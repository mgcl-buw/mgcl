/**
 * @date 30.06.2025
 * This file contains code for benchmarking the Jacobi version where iteration of inner gps is overlaped with
 * ghost update in multi-gpu case. For scalar varying stencil only.
 *
 * To remain flexbility and reproducability, no actual production code is used, but instead the driver functions
 * are copied here.
 *
 */

#include "bench_util.hpp"
#include "nanobench.h"

#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"

#include <CL/cl.h>
#include <catch2/catch_message.hpp>
#include <chrono>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <sstream>
#include <vector>
using namespace std::chrono_literals;

#include "../src/mgcl/cuboid.hpp"
#include "../src/mgcl/level.hpp"
#include "../src/mgcl/mpi_util.hpp"
#include "../src/mgcl/multigrid_engine.hpp"
#include "../src/mgcl/problem.hpp"
#include "../test/test_utility.hpp"
#include "cli_args.hpp"

namespace mgcl_bench_jacobi_varying_overlapped
{
    enum class KernelVersion
    {
        DEFAULT,
        OVERLAPPED
    };

    enum class OverlappedKernelVersion
    {
        BOUNDARY_INNER_PARALLEL, // enqueue boundary and inner kernels back to back
        EXTRACT_INNER_PARALLEL,  // enqueue extract and inner kernels back to back
        MEMCPY_INNER_PARALLEL    // wait for extract to finish and then enqueue inner, s.t. it is overlapped with memcpy
    };

    using size_t3 = struct
    {
        int x, y, z;
    };

    // using JacobiArgs = struct
    // {
    //     mgcl::CuboidGpu& f;
    //     mgcl::CuboidGpu& v_in;
    //     mgcl::CuboidGpu& v_out;
    //     mgcl::CuboidGpu& r;
    //     mgcl::MGCL_RESIDUAL_NORM resnorm;
    //     mgcl::VaryingStencilGpu& stencilValues;
    //     mgcl::CuboidBSGpu* dRsq;

    //     bool returnResidualNorm;
    //     bool periodic;
    //     bool updateGhostsLocally;
    //     int maxiter;
    //     int stepsPerIter;
    //     double omega;

    //     mgcl::BufferGpu* dPlanesBuf;
    //     std::vector<double>* sendBuf;
    //     std::vector<double>* recvBuf;

    //     cl_program program;
    //     cl_command_queue queue;
    //     cl_context context;

    //     int moff = 0;
    //     int noff = 0;
    //     int ooff = 0;
    //     mgcl::MPILevelData* mpiData = nullptr;

    //     mgcl::conf::KernelConfig* conf = nullptr;
    //     mgcl::ProfilingData* pd = nullptr;

    //     KernelVersion kernelVersion;
    // };

    /* Runs jacobi method using OpenCL.
     * Doesn't creates ocl buffers and doesn't copy data from host to device and vice versa
     * v, f and r must be of size [m][n][o] for periodic boundary condition. Ghosts of v and f must be updated.
     * m, n and o must be the dimensions of grid + 2*ghosts
     * If return_residual is true, the residual's 2-norm or inf-norm will be read back from device and returned, else -1.
     * It's not
     * really performant to do so because we have to wait for all kernels to complete and reading a buffer to host is slow.
     * stepsPerIter is amount iterations without ghost update in-between. Ghost cells must be adequate. Defaults to 1.
     */
    double jacobiDefault(mgcl::Problem& problem, mgcl::Level& level, int maxiter, bool return_residual, int stepsPerIter)
    {
        int err;
        int mgh = level.getMgh();
        int ngh = level.getNgh();
        int ogh = level.getOgh();
        int store_res = 0;
        double res = -1;
        int idx_start = 0;

        // decrease stepsPerIter if it's less than maxIter
        if (maxiter < stepsPerIter)
            stepsPerIter = maxiter;

        // Ghosts only need to be updated in the periodic case, so set stepsPerIter = 1 for non-periodic.
        // TODO adjust for MPI
        if (!problem.isPeriodic())
            stepsPerIter = 1;

        // Check if amount of ghost cells is large enough
        if (problem.getGhosts() < stepsPerIter)
        {
            error("#ghosts must be >= stepsPerIter!");
        }

        cl_event ev;

        double h2 = 1.0 / static_cast<double>((problem.getMGlobal() >> level.getNum()) * (problem.getMGlobal() >> level.getNum()));
        double dinv = h2 / 6.0;
        double h2inv = level.getStencilFactor(); // divisor of the stencil, inverted to use * instead of / in kernel
        // TODO refactor stencilFactor

        // Create the compute kernel from the program
        const char* kernelName;
        if (problem.getStencilType() == mgcl::MGCL_LAPLACE_7POINT)
            kernelName = "jacobi_iter_7point";
        else if (problem.getStencilType() == mgcl::MGCL_LAPLACE_19POINT)
        {
            kernelName = "jacobi_iter_19point";
            dinv = (6.0 * h2) / 24.0;
        }
        else if (problem.getStencilType() == mgcl::MGCL_LAPLACE_27POINT)
        {
            kernelName = "jacobi_iter_27point";
            dinv = (26.0 * h2) / 88.0;
        }
        else if (problem.getStencilType() == mgcl::MGCL_VARYING)
        {
            kernelName = "jacobi_iter_27point_varying_stencil_1d";
        }
        else if (problem.getStencilType() == mgcl::MGCL_FIXED)
        {
            kernelName = "jacobi_iter_27point_fixed_stencil_1d";
        }

        cl_kernel kernel = clCreateKernel(problem.getProgram(), kernelName, &err);
        mgcl::mgclCheckError(err, "Creating kernel");

        cl_mem dVIn = level.getDVIn().getBuffer();
        cl_mem dVOut = level.getDVOut().getBuffer();
        cl_mem dF = level.getDF().getBuffer();
        cl_mem dR = level.getDR().getBuffer();

        // assign kernel arguments
        int pos = 0;
        int pos_idxstart = -1;
        int pos_storeres = -1;

        double omega = problem.getOmega();
        int ghosts = problem.getGhosts();

        if (problem.getStencilType() == mgcl::MGCL_VARYING)
        {
            auto svbuf = level.getStencilValuesGpu()->getBuf();
            int svgh = level.getStencilValuesGpu()->getGh();
            int svmgh = level.getStencilValuesGpu()->getMgh();
            int svngh = level.getStencilValuesGpu()->getNgh();
            int svogh = level.getStencilValuesGpu()->getOgh();
            int svGridSize = svmgh * svngh * svogh;
            err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &dVIn);
            err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dVOut);
            err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dF);
            err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dR);
            err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &svbuf);
            err |= clSetKernelArg(kernel, ++pos, sizeof(double), &omega);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &mgh);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ngh);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ogh);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svmgh);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svngh);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svogh);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghosts);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svgh);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svGridSize);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &idx_start);
            pos_idxstart = pos;
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &store_res);
            pos_storeres = pos;
        }
        else if (problem.getStencilType() == mgcl::MGCL_FIXED)
        {
            auto& fs = *level.getFixedStencil();
            err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &dVIn);
            err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dVOut);
            err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dF);
            err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dR);
            err |= clSetKernelArg(kernel, ++pos, sizeof(double), &omega);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &mgh);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ngh);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ogh);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghosts);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &idx_start);
            pos_idxstart = pos;
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &store_res);
            pos_storeres = pos;
            err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[0][0][0]);
            err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[0][0][1]);
            err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[0][0][2]);
            err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[0][1][0]);
            err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[0][1][1]);
            err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[0][1][2]);
            err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[0][2][0]);
            err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[0][2][1]);
            err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[0][2][2]);
            err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[1][0][0]);
            err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[1][0][1]);
            err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[1][0][2]);
            err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[1][1][0]);
            err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[1][1][1]);
            err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[1][1][2]);
            err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[1][2][0]);
            err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[1][2][1]);
            err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[1][2][2]);
            err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[2][0][0]);
            err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[2][0][1]);
            err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[2][0][2]);
            err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[2][1][0]);
            err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[2][1][1]);
            err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[2][1][2]);
            err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[2][2][0]);
            err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[2][2][1]);
            err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[2][2][2]);
        }
        else
        {
            err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &dVIn);
            err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dVOut);
            err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dF);
            err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dR);
            err |= clSetKernelArg(kernel, ++pos, sizeof(double), &h2inv);
            err |= clSetKernelArg(kernel, ++pos, sizeof(double), &dinv);
            err |= clSetKernelArg(kernel, ++pos, sizeof(double), &omega);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &mgh);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ngh);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ogh);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghosts);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &idx_start);
            pos_idxstart = pos;
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &store_res);
            pos_storeres = pos;
        }
        mgcl::mgclCheckError(err, "Setting kernel arguments");

        // One work-item per cell (including ghost cells).
        size_t global[2] = {static_cast<size_t>(mgh * ngh * ogh), static_cast<size_t>(0)};
        const auto& c = mgcl::conf::getWorkGroupSizeForKernelAndWiCount(problem.getKernelConfig(), kernelName, 1);
        size_t local[2] = {c[0], c[1]};

        // kernels that use constant Laplace stencils are 2d and need different global and local sizes
        if (problem.getStencilType() != mgcl::MGCL_VARYING && problem.getStencilType() != mgcl::MGCL_FIXED)
        {
            global[0] = static_cast<size_t>(ngh);
            global[1] = static_cast<size_t>(ogh);
            // local[0] = static_cast<size_t>(1);
            // local[1] = static_cast<size_t>(64);
        }

        // Pad global sizes to fit to local sizes
        int kernelDims = (problem.getStencilType() == mgcl::MGCL_VARYING || problem.getStencilType() == mgcl::MGCL_FIXED) ? 1 : 2;
        for (int i = 0; i < kernelDims; i++)
            if (global[i] % local[i] != 0)
            {
                global[i] += local[i] - (global[i] % local[i]);
            }

        int globalIter = 0;
        while (globalIter < maxiter)
        {
            // Update ghosts of current input v
            if (globalIter % 2 == 1)
            {
                err = mgcl::MultigridEngine::updateGhosts(problem, level.getDVOut(),
                                                          level.getMpiDataPtr(), level.isCalculatedLocally());
                mgcl::mgclCheckError(err, "Updating ghosts");
            }
            else
            {
                err = mgcl::MultigridEngine::updateGhosts(problem, level.getDVIn(),
                                                          level.getMpiDataPtr(), level.isCalculatedLocally());
                mgcl::mgclCheckError(err, "Updating ghosts");
            }

            // if stepsPerIter > 1, multiple iterations can be done without updating ghosts in-between
            for (int innerIter = 0; innerIter < stepsPerIter && globalIter < maxiter; innerIter++, globalIter++)
            {
                // damped/weighted iteration formula: u_(m+1) = u_(m) + omega * D^-1 * r_(m)

                // switch arguments dVIn -> dVOut to use latest values in next iteration
                if (globalIter % 2 == 1)
                {
                    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &dVIn);
                    err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &dVOut);
                    mgcl::mgclCheckError(err, "Setting kernel arguments");
                }
                else
                {
                    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &dVIn);
                    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &dVOut);
                    mgcl::mgclCheckError(err, "Setting kernel arguments");
                }

                // set flag to store residual in last iteration
                if (globalIter == maxiter - 1)
                {
                    store_res = 1;
                    err = clSetKernelArg(kernel, pos_storeres, sizeof(int), &store_res);
                    mgcl::mgclCheckError(err, "Setting kernel arguments");
                }

                // recalculate and set idx_start
                idx_start = problem.getGhosts() - ((stepsPerIter - innerIter) - 1);
                err = clSetKernelArg(kernel, pos_idxstart, sizeof(int), &idx_start);
                mgcl::mgclCheckError(err, "Setting kernel arguments");

                err = clEnqueueNDRangeKernel(problem.getOpenCLHelper().getCommands(), kernel, kernelDims, NULL, global, local, 0, NULL, &ev);
                mgcl::mgclCheckError(err, "Enqueueing kernel");

                if (problem.isProfilingEnabled())
                {
                    problem.getProfilingData()->addMeasurement(problem.getCommands(), ev, kernelName,
                                                               {global[0], global[1], 0},
                                                               {local[0], local[1], 1});
                }
                mgcl::mgclCheckError(clReleaseEvent(ev), "clReleaseEvent");
            }
        }

        // copy result into dVIn if needed
        if (maxiter % 2 == 1)
            // level.getDVOut().copyTo(problem.getOpenCLHelper().getCommands(), level.getDVIn());
            mgcl::CuboidGpu::swap(level.getDVIn(), level.getDVOut());

        // Update ghosts of dVIn
        err = mgcl::MultigridEngine::updateGhosts(problem, level.getDVIn(),
                                                  level.getMpiDataPtr(), level.isCalculatedLocally());
        mgcl::mgclCheckError(err, "Updating ghosts");

        // calculate residual and its norm
        if (return_residual)
        {
            // update residual to use current approximation v
            res = mgcl::MultigridEngine::residual(problem, level, true);
        }

        clReleaseKernel(kernel);

        return res;
    }

    namespace overlapped_helpers
    {

        // Starts the kernel for one iteration of Jacobi for boundary gps.
        // No ghost update included. No final residual calculation included.
        double jacobiBoundary(mgcl::Problem& problem, mgcl::Level& level,
                              cl_mem dVIn, cl_mem dVOut, int store_res,
                              cl_command_queue queue, cl_kernel kernel,
                              size_t global[3], size_t local[3],
                              std::string kernelName)
        {
            int err;
            int mgh = level.getMgh();
            int ngh = level.getNgh();
            int ogh = level.getOgh();
            double res = -1;
            int idx_start = level.getDVIn().getGhostsM(); // only inner gps.

            cl_event ev;

            double h2 = 1.0 / static_cast<double>((problem.getMGlobal() >> level.getNum()) * (problem.getMGlobal() >> level.getNum()));
            double dinv = h2 / 6.0;
            double h2inv = level.getStencilFactor(); // divisor of the stencil, inverted to use * instead of / in kernel
            // TODO refactor stencilFactor

            // Create the compute kernel from the program
            if (problem.getStencilType() == mgcl::MGCL_LAPLACE_19POINT)
            {
                dinv = (6.0 * h2) / 24.0;
            }
            else if (problem.getStencilType() == mgcl::MGCL_LAPLACE_27POINT)
            {
                dinv = (26.0 * h2) / 88.0;
            }

            // cl_mem dVIn = level.getDVIn().getBuffer();
            // cl_mem dVOut = level.getDVOut().getBuffer();
            cl_mem dF = level.getDF().getBuffer();
            cl_mem dR = level.getDR().getBuffer();

            // assign kernel arguments
            int pos = 0;

            double omega = problem.getOmega();
            int ghosts = problem.getGhosts();

            if (problem.getStencilType() == mgcl::MGCL_VARYING)
            {
                auto svbuf = level.getStencilValuesGpu()->getBuf();
                int svgh = level.getStencilValuesGpu()->getGh();
                int svmgh = level.getStencilValuesGpu()->getMgh();
                int svngh = level.getStencilValuesGpu()->getNgh();
                int svogh = level.getStencilValuesGpu()->getOgh();
                int svGridSize = svmgh * svngh * svogh;
                err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &dVIn);
                err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dVOut);
                err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dF);
                err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dR);
                err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &svbuf);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &omega);
                err |= clSetKernelArg(kernel, ++pos, sizeof(int), &mgh);
                err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ngh);
                err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ogh);
                err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svmgh);
                err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svngh);
                err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svogh);
                err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghosts);
                err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svgh);
                err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svGridSize);
                err |= clSetKernelArg(kernel, ++pos, sizeof(int), &idx_start);
                err |= clSetKernelArg(kernel, ++pos, sizeof(int), &store_res);
            }
            else if (problem.getStencilType() == mgcl::MGCL_FIXED)
            {
                auto& fs = *level.getFixedStencil();
                err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &dVIn);
                err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dVOut);
                err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dF);
                err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dR);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &omega);
                err |= clSetKernelArg(kernel, ++pos, sizeof(int), &mgh);
                err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ngh);
                err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ogh);
                err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghosts);
                err |= clSetKernelArg(kernel, ++pos, sizeof(int), &idx_start);
                err |= clSetKernelArg(kernel, ++pos, sizeof(int), &store_res);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[0][0][0]);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[0][0][1]);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[0][0][2]);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[0][1][0]);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[0][1][1]);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[0][1][2]);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[0][2][0]);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[0][2][1]);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[0][2][2]);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[1][0][0]);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[1][0][1]);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[1][0][2]);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[1][1][0]);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[1][1][1]);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[1][1][2]);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[1][2][0]);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[1][2][1]);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[1][2][2]);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[2][0][0]);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[2][0][1]);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[2][0][2]);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[2][1][0]);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[2][1][1]);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[2][1][2]);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[2][2][0]);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[2][2][1]);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[2][2][2]);
            }
            else
            {
                err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &dVIn);
                err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dVOut);
                err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dF);
                err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dR);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &h2inv);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &dinv);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &omega);
                err |= clSetKernelArg(kernel, ++pos, sizeof(int), &mgh);
                err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ngh);
                err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ogh);
                err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghosts);
                err |= clSetKernelArg(kernel, ++pos, sizeof(int), &idx_start);
                err |= clSetKernelArg(kernel, ++pos, sizeof(int), &store_res);
            }
            mgcl::mgclCheckError(err, "Setting kernel arguments");

            int kernelDims = 2;
            if (problem.getStencilType() == mgcl::MGCL_VARYING || problem.getStencilType() == mgcl::MGCL_FIXED)
                kernelDims = 1;

            err = clEnqueueNDRangeKernel(queue, kernel, kernelDims, NULL, global, local, 0, NULL, &ev);
            mgcl::mgclCheckError(err, "Enqueueing kernel");

            if (problem.isProfilingEnabled())
            {
                problem.getProfilingData()->addMeasurement(queue, ev, kernelName,
                                                           {global[0], global[1], 0},
                                                           {local[0], local[1], 1});
            }
            mgcl::mgclCheckError(clReleaseEvent(ev), "clReleaseEvent");

            return res;
        }

        // Starts the kernel for one iteration of Jacobi for inner gps.
        // No ghost update included. No final residual calculation included.
        double jacobiInner(mgcl::Problem& problem, mgcl::Level& level,
                           cl_mem dVIn, cl_mem dVOut, int store_res,
                           cl_command_queue queue, cl_kernel kernel,
                           size_t global[3], size_t local[3],
                           std::string kernelName)
        {
            int err;
            int mgh = level.getMgh();
            int ngh = level.getNgh();
            int ogh = level.getOgh();
            double res = -1;
            int idx_start = level.getDVIn().getGhostsM(); // only inner gps.

            cl_event ev;

            double h2 = 1.0 / static_cast<double>((problem.getMGlobal() >> level.getNum()) * (problem.getMGlobal() >> level.getNum()));
            double dinv = h2 / 6.0;
            double h2inv = level.getStencilFactor(); // divisor of the stencil, inverted to use * instead of / in kernel
            // TODO refactor stencilFactor

            // Create the compute kernel from the program
            if (problem.getStencilType() == mgcl::MGCL_LAPLACE_19POINT)
            {
                dinv = (6.0 * h2) / 24.0;
            }
            else if (problem.getStencilType() == mgcl::MGCL_LAPLACE_27POINT)
            {
                dinv = (26.0 * h2) / 88.0;
            }

            // cl_mem dVIn = level.getDVIn().getBuffer();
            // cl_mem dVOut = level.getDVOut().getBuffer();
            cl_mem dF = level.getDF().getBuffer();
            cl_mem dR = level.getDR().getBuffer();

            // assign kernel arguments
            int pos = 0;

            double omega = problem.getOmega();
            int ghosts = problem.getGhosts();

            if (problem.getStencilType() == mgcl::MGCL_VARYING)
            {
                auto svbuf = level.getStencilValuesGpu()->getBuf();
                int svgh = level.getStencilValuesGpu()->getGh();
                int svmgh = level.getStencilValuesGpu()->getMgh();
                int svngh = level.getStencilValuesGpu()->getNgh();
                int svogh = level.getStencilValuesGpu()->getOgh();
                int svGridSize = svmgh * svngh * svogh;
                err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &dVIn);
                err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dVOut);
                err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dF);
                err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dR);
                err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &svbuf);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &omega);
                err |= clSetKernelArg(kernel, ++pos, sizeof(int), &mgh);
                err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ngh);
                err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ogh);
                err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svmgh);
                err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svngh);
                err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svogh);
                err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghosts);
                err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svgh);
                err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svGridSize);
                err |= clSetKernelArg(kernel, ++pos, sizeof(int), &idx_start);
                err |= clSetKernelArg(kernel, ++pos, sizeof(int), &store_res);
            }
            else if (problem.getStencilType() == mgcl::MGCL_FIXED)
            {
                auto& fs = *level.getFixedStencil();
                err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &dVIn);
                err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dVOut);
                err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dF);
                err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dR);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &omega);
                err |= clSetKernelArg(kernel, ++pos, sizeof(int), &mgh);
                err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ngh);
                err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ogh);
                err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghosts);
                err |= clSetKernelArg(kernel, ++pos, sizeof(int), &idx_start);
                err |= clSetKernelArg(kernel, ++pos, sizeof(int), &store_res);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[0][0][0]);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[0][0][1]);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[0][0][2]);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[0][1][0]);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[0][1][1]);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[0][1][2]);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[0][2][0]);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[0][2][1]);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[0][2][2]);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[1][0][0]);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[1][0][1]);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[1][0][2]);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[1][1][0]);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[1][1][1]);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[1][1][2]);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[1][2][0]);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[1][2][1]);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[1][2][2]);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[2][0][0]);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[2][0][1]);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[2][0][2]);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[2][1][0]);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[2][1][1]);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[2][1][2]);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[2][2][0]);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[2][2][1]);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &fs[2][2][2]);
            }
            else
            {
                err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &dVIn);
                err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dVOut);
                err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dF);
                err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dR);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &h2inv);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &dinv);
                err |= clSetKernelArg(kernel, ++pos, sizeof(double), &omega);
                err |= clSetKernelArg(kernel, ++pos, sizeof(int), &mgh);
                err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ngh);
                err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ogh);
                err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghosts);
                err |= clSetKernelArg(kernel, ++pos, sizeof(int), &idx_start);
                err |= clSetKernelArg(kernel, ++pos, sizeof(int), &store_res);
            }
            mgcl::mgclCheckError(err, "Setting kernel arguments");

            int kernelDims = 2;
            if (problem.getStencilType() == mgcl::MGCL_VARYING || problem.getStencilType() == mgcl::MGCL_FIXED)
                kernelDims = 1;

            err = clEnqueueNDRangeKernel(queue, kernel, kernelDims, NULL, global, local, 0, NULL, &ev);
            mgcl::mgclCheckError(err, "Enqueueing kernel");

            if (problem.isProfilingEnabled())
            {
                problem.getProfilingData()->addMeasurement(queue, ev, kernelName,
                                                           {global[0], global[1], 0},
                                                           {local[0], local[1], 1});
            }
            mgcl::mgclCheckError(clReleaseEvent(ev), "clReleaseEvent");

            return res;
        }
    }

    /* Runs jacobi method using OpenCL.
     * Doesn't creates ocl buffers and doesn't copy data from host to device and vice versa
     * v, f and r must be of size [m][n][o] for periodic boundary condition. Ghosts of v and f must be updated.
     * m, n and o must be the dimensions of grid + 2*ghosts
     * If return_residual is true, the residual's 2-norm or inf-norm will be read back from device and returned, else -1.
     * It's not
     * really performant to do so because we have to wait for all kernels to complete and reading a buffer to host is slow.
     * stepsPerIter is amount iterations without ghost update in-between. Ghost cells must be adequate. Defaults to 1.
     */
    double jacobiOverlapped(mgcl::Problem& problem, mgcl::Level& level, int maxiter, bool return_residual, int stepsPerIter,
                            cl_command_queue queue2, OverlappedKernelVersion kv)
    {
        int err;
        double res = -1;
        bool store_res = false;

        // decrease stepsPerIter if it's less than maxIter
        if (maxiter < stepsPerIter)
            stepsPerIter = maxiter;

        // Ghosts only need to be updated in the periodic case, so set stepsPerIter = 1 for non-periodic.
        // TODO adjust for MPI
        if (!problem.isPeriodic())
            stepsPerIter = 1;

        // Check if amount of ghost cells is large enough
        if (problem.getGhosts() < stepsPerIter)
        {
            error("#ghosts must be >= stepsPerIter!");
        }

        cl_mem dVIn = level.getDVIn().getBuffer();
        cl_mem dVOut = level.getDVOut().getBuffer();

        int mgh = level.getMgh();
        int ngh = level.getNgh();
        int ogh = level.getOgh();
        int idx_start = level.getDVIn().getGhostsM(); // only inner gps.

        cl_event ev;

        double h2 = 1.0 / static_cast<double>((problem.getMGlobal() >> level.getNum()) * (problem.getMGlobal() >> level.getNum()));
        double dinv = h2 / 6.0;
        double h2inv = level.getStencilFactor(); // divisor of the stencil, inverted to use * instead of / in kernel
        // TODO refactor stencilFactor

        /********************** create boundary kernel *********************/
        const char* kernelNameBoundary;
        if (problem.getStencilType() == mgcl::MGCL_LAPLACE_7POINT)
            kernelNameBoundary = "jacobi_iter_7point_boundary";
        else if (problem.getStencilType() == mgcl::MGCL_LAPLACE_19POINT)
        {
            kernelNameBoundary = "jacobi_iter_19point_boundary";
            dinv = (6.0 * h2) / 24.0;
        }
        else if (problem.getStencilType() == mgcl::MGCL_LAPLACE_27POINT)
        {
            kernelNameBoundary = "jacobi_iter_27point_boundary";
            dinv = (26.0 * h2) / 88.0;
        }
        else if (problem.getStencilType() == mgcl::MGCL_VARYING)
        {
            kernelNameBoundary = "jacobi_iter_27point_varying_stencil_1d_boundary";
        }
        else if (problem.getStencilType() == mgcl::MGCL_FIXED)
        {
            kernelNameBoundary = "jacobi_iter_27point_fixed_stencil_1d_boundary";
        }
        cl_kernel boundaryKernel = clCreateKernel(problem.getProgram(), kernelNameBoundary, &err);
        mgcl::mgclCheckError(err, "Creating boundaryKernel");

        /********************** create inner kernel *********************/
        const char* kernelNameInner;
        if (problem.getStencilType() == mgcl::MGCL_LAPLACE_7POINT)
            kernelNameInner = "jacobi_iter_7point_inner";
        else if (problem.getStencilType() == mgcl::MGCL_LAPLACE_19POINT)
        {
            kernelNameInner = "jacobi_iter_19point_inner";
            dinv = (6.0 * h2) / 24.0;
        }
        else if (problem.getStencilType() == mgcl::MGCL_LAPLACE_27POINT)
        {
            kernelNameInner = "jacobi_iter_27point_inner";
            dinv = (26.0 * h2) / 88.0;
        }
        else if (problem.getStencilType() == mgcl::MGCL_VARYING)
        {
            kernelNameInner = "jacobi_iter_27point_varying_stencil_1d_inner";
        }
        else if (problem.getStencilType() == mgcl::MGCL_FIXED)
        {
            kernelNameInner = "jacobi_iter_27point_fixed_stencil_1d_inner";
        }
        cl_kernel innerKernel = clCreateKernel(problem.getProgram(), kernelNameInner, &err);
        mgcl::mgclCheckError(err, "Creating innerKernel");

        // NOTE: using the same conf for inner and boundary kernels

        // One work-item per cell (including ghost cells).
        size_t global[3] = {static_cast<size_t>(mgh * ngh * ogh), static_cast<size_t>(0), static_cast<size_t>(0)};
        const auto& c = mgcl::conf::getWorkGroupSizeForKernelAndWiCount(problem.getKernelConfig(), kernelNameBoundary, 1);
        size_t local[3] = {c[0], c[1], c[2]};

        // kernels that use constant Laplace stencils are 2d and need different global and local sizes
        if (problem.getStencilType() != mgcl::MGCL_VARYING && problem.getStencilType() != mgcl::MGCL_FIXED)
        {
            global[0] = static_cast<size_t>(ngh);
            global[1] = static_cast<size_t>(ogh);
            // local[0] = static_cast<size_t>(1);
            // local[1] = static_cast<size_t>(64);
        }

        // Pad global sizes to fit to local sizes
        int kernelDims = (problem.getStencilType() == mgcl::MGCL_VARYING || problem.getStencilType() == mgcl::MGCL_FIXED) ? 1 : 2;
        for (int i = 0; i < kernelDims; i++)
            if (global[i] % local[i] != 0)
            {
                global[i] += local[i] - (global[i] % local[i]);
            }

        // Update ghosts use the default command queue of the problem instance
        err = mgcl::MultigridEngine::updateGhosts(problem, level.getDVIn(),
                                                  level.getMpiDataPtr(), level.isCalculatedLocally());
        mgcl::mgclCheckError(err, "Updating ghosts");

        int globalIter = 0;
        auto ptr_dvin_wrapper = &level.getDVIn();
        auto ptr_dvout_wrapper = &level.getDVOut();
        auto tmp = ptr_dvin_wrapper;
        while (globalIter < maxiter)
        {
            // if stepsPerIter > 1, multiple iterations can be done without updating ghosts in-between
            for (int innerIter = 0; innerIter < stepsPerIter && globalIter < maxiter; innerIter++, globalIter++)
            {
                auto dVIn = ptr_dvin_wrapper->getBuffer();
                auto dVOut = ptr_dvout_wrapper->getBuffer();

                store_res = globalIter == maxiter - 1;

                // calculate boundary points and inner points in separate queues concurrently
                overlapped_helpers::jacobiBoundary(problem, level, dVIn, dVOut, store_res, problem.getCommands(), boundaryKernel, global, local, kernelNameBoundary);

                if (kv == OverlappedKernelVersion::BOUNDARY_INNER_PARALLEL)
                {
                    overlapped_helpers::jacobiInner(problem, level, dVIn, dVOut, store_res, queue2, innerKernel, global, local, kernelNameInner);

                    // No need for waiting for the boundary kernel to finish, because extractBorderPlanes is in same in-order queue
                    err = mgcl::MultigridEngine::updateGhosts(problem, *ptr_dvout_wrapper,
                                                              level.getMpiDataPtr(), level.isCalculatedLocally());
                    mgcl::mgclCheckError(err, "Updating ghosts");
                }
                else
                {
                    auto& d_buf = *ptr_dvout_wrapper;
                    if (problem.getDPlanesBufPtr() == nullptr)
                        error("MultigridEngine::updateGhostsOclMpi: dPlanesBufPtr is null");

                    // Use temporary buffer for extracting and pasting planes. Check if it's large enough beforehand.
                    // TODO maybe disable check in UNSAFE mode
                    int yz = d_buf.getNgh() * d_buf.getOgh();
                    int xz = d_buf.getMgh() * d_buf.getOgh();
                    int xy = d_buf.getMgh() * d_buf.getNgh();
                    int ressize = 2 * yz * d_buf.getGhostsM() + 2 * xz * d_buf.getGhostsN() + 2 * xy * d_buf.getGhostsO();

                    auto dPlanesBuf = problem.getDPlanesBufPtr();
                    if (dPlanesBuf->getSize() < ressize)
                        error("MultigridEngine::updateGhostsOclMpi: dPlanesBuf is too small. Need at least " + std::to_string(ressize) + ", but is " + std::to_string(dPlanesBuf->getSize()));

                    auto hPlanesBufSend = problem.getHPlanesBufSendPtr();
                    auto hPlanesBufRecv = problem.getHPlanesBufRecvPtr();
                    if (hPlanesBufSend->size() < ressize || hPlanesBufRecv->size() < ressize)
                        throw "MultigridEngine::updateGhostsOclMpi: hPlanesBufSend or hPlanesBufRecv is too small. Need at least " +
                            std::to_string(ressize) + ", but is " + std::to_string(hPlanesBufSend->size()) +
                            " (send) and " + std::to_string(hPlanesBufRecv->size()) + " (recv)";

                    // Extract border planes from the buffer
                    // d_buf.extractBorderPlanes(problem.getCommands(), problem.getProgram(),
                    //                           dPlanesBuf, hPlanesBufSend,
                    //                           &problem.getKernelConfig(), problem.getProfilingData());
                    d_buf.extractBorderPlanes(problem.getCommands(), problem.getProgram(),
                                              dPlanesBuf, nullptr,
                                              &problem.getKernelConfig(), problem.getProfilingData(), false);
                    auto& sbuf = *hPlanesBufSend;
                    auto& rbuf = *hPlanesBufRecv;

                    if (kv == OverlappedKernelVersion::MEMCPY_INNER_PARALLEL)
                    {
                        problem.finish(); // if we wait here, inner Jacobi and memcpy will be executed in parallel
                    }

                    overlapped_helpers::jacobiInner(problem, level, dVIn, dVOut, store_res, queue2, innerKernel, global, local, kernelNameInner);

                    dPlanesBuf->read(problem.getCommands(), hPlanesBufSend->data(), true, ressize, nullptr);

                    // Send our planes to neighbours and receive their planes
                    mgcl::mpi_util::sendBorderPlanes(d_buf.getMgh(), d_buf.getNgh(), d_buf.getOgh(),
                                                     d_buf.getGhostsM(), d_buf.getGhostsN(), d_buf.getGhostsO(), 1,
                                                     sbuf, rbuf, *level.getMpiDataPtr());

                    // Paste planes back into the buffer.
                    dPlanesBuf->write(problem.getCommands(), rbuf, false, ressize, problem.getProfilingData());
                    d_buf.pasteGhostsFromBorderPlanes(problem.getContext(), problem.getCommands(), problem.getProgram(),
                                                      dPlanesBuf, nullptr,
                                                      &problem.getKernelConfig(), problem.getProfilingData());
                }

                // Wait for inner kernel and ghost update to finish
                mgcl::mgclCheckError(clFinish(problem.getCommands()), "clFinish queue1");
                mgcl::mgclCheckError(clFinish(queue2), "clFinish queue2");

                // swap dVIn and dVOut for next iteration
                tmp = ptr_dvout_wrapper;
                ptr_dvout_wrapper = ptr_dvin_wrapper;
                ptr_dvin_wrapper = tmp;
            }
        }

        mgcl::mgclCheckError(clReleaseKernel(boundaryKernel), "Releasing boundaryKernel");
        mgcl::mgclCheckError(clReleaseKernel(innerKernel), "Releasing innerKernel");

        // copy result into dVIn if needed
        if (maxiter % 2 == 1)
            // level.getDVOut().copyTo(problem.getOpenCLHelper().getCommands(), level.getDVIn());
            mgcl::CuboidGpu::swap(level.getDVIn(), level.getDVOut());

        // Update ghosts of dVIn
        // err = mgcl::MultigridEngine::updateGhosts(problem, level.getDVIn(),
        //                                           level.getMpiDataPtr(), level.isCalculatedLocally());
        // mgcl::mgclCheckError(err, "Updating ghosts");

        // calculate residual and its norm
        if (return_residual)
        {
            // update residual to use current approximation v
            res = mgcl::MultigridEngine::residual(problem, level, true);
        }

        return res;
    }

    // /* Runs jacobi method using OpenCL.
    //  * Doesn't creates ocl buffers and doesn't copy data from host to device and vice versa
    //  * v, f and r must be of size [m][n][o] for periodic boundary condition. Ghosts of v and f must be updated.
    //  * m, n and o must be the dimensions of grid + 2*ghosts
    //  * If return_residual is true, the residual's 2-norm or inf-norm will be read back from device and returned, else -1.
    //  * It's not
    //  * really performant to do so because we have to wait for all kernels to complete and reading a buffer to host is slow.
    //  * stepsPerIter is amount iterations without ghost update in-between. Ghost cells must be adequate. Defaults to 1.
    //  */
    // double jacobiOverlapped(JacobiArgs& args)
    // {
    //     int err;
    //     int mgh = args.f.getMgh();
    //     int ngh = args.f.getNgh();
    //     int ogh = args.f.getOgh();
    //     int store_res = 0;
    //     double res = -1;
    //     int idx_start = 0;

    //     // decrease stepsPerIter if it's less than maxIter
    //     if (args.maxiter < args.stepsPerIter)
    //         args.stepsPerIter = args.maxiter;

    //     // Ghosts only need to be updated in the periodic case, so set stepsPerIter = 1 for non-periodic.
    //     // TODO adjust for MPI
    //     if (!args.periodic)
    //         args.stepsPerIter = 1;

    //     // Check if amount of ghost cells is large enough
    //     if (args.f.getGhostsM() < args.stepsPerIter)
    //     {
    //         error("#ghosts must be >= stepsPerIter!");
    //     }

    //     cl_event ev;

    //     // Create the compute kernel from the program
    //     const char* kernelName = "jacobi_iter_27point_varying_stencil_1d";

    //     cl_kernel kernel = clCreateKernel(args.program, kernelName, &err);
    //     mgcl::mgclCheckError(err, "Creating kernel");

    //     cl_mem dVIn = args.v_in.getBuffer();
    //     cl_mem dVOut = args.v_out.getBuffer();
    //     cl_mem dF = args.f.getBuffer();
    //     cl_mem dR = args.r.getBuffer();

    //     // assign kernel arguments
    //     int pos = 0;
    //     int pos_idxstart = -1;
    //     int pos_storeres = -1;

    //     // if (problem.stencilType == MGCL_VARYING)
    //     {
    //         auto svbuf = args.stencilValues.getBuf();
    //         int svgh = args.stencilValues.getGh();
    //         int svmgh = args.stencilValues.getMgh();
    //         int svngh = args.stencilValues.getNgh();
    //         int svogh = args.stencilValues.getOgh();
    //         int svGridSize = svmgh * svngh * svogh;
    //         int ghosts = args.f.getGhostsM();

    //         err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &dVIn);
    //         err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dVOut);
    //         err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dF);
    //         err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dR);
    //         err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &svbuf);
    //         err |= clSetKernelArg(kernel, ++pos, sizeof(double), &args.omega);
    //         err |= clSetKernelArg(kernel, ++pos, sizeof(int), &mgh);
    //         err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ngh);
    //         err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ogh);
    //         err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svmgh);
    //         err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svngh);
    //         err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svogh);
    //         err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghosts);
    //         err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svgh);
    //         err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svGridSize);
    //         err |= clSetKernelArg(kernel, ++pos, sizeof(int), &idx_start);
    //         pos_idxstart = pos;
    //         err |= clSetKernelArg(kernel, ++pos, sizeof(int), &store_res);
    //         pos_storeres = pos;
    //     }
    //     mgcl::mgclCheckError(err, "Setting kernel arguments");

    //     // One work-item per cell (including ghost cells).
    //     size_t global = static_cast<size_t>(mgh * ngh * ogh);
    //     size_t local = 32;

    //     if (args.conf)
    //     {
    //         const auto& c = mgcl::conf::getWorkGroupSizeForKernelAndWiCount(*args.conf, kernelName, 1);
    //         local = c[0];
    //     }

    //     // Pad global sizes to fit to local sizes
    //     int kernelDims = 1;
    //     if (global % local != 0)
    //     {
    //         global += local - (global % local);
    //     }

    //     int globalIter = 0;
    //     while (globalIter < args.maxiter)
    //     {
    //         // Update ghosts of current input v
    //         if (globalIter % 2 == 1)
    //         {
    //             err = mgcl::MultigridEngine::updateGhosts(problem, level.getDVOut(),
    //                                                       level.getMpiDataPtr(), level.isCalculatedLocally());
    //             mgclCheckError(err, "Updating ghosts");
    //         }
    //         else
    //         {
    //             err = MultigridEngine::updateGhosts(problem, level.getDVIn(),
    //                                                 level.getMpiDataPtr(), level.isCalculatedLocally());
    //             mgclCheckError(err, "Updating ghosts");
    //         }

    //         // if stepsPerIter > 1, multiple iterations can be done without updating ghosts in-between
    //         for (int innerIter = 0; innerIter < stepsPerIter && globalIter < maxiter; innerIter++, globalIter++)
    //         {
    //             // damped/weighted iteration formula: u_(m+1) = u_(m) + omega * D^-1 * r_(m)

    //             // switch arguments dVIn -> dVOut to use latest values in next iteration
    //             if (globalIter % 2 == 1)
    //             {
    //                 err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &dVIn);
    //                 err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &dVOut);
    //                 mgclCheckError(err, "Setting kernel arguments");
    //             }
    //             else
    //             {
    //                 err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &dVIn);
    //                 err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &dVOut);
    //                 mgclCheckError(err, "Setting kernel arguments");
    //             }

    //             // set flag to store residual in last iteration
    //             if (globalIter == maxiter - 1)
    //             {
    //                 store_res = 1;
    //                 err = clSetKernelArg(kernel, pos_storeres, sizeof(int), &store_res);
    //                 mgclCheckError(err, "Setting kernel arguments");
    //             }

    //             // recalculate and set idx_start
    //             idx_start = problem.ghosts - ((stepsPerIter - innerIter) - 1);
    //             err = clSetKernelArg(kernel, pos_idxstart, sizeof(int), &idx_start);
    //             mgclCheckError(err, "Setting kernel arguments");

    //             err = clEnqueueNDRangeKernel(problem.getOpenCLHelper().getCommands(), kernel, kernelDims, NULL, global, local, 0, NULL, &ev);
    //             mgclCheckError(err, "Enqueueing kernel");

    //             if (problem.isProfilingEnabled())
    //             {
    //                 problem.getProfilingData()->addMeasurement(problem.getCommands(), ev, kernelName,
    //                                                            {global[0], global[1], 0},
    //                                                            {local[0], local[1], 1});
    //             }
    //             mgclCheckError(clReleaseEvent(ev), "clReleaseEvent");
    //         }
    //     }

    //     if (store_res)
    //     {
    //         // TODO check for mpi
    //         err = MultigridEngine::updateGhosts(problem, level.getDR(), level.getMpiDataPtr(),
    //                                             level.isCalculatedLocally());
    //         mgclCheckError(err, "Updating ghosts of dR");
    //     }

    //     // copy result into dVIn if needed
    //     if (maxiter % 2 == 1)
    //         level.getDVOut().copyTo(problem.getOpenCLHelper().getCommands(), level.getDVIn());

    //     // Update ghosts of dVIn
    //     err = MultigridEngine::updateGhosts(problem, level.getDVIn(),
    //                                         level.getMpiDataPtr(), level.isCalculatedLocally());
    //     mgclCheckError(err, "Updating ghosts");

    //     // calculate residual and its norm
    //     if (return_residual)
    //     {
    //         // update residual to use current approximation v
    //         res = MultigridEngine::residual(problem, level, true);
    //     }

    //     clReleaseKernel(kernel);

    //     return res;
    // }

    // Benchs the various residual fixed stencil kernel versions
    // Note that, while kernel profiling works and reports adequate timings, it needs to wait for each kernel to be finished before starting the next one,
    // which might hurt performance, even when using two queues.
    TEST_CASE("benchJacobiOverlappedGhostUpdate")
    {
        using std::min;

        if (CLI_ARGS::grids.size() == 0 && (CLI_ARGS::gridsMin.size() == 0 || CLI_ARGS::gridsMax.size() == 0))
            throw "Need to specify at least one local grid size, e.g. using --grids 4,8,16 or --gridsMin 4,4,4 AND --gridsMax 32,32,32";

        if (CLI_ARGS::jacobiIters.empty())
            throw "Need to specify --jacobiIters, e.g. --jacobiIters 1,2,3";

        if (CLI_ARGS::jacobiStepsPerIter.empty())
            throw "Need to specify --spi, e.g. --spi 1,2,3";

        // build grids to be tested from CLI args
        std::vector<std::vector<int>> gridsTBT;
        for (auto N : CLI_ARGS::grids)
            gridsTBT.push_back({N, N, N});
        if (CLI_ARGS::gridsMin.size() > 0 && CLI_ARGS::gridsMax.size() > 0)
            for (int m = CLI_ARGS::gridsMin[0]; m <= CLI_ARGS::gridsMax[0]; m *= 2)
                for (int n = CLI_ARGS::gridsMin[1]; n <= CLI_ARGS::gridsMax[1]; n *= 2)
                    for (int o = CLI_ARGS::gridsMin[2]; o <= CLI_ARGS::gridsMax[2]; o *= 2)
                        gridsTBT.push_back({m, n, o});

        std::vector<bench_util::ResultMpi> results;

        bool return_residual = false;
        int ghosts = 1;
        int periodic = 1;
        std::stringstream kernelProfilesStream;

        for (auto stepsPerIter : CLI_ARGS::jacobiStepsPerIter)
            if (stepsPerIter > ghosts)
                throw "stepsPerIter must be <= ghosts. Not supported yet.";

        // check if mpi is initialized
        int isInitialized = 0;
        MPI_Initialized(&isInitialized);
        REQUIRE(isInitialized);

        MPI_Comm mpi_comm = MPI_COMM_WORLD;

        // check number of processes
        int mpi_size = -1;
        MPI_Comm_size(mpi_comm, &mpi_size);
        // REQUIRE(mpi_size == 8);

        /* MPI variables */
        int mpi_rank;
        int mpi_dims[3] = {0, 0, 0};
        int mpi_periods[3] = {periodic, periodic, periodic};
        int mpi_coords[3];

        /* Initialize cartesian process grid */
        MPI_Comm_size(mpi_comm, &mpi_size);
        MPI_Dims_create(mpi_size, 3, mpi_dims);
        MPI_Cart_create(mpi_comm, 3, mpi_dims, mpi_periods, 1, &mpi_comm);
        MPI_Comm_rank(mpi_comm, &mpi_rank);
        MPI_Cart_coords(mpi_comm, mpi_rank, 3, mpi_coords);

        for (auto gr : gridsTBT)
        {
            int ml = gr[0];
            int nl = gr[1];
            int ol = gr[2];
            int mglob = ml * mpi_dims[0];
            int nglob = nl * mpi_dims[1];
            int oglob = ol * mpi_dims[2];

            CAPTURE(ml, nl, ol, mglob, nglob, oglob);

            // print coords and boundaries per rank
            // if (mpi_rank == 0)
            //     std::cout << "rank;coords[0];coords[1];coords[2];ms;me;ns;ne;os;oe" << std::endl;

            // for (int i = 0; i < mpi_size; i++)
            // {
            //     MPI_Barrier(mpi_comm);
            //     if (mpi_rank == i)
            //     {
            //         std::cout << mpi_rank << ";" << mpi_coords[0] << ";" << mpi_coords[1] << ";" << mpi_coords[2] << ";"
            //                   << m_start << ";" << m_end << ";"
            //                   << n_start << ";" << n_end << ";"
            //                   << o_start << ";" << o_end << std::endl;
            //     }
            // }

            REQUIRE(ml > 0);
            REQUIRE(ml <= mglob);
            REQUIRE(nl > 0);
            REQUIRE(nl <= nglob);
            REQUIRE(ol > 0);
            REQUIRE(ol <= oglob);

            auto v_in = std::make_shared<mgcl::Cuboid>(ml, nl, ol, 0, 0, 0);
            auto f_in = std::make_shared<mgcl::Cuboid>(ml, nl, ol, 0, 0, 0);
            v_in->fill1dIndex(true);
            f_in->fill1dIndex(true);
            // v_in->fillRandom();
            // f_in->fillRandom();

            // Create dummy problem to initialize OpenCL
            mgcl::Problem p(ml, nl, ol, f_in, v_in, mglob, nglob, oglob);
            p.setSilent(true);
            p.setKernelFile("residual_kernels_inner_vs_boundary.cl");
            if (CLI_ARGS::useBinaryFile)
            {
                p.setBinaryFile("residualInnerVsBoundary.bin");
            }
            p.setUseOpencl(true);
            p.setGhosts(ghosts);
            p.setStencilType(mgcl::MGCL_VARYING);
            p.setDeviceType(CL_DEVICE_TYPE_GPU);
            p.setDeviceStrategy(mgcl::OCL_DEVICE_STRATEGY::DISTRIBUTE_EVENLY);
            p.setProfilingEnabled(CLI_ARGS::enableKernelProfiling);
            auto sv = p.createStencilValues();
            mgcl_test::fill27pLaplace(*sv, 1.0 / static_cast<double>(mglob), false);
            p.setMpiComm(mpi_comm);

            auto& conf = p.getKernelConfig();
            // Jacobi kernels
            conf["jacobi_iter_27point_varying_stencil_1d_boundary"] = mgcl::conf::KernelWorkgroupSizes{{1, {128, 1, 1}}};
            conf["jacobi_iter_27point_fixed_stencil_1d_boundary"] = mgcl::conf::KernelWorkgroupSizes{{1, {128, 1, 1}}};
            conf["jacobi_iter_7point_boundary"] = mgcl::conf::KernelWorkgroupSizes{{1, {1, 64, 1}}};
            conf["jacobi_iter_19point_boundary"] = mgcl::conf::KernelWorkgroupSizes{{1, {1, 64, 1}}};
            conf["jacobi_iter_27point_boundary"] = mgcl::conf::KernelWorkgroupSizes{{1, {1, 64, 1}}};
            conf["jacobi_iter_27point_varying_stencil_1d_inner"] = mgcl::conf::KernelWorkgroupSizes{{1, {128, 1, 1}}};
            conf["jacobi_iter_27point_fixed_stencil_1d_inner"] = mgcl::conf::KernelWorkgroupSizes{{1, {128, 1, 1}}};
            conf["jacobi_iter_7point_inner"] = mgcl::conf::KernelWorkgroupSizes{{1, {1, 64, 1}}};
            conf["jacobi_iter_19point_inner"] = mgcl::conf::KernelWorkgroupSizes{{1, {1, 64, 1}}};
            conf["jacobi_iter_27point_inner"] = mgcl::conf::KernelWorkgroupSizes{{1, {1, 64, 1}}};
            p.init();

            if (CLI_ARGS::enableKernelProfiling)
                p.getProfilingData()->getMeasurements().clear();

            auto& lv0 = p.getLevelAt(0);

            ankerl::nanobench::Bench bench;
            bench.timeUnit(1ms, "ms")
                .epochs(CLI_ARGS::bench_epochs)
                .epochIterations(CLI_ARGS::bench_iterations)
                .relative(false);

            if (mpi_rank > 0)
                bench.output(nullptr);

            if (CLI_ARGS::checkResults)
            {
                bench.epochs(1).epochIterations(1);
            }

            for (auto maxiter : CLI_ARGS::jacobiIters)
                for (auto stepsPerIter : CLI_ARGS::jacobiStepsPerIter)
                {
                    if (stepsPerIter > maxiter)
                        continue;

                    std::unique_ptr<mgcl::Cuboid> v_out_default = nullptr;
                    std::unique_ptr<mgcl::Cuboid> v_out_overlapped = nullptr;

                    {
                        lv0.getDVIn().fill(p.getProgram(), p.getCommands(), 0.0, false, nullptr, nullptr);
                        lv0.getDVIn().fill1dIndex(p.getProgram(), p.getCommands(), true, true, nullptr, nullptr);

                        std::string name = std::string("jacobi_default_")
                                               .append(std::to_string(maxiter))
                                               .append("iters_")
                                               .append(std::to_string(mglob))
                                               .append("_")
                                               .append(std::to_string(nglob))
                                               .append("_")
                                               .append(std::to_string(oglob));

                        bench.run(std::string(name).c_str(), [&] { //
                            jacobiDefault(p, lv0, maxiter, return_residual, stepsPerIter);
                            p.finish();
                        });

                        bench_util::ResultMpi res;
                        res.name = name;
                        res.minTime = bench_util::getMinTime(bench, name);
                        res.medianTime = bench_util::getMedianTime(bench, name);
                        res.avgTime = bench_util::getAvgTime(bench, name);
                        res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
                        res.m = ml;
                        res.n = nl;
                        res.o = ol;
                        res.mglob = mglob;
                        res.nglob = nglob;
                        res.oglob = oglob;
                        res.gpus = mpi_size;
                        res.LT = -1;
                        results.push_back(res);

                        if (CLI_ARGS::checkResults)
                        {
                            v_out_default = std::make_unique<mgcl::Cuboid>(ml, nl, ol, ghosts, ghosts, ghosts);
                            lv0.getDVIn().read(p.getCommands(), v_out_default.get(), true);
                        }
                    }

                    {
                        lv0.getDVIn().fill(p.getProgram(), p.getCommands(), 0.0, false, nullptr, nullptr);
                        lv0.getDVIn().fill1dIndex(p.getProgram(), p.getCommands(), true, true, nullptr, nullptr);

                        int err;
                        cl_command_queue_properties props = p.isProfilingEnabled() ? CL_QUEUE_PROFILING_ENABLE : 0;
                        cl_command_queue queue2 = clCreateCommandQueue(p.getContext(), p.getOpenCLHelper().getDeviceId(), props, &err);
                        mgcl::mgclCheckError(err, "Creating command queue");

                        std::string name = std::string("jacobi_overlapped_boundaryInnerParallel_")
                                               .append(std::to_string(maxiter))
                                               .append("iters_")
                                               .append(std::to_string(mglob))
                                               .append("_")
                                               .append(std::to_string(nglob))
                                               .append("_")
                                               .append(std::to_string(oglob));

                        bench.run(std::string(name).c_str(), [&] { //
                            jacobiOverlapped(p, lv0, maxiter, return_residual, stepsPerIter, queue2, OverlappedKernelVersion::BOUNDARY_INNER_PARALLEL);
                            p.finish();
                        });

                        bench_util::ResultMpi res;
                        res.name = name;
                        res.minTime = bench_util::getMinTime(bench, name);
                        res.medianTime = bench_util::getMedianTime(bench, name);
                        res.avgTime = bench_util::getAvgTime(bench, name);
                        res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
                        res.m = ml;
                        res.n = nl;
                        res.o = ol;
                        res.mglob = mglob;
                        res.nglob = nglob;
                        res.oglob = oglob;
                        res.gpus = mpi_size;
                        res.LT = -1;
                        results.push_back(res);

                        if (CLI_ARGS::checkResults)
                        {
                            v_out_overlapped = std::make_unique<mgcl::Cuboid>(ml, nl, ol, ghosts, ghosts, ghosts);
                            lv0.getDVIn().read(p.getCommands(), v_out_overlapped.get(), true);
                        }
                    }

                    {
                        lv0.getDVIn().fill(p.getProgram(), p.getCommands(), 0.0, false, nullptr, nullptr);
                        lv0.getDVIn().fill1dIndex(p.getProgram(), p.getCommands(), true, true, nullptr, nullptr);

                        int err;
                        cl_command_queue_properties props = p.isProfilingEnabled() ? CL_QUEUE_PROFILING_ENABLE : 0;
                        cl_command_queue queue2 = clCreateCommandQueue(p.getContext(), p.getOpenCLHelper().getDeviceId(), props, &err);
                        mgcl::mgclCheckError(err, "Creating command queue");

                        std::string name = std::string("jacobi_overlapped_extractInnerParallel")
                                               .append(std::to_string(maxiter))
                                               .append("iters_")
                                               .append(std::to_string(mglob))
                                               .append("_")
                                               .append(std::to_string(nglob))
                                               .append("_")
                                               .append(std::to_string(oglob));

                        bench.run(std::string(name).c_str(), [&] { //
                            jacobiOverlapped(p, lv0, maxiter, return_residual, stepsPerIter, queue2, OverlappedKernelVersion::EXTRACT_INNER_PARALLEL);
                            p.finish();
                        });

                        bench_util::ResultMpi res;
                        res.name = name;
                        res.minTime = bench_util::getMinTime(bench, name);
                        res.medianTime = bench_util::getMedianTime(bench, name);
                        res.avgTime = bench_util::getAvgTime(bench, name);
                        res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
                        res.m = ml;
                        res.n = nl;
                        res.o = ol;
                        res.mglob = mglob;
                        res.nglob = nglob;
                        res.oglob = oglob;
                        res.gpus = mpi_size;
                        res.LT = -1;
                        results.push_back(res);

                        if (CLI_ARGS::checkResults)
                        {
                            v_out_overlapped = std::make_unique<mgcl::Cuboid>(ml, nl, ol, ghosts, ghosts, ghosts);
                            lv0.getDVIn().read(p.getCommands(), v_out_overlapped.get(), true);
                        }
                    }

                    {
                        lv0.getDVIn().fill(p.getProgram(), p.getCommands(), 0.0, false, nullptr, nullptr);
                        lv0.getDVIn().fill1dIndex(p.getProgram(), p.getCommands(), true, true, nullptr, nullptr);

                        int err;
                        cl_command_queue_properties props = p.isProfilingEnabled() ? CL_QUEUE_PROFILING_ENABLE : 0;
                        cl_command_queue queue2 = clCreateCommandQueue(p.getContext(), p.getOpenCLHelper().getDeviceId(), props, &err);
                        mgcl::mgclCheckError(err, "Creating command queue");

                        std::string name = std::string("jacobi_overlapped_memcpyInnerParallel")
                                               .append(std::to_string(maxiter))
                                               .append("iters_")
                                               .append(std::to_string(mglob))
                                               .append("_")
                                               .append(std::to_string(nglob))
                                               .append("_")
                                               .append(std::to_string(oglob));

                        bench.run(std::string(name).c_str(), [&] { //
                            jacobiOverlapped(p, lv0, maxiter, return_residual, stepsPerIter, queue2, OverlappedKernelVersion::MEMCPY_INNER_PARALLEL);
                            p.finish();
                        });

                        bench_util::ResultMpi res;
                        res.name = name;
                        res.minTime = bench_util::getMinTime(bench, name);
                        res.medianTime = bench_util::getMedianTime(bench, name);
                        res.avgTime = bench_util::getAvgTime(bench, name);
                        res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
                        res.m = ml;
                        res.n = nl;
                        res.o = ol;
                        res.mglob = mglob;
                        res.nglob = nglob;
                        res.oglob = oglob;
                        res.gpus = mpi_size;
                        res.LT = -1;
                        results.push_back(res);

                        if (CLI_ARGS::checkResults)
                        {
                            v_out_overlapped = std::make_unique<mgcl::Cuboid>(ml, nl, ol, ghosts, ghosts, ghosts);
                            lv0.getDVIn().read(p.getCommands(), v_out_overlapped.get(), true);
                        }
                    }

                    // Check results for kernels that it is valid for
                    if (CLI_ARGS::checkResults)
                    {
                        // v_out_default->dumpToFile("v_out_default_" + std::to_string(mpi_rank) + ".txt", false);
                        // v_out_overlapped->dumpToFile("v_out_overlapped_" + std::to_string(mpi_rank) + ".txt", false);
                        REQUIRE(v_out_default->isEqual(*v_out_overlapped));
                        std::cout << "Results seem good on rank " << mpi_rank << std::endl;
                    }

                    if (CLI_ARGS::enableKernelProfiling)
                    {
                        p.getProfilingData()->printBestTimingsPerKernel(kernelProfilesStream);
                    }
                }
        }

        MPI_Barrier(mpi_comm);
        bench_util::printCsvFormat(results, mpi_comm, mpi_rank);
        MPI_Barrier(mpi_comm);

        if (CLI_ARGS::enableKernelProfiling)
        {
            kernelProfilesStream << "rank: " << mpi_rank << std::endl;
            std::cout << kernelProfilesStream.str() << std::endl;
        }
        MPI_Barrier(mpi_comm);
    }
}