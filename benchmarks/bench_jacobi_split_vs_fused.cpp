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
#include "../src/mgcl/multigrid_engine.hpp"
#include "../src/mgcl/problem.hpp"
#include "../test/test_utility.hpp"
#include "cli_args.hpp"

namespace mgcl_bench_jacobi_split_vs_fused
{
    enum class KernelVersion
    {
        DEFAULT,
        OVERLAPPED
    };

    using size_t3 = struct
    {
        int x, y, z;
    };

    // Regular jacobi method like in production code
    double jacobiFused(mgcl::Problem& problem, mgcl::Level& level, int maxiter, bool return_residual, int stepsPerIter)
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
                err = mgcl::MultigridEngine::updateGhosts(problem, level, level.getDVOut(),
                                                          level.getMpiDataPtr(), level.isCalculatedLocally());
                mgcl::mgclCheckError(err, "Updating ghosts");
            }
            else
            {
                err = mgcl::MultigridEngine::updateGhosts(problem, level, level.getDVIn(),
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
        {
            // level.getDVOut().copyTo(problem.getOpenCLHelper().getCommands(), level.getDVIn());
            mgcl::CuboidGpu::swap(level.getDVIn(), level.getDVOut());
        }

        // Update ghosts of dVIn
        err = mgcl::MultigridEngine::updateGhosts(problem, level, level.getDVIn(),
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

    // Jacobi having residual and jacobi update step split, requiring only a single v array.
    double jacobiSplit(mgcl::Problem& problem, mgcl::Level& level, int maxiter, bool return_residual, int stepsPerIter)
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

        cl_event evJacobi;
        cl_event evResidual;

        double h2 = 1.0 / static_cast<double>((problem.getMGlobal() >> level.getNum()) * (problem.getMGlobal() >> level.getNum()));
        double dinv = h2 / 6.0;
        double h2inv = level.getStencilFactor(); // divisor of the stencil, inverted to use * instead of / in kernel
        // TODO refactor stencilFactor

        // Create the compute kernel from the program
        const char* kernelNameJacobiUpdate;
        // if (problem.getStencilType() == mgcl::MGCL_LAPLACE_7POINT)
        //     kernelNameJacobiUpdate = "jacobi_iter_7point";
        // else if (problem.getStencilType() == mgcl::MGCL_LAPLACE_19POINT)
        // {
        //     kernelNameJacobiUpdate = "jacobi_iter_19point";
        //     dinv = (6.0 * h2) / 24.0;
        // }
        // else if (problem.getStencilType() == mgcl::MGCL_LAPLACE_27POINT)
        // {
        //     kernelNameJacobiUpdate = "jacobi_iter_27point";
        //     dinv = (26.0 * h2) / 88.0;
        // }
        // else if (problem.getStencilType() == mgcl::MGCL_VARYING)
        // {
        kernelNameJacobiUpdate = "jacobi_iter_27point_varying_stencil_1d_update_step_only";
        // }
        // else if (problem.getStencilType() == mgcl::MGCL_FIXED)
        // {
        //     kernelNameJacobiUpdate = "jacobi_iter_27point_fixed_stencil_1d";
        // }

        cl_kernel kernelJacobiUpdate = clCreateKernel(problem.getProgram(), kernelNameJacobiUpdate, &err);
        mgcl::mgclCheckError(err, "Creating kernel jacobi update");

        // Create the compute kernel from the program
        const char* kernelNameResidual;
        if (problem.getStencilType() == mgcl::MGCL_LAPLACE_7POINT)
            kernelNameResidual = "residual_7point";
        else if (problem.getStencilType() == mgcl::MGCL_LAPLACE_19POINT)
        {
            kernelNameResidual = "residual_19point";
            h2inv = 1.0 / (6.0 * h2);
        }
        else if (problem.getStencilType() == mgcl::MGCL_LAPLACE_27POINT)
        {
            kernelNameResidual = "residual_27point";
            h2inv = 1.0 / (26.0 * h2);
        }
        else if (problem.getStencilType() == mgcl::MGCL_VARYING)
        {
            kernelNameResidual = "residual_27point_varying_stencil";
        }
        else if (problem.getStencilType() == mgcl::MGCL_FIXED)
        {
            kernelNameResidual = "residual_27point_fixed_stencil";
        }

        cl_kernel kernelResidual = clCreateKernel(problem.getProgram(), kernelNameResidual, &err);
        mgcl::mgclCheckError(err, "Creating kernel residual");

        cl_mem dVIn = level.getDVIn().getBuffer();
        cl_mem dF = level.getDF().getBuffer();
        cl_mem dR = level.getDR().getBuffer();

        // assign kernel arguments
        int pos = 0;
        int pos_idxstart = -1;

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
            err = clSetKernelArg(kernelJacobiUpdate, pos, sizeof(cl_mem), &dVIn);
            mgcl::mgclCheckError(err, "Setting Jacobi step kernel arguments");
            err |= clSetKernelArg(kernelJacobiUpdate, ++pos, sizeof(cl_mem), &dF);
            mgcl::mgclCheckError(err, "Setting Jacobi step kernel arguments");
            err |= clSetKernelArg(kernelJacobiUpdate, ++pos, sizeof(cl_mem), &dR);
            mgcl::mgclCheckError(err, "Setting Jacobi step kernel arguments");
            err |= clSetKernelArg(kernelJacobiUpdate, ++pos, sizeof(cl_mem), &svbuf);
            mgcl::mgclCheckError(err, "Setting Jacobi step kernel arguments");
            err |= clSetKernelArg(kernelJacobiUpdate, ++pos, sizeof(double), &omega);
            mgcl::mgclCheckError(err, "Setting Jacobi step kernel arguments");
            err |= clSetKernelArg(kernelJacobiUpdate, ++pos, sizeof(int), &mgh);
            mgcl::mgclCheckError(err, "Setting Jacobi step kernel arguments");
            err |= clSetKernelArg(kernelJacobiUpdate, ++pos, sizeof(int), &ngh);
            mgcl::mgclCheckError(err, "Setting Jacobi step kernel arguments");
            err |= clSetKernelArg(kernelJacobiUpdate, ++pos, sizeof(int), &ogh);
            mgcl::mgclCheckError(err, "Setting Jacobi step kernel arguments");
            err |= clSetKernelArg(kernelJacobiUpdate, ++pos, sizeof(int), &svmgh);
            err |= clSetKernelArg(kernelJacobiUpdate, ++pos, sizeof(int), &svngh);
            err |= clSetKernelArg(kernelJacobiUpdate, ++pos, sizeof(int), &svogh);
            err |= clSetKernelArg(kernelJacobiUpdate, ++pos, sizeof(int), &ghosts);
            err |= clSetKernelArg(kernelJacobiUpdate, ++pos, sizeof(int), &svgh);
            err |= clSetKernelArg(kernelJacobiUpdate, ++pos, sizeof(int), &svGridSize);
            err |= clSetKernelArg(kernelJacobiUpdate, ++pos, sizeof(int), &idx_start);
            pos_idxstart = pos;
        }
        else if (problem.getStencilType() == mgcl::MGCL_FIXED)
        {
            auto& fs = *level.getFixedStencil();
            err = clSetKernelArg(kernelJacobiUpdate, pos, sizeof(cl_mem), &dVIn);
            err |= clSetKernelArg(kernelJacobiUpdate, ++pos, sizeof(cl_mem), &dF);
            err |= clSetKernelArg(kernelJacobiUpdate, ++pos, sizeof(cl_mem), &dR);
            err |= clSetKernelArg(kernelJacobiUpdate, ++pos, sizeof(double), &omega);
            err |= clSetKernelArg(kernelJacobiUpdate, ++pos, sizeof(int), &mgh);
            err |= clSetKernelArg(kernelJacobiUpdate, ++pos, sizeof(int), &ngh);
            err |= clSetKernelArg(kernelJacobiUpdate, ++pos, sizeof(int), &ogh);
            err |= clSetKernelArg(kernelJacobiUpdate, ++pos, sizeof(int), &ghosts);
            err |= clSetKernelArg(kernelJacobiUpdate, ++pos, sizeof(int), &idx_start);
            pos_idxstart = pos;
            err |= clSetKernelArg(kernelJacobiUpdate, ++pos, sizeof(double), &fs[0][0][0]);
            err |= clSetKernelArg(kernelJacobiUpdate, ++pos, sizeof(double), &fs[0][0][1]);
            err |= clSetKernelArg(kernelJacobiUpdate, ++pos, sizeof(double), &fs[0][0][2]);
            err |= clSetKernelArg(kernelJacobiUpdate, ++pos, sizeof(double), &fs[0][1][0]);
            err |= clSetKernelArg(kernelJacobiUpdate, ++pos, sizeof(double), &fs[0][1][1]);
            err |= clSetKernelArg(kernelJacobiUpdate, ++pos, sizeof(double), &fs[0][1][2]);
            err |= clSetKernelArg(kernelJacobiUpdate, ++pos, sizeof(double), &fs[0][2][0]);
            err |= clSetKernelArg(kernelJacobiUpdate, ++pos, sizeof(double), &fs[0][2][1]);
            err |= clSetKernelArg(kernelJacobiUpdate, ++pos, sizeof(double), &fs[0][2][2]);
            err |= clSetKernelArg(kernelJacobiUpdate, ++pos, sizeof(double), &fs[1][0][0]);
            err |= clSetKernelArg(kernelJacobiUpdate, ++pos, sizeof(double), &fs[1][0][1]);
            err |= clSetKernelArg(kernelJacobiUpdate, ++pos, sizeof(double), &fs[1][0][2]);
            err |= clSetKernelArg(kernelJacobiUpdate, ++pos, sizeof(double), &fs[1][1][0]);
            err |= clSetKernelArg(kernelJacobiUpdate, ++pos, sizeof(double), &fs[1][1][1]);
            err |= clSetKernelArg(kernelJacobiUpdate, ++pos, sizeof(double), &fs[1][1][2]);
            err |= clSetKernelArg(kernelJacobiUpdate, ++pos, sizeof(double), &fs[1][2][0]);
            err |= clSetKernelArg(kernelJacobiUpdate, ++pos, sizeof(double), &fs[1][2][1]);
            err |= clSetKernelArg(kernelJacobiUpdate, ++pos, sizeof(double), &fs[1][2][2]);
            err |= clSetKernelArg(kernelJacobiUpdate, ++pos, sizeof(double), &fs[2][0][0]);
            err |= clSetKernelArg(kernelJacobiUpdate, ++pos, sizeof(double), &fs[2][0][1]);
            err |= clSetKernelArg(kernelJacobiUpdate, ++pos, sizeof(double), &fs[2][0][2]);
            err |= clSetKernelArg(kernelJacobiUpdate, ++pos, sizeof(double), &fs[2][1][0]);
            err |= clSetKernelArg(kernelJacobiUpdate, ++pos, sizeof(double), &fs[2][1][1]);
            err |= clSetKernelArg(kernelJacobiUpdate, ++pos, sizeof(double), &fs[2][1][2]);
            err |= clSetKernelArg(kernelJacobiUpdate, ++pos, sizeof(double), &fs[2][2][0]);
            err |= clSetKernelArg(kernelJacobiUpdate, ++pos, sizeof(double), &fs[2][2][1]);
            err |= clSetKernelArg(kernelJacobiUpdate, ++pos, sizeof(double), &fs[2][2][2]);
        }
        else
        {
            err = clSetKernelArg(kernelJacobiUpdate, pos, sizeof(cl_mem), &dVIn);
            err |= clSetKernelArg(kernelJacobiUpdate, ++pos, sizeof(cl_mem), &dF);
            err |= clSetKernelArg(kernelJacobiUpdate, ++pos, sizeof(cl_mem), &dR);
            err |= clSetKernelArg(kernelJacobiUpdate, ++pos, sizeof(double), &h2inv);
            err |= clSetKernelArg(kernelJacobiUpdate, ++pos, sizeof(double), &dinv);
            err |= clSetKernelArg(kernelJacobiUpdate, ++pos, sizeof(double), &omega);
            err |= clSetKernelArg(kernelJacobiUpdate, ++pos, sizeof(int), &mgh);
            err |= clSetKernelArg(kernelJacobiUpdate, ++pos, sizeof(int), &ngh);
            err |= clSetKernelArg(kernelJacobiUpdate, ++pos, sizeof(int), &ogh);
            err |= clSetKernelArg(kernelJacobiUpdate, ++pos, sizeof(int), &ghosts);
            err |= clSetKernelArg(kernelJacobiUpdate, ++pos, sizeof(int), &idx_start);
            pos_idxstart = pos;
        }
        mgcl::mgclCheckError(err, "Setting Jacobi step kernel arguments");

        // assign residual kernel arguments
        pos = 0;
        int moff = 0;
        int noff = 0;
        int ooff = 0;
        if (problem.getStencilType() == mgcl::MGCL_VARYING)
        {
            auto svbuf = level.getStencilValuesGpu()->getBuf();
            int svgh = level.getStencilValuesGpu()->getGh();
            int svmgh = level.getStencilValuesGpu()->getMgh();
            int svngh = level.getStencilValuesGpu()->getNgh();
            int svogh = level.getStencilValuesGpu()->getOgh();
            int svGridSize = svmgh * svngh * svogh;

            err = clSetKernelArg(kernelResidual, pos, sizeof(cl_mem), &dVIn);
            err |= clSetKernelArg(kernelResidual, ++pos, sizeof(cl_mem), &dF);
            err |= clSetKernelArg(kernelResidual, ++pos, sizeof(cl_mem), &dR);
            err |= clSetKernelArg(kernelResidual, ++pos, sizeof(cl_mem), &svbuf);
            err |= clSetKernelArg(kernelResidual, ++pos, sizeof(int), &mgh);
            err |= clSetKernelArg(kernelResidual, ++pos, sizeof(int), &ngh);
            err |= clSetKernelArg(kernelResidual, ++pos, sizeof(int), &ogh);
            err |= clSetKernelArg(kernelResidual, ++pos, sizeof(int), &svmgh);
            err |= clSetKernelArg(kernelResidual, ++pos, sizeof(int), &svngh);
            err |= clSetKernelArg(kernelResidual, ++pos, sizeof(int), &svogh);
            err |= clSetKernelArg(kernelResidual, ++pos, sizeof(int), &ghosts);
            err |= clSetKernelArg(kernelResidual, ++pos, sizeof(int), &svgh);
            err |= clSetKernelArg(kernelResidual, ++pos, sizeof(int), &svGridSize);
            err |= clSetKernelArg(kernelResidual, ++pos, sizeof(int), &moff);
            err |= clSetKernelArg(kernelResidual, ++pos, sizeof(int), &noff);
            err |= clSetKernelArg(kernelResidual, ++pos, sizeof(int), &ooff);
        }
        else if (problem.getStencilType() == mgcl::MGCL_FIXED)
        {
            auto& fs = *level.getFixedStencil();
            err = clSetKernelArg(kernelResidual, pos, sizeof(cl_mem), &dVIn);
            err |= clSetKernelArg(kernelResidual, ++pos, sizeof(cl_mem), &dF);
            err |= clSetKernelArg(kernelResidual, ++pos, sizeof(cl_mem), &dR);
            err |= clSetKernelArg(kernelResidual, ++pos, sizeof(int), &mgh);
            err |= clSetKernelArg(kernelResidual, ++pos, sizeof(int), &ngh);
            err |= clSetKernelArg(kernelResidual, ++pos, sizeof(int), &ogh);
            err |= clSetKernelArg(kernelResidual, ++pos, sizeof(int), &ghosts);
            err |= clSetKernelArg(kernelResidual, ++pos, sizeof(int), &moff);
            err |= clSetKernelArg(kernelResidual, ++pos, sizeof(int), &noff);
            err |= clSetKernelArg(kernelResidual, ++pos, sizeof(int), &ooff);
            err |= clSetKernelArg(kernelResidual, ++pos, sizeof(double), &fs[0][0][0]);
            err |= clSetKernelArg(kernelResidual, ++pos, sizeof(double), &fs[0][0][1]);
            err |= clSetKernelArg(kernelResidual, ++pos, sizeof(double), &fs[0][0][2]);
            err |= clSetKernelArg(kernelResidual, ++pos, sizeof(double), &fs[0][1][0]);
            err |= clSetKernelArg(kernelResidual, ++pos, sizeof(double), &fs[0][1][1]);
            err |= clSetKernelArg(kernelResidual, ++pos, sizeof(double), &fs[0][1][2]);
            err |= clSetKernelArg(kernelResidual, ++pos, sizeof(double), &fs[0][2][0]);
            err |= clSetKernelArg(kernelResidual, ++pos, sizeof(double), &fs[0][2][1]);
            err |= clSetKernelArg(kernelResidual, ++pos, sizeof(double), &fs[0][2][2]);
            err |= clSetKernelArg(kernelResidual, ++pos, sizeof(double), &fs[1][0][0]);
            err |= clSetKernelArg(kernelResidual, ++pos, sizeof(double), &fs[1][0][1]);
            err |= clSetKernelArg(kernelResidual, ++pos, sizeof(double), &fs[1][0][2]);
            err |= clSetKernelArg(kernelResidual, ++pos, sizeof(double), &fs[1][1][0]);
            err |= clSetKernelArg(kernelResidual, ++pos, sizeof(double), &fs[1][1][1]);
            err |= clSetKernelArg(kernelResidual, ++pos, sizeof(double), &fs[1][1][2]);
            err |= clSetKernelArg(kernelResidual, ++pos, sizeof(double), &fs[1][2][0]);
            err |= clSetKernelArg(kernelResidual, ++pos, sizeof(double), &fs[1][2][1]);
            err |= clSetKernelArg(kernelResidual, ++pos, sizeof(double), &fs[1][2][2]);
            err |= clSetKernelArg(kernelResidual, ++pos, sizeof(double), &fs[2][0][0]);
            err |= clSetKernelArg(kernelResidual, ++pos, sizeof(double), &fs[2][0][1]);
            err |= clSetKernelArg(kernelResidual, ++pos, sizeof(double), &fs[2][0][2]);
            err |= clSetKernelArg(kernelResidual, ++pos, sizeof(double), &fs[2][1][0]);
            err |= clSetKernelArg(kernelResidual, ++pos, sizeof(double), &fs[2][1][1]);
            err |= clSetKernelArg(kernelResidual, ++pos, sizeof(double), &fs[2][1][2]);
            err |= clSetKernelArg(kernelResidual, ++pos, sizeof(double), &fs[2][2][0]);
            err |= clSetKernelArg(kernelResidual, ++pos, sizeof(double), &fs[2][2][1]);
            err |= clSetKernelArg(kernelResidual, ++pos, sizeof(double), &fs[2][2][2]);
        }
        else
        {
            err = clSetKernelArg(kernelResidual, pos, sizeof(cl_mem), &dVIn);
            err |= clSetKernelArg(kernelResidual, ++pos, sizeof(cl_mem), &dF);
            err |= clSetKernelArg(kernelResidual, ++pos, sizeof(cl_mem), &dR);
            err |= clSetKernelArg(kernelResidual, ++pos, sizeof(double), &h2inv);
            err |= clSetKernelArg(kernelResidual, ++pos, sizeof(int), &mgh);
            err |= clSetKernelArg(kernelResidual, ++pos, sizeof(int), &ngh);
            err |= clSetKernelArg(kernelResidual, ++pos, sizeof(int), &ogh);
            err |= clSetKernelArg(kernelResidual, ++pos, sizeof(int), &ghosts);
            err |= clSetKernelArg(kernelResidual, ++pos, sizeof(int), &moff);
            err |= clSetKernelArg(kernelResidual, ++pos, sizeof(int), &noff);
            err |= clSetKernelArg(kernelResidual, ++pos, sizeof(int), &ooff);
        }

        mgcl::mgclCheckError(err, "Setting residual kernel arguments");

        // One work-item per cell (including ghost cells).
        size_t global[2] = {static_cast<size_t>(mgh * ngh * ogh), static_cast<size_t>(0)};
        const auto& c = mgcl::conf::getWorkGroupSizeForKernelAndWiCount(problem.getKernelConfig(), kernelNameJacobiUpdate, 1);
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
            err = mgcl::MultigridEngine::updateGhosts(problem, level, level.getDVIn(),
                                                      level.getMpiDataPtr(), level.isCalculatedLocally());
            mgcl::mgclCheckError(err, "Updating ghosts");

            // if stepsPerIter > 1, multiple iterations can be done without updating ghosts in-between
            for (int innerIter = 0; innerIter < stepsPerIter && globalIter < maxiter; innerIter++, globalIter++)
            {
                // damped/weighted iteration formula: u_(m+1) = u_(m) + omega * D^-1 * r_(m)

                // recalculate and set idx_start
                idx_start = problem.getGhosts() - ((stepsPerIter - innerIter) - 1);
                err = clSetKernelArg(kernelJacobiUpdate, pos_idxstart, sizeof(int), &idx_start);
                mgcl::mgclCheckError(err, "Setting kernel arguments");

                err = clEnqueueNDRangeKernel(problem.getOpenCLHelper().getCommands(), kernelResidual, kernelDims, NULL, global, local, 0, NULL, &evResidual);
                mgcl::mgclCheckError(err, "Enqueueing Residual kernel");
                err = clEnqueueNDRangeKernel(problem.getOpenCLHelper().getCommands(), kernelJacobiUpdate, kernelDims, NULL, global, local, 0, NULL, &evJacobi);
                mgcl::mgclCheckError(err, "Enqueueing Jacobi step kernel");

                if (problem.isProfilingEnabled())
                {
                    problem.getProfilingData()->addMeasurement(problem.getCommands(), evResidual, kernelNameResidual,
                                                               {global[0], global[1], 0},
                                                               {local[0], local[1], 1});
                    problem.getProfilingData()->addMeasurement(problem.getCommands(), evJacobi, kernelNameJacobiUpdate,
                                                               {global[0], global[1], 0},
                                                               {local[0], local[1], 1});
                }
                mgcl::mgclCheckError(clReleaseEvent(evResidual), "clReleaseEvent");
                mgcl::mgclCheckError(clReleaseEvent(evJacobi), "clReleaseEvent");
            }
        }

        // if (store_res)
        // {
        // TODO check for mpi
        // err = mgcl::MultigridEngine::updateGhosts(problem, level.getDR(), level.getMpiDataPtr(),
        //                                           level.isCalculatedLocally());
        // mgcl::mgclCheckError(err, "Updating ghosts of dR");
        // }

        // Update ghosts of dVIn
        err = mgcl::MultigridEngine::updateGhosts(problem, level, level.getDVIn(),
                                                  level.getMpiDataPtr(), level.isCalculatedLocally());
        mgcl::mgclCheckError(err, "Updating ghosts");

        // calculate residual and its norm
        if (return_residual)
        {
            // update residual to use current approximation v
            res = mgcl::MultigridEngine::residual(problem, level, true);
        }

        clReleaseKernel(kernelJacobiUpdate);
        clReleaseKernel(kernelResidual);

        return res;
    }

    // Benchs the various residual fixed stencil kernel versions
    // Note that, while kernel profiling works and reports adequate timings, it needs to wait for each kernel to be finished before starting the next one,
    // which might hurt performance, even when using two queues.
    TEST_CASE("benchJacobiSplitVsFused")
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
            p.setKernelFile("jacobi_kernels_split_vs_fused.cl");
            if (CLI_ARGS::useBinaryFile)
            {
                p.setBinaryFile("jacobi_kernels_inner_vs_boundary.bin");
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
            conf["jacobi_iter_27point_varying_stencil_1d_update_step_only"] = mgcl::conf::KernelWorkgroupSizes{{1, {32, 1, 1}}};
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
                    std::unique_ptr<mgcl::Cuboid> v_out_split = nullptr;

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
                            jacobiFused(p, lv0, maxiter, return_residual, stepsPerIter);
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

                        std::string name = std::string("jacobi_split_")
                                               .append(std::to_string(maxiter))
                                               .append("iters_")
                                               .append(std::to_string(mglob))
                                               .append("_")
                                               .append(std::to_string(nglob))
                                               .append("_")
                                               .append(std::to_string(oglob));

                        bench.run(std::string(name).c_str(), [&] { //
                            jacobiSplit(p, lv0, maxiter, return_residual, stepsPerIter);
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
                            v_out_split = std::make_unique<mgcl::Cuboid>(ml, nl, ol, ghosts, ghosts, ghosts);
                            lv0.getDVIn().read(p.getCommands(), v_out_split.get(), true);
                        }
                    }

                    // Check results for kernels that it is valid for
                    if (CLI_ARGS::checkResults)
                    {
                        // v_out_default->dumpToFile("v_out_default_" + std::to_string(mpi_rank) + ".txt", false);
                        // v_out_overlapped->dumpToFile("v_out_overlapped_" + std::to_string(mpi_rank) + ".txt", false);
                        REQUIRE(v_out_default->isEqual(*v_out_split));
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