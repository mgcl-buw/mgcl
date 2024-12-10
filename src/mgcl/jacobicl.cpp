#include "cuboid.hpp"    // for Cuboid
#include "hypercube.hpp" // for Hypercube6d
#include "level.hpp"     // for Level
#include "mgcl.hpp"      // for mgcl_debug, MGCL_LAPLACE_19POINT
#include "mpi_level_data.hpp"
#include "multigrid_engine.hpp" // for Problem, VaryingStencil3x3x3, Multig...
#include "opencl_helper.hpp"
#include "problem.hpp" // for Problem
#include "stencil.hpp" // for mgclCheckError, VaryingStencil3x3x3
#include "util.hpp"

#include <cstddef>
#include <cstdio> // for printf, size_t, NULL
// #include <iostream>
#include <math.h> // for fabs, sqrt, ceil

#ifdef __APPLE__
#include <OpenCL/cl.h>          // for clSetKernelArg, _cl_mem, cl_mem, clE...
#include <OpenCL/cl_platform.h> // for cl_ulong
#else
#include <CL/cl.h>          // for clSetKernelArg, _cl_mem, cl_mem, clE...
#include <CL/cl_platform.h> // for cl_ulong
#endif

namespace mgcl
{
    /* Runs jacobi method sequentially.
     * v, f and r must be of size [m+2][n+2][o+2] for periodic boundary condition.
     * m,n,o is size of real grid
     * stepsPerIter is the amount of iterations done without ghost update in-between. v and r must have adequate ghost
     * cells, i.e. v_gh >= stepsPerIter per border. Defaults to 1. */
    double MultigridEngine::jacobiSeq(Cuboid& v, Cuboid& f, Cuboid& r, double omega, double h2,
                                      int maxiter, MGCL_RESIDUAL_NORM resnorm, MGCL_STENCIL stencilType,
                                      double stencilFactor, VaryingStencil* stencilValues,
                                      FixedStencil* fixedStencil, bool returnResidualNorm,
                                      bool periodic, bool updateGhostsLocally, int stepsPerIter, MPILevelData* mpiData)
    {
        double res = 0.0;
        double*** vraw = v.getData();

        // TODO adjust for mpi, need global m, not local m
        // double h2 = 1.0 / ((double)(v.getM() * v.getM()));
        double dinv = h2 / 6.0;
        if (stencilType == MGCL_LAPLACE_19POINT)
            dinv = (6.0 * h2) / 24.0;
        else if (stencilType == MGCL_LAPLACE_27POINT)
            dinv = (26.0 * h2) / 88.0;

        // decrease stepsPerIter if it's less than maxIter
        if (maxiter < stepsPerIter)
            stepsPerIter = maxiter;

        // Ghosts only need to be updated in the periodic case, so set stepsPerIter = 1 for non-periodic.
        // TODO adjust for MPI
        if (!periodic)
            stepsPerIter = 1;

        // Check if amount of ghost cells is large enough
        if (util::seq::min3(v.getGhostsM(), v.getGhostsN(), v.getGhostsO()) < stepsPerIter)
        {
            error("#ghosts of v must be >= stepsPerIter!");
        }

        if (util::seq::min3(r.getGhostsM(), r.getGhostsN(), r.getGhostsO()) < stepsPerIter - 1)
        {
            error("#ghosts of r must be >= stepsPerIter - 1!");
        }

        if (util::seq::min3(f.getGhostsM(), f.getGhostsN(), f.getGhostsO()) < stepsPerIter - 1)
        {
            error("#ghosts of f must be >= stepsPerIter - 1!");
        }

        // check that stencilValues is not null if stencil type is varying
        if (stencilType == MGCL_VARYING && stencilValues == nullptr)
            error("stencilType is varying but stencilValues is null!");

        for (int iter = 0; iter < maxiter; iter += stepsPerIter)
        {
            // update ghost cells for periodic boundary condition
            if (periodic)
                MultigridEngine::updateGhostsSeq(v, mpiData, periodic, updateGhostsLocally);
            // TODO else update only neighboring processes if using mpi

            // if stepsPerIter > 1, multiple iterations can be done without updating ghosts in-between
            for (int innerIter = 0; innerIter < stepsPerIter && iter + innerIter < maxiter; innerIter++)
            {
                // damped/weighted iteration formula: u_(m+1) = u_(m) + omega * D^-1 * r_(m)

                int off = (stepsPerIter - innerIter) - 1;
                int istart_v = v.getGhostsM() - off;
                int jstart_v = v.getGhostsN() - off;
                int kstart_v = v.getGhostsO() - off;
                int iend_v = v.getMgh() - istart_v;
                int jend_v = v.getNgh() - jstart_v;
                int kend_v = v.getOgh() - kstart_v;
                int istart_r = r.getGhostsM() - off;
                int jstart_r = r.getGhostsN() - off;
                int kstart_r = r.getGhostsO() - off;
                int istart_sv = stencilValues ? stencilValues->getGhostsM() - off : 0;
                int jstart_sv = stencilValues ? stencilValues->getGhostsN() - off : 0;
                int kstart_sv = stencilValues ? stencilValues->getGhostsO() - off : 0;

                // r = f - A*v
                res = residualSeq(f, v, r, resnorm, stencilType, stencilFactor, stencilValues, fixedStencil,
                                  false, periodic,
                                  updateGhostsLocally, -off, -off, -off, mpiData);

                if (stencilType == MGCL_LAPLACE_7POINT || stencilType == MGCL_LAPLACE_19POINT || stencilType == MGCL_LAPLACE_27POINT)
                {
                    for (int iv = istart_v, ir = istart_r; iv < iend_v; iv++, ir++)
                        for (int jv = jstart_v, jr = jstart_r; jv < jend_v; jv++, jr++)
                            for (int kv = kstart_v, kr = kstart_r; kv < kend_v; kv++, kr++)
                            {
                                // if (i == 1 && j == 1 && k == 1)
                                //     printf("v[%d][%d][%d] = %f, r[%d][%d][%d] = %f, omega = %f\n", i,j,k, vraw[i][j][k],
                                //     i,j,k,r[i][j][k], omega);
                                vraw[iv][jv][kv] = vraw[iv][jv][kv] + omega * dinv * r[ir][jr][kr];
                            }
                }
                else
                {
                    // printf("seq x = %d, omega = %e, res = %e, v_out = %e, sv_self = %e\n", 1, omega, r[1][2][5], vraw[1][2][5], stencilValues[2][3][6][1][1][1]);
                    // print_7point(v_in, index, ioff, joff, koff);
                    // print27point(v, 1, 2, 5);
                    // print27point_sv(v, 1, 2, 5, stencilValues, 2, 3, 6);

                    for (int iv = istart_v, ir = istart_r, isv = istart_sv; iv < iend_v; iv++, ir++, isv++)
                        for (int jv = jstart_v, jr = jstart_r, jsv = jstart_sv; jv < jend_v; jv++, jr++, jsv++)
                            for (int kv = kstart_v, kr = kstart_r, ksv = kstart_sv; kv < kend_v; kv++, kr++, ksv++)
                            {
                                // if (i == 1 && j == 1 && k == 1)
                                //     printf("v[%d][%d][%d] = %f, r[%d][%d][%d] = %f, omega = %f\n", i,j,k, vraw[i][j][k],
                                //     i,j,k,r[i][j][k], omega);
                                vraw[iv][jv][kv] = vraw[iv][jv][kv] + omega * (1.0 / (*stencilValues)[1][1][1][isv][jsv][ksv]) * r[ir][jr][kr];

                                // if (j == 2 && k == 5 && i == 1)
                                // {
                                //     // printf("seq omega * (1.0 / sv_self) * res = %e\n", omega * (1.0 / stencilValues[isv][jsv][ksv][1][1][1]) * r[i][j][k]);
                                //     // printf("seq omega = %e, (1.0 / sv_self) = %e, res = %e\n", omega, (1.0 / stencilValues[isv][jsv][ksv][1][1][1]), r[i][j][k]);
                                //     // printf("seq res = %e, f = %e\n", r[i][j][k], f[i][j][k]);
                                //     // printf("seq x = %d, omega = %e, res = %e, v_out = %e, sv_self = %e\n", i, omega, r[i][j][k], vraw[i][j][k], stencilValues[isv][jsv][ksv][1][1][1]);
                                //     // // print_7point(v_in, index, ioff, joff, koff);
                                //     // print27point(v, i, j, k);
                                // }
                            }
                }
            }
        }

        if (periodic)
            MultigridEngine::updateGhostsSeq(v, mpiData, periodic, updateGhostsLocally);

        if (returnResidualNorm)
            res = residualSeq(f, v, r, resnorm, stencilType, stencilFactor, stencilValues, fixedStencil,
                              returnResidualNorm, periodic,
                              updateGhostsLocally, 0, 0, 0, mpiData);

        return res;
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
    double MultigridEngine::jacobi(Problem& problem, Level& level, int maxiter, bool return_residual, int stepsPerIter)
    {
        int err;
        int mgh = level.mgh;
        int ngh = level.ngh;
        int ogh = level.ogh;
        int store_res = 0;
        double res = -1;
        int idx_start = 0;

        if (problem.use_local_memory)
        {
            res = jacobiLocalMem(problem, level, maxiter, return_residual);
            if (res != -2)
                return res;

            mgcl_debug("jacobiLocalMem apparently failed. Running global mem version instead.\n");
        }

        // decrease stepsPerIter if it's less than maxIter
        if (maxiter < stepsPerIter)
            stepsPerIter = maxiter;

        // Ghosts only need to be updated in the periodic case, so set stepsPerIter = 1 for non-periodic.
        // TODO adjust for MPI
        if (!problem.isPeriodic())
            stepsPerIter = 1;

        // Check if amount of ghost cells is large enough
        if (problem.ghosts < stepsPerIter)
        {
            error("#ghosts must be >= stepsPerIter!");
        }

        cl_event ev;

        double h2 = 1.0 / static_cast<double>((problem.getMGlobal() >> level.num) * (problem.getMGlobal() >> level.num));
        double dinv = h2 / 6.0;
        double h2inv = level.stencilFactor; // divisor of the stencil, inverted to use * instead of / in kernel
        // TODO refactor stencilFactor

        // Create the compute kernel from the program
        const char* kernelName;
        if (problem.stencilType == MGCL_LAPLACE_7POINT)
            kernelName = "jacobi_iter_7point";
        else if (problem.stencilType == MGCL_LAPLACE_19POINT)
        {
            kernelName = "jacobi_iter_19point";
            dinv = (6.0 * h2) / 24.0;
        }
        else if (problem.stencilType == MGCL_LAPLACE_27POINT)
        {
            kernelName = "jacobi_iter_27point";
            dinv = (26.0 * h2) / 88.0;
        }
        else if (problem.stencilType == MGCL_VARYING)
        {
            kernelName = "jacobi_iter_27point_varying_stencil_1d";
        }

        cl_kernel kernel = clCreateKernel(problem.openCLHelper.getProgram(), kernelName, &err);
        mgclCheckError(err, "Creating kernel");

        cl_mem dVIn = level.getDVIn().getBuffer();
        cl_mem dVOut = level.getDVOut().getBuffer();
        cl_mem dF = level.getDF().getBuffer();
        cl_mem dR = level.getDR().getBuffer();

        // assign kernel arguments
        int pos = 0;

        if (problem.stencilType == MGCL_VARYING)
        {
            auto svbuf = level.stencilValuesGpu->getBuf();
            int svgh = level.stencilValuesGpu->getGh();
            int svmgh = level.stencilValuesGpu->getMgh();
            int svngh = level.stencilValuesGpu->getNgh();
            int svogh = level.stencilValuesGpu->getOgh();
            int svGridSize = svmgh * svngh * svogh;
            err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &dVIn);
            err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dVOut);
            err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dF);
            err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dR);
            err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &svbuf);
            err |= clSetKernelArg(kernel, ++pos, sizeof(double), &problem.omega);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &mgh);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ngh);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ogh);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svmgh);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svngh);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svogh);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &problem.ghosts);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svgh);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svGridSize);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &idx_start);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &store_res);
        }
        else
        {
            err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &dVIn);
            err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dVOut);
            err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dF);
            err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dR);
            err |= clSetKernelArg(kernel, ++pos, sizeof(double), &h2inv);
            err |= clSetKernelArg(kernel, ++pos, sizeof(double), &dinv);
            err |= clSetKernelArg(kernel, ++pos, sizeof(double), &problem.omega);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &mgh);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ngh);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ogh);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &problem.ghosts);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &idx_start);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &store_res);
        }
        mgclCheckError(err, "Setting kernel arguments");

        // One work-item per cell (including ghost cells).
        size_t global[2] = {static_cast<size_t>(mgh * ngh * ogh), static_cast<size_t>(0)};
        const auto& c = conf::getWorkGroupSizeForKernelAndWiCount(problem.getKernelConfig(), kernelName, 1);
        size_t local[2] = {c[0], c[1]};

        // kernels that use constant Laplace stencils are 2d and need different global and local sizes
        if (problem.stencilType != MGCL_VARYING)
        {
            global[0] = static_cast<size_t>(ngh);
            global[1] = static_cast<size_t>(ogh);
            // local[0] = static_cast<size_t>(1);
            // local[1] = static_cast<size_t>(64);
        }

        // Pad global sizes to fit to local sizes
        for (int i = 0; i < (problem.stencilType == MGCL_VARYING ? 1 : 2); i++)
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
                err = MultigridEngine::updateGhosts(problem, level.getDVOut(),
                                                    level.getMpiDataPtr(), level.isCalculatedLocally());
                mgclCheckError(err, "Updating ghosts");
            }
            else
            {
                err = MultigridEngine::updateGhosts(problem, level.getDVIn(),
                                                    level.getMpiDataPtr(), level.isCalculatedLocally());
                mgclCheckError(err, "Updating ghosts");
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
                    mgclCheckError(err, "Setting kernel arguments");
                }
                else
                {
                    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &dVIn);
                    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &dVOut);
                    mgclCheckError(err, "Setting kernel arguments");
                }

                // set flag to store residual in last iteration
                if (globalIter == maxiter - 1)
                {
                    store_res = 1;
                    err = clSetKernelArg(kernel, pos, sizeof(int), &store_res);
                    mgclCheckError(err, "Setting kernel arguments");
                }

                // recalculate and set idx_start
                idx_start = problem.ghosts - ((stepsPerIter - innerIter) - 1);
                err = clSetKernelArg(kernel, pos - 1, sizeof(int), &idx_start);
                mgclCheckError(err, "Setting kernel arguments");

                if (problem.stencilType != MGCL_VARYING)
                    err = clEnqueueNDRangeKernel(problem.getOpenCLHelper().getCommands(), kernel, 2, NULL, global, local, 0, NULL, &ev);
                else
                    err = clEnqueueNDRangeKernel(problem.getOpenCLHelper().getCommands(), kernel, 1, NULL, global, local, 0, NULL, &ev);
                mgclCheckError(err, "Enqueueing kernel");

                if (problem.isProfilingEnabled())
                {
                    problem.getProfilingData()->addMeasurement(problem.getCommands(), ev, kernelName,
                                                               {global[0], global[1], 0},
                                                               {local[0], local[1], 1});
                }
                mgclCheckError(clReleaseEvent(ev), "clReleaseEvent");
            }
        }

        if (store_res)
        {
            // TODO check for mpi
            err = MultigridEngine::updateGhosts(problem, level.getDR(), level.getMpiDataPtr(),
                                                level.isCalculatedLocally());
            mgclCheckError(err, "Updating ghosts of dR");
        }

        // copy result into dVIn if needed
        if (maxiter % 2 == 1)
            level.getDVOut().copyTo(problem.getOpenCLHelper().getCommands(), level.getDVIn());

        // Update ghosts of dVIn
        err = MultigridEngine::updateGhosts(problem, level.getDVIn(),
                                            level.getMpiDataPtr(), level.isCalculatedLocally());
        mgclCheckError(err, "Updating ghosts");

        // calculate residual and its norm
        if (return_residual)
        {
            // update residual to use current approximation v
            res = MultigridEngine::residual(problem, level, true);
        }

        clReleaseKernel(kernel);

        return res;
    }

    /* Runs jacobi method using OpenCL.
     * Doesn't creates ocl buffers and doesn't copy data from host to device and vice versa
     * v, f and r must be of size [m][n][o] for periodic boundary condition.
     * m, n and o must be the dimensions of grid + 2*ghosts
     * If return_residual is true, the residual's 2-norm or inf-norm will be read back from device and returned, else -1.
     * It's not really performant to do so because we have to wait for all kernels to complete and reading a buffer to host
     * is slow. Runs multiple iterations per kernel call using local memory if enough is available Only jacobi_wg_size_x is
     * used for now. If there is not enough local memory available the kernel will not be called and -2 is returned. */
    double MultigridEngine::jacobiLocalMem(Problem& problem, Level& level, int maxiter, int return_residual)
    {
        int err;
        int mgh = level.mgh;
        int ngh = level.ngh;
        int ogh = level.ogh;
        int store_res = 0;
        double res = -1;
        int ipk = problem.jacobi_iterations_per_kernel;
        int wg_size = problem.jacobi_wg_size_x;

        if (level.n < wg_size)
            wg_size = level.n;
        if (ogh < ngh)
            wg_size = level.o;
        mgcl_debug("Using wg_size = %d (problem.jacobi_wg_size_x = %d)\n", wg_size, problem.jacobi_wg_size_x);

        if (problem.ghosts < problem.jacobi_iterations_per_kernel)
        {
            ipk = problem.ghosts;
            mgcl_debug("Reducing iterations_per_kernel, ghosts = %d < %d = ipk\n", problem.ghosts,
                       problem.jacobi_iterations_per_kernel);
        }

        if (maxiter < problem.jacobi_iterations_per_kernel)
        {
            ipk = maxiter;
            mgcl_debug("Reducing iterations_per_kernel, maxiter = %d < %d = ipk\n", maxiter,
                       problem.jacobi_iterations_per_kernel);
        }

        // check if there is enough local memory available on device for given problem.ghosts = iterations per kernel call
        // TODO do in mgcl_init?
        cl_ulong available_local_mem;
        err = clGetDeviceInfo(problem.openCLHelper.getDeviceId(), CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &available_local_mem, 0);
        mgclCheckError(err, "Querying local memory size info");
        mgcl_debug("Available local memory on device: %ld Bytes\n", available_local_mem);

        // size of shared memory size for one work-group
        size_t locmem_size_wg = 3 * ipk * (wg_size + 2 * ipk) * (wg_size + 2 * ipk) * sizeof(double);

        // halve wg size until local memory fits
        while (available_local_mem < locmem_size_wg)
        {
            if (wg_size == 1)
            {
                printf("Not enough local memory available to start Jacobi kernel using local memory. Please set "
                       "problem.use_local_memory to false. Aborting.\n");
                return -2;
            }

            wg_size >>= 1;
            locmem_size_wg = 3 * ipk * (wg_size + 2 * ipk) * (wg_size + 2 * ipk) * sizeof(double);
            mgcl_debug("reducing wg_size from %d to %d (now %d Bytes of local memory needed)\n", wg_size << 1, wg_size,
                       locmem_size_wg);
        }
        // TODO pad wg size?
        mgcl_debug("using %d Bytes of local memory\n", locmem_size_wg);

        // ghosted wg size (ghosted grid excluding outmost ghosted border)
        size_t wg_size_ghosted = wg_size + 2 * (ipk - 1);

        double h2 = (1.0 / (double)level.m) *
                    (1.0 / (double)level.m); // TODO minimum of m,n,o when not cube?
        double dinv = h2 / 6.0;
        double h2inv = level.stencilFactor; // divisor of the stencil, inverted to use * instead of / in kernel

        // Create the compute kernel from the program
        const char* kernel_name;
        if (problem.stencilType == MGCL_LAPLACE_7POINT)
            kernel_name = "jacobi_stream_shmem_7point";
        else if (problem.stencilType == MGCL_LAPLACE_19POINT)
        {
            kernel_name = "jacobi_stream_shmem_19point";
            dinv = (6.0 * h2) / 24.0;
        }
        else if (problem.stencilType == MGCL_LAPLACE_27POINT)
        {
            kernel_name = "jacobi_stream_shmem_27point";
            dinv = (26.0 * h2) / 88.0;
        }
        cl_kernel kernel = clCreateKernel(problem.openCLHelper.getProgram(), kernel_name, &err);
        mgclCheckError(err, "Creating kernel");

        cl_mem dVIn = level.getDVIn().getBuffer();
        cl_mem dVOut = level.getDVOut().getBuffer();
        cl_mem dF = level.getDF().getBuffer();
        cl_mem dR = level.getDR().getBuffer();

        // assign kernel arguments
        int pos = 0;
        err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &dVIn);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dVOut);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dF);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dR);
        err |= clSetKernelArg(kernel, ++pos, locmem_size_wg, NULL);
        err |= clSetKernelArg(kernel, ++pos, sizeof(double), &h2inv);
        err |= clSetKernelArg(kernel, ++pos, sizeof(double), &dinv);
        err |= clSetKernelArg(kernel, ++pos, sizeof(double), &problem.omega);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &mgh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ngh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ogh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &problem.ghosts);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ipk);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &store_res);
        mgclCheckError(err, "Setting kernel arguments");

        // initial kernel dimensions
        size_t global_n = ceil((double)ngh / (double)wg_size) * wg_size_ghosted;
        size_t global_o = ceil((double)ogh / (double)wg_size) * wg_size_ghosted;
        size_t global[2] = {global_n, global_o};
        size_t local[2] = {wg_size_ghosted, wg_size_ghosted};
        mgcl_debug("Running Jacobi kernel with %ld,%ld work-items and %ld,%ld work group size\n", global[0], global[1],
                   local[0], local[1]);

        if (ipk == maxiter)
        {
            // set flag to store residual of last iteration
            store_res = 1;
            err = clSetKernelArg(kernel, pos, sizeof(int), &store_res);
            mgclCheckError(err, "Setting kernel arguments");

            err = MultigridEngine::updateGhosts(problem, level.getDVIn(), level.getMpiDataPtr(),
                                                level.isCalculatedLocally());
            mgclCheckError(err, "Updating ghosts");

            err = clEnqueueNDRangeKernel(problem.getOpenCLHelper().getCommands(), kernel, 2, NULL, global, local, 0, NULL, NULL);
            mgclCheckError(err, "Enqueueing kernel");

            // swap pointers so result is in dVIn
            CuboidGpu::swap(level.getDVIn(), level.getDVOut());
            dVIn = level.getDVIn().getBuffer();
            dVOut = level.getDVOut().getBuffer();
        }
        else
        {
            // start multiple kernels with lesser iterations
            int call_count = ((double)maxiter) / ((double)ipk);
            int iter_rest = maxiter % ipk;
            mgcl_debug("starting kernels %d time(s) for %d iterations and once for %d iteration(s)\n", call_count, ipk,
                       iter_rest);

            // call kernel multiple times with given ipk
            for (int k = 0; k < call_count; k++)
            {
                if (k == call_count - 1 && !iter_rest)
                {
                    // set flag to store residual of last iteration
                    store_res = 1;
                    err = clSetKernelArg(kernel, pos, sizeof(int), &store_res);
                    mgclCheckError(err, "Setting kernel argument store_residual");
                }

                err = clEnqueueNDRangeKernel(problem.getOpenCLHelper().getCommands(), kernel, 2, NULL, global, local, 0, NULL, NULL);
                mgclCheckError(err, "Enqueueing kernel");

                err = MultigridEngine::updateGhosts(problem, level.getDVOut(), level.getMpiDataPtr(),
                                                    level.isCalculatedLocally());
                mgclCheckError(err, "Updating ghosts");

                // swap pointers so result is in dVIn
                CuboidGpu::swap(level.getDVIn(), level.getDVOut());
                dVIn = level.getDVIn().getBuffer();
                dVOut = level.getDVOut().getBuffer();

                err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &dVIn);
                err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &dVOut);
            }

            // call once for remaining iterations
            if (iter_rest)
            {
                // set flag to store residual of last iteration
                store_res = 1;
                err = clSetKernelArg(kernel, pos, sizeof(int), &store_res);
                mgclCheckError(err, "Setting kernel argument store_residual");

                // set ipk argument for this kernel call
                err = clSetKernelArg(kernel, pos - 1, sizeof(int), &iter_rest);
                mgclCheckError(err, "Setting kernel arguments iter_rest");

                err = clEnqueueNDRangeKernel(problem.getOpenCLHelper().getCommands(), kernel, 2, NULL, global, local, 0, NULL, NULL);
                mgclCheckError(err, "Enqueueing kernel");

                err = MultigridEngine::updateGhosts(problem, level.getDVOut(), level.getMpiDataPtr(),
                                                    level.isCalculatedLocally());
                mgclCheckError(err, "Updating ghosts");

                // swap pointers so result is in dVIn
                CuboidGpu::swap(level.getDVIn(), level.getDVOut());
                dVIn = level.getDVIn().getBuffer();
                dVOut = level.getDVOut().getBuffer();
            }
        }
        // result is in dVIn now since pointers were swapped at the end of the loops above

        err = MultigridEngine::updateGhosts(problem, level.getDVIn(), level.getMpiDataPtr(),
                                            level.isCalculatedLocally());
        mgclCheckError(err, "Updating ghosts of v_in");

        // calculate residual's 2-norm. Square elements on device and sum up on host
        if (return_residual)
        {
            // update residual to use current approximation v
            res = MultigridEngine::residual(problem, level, true);
        }

        clReleaseKernel(kernel); // TODO maybe clFinish before release?

        return res;
    }

    /* Calculates the residual using OpenCL.
     * Doesn't creates ocl buffers and doesn't copy data from host to device and vice versa
     * v, f and r must be of size [m][n][o] for periodic boundary condition.
     * m, n and o must be the dimensions of ghosted grid.
     * If return_residual is true, the residual's 2-norm or inf-norm will be read back from device and returned, else -1.
     * It's not really performant to do so because we have to wait for all kernels to complete and
     * reading a buffer to host is slow.
     * v needs to have updated ghost cells if the problem is periodic!
     * moff, noff and ooff can be used to change the size of the grid that the residual shall be calculated for.
     *   Per default only real cells are considered (moff = 0), but with e.g. moff = -1, the first ghost cell border is
     *   considered, too. Analogously, with moff = 1 the outermost set of real cells is ignored. The calculation
     *   of the boundaries is e.g. istart = v.ghosts_m + moff.
     */
    double MultigridEngine::residual(Problem& problem, Level& level, bool return_residual,
                                     int moff, int noff, int ooff)
    {
        int err;
        int mgh = level.mgh;
        int ngh = level.ngh;
        int ogh = level.ogh;
        double res = 0.0;

        double h2 = 1.0 / static_cast<double>((problem.getMGlobal() >> level.num) * (problem.getMGlobal() >> level.num));
        double h2inv = 1.0 / h2; // divisor of the stencil, inverted to use * instead of / in kernel

        // check if off is too small (i.e. start < 0)
        // TODO refactor to use GPUCuboid and check against v.getGhosts
        if (moff <= -problem.ghosts || noff <= -problem.ghosts || ooff <= -problem.ghosts)
            error("moff, noff and ooff must not be <= -ghosts");

        // check if off is too large (i.e. start > end)
        if (moff * 2 >= level.m || noff * 2 >= level.n || ooff * 2 >= level.o)
            error("2*moff, 2*noff and 2*ooff must not be >= m, n or o");

        // Create the compute kernel from the program
        const char* kernelName;
        if (problem.stencilType == MGCL_LAPLACE_7POINT)
            kernelName = "residual_7point";
        else if (problem.stencilType == MGCL_LAPLACE_19POINT)
        {
            kernelName = "residual_19point";
            h2inv = 1.0 / (6.0 * h2);
        }
        else if (problem.stencilType == MGCL_LAPLACE_27POINT)
        {
            kernelName = "residual_27point";
            h2inv = 1.0 / (26.0 * h2);
        }
        else if (problem.stencilType == MGCL_VARYING)
        {
            kernelName = "residual_27point_varying_stencil";
        }

        cl_event ev;

        // Create the compute kernel from the program
        cl_kernel kernel = clCreateKernel(problem.openCLHelper.getProgram(), kernelName, &err);
        mgclCheckError(err, "Creating kernel");

        cl_mem dVIn = level.getDVIn().getBuffer();
        cl_mem dF = level.getDF().getBuffer();
        cl_mem dR = level.getDR().getBuffer();

        // assign kernel arguments
        int pos = 0;
        if (problem.stencilType == MGCL_VARYING)
        {
            auto svbuf = level.stencilValuesGpu->getBuf();
            int svgh = level.stencilValuesGpu->getGh();
            int svmgh = level.stencilValuesGpu->getMgh();
            int svngh = level.stencilValuesGpu->getNgh();
            int svogh = level.stencilValuesGpu->getOgh();
            int svGridSize = svmgh * svngh * svogh;

            err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &dVIn);
            err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dF);
            err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dR);
            err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &svbuf);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &mgh);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ngh);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ogh);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svmgh);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svngh);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svogh);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &problem.ghosts);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svgh);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svGridSize);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &moff);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &noff);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ooff);
        }
        else
        {
            err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &dVIn);
            err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dF);
            err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dR);
            err |= clSetKernelArg(kernel, ++pos, sizeof(double), &h2inv);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &mgh);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ngh);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ogh);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &problem.ghosts);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &moff);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &noff);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ooff);
        }

        mgclCheckError(err, "Setting residual kernel arguments");

        // one work-item per cell (including ghost cells). Pad global sizes to fit to local sizes
        size_t global = mgh * ngh * ogh;
        const auto& c = conf::getWorkGroupSizeForKernelAndWiCount(problem.getKernelConfig(), kernelName, global);
        size_t local = c[0];

        if (global % local != 0)
            global += local - (global % local);

        err = clEnqueueNDRangeKernel(problem.getOpenCLHelper().getCommands(), kernel, 1, NULL, &global, &local, 0, NULL, &ev);
        mgclCheckError(err, "Enqueueing residual kernel");

        if (problem.isProfilingEnabled())
        {
            problem.getProfilingData()->addMeasurement(problem.getCommands(), ev, kernelName,
                                                       {global, 0, 0},
                                                       {local, 1, 1});
        }
        mgclCheckError(clReleaseEvent(ev), "clReleaseEvent");

        if (problem.isPeriodic())
        {
            err = MultigridEngine::updateGhosts(problem, level.getDR(), level.getMpiDataPtr(),
                                                level.isCalculatedLocally());
            mgclCheckError(err, "Updating ghosts of r");
        }

        // calculate residual's 2-norm. Square elements on device and sum up on host
        if (return_residual)
        {
            if (problem.residual_norm == MGCL_L2)
            {
                // calculate 2-Norm
                auto& dRsquares = level.getDRsq();
                dRsquares.fill(problem.getProgram(), problem.getCommands(), 0.0, false, &problem.getKernelConfig(), problem.getProfilingData()); // reset to zero

                // Create the compute kernel from the program
                const char* kernelName = "residual_squared";
                cl_kernel kernel_square = clCreateKernel(problem.openCLHelper.getProgram(), kernelName, &err);
                mgclCheckError(err, "Creating residual squared kernel");

                pos = 0;
                err = clSetKernelArg(kernel_square, pos, sizeof(cl_mem), &dR);
                err |= clSetKernelArg(kernel_square, ++pos, sizeof(cl_mem), &dRsquares);
                err |= clSetKernelArg(kernel_square, ++pos, sizeof(int), &level.m);
                err |= clSetKernelArg(kernel_square, ++pos, sizeof(int), &level.n);
                err |= clSetKernelArg(kernel_square, ++pos, sizeof(int), &level.o);
                err |= clSetKernelArg(kernel_square, ++pos, sizeof(int), &problem.ghosts);
                mgclCheckError(err, "Setting residual squared kernel arguments");

                size_t global = level.getM() * level.getN() * level.getO();
                const auto& c_sq = conf::getWorkGroupSizeForKernelAndWiCount(problem.getKernelConfig(), kernelName, global);
                size_t local_sq = c_sq[0];

                if (global % local_sq != 0)
                    global += local_sq - (global % local_sq);

                cl_event ev;
                err = clEnqueueNDRangeKernel(problem.getOpenCLHelper().getCommands(), kernel_square, 1, NULL, &global, &local_sq, 0, NULL, &ev);
                mgclCheckError(err, "Enqueueing residual squared kernel");

                if (problem.isProfilingEnabled())
                {
                    problem.getProfilingData()->addMeasurement(problem.getCommands(), ev, kernelName,
                                                               {global, 0, 0},
                                                               {local_sq, 1, 1});
                }
                mgclCheckError(clReleaseEvent(ev), "clReleaseEvent");

                // sum up residual squares
                res = sqrt(util::sum(dRsquares, problem.getProgram(), problem.getCommands(), true, &problem.getKernelConfig(), problem.getProfilingData()));

                clReleaseKernel(kernel_square);
            }
            else
            {
                // calculate Infinity-Norm
                res = util::max_abs(level.getDR(), problem.getProgram(), problem.getCommands(), true, &problem.getKernelConfig(), problem.getProfilingData());
            }
        }

        clReleaseKernel(kernel); // TODO maybe clFinish before release?
        return res;
    }

    /* Calculates r = f - A*v using 7-point, 19-point or 27-point stencil of 3D laplacian or a varying stencil.
     * m,n,o is the size of the real grid.
     * v needs to have updated ghost cells if the problem is periodic!
     * moff, noff and ooff can be used to change the size of the grid that the residual shall be calculated for.
     *   Per default only real cells are considered (moff = 0), but with e.g. moff = -1, the first ghost cell border is
     *   considered, too. Analogously, with moff = 1 the outermost set of real cells is ignored. The calculation
     *   of the boundaries is e.g. istart = v.ghosts_m + moff.
     */
    double MultigridEngine::residualSeq(Cuboid& f, Cuboid& v, Cuboid& r, MGCL_RESIDUAL_NORM resnorm,
                                        MGCL_STENCIL stencilType, double stencilFactor,
                                        VaryingStencil* stencilValuesCuboid, FixedStencil* fixedStencil,
                                        bool returnResidualNorm,
                                        bool periodic, bool updateGhostsLocally, int moff, int noff, int ooff, MPILevelData* mpiData)
    {
        double res = 0.0;
        double stencilsum = 0;
        double****** stencilValues;
        double*** vraw = v.getData();
        double*** fsRaw;

        // check if off is too small (i.e. start < 0)
        if (moff <= -v.getGhostsM() || noff <= -v.getGhostsN() || ooff <= -v.getGhostsO())
            error("moff, noff and ooff must not be <= -ghosts");

        // check if off is too large (i.e. start > end)
        if (moff * 2 >= v.getM() || noff * 2 >= v.getN() || ooff * 2 >= v.getO())
            error("2*moff, 2*noff and 2*ooff must not be >= m, n or o");

        // check that stencilValues is not null if stencil type is varying
        if (stencilType == MGCL_VARYING && stencilValuesCuboid == nullptr)
            error("stencilType is varying but stencilValues is null!");

        if (stencilType == MGCL_VARYING)
            stencilValues = stencilValuesCuboid->getData();

        if (stencilType == MGCL_FIXED)
        {
            fsRaw = fixedStencil->getData(); // TODO
        }

        int istart_v = v.getGhostsM() + moff;
        int jstart_v = v.getGhostsN() + noff;
        int kstart_v = v.getGhostsO() + ooff;
        int iend_v = v.getMgh() - v.getGhostsM() - moff;
        int jend_v = v.getNgh() - v.getGhostsN() - noff;
        int kend_v = v.getOgh() - v.getGhostsO() - ooff;
        int istart_r = r.getGhostsM() + moff;
        int jstart_r = r.getGhostsN() + noff;
        int kstart_r = r.getGhostsO() + ooff;
        int istart_f = f.getGhostsM() + moff;
        int jstart_f = f.getGhostsN() + noff;
        int kstart_f = f.getGhostsO() + ooff;
        int istart_sv = stencilValuesCuboid ? stencilValuesCuboid->getGhostsM() + moff : 0;
        int jstart_sv = stencilValuesCuboid ? stencilValuesCuboid->getGhostsN() + noff : 0;
        int kstart_sv = stencilValuesCuboid ? stencilValuesCuboid->getGhostsO() + ooff : 0;

        for (int iv = istart_v, ir = istart_r, fi = istart_f, isv = istart_sv; iv < iend_v; iv++, ir++, fi++, isv++)
            for (int jv = jstart_v, jr = jstart_r, fj = jstart_f, jsv = jstart_sv; jv < jend_v; jv++, jr++, fj++, jsv++)
                for (int kv = kstart_v, kr = kstart_r, fk = kstart_f, ksv = kstart_sv; kv < kend_v; kv++, kr++, fk++, ksv++)
                {
                    // A*v
                    if (stencilType == MGCL_LAPLACE_7POINT)
                    {
                        // clang-format off
                        stencilsum = (6.0 * vraw[iv][jv][kv]
                            - vraw[iv][jv][kv - 1] - vraw[iv][jv][kv + 1]
                            - vraw[iv][jv - 1][kv] - vraw[iv][jv + 1][kv]
                            - vraw[iv - 1][jv][kv] - vraw[iv + 1][jv][kv]
                            ) * stencilFactor;
                        // clang-format on
                    }
                    else if (stencilType == MGCL_LAPLACE_19POINT)
                    {
                        // clang-format off
                        stencilsum = (24.0 * vraw[iv][jv][kv]
                                - 2.0 * vraw[iv][jv][kv - 1] - 2.0 * vraw[iv][jv][kv + 1]
                                - 2.0 * vraw[iv][jv - 1][kv] - 2.0 * vraw[iv][jv + 1][kv]
                                - 2.0 * vraw[iv - 1][jv][kv] - 2.0 * vraw[iv + 1][jv][kv]
                                
                                - vraw[iv][jv - 1][kv - 1] - vraw[iv][jv - 1][kv + 1]
                                - vraw[iv][jv + 1][kv - 1] - vraw[iv][jv + 1][kv + 1]
                                - vraw[iv - 1][jv][kv - 1] - vraw[iv - 1][jv][kv + 1]
                                - vraw[iv + 1][jv][kv - 1] - vraw[iv + 1][jv][kv + 1]
                                - vraw[iv - 1][jv - 1][kv] - vraw[iv - 1][jv + 1][kv]
                                - vraw[iv + 1][jv - 1][kv] - vraw[iv + 1][jv + 1][kv]
                                ) * stencilFactor;
                        // clang-format on
                    }
                    else if (stencilType == MGCL_LAPLACE_27POINT)
                    {
                        // clang-format off
                        stencilsum = (88.0 * vraw[iv][jv][kv]
                                - 6.0 * vraw[iv][jv][kv - 1] - 6.0 * vraw[iv][jv][kv + 1]
                                - 6.0 * vraw[iv][jv - 1][kv] - 6.0 * vraw[iv][jv + 1][kv]
                                - 6.0 * vraw[iv - 1][jv][kv] - 6.0 * vraw[iv + 1][jv][kv]

                                - 3.0 * vraw[iv][jv - 1][kv - 1] - 3.0 * vraw[iv][jv - 1][kv + 1]
                                - 3.0 * vraw[iv][jv + 1][kv - 1] - 3.0 * vraw[iv][jv + 1][kv + 1]
                                - 3.0 * vraw[iv - 1][jv][kv - 1] - 3.0 * vraw[iv - 1][jv][kv + 1]
                                - 3.0 * vraw[iv + 1][jv][kv - 1] - 3.0 * vraw[iv + 1][jv][kv + 1]
                                - 3.0 * vraw[iv - 1][jv - 1][kv] - 3.0 * vraw[iv - 1][jv + 1][kv]
                                - 3.0 * vraw[iv + 1][jv - 1][kv] - 3.0 * vraw[iv + 1][jv + 1][kv]

                                - 2.0 * vraw[iv - 1][jv - 1][kv - 1] - 2.0 * vraw[iv - 1][jv - 1][kv + 1]
                                - 2.0 * vraw[iv - 1][jv + 1][kv - 1] - 2.0 * vraw[iv - 1][jv + 1][kv + 1]
                                - 2.0 * vraw[iv + 1][jv - 1][kv - 1] - 2.0 * vraw[iv + 1][jv - 1][kv + 1]
                                - 2.0 * vraw[iv + 1][jv + 1][kv - 1] - 2.0 * vraw[iv + 1][jv + 1][kv + 1]
                                ) * stencilFactor;
                        // clang-format on
                    }
                    else if (stencilType == MGCL_FIXED)
                    {
                        // clang-format off
                        stencilsum = (fsRaw[1][1][1] * vraw[iv][jv][kv]
                                - fsRaw[1][1][0] * vraw[iv][jv][kv - 1] - fsRaw[1][1][2] * vraw[iv][jv][kv + 1]
                                - fsRaw[1][0][1] * vraw[iv][jv - 1][kv] - fsRaw[1][2][1] * vraw[iv][jv + 1][kv]
                                - fsRaw[0][1][1] * vraw[iv - 1][jv][kv] - fsRaw[2][1][1] * vraw[iv + 1][jv][kv]

                                - fsRaw[1][0][0] * vraw[iv][jv - 1][kv - 1] - fsRaw[1][0][2] * vraw[iv][jv - 1][kv + 1]
                                - fsRaw[1][2][0] * vraw[iv][jv + 1][kv - 1] - fsRaw[1][2][2] * vraw[iv][jv + 1][kv + 1]
                                - fsRaw[0][1][0] * vraw[iv - 1][jv][kv - 1] - fsRaw[0][1][2] * vraw[iv - 1][jv][kv + 1]
                                - fsRaw[2][1][0] * vraw[iv + 1][jv][kv - 1] - fsRaw[2][1][2] * vraw[iv + 1][jv][kv + 1]
                                - fsRaw[0][0][1] * vraw[iv - 1][jv - 1][kv] - fsRaw[0][2][1] * vraw[iv - 1][jv + 1][kv]
                                - fsRaw[2][0][1] * vraw[iv + 1][jv - 1][kv] - fsRaw[2][2][1] * vraw[iv + 1][jv + 1][kv]

                                - fsRaw[0][0][0] * vraw[iv - 1][jv - 1][kv - 1] - fsRaw[0][0][2] * vraw[iv - 1][jv - 1][kv + 1]
                                - fsRaw[0][2][0] * vraw[iv - 1][jv + 1][kv - 1] - fsRaw[0][2][2] * vraw[iv - 1][jv + 1][kv + 1]
                                - fsRaw[2][0][0] * vraw[iv + 1][jv - 1][kv - 1] - fsRaw[2][0][2] * vraw[iv + 1][jv - 1][kv + 1]
                                - fsRaw[2][2][0] * vraw[iv + 1][jv + 1][kv - 1] - fsRaw[2][2][2] * vraw[iv + 1][jv + 1][kv + 1]
                                ) * stencilFactor;
                        // clang-format on
                    }
                    else if (stencilType == MGCL_VARYING)
                    {
                        // clang-format off
                        stencilsum = stencilValues[1][1][1][isv][jsv][ksv]  * vraw[iv][jv][kv]
                            + stencilValues[1][1][0][isv][jsv][ksv] * vraw[ iv ][ jv ][kv-1]
                            + stencilValues[1][1][2][isv][jsv][ksv] * vraw[ iv ][ jv ][kv+1]
                            + stencilValues[1][0][1][isv][jsv][ksv] * vraw[ iv ][jv-1][ kv ]
                            + stencilValues[1][2][1][isv][jsv][ksv] * vraw[ iv ][jv+1][ kv ]
                            + stencilValues[0][1][1][isv][jsv][ksv] * vraw[iv-1][ jv ][ kv ]
                            + stencilValues[2][1][1][isv][jsv][ksv] * vraw[iv+1][ jv ][ kv ]
                            
                            + stencilValues[1][0][0][isv][jsv][ksv] * vraw[ iv ][jv-1][kv-1]
                            + stencilValues[1][0][2][isv][jsv][ksv] * vraw[ iv ][jv-1][kv+1]
                            + stencilValues[1][2][0][isv][jsv][ksv] * vraw[ iv ][jv+1][kv-1]
                            + stencilValues[1][2][2][isv][jsv][ksv] * vraw[ iv ][jv+1][kv+1]
                            + stencilValues[0][1][0][isv][jsv][ksv] * vraw[iv-1][ jv ][kv-1]
                            + stencilValues[0][1][2][isv][jsv][ksv] * vraw[iv-1][ jv ][kv+1]
                            + stencilValues[2][1][0][isv][jsv][ksv] * vraw[iv+1][ jv ][kv-1]
                            + stencilValues[2][1][2][isv][jsv][ksv] * vraw[iv+1][ jv ][kv+1]
                            + stencilValues[0][0][1][isv][jsv][ksv] * vraw[iv-1][jv-1][ kv ]
                            + stencilValues[0][2][1][isv][jsv][ksv] * vraw[iv-1][jv+1][ kv ]
                            + stencilValues[2][0][1][isv][jsv][ksv] * vraw[iv+1][jv-1][ kv ]
                            + stencilValues[2][2][1][isv][jsv][ksv] * vraw[iv+1][jv+1][ kv ]
                            
                            + stencilValues[0][0][0][isv][jsv][ksv] * vraw[iv-1][jv-1][kv-1]
                            + stencilValues[0][0][2][isv][jsv][ksv] * vraw[iv-1][jv-1][kv+1]
                            + stencilValues[0][2][0][isv][jsv][ksv] * vraw[iv-1][jv+1][kv-1]
                            + stencilValues[0][2][2][isv][jsv][ksv] * vraw[iv-1][jv+1][kv+1]
                            + stencilValues[2][0][0][isv][jsv][ksv] * vraw[iv+1][jv-1][kv-1]
                            + stencilValues[2][0][2][isv][jsv][ksv] * vraw[iv+1][jv-1][kv+1]
                            + stencilValues[2][2][0][isv][jsv][ksv] * vraw[iv+1][jv+1][kv-1]
                            + stencilValues[2][2][2][isv][jsv][ksv] * vraw[iv+1][jv+1][kv+1];
                        // clang-format on

                        // if (j == 2 && k == 2 && i == 2)
                        // {
                        //     printf("seq stencilsum = %e\n", stencilsum);
                        //     print27point_sv(v, i, j, k, stencilValuesCuboid, isv, jsv, ksv);
                        // }
                    }

                    // r = f - A*v
                    r[ir][jr][kr] = f[fi][fj][fk] - stencilsum;

                    if (returnResidualNorm)
                    {
                        if (resnorm == MGCL_L2)
                            res += r[ir][jr][kr] * r[ir][jr][kr];
                        else if (fabs(r[ir][jr][kr]) > res)
                            res = fabs(r[ir][jr][kr]);
                    }
                }

        if (periodic)
            MultigridEngine::updateGhostsSeq(r, mpiData, periodic, updateGhostsLocally);

        return (returnResidualNorm && resnorm == MGCL_L2) ? sqrt(res) : res;
    }

    /* Prints components of 7-point laplacian stencil for debugging purposes */
    void MultigridEngine::print7point(Cuboid& v, int i, int j, int k)
    {
        printf("7point stencil at %d,%d,%d:\n", i, j, k);
        printf("v[self] = %e\n", v[i][j][k]);
        printf(" v[k-1] = %e\n", v[i][j][k - 1]);
        printf(" v[k+1] = %e\n", v[i][j][k + 1]);
        printf(" v[j-1] = %e\n", v[i][j - 1][k]);
        printf(" v[j+1] = %e\n", v[i][j + 1][k]);
        printf(" v[i-1] = %e\n", v[i - 1][j][k]);
        printf(" v[i+1] = %e\n", v[i + 1][j][k]);
    }

    void MultigridEngine::print19point(Cuboid& v, int i, int j, int k)
    {
        printf("19point stencil at %d,%d,%d:\n", i, j, k);
        printf("v[self] = %e\n", v[i][j][k]);
        printf(" v[ i ][ j ][k-1] = %e\n", v[i][j][k - 1]);
        printf(" v[ i ][ j ][k+1] = %e\n", v[i][j][k + 1]);
        printf(" v[ i ][j-1][ k ] = %e\n", v[i][j - 1][k]);
        printf(" v[ i ][j+1][ k ] = %e\n", v[i][j + 1][k]);
        printf(" v[i-1][ j ][ k ] = %e\n", v[i - 1][j][k]);
        printf(" v[i+1][ j ][ k ] = %e\n", v[i + 1][j][k]);
        printf(" v[ i ][j-1][k-1] = %e\n", v[i][j - 1][k - 1]);
        printf(" v[ i ][j-1][k+1] = %e\n", v[i][j - 1][k + 1]);
        printf(" v[ i ][j+1][k-1] = %e\n", v[i][j + 1][k - 1]);
        printf(" v[ i ][j+1][k+1] = %e\n", v[i][j + 1][k + 1]);
        printf(" v[i-1][ j ][k-1] = %e\n", v[i - 1][j][k - 1]);
        printf(" v[i-1][ j ][k+1] = %e\n", v[i - 1][j][k + 1]);
        printf(" v[i+1][ j ][k-1] = %e\n", v[i + 1][j][k - 1]);
        printf(" v[i+1][ j ][k+1] = %e\n", v[i + 1][j][k + 1]);
        printf(" v[i-1][j-1][ k ] = %e\n", v[i - 1][j - 1][k]);
        printf(" v[i-1][j+1][ k ] = %e\n", v[i - 1][j + 1][k]);
        printf(" v[i+1][j-1][ k ] = %e\n", v[i + 1][j - 1][k]);
        printf(" v[i+1][j+1][ k ] = %e\n", v[i + 1][j + 1][k]);
    }

    void MultigridEngine::print27point(Cuboid& v, int i, int j, int k)
    {
        printf("27point stencil at %d,%d,%d:\n", i, j, k);
        printf("v[self] = %e\n", v[i][j][k]);
        printf(" v[ i ][ j ][k-1] = %e\n", v[i][j][k - 1]);
        printf(" v[ i ][ j ][k+1] = %e\n", v[i][j][k + 1]);
        printf(" v[ i ][j-1][ k ] = %e\n", v[i][j - 1][k]);
        printf(" v[ i ][j+1][ k ] = %e\n", v[i][j + 1][k]);
        printf(" v[i-1][ j ][ k ] = %e\n", v[i - 1][j][k]);
        printf(" v[i+1][ j ][ k ] = %e\n", v[i + 1][j][k]);
        printf(" v[ i ][j-1][k-1] = %e\n", v[i][j - 1][k - 1]);
        printf(" v[ i ][j-1][k+1] = %e\n", v[i][j - 1][k + 1]);
        printf(" v[ i ][j+1][k-1] = %e\n", v[i][j + 1][k - 1]);
        printf(" v[ i ][j+1][k+1] = %e\n", v[i][j + 1][k + 1]);
        printf(" v[i-1][ j ][k-1] = %e\n", v[i - 1][j][k - 1]);
        printf(" v[i-1][ j ][k+1] = %e\n", v[i - 1][j][k + 1]);
        printf(" v[i+1][ j ][k-1] = %e\n", v[i + 1][j][k - 1]);
        printf(" v[i+1][ j ][k+1] = %e\n", v[i + 1][j][k + 1]);
        printf(" v[i-1][j-1][ k ] = %e\n", v[i - 1][j - 1][k]);
        printf(" v[i-1][j+1][ k ] = %e\n", v[i - 1][j + 1][k]);
        printf(" v[i+1][j-1][ k ] = %e\n", v[i + 1][j - 1][k]);
        printf(" v[i+1][j+1][ k ] = %e\n", v[i + 1][j + 1][k]);
        printf(" v[i-1][j-1][k-1] = %e\n", v[i - 1][j - 1][k - 1]);
        printf(" v[i-1][j-1][k+1] = %e\n", v[i - 1][j - 1][k + 1]);
        printf(" v[i-1][j+1][k-1] = %e\n", v[i - 1][j + 1][k - 1]);
        printf(" v[i-1][j+1][k+1] = %e\n", v[i - 1][j + 1][k + 1]);
        printf(" v[i+1][j-1][k-1] = %e\n", v[i + 1][j - 1][k - 1]);
        printf(" v[i+1][j-1][k+1] = %e\n", v[i + 1][j - 1][k + 1]);
        printf(" v[i+1][j+1][k-1] = %e\n", v[i + 1][j + 1][k - 1]);
        printf(" v[i+1][j+1][k+1] = %e\n", v[i + 1][j + 1][k + 1]);
    }

    void MultigridEngine::print27point_sv(Cuboid& v, int i, int j, int k,
                                          VaryingStencil& sv, int i_sv, int j_sv, int k_sv)
    {
        // clang-format off
        printf("27point stencil at %d,%d,%d; %d,%d,%d:\n", i_sv, j_sv, k_sv, i, j, k);
        printf(" sv * v[    self     ] = %e * %e\n", sv[1][1][1][i_sv][j_sv][k_sv], v[ i ][ j ][ k ]);
        printf(" sv * v[ i ][ j ][k-1] = %e * %e\n", sv[1][1][0][i_sv][j_sv][k_sv], v[ i ][ j ][k-1]);
        printf(" sv * v[ i ][ j ][k+1] = %e * %e\n", sv[1][1][2][i_sv][j_sv][k_sv], v[ i ][ j ][k+1]);
        printf(" sv * v[ i ][j-1][ k ] = %e * %e\n", sv[1][0][1][i_sv][j_sv][k_sv], v[ i ][j-1][ k ]);
        printf(" sv * v[ i ][j+1][ k ] = %e * %e\n", sv[1][2][1][i_sv][j_sv][k_sv], v[ i ][j+1][ k ]);
        printf(" sv * v[i-1][ j ][ k ] = %e * %e\n", sv[0][1][1][i_sv][j_sv][k_sv], v[i-1][ j ][ k ]);
        printf(" sv * v[i+1][ j ][ k ] = %e * %e\n", sv[2][1][1][i_sv][j_sv][k_sv], v[i+1][ j ][ k ]);
        printf(" sv * v[ i ][j-1][k-1] = %e * %e\n", sv[1][0][0][i_sv][j_sv][k_sv], v[ i ][j-1][k-1]);
        printf(" sv * v[ i ][j-1][k+1] = %e * %e\n", sv[1][0][2][i_sv][j_sv][k_sv], v[ i ][j-1][k+1]);
        printf(" sv * v[ i ][j+1][k-1] = %e * %e\n", sv[1][2][0][i_sv][j_sv][k_sv], v[ i ][j+1][k-1]);
        printf(" sv * v[ i ][j+1][k+1] = %e * %e\n", sv[1][2][2][i_sv][j_sv][k_sv], v[ i ][j+1][k+1]);
        printf(" sv * v[i-1][ j ][k-1] = %e * %e\n", sv[0][1][0][i_sv][j_sv][k_sv], v[i-1][ j ][k-1]);
        printf(" sv * v[i-1][ j ][k+1] = %e * %e\n", sv[0][1][2][i_sv][j_sv][k_sv], v[i-1][ j ][k+1]);
        printf(" sv * v[i+1][ j ][k-1] = %e * %e\n", sv[2][1][0][i_sv][j_sv][k_sv], v[i+1][ j ][k-1]);
        printf(" sv * v[i+1][ j ][k+1] = %e * %e\n", sv[2][1][2][i_sv][j_sv][k_sv], v[i+1][ j ][k+1]);
        printf(" sv * v[i-1][j-1][ k ] = %e * %e\n", sv[0][0][1][i_sv][j_sv][k_sv], v[i-1][j-1][ k ]);
        printf(" sv * v[i-1][j+1][ k ] = %e * %e\n", sv[0][2][1][i_sv][j_sv][k_sv], v[i-1][j+1][ k ]);
        printf(" sv * v[i+1][j-1][ k ] = %e * %e\n", sv[2][0][1][i_sv][j_sv][k_sv], v[i+1][j-1][ k ]);
        printf(" sv * v[i+1][j+1][ k ] = %e * %e\n", sv[2][2][1][i_sv][j_sv][k_sv], v[i+1][j+1][ k ]);
        printf(" sv * v[i-1][j-1][k-1] = %e * %e\n", sv[0][0][0][i_sv][j_sv][k_sv], v[i-1][j-1][k-1]);
        printf(" sv * v[i-1][j-1][k+1] = %e * %e\n", sv[0][0][2][i_sv][j_sv][k_sv], v[i-1][j-1][k+1]);
        printf(" sv * v[i-1][j+1][k-1] = %e * %e\n", sv[0][2][0][i_sv][j_sv][k_sv], v[i-1][j+1][k-1]);
        printf(" sv * v[i-1][j+1][k+1] = %e * %e\n", sv[0][2][2][i_sv][j_sv][k_sv], v[i-1][j+1][k+1]);
        printf(" sv * v[i+1][j-1][k-1] = %e * %e\n", sv[2][0][0][i_sv][j_sv][k_sv], v[i+1][j-1][k-1]);
        printf(" sv * v[i+1][j-1][k+1] = %e * %e\n", sv[2][0][2][i_sv][j_sv][k_sv], v[i+1][j-1][k+1]);
        printf(" sv * v[i+1][j+1][k-1] = %e * %e\n", sv[2][2][0][i_sv][j_sv][k_sv], v[i+1][j+1][k-1]);
        printf(" sv * v[i+1][j+1][k+1] = %e * %e\n", sv[2][2][2][i_sv][j_sv][k_sv], v[i+1][j+1][k+1]);
        // clang-format on
    }
}
