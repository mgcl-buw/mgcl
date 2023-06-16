#include "cuboid.hpp"           // for Cuboid
#include "hypercube.hpp"        // for Hypercube6d
#include "level.hpp"            // for Level
#include "mgcl.hpp"             // for mgcl_debug, MGCL_LAPLACE_19POINT
#include "multigrid_engine.hpp" // for Problem, VaryingStencil3x3x3, Multig...
#include "problem.hpp"          // for Problem
#include "stencil.hpp"          // for mgclCheckError, VaryingStencil3x3x3
#include "util.hpp"

#include <cstdio> // for printf, size_t, NULL
#include <math.h> // for fabs, sqrt, ceil
#include <memory> // for __shared_ptr_access, shared_ptr

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
     * m,n,o is size of real grid */
    double MultigridEngine::jacobiSeq(Cuboid &v, Cuboid &f, Cuboid &r, double omega,
                                      int maxiter, MGCL_RESIDUAL_NORM resnorm, MGCL_STENCIL stencilType,
                                      double stencilFactor, VaryingStencil3x3x3 &stencilValues, bool returnResidualNorm,
                                      bool periodic)
    {
        double res = 0.0;
        double h2 = 1.0 / ((double)(v.getM() * v.getM()));
        double dinv = h2 / 6.0;
        if (stencilType == MGCL_LAPLACE_19POINT)
            dinv = (6.0 * h2) / 24.0;
        else if (stencilType == MGCL_LAPLACE_27POINT)
            dinv = (30.0 * h2) / 128.0;

        double ***vraw = v.getData();

        for (int iter = 0; iter < maxiter; iter++)
        {
            // update ghost cells for periodic boundary condition
            if (periodic)
                MultigridEngine::updateGhostsSeq(v);

            // damped/weighted iteration formula: u_(m+1) = u_(m) + omega * D^-1 * r_(m)

            // r = f - A*v
            res = residualSeq(f, v, r, resnorm, stencilType, stencilFactor, stencilValues, false, periodic);

            if (stencilType == MGCL_LAPLACE_7POINT || stencilType == MGCL_LAPLACE_19POINT || stencilType == MGCL_LAPLACE_27POINT)
            {
                for (int i = f.getGhostsM(); i < f.getM() + f.getGhostsM(); i++)
                    for (int j = f.getGhostsN(); j < f.getN() + f.getGhostsN(); j++)
                        for (int k = f.getGhostsO(); k < f.getO() + f.getGhostsO(); k++)
                        {
                            // if (i == 1 && j == 1 && k == 1)
                            //     printf("v[%d][%d][%d] = %f, r[%d][%d][%d] = %f, omega = %f\n", i,j,k, vraw[i][j][k],
                            //     i,j,k,r[i][j][k], omega);
                            vraw[i][j][k] = vraw[i][j][k] + omega * dinv * r[i][j][k];
                        }
            }
            else
            {
                // printf("seq x = %d, omega = %e, res = %e, v_out = %e, sv_self = %e\n", 1, omega, r[1][2][5], vraw[1][2][5], stencilValues[2][3][6][1][1][1]);
                // print_7point(v_in, index, ioff, joff, koff);
                // print27point(v, 1, 2, 5);
                // print27point_sv(v, 1, 2, 5, stencilValues, 2, 3, 6);

                int ghmsv = stencilValues.getGhostsDim1();
                int ghnsv = stencilValues.getGhostsDim2();
                int ghosv = stencilValues.getGhostsDim3();
                for (int i = f.getGhostsM(), isv = ghmsv; i < f.getM() + f.getGhostsM(); i++, isv++)
                    for (int j = f.getGhostsN(), jsv = ghnsv; j < f.getN() + f.getGhostsN(); j++, jsv++)
                        for (int k = f.getGhostsO(), ksv = ghosv; k < f.getO() + f.getGhostsO(); k++, ksv++)
                        {
                            // if (i == 1 && j == 1 && k == 1)
                            //     printf("v[%d][%d][%d] = %f, r[%d][%d][%d] = %f, omega = %f\n", i,j,k, vraw[i][j][k],
                            //     i,j,k,r[i][j][k], omega);
                            vraw[i][j][k] = vraw[i][j][k] + omega * (1.0 / stencilValues[isv][jsv][ksv][1][1][1]) * r[i][j][k];

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

        if (periodic)
            MultigridEngine::updateGhostsSeq(v);

        if (returnResidualNorm)
            res = residualSeq(f, v, r, resnorm, stencilType, stencilFactor, stencilValues, returnResidualNorm, periodic);

        return res;
    }

    /* Runs jacobi method using OpenCL.
     * Doesn't creates ocl buffers and doesn't copy data from host to device and vice versa
     * v, f and r must be of size [m][n][o] for periodic boundary condition.
     * m, n and o must be the dimensions of grid + 2*ghosts
     * If return_residual is true, the residual's 2-norm or inf-norm will be read back from device and returned, else -1.
     * It's not
     * really performant to do so because we have to wait for all kernels to complete and reading a buffer to host is slow.
     */
    double MultigridEngine::jacobi(Problem &problem, Level &level, int maxiter, bool return_residual)
    {
        int err;
        int mgh = level.mgh;
        int ngh = level.ngh;
        int ogh = level.ogh;
        int store_res = 0;
        double res = -1;

        if (problem.use_local_memory)
        {
            res = jacobiLocalMem(problem, level, maxiter, return_residual);
            if (res != -2)
                return res;

            mgcl_debug("jacobiLocalMem apparently failed. Running global mem version instead.\n");
        }

        double h2 = (1.0 / (double)level.m) *
                    (1.0 / (double)level.m); // TODO minimum of m,n,o when not cube?
        double dinv = h2 / 6.0;
        double h2inv = level.stencilFactor; // divisor of the stencil, inverted to use * instead of / in kernel

        // Create the compute kernel from the program
        const char *kernel_name;
        if (problem.stencilType == MGCL_LAPLACE_7POINT)
            kernel_name = "jacobi_iter_7point";
        else if (problem.stencilType == MGCL_LAPLACE_19POINT)
        {
            kernel_name = "jacobi_iter_19point";
            dinv = (6.0 * h2) / 24.0;
        }
        else if (problem.stencilType == MGCL_LAPLACE_27POINT)
        {
            kernel_name = "jacobi_iter_27point";
            dinv = (30.0 * h2) / 128.0;
        }
        else if (problem.stencilType == MGCL_VARYING)
        {
            kernel_name = "jacobi_iter_27point_varying_stencil";
        }

        cl_kernel kernel = clCreateKernel(problem.openCLHelper.getProgram(), kernel_name, &err);
        mgclCheckError(err, "Creating kernel");

        // assign kernel arguments
        int pos = 0;

        if (problem.stencilType == MGCL_VARYING)
        {
            auto svbuf = level.stencilValuesGpu->getBuf();
            int svgh = level.stencilValuesGpu->getGh();
            err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &level.dVIn);
            err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &level.dVOut);
            err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &level.dF);
            err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &level.dR);
            err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &svbuf);
            err |= clSetKernelArg(kernel, ++pos, sizeof(double), &problem.omega);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &mgh);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ngh);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ogh);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &problem.ghosts);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svgh);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &store_res);
        }
        else
        {
            err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &level.dVIn);
            err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &level.dVOut);
            err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &level.dF);
            err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &level.dR);
            err |= clSetKernelArg(kernel, ++pos, sizeof(double), &h2inv);
            err |= clSetKernelArg(kernel, ++pos, sizeof(double), &dinv);
            err |= clSetKernelArg(kernel, ++pos, sizeof(double), &problem.omega);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &mgh);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ngh);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ogh);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &problem.ghosts);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &store_res);
        }
        mgclCheckError(err, "Setting kernel arguments");

        // one work-item per cell (including ghost cells). Pad global sizes to fit to local sizes
        size_t global[2] = {static_cast<size_t>(ngh), static_cast<size_t>(ogh)};
        const size_t local[2] = {static_cast<size_t>(ngh > 4 ? 4 : ngh),
                                 static_cast<size_t>(ogh > 8 ? 8 : ogh)}; // TODO problem.jacobi_wg_size_x

        for (int i = 0; i < 2; i++)
            if (global[i] % local[i] != 0)
            {
                // printf("padding global size %d from %ld to ", i, global[i]);
                global[i] += local[i] - (global[i] % local[i]);
                // printf("%ld (multiple of %ld)\n", global[i], local[i]);
            }

        for (int iter = 0; iter < maxiter; iter++)
        {
            // switch arguments dVIn -> dVOut to use latest values in next iteration
            if (iter % 2 == 1)
            {
                err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &level.dVIn);
                err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &level.dVOut);
                mgclCheckError(err, "Setting kernel arguments");

                err = MultigridEngine::updateGhosts(problem, level.dVOut, mgh, ngh, ogh, problem.ghosts, problem.ghosts, problem.ghosts);
                mgclCheckError(err, "Updating ghosts");
            }
            else
            {
                err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &level.dVIn);
                err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &level.dVOut);
                mgclCheckError(err, "Setting kernel arguments");

                err = MultigridEngine::updateGhosts(problem, level.dVIn, mgh, ngh, ogh, problem.ghosts, problem.ghosts, problem.ghosts);
                mgclCheckError(err, "Updating ghosts");
            }

            // set flag to store residual in last iteration
            if (iter == maxiter - 1)
            {
                store_res = 1;
                err = clSetKernelArg(kernel, pos, sizeof(int), &store_res);
                mgclCheckError(err, "Setting kernel arguments");
            }

            err = clEnqueueNDRangeKernel(problem.getOpenCLHelper().getCommands(), kernel, 2, NULL, global, local, 0, NULL, NULL);
            mgclCheckError(err, "Enqueueing kernel");
        }

        if (store_res)
        {
            err = MultigridEngine::updateGhosts(problem, level.dR, mgh, ngh, ogh, problem.ghosts, problem.ghosts, problem.ghosts);
            mgclCheckError(err, "Updating ghosts of dR");
        }

        // copy result into dVIn if needed
        if (maxiter % 2 == 1)
        {
            err = clEnqueueCopyBuffer(problem.getOpenCLHelper().getCommands(), level.dVOut, level.dVIn, 0, 0, sizeof(double) * mgh * ngh * ogh, 0,
                                      NULL, NULL);
            mgclCheckError(err, "Update v");
        }

        err = MultigridEngine::updateGhosts(problem, level.dVIn, mgh, ngh, ogh, problem.ghosts, problem.ghosts, problem.ghosts);
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

    /* Runs jacobi method using OpenCL.
     * Doesn't creates ocl buffers and doesn't copy data from host to device and vice versa
     * v, f and r must be of size [m][n][o] for periodic boundary condition.
     * m, n and o must be the dimensions of grid + 2*ghosts
     * If return_residual is true, the residual's 2-norm or inf-norm will be read back from device and returned, else -1.
     * It's not really performant to do so because we have to wait for all kernels to complete and reading a buffer to host
     * is slow. Runs multiple iterations per kernel call using local memory if enough is available Only jacobi_wg_size_x is
     * used for now. If there is not enough local memory available the kernel will not be called and -2 is returned. */
    double MultigridEngine::jacobiLocalMem(Problem &problem, Level &level, int maxiter, int return_residual)
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
        int locmem_size_wg = 3 * ipk * (wg_size + 2 * ipk) * (wg_size + 2 * ipk) * sizeof(double);

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
        const char *kernel_name;
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
            dinv = (30.0 * h2) / 128.0;
        }
        cl_kernel kernel = clCreateKernel(problem.openCLHelper.getProgram(), kernel_name, &err);
        mgclCheckError(err, "Creating kernel");

        // assign kernel arguments
        int pos = 0;
        err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &level.dVIn);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &level.dVOut);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &level.dF);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &level.dR);
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

        cl_mem tmp;
        if (ipk == maxiter)
        {
            // set flag to store residual of last iteration
            store_res = 1;
            err = clSetKernelArg(kernel, pos, sizeof(int), &store_res);
            mgclCheckError(err, "Setting kernel arguments");

            err = MultigridEngine::updateGhosts(problem, level.dVIn, mgh, ngh, ogh, problem.ghosts, problem.ghosts, problem.ghosts);
            mgclCheckError(err, "Updating ghosts");

            err = clEnqueueNDRangeKernel(problem.getOpenCLHelper().getCommands(), kernel, 2, NULL, global, local, 0, NULL, NULL);
            mgclCheckError(err, "Enqueueing kernel");

            // swap pointers so result is in dVIn
            tmp = level.dVIn;
            level.dVIn = level.dVOut;
            level.dVOut = tmp;
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

                err = MultigridEngine::updateGhosts(problem, level.dVOut, mgh, ngh, ogh, problem.ghosts, problem.ghosts, problem.ghosts);
                mgclCheckError(err, "Updating ghosts");

                // swap pointers so result is in dVIn
                tmp = level.dVIn;
                level.dVIn = level.dVOut;
                level.dVOut = tmp;

                err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &level.dVIn);
                err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &level.dVOut);
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

                err = MultigridEngine::updateGhosts(problem, level.dVOut, mgh, ngh, ogh, problem.ghosts, problem.ghosts, problem.ghosts);
                mgclCheckError(err, "Updating ghosts");

                // swap pointers so result is in dVIn
                tmp = level.dVIn;
                level.dVIn = level.dVOut;
                level.dVOut = tmp;
            }
        }
        // result is in dVIn now since pointers were swapped at the end of the loops above

        err = MultigridEngine::updateGhosts(problem, level.dVIn, mgh, ngh, ogh, problem.ghosts, problem.ghosts, problem.ghosts);
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
     */
    double MultigridEngine::residual(Problem &problem, Level &level, bool return_residual)
    {
        int err;
        int mgh = level.mgh;
        int ngh = level.ngh;
        int ogh = level.ogh;
        double res = 0.0;

        double h2 = (1.0 / (double)level.m) *
                    (1.0 / (double)level.m); // TODO minimum of m,n,o when not cube?
        double h2inv = 1.0 / h2;             // divisor of the stencil, inverted to use * instead of / in kernel

        // Create the compute kernel from the program
        const char *kernel_name;
        if (problem.stencilType == MGCL_LAPLACE_7POINT)
            kernel_name = "residual_7point";
        else if (problem.stencilType == MGCL_LAPLACE_19POINT)
        {
            kernel_name = "residual_19point";
            h2inv = 1.0 / (6.0 * h2);
        }
        else if (problem.stencilType == MGCL_LAPLACE_27POINT)
        {
            kernel_name = "residual_27point";
            h2inv = 1.0 / (30.0 * h2);
        }
        else if (problem.stencilType == MGCL_VARYING)
        {
            kernel_name = "residual_27point_varying_stencil";
        }

        // Create the compute kernel from the program
        cl_kernel kernel = clCreateKernel(problem.openCLHelper.getProgram(), kernel_name, &err);
        mgclCheckError(err, "Creating kernel");

        // assign kernel arguments
        int pos = 0;

        if (problem.stencilType == MGCL_VARYING)
        {
            auto svbuf = level.stencilValuesGpu->getBuf();
            int svgh = level.stencilValuesGpu->getGh();
            err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &level.dVIn);
            err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &level.dF);
            err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &level.dR);
            err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &svbuf);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &mgh);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ngh);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ogh);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &problem.ghosts);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svgh);
        }
        else
        {
            err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &level.dVIn);
            err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &level.dF);
            err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &level.dR);
            err |= clSetKernelArg(kernel, ++pos, sizeof(double), &h2inv);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &mgh);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ngh);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ogh);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &problem.ghosts);
        }

        mgclCheckError(err, "Setting residual kernel arguments");

        // one work-item per cell (including ghost cells). Pad global sizes to fit to local sizes
        size_t global[3] = {static_cast<size_t>(mgh), static_cast<size_t>(ngh), static_cast<size_t>(ogh)};
        const size_t local[3] = {static_cast<size_t>(mgh > 4 ? 4 : mgh), static_cast<size_t>(ngh > 4 ? 4 : ngh),
                                 static_cast<size_t>(ogh > 4 ? 4 : ogh)};

        for (int i = 0; i < 3; i++)
            if (global[i] % local[i] != 0)
            {
                // printf("padding global size %d from %ld to ", i, global[i]);
                global[i] += local[i] - (global[i] % local[i]);
                // printf("%ld (multiple of %ld)\n", global[i], local[i]);
            }

        err = clEnqueueNDRangeKernel(problem.getOpenCLHelper().getCommands(), kernel, 3, NULL, global, local, 0, NULL, NULL);
        mgclCheckError(err, "Enqueueing residual kernel");

        if (problem.isPeriodic())
        {
            err = MultigridEngine::updateGhosts(problem, level.dR, mgh, ngh, ogh, problem.ghosts, problem.ghosts, problem.ghosts);
            mgclCheckError(err, "Updating ghosts of r");
        }

        // calculate residual's 2-norm. Square elements on device and sum up on host
        if (return_residual)
        {
            if (problem.residual_norm == MGCL_L2)
            {
                // calculate 2-Norm
                Cuboid rsq(mgh, ngh, ogh);
                double ***rsquares = rsq.getData();
                int pointer_flag = problem.getOpenCLHelper().getDeviceType() == CL_DEVICE_TYPE_GPU ? CL_MEM_COPY_HOST_PTR : CL_MEM_USE_HOST_PTR;
                cl_mem dRsquares = clCreateBuffer(problem.getOpenCLHelper().getContext(), CL_MEM_WRITE_ONLY | pointer_flag,
                                                  sizeof(double) * mgh * ngh * ogh, rsquares[0][0], &err);
                mgclCheckError(err, "Creating rsquares buffer");

                // Create the compute kernel from the program
                cl_kernel kernel_square = clCreateKernel(problem.openCLHelper.getProgram(), "residual_squared", &err);
                mgclCheckError(err, "Creating residual squared kernel");

                pos = 0;
                err = clSetKernelArg(kernel_square, pos, sizeof(cl_mem), &level.dR);
                err |= clSetKernelArg(kernel_square, ++pos, sizeof(cl_mem), &dRsquares);
                err |= clSetKernelArg(kernel_square, ++pos, sizeof(int), &mgh);
                err |= clSetKernelArg(kernel_square, ++pos, sizeof(int), &ngh);
                err |= clSetKernelArg(kernel_square, ++pos, sizeof(int), &ogh);
                err |= clSetKernelArg(kernel_square, ++pos, sizeof(int), &problem.ghosts);
                mgclCheckError(err, "Setting residual squared kernel arguments");

                err = clEnqueueNDRangeKernel(problem.getOpenCLHelper().getCommands(), kernel_square, 3, NULL, global, local, 0, NULL, NULL);
                mgclCheckError(err, "Enqueueing residual squared kernel");

                // sum up residual squares
                res = sqrt(util::sum(dRsquares, mgh * ngh * ogh,
                                     problem.getContext(), problem.getProgram(), problem.getCommands(), true));

                clReleaseMemObject(dRsquares);
                clReleaseKernel(kernel_square);
            }
            else
            {
                // calculate Infinity-Norm
                res = util::max_abs(level.dR, mgh * ngh * ogh,
                                    problem.getContext(), problem.getProgram(), problem.getCommands(), true);
            }
        }

        clReleaseKernel(kernel); // TODO maybe clFinish before release?
        return res;
    }

    /* Calculates r = f - A*v using 7-point stencil of 3D laplacian.
     * m,n,o is size of real grid
     * v needs to have updated ghost cells if the problem is periodic!
     */
    double MultigridEngine::residualSeq(Cuboid &f, Cuboid &v, Cuboid &r, MGCL_RESIDUAL_NORM resnorm,
                                        MGCL_STENCIL stencilType, double stencilFactor, VaryingStencil3x3x3 &stencilValuesCuboid, bool returnResidualNorm, bool periodic)
    {
        // TODO adjust when stencilValues are stored with ghosts
        double res = 0.0;
        double stencilsum = 0;
        double ******stencilValues;
        double ***vraw = v.getData();

        int ghm = f.getGhostsM();
        int ghn = f.getGhostsN();
        int gho = f.getGhostsO();
        int ghmsv = 0;
        int ghnsv = 0;
        int ghosv = 0;

        if (stencilType == MGCL_VARYING)
        {
            stencilValues = stencilValuesCuboid.getData();
            ghmsv = stencilValuesCuboid.getGhostsDim1();
            ghnsv = stencilValuesCuboid.getGhostsDim2();
            ghosv = stencilValuesCuboid.getGhostsDim3();
            // if (ghmsv != ghm ||
            //     ghnsv != ghn ||
            //     ghosv != gho)
            //     throw "Ghosts of inputs and stencilValues must be the same!";
        }

        for (int i = ghm, isv = ghmsv; i < f.getM() + ghm; i++, isv++)
            for (int j = ghn, jsv = ghnsv; j < f.getN() + ghn; j++, jsv++)
                for (int k = gho, ksv = ghosv; k < f.getO() + gho; k++, ksv++)
                {
                    // A*v
                    if (stencilType == MGCL_LAPLACE_7POINT)
                    {
                        // clang-format off
                        stencilsum = (6.0 * vraw[i][j][k]
                            - vraw[i][j][k - 1] - vraw[i][j][k + 1]
                            - vraw[i][j - 1][k] - vraw[i][j + 1][k]
                            - vraw[i - 1][j][k] - vraw[i + 1][j][k]
                            ) * stencilFactor;
                        // clang-format on
                    }
                    else if (stencilType == MGCL_LAPLACE_19POINT)
                    {
                        // clang-format off
                        stencilsum = (24.0 * vraw[i][j][k]
                                - 2.0 * vraw[i][j][k - 1] - 2.0 * vraw[i][j][k + 1]
                                - 2.0 * vraw[i][j - 1][k] - 2.0 * vraw[i][j + 1][k]
                                - 2.0 * vraw[i - 1][j][k] - 2.0 * vraw[i + 1][j][k]
                                
                                - vraw[i][j - 1][k - 1] - vraw[i][j - 1][k + 1]
                                - vraw[i][j + 1][k - 1] - vraw[i][j + 1][k + 1]
                                - vraw[i - 1][j][k - 1] - vraw[i - 1][j][k + 1]
                                - vraw[i + 1][j][k - 1] - vraw[i + 1][j][k + 1]
                                - vraw[i - 1][j - 1][k] - vraw[i - 1][j + 1][k]
                                - vraw[i + 1][j - 1][k] - vraw[i + 1][j + 1][k]
                                ) * stencilFactor;
                        // clang-format on
                    }
                    else if (stencilType == MGCL_LAPLACE_27POINT)
                    {
                        // clang-format off
                        stencilsum = (128.0 * vraw[i][j][k]
                                - 14.0 * vraw[i][j][k - 1] - 14.0 * vraw[i][j][k + 1]
                                - 14.0 * vraw[i][j - 1][k] - 14.0 * vraw[i][j + 1][k]
                                - 14.0 * vraw[i - 1][j][k] - 14.0 * vraw[i + 1][j][k]

                                - 3.0 * vraw[i][j - 1][k - 1] - 3.0 * vraw[i][j - 1][k + 1]
                                - 3.0 * vraw[i][j + 1][k - 1] - 3.0 * vraw[i][j + 1][k + 1]
                                - 3.0 * vraw[i - 1][j][k - 1] - 3.0 * vraw[i - 1][j][k + 1]
                                - 3.0 * vraw[i + 1][j][k - 1] - 3.0 * vraw[i + 1][j][k + 1]
                                - 3.0 * vraw[i - 1][j - 1][k] - 3.0 * vraw[i - 1][j + 1][k]
                                - 3.0 * vraw[i + 1][j - 1][k] - 3.0 * vraw[i + 1][j + 1][k]

                                - vraw[i - 1][j - 1][k - 1] - vraw[i - 1][j - 1][k + 1]
                                - vraw[i - 1][j + 1][k - 1] - vraw[i - 1][j + 1][k + 1]
                                - vraw[i + 1][j - 1][k - 1] - vraw[i + 1][j - 1][k + 1]
                                - vraw[i + 1][j + 1][k - 1] - vraw[i + 1][j + 1][k + 1]
                                ) * stencilFactor;
                        // clang-format on
                    }
                    else if (stencilType == MGCL_VARYING)
                    {
                        // clang-format off
                        stencilsum = stencilValues[isv][jsv][ksv][1][1][1]  * vraw[i][j][k]
                            + stencilValues[isv][jsv][ksv][1][1][0] * vraw[ i ][ j ][k-1]
                            + stencilValues[isv][jsv][ksv][1][1][2] * vraw[ i ][ j ][k+1]
                            + stencilValues[isv][jsv][ksv][1][0][1] * vraw[ i ][j-1][ k ]
                            + stencilValues[isv][jsv][ksv][1][2][1] * vraw[ i ][j+1][ k ]
                            + stencilValues[isv][jsv][ksv][0][1][1] * vraw[i-1][ j ][ k ]
                            + stencilValues[isv][jsv][ksv][2][1][1] * vraw[i+1][ j ][ k ]
                            
                            + stencilValues[isv][jsv][ksv][1][0][0] * vraw[ i ][j-1][k-1]
                            + stencilValues[isv][jsv][ksv][1][0][2] * vraw[ i ][j-1][k+1]
                            + stencilValues[isv][jsv][ksv][1][2][0] * vraw[ i ][j+1][k-1]
                            + stencilValues[isv][jsv][ksv][1][2][2] * vraw[ i ][j+1][k+1]
                            + stencilValues[isv][jsv][ksv][0][1][0] * vraw[i-1][ j ][k-1]
                            + stencilValues[isv][jsv][ksv][0][1][2] * vraw[i-1][ j ][k+1]
                            + stencilValues[isv][jsv][ksv][2][1][0] * vraw[i+1][ j ][k-1]
                            + stencilValues[isv][jsv][ksv][2][1][2] * vraw[i+1][ j ][k+1]
                            + stencilValues[isv][jsv][ksv][0][0][1] * vraw[i-1][j-1][ k ]
                            + stencilValues[isv][jsv][ksv][0][2][1] * vraw[i-1][j+1][ k ]
                            + stencilValues[isv][jsv][ksv][2][0][1] * vraw[i+1][j-1][ k ]
                            + stencilValues[isv][jsv][ksv][2][2][1] * vraw[i+1][j+1][ k ]
                            
                            + stencilValues[isv][jsv][ksv][0][0][0] * vraw[i-1][j-1][k-1]
                            + stencilValues[isv][jsv][ksv][0][0][2] * vraw[i-1][j-1][k+1]
                            + stencilValues[isv][jsv][ksv][0][2][0] * vraw[i-1][j+1][k-1]
                            + stencilValues[isv][jsv][ksv][0][2][2] * vraw[i-1][j+1][k+1]
                            + stencilValues[isv][jsv][ksv][2][0][0] * vraw[i+1][j-1][k-1]
                            + stencilValues[isv][jsv][ksv][2][0][2] * vraw[i+1][j-1][k+1]
                            + stencilValues[isv][jsv][ksv][2][2][0] * vraw[i+1][j+1][k-1]
                            + stencilValues[isv][jsv][ksv][2][2][2] * vraw[i+1][j+1][k+1];
                        // clang-format on

                        // if (j == 2 && k == 2 && i == 2)
                        // {
                        //     printf("seq stencilsum = %e\n", stencilsum);
                        //     print27point_sv(v, i, j, k, stencilValuesCuboid, isv, jsv, ksv);
                        // }
                    }

                    // r = f - A*v
                    r[i][j][k] = f[i][j][k] - stencilsum;

                    if (returnResidualNorm)
                    {
                        if (resnorm == MGCL_L2)
                            res += r[i][j][k] * r[i][j][k];
                        else if (fabs(r[i][j][k]) > res)
                            res = fabs(r[i][j][k]);
                    }
                }

        if (periodic)
            MultigridEngine::updateGhostsSeq(r);

        return (returnResidualNorm && resnorm == MGCL_L2) ? sqrt(res) : res;
    }

    /* Prints components of 7-point laplacian stencil for debugging purposes */
    void MultigridEngine::print7point(Cuboid &v, int i, int j, int k)
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

    void MultigridEngine::print19point(Cuboid &v, int i, int j, int k)
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

    void MultigridEngine::print27point(Cuboid &v, int i, int j, int k)
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

    void MultigridEngine::print27point_sv(Cuboid &v, int i, int j, int k,
                                          VaryingStencil3x3x3 &sv, int i_sv, int j_sv, int k_sv)
    {
        // clang-format off
        printf("27point stencil at %d,%d,%d; %d,%d,%d:\n", i_sv, j_sv, k_sv, i, j, k);
        printf(" sv * v[    self     ] = %e * %e\n", sv[i_sv][j_sv][k_sv][1][1][1], v[ i ][ j ][ k ]);
        printf(" sv * v[ i ][ j ][k-1] = %e * %e\n", sv[i_sv][j_sv][k_sv][1][1][0], v[ i ][ j ][k-1]);
        printf(" sv * v[ i ][ j ][k+1] = %e * %e\n", sv[i_sv][j_sv][k_sv][1][1][2], v[ i ][ j ][k+1]);
        printf(" sv * v[ i ][j-1][ k ] = %e * %e\n", sv[i_sv][j_sv][k_sv][1][0][1], v[ i ][j-1][ k ]);
        printf(" sv * v[ i ][j+1][ k ] = %e * %e\n", sv[i_sv][j_sv][k_sv][1][2][1], v[ i ][j+1][ k ]);
        printf(" sv * v[i-1][ j ][ k ] = %e * %e\n", sv[i_sv][j_sv][k_sv][0][1][1], v[i-1][ j ][ k ]);
        printf(" sv * v[i+1][ j ][ k ] = %e * %e\n", sv[i_sv][j_sv][k_sv][2][1][1], v[i+1][ j ][ k ]);
        printf(" sv * v[ i ][j-1][k-1] = %e * %e\n", sv[i_sv][j_sv][k_sv][1][0][0], v[ i ][j-1][k-1]);
        printf(" sv * v[ i ][j-1][k+1] = %e * %e\n", sv[i_sv][j_sv][k_sv][1][0][2], v[ i ][j-1][k+1]);
        printf(" sv * v[ i ][j+1][k-1] = %e * %e\n", sv[i_sv][j_sv][k_sv][1][2][0], v[ i ][j+1][k-1]);
        printf(" sv * v[ i ][j+1][k+1] = %e * %e\n", sv[i_sv][j_sv][k_sv][1][2][2], v[ i ][j+1][k+1]);
        printf(" sv * v[i-1][ j ][k-1] = %e * %e\n", sv[i_sv][j_sv][k_sv][0][1][0], v[i-1][ j ][k-1]);
        printf(" sv * v[i-1][ j ][k+1] = %e * %e\n", sv[i_sv][j_sv][k_sv][0][1][2], v[i-1][ j ][k+1]);
        printf(" sv * v[i+1][ j ][k-1] = %e * %e\n", sv[i_sv][j_sv][k_sv][2][1][0], v[i+1][ j ][k-1]);
        printf(" sv * v[i+1][ j ][k+1] = %e * %e\n", sv[i_sv][j_sv][k_sv][2][1][2], v[i+1][ j ][k+1]);
        printf(" sv * v[i-1][j-1][ k ] = %e * %e\n", sv[i_sv][j_sv][k_sv][0][0][1], v[i-1][j-1][ k ]);
        printf(" sv * v[i-1][j+1][ k ] = %e * %e\n", sv[i_sv][j_sv][k_sv][0][2][1], v[i-1][j+1][ k ]);
        printf(" sv * v[i+1][j-1][ k ] = %e * %e\n", sv[i_sv][j_sv][k_sv][2][0][1], v[i+1][j-1][ k ]);
        printf(" sv * v[i+1][j+1][ k ] = %e * %e\n", sv[i_sv][j_sv][k_sv][2][2][1], v[i+1][j+1][ k ]);
        printf(" sv * v[i-1][j-1][k-1] = %e * %e\n", sv[i_sv][j_sv][k_sv][0][0][0], v[i-1][j-1][k-1]);
        printf(" sv * v[i-1][j-1][k+1] = %e * %e\n", sv[i_sv][j_sv][k_sv][0][0][2], v[i-1][j-1][k+1]);
        printf(" sv * v[i-1][j+1][k-1] = %e * %e\n", sv[i_sv][j_sv][k_sv][0][2][0], v[i-1][j+1][k-1]);
        printf(" sv * v[i-1][j+1][k+1] = %e * %e\n", sv[i_sv][j_sv][k_sv][0][2][2], v[i-1][j+1][k+1]);
        printf(" sv * v[i+1][j-1][k-1] = %e * %e\n", sv[i_sv][j_sv][k_sv][2][0][0], v[i+1][j-1][k-1]);
        printf(" sv * v[i+1][j-1][k+1] = %e * %e\n", sv[i_sv][j_sv][k_sv][2][0][2], v[i+1][j-1][k+1]);
        printf(" sv * v[i+1][j+1][k-1] = %e * %e\n", sv[i_sv][j_sv][k_sv][2][2][0], v[i+1][j+1][k-1]);
        printf(" sv * v[i+1][j+1][k+1] = %e * %e\n", sv[i_sv][j_sv][k_sv][2][2][2], v[i+1][j+1][k+1]);
        // clang-format on
    }
}
