#include <cstdio>
#include <ctgmath>

#include "cuboid.hpp"
#include "multigrid_engine.hpp"

namespace mgcl
{
    /* Runs jacobi method sequentially.
     * v, f and r must be of size [m+2][n+2][o+2] for periodic boundary condition.
     * m,n,o is size of real grid */
    double MultigridEngine::jacobiSeq(Cuboid &v, Cuboid &f, Cuboid &r, double omega,
                                      int maxiter, MGCL_RESIDUAL_NORM resnorm, MGCL_STENCIL stencil)
    {
        double res = 0.0;
        double h2 = 1.0 / ((double)(v.getM() * v.getM()));
        double dinv = h2 / 6.0;
        if (stencil == MGCL_19POINT)
            dinv = (6.0 * h2) / 24.0;
        else if (stencil == MGCL_27POINT)
            dinv = (30.0 * h2) / 128.0;

        for (int iter = 0; iter < maxiter; iter++)
        {
            // update ghost cells for periodic boundary condition
            MultigridEngine::updateGhostsSeq(v);

            // damped/weighted iteration formula: u_(m+1) = u_(m) + omega * D^-1 * r_(m)

            // r = f - A*v
            res = residualSeq(f, v, r, resnorm, stencil);
            for (int i = f.getGhostsM(); i < f.getM() + f.getGhostsM(); i++)
                for (int j = f.getGhostsN(); j < f.getN() + f.getGhostsN(); j++)
                    for (int k = f.getGhostsO(); k < f.getO() + f.getGhostsO(); k++)
                    {
                        // if (i == 1 && j == 1 && k == 1)
                        //     printf("v[%d][%d][%d] = %f, r[%d][%d][%d] = %f, omega = %f\n", i,j,k, v[i][j][k],
                        //     i,j,k,r[i][j][k], omega);
                        v[i][j][k] = v[i][j][k] + omega * dinv * r[i][j][k];
                    }
        }
        MultigridEngine::updateGhostsSeq(v);
        return res;
    }

    /* Tests jacobi method using OpenCL. Creates buffers and copies memory from host to device and back.
     * v, f and r must be of size [m][n][o] for periodic boundary condition.
     * m, n and o must be the dimensions of grid + 2*ghosts */
    void MultigridEngine::jacobiTest(Problem &problem, double ***v, double ***f, double ***r, int m, int n, int o, int maxiter,
                                     int readResults)
    {
        // int err;

        // // create device buffers
        // int pointer_flag = problem.getOpenCLHelper().getDeviceType() == CL_DEVICE_TYPE_GPU ? CL_MEM_COPY_HOST_PTR : CL_MEM_USE_HOST_PTR;
        // cl_mem dVIn =
        //     clCreateBuffer(problem.getOpenCLHelper().getContext(), CL_MEM_READ_WRITE | pointer_flag, sizeof(double) * m * n * o, v[0][0], &err);
        // cl_mem dVOut =
        //     clCreateBuffer(problem.getOpenCLHelper().getContext(), CL_MEM_READ_WRITE | pointer_flag, sizeof(double) * m * n * o, v[0][0], &err);
        // cl_mem d_f =
        //     clCreateBuffer(problem.getOpenCLHelper().getContext(), CL_MEM_READ_ONLY | pointer_flag, sizeof(double) * m * n * o, f[0][0], &err);
        // cl_mem dR =
        //     clCreateBuffer(problem.getOpenCLHelper().getContext(), CL_MEM_READ_WRITE | pointer_flag, sizeof(double) * m * n * o, r[0][0], &err);

        // // create level data
        // mgcl_level_data data;
        // data.dVIn = dVIn;
        // data.dVOut = dVOut;
        // data.dF = d_f;
        // data.dR = dR;
        // data.m = m;
        // data.n = n;
        // data.o = o;

        // MultigridEngine::updateGhosts(problem, dVIn, m, n, o, problem.ghosts, problem.ghosts, problem.ghosts);
        // MultigridEngine::updateGhosts(problem, d_f, m, n, o, problem.ghosts, problem.ghosts, problem.ghosts);

        // auto t_start_iter = std::chrono::steady_clock::now();
        // jacobi(problem, &data, maxiter, readResults);

        // // Wait for the commands to complete before stopping the timer
        // err = clFinish(problem.getOpenCLHelper().getCommands());
        // mgclCheckError(err, "Waiting for kernel to finish");
        // auto t_end_iter = mgcl_since(t_start_iter).count() * 1000.0;
        // printf("jacobi on opencl took %.3e s\n", t_end_iter);

        // // reassign pointers to v since they might've been swapped
        // dVIn = data.dVIn;
        // dVOut = data.dVOut;

        // // read back results TODO: only for testing purposes, maybe define TESTING?
        // err = clEnqueueReadBuffer(problem.getOpenCLHelper().getCommands(), dVIn, CL_FALSE, 0, sizeof(double) * m * n * o, v[0][0], 0, NULL, NULL);
        // err = clEnqueueReadBuffer(problem.getOpenCLHelper().getCommands(), dR, CL_TRUE, 0, sizeof(double) * m * n * o, r[0][0], 0, NULL, NULL);
        // if (err != CL_SUCCESS)
        // {
        //     printf("Error: Failed to read output arrays from device!\n%s\n", mgcl_err_code(err));
        //     exit(1);
        // }

        // clReleaseMemObject(dVIn);
        // clReleaseMemObject(dVOut);
        // clReleaseMemObject(d_f);
        // clReleaseMemObject(dR);
    }

    /* Runs jacobi method using OpenCL.
     * Doesn't creates ocl buffers and doesn't copy data from host to device and vice versa
     * v, f and r must be of size [m][n][o] for periodic boundary condition.
     * m, n and o must be the dimensions of grid + 2*ghosts
     * If return_residual is true, the residual's 2-norm or inf-norm will be read back from device and returned, else -1.
     * It's not
     * really performant to do so because we have to wait for all kernels to complete and reading a buffer to host is slow.
     */
    double MultigridEngine::jacobi(Problem &problem, Level &level, int maxiter, int return_residual)
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
        double h2inv = 1.0 / h2; // divisor of the stencil, inverted to use * instead of / in kernel

        // Create the compute kernel from the program
        const char *kernel_name;
        if (problem.stencil == MGCL_7POINT)
            kernel_name = "jacobi_iter_7point";
        else if (problem.stencil == MGCL_19POINT)
        {
            kernel_name = "jacobi_iter_19point";
            h2inv = 1.0 / (6.0 * h2);
            dinv = (6.0 * h2) / 24.0;
        }
        else if (problem.stencil == MGCL_27POINT)
        {
            kernel_name = "jacobi_iter_27point";
            h2inv = 1.0 / (30.0 * h2);
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
        err |= clSetKernelArg(kernel, ++pos, sizeof(double), &h2inv);
        err |= clSetKernelArg(kernel, ++pos, sizeof(double), &dinv);
        err |= clSetKernelArg(kernel, ++pos, sizeof(double), &problem.omega);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &mgh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ngh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ogh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &problem.ghosts);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &store_res);
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
            if (problem.residual_norm == MGCL_L2)
            {
                // calculate 2-Norm
                double ***rsquares = cuboid_alloc(mgh, ngh, ogh);
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

                // one work-item per cell (including ghost cells). Pad global sizes to fit to local sizes
                size_t global3d[3] = {static_cast<size_t>(mgh), static_cast<size_t>(ngh), static_cast<size_t>(ogh)};
                const size_t local3d[3] = {static_cast<size_t>(mgh > 4 ? 4 : mgh), static_cast<size_t>(ngh > 4 ? 4 : ngh),
                                           static_cast<size_t>(ogh > 4 ? 4 : ogh)};

                for (int i = 0; i < 3; i++)
                    if (global3d[i] % local3d[i] != 0)
                    {
                        // printf("padding global size %d from %ld to ", i, global[i]);
                        global3d[i] += local3d[i] - (global3d[i] % local3d[i]);
                        // printf("%ld (multiple of %ld)\n", global[i], local[i]);
                    }

                err = clEnqueueNDRangeKernel(problem.getOpenCLHelper().getCommands(), kernel_square, 3, NULL, global3d, local3d, 0, NULL, NULL);
                mgclCheckError(err, "Enqueueing residual squared kernel");

                err = clFinish(problem.getOpenCLHelper().getCommands());
                mgclCheckError(err, "Waiting for kernels to finish");

                err = clEnqueueReadBuffer(problem.getOpenCLHelper().getCommands(), dRsquares, CL_TRUE, 0, sizeof(double) * mgh * ngh * ogh,
                                          rsquares[0][0], 0, NULL, NULL);
                mgclCheckError(err, "Error: Failed to read rsquares array from device!");

                // sum up residual squares
                res = 0;
                for (int i = problem.ghosts; i < mgh - problem.ghosts; i++)
                    for (int j = problem.ghosts; j < ngh - problem.ghosts; j++)
                        for (int k = problem.ghosts; k < ogh - problem.ghosts; k++)
                            res += rsquares[i][j][k];
                res = sqrt(res);

                clReleaseMemObject(dRsquares);
                cuboid_free(rsquares, mgh, ngh, ogh);
                clReleaseKernel(kernel_square);
            }
            else
            {
                // calculate Infinity-Norm (on host, TODO do on opencl)
                err = clFinish(problem.getOpenCLHelper().getCommands());
                mgclCheckError(err, "Waiting for kernels to finish");

                double ***rtmp = cuboid_alloc(mgh, ngh, ogh);

                err = clEnqueueReadBuffer(problem.getOpenCLHelper().getCommands(), level.dR, CL_TRUE, 0, sizeof(double) * mgh * ngh * ogh, rtmp[0][0], 0,
                                          NULL, NULL);
                mgclCheckError(err, "Error: Failed to read rsquares array from device!");

                // find maximum residual
                res = 0;
                for (int i = problem.ghosts; i < mgh - problem.ghosts; i++)
                    for (int j = problem.ghosts; j < ngh - problem.ghosts; j++)
                        for (int k = problem.ghosts; k < ogh - problem.ghosts; k++)
                            if (fabs(rtmp[i][j][k]) > res)
                                res = rtmp[i][j][k];

                cuboid_free(rtmp, mgh, ngh, ogh);
            }
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

        if (ngh - 2 * problem.ghosts < wg_size)
            wg_size = ngh - 2 * problem.ghosts;
        if (ogh < ngh)
            wg_size = ogh - 2 * problem.ghosts;
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

            wg_size >> 1;
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
        double h2inv = 1 / h2; // divisor of the stencil, inverted to use * instead of / in kernel

        // Create the compute kernel from the program
        const char *kernel_name;
        if (problem.stencil == MGCL_7POINT)
            kernel_name = "jacobi_stream_shmem_7point";
        else if (problem.stencil == MGCL_19POINT)
        {
            kernel_name = "jacobi_stream_shmem_19point";
            h2inv = 1.0 / (6.0 * h2);
            dinv = (6.0 * h2) / 24.0;
        }
        else if (problem.stencil == MGCL_27POINT)
        {
            kernel_name = "jacobi_stream_shmem_27point";
            h2inv = 1.0 / (30.0 * h2);
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
            if (problem.residual_norm == MGCL_L2)
            {
                // calculate 2-Norm
                double ***rsquares = cuboid_alloc(mgh, ngh, ogh);
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

                // one work-item per cell (including ghost cells). Pad global sizes to fit to local sizes
                size_t global3d[3] = {static_cast<size_t>(mgh), static_cast<size_t>(ngh), static_cast<size_t>(ogh)};
                const size_t local3d[3] = {static_cast<size_t>(mgh > 4 ? 4 : mgh), static_cast<size_t>(ngh > 4 ? 4 : ngh),
                                           static_cast<size_t>(ogh > 4 ? 4 : ogh)};

                for (int i = 0; i < 3; i++)
                    if (global3d[i] % local3d[i] != 0)
                    {
                        // printf("padding global size %d from %ld to ", i, global[i]);
                        global3d[i] += local3d[i] - (global3d[i] % local3d[i]);
                        // printf("%ld (multiple of %ld)\n", global[i], local[i]);
                    }

                err = clEnqueueNDRangeKernel(problem.getOpenCLHelper().getCommands(), kernel_square, 3, NULL, global3d, local3d, 0, NULL, NULL);
                mgclCheckError(err, "Enqueueing residual squared kernel");

                err = clFinish(problem.getOpenCLHelper().getCommands());
                mgclCheckError(err, "Waiting for kernels to finish");

                err = clEnqueueReadBuffer(problem.getOpenCLHelper().getCommands(), dRsquares, CL_TRUE, 0, sizeof(double) * mgh * ngh * ogh,
                                          rsquares[0][0], 0, NULL, NULL);
                mgclCheckError(err, "Error: Failed to read rsquares array from device!");

                // sum up residual squares
                res = 0;
                for (int i = problem.ghosts; i < mgh - problem.ghosts; i++)
                    for (int j = problem.ghosts; j < ngh - problem.ghosts; j++)
                        for (int k = problem.ghosts; k < ogh - problem.ghosts; k++)
                            res += rsquares[i][j][k];
                res = sqrt(res);

                clReleaseMemObject(dRsquares);
                cuboid_free(rsquares, mgh, ngh, ogh);
                clReleaseKernel(kernel_square);
            }
            else
            {
                // calculate Infinity-Norm (on host, TODO do on opencl)
                err = clFinish(problem.getOpenCLHelper().getCommands());
                mgclCheckError(err, "Waiting for kernels to finish");

                double ***rtmp = cuboid_alloc(mgh, ngh, ogh);

                err = clEnqueueReadBuffer(problem.getOpenCLHelper().getCommands(), level.dR, CL_TRUE, 0, sizeof(double) * mgh * ngh * ogh, rtmp[0][0], 0,
                                          NULL, NULL);
                mgclCheckError(err, "Error: Failed to read rsquares array from device!");

                // find maximum residual
                res = 0;
                for (int i = problem.ghosts; i < mgh - problem.ghosts; i++)
                    for (int j = problem.ghosts; j < ngh - problem.ghosts; j++)
                        for (int k = problem.ghosts; k < ogh - problem.ghosts; k++)
                            if (fabs(rtmp[i][j][k]) > res)
                                res = rtmp[i][j][k];

                cuboid_free(rtmp, mgh, ngh, ogh);
            }
        }

        clReleaseKernel(kernel); // TODO maybe clFinish before release?

        return res;
    }

    /* Calculates the residual using OpenCL.
     * Doesn't creates ocl buffers and doesn't copy data from host to device and vice versa
     * v, f and r must be of size [m][n][o] for periodic boundary condition.
     * m, n and o must be the dimensions of ghosted grid.
     * If return_residual is true, the residual's 2-norm or inf-norm will be read back from device and returned, else -1.
     * It's not
     * really performant to do so because we have to wait for all kernels to complete and reading a buffer to host is slow.
     */
    double MultigridEngine::residual(Problem &problem, Level &level, int return_residual)
    {
        int err;
        int mgh = level.mgh;
        int ngh = level.ngh;
        int ogh = level.ogh;
        double res = -1;

        double h2 = (1.0 / (double)level.m) *
                    (1.0 / (double)level.m); // TODO minimum of m,n,o when not cube?
        double h2inv = 1.0 / h2;             // divisor of the stencil, inverted to use * instead of / in kernel

        // Create the compute kernel from the program
        const char *kernel_name;
        if (problem.stencil == MGCL_7POINT)
            kernel_name = "residual_7point";
        else if (problem.stencil == MGCL_19POINT)
        {
            kernel_name = "residual_19point";
        }
        else if (problem.stencil == MGCL_27POINT)
        {
            kernel_name = "residual_27point";
        }

        // Create the compute kernel from the program
        cl_kernel kernel = clCreateKernel(problem.openCLHelper.getProgram(), kernel_name, &err);
        mgclCheckError(err, "Creating kernel");

        // assign kernel arguments
        int pos = 0;
        err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &level.dVIn);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &level.dF);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &level.dR);
        err |= clSetKernelArg(kernel, ++pos, sizeof(double), &h2inv);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &mgh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ngh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ogh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &problem.ghosts);
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

        err = MultigridEngine::updateGhosts(problem, level.dVIn, mgh, ngh, ogh, problem.ghosts, problem.ghosts, problem.ghosts);
        mgclCheckError(err, "Updating ghosts");
        err = clEnqueueNDRangeKernel(problem.getOpenCLHelper().getCommands(), kernel, 3, NULL, global, local, 0, NULL, NULL);
        mgclCheckError(err, "Enqueueing residual kernel");
        err = MultigridEngine::updateGhosts(problem, level.dR, mgh, ngh, ogh, problem.ghosts, problem.ghosts, problem.ghosts);
        mgclCheckError(err, "Updating ghosts of r");

        // calculate residual's 2-norm. Square elements on device and sum up on host
        if (return_residual)
        {
            if (problem.residual_norm == MGCL_L2)
            {
                // calculate 2-Norm
                double ***rsquares = cuboid_alloc(mgh, ngh, ogh);
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

                err = clFinish(problem.getOpenCLHelper().getCommands());
                mgclCheckError(err, "Waiting for kernels to finish");

                err = clEnqueueReadBuffer(problem.getOpenCLHelper().getCommands(), dRsquares, CL_TRUE, 0, sizeof(double) * mgh * ngh * ogh,
                                          rsquares[0][0], 0, NULL, NULL);
                mgclCheckError(err, "Error: Failed to read rsquares array from device!");

                // sum up residual squares
                res = 0;
                for (int i = problem.ghosts; i < mgh - problem.ghosts; i++)
                    for (int j = problem.ghosts; j < ngh - problem.ghosts; j++)
                        for (int k = problem.ghosts; k < ogh - problem.ghosts; k++)
                            res += rsquares[i][j][k];
                res = sqrt(res);

                clReleaseMemObject(dRsquares);
                cuboid_free(rsquares, mgh, ngh, ogh);
                clReleaseKernel(kernel_square);
            }
            else
            {
                // calculate Infinity-Norm (on host, TODO do on opencl)
                err = clFinish(problem.getOpenCLHelper().getCommands());
                mgclCheckError(err, "Waiting for kernels to finish");

                double ***rtmp = cuboid_alloc(mgh, ngh, ogh);

                err = clEnqueueReadBuffer(problem.getOpenCLHelper().getCommands(), level.dR, CL_TRUE, 0, sizeof(double) * mgh * ngh * ogh, rtmp[0][0], 0,
                                          NULL, NULL);
                mgclCheckError(err, "Error: Failed to read rsquares array from device!");

                // find maximum residual
                res = 0;
                for (int i = problem.ghosts; i < mgh - problem.ghosts; i++)
                    for (int j = problem.ghosts; j < ngh - problem.ghosts; j++)
                        for (int k = problem.ghosts; k < ogh - problem.ghosts; k++)
                            if (fabs(rtmp[i][j][k]) > res)
                                res = rtmp[i][j][k];

                cuboid_free(rtmp, mgh, ngh, ogh);
            }
        }

        clReleaseKernel(kernel); // TODO maybe clFinish before release?
        return res;
    }

    /* Tests residual calculation using OpenCL.
     * Creates ocl buffers, copies data from host to device and vice versa
     * v, f and r must be of size [m][n][o] for periodic boundary condition.
     * m, n and o must be the dimensions of grid + 2.
     * If return_residual is true, the residual's 2-norm will be read back from device and returned, else -1. It's not
     * really performant to do so because we have to wait for all kernels to complete and reading a buffer to host is slow.
     */
    double MultigridEngine::residualTest(Problem &problem, double ***v, double ***f, double ***r, int m, int n, int o,
                                         int return_residual)
    {
        // int err;

        // // create device buffers
        // int pointer_flag = problem.getOpenCLHelper().getDeviceType() == CL_DEVICE_TYPE_GPU ? CL_MEM_COPY_HOST_PTR : CL_MEM_USE_HOST_PTR;
        // cl_mem d_v =
        //     clCreateBuffer(problem.getOpenCLHelper().getContext(), CL_MEM_READ_WRITE | pointer_flag, sizeof(double) * m * n * o, v[0][0], &err);
        // cl_mem d_f =
        //     clCreateBuffer(problem.getOpenCLHelper().getContext(), CL_MEM_READ_WRITE | pointer_flag, sizeof(double) * m * n * o, f[0][0], &err);
        // cl_mem dR =
        //     clCreateBuffer(problem.getOpenCLHelper().getContext(), CL_MEM_READ_WRITE | pointer_flag, sizeof(double) * m * n * o, r[0][0], &err);

        // mgcl_level_data data;
        // data.dR = dR;
        // data.dVIn = d_v;
        // data.dF = d_f;
        // data.m = m;
        // data.n = n;
        // data.o = o;

        // double res = mgcl_residual(problem, &data, 1);

        // // read back results
        // err = clEnqueueReadBuffer(problem.getOpenCLHelper().getCommands(), d_v, CL_FALSE, 0, sizeof(double) * m * n * o, v[0][0], 0, NULL, NULL);
        // err = clEnqueueReadBuffer(problem.getOpenCLHelper().getCommands(), d_f, CL_FALSE, 0, sizeof(double) * m * n * o, f[0][0], 0, NULL, NULL);
        // err = clEnqueueReadBuffer(problem.getOpenCLHelper().getCommands(), dR, CL_TRUE, 0, sizeof(double) * m * n * o, r[0][0], 0, NULL, NULL);
        // if (err != CL_SUCCESS)
        // {
        //     printf("Error: Failed to read output arrays from device!\n%s\n", mgcl_err_code(err));
        //     exit(1);
        // }

        // clReleaseMemObject(d_v);
        // clReleaseMemObject(d_f);
        // clReleaseMemObject(dR);
        // return res;
        return 0;
    }

    /* Calculates r = f - A*v using 7-point stencil of 3D laplacian.
     * m,n,o is size of real grid */
    double MultigridEngine::residualSeq(Cuboid &f, Cuboid &v, Cuboid &r, MGCL_RESIDUAL_NORM resnorm,
                                        MGCL_STENCIL stencil)
    {
        double res = 0.0;
        double h2 = 1.0 / ((double)(f.getM() * f.getM()));
        double stencilsum = 0;
        double h2inv = 1.0 / h2;
        if (stencil == MGCL_19POINT)
            h2inv = 1.0 / (6.0 * h2);
        else if (stencil == MGCL_27POINT)
            h2inv = 1.0 / (30.0 * h2);

        for (int i = f.getGhostsM(); i < f.getM() + f.getGhostsM(); i++)
            for (int j = f.getGhostsN(); j < f.getN() + f.getGhostsN(); j++)
                for (int k = f.getGhostsO(); k < f.getO() + f.getGhostsO(); k++)
                {
                    // A*v
                    if (stencil == MGCL_7POINT)
                        stencilsum = (6.0 * v[i][j][k] - v[i][j][k - 1] - v[i][j][k + 1] - v[i][j - 1][k] - v[i][j + 1][k] -
                                      v[i - 1][j][k] - v[i + 1][j][k]) *
                                     h2inv;
                    else if (stencil == MGCL_19POINT)
                        stencilsum =
                            (24.0 * v[i][j][k] - 2.0 * v[i][j][k - 1] - 2.0 * v[i][j][k + 1] - 2.0 * v[i][j - 1][k] -
                             2.0 * v[i][j + 1][k] - 2.0 * v[i - 1][j][k] - 2.0 * v[i + 1][j][k] - v[i][j - 1][k - 1] -
                             v[i][j - 1][k + 1] - v[i][j + 1][k - 1] - v[i][j + 1][k + 1] - v[i - 1][j][k - 1] -
                             v[i - 1][j][k + 1] - v[i + 1][j][k - 1] - v[i + 1][j][k + 1] - v[i - 1][j - 1][k] -
                             v[i - 1][j + 1][k] - v[i + 1][j - 1][k] - v[i + 1][j + 1][k]) *
                            h2inv;
                    else if (stencil == MGCL_27POINT)
                        stencilsum =
                            (128.0 * v[i][j][k] - 14.0 * v[i][j][k - 1] - 14.0 * v[i][j][k + 1] - 14.0 * v[i][j - 1][k] -
                             14.0 * v[i][j + 1][k] - 14.0 * v[i - 1][j][k] - 14.0 * v[i + 1][j][k]

                             - 3.0 * v[i][j - 1][k - 1] - 3.0 * v[i][j - 1][k + 1] - 3.0 * v[i][j + 1][k - 1] -
                             3.0 * v[i][j + 1][k + 1] - 3.0 * v[i - 1][j][k - 1] - 3.0 * v[i - 1][j][k + 1] -
                             3.0 * v[i + 1][j][k - 1] - 3.0 * v[i + 1][j][k + 1] - 3.0 * v[i - 1][j - 1][k] -
                             3.0 * v[i - 1][j + 1][k] - 3.0 * v[i + 1][j - 1][k] - 3.0 * v[i + 1][j + 1][k]

                             - v[i - 1][j - 1][k - 1] - v[i - 1][j - 1][k + 1] - v[i - 1][j + 1][k - 1] -
                             v[i - 1][j + 1][k + 1] - v[i + 1][j - 1][k - 1] - v[i + 1][j - 1][k + 1] -
                             v[i + 1][j + 1][k - 1] - v[i + 1][j + 1][k + 1]) *
                            h2inv;

                    // if (i == 1 && j == 1 && k == 2)
                    //     printf("stencilsum = %e\n", stencilsum);
                    // if (i >= 0 && i <= 6 && j >= 0 && j <= 6 && k >= 0 && k <= 6 && m > 4 && stencil == MGCL_19POINT)
                    //     print_19point(v, i, j, k);

                    // if (i == ghosts && j == ghosts && k == ghosts)
                    // {
                    //     print_7point(v, i, j, k);
                    //     printf("stencil = %d\n", stencil);
                    //     printf("stencil_values[i][j][kst] = %e, stencilsum = %e\n", 6.0 * h2inv, stencilsum);
                    //     printf("stencil_values: %f, %f, %f, %f\n", 6.0 * h2inv, -1.0 * h2inv, -1.0 * h2inv, -1.0 *
                    //     h2inv);
                    // }

                    // r = f - A*v
                    r[i][j][k] = f[i][j][k] - stencilsum;
                    if (resnorm == MGCL_L2)
                        res += r[i][j][k] * r[i][j][k];
                    else if (fabs(r[i][j][k]) > res)
                        res = r[i][j][k];
                }
        MultigridEngine::updateGhostsSeq(r);
        return resnorm == MGCL_L2 ? sqrt(res) : res;
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
}
