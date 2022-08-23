#include "multigrid_engine.hpp"

namespace mgcl
{

    /* Restricts residual to coarser grid using full-weighted restriction operator.
     * m, n and o must be the dimensions of the coarser grid without ghost cells. */
    void MultigridEngine::restrictSeq(Level &fine, Level &coarse, double ***fine_vals, double ***coarse_vals)
    {
        int ghosts = coarse.problem->getGhosts();
        int m = coarse.m;
        int n = coarse.n;
        int o = coarse.o;

        MultigridEngine::updateGhostsSeq(fine_vals, m * 2, n * 2, o * 2, ghosts, ghosts, ghosts);
        int ioff = 1, joff = 1, koff = 1; // offset grows by 1 for each step
        int i2, j2, k2;
        for (int i = ghosts; i < m + ghosts; i++, ioff++)
        {
            i2 = i + ioff; // == i*2+ghosts+1
            joff = 1;
            for (int j = ghosts; j < n + ghosts; j++, joff++)
            {
                j2 = j + joff;
                koff = 1;
                for (int k = ghosts; k < o + ghosts; k++, koff++)
                {
                    k2 = k + koff;
                    coarse_vals[i][j][k] =
                        0.125 * fine_vals[i2][j2][k2] // self
                        // direct neighbours
                        + 0.0625 * fine_vals[i2 - 1][j2][k2] + 0.0625 * fine_vals[i2 + 1][j2][k2] +
                        0.0625 * fine_vals[i2][j2 - 1][k2] + 0.0625 * fine_vals[i2][j2 + 1][k2] +
                        0.0625 * fine_vals[i2][j2][k2 - 1] +
                        0.0625 * fine_vals[i2][j2][k2 + 1]
                        // edge midpoints xy-plane
                        + 0.03125 * fine_vals[i2 - 1][j2 - 1][k2] + 0.03125 * fine_vals[i2 - 1][j2 + 1][k2] +
                        0.03125 * fine_vals[i2 + 1][j2 - 1][k2] +
                        0.03125 * fine_vals[i2 + 1][j2 + 1][k2]
                        // edge midpoints xz-plane
                        + 0.03125 * fine_vals[i2 - 1][j2][k2 - 1] + 0.03125 * fine_vals[i2 - 1][j2][k2 + 1] +
                        0.03125 * fine_vals[i2 + 1][j2][k2 - 1] +
                        0.03125 * fine_vals[i2 + 1][j2][k2 + 1]
                        // edge midpoints yz-plane
                        + 0.03125 * fine_vals[i2][j2 - 1][k2 - 1] + 0.03125 * fine_vals[i2][j2 - 1][k2 + 1] +
                        0.03125 * fine_vals[i2][j2 + 1][k2 - 1] +
                        0.03125 * fine_vals[i2][j2 + 1][k2 + 1]
                        // corners
                        + 0.015625 * fine_vals[i2 - 1][j2 - 1][k2 - 1] + 0.015625 * fine_vals[i2 - 1][j2 - 1][k2 + 1] +
                        0.015625 * fine_vals[i2 - 1][j2 + 1][k2 - 1] + 0.015625 * fine_vals[i2 - 1][j2 + 1][k2 + 1] +
                        0.015625 * fine_vals[i2 + 1][j2 - 1][k2 - 1] + 0.015625 * fine_vals[i2 + 1][j2 - 1][k2 + 1] +
                        0.015625 * fine_vals[i2 + 1][j2 + 1][k2 - 1] + 0.015625 * fine_vals[i2 + 1][j2 + 1][k2 + 1];
                }
            }
        }
        MultigridEngine::updateGhostsSeq(coarse_vals, m, n, o, ghosts, ghosts, ghosts);
    }

    void MultigridEngine::restrict(Level &fine, Level &coarse)
    {
        int err;
        int mreal = coarse.m - 2 * fine.problem->ghosts;
        int nreal = coarse.n - 2 * fine.problem->ghosts;
        int oreal = coarse.o - 2 * fine.problem->ghosts;
        auto problem = fine.problem;

        // Create the compute kernel from the program
        cl_kernel kernel = clCreateKernel(problem->openCLHelper.getProgram(), "restrict_to_coarse", &err);
        mgclCheckError(err, "Creating kernel");

        // assign kernel arguments
        int pos = 0;
        err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &fine.dR);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &coarse.dF);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &coarse.m);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &coarse.n);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &coarse.o);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &problem->ghosts);
        mgclCheckError(err, "Setting kernel arguments");

        // one work-item per cell (excluding ghost cells). Pad global sizes to fit to local sizes
        size_t global[3] = {static_cast<size_t>(mreal), static_cast<size_t>(nreal), static_cast<size_t>(oreal)};
        const size_t local[3] = {static_cast<size_t>(mreal > 4 ? 4 : mreal), static_cast<size_t>(nreal > 4 ? 4 : nreal),
                                 static_cast<size_t>(oreal > 4 ? 4 : oreal)};

        for (int i = 0; i < 3; i++)
            if (global[i] % local[i] != 0)
            {
                // printf("padding global size %d from %ld to ", i, global[i]);
                global[i] += local[i] - (global[i] % local[i]);
                // printf("%ld (multiple of %ld)\n", global[i], local[i]);
            }

        err = MultigridEngine::updateGhosts(*problem, fine.dR, fine.m, fine.n, fine.o, problem->ghosts, problem->ghosts, problem->ghosts);
        mgclCheckError(err, "Updating fine ghosts");
        err = clEnqueueNDRangeKernel(problem->openCLHelper.getCommands(), kernel, 3, NULL, global, local, 0, NULL, NULL);
        mgclCheckError(err, "Enqueueing restriction kernel");
        err = MultigridEngine::updateGhosts(*problem, coarse.dF, coarse.m, coarse.n, coarse.o, problem->ghosts, problem->ghosts,
                                            problem->ghosts);
        mgclCheckError(err, "Updating coarse ghosts");

        clReleaseKernel(kernel);
    }
}
