#include "cuboid.hpp"           // for Cuboid
#include "level.hpp"            // for Level
#include "multigrid_engine.hpp" // for Problem, MultigridEngine
#include "opencl_helper.hpp"    // for mgclCheckError, OpenCLHelper
#include "problem.hpp"

#include <cstddef> // for size_t, NULL

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

namespace mgcl
{
    using std::size_t;

    /* Restricts residual to coarser grid using full-weighted restriction operator.
     * m, n and o must be the dimensions of the coarser grid without ghost cells. */
    void MultigridEngine::restrictSeq(Level &fine, Level &coarse, Cuboid &fine_vals, Cuboid &coarse_vals)
    {
        int ghosts = coarse.problem->getGhosts();
        int m = coarse.m;
        int n = coarse.n;
        int o = coarse.o;

        MultigridEngine::updateGhostsSeq(fine_vals);
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
        MultigridEngine::updateGhostsSeq(coarse_vals);
    }

    void MultigridEngine::restrict(Level &fine, Level &coarse, cl_mem d_fine_values, cl_mem d_coarse_values)
    {
        int err;
        int mreal = coarse.m;
        int nreal = coarse.n;
        int oreal = coarse.o;
        auto problem = fine.problem;

        // Create the compute kernel from the program
        cl_kernel kernel = clCreateKernel(problem->openCLHelper.getProgram(), "restrict_to_coarse", &err);
        mgclCheckError(err, "Creating kernel");

        // assign kernel arguments
        int pos = 0;
        err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &d_fine_values);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &d_coarse_values);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &coarse.mgh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &coarse.ngh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &coarse.ogh);
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

        err = MultigridEngine::updateGhosts(*problem, d_fine_values, fine.mgh, fine.ngh, fine.ogh, problem->ghosts, problem->ghosts, problem->ghosts);
        mgclCheckError(err, "Updating fine ghosts");
        err = clEnqueueNDRangeKernel(problem->openCLHelper.getCommands(), kernel, 3, NULL, global, local, 0, NULL, NULL);
        mgclCheckError(err, "Enqueueing restriction kernel");
        err = MultigridEngine::updateGhosts(*problem, d_coarse_values, coarse.mgh, coarse.ngh, coarse.ogh, problem->ghosts, problem->ghosts,
                                            problem->ghosts);
        mgclCheckError(err, "Updating coarse ghosts");

        clReleaseKernel(kernel);
    }
}
