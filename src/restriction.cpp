#include "cuboid.hpp"           // for Cuboid
#include "level.hpp"            // for Level
#include "mgcl.hpp"             // for BC
#include "multigrid_engine.hpp" // for Problem, MultigridEngine
#include "opencl_helper.hpp"    // for mgclCheckError, OpenCLHelper
#include "problem.hpp"

#include <cstddef> // for size_t, NULL
#include <iostream>

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
    void MultigridEngine::restrictSeq(Level& fine, Level& coarse, Cuboid& fine_vals, Cuboid& coarse_vals)
    {
        int ghosts = coarse.problem->getGhosts();
        // Shift fine levels instead of using coarse level directly since coarse might have different sizes when
        // using mpi and coarse.num == mpiLevelThreshold.
        int m = fine.m >> 1;
        int n = fine.n >> 1;
        int o = fine.o >> 1;

        if (fine.problem->isPeriodic())
            MultigridEngine::updateGhostsSeq(fine_vals, fine.getMpiDataPtr(), fine.problem->isPeriodic(),
                                             fine.isCalculatedLocally());

        int ioff = 1;
        for (int i = ghosts; i < m + ghosts; i++, ioff++)
        {
            int i2 = i + ioff; // == i*2+ghosts+1
            int joff = 1;
            for (int j = ghosts; j < n + ghosts; j++, joff++)
            {
                int j2 = j + joff;
                int koff = 1;
                for (int k = ghosts; k < o + ghosts; k++, koff++)
                {
                    int k2 = k + koff;
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

        if (coarse.problem->isPeriodic())
            MultigridEngine::updateGhostsSeq(coarse_vals, coarse.getMpiDataPtr(), coarse.problem->isPeriodic(),
                                             coarse.isCalculatedLocally());
    }

    void MultigridEngine::restrict(Level& fine, Level& coarse, cl_mem d_fine_values, cl_mem d_coarse_values)
    {
        int err;
        int mreal = fine.m >> 1;
        int nreal = fine.n >> 1;
        int oreal = fine.o >> 1;
        auto problem = fine.problem;

        // Create the compute kernel from the program
        cl_kernel kernel = clCreateKernel(problem->openCLHelper.getProgram(), "restrict_to_coarse", &err);
        mgclCheckError(err, "Creating kernel");

        // assign kernel arguments
        int pos = 0;
        err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &d_fine_values);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &d_coarse_values);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &coarse.mgh); // TODO check here when using MPI
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

        err = MultigridEngine::updateGhosts(*problem, d_fine_values, fine.mgh, fine.ngh, fine.ogh,
                                            problem->ghosts, problem->ghosts, problem->ghosts, fine.getMpiDataPtr(),
                                            fine.isCalculatedLocally());
        mgclCheckError(err, "Updating fine ghosts");
        err = clEnqueueNDRangeKernel(problem->openCLHelper.getCommands(), kernel, 3, NULL, global, local, 0, NULL, NULL);
        mgclCheckError(err, "Enqueueing restriction kernel");
        err = MultigridEngine::updateGhosts(*problem, d_coarse_values, coarse.mgh, coarse.ngh, coarse.ogh,
                                            problem->ghosts, problem->ghosts, problem->ghosts, coarse.getMpiDataPtr(),
                                            coarse.isCalculatedLocally());
        mgclCheckError(err, "Updating coarse ghosts");

        clReleaseKernel(kernel);
    }
}
