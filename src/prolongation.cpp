#include "cuboid.hpp"           // for Cuboid
#include "level.hpp"            // for Level
#include "mgcl.hpp"             // for BC
#include "multigrid_engine.hpp" // for Problem, MultigridEngine
#include "opencl_helper.hpp"    // for mgclCheckError, OpenCLHelper
#include "problem.hpp"          // for Problem

#include <cstddef> // for size_t, NULL

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

namespace mgcl
{
    using std::size_t;

    /* Prolongates from coarse to fine grid.
     * m, n and o must be dimensions of the fine grid without ghost cells. */
    void MultigridEngine::prolongateSeq(Level &fine, Level &coarse, Cuboid &fineVals, Cuboid &coarseVals)
    {
        int ghosts = coarse.problem->getGhosts();
        int m = fine.m;
        int n = fine.n;
        int o = fine.o;

        if (fine.problem->isPeriodic())
            MultigridEngine::updateGhostsSeq(coarseVals, coarse.getMpiDataPtr(), fine.problem->isPeriodic(),
                                             !coarse.isBelowMpiLevelThreshold());

        int ioff = 1, joff = 1, koff = 1; // offset grows by 1 for each step
        int i2, j2, k2;
        for (int i = ghosts; i < m / 2 + ghosts; i++, ioff++)
        {
            i2 = i + ioff; // == i*2+ghosts+1
            joff = 1;
            for (int j = ghosts; j < n / 2 + ghosts; j++, joff++)
            {
                j2 = j + joff;
                koff = 1;
                for (int k = ghosts; k < o / 2 + ghosts; k++, koff++)
                {
                    k2 = k + koff;
                    fineVals[i2][j2][k2] = coarseVals[i][j][k];

                    fineVals[i2][j2][k2 - 1] = 0.5 * (coarseVals[i][j][k] + coarseVals[i][j][k - 1]);
                    fineVals[i2][j2 - 1][k2] = 0.5 * (coarseVals[i][j][k] + coarseVals[i][j - 1][k]);
                    fineVals[i2 - 1][j2][k2] = 0.5 * (coarseVals[i][j][k] + coarseVals[i - 1][j][k]);

                    fineVals[i2][j2 - 1][k2 - 1] =
                        0.25 * (coarseVals[i][j][k] + coarseVals[i][j][k - 1] + coarseVals[i][j - 1][k] + coarseVals[i][j - 1][k - 1]);
                    fineVals[i2 - 1][j2][k2 - 1] =
                        0.25 * (coarseVals[i][j][k] + coarseVals[i][j][k - 1] + coarseVals[i - 1][j][k] + coarseVals[i - 1][j][k - 1]);
                    fineVals[i2 - 1][j2 - 1][k2] =
                        0.25 * (coarseVals[i][j][k] + coarseVals[i][j - 1][k] + coarseVals[i - 1][j][k] + coarseVals[i - 1][j - 1][k]);

                    fineVals[i2 - 1][j2 - 1][k2 - 1] =
                        0.125 * (coarseVals[i][j][k] + coarseVals[i][j][k - 1] + coarseVals[i][j - 1][k] + coarseVals[i][j - 1][k - 1] +
                                 coarseVals[i - 1][j][k] + coarseVals[i - 1][j][k - 1] + coarseVals[i - 1][j - 1][k] +
                                 coarseVals[i - 1][j - 1][k - 1]);
                }
            }
        }
    }

    /* Prolongates from coarse to fine grid.
     * Doesn't create buffers or copy memory from or to device. */
    void MultigridEngine::prolongate(Level &fine, Level &coarse, cl_mem d_fine_values, cl_mem d_coarse_values)
    {
        int err;
        auto problem = fine.problem;

        // Create the compute kernel from the program
        cl_kernel kernel = clCreateKernel(problem->openCLHelper.getProgram(), "prolongate_to_fine", &err);
        mgclCheckError(err, "Creating kernel");

        // assign kernel arguments
        int pos = 0;
        err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &d_fine_values);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &d_coarse_values);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &fine.mgh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &fine.ngh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &fine.ogh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &problem->ghosts);
        mgclCheckError(err, "Setting kernel arguments");

        // one work-item per cell (including ghost cells). Pad global sizes to fit to local sizes
        size_t global[3] = {static_cast<size_t>(coarse.mgh), static_cast<size_t>(coarse.ngh), static_cast<size_t>(coarse.ogh)};
        const size_t local[3] = {static_cast<size_t>(coarse.mgh > 4 ? 4 : coarse.mgh),
                                 static_cast<size_t>(coarse.ngh > 4 ? 4 : coarse.ngh),
                                 static_cast<size_t>(coarse.ogh > 4 ? 4 : coarse.ogh)};

        for (int i = 0; i < 3; i++)
            if (global[i] % local[i] != 0)
            {
                // printf("padding global size %d from %ld to ", i, global[i]);
                global[i] += local[i] - (global[i] % local[i]);
                // printf("%ld (multiple of %ld)\n", global[i], local[i]);
            }

        err = MultigridEngine::updateGhosts(*problem, d_coarse_values, coarse.mgh, coarse.ngh, coarse.ogh,
                                            problem->ghosts, problem->ghosts, problem->ghosts, coarse.getMpiDataPtr(),
                                            !coarse.isBelowMpiLevelThreshold());
        mgclCheckError(err, "Updating ghosts coarse");
        err = clEnqueueNDRangeKernel(problem->openCLHelper.getCommands(), kernel, 3, NULL, global, local, 0, NULL, NULL);
        mgclCheckError(err, "Enqueueing kernel");
        // err = MultigridEngine::updateGhosts(conf, fine.d_r, fine.m, fine.n, fine.o, conf->ghosts, conf->ghosts, conf->ghosts);
        // mgclCheckError(err, "Updating ghosts fine");

        clReleaseKernel(kernel);
    }
}