#include "cuboid.hpp"           // for Cuboid
#include "level.hpp"            // for Level
#include "multigrid_engine.hpp" // for Problem, MultigridEngine
#include "opencl_helper.hpp"    // for mgclCheckError, OpenCLHelper
#include "problem.hpp"

#include <cstddef> // for size_t, NULL
// #include <iostream>

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

        MultigridEngine::updateGhostsSeq(coarse_vals, coarse.getMpiDataPtr(), coarse.problem->isPeriodic(),
                                         coarse.isCalculatedLocally());
    }

    void MultigridEngine::restrict(Level& fine, Level& coarse, CuboidGpu& d_fine_values, CuboidGpu& d_coarse_values)
    {
        int err;
        int mreal = fine.m >> 1;
        int nreal = fine.n >> 1;
        int oreal = fine.o >> 1;
        auto problem = fine.problem;

        // Create the compute kernel from the program
        const char* kernelName = "restrict_to_coarse";
        cl_kernel kernel = clCreateKernel(problem->openCLHelper.getProgram(), kernelName, &err);
        mgclCheckError(err, "Creating kernel");

        cl_mem buf_fine = d_fine_values.getBuffer();
        cl_mem buf_coarse = d_coarse_values.getBuffer();

        // Shift fine levels instead of using coarse level directly since coarse might have different sizes when
        // using mpi and coarse.num == mpiLevelThreshold.
        int mcgh = mreal + 2 * problem->getGhosts();
        int ncgh = nreal + 2 * problem->getGhosts();
        int ocgh = oreal + 2 * problem->getGhosts();
        int ngh_vals_coarse = d_coarse_values.getNgh();
        int ogh_vals_coarse = d_coarse_values.getOgh();

        // assign kernel arguments
        int pos = 0;
        err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &buf_fine);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &buf_coarse);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &mcgh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ncgh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ocgh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &problem->ghosts);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ngh_vals_coarse);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ogh_vals_coarse);
        mgclCheckError(err, "Setting kernel arguments");

        // one work-item per cell (excluding ghost cells). Pad global sizes to fit to local sizes
        size_t global[3] = {static_cast<size_t>(mreal), static_cast<size_t>(nreal), static_cast<size_t>(oreal)};
        const auto& c = conf::getWorkGroupSizeForKernelAndWiCount(fine.problem->getKernelConfig(), kernelName, 1);
        const size_t local[3] = {static_cast<size_t>(mreal > c[0] ? c[0] : mreal),
                                 static_cast<size_t>(nreal > c[1] ? c[1] : nreal),
                                 static_cast<size_t>(oreal > c[2] ? c[2] : oreal)};

        for (int i = 0; i < 3; i++)
            if (global[i] % local[i] != 0)
                global[i] += local[i] - (global[i] % local[i]);

        cl_event ev;

        err = MultigridEngine::updateGhosts(*problem, fine, d_fine_values, fine.getMpiDataPtr(),
                                            fine.isCalculatedLocally());
        mgclCheckError(err, "Updating fine ghosts");
        err = clEnqueueNDRangeKernel(problem->openCLHelper.getCommands(), kernel, 3, NULL, global, local, 0, NULL, &ev);
        mgclCheckError(err, "Enqueueing restriction kernel");
        // err = MultigridEngine::updateGhosts(*problem, d_coarse_values, coarse.getMpiDataPtr(),
        //                                     coarse.isCalculatedLocally());
        // mgclCheckError(err, "Updating coarse ghosts");

        if (problem->isProfilingEnabled())
        {
            problem->getProfilingData()->addMeasurement(problem->getCommands(), ev, kernelName,
                                                        {global[0], global[1], global[2]},
                                                        {local[0], local[1], local[2]});
        }
        mgclCheckError(clReleaseEvent(ev), "clReleaseEvent");

        clReleaseKernel(kernel);
    }

    /**
     * Restricts residual to coarser grid using user-defined restriction operator defined by a blockstencil.
     **/
    void MultigridEngine::restrictSeqBlockstencil(args::RestrictionBSSeqArgs& args)
    {
        int ghosts = args.fine.getGhostsM();
        // Shift fine levels instead of using coarse level directly since coarse might have different sizes when
        // using mpi and coarse.num == mpiLevelThreshold.
        int m = args.fine.getM() >> 1;
        int n = args.fine.getN() >> 1;
        int o = args.fine.getO() >> 1;

        auto fraw = args.fine.getData();
        auto bsraw = args.rbs.getData();

        args.fine.updateGhosts(args.mpiDataFine, args.updateFineGhostsLocally, args.periodic);

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

                    for (int bi = 0; bi < args.rbs.getBlocksize(); bi++)
                    {
                        double stencilsum = 0;
                        for (int bj = 0; bj < args.rbs.getBlocksize(); bj++)
                        {
                            // clang-format off
                            stencilsum += bsraw[bi][bj][1][1][1] * fraw[bj][i2][j2][k2] // self
                                + bsraw[bi][bj][1][1][0] * fraw[bj][ i2 ][ j2 ][k2-1]
                                + bsraw[bi][bj][1][1][2] * fraw[bj][ i2 ][ j2 ][k2+1]
                                + bsraw[bi][bj][1][0][1] * fraw[bj][ i2 ][j2-1][ k2 ]
                                + bsraw[bi][bj][1][2][1] * fraw[bj][ i2 ][j2+1][ k2 ]
                                + bsraw[bi][bj][0][1][1] * fraw[bj][i2-1][ j2 ][ k2 ]
                                + bsraw[bi][bj][2][1][1] * fraw[bj][i2+1][ j2 ][ k2 ]
                                
                                + bsraw[bi][bj][1][0][0] * fraw[bj][ i2 ][j2-1][k2-1]
                                + bsraw[bi][bj][1][0][2] * fraw[bj][ i2 ][j2-1][k2+1]
                                + bsraw[bi][bj][1][2][0] * fraw[bj][ i2 ][j2+1][k2-1]
                                + bsraw[bi][bj][1][2][2] * fraw[bj][ i2 ][j2+1][k2+1]
                                + bsraw[bi][bj][0][1][0] * fraw[bj][i2-1][ j2 ][k2-1]
                                + bsraw[bi][bj][0][1][2] * fraw[bj][i2-1][ j2 ][k2+1]
                                + bsraw[bi][bj][2][1][0] * fraw[bj][i2+1][ j2 ][k2-1]
                                + bsraw[bi][bj][2][1][2] * fraw[bj][i2+1][ j2 ][k2+1]
                                + bsraw[bi][bj][0][0][1] * fraw[bj][i2-1][j2-1][ k2 ]
                                + bsraw[bi][bj][0][2][1] * fraw[bj][i2-1][j2+1][ k2 ]
                                + bsraw[bi][bj][2][0][1] * fraw[bj][i2+1][j2-1][ k2 ]
                                + bsraw[bi][bj][2][2][1] * fraw[bj][i2+1][j2+1][ k2 ]
                                
                                + bsraw[bi][bj][0][0][0] * fraw[bj][i2-1][j2-1][k2-1]
                                + bsraw[bi][bj][0][0][2] * fraw[bj][i2-1][j2-1][k2+1]
                                + bsraw[bi][bj][0][2][0] * fraw[bj][i2-1][j2+1][k2-1]
                                + bsraw[bi][bj][0][2][2] * fraw[bj][i2-1][j2+1][k2+1]
                                + bsraw[bi][bj][2][0][0] * fraw[bj][i2+1][j2-1][k2-1]
                                + bsraw[bi][bj][2][0][2] * fraw[bj][i2+1][j2-1][k2+1]
                                + bsraw[bi][bj][2][2][0] * fraw[bj][i2+1][j2+1][k2-1]
                                + bsraw[bi][bj][2][2][2] * fraw[bj][i2+1][j2+1][k2+1];
                            // clang-format on
                        }

                        args.coarse[bi][i][j][k] = stencilsum;
                    }
                }
            }
        }

        args.coarse.updateGhosts(args.mpiDataCoarse, args.updateCoarseGhostsLocally, args.periodic);
    }

    void MultigridEngine::restrictBlockstencil(args::RestrictionBSOclArgs& args)
    {
        int err;
        int mreal = args.fine.getM() >> 1;
        int nreal = args.fine.getN() >> 1;
        int oreal = args.fine.getO() >> 1;

        // Create the compute kernel from the program
        const char* kernelName = "restrict_to_coarse_blockstencil";
        cl_kernel kernel = clCreateKernel(args.program, kernelName, &err);
        mgclCheckError(err, "Creating kernel");

        cl_mem buf_fine = args.fine.getBuffer();
        cl_mem buf_coarse = args.coarse.getBuffer();
        cl_mem buf_rbs = args.rbs.getBuf().getBuf();

        // Shift fine levels instead of using coarse level directly since coarse might have different sizes when
        // using mpi and coarse.num == mpiLevelThreshold.
        int mcgh = mreal + 2 * args.coarse.getGhostsM();
        int ncgh = nreal + 2 * args.coarse.getGhostsN();
        int ocgh = oreal + 2 * args.coarse.getGhostsO();
        int ngh_vals_coarse = args.coarse.getNgh();
        int ogh_vals_coarse = args.coarse.getOgh();

        // assign kernel arguments
        int pos = 0;
        int gh = args.coarse.getGhostsM();
        int blocksize = args.coarse.getBlocksize();
        err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &buf_fine);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &buf_coarse);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &buf_rbs);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &mcgh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ncgh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ocgh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &gh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ngh_vals_coarse);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ogh_vals_coarse);
        mgclCheckError(err, "Setting kernel arguments");

        // one work-item per cell (excluding ghost cells). Pad global sizes to fit to local sizes
        size_t global[3] = {static_cast<size_t>(mreal), static_cast<size_t>(nreal), static_cast<size_t>(oreal)};
        size_t local[3] = {4, 4, 4};
        if (args.conf)
        {

            const auto& c = conf::getWorkGroupSizeForKernelAndWiCount(*args.conf, kernelName, 1);
            local[0] = c[0];
            local[1] = c[1];
            local[2] = c[2];
        }

        local[0] = static_cast<size_t>(mreal > local[0] ? local[0] : mreal);
        local[0] = static_cast<size_t>(nreal > local[1] ? local[1] : nreal);
        local[0] = static_cast<size_t>(oreal > local[2] ? local[2] : oreal);

        for (int i = 0; i < 3; i++)
            if (global[i] % local[i] != 0)
                global[i] += local[i] - (global[i] % local[i]);

        cl_event ev;

        args.fine.updateGhostsOclMpi(args.program, args.queue, args.dPlanesBuf, args.sendBuf, args.recvBuf, args.mpiDataFine, args.updateFineGhostsLocally, args.periodic, args.conf, args.pd);

        err = clEnqueueNDRangeKernel(args.queue, kernel, 3, NULL, global, local, 0, NULL, &ev);
        mgclCheckError(err, "Enqueueing restriction kernel");

        args.coarse.updateGhostsOclMpi(args.program, args.queue, args.dPlanesBuf, args.sendBuf, args.recvBuf, args.mpiDataCoarse, args.updateCoarseGhostsLocally, args.periodic, args.conf, args.pd);

        if (args.pd)
        {
            args.pd->addMeasurement(args.queue, ev, kernelName,
                                    {global[0], global[1], global[2]},
                                    {local[0], local[1], local[2]});
        }
        mgclCheckError(clReleaseEvent(ev), "clReleaseEvent");

        clReleaseKernel(kernel);
    }
}
