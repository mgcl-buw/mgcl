#include "blockstencil.hpp"
#include "cuboid.hpp" // for Cuboid
#include "cuboid_gpu.hpp"
#include "hypercube.hpp" // for Hypercube6d
#include "level.hpp"     // for Level
#include "mgcl.hpp"      // for mgcl_debug, MGCL_LAPLACE_19POINT
#include "mpi_level_data.hpp"
#include "mpi_util.hpp"
#include "multigrid_engine.hpp" // for Problem, VaryingStencil3x3x3, Multig...
#include "opencl_helper.hpp"
#include "problem.hpp" // for Problem
#include "stencil.hpp" // for mgclCheckError, VaryingStencil3x3x3
#include "util.hpp"

#include <cstddef>
#include <cstdio> // for printf, size_t, NULL
// #include <iostream>
#include <iostream>
#include <math.h> // for fabs, sqrt, ceil
#include <memory>
#include <string>
#include <variant>

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

        if (stencilType == MGCL_FIXED && fixedStencil == nullptr)
        {
            error("stencilType is fixed but fixedStencil is null!");
        }

        for (int iter = 0; iter < maxiter; iter += stepsPerIter)
        {
            // update ghost cells for periodic boundary condition
            MultigridEngine::updateGhostsSeq(v, mpiData, periodic, updateGhostsLocally);
            // TODO else update only neighboring processes if using mpi

            // if (iter == 1)
            // {
            //     f.dumpToFile(std::to_string(mpiData ? mpiData->rank : 0) + "fSeq.txt");
            //     v.dumpToFile(std::to_string(mpiData ? mpiData->rank : 0) + "vSeq.txt");
            //     r.dumpToFile(std::to_string(mpiData ? mpiData->rank : 0) + "rSeq.txt");
            //     if (mpiData)
            //         MPI_Barrier(mpiData->comm);
            //     exit(0);
            // }

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
                else if (stencilType == MGCL_VARYING)
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
                else if (stencilType == MGCL_FIXED)
                {
                    for (int iv = istart_v, ir = istart_r, isv = istart_sv; iv < iend_v; iv++, ir++, isv++)
                        for (int jv = jstart_v, jr = jstart_r, jsv = jstart_sv; jv < jend_v; jv++, jr++, jsv++)
                            for (int kv = kstart_v, kr = kstart_r, ksv = kstart_sv; kv < kend_v; kv++, kr++, ksv++)
                            {

                                vraw[iv][jv][kv] = vraw[iv][jv][kv] + omega * (1.0 / (*fixedStencil)[1][1][1]) * r[ir][jr][kr];

                                // if (iv >= 1 && iv <= 2 && jv >= 1 && jv <= 2 && kv >= 1 && kv <= 2)
                                // {
                                //     // print27point(v, iv, jv, kv, *fixedStencil);
                                //     std::cout << "(1.0 / (*fixedStencil)[1][1][1]) * r[ir][jr][kr] = (" << 1.0 / (*fixedStencil)[1][1][1] << ") * " << r[ir][jr][kr] << " = " << (1.0 / (*fixedStencil)[1][1][1]) * r[ir][jr][kr] << std::endl;
                                // }
                            }
                }
            }
        }

        MultigridEngine::updateGhostsSeq(v, mpiData, periodic, updateGhostsLocally);

        if (returnResidualNorm)
            res = residualSeq(f, v, r, resnorm, stencilType, stencilFactor, stencilValues, fixedStencil,
                              returnResidualNorm, periodic,
                              updateGhostsLocally, 0, 0, 0, mpiData);

        return res;
    }

    double MultigridEngine::jacobiSeq(args::JacobiBSSeqArgs& args)
    {
        double res = 0.0;
        auto vraw = args.v.getData();
        int stepsPerIter = args.stepsPerIter;

        // decrease stepsPerIter if it's less than maxIter
        if (args.maxiter < stepsPerIter)
            stepsPerIter = args.maxiter;

        // Ghosts only need to be updated in the periodic case, so set stepsPerIter = 1 for non-periodic.
        // TODO adjust for MPI
        if (!args.periodic)
            stepsPerIter = 1;

        // Check if amount of ghost cells is large enough
        if (util::seq::min3(args.v.getGhostsM(), args.v.getGhostsN(), args.v.getGhostsO()) < stepsPerIter)
        {
            error("#ghosts of v must be >= stepsPerIter!");
        }

        if (util::seq::min3(args.r.getGhostsM(), args.r.getGhostsN(), args.r.getGhostsO()) < stepsPerIter - 1)
        {
            error("#ghosts of r must be >= stepsPerIter - 1!");
        }

        if (util::seq::min3(args.f.getGhostsM(), args.f.getGhostsN(), args.f.getGhostsO()) < stepsPerIter - 1)
        {
            error("#ghosts of f must be >= stepsPerIter - 1!");
        }

        if (!std::holds_alternative<std::shared_ptr<CuboidBS>>(args.bs_inv) && !std::holds_alternative<std::shared_ptr<Blockstencil>>(args.bs_inv))
        {
            error("bs_inv must be a shared_ptr to either CuboidBS or Blockstencil!");
        }

        if (auto bs_inv_ptr = std::get_if<std::shared_ptr<Blockstencil>>(&args.bs_inv))
        {
            auto& bs_inv = *bs_inv_ptr->get();
            if (bs_inv.getWidth() != 1)
            {
                error("width of bs_inv must be 1!");
            }
        }

        for (int iter = 0; iter < args.maxiter; iter += stepsPerIter)
        {
            // update ghost cells for periodic boundary condition
            args.v.updateGhosts(args.mpiData, args.updateGhostsLocally, args.periodic);
            // TODO else update only neighboring processes if using mpi

            // if stepsPerIter > 1, multiple iterations can be done without updating ghosts in-between
            for (int innerIter = 0; innerIter < stepsPerIter && iter + innerIter < args.maxiter; innerIter++)
            {
                // damped/weighted iteration formula: u_(m+1) = u_(m) + omega * D^-1 * r_(m)

                int off = (stepsPerIter - innerIter) - 1;
                int istart_v = args.v.getGhostsM() - off;
                int jstart_v = args.v.getGhostsN() - off;
                int kstart_v = args.v.getGhostsO() - off;
                int iend_v = args.v.getMgh() - istart_v;
                int jend_v = args.v.getNgh() - jstart_v;
                int kend_v = args.v.getOgh() - kstart_v;
                int istart_r = args.r.getGhostsM() - off;
                int jstart_r = args.r.getGhostsN() - off;
                int kstart_r = args.r.getGhostsO() - off;
                int istart_sv = args.bs.getGhostsM() - off;
                int jstart_sv = args.bs.getGhostsN() - off;
                int kstart_sv = args.bs.getGhostsO() - off;

                // r = f - A*v
                // TODO move creation of args outside of loop?
                args::ResidualBSSeqArgs residualArgs{
                    args.f,
                    args.v,
                    args.r,
                    args.resnorm,
                    args.bs,
                    args.returnResidualNorm,
                    args.periodic,
                    args.updateGhostsLocally,
                    off, off, off,
                    args.mpiData};
                res = residualSeq(residualArgs);

                // smoother type is Jacobi_Block: bs_inv is a matrix
                if (auto bs_inv_ptr = std::get_if<std::shared_ptr<Blockstencil>>(&args.bs_inv))
                {
                    auto& bs_inv = *bs_inv_ptr->get();
                    for (int iv = istart_v, ir = istart_r, isv = istart_sv; iv < iend_v; iv++, ir++, isv++)
                        for (int jv = jstart_v, jr = jstart_r, jsv = jstart_sv; jv < jend_v; jv++, jr++, jsv++)
                            for (int kv = kstart_v, kr = kstart_r, ksv = kstart_sv; kv < kend_v; kv++, kr++, ksv++)
                            {
                                for (int bi = 0; bi < args.v.getBlocksize(); bi++)
                                {
                                    double sum = 0;
                                    // calculate bs_inv * r first
                                    for (int bj = 0; bj < args.v.getBlocksize(); bj++)
                                    {
                                        sum += bs_inv[bi][bj][0][0][0][isv][jsv][ksv] * args.r[bj][ir][jr][kr];

                                        // if (iv == 1 && jv == 1 && kv == 1)
                                        // {
                                        //     // print27point(v, iv, jv, kv, *fixedStencil);
                                        //     std::cout << "bs_inv * r = " << args.bs_inv[bi][bj][0][0][0][isv][jsv][ksv] << " * " << args.r[ir][jr][kr][bj] << " = " << args.bs_inv[bi][bj][0][0][0][isv][jsv][ksv] * args.r[ir][jr][kr][bj] << std::endl;
                                        // }
                                    }

                                    // if (iv == 1 && jv == 1 && kv == 1)
                                    // {
                                    //     // print27point(v, iv, jv, kv, *fixedStencil);
                                    //     std::cout << "sum = " << sum << std::endl;
                                    // }

                                    // update v, i.e. v_{i+1} = v_i + omega * bs_inv * r
                                    vraw[bi][iv][jv][kv] = vraw[bi][iv][jv][kv] + args.omega * sum;
                                    // vraw[iv][jv][kv][bi] = vraw[iv][jv][kv][bi] + args.omega * args.bs_inv[bi][bi][0][0][0][isv][jsv][ksv] * args.r[ir][jr][kr][bi];
                                }
                            }
                }
                else
                {
                    // smoother type is Jacobi_Scalar: bs_inv is scalar
                    auto& bs_inv = *std::get_if<std::shared_ptr<CuboidBS>>(&args.bs_inv)->get();
                    for (int iv = istart_v, ir = istart_r, isv = istart_sv; iv < iend_v; iv++, ir++, isv++)
                        for (int jv = jstart_v, jr = jstart_r, jsv = jstart_sv; jv < jend_v; jv++, jr++, jsv++)
                            for (int kv = kstart_v, kr = kstart_r, ksv = kstart_sv; kv < kend_v; kv++, kr++, ksv++)
                                for (int bi = 0; bi < args.v.getBlocksize(); bi++)
                                {
                                    // update v, i.e. v_{i+1} = v_i + omega * bs_inv * r
                                    vraw[bi][iv][jv][kv] = vraw[bi][iv][jv][kv] + args.omega * bs_inv[bi][isv][jsv][ksv] * args.r[bi][ir][jr][kr];

                                    // if (iv == 1 && jv == 1 && kv == 1)
                                    // {
                                    //     std::cout << "bs_inv * r[ir][jr][kr] = " << bs_inv[isv][jsv][ksv][bi] << " * " << args.r[ir][jr][kr][bi] << " = " << bs_inv[isv][jsv][ksv][bi] * args.r[ir][jr][kr][bi] << std::endl;
                                    // }
                                }
                }
            }
        }

        args.v.updateGhosts(args.mpiData, args.updateGhostsLocally, args.periodic);

        if (args.returnResidualNorm)
        {
            args::ResidualBSSeqArgs residualArgs{
                args.f,
                args.v,
                args.r,
                args.resnorm,
                args.bs,
                args.returnResidualNorm,
                args.periodic,
                args.updateGhostsLocally,
                0, 0, 0,
                args.mpiData};
            res = residualSeq(residualArgs);
        }

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

        // double h2 = 1.0 / static_cast<double>((problem.getMGlobal() >> level.num) * (problem.getMGlobal() >> level.num));
        double h2 = level.getH() * level.getH();
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
        else if (problem.stencilType == MGCL_FIXED)
        {
            kernelName = "jacobi_iter_27point_fixed_stencil_1d";
        }

        // Use two kernel objects for alternating iterations to avoid race conditions with dvin and dvout
        cl_kernel kernel1 = clCreateKernel(problem.openCLHelper.getProgram(), kernelName, &err);
        mgclCheckError(err, "Creating kernel1");
        cl_kernel kernel2 = clCreateKernel(problem.openCLHelper.getProgram(), kernelName, &err);
        mgclCheckError(err, "Creating kernel2");

        cl_mem dVIn = level.getDVIn().getBuffer();
        cl_mem dVOut = level.getDVOut().getBuffer();
        cl_mem dF = level.getDF().getBuffer();
        cl_mem dR = level.getDR().getBuffer();

        // assign kernel arguments
        int pos = 0;
        int pos_idxstart = -1;
        if (problem.stencilType == MGCL_VARYING)
        {
            auto svbuf = level.stencilValuesGpu->getBuf();
            int svgh = level.stencilValuesGpu->getGh();
            int svmgh = level.stencilValuesGpu->getMgh();
            int svngh = level.stencilValuesGpu->getNgh();
            int svogh = level.stencilValuesGpu->getOgh();
            int svGridSize = svmgh * svngh * svogh;
            err = clSetKernelArg(kernel1, pos, sizeof(cl_mem), &dVIn);
            err |= clSetKernelArg(kernel1, ++pos, sizeof(cl_mem), &dVOut);
            err |= clSetKernelArg(kernel1, ++pos, sizeof(cl_mem), &dF);
            err |= clSetKernelArg(kernel1, ++pos, sizeof(cl_mem), &dR);
            err |= clSetKernelArg(kernel1, ++pos, sizeof(cl_mem), &svbuf);
            err |= clSetKernelArg(kernel1, ++pos, sizeof(double), &problem.omega);
            err |= clSetKernelArg(kernel1, ++pos, sizeof(int), &mgh);
            err |= clSetKernelArg(kernel1, ++pos, sizeof(int), &ngh);
            err |= clSetKernelArg(kernel1, ++pos, sizeof(int), &ogh);
            err |= clSetKernelArg(kernel1, ++pos, sizeof(int), &svmgh);
            err |= clSetKernelArg(kernel1, ++pos, sizeof(int), &svngh);
            err |= clSetKernelArg(kernel1, ++pos, sizeof(int), &svogh);
            err |= clSetKernelArg(kernel1, ++pos, sizeof(int), &problem.ghosts);
            err |= clSetKernelArg(kernel1, ++pos, sizeof(int), &svgh);
            err |= clSetKernelArg(kernel1, ++pos, sizeof(int), &svGridSize);
            err |= clSetKernelArg(kernel1, ++pos, sizeof(int), &idx_start);
            pos_idxstart = pos;
            err |= clSetKernelArg(kernel1, ++pos, sizeof(int), &store_res);
        }
        else if (problem.stencilType == MGCL_FIXED)
        {
            auto& fs = *level.getFixedStencil();
            err = clSetKernelArg(kernel1, pos, sizeof(cl_mem), &dVIn);
            err |= clSetKernelArg(kernel1, ++pos, sizeof(cl_mem), &dVOut);
            err |= clSetKernelArg(kernel1, ++pos, sizeof(cl_mem), &dF);
            err |= clSetKernelArg(kernel1, ++pos, sizeof(cl_mem), &dR);
            err |= clSetKernelArg(kernel1, ++pos, sizeof(double), &problem.omega);
            err |= clSetKernelArg(kernel1, ++pos, sizeof(int), &mgh);
            err |= clSetKernelArg(kernel1, ++pos, sizeof(int), &ngh);
            err |= clSetKernelArg(kernel1, ++pos, sizeof(int), &ogh);
            err |= clSetKernelArg(kernel1, ++pos, sizeof(int), &problem.ghosts);
            err |= clSetKernelArg(kernel1, ++pos, sizeof(int), &idx_start);
            pos_idxstart = pos;
            err |= clSetKernelArg(kernel1, ++pos, sizeof(int), &store_res);
            err |= clSetKernelArg(kernel1, ++pos, sizeof(double), &fs[0][0][0]);
            err |= clSetKernelArg(kernel1, ++pos, sizeof(double), &fs[0][0][1]);
            err |= clSetKernelArg(kernel1, ++pos, sizeof(double), &fs[0][0][2]);
            err |= clSetKernelArg(kernel1, ++pos, sizeof(double), &fs[0][1][0]);
            err |= clSetKernelArg(kernel1, ++pos, sizeof(double), &fs[0][1][1]);
            err |= clSetKernelArg(kernel1, ++pos, sizeof(double), &fs[0][1][2]);
            err |= clSetKernelArg(kernel1, ++pos, sizeof(double), &fs[0][2][0]);
            err |= clSetKernelArg(kernel1, ++pos, sizeof(double), &fs[0][2][1]);
            err |= clSetKernelArg(kernel1, ++pos, sizeof(double), &fs[0][2][2]);
            err |= clSetKernelArg(kernel1, ++pos, sizeof(double), &fs[1][0][0]);
            err |= clSetKernelArg(kernel1, ++pos, sizeof(double), &fs[1][0][1]);
            err |= clSetKernelArg(kernel1, ++pos, sizeof(double), &fs[1][0][2]);
            err |= clSetKernelArg(kernel1, ++pos, sizeof(double), &fs[1][1][0]);
            err |= clSetKernelArg(kernel1, ++pos, sizeof(double), &fs[1][1][1]);
            err |= clSetKernelArg(kernel1, ++pos, sizeof(double), &fs[1][1][2]);
            err |= clSetKernelArg(kernel1, ++pos, sizeof(double), &fs[1][2][0]);
            err |= clSetKernelArg(kernel1, ++pos, sizeof(double), &fs[1][2][1]);
            err |= clSetKernelArg(kernel1, ++pos, sizeof(double), &fs[1][2][2]);
            err |= clSetKernelArg(kernel1, ++pos, sizeof(double), &fs[2][0][0]);
            err |= clSetKernelArg(kernel1, ++pos, sizeof(double), &fs[2][0][1]);
            err |= clSetKernelArg(kernel1, ++pos, sizeof(double), &fs[2][0][2]);
            err |= clSetKernelArg(kernel1, ++pos, sizeof(double), &fs[2][1][0]);
            err |= clSetKernelArg(kernel1, ++pos, sizeof(double), &fs[2][1][1]);
            err |= clSetKernelArg(kernel1, ++pos, sizeof(double), &fs[2][1][2]);
            err |= clSetKernelArg(kernel1, ++pos, sizeof(double), &fs[2][2][0]);
            err |= clSetKernelArg(kernel1, ++pos, sizeof(double), &fs[2][2][1]);
            err |= clSetKernelArg(kernel1, ++pos, sizeof(double), &fs[2][2][2]);
        }
        else
        {
            err = clSetKernelArg(kernel1, pos, sizeof(cl_mem), &dVIn);
            err |= clSetKernelArg(kernel1, ++pos, sizeof(cl_mem), &dVOut);
            err |= clSetKernelArg(kernel1, ++pos, sizeof(cl_mem), &dF);
            err |= clSetKernelArg(kernel1, ++pos, sizeof(cl_mem), &dR);
            err |= clSetKernelArg(kernel1, ++pos, sizeof(double), &h2inv);
            err |= clSetKernelArg(kernel1, ++pos, sizeof(double), &dinv);
            err |= clSetKernelArg(kernel1, ++pos, sizeof(double), &problem.omega);
            err |= clSetKernelArg(kernel1, ++pos, sizeof(int), &mgh);
            err |= clSetKernelArg(kernel1, ++pos, sizeof(int), &ngh);
            err |= clSetKernelArg(kernel1, ++pos, sizeof(int), &ogh);
            err |= clSetKernelArg(kernel1, ++pos, sizeof(int), &problem.ghosts);
            err |= clSetKernelArg(kernel1, ++pos, sizeof(int), &idx_start);
            pos_idxstart = pos;
            err |= clSetKernelArg(kernel1, ++pos, sizeof(int), &store_res);
        }
        mgclCheckError(err, "Setting kernel arguments");

        // assign kernel arguments
        pos = 0;
        pos_idxstart = -1;
        if (problem.stencilType == MGCL_VARYING)
        {
            auto svbuf = level.stencilValuesGpu->getBuf();
            int svgh = level.stencilValuesGpu->getGh();
            int svmgh = level.stencilValuesGpu->getMgh();
            int svngh = level.stencilValuesGpu->getNgh();
            int svogh = level.stencilValuesGpu->getOgh();
            int svGridSize = svmgh * svngh * svogh;
            err = clSetKernelArg(kernel2, pos, sizeof(cl_mem), &dVOut);
            err |= clSetKernelArg(kernel2, ++pos, sizeof(cl_mem), &dVIn);
            err |= clSetKernelArg(kernel2, ++pos, sizeof(cl_mem), &dF);
            err |= clSetKernelArg(kernel2, ++pos, sizeof(cl_mem), &dR);
            err |= clSetKernelArg(kernel2, ++pos, sizeof(cl_mem), &svbuf);
            err |= clSetKernelArg(kernel2, ++pos, sizeof(double), &problem.omega);
            err |= clSetKernelArg(kernel2, ++pos, sizeof(int), &mgh);
            err |= clSetKernelArg(kernel2, ++pos, sizeof(int), &ngh);
            err |= clSetKernelArg(kernel2, ++pos, sizeof(int), &ogh);
            err |= clSetKernelArg(kernel2, ++pos, sizeof(int), &svmgh);
            err |= clSetKernelArg(kernel2, ++pos, sizeof(int), &svngh);
            err |= clSetKernelArg(kernel2, ++pos, sizeof(int), &svogh);
            err |= clSetKernelArg(kernel2, ++pos, sizeof(int), &problem.ghosts);
            err |= clSetKernelArg(kernel2, ++pos, sizeof(int), &svgh);
            err |= clSetKernelArg(kernel2, ++pos, sizeof(int), &svGridSize);
            err |= clSetKernelArg(kernel2, ++pos, sizeof(int), &idx_start);
            pos_idxstart = pos;
            err |= clSetKernelArg(kernel2, ++pos, sizeof(int), &store_res);
        }
        else if (problem.stencilType == MGCL_FIXED)
        {
            auto& fs = *level.getFixedStencil();
            err = clSetKernelArg(kernel2, pos, sizeof(cl_mem), &dVOut);
            err |= clSetKernelArg(kernel2, ++pos, sizeof(cl_mem), &dVIn);
            err |= clSetKernelArg(kernel2, ++pos, sizeof(cl_mem), &dF);
            err |= clSetKernelArg(kernel2, ++pos, sizeof(cl_mem), &dR);
            err |= clSetKernelArg(kernel2, ++pos, sizeof(double), &problem.omega);
            err |= clSetKernelArg(kernel2, ++pos, sizeof(int), &mgh);
            err |= clSetKernelArg(kernel2, ++pos, sizeof(int), &ngh);
            err |= clSetKernelArg(kernel2, ++pos, sizeof(int), &ogh);
            err |= clSetKernelArg(kernel2, ++pos, sizeof(int), &problem.ghosts);
            err |= clSetKernelArg(kernel2, ++pos, sizeof(int), &idx_start);
            pos_idxstart = pos;
            err |= clSetKernelArg(kernel2, ++pos, sizeof(int), &store_res);
            err |= clSetKernelArg(kernel2, ++pos, sizeof(double), &fs[0][0][0]);
            err |= clSetKernelArg(kernel2, ++pos, sizeof(double), &fs[0][0][1]);
            err |= clSetKernelArg(kernel2, ++pos, sizeof(double), &fs[0][0][2]);
            err |= clSetKernelArg(kernel2, ++pos, sizeof(double), &fs[0][1][0]);
            err |= clSetKernelArg(kernel2, ++pos, sizeof(double), &fs[0][1][1]);
            err |= clSetKernelArg(kernel2, ++pos, sizeof(double), &fs[0][1][2]);
            err |= clSetKernelArg(kernel2, ++pos, sizeof(double), &fs[0][2][0]);
            err |= clSetKernelArg(kernel2, ++pos, sizeof(double), &fs[0][2][1]);
            err |= clSetKernelArg(kernel2, ++pos, sizeof(double), &fs[0][2][2]);
            err |= clSetKernelArg(kernel2, ++pos, sizeof(double), &fs[1][0][0]);
            err |= clSetKernelArg(kernel2, ++pos, sizeof(double), &fs[1][0][1]);
            err |= clSetKernelArg(kernel2, ++pos, sizeof(double), &fs[1][0][2]);
            err |= clSetKernelArg(kernel2, ++pos, sizeof(double), &fs[1][1][0]);
            err |= clSetKernelArg(kernel2, ++pos, sizeof(double), &fs[1][1][1]);
            err |= clSetKernelArg(kernel2, ++pos, sizeof(double), &fs[1][1][2]);
            err |= clSetKernelArg(kernel2, ++pos, sizeof(double), &fs[1][2][0]);
            err |= clSetKernelArg(kernel2, ++pos, sizeof(double), &fs[1][2][1]);
            err |= clSetKernelArg(kernel2, ++pos, sizeof(double), &fs[1][2][2]);
            err |= clSetKernelArg(kernel2, ++pos, sizeof(double), &fs[2][0][0]);
            err |= clSetKernelArg(kernel2, ++pos, sizeof(double), &fs[2][0][1]);
            err |= clSetKernelArg(kernel2, ++pos, sizeof(double), &fs[2][0][2]);
            err |= clSetKernelArg(kernel2, ++pos, sizeof(double), &fs[2][1][0]);
            err |= clSetKernelArg(kernel2, ++pos, sizeof(double), &fs[2][1][1]);
            err |= clSetKernelArg(kernel2, ++pos, sizeof(double), &fs[2][1][2]);
            err |= clSetKernelArg(kernel2, ++pos, sizeof(double), &fs[2][2][0]);
            err |= clSetKernelArg(kernel2, ++pos, sizeof(double), &fs[2][2][1]);
            err |= clSetKernelArg(kernel2, ++pos, sizeof(double), &fs[2][2][2]);
        }
        else
        {
            err = clSetKernelArg(kernel2, pos, sizeof(cl_mem), &dVOut);
            err |= clSetKernelArg(kernel2, ++pos, sizeof(cl_mem), &dVIn);
            err |= clSetKernelArg(kernel2, ++pos, sizeof(cl_mem), &dF);
            err |= clSetKernelArg(kernel2, ++pos, sizeof(cl_mem), &dR);
            err |= clSetKernelArg(kernel2, ++pos, sizeof(double), &h2inv);
            err |= clSetKernelArg(kernel2, ++pos, sizeof(double), &dinv);
            err |= clSetKernelArg(kernel2, ++pos, sizeof(double), &problem.omega);
            err |= clSetKernelArg(kernel2, ++pos, sizeof(int), &mgh);
            err |= clSetKernelArg(kernel2, ++pos, sizeof(int), &ngh);
            err |= clSetKernelArg(kernel2, ++pos, sizeof(int), &ogh);
            err |= clSetKernelArg(kernel2, ++pos, sizeof(int), &problem.ghosts);
            err |= clSetKernelArg(kernel2, ++pos, sizeof(int), &idx_start);
            pos_idxstart = pos;
            err |= clSetKernelArg(kernel2, ++pos, sizeof(int), &store_res);
        }
        mgclCheckError(err, "Setting kernel arguments");

        // One work-item per cell (including ghost cells).
        size_t global[2] = {static_cast<size_t>(mgh * ngh * ogh), static_cast<size_t>(0)};
        const auto& c = conf::getWorkGroupSizeForKernelAndWiCount(problem.getKernelConfig(), kernelName, 1);
        size_t local[2] = {c[0], c[1]};

        // kernels that use constant Laplace stencils are 2d and need different global and local sizes
        if (problem.stencilType != MGCL_VARYING && problem.stencilType != MGCL_FIXED)
        {
            global[0] = static_cast<size_t>(ngh);
            global[1] = static_cast<size_t>(ogh);
            // local[0] = static_cast<size_t>(1);
            // local[1] = static_cast<size_t>(64);
        }

        // Pad global sizes to fit to local sizes
        int kernelDims = (problem.stencilType == MGCL_VARYING || problem.stencilType == MGCL_FIXED) ? 1 : 2;
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

            // if (globalIter == 1)
            // {
            //     level.getDF().dumpToFile(problem.getCommands(), std::to_string(level.getMpiDataPtr() ? level.getMpiData().rank : 0) + "fOcl.txt");
            //     level.getDVOut().dumpToFile(problem.getCommands(), std::to_string(level.getMpiDataPtr() ? level.getMpiData().rank : 0) + "vOcl.txt");
            //     level.getDR().dumpToFile(problem.getCommands(), std::to_string(level.getMpiDataPtr() ? level.getMpiData().rank : 0) + "rOcl.txt");
            //     if (level.getMpiDataPtr())
            //         MPI_Barrier(level.getMpiDataPtr()->comm);
            //     exit(0);
            // }

            // if stepsPerIter > 1, multiple iterations can be done without updating ghosts in-between
            for (int innerIter = 0; innerIter < stepsPerIter && globalIter < maxiter; innerIter++, globalIter++)
            {
                // damped/weighted iteration formula: u_(m+1) = u_(m) + omega * D^-1 * r_(m)

                // recalculate and set idx_start
                idx_start = problem.ghosts - (std::min((stepsPerIter - innerIter), maxiter - globalIter) - 1);
                err = clSetKernelArg(kernel1, pos_idxstart, sizeof(int), &idx_start);
                err |= clSetKernelArg(kernel2, pos_idxstart, sizeof(int), &idx_start);
                mgclCheckError(err, "Setting kernel arguments");

                err = clEnqueueNDRangeKernel(problem.getOpenCLHelper().getCommands(), globalIter % 2 == 0 ? kernel1 : kernel2, kernelDims, NULL, global, local, 0, NULL, &ev);
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

        // copy result into dVIn if needed
        if (maxiter % 2 == 1)
        {
            // level.getDVOut().copyTo(problem.getOpenCLHelper().getCommands(), level.getDVIn());
            // problem.finish();
            CuboidGpu::swap(level.getDVIn(), level.getDVOut());
        }

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

        clReleaseKernel(kernel1);
        clReleaseKernel(kernel2);

        return res;
    }

    // Starts the kernel for one iteration of Jacobi for boundary gps.
    // No ghost update included. No final residual calculation included.
    double MultigridEngine::jacobi_overlapped_helpers::jacobiBoundary(mgcl::Problem& problem, mgcl::Level& level,
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
    double MultigridEngine::jacobi_overlapped_helpers::jacobiInner(mgcl::Problem& problem, mgcl::Level& level,
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

    /**
     * @brief Runs Jacobi inner parallel to start of data transfer for ghost update.
     */
    void MultigridEngine::jacobi_overlapped_helpers::jacobiOverlappedCore(mgcl::Problem& problem, mgcl::Level& level,
                                                                          cl_mem dVInBuf, CuboidGpu& dVOut, int store_res,
                                                                          cl_command_queue queue2,
                                                                          cl_kernel innerKernel, std::string kernelNameInner,
                                                                          size_t globalInner[3], size_t localInner[3])
    {
        if (problem.getDPlanesBufPtr() == nullptr)
            error("MultigridEngine::updateGhostsOclMpi: dPlanesBufPtr is null");

        // Use temporary buffer for extracting and pasting planes. Check if it's large enough beforehand.
        // TODO maybe disable check in UNSAFE mode
        int yz = dVOut.getNgh() * dVOut.getOgh();
        int xz = dVOut.getMgh() * dVOut.getOgh();
        int xy = dVOut.getMgh() * dVOut.getNgh();
        int ressize = 2 * yz * dVOut.getGhostsM() + 2 * xz * dVOut.getGhostsN() + 2 * xy * dVOut.getGhostsO();

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
        // dVOut.extractBorderPlanes(problem.getCommands(), problem.getProgram(),
        //                           dPlanesBuf, hPlanesBufSend,
        //                           &problem.getKernelConfig(), problem.getProfilingData());
        dVOut.extractBorderPlanes(problem.getCommands(), problem.getProgram(),
                                  dPlanesBuf, nullptr,
                                  &problem.getKernelConfig(), problem.getProfilingData(), false);
        auto& sbuf = *hPlanesBufSend;
        auto& rbuf = *hPlanesBufRecv;

        // if we wait here, inner Jacobi and memcpy will be executed in parallel
        problem.finish();

        jacobi_overlapped_helpers::jacobiInner(problem, level, dVInBuf, dVOut.getBuffer(), store_res, queue2, innerKernel, globalInner, localInner, kernelNameInner);

        dPlanesBuf->read(problem.getCommands(), hPlanesBufSend->data(), true, ressize, nullptr);

        // Send our planes to neighbours and receive their planes
        mgcl::mpi_util::sendBorderPlanes(dVOut.getMgh(), dVOut.getNgh(), dVOut.getOgh(),
                                         dVOut.getGhostsM(), dVOut.getGhostsN(), dVOut.getGhostsO(), 1,
                                         sbuf, rbuf, *level.getMpiDataPtr());

        // Paste planes back into the buffer.
        dPlanesBuf->write(problem.getCommands(), rbuf, false, ressize, problem.getProfilingData());
        dVOut.pasteGhostsFromBorderPlanes(problem.getContext(), problem.getCommands(), problem.getProgram(),
                                          dPlanesBuf, nullptr,
                                          &problem.getKernelConfig(), problem.getProfilingData());
    }

    /* Runs jacobi method using OpenCL.
     * Doesn't creates ocl buffers and doesn't copy data from host to device and vice versa
     * v, f and r must be of size [m][n][o] for periodic boundary condition. Ghosts of v and f must be updated.
     * m, n and o must be the dimensions of grid + 2*ghosts
     * If return_residual is true, the residual's 2-norm or inf-norm will be read back from device and returned, else -1.
     * queue2 must be different from problem.getCommands().
     */
    double MultigridEngine::jacobiOverlapped(mgcl::Problem& problem, mgcl::Level& level, int maxiter, bool return_residual, int stepsPerIter,
                                             cl_command_queue queue2)
    {
        assert(problem.getCommands() != queue2 && "queue2 must not be the same as problem.getCommands()");

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

        /********************** create boundary kernel *********************/
        const char* kernelNameBoundary;
        if (problem.getStencilType() == mgcl::MGCL_LAPLACE_7POINT)
            kernelNameBoundary = "jacobi_iter_7point_boundary";
        else if (problem.getStencilType() == mgcl::MGCL_LAPLACE_19POINT)
        {
            kernelNameBoundary = "jacobi_iter_19point_boundary";
        }
        else if (problem.getStencilType() == mgcl::MGCL_LAPLACE_27POINT)
        {
            kernelNameBoundary = "jacobi_iter_27point_boundary";
        }
        else if (problem.getStencilType() == mgcl::MGCL_VARYING)
        {
            kernelNameBoundary = "jacobi_iter_27point_varying_stencil_1d_boundary";
        }
        else if (problem.getStencilType() == mgcl::MGCL_FIXED)
        {
            kernelNameBoundary = "jacobi_iter_27point_fixed_stencil_1d_boundary";
        }
        else
        {
            assert(false && "Unknown stencil type for Jacobi inner kernel");
        }

        cl_kernel boundaryKernel = clCreateKernel(problem.getProgram(), kernelNameBoundary, &err);
        mgcl::mgclCheckError(err, "Creating boundaryKernel");

        /********************** create inner kernel *********************/
        // These are the default Jacobi kernels with proper idx_start
        const char* kernelNameInner;
        if (problem.getStencilType() == mgcl::MGCL_LAPLACE_7POINT)
            kernelNameInner = "jacobi_iter_7point";
        else if (problem.getStencilType() == mgcl::MGCL_LAPLACE_19POINT)
        {
            kernelNameInner = "jacobi_iter_19point";
        }
        else if (problem.getStencilType() == mgcl::MGCL_LAPLACE_27POINT)
        {
            kernelNameInner = "jacobi_iter_27point";
        }
        else if (problem.getStencilType() == mgcl::MGCL_VARYING)
        {
            kernelNameInner = "jacobi_iter_27point_varying_stencil_1d";
        }
        else if (problem.getStencilType() == mgcl::MGCL_FIXED)
        {
            kernelNameInner = "jacobi_iter_27point_fixed_stencil_1d";
        }
        else
        {
            assert(false && "Unknown stencil type for Jacobi inner kernel");
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

        int globalIter = 0;
        auto ptr_dvin_wrapper = &level.getDVIn();
        auto ptr_dvout_wrapper = &level.getDVOut();
        auto tmp = ptr_dvin_wrapper;

        // No need for waiting for the boundary kernel to finish, because extractBorderPlanes is in same in-order queue
        err = mgcl::MultigridEngine::updateGhosts(problem, *ptr_dvin_wrapper,
                                                  level.getMpiDataPtr(), level.isCalculatedLocally());
        mgcl::mgclCheckError(err, "Updating ghosts");

        while (globalIter < maxiter)
        {
            // if stepsPerIter > 1, multiple iterations can be done without updating ghosts in-between
            for (int innerIter = 0; innerIter < stepsPerIter && globalIter < maxiter; innerIter++, globalIter++)
            {
                auto dVIn = ptr_dvin_wrapper->getBuffer();
                auto dVOut = ptr_dvout_wrapper->getBuffer();

                store_res = globalIter == maxiter - 1;

                // calculate boundary points and inner points in separate queues concurrently
                jacobi_overlapped_helpers::jacobiBoundary(problem, level, dVIn, dVOut, store_res, problem.getCommands(), boundaryKernel, global, local, kernelNameBoundary);
                // jacobi_overlapped_helpers::jacobiInner(problem, level, dVIn, dVOut, store_res, queue2, innerKernel, global, local, kernelNameInner);

                // // No need for waiting for the boundary kernel to finish, because extractBorderPlanes is in same in-order queue
                // err = mgcl::MultigridEngine::updateGhosts(problem, *ptr_dvout_wrapper,
                //                                           level.getMpiDataPtr(), level.isCalculatedLocally());
                // mgcl::mgclCheckError(err, "Updating ghosts");

                if (!level.isCalculatedLocally())
                {
                    jacobi_overlapped_helpers::jacobiOverlappedCore(problem, level, dVIn, *ptr_dvout_wrapper, store_res, queue2, innerKernel, kernelNameInner, global, local);
                }
                else
                {
                    jacobi_overlapped_helpers::jacobiInner(problem, level, dVIn, dVOut, store_res, queue2, innerKernel, global, local, kernelNameInner);
                    err = mgcl::MultigridEngine::updateGhosts(problem, *ptr_dvout_wrapper,
                                                              level.getMpiDataPtr(), level.isCalculatedLocally());
                    mgcl::mgclCheckError(err, "Updating ghosts");
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

        if (store_res)
        {
            // TODO check for mpi
            err = mgcl::MultigridEngine::updateGhosts(problem, level.getDR(), level.getMpiDataPtr(),
                                                      level.isCalculatedLocally());
            mgcl::mgclCheckError(err, "Updating ghosts of dR");
        }

        // copy result into dVIn if needed
        if (maxiter % 2 == 1)
            level.getDVOut().copyTo(problem.getOpenCLHelper().getCommands(), level.getDVIn());

        // calculate residual and its norm
        if (return_residual)
        {
            // update residual to use current approximation v
            res = mgcl::MultigridEngine::residual(problem, level, true);
        }

        return res;
    }

    /* Runs jacobi method using OpenCL.
     * Uses Blockstencil.
     */
    double MultigridEngine::jacobi(args::JacobiBSOclArgs& args)
    {
        int err;
        int mgh = args.v_in.getMgh();
        int ngh = args.v_in.getNgh();
        int ogh = args.v_in.getOgh();
        int store_res = 0;
        double res = -1;
        int idx_start = 0;

        // decrease stepsPerIter if it's less than maxIter
        if (args.maxiter < args.stepsPerIter)
            args.stepsPerIter = args.maxiter;

        // Ghosts only need to be updated in the periodic case, so set stepsPerIter = 1 for non-periodic.
        // TODO adjust for MPI
        if (!args.periodic)
            args.stepsPerIter = 1;

        // Check if amount of ghost cells is large enough
        if (args.v_in.getGhostsM() < args.stepsPerIter)
        {
            error("#ghosts must be >= stepsPerIter!");
        }

        if (!std::holds_alternative<std::shared_ptr<CuboidBSGpu>>(args.bs_inv) && !std::holds_alternative<std::shared_ptr<BlockstencilGpu>>(args.bs_inv))
        {
            error("bs_inv must be a shared_ptr to either CuboidBSGpu or BlockstencilGpu!");
        }

        std::string kernelName = "jacobi_iter_27point_blockstencil_block_first_v_block_first_scalarjacobi";
        if (auto bs_inv_ptr = std::get_if<std::shared_ptr<BlockstencilGpu>>(&args.bs_inv))
        {
            auto& bs_inv = *bs_inv_ptr->get();
            if (bs_inv.getWidth() != 1)
            {
                error("width of bs_inv must be 1!");
            }
            kernelName = "jacobi_iter_27point_blockstencil_block_first_v_block_first_blockjacobi";
        }

        cl_event ev;

        // Create the compute kernel from the program
        cl_kernel kernel = clCreateKernel(args.program, kernelName.c_str(), &err);
        mgclCheckError(err, "Creating kernel");

        cl_mem dVIn = args.v_in.getBuffer();
        cl_mem dVOut = args.v_out.getBuffer();
        cl_mem dF = args.f.getBuffer();
        cl_mem dR = args.r.getBuffer();

        // assign kernel arguments
        int pos = 0;
        int pos_idxstart = -1;
        int pos_storeres = -1;

        auto svbuf = args.bs.getBuf();
        int svgh = args.bs.getGh();
        int svmgh = args.bs.getMgh();
        int svngh = args.bs.getNgh();
        int svogh = args.bs.getOgh();
        int svGridSize = svmgh * svngh * svogh;
        int gh = args.v_in.getGhostsM();
        cl_mem bs_inv_buf;
        if (auto bs_inv_ptr = std::get_if<std::shared_ptr<BlockstencilGpu>>(&args.bs_inv))
        {
            bs_inv_buf = bs_inv_ptr->get()->getBuf();
        }
        else
        {
            bs_inv_buf = std::get<std::shared_ptr<CuboidBSGpu>>(args.bs_inv)->getBuffer();
        }
        int svGridSizeBlock = 27 * svGridSize;
        err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &dVIn);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dVOut);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dF);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dR);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &svbuf);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &bs_inv_buf);
        err |= clSetKernelArg(kernel, ++pos, sizeof(double), &args.omega);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &mgh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ngh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ogh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svmgh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svngh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svogh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &gh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svgh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svGridSize);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svGridSizeBlock);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &idx_start);
        pos_idxstart = pos;
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &store_res);
        pos_storeres = pos;
        mgclCheckError(err, "Setting kernel arguments");

        // One work-item per cell (including ghost cells).
        size_t global = static_cast<size_t>(mgh * ngh * ogh);
        size_t local = static_cast<size_t>(128);
        if (args.conf)
        {
            const auto& c = conf::getWorkGroupSizeForKernelAndWiCount(*args.conf, kernelName, 1);
            local = c[0];
        }

        // Pad global sizes to fit to local sizes
        int kernelDims = 1;
        if (global % local != 0)
            global += local - (global % local);

        int globalIter = 0;
        while (globalIter < args.maxiter)
        {
            // Update ghosts of current input v
            if (globalIter % 2 == 1)
            {
                args.v_out.updateGhostsOclMpi(args.program, args.queue, args.dPlanesBuf, args.sendBuf, args.recvBuf, args.mpiData, args.updateGhostsLocally, args.periodic, args.conf, args.pd);
                // err = MultigridEngine::updateGhosts(problem, level.getDVOut(),
                //                                     level.getMpiDataPtr(), level.isCalculatedLocally());
                // mgclCheckError(err, "Updating ghosts");
            }
            else
            {
                args.v_in.updateGhostsOclMpi(args.program, args.queue, args.dPlanesBuf, args.sendBuf, args.recvBuf, args.mpiData, args.updateGhostsLocally, args.periodic, args.conf, args.pd);
                // err = MultigridEngine::updateGhosts(problem, level.getDVIn(),
                //                                     level.getMpiDataPtr(), level.isCalculatedLocally());
                // mgclCheckError(err, "Updating ghosts");
            }

            // if stepsPerIter > 1, multiple iterations can be done without updating ghosts in-between
            for (int innerIter = 0; innerIter < args.stepsPerIter && globalIter < args.maxiter; innerIter++, globalIter++)
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
                if (globalIter == args.maxiter - 1)
                {
                    store_res = 1;
                    err = clSetKernelArg(kernel, pos_storeres, sizeof(int), &store_res);
                    mgclCheckError(err, "Setting kernel arguments");
                }

                // recalculate and set idx_start
                idx_start = args.v_in.getGhostsM() - ((args.stepsPerIter - innerIter) - 1);
                err = clSetKernelArg(kernel, pos_idxstart, sizeof(int), &idx_start);
                mgclCheckError(err, "Setting kernel arguments");

                err = clEnqueueNDRangeKernel(args.queue, kernel, kernelDims, NULL, &global, &local, 0, NULL, &ev);
                mgclCheckError(err, "Enqueueing kernel");

                if (args.pd)
                {
                    args.pd->addMeasurement(args.queue, ev, kernelName,
                                            {global, 0, 0},
                                            {local, 1, 1});
                }
                mgclCheckError(clReleaseEvent(ev), "clReleaseEvent");
            }
        }

        if (store_res)
        {
            // TODO check for mpi
            args.r.updateGhostsOclMpi(args.program, args.queue, args.dPlanesBuf, args.sendBuf, args.recvBuf, args.mpiData, args.updateGhostsLocally, args.periodic, args.conf, args.pd);
            // err = MultigridEngine::updateGhosts(problem, level.getDR(), level.getMpiDataPtr(),
            //                                     level.isCalculatedLocally());
            // mgclCheckError(err, "Updating ghosts of dR");
        }

        // copy result into dVIn if needed
        if (args.maxiter % 2 == 1)
            args.v_out.copyTo(args.queue, args.v_in);

        // Update ghosts of dVIn
        args.v_in.updateGhostsOclMpi(args.program, args.queue, args.dPlanesBuf, args.sendBuf, args.recvBuf, args.mpiData, args.updateGhostsLocally, args.periodic, args.conf, args.pd);
        // err = MultigridEngine::updateGhosts(problem, level.getDVIn(),
        //                                     level.getMpiDataPtr(), level.isCalculatedLocally());
        // mgclCheckError(err, "Updating ghosts");

        // calculate residual and its norm
        if (args.returnResidualNorm)
        {
            // update residual to use current approximation v
            args::ResidualBSOclArgs resargs{
                args.f,
                args.v_in,
                args.r,
                args.resnorm,
                args.bs,
                args.dRsq,
                args.returnResidualNorm,
                args.periodic,
                args.updateGhostsLocally,
                args.dPlanesBuf,
                args.sendBuf,
                args.recvBuf,
                args.program,
                args.queue,
                args.context,
                args.moff,
                args.noff,
                args.ooff,
                args.mpiData,
                args.conf,
                args.pd};
            res = MultigridEngine::residual(resargs);
        }

        clReleaseKernel(kernel);

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

        double h2 = level.getH() * level.getH();
        double h2inv = 1.0 / h2; // divisor of the stencil, inverted to use * instead of / in kernel

        // check if off is too small (i.e. start < 0)
        // TODO refactor to use GPUCuboid and check against v.getGhosts
        if (moff <= -problem.ghosts || noff <= -problem.ghosts || ooff <= -problem.ghosts)
            error("moff, noff and ooff must not be <= -ghosts");

        // check if off is too large (i.e. start > end)
        if (moff * 2 >= level.m || noff * 2 >= level.n || ooff * 2 >= level.o)
            error("2*moff, 2*noff and 2*ooff must not be >= m, n or o");

        if (level.getDVIn().getGhostsM() < 1 || level.getDVIn().getGhostsN() < 1 || level.getDVIn().getGhostsO() < 1)
        {
            error("level.getDVIn() must have at least 1 ghost cell in each dimension");
        }

        if (level.getDF().getGhostsM() < 1 || level.getDF().getGhostsN() < 1 || level.getDF().getGhostsO() < 1)
        {
            error("level.getDF must have at least 1 ghost cell in each dimension");
        }

        if (level.getDR().getGhostsM() < 1 || level.getDR().getGhostsN() < 1 || level.getDR().getGhostsO() < 1)
        {
            error("level.getDR must have at least 1 ghost cell in each dimension");
        }

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
        else if (problem.stencilType == MGCL_FIXED)
        {
            kernelName = "residual_27point_fixed_stencil";
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
        else if (problem.stencilType == MGCL_FIXED)
        {
            auto& fs = *level.getFixedStencil();
            err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &dVIn);
            err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dF);
            err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dR);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &mgh);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ngh);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ogh);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &problem.ghosts);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &moff);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &noff);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ooff);
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

        err = MultigridEngine::updateGhosts(problem, level.getDR(), level.getMpiDataPtr(),
                                            level.isCalculatedLocally());
        mgclCheckError(err, "Updating ghosts of r");

        // calculate residual's 2-norm. Square elements on device and sum up on host
        if (return_residual)
        {
            if (problem.residual_norm == MGCL_L2)
            {
                auto& dRsquares = level.getDRsq();
                dRsquares.fill(problem.getProgram(), problem.getCommands(), 0.0, false, &problem.getKernelConfig(), problem.getProfilingData()); // reset to zero
                res = level.getDR().l2norm(problem.getProgram(), problem.getCommands(), &dRsquares, level.isCalculatedLocally() ? nullptr : problem.getMpiComm(),
                                           &problem.getKernelConfig(), problem.getProfilingData());
            }
            else
            {
                // calculate Infinity-Norm
                res = util::max_abs(level.getDR(), problem.getProgram(), problem.getCommands(), true, util::DEFAULT_REDUCTION_MAX_WG_SIZE, &problem.getKernelConfig(), problem.getProfilingData());
            }
        }

        clReleaseKernel(kernel); // TODO maybe clFinish before release?
        return res;
    }

    /* Calculates r = f - A*v using a Blockstencil and CuboidBS.
     * m,n,o is the size of the real grid.
     * v needs to have updated ghost cells if the problem is periodic!
     * moff, noff and ooff can be used to change the size of the grid that the residual shall be calculated for.
     *   Per default only real cells are considered (moff = 0), but with e.g. moff = -1, the first ghost cell border is
     *   considered, too. Analogously, with moff = 1 the outermost set of real cells is ignored. The calculation
     *   of the boundaries is e.g. istart = v.ghosts_m + moff.
     *   moff,noff,ooff not supported yet!
     */
    double MultigridEngine::residual(args::ResidualBSOclArgs& args)
    {
        CuboidBSGpu& v = args.v;
        CuboidBSGpu& f = args.f;
        CuboidBSGpu& r = args.r;
        double res = 0.0;
        int err;

        // check if off is too small (i.e. start < 0)
        // if (moff <= -v.getGhostsM() || noff <= -v.getGhostsN() || ooff <= -v.getGhostsO())
        //     error("moff, noff and ooff must not be <= -ghosts");

        // // check if off is too large (i.e. start > end)
        // if (moff * 2 >= v.getM() || noff * 2 >= v.getN() || ooff * 2 >= v.getO())
        //     error("2*moff, 2*noff and 2*ooff must not be >= m, n or o");

        if (v.getGhostsM() < 1 || v.getGhostsN() < 1 || v.getGhostsO() < 1)
        {
            error("v must have at least 1 ghost cell in each dimension");
        }
        if (f.getGhostsM() < 1 || f.getGhostsN() < 1 || f.getGhostsO() < 1)
        {
            error("f must have at least 1 ghost cell in each dimension");
        }
        if (r.getGhostsM() < 1 || r.getGhostsN() < 1 || r.getGhostsO() < 1)
        {
            error("r must have at least 1 ghost cell in each dimension");
        }

        const char* kernelName = "residual_27point_blockstencil_block_first_v_block_first";

        cl_event ev;

        // Create the compute kernel from the program
        cl_kernel kernel = clCreateKernel(args.program, kernelName, &err);
        mgclCheckError(err, "Creating kernel");

        // assign kernel arguments
        int pos = 0;
        cl_mem dVIn = v.getBuffer();
        cl_mem dF = f.getBuffer();
        cl_mem dR = r.getBuffer();
        cl_mem svbuf = args.bs.getBuf();
        int mgh = v.getMgh();
        int ngh = v.getNgh();
        int ogh = v.getOgh();
        int ghosts = v.getGhostsM();
        int svgh = args.bs.getGh();
        int svmgh = args.bs.getMgh();
        int svngh = args.bs.getNgh();
        int svogh = args.bs.getOgh();
        int svGridSize = svmgh * svngh * svogh;
        int svGridSizeBlock = 27 * svGridSize;

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
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghosts);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svgh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svGridSize);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svGridSizeBlock);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &args.moff);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &args.noff);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &args.ooff);

        mgclCheckError(err, "Setting residual kernel arguments");

        // one work-item per cell (including ghost cells). Pad global sizes to fit to local sizes
        size_t global = mgh * ngh * ogh;
        size_t local = 32;
        if (args.conf)
        {
            const auto& c = conf::getWorkGroupSizeForKernelAndWiCount(*args.conf, kernelName, global);
            local = c[0];
        }

        if (global % local != 0)
            global += local - (global % local);

        err = clEnqueueNDRangeKernel(args.queue, kernel, 1, NULL, &global, &local, 0, NULL, &ev);
        mgclCheckError(err, "Enqueueing residual kernel");

        if (args.pd)
        {
            args.pd->addMeasurement(args.queue, ev, kernelName,
                                    {global, 0, 0},
                                    {local, 1, 1});
        }
        mgclCheckError(clReleaseEvent(ev), "clReleaseEvent");

        r.updateGhostsOclMpi(args.program, args.queue, args.dPlanesBuf, args.sendBuf, args.recvBuf, args.mpiData, args.updateGhostsLocally, args.periodic, args.conf, args.pd);

        // calculate residual's 2-norm. Square elements on device and sum up on host
        if (args.returnResidualNorm)
        {
            if (args.resnorm == MGCL_L2)
            {
                if (args.dRsq == nullptr)
                {
                    error("dRsq is null.");
                }

                // calculate 2-Norm
                args.dRsq->fill(args.program, args.queue, 0.0, false, args.conf, args.pd); // reset to zero
                res = r.l2norm(args.program, args.queue, args.dRsq, args.updateGhostsLocally ? nullptr : args.mpiData->comm, args.conf, args.pd);
            }
            else
            {
                // calculate Infinity-Norm
                res = util::max_abs(r.getBuffer(), r.getSize(), args.program, args.queue, args.context, true, util::DEFAULT_REDUCTION_MAX_WG_SIZE, args.conf, args.pd);
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

        if (stencilType == MGCL_FIXED && fixedStencil == nullptr)
        {
            error("stencilType is fixed but fixedStencil is null!");
        }

        if (stencilType == MGCL_VARYING)
            stencilValues = stencilValuesCuboid->getData();

        if (stencilType == MGCL_FIXED)
        {
            fsRaw = fixedStencil->getData();
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
                        stencilsum = fsRaw[1][1][1]  * vraw[iv][jv][kv]
                            + fsRaw[1][1][0] * vraw[ iv ][ jv ][kv-1]
                            + fsRaw[1][1][2] * vraw[ iv ][ jv ][kv+1]
                            + fsRaw[1][0][1] * vraw[ iv ][jv-1][ kv ]
                            + fsRaw[1][2][1] * vraw[ iv ][jv+1][ kv ]
                            + fsRaw[0][1][1] * vraw[iv-1][ jv ][ kv ]
                            + fsRaw[2][1][1] * vraw[iv+1][ jv ][ kv ]
                            
                            + fsRaw[1][0][0] * vraw[ iv ][jv-1][kv-1]
                            + fsRaw[1][0][2] * vraw[ iv ][jv-1][kv+1]
                            + fsRaw[1][2][0] * vraw[ iv ][jv+1][kv-1]
                            + fsRaw[1][2][2] * vraw[ iv ][jv+1][kv+1]
                            + fsRaw[0][1][0] * vraw[iv-1][ jv ][kv-1]
                            + fsRaw[0][1][2] * vraw[iv-1][ jv ][kv+1]
                            + fsRaw[2][1][0] * vraw[iv+1][ jv ][kv-1]
                            + fsRaw[2][1][2] * vraw[iv+1][ jv ][kv+1]
                            + fsRaw[0][0][1] * vraw[iv-1][jv-1][ kv ]
                            + fsRaw[0][2][1] * vraw[iv-1][jv+1][ kv ]
                            + fsRaw[2][0][1] * vraw[iv+1][jv-1][ kv ]
                            + fsRaw[2][2][1] * vraw[iv+1][jv+1][ kv ]
                            
                            + fsRaw[0][0][0] * vraw[iv-1][jv-1][kv-1]
                            + fsRaw[0][0][2] * vraw[iv-1][jv-1][kv+1]
                            + fsRaw[0][2][0] * vraw[iv-1][jv+1][kv-1]
                            + fsRaw[0][2][2] * vraw[iv-1][jv+1][kv+1]
                            + fsRaw[2][0][0] * vraw[iv+1][jv-1][kv-1]
                            + fsRaw[2][0][2] * vraw[iv+1][jv-1][kv+1]
                            + fsRaw[2][2][0] * vraw[iv+1][jv+1][kv-1]
                            + fsRaw[2][2][2] * vraw[iv+1][jv+1][kv+1];
                        // clang-format on

                        // if (iv >= 1 && iv <= 2 && jv >= 1 && jv <= 2 && kv >= 1 && kv <= 2)
                        // {
                        //     // print27point(v, iv, jv, kv, *fixedStencil);
                        //     std::cout << "fs stencilsum: " << stencilsum << std::endl;
                        // }
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

        MultigridEngine::updateGhostsSeq(r, mpiData, periodic, updateGhostsLocally);

        // f.dumpToFile(std::to_string(mpiData ? mpiData->rank : 0) + "fSeq.txt");
        // v.dumpToFile(std::to_string(mpiData ? mpiData->rank : 0) + "vSeq.txt");
        // r.dumpToFile(std::to_string(mpiData ? mpiData->rank : 0) + "rSeq.txt");
        // if (mpiData)
        //     MPI_Barrier(mpiData->comm);
        // exit(0);

        if (!updateGhostsLocally)
        {
            double globalSum = 0;
            MPI_Allreduce(&res, &globalSum, 1, MPI_DOUBLE, MPI_SUM, mpiData->comm);
            res = globalSum;
        }

        return (returnResidualNorm && resnorm == MGCL_L2) ? sqrt(res) : res;
    }

    /* Calculates r = f - A*v using a Blockstencil and CuboidBS.
     * m,n,o is the size of the real grid.
     * v needs to have updated ghost cells if the problem is periodic!
     * moff, noff and ooff can be used to change the size of the grid that the residual shall be calculated for.
     *   Per default only real cells are considered (moff = 0), but with e.g. moff = -1, the first ghost cell border is
     *   considered, too. Analogously, with moff = 1 the outermost set of real cells is ignored. The calculation
     *   of the boundaries is e.g. istart = v.ghosts_m + moff.
     *   moff,noff,ooff not supported yet!
     */
    double MultigridEngine::residualSeq(args::ResidualBSSeqArgs& args)
    {
        CuboidBS& v = args.v;
        CuboidBS& f = args.f;
        CuboidBS& r = args.r;
        double res = 0.0;
        double**** vraw = v.getData();
        double******** bsraw = args.bs.getData();

        // check if off is too small (i.e. start < 0)
        // if (moff <= -v.getGhostsM() || noff <= -v.getGhostsN() || ooff <= -v.getGhostsO())
        //     error("moff, noff and ooff must not be <= -ghosts");

        // // check if off is too large (i.e. start > end)
        // if (moff * 2 >= v.getM() || noff * 2 >= v.getN() || ooff * 2 >= v.getO())
        //     error("2*moff, 2*noff and 2*ooff must not be >= m, n or o");

        int istart_v = v.getGhostsM() + args.moff;
        int jstart_v = v.getGhostsN() + args.noff;
        int kstart_v = v.getGhostsO() + args.ooff;
        int iend_v = v.getMgh() - v.getGhostsM() - args.moff;
        int jend_v = v.getNgh() - v.getGhostsN() - args.noff;
        int kend_v = v.getOgh() - v.getGhostsO() - args.ooff;
        int istart_r = r.getGhostsM() + args.moff;
        int jstart_r = r.getGhostsN() + args.noff;
        int kstart_r = r.getGhostsO() + args.ooff;
        int istart_f = f.getGhostsM() + args.moff;
        int jstart_f = f.getGhostsN() + args.noff;
        int kstart_f = f.getGhostsO() + args.ooff;
        int istart_sv = args.bs.getGhostsM() + args.moff;
        int jstart_sv = args.bs.getGhostsN() + args.noff;
        int kstart_sv = args.bs.getGhostsO() + args.ooff;

        for (int iv = istart_v, ir = istart_r, fi = istart_f, isv = istart_sv; iv < iend_v; iv++, ir++, fi++, isv++)
            for (int jv = jstart_v, jr = jstart_r, fj = jstart_f, jsv = jstart_sv; jv < jend_v; jv++, jr++, fj++, jsv++)
                for (int kv = kstart_v, kr = kstart_r, fk = kstart_f, ksv = kstart_sv; kv < kend_v; kv++, kr++, fk++, ksv++)
                {
                    for (int bi = 0; bi < args.bs.getBlocksize(); bi++)
                    {
                        double stencilsum = 0;
                        for (int bj = 0; bj < args.bs.getBlocksize(); bj++)
                        {
                            // clang-format off
                            stencilsum += bsraw[bi][bj][1][1][1][isv][jsv][ksv] * vraw[bj][iv][jv][kv]
                                + bsraw[bi][bj][1][1][0][isv][jsv][ksv] * vraw[bj][ iv ][ jv ][kv-1]
                                + bsraw[bi][bj][1][1][2][isv][jsv][ksv] * vraw[bj][ iv ][ jv ][kv+1]
                                + bsraw[bi][bj][1][0][1][isv][jsv][ksv] * vraw[bj][ iv ][jv-1][ kv ]
                                + bsraw[bi][bj][1][2][1][isv][jsv][ksv] * vraw[bj][ iv ][jv+1][ kv ]
                                + bsraw[bi][bj][0][1][1][isv][jsv][ksv] * vraw[bj][iv-1][ jv ][ kv ]
                                + bsraw[bi][bj][2][1][1][isv][jsv][ksv] * vraw[bj][iv+1][ jv ][ kv ]
                                
                                + bsraw[bi][bj][1][0][0][isv][jsv][ksv] * vraw[bj][ iv ][jv-1][kv-1]
                                + bsraw[bi][bj][1][0][2][isv][jsv][ksv] * vraw[bj][ iv ][jv-1][kv+1]
                                + bsraw[bi][bj][1][2][0][isv][jsv][ksv] * vraw[bj][ iv ][jv+1][kv-1]
                                + bsraw[bi][bj][1][2][2][isv][jsv][ksv] * vraw[bj][ iv ][jv+1][kv+1]
                                + bsraw[bi][bj][0][1][0][isv][jsv][ksv] * vraw[bj][iv-1][ jv ][kv-1]
                                + bsraw[bi][bj][0][1][2][isv][jsv][ksv] * vraw[bj][iv-1][ jv ][kv+1]
                                + bsraw[bi][bj][2][1][0][isv][jsv][ksv] * vraw[bj][iv+1][ jv ][kv-1]
                                + bsraw[bi][bj][2][1][2][isv][jsv][ksv] * vraw[bj][iv+1][ jv ][kv+1]
                                + bsraw[bi][bj][0][0][1][isv][jsv][ksv] * vraw[bj][iv-1][jv-1][ kv ]
                                + bsraw[bi][bj][0][2][1][isv][jsv][ksv] * vraw[bj][iv-1][jv+1][ kv ]
                                + bsraw[bi][bj][2][0][1][isv][jsv][ksv] * vraw[bj][iv+1][jv-1][ kv ]
                                + bsraw[bi][bj][2][2][1][isv][jsv][ksv] * vraw[bj][iv+1][jv+1][ kv ]
                                
                                + bsraw[bi][bj][0][0][0][isv][jsv][ksv] * vraw[bj][iv-1][jv-1][kv-1]
                                + bsraw[bi][bj][0][0][2][isv][jsv][ksv] * vraw[bj][iv-1][jv-1][kv+1]
                                + bsraw[bi][bj][0][2][0][isv][jsv][ksv] * vraw[bj][iv-1][jv+1][kv-1]
                                + bsraw[bi][bj][0][2][2][isv][jsv][ksv] * vraw[bj][iv-1][jv+1][kv+1]
                                + bsraw[bi][bj][2][0][0][isv][jsv][ksv] * vraw[bj][iv+1][jv-1][kv-1]
                                + bsraw[bi][bj][2][0][2][isv][jsv][ksv] * vraw[bj][iv+1][jv-1][kv+1]
                                + bsraw[bi][bj][2][2][0][isv][jsv][ksv] * vraw[bj][iv+1][jv+1][kv-1]
                                + bsraw[bi][bj][2][2][2][isv][jsv][ksv] * vraw[bj][iv+1][jv+1][kv+1];
                            // clang-format on

                            // if (iv == 1 && jv == 1 && kv == 1)
                            // // if (iv == 2 && jv == 2 && kv == 2 && bi == 6)
                            // {
                            //     // print27point(v, iv, jv, kv, args.bs, isv, jsv, ksv, bi, bj);
                            //     std::cout << "bs stencilsum: " << stencilsum << std::endl;
                            // }
                        }

                        // if (iv == 1 && jv == 1 && kv == 1)
                        // {
                        //     std::cout << "stencilsum: " << stencilsum << std::endl;
                        // }

                        // r = f - A*v
                        r[bi][ir][jr][kr] = f[bi][fi][fj][fk] - stencilsum;

                        if (args.returnResidualNorm)
                        {
                            if (args.resnorm == MGCL_L2)
                                res += r[bi][ir][jr][kr] * r[bi][ir][jr][kr];
                            else if (fabs(r[bi][ir][jr][kr]) > res)
                                res = fabs(r[bi][ir][jr][kr]);
                        }
                    }
                }

        r.updateGhosts(args.mpiData, args.updateGhostsLocally, args.periodic);

        if (!args.updateGhostsLocally)
        {
            double globalSum = 0;
            MPI_Allreduce(&res, &globalSum, 1, MPI_DOUBLE, MPI_SUM, args.mpiData->comm);
            res = globalSum;
        }

        return (args.returnResidualNorm && args.resnorm == MGCL_L2) ? sqrt(res) : res;
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

    void MultigridEngine::print27point(Cuboid& v, int i, int j, int k, FixedStencil& sv)
    {
        // clang-format off
        printf("27point stencil at %d,%d,%d:\n", i, j, k);
        printf(" fs * v[    self     ] = %e * %e\n", sv[1][1][1], v[ i ][ j ][ k ]);
        printf(" fs * v[ i ][ j ][k-1] = %e * %e\n", sv[1][1][0], v[ i ][ j ][k-1]);
        printf(" fs * v[ i ][ j ][k+1] = %e * %e\n", sv[1][1][2], v[ i ][ j ][k+1]);
        printf(" fs * v[ i ][j-1][ k ] = %e * %e\n", sv[1][0][1], v[ i ][j-1][ k ]);
        printf(" fs * v[ i ][j+1][ k ] = %e * %e\n", sv[1][2][1], v[ i ][j+1][ k ]);
        printf(" fs * v[i-1][ j ][ k ] = %e * %e\n", sv[0][1][1], v[i-1][ j ][ k ]);
        printf(" fs * v[i+1][ j ][ k ] = %e * %e\n", sv[2][1][1], v[i+1][ j ][ k ]);
        printf(" fs * v[ i ][j-1][k-1] = %e * %e\n", sv[1][0][0], v[ i ][j-1][k-1]);
        printf(" fs * v[ i ][j-1][k+1] = %e * %e\n", sv[1][0][2], v[ i ][j-1][k+1]);
        printf(" fs * v[ i ][j+1][k-1] = %e * %e\n", sv[1][2][0], v[ i ][j+1][k-1]);
        printf(" fs * v[ i ][j+1][k+1] = %e * %e\n", sv[1][2][2], v[ i ][j+1][k+1]);
        printf(" fs * v[i-1][ j ][k-1] = %e * %e\n", sv[0][1][0], v[i-1][ j ][k-1]);
        printf(" fs * v[i-1][ j ][k+1] = %e * %e\n", sv[0][1][2], v[i-1][ j ][k+1]);
        printf(" fs * v[i+1][ j ][k-1] = %e * %e\n", sv[2][1][0], v[i+1][ j ][k-1]);
        printf(" fs * v[i+1][ j ][k+1] = %e * %e\n", sv[2][1][2], v[i+1][ j ][k+1]);
        printf(" fs * v[i-1][j-1][ k ] = %e * %e\n", sv[0][0][1], v[i-1][j-1][ k ]);
        printf(" fs * v[i-1][j+1][ k ] = %e * %e\n", sv[0][2][1], v[i-1][j+1][ k ]);
        printf(" fs * v[i+1][j-1][ k ] = %e * %e\n", sv[2][0][1], v[i+1][j-1][ k ]);
        printf(" fs * v[i+1][j+1][ k ] = %e * %e\n", sv[2][2][1], v[i+1][j+1][ k ]);
        printf(" fs * v[i-1][j-1][k-1] = %e * %e\n", sv[0][0][0], v[i-1][j-1][k-1]);
        printf(" fs * v[i-1][j-1][k+1] = %e * %e\n", sv[0][0][2], v[i-1][j-1][k+1]);
        printf(" fs * v[i-1][j+1][k-1] = %e * %e\n", sv[0][2][0], v[i-1][j+1][k-1]);
        printf(" fs * v[i-1][j+1][k+1] = %e * %e\n", sv[0][2][2], v[i-1][j+1][k+1]);
        printf(" fs * v[i+1][j-1][k-1] = %e * %e\n", sv[2][0][0], v[i+1][j-1][k-1]);
        printf(" fs * v[i+1][j-1][k+1] = %e * %e\n", sv[2][0][2], v[i+1][j-1][k+1]);
        printf(" fs * v[i+1][j+1][k-1] = %e * %e\n", sv[2][2][0], v[i+1][j+1][k-1]);
        printf(" fs * v[i+1][j+1][k+1] = %e * %e\n", sv[2][2][2], v[i+1][j+1][k+1]);
        // clang-format on
    }

    void MultigridEngine::print27point(CuboidBS& v, int i, int j, int k,
                                       const Blockstencil& sv, int i_sv, int j_sv, int k_sv, int bi, int bj)
    {
        // clang-format off
        printf("27point stencil at %d,%d,%d for block entry %d,%d; grid point at %d,%d,%d\n", i_sv, j_sv, k_sv, bi, bj,i,j,k);
        printf(" bs * v[    self     ] = %e * %e\n", sv[bi][bj][1][1][1][i_sv][j_sv][k_sv], v[bj][ i ][ j ][ k ]);
        printf(" bs * v[ i ][ j ][k-1] = %e * %e\n", sv[bi][bj][1][1][0][i_sv][j_sv][k_sv], v[bj][ i ][ j ][k-1]);
        printf(" bs * v[ i ][ j ][k+1] = %e * %e\n", sv[bi][bj][1][1][2][i_sv][j_sv][k_sv], v[bj][ i ][ j ][k+1]);
        printf(" bs * v[ i ][j-1][ k ] = %e * %e\n", sv[bi][bj][1][0][1][i_sv][j_sv][k_sv], v[bj][ i ][j-1][ k ]);
        printf(" bs * v[ i ][j+1][ k ] = %e * %e\n", sv[bi][bj][1][2][1][i_sv][j_sv][k_sv], v[bj][ i ][j+1][ k ]);
        printf(" bs * v[i-1][ j ][ k ] = %e * %e\n", sv[bi][bj][0][1][1][i_sv][j_sv][k_sv], v[bj][i-1][ j ][ k ]);
        printf(" bs * v[i+1][ j ][ k ] = %e * %e\n", sv[bi][bj][2][1][1][i_sv][j_sv][k_sv], v[bj][i+1][ j ][ k ]);
        printf(" bs * v[ i ][j-1][k-1] = %e * %e\n", sv[bi][bj][1][0][0][i_sv][j_sv][k_sv], v[bj][ i ][j-1][k-1]);
        printf(" bs * v[ i ][j-1][k+1] = %e * %e\n", sv[bi][bj][1][0][2][i_sv][j_sv][k_sv], v[bj][ i ][j-1][k+1]);
        printf(" bs * v[ i ][j+1][k-1] = %e * %e\n", sv[bi][bj][1][2][0][i_sv][j_sv][k_sv], v[bj][ i ][j+1][k-1]);
        printf(" bs * v[ i ][j+1][k+1] = %e * %e\n", sv[bi][bj][1][2][2][i_sv][j_sv][k_sv], v[bj][ i ][j+1][k+1]);
        printf(" bs * v[i-1][ j ][k-1] = %e * %e\n", sv[bi][bj][0][1][0][i_sv][j_sv][k_sv], v[bj][i-1][ j ][k-1]);
        printf(" bs * v[i-1][ j ][k+1] = %e * %e\n", sv[bi][bj][0][1][2][i_sv][j_sv][k_sv], v[bj][i-1][ j ][k+1]);
        printf(" bs * v[i+1][ j ][k-1] = %e * %e\n", sv[bi][bj][2][1][0][i_sv][j_sv][k_sv], v[bj][i+1][ j ][k-1]);
        printf(" bs * v[i+1][ j ][k+1] = %e * %e\n", sv[bi][bj][2][1][2][i_sv][j_sv][k_sv], v[bj][i+1][ j ][k+1]);
        printf(" bs * v[i-1][j-1][ k ] = %e * %e\n", sv[bi][bj][0][0][1][i_sv][j_sv][k_sv], v[bj][i-1][j-1][ k ]);
        printf(" bs * v[i-1][j+1][ k ] = %e * %e\n", sv[bi][bj][0][2][1][i_sv][j_sv][k_sv], v[bj][i-1][j+1][ k ]);
        printf(" bs * v[i+1][j-1][ k ] = %e * %e\n", sv[bi][bj][2][0][1][i_sv][j_sv][k_sv], v[bj][i+1][j-1][ k ]);
        printf(" bs * v[i+1][j+1][ k ] = %e * %e\n", sv[bi][bj][2][2][1][i_sv][j_sv][k_sv], v[bj][i+1][j+1][ k ]);
        printf(" bs * v[i-1][j-1][k-1] = %e * %e\n", sv[bi][bj][0][0][0][i_sv][j_sv][k_sv], v[bj][i-1][j-1][k-1]);
        printf(" bs * v[i-1][j-1][k+1] = %e * %e\n", sv[bi][bj][0][0][2][i_sv][j_sv][k_sv], v[bj][i-1][j-1][k+1]);
        printf(" bs * v[i-1][j+1][k-1] = %e * %e\n", sv[bi][bj][0][2][0][i_sv][j_sv][k_sv], v[bj][i-1][j+1][k-1]);
        printf(" bs * v[i-1][j+1][k+1] = %e * %e\n", sv[bi][bj][0][2][2][i_sv][j_sv][k_sv], v[bj][i-1][j+1][k+1]);
        printf(" bs * v[i+1][j-1][k-1] = %e * %e\n", sv[bi][bj][2][0][0][i_sv][j_sv][k_sv], v[bj][i+1][j-1][k-1]);
        printf(" bs * v[i+1][j-1][k+1] = %e * %e\n", sv[bi][bj][2][0][2][i_sv][j_sv][k_sv], v[bj][i+1][j-1][k+1]);
        printf(" bs * v[i+1][j+1][k-1] = %e * %e\n", sv[bi][bj][2][2][0][i_sv][j_sv][k_sv], v[bj][i+1][j+1][k-1]);
        printf(" bs * v[i+1][j+1][k+1] = %e * %e\n", sv[bi][bj][2][2][2][i_sv][j_sv][k_sv], v[bj][i+1][j+1][k+1]);
        // clang-format on
    }
}
