/**
 * @date 18.06.2025
 * This file contains tests to compare the runtime performance of point-wise vs block Jacobi for a
 * scalar vs. vector-valued problem.
 *
 */

#include "bench_util.hpp"
#include "nanobench.h"

#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"

#include <CL/cl.h>
#include <catch2/catch_message.hpp>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>
using namespace std::chrono_literals;

#include "../src/mgcl/cuboid.hpp"
#include "../src/mgcl/multigrid_engine.hpp"
#include "../src/mgcl/problem.hpp"
#include "../src/mgcl/util.hpp"
#include "../test/test_utility.hpp"
#include "cli_args.hpp"

namespace mgcl_bench_jacobi_blockstencil
{
    using size_t2 = struct
    {
        size_t x, y;
    };

    struct JacobiBSOclArgs
    {
        mgcl::CuboidBSGpu& f;
        mgcl::CuboidBSGpu& v_in;
        mgcl::CuboidBSGpu& v_out;
        mgcl::CuboidBSGpu& r;
        mgcl::MGCL_RESIDUAL_NORM resnorm;
        mgcl::BlockstencilGpu& bs;
        mgcl::TBlockstencilInv& bs_inv;
        mgcl::CuboidBSGpu* dRsq;

        bool returnResidualNorm;
        bool periodic;
        bool updateGhostsLocally;
        int maxiter;
        int stepsPerIter;
        double omega;

        mgcl::BufferGpu* dPlanesBuf;
        std::vector<double>* sendBuf;
        std::vector<double>* recvBuf;

        cl_program program;
        cl_command_queue queue;
        cl_context context;

        int moff = 0;
        int noff = 0;
        int ooff = 0;
        mgcl::MPILevelData* mpiData = nullptr;

        mgcl::conf::KernelConfig* conf = nullptr;
        mgcl::ProfilingData* pd = nullptr;
    };

    struct JacobiBSSeqArgs
    {
        mgcl::CuboidBS& f;
        mgcl::CuboidBS& v;
        mgcl::CuboidBS& r;
        mgcl::MGCL_RESIDUAL_NORM resnorm;
        mgcl::Blockstencil& bs;
        mgcl::TBlockstencilInv& bs_inv;

        bool returnResidualNorm;
        bool periodic;
        bool updateGhostsLocally;
        int maxiter;
        int stepsPerIter;
        double omega;

        mgcl::MPILevelData* mpiData = nullptr;
    };

    double jacobi(JacobiBSOclArgs& args)
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

        if (!std::holds_alternative<std::shared_ptr<mgcl::CuboidBSGpu>>(args.bs_inv) && !std::holds_alternative<std::shared_ptr<mgcl::BlockstencilGpu>>(args.bs_inv))
        {
            error("bs_inv must be a shared_ptr to either CuboidBSGpu or BlockstencilGpu!");
        }

        std::string kernelName = "jacobi_iter_27point_blockstencil_block_first_v_block_first_scalarjacobi";
        if (auto bs_inv_ptr = std::get_if<std::shared_ptr<mgcl::BlockstencilGpu>>(&args.bs_inv))
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
        mgcl::mgclCheckError(err, "Creating kernel");

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
        if (auto bs_inv_ptr = std::get_if<std::shared_ptr<mgcl::BlockstencilGpu>>(&args.bs_inv))
        {
            bs_inv_buf = bs_inv_ptr->get()->getBuf();
        }
        else
        {
            bs_inv_buf = std::get<std::shared_ptr<mgcl::CuboidBSGpu>>(args.bs_inv)->getBuffer();
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
        mgcl::mgclCheckError(err, "Setting kernel arguments");

        // One work-item per cell (including ghost cells).
        size_t global = static_cast<size_t>(mgh * ngh * ogh);
        size_t local = static_cast<size_t>(128);
        if (args.conf)
        {
            const auto& c = mgcl::conf::getWorkGroupSizeForKernelAndWiCount(*args.conf, kernelName, 1);
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
                    mgcl::mgclCheckError(err, "Setting kernel arguments");
                }
                else
                {
                    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &dVIn);
                    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &dVOut);
                    mgcl::mgclCheckError(err, "Setting kernel arguments");
                }

                // set flag to store residual in last iteration
                if (globalIter == args.maxiter - 1)
                {
                    store_res = 1;
                    err = clSetKernelArg(kernel, pos_storeres, sizeof(int), &store_res);
                    mgcl::mgclCheckError(err, "Setting kernel arguments");
                }

                // recalculate and set idx_start
                idx_start = args.v_in.getGhostsM() - ((args.stepsPerIter - innerIter) - 1);
                err = clSetKernelArg(kernel, pos_idxstart, sizeof(int), &idx_start);
                mgcl::mgclCheckError(err, "Setting kernel arguments");

                err = clEnqueueNDRangeKernel(args.queue, kernel, kernelDims, NULL, &global, &local, 0, NULL, &ev);
                mgcl::mgclCheckError(err, "Enqueueing kernel");

                if (args.pd)
                {
                    args.pd->addMeasurement(args.queue, ev, kernelName,
                                            {global, 0, 0},
                                            {local, 1, 1});
                }
                mgcl::mgclCheckError(clReleaseEvent(ev), "clReleaseEvent");
            }
        }

        // copy result into dVIn if needed
        if (args.maxiter % 2 == 1)
        {
            mgcl::CuboidBSGpu::swap(args.v_in, args.v_out);
        }

        // Update ghosts of dVIn
        args.v_in.updateGhostsOclMpi(args.program, args.queue, args.dPlanesBuf, args.sendBuf, args.recvBuf, args.mpiData, args.updateGhostsLocally, args.periodic, args.conf, args.pd);
        // err = MultigridEngine::updateGhosts(problem, level.getDVIn(),
        //                                     level.getMpiDataPtr(), level.isCalculatedLocally());
        // mgclCheckError(err, "Updating ghosts");

        // calculate residual and its norm
        if (args.returnResidualNorm)
        {
            // update residual to use current approximation v
            mgcl::args::ResidualBSOclArgs resargs{
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
            res = mgcl::MultigridEngine::residual(resargs);
        }

        clReleaseKernel(kernel);

        return res;
    }

    double jacobiSeq(JacobiBSSeqArgs& args)
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
        if (mgcl::util::seq::min3(args.v.getGhostsM(), args.v.getGhostsN(), args.v.getGhostsO()) < stepsPerIter)
        {
            error("#ghosts of v must be >= stepsPerIter!");
        }

        if (mgcl::util::seq::min3(args.r.getGhostsM(), args.r.getGhostsN(), args.r.getGhostsO()) < stepsPerIter - 1)
        {
            error("#ghosts of r must be >= stepsPerIter - 1!");
        }

        if (mgcl::util::seq::min3(args.f.getGhostsM(), args.f.getGhostsN(), args.f.getGhostsO()) < stepsPerIter - 1)
        {
            error("#ghosts of f must be >= stepsPerIter - 1!");
        }

        if (!std::holds_alternative<std::shared_ptr<mgcl::CuboidBS>>(args.bs_inv) && !std::holds_alternative<std::shared_ptr<mgcl::Blockstencil>>(args.bs_inv))
        {
            error("bs_inv must be a shared_ptr to either CuboidBS or Blockstencil!");
        }

        if (auto bs_inv_ptr = std::get_if<std::shared_ptr<mgcl::Blockstencil>>(&args.bs_inv))
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
                mgcl::args::ResidualBSSeqArgs residualArgs{
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
                res = mgcl::MultigridEngine::residualSeq(residualArgs);

                // args.r.dumpToFile("r_vectorial.txt");

                // smoother type is Jacobi_Block: bs_inv is a matrix
                if (auto bs_inv_ptr = std::get_if<std::shared_ptr<mgcl::Blockstencil>>(&args.bs_inv))
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
                                        sum += bs_inv[bi][bj][0][0][0][isv][jsv][ksv] * args.r[ir][jr][kr][bj];

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
                                    vraw[iv][jv][kv][bi] = vraw[iv][jv][kv][bi] + args.omega * sum;
                                    // vraw[iv][jv][kv][bi] = vraw[iv][jv][kv][bi] + args.omega * args.bs_inv[bi][bi][0][0][0][isv][jsv][ksv] * args.r[ir][jr][kr][bi];
                                }
                            }
                }
                else
                {
                    // smoother type is Jacobi_Scalar: bs_inv is scalar
                    auto& bs_inv = *std::get_if<std::shared_ptr<mgcl::CuboidBS>>(&args.bs_inv)->get();
                    for (int iv = istart_v, ir = istart_r, isv = istart_sv; iv < iend_v; iv++, ir++, isv++)
                        for (int jv = jstart_v, jr = jstart_r, jsv = jstart_sv; jv < jend_v; jv++, jr++, jsv++)
                            for (int kv = kstart_v, kr = kstart_r, ksv = kstart_sv; kv < kend_v; kv++, kr++, ksv++)
                                for (int bi = 0; bi < args.v.getBlocksize(); bi++)
                                {
                                    // update v, i.e. v_{i+1} = v_i + omega * bs_inv * r
                                    vraw[iv][jv][kv][bi] = vraw[iv][jv][kv][bi] + args.omega * bs_inv[isv][jsv][ksv][bi] * args.r[ir][jr][kr][bi];

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
            mgcl::args::ResidualBSSeqArgs residualArgs{
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
            res = mgcl::MultigridEngine::residualSeq(residualArgs);
        }

        return res;
    }

    // Benchs Jacobi scalar vs vector-valued problem
    // run with e.g. ./benchmarks bench_ocl_jacobi_bs_scalar_vs_vector --grids 8 --jacobiIters 1,2
    TEST_CASE("bench_ocl_jacobi_bs_scalar_vs_vector")
    {
        using std::min;

        if (CLI_ARGS::grids.size() == 0 && (CLI_ARGS::gridsMin.size() == 0 || CLI_ARGS::gridsMax.size() == 0))
            throw "Need to specify at least one local grid size, e.g. using --grids 4,8,16 or --gridsMin 4,4,4 AND --gridsMax 32,32,32";

        if (CLI_ARGS::jacobiIters.empty())
        {
            throw "Need to specify jacobiIters, e.g. --jacobiIters 1,2,3";
        }

        // if (CLI_ARGS::jacobiStepsPerIter.empty())
        // {
        //     throw "Need to specify jacobiStepsPerIter, e.g. --spi 1,2,3";
        // }

        if (!CLI_ARGS::jacobiStepsPerIter.empty())
        {
            std::cout << "Currently only for 1 jacobiStepsPerIter! --spi ignored." << std::endl;
        }
        CLI_ARGS::jacobiStepsPerIter = {1};

        // build grids to be tested from CLI args
        std::vector<std::vector<int>> gridsTBT;
        for (auto N : CLI_ARGS::grids)
            gridsTBT.push_back({N, N, N});
        if (CLI_ARGS::gridsMin.size() > 0 && CLI_ARGS::gridsMax.size() > 0)
            for (int m = CLI_ARGS::gridsMin[0]; m <= CLI_ARGS::gridsMax[0]; m *= 2)
                for (int n = CLI_ARGS::gridsMin[1]; n <= CLI_ARGS::gridsMax[1]; n *= 2)
                    for (int o = CLI_ARGS::gridsMin[2]; o <= CLI_ARGS::gridsMax[2]; o *= 2)
                        gridsTBT.push_back({m, n, o});

        std::vector<bench_util::ResultJacobiBlockstencil> results;
        std::stringstream profilingStream;

        int ghosts_in = 1;
        bool returnResidualNorm = false;
        bool periodic = false;

        for (auto gr : gridsTBT)
        {
            int m = gr[0];
            int n = gr[1];
            int o = gr[2];

            double omega = 0.8;
            double h2 = 1.0 / (double)(m * m);
            mgcl::MGCL_RESIDUAL_NORM resnorm = mgcl::MGCL_L2;
            mgcl::MGCL_STENCIL stencilType = mgcl::MGCL_VARYING;

            // scalar Problem, point-wise Jacobi
            {
                auto v_in = std::make_shared<mgcl::Cuboid>(m, n, o, ghosts_in, ghosts_in, ghosts_in);
                auto f_in = std::make_shared<mgcl::Cuboid>(m, n, o, ghosts_in, ghosts_in, ghosts_in);
                auto r_in = std::make_shared<mgcl::Cuboid>(m, n, o, ghosts_in, ghosts_in, ghosts_in);
                // v_in->fill1dIndex(true);
                // f_in->fill1dIndex(true);
                v_in->fillRandom();
                f_in->fillRandom();

                mgcl::Problem p(m, n, o, f_in, v_in);
                p.setGhostsIn(ghosts_in);
                p.setUseOpencl(true);
                p.setSilent(true);
                p.setStencilType(mgcl::MGCL_VARYING);
                p.setProfilingEnabled(CLI_ARGS::enableKernelProfiling);
                p.setSmootherType(mgcl::MGCL_JACOBI_SCALAR);
                p.setJacobiIterationsPerKernel(1);
                // p.setKernelFile("kernel_optimizations.cl");
                if (CLI_ARGS::useBinaryFile)
                {
                    p.setBinaryFile("jacobiBenchBlockstencilScalarProblem.bin");
                }
                p.setDeviceType(CL_DEVICE_TYPE_GPU);

                auto& sv = p.createStencilValues();
                sv->fill1dIndex(false);

                p.init();

                if (CLI_ARGS::enableKernelProfiling)
                    p.getProfilingData()->getMeasurements().clear();

                auto& lv0 = p.getLevelAt(0);

                ankerl::nanobench::Bench bench;
                bench.timeUnit(1ms, "ms")
                    .epochs(CLI_ARGS::bench_epochs)
                    .epochIterations(CLI_ARGS::bench_iterations)
                    .relative(false);

                for (int iters : CLI_ARGS::jacobiIters)
                    for (int stepsPerIter : CLI_ARGS::jacobiStepsPerIter)
                    {
                        if (stepsPerIter > iters)
                        {
                            continue;
                        }

                        std::string name = std::string("jacobi_scalarproblem_pointwiseJacobi_")
                                               .append(std::to_string(m))
                                               .append("x")
                                               .append(std::to_string(n))
                                               .append("x")
                                               .append(std::to_string(o))
                                               .append("_iters")
                                               .append(std::to_string(iters));

                        bench.run(std::string(name).c_str(), [&] { //
                            mgcl::MultigridEngine::jacobi(p, lv0, iters, false, stepsPerIter);
                            p.finish();
                        });

                        bench_util::ResultJacobiBlockstencil res;
                        res.name = name;
                        res.minTime = bench_util::getMinTime(bench, name);
                        res.medianTime = bench_util::getMedianTime(bench, name);
                        res.avgTime = bench_util::getAvgTime(bench, name);
                        res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
                        res.m = m;
                        res.n = n;
                        res.o = o;
                        res.iters = iters;
                        res.spi = stepsPerIter;
                        res.blocksize = 1;
                        results.push_back(res);
                    }

                if (CLI_ARGS::enableKernelProfiling)
                {
                    profilingStream << "jacobi_scalarproblem_pointwiseJacobi" << std::endl;
                    p.getProfilingData()->printBestTimingsPerKernelAsCsv(profilingStream);
                }
            }

            int ghosts = 1;
            int width = 3;
            int ghosts_bs = 1;

            // vector-valued Problem, blocksize 2^3, point-wise Jacobi
            {
                int mbs = m / 2;
                int nbs = n / 2;
                int obs = o / 2;
                int blocksize = 8;

                // create dummy problem
                auto v_dummy = std::make_shared<mgcl::Cuboid>(1, 1, 1);
                auto f_dummy = std::make_shared<mgcl::Cuboid>(1, 1, 1);
                mgcl::Problem p(1, 1, 1, f_dummy, v_dummy);
                p.setUseOpencl(true);
                p.setSilent(true);
                p.setDeviceType(CL_DEVICE_TYPE_GPU);
                p.setProfilingEnabled(CLI_ARGS::enableKernelProfiling);
                p.getOpenCLHelper().setPreprocessorConstant("BLOCKSIZE", std::to_string(blocksize));
                p.init();

                if (CLI_ARGS::enableKernelProfiling)
                    p.getProfilingData()->getMeasurements().clear();

                auto v_in = std::make_shared<mgcl::CuboidBS>(mbs, nbs, obs, ghosts_in, ghosts_in, ghosts_in, blocksize);
                auto f_in = std::make_shared<mgcl::CuboidBS>(mbs, nbs, obs, ghosts_in, ghosts_in, ghosts_in, blocksize);
                auto r_in = std::make_shared<mgcl::CuboidBS>(mbs, nbs, obs, ghosts_in, ghosts_in, ghosts_in, blocksize);
                // v_in->fill1dIndex(true);
                // f_in->fill1dIndex(true);
                v_in->fillRandom();
                f_in->fillRandom();

                mgcl::CuboidBS v(mbs, nbs, obs, ghosts, ghosts, ghosts, blocksize);
                mgcl::CuboidBS r(mbs, nbs, obs, ghosts, ghosts, ghosts, blocksize);
                mgcl::CuboidBS f(mbs, nbs, obs, ghosts, ghosts, ghosts, blocksize);
                mgcl::Blockstencil bs(mbs, nbs, obs, width, blocksize, ghosts_bs, ghosts_bs, ghosts_bs);
                mgcl::CuboidBS bs_inv(mbs, nbs, obs, ghosts_bs, ghosts_bs, ghosts_bs, blocksize);

                mgcl::CuboidBSGpu d_v_in(p.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, v);
                mgcl::CuboidBSGpu d_v_out(p.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, v);
                mgcl::CuboidBSGpu d_r(p.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, r);
                mgcl::CuboidBSGpu d_f(p.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, f);
                mgcl::BlockstencilGpu d_bs(bs, p.getContext(), p.getCommands(), p.getProgram());
                auto d_bs_inv = std::make_shared<mgcl::CuboidBSGpu>(p.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bs_inv);
                mgcl::TBlockstencilInv d_bs_inv_variant = d_bs_inv;
                mgcl::CuboidBSGpu dRSquares(p.getContext(), CL_MEM_READ_WRITE, mbs, nbs, obs, 0, 0, 0, blocksize);

                ankerl::nanobench::Bench bench;
                bench.timeUnit(1ms, "ms")
                    .epochs(CLI_ARGS::bench_epochs)
                    .epochIterations(CLI_ARGS::bench_iterations)
                    .relative(false);

                for (int iters : CLI_ARGS::jacobiIters)
                    for (int stepsPerIter : CLI_ARGS::jacobiStepsPerIter)
                    {
                        if (stepsPerIter > iters)
                        {
                            continue;
                        }

                        JacobiBSOclArgs args{
                            d_f,
                            d_v_in,
                            d_v_out,
                            d_r,
                            resnorm,
                            d_bs,
                            d_bs_inv_variant,
                            &dRSquares,
                            true,
                            periodic,
                            true, iters, stepsPerIter, omega,
                            nullptr, nullptr, nullptr,
                            p.getProgram(), p.getCommands(), p.getContext(),
                            0, 0, 0, nullptr,
                            &p.getKernelConfig(),
                            p.getProfilingData()};

                        std::string name = std::string("jacobi_vectorproblem_pointwiseJacobi_")
                                               .append(std::to_string(mbs))
                                               .append("x")
                                               .append(std::to_string(nbs))
                                               .append("x")
                                               .append(std::to_string(obs))
                                               .append("_blocksize_")
                                               .append(std::to_string(blocksize))
                                               .append("_iters")
                                               .append(std::to_string(iters));

                        bench.run(std::string(name).c_str(), [&] { //
                            jacobi(args);
                            p.finish();
                        });

                        bench_util::ResultJacobiBlockstencil res;
                        res.name = name;
                        res.minTime = bench_util::getMinTime(bench, name);
                        res.medianTime = bench_util::getMedianTime(bench, name);
                        res.avgTime = bench_util::getAvgTime(bench, name);
                        res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
                        res.m = mbs;
                        res.n = nbs;
                        res.o = obs;
                        res.iters = iters;
                        res.spi = stepsPerIter;
                        res.blocksize = blocksize;
                        results.push_back(res);
                    }
                if (CLI_ARGS::enableKernelProfiling)
                {
                    profilingStream << "jacobi_vectorproblem8_pointwiseJacobi" << std::endl;
                    p.getProfilingData()->printBestTimingsPerKernelAsCsv(profilingStream);
                }
            }

            // vector-valued Problem, blocksize 2^3, block Jacobi
            {
                int mbs = m / 2;
                int nbs = n / 2;
                int obs = o / 2;
                int blocksize = 8;

                // create dummy problem
                auto v_dummy = std::make_shared<mgcl::Cuboid>(1, 1, 1);
                auto f_dummy = std::make_shared<mgcl::Cuboid>(1, 1, 1);
                mgcl::Problem p(1, 1, 1, f_dummy, v_dummy);
                p.setUseOpencl(true);
                p.setSilent(true);
                p.setDeviceType(CL_DEVICE_TYPE_GPU);
                p.setProfilingEnabled(CLI_ARGS::enableKernelProfiling);
                p.getOpenCLHelper().setPreprocessorConstant("BLOCKSIZE", std::to_string(blocksize));
                p.init();

                if (CLI_ARGS::enableKernelProfiling)
                    p.getProfilingData()->getMeasurements().clear();

                auto v_in = std::make_shared<mgcl::CuboidBS>(mbs, nbs, obs, ghosts_in, ghosts_in, ghosts_in, blocksize);
                auto f_in = std::make_shared<mgcl::CuboidBS>(mbs, nbs, obs, ghosts_in, ghosts_in, ghosts_in, blocksize);
                auto r_in = std::make_shared<mgcl::CuboidBS>(mbs, nbs, obs, ghosts_in, ghosts_in, ghosts_in, blocksize);
                // v_in->fill1dIndex(true);
                // f_in->fill1dIndex(true);
                v_in->fillRandom();
                f_in->fillRandom();

                mgcl::CuboidBS v(mbs, nbs, obs, ghosts, ghosts, ghosts, blocksize);
                mgcl::CuboidBS r(mbs, nbs, obs, ghosts, ghosts, ghosts, blocksize);
                mgcl::CuboidBS f(mbs, nbs, obs, ghosts, ghosts, ghosts, blocksize);
                mgcl::Blockstencil bs(mbs, nbs, obs, width, blocksize, ghosts_bs, ghosts_bs, ghosts_bs);
                mgcl::Blockstencil bs_inv(mbs, nbs, obs, 1, blocksize, ghosts_bs, ghosts_bs, ghosts_bs);

                mgcl::CuboidBSGpu d_v_in(p.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, v);
                mgcl::CuboidBSGpu d_v_out(p.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, v);
                mgcl::CuboidBSGpu d_r(p.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, r);
                mgcl::CuboidBSGpu d_f(p.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, f);
                mgcl::BlockstencilGpu d_bs(bs, p.getContext(), p.getCommands(), p.getProgram());
                auto d_bs_inv = std::make_shared<mgcl::BlockstencilGpu>(bs_inv, p.getContext(), p.getCommands(), p.getProgram());
                mgcl::TBlockstencilInv d_bs_inv_variant = d_bs_inv;
                mgcl::CuboidBSGpu dRSquares(p.getContext(), CL_MEM_READ_WRITE, mbs, nbs, obs, 0, 0, 0, blocksize);

                ankerl::nanobench::Bench bench;
                bench.timeUnit(1ms, "ms")
                    .epochs(CLI_ARGS::bench_epochs)
                    .epochIterations(CLI_ARGS::bench_iterations)
                    .relative(false);

                for (int iters : CLI_ARGS::jacobiIters)
                    for (int stepsPerIter : CLI_ARGS::jacobiStepsPerIter)
                    {
                        if (stepsPerIter > iters)
                        {
                            continue;
                        }

                        JacobiBSOclArgs args{
                            d_f,
                            d_v_in,
                            d_v_out,
                            d_r,
                            resnorm,
                            d_bs,
                            d_bs_inv_variant,
                            &dRSquares,
                            true,
                            periodic,
                            true, iters, stepsPerIter, omega,
                            nullptr, nullptr, nullptr,
                            p.getProgram(), p.getCommands(), p.getContext(),
                            0, 0, 0, nullptr,
                            &p.getKernelConfig(),
                            p.getProfilingData()};

                        std::string name = std::string("jacobi_vectorproblem_blockJacobi_")
                                               .append(std::to_string(mbs))
                                               .append("x")
                                               .append(std::to_string(nbs))
                                               .append("x")
                                               .append(std::to_string(obs))
                                               .append("_blocksize_")
                                               .append(std::to_string(blocksize))
                                               .append("_iters")
                                               .append(std::to_string(iters));

                        bench.run(std::string(name).c_str(), [&] { //
                            jacobi(args);
                            p.finish();
                        });

                        bench_util::ResultJacobiBlockstencil res;
                        res.name = name;
                        res.minTime = bench_util::getMinTime(bench, name);
                        res.medianTime = bench_util::getMedianTime(bench, name);
                        res.avgTime = bench_util::getAvgTime(bench, name);
                        res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
                        res.m = mbs;
                        res.n = nbs;
                        res.o = obs;
                        res.iters = iters;
                        res.spi = stepsPerIter;
                        res.blocksize = blocksize;
                        results.push_back(res);
                    }
                if (CLI_ARGS::enableKernelProfiling)
                {
                    profilingStream << "jacobi_vectorproblem8_blockJacobi" << std::endl;
                    p.getProfilingData()->printBestTimingsPerKernelAsCsv(profilingStream);
                }
            }
            // vector-valued Problem, blocksize 4^3, point-wise Jacobi
            {
                int mbs = m / 4;
                int nbs = n / 4;
                int obs = o / 4;
                int blocksize = 64;

                // create dummy problem
                auto v_dummy = std::make_shared<mgcl::Cuboid>(1, 1, 1);
                auto f_dummy = std::make_shared<mgcl::Cuboid>(1, 1, 1);
                mgcl::Problem p(1, 1, 1, f_dummy, v_dummy);
                p.setUseOpencl(true);
                p.setSilent(true);
                p.setDeviceType(CL_DEVICE_TYPE_GPU);
                p.setProfilingEnabled(CLI_ARGS::enableKernelProfiling);
                p.getOpenCLHelper().setPreprocessorConstant("BLOCKSIZE", std::to_string(blocksize));
                p.init();

                if (CLI_ARGS::enableKernelProfiling)
                    p.getProfilingData()->getMeasurements().clear();

                auto v_in = std::make_shared<mgcl::CuboidBS>(mbs, nbs, obs, ghosts_in, ghosts_in, ghosts_in, blocksize);
                auto f_in = std::make_shared<mgcl::CuboidBS>(mbs, nbs, obs, ghosts_in, ghosts_in, ghosts_in, blocksize);
                auto r_in = std::make_shared<mgcl::CuboidBS>(mbs, nbs, obs, ghosts_in, ghosts_in, ghosts_in, blocksize);
                // v_in->fill1dIndex(true);
                // f_in->fill1dIndex(true);
                v_in->fillRandom();
                f_in->fillRandom();

                mgcl::CuboidBS v(mbs, nbs, obs, ghosts, ghosts, ghosts, blocksize);
                mgcl::CuboidBS r(mbs, nbs, obs, ghosts, ghosts, ghosts, blocksize);
                mgcl::CuboidBS f(mbs, nbs, obs, ghosts, ghosts, ghosts, blocksize);
                mgcl::Blockstencil bs(mbs, nbs, obs, width, blocksize, ghosts_bs, ghosts_bs, ghosts_bs);
                mgcl::CuboidBS bs_inv(mbs, nbs, obs, ghosts_bs, ghosts_bs, ghosts_bs, blocksize);

                mgcl::CuboidBSGpu d_v_in(p.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, v);
                mgcl::CuboidBSGpu d_v_out(p.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, v);
                mgcl::CuboidBSGpu d_r(p.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, r);
                mgcl::CuboidBSGpu d_f(p.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, f);
                mgcl::BlockstencilGpu d_bs(bs, p.getContext(), p.getCommands(), p.getProgram());
                auto d_bs_inv = std::make_shared<mgcl::CuboidBSGpu>(p.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bs_inv);
                mgcl::TBlockstencilInv d_bs_inv_variant = d_bs_inv;
                mgcl::CuboidBSGpu dRSquares(p.getContext(), CL_MEM_READ_WRITE, mbs, nbs, obs, 0, 0, 0, blocksize);

                ankerl::nanobench::Bench bench;
                bench.timeUnit(1ms, "ms")
                    .epochs(CLI_ARGS::bench_epochs)
                    .epochIterations(CLI_ARGS::bench_iterations)
                    .relative(false);

                for (int iters : CLI_ARGS::jacobiIters)
                    for (int stepsPerIter : CLI_ARGS::jacobiStepsPerIter)
                    {
                        if (stepsPerIter > iters)
                        {
                            continue;
                        }

                        JacobiBSOclArgs args{
                            d_f,
                            d_v_in,
                            d_v_out,
                            d_r,
                            resnorm,
                            d_bs,
                            d_bs_inv_variant,
                            &dRSquares,
                            true,
                            periodic,
                            true, iters, stepsPerIter, omega,
                            nullptr, nullptr, nullptr,
                            p.getProgram(), p.getCommands(), p.getContext(),
                            0, 0, 0, nullptr,
                            &p.getKernelConfig(),
                            p.getProfilingData()};

                        std::string name = std::string("jacobi_vectorproblem_pointwiseJacobi_")
                                               .append(std::to_string(mbs))
                                               .append("x")
                                               .append(std::to_string(nbs))
                                               .append("x")
                                               .append(std::to_string(obs))
                                               .append("_blocksize_")
                                               .append(std::to_string(blocksize))
                                               .append("_iters")
                                               .append(std::to_string(iters));

                        bench.run(std::string(name).c_str(), [&] { //
                            jacobi(args);
                            p.finish();
                        });

                        bench_util::ResultJacobiBlockstencil res;
                        res.name = name;
                        res.minTime = bench_util::getMinTime(bench, name);
                        res.medianTime = bench_util::getMedianTime(bench, name);
                        res.avgTime = bench_util::getAvgTime(bench, name);
                        res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
                        res.m = mbs;
                        res.n = nbs;
                        res.o = obs;
                        res.iters = iters;
                        res.spi = stepsPerIter;
                        res.blocksize = blocksize;
                        results.push_back(res);
                    }
                if (CLI_ARGS::enableKernelProfiling)
                {
                    profilingStream << "jacobi_vectorproblem64_pointwiseJacobi" << std::endl;
                    p.getProfilingData()->printBestTimingsPerKernelAsCsv(profilingStream);
                }
            }

            // vector-valued Problem, blocksize 4^3, block Jacobi
            {
                int mbs = m / 4;
                int nbs = n / 4;
                int obs = o / 4;
                int blocksize = 64;

                // create dummy problem
                auto v_dummy = std::make_shared<mgcl::Cuboid>(1, 1, 1);
                auto f_dummy = std::make_shared<mgcl::Cuboid>(1, 1, 1);
                mgcl::Problem p(1, 1, 1, f_dummy, v_dummy);
                p.setUseOpencl(true);
                p.setSilent(true);
                p.setDeviceType(CL_DEVICE_TYPE_GPU);
                p.setProfilingEnabled(CLI_ARGS::enableKernelProfiling);
                p.getOpenCLHelper().setPreprocessorConstant("BLOCKSIZE", std::to_string(blocksize));
                p.init();

                if (CLI_ARGS::enableKernelProfiling)
                    p.getProfilingData()->getMeasurements().clear();

                auto v_in = std::make_shared<mgcl::CuboidBS>(mbs, nbs, obs, ghosts_in, ghosts_in, ghosts_in, blocksize);
                auto f_in = std::make_shared<mgcl::CuboidBS>(mbs, nbs, obs, ghosts_in, ghosts_in, ghosts_in, blocksize);
                auto r_in = std::make_shared<mgcl::CuboidBS>(mbs, nbs, obs, ghosts_in, ghosts_in, ghosts_in, blocksize);
                // v_in->fill1dIndex(true);
                // f_in->fill1dIndex(true);
                v_in->fillRandom();
                f_in->fillRandom();

                mgcl::CuboidBS v(mbs, nbs, obs, ghosts, ghosts, ghosts, blocksize);
                mgcl::CuboidBS r(mbs, nbs, obs, ghosts, ghosts, ghosts, blocksize);
                mgcl::CuboidBS f(mbs, nbs, obs, ghosts, ghosts, ghosts, blocksize);
                mgcl::Blockstencil bs(mbs, nbs, obs, width, blocksize, ghosts_bs, ghosts_bs, ghosts_bs);
                mgcl::Blockstencil bs_inv(mbs, nbs, obs, 1, blocksize, ghosts_bs, ghosts_bs, ghosts_bs);

                mgcl::CuboidBSGpu d_v_in(p.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, v);
                mgcl::CuboidBSGpu d_v_out(p.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, v);
                mgcl::CuboidBSGpu d_r(p.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, r);
                mgcl::CuboidBSGpu d_f(p.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, f);
                mgcl::BlockstencilGpu d_bs(bs, p.getContext(), p.getCommands(), p.getProgram());
                auto d_bs_inv = std::make_shared<mgcl::BlockstencilGpu>(bs_inv, p.getContext(), p.getCommands(), p.getProgram());
                mgcl::TBlockstencilInv d_bs_inv_variant = d_bs_inv;
                mgcl::CuboidBSGpu dRSquares(p.getContext(), CL_MEM_READ_WRITE, mbs, nbs, obs, 0, 0, 0, blocksize);

                ankerl::nanobench::Bench bench;
                bench.timeUnit(1ms, "ms")
                    .epochs(CLI_ARGS::bench_epochs)
                    .epochIterations(CLI_ARGS::bench_iterations)
                    .relative(false);

                for (int iters : CLI_ARGS::jacobiIters)
                    for (int stepsPerIter : CLI_ARGS::jacobiStepsPerIter)
                    {
                        if (stepsPerIter > iters)
                        {
                            continue;
                        }

                        JacobiBSOclArgs args{
                            d_f,
                            d_v_in,
                            d_v_out,
                            d_r,
                            resnorm,
                            d_bs,
                            d_bs_inv_variant,
                            &dRSquares,
                            true,
                            periodic,
                            true, iters, stepsPerIter, omega,
                            nullptr, nullptr, nullptr,
                            p.getProgram(), p.getCommands(), p.getContext(),
                            0, 0, 0, nullptr,
                            &p.getKernelConfig(),
                            p.getProfilingData()};

                        std::string name = std::string("jacobi_vectorproblem_blockJacobi_")
                                               .append(std::to_string(mbs))
                                               .append("x")
                                               .append(std::to_string(nbs))
                                               .append("x")
                                               .append(std::to_string(obs))
                                               .append("_blocksize_")
                                               .append(std::to_string(blocksize))
                                               .append("_iters")
                                               .append(std::to_string(iters));

                        bench.run(std::string(name).c_str(), [&] { //
                            jacobi(args);
                            p.finish();
                        });

                        bench_util::ResultJacobiBlockstencil res;
                        res.name = name;
                        res.minTime = bench_util::getMinTime(bench, name);
                        res.medianTime = bench_util::getMedianTime(bench, name);
                        res.avgTime = bench_util::getAvgTime(bench, name);
                        res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
                        res.m = mbs;
                        res.n = nbs;
                        res.o = obs;
                        res.iters = iters;
                        res.spi = stepsPerIter;
                        res.blocksize = blocksize;
                        results.push_back(res);
                    }

                if (CLI_ARGS::enableKernelProfiling)
                {
                    profilingStream << "jacobi_vectorproblem64_blockJacobi" << std::endl;
                    p.getProfilingData()->printBestTimingsPerKernelAsCsv(profilingStream);
                }
            }
        }
        bench_util::printCsvFormat(results);

        std::cout << profilingStream.str() << std::endl;
    }

    // Benchs Jacobi scalar vs vector-valued problem
    // run with e.g. ./benchmarks bench_ocl_jacobi_bs_scalar_vs_vector --grids 8 --jacobiIters 1,2
    TEST_CASE("bench_seq_jacobi_bs_scalar_vs_vector")
    {
        using std::min;

        if (CLI_ARGS::grids.size() == 0 && (CLI_ARGS::gridsMin.size() == 0 || CLI_ARGS::gridsMax.size() == 0))
            throw "Need to specify at least one local grid size, e.g. using --grids 4,8,16 or --gridsMin 4,4,4 AND --gridsMax 32,32,32";

        if (CLI_ARGS::jacobiIters.empty())
        {
            throw "Need to specify jacobiIters, e.g. --jacobiIters 1,2,3";
        }

        // if (CLI_ARGS::jacobiStepsPerIter.empty())
        // {
        //     throw "Need to specify jacobiStepsPerIter, e.g. --spi 1,2,3";
        // }

        if (!CLI_ARGS::jacobiStepsPerIter.empty())
        {
            std::cout << "Currently only for 1 jacobiStepsPerIter! --spi ignored." << std::endl;
        }
        CLI_ARGS::jacobiStepsPerIter = {1};

        // build grids to be tested from CLI args
        std::vector<std::vector<int>> gridsTBT;
        for (auto N : CLI_ARGS::grids)
            gridsTBT.push_back({N, N, N});
        if (CLI_ARGS::gridsMin.size() > 0 && CLI_ARGS::gridsMax.size() > 0)
            for (int m = CLI_ARGS::gridsMin[0]; m <= CLI_ARGS::gridsMax[0]; m *= 2)
                for (int n = CLI_ARGS::gridsMin[1]; n <= CLI_ARGS::gridsMax[1]; n *= 2)
                    for (int o = CLI_ARGS::gridsMin[2]; o <= CLI_ARGS::gridsMax[2]; o *= 2)
                        gridsTBT.push_back({m, n, o});

        std::vector<bench_util::ResultJacobiBlockstencil> results;

        int ghosts_in = 1;
        bool returnResidualNorm = false;
        bool periodic = false;

        for (auto gr : gridsTBT)
        {
            int m = gr[0];
            int n = gr[1];
            int o = gr[2];

            double omega = 0.8;
            double h2 = 1.0 / (double)(m * m);
            mgcl::MGCL_RESIDUAL_NORM resnorm = mgcl::MGCL_L2;
            mgcl::MGCL_STENCIL stencilType = mgcl::MGCL_VARYING;

            // scalar Problem, point-wise Jacobi
            {
                auto v_in = std::make_shared<mgcl::Cuboid>(m, n, o, ghosts_in, ghosts_in, ghosts_in);
                auto f_in = std::make_shared<mgcl::Cuboid>(m, n, o, ghosts_in, ghosts_in, ghosts_in);
                auto r_in = std::make_shared<mgcl::Cuboid>(m, n, o, ghosts_in, ghosts_in, ghosts_in);
                // v_in->fill1dIndex(true);
                // f_in->fill1dIndex(true);
                v_in->fillRandom();
                f_in->fillRandom();

                mgcl::Problem p(m, n, o, f_in, v_in);
                p.setGhostsIn(ghosts_in);
                p.setUseOpencl(false);
                p.setSilent(true);
                p.setStencilType(mgcl::MGCL_VARYING);
                p.setSmootherType(mgcl::MGCL_JACOBI_SCALAR);
                p.setJacobiIterationsPerKernel(1);

                auto& sv = p.createStencilValues();
                sv->fill1dIndex(false);

                p.init();

                auto& lv0 = p.getLevelAt(0);

                ankerl::nanobench::Bench bench;
                bench.timeUnit(1ms, "ms")
                    .epochs(CLI_ARGS::bench_epochs)
                    .epochIterations(CLI_ARGS::bench_iterations)
                    .relative(false);

                for (int iters : CLI_ARGS::jacobiIters)
                    for (int stepsPerIter : CLI_ARGS::jacobiStepsPerIter)
                    {
                        if (stepsPerIter > iters)
                        {
                            continue;
                        }

                        std::string name = std::string("jacobi_scalarproblem_pointwiseJacobi_")
                                               .append(std::to_string(m))
                                               .append("x")
                                               .append(std::to_string(n))
                                               .append("x")
                                               .append(std::to_string(o))
                                               .append("_iters")
                                               .append(std::to_string(iters));

                        bench.run(std::string(name).c_str(), [&] { //
                            mgcl::MultigridEngine::jacobiSeq(lv0.getV(), lv0.getF(), lv0.getR(), omega, h2, iters, resnorm, stencilType, 0, sv.get(), nullptr, returnResidualNorm, periodic, true, stepsPerIter, nullptr);
                            p.finish();
                        });

                        bench_util::ResultJacobiBlockstencil res;
                        res.name = name;
                        res.minTime = bench_util::getMinTime(bench, name);
                        res.medianTime = bench_util::getMedianTime(bench, name);
                        res.avgTime = bench_util::getAvgTime(bench, name);
                        res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
                        res.m = m;
                        res.n = n;
                        res.o = o;
                        res.iters = iters;
                        res.spi = stepsPerIter;
                        res.blocksize = 1;
                        results.push_back(res);
                    }
            }

            int ghosts = 1;
            int width = 3;
            int ghosts_bs = 1;

            // vector-valued Problem, blocksize 2^3, point-wise Jacobi
            {
                int mbs = m / 2;
                int nbs = n / 2;
                int obs = o / 2;
                int blocksize = 8;

                auto v_in = std::make_shared<mgcl::CuboidBS>(mbs, nbs, obs, ghosts_in, ghosts_in, ghosts_in, blocksize);
                auto f_in = std::make_shared<mgcl::CuboidBS>(mbs, nbs, obs, ghosts_in, ghosts_in, ghosts_in, blocksize);
                auto r_in = std::make_shared<mgcl::CuboidBS>(mbs, nbs, obs, ghosts_in, ghosts_in, ghosts_in, blocksize);
                // v_in->fill1dIndex(true);
                // f_in->fill1dIndex(true);
                v_in->fillRandom();
                f_in->fillRandom();

                mgcl::CuboidBS v(mbs, nbs, obs, ghosts, ghosts, ghosts, blocksize);
                mgcl::CuboidBS r(mbs, nbs, obs, ghosts, ghosts, ghosts, blocksize);
                mgcl::CuboidBS f(mbs, nbs, obs, ghosts, ghosts, ghosts, blocksize);
                mgcl::Blockstencil bs(mbs, nbs, obs, width, blocksize, ghosts_bs, ghosts_bs, ghosts_bs);
                auto bs_inv = std::make_shared<mgcl::CuboidBS>(mbs, nbs, obs, ghosts_bs, ghosts_bs, ghosts_bs, blocksize);
                mgcl::TBlockstencilInv bs_inv_variant = bs_inv;

                ankerl::nanobench::Bench bench;
                bench.timeUnit(1ms, "ms")
                    .epochs(CLI_ARGS::bench_epochs)
                    .epochIterations(CLI_ARGS::bench_iterations)
                    .relative(false);

                for (int iters : CLI_ARGS::jacobiIters)
                    for (int stepsPerIter : CLI_ARGS::jacobiStepsPerIter)
                    {
                        if (stepsPerIter > iters)
                        {
                            continue;
                        }

                        JacobiBSSeqArgs args{
                            f,
                            v,
                            r,
                            resnorm,
                            bs,
                            bs_inv_variant,
                            returnResidualNorm,
                            periodic,
                            true, iters, stepsPerIter, omega,
                            nullptr};

                        std::string name = std::string("jacobi_vectorproblem_pointwiseJacobi_")
                                               .append(std::to_string(mbs))
                                               .append("x")
                                               .append(std::to_string(nbs))
                                               .append("x")
                                               .append(std::to_string(obs))
                                               .append("_blocksize_")
                                               .append(std::to_string(blocksize))
                                               .append("_iters")
                                               .append(std::to_string(iters));

                        bench.run(std::string(name).c_str(), [&] { //
                            jacobiSeq(args);
                        });

                        bench_util::ResultJacobiBlockstencil res;
                        res.name = name;
                        res.minTime = bench_util::getMinTime(bench, name);
                        res.medianTime = bench_util::getMedianTime(bench, name);
                        res.avgTime = bench_util::getAvgTime(bench, name);
                        res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
                        res.m = mbs;
                        res.n = nbs;
                        res.o = obs;
                        res.iters = iters;
                        res.spi = stepsPerIter;
                        res.blocksize = blocksize;
                        results.push_back(res);
                    }
            }

            // vector-valued Problem, blocksize 2^3, block-Jacobi
            {
                int mbs = m / 2;
                int nbs = n / 2;
                int obs = o / 2;
                int blocksize = 8;

                auto v_in = std::make_shared<mgcl::CuboidBS>(mbs, nbs, obs, ghosts_in, ghosts_in, ghosts_in, blocksize);
                auto f_in = std::make_shared<mgcl::CuboidBS>(mbs, nbs, obs, ghosts_in, ghosts_in, ghosts_in, blocksize);
                auto r_in = std::make_shared<mgcl::CuboidBS>(mbs, nbs, obs, ghosts_in, ghosts_in, ghosts_in, blocksize);
                // v_in->fill1dIndex(true);
                // f_in->fill1dIndex(true);
                v_in->fillRandom();
                f_in->fillRandom();

                mgcl::CuboidBS v(mbs, nbs, obs, ghosts, ghosts, ghosts, blocksize);
                mgcl::CuboidBS r(mbs, nbs, obs, ghosts, ghosts, ghosts, blocksize);
                mgcl::CuboidBS f(mbs, nbs, obs, ghosts, ghosts, ghosts, blocksize);
                mgcl::Blockstencil bs(mbs, nbs, obs, width, blocksize, ghosts_bs, ghosts_bs, ghosts_bs);
                auto bs_inv = std::make_shared<mgcl::Blockstencil>(mbs, nbs, obs, 1, blocksize, ghosts_bs, ghosts_bs, ghosts_bs);
                mgcl::TBlockstencilInv bs_inv_variant = bs_inv;

                ankerl::nanobench::Bench bench;
                bench.timeUnit(1ms, "ms")
                    .epochs(CLI_ARGS::bench_epochs)
                    .epochIterations(CLI_ARGS::bench_iterations)
                    .relative(false);

                for (int iters : CLI_ARGS::jacobiIters)
                    for (int stepsPerIter : CLI_ARGS::jacobiStepsPerIter)
                    {
                        if (stepsPerIter > iters)
                        {
                            continue;
                        }

                        JacobiBSSeqArgs args{
                            f,
                            v,
                            r,
                            resnorm,
                            bs,
                            bs_inv_variant,
                            returnResidualNorm,
                            periodic,
                            true, iters, stepsPerIter, omega,
                            nullptr};

                        std::string name = std::string("jacobi_vectorproblem_blockJacobi_")
                                               .append(std::to_string(mbs))
                                               .append("x")
                                               .append(std::to_string(nbs))
                                               .append("x")
                                               .append(std::to_string(obs))
                                               .append("_blocksize_")
                                               .append(std::to_string(blocksize))
                                               .append("_iters")
                                               .append(std::to_string(iters));

                        bench.run(std::string(name).c_str(), [&] { //
                            jacobiSeq(args);
                        });

                        bench_util::ResultJacobiBlockstencil res;
                        res.name = name;
                        res.minTime = bench_util::getMinTime(bench, name);
                        res.medianTime = bench_util::getMedianTime(bench, name);
                        res.avgTime = bench_util::getAvgTime(bench, name);
                        res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
                        res.m = mbs;
                        res.n = nbs;
                        res.o = obs;
                        res.iters = iters;
                        res.spi = stepsPerIter;
                        res.blocksize = blocksize;
                        results.push_back(res);
                    }
            }

            // vector-valued Problem, blocksize 4^3, point-wise Jacobi
            {
                int mbs = m / 4;
                int nbs = n / 4;
                int obs = o / 4;
                int blocksize = 64;

                auto v_in = std::make_shared<mgcl::CuboidBS>(mbs, nbs, obs, ghosts_in, ghosts_in, ghosts_in, blocksize);
                auto f_in = std::make_shared<mgcl::CuboidBS>(mbs, nbs, obs, ghosts_in, ghosts_in, ghosts_in, blocksize);
                auto r_in = std::make_shared<mgcl::CuboidBS>(mbs, nbs, obs, ghosts_in, ghosts_in, ghosts_in, blocksize);
                // v_in->fill1dIndex(true);
                // f_in->fill1dIndex(true);
                v_in->fillRandom();
                f_in->fillRandom();

                mgcl::CuboidBS v(mbs, nbs, obs, ghosts, ghosts, ghosts, blocksize);
                mgcl::CuboidBS r(mbs, nbs, obs, ghosts, ghosts, ghosts, blocksize);
                mgcl::CuboidBS f(mbs, nbs, obs, ghosts, ghosts, ghosts, blocksize);
                mgcl::Blockstencil bs(mbs, nbs, obs, width, blocksize, ghosts_bs, ghosts_bs, ghosts_bs);
                auto bs_inv = std::make_shared<mgcl::CuboidBS>(mbs, nbs, obs, ghosts_bs, ghosts_bs, ghosts_bs, blocksize);
                mgcl::TBlockstencilInv bs_inv_variant = bs_inv;

                ankerl::nanobench::Bench bench;
                bench.timeUnit(1ms, "ms")
                    .epochs(CLI_ARGS::bench_epochs)
                    .epochIterations(CLI_ARGS::bench_iterations)
                    .relative(false);

                for (int iters : CLI_ARGS::jacobiIters)
                    for (int stepsPerIter : CLI_ARGS::jacobiStepsPerIter)
                    {
                        if (stepsPerIter > iters)
                        {
                            continue;
                        }

                        JacobiBSSeqArgs args{
                            f,
                            v,
                            r,
                            resnorm,
                            bs,
                            bs_inv_variant,
                            returnResidualNorm,
                            periodic,
                            true, iters, stepsPerIter, omega,
                            nullptr};

                        std::string name = std::string("jacobi_vectorproblem_pointwiseJacobi_")
                                               .append(std::to_string(mbs))
                                               .append("x")
                                               .append(std::to_string(nbs))
                                               .append("x")
                                               .append(std::to_string(obs))
                                               .append("_blocksize_")
                                               .append(std::to_string(blocksize))
                                               .append("_iters")
                                               .append(std::to_string(iters));

                        bench.run(std::string(name).c_str(), [&] { //
                            jacobiSeq(args);
                        });

                        bench_util::ResultJacobiBlockstencil res;
                        res.name = name;
                        res.minTime = bench_util::getMinTime(bench, name);
                        res.medianTime = bench_util::getMedianTime(bench, name);
                        res.avgTime = bench_util::getAvgTime(bench, name);
                        res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
                        res.m = mbs;
                        res.n = nbs;
                        res.o = obs;
                        res.iters = iters;
                        res.spi = stepsPerIter;
                        res.blocksize = blocksize;
                        results.push_back(res);
                    }
            }

            // vector-valued Problem, blocksize 4^3, block-Jacobi
            {
                int mbs = m / 4;
                int nbs = n / 4;
                int obs = o / 4;
                int blocksize = 64;

                auto v_in = std::make_shared<mgcl::CuboidBS>(mbs, nbs, obs, ghosts_in, ghosts_in, ghosts_in, blocksize);
                auto f_in = std::make_shared<mgcl::CuboidBS>(mbs, nbs, obs, ghosts_in, ghosts_in, ghosts_in, blocksize);
                auto r_in = std::make_shared<mgcl::CuboidBS>(mbs, nbs, obs, ghosts_in, ghosts_in, ghosts_in, blocksize);
                // v_in->fill1dIndex(true);
                // f_in->fill1dIndex(true);
                v_in->fillRandom();
                f_in->fillRandom();

                mgcl::CuboidBS v(mbs, nbs, obs, ghosts, ghosts, ghosts, blocksize);
                mgcl::CuboidBS r(mbs, nbs, obs, ghosts, ghosts, ghosts, blocksize);
                mgcl::CuboidBS f(mbs, nbs, obs, ghosts, ghosts, ghosts, blocksize);
                mgcl::Blockstencil bs(mbs, nbs, obs, width, blocksize, ghosts_bs, ghosts_bs, ghosts_bs);
                auto bs_inv = std::make_shared<mgcl::Blockstencil>(mbs, nbs, obs, 1, blocksize, ghosts_bs, ghosts_bs, ghosts_bs);
                mgcl::TBlockstencilInv bs_inv_variant = bs_inv;

                ankerl::nanobench::Bench bench;
                bench.timeUnit(1ms, "ms")
                    .epochs(CLI_ARGS::bench_epochs)
                    .epochIterations(CLI_ARGS::bench_iterations)
                    .relative(false);

                for (int iters : CLI_ARGS::jacobiIters)
                    for (int stepsPerIter : CLI_ARGS::jacobiStepsPerIter)
                    {
                        if (stepsPerIter > iters)
                        {
                            continue;
                        }

                        JacobiBSSeqArgs args{
                            f,
                            v,
                            r,
                            resnorm,
                            bs,
                            bs_inv_variant,
                            returnResidualNorm,
                            periodic,
                            true, iters, stepsPerIter, omega,
                            nullptr};

                        std::string name = std::string("jacobi_vectorproblem_blockJacobi_")
                                               .append(std::to_string(mbs))
                                               .append("x")
                                               .append(std::to_string(nbs))
                                               .append("x")
                                               .append(std::to_string(obs))
                                               .append("_blocksize_")
                                               .append(std::to_string(blocksize))
                                               .append("_iters")
                                               .append(std::to_string(iters));

                        bench.run(std::string(name).c_str(), [&] { //
                            jacobiSeq(args);
                        });

                        bench_util::ResultJacobiBlockstencil res;
                        res.name = name;
                        res.minTime = bench_util::getMinTime(bench, name);
                        res.medianTime = bench_util::getMedianTime(bench, name);
                        res.avgTime = bench_util::getAvgTime(bench, name);
                        res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
                        res.m = mbs;
                        res.n = nbs;
                        res.o = obs;
                        res.iters = iters;
                        res.spi = stepsPerIter;
                        res.blocksize = blocksize;
                        results.push_back(res);
                    }
            }
        }
        bench_util::printCsvFormat(results);
    }
}