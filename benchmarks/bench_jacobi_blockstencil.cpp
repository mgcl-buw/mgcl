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

        std::string kernelName = "jacobi_iter_27point_blockstencil_block_first_v_gp_first_scalarjacobi";
        if (auto bs_inv_ptr = std::get_if<std::shared_ptr<mgcl::BlockstencilGpu>>(&args.bs_inv))
        {
            auto& bs_inv = *bs_inv_ptr->get();
            if (bs_inv.getWidth() != 1)
            {
                error("width of bs_inv must be 1!");
            }
            kernelName = "jacobi_iter_27point_blockstencil_block_first_v_gp_first_blockjacobi";
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
                args.v_out.updateGhostsOclMpi(args.program, args.queue, args.dPlanesBuf, args.sendBuf, args.recvBuf, args.mpiData, args.updateGhostsLocally, args.conf, args.pd);
                // err = MultigridEngine::updateGhosts(problem, level.getDVOut(),
                //                                     level.getMpiDataPtr(), level.isCalculatedLocally());
                // mgclCheckError(err, "Updating ghosts");
            }
            else
            {
                args.v_in.updateGhostsOclMpi(args.program, args.queue, args.dPlanesBuf, args.sendBuf, args.recvBuf, args.mpiData, args.updateGhostsLocally, args.conf, args.pd);
                // err = MultigridEngine::updateGhosts(problem, level.getDVIn(),
                //                                     level.getMpiDataPtr(), level.isCalculatedLocally());
                // mgclCheckError(err, "Updating ghosts");
            }

            if (globalIter == 1)
            {
                args.v_out.dumpToFile(args.queue, "v_out.txt");
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

        if (store_res)
        {
            // TODO check for mpi
            args.r.updateGhostsOclMpi(args.program, args.queue, args.dPlanesBuf, args.sendBuf, args.recvBuf, args.mpiData, args.updateGhostsLocally, args.conf, args.pd);
            // err = MultigridEngine::updateGhosts(problem, level.getDR(), level.getMpiDataPtr(),
            //                                     level.isCalculatedLocally());
            // mgclCheckError(err, "Updating ghosts of dR");
        }

        // copy result into dVIn if needed
        if (args.maxiter % 2 == 1)
            args.v_out.copyTo(args.queue, args.v_in);

        // Update ghosts of dVIn
        args.v_in.updateGhostsOclMpi(args.program, args.queue, args.dPlanesBuf, args.sendBuf, args.recvBuf, args.mpiData, args.updateGhostsLocally, args.conf, args.pd);
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

    // Benchs the various residual fixed stencil kernel versions
    TEST_CASE("bench_ocl_jacobi_bs_scalar_vs_vector")
    {
        using std::min;

        if (CLI_ARGS::grids.size() == 0 && (CLI_ARGS::gridsMin.size() == 0 || CLI_ARGS::gridsMax.size() == 0))
            throw "Need to specify at least one local grid size, e.g. using --grids 4,8,16 or --gridsMin 4,4,4 AND --gridsMax 32,32,32";

        if (CLI_ARGS::jacobiIters.empty())
        {
            throw "Need to specify jacobiIters, e.g. --jacobiIters 1,2,3";
        }

        if (CLI_ARGS::jacobiStepsPerIter.empty())
        {
            throw "Need to specify jacobiStepsPerIter, e.g. --spi 1,2,3";
        }

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

        int ghosts = 1;
        bool returnResidualNorm = false;
        bool periodic = false;

        for (auto gr : gridsTBT)
        {
            int m = gr[0];
            int n = gr[1];
            int o = gr[2];

            int mgh = m + 2 * ghosts;
            int ngh = n + 2 * ghosts;
            int ogh = o + 2 * ghosts;

            double omega = 0.8;
            double h2 = 1.0 / (double)(m * m);
            mgcl::MGCL_RESIDUAL_NORM resnorm = mgcl::MGCL_L2;
            mgcl::MGCL_STENCIL stencilType = mgcl::MGCL_VARYING;

            // scalar Problem, point-wise Jacobi
            {
                auto v_in = std::make_shared<mgcl::Cuboid>(m, n, o, ghosts, ghosts, ghosts);
                auto f_in = std::make_shared<mgcl::Cuboid>(m, n, o, ghosts, ghosts, ghosts);
                auto r_in = std::make_shared<mgcl::Cuboid>(m, n, o, ghosts, ghosts, ghosts);
                // v_in->fill1dIndex(true);
                // f_in->fill1dIndex(true);
                v_in->fillRandom();
                f_in->fillRandom();

                mgcl::Problem p(m, n, o, f_in, v_in);
                p.setGhostsIn(ghosts);
                p.setUseOpencl(true);
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

                auto& sv = p.getStencilValues();
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
                                               .append(std::to_string(o));

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
            }

            // vector-valued Problem, blocksize 2^3, point-wise Jacobi
            {
                int mbs = m / 2;
                int nbs = n / 2;
                int obs = o / 2;
                int blocksize = 8;
                auto v_in = std::make_shared<mgcl::CuboidBS>(mbs, nbs, obs, ghosts, ghosts, ghosts, blocksize);
                auto f_in = std::make_shared<mgcl::CuboidBS>(mbs, nbs, obs, ghosts, ghosts, ghosts, blocksize);
                auto r_in = std::make_shared<mgcl::CuboidBS>(mbs, nbs, obs, ghosts, ghosts, ghosts, blocksize);
                // v_in->fill1dIndex(true);
                // f_in->fill1dIndex(true);
                v_in->fillRandom();
                f_in->fillRandom();

                mgcl::Problem p(mbs, nbs, obs, f_in, v_in);
                p.setGhostsIn(ghosts);
                p.setUseOpencl(true);
                p.setStencilType(mgcl::MGCL_BLOCKSTENCIL);
                p.setProfilingEnabled(CLI_ARGS::enableKernelProfiling);
                p.setSmootherType(mgcl::MGCL_JACOBI_SCALAR);
                p.setSilent(true);
                // p.setJacobiIterationsPerKernel(1);
                // p.setKernelFile("kernel_optimizations.cl");
                if (CLI_ARGS::useBinaryFile)
                {
                    p.setBinaryFile("jacobiBenchBlockstencilScalarProblem.bin");
                }
                p.setDeviceType(CL_DEVICE_TYPE_GPU);

                auto sv = p.getBlockstencil();
                mgcl::FixedStencil fs1(3);
                mgcl_test::fill27pLaplace(fs1, 1.0 / (double)(mbs), false);
                mgcl_test::fillBlockstencilFromFixedStencil(*sv, fs1);

                // sv->dumpToFile("sv.txt");

                auto rbs = p.getRestrictionBlockstencil();
                auto pbs = p.getProlongationBlockstencil();
                mgcl_test::fill3dFullWeightRestrictionBlockstencil(*rbs);
                mgcl_test::fill3dBilinearProlongationBlockstencil(*pbs);

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

                        mgcl::args::JacobiBSOclArgs args{
                            lv0.getDFBS(),
                            lv0.getDVBSIn(),
                            lv0.getDVBSOut(),
                            lv0.getDRBS(),
                            resnorm,
                            *lv0.getBlockstencilGpu(),
                            lv0.getBlockstencilInvVariant(),
                            lv0.getDRsqBSPtr().get(),
                            returnResidualNorm, periodic,
                            true, iters, stepsPerIter, omega,
                            nullptr, nullptr, nullptr,
                            p.getProgram(), p.getCommands(), p.getContext(),
                            0, 0, 0, nullptr,
                            &p.getKernelConfig(), p.getProfilingData()};

                        std::string name = std::string("jacobi_vectorproblem_pointwiseJacobi_")
                                               .append(std::to_string(m))
                                               .append("x")
                                               .append(std::to_string(n))
                                               .append("x")
                                               .append(std::to_string(o));

                        bench.run(std::string(name).c_str(), [&] { //
                            mgcl::MultigridEngine::jacobi(args);
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
        }
        bench_util::printCsvFormat(results);
    }
}