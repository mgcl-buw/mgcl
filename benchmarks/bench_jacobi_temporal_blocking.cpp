/**
 * @date 10.03.2025
 * This file contains tests to compare the runtime performance of simple Jacobi kernel vs. temporal blocking
 * using local memory.
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
#include "cli_args.hpp"

namespace mgcl_bench_residual_varying
{
    using size_t2 = struct
    {
        size_t x, y;
    };

    struct JacobiTBArgs
    {
        bool return_residual;
        int m;
        int n;
        int o;
        int mgh;
        int ngh;
        int ogh;
        double h2;
        int ghosts;
        double omega;
        int num_x_planes;

        cl_program program;
        cl_command_queue commands;
        size_t2 wgsize;

        mgcl::CuboidGpu& c_dVIn;
        mgcl::CuboidGpu& c_dVOut;
        mgcl::CuboidGpu& c_dF;
        mgcl::CuboidGpu& c_dR;
        mgcl::VaryingStencilGpu& c_stencilValues;

        mgcl::ProfilingData* pd;

        int moff;
        int noff;
        int ooff;
    };

    void jacobi_ocl_tb_2iters(JacobiTBArgs& args)
    {
        int err;
        int m = args.m;
        int n = args.n;
        int o = args.o;
        int mgh = args.mgh;
        int ngh = args.ngh;
        int ogh = args.ogh;
        int store_res = 0;
        double res = -1;
        int idx_start = 0;

        cl_event ev;

        double h2 = 1.0 / static_cast<double>(m * m);
        double dinv = h2 / 6.0;

        // Create the compute kernel from the program
        const char* kernelName = "jacobi_iter_27point_varying_stencil_2d_local_mem_2iters";

        cl_kernel kernel = clCreateKernel(args.program, kernelName, &err);
        mgcl::mgclCheckError(err, "Creating kernel");

        cl_mem dVIn = args.c_dVIn.getBuffer();
        cl_mem dVOut = args.c_dVOut.getBuffer();
        cl_mem dF = args.c_dF.getBuffer();
        cl_mem dR = args.c_dR.getBuffer();

        // assign kernel arguments
        int pos = 0;
        int pos_idxstart = -1;
        int pos_storeres = -1;

        auto svbuf = args.c_stencilValues.getBuf();
        int svgh = args.c_stencilValues.getGh();
        int svmgh = args.c_stencilValues.getMgh();
        int svngh = args.c_stencilValues.getNgh();
        int svogh = args.c_stencilValues.getOgh();
        int svGridSize = svmgh * svngh * svogh;
        int locmem_size = (args.wgsize.x + 4) * (args.wgsize.y + 4) * 5; // 5 planes in local memory

        err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &dVIn);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dVOut);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dF);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dR);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &svbuf);
        err |= clSetKernelArg(kernel, ++pos, sizeof(double) * locmem_size, NULL);
        err |= clSetKernelArg(kernel, ++pos, sizeof(double), &args.omega);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &m);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &n);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &o);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &mgh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ngh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ogh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svmgh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svngh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svogh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &args.ghosts);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svgh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svGridSize);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &idx_start);
        pos_idxstart = pos;
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &store_res);
        pos_storeres = pos;
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &args.num_x_planes);
        mgcl::mgclCheckError(err, "Setting kernel arguments");

        // One work-item per real grid point in a yz-plane
        // TODO adjust when not streaming along whole x-dim
        size_t num_dim_x_wis = m / args.num_x_planes;
        size_t global[3] = {num_dim_x_wis, static_cast<size_t>(n), static_cast<size_t>(o)};
        size_t local[3] = {1, args.wgsize.x, args.wgsize.y};

        // No padding of work-items. Instead, sum of wgs must match global grid size
        assert((n / args.wgsize.x) * args.wgsize.x == n && "n is not divisible by wgsize.x!");
        assert((o / args.wgsize.y) * args.wgsize.y == o && "o is not divisible by wgsize.y!");
        assert((m / args.num_x_planes) * args.num_x_planes == m && "m is not divisible by num_x_planes!");

        err = clEnqueueNDRangeKernel(args.commands, kernel, 3, NULL, global, local, 0, NULL, &ev);
        mgcl::mgclCheckError(err, "Enqueueing kernel");

        if (args.pd)
        {
            args.pd->addMeasurement(args.commands, ev, kernelName,
                                    {global[0], global[1], global[2]},
                                    {local[0], local[1], local[2]});
        }
        mgcl::mgclCheckError(clReleaseEvent(ev), "clReleaseEvent");

        clReleaseKernel(kernel);
    };

    // Benchs the various residual fixed stencil kernel versions
    TEST_CASE("jacobi2itersTemporalBlocking")
    {
        using std::min;

        if (CLI_ARGS::grids.size() == 0 && (CLI_ARGS::gridsMin.size() == 0 || CLI_ARGS::gridsMax.size() == 0))
            throw "Need to specify at least one local grid size, e.g. using --grids 4,8,16 or --gridsMin 4,4,4 AND --gridsMax 32,32,32";

        // build grids to be tested from CLI args
        std::vector<std::vector<int>> gridsTBT;
        for (auto N : CLI_ARGS::grids)
            gridsTBT.push_back({N, N, N});
        if (CLI_ARGS::gridsMin.size() > 0 && CLI_ARGS::gridsMax.size() > 0)
            for (int m = CLI_ARGS::gridsMin[0]; m <= CLI_ARGS::gridsMax[0]; m *= 2)
                for (int n = CLI_ARGS::gridsMin[1]; n <= CLI_ARGS::gridsMax[1]; n *= 2)
                    for (int o = CLI_ARGS::gridsMin[2]; o <= CLI_ARGS::gridsMax[2]; o *= 2)
                        gridsTBT.push_back({m, n, o});

        std::vector<bench_util::ResultJacobiTempBlock> results;

        // // Create dummy problem to initialize OpenCL
        // auto v_dummy = std::make_shared<mgcl::Cuboid>(1, 1, 1);
        // auto f_dummy = std::make_shared<mgcl::Cuboid>(1, 1, 1);
        // mgcl::Problem p(1, 1, 1, f_dummy, v_dummy);
        // p.setKernelFile("kernel_optimizations.cl");
        // if (CLI_ARGS::useBinaryFile)
        // {
        //     p.setBinaryFile("jacobiTempBlockingBench.bin");
        // }
        // p.setUseOpencl(true);
        // p.setDeviceType(CL_DEVICE_TYPE_GPU);
        // p.setProfilingEnabled(CLI_ARGS::enableKernelProfiling);
        // p.init();

        int ghosts = 1;

        std::vector<std::vector<size_t>> wg_sizes{{4, 4}, {4, 8}, {8, 8}, {8, 16}, {16, 16}, {4, 32}, {8, 32}, {4, 64}, {2, 32}, {2, 64}};
        std::vector<size_t> num_x_planes_divisors{1, 2, 4, 8}; // divisors for num_x_planes, i.e. num_x_planes = m / num_x_planes_divisor

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
            p.setJacobiIterationsPerKernel(1);
            p.setKernelFile("kernel_optimizations.cl");
            if (CLI_ARGS::useBinaryFile)
            {
                p.setBinaryFile("jacobiTempBlockingBench.bin");
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

            std::unique_ptr<mgcl::Cuboid> r_out_globmem_1iter_per_call = nullptr;
            // std::unique_ptr<mgcl::Cuboid> r_out_global_coeffs_without_ghosts = nullptr;
            std::unique_ptr<mgcl::Cuboid> r_out_global_coeffs_4_gp_per_thread = nullptr;
            if (CLI_ARGS::checkResults)
            {
                bench.epochs(1).epochIterations(1);
            }

            {
                std::string name = std::string("jacobi_globmem_one_iter_per_kernel_call_mxnxno")
                                       .append(std::to_string(m))
                                       .append("x")
                                       .append(std::to_string(n))
                                       .append("x")
                                       .append(std::to_string(o));

                bench.run(std::string(name).c_str(), [&] { //
                    mgcl::MultigridEngine::jacobi(p, lv0, 2, false, 1);
                    p.finish();
                });

                bench_util::ResultJacobiTempBlock res;
                res.name = name;
                res.minTime = bench_util::getMinTime(bench, name);
                res.medianTime = bench_util::getMedianTime(bench, name);
                res.avgTime = bench_util::getAvgTime(bench, name);
                res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
                res.m = m;
                res.n = n;
                res.o = o;
                const auto& c = mgcl::conf::getWorkGroupSizeForKernelAndWiCount(p.getKernelConfig(), "jacobi_iter_27point_varying_stencil_1d", 1);
                res.wgx = c[0];
                res.wgy = c[1];
                res.wgz = c[2];
                res.num_x_planes = 0;
                results.push_back(res);

                // if (CLI_ARGS::checkResults)
                // {
                //     r_out_globmem_1iter_per_call = std::make_unique<mgcl::Cuboid>(m, n, o, ghosts, ghosts, ghosts);
                //     lv0.getDVIn().read(p.getCommands(), r_out_globmem_1iter_per_call.get(), true);
                // }
            }

            {
                for (auto wg : wg_sizes)
                {
                    for (auto xpdiv : num_x_planes_divisors)
                    {

                        size_t wg_size_x = wg[0]; // GENERATE(4, 8);
                        size_t wg_size_y = wg[1];
                        int grid_size = mgh * ngh * ogh;
                        int num_x_planes = m / xpdiv;

                        JacobiTBArgs args{
                            false,
                            m,
                            n,
                            o,
                            mgh,
                            ngh,
                            ogh,
                            h2,
                            ghosts,
                            omega,
                            num_x_planes,
                            p.getProgram(),
                            p.getCommands(),
                            {wg_size_x, wg_size_y},
                            // c_dVIn,
                            // c_dVOut,
                            // c_dF,
                            // c_dR,
                            // *c_dSv,
                            lv0.getDVIn(),
                            lv0.getDVOut(),
                            lv0.getDF(),
                            lv0.getDR(),
                            *lv0.getStencilValuesGpu(),
                            p.getProfilingData(),
                            0,
                            0,
                            0};

                        // m must be divisable by num_x_planes_divisor and n,o must be divisable by wg_size
                        if ((m / num_x_planes) * num_x_planes != m || (n / args.wgsize.x) * args.wgsize.x != n || (o / args.wgsize.y) * args.wgsize.y != o)
                        {
                            continue;
                        }

                        std::string name = std::string("jacobi_tempblock_2iters_wg")
                                               .append(std::to_string(wg_size_x))
                                               .append("x")
                                               .append(std::to_string(wg_size_y))
                                               .append("_num-x-planes")
                                               .append(std::to_string(num_x_planes))
                                               .append("_mxnxo")
                                               .append(std::to_string(m))
                                               .append("x")
                                               .append(std::to_string(n))
                                               .append("x")
                                               .append(std::to_string(o));

                        bench.run(std::string(name).c_str(), [&] { //
                            // Update ghosts before and after to match globmem jacobi more closely
                            int err = mgcl::MultigridEngine::updateGhosts(p, args.c_dVIn,
                                                                          nullptr, true);
                            mgcl::mgclCheckError(err, "Updating ghosts");

                            jacobi_ocl_tb_2iters(args);

                            err = mgcl::MultigridEngine::updateGhosts(p, args.c_dVOut,
                                                                      nullptr, true);
                            mgcl::mgclCheckError(err, "Updating ghosts");
                            p.finish();
                        });

                        bench_util::ResultJacobiTempBlock res;
                        res.name = name;
                        res.minTime = bench_util::getMinTime(bench, name);
                        res.medianTime = bench_util::getMedianTime(bench, name);
                        res.avgTime = bench_util::getAvgTime(bench, name);
                        res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
                        res.m = m;
                        res.n = n;
                        res.o = o;
                        res.wgx = 1;
                        res.wgy = wg_size_x;
                        res.wgz = wg_size_y;
                        res.num_x_planes = num_x_planes;
                        results.push_back(res);

                        // if (CLI_ARGS::checkResults)
                        // {
                        //     r_out_globmem_1iter_per_call = std::make_unique<mgcl::Cuboid>(m, n, o, ghosts, ghosts, ghosts);
                        //     lv0.getDVIn().read(p.getCommands(), r_out_globmem_1iter_per_call.get(), true);
                        // } }
                    }
                }

                // Check results for kernels that it is valid for
                // if (CLI_ARGS::checkResults)
                // {
                //     REQUIRE(r_out_global_coeffs_first->isEqual(*r_out_global_coeffs_4_gp_per_thread));
                // }
                if (CLI_ARGS::enableKernelProfiling)
                {
                    p.getProfilingData()->printBestTimingsPerKernel();
                }
            }

            // if (CLI_ARGS::enableKernelProfiling)
            // {
            //     p.getProfilingData()->printBestTimingsPerKernel();
            // }
        }
        bench_util::printCsvFormat(results);
    }
}