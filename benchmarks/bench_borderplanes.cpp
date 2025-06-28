/**
 * @file bench_borderplanes.cpp
 * @brief Contains benchmarks for extract_border_planes and paste_from_ghosts.
 * @date 26.06.2025
 *
 */
#include "nanobench.h"

#include "catch2/catch_test_macros.hpp"

#include <chrono>
#include <iostream>
#include <vector>
using namespace std::chrono_literals;

#include "../src/mgcl/cuboid.hpp"
#include "../src/mgcl/mpi_util.hpp"
#include "../src/mgcl/multigrid_engine.hpp"
#include "../src/mgcl/opencl_helper.hpp"
#include "../src/mgcl/problem.hpp"
#include "bench_util.hpp"
#include "cli_args.hpp"

// Benchmarks extract ghosts kernel plus data transfer between host and device without MPI.
// If called with multiple processes, only root process runs the benchmarks.
// These benchmarks seem to be very unstable.
// Run with e.g.: benchmarks bench_data_transfer_host_device
TEST_CASE("bench_borderplanes")
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

    std::vector<bench_util::Result> results;

    // Create a dummy problem
    auto v = std::make_shared<mgcl::Cuboid>(1, 1, 1);
    auto f = std::make_shared<mgcl::Cuboid>(1, 1, 1);
    mgcl::Problem p(1, 1, 1, f, v);
    p.setUseOpencl(true);
    p.setDeviceType(CL_DEVICE_TYPE_GPU);
    p.setKernelFile("borderplanes_kernels.cl");
    if (CLI_ARGS::useBinaryFile)
    {
        p.setBinaryFile("benchBorderPlanes.bin");
    }
    p.setProfilingEnabled(CLI_ARGS::enableKernelProfiling);
    p.setSilent(true);
    p.init();

    bool printedGpu = false;
    for (auto gr : gridsTBT)
    {
        int m = gr[0];
        int n = gr[1];
        int o = gr[2];
        int ghosts = 1;
        int mgh = m + 2 * ghosts;
        int ngh = n + 2 * ghosts;
        int ogh = o + 2 * ghosts;

        int err;

        // actual test buffers
        mgcl::Cuboid c_h(m, n, o, ghosts, ghosts, ghosts);
        c_h.fillRandom();
        mgcl::CuboidGpu c_d(p.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, c_h);

        int yz = c_d.getNgh() * c_d.getOgh();
        int xz = c_d.getMgh() * c_d.getOgh();
        int xy = c_d.getMgh() * c_d.getNgh();
        int ressize = 2 * yz * c_d.getGhostsM() + 2 * xz * c_d.getGhostsN() + 2 * xy * c_d.getGhostsO();
        mgcl::BufferGpu d_planesbuf(p.getContext(), CL_MEM_READ_ONLY, ressize);

        ankerl::nanobench::Bench bench;
        bench.timeUnit(1ms, "ms")
            .epochs(CLI_ARGS::bench_epochs)
            .epochIterations(CLI_ARGS::bench_iterations);

        {
            std::string name = std::string("extractBorderPlanes_newret_")
                                   .append(std::to_string(m))
                                   .append("_")
                                   .append(std::to_string(n))
                                   .append("_")
                                   .append(std::to_string(o))
                                   .append("_gh")
                                   .append(std::to_string(ghosts));

            bench.run(std::string(name).c_str(), [&] { //
                auto sbuf_ptr = c_d.extractBorderPlanes(p.getCommands(), p.getProgram(), &d_planesbuf, nullptr, nullptr, p.getProfilingData());
                p.getOpenCLHelper().finish();
            });

            bench_util::Result res;
            res.name = name;
            res.minTime = bench_util::getMinTime(bench, name);
            res.medianTime = bench_util::getMedianTime(bench, name);
            res.avgTime = bench_util::getAvgTime(bench, name);
            res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
            res.m = m;
            res.n = n;
            res.o = o;
            results.push_back(res);
        }

        {
            std::vector<double> h_planesbuf(ressize);
            std::string name = std::string("extractBorderPlanes_reuseret_")
                                   .append(std::to_string(m))
                                   .append("_")
                                   .append(std::to_string(n))
                                   .append("_")
                                   .append(std::to_string(o))
                                   .append("_gh")
                                   .append(std::to_string(ghosts));

            bench.run(std::string(name).c_str(), [&] { //
                c_d.extractBorderPlanes(p.getCommands(), p.getProgram(), &d_planesbuf, &h_planesbuf, nullptr, p.getProfilingData());
                p.getOpenCLHelper().finish();
            });

            bench_util::Result res;
            res.name = name;
            res.minTime = bench_util::getMinTime(bench, name);
            res.medianTime = bench_util::getMedianTime(bench, name);
            res.avgTime = bench_util::getAvgTime(bench, name);
            res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
            res.m = m;
            res.n = n;
            res.o = o;
            results.push_back(res);
        }

        {
            std::vector<double> rbuf(ressize);
            std::string name = std::string("pasteGhostsFromBorderPlanes_newdevicebuf_N")
                                   .append(std::to_string(m))
                                   .append("_")
                                   .append(std::to_string(n))
                                   .append("_")
                                   .append(std::to_string(o))
                                   .append("_gh")
                                   .append(std::to_string(ghosts));

            bench.run(std::string(name).c_str(), [&] { //
                c_d.pasteGhostsFromBorderPlanes(p.getContext(), p.getCommands(), p.getProgram(), nullptr, &rbuf, nullptr, p.getProfilingData());
                p.getOpenCLHelper().finish();
            });

            bench_util::Result res;
            res.name = name;
            res.minTime = bench_util::getMinTime(bench, name);
            res.medianTime = bench_util::getMedianTime(bench, name);
            res.avgTime = bench_util::getAvgTime(bench, name);
            res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
            res.m = m;
            res.n = n;
            res.o = o;
            results.push_back(res);
        }

        {
            std::string name = std::string("pasteGhostsFromBorderPlanes_reusedevicebuf_N")
                                   .append(std::to_string(m))
                                   .append("_")
                                   .append(std::to_string(n))
                                   .append("_")
                                   .append(std::to_string(o))
                                   .append("_gh")
                                   .append(std::to_string(ghosts));

            bench.run(std::string(name).c_str(), [&] { //
                c_d.pasteGhostsFromBorderPlanes(p.getContext(), p.getCommands(), p.getProgram(), &d_planesbuf, nullptr, nullptr, p.getProfilingData());
                p.getOpenCLHelper().finish();
            });

            bench_util::Result res;
            res.name = name;
            res.minTime = bench_util::getMinTime(bench, name);
            res.medianTime = bench_util::getMedianTime(bench, name);
            res.avgTime = bench_util::getAvgTime(bench, name);
            res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
            res.m = m;
            res.n = n;
            res.o = o;
            results.push_back(res);
        }
    }

    bench_util::printCsvFormat(results);

    if (CLI_ARGS::enableKernelProfiling)
    {
        p.getProfilingData()->printBestTimingsPerKernel();
    }
}
