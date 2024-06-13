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

// Benchs clEnqueueFillBuffer vs custom fill buffer
TEST_CASE("bench_fill_custom_vs_api")
{
    using std::min;

    if (CLI_ARGS::elements.size() == 0)
        throw "Need to specify at least one element count, e.g. using --elements 16384,32768";

    std::cout << "Testing the following element counts" << std::endl
              << "  ";
    for (auto e : CLI_ARGS::elements)
    {
        std::cout << e << ", ";
    }
    std::cout << std::endl;

    std::vector<bench_util::Result1d> results;
    bool printedGpu = false;

    // Create dummy test problem
    auto v = std::make_shared<mgcl::Cuboid>(1, 1, 1);
    auto f = std::make_shared<mgcl::Cuboid>(1, 1, 1);
    mgcl::Problem p(1, 1, 1, f, v);
    p.setUseOpencl(true);
    p.setDeviceType(CL_DEVICE_TYPE_GPU);
    p.init();

    for (int el : CLI_ARGS::elements)
    {
        mgcl::CuboidGpu cbuf(p.getContext(), CL_MEM_READ_WRITE, 1, 1, el, 0, 0, 0, 0);
        cl_mem buf = cbuf.getBuffer();

        double one = 1.0;

        ankerl::nanobench::Bench bench;
        bench.timeUnit(1ms, "ms")
            .epochs(CLI_ARGS::bench_epochs)
            .epochIterations(CLI_ARGS::bench_iterations)
            // .minEpochTime(100ms)
            .relative(CLI_ARGS::jacobiIters.size() > 1);

        {
            std::string name = std::string("api_")
                                   .append(std::to_string(el));
            bench.run(std::string(name).c_str(), [&] { //
                clEnqueueFillBuffer(p.getCommands(), buf, &one, sizeof(double), 0, el, 0, NULL, NULL);
                p.getOpenCLHelper().finish();
            });

            bench_util::Result1d res;
            res.name = name;
            res.minTime = bench_util::getMinTime(bench, name);
            res.medianTime = bench_util::getMedianTime(bench, name);
            res.avgTime = bench_util::getAvgTime(bench, name);
            res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
            res.elements = el;
            results.push_back(res);
        }

        {
            std::string name = std::string("custom_")
                                   .append(std::to_string(el));
            bench.run(std::string(name).c_str(), [&] { //
                cbuf.fill(p.getProgram(), p.getCommands(), 1.0, false, nullptr, nullptr);
                p.getOpenCLHelper().finish();
            });

            bench_util::Result1d res;
            res.name = name;
            res.minTime = bench_util::getMinTime(bench, name);
            res.medianTime = bench_util::getMedianTime(bench, name);
            res.avgTime = bench_util::getAvgTime(bench, name);
            res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
            res.elements = el;
            results.push_back(res);
        }
    }

    bench_util::printCsvFormat(results);
}