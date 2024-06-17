#include "cli_args.hpp"
#include "nanobench.h"

#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"

#include <chrono>
#include <iostream>
using namespace std::chrono_literals;

#include "../src/mgcl/stencil.hpp"
#include "../test/test_utility.hpp"

TEST_CASE("stencil_arithmetic")
{
    int N = GENERATE(8, 16, 32, 64);
    // int N = 16;
    int m = N;
    int n = N;
    int o = N;

    ankerl::nanobench::Bench b;
    b.timeUnit(1ms, "ms")
        // .epochs(1)
        // .epochIterations(1)
        .minEpochTime(100ms)
        .relative(false);

    if (N >= 32)
        b.epochs(1).epochIterations(2);

    if (N >= 64)
        b.epochs(1).epochIterations(1);

    auto tu = std::make_shared<mgcl_test::TestUtility>();
    bool gpuAvailable = tu->deviceAvailable("Quadro", CL_DEVICE_TYPE_GPU);
    bool cpuAvailable = tu->deviceAvailable("", CL_DEVICE_TYPE_CPU);

    SECTION(std::string("N = ").append(std::to_string(N)).c_str())
    {
        int gh = 1;

        {
            mgcl::VaryingStencil v(m, n, o, 3, gh, gh, gh);
            mgcl::VaryingStencil f(m, n, o, 3, gh, gh, gh);
            v.fillRandom(0, 10);
            f.fillRandom(0, 10);

            std::string name = std::string("seq var*var, N = ")
                                   .append(std::to_string(N));

            b.run(std::string(name).c_str(), [&]
                  { ankerl::nanobench::doNotOptimizeAway(v.multiply(f, 0, nullptr, false, true)); });
        }

        {
            mgcl::FixedStencil v(3);
            mgcl::VaryingStencil f(m, n, o, 3, gh, gh, gh);
            v.fillRandom(0, 10);
            f.fillRandom(0, 10);

            std::string name = std::string("seq fix*var, N = ")
                                   .append(std::to_string(N));

            b.run(std::string(name).c_str(), [&]
                  { ankerl::nanobench::doNotOptimizeAway(v.multiply(f, 0, nullptr, false, true)); });
        }

        {
            mgcl::FixedStencil v(3);
            mgcl::VaryingStencil f(m, n, o, 3, gh, gh, gh);
            v.fillRandom(0, 10);
            f.fillRandom(0, 10);

            std::string name = std::string("seq var*fix, N = ")
                                   .append(std::to_string(N));

            b.run(std::string(name).c_str(), [&]
                  { ankerl::nanobench::doNotOptimizeAway(f.multiply(v, 0, nullptr, false, true)); });
        }

        if (gpuAvailable)
        {
            mgcl::VaryingStencil v(m, n, o, 3, gh, gh, gh);
            mgcl::VaryingStencil f(m, n, o, 3, gh, gh, gh);
            v.fillRandom(0, 10);
            f.fillRandom(0, 10);

            mgcl::VaryingStencilGpu vd(m, n, o, 3, gh, tu->getContext(), tu->getCommands(), tu->getProgram());
            mgcl::VaryingStencilGpu fd(m, n, o, 3, gh, tu->getContext(), tu->getCommands(), tu->getProgram());
            vd.fill(v, tu->getCommands(), true);
            fd.fill(f, tu->getCommands(), true);
            tu->finish();

            std::string name = std::string("gpu var*var, N = ")
                                   .append(std::to_string(N));

            b.run(std::string(name).c_str(), [&]
                  { 
                    ankerl::nanobench::doNotOptimizeAway(vd.multiply(fd, 0, tu->getProgram(), tu->getCommands(), tu->getContext(), nullptr, true, true, nullptr, nullptr)); 
                    tu->finish(); });
        }

        if (gpuAvailable)
        {
            mgcl::FixedStencil v(3);
            mgcl::VaryingStencil f(m, n, o, 3, gh, gh, gh);
            v.fillRandom(0, 10);
            f.fillRandom(0, 10);

            mgcl::FixedStencilGpu vd(3, tu->getContext(), tu->getCommands(), tu->getProgram());
            mgcl::VaryingStencilGpu fd(m, n, o, 3, gh, tu->getContext(), tu->getCommands(), tu->getProgram());
            vd.fill(v, tu->getCommands(), true);
            fd.fill(f, tu->getCommands(), true);
            tu->finish();

            std::string name = std::string("gpu fix*var, N = ")
                                   .append(std::to_string(N));

            b.run(std::string(name).c_str(), [&]
                  { 
                    ankerl::nanobench::doNotOptimizeAway(vd.multiply(fd, 0, tu->getProgram(), tu->getCommands(), tu->getContext(), nullptr, true, true, nullptr, nullptr)); 
                    tu->finish(); });
        }

        if (gpuAvailable)
        {
            mgcl::FixedStencil v(3);
            mgcl::VaryingStencil f(m, n, o, 3, gh, gh, gh);
            v.fillRandom(0, 10);
            f.fillRandom(0, 10);

            mgcl::FixedStencilGpu vd(3, tu->getContext(), tu->getCommands(), tu->getProgram());
            mgcl::VaryingStencilGpu fd(m, n, o, 3, gh, tu->getContext(), tu->getCommands(), tu->getProgram());
            vd.fill(v, tu->getCommands(), true);
            fd.fill(f, tu->getCommands(), true);
            tu->finish();

            std::string name = std::string("gpu var*fix, N = ")
                                   .append(std::to_string(N));

            b.run(std::string(name).c_str(), [&]
                  { 
                    ankerl::nanobench::doNotOptimizeAway(fd.multiply(vd, 0, tu->getProgram(), tu->getCommands(), tu->getContext(), nullptr, true, true, nullptr, nullptr)); 
                    tu->finish(); });
        }
    }
}

TEST_CASE("stencil_arithmetic_ocl_only")
{
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

    for (auto gr : gridsTBT)
    {
        int m = gr[0];
        int n = gr[1];
        int o = gr[2];

        ankerl::nanobench::Bench b;
        b.timeUnit(1ms, "ms")
            .epochs(CLI_ARGS::bench_epochs)
            .epochIterations(CLI_ARGS::bench_iterations)
            .minEpochTime(100ms)
            .relative(false);

        auto tu = std::make_shared<mgcl_test::TestUtility>(CL_DEVICE_TYPE_GPU);
        bool gpuAvailable = tu->deviceAvailable("Quadro", CL_DEVICE_TYPE_GPU);

        if (!gpuAvailable)
            throw "No GPU found!";

        std::cout << "GPU info:" << std::endl;
        tu->getProblem().getOpenCLHelper().outputDeviceInfo(tu->getProblem().getOpenCLHelper().getDeviceId());

        std::string name_grid = std::to_string(m)
                                    .append("_")
                                    .append(std::to_string(n))
                                    .append("_")
                                    .append(std::to_string(o));

        int gh = 1;
        int width = 3;

        {
            mgcl::VaryingStencil v(m, n, o, width, gh, gh, gh);
            mgcl::VaryingStencil f(m, n, o, width, gh, gh, gh);
            v.fillRandom(0, 10);
            f.fillRandom(0, 10);

            mgcl::VaryingStencilGpu vd(m, n, o, width, gh, tu->getContext(), tu->getCommands(), tu->getProgram());
            mgcl::VaryingStencilGpu fd(m, n, o, width, gh, tu->getContext(), tu->getCommands(), tu->getProgram());
            vd.fill(v, tu->getCommands(), true);
            fd.fill(f, tu->getCommands(), true);
            tu->finish();

            std::string name = std::string("gpu var*var_").append(name_grid);
            b.run(std::string(name).c_str(), [&]
                  { 
                    ankerl::nanobench::doNotOptimizeAway(vd.multiply(fd, 0, tu->getProgram(), tu->getCommands(), tu->getContext(), nullptr, true, true, nullptr, nullptr)); 
                    tu->finish(); });
        }

        {
            mgcl::FixedStencil v(width);
            mgcl::VaryingStencil f(m, n, o, width, gh, gh, gh);
            v.fillRandom(0, 10);
            f.fillRandom(0, 10);

            mgcl::FixedStencilGpu vd(width, tu->getContext(), tu->getCommands(), tu->getProgram());
            mgcl::VaryingStencilGpu fd(m, n, o, width, gh, tu->getContext(), tu->getCommands(), tu->getProgram());
            vd.fill(v, tu->getCommands(), true);
            fd.fill(f, tu->getCommands(), true);
            tu->finish();

            std::string name = std::string("gpu fix*var_").append(name_grid);
            b.run(std::string(name).c_str(), [&]
                  { 
                    ankerl::nanobench::doNotOptimizeAway(vd.multiply(fd, 0, tu->getProgram(), tu->getCommands(), tu->getContext(), nullptr, true, true, nullptr, nullptr)); 
                    tu->finish(); });
        }

        {
            mgcl::FixedStencil v(width);
            mgcl::VaryingStencil f(m, n, o, width, gh, gh, gh);
            v.fillRandom(0, 10);
            f.fillRandom(0, 10);

            mgcl::FixedStencilGpu vd(width, tu->getContext(), tu->getCommands(), tu->getProgram());
            mgcl::VaryingStencilGpu fd(m, n, o, width, gh, tu->getContext(), tu->getCommands(), tu->getProgram());
            vd.fill(v, tu->getCommands(), true);
            fd.fill(f, tu->getCommands(), true);
            tu->finish();

            std::string name = std::string("gpu var*fix_").append(name_grid);
            b.run(std::string(name).c_str(), [&]
                  { 
                    ankerl::nanobench::doNotOptimizeAway(fd.multiply(vd, 0, tu->getProgram(), tu->getCommands(), tu->getContext(), nullptr, true, true, nullptr, nullptr)); 
                    tu->finish(); });
        }
    }
}
