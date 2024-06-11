#include "nanobench.h"

#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"

#include <chrono>
using namespace std::chrono_literals;

#include "../src/mgcl/stencil.hpp"
#include "../test/test_utility.hpp"

// TODO implement + add to cmake
TEST_CASE("stencil arithmetic", "[console][fixedVsVarying]")
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
