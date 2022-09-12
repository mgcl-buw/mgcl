#include "nanobench.h"

#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"

#include <chrono>
#include <iostream>
#include <vector>
using namespace std::chrono_literals;

#include "../cuboid.hpp"
#include "../problem.hpp"
#include "../test/test_utility.hpp"

TEST_CASE("mgcl benchmarks console: init", "[!benchmark]")
{
    int N = GENERATE(16, 32, 64, 128);
    // int N = 16;
    int m = N;
    int n = N;
    int o = N;

    ankerl::nanobench::Bench b;
    b.timeUnit(1ms, "ms")
        .epochs(1)
        .epochIterations(1)
        // .minEpochTime(100ms)
        .relative(true);

    auto tu = new mgcl_test::TestUtility();
    bool gpuAvailable = tu->deviceAvailable("", CL_DEVICE_TYPE_GPU);
    bool cpuAvailable = tu->deviceAvailable("", CL_DEVICE_TYPE_CPU);

    SECTION(std::string("N = ").append(std::to_string(N)).c_str())
    {
        {
            auto v = std::make_shared<mgcl::Cuboid>(m, n, o);
            auto f = std::make_shared<mgcl::Cuboid>(m, n, o);
            v->fillRandom(0, 10);
            f->fillRandom(0, 10);

            mgcl::Problem p(m, n, o, f, v);
            p.setSilent(true);

            bool ret;
            b.run(std::string("sequential, N = ").append(std::to_string(N)).c_str(), [&]
                  { ret = p.init(); });
            REQUIRE(ret);
        }

        if (gpuAvailable)
        {
            auto v = std::make_shared<mgcl::Cuboid>(m, n, o);
            auto f = std::make_shared<mgcl::Cuboid>(m, n, o);
            v->fillRandom(0, 10);
            f->fillRandom(0, 10);

            mgcl::Problem p(m, n, o, f, v);
            p.setUseOpencl(true);
            p.setDeviceType(CL_DEVICE_TYPE_GPU);
            p.setSilent(true);

            bool ret;
            b.run(std::string("opencl GPU, N = ").append(std::to_string(N)).c_str(), [&]
                  { ret = p.init(); });
            REQUIRE(ret);
        }

        if (gpuAvailable)
        {
            auto v = std::make_shared<mgcl::Cuboid>(m, n, o);
            auto f = std::make_shared<mgcl::Cuboid>(m, n, o);
            v->fillRandom(0, 10);
            f->fillRandom(0, 10);

            mgcl::Problem p(m, n, o, f, v);
            p.setUseOpencl(true);
            p.setDeviceType(CL_DEVICE_TYPE_GPU);
            p.setSilent(true);

            // this is equal to Problem::reuseOpenCL for our matter
            REQUIRE(p.initOpenCL() == CL_SUCCESS);

            bool ret;
            b.run(std::string("opencl GPU reusing platform, N = ").append(std::to_string(N)).c_str(), [&]
                  { ret = p.init(); });
            REQUIRE(ret);
        }

        if (cpuAvailable)
        {
            auto v = std::make_shared<mgcl::Cuboid>(m, n, o);
            auto f = std::make_shared<mgcl::Cuboid>(m, n, o);
            v->fillRandom(0, 10);
            f->fillRandom(0, 10);

            mgcl::Problem p(m, n, o, f, v);
            p.setUseOpencl(true);
            p.setDeviceType(CL_DEVICE_TYPE_CPU);
            p.setSilent(true);

            bool ret;
            b.run(std::string("opencl CPU, N = ").append(std::to_string(N)).c_str(), [&]
                  { ret = p.init(); });
            REQUIRE(ret);
        }

        if (cpuAvailable)
        {
            auto v = std::make_shared<mgcl::Cuboid>(m, n, o);
            auto f = std::make_shared<mgcl::Cuboid>(m, n, o);
            v->fillRandom(0, 10);
            f->fillRandom(0, 10);

            mgcl::Problem p(m, n, o, f, v);
            p.setUseOpencl(true);
            p.setDeviceType(CL_DEVICE_TYPE_CPU);
            p.setSilent(true);

            // this is equal to Problem::reuseOpenCL for our matter
            REQUIRE(p.initOpenCL() == CL_SUCCESS);

            bool ret;
            b.run(std::string("opencl CPU reusing platform, N = ").append(std::to_string(N)).c_str(), [&]
                  { ret = p.init(); });
            REQUIRE(ret);
        }
    }
}
