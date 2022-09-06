#define ANKERL_NANOBENCH_IMPLEMENT
#include "nanobench.h"

#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"

#include <chrono>
#include <vector>
using namespace std::chrono_literals;

#include "../cuboid.hpp"
#include "../problem.hpp"
#include "test_utility.hpp"

TEST_CASE("mgcl benchmarks", "[!benchmark]")
{
    int N = GENERATE(16, 32, 64, 128);
    // int N = 16;
    int m = N;
    int n = N;
    int o = N;

    ankerl::nanobench::Bench b;
    b.timeUnit(1ms, "ms")
        .warmup(3)
        .minEpochTime(100ms)
        .relative(true);

    SECTION(std::string("N = ").append(std::to_string(N)).c_str())
    {
        {
            auto v = std::make_shared<mgcl::Cuboid>(m, n, o);
            auto f = std::make_shared<mgcl::Cuboid>(m, n, o);
            v->fillRandom(0, 10);
            f->fillRandom(0, 10);

            mgcl::Problem p(m, n, o, f, v);
            p.setSilent(true);
            p.init();

            b.run(std::string("sequential random values, N = ").append(std::to_string(N)).c_str(), [&]
                  { p.solveSeq(); });
        }

        {
            auto v = std::make_shared<mgcl::Cuboid>(m, n, o);
            auto f = std::make_shared<mgcl::Cuboid>(m, n, o);
            v->fillRandom(0, 10);
            f->fillRandom(0, 10);

            mgcl::Problem p(m, n, o, f, v);
            p.setUseOpencl(true);
            p.setDeviceType(CL_DEVICE_TYPE_GPU);
            p.setSilent(true);

            mgcl_test::TestUtility tu;
            if (tu.deviceAvailable("Quadro", p.getDeviceType()))
                p.setDeviceName("Quadro");

            p.init();
            b.run(std::string("sequential random values, N = ").append(std::to_string(N)).c_str(), [&]
                  { p.solve(); });
        }
    }
}
