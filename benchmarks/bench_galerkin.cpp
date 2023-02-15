#include "nanobench.h"

#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
using namespace std::chrono_literals;

#include "../src/cuboid.hpp"
#include "../src/problem.hpp"
#include "../test/test_utility.hpp"
#include "bench_render_templates.hpp"

TEST_CASE("galerkin init vs solve", "[console][galerkinInitVsSolve]")
{
    int N = GENERATE(16, 32, 64);
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

    // auto tu = new mgcl_test::TestUtility();
    // bool gpuAvailable = tu->deviceAvailable("", CL_DEVICE_TYPE_GPU);
    // bool cpuAvailable = tu->deviceAvailable("", CL_DEVICE_TYPE_CPU);

    SECTION(std::string("N = ").append(std::to_string(N)).c_str())
    {
        {
            auto v = std::make_shared<mgcl::Cuboid>(m, n, o);
            auto f = std::make_shared<mgcl::Cuboid>(m, n, o);
            v->fillRandom(0, 10);
            f->fillRandom(0, 10);

            mgcl::Problem p(m, n, o, f, v);
            p.setSilent(true);

            std::string name = std::string("seq fixed 7p Laplace, N = ")
                                   .append(std::to_string(N))
                                   .append(", iters: ")
                                   .append(std::to_string(p.getMaxiterVcycles()));

            bool ret;
            b.run(std::string(name).append(", init").c_str(), [&]
                  { ret = p.init(); });
            REQUIRE(ret);

            b.run(std::string(name).append(", init+solve").c_str(), [&]
                  { p.solve(); });
        }

        {
            auto v = std::make_shared<mgcl::Cuboid>(m, n, o);
            auto f = std::make_shared<mgcl::Cuboid>(m, n, o);
            v->fillRandom(0, 10);
            f->fillRandom(0, 10);

            mgcl::Problem p(m, n, o, f, v);
            p.setSilent(true);
            p.setStencilType(mgcl::MGCL_VARYING_7POINT);
            auto &s = *p.getStencilValues();

            // Fill with 7-point Laplace, which is also used by the other two Sections in this test case
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    for (int k = 0; k < N; k++)
                    {
                        // 7-point Laplace
                        s[i][j][k][0][1][1] = 1;
                        s[i][j][k][1][0][1] = 1;
                        s[i][j][k][1][1][0] = 1;
                        s[i][j][k][1][1][1] = -6;
                        s[i][j][k][1][1][2] = 1;
                        s[i][j][k][1][2][1] = 1;
                        s[i][j][k][2][1][1] = 1;
                    }

            std::string name = std::string("seq varying 7p Laplace, N = ")
                                   .append(std::to_string(N))
                                   .append(", iters: ")
                                   .append(std::to_string(p.getMaxiterVcycles()));

            if (N >= 32)
                b.epochs(1).epochIterations(1);

            bool ret;
            b.run(std::string(name).append(", init").c_str(), [&]
                  { ret = p.init(); });
            REQUIRE(ret);

            b.run(std::string(name).append(", init+solve").c_str(), [&]
                  { p.solve(); });

            if (N >= 32)
                b.epochs(11).epochIterations(0);
        }
    }
}
