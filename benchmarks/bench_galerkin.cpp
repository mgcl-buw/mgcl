#include "nanobench.h"

#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"

#include <chrono>
#include <fstream>
#include <functional> // for function
#include <iostream>
#include <vector>
using namespace std::chrono_literals;

#include "../src/cuboid.hpp"
#include "../src/problem.hpp"
#include "../test/test_utility.hpp"
#include "bench_render_templates.hpp"

TEST_CASE("galerkin init vs solve", "[console][galerkinInitVsSolve]")
{
    // int N = GENERATE(8, 16, 32, 64);
    int N = 64;
    int m = N;
    int n = N;
    int o = N;

    int vcycleIters = 30;

    ankerl::nanobench::Bench b;
    b.timeUnit(1ns, "ns")
        // .epochs(1)
        // .epochIterations(1)
        .minEpochTime(100ms)
        .relative(false);
    // .warmup(1); // especially for init OpenCL environment

    bool gpuAvailable = mgcl_test::TestUtility::deviceAvailable("", CL_DEVICE_TYPE_GPU);
    // bool cpuAvailable = tu->deviceAvailable("", CL_DEVICE_TYPE_CPU);

    SECTION(std::string("N = ").append(std::to_string(N)).c_str())
    {
        {
            // CPU fixed Laplace stencil
            auto v = std::make_shared<mgcl::Cuboid>(m, n, o);
            auto f = std::make_shared<mgcl::Cuboid>(m, n, o);
            v->fillRandom(0, 10);
            f->fillRandom(0, 10);

            std::string name = std::string("seq fixed 7p Laplace, N = ")
                                   .append(std::to_string(N))
                                   .append(", iters: ")
                                   .append(std::to_string(vcycleIters));

            b.run(std::string(name).append(", overhead").c_str(), [&]
                  {
                      auto p = new mgcl::Problem(m, n, o, f, v);
                      p->setSilent(true);
                      p->setMaxiterVcycles(vcycleIters);
                      // p.init();
                      delete p; //
                  });

            b.run(std::string(name).append(", init").c_str(), [&]
                  {
                      auto p = new mgcl::Problem(m, n, o, f, v);
                      p->setSilent(true);
                      p->setMaxiterVcycles(vcycleIters);
                      p->init();
                      delete p; //
                  });

            b.run(std::string(name).append(", init+solve").c_str(), [&]
                  {
                      auto p = new mgcl::Problem(m, n, o, f, v);
                      p->setSilent(true);
                      p->setMaxiterVcycles(vcycleIters);
                      p->solve();
                      delete p; //
                  });
        }

        {
            // CPU varying stencil (Galerkin)
            auto v = std::make_shared<mgcl::Cuboid>(m, n, o);
            auto f = std::make_shared<mgcl::Cuboid>(m, n, o);
            v->fillRandom(0, 10);
            f->fillRandom(0, 10);

            // mgcl::Problem p(m, n, o, f, v);
            // p.setSilent(true);
            // p.setStencilType(mgcl::MGCL_VARYING);
            // auto &s = *p.getStencilValues();

            std::function<void(mgcl::VaryingStencil3x3x3 &)> fillStencil = [N](mgcl::VaryingStencil3x3x3 &s)
            {
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
            };

            std::string name = std::string("seq varying 7p Laplace, N = ")
                                   .append(std::to_string(N))
                                   .append(", iters: ")
                                   .append(std::to_string(vcycleIters));

            b.run(std::string(name).append(", overhead").c_str(), [&]
                  {
                      auto p = new mgcl::Problem(m, n, o, f, v);
                      p->setSilent(true);
                      p->setMaxiterVcycles(vcycleIters);
                      p->setStencilType(mgcl::MGCL_VARYING);
                      fillStencil(*p->getStencilValues());

                      // p.init();
                      delete p; //
                  });

            if (N >= 32)
                b.epochs(1).epochIterations(1);

            b.run(std::string(name).append(", init").c_str(), [&]
                  {
                      auto p = new mgcl::Problem(m, n, o, f, v);
                      p->setSilent(true);
                      p->setMaxiterVcycles(vcycleIters);
                      p->setStencilType(mgcl::MGCL_VARYING);
                      fillStencil(*p->getStencilValues());

                      p->init();
                      delete p; //
                  });

            b.run(std::string(name).append(", init+solve").c_str(), [&]
                  {
                      auto p = new mgcl::Problem(m, n, o, f, v);
                      p->setSilent(true);
                      p->setMaxiterVcycles(vcycleIters);
                      p->setStencilType(mgcl::MGCL_VARYING);
                      fillStencil(*p->getStencilValues());

                      p->solve();
                      delete p; //
                  });

            if (N >= 32)
                b.epochs(11).epochIterations(0);
        }

        // if (gpuAvailable)
        // {
        //     // OpenCL fixed Laplace stencil
        //     auto v = std::make_shared<mgcl::Cuboid>(m, n, o);
        //     auto f = std::make_shared<mgcl::Cuboid>(m, n, o);
        //     v->fillRandom(0, 10);
        //     f->fillRandom(0, 10);

        //     auto p = new mgcl::Problem(m, n, o, f, v);
        //     p->setSilent(true);
        //     p->setUseOpencl(true);

        //     std::string name = std::string("ocl fixed 7p Laplace, N = ")
        //                            .append(std::to_string(N))
        //                            .append(", iters: ")
        //                            .append(std::to_string(p->getMaxiterVcycles()));

        //     bool ret;
        //     b.run(std::string(name).append(", init").c_str(), [&]
        //           {
        //               ret = p->init();
        //               clFinish(p->getCommands());
        //               p->reset();
        //               // delete p;
        //           });
        //     REQUIRE(ret);

        //     p = new mgcl::Problem(m, n, o, f, v);
        //     p->setSilent(true);
        //     p->setUseOpencl(true);

        //     b.run(std::string(name).append(", init+solve").c_str(), [&]
        //           {
        //               p->solve();
        //               clFinish(p->getCommands());
        //               p->reset();
        //               // delete p;
        //           });
        // }

        // if (gpuAvailable)
        // {
        //     // OpenCL varying stencil (Galerkin)
        //     auto v = std::make_shared<mgcl::Cuboid>(m, n, o);
        //     auto f = std::make_shared<mgcl::Cuboid>(m, n, o);
        //     v->fillRandom(0, 10);
        //     f->fillRandom(0, 10);

        //     mgcl::Problem p(m, n, o, f, v);
        //     p.setSilent(true);
        //     p.setStencilType(mgcl::MGCL_VARYING);
        //     p.setUseOpencl(true);
        //     auto &s = *p.getStencilValues();

        //     // Fill with 7-point Laplace, which is also used by the other two Sections in this test case
        //     for (int i = 0; i < N; i++)
        //         for (int j = 0; j < N; j++)
        //             for (int k = 0; k < N; k++)
        //             {
        //                 // 7-point Laplace
        //                 s[i][j][k][0][1][1] = 1;
        //                 s[i][j][k][1][0][1] = 1;
        //                 s[i][j][k][1][1][0] = 1;
        //                 s[i][j][k][1][1][1] = -6;
        //                 s[i][j][k][1][1][2] = 1;
        //                 s[i][j][k][1][2][1] = 1;
        //                 s[i][j][k][2][1][1] = 1;
        //             }

        //     std::string name = std::string("ocl varying 7p Laplace, N = ")
        //                            .append(std::to_string(N))
        //                            .append(", iters: ")
        //                            .append(std::to_string(p.getMaxiterVcycles()));

        //     // if (N >= 32)
        //     //     b.epochs(1).epochIterations(1);

        //     bool ret;
        //     b.run(std::string(name).append(", init").c_str(), [&]
        //           {
        //             ret = p.init();
        //             clFinish(p.getCommands()); });
        //     REQUIRE(ret);

        //     b.run(std::string(name).append(", init+solve").c_str(), [&]
        //           {
        //             p.solve();
        //             clFinish(p.getCommands()); });

        //     // if (N >= 32)
        //     //     b.epochs(11).epochIterations(0);
        // }
    }
}
