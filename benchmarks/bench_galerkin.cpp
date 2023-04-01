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
    // Problem parameters
    double tol = 1e-20;
    int nu1 = 2;
    int nu2 = 2;
    double omega = 0.8;
    int vcycleIters = 30;

    std::cout << "Problem parameters:" << std::endl
              << "  tol: " << tol << std::endl
              << "  nu1: " << nu1 << std::endl
              << "  nu2: " << nu2 << std::endl
              << "  omega: " << omega << std::endl
              << "  v-cycle iterations: " << vcycleIters << std::endl;

    // int N = GENERATE(8, 16, 32, 64);
    int N = 64;
    int m = N;
    int n = N;
    int o = N;

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

            std::function<mgcl::Problem *()> createProblem = [m, n, o, &f, &v, vcycleIters]()
            {
                auto p = new mgcl::Problem(m, n, o, f, v);
                p->setSilent(true);
                p->setMaxiterVcycles(vcycleIters);
                return p;
            };

            std::string name = std::string("seq fixed 7p Laplace, N = ")
                                   .append(std::to_string(N))
                                   .append(", iters: ")
                                   .append(std::to_string(vcycleIters));

            // Increase epoch count for overhead and init
            auto tmp = b.minEpochIterations();
            b.minEpochIterations(300);

            b.run(std::string(name).append(", overhead (oh)").c_str(), [&]
                  {
                      auto p = createProblem();
                      delete p; //
                  });

            b.run(std::string(name).append(", init+oh").c_str(), [&]
                  {
                      auto p = createProblem();
                      p->init();
                      delete p; //
                  });

            b.minEpochIterations(tmp);

            b.run(std::string(name).append(", init+oh+solve").c_str(), [&]
                  {
                      auto p = createProblem();
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

            std::function<mgcl::Problem *()> createProblem = [m, n, o, &f, &v, vcycleIters, N]()
            {
                auto p = new mgcl::Problem(m, n, o, f, v);
                p->setSilent(true);
                p->setMaxiterVcycles(vcycleIters);
                p->setStencilType(mgcl::MGCL_VARYING);
                auto &s = *p->getStencilValues();

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

                return p;
            };

            std::string name = std::string("seq varying 7p Laplace, N = ")
                                   .append(std::to_string(N))
                                   .append(", iters: ")
                                   .append(std::to_string(vcycleIters));

            b.run(std::string(name).append(", overhead (oh)").c_str(), [&]
                  {
                      auto p = createProblem();
                      delete p; //
                  });

            if (N >= 32)
                b.epochs(1).epochIterations(1);

            b.run(std::string(name).append(", init+oh").c_str(), [&]
                  {
                      auto p = createProblem();
                      p->init();
                      delete p; //
                  });

            b.run(std::string(name).append(", init+oh+solve").c_str(), [&]
                  {
                      auto p = createProblem();
                      p->solve();
                      delete p; //
                  });

            if (N >= 32)
                b.epochs(11).epochIterations(0);
        }

        if (gpuAvailable)
        {
            // OpenCL fixed Laplace stencil
            auto v = std::make_shared<mgcl::Cuboid>(m, n, o);
            auto f = std::make_shared<mgcl::Cuboid>(m, n, o);
            v->fillRandom(0, 10);
            f->fillRandom(0, 10);

            std::function<mgcl::Problem *()> createProblem = [m, n, o, &f, &v, vcycleIters]()
            {
                auto p = new mgcl::Problem(m, n, o, f, v);
                p->setSilent(true);
                p->setMaxiterVcycles(vcycleIters);
                p->setUseOpencl(true);
                clFinish(p->getCommands());
                return p;
            };

            std::string name = std::string("ocl fixed 7p Laplace, N = ")
                                   .append(std::to_string(N))
                                   .append(", iters: ")
                                   .append(std::to_string(vcycleIters));

            // Increase epoch count for overhead
            auto tmp = b.minEpochIterations();
            b.minEpochIterations(300);

            b.run(std::string(name).append(", overhead (oh)").c_str(), [&]
                  {
                      auto p = createProblem();
                      delete p; //
                  });

            b.minEpochIterations(tmp);

            b.run(std::string(name).append(", init ocl env").c_str(), [&]
                  {
                      auto p = createProblem();
                      p->getOpenCLHelper().init();
                      delete p; //
                  });

            b.run(std::string(name).append(", init+oh").c_str(), [&]
                  {
                      auto p = createProblem();
                      p->init();
                      delete p; //
                  });

            b.run(std::string(name).append(", init+oh+solve").c_str(), [&]
                  {
                      auto p = createProblem();
                      p->solve();
                      delete p; //
                  });
        }

        if (gpuAvailable)
        {
            // OpenCL varying stencil (Galerkin)
            auto v = std::make_shared<mgcl::Cuboid>(m, n, o);
            auto f = std::make_shared<mgcl::Cuboid>(m, n, o);
            v->fillRandom(0, 10);
            f->fillRandom(0, 10);

            std::function<mgcl::Problem *()> createProblem = [m, n, o, &f, &v, vcycleIters, N]()
            {
                auto p = new mgcl::Problem(m, n, o, f, v);
                p->setSilent(true);
                p->setUseOpencl(true);
                p->setMaxiterVcycles(vcycleIters);
                p->setStencilType(mgcl::MGCL_VARYING);
                auto &s = *p->getStencilValues();

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

                return p;
            };

            std::string name = std::string("ocl varying 7p Laplace, N = ")
                                   .append(std::to_string(N))
                                   .append(", iters: ")
                                   .append(std::to_string(vcycleIters));

            // if (N >= 32)
            //     b.epochs(1).epochIterations(1);

            b.run(std::string(name).append(", overhead (oh)").c_str(), [&]
                  {
                      auto p = createProblem();
                      delete p; //
                  });

            b.run(std::string(name).append(", init ocl env").c_str(), [&]
                  {
                      auto p = createProblem();
                      p->getOpenCLHelper().init();
                      delete p; //
                  });

            b.run(std::string(name).append(", init+oh").c_str(), [&]
                  {
                      auto p = createProblem();
                      p->init();
                      delete p; //
                  });

            b.run(std::string(name).append(", init+oh+solve").c_str(), [&]
                  {
                      auto p = createProblem();
                      p->solve();
                      delete p; //
                  });

            // if (N >= 32)
            //     b.epochs(11).epochIterations(0);
        }
    }

    std::ofstream renderOutCsv(std::string("bench_initVsSolve_").append(std::to_string(N)).append(".csv"));
    b.render(ankerl::nanobench::templates::csv(), renderOutCsv);
}
