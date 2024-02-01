#include "nanobench.h"

#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"

#include <chrono>
#include <fstream>
#include <functional> // for function
#include <iostream>
#include <sstream>
#include <vector>
using namespace std::chrono_literals;

#include "../src/mgcl/cuboid.hpp"
#include "../src/mgcl/problem.hpp"
#include "../test/test_utility.hpp"
#include "bench_render_templates.hpp"
#include "cli_args.hpp"

TEST_CASE("galerkin init vs solve", "[galerkinInitVsSolve][all]")
{
    // Problem parameters
    double tol = 1e-20;
    int nu1 = 2;
    int nu2 = 2;
    double omega = 0.8;
    int vcycleIters = CLI_ARGS::vCycleIterations;

    std::cout << "Problem parameters:" << std::endl
              << "  tol: " << tol << std::endl
              << "  nu1: " << nu1 << std::endl
              << "  nu2: " << nu2 << std::endl
              << "  omega: " << omega << std::endl
              << "  v-cycle iterations: " << vcycleIters << std::endl;

    // Build csv with aggregated values
    bool exportAggregated = true;
    std::stringstream ss;
    ss << std::scientific << "\"grid\";\"type\";\"step\";\"minTime\";\"medianTime\";\"maxTime\";\"avgTime\"\n";

    // auto grids = CLI_ARGS::grids;
    auto grids = CLI_ARGS::grids;
    if (grids.empty())
        std::vector<int> grids = {8, 16, 32, 64};

    int minEpochIterations = CLI_ARGS::minEpochIterations;

    for (auto N : grids)
    {
        int m = N;
        int n = N;
        int o = N;

        ankerl::nanobench::Bench b;
        b.timeUnit(1ns, "ns")
            // .epochs(1)
            // .epochIterations(1)
            .minEpochTime(100ms)
            .minEpochIterations(minEpochIterations)
            .relative(false);
        // .warmup(1); // especially for init OpenCL environment

        bool gpuAvailable = mgcl_test::TestUtility::deviceAvailable("", CL_DEVICE_TYPE_GPU);
        // bool cpuAvailable = tu->deviceAvailable("", CL_DEVICE_TYPE_CPU);

        {
            // CPU fixed Laplace stencil
            auto v = std::make_shared<mgcl::Cuboid>(m, n, o);
            auto f = std::make_shared<mgcl::Cuboid>(m, n, o);
            v->fillRandom(0, 10);
            f->fillRandom(0, 10);

            std::function<mgcl::Problem*()> createProblem = [m, n, o, &f, &v, vcycleIters]()
            {
                auto p = new mgcl::Problem(m, n, o, f, v);
                p->setSilent(true);
                p->setMaxiterVcycles(vcycleIters);
                p->setStencilType(mgcl::MGCL_LAPLACE_27POINT);
                return p;
            };

            std::string name = std::string("seq fixed 27p Laplace, N = ")
                                   .append(std::to_string(N))
                                   .append(", iters: ")
                                   .append(std::to_string(vcycleIters));

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

            std::function<mgcl::Problem*()> createProblem = [m, n, o, &f, &v, vcycleIters, N]()
            {
                auto p = new mgcl::Problem(m, n, o, f, v);
                p->setSilent(true);
                p->setMaxiterVcycles(vcycleIters);
                p->setStencilType(mgcl::MGCL_VARYING);
                auto& s = *p->getStencilValues();

                mgcl_test::fill7pLaplace(s, true);

                return p;
            };

            std::string name = std::string("seq varying 27p Laplace, N = ")
                                   .append(std::to_string(N))
                                   .append(", iters: ")
                                   .append(std::to_string(vcycleIters));

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

            b.run(std::string(name).append(", init+oh+solve").c_str(), [&]
                  {
                      auto p = createProblem();
                      p->solve();
                      delete p; //
                  });
        }

        if (gpuAvailable)
        {
            // OpenCL fixed Laplace stencil
            auto v = std::make_shared<mgcl::Cuboid>(m, n, o);
            auto f = std::make_shared<mgcl::Cuboid>(m, n, o);
            v->fillRandom(0, 10);
            f->fillRandom(0, 10);

            std::function<mgcl::Problem*()> createProblem = [m, n, o, &f, &v, vcycleIters]()
            {
                auto p = new mgcl::Problem(m, n, o, f, v);
                p->setSilent(true);
                p->setMaxiterVcycles(vcycleIters);
                p->setStencilType(mgcl::MGCL_LAPLACE_27POINT);
                p->setUseOpencl(true);
                clFinish(p->getCommands());
                return p;
            };

            std::string name = std::string("ocl fixed 27p Laplace, N = ")
                                   .append(std::to_string(N))
                                   .append(", iters: ")
                                   .append(std::to_string(vcycleIters));

            b.run(std::string(name).append(", overhead (oh)").c_str(), [&]
                  {
                      auto p = createProblem();
                      delete p; //
                  });

            b.run(std::string(name).append(", oh+init_ocl").c_str(), [&]
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

            std::function<mgcl::Problem*()> createProblem = [m, n, o, &f, &v, vcycleIters, N]()
            {
                auto p = new mgcl::Problem(m, n, o, f, v);
                p->setSilent(true);
                p->setUseOpencl(true);
                p->setMaxiterVcycles(vcycleIters);
                p->setStencilType(mgcl::MGCL_VARYING);
                auto& s = *p->getStencilValues();

                mgcl_test::fill7pLaplace(s, true);

                return p;
            };

            std::string name = std::string("ocl varying 27p Laplace, N = ")
                                   .append(std::to_string(N))
                                   .append(", iters: ")
                                   .append(std::to_string(vcycleIters));

            b.run(std::string(name).append(", overhead (oh)").c_str(), [&]
                  {
                      auto p = createProblem();
                      delete p; //
                  });

            b.run(std::string(name).append(", oh+init_ocl").c_str(), [&]
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

        if (exportAggregated)
        {
            // aggregate results, i.e. calculate solve from oh+init+solve, etc.
            // format: grid;type;step;minTime
            // depends on indices of results (hardcoded!)
            auto elapsed = ankerl::nanobench::Result::Measure::elapsed;
            auto resSize = b.results().size();
            int idx = 0;

            auto wr = [&b, &idx, &ss](int N, std::string type, std::string step, int d1, int d2)
            {
                auto elapsed = ankerl::nanobench::Result::Measure::elapsed;

                ss << "\"" << std::to_string(N) << "\";";
                ss << "\"" << type << "\";";
                ss << "\"" << step << "\";";
                ss << b.results()[idx + d1].minimum(elapsed) - b.results()[idx + d2].minimum(elapsed) << ";";
                ss << b.results()[idx + d1].median(elapsed) - b.results()[idx + d2].median(elapsed) << ";";
                ss << b.results()[idx + d1].maximum(elapsed) - b.results()[idx + d2].maximum(elapsed) << ";";
                ss << b.results()[idx + d1].average(elapsed) - b.results()[idx + d2].average(elapsed);
                ss << "\n";
            };

            if (resSize >= 2)
            {
                idx = 0;
                wr(N, "seq fixed 27p", "init", 1, 0);
                wr(N, "seq fixed 27p", "solve", 2, 1);
            }

            if (resSize >= 5)
            {
                idx = 3;
                wr(N, "seq varying 27p", "init", 1, 0);
                wr(N, "seq varying 27p", "solve", 2, 1);
            }

            if (resSize >= 9)
            {
                idx = 6;
                wr(N, "ocl fixed 27p", "init_ocl", 1, 0);
                wr(N, "ocl fixed 27p", "init", 2, 1);
                wr(N, "ocl fixed 27p", "solve", 3, 2);
            }

            if (resSize >= 13)
            {
                idx = 10;
                wr(N, "ocl varying 27p", "init_ocl", 1, 0);
                wr(N, "ocl varying 27p", "init", 2, 1);
                wr(N, "ocl varying 27p", "solve", 3, 2);
            }
        }
    }

    if (exportAggregated)
    {
        std::ofstream out;
        out.open(CLI_ARGS::outputPath + "bench_initVsSolve.csv");
        out << ss.str();
        out.close();
    }

    // std::ofstream renderOutCsv(std::string("bench_initVsSolve_").append(std::to_string(N)).append(".csv"));
    // b.render(ankerl::nanobench::templates::csv(), renderOutCsv);
}

TEST_CASE("galerkin init vs solve", "[galerkinInitVsSolve][oclOnly]")
{
    // Problem parameters
    double tol = 1e-20;
    int nu1 = 2;
    int nu2 = 2;
    double omega = 0.8;
    int vcycleIters = CLI_ARGS::vCycleIterations;

    std::cout << "Problem parameters:" << std::endl
              << "  tol: " << tol << std::endl
              << "  nu1: " << nu1 << std::endl
              << "  nu2: " << nu2 << std::endl
              << "  omega: " << omega << std::endl
              << "  v-cycle iterations: " << vcycleIters << std::endl;

    // Build csv with aggregated values
    bool exportAggregated = true;
    std::stringstream ss;
    ss << std::scientific << "\"grid\";\"type\";\"step\";\"minTime\";\"medianTime\";\"maxTime\";\"avgTime\"\n";

    // auto grids = CLI_ARGS::grids;
    auto grids = CLI_ARGS::grids;
    if (grids.empty())
        std::vector<int> grids = {8, 16, 32, 64};

    int minEpochIterations = CLI_ARGS::minEpochIterations;

    // std::vector<int> grids = {8, 16, 32, 64, 128};
    for (auto N : grids)
    {
        int m = N;
        int n = N;
        int o = N;

        ankerl::nanobench::Bench b;
        b.timeUnit(1ns, "ns")
            // .epochs(1)
            // .epochIterations(1)
            .minEpochIterations(minEpochIterations)
            .minEpochTime(100ms)
            .relative(false);
        // .warmup(1); // especially for init OpenCL environment

        bool gpuAvailable = mgcl_test::TestUtility::deviceAvailable("", CL_DEVICE_TYPE_GPU);
        // bool cpuAvailable = tu->deviceAvailable("", CL_DEVICE_TYPE_CPU);

        if (gpuAvailable)
        {
            // OpenCL fixed Laplace stencil
            auto v = std::make_shared<mgcl::Cuboid>(m, n, o);
            auto f = std::make_shared<mgcl::Cuboid>(m, n, o);
            v->fillRandom(0, 10);
            f->fillRandom(0, 10);

            std::function<mgcl::Problem*()> createProblem = [m, n, o, &f, &v, vcycleIters]()
            {
                auto p = new mgcl::Problem(m, n, o, f, v);
                p->setSilent(true);
                p->setMaxiterVcycles(vcycleIters);
                p->setStencilType(mgcl::MGCL_LAPLACE_27POINT);
                p->setUseOpencl(true);
                clFinish(p->getCommands());
                return p;
            };

            std::string name = std::string("ocl fixed 27p Laplace, N = ")
                                   .append(std::to_string(N))
                                   .append(", iters: ")
                                   .append(std::to_string(vcycleIters));

            b.run(std::string(name).append(", overhead (oh)").c_str(), [&]
                  {
                      auto p = createProblem();
                      delete p; //
                  });

            b.run(std::string(name).append(", oh+init_ocl").c_str(), [&]
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

            std::function<mgcl::Problem*()> createProblem = [m, n, o, &f, &v, vcycleIters, N]()
            {
                auto p = new mgcl::Problem(m, n, o, f, v);
                p->setSilent(true);
                p->setUseOpencl(true);
                p->setMaxiterVcycles(vcycleIters);
                p->setStencilType(mgcl::MGCL_VARYING);
                auto& s = *p->getStencilValues();

                mgcl_test::fill7pLaplace(s, true);

                return p;
            };

            std::string name = std::string("ocl varying 27p Laplace, N = ")
                                   .append(std::to_string(N))
                                   .append(", iters: ")
                                   .append(std::to_string(vcycleIters));

            b.run(std::string(name).append(", overhead (oh)").c_str(), [&]
                  {
                      auto p = createProblem();
                      delete p; //
                  });

            b.run(std::string(name).append(", oh+init_ocl").c_str(), [&]
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

        if (exportAggregated)
        {
            // aggregate results, i.e. calculate solve from oh+init+solve, etc.
            // format: grid;type;step;minTime
            // depends on indices of results (hardcoded!)
            auto elapsed = ankerl::nanobench::Result::Measure::elapsed;
            auto resSize = b.results().size();
            int idx = 0;

            auto wr = [&b, &idx, &ss](int N, std::string type, std::string step, int d1, int d2)
            {
                auto elapsed = ankerl::nanobench::Result::Measure::elapsed;

                ss << "\"" << std::to_string(N) << "\";";
                ss << "\"" << type << "\";";
                ss << "\"" << step << "\";";
                ss << b.results()[idx + d1].minimum(elapsed) - b.results()[idx + d2].minimum(elapsed) << ";";
                ss << b.results()[idx + d1].median(elapsed) - b.results()[idx + d2].median(elapsed) << ";";
                ss << b.results()[idx + d1].maximum(elapsed) - b.results()[idx + d2].maximum(elapsed) << ";";
                ss << b.results()[idx + d1].average(elapsed) - b.results()[idx + d2].average(elapsed);
                ss << "\n";
            };

            if (resSize >= 4)
            {
                idx = 0;
                wr(N, "ocl fixed 27p", "init_ocl", 1, 0);
                wr(N, "ocl fixed 27p", "init", 2, 1);
                wr(N, "ocl fixed 27p", "solve", 3, 2);
            }

            if (resSize >= 8)
            {
                idx = 4;
                wr(N, "ocl varying 27p", "init_ocl", 1, 0);
                wr(N, "ocl varying 27p", "init", 2, 1);
                wr(N, "ocl varying 27p", "solve", 3, 2);
            }
        }
    }

    if (exportAggregated)
    {
        std::ofstream out;
        out.open(CLI_ARGS::outputPath + "bench_initVsSolve_oclOnly.csv");
        out << ss.str();
        out.close();
    }

    // std::ofstream renderOutCsv(std::string("bench_initVsSolve_").append(std::to_string(N)).append(".csv"));
    // b.render(ankerl::nanobench::templates::csv(), renderOutCsv);
}
