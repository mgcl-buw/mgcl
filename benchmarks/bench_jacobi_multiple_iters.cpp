#include "nanobench.h"

#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
using namespace std::chrono_literals;

#include "../src/cuboid.hpp"
#include "../src/multigrid_engine.hpp"
#include "../src/opencl_helper.hpp"
#include "../src/problem.hpp"
#include "../test/ocl_wrapper.hpp"
#include "../test/test_utility.hpp"
#include "bench_render_templates.hpp"

/**
 * Benchmarks Jacobi multiple iterations without ghost-update in-between vs. standard Jacobi.
 *
 */
TEST_CASE("benchmark Jacobi seq multiple iters", "[console][jacobiMulti][seq]")
{
    int N = GENERATE(8, 16, 32, 64);
    // int N = 16;
    int m = N;
    int n = N;
    int o = N;
    double h = 1.0 / (double)N;

    int maxStepsPerIter = 3;
    int iters = 3;

    double omega = 0.8;
    int maxiter = 3;
    mgcl::MGCL_RESIDUAL_NORM resnorm = mgcl::MGCL_L2;
    mgcl::MGCL_STENCIL stencilType = mgcl::MGCL_LAPLACE_27POINT;
    double stencilFactor = 1.0 / (30.0 * h * h);

    ankerl::nanobench::Bench bench;
    bench.timeUnit(1ms, "ms")
        // .epochs(1)
        // .epochIterations(1)
        .minEpochTime(100ms)
        .relative(true);

    for (int spi = 1; spi <= maxStepsPerIter; spi++)
    {
        mgcl::Cuboid v(m, n, o, spi, spi, spi);
        mgcl::Cuboid r(m, n, o, spi, spi, spi);
        mgcl::Cuboid f(m, n, o, spi, spi, spi);
        v.fillRandom();
        f.fillRandom();

        std::string name = std::string("seq, N = ")
                               .append(std::to_string(N))
                               .append(", spi = ")
                               .append(std::to_string(spi));
        bench.run(std::string(name).c_str(), [&]
                  { mgcl::MultigridEngine::jacobiSeq(v, f, r, omega, iters, resnorm, stencilType, stencilFactor,
                                                     nullptr, false, true, spi); });
    }
}

// Same as above but with OCL.
TEST_CASE("benchmark Jacobi OCL multiple iters", "[console][jacobiMulti][ocl]")
{
    int maxStepsPerIter = 3;
    std::stringstream ss;
    ss << "N;iters;spi;ns" << std::endl;

    // Vector to collect all minimum times per spi, in order to get avg results later.
    std::vector<std::vector<int>> mintimesPerSpi(maxStepsPerIter);

    std::vector<int> Ns = {8, 16, 32, 64};
    std::vector<int> itersAll = {3, 5, 10};
    for (auto N : Ns)
    {
        int m = N;
        int n = N;
        int o = N;
        double h = 1.0 / (double)N;

        double omega = 0.8;
        int maxiter = 3;
        mgcl::MGCL_RESIDUAL_NORM resnorm = mgcl::MGCL_L2;
        mgcl::MGCL_STENCIL stencilType = mgcl::MGCL_LAPLACE_27POINT;
        double stencilFactor = 1.0 / (30.0 * h * h);

        for (auto iters : itersAll)
        {
            ankerl::nanobench::Bench bench;
            bench.timeUnit(1ns, "ns")
                // .epochs(1)
                // .epochIterations(1)
                .minEpochTime(100ms)
                .relative(true);

            for (int spi = 1; spi <= maxStepsPerIter; spi++)
            {
                auto v = std::make_shared<mgcl::Cuboid>(m, n, o, spi, spi, spi);
                auto r = std::make_shared<mgcl::Cuboid>(m, n, o, spi, spi, spi);
                auto f = std::make_shared<mgcl::Cuboid>(m, n, o, spi, spi, spi);
                v->fillRandom();
                f->fillRandom();

                auto p = std::make_shared<mgcl::Problem>(m, n, o, v, f);
                p->setGhosts(spi);
                p->setJacobiIterationsPerKernel(spi);
                p->setUseOpencl(true);
                p->setSilent(true);
                p->init();
                auto &level = p->getLevelAt(0);
                mgcl_test::TestUtility tu(p);

                std::string name = std::string("ocl, N = ")
                                       .append(std::to_string(N))
                                       .append(", spi = ")
                                       .append(std::to_string(spi))
                                       .append(", iters = ")
                                       .append(std::to_string(iters));
                bench.run(std::string(name).c_str(), [&] { //
                    mgcl::MultigridEngine::jacobi(*p, level, iters, false, spi);
                    tu.finish(); //
                });

                // Get minimum of all epochs in ns
                double min = 1000000;
                for (auto r : bench.results())
                    if (r.minimum(ankerl::nanobench::Result::Measure::elapsed) < min)
                        min = r.minimum(ankerl::nanobench::Result::Measure::elapsed) * 1000.0 * 1000.0 * 1000.0;

                ss << N << ";" << iters << ";" << spi << ";" << min << std::endl;
                mintimesPerSpi[spi - 1].push_back(min);

                // std::cout << "bench.results()[0].size(): " << bench.results()[0].size() << std::endl;
            }

            // std::ofstream renderOutCsv(std::string("jacobiMultiIterOcl_")
            //                                .append(std::to_string(N))
            //                                .append("_")
            //                                .append(".csv"));

            // std::streambuf *old = std::cout.rdbuf(ss.rdbuf());
            // bench.render(ankerl::nanobench::templates::csv(), std::cout);
            // std::cout.rdbuf(old);
        }
    }

    std::cout << ss.str() << std::endl;

    std::vector<double> avgs = {0, 0, 0};
    for (int spi = 0; spi < maxStepsPerIter; spi++)
    {
        for (int val : mintimesPerSpi[spi])
            avgs[spi] += val;

        avgs[spi] /= (double)mintimesPerSpi[spi].size();
    }

    std::cout << "avgs:" << std::endl
              << "  spi: 1: " << avgs[0] << " ns" << std::endl
              << "  spi: 2: " << avgs[1] << " ns" << std::endl
              << "  spi: 3: " << avgs[2] << " ns" << std::endl;
}

// // Same as above but with MPI.
// TEST_CASE("benchmark Jacobi OCL MPI multiple iters", "[console][jacobiMulti][ocl][mpi]")
// {
//     int maxStepsPerIter = 3;

//     std::vector<int> Ns = {8, 16, 32, 64};
//     std::vector<int> itersAll = {3, 5, 10, 20, 50};
//     for (auto N : Ns)
//     {
//         int m = N;
//         int n = N;
//         int o = N;
//         double h = 1.0 / (double)N;

//         double omega = 0.8;
//         int maxiter = 3;
//         mgcl::MGCL_RESIDUAL_NORM resnorm = mgcl::MGCL_L2;
//         mgcl::MGCL_STENCIL stencilType = mgcl::MGCL_LAPLACE_27POINT;
//         double stencilFactor = 1.0 / (30.0 * h * h);

//         for (auto iters : itersAll)
//         {
//             ankerl::nanobench::Bench bench;
//             bench.timeUnit(1ns, "ns")
//                 // .epochs(1)
//                 // .epochIterations(1)
//                 .minEpochTime(100ms)
//                 .relative(true);

//             for (int spi = 1; spi <= maxStepsPerIter; spi++)
//             {
//                 auto v = std::make_shared<mgcl::Cuboid>(m, n, o, spi, spi, spi);
//                 auto r = std::make_shared<mgcl::Cuboid>(m, n, o, spi, spi, spi);
//                 auto f = std::make_shared<mgcl::Cuboid>(m, n, o, spi, spi, spi);
//                 v->fillRandom();
//                 f->fillRandom();

//                 auto p = std::make_shared<mgcl::Problem>(m, n, o, v, f);
//                 p->setGhosts(spi);
//                 p->setJacobiIterationsPerKernel(spi);
//                 p->setUseOpencl(true);
//                 p->setSilent(true);
//                 p->init();
//                 auto &level = p->getLevelAt(0);
//                 mgcl_test::TestUtility tu(p);

//                 std::string name = std::string("ocl, N = ")
//                                        .append(std::to_string(N))
//                                        .append(", spi = ")
//                                        .append(std::to_string(spi))
//                                        .append(", iters = ")
//                                        .append(std::to_string(iters));
//                 bench.run(std::string(name).c_str(), [&] { //
//                     mgcl::MultigridEngine::jacobi(*p, level, iters, false, spi);
//                     tu.finish(); //
//                 });
//             }
//         }
//     }
// }
