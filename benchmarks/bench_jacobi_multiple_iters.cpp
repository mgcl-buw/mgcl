#include "nanobench.h"

#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
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

// TODO jacobi with mpi ghost update before this makes sense
TEST_CASE("benchmark Jacobi seq multiple iters", "[console][jacobiMulti][ocl]")
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
                               .append(std::to_string(spi));
        bench.run(std::string(name).c_str(), [&] { //
            mgcl::MultigridEngine::jacobi(*p, level, iters, false, spi);
            tu.finish(); //
        });
    }
}
