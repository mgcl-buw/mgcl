#include "nanobench.h"

#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"

#include <chrono>
#include <fstream>
#include <vector>
using namespace std::chrono_literals;

#include "../cuboid.hpp"
#include "../multigrid_engine.hpp"
#include "../problem.hpp"
#include "../test/test_utility.hpp"
#include "bench_render_templates.hpp"

TEST_CASE("mgcl benchmarks console: steps", "[!benchmark][steps][console]")
{
    int N = GENERATE(16, 32, 64, 128);
    //     int N = 16;
    int m = N;
    int n = N;
    int o = N;

    ankerl::nanobench::Bench b;
    b.timeUnit(1ms, "ms")
        // .epochs(1)
        // .epochIterations(1)
        .minEpochTime(100ms)
        .maxEpochTime(5s)
        .relative(true);

    SECTION(std::string("seq N = ").append(std::to_string(N)).c_str())
    {
        auto f = std::make_shared<mgcl::Cuboid>(m, n, o);
        auto v = std::make_shared<mgcl::Cuboid>(m, n, o);
        f->fillRandom(0, 10);

        mgcl::Problem p(m, n, o, f, v);
        p.init();

        auto &v0 = p.getLevelAt(0).getV();
        auto &f0 = p.getLevelAt(0).getF();
        auto &r0 = p.getLevelAt(0).getR();
        auto &fine = p.getLevelAt(0);
        auto &coarse = p.getLevelAt(1);

        // if (N >= 128)
        //     b.epochs(3);

        // jacobi
        b.run(std::string("seq jacobi, N = ").append(std::to_string(N)).c_str(), [&]
              { mgcl::MultigridEngine::jacobiSeq(v0, f0, r0, p.getOmega(), p.getNu1(), p.getResidualNorm(), *p.getStencil(), false); });

        // residual
        b.run(std::string("seq residual, N = ").append(std::to_string(N)).c_str(), [&]
              { mgcl::MultigridEngine::residualSeq(f0, v0, r0, p.getResidualNorm(), *p.getStencil(), false); });

        // updateGhosts
        b.run(std::string("seq updateGhosts, N = ").append(std::to_string(N)).c_str(), [&]
              { mgcl::MultigridEngine::updateGhostsSeq(v0); });

        // restriction
        b.run(std::string("seq restrict, N = ").append(std::to_string(N)).c_str(), [&]
              { mgcl::MultigridEngine::restrictSeq(fine, coarse, fine.getR(), coarse.getF()); });

        // prolongation
        b.run(std::string("seq prolongate, N = ").append(std::to_string(N)).c_str(), [&]
              { mgcl::MultigridEngine::prolongateSeq(fine, coarse, fine.getR(), coarse.getV()); });

        std::ofstream renderOut(std::string("solvingBoxplot_").append(std::to_string(N)).append(".html"));
        b.render(ankerl::nanobench::templates::htmlBoxplot(), renderOut);
    }
}
