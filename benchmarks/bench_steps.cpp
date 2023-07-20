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
#include "../src/problem.hpp"
#include "../test/test_utility.hpp"
#include "bench_render_templates.hpp"

/**
 * @brief Measures each step of vcycle and compares different steps once for seq and once for ocl.
 *
 */
TEST_CASE("mgcl benchmarks console: steps", "[!benchmark][steps][console][stepvsstep]")
{
    int N = GENERATE(16, 32, 64, 128);
    //     int N = 16;
    int m = N;
    int n = N;
    int o = N;

    mgcl::BC bc = mgcl::BC::PERIODIC;
    bool periodic = bc == mgcl::BC::PERIODIC;

    ankerl::nanobench::Bench b;
    b.timeUnit(1us, "us")
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
        p.setBc(bc);
        p.setSilent(true);
        p.init();

        auto &v0 = p.getLevelAt(0).getV();
        auto &f0 = p.getLevelAt(0).getF();
        auto &r0 = p.getLevelAt(0).getR();
        auto &fine = p.getLevelAt(0);
        auto &coarse = p.getLevelAt(1);
        double stencilFactor0 = fine.getStencilFactor();
        double stencilFactor1 = coarse.getStencilFactor();

        // jacobi
        b.run(std::string("seq jacobi, N = ").append(std::to_string(N)).c_str(), [&]
              { mgcl::MultigridEngine::jacobiSeq(v0, f0, r0, p.getOmega(), p.getNu1(), p.getResidualNorm(),
                                                 p.getStencilType(), stencilFactor0, p.getStencilValues().get(), false, periodic); });

        // residual
        b.run(std::string("seq residual, N = ").append(std::to_string(N)).c_str(), [&]
              { mgcl::MultigridEngine::residualSeq(f0, v0, r0, p.getResidualNorm(), p.getStencilType(), stencilFactor0, p.getStencilValues().get(), false, periodic); });

        // updateGhosts
        b.run(std::string("seq updateGhosts, N = ").append(std::to_string(N)).c_str(), [&]
              { mgcl::MultigridEngine::updateGhostsSeq(v0); });

        // restriction
        b.run(std::string("seq restrict, N = ").append(std::to_string(N)).c_str(), [&]
              { mgcl::MultigridEngine::restrictSeq(fine, coarse, fine.getR(), coarse.getF()); });

        // prolongation
        b.run(std::string("seq prolongate, N = ").append(std::to_string(N)).c_str(), [&]
              { mgcl::MultigridEngine::prolongateSeq(fine, coarse, fine.getR(), coarse.getV()); });

        //   std::ofstream renderOut(std::string("solvingBoxplot_").append(std::to_string(N)).append(".html"));
        //   b.render(ankerl::nanobench::templates::htmlBoxplot(), renderOut);
        std::cout << "=============" << std::endl;
    }

    SECTION(std::string("ocl gpu N = ").append(std::to_string(N)).c_str())
    {
        auto f = std::make_shared<mgcl::Cuboid>(m, n, o);
        auto v = std::make_shared<mgcl::Cuboid>(m, n, o);
        f->fillRandom(0, 10);

        mgcl::Problem p(m, n, o, f, v);
        p.setUseOpencl(true);
        if (mgcl_test::TestUtility::deviceAvailable("Quadro", CL_DEVICE_TYPE_GPU))
            p.setDeviceName("Quadro");
        p.setSilent(true);
        p.init();

        auto &v0 = p.getLevelAt(0).getV();
        auto v0d = p.getLevelAt(0).getDVIn();
        auto &fine = p.getLevelAt(0);
        auto &coarse = p.getLevelAt(1);

        // jacobi
        b.run(std::string("ocl gpu jacobi, N = ").append(std::to_string(N)).c_str(),
              [&]
              {
                  mgcl::MultigridEngine::jacobi(p, fine, p.getNu1(), false);
                  clFinish(p.getCommands());
              });

        // residual
        b.run(std::string("ocl gpu residual, N = ").append(std::to_string(N)).c_str(),
              [&]
              {
                  mgcl::MultigridEngine::residual(p, fine, false);
                  clFinish(p.getCommands());
              });

        // updateGhosts
        b.run(std::string("ocl gpu updateGhosts, N = ").append(std::to_string(N)).c_str(),
              [&]
              {
                  mgcl::MultigridEngine::updateGhosts(p, v0d, v0.getMgh(), v0.getNgh(), v0.getOgh(),
                                                      v0.getGhostsM(), v0.getGhostsN(), v0.getGhostsO());
                  clFinish(p.getCommands());
              });

        // restriction
        b.run(std::string("ocl gpu restrict, N = ").append(std::to_string(N)).c_str(),
              [&]
              {
                  mgcl::MultigridEngine::restrict(fine, coarse, fine.getDR(), coarse.getDF());
                  clFinish(p.getCommands());
              });

        // prolongation
        b.run(std::string("ocl gpu prolongate, N = ").append(std::to_string(N)).c_str(),
              [&]
              {
                  mgcl::MultigridEngine::prolongate(fine, coarse, fine.getDR(), coarse.getDVIn());
                  clFinish(p.getCommands());
              });

        //   std::ofstream renderOut(std::string("solvingBoxplot_").append(std::to_string(N)).append(".html"));
        //   b.render(ankerl::nanobench::templates::htmlBoxplot(), renderOut);
        std::cout << "=============" << std::endl;
    }
}

/**
 * @brief Measures each step of vcycle and compares seq vs ocl versions
 *
 */
TEST_CASE("mgcl benchmarks console: steps", "[!benchmark][steps][console][seqvsocl]")
{
    int N = GENERATE(16, 32, 64, 128);
    //     int N = 16;
    int m = N;
    int n = N;
    int o = N;

    mgcl::BC bc = mgcl::BC::PERIODIC;
    bool periodic = bc == mgcl::BC::PERIODIC;

    ankerl::nanobench::Bench b;
    b.timeUnit(1us, "us")
        // .epochs(1)
        // .epochIterations(1)
        .minEpochTime(100ms)
        .maxEpochTime(5s)
        .relative(true);

    SECTION(std::string("jacobi N = ").append(std::to_string(N)).c_str())
    {
        auto f = std::make_shared<mgcl::Cuboid>(m, n, o);
        f->fillRandom(0, 10);

        {
            auto v = std::make_shared<mgcl::Cuboid>(m, n, o);

            mgcl::Problem p(m, n, o, f, v);
            p.setBc(bc);
            p.setSilent(true);
            p.init();

            auto &v0 = p.getLevelAt(0).getV();
            auto &f0 = p.getLevelAt(0).getF();
            auto &r0 = p.getLevelAt(0).getR();
            auto &fine = p.getLevelAt(0);
            auto &coarse = p.getLevelAt(1);
            double stencilFactor0 = fine.getStencilFactor();
            double stencilFactor1 = coarse.getStencilFactor();

            b.run(std::string("seq jacobi, N = ").append(std::to_string(N)).c_str(), [&]
                  { mgcl::MultigridEngine::jacobiSeq(v0, f0, r0, p.getOmega(), p.getNu1(), p.getResidualNorm(),
                                                     p.getStencilType(), stencilFactor0, p.getStencilValues().get(), false, periodic); });
        }

        {
            auto v = std::make_shared<mgcl::Cuboid>(m, n, o);

            mgcl::Problem p(m, n, o, f, v);
            p.setUseOpencl(true);
            if (mgcl_test::TestUtility::deviceAvailable("Quadro", CL_DEVICE_TYPE_GPU))
                p.setDeviceName("Quadro");
            p.setSilent(true);
            p.init();

            auto &v0 = p.getLevelAt(0).getV();
            auto v0d = p.getLevelAt(0).getDVIn();
            auto &fine = p.getLevelAt(0);
            auto &coarse = p.getLevelAt(1);

            // jacobi
            b.run(std::string("ocl gpu jacobi, N = ").append(std::to_string(N)).c_str(),
                  [&]
                  {
                      mgcl::MultigridEngine::jacobi(p, fine, p.getNu1(), false);
                      clFinish(p.getCommands());
                  });
        }
        std::cout << "=============" << std::endl;
    }

    SECTION(std::string("residual N = ").append(std::to_string(N)).c_str())
    {
        auto f = std::make_shared<mgcl::Cuboid>(m, n, o);
        f->fillRandom(0, 10);

        {
            auto v = std::make_shared<mgcl::Cuboid>(m, n, o);

            mgcl::Problem p(m, n, o, f, v);
            p.setSilent(true);
            p.init();

            auto &v0 = p.getLevelAt(0).getV();
            auto &f0 = p.getLevelAt(0).getF();
            auto &r0 = p.getLevelAt(0).getR();
            auto &fine = p.getLevelAt(0);
            auto &coarse = p.getLevelAt(1);
            double stencilFactor0 = fine.getStencilFactor();
            double stencilFactor1 = coarse.getStencilFactor();

            b.run(std::string("seq residual, N = ").append(std::to_string(N)).c_str(), [&]
                  { mgcl::MultigridEngine::residualSeq(f0, v0, r0, p.getResidualNorm(), p.getStencilType(),
                                                       stencilFactor0, p.getStencilValues().get(), false, periodic); });
        }

        {
            auto v = std::make_shared<mgcl::Cuboid>(m, n, o);

            mgcl::Problem p(m, n, o, f, v);
            p.setUseOpencl(true);
            if (mgcl_test::TestUtility::deviceAvailable("Quadro", CL_DEVICE_TYPE_GPU))
                p.setDeviceName("Quadro");
            p.setSilent(true);
            p.init();

            auto &v0 = p.getLevelAt(0).getV();
            auto v0d = p.getLevelAt(0).getDVIn();
            auto &fine = p.getLevelAt(0);
            auto &coarse = p.getLevelAt(1);

            b.run(std::string("ocl gpu residual, N = ").append(std::to_string(N)).c_str(),
                  [&]
                  {
                      mgcl::MultigridEngine::residual(p, fine, false);
                      clFinish(p.getCommands());
                  });
        }
        std::cout << "=============" << std::endl;
    }

    SECTION(std::string("updateGhosts N = ").append(std::to_string(N)).c_str())
    {
        auto f = std::make_shared<mgcl::Cuboid>(m, n, o);
        f->fillRandom(0, 10);

        {
            auto v = std::make_shared<mgcl::Cuboid>(m, n, o);

            mgcl::Problem p(m, n, o, f, v);
            p.setSilent(true);
            p.init();

            auto &v0 = p.getLevelAt(0).getV();
            auto &f0 = p.getLevelAt(0).getF();
            auto &r0 = p.getLevelAt(0).getR();
            auto &fine = p.getLevelAt(0);
            auto &coarse = p.getLevelAt(1);

            b.run(std::string("seq updateGhosts, N = ").append(std::to_string(N)).c_str(), [&]
                  { mgcl::MultigridEngine::updateGhostsSeq(v0); });
        }

        {
            auto v = std::make_shared<mgcl::Cuboid>(m, n, o);

            mgcl::Problem p(m, n, o, f, v);
            p.setUseOpencl(true);
            if (mgcl_test::TestUtility::deviceAvailable("Quadro", CL_DEVICE_TYPE_GPU))
                p.setDeviceName("Quadro");
            p.setSilent(true);
            p.init();

            auto &v0 = p.getLevelAt(0).getV();
            auto v0d = p.getLevelAt(0).getDVIn();
            auto &fine = p.getLevelAt(0);
            auto &coarse = p.getLevelAt(1);

            b.run(std::string("ocl gpu updateGhosts, N = ").append(std::to_string(N)).c_str(),
                  [&]
                  {
                      mgcl::MultigridEngine::updateGhosts(p, v0d, v0.getMgh(), v0.getNgh(), v0.getOgh(),
                                                          v0.getGhostsM(), v0.getGhostsN(), v0.getGhostsO());
                      clFinish(p.getCommands());
                  });
        }
        std::cout << "=============" << std::endl;
    }

    SECTION(std::string("restrict N = ").append(std::to_string(N)).c_str())
    {
        auto f = std::make_shared<mgcl::Cuboid>(m, n, o);
        f->fillRandom(0, 10);

        {
            auto v = std::make_shared<mgcl::Cuboid>(m, n, o);

            mgcl::Problem p(m, n, o, f, v);
            p.setSilent(true);
            p.init();

            auto &v0 = p.getLevelAt(0).getV();
            auto &f0 = p.getLevelAt(0).getF();
            auto &r0 = p.getLevelAt(0).getR();
            auto &fine = p.getLevelAt(0);
            auto &coarse = p.getLevelAt(1);

            b.run(std::string("seq restrict, N = ").append(std::to_string(N)).c_str(), [&]
                  { mgcl::MultigridEngine::restrictSeq(fine, coarse, fine.getR(), coarse.getF()); });
        }

        {
            auto v = std::make_shared<mgcl::Cuboid>(m, n, o);

            mgcl::Problem p(m, n, o, f, v);
            p.setUseOpencl(true);
            if (mgcl_test::TestUtility::deviceAvailable("Quadro", CL_DEVICE_TYPE_GPU))
                p.setDeviceName("Quadro");
            p.setSilent(true);
            p.init();

            auto &v0 = p.getLevelAt(0).getV();
            auto v0d = p.getLevelAt(0).getDVIn();
            auto &fine = p.getLevelAt(0);
            auto &coarse = p.getLevelAt(1);

            b.run(std::string("ocl gpu restrict, N = ").append(std::to_string(N)).c_str(),
                  [&]
                  {
                      mgcl::MultigridEngine::restrict(fine, coarse, fine.getDR(), coarse.getDF());
                      clFinish(p.getCommands());
                  });
        }
        std::cout << "=============" << std::endl;
    }

    SECTION(std::string("prolongate N = ").append(std::to_string(N)).c_str())
    {
        auto f = std::make_shared<mgcl::Cuboid>(m, n, o);
        f->fillRandom(0, 10);

        {
            auto v = std::make_shared<mgcl::Cuboid>(m, n, o);

            mgcl::Problem p(m, n, o, f, v);
            p.setSilent(true);
            p.init();

            auto &v0 = p.getLevelAt(0).getV();
            auto &f0 = p.getLevelAt(0).getF();
            auto &r0 = p.getLevelAt(0).getR();
            auto &fine = p.getLevelAt(0);
            auto &coarse = p.getLevelAt(1);

            b.run(std::string("seq prolongate, N = ").append(std::to_string(N)).c_str(), [&]
                  { mgcl::MultigridEngine::prolongateSeq(fine, coarse, fine.getR(), coarse.getV()); });
        }

        {
            auto v = std::make_shared<mgcl::Cuboid>(m, n, o);

            mgcl::Problem p(m, n, o, f, v);
            p.setUseOpencl(true);
            if (mgcl_test::TestUtility::deviceAvailable("Quadro", CL_DEVICE_TYPE_GPU))
                p.setDeviceName("Quadro");
            p.setSilent(true);
            p.init();

            auto &v0 = p.getLevelAt(0).getV();
            auto v0d = p.getLevelAt(0).getDVIn();
            auto &fine = p.getLevelAt(0);
            auto &coarse = p.getLevelAt(1);

            b.run(std::string("ocl gpu prolongate, N = ").append(std::to_string(N)).c_str(),
                  [&]
                  {
                      mgcl::MultigridEngine::prolongate(fine, coarse, fine.getDR(), coarse.getDVIn());
                      clFinish(p.getCommands());
                  });
        }
        std::cout << "=============" << std::endl;
    }
}
