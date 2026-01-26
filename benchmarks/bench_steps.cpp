#include "bench_util.hpp"
#include "cli_args.hpp"
#include "nanobench.h"

#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"

#include <chrono>
#include <iostream>
#include <mpi.h>
using namespace std::chrono_literals;

#include "../src/mgcl/cuboid.hpp"
#include "../src/mgcl/multigrid_engine.hpp"
#include "../src/mgcl/problem.hpp"
#include "../test/test_utility.hpp"

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
    double h = 1.0 / static_cast<double>(N);

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

        auto& v0 = p.getLevelAt(0).getV();
        auto& f0 = p.getLevelAt(0).getF();
        auto& r0 = p.getLevelAt(0).getR();
        auto& fine = p.getLevelAt(0);
        auto& coarse = p.getLevelAt(1);
        double stencilFactor0 = fine.getStencilFactor();
        double stencilFactor1 = coarse.getStencilFactor();

        // jacobi
        b.run(std::string("seq jacobi, N = ").append(std::to_string(N)).c_str(), [&]
              { mgcl::MultigridEngine::jacobiSeq(v0, f0, r0, p.getOmega(), h * h, p.getNu1(), p.getResidualNorm(),
                                                 p.getStencilType(), stencilFactor0, p.getStencilValues().get(),
                                                 nullptr, false, periodic, false); });

        // residual
        b.run(std::string("seq residual, N = ").append(std::to_string(N)).c_str(), [&]
              { mgcl::MultigridEngine::residualSeq(f0, v0, r0, p.getResidualNorm(), p.getStencilType(), stencilFactor0, p.getStencilValues().get(), nullptr, false, periodic, false); });

        // updateGhosts
        b.run(std::string("seq updateGhosts, N = ").append(std::to_string(N)).c_str(), [&]
              { mgcl::MultigridEngine::updateGhostsSeq(v0, nullptr, true, false); });

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

        auto& v0 = p.getLevelAt(0).getV();
        auto& v0d = p.getLevelAt(0).getDVIn();
        auto& fine = p.getLevelAt(0);
        auto& coarse = p.getLevelAt(1);

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
                  mgcl::MultigridEngine::updateGhosts(p, v0d, nullptr, false);
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
 * @brief Measures each step of vcycle and compares against full solve (to get overhead).
 *
 */
TEST_CASE("mgcl_steps_galerkin_vs_solve")
{
    int N = GENERATE(8, 16, 32, 64);
    //     int N = 16;
    int m = N;
    int n = N;
    int o = N;
    double h = 1.0 / static_cast<double>(N);

    mgcl::BC bc = mgcl::BC::PERIODIC;
    bool periodic = bc == mgcl::BC::PERIODIC;

    ankerl::nanobench::Bench b;
    b.timeUnit(1us, "us")
        // .epochs(1)
        // .epochIterations(1)
        .minEpochTime(100ms)
        .maxEpochTime(5s)
        .relative(false);

    SECTION(std::string("gpu_ocl gpu N = ").append(std::to_string(N)).c_str())
    {
        auto f = std::make_shared<mgcl::Cuboid>(m, n, o);
        auto v = std::make_shared<mgcl::Cuboid>(m, n, o);
        f->fillRandom(0, 10);

        mgcl::Problem p(m, n, o, f, v);
        p.setUseOpencl(true);
        if (mgcl_test::TestUtility::deviceAvailable("Quadro", CL_DEVICE_TYPE_GPU))
            p.setDeviceName("Quadro");
        p.setSilent(true);
        p.setStencilType(mgcl::MGCL_VARYING);

        auto& sv = p.createStencilValues();
        sv->fill1dIndex(true);

        p.init();

        auto& v0 = p.getLevelAt(0).getV();
        auto& v0d = p.getLevelAt(0).getDVIn();
        auto& fine = p.getLevelAt(0);
        auto& coarse = p.getLevelAt(1);

        // jacobi
        b.run(std::string("gpu_jacobiN").append(std::to_string(N)).c_str(),
              [&]
              {
                  mgcl::MultigridEngine::jacobi(p, fine, p.getNu1(), false);
                  clFinish(p.getCommands());
              });

        // residual
        b.run(std::string("gpu_residualN").append(std::to_string(N)).c_str(),
              [&]
              {
                  mgcl::MultigridEngine::residual(p, fine, false);
                  clFinish(p.getCommands());
              });

        // updateGhosts
        b.run(std::string("gpu_updateGhostsN").append(std::to_string(N)).c_str(),
              [&]
              {
                  mgcl::MultigridEngine::updateGhosts(p, v0d, nullptr, false);
                  clFinish(p.getCommands());
              });

        // restriction
        b.run(std::string("gpu_restrictN").append(std::to_string(N)).c_str(),
              [&]
              {
                  mgcl::MultigridEngine::restrict(fine, coarse, fine.getDR(), coarse.getDF());
                  clFinish(p.getCommands());
              });

        // prolongation
        b.run(std::string("gpu_prolongateN").append(std::to_string(N)).c_str(),
              [&]
              {
                  mgcl::MultigridEngine::prolongate(fine, coarse, fine.getDR(), coarse.getDVIn());
                  clFinish(p.getCommands());
              });

        // prolongation
        b.run(std::string("gpu_galerkinN").append(std::to_string(N)).c_str(),
              [&]
              {
                  auto& fsv = *fine.getStencilValuesGpu();
                  mgcl::MultigridEngine::galerkinOptimized(fsv, 2, fsv.getM() >> 1, fsv.getN() >> 1, fsv.getO() >> 1,
                                                           p.getProgram(), p.getCommands(), p.getContext(),
                                                           nullptr, nullptr);
                  clFinish(p.getCommands());
              });

        //   std::ofstream renderOut(std::string("gpu_solvingBoxplot_").append(std::to_string(N)).append(".html"));
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
    double h = 1.0 / static_cast<double>(N);

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

            auto& v0 = p.getLevelAt(0).getV();
            auto& f0 = p.getLevelAt(0).getF();
            auto& r0 = p.getLevelAt(0).getR();
            auto& fine = p.getLevelAt(0);
            auto& coarse = p.getLevelAt(1);
            double stencilFactor0 = fine.getStencilFactor();
            double stencilFactor1 = coarse.getStencilFactor();

            b.run(std::string("seq jacobi, N = ").append(std::to_string(N)).c_str(), [&]
                  { mgcl::MultigridEngine::jacobiSeq(v0, f0, r0, p.getOmega(), h * h, p.getNu1(), p.getResidualNorm(),
                                                     p.getStencilType(), stencilFactor0, p.getStencilValues().get(),
                                                     nullptr, false, periodic, false); });
        }

        {
            auto v = std::make_shared<mgcl::Cuboid>(m, n, o);

            mgcl::Problem p(m, n, o, f, v);
            p.setUseOpencl(true);
            if (mgcl_test::TestUtility::deviceAvailable("Quadro", CL_DEVICE_TYPE_GPU))
                p.setDeviceName("Quadro");
            p.setSilent(true);
            p.init();

            auto& v0 = p.getLevelAt(0).getV();
            auto& v0d = p.getLevelAt(0).getDVIn();
            auto& fine = p.getLevelAt(0);
            auto& coarse = p.getLevelAt(1);

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

            auto& v0 = p.getLevelAt(0).getV();
            auto& f0 = p.getLevelAt(0).getF();
            auto& r0 = p.getLevelAt(0).getR();
            auto& fine = p.getLevelAt(0);
            auto& coarse = p.getLevelAt(1);
            double stencilFactor0 = fine.getStencilFactor();
            double stencilFactor1 = coarse.getStencilFactor();

            b.run(std::string("seq residual, N = ").append(std::to_string(N)).c_str(), [&]
                  { mgcl::MultigridEngine::residualSeq(f0, v0, r0, p.getResidualNorm(), p.getStencilType(),
                                                       stencilFactor0, p.getStencilValues().get(), nullptr, false, periodic, false); });
        }

        {
            auto v = std::make_shared<mgcl::Cuboid>(m, n, o);

            mgcl::Problem p(m, n, o, f, v);
            p.setUseOpencl(true);
            if (mgcl_test::TestUtility::deviceAvailable("Quadro", CL_DEVICE_TYPE_GPU))
                p.setDeviceName("Quadro");
            p.setSilent(true);
            p.init();

            auto& v0 = p.getLevelAt(0).getV();
            auto& v0d = p.getLevelAt(0).getDVIn();
            auto& fine = p.getLevelAt(0);
            auto& coarse = p.getLevelAt(1);

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

            auto& v0 = p.getLevelAt(0).getV();
            auto& f0 = p.getLevelAt(0).getF();
            auto& r0 = p.getLevelAt(0).getR();
            auto& fine = p.getLevelAt(0);
            auto& coarse = p.getLevelAt(1);

            b.run(std::string("seq updateGhosts, N = ").append(std::to_string(N)).c_str(), [&]
                  { mgcl::MultigridEngine::updateGhostsSeq(v0, nullptr, true, false); });
        }

        {
            auto v = std::make_shared<mgcl::Cuboid>(m, n, o);

            mgcl::Problem p(m, n, o, f, v);
            p.setUseOpencl(true);
            if (mgcl_test::TestUtility::deviceAvailable("Quadro", CL_DEVICE_TYPE_GPU))
                p.setDeviceName("Quadro");
            p.setSilent(true);
            p.init();

            auto& v0 = p.getLevelAt(0).getV();
            auto& v0d = p.getLevelAt(0).getDVIn();
            auto& fine = p.getLevelAt(0);
            auto& coarse = p.getLevelAt(1);

            b.run(std::string("ocl gpu updateGhosts, N = ").append(std::to_string(N)).c_str(),
                  [&]
                  {
                      mgcl::MultigridEngine::updateGhosts(p, v0d, nullptr, false);
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

            auto& v0 = p.getLevelAt(0).getV();
            auto& f0 = p.getLevelAt(0).getF();
            auto& r0 = p.getLevelAt(0).getR();
            auto& fine = p.getLevelAt(0);
            auto& coarse = p.getLevelAt(1);

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

            auto& v0 = p.getLevelAt(0).getV();
            auto& v0d = p.getLevelAt(0).getDVIn();
            auto& fine = p.getLevelAt(0);
            auto& coarse = p.getLevelAt(1);

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

            auto& v0 = p.getLevelAt(0).getV();
            auto& f0 = p.getLevelAt(0).getF();
            auto& r0 = p.getLevelAt(0).getR();
            auto& fine = p.getLevelAt(0);
            auto& coarse = p.getLevelAt(1);

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

            auto& v0 = p.getLevelAt(0).getV();
            auto& v0d = p.getLevelAt(0).getDVIn();
            auto& fine = p.getLevelAt(0);
            auto& coarse = p.getLevelAt(1);

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

/**
 * @brief Measures each step of vcycle using the profiler, i.e. prints kernel timings only.
 *
 */
TEST_CASE("benchStepsProfile")
{
    if (CLI_ARGS::grids.size() == 0 && (CLI_ARGS::gridsMin.size() == 0 || CLI_ARGS::gridsMax.size() == 0))
        throw "Need to specify at least one local grid size, e.g. using --grids 4,8,16 or --gridsMin 4,4,4 AND --gridsMax 32,32,32";

    // build grids to be tested from CLI args
    std::vector<std::vector<int>> gridsTBT;
    for (auto N : CLI_ARGS::grids)
        gridsTBT.push_back({N, N, N});
    if (CLI_ARGS::gridsMin.size() > 0 && CLI_ARGS::gridsMax.size() > 0)
        for (int m = CLI_ARGS::gridsMin[0]; m <= CLI_ARGS::gridsMax[0]; m *= 2)
            for (int n = CLI_ARGS::gridsMin[1]; n <= CLI_ARGS::gridsMax[1]; n *= 2)
                for (int o = CLI_ARGS::gridsMin[2]; o <= CLI_ARGS::gridsMax[2]; o *= 2)
                    gridsTBT.push_back({m, n, o});

    std::cout << "Testing the following grid sizes" << std::endl;
    for (auto gr : gridsTBT)
    {
        int m = gr[0];
        int n = gr[1];
        int o = gr[2];
        std::cout << "  " << m << "," << n << "," << o << std::endl;
    }

    for (auto Ns : gridsTBT)
    {
        int m = Ns[0];
        int n = Ns[1];
        int o = Ns[2];
        int ghosts_m = 1;
        int ghosts_n = 1;
        int ghosts_o = 1;
        double h = 1.0 / static_cast<double>(m);

        mgcl::BC bc = mgcl::BC::PERIODIC;
        bool periodic = bc == mgcl::BC::PERIODIC;
        int maxiter = 1;
        int benchiters = 10;

        ankerl::nanobench::Bench b;
        b.timeUnit(1us, "us")
            // .epochs(1)
            // .epochIterations(1)
            .minEpochTime(100ms)
            .maxEpochTime(5s)
            .relative(true);

        auto f = std::make_shared<mgcl::Cuboid>(m, n, o);
        auto v = std::make_shared<mgcl::Cuboid>(m, n, o);
        f->fillRandom(0, 10);

        mgcl::Problem p(m, n, o, f, v);
        p.setUseOpencl(true);
        if (mgcl_test::TestUtility::deviceAvailable("Quadro", CL_DEVICE_TYPE_GPU))
            p.setDeviceName("Quadro");
        p.setSilent(true);
        p.setProfilingEnabled(true);
        p.setStencilType(mgcl::MGCL_VARYING);
        p.createStencilValues()->fillRandom();
        p.init();

        // Clear measurements from init
        auto pd = p.getProfilingData();
        pd->getMeasurements().clear();

        auto& v0 = p.getLevelAt(0).getV();
        auto& v0d = p.getLevelAt(0).getDVIn();
        auto& fine = p.getLevelAt(0);
        auto& coarse = p.getLevelAt(1);

        for (int i = 0; i < benchiters; i++)
        {
            mgcl::MultigridEngine::jacobi(p, fine, maxiter, false);
            mgcl::MultigridEngine::residual(p, fine, true);
            mgcl::MultigridEngine::updateGhosts(p, v0d, nullptr, false);
            mgcl::MultigridEngine::restrict(fine, coarse, fine.getDR(), coarse.getDF());
            mgcl::MultigridEngine::prolongate(fine, coarse, fine.getDR(), coarse.getDVIn());
        }
        clFinish(p.getCommands());

        // Print profiling measurements
        std::cout << "m,n,o: " << m << "," << n << "," << o << std::endl;
        pd->printBestTimingsPerKernel();
        std::cout << "========" << std::endl;
    }
}

/**
 * @brief Measures each step of vcycle using the profiler, i.e. prints kernel timings only.
 *
 */
TEST_CASE("benchStepsProfileMpi")
{
    using std::min;

    if (CLI_ARGS::grids.size() == 0 && (CLI_ARGS::gridsMin.size() == 0 || CLI_ARGS::gridsMax.size() == 0))
        throw "Need to specify at least one local grid size, e.g. using --grids 4,8,16 or --gridsMin 4,4,4 AND --gridsMax 32,32,32";

    // build grids to be tested from CLI args
    std::vector<std::vector<int>> gridsTBT;
    for (auto N : CLI_ARGS::grids)
        gridsTBT.push_back({N, N, N});
    if (CLI_ARGS::gridsMin.size() > 0 && CLI_ARGS::gridsMax.size() > 0)
        for (int m = CLI_ARGS::gridsMin[0]; m <= CLI_ARGS::gridsMax[0]; m *= 2)
            for (int n = CLI_ARGS::gridsMin[1]; n <= CLI_ARGS::gridsMax[1]; n *= 2)
                for (int o = CLI_ARGS::gridsMin[2]; o <= CLI_ARGS::gridsMax[2]; o *= 2)
                    gridsTBT.push_back({m, n, o});

    // Check if mpi is initialized
    int isInitialized = 0;
    MPI_Initialized(&isInitialized);
    REQUIRE(isInitialized);

    MPI_Comm mpi_comm = MPI_COMM_WORLD;

    // Check number of processes
    int mpi_size = -1;
    MPI_Comm_size(mpi_comm, &mpi_size);
    // REQUIRE(mpi_size == 1);

    int periodic = 1;

    /* MPI variables */
    int mpi_rank;
    int mpi_dims[3] = {0, 0, 0};
    int mpi_periods[3] = {periodic, periodic, periodic};
    int mpi_coords[3];

    /* Initialize cartesian process grid */
    MPI_Comm_size(mpi_comm, &mpi_size);
    MPI_Dims_create(mpi_size, 3, mpi_dims);
    MPI_Cart_create(mpi_comm, 3, mpi_dims, mpi_periods, 1, &mpi_comm);
    MPI_Comm_rank(mpi_comm, &mpi_rank);
    MPI_Cart_coords(mpi_comm, mpi_rank, 3, mpi_coords);

    if (mpi_rank == 0)
    {
        std::cout << "Testing the following grid sizes" << std::endl;
        for (auto gr : gridsTBT)
        {
            int m = gr[0];
            int n = gr[1];
            int o = gr[2];
            std::cout << "  " << m << "," << n << "," << o << std::endl;
        }
    }

    std::vector<bench_util::ResultGhostUpdateMpi> results;

    bool printedGpu = false;
    for (auto gr : gridsTBT)
    {
        int m = gr[0];
        int n = gr[1];
        int o = gr[2];
        int mglob = m * mpi_dims[0];
        int nglob = n * mpi_dims[1];
        int oglob = o * mpi_dims[2];
        int ghosts = 1;
        int mgh = m + 2 * ghosts;
        int ngh = n + 2 * ghosts;
        int ogh = o + 2 * ghosts;

        mgcl::BC bc = mgcl::BC::PERIODIC;
        bool periodic = bc == mgcl::BC::PERIODIC;
        int maxiter = 1;
        int benchiters = CLI_ARGS::bench_iterations;

        auto f = std::make_shared<mgcl::Cuboid>(m, n, o);
        auto v = std::make_shared<mgcl::Cuboid>(m, n, o);
        f->fillRandom(0, 10);

        mgcl::Problem p(m, n, o, f, v, mglob, nglob, oglob);
        p.setUseOpencl(true);
        if (mgcl_test::TestUtility::deviceAvailable("Quadro", CL_DEVICE_TYPE_GPU))
            p.setDeviceName("Quadro");
        p.setSilent(true);
        p.setProfilingEnabled(true);
        p.setStencilType(mgcl::MGCL_VARYING);
        p.createStencilValues()->fillRandom();
        p.setMpiComm(mpi_comm);
        p.init();

        if (!printedGpu)
        {
            for (int i = 0; i < mpi_size; i++)
            {
                MPI_Barrier(mpi_comm);
                if (i == mpi_rank)
                {
                    std::cout << "on rank " << mpi_rank << ", GPU info: ";
                    p.getOpenCLHelper().outputDeviceInfo();
                }
            }
            printedGpu = true;
        }

        // Clear measurements from init
        auto pd = p.getProfilingData();
        pd->getMeasurements().clear();

        auto& v0 = p.getLevelAt(0).getV();
        auto& v0d = p.getLevelAt(0).getDVIn();
        auto& fine = p.getLevelAt(0);
        auto& coarse = p.getLevelAt(1);
        auto& mpiData = p.getLevelAt(0).getMpiData();

        for (int i = 0; i < benchiters; i++)
        {
            MPI_Barrier(mpi_comm);
            mgcl::MultigridEngine::jacobi(p, fine, maxiter, false);
            mgcl::MultigridEngine::residual(p, fine, true);
            mgcl::MultigridEngine::updateGhosts(p, v0d, &mpiData, false);
            mgcl::MultigridEngine::restrict(fine, coarse, fine.getDR(), coarse.getDF());
            mgcl::MultigridEngine::prolongate(fine, coarse, fine.getDR(), coarse.getDVIn());
        }
        clFinish(p.getCommands());
        MPI_Barrier(mpi_comm);

        // Print profiling measurements
        if (mpi_rank == 0)
        {
            std::cout << "m,n,o: " << m << "," << n << "," << o << std::endl;
            pd->printBestTimingsPerKernel();
            std::cout << "========" << std::endl;
        }

        MPI_Barrier(mpi_comm);
    }
}