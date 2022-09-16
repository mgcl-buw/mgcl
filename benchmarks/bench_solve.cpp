#include "nanobench.h"

#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
using namespace std::chrono_literals;

#include "../cuboid.hpp"
#include "../problem.hpp"
#include "../test/test_utility.hpp"
#include "bench_render_templates.hpp"
#include "pmg_utility.hpp"

#include "../thirdparty/pmg/mg.h"

TEST_CASE("mgcl benchmarks console: solve", "[!benchmark][solve][console]")
{
    std::vector grids{16, 32, 64, 128};
    for (auto N : grids)
    {
        // int N = 16;
        int m = N;
        int n = N;
        int o = N;

        int maxIterVCycles = 30;

        ankerl::nanobench::Bench b;
        b.timeUnit(1ms, "ms")
            // .epochs(1)
            // .epochIterations(1)
            .minEpochTime(100ms)
            .maxEpochTime(5s)
            .relative(true);

        auto f = std::make_shared<mgcl::Cuboid>(m, n, o);
        f->fillRandom(0, 10);

        // if (N >= 128)
        //     b.epochs(3);

        {
            auto v = std::make_shared<mgcl::Cuboid>(m, n, o);

            mgcl::Problem p(m, n, o, f, v);
            p.setMaxiterVcycles(maxIterVCycles);
            p.setIgnoreTol(true);
            p.setSilent(true);
            p.init();

            b.run(std::string("sequential random values, N = ").append(std::to_string(N)).c_str(), [&]
                  { p.solveSeq(); });
        }

        if (mgcl_test::TestUtility::deviceAvailable("", CL_DEVICE_TYPE_GPU))
        {
            auto v = std::make_shared<mgcl::Cuboid>(m, n, o);

            mgcl::Problem p(m, n, o, f, v);
            p.setMaxiterVcycles(maxIterVCycles);
            p.setIgnoreTol(true);
            p.setUseOpencl(true);
            p.setDeviceType(CL_DEVICE_TYPE_GPU);
            p.setSilent(true);

            if (mgcl_test::TestUtility::deviceAvailable("Quadro", p.getDeviceType()))
                p.setDeviceName("Quadro");

            p.init();
            b.run(std::string("opencl gpu random values, N = ").append(std::to_string(N)).c_str(), [&]
                  { p.solve(); });
        }

        {
            // pmg

            // setup MPI
            MPI_Comm mpi_comm_cart = *init_mpi_for_pmg();

            if (mpi_comm_cart == MPI_COMM_NULL)
            {
                std::cout << "mpi_comm_cart is null! Cannot test against pmg." << std::endl;
            }
            else
            {
                // Problem parameters
                auto v = std::make_shared<mgcl::Cuboid>(m, n, o);
                double tol = 1e-20;
                int nu1 = 2;
                int nu2 = 2;
                double omega = 0.8;

                // init 7-point stencil for pmg's jacobi
                int size = 7;
                double h2 = 1.0 / (double)(N * N);
                double *values = new double[size]();
                int *xoff = new int[size]();
                int *yoff = new int[size]();
                int *zoff = new int[size]();

                values[0] = 6.0 / h2;
                for (int i = 1; i <= 6; i++)
                    values[i] = (-1.0) / h2;

                xoff[0] = 0;
                xoff[1] = 1;
                xoff[2] = -1;
                for (int i = 3; i <= 6; i++)
                    xoff[i] = 0;

                for (int i = 0; i <= 2; i++)
                    yoff[i] = 0;
                yoff[3] = 1;
                yoff[4] = -1;
                yoff[5] = 0;
                yoff[6] = 0;

                for (int i = 0; i <= 4; i++)
                    zoff[i] = 0;
                zoff[5] = 1;
                zoff[6] = -1;

                // run with a tolerance that will never be reached thus all vcycle iters are executed
                b.run(std::string("pmg random values, N = ").append(std::to_string(N)).c_str(), [&]
                      { mg(v->getData(), f->getData(), maxIterVCycles, tol, m, n, o,
                           0, m - 1, 0, n - 1, 0, o - 1,
                           1, nu1, nu2, omega, size, values, xoff, yoff, zoff, mpi_comm_cart, 1); });
            }
        }

        // if (mgcl_test::TestUtility::deviceAvailable("", CL_DEVICE_TYPE_CPU))
        // {
        //     b.epochs(1).epochIterations(1);

        //     auto v = std::make_shared<mgcl::Cuboid>(m, n, o);

        //     mgcl::Problem p(m, n, o, f, v);
        //     p.setMaxiterVcycles(maxIterVCycles);
        //     p.setIgnoreTol(true);
        //     p.setUseOpencl(true);
        //     p.setDeviceType(CL_DEVICE_TYPE_CPU);
        //     p.setSilent(true);

        //     if (mgcl_test::TestUtility::deviceAvailable("i7-10875H", p.getDeviceType()))
        //         p.setDeviceName("i7-10875H");

        //     p.init();
        //     b.run(std::string("opencl cpu random values, N = ").append(std::to_string(N)).c_str(), [&]
        //           { p.solve(); });
        // }

        std::ofstream renderOut(std::string("solvingBoxplot_").append(std::to_string(N)).append(".html"));
        b.render(ankerl::nanobench::templates::htmlBoxplot(), renderOut);
    }
    MPI_Finalize();
}

TEST_CASE("mgcl benchmarks lineplot: solve", "[!benchmark][solve][plot]")
{
    ankerl::nanobench::Bench b;
    b.timeUnit(1ms, "ms")
        .minEpochTime(100ms);

    std::vector<int> grids{16, 32, 64, 128};
    for (auto N : grids)
    {
        int m = N;
        int n = N;
        int o = N;

        {
            auto v = std::make_shared<mgcl::Cuboid>(m, n, o);
            auto f = std::make_shared<mgcl::Cuboid>(m, n, o);
            v->fillRandom(0, 10);
            f->fillRandom(0, 10);

            mgcl::Problem p(m, n, o, f, v);
            p.setMaxiterVcycles(10);
            p.setIgnoreTol(true);
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
            p.setMaxiterVcycles(10);
            p.setIgnoreTol(true);
            p.setUseOpencl(true);
            p.setDeviceType(CL_DEVICE_TYPE_GPU);
            p.setSilent(true);

            mgcl_test::TestUtility tu;
            if (tu.deviceAvailable("Quadro", p.getDeviceType()))
                p.setDeviceName("Quadro");

            p.init();
            b.run(std::string("opencl random values, N = ").append(std::to_string(N)).c_str(), [&]
                  { p.solve(); });
        }
    }

    std::ofstream renderOut(std::string("benchresults_lineplot").append(".html"));
    ankerl::nanobench::render(mgcl_test::htmlLineComparingN(), b, renderOut);
}

/**
 * @brief Runs mgcl::solve with different vcycle iteration counts. Results are compared for only one grid size at
 * a time. The resulting line plot shows the ratio sequential/opencl on y-axis and vcycleIterations on x-axis.
 *
 */
TEST_CASE("bench vcycle iterations lineplot: solve", "[!benchmark][solve][plot][vcycleiters]")
{
    // std::vector<int> grids{16, 32, 64, 128};
    std::vector<int> grids{32};

    int iters_start = 10;
    int iters_stop = 100;
    int iters_step = 5;

    for (auto N : grids)
    {
        ankerl::nanobench::Bench b;
        b.timeUnit(1ms, "ms")
            .minEpochTime(100ms);

        int m = N;
        int n = N;
        int o = N;

        auto f = std::make_shared<mgcl::Cuboid>(m, n, o);
        f->fillRandom(0, 10);

        for (int iters = iters_start; iters <= iters_stop; iters += iters_step)
        {
            {
                auto v = std::make_shared<mgcl::Cuboid>(m, n, o);
                mgcl::Problem p(m, n, o, f, v);
                p.setMaxiterVcycles(iters);
                p.setIgnoreTol(true);
                p.setSilent(true);
                p.init();

                b.run(std::string("seq, N = ").append(std::to_string(N).append(", vcycle iters = ").append(std::to_string(iters))).c_str(), [&]
                      { p.solveSeq(); });
            }

            {
                auto v = std::make_shared<mgcl::Cuboid>(m, n, o);

                mgcl::Problem p(m, n, o, f, v);
                p.setMaxiterVcycles(iters);
                p.setIgnoreTol(true);
                p.setUseOpencl(true);
                p.setDeviceType(CL_DEVICE_TYPE_GPU);
                p.setSilent(true);

                mgcl_test::TestUtility tu;
                if (tu.deviceAvailable("Quadro", p.getDeviceType()))
                    p.setDeviceName("Quadro");

                p.init();
                b.run(std::string("ocl, N = ").append(std::to_string(N).append(", vcycle iters = ").append(std::to_string(iters))).c_str(), [&]
                      { p.solve(); });
            }
        }

        std::ofstream renderOut(std::string("benchresults_vcycleiters_").append(std::to_string(N)).append(".html"));
        ankerl::nanobench::render(mgcl_test::htmlLineComparingVcycleIters(), b, renderOut);
    }
}
