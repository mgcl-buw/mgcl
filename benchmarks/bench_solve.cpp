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

#include "../thirdparty/mgcl_c/mgcl.hpp"
#include "../thirdparty/pmg/mg.h"

TEST_CASE("mgcl benchmarks console: solve", "[!benchmark][solve][console][Laplace7p]")
{
    std::vector grids{16, 32, 64, 128};
    for (auto N : grids)
    {
        // int N = 16;
        int m = N;
        int n = N;
        int o = N;

        // Problem parameters
        double tol = 1e-20;
        int nu1 = 2;
        int nu2 = 2;
        double omega = 0.8;
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
            p.setNu1(nu1);
            p.setNu2(nu2);
            p.setOmega(omega);
            // p.init();

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
            p.setNu1(nu1);
            p.setNu2(nu2);
            p.setOmega(omega);

            if (mgcl_test::TestUtility::deviceAvailable("Quadro", p.getDeviceType()))
                p.setDeviceName("Quadro");

            // p.init();
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
                auto v = std::make_shared<mgcl::Cuboid>(m, n, o);

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

        {
            // old mgcl c implementation seq

            auto v = std::make_shared<mgcl::Cuboid>(m, n, o);

            mgcl_config *conf;
            mgcl_generate_config(&conf);

            conf->v = v->getData();
            conf->f = f->getData();
            conf->m = N;
            conf->n = N;
            conf->o = N;
            conf->ghosts_in = 0;
            conf->nu1 = nu1;
            conf->nu2 = nu2;
            conf->omega = omega;
            conf->maxiter_vcycles = maxIterVCycles;
            conf->silent = 1;
            conf->ignoreTol = 1;

            b.run(std::string("old mgcl_c seq random values, N = ").append(std::to_string(N)).c_str(), [&]
                  { mgcl_c_mgcl_seq(conf); });
        }

        if (mgcl_test::TestUtility::deviceAvailable("", CL_DEVICE_TYPE_GPU))
        {
            // old mgcl c implementation ocl

            auto v = std::make_shared<mgcl::Cuboid>(m, n, o);

            mgcl_config *conf;
            mgcl_generate_config(&conf);

            conf->v = v->getData();
            conf->f = f->getData();
            conf->m = N;
            conf->n = N;
            conf->o = N;
            conf->ghosts_in = 0;
            conf->nu1 = nu1;
            conf->nu2 = nu2;
            conf->omega = omega;
            conf->maxiter_vcycles = maxIterVCycles;
            conf->device_type = CL_DEVICE_TYPE_GPU;
            // conf->kernel_dir = ".";

            if (mgcl_test::TestUtility::deviceAvailable("Quadro", CL_DEVICE_TYPE_GPU))
                conf->device_name = "Quadro";

            conf->silent = 1;
            conf->ignoreTol = 1;
            conf->use_opencl = 1;
            conf->read_results = 1;

            b.run(std::string("old mgcl_c ocl random values, N = ").append(std::to_string(N)).c_str(), [&]
                  { mgcl_c_mgcl(conf); });
        }

        // std::ofstream renderOut(std::string("solvingBoxplot_").append(std::to_string(N)).append(".html"));
        // b.render(ankerl::nanobench::templates::htmlBoxplot(), renderOut);

        std::ofstream renderOutCsv(std::string("solvingCsv_").append(std::to_string(N)).append(".csv"));
        b.render(ankerl::nanobench::templates::csv(), renderOutCsv);
    }
    MPI_Finalize();
}

TEST_CASE("mgcl bench: solve, all stencils", "[!benchmark][solve][console][allStencils]")
{
    std::vector grids{4, 8, 16, 32, 64, 128};
    std::vector stencils{
        // mgcl::MGCL_LAPLACE_7POINT,
        // mgcl::MGCL_LAPLACE_19POINT,
        // mgcl::MGCL_LAPLACE_27POINT,
        // mgcl::MGCL_VARYING_7POINT,
        // mgcl::MGCL_VARYING_19POINT,
        mgcl::MGCL_VARYING_27POINT};

    // Problem parameters
    double tol = 1e-20;
    int nu1 = 2;
    int nu2 = 2;
    double omega = 0.8;
    int maxIterVCycles = 30;

    std::cout << "Problem parameters:" << std::endl
              << "  tol: " << tol << std::endl
              << "  nu1: " << nu1 << std::endl
              << "  nu2: " << nu2 << std::endl
              << "  omega: " << omega << std::endl
              << "  v-cycle iterations: " << maxIterVCycles << std::endl;

    for (auto N : grids)
    {
        for (auto stencil : stencils)
        {
            // int N = 16;
            int m = N;
            int n = N;
            int o = N;

            ankerl::nanobench::Bench b;
            b.timeUnit(1ms, "ms")
                // .epochs(1)
                // .epochIterations(1)
                .minEpochTime(100ms)
                .relative(true);

            if (N <= 16)
                b.minEpochIterations(20);

            auto f = std::make_shared<mgcl::Cuboid>(m, n, o);
            f->fillRandom(0, 10);

            std::string stencilName = "Laplace7p";
            if (stencil == mgcl::MGCL_LAPLACE_19POINT)
                stencilName = "Laplace19p";
            else if (stencil == mgcl::MGCL_LAPLACE_27POINT)
                stencilName = "Laplace27p";
            else if (stencil == mgcl::MGCL_VARYING_7POINT)
                stencilName = "Varying7p";
            else if (stencil == mgcl::MGCL_VARYING_19POINT)
                stencilName = "Varying19p";
            else if (stencil == mgcl::MGCL_VARYING_27POINT)
                stencilName = "Varying27p";

            // auto stencilValues;

            // if (N >= 128)
            //     b.epochs(3);

            {
                auto v = std::make_shared<mgcl::Cuboid>(m, n, o);

                mgcl::Problem p(m, n, o, f, v);
                p.setMaxiterVcycles(maxIterVCycles);
                p.setIgnoreTol(true);
                p.setSilent(true);
                p.setNu1(nu1);
                p.setNu2(nu2);
                p.setOmega(omega);
                p.setStencilType(stencil);
                p.setReadResults(true);

                if (stencil == mgcl::MGCL_VARYING_27POINT)
                    p.getStencilValues()->fillRandomInt();
                // p.init();

                b.run(std::string("sequential random values, N = ").append(std::to_string(N)).append(", ").append(stencilName).c_str(), [&]
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
                p.setNu1(nu1);
                p.setNu2(nu2);
                p.setOmega(omega);
                p.setStencilType(stencil);
                p.setReadResults(true);

                if (stencil == mgcl::MGCL_VARYING_27POINT)
                    p.getStencilValues()->fillRandomInt();

                if (mgcl_test::TestUtility::deviceAvailable("Quadro", p.getDeviceType()))
                    p.setDeviceName("Quadro");

                // p.init();
                b.run(std::string("opencl gpu random values, N = ").append(std::to_string(N)).append(", ").append(stencilName).c_str(), [&]
                      { p.solve(); });
            }

            // std::ofstream renderOut(std::string("solvingBoxplot_").append(std::to_string(N)).append(".html"));
            // b.render(ankerl::nanobench::templates::htmlBoxplot(), renderOut);

            // std::ofstream renderOutCsv(std::string("solvingCsv_").append(std::to_string(N)).append(".csv"));
            // b.render(ankerl::nanobench::templates::csv(), renderOutCsv);
        }
    }
}

TEST_CASE("mgcl old vs new: solve equality", "[!benchmark][solve][console][equality]")
{
    std::vector grids{16};
    for (auto N : grids)
    {
        // int N = 16;
        int m = N;
        int n = N;
        int o = N;

        // Problem parameters
        double tol = 1e-20;
        int nu1 = 2;
        int nu2 = 2;
        double omega = 0.8;
        int maxIterVCycles = 30;

        auto f = std::make_shared<mgcl::Cuboid>(m, n, o);
        f->fillRandom(0, 10);

        // new mgcl seq
        auto vseq = std::make_shared<mgcl::Cuboid>(m, n, o);

        mgcl::Problem p(m, n, o, f, vseq);
        p.setMaxiterVcycles(maxIterVCycles);
        p.setIgnoreTol(true);
        p.setSilent(true);
        p.setNu1(nu1);
        p.setNu2(nu2);
        p.setOmega(omega);
        // p.init();
        p.solveSeq();

        // new mgcl ocl
        auto vocl = std::make_shared<mgcl::Cuboid>(m, n, o);

        mgcl::Problem pocl(m, n, o, f, vocl);
        pocl.setMaxiterVcycles(maxIterVCycles);
        pocl.setIgnoreTol(true);
        pocl.setSilent(true);
        pocl.setNu1(nu1);
        pocl.setNu2(nu2);
        pocl.setOmega(omega);
        pocl.solveSeq();

        // old mgcl c implementation seq
        auto vold = std::make_shared<mgcl::Cuboid>(m, n, o);

        mgcl_config *conf;
        mgcl_generate_config(&conf);

        conf->v = vold->getData();
        conf->f = f->getData();
        conf->m = N;
        conf->n = N;
        conf->o = N;
        conf->ghosts_in = 0;
        conf->nu1 = nu1;
        conf->nu2 = nu2;
        conf->omega = omega;
        conf->maxiter_vcycles = maxIterVCycles;
        conf->silent = 1;
        conf->ignoreTol = 1;

        mgcl_c_mgcl_seq(conf);

        // old mgcl c implementation ocl
        auto voldocl = std::make_shared<mgcl::Cuboid>(m, n, o);

        mgcl_config *conf_ocl;
        mgcl_generate_config(&conf_ocl);

        conf_ocl->v = voldocl->getData();
        conf_ocl->f = f->getData();
        conf_ocl->m = N;
        conf_ocl->n = N;
        conf_ocl->o = N;
        conf_ocl->ghosts_in = 0;
        conf_ocl->nu1 = nu1;
        conf_ocl->nu2 = nu2;
        conf_ocl->omega = omega;
        conf_ocl->maxiter_vcycles = maxIterVCycles;
        conf_ocl->silent = 1;
        conf_ocl->ignoreTol = 1;
        conf_ocl->use_opencl = 1;
        conf_ocl->device_type = CL_DEVICE_TYPE_GPU;
        conf_ocl->read_results = 1;

        if (mgcl_test::TestUtility::deviceAvailable("Quadro", CL_DEVICE_TYPE_GPU))
            conf_ocl->device_name = "Quadro";

        mgcl_c_mgcl(conf_ocl);

        // vocl->dumpToFile("out_vocl.txt");
        // voldocl->dumpToFile("out_voldocl.txt");

        REQUIRE(vseq->isEqual(*vold));
        REQUIRE(vocl->isEqual(*voldocl));
    }
}

TEST_CASE("mgcl benchmarks console: solve", "[!benchmark][solveWithoutInit][console]")
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
