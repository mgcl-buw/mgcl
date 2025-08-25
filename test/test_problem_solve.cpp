#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>

#include "../benchmarks/pmg_utility.hpp"
#include "../src/mgcl/cuboid.hpp"
#include "../src/mgcl/problem.hpp"
#include "../thirdparty/pmg/mg.h"
#include "cli_args.hpp"
#include "device_type_generator.hpp"
#include "test_utility.hpp"

/**
 * @brief Tests if solving works correctly for u = x^4 * (x-1)^4.
 *
 */
TEST_CASE("solve_periodic")
{
    int N = 16;
    double h = 1.0 / (double)N;

    // Problem parameters
    double tol = 1e-14;
    int nu1 = 2;
    int nu2 = 2;
    double omega = 0.8;
    int maxIterVCycles = 10;
    int maxlevel = 10;

    auto v = std::make_shared<mgcl::Cuboid>(N, N, N);
    auto f = std::make_shared<mgcl::Cuboid>(N, N, N);
    auto solution = mgcl::Cuboid(N, N, N);

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
            {
                double zs = i * h;
                double ys = j * h;
                double xs = k * h;
                double xs2 = xs * xs;
                double ys2 = ys * ys;
                double zs2 = zs * zs;
                double xsm1_2 = (xs - 1) * (xs - 1);
                double ysm1_2 = (ys - 1) * (ys - 1);
                double zsm1_2 = (zs - 1) * (zs - 1);
                double xs3 = xs * xs * xs;
                double ys3 = ys * ys * ys;
                double zs3 = zs * zs * zs;
                double xsm1_3 = (xs - 1) * (xs - 1) * (xs - 1);
                double ysm1_3 = (ys - 1) * (ys - 1) * (ys - 1);
                double zsm1_3 = (zs - 1) * (zs - 1) * (zs - 1);
                double xs4 = xs * xs * xs * xs;
                double ys4 = ys * ys * ys * ys;
                double zs4 = zs * zs * zs * zs;
                double xsm1_4 = (xs - 1) * (xs - 1) * (xs - 1) * (xs - 1);
                double ysm1_4 = (ys - 1) * (ys - 1) * (ys - 1) * (ys - 1);
                double zsm1_4 = (zs - 1) * (zs - 1) * (zs - 1) * (zs - 1);
                (*v)[i][j][k] = 0;
                solution[i][j][k] = 1000000 * (xs * (xs - 1)) * (xs * (xs - 1)) * (xs * (xs - 1)) * (xs * (xs - 1)) *
                                    (ys * (ys - 1)) * (ys * (ys - 1)) * (ys * (ys - 1)) * (ys * (ys - 1)) *
                                    (zs * (zs - 1)) * (zs * (zs - 1)) * (zs * (zs - 1)) * (zs * (zs - 1));
                (*f)[i][j][k] =
                    -1000000 *
                    (12 * xs4 * ys4 * zs4 * xsm1_4 * ysm1_4 * zsm1_2 + 12 * xs4 * ys4 * zs4 * xsm1_4 * ysm1_2 * zsm1_4 +
                     12 * xs4 * ys4 * zs4 * xsm1_2 * ysm1_4 * zsm1_4 + 32 * xs4 * ys4 * zs3 * xsm1_4 * ysm1_4 * zsm1_3 +
                     12 * xs4 * ys4 * zs2 * xsm1_4 * ysm1_4 * zsm1_4 + 32 * xs4 * ys3 * zs4 * xsm1_4 * ysm1_3 * zsm1_4 +
                     12 * xs4 * ys2 * zs4 * xsm1_4 * ysm1_4 * zsm1_4 + 32 * xs3 * ys4 * zs4 * xsm1_3 * ysm1_4 * zsm1_4 +
                     12 * xs2 * ys4 * zs4 * xsm1_4 * ysm1_4 * zsm1_4);
            }

    mgcl::Problem p(N, N, N, f, v);
    p.setMaxiterVcycles(maxIterVCycles);
    p.setTol(tol);
    p.setNu1(nu1);
    p.setNu2(nu2);
    p.setOmega(omega);
    p.setMaxlevel(maxlevel);

    SECTION("Sequential")
    {
        SECTION("Laplace_7p")
        {
            p.solveSeq();

            // check if input v is equal to the v stored in Problem instance
            REQUIRE(v.get() == p.getVPtr().get());
            REQUIRE(v->isEqual(p.getV()));

            // check if solution is good
            auto err = mgcl_test::calculateError(solution, *v);
            auto errNorm = mgcl_test::calculateErrorNorm(1.0 / (double)N, *err);
            auto errMax = mgcl_test::calculateMaxError(*err);

            // solution.dumpToFile("out_solution.txt");
            // (*v).dumpToFile("out_v.txt");

            std::cout
                << "seq Laplace 7p" << std::endl
                << std::scientific << std::setprecision(17) << "  ||e||_2 = " << errNorm << std::endl
                << std::scientific << std::setprecision(17) << "  e_max = " << errMax << std::endl;

            CHECK(errNorm < 1e-2);
            CHECK(errMax < 1e-2);

            // check if error is equal to old mgcl implementation (problem params must match)
            if (p.getMaxiterVcycles() == 10 && N == 32 && p.getTol() == 1e-14 &&
                p.getNu1() == 2 && p.getNu2() == 2 && p.getOmega() == 0.8 &&
                p.getStencilType() == mgcl::MGCL_LAPLACE_7POINT)
            {
                CHECK(fabs(errNorm - 3.93115528889639940e-03) < 1e-14);
                CHECK(fabs(errMax - 3.95723982871564600e-03) < 1e-14);
            }
        }

        SECTION("Galerkin_7p")
        {
            p.setStencilType(mgcl::MGCL_VARYING);
            auto& s = *p.getStencilValues();

            double h = 1.0 / static_cast<double>(N);
            mgcl_test::fill7pLaplace(s, h, false);

            p.solveSeq();

            // check if input v is equal to the v stored in Problem instance
            REQUIRE(v.get() == p.getVPtr().get());
            REQUIRE(v->isEqual(p.getV()));

            // check if solution is good
            auto err = mgcl_test::calculateError(solution, *v);
            auto errNorm = mgcl_test::calculateErrorNorm(1.0 / (double)N, *err);
            auto errMax = mgcl_test::calculateMaxError(*err);

            // solution.dumpToFile("out_solution.txt");
            // (*v).dumpToFile("out_v.txt");

            std::cout
                << "seq Galerkin 7p" << std::endl
                << std::scientific << std::setprecision(17) << "  ||e||_2 = " << errNorm << std::endl
                << std::scientific << std::setprecision(17) << "  e_max = " << errMax << std::endl;

            CHECK(errNorm < 1e-2);
            CHECK(errMax < 1e-2);
        }

        SECTION("Laplace_27p")
        {
            p.setStencilType(mgcl::MGCL_LAPLACE_27POINT);
            p.solveSeq();

            // check if input v is equal to the v stored in Problem instance
            REQUIRE(v.get() == p.getVPtr().get());
            REQUIRE(v->isEqual(p.getV()));

            // check if solution is good
            auto err = mgcl_test::calculateError(solution, *v);
            auto errNorm = mgcl_test::calculateErrorNorm(1.0 / (double)N, *err);
            auto errMax = mgcl_test::calculateMaxError(*err);

            // solution.dumpToFile("out_solution.txt");
            // (*v).dumpToFile("out_v.txt");

            std::cout
                << "seq Laplace 27p" << std::endl
                << std::scientific << std::setprecision(17) << "  ||e||_2 = " << errNorm << std::endl
                << std::scientific << std::setprecision(17) << "  e_max = " << errMax << std::endl;

            CHECK(errNorm < 1e-2);
            CHECK(errMax < 1e-2);

            // check if error is equal to old mgcl implementation (problem params must match)
            if (p.getMaxiterVcycles() == 10 && N == 32 && p.getTol() == 1e-14 &&
                p.getNu1() == 2 && p.getNu2() == 2 && p.getOmega() == 0.8 &&
                p.getStencilType() == mgcl::MGCL_LAPLACE_7POINT)
            {
                CHECK(fabs(errNorm - 3.93115528889639940e-03) < 1e-14);
                CHECK(fabs(errMax - 3.95723982871564600e-03) < 1e-14);
            }
        }

        SECTION("Galerkin_27p")
        {
            p.setStencilType(mgcl::MGCL_VARYING);
            auto& s = *p.getStencilValues();

            double h = 1.0 / static_cast<double>(N);
            mgcl_test::fill27pLaplace(s, h, false);

            p.solveSeq();

            // check if input v is equal to the v stored in Problem instance
            REQUIRE(v.get() == p.getVPtr().get());
            REQUIRE(v->isEqual(p.getV()));

            // check if solution is good
            auto err = mgcl_test::calculateError(solution, *v);
            auto errNorm = mgcl_test::calculateErrorNorm(1.0 / (double)N, *err);
            auto errMax = mgcl_test::calculateMaxError(*err);

            // solution.dumpToFile("out_solution.txt");
            // (*v).dumpToFile("out_v.txt");

            std::cout
                << "seq Galerkin 27p" << std::endl
                << std::scientific << std::setprecision(17) << "  ||e||_2 = " << errNorm << std::endl
                << std::scientific << std::setprecision(17) << "  e_max = " << errMax << std::endl;

            CHECK(errNorm < 1e-2);
            CHECK(errMax < 1e-2);
        }
    }

    SECTION("OpenCL")
    {
        auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));

        std::string oclDeviceType = deviceType == CL_DEVICE_TYPE_GPU ? "GPU" : "CPU";

        p.setUseOpencl(true);
        p.setDeviceType(CL_DEVICE_TYPE_GPU);
        p.setReadResults(true);
        p.setDeviceType(deviceType);
        // p.setDeviceName("Quadro");

        SECTION("Laplace_7p")
        {
            p.solve();

            // check if solution is good
            auto err = mgcl_test::calculateError(solution, *v);
            auto errNorm = mgcl_test::calculateErrorNorm(1.0 / (double)N, *err);
            auto errMax = mgcl_test::calculateMaxError(*err);

            std::cout
                << "ocl " << oclDeviceType << " Laplace 7p" << std::endl
                << std::scientific << "  ||e||_2 = " << errNorm << std::endl
                << std::scientific << "  e_max = " << errMax << std::endl;

            CHECK(errNorm < 1e-2);
            CHECK(errMax < 1e-2);

            // check if error is equal to old mgcl implementation (problem params must match)
            if (p.getMaxiterVcycles() == 10 && N == 32 && p.getTol() == 1e-14 &&
                p.getNu1() == 2 && p.getNu2() == 2 && p.getOmega() == 0.8 &&
                p.getDeviceName() == "Quadro" && p.getDeviceType() == CL_DEVICE_TYPE_GPU)
            {
                CHECK(fabs(errNorm - 3.93115528889612358e-03) < 1e-14);
                CHECK(fabs(errMax - 3.95723982871536324e-03) < 1e-14);
            }
        }

        SECTION("Galerkin_7p")
        {
            p.setStencilType(mgcl::MGCL_VARYING);
            auto& s = *p.getStencilValues();

            double h = 1.0 / static_cast<double>(N);
            mgcl_test::fill7pLaplace(s, h, false);

            p.solve();

            // check if solution is good
            auto err = mgcl_test::calculateError(solution, *v);
            auto errNorm = mgcl_test::calculateErrorNorm(1.0 / (double)N, *err);
            auto errMax = mgcl_test::calculateMaxError(*err);

            // solution.dumpToFile("out_solution.txt");
            // (*v).dumpToFile("out_v.txt");

            std::cout
                << "ocl " << oclDeviceType << " Galerkin 7p" << std::endl
                << std::scientific << std::setprecision(17) << "  ||e||_2 = " << errNorm << std::endl
                << std::scientific << std::setprecision(17) << "  e_max = " << errMax << std::endl;

            CHECK(errNorm < 1e-2);
            CHECK(errMax < 1e-2);
        }

        SECTION("Laplace_27p")
        {
            p.setStencilType(mgcl::MGCL_LAPLACE_27POINT);
            p.solve();

            // check if solution is good
            auto err = mgcl_test::calculateError(solution, *v);
            auto errNorm = mgcl_test::calculateErrorNorm(1.0 / (double)N, *err);
            auto errMax = mgcl_test::calculateMaxError(*err);

            std::cout
                << "ocl " << oclDeviceType << " Laplace 27p" << std::endl
                << std::scientific << "  ||e||_2 = " << errNorm << std::endl
                << std::scientific << "  e_max = " << errMax << std::endl;

            CHECK(errNorm < 1e-2);
            CHECK(errMax < 1e-2);
        }

        SECTION("Galerkin_27p")
        {
            p.setStencilType(mgcl::MGCL_VARYING);
            auto& s = *p.getStencilValues();

            double h = 1.0 / static_cast<double>(N);
            mgcl_test::fill27pLaplace(s, h, false);

            p.solve();

            // check if solution is good
            auto err = mgcl_test::calculateError(solution, *v);
            auto errNorm = mgcl_test::calculateErrorNorm(1.0 / (double)N, *err);
            auto errMax = mgcl_test::calculateMaxError(*err);

            // solution.dumpToFile("out_solution.txt");
            // (*v).dumpToFile("out_v.txt");

            std::cout
                << "ocl " << oclDeviceType << " Galerkin 27p" << std::endl
                << std::scientific << std::setprecision(17) << "  ||e||_2 = " << errNorm << std::endl
                << std::scientific << std::setprecision(17) << "  e_max = " << errMax << std::endl;

            CHECK(errNorm < 1e-2);
            CHECK(errMax < 1e-2);
        }
    }

    SECTION("pmg")
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
            // auto v = std::make_shared<mgcl::Cuboid>(N, N, N);

            // init 7-point stencil for pmg's jacobi
            int size = 7;
            double h2 = 1.0 / (double)(N * N);
            double* values = new double[size]();
            int* xoff = new int[size]();
            int* yoff = new int[size]();
            int* zoff = new int[size]();

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
            mg_with_maxlv(v->getData(), f->getData(), maxIterVCycles, tol, N, N, N, 0, N - 1, 0, N - 1, 0, N - 1,
                          1, nu1, nu2, omega, size, values, xoff, yoff, zoff, mpi_comm_cart, 0, maxlevel + 1);

            // check if solution is good
            auto err = mgcl_test::calculateError(solution, *v);
            auto errNorm = mgcl_test::calculateErrorNorm(1.0 / (double)N, *err);
            auto errMax = mgcl_test::calculateMaxError(*err);

            std::cout
                << "pmg" << std::endl
                << std::scientific << "  ||e||_2 = " << errNorm << std::endl
                << std::scientific << "  e_max = " << errMax << std::endl;

            CHECK(errNorm < 1e-2);
            CHECK(errMax < 1e-2);

            // check if error is equal to old mgcl implementation (problem params must match)
            if (p.getMaxiterVcycles() == 10 && N == 32 && p.getTol() == 1e-14 &&
                p.getNu1() == 2 && p.getNu2() == 2 && p.getOmega() == 0.8 &&
                p.getDeviceName() == "Quadro" && p.getDeviceType() == CL_DEVICE_TYPE_GPU)
            {
                CHECK(fabs(errNorm - 3.93115528889612358e-03) < 1e-14);
                CHECK(fabs(errMax - 3.95723982871536324e-03) < 1e-14);
            }
        }

        // Gets called in custom catch2 main
        // MPI_Finalize();
    }
}

TEST_CASE("solve_dirichlet")
{
    int N = 16;
    double h = 1.0 / (double)N;

    // Problem parameters
    double tol = 1e-7;
    int nu1 = 2;
    int nu2 = 2;
    double omega = 0.8;
    int maxIterVCycles = 20;
    int maxlevel = 10;

    mgcl::BC bc = mgcl::BC::DIRICHLET;
    int ghin = 1;

    auto v = std::make_shared<mgcl::Cuboid>(N, N, N, ghin, ghin, ghin);
    auto f = std::make_shared<mgcl::Cuboid>(N, N, N, ghin, ghin, ghin);
    auto solution = mgcl::Cuboid(N, N, N);

    // Set boundaries to 0.
    for (int i = 0; i < ghin; i++)
        for (int j = 0; j < ghin; j++)
            for (int k = 0; k < ghin; k++)
            {
                (*v)[i][j][k] = 0.0;
                (*f)[i][j][k] = 0.0;

                (*v)[i + N][j + N][k + N] = 0.0;
                (*f)[i + N][j + N][k + N] = 0.0;
            }

    for (int i = ghin; i < N + ghin; i++)
        for (int j = ghin; j < N + ghin; j++)
            for (int k = ghin; k < N + ghin; k++)
            {
                double zs = i * h;
                double ys = j * h;
                double xs = k * h;
                double xs2 = xs * xs;
                double ys2 = ys * ys;
                double zs2 = zs * zs;
                double xsm1_2 = (xs - 1) * (xs - 1);
                double ysm1_2 = (ys - 1) * (ys - 1);
                double zsm1_2 = (zs - 1) * (zs - 1);
                double xs3 = xs * xs * xs;
                double ys3 = ys * ys * ys;
                double zs3 = zs * zs * zs;
                double xsm1_3 = (xs - 1) * (xs - 1) * (xs - 1);
                double ysm1_3 = (ys - 1) * (ys - 1) * (ys - 1);
                double zsm1_3 = (zs - 1) * (zs - 1) * (zs - 1);
                double xs4 = xs * xs * xs * xs;
                double ys4 = ys * ys * ys * ys;
                double zs4 = zs * zs * zs * zs;
                double xsm1_4 = (xs - 1) * (xs - 1) * (xs - 1) * (xs - 1);
                double ysm1_4 = (ys - 1) * (ys - 1) * (ys - 1) * (ys - 1);
                double zsm1_4 = (zs - 1) * (zs - 1) * (zs - 1) * (zs - 1);
                (*v)[i][j][k] = 0;
                solution[i - ghin][j - ghin][k - ghin] = 1000000 * (xs * (xs - 1)) * (xs * (xs - 1)) * (xs * (xs - 1)) * (xs * (xs - 1)) *
                                                         (ys * (ys - 1)) * (ys * (ys - 1)) * (ys * (ys - 1)) * (ys * (ys - 1)) *
                                                         (zs * (zs - 1)) * (zs * (zs - 1)) * (zs * (zs - 1)) * (zs * (zs - 1));
                (*f)[i][j][k] =
                    -1000000 *
                    (12 * xs4 * ys4 * zs4 * xsm1_4 * ysm1_4 * zsm1_2 + 12 * xs4 * ys4 * zs4 * xsm1_4 * ysm1_2 * zsm1_4 +
                     12 * xs4 * ys4 * zs4 * xsm1_2 * ysm1_4 * zsm1_4 + 32 * xs4 * ys4 * zs3 * xsm1_4 * ysm1_4 * zsm1_3 +
                     12 * xs4 * ys4 * zs2 * xsm1_4 * ysm1_4 * zsm1_4 + 32 * xs4 * ys3 * zs4 * xsm1_4 * ysm1_3 * zsm1_4 +
                     12 * xs4 * ys2 * zs4 * xsm1_4 * ysm1_4 * zsm1_4 + 32 * xs3 * ys4 * zs4 * xsm1_3 * ysm1_4 * zsm1_4 +
                     12 * xs2 * ys4 * zs4 * xsm1_4 * ysm1_4 * zsm1_4);
            }

    mgcl::Problem p(N, N, N, f, v);
    p.setMaxiterVcycles(maxIterVCycles);
    p.setTol(tol);
    p.setNu1(nu1);
    p.setNu2(nu2);
    p.setOmega(omega);
    p.setMaxlevel(maxlevel);
    p.setBc(bc);
    p.setGhostsIn(ghin);

    SECTION("Sequential")
    {
        SECTION("Laplace_7p")
        {
            p.solveSeq();

            // check if input v is equal to the v stored in Problem instance
            REQUIRE(v.get() == p.getVPtr().get());
            REQUIRE(v->isEqual(p.getV()));

            // check if solution is good
            auto err = mgcl_test::calculateError(solution, *v);
            auto errNorm = mgcl_test::calculateErrorNorm(1.0 / (double)N, *err);
            auto errMax = mgcl_test::calculateMaxError(*err);

            // solution.dumpToFile("out_solution.txt");
            // (*v).dumpToFile("out_v.txt");

            std::cout
                << "seq Laplace 7p" << std::endl
                << std::scientific << std::setprecision(17) << "  ||e||_2 = " << errNorm << std::endl
                << std::scientific << std::setprecision(17) << "  e_max = " << errMax << std::endl;

            CHECK(errNorm < 1e-2);
            CHECK(errMax < 1e-2);
        }

        SECTION("Galerkin_7p")
        {
            p.setStencilType(mgcl::MGCL_VARYING);
            auto& s = *p.getStencilValues();
            double h2inv = N * N; // h = 1/N -> 1/h = N

            double h = 1.0 / static_cast<double>(N);
            mgcl_test::fill7pLaplace(s, h, false);

            p.solveSeq();

            // check if input v is equal to the v stored in Problem instance
            REQUIRE(v.get() == p.getVPtr().get());
            REQUIRE(v->isEqual(p.getV()));

            // check if solution is good
            auto err = mgcl_test::calculateError(solution, *v);
            auto errNorm = mgcl_test::calculateErrorNorm(1.0 / (double)N, *err);
            auto errMax = mgcl_test::calculateMaxError(*err);

            // solution.dumpToFile("out_solution.txt");
            // (*v).dumpToFile("out_v.txt");

            std::cout
                << "seq Galerkin 7p" << std::endl
                << std::scientific << std::setprecision(17) << "  ||e||_2 = " << errNorm << std::endl
                << std::scientific << std::setprecision(17) << "  e_max = " << errMax << std::endl;

            CHECK(errNorm < 1e-2);
            CHECK(errMax < 1e-2);
        }

        SECTION("Laplace_27p")
        {
            p.setStencilType(mgcl::MGCL_LAPLACE_27POINT);
            p.solveSeq();

            // check if input v is equal to the v stored in Problem instance
            REQUIRE(v.get() == p.getVPtr().get());
            REQUIRE(v->isEqual(p.getV()));

            // check if solution is good
            auto err = mgcl_test::calculateError(solution, *v);
            auto errNorm = mgcl_test::calculateErrorNorm(1.0 / (double)N, *err);
            auto errMax = mgcl_test::calculateMaxError(*err);

            // solution.dumpToFile("out_solution.txt");
            // (*v).dumpToFile("out_v.txt");

            std::cout
                << "seq Laplace 27p" << std::endl
                << std::scientific << std::setprecision(17) << "  ||e||_2 = " << errNorm << std::endl
                << std::scientific << std::setprecision(17) << "  e_max = " << errMax << std::endl;

            CHECK(errNorm < 1e-2);
            CHECK(errMax < 1e-2);
        }

        SECTION("Galerkin_27p")
        {
            p.setStencilType(mgcl::MGCL_VARYING);
            auto& s = *p.getStencilValues();
            double h2inv = N * N; // h = 1/N -> 1/h = N

            double h = 1.0 / static_cast<double>(N);
            mgcl_test::fill27pLaplace(s, h, false);

            p.solveSeq();

            // check if input v is equal to the v stored in Problem instance
            REQUIRE(v.get() == p.getVPtr().get());
            REQUIRE(v->isEqual(p.getV()));

            // check if solution is good
            auto err = mgcl_test::calculateError(solution, *v);
            auto errNorm = mgcl_test::calculateErrorNorm(1.0 / (double)N, *err);
            auto errMax = mgcl_test::calculateMaxError(*err);

            // solution.dumpToFile("out_solution.txt");
            // (*v).dumpToFile("out_v.txt");

            std::cout
                << "seq Galerkin 27p" << std::endl
                << std::scientific << std::setprecision(17) << "  ||e||_2 = " << errNorm << std::endl
                << std::scientific << std::setprecision(17) << "  e_max = " << errMax << std::endl;

            CHECK(errNorm < 1e-2);
            CHECK(errMax < 1e-2);
        }
    }

    SECTION("OpenCL")
    {
        auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));

        std::string oclDeviceType = deviceType == CL_DEVICE_TYPE_GPU ? "GPU" : "CPU";

        p.setUseOpencl(true);
        p.setDeviceType(CL_DEVICE_TYPE_GPU);
        p.setReadResults(true);
        p.setDeviceType(deviceType);
        // p.setDeviceName("Quadro");

        SECTION("Laplace_7p")
        {
            p.solve();

            // check if solution is good
            auto err = mgcl_test::calculateError(solution, *v);
            auto errNorm = mgcl_test::calculateErrorNorm(1.0 / (double)N, *err);
            auto errMax = mgcl_test::calculateMaxError(*err);

            std::cout
                << "ocl " << oclDeviceType << " Laplace 7p" << std::endl
                << std::scientific << "  ||e||_2 = " << errNorm << std::endl
                << std::scientific << "  e_max = " << errMax << std::endl;

            CHECK(errNorm < 1e-2);
            CHECK(errMax < 1e-2);
        }

        SECTION("Galerkin_7p")
        {
            p.setStencilType(mgcl::MGCL_VARYING);
            auto& s = *p.getStencilValues();
            double h2inv = N * N; // h = 1/N -> 1/h = N

            double h = 1.0 / static_cast<double>(N);
            mgcl_test::fill7pLaplace(s, h, false);

            p.solve();

            // check if solution is good
            auto err = mgcl_test::calculateError(solution, *v);
            auto errNorm = mgcl_test::calculateErrorNorm(1.0 / (double)N, *err);
            auto errMax = mgcl_test::calculateMaxError(*err);

            // solution.dumpToFile("out_solution.txt");
            // (*v).dumpToFile("out_v.txt");

            std::cout
                << "ocl " << oclDeviceType << " Galerkin 7p" << std::endl
                << std::scientific << std::setprecision(17) << "  ||e||_2 = " << errNorm << std::endl
                << std::scientific << std::setprecision(17) << "  e_max = " << errMax << std::endl;

            CHECK(errNorm < 1e-2);
            CHECK(errMax < 1e-2);
        }

        SECTION("Laplace_27p")
        {
            p.setStencilType(mgcl::MGCL_LAPLACE_27POINT);
            p.solve();

            // check if solution is good
            auto err = mgcl_test::calculateError(solution, *v);
            auto errNorm = mgcl_test::calculateErrorNorm(1.0 / (double)N, *err);
            auto errMax = mgcl_test::calculateMaxError(*err);

            std::cout
                << "ocl " << oclDeviceType << " Laplace 27p" << std::endl
                << std::scientific << "  ||e||_2 = " << errNorm << std::endl
                << std::scientific << "  e_max = " << errMax << std::endl;

            CHECK(errNorm < 1e-2);
            CHECK(errMax < 1e-2);
        }

        SECTION("Galerkin_27p")
        {
            p.setStencilType(mgcl::MGCL_VARYING);
            auto& s = *p.getStencilValues();
            double h2inv = N * N; // h = 1/N -> 1/h = N

            double h = 1.0 / static_cast<double>(N);
            mgcl_test::fill27pLaplace(s, h, false);

            p.solve();

            // check if solution is good
            auto err = mgcl_test::calculateError(solution, *v);
            auto errNorm = mgcl_test::calculateErrorNorm(1.0 / (double)N, *err);
            auto errMax = mgcl_test::calculateMaxError(*err);

            // solution.dumpToFile("out_solution.txt");
            // (*v).dumpToFile("out_v.txt");

            std::cout
                << "ocl " << oclDeviceType << " Galerkin 27p" << std::endl
                << std::scientific << std::setprecision(17) << "  ||e||_2 = " << errNorm << std::endl
                << std::scientific << std::setprecision(17) << "  e_max = " << errMax << std::endl;

            CHECK(errNorm < 1e-2);
            CHECK(errMax < 1e-2);
        }
    }

    // pmg does not work without touching the code, skip for now
    SECTION("pmg")
    {
        // pmg

        // decrement N by 1 for Dirichlet bc's
        int Norig = N;
        N = N - 1;

        // setup MPI
        MPI_Comm mpi_comm_cart = *init_mpi_for_pmg();

        if (mpi_comm_cart == MPI_COMM_NULL)
        {
            std::cout << "mpi_comm_cart is null! Cannot test against pmg." << std::endl;
        }
        else
        {
            int periodic = 0;
            // auto v = std::make_shared<mgcl::Cuboid>(N, N, N);

            // Copy non-ghosted versions of v and f for pmg
            mgcl::Cuboid vpmg(N, N, N);
            mgcl::Cuboid fpmg(N, N, N);

            for (int i = ghin; i < N + ghin; i++)
                for (int j = ghin; j < N + ghin; j++)
                    for (int k = ghin; k < N + ghin; k++)
                    {
                        vpmg[i - ghin][j - ghin][k - ghin] = (*v)[i][j][k];
                        fpmg[i - ghin][j - ghin][k - ghin] = (*f)[i][j][k];
                    }

            mgcl::Cuboid solutionpmg(N, N, N);
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    for (int k = 0; k < N; k++)
                    {
                        solutionpmg[i][j][k] = solution[i][j][k];
                    }

            // init 7-point stencil for pmg's jacobi
            int size = 7;
            double h2 = 1.0 / (double)(Norig * Norig);
            double* values = new double[size]();
            int* xoff = new int[size]();
            int* yoff = new int[size]();
            int* zoff = new int[size]();

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
            mg_with_maxlv(vpmg.getData(), fpmg.getData(), maxIterVCycles, tol, N, N, N, 0, N - 1, 0, N - 1, 0, N - 1,
                          periodic, nu1, nu2, omega, size, values, xoff, yoff, zoff, mpi_comm_cart, 1, maxlevel + 1);

            // vpmg.dumpToFile("vpmg.txt");
            // fpmg.dumpToFile("fpmg.txt");

            // check if solution is good
            auto err = mgcl_test::calculateError(solutionpmg, vpmg);
            auto errNorm = mgcl_test::calculateErrorNorm(1.0 / (double)Norig, *err);
            auto errMax = mgcl_test::calculateMaxError(*err);

            std::cout
                << "pmg" << std::endl
                << std::scientific << "  ||e||_2 = " << errNorm << std::endl
                << std::scientific << "  e_max = " << errMax << std::endl;

            CHECK(errNorm < 1e-2);
            CHECK(errMax < 1e-2);
        }

        // Gets called in custom catch2 main
        // MPI_Finalize();
    }
}

/**
 * @brief Tests if solving works correctly for u = x^4 * (x-1)^4.
 * Tests different amounts of jacobi iterations without ghost update in-between.
 *
 */
TEST_CASE("Problem_solving:_periodic_4th_order_Jacobi_iters")
{
    int N = 16;
    double h = 1.0 / (double)N;

    // Problem parameters
    double tol = 1e-7;
    int nu1 = 2;
    int nu2 = 2;
    double omega = 0.8;
    // int maxIterVCycles = 1;
    int maxIterVCycles = 20;
    int maxlevel = 10;

    int jacobiIters = GENERATE(1, 2, 3);
    int gh = jacobiIters;
    CAPTURE(jacobiIters);
    std::cerr << "Testing with jacobiIters = " << jacobiIters << std::endl;

    auto v = std::make_shared<mgcl::Cuboid>(N, N, N);
    auto f = std::make_shared<mgcl::Cuboid>(N, N, N);
    auto solution = mgcl::Cuboid(N, N, N);

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
            {
                double zs = i * h;
                double ys = j * h;
                double xs = k * h;
                double xs2 = xs * xs;
                double ys2 = ys * ys;
                double zs2 = zs * zs;
                double xsm1_2 = (xs - 1) * (xs - 1);
                double ysm1_2 = (ys - 1) * (ys - 1);
                double zsm1_2 = (zs - 1) * (zs - 1);
                double xs3 = xs * xs * xs;
                double ys3 = ys * ys * ys;
                double zs3 = zs * zs * zs;
                double xsm1_3 = (xs - 1) * (xs - 1) * (xs - 1);
                double ysm1_3 = (ys - 1) * (ys - 1) * (ys - 1);
                double zsm1_3 = (zs - 1) * (zs - 1) * (zs - 1);
                double xs4 = xs * xs * xs * xs;
                double ys4 = ys * ys * ys * ys;
                double zs4 = zs * zs * zs * zs;
                double xsm1_4 = (xs - 1) * (xs - 1) * (xs - 1) * (xs - 1);
                double ysm1_4 = (ys - 1) * (ys - 1) * (ys - 1) * (ys - 1);
                double zsm1_4 = (zs - 1) * (zs - 1) * (zs - 1) * (zs - 1);
                (*v)[i][j][k] = 0;
                solution[i][j][k] = 1000000 * (xs * (xs - 1)) * (xs * (xs - 1)) * (xs * (xs - 1)) * (xs * (xs - 1)) *
                                    (ys * (ys - 1)) * (ys * (ys - 1)) * (ys * (ys - 1)) * (ys * (ys - 1)) *
                                    (zs * (zs - 1)) * (zs * (zs - 1)) * (zs * (zs - 1)) * (zs * (zs - 1));
                (*f)[i][j][k] =
                    -1000000 *
                    (12 * xs4 * ys4 * zs4 * xsm1_4 * ysm1_4 * zsm1_2 + 12 * xs4 * ys4 * zs4 * xsm1_4 * ysm1_2 * zsm1_4 +
                     12 * xs4 * ys4 * zs4 * xsm1_2 * ysm1_4 * zsm1_4 + 32 * xs4 * ys4 * zs3 * xsm1_4 * ysm1_4 * zsm1_3 +
                     12 * xs4 * ys4 * zs2 * xsm1_4 * ysm1_4 * zsm1_4 + 32 * xs4 * ys3 * zs4 * xsm1_4 * ysm1_3 * zsm1_4 +
                     12 * xs4 * ys2 * zs4 * xsm1_4 * ysm1_4 * zsm1_4 + 32 * xs3 * ys4 * zs4 * xsm1_3 * ysm1_4 * zsm1_4 +
                     12 * xs2 * ys4 * zs4 * xsm1_4 * ysm1_4 * zsm1_4);
            }

    mgcl::Problem p(N, N, N, f, v);
    p.setMaxiterVcycles(maxIterVCycles);
    p.setTol(tol);
    p.setNu1(nu1);
    p.setNu2(nu2);
    p.setOmega(omega);
    p.setMaxlevel(maxlevel);
    p.setGhosts(gh);
    p.setJacobiIterationsPerKernel(jacobiIters);

    SECTION("Sequential")
    {
        SECTION("Laplace")
        {
            p.solveSeq();

            // check if input v is equal to the v stored in Problem instance
            REQUIRE(v.get() == p.getVPtr().get());
            REQUIRE(v->isEqual(p.getV()));

            // check if solution is good
            auto err = mgcl_test::calculateError(solution, *v);
            auto errNorm = mgcl_test::calculateErrorNorm(1.0 / (double)N, *err);
            auto errMax = mgcl_test::calculateMaxError(*err);

            // solution.dumpToFile("out_solution.txt");
            // (*v).dumpToFile("out_v.txt");

            std::cout
                << "seq Laplace" << std::endl
                << std::scientific << std::setprecision(17) << "  ||e||_2 = " << errNorm << std::endl
                << std::scientific << std::setprecision(17) << "  e_max = " << errMax << std::endl;

            CHECK(errNorm < 1e-2);
            CHECK(errMax < 1e-2);

            // check if error is equal to old mgcl implementation (problem params must match)
            if (p.getMaxiterVcycles() == 10 && N == 32 && p.getTol() == 1e-14 &&
                p.getNu1() == 2 && p.getNu2() == 2 && p.getOmega() == 0.8 &&
                p.getStencilType() == mgcl::MGCL_LAPLACE_7POINT)
            {
                CHECK(fabs(errNorm - 3.93115528889639940e-03) < 1e-14);
                CHECK(fabs(errMax - 3.95723982871564600e-03) < 1e-14);
            }
        }

        SECTION("Galerkin")
        {
            p.setStencilType(mgcl::MGCL_VARYING);
            auto& s = *p.getStencilValues();
            double h2inv = N * N; // h = 1/N -> 1/h = N

            double h = 1.0 / static_cast<double>(N);
            mgcl_test::fill7pLaplace(s, h, false);

            p.solveSeq();

            // check if input v is equal to the v stored in Problem instance
            REQUIRE(v.get() == p.getVPtr().get());
            REQUIRE(v->isEqual(p.getV()));

            // check if solution is good
            auto err = mgcl_test::calculateError(solution, *v);
            auto errNorm = mgcl_test::calculateErrorNorm(1.0 / (double)N, *err);
            auto errMax = mgcl_test::calculateMaxError(*err);

            // solution.dumpToFile("out_solution.txt");
            // (*v).dumpToFile("out_v.txt");

            std::cout
                << "seq Galerkin" << std::endl
                << std::scientific << std::setprecision(17) << "  ||e||_2 = " << errNorm << std::endl
                << std::scientific << std::setprecision(17) << "  e_max = " << errMax << std::endl;

            CHECK(errNorm < 1e-2);
            CHECK(errMax < 1e-2);
        }
    }

    SECTION("OpenCL")
    {
        auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));

        std::string oclDeviceType = deviceType == CL_DEVICE_TYPE_GPU ? "GPU" : "CPU";

        p.setUseOpencl(true);
        p.setDeviceType(CL_DEVICE_TYPE_GPU);
        p.setReadResults(true);
        p.setDeviceType(deviceType);
        // p.setDeviceName("Quadro");

        SECTION("Laplace")
        {
            p.solve();

            // check if solution is good
            auto err = mgcl_test::calculateError(solution, *v);
            auto errNorm = mgcl_test::calculateErrorNorm(1.0 / (double)N, *err);
            auto errMax = mgcl_test::calculateMaxError(*err);

            std::cout
                << "ocl " << oclDeviceType << " Laplace" << std::endl
                << std::scientific << "  ||e||_2 = " << errNorm << std::endl
                << std::scientific << "  e_max = " << errMax << std::endl;

            CHECK(errNorm < 1e-2);
            CHECK(errMax < 1e-2);

            // check if error is equal to old mgcl implementation (problem params must match)
            if (p.getMaxiterVcycles() == 10 && N == 32 && p.getTol() == 1e-14 &&
                p.getNu1() == 2 && p.getNu2() == 2 && p.getOmega() == 0.8 &&
                p.getDeviceName() == "Quadro" && p.getDeviceType() == CL_DEVICE_TYPE_GPU)
            {
                CHECK(fabs(errNorm - 3.93115528889612358e-03) < 1e-14);
                CHECK(fabs(errMax - 3.95723982871536324e-03) < 1e-14);
            }
        }

        SECTION("Galerkin")
        {
            p.setStencilType(mgcl::MGCL_VARYING);
            auto& s = *p.getStencilValues();
            double h2inv = N * N; // h = 1/N -> 1/h = N

            double h = 1.0 / static_cast<double>(N);
            mgcl_test::fill7pLaplace(s, h, false);

            p.solve();

            // check if solution is good
            auto err = mgcl_test::calculateError(solution, *v);
            auto errNorm = mgcl_test::calculateErrorNorm(1.0 / (double)N, *err);
            auto errMax = mgcl_test::calculateMaxError(*err);

            // solution.dumpToFile("out_solution.txt");
            // (*v).dumpToFile("out_v.txt");

            std::cout
                << "ocl " << oclDeviceType << " Galerkin" << std::endl
                << std::scientific << std::setprecision(17) << "  ||e||_2 = " << errNorm << std::endl
                << std::scientific << std::setprecision(17) << "  e_max = " << errMax << std::endl;

            CHECK(errNorm < 1e-2);
            CHECK(errMax < 1e-2);
        }
    }
}

/**
 * @brief Tests if tolerance is ignored if ignoreTol is true, i.e. maximum amount of v-cycle iterations
 * is done although the tolerance is low.
 *
 */
TEST_CASE("Problem_ignore_tol")
{
    int N = 16;
    double h = 1.0 / (double)N;

    // Problem parameters
    double tol = 1e-1; // will be reached really quick
    int nu1 = 2;
    int nu2 = 2;
    double omega = 0.8;
    int maxIterVCycles = 10;
    int maxlevel = 10;

    auto v = std::make_shared<mgcl::Cuboid>(N, N, N);
    auto f = std::make_shared<mgcl::Cuboid>(N, N, N);
    auto solution = mgcl::Cuboid(N, N, N);

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
            {
                double zs = i * h;
                double ys = j * h;
                double xs = k * h;
                double xs2 = xs * xs;
                double ys2 = ys * ys;
                double zs2 = zs * zs;
                double xsm1_2 = (xs - 1) * (xs - 1);
                double ysm1_2 = (ys - 1) * (ys - 1);
                double zsm1_2 = (zs - 1) * (zs - 1);
                double xs3 = xs * xs * xs;
                double ys3 = ys * ys * ys;
                double zs3 = zs * zs * zs;
                double xsm1_3 = (xs - 1) * (xs - 1) * (xs - 1);
                double ysm1_3 = (ys - 1) * (ys - 1) * (ys - 1);
                double zsm1_3 = (zs - 1) * (zs - 1) * (zs - 1);
                double xs4 = xs * xs * xs * xs;
                double ys4 = ys * ys * ys * ys;
                double zs4 = zs * zs * zs * zs;
                double xsm1_4 = (xs - 1) * (xs - 1) * (xs - 1) * (xs - 1);
                double ysm1_4 = (ys - 1) * (ys - 1) * (ys - 1) * (ys - 1);
                double zsm1_4 = (zs - 1) * (zs - 1) * (zs - 1) * (zs - 1);
                (*v)[i][j][k] = 0;
                solution[i][j][k] = 1000000 * (xs * (xs - 1)) * (xs * (xs - 1)) * (xs * (xs - 1)) * (xs * (xs - 1)) *
                                    (ys * (ys - 1)) * (ys * (ys - 1)) * (ys * (ys - 1)) * (ys * (ys - 1)) *
                                    (zs * (zs - 1)) * (zs * (zs - 1)) * (zs * (zs - 1)) * (zs * (zs - 1));
                (*f)[i][j][k] =
                    -1000000 *
                    (12 * xs4 * ys4 * zs4 * xsm1_4 * ysm1_4 * zsm1_2 + 12 * xs4 * ys4 * zs4 * xsm1_4 * ysm1_2 * zsm1_4 +
                     12 * xs4 * ys4 * zs4 * xsm1_2 * ysm1_4 * zsm1_4 + 32 * xs4 * ys4 * zs3 * xsm1_4 * ysm1_4 * zsm1_3 +
                     12 * xs4 * ys4 * zs2 * xsm1_4 * ysm1_4 * zsm1_4 + 32 * xs4 * ys3 * zs4 * xsm1_4 * ysm1_3 * zsm1_4 +
                     12 * xs4 * ys2 * zs4 * xsm1_4 * ysm1_4 * zsm1_4 + 32 * xs3 * ys4 * zs4 * xsm1_3 * ysm1_4 * zsm1_4 +
                     12 * xs2 * ys4 * zs4 * xsm1_4 * ysm1_4 * zsm1_4);
            }

    mgcl::Problem p(N, N, N, f, v);
    p.setMaxiterVcycles(maxIterVCycles);
    p.setTol(tol);
    p.setNu1(nu1);
    p.setNu2(nu2);
    p.setOmega(omega);
    p.setMaxlevel(maxlevel);

    SECTION("Sequential")
    {
        SECTION("ignoreTolFalse")
        {
            p.setIgnoreTol(false);
            p.solveSeq();

            REQUIRE(p.getElapsedIterations() < maxIterVCycles);
        }

        SECTION("ignoreTolTrue")
        {
            p.setIgnoreTol(true);
            p.solveSeq();

            REQUIRE(p.getElapsedIterations() == maxIterVCycles);
        }
    }

    SECTION("OpenCL")
    {
        auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));

        std::string oclDeviceType = deviceType == CL_DEVICE_TYPE_GPU ? "GPU" : "CPU";

        p.setUseOpencl(true);
        p.setDeviceType(CL_DEVICE_TYPE_GPU);
        p.setDeviceType(deviceType);

        SECTION("ignoreTolFalse")
        {
            p.setIgnoreTol(false);
            p.solve();

            REQUIRE(p.getElapsedIterations() < maxIterVCycles);
        }

        SECTION("ignoreTolTrue")
        {
            p.setIgnoreTol(true);
            p.solve();

            REQUIRE(p.getElapsedIterations() == maxIterVCycles);
        }
    }
}

/**
 * @brief Checks whether using FixedStencil yields the same solution as using VaryingStencil with values of FixedStencil
 *
 */
TEST_CASE("solve_fixed_vs_varying_stencil")
{
    int N = 16;
    double h = 1.0 / (double)N;

    // Problem parameters
    double tol = 1e-1; // will be reached really quick
    int nu1 = 2;
    int nu2 = 2;
    double omega = 0.8;
    int maxIterVCycles = 10;
    int maxlevel = 10;

    auto v_fixed = std::make_shared<mgcl::Cuboid>(N, N, N);
    auto v_varying = std::make_shared<mgcl::Cuboid>(N, N, N);
    auto f = std::make_shared<mgcl::Cuboid>(N, N, N);
    auto solution = mgcl::Cuboid(N, N, N);

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
            {
                double zs = i * h;
                double ys = j * h;
                double xs = k * h;
                double xs2 = xs * xs;
                double ys2 = ys * ys;
                double zs2 = zs * zs;
                double xsm1_2 = (xs - 1) * (xs - 1);
                double ysm1_2 = (ys - 1) * (ys - 1);
                double zsm1_2 = (zs - 1) * (zs - 1);
                double xs3 = xs * xs * xs;
                double ys3 = ys * ys * ys;
                double zs3 = zs * zs * zs;
                double xsm1_3 = (xs - 1) * (xs - 1) * (xs - 1);
                double ysm1_3 = (ys - 1) * (ys - 1) * (ys - 1);
                double zsm1_3 = (zs - 1) * (zs - 1) * (zs - 1);
                double xs4 = xs * xs * xs * xs;
                double ys4 = ys * ys * ys * ys;
                double zs4 = zs * zs * zs * zs;
                double xsm1_4 = (xs - 1) * (xs - 1) * (xs - 1) * (xs - 1);
                double ysm1_4 = (ys - 1) * (ys - 1) * (ys - 1) * (ys - 1);
                double zsm1_4 = (zs - 1) * (zs - 1) * (zs - 1) * (zs - 1);
                (*v_fixed)[i][j][k] = 0;
                (*v_varying)[i][j][k] = 0;
                solution[i][j][k] = 1000000 * (xs * (xs - 1)) * (xs * (xs - 1)) * (xs * (xs - 1)) * (xs * (xs - 1)) *
                                    (ys * (ys - 1)) * (ys * (ys - 1)) * (ys * (ys - 1)) * (ys * (ys - 1)) *
                                    (zs * (zs - 1)) * (zs * (zs - 1)) * (zs * (zs - 1)) * (zs * (zs - 1));
                (*f)[i][j][k] =
                    -1000000 *
                    (12 * xs4 * ys4 * zs4 * xsm1_4 * ysm1_4 * zsm1_2 + 12 * xs4 * ys4 * zs4 * xsm1_4 * ysm1_2 * zsm1_4 +
                     12 * xs4 * ys4 * zs4 * xsm1_2 * ysm1_4 * zsm1_4 + 32 * xs4 * ys4 * zs3 * xsm1_4 * ysm1_4 * zsm1_3 +
                     12 * xs4 * ys4 * zs2 * xsm1_4 * ysm1_4 * zsm1_4 + 32 * xs4 * ys3 * zs4 * xsm1_4 * ysm1_3 * zsm1_4 +
                     12 * xs4 * ys2 * zs4 * xsm1_4 * ysm1_4 * zsm1_4 + 32 * xs3 * ys4 * zs4 * xsm1_3 * ysm1_4 * zsm1_4 +
                     12 * xs2 * ys4 * zs4 * xsm1_4 * ysm1_4 * zsm1_4);
            }

    mgcl::Problem p_fixed(N, N, N, f, v_fixed);
    p_fixed.setMaxiterVcycles(maxIterVCycles);
    p_fixed.setTol(tol);
    p_fixed.setNu1(nu1);
    p_fixed.setNu2(nu2);
    p_fixed.setOmega(omega);
    p_fixed.setMaxlevel(maxlevel);

    p_fixed.setStencilType(mgcl::MGCL_FIXED);
    auto& fixedStencil = p_fixed.getFixedStencil();
    fixedStencil->fillRandom();
    (*fixedStencil)[1][1][1] = 1.0; // make sure the stencil does not produce nan

    mgcl::Problem p_varying(N, N, N, f, v_varying);
    p_varying.setMaxiterVcycles(maxIterVCycles);
    p_varying.setTol(tol);
    p_varying.setNu1(nu1);
    p_varying.setNu2(nu2);
    p_varying.setOmega(omega);
    p_varying.setMaxlevel(maxlevel);

    p_varying.setStencilType(mgcl::MGCL_VARYING);
    auto& sv = p_varying.getStencilValues();
    // copy from fixed into varying
    // clang-format off
    for (int i = 0; i < sv->getMgh(); i++)
    for (int j = 0; j < sv->getNgh(); j++)
    for (int k = 0; k < sv->getOgh(); k++)
        for (int ii = 0; ii < 3; ii++)
        for (int jj = 0; jj < 3; jj++)
        for (int kk = 0; kk < 3; kk++)
        {
            (*sv)[ii][jj][kk][i][j][k] = (*fixedStencil)[ii][jj][kk];
        }
    // clang-format on

    SECTION("Sequential")
    {
        p_fixed.solve();
        p_varying.solve();

        REQUIRE(v_fixed->isEqual(*v_varying));
    }

    SECTION("OpenCL")
    {
        auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));

        std::string oclDeviceType = deviceType == CL_DEVICE_TYPE_GPU ? "GPU" : "CPU";

        p_fixed.setUseOpencl(true);
        p_fixed.setDeviceType(CL_DEVICE_TYPE_GPU);
        p_fixed.setDeviceType(deviceType);
        p_varying.setUseOpencl(true);
        p_varying.setDeviceType(CL_DEVICE_TYPE_GPU);
        p_varying.setDeviceType(deviceType);

        p_fixed.solve();
        p_varying.solve();

        REQUIRE(v_fixed->isEqual(*v_varying));
    }
}

/**
 * @brief Checks whether vcycle works with levels partly calculated with OpenCL and partly calculated on host
 *
 */
TEST_CASE("solve_maxLevelUsingOcl")
{
    int N = 16;
    double h = 1.0 / (double)N;

    // Problem parameters
    double tol = 1e-1; // will be reached really quick
    int nu1 = 2;
    int nu2 = 2;
    double omega = 0.8;
    int maxIterVCycles = 10;
    int maxlevel = 10;
    int maxLevelUsingOcl = GENERATE(0, 1, 2);

    CAPTURE(maxLevelUsingOcl);

    auto v_fixed = std::make_shared<mgcl::Cuboid>(N, N, N);
    auto v_varying = std::make_shared<mgcl::Cuboid>(N, N, N);
    auto f = std::make_shared<mgcl::Cuboid>(N, N, N);
    auto solution = mgcl::Cuboid(N, N, N);

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
            {
                double zs = i * h;
                double ys = j * h;
                double xs = k * h;
                double xs2 = xs * xs;
                double ys2 = ys * ys;
                double zs2 = zs * zs;
                double xsm1_2 = (xs - 1) * (xs - 1);
                double ysm1_2 = (ys - 1) * (ys - 1);
                double zsm1_2 = (zs - 1) * (zs - 1);
                double xs3 = xs * xs * xs;
                double ys3 = ys * ys * ys;
                double zs3 = zs * zs * zs;
                double xsm1_3 = (xs - 1) * (xs - 1) * (xs - 1);
                double ysm1_3 = (ys - 1) * (ys - 1) * (ys - 1);
                double zsm1_3 = (zs - 1) * (zs - 1) * (zs - 1);
                double xs4 = xs * xs * xs * xs;
                double ys4 = ys * ys * ys * ys;
                double zs4 = zs * zs * zs * zs;
                double xsm1_4 = (xs - 1) * (xs - 1) * (xs - 1) * (xs - 1);
                double ysm1_4 = (ys - 1) * (ys - 1) * (ys - 1) * (ys - 1);
                double zsm1_4 = (zs - 1) * (zs - 1) * (zs - 1) * (zs - 1);
                (*v_fixed)[i][j][k] = 0;
                (*v_varying)[i][j][k] = 0;
                solution[i][j][k] = 1000000 * (xs * (xs - 1)) * (xs * (xs - 1)) * (xs * (xs - 1)) * (xs * (xs - 1)) *
                                    (ys * (ys - 1)) * (ys * (ys - 1)) * (ys * (ys - 1)) * (ys * (ys - 1)) *
                                    (zs * (zs - 1)) * (zs * (zs - 1)) * (zs * (zs - 1)) * (zs * (zs - 1));
                (*f)[i][j][k] =
                    -1000000 *
                    (12 * xs4 * ys4 * zs4 * xsm1_4 * ysm1_4 * zsm1_2 + 12 * xs4 * ys4 * zs4 * xsm1_4 * ysm1_2 * zsm1_4 +
                     12 * xs4 * ys4 * zs4 * xsm1_2 * ysm1_4 * zsm1_4 + 32 * xs4 * ys4 * zs3 * xsm1_4 * ysm1_4 * zsm1_3 +
                     12 * xs4 * ys4 * zs2 * xsm1_4 * ysm1_4 * zsm1_4 + 32 * xs4 * ys3 * zs4 * xsm1_4 * ysm1_3 * zsm1_4 +
                     12 * xs4 * ys2 * zs4 * xsm1_4 * ysm1_4 * zsm1_4 + 32 * xs3 * ys4 * zs4 * xsm1_3 * ysm1_4 * zsm1_4 +
                     12 * xs2 * ys4 * zs4 * xsm1_4 * ysm1_4 * zsm1_4);
            }

    mgcl::Problem pAllLevelsUsingOcl(N, N, N, f, v_fixed);
    pAllLevelsUsingOcl.setMaxiterVcycles(maxIterVCycles);
    pAllLevelsUsingOcl.setTol(tol);
    pAllLevelsUsingOcl.setNu1(nu1);
    pAllLevelsUsingOcl.setNu2(nu2);
    pAllLevelsUsingOcl.setOmega(omega);
    pAllLevelsUsingOcl.setMaxlevel(maxlevel);

    pAllLevelsUsingOcl.setStencilType(mgcl::MGCL_FIXED);
    auto& fixedStencil = pAllLevelsUsingOcl.getFixedStencil();
    fixedStencil->fillRandom();
    (*fixedStencil)[1][1][1] = 1.0; // make sure the stencil does not produce nan

    mgcl::Problem pUsingOclLevelThreshold(N, N, N, f, v_varying);
    pUsingOclLevelThreshold.setMaxiterVcycles(maxIterVCycles);
    pUsingOclLevelThreshold.setTol(tol);
    pUsingOclLevelThreshold.setNu1(nu1);
    pUsingOclLevelThreshold.setNu2(nu2);
    pUsingOclLevelThreshold.setOmega(omega);
    pUsingOclLevelThreshold.setMaxlevel(maxlevel);
    pUsingOclLevelThreshold.setMaxLevelUsingOcl(maxLevelUsingOcl);

    pUsingOclLevelThreshold.setStencilType(mgcl::MGCL_VARYING);
    auto& sv = pUsingOclLevelThreshold.getStencilValues();
    // copy from fixed into varying
    // clang-format off
    for (int i = 0; i < sv->getMgh(); i++)
    for (int j = 0; j < sv->getNgh(); j++)
    for (int k = 0; k < sv->getOgh(); k++)
        for (int ii = 0; ii < 3; ii++)
        for (int jj = 0; jj < 3; jj++)
        for (int kk = 0; kk < 3; kk++)
        {
            (*sv)[ii][jj][kk][i][j][k] = (*fixedStencil)[ii][jj][kk];
        }
    // clang-format on

    auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));

    // std::string oclDeviceType = deviceType == CL_DEVICE_TYPE_GPU ? "GPU" : "CPU";

    pAllLevelsUsingOcl.setUseOpencl(true);
    pAllLevelsUsingOcl.setDeviceType(CL_DEVICE_TYPE_GPU);
    pAllLevelsUsingOcl.setDeviceType(deviceType);
    pUsingOclLevelThreshold.setUseOpencl(true);
    pUsingOclLevelThreshold.setDeviceType(CL_DEVICE_TYPE_GPU);
    pUsingOclLevelThreshold.setDeviceType(deviceType);

    pAllLevelsUsingOcl.solve();
    pUsingOclLevelThreshold.solve();

    REQUIRE(v_fixed->isEqual(*v_varying));
}