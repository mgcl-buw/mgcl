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
#include "test_utility.hpp"

std::shared_ptr<mgcl::Cuboid> calculateError(mgcl::Cuboid& solution, mgcl::Cuboid& approximation);
double calculateMaxError(mgcl::Cuboid& error);
double calculateErrorNorm(double h, mgcl::Cuboid& error);

/**
 * @brief Tests if solving works correctly for u = x^4 * (x-1)^4.
 *
 */
TEST_CASE("Problem solving: periodic 4th order", "[periodic]")
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
        SECTION("Laplace")
        {
            p.solveSeq();

            // check if input v is equal to the v stored in Problem instance
            REQUIRE(v.get() == p.getVPtr().get());
            REQUIRE(v->isEqual(p.getV()));

            // check if solution is good
            auto err = calculateError(solution, *v);
            auto errNorm = calculateErrorNorm(1.0 / (double)N, *err);
            auto errMax = calculateMaxError(*err);

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

        SECTION("Galerkin (varying stencil)")
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
            auto err = calculateError(solution, *v);
            auto errNorm = calculateErrorNorm(1.0 / (double)N, *err);
            auto errMax = calculateMaxError(*err);

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
        auto deviceType = GENERATE(CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU);

        if (!mgcl_test::TestUtility::deviceAvailable("", deviceType))
        {
            std::string typeName = deviceType == CL_DEVICE_TYPE_GPU ? "CL_DEVICE_TYPE_GPU" : "CL_DEVICE_TYPE_CPU";
            std::cout << "Skipping non-available device type '" << typeName << "'" << std::endl;
            return;
        }

        std::string oclDeviceType = deviceType == CL_DEVICE_TYPE_GPU ? "GPU" : "CPU";

        p.setUseOpencl(true);
        p.setReadResults(true);
        p.setDeviceType(deviceType);
        // p.setDeviceName("Quadro");

        SECTION("Laplace")
        {
            p.solve();

            // check if solution is good
            auto err = calculateError(solution, *v);
            auto errNorm = calculateErrorNorm(1.0 / (double)N, *err);
            auto errMax = calculateMaxError(*err);

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

        SECTION("Galerkin (varying stencil)")
        {
            p.setStencilType(mgcl::MGCL_VARYING);
            auto& s = *p.getStencilValues();
            double h2inv = N * N; // h = 1/N -> 1/h = N

            double h = 1.0 / static_cast<double>(N);
            mgcl_test::fill7pLaplace(s, h, false);

            p.solve();

            // check if solution is good
            auto err = calculateError(solution, *v);
            auto errNorm = calculateErrorNorm(1.0 / (double)N, *err);
            auto errMax = calculateMaxError(*err);

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
                          1, nu1, nu2, omega, size, values, xoff, yoff, zoff, mpi_comm_cart, 1, maxlevel + 1);

            // check if solution is good
            auto err = calculateError(solution, *v);
            auto errNorm = calculateErrorNorm(1.0 / (double)N, *err);
            auto errMax = calculateMaxError(*err);

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

TEST_CASE("Problem solving: Dirichlet 4th order", "[dirichlet]")
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
        SECTION("Laplace")
        {
            p.solveSeq();

            // check if input v is equal to the v stored in Problem instance
            REQUIRE(v.get() == p.getVPtr().get());
            REQUIRE(v->isEqual(p.getV()));

            // check if solution is good
            auto err = calculateError(solution, *v);
            auto errNorm = calculateErrorNorm(1.0 / (double)N, *err);
            auto errMax = calculateMaxError(*err);

            // solution.dumpToFile("out_solution.txt");
            // (*v).dumpToFile("out_v.txt");

            std::cout
                << "seq Laplace" << std::endl
                << std::scientific << std::setprecision(17) << "  ||e||_2 = " << errNorm << std::endl
                << std::scientific << std::setprecision(17) << "  e_max = " << errMax << std::endl;

            CHECK(errNorm < 1e-2);
            CHECK(errMax < 1e-2);
        }

        SECTION("Galerkin (varying stencil)")
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
            auto err = calculateError(solution, *v);
            auto errNorm = calculateErrorNorm(1.0 / (double)N, *err);
            auto errMax = calculateMaxError(*err);

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
        auto deviceType = GENERATE(CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU);

        if (!mgcl_test::TestUtility::deviceAvailable("", deviceType))
        {
            std::string typeName = deviceType == CL_DEVICE_TYPE_GPU ? "CL_DEVICE_TYPE_GPU" : "CL_DEVICE_TYPE_CPU";
            std::cout << "Skipping non-available device type '" << typeName << "'" << std::endl;
            return;
        }

        std::string oclDeviceType = deviceType == CL_DEVICE_TYPE_GPU ? "GPU" : "CPU";

        p.setUseOpencl(true);
        p.setReadResults(true);
        p.setDeviceType(deviceType);
        // p.setDeviceName("Quadro");

        SECTION("Laplace")
        {
            p.solve();

            // check if solution is good
            auto err = calculateError(solution, *v);
            auto errNorm = calculateErrorNorm(1.0 / (double)N, *err);
            auto errMax = calculateMaxError(*err);

            std::cout
                << "ocl " << oclDeviceType << " Laplace" << std::endl
                << std::scientific << "  ||e||_2 = " << errNorm << std::endl
                << std::scientific << "  e_max = " << errMax << std::endl;

            CHECK(errNorm < 1e-2);
            CHECK(errMax < 1e-2);
        }

        SECTION("Galerkin (varying stencil)")
        {
            p.setStencilType(mgcl::MGCL_VARYING);
            auto& s = *p.getStencilValues();
            double h2inv = N * N; // h = 1/N -> 1/h = N

            double h = 1.0 / static_cast<double>(N);
            mgcl_test::fill7pLaplace(s, h, false);

            p.solve();

            // check if solution is good
            auto err = calculateError(solution, *v);
            auto errNorm = calculateErrorNorm(1.0 / (double)N, *err);
            auto errMax = calculateMaxError(*err);

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
            auto err = calculateError(solutionpmg, vpmg);
            auto errNorm = calculateErrorNorm(1.0 / (double)Norig, *err);
            auto errMax = calculateMaxError(*err);

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
            auto err = calculateError(solution, *v);
            auto errNorm = calculateErrorNorm(1.0 / (double)N, *err);
            auto errMax = calculateMaxError(*err);

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
            auto err = calculateError(solution, *v);
            auto errNorm = calculateErrorNorm(1.0 / (double)N, *err);
            auto errMax = calculateMaxError(*err);

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
        auto deviceType = GENERATE(CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU);

        if (!mgcl_test::TestUtility::deviceAvailable("", deviceType))
        {
            std::string typeName = deviceType == CL_DEVICE_TYPE_GPU ? "CL_DEVICE_TYPE_GPU" : "CL_DEVICE_TYPE_CPU";
            std::cout << "Skipping non-available device type '" << typeName << "'" << std::endl;
            return;
        }

        std::string oclDeviceType = deviceType == CL_DEVICE_TYPE_GPU ? "GPU" : "CPU";

        p.setUseOpencl(true);
        p.setReadResults(true);
        p.setDeviceType(deviceType);
        // p.setDeviceName("Quadro");

        SECTION("Laplace")
        {
            p.solve();

            // check if solution is good
            auto err = calculateError(solution, *v);
            auto errNorm = calculateErrorNorm(1.0 / (double)N, *err);
            auto errMax = calculateMaxError(*err);

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
            auto err = calculateError(solution, *v);
            auto errNorm = calculateErrorNorm(1.0 / (double)N, *err);
            auto errMax = calculateMaxError(*err);

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
        auto deviceType = GENERATE(CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU);

        if (!mgcl_test::TestUtility::deviceAvailable("", deviceType))
        {
            std::string typeName = deviceType == CL_DEVICE_TYPE_GPU ? "CL_DEVICE_TYPE_GPU" : "CL_DEVICE_TYPE_CPU";
            std::cout << "Skipping non-available device type '" << typeName << "'" << std::endl;
            return;
        }

        std::string oclDeviceType = deviceType == CL_DEVICE_TYPE_GPU ? "GPU" : "CPU";

        p.setUseOpencl(true);
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
 * @brief Calculates error for each cell, e.g. difference between solution and approximation. Dimensions must match.
 *
 * @param solution
 * @param approximation
 * @return std::shared_ptr<mgcl::Cuboid>
 */
std::shared_ptr<mgcl::Cuboid> calculateError(mgcl::Cuboid& solution, mgcl::Cuboid& approximation)
{
    if (solution.getM() != approximation.getM() ||
        solution.getN() != approximation.getN() ||
        solution.getO() != approximation.getO())
        throw std::invalid_argument("Dimensions do not match.");

    auto ret = std::make_shared<mgcl::Cuboid>(solution.getM(), solution.getN(), solution.getO());
    for (int i = 0, is = solution.getGhostsM(), ia = approximation.getGhostsM(); is < solution.getMgh(); i++, is++, ia++)
        for (int j = 0, js = solution.getGhostsN(), ja = approximation.getGhostsN(); js < solution.getNgh(); j++, js++, ja++)
            for (int k = 0, ks = solution.getGhostsO(), ka = approximation.getGhostsO(); ks < solution.getOgh(); k++, ks++, ka++)
            {
                (*ret)[i][j][k] = fabs(solution[is][js][ks] - approximation[ia][ja][ka]);
            }

    return ret;
}

/**
 * @brief Returns the maximum absolute error. calculateError should have been called first.
 *
 * @param error
 * @return double
 */
double calculateMaxError(mgcl::Cuboid& error)
{
    double max = 0;
    for (int i = 0; i < error.getM(); i++)
        for (int j = 0; j < error.getN(); j++)
            for (int k = 0; k < error.getO(); k++)
            {
                if (max < error[i][j][k])
                    max = error[i][j][k];
            }
    return max;
}

/**
 * @brief Returns the 2-norm of the given error. calculateError should have been called first.
 *
 * @param h width of one cell
 * @param error precalculated error per cell
 * @return double Error norm of form ||e||_2 * h^3
 */
double calculateErrorNorm(double h, mgcl::Cuboid& error)
{
    double sum = 0;

    for (int i = 0; i < error.getM(); i++)
        for (int j = 0; j < error.getN(); j++)
            for (int k = 0; k < error.getO(); k++)
            {
                sum += error[i][j][k] * error[i][j][k];
            }

    return sqrt(sum * h * h * h);
}
