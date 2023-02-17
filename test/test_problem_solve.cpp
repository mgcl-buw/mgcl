#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>

#include "../benchmarks/pmg_utility.hpp"
#include "../src/cuboid.hpp"
#include "../src/problem.hpp"
#include "../thirdparty/pmg/mg.h"
#include "test_utility.hpp"

std::shared_ptr<mgcl::Cuboid> calculateError(mgcl::Cuboid &solution, mgcl::Cuboid &approximation);
double calculateMaxError(mgcl::Cuboid &error);
double calculateErrorNorm(double h, mgcl::Cuboid &error);

/**
 * @brief Tests if solving works correctly.
 *
 */
TEST_CASE("Problem solving: periodic 4th order")
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
            auto &s = *p.getStencilValues();
            double h2inv = N * N; // h = 1/N -> 1/h = N

            // Fill with 7-point Laplace, which is also used by the other two Sections in this test case
            for (int i = 0; i < s.getDim1gh(); i++)
                for (int j = 0; j < s.getDim2gh(); j++)
                    for (int k = 0; k < s.getDim3gh(); k++)
                    {
                        // 7-point Laplace
                        s[i][j][k][0][1][1] = h2inv * -1.0;
                        s[i][j][k][1][0][1] = h2inv * -1.0;
                        s[i][j][k][1][1][0] = h2inv * -1.0;
                        s[i][j][k][1][1][1] = h2inv * 6.0;
                        s[i][j][k][1][1][2] = h2inv * -1.0;
                        s[i][j][k][1][2][1] = h2inv * -1.0;
                        s[i][j][k][2][1][1] = h2inv * -1.0;
                    }

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
                << "ocl Laplace" << std::endl
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
            auto &s = *p.getStencilValues();
            double h2inv = N * N; // h = 1/N -> 1/h = N

            // Fill with 7-point Laplace, which is also used by the other two Sections in this test case
            for (int i = 0; i < s.getDim1gh(); i++)
                for (int j = 0; j < s.getDim2gh(); j++)
                    for (int k = 0; k < s.getDim3gh(); k++)
                    {
                        // 7-point Laplace
                        s[i][j][k][0][1][1] = h2inv * -1.0;
                        s[i][j][k][1][0][1] = h2inv * -1.0;
                        s[i][j][k][1][1][0] = h2inv * -1.0;
                        s[i][j][k][1][1][1] = h2inv * 6.0;
                        s[i][j][k][1][1][2] = h2inv * -1.0;
                        s[i][j][k][1][2][1] = h2inv * -1.0;
                        s[i][j][k][2][1][1] = h2inv * -1.0;
                    }

            p.solve();

            // check if solution is good
            auto err = calculateError(solution, *v);
            auto errNorm = calculateErrorNorm(1.0 / (double)N, *err);
            auto errMax = calculateMaxError(*err);

            // solution.dumpToFile("out_solution.txt");
            // (*v).dumpToFile("out_v.txt");

            std::cout
                << "ocl Galerkin" << std::endl
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
            auto v = std::make_shared<mgcl::Cuboid>(N, N, N);

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
        MPI_Finalize();
    }
}

/**
 * @brief Calculates error for each cell, e.g. difference between solution and approximation. Dimensions must match.
 *
 * @param solution
 * @param approximation
 * @return std::shared_ptr<mgcl::Cuboid>
 */
std::shared_ptr<mgcl::Cuboid> calculateError(mgcl::Cuboid &solution, mgcl::Cuboid &approximation)
{
    if (solution.getM() != approximation.getM() ||
        solution.getN() != approximation.getN() ||
        solution.getO() != approximation.getO())
        throw std::invalid_argument("Dimensions do not match.");

    auto ret = std::make_shared<mgcl::Cuboid>(solution.getM(), solution.getN(), solution.getO());
    for (int i = 0; i < solution.getM(); i++)
        for (int j = 0; j < solution.getN(); j++)
            for (int k = 0; k < solution.getO(); k++)
            {
                (*ret)[i][j][k] = fabs(solution[i][j][k] - approximation[i][j][k]);
            }

    return ret;
}

/**
 * @brief Returns the maximum absolute error. calculateError should have been called first.
 *
 * @param error
 * @return double
 */
double calculateMaxError(mgcl::Cuboid &error)
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
double calculateErrorNorm(double h, mgcl::Cuboid &error)
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
