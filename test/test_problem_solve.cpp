#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>

#include "../cuboid.hpp"
#include "../problem.hpp"

mgcl::Cuboid calculateError(mgcl::Cuboid &solution, mgcl::Cuboid &approximation);
double calculateMaxError(mgcl::Cuboid &error);
double calculateErrorNorm(double h, mgcl::Cuboid &error);

/**
 * @brief Tests if solving works correctly.
 *
 */
TEST_CASE("Problem solving: periodic 4th order")
{
    int N = 32;
    double h = 1.0 / (double)N;

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
    p.setMaxiterVcycles(10);
    p.setTol(1e-14);
    p.setNu1(2);
    p.setNu2(2);
    p.setOmega(0.8);

    SECTION("Sequential")
    {
        p.solveSeq();

        // check if input v is equal to the v stored in Problem instance
        REQUIRE(v.get() == p.getVPtr().get());
        REQUIRE(v->isEqual(p.getV()));

        // check if solution is good
        auto err = calculateError(solution, *v);
        auto errNorm = calculateErrorNorm(1.0 / (double)N, err);
        auto errMax = calculateMaxError(err);

        solution.dumpToFile("out_solution.txt");
        (*v).dumpToFile("out_v.txt");

        std::cout
            << std::scientific << std::setprecision(17) << "||e||_2 = " << errNorm << std::endl
            << std::scientific << std::setprecision(17) << "e_max = " << errMax << std::endl;

        CHECK(errNorm < 1e-2);
        CHECK(errMax < 1e-2);

        // check if error is equal to old mgcl implementation (problem params must match)
        if (p.getMaxiterVcycles() == 10 && N == 32 && p.getTol() == 1e-14 &&
            p.getNu1() == 2 && p.getNu2() == 2 && p.getOmega() == 0.8 &&
            p.getStencil()->getType() == mgcl::MGCL_LAPLACE_7POINT)
        {
            CHECK(fabs(errNorm - 3.93115528889639940e-03) < 1e-14);
            CHECK(fabs(errMax - 3.95723982871564600e-03) < 1e-14);
        }
    }

    SECTION("using OpenCL")
    {
        p.setUseOpencl(true);
        p.setReadResults(true);
        p.setDeviceType(CL_DEVICE_TYPE_GPU);
        // p.setDeviceName("Quadro");
        p.solve();

        // check if solution is good
        auto err = calculateError(solution, *v);
        auto errNorm = calculateErrorNorm(1.0 / (double)N, err);
        auto errMax = calculateMaxError(err);

        std::cout << std::scientific << "||e||_2 = " << errNorm << std::endl
                  << std::scientific << "e_max = " << errMax << std::endl;

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
}

/**
 * @brief Calculates error for each cell, e.g. difference between solution and approximation. Dimensions must match.
 *
 * @param solution
 * @param approximation
 * @return mgcl::Cuboid
 */
mgcl::Cuboid calculateError(mgcl::Cuboid &solution, mgcl::Cuboid &approximation)
{
    if (solution.getM() != approximation.getM() ||
        solution.getN() != approximation.getN() ||
        solution.getO() != approximation.getO())
        throw std::invalid_argument("Dimensions do not match.");

    mgcl::Cuboid ret(solution.getM(), solution.getN(), solution.getO());
    for (int i = 0; i < solution.getM(); i++)
        for (int j = 0; j < solution.getN(); j++)
            for (int k = 0; k < solution.getO(); k++)
            {
                ret[i][j][k] = fabs(solution[i][j][k] - approximation[i][j][k]);
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
