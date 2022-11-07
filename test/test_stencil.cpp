#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <memory>
#include <random>

#include "../cuboid.hpp"
#include "../multigrid_engine.hpp"
#include "../stencil.hpp"

#include "matrix2d.hpp"

// fills real cells of a stencil with random values
template <int N>
void fillRandomStencil(mgcl::VaryingStencil<N> &s)
{
    // specify fixed seed to get the same random values each run
    std::mt19937 rng(123);
    std::uniform_real_distribution<std::mt19937::result_type> rand(1, 9); // distribution in range [1, 9]

    int ghm = s.getGhostsDim1();
    int ghn = s.getGhostsDim2();
    int gho = s.getGhostsDim3();

    for (int i = ghm; i < s.getDim1() + ghm; i++)
        for (int j = ghn; j < s.getDim2() + ghn; j++)
            for (int k = gho; k < s.getDim3() + gho; k++)
                for (int ii = 0; i < s.getDim1(); i++)
                    for (int jj = 0; j < s.getDim2(); j++)
                        for (int kk = 0; k < s.getDim3(); k++)
                        {
                            s[i][j][k][ii][jj][kk] = rand(rng);
                        }
}

TEST_CASE("StencilLaplace7p")
{
    int N = 4;
    int gh = 1;

    mgcl::Cuboid v(N, N, N, gh, gh, gh);
    v[1][1][1] = 8;
    v[1][1][0] = 4;
    v[1][1][2] = 16;
    v[1][0][1] = 4;
    v[1][2][1] = 2;
    v[0][1][1] = 1;
    v[2][1][1] = 2;

    double expectedFactor = 4.0 * 4.0;

    mgcl::Cuboid f(N, N, N, gh, gh, gh);
    mgcl::Cuboid r(N, N, N, gh, gh, gh);
    mgcl::VaryingStencil3x3x3 stencilValues(1, 1, 1, 0, 0, 0); // just a dummy

    mgcl::MultigridEngine::residualSeq(f, v, r, mgcl::MGCL_L2, mgcl::MGCL_LAPLACE_7POINT, expectedFactor, stencilValues, 0);

    double expected = expectedFactor * (6.0 * 8 - 4 - 16 - 4 - 2 - 1 - 2);
    REQUIRE(-r[1][1][1] == Catch::Approx(expected));
}

TEST_CASE("StencilLaplace19p")
{
    int N = 4;
    int gh = 1;

    mgcl::Cuboid v(N, N, N, gh, gh, gh);
    v[1][1][1] = 8;
    v[1][1][0] = 4;
    v[1][1][2] = 16;
    v[1][0][1] = 4;
    v[1][2][1] = 2;
    v[0][1][1] = 1;
    v[2][1][1] = 2;

    v[1][0][0] = 8;
    v[1][0][2] = 4;
    v[1][2][0] = 1;
    v[1][2][2] = 2;

    v[0][1][0] = 16;
    v[0][1][2] = 32;
    v[2][1][0] = 8;
    v[2][1][2] = 4;

    v[0][0][1] = 4;
    v[0][2][1] = 4;
    v[2][0][1] = 2;
    v[2][2][1] = 1;

    double expectedFactor = (4.0 * 4.0) / 6.0;

    mgcl::Cuboid f(N, N, N, gh, gh, gh);
    mgcl::Cuboid r(N, N, N, gh, gh, gh);
    mgcl::VaryingStencil3x3x3 stencilValues(1, 1, 1, 0, 0, 0); // just a dummy

    mgcl::MultigridEngine::residualSeq(f, v, r, mgcl::MGCL_L2, mgcl::MGCL_LAPLACE_19POINT, expectedFactor, stencilValues, 0);

    double expected = expectedFactor * (24.0 * 8 - 2.0 * (4 + 16 + 4 + 2 + 1 + 2) - 8 - 4 - 1 - 2 - 16 - 32 - 8 - 4 - 4 - 4 - 2 - 1);
    REQUIRE(-r[1][1][1] == Catch::Approx(expected));
}

TEST_CASE("StencilLaplace27p")
{
    int N = 4;
    int gh = 1;

    mgcl::Cuboid v(N, N, N, gh, gh, gh);
    v[1][1][1] = 8;
    v[1][1][0] = 4;
    v[1][1][2] = 16;
    v[1][0][1] = 4;
    v[1][2][1] = 2;
    v[0][1][1] = 1;
    v[2][1][1] = 2;

    v[1][0][0] = 8;
    v[1][0][2] = 4;
    v[1][2][0] = 1;
    v[1][2][2] = 2;

    v[0][1][0] = 16;
    v[0][1][2] = 32;
    v[2][1][0] = 8;
    v[2][1][2] = 4;

    v[0][0][1] = 4;
    v[0][2][1] = 4;
    v[2][0][1] = 2;
    v[2][2][1] = 1;

    v[0][0][0] = 8;
    v[0][0][2] = 4;
    v[0][2][0] = 1;
    v[0][2][2] = 2;
    v[2][0][0] = 16;
    v[2][0][2] = 4;
    v[2][2][0] = 8;
    v[2][2][2] = 2;

    double expectedFactor = (4.0 * 4.0) / 30.0;

    mgcl::Cuboid f(N, N, N, gh, gh, gh);
    mgcl::Cuboid r(N, N, N, gh, gh, gh);
    mgcl::VaryingStencil3x3x3 stencilValues(1, 1, 1, 0, 0, 0); // just a dummy

    mgcl::MultigridEngine::residualSeq(f, v, r, mgcl::MGCL_L2, mgcl::MGCL_LAPLACE_27POINT, expectedFactor, stencilValues, 0);

    double expected = expectedFactor * (128.0 * 8 - 14.0 * (4 + 16 + 4 + 2 + 1 + 2) -
                                        3.0 * (8 + 4 + 1 + 2 + 16 + 32 + 8 + 4 + 4 + 4 + 2 + 1) -
                                        8 - 4 - 1 - 2 - 16 - 4 - 8 - 2);
    REQUIRE(-r[1][1][1] == Catch::Approx(expected));
}

TEST_CASE("StencilVarying7p")
{
    int N = 4;
    int gh = 1;

    mgcl::Cuboid v(N, N, N, gh, gh, gh);
    v[1][1][1] = 8;
    v[1][1][0] = 4;
    v[1][1][2] = 16;
    v[1][0][1] = 4;
    v[1][2][1] = 2;
    v[0][1][1] = 1;
    v[2][1][1] = 2;

    mgcl::VaryingStencil3x3x3 vals(N, N, N, gh, gh, gh);
    REQUIRE(vals.getDim1() == v.getM());
    REQUIRE(vals.getDim2() == v.getN());
    REQUIRE(vals.getDim3() == v.getO());

    // fill varying stencil values with 7p Laplace stencil
    double h2inv = 4.0 * 4.0;
    for (int i = 0; i < vals.getDim1gh(); i++)
        for (int j = 0; j < vals.getDim2gh(); j++)
            for (int k = 0; k < vals.getDim3gh(); k++)
            {
                vals[i][j][k][1][1][1] = 6.0 * h2inv;
                vals[i][j][k][1][1][0] = -1.0 * h2inv;
                vals[i][j][k][1][1][2] = -1.0 * h2inv;
                vals[i][j][k][1][0][1] = -1.0 * h2inv;
                vals[i][j][k][1][2][1] = -1.0 * h2inv;
                vals[i][j][k][0][1][1] = -1.0 * h2inv;
                vals[i][j][k][2][1][1] = -1.0 * h2inv;
            }

    mgcl::Cuboid f(N, N, N, gh, gh, gh);
    mgcl::Cuboid r(N, N, N, gh, gh, gh);

    mgcl::MultigridEngine::residualSeq(f, v, r, mgcl::MGCL_L2, mgcl::MGCL_VARYING_7POINT, 1, vals, 0);

    double expected = h2inv * (6.0 * 8 - 4 - 16 - 4 - 2 - 1 - 2);
    REQUIRE(-r[1][1][1] == Catch::Approx(expected));
}

TEST_CASE("StencilVarying19p")
{
    int N = 4;
    int gh = 1;

    mgcl::Cuboid v(N, N, N, gh, gh, gh);
    v[1][1][1] = 8;
    v[1][1][0] = 4;
    v[1][1][2] = 16;
    v[1][0][1] = 4;
    v[1][2][1] = 2;
    v[0][1][1] = 1;
    v[2][1][1] = 2;

    v[1][0][0] = 8;
    v[1][0][2] = 4;
    v[1][2][0] = 1;
    v[1][2][2] = 2;

    v[0][1][0] = 16;
    v[0][1][2] = 32;
    v[2][1][0] = 8;
    v[2][1][2] = 4;

    v[0][0][1] = 4;
    v[0][2][1] = 4;
    v[2][0][1] = 2;
    v[2][2][1] = 1;

    mgcl::VaryingStencil3x3x3 vals(N, N, N, gh, gh, gh);
    REQUIRE(vals.getDim1() == v.getM());
    REQUIRE(vals.getDim2() == v.getN());
    REQUIRE(vals.getDim3() == v.getO());

    // fill varying stencil with 19p Laplace stencil
    double h2inv = (4.0 * 4.0) / 6.0;
    for (int i = 0; i < vals.getDim1gh(); i++)
        for (int j = 0; j < vals.getDim2gh(); j++)
            for (int k = 0; k < vals.getDim3gh(); k++)
            {
                vals[i][j][k][1][1][1] = 24.0 * h2inv;
                vals[i][j][k][1][1][0] = -2.0 * h2inv;
                vals[i][j][k][1][1][2] = -2.0 * h2inv;
                vals[i][j][k][1][0][1] = -2.0 * h2inv;
                vals[i][j][k][1][2][1] = -2.0 * h2inv;
                vals[i][j][k][0][1][1] = -2.0 * h2inv;
                vals[i][j][k][2][1][1] = -2.0 * h2inv;
                vals[i][j][k][1][0][0] = -h2inv;
                vals[i][j][k][1][0][2] = -h2inv;
                vals[i][j][k][1][2][0] = -h2inv;
                vals[i][j][k][1][2][2] = -h2inv;
                vals[i][j][k][0][1][0] = -h2inv;
                vals[i][j][k][0][1][2] = -h2inv;
                vals[i][j][k][2][1][0] = -h2inv;
                vals[i][j][k][2][1][2] = -h2inv;
                vals[i][j][k][0][0][1] = -h2inv;
                vals[i][j][k][0][2][1] = -h2inv;
                vals[i][j][k][2][0][1] = -h2inv;
                vals[i][j][k][2][2][1] = -h2inv;
            }

    mgcl::Cuboid f(N, N, N, gh, gh, gh);
    mgcl::Cuboid r(N, N, N, gh, gh, gh);

    mgcl::MultigridEngine::residualSeq(f, v, r, mgcl::MGCL_L2, mgcl::MGCL_VARYING_19POINT, 1, vals, 0);

    double expected = h2inv * (24.0 * 8 - 2.0 * (4 + 16 + 4 + 2 + 1 + 2) - 8 - 4 - 1 - 2 - 16 - 32 - 8 - 4 - 4 - 4 - 2 - 1);
    REQUIRE(-r[1][1][1] == Catch::Approx(expected));
}

TEST_CASE("StencilVarying27p")
{
    int N = 4;
    int gh = 1;

    mgcl::Cuboid v(N, N, N, gh, gh, gh);
    v[1][1][1] = 8;
    v[1][1][0] = 4;
    v[1][1][2] = 16;
    v[1][0][1] = 4;
    v[1][2][1] = 2;
    v[0][1][1] = 1;
    v[2][1][1] = 2;

    v[1][0][0] = 8;
    v[1][0][2] = 4;
    v[1][2][0] = 1;
    v[1][2][2] = 2;

    v[0][1][0] = 16;
    v[0][1][2] = 32;
    v[2][1][0] = 8;
    v[2][1][2] = 4;

    v[0][0][1] = 4;
    v[0][2][1] = 4;
    v[2][0][1] = 2;
    v[2][2][1] = 1;

    v[0][0][0] = 8;
    v[0][0][2] = 4;
    v[0][2][0] = 1;
    v[0][2][2] = 2;
    v[2][0][0] = 16;
    v[2][0][2] = 4;
    v[2][2][0] = 8;
    v[2][2][2] = 2;

    mgcl::VaryingStencil3x3x3 vals(N, N, N, gh, gh, gh);
    REQUIRE(vals.getDim1() == v.getM());
    REQUIRE(vals.getDim2() == v.getN());
    REQUIRE(vals.getDim3() == v.getO());

    // fill varying stencil with 19p Laplace stencil
    double h2inv = (4.0 * 4.0) / 30.0;
    for (int i = 0; i < vals.getDim1gh(); i++)
        for (int j = 0; j < vals.getDim2gh(); j++)
            for (int k = 0; k < vals.getDim3gh(); k++)
            {
                vals[i][j][k][1][1][1] = 128.0 * h2inv;
                vals[i][j][k][1][1][0] = -14.0 * h2inv;
                vals[i][j][k][1][1][2] = -14.0 * h2inv;
                vals[i][j][k][1][0][1] = -14.0 * h2inv;
                vals[i][j][k][1][2][1] = -14.0 * h2inv;
                vals[i][j][k][0][1][1] = -14.0 * h2inv;
                vals[i][j][k][2][1][1] = -14.0 * h2inv;
                vals[i][j][k][1][0][0] = -3.0 * h2inv;
                vals[i][j][k][1][0][2] = -3.0 * h2inv;
                vals[i][j][k][1][2][0] = -3.0 * h2inv;
                vals[i][j][k][1][2][2] = -3.0 * h2inv;
                vals[i][j][k][0][1][0] = -3.0 * h2inv;
                vals[i][j][k][0][1][2] = -3.0 * h2inv;
                vals[i][j][k][2][1][0] = -3.0 * h2inv;
                vals[i][j][k][2][1][2] = -3.0 * h2inv;
                vals[i][j][k][0][0][1] = -3.0 * h2inv;
                vals[i][j][k][0][2][1] = -3.0 * h2inv;
                vals[i][j][k][2][0][1] = -3.0 * h2inv;
                vals[i][j][k][2][2][1] = -3.0 * h2inv;
                vals[i][j][k][0][0][0] = -h2inv;
                vals[i][j][k][0][0][2] = -h2inv;
                vals[i][j][k][0][2][0] = -h2inv;
                vals[i][j][k][0][2][2] = -h2inv;
                vals[i][j][k][2][0][0] = -h2inv;
                vals[i][j][k][2][0][2] = -h2inv;
                vals[i][j][k][2][2][0] = -h2inv;
                vals[i][j][k][2][2][2] = -h2inv;
            }

    mgcl::Cuboid f(N, N, N, gh, gh, gh);
    mgcl::Cuboid r(N, N, N, gh, gh, gh);

    mgcl::MultigridEngine::residualSeq(f, v, r, mgcl::MGCL_L2, mgcl::MGCL_VARYING_27POINT, 1, vals, 0);

    double expected = h2inv * (128.0 * 8 - 14.0 * (4 + 16 + 4 + 2 + 1 + 2) -
                               3.0 * (8 + 4 + 1 + 2 + 16 + 32 + 8 + 4 + 4 + 4 + 2 + 1) -
                               8 - 4 - 1 - 2 - 16 - 4 - 8 - 2);
    REQUIRE(-r[1][1][1] == Catch::Approx(expected));
}

TEST_CASE("VaryingStencil move ctor")
{
    int n = GENERATE(1, 2, 3);
    int m = GENERATE(1, 2, 3);
    int o = GENERATE(1, 2, 3);

    mgcl::VaryingStencil3x3x3 h(m, n, o, 0, 0, 0);
    h.fillRandom();

    // copy manually for checking results
    mgcl::VaryingStencil3x3x3 h_check(m, n, o, 0, 0, 0);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < o; k++)
                for (int ii = 0; ii < 3; ii++)
                    for (int jj = 0; jj < 3; jj++)
                        for (int kk = 0; kk < 3; kk++)
                        {
                            h_check[i][j][k][ii][jj][kk] = h[i][j][k][ii][jj][kk];
                        }

    // check move ctor
    auto h2(std::move(h));

    CHECK(h.getDim1() == 0);
    CHECK(h.getDim2() == 0);
    CHECK(h.getDim3() == 0);
    CHECK(h.getDim4() == 0);
    CHECK(h.getDim5() == 0);
    CHECK(h.getDim6() == 0);
    CHECK(h.getData() == nullptr);
    CHECK(h2.isEqual(h_check));

    // check move assignment
    auto h3 = std::move(h2);

    CHECK(h2.getDim1() == 0);
    CHECK(h2.getDim2() == 0);
    CHECK(h2.getDim3() == 0);
    CHECK(h2.getDim4() == 0);
    CHECK(h2.getDim5() == 0);
    CHECK(h2.getDim6() == 0);
    CHECK(h2.getData() == nullptr);
    CHECK(h3.isEqual(h_check));
}

TEST_CASE("VaryingStencil::multiply")
{
    SECTION("valid N=3, not periodic")
    {
        int m = GENERATE(2, 3, 4);
        int n = GENERATE(2, 3, 4);
        int o = GENERATE(2, 3, 4);

        mgcl::VaryingStencil3x3x3 a(m, n, o, 0, 0, 0);
        mgcl::VaryingStencil3x3x3 b(m, n, o, 1, 1, 1);

        double factor1 = 8.0 / 64.0;
        double factor2 = 4.0 / 64.0;
        double factor3 = 2.0 / 64.0;
        double factor4 = 1.0 / 64.0;

        // fill a and b
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < o; k++)
                {
                    // 7-point Laplace
                    a[i][j][k][0][1][1] = 1;
                    a[i][j][k][1][0][1] = 1;
                    a[i][j][k][1][1][0] = 1;
                    a[i][j][k][1][1][1] = -6;
                    a[i][j][k][1][1][2] = 1;
                    a[i][j][k][1][2][1] = 1;
                    a[i][j][k][2][1][1] = 1;

                    // full-weight restriction
                    // fill only real cells, ghosts cells must be zero (Dirichlet condition)
                    b[i + 1][j + 1][k + 1][0][0][0] = factor4;
                    b[i + 1][j + 1][k + 1][0][0][1] = factor3;
                    b[i + 1][j + 1][k + 1][0][0][2] = factor4;
                    b[i + 1][j + 1][k + 1][0][1][0] = factor3;
                    b[i + 1][j + 1][k + 1][0][1][1] = factor2;
                    b[i + 1][j + 1][k + 1][0][1][2] = factor3;
                    b[i + 1][j + 1][k + 1][0][2][0] = factor4;
                    b[i + 1][j + 1][k + 1][0][2][1] = factor3;
                    b[i + 1][j + 1][k + 1][0][2][2] = factor4;
                    b[i + 1][j + 1][k + 1][1][0][0] = factor3;
                    b[i + 1][j + 1][k + 1][1][0][1] = factor2;
                    b[i + 1][j + 1][k + 1][1][0][2] = factor3;
                    b[i + 1][j + 1][k + 1][1][1][0] = factor2;
                    b[i + 1][j + 1][k + 1][1][1][1] = factor1;
                    b[i + 1][j + 1][k + 1][1][1][2] = factor2;
                    b[i + 1][j + 1][k + 1][1][2][0] = factor3;
                    b[i + 1][j + 1][k + 1][1][2][1] = factor2;
                    b[i + 1][j + 1][k + 1][1][2][2] = factor3;
                    b[i + 1][j + 1][k + 1][2][0][0] = factor4;
                    b[i + 1][j + 1][k + 1][2][0][1] = factor3;
                    b[i + 1][j + 1][k + 1][2][0][2] = factor4;
                    b[i + 1][j + 1][k + 1][2][1][0] = factor3;
                    b[i + 1][j + 1][k + 1][2][1][1] = factor2;
                    b[i + 1][j + 1][k + 1][2][1][2] = factor3;
                    b[i + 1][j + 1][k + 1][2][2][0] = factor4;
                    b[i + 1][j + 1][k + 1][2][2][1] = factor3;
                    b[i + 1][j + 1][k + 1][2][2][2] = factor4;
                }

        auto c = a.multiply(b);
        auto cstar = a * b;

        // check result against explicitly calculated matrix product
        auto c_expected = mgcl_test::Matrix2d::laplace7p3d(m, n, o, false) * mgcl_test::Matrix2d::restrictionFullWeight(m, n, o, false);
        auto c_m2d = mgcl_test::Matrix2d::fromVaryingStencil(c, false);
        auto cstar_m2d = mgcl_test::Matrix2d::fromVaryingStencil(cstar, false);

        REQUIRE(c_expected.getM() == c_m2d.getM());
        REQUIRE(c_expected.getN() == c_m2d.getN());
        REQUIRE(c_expected.getM() == cstar_m2d.getM());
        REQUIRE(c_expected.getN() == cstar_m2d.getN());

        CHECK(c_m2d == c_expected);
        CHECK(cstar_m2d == c_expected);
    }

    SECTION("valid N=3, periodic")
    {
        int m = GENERATE(2, 3, 4);
        int n = GENERATE(2, 3, 4);
        int o = GENERATE(2, 3, 4);

        mgcl::VaryingStencil3x3x3 a(m, n, o, 0, 0, 0);
        mgcl::VaryingStencil3x3x3 b(m, n, o, 1, 1, 1);

        double factor1 = 8.0 / 64.0;
        double factor2 = 4.0 / 64.0;
        double factor3 = 2.0 / 64.0;
        double factor4 = 1.0 / 64.0;

        // fill a and b
        for (int i = 0; i < m + 2; i++)
            for (int j = 0; j < n + 2; j++)
                for (int k = 0; k < o + 2; k++)
                {
                    // 7-point Laplace, real cells only
                    if (i < m && j < n && k < o)
                    {
                        a[i][j][k][0][1][1] = 1;
                        a[i][j][k][1][0][1] = 1;
                        a[i][j][k][1][1][0] = 1;
                        a[i][j][k][1][1][1] = -6;
                        a[i][j][k][1][1][2] = 1;
                        a[i][j][k][1][2][1] = 1;
                        a[i][j][k][2][1][1] = 1;
                    }

                    // full-weight restriction
                    b[i][j][k][0][0][0] = factor4;
                    b[i][j][k][0][0][1] = factor3;
                    b[i][j][k][0][0][2] = factor4;
                    b[i][j][k][0][1][0] = factor3;
                    b[i][j][k][0][1][1] = factor2;
                    b[i][j][k][0][1][2] = factor3;
                    b[i][j][k][0][2][0] = factor4;
                    b[i][j][k][0][2][1] = factor3;
                    b[i][j][k][0][2][2] = factor4;
                    b[i][j][k][1][0][0] = factor3;
                    b[i][j][k][1][0][1] = factor2;
                    b[i][j][k][1][0][2] = factor3;
                    b[i][j][k][1][1][0] = factor2;
                    b[i][j][k][1][1][1] = factor1;
                    b[i][j][k][1][1][2] = factor2;
                    b[i][j][k][1][2][0] = factor3;
                    b[i][j][k][1][2][1] = factor2;
                    b[i][j][k][1][2][2] = factor3;
                    b[i][j][k][2][0][0] = factor4;
                    b[i][j][k][2][0][1] = factor3;
                    b[i][j][k][2][0][2] = factor4;
                    b[i][j][k][2][1][0] = factor3;
                    b[i][j][k][2][1][1] = factor2;
                    b[i][j][k][2][1][2] = factor3;
                    b[i][j][k][2][2][0] = factor4;
                    b[i][j][k][2][2][1] = factor3;
                    b[i][j][k][2][2][2] = factor4;
                }

        auto c = a.multiply(b);
        auto cstar = a * b;

        // check result against explicitly calculated matrix product
        auto c_expected = mgcl_test::Matrix2d::laplace7p3d(m, n, o) * mgcl_test::Matrix2d::restrictionFullWeight(m, n, o);
        auto c_m2d = mgcl_test::Matrix2d::fromVaryingStencil(c, true);
        auto cstar_m2d = mgcl_test::Matrix2d::fromVaryingStencil(cstar, true);

        REQUIRE(c_expected.getM() == c_m2d.getM());
        REQUIRE(c_expected.getN() == c_m2d.getN());
        REQUIRE(c_expected.getM() == cstar_m2d.getM());
        REQUIRE(c_expected.getN() == cstar_m2d.getN());

        CHECK(c_m2d == c_expected);
        CHECK(cstar_m2d == c_expected);
    }

    SECTION("different Ns, random values")
    {
        int m = GENERATE(2, 3, 4);
        int n = GENERATE(2, 3, 4);
        int o = GENERATE(2, 3, 4);

        mgcl::VaryingStencil3x3x3 a3(m, n, o, 0, 0, 0);
        fillRandomStencil<3>(a3);
        mgcl::VaryingStencil5x5x5 a5(m, n, o, 0, 0, 0);
        fillRandomStencil<5>(a5);
        mgcl::VaryingStencil7x7x7 a7(m, n, o, 0, 0, 0);
        fillRandomStencil<7>(a7);

        mgcl::VaryingStencil3x3x3 b3(m, n, o, 1, 1, 1);
        fillRandomStencil<3>(b3);
        mgcl::VaryingStencil5x5x5 b5(m, n, o, 2, 2, 2);
        fillRandomStencil<5>(b5);
        mgcl::VaryingStencil7x7x7 b7(m, n, o, 3, 3, 3);
        fillRandomStencil<7>(b7);

        // TODO update ghosts

        // TODO multiply each versions

        auto c33 = a3 * b3;

        // // check result against explicitly calculated matrix product
        // auto c_expected = mgcl_test::Matrix2d::laplace7p3d(m, n, o) * mgcl_test::Matrix2d::restrictionFullWeight(m, n, o);
        // auto c_m2d = mgcl_test::Matrix2d::fromVaryingStencil(c, true);

        // REQUIRE(c_expected.getM() == c_m2d.getM());
        // REQUIRE(c_expected.getN() == c_m2d.getN());

        // CHECK(c_m2d == c_expected);
    }

    SECTION("throwing")
    {
        int m = GENERATE(1, 2, 3);
        int n = GENERATE(1, 2, 3);
        int o = GENERATE(1, 2, 3);
        int ghm = GENERATE(1, 2, 3);
        int ghn = GENERATE(1, 2, 3);
        int gho = GENERATE(1, 2, 3);

        mgcl::VaryingStencil3x3x3 a(m, n, o, ghm, ghn, gho);
        mgcl::VaryingStencil3x3x3 b(m, n, o, ghm, ghn, gho);

        if (a.getDim1gh() != b.getDim1gh() || a.getDim2gh() != b.getDim2gh() || a.getDim3gh() != b.getDim3gh())
            REQUIRE_THROWS(a * b);
        else
            REQUIRE_NOTHROW(a * b);
    }
}

TEST_CASE("VaryingStencil::create3dFullWeightRestriction")
{
    int m = GENERATE(1, 2, 3);
    int n = GENERATE(1, 2, 3);
    int o = GENERATE(1, 2, 3);
    int ghm = GENERATE(1, 2);
    int ghn = GENERATE(1, 2);
    int gho = GENERATE(1, 2);

    auto rptr = mgcl::create3dFullWeightRestrictionStencil(m, n, o, ghm, ghn, gho);
    auto &r = *rptr;

    REQUIRE(r.getDim1() == m);
    REQUIRE(r.getDim2() == n);
    REQUIRE(r.getDim3() == o);

    double factor = 1.0 / 64.0;

    int mgh = m + 2 * ghm;
    int ngh = n + 2 * ghn;
    int ogh = o + 2 * gho;

    for (int i = 0; i < mgh; i++)
        for (int j = 0; j < ngh; j++)
            for (int k = 0; k < ogh; k++)
            {

                // full-weight restriction, scaled by 64
                CHECK(r[i][j][k][0][0][0] == 1 * factor);
                CHECK(r[i][j][k][0][0][1] == 2 * factor);
                CHECK(r[i][j][k][0][0][2] == 1 * factor);
                CHECK(r[i][j][k][0][1][0] == 2 * factor);
                CHECK(r[i][j][k][0][1][1] == 4 * factor);
                CHECK(r[i][j][k][0][1][2] == 2 * factor);
                CHECK(r[i][j][k][0][2][0] == 1 * factor);
                CHECK(r[i][j][k][0][2][1] == 2 * factor);
                CHECK(r[i][j][k][0][2][2] == 1 * factor);
                CHECK(r[i][j][k][1][0][0] == 2 * factor);
                CHECK(r[i][j][k][1][0][1] == 4 * factor);
                CHECK(r[i][j][k][1][0][2] == 2 * factor);
                CHECK(r[i][j][k][1][1][0] == 4 * factor);
                CHECK(r[i][j][k][1][1][1] == 8 * factor);
                CHECK(r[i][j][k][1][1][2] == 4 * factor);
                CHECK(r[i][j][k][1][2][0] == 2 * factor);
                CHECK(r[i][j][k][1][2][1] == 4 * factor);
                CHECK(r[i][j][k][1][2][2] == 2 * factor);
                CHECK(r[i][j][k][2][0][0] == 1 * factor);
                CHECK(r[i][j][k][2][0][1] == 2 * factor);
                CHECK(r[i][j][k][2][0][2] == 1 * factor);
                CHECK(r[i][j][k][2][1][0] == 2 * factor);
                CHECK(r[i][j][k][2][1][1] == 4 * factor);
                CHECK(r[i][j][k][2][1][2] == 2 * factor);
                CHECK(r[i][j][k][2][2][0] == 1 * factor);
                CHECK(r[i][j][k][2][2][1] == 2 * factor);
                CHECK(r[i][j][k][2][2][2] == 1 * factor);
            }
}
