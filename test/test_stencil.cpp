#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <memory>

#include "../cuboid.hpp"
#include "../multigrid_engine.hpp"
#include "../stencil.hpp"

#include "matrix2d.hpp"

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
    mgcl::Hypercube4d stencilValues(1, 1, 1, 1); // just a dummy

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
    mgcl::Hypercube4d stencilValues(1, 1, 1, 1); // just a dummy

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
    mgcl::Hypercube4d stencilValues(1, 1, 1, 1); // just a dummy

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

    mgcl::Hypercube4d vals(N, N, N, 7, 1, 1, 1, 0);
    REQUIRE(vals.getDim1() == v.getM());
    REQUIRE(vals.getDim2() == v.getN());
    REQUIRE(vals.getDim3() == v.getO());
    REQUIRE(vals.getDim4() == 7);

    // fill varying stencil values with 7p Laplace stencil
    double h2inv = 4.0 * 4.0;
    for (int i = 0; i < vals.getDim1gh(); i++)
        for (int j = 0; j < vals.getDim2gh(); j++)
            for (int k = 0; k < vals.getDim3gh(); k++)
            {
                vals[i][j][k][0] = 6.0 * h2inv;
                vals[i][j][k][1] = -1.0 * h2inv;
                vals[i][j][k][2] = -1.0 * h2inv;
                vals[i][j][k][3] = -1.0 * h2inv;
                vals[i][j][k][4] = -1.0 * h2inv;
                vals[i][j][k][5] = -1.0 * h2inv;
                vals[i][j][k][6] = -1.0 * h2inv;
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

    mgcl::Hypercube4d vals(N, N, N, 19, 1, 1, 1, 0);
    REQUIRE(vals.getDim1() == v.getM());
    REQUIRE(vals.getDim2() == v.getN());
    REQUIRE(vals.getDim3() == v.getO());
    REQUIRE(vals.getDim4() == 19);

    // fill varying stencil with 19p Laplace stencil
    double h2inv = (4.0 * 4.0) / 6.0;
    for (int i = 0; i < vals.getDim1gh(); i++)
        for (int j = 0; j < vals.getDim2gh(); j++)
            for (int k = 0; k < vals.getDim3gh(); k++)
            {
                vals[i][j][k][0] = 24.0 * h2inv;
                vals[i][j][k][1] = -2.0 * h2inv;
                vals[i][j][k][2] = -2.0 * h2inv;
                vals[i][j][k][3] = -2.0 * h2inv;
                vals[i][j][k][4] = -2.0 * h2inv;
                vals[i][j][k][5] = -2.0 * h2inv;
                vals[i][j][k][6] = -2.0 * h2inv;
                vals[i][j][k][7] = -h2inv;
                vals[i][j][k][8] = -h2inv;
                vals[i][j][k][9] = -h2inv;
                vals[i][j][k][10] = -h2inv;
                vals[i][j][k][11] = -h2inv;
                vals[i][j][k][12] = -h2inv;
                vals[i][j][k][13] = -h2inv;
                vals[i][j][k][14] = -h2inv;
                vals[i][j][k][15] = -h2inv;
                vals[i][j][k][16] = -h2inv;
                vals[i][j][k][17] = -h2inv;
                vals[i][j][k][18] = -h2inv;
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

    mgcl::Hypercube4d vals(N, N, N, 27, 1, 1, 1, 0);
    REQUIRE(vals.getDim1() == v.getM());
    REQUIRE(vals.getDim2() == v.getN());
    REQUIRE(vals.getDim3() == v.getO());
    REQUIRE(vals.getDim4() == 27);

    // fill varying stencil with 19p Laplace stencil
    double h2inv = (4.0 * 4.0) / 30.0;
    for (int i = 0; i < vals.getDim1gh(); i++)
        for (int j = 0; j < vals.getDim2gh(); j++)
            for (int k = 0; k < vals.getDim3gh(); k++)
            {
                vals[i][j][k][0] = 128.0 * h2inv;
                vals[i][j][k][1] = -14.0 * h2inv;
                vals[i][j][k][2] = -14.0 * h2inv;
                vals[i][j][k][3] = -14.0 * h2inv;
                vals[i][j][k][4] = -14.0 * h2inv;
                vals[i][j][k][5] = -14.0 * h2inv;
                vals[i][j][k][6] = -14.0 * h2inv;
                vals[i][j][k][7] = -3.0 * h2inv;
                vals[i][j][k][8] = -3.0 * h2inv;
                vals[i][j][k][9] = -3.0 * h2inv;
                vals[i][j][k][10] = -3.0 * h2inv;
                vals[i][j][k][11] = -3.0 * h2inv;
                vals[i][j][k][12] = -3.0 * h2inv;
                vals[i][j][k][13] = -3.0 * h2inv;
                vals[i][j][k][14] = -3.0 * h2inv;
                vals[i][j][k][15] = -3.0 * h2inv;
                vals[i][j][k][16] = -3.0 * h2inv;
                vals[i][j][k][17] = -3.0 * h2inv;
                vals[i][j][k][18] = -3.0 * h2inv;
                vals[i][j][k][19] = -h2inv;
                vals[i][j][k][20] = -h2inv;
                vals[i][j][k][21] = -h2inv;
                vals[i][j][k][22] = -h2inv;
                vals[i][j][k][23] = -h2inv;
                vals[i][j][k][24] = -h2inv;
                vals[i][j][k][25] = -h2inv;
                vals[i][j][k][26] = -h2inv;
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
    SECTION("valid N=3")
    {
        int m = GENERATE(1, 2, 3, 4);
        int n = GENERATE(1, 2, 3, 4);
        int o = GENERATE(1, 2, 3, 4);
        int ghm = GENERATE(1, 2);
        int ghn = GENERATE(1, 2);
        int gho = GENERATE(1, 2);
        int mgh = m + 2 * ghm;
        int ngh = n + 2 * ghn;
        int ogh = o + 2 * gho;

        mgcl::VaryingStencil3x3x3 a(m, n, o, ghm, ghn, gho);
        mgcl::VaryingStencil3x3x3 b(m, n, o, ghm, ghn, gho);

        // fill a and b
        for (int i = 0; i < mgh; i++)
            for (int j = 0; j < ngh; j++)
                for (int k = 0; k < ogh; k++)
                {
                    // 7-point Laplace
                    a[i][j][k][0][1][1] = 1;
                    a[i][j][k][1][0][1] = 1;
                    a[i][j][k][1][1][0] = 1;
                    a[i][j][k][1][1][1] = -6;
                    a[i][j][k][1][1][2] = 1;
                    a[i][j][k][1][2][1] = 1;
                    a[i][j][k][2][1][1] = 1;

                    // full-weight restriction, scaled by 64
                    b[i][j][k][0][0][0] = 1;
                    b[i][j][k][0][0][1] = 2;
                    b[i][j][k][0][0][2] = 1;
                    b[i][j][k][0][1][0] = 2;
                    b[i][j][k][0][1][1] = 4;
                    b[i][j][k][0][1][2] = 2;
                    b[i][j][k][0][2][0] = 1;
                    b[i][j][k][0][2][1] = 2;
                    b[i][j][k][0][2][2] = 1;
                    b[i][j][k][1][0][0] = 2;
                    b[i][j][k][1][0][1] = 4;
                    b[i][j][k][1][0][2] = 2;
                    b[i][j][k][1][1][0] = 4;
                    b[i][j][k][1][1][1] = 8;
                    b[i][j][k][1][1][2] = 4;
                    b[i][j][k][1][2][0] = 2;
                    b[i][j][k][1][2][1] = 4;
                    b[i][j][k][1][2][2] = 2;
                    b[i][j][k][2][0][0] = 1;
                    b[i][j][k][2][0][1] = 2;
                    b[i][j][k][2][0][2] = 1;
                    b[i][j][k][2][1][0] = 2;
                    b[i][j][k][2][1][1] = 4;
                    b[i][j][k][2][1][2] = 2;
                    b[i][j][k][2][2][0] = 1;
                    b[i][j][k][2][2][1] = 2;
                    b[i][j][k][2][2][2] = 1;
                }

        auto c = a.multiply(b);
        auto cstar = a * b;

        // check result against explicitly calculated matrix product
        auto c_expected = mgcl_test::Matrix2d::laplace7p3d(m, n, o) * mgcl_test::Matrix2d::restrictionFullWeight(m, n, o);
        auto c_m2d = mgcl_test::Matrix2d::fromVaryingStencil(c);
        auto cstar_m2d = mgcl_test::Matrix2d::fromVaryingStencil(cstar);

        REQUIRE(c_expected.getM() == c_m2d.getM());
        REQUIRE(c_expected.getN() == c_m2d.getN());
        REQUIRE(c_expected.getM() == cstar_m2d.getM());
        REQUIRE(c_expected.getN() == cstar_m2d.getN());

        CHECK(c_m2d == c_expected);
        CHECK(cstar_m2d == c_expected);
    }

    SECTION("different Ns")
    {
        int m = GENERATE(2, 3, 4);
        int n = GENERATE(2, 3, 4);
        int o = GENERATE(2, 3, 4);

        mgcl::VaryingStencil3x3x3 a(m, n, o, 0, 0, 0);
        mgcl::VaryingStencil3x3x3 b(m, n, o, 0, 0, 0);

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

                    // full-weight restriction, scaled by 64
                    b[i][j][k][0][0][0] = 1;
                    b[i][j][k][0][0][1] = 2;
                    b[i][j][k][0][0][2] = 1;
                    b[i][j][k][0][1][0] = 2;
                    b[i][j][k][0][1][1] = 4;
                    b[i][j][k][0][1][2] = 2;
                    b[i][j][k][0][2][0] = 1;
                    b[i][j][k][0][2][1] = 2;
                    b[i][j][k][0][2][2] = 1;
                    b[i][j][k][1][0][0] = 2;
                    b[i][j][k][1][0][1] = 4;
                    b[i][j][k][1][0][2] = 2;
                    b[i][j][k][1][1][0] = 4;
                    b[i][j][k][1][1][1] = 8;
                    b[i][j][k][1][1][2] = 4;
                    b[i][j][k][1][2][0] = 2;
                    b[i][j][k][1][2][1] = 4;
                    b[i][j][k][1][2][2] = 2;
                    b[i][j][k][2][0][0] = 1;
                    b[i][j][k][2][0][1] = 2;
                    b[i][j][k][2][0][2] = 1;
                    b[i][j][k][2][1][0] = 2;
                    b[i][j][k][2][1][1] = 4;
                    b[i][j][k][2][1][2] = 2;
                    b[i][j][k][2][2][0] = 1;
                    b[i][j][k][2][2][1] = 2;
                    b[i][j][k][2][2][2] = 1;
                }

        auto c = b * a * b;

        // check result against explicitly calculated matrix product
        auto c_expected = mgcl_test::Matrix2d::restrictionFullWeight(m, n, o) *
                          mgcl_test::Matrix2d::laplace7p3d(m, n, o) *
                          mgcl_test::Matrix2d::restrictionFullWeight(m, n, o);
        auto c_m2d = mgcl_test::Matrix2d::fromVaryingStencil(c);

        REQUIRE(c_expected.getM() == c_m2d.getM());
        REQUIRE(c_expected.getN() == c_m2d.getN());

        CHECK(c_m2d == c_expected);

        // check also associativity
        auto c_assoc = b * (a * b);

        // check result against explicitly calculated matrix product
        auto c_expected_assoc = mgcl_test::Matrix2d::restrictionFullWeight(m, n, o) *
                                (mgcl_test::Matrix2d::laplace7p3d(m, n, o) *
                                 mgcl_test::Matrix2d::restrictionFullWeight(m, n, o));
        auto c_m2d_assoc = mgcl_test::Matrix2d::fromVaryingStencil(c_assoc);

        REQUIRE(c_expected_assoc.getM() == c_m2d_assoc.getM());
        REQUIRE(c_expected_assoc.getN() == c_m2d_assoc.getN());

        CHECK(c_m2d_assoc == c_expected_assoc);
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
