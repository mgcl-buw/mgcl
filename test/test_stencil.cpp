#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <memory>

#include "../src/cuboid.hpp"
#include "../src/multigrid_engine.hpp"
#include "../src/stencil.hpp"

#include "matrix2d.hpp"

TEST_CASE("StencilLaplace7p periodic")
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

    mgcl::MultigridEngine::residualSeq(f, v, r, mgcl::MGCL_L2, mgcl::MGCL_LAPLACE_7POINT, expectedFactor, nullptr, 0, true);

    double expected = expectedFactor * (6.0 * 8 - 4 - 16 - 4 - 2 - 1 - 2);
    REQUIRE(-r[1][1][1] == Catch::Approx(expected));
}

TEST_CASE("StencilLaplace19p periodic")
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

    mgcl::MultigridEngine::residualSeq(f, v, r, mgcl::MGCL_L2, mgcl::MGCL_LAPLACE_19POINT, expectedFactor, nullptr, 0, true);

    double expected = expectedFactor * (24.0 * 8 - 2.0 * (4 + 16 + 4 + 2 + 1 + 2) - 8 - 4 - 1 - 2 - 16 - 32 - 8 - 4 - 4 - 4 - 2 - 1);
    REQUIRE(-r[1][1][1] == Catch::Approx(expected));
}

TEST_CASE("StencilLaplace27p periodic")
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

    mgcl::MultigridEngine::residualSeq(f, v, r, mgcl::MGCL_L2, mgcl::MGCL_LAPLACE_27POINT, expectedFactor, nullptr, 0, true);

    double expected = expectedFactor * (128.0 * 8 - 14.0 * (4 + 16 + 4 + 2 + 1 + 2) -
                                        3.0 * (8 + 4 + 1 + 2 + 16 + 32 + 8 + 4 + 4 + 4 + 2 + 1) -
                                        8 - 4 - 1 - 2 - 16 - 4 - 8 - 2);
    REQUIRE(-r[1][1][1] == Catch::Approx(expected));
}

TEST_CASE("StencilVarying7p periodic")
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

    mgcl::MultigridEngine::residualSeq(f, v, r, mgcl::MGCL_L2, mgcl::MGCL_VARYING, 1, &vals, 0, true);

    double expected = h2inv * (6.0 * 8 - 4 - 16 - 4 - 2 - 1 - 2);
    REQUIRE(-r[1][1][1] == Catch::Approx(expected));
}

TEST_CASE("StencilVarying19p periodic")
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

    mgcl::MultigridEngine::residualSeq(f, v, r, mgcl::MGCL_L2, mgcl::MGCL_VARYING, 1, &vals, 0, true);

    double expected = h2inv * (24.0 * 8 - 2.0 * (4 + 16 + 4 + 2 + 1 + 2) - 8 - 4 - 1 - 2 - 16 - 32 - 8 - 4 - 4 - 4 - 2 - 1);
    REQUIRE(-r[1][1][1] == Catch::Approx(expected));
}

TEST_CASE("StencilVarying27p periodic")
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

    mgcl::MultigridEngine::residualSeq(f, v, r, mgcl::MGCL_L2, mgcl::MGCL_VARYING, 1, &vals, 0, true);

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

TEST_CASE("VaryingStencil::updateGhosts")
{
    SECTION("m,n,o >= gh")
    {
        // regular ghost update
        int m = GENERATE(2, 3, 4);
        int n = GENERATE(2, 3, 4);
        int o = GENERATE(2, 3, 4);

        int gh = GENERATE(1, 2);

        SECTION("N = 3")
        {
            mgcl::VaryingStencil3x3x3 a(m, n, o, gh, gh, gh);
            a.fillRandomInt();
            a.updateGhosts();

            // check in z-direction
            for (int i = 0; i < gh; i++)
                for (int j = 0; j < n + 2 * gh; j++)
                    for (int k = 0; k < o + 2 * gh; k++)
                        for (int ii = 0; ii < 3; ii++)
                            for (int jj = 0; jj < 3; jj++)
                                for (int kk = 0; kk < 3; kk++)
                                {
                                    CHECK(a[i][j][k][ii][jj][kk] == a[i + m][j][k][ii][jj][kk]);
                                    CHECK(a[i + gh][j][k][ii][jj][kk] == a[i + gh + m][j][k][ii][jj][kk]);
                                }

            // check in y-direction
            for (int i = 0; i < m + 2 * gh; i++)
                for (int j = 0; j < gh; j++)
                    for (int k = 0; k < o + 2 * gh; k++)
                        for (int ii = 0; ii < 3; ii++)
                            for (int jj = 0; jj < 3; jj++)
                                for (int kk = 0; kk < 3; kk++)
                                {
                                    CHECK(a[i][j][k][ii][jj][kk] == a[i][j + n][k][ii][jj][kk]);
                                    CHECK(a[i][j + gh][k][ii][jj][kk] == a[i][j + gh + n][k][ii][jj][kk]);
                                }

            // check in x-direction
            for (int i = 0; i < m + 2 * gh; i++)
                for (int j = 0; j < n + 2 * gh; j++)
                    for (int k = 0; k < gh; k++)
                        for (int ii = 0; ii < 3; ii++)
                            for (int jj = 0; jj < 3; jj++)
                                for (int kk = 0; kk < 3; kk++)
                                {
                                    CHECK(a[i][j][k][ii][jj][kk] == a[i][j][k + o][ii][jj][kk]);
                                    CHECK(a[i][j][k + gh][ii][jj][kk] == a[i][j][k + gh + o][ii][jj][kk]);
                                }
        }

        SECTION("N = 5")
        {
            mgcl::VaryingStencil5x5x5 a(m, n, o, gh, gh, gh);
            a.fillRandomInt();
            a.updateGhosts();

            // check in z-direction
            for (int i = 0; i < gh; i++)
                for (int j = 0; j < n + 2 * gh; j++)
                    for (int k = 0; k < o + 2 * gh; k++)
                        for (int ii = 0; ii < 5; ii++)
                            for (int jj = 0; jj < 5; jj++)
                                for (int kk = 0; kk < 5; kk++)
                                {
                                    CHECK(a[i][j][k][ii][jj][kk] == a[i + m][j][k][ii][jj][kk]);
                                    CHECK(a[i + gh][j][k][ii][jj][kk] == a[i + gh + m][j][k][ii][jj][kk]);
                                }

            // check in y-direction
            for (int i = 0; i < m + 2 * gh; i++)
                for (int j = 0; j < gh; j++)
                    for (int k = 0; k < o + 2 * gh; k++)
                        for (int ii = 0; ii < 5; ii++)
                            for (int jj = 0; jj < 5; jj++)
                                for (int kk = 0; kk < 5; kk++)
                                {
                                    CHECK(a[i][j][k][ii][jj][kk] == a[i][j + n][k][ii][jj][kk]);
                                    CHECK(a[i][j + gh][k][ii][jj][kk] == a[i][j + gh + n][k][ii][jj][kk]);
                                }

            // check in x-direction
            for (int i = 0; i < m + 2 * gh; i++)
                for (int j = 0; j < n + 2 * gh; j++)
                    for (int k = 0; k < gh; k++)
                        for (int ii = 0; ii < 5; ii++)
                            for (int jj = 0; jj < 5; jj++)
                                for (int kk = 0; kk < 5; kk++)
                                {
                                    CHECK(a[i][j][k][ii][jj][kk] == a[i][j][k + o][ii][jj][kk]);
                                    CHECK(a[i][j][k + gh][ii][jj][kk] == a[i][j][k + gh + o][ii][jj][kk]);
                                }
        }
    }

    SECTION("m,n,o < gh")
    {
        // periodic ghost update, i.e. real grid is repeated (partially) more than once in ghost cells
        int m = GENERATE(1, 2, 3);
        int n = GENERATE(1, 2, 3);
        int o = GENERATE(1, 2, 3);

        int gh = GENERATE(4, 5);

        SECTION("N = 3")
        {
            mgcl::VaryingStencil3x3x3 a(m, n, o, gh, gh, gh);
            a.fillRandomInt();
            a.updateGhosts();

            // check in z-direction
            for (int i = 0; i < gh; i++)
                for (int j = 0; j < n + 2 * gh; j++)
                    for (int k = 0; k < o + 2 * gh; k++)
                        for (int ii = 0; ii < 3; ii++)
                            for (int jj = 0; jj < 3; jj++)
                                for (int kk = 0; kk < 3; kk++)
                                {
                                    CHECK(a[i][j][k][ii][jj][kk] == a[i + m][j][k][ii][jj][kk]);
                                    CHECK(a[i + gh][j][k][ii][jj][kk] == a[i + gh + m][j][k][ii][jj][kk]);
                                }

            // check in y-direction
            for (int i = 0; i < m + 2 * gh; i++)
                for (int j = 0; j < gh; j++)
                    for (int k = 0; k < o + 2 * gh; k++)
                        for (int ii = 0; ii < 3; ii++)
                            for (int jj = 0; jj < 3; jj++)
                                for (int kk = 0; kk < 3; kk++)
                                {
                                    CHECK(a[i][j][k][ii][jj][kk] == a[i][j + n][k][ii][jj][kk]);
                                    CHECK(a[i][j + gh][k][ii][jj][kk] == a[i][j + gh + n][k][ii][jj][kk]);
                                }

            // check in x-direction
            for (int i = 0; i < m + 2 * gh; i++)
                for (int j = 0; j < n + 2 * gh; j++)
                    for (int k = 0; k < gh; k++)
                        for (int ii = 0; ii < 3; ii++)
                            for (int jj = 0; jj < 3; jj++)
                                for (int kk = 0; kk < 3; kk++)
                                {
                                    CHECK(a[i][j][k][ii][jj][kk] == a[i][j][k + o][ii][jj][kk]);
                                    CHECK(a[i][j][k + gh][ii][jj][kk] == a[i][j][k + gh + o][ii][jj][kk]);
                                }
        }

        SECTION("N = 5")
        {
            mgcl::VaryingStencil5x5x5 a(m, n, o, gh, gh, gh);
            a.fillRandomInt();
            a.updateGhosts();

            // check in z-direction
            for (int i = 0; i < gh; i++)
                for (int j = 0; j < n + 2 * gh; j++)
                    for (int k = 0; k < o + 2 * gh; k++)
                        for (int ii = 0; ii < 5; ii++)
                            for (int jj = 0; jj < 5; jj++)
                                for (int kk = 0; kk < 5; kk++)
                                {
                                    CHECK(a[i][j][k][ii][jj][kk] == a[i + m][j][k][ii][jj][kk]);
                                    CHECK(a[i + gh][j][k][ii][jj][kk] == a[i + gh + m][j][k][ii][jj][kk]);
                                }

            // check in y-direction
            for (int i = 0; i < m + 2 * gh; i++)
                for (int j = 0; j < gh; j++)
                    for (int k = 0; k < o + 2 * gh; k++)
                        for (int ii = 0; ii < 5; ii++)
                            for (int jj = 0; jj < 5; jj++)
                                for (int kk = 0; kk < 5; kk++)
                                {
                                    CHECK(a[i][j][k][ii][jj][kk] == a[i][j + n][k][ii][jj][kk]);
                                    CHECK(a[i][j + gh][k][ii][jj][kk] == a[i][j + gh + n][k][ii][jj][kk]);
                                }

            // check in x-direction
            for (int i = 0; i < m + 2 * gh; i++)
                for (int j = 0; j < n + 2 * gh; j++)
                    for (int k = 0; k < gh; k++)
                        for (int ii = 0; ii < 5; ii++)
                            for (int jj = 0; jj < 5; jj++)
                                for (int kk = 0; kk < 5; kk++)
                                {
                                    CHECK(a[i][j][k][ii][jj][kk] == a[i][j][k + o][ii][jj][kk]);
                                    CHECK(a[i][j][k + gh][ii][jj][kk] == a[i][j][k + gh + o][ii][jj][kk]);
                                }
        }
    }

    SECTION("mixed")
    {
        // mixed periodic and regular ghost update, e.g. m > gh but n < gh
        int m = GENERATE(2, 3);
        int n = GENERATE(2, 3);
        int o = GENERATE(2, 3);

        int ghm = GENERATE(1, 3, 5);
        int ghn = GENERATE(1, 3, 5);
        int gho = GENERATE(1, 3, 5);

        SECTION("N = 3")
        {
            mgcl::VaryingStencil3x3x3 a(m, n, o, ghm, ghn, gho);
            a.fillRandomInt();
            a.updateGhosts();

            // check in z-direction
            for (int i = 0; i < ghm; i++)
                for (int j = 0; j < n + 2 * ghn; j++)
                    for (int k = 0; k < o + 2 * gho; k++)
                        for (int ii = 0; ii < 3; ii++)
                            for (int jj = 0; jj < 3; jj++)
                                for (int kk = 0; kk < 3; kk++)
                                {
                                    CHECK(a[i][j][k][ii][jj][kk] == a[i + m][j][k][ii][jj][kk]);
                                    CHECK(a[i + ghm][j][k][ii][jj][kk] == a[i + ghm + m][j][k][ii][jj][kk]);
                                }

            // check in y-direction
            for (int i = 0; i < m + 2 * ghm; i++)
                for (int j = 0; j < ghn; j++)
                    for (int k = 0; k < o + 2 * gho; k++)
                        for (int ii = 0; ii < 3; ii++)
                            for (int jj = 0; jj < 3; jj++)
                                for (int kk = 0; kk < 3; kk++)
                                {
                                    CHECK(a[i][j][k][ii][jj][kk] == a[i][j + n][k][ii][jj][kk]);
                                    CHECK(a[i][j + ghn][k][ii][jj][kk] == a[i][j + ghn + n][k][ii][jj][kk]);
                                }

            // check in x-direction
            for (int i = 0; i < m + 2 * ghm; i++)
                for (int j = 0; j < n + 2 * ghn; j++)
                    for (int k = 0; k < gho; k++)
                        for (int ii = 0; ii < 3; ii++)
                            for (int jj = 0; jj < 3; jj++)
                                for (int kk = 0; kk < 3; kk++)
                                {
                                    CHECK(a[i][j][k][ii][jj][kk] == a[i][j][k + o][ii][jj][kk]);
                                    CHECK(a[i][j][k + gho][ii][jj][kk] == a[i][j][k + gho + o][ii][jj][kk]);
                                }
        }

        SECTION("N = 5")
        {
            mgcl::VaryingStencil5x5x5 a(m, n, o, ghm, ghn, gho);
            a.fillRandomInt();
            a.updateGhosts();

            // check in z-direction
            for (int i = 0; i < ghm; i++)
                for (int j = 0; j < n + 2 * ghn; j++)
                    for (int k = 0; k < o + 2 * gho; k++)
                        for (int ii = 0; ii < 5; ii++)
                            for (int jj = 0; jj < 5; jj++)
                                for (int kk = 0; kk < 5; kk++)
                                {
                                    CHECK(a[i][j][k][ii][jj][kk] == a[i + m][j][k][ii][jj][kk]);
                                    CHECK(a[i + ghm][j][k][ii][jj][kk] == a[i + ghm + m][j][k][ii][jj][kk]);
                                }

            // check in y-direction
            for (int i = 0; i < m + 2 * ghm; i++)
                for (int j = 0; j < ghn; j++)
                    for (int k = 0; k < o + 2 * gho; k++)
                        for (int ii = 0; ii < 5; ii++)
                            for (int jj = 0; jj < 5; jj++)
                                for (int kk = 0; kk < 5; kk++)
                                {
                                    CHECK(a[i][j][k][ii][jj][kk] == a[i][j + n][k][ii][jj][kk]);
                                    CHECK(a[i][j + ghn][k][ii][jj][kk] == a[i][j + ghn + n][k][ii][jj][kk]);
                                }

            // check in x-direction
            for (int i = 0; i < m + 2 * ghm; i++)
                for (int j = 0; j < n + 2 * ghn; j++)
                    for (int k = 0; k < gho; k++)
                        for (int ii = 0; ii < 5; ii++)
                            for (int jj = 0; jj < 5; jj++)
                                for (int kk = 0; kk < 5; kk++)
                                {
                                    CHECK(a[i][j][k][ii][jj][kk] == a[i][j][k + o][ii][jj][kk]);
                                    CHECK(a[i][j][k + gho][ii][jj][kk] == a[i][j][k + gho + o][ii][jj][kk]);
                                }
        }
    }
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

        // check result against explicitly calculated matrix product
        auto c_expected = mgcl_test::Matrix2d::laplace7p3d(m, n, o, false) * mgcl_test::Matrix2d::fullWeightNonCut(m, n, o, false);
        auto c_m2d = mgcl_test::Matrix2d::fromVaryingStencil(c, false);

        REQUIRE(c_expected.getM() == c_m2d.getM());
        REQUIRE(c_expected.getN() == c_m2d.getN());

        CHECK(c_m2d == c_expected);
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

        // check result against explicitly calculated matrix product
        auto c_expected = mgcl_test::Matrix2d::laplace7p3d(m, n, o) * mgcl_test::Matrix2d::fullWeightNonCut(m, n, o);
        auto c_m2d = mgcl_test::Matrix2d::fromVaryingStencil(c, true);

        REQUIRE(c_expected.getM() == c_m2d.getM());
        REQUIRE(c_expected.getN() == c_m2d.getN());

        CHECK(c_m2d == c_expected);
    }

    SECTION("different Ns, random values")
    {
        int m = GENERATE(2, 3, 4);
        int n = GENERATE(2, 3, 4);
        int o = GENERATE(2, 3, 4);

        SECTION("widths 3 * {3,5,7}")
        {
            int ghb = GENERATE(1, 2);
            int gha = GENERATE(0, 1, 2);

            mgcl::VaryingStencil3x3x3 b3p(m, n, o, ghb, ghb, ghb);
            mgcl::VaryingStencil5x5x5 b5p(m, n, o, ghb, ghb, ghb);
            mgcl::VaryingStencil7x7x7 b7p(m, n, o, ghb, ghb, ghb);
            mgcl::VaryingStencil3x3x3 b3np(m, n, o, ghb, ghb, ghb);
            mgcl::VaryingStencil5x5x5 b5np(m, n, o, ghb, ghb, ghb);
            mgcl::VaryingStencil7x7x7 b7np(m, n, o, ghb, ghb, ghb);

            b3p.fillRandomInt(1, 9);
            b5p.fillRandomInt(1, 9);
            b7p.fillRandomInt(1, 9);
            // fill only real cells for non-periodic stencils, ghosts are 0.
            b3np.fillRandomInt(1, 9, true);
            b5np.fillRandomInt(1, 9, true);
            b7np.fillRandomInt(1, 9, true);

            // update ghosts only for periodic stencils
            b3p.updateGhosts();
            b5p.updateGhosts();
            b7p.updateGhosts();

            // build matrices from stencils to check results
            auto mb3p = mgcl_test::Matrix2d::fromVaryingStencil(b3p, true);
            auto mb5p = mgcl_test::Matrix2d::fromVaryingStencil(b5p, true);
            auto mb7p = mgcl_test::Matrix2d::fromVaryingStencil(b7p, true);
            auto mb3np = mgcl_test::Matrix2d::fromVaryingStencil(b3np, false);
            auto mb5np = mgcl_test::Matrix2d::fromVaryingStencil(b5np, false);
            auto mb7np = mgcl_test::Matrix2d::fromVaryingStencil(b7np, false);

            mgcl::VaryingStencil3x3x3 a3(m, n, o, gha, gha, gha);
            a3.fillRandomInt(1, 9);

            // build matrices from stencils to check results later
            auto ma3p = mgcl_test::Matrix2d::fromVaryingStencil(a3, true);
            auto ma3np = mgcl_test::Matrix2d::fromVaryingStencil(a3, false);

            {
                auto c33p = a3.multiply(b3p);
                auto c33np = a3.multiply(b3np);
                auto c33_expected_p = ma3p * mb3p;
                auto c33_expected_np = ma3np * mb3np;
                auto mc33p = mgcl_test::Matrix2d::fromVaryingStencil(c33p, true);
                auto mc33np = mgcl_test::Matrix2d::fromVaryingStencil(c33np, false);
                REQUIRE(mc33p == c33_expected_p);
                REQUIRE(mc33np == c33_expected_np);
            }

            {
                auto c35p = a3.multiply(b5p);
                auto c35np = a3.multiply(b5np);
                auto c35_expected_p = ma3p * mb5p;
                auto c35_expected_np = ma3np * mb5np;
                auto mc35p = mgcl_test::Matrix2d::fromVaryingStencil(c35p, true);
                auto mc35np = mgcl_test::Matrix2d::fromVaryingStencil(c35np, false);
                REQUIRE(mc35p == c35_expected_p);
                REQUIRE(mc35np == c35_expected_np);
            }

            {
                auto c37p = a3.multiply(b7p);
                auto c37np = a3.multiply(b7np);
                auto c37_expected_p = ma3p * mb7p;
                auto c37_expected_np = ma3np * mb7np;
                auto mc37p = mgcl_test::Matrix2d::fromVaryingStencil(c37p, true);
                auto mc37np = mgcl_test::Matrix2d::fromVaryingStencil(c37np, false);
                REQUIRE(mc37p == c37_expected_p);
                REQUIRE(mc37np == c37_expected_np);
            }
        }

        SECTION("widths 5 * {3,5,7}")
        {
            mgcl::VaryingStencil3x3x3 b3p(m, n, o, 2, 2, 2);
            mgcl::VaryingStencil5x5x5 b5p(m, n, o, 2, 2, 2);
            mgcl::VaryingStencil7x7x7 b7p(m, n, o, 2, 2, 2);
            mgcl::VaryingStencil3x3x3 b3np(m, n, o, 2, 2, 2);
            mgcl::VaryingStencil5x5x5 b5np(m, n, o, 2, 2, 2);
            mgcl::VaryingStencil7x7x7 b7np(m, n, o, 2, 2, 2);

            b3p.fillRandomInt(1, 9);
            b5p.fillRandomInt(1, 9);
            b7p.fillRandomInt(1, 9);
            // fill only real cells for non-periodic stencils, ghosts are 0.
            b3np.fillRandomInt(1, 9, true);
            b5np.fillRandomInt(1, 9, true);
            b7np.fillRandomInt(1, 9, true);

            // update ghosts only for periodic stencils
            b3p.updateGhosts();
            b5p.updateGhosts();
            b7p.updateGhosts();

            // build matrices from stencils to check results
            auto mb3p = mgcl_test::Matrix2d::fromVaryingStencil(b3p, true);
            auto mb5p = mgcl_test::Matrix2d::fromVaryingStencil(b5p, true);
            auto mb7p = mgcl_test::Matrix2d::fromVaryingStencil(b7p, true);
            auto mb3np = mgcl_test::Matrix2d::fromVaryingStencil(b3np, false);
            auto mb5np = mgcl_test::Matrix2d::fromVaryingStencil(b5np, false);
            auto mb7np = mgcl_test::Matrix2d::fromVaryingStencil(b7np, false);

            mgcl::VaryingStencil5x5x5 a5(m, n, o, 0, 0, 0);
            a5.fillRandomInt(1, 9);

            // build matrices from stencils to check results later
            auto ma5p = mgcl_test::Matrix2d::fromVaryingStencil(a5, true);
            auto ma5np = mgcl_test::Matrix2d::fromVaryingStencil(a5, false);

            {
                auto c53p = a5.multiply(b3p);
                auto c53np = a5.multiply(b3np);
                auto c53_expected_p = ma5p * mb3p;
                auto c53_expected_np = ma5np * mb3np;
                auto mc53p = mgcl_test::Matrix2d::fromVaryingStencil(c53p, true);
                auto mc53np = mgcl_test::Matrix2d::fromVaryingStencil(c53np, false);
                REQUIRE(mc53p == c53_expected_p);
                REQUIRE(mc53np == c53_expected_np);
            }

            {
                auto c55p = a5.multiply(b5p);
                auto c55np = a5.multiply(b5np);
                auto c55_expected_p = ma5p * mb5p;
                auto c55_expected_np = ma5np * mb5np;
                auto mc55p = mgcl_test::Matrix2d::fromVaryingStencil(c55p, true);
                auto mc55np = mgcl_test::Matrix2d::fromVaryingStencil(c55np, false);
                REQUIRE(mc55p == c55_expected_p);
                REQUIRE(mc55np == c55_expected_np);
            }

            {
                auto c57p = a5.multiply(b7p);
                auto c57np = a5.multiply(b7np);
                auto c57_expected_p = ma5p * mb7p;
                auto c57_expected_np = ma5np * mb7np;
                auto mc57p = mgcl_test::Matrix2d::fromVaryingStencil(c57p, true);
                auto mc57np = mgcl_test::Matrix2d::fromVaryingStencil(c57np, false);
                REQUIRE(mc57p == c57_expected_p);
                REQUIRE(mc57np == c57_expected_np);
            }
        }

        SECTION("widths 7 * {3,5,7}")
        {
            mgcl::VaryingStencil3x3x3 b3p(m, n, o, 3, 3, 3);
            mgcl::VaryingStencil5x5x5 b5p(m, n, o, 3, 3, 3);
            mgcl::VaryingStencil7x7x7 b7p(m, n, o, 3, 3, 3);
            mgcl::VaryingStencil3x3x3 b3np(m, n, o, 3, 3, 3);
            mgcl::VaryingStencil5x5x5 b5np(m, n, o, 3, 3, 3);
            mgcl::VaryingStencil7x7x7 b7np(m, n, o, 3, 3, 3);

            b3p.fillRandomInt(1, 9);
            b5p.fillRandomInt(1, 9);
            b7p.fillRandomInt(1, 9);
            // fill only real cells for non-periodic stencils, ghosts are 0.
            b3np.fillRandomInt(1, 9, true);
            b5np.fillRandomInt(1, 9, true);
            b7np.fillRandomInt(1, 9, true);

            // update ghosts only for periodic stencils
            b3p.updateGhosts();
            b5p.updateGhosts();
            b7p.updateGhosts();

            // build matrices from stencils to check results
            auto mb3p = mgcl_test::Matrix2d::fromVaryingStencil(b3p, true);
            auto mb5p = mgcl_test::Matrix2d::fromVaryingStencil(b5p, true);
            auto mb7p = mgcl_test::Matrix2d::fromVaryingStencil(b7p, true);
            auto mb3np = mgcl_test::Matrix2d::fromVaryingStencil(b3np, false);
            auto mb5np = mgcl_test::Matrix2d::fromVaryingStencil(b5np, false);
            auto mb7np = mgcl_test::Matrix2d::fromVaryingStencil(b7np, false);

            mgcl::VaryingStencil7x7x7 a7(m, n, o, 0, 0, 0);
            a7.fillRandomInt(1, 9);

            // build matrices from stencils to check results later
            auto ma7p = mgcl_test::Matrix2d::fromVaryingStencil(a7, true);
            auto ma7np = mgcl_test::Matrix2d::fromVaryingStencil(a7, false);

            {
                auto c73p = a7.multiply(b3p);
                auto c73np = a7.multiply(b3np);
                auto c73_expected_p = ma7p * mb3p;
                auto c73_expected_np = ma7np * mb3np;
                auto mc73p = mgcl_test::Matrix2d::fromVaryingStencil(c73p, true);
                auto mc73np = mgcl_test::Matrix2d::fromVaryingStencil(c73np, false);
                REQUIRE(mc73p == c73_expected_p);
                REQUIRE(mc73np == c73_expected_np);
            }

            if (m > 3 && n > 3 && o > 3)
            {
                auto c75p = a7.multiply(b5p);
                auto c75np = a7.multiply(b5np);
                auto c75_expected_p = ma7p * mb5p;
                auto c75_expected_np = ma7np * mb5np;
                auto mc75p = mgcl_test::Matrix2d::fromVaryingStencil(c75p, true);
                auto mc75np = mgcl_test::Matrix2d::fromVaryingStencil(c75np, false);
                REQUIRE(mc75p == c75_expected_p);
                REQUIRE(mc75np == c75_expected_np);
            }

            {
                auto c77p = a7.multiply(b7p);
                auto c77np = a7.multiply(b7np);
                auto c77_expected_p = ma7p * mb7p;
                auto c77_expected_np = ma7np * mb7np;
                auto mc77p = mgcl_test::Matrix2d::fromVaryingStencil(c77p, true);
                auto mc77np = mgcl_test::Matrix2d::fromVaryingStencil(c77np, false);
                REQUIRE(mc77p == c77_expected_p);
                REQUIRE(mc77np == c77_expected_np);
            }
        }
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

        bool dimsNotEqual = a.getDim1() != b.getDim1() || a.getDim2() != b.getDim2() || a.getDim3() != b.getDim3();
        bool ghostsNotBigEnough = b.getGhostsDim1() < 1 || b.getGhostsDim2() < 1 || b.getGhostsDim3() < 1;
        bool ghostsNotEqual = b.getGhostsDim1() != b.getGhostsDim2() || b.getGhostsDim1() != b.getGhostsDim3();

        if (dimsNotEqual || ghostsNotEqual || ghostsNotBigEnough)
            REQUIRE_THROWS(a.multiply(b));
        else
            REQUIRE_NOTHROW(a.multiply(b));
    }
}

TEST_CASE("VaryingStencil::multiply(FixedStencil)")
{
    int m = GENERATE(2, 3, 4);
    int n = GENERATE(2, 3, 4);
    int o = GENERATE(2, 3, 4);

    mgcl::FixedStencil3x3x3 f;
    f.fillRandom(0, 10);

    // create varying stencil from fixed
    mgcl::VaryingStencil3x3x3 vf(m, n, o, 1, 1, 1);
    // clang-format off
    for (int i = 0; i < vf.getDim1(); i++)
    for (int j = 0; j < vf.getDim2(); j++)
    for (int k = 0; k < vf.getDim3(); k++)
        for (int ii = 0; ii < vf.getDim4(); ii++)
        for (int jj = 0; jj < vf.getDim5(); jj++)
        for (int kk = 0; kk < vf.getDim6(); kk++)
        {
            vf[i][j][k][ii][jj][kk] = f[ii][jj][kk];
        }
    // clang-format on

    // create random lhs varying stencil
    mgcl::VaryingStencil3x3x3 vr(m, n, o, 0, 0, 0);

    auto fres = vr.multiply(f, 2);
    auto vres = vr.multiply(vf);

    // Check dimensions
    REQUIRE(fres.getDim1() == vres.getDim1());
    REQUIRE(fres.getDim2() == vres.getDim2());
    REQUIRE(fres.getDim3() == vres.getDim3());
    REQUIRE(fres.getDim4() == vres.getDim4());
    REQUIRE(fres.getDim5() == vres.getDim5());
    REQUIRE(fres.getDim6() == vres.getDim6());
    REQUIRE(fres.getGhostsDim1() == vres.getGhostsDim1());
    REQUIRE(fres.getGhostsDim2() == vres.getGhostsDim2());
    REQUIRE(fres.getGhostsDim3() == vres.getGhostsDim3());
    REQUIRE(fres.getGhostsDim4() == vres.getGhostsDim4());
    REQUIRE(fres.getGhostsDim5() == vres.getGhostsDim5());
    REQUIRE(fres.getGhostsDim6() == vres.getGhostsDim6());

    // check results

    // clang-format off
    for (int i = 0; i < vf.getDim1(); i++)
    for (int j = 0; j < vf.getDim2(); j++)
    for (int k = 0; k < vf.getDim3(); k++)
        for (int ii = 0; ii < vf.getDim4(); ii++)
        for (int jj = 0; jj < vf.getDim5(); jj++)
        for (int kk = 0; kk < vf.getDim6(); kk++)
        {
            REQUIRE(fres[i][j][k][ii][jj][kk] == vres[i][j][k][ii][jj][kk]);
        }
    // clang-format on
}

TEST_CASE("FixedStencil::multiply")
{
    int m = GENERATE(2, 3, 4);
    int n = GENERATE(2, 3, 4);
    int o = GENERATE(2, 3, 4);

    mgcl::FixedStencil3x3x3 f;
    f.fillRandom(0, 10);

    // create varying stencil from fixed
    mgcl::VaryingStencil3x3x3 vf(m, n, o, 0, 0, 0);
    // clang-format off
    for (int i = 0; i < vf.getDim1(); i++)
    for (int j = 0; j < vf.getDim2(); j++)
    for (int k = 0; k < vf.getDim3(); k++)
        for (int ii = 0; ii < vf.getDim4(); ii++)
        for (int jj = 0; jj < vf.getDim5(); jj++)
        for (int kk = 0; kk < vf.getDim6(); kk++)
        {
            vf[i][j][k][ii][jj][kk] = f[ii][jj][kk];
        }
    // clang-format on

    // create random rhs varying stencil
    mgcl::VaryingStencil3x3x3 vr(m, n, o, 1, 1, 1);

    auto fres = f.multiply(vr, 2);
    auto vres = vf.multiply(vr);

    // Check dimensions
    REQUIRE(fres.getDim1() == vres.getDim1());
    REQUIRE(fres.getDim2() == vres.getDim2());
    REQUIRE(fres.getDim3() == vres.getDim3());
    REQUIRE(fres.getDim4() == vres.getDim4());
    REQUIRE(fres.getDim5() == vres.getDim5());
    REQUIRE(fres.getDim6() == vres.getDim6());
    REQUIRE(fres.getGhostsDim1() == vres.getGhostsDim1());
    REQUIRE(fres.getGhostsDim2() == vres.getGhostsDim2());
    REQUIRE(fres.getGhostsDim3() == vres.getGhostsDim3());
    REQUIRE(fres.getGhostsDim4() == vres.getGhostsDim4());
    REQUIRE(fres.getGhostsDim5() == vres.getGhostsDim5());
    REQUIRE(fres.getGhostsDim6() == vres.getGhostsDim6());

    // check results

    // clang-format off
    for (int i = 0; i < vf.getDim1(); i++)
    for (int j = 0; j < vf.getDim2(); j++)
    for (int k = 0; k < vf.getDim3(); k++)
        for (int ii = 0; ii < vf.getDim4(); ii++)
        for (int jj = 0; jj < vf.getDim5(); jj++)
        for (int kk = 0; kk < vf.getDim6(); kk++)
        {
            REQUIRE(fres[i][j][k][ii][jj][kk] == vres[i][j][k][ii][jj][kk]);
        }
    // clang-format on
}

TEST_CASE("FixedStencil::create3dFullWeightRestriction")
{
    auto r = mgcl::create3dFullWeightRestrictionStencil();

    double factor = 1.0 / 64.0;

    // full-weight restriction, scaled by 64
    REQUIRE(r[0][0][0] == 1 * factor);
    REQUIRE(r[0][0][1] == 2 * factor);
    REQUIRE(r[0][0][2] == 1 * factor);
    REQUIRE(r[0][1][0] == 2 * factor);
    REQUIRE(r[0][1][1] == 4 * factor);
    REQUIRE(r[0][1][2] == 2 * factor);
    REQUIRE(r[0][2][0] == 1 * factor);
    REQUIRE(r[0][2][1] == 2 * factor);
    REQUIRE(r[0][2][2] == 1 * factor);
    REQUIRE(r[1][0][0] == 2 * factor);
    REQUIRE(r[1][0][1] == 4 * factor);
    REQUIRE(r[1][0][2] == 2 * factor);
    REQUIRE(r[1][1][0] == 4 * factor);
    REQUIRE(r[1][1][1] == 8 * factor);
    REQUIRE(r[1][1][2] == 4 * factor);
    REQUIRE(r[1][2][0] == 2 * factor);
    REQUIRE(r[1][2][1] == 4 * factor);
    REQUIRE(r[1][2][2] == 2 * factor);
    REQUIRE(r[2][0][0] == 1 * factor);
    REQUIRE(r[2][0][1] == 2 * factor);
    REQUIRE(r[2][0][2] == 1 * factor);
    REQUIRE(r[2][1][0] == 2 * factor);
    REQUIRE(r[2][1][1] == 4 * factor);
    REQUIRE(r[2][1][2] == 2 * factor);
    REQUIRE(r[2][2][0] == 1 * factor);
    REQUIRE(r[2][2][1] == 2 * factor);
    REQUIRE(r[2][2][2] == 1 * factor);
}

TEST_CASE("FixedStencil::create3dBilinearProlongationStencil")
{
    auto r = mgcl::create3dBilinearProlongationStencil();

    double factor = 1.0 / 8.0;

    // full-weight restriction, scaled by 64
    REQUIRE(r[0][0][0] == 1 * factor);
    REQUIRE(r[0][0][1] == 2 * factor);
    REQUIRE(r[0][0][2] == 1 * factor);
    REQUIRE(r[0][1][0] == 2 * factor);
    REQUIRE(r[0][1][1] == 4 * factor);
    REQUIRE(r[0][1][2] == 2 * factor);
    REQUIRE(r[0][2][0] == 1 * factor);
    REQUIRE(r[0][2][1] == 2 * factor);
    REQUIRE(r[0][2][2] == 1 * factor);
    REQUIRE(r[1][0][0] == 2 * factor);
    REQUIRE(r[1][0][1] == 4 * factor);
    REQUIRE(r[1][0][2] == 2 * factor);
    REQUIRE(r[1][1][0] == 4 * factor);
    REQUIRE(r[1][1][1] == 8 * factor);
    REQUIRE(r[1][1][2] == 4 * factor);
    REQUIRE(r[1][2][0] == 2 * factor);
    REQUIRE(r[1][2][1] == 4 * factor);
    REQUIRE(r[1][2][2] == 2 * factor);
    REQUIRE(r[2][0][0] == 1 * factor);
    REQUIRE(r[2][0][1] == 2 * factor);
    REQUIRE(r[2][0][2] == 1 * factor);
    REQUIRE(r[2][1][0] == 2 * factor);
    REQUIRE(r[2][1][1] == 4 * factor);
    REQUIRE(r[2][1][2] == 2 * factor);
    REQUIRE(r[2][2][0] == 1 * factor);
    REQUIRE(r[2][2][1] == 2 * factor);
    REQUIRE(r[2][2][2] == 1 * factor);
}
