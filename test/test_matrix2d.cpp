#include "matrix2d.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>

#include "test_results.hpp"

TEST_CASE("Matrix2d::constructor")
{
    mgcl_test::Matrix2d a(3, 2);
    REQUIRE(a.getM() == 3);
    REQUIRE(a.getN() == 2);
    REQUIRE_NOTHROW(a.at(0).at(0));
    REQUIRE_NOTHROW(a.at(0).at(1));
    REQUIRE_NOTHROW(a.at(1).at(0));
    REQUIRE_NOTHROW(a.at(1).at(1));
    REQUIRE_NOTHROW(a.at(2).at(0));
    REQUIRE_NOTHROW(a.at(2).at(1));
}

TEST_CASE("Matrix2d::kronecker")
{
    SECTION("correct result")
    {
        mgcl_test::Matrix2d a(1, 2);
        mgcl_test::Matrix2d b(2, 2);

        a[0][0] = 1;
        a[0][1] = 2;
        b[0][0] = 3;
        b[0][1] = 4;
        b[1][0] = 5;
        b[1][1] = 6;

        auto c = a.kronecker(b);

        REQUIRE(c.getM() == a.getM() * b.getM());
        REQUIRE(c.getN() == a.getN() * b.getN());

        CHECK(c[0][0] == a[0][0] * b[0][0]);
        CHECK(c[0][1] == a[0][0] * b[0][1]);
        CHECK(c[0][2] == a[0][1] * b[0][0]);
        CHECK(c[0][3] == a[0][1] * b[0][1]);
        CHECK(c[1][0] == a[0][0] * b[1][0]);
        CHECK(c[1][1] == a[0][0] * b[1][1]);
        CHECK(c[1][2] == a[0][1] * b[1][0]);
        CHECK(c[1][3] == a[0][1] * b[1][1]);
    }

    SECTION("no sigsegv")
    {
        int m1 = GENERATE(1, 2, 3, 4, 5);
        int n1 = GENERATE(1, 2, 3, 4, 5);
        int m2 = GENERATE(1, 2, 3, 4, 5);
        int n2 = GENERATE(1, 2, 3, 4, 5);

        mgcl_test::Matrix2d a(m1, n1);
        mgcl_test::Matrix2d b(m2, n2);

        auto c = a.kronecker(b);

        REQUIRE(c.getM() == a.getM() * b.getM());
        REQUIRE(c.getN() == a.getN() * b.getN());
    }
}

TEST_CASE("Matrix2d::eye(m,n)")
{
    int m = GENERATE(1, 2, 3, 4);
    int n = GENERATE(1, 2, 3, 4);
    auto a = mgcl_test::Matrix2d::eye(m, n);

    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
        {
            if (i == j)
                CHECK(a[i][j] == 1);
            else
                CHECK(a[i][j] == 0);
        }
}

TEST_CASE("Matrix2d::eye(m)")
{
    int m = GENERATE(1, 2, 3, 4);

    auto a = mgcl_test::Matrix2d::eye(m, m);
    auto b = mgcl_test::Matrix2d::eye(m);

    CHECK(a == b);
}

TEST_CASE("Matrix2d::operator+")
{
    SECTION("valid")
    {
        mgcl_test::Matrix2d a(2, 2);
        mgcl_test::Matrix2d b(2, 2);

        a[0][0] = 1;
        a[0][1] = 2;
        b[0][0] = 3;
        b[0][1] = 4;
        b[1][0] = 5;
        b[1][1] = 6;

        auto c = a + b;

        REQUIRE(c.getM() == a.getM());
        REQUIRE(c.getN() == a.getN());

        CHECK(c[0][0] == a[0][0] + b[0][0]);
        CHECK(c[0][1] == a[0][1] + b[0][1]);
        CHECK(c[1][0] == a[1][0] + b[1][0]);
        CHECK(c[1][1] == a[1][1] + b[1][1]);
    }

    SECTION("throwing")
    {
        int m1 = GENERATE(1, 2, 3);
        int n1 = GENERATE(1, 2, 3);
        int m2 = GENERATE(1, 2, 3);
        int n2 = GENERATE(1, 2, 3);

        if (m1 == m2 && n1 == n2)
            REQUIRE_NOTHROW(mgcl_test::Matrix2d(m1, n1) + mgcl_test::Matrix2d(m2, n2));
        else
            REQUIRE_THROWS(mgcl_test::Matrix2d(m1, n1) + mgcl_test::Matrix2d(m2, n2));
    }
}

TEST_CASE("Matrix2d::operator+=")
{
    SECTION("valid")
    {
        mgcl_test::Matrix2d a(2, 2);
        mgcl_test::Matrix2d b(2, 2);

        a[0][0] = 1;
        a[0][1] = 2;
        b[0][0] = 3;
        b[0][1] = 4;
        b[1][0] = 5;
        b[1][1] = 6;

        auto a_old = a;

        a += b;

        CHECK(a[0][0] == a_old[0][0] + b[0][0]);
        CHECK(a[0][1] == a_old[0][1] + b[0][1]);
        CHECK(a[1][0] == a_old[1][0] + b[1][0]);
        CHECK(a[1][1] == a_old[1][1] + b[1][1]);
    }

    SECTION("throwing")
    {
        int m1 = GENERATE(1, 2, 3);
        int n1 = GENERATE(1, 2, 3);
        int m2 = GENERATE(1, 2, 3);
        int n2 = GENERATE(1, 2, 3);

        if (m1 == m2 && n1 == n2)
            REQUIRE_NOTHROW(mgcl_test::Matrix2d(m1, n1) += mgcl_test::Matrix2d(m2, n2));
        else
            REQUIRE_THROWS(mgcl_test::Matrix2d(m1, n1) += mgcl_test::Matrix2d(m2, n2));
    }
}

TEST_CASE("Matrix2d::operator== and Matrix2d::operator!=")
{
    mgcl_test::Matrix2d a(2, 2);
    mgcl_test::Matrix2d b(2, 2);
    mgcl_test::Matrix2d c(1, 2);
    mgcl_test::Matrix2d d(2, 1);

    a[0][0] = 1;
    a[0][1] = 2;
    a[1][0] = 1;
    a[1][1] = 2;

    b[0][0] = 1;
    b[0][1] = 2;
    b[1][0] = 1;
    b[1][1] = 2;

    CHECK(a == b);
    CHECK(b == a);
    CHECK(a != c);
    CHECK(a != d);
    CHECK(c != a);
    CHECK(d != a);
}

TEST_CASE("Matrix2d::operator*")
{
    SECTION("m,n = 1,1")
    {
        mgcl_test::Matrix2d a(1, 1);
        mgcl_test::Matrix2d b(1, 1);

        a[0][0] = 3;
        b[0][0] = 2;

        auto c = a * b;

        REQUIRE(c.getM() == a.getM());
        REQUIRE(c.getN() == b.getN());

        CHECK(c[0][0] == 6);
    }

    SECTION("m,n = 2,2")
    {
        mgcl_test::Matrix2d a(2, 2);
        mgcl_test::Matrix2d b(2, 2);

        a[0][0] = 1;
        a[0][1] = 2;
        a[1][0] = 3;
        a[1][1] = 4;
        b[0][0] = 5;
        b[0][1] = 6;
        b[1][0] = 7;
        b[1][1] = 8;

        auto c = a * b;

        REQUIRE(c.getM() == a.getM());
        REQUIRE(c.getN() == b.getN());

        CHECK(c[0][0] == 19);
        CHECK(c[0][1] == 22);
        CHECK(c[1][0] == 43);
        CHECK(c[1][1] == 50);
    }

    SECTION("a m,n = 3,2; b m,n = 2,4")
    {
        mgcl_test::Matrix2d a(3, 2);
        mgcl_test::Matrix2d b(2, 4);

        a[0][0] = 1;
        a[0][1] = 2;
        a[1][0] = 3;
        a[1][1] = 4;
        a[2][0] = 5;
        a[2][1] = 6;
        b[0][0] = 7;
        b[0][1] = 8;
        b[0][2] = 9;
        b[0][3] = 10;
        b[1][0] = 11;
        b[1][1] = 12;
        b[1][2] = 13;
        b[1][3] = 14;

        auto c = a * b;

        REQUIRE(c.getM() == a.getM());
        REQUIRE(c.getN() == b.getN());

        CHECK(c[0][0] == 29);
        CHECK(c[0][1] == 32);
        CHECK(c[0][2] == 35);
        CHECK(c[0][3] == 38);
        CHECK(c[1][0] == 65);
        CHECK(c[1][1] == 72);
        CHECK(c[1][2] == 79);
        CHECK(c[1][3] == 86);
        CHECK(c[2][0] == 101);
        CHECK(c[2][1] == 112);
        CHECK(c[2][2] == 123);
        CHECK(c[2][3] == 134);
    }

    SECTION("a m,n = 2,3; b m,n = 3,1")
    {
        mgcl_test::Matrix2d a(2, 3);
        mgcl_test::Matrix2d b(3, 1);

        a[0][0] = 1;
        a[0][1] = 2;
        a[0][2] = 3;
        a[1][0] = 4;
        a[1][1] = 5;
        a[1][2] = 6;
        b[0][0] = 7;
        b[1][0] = 8;
        b[2][0] = 9;

        auto c = a * b;

        REQUIRE(c.getM() == a.getM());
        REQUIRE(c.getN() == b.getN());

        CHECK(c[0][0] == 50);
        CHECK(c[1][0] == 122);
    }

    SECTION("laplace3d * fullWeightRestriction")
    {
        auto a = mgcl_test::Matrix2d::laplace7p3d(3, 2, 1, false);
        auto b = 64 * mgcl_test::Matrix2d::restrictionFullWeight(3, 2, 1, false);

        auto c = a * b;

        REQUIRE(c.getM() == a.getM());
        REQUIRE(c.getN() == b.getN());

        mgcl_test::Matrix2d c_check({{-40, -14, -14, -4, 4, 2},
                                     {-14, -40, -4, -14, 2, 4},
                                     {-14, -4, -36, -12, -14, -4},
                                     {-4, -14, -12, -36, -4, -14},
                                     {4, 2, -14, -4, -40, -14},
                                     {2, 4, -4, -14, -14, -40}});
        CHECK(c == c_check);
    }

    SECTION("throws")
    {
        REQUIRE_THROWS(mgcl_test::Matrix2d(1, 1) * mgcl_test::Matrix2d(3, 1));
    }
}

TEST_CASE("Matrix2d::diag")
{
    int m = GENERATE(1, 2, 3);
    int n = GENERATE(1, 2, 3, 4, 5);

    std::vector<std::tuple<double, int>> vals{{4, 0}, {-1, -1}, {1, 1}};
    auto a = mgcl_test::Matrix2d::diag(vals, m, n);

    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
        {
            if (i == j)
                CHECK(a[i][j] == 4);
            else if (i == j - 1)
                CHECK(a[i][j] == 1);
            else if (i == j + 1)
                CHECK(a[i][j] == -1);
            else
                CHECK(a[i][j] == 0);
        }
}

TEST_CASE("Matrix2d::fromVaryingStencil")
{
    SECTION("laplace3d, not periodic")
    {
        int m = GENERATE(1, 2, 3);
        int n = GENERATE(1, 2, 3);
        int o = GENERATE(1, 2, 3);
        int gh = 0;

        auto a = mgcl_test::Matrix2d::laplace7p3d(m, n, o, false);
        mgcl::VaryingStencil3x3x3 s(m, n, o, gh, gh, gh);
        for (int i = 0; i < m + 2 * gh; i++)
            for (int j = 0; j < n + 2 * gh; j++)
                for (int k = 0; k < o + 2 * gh; k++)
                {
                    // 7-point Laplace
                    s[i][j][k][0][1][1] = 1;
                    s[i][j][k][1][0][1] = 1;
                    s[i][j][k][1][1][0] = 1;
                    s[i][j][k][1][1][1] = -6;
                    s[i][j][k][1][1][2] = 1;
                    s[i][j][k][1][2][1] = 1;
                    s[i][j][k][2][1][1] = 1;
                }

        auto c = mgcl_test::Matrix2d::fromVaryingStencil(s, false);
        CHECK(a == c);
    }

    SECTION("laplace3d, periodic")
    {
        int m = GENERATE(1, 2, 3);
        int n = GENERATE(1, 2, 3);
        int o = GENERATE(1, 2, 3);
        int gh = 1;

        auto a = mgcl_test::Matrix2d::laplace7p3d(m, n, o);
        mgcl::VaryingStencil3x3x3 s(m, n, o, gh, gh, gh);
        for (int i = 0; i < m + 2 * gh; i++)
            for (int j = 0; j < n + 2 * gh; j++)
                for (int k = 0; k < o + 2 * gh; k++)
                {
                    // 7-point Laplace
                    s[i][j][k][0][1][1] = 1;
                    s[i][j][k][1][0][1] = 1;
                    s[i][j][k][1][1][0] = 1;
                    s[i][j][k][1][1][1] = -6;
                    s[i][j][k][1][1][2] = 1;
                    s[i][j][k][1][2][1] = 1;
                    s[i][j][k][2][1][1] = 1;
                }

        auto c = mgcl_test::Matrix2d::fromVaryingStencil(s, true);
        CHECK(a == c);
    }

    SECTION("varying, not periodic")
    {
        auto a = mgcl_test::Matrix2d_fromVaryingStencil::matrix2d24x24RandomNotPeriodic();
        auto sptr = mgcl_test::Matrix2d_fromVaryingStencil::varyingStencil2x3x4RandomPeriodic();
        auto &s = *sptr;

        auto c = mgcl_test::Matrix2d::fromVaryingStencil(s, false);

        CHECK(a == c);
    }

    SECTION("varying, periodic")
    {
        auto a = mgcl_test::Matrix2d_fromVaryingStencil::matrix2d24x24RandomPeriodic();
        auto sptr = mgcl_test::Matrix2d_fromVaryingStencil::varyingStencil2x3x4RandomPeriodic();
        auto &s = *sptr;

        auto c = mgcl_test::Matrix2d::fromVaryingStencil(s, true);

        CHECK(a == c);
    }
}

TEST_CASE("Matrix2d::diag Laplace")
{
    int m = 3;
    int n = 3;

    std::vector<std::tuple<double, int>> vals{{-2, 0}, {1, -1}, {1, 1}};
    auto a = mgcl_test::Matrix2d::diag(vals, m, n);

    CHECK(a[0][0] == -2);
    CHECK(a[0][1] == 1);
    CHECK(a[0][2] == 0);
    CHECK(a[1][0] == 1);
    CHECK(a[1][1] == -2);
    CHECK(a[1][2] == 1);
    CHECK(a[2][0] == 0);
    CHECK(a[2][1] == 1);
    CHECK(a[2][2] == -2);
}

TEST_CASE("Matrix2d::laplace7p3d")
{
    int m = GENERATE(1, 2, 3, 4);
    int n = GENERATE(1, 2, 3, 4);
    int o = GENERATE(1, 2, 3, 4);

    SECTION("not periodic")
    {
        auto a = mgcl_test::Matrix2d::laplace7p3d(m, n, o, false);
        REQUIRE(a.getM() == m * n * o);
        REQUIRE(a.getN() == m * n * o);

        // clang-format off
        for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
        for (int k = 0; k < o; k++)
            for (int ii = 0; ii < 3; ii++)
            for (int jj = 0; jj < 3; jj++)
            for (int kk = 0; kk < 3; kk++)
                // check if current stencil entry maps to a real grid point
                if (i + ii >= 1 &&
                    i + ii <= m &&
                    j + jj >= 1 &&
                    j + jj <= n &&
                    k + kk >= 1 &&
                    k + kk <= o)
                {
                    // center of stencil
                    if (ii == 1 && jj == 1 && kk == 1)
                        CHECK(a[i * n * o + j * o + k][i * n * o + j * o + k + (ii - 1) * n * o + (jj - 1) * o + (kk - 1)] == -6.0);
                    // adjacent to center
                    else if (ii == 1 && jj == 1 && (kk == 0 || kk == 2) ||
                            ii == 1 && (jj == 0 || jj == 2) && kk == 1 ||
                            (ii == 0 || ii == 2) && jj == 1 && kk == 1)
                        CHECK(a[i * n * o + j * o + k][i * n * o + j * o + k + (ii - 1) * n * o + (jj - 1) * o + (kk - 1)] == 1.0);
                    else
                        CHECK(a[i * n * o + j * o + k][i * n * o + j * o + k + (ii - 1) * n * o + (jj - 1) * o + (kk - 1)] == 0.0);
                }
        // clang-format on
    }

    SECTION("periodic")
    {
        auto a = mgcl_test::Matrix2d::laplace7p3d(m, n, o);
        REQUIRE(a.getM() == m * n * o);
        REQUIRE(a.getN() == m * n * o);

        // stores the amount of stencil entries that map to a grid point. Tuple entries:
        // 0: Amount of center entries (should be 0 or 1)
        // 1: Amount of entries adjacent to center (should be 0 to 4)
        // 2: Amount of corner entries (should be 0 to 22)
        std::tuple<int, int, int> mappingsPerEntry[m * n * o][m * n * o];

        // init with 0
        for (int i = 0; i < m * n * o; i++)
            for (int j = 0; j < m * n * o; j++)
            {
                std::get<0>(mappingsPerEntry[i][j]) = 0;
                std::get<1>(mappingsPerEntry[i][j]) = 0;
                std::get<2>(mappingsPerEntry[i][j]) = 0;
            }

        // clang-format off
        for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
        for (int k = 0; k < o; k++)
            for (int ii = 0; ii < 3; ii++)
            for (int jj = 0; jj < 3; jj++)
            for (int kk = 0; kk < 3; kk++)
            {
                // taken from fromVaryingStencil
                int gpi = i + (ii - 1); // grid point index mapped to by stencil entry in x-direction
                int gpj = j + (jj - 1); // grid point index mapped to by stencil entry in y-direction
                int gpk = k + (kk - 1); // grid point index mapped to by stencil entry in z-direction

                // wrap around for periodic bc
                if (gpi < 0)
                    gpi += m;
                else if (gpi >= m)
                    gpi -= m;
                
                if (gpj < 0)
                    gpj += n;
                else if (gpj >= n)
                    gpj -= n;
                
                if (gpk < 0)
                    gpk += o;
                else if (gpk >= o)
                    gpk -= o;

                // center of stencil
                if (ii == 1 && jj == 1 && kk == 1)
                    std::get<0>(mappingsPerEntry[i * n * o + j * o + k][gpi * n * o + gpj * o + gpk])++;
                // adjacent to center
                else if (ii == 1 && jj == 1 && (kk == 0 || kk == 2) ||
                        ii == 1 && (jj == 0 || jj == 2) && kk == 1 ||
                        (ii == 0 || ii == 2) && jj == 1 && kk == 1)
                        std::get<1>(mappingsPerEntry[i * n * o + j * o + k][gpi * n * o + gpj * o + gpk])++;
                else
                    std::get<2>(mappingsPerEntry[i * n * o + j * o + k][gpi * n * o + gpj * o + gpk])++;
            }
        // clang-format on

        // Now check matrix entries, which should be sums of the mappings
        for (int i = 0; i < m * n * o; i++)
            for (int j = 0; j < m * n * o; j++)
            {
                double sum = std::get<0>(mappingsPerEntry[i][j]) * -6.0 + std::get<1>(mappingsPerEntry[i][j]) * 1.0;
                CHECK(a[i][j] == sum);
            }
    }
}

TEST_CASE("Matrix2d::restrictionFullWeight")
{
    int m = GENERATE(1, 2, 3, 4);
    int n = GENERATE(1, 2, 3, 4);
    int o = GENERATE(1, 2, 3, 4);

    SECTION("not periodic")
    {
        auto a = mgcl_test::Matrix2d::restrictionFullWeight(m, n, o, false);

        REQUIRE(a.getM() == m * n * o);
        REQUIRE(a.getN() == m * n * o);

        double factor1 = 8.0 / 64.0;
        double factor2 = 4.0 / 64.0;
        double factor3 = 2.0 / 64.0;
        double factor4 = 1.0 / 64.0;

        // clang-format off
        for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
        for (int k = 0; k < o; k++)
            for (int ii = 0; ii < 3; ii++)
            for (int jj = 0; jj < 3; jj++)
            for (int kk = 0; kk < 3; kk++)
                // check if current stencil entry maps to a real grid point
                if (i + ii >= 1 &&
                    i + ii <= m &&
                    j + jj >= 1 &&
                    j + jj <= n &&
                    k + kk >= 1 &&
                    k + kk <= o)
                {
                    // center of stencil
                    if (ii == 1 && jj == 1 && kk == 1)
                        CHECK(a[i * n * o + j * o + k][i * n * o + j * o + k + (ii - 1) * n * o + (jj - 1) * o + (kk - 1)] == factor1);
                    // adjacent to center
                    else if (ii == 1 && jj == 1 && kk != 1 ||
                            ii == 1 && jj != 1 && kk == 1 ||
                            ii != 1 && jj == 1 && kk == 1)
                        CHECK(a[i * n * o + j * o + k][i * n * o + j * o + k + (ii - 1) * n * o + (jj - 1) * o + (kk - 1)] == factor2);
                    // diagonally adjacent to center
                    else if (ii == 1 && jj != 1 && kk != 1 ||
                            ii != 1 && jj != 1 && kk == 1 ||
                            ii != 1 && jj == 1 && kk != 1)
                        CHECK(a[i * n * o + j * o + k][i * n * o + j * o + k + (ii - 1) * n * o + (jj - 1) * o + (kk - 1)] == factor3);
                    // corner of stencil
                    else if (ii != 1 && jj != 1 && kk != 1)
                        CHECK(a[i * n * o + j * o + k][i * n * o + j * o + k + (ii - 1) * n * o + (jj - 1) * o + (kk - 1)] == factor4);
                    else
                        CHECK(a[i * n * o + j * o + k][i * n * o + j * o + k + (ii - 1) * n * o + (jj - 1) * o + (kk - 1)] == 0.0);
                }
        // clang-format on
    }

    SECTION("periodic")
    {
        auto a = mgcl_test::Matrix2d::restrictionFullWeight(m, n, o);

        REQUIRE(a.getM() == m * n * o);
        REQUIRE(a.getN() == m * n * o);

        double factor1 = 8.0 / 64.0;
        double factor2 = 4.0 / 64.0;
        double factor3 = 2.0 / 64.0;
        double factor4 = 1.0 / 64.0;

        // stores the amount of stencil entries that map to a grid point. Tuple entries:
        // 0: Amount of center entries (should be 0 or 1)
        // 1: Amount of entries adjacent to center (should be 0 to 4)
        // 2: Amount of entries diagonally adjacent to center (should be 0 to 14)
        // 3: Amount of outer corner entries (should be 0 to 8)
        std::tuple<int, int, int, int> mappingsPerEntry[m * n * o][m * n * o];

        // init with 0
        for (int i = 0; i < m * n * o; i++)
            for (int j = 0; j < m * n * o; j++)
            {
                std::get<0>(mappingsPerEntry[i][j]) = 0;
                std::get<1>(mappingsPerEntry[i][j]) = 0;
                std::get<2>(mappingsPerEntry[i][j]) = 0;
                std::get<3>(mappingsPerEntry[i][j]) = 0;
            }

        // clang-format off
        for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
        for (int k = 0; k < o; k++)
            for (int ii = 0; ii < 3; ii++)
            for (int jj = 0; jj < 3; jj++)
            for (int kk = 0; kk < 3; kk++)
            {
                // taken from fromVaryingStencil
                int gpi = i + (ii - 1); // grid point index mapped to by stencil entry in x-direction
                int gpj = j + (jj - 1); // grid point index mapped to by stencil entry in y-direction
                int gpk = k + (kk - 1); // grid point index mapped to by stencil entry in z-direction

                // wrap around for periodic bc
                if (gpi < 0)
                    gpi += m;
                else if (gpi >= m)
                    gpi -= m;
                
                if (gpj < 0)
                    gpj += n;
                else if (gpj >= n)
                    gpj -= n;
                
                if (gpk < 0)
                    gpk += o;
                else if (gpk >= o)
                    gpk -= o;

                // center of stencil
                if (ii == 1 && jj == 1 && kk == 1)
                    std::get<0>(mappingsPerEntry[i * n * o + j * o + k][gpi * n * o + gpj * o + gpk])++;
                // adjacent to center
                else if (ii == 1 && jj == 1 && kk != 1 ||
                        ii == 1 && jj != 1 && kk == 1 ||
                        ii != 1 && jj == 1 && kk == 1)
                    std::get<1>(mappingsPerEntry[i * n * o + j * o + k][gpi * n * o + gpj * o + gpk])++;
                // diagonally adjacent to center
                else if (ii == 1 && jj != 1 && kk != 1 ||
                        ii != 1 && jj != 1 && kk == 1 ||
                        ii != 1 && jj == 1 && kk != 1)
                    std::get<2>(mappingsPerEntry[i * n * o + j * o + k][gpi * n * o + gpj * o + gpk])++;
                // corner of stencil
                else if (ii != 1 && jj != 1 && kk != 1)
                    std::get<3>(mappingsPerEntry[i * n * o + j * o + k][gpi * n * o + gpj * o + gpk])++;
            }
        // clang-format on

        // Now check matrix entries, which should be sums of the mappings
        for (int i = 0; i < m * n * o; i++)
            for (int j = 0; j < m * n * o; j++)
            {
                double sum = std::get<0>(mappingsPerEntry[i][j]) * factor1 +
                             std::get<1>(mappingsPerEntry[i][j]) * factor2 +
                             std::get<2>(mappingsPerEntry[i][j]) * factor3 +
                             std::get<3>(mappingsPerEntry[i][j]) * factor4;
                CHECK(a[i][j] == sum);
            }
    }
}

TEST_CASE("Matrix2d::cuttingMatrix1d")
{
    int m = GENERATE(1, 2, 3, 4);

    auto a = mgcl_test::Matrix2d::cuttingMatrix1d(m);

    REQUIRE(a.getM() == m);
    REQUIRE(a.getN() == m * 2);

    for (int i = 0; i < a.getM(); i++)
        for (int j = 0; j < a.getN(); j++)
        {
            if (j == (i + 1) * 2 - 1)
                CHECK(a[i][j] == 1);
            else
                CHECK(a[i][j] == 0);
        }
}

TEST_CASE("Matrix2d::cuttingMatrix3d")
{
    int m = GENERATE(1, 2, 3);
    int n = GENERATE(1, 2, 3);
    int o = GENERATE(1, 2, 3);

    auto a = mgcl_test::Matrix2d::cuttingMatrix3d(m, n, o);

    REQUIRE(a.getM() == m * n * o);
    REQUIRE(a.getN() == m * n * o * 2 * 2 * 2);

    int jstart = n * o * 2 * 2 + o * 2 + 1;

    for (int i = 0; i < a.getM(); i++)
    {
        // find 1d index of current coarse grid point, that must equal the current column
        int idx1d = jstart + (i % o) * 2 + (i / o) * 2 * 2 * o + (i / (n * o)) * 2 * 2 * n * o;
        for (int j = 0; j < a.getN(); j++)
        {
            if (j == idx1d)
                CHECK(a[i][j] == 1);
            else
                CHECK(a[i][j] == 0);
        }
    }
}

TEST_CASE("Matrix2d::transposed")
{
    int m = GENERATE(1, 2, 3);
    int n = GENERATE(1, 2, 3);

    auto a = mgcl_test::Matrix2d(m, n);
    auto b = a.transposed();

    REQUIRE(a.getM() == m);
    REQUIRE(a.getN() == n);
    REQUIRE(b.getM() == n);
    REQUIRE(b.getN() == m);

    for (int i = 0; i < a.getM(); i++)
        for (int j = 0; j < a.getN(); j++)
        {
            CHECK(a[i][j] == b[j][i]);
        }
}
