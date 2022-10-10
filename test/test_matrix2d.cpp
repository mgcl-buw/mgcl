#include "matrix2d.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>

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

TEST_CASE("Matrix2d::fromVaryingStencil", "[.]")
{
    int m = GENERATE(1, 2, 3);
    int n = GENERATE(1, 2, 3);
    int o = GENERATE(1, 2, 3);
    int gh = 1;

    auto a = mgcl_test::Matrix2d::laplace7p3d(m, n, o);
    mgcl::VaryingStencil s(m, n, o, gh, gh, gh);
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

    auto c = mgcl_test::Matrix2d::fromVaryingStencil(s);
    CHECK(a == c);
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
    SECTION("check valid result")
    {
        int m = 3;
        int n = 3;
        int o = 3;

        auto a = mgcl_test::Matrix2d::laplace7p3d(m, n, o);

        REQUIRE(a.getM() == m * n * o);
        REQUIRE(a.getN() == m * n * o);

        mgcl_test::Matrix2d a_check({{-6, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                     {1, -6, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                     {0, 1, -6, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                     {1, 0, 0, -6, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                     {0, 1, 0, 1, -6, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                     {0, 0, 1, 0, 1, -6, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                     {0, 0, 0, 1, 0, 0, -6, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                     {0, 0, 0, 0, 1, 0, 1, -6, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                     {0, 0, 0, 0, 0, 1, 0, 1, -6, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                     {1, 0, 0, 0, 0, 0, 0, 0, 0, -6, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
                                     {0, 1, 0, 0, 0, 0, 0, 0, 0, 1, -6, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
                                     {0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, -6, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
                                     {0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, -6, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
                                     {0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, -6, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
                                     {0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, -6, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0},
                                     {0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, -6, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
                                     {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, -6, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0},
                                     {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, -6, 0, 0, 0, 0, 0, 0, 0, 0, 1},
                                     {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, -6, 1, 0, 1, 0, 0, 0, 0, 0},
                                     {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, -6, 1, 0, 1, 0, 0, 0, 0},
                                     {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, -6, 0, 0, 1, 0, 0, 0},
                                     {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, -6, 1, 0, 1, 0, 0},
                                     {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, -6, 1, 0, 1, 0},
                                     {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, -6, 0, 0, 1},
                                     {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, -6, 1, 0},
                                     {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, -6, 1},
                                     {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, -6}});

        CHECK(a == a_check);
    }

    SECTION("check no sigsegv")
    {
        int m = GENERATE(1, 2, 3);
        int n = GENERATE(1, 2, 3);
        int o = GENERATE(1, 2, 3);

        auto a = mgcl_test::Matrix2d::laplace7p3d(m, n, o);
        REQUIRE(a.getM() == m * n * o);
        REQUIRE(a.getN() == m * n * o);
    }
}

TEST_CASE("Matrix2d::restrictionFullWeight")
{
    int m = 3;
    int n = 3;
    int o = 3;

    auto a = mgcl_test::Matrix2d::restrictionFullWeight(m, n, o);

    REQUIRE(a.getM() == m * n * o);
    REQUIRE(a.getN() == m * n * o);

    mgcl_test::Matrix2d a_check({{8, 4, 0, 4, 2, 0, 0, 0, 0, 4, 2, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                 {4, 8, 4, 2, 4, 2, 0, 0, 0, 2, 4, 2, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                 {0, 4, 8, 0, 2, 4, 0, 0, 0, 0, 2, 4, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                 {4, 2, 0, 8, 4, 0, 4, 2, 0, 2, 1, 0, 4, 2, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                 {2, 4, 2, 4, 8, 4, 2, 4, 2, 1, 2, 1, 2, 4, 2, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                 {0, 2, 4, 0, 4, 8, 0, 2, 4, 0, 1, 2, 0, 2, 4, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                 {0, 0, 0, 4, 2, 0, 8, 4, 0, 0, 0, 0, 2, 1, 0, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                 {0, 0, 0, 2, 4, 2, 4, 8, 4, 0, 0, 0, 1, 2, 1, 2, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                 {0, 0, 0, 0, 2, 4, 0, 4, 8, 0, 0, 0, 0, 1, 2, 0, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                 {4, 2, 0, 2, 1, 0, 0, 0, 0, 8, 4, 0, 4, 2, 0, 0, 0, 0, 4, 2, 0, 2, 1, 0, 0, 0, 0},
                                 {2, 4, 2, 1, 2, 1, 0, 0, 0, 4, 8, 4, 2, 4, 2, 0, 0, 0, 2, 4, 2, 1, 2, 1, 0, 0, 0},
                                 {0, 2, 4, 0, 1, 2, 0, 0, 0, 0, 4, 8, 0, 2, 4, 0, 0, 0, 0, 2, 4, 0, 1, 2, 0, 0, 0},
                                 {2, 1, 0, 4, 2, 0, 2, 1, 0, 4, 2, 0, 8, 4, 0, 4, 2, 0, 2, 1, 0, 4, 2, 0, 2, 1, 0},
                                 {1, 2, 1, 2, 4, 2, 1, 2, 1, 2, 4, 2, 4, 8, 4, 2, 4, 2, 1, 2, 1, 2, 4, 2, 1, 2, 1},
                                 {0, 1, 2, 0, 2, 4, 0, 1, 2, 0, 2, 4, 0, 4, 8, 0, 2, 4, 0, 1, 2, 0, 2, 4, 0, 1, 2},
                                 {0, 0, 0, 2, 1, 0, 4, 2, 0, 0, 0, 0, 4, 2, 0, 8, 4, 0, 0, 0, 0, 2, 1, 0, 4, 2, 0},
                                 {0, 0, 0, 1, 2, 1, 2, 4, 2, 0, 0, 0, 2, 4, 2, 4, 8, 4, 0, 0, 0, 1, 2, 1, 2, 4, 2},
                                 {0, 0, 0, 0, 1, 2, 0, 2, 4, 0, 0, 0, 0, 2, 4, 0, 4, 8, 0, 0, 0, 0, 1, 2, 0, 2, 4},
                                 {0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 2, 0, 2, 1, 0, 0, 0, 0, 8, 4, 0, 4, 2, 0, 0, 0, 0},
                                 {0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 2, 1, 2, 1, 0, 0, 0, 4, 8, 4, 2, 4, 2, 0, 0, 0},
                                 {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 0, 1, 2, 0, 0, 0, 0, 4, 8, 0, 2, 4, 0, 0, 0},
                                 {0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 4, 2, 0, 2, 1, 0, 4, 2, 0, 8, 4, 0, 4, 2, 0},
                                 {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 2, 4, 2, 1, 2, 1, 2, 4, 2, 4, 8, 4, 2, 4, 2},
                                 {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 2, 4, 0, 1, 2, 0, 2, 4, 0, 4, 8, 0, 2, 4},
                                 {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 4, 2, 0, 0, 0, 0, 4, 2, 0, 8, 4, 0},
                                 {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 2, 4, 2, 0, 0, 0, 2, 4, 2, 4, 8, 4},
                                 {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 2, 4, 0, 0, 0, 0, 2, 4, 0, 4, 8}});

    CHECK(a == a_check);
}
