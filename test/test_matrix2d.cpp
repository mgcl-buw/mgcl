#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>

#include "matrix2d.hpp"

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

TEST_CASE("Matrix2d::operator+=")
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

TEST_CASE("Matrix2d 3d-Laplace")
{
    using M2d = mgcl_test::Matrix2d;

    int m = 3;
    int n = 3;
    int o = 3;

    std::vector<std::tuple<double, int>> vals{{-2, 0}, {1, -1}, {1, 1}};
    auto Dxx = mgcl_test::Matrix2d::diag(vals, m, m);
    auto Dyy = mgcl_test::Matrix2d::diag(vals, n, n);
    auto Dzz = mgcl_test::Matrix2d::diag(vals, o, o);

    auto a = Dzz.kronecker(M2d::eye(n)).kronecker(M2d::eye(m)) +
             M2d::eye(o).kronecker(Dyy).kronecker(M2d::eye(n)) +
             M2d::eye(o).kronecker(M2d::eye(n)).kronecker(Dxx);

    REQUIRE(a.getM() == m * n * o);
    REQUIRE(a.getN() == m * n * o);

    M2d a_check({{-6, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
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
