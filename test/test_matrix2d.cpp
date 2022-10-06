#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <cstdio>
#include <fstream>

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

TEST_CASE("Matrix2d::eye")
{
    int n = 3;
    auto a = mgcl_test::Matrix2d::eye(n);

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
        {
            if (i == j)
                CHECK(a[i][j] == 1);
            else
                CHECK(a[i][j] == 0);
        }
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
