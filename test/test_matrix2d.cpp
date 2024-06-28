#include "matrix2d.hpp"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <ostream>

#include "test_results.hpp"

TEST_CASE("tmp_GalerkinOptimization")
{
    int m = 4;
    int n = 4;
    int o = 4;
    int N = m * n * o;

    std::vector<std::tuple<double, int>> vals{{1, 0}, {0.5, -1}, {0.5, 1}};
    auto SP = mgcl_test::Matrix2d::diag(vals, N, N);
    auto A = mgcl_test::Matrix2d::laplace7p3d(4, 4, 4, true);
    auto SR = mgcl_test::Matrix2d::diag(vals, N, N);

    auto A_SP = A * SP;
    auto SR_A_SP = SR * A_SP;

    // SP.dumpToFileDecimalFormat("SP.csv", 2);
    // A.dumpToFileDecimalFormat("A.csv", 2);
    // A_SP.dumpToFileDecimalFormat("A_SP.csv", 2);
    // SR.dumpToFileDecimalFormat("SR.csv", 2);
    // SR_A_SP.dumpToFileDecimalFormat("SR_A_SP.csv", 2);

    // cutting only in the 3rd dimension
    auto K = mgcl_test::Matrix2d::cuttingMatrix3d(m, n, o / 2); // TODO create only for 3rd dimension
    std::cout << "K: " << K.getM() << " x " << K.getN() << std::endl;
    auto R = mgcl_test::Matrix2d::cuttingMatrix3d(m, n, o / 2) * SR;
    // auto P = SP * mgcl_test::Matrix2d::cuttingMatrix3d(m, n, o / 2).transposed();
    // auto RAP = R * A * P;
    // R.dumpToFileDecimalFormat("R.csv", 2);
    // P.dumpToFileDecimalFormat("P.csv", 2);
    // RAP.dumpToFileDecimalFormat("RAP.csv", 2);
}

TEST_CASE("MatrixGalerkinTrace")
{

    int m = 8;
    int n = 8;
    int o = 8;
    int N = m * n * o;
    double h2inv = static_cast<double>(N * N);
    bool periodic = true;

    std::cout << "mxnxo = " << m << "x" << n << "x" << o << std::endl;

    // Create 27p stencil and generate matrix from it
    mgcl::VaryingStencil* vst = new mgcl::VaryingStencil(m, n, o, 3, 0, 0, 0);
    vst->fill1dIndex(false);
    (*vst)[0][0][0][0][0][0] = 0.5; // fill with non zero value
    auto ah = mgcl_test::Matrix2d::fromVaryingStencil(*vst, periodic);
    delete vst;

    // auto ah = mgcl_test::Matrix2d::laplace7p3d(m, n, o) * h2inv;
    auto r = mgcl_test::Matrix2d::restrictionFullWeight(m, n, o);
    auto p = mgcl_test::Matrix2d::prolongationBilinear(m, n, o);
    auto r_ah = r * ah;
    auto r_ah_p = r_ah * p;

    std::cout << "ah rows x cols = " << ah.getM() << "x" << ah.getN() << std::endl;
    std::cout << "r rows x cols = " << r.getM() << "x" << r.getN() << std::endl;
    std::cout << "p rows x cols = " << p.getM() << "x" << p.getN() << std::endl;
    std::cout << "r_ah rows x cols = " << r_ah.getM() << "x" << r_ah.getN() << std::endl;
    std::cout << "r_ah_p rows x cols = " << r_ah_p.getM() << "x" << r_ah_p.getN() << std::endl;

    /* r.dumpToFileWithIndices("r.csv"); */
    /* ah.dumpToFileWithIndices("ah.csv"); */
    /* r_ah.dumpToFileWithIndices("r_ah.csv"); */
    /* p.dumpToFileWithIndices("p.csv"); */
    /* r_ah_p.dumpToFileWithIndices("r_ah_p.csv"); */

    using Trace = std::vector<std::vector<std::string>>;
    // Non-zero trace
    Trace trace_nnz(N, std::vector<std::string>(N));
    Trace trace_nnz_tmp(N, std::vector<std::string>(N));
    Trace stencil_trace_nnz(N, std::vector<std::string>(N));
    Trace stencil_trace_nnz_tmp(N, std::vector<std::string>(N));

    int stencilWidth = 3;
    int stencilWidth2 = stencilWidth >> 1;
    int m2 = m >> 1;
    int n2 = n >> 1;
    int o2 = o >> 1;

    // r * a_h
    for (int ci = 0; ci < r_ah.getM(); ci++)
        for (int cj = 0; cj < r_ah.getN(); cj++)
        {
            std::stringstream ss;
            ss << "(";
            std::stringstream ssst;
            ssst << "(";

            bool nnzfound = false;
            for (int idx = 0; idx < r.getN(); idx++)
            {
                // if (ci == 0 && cj == 0)
                //     std::cout << "r_ah[" << ci << "][" << cj << "] += "
                //               << "r[" << ci << "][" << idx << "] * a[" << idx << "][" << cj << "] = " << r[ci][idx] << " * " << ah[idx][cj] << std::endl;
                if (r[ci][idx] != 0 && ah[idx][cj] != 0)
                {
                    ss << (!nnzfound ? "" : " + ") << "r[" << ci << "][" << idx << "] * a[" << idx << "][" << cj << "]";

                    auto ind_ah = ah.getStencilIndicesForEntry(idx, cj, m, n, o, stencilWidth, periodic);
                    auto ind_r = r.getStencilIndicesForEntry(ci, idx, m, n, o, stencilWidth, periodic);

                    // Indices must exist since matrix entry is not zero
                    CAPTURE(ah.getM(), ah.getN(), r.getM(), r.getN(), ci, cj, idx, ind_ah, ind_r);
                    REQUIRE(ind_ah[0] != -1);
                    REQUIRE(ind_r[0] != -1);

                    ssst << (!nnzfound ? "" : " + ") << "r[" << ind_r[3] << "][" << ind_r[4] << "][" << ind_r[5] << "] * a["
                         << ind_ah[3] << "][" << ind_ah[4] << "][" << ind_ah[5] << "][" << ind_ah[0] << "][" << ind_ah[1] << "][" << ind_ah[2] << "]";

                    nnzfound = true;
                }
            }
            ss << ")";
            ssst << ")";

            trace_nnz_tmp[ci][cj] = ss.str();
            stencil_trace_nnz_tmp[ci][cj] = ssst.str();
        }

    // // print trace for r*a
    // {
    //     int nnz = 0;
    //     for (int ci = 0; ci < r_ah.getM(); ci++)
    //     {
    //         for (int cj = 0; cj < r_ah.getN(); cj++)
    //         {
    //             // check that only one non-zero entries are in the trace
    //             CAPTURE(ci, cj, r_ah[ci][cj], trace_nnz[ci][cj]);
    //             CAPTURE(r[0][0], ah[0][0], r[0][1], ah[1][0]);
    //             REQUIRE(((r_ah[ci][cj] == 0 && trace_nnz[ci][cj] == "()") || (trace_nnz[ci][cj] != "()")));
    //             if (r_ah[ci][cj] != 0)
    //             {
    //                 nnz++;
    //                 std::cout << "ra[" + std::to_string(ci) + "][" + std::to_string(cj) + "] = " << trace_nnz_tmp[ci][cj] << std::endl;
    //             }
    //         }
    //     }

    //     // print stencil trace for r*a
    //     std::cout << std::endl;
    //     for (int ci = 0; ci < r_ah.getM(); ci++)
    //     {
    //         for (int cj = 0; cj < r_ah.getN(); cj++)
    //         {
    //             // check that only one non-zero entries are in the trace
    //             if (r_ah[ci][cj] != 0)
    //             {
    //                 nnz++;
    //                 std::cout << "ra[" + std::to_string(ci) + "][" + std::to_string(cj) + "] = " << stencil_trace_nnz_tmp[ci][cj] << std::endl
    //                           << std::endl;
    //             }
    //         }
    //     }
    // }
    // return;

    // (r * a_h) * p
    for (int ci = 0; ci < r_ah_p.getM(); ci++)
        for (int cj = 0; cj < r_ah_p.getN(); cj++)
        {
            std::stringstream ss;
            ss << "(";
            std::stringstream ssst;
            ssst << "(";

            bool nnzfound = false;
            for (int idx = 0; idx < r_ah.getN(); idx++)
                if (r_ah[ci][idx] != 0 && p[idx][cj] != 0)
                {
                    ss << (!nnzfound ? "" : "\n + ") << trace_nnz_tmp[ci][idx] << " * p[" << idx << "][" << cj << "]";

                    // auto ind_r_ah = r_ah.getStencilIndicesForEntry(idx, cj, m, n, o, stencilWidth, periodic);
                    auto ind_p = p.getStencilIndicesForEntry(idx, cj, m, n, o, stencilWidth, periodic);

                    // Indices must exist since matrix entry is not zero
                    CAPTURE(ah.getM(), ah.getN(), r.getM(), r.getN(), ci, cj, idx, ind_p);
                    // CAPTURE(ah.getM(), ah.getN(), r.getM(), r.getN(), ci, cj, idx, ind_r_ah, ind_p);
                    // REQUIRE(ind_r_ah[0] != -1);
                    REQUIRE(ind_p[0] != -1);

                    ssst << (!nnzfound ? "" : " + ") << stencil_trace_nnz_tmp[ci][idx] << "] * p["
                         << ind_p[3] << "][" << ind_p[4] << "][" << ind_p[5] << "]";

                    nnzfound = true;
                }

            ss << ")";
            ssst << ")";

            trace_nnz[ci][cj] = ss.str();
            stencil_trace_nnz[ci][cj] = ssst.str();

            CAPTURE(ci, cj, r_ah_p[ci][cj], trace_nnz[ci][cj]);
            REQUIRE((r_ah_p[ci][cj] == 0 || (r_ah_p[ci][cj] != 0 && trace_nnz[ci][cj].find("()") == std::string::npos)));
        }

    std::cout << "==========" << std::endl;
    std::cout << "Non-Zero Trace:" << std::endl;
    std::cout << "RAP rows x cols = " << r_ah_p.getM() << "x" << r_ah_p.getN() << std::endl;

    // print trace
    int nnz = 0;
    for (int ci = 0; ci < r_ah_p.getM(); ci++)
    {
        for (int cj = 0; cj < r_ah_p.getN(); cj++)
            if (r_ah_p[ci][cj] != 0)
            {
                nnz++;
                std::cout << "rap[" + std::to_string(ci) + "][" + std::to_string(cj) + "] = " << trace_nnz[ci][cj] << std::endl
                          << std::endl;
            }
    }
    std::cout << "non-zero values:" << nnz << std::endl;

    // print stencil trace
    std::cout << std::endl;
    for (int ci = 0; ci < r_ah_p.getM(); ci++)
    {
        for (int cj = 0; cj < r_ah_p.getN(); cj++)
        {
            // check that only one non-zero entries are in the trace
            if (r_ah_p[ci][cj] != 0)
            {
                nnz++;
                std::cout << "rap[" + std::to_string(ci) + "][" + std::to_string(cj) + "] = " << stencil_trace_nnz[ci][cj] << std::endl
                          << std::endl;
            }
        }
    }
}

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

TEST_CASE("Matrix2d::fromVaryingStencil")
{
    SECTION("laplace3d, not periodic")
    {
        int m = GENERATE(1, 2, 3);
        int n = GENERATE(1, 2, 3);
        int o = GENERATE(1, 2, 3);
        int gh = 0;

        auto a = mgcl_test::Matrix2d::laplace7p3d(m, n, o, false);
        mgcl::VaryingStencil s(m, n, o, 3, gh, gh, gh);
        for (int i = 0; i < m + 2 * gh; i++)
            for (int j = 0; j < n + 2 * gh; j++)
                for (int k = 0; k < o + 2 * gh; k++)
                {
                    // 7-point Laplace
                    s[0][1][1][i][j][k] = 1;
                    s[1][0][1][i][j][k] = 1;
                    s[1][1][0][i][j][k] = 1;
                    s[1][1][1][i][j][k] = -6;
                    s[1][1][2][i][j][k] = 1;
                    s[1][2][1][i][j][k] = 1;
                    s[2][1][1][i][j][k] = 1;
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
        mgcl::VaryingStencil s(m, n, o, 3, gh, gh, gh);
        for (int i = 0; i < m + 2 * gh; i++)
            for (int j = 0; j < n + 2 * gh; j++)
                for (int k = 0; k < o + 2 * gh; k++)
                {
                    // 7-point Laplace
                    s[0][1][1][i][j][k] = 1;
                    s[1][0][1][i][j][k] = 1;
                    s[1][1][0][i][j][k] = 1;
                    s[1][1][1][i][j][k] = -6;
                    s[1][1][2][i][j][k] = 1;
                    s[1][2][1][i][j][k] = 1;
                    s[2][1][1][i][j][k] = 1;
                }

        auto c = mgcl_test::Matrix2d::fromVaryingStencil(s, true);
        CHECK(a == c);
    }

    SECTION("varying, not periodic")
    {
        auto a = mgcl_test::Matrix2d_fromVaryingStencil::matrix2d24x24RandomNotPeriodic();
        auto sptr = mgcl_test::Matrix2d_fromVaryingStencil::varyingStencil2x3x4RandomPeriodic();
        auto& s = *sptr;

        auto c = mgcl_test::Matrix2d::fromVaryingStencil(s, false);

        CHECK(a == c);
    }

    SECTION("varying, periodic")
    {
        auto a = mgcl_test::Matrix2d_fromVaryingStencil::matrix2d24x24RandomPeriodic();
        auto sptr = mgcl_test::Matrix2d_fromVaryingStencil::varyingStencil2x3x4RandomPeriodic();
        auto& s = *sptr;

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
    int m = GENERATE(3, 4);
    int n = GENERATE(1, 3);
    int o = GENERATE(1, 2);

    SECTION("not periodic")
    {
        int m2 = 2 * m;
        int n2 = 2 * n;
        int o2 = 2 * o;
        auto a = mgcl_test::Matrix2d::restrictionFullWeight(m2, n2, o2, false);

        REQUIRE(a.getM() == m * n * o);
        REQUIRE(a.getN() == m2 * n2 * o2);

        double factor1 = 8.0 / 64.0;
        double factor2 = 4.0 / 64.0;
        double factor3 = 2.0 / 64.0;
        double factor4 = 1.0 / 64.0;

        // clang-format off
        for (int i = 0, i2 = 1; i < m; i++, i2 += 2)
        for (int j = 0, j2 = 1; j < n; j++, j2 += 2)
        for (int k = 0, k2 = 1; k < o; k++, k2 += 2)
            for (int ii = 0; ii < 3; ii++)
            for (int jj = 0; jj < 3; jj++)
            for (int kk = 0; kk < 3; kk++)
                // check if current stencil entry maps to a real grid point
                if (i2 + ii >= 1 &&
                    i2 + ii <= m2 &&
                    j2 + jj >= 1 &&
                    j2 + jj <= n2 &&
                    k2 + kk >= 1 &&
                    k2 + kk <= o2)
                {
                    int row = i * n * o + j * o + k;
                    int col = i2 * n2 * o2 + j2 * o2 + k2 + (ii - 1) * n2 * o2 + (jj - 1) * o2 + (kk - 1);

                    // center of stencil
                    if (ii == 1 && jj == 1 && kk == 1)
                        REQUIRE(a[row][col] == factor1);
                    // adjacent to center
                    else if (ii == 1 && jj == 1 && kk != 1 ||
                            ii == 1 && jj != 1 && kk == 1 ||
                            ii != 1 && jj == 1 && kk == 1)
                        REQUIRE(a[row][col] == factor2);
                    // diagonally adjacent to center
                    else if (ii == 1 && jj != 1 && kk != 1 ||
                            ii != 1 && jj != 1 && kk == 1 ||
                            ii != 1 && jj == 1 && kk != 1)
                        REQUIRE(a[row][col] == factor3);
                    // corner of stencil
                    else if (ii != 1 && jj != 1 && kk != 1)
                        REQUIRE(a[row][col] == factor4);
                    else
                        REQUIRE(a[row][col] == 0.0);
                }
        // clang-format on
    }

    SECTION("periodic")
    {
        int m2 = 2 * m;
        int n2 = 2 * n;
        int o2 = 2 * o;
        auto a = mgcl_test::Matrix2d::restrictionFullWeight(m2, n2, o2);

        REQUIRE(a.getM() == m * n * o);
        REQUIRE(a.getN() == m2 * n2 * o2);

        double factor1 = 8.0 / 64.0;
        double factor2 = 4.0 / 64.0;
        double factor3 = 2.0 / 64.0;
        double factor4 = 1.0 / 64.0;

        // stores the amount of stencil entries that map to a grid point. Tuple entries:
        // 0: Amount of center entries (should be 0 or 1)
        // 1: Amount of entries adjacent to center (should be 0 to 4)
        // 2: Amount of entries diagonally adjacent to center (should be 0 to 14)
        // 3: Amount of outer corner entries (should be 0 to 8)
        std::tuple<int, int, int, int> mappingsPerEntry[m * n * o][m2 * n2 * o2];

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
        for (int i = 0, i2 = 1; i < m; i++, i2 += 2)
        for (int j = 0, j2 = 1; j < n; j++, j2 += 2)
        for (int k = 0, k2 = 1; k < o; k++, k2 += 2)
            for (int ii = 0; ii < 3; ii++)
            for (int jj = 0; jj < 3; jj++)
            for (int kk = 0; kk < 3; kk++)
            {
                // taken from fromVaryingStencil
                int gpi = i2 + (ii - 1); // grid point index mapped to by stencil entry in x-direction
                int gpj = j2 + (jj - 1); // grid point index mapped to by stencil entry in y-direction
                int gpk = k2 + (kk - 1); // grid point index mapped to by stencil entry in z-direction

                // wrap around for periodic bc
                gpi = gpi % m2;
                gpj = gpj % n2;
                gpk = gpk % o2;

                // shift mod result into positive range
                if (gpi < 0)
                    gpi += m2;

                if (gpj < 0)
                    gpj += n2;

                if (gpk < 0)
                    gpk += o2;

                int row = i * n * o + j * o + k;
                int col = gpi * n2 * o2 + gpj * o2 + gpk;

                // center of stencil
                if (ii == 1 && jj == 1 && kk == 1)
                    std::get<0>(mappingsPerEntry[row][col])++;
                // adjacent to center
                else if (ii == 1 && jj == 1 && kk != 1 ||
                        ii == 1 && jj != 1 && kk == 1 ||
                        ii != 1 && jj == 1 && kk == 1)
                    std::get<1>(mappingsPerEntry[row][col])++;
                // diagonally adjacent to center
                else if (ii == 1 && jj != 1 && kk != 1 ||
                        ii != 1 && jj != 1 && kk == 1 ||
                        ii != 1 && jj == 1 && kk != 1)
                    std::get<2>(mappingsPerEntry[row][col])++;
                // corner of stencil
                else if (ii != 1 && jj != 1 && kk != 1)
                    std::get<3>(mappingsPerEntry[row][col])++;
            }
        // clang-format on

        // Now check matrix entries, which should be sums of the mappings
        for (int i = 0; i < m * n * o; i++)
            for (int j = 0; j < m2 * n2 * o2; j++)
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

TEST_CASE("Matrix2d::getStencilIndicesForEntry")
{
    int m = 4; // GENERATE(2, 4);
    int n = 4; // GENERATE(2, 4);
    int o = 4; // GENERATE(2, 4);
    int m2 = m / 2;
    int n2 = n / 2;
    int o2 = o / 2;
    bool periodic = true;
    int width = 3;

    CAPTURE(m, n, o, m2, n2, o2, periodic, width);

    // square matrix
    {
        mgcl::VaryingStencil st(m, n, o, width, 0, 0, 0);
        st.fill1dIndex(false);
        auto a = mgcl_test::Matrix2d::fromVaryingStencil(st, periodic);

        // fill a with 1d index
        for (int i = 0; i < a.getM(); i++)
            for (int j = 0; j < a.getN(); j++)
            {
                CAPTURE(i, j);
                auto s = a.getStencilIndicesForEntry(i, j, m, n, o, width, periodic);
                CAPTURE(s[0], s[1], s[2], s[3], s[4], s[5]);

                // stencil entry exists
                REQUIRE((a[i][j] == 0 || (a[i][j] > 0 && s[0] != -1)));
                if (s[0] != -1)
                {
                    CAPTURE(a[i][j], st[s[3]][s[4]][s[5]][s[0]][s[1]][s[2]]);
                    REQUIRE((a[i][j] == st[s[3]][s[4]][s[5]][s[0]][s[1]][s[2]]));
                }
            }
    }

    // restriction matrix
    {
        mgcl::VaryingStencil st(m, n, o, width, 0, 0, 0);
        st.fill1dIndex(false);
        auto a_tmp = mgcl_test::Matrix2d::fromVaryingStencil(st, periodic);
        auto a = mgcl_test::Matrix2d::cuttingMatrix3d(m2, n2, o2) * a_tmp;

        CAPTURE(a.getM(), a.getN());

        // fill a with 1d index
        for (int i = 0; i < a.getM(); i++)
            for (int j = 0; j < a.getN(); j++)
            {
                CAPTURE(i, j);
                auto s = a.getStencilIndicesForEntry(i, j, m, n, o, width, periodic);
                CAPTURE(s[0], s[1], s[2], s[3], s[4], s[5]);

                // stencil entry exists
                REQUIRE((a[i][j] == 0 || (a[i][j] > 0 && s[0] != -1)));
                if (s[0] != -1)
                {
                    CAPTURE(a[i][j], st[s[3]][s[4]][s[5]][s[0]][s[1]][s[2]]);
                    REQUIRE((a[i][j] == st[s[3]][s[4]][s[5]][s[0]][s[1]][s[2]]));
                }
            }
    }

    // prologation matrix
    {
        mgcl::VaryingStencil st(m, n, o, width, 0, 0, 0);
        st.fill1dIndex(false);
        auto a_tmp = mgcl_test::Matrix2d::fromVaryingStencil(st, periodic);
        auto a = a_tmp * mgcl_test::Matrix2d::cuttingMatrix3d(m2, n2, o2).transposed();

        CAPTURE(a.getM(), a.getN());

        // fill a with 1d index
        for (int i = 0; i < a.getM(); i++)
            for (int j = 0; j < a.getN(); j++)
            {
                CAPTURE(i, j);
                auto s = a.getStencilIndicesForEntry(i, j, m, n, o, width, periodic);
                CAPTURE(s[0], s[1], s[2], s[3], s[4], s[5]);

                // stencil entry exists
                REQUIRE((a[i][j] == 0 || (a[i][j] > 0 && s[0] != -1)));
                if (s[0] != -1)
                {
                    CAPTURE(a[i][j], st[s[3]][s[4]][s[5]][s[0]][s[1]][s[2]]);
                    REQUIRE((a[i][j] == st[s[3]][s[4]][s[5]][s[0]][s[1]][s[2]]));
                }
            }
    }
}
