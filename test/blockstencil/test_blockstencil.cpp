
#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <iostream>
#include <memory>

#include "../../src/mgcl/blockstencil.hpp"
#include "../../src/mgcl/cuboid.hpp"
#include "../../src/mgcl/matrix.hpp"
#include "../../src/mgcl/multigrid_engine.hpp"
#include "../../src/mgcl/stencil.hpp"

TEST_CASE("Blockstencil::updateGhostsLocally")
{
    SECTION("m,n,o >= gh")
    {
        // regular ghost update
        int m = 2;
        int n = 3;
        int o = 4;
        // int m = GENERATE(2, 3, 4);
        // int n = GENERATE(2, 3, 4);
        // int o = GENERATE(2, 3, 4);

        int gh = GENERATE(1, 2);
        int blocksize = 2;
        int width = GENERATE(3, 5);

        mgcl::Blockstencil a(m, n, o, width, blocksize, gh, gh, gh);
        a.fillRandomInt();
        a.updateGhostsLocally();

        // check in z-direction
        for (int i = 0; i < gh; i++)
            for (int j = 0; j < n + 2 * gh; j++)
                for (int k = 0; k < o + 2 * gh; k++)
                    for (int ii = 0; ii < width; ii++)
                        for (int jj = 0; jj < width; jj++)
                            for (int kk = 0; kk < width; kk++)
                                for (int bi = 0; bi < blocksize; bi++)
                                    for (int bj = 0; bj < blocksize; bj++)
                                    {
                                        REQUIRE(a[bi][bj][ii][jj][kk][i][j][k] == a[bi][bj][ii][jj][kk][i + m][j][k]);
                                        REQUIRE(a[bi][bj][ii][jj][kk][i + gh][j][k] == a[bi][bj][ii][jj][kk][i + gh + m][j][k]);
                                    }

        // check in y-direction
        for (int i = 0; i < m + 2 * gh; i++)
            for (int j = 0; j < gh; j++)
                for (int k = 0; k < o + 2 * gh; k++)
                    for (int ii = 0; ii < width; ii++)
                        for (int jj = 0; jj < width; jj++)
                            for (int kk = 0; kk < width; kk++)
                                for (int bi = 0; bi < blocksize; bi++)
                                    for (int bj = 0; bj < blocksize; bj++)
                                    {
                                        REQUIRE(a[bi][bj][ii][jj][kk][i][j][k] == a[bi][bj][ii][jj][kk][i][j + n][k]);
                                        REQUIRE(a[bi][bj][ii][jj][kk][i][j + gh][k] == a[bi][bj][ii][jj][kk][i][j + gh + n][k]);
                                    }

        // check in x-direction
        for (int i = 0; i < m + 2 * gh; i++)
            for (int j = 0; j < n + 2 * gh; j++)
                for (int k = 0; k < gh; k++)
                    for (int ii = 0; ii < width; ii++)
                        for (int jj = 0; jj < width; jj++)
                            for (int kk = 0; kk < width; kk++)
                                for (int bi = 0; bi < blocksize; bi++)
                                    for (int bj = 0; bj < blocksize; bj++)
                                    {
                                        REQUIRE(a[bi][bj][ii][jj][kk][i][j][k] == a[bi][bj][ii][jj][kk][i][j][k + o]);
                                        REQUIRE(a[bi][bj][ii][jj][kk][i][j][k + gh] == a[bi][bj][ii][jj][kk][i][j][k + gh + o]);
                                    }
    }

    SECTION("m,n,o < gh")
    {
        // periodic ghost update, i.e. real grid is repeated (partially) more than once in ghost cells
        int m = 2;
        int n = 1;
        int o = 3;
        // int m = GENERATE(1, 2, 3);
        // int n = GENERATE(1, 2, 3);
        // int o = GENERATE(1, 2, 3);

        int gh = GENERATE(4, 5);
        int blocksize = 2;
        int width = GENERATE(3, 5);

        mgcl::Blockstencil a(m, n, o, width, blocksize, gh, gh, gh);
        a.fillRandomInt();
        a.updateGhostsLocally();

        // check in z-direction
        for (int i = 0; i < gh; i++)
            for (int j = 0; j < n + 2 * gh; j++)
                for (int k = 0; k < o + 2 * gh; k++)
                    for (int ii = 0; ii < width; ii++)
                        for (int jj = 0; jj < width; jj++)
                            for (int kk = 0; kk < width; kk++)
                                for (int bi = 0; bi < blocksize; bi++)
                                    for (int bj = 0; bj < blocksize; bj++)
                                    {
                                        REQUIRE(a[bi][bj][ii][jj][kk][i][j][k] == a[bi][bj][ii][jj][kk][i + m][j][k]);
                                        REQUIRE(a[bi][bj][ii][jj][kk][i + gh][j][k] == a[bi][bj][ii][jj][kk][i + gh + m][j][k]);
                                    }

        // check in y-direction
        for (int i = 0; i < m + 2 * gh; i++)
            for (int j = 0; j < gh; j++)
                for (int k = 0; k < o + 2 * gh; k++)
                    for (int ii = 0; ii < width; ii++)
                        for (int jj = 0; jj < width; jj++)
                            for (int kk = 0; kk < width; kk++)
                                for (int bi = 0; bi < blocksize; bi++)
                                    for (int bj = 0; bj < blocksize; bj++)
                                    {
                                        REQUIRE(a[bi][bj][ii][jj][kk][i][j][k] == a[bi][bj][ii][jj][kk][i][j + n][k]);
                                        REQUIRE(a[bi][bj][ii][jj][kk][i][j + gh][k] == a[bi][bj][ii][jj][kk][i][j + gh + n][k]);
                                    }

        // check in x-direction
        for (int i = 0; i < m + 2 * gh; i++)
            for (int j = 0; j < n + 2 * gh; j++)
                for (int k = 0; k < gh; k++)
                    for (int ii = 0; ii < width; ii++)
                        for (int jj = 0; jj < width; jj++)
                            for (int kk = 0; kk < width; kk++)
                                for (int bi = 0; bi < blocksize; bi++)
                                    for (int bj = 0; bj < blocksize; bj++)
                                    {
                                        REQUIRE(a[bi][bj][ii][jj][kk][i][j][k] == a[bi][bj][ii][jj][kk][i][j][k + o]);
                                        REQUIRE(a[bi][bj][ii][jj][kk][i][j][k + gh] == a[bi][bj][ii][jj][kk][i][j][k + gh + o]);
                                    }
    }

    SECTION("mixed")
    {
        // mixed periodic and regular ghost update, e.g. m > gh but n < gh
        int m = 2;
        int n = 2;
        int o = 3;

        int ghm = GENERATE(1, 5);
        int ghn = GENERATE(1, 3);
        int gho = GENERATE(3, 5);

        // int m = GENERATE(2, 3);
        // int n = GENERATE(2, 3);
        // int o = GENERATE(2, 3);
        //
        // int ghm = GENERATE(1, 3, 5);
        // int ghn = GENERATE(1, 3, 5);
        // int gho = GENERATE(1, 3, 5);

        int blocksize = 2;
        int width = GENERATE(3, 5);

        mgcl::Blockstencil a(m, n, o, width, blocksize, ghm, ghn, gho);

        a.fillRandomInt();
        a.updateGhostsLocally();

        // check in z-direction
        for (int i = 0; i < ghm; i++)
            for (int j = 0; j < n + 2 * ghn; j++)
                for (int k = 0; k < o + 2 * gho; k++)
                    for (int ii = 0; ii < width; ii++)
                        for (int jj = 0; jj < width; jj++)
                            for (int kk = 0; kk < width; kk++)
                                for (int bi = 0; bi < blocksize; bi++)
                                    for (int bj = 0; bj < blocksize; bj++)
                                    {
                                        REQUIRE(a[bi][bj][ii][jj][kk][i][j][k] == a[bi][bj][ii][jj][kk][i + m][j][k]);
                                        REQUIRE(a[bi][bj][ii][jj][kk][i + ghm][j][k] == a[bi][bj][ii][jj][kk][i + ghm + m][j][k]);
                                    }

        // check in y-direction
        for (int i = 0; i < m + 2 * ghm; i++)
            for (int j = 0; j < ghn; j++)
                for (int k = 0; k < o + 2 * gho; k++)
                    for (int ii = 0; ii < width; ii++)
                        for (int jj = 0; jj < width; jj++)
                            for (int kk = 0; kk < width; kk++)
                                for (int bi = 0; bi < blocksize; bi++)
                                    for (int bj = 0; bj < blocksize; bj++)
                                    {
                                        REQUIRE(a[bi][bj][ii][jj][kk][i][j][k] == a[bi][bj][ii][jj][kk][i][j + n][k]);
                                        REQUIRE(a[bi][bj][ii][jj][kk][i][j + ghn][k] == a[bi][bj][ii][jj][kk][i][j + ghn + n][k]);
                                    }

        // check in x-direction
        for (int i = 0; i < m + 2 * ghm; i++)
            for (int j = 0; j < n + 2 * ghn; j++)
                for (int k = 0; k < gho; k++)
                    for (int ii = 0; ii < width; ii++)
                        for (int jj = 0; jj < width; jj++)
                            for (int kk = 0; kk < width; kk++)
                                for (int bi = 0; bi < blocksize; bi++)
                                    for (int bj = 0; bj < blocksize; bj++)
                                    {
                                        REQUIRE(a[bi][bj][ii][jj][kk][i][j][k] == a[bi][bj][ii][jj][kk][i][j][k + o]);
                                        REQUIRE(a[bi][bj][ii][jj][kk][i][j][k + gho] == a[bi][bj][ii][jj][kk][i][j][k + gho + o]);
                                    }
    }
}

// TEST_CASE("Blockstencil::multiply")
// {
//     SECTION("valid N=3, not periodic")
//     {
//         int m = 3;
//         int n = 2;
//         int o = 4;
//         // int m = GENERATE(2, 3, 4);
//         // int n = GENERATE(2, 3, 4);
//         // int o = GENERATE(2, 3, 4);

//         mgcl::Blockstencil a(m, n, o, 3, 0, 0, 0);
//         mgcl::Blockstencil b(m, n, o, 3, 1, 1, 1);

//         double factor1 = 8.0 / 64.0;
//         double factor2 = 4.0 / 64.0;
//         double factor3 = 2.0 / 64.0;
//         double factor4 = 1.0 / 64.0;

//         // fill a and b
//         for (int i = 0; i < m; i++)
//             for (int j = 0; j < n; j++)
//                 for (int k = 0; k < o; k++)
//                 {
//                     // 7-point Laplace
//                     a[0][1][1][i][j][k] = 1;
//                     a[1][0][1][i][j][k] = 1;
//                     a[1][1][0][i][j][k] = 1;
//                     a[1][1][1][i][j][k] = -6;
//                     a[1][1][2][i][j][k] = 1;
//                     a[1][2][1][i][j][k] = 1;
//                     a[2][1][1][i][j][k] = 1;

//                     // full-weight restriction
//                     // fill only real cells, ghosts cells must be zero (Dirichlet condition)
//                     b[0][0][0][i + 1][j + 1][k + 1] = factor4;
//                     b[0][0][1][i + 1][j + 1][k + 1] = factor3;
//                     b[0][0][2][i + 1][j + 1][k + 1] = factor4;
//                     b[0][1][0][i + 1][j + 1][k + 1] = factor3;
//                     b[0][1][1][i + 1][j + 1][k + 1] = factor2;
//                     b[0][1][2][i + 1][j + 1][k + 1] = factor3;
//                     b[0][2][0][i + 1][j + 1][k + 1] = factor4;
//                     b[0][2][1][i + 1][j + 1][k + 1] = factor3;
//                     b[0][2][2][i + 1][j + 1][k + 1] = factor4;
//                     b[1][0][0][i + 1][j + 1][k + 1] = factor3;
//                     b[1][0][1][i + 1][j + 1][k + 1] = factor2;
//                     b[1][0][2][i + 1][j + 1][k + 1] = factor3;
//                     b[1][1][0][i + 1][j + 1][k + 1] = factor2;
//                     b[1][1][1][i + 1][j + 1][k + 1] = factor1;
//                     b[1][1][2][i + 1][j + 1][k + 1] = factor2;
//                     b[1][2][0][i + 1][j + 1][k + 1] = factor3;
//                     b[1][2][1][i + 1][j + 1][k + 1] = factor2;
//                     b[1][2][2][i + 1][j + 1][k + 1] = factor3;
//                     b[2][0][0][i + 1][j + 1][k + 1] = factor4;
//                     b[2][0][1][i + 1][j + 1][k + 1] = factor3;
//                     b[2][0][2][i + 1][j + 1][k + 1] = factor4;
//                     b[2][1][0][i + 1][j + 1][k + 1] = factor3;
//                     b[2][1][1][i + 1][j + 1][k + 1] = factor2;
//                     b[2][1][2][i + 1][j + 1][k + 1] = factor3;
//                     b[2][2][0][i + 1][j + 1][k + 1] = factor4;
//                     b[2][2][1][i + 1][j + 1][k + 1] = factor3;
//                     b[2][2][2][i + 1][j + 1][k + 1] = factor4;
//                 }

//         auto c = a.multiply(b, 2, nullptr, false, true);

//         // check result against explicitly calculated matrix product
//         auto c_expected = mgcl_test::Matrix2d::laplace7p3d(m, n, o, false) * mgcl_test::Matrix2d::fullWeightNonCut(m, n, o, false);
//         auto c_m2d = mgcl_test::Matrix2d::fromBlockstencil(c, false);

//         REQUIRE(c_expected.getM() == c_m2d.getM());
//         REQUIRE(c_expected.getN() == c_m2d.getN());

//         CHECK(c_m2d == c_expected);
//     }

//     SECTION("valid N=3, periodic")
//     {
//         int m = 3;
//         int n = 2;
//         int o = 4;
//         // int m = GENERATE(2, 3, 4);
//         // int n = GENERATE(2, 3, 4);
//         // int o = GENERATE(2, 3, 4);

//         mgcl::Blockstencil a(m, n, o, 3, 0, 0, 0);
//         mgcl::Blockstencil b(m, n, o, 3, 1, 1, 1);

//         double factor1 = 8.0 / 64.0;
//         double factor2 = 4.0 / 64.0;
//         double factor3 = 2.0 / 64.0;
//         double factor4 = 1.0 / 64.0;

//         // fill a and b
//         for (int i = 0; i < m + 2; i++)
//             for (int j = 0; j < n + 2; j++)
//                 for (int k = 0; k < o + 2; k++)
//                 {
//                     // 7-point Laplace, real cells only
//                     if (i < m && j < n && k < o)
//                     {
//                         a[0][1][1][i][j][k] = 1;
//                         a[1][0][1][i][j][k] = 1;
//                         a[1][1][0][i][j][k] = 1;
//                         a[1][1][1][i][j][k] = -6;
//                         a[1][1][2][i][j][k] = 1;
//                         a[1][2][1][i][j][k] = 1;
//                         a[2][1][1][i][j][k] = 1;
//                     }

//                     // full-weight restriction
//                     b[0][0][0][i][j][k] = factor4;
//                     b[0][0][1][i][j][k] = factor3;
//                     b[0][0][2][i][j][k] = factor4;
//                     b[0][1][0][i][j][k] = factor3;
//                     b[0][1][1][i][j][k] = factor2;
//                     b[0][1][2][i][j][k] = factor3;
//                     b[0][2][0][i][j][k] = factor4;
//                     b[0][2][1][i][j][k] = factor3;
//                     b[0][2][2][i][j][k] = factor4;
//                     b[1][0][0][i][j][k] = factor3;
//                     b[1][0][1][i][j][k] = factor2;
//                     b[1][0][2][i][j][k] = factor3;
//                     b[1][1][0][i][j][k] = factor2;
//                     b[1][1][1][i][j][k] = factor1;
//                     b[1][1][2][i][j][k] = factor2;
//                     b[1][2][0][i][j][k] = factor3;
//                     b[1][2][1][i][j][k] = factor2;
//                     b[1][2][2][i][j][k] = factor3;
//                     b[2][0][0][i][j][k] = factor4;
//                     b[2][0][1][i][j][k] = factor3;
//                     b[2][0][2][i][j][k] = factor4;
//                     b[2][1][0][i][j][k] = factor3;
//                     b[2][1][1][i][j][k] = factor2;
//                     b[2][1][2][i][j][k] = factor3;
//                     b[2][2][0][i][j][k] = factor4;
//                     b[2][2][1][i][j][k] = factor3;
//                     b[2][2][2][i][j][k] = factor4;
//                 }

//         auto c = a.multiply(b, 2, nullptr, true, true);

//         // check result against explicitly calculated matrix product
//         auto c_expected = mgcl_test::Matrix2d::laplace7p3d(m, n, o) * mgcl_test::Matrix2d::fullWeightNonCut(m, n, o);
//         auto c_m2d = mgcl_test::Matrix2d::fromBlockstencil(c, true);

//         REQUIRE(c_expected.getM() == c_m2d.getM());
//         REQUIRE(c_expected.getN() == c_m2d.getN());

//         CHECK(c_m2d == c_expected);
//     }

//     SECTION("different Ns, random values")
//     {
//         int m = 3;
//         int n = 2;
//         int o = 4;
//         // int m = GENERATE(2, 3, 4);
//         // int n = GENERATE(2, 3, 4);
//         // int o = GENERATE(2, 3, 4);

//         SECTION("widths 3 * {3,5,7}")
//         {
//             int ghb = GENERATE(1, 2);
//             int gha = GENERATE(0, 1, 2);

//             mgcl::Blockstencil b3p(m, n, o, 3, ghb, ghb, ghb);
//             mgcl::Blockstencil b5p(m, n, o, 5, ghb, ghb, ghb);
//             mgcl::Blockstencil b7p(m, n, o, 7, ghb, ghb, ghb);
//             mgcl::Blockstencil b3np(m, n, o, 3, ghb, ghb, ghb);
//             mgcl::Blockstencil b5np(m, n, o, 5, ghb, ghb, ghb);
//             mgcl::Blockstencil b7np(m, n, o, 7, ghb, ghb, ghb);

//             b3p.fillRandomInt(1, 9);
//             b5p.fillRandomInt(1, 9);
//             b7p.fillRandomInt(1, 9);
//             // fill only real cells for non-periodic stencils, ghosts are 0.
//             b3np.fillRandomInt(1, 9, true);
//             b5np.fillRandomInt(1, 9, true);
//             b7np.fillRandomInt(1, 9, true);

//             // update ghosts only for periodic stencils
//             b3p.updateGhosts();
//             b5p.updateGhosts();
//             b7p.updateGhosts();

//             // build matrices from stencils to check results
//             auto mb3p = mgcl_test::Matrix2d::fromBlockstencil(b3p, true);
//             auto mb5p = mgcl_test::Matrix2d::fromBlockstencil(b5p, true);
//             auto mb7p = mgcl_test::Matrix2d::fromBlockstencil(b7p, true);
//             auto mb3np = mgcl_test::Matrix2d::fromBlockstencil(b3np, false);
//             auto mb5np = mgcl_test::Matrix2d::fromBlockstencil(b5np, false);
//             auto mb7np = mgcl_test::Matrix2d::fromBlockstencil(b7np, false);

//             mgcl::Blockstencil a3(m, n, o, 3, gha, gha, gha);
//             a3.fillRandomInt(1, 9);

//             // build matrices from stencils to check results later
//             auto ma3p = mgcl_test::Matrix2d::fromBlockstencil(a3, true);
//             auto ma3np = mgcl_test::Matrix2d::fromBlockstencil(a3, false);

//             {
//                 auto c33p = a3.multiply(b3p, 2, nullptr, true, true);
//                 auto c33np = a3.multiply(b3np, 2, nullptr, false, true);
//                 auto c33_expected_p = ma3p * mb3p;
//                 auto c33_expected_np = ma3np * mb3np;
//                 auto mc33p = mgcl_test::Matrix2d::fromBlockstencil(c33p, true);
//                 auto mc33np = mgcl_test::Matrix2d::fromBlockstencil(c33np, false);
//                 REQUIRE(mc33p == c33_expected_p);
//                 REQUIRE(mc33np == c33_expected_np);
//             }

//             {
//                 auto c35p = a3.multiply(b5p, 2, nullptr, true, true);
//                 auto c35np = a3.multiply(b5np, 2, nullptr, false, true);
//                 auto c35_expected_p = ma3p * mb5p;
//                 auto c35_expected_np = ma3np * mb5np;
//                 auto mc35p = mgcl_test::Matrix2d::fromBlockstencil(c35p, true);
//                 auto mc35np = mgcl_test::Matrix2d::fromBlockstencil(c35np, false);
//                 REQUIRE(mc35p == c35_expected_p);
//                 REQUIRE(mc35np == c35_expected_np);
//             }

//             {
//                 auto c37p = a3.multiply(b7p, 2, nullptr, true, true);
//                 auto c37np = a3.multiply(b7np, 2, nullptr, false, true);
//                 auto c37_expected_p = ma3p * mb7p;
//                 auto c37_expected_np = ma3np * mb7np;
//                 auto mc37p = mgcl_test::Matrix2d::fromBlockstencil(c37p, true);
//                 auto mc37np = mgcl_test::Matrix2d::fromBlockstencil(c37np, false);
//                 REQUIRE(mc37p == c37_expected_p);
//                 REQUIRE(mc37np == c37_expected_np);
//             }
//         }

//         SECTION("widths 5 * {3,5,7}")
//         {
//             mgcl::Blockstencil b3p(m, n, o, 3, 2, 2, 2);
//             mgcl::Blockstencil b5p(m, n, o, 5, 2, 2, 2);
//             mgcl::Blockstencil b7p(m, n, o, 7, 2, 2, 2);
//             mgcl::Blockstencil b3np(m, n, o, 3, 2, 2, 2);
//             mgcl::Blockstencil b5np(m, n, o, 5, 2, 2, 2);
//             mgcl::Blockstencil b7np(m, n, o, 7, 2, 2, 2);

//             b3p.fillRandomInt(1, 9);
//             b5p.fillRandomInt(1, 9);
//             b7p.fillRandomInt(1, 9);
//             // fill only real cells for non-periodic stencils, ghosts are 0.
//             b3np.fillRandomInt(1, 9, true);
//             b5np.fillRandomInt(1, 9, true);
//             b7np.fillRandomInt(1, 9, true);

//             // update ghosts only for periodic stencils
//             b3p.updateGhosts();
//             b5p.updateGhosts();
//             b7p.updateGhosts();

//             // build matrices from stencils to check results
//             auto mb3p = mgcl_test::Matrix2d::fromBlockstencil(b3p, true);
//             auto mb5p = mgcl_test::Matrix2d::fromBlockstencil(b5p, true);
//             auto mb7p = mgcl_test::Matrix2d::fromBlockstencil(b7p, true);
//             auto mb3np = mgcl_test::Matrix2d::fromBlockstencil(b3np, false);
//             auto mb5np = mgcl_test::Matrix2d::fromBlockstencil(b5np, false);
//             auto mb7np = mgcl_test::Matrix2d::fromBlockstencil(b7np, false);

//             mgcl::Blockstencil a5(m, n, o, 5, 0, 0, 0);
//             a5.fillRandomInt(1, 9);

//             // build matrices from stencils to check results later
//             auto ma5p = mgcl_test::Matrix2d::fromBlockstencil(a5, true);
//             auto ma5np = mgcl_test::Matrix2d::fromBlockstencil(a5, false);

//             {
//                 auto c53p = a5.multiply(b3p, 2, nullptr, true, true);
//                 auto c53np = a5.multiply(b3np, 2, nullptr, false, true);
//                 auto c53_expected_p = ma5p * mb3p;
//                 auto c53_expected_np = ma5np * mb3np;
//                 auto mc53p = mgcl_test::Matrix2d::fromBlockstencil(c53p, true);
//                 auto mc53np = mgcl_test::Matrix2d::fromBlockstencil(c53np, false);
//                 REQUIRE(mc53p == c53_expected_p);
//                 REQUIRE(mc53np == c53_expected_np);
//             }

//             {
//                 auto c55p = a5.multiply(b5p, 2, nullptr, true, true);
//                 auto c55np = a5.multiply(b5np, 2, nullptr, false, true);
//                 auto c55_expected_p = ma5p * mb5p;
//                 auto c55_expected_np = ma5np * mb5np;
//                 auto mc55p = mgcl_test::Matrix2d::fromBlockstencil(c55p, true);
//                 auto mc55np = mgcl_test::Matrix2d::fromBlockstencil(c55np, false);
//                 REQUIRE(mc55p == c55_expected_p);
//                 REQUIRE(mc55np == c55_expected_np);
//             }

//             {
//                 auto c57p = a5.multiply(b7p, 2, nullptr, true, true);
//                 auto c57np = a5.multiply(b7np, 2, nullptr, false, true);
//                 auto c57_expected_p = ma5p * mb7p;
//                 auto c57_expected_np = ma5np * mb7np;
//                 auto mc57p = mgcl_test::Matrix2d::fromBlockstencil(c57p, true);
//                 auto mc57np = mgcl_test::Matrix2d::fromBlockstencil(c57np, false);
//                 REQUIRE(mc57p == c57_expected_p);
//                 REQUIRE(mc57np == c57_expected_np);
//             }
//         }

//         SECTION("widths 7 * {3,5,7}")
//         {
//             mgcl::Blockstencil b3p(m, n, o, 3, 3, 3, 3);
//             mgcl::Blockstencil b5p(m, n, o, 5, 3, 3, 3);
//             mgcl::Blockstencil b7p(m, n, o, 7, 3, 3, 3);
//             mgcl::Blockstencil b3np(m, n, o, 3, 3, 3, 3);
//             mgcl::Blockstencil b5np(m, n, o, 5, 3, 3, 3);
//             mgcl::Blockstencil b7np(m, n, o, 7, 3, 3, 3);

//             b3p.fillRandomInt(1, 9);
//             b5p.fillRandomInt(1, 9);
//             b7p.fillRandomInt(1, 9);
//             // fill only real cells for non-periodic stencils, ghosts are 0.
//             b3np.fillRandomInt(1, 9, true);
//             b5np.fillRandomInt(1, 9, true);
//             b7np.fillRandomInt(1, 9, true);

//             // update ghosts only for periodic stencils
//             b3p.updateGhosts();
//             b5p.updateGhosts();
//             b7p.updateGhosts();

//             // build matrices from stencils to check results
//             auto mb3p = mgcl_test::Matrix2d::fromBlockstencil(b3p, true);
//             auto mb5p = mgcl_test::Matrix2d::fromBlockstencil(b5p, true);
//             auto mb7p = mgcl_test::Matrix2d::fromBlockstencil(b7p, true);
//             auto mb3np = mgcl_test::Matrix2d::fromBlockstencil(b3np, false);
//             auto mb5np = mgcl_test::Matrix2d::fromBlockstencil(b5np, false);
//             auto mb7np = mgcl_test::Matrix2d::fromBlockstencil(b7np, false);

//             mgcl::Blockstencil a7(m, n, o, 7, 0, 0, 0);
//             a7.fillRandomInt(1, 9);

//             // build matrices from stencils to check results later
//             auto ma7p = mgcl_test::Matrix2d::fromBlockstencil(a7, true);
//             auto ma7np = mgcl_test::Matrix2d::fromBlockstencil(a7, false);

//             {
//                 auto c73p = a7.multiply(b3p, 2, nullptr, true, true);
//                 auto c73np = a7.multiply(b3np, 2, nullptr, false, true);
//                 auto c73_expected_p = ma7p * mb3p;
//                 auto c73_expected_np = ma7np * mb3np;
//                 auto mc73p = mgcl_test::Matrix2d::fromBlockstencil(c73p, true);
//                 auto mc73np = mgcl_test::Matrix2d::fromBlockstencil(c73np, false);
//                 REQUIRE(mc73p == c73_expected_p);
//                 REQUIRE(mc73np == c73_expected_np);
//             }

//             if (m > 3 && n > 3 && o > 3)
//             {
//                 auto c75p = a7.multiply(b5p, 2, nullptr, true, true);
//                 auto c75np = a7.multiply(b5np, 2, nullptr, false, true);
//                 auto c75_expected_p = ma7p * mb5p;
//                 auto c75_expected_np = ma7np * mb5np;
//                 auto mc75p = mgcl_test::Matrix2d::fromBlockstencil(c75p, true);
//                 auto mc75np = mgcl_test::Matrix2d::fromBlockstencil(c75np, false);
//                 REQUIRE(mc75p == c75_expected_p);
//                 REQUIRE(mc75np == c75_expected_np);
//             }

//             {
//                 auto c77p = a7.multiply(b7p, 2, nullptr, true, true);
//                 auto c77np = a7.multiply(b7np, 2, nullptr, false, true);
//                 auto c77_expected_p = ma7p * mb7p;
//                 auto c77_expected_np = ma7np * mb7np;
//                 auto mc77p = mgcl_test::Matrix2d::fromBlockstencil(c77p, true);
//                 auto mc77np = mgcl_test::Matrix2d::fromBlockstencil(c77np, false);
//                 REQUIRE(mc77p == c77_expected_p);
//                 REQUIRE(mc77np == c77_expected_np);
//             }
//         }
//     }

//     SECTION("throwing")
//     {
//         int m = GENERATE(1, 2, 3);
//         int n = GENERATE(1, 2, 3);
//         int o = GENERATE(1, 2, 3);
//         int ghm = GENERATE(1, 2, 3);
//         int ghn = GENERATE(1, 2, 3);
//         int gho = GENERATE(1, 2, 3);

//         mgcl::Blockstencil a(m, n, o, 3, ghm, ghn, gho);
//         mgcl::Blockstencil b(m, n, o, 3, ghm, ghn, gho);

//         bool dimsNotEqual = a.getM() != b.getM() || a.getN() != b.getN() || a.getO() != b.getO();
//         bool ghostsNotBigEnough = b.getGhostsM() < 1 || b.getGhostsN() < 1 || b.getGhostsO() < 1;
//         bool ghostsNotEqual = b.getGhostsM() != b.getGhostsN() || b.getGhostsM() != b.getGhostsO();

//         if (dimsNotEqual || ghostsNotEqual || ghostsNotBigEnough)
//             REQUIRE_THROWS(a.multiply(b, 2, nullptr, true, true));
//         else
//             REQUIRE_NOTHROW(a.multiply(b, 2, nullptr, true, true));
//     }
// }

TEST_CASE("Blockstencil::slice")
{
    int m = 4;
    int n = 4;
    int o = 4;
    int blocksize = 2;

    mgcl::Blockstencil cb(m, n, o, 3, blocksize, 1, 1, 1);
    // cb.fillRandom();
    cb.fill1dIndex(true);

    SECTION("throwing")
    {
        REQUIRE_THROWS(cb.slice(-1, 0, 0, 0, 0, 0));
        REQUIRE_THROWS(cb.slice(0, 0, -1, 0, 0, 0));
        REQUIRE_THROWS(cb.slice(0, 0, 0, 0, -1, 0));

        REQUIRE_THROWS(cb.slice(0, m + 3, 0, 0, 0, 0));
        REQUIRE_THROWS(cb.slice(0, 0, 0, n + 3, 0, 0));
        REQUIRE_THROWS(cb.slice(0, 0, 0, 0, 0, n + 3));
    }

    SECTION("success")
    {
        auto cs = cb.slice(0, 1, 0, 2, 2, 3);

        REQUIRE(cs->getM() == 2);
        REQUIRE(cs->getN() == 3);
        REQUIRE(cs->getO() == 2);
        REQUIRE(cs->getWidth() == cb.getWidth());
        REQUIRE(cs->getBlocksize() == cb.getBlocksize());
        REQUIRE(cs->getGhostsM() == cb.getGhostsM());
        REQUIRE(cs->getGhostsN() == cb.getGhostsN());
        REQUIRE(cs->getGhostsO() == cb.getGhostsO());

        for (int d1 = cs->getGhostsM(); d1 < cs->getM() + cs->getGhostsM(); d1++)
            for (int d2 = cs->getGhostsN(); d2 < cs->getN() + cs->getGhostsN(); d2++)
                for (int d3 = cs->getGhostsO(); d3 < cs->getO() + cs->getGhostsO(); d3++)
                    for (int d4 = 0; d4 < cs->getWidth(); d4++)
                        for (int d5 = 0; d5 < cs->getWidth(); d5++)
                            for (int d6 = 0; d6 < cs->getWidth(); d6++)
                                for (int d7 = 0; d7 < cs->getBlocksize(); d7++)
                                    for (int d8 = 0; d8 < cs->getBlocksize(); d8++)
                                    {
                                        CAPTURE(d1, d2, d3, d4, d5, d6, d7, d8);
                                        REQUIRE(cs->getData()[d7][d8][d4][d5][d6][d1][d2][d3] == cb[d7][d8][d4][d5][d6][d1][d2][d3 + 2]);
                                    }
    }
}

TEST_CASE("Blockstencil::sliceIncGhosts")
{
    int m = 4;
    int n = 4;
    int o = 4;
    int blocksize = 2;

    mgcl::Blockstencil cb(m, n, o, 3, blocksize, 1, 1, 1);
    cb.fillRandom();

    SECTION("throwing")
    {
        REQUIRE_THROWS(cb.sliceIncGhosts(-1, 0, 0, 0, 0, 0));
        REQUIRE_THROWS(cb.sliceIncGhosts(0, 0, -1, 0, 0, 0));
        REQUIRE_THROWS(cb.sliceIncGhosts(0, 0, 0, 0, -1, 0));

        REQUIRE_THROWS(cb.sliceIncGhosts(0, m + 3, 0, 0, 0, 0));
        REQUIRE_THROWS(cb.sliceIncGhosts(0, 0, 0, n + 3, 0, 0));
        REQUIRE_THROWS(cb.sliceIncGhosts(0, 0, 0, 0, 0, n + 3));
    }

    SECTION("success")
    {
        auto cs = cb.sliceIncGhosts(0, 1, 0, 2, 2, 3);

        REQUIRE(cs->getM() == 2);
        REQUIRE(cs->getN() == 3);
        REQUIRE(cs->getO() == 2);
        REQUIRE(cs->getWidth() == cb.getWidth());
        REQUIRE(cs->getBlocksize() == cb.getBlocksize());
        REQUIRE(cs->getGhostsM() == 0);
        REQUIRE(cs->getGhostsN() == 0);
        REQUIRE(cs->getGhostsO() == 0);

        for (int d1 = 0; d1 < cs->getMgh(); d1++)
            for (int d2 = 0; d2 < cs->getNgh(); d2++)
                for (int d3 = 0; d3 < cs->getOgh(); d3++)
                    for (int d4 = 0; d4 < cs->getWidth(); d4++)
                        for (int d5 = 0; d5 < cs->getWidth(); d5++)
                            for (int d6 = 0; d6 < cs->getWidth(); d6++)
                                for (int d7 = 0; d7 < cs->getBlocksize(); d7++)
                                    for (int d8 = 0; d8 < cs->getBlocksize(); d8++)
                                    {
                                        REQUIRE(cs->getData()[d7][d8][d4][d5][d6][d1][d2][d3] == cb[d7][d8][d4][d5][d6][d1][d2][d3 + 2]);
                                    }
    }
}

TEST_CASE("Blockstencil::copyShallow")
{
    mgcl::Blockstencil s1(2, 3, 4, 3, 2, 5, 6, 7);
    s1.fillRandom();
    auto s2 = s1.copyShallow();

    REQUIRE(s1.getM() == s2->getM());
    REQUIRE(s1.getN() == s2->getN());
    REQUIRE(s1.getO() == s2->getO());
    REQUIRE(s1.getWidth() == s2->getWidth());
    REQUIRE(s1.getBlocksize() == s2->getBlocksize());
    REQUIRE(s1.getGhostsM() == s2->getGhostsM());
    REQUIRE(s1.getGhostsN() == s2->getGhostsN());
    REQUIRE(s1.getGhostsO() == s2->getGhostsO());
    REQUIRE(s1.getGhostsDim4() == s2->getGhostsDim4());
    REQUIRE(s1.getGhostsDim5() == s2->getGhostsDim5());
    REQUIRE(s1.getGhostsDim6() == s2->getGhostsDim6());
    REQUIRE(s1.getGhostsDim7() == s2->getGhostsDim7());
    REQUIRE(s1.getGhostsDim8() == s2->getGhostsDim8());
    REQUIRE(s1.getSize() == s2->getSize());
}

TEST_CASE("Blockstencil::invertCenterMatrices")
{
    int m = 1;
    int n = 2;
    int o = 3;
    int ghm = GENERATE(0, 1);
    int ghn = 1;
    int gho = 1;
    int width = 3;
    int blocksize = 2;

    mgcl::Blockstencil s(m, n, o, width, blocksize, ghm, ghn, gho);
    // fill with 1d real cell indices, as if there were no ghosts
    int cnt = 0;
    for (int d1 = s.getGhostsDim1(); d1 < s.getDim1() + s.getGhostsDim1(); d1++)
        for (int d2 = s.getGhostsDim2(); d2 < s.getDim2() + s.getGhostsDim2(); d2++)
            for (int d3 = s.getGhostsDim3(); d3 < s.getDim3() + s.getGhostsDim3(); d3++)
                for (int d4 = s.getGhostsDim4(); d4 < s.getDim4() + s.getGhostsDim4(); d4++)
                    for (int d5 = s.getGhostsDim5(); d5 < s.getDim5() + s.getGhostsDim5(); d5++)
                        for (int d6 = s.getGhostsDim6(); d6 < s.getDim6() + s.getGhostsDim6(); d6++)
                            for (int d7 = s.getGhostsDim7(); d7 < s.getDim7() + s.getGhostsDim7(); d7++)
                                for (int d8 = s.getGhostsDim8(); d8 < s.getDim8() + s.getGhostsDim8(); d8++)
                                {
                                    s[d1][d2][d3][d4][d5][d6][d7][d8] = cnt++;
                                }
    // s.dumpToFile("s.txt");

    SECTION("success")
    {
        // print center coeffs for Matlab
        // for (int i = 0; i < m; i++)
        //     for (int j = 0; j < n; j++)
        //         for (int k = 0; k < o; k++)
        //         {
        //             std::cout << "a" << i << j << k << " = [";
        //             for (int bi = 0; bi < blocksize; bi++)
        //             {
        //                 for (int bj = 0; bj < blocksize; bj++)
        //                 {
        //                     std::cout << s[bi][bj][width / 2][width / 2][width / 2][i][j][k] << " ";
        //                 }

        //                 if (bi != blocksize - 1)
        //                     std::cout << " ; ";
        //             }
        //             std::cout << "]" << std::endl;
        //         }

        mgcl::Matrix a000_inv(blocksize, blocksize);
        a000_inv[0][0] = -0.0107;
        a000_inv[0][1] = 0.0046;
        a000_inv[1][0] = 0.0077;
        a000_inv[1][1] = -0.0015;

        mgcl::Matrix a001_inv(blocksize, blocksize);
        a001_inv[0][0] = -0.0108;
        a001_inv[0][1] = 0.0046;
        a001_inv[1][0] = 0.0077;
        a001_inv[1][1] = -0.0015;

        mgcl::Matrix a002_inv(blocksize, blocksize);
        a002_inv[0][0] = -0.0108;
        a002_inv[0][1] = 0.0046;
        a002_inv[1][0] = 0.0077;
        a002_inv[1][1] = -0.0015;

        mgcl::Matrix a010_inv(blocksize, blocksize);
        a010_inv[0][0] = -0.0108;
        a010_inv[0][1] = 0.0046;
        a010_inv[1][0] = 0.0077;
        a010_inv[1][1] = -0.0015;

        mgcl::Matrix a011_inv(blocksize, blocksize);
        a011_inv[0][0] = -0.0108;
        a011_inv[0][1] = 0.0046;
        a011_inv[1][0] = 0.0077;
        a011_inv[1][1] = -0.0016;

        mgcl::Matrix a012_inv(blocksize, blocksize);
        a012_inv[0][0] = -0.0108;
        a012_inv[0][1] = 0.0047;
        a012_inv[1][0] = 0.0078;
        a012_inv[1][1] = -0.0016;

        auto s2 = s.invertCenterMatrices();

        REQUIRE_THAT((*s2)[0][0][0][0][0][0][0][0], Catch::Matchers::WithinAbs(a000_inv[0][0], 1e-4));
        REQUIRE_THAT((*s2)[0][1][0][0][0][0][0][0], Catch::Matchers::WithinAbs(a000_inv[0][1], 1e-4));
        REQUIRE_THAT((*s2)[1][0][0][0][0][0][0][0], Catch::Matchers::WithinAbs(a000_inv[1][0], 1e-4));
        REQUIRE_THAT((*s2)[1][1][0][0][0][0][0][0], Catch::Matchers::WithinAbs(a000_inv[1][1], 1e-4));
        REQUIRE_THAT((*s2)[0][0][0][0][0][0][0][1], Catch::Matchers::WithinAbs(a001_inv[0][0], 1e-4));
        REQUIRE_THAT((*s2)[0][1][0][0][0][0][0][1], Catch::Matchers::WithinAbs(a001_inv[0][1], 1e-4));
        REQUIRE_THAT((*s2)[1][0][0][0][0][0][0][1], Catch::Matchers::WithinAbs(a001_inv[1][0], 1e-4));
        REQUIRE_THAT((*s2)[1][1][0][0][0][0][0][1], Catch::Matchers::WithinAbs(a001_inv[1][1], 1e-4));
        REQUIRE_THAT((*s2)[0][0][0][0][0][0][0][2], Catch::Matchers::WithinAbs(a002_inv[0][0], 1e-4));
        REQUIRE_THAT((*s2)[0][1][0][0][0][0][0][2], Catch::Matchers::WithinAbs(a002_inv[0][1], 1e-4));
        REQUIRE_THAT((*s2)[1][0][0][0][0][0][0][2], Catch::Matchers::WithinAbs(a002_inv[1][0], 1e-4));
        REQUIRE_THAT((*s2)[1][1][0][0][0][0][0][2], Catch::Matchers::WithinAbs(a002_inv[1][1], 1e-4));
        REQUIRE_THAT((*s2)[0][0][0][0][0][0][1][0], Catch::Matchers::WithinAbs(a010_inv[0][0], 1e-4));
        REQUIRE_THAT((*s2)[0][1][0][0][0][0][1][0], Catch::Matchers::WithinAbs(a010_inv[0][1], 1e-4));
        REQUIRE_THAT((*s2)[1][0][0][0][0][0][1][0], Catch::Matchers::WithinAbs(a010_inv[1][0], 1e-4));
        REQUIRE_THAT((*s2)[1][1][0][0][0][0][1][0], Catch::Matchers::WithinAbs(a010_inv[1][1], 1e-4));
        REQUIRE_THAT((*s2)[0][0][0][0][0][0][1][1], Catch::Matchers::WithinAbs(a011_inv[0][0], 1e-4));
        REQUIRE_THAT((*s2)[0][1][0][0][0][0][1][1], Catch::Matchers::WithinAbs(a011_inv[0][1], 1e-4));
        REQUIRE_THAT((*s2)[1][0][0][0][0][0][1][1], Catch::Matchers::WithinAbs(a011_inv[1][0], 1e-4));
        REQUIRE_THAT((*s2)[1][1][0][0][0][0][1][1], Catch::Matchers::WithinAbs(a011_inv[1][1], 1e-4));
        REQUIRE_THAT((*s2)[0][0][0][0][0][0][1][2], Catch::Matchers::WithinAbs(a012_inv[0][0], 1e-4));
        REQUIRE_THAT((*s2)[0][1][0][0][0][0][1][2], Catch::Matchers::WithinAbs(a012_inv[0][1], 1e-4));
        REQUIRE_THAT((*s2)[1][0][0][0][0][0][1][2], Catch::Matchers::WithinAbs(a012_inv[1][0], 1e-4));
        REQUIRE_THAT((*s2)[1][1][0][0][0][0][1][2], Catch::Matchers::WithinAbs(a012_inv[1][1], 1e-4));
    }

    // Check that inversion of a singular matrix returns nullptr.
    SECTION("singular")
    {
        s[0][0][1][1][1][ghm][ghn][gho] = 0;
        s[0][1][1][1][1][ghm][ghn][gho] = 0;

        REQUIRE_THROWS(s.invertCenterMatrices());
    }
}

// TODO
TEST_CASE("Blockstencil::invertDiagonal")
{
    int m = 1;
    int n = 2;
    int o = 3;
    int ghm = GENERATE(0, 1);
    int ghn = 1;
    int gho = 1;
    int width = 3;
    int blocksize = 2;

    mgcl::Blockstencil s(m, n, o, width, blocksize, ghm, ghn, gho);
    // fill with 1d real cell indices, as if there were no ghosts
    int cnt = 0;
    for (int d1 = s.getGhostsDim1(); d1 < s.getDim1() + s.getGhostsDim1(); d1++)
        for (int d2 = s.getGhostsDim2(); d2 < s.getDim2() + s.getGhostsDim2(); d2++)
            for (int d3 = s.getGhostsDim3(); d3 < s.getDim3() + s.getGhostsDim3(); d3++)
                for (int d4 = s.getGhostsDim4(); d4 < s.getDim4() + s.getGhostsDim4(); d4++)
                    for (int d5 = s.getGhostsDim5(); d5 < s.getDim5() + s.getGhostsDim5(); d5++)
                        for (int d6 = s.getGhostsDim6(); d6 < s.getDim6() + s.getGhostsDim6(); d6++)
                            for (int d7 = s.getGhostsDim7(); d7 < s.getDim7() + s.getGhostsDim7(); d7++)
                                for (int d8 = s.getGhostsDim8(); d8 < s.getDim8() + s.getGhostsDim8(); d8++)
                                {
                                    s[d1][d2][d3][d4][d5][d6][d7][d8] = cnt++;
                                }
    // s.dumpToFile("s.txt");

    SECTION("success")
    {
        auto s2 = s.invertDiagonal();

        REQUIRE(s2->getM() == m);
        REQUIRE(s2->getN() == n);
        REQUIRE(s2->getO() == o);
        REQUIRE(s2->getBlocksize() == blocksize);

        for (int i = ghm; i < m + ghm; i++)
            for (int j = ghn; j < n + ghn; j++)
                for (int k = gho; k < o + gho; k++)
                    for (size_t b = 0; b < blocksize; b++)
                    {
                        CAPTURE(i, j, k, b);
                        REQUIRE_THAT((*s2)[i - ghm][j - ghn][k - gho][b], Catch::Matchers::WithinAbs(1.0 / s[b][b][width / 2][width / 2][width / 2][i][j][k], 1e-4));
                    }
    }

    // Check that inversion of a singular matrix returns nullptr.
    SECTION("singular")
    {
        s[0][0][1][1][1][ghm][ghn][gho] = 0;
        s[0][1][1][1][1][ghm][ghn][gho] = 0;

        REQUIRE_THROWS(s.invertDiagonal());
    }
}