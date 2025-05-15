
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <memory>

#include "../../src/mgcl/blockstencil.hpp"
#include "../../src/mgcl/blockstencil_gpu.hpp"
#include "../../src/mgcl/cuboid.hpp"
#include "../../src/mgcl/multigrid_engine.hpp"
#include "../../src/mgcl/problem.hpp"

#include "../cli_args.hpp"
#include "../device_type_generator.hpp"

TEST_CASE("BlockstencilGpu::updateGhosts")
{
    std::shared_ptr<mgcl::Cuboid> v_dummy = std::make_shared<mgcl::Cuboid>(1, 1, 1);
    std::shared_ptr<mgcl::Cuboid> f_dummy = std::make_shared<mgcl::Cuboid>(1, 1, 1);
    mgcl::Problem p(1, 1, 1, f_dummy, v_dummy);
    p.setUseOpencl(true);
    p.init();

    SECTION("m,n,o >= gh")
    {
        // regular ghost update
        int m = 2;
        int n = 3;
        int o = 4;
        // int m = GENERATE(2, 3, 4);
        // int n = GENERATE(2, 3, 4);
        // int o = GENERATE(2, 3, 4);

        int gh = 1; // GENERATE(1, 2);
        int blocksize = 2;
        int width = 3; // GENERATE(3, 5);

        mgcl::Blockstencil h_a(m, n, o, width, blocksize, gh, gh, gh);
        h_a.fillRandomInt();
        mgcl::BlockstencilGpu d_a(h_a, p.getContext(), p.getCommands(), p.getProgram());

        h_a.updateGhostsLocally();
        d_a.updateGhostsLocally(p.getProgram(), p.getCommands(), &p.getKernelConfig(), p.getProfilingData());

        p.finish();
        // h_a.dumpToFile("ha.txt");
        // d_a.dumpToFile(p.getCommands(), "da.txt", false);

        REQUIRE(d_a.isEqualIncGhosts(p.getCommands(), h_a));
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

        mgcl::Blockstencil h_a(m, n, o, width, blocksize, gh, gh, gh);
        h_a.fillRandomInt();
        mgcl::BlockstencilGpu d_a(h_a, p.getContext(), p.getCommands(), p.getProgram());

        h_a.updateGhostsLocally();
        d_a.updateGhostsLocally(p.getProgram(), p.getCommands(), &p.getKernelConfig(), p.getProfilingData());

        REQUIRE(d_a.isEqualIncGhosts(p.getCommands(), h_a));
    }

    SECTION("mixed")
    {
        // mixed periodic and regular ghost update, e.g. m > gh but n < gh
        int m = 2;
        int n = 2;
        int o = 3;

        int gh = GENERATE(1, 2);

        // int m = GENERATE(2, 3);
        // int n = GENERATE(2, 3);
        // int o = GENERATE(2, 3);
        //
        // int ghm = GENERATE(1, 3, 5);
        // int ghn = GENERATE(1, 3, 5);
        // int gho = GENERATE(1, 3, 5);

        int blocksize = 2;
        int width = GENERATE(3, 5);

        mgcl::Blockstencil h_a(m, n, o, width, blocksize, gh, gh, gh);
        h_a.fillRandomInt();
        mgcl::BlockstencilGpu d_a(h_a, p.getContext(), p.getCommands(), p.getProgram());

        h_a.updateGhostsLocally();
        d_a.updateGhostsLocally(p.getProgram(), p.getCommands(), &p.getKernelConfig(), p.getProfilingData());
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

// TEST_CASE("Blockstencil::slice")
// {
//     int m = 4;
//     int n = 4;
//     int o = 4;
//     int blocksize = 2;

//     mgcl::Blockstencil cb(m, n, o, 3, blocksize, 1, 1, 1);
//     // cb.fillRandom();
//     cb.fill1dIndex(true);

//     SECTION("throwing")
//     {
//         REQUIRE_THROWS(cb.slice(-1, 0, 0, 0, 0, 0));
//         REQUIRE_THROWS(cb.slice(0, 0, -1, 0, 0, 0));
//         REQUIRE_THROWS(cb.slice(0, 0, 0, 0, -1, 0));

//         REQUIRE_THROWS(cb.slice(0, m + 3, 0, 0, 0, 0));
//         REQUIRE_THROWS(cb.slice(0, 0, 0, n + 3, 0, 0));
//         REQUIRE_THROWS(cb.slice(0, 0, 0, 0, 0, n + 3));
//     }

//     SECTION("success")
//     {
//         auto cs = cb.slice(0, 1, 0, 2, 2, 3);

//         REQUIRE(cs->getM() == 2);
//         REQUIRE(cs->getN() == 3);
//         REQUIRE(cs->getO() == 2);
//         REQUIRE(cs->getWidth() == cb.getWidth());
//         REQUIRE(cs->getBlocksize() == cb.getBlocksize());
//         REQUIRE(cs->getGhostsM() == cb.getGhostsM());
//         REQUIRE(cs->getGhostsN() == cb.getGhostsN());
//         REQUIRE(cs->getGhostsO() == cb.getGhostsO());

//         for (int d1 = cs->getGhostsM(); d1 < cs->getM() + cs->getGhostsM(); d1++)
//             for (int d2 = cs->getGhostsN(); d2 < cs->getN() + cs->getGhostsN(); d2++)
//                 for (int d3 = cs->getGhostsO(); d3 < cs->getO() + cs->getGhostsO(); d3++)
//                     for (int d4 = 0; d4 < cs->getWidth(); d4++)
//                         for (int d5 = 0; d5 < cs->getWidth(); d5++)
//                             for (int d6 = 0; d6 < cs->getWidth(); d6++)
//                                 for (int d7 = 0; d7 < cs->getBlocksize(); d7++)
//                                     for (int d8 = 0; d8 < cs->getBlocksize(); d8++)
//                                     {
//                                         CAPTURE(d1, d2, d3, d4, d5, d6, d7, d8);
//                                         REQUIRE(cs->getData()[d7][d8][d4][d5][d6][d1][d2][d3] == cb[d7][d8][d4][d5][d6][d1][d2][d3 + 2]);
//                                     }
//     }
// }

// TEST_CASE("Blockstencil::sliceIncGhosts")
// {
//     int m = 4;
//     int n = 4;
//     int o = 4;
//     int blocksize = 2;

//     mgcl::Blockstencil cb(m, n, o, 3, blocksize, 1, 1, 1);
//     cb.fillRandom();

//     SECTION("throwing")
//     {
//         REQUIRE_THROWS(cb.sliceIncGhosts(-1, 0, 0, 0, 0, 0));
//         REQUIRE_THROWS(cb.sliceIncGhosts(0, 0, -1, 0, 0, 0));
//         REQUIRE_THROWS(cb.sliceIncGhosts(0, 0, 0, 0, -1, 0));

//         REQUIRE_THROWS(cb.sliceIncGhosts(0, m + 3, 0, 0, 0, 0));
//         REQUIRE_THROWS(cb.sliceIncGhosts(0, 0, 0, n + 3, 0, 0));
//         REQUIRE_THROWS(cb.sliceIncGhosts(0, 0, 0, 0, 0, n + 3));
//     }

//     SECTION("success")
//     {
//         auto cs = cb.sliceIncGhosts(0, 1, 0, 2, 2, 3);

//         REQUIRE(cs->getM() == 2);
//         REQUIRE(cs->getN() == 3);
//         REQUIRE(cs->getO() == 2);
//         REQUIRE(cs->getWidth() == cb.getWidth());
//         REQUIRE(cs->getBlocksize() == cb.getBlocksize());
//         REQUIRE(cs->getGhostsM() == 0);
//         REQUIRE(cs->getGhostsN() == 0);
//         REQUIRE(cs->getGhostsO() == 0);

//         for (int d1 = 0; d1 < cs->getMgh(); d1++)
//             for (int d2 = 0; d2 < cs->getNgh(); d2++)
//                 for (int d3 = 0; d3 < cs->getOgh(); d3++)
//                     for (int d4 = 0; d4 < cs->getWidth(); d4++)
//                         for (int d5 = 0; d5 < cs->getWidth(); d5++)
//                             for (int d6 = 0; d6 < cs->getWidth(); d6++)
//                                 for (int d7 = 0; d7 < cs->getBlocksize(); d7++)
//                                     for (int d8 = 0; d8 < cs->getBlocksize(); d8++)
//                                     {
//                                         REQUIRE(cs->getData()[d7][d8][d4][d5][d6][d1][d2][d3] == cb[d7][d8][d4][d5][d6][d1][d2][d3 + 2]);
//                                     }
//     }
// }

// TEST_CASE("Blockstencil::copyShallow")
// {
//     mgcl::Blockstencil s1(2, 3, 4, 3, 2, 5, 6, 7);
//     s1.fillRandom();
//     auto s2 = s1.copyShallow();

//     REQUIRE(s1.getM() == s2->getM());
//     REQUIRE(s1.getN() == s2->getN());
//     REQUIRE(s1.getO() == s2->getO());
//     REQUIRE(s1.getWidth() == s2->getWidth());
//     REQUIRE(s1.getBlocksize() == s2->getBlocksize());
//     REQUIRE(s1.getGhostsM() == s2->getGhostsM());
//     REQUIRE(s1.getGhostsN() == s2->getGhostsN());
//     REQUIRE(s1.getGhostsO() == s2->getGhostsO());
//     REQUIRE(s1.getGhostsDim4() == s2->getGhostsDim4());
//     REQUIRE(s1.getGhostsDim5() == s2->getGhostsDim5());
//     REQUIRE(s1.getGhostsDim6() == s2->getGhostsDim6());
//     REQUIRE(s1.getGhostsDim7() == s2->getGhostsDim7());
//     REQUIRE(s1.getGhostsDim8() == s2->getGhostsDim8());
//     REQUIRE(s1.getSize() == s2->getSize());
// }

TEST_CASE("BlockstencilGpu::extract_border_planes")
{

    auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));

    // Create dummy problem
    auto v = std::make_shared<mgcl::Cuboid>(1, 1, 1);
    auto f = std::make_shared<mgcl::Cuboid>(1, 1, 1);
    mgcl::Problem p(1, 1, 1, f, v);
    p.setUseOpencl(true);
    p.setDeviceType(deviceType);
    p.setProfilingEnabled(true);
    p.init();

    int m = 3;
    int n = 5;
    int o = 7;
    int ghosts_m = 1;
    int ghosts_n = 1; // BlockstencilGpu currently does not support different ghosts per dimension
    int ghosts_o = 1;
    int mgh = m + 2 * ghosts_m;
    int ngh = n + 2 * ghosts_n;
    int ogh = o + 2 * ghosts_o;
    int yz = ngh * ogh;
    int xz = mgh * ogh;
    int xy = mgh * ngh;
    int blocksize = 2;
    int ressize = (2 * yz * ghosts_m + 2 * xz * ghosts_n + 2 * xy * ghosts_o) * 27 * blocksize * blocksize;

    mgcl::Blockstencil h_stencil(m, n, o, 3, blocksize, ghosts_m, ghosts_n, ghosts_o);
    h_stencil.fill1dIndex(false);
    const double* buf_stencil = h_stencil.field1d().data();

    mgcl::BlockstencilGpu d_stencil(h_stencil, p.getContext(), p.getCommands(), p.getProgram());

    std::vector<double> h_ret(ressize, -1);
    mgcl::BufferGpu d_tmp(p.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, h_ret);

    d_stencil.extractBorderPlanes(p.getCommands(), p.getProgram(), d_tmp, h_ret, &p.getKernelConfig(), p.getProfilingData());
    p.finish();

    // Check that every index was written to
    for (size_t i = 0; i < ressize; i++)
    {
        CAPTURE(i, ressize);
        REQUIRE(h_ret[i] >= 0);
    }

    int cnt = 0;
    // front planes (yz)
    for (int bi = 0; bi < blocksize; bi++)
        for (int bj = 0; bj < blocksize; bj++)
            for (int ii = 0; ii < 3; ii++)
                for (int jj = 0; jj < 3; jj++)
                    for (int kk = 0; kk < 3; kk++)
                        for (int i = ghosts_m; i < 2 * ghosts_m; i++) // ghm real planes in the front
                            for (int j = 0; j < ngh; j++)             // all cells in y-dir
                                for (int k = 0; k < ogh; k++)         // all cells in z-dir
                                {
                                    CAPTURE(i, j, k, ii, jj, kk, bi, bj, cnt);
                                    REQUIRE(h_ret[cnt++] == h_stencil[bi][bj][ii][jj][kk][i][j][k]);
                                }

    // back planes (yz)
    for (int bi = 0; bi < blocksize; bi++)
        for (int bj = 0; bj < blocksize; bj++)
            for (int ii = 0; ii < 3; ii++)
                for (int jj = 0; jj < 3; jj++)
                    for (int kk = 0; kk < 3; kk++)
                        for (int i = m; i < m + ghosts_m; i++) // ghosts_m real planes in the back
                            for (int j = 0; j < ngh; j++)      // all cells in y-dir
                                for (int k = 0; k < ogh; k++)  // all cells in z-dir
                                    REQUIRE(h_ret[cnt++] == h_stencil[bi][bj][ii][jj][kk][i][j][k]);

    // top planes (xz)
    for (int bi = 0; bi < blocksize; bi++)
        for (int bj = 0; bj < blocksize; bj++)
            for (int ii = 0; ii < 3; ii++)
                for (int jj = 0; jj < 3; jj++)
                    for (int kk = 0; kk < 3; kk++)
                        for (int j = ghosts_n; j < 2 * ghosts_n; j++)
                            for (int i = 0; i < mgh; i++)
                                for (int k = 0; k < ogh; k++)
                                    REQUIRE(h_ret[cnt++] == h_stencil[bi][bj][ii][jj][kk][i][j][k]);

    // bottom planes (xz)
    for (int bi = 0; bi < blocksize; bi++)
        for (int bj = 0; bj < blocksize; bj++)
            for (int ii = 0; ii < 3; ii++)
                for (int jj = 0; jj < 3; jj++)
                    for (int kk = 0; kk < 3; kk++)
                        for (int j = n; j < n + ghosts_n; j++)
                            for (int i = 0; i < mgh; i++)
                                for (int k = 0; k < ogh; k++)
                                    REQUIRE(h_ret[cnt++] == h_stencil[bi][bj][ii][jj][kk][i][j][k]);

    // left planes (xy)
    for (int bi = 0; bi < blocksize; bi++)
        for (int bj = 0; bj < blocksize; bj++)
            for (int ii = 0; ii < 3; ii++)
                for (int jj = 0; jj < 3; jj++)
                    for (int kk = 0; kk < 3; kk++)
                        for (int k = ghosts_o; k < 2 * ghosts_o; k++)
                            for (int i = 0; i < mgh; i++)
                                for (int j = 0; j < ngh; j++)
                                    REQUIRE(h_ret[cnt++] == h_stencil[bi][bj][ii][jj][kk][i][j][k]);

    // right planes (xy)
    for (int bi = 0; bi < blocksize; bi++)
        for (int bj = 0; bj < blocksize; bj++)
            for (int ii = 0; ii < 3; ii++)
                for (int jj = 0; jj < 3; jj++)
                    for (int kk = 0; kk < 3; kk++)
                        for (int k = o; k < o + ghosts_o; k++)
                            for (int i = 0; i < mgh; i++)
                                for (int j = 0; j < ngh; j++)
                                    REQUIRE(h_ret[cnt++] == h_stencil[bi][bj][ii][jj][kk][i][j][k]);
}

TEST_CASE("BlockstencilGpu::pasteGhostsFromBorderPlanes")
{
    int m = 3;
    int n = 5;
    int o = 7;
    int ghosts_m = 1;
    int ghosts_n = 1; // 2;
    int ghosts_o = 1; // 3;
    int mgh = m + 2 * ghosts_m;
    int ngh = n + 2 * ghosts_n;
    int ogh = o + 2 * ghosts_o;
    int yz = ngh * ogh;
    int xz = mgh * ogh;
    int xy = mgh * ngh;
    int gridsize = mgh * ngh * ogh;
    int blocksize = 2;
    int ressize = (2 * yz * ghosts_m + 2 * xz * ghosts_n + 2 * xy * ghosts_o) * 27 * blocksize * blocksize;

    auto checkResult = [&](mgcl::Blockstencil& h_stencil, double* buf_ghosts)
    {
        // Check that all ghost cells were filled with any value and all real cells were left untouched
        for (int bi = 0; bi < blocksize; bi++)
            for (int bj = 0; bj < blocksize; bj++)
                for (int ii = 0; ii < 3; ii++)
                    for (int jj = 0; jj < 3; jj++)
                        for (int kk = 0; kk < 3; kk++)
                            for (int i = 0; i < mgh; i++)
                                for (int j = 0; j < ngh; j++)
                                    for (int k = 0; k < ogh; k++)
                                    {
                                        CAPTURE(i, j, k, ii, jj, kk, bi, bj);
                                        if ((i < ghosts_m || i >= m + ghosts_m) || (j < ghosts_n || j >= n + ghosts_n) || (k < ghosts_o || k >= o + ghosts_o))
                                        {
                                            REQUIRE(h_stencil[bi][bj][ii][jj][kk][i][j][k] >= 0);
                                            REQUIRE(!std::isnan(h_stencil[bi][bj][ii][jj][kk][i][j][k]));
                                        }
                                        else
                                        {
                                            REQUIRE(h_stencil[bi][bj][ii][jj][kk][i][j][k] == -1);
                                        }
                                    }

        int cnt = 0;
        // back ghosts (yz)
        for (int bi = 0; bi < blocksize; bi++)
            for (int bj = 0; bj < blocksize; bj++)
                for (int ii = 0; ii < 3; ii++)
                    for (int jj = 0; jj < 3; jj++)
                        for (int kk = 0; kk < 3; kk++)
                            for (int i = m + ghosts_m; i < mgh; i++) // ghosts_m real planes in the back
                                for (int j = 0; j < ngh; j++)        // all cells in y-dir
                                    for (int k = 0; k < ogh; k++)    // all cells in z-dir
                                    {
                                        // No corners or edges, only ghosts directly adjacent to real back face
                                        if (j >= ghosts_n && j < n + ghosts_n && k >= ghosts_o && k < o + ghosts_o)
                                        {
                                            CAPTURE(i, j, k, ii, jj, kk, cnt);
                                            REQUIRE_THAT(buf_ghosts[cnt], Catch::Matchers::WithinAbs(h_stencil[bi][bj][ii][jj][kk][i][j][k], 1e-15));
                                        }
                                        cnt++;
                                    }

        // front ghosts (yz)
        for (int bi = 0; bi < blocksize; bi++)
            for (int bj = 0; bj < blocksize; bj++)
                for (int ii = 0; ii < 3; ii++)
                    for (int jj = 0; jj < 3; jj++)
                        for (int kk = 0; kk < 3; kk++)
                            for (int i = 0; i < ghosts_m; i++)    // ghm ghost planes in the front
                                for (int j = 0; j < ngh; j++)     // all cells in y-dir
                                    for (int k = 0; k < ogh; k++) // all cells in z-dir
                                    {                             // No corners or edges, only ghosts directly adjacent to real back face
                                        if (j >= ghosts_n && j < n + ghosts_n && k >= ghosts_o && k < o + ghosts_o)
                                        {
                                            CAPTURE(i, j, k, ii, jj, kk, cnt);
                                            REQUIRE_THAT(buf_ghosts[cnt], Catch::Matchers::WithinAbs(h_stencil[bi][bj][ii][jj][kk][i][j][k], 1e-15));
                                        }
                                        cnt++;
                                    }

        // bottom ghosts (xz)
        for (int bi = 0; bi < blocksize; bi++)
            for (int bj = 0; bj < blocksize; bj++)
                for (int ii = 0; ii < 3; ii++)
                    for (int jj = 0; jj < 3; jj++)
                        for (int kk = 0; kk < 3; kk++)
                            for (int j = n + ghosts_n; j < ngh; j++)
                                for (int i = 0; i < mgh; i++)
                                    for (int k = 0; k < ogh; k++)
                                    {
                                        // Ignore left and right ghost cells, but include front and back ghosts
                                        if (k >= ghosts_o && k < o + ghosts_o)
                                        {
                                            CAPTURE(i, j, k, ii, jj, kk, cnt);
                                            REQUIRE_THAT(buf_ghosts[cnt], Catch::Matchers::WithinAbs(h_stencil[bi][bj][ii][jj][kk][i][j][k], 1e-15));
                                        }
                                        cnt++;
                                    }

        // top ghosts (xz)
        for (int bi = 0; bi < blocksize; bi++)
            for (int bj = 0; bj < blocksize; bj++)
                for (int ii = 0; ii < 3; ii++)
                    for (int jj = 0; jj < 3; jj++)
                        for (int kk = 0; kk < 3; kk++)
                            for (int j = 0; j < ghosts_n; j++)
                                for (int i = 0; i < mgh; i++)
                                    for (int k = 0; k < ogh; k++)
                                    {
                                        // Ignore left and right ghost cells, but include front and back ghosts
                                        if (k >= ghosts_o && k < o + ghosts_o)
                                        {
                                            CAPTURE(i, j, k, ii, jj, kk, cnt);
                                            REQUIRE_THAT(buf_ghosts[cnt], Catch::Matchers::WithinAbs(h_stencil[bi][bj][ii][jj][kk][i][j][k], 1e-15));
                                        }
                                        cnt++;
                                    }

        // right ghosts (xy)
        for (int bi = 0; bi < blocksize; bi++)
            for (int bj = 0; bj < blocksize; bj++)
                for (int ii = 0; ii < 3; ii++)
                    for (int jj = 0; jj < 3; jj++)
                        for (int kk = 0; kk < 3; kk++)
                            for (int k = o + ghosts_o; k < ogh; k++)
                                for (int i = 0; i < mgh; i++)
                                    for (int j = 0; j < ngh; j++)
                                    {
                                        CAPTURE(i, j, k, ii, jj, kk, cnt);
                                        REQUIRE_THAT(buf_ghosts[cnt++], Catch::Matchers::WithinAbs(h_stencil[bi][bj][ii][jj][kk][i][j][k], 1e-15));
                                    }

        // left ghosts (xy)
        for (int bi = 0; bi < blocksize; bi++)
            for (int bj = 0; bj < blocksize; bj++)
                for (int ii = 0; ii < 3; ii++)
                    for (int jj = 0; jj < 3; jj++)
                        for (int kk = 0; kk < 3; kk++)
                            for (int k = 0; k < ghosts_o; k++)
                                for (int i = 0; i < mgh; i++)
                                    for (int j = 0; j < ngh; j++)
                                    {
                                        CAPTURE(i, j, k, ii, jj, kk, cnt);
                                        REQUIRE_THAT(buf_ghosts[cnt++], Catch::Matchers::WithinAbs(h_stencil[bi][bj][ii][jj][kk][i][j][k], 1e-15));
                                    }
    };

    auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));

    // Create dummy problem
    auto v = std::make_shared<mgcl::Cuboid>(1, 1, 1);
    auto f = std::make_shared<mgcl::Cuboid>(1, 1, 1);
    mgcl::Problem p(1, 1, 1, f, v);
    p.setUseOpencl(true);
    p.setDeviceType(deviceType);
    p.init();

    mgcl::Blockstencil h_stencil(m, n, o, 3, blocksize, ghosts_m, ghosts_n, ghosts_o);
    h_stencil.fill(-1, false);

    // 1d ghosts buffer, filled with 1d index
    std::vector<double> h_planes(ressize);
    for (size_t i = 0; i < ressize; i++)
    {
        h_planes[i] = i;
    }

    mgcl::BlockstencilGpu d_stencil(h_stencil, p.getContext(), p.getCommands(), p.getProgram());
    mgcl::BufferGpu d_planes(p.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, h_planes);

    // paste planes
    d_stencil.pasteGhostsFromBorderPlanes(p.getCommands(), p.getProgram(), d_planes, nullptr, nullptr);
    p.finish();

    // read result
    d_stencil.read(p.getCommands(), true, h_stencil);

    checkResult(h_stencil, h_planes.data());
}

TEST_CASE("BlockstencilGpu::invertDiagonal")
{
    auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));

    int m = 4;
    int n = 4;
    int o = 4;
    int gh = 1;
    int blocksize = 2;

    // create dummy problem
    auto v_dummy = std::make_shared<mgcl::Cuboid>(1, 1, 1);
    auto f_dummy = std::make_shared<mgcl::Cuboid>(1, 1, 1);
    mgcl::Problem p(1, 1, 1, f_dummy, v_dummy);
    p.setUseOpencl(true);
    p.setDeviceType(deviceType);
    p.setProfilingEnabled(true);
    p.init();

    mgcl::Blockstencil bs(m, n, o, 3, blocksize, gh, gh, gh);
    bs.fill1dIndex(true);
    auto bs_inv = bs.invertDiagonal();
    REQUIRE(bs_inv);

    mgcl::BlockstencilGpu d_bs(bs, p.getContext(), p.getCommands(), p.getProgram());

    auto d_bs_inv = d_bs.invertDiagonal(p.getContext(), p.getCommands(), p.getProgram());
    p.finish();

    auto d_bs_inv_test = d_bs_inv->read(p.getCommands(), true);

    REQUIRE(d_bs_inv_test.isEqual(*bs_inv));
}