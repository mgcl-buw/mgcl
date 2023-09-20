#include "matrix2d.hpp"

#include <cmath>

#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"

#include "../src/level.hpp"
#include "../src/multigrid_engine.hpp"
#include "../src/stencil.hpp"

#include "test_utility.hpp"

TEST_CASE("galerkin Laplace vs Matrix")
{
    int m = GENERATE(2, 4, 8);
    int n = GENERATE(2, 4, 8);
    int o = GENERATE(2, 4, 8);

    // Fill varying stencil on fine grid with 7p Laplace
    mgcl::VaryingStencil a_h(m, n, o, 3, 2, 2, 2);
    for (int i = 0; i < m + 4; i++)
        for (int j = 0; j < n + 4; j++)
            for (int k = 0; k < o + 4; k++)
            {
                // 7-point Laplace
                a_h[i][j][k][0][1][1] = 1;
                a_h[i][j][k][1][0][1] = 1;
                a_h[i][j][k][1][1][0] = 1;
                a_h[i][j][k][1][1][1] = -6;
                a_h[i][j][k][1][1][2] = 1;
                a_h[i][j][k][1][2][1] = 1;
                a_h[i][j][k][2][1][1] = 1;
            }

    auto a_2h = mgcl::MultigridEngine::galerkin(a_h, nullptr, nullptr, true, true, true, false);
    auto a2hm = mgcl_test::Matrix2d::fromVaryingStencil(a_2h, true);

    // calculate results with Matrices to check
    auto ah = mgcl_test::Matrix2d::laplace7p3d(m, n, o);
    auto r = mgcl_test::Matrix2d::restrictionFullWeight(m, n, o);
    auto p = mgcl_test::Matrix2d::prolongationBilinear(m, n, o);

    auto a2h = r * ah * p;

    REQUIRE(a2hm == a2h);
}

TEST_CASE("galerkin random values periodic vs Matrix")
{
    int m = GENERATE(2, 4, 8);
    int n = GENERATE(2, 4, 8);
    int o = GENERATE(2, 4, 8);

    double tol = 1e-12;

    // Fill varying stencil on fine grid with 27p random values
    mgcl::VaryingStencil a_h(m, n, o, 3, 2, 2, 2);
    a_h.fillRandom(-10, 10);
    a_h.updateGhosts();

    auto a_2h = mgcl::MultigridEngine::galerkin(a_h, nullptr, nullptr, true, true, true, false);
    auto a2hm = mgcl_test::Matrix2d::fromVaryingStencil(a_2h, true);

    // calculate results with Matrices to check
    auto ah = mgcl_test::Matrix2d::fromVaryingStencil(a_h, true);
    auto r = mgcl_test::Matrix2d::restrictionFullWeight(m, n, o);
    auto p = mgcl_test::Matrix2d::prolongationBilinear(m, n, o);

    auto a2h = r * ah * p;

    CHECK(a2hm.isEqual(a2h, tol));
}

TEST_CASE("GPU galerkin random values periodic")
{
    auto deviceType = GENERATE(CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU);

    if (!mgcl_test::TestUtility::deviceAvailable("", deviceType))
    {
        std::string typeName = deviceType == CL_DEVICE_TYPE_GPU ? "CL_DEVICE_TYPE_GPU" : "CL_DEVICE_TYPE_CPU";
        std::cout << "Skipping non-available device type '" << typeName << "'" << std::endl;
        return;
    }

    mgcl_test::TestUtility t(deviceType);

    int m = GENERATE(2, 4, 8);
    int n = GENERATE(4, 8);
    int o = GENERATE(2, 4);

    double tol = 1e-12;
    int gh = 2;

    // Fill varying stencil on fine grid with 27p random values
    mgcl::VaryingStencilGpu a_h_gpu(m, n, o, 3, gh, t.getContext(), t.getCommands());
    mgcl::VaryingStencil a_h(m, n, o, 3, gh, gh, gh);
    a_h.fillRandom(-10, 10);
    a_h.updateGhosts();
    a_h_gpu.fill(a_h, t.getCommands(), true);

    auto a_2h = mgcl::MultigridEngine::galerkin(a_h, nullptr, nullptr, true, true, true, false);
    auto a_2h_gpu = mgcl::MultigridEngine::galerkin(a_h_gpu, t.getProgram(), t.getCommands(), t.getContext(),
                                                    nullptr, nullptr, true, true, true, false);
    t.finish();

    auto ret = a_2h_gpu.read(t.getCommands(), true);

    REQUIRE(a_2h.isEqual(ret, tol));
}

TEST_CASE("galerkin multiple levels random values periodic")
{
    int m = GENERATE(2, 4, 8);
    int n = GENERATE(2, 4, 8);
    int o = GENERATE(2, 4, 8);

    double tol = 1e-12;

    int minsize = m < n ? m : n;
    minsize = minsize < o ? minsize : o;
    int maxlv = log2(minsize);

    std::cout << "Testing for m,n,o with maxlv: " << m << "," << n << "," << o << ", " << maxlv << std::endl;

    // Fill varying stencil on fine grid with 27p random values initially
    auto a_h = std::make_unique<mgcl::VaryingStencil>(m, n, o, 3, 2, 2, 2);
    a_h->fillRandom(-10, 10);
    a_h->updateGhosts();

    std::unique_ptr<mgcl::VaryingStencil> a_2h = nullptr;

    for (int lv = 0; lv < maxlv; lv++)
    {
        a_2h = std::make_unique<mgcl::VaryingStencil>(mgcl::MultigridEngine::galerkin(*a_h, nullptr, nullptr, true, true, true, false));
        auto a2hm = mgcl_test::Matrix2d::fromVaryingStencil(*a_2h, true);

        // calculate results with Matrices to check
        auto ah = mgcl_test::Matrix2d::fromVaryingStencil(*a_h, true);
        auto r = mgcl_test::Matrix2d::restrictionFullWeight(m >> lv, n >> lv, o >> lv);
        auto p = mgcl_test::Matrix2d::prolongationBilinear(m >> lv, n >> lv, o >> lv);

        auto a2h = r * ah * p;

        REQUIRE(a2hm.isEqual(a2h, tol));

        // a_2h is a_h for next level
        a_h = std::move(a_2h);
    }
}

// If the fine grid operator is symmetric, the coarse grid operator shall be too (see "A Multigrid tutorial")
TEST_CASE("galerkin symmetric")
{
    int m = GENERATE(2, 4, 8);
    int n = GENERATE(2, 4, 8);
    int o = GENERATE(2, 4, 8);

    // Fill varying stencil on fine grid with 7p Laplace
    mgcl::VaryingStencil a_h(m, n, o, 3, 2, 2, 2);
    for (int i = 0; i < m + 4; i++)
        for (int j = 0; j < n + 4; j++)
            for (int k = 0; k < o + 4; k++)
            {
                // 7-point Laplace
                a_h[i][j][k][0][1][1] = 1;
                a_h[i][j][k][1][0][1] = 1;
                a_h[i][j][k][1][1][0] = 1;
                a_h[i][j][k][1][1][1] = -6;
                a_h[i][j][k][1][1][2] = 1;
                a_h[i][j][k][1][2][1] = 1;
                a_h[i][j][k][2][1][1] = 1;
            }

    auto a_2h = mgcl::MultigridEngine::galerkin(a_h, nullptr, nullptr, true, true, true, false);
    auto a2hm = mgcl_test::Matrix2d::fromVaryingStencil(a_2h, true);

    for (int i = 0; i < a2hm.getM(); i++)
        for (int j = 0; j < a2hm.getN(); j++)
        {
            REQUIRE(a2hm[i][j] == a2hm[j][i]);
        }
}

// Checks if SA == AS which it should since S = S^T and A = A^T
TEST_CASE("galerkin Laplace SA == AS")
{
    int m = 8; // GENERATE(2, 4, 8);
    int n = 8; // GENERATE(2, 4, 8);
    int o = 8; // GENERATE(2, 4, 8);
    double h = 1.0 / (double)m;
    double h2inv = static_cast<double>(m * m);

    // Fill varying stencil on fine grid with 7p Laplace
    mgcl::VaryingStencil a(m, n, o, 3, 2, 2, 2);
    for (int i = 0; i < m + 4; i++)
        for (int j = 0; j < n + 4; j++)
            for (int k = 0; k < o + 4; k++)
            {
                // 7-point Laplace
                a[i][j][k][0][1][1] = h2inv * 1.0;
                a[i][j][k][1][0][1] = h2inv * 1.0;
                a[i][j][k][1][1][0] = h2inv * 1.0;
                a[i][j][k][1][1][1] = h2inv * -6.0;
                a[i][j][k][1][1][2] = h2inv * 1.0;
                a[i][j][k][1][2][1] = h2inv * 1.0;
                a[i][j][k][2][1][1] = h2inv * 1.0;
            }

    auto s = mgcl::create3dFullWeightRestrictionStencil();

    auto as = a.multiply(s, 2, nullptr, true, true);
    auto sa = s.multiply(a, 2, nullptr, true, true);

    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < o; k++)
                for (int ii = 0; ii < as.getDim4gh(); ii++)
                    for (int jj = 0; jj < as.getDim5gh(); jj++)
                        for (int kk = 0; kk < as.getDim6gh(); kk++)
                        {
                            REQUIRE(as[i][j][k][ii][jj][kk] == sa[i][j][k][ii][jj][kk]);
                        }
}

/**
 * For the Laplace operator the Galerkin coarse grid operator shall be equal to the rediscretization of the Laplace
 * operator on the coarse grid (using 2nd order differences, see "A Multigrid Tutorial", p. 79) in 1d.
 * In higher dimensions the full stencil is being used, i.e. a 7p stencil on the fine grid results in a 27p stencil on
 * the coarse grid with constant factors, depending just on h. The factors were calculated using Matlab.
 */
TEST_CASE("galerkin Laplace rediscretized")
{
    int m = 8; // GENERATE(2, 4, 8);
    int n = 8; // GENERATE(2, 4, 8);
    int o = 8; // GENERATE(2, 4, 8);
    double h = 1.0 / (double)m;
    double h2inv = static_cast<double>(m * m);

    // Fill varying stencil on fine grid with 7p Laplace
    mgcl::VaryingStencil a_h(m, n, o, 3, 2, 2, 2);
    for (int i = 0; i < m + 4; i++)
        for (int j = 0; j < n + 4; j++)
            for (int k = 0; k < o + 4; k++)
            {
                // 7-point Laplace
                a_h[i][j][k][0][1][1] = h2inv * 1.0;
                a_h[i][j][k][1][0][1] = h2inv * 1.0;
                a_h[i][j][k][1][1][0] = h2inv * 1.0;
                a_h[i][j][k][1][1][1] = h2inv * -6.0;
                a_h[i][j][k][1][1][2] = h2inv * 1.0;
                a_h[i][j][k][1][2][1] = h2inv * 1.0;
                a_h[i][j][k][2][1][1] = h2inv * 1.0;
            }

    auto a_2h = mgcl::MultigridEngine::galerkin(a_h, nullptr, nullptr, true, true, true, false);

    // Test with fine h (result from Matlab's Symbolic Toolbox)
    double factor1 = h2inv * (3.0 / 256.0);  // corners
    double factor2 = h2inv * (5.0 / 128.0);  // diagonally off
    double factor3 = h2inv * (3.0 / 64.0);   // straight off
    double factor4 = h2inv * (-27.0 / 32.0); // center

    for (int i = 0; i < a_2h.getDim1(); i++)
        for (int j = 0; j < a_2h.getDim2(); j++)
            for (int k = 0; k < a_2h.getDim3(); k++)
            {
                REQUIRE(a_2h[i][j][k][0][0][0] == factor1);
                REQUIRE(a_2h[i][j][k][0][0][1] == factor2);
                REQUIRE(a_2h[i][j][k][0][0][2] == factor1);
                REQUIRE(a_2h[i][j][k][0][1][0] == factor2);
                REQUIRE(a_2h[i][j][k][0][1][1] == factor3);
                REQUIRE(a_2h[i][j][k][0][1][2] == factor2);
                REQUIRE(a_2h[i][j][k][0][2][0] == factor1);
                REQUIRE(a_2h[i][j][k][0][2][1] == factor2);
                REQUIRE(a_2h[i][j][k][0][2][2] == factor1);
                REQUIRE(a_2h[i][j][k][1][0][0] == factor2);
                REQUIRE(a_2h[i][j][k][1][0][1] == factor3);
                REQUIRE(a_2h[i][j][k][1][0][2] == factor2);
                REQUIRE(a_2h[i][j][k][1][1][0] == factor3);
                REQUIRE(a_2h[i][j][k][1][1][1] == factor4);
                REQUIRE(a_2h[i][j][k][1][1][2] == factor3);
                REQUIRE(a_2h[i][j][k][1][2][0] == factor2);
                REQUIRE(a_2h[i][j][k][1][2][1] == factor3);
                REQUIRE(a_2h[i][j][k][1][2][2] == factor2);
                REQUIRE(a_2h[i][j][k][2][0][0] == factor1);
                REQUIRE(a_2h[i][j][k][2][0][1] == factor2);
                REQUIRE(a_2h[i][j][k][2][0][2] == factor1);
                REQUIRE(a_2h[i][j][k][2][1][0] == factor2);
                REQUIRE(a_2h[i][j][k][2][1][1] == factor3);
                REQUIRE(a_2h[i][j][k][2][1][2] == factor2);
                REQUIRE(a_2h[i][j][k][2][2][0] == factor1);
                REQUIRE(a_2h[i][j][k][2][2][1] == factor2);
                REQUIRE(a_2h[i][j][k][2][2][2] == factor1);
            }
}
