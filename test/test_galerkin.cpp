#include "matrix2d.hpp"

#include <cmath>

#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"

#include "../level.hpp"
#include "../multigrid_engine.hpp"
#include "../stencil.hpp"

TEST_CASE("galerkin Laplace")
{
    int m = GENERATE(2, 4, 8);
    int n = GENERATE(2, 4, 8);
    int o = GENERATE(2, 4, 8);

    // Fill varying stencil on fine grid with 7p Laplace
    mgcl::VaryingStencil3x3x3 a_h(m, n, o, 2, 2, 2);
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

    auto a_2h = mgcl::MultigridEngine::galerkin(a_h);
    auto a2hm = mgcl_test::Matrix2d::fromVaryingStencil(a_2h, true);

    // calculate results with Matrices to check
    auto ah = mgcl_test::Matrix2d::laplace7p3d(m, n, o);
    auto s = mgcl_test::Matrix2d::restrictionFullWeight(m, n, o);
    auto k = mgcl_test::Matrix2d::cuttingMatrix3d(m / 2, n / 2, o / 2);

    auto r = k * s;
    auto p = r.transposed();

    auto a2h = r * ah * p;

    CHECK(a2hm == a2h);
}

TEST_CASE("galerkin random values periodic")
{
    int m = GENERATE(2, 4, 8);
    int n = GENERATE(2, 4, 8);
    int o = GENERATE(2, 4, 8);

    double tol = 1e-12;

    // Fill varying stencil on fine grid with 27p random values
    mgcl::VaryingStencil3x3x3 a_h(m, n, o, 2, 2, 2);
    a_h.fillRandom(-10, 10);
    a_h.updateGhosts();

    auto a_2h = mgcl::MultigridEngine::galerkin(a_h);
    auto a2hm = mgcl_test::Matrix2d::fromVaryingStencil(a_2h, true);

    // calculate results with Matrices to check
    auto ah = mgcl_test::Matrix2d::fromVaryingStencil(a_h, true);
    auto s = mgcl_test::Matrix2d::restrictionFullWeight(m, n, o);
    auto k = mgcl_test::Matrix2d::cuttingMatrix3d(m / 2, n / 2, o / 2);

    auto r = k * s;
    auto p = r.transposed();

    auto a2h = r * ah * p;

    CHECK(a2hm.isEqual(a2h, tol));
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
    auto a_h = std::make_unique<mgcl::VaryingStencil3x3x3>(m, n, o, 2, 2, 2);
    a_h->fillRandom(-10, 10);
    a_h->updateGhosts();

    std::unique_ptr<mgcl::VaryingStencil3x3x3> a_2h = nullptr;

    for (int lv = 0; lv < maxlv; lv++)
    {
        a_2h = std::make_unique<mgcl::VaryingStencil3x3x3>(mgcl::MultigridEngine::galerkin(*a_h));
        auto a2hm = mgcl_test::Matrix2d::fromVaryingStencil(*a_2h, true);

        // calculate results with Matrices to check
        auto ah = mgcl_test::Matrix2d::fromVaryingStencil(*a_h, true);
        auto s = mgcl_test::Matrix2d::restrictionFullWeight(m >> lv, n >> lv, o >> lv);
        auto k = mgcl_test::Matrix2d::cuttingMatrix3d(m >> (lv + 1), n >> (lv + 1), o >> (lv + 1));

        auto r = k * s;
        auto p = r.transposed();

        auto a2h = r * ah * p;

        REQUIRE(a2hm.isEqual(a2h, tol));

        // a_2h is a_h for next level
        a_h = std::move(a_2h);
    }
}
