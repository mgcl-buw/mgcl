#include "matrix2d.hpp"

#include <cmath>

#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"

#include "../src/mgcl/level.hpp"
#include "../src/mgcl/multigrid_engine.hpp"
#include "../src/mgcl/stencil.hpp"

#include "test_utility.hpp"

TEST_CASE("galerkin Laplace vs Matrix")
{
    int m = GENERATE(2, 4, 8);
    int n = GENERATE(2, 4, 8);
    int o = GENERATE(2, 4, 8);
    double h2inv = static_cast<double>(m) * static_cast<double>(m);

    // Fill varying stencil on fine grid with 7p Laplace
    mgcl::VaryingStencil a_h(m, n, o, 3, 2, 2, 2);
    double h = 1.0 / static_cast<double>(m);
    mgcl_test::fill7pLaplace(a_h, h, true);

    auto a_2h = mgcl::MultigridEngine::galerkin(a_h, 2, nullptr, nullptr, true, true, true, false);
    auto a2hm = mgcl_test::Matrix2d::fromVaryingStencil(a_2h, true);

    // calculate results with Matrices to check
    auto ah = mgcl_test::Matrix2d::laplace7p3d(m, n, o) * h2inv;
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

    auto a_2h = mgcl::MultigridEngine::galerkin(a_h, 2, nullptr, nullptr, true, true, true, false);
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
    auto deviceType = CL_DEVICE_TYPE_GPU; // GENERATE(CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU);

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

    auto a_2h = mgcl::MultigridEngine::galerkin(a_h, 2, nullptr, nullptr, true, true, true, false);
    auto a_2h_gpu = mgcl::MultigridEngine::galerkin(a_h_gpu, 2, t.getProgram(), t.getCommands(), t.getContext(),
                                                    nullptr, nullptr, true, true, true, false, nullptr, nullptr);
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
        a_2h = std::make_unique<mgcl::VaryingStencil>(mgcl::MultigridEngine::galerkin(*a_h, 2, nullptr, nullptr, true, true, true, false));
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
    double h = 1.0 / static_cast<double>(m);
    mgcl_test::fill7pLaplace(a_h, h, true);

    auto a_2h = mgcl::MultigridEngine::galerkin(a_h, 2, nullptr, nullptr, true, true, true, false);
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

    // Fill varying stencil on fine grid with 7p Laplace
    mgcl::VaryingStencil a(m, n, o, 3, 2, 2, 2);
    double h = 1.0 / static_cast<double>(m);
    mgcl_test::fill7pLaplace(a, h, true);

    auto s = mgcl::create3dFullWeightRestrictionStencil();

    auto as = a.multiply(s, 2, nullptr, true, true);
    auto sa = s.multiply(a, 2, nullptr, true, true);

    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < o; k++)
                for (int ii = 0; ii < as.getWidth(); ii++)
                    for (int jj = 0; jj < as.getWidth(); jj++)
                        for (int kk = 0; kk < as.getWidth(); kk++)
                        {
                            REQUIRE(as[ii][jj][kk][i][j][k] == sa[ii][jj][kk][i][j][k]);
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
    double h2inv = static_cast<double>(m * m);

    // Fill varying stencil on fine grid with 7p Laplace
    mgcl::VaryingStencil a_h(m, n, o, 3, 2, 2, 2);
    double h = 1.0 / static_cast<double>(m);
    mgcl_test::fill7pLaplace(a_h, h, true);

    auto a_2h = mgcl::MultigridEngine::galerkin(a_h, 2, nullptr, nullptr, true, true, true, false);

    // Test with fine h (result from Matlab's Symbolic Toolbox)
    double factor1 = h2inv * (3.0 / 256.0);  // corners
    double factor2 = h2inv * (5.0 / 128.0);  // diagonally off
    double factor3 = h2inv * (3.0 / 64.0);   // straight off
    double factor4 = h2inv * (-27.0 / 32.0); // center

    for (int i = 0; i < a_2h.getM(); i++)
        for (int j = 0; j < a_2h.getN(); j++)
            for (int k = 0; k < a_2h.getO(); k++)
            {
                REQUIRE(a_2h[0][0][0][i][j][k] == factor1);
                REQUIRE(a_2h[0][0][1][i][j][k] == factor2);
                REQUIRE(a_2h[0][0][2][i][j][k] == factor1);
                REQUIRE(a_2h[0][1][0][i][j][k] == factor2);
                REQUIRE(a_2h[0][1][1][i][j][k] == factor3);
                REQUIRE(a_2h[0][1][2][i][j][k] == factor2);
                REQUIRE(a_2h[0][2][0][i][j][k] == factor1);
                REQUIRE(a_2h[0][2][1][i][j][k] == factor2);
                REQUIRE(a_2h[0][2][2][i][j][k] == factor1);
                REQUIRE(a_2h[1][0][0][i][j][k] == factor2);
                REQUIRE(a_2h[1][0][1][i][j][k] == factor3);
                REQUIRE(a_2h[1][0][2][i][j][k] == factor2);
                REQUIRE(a_2h[1][1][0][i][j][k] == factor3);
                REQUIRE(a_2h[1][1][1][i][j][k] == factor4);
                REQUIRE(a_2h[1][1][2][i][j][k] == factor3);
                REQUIRE(a_2h[1][2][0][i][j][k] == factor2);
                REQUIRE(a_2h[1][2][1][i][j][k] == factor3);
                REQUIRE(a_2h[1][2][2][i][j][k] == factor2);
                REQUIRE(a_2h[2][0][0][i][j][k] == factor1);
                REQUIRE(a_2h[2][0][1][i][j][k] == factor2);
                REQUIRE(a_2h[2][0][2][i][j][k] == factor1);
                REQUIRE(a_2h[2][1][0][i][j][k] == factor2);
                REQUIRE(a_2h[2][1][1][i][j][k] == factor3);
                REQUIRE(a_2h[2][1][2][i][j][k] == factor2);
                REQUIRE(a_2h[2][2][0][i][j][k] == factor1);
                REQUIRE(a_2h[2][2][1][i][j][k] == factor2);
                REQUIRE(a_2h[2][2][2][i][j][k] == factor1);
            }
}

// Test that Galerkin yields the same result when using different amount of jacobiIterationsPerKernel
TEST_CASE("galerkin_different_jacobi_iters_per_kernel")
{
    int N = 16;

    auto v = std::make_shared<mgcl::Cuboid>(N, N, N);
    auto f = std::make_shared<mgcl::Cuboid>(N, N, N);
    v->fillRandom();
    f->fillRandom();

    mgcl::Problem p1(N, N, N, f, v);
    p1.setGhosts(1);
    p1.setJacobiIterationsPerKernel(1);
    p1.setStencilType(mgcl::MGCL_VARYING);

    mgcl::Problem p2(N, N, N, f, v);
    p2.setGhosts(2);
    p2.setJacobiIterationsPerKernel(2);
    p2.setStencilType(mgcl::MGCL_VARYING);

    mgcl::Problem p3(N, N, N, f, v);
    p3.setGhosts(3);
    p3.setJacobiIterationsPerKernel(3);
    p3.setStencilType(mgcl::MGCL_VARYING);

    REQUIRE(p1.getMaxlevel() == p2.getMaxlevel());
    REQUIRE(p1.getMaxlevel() == p3.getMaxlevel());

    auto sv1 = p1.getStencilValues();
    auto sv2 = p2.getStencilValues();
    auto sv3 = p3.getStencilValues();
    sv1->fill1dIndex(false);
    sv2->copyRealFrom(*sv1);
    sv3->copyRealFrom(*sv1);

    SECTION("Sequential")
    {
        p1.init();
        p2.init();
        p3.init();

        // check stencil values for each level
        for (int i = 0; i < p1.getMaxlevel(); i++)
        {
            CAPTURE(i);
            REQUIRE(p1.getLevelAt(i).getStencilValues()->getGhostsM() == 2);
            REQUIRE(p1.getLevelAt(i).getStencilValues()->getGhostsN() == 2);
            REQUIRE(p1.getLevelAt(i).getStencilValues()->getGhostsO() == 2);
            REQUIRE(p2.getLevelAt(i).getStencilValues()->getGhostsM() == 2);
            REQUIRE(p2.getLevelAt(i).getStencilValues()->getGhostsN() == 2);
            REQUIRE(p2.getLevelAt(i).getStencilValues()->getGhostsO() == 2);
            REQUIRE(p3.getLevelAt(i).getStencilValues()->getGhostsM() == 3);
            REQUIRE(p3.getLevelAt(i).getStencilValues()->getGhostsN() == 3);
            REQUIRE(p3.getLevelAt(i).getStencilValues()->getGhostsO() == 3);

            REQUIRE(p1.getLevelAt(i).getStencilValues()->isEqual(*p2.getLevelAt(i).getStencilValues()));
            REQUIRE(p1.getLevelAt(i).getStencilValues()->isEqual(*p3.getLevelAt(i).getStencilValues()));
        }
    }

    SECTION("OpenCL")
    {
        p1.setUseOpencl(true);
        p1.setDeviceType(CL_DEVICE_TYPE_GPU);
        p2.setUseOpencl(true);
        p2.setDeviceType(CL_DEVICE_TYPE_GPU);
        p3.setUseOpencl(true);
        p3.setDeviceType(CL_DEVICE_TYPE_GPU);
        p1.init();
        p2.init();
        p3.init();

        // check stencil values for each level
        for (int i = 0; i < p1.getMaxlevel(); i++)
        {
            CAPTURE(i);
            REQUIRE(p1.getLevelAt(i).getStencilValuesGpu()->getGh() == 2);
            REQUIRE(p2.getLevelAt(i).getStencilValuesGpu()->getGh() == 2);
            REQUIRE(p3.getLevelAt(i).getStencilValuesGpu()->getGh() == 3);

            auto sv1 = p1.getLevelAt(i).getStencilValuesGpu()->read(p1.getCommands(), true);
            auto sv2 = p2.getLevelAt(i).getStencilValuesGpu()->read(p2.getCommands(), true);
            auto sv3 = p3.getLevelAt(i).getStencilValuesGpu()->read(p3.getCommands(), true);

            REQUIRE(sv1.isEqual(sv2));
            REQUIRE(sv1.isEqual(sv3));
        }
    }
}