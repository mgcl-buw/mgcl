#include "matrix2d.hpp"

#include <catch2/catch_message.hpp>
#include <cmath>
#include <memory>

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
    mgcl::VaryingStencil a_h(m, n, o, 3, 1, 1, 1);
    double h = 1.0 / static_cast<double>(m);
    mgcl_test::fill7pLaplace(a_h, h, true);

    auto a_2h = mgcl::MultigridEngine::galerkinOptimized(a_h, 1, m >> 1, n >> 1, o >> 1);
    auto a2hm = mgcl_test::Matrix2d::fromVaryingStencil(*a_2h, true);

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
    mgcl::VaryingStencil a_h(m, n, o, 3, 1, 1, 1);
    a_h.fillRandom(-10, 10);
    a_h.updateGhosts();

    auto a_2h = mgcl::MultigridEngine::galerkinOptimized(a_h, 1, m >> 1, n >> 1, o >> 1);
    auto a2hm = mgcl_test::Matrix2d::fromVaryingStencil(*a_2h, true);

    // calculate results with Matrices to check
    auto ah = mgcl_test::Matrix2d::fromVaryingStencil(a_h, true);
    auto r = mgcl_test::Matrix2d::restrictionFullWeight(m, n, o);
    auto p = mgcl_test::Matrix2d::prolongationBilinear(m, n, o);

    auto a2h = r * ah * p;

    CHECK(a2hm.isEqual(a2h, tol));
}

TEST_CASE("seq_galerkin_optimized_random_values_periodic_vs_matrix")
{
    int m = GENERATE(2, 4, 8);
    int n = GENERATE(2, 4, 8);
    int o = GENERATE(2, 4, 8);

    double tol = 1e-12;

    // Fill varying stencil on fine grid with 27p random values
    mgcl::VaryingStencil a_h(m, n, o, 3, 1, 1, 1);
    a_h.fill1dIndex(true);
    a_h[0][0][0][0][0][0] = 0.5; // fill with non zero value
    a_h.updateGhosts();

    auto a_2h = mgcl::MultigridEngine::galerkinOptimized(a_h, 1, m >> 1, n >> 1, o >> 1);
    auto a2hm = mgcl_test::Matrix2d::fromVaryingStencil(*a_2h, true);

    // calculate results with Matrices to check
    auto ah = mgcl_test::Matrix2d::fromVaryingStencil(a_h, true);
    auto r = mgcl_test::Matrix2d::restrictionFullWeight(m, n, o);
    auto p = mgcl_test::Matrix2d::prolongationBilinear(m, n, o);

    auto a2h = r * ah * p;

    // a_h.dumpToFile("a_h.txt", false);
    // ah.dumpToFileWithIndices("ah_matrix.txt");
    // a2hm.dumpToFile("a2hm.txt");
    // a2h.dumpToFile("a2h.txt");

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
    mgcl::VaryingStencilGpu a_h_gpu(m, n, o, 3, gh, t.getContext(), t.getCommands(), t.getProgram());
    mgcl::VaryingStencil a_h(m, n, o, 3, gh, gh, gh);
    a_h.fillRandom(-10, 10);
    a_h.updateGhosts();
    a_h_gpu.fill(a_h, t.getCommands(), true);

    auto a_2h = mgcl::MultigridEngine::galerkinOptimized(a_h, 2, m >> 1, n >> 1, o >> 1);
    auto a_2h_gpu = mgcl::MultigridEngine::galerkinOptimized(a_h_gpu, 1, m >> 1, n >> 1, o >> 1,
                                                             t.getProgram(), t.getCommands(), t.getContext(),
                                                             nullptr, nullptr);
    t.finish();

    auto ret = a_2h_gpu->read(t.getCommands(), true);

    REQUIRE(a_2h->isEqual(ret, tol));
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
    auto a_h = std::make_unique<mgcl::VaryingStencil>(m, n, o, 3, 1, 1, 1);
    a_h->fillRandom(-10, 10);
    a_h->updateGhosts();

    std::unique_ptr<mgcl::VaryingStencil> a_2h = nullptr;

    for (int lv = 0; lv < maxlv; lv++)
    {
        a_2h = mgcl::MultigridEngine::galerkinOptimized(*a_h, 1, a_h->getM() >> 1, a_h->getN() >> 1, a_h->getO() >> 1);
        a_2h->updateGhosts();
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

    auto a_2h = mgcl::MultigridEngine::galerkinOptimized(a_h, 1, m >> 1, n >> 1, o >> 1);
    auto a2hm = mgcl_test::Matrix2d::fromVaryingStencil(*a_2h, true);

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

    auto a_2h_ptr = mgcl::MultigridEngine::galerkinOptimized(a_h, 1, m >> 1, n >> 1, o >> 1);
    auto& a_2h = *a_2h_ptr;
    a_2h.updateGhosts();

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
            REQUIRE(p1.getLevelAt(i).getStencilValues()->getGhostsM() == 1);
            REQUIRE(p1.getLevelAt(i).getStencilValues()->getGhostsN() == 1);
            REQUIRE(p1.getLevelAt(i).getStencilValues()->getGhostsO() == 1);
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

        CAPTURE(p1.getMaxlevel());

        // check stencil values for each level
        for (int i = 0; i < p1.getMaxlevel(); i++)
        {
            CAPTURE(i);
            REQUIRE(p1.getLevelAt(i).getStencilValuesGpu());
            REQUIRE(p2.getLevelAt(i).getStencilValuesGpu());
            REQUIRE(p3.getLevelAt(i).getStencilValuesGpu());
            REQUIRE(p1.getLevelAt(i).getStencilValuesGpu()->getGh() == 1);
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

// OpenCL uses a derivate of C99. This test checks whether the port of C++ to C99 works correctly.
// Does not actually use OpenCL.
TEST_CASE("GalerkinC99")
{
    int m = 4;
    int n = 4;
    int o = 4;

    mgcl::VaryingStencil a_h(m, n, o, 3, 1, 1, 1);
    a_h.fill1dIndex(false);
    a_h.updateGhosts();

    auto a_2h_cpp = mgcl::MultigridEngine::galerkinOptimized(a_h, 1, m >> 1, n >> 1, o >> 1);

    auto a_2h_c99 = a_2h_cpp->copyShallow();
    auto r = mgcl::create3dFullWeightRestrictionStencil();
    auto p = mgcl::create3dBilinearProlongationStencil();
    {
        // Helper struct for galerkin. Defines an interval with integer start and end.
        typedef struct Interval
        {
            int start;
            int end;
        } Interval;

        // Helper struct for galerkin. Defines a grid point with integer 3d coordinates.
        typedef struct Point
        {
            int x;
            int y;
            int z;
        } Point;

        // Helper class for declaring the helper functions as static functions of the class, since functions
        // inside functions are actually not allowed. We could also use lambdas, but we want to be as close to
        // C99 syntax as possible.
        struct Fns
        {
            // Returns the intersection of two intervals or [-1,-1] if they don't overlap
            static Interval intersect(Interval a, Interval b)
            {
                Interval ret;
                // Check if intervals overlap
                if (a.start <= b.end && b.start <= a.end)
                {
                    // Calculate start and end points of intersection
                    ret.start = (a.start > b.start) ? a.start : b.start;
                    ret.end = (a.end < b.end) ? a.end : b.end;
                    return ret;
                }
                else
                {
                    // Intervals do not overlap
                    ret.start = -1;
                    ret.end = -1;
                    return ret;
                }
            };

            // Returns the stencil entry indices of the stencil sitting at locationOfStencil that maps to mapsTo.
            // No check is done, if the mapping is possible, i.e. the returned value might be outside of range [0,2].
            // The result is just the difference of the indices plus one, since the stencil entry indices start at 0
            // and not at -1.
            static Point stencilEntryThatMapsTo(Point locationOfStencil, Point mapsTo)
            {
                Point ret;
                ret.x = mapsTo.x - locationOfStencil.x + 1;
                ret.y = mapsTo.y - locationOfStencil.y + 1;
                ret.z = mapsTo.z - locationOfStencil.z + 1;
                return ret;
            };

            // Returns the grid point indices that is mapped to by the stencil entry of another point.
            // stencilEntry must be 0-based, hence the substraction by 1.
            static Point pointMappedToByStencilEntry(Point locationOfStencil, Point stencilEntry)
            {
                Point ret;
                ret.x = locationOfStencil.x + (stencilEntry.x - 1);
                ret.y = locationOfStencil.y + (stencilEntry.y - 1);
                ret.z = locationOfStencil.z + (stencilEntry.z - 1);
                return ret;
            };

            // Returns the point on the fine grid that is related to the coarse grid point, respecting ghost cells.
            static Point coarseToFine(Point p, int ghc, int ghf)
            {
                Point ret;
                ret.x = (p.x - ghc) * 2 + 1 + ghf;
                ret.y = (p.y - ghc) * 2 + 1 + ghf;
                ret.z = (p.z - ghc) * 2 + 1 + ghf;
                return ret;
            };
        };

        // alias for making things easier
        mgcl::VaryingStencil* a_2h = a_2h_c99.get();

        /********* Start copy of sequential C++ version of Galerkin **********/

        // for each real resulting stencil and stencil entry...
        // (only until a_h >> 1, since on root and on threshold level, a_2h has global size, but only local part can be filled.)
        for (int i = a_2h->getGhostsM(); i < (a_h.getM() >> 1) + a_2h->getGhostsM(); i++)
            for (int j = a_2h->getGhostsN(); j < (a_h.getN() >> 1) + a_2h->getGhostsN(); j++)
                for (int k = a_2h->getGhostsO(); k < (a_h.getO() >> 1) + a_2h->getGhostsO(); k++)
                    for (int ii = 0; ii < a_2h->getWidth(); ii++)
                        for (int jj = 0; jj < a_2h->getWidth(); jj++)
                            for (int kk = 0; kk < a_2h->getWidth(); kk++)
                            {
                                // calculate fine grid point indices
                                Point gp_c = {i, j, k};
                                Point gp_f = Fns::coarseToFine(gp_c, a_2h->getGhostsM(), a_h.getGhostsM());
                                Point entry_gpc = {ii, jj, kk};
                                Point entry_gpf = Fns::coarseToFine(
                                    Fns::pointMappedToByStencilEntry(gp_c, entry_gpc),
                                    a_2h->getGhostsM(), a_h.getGhostsM());

                                // find intersection S_P of neighbouring points for entry_gpf with reach=1 and gp_f with reach=2
                                Interval xa = {.start = gp_f.x - 2, .end = gp_f.x + 2};
                                Interval ya = {.start = gp_f.y - 2, .end = gp_f.y + 2};
                                Interval za = {.start = gp_f.z - 2, .end = gp_f.z + 2};
                                Interval xb = {.start = entry_gpf.x - 1, .end = entry_gpf.x + 1};
                                Interval yb = {.start = entry_gpf.y - 1, .end = entry_gpf.y + 1};
                                Interval zb = {.start = entry_gpf.z - 1, .end = entry_gpf.z + 1};

                                Interval S_P[3] = {
                                    Fns::intersect(xa, xb),
                                    Fns::intersect(ya, yb),
                                    Fns::intersect(za, zb),
                                };

                                // Start calc (R*A)*P
                                double res = 0;

                                // for each fine grid point gp_sp in S_P:
                                for (int spi = S_P[0].start; spi <= S_P[0].end; spi++)
                                    for (int spj = S_P[1].start; spj <= S_P[1].end; spj++)
                                        for (int spk = S_P[2].start; spk <= S_P[2].end; spk++)
                                        {
                                            Point gp_sp = {spi, spj, spk};
                                            // tmp_p <- in stencil P located at gp_sp: Find stencil entry entry_p that maps to entry_gpf. Since
                                            // gp_sp is in S_P, it is ensured that the stencil has a stencil entry that maps to entry_gpf.
                                            Point tmp_p_indices = Fns::stencilEntryThatMapsTo(gp_sp, entry_gpf);
                                            double tmp_p = p[tmp_p_indices.x][tmp_p_indices.y][tmp_p_indices.z];

                                            // Start calc R*A
                                            // find intersection S_R of neighbouring points for gp_f and gp_sp, both with reach=1
                                            xa.start = gp_f.x - 1;
                                            xa.end = gp_f.x + 1;
                                            ya.start = gp_f.y - 1;
                                            ya.end = gp_f.y + 1;
                                            za.start = gp_f.z - 1;
                                            za.end = gp_f.z + 1;
                                            xb.start = spi - 1;
                                            xb.end = spi + 1;
                                            yb.start = spj - 1;
                                            yb.end = spj + 1;
                                            zb.start = spk - 1;
                                            zb.end = spk + 1;

                                            Interval S_R[3] = {
                                                Fns::intersect(xa, xb),
                                                Fns::intersect(ya, yb),
                                                Fns::intersect(za, zb),
                                            };

                                            double sum = 0;
                                            // for each fine grid point gp_sr in S_R:
                                            for (int sri = S_R[0].start; sri <= S_R[0].end; sri++)
                                                for (int srj = S_R[1].start; srj <= S_R[1].end; srj++)
                                                    for (int srk = S_R[2].start; srk <= S_R[2].end; srk++)
                                                    {
                                                        Point gp_sr = {sri, srj, srk};
                                                        // tmp_r <- in stencil R located at gp_f: Find stencil entry entry_r that maps to gp_sr
                                                        Point tmp_r_indices = Fns::stencilEntryThatMapsTo(gp_f, gp_sr);
                                                        double tmp_r = r[tmp_r_indices.x][tmp_r_indices.y][tmp_r_indices.z];
                                                        // tmp_a <- in stencil A located at gp_sr: Find stencil entry that maps to gp_sp
                                                        Point tmp_a_indices = Fns::stencilEntryThatMapsTo(gp_sr, gp_sp);

                                                        double tmp_a = a_h[tmp_a_indices.x][tmp_a_indices.y][tmp_a_indices.z][gp_sr.x][gp_sr.y][gp_sr.z];
                                                        // sum <- sum + tmp_r * tmp_a
                                                        sum += tmp_r * tmp_a;
                                                        // End calc R*A
                                                    }

                                            //   res <- res + sum * tmp_p
                                            res += sum * tmp_p;
                                            // End calc (R*A)*P
                                        }

                                // store res in rap
                                (*a_2h)[ii][jj][kk][i][j][k] = res;
                            }

        /********* End sequential C++ version of Galerkin **********/
    }

    REQUIRE(a_2h_cpp->isEqual(*a_2h_c99));
}

// This is a regression test.
// Checks if Galerkin is correct, if resm, resn, reso is different from the actual buffer size.
// This happens when using MPI on the threshold level (but this test does NOT use MPI).
TEST_CASE("galerkinOpimizedGpuCoarseGridSizeDifferentFromBufferSize")
{
    int m = 4;
    int n = 4;
    int o = 4;
    int gh = 1;

    mgcl_test::TestUtility tu(CL_DEVICE_TYPE_GPU);

    mgcl::VaryingStencil a_h(m, n, o, 3, gh, gh, gh);
    a_h.fill1dIndex(false);
    a_h.updateGhosts();

    mgcl::VaryingStencilGpu a_h_gpu(m, n, o, 3, gh, tu.getContext(), tu.getCommands(), tu.getProgram());
    a_h_gpu.fill(a_h, tu.getCommands(), true);

    auto a_2h_same_sizes_gpu = mgcl::MultigridEngine::galerkinOptimized(a_h_gpu, gh, m >> 1, n >> 1, o >> 1, tu.getProgram(), tu.getCommands(), tu.getContext(), nullptr, nullptr);
    auto a_2h_different_sizes_gpu = mgcl::MultigridEngine::galerkinOptimized(a_h_gpu, gh, m, n, o, tu.getProgram(), tu.getCommands(), tu.getContext(), nullptr, nullptr);

    auto a_2h_same_sizes = a_2h_same_sizes_gpu->read(tu.getCommands(), true);
    auto a_2h_different_sizes = a_2h_different_sizes_gpu->read(tu.getCommands(), true);

    // Check that the local portions of results are equal
    // clang-format off
    for (int i = gh; i < a_2h_same_sizes.getM() + gh; i++)
    for (int j = gh; j < a_2h_same_sizes.getN() + gh; j++)
    for (int k = gh; k < a_2h_same_sizes.getO() + gh; k++)
        for (int ii = 0; ii < 3; ii++)
        for (int jj = 0; jj < 3; jj++)
        for (int kk = 0; kk < 3; kk++)
        {
            REQUIRE(a_2h_same_sizes[ii][jj][kk][i][j][k] == a_2h_different_sizes[ii][jj][kk][i][j][k]);
        }
    // clang-format on
}