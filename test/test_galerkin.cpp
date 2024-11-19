#include "matrix2d.hpp"

#include <catch2/catch_message.hpp>
#include <cmath>
#include <memory>
#include <sstream>
#include <vector>

#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"

#include "../src/mgcl/level.hpp"
#include "../src/mgcl/multigrid_engine.hpp"
#include "../src/mgcl/stencil.hpp"

#include "cli_args.hpp"
#include "device_type_generator.hpp"
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
    auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));

    mgcl_test::TestUtility t(deviceType);

    int m = 4; // GENERATE(2, 4, 8);
    int n = 4; // GENERATE(4, 8);
    int o = 4; // GENERATE(2, 4);

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
    auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));

    int m = 4;
    int n = 4;
    int o = 4;
    int gh = 1;

    mgcl_test::TestUtility tu(deviceType);

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

TEST_CASE("galerkinOptimized_parallel_iijjkk")
{
    int m_c_loc = GENERATE(2, 4);
    int n_c_loc = GENERATE(4, 6);
    int o_c_loc = GENERATE(1, 2, 4);

    int totalsize = m_c_loc * n_c_loc * o_c_loc * 27;

    // Checks that the index calculation in the optimized galerkin parallel iijjkk kernel works correctly.
    SECTION("indices")
    {
        int indices[6] = {0, 0, 0, 0, 0, 0};
        auto add_one = [o_c_loc, n_c_loc, m_c_loc](int* indices)
        {
            indices[5]++;
            if (indices[5] >= o_c_loc)
                indices[4]++;
            if (indices[4] >= n_c_loc)
                indices[3]++;
            if (indices[3] >= m_c_loc)
                indices[2]++;
            if (indices[2] >= 3)
                indices[1]++;
            if (indices[1] >= 3)
                indices[0]++;

            indices[5] = indices[5] % o_c_loc;
            indices[4] = indices[4] % n_c_loc;
            indices[3] = indices[3] % m_c_loc;
            indices[2] = indices[2] % 3;
            indices[1] = indices[1] % 3;
            indices[0] = indices[0] % 3;
        };

        for (size_t idx = 0; idx < totalsize; idx++)
        {
            int gridsize = m_c_loc * n_c_loc * o_c_loc;
            int stencil_idx = idx / gridsize;
            int grididx = idx % gridsize;

            int no = n_c_loc * o_c_loc;
            int i = grididx / no;
            int j = (grididx - i * no) / o_c_loc;
            int k = grididx % o_c_loc;

            int ii = stencil_idx / 9;
            int jj = (stencil_idx - ii * 9) / 3;
            int kk = stencil_idx % 3;

            CAPTURE(idx, grididx, i, j, k, ii, jj, kk);

            REQUIRE(ii == indices[0]);
            REQUIRE(jj == indices[1]);
            REQUIRE(kk == indices[2]);
            REQUIRE(i == indices[3]);
            REQUIRE(j == indices[4]);
            REQUIRE(k == indices[5]);

            add_one(indices);
        }
    }

    // Checks that the optimized galerkin parallel iijjkk kernel writes to every real cell exactly once.
    SECTION("cells_written_once")
    {
        int gh = 1;

        mgcl::VaryingStencil s(m_c_loc, n_c_loc, o_c_loc, 3, gh, gh, gh);
        s.fill(0, false);

        for (size_t idx = 0; idx < totalsize; idx++)
        {
            int gridsize = m_c_loc * n_c_loc * o_c_loc;
            int stencil_idx = idx / gridsize;
            int grididx = idx % gridsize;

            int no = n_c_loc * o_c_loc;
            int i = grididx / no;
            int j = (grididx - i * no) / o_c_loc;
            int k = grididx % o_c_loc;

            int ii = stencil_idx / 9;
            int jj = (stencil_idx - ii * 9) / 3;
            int kk = stencil_idx % 3;

            i += gh;
            j += gh;
            k += gh;

            if (i < m_c_loc + gh && j < n_c_loc + gh && k < o_c_loc + gh && stencil_idx < 27)
                s[ii][jj][kk][i][j][k] += 1;
        }

        // clang-format off
        for (int ii = 0; ii < 3; ii++)
        for (int jj = 0; jj < 3; jj++)
        for (int kk = 0; kk < 3; kk++)
            for (int i = gh; i < m_c_loc + gh; i++)
            for (int j = gh; j < n_c_loc + gh; j++)
            for (int k = gh; k < o_c_loc + gh; k++)
            {
                CAPTURE(i,j,k,ii,jj,kk);
                REQUIRE(s[ii][jj][kk][i][j][k] == 1);
            }
        // clang-format on
    }
}

// Checks whether the handcrafted equations for calculating RAP yield the same result as the interval-based
// optimized galerkin kernel. The idea is that the equations are the same for every coarse grid point.
// The equations were deduced with a python script from the matrix trace. Only works for 3x3x3 stencils.
// This is equivalent to what Hypre does for rap_type = 0 (albeit here being full-coarsening instead of semi-coarsening).
// Note: This test does not evaluate any sequential or opencl-using implementation of building the coarse grid
// operator, but instead serves as a proof of conecpt.
TEST_CASE("galerkinHandcrafted_proofOfConcept")
{
    auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));

    int m = 8;
    int n = 8;
    int o = 8;
    int gh = 1;

    mgcl_test::TestUtility tu(deviceType);

    mgcl::VaryingStencil a_h(m, n, o, 3, gh, gh, gh);
    a_h.fill1dIndex(false);
    a_h.updateGhosts();

    mgcl::VaryingStencilGpu a_h_gpu(m, n, o, 3, gh, tu.getContext(), tu.getCommands(), tu.getProgram());
    a_h_gpu.fill(a_h, tu.getCommands(), true);

    auto a_2h_gpu = mgcl::MultigridEngine::galerkinOptimized(a_h_gpu, gh, m >> 1, n >> 1, o >> 1, tu.getProgram(), tu.getCommands(), tu.getContext(), nullptr, nullptr);
    auto a_2h = a_2h_gpu->read(tu.getCommands(), true);

    auto r = mgcl::create3dFullWeightRestrictionStencil();
    auto p = mgcl::create3dBilinearProlongationStencil();

    int mc = m >> 1;
    int nc = n >> 1;
    int oc = o >> 1;

    // for each real coarse grid point and the corresponding fine grid point
    for (int ci = gh, fi = gh + 1; ci < mc + gh; ci++, fi += 2)
        for (int cj = gh, fj = gh + 1; cj < nc + gh; cj++, fj += 2)
            for (int ck = gh, fk = gh + 1; ck < oc + gh; ck++, fk += 2)
            {
                // front-top-left coefficient
                REQUIRE(a_2h[0][0][0][ci][cj][ck] == ((r[0][0][0] * a_h[0][0][0][fi + -1][fj + -1][fk + -1]) * p[1][1][1] + (r[0][0][0] * a_h[0][0][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[0][0][0][fi + -1][fj + -1][fk + 0]) * p[1][1][0] + (r[0][0][0] * a_h[0][1][0][fi + -1][fj + -1][fk + -1] + r[0][1][0] * a_h[0][0][0][fi + -1][fj + 0][fk + -1]) * p[1][0][1] + (r[0][0][0] * a_h[0][1][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[0][1][0][fi + -1][fj + -1][fk + 0] + r[0][1][0] * a_h[0][0][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[0][0][0][fi + -1][fj + 0][fk + 0]) * p[1][0][0] + (r[0][0][0] * a_h[1][0][0][fi + -1][fj + -1][fk + -1] + r[1][0][0] * a_h[0][0][0][fi + 0][fj + -1][fk + -1]) * p[0][1][1] + (r[0][0][0] * a_h[1][0][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[1][0][0][fi + -1][fj + -1][fk + 0] + r[1][0][0] * a_h[0][0][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[0][0][0][fi + 0][fj + -1][fk + 0]) * p[0][1][0] + (r[0][0][0] * a_h[1][1][0][fi + -1][fj + -1][fk + -1] + r[0][1][0] * a_h[1][0][0][fi + -1][fj + 0][fk + -1] + r[1][0][0] * a_h[0][1][0][fi + 0][fj + -1][fk + -1] + r[1][1][0] * a_h[0][0][0][fi + 0][fj + 0][fk + -1]) * p[0][0][1] + (r[0][0][0] * a_h[1][1][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[1][1][0][fi + -1][fj + -1][fk + 0] + r[0][1][0] * a_h[1][0][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[1][0][0][fi + -1][fj + 0][fk + 0] + r[1][0][0] * a_h[0][1][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[0][1][0][fi + 0][fj + -1][fk + 0] + r[1][1][0] * a_h[0][0][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[0][0][0][fi + 0][fj + 0][fk + 0]) * p[0][0][0]));

                REQUIRE(a_2h[0][0][1][ci][cj][ck] == ((r[0][0][0] * a_h[0][0][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[0][0][0][fi + -1][fj + -1][fk + 0]) * p[1][1][2] + (r[0][0][0] * a_h[0][0][2][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[0][0][1][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[0][0][0][fi + -1][fj + -1][fk + 1]) * p[1][1][1] + (r[0][0][1] * a_h[0][0][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[0][0][1][fi + -1][fj + -1][fk + 1]) * p[1][1][0] + (r[0][0][0] * a_h[0][1][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[0][1][0][fi + -1][fj + -1][fk + 0] + r[0][1][0] * a_h[0][0][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[0][0][0][fi + -1][fj + 0][fk + 0]) * p[1][0][2] + (r[0][0][0] * a_h[0][1][2][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[0][1][1][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[0][1][0][fi + -1][fj + -1][fk + 1] + r[0][1][0] * a_h[0][0][2][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[0][0][1][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[0][0][0][fi + -1][fj + 0][fk + 1]) * p[1][0][1] + (r[0][0][1] * a_h[0][1][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[0][1][1][fi + -1][fj + -1][fk + 1] + r[0][1][1] * a_h[0][0][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[0][0][1][fi + -1][fj + 0][fk + 1]) * p[1][0][0] + (r[0][0][0] * a_h[1][0][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[1][0][0][fi + -1][fj + -1][fk + 0] + r[1][0][0] * a_h[0][0][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[0][0][0][fi + 0][fj + -1][fk + 0]) * p[0][1][2] + (r[0][0][0] * a_h[1][0][2][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[1][0][1][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[1][0][0][fi + -1][fj + -1][fk + 1] + r[1][0][0] * a_h[0][0][2][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[0][0][1][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[0][0][0][fi + 0][fj + -1][fk + 1]) * p[0][1][1] + (r[0][0][1] * a_h[1][0][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[1][0][1][fi + -1][fj + -1][fk + 1] + r[1][0][1] * a_h[0][0][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[0][0][1][fi + 0][fj + -1][fk + 1]) * p[0][1][0] + (r[0][0][0] * a_h[1][1][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[1][1][0][fi + -1][fj + -1][fk + 0] + r[0][1][0] * a_h[1][0][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[1][0][0][fi + -1][fj + 0][fk + 0] + r[1][0][0] * a_h[0][1][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[0][1][0][fi + 0][fj + -1][fk + 0] + r[1][1][0] * a_h[0][0][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[0][0][0][fi + 0][fj + 0][fk + 0]) * p[0][0][2] + (r[0][0][0] * a_h[1][1][2][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[1][1][1][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[1][1][0][fi + -1][fj + -1][fk + 1] + r[0][1][0] * a_h[1][0][2][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[1][0][1][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[1][0][0][fi + -1][fj + 0][fk + 1] + r[1][0][0] * a_h[0][1][2][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[0][1][1][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[0][1][0][fi + 0][fj + -1][fk + 1] + r[1][1][0] * a_h[0][0][2][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[0][0][1][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[0][0][0][fi + 0][fj + 0][fk + 1]) * p[0][0][1] + (r[0][0][1] * a_h[1][1][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[1][1][1][fi + -1][fj + -1][fk + 1] + r[0][1][1] * a_h[1][0][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[1][0][1][fi + -1][fj + 0][fk + 1] + r[1][0][1] * a_h[0][1][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[0][1][1][fi + 0][fj + -1][fk + 1] + r[1][1][1] * a_h[0][0][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[0][0][1][fi + 0][fj + 0][fk + 1]) * p[0][0][0]));

                REQUIRE(a_2h[0][0][2][ci][cj][ck] == ((r[0][0][1] * a_h[0][0][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[0][0][1][fi + -1][fj + -1][fk + 1]) * p[1][1][2] + (r[0][0][2] * a_h[0][0][2][fi + -1][fj + -1][fk + 1]) * p[1][1][1] + (r[0][0][1] * a_h[0][1][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[0][1][1][fi + -1][fj + -1][fk + 1] + r[0][1][1] * a_h[0][0][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[0][0][1][fi + -1][fj + 0][fk + 1]) * p[1][0][2] + (r[0][0][2] * a_h[0][1][2][fi + -1][fj + -1][fk + 1] + r[0][1][2] * a_h[0][0][2][fi + -1][fj + 0][fk + 1]) * p[1][0][1] + (r[0][0][1] * a_h[1][0][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[1][0][1][fi + -1][fj + -1][fk + 1] + r[1][0][1] * a_h[0][0][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[0][0][1][fi + 0][fj + -1][fk + 1]) * p[0][1][2] + (r[0][0][2] * a_h[1][0][2][fi + -1][fj + -1][fk + 1] + r[1][0][2] * a_h[0][0][2][fi + 0][fj + -1][fk + 1]) * p[0][1][1] + (r[0][0][1] * a_h[1][1][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[1][1][1][fi + -1][fj + -1][fk + 1] + r[0][1][1] * a_h[1][0][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[1][0][1][fi + -1][fj + 0][fk + 1] + r[1][0][1] * a_h[0][1][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[0][1][1][fi + 0][fj + -1][fk + 1] + r[1][1][1] * a_h[0][0][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[0][0][1][fi + 0][fj + 0][fk + 1]) * p[0][0][2] + (r[0][0][2] * a_h[1][1][2][fi + -1][fj + -1][fk + 1] + r[0][1][2] * a_h[1][0][2][fi + -1][fj + 0][fk + 1] + r[1][0][2] * a_h[0][1][2][fi + 0][fj + -1][fk + 1] + r[1][1][2] * a_h[0][0][2][fi + 0][fj + 0][fk + 1]) * p[0][0][1]));

                REQUIRE(a_2h[0][1][0][ci][cj][ck] == ((r[0][0][0] * a_h[0][1][0][fi + -1][fj + -1][fk + -1] + r[0][1][0] * a_h[0][0][0][fi + -1][fj + 0][fk + -1]) * p[1][2][1] + (r[0][0][0] * a_h[0][1][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[0][1][0][fi + -1][fj + -1][fk + 0] + r[0][1][0] * a_h[0][0][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[0][0][0][fi + -1][fj + 0][fk + 0]) * p[1][2][0] + (r[0][0][0] * a_h[0][2][0][fi + -1][fj + -1][fk + -1] + r[0][1][0] * a_h[0][1][0][fi + -1][fj + 0][fk + -1] + r[0][2][0] * a_h[0][0][0][fi + -1][fj + 1][fk + -1]) * p[1][1][1] + (r[0][0][0] * a_h[0][2][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[0][2][0][fi + -1][fj + -1][fk + 0] + r[0][1][0] * a_h[0][1][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[0][1][0][fi + -1][fj + 0][fk + 0] + r[0][2][0] * a_h[0][0][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[0][0][0][fi + -1][fj + 1][fk + 0]) * p[1][1][0] + (r[0][1][0] * a_h[0][2][0][fi + -1][fj + 0][fk + -1] + r[0][2][0] * a_h[0][1][0][fi + -1][fj + 1][fk + -1]) * p[1][0][1] + (r[0][1][0] * a_h[0][2][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[0][2][0][fi + -1][fj + 0][fk + 0] + r[0][2][0] * a_h[0][1][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[0][1][0][fi + -1][fj + 1][fk + 0]) * p[1][0][0] + (r[0][0][0] * a_h[1][1][0][fi + -1][fj + -1][fk + -1] + r[0][1][0] * a_h[1][0][0][fi + -1][fj + 0][fk + -1] + r[1][0][0] * a_h[0][1][0][fi + 0][fj + -1][fk + -1] + r[1][1][0] * a_h[0][0][0][fi + 0][fj + 0][fk + -1]) * p[0][2][1] + (r[0][0][0] * a_h[1][1][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[1][1][0][fi + -1][fj + -1][fk + 0] + r[0][1][0] * a_h[1][0][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[1][0][0][fi + -1][fj + 0][fk + 0] + r[1][0][0] * a_h[0][1][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[0][1][0][fi + 0][fj + -1][fk + 0] + r[1][1][0] * a_h[0][0][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[0][0][0][fi + 0][fj + 0][fk + 0]) * p[0][2][0] + (r[0][0][0] * a_h[1][2][0][fi + -1][fj + -1][fk + -1] + r[0][1][0] * a_h[1][1][0][fi + -1][fj + 0][fk + -1] + r[0][2][0] * a_h[1][0][0][fi + -1][fj + 1][fk + -1] + r[1][0][0] * a_h[0][2][0][fi + 0][fj + -1][fk + -1] + r[1][1][0] * a_h[0][1][0][fi + 0][fj + 0][fk + -1] + r[1][2][0] * a_h[0][0][0][fi + 0][fj + 1][fk + -1]) * p[0][1][1] + (r[0][0][0] * a_h[1][2][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[1][2][0][fi + -1][fj + -1][fk + 0] + r[0][1][0] * a_h[1][1][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[1][1][0][fi + -1][fj + 0][fk + 0] + r[0][2][0] * a_h[1][0][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[1][0][0][fi + -1][fj + 1][fk + 0] + r[1][0][0] * a_h[0][2][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[0][2][0][fi + 0][fj + -1][fk + 0] + r[1][1][0] * a_h[0][1][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[0][1][0][fi + 0][fj + 0][fk + 0] + r[1][2][0] * a_h[0][0][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[0][0][0][fi + 0][fj + 1][fk + 0]) * p[0][1][0] + (r[0][1][0] * a_h[1][2][0][fi + -1][fj + 0][fk + -1] + r[0][2][0] * a_h[1][1][0][fi + -1][fj + 1][fk + -1] + r[1][1][0] * a_h[0][2][0][fi + 0][fj + 0][fk + -1] + r[1][2][0] * a_h[0][1][0][fi + 0][fj + 1][fk + -1]) * p[0][0][1] + (r[0][1][0] * a_h[1][2][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[1][2][0][fi + -1][fj + 0][fk + 0] + r[0][2][0] * a_h[1][1][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[1][1][0][fi + -1][fj + 1][fk + 0] + r[1][1][0] * a_h[0][2][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[0][2][0][fi + 0][fj + 0][fk + 0] + r[1][2][0] * a_h[0][1][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[0][1][0][fi + 0][fj + 1][fk + 0]) * p[0][0][0]));

                REQUIRE(a_2h[0][1][1][ci][cj][ck] == ((r[0][0][0] * a_h[0][1][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[0][1][0][fi + -1][fj + -1][fk + 0] + r[0][1][0] * a_h[0][0][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[0][0][0][fi + -1][fj + 0][fk + 0]) * p[1][2][2] + (r[0][0][0] * a_h[0][1][2][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[0][1][1][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[0][1][0][fi + -1][fj + -1][fk + 1] + r[0][1][0] * a_h[0][0][2][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[0][0][1][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[0][0][0][fi + -1][fj + 0][fk + 1]) * p[1][2][1] + (r[0][0][1] * a_h[0][1][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[0][1][1][fi + -1][fj + -1][fk + 1] + r[0][1][1] * a_h[0][0][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[0][0][1][fi + -1][fj + 0][fk + 1]) * p[1][2][0] + (r[0][0][0] * a_h[0][2][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[0][2][0][fi + -1][fj + -1][fk + 0] + r[0][1][0] * a_h[0][1][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[0][1][0][fi + -1][fj + 0][fk + 0] + r[0][2][0] * a_h[0][0][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[0][0][0][fi + -1][fj + 1][fk + 0]) * p[1][1][2] + (r[0][0][0] * a_h[0][2][2][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[0][2][1][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[0][2][0][fi + -1][fj + -1][fk + 1] + r[0][1][0] * a_h[0][1][2][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[0][1][1][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[0][1][0][fi + -1][fj + 0][fk + 1] + r[0][2][0] * a_h[0][0][2][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[0][0][1][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[0][0][0][fi + -1][fj + 1][fk + 1]) * p[1][1][1] + (r[0][0][1] * a_h[0][2][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[0][2][1][fi + -1][fj + -1][fk + 1] + r[0][1][1] * a_h[0][1][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[0][1][1][fi + -1][fj + 0][fk + 1] + r[0][2][1] * a_h[0][0][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[0][0][1][fi + -1][fj + 1][fk + 1]) * p[1][1][0] + (r[0][1][0] * a_h[0][2][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[0][2][0][fi + -1][fj + 0][fk + 0] + r[0][2][0] * a_h[0][1][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[0][1][0][fi + -1][fj + 1][fk + 0]) * p[1][0][2] + (r[0][1][0] * a_h[0][2][2][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[0][2][1][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[0][2][0][fi + -1][fj + 0][fk + 1] + r[0][2][0] * a_h[0][1][2][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[0][1][1][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[0][1][0][fi + -1][fj + 1][fk + 1]) * p[1][0][1] + (r[0][1][1] * a_h[0][2][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[0][2][1][fi + -1][fj + 0][fk + 1] + r[0][2][1] * a_h[0][1][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[0][1][1][fi + -1][fj + 1][fk + 1]) * p[1][0][0] + (r[0][0][0] * a_h[1][1][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[1][1][0][fi + -1][fj + -1][fk + 0] + r[0][1][0] * a_h[1][0][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[1][0][0][fi + -1][fj + 0][fk + 0] + r[1][0][0] * a_h[0][1][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[0][1][0][fi + 0][fj + -1][fk + 0] + r[1][1][0] * a_h[0][0][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[0][0][0][fi + 0][fj + 0][fk + 0]) * p[0][2][2] + (r[0][0][0] * a_h[1][1][2][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[1][1][1][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[1][1][0][fi + -1][fj + -1][fk + 1] + r[0][1][0] * a_h[1][0][2][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[1][0][1][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[1][0][0][fi + -1][fj + 0][fk + 1] + r[1][0][0] * a_h[0][1][2][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[0][1][1][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[0][1][0][fi + 0][fj + -1][fk + 1] + r[1][1][0] * a_h[0][0][2][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[0][0][1][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[0][0][0][fi + 0][fj + 0][fk + 1]) * p[0][2][1] + (r[0][0][1] * a_h[1][1][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[1][1][1][fi + -1][fj + -1][fk + 1] + r[0][1][1] * a_h[1][0][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[1][0][1][fi + -1][fj + 0][fk + 1] + r[1][0][1] * a_h[0][1][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[0][1][1][fi + 0][fj + -1][fk + 1] + r[1][1][1] * a_h[0][0][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[0][0][1][fi + 0][fj + 0][fk + 1]) * p[0][2][0] + (r[0][0][0] * a_h[1][2][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[1][2][0][fi + -1][fj + -1][fk + 0] + r[0][1][0] * a_h[1][1][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[1][1][0][fi + -1][fj + 0][fk + 0] + r[0][2][0] * a_h[1][0][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[1][0][0][fi + -1][fj + 1][fk + 0] + r[1][0][0] * a_h[0][2][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[0][2][0][fi + 0][fj + -1][fk + 0] + r[1][1][0] * a_h[0][1][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[0][1][0][fi + 0][fj + 0][fk + 0] + r[1][2][0] * a_h[0][0][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[0][0][0][fi + 0][fj + 1][fk + 0]) * p[0][1][2] + (r[0][0][0] * a_h[1][2][2][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[1][2][1][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[1][2][0][fi + -1][fj + -1][fk + 1] + r[0][1][0] * a_h[1][1][2][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[1][1][1][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[1][1][0][fi + -1][fj + 0][fk + 1] + r[0][2][0] * a_h[1][0][2][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[1][0][1][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[1][0][0][fi + -1][fj + 1][fk + 1] + r[1][0][0] * a_h[0][2][2][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[0][2][1][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[0][2][0][fi + 0][fj + -1][fk + 1] + r[1][1][0] * a_h[0][1][2][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[0][1][1][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[0][1][0][fi + 0][fj + 0][fk + 1] + r[1][2][0] * a_h[0][0][2][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[0][0][1][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[0][0][0][fi + 0][fj + 1][fk + 1]) * p[0][1][1] + (r[0][0][1] * a_h[1][2][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[1][2][1][fi + -1][fj + -1][fk + 1] + r[0][1][1] * a_h[1][1][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[1][1][1][fi + -1][fj + 0][fk + 1] + r[0][2][1] * a_h[1][0][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[1][0][1][fi + -1][fj + 1][fk + 1] + r[1][0][1] * a_h[0][2][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[0][2][1][fi + 0][fj + -1][fk + 1] + r[1][1][1] * a_h[0][1][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[0][1][1][fi + 0][fj + 0][fk + 1] + r[1][2][1] * a_h[0][0][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[0][0][1][fi + 0][fj + 1][fk + 1]) * p[0][1][0] + (r[0][1][0] * a_h[1][2][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[1][2][0][fi + -1][fj + 0][fk + 0] + r[0][2][0] * a_h[1][1][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[1][1][0][fi + -1][fj + 1][fk + 0] + r[1][1][0] * a_h[0][2][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[0][2][0][fi + 0][fj + 0][fk + 0] + r[1][2][0] * a_h[0][1][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[0][1][0][fi + 0][fj + 1][fk + 0]) * p[0][0][2] + (r[0][1][0] * a_h[1][2][2][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[1][2][1][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[1][2][0][fi + -1][fj + 0][fk + 1] + r[0][2][0] * a_h[1][1][2][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[1][1][1][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[1][1][0][fi + -1][fj + 1][fk + 1] + r[1][1][0] * a_h[0][2][2][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[0][2][1][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[0][2][0][fi + 0][fj + 0][fk + 1] + r[1][2][0] * a_h[0][1][2][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[0][1][1][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[0][1][0][fi + 0][fj + 1][fk + 1]) * p[0][0][1] + (r[0][1][1] * a_h[1][2][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[1][2][1][fi + -1][fj + 0][fk + 1] + r[0][2][1] * a_h[1][1][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[1][1][1][fi + -1][fj + 1][fk + 1] + r[1][1][1] * a_h[0][2][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[0][2][1][fi + 0][fj + 0][fk + 1] + r[1][2][1] * a_h[0][1][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[0][1][1][fi + 0][fj + 1][fk + 1]) * p[0][0][0]));

                REQUIRE(a_2h[0][1][2][ci][cj][ck] == ((r[0][0][1] * a_h[0][1][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[0][1][1][fi + -1][fj + -1][fk + 1] + r[0][1][1] * a_h[0][0][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[0][0][1][fi + -1][fj + 0][fk + 1]) * p[1][2][2] + (r[0][0][2] * a_h[0][1][2][fi + -1][fj + -1][fk + 1] + r[0][1][2] * a_h[0][0][2][fi + -1][fj + 0][fk + 1]) * p[1][2][1] + (r[0][0][1] * a_h[0][2][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[0][2][1][fi + -1][fj + -1][fk + 1] + r[0][1][1] * a_h[0][1][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[0][1][1][fi + -1][fj + 0][fk + 1] + r[0][2][1] * a_h[0][0][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[0][0][1][fi + -1][fj + 1][fk + 1]) * p[1][1][2] + (r[0][0][2] * a_h[0][2][2][fi + -1][fj + -1][fk + 1] + r[0][1][2] * a_h[0][1][2][fi + -1][fj + 0][fk + 1] + r[0][2][2] * a_h[0][0][2][fi + -1][fj + 1][fk + 1]) * p[1][1][1] + (r[0][1][1] * a_h[0][2][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[0][2][1][fi + -1][fj + 0][fk + 1] + r[0][2][1] * a_h[0][1][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[0][1][1][fi + -1][fj + 1][fk + 1]) * p[1][0][2] + (r[0][1][2] * a_h[0][2][2][fi + -1][fj + 0][fk + 1] + r[0][2][2] * a_h[0][1][2][fi + -1][fj + 1][fk + 1]) * p[1][0][1] + (r[0][0][1] * a_h[1][1][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[1][1][1][fi + -1][fj + -1][fk + 1] + r[0][1][1] * a_h[1][0][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[1][0][1][fi + -1][fj + 0][fk + 1] + r[1][0][1] * a_h[0][1][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[0][1][1][fi + 0][fj + -1][fk + 1] + r[1][1][1] * a_h[0][0][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[0][0][1][fi + 0][fj + 0][fk + 1]) * p[0][2][2] + (r[0][0][2] * a_h[1][1][2][fi + -1][fj + -1][fk + 1] + r[0][1][2] * a_h[1][0][2][fi + -1][fj + 0][fk + 1] + r[1][0][2] * a_h[0][1][2][fi + 0][fj + -1][fk + 1] + r[1][1][2] * a_h[0][0][2][fi + 0][fj + 0][fk + 1]) * p[0][2][1] + (r[0][0][1] * a_h[1][2][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[1][2][1][fi + -1][fj + -1][fk + 1] + r[0][1][1] * a_h[1][1][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[1][1][1][fi + -1][fj + 0][fk + 1] + r[0][2][1] * a_h[1][0][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[1][0][1][fi + -1][fj + 1][fk + 1] + r[1][0][1] * a_h[0][2][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[0][2][1][fi + 0][fj + -1][fk + 1] + r[1][1][1] * a_h[0][1][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[0][1][1][fi + 0][fj + 0][fk + 1] + r[1][2][1] * a_h[0][0][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[0][0][1][fi + 0][fj + 1][fk + 1]) * p[0][1][2] + (r[0][0][2] * a_h[1][2][2][fi + -1][fj + -1][fk + 1] + r[0][1][2] * a_h[1][1][2][fi + -1][fj + 0][fk + 1] + r[0][2][2] * a_h[1][0][2][fi + -1][fj + 1][fk + 1] + r[1][0][2] * a_h[0][2][2][fi + 0][fj + -1][fk + 1] + r[1][1][2] * a_h[0][1][2][fi + 0][fj + 0][fk + 1] + r[1][2][2] * a_h[0][0][2][fi + 0][fj + 1][fk + 1]) * p[0][1][1] + (r[0][1][1] * a_h[1][2][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[1][2][1][fi + -1][fj + 0][fk + 1] + r[0][2][1] * a_h[1][1][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[1][1][1][fi + -1][fj + 1][fk + 1] + r[1][1][1] * a_h[0][2][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[0][2][1][fi + 0][fj + 0][fk + 1] + r[1][2][1] * a_h[0][1][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[0][1][1][fi + 0][fj + 1][fk + 1]) * p[0][0][2] + (r[0][1][2] * a_h[1][2][2][fi + -1][fj + 0][fk + 1] + r[0][2][2] * a_h[1][1][2][fi + -1][fj + 1][fk + 1] + r[1][1][2] * a_h[0][2][2][fi + 0][fj + 0][fk + 1] + r[1][2][2] * a_h[0][1][2][fi + 0][fj + 1][fk + 1]) * p[0][0][1]));

                REQUIRE(a_2h[0][2][0][ci][cj][ck] == ((r[0][1][0] * a_h[0][2][0][fi + -1][fj + 0][fk + -1] + r[0][2][0] * a_h[0][1][0][fi + -1][fj + 1][fk + -1]) * p[1][2][1] + (r[0][1][0] * a_h[0][2][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[0][2][0][fi + -1][fj + 0][fk + 0] + r[0][2][0] * a_h[0][1][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[0][1][0][fi + -1][fj + 1][fk + 0]) * p[1][2][0] + (r[0][2][0] * a_h[0][2][0][fi + -1][fj + 1][fk + -1]) * p[1][1][1] + (r[0][2][0] * a_h[0][2][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[0][2][0][fi + -1][fj + 1][fk + 0]) * p[1][1][0] + (r[0][1][0] * a_h[1][2][0][fi + -1][fj + 0][fk + -1] + r[0][2][0] * a_h[1][1][0][fi + -1][fj + 1][fk + -1] + r[1][1][0] * a_h[0][2][0][fi + 0][fj + 0][fk + -1] + r[1][2][0] * a_h[0][1][0][fi + 0][fj + 1][fk + -1]) * p[0][2][1] + (r[0][1][0] * a_h[1][2][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[1][2][0][fi + -1][fj + 0][fk + 0] + r[0][2][0] * a_h[1][1][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[1][1][0][fi + -1][fj + 1][fk + 0] + r[1][1][0] * a_h[0][2][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[0][2][0][fi + 0][fj + 0][fk + 0] + r[1][2][0] * a_h[0][1][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[0][1][0][fi + 0][fj + 1][fk + 0]) * p[0][2][0] + (r[0][2][0] * a_h[1][2][0][fi + -1][fj + 1][fk + -1] + r[1][2][0] * a_h[0][2][0][fi + 0][fj + 1][fk + -1]) * p[0][1][1] + (r[0][2][0] * a_h[1][2][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[1][2][0][fi + -1][fj + 1][fk + 0] + r[1][2][0] * a_h[0][2][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[0][2][0][fi + 0][fj + 1][fk + 0]) * p[0][1][0]));

                REQUIRE(a_2h[0][2][1][ci][cj][ck] == ((r[0][1][0] * a_h[0][2][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[0][2][0][fi + -1][fj + 0][fk + 0] + r[0][2][0] * a_h[0][1][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[0][1][0][fi + -1][fj + 1][fk + 0]) * p[1][2][2] + (r[0][1][0] * a_h[0][2][2][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[0][2][1][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[0][2][0][fi + -1][fj + 0][fk + 1] + r[0][2][0] * a_h[0][1][2][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[0][1][1][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[0][1][0][fi + -1][fj + 1][fk + 1]) * p[1][2][1] + (r[0][1][1] * a_h[0][2][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[0][2][1][fi + -1][fj + 0][fk + 1] + r[0][2][1] * a_h[0][1][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[0][1][1][fi + -1][fj + 1][fk + 1]) * p[1][2][0] + (r[0][2][0] * a_h[0][2][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[0][2][0][fi + -1][fj + 1][fk + 0]) * p[1][1][2] + (r[0][2][0] * a_h[0][2][2][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[0][2][1][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[0][2][0][fi + -1][fj + 1][fk + 1]) * p[1][1][1] + (r[0][2][1] * a_h[0][2][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[0][2][1][fi + -1][fj + 1][fk + 1]) * p[1][1][0] + (r[0][1][0] * a_h[1][2][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[1][2][0][fi + -1][fj + 0][fk + 0] + r[0][2][0] * a_h[1][1][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[1][1][0][fi + -1][fj + 1][fk + 0] + r[1][1][0] * a_h[0][2][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[0][2][0][fi + 0][fj + 0][fk + 0] + r[1][2][0] * a_h[0][1][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[0][1][0][fi + 0][fj + 1][fk + 0]) * p[0][2][2] + (r[0][1][0] * a_h[1][2][2][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[1][2][1][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[1][2][0][fi + -1][fj + 0][fk + 1] + r[0][2][0] * a_h[1][1][2][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[1][1][1][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[1][1][0][fi + -1][fj + 1][fk + 1] + r[1][1][0] * a_h[0][2][2][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[0][2][1][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[0][2][0][fi + 0][fj + 0][fk + 1] + r[1][2][0] * a_h[0][1][2][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[0][1][1][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[0][1][0][fi + 0][fj + 1][fk + 1]) * p[0][2][1] + (r[0][1][1] * a_h[1][2][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[1][2][1][fi + -1][fj + 0][fk + 1] + r[0][2][1] * a_h[1][1][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[1][1][1][fi + -1][fj + 1][fk + 1] + r[1][1][1] * a_h[0][2][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[0][2][1][fi + 0][fj + 0][fk + 1] + r[1][2][1] * a_h[0][1][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[0][1][1][fi + 0][fj + 1][fk + 1]) * p[0][2][0] + (r[0][2][0] * a_h[1][2][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[1][2][0][fi + -1][fj + 1][fk + 0] + r[1][2][0] * a_h[0][2][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[0][2][0][fi + 0][fj + 1][fk + 0]) * p[0][1][2] + (r[0][2][0] * a_h[1][2][2][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[1][2][1][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[1][2][0][fi + -1][fj + 1][fk + 1] + r[1][2][0] * a_h[0][2][2][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[0][2][1][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[0][2][0][fi + 0][fj + 1][fk + 1]) * p[0][1][1] + (r[0][2][1] * a_h[1][2][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[1][2][1][fi + -1][fj + 1][fk + 1] + r[1][2][1] * a_h[0][2][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[0][2][1][fi + 0][fj + 1][fk + 1]) * p[0][1][0]));

                REQUIRE(a_2h[0][2][2][ci][cj][ck] == ((r[0][1][1] * a_h[0][2][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[0][2][1][fi + -1][fj + 0][fk + 1] + r[0][2][1] * a_h[0][1][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[0][1][1][fi + -1][fj + 1][fk + 1]) * p[1][2][2] + (r[0][1][2] * a_h[0][2][2][fi + -1][fj + 0][fk + 1] + r[0][2][2] * a_h[0][1][2][fi + -1][fj + 1][fk + 1]) * p[1][2][1] + (r[0][2][1] * a_h[0][2][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[0][2][1][fi + -1][fj + 1][fk + 1]) * p[1][1][2] + (r[0][2][2] * a_h[0][2][2][fi + -1][fj + 1][fk + 1]) * p[1][1][1] + (r[0][1][1] * a_h[1][2][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[1][2][1][fi + -1][fj + 0][fk + 1] + r[0][2][1] * a_h[1][1][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[1][1][1][fi + -1][fj + 1][fk + 1] + r[1][1][1] * a_h[0][2][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[0][2][1][fi + 0][fj + 0][fk + 1] + r[1][2][1] * a_h[0][1][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[0][1][1][fi + 0][fj + 1][fk + 1]) * p[0][2][2] + (r[0][1][2] * a_h[1][2][2][fi + -1][fj + 0][fk + 1] + r[0][2][2] * a_h[1][1][2][fi + -1][fj + 1][fk + 1] + r[1][1][2] * a_h[0][2][2][fi + 0][fj + 0][fk + 1] + r[1][2][2] * a_h[0][1][2][fi + 0][fj + 1][fk + 1]) * p[0][2][1] + (r[0][2][1] * a_h[1][2][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[1][2][1][fi + -1][fj + 1][fk + 1] + r[1][2][1] * a_h[0][2][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[0][2][1][fi + 0][fj + 1][fk + 1]) * p[0][1][2] + (r[0][2][2] * a_h[1][2][2][fi + -1][fj + 1][fk + 1] + r[1][2][2] * a_h[0][2][2][fi + 0][fj + 1][fk + 1]) * p[0][1][1]));

                REQUIRE(a_2h[1][0][0][ci][cj][ck] == ((r[0][0][0] * a_h[1][0][0][fi + -1][fj + -1][fk + -1] + r[1][0][0] * a_h[0][0][0][fi + 0][fj + -1][fk + -1]) * p[2][1][1] + (r[0][0][0] * a_h[1][0][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[1][0][0][fi + -1][fj + -1][fk + 0] + r[1][0][0] * a_h[0][0][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[0][0][0][fi + 0][fj + -1][fk + 0]) * p[2][1][0] + (r[0][0][0] * a_h[1][1][0][fi + -1][fj + -1][fk + -1] + r[0][1][0] * a_h[1][0][0][fi + -1][fj + 0][fk + -1] + r[1][0][0] * a_h[0][1][0][fi + 0][fj + -1][fk + -1] + r[1][1][0] * a_h[0][0][0][fi + 0][fj + 0][fk + -1]) * p[2][0][1] + (r[0][0][0] * a_h[1][1][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[1][1][0][fi + -1][fj + -1][fk + 0] + r[0][1][0] * a_h[1][0][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[1][0][0][fi + -1][fj + 0][fk + 0] + r[1][0][0] * a_h[0][1][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[0][1][0][fi + 0][fj + -1][fk + 0] + r[1][1][0] * a_h[0][0][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[0][0][0][fi + 0][fj + 0][fk + 0]) * p[2][0][0] + (r[0][0][0] * a_h[2][0][0][fi + -1][fj + -1][fk + -1] + r[1][0][0] * a_h[1][0][0][fi + 0][fj + -1][fk + -1] + r[2][0][0] * a_h[0][0][0][fi + 1][fj + -1][fk + -1]) * p[1][1][1] + (r[0][0][0] * a_h[2][0][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[2][0][0][fi + -1][fj + -1][fk + 0] + r[1][0][0] * a_h[1][0][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[1][0][0][fi + 0][fj + -1][fk + 0] + r[2][0][0] * a_h[0][0][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[0][0][0][fi + 1][fj + -1][fk + 0]) * p[1][1][0] + (r[0][0][0] * a_h[2][1][0][fi + -1][fj + -1][fk + -1] + r[0][1][0] * a_h[2][0][0][fi + -1][fj + 0][fk + -1] + r[1][0][0] * a_h[1][1][0][fi + 0][fj + -1][fk + -1] + r[1][1][0] * a_h[1][0][0][fi + 0][fj + 0][fk + -1] + r[2][0][0] * a_h[0][1][0][fi + 1][fj + -1][fk + -1] + r[2][1][0] * a_h[0][0][0][fi + 1][fj + 0][fk + -1]) * p[1][0][1] + (r[0][0][0] * a_h[2][1][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[2][1][0][fi + -1][fj + -1][fk + 0] + r[0][1][0] * a_h[2][0][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[2][0][0][fi + -1][fj + 0][fk + 0] + r[1][0][0] * a_h[1][1][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[1][1][0][fi + 0][fj + -1][fk + 0] + r[1][1][0] * a_h[1][0][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[1][0][0][fi + 0][fj + 0][fk + 0] + r[2][0][0] * a_h[0][1][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[0][1][0][fi + 1][fj + -1][fk + 0] + r[2][1][0] * a_h[0][0][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[0][0][0][fi + 1][fj + 0][fk + 0]) * p[1][0][0] + (r[1][0][0] * a_h[2][0][0][fi + 0][fj + -1][fk + -1] + r[2][0][0] * a_h[1][0][0][fi + 1][fj + -1][fk + -1]) * p[0][1][1] + (r[1][0][0] * a_h[2][0][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[2][0][0][fi + 0][fj + -1][fk + 0] + r[2][0][0] * a_h[1][0][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[1][0][0][fi + 1][fj + -1][fk + 0]) * p[0][1][0] + (r[1][0][0] * a_h[2][1][0][fi + 0][fj + -1][fk + -1] + r[1][1][0] * a_h[2][0][0][fi + 0][fj + 0][fk + -1] + r[2][0][0] * a_h[1][1][0][fi + 1][fj + -1][fk + -1] + r[2][1][0] * a_h[1][0][0][fi + 1][fj + 0][fk + -1]) * p[0][0][1] + (r[1][0][0] * a_h[2][1][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[2][1][0][fi + 0][fj + -1][fk + 0] + r[1][1][0] * a_h[2][0][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[2][0][0][fi + 0][fj + 0][fk + 0] + r[2][0][0] * a_h[1][1][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[1][1][0][fi + 1][fj + -1][fk + 0] + r[2][1][0] * a_h[1][0][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[1][0][0][fi + 1][fj + 0][fk + 0]) * p[0][0][0]));

                REQUIRE(a_2h[1][0][1][ci][cj][ck] == ((r[0][0][0] * a_h[1][0][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[1][0][0][fi + -1][fj + -1][fk + 0] + r[1][0][0] * a_h[0][0][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[0][0][0][fi + 0][fj + -1][fk + 0]) * p[2][1][2] + (r[0][0][0] * a_h[1][0][2][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[1][0][1][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[1][0][0][fi + -1][fj + -1][fk + 1] + r[1][0][0] * a_h[0][0][2][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[0][0][1][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[0][0][0][fi + 0][fj + -1][fk + 1]) * p[2][1][1] + (r[0][0][1] * a_h[1][0][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[1][0][1][fi + -1][fj + -1][fk + 1] + r[1][0][1] * a_h[0][0][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[0][0][1][fi + 0][fj + -1][fk + 1]) * p[2][1][0] + (r[0][0][0] * a_h[1][1][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[1][1][0][fi + -1][fj + -1][fk + 0] + r[0][1][0] * a_h[1][0][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[1][0][0][fi + -1][fj + 0][fk + 0] + r[1][0][0] * a_h[0][1][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[0][1][0][fi + 0][fj + -1][fk + 0] + r[1][1][0] * a_h[0][0][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[0][0][0][fi + 0][fj + 0][fk + 0]) * p[2][0][2] + (r[0][0][0] * a_h[1][1][2][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[1][1][1][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[1][1][0][fi + -1][fj + -1][fk + 1] + r[0][1][0] * a_h[1][0][2][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[1][0][1][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[1][0][0][fi + -1][fj + 0][fk + 1] + r[1][0][0] * a_h[0][1][2][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[0][1][1][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[0][1][0][fi + 0][fj + -1][fk + 1] + r[1][1][0] * a_h[0][0][2][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[0][0][1][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[0][0][0][fi + 0][fj + 0][fk + 1]) * p[2][0][1] + (r[0][0][1] * a_h[1][1][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[1][1][1][fi + -1][fj + -1][fk + 1] + r[0][1][1] * a_h[1][0][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[1][0][1][fi + -1][fj + 0][fk + 1] + r[1][0][1] * a_h[0][1][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[0][1][1][fi + 0][fj + -1][fk + 1] + r[1][1][1] * a_h[0][0][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[0][0][1][fi + 0][fj + 0][fk + 1]) * p[2][0][0] + (r[0][0][0] * a_h[2][0][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[2][0][0][fi + -1][fj + -1][fk + 0] + r[1][0][0] * a_h[1][0][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[1][0][0][fi + 0][fj + -1][fk + 0] + r[2][0][0] * a_h[0][0][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[0][0][0][fi + 1][fj + -1][fk + 0]) * p[1][1][2] + (r[0][0][0] * a_h[2][0][2][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[2][0][1][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[2][0][0][fi + -1][fj + -1][fk + 1] + r[1][0][0] * a_h[1][0][2][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[1][0][1][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[1][0][0][fi + 0][fj + -1][fk + 1] + r[2][0][0] * a_h[0][0][2][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[0][0][1][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[0][0][0][fi + 1][fj + -1][fk + 1]) * p[1][1][1] + (r[0][0][1] * a_h[2][0][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[2][0][1][fi + -1][fj + -1][fk + 1] + r[1][0][1] * a_h[1][0][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[1][0][1][fi + 0][fj + -1][fk + 1] + r[2][0][1] * a_h[0][0][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[0][0][1][fi + 1][fj + -1][fk + 1]) * p[1][1][0] + (r[0][0][0] * a_h[2][1][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[2][1][0][fi + -1][fj + -1][fk + 0] + r[0][1][0] * a_h[2][0][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[2][0][0][fi + -1][fj + 0][fk + 0] + r[1][0][0] * a_h[1][1][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[1][1][0][fi + 0][fj + -1][fk + 0] + r[1][1][0] * a_h[1][0][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[1][0][0][fi + 0][fj + 0][fk + 0] + r[2][0][0] * a_h[0][1][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[0][1][0][fi + 1][fj + -1][fk + 0] + r[2][1][0] * a_h[0][0][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[0][0][0][fi + 1][fj + 0][fk + 0]) * p[1][0][2] + (r[0][0][0] * a_h[2][1][2][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[2][1][1][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[2][1][0][fi + -1][fj + -1][fk + 1] + r[0][1][0] * a_h[2][0][2][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[2][0][1][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[2][0][0][fi + -1][fj + 0][fk + 1] + r[1][0][0] * a_h[1][1][2][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[1][1][1][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[1][1][0][fi + 0][fj + -1][fk + 1] + r[1][1][0] * a_h[1][0][2][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[1][0][1][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[1][0][0][fi + 0][fj + 0][fk + 1] + r[2][0][0] * a_h[0][1][2][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[0][1][1][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[0][1][0][fi + 1][fj + -1][fk + 1] + r[2][1][0] * a_h[0][0][2][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[0][0][1][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[0][0][0][fi + 1][fj + 0][fk + 1]) * p[1][0][1] + (r[0][0][1] * a_h[2][1][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[2][1][1][fi + -1][fj + -1][fk + 1] + r[0][1][1] * a_h[2][0][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[2][0][1][fi + -1][fj + 0][fk + 1] + r[1][0][1] * a_h[1][1][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[1][1][1][fi + 0][fj + -1][fk + 1] + r[1][1][1] * a_h[1][0][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[1][0][1][fi + 0][fj + 0][fk + 1] + r[2][0][1] * a_h[0][1][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[0][1][1][fi + 1][fj + -1][fk + 1] + r[2][1][1] * a_h[0][0][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[0][0][1][fi + 1][fj + 0][fk + 1]) * p[1][0][0] + (r[1][0][0] * a_h[2][0][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[2][0][0][fi + 0][fj + -1][fk + 0] + r[2][0][0] * a_h[1][0][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[1][0][0][fi + 1][fj + -1][fk + 0]) * p[0][1][2] + (r[1][0][0] * a_h[2][0][2][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[2][0][1][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[2][0][0][fi + 0][fj + -1][fk + 1] + r[2][0][0] * a_h[1][0][2][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[1][0][1][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[1][0][0][fi + 1][fj + -1][fk + 1]) * p[0][1][1] + (r[1][0][1] * a_h[2][0][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[2][0][1][fi + 0][fj + -1][fk + 1] + r[2][0][1] * a_h[1][0][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[1][0][1][fi + 1][fj + -1][fk + 1]) * p[0][1][0] + (r[1][0][0] * a_h[2][1][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[2][1][0][fi + 0][fj + -1][fk + 0] + r[1][1][0] * a_h[2][0][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[2][0][0][fi + 0][fj + 0][fk + 0] + r[2][0][0] * a_h[1][1][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[1][1][0][fi + 1][fj + -1][fk + 0] + r[2][1][0] * a_h[1][0][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[1][0][0][fi + 1][fj + 0][fk + 0]) * p[0][0][2] + (r[1][0][0] * a_h[2][1][2][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[2][1][1][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[2][1][0][fi + 0][fj + -1][fk + 1] + r[1][1][0] * a_h[2][0][2][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[2][0][1][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[2][0][0][fi + 0][fj + 0][fk + 1] + r[2][0][0] * a_h[1][1][2][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[1][1][1][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[1][1][0][fi + 1][fj + -1][fk + 1] + r[2][1][0] * a_h[1][0][2][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[1][0][1][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[1][0][0][fi + 1][fj + 0][fk + 1]) * p[0][0][1] + (r[1][0][1] * a_h[2][1][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[2][1][1][fi + 0][fj + -1][fk + 1] + r[1][1][1] * a_h[2][0][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[2][0][1][fi + 0][fj + 0][fk + 1] + r[2][0][1] * a_h[1][1][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[1][1][1][fi + 1][fj + -1][fk + 1] + r[2][1][1] * a_h[1][0][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[1][0][1][fi + 1][fj + 0][fk + 1]) * p[0][0][0]));

                REQUIRE(a_2h[1][0][2][ci][cj][ck] == ((r[0][0][1] * a_h[1][0][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[1][0][1][fi + -1][fj + -1][fk + 1] + r[1][0][1] * a_h[0][0][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[0][0][1][fi + 0][fj + -1][fk + 1]) * p[2][1][2] + (r[0][0][2] * a_h[1][0][2][fi + -1][fj + -1][fk + 1] + r[1][0][2] * a_h[0][0][2][fi + 0][fj + -1][fk + 1]) * p[2][1][1] + (r[0][0][1] * a_h[1][1][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[1][1][1][fi + -1][fj + -1][fk + 1] + r[0][1][1] * a_h[1][0][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[1][0][1][fi + -1][fj + 0][fk + 1] + r[1][0][1] * a_h[0][1][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[0][1][1][fi + 0][fj + -1][fk + 1] + r[1][1][1] * a_h[0][0][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[0][0][1][fi + 0][fj + 0][fk + 1]) * p[2][0][2] + (r[0][0][2] * a_h[1][1][2][fi + -1][fj + -1][fk + 1] + r[0][1][2] * a_h[1][0][2][fi + -1][fj + 0][fk + 1] + r[1][0][2] * a_h[0][1][2][fi + 0][fj + -1][fk + 1] + r[1][1][2] * a_h[0][0][2][fi + 0][fj + 0][fk + 1]) * p[2][0][1] + (r[0][0][1] * a_h[2][0][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[2][0][1][fi + -1][fj + -1][fk + 1] + r[1][0][1] * a_h[1][0][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[1][0][1][fi + 0][fj + -1][fk + 1] + r[2][0][1] * a_h[0][0][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[0][0][1][fi + 1][fj + -1][fk + 1]) * p[1][1][2] + (r[0][0][2] * a_h[2][0][2][fi + -1][fj + -1][fk + 1] + r[1][0][2] * a_h[1][0][2][fi + 0][fj + -1][fk + 1] + r[2][0][2] * a_h[0][0][2][fi + 1][fj + -1][fk + 1]) * p[1][1][1] + (r[0][0][1] * a_h[2][1][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[2][1][1][fi + -1][fj + -1][fk + 1] + r[0][1][1] * a_h[2][0][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[2][0][1][fi + -1][fj + 0][fk + 1] + r[1][0][1] * a_h[1][1][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[1][1][1][fi + 0][fj + -1][fk + 1] + r[1][1][1] * a_h[1][0][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[1][0][1][fi + 0][fj + 0][fk + 1] + r[2][0][1] * a_h[0][1][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[0][1][1][fi + 1][fj + -1][fk + 1] + r[2][1][1] * a_h[0][0][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[0][0][1][fi + 1][fj + 0][fk + 1]) * p[1][0][2] + (r[0][0][2] * a_h[2][1][2][fi + -1][fj + -1][fk + 1] + r[0][1][2] * a_h[2][0][2][fi + -1][fj + 0][fk + 1] + r[1][0][2] * a_h[1][1][2][fi + 0][fj + -1][fk + 1] + r[1][1][2] * a_h[1][0][2][fi + 0][fj + 0][fk + 1] + r[2][0][2] * a_h[0][1][2][fi + 1][fj + -1][fk + 1] + r[2][1][2] * a_h[0][0][2][fi + 1][fj + 0][fk + 1]) * p[1][0][1] + (r[1][0][1] * a_h[2][0][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[2][0][1][fi + 0][fj + -1][fk + 1] + r[2][0][1] * a_h[1][0][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[1][0][1][fi + 1][fj + -1][fk + 1]) * p[0][1][2] + (r[1][0][2] * a_h[2][0][2][fi + 0][fj + -1][fk + 1] + r[2][0][2] * a_h[1][0][2][fi + 1][fj + -1][fk + 1]) * p[0][1][1] + (r[1][0][1] * a_h[2][1][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[2][1][1][fi + 0][fj + -1][fk + 1] + r[1][1][1] * a_h[2][0][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[2][0][1][fi + 0][fj + 0][fk + 1] + r[2][0][1] * a_h[1][1][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[1][1][1][fi + 1][fj + -1][fk + 1] + r[2][1][1] * a_h[1][0][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[1][0][1][fi + 1][fj + 0][fk + 1]) * p[0][0][2] + (r[1][0][2] * a_h[2][1][2][fi + 0][fj + -1][fk + 1] + r[1][1][2] * a_h[2][0][2][fi + 0][fj + 0][fk + 1] + r[2][0][2] * a_h[1][1][2][fi + 1][fj + -1][fk + 1] + r[2][1][2] * a_h[1][0][2][fi + 1][fj + 0][fk + 1]) * p[0][0][1]));

                REQUIRE(a_2h[1][1][0][ci][cj][ck] == ((r[0][0][0] * a_h[1][1][0][fi + -1][fj + -1][fk + -1] + r[0][1][0] * a_h[1][0][0][fi + -1][fj + 0][fk + -1] + r[1][0][0] * a_h[0][1][0][fi + 0][fj + -1][fk + -1] + r[1][1][0] * a_h[0][0][0][fi + 0][fj + 0][fk + -1]) * p[2][2][1] + (r[0][0][0] * a_h[1][1][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[1][1][0][fi + -1][fj + -1][fk + 0] + r[0][1][0] * a_h[1][0][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[1][0][0][fi + -1][fj + 0][fk + 0] + r[1][0][0] * a_h[0][1][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[0][1][0][fi + 0][fj + -1][fk + 0] + r[1][1][0] * a_h[0][0][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[0][0][0][fi + 0][fj + 0][fk + 0]) * p[2][2][0] + (r[0][0][0] * a_h[1][2][0][fi + -1][fj + -1][fk + -1] + r[0][1][0] * a_h[1][1][0][fi + -1][fj + 0][fk + -1] + r[0][2][0] * a_h[1][0][0][fi + -1][fj + 1][fk + -1] + r[1][0][0] * a_h[0][2][0][fi + 0][fj + -1][fk + -1] + r[1][1][0] * a_h[0][1][0][fi + 0][fj + 0][fk + -1] + r[1][2][0] * a_h[0][0][0][fi + 0][fj + 1][fk + -1]) * p[2][1][1] + (r[0][0][0] * a_h[1][2][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[1][2][0][fi + -1][fj + -1][fk + 0] + r[0][1][0] * a_h[1][1][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[1][1][0][fi + -1][fj + 0][fk + 0] + r[0][2][0] * a_h[1][0][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[1][0][0][fi + -1][fj + 1][fk + 0] + r[1][0][0] * a_h[0][2][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[0][2][0][fi + 0][fj + -1][fk + 0] + r[1][1][0] * a_h[0][1][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[0][1][0][fi + 0][fj + 0][fk + 0] + r[1][2][0] * a_h[0][0][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[0][0][0][fi + 0][fj + 1][fk + 0]) * p[2][1][0] + (r[0][1][0] * a_h[1][2][0][fi + -1][fj + 0][fk + -1] + r[0][2][0] * a_h[1][1][0][fi + -1][fj + 1][fk + -1] + r[1][1][0] * a_h[0][2][0][fi + 0][fj + 0][fk + -1] + r[1][2][0] * a_h[0][1][0][fi + 0][fj + 1][fk + -1]) * p[2][0][1] + (r[0][1][0] * a_h[1][2][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[1][2][0][fi + -1][fj + 0][fk + 0] + r[0][2][0] * a_h[1][1][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[1][1][0][fi + -1][fj + 1][fk + 0] + r[1][1][0] * a_h[0][2][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[0][2][0][fi + 0][fj + 0][fk + 0] + r[1][2][0] * a_h[0][1][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[0][1][0][fi + 0][fj + 1][fk + 0]) * p[2][0][0] + (r[0][0][0] * a_h[2][1][0][fi + -1][fj + -1][fk + -1] + r[0][1][0] * a_h[2][0][0][fi + -1][fj + 0][fk + -1] + r[1][0][0] * a_h[1][1][0][fi + 0][fj + -1][fk + -1] + r[1][1][0] * a_h[1][0][0][fi + 0][fj + 0][fk + -1] + r[2][0][0] * a_h[0][1][0][fi + 1][fj + -1][fk + -1] + r[2][1][0] * a_h[0][0][0][fi + 1][fj + 0][fk + -1]) * p[1][2][1] + (r[0][0][0] * a_h[2][1][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[2][1][0][fi + -1][fj + -1][fk + 0] + r[0][1][0] * a_h[2][0][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[2][0][0][fi + -1][fj + 0][fk + 0] + r[1][0][0] * a_h[1][1][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[1][1][0][fi + 0][fj + -1][fk + 0] + r[1][1][0] * a_h[1][0][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[1][0][0][fi + 0][fj + 0][fk + 0] + r[2][0][0] * a_h[0][1][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[0][1][0][fi + 1][fj + -1][fk + 0] + r[2][1][0] * a_h[0][0][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[0][0][0][fi + 1][fj + 0][fk + 0]) * p[1][2][0] + (r[0][0][0] * a_h[2][2][0][fi + -1][fj + -1][fk + -1] + r[0][1][0] * a_h[2][1][0][fi + -1][fj + 0][fk + -1] + r[0][2][0] * a_h[2][0][0][fi + -1][fj + 1][fk + -1] + r[1][0][0] * a_h[1][2][0][fi + 0][fj + -1][fk + -1] + r[1][1][0] * a_h[1][1][0][fi + 0][fj + 0][fk + -1] + r[1][2][0] * a_h[1][0][0][fi + 0][fj + 1][fk + -1] + r[2][0][0] * a_h[0][2][0][fi + 1][fj + -1][fk + -1] + r[2][1][0] * a_h[0][1][0][fi + 1][fj + 0][fk + -1] + r[2][2][0] * a_h[0][0][0][fi + 1][fj + 1][fk + -1]) * p[1][1][1] + (r[0][0][0] * a_h[2][2][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[2][2][0][fi + -1][fj + -1][fk + 0] + r[0][1][0] * a_h[2][1][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[2][1][0][fi + -1][fj + 0][fk + 0] + r[0][2][0] * a_h[2][0][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[2][0][0][fi + -1][fj + 1][fk + 0] + r[1][0][0] * a_h[1][2][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[1][2][0][fi + 0][fj + -1][fk + 0] + r[1][1][0] * a_h[1][1][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[1][1][0][fi + 0][fj + 0][fk + 0] + r[1][2][0] * a_h[1][0][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[1][0][0][fi + 0][fj + 1][fk + 0] + r[2][0][0] * a_h[0][2][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[0][2][0][fi + 1][fj + -1][fk + 0] + r[2][1][0] * a_h[0][1][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[0][1][0][fi + 1][fj + 0][fk + 0] + r[2][2][0] * a_h[0][0][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[0][0][0][fi + 1][fj + 1][fk + 0]) * p[1][1][0] + (r[0][1][0] * a_h[2][2][0][fi + -1][fj + 0][fk + -1] + r[0][2][0] * a_h[2][1][0][fi + -1][fj + 1][fk + -1] + r[1][1][0] * a_h[1][2][0][fi + 0][fj + 0][fk + -1] + r[1][2][0] * a_h[1][1][0][fi + 0][fj + 1][fk + -1] + r[2][1][0] * a_h[0][2][0][fi + 1][fj + 0][fk + -1] + r[2][2][0] * a_h[0][1][0][fi + 1][fj + 1][fk + -1]) * p[1][0][1] + (r[0][1][0] * a_h[2][2][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[2][2][0][fi + -1][fj + 0][fk + 0] + r[0][2][0] * a_h[2][1][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[2][1][0][fi + -1][fj + 1][fk + 0] + r[1][1][0] * a_h[1][2][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[1][2][0][fi + 0][fj + 0][fk + 0] + r[1][2][0] * a_h[1][1][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[1][1][0][fi + 0][fj + 1][fk + 0] + r[2][1][0] * a_h[0][2][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[0][2][0][fi + 1][fj + 0][fk + 0] + r[2][2][0] * a_h[0][1][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[0][1][0][fi + 1][fj + 1][fk + 0]) * p[1][0][0] + (r[1][0][0] * a_h[2][1][0][fi + 0][fj + -1][fk + -1] + r[1][1][0] * a_h[2][0][0][fi + 0][fj + 0][fk + -1] + r[2][0][0] * a_h[1][1][0][fi + 1][fj + -1][fk + -1] + r[2][1][0] * a_h[1][0][0][fi + 1][fj + 0][fk + -1]) * p[0][2][1] + (r[1][0][0] * a_h[2][1][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[2][1][0][fi + 0][fj + -1][fk + 0] + r[1][1][0] * a_h[2][0][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[2][0][0][fi + 0][fj + 0][fk + 0] + r[2][0][0] * a_h[1][1][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[1][1][0][fi + 1][fj + -1][fk + 0] + r[2][1][0] * a_h[1][0][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[1][0][0][fi + 1][fj + 0][fk + 0]) * p[0][2][0] + (r[1][0][0] * a_h[2][2][0][fi + 0][fj + -1][fk + -1] + r[1][1][0] * a_h[2][1][0][fi + 0][fj + 0][fk + -1] + r[1][2][0] * a_h[2][0][0][fi + 0][fj + 1][fk + -1] + r[2][0][0] * a_h[1][2][0][fi + 1][fj + -1][fk + -1] + r[2][1][0] * a_h[1][1][0][fi + 1][fj + 0][fk + -1] + r[2][2][0] * a_h[1][0][0][fi + 1][fj + 1][fk + -1]) * p[0][1][1] + (r[1][0][0] * a_h[2][2][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[2][2][0][fi + 0][fj + -1][fk + 0] + r[1][1][0] * a_h[2][1][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[2][1][0][fi + 0][fj + 0][fk + 0] + r[1][2][0] * a_h[2][0][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[2][0][0][fi + 0][fj + 1][fk + 0] + r[2][0][0] * a_h[1][2][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[1][2][0][fi + 1][fj + -1][fk + 0] + r[2][1][0] * a_h[1][1][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[1][1][0][fi + 1][fj + 0][fk + 0] + r[2][2][0] * a_h[1][0][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[1][0][0][fi + 1][fj + 1][fk + 0]) * p[0][1][0] + (r[1][1][0] * a_h[2][2][0][fi + 0][fj + 0][fk + -1] + r[1][2][0] * a_h[2][1][0][fi + 0][fj + 1][fk + -1] + r[2][1][0] * a_h[1][2][0][fi + 1][fj + 0][fk + -1] + r[2][2][0] * a_h[1][1][0][fi + 1][fj + 1][fk + -1]) * p[0][0][1] + (r[1][1][0] * a_h[2][2][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[2][2][0][fi + 0][fj + 0][fk + 0] + r[1][2][0] * a_h[2][1][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[2][1][0][fi + 0][fj + 1][fk + 0] + r[2][1][0] * a_h[1][2][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[1][2][0][fi + 1][fj + 0][fk + 0] + r[2][2][0] * a_h[1][1][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[1][1][0][fi + 1][fj + 1][fk + 0]) * p[0][0][0]));

                REQUIRE(a_2h[1][1][1][ci][cj][ck] == ((r[0][0][0] * a_h[1][1][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[1][1][0][fi + -1][fj + -1][fk + 0] + r[0][1][0] * a_h[1][0][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[1][0][0][fi + -1][fj + 0][fk + 0] + r[1][0][0] * a_h[0][1][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[0][1][0][fi + 0][fj + -1][fk + 0] + r[1][1][0] * a_h[0][0][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[0][0][0][fi + 0][fj + 0][fk + 0]) * p[2][2][2] + (r[0][0][0] * a_h[1][1][2][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[1][1][1][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[1][1][0][fi + -1][fj + -1][fk + 1] + r[0][1][0] * a_h[1][0][2][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[1][0][1][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[1][0][0][fi + -1][fj + 0][fk + 1] + r[1][0][0] * a_h[0][1][2][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[0][1][1][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[0][1][0][fi + 0][fj + -1][fk + 1] + r[1][1][0] * a_h[0][0][2][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[0][0][1][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[0][0][0][fi + 0][fj + 0][fk + 1]) * p[2][2][1] + (r[0][0][1] * a_h[1][1][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[1][1][1][fi + -1][fj + -1][fk + 1] + r[0][1][1] * a_h[1][0][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[1][0][1][fi + -1][fj + 0][fk + 1] + r[1][0][1] * a_h[0][1][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[0][1][1][fi + 0][fj + -1][fk + 1] + r[1][1][1] * a_h[0][0][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[0][0][1][fi + 0][fj + 0][fk + 1]) * p[2][2][0] + (r[0][0][0] * a_h[1][2][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[1][2][0][fi + -1][fj + -1][fk + 0] + r[0][1][0] * a_h[1][1][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[1][1][0][fi + -1][fj + 0][fk + 0] + r[0][2][0] * a_h[1][0][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[1][0][0][fi + -1][fj + 1][fk + 0] + r[1][0][0] * a_h[0][2][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[0][2][0][fi + 0][fj + -1][fk + 0] + r[1][1][0] * a_h[0][1][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[0][1][0][fi + 0][fj + 0][fk + 0] + r[1][2][0] * a_h[0][0][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[0][0][0][fi + 0][fj + 1][fk + 0]) * p[2][1][2] + (r[0][0][0] * a_h[1][2][2][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[1][2][1][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[1][2][0][fi + -1][fj + -1][fk + 1] + r[0][1][0] * a_h[1][1][2][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[1][1][1][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[1][1][0][fi + -1][fj + 0][fk + 1] + r[0][2][0] * a_h[1][0][2][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[1][0][1][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[1][0][0][fi + -1][fj + 1][fk + 1] + r[1][0][0] * a_h[0][2][2][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[0][2][1][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[0][2][0][fi + 0][fj + -1][fk + 1] + r[1][1][0] * a_h[0][1][2][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[0][1][1][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[0][1][0][fi + 0][fj + 0][fk + 1] + r[1][2][0] * a_h[0][0][2][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[0][0][1][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[0][0][0][fi + 0][fj + 1][fk + 1]) * p[2][1][1] + (r[0][0][1] * a_h[1][2][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[1][2][1][fi + -1][fj + -1][fk + 1] + r[0][1][1] * a_h[1][1][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[1][1][1][fi + -1][fj + 0][fk + 1] + r[0][2][1] * a_h[1][0][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[1][0][1][fi + -1][fj + 1][fk + 1] + r[1][0][1] * a_h[0][2][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[0][2][1][fi + 0][fj + -1][fk + 1] + r[1][1][1] * a_h[0][1][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[0][1][1][fi + 0][fj + 0][fk + 1] + r[1][2][1] * a_h[0][0][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[0][0][1][fi + 0][fj + 1][fk + 1]) * p[2][1][0] + (r[0][1][0] * a_h[1][2][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[1][2][0][fi + -1][fj + 0][fk + 0] + r[0][2][0] * a_h[1][1][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[1][1][0][fi + -1][fj + 1][fk + 0] + r[1][1][0] * a_h[0][2][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[0][2][0][fi + 0][fj + 0][fk + 0] + r[1][2][0] * a_h[0][1][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[0][1][0][fi + 0][fj + 1][fk + 0]) * p[2][0][2] + (r[0][1][0] * a_h[1][2][2][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[1][2][1][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[1][2][0][fi + -1][fj + 0][fk + 1] + r[0][2][0] * a_h[1][1][2][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[1][1][1][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[1][1][0][fi + -1][fj + 1][fk + 1] + r[1][1][0] * a_h[0][2][2][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[0][2][1][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[0][2][0][fi + 0][fj + 0][fk + 1] + r[1][2][0] * a_h[0][1][2][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[0][1][1][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[0][1][0][fi + 0][fj + 1][fk + 1]) * p[2][0][1] + (r[0][1][1] * a_h[1][2][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[1][2][1][fi + -1][fj + 0][fk + 1] + r[0][2][1] * a_h[1][1][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[1][1][1][fi + -1][fj + 1][fk + 1] + r[1][1][1] * a_h[0][2][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[0][2][1][fi + 0][fj + 0][fk + 1] + r[1][2][1] * a_h[0][1][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[0][1][1][fi + 0][fj + 1][fk + 1]) * p[2][0][0] + (r[0][0][0] * a_h[2][1][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[2][1][0][fi + -1][fj + -1][fk + 0] + r[0][1][0] * a_h[2][0][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[2][0][0][fi + -1][fj + 0][fk + 0] + r[1][0][0] * a_h[1][1][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[1][1][0][fi + 0][fj + -1][fk + 0] + r[1][1][0] * a_h[1][0][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[1][0][0][fi + 0][fj + 0][fk + 0] + r[2][0][0] * a_h[0][1][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[0][1][0][fi + 1][fj + -1][fk + 0] + r[2][1][0] * a_h[0][0][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[0][0][0][fi + 1][fj + 0][fk + 0]) * p[1][2][2] + (r[0][0][0] * a_h[2][1][2][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[2][1][1][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[2][1][0][fi + -1][fj + -1][fk + 1] + r[0][1][0] * a_h[2][0][2][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[2][0][1][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[2][0][0][fi + -1][fj + 0][fk + 1] + r[1][0][0] * a_h[1][1][2][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[1][1][1][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[1][1][0][fi + 0][fj + -1][fk + 1] + r[1][1][0] * a_h[1][0][2][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[1][0][1][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[1][0][0][fi + 0][fj + 0][fk + 1] + r[2][0][0] * a_h[0][1][2][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[0][1][1][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[0][1][0][fi + 1][fj + -1][fk + 1] + r[2][1][0] * a_h[0][0][2][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[0][0][1][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[0][0][0][fi + 1][fj + 0][fk + 1]) * p[1][2][1] + (r[0][0][1] * a_h[2][1][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[2][1][1][fi + -1][fj + -1][fk + 1] + r[0][1][1] * a_h[2][0][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[2][0][1][fi + -1][fj + 0][fk + 1] + r[1][0][1] * a_h[1][1][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[1][1][1][fi + 0][fj + -1][fk + 1] + r[1][1][1] * a_h[1][0][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[1][0][1][fi + 0][fj + 0][fk + 1] + r[2][0][1] * a_h[0][1][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[0][1][1][fi + 1][fj + -1][fk + 1] + r[2][1][1] * a_h[0][0][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[0][0][1][fi + 1][fj + 0][fk + 1]) * p[1][2][0] + (r[0][0][0] * a_h[2][2][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[2][2][0][fi + -1][fj + -1][fk + 0] + r[0][1][0] * a_h[2][1][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[2][1][0][fi + -1][fj + 0][fk + 0] + r[0][2][0] * a_h[2][0][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[2][0][0][fi + -1][fj + 1][fk + 0] + r[1][0][0] * a_h[1][2][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[1][2][0][fi + 0][fj + -1][fk + 0] + r[1][1][0] * a_h[1][1][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[1][1][0][fi + 0][fj + 0][fk + 0] + r[1][2][0] * a_h[1][0][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[1][0][0][fi + 0][fj + 1][fk + 0] + r[2][0][0] * a_h[0][2][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[0][2][0][fi + 1][fj + -1][fk + 0] + r[2][1][0] * a_h[0][1][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[0][1][0][fi + 1][fj + 0][fk + 0] + r[2][2][0] * a_h[0][0][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[0][0][0][fi + 1][fj + 1][fk + 0]) * p[1][1][2] + (r[0][0][0] * a_h[2][2][2][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[2][2][1][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[2][2][0][fi + -1][fj + -1][fk + 1] + r[0][1][0] * a_h[2][1][2][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[2][1][1][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[2][1][0][fi + -1][fj + 0][fk + 1] + r[0][2][0] * a_h[2][0][2][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[2][0][1][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[2][0][0][fi + -1][fj + 1][fk + 1] + r[1][0][0] * a_h[1][2][2][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[1][2][1][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[1][2][0][fi + 0][fj + -1][fk + 1] + r[1][1][0] * a_h[1][1][2][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[1][1][1][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[1][1][0][fi + 0][fj + 0][fk + 1] + r[1][2][0] * a_h[1][0][2][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[1][0][1][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[1][0][0][fi + 0][fj + 1][fk + 1] + r[2][0][0] * a_h[0][2][2][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[0][2][1][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[0][2][0][fi + 1][fj + -1][fk + 1] + r[2][1][0] * a_h[0][1][2][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[0][1][1][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[0][1][0][fi + 1][fj + 0][fk + 1] + r[2][2][0] * a_h[0][0][2][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[0][0][1][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[0][0][0][fi + 1][fj + 1][fk + 1]) * p[1][1][1] + (r[0][0][1] * a_h[2][2][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[2][2][1][fi + -1][fj + -1][fk + 1] + r[0][1][1] * a_h[2][1][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[2][1][1][fi + -1][fj + 0][fk + 1] + r[0][2][1] * a_h[2][0][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[2][0][1][fi + -1][fj + 1][fk + 1] + r[1][0][1] * a_h[1][2][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[1][2][1][fi + 0][fj + -1][fk + 1] + r[1][1][1] * a_h[1][1][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[1][1][1][fi + 0][fj + 0][fk + 1] + r[1][2][1] * a_h[1][0][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[1][0][1][fi + 0][fj + 1][fk + 1] + r[2][0][1] * a_h[0][2][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[0][2][1][fi + 1][fj + -1][fk + 1] + r[2][1][1] * a_h[0][1][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[0][1][1][fi + 1][fj + 0][fk + 1] + r[2][2][1] * a_h[0][0][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[0][0][1][fi + 1][fj + 1][fk + 1]) * p[1][1][0] + (r[0][1][0] * a_h[2][2][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[2][2][0][fi + -1][fj + 0][fk + 0] + r[0][2][0] * a_h[2][1][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[2][1][0][fi + -1][fj + 1][fk + 0] + r[1][1][0] * a_h[1][2][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[1][2][0][fi + 0][fj + 0][fk + 0] + r[1][2][0] * a_h[1][1][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[1][1][0][fi + 0][fj + 1][fk + 0] + r[2][1][0] * a_h[0][2][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[0][2][0][fi + 1][fj + 0][fk + 0] + r[2][2][0] * a_h[0][1][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[0][1][0][fi + 1][fj + 1][fk + 0]) * p[1][0][2] + (r[0][1][0] * a_h[2][2][2][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[2][2][1][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[2][2][0][fi + -1][fj + 0][fk + 1] + r[0][2][0] * a_h[2][1][2][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[2][1][1][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[2][1][0][fi + -1][fj + 1][fk + 1] + r[1][1][0] * a_h[1][2][2][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[1][2][1][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[1][2][0][fi + 0][fj + 0][fk + 1] + r[1][2][0] * a_h[1][1][2][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[1][1][1][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[1][1][0][fi + 0][fj + 1][fk + 1] + r[2][1][0] * a_h[0][2][2][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[0][2][1][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[0][2][0][fi + 1][fj + 0][fk + 1] + r[2][2][0] * a_h[0][1][2][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[0][1][1][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[0][1][0][fi + 1][fj + 1][fk + 1]) * p[1][0][1] + (r[0][1][1] * a_h[2][2][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[2][2][1][fi + -1][fj + 0][fk + 1] + r[0][2][1] * a_h[2][1][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[2][1][1][fi + -1][fj + 1][fk + 1] + r[1][1][1] * a_h[1][2][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[1][2][1][fi + 0][fj + 0][fk + 1] + r[1][2][1] * a_h[1][1][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[1][1][1][fi + 0][fj + 1][fk + 1] + r[2][1][1] * a_h[0][2][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[0][2][1][fi + 1][fj + 0][fk + 1] + r[2][2][1] * a_h[0][1][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[0][1][1][fi + 1][fj + 1][fk + 1]) * p[1][0][0] + (r[1][0][0] * a_h[2][1][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[2][1][0][fi + 0][fj + -1][fk + 0] + r[1][1][0] * a_h[2][0][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[2][0][0][fi + 0][fj + 0][fk + 0] + r[2][0][0] * a_h[1][1][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[1][1][0][fi + 1][fj + -1][fk + 0] + r[2][1][0] * a_h[1][0][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[1][0][0][fi + 1][fj + 0][fk + 0]) * p[0][2][2] + (r[1][0][0] * a_h[2][1][2][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[2][1][1][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[2][1][0][fi + 0][fj + -1][fk + 1] + r[1][1][0] * a_h[2][0][2][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[2][0][1][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[2][0][0][fi + 0][fj + 0][fk + 1] + r[2][0][0] * a_h[1][1][2][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[1][1][1][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[1][1][0][fi + 1][fj + -1][fk + 1] + r[2][1][0] * a_h[1][0][2][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[1][0][1][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[1][0][0][fi + 1][fj + 0][fk + 1]) * p[0][2][1] + (r[1][0][1] * a_h[2][1][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[2][1][1][fi + 0][fj + -1][fk + 1] + r[1][1][1] * a_h[2][0][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[2][0][1][fi + 0][fj + 0][fk + 1] + r[2][0][1] * a_h[1][1][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[1][1][1][fi + 1][fj + -1][fk + 1] + r[2][1][1] * a_h[1][0][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[1][0][1][fi + 1][fj + 0][fk + 1]) * p[0][2][0] + (r[1][0][0] * a_h[2][2][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[2][2][0][fi + 0][fj + -1][fk + 0] + r[1][1][0] * a_h[2][1][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[2][1][0][fi + 0][fj + 0][fk + 0] + r[1][2][0] * a_h[2][0][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[2][0][0][fi + 0][fj + 1][fk + 0] + r[2][0][0] * a_h[1][2][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[1][2][0][fi + 1][fj + -1][fk + 0] + r[2][1][0] * a_h[1][1][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[1][1][0][fi + 1][fj + 0][fk + 0] + r[2][2][0] * a_h[1][0][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[1][0][0][fi + 1][fj + 1][fk + 0]) * p[0][1][2] + (r[1][0][0] * a_h[2][2][2][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[2][2][1][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[2][2][0][fi + 0][fj + -1][fk + 1] + r[1][1][0] * a_h[2][1][2][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[2][1][1][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[2][1][0][fi + 0][fj + 0][fk + 1] + r[1][2][0] * a_h[2][0][2][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[2][0][1][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[2][0][0][fi + 0][fj + 1][fk + 1] + r[2][0][0] * a_h[1][2][2][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[1][2][1][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[1][2][0][fi + 1][fj + -1][fk + 1] + r[2][1][0] * a_h[1][1][2][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[1][1][1][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[1][1][0][fi + 1][fj + 0][fk + 1] + r[2][2][0] * a_h[1][0][2][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[1][0][1][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[1][0][0][fi + 1][fj + 1][fk + 1]) * p[0][1][1] + (r[1][0][1] * a_h[2][2][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[2][2][1][fi + 0][fj + -1][fk + 1] + r[1][1][1] * a_h[2][1][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[2][1][1][fi + 0][fj + 0][fk + 1] + r[1][2][1] * a_h[2][0][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[2][0][1][fi + 0][fj + 1][fk + 1] + r[2][0][1] * a_h[1][2][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[1][2][1][fi + 1][fj + -1][fk + 1] + r[2][1][1] * a_h[1][1][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[1][1][1][fi + 1][fj + 0][fk + 1] + r[2][2][1] * a_h[1][0][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[1][0][1][fi + 1][fj + 1][fk + 1]) * p[0][1][0] + (r[1][1][0] * a_h[2][2][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[2][2][0][fi + 0][fj + 0][fk + 0] + r[1][2][0] * a_h[2][1][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[2][1][0][fi + 0][fj + 1][fk + 0] + r[2][1][0] * a_h[1][2][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[1][2][0][fi + 1][fj + 0][fk + 0] + r[2][2][0] * a_h[1][1][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[1][1][0][fi + 1][fj + 1][fk + 0]) * p[0][0][2] + (r[1][1][0] * a_h[2][2][2][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[2][2][1][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[2][2][0][fi + 0][fj + 0][fk + 1] + r[1][2][0] * a_h[2][1][2][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[2][1][1][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[2][1][0][fi + 0][fj + 1][fk + 1] + r[2][1][0] * a_h[1][2][2][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[1][2][1][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[1][2][0][fi + 1][fj + 0][fk + 1] + r[2][2][0] * a_h[1][1][2][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[1][1][1][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[1][1][0][fi + 1][fj + 1][fk + 1]) * p[0][0][1] + (r[1][1][1] * a_h[2][2][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[2][2][1][fi + 0][fj + 0][fk + 1] + r[1][2][1] * a_h[2][1][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[2][1][1][fi + 0][fj + 1][fk + 1] + r[2][1][1] * a_h[1][2][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[1][2][1][fi + 1][fj + 0][fk + 1] + r[2][2][1] * a_h[1][1][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[1][1][1][fi + 1][fj + 1][fk + 1]) * p[0][0][0]));

                REQUIRE(a_2h[1][1][2][ci][cj][ck] == ((r[0][0][1] * a_h[1][1][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[1][1][1][fi + -1][fj + -1][fk + 1] + r[0][1][1] * a_h[1][0][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[1][0][1][fi + -1][fj + 0][fk + 1] + r[1][0][1] * a_h[0][1][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[0][1][1][fi + 0][fj + -1][fk + 1] + r[1][1][1] * a_h[0][0][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[0][0][1][fi + 0][fj + 0][fk + 1]) * p[2][2][2] + (r[0][0][2] * a_h[1][1][2][fi + -1][fj + -1][fk + 1] + r[0][1][2] * a_h[1][0][2][fi + -1][fj + 0][fk + 1] + r[1][0][2] * a_h[0][1][2][fi + 0][fj + -1][fk + 1] + r[1][1][2] * a_h[0][0][2][fi + 0][fj + 0][fk + 1]) * p[2][2][1] + (r[0][0][1] * a_h[1][2][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[1][2][1][fi + -1][fj + -1][fk + 1] + r[0][1][1] * a_h[1][1][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[1][1][1][fi + -1][fj + 0][fk + 1] + r[0][2][1] * a_h[1][0][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[1][0][1][fi + -1][fj + 1][fk + 1] + r[1][0][1] * a_h[0][2][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[0][2][1][fi + 0][fj + -1][fk + 1] + r[1][1][1] * a_h[0][1][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[0][1][1][fi + 0][fj + 0][fk + 1] + r[1][2][1] * a_h[0][0][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[0][0][1][fi + 0][fj + 1][fk + 1]) * p[2][1][2] + (r[0][0][2] * a_h[1][2][2][fi + -1][fj + -1][fk + 1] + r[0][1][2] * a_h[1][1][2][fi + -1][fj + 0][fk + 1] + r[0][2][2] * a_h[1][0][2][fi + -1][fj + 1][fk + 1] + r[1][0][2] * a_h[0][2][2][fi + 0][fj + -1][fk + 1] + r[1][1][2] * a_h[0][1][2][fi + 0][fj + 0][fk + 1] + r[1][2][2] * a_h[0][0][2][fi + 0][fj + 1][fk + 1]) * p[2][1][1] + (r[0][1][1] * a_h[1][2][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[1][2][1][fi + -1][fj + 0][fk + 1] + r[0][2][1] * a_h[1][1][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[1][1][1][fi + -1][fj + 1][fk + 1] + r[1][1][1] * a_h[0][2][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[0][2][1][fi + 0][fj + 0][fk + 1] + r[1][2][1] * a_h[0][1][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[0][1][1][fi + 0][fj + 1][fk + 1]) * p[2][0][2] + (r[0][1][2] * a_h[1][2][2][fi + -1][fj + 0][fk + 1] + r[0][2][2] * a_h[1][1][2][fi + -1][fj + 1][fk + 1] + r[1][1][2] * a_h[0][2][2][fi + 0][fj + 0][fk + 1] + r[1][2][2] * a_h[0][1][2][fi + 0][fj + 1][fk + 1]) * p[2][0][1] + (r[0][0][1] * a_h[2][1][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[2][1][1][fi + -1][fj + -1][fk + 1] + r[0][1][1] * a_h[2][0][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[2][0][1][fi + -1][fj + 0][fk + 1] + r[1][0][1] * a_h[1][1][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[1][1][1][fi + 0][fj + -1][fk + 1] + r[1][1][1] * a_h[1][0][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[1][0][1][fi + 0][fj + 0][fk + 1] + r[2][0][1] * a_h[0][1][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[0][1][1][fi + 1][fj + -1][fk + 1] + r[2][1][1] * a_h[0][0][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[0][0][1][fi + 1][fj + 0][fk + 1]) * p[1][2][2] + (r[0][0][2] * a_h[2][1][2][fi + -1][fj + -1][fk + 1] + r[0][1][2] * a_h[2][0][2][fi + -1][fj + 0][fk + 1] + r[1][0][2] * a_h[1][1][2][fi + 0][fj + -1][fk + 1] + r[1][1][2] * a_h[1][0][2][fi + 0][fj + 0][fk + 1] + r[2][0][2] * a_h[0][1][2][fi + 1][fj + -1][fk + 1] + r[2][1][2] * a_h[0][0][2][fi + 1][fj + 0][fk + 1]) * p[1][2][1] + (r[0][0][1] * a_h[2][2][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[2][2][1][fi + -1][fj + -1][fk + 1] + r[0][1][1] * a_h[2][1][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[2][1][1][fi + -1][fj + 0][fk + 1] + r[0][2][1] * a_h[2][0][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[2][0][1][fi + -1][fj + 1][fk + 1] + r[1][0][1] * a_h[1][2][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[1][2][1][fi + 0][fj + -1][fk + 1] + r[1][1][1] * a_h[1][1][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[1][1][1][fi + 0][fj + 0][fk + 1] + r[1][2][1] * a_h[1][0][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[1][0][1][fi + 0][fj + 1][fk + 1] + r[2][0][1] * a_h[0][2][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[0][2][1][fi + 1][fj + -1][fk + 1] + r[2][1][1] * a_h[0][1][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[0][1][1][fi + 1][fj + 0][fk + 1] + r[2][2][1] * a_h[0][0][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[0][0][1][fi + 1][fj + 1][fk + 1]) * p[1][1][2] + (r[0][0][2] * a_h[2][2][2][fi + -1][fj + -1][fk + 1] + r[0][1][2] * a_h[2][1][2][fi + -1][fj + 0][fk + 1] + r[0][2][2] * a_h[2][0][2][fi + -1][fj + 1][fk + 1] + r[1][0][2] * a_h[1][2][2][fi + 0][fj + -1][fk + 1] + r[1][1][2] * a_h[1][1][2][fi + 0][fj + 0][fk + 1] + r[1][2][2] * a_h[1][0][2][fi + 0][fj + 1][fk + 1] + r[2][0][2] * a_h[0][2][2][fi + 1][fj + -1][fk + 1] + r[2][1][2] * a_h[0][1][2][fi + 1][fj + 0][fk + 1] + r[2][2][2] * a_h[0][0][2][fi + 1][fj + 1][fk + 1]) * p[1][1][1] + (r[0][1][1] * a_h[2][2][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[2][2][1][fi + -1][fj + 0][fk + 1] + r[0][2][1] * a_h[2][1][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[2][1][1][fi + -1][fj + 1][fk + 1] + r[1][1][1] * a_h[1][2][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[1][2][1][fi + 0][fj + 0][fk + 1] + r[1][2][1] * a_h[1][1][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[1][1][1][fi + 0][fj + 1][fk + 1] + r[2][1][1] * a_h[0][2][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[0][2][1][fi + 1][fj + 0][fk + 1] + r[2][2][1] * a_h[0][1][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[0][1][1][fi + 1][fj + 1][fk + 1]) * p[1][0][2] + (r[0][1][2] * a_h[2][2][2][fi + -1][fj + 0][fk + 1] + r[0][2][2] * a_h[2][1][2][fi + -1][fj + 1][fk + 1] + r[1][1][2] * a_h[1][2][2][fi + 0][fj + 0][fk + 1] + r[1][2][2] * a_h[1][1][2][fi + 0][fj + 1][fk + 1] + r[2][1][2] * a_h[0][2][2][fi + 1][fj + 0][fk + 1] + r[2][2][2] * a_h[0][1][2][fi + 1][fj + 1][fk + 1]) * p[1][0][1] + (r[1][0][1] * a_h[2][1][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[2][1][1][fi + 0][fj + -1][fk + 1] + r[1][1][1] * a_h[2][0][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[2][0][1][fi + 0][fj + 0][fk + 1] + r[2][0][1] * a_h[1][1][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[1][1][1][fi + 1][fj + -1][fk + 1] + r[2][1][1] * a_h[1][0][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[1][0][1][fi + 1][fj + 0][fk + 1]) * p[0][2][2] + (r[1][0][2] * a_h[2][1][2][fi + 0][fj + -1][fk + 1] + r[1][1][2] * a_h[2][0][2][fi + 0][fj + 0][fk + 1] + r[2][0][2] * a_h[1][1][2][fi + 1][fj + -1][fk + 1] + r[2][1][2] * a_h[1][0][2][fi + 1][fj + 0][fk + 1]) * p[0][2][1] + (r[1][0][1] * a_h[2][2][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[2][2][1][fi + 0][fj + -1][fk + 1] + r[1][1][1] * a_h[2][1][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[2][1][1][fi + 0][fj + 0][fk + 1] + r[1][2][1] * a_h[2][0][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[2][0][1][fi + 0][fj + 1][fk + 1] + r[2][0][1] * a_h[1][2][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[1][2][1][fi + 1][fj + -1][fk + 1] + r[2][1][1] * a_h[1][1][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[1][1][1][fi + 1][fj + 0][fk + 1] + r[2][2][1] * a_h[1][0][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[1][0][1][fi + 1][fj + 1][fk + 1]) * p[0][1][2] + (r[1][0][2] * a_h[2][2][2][fi + 0][fj + -1][fk + 1] + r[1][1][2] * a_h[2][1][2][fi + 0][fj + 0][fk + 1] + r[1][2][2] * a_h[2][0][2][fi + 0][fj + 1][fk + 1] + r[2][0][2] * a_h[1][2][2][fi + 1][fj + -1][fk + 1] + r[2][1][2] * a_h[1][1][2][fi + 1][fj + 0][fk + 1] + r[2][2][2] * a_h[1][0][2][fi + 1][fj + 1][fk + 1]) * p[0][1][1] + (r[1][1][1] * a_h[2][2][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[2][2][1][fi + 0][fj + 0][fk + 1] + r[1][2][1] * a_h[2][1][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[2][1][1][fi + 0][fj + 1][fk + 1] + r[2][1][1] * a_h[1][2][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[1][2][1][fi + 1][fj + 0][fk + 1] + r[2][2][1] * a_h[1][1][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[1][1][1][fi + 1][fj + 1][fk + 1]) * p[0][0][2] + (r[1][1][2] * a_h[2][2][2][fi + 0][fj + 0][fk + 1] + r[1][2][2] * a_h[2][1][2][fi + 0][fj + 1][fk + 1] + r[2][1][2] * a_h[1][2][2][fi + 1][fj + 0][fk + 1] + r[2][2][2] * a_h[1][1][2][fi + 1][fj + 1][fk + 1]) * p[0][0][1]));

                REQUIRE(a_2h[1][2][0][ci][cj][ck] == ((r[0][1][0] * a_h[1][2][0][fi + -1][fj + 0][fk + -1] + r[0][2][0] * a_h[1][1][0][fi + -1][fj + 1][fk + -1] + r[1][1][0] * a_h[0][2][0][fi + 0][fj + 0][fk + -1] + r[1][2][0] * a_h[0][1][0][fi + 0][fj + 1][fk + -1]) * p[2][2][1] + (r[0][1][0] * a_h[1][2][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[1][2][0][fi + -1][fj + 0][fk + 0] + r[0][2][0] * a_h[1][1][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[1][1][0][fi + -1][fj + 1][fk + 0] + r[1][1][0] * a_h[0][2][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[0][2][0][fi + 0][fj + 0][fk + 0] + r[1][2][0] * a_h[0][1][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[0][1][0][fi + 0][fj + 1][fk + 0]) * p[2][2][0] + (r[0][2][0] * a_h[1][2][0][fi + -1][fj + 1][fk + -1] + r[1][2][0] * a_h[0][2][0][fi + 0][fj + 1][fk + -1]) * p[2][1][1] + (r[0][2][0] * a_h[1][2][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[1][2][0][fi + -1][fj + 1][fk + 0] + r[1][2][0] * a_h[0][2][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[0][2][0][fi + 0][fj + 1][fk + 0]) * p[2][1][0] + (r[0][1][0] * a_h[2][2][0][fi + -1][fj + 0][fk + -1] + r[0][2][0] * a_h[2][1][0][fi + -1][fj + 1][fk + -1] + r[1][1][0] * a_h[1][2][0][fi + 0][fj + 0][fk + -1] + r[1][2][0] * a_h[1][1][0][fi + 0][fj + 1][fk + -1] + r[2][1][0] * a_h[0][2][0][fi + 1][fj + 0][fk + -1] + r[2][2][0] * a_h[0][1][0][fi + 1][fj + 1][fk + -1]) * p[1][2][1] + (r[0][1][0] * a_h[2][2][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[2][2][0][fi + -1][fj + 0][fk + 0] + r[0][2][0] * a_h[2][1][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[2][1][0][fi + -1][fj + 1][fk + 0] + r[1][1][0] * a_h[1][2][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[1][2][0][fi + 0][fj + 0][fk + 0] + r[1][2][0] * a_h[1][1][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[1][1][0][fi + 0][fj + 1][fk + 0] + r[2][1][0] * a_h[0][2][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[0][2][0][fi + 1][fj + 0][fk + 0] + r[2][2][0] * a_h[0][1][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[0][1][0][fi + 1][fj + 1][fk + 0]) * p[1][2][0] + (r[0][2][0] * a_h[2][2][0][fi + -1][fj + 1][fk + -1] + r[1][2][0] * a_h[1][2][0][fi + 0][fj + 1][fk + -1] + r[2][2][0] * a_h[0][2][0][fi + 1][fj + 1][fk + -1]) * p[1][1][1] + (r[0][2][0] * a_h[2][2][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[2][2][0][fi + -1][fj + 1][fk + 0] + r[1][2][0] * a_h[1][2][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[1][2][0][fi + 0][fj + 1][fk + 0] + r[2][2][0] * a_h[0][2][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[0][2][0][fi + 1][fj + 1][fk + 0]) * p[1][1][0] + (r[1][1][0] * a_h[2][2][0][fi + 0][fj + 0][fk + -1] + r[1][2][0] * a_h[2][1][0][fi + 0][fj + 1][fk + -1] + r[2][1][0] * a_h[1][2][0][fi + 1][fj + 0][fk + -1] + r[2][2][0] * a_h[1][1][0][fi + 1][fj + 1][fk + -1]) * p[0][2][1] + (r[1][1][0] * a_h[2][2][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[2][2][0][fi + 0][fj + 0][fk + 0] + r[1][2][0] * a_h[2][1][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[2][1][0][fi + 0][fj + 1][fk + 0] + r[2][1][0] * a_h[1][2][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[1][2][0][fi + 1][fj + 0][fk + 0] + r[2][2][0] * a_h[1][1][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[1][1][0][fi + 1][fj + 1][fk + 0]) * p[0][2][0] + (r[1][2][0] * a_h[2][2][0][fi + 0][fj + 1][fk + -1] + r[2][2][0] * a_h[1][2][0][fi + 1][fj + 1][fk + -1]) * p[0][1][1] + (r[1][2][0] * a_h[2][2][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[2][2][0][fi + 0][fj + 1][fk + 0] + r[2][2][0] * a_h[1][2][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[1][2][0][fi + 1][fj + 1][fk + 0]) * p[0][1][0]));

                REQUIRE(a_2h[1][2][1][ci][cj][ck] == ((r[0][1][0] * a_h[1][2][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[1][2][0][fi + -1][fj + 0][fk + 0] + r[0][2][0] * a_h[1][1][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[1][1][0][fi + -1][fj + 1][fk + 0] + r[1][1][0] * a_h[0][2][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[0][2][0][fi + 0][fj + 0][fk + 0] + r[1][2][0] * a_h[0][1][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[0][1][0][fi + 0][fj + 1][fk + 0]) * p[2][2][2] + (r[0][1][0] * a_h[1][2][2][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[1][2][1][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[1][2][0][fi + -1][fj + 0][fk + 1] + r[0][2][0] * a_h[1][1][2][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[1][1][1][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[1][1][0][fi + -1][fj + 1][fk + 1] + r[1][1][0] * a_h[0][2][2][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[0][2][1][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[0][2][0][fi + 0][fj + 0][fk + 1] + r[1][2][0] * a_h[0][1][2][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[0][1][1][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[0][1][0][fi + 0][fj + 1][fk + 1]) * p[2][2][1] + (r[0][1][1] * a_h[1][2][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[1][2][1][fi + -1][fj + 0][fk + 1] + r[0][2][1] * a_h[1][1][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[1][1][1][fi + -1][fj + 1][fk + 1] + r[1][1][1] * a_h[0][2][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[0][2][1][fi + 0][fj + 0][fk + 1] + r[1][2][1] * a_h[0][1][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[0][1][1][fi + 0][fj + 1][fk + 1]) * p[2][2][0] + (r[0][2][0] * a_h[1][2][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[1][2][0][fi + -1][fj + 1][fk + 0] + r[1][2][0] * a_h[0][2][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[0][2][0][fi + 0][fj + 1][fk + 0]) * p[2][1][2] + (r[0][2][0] * a_h[1][2][2][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[1][2][1][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[1][2][0][fi + -1][fj + 1][fk + 1] + r[1][2][0] * a_h[0][2][2][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[0][2][1][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[0][2][0][fi + 0][fj + 1][fk + 1]) * p[2][1][1] + (r[0][2][1] * a_h[1][2][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[1][2][1][fi + -1][fj + 1][fk + 1] + r[1][2][1] * a_h[0][2][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[0][2][1][fi + 0][fj + 1][fk + 1]) * p[2][1][0] + (r[0][1][0] * a_h[2][2][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[2][2][0][fi + -1][fj + 0][fk + 0] + r[0][2][0] * a_h[2][1][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[2][1][0][fi + -1][fj + 1][fk + 0] + r[1][1][0] * a_h[1][2][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[1][2][0][fi + 0][fj + 0][fk + 0] + r[1][2][0] * a_h[1][1][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[1][1][0][fi + 0][fj + 1][fk + 0] + r[2][1][0] * a_h[0][2][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[0][2][0][fi + 1][fj + 0][fk + 0] + r[2][2][0] * a_h[0][1][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[0][1][0][fi + 1][fj + 1][fk + 0]) * p[1][2][2] + (r[0][1][0] * a_h[2][2][2][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[2][2][1][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[2][2][0][fi + -1][fj + 0][fk + 1] + r[0][2][0] * a_h[2][1][2][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[2][1][1][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[2][1][0][fi + -1][fj + 1][fk + 1] + r[1][1][0] * a_h[1][2][2][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[1][2][1][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[1][2][0][fi + 0][fj + 0][fk + 1] + r[1][2][0] * a_h[1][1][2][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[1][1][1][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[1][1][0][fi + 0][fj + 1][fk + 1] + r[2][1][0] * a_h[0][2][2][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[0][2][1][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[0][2][0][fi + 1][fj + 0][fk + 1] + r[2][2][0] * a_h[0][1][2][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[0][1][1][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[0][1][0][fi + 1][fj + 1][fk + 1]) * p[1][2][1] + (r[0][1][1] * a_h[2][2][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[2][2][1][fi + -1][fj + 0][fk + 1] + r[0][2][1] * a_h[2][1][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[2][1][1][fi + -1][fj + 1][fk + 1] + r[1][1][1] * a_h[1][2][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[1][2][1][fi + 0][fj + 0][fk + 1] + r[1][2][1] * a_h[1][1][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[1][1][1][fi + 0][fj + 1][fk + 1] + r[2][1][1] * a_h[0][2][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[0][2][1][fi + 1][fj + 0][fk + 1] + r[2][2][1] * a_h[0][1][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[0][1][1][fi + 1][fj + 1][fk + 1]) * p[1][2][0] + (r[0][2][0] * a_h[2][2][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[2][2][0][fi + -1][fj + 1][fk + 0] + r[1][2][0] * a_h[1][2][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[1][2][0][fi + 0][fj + 1][fk + 0] + r[2][2][0] * a_h[0][2][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[0][2][0][fi + 1][fj + 1][fk + 0]) * p[1][1][2] + (r[0][2][0] * a_h[2][2][2][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[2][2][1][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[2][2][0][fi + -1][fj + 1][fk + 1] + r[1][2][0] * a_h[1][2][2][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[1][2][1][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[1][2][0][fi + 0][fj + 1][fk + 1] + r[2][2][0] * a_h[0][2][2][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[0][2][1][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[0][2][0][fi + 1][fj + 1][fk + 1]) * p[1][1][1] + (r[0][2][1] * a_h[2][2][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[2][2][1][fi + -1][fj + 1][fk + 1] + r[1][2][1] * a_h[1][2][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[1][2][1][fi + 0][fj + 1][fk + 1] + r[2][2][1] * a_h[0][2][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[0][2][1][fi + 1][fj + 1][fk + 1]) * p[1][1][0] + (r[1][1][0] * a_h[2][2][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[2][2][0][fi + 0][fj + 0][fk + 0] + r[1][2][0] * a_h[2][1][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[2][1][0][fi + 0][fj + 1][fk + 0] + r[2][1][0] * a_h[1][2][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[1][2][0][fi + 1][fj + 0][fk + 0] + r[2][2][0] * a_h[1][1][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[1][1][0][fi + 1][fj + 1][fk + 0]) * p[0][2][2] + (r[1][1][0] * a_h[2][2][2][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[2][2][1][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[2][2][0][fi + 0][fj + 0][fk + 1] + r[1][2][0] * a_h[2][1][2][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[2][1][1][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[2][1][0][fi + 0][fj + 1][fk + 1] + r[2][1][0] * a_h[1][2][2][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[1][2][1][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[1][2][0][fi + 1][fj + 0][fk + 1] + r[2][2][0] * a_h[1][1][2][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[1][1][1][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[1][1][0][fi + 1][fj + 1][fk + 1]) * p[0][2][1] + (r[1][1][1] * a_h[2][2][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[2][2][1][fi + 0][fj + 0][fk + 1] + r[1][2][1] * a_h[2][1][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[2][1][1][fi + 0][fj + 1][fk + 1] + r[2][1][1] * a_h[1][2][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[1][2][1][fi + 1][fj + 0][fk + 1] + r[2][2][1] * a_h[1][1][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[1][1][1][fi + 1][fj + 1][fk + 1]) * p[0][2][0] + (r[1][2][0] * a_h[2][2][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[2][2][0][fi + 0][fj + 1][fk + 0] + r[2][2][0] * a_h[1][2][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[1][2][0][fi + 1][fj + 1][fk + 0]) * p[0][1][2] + (r[1][2][0] * a_h[2][2][2][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[2][2][1][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[2][2][0][fi + 0][fj + 1][fk + 1] + r[2][2][0] * a_h[1][2][2][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[1][2][1][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[1][2][0][fi + 1][fj + 1][fk + 1]) * p[0][1][1] + (r[1][2][1] * a_h[2][2][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[2][2][1][fi + 0][fj + 1][fk + 1] + r[2][2][1] * a_h[1][2][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[1][2][1][fi + 1][fj + 1][fk + 1]) * p[0][1][0]));

                REQUIRE(a_2h[1][2][2][ci][cj][ck] == ((r[0][1][1] * a_h[1][2][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[1][2][1][fi + -1][fj + 0][fk + 1] + r[0][2][1] * a_h[1][1][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[1][1][1][fi + -1][fj + 1][fk + 1] + r[1][1][1] * a_h[0][2][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[0][2][1][fi + 0][fj + 0][fk + 1] + r[1][2][1] * a_h[0][1][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[0][1][1][fi + 0][fj + 1][fk + 1]) * p[2][2][2] + (r[0][1][2] * a_h[1][2][2][fi + -1][fj + 0][fk + 1] + r[0][2][2] * a_h[1][1][2][fi + -1][fj + 1][fk + 1] + r[1][1][2] * a_h[0][2][2][fi + 0][fj + 0][fk + 1] + r[1][2][2] * a_h[0][1][2][fi + 0][fj + 1][fk + 1]) * p[2][2][1] + (r[0][2][1] * a_h[1][2][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[1][2][1][fi + -1][fj + 1][fk + 1] + r[1][2][1] * a_h[0][2][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[0][2][1][fi + 0][fj + 1][fk + 1]) * p[2][1][2] + (r[0][2][2] * a_h[1][2][2][fi + -1][fj + 1][fk + 1] + r[1][2][2] * a_h[0][2][2][fi + 0][fj + 1][fk + 1]) * p[2][1][1] + (r[0][1][1] * a_h[2][2][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[2][2][1][fi + -1][fj + 0][fk + 1] + r[0][2][1] * a_h[2][1][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[2][1][1][fi + -1][fj + 1][fk + 1] + r[1][1][1] * a_h[1][2][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[1][2][1][fi + 0][fj + 0][fk + 1] + r[1][2][1] * a_h[1][1][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[1][1][1][fi + 0][fj + 1][fk + 1] + r[2][1][1] * a_h[0][2][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[0][2][1][fi + 1][fj + 0][fk + 1] + r[2][2][1] * a_h[0][1][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[0][1][1][fi + 1][fj + 1][fk + 1]) * p[1][2][2] + (r[0][1][2] * a_h[2][2][2][fi + -1][fj + 0][fk + 1] + r[0][2][2] * a_h[2][1][2][fi + -1][fj + 1][fk + 1] + r[1][1][2] * a_h[1][2][2][fi + 0][fj + 0][fk + 1] + r[1][2][2] * a_h[1][1][2][fi + 0][fj + 1][fk + 1] + r[2][1][2] * a_h[0][2][2][fi + 1][fj + 0][fk + 1] + r[2][2][2] * a_h[0][1][2][fi + 1][fj + 1][fk + 1]) * p[1][2][1] + (r[0][2][1] * a_h[2][2][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[2][2][1][fi + -1][fj + 1][fk + 1] + r[1][2][1] * a_h[1][2][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[1][2][1][fi + 0][fj + 1][fk + 1] + r[2][2][1] * a_h[0][2][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[0][2][1][fi + 1][fj + 1][fk + 1]) * p[1][1][2] + (r[0][2][2] * a_h[2][2][2][fi + -1][fj + 1][fk + 1] + r[1][2][2] * a_h[1][2][2][fi + 0][fj + 1][fk + 1] + r[2][2][2] * a_h[0][2][2][fi + 1][fj + 1][fk + 1]) * p[1][1][1] + (r[1][1][1] * a_h[2][2][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[2][2][1][fi + 0][fj + 0][fk + 1] + r[1][2][1] * a_h[2][1][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[2][1][1][fi + 0][fj + 1][fk + 1] + r[2][1][1] * a_h[1][2][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[1][2][1][fi + 1][fj + 0][fk + 1] + r[2][2][1] * a_h[1][1][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[1][1][1][fi + 1][fj + 1][fk + 1]) * p[0][2][2] + (r[1][1][2] * a_h[2][2][2][fi + 0][fj + 0][fk + 1] + r[1][2][2] * a_h[2][1][2][fi + 0][fj + 1][fk + 1] + r[2][1][2] * a_h[1][2][2][fi + 1][fj + 0][fk + 1] + r[2][2][2] * a_h[1][1][2][fi + 1][fj + 1][fk + 1]) * p[0][2][1] + (r[1][2][1] * a_h[2][2][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[2][2][1][fi + 0][fj + 1][fk + 1] + r[2][2][1] * a_h[1][2][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[1][2][1][fi + 1][fj + 1][fk + 1]) * p[0][1][2] + (r[1][2][2] * a_h[2][2][2][fi + 0][fj + 1][fk + 1] + r[2][2][2] * a_h[1][2][2][fi + 1][fj + 1][fk + 1]) * p[0][1][1]));

                REQUIRE(a_2h[2][0][0][ci][cj][ck] == ((r[1][0][0] * a_h[2][0][0][fi + 0][fj + -1][fk + -1] + r[2][0][0] * a_h[1][0][0][fi + 1][fj + -1][fk + -1]) * p[2][1][1] + (r[1][0][0] * a_h[2][0][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[2][0][0][fi + 0][fj + -1][fk + 0] + r[2][0][0] * a_h[1][0][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[1][0][0][fi + 1][fj + -1][fk + 0]) * p[2][1][0] + (r[1][0][0] * a_h[2][1][0][fi + 0][fj + -1][fk + -1] + r[1][1][0] * a_h[2][0][0][fi + 0][fj + 0][fk + -1] + r[2][0][0] * a_h[1][1][0][fi + 1][fj + -1][fk + -1] + r[2][1][0] * a_h[1][0][0][fi + 1][fj + 0][fk + -1]) * p[2][0][1] + (r[1][0][0] * a_h[2][1][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[2][1][0][fi + 0][fj + -1][fk + 0] + r[1][1][0] * a_h[2][0][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[2][0][0][fi + 0][fj + 0][fk + 0] + r[2][0][0] * a_h[1][1][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[1][1][0][fi + 1][fj + -1][fk + 0] + r[2][1][0] * a_h[1][0][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[1][0][0][fi + 1][fj + 0][fk + 0]) * p[2][0][0] + (r[2][0][0] * a_h[2][0][0][fi + 1][fj + -1][fk + -1]) * p[1][1][1] + (r[2][0][0] * a_h[2][0][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[2][0][0][fi + 1][fj + -1][fk + 0]) * p[1][1][0] + (r[2][0][0] * a_h[2][1][0][fi + 1][fj + -1][fk + -1] + r[2][1][0] * a_h[2][0][0][fi + 1][fj + 0][fk + -1]) * p[1][0][1] + (r[2][0][0] * a_h[2][1][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[2][1][0][fi + 1][fj + -1][fk + 0] + r[2][1][0] * a_h[2][0][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[2][0][0][fi + 1][fj + 0][fk + 0]) * p[1][0][0]));

                REQUIRE(a_2h[2][0][1][ci][cj][ck] == ((r[1][0][0] * a_h[2][0][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[2][0][0][fi + 0][fj + -1][fk + 0] + r[2][0][0] * a_h[1][0][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[1][0][0][fi + 1][fj + -1][fk + 0]) * p[2][1][2] + (r[1][0][0] * a_h[2][0][2][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[2][0][1][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[2][0][0][fi + 0][fj + -1][fk + 1] + r[2][0][0] * a_h[1][0][2][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[1][0][1][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[1][0][0][fi + 1][fj + -1][fk + 1]) * p[2][1][1] + (r[1][0][1] * a_h[2][0][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[2][0][1][fi + 0][fj + -1][fk + 1] + r[2][0][1] * a_h[1][0][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[1][0][1][fi + 1][fj + -1][fk + 1]) * p[2][1][0] + (r[1][0][0] * a_h[2][1][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[2][1][0][fi + 0][fj + -1][fk + 0] + r[1][1][0] * a_h[2][0][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[2][0][0][fi + 0][fj + 0][fk + 0] + r[2][0][0] * a_h[1][1][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[1][1][0][fi + 1][fj + -1][fk + 0] + r[2][1][0] * a_h[1][0][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[1][0][0][fi + 1][fj + 0][fk + 0]) * p[2][0][2] + (r[1][0][0] * a_h[2][1][2][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[2][1][1][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[2][1][0][fi + 0][fj + -1][fk + 1] + r[1][1][0] * a_h[2][0][2][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[2][0][1][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[2][0][0][fi + 0][fj + 0][fk + 1] + r[2][0][0] * a_h[1][1][2][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[1][1][1][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[1][1][0][fi + 1][fj + -1][fk + 1] + r[2][1][0] * a_h[1][0][2][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[1][0][1][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[1][0][0][fi + 1][fj + 0][fk + 1]) * p[2][0][1] + (r[1][0][1] * a_h[2][1][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[2][1][1][fi + 0][fj + -1][fk + 1] + r[1][1][1] * a_h[2][0][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[2][0][1][fi + 0][fj + 0][fk + 1] + r[2][0][1] * a_h[1][1][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[1][1][1][fi + 1][fj + -1][fk + 1] + r[2][1][1] * a_h[1][0][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[1][0][1][fi + 1][fj + 0][fk + 1]) * p[2][0][0] + (r[2][0][0] * a_h[2][0][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[2][0][0][fi + 1][fj + -1][fk + 0]) * p[1][1][2] + (r[2][0][0] * a_h[2][0][2][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[2][0][1][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[2][0][0][fi + 1][fj + -1][fk + 1]) * p[1][1][1] + (r[2][0][1] * a_h[2][0][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[2][0][1][fi + 1][fj + -1][fk + 1]) * p[1][1][0] + (r[2][0][0] * a_h[2][1][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[2][1][0][fi + 1][fj + -1][fk + 0] + r[2][1][0] * a_h[2][0][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[2][0][0][fi + 1][fj + 0][fk + 0]) * p[1][0][2] + (r[2][0][0] * a_h[2][1][2][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[2][1][1][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[2][1][0][fi + 1][fj + -1][fk + 1] + r[2][1][0] * a_h[2][0][2][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[2][0][1][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[2][0][0][fi + 1][fj + 0][fk + 1]) * p[1][0][1] + (r[2][0][1] * a_h[2][1][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[2][1][1][fi + 1][fj + -1][fk + 1] + r[2][1][1] * a_h[2][0][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[2][0][1][fi + 1][fj + 0][fk + 1]) * p[1][0][0]));

                REQUIRE(a_2h[2][0][2][ci][cj][ck] == ((r[1][0][1] * a_h[2][0][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[2][0][1][fi + 0][fj + -1][fk + 1] + r[2][0][1] * a_h[1][0][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[1][0][1][fi + 1][fj + -1][fk + 1]) * p[2][1][2] + (r[1][0][2] * a_h[2][0][2][fi + 0][fj + -1][fk + 1] + r[2][0][2] * a_h[1][0][2][fi + 1][fj + -1][fk + 1]) * p[2][1][1] + (r[1][0][1] * a_h[2][1][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[2][1][1][fi + 0][fj + -1][fk + 1] + r[1][1][1] * a_h[2][0][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[2][0][1][fi + 0][fj + 0][fk + 1] + r[2][0][1] * a_h[1][1][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[1][1][1][fi + 1][fj + -1][fk + 1] + r[2][1][1] * a_h[1][0][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[1][0][1][fi + 1][fj + 0][fk + 1]) * p[2][0][2] + (r[1][0][2] * a_h[2][1][2][fi + 0][fj + -1][fk + 1] + r[1][1][2] * a_h[2][0][2][fi + 0][fj + 0][fk + 1] + r[2][0][2] * a_h[1][1][2][fi + 1][fj + -1][fk + 1] + r[2][1][2] * a_h[1][0][2][fi + 1][fj + 0][fk + 1]) * p[2][0][1] + (r[2][0][1] * a_h[2][0][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[2][0][1][fi + 1][fj + -1][fk + 1]) * p[1][1][2] + (r[2][0][2] * a_h[2][0][2][fi + 1][fj + -1][fk + 1]) * p[1][1][1] + (r[2][0][1] * a_h[2][1][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[2][1][1][fi + 1][fj + -1][fk + 1] + r[2][1][1] * a_h[2][0][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[2][0][1][fi + 1][fj + 0][fk + 1]) * p[1][0][2] + (r[2][0][2] * a_h[2][1][2][fi + 1][fj + -1][fk + 1] + r[2][1][2] * a_h[2][0][2][fi + 1][fj + 0][fk + 1]) * p[1][0][1]));

                REQUIRE(a_2h[2][1][0][ci][cj][ck] == ((r[1][0][0] * a_h[2][1][0][fi + 0][fj + -1][fk + -1] + r[1][1][0] * a_h[2][0][0][fi + 0][fj + 0][fk + -1] + r[2][0][0] * a_h[1][1][0][fi + 1][fj + -1][fk + -1] + r[2][1][0] * a_h[1][0][0][fi + 1][fj + 0][fk + -1]) * p[2][2][1] + (r[1][0][0] * a_h[2][1][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[2][1][0][fi + 0][fj + -1][fk + 0] + r[1][1][0] * a_h[2][0][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[2][0][0][fi + 0][fj + 0][fk + 0] + r[2][0][0] * a_h[1][1][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[1][1][0][fi + 1][fj + -1][fk + 0] + r[2][1][0] * a_h[1][0][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[1][0][0][fi + 1][fj + 0][fk + 0]) * p[2][2][0] + (r[1][0][0] * a_h[2][2][0][fi + 0][fj + -1][fk + -1] + r[1][1][0] * a_h[2][1][0][fi + 0][fj + 0][fk + -1] + r[1][2][0] * a_h[2][0][0][fi + 0][fj + 1][fk + -1] + r[2][0][0] * a_h[1][2][0][fi + 1][fj + -1][fk + -1] + r[2][1][0] * a_h[1][1][0][fi + 1][fj + 0][fk + -1] + r[2][2][0] * a_h[1][0][0][fi + 1][fj + 1][fk + -1]) * p[2][1][1] + (r[1][0][0] * a_h[2][2][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[2][2][0][fi + 0][fj + -1][fk + 0] + r[1][1][0] * a_h[2][1][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[2][1][0][fi + 0][fj + 0][fk + 0] + r[1][2][0] * a_h[2][0][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[2][0][0][fi + 0][fj + 1][fk + 0] + r[2][0][0] * a_h[1][2][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[1][2][0][fi + 1][fj + -1][fk + 0] + r[2][1][0] * a_h[1][1][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[1][1][0][fi + 1][fj + 0][fk + 0] + r[2][2][0] * a_h[1][0][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[1][0][0][fi + 1][fj + 1][fk + 0]) * p[2][1][0] + (r[1][1][0] * a_h[2][2][0][fi + 0][fj + 0][fk + -1] + r[1][2][0] * a_h[2][1][0][fi + 0][fj + 1][fk + -1] + r[2][1][0] * a_h[1][2][0][fi + 1][fj + 0][fk + -1] + r[2][2][0] * a_h[1][1][0][fi + 1][fj + 1][fk + -1]) * p[2][0][1] + (r[1][1][0] * a_h[2][2][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[2][2][0][fi + 0][fj + 0][fk + 0] + r[1][2][0] * a_h[2][1][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[2][1][0][fi + 0][fj + 1][fk + 0] + r[2][1][0] * a_h[1][2][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[1][2][0][fi + 1][fj + 0][fk + 0] + r[2][2][0] * a_h[1][1][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[1][1][0][fi + 1][fj + 1][fk + 0]) * p[2][0][0] + (r[2][0][0] * a_h[2][1][0][fi + 1][fj + -1][fk + -1] + r[2][1][0] * a_h[2][0][0][fi + 1][fj + 0][fk + -1]) * p[1][2][1] + (r[2][0][0] * a_h[2][1][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[2][1][0][fi + 1][fj + -1][fk + 0] + r[2][1][0] * a_h[2][0][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[2][0][0][fi + 1][fj + 0][fk + 0]) * p[1][2][0] + (r[2][0][0] * a_h[2][2][0][fi + 1][fj + -1][fk + -1] + r[2][1][0] * a_h[2][1][0][fi + 1][fj + 0][fk + -1] + r[2][2][0] * a_h[2][0][0][fi + 1][fj + 1][fk + -1]) * p[1][1][1] + (r[2][0][0] * a_h[2][2][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[2][2][0][fi + 1][fj + -1][fk + 0] + r[2][1][0] * a_h[2][1][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[2][1][0][fi + 1][fj + 0][fk + 0] + r[2][2][0] * a_h[2][0][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[2][0][0][fi + 1][fj + 1][fk + 0]) * p[1][1][0] + (r[2][1][0] * a_h[2][2][0][fi + 1][fj + 0][fk + -1] + r[2][2][0] * a_h[2][1][0][fi + 1][fj + 1][fk + -1]) * p[1][0][1] + (r[2][1][0] * a_h[2][2][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[2][2][0][fi + 1][fj + 0][fk + 0] + r[2][2][0] * a_h[2][1][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[2][1][0][fi + 1][fj + 1][fk + 0]) * p[1][0][0]));

                REQUIRE(a_2h[2][1][1][ci][cj][ck] == ((r[1][0][0] * a_h[2][1][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[2][1][0][fi + 0][fj + -1][fk + 0] + r[1][1][0] * a_h[2][0][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[2][0][0][fi + 0][fj + 0][fk + 0] + r[2][0][0] * a_h[1][1][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[1][1][0][fi + 1][fj + -1][fk + 0] + r[2][1][0] * a_h[1][0][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[1][0][0][fi + 1][fj + 0][fk + 0]) * p[2][2][2] + (r[1][0][0] * a_h[2][1][2][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[2][1][1][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[2][1][0][fi + 0][fj + -1][fk + 1] + r[1][1][0] * a_h[2][0][2][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[2][0][1][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[2][0][0][fi + 0][fj + 0][fk + 1] + r[2][0][0] * a_h[1][1][2][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[1][1][1][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[1][1][0][fi + 1][fj + -1][fk + 1] + r[2][1][0] * a_h[1][0][2][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[1][0][1][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[1][0][0][fi + 1][fj + 0][fk + 1]) * p[2][2][1] + (r[1][0][1] * a_h[2][1][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[2][1][1][fi + 0][fj + -1][fk + 1] + r[1][1][1] * a_h[2][0][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[2][0][1][fi + 0][fj + 0][fk + 1] + r[2][0][1] * a_h[1][1][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[1][1][1][fi + 1][fj + -1][fk + 1] + r[2][1][1] * a_h[1][0][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[1][0][1][fi + 1][fj + 0][fk + 1]) * p[2][2][0] + (r[1][0][0] * a_h[2][2][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[2][2][0][fi + 0][fj + -1][fk + 0] + r[1][1][0] * a_h[2][1][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[2][1][0][fi + 0][fj + 0][fk + 0] + r[1][2][0] * a_h[2][0][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[2][0][0][fi + 0][fj + 1][fk + 0] + r[2][0][0] * a_h[1][2][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[1][2][0][fi + 1][fj + -1][fk + 0] + r[2][1][0] * a_h[1][1][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[1][1][0][fi + 1][fj + 0][fk + 0] + r[2][2][0] * a_h[1][0][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[1][0][0][fi + 1][fj + 1][fk + 0]) * p[2][1][2] + (r[1][0][0] * a_h[2][2][2][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[2][2][1][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[2][2][0][fi + 0][fj + -1][fk + 1] + r[1][1][0] * a_h[2][1][2][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[2][1][1][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[2][1][0][fi + 0][fj + 0][fk + 1] + r[1][2][0] * a_h[2][0][2][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[2][0][1][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[2][0][0][fi + 0][fj + 1][fk + 1] + r[2][0][0] * a_h[1][2][2][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[1][2][1][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[1][2][0][fi + 1][fj + -1][fk + 1] + r[2][1][0] * a_h[1][1][2][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[1][1][1][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[1][1][0][fi + 1][fj + 0][fk + 1] + r[2][2][0] * a_h[1][0][2][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[1][0][1][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[1][0][0][fi + 1][fj + 1][fk + 1]) * p[2][1][1] + (r[1][0][1] * a_h[2][2][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[2][2][1][fi + 0][fj + -1][fk + 1] + r[1][1][1] * a_h[2][1][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[2][1][1][fi + 0][fj + 0][fk + 1] + r[1][2][1] * a_h[2][0][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[2][0][1][fi + 0][fj + 1][fk + 1] + r[2][0][1] * a_h[1][2][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[1][2][1][fi + 1][fj + -1][fk + 1] + r[2][1][1] * a_h[1][1][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[1][1][1][fi + 1][fj + 0][fk + 1] + r[2][2][1] * a_h[1][0][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[1][0][1][fi + 1][fj + 1][fk + 1]) * p[2][1][0] + (r[1][1][0] * a_h[2][2][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[2][2][0][fi + 0][fj + 0][fk + 0] + r[1][2][0] * a_h[2][1][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[2][1][0][fi + 0][fj + 1][fk + 0] + r[2][1][0] * a_h[1][2][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[1][2][0][fi + 1][fj + 0][fk + 0] + r[2][2][0] * a_h[1][1][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[1][1][0][fi + 1][fj + 1][fk + 0]) * p[2][0][2] + (r[1][1][0] * a_h[2][2][2][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[2][2][1][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[2][2][0][fi + 0][fj + 0][fk + 1] + r[1][2][0] * a_h[2][1][2][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[2][1][1][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[2][1][0][fi + 0][fj + 1][fk + 1] + r[2][1][0] * a_h[1][2][2][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[1][2][1][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[1][2][0][fi + 1][fj + 0][fk + 1] + r[2][2][0] * a_h[1][1][2][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[1][1][1][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[1][1][0][fi + 1][fj + 1][fk + 1]) * p[2][0][1] + (r[1][1][1] * a_h[2][2][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[2][2][1][fi + 0][fj + 0][fk + 1] + r[1][2][1] * a_h[2][1][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[2][1][1][fi + 0][fj + 1][fk + 1] + r[2][1][1] * a_h[1][2][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[1][2][1][fi + 1][fj + 0][fk + 1] + r[2][2][1] * a_h[1][1][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[1][1][1][fi + 1][fj + 1][fk + 1]) * p[2][0][0] + (r[2][0][0] * a_h[2][1][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[2][1][0][fi + 1][fj + -1][fk + 0] + r[2][1][0] * a_h[2][0][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[2][0][0][fi + 1][fj + 0][fk + 0]) * p[1][2][2] + (r[2][0][0] * a_h[2][1][2][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[2][1][1][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[2][1][0][fi + 1][fj + -1][fk + 1] + r[2][1][0] * a_h[2][0][2][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[2][0][1][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[2][0][0][fi + 1][fj + 0][fk + 1]) * p[1][2][1] + (r[2][0][1] * a_h[2][1][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[2][1][1][fi + 1][fj + -1][fk + 1] + r[2][1][1] * a_h[2][0][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[2][0][1][fi + 1][fj + 0][fk + 1]) * p[1][2][0] + (r[2][0][0] * a_h[2][2][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[2][2][0][fi + 1][fj + -1][fk + 0] + r[2][1][0] * a_h[2][1][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[2][1][0][fi + 1][fj + 0][fk + 0] + r[2][2][0] * a_h[2][0][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[2][0][0][fi + 1][fj + 1][fk + 0]) * p[1][1][2] + (r[2][0][0] * a_h[2][2][2][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[2][2][1][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[2][2][0][fi + 1][fj + -1][fk + 1] + r[2][1][0] * a_h[2][1][2][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[2][1][1][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[2][1][0][fi + 1][fj + 0][fk + 1] + r[2][2][0] * a_h[2][0][2][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[2][0][1][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[2][0][0][fi + 1][fj + 1][fk + 1]) * p[1][1][1] + (r[2][0][1] * a_h[2][2][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[2][2][1][fi + 1][fj + -1][fk + 1] + r[2][1][1] * a_h[2][1][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[2][1][1][fi + 1][fj + 0][fk + 1] + r[2][2][1] * a_h[2][0][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[2][0][1][fi + 1][fj + 1][fk + 1]) * p[1][1][0] + (r[2][1][0] * a_h[2][2][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[2][2][0][fi + 1][fj + 0][fk + 0] + r[2][2][0] * a_h[2][1][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[2][1][0][fi + 1][fj + 1][fk + 0]) * p[1][0][2] + (r[2][1][0] * a_h[2][2][2][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[2][2][1][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[2][2][0][fi + 1][fj + 0][fk + 1] + r[2][2][0] * a_h[2][1][2][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[2][1][1][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[2][1][0][fi + 1][fj + 1][fk + 1]) * p[1][0][1] + (r[2][1][1] * a_h[2][2][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[2][2][1][fi + 1][fj + 0][fk + 1] + r[2][2][1] * a_h[2][1][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[2][1][1][fi + 1][fj + 1][fk + 1]) * p[1][0][0]));

                REQUIRE(a_2h[2][1][2][ci][cj][ck] == ((r[1][0][1] * a_h[2][1][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[2][1][1][fi + 0][fj + -1][fk + 1] + r[1][1][1] * a_h[2][0][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[2][0][1][fi + 0][fj + 0][fk + 1] + r[2][0][1] * a_h[1][1][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[1][1][1][fi + 1][fj + -1][fk + 1] + r[2][1][1] * a_h[1][0][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[1][0][1][fi + 1][fj + 0][fk + 1]) * p[2][2][2] + (r[1][0][2] * a_h[2][1][2][fi + 0][fj + -1][fk + 1] + r[1][1][2] * a_h[2][0][2][fi + 0][fj + 0][fk + 1] + r[2][0][2] * a_h[1][1][2][fi + 1][fj + -1][fk + 1] + r[2][1][2] * a_h[1][0][2][fi + 1][fj + 0][fk + 1]) * p[2][2][1] + (r[1][0][1] * a_h[2][2][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[2][2][1][fi + 0][fj + -1][fk + 1] + r[1][1][1] * a_h[2][1][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[2][1][1][fi + 0][fj + 0][fk + 1] + r[1][2][1] * a_h[2][0][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[2][0][1][fi + 0][fj + 1][fk + 1] + r[2][0][1] * a_h[1][2][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[1][2][1][fi + 1][fj + -1][fk + 1] + r[2][1][1] * a_h[1][1][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[1][1][1][fi + 1][fj + 0][fk + 1] + r[2][2][1] * a_h[1][0][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[1][0][1][fi + 1][fj + 1][fk + 1]) * p[2][1][2] + (r[1][0][2] * a_h[2][2][2][fi + 0][fj + -1][fk + 1] + r[1][1][2] * a_h[2][1][2][fi + 0][fj + 0][fk + 1] + r[1][2][2] * a_h[2][0][2][fi + 0][fj + 1][fk + 1] + r[2][0][2] * a_h[1][2][2][fi + 1][fj + -1][fk + 1] + r[2][1][2] * a_h[1][1][2][fi + 1][fj + 0][fk + 1] + r[2][2][2] * a_h[1][0][2][fi + 1][fj + 1][fk + 1]) * p[2][1][1] + (r[1][1][1] * a_h[2][2][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[2][2][1][fi + 0][fj + 0][fk + 1] + r[1][2][1] * a_h[2][1][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[2][1][1][fi + 0][fj + 1][fk + 1] + r[2][1][1] * a_h[1][2][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[1][2][1][fi + 1][fj + 0][fk + 1] + r[2][2][1] * a_h[1][1][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[1][1][1][fi + 1][fj + 1][fk + 1]) * p[2][0][2] + (r[1][1][2] * a_h[2][2][2][fi + 0][fj + 0][fk + 1] + r[1][2][2] * a_h[2][1][2][fi + 0][fj + 1][fk + 1] + r[2][1][2] * a_h[1][2][2][fi + 1][fj + 0][fk + 1] + r[2][2][2] * a_h[1][1][2][fi + 1][fj + 1][fk + 1]) * p[2][0][1] + (r[2][0][1] * a_h[2][1][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[2][1][1][fi + 1][fj + -1][fk + 1] + r[2][1][1] * a_h[2][0][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[2][0][1][fi + 1][fj + 0][fk + 1]) * p[1][2][2] + (r[2][0][2] * a_h[2][1][2][fi + 1][fj + -1][fk + 1] + r[2][1][2] * a_h[2][0][2][fi + 1][fj + 0][fk + 1]) * p[1][2][1] + (r[2][0][1] * a_h[2][2][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[2][2][1][fi + 1][fj + -1][fk + 1] + r[2][1][1] * a_h[2][1][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[2][1][1][fi + 1][fj + 0][fk + 1] + r[2][2][1] * a_h[2][0][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[2][0][1][fi + 1][fj + 1][fk + 1]) * p[1][1][2] + (r[2][0][2] * a_h[2][2][2][fi + 1][fj + -1][fk + 1] + r[2][1][2] * a_h[2][1][2][fi + 1][fj + 0][fk + 1] + r[2][2][2] * a_h[2][0][2][fi + 1][fj + 1][fk + 1]) * p[1][1][1] + (r[2][1][1] * a_h[2][2][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[2][2][1][fi + 1][fj + 0][fk + 1] + r[2][2][1] * a_h[2][1][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[2][1][1][fi + 1][fj + 1][fk + 1]) * p[1][0][2] + (r[2][1][2] * a_h[2][2][2][fi + 1][fj + 0][fk + 1] + r[2][2][2] * a_h[2][1][2][fi + 1][fj + 1][fk + 1]) * p[1][0][1]));

                REQUIRE(a_2h[2][2][0][ci][cj][ck] == ((r[1][1][0] * a_h[2][2][0][fi + 0][fj + 0][fk + -1] + r[1][2][0] * a_h[2][1][0][fi + 0][fj + 1][fk + -1] + r[2][1][0] * a_h[1][2][0][fi + 1][fj + 0][fk + -1] + r[2][2][0] * a_h[1][1][0][fi + 1][fj + 1][fk + -1]) * p[2][2][1] + (r[1][1][0] * a_h[2][2][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[2][2][0][fi + 0][fj + 0][fk + 0] + r[1][2][0] * a_h[2][1][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[2][1][0][fi + 0][fj + 1][fk + 0] + r[2][1][0] * a_h[1][2][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[1][2][0][fi + 1][fj + 0][fk + 0] + r[2][2][0] * a_h[1][1][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[1][1][0][fi + 1][fj + 1][fk + 0]) * p[2][2][0] + (r[1][2][0] * a_h[2][2][0][fi + 0][fj + 1][fk + -1] + r[2][2][0] * a_h[1][2][0][fi + 1][fj + 1][fk + -1]) * p[2][1][1] + (r[1][2][0] * a_h[2][2][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[2][2][0][fi + 0][fj + 1][fk + 0] + r[2][2][0] * a_h[1][2][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[1][2][0][fi + 1][fj + 1][fk + 0]) * p[2][1][0] + (r[2][1][0] * a_h[2][2][0][fi + 1][fj + 0][fk + -1] + r[2][2][0] * a_h[2][1][0][fi + 1][fj + 1][fk + -1]) * p[1][2][1] + (r[2][1][0] * a_h[2][2][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[2][2][0][fi + 1][fj + 0][fk + 0] + r[2][2][0] * a_h[2][1][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[2][1][0][fi + 1][fj + 1][fk + 0]) * p[1][2][0] + (r[2][2][0] * a_h[2][2][0][fi + 1][fj + 1][fk + -1]) * p[1][1][1] + (r[2][2][0] * a_h[2][2][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[2][2][0][fi + 1][fj + 1][fk + 0]) * p[1][1][0]));

                REQUIRE(a_2h[2][2][1][ci][cj][ck] == ((r[1][1][0] * a_h[2][2][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[2][2][0][fi + 0][fj + 0][fk + 0] + r[1][2][0] * a_h[2][1][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[2][1][0][fi + 0][fj + 1][fk + 0] + r[2][1][0] * a_h[1][2][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[1][2][0][fi + 1][fj + 0][fk + 0] + r[2][2][0] * a_h[1][1][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[1][1][0][fi + 1][fj + 1][fk + 0]) * p[2][2][2] + (r[1][1][0] * a_h[2][2][2][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[2][2][1][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[2][2][0][fi + 0][fj + 0][fk + 1] + r[1][2][0] * a_h[2][1][2][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[2][1][1][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[2][1][0][fi + 0][fj + 1][fk + 1] + r[2][1][0] * a_h[1][2][2][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[1][2][1][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[1][2][0][fi + 1][fj + 0][fk + 1] + r[2][2][0] * a_h[1][1][2][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[1][1][1][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[1][1][0][fi + 1][fj + 1][fk + 1]) * p[2][2][1] + (r[1][1][1] * a_h[2][2][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[2][2][1][fi + 0][fj + 0][fk + 1] + r[1][2][1] * a_h[2][1][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[2][1][1][fi + 0][fj + 1][fk + 1] + r[2][1][1] * a_h[1][2][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[1][2][1][fi + 1][fj + 0][fk + 1] + r[2][2][1] * a_h[1][1][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[1][1][1][fi + 1][fj + 1][fk + 1]) * p[2][2][0] + (r[1][2][0] * a_h[2][2][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[2][2][0][fi + 0][fj + 1][fk + 0] + r[2][2][0] * a_h[1][2][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[1][2][0][fi + 1][fj + 1][fk + 0]) * p[2][1][2] + (r[1][2][0] * a_h[2][2][2][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[2][2][1][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[2][2][0][fi + 0][fj + 1][fk + 1] + r[2][2][0] * a_h[1][2][2][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[1][2][1][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[1][2][0][fi + 1][fj + 1][fk + 1]) * p[2][1][1] + (r[1][2][1] * a_h[2][2][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[2][2][1][fi + 0][fj + 1][fk + 1] + r[2][2][1] * a_h[1][2][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[1][2][1][fi + 1][fj + 1][fk + 1]) * p[2][1][0] + (r[2][1][0] * a_h[2][2][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[2][2][0][fi + 1][fj + 0][fk + 0] + r[2][2][0] * a_h[2][1][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[2][1][0][fi + 1][fj + 1][fk + 0]) * p[1][2][2] + (r[2][1][0] * a_h[2][2][2][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[2][2][1][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[2][2][0][fi + 1][fj + 0][fk + 1] + r[2][2][0] * a_h[2][1][2][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[2][1][1][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[2][1][0][fi + 1][fj + 1][fk + 1]) * p[1][2][1] + (r[2][1][1] * a_h[2][2][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[2][2][1][fi + 1][fj + 0][fk + 1] + r[2][2][1] * a_h[2][1][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[2][1][1][fi + 1][fj + 1][fk + 1]) * p[1][2][0] + (r[2][2][0] * a_h[2][2][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[2][2][0][fi + 1][fj + 1][fk + 0]) * p[1][1][2] + (r[2][2][0] * a_h[2][2][2][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[2][2][1][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[2][2][0][fi + 1][fj + 1][fk + 1]) * p[1][1][1] + (r[2][2][1] * a_h[2][2][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[2][2][1][fi + 1][fj + 1][fk + 1]) * p[1][1][0]));

                REQUIRE(a_2h[2][2][2][ci][cj][ck] == ((r[1][1][1] * a_h[2][2][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[2][2][1][fi + 0][fj + 0][fk + 1] + r[1][2][1] * a_h[2][1][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[2][1][1][fi + 0][fj + 1][fk + 1] + r[2][1][1] * a_h[1][2][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[1][2][1][fi + 1][fj + 0][fk + 1] + r[2][2][1] * a_h[1][1][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[1][1][1][fi + 1][fj + 1][fk + 1]) * p[2][2][2] + (r[1][1][2] * a_h[2][2][2][fi + 0][fj + 0][fk + 1] + r[1][2][2] * a_h[2][1][2][fi + 0][fj + 1][fk + 1] + r[2][1][2] * a_h[1][2][2][fi + 1][fj + 0][fk + 1] + r[2][2][2] * a_h[1][1][2][fi + 1][fj + 1][fk + 1]) * p[2][2][1] + (r[1][2][1] * a_h[2][2][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[2][2][1][fi + 0][fj + 1][fk + 1] + r[2][2][1] * a_h[1][2][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[1][2][1][fi + 1][fj + 1][fk + 1]) * p[2][1][2] + (r[1][2][2] * a_h[2][2][2][fi + 0][fj + 1][fk + 1] + r[2][2][2] * a_h[1][2][2][fi + 1][fj + 1][fk + 1]) * p[2][1][1] + (r[2][1][1] * a_h[2][2][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[2][2][1][fi + 1][fj + 0][fk + 1] + r[2][2][1] * a_h[2][1][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[2][1][1][fi + 1][fj + 1][fk + 1]) * p[1][2][2] + (r[2][1][2] * a_h[2][2][2][fi + 1][fj + 0][fk + 1] + r[2][2][2] * a_h[2][1][2][fi + 1][fj + 1][fk + 1]) * p[1][2][1] + (r[2][2][1] * a_h[2][2][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[2][2][1][fi + 1][fj + 1][fk + 1]) * p[1][1][2] + (r[2][2][2] * a_h[2][2][2][fi + 1][fj + 1][fk + 1]) * p[1][1][1]));
            }
}

// Checks if the handcrafted version of calculating the Galerkin operator yields the same result as the previously
// optimized one (the one that uses intervals).
TEST_CASE("seq_galerkinHandcrafted_vs_galerkinOptimized")
{
    auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));

    int m = 8;
    int n = 8;
    int o = 8;
    int gh = 1;

    mgcl_test::TestUtility tu(deviceType);

    mgcl::VaryingStencil a_h(m, n, o, 3, gh, gh, gh);
    a_h.fill1dIndex(false);
    a_h.updateGhosts();

    mgcl::VaryingStencilGpu a_h_gpu(m, n, o, 3, gh, tu.getContext(), tu.getCommands(), tu.getProgram());
    a_h_gpu.fill(a_h, tu.getCommands(), true);

    auto a_2h_gpu = mgcl::MultigridEngine::galerkinOptimized(a_h_gpu, gh, m >> 1, n >> 1, o >> 1, tu.getProgram(), tu.getCommands(), tu.getContext(), nullptr, nullptr);
    auto a_2h = a_2h_gpu->read(tu.getCommands(), true);

    auto a_2h_hc = mgcl::MultigridEngine::galerkinHandcrafted(a_h, gh, m >> 1, n >> 1, o >> 1);

    REQUIRE(a_2h.isEqual(*a_2h_hc));
}

// Checks if the handcrafted version of calculating the Galerkin operator yields the same result as the previously
// optimized one (the one that uses intervals).
TEST_CASE("ocl_galerkinHandcrafted_vs_galerkinOptimized")
{
    auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));

    int m = 8;
    int n = 8;
    int o = 8;
    int gh = 1;

    mgcl_test::TestUtility tu(deviceType, true);

    mgcl::VaryingStencil a_h(m, n, o, 3, gh, gh, gh);
    a_h.fill1dIndex(false);
    a_h.updateGhosts();

    mgcl::VaryingStencilGpu a_h_gpu(m, n, o, 3, gh, tu.getContext(), tu.getCommands(), tu.getProgram());
    a_h_gpu.fill(a_h, tu.getCommands(), true);

    auto pd = std::make_unique<mgcl::ProfilingData>();

    auto a_2h_gpu = mgcl::MultigridEngine::galerkinOptimized(a_h_gpu, gh, m >> 1, n >> 1, o >> 1, tu.getProgram(), tu.getCommands(), tu.getContext(), nullptr, pd.get());
    auto a_2h = a_2h_gpu->read(tu.getCommands(), true);

    auto a_2h_hc_gpu = mgcl::MultigridEngine::galerkinHandcrafted(a_h_gpu, gh, m >> 1, n >> 1, o >> 1,
                                                                  tu.getProgram(), tu.getCommands(), tu.getContext(), nullptr, pd.get());
    auto a_2h_hc = a_2h_hc_gpu->read(tu.getCommands(), true);

    pd->printBestTimingsPerKernel();

    REQUIRE(a_2h.isEqual(a_2h_hc));
}

// Helper struct for galerkin. Defines an interval with integer start and end.
typedef struct Interval
{
    int start;
    int end;
} Interval;

// Returns the intersection of two intervals or [-1,-1] if they don't overlap
Interval intersect(Interval a, Interval b)
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

// Helper struct for galerkin. Defines a grid point with integer 3d coordinates.
typedef struct Point
{
    int x;
    int y;
    int z;
} Point;

// Returns the stencil entry indices of the stencil sitting at locationOfStencil that maps to mapsTo.
// No check is done, if the mapping is possible, i.e. the returned value might be outside of range [0,2].
// The result is just the difference of the indices plus one, since the stencil entry indices start at 0
// and not at -1.
Point stencilEntryThatMapsTo(Point locationOfStencil, Point mapsTo)
{
    Point ret;
    ret.x = mapsTo.x - locationOfStencil.x + 1;
    ret.y = mapsTo.y - locationOfStencil.y + 1;
    ret.z = mapsTo.z - locationOfStencil.z + 1;
    return ret;
};

// Returns the stencil entry indices of the stencil sitting at locationOfStencil that maps to mapsTo with radius=2.
// No check is done, if the mapping is possible, i.e. the returned value might be outside of range [0,4].
// The result is just the difference of the indices plus two, since the stencil entry indices start at 0
// and not at -2.
Point stencilEntryThatMapsToRad2(Point locationOfStencil, Point mapsTo)
{
    Point ret;
    ret.x = mapsTo.x - locationOfStencil.x + 2;
    ret.y = mapsTo.y - locationOfStencil.y + 2;
    ret.z = mapsTo.z - locationOfStencil.z + 2;
    return ret;
};

// Returns the grid point indices that is mapped to by the stencil entry of another point's stencil with radius=1.
// stencilEntry must be 0-based, hence the substraction by 1.
Point pointMappedToByStencilEntry(Point locationOfStencil, Point stencilEntry)
{
    Point ret;
    ret.x = locationOfStencil.x + (stencilEntry.x - 1);
    ret.y = locationOfStencil.y + (stencilEntry.y - 1);
    ret.z = locationOfStencil.z + (stencilEntry.z - 1);
    return ret;
};

// Returns the grid point indices that is mapped to by the stencil entry of another point's stencil with radius=2.
// stencilEntry must be 0-based, hence the substraction by 2.
Point pointMappedToByStencilEntryRad2(Point locationOfStencil, Point stencilEntry)
{
    Point ret;
    ret.x = locationOfStencil.x + (stencilEntry.x - 2);
    ret.y = locationOfStencil.y + (stencilEntry.y - 2);
    ret.z = locationOfStencil.z + (stencilEntry.z - 2);
    return ret;
};

// Returns the point on the fine grid that is related to the coarse grid point, respecting ghost cells.
Point coarseToFine(Point p, int ghc, int ghf)
{
    Point ret;
    ret.x = (p.x - ghc) * 2 + 1 + ghf;
    ret.y = (p.y - ghc) * 2 + 1 + ghf;
    ret.z = (p.z - ghc) * 2 + 1 + ghf;
    return ret;
};

TEST_CASE("galerkinOptimizedCachedRA")
{
    auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));

    int m = 8;
    int n = 8;
    int o = 8;
    int gh = 1;

    mgcl_test::TestUtility tu(deviceType);

    mgcl::VaryingStencil a_h(m, n, o, 3, gh, gh, gh);
    a_h.fill1dIndex(false);
    a_h.updateGhosts();

    mgcl::VaryingStencilGpu a_h_gpu(m, n, o, 3, gh, tu.getContext(), tu.getCommands(), tu.getProgram());
    a_h_gpu.fill(a_h, tu.getCommands(), true);

    auto a_2h_gpu = mgcl::MultigridEngine::galerkinOptimized(a_h_gpu, gh, m >> 1, n >> 1, o >> 1, tu.getProgram(), tu.getCommands(), tu.getContext(), nullptr, nullptr);
    auto a_2h_check = a_2h_gpu->read(tu.getCommands(), true);

    auto r = mgcl::create3dFullWeightRestrictionStencil();
    auto p = mgcl::create3dBilinearProlongationStencil();

    int mc = m >> 1;
    int nc = n >> 1;
    int oc = o >> 1;

    mgcl::VaryingStencil a_2h(mc, nc, oc, 3, gh, gh, gh);

    // define kernel arguments
    int m_c_loc = mc;
    int n_c_loc = nc;
    int o_c_loc = oc;
    int gh_f = gh;
    int gh_c = gh;
    int mgh_f = a_h.getMgh();
    int ngh_f = a_h.getNgh();
    int ogh_f = a_h.getOgh();
    int m_c_buf = mc;
    int n_c_buf = nc;
    int o_c_buf = oc;

    try
    {

        // for each real coarse grid point and the corresponding fine grid point
        // for (int ci = gh, fi = gh + 1; ci < mc + gh; ci++, fi += 2)
        //     for (int cj = gh, fj = gh + 1; cj < nc + gh; cj++, fj += 2)
        //         for (int ck = gh, fk = gh + 1; ck < oc + gh; ck++, fk += 2)
        for (int idx = 0; idx < mc * nc * oc; idx++)
        {
            int no = n_c_loc * o_c_loc;
            int i = idx / no;
            int j = (idx - i * no) / o_c_loc;
            int k = idx % o_c_loc;

            // only for real cells of coarse grid
            i += gh_c;
            j += gh_c;
            k += gh_c;

            // plane and grid size of ghosted fine grid
            int nogh_f = ngh_f * ogh_f;
            int mnogh_f = mgh_f * nogh_f;

            // plane and grid size of ghosted coarse grid
            int nogh_c = (n_c_buf + 2 * gh_c) * (o_c_buf + 2 * gh_c);
            int mnogh_c = (m_c_buf + 2 * gh_c) * nogh_c;

            CAPTURE(idx);

            // Calculate only for real cells of coarse grid
            if (i < m_c_loc + gh_c && n_c_loc + gh_c && o_c_loc + gh_c)
            {
                // calculate RA, which does not change for the current RAP entry. Locations of RAP and RA are equal.
                double ra[125];
                // for each coefficient of RA (which is a 5x5x5 stencil)
                for (int ii = 0; ii < 5; ii++)
                    for (int jj = 0; jj < 5; jj++)
                        for (int kk = 0; kk < 5; kk++)
                        {
                            // calculate fine grid point indices
                            Point gp_c = {i, j, k};
                            Point gp_f = coarseToFine(gp_c, gh_c, gh_f);
                            Point coeff_RA = {ii, jj, kk};

                            // RA_gp_mapto = find grid point mapped to by coeff
                            Point RA_gp_mapto = pointMappedToByStencilEntryRad2(gp_f, coeff_RA);
                            // RA_coeff_gps = set of grid points around RA_gp_mapto with radius = 1
                            Interval RA_coeff_gps[3] = {
                                {.start = RA_gp_mapto.x - 1, .end = RA_gp_mapto.x + 1},
                                {.start = RA_gp_mapto.y - 1, .end = RA_gp_mapto.y + 1},
                                {.start = RA_gp_mapto.z - 1, .end = RA_gp_mapto.z + 1},
                            };
                            // R_gps = set of grid points around R with radius = 1
                            Interval R_gps[3] = {
                                {.start = gp_f.x - 1, .end = gp_f.x + 1},
                                {.start = gp_f.y - 1, .end = gp_f.y + 1},
                                {.start = gp_f.z - 1, .end = gp_f.z + 1},
                            };
                            // S_RA_R = intersection(R_gps, RA_coeff_gps)
                            Interval S_RA_R[3] = {
                                intersect(R_gps[0], RA_coeff_gps[0]),
                                intersect(R_gps[1], RA_coeff_gps[1]),
                                intersect(R_gps[2], RA_coeff_gps[2]),
                            };
                            // foreach gp_s_ra_r in S_RA_R:
                            double sum = 0;
                            for (int srar_i = S_RA_R[0].start; srar_i <= S_RA_R[0].end; srar_i++)
                                for (int srar_j = S_RA_R[1].start; srar_j <= S_RA_R[1].end; srar_j++)
                                    for (int srar_k = S_RA_R[2].start; srar_k <= S_RA_R[2].end; srar_k++)
                                    {
                                        // Point that the coeff of R maps to, or where A is located at
                                        Point gp_srar = {srar_i, srar_j, srar_k};
                                        Point coeff_r = stencilEntryThatMapsTo(gp_f, gp_srar);

                                        // Coeff in A that maps to the resulting point RA_gp_mapto
                                        Point coeff_a = stencilEntryThatMapsTo(gp_srar, RA_gp_mapto);

                                        CAPTURE(coeff_a.x, coeff_a.y, coeff_a.z,
                                                gp_srar.x, gp_srar.y, gp_srar.z,
                                                coeff_r.x, coeff_r.y, coeff_r.z,
                                                RA_gp_mapto.x, RA_gp_mapto.y, RA_gp_mapto.z,
                                                gp_f.x, gp_f.y, gp_f.z,
                                                i, j, k,
                                                ii, jj, kk,
                                                gp_f.x, gp_f.y, gp_f.z,
                                                gp_srar.x, gp_srar.y, gp_srar.z);

                                        if (coeff_a.x * 9 * mnogh_f + coeff_a.y * 3 * mnogh_f + coeff_a.z * mnogh_f + gp_srar.x * nogh_f + gp_srar.y * ogh_f + gp_srar.z >= a_h.field1d().size() ||
                                            coeff_r.x * 9 + coeff_r.y * 3 + coeff_r.z >= 27)
                                        {
                                            std::cout << "accessing a_h out of bounds:" << std::endl
                                                      << "index a_h: " << coeff_a.x * 9 * mnogh_f + coeff_a.y * 3 * mnogh_f + coeff_a.z * mnogh_f + gp_srar.x * nogh_f + gp_srar.y * ogh_f + gp_srar.z << std::endl
                                                      << "index r_h: " << coeff_r.x * 9 + coeff_r.y * 3 + coeff_r.z << std::endl
                                                      << "coeff_a.xyz: " << coeff_a.x << ", " << coeff_a.y << ", " << coeff_a.z << std::endl
                                                      << "coeff_r.xyz: " << coeff_r.x << ", " << coeff_r.y << ", " << coeff_r.z << std::endl
                                                      << "gp_srar.xyz: " << gp_srar.x << ", " << gp_srar.y << ", " << gp_srar.z << std::endl
                                                      << "RA_gp_mapto: " << RA_gp_mapto.x << ", " << RA_gp_mapto.y << ", " << RA_gp_mapto.z << std::endl
                                                      << "S_RA_R: " << S_RA_R[0].start << ", " << S_RA_R[0].end << ", " << S_RA_R[1].start << ", " << S_RA_R[1].end << ", " << S_RA_R[2].start << ", " << S_RA_R[2].end << std::endl
                                                      << "gp_f: " << gp_f.x << ", " << gp_f.y << ", " << gp_f.z << std::endl
                                                      << "ii, jj, kk: " << ii << ", " << jj << ", " << kk << std::endl;
                                            FAIL("accessing a_h or r out of bounds");
                                        }

                                        // RA(gp, coeff) += R(gp, gp_s_ra_r) * A(gp_s_ra_r, coeff)
                                        sum += r.field1d().at(coeff_r.x * 9 + coeff_r.y * 3 + coeff_r.z) * a_h.field1d().at(coeff_a.x * 9 * mnogh_f + coeff_a.y * 3 * mnogh_f + coeff_a.z * mnogh_f + gp_srar.x * nogh_f + gp_srar.y * ogh_f + gp_srar.z);
                                    }

                            // Store in private RA
                            ra[ii * 25 + jj * 5 + kk] = sum;
                        }

                // for (int i = a_2h->getGhostsM(); i < (a_h.getM() >> 1) + a_2h->getGhostsM(); i++)
                //     for (int j = a_2h->getGhostsN(); j < (a_h.getN() >> 1) + a_2h->getGhostsN(); j++)
                //         for (int k = a_2h->getGhostsO(); k < (a_h.getO() >> 1) + a_2h->getGhostsO(); k++)
                // for each stencil entry of the coarse grid poiint this work-item maps to
                for (int ii = 0; ii < 3; ii++)
                    for (int jj = 0; jj < 3; jj++)
                        for (int kk = 0; kk < 3; kk++)
                        {
                            // calculate fine grid point indices
                            Point gp_c = {i, j, k};
                            Point gp_f = coarseToFine(gp_c, gh_c, gh_f);
                            Point entry_gpc = {ii, jj, kk};
                            Point entry_gpf = coarseToFine(
                                pointMappedToByStencilEntry(gp_c, entry_gpc),
                                gh_c, gh_f);

                            // find intersection S_P of neighbouring points for entry_gpf with reach=1 and gp_f with reach=2
                            Interval xa = {.start = gp_f.x - 2, .end = gp_f.x + 2};
                            Interval ya = {.start = gp_f.y - 2, .end = gp_f.y + 2};
                            Interval za = {.start = gp_f.z - 2, .end = gp_f.z + 2};
                            Interval xb = {.start = entry_gpf.x - 1, .end = entry_gpf.x + 1};
                            Interval yb = {.start = entry_gpf.y - 1, .end = entry_gpf.y + 1};
                            Interval zb = {.start = entry_gpf.z - 1, .end = entry_gpf.z + 1};

                            Interval S_P[3] = {
                                intersect(xa, xb),
                                intersect(ya, yb),
                                intersect(za, zb),
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
                                        Point tmp_p_indices = stencilEntryThatMapsTo(gp_sp, entry_gpf);

                                        // Use cached RA
                                        // Find coeff that is mapping to the current gp_sp
                                        Point coeff_ra = stencilEntryThatMapsToRad2(gp_f, gp_sp);

                                        if (tmp_p_indices.x * 9 + tmp_p_indices.y * 3 + tmp_p_indices.z >= 27)
                                        {
                                            std::cout << "accessing p out of bounds:" << std::endl
                                                      << "idx: " << tmp_p_indices.x * 9 + tmp_p_indices.y * 3 + tmp_p_indices.z << std::endl
                                                      << "size: " << p.field1d().size() << std::endl
                                                      << "gp_sp: " << gp_sp.x << ", " << gp_sp.y << ", " << gp_sp.z << std::endl
                                                      << "tmp_p_indices: " << tmp_p_indices.x << ", " << tmp_p_indices.y << ", " << tmp_p_indices.z << std::endl
                                                      << "coeff_ra: " << coeff_ra.x << ", " << coeff_ra.y << ", " << coeff_ra.z << std::endl;
                                            FAIL("accessing p out of bounds");
                                        }

                                        res += ra[coeff_ra.x * 25 + coeff_ra.y * 5 + coeff_ra.z] * p.field1d().at(tmp_p_indices.x * 9 + tmp_p_indices.y * 3 + tmp_p_indices.z);
                                        // End calc (R*A)*P
                                    }

                            // store res in rap
                            // (*a_2h)[ii][jj][kk][i][j][k] = res;
                            if (ii * 9 * mnogh_c + jj * 3 * mnogh_c + kk * mnogh_c + i * nogh_c + j * (o_c_buf + 2 * gh_c) + k >= a_2h.field1d().size())
                            {
                                std::cout << "accessing a_2h out of bounds:" << std::endl
                                          << "idx: " << ii * 9 * mnogh_c + jj * 3 * mnogh_c + kk * mnogh_c + i * nogh_c + j * (o_c_buf + 2 * gh_c) + k << std::endl
                                          << "size: " << a_2h.field1d().size() << std::endl
                                          << "ii,jj,kk: " << ii << "," << jj << "," << kk << std::endl
                                          << "i,j,k: " << i << "," << j << "," << k << std::endl;
                            }
                            a_2h.field1d().at(ii * 9 * mnogh_c + jj * 3 * mnogh_c + kk * mnogh_c + i * nogh_c + j * (o_c_buf + 2 * gh_c) + k) = res;
                        }
            }
        } // end for idx
    }
    catch (...)
    {
        // std::cout << str << std::endl;
        FAIL();
    }

    REQUIRE(a_2h_check.isEqual(a_2h));
}

TEST_CASE("galerkinOptimizedCachedRALocalMem")
{
    auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));

    int m = 8;
    int n = 8;
    int o = 8;
    int gh = 1;

    mgcl_test::TestUtility tu(deviceType);

    mgcl::VaryingStencil a_h(m, n, o, 3, gh, gh, gh);
    a_h.fill1dIndex(false);
    a_h.updateGhosts();

    mgcl::VaryingStencilGpu a_h_gpu(m, n, o, 3, gh, tu.getContext(), tu.getCommands(), tu.getProgram());
    a_h_gpu.fill(a_h, tu.getCommands(), true);

    auto a_2h_gpu = mgcl::MultigridEngine::galerkinOptimized(a_h_gpu, gh, m >> 1, n >> 1, o >> 1, tu.getProgram(), tu.getCommands(), tu.getContext(), nullptr, nullptr);
    auto a_2h_check = a_2h_gpu->read(tu.getCommands(), true);

    auto r = mgcl::create3dFullWeightRestrictionStencil();
    auto p = mgcl::create3dBilinearProlongationStencil();

    int mc = m >> 1;
    int nc = n >> 1;
    int oc = o >> 1;

    mgcl::VaryingStencil a_2h(mc, nc, oc, 3, gh, gh, gh);

    size_t local = 128;
    std::vector<double> ra_base_vec(mc * nc * oc * 125); // local memory for all work-groups
    double* ra_base = ra_base_vec.data();

    // define kernel arguments
    int m_c_loc = mc;
    int n_c_loc = nc;
    int o_c_loc = oc;
    int gh_f = gh;
    int gh_c = gh;
    int mgh_f = a_h.getMgh();
    int ngh_f = a_h.getNgh();
    int ogh_f = a_h.getOgh();
    int m_c_buf = mc;
    int n_c_buf = nc;
    int o_c_buf = oc;

    try
    {

        // for each real coarse grid point and the corresponding fine grid point
        // for (int ci = gh, fi = gh + 1; ci < mc + gh; ci++, fi += 2)
        //     for (int cj = gh, fj = gh + 1; cj < nc + gh; cj++, fj += 2)
        //         for (int ck = gh, fk = gh + 1; ck < oc + gh; ck++, fk += 2)
        for (int idx = 0; idx < mc * nc * oc; idx++)
        {
            int no = n_c_loc * o_c_loc;
            int i = idx / no;
            int j = (idx - i * no) / o_c_loc;
            int k = idx % o_c_loc;

            // only for real cells of coarse grid
            i += gh_c;
            j += gh_c;
            k += gh_c;

            // plane and grid size of ghosted fine grid
            int nogh_f = ngh_f * ogh_f;
            int mnogh_f = mgh_f * nogh_f;

            // plane and grid size of ghosted coarse grid
            int nogh_c = (n_c_buf + 2 * gh_c) * (o_c_buf + 2 * gh_c);
            int mnogh_c = (m_c_buf + 2 * gh_c) * nogh_c;

            // Base of RA for current work-item
            // get_local_id simulated by idx % wg-size
            double* ra = ra_base + (idx % local) * 125;

            // Calculate only for real cells of coarse grid
            if (i < m_c_loc + gh_c && n_c_loc + gh_c && o_c_loc + gh_c)
            {
                // calculate RA, which does not change for the current RAP entry. Locations of RAP and RA are equal.
                // for each coefficient of RA (which is a 5x5x5 stencil)
                for (int ii = 0; ii < 5; ii++)
                    for (int jj = 0; jj < 5; jj++)
                        for (int kk = 0; kk < 5; kk++)
                        {
                            // calculate fine grid point indices
                            Point gp_c = {i, j, k};
                            Point gp_f = coarseToFine(gp_c, gh_c, gh_f);
                            Point coeff_RA = {ii, jj, kk};

                            // RA_gp_mapto = find grid point mapped to by coeff
                            Point RA_gp_mapto = pointMappedToByStencilEntryRad2(gp_f, coeff_RA);
                            // RA_coeff_gps = set of grid points around RA_gp_mapto with radius = 1
                            Interval RA_coeff_gps[3] = {
                                {.start = RA_gp_mapto.x - 1, .end = RA_gp_mapto.x + 1},
                                {.start = RA_gp_mapto.y - 1, .end = RA_gp_mapto.y + 1},
                                {.start = RA_gp_mapto.z - 1, .end = RA_gp_mapto.z + 1},
                            };
                            // R_gps = set of grid points around R with radius = 1
                            Interval R_gps[3] = {
                                {.start = gp_f.x - 1, .end = gp_f.x + 1},
                                {.start = gp_f.y - 1, .end = gp_f.y + 1},
                                {.start = gp_f.z - 1, .end = gp_f.z + 1},
                            };
                            // S_RA_R = intersection(R_gps, RA_coeff_gps)
                            Interval S_RA_R[3] = {
                                intersect(R_gps[0], RA_coeff_gps[0]),
                                intersect(R_gps[1], RA_coeff_gps[1]),
                                intersect(R_gps[2], RA_coeff_gps[2]),
                            };
                            // foreach gp_s_ra_r in S_RA_R:
                            double sum = 0;
                            for (int srar_i = S_RA_R[0].start; srar_i <= S_RA_R[0].end; srar_i++)
                                for (int srar_j = S_RA_R[1].start; srar_j <= S_RA_R[1].end; srar_j++)
                                    for (int srar_k = S_RA_R[2].start; srar_k <= S_RA_R[2].end; srar_k++)
                                    {
                                        // Point that the coeff of R maps to, or where A is located at
                                        Point gp_srar = {srar_i, srar_j, srar_k};
                                        Point coeff_r = stencilEntryThatMapsTo(gp_f, gp_srar);

                                        // Coeff in A that maps to the resulting point RA_gp_mapto
                                        Point coeff_a = stencilEntryThatMapsTo(gp_srar, RA_gp_mapto);

                                        // RA(gp, coeff) += R(gp, gp_s_ra_r) * A(gp_s_ra_r, coeff)
                                        sum += r.field1d().at(coeff_r.x * 9 + coeff_r.y * 3 + coeff_r.z) * a_h.field1d().at(coeff_a.x * 9 * mnogh_f + coeff_a.y * 3 * mnogh_f + coeff_a.z * mnogh_f + gp_srar.x * nogh_f + gp_srar.y * ogh_f + gp_srar.z);
                                    }

                            // Store in private RA
                            ra[ii * 25 + jj * 5 + kk] = sum;
                        }

                // for (int i = a_2h->getGhostsM(); i < (a_h.getM() >> 1) + a_2h->getGhostsM(); i++)
                //     for (int j = a_2h->getGhostsN(); j < (a_h.getN() >> 1) + a_2h->getGhostsN(); j++)
                //         for (int k = a_2h->getGhostsO(); k < (a_h.getO() >> 1) + a_2h->getGhostsO(); k++)
                // for each stencil entry of the coarse grid poiint this work-item maps to
                for (int ii = 0; ii < 3; ii++)
                    for (int jj = 0; jj < 3; jj++)
                        for (int kk = 0; kk < 3; kk++)
                        {
                            // calculate fine grid point indices
                            Point gp_c = {i, j, k};
                            Point gp_f = coarseToFine(gp_c, gh_c, gh_f);
                            Point entry_gpc = {ii, jj, kk};
                            Point entry_gpf = coarseToFine(
                                pointMappedToByStencilEntry(gp_c, entry_gpc),
                                gh_c, gh_f);

                            // find intersection S_P of neighbouring points for entry_gpf with reach=1 and gp_f with reach=2
                            Interval xa = {.start = gp_f.x - 2, .end = gp_f.x + 2};
                            Interval ya = {.start = gp_f.y - 2, .end = gp_f.y + 2};
                            Interval za = {.start = gp_f.z - 2, .end = gp_f.z + 2};
                            Interval xb = {.start = entry_gpf.x - 1, .end = entry_gpf.x + 1};
                            Interval yb = {.start = entry_gpf.y - 1, .end = entry_gpf.y + 1};
                            Interval zb = {.start = entry_gpf.z - 1, .end = entry_gpf.z + 1};

                            Interval S_P[3] = {
                                intersect(xa, xb),
                                intersect(ya, yb),
                                intersect(za, zb),
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
                                        Point tmp_p_indices = stencilEntryThatMapsTo(gp_sp, entry_gpf);

                                        // Use cached RA
                                        // Find coeff that is mapping to the current gp_sp
                                        Point coeff_ra = stencilEntryThatMapsToRad2(gp_f, gp_sp);

                                        res += ra[coeff_ra.x * 25 + coeff_ra.y * 5 + coeff_ra.z] * p.field1d().at(tmp_p_indices.x * 9 + tmp_p_indices.y * 3 + tmp_p_indices.z);
                                        // End calc (R*A)*P
                                    }

                            // store res in rap
                            a_2h.field1d().at(ii * 9 * mnogh_c + jj * 3 * mnogh_c + kk * mnogh_c + i * nogh_c + j * (o_c_buf + 2 * gh_c) + k) = res;
                        }
            }
        } // end for idx
    }
    catch (...)
    {
        // std::cout << str << std::endl;
        FAIL();
    }

    REQUIRE(a_2h_check.isEqual(a_2h));
}

// Build a trace from the galerkin_optimized_cached_RA kernel for a single grid point
TEST_CASE("galerkinTraceCachedRA", "[.]")
{
    int m = 8;
    int n = 8;
    int o = 8;
    int gh = 1;

    mgcl::VaryingStencil a_h(m, n, o, 3, gh, gh, gh);
    a_h.fill1dIndex(false);
    a_h.updateGhosts();

    auto r = mgcl::create3dFullWeightRestrictionStencil();
    auto p = mgcl::create3dBilinearProlongationStencil();

    int mc = m >> 1;
    int nc = n >> 1;
    int oc = o >> 1;

    mgcl::VaryingStencil a_2h(mc, nc, oc, 3, gh, gh, gh);

    // define kernel arguments
    int m_c_loc = mc;
    int n_c_loc = nc;
    int o_c_loc = oc;
    int gh_f = gh;
    int gh_c = gh;
    int mgh_f = a_h.getMgh();
    int ngh_f = a_h.getNgh();
    int ogh_f = a_h.getOgh();
    int m_c_buf = mc;
    int n_c_buf = nc;
    int o_c_buf = oc;

    std::stringstream ss;

    // Calculate for coarse gp 1,1,1
    int ci = 1;
    int cj = 1;
    int ck = 1;
    int fi = ci * 2 + 1;
    int fj = cj * 2 + 1;
    int fk = ck * 2 + 1;
    int idx = ci * nc * oc + cj * oc + ck;
    // for (int idx = 0; idx < mc * nc * oc; idx++)
    {
        int no = n_c_loc * o_c_loc;
        int i = idx / no;
        int j = (idx - i * no) / o_c_loc;
        int k = idx % o_c_loc;

        // only for real cells of coarse grid
        i += gh_c;
        j += gh_c;
        k += gh_c;

        // plane and grid size of ghosted fine grid
        int nogh_f = ngh_f * ogh_f;
        int mnogh_f = mgh_f * nogh_f;

        // plane and grid size of ghosted coarse grid
        int nogh_c = (n_c_buf + 2 * gh_c) * (o_c_buf + 2 * gh_c);
        int mnogh_c = (m_c_buf + 2 * gh_c) * nogh_c;

        // Calculate only for real cells of coarse grid
        if (i < m_c_loc + gh_c && n_c_loc + gh_c && o_c_loc + gh_c)
        {
            // calculate RA, which does not change for the current RAP entry. Locations of RAP and RA are equal.
            double ra[125];
            // for each coefficient of RA (which is a 5x5x5 stencil)
            for (int ii = 0; ii < 5; ii++)
                for (int jj = 0; jj < 5; jj++)
                    for (int kk = 0; kk < 5; kk++)
                    {
                        // calculate fine grid point indices
                        Point gp_c = {i, j, k};
                        Point gp_f = coarseToFine(gp_c, gh_c, gh_f);
                        Point coeff_RA = {ii, jj, kk};

                        // RA_gp_mapto = find grid point mapped to by coeff
                        Point RA_gp_mapto = pointMappedToByStencilEntryRad2(gp_f, coeff_RA);
                        // RA_coeff_gps = set of grid points around RA_gp_mapto with radius = 1
                        Interval RA_coeff_gps[3] = {
                            {.start = RA_gp_mapto.x - 1, .end = RA_gp_mapto.x + 1},
                            {.start = RA_gp_mapto.y - 1, .end = RA_gp_mapto.y + 1},
                            {.start = RA_gp_mapto.z - 1, .end = RA_gp_mapto.z + 1},
                        };
                        // R_gps = set of grid points around R with radius = 1
                        Interval R_gps[3] = {
                            {.start = gp_f.x - 1, .end = gp_f.x + 1},
                            {.start = gp_f.y - 1, .end = gp_f.y + 1},
                            {.start = gp_f.z - 1, .end = gp_f.z + 1},
                        };
                        // S_RA_R = intersection(R_gps, RA_coeff_gps)
                        Interval S_RA_R[3] = {
                            intersect(R_gps[0], RA_coeff_gps[0]),
                            intersect(R_gps[1], RA_coeff_gps[1]),
                            intersect(R_gps[2], RA_coeff_gps[2]),
                        };
                        // foreach gp_s_ra_r in S_RA_R:
                        double sum = 0;
                        ss << "ra[" << ii << " * 25 + " << jj << " * 5 + " << kk << "] = ";
                        for (int srar_i = S_RA_R[0].start; srar_i <= S_RA_R[0].end; srar_i++)
                            for (int srar_j = S_RA_R[1].start; srar_j <= S_RA_R[1].end; srar_j++)
                                for (int srar_k = S_RA_R[2].start; srar_k <= S_RA_R[2].end; srar_k++)
                                {
                                    // Point that the coeff of R maps to, or where A is located at
                                    Point gp_srar = {srar_i, srar_j, srar_k};
                                    Point coeff_r = stencilEntryThatMapsTo(gp_f, gp_srar);

                                    // Coeff in A that maps to the resulting point RA_gp_mapto
                                    Point coeff_a = stencilEntryThatMapsTo(gp_srar, RA_gp_mapto);

                                    // for the trace: How can I get from f_point, where R is located at, to gp_srar, where A is located at?
                                    // -> equal to coeff_r! -> e.g. fi + coeff_r.x = srar_i
                                    // Point coeff_gp_srar = stencilEntryThatMapsTo(gp_f, gp_srar);

                                    // RA(gp, coeff) += R(gp, gp_s_ra_r) * A(gp_s_ra_r, coeff)
                                    // sum += r[coeff_r.x * 9 + coeff_r.y * 3 + coeff_r.z] * a_h[coeff_a.x * 9 * mnogh_f + coeff_a.y * 3 * mnogh_f + coeff_a.z * mnogh_f + gp_srar.x * nogh_f + gp_srar.y * ogh_f + gp_srar.z];
                                    ss << "r[" << coeff_r.x << " * 9 + " << coeff_r.y << " * 3 + " << coeff_r.z << "] * a_h[" << coeff_a.x << " * 9 * mnogh_f + " << coeff_a.y << " * 3 * mnogh_f + " << coeff_a.z << " * mnogh_f + (fi + " << coeff_r.x - 1 << ") * nogh_f + (fj + " << coeff_r.y - 1 << ") * ogh_f + fk + " << coeff_r.z - 1
                                       << "] + ";
                                }

                        // Store in private RA
                        // ra[ii * 25 + jj * 5 + kk] = sum;
                        ss << ";" << std::endl;
                    }

            ss << std::endl;

            // for (int i = a_2h->getGhostsM(); i < (a_h.getM() >> 1) + a_2h->getGhostsM(); i++)
            //     for (int j = a_2h->getGhostsN(); j < (a_h.getN() >> 1) + a_2h->getGhostsN(); j++)
            //         for (int k = a_2h->getGhostsO(); k < (a_h.getO() >> 1) + a_2h->getGhostsO(); k++)
            // for each stencil entry of the coarse grid poiint this work-item maps to
            for (int ii = 0; ii < 3; ii++)
                for (int jj = 0; jj < 3; jj++)
                    for (int kk = 0; kk < 3; kk++)
                    {
                        // calculate fine grid point indices
                        Point gp_c = {i, j, k};
                        Point gp_f = coarseToFine(gp_c, gh_c, gh_f);
                        Point entry_gpc = {ii, jj, kk};
                        Point entry_gpf = coarseToFine(
                            pointMappedToByStencilEntry(gp_c, entry_gpc),
                            gh_c, gh_f);

                        // find intersection S_P of neighbouring points for entry_gpf with reach=1 and gp_f with reach=2
                        Interval xa = {.start = gp_f.x - 2, .end = gp_f.x + 2};
                        Interval ya = {.start = gp_f.y - 2, .end = gp_f.y + 2};
                        Interval za = {.start = gp_f.z - 2, .end = gp_f.z + 2};
                        Interval xb = {.start = entry_gpf.x - 1, .end = entry_gpf.x + 1};
                        Interval yb = {.start = entry_gpf.y - 1, .end = entry_gpf.y + 1};
                        Interval zb = {.start = entry_gpf.z - 1, .end = entry_gpf.z + 1};

                        Interval S_P[3] = {
                            intersect(xa, xb),
                            intersect(ya, yb),
                            intersect(za, zb),
                        };

                        // Start calc (R*A)*P
                        double res = 0;

                        ss << "a_2h[" << ii << " * 9 * mnogh_c + " << jj << " * 3 * mnogh_c + " << kk << " * mnogh_c + ci * nogh_c + cj * (o_c_buf + 2 * gh_c) + ck] = ";
                        // for each fine grid point gp_sp in S_P:
                        for (int spi = S_P[0].start; spi <= S_P[0].end; spi++)
                            for (int spj = S_P[1].start; spj <= S_P[1].end; spj++)
                                for (int spk = S_P[2].start; spk <= S_P[2].end; spk++)
                                {
                                    Point gp_sp = {spi, spj, spk};
                                    // tmp_p <- in stencil P located at gp_sp: Find stencil entry entry_p that maps to entry_gpf. Since
                                    // gp_sp is in S_P, it is ensured that the stencil has a stencil entry that maps to entry_gpf.
                                    Point tmp_p_indices = stencilEntryThatMapsTo(gp_sp, entry_gpf);

                                    // Use cached RA
                                    // Find coeff that is mapping to the current gp_sp
                                    Point coeff_ra = stencilEntryThatMapsToRad2(gp_f, gp_sp);

                                    // res += ra[coeff_ra.x * 25 + coeff_ra.y * 5 + coeff_ra.z] * p[tmp_p_indices.x * 9 + tmp_p_indices.y * 3 + tmp_p_indices.z];
                                    ss << "ra[" << coeff_ra.x << " * 25 + " << coeff_ra.y << " * 5 + " << coeff_ra.z << "] * p[" << tmp_p_indices.x << " * 9 + " << tmp_p_indices.y << " * 3 + " << tmp_p_indices.z << "] + ";
                                    // End calc (R*A)*P
                                }

                        // store res in rap
                        // a_2h[ii * 9 * mnogh_c + jj * 3 * mnogh_c + kk * mnogh_c + i * nogh_c + j * (o_c_buf + 2 * gh_c) + k] = res;
                        ss << ";" << std::endl;
                    }
        }
    }

    std::cout << ss.str() << std::endl;
}