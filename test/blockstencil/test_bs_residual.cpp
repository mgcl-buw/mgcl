#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <iostream>
#include <memory>

#include "../../src/mgcl/blockstencil.hpp"
#include "../../src/mgcl/cuboid_bs.hpp"
#include "../../src/mgcl/level.hpp"
#include "../../src/mgcl/multigrid_engine.hpp"
#include "../cli_args.hpp"
#include "../device_type_generator.hpp"
#include "../test_results.hpp"
#include "../test_utility.hpp"

// Test applying a blockstencil on one point
TEST_CASE("seq_bs_residual_single_point")
{
    int m = 1;
    int n = 1;
    int o = 1;
    int gh = 1;
    int blocksize = 2;
    int width = 3;
    int mgh = m + 2 * gh;
    int ngh = n + 2 * gh;
    int ogh = o + 2 * gh;

    bool periodic = true;

    mgcl::MGCL_RESIDUAL_NORM resnorm = mgcl::MGCL_L2;

    mgcl::CuboidBS v(m, n, o, gh, gh, gh, blocksize);
    mgcl::CuboidBS r(m, n, o, gh, gh, gh, blocksize);
    mgcl::CuboidBS f(m, n, o, gh, gh, gh, blocksize);
    mgcl::Blockstencil bs(m, n, o, width, blocksize, gh, gh, gh);

    v.fill1dIndex(false);
    f.fill1dIndex(false);
    bs.fill(0.0);

    // fill blockstencil for gp 1,1,1 with increasing values starting from 0
    int cnt = 0;
    for (int ci = 0; ci < width; ci++)
        for (int cj = 0; cj < width; cj++)
            for (int ck = 0; ck < width; ck++)
                for (int bi = 0; bi < blocksize; bi++)
                    for (int bj = 0; bj < blocksize; bj++)
                    {
                        bs[bi][bj][ci][cj][ck][1][1][1] = cnt++;
                    }

    mgcl::args::ResidualBSSeqArgs args{
        f,
        v,
        r,
        resnorm,
        bs,
        true,
        periodic,
        true, 0, 0, 0, nullptr

    };

    double res = mgcl::MultigridEngine::residualSeq(args);

    // calculated by hand
    REQUIRE(r[1][1][1][0] == -101323);
    REQUIRE(r[1][1][1][1] == -104184);
}

// Test applying a blockstencil on multiple grid points with independent quantitites, i.e.
// a blockstencil only has entries on its diagonal.
// The result is compared to two residual separate calculations of the two quantities.
TEST_CASE("seq_bs_residual_independent_quantities")
{
    int m = 4;
    int n = 4;
    int o = 4;
    int gh = 1;
    int blocksize = 2;
    int width = 3;
    int mgh = m + 2 * gh;
    int ngh = n + 2 * gh;
    int ogh = o + 2 * gh;

    bool periodic = true;

    mgcl::MGCL_RESIDUAL_NORM resnorm = mgcl::MGCL_L2;

    mgcl::CuboidBS v(m, n, o, gh, gh, gh, blocksize);
    mgcl::CuboidBS r(m, n, o, gh, gh, gh, blocksize);
    mgcl::CuboidBS f(m, n, o, gh, gh, gh, blocksize);
    mgcl::Blockstencil bs(m, n, o, width, blocksize, 0, 0, 0);

    mgcl::Cuboid v1(m, n, o, gh, gh, gh);
    mgcl::Cuboid f1(m, n, o, gh, gh, gh);
    mgcl::Cuboid r1(m, n, o, gh, gh, gh);
    mgcl::Cuboid v2(m, n, o, gh, gh, gh);
    mgcl::Cuboid f2(m, n, o, gh, gh, gh);
    mgcl::Cuboid r2(m, n, o, gh, gh, gh);
    mgcl::VaryingStencil sv1(m, n, o, width, 0, 0, 0);
    mgcl::VaryingStencil sv2(m, n, o, width, 0, 0, 0);

    v.fill1dIndex(false);
    f.fill1dIndex(false);
    bs.fill1dIndex(false);

    // fill different coefficients for v1 and v2
    // clang-format off
    int cnt = 0;
    for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
    for (int k = 0; k < o; k++)
        for (int ii = 0; ii < width; ii++)
        for (int jj = 0; jj < width; jj++)
        for (int kk = 0; kk < width; kk++)
        {
            sv1[ii][jj][kk][i][j][k] = cnt++;
            sv2[ii][jj][kk][i][j][k] = -cnt;
        }
    // clang-format on

    // fill blockstencil diagonal with values from sv1 and sv2
    for (int ci = 0; ci < width; ci++)
        for (int cj = 0; cj < width; cj++)
            for (int ck = 0; ck < width; ck++)
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                        for (int k = 0; k < o; k++)
                        {
                            bs[0][0][ci][cj][ck][i][j][k] = sv1[ci][cj][ck][i][j][k];
                            bs[1][1][ci][cj][ck][i][j][k] = sv2[ci][cj][ck][i][j][k];
                            bs[0][1][ci][cj][ck][i][j][k] = 0;
                            bs[1][0][ci][cj][ck][i][j][k] = 0;
                        }

    // fill v1 and v2 with values of v, vice versa for f
    for (int i = 0; i < mgh; i++)
        for (int j = 0; j < ngh; j++)
            for (int k = 0; k < ogh; k++)
            {
                v1[i][j][k] = v[i][j][k][0];
                v2[i][j][k] = v[i][j][k][1];
                f1[i][j][k] = f[i][j][k][0];
                f2[i][j][k] = f[i][j][k][1];
            }

    v.updateGhosts(nullptr, true);
    mgcl::MultigridEngine::updateGhostsSeq(v1, nullptr, true, false);
    mgcl::MultigridEngine::updateGhostsSeq(v2, nullptr, true, false);

    mgcl::args::ResidualBSSeqArgs args{
        f,
        v,
        r,
        resnorm,
        bs,
        true,
        periodic,
        true, 0, 0, 0, nullptr

    };

    double res = mgcl::MultigridEngine::residualSeq(args);
    double res1 = mgcl::MultigridEngine::residualSeq(f1, v1, r1, resnorm, mgcl::MGCL_VARYING, 0, &sv1, nullptr,
                                                     true, true, true);
    double res2 = mgcl::MultigridEngine::residualSeq(f2, v2, r2, resnorm, mgcl::MGCL_VARYING, 0, &sv2, nullptr,
                                                     true, true, true);

    // Check both r components
    for (int i = gh; i < m + gh; i++)
        for (int j = gh; j < n + gh; j++)
            for (int k = gh; k < o + gh; k++)
            {
                CAPTURE(i, j, k);
                REQUIRE(r1[i][j][k] == r[i][j][k][0]);
                REQUIRE(r2[i][j][k] == r[i][j][k][1]);
            }
}