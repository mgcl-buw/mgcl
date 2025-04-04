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
TEST_CASE("seq_bs_residual")
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