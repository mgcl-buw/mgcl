#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cmath>
#include <iostream>

#include "../../src/mgcl/cuboid_bs.hpp"
#include "../../src/mgcl/level.hpp"
#include "../../src/mgcl/multigrid_engine.hpp"
#include "../../src/mgcl/stencil.hpp"
#include "../cli_args.hpp"
#include "../device_type_generator.hpp"
#include "../test_results.hpp"
#include "../test_utility.hpp"

// Tests blokcstencil restriction one one c-point, i.e. from real grids 2x2x2 to 1x1x1
TEST_CASE("seq_restriction_single_point")
{
    int mf = 2;
    int nf = 2;
    int of = 2;
    int mc = mf / 2;
    int nc = nf / 2;
    int oc = of / 2;
    int ghosts_m = 1;
    int ghosts_n = 1;
    int ghosts_o = 1;
    int mfgh = mf + 2 * ghosts_m;
    int nfgh = nf + 2 * ghosts_n;
    int ofgh = of + 2 * ghosts_o;
    int mcgh = mc + 2 * ghosts_m;
    int ncgh = nc + 2 * ghosts_n;
    int ocgh = oc + 2 * ghosts_o;
    int blocksize = 2;
    int width = 3;

    bool periodic = false;

    mgcl::CuboidBS c_fine(mf, nf, of, ghosts_m, ghosts_n, ghosts_o, blocksize);
    mgcl::CuboidBS c_coarse(mc, nc, oc, ghosts_m, ghosts_n, ghosts_o, blocksize);
    c_fine.fill1dIndex(false);
    // c_fine.dumpToFile("c_fine.txt", false);

    // fill diagonals with full-weighted restriction operator, which is the same as in the scalar case
    mgcl::FixedBlockstencil fbs(width, blocksize);
    fbs.fill(0.0);

    // fill blockstencil for gp 1,1,1 with increasing values starting from 0
    int cnt = 0;
    for (int ci = 0; ci < width; ci++)
        for (int cj = 0; cj < width; cj++)
            for (int ck = 0; ck < width; ck++)
                for (int bi = 0; bi < blocksize; bi++)
                    for (int bj = 0; bj < blocksize; bj++)
                    {
                        fbs[bi][bj][ci][cj][ck] = cnt++;
                    }
    // fbs.dumpToFile("fbs.txt", false);

    mgcl::args::RestrictionBSSeqArgs args{
        c_fine,
        c_coarse,
        fbs,
        periodic, true, true,
        nullptr, nullptr};

    mgcl::MultigridEngine::restrictSeqBlockstencil(args);

    // c_coarse.dumpToFile("c_coarse.txt", false);

    // Check against manually calculated results
    REQUIRE(c_coarse[ghosts_m][ghosts_n][ghosts_o][0] == 284787);
    REQUIRE(c_coarse[ghosts_m][ghosts_n][ghosts_o][1] == 293913);
}

// Tests restriction using a blockstencil and checks results against multiple restrictions using scalar stencils, while
// the blockstencil only has entries on its diagonal, i.e. is only affecting one quantity at a time.
TEST_CASE("seq_restriction_blockstencil_independent_quantities")
{
    int m = 16;
    int n = 16;
    int o = 16;
    int ghosts_m = 1;
    int ghosts_n = 1;
    int ghosts_o = 1;
    int mgh = m + 2 * ghosts_m;
    int ngh = n + 2 * ghosts_n;
    int ogh = o + 2 * ghosts_o;
    int blocksize = 2;

    bool periodic = true;

    auto p1 = std::make_shared<mgcl::Problem>(m, n, o);
    p1->setBc(periodic ? mgcl::BC::PERIODIC : mgcl::BC::DIRICHLET);
    mgcl::Level lv_fine1(p1.get(), 0);
    mgcl::Level lv_coarse1(p1.get(), 1);
    auto p2 = std::make_shared<mgcl::Problem>(m, n, o);
    p2->setBc(periodic ? mgcl::BC::PERIODIC : mgcl::BC::DIRICHLET);
    mgcl::Level lv_fine2(p2.get(), 0);
    mgcl::Level lv_coarse2(p2.get(), 1);

    mgcl::Cuboid c_fine1(m, n, o, ghosts_m, ghosts_n, ghosts_o);
    mgcl::Cuboid c_coarse1(m / 2, n / 2, o / 2, ghosts_m, ghosts_n, ghosts_o);
    mgcl::Cuboid c_fine2(m, n, o, ghosts_m, ghosts_n, ghosts_o);
    mgcl::Cuboid c_coarse2(m / 2, n / 2, o / 2, ghosts_m, ghosts_n, ghosts_o);
    mgcl::CuboidBS c_finebs(m, n, o, ghosts_m, ghosts_n, ghosts_o, blocksize);
    mgcl::CuboidBS c_coarsebs(c_coarse1.getM(), c_coarse1.getN(), c_coarse1.getO(), ghosts_m, ghosts_n, ghosts_o, blocksize);
    c_finebs.fill1dIndex(false);

    // Fill scalar cuboids with values of block cuboid
    for (int i = 0; i < mgh; i++)
        for (int j = 0; j < ngh; j++)
            for (int k = 0; k < ogh; k++)
            {
                c_fine1[i][j][k] = c_finebs[i][j][k][0];
                c_fine2[i][j][k] = c_finebs[i][j][k][1];
            }

    // fill diagonals with full-weighted restriction operator, which is the same as in the scalar case
    mgcl::FixedBlockstencil fbs(3, blocksize);
    for (int b = 0; b < blocksize; b++)
    {
        fbs[b][b][1][1][1] = 0.125;
        fbs[b][b][1][1][0] = 0.0625;
        fbs[b][b][1][1][2] = 0.0625;
        fbs[b][b][1][0][1] = 0.0625;
        fbs[b][b][1][2][1] = 0.0625;
        fbs[b][b][0][1][1] = 0.0625;
        fbs[b][b][2][1][1] = 0.0625;

        fbs[b][b][1][0][0] = 0.03125;
        fbs[b][b][1][0][2] = 0.03125;
        fbs[b][b][1][2][0] = 0.03125;
        fbs[b][b][1][2][2] = 0.03125;
        fbs[b][b][0][1][0] = 0.03125;
        fbs[b][b][0][1][2] = 0.03125;
        fbs[b][b][2][1][0] = 0.03125;
        fbs[b][b][2][1][2] = 0.03125;
        fbs[b][b][0][0][1] = 0.03125;
        fbs[b][b][0][2][1] = 0.03125;
        fbs[b][b][2][0][1] = 0.03125;
        fbs[b][b][2][2][1] = 0.03125;

        fbs[b][b][0][0][0] = 0.015625;
        fbs[b][b][0][0][2] = 0.015625;
        fbs[b][b][0][2][0] = 0.015625;
        fbs[b][b][0][2][2] = 0.015625;
        fbs[b][b][2][0][0] = 0.015625;
        fbs[b][b][2][0][2] = 0.015625;
        fbs[b][b][2][2][0] = 0.015625;
        fbs[b][b][2][2][2] = 0.015625;
    }

    SECTION("restrictSeq")
    {
        mgcl::MultigridEngine::restrictSeq(lv_fine1, lv_coarse1, c_fine1, c_coarse1);
        mgcl::MultigridEngine::restrictSeq(lv_fine2, lv_coarse2, c_fine2, c_coarse2);

        mgcl::args::RestrictionBSSeqArgs args{
            c_finebs,
            c_coarsebs,
            fbs,
            true, true, true,
            nullptr, nullptr};

        mgcl::MultigridEngine::restrictSeqBlockstencil(args);

        // Check both components
        for (int i = ghosts_m; i < m + ghosts_m; i++)
            for (int j = ghosts_n; j < n + ghosts_n; j++)
                for (int k = ghosts_o; k < o + ghosts_o; k++)
                {
                    CAPTURE(i, j, k);
                    REQUIRE(c_fine1[i][j][k] == c_finebs[i][j][k][0]);
                    REQUIRE(c_fine2[i][j][k] == c_finebs[i][j][k][1]);
                }
    }

    // SECTION("restrict OpenCL")
    // {
    //     auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));

    //     p->setDeviceType(deviceType);

    //     mgcl_test::TestUtility tu(p);
    //     mgcl::CuboidGpu d_c_fine(tu.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, *c_fine);
    //     mgcl::CuboidGpu d_c_coarse(tu.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, *c_coarse);

    //     mgcl::MultigridEngine::restrict(lv_fine, lv_coarse, d_c_fine, d_c_coarse);
    //     tu.finish();

    //     auto c_fine_out = d_c_fine.read(tu.getCommands(), nullptr, true);
    //     auto c_coarse_out = d_c_coarse.read(tu.getCommands(), nullptr, true);

    //     REQUIRE(c_fine_out->isEqual(*c_expected_fine));
    //     REQUIRE(c_coarse_out->isEqual(*c_expected_coarse));
    // }
}
