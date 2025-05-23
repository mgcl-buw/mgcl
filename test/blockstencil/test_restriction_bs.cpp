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
TEST_CASE("restriction_blockstencil_independent_quantities")
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

    mgcl::MultigridEngine::restrictSeq(lv_fine1, lv_coarse1, c_fine1, c_coarse1);
    mgcl::MultigridEngine::restrictSeq(lv_fine2, lv_coarse2, c_fine2, c_coarse2);

    SECTION("seq")
    {
        mgcl::args::RestrictionBSSeqArgs args{
            c_finebs,
            c_coarsebs,
            fbs,
            true, true, true,
            nullptr, nullptr};

        mgcl::MultigridEngine::restrictSeqBlockstencil(args);

        // Check both components
        for (int i = ghosts_m; i < m / 2 + ghosts_m; i++)
            for (int j = ghosts_n; j < n / 2 + ghosts_n; j++)
                for (int k = ghosts_o; k < o / 2 + ghosts_o; k++)
                {
                    CAPTURE(i, j, k);
                    REQUIRE(c_coarse1[i][j][k] == c_coarsebs[i][j][k][0]);
                    REQUIRE(c_coarse2[i][j][k] == c_coarsebs[i][j][k][1]);
                }
    }

    // TODO test kernel with fixed R

    SECTION("ocl")
    {
        auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));

        auto v_dummy = std::make_shared<mgcl::Cuboid>(1, 1, 1);
        auto f_dummy = std::make_shared<mgcl::Cuboid>(1, 1, 1);
        auto p_dummy = std::make_shared<mgcl::Problem>(1, 1, 1, f_dummy, v_dummy);
        p_dummy->setUseOpencl(true);
        p_dummy->setDeviceType(deviceType);
        p_dummy->setProfilingEnabled(true);
        p_dummy->getOpenCLHelper().setPreprocessorConstant("BLOCKSIZE", std::to_string(blocksize));
        p_dummy->init();

        mgcl::CuboidBSGpu d_c_fine(p_dummy->getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, c_finebs);
        mgcl::CuboidBSGpu d_c_coarse(p_dummy->getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, c_coarsebs);
        mgcl::FixedBlockstencilGpu d_fbs(fbs, p_dummy->getContext(), p_dummy->getCommands());

        // We don't need planebuf and send/recv bufs since we're not using MPI. So just nullptr.
        mgcl::args::RestrictionBSOclArgs args{
            d_c_fine, d_c_coarse, d_fbs,
            periodic, true, true,
            nullptr, nullptr, nullptr,
            p_dummy->getProgram(), p_dummy->getCommands(), p_dummy->getContext(),
            nullptr, nullptr,
            &p_dummy->getKernelConfig(), p_dummy->getProfilingData()};

        mgcl::MultigridEngine::restrictBlockstencil(args);

        p_dummy->finish();
        d_c_coarse.read(p_dummy->getCommands(), &c_coarsebs, true);

        // Check both components
        for (int i = ghosts_m; i < m / 2 + ghosts_m; i++)
            for (int j = ghosts_n; j < n / 2 + ghosts_n; j++)
                for (int k = ghosts_o; k < o / 2 + ghosts_o; k++)
                {
                    CAPTURE(i, j, k);
                    REQUIRE(c_coarse1[i][j][k] == c_coarsebs[i][j][k][0]);
                    REQUIRE(c_coarse2[i][j][k] == c_coarsebs[i][j][k][1]);
                }
    }
}
