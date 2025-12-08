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

// Tests blokcstencil prolongation for one c-point, i.e. from real grids 4x4x4 to 8x8x8
TEST_CASE("prolongation_single_point")
{
    int mf = 8;
    int nf = 8;
    int of = 8;
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
    c_fine.fillRandom();
    int cnt = 0;
    for (int i = ghosts_m; i < mc + ghosts_m; i++)
        for (int j = ghosts_n; j < nc + ghosts_n; j++)
            for (int k = ghosts_o; k < oc + ghosts_o; k++)
                for (int b = 0; b < blocksize; b++)
                {
                    c_coarse[b][i][j][k] = cnt++;
                }
    // c_coarse.dumpToFile("c_coarse.txt", false);

    // fill diagonals with full-weighted restriction operator, which is the same as in the scalar case
    mgcl::FixedBlockstencil fbs(width, blocksize);
    fbs.fill(0.0);

    // fill blockstencil for gp 1,1,1 with increasing values starting from 0
    cnt = 0;
    for (int ci = 0; ci < width; ci++)
        for (int cj = 0; cj < width; cj++)
            for (int ck = 0; ck < width; ck++)
                for (int bi = 0; bi < blocksize; bi++)
                    for (int bj = 0; bj < blocksize; bj++)
                    {
                        fbs[bi][bj][ci][cj][ck] = cnt++;
                    }
    // fbs.dumpToFile("fbs.txt", false);

    SECTION("seq")
    {
        mgcl::args::ProlongationBSSeqArgs args{
            c_fine,
            c_coarse,
            fbs,
            periodic, true, true,
            nullptr, nullptr};

        mgcl::MultigridEngine::prolongateSeqBlockstencil(args);
    }

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

        mgcl::CuboidBSGpu d_c_fine(p_dummy->getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, c_fine);
        mgcl::CuboidBSGpu d_c_coarse(p_dummy->getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, c_coarse);
        mgcl::FixedBlockstencilGpu d_fbs(fbs, p_dummy->getContext(), p_dummy->getCommands());

        // We don't need planebuf and send/recv bufs since we're not using MPI. So just nullptr.
        mgcl::args::ProlongationBSOclArgs args{
            d_c_fine, d_c_coarse, d_fbs,
            periodic, true, true,
            nullptr, nullptr, nullptr,
            p_dummy->getProgram(), p_dummy->getCommands(), p_dummy->getContext(),
            nullptr, nullptr,
            &p_dummy->getKernelConfig(), p_dummy->getProfilingData()};

        mgcl::MultigridEngine::prolongateBlockstencil(args);

        p_dummy->finish();
        d_c_fine.read(p_dummy->getCommands(), &c_fine, true);
    }

    // c_fine.dumpToFile("c_fine.txt", false);

    // Check against manually calculated results
    REQUIRE(c_fine[0][ghosts_m + 3][ghosts_n + 3][ghosts_o + 3] == 4463); // self
    REQUIRE(c_fine[1][ghosts_m + 3][ghosts_n + 3][ghosts_o + 3] == 4633);
    REQUIRE(c_fine[0][ghosts_m + 3][ghosts_n + 3][ghosts_o + 2] == 8700); // left
    REQUIRE(c_fine[1][ghosts_m + 3][ghosts_n + 3][ghosts_o + 2] == 9032);
    REQUIRE(c_fine[0][ghosts_m + 3][ghosts_n + 2][ghosts_o + 3] == 7894); // top
    REQUIRE(c_fine[1][ghosts_m + 3][ghosts_n + 2][ghosts_o + 3] == 8202);
    REQUIRE(c_fine[0][ghosts_m + 2][ghosts_n + 3][ghosts_o + 3] == 3262); // front
    REQUIRE(c_fine[1][ghosts_m + 2][ghosts_n + 3][ghosts_o + 3] == 3474);
    REQUIRE(c_fine[0][ghosts_m + 3][ghosts_n + 2][ghosts_o + 2] == 15336); // top left
    REQUIRE(c_fine[1][ghosts_m + 3][ghosts_n + 2][ghosts_o + 2] == 15936);
    REQUIRE(c_fine[0][ghosts_m + 2][ghosts_n + 3][ghosts_o + 2] == 6072); // front left
    REQUIRE(c_fine[1][ghosts_m + 2][ghosts_n + 3][ghosts_o + 2] == 6480);
    REQUIRE(c_fine[0][ghosts_m + 2][ghosts_n + 2][ghosts_o + 3] == 4460); // front top
    REQUIRE(c_fine[1][ghosts_m + 2][ghosts_n + 2][ghosts_o + 3] == 4820);
    REQUIRE(c_fine[0][ghosts_m + 2][ghosts_n + 2][ghosts_o + 2] == 8016); // front top left
    REQUIRE(c_fine[1][ghosts_m + 2][ghosts_n + 2][ghosts_o + 2] == 8704);
}

// Tests restriction using a blockstencil and checks results against multiple restrictions using scalar stencils, while
// the blockstencil only has entries on its diagonal, i.e. is only affecting one quantity at a time.
TEST_CASE("prolongation_blockstencil_independent_quantities")
{
    int mf = 16;
    int nf = 16;
    int of = 16;
    int ghosts_m = 1;
    int ghosts_n = 1;
    int ghosts_o = 1;
    int mghf = mf + 2 * ghosts_m;
    int nghf = nf + 2 * ghosts_n;
    int oghf = of + 2 * ghosts_o;
    int mc = mf / 2;
    int nc = nf / 2;
    int oc = of / 2;
    int mghc = mc + 2 * ghosts_m;
    int nghc = nc + 2 * ghosts_n;
    int oghc = oc + 2 * ghosts_o;
    int blocksize = 2;

    bool periodic = true;

    auto p1 = std::make_shared<mgcl::Problem>(mf, nf, of);
    p1->setBc(periodic ? mgcl::BC::PERIODIC : mgcl::BC::DIRICHLET);
    mgcl::Level lv_fine1(p1.get(), 0);
    mgcl::Level lv_coarse1(p1.get(), 1);
    auto p2 = std::make_shared<mgcl::Problem>(mf, nf, of);
    p2->setBc(periodic ? mgcl::BC::PERIODIC : mgcl::BC::DIRICHLET);
    mgcl::Level lv_fine2(p2.get(), 0);
    mgcl::Level lv_coarse2(p2.get(), 1);

    mgcl::Cuboid c_fine1(mf, nf, of, ghosts_m, ghosts_n, ghosts_o);
    mgcl::Cuboid c_coarse1(mc, nc, oc, ghosts_m, ghosts_n, ghosts_o);
    mgcl::Cuboid c_fine2(mf, nf, of, ghosts_m, ghosts_n, ghosts_o);
    mgcl::Cuboid c_coarse2(mc, nc, oc, ghosts_m, ghosts_n, ghosts_o);
    mgcl::CuboidBS c_finebs(mf, nf, of, ghosts_m, ghosts_n, ghosts_o, blocksize);
    mgcl::CuboidBS c_coarsebs(c_coarse1.getM(), c_coarse1.getN(), c_coarse1.getO(), ghosts_m, ghosts_n, ghosts_o, blocksize);
    c_coarsebs.fill1dIndex(false);

    // Fill random values for fine cuboids to make sure it won't have an effect
    c_fine1.fillRandom();
    c_fine2.fillRandom();

    // Fill scalar cuboids with values of block cuboid
    for (int i = 0; i < mghc; i++)
        for (int j = 0; j < nghc; j++)
            for (int k = 0; k < oghc; k++)
            {
                c_coarse1[i][j][k] = c_coarsebs[0][i][j][k];
                c_coarse2[i][j][k] = c_coarsebs[1][i][j][k];
            }

    // fill diagonals with full-weighted restriction operator, which is the same as in the scalar case
    mgcl::FixedBlockstencil fbs(3, blocksize);
    for (int b = 0; b < blocksize; b++)
    {
        fbs[b][b][1][1][1] = 1.0;
        fbs[b][b][1][1][0] = 0.5;
        fbs[b][b][1][1][2] = 0.5;
        fbs[b][b][1][0][1] = 0.5;
        fbs[b][b][1][2][1] = 0.5;
        fbs[b][b][0][1][1] = 0.5;
        fbs[b][b][2][1][1] = 0.5;

        fbs[b][b][1][0][0] = 0.25;
        fbs[b][b][1][0][2] = 0.25;
        fbs[b][b][1][2][0] = 0.25;
        fbs[b][b][1][2][2] = 0.25;
        fbs[b][b][0][1][0] = 0.25;
        fbs[b][b][0][1][2] = 0.25;
        fbs[b][b][2][1][0] = 0.25;
        fbs[b][b][2][1][2] = 0.25;
        fbs[b][b][0][0][1] = 0.25;
        fbs[b][b][0][2][1] = 0.25;
        fbs[b][b][2][0][1] = 0.25;
        fbs[b][b][2][2][1] = 0.25;

        fbs[b][b][0][0][0] = 0.125;
        fbs[b][b][0][0][2] = 0.125;
        fbs[b][b][0][2][0] = 0.125;
        fbs[b][b][0][2][2] = 0.125;
        fbs[b][b][2][0][0] = 0.125;
        fbs[b][b][2][0][2] = 0.125;
        fbs[b][b][2][2][0] = 0.125;
        fbs[b][b][2][2][2] = 0.125;
    }

    mgcl::MultigridEngine::prolongateSeq(lv_fine1, lv_coarse1, c_fine1, c_coarse1);
    mgcl::MultigridEngine::prolongateSeq(lv_fine2, lv_coarse2, c_fine2, c_coarse2);

    SECTION("seq")
    {
        mgcl::args::ProlongationBSSeqArgs args{
            c_finebs,
            c_coarsebs,
            fbs,
            true, true, true,
            nullptr, nullptr};

        mgcl::MultigridEngine::prolongateSeqBlockstencil(args);

        // Check both components
        for (int i = ghosts_m; i < mf + ghosts_m; i++)
            for (int j = ghosts_n; j < nf + ghosts_n; j++)
                for (int k = ghosts_o; k < of + ghosts_o; k++)
                {
                    CAPTURE(i, j, k);
                    REQUIRE(c_fine1[i][j][k] == c_finebs[0][i][j][k]);
                    REQUIRE(c_fine2[i][j][k] == c_finebs[1][i][j][k]);
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
        mgcl::args::ProlongationBSOclArgs args{
            d_c_fine, d_c_coarse, d_fbs,
            periodic, true, true,
            nullptr, nullptr, nullptr,
            p_dummy->getProgram(), p_dummy->getCommands(), p_dummy->getContext(),
            nullptr, nullptr,
            &p_dummy->getKernelConfig(), p_dummy->getProfilingData()};

        mgcl::MultigridEngine::prolongateBlockstencil(args);

        p_dummy->finish();
        d_c_fine.read(p_dummy->getCommands(), &c_finebs, true);

        // Check both components
        for (int i = ghosts_m; i < mf + ghosts_m; i++)
            for (int j = ghosts_n; j < nf + ghosts_n; j++)
                for (int k = ghosts_o; k < of + ghosts_o; k++)
                {
                    CAPTURE(i, j, k);
                    REQUIRE(c_fine1[i][j][k] == c_finebs[0][i][j][k]);
                    REQUIRE(c_fine2[i][j][k] == c_finebs[1][i][j][k]);
                }
    }
}
