#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <iostream>
#include <memory>
#include <variant>

#include "../../src/mgcl/blockstencil.hpp"
#include "../../src/mgcl/cuboid_bs.hpp"
#include "../../src/mgcl/level.hpp"
#include "../../src/mgcl/multigrid_engine.hpp"
#include "../cli_args.hpp"
#include "../device_type_generator.hpp"
#include "../test_results.hpp"
#include "../test_utility.hpp"

// Test applying a blockstencil on one point
// TEST_CASE("seq_bs_residual_single_point")
// {
//     int m = 1;
//     int n = 1;
//     int o = 1;
//     int gh = 1;
//     int blocksize = 2;
//     int width = 3;
//     int mgh = m + 2 * gh;
//     int ngh = n + 2 * gh;
//     int ogh = o + 2 * gh;

//     bool periodic = true;

//     mgcl::MGCL_RESIDUAL_NORM resnorm = mgcl::MGCL_L2;

//     mgcl::CuboidBS v(m, n, o, gh, gh, gh, blocksize);
//     mgcl::CuboidBS r(m, n, o, gh, gh, gh, blocksize);
//     mgcl::CuboidBS f(m, n, o, gh, gh, gh, blocksize);
//     mgcl::Blockstencil bs(m, n, o, width, blocksize, gh, gh, gh);

//     v.fill1dIndex(false);
//     f.fill1dIndex(false);
//     bs.fill(0.0);

//     // fill blockstencil for gp 1,1,1 with increasing values starting from 0
//     int cnt = 0;
//     for (int ci = 0; ci < width; ci++)
//         for (int cj = 0; cj < width; cj++)
//             for (int ck = 0; ck < width; ck++)
//                 for (int bi = 0; bi < blocksize; bi++)
//                     for (int bj = 0; bj < blocksize; bj++)
//                     {
//                         bs[bi][bj][ci][cj][ck][1][1][1] = cnt++;
//                     }

//     mgcl::args::ResidualBSSeqArgs args{
//         f,
//         v,
//         r,
//         resnorm,
//         bs,
//         true,
//         periodic,
//         true, 0, 0, 0, nullptr

//     };

//     double res = mgcl::MultigridEngine::residualSeq(args);

//     // calculated by hand
//     REQUIRE(r[1][1][1][0] == -101323);
//     REQUIRE(r[1][1][1][1] == -104184);
// }

// Test applying a blockstencil on multiple grid points with independent quantitites, i.e.
// a blockstencil only has entries on its diagonal.
// The result is compared to two residual separate calculations of the two quantities.
TEST_CASE("bs_jacobi_independent_quantities")
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
    double omega = 0.8;
    int iters = 3;
    int stepsPerIter = 1;

    bool periodic = true;

    mgcl::MGCL_RESIDUAL_NORM resnorm = mgcl::MGCL_L2;

    // TODO test with ghosted bs

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

    mgcl::TBlockstencilInv bs_inv_variant = bs.invertDiagonal();
    REQUIRE(std::holds_alternative<std::shared_ptr<mgcl::CuboidBS>>(bs_inv_variant));
    auto& bs_inv = std::get<std::shared_ptr<mgcl::CuboidBS>>(bs_inv_variant);
    // auto bs_inv_ptr = bs.invertCenterMatrices();
    // REQUIRE(bs_inv_ptr != nullptr);
    // auto& bs_inv = *bs_inv_ptr;

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

    double res1 = mgcl::MultigridEngine::jacobiSeq(v1, f1, r1, omega, 0, iters, resnorm, mgcl::MGCL_VARYING, 0, &sv1, nullptr,
                                                   true, true, true, stepsPerIter);
    double res2 = mgcl::MultigridEngine::jacobiSeq(v2, f2, r2, omega, 0, iters, resnorm, mgcl::MGCL_VARYING, 0, &sv2, nullptr,
                                                   true, true, true, stepsPerIter);

    SECTION("seq")
    {
        mgcl::args::JacobiBSSeqArgs args{
            f,
            v,
            r,
            resnorm,
            bs,
            bs_inv_variant,
            true,
            periodic,
            true, iters, stepsPerIter, omega,
            nullptr};

        double res = mgcl::MultigridEngine::jacobiSeq(args);

        // r.dumpToFile("r.txt");
        // r1.dumpToFile("r1.txt");
    }

    SECTION("ocl")
    {
        // create dummy problem
        auto v_dummy = std::make_shared<mgcl::Cuboid>(1, 1, 1);
        auto f_dummy = std::make_shared<mgcl::Cuboid>(1, 1, 1);
        mgcl::Problem p(1, 1, 1, f_dummy, v_dummy);
        p.setUseOpencl(true);
        p.setProfilingEnabled(true);
        p.getOpenCLHelper().setPreprocessorConstant("BLOCKSIZE", std::to_string(blocksize));
        p.init();

        // bs_inv.dumpToFile("bs_inv.txt");

        mgcl::CuboidBSGpu d_v_in(p.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, v);
        mgcl::CuboidBSGpu d_v_out(p.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, v);
        mgcl::CuboidBSGpu d_r(p.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, r);
        mgcl::CuboidBSGpu d_f(p.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, f);
        mgcl::BlockstencilGpu d_bs(bs, p.getContext(), p.getCommands(), p.getProgram());
        // mgcl::BlockstencilGpu d_bs_inv(*bs_inv, p.getContext(), p.getCommands(), p.getProgram());
        auto d_bs_inv = std::make_shared<mgcl::CuboidBSGpu>(p.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, *bs_inv);
        mgcl::TBlockstencilInv d_bs_inv_variant = d_bs_inv;
        mgcl::CuboidBSGpu dRSquares(p.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, r);

        mgcl::args::JacobiBSOclArgs args{
            d_f,
            d_v_in,
            d_v_out,
            d_r,
            resnorm,
            d_bs,
            d_bs_inv_variant,
            &dRSquares,
            true,
            periodic,
            true, iters, stepsPerIter, omega,
            nullptr, nullptr, nullptr,
            p.getProgram(), p.getCommands(), p.getContext(),
            0, 0, 0, nullptr,
            &p.getKernelConfig(),
            p.getProfilingData()};

        double res = mgcl::MultigridEngine::jacobi(args);

        p.finish();

        d_v_in.read(p.getCommands(), &v, true);
        d_r.read(p.getCommands(), &r, true);

        // r.dumpToFile("r.txt");
        // r1.dumpToFile("r1.txt");
    }

    // Check both r and v components
    for (int i = gh; i < m + gh; i++)
        for (int j = gh; j < n + gh; j++)
            for (int k = gh; k < o + gh; k++)
            {
                CAPTURE(i, j, k);
                REQUIRE_THAT(r1[i][j][k], Catch::Matchers::WithinAbs(r[i][j][k][0], 1e-4));
                REQUIRE_THAT(r2[i][j][k], Catch::Matchers::WithinAbs(r[i][j][k][1], 1e-4));
                REQUIRE_THAT(v1[i][j][k], Catch::Matchers::WithinAbs(v[i][j][k][0], 1e-4));
                REQUIRE_THAT(v2[i][j][k], Catch::Matchers::WithinAbs(v[i][j][k][1], 1e-4));
            }
}

// We can construct a vectorial problem by combining two scalar unknowns into one vector and carefully construct
// the according blockstencil. The result must be the same as if we would solve the fine scalar problems.
// We use a fixed stencil which has the same coefficients for all grid points to make things easier.
// See Notizen/2025-05-14_blockstencil_tests_scalar_to_block.ods.
TEST_CASE("bs_jacobi_combined_scalars")
{
    int mf = 8;
    int nf = 8;
    int of = 8;
    int mc = mf / 2;
    int nc = nf / 2;
    int oc = of / 2;
    int gh = 1;
    int blocksize = 8;
    int width = 3;
    int mfgh = mf + 2 * gh;
    int nfgh = nf + 2 * gh;
    int ofgh = of + 2 * gh;
    int mcgh = mc + 2 * gh;
    int ncgh = nc + 2 * gh;
    int ocgh = oc + 2 * gh;
    double omega = 0.8;
    int iters = 1;
    int stepsPerIter = 1;

    bool periodic = true;

    mgcl::MGCL_RESIDUAL_NORM resnorm = mgcl::MGCL_L2;

    // TODO test with ghosted bs

    mgcl::CuboidBS v(mc, nc, oc, gh, gh, gh, blocksize);
    mgcl::CuboidBS r(mc, nc, oc, gh, gh, gh, blocksize);
    mgcl::CuboidBS f(mc, nc, oc, gh, gh, gh, blocksize);
    mgcl::Blockstencil bs(mc, nc, oc, width, blocksize, 0, 0, 0);

    mgcl::Cuboid v1(mf, nf, of, gh, gh, gh);
    mgcl::Cuboid f1(mf, nf, of, gh, gh, gh);
    mgcl::Cuboid r1(mf, nf, of, gh, gh, gh);
    // mgcl::VaryingStencil sv1(mf, nf, of, width, 0, 0, 0);
    // mgcl::VaryingStencil sv2(mf, nf, of, width, 0, 0, 0);
    mgcl::FixedStencil fs1(width);

    // v1.fill1dIndex(false);
    v1.fillRandom();
    f1.fill1dIndex(false);

    // // fill same coefficients for sv1 and sv2
    // // clang-format off
    // int cnt = 0;
    // for (int i = 0; i < mf; i++)
    // for (int j = 0; j < nf; j++)
    // for (int k = 0; k < of; k++)
    //     for (int ii = 0; ii < width; ii++)
    //     for (int jj = 0; jj < width; jj++)
    //     for (int kk = 0; kk < width; kk++)
    //     {
    //         sv1[ii][jj][kk][i][j][k] = cnt++;
    //         sv2[ii][jj][kk][i][j][k] = cnt;
    //     }
    // int cnt = 0;
    // for (int ii = 0; ii < width; ii++)
    //     for (int jj = 0; jj < width; jj++)
    //         for (int kk = 0; kk < width; kk++)
    //         {
    //             fs1[ii][jj][kk] = cnt;
    //         }
    mgcl_test::fill7pLaplace(fs1, 1.0 / (double)mf, false);
    double ftl = fs1[0][0][0];
    double ftc = fs1[0][0][1];
    double ftr = fs1[0][0][2];
    double fcl = fs1[0][1][0];
    double fcc = fs1[0][1][1];
    double fcr = fs1[0][1][2];
    double fbl = fs1[0][2][0];
    double fbc = fs1[0][2][1];
    double fbr = fs1[0][2][2];
    double ctl = fs1[1][0][0];
    double ctc = fs1[1][0][1];
    double ctr = fs1[1][0][2];
    double ccl = fs1[1][1][0];
    double ccc = fs1[1][1][1];
    double ccr = fs1[1][1][2];
    double cbl = fs1[1][2][0];
    double cbc = fs1[1][2][1];
    double cbr = fs1[1][2][2];
    double btl = fs1[2][0][0];
    double btc = fs1[2][0][1];
    double btr = fs1[2][0][2];
    double bcl = fs1[2][1][0];
    double bcc = fs1[2][1][1];
    double bcr = fs1[2][1][2];
    double bbl = fs1[2][2][0];
    double bbc = fs1[2][2][1];
    double bbr = fs1[2][2][2];

    // fill blockstencil with values from sv1 and sv2
    for (int i = 0; i < mc; i++)
        for (int j = 0; j < nc; j++)
            for (int k = 0; k < oc; k++)
            {
                // ***** front *****
                // b_ftl
                bs[0][7][0][0][0][i][j][k] = ftl;

                // b_ftc
                bs[0][6][0][0][1][i][j][k] = ftc;
                bs[0][7][0][0][1][i][j][k] = ftr;
                bs[1][6][0][0][1][i][j][k] = ftl;
                bs[1][7][0][0][1][i][j][k] = ftc;

                // b_ftr
                bs[1][6][0][0][2][i][j][k] = ftr;

                // b_fcl
                bs[0][5][0][1][0][i][j][k] = fcl;
                bs[0][7][0][1][0][i][j][k] = fbl;
                bs[2][5][0][1][0][i][j][k] = ftl;
                bs[2][7][0][1][0][i][j][k] = fcl;

                // b_fcc
                bs[0][4][0][1][1][i][j][k] = fcc;
                bs[0][5][0][1][1][i][j][k] = fcr;
                bs[0][6][0][1][1][i][j][k] = fbc;
                bs[0][7][0][1][1][i][j][k] = fbr;
                bs[1][4][0][1][1][i][j][k] = fcl;
                bs[1][5][0][1][1][i][j][k] = fcc;
                bs[1][6][0][1][1][i][j][k] = fbl;
                bs[1][7][0][1][1][i][j][k] = fbc;
                bs[2][4][0][1][1][i][j][k] = ftc;
                bs[2][5][0][1][1][i][j][k] = ftr;
                bs[2][6][0][1][1][i][j][k] = fcc;
                bs[2][7][0][1][1][i][j][k] = fcr;
                bs[3][4][0][1][1][i][j][k] = ftl;
                bs[3][5][0][1][1][i][j][k] = ftc;
                bs[3][6][0][1][1][i][j][k] = fcl;
                bs[3][7][0][1][1][i][j][k] = fcc;

                // b_fcr
                bs[1][4][0][1][2][i][j][k] = fcr;
                bs[1][6][0][1][2][i][j][k] = fbr;
                bs[3][4][0][1][2][i][j][k] = ftr;
                bs[3][6][0][1][2][i][j][k] = fcr;

                // b_fbl
                bs[2][5][0][2][0][i][j][k] = fbl;

                // b_fbc
                bs[2][4][0][2][1][i][j][k] = fbc;
                bs[2][5][0][2][1][i][j][k] = fbr;
                bs[3][4][0][2][1][i][j][k] = fbl;
                bs[3][5][0][2][1][i][j][k] = fbc;

                // b_fbr
                bs[3][4][0][2][2][i][j][k] = fbr;

                // ***** center *****
                // b_ctl
                bs[0][3][1][0][0][i][j][k] = ctl;
                bs[0][7][1][0][0][i][j][k] = btl;
                bs[4][3][1][0][0][i][j][k] = ftl;
                bs[4][7][1][0][0][i][j][k] = ctl;

                // b_ctc
                bs[0][2][1][0][1][i][j][k] = ctc;
                bs[0][3][1][0][1][i][j][k] = ctr;
                bs[0][6][1][0][1][i][j][k] = btc;
                bs[0][7][1][0][1][i][j][k] = btr;
                bs[1][2][1][0][1][i][j][k] = ctl;
                bs[1][3][1][0][1][i][j][k] = ctc;
                bs[1][6][1][0][1][i][j][k] = btl;
                bs[1][7][1][0][1][i][j][k] = btc;
                bs[4][2][1][0][1][i][j][k] = ftc;
                bs[4][3][1][0][1][i][j][k] = ftr;
                bs[4][6][1][0][1][i][j][k] = ctc;
                bs[4][7][1][0][1][i][j][k] = ctr;
                bs[5][2][1][0][1][i][j][k] = ftl;
                bs[5][3][1][0][1][i][j][k] = ftc;
                bs[5][6][1][0][1][i][j][k] = ctl;
                bs[5][7][1][0][1][i][j][k] = ctc;

                // b_ctr
                bs[1][2][1][0][2][i][j][k] = ctr;
                bs[1][6][1][0][2][i][j][k] = btr;
                bs[5][2][1][0][2][i][j][k] = ftr;
                bs[5][6][1][0][2][i][j][k] = ctr;

                // b_ccl
                bs[0][1][1][1][0][i][j][k] = ccl;
                bs[0][3][1][1][0][i][j][k] = cbl;
                bs[0][5][1][1][0][i][j][k] = bcl;
                bs[0][7][1][1][0][i][j][k] = bbl;
                bs[2][1][1][1][0][i][j][k] = ctl;
                bs[2][3][1][1][0][i][j][k] = ccl;
                bs[2][5][1][1][0][i][j][k] = btl;
                bs[2][7][1][1][0][i][j][k] = bcl;
                bs[4][1][1][1][0][i][j][k] = fcl;
                bs[4][3][1][1][0][i][j][k] = fbl;
                bs[4][5][1][1][0][i][j][k] = ccl;
                bs[4][7][1][1][0][i][j][k] = cbl;
                bs[6][1][1][1][0][i][j][k] = ftl;
                bs[6][3][1][1][0][i][j][k] = fcl;
                bs[6][5][1][1][0][i][j][k] = ctl;
                bs[6][7][1][1][0][i][j][k] = ccl;

                // b_ccc
                bs[0][0][1][1][1][i][j][k] = ccc;
                bs[0][1][1][1][1][i][j][k] = ccr;
                bs[0][2][1][1][1][i][j][k] = cbc;
                bs[0][3][1][1][1][i][j][k] = cbr;
                bs[0][4][1][1][1][i][j][k] = bcc;
                bs[0][5][1][1][1][i][j][k] = bcr;
                bs[0][6][1][1][1][i][j][k] = bbc;
                bs[0][7][1][1][1][i][j][k] = bbr;
                bs[1][0][1][1][1][i][j][k] = ccl;
                bs[1][1][1][1][1][i][j][k] = ccc;
                bs[1][2][1][1][1][i][j][k] = cbl;
                bs[1][3][1][1][1][i][j][k] = cbc;
                bs[1][4][1][1][1][i][j][k] = bcl;
                bs[1][5][1][1][1][i][j][k] = bcc;
                bs[1][6][1][1][1][i][j][k] = bbl;
                bs[1][7][1][1][1][i][j][k] = bbc;
                bs[2][0][1][1][1][i][j][k] = ctc;
                bs[2][1][1][1][1][i][j][k] = ctr;
                bs[2][2][1][1][1][i][j][k] = ccc;
                bs[2][3][1][1][1][i][j][k] = ccr;
                bs[2][4][1][1][1][i][j][k] = btc;
                bs[2][5][1][1][1][i][j][k] = btr;
                bs[2][6][1][1][1][i][j][k] = bcc;
                bs[2][7][1][1][1][i][j][k] = bcr;
                bs[3][0][1][1][1][i][j][k] = ctl;
                bs[3][1][1][1][1][i][j][k] = ctc;
                bs[3][2][1][1][1][i][j][k] = ccl;
                bs[3][3][1][1][1][i][j][k] = ccc;
                bs[3][4][1][1][1][i][j][k] = btl;
                bs[3][5][1][1][1][i][j][k] = btc;
                bs[3][6][1][1][1][i][j][k] = bcl;
                bs[3][7][1][1][1][i][j][k] = bcc;
                bs[4][0][1][1][1][i][j][k] = fcc;
                bs[4][1][1][1][1][i][j][k] = fcr;
                bs[4][2][1][1][1][i][j][k] = fbc;
                bs[4][3][1][1][1][i][j][k] = fbr;
                bs[4][4][1][1][1][i][j][k] = ccc;
                bs[4][5][1][1][1][i][j][k] = ccr;
                bs[4][6][1][1][1][i][j][k] = cbc;
                bs[4][7][1][1][1][i][j][k] = cbr;
                bs[5][0][1][1][1][i][j][k] = fcl;
                bs[5][1][1][1][1][i][j][k] = fcc;
                bs[5][2][1][1][1][i][j][k] = fbl;
                bs[5][3][1][1][1][i][j][k] = fbc;
                bs[5][4][1][1][1][i][j][k] = ccl;
                bs[5][5][1][1][1][i][j][k] = ccc;
                bs[5][6][1][1][1][i][j][k] = cbl;
                bs[5][7][1][1][1][i][j][k] = cbc;
                bs[6][0][1][1][1][i][j][k] = ftc;
                bs[6][1][1][1][1][i][j][k] = ftr;
                bs[6][2][1][1][1][i][j][k] = fcc;
                bs[6][3][1][1][1][i][j][k] = fcr;
                bs[6][4][1][1][1][i][j][k] = ctc;
                bs[6][5][1][1][1][i][j][k] = ctr;
                bs[6][6][1][1][1][i][j][k] = ccc;
                bs[6][7][1][1][1][i][j][k] = ccr;
                bs[7][0][1][1][1][i][j][k] = ftl;
                bs[7][1][1][1][1][i][j][k] = ftc;
                bs[7][2][1][1][1][i][j][k] = fcl;
                bs[7][3][1][1][1][i][j][k] = fcc;
                bs[7][4][1][1][1][i][j][k] = ctl;
                bs[7][5][1][1][1][i][j][k] = ctc;
                bs[7][6][1][1][1][i][j][k] = ccl;
                bs[7][7][1][1][1][i][j][k] = ccc;

                // b_ccr
                bs[1][0][1][1][2][i][j][k] = ccr;
                bs[3][0][1][1][2][i][j][k] = ctr;
                bs[5][0][1][1][2][i][j][k] = fcr;
                bs[7][0][1][1][2][i][j][k] = ftr;
                bs[1][2][1][1][2][i][j][k] = cbr;
                bs[3][2][1][1][2][i][j][k] = ccr;
                bs[5][2][1][1][2][i][j][k] = fbr;
                bs[7][2][1][1][2][i][j][k] = fcr;
                bs[1][4][1][1][2][i][j][k] = bcr;
                bs[3][4][1][1][2][i][j][k] = btr;
                bs[5][4][1][1][2][i][j][k] = ccr;
                bs[7][4][1][1][2][i][j][k] = ctr;
                bs[1][6][1][1][2][i][j][k] = bbr;
                bs[3][6][1][1][2][i][j][k] = bcr;
                bs[5][6][1][1][2][i][j][k] = cbr;
                bs[7][6][1][1][2][i][j][k] = ccr;

                // b_cbl
                bs[2][1][1][2][0][i][j][k] = cbl;
                bs[2][5][1][2][0][i][j][k] = bbl;
                bs[6][1][1][2][0][i][j][k] = fbl;
                bs[6][5][1][2][0][i][j][k] = cbl;

                // b_cbc
                bs[2][0][1][2][1][i][j][k] = cbc;
                bs[2][1][1][2][1][i][j][k] = cbr;
                bs[2][4][1][2][1][i][j][k] = bbc;
                bs[2][5][1][2][1][i][j][k] = bbr;
                bs[3][0][1][2][1][i][j][k] = cbr;
                bs[3][1][1][2][1][i][j][k] = cbc;
                bs[3][4][1][2][1][i][j][k] = bbr;
                bs[3][5][1][2][1][i][j][k] = bbc;
                bs[6][0][1][2][1][i][j][k] = fbc;
                bs[6][1][1][2][1][i][j][k] = fbr;
                bs[6][4][1][2][1][i][j][k] = cbc;
                bs[6][5][1][2][1][i][j][k] = cbr;
                bs[7][0][1][2][1][i][j][k] = fbr;
                bs[7][1][1][2][1][i][j][k] = fbc;
                bs[7][4][1][2][1][i][j][k] = cbr;
                bs[7][5][1][2][1][i][j][k] = cbc;

                // b_cbr
                bs[3][0][1][2][2][i][j][k] = cbr;
                bs[3][4][1][2][2][i][j][k] = bbr;
                bs[7][0][1][2][2][i][j][k] = fbr;
                bs[7][4][1][2][2][i][j][k] = cbr;

                // **** back ****
                // b_btl
                bs[4][3][2][0][0][i][j][k] = btl;

                // b_btc
                bs[4][2][2][0][1][i][j][k] = btc;
                bs[4][3][2][0][1][i][j][k] = btr;
                bs[5][2][2][0][1][i][j][k] = btl;
                bs[5][3][2][0][1][i][j][k] = btc;

                // b_btr
                bs[5][2][2][0][2][i][j][k] = btr;

                // b_bcl
                bs[4][1][2][1][0][i][j][k] = bcl;
                bs[4][3][2][1][0][i][j][k] = bbl;
                bs[6][1][2][1][0][i][j][k] = btl;
                bs[6][3][2][1][0][i][j][k] = bcl;

                // b_bcc
                bs[4][0][2][1][1][i][j][k] = bcc;
                bs[4][1][2][1][1][i][j][k] = bcr;
                bs[4][2][2][1][1][i][j][k] = bbc;
                bs[4][3][2][1][1][i][j][k] = bbr;
                bs[5][0][2][1][1][i][j][k] = bcl;
                bs[5][1][2][1][1][i][j][k] = bcc;
                bs[5][2][2][1][1][i][j][k] = bbl;
                bs[5][3][2][1][1][i][j][k] = bbc;
                bs[6][0][2][1][1][i][j][k] = btc;
                bs[6][1][2][1][1][i][j][k] = btr;
                bs[6][2][2][1][1][i][j][k] = bcc;
                bs[6][3][2][1][1][i][j][k] = bcr;
                bs[7][0][2][1][1][i][j][k] = btl;
                bs[7][1][2][1][1][i][j][k] = btc;
                bs[7][2][2][1][1][i][j][k] = bcl;
                bs[7][3][2][1][1][i][j][k] = bcc;

                // b_bcr
                bs[5][0][2][1][2][i][j][k] = bcr;
                bs[5][2][2][1][2][i][j][k] = bbr;
                bs[7][0][2][1][2][i][j][k] = btr;
                bs[7][2][2][1][2][i][j][k] = bcr;

                // b_bbr
                bs[6][1][2][2][0][i][j][k] = bbl;

                // b_bbc
                bs[6][0][2][2][1][i][j][k] = bbc;
                bs[6][1][2][2][1][i][j][k] = bbr;
                bs[7][0][2][2][1][i][j][k] = bbl;
                bs[7][1][2][2][1][i][j][k] = bbc;

                // b_bbr
                bs[7][0][2][2][2][i][j][k] = bbr;
            }

    mgcl::TBlockstencilInv bs_inv_variant = bs.invertDiagonal();
    REQUIRE(std::holds_alternative<std::shared_ptr<mgcl::CuboidBS>>(bs_inv_variant));
    auto& bs_inv = std::get<std::shared_ptr<mgcl::CuboidBS>>(bs_inv_variant);

    // bs_inv.dumpToFile("bs_inv.txt");
    // bs.dumpToFile("bs.txt");

    // fill v with values of v1 and v2, vice versa for f
    for (int i = gh, i2 = gh; i < mc + gh; i++, i2 += 2)
        for (int j = gh, j2 = gh; j < nc + gh; j++, j2 += 2)
            for (int k = gh, k2 = gh; k < oc + gh; k++, k2 += 2)
            {
                v[i][j][k][0] = v1[i2][j2][k2];
                v[i][j][k][1] = v1[i2][j2][k2 + 1];
                v[i][j][k][2] = v1[i2][j2 + 1][k2];
                v[i][j][k][3] = v1[i2][j2 + 1][k2 + 1];
                v[i][j][k][4] = v1[i2 + 1][j2][k2];
                v[i][j][k][5] = v1[i2 + 1][j2][k2 + 1];
                v[i][j][k][6] = v1[i2 + 1][j2 + 1][k2];
                v[i][j][k][7] = v1[i2 + 1][j2 + 1][k2 + 1];
                f[i][j][k][0] = f1[i2][j2][k2];
                f[i][j][k][1] = f1[i2][j2][k2 + 1];
                f[i][j][k][2] = f1[i2][j2 + 1][k2];
                f[i][j][k][3] = f1[i2][j2 + 1][k2 + 1];
                f[i][j][k][4] = f1[i2 + 1][j2][k2];
                f[i][j][k][5] = f1[i2 + 1][j2][k2 + 1];
                f[i][j][k][6] = f1[i2 + 1][j2 + 1][k2];
                f[i][j][k][7] = f1[i2 + 1][j2 + 1][k2 + 1];
            }

    v.updateGhosts(nullptr, true);
    mgcl::MultigridEngine::updateGhostsSeq(v1, nullptr, true, false);
    f.updateGhosts(nullptr, true);
    mgcl::MultigridEngine::updateGhostsSeq(f1, nullptr, true, false);

    // v.dumpToFile("v.txt");
    // v1.dumpToFile("v1.txt");

    double res1 = mgcl::MultigridEngine::jacobiSeq(v1, f1, r1, omega, 0, iters, resnorm, mgcl::MGCL_FIXED, 0, nullptr, &fs1,
                                                   true, true, true, stepsPerIter);

    SECTION("seq")
    {
        mgcl::args::JacobiBSSeqArgs args{
            f,
            v,
            r,
            resnorm,
            bs,
            bs_inv_variant,
            true,
            periodic,
            true, iters, stepsPerIter, omega,
            nullptr};

        double res = mgcl::MultigridEngine::jacobiSeq(args);

        // r.dumpToFile("r.txt");
        // r1.dumpToFile("r1.txt");
    }

    SECTION("ocl")
    {
        // create dummy problem
        auto v_dummy = std::make_shared<mgcl::Cuboid>(1, 1, 1);
        auto f_dummy = std::make_shared<mgcl::Cuboid>(1, 1, 1);
        mgcl::Problem p(1, 1, 1, f_dummy, v_dummy);
        p.setUseOpencl(true);
        p.setProfilingEnabled(true);
        p.getOpenCLHelper().setPreprocessorConstant("BLOCKSIZE", std::to_string(blocksize));
        p.init();

        // bs_inv.dumpToFile("bs_inv.txt");

        mgcl::CuboidBSGpu d_v_in(p.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, v);
        mgcl::CuboidBSGpu d_v_out(p.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, v);
        mgcl::CuboidBSGpu d_r(p.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, r);
        mgcl::CuboidBSGpu d_f(p.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, f);
        mgcl::BlockstencilGpu d_bs(bs, p.getContext(), p.getCommands(), p.getProgram());
        auto d_bs_inv = std::make_shared<mgcl::CuboidBSGpu>(p.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, *bs_inv);
        mgcl::TBlockstencilInv d_bs_inv_variant = d_bs_inv;
        mgcl::CuboidBSGpu dRSquares(p.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, r);

        mgcl::args::JacobiBSOclArgs args{
            d_f,
            d_v_in,
            d_v_out,
            d_r,
            resnorm,
            d_bs,
            d_bs_inv_variant,
            &dRSquares,
            true,
            periodic,
            true, iters, stepsPerIter, omega,
            nullptr, nullptr, nullptr,
            p.getProgram(), p.getCommands(), p.getContext(),
            0, 0, 0, nullptr,
            &p.getKernelConfig(),
            p.getProfilingData()};

        double res = mgcl::MultigridEngine::jacobi(args);

        p.finish();

        d_v_in.read(p.getCommands(), &v, true);
        d_r.read(p.getCommands(), &r, true);

        // r.dumpToFile("r.txt");
        // r1.dumpToFile("r1.txt");
    }

    // Check both r and v components
    for (int i = gh, i2 = gh; i < mc + gh; i++, i2 += 2)
        for (int j = gh, j2 = gh; j < nc + gh; j++, j2 += 2)
            for (int k = gh, k2 = gh; k < oc + gh; k++, k2 += 2)
            {
                CAPTURE(i, j, k, i2, j2, k2);
                REQUIRE_THAT(v[i][j][k][0], Catch::Matchers::WithinAbs(v1[i2][j2][k2], 1e-4));
                REQUIRE_THAT(v[i][j][k][1], Catch::Matchers::WithinAbs(v1[i2][j2][k2 + 1], 1e-4));
                REQUIRE_THAT(v[i][j][k][2], Catch::Matchers::WithinAbs(v1[i2][j2 + 1][k2], 1e-4));
                REQUIRE_THAT(v[i][j][k][3], Catch::Matchers::WithinAbs(v1[i2][j2 + 1][k2 + 1], 1e-4));
                REQUIRE_THAT(v[i][j][k][4], Catch::Matchers::WithinAbs(v1[i2 + 1][j2][k2], 1e-4));
                REQUIRE_THAT(v[i][j][k][5], Catch::Matchers::WithinAbs(v1[i2 + 1][j2][k2 + 1], 1e-4));
                REQUIRE_THAT(v[i][j][k][6], Catch::Matchers::WithinAbs(v1[i2 + 1][j2 + 1][k2], 1e-4));
                REQUIRE_THAT(v[i][j][k][7], Catch::Matchers::WithinAbs(v1[i2 + 1][j2 + 1][k2 + 1], 1e-4));
                REQUIRE_THAT(r[i][j][k][0], Catch::Matchers::WithinAbs(r1[i2][j2][k2], 1e-4));
                REQUIRE_THAT(r[i][j][k][1], Catch::Matchers::WithinAbs(r1[i2][j2][k2 + 1], 1e-4));
                REQUIRE_THAT(r[i][j][k][2], Catch::Matchers::WithinAbs(r1[i2][j2 + 1][k2], 1e-4));
                REQUIRE_THAT(r[i][j][k][3], Catch::Matchers::WithinAbs(r1[i2][j2 + 1][k2 + 1], 1e-4));
                REQUIRE_THAT(r[i][j][k][4], Catch::Matchers::WithinAbs(r1[i2 + 1][j2][k2], 1e-4));
                REQUIRE_THAT(r[i][j][k][5], Catch::Matchers::WithinAbs(r1[i2 + 1][j2][k2 + 1], 1e-4));
                REQUIRE_THAT(r[i][j][k][6], Catch::Matchers::WithinAbs(r1[i2 + 1][j2 + 1][k2], 1e-4));
                REQUIRE_THAT(r[i][j][k][7], Catch::Matchers::WithinAbs(r1[i2 + 1][j2 + 1][k2 + 1], 1e-4));
            }
}

// // Test applying a blockstencil on one point
// TEST_CASE("ocl_bs_residual_single_point")
// {
//     int m = 1;
//     int n = 1;
//     int o = 1;
//     int gh = 1;
//     int blocksize = 2;
//     int width = 3;
//     int mgh = m + 2 * gh;
//     int ngh = n + 2 * gh;
//     int ogh = o + 2 * gh;

//     bool periodic = true;

//     mgcl::MGCL_RESIDUAL_NORM resnorm = mgcl::MGCL_L2;

//     mgcl::CuboidBS v(m, n, o, gh, gh, gh, blocksize);
//     mgcl::CuboidBS r(m, n, o, gh, gh, gh, blocksize);
//     mgcl::CuboidBS f(m, n, o, gh, gh, gh, blocksize);
//     mgcl::Blockstencil bs(m, n, o, width, blocksize, gh, gh, gh);

//     v.fill1dIndex(false);
//     f.fill1dIndex(false);
//     bs.fill(0.0);

//     // fill blockstencil for gp 1,1,1 with increasing values starting from 0
//     int cnt = 0;
//     for (int ci = 0; ci < width; ci++)
//         for (int cj = 0; cj < width; cj++)
//             for (int ck = 0; ck < width; ck++)
//                 for (int bi = 0; bi < blocksize; bi++)
//                     for (int bj = 0; bj < blocksize; bj++)
//                     {
//                         bs[bi][bj][ci][cj][ck][1][1][1] = cnt++;
//                     }

//     // create dummy problem
//     auto v_dummy = std::make_shared<mgcl::Cuboid>(1, 1, 1);
//     auto f_dummy = std::make_shared<mgcl::Cuboid>(1, 1, 1);
//     mgcl::Problem p(1, 1, 1, f_dummy, v_dummy);
//     p.setUseOpencl(true);
//     p.setProfilingEnabled(true);
//     p.init();

//     mgcl::CuboidBSGpu d_v(p.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, v);
//     mgcl::CuboidBSGpu d_f(p.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, f);
//     mgcl::CuboidBSGpu d_r(p.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, r);
//     mgcl::BlockstencilGpu d_bs(bs, p.getContext(), p.getCommands(), p.getProgram());
//     mgcl::CuboidBSGpu dRSquares(p.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, r);
//     // mgcl::BufferGpu dPlanesBuf(p.getContext(), CL_MEM_READ_WRITE, 2 * gh * (mgh * ngh + mgh * ogh + ngh * ogh));
//     // std::vector<double> sendBuf(2 * gh * (mgh * ngh + mgh * ogh + ngh * ogh));
//     // std::vector<double> recvBuf(2 * gh * (mgh * ngh + mgh * ogh + ngh * ogh));

//     mgcl::args::ResidualBSOclArgs args{
//         d_f,
//         d_v,
//         d_r,
//         resnorm,
//         d_bs,
//         &dRSquares,
//         true,
//         periodic,
//         true,
//         // &dPlanesBuf,
//         // &sendBuf,
//         // &recvBuf,
//         nullptr,
//         nullptr,
//         nullptr,
//         p.getProgram(),
//         p.getCommands(),
//         p.getContext(),
//         0, 0, 0, nullptr,
//         &p.getKernelConfig(),
//         p.getProfilingData()};

//     // d_v.dumpToFile(p.getCommands(), "v.txt", false);
//     // d_f.dumpToFile(p.getCommands(), "f.txt", false);

//     double res = mgcl::MultigridEngine::residual(args);
//     p.finish();

//     // d_r.dumpToFile(p.getCommands(), "r.txt", false);
//     // d_bs.dumpToFile(p.getCommands(), "bs.txt", false);

//     d_r.read(p.getCommands(), &r, true);

//     // calculated by hand
//     REQUIRE(r[1][1][1][0] == -101323);
//     REQUIRE(r[1][1][1][1] == -104184);
// }

// // Test applying a blockstencil on multiple grid points with independent quantitites, i.e.
// // a blockstencil only has entries on its diagonal.
// // The result is compared to two residual separate calculations of the two quantities.
// TEST_CASE("ocl_bs_residual_independent_quantities")
// {
//     int m = 4;
//     int n = 4;
//     int o = 4;
//     int gh = 1;
//     int blocksize = 2;
//     int width = 3;
//     int mgh = m + 2 * gh;
//     int ngh = n + 2 * gh;
//     int ogh = o + 2 * gh;

//     bool periodic = true;

//     mgcl::MGCL_RESIDUAL_NORM resnorm = mgcl::MGCL_L2;

//     mgcl::CuboidBS v(m, n, o, gh, gh, gh, blocksize);
//     mgcl::CuboidBS r(m, n, o, gh, gh, gh, blocksize);
//     mgcl::CuboidBS f(m, n, o, gh, gh, gh, blocksize);
//     mgcl::Blockstencil bs(m, n, o, width, blocksize, gh, gh, gh);

//     auto v1_ptr = std::make_shared<mgcl::Cuboid>(m, n, o, gh, gh, gh);
//     auto f1_ptr = std::make_shared<mgcl::Cuboid>(m, n, o, gh, gh, gh);
//     auto& v1 = *v1_ptr;
//     auto& f1 = *f1_ptr;
//     mgcl::Cuboid r1(m, n, o, gh, gh, gh);
//     auto v2_ptr = std::make_shared<mgcl::Cuboid>(m, n, o, gh, gh, gh);
//     auto f2_ptr = std::make_shared<mgcl::Cuboid>(m, n, o, gh, gh, gh);
//     auto& v2 = *v2_ptr;
//     auto& f2 = *f2_ptr;
//     mgcl::Cuboid r2(m, n, o, gh, gh, gh);

//     mgcl::Problem p1(m, n, o, f1_ptr, v1_ptr);
//     p1.setUseOpencl(true);
//     p1.setProfilingEnabled(true);
//     p1.setGhostsIn(gh);
//     p1.setStencilType(mgcl::MGCL_VARYING);

//     mgcl::Problem p2(m, n, o, f2_ptr, v2_ptr);
//     p2.setUseOpencl(true);
//     p2.setProfilingEnabled(true);
//     p2.setGhostsIn(gh);
//     p2.setStencilType(mgcl::MGCL_VARYING);

//     auto& sv1_ptr = p1.getStencilValues();
//     auto& sv2_ptr = p2.getStencilValues();
//     auto& sv1 = *sv1_ptr;
//     auto& sv2 = *sv2_ptr;

//     v.fill1dIndex(false);
//     f.fill1dIndex(false);

//     // fill different coefficients for v1 and v2
//     // clang-format off
//     int cnt = 0;
//     for (int i = sv1.getGhostsM(); i < sv1.getGhostsM() + m; i++)
//     for (int j = sv1.getGhostsN(); j < sv1.getGhostsN() + n; j++)
//     for (int k = sv1.getGhostsO(); k < sv1.getGhostsO() + o; k++)
//         for (int ii = 0; ii < width; ii++)
//         for (int jj = 0; jj < width; jj++)
//         for (int kk = 0; kk < width; kk++)
//         {
//             sv1[ii][jj][kk][i][j][k] = cnt++;
//             sv2[ii][jj][kk][i][j][k] = -cnt;
//         }
//     // clang-format on

//     p1.init();
//     auto& lv01 = p1.getLevelAt(0);
//     p2.init();
//     auto& lv02 = p2.getLevelAt(0);

//     // fill blockstencil diagonal with values from sv1 and sv2
//     for (int ci = 0; ci < width; ci++)
//         for (int cj = 0; cj < width; cj++)
//             for (int ck = 0; ck < width; ck++)
//                 for (int i = gh; i < m + gh; i++)
//                     for (int j = gh; j < n + gh; j++)
//                         for (int k = gh; k < o + gh; k++)
//                         {
//                             bs[0][0][ci][cj][ck][i][j][k] = sv1[ci][cj][ck][i][j][k];
//                             bs[1][1][ci][cj][ck][i][j][k] = sv2[ci][cj][ck][i][j][k];
//                             bs[0][1][ci][cj][ck][i][j][k] = 0;
//                             bs[1][0][ci][cj][ck][i][j][k] = 0;
//                         }

//     // fill v1 and v2 with values of v, vice versa for f
//     for (int i = 0; i < mgh; i++)
//         for (int j = 0; j < ngh; j++)
//             for (int k = 0; k < ogh; k++)
//             {
//                 v1[i][j][k] = v[i][j][k][0];
//                 v2[i][j][k] = v[i][j][k][1];
//                 f1[i][j][k] = f[i][j][k][0];
//                 f2[i][j][k] = f[i][j][k][1];
//             }

//     v.updateGhosts(nullptr, true);
//     mgcl::MultigridEngine::updateGhostsSeq(v1, nullptr, true, false);
//     mgcl::MultigridEngine::updateGhostsSeq(v2, nullptr, true, false);

//     // create dummy problem
//     auto v_dummy = std::make_shared<mgcl::Cuboid>(1, 1, 1);
//     auto f_dummy = std::make_shared<mgcl::Cuboid>(1, 1, 1);
//     mgcl::Problem p(1, 1, 1, f_dummy, v_dummy);
//     p.setUseOpencl(true);
//     p.setProfilingEnabled(true);
//     p.init();

//     // ********** Blockstencil args ***********
//     mgcl::CuboidBSGpu d_v(p.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, v);
//     mgcl::CuboidBSGpu d_f(p.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, f);
//     mgcl::CuboidBSGpu d_r(p.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, r);
//     mgcl::BlockstencilGpu d_bs(bs, p.getContext(), p.getCommands(), p.getProgram());
//     mgcl::CuboidBSGpu dRSquares(p.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, r);
//     // mgcl::BufferGpu dPlanesBuf(p.getContext(), CL_MEM_READ_WRITE, 2 * gh * (mgh * ngh + mgh * ogh + ngh * ogh));
//     // std::vector<double> sendBuf(2 * gh * (mgh * ngh + mgh * ogh + ngh * ogh));
//     // std::vector<double> recvBuf(2 * gh * (mgh * ngh + mgh * ogh + ngh * ogh));

//     mgcl::args::ResidualBSOclArgs args{
//         d_f,
//         d_v,
//         d_r,
//         resnorm,
//         d_bs,
//         &dRSquares,
//         true,
//         periodic,
//         true,
//         // &dPlanesBuf,
//         // &sendBuf,
//         // &recvBuf,
//         nullptr,
//         nullptr,
//         nullptr,
//         p.getProgram(),
//         p.getCommands(),
//         p.getContext(),
//         0, 0, 0, nullptr,
//         &p.getKernelConfig(),
//         p.getProfilingData()};

//     // ********** Scalar stencil args ***********
//     // mgcl::CuboidGpu d_v1(p.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, v1);
//     // mgcl::CuboidGpu d_f1(p.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, f1);
//     // mgcl::CuboidGpu d_r1(p.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, r1);
//     // mgcl::VaryingStencilGpu d_sv1(sv1, p.getContext(), p.getCommands(), p.getProgram());
//     // mgcl::CuboidGpu d_v2(p.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, v2);
//     // mgcl::CuboidGpu d_f2(p.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, f2);
//     // mgcl::CuboidGpu d_r2(p.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, r2);
//     // mgcl::VaryingStencilGpu d_sv2(sv2, p.getContext(), p.getCommands(), p.getProgram());

//     lv01.getDVIn().write(p1.getCommands(), v1, false);
//     lv01.getDF().write(p1.getCommands(), f1, false);
//     lv01.getStencilValuesGpu()->write(p1.getCommands(), false, sv1);
//     lv02.getDVIn().write(p2.getCommands(), v2, false);
//     lv02.getDF().write(p2.getCommands(), f2, false);
//     lv02.getStencilValuesGpu()->write(p2.getCommands(), true, sv2);

//     double res = mgcl::MultigridEngine::residual(args);
//     double res1 = mgcl::MultigridEngine::residual(p1, lv01, true);
//     double res2 = mgcl::MultigridEngine::residual(p2, lv02, true);
//     p.finish();
//     p1.finish();
//     p2.finish();

//     lv01.getDR().read(p1.getCommands(), &r1, true);
//     lv02.getDR().read(p2.getCommands(), &r2, true);
//     d_r.read(p.getCommands(), &r, true);

//     // Check both r components
//     for (int i = gh; i < m + gh; i++)
//         for (int j = gh; j < n + gh; j++)
//             for (int k = gh; k < o + gh; k++)
//             {
//                 CAPTURE(i, j, k);
//                 REQUIRE(r1[i][j][k] == r[i][j][k][0]);
//                 REQUIRE(r2[i][j][k] == r[i][j][k][1]);
//             }
// }