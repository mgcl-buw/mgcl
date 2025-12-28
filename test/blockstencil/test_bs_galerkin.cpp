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

// Test calculating the Galerkin operator using blockstencils, which only have entries on the diagonal, i.e.
// the quantities are independent.
// Test against two calculations of scalar Galerkin operators.
TEST_CASE("bs_galerkin_independent_quantities")
{
    int m = 4; // GENERATE(2, 4, 8);
    int n = 4; // GENERATE(2, 4, 8);
    int o = 4; // GENERATE(2, 4, 8);
    int gh_ah = 1;
    int gh_a2h = 1;
    int blocksize = 2;
    bool periodic = true;

    double tol = 1e-12;

    // Fill diagonals of R and P Blockstencils with values of scalar R and P
    mgcl::FixedBlockstencil r(3, blocksize);
    mgcl::FixedBlockstencil p(3, blocksize);
    mgcl_test::fill3dFullWeightRestrictionBlockstencil(r);
    mgcl_test::fill3dBilinearProlongationBlockstencil(p);

    // Fill two varying stencils on fine grid with 27p random values and fill diagonal of blockstencil appropriately
    mgcl::VaryingStencil a_h1(m, n, o, 3, gh_ah, gh_ah, gh_ah);
    mgcl::VaryingStencil a_h2(m, n, o, 3, gh_ah, gh_ah, gh_ah);
    mgcl::Blockstencil a_h_bs(m, n, o, 3, blocksize, gh_ah, gh_ah, gh_ah);
    a_h1.fillRandom();
    a_h2.fillRandom();
    for (int i = gh_ah; i < m + gh_ah; i++)
        for (int j = gh_ah; j < n + gh_ah; j++)
            for (int k = gh_ah; k < o + gh_ah; k++)
                for (int ii = 0; ii < 3; ii++)
                    for (int jj = 0; jj < 3; jj++)
                        for (int kk = 0; kk < 3; kk++)
                        {
                            a_h_bs[0][0][ii][jj][kk][i][j][k] = a_h1[ii][jj][kk][i][j][k];
                            a_h_bs[1][1][ii][jj][kk][i][j][k] = a_h2[ii][jj][kk][i][j][k];
                        }

    a_h_bs.updateGhosts(nullptr, true, periodic);
    a_h1.updateGhosts();
    a_h2.updateGhosts();

    std::unique_ptr<mgcl::Blockstencil> a_2h_bs;
    auto a_2h1 = mgcl::MultigridEngine::galerkinOptimized(a_h1, gh_a2h, m >> 1, n >> 1, o >> 1);
    auto a_2h2 = mgcl::MultigridEngine::galerkinOptimized(a_h2, gh_a2h, m >> 1, n >> 1, o >> 1);

    SECTION("seq")
    {
        a_2h_bs = mgcl::MultigridEngine::galerkinOptimized(a_h_bs, r, p, gh_a2h, m >> 1, n >> 1, o >> 1);
    }

    SECTION("ocl")
    {
        // create dummy problem
        auto v_dummy = std::make_shared<mgcl::Cuboid>(1, 1, 1);
        auto f_dummy = std::make_shared<mgcl::Cuboid>(1, 1, 1);
        mgcl::Problem p_dummy(1, 1, 1, f_dummy, v_dummy);
        p_dummy.setUseOpencl(true);
        p_dummy.setProfilingEnabled(true);
        p_dummy.getOpenCLHelper().setPreprocessorConstant("BLOCKSIZE", std::to_string(blocksize));
        p_dummy.init();

        // bs_inv.dumpToFile("bs_inv.txt");

        mgcl::BlockstencilGpu d_a_h_bs(a_h_bs, p_dummy.getContext(), p_dummy.getCommands(), p_dummy.getProgram());
        mgcl::FixedBlockstencilGpu d_r(r, p_dummy.getContext(), p_dummy.getCommands());
        mgcl::FixedBlockstencilGpu d_p(p, p_dummy.getContext(), p_dummy.getCommands());

        // d_a_h_bs.dumpToFile(p_dummy.getCommands(), "a.txt", false);
        // d_r.dumpToFile(p_dummy.getCommands(), "r.txt", false);
        auto d_2h_bs = mgcl::MultigridEngine::galerkinOptimized(d_a_h_bs, d_r, d_p, gh_a2h, m >> 1, n >> 1, o >> 1,
                                                                p_dummy.getProgram(), p_dummy.getCommands(), p_dummy.getContext(),
                                                                &p_dummy.getKernelConfig(), p_dummy.getProfilingData());

        p_dummy.finish();

        a_2h_bs = d_2h_bs->read_ptr(p_dummy.getCommands(), true);
        // r.dumpToFile("r.txt");
        // r1.dumpToFile("r1.txt");
    }

    // Check both blockstencil vs scalar
    for (int i = gh_a2h; i < (m >> 1) + gh_a2h; i++)
        for (int j = gh_a2h; j < (n >> 1) + gh_a2h; j++)
            for (int k = gh_a2h; k < (o >> 1) + gh_a2h; k++)
                for (int ii = 0; ii < 3; ii++)
                    for (int jj = 0; jj < 3; jj++)
                        for (int kk = 0; kk < 3; kk++)
                        {
                            CAPTURE(i, j, k, ii, jj, kk);
                            REQUIRE_THAT((*a_2h1)[ii][jj][kk][i][j][k], Catch::Matchers::WithinAbs((*a_2h_bs)[0][0][ii][jj][kk][i][j][k], 1e-4));
                            REQUIRE_THAT((*a_2h2)[ii][jj][kk][i][j][k], Catch::Matchers::WithinAbs((*a_2h_bs)[1][1][ii][jj][kk][i][j][k], 1e-4));
                        }
}

// Test calculating the Galerkin operator using a blockstencils containing combined scalars. Same scenario as for
// the Jacobi test. Using a FixedStencil for now.
TEST_CASE("bs_galerkin_combined_scalars")
{
    // blockstencil dims
    int mbs = 4; // GENERATE(2, 4, 8);
    int nbs = 4; // GENERATE(2, 4, 8);
    int obs = 4; // GENERATE(2, 4, 8);
    // scalar dims
    int msc = mbs * 2; // GENERATE(2, 4, 8);
    int nsc = nbs * 2; // GENERATE(2, 4, 8);
    int osc = obs * 2; // GENERATE(2, 4, 8);
    int gh_ah = 1;
    int gh_a2h = 1;
    int blocksize = 8;
    bool periodic = true;

    double tol = 1e-12;

    // Fill diagonals of R and P Blockstencils with values of scalar R and P
    mgcl::FixedBlockstencil r(3, blocksize);
    mgcl::FixedBlockstencil p(3, blocksize);
    mgcl_test::fill3dFullWeightRestrictionBlockstencil(r);
    mgcl_test::fill3dBilinearProlongationBlockstencil(p);

    // Fill two varying stencils on fine grid with 27p random values and fill diagonal of blockstencil appropriately
    mgcl::FixedStencil fs(3);
    mgcl::VaryingStencil a_h_sc(msc, nsc, osc, 3, gh_ah, gh_ah, gh_ah);
    mgcl::Blockstencil a_h_bs(mbs, nbs, obs, 3, blocksize, gh_ah, gh_ah, gh_ah);
    mgcl_test::fillBlockstencilFromFixedStencil(a_h_bs, fs);
    mgcl_test::fillVaryingStencilFromFixedStencil(a_h_sc, fs);

    a_h_bs.updateGhosts(nullptr, true, periodic);
    a_h_sc.updateGhosts();

    std::unique_ptr<mgcl::Blockstencil> a_2h_bs;
    auto a_2h_sc = mgcl::MultigridEngine::galerkinOptimized(a_h_sc, gh_a2h, msc >> 1, nsc >> 1, osc >> 1);

    SECTION("seq")
    {
        a_2h_bs = mgcl::MultigridEngine::galerkinOptimized(a_h_bs, r, p, gh_a2h, mbs >> 1, nbs >> 1, obs >> 1);
    }

    // SECTION("ocl")
    // {
    //     // create dummy problem
    //     auto v_dummy = std::make_shared<mgcl::Cuboid>(1, 1, 1);
    //     auto f_dummy = std::make_shared<mgcl::Cuboid>(1, 1, 1);
    //     mgcl::Problem p_dummy(1, 1, 1, f_dummy, v_dummy);
    //     p_dummy.setUseOpencl(true);
    //     p_dummy.setProfilingEnabled(true);
    //     p_dummy.getOpenCLHelper().setPreprocessorConstant("BLOCKSIZE", std::to_string(blocksize));
    //     p_dummy.init();

    //     // bs_inv.dumpToFile("bs_inv.txt");

    //     mgcl::BlockstencilGpu d_a_h_bs(a_h_bs, p_dummy.getContext(), p_dummy.getCommands(), p_dummy.getProgram());
    //     mgcl::FixedBlockstencilGpu d_r(r, p_dummy.getContext(), p_dummy.getCommands());
    //     mgcl::FixedBlockstencilGpu d_p(p, p_dummy.getContext(), p_dummy.getCommands());

    //     // d_a_h_bs.dumpToFile(p_dummy.getCommands(), "a.txt", false);
    //     // d_r.dumpToFile(p_dummy.getCommands(), "r.txt", false);
    //     auto d_2h_bs = mgcl::MultigridEngine::galerkinOptimized(d_a_h_bs, d_r, d_p, gh_a2h, mbs >> 1, nbs >> 1, obs >> 1,
    //                                                             p_dummy.getProgram(), p_dummy.getCommands(), p_dummy.getContext(),
    //                                                             &p_dummy.getKernelConfig(), p_dummy.getProfilingData());

    //     p_dummy.finish();

    //     a_2h_bs = d_2h_bs->read_ptr(p_dummy.getCommands(), true);
    //     // r.dumpToFile("r.txt");
    //     // r1.dumpToFile("r1.txt");
    // }

    mgcl::FixedStencil fs_bs_res(3);
    mgcl_test::fillFixedStencilFromBlockstencil(*a_2h_bs, fs_bs_res);
    // Check both blockstencil vs scalar
    for (int i = gh_a2h; i < (msc >> 1) + gh_a2h; i++)
        for (int j = gh_a2h; j < (nsc >> 1) + gh_a2h; j++)
            for (int k = gh_a2h; k < (osc >> 1) + gh_a2h; k++)
                for (int ii = 0; ii < 3; ii++)
                    for (int jj = 0; jj < 3; jj++)
                        for (int kk = 0; kk < 3; kk++)
                        {
                            CAPTURE(i, j, k, ii, jj, kk);
                            REQUIRE_THAT((*a_2h_sc)[ii][jj][kk][i][j][k], Catch::Matchers::WithinAbs(fs_bs_res[ii][jj][kk], 1e-4));
                        }
}
