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
TEST_CASE("seq_bs_galerkin_independent_quantities")
{
    int m = 4; // GENERATE(2, 4, 8);
    int n = 4; // GENERATE(2, 4, 8);
    int o = 4; // GENERATE(2, 4, 8);
    int gh_ah = 1;
    int gh_a2h = 1;
    int blocksize = 2;

    double tol = 1e-12;

    // Fill diagonals of R and P Blockstencils with values of scalar R and P
    auto r_fs = mgcl::create3dFullWeightRestrictionStencil();
    auto p_fs = mgcl::create3dBilinearProlongationStencil();
    mgcl::FixedBlockstencil r(3, blocksize);
    mgcl::FixedBlockstencil p(3, blocksize);
    for (size_t bi = 0; bi < blocksize; bi++)
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                for (int k = 0; k < 3; k++)
                {
                    r[bi][bi][i][j][k] = r_fs[i][j][k];
                    p[bi][bi][i][j][k] = p_fs[i][j][k];
                }

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

    a_h_bs.updateGhosts(nullptr, true);
    a_h1.updateGhosts();
    a_h2.updateGhosts();

    auto a_2h_bs = mgcl::MultigridEngine::galerkinOptimized(a_h_bs, r, p, gh_a2h, m >> 1, n >> 1, o >> 1);
    auto a_2h1 = mgcl::MultigridEngine::galerkinOptimized(a_h1, gh_a2h, m >> 1, n >> 1, o >> 1);
    auto a_2h2 = mgcl::MultigridEngine::galerkinOptimized(a_h2, gh_a2h, m >> 1, n >> 1, o >> 1);

    // SECTION("seq")
    // {
    //     mgcl::args::JacobiBSSeqArgs args{
    //         f,
    //         v,
    //         r,
    //         resnorm,
    //         bs,
    //         bs_inv,
    //         true,
    //         periodic,
    //         true, iters, stepsPerIter, omega,
    //         0, 0, 0, nullptr};

    //     double res = mgcl::MultigridEngine::jacobiSeq(args);

    //     // r.dumpToFile("r.txt");
    //     // r1.dumpToFile("r1.txt");
    // }

    // SECTION("ocl")
    // {
    //     // create dummy problem
    //     auto v_dummy = std::make_shared<mgcl::Cuboid>(1, 1, 1);
    //     auto f_dummy = std::make_shared<mgcl::Cuboid>(1, 1, 1);
    //     mgcl::Problem p(1, 1, 1, f_dummy, v_dummy);
    //     p.setUseOpencl(true);
    //     p.setProfilingEnabled(true);
    //     p.init();

    //     // bs_inv.dumpToFile("bs_inv.txt");

    //     mgcl::CuboidBSGpu d_v_in(p.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, v);
    //     mgcl::CuboidBSGpu d_v_out(p.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, v);
    //     mgcl::CuboidBSGpu d_r(p.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, r);
    //     mgcl::CuboidBSGpu d_f(p.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, f);
    //     mgcl::BlockstencilGpu d_bs(bs, p.getContext(), p.getCommands(), p.getProgram());
    //     mgcl::BlockstencilGpu d_bs_inv(bs_inv, p.getContext(), p.getCommands(), p.getProgram());
    //     mgcl::CuboidBSGpu dRSquares(p.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, r);

    //     mgcl::args::JacobiBSOclArgs args{
    //         d_f,
    //         d_v_in,
    //         d_v_out,
    //         d_r,
    //         resnorm,
    //         d_bs,
    //         d_bs_inv,
    //         &dRSquares,
    //         true,
    //         periodic,
    //         true, iters, stepsPerIter, omega,
    //         nullptr, nullptr, nullptr,
    //         p.getProgram(), p.getCommands(), p.getContext(),
    //         0, 0, 0, nullptr,
    //         &p.getKernelConfig(),
    //         p.getProfilingData()};

    //     double res = mgcl::MultigridEngine::jacobi(args);

    //     p.finish();

    //     d_v_in.read(p.getCommands(), &v, true);
    //     d_r.read(p.getCommands(), &r, true);

    //     // r.dumpToFile("r.txt");
    //     // r1.dumpToFile("r1.txt");
    // }

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
