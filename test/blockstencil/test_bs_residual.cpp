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

    // r.dumpToFile("r.txt");
    // r1.dumpToFile("r1.txt");

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

// Test applying a blockstencil on one point
TEST_CASE("ocl_bs_residual_single_point")
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

    // create dummy problem
    auto v_dummy = std::make_shared<mgcl::Cuboid>(1, 1, 1);
    auto f_dummy = std::make_shared<mgcl::Cuboid>(1, 1, 1);
    mgcl::Problem p(1, 1, 1, f_dummy, v_dummy);
    p.setUseOpencl(true);
    p.setProfilingEnabled(true);
    p.getOpenCLHelper().setPreprocessorConstant("BLOCKSIZE", std::to_string(blocksize));
    p.init();

    mgcl::CuboidBSGpu d_v(p.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, v);
    mgcl::CuboidBSGpu d_f(p.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, f);
    mgcl::CuboidBSGpu d_r(p.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, r);
    mgcl::BlockstencilGpu d_bs(bs, p.getContext(), p.getCommands(), p.getProgram());
    mgcl::CuboidBSGpu dRSquares(p.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, r);
    // mgcl::BufferGpu dPlanesBuf(p.getContext(), CL_MEM_READ_WRITE, 2 * gh * (mgh * ngh + mgh * ogh + ngh * ogh));
    // std::vector<double> sendBuf(2 * gh * (mgh * ngh + mgh * ogh + ngh * ogh));
    // std::vector<double> recvBuf(2 * gh * (mgh * ngh + mgh * ogh + ngh * ogh));

    mgcl::args::ResidualBSOclArgs args{
        d_f,
        d_v,
        d_r,
        resnorm,
        d_bs,
        &dRSquares,
        true,
        periodic,
        true,
        // &dPlanesBuf,
        // &sendBuf,
        // &recvBuf,
        nullptr,
        nullptr,
        nullptr,
        p.getProgram(),
        p.getCommands(),
        p.getContext(),
        0, 0, 0, nullptr,
        &p.getKernelConfig(),
        p.getProfilingData()};

    // d_v.dumpToFile(p.getCommands(), "v.txt", false);
    // d_f.dumpToFile(p.getCommands(), "f.txt", false);

    double res = mgcl::MultigridEngine::residual(args);
    p.finish();

    // d_r.dumpToFile(p.getCommands(), "r.txt", false);
    // d_bs.dumpToFile(p.getCommands(), "bs.txt", false);

    d_r.read(p.getCommands(), &r, true);

    // calculated by hand
    REQUIRE(r[1][1][1][0] == -101323);
    REQUIRE(r[1][1][1][1] == -104184);
}

// Test applying a blockstencil on multiple grid points with independent quantitites, i.e.
// a blockstencil only has entries on its diagonal.
// The result is compared to two residual separate calculations of the two quantities.
TEST_CASE("ocl_bs_residual_independent_quantities")
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
    mgcl::Blockstencil bs(m, n, o, width, blocksize, gh, gh, gh);

    auto v1_ptr = std::make_shared<mgcl::Cuboid>(m, n, o, gh, gh, gh);
    auto f1_ptr = std::make_shared<mgcl::Cuboid>(m, n, o, gh, gh, gh);
    auto& v1 = *v1_ptr;
    auto& f1 = *f1_ptr;
    mgcl::Cuboid r1(m, n, o, gh, gh, gh);
    auto v2_ptr = std::make_shared<mgcl::Cuboid>(m, n, o, gh, gh, gh);
    auto f2_ptr = std::make_shared<mgcl::Cuboid>(m, n, o, gh, gh, gh);
    auto& v2 = *v2_ptr;
    auto& f2 = *f2_ptr;
    mgcl::Cuboid r2(m, n, o, gh, gh, gh);

    mgcl::Problem p1(m, n, o, f1_ptr, v1_ptr);
    p1.setUseOpencl(true);
    p1.setProfilingEnabled(true);
    p1.setGhostsIn(gh);
    p1.setStencilType(mgcl::MGCL_VARYING);

    mgcl::Problem p2(m, n, o, f2_ptr, v2_ptr);
    p2.setUseOpencl(true);
    p2.setProfilingEnabled(true);
    p2.setGhostsIn(gh);
    p2.setStencilType(mgcl::MGCL_VARYING);

    auto& sv1_ptr = p1.getStencilValues();
    auto& sv2_ptr = p2.getStencilValues();
    auto& sv1 = *sv1_ptr;
    auto& sv2 = *sv2_ptr;

    v.fill1dIndex(false);
    f.fill1dIndex(false);

    // fill different coefficients for v1 and v2
    // clang-format off
    int cnt = 0;
    for (int i = sv1.getGhostsM(); i < sv1.getGhostsM() + m; i++)
    for (int j = sv1.getGhostsN(); j < sv1.getGhostsN() + n; j++)
    for (int k = sv1.getGhostsO(); k < sv1.getGhostsO() + o; k++)
        for (int ii = 0; ii < width; ii++)
        for (int jj = 0; jj < width; jj++)
        for (int kk = 0; kk < width; kk++)
        {
            sv1[ii][jj][kk][i][j][k] = cnt++;
            sv2[ii][jj][kk][i][j][k] = -cnt;
        }
    // clang-format on

    p1.init();
    auto& lv01 = p1.getLevelAt(0);
    p2.init();
    auto& lv02 = p2.getLevelAt(0);

    // fill blockstencil diagonal with values from sv1 and sv2
    for (int ci = 0; ci < width; ci++)
        for (int cj = 0; cj < width; cj++)
            for (int ck = 0; ck < width; ck++)
                for (int i = gh; i < m + gh; i++)
                    for (int j = gh; j < n + gh; j++)
                        for (int k = gh; k < o + gh; k++)
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

    // create dummy problem
    auto v_dummy = std::make_shared<mgcl::Cuboid>(1, 1, 1);
    auto f_dummy = std::make_shared<mgcl::Cuboid>(1, 1, 1);
    mgcl::Problem p(1, 1, 1, f_dummy, v_dummy);
    p.setUseOpencl(true);
    p.setProfilingEnabled(true);
    p.getOpenCLHelper().setPreprocessorConstant("BLOCKSIZE", std::to_string(blocksize));
    p.init();

    // ********** Blockstencil args ***********
    mgcl::CuboidBSGpu d_v(p.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, v);
    mgcl::CuboidBSGpu d_f(p.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, f);
    mgcl::CuboidBSGpu d_r(p.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, r);
    mgcl::BlockstencilGpu d_bs(bs, p.getContext(), p.getCommands(), p.getProgram());
    mgcl::CuboidBSGpu dRSquares(p.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, r);
    // mgcl::BufferGpu dPlanesBuf(p.getContext(), CL_MEM_READ_WRITE, 2 * gh * (mgh * ngh + mgh * ogh + ngh * ogh));
    // std::vector<double> sendBuf(2 * gh * (mgh * ngh + mgh * ogh + ngh * ogh));
    // std::vector<double> recvBuf(2 * gh * (mgh * ngh + mgh * ogh + ngh * ogh));

    mgcl::args::ResidualBSOclArgs args{
        d_f,
        d_v,
        d_r,
        resnorm,
        d_bs,
        &dRSquares,
        true,
        periodic,
        true,
        // &dPlanesBuf,
        // &sendBuf,
        // &recvBuf,
        nullptr,
        nullptr,
        nullptr,
        p.getProgram(),
        p.getCommands(),
        p.getContext(),
        0, 0, 0, nullptr,
        &p.getKernelConfig(),
        p.getProfilingData()};

    // ********** Scalar stencil args ***********
    // mgcl::CuboidGpu d_v1(p.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, v1);
    // mgcl::CuboidGpu d_f1(p.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, f1);
    // mgcl::CuboidGpu d_r1(p.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, r1);
    // mgcl::VaryingStencilGpu d_sv1(sv1, p.getContext(), p.getCommands(), p.getProgram());
    // mgcl::CuboidGpu d_v2(p.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, v2);
    // mgcl::CuboidGpu d_f2(p.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, f2);
    // mgcl::CuboidGpu d_r2(p.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, r2);
    // mgcl::VaryingStencilGpu d_sv2(sv2, p.getContext(), p.getCommands(), p.getProgram());

    lv01.getDVIn().write(p1.getCommands(), v1, false);
    lv01.getDF().write(p1.getCommands(), f1, false);
    lv01.getStencilValuesGpu()->write(p1.getCommands(), false, sv1);
    lv02.getDVIn().write(p2.getCommands(), v2, false);
    lv02.getDF().write(p2.getCommands(), f2, false);
    lv02.getStencilValuesGpu()->write(p2.getCommands(), true, sv2);

    double res = mgcl::MultigridEngine::residual(args);
    double res1 = mgcl::MultigridEngine::residual(p1, lv01, true);
    double res2 = mgcl::MultigridEngine::residual(p2, lv02, true);
    p.finish();
    p1.finish();
    p2.finish();

    lv01.getDR().read(p1.getCommands(), &r1, true);
    lv02.getDR().read(p2.getCommands(), &r2, true);
    d_r.read(p.getCommands(), &r, true);

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

// Calculates residual on combined scalar values and compares it to result of scalar residual
TEST_CASE("seq_bs_residual_combined_scalars")
{
    int mf = 8;
    int nf = 8;
    int of = 8;
    int mc = mf / 2;
    int nc = nf / 2;
    int oc = of / 2;
    int blocksize = 8;
    int width = 3;
    int gh_sc = GENERATE(1, 2);
    int gh_bs = GENERATE(1, 2);
    int mfgh = mf + 2 * gh_sc;
    int nfgh = nf + 2 * gh_sc;
    int ofgh = of + 2 * gh_sc;
    int mcgh = mc + 2 * gh_bs;
    int ncgh = nc + 2 * gh_bs;
    int ocgh = oc + 2 * gh_bs;

    bool periodic = true;

    mgcl::MGCL_RESIDUAL_NORM resnorm = mgcl::MGCL_L2;

    mgcl::CuboidBS v(mc, nc, oc, gh_bs, gh_bs, gh_bs, blocksize);
    mgcl::CuboidBS r(mc, nc, oc, gh_bs, gh_bs, gh_bs, blocksize);
    mgcl::CuboidBS f(mc, nc, oc, gh_bs, gh_bs, gh_bs, blocksize);
    mgcl::Blockstencil bs(mc, nc, oc, width, blocksize, gh_bs, gh_bs, gh_bs);

    mgcl::Cuboid v1(mf, nf, of, gh_sc, gh_sc, gh_sc);
    mgcl::Cuboid f1(mf, nf, of, gh_sc, gh_sc, gh_sc);
    mgcl::Cuboid r1(mf, nf, of, gh_sc, gh_sc, gh_sc);
    // mgcl::VaryingStencil sv1(mf, nf, of, width, 0, 0, 0);
    // mgcl::VaryingStencil sv2(mf, nf, of, width, 0, 0, 0);
    mgcl::FixedStencil fs1(width);

    // fill with 4th order periodic Problem
    double hm = 1.0 / (double)mf;
    double hn = 1.0 / (double)nf;
    double ho = 1.0 / (double)of;
    for (int i = 0; i < mf; i++)
        for (int j = 0; j < nf; j++)
            for (int k = 0; k < of; k++)
            {
                double zs = i * ho;
                double ys = j * hn;
                double xs = k * hm;
                double xs2 = xs * xs;
                double ys2 = ys * ys;
                double zs2 = zs * zs;
                double xsm1_2 = (xs - 1) * (xs - 1);
                double ysm1_2 = (ys - 1) * (ys - 1);
                double zsm1_2 = (zs - 1) * (zs - 1);
                double xs3 = xs * xs * xs;
                double ys3 = ys * ys * ys;
                double zs3 = zs * zs * zs;
                double xsm1_3 = (xs - 1) * (xs - 1) * (xs - 1);
                double ysm1_3 = (ys - 1) * (ys - 1) * (ys - 1);
                double zsm1_3 = (zs - 1) * (zs - 1) * (zs - 1);
                double xs4 = xs * xs * xs * xs;
                double ys4 = ys * ys * ys * ys;
                double zs4 = zs * zs * zs * zs;
                double xsm1_4 = (xs - 1) * (xs - 1) * (xs - 1) * (xs - 1);
                double ysm1_4 = (ys - 1) * (ys - 1) * (ys - 1) * (ys - 1);
                double zsm1_4 = (zs - 1) * (zs - 1) * (zs - 1) * (zs - 1);
                v1[i][j][k] = 0;
                // solutionsc[i][j][k] = 1000000 * (xs * (xs - 1)) * (xs * (xs - 1)) * (xs * (xs - 1)) * (xs * (xs - 1)) *
                //                       (ys * (ys - 1)) * (ys * (ys - 1)) * (ys * (ys - 1)) * (ys * (ys - 1)) *
                //                       (zs * (zs - 1)) * (zs * (zs - 1)) * (zs * (zs - 1)) * (zs * (zs - 1));
                f1[i][j][k] =
                    -1000000 *
                    (12 * xs4 * ys4 * zs4 * xsm1_4 * ysm1_4 * zsm1_2 + 12 * xs4 * ys4 * zs4 * xsm1_4 * ysm1_2 * zsm1_4 +
                     12 * xs4 * ys4 * zs4 * xsm1_2 * ysm1_4 * zsm1_4 + 32 * xs4 * ys4 * zs3 * xsm1_4 * ysm1_4 * zsm1_3 +
                     12 * xs4 * ys4 * zs2 * xsm1_4 * ysm1_4 * zsm1_4 + 32 * xs4 * ys3 * zs4 * xsm1_4 * ysm1_3 * zsm1_4 +
                     12 * xs4 * ys2 * zs4 * xsm1_4 * ysm1_4 * zsm1_4 + 32 * xs3 * ys4 * zs4 * xsm1_3 * ysm1_4 * zsm1_4 +
                     12 * xs2 * ys4 * zs4 * xsm1_4 * ysm1_4 * zsm1_4);
            }

    mgcl_test::fill7pLaplace(fs1, 1.0 / (double)mf, false);
    mgcl_test::fillBlockstencilFromFixedStencil(bs, fs1);

    mgcl_test::copyCuboidToCuboidBS(v1, v);
    mgcl_test::copyCuboidToCuboidBS(f1, f);

    v.updateGhosts(nullptr, true);
    mgcl::MultigridEngine::updateGhostsSeq(v1, nullptr, true, false);
    f.updateGhosts(nullptr, true);
    mgcl::MultigridEngine::updateGhostsSeq(f1, nullptr, true, false);

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
    double res1 = mgcl::MultigridEngine::residualSeq(f1, v1, r1, resnorm, mgcl::MGCL_FIXED, 0, nullptr, &fs1, true, true, true);

    // r.dumpToFile("r.txt");
    // r1.dumpToFile("r1.txt");

    // Check r
    auto r_bs_as_sc_ptr = r1.copyShallow();
    auto& r_bs_as_sc = *r_bs_as_sc_ptr;
    mgcl_test::copyCuboidBSToCuboid(r, r_bs_as_sc);
    REQUIRE(r1.isEqual(r_bs_as_sc));

    for (int i = gh_bs, i2 = gh_sc; i < mc + gh_bs; i++, i2 += 2)
        for (int j = gh_bs, j2 = gh_sc; j < nc + gh_bs; j++, j2 += 2)
            for (int k = gh_bs, k2 = gh_sc; k < oc + gh_bs; k++, k2 += 2)
            {
                CAPTURE(i, j, k, i2, j2, k2);
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