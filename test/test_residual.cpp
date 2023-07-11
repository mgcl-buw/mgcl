#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <iostream>

#include "../src/cuboid.hpp"
#include "../src/multigrid_engine.hpp"
#include "test_results.hpp"
#include "test_utility.hpp"

std::shared_ptr<mgcl::Cuboid> residualTestInputF();
std::shared_ptr<mgcl::Cuboid> residualTestInputV();
std::shared_ptr<mgcl::Cuboid> residualTestOutputR();

TEST_CASE("residual")
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

    auto c_in_f = mgcl_test::test_residual::inputF16();
    auto c_in_v = mgcl_test::test_residual::inputV16();
    auto c_in_r = std::make_unique<mgcl::Cuboid>(m, n, o, ghosts_m, ghosts_n, ghosts_o);
    auto c_expected_out_r = mgcl_test::test_residual::outputR16();

    auto stencil = mgcl::MGCL_LAPLACE_7POINT;
    double h = 1.0 / (double)m;
    double stencilFactor = 1.0 / (h * h);

    SECTION("residualSeq L2-norm 7point periodic")
    {
        auto stencilValues = std::make_unique<mgcl::VaryingStencil3x3x3>(1, 1, 1, 0, 0, 0); // just a dummy
        double res = mgcl::MultigridEngine::residualSeq(*c_in_f, *c_in_v, *c_in_r, mgcl::MGCL_L2,
                                                        mgcl::MGCL_LAPLACE_7POINT, stencilFactor, *stencilValues, true, true);

        CHECK(fabs(res - 3.00209960095333271e+07) < 1e-7);
        REQUIRE(c_in_r->isEqual(*c_expected_out_r));
    }

    SECTION("residual OpenCL L2-norm 7point")
    {
        auto deviceType = GENERATE(CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU);

        if (!mgcl_test::TestUtility::deviceAvailable("", deviceType))
        {
            std::string typeName = deviceType == CL_DEVICE_TYPE_GPU ? "CL_DEVICE_TYPE_GPU" : "CL_DEVICE_TYPE_CPU";
            std::cout << "Skipping non-available device type '" << typeName << "'" << std::endl;
            return;
        }

        auto p = std::make_shared<mgcl::Problem>(m, n, o);
        p->setResidualNorm(mgcl::MGCL_L2);
        p->setStencilType(stencil);
        p->setGhosts(1);
        p->setDeviceType(deviceType);

        mgcl_test::TestUtility tu(p);
        cl_mem d_in_f = tu.createOpenCLBuffer(*c_in_f);
        cl_mem d_in_v = tu.createOpenCLBuffer(*c_in_v);
        cl_mem d_in_r = tu.createOpenCLBuffer(*c_in_r);

        mgcl::Level level(p.get(), 0);
        level.setDF(d_in_f);
        level.setDVIn(d_in_v);
        level.setDR(d_in_r);

        double res = mgcl::MultigridEngine::residual(*p, level, 1);
        tu.finish();

        auto c_r_out = tu.readOpenCLBuffer(d_in_r, m, n, o, ghosts_m, ghosts_n, ghosts_o);

        CHECK(fabs(res - 3.00209960095333271e+07) < 1e-7);
        REQUIRE(c_r_out->isEqual(*c_expected_out_r));
    }

    SECTION("residualSeq L2-norm 7point varying stencil periodic")
    {
        int n = 16;
        auto vals = mgcl::VaryingStencil3x3x3(n, n, n, 1, 1, 1);

        double h2inv = static_cast<double>(n * n);
        for (int i = 0; i < vals.getDim1gh(); i++)
            for (int j = 0; j < vals.getDim2gh(); j++)
                for (int k = 0; k < vals.getDim3gh(); k++)
                {
                    vals[i][j][k][1][1][1] = 6.0 * h2inv;
                    vals[i][j][k][1][1][0] = -1.0 * h2inv;
                    vals[i][j][k][1][1][2] = -1.0 * h2inv;
                    vals[i][j][k][1][0][1] = -1.0 * h2inv;
                    vals[i][j][k][1][2][1] = -1.0 * h2inv;
                    vals[i][j][k][0][1][1] = -1.0 * h2inv;
                    vals[i][j][k][2][1][1] = -1.0 * h2inv;
                }

        double res = mgcl::MultigridEngine::residualSeq(*c_in_f, *c_in_v, *c_in_r, mgcl::MGCL_L2,
                                                        mgcl::MGCL_VARYING, stencilFactor, vals, true, true);

        CHECK(fabs(res - 3.00209960095333271e+07) < 1e-7);
        REQUIRE(c_in_r->isEqual(*c_expected_out_r));
    }
}

TEST_CASE("residual periodic Laplace seq vs ocl")
{
    auto deviceType = GENERATE(CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU);

    if (!mgcl_test::TestUtility::deviceAvailable("", deviceType))
    {
        std::string typeName = deviceType == CL_DEVICE_TYPE_GPU ? "CL_DEVICE_TYPE_GPU" : "CL_DEVICE_TYPE_CPU";
        std::cout << "Skipping non-available device type '" << typeName << "'" << std::endl;
        return;
    }

    int m = 8;
    int n = 8;
    int o = 8;
    mgcl::MGCL_RESIDUAL_NORM resnorm = mgcl::MGCL_L2;
    mgcl::MGCL_STENCIL stencilType = GENERATE(mgcl::MGCL_LAPLACE_7POINT, mgcl::MGCL_LAPLACE_19POINT, mgcl::MGCL_LAPLACE_27POINT);
    mgcl::BC bc = mgcl::BC::PERIODIC;

    auto v_in = std::make_shared<mgcl::Cuboid>(m, n, o, 0, 0, 0);
    auto f_in = std::make_shared<mgcl::Cuboid>(m, n, o, 0, 0, 0);
    v_in->fillRandomInt();
    f_in->fillRandomInt();

    // init sequential Problem
    auto p_seq = std::make_shared<mgcl::Problem>(m, n, o);
    p_seq->setResidualNorm(resnorm);
    p_seq->setDeviceType(deviceType);
    p_seq->setV(v_in);
    p_seq->setF(f_in);
    p_seq->setStencilType(stencilType);
    p_seq->setBc(bc);

    p_seq->init();
    auto &level0_seq = p_seq->getLevelAt(0);

    // init OpenCL problem
    auto p_gpu = std::make_shared<mgcl::Problem>(m, n, o);
    p_gpu->setResidualNorm(resnorm);
    p_gpu->setDeviceType(deviceType);
    p_gpu->setV(v_in);
    p_gpu->setF(f_in);
    p_gpu->setUseOpencl(true);
    p_gpu->setStencilType(stencilType);
    p_gpu->setBc(bc);

    p_gpu->init();
    auto &level0_gpu = p_gpu->getLevelAt(0);

    auto &v_in_lv0 = level0_seq.getV();
    auto &f_in_lv0 = level0_seq.getF();
    auto &r_in_lv0 = level0_seq.getR();

    mgcl_test::TestUtility tu(p_gpu);

    mgcl::MultigridEngine::updateGhosts(*p_gpu, level0_gpu.getDVIn(),
                                        level0_gpu.getMgh(), level0_gpu.getNgh(), level0_gpu.getOgh(), 1, 1, 1);
    tu.finish();
    mgcl::MultigridEngine::updateGhostsSeq(v_in_lv0);

    // make sure input is equal
    auto c_r_in = tu.readOpenCLBuffer(level0_gpu.getDR(), m, n, o, 1, 1, 1);
    auto c_v_in = tu.readOpenCLBuffer(level0_gpu.getDVIn(), m, n, o, 1, 1, 1);
    auto c_f_in = tu.readOpenCLBuffer(level0_gpu.getDF(), m, n, o, 1, 1, 1);
    tu.finish();
    REQUIRE(c_r_in->isEqual(r_in_lv0));
    REQUIRE(c_v_in->isEqual(v_in_lv0));
    REQUIRE(c_f_in->isEqual(f_in_lv0));

    double res_gpu = mgcl::MultigridEngine::residual(*p_gpu, level0_gpu, true);
    tu.finish();

    auto stencilValues = std::make_unique<mgcl::VaryingStencil3x3x3>(1, 1, 1, 0, 0, 0); // just a dummy
    double res_seq = mgcl::MultigridEngine::residualSeq(f_in_lv0, v_in_lv0, r_in_lv0, resnorm,
                                                        stencilType, level0_seq.getStencilFactor(), *stencilValues, true, true);

    auto c_r_out = tu.readOpenCLBuffer(level0_gpu.getDR(), m, n, o, 1, 1, 1);
    auto c_v_out = tu.readOpenCLBuffer(level0_gpu.getDVIn(), m, n, o, 1, 1, 1);
    auto c_f_out = tu.readOpenCLBuffer(level0_gpu.getDF(), m, n, o, 1, 1, 1);

    REQUIRE(c_r_out->getM() == r_in_lv0.getM());
    REQUIRE(c_r_out->getN() == r_in_lv0.getN());
    REQUIRE(c_r_out->getO() == r_in_lv0.getO());
    REQUIRE(c_r_out->getMgh() == r_in_lv0.getMgh());
    REQUIRE(c_r_out->getNgh() == r_in_lv0.getNgh());
    REQUIRE(c_r_out->getOgh() == r_in_lv0.getOgh());

    // c_r_out->dumpToFile("../r_gpu.txt");
    // r_in_lv0.dumpToFile("../r_seq.txt");
    // c_v_out->dumpToFile("../v_gpu.txt");
    // v_in_lv0.dumpToFile("../v_seq.txt");

    // sv_in_lv0->dumpToFile("../sv_seq.txt");
    // c_sv_in.dumpToFile("../sv_gpu.txt");

    REQUIRE(c_v_out->isEqual(v_in_lv0)); // should be untouched
    REQUIRE(c_f_out->isEqual(f_in_lv0)); // should be untouched

    REQUIRE_THAT(res_seq, Catch::Matchers::WithinAbs(res_gpu, 1e-7));
    REQUIRE(c_r_out->isEqual(r_in_lv0));
}

TEST_CASE("residual periodic varying stencil seq vs ocl")
{
    auto deviceType = GENERATE(CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU);

    if (!mgcl_test::TestUtility::deviceAvailable("", deviceType))
    {
        std::string typeName = deviceType == CL_DEVICE_TYPE_GPU ? "CL_DEVICE_TYPE_GPU" : "CL_DEVICE_TYPE_CPU";
        std::cout << "Skipping non-available device type '" << typeName << "'" << std::endl;
        return;
    }

    int m = 8;
    int n = 8;
    int o = 8;
    mgcl::MGCL_RESIDUAL_NORM resnorm = mgcl::MGCL_L2;
    mgcl::MGCL_STENCIL stencilType = mgcl::MGCL_VARYING;
    mgcl::BC bc = mgcl::BC::PERIODIC;

    auto v_in = std::make_shared<mgcl::Cuboid>(m, n, o, 0, 0, 0);
    auto f_in = std::make_shared<mgcl::Cuboid>(m, n, o, 0, 0, 0);
    v_in->fillRandomInt();
    f_in->fillRandomInt();

    // init sequential Problem
    auto p_seq = std::make_shared<mgcl::Problem>(m, n, o);
    p_seq->setResidualNorm(resnorm);
    p_seq->setDeviceType(deviceType);
    p_seq->setV(v_in);
    p_seq->setF(f_in);
    p_seq->setBc(bc);

    p_seq->setStencilType(stencilType);
    auto &sv = p_seq->getStencilValues();
    sv->fillRandomInt();

    p_seq->init();
    auto &level0_seq = p_seq->getLevelAt(0);
    REQUIRE(level0_seq.getStencilValues().get() == sv.get());

    // init OpenCL problem
    auto p_gpu = std::make_shared<mgcl::Problem>(m, n, o);
    p_gpu->setResidualNorm(resnorm);
    p_gpu->setDeviceType(deviceType);
    p_gpu->setV(v_in);
    p_gpu->setF(f_in);
    p_gpu->setUseOpencl(true);
    p_gpu->setBc(bc);

    p_gpu->setStencilType(stencilType);
    auto &sv_gpu = p_gpu->getStencilValues();

    // copy stencil values
    REQUIRE(sv->field1d().size() == sv_gpu->field1d().size());
    for (int i = 0; i < sv->field1d().size(); i++)
        sv_gpu->field1d()[i] = sv->field1d()[i];

    p_gpu->init();
    auto &level0_gpu = p_gpu->getLevelAt(0);
    REQUIRE(level0_gpu.getStencilValuesGpu() != nullptr);

    auto &v_in_lv0 = level0_seq.getV();
    auto &f_in_lv0 = level0_seq.getF();
    auto &r_in_lv0 = level0_seq.getR();
    auto &sv_in_lv0 = level0_seq.getStencilValues();

    mgcl_test::TestUtility tu(p_gpu);

    mgcl::MultigridEngine::updateGhosts(*p_gpu, level0_gpu.getDVIn(),
                                        level0_gpu.getMgh(), level0_gpu.getNgh(), level0_gpu.getOgh(), 1, 1, 1);
    tu.finish();
    mgcl::MultigridEngine::updateGhostsSeq(v_in_lv0);

    // make sure input is equal
    auto c_r_in = tu.readOpenCLBuffer(level0_gpu.getDR(), m, n, o, 1, 1, 1);
    auto c_v_in = tu.readOpenCLBuffer(level0_gpu.getDVIn(), m, n, o, 1, 1, 1);
    auto c_f_in = tu.readOpenCLBuffer(level0_gpu.getDF(), m, n, o, 1, 1, 1);
    auto c_sv_in = level0_gpu.getStencilValuesGpu()->read<3>(tu.getCommands());
    tu.finish();

    REQUIRE(c_r_in->isEqual(r_in_lv0));
    REQUIRE(c_v_in->isEqual(v_in_lv0));
    REQUIRE(c_f_in->isEqual(f_in_lv0));
    REQUIRE(c_sv_in.isEqual(*sv_in_lv0));

    double res_gpu = mgcl::MultigridEngine::residual(*p_gpu, level0_gpu, true);
    tu.finish();
    double res_seq = mgcl::MultigridEngine::residualSeq(f_in_lv0, v_in_lv0, r_in_lv0, resnorm,
                                                        stencilType, level0_seq.getStencilFactor(), *sv_in_lv0, true, true);

    auto c_r_out = tu.readOpenCLBuffer(level0_gpu.getDR(), m, n, o, 1, 1, 1);
    auto c_v_out = tu.readOpenCLBuffer(level0_gpu.getDVIn(), m, n, o, 1, 1, 1);
    auto c_f_out = tu.readOpenCLBuffer(level0_gpu.getDF(), m, n, o, 1, 1, 1);
    tu.finish();

    REQUIRE(c_r_out->getM() == r_in_lv0.getM());
    REQUIRE(c_r_out->getN() == r_in_lv0.getN());
    REQUIRE(c_r_out->getO() == r_in_lv0.getO());
    REQUIRE(c_r_out->getMgh() == r_in_lv0.getMgh());
    REQUIRE(c_r_out->getNgh() == r_in_lv0.getNgh());
    REQUIRE(c_r_out->getOgh() == r_in_lv0.getOgh());

    // c_r_out->dumpToFile("../r_gpu.txt");
    // r_in_lv0.dumpToFile("../r_seq.txt");
    // c_v_out->dumpToFile("../v_gpu.txt");
    // v_in_lv0.dumpToFile("../v_seq.txt");

    // sv_in_lv0->dumpToFile("../sv_seq.txt");
    // c_sv_in.dumpToFile("../sv_gpu.txt");

    REQUIRE(c_v_out->isEqual(v_in_lv0)); // should be untouched
    REQUIRE(c_f_out->isEqual(f_in_lv0)); // should be untouched

    REQUIRE_THAT(res_seq, Catch::Matchers::WithinAbs(res_gpu, 1e-7));
    REQUIRE(c_r_out->isEqual(r_in_lv0));
}

// tests if residual works if v_gh > 1 or r_gh > 1 or f_gh > 1
TEST_CASE("residual gh > 1")
{
    int m = 16;
    int n = 16;
    int o = 16;
    int ghm_v = 1;
    int ghn_v = GENERATE(1, 2);
    int gho_v = GENERATE(1, 2);
    int ghm_r = GENERATE(1, 2);
    int ghn_r = 2;
    int gho_r = GENERATE(1, 2);
    int ghm_f = GENERATE(1, 2);
    int ghn_f = GENERATE(1, 2);
    int gho_f = 1;

    double h = 1.0 / ((double)m);
    mgcl::MGCL_STENCIL stencilType = mgcl::MGCL_LAPLACE_27POINT;
    double stencilFactor = 1.0 / (30.0 * h * h);
    mgcl::MGCL_RESIDUAL_NORM resnorm = mgcl::MGCL_L2;

    // v, r and f with extended ghosts, i.e. no ghost update between iterations
    mgcl::Cuboid v_in_gh(m, n, o, ghm_v, ghn_v, gho_v);
    mgcl::Cuboid r_in_gh(m, n, o, ghm_r, ghn_r, gho_r);
    mgcl::Cuboid f_in_gh(m, n, o, ghm_f, ghn_f, gho_f);

    // v and r with gh = 1, i.e. regular ghost update between iterations
    mgcl::Cuboid v_in(m, n, o, 1, 1, 1);
    mgcl::Cuboid r_in(m, n, o, 1, 1, 1);
    mgcl::Cuboid f_in(m, n, o, 1, 1, 1);

    v_in.fillRandom(-10, 10);
    f_in.fillRandom(-10, 10);
    mgcl::MultigridEngine::updateGhostsSeq(v_in);
    mgcl::MultigridEngine::updateGhostsSeq(f_in);

    // copy real cells from v_in to v_in_gh
    v_in_gh.fillRealFrom(v_in);
    f_in_gh.fillRealFrom(f_in);
    mgcl::MultigridEngine::updateGhostsSeq(v_in_gh);
    mgcl::MultigridEngine::updateGhostsSeq(f_in_gh);

    mgcl::VaryingStencil3x3x3 dummy(1, 1, 1, 0, 0, 0);

    SECTION("seq")
    {
        SECTION("Laplace 27p")
        {
            // First calculate exptected result with gh = 1
            double res_exp = mgcl::MultigridEngine::residualSeq(f_in, v_in, r_in, resnorm, stencilType, stencilFactor, dummy, true, true);

            // Now calculate with gh > 1
            double res_act = mgcl::MultigridEngine::residualSeq(f_in_gh, v_in_gh, r_in_gh, resnorm, stencilType, stencilFactor, dummy, true, true);

            REQUIRE_THAT(res_exp, Catch::Matchers::WithinAbs(res_act, 1e-7));
            REQUIRE(v_in.isEqual(v_in_gh));
            REQUIRE(r_in.isEqual(r_in_gh));
        }

        SECTION("Varying 27p")
        {
            mgcl::VaryingStencil3x3x3 sv(m, n, o, 2, 2, 2);
            sv.fillRandom();
            sv.updateGhosts();

            // First calculate exptected result with gh = 1
            double res_exp = mgcl::MultigridEngine::residualSeq(f_in, v_in, r_in, resnorm, mgcl::MGCL_VARYING,
                                                                stencilFactor, sv, true, true);

            // Now calculate with gh > 1
            double res_act = mgcl::MultigridEngine::residualSeq(f_in_gh, v_in_gh, r_in_gh, resnorm, mgcl::MGCL_VARYING,
                                                                stencilFactor, sv, true, true);

            REQUIRE_THAT(res_exp, Catch::Matchers::WithinAbs(res_act, 1e-7));
            REQUIRE(v_in.isEqual(v_in_gh));
            REQUIRE(r_in.isEqual(r_in_gh));
        }
    }

    // TODO test varying stencil
}

// TODO test exceptions

// Tests if residual works if moff, noff or koff < 0, i.e. changing the grid size the residual shall be calculated for.
// TODO off > 0 is not tested yet since it's not needed in practice.
TEST_CASE("residual moff, noff, koff < 0")
{
    int m = 8;
    int n = 8;
    int o = 8;

    double h = 1.0 / ((double)m);
    mgcl::MGCL_STENCIL stencilType = mgcl::MGCL_LAPLACE_27POINT;
    double stencilFactor = 1.0 / (30.0 * h * h);
    mgcl::MGCL_RESIDUAL_NORM resnorm = mgcl::MGCL_L2;

    int moff = -1; // GENERATE(-2,-1,0);
    int noff = -1; // GENERATE(-1,0);
    int ooff = -1; // GENERATE(-1,0);

    // calculate grid sizes for expected result (ghosts = 1)
    int exp_m = m - 2 * moff;
    int exp_n = n - 2 * noff;
    int exp_o = o - 2 * ooff;

    // TODO adjust if tests for off > 0 are added
    int act_ghm = -moff + 1;
    int act_ghn = -noff + 1;
    int act_gho = -ooff + 1;

    // v, r and f with extended ghosts, i.e. no ghost update between iterations
    mgcl::Cuboid v_act(m, n, o, act_ghm, act_ghn, act_gho);
    mgcl::Cuboid r_act(m, n, o, act_ghm, act_ghn, act_gho);
    mgcl::Cuboid f_act(m, n, o, act_ghm, act_ghn, act_gho);

    v_act.fillRandomInt(-10, 10);
    f_act.fillRandomInt(-10, 10);
    mgcl::MultigridEngine::updateGhostsSeq(v_act);
    mgcl::MultigridEngine::updateGhostsSeq(f_act);

    // v and r with gh = 1, i.e. regular ghost update between iterations.
    // make this one bigger to include ghosts of the other one, so results should be equal
    mgcl::Cuboid v_exp(exp_m, exp_n, exp_o, 1, 1, 1);
    mgcl::Cuboid r_exp(exp_m, exp_n, exp_o, 1, 1, 1);
    mgcl::Cuboid f_exp(exp_m, exp_n, exp_o, 1, 1, 1);

    // Dimensions must fit
    REQUIRE(v_exp.getMgh() == v_act.getMgh());
    REQUIRE(v_exp.getNgh() == v_act.getNgh());
    REQUIRE(v_exp.getOgh() == v_act.getOgh());
    REQUIRE(r_exp.getMgh() == r_act.getMgh());
    REQUIRE(r_exp.getNgh() == r_act.getNgh());
    REQUIRE(r_exp.getOgh() == r_act.getOgh());
    REQUIRE(f_exp.getMgh() == f_act.getMgh());
    REQUIRE(f_exp.getNgh() == f_act.getNgh());
    REQUIRE(f_exp.getOgh() == f_act.getOgh());

    // copy all cells from extended ghost variants to standard variants for inputs v and f
    v_act.fillAllFrom(v_exp);
    f_act.fillAllFrom(f_exp);

    mgcl::VaryingStencil3x3x3 dummy(1, 1, 1, 0, 0, 0);

    SECTION("seq")
    {
        SECTION("throwing")
        {
            REQUIRE_THROWS(mgcl::MultigridEngine::residualSeq(f_act, v_act, r_act, resnorm, stencilType,
                                                              stencilFactor, dummy, true, true, -50, noff, ooff));
            REQUIRE_THROWS(mgcl::MultigridEngine::residualSeq(f_act, v_act, r_act, resnorm, stencilType,
                                                              stencilFactor, dummy, true, true, 50, noff, ooff));
            REQUIRE_THROWS(mgcl::MultigridEngine::residualSeq(f_act, v_act, r_act, resnorm, stencilType,
                                                              stencilFactor, dummy, true, true, moff, -50, ooff));
            REQUIRE_THROWS(mgcl::MultigridEngine::residualSeq(f_act, v_act, r_act, resnorm, stencilType,
                                                              stencilFactor, dummy, true, true, moff, 50, ooff));
            REQUIRE_THROWS(mgcl::MultigridEngine::residualSeq(f_act, v_act, r_act, resnorm, stencilType,
                                                              stencilFactor, dummy, true, true, moff, -50, ooff));
            REQUIRE_THROWS(mgcl::MultigridEngine::residualSeq(f_act, v_act, r_act, resnorm, stencilType,
                                                              stencilFactor, dummy, true, true, moff, 50, ooff));
        }

        SECTION("success")
        {
            // First calculate exptected result with gh = 1, off = 0
            double res_exp = mgcl::MultigridEngine::residualSeq(f_exp, v_exp, r_exp, resnorm, stencilType, stencilFactor,
                                                                dummy, true, true);

            // Now calculate with gh > 1 and off < 0
            double res_act = mgcl::MultigridEngine::residualSeq(f_act, v_act, r_act, resnorm, stencilType,
                                                                stencilFactor, dummy, true, true, moff, noff, ooff);

            REQUIRE_THAT(res_exp, Catch::Matchers::WithinAbs(res_act, 1e-7));
            REQUIRE(r_act.isEqualAllCells(r_exp));
        }
    }

    // TODO test varying stencil
}
