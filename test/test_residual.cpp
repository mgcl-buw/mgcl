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
