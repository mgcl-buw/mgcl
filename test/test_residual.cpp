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
        double res = mgcl::MultigridEngine::residualSeq(*c_in_f, *c_in_v, *c_in_r, mgcl::MGCL_L2,
                                                        mgcl::MGCL_LAPLACE_7POINT, stencilFactor, nullptr, true, true,
                                                        true);

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
        auto d_in_f = std::make_shared<mgcl::CuboidGpu>(tu.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, c_in_f);
        auto d_in_v = std::make_shared<mgcl::CuboidGpu>(tu.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, c_in_v);
        auto d_in_r = std::make_shared<mgcl::CuboidGpu>(tu.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, c_in_r);

        mgcl::Level level(p.get(), 0);
        level.setDF(d_in_f);
        level.setDVIn(d_in_v);
        level.setDR(d_in_r);

        double res = mgcl::MultigridEngine::residual(*p, level, true);
        tu.finish();

        auto c_r_out = d_in_r->read(tu.getCommands(), nullptr, true);

        CHECK(fabs(res - 3.00209960095333271e+07) < 1e-7);
        REQUIRE(c_r_out->isEqual(*c_expected_out_r));
    }

    SECTION("residualSeq L2-norm 7point varying stencil periodic")
    {
        int n = 16;
        auto vals = mgcl::VaryingStencil(n, n, n, 3, 1, 1, 1);

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
                                                        mgcl::MGCL_VARYING, stencilFactor, &vals, true, true,
                                                        true);

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
    auto& level0_seq = p_seq->getLevelAt(0);

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
    auto& level0_gpu = p_gpu->getLevelAt(0);

    auto& v_in_lv0 = level0_seq.getV();
    auto& f_in_lv0 = level0_seq.getF();
    auto& r_in_lv0 = level0_seq.getR();

    mgcl_test::TestUtility tu(p_gpu);

    mgcl::MultigridEngine::updateGhosts(*p_gpu, level0_gpu.getDVIn(),
                                        level0_gpu.getMgh(), level0_gpu.getNgh(), level0_gpu.getOgh(), 1, 1, 1, nullptr,
                                        true);
    tu.finish();
    mgcl::MultigridEngine::updateGhostsSeq(v_in_lv0, nullptr, true, false);

    // make sure input is equal
    auto c_r_in = level0_gpu.getDR().read(tu.getCommands(), nullptr, true);
    auto c_v_in = level0_gpu.getDVIn().read(tu.getCommands(), nullptr, true);
    auto c_f_in = level0_gpu.getDF().read(tu.getCommands(), nullptr, true);
    tu.finish();
    REQUIRE(c_r_in->isEqual(r_in_lv0));
    REQUIRE(c_v_in->isEqual(v_in_lv0));
    REQUIRE(c_f_in->isEqual(f_in_lv0));

    double res_gpu = mgcl::MultigridEngine::residual(*p_gpu, level0_gpu, true);
    tu.finish();

    double res_seq = mgcl::MultigridEngine::residualSeq(f_in_lv0, v_in_lv0, r_in_lv0, resnorm,
                                                        stencilType, level0_seq.getStencilFactor(), nullptr, true, true,
                                                        true);

    auto c_r_out = level0_gpu.getDR().read(tu.getCommands(), nullptr, true);
    auto c_v_out = level0_gpu.getDVIn().read(tu.getCommands(), nullptr, true);
    auto c_f_out = level0_gpu.getDF().read(tu.getCommands(), nullptr, true);

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
    auto& sv = p_seq->getStencilValues();
    sv->fillRandomInt();

    p_seq->init();
    auto& level0_seq = p_seq->getLevelAt(0);
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
    auto& sv_gpu = p_gpu->getStencilValues();

    // copy stencil values
    REQUIRE(sv->field1d().size() == sv_gpu->field1d().size());
    for (int i = 0; i < sv->field1d().size(); i++)
        sv_gpu->field1d()[i] = sv->field1d()[i];

    p_gpu->init();
    auto& level0_gpu = p_gpu->getLevelAt(0);
    REQUIRE(level0_gpu.getStencilValuesGpu() != nullptr);

    auto& v_in_lv0 = level0_seq.getV();
    auto& f_in_lv0 = level0_seq.getF();
    auto& r_in_lv0 = level0_seq.getR();
    auto& sv_in_lv0 = level0_seq.getStencilValues();

    mgcl_test::TestUtility tu(p_gpu);

    mgcl::MultigridEngine::updateGhosts(*p_gpu, level0_gpu.getDVIn(),
                                        level0_gpu.getMgh(), level0_gpu.getNgh(), level0_gpu.getOgh(), 1, 1, 1, nullptr,
                                        true);
    tu.finish();
    mgcl::MultigridEngine::updateGhostsSeq(v_in_lv0, nullptr, true, false);

    // make sure input is equal
    auto c_r_in = level0_gpu.getDR().read(tu.getCommands(), nullptr, true);
    auto c_v_in = level0_gpu.getDVIn().read(tu.getCommands(), nullptr, true);
    auto c_f_in = level0_gpu.getDF().read(tu.getCommands(), nullptr, true);
    auto c_sv_in = level0_gpu.getStencilValuesGpu()->read(tu.getCommands());
    tu.finish();

    REQUIRE(c_r_in->isEqual(r_in_lv0));
    REQUIRE(c_v_in->isEqual(v_in_lv0));
    REQUIRE(c_f_in->isEqual(f_in_lv0));
    REQUIRE(c_sv_in.isEqual(*sv_in_lv0));

    double res_gpu = mgcl::MultigridEngine::residual(*p_gpu, level0_gpu, true);
    tu.finish();
    double res_seq = mgcl::MultigridEngine::residualSeq(f_in_lv0, v_in_lv0, r_in_lv0, resnorm,
                                                        stencilType, level0_seq.getStencilFactor(), sv_in_lv0.get(),
                                                        true, true, true);

    auto c_r_out = level0_gpu.getDR().read(tu.getCommands(), nullptr, true);
    auto c_v_out = level0_gpu.getDVIn().read(tu.getCommands(), nullptr, true);
    auto c_f_out = level0_gpu.getDF().read(tu.getCommands(), nullptr, true);
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

// tests if residual seq works if v_gh > 1 or r_gh > 1 or f_gh > 1
TEST_CASE("residual seq gh > 1")
{
    int m = 16;
    int n = 16;
    int o = 16;
    int ghm_v = 2;
    int ghn_v = 3;
    int gho_v = 4;
    int ghm_r = 3;
    int ghn_r = 2;
    int gho_r = 4;
    int ghm_f = 4;
    int ghn_f = 3;
    int gho_f = 2;

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
    mgcl::MultigridEngine::updateGhostsSeq(v_in, nullptr, true, false);
    mgcl::MultigridEngine::updateGhostsSeq(f_in, nullptr, true, false);

    // copy real cells from v_in to v_in_gh
    v_in_gh.fillRealFrom(v_in);
    f_in_gh.fillRealFrom(f_in);
    mgcl::MultigridEngine::updateGhostsSeq(v_in_gh, nullptr, true, false);
    mgcl::MultigridEngine::updateGhostsSeq(f_in_gh, nullptr, true, false);

    SECTION("Laplace 27p")
    {
        // First calculate exptected result with gh = 1
        double res_exp = mgcl::MultigridEngine::residualSeq(f_in, v_in, r_in, resnorm, stencilType, stencilFactor,
                                                            nullptr, true, true, true);

        // Now calculate with gh > 1
        double res_act = mgcl::MultigridEngine::residualSeq(f_in_gh, v_in_gh, r_in_gh, resnorm, stencilType,
                                                            stencilFactor, nullptr, true, true, true);

        REQUIRE_THAT(res_exp, Catch::Matchers::WithinAbs(res_act, 1e-7));
        REQUIRE(v_in.isEqual(v_in_gh));
        REQUIRE(r_in.isEqual(r_in_gh));
    }

    SECTION("Varying 27p")
    {
        mgcl::VaryingStencil sv(m, n, o, 3, 2, 2, 2);
        sv.fillRandom();
        sv.updateGhosts();

        // First calculate exptected result with gh = 1
        double res_exp = mgcl::MultigridEngine::residualSeq(f_in, v_in, r_in, resnorm, mgcl::MGCL_VARYING,
                                                            stencilFactor, &sv, true, true, true);

        // Now calculate with gh > 1
        double res_act = mgcl::MultigridEngine::residualSeq(f_in_gh, v_in_gh, r_in_gh, resnorm, mgcl::MGCL_VARYING,
                                                            stencilFactor, &sv, true, true, true);

        REQUIRE_THAT(res_exp, Catch::Matchers::WithinAbs(res_act, 1e-7));
        REQUIRE(v_in.isEqual(v_in_gh));
        REQUIRE(r_in.isEqual(r_in_gh));
    }
}

// tests if residual gpu works if v_gh > 1 or r_gh > 1 or f_gh > 1
TEST_CASE("residual gpu gh > 1")
{
    int m = 16;
    int n = 16;
    int o = 16;

    // TODO currently ocl version of residual only supports one ghost cell amount for all m,n,o and v,f,r. Maybe adjust.
    // int ghm_v = 2;
    // int ghn_v = 3;
    // int gho_v = 4;
    // int ghm_r = 3;
    // int ghn_r = 2;
    // int gho_r = 4;
    // int ghm_f = 4;
    // int ghn_f = 3;
    // int gho_f = 2;
    int gh = 2;

    double h = 1.0 / ((double)m);
    mgcl::MGCL_STENCIL stencilType = mgcl::MGCL_LAPLACE_27POINT;
    double stencilFactor = 1.0 / (30.0 * h * h);
    mgcl::MGCL_RESIDUAL_NORM resnorm = mgcl::MGCL_L2;

    // v, r and f with extended ghosts, i.e. no ghost update between iterations
    mgcl::Cuboid v_in_gh(m, n, o, gh, gh, gh);
    mgcl::Cuboid r_in_gh(m, n, o, gh, gh, gh);
    mgcl::Cuboid f_in_gh(m, n, o, gh, gh, gh);

    // v and r with gh = 1, i.e. regular ghost update between iterations
    mgcl::Cuboid v_in(m, n, o, 1, 1, 1);
    mgcl::Cuboid r_in(m, n, o, 1, 1, 1);
    mgcl::Cuboid f_in(m, n, o, 1, 1, 1);

    v_in.fillRandom(-10, 10);
    f_in.fillRandom(-10, 10);
    mgcl::MultigridEngine::updateGhostsSeq(v_in, nullptr, true, true);
    mgcl::MultigridEngine::updateGhostsSeq(f_in, nullptr, true, true);

    // copy real cells from v_in to v_in_gh
    v_in_gh.fillRealFrom(v_in);
    f_in_gh.fillRealFrom(f_in);
    mgcl::MultigridEngine::updateGhostsSeq(v_in_gh, nullptr, true, true);
    mgcl::MultigridEngine::updateGhostsSeq(f_in_gh, nullptr, true, true);

    mgcl::VaryingStencil dummy(1, 1, 1, 3, 0, 0, 0);

    if (mgcl_test::TestUtility::deviceAvailable("", CL_DEVICE_TYPE_GPU))
    {
        // init problem with ghosts = 1
        auto p = std::make_shared<mgcl::Problem>(m, n, o);
        p->setResidualNorm(mgcl::MGCL_L2);
        p->setGhosts(1);
        p->setDeviceType(CL_DEVICE_TYPE_GPU);

        mgcl_test::TestUtility tu(p);
        auto d_in_f = std::make_shared<mgcl::CuboidGpu>(tu.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, f_in);
        auto d_in_v = std::make_shared<mgcl::CuboidGpu>(tu.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, v_in);
        auto d_in_r = std::make_shared<mgcl::CuboidGpu>(tu.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, r_in);

        mgcl::Level level(p.get(), 0);
        level.setDF(d_in_f);
        level.setDVIn(d_in_v);
        level.setDR(d_in_r);

        // init problem with ghosts = gh
        auto pgh = std::make_shared<mgcl::Problem>(m, n, o);
        pgh->setResidualNorm(mgcl::MGCL_L2);
        pgh->setGhosts(gh);
        pgh->setDeviceType(CL_DEVICE_TYPE_GPU);

        mgcl_test::TestUtility tu_gh(pgh);
        auto d_in_f_gh = std::make_shared<mgcl::CuboidGpu>(tu_gh.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, f_in_gh);
        auto d_in_v_gh = std::make_shared<mgcl::CuboidGpu>(tu_gh.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, v_in_gh);
        auto d_in_r_gh = std::make_shared<mgcl::CuboidGpu>(tu_gh.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, r_in_gh);

        mgcl::Level level_gh(pgh.get(), 0);
        level_gh.setDF(d_in_f_gh);
        level_gh.setDVIn(d_in_v_gh);
        level_gh.setDR(d_in_r_gh);

        // Make sure input is equal.
        REQUIRE(v_in.isEqual(v_in_gh));
        REQUIRE(r_in.isEqual(r_in_gh));
        REQUIRE(f_in.isEqual(f_in_gh));

        SECTION("Laplace 7p")
        {
            p->setStencilType(mgcl::MGCL_LAPLACE_7POINT);
            pgh->setStencilType(mgcl::MGCL_LAPLACE_7POINT);

            // First calculate exptected result with gh = 1
            double res_exp = mgcl::MultigridEngine::residual(*p, level, true);
            tu.finish();

            // Now calculate with gh > 1
            double res_act = mgcl::MultigridEngine::residual(*pgh, level_gh, true);
            tu_gh.finish();

            auto r_out = d_in_r->read(tu.getCommands(), nullptr, true);
            auto r_out_gh = d_in_r_gh->read(tu_gh.getCommands(), nullptr, true);

            // r_out->dumpToFile("/home/simon/tmp/r_out.txt", true);
            // r_out_gh->dumpToFile("/home/simon/tmp/r_out_gh.txt", true);

            REQUIRE_THAT(res_exp, Catch::Matchers::WithinAbs(res_act, 1e-7));
            REQUIRE(r_out->isEqual(*r_out_gh));
        }

        SECTION("Laplace 19p")
        {
            p->setStencilType(mgcl::MGCL_LAPLACE_7POINT);
            pgh->setStencilType(mgcl::MGCL_LAPLACE_7POINT);

            // First calculate exptected result with gh = 1
            double res_exp = mgcl::MultigridEngine::residual(*p, level, true);
            tu.finish();

            // Now calculate with gh > 1
            double res_act = mgcl::MultigridEngine::residual(*pgh, level_gh, true);
            tu_gh.finish();

            auto r_out = d_in_r->read(tu.getCommands(), nullptr, true);
            auto r_out_gh = d_in_r_gh->read(tu_gh.getCommands(), nullptr, true);

            // r_out->dumpToFile("/home/simon/tmp/r_out.txt", true);
            // r_out_gh->dumpToFile("/home/simon/tmp/r_out_gh.txt", true);

            REQUIRE_THAT(res_exp, Catch::Matchers::WithinAbs(res_act, 1e-7));
            REQUIRE(r_out->isEqual(*r_out_gh));
        }

        SECTION("Laplace 27p")
        {
            p->setStencilType(mgcl::MGCL_LAPLACE_27POINT);
            pgh->setStencilType(mgcl::MGCL_LAPLACE_27POINT);

            // First calculate exptected result with gh = 1
            double res_exp = mgcl::MultigridEngine::residual(*p, level, true);
            tu.finish();

            // Now calculate with gh > 1
            double res_act = mgcl::MultigridEngine::residual(*pgh, level_gh, true);
            tu_gh.finish();

            auto r_out = d_in_r->read(tu.getCommands(), nullptr, true);
            auto r_out_gh = d_in_r_gh->read(tu_gh.getCommands(), nullptr, true);

            // r_out->dumpToFile("/home/simon/tmp/r_out.txt", true);
            // r_out_gh->dumpToFile("/home/simon/tmp/r_out_gh.txt", true);

            REQUIRE_THAT(res_exp, Catch::Matchers::WithinAbs(res_act, 1e-7));
            REQUIRE(r_out->isEqual(*r_out_gh));
        }

        SECTION("Varying 27p")
        {
            p->setStencilType(mgcl::MGCL_VARYING);
            pgh->setStencilType(mgcl::MGCL_VARYING);

            auto sv = p->getStencilValues();
            sv->fillRandom();
            sv->updateGhosts();

            auto d_in_sv = std::make_shared<mgcl::VaryingStencilGpu>(sv->getDim1(), sv->getDim2(), sv->getDim3(), 3, sv->getGhostsDim1(),
                                                                     tu.getContext(), tu.getCommands());
            auto d_in_sv_gh = std::make_shared<mgcl::VaryingStencilGpu>(sv->getDim1(), sv->getDim2(), sv->getDim3(), 3, sv->getGhostsDim1(),
                                                                        tu_gh.getContext(), tu_gh.getCommands());

            d_in_sv->fill(*sv, tu.getCommands());
            d_in_sv_gh->fill(*sv, tu_gh.getCommands());

            level.setStencilValuesGpu(d_in_sv);
            level_gh.setStencilValuesGpu(d_in_sv_gh);

            // First calculate exptected result with gh = 1
            double res_exp = mgcl::MultigridEngine::residual(*p, level, true);
            tu.finish();

            // Now calculate with gh > 1
            double res_act = mgcl::MultigridEngine::residual(*pgh, level_gh, true);
            tu_gh.finish();

            auto r_out = d_in_r->read(tu.getCommands(), nullptr, true);
            auto r_out_gh = d_in_r_gh->read(tu_gh.getCommands(), nullptr, true);

            // r_out->dumpToFile("/home/simon/tmp/r_out.txt", true);
            // r_out_gh->dumpToFile("/home/simon/tmp/r_out_gh.txt", true);

            REQUIRE_THAT(res_exp, Catch::Matchers::WithinAbs(res_act, 1e-7));
            REQUIRE(r_out->isEqual(*r_out_gh));
        }
    }
}

// Test if exceptions are thrown for wrong parameters in residual. Only needed for periodic case.
// TODO check Dirichlet MPI
TEST_CASE("residual throwing")
{
    int m = 16;
    int n = 16;
    int o = 16;

    double omega = 0.8;
    double h = 1.0 / ((double)m);
    mgcl::MGCL_STENCIL stencilType = mgcl::MGCL_LAPLACE_27POINT;
    double stencilFactor = 1.0 / (30.0 * h * h);
    mgcl::MGCL_RESIDUAL_NORM resnorm = mgcl::MGCL_L2;

    int moff = -1;
    int noff = moff;
    int ooff = moff;

    // Ghost cell amounts needed for off
    int act_ghm = -moff + 1;
    int act_ghn = -noff + 1;
    int act_gho = -ooff + 1;

    // Create ghosted fields that are ok and fields with too little ghost cells
    mgcl::Cuboid v_gh(m, n, o, act_ghm, act_ghn, act_gho);
    mgcl::Cuboid r_gh(m, n, o, act_ghm, act_ghn, act_gho);
    mgcl::Cuboid f_gh(m, n, o, act_ghm, act_ghn, act_gho);

    mgcl::Cuboid v(m, n, o, 1, 1, 1);
    mgcl::Cuboid r(m, n, o, 1, 1, 1);
    mgcl::Cuboid f(m, n, o, 1, 1, 1);

    SECTION("seq")
    {
        SECTION("off too little or too large")
        {
            REQUIRE_THROWS(mgcl::MultigridEngine::residualSeq(f_gh, v_gh, r_gh, resnorm, stencilType,
                                                              stencilFactor, nullptr, true, true, true, -50, noff, ooff));
            REQUIRE_THROWS(mgcl::MultigridEngine::residualSeq(f_gh, v_gh, r_gh, resnorm, stencilType,
                                                              stencilFactor, nullptr, true, true, true, 50, noff, ooff));
            REQUIRE_THROWS(mgcl::MultigridEngine::residualSeq(f_gh, v_gh, r_gh, resnorm, stencilType,
                                                              stencilFactor, nullptr, true, true, true, moff, -50, ooff));
            REQUIRE_THROWS(mgcl::MultigridEngine::residualSeq(f_gh, v_gh, r_gh, resnorm, stencilType,
                                                              stencilFactor, nullptr, true, true, true, moff, 50, ooff));
            REQUIRE_THROWS(mgcl::MultigridEngine::residualSeq(f_gh, v_gh, r_gh, resnorm, stencilType,
                                                              stencilFactor, nullptr, true, true, true, moff, -50, ooff));
            REQUIRE_THROWS(mgcl::MultigridEngine::residualSeq(f_gh, v_gh, r_gh, resnorm, stencilType,
                                                              stencilFactor, nullptr, true, true, true, moff, 50, ooff));
        }

        SECTION("stencilValues null and stencilType varying")
        {
            REQUIRE_THROWS(mgcl::MultigridEngine::residualSeq(f_gh, v_gh, r_gh, resnorm, mgcl::MGCL_VARYING,
                                                              stencilFactor, nullptr, true, true, moff, noff, ooff));
        }
    }

    SECTION("gpu")
    {
        if (mgcl_test::TestUtility::deviceAvailable("", CL_DEVICE_TYPE_GPU))
        {
            // init problem with ghosts = 1
            auto p_exp = std::make_shared<mgcl::Problem>(m, n, o);
            p_exp->setGhosts(1);
            p_exp->setDeviceType(CL_DEVICE_TYPE_GPU);

            mgcl_test::TestUtility tu_exp(p_exp);
            auto d_f_exp = std::make_shared<mgcl::CuboidGpu>(tu_exp.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, f);
            auto d_v_exp = std::make_shared<mgcl::CuboidGpu>(tu_exp.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, v);
            auto d_v_out_exp = std::make_shared<mgcl::CuboidGpu>(tu_exp.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, v);
            auto d_r_exp = std::make_shared<mgcl::CuboidGpu>(tu_exp.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, r);

            mgcl::Level level_exp(p_exp.get(), 0);
            level_exp.setDF(d_f_exp);
            level_exp.setDVIn(d_v_exp);
            level_exp.setDVOut(d_v_out_exp);
            level_exp.setDR(d_r_exp);

            SECTION("off too little or too large")
            {
                REQUIRE_THROWS(mgcl::MultigridEngine::residual(*p_exp, level_exp, false, 50, noff, ooff));
                REQUIRE_THROWS(mgcl::MultigridEngine::residual(*p_exp, level_exp, false, -50, noff, ooff));
                REQUIRE_THROWS(mgcl::MultigridEngine::residual(*p_exp, level_exp, false, moff, 50, ooff));
                REQUIRE_THROWS(mgcl::MultigridEngine::residual(*p_exp, level_exp, false, moff, -50, ooff));
                REQUIRE_THROWS(mgcl::MultigridEngine::residual(*p_exp, level_exp, false, moff, noff, 50));
                REQUIRE_THROWS(mgcl::MultigridEngine::residual(*p_exp, level_exp, false, moff, noff, -50));
            }
        }
    }
}

// Tests if residual works if moff, noff or koff < 0, i.e. changing the grid size the residual shall be calculated for.
// TODO off > 0 is not tested yet since it's not needed in practice.
TEST_CASE("residual seq moff, noff, koff < 0")
{
    int m = 8;
    int n = 8;
    int o = 8;

    double h = 1.0 / ((double)m);
    mgcl::MGCL_STENCIL stencilType = mgcl::MGCL_LAPLACE_27POINT;
    double stencilFactor = 1.0 / (30.0 * h * h);
    mgcl::MGCL_RESIDUAL_NORM resnorm = mgcl::MGCL_L2;

    int moff = GENERATE(-2, -1, 0);
    int noff = GENERATE(-1, 0);
    int ooff = GENERATE(-1, 0);

    // calculate grid sizes for expected result (ghosts = 1)
    int exp_m = m - 2 * moff;
    int exp_n = n - 2 * noff;
    int exp_o = o - 2 * ooff;

    // TODO adjust if tests for off > 0 are added
    int act_ghm = -moff + 1;
    int act_ghn = -noff + 1;
    int act_gho = -ooff + 1;

    CAPTURE(moff, noff, ooff, exp_m, exp_n, exp_o, act_ghm, act_ghn, act_gho);

    // v, r and f with extended ghosts, i.e. no ghost update between iterations
    mgcl::Cuboid v_act(m, n, o, act_ghm, act_ghn, act_gho);
    mgcl::Cuboid r_act(m, n, o, act_ghm, act_ghn, act_gho);
    mgcl::Cuboid f_act(m, n, o, act_ghm, act_ghn, act_gho);

    v_act.fillRandomInt(-10, 10);
    f_act.fillRandomInt(-10, 10);
    mgcl::MultigridEngine::updateGhostsSeq(v_act, nullptr, true, true);
    mgcl::MultigridEngine::updateGhostsSeq(f_act, nullptr, true, true);

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

    SECTION("Laplace 27p")
    {
        // First calculate exptected result with gh = 1, off = 0
        double res_exp = mgcl::MultigridEngine::residualSeq(f_exp, v_exp, r_exp, resnorm, stencilType, stencilFactor,
                                                            nullptr, true, true, true);

        // Now calculate with gh > 1 and off < 0
        double res_act = mgcl::MultigridEngine::residualSeq(f_act, v_act, r_act, resnorm, stencilType,
                                                            stencilFactor, nullptr, true, true, true, moff, noff, ooff);

        REQUIRE_THAT(res_exp, Catch::Matchers::WithinAbs(res_act, 1e-7));
        REQUIRE(r_act.isEqualAllCells(r_exp));
    }

    SECTION("Varying 27p")
    {
        mgcl::VaryingStencil sv(exp_m, exp_n, exp_o, 3, 2, 2, 2);
        sv.fillRandom();
        sv.updateGhosts();

        // First calculate exptected result with gh = 1
        double res_exp = mgcl::MultigridEngine::residualSeq(f_exp, v_exp, r_exp, resnorm, mgcl::MGCL_VARYING,
                                                            stencilFactor, &sv, true, true, true);

        // Now calculate with gh > 1
        double res_act = mgcl::MultigridEngine::residualSeq(f_act, v_act, r_act, resnorm, mgcl::MGCL_VARYING,
                                                            stencilFactor, &sv, true, true, true, moff, noff, ooff);

        REQUIRE_THAT(res_exp, Catch::Matchers::WithinAbs(res_act, 1e-7));
        REQUIRE(r_act.isEqualAllCells(r_exp));
    }
}

// TODO adjust for gpu
// Tests if residual works if moff, noff or koff < 0, i.e. changing the grid size the residual shall be calculated for.
// TODO off > 0 is not tested yet since it's not needed in practice.
TEST_CASE("residual gpu moff, noff, koff < 0")
{
    int m = 8;
    int n = 8;
    int o = 8;

    double h = 1.0 / ((double)m);
    mgcl::MGCL_STENCIL stencilType = mgcl::MGCL_LAPLACE_27POINT;
    double stencilFactor = 1.0 / (30.0 * h * h);
    mgcl::MGCL_RESIDUAL_NORM resnorm = mgcl::MGCL_L2;

    // Must be equal for now since gpu residual does not support varying ghosts for different dimensions yet.
    int moff = -1;   // GENERATE(-2,-1,0);
    int noff = moff; // -1; // GENERATE(-1,0);
    int ooff = moff; // -1; // GENERATE(-1,0);

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
    mgcl::MultigridEngine::updateGhostsSeq(v_act, nullptr, true, true);
    mgcl::MultigridEngine::updateGhostsSeq(f_act, nullptr, true, true);

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

    mgcl::VaryingStencil dummy(1, 1, 1, 3, 0, 0, 0);

    if (mgcl_test::TestUtility::deviceAvailable("", CL_DEVICE_TYPE_GPU))
    {
        // init problem with ghosts = 1
        auto p_exp = std::make_shared<mgcl::Problem>(m, n, o);
        p_exp->setResidualNorm(mgcl::MGCL_L2);
        p_exp->setGhosts(1);
        p_exp->setDeviceType(CL_DEVICE_TYPE_GPU);

        mgcl_test::TestUtility tu_exp(p_exp);
        auto d_f_exp = std::make_shared<mgcl::CuboidGpu>(tu_exp.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, f_exp);
        auto d_v_exp = std::make_shared<mgcl::CuboidGpu>(tu_exp.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, v_exp);
        auto d_r_exp = std::make_shared<mgcl::CuboidGpu>(tu_exp.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, r_exp);

        mgcl::Level level_exp(p_exp.get(), 0);
        level_exp.setDF(d_f_exp);
        level_exp.setDVIn(d_v_exp);
        level_exp.setDR(d_r_exp);

        // init problem with ghosts = act_ghm
        auto p_act = std::make_shared<mgcl::Problem>(m, n, o);
        p_act->setResidualNorm(mgcl::MGCL_L2);
        p_act->setGhosts(act_ghm);
        p_act->setDeviceType(CL_DEVICE_TYPE_GPU);

        mgcl_test::TestUtility tu_act(p_act);
        auto d_f_act = std::make_shared<mgcl::CuboidGpu>(tu_act.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, f_act);
        auto d_v_act = std::make_shared<mgcl::CuboidGpu>(tu_act.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, v_act);
        auto d_r_act = std::make_shared<mgcl::CuboidGpu>(tu_act.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, r_act);

        mgcl::Level level_act(p_act.get(), 0);
        level_act.setDF(d_f_act);
        level_act.setDVIn(d_v_act);
        level_act.setDR(d_r_act);

        // Make sure input is equal.
        REQUIRE(v_exp.isEqualAllCells(v_act));
        REQUIRE(r_exp.isEqualAllCells(r_act));
        REQUIRE(f_exp.isEqualAllCells(f_act));

        SECTION("Laplace 7p")
        {
            p_exp->setStencilType(mgcl::MGCL_LAPLACE_7POINT);
            p_act->setStencilType(mgcl::MGCL_LAPLACE_7POINT);

            // First calculate exptected result with gh = 1, off = 0
            double res_exp = mgcl::MultigridEngine::residual(*p_exp, level_exp, true);
            tu_exp.finish();

            // Now calculate with gh > 1 and off < 0
            double res_act = mgcl::MultigridEngine::residual(*p_act, level_act, true, moff, noff, ooff);
            tu_act.finish();

            auto r_out_exp = d_r_exp->read(tu_exp.getCommands(), nullptr, true);
            auto r_out_act = d_r_act->read(tu_act.getCommands(), nullptr, true);

            REQUIRE_THAT(res_exp, Catch::Matchers::WithinAbs(res_act, 1e-7));
            REQUIRE(r_out_act->isEqualAllCells(*r_out_exp));
        }

        SECTION("Laplace 19p")
        {
            p_exp->setStencilType(mgcl::MGCL_LAPLACE_19POINT);
            p_act->setStencilType(mgcl::MGCL_LAPLACE_19POINT);

            // First calculate exptected result with gh = 1, off = 0
            double res_exp = mgcl::MultigridEngine::residual(*p_exp, level_exp, true);
            tu_exp.finish();

            // Now calculate with gh > 1 and off < 0
            double res_act = mgcl::MultigridEngine::residual(*p_act, level_act, true, moff, noff, ooff);
            tu_act.finish();

            auto r_out_exp = d_r_exp->read(tu_exp.getCommands(), nullptr, true);
            auto r_out_act = d_r_act->read(tu_act.getCommands(), nullptr, true);

            REQUIRE_THAT(res_exp, Catch::Matchers::WithinAbs(res_act, 1e-7));
            REQUIRE(r_out_act->isEqualAllCells(*r_out_exp));
        }

        SECTION("Laplace 27p")
        {
            p_exp->setStencilType(mgcl::MGCL_LAPLACE_27POINT);
            p_act->setStencilType(mgcl::MGCL_LAPLACE_27POINT);

            // First calculate exptected result with gh = 1, off = 0
            double res_exp = mgcl::MultigridEngine::residual(*p_exp, level_exp, true);
            tu_exp.finish();

            // Now calculate with gh > 1 and off < 0
            double res_act = mgcl::MultigridEngine::residual(*p_act, level_act, true, moff, noff, ooff);
            tu_act.finish();

            auto r_out_exp = d_r_exp->read(tu_exp.getCommands(), nullptr, true);
            auto r_out_act = d_r_act->read(tu_act.getCommands(), nullptr, true);

            REQUIRE_THAT(res_exp, Catch::Matchers::WithinAbs(res_act, 1e-7));
            REQUIRE(r_out_act->isEqualAllCells(*r_out_exp));
        }

        SECTION("Varying 27p")
        {
            p_exp->setStencilType(mgcl::MGCL_VARYING);
            p_act->setStencilType(mgcl::MGCL_VARYING);

            auto sv_exp = p_exp->getStencilValues();
            sv_exp->fillRandom();
            sv_exp->updateGhosts();

            // copy stencil values
            auto sv_act = p_act->getStencilValues();
            for (int i = 0; i < sv_exp->field1d().size(); i++)
                sv_act->field1d()[i] = sv_exp->field1d()[i];

            auto d_sv_exp = std::make_shared<mgcl::VaryingStencilGpu>(sv_exp->getDim1(), sv_exp->getDim2(), sv_exp->getDim3(), 3,
                                                                      sv_exp->getGhostsDim1(),
                                                                      tu_exp.getContext(), tu_exp.getCommands());
            auto d_sv_act = std::make_shared<mgcl::VaryingStencilGpu>(sv_act->getDim1(), sv_act->getDim2(), sv_act->getDim3(), 3,
                                                                      sv_act->getGhostsDim1(),
                                                                      tu_act.getContext(), tu_act.getCommands());

            d_sv_exp->fill(*sv_exp, tu_exp.getCommands());
            d_sv_act->fill(*sv_act, tu_act.getCommands());

            level_exp.setStencilValuesGpu(d_sv_exp);
            level_act.setStencilValuesGpu(d_sv_act);

            // First calculate exptected result with gh = 1
            double res_exp = mgcl::MultigridEngine::residual(*p_exp, level_exp, true);
            tu_exp.finish();

            // Now calculate with gh > 1
            double res_act = mgcl::MultigridEngine::residual(*p_act, level_act, true, moff, noff, ooff);
            tu_act.finish();

            auto r_out_exp = d_r_exp->read(tu_exp.getCommands(), nullptr, true);
            auto r_out_act = d_r_act->read(tu_act.getCommands(), nullptr, true);

            REQUIRE_THAT(res_exp, Catch::Matchers::WithinAbs(res_act, 1e-7));
            REQUIRE(r_out_act->isEqualAllCells(*r_out_exp));
        }
    }
}