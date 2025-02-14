#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>

#include "../src/mgcl/cuboid.hpp"
#include "../src/mgcl/level.hpp"
#include "../src/mgcl/multigrid_engine.hpp"
#include "cli_args.hpp"
#include "device_type_generator.hpp"
#include "test_results.hpp"
#include "test_utility.hpp"

std::shared_ptr<mgcl::Cuboid> jacobiTestInputF();
std::shared_ptr<mgcl::Cuboid> jacobiTestInputV();
std::shared_ptr<mgcl::Cuboid> jacobiTestInputR();
std::shared_ptr<mgcl::Cuboid> jacobiTestOutputV();
std::shared_ptr<mgcl::Cuboid> jacobiTestOutputR();

TEST_CASE("jacobi")
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
    double omega = 0.8;
    int maxiter = 5;

    auto c_in_f = mgcl_test::test_jacobi::inputF16();
    auto c_in_v = mgcl_test::test_jacobi::inputV16();
    auto c_in_r = mgcl_test::test_jacobi::inputR16();
    auto c_expected_out_v = mgcl_test::test_jacobi::outputV16();
    auto c_expected_out_r = mgcl_test::test_jacobi::outputR16();

    auto stencil = mgcl::MGCL_LAPLACE_7POINT;
    double h = 1.0 / (double)m;
    double stencilFactor = 1.0 / (h * h);

    // TODO adjust expected results for expected r and norm of r?
    SECTION("seq L2-norm 7point periodic")
    {
        double res = mgcl::MultigridEngine::jacobiSeq(*c_in_v, *c_in_f, *c_in_r, omega, h * h, maxiter,
                                                      mgcl::MGCL_L2, mgcl::MGCL_LAPLACE_7POINT, stencilFactor,
                                                      nullptr, nullptr, true, true, true);

        // REQUIRE_THAT(res, Catch::Matchers::WithinAbs(4.02895897954478714e+04, 1e-7));
        CHECK(c_in_v->isEqual(*c_expected_out_v));
        // CHECK(c_in_r->isEqual(*c_expected_out_r));
    }

    // TODO adjust expected results for expected r and norm of r?
    SECTION("OpenCL GPU L2-norm 7point periodic")
    {
        auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));

        auto p = std::make_shared<mgcl::Problem>(m, n, o);
        p->setResidualNorm(mgcl::MGCL_L2);
        p->setStencilType(stencil);
        p->setGhosts(1);
        p->setOmega(omega);
        p->setDeviceType(deviceType);

        mgcl_test::TestUtility tu(p);
        auto d_in_f = std::make_shared<mgcl::CuboidGpu>(tu.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, *c_in_f);
        auto d_in_v = std::make_shared<mgcl::CuboidGpu>(tu.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, *c_in_v);
        auto d_in_v_out = std::make_shared<mgcl::CuboidGpu>(tu.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, *c_in_v);
        auto d_in_r = std::make_shared<mgcl::CuboidGpu>(tu.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, *c_in_r);
        auto d_in_rsq = std::make_shared<mgcl::CuboidGpu>(tu.getContext(), CL_MEM_READ_ONLY, m, n, o, 0, 0, 0);

        mgcl::Level level(p.get(), 0);
        level.setDF(d_in_f);
        level.setDVIn(d_in_v);
        level.setDVOut(d_in_v_out);
        level.setDR(d_in_r);
        level.setDRsq(d_in_rsq);

        double res = mgcl::MultigridEngine::jacobi(*p, level, maxiter, 1);
        tu.finish();

        auto c_r_out = d_in_r->read(p->getCommands(), nullptr, true);
        auto c_v_out = d_in_v->read(p->getCommands(), nullptr, true);

        // REQUIRE_THAT(res, Catch::Matchers::WithinAbs(4.02895897954478714e+04, 1e-7));
        // CHECK(fabs(res - 4.02895897954478714e+04) < 1e-7);
        CHECK(c_v_out->isEqual(*c_expected_out_v));
        // CHECK(c_r_out->isEqual(*c_expected_out_r));
    }

    SECTION("Inf-norm seq vs ocl")
    {
        auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));

        auto p = std::make_shared<mgcl::Problem>(m, n, o);
        p->setResidualNorm(mgcl::MGCL_INF);
        p->setStencilType(stencil);
        p->setGhosts(1);
        p->setOmega(omega);
        p->setDeviceType(deviceType);

        mgcl_test::TestUtility tu(p);
        auto d_in_f = std::make_shared<mgcl::CuboidGpu>(tu.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, *c_in_f);
        auto d_in_v = std::make_shared<mgcl::CuboidGpu>(tu.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, *c_in_v);
        auto d_in_v_out = std::make_shared<mgcl::CuboidGpu>(tu.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, *c_in_v);
        auto d_in_r = std::make_shared<mgcl::CuboidGpu>(tu.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, *c_in_r);

        mgcl::Level level(p.get(), 0);
        level.setDF(d_in_f);
        level.setDVIn(d_in_v);
        level.setDVOut(d_in_v_out);
        level.setDR(d_in_r);

        double res_ocl = mgcl::MultigridEngine::jacobi(*p, level, maxiter, 1);
        tu.finish();

        auto c_r_out_ocl = d_in_r->read(p->getCommands(), nullptr, true);
        auto c_v_out_ocl = d_in_v->read(p->getCommands(), nullptr, true);

        double res_seq = mgcl::MultigridEngine::jacobiSeq(*c_in_v, *c_in_f, *c_in_r, omega, h * h, maxiter,
                                                          mgcl::MGCL_INF, mgcl::MGCL_LAPLACE_7POINT,
                                                          stencilFactor, nullptr, nullptr, true, true, true);

        // c_r_out_ocl->dumpToFile("/home/simon/tmp/c_r_out_ocl.csv");
        // c_in_r->dumpToFile("/home/simon/tmp/c_in_r.csv");

        CHECK(c_in_v->isEqual(*c_v_out_ocl));
        CHECK(c_in_r->isEqual(*c_r_out_ocl));
        REQUIRE_THAT(res_seq, Catch::Matchers::WithinAbs(res_ocl, 1e-7));
    }
}

TEST_CASE("jacobi GPU varying stencil")
{
    // checks 1d indexing inside 2d kernel
    SECTION("indices_2d")
    {
        int mreal = 8;
        int nreal = 8;
        int oreal = 8;
        int ghosts = 1;
        int mgh = 8 + 2 * ghosts;
        int ngh = 8 + 2 * ghosts;
        int ogh = 8 + 2 * ghosts;

        // Set different grid size for stencilValues, which happens when using MPI at
        // the level of mpiLevelTreshold.
        int ghosts_sv = 2;
        int svm = 8;
        int svn = 16;
        int svo = 16;
        int svmgh = svm + 2 * ghosts_sv;
        int svngh = svn + 2 * ghosts_sv;
        int svogh = svo + 2 * ghosts_sv;
        int svGridSize = svmgh * svngh * svogh;

        mgcl::VaryingStencil stencilValues(svm, svn, svo, 3, ghosts_sv, ghosts_sv, ghosts_sv);
        for (int i = 0; i < stencilValues.field1d().size(); i++)
            stencilValues.field1d()[i] = i;

        int idx_start = ghosts;

        for (int j = ghosts, jsv = ghosts_sv; j < ngh - ghosts; j++, jsv++)
            for (int k = ghosts, ksv = ghosts_sv; k < ogh - ghosts; k++, ksv++)
            {
                int ioff = ngh * ogh;
                int joff = ogh;
                int koff = 1;
                int index = ghosts * ioff + j * ogh + k;

                int svno = svngh * svogh;
                // offset inside one coefficient grid that points to the coefficient for the current grid point. Must consider different amount of ghosts for v and sv.
                int index_sv = (idx_start - ghosts + ghosts_sv) * svno + (j - ghosts + ghosts_sv) * svogh + (k - ghosts + ghosts_sv);

                for (int i = idx_start, isv = ghosts_sv; i < mgh - ghosts; i++, isv++)
                {
                    CAPTURE(i, j, k, isv, jsv, ksv);

                    // make sure isv, jsv and ksv are correct (only needed in this test, not in the actual kernel)
                    REQUIRE(isv == (i - ghosts + ghosts_sv));
                    REQUIRE(jsv == (j - ghosts + ghosts_sv));
                    REQUIRE(ksv == (k - ghosts + ghosts_sv));

                    // Check that index_sv is updated correctly after each iteration
                    REQUIRE(index_sv == (i - ghosts + ghosts_sv) * svno + (j - ghosts + ghosts_sv) * svogh + (k - ghosts + ghosts_sv));

                    // Check actual values
                    REQUIRE(stencilValues[1][1][1][isv][jsv][ksv] == stencilValues.field1d()[index_sv + (9 + 3 + 1) * svGridSize]);
                    REQUIRE(stencilValues[1][1][0][isv][jsv][ksv] == stencilValues.field1d()[index_sv + (9 + 3) * svGridSize]);
                    REQUIRE(stencilValues[1][1][2][isv][jsv][ksv] == stencilValues.field1d()[index_sv + (9 + 3 + 2) * svGridSize]);
                    REQUIRE(stencilValues[1][0][1][isv][jsv][ksv] == stencilValues.field1d()[index_sv + (9 + 1) * svGridSize]);
                    REQUIRE(stencilValues[1][2][1][isv][jsv][ksv] == stencilValues.field1d()[index_sv + (9 + 6 + 1) * svGridSize]);
                    REQUIRE(stencilValues[0][1][1][isv][jsv][ksv] == stencilValues.field1d()[index_sv + (3 + 1) * svGridSize]);
                    REQUIRE(stencilValues[2][1][1][isv][jsv][ksv] == stencilValues.field1d()[index_sv + (18 + 3 + 1) * svGridSize]);
                    REQUIRE(stencilValues[1][0][0][isv][jsv][ksv] == stencilValues.field1d()[index_sv + (9) * svGridSize]);
                    REQUIRE(stencilValues[1][0][2][isv][jsv][ksv] == stencilValues.field1d()[index_sv + (9 + 2) * svGridSize]);
                    REQUIRE(stencilValues[1][2][0][isv][jsv][ksv] == stencilValues.field1d()[index_sv + (9 + 6) * svGridSize]);
                    REQUIRE(stencilValues[1][2][2][isv][jsv][ksv] == stencilValues.field1d()[index_sv + (9 + 6 + 2) * svGridSize]);
                    REQUIRE(stencilValues[0][1][0][isv][jsv][ksv] == stencilValues.field1d()[index_sv + (3) * svGridSize]);
                    REQUIRE(stencilValues[0][1][2][isv][jsv][ksv] == stencilValues.field1d()[index_sv + (3 + 2) * svGridSize]);
                    REQUIRE(stencilValues[2][1][0][isv][jsv][ksv] == stencilValues.field1d()[index_sv + (18 + 3) * svGridSize]);
                    REQUIRE(stencilValues[2][1][2][isv][jsv][ksv] == stencilValues.field1d()[index_sv + (18 + 3 + 2) * svGridSize]);
                    REQUIRE(stencilValues[0][0][1][isv][jsv][ksv] == stencilValues.field1d()[index_sv + (1) * svGridSize]);
                    REQUIRE(stencilValues[0][2][1][isv][jsv][ksv] == stencilValues.field1d()[index_sv + (6 + 1) * svGridSize]);
                    REQUIRE(stencilValues[2][0][1][isv][jsv][ksv] == stencilValues.field1d()[index_sv + (18 + 1) * svGridSize]);
                    REQUIRE(stencilValues[2][2][1][isv][jsv][ksv] == stencilValues.field1d()[index_sv + (18 + 6 + 1) * svGridSize]);
                    REQUIRE(stencilValues[0][0][0][isv][jsv][ksv] == stencilValues.field1d()[index_sv]);
                    REQUIRE(stencilValues[0][0][2][isv][jsv][ksv] == stencilValues.field1d()[index_sv + (2) * svGridSize]);
                    REQUIRE(stencilValues[0][2][0][isv][jsv][ksv] == stencilValues.field1d()[index_sv + (6) * svGridSize]);
                    REQUIRE(stencilValues[0][2][2][isv][jsv][ksv] == stencilValues.field1d()[index_sv + (6 + 2) * svGridSize]);
                    REQUIRE(stencilValues[2][0][0][isv][jsv][ksv] == stencilValues.field1d()[index_sv + (18) * svGridSize]);
                    REQUIRE(stencilValues[2][0][2][isv][jsv][ksv] == stencilValues.field1d()[index_sv + (18 + 2) * svGridSize]);
                    REQUIRE(stencilValues[2][2][0][isv][jsv][ksv] == stencilValues.field1d()[index_sv + (18 + 6) * svGridSize]);
                    REQUIRE(stencilValues[2][2][2][isv][jsv][ksv] == stencilValues.field1d()[index_sv + (18 + 6 + 2) * svGridSize]);

                    index_sv += svno;
                }
            }
    }
    // checks 1d indexing inside 3d kernel
    SECTION("indices_3d")
    {
        int mreal = 8;
        int nreal = 8;
        int oreal = 8;
        int ghosts = 1;
        int mgh = 8 + 2 * ghosts;
        int ngh = 8 + 2 * ghosts;
        int ogh = 8 + 2 * ghosts;

        // Set different grid size for stencilValues, which happens when using MPI at
        // the level of mpiLevelTreshold.
        int ghosts_sv = 2;
        int svm = 8;
        int svn = 16;
        int svo = 16;
        int svmgh = svm + 2 * ghosts_sv;
        int svngh = svn + 2 * ghosts_sv;
        int svogh = svo + 2 * ghosts_sv;
        int svGridSize = svmgh * svngh * svogh;

        mgcl::VaryingStencil stencilValues(svm, svn, svo, 3, ghosts_sv, ghosts_sv, ghosts_sv);
        for (int i = 0; i < stencilValues.field1d().size(); i++)
            stencilValues.field1d()[i] = i;

        int idx_start = ghosts;

        for (int i = idx_start, isv = ghosts_sv; i < ngh - ghosts; i++, isv++)
            for (int j = idx_start, jsv = ghosts_sv; j < ngh - ghosts; j++, jsv++)
                for (int k = idx_start, ksv = ghosts_sv; k < ogh - ghosts; k++, ksv++)
                {
                    int ioff = ngh * ogh;
                    int joff = ogh;
                    int koff = 1;
                    int index = ghosts * ioff + j * ogh + k;

                    int svno = svngh * svogh;
                    // offset inside one coefficient grid that points to the coefficient for the current grid point. Must consider different amount of ghosts for v and sv.
                    int index_sv = (i - ghosts + ghosts_sv) * svno + (j - ghosts + ghosts_sv) * svogh + (k - ghosts + ghosts_sv);

                    CAPTURE(i, j, k, isv, jsv, ksv);

                    // make sure isv, jsv and ksv are correct (only needed in this test, not in the actual kernel)
                    REQUIRE(isv == (i - ghosts + ghosts_sv));
                    REQUIRE(jsv == (j - ghosts + ghosts_sv));
                    REQUIRE(ksv == (k - ghosts + ghosts_sv));

                    // Check actual values
                    REQUIRE(stencilValues[1][1][1][isv][jsv][ksv] == stencilValues.field1d()[index_sv + (9 + 3 + 1) * svGridSize]);
                    REQUIRE(stencilValues[1][1][0][isv][jsv][ksv] == stencilValues.field1d()[index_sv + (9 + 3) * svGridSize]);
                    REQUIRE(stencilValues[1][1][2][isv][jsv][ksv] == stencilValues.field1d()[index_sv + (9 + 3 + 2) * svGridSize]);
                    REQUIRE(stencilValues[1][0][1][isv][jsv][ksv] == stencilValues.field1d()[index_sv + (9 + 1) * svGridSize]);
                    REQUIRE(stencilValues[1][2][1][isv][jsv][ksv] == stencilValues.field1d()[index_sv + (9 + 6 + 1) * svGridSize]);
                    REQUIRE(stencilValues[0][1][1][isv][jsv][ksv] == stencilValues.field1d()[index_sv + (3 + 1) * svGridSize]);
                    REQUIRE(stencilValues[2][1][1][isv][jsv][ksv] == stencilValues.field1d()[index_sv + (18 + 3 + 1) * svGridSize]);
                    REQUIRE(stencilValues[1][0][0][isv][jsv][ksv] == stencilValues.field1d()[index_sv + (9) * svGridSize]);
                    REQUIRE(stencilValues[1][0][2][isv][jsv][ksv] == stencilValues.field1d()[index_sv + (9 + 2) * svGridSize]);
                    REQUIRE(stencilValues[1][2][0][isv][jsv][ksv] == stencilValues.field1d()[index_sv + (9 + 6) * svGridSize]);
                    REQUIRE(stencilValues[1][2][2][isv][jsv][ksv] == stencilValues.field1d()[index_sv + (9 + 6 + 2) * svGridSize]);
                    REQUIRE(stencilValues[0][1][0][isv][jsv][ksv] == stencilValues.field1d()[index_sv + (3) * svGridSize]);
                    REQUIRE(stencilValues[0][1][2][isv][jsv][ksv] == stencilValues.field1d()[index_sv + (3 + 2) * svGridSize]);
                    REQUIRE(stencilValues[2][1][0][isv][jsv][ksv] == stencilValues.field1d()[index_sv + (18 + 3) * svGridSize]);
                    REQUIRE(stencilValues[2][1][2][isv][jsv][ksv] == stencilValues.field1d()[index_sv + (18 + 3 + 2) * svGridSize]);
                    REQUIRE(stencilValues[0][0][1][isv][jsv][ksv] == stencilValues.field1d()[index_sv + (1) * svGridSize]);
                    REQUIRE(stencilValues[0][2][1][isv][jsv][ksv] == stencilValues.field1d()[index_sv + (6 + 1) * svGridSize]);
                    REQUIRE(stencilValues[2][0][1][isv][jsv][ksv] == stencilValues.field1d()[index_sv + (18 + 1) * svGridSize]);
                    REQUIRE(stencilValues[2][2][1][isv][jsv][ksv] == stencilValues.field1d()[index_sv + (18 + 6 + 1) * svGridSize]);
                    REQUIRE(stencilValues[0][0][0][isv][jsv][ksv] == stencilValues.field1d()[index_sv]);
                    REQUIRE(stencilValues[0][0][2][isv][jsv][ksv] == stencilValues.field1d()[index_sv + (2) * svGridSize]);
                    REQUIRE(stencilValues[0][2][0][isv][jsv][ksv] == stencilValues.field1d()[index_sv + (6) * svGridSize]);
                    REQUIRE(stencilValues[0][2][2][isv][jsv][ksv] == stencilValues.field1d()[index_sv + (6 + 2) * svGridSize]);
                    REQUIRE(stencilValues[2][0][0][isv][jsv][ksv] == stencilValues.field1d()[index_sv + (18) * svGridSize]);
                    REQUIRE(stencilValues[2][0][2][isv][jsv][ksv] == stencilValues.field1d()[index_sv + (18 + 2) * svGridSize]);
                    REQUIRE(stencilValues[2][2][0][isv][jsv][ksv] == stencilValues.field1d()[index_sv + (18 + 6) * svGridSize]);
                    REQUIRE(stencilValues[2][2][2][isv][jsv][ksv] == stencilValues.field1d()[index_sv + (18 + 6 + 2) * svGridSize]);
                }
    }

    SECTION("ocl vs seq periodic")
    {
        auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));

        int m = 8;
        int n = 8;
        int o = 8;
        double omega = 0.8;
        double h = 1.0 / static_cast<double>(m);
        // int maxiter = GENERATE(1, 2, 3, 4);
        int maxiter = 1;
        mgcl::MGCL_RESIDUAL_NORM resnorm = mgcl::MGCL_L2;
        mgcl::MGCL_STENCIL stencilType = mgcl::MGCL_VARYING;
        mgcl::BC bc = mgcl::BC::PERIODIC;

        auto v_in = std::make_shared<mgcl::Cuboid>(m, n, o, 0, 0, 0);
        auto f_in = std::make_shared<mgcl::Cuboid>(m, n, o, 0, 0, 0);
        auto r_in = std::make_shared<mgcl::Cuboid>(m, n, o, 0, 0, 0);
        f_in->fillRandomInt();
        v_in->fillRandomInt();

        // init sequential Problem
        auto p_seq = std::make_shared<mgcl::Problem>(m, n, o);
        p_seq->setResidualNorm(resnorm);
        p_seq->setOmega(omega);
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
        p_gpu->setOmega(omega);
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

        mgcl::MultigridEngine::updateGhosts(*p_gpu, level0_gpu.getDVIn(), nullptr, true);
        tu.finish();
        mgcl::MultigridEngine::updateGhostsSeq(v_in_lv0, nullptr, true, true);

        // make sure input is equal
        auto c_r_in = level0_gpu.getDR().read(tu.getCommands(), nullptr, true);
        auto c_v_in = level0_gpu.getDVIn().read(tu.getCommands(), nullptr, true);
        auto c_f_in = level0_gpu.getDF().read(tu.getCommands(), nullptr, true);
        auto c_sv_in = level0_gpu.getStencilValuesGpu()->read(tu.getCommands(), true);
        tu.finish();
        REQUIRE(c_r_in->isEqual(r_in_lv0));
        REQUIRE(c_v_in->isEqual(v_in_lv0));
        REQUIRE(c_f_in->isEqual(f_in_lv0));
        REQUIRE(c_sv_in.isEqual(*sv_in_lv0));

        double res_gpu = mgcl::MultigridEngine::jacobi(*p_gpu, level0_gpu, maxiter, true);
        tu.finish();
        double res_seq = mgcl::MultigridEngine::jacobiSeq(v_in_lv0, f_in_lv0, r_in_lv0, omega, h * h, maxiter, resnorm,
                                                          stencilType, level0_seq.getStencilFactor(), sv_in_lv0.get(),
                                                          nullptr, true, true, true);

        // res_gpu = mgcl::MultigridEngine::residual(*p_gpu, level0_gpu, true);
        // res_seq = mgcl::MultigridEngine::residualSeq(f_in_lv0, v_in_lv0, r_in_lv0, mgcl::MGCL_L2,
        //                                              mgcl::MGCL_VARYING, 1, *sv_in_lv0, true, true);

        auto c_r_out = level0_gpu.getDR().read(tu.getCommands(), nullptr, true);
        auto c_v_out = level0_gpu.getDVIn().read(tu.getCommands(), nullptr, true);
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

        // TODO check
        REQUIRE(c_v_out->isEqual(v_in_lv0));
        REQUIRE_THAT(res_seq, Catch::Matchers::WithinAbs(res_gpu, 1e-7));
        // REQUIRE(fabs(res_seq - res_gpu) < 1e-13);
        REQUIRE(c_r_out->isEqual(r_in_lv0));
    }
}

// omit this test for now, fix later
// TEST_CASE("jacobi OpenCL L2-norm 7point localMemory", "[.]")
// {
//     int m = 16;
//     int n = 16;
//     int o = 16;
//     int ghosts_m = 1;
//     int ghosts_n = 1;
//     int ghosts_o = 1;
//     int mgh = m + 2 * ghosts_m;
//     int ngh = n + 2 * ghosts_n;
//     int ogh = o + 2 * ghosts_o;
//     double omega = 0.8;
//     int maxiter = 5;

//     auto c_in_f = mgcl_test::test_jacobi::inputF16();
//     auto c_in_v = mgcl_test::test_jacobi::inputV16();
//     auto c_in_r = mgcl_test::test_jacobi::inputR16();
//     auto c_expected_out_v = mgcl_test::test_jacobi::outputV16();
//     auto c_expected_out_r = mgcl_test::test_jacobi::outputR16();

//     auto stencil = mgcl::MGCL_LAPLACE_7POINT;
//     int ghosts = 3;

//     auto p = std::make_shared<mgcl::Problem>(m, n, o);
//     p->setResidualNorm(mgcl::MGCL_L2);
//     p->setStencilType(stencil);
//     p->setGhosts(ghosts);
//     p->setOmega(omega);
//     p->setUseLocalMemory(true);
//     p->setJacobiWgSizeX(8);
//     p->setJacobiWgSizeY(8);

//     mgcl_test::TestUtility* tu_tmp = new mgcl_test::TestUtility();
//     if (tu_tmp->deviceAvailable("Quadro", p->getDeviceType()))
//         p->setDeviceName("Quadro");
//     delete tu_tmp;

//     // create input Cuboids with different ghost cell count
//     mgcl::Cuboid c_in_f_gh3(c_in_f->getM(), c_in_f->getN(), c_in_f->getO(), ghosts, ghosts, ghosts);
//     mgcl::Cuboid c_in_v_gh3(c_in_v->getM(), c_in_v->getN(), c_in_v->getO(), ghosts, ghosts, ghosts);
//     mgcl::Cuboid c_in_r_gh3(c_in_r->getM(), c_in_r->getN(), c_in_r->getO(), ghosts, ghosts, ghosts);
//     for (int i = 0; i < c_in_f->getM(); i++)
//         for (int j = 0; j < c_in_f->getN(); j++)
//             for (int k = 0; k < c_in_f->getO(); k++)
//             {
//                 c_in_f_gh3[i + ghosts][j + ghosts][k + ghosts] = (*c_in_f)[i + 1][j + 1][k + 1];
//                 c_in_v_gh3[i + ghosts][j + ghosts][k + ghosts] = (*c_in_v)[i + 1][j + 1][k + 1];
//                 c_in_r_gh3[i + ghosts][j + ghosts][k + ghosts] = (*c_in_r)[i + 1][j + 1][k + 1];
//             }
//     mgcl::MultigridEngine::updateGhostsSeq(c_in_f_gh3, nullptr, true, true);
//     mgcl::MultigridEngine::updateGhostsSeq(c_in_v_gh3, nullptr, true, true);
//     mgcl::MultigridEngine::updateGhostsSeq(c_in_r_gh3, nullptr, true, true);

//     mgcl_test::TestUtility tu(p);
//     auto d_in_f = std::make_shared<mgcl::CuboidGpu>(tu.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, c_in_f_gh3);
//     auto d_in_v = std::make_shared<mgcl::CuboidGpu>(tu.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, c_in_v_gh3);
//     auto d_in_v_out = std::make_shared<mgcl::CuboidGpu>(tu.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, c_in_v_gh3);
//     auto d_in_r = std::make_shared<mgcl::CuboidGpu>(tu.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, c_in_r_gh3);

//     mgcl::Level level(p.get(), 0);
//     level.setDF(d_in_f);
//     level.setDVIn(d_in_v);
//     level.setDVOut(d_in_v_out);
//     level.setDR(d_in_r);

//     double res = mgcl::MultigridEngine::jacobi(*p, level, maxiter, 1);
//     tu.finish();

//     // read back from device and copy to Cuboid with ghosts = 1
//     auto c_r_out = d_in_r->read(p->getCommands(), nullptr, true);
//     auto c_v_out = d_in_v->read(p->getCommands(), nullptr, true);
//     for (int i = 0; i < c_in_f->getM(); i++)
//         for (int j = 0; j < c_in_f->getN(); j++)
//             for (int k = 0; k < c_in_f->getO(); k++)
//             {
//                 (*c_in_v)[i + 1][j + 1][k + 1] = (*c_v_out)[i + ghosts][j + ghosts][k + ghosts];
//                 (*c_in_r)[i + 1][j + 1][k + 1] = (*c_r_out)[i + ghosts][j + ghosts][k + ghosts];
//             }

//     // c_in_r->dumpToFile("c_in_r->txt");
//     // c_expected_out_r->dumpToFile("c_expected_out_r.txt");

//     REQUIRE_THAT(res, Catch::Matchers::WithinAbs(4.02895897954478714e+04, 1e-7));
//     // CHECK(fabs(res - 4.02895897954478714e+04) < 1e-7);
//     CHECK(c_in_v->isEqual(*c_expected_out_v));
//     CHECK(c_in_r->isEqual(*c_expected_out_r));
// }

// Test if exceptions are thrown for wrong parameters in Jacobi. Only needed for periodic case.
// TODO check Dirichlet MPI
TEST_CASE("jacobi throwing")
{
    int iters = 5;
    int stepsPerIter = 3;

    int m = 16;
    int n = 16;
    int o = 16;

    double omega = 0.8;
    double h = 1.0 / ((double)m);
    mgcl::MGCL_STENCIL stencilType = mgcl::MGCL_LAPLACE_27POINT;
    double stencilFactor = 1.0 / (26.0 * h * h);
    mgcl::MGCL_RESIDUAL_NORM resnorm = mgcl::MGCL_L2;

    // Ghost cell amounts needed for stepsPerIter
    int ghm_v = stepsPerIter;
    int ghn_v = stepsPerIter;
    int gho_v = stepsPerIter;
    int ghm_rf = stepsPerIter - 1;
    int ghn_rf = stepsPerIter - 1;
    int gho_rf = stepsPerIter - 1;

    // Create ghosted fields that are ok and fields with too little ghost cells
    mgcl::Cuboid v_gh(m, n, o, ghm_v, ghn_v, gho_v);
    mgcl::Cuboid r_gh(m, n, o, ghm_rf, ghn_rf, gho_rf);
    mgcl::Cuboid f_gh(m, n, o, ghm_rf, ghn_rf, gho_rf);

    mgcl::Cuboid v(m, n, o, 1, 1, 1);
    mgcl::Cuboid r(m, n, o, 1, 1, 1);
    mgcl::Cuboid f(m, n, o, 1, 1, 1);

    SECTION("seq")
    {
        SECTION("throws when stepsPerIter > ghosts and periodic")
        {
            // ghosts of v too small
            REQUIRE_THROWS(mgcl::MultigridEngine::jacobiSeq(v, f_gh, r_gh, omega, h * h, iters, resnorm, stencilType,
                                                            stencilFactor, nullptr, nullptr, false, true, true, stepsPerIter));

            // ghosts of f too small
            REQUIRE_THROWS(mgcl::MultigridEngine::jacobiSeq(v_gh, f, r_gh, omega, h * h, iters, resnorm, stencilType,
                                                            stencilFactor, nullptr, nullptr, false, true, true, stepsPerIter));

            // ghosts of r too small
            REQUIRE_THROWS(mgcl::MultigridEngine::jacobiSeq(v_gh, f_gh, r, omega, h * h, iters, resnorm, stencilType,
                                                            stencilFactor, nullptr, nullptr, false, true, true, stepsPerIter));
        }

        SECTION("throws when stencilValues null and stencilType varying")
        {
            REQUIRE_THROWS(mgcl::MultigridEngine::jacobiSeq(v_gh, f_gh, r_gh, omega, h * h, iters, resnorm, mgcl::MGCL_VARYING,
                                                            stencilFactor, nullptr, nullptr, false, true, true, stepsPerIter));
        }
    }

    SECTION("gpu")
    {
        auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));

        {
            // init problem with ghosts = 1
            auto p_exp = std::make_shared<mgcl::Problem>(m, n, o);
            p_exp->setGhosts(1);
            p_exp->setDeviceType(deviceType);

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

            SECTION("throws when stepsPerIter > ghosts and periodic")
            {
                REQUIRE_THROWS(mgcl::MultigridEngine::jacobi(*p_exp, level_exp, iters, false, stepsPerIter));
            }
        }
    }
}

// tests if jacobi seq works if v_gh > 1, i.e. multiple iterations can be done without ghost update in between
// TODO test for non-periodic and varying stencil
TEST_CASE("jacobi seq gh > 1 multiple iters")
{
    int iters = GENERATE(1, 2, 5);
    int stepsPerIter = GENERATE(1, 2, 3);

    int N = GENERATE(1, 16);
    int m = N;
    int n = N;
    int o = N;
    int ghm_v = GENERATE(3, 4);
    int ghn_v = GENERATE(3, 4);
    int gho_v = stepsPerIter;
    int ghm_rf = GENERATE(2, 3);
    int ghn_rf = stepsPerIter - 1;
    int gho_rf = stepsPerIter - 1;

    // std::cout << "iters, stepsPerIter: " << iters << ", " << stepsPerIter << std::endl;
    // std::cout << "gh_v: " << ghm_v << "," << ghn_v << "," << gho_v << std::endl;
    // std::cout << "gh_rf: " << ghm_rf << "," << ghn_rf << "," << gho_rf << std::endl;

    double omega = 0.8;

    double h = 1.0 / ((double)m);

    double stencilFactor = 1.0 / (26.0 * h * h);
    mgcl::MGCL_RESIDUAL_NORM resnorm = mgcl::MGCL_L2;

    // v, r and f with extended ghosts, i.e. no ghost update between iterations
    mgcl::Cuboid v_in_gh(m, n, o, ghm_v, ghn_v, gho_v);
    mgcl::Cuboid r_in_gh(m, n, o, ghm_rf, ghn_rf, gho_rf);
    mgcl::Cuboid f_in_gh(m, n, o, ghm_rf, ghn_rf, gho_rf);

    // v and r with gh = 1, i.e. regular ghost update between iterations
    mgcl::Cuboid v_in(m, n, o, 1, 1, 1);
    mgcl::Cuboid r_in(m, n, o, 1, 1, 1);
    mgcl::Cuboid f_in(m, n, o, 0, 0, 0);
    v_in.fillRandom(-10, 10, true);
    f_in.fillRandom(-10, 10, true);

    // copy real cells from v_in to v_in_gh
    v_in_gh.fillRealFrom(v_in);
    f_in_gh.fillRealFrom(f_in);

    SECTION("periodic 27p-Laplace")
    {
        mgcl::MGCL_STENCIL stencilType = mgcl::MGCL_LAPLACE_27POINT;

        mgcl::MultigridEngine::updateGhostsSeq(v_in, nullptr, true, true);
        mgcl::MultigridEngine::updateGhostsSeq(v_in_gh, nullptr, true, true);
        mgcl::MultigridEngine::updateGhostsSeq(f_in_gh, nullptr, true, true);

        // First calculate exptected result with regular ghost updates between iterations
        double res_exp = mgcl::MultigridEngine::jacobiSeq(v_in, f_in, r_in, omega, h * h, iters, resnorm, stencilType, stencilFactor, nullptr, nullptr, true, true, 1);

        // Now calculate with gh > 1
        double res_act = mgcl::MultigridEngine::jacobiSeq(v_in_gh, f_in_gh, r_in_gh, omega, h * h, iters, resnorm, stencilType, stencilFactor, nullptr, nullptr, true, true, stepsPerIter);

        REQUIRE_THAT(res_exp, Catch::Matchers::WithinAbs(res_act, 1e-7));
        REQUIRE(v_in.isEqual(v_in_gh));
        REQUIRE(r_in.isEqual(r_in_gh));
    }

    SECTION("Dirichlet 27p-Laplace")
    {
        mgcl::MGCL_STENCIL stencilType = mgcl::MGCL_LAPLACE_27POINT;

        // First calculate exptected result with regular ghost updates between iterations
        double res_exp = mgcl::MultigridEngine::jacobiSeq(v_in, f_in, r_in, omega, h * h, iters, resnorm, stencilType, stencilFactor, nullptr, nullptr, true, false, 1);

        // Now calculate with gh > 1
        double res_act = mgcl::MultigridEngine::jacobiSeq(v_in_gh, f_in_gh, r_in_gh, omega, h * h, iters, resnorm, stencilType, stencilFactor, nullptr, nullptr, true, false, stepsPerIter);

        REQUIRE_THAT(res_exp, Catch::Matchers::WithinAbs(res_act, 1e-7));
        REQUIRE(v_in.isEqual(v_in_gh));
        REQUIRE(r_in.isEqual(r_in_gh));
    }

    SECTION("periodic varying stencil")
    {
        mgcl::MGCL_STENCIL stencilType = mgcl::MGCL_VARYING;
        mgcl::VaryingStencil stencilValuesExp(m, n, o, 3, v_in.getGhostsM(), v_in.getGhostsN(), v_in.getGhostsO());
        mgcl::VaryingStencil stencilValuesAct(m, n, o, 3, v_in_gh.getGhostsM(), v_in_gh.getGhostsN(), v_in_gh.getGhostsO());
        stencilValuesExp.fillRandom(-10, 10);
        stencilValuesAct.copyRealFrom(stencilValuesExp);
        stencilValuesExp.updateGhosts();
        stencilValuesAct.updateGhosts();

        mgcl::MultigridEngine::updateGhostsSeq(v_in, nullptr, true, true);
        mgcl::MultigridEngine::updateGhostsSeq(v_in_gh, nullptr, true, true);
        mgcl::MultigridEngine::updateGhostsSeq(f_in_gh, nullptr, true, true);

        // First calculate exptected result with regular ghost updates between iterations
        double res_exp = mgcl::MultigridEngine::jacobiSeq(v_in, f_in, r_in, omega, h * h, iters, resnorm,
                                                          stencilType, stencilFactor, &stencilValuesExp, nullptr, true, true, 1);

        // Now calculate with gh > 1
        double res_act = mgcl::MultigridEngine::jacobiSeq(v_in_gh, f_in_gh, r_in_gh, omega, h * h, iters, resnorm,
                                                          stencilType, stencilFactor, &stencilValuesAct, nullptr,
                                                          true, true, stepsPerIter);

        REQUIRE_THAT(res_exp, Catch::Matchers::WithinAbs(res_act, 1e-7));
        REQUIRE(v_in.isEqual(v_in_gh));
        REQUIRE(r_in.isEqual(r_in_gh));
    }
}

// tests if jacobi on gpu works if v_gh > 1, i.e. multiple iterations can be done without ghost update in between
// TODO test for non-periodic and varying stencil
TEST_CASE("jacobi gpu gh > 1 multiple iters")
{
    auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));

    int iters = GENERATE(1, 2, 5);
    int stepsPerIter = GENERATE(1, 2, 3);
    // int iters = 1;
    // int stepsPerIter = 3;

    int N = GENERATE(1, 16);
    int m = N;
    int n = N;
    int o = N;

    // gpu implementation currently only supports one ghost cell amount for all fields and directions
    // int gh = std::max(4, stepsPerIter);
    int gh = stepsPerIter;
    // int ghm_v = GENERATE(3, 4);
    // int ghn_v = GENERATE(3, 4);
    // int gho_v = stepsPerIter;
    // int ghm_rf = GENERATE(2, 3);
    // int ghn_rf = stepsPerIter - 1;
    // int gho_rf = stepsPerIter - 1;

    CAPTURE(iters, stepsPerIter, gh);

    double omega = 0.8;

    double h = 1.0 / ((double)m);
    mgcl::MGCL_STENCIL stencilType = mgcl::MGCL_LAPLACE_27POINT;
    double stencilFactor = 1.0 / (26.0 * h * h);
    mgcl::MGCL_RESIDUAL_NORM resnorm = mgcl::MGCL_L2;

    // v, r and f with extended ghosts, i.e. no ghost update between iterations
    mgcl::Cuboid v_act(m, n, o, gh, gh, gh);
    mgcl::Cuboid r_act(m, n, o, gh, gh, gh);
    mgcl::Cuboid f_act(m, n, o, gh, gh, gh);

    // v and r with gh = 1, i.e. regular ghost update between iterations
    mgcl::Cuboid v_exp(m, n, o, 1, 1, 1);
    mgcl::Cuboid r_exp(m, n, o, 1, 1, 1);
    mgcl::Cuboid f_exp(m, n, o, 1, 1, 1);

    // Fill with 4th order periodic Problem
    mgcl::Cuboid solution(m, n, o);
    mgcl_test::create4hOrderPeriodicProblem(v_exp, f_exp, solution);

    // v_exp.fillRandom(-10, 10, true);
    // f_exp.fillRandom(-10, 10, true);

    // copy real cells from exp to act for v and f
    v_act.fillRealFrom(v_exp);
    f_act.fillRealFrom(f_exp);

    {
        // init problem with ghosts = 1
        auto p_exp = std::make_shared<mgcl::Problem>(m, n, o);
        p_exp->setResidualNorm(mgcl::MGCL_L2);
        p_exp->setGhosts(1);
        p_exp->setDeviceType(deviceType);
        p_exp->setJacobiIterationsPerKernel(1);

        mgcl_test::TestUtility tu_exp(p_exp);
        auto d_f_exp = std::make_shared<mgcl::CuboidGpu>(tu_exp.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, f_exp);
        auto d_v_exp = std::make_shared<mgcl::CuboidGpu>(tu_exp.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, v_exp);
        auto d_v_out_exp = std::make_shared<mgcl::CuboidGpu>(tu_exp.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, v_exp);
        auto d_r_exp = std::make_shared<mgcl::CuboidGpu>(tu_exp.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, r_exp);
        auto d_rsq_exp = std::make_shared<mgcl::CuboidGpu>(tu_exp.getContext(), CL_MEM_READ_ONLY, m, n, o, 0, 0, 0);

        mgcl::Level level_exp(p_exp.get(), 0);
        level_exp.setDF(d_f_exp);
        level_exp.setDVIn(d_v_exp);
        level_exp.setDVOut(d_v_out_exp);
        level_exp.setDR(d_r_exp);
        level_exp.setDRsq(d_rsq_exp);

        // init problem with ghosts = act_ghm
        auto p_act = std::make_shared<mgcl::Problem>(m, n, o);
        p_act->setResidualNorm(mgcl::MGCL_L2);
        p_act->setGhosts(gh);
        p_act->setDeviceType(CL_DEVICE_TYPE_GPU);
        p_act->setJacobiIterationsPerKernel(stepsPerIter);

        mgcl_test::TestUtility tu_act(p_act);
        auto d_f_act = std::make_shared<mgcl::CuboidGpu>(tu_act.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, f_act);
        auto d_v_act = std::make_shared<mgcl::CuboidGpu>(tu_act.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, v_act);
        auto d_v_out_act = std::make_shared<mgcl::CuboidGpu>(tu_act.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, v_act);
        auto d_r_act = std::make_shared<mgcl::CuboidGpu>(tu_act.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, r_act);
        auto d_rsq_act = std::make_shared<mgcl::CuboidGpu>(tu_act.getContext(), CL_MEM_READ_ONLY, m, n, o, 0, 0, 0);

        mgcl::Level level_act(p_act.get(), 0);
        level_act.setDF(d_f_act);
        level_act.setDVIn(d_v_act);
        level_act.setDVOut(d_v_out_act);
        level_act.setDR(d_r_act);
        level_act.setDRsq(d_rsq_act);

        // Make sure input is equal (real cells only).
        REQUIRE(v_exp.isEqual(v_act));
        REQUIRE(r_exp.isEqual(r_act));
        REQUIRE(f_exp.isEqual(f_act));

        SECTION("periodic 7p-Laplace")
        {
            p_exp->setStencilType(mgcl::MGCL_LAPLACE_7POINT);
            p_act->setStencilType(mgcl::MGCL_LAPLACE_7POINT);

            mgcl::MultigridEngine::updateGhosts(*p_exp, *d_v_exp, nullptr, true);
            mgcl::MultigridEngine::updateGhosts(*p_exp, *d_f_exp, nullptr, true);
            mgcl::MultigridEngine::updateGhosts(*p_act, *d_v_act, nullptr, true);
            mgcl::MultigridEngine::updateGhosts(*p_act, *d_f_act, nullptr, true);
            tu_act.finish();
            tu_exp.finish();

            // First calculate exptected result with regular ghost updates between iterations
            double res_exp = mgcl::MultigridEngine::jacobi(*p_exp, level_exp, iters, true, 1);
            tu_exp.finish();

            // Now calculate with gh > 1
            double res_act = mgcl::MultigridEngine::jacobi(*p_act, level_act, iters, true, stepsPerIter);
            tu_act.finish();

            auto r_out_exp = d_r_exp->read(tu_exp.getCommands(), nullptr, true);
            auto r_out_act = d_r_act->read(tu_act.getCommands(), nullptr, true);
            auto v_out_exp = d_v_exp->read(tu_exp.getCommands(), nullptr, true);
            auto v_out_act = d_v_act->read(tu_act.getCommands(), nullptr, true);

            // v_out_exp->dumpToFile("v_out_exp.txt", true);
            // v_out_act->dumpToFile("v_out_act.txt", true);

            REQUIRE_THAT(res_exp, Catch::Matchers::WithinAbs(res_act, 1e-7));
            REQUIRE(r_out_act->isEqual(*r_out_exp));
            REQUIRE(v_out_act->isEqual(*v_out_exp));
        }

        SECTION("periodic 19p-Laplace")
        {
            p_exp->setStencilType(mgcl::MGCL_LAPLACE_19POINT);
            p_act->setStencilType(mgcl::MGCL_LAPLACE_19POINT);

            mgcl::MultigridEngine::updateGhosts(*p_exp, *d_v_exp, nullptr, true);
            mgcl::MultigridEngine::updateGhosts(*p_exp, *d_f_exp, nullptr, true);
            mgcl::MultigridEngine::updateGhosts(*p_act, *d_v_act, nullptr, true);
            mgcl::MultigridEngine::updateGhosts(*p_act, *d_f_act, nullptr, true);
            tu_act.finish();
            tu_exp.finish();

            // First calculate exptected result with regular ghost updates between iterations
            double res_exp = mgcl::MultigridEngine::jacobi(*p_exp, level_exp, iters, true, 1);
            tu_exp.finish();

            // Now calculate with gh > 1
            double res_act = mgcl::MultigridEngine::jacobi(*p_act, level_act, iters, true, stepsPerIter);
            tu_act.finish();

            auto r_out_exp = d_r_exp->read(tu_exp.getCommands(), nullptr, true);
            auto r_out_act = d_r_act->read(tu_act.getCommands(), nullptr, true);
            auto v_out_exp = d_v_exp->read(tu_exp.getCommands(), nullptr, true);
            auto v_out_act = d_v_act->read(tu_act.getCommands(), nullptr, true);

            REQUIRE_THAT(res_exp, Catch::Matchers::WithinAbs(res_act, 1e-7));
            REQUIRE(r_out_act->isEqual(*r_out_exp));
            REQUIRE(v_out_act->isEqual(*v_out_exp));
        }

        SECTION("periodic 27p-Laplace")
        {
            p_exp->setStencilType(mgcl::MGCL_LAPLACE_27POINT);
            p_act->setStencilType(mgcl::MGCL_LAPLACE_27POINT);

            mgcl::MultigridEngine::updateGhosts(*p_exp, *d_v_exp, nullptr, true);
            mgcl::MultigridEngine::updateGhosts(*p_exp, *d_f_exp, nullptr, true);
            mgcl::MultigridEngine::updateGhosts(*p_act, *d_v_act, nullptr, true);
            mgcl::MultigridEngine::updateGhosts(*p_act, *d_f_act, nullptr, true);
            tu_act.finish();
            tu_exp.finish();

            // First calculate exptected result with regular ghost updates between iterations
            double res_exp = mgcl::MultigridEngine::jacobi(*p_exp, level_exp, iters, true, 1);
            tu_exp.finish();

            // Now calculate with gh > 1
            double res_act = mgcl::MultigridEngine::jacobi(*p_act, level_act, iters, true, stepsPerIter);
            tu_act.finish();

            auto r_out_exp = d_r_exp->read(tu_exp.getCommands(), nullptr, true);
            auto r_out_act = d_r_act->read(tu_act.getCommands(), nullptr, true);
            auto v_out_exp = d_v_exp->read(tu_exp.getCommands(), nullptr, true);
            auto v_out_act = d_v_act->read(tu_act.getCommands(), nullptr, true);

            REQUIRE_THAT(res_exp, Catch::Matchers::WithinAbs(res_act, 1e-7));
            REQUIRE(r_out_act->isEqual(*r_out_exp));
            REQUIRE(v_out_act->isEqual(*v_out_exp));
        }

        // SECTION("Dirichlet 27p-Laplace")
        // {
        //     // First calculate exptected result with regular ghost updates between iterations
        //     double res_exp = mgcl::MultigridEngine::jacobiSeq(v_in, f_in, r_in, omega, h*h,iters, resnorm, stencilType, stencilFactor, dummy, true, false, 1);

        //     // Now calculate with gh > 1
        //     double res_act = mgcl::MultigridEngine::jacobiSeq(v_in_gh, f_in_gh, r_in_gh, omega, h*h,iters, resnorm, stencilType, stencilFactor, dummy, true, false, stepsPerIter);

        //     REQUIRE_THAT(res_exp, Catch::Matchers::WithinAbs(res_act, 1e-7));
        //     REQUIRE(v_in.isEqual(v_in_gh));
        //     REQUIRE(r_in.isEqual(r_in_gh));
        // }

        SECTION("Varying 27p")
        {
            p_exp->setStencilType(mgcl::MGCL_VARYING);
            p_act->setStencilType(mgcl::MGCL_VARYING);

            mgcl::MultigridEngine::updateGhosts(*p_exp, *d_v_exp, nullptr, true);
            mgcl::MultigridEngine::updateGhosts(*p_exp, *d_f_exp, nullptr, true);
            mgcl::MultigridEngine::updateGhosts(*p_act, *d_v_act, nullptr, true);
            mgcl::MultigridEngine::updateGhosts(*p_act, *d_f_act, nullptr, true);
            tu_act.finish();
            tu_exp.finish();

            auto sv_exp = p_exp->getStencilValues();
            sv_exp->fill1dIndex(true);
            sv_exp->updateGhosts();

            // copy stencil values
            auto sv_act = p_act->getStencilValues();
            sv_act->copyRealFrom(*sv_exp);
            sv_act->updateGhosts();

            REQUIRE(sv_exp->getGhostsM() == 1);
            REQUIRE(sv_exp->getGhostsN() == 1);
            REQUIRE(sv_exp->getGhostsO() == 1);
            REQUIRE(sv_act->getGhostsM() == std::max(1, stepsPerIter));
            REQUIRE(sv_act->getGhostsN() == std::max(1, stepsPerIter));
            REQUIRE(sv_act->getGhostsO() == std::max(1, stepsPerIter));

            auto d_sv_exp = std::make_shared<mgcl::VaryingStencilGpu>(sv_exp->getM(), sv_exp->getN(), sv_exp->getO(), 3,
                                                                      sv_exp->getGhostsM(),
                                                                      tu_exp.getContext(), tu_exp.getCommands(), tu_exp.getProgram());
            auto d_sv_act = std::make_shared<mgcl::VaryingStencilGpu>(sv_act->getM(), sv_act->getN(), sv_act->getO(), 3,
                                                                      sv_act->getGhostsM(),
                                                                      tu_act.getContext(), tu_act.getCommands(), tu_act.getProgram());

            d_sv_exp->fill(*sv_exp, tu_exp.getCommands(), true);
            d_sv_act->fill(*sv_act, tu_act.getCommands(), true);

            level_exp.setStencilValuesGpu(d_sv_exp);
            level_act.setStencilValuesGpu(d_sv_act);

            // First calculate exptected result with gh = 1
            double res_exp = mgcl::MultigridEngine::jacobi(*p_exp, level_exp, iters, true, 1);
            tu_exp.finish();

            // Now calculate with gh > 1
            double res_act = mgcl::MultigridEngine::jacobi(*p_act, level_act, iters, true, stepsPerIter);
            tu_act.finish();

            auto r_out_exp = d_r_exp->read(tu_exp.getCommands(), nullptr, true);
            auto r_out_act = d_r_act->read(tu_act.getCommands(), nullptr, true);
            auto v_out_exp = d_v_exp->read(tu_exp.getCommands(), nullptr, true);
            auto v_out_act = d_v_act->read(tu_act.getCommands(), nullptr, true);

            REQUIRE_THAT(res_exp, Catch::Matchers::WithinAbs(res_act, 1e-7));
            REQUIRE(r_out_act->isEqual(*r_out_exp));
            REQUIRE(v_out_act->isEqual(*v_out_exp));
        }
    }
}

TEST_CASE("jacobi_seq_fixed")
{
    int m = 4;
    int n = 4;
    int o = 4;
    double omega = 0.8;
    double h2 = 1.0 / (double)(m * m);
    int maxiter = GENERATE(1, 2, 3);
    int stepsPerIter = GENERATE(1, 2, 3);
    int gh = stepsPerIter;

    if (maxiter < stepsPerIter)
    {
        SUCCEED("maxiter < stepsPerIter, skipping test");
    }

    mgcl::MGCL_RESIDUAL_NORM resnorm = mgcl::MGCL_L2;
    bool periodic = true;

    auto vin = std::make_shared<mgcl::Cuboid>(m, n, o, gh, gh, gh);
    auto fin = std::make_shared<mgcl::Cuboid>(m, n, o, gh, gh, gh);
    vin->fillRandom();
    fin->fillRandom();

    mgcl::Problem pfixed(m, n, o, fin, vin);
    pfixed.setResidualNorm(resnorm);
    pfixed.setGhostsIn(gh);
    pfixed.setGhosts(gh);
    pfixed.setJacobiIterationsPerKernel(stepsPerIter);
    pfixed.setStencilType(mgcl::MGCL_FIXED);

    auto& fixedStencil = pfixed.getFixedStencil();
    fixedStencil->fillRandom();
    (*fixedStencil)[1][1][1] = 1.0; // make sure the stencil does not produce nan

    pfixed.init();

    auto& lv0_fixed = pfixed.getLevelAt(0);
    auto& r_fixed = lv0_fixed.getR();
    auto& v_fixed = lv0_fixed.getV();
    auto& f_fixed = lv0_fixed.getF();

    double res_norm_fixed = mgcl::MultigridEngine::jacobiSeq(v_fixed, f_fixed, r_fixed, omega, h2, maxiter, resnorm, mgcl::MGCL_FIXED,
                                                             0, nullptr, fixedStencil.get(), true, periodic, true, stepsPerIter);

    // Varying
    mgcl::Problem pvarying(m, n, o, fin, vin);
    pvarying.setResidualNorm(resnorm);
    pvarying.setGhostsIn(gh);
    pvarying.setGhosts(gh);
    pvarying.setJacobiIterationsPerKernel(stepsPerIter);
    pvarying.setStencilType(mgcl::MGCL_VARYING);

    auto& sv = pvarying.getStencilValues();
    // copy from fixed into varying
    // clang-format off
    for (int i = 0; i < sv->getMgh(); i++)
    for (int j = 0; j < sv->getNgh(); j++)
    for (int k = 0; k < sv->getOgh(); k++)
        for (int ii = 0; ii < 3; ii++)
        for (int jj = 0; jj < 3; jj++)
        for (int kk = 0; kk < 3; kk++)
        {
            (*sv)[ii][jj][kk][i][j][k] = (*fixedStencil)[ii][jj][kk];
        }
    // clang-format on

    pvarying.init();

    auto& lv0_varying = pvarying.getLevelAt(0);
    auto& r_varying = lv0_varying.getR();
    auto& v_varying = lv0_varying.getV();
    auto& f_varying = lv0_varying.getF();

    double res_norm_varying = mgcl::MultigridEngine::jacobiSeq(v_varying, f_varying, r_varying, omega, h2, maxiter, resnorm, mgcl::MGCL_VARYING,
                                                               0, sv.get(), nullptr, true, periodic, true, stepsPerIter);

    REQUIRE(!std::isnan(res_norm_fixed));
    REQUIRE(!std::isnan(res_norm_varying));
    REQUIRE(res_norm_fixed == res_norm_varying);
    REQUIRE(r_fixed.isEqual(r_varying));
    REQUIRE(v_fixed.isEqual(v_varying));
}

TEST_CASE("jacobi_ocl_fixed")
{
    int m = 4;
    int n = 4;
    int o = 4;
    double omega = 0.8;
    double h2 = 1.0 / (double)(m * m);
    int maxiter = 1;      // GENERATE(1, 2, 3);
    int stepsPerIter = 1; // GENERATE(1, 2, 3);
    int gh = stepsPerIter;

    if (maxiter < stepsPerIter)
    {
        SUCCEED("maxiter < stepsPerIter, skipping test");
    }

    mgcl::MGCL_RESIDUAL_NORM resnorm = mgcl::MGCL_L2;
    bool periodic = true;

    auto vin = std::make_shared<mgcl::Cuboid>(m, n, o, gh, gh, gh);
    auto fin = std::make_shared<mgcl::Cuboid>(m, n, o, gh, gh, gh);
    vin->fillRandom();
    fin->fillRandom();

    mgcl::Problem pfixed(m, n, o, fin, vin);
    pfixed.setUseOpencl(true);
    pfixed.setResidualNorm(resnorm);
    pfixed.setGhostsIn(gh);
    pfixed.setGhosts(gh);
    pfixed.setJacobiIterationsPerKernel(stepsPerIter);
    pfixed.setStencilType(mgcl::MGCL_FIXED);

    auto& fixedStencil = pfixed.getFixedStencil();
    fixedStencil->fillRandom();
    (*fixedStencil)[1][1][1] = 1.0; // make sure the stencil does not produce nan

    pfixed.init();

    auto& lv0_fixed = pfixed.getLevelAt(0);

    double res_norm_fixed = mgcl::MultigridEngine::jacobi(pfixed, lv0_fixed, maxiter, true, stepsPerIter);

    auto v_fixed_out = lv0_fixed.getDVIn().read(pfixed.getCommands(), nullptr, true);

    // Varying
    mgcl::Problem pvarying(m, n, o, fin, vin);
    pvarying.setUseOpencl(true);
    pvarying.setResidualNorm(resnorm);
    pvarying.setGhostsIn(gh);
    pvarying.setGhosts(gh);
    pvarying.setJacobiIterationsPerKernel(stepsPerIter);
    pvarying.setStencilType(mgcl::MGCL_VARYING);

    auto& sv = pvarying.getStencilValues();
    // copy from fixed into varying
    // clang-format off
    for (int i = 0; i < sv->getMgh(); i++)
    for (int j = 0; j < sv->getNgh(); j++)
    for (int k = 0; k < sv->getOgh(); k++)
        for (int ii = 0; ii < 3; ii++)
        for (int jj = 0; jj < 3; jj++)
        for (int kk = 0; kk < 3; kk++)
        {
            (*sv)[ii][jj][kk][i][j][k] = (*fixedStencil)[ii][jj][kk];
        }
    // clang-format on

    pvarying.init();

    auto& lv0_varying = pvarying.getLevelAt(0);

    double res_norm_varying = mgcl::MultigridEngine::jacobi(pvarying, lv0_varying, maxiter, true, stepsPerIter);

    auto v_varying_out = lv0_varying.getDVIn().read(pvarying.getCommands(), nullptr, true);

    REQUIRE(!std::isnan(res_norm_fixed));
    REQUIRE(!std::isnan(res_norm_varying));
    REQUIRE(res_norm_fixed == res_norm_varying); // TODO check
    REQUIRE(v_fixed_out->isEqual(*v_varying_out));
}

// Tests local memory kernel using temporal tiling, i.e. doing multiple iterations in one kernel call,
// but without the need of having more than one ghost layer at each border.
TEST_CASE("temporal_tiling_localmem_1d")
{
    // Test whether fetching into local memory works
    SECTION("fetch_into_localmem")
    {
        int m = 16;
        int n = 16;
        int o = 16;
        int gh = 1;
        int mgh = m + 2 * gh;
        int ngh = n + 2 * gh;
        int ogh = o + 2 * gh;

        // int wg_size = 32;
        int wg_size = 4;
        int grid_size = mgh * ngh * ogh;

        std::vector<double> shmem(5 * 5 * (wg_size + 4));
        mgcl::Cuboid v_cub(m, n, o, gh, gh, gh);
        v_cub.fill1dIndex(false);
        auto& v = v_cub.field1d();

        // for (int idx = 1372; idx < grid_size; idx++)
        for (int idx = 0; idx < grid_size; idx++)
        {
            // int idx = get_global_id(0);
            int no = ngh * ogh;
            int i = idx / no;
            int j = (idx - i * no) / ogh;
            int k = idx % ogh;

            int idx_tmp = idx;
            int iloc = 0;
            int jloc = 0;

            int loc_idx = idx % wg_size;

            // fetch k left:
            for (int icnt = 0, ioff = -2 * no; icnt < 5; icnt++, ioff += no)
                for (int jcnt = 0, joff = -2 * ogh; jcnt < 5; jcnt++, joff += ogh)
                {
                    idx_tmp = idx + ioff + joff - 2;
                    if (idx_tmp < 0)
                        idx_tmp += mgh * ngh * ogh; // wrap around is entirely handled by this line
                    int i_fetch = idx_tmp / no;
                    int j_fetch = (idx_tmp - i_fetch * no) / ogh;
                    int k_fetch = idx_tmp % ogh;
                    shmem[icnt * 5 * (wg_size + 4) + jcnt * (wg_size + 4) + loc_idx] = v[i_fetch * no + j_fetch * ogh + k_fetch];
                    std::cout << "shmem[" << icnt << "," << jcnt << "," << loc_idx << "] <- v[" << i_fetch << "," << j_fetch << "," << k_fetch << "]" << std::endl;
                }

            // fetch k right:
            if (loc_idx > wg_size - 4)
            {
                for (int icnt = 0, ioff = -2 * no; icnt < 5; icnt++, ioff += no)
                    for (int jcnt = 0, joff = -2 * ogh; jcnt < 5; jcnt++, joff += ogh)
                    {
                        idx_tmp = idx + ioff + joff + 2;
                        if (idx_tmp < 0)
                            idx_tmp += mgh * ngh * ogh; // wrap around is entirely handled by this line
                        int i_fetch = idx_tmp / no;
                        int j_fetch = (idx_tmp - i_fetch * no) / ogh;
                        int k_fetch = idx_tmp % ogh;
                        shmem[icnt * 5 * (wg_size + 4) + jcnt * (wg_size + 4) + loc_idx + 4] = v[i_fetch * no + j_fetch * ogh + k_fetch];
                    }
            }

            // loop over shmem to check if values are correct, if idx_loc = wg_size-1 reached (looped over one wg)
            if (loc_idx == wg_size - 1)
            {
                // std::string filename = "shmem.txt";
                // std::ofstream of(filename);
                // for (int ii = 0; ii < 5 * 5 * (wg_size + 4); ii++)
                // {
                //     int no = ngh * ogh;
                //     int i = ii / no;
                //     int j = (ii - i * no) / ogh;
                //     int k = ii % ogh;
                //     of << i << "," << j << "," << k << ": " << shmem[ii] << std::endl;
                // }
                // of.close();
                // return;

                for (int ioff = -2; ioff < 3; ioff++)
                    for (int joff = -2; joff < 3; joff++)
                        for (int koff = -2; koff < 3; koff++)
                        {
                            int iget = (i + ioff < 0) ? i + ioff + mgh : i + ioff;
                            int jget = (j + joff < 0) ? j + joff + ngh : j + joff;
                            int kget = (k - loc_idx + koff < 0) ? k - loc_idx + koff + ogh : k - loc_idx + koff;
                            int idxget = iget * no + jget * ogh + kget;
                            CAPTURE(idx, loc_idx, i, j, k, ioff, joff, koff, iget, jget, kget);
                            REQUIRE(shmem[(ioff + 2) * 5 * (wg_size + 4) + (joff + 2) * (wg_size + 4) + (koff + 2)] == v[idxget]);
                        }
            }
        }
    }
}

// Tests local memory kernel using temporal tiling, i.e. doing multiple iterations in one kernel call,
// but without the need of having more than one ghost layer at each border.
// 2d kernel that loops over the 1st dimension. One wi per real grid point.
TEST_CASE("temporal_tiling_localmem_2d_stream")
{

    // kernel
    // set any idx_ps ("index_plane_start"), i.e. the plane index that the wg maps to
    // fetch v plane i=idx_ps-2 and store in locmem[0] (need to get from back, if idx_ps<2, be careful about ghosts)
    // fetch v plane i=idx_ps-1 and store in locmem[1] (need to get from back, if idx_ps<1, be careful about ghosts)
    // fetch v plane i=idx_ps and store in locmem[2]
    // fetch v plane i=idx_ps+1 and store in locmem[3]
    // fetch v plane i=idx_ps+2 and store in locmem[4]
    //
    // apply stencils in locmem[1] for grid points except outer border and store in locmem[0] (t=0)
    // apply stencils in locmem[2] for grid points except outer border and store in locmem[1] (t=0)
    // apply stencils in locmem[3] for grid points except outer border and store in locmem[2] (t=0)
    // apply stencils in locmem[1] for grid points except outer 2 borders and store in global memory (t=1)
    //
    // next_buf = 0
    //
    // current_glob_plane = 1 .. m // 1: the second real plane, m: the last real plane
    //   load next entire plane in locmem[next_buf] (with 2 ghost layers)
    //   apply stencils in locmem[(next_buf-1) % 5] for grid points except outer border and store in locmem[(next_buf-2) % 5] (t=0)
    //   apply stencils in locmem[(next_buf-3) % 5] for grid points except outer border and store in global memory (t=1)
    //   next_buf = (next_buf + 1) % 5
    //
    // Notes:
    // Applying stencil: Need to divide stencil into planes and apply to each plane separately, since we don't want to
    // shift data in local memory. E.g. when applying stencil in plane 0, we need to take values from planes 4, 0 and 1.
    // The planes indices are always in increasing order, whereas after plane 4 it restarts with plane 0. So, if
    // the current plane has index locmem[p], we need to get values from locmem[(p-1) % 5], locmem[p] and locmem[(p+1) % 5].
    // Maybe we could also use 3 counters to avoid the modulo op...
    //
    // Loading a real plane p: Each wi loads its grid point. Then, the outer 2 layers of wis load the ghost values. E.g.:
    // // k_loc is the local memory index in the z dimension and in range 0..loc_size_y-1 (wg is only part of real grid), i.e. loc_size_{x,y} is without local ghosts.
    // // locmem has size [loc_size_x + 4][loc_size_y + 4].
    // // ploc is the target plane in local memory, i.e. in range 0..4
    // // pglob is the global plane index of v to be fetched
    //
    // Implementation see below

    /**
     * @brief
     * pglob: Real global plane index, i.e. in range 0...m-1.
     * ploc: Local memory plane index, i.e. in range 0..4.
     * j_loc: Local memory index in y dimension, i.e. in range 0..loc_size_y-1.
     * k_loc: Local memory index in z dimension, i.e. in range 0..loc_size_y-1.
     * loc_size_x: Size of work-group in x dimension, i.e. without ghosts of local memory.
     * loc_size_y: Size of work-group in y dimension, i.e. without ghosts of local memory.
     * locmem: Local memory.
     * v: Global memory.
     * m: Size in x dimension of global memory.
     * n: Size in y dimension of global memory.
     * o: Size in z dimension of global memory.
     * j: Global index in y dimension of REAL grid point to be loaded.
     * k: Global index in z dimension of REAL grid point to be loaded.
     * vgh: Number of ghost cells in global memory.
     */
    auto load_plane = [](int pglob, int ploc,
                         int j_loc, int k_loc,
                         int loc_size_x, int loc_size_y,
                         mgcl::Cuboid& locmem, mgcl::Cuboid& v,
                         int m, int n, int o,
                         int j, int k,
                         int vgh)
    {
        // int p = pglob - 2;
        // Handle ghost planes at the beginning and at the end
        // if (pglob < 2)
        //     p = (pglob - 2 < 0 ? pglob - 2 + m : pglob - 2) % m;
        // else if (pglob >= m)
        //     p = (pglob + 2) % m;
        int p = pglob + vgh; // Don't do ghost handling for x-plane here, instead do it in algorithm. But respect amount of ghosts in v.

        locmem[ploc][j_loc + 2][k_loc + 2] = v[p][j + vgh][k + vgh]; // load self

        // Load ghosts in z dimension.
        if (k_loc < 2)
            locmem[ploc][j_loc + 2][k_loc] = v[p][j + vgh][((k - 2 < 0 ? k - 2 + o : k - 2) % o) + vgh];
        else if (k_loc >= loc_size_y - 2)
            locmem[ploc][j_loc + 2][k_loc + 4] = v[p][j + vgh][((k + 2) % o) + vgh];

        // Load ghosts in y dimension.
        if (j_loc < 2)
            locmem[ploc][j_loc][k_loc + 2] = v[p][((j - 2 < 0 ? j - 2 + n : j - 2) % n) + vgh][k + vgh];
        else if (j_loc >= loc_size_x - 2)
            locmem[ploc][j_loc + 4][k_loc + 2] = v[p][((j + 2) % n) + vgh][k + vgh];

        // Load ghosts in corners.
        if (j_loc < 2 && k_loc < 2) // upper left
            locmem[ploc][j_loc][k_loc] = v[p][((j - 2 < 0 ? j - 2 + n : j - 2) % n) + vgh][((k - 2 < 0 ? k - 2 + o : k - 2) % o) + vgh];
        else if (j_loc < 2 && k_loc >= loc_size_y - 2) // upper right
            locmem[ploc][j_loc][k_loc + 4] = v[p][((j - 2 < 0 ? j - 2 + n : j - 2) % n) + vgh][((k + 2) % o) + vgh];
        else if (j_loc >= loc_size_x - 2 && k_loc < 2) // lower left
            locmem[ploc][j_loc + 4][k_loc] = v[p][((j + 2) % n) + vgh][((k - 2 < 0 ? k - 2 + o : k - 2) % o) + vgh];
        else if (j_loc >= loc_size_x - 2 && k_loc >= loc_size_y - 2) // lower right
            locmem[ploc][j_loc + 4][k_loc + 4] = v[p][((j + 2) % n) + vgh][((k + 2) % o) + vgh];
    };

    // Test whether loading a plane into local memory works
    SECTION("load_plane")
    {
        int m = 16;
        int n = 16;
        int o = 16;
        int gh = 1;
        int mgh = m + 2 * gh;
        int ngh = n + 2 * gh;
        int ogh = o + 2 * gh;

        // int wg_size = 32;
        int wg_size_x = GENERATE(4, 8);
        int wg_size_y = 4;
        int grid_size = mgh * ngh * ogh;

        // std::vector<double> shmem((wg_size_x + 4) * (wg_size_y + 4) * 5); // need 5 planes in buffer
        mgcl::Cuboid locmem(5, (wg_size_x + 4), (wg_size_y + 4)); // need 5 planes in buffer
        mgcl::Cuboid v_cub(m, n, o, gh, gh, gh);
        mgcl::Cuboid v_cub_nogh(m, n, o);
        v_cub.fill1dIndex(false);
        v_cub_nogh.fill1dIndex(false);
        auto& v = v_cub.field1d();

        {
            // Test the following wg cases for a non-ghosted v:
            // {not touching border, touching upper border, touching lower border, touching left border, touching right border,
            //  all 4 corners (4 cases)}
            std::vector<std::vector<int>> wgs{
                {1, 1},
                {0, 1},
                {n / wg_size_x - 1, 1},
                {1, 0},
                {1, o / wg_size_y - 1},
                {0, 0},
                {n / wg_size_x - 1, 0},
                {0, o / wg_size_y - 1},
                {n / wg_size_x - 1, o / wg_size_y - 1}};

            for (auto wg : wgs)
            {
                // Test wg in the upper left corner
                int wg_num_x = wg[0];
                int wg_num_y = wg[1];

                // for (int idx = 1372; idx < grid_size; idx++)
                // loop over first work-group, i.e. in upper left corner of the grid. Try without handling ghosts of global v
                // first.
                for (int jloc = 0; jloc < wg_size_x; jloc++)
                    for (int kloc = 0; kloc < wg_size_y; kloc++)
                    {
                        int j = wg_num_x * wg_size_x + jloc;
                        int k = wg_num_y * wg_size_y + kloc;
                        // int idx = get_global_id(0);
                        // int no = ngh * ogh;
                        // int i = idx / no;
                        // int j = (idx - i * no) / ogh;
                        // int k = idx % ogh;

                        load_plane(3, 0, jloc, kloc, wg_size_x, wg_size_y, locmem, v_cub_nogh, m, n, o, j, k, 0);
                        load_plane(5, 3, jloc, kloc, wg_size_x, wg_size_y, locmem, v_cub_nogh, m, n, o, j, k, 0);
                        load_plane(6, 2, jloc, kloc, wg_size_x, wg_size_y, locmem, v_cub_nogh, m, n, o, j, k, 0);
                        load_plane(1, 4, jloc, kloc, wg_size_x, wg_size_y, locmem, v_cub_nogh, m, n, o, j, k, 0);
                        load_plane(m - 1, 1, jloc, kloc, wg_size_x, wg_size_y, locmem, v_cub_nogh, m, n, o, j, k, 0);

                        // v_cub_nogh.dumpToFile("v_cub_nogh.txt");
                        // locmem.dumpToFile("locmem.txt");
                    }

                // Check results
                for (int jloc = 0; jloc < locmem.getN(); jloc++)
                    for (int kloc = 0; kloc < locmem.getO(); kloc++)
                    {
                        int j = wg_num_x * wg_size_x + jloc - 2;
                        int k = wg_num_y * wg_size_y + kloc - 2;
                        // int pglob_real = pglob - 2;

                        // if (pglob_real < 2)
                        //     pglob_real += m;
                        // else if (pglob_real >= m + 2)
                        //     pglob_real -= m;
                        if (j < 0)
                            j += n;
                        else if (j >= n)
                            j -= n;
                        if (k < 0)
                            k += o;
                        else if (k >= o)
                            k -= o;

                        CAPTURE(jloc, kloc, j, k);
                        REQUIRE(locmem[0][jloc][kloc] == v_cub_nogh[3][j][k]);
                        REQUIRE(locmem[3][jloc][kloc] == v_cub_nogh[5][j][k]);
                        REQUIRE(locmem[2][jloc][kloc] == v_cub_nogh[6][j][k]);
                        REQUIRE(locmem[4][jloc][kloc] == v_cub_nogh[1][j][k]);
                        REQUIRE(locmem[1][jloc][kloc] == v_cub_nogh[m - 1][j][k]);
                    }
            }
        }

        {
            // Test the following wg cases for a ghosted v:
            // {not touching border, touching upper border, touching lower border, touching left border, touching right border,
            //  all 4 corners (4 cases)}
            std::vector<std::vector<int>> wgs{
                {1, 1},
                {0, 1},
                {n / wg_size_x - 1, 1},
                {1, 0},
                {1, o / wg_size_y - 1},
                {0, 0},
                {n / wg_size_x - 1, 0},
                {0, o / wg_size_y - 1},
                {n / wg_size_x - 1, o / wg_size_y - 1}};

            locmem.field1d().clear();

            for (auto wg : wgs)
            {
                // Test wg in the upper left corner
                int wg_num_x = wg[0];
                int wg_num_y = wg[1];

                // for (int idx = 1372; idx < grid_size; idx++)
                // loop over first work-group, i.e. in upper left corner of the grid. Try without handling ghosts of global v
                // first.
                for (int jloc = 0; jloc < wg_size_x; jloc++)
                    for (int kloc = 0; kloc < wg_size_y; kloc++)
                    {
                        int j = wg_num_x * wg_size_x + jloc;
                        int k = wg_num_y * wg_size_y + kloc;
                        // int idx = get_global_id(0);
                        // int no = ngh * ogh;
                        // int i = idx / no;
                        // int j = (idx - i * no) / ogh;
                        // int k = idx % ogh;

                        load_plane(3, 0, jloc, kloc, wg_size_x, wg_size_y, locmem, v_cub, m, n, o, j, k, gh);
                        load_plane(5, 3, jloc, kloc, wg_size_x, wg_size_y, locmem, v_cub, m, n, o, j, k, gh);
                        load_plane(6, 2, jloc, kloc, wg_size_x, wg_size_y, locmem, v_cub, m, n, o, j, k, gh);
                        load_plane(1, 4, jloc, kloc, wg_size_x, wg_size_y, locmem, v_cub, m, n, o, j, k, gh);
                        load_plane(m - 1, 1, jloc, kloc, wg_size_x, wg_size_y, locmem, v_cub, m, n, o, j, k, gh);

                        // v_cub_nogh.dumpToFile("v_cub.txt");
                        // locmem.dumpToFile("locmem.txt");
                    }

                // Check results
                for (int jloc = 0; jloc < locmem.getN(); jloc++)
                    for (int kloc = 0; kloc < locmem.getO(); kloc++)
                    {
                        int j = wg_num_x * wg_size_x + jloc - 2;
                        int k = wg_num_y * wg_size_y + kloc - 2;
                        // int pglob_real = pglob - 2;

                        // if (pglob_real < 2)
                        //     pglob_real += m;
                        // else if (pglob_real >= m + 2)
                        //     pglob_real -= m;
                        if (j < 0)
                            j += n;
                        else if (j >= n)
                            j -= n;
                        if (k < 0)
                            k += o;
                        else if (k >= o)
                            k -= o;

                        CAPTURE(jloc, kloc, j, k);
                        REQUIRE(locmem[0][jloc][kloc] == v_cub[3 + gh][j + gh][k + gh]);
                        REQUIRE(locmem[3][jloc][kloc] == v_cub[5 + gh][j + gh][k + gh]);
                        REQUIRE(locmem[2][jloc][kloc] == v_cub[6 + gh][j + gh][k + gh]);
                        REQUIRE(locmem[4][jloc][kloc] == v_cub[1 + gh][j + gh][k + gh]);
                        REQUIRE(locmem[1][jloc][kloc] == v_cub[m - 1 + gh][j + gh][k + gh]);
                    }
            }
        }
    }

    // // front is i-1, center is i, back is i+1
    // auto apply_stencil = [](double* front, double* center, double* back, double* stencilValues, int svGridSize, int index_sv, int index, int joff, int koff)
    // {
    //     // clang-format off
    //         double stencilsum = stencilValues[index_sv + (9 + 3 + 1) * svGridSize] * center[index]
    //             + stencilValues[index_sv + (9 + 3) * svGridSize]      * center[index - 1]
    //             + stencilValues[index_sv + (9 + 3 + 2) * svGridSize]  * center[index + 1]
    //             + stencilValues[index_sv + (9 + 1) * svGridSize]      * center[index - joff]
    //             + stencilValues[index_sv + (9 + 6 + 1) * svGridSize]  * center[index + joff]
    //             + stencilValues[index_sv + (3 + 1) * svGridSize]      * front[index]
    //             + stencilValues[index_sv + (18 + 3 + 1) * svGridSize] * back[index]

    //             + stencilValues[index_sv + (9) * svGridSize]          * center[index - joff - koff]
    //             + stencilValues[index_sv + (9 + 2) * svGridSize]      * center[index - joff + koff]
    //             + stencilValues[index_sv + (9 + 6) * svGridSize]      * center[index + joff - koff]
    //             + stencilValues[index_sv + (9 + 6 + 2) * svGridSize]  * center[index + joff + koff]
    //             + stencilValues[svGridSize * 3 + index_sv]            * front[index  - koff]
    //             + stencilValues[index_sv + (3 + 2) * svGridSize]      * front[index  + koff]
    //             + stencilValues[index_sv + (18 + 3) * svGridSize]     * back[index  - koff]
    //             + stencilValues[index_sv + (18 + 3 + 2) * svGridSize] * back[index  + koff]
    //             + stencilValues[svGridSize + index_sv]                * front[index  - joff]
    //             + stencilValues[index_sv + (6 + 1) * svGridSize]      * front[index  + joff]
    //             + stencilValues[index_sv + (18 + 1) * svGridSize]     * back[index  - joff]
    //             + stencilValues[index_sv + (18 + 6 + 1) * svGridSize] * back[index  + joff]

    //             + stencilValues[index_sv]                           * front[index  - joff - koff]
    //             + stencilValues[svGridSize * 2 + index_sv]            * front[index  - joff + koff]
    //             + stencilValues[index_sv + (6) * svGridSize]          * front[index  + joff - koff]
    //             + stencilValues[index_sv + (6 + 2) * svGridSize]      * front[index  + joff + koff]
    //             + stencilValues[index_sv + (18) * svGridSize]         * back[index  - joff - koff]
    //             + stencilValues[index_sv + (18 + 2) * svGridSize]     * back[index  - joff + koff]
    //             + stencilValues[index_sv + (18 + 6) * svGridSize]     * back[index  + joff - koff]
    //             + stencilValues[index_sv + (18 + 6 + 2) * svGridSize] * back[index  + joff + koff];
    //     // clang-format on
    // };

    // SECTION("apply_stencil")
    // {
    // }
}