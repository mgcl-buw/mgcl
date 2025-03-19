#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <functional>
#include <ios>
#include <iostream>
#include <memory>

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
// 2d kernel that loops over the 1st dimension. One wi per real grid point.
TEST_CASE("seq_temporal_tiling_localmem_2d_stream", "[.]")
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
    // Ghosts of stencilValues are irrelevant, since for ghosted grid points (the first iteration) we can just wrap around.
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
     * j: Global index in y dimension of REAL grid point to be loaded, i.e. in range 0..n-1.
     * k: Global index in z dimension of REAL grid point to be loaded, i.e. in range 0..o-1.
     * vgh: Number of ghost cells in global memory.
     */
    auto load_plane = [](int pglob, int ploc,
                         int j_loc, int k_loc,
                         int loc_size_x, int loc_size_y,
                         int locmem_size_x, int locmem_size_y,
                         double* locmem, double* v,
                         int m, int n, int o,
                         int mgh, int ngh, int ogh,
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

        int locmem_size_xy = locmem_size_x * locmem_size_y;
        int pnogh = p * ngh * ogh;

        locmem[ploc * locmem_size_xy + (j_loc + 2) * locmem_size_y + (k_loc + 2)] = v[pnogh + (j + vgh) * ogh + (k + vgh)]; // load self

        // Load ghosts in z dimension.
        if (k_loc < 2)
            locmem[ploc * locmem_size_xy + (j_loc + 2) * locmem_size_y + (k_loc)] = v[pnogh + (j + vgh) * ogh + (((k - 2 < 0 ? k - 2 + o : k - 2) % o) + vgh)];
        else if (k_loc >= loc_size_y - 2)
            locmem[ploc * locmem_size_xy + (j_loc + 2) * locmem_size_y + (k_loc + 4)] = v[pnogh + (j + vgh) * ogh + (((k + 2) % o) + vgh)];

        // Load ghosts in y dimension.
        if (j_loc < 2)
            locmem[ploc * locmem_size_xy + (j_loc)*locmem_size_y + (k_loc + 2)] = v[pnogh + (((j - 2 < 0 ? j - 2 + n : j - 2) % n) + vgh) * ogh + (k + vgh)];
        else if (j_loc >= loc_size_x - 2)
            locmem[ploc * locmem_size_xy + (j_loc + 4) * locmem_size_y + (k_loc + 2)] = v[pnogh + (((j + 2) % n) + vgh) * ogh + (k + vgh)];

        // Load ghosts in corners.
        if (j_loc < 2 && k_loc < 2) // upper left
            locmem[ploc * locmem_size_xy + (j_loc)*locmem_size_y + (k_loc)] = v[pnogh + (((j - 2 < 0 ? j - 2 + n : j - 2) % n) + vgh) * ogh + (((k - 2 < 0 ? k - 2 + o : k - 2) % o) + vgh)];
        else if (j_loc < 2 && k_loc >= loc_size_y - 2) // upper right
            locmem[ploc * locmem_size_xy + (j_loc)*locmem_size_y + (k_loc + 4)] = v[pnogh + (((j - 2 < 0 ? j - 2 + n : j - 2) % n) + vgh) * ogh + (((k + 2) % o) + vgh)];
        else if (j_loc >= loc_size_x - 2 && k_loc < 2) // lower left
            locmem[ploc * locmem_size_xy + (j_loc + 4) * locmem_size_y + (k_loc)] = v[pnogh + (((j + 2) % n) + vgh) * ogh + (((k - 2 < 0 ? k - 2 + o : k - 2) % o) + vgh)];
        else if (j_loc >= loc_size_x - 2 && k_loc >= loc_size_y - 2) // lower right
            locmem[ploc * locmem_size_xy + (j_loc + 4) * locmem_size_y + (k_loc + 4)] = v[pnogh + (((j + 2) % n) + vgh) * ogh + (((k + 2) % o) + vgh)];
    };

    // Test whether loading a plane into local memory works
    SECTION("load_plane")
    {
        int m = 16;
        int n = 16;
        int o = 16;
        int gh = GENERATE(0, 1);
        int mgh = m + 2 * gh;
        int ngh = n + 2 * gh;
        int ogh = o + 2 * gh;

        // int wg_size = 32;
        int wg_size_x = GENERATE(4, 8);
        int wg_size_y = 4;
        int grid_size = mgh * ngh * ogh;

        int locmem_size_x = (wg_size_x + 4);
        int locmem_size_y = (wg_size_y + 4);

        // std::vector<double> shmem((wg_size_x + 4) * (wg_size_y + 4) * 5); // need 5 planes in buffer
        mgcl::Cuboid locmem(5, locmem_size_x, locmem_size_y); // need 5 planes in buffer
        mgcl::Cuboid v_cub(m, n, o, gh, gh, gh);
        v_cub.fill1dIndex(false);
        auto& v = v_cub.field1d();

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

                        load_plane(3, 0, jloc, kloc, wg_size_x, wg_size_y, locmem_size_x, locmem_size_y, locmem.field1d().data(), v_cub.field1d().data(), m, n, o, mgh, ngh, ogh, j, k, gh);
                        load_plane(5, 3, jloc, kloc, wg_size_x, wg_size_y, locmem_size_x, locmem_size_y, locmem.field1d().data(), v_cub.field1d().data(), m, n, o, mgh, ngh, ogh, j, k, gh);
                        load_plane(6, 2, jloc, kloc, wg_size_x, wg_size_y, locmem_size_x, locmem_size_y, locmem.field1d().data(), v_cub.field1d().data(), m, n, o, mgh, ngh, ogh, j, k, gh);
                        load_plane(1, 4, jloc, kloc, wg_size_x, wg_size_y, locmem_size_x, locmem_size_y, locmem.field1d().data(), v_cub.field1d().data(), m, n, o, mgh, ngh, ogh, j, k, gh);
                        load_plane(m - 1, 1, jloc, kloc, wg_size_x, wg_size_y, locmem_size_x, locmem_size_y, locmem.field1d().data(), v_cub.field1d().data(), m, n, o, mgh, ngh, ogh, j, k, gh);

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

    // front is i-1, center is i, back is i+1
    // front: address of front plane in local memory
    // center: address of center plane in local memory
    // back: address of back plane in local memory
    // stencilValues: Coefficients in global memory
    // svGridSize: Size of ghosted grid of coefficients array in global memory
    // index_sv: Index of stencil (top left front coefficient) in global memory
    // index: Index of grid point in local memory, i.e. center of stencil
    // joff: Distance to the next grid point in y-direction in local memory
    auto apply_stencil = [](double* front, double* center, double* back, double* stencilValues, int svGridSize, int index_sv, int index, int joff)
    {
        // clang-format off
            double stencilsum = stencilValues[index_sv + (9 + 3 + 1) * svGridSize] * center[index]
                + stencilValues[index_sv + (9 + 3) * svGridSize]      * center[index - 1]
                + stencilValues[index_sv + (9 + 3 + 2) * svGridSize]  * center[index + 1]
                + stencilValues[index_sv + (9 + 1) * svGridSize]      * center[index - joff]
                + stencilValues[index_sv + (9 + 6 + 1) * svGridSize]  * center[index + joff]
                + stencilValues[index_sv + (3 + 1) * svGridSize]      * front[index]
                + stencilValues[index_sv + (18 + 3 + 1) * svGridSize] * back[index]

                + stencilValues[index_sv + (9) * svGridSize]          * center[index - joff - 1]
                + stencilValues[index_sv + (9 + 2) * svGridSize]      * center[index - joff + 1]
                + stencilValues[index_sv + (9 + 6) * svGridSize]      * center[index + joff - 1]
                + stencilValues[index_sv + (9 + 6 + 2) * svGridSize]  * center[index + joff + 1]
                + stencilValues[svGridSize * 3 + index_sv]            * front[index - 1]
                + stencilValues[index_sv + (3 + 2) * svGridSize]      * front[index + 1]
                + stencilValues[index_sv + (18 + 3) * svGridSize]     * back[index - 1]
                + stencilValues[index_sv + (18 + 3 + 2) * svGridSize] * back[index + 1]
                + stencilValues[svGridSize + index_sv]                * front[index - joff]
                + stencilValues[index_sv + (6 + 1) * svGridSize]      * front[index + joff]
                + stencilValues[index_sv + (18 + 1) * svGridSize]     * back[index - joff]
                + stencilValues[index_sv + (18 + 6 + 1) * svGridSize] * back[index + joff]

                + stencilValues[index_sv]                           * front[index - joff - 1]
                + stencilValues[svGridSize * 2 + index_sv]            * front[index - joff + 1]
                + stencilValues[index_sv + (6) * svGridSize]          * front[index + joff - 1]
                + stencilValues[index_sv + (6 + 2) * svGridSize]      * front[index + joff + 1]
                + stencilValues[index_sv + (18) * svGridSize]         * back[index - joff - 1]
                + stencilValues[index_sv + (18 + 2) * svGridSize]     * back[index - joff + 1]
                + stencilValues[index_sv + (18 + 6) * svGridSize]     * back[index + joff - 1]
                + stencilValues[index_sv + (18 + 6 + 2) * svGridSize] * back[index + joff + 1];
        // clang-format on

        // std::cout << "center[index]: " << center[index] << std::endl;
        // std::cout << "center[index - 1]: " << center[index - 1] << std::endl;
        // std::cout << "center[index + 1]: " << center[index + 1] << std::endl;
        // std::cout << "center[index - joff]: " << center[index - joff] << std::endl;
        // std::cout << "center[index + joff]: " << center[index + joff] << std::endl;
        // std::cout << "front[index]: " << front[index] << std::endl;
        // std::cout << "back[index]: " << back[index] << std::endl;
        // std::cout << "center[index - joff - 1]: " << center[index - joff - 1] << std::endl;
        // std::cout << "center[index - joff + 1]: " << center[index - joff + 1] << std::endl;
        // std::cout << "center[index + joff - 1]: " << center[index + joff - 1] << std::endl;
        // std::cout << "center[index + joff + 1]: " << center[index + joff + 1] << std::endl;
        // std::cout << "front[index - 1]: " << front[index - 1] << std::endl;
        // std::cout << "front[index + 1]: " << front[index + 1] << std::endl;
        // std::cout << "back[index - 1]: " << back[index - 1] << std::endl;
        // std::cout << "back[index + 1]: " << back[index + 1] << std::endl;
        // std::cout << "front[index - joff]: " << front[index - joff] << std::endl;
        // std::cout << "front[index + joff]: " << front[index + joff] << std::endl;
        // std::cout << "back[index - joff]: " << back[index - joff] << std::endl;
        // std::cout << "back[index + joff]: " << back[index + joff] << std::endl;
        // std::cout << "front[index - joff - 1]: " << front[index - joff - 1] << std::endl;
        // std::cout << "front[index - joff + 1]: " << front[index - joff + 1] << std::endl;
        // std::cout << "front[index + joff - 1]: " << front[index + joff - 1] << std::endl;
        // std::cout << "front[index + joff + 1]: " << front[index + joff + 1] << std::endl;
        // std::cout << "back[index - joff - 1]: " << back[index - joff - 1] << std::endl;
        // std::cout << "back[index - joff + 1]: " << back[index - joff + 1] << std::endl;
        // std::cout << "back[index + joff - 1]: " << back[index + joff - 1] << std::endl;
        // std::cout << "back[index + joff + 1]: " << back[index + joff + 1] << std::endl;

        return stencilsum;
    };

    // regular stencil application for checking results
    auto apply_stencil_check = [](double* v_in, double* stencilValues, int svGridSize, int index_sv, int index, int joff, int ioff)
    {
        // clang-format off
            double stencilsum = stencilValues[index_sv + (9 + 3 + 1) * svGridSize] * v_in[index]
                + stencilValues[index_sv + (9 + 3) * svGridSize]      * v_in[index - 1]
                + stencilValues[index_sv + (9 + 3 + 2) * svGridSize]  * v_in[index + 1]
                + stencilValues[index_sv + (9 + 1) * svGridSize]      * v_in[index - joff]
                + stencilValues[index_sv + (9 + 6 + 1) * svGridSize]  * v_in[index + joff]
                + stencilValues[index_sv + (3 + 1) * svGridSize]      * v_in[index - ioff]
                + stencilValues[index_sv + (18 + 3 + 1) * svGridSize] * v_in[index + ioff]
                
                + stencilValues[index_sv + (9) * svGridSize]          * v_in[index - joff - 1]
                + stencilValues[index_sv + (9 + 2) * svGridSize]      * v_in[index - joff + 1]
                + stencilValues[index_sv + (9 + 6) * svGridSize]      * v_in[index + joff - 1]
                + stencilValues[index_sv + (9 + 6 + 2) * svGridSize]  * v_in[index + joff + 1]
                + stencilValues[svGridSize * 3 + index_sv]            * v_in[index - ioff - 1]
                + stencilValues[index_sv + (3 + 2) * svGridSize]      * v_in[index - ioff + 1]
                + stencilValues[index_sv + (18 + 3) * svGridSize]     * v_in[index + ioff - 1]
                + stencilValues[index_sv + (18 + 3 + 2) * svGridSize] * v_in[index + ioff + 1]
                + stencilValues[svGridSize + index_sv]                * v_in[index - ioff - joff]
                + stencilValues[index_sv + (6 + 1) * svGridSize]      * v_in[index - ioff + joff]
                + stencilValues[index_sv + (18 + 1) * svGridSize]     * v_in[index + ioff - joff]
                + stencilValues[index_sv + (18 + 6 + 1) * svGridSize] * v_in[index + ioff + joff]

                + stencilValues[index_sv]                           * v_in[index - ioff - joff - 1]
                + stencilValues[svGridSize * 2 + index_sv]            * v_in[index - ioff - joff + 1]
                + stencilValues[index_sv + (6) * svGridSize]          * v_in[index - ioff + joff - 1]
                + stencilValues[index_sv + (6 + 2) * svGridSize]      * v_in[index - ioff + joff + 1]
                + stencilValues[index_sv + (18) * svGridSize]         * v_in[index + ioff - joff - 1]
                + stencilValues[index_sv + (18 + 2) * svGridSize]     * v_in[index + ioff - joff + 1]
                + stencilValues[index_sv + (18 + 6) * svGridSize]     * v_in[index + ioff + joff - 1]
                + stencilValues[index_sv + (18 + 6 + 2) * svGridSize] * v_in[index + ioff + joff + 1];
        // clang-format on

        // std::cout << "v_in[index]: " << v_in[index] << std::endl;
        // std::cout << "v_in[index - 1]: " << v_in[index - 1] << std::endl;
        // std::cout << "v_in[index + 1]: " << v_in[index + 1] << std::endl;
        // std::cout << "v_in[index - joff]: " << v_in[index - joff] << std::endl;
        // std::cout << "v_in[index + joff]: " << v_in[index + joff] << std::endl;
        // std::cout << "v_in[index - ioff]: " << v_in[index - ioff] << std::endl;
        // std::cout << "v_in[index + ioff]: " << v_in[index + ioff] << std::endl;
        // std::cout << "v_in[index - joff - 1]: " << v_in[index - joff - 1] << std::endl;
        // std::cout << "v_in[index - joff + 1]: " << v_in[index - joff + 1] << std::endl;
        // std::cout << "v_in[index + joff - 1]: " << v_in[index + joff - 1] << std::endl;
        // std::cout << "v_in[index + joff + 1]: " << v_in[index + joff + 1] << std::endl;
        // std::cout << "v_in[index - ioff - 1]: " << v_in[index - ioff - 1] << std::endl;
        // std::cout << "v_in[index - ioff + 1]: " << v_in[index - ioff + 1] << std::endl;
        // std::cout << "v_in[index + ioff - 1]: " << v_in[index + ioff - 1] << std::endl;
        // std::cout << "v_in[index + ioff + 1]: " << v_in[index + ioff + 1] << std::endl;
        // std::cout << "v_in[index - ioff - joff]: " << v_in[index - ioff - joff] << std::endl;
        // std::cout << "v_in[index - ioff + joff]: " << v_in[index - ioff + joff] << std::endl;
        // std::cout << "v_in[index + ioff - joff]: " << v_in[index + ioff - joff] << std::endl;
        // std::cout << "v_in[index + ioff + joff]: " << v_in[index + ioff + joff] << std::endl;
        // std::cout << "v_in[index - ioff - joff - 1]: " << v_in[index - ioff - joff - 1] << std::endl;
        // std::cout << "v_in[index - ioff - joff + 1]: " << v_in[index - ioff - joff + 1] << std::endl;
        // std::cout << "v_in[index - ioff + joff - 1]: " << v_in[index - ioff + joff - 1] << std::endl;
        // std::cout << "v_in[index - ioff + joff + 1]: " << v_in[index - ioff + joff + 1] << std::endl;
        // std::cout << "v_in[index + ioff - joff - 1]: " << v_in[index + ioff - joff - 1] << std::endl;
        // std::cout << "v_in[index + ioff - joff + 1]: " << v_in[index + ioff - joff + 1] << std::endl;
        // std::cout << "v_in[index + ioff + joff - 1]: " << v_in[index + ioff + joff - 1] << std::endl;
        // std::cout << "v_in[index + ioff + joff + 1]: " << v_in[index + ioff + joff + 1] << std::endl;

        return stencilsum;
    };

    SECTION("apply_stencil")
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

        int locmem_size_x = (wg_size_x + 4);
        int locmem_size_y = (wg_size_y + 4);

        // std::vector<double> shmem((wg_size_x + 4) * (wg_size_y + 4) * 5); // need 5 planes in buffer
        mgcl::Cuboid locmem(5, locmem_size_x, locmem_size_y); // need 5 planes in buffer
        mgcl::Cuboid v_cub(m, n, o, gh, gh, gh);
        v_cub.fill1dIndex(false);
        mgcl::MultigridEngine::updateGhostsSeq(v_cub, nullptr, true, true);
        auto& v = v_cub.field1d();

        int svgh = 1;
        mgcl::VaryingStencil stencilValues(m, n, o, 3, svgh, svgh, svgh);
        stencilValues.fill1dIndex(false);
        // stencilValues.fill(1.0, false); // TODO varying
        stencilValues.updateGhosts();

        int svGridSize = stencilValues.getMgh() * stencilValues.getNgh() * stencilValues.getOgh();

        {
            // Test the following wg cases for a ghosted v:
            // {not touching border, touching upper border, touching lower border, touching left border, touching right border,
            //  all 4 corners (4 cases)}
            std::vector<std::vector<int>> wgs{
                {1, 1},
                {0, 1},
                // {n / wg_size_x - 1, 1},
                // {1, 0},
                // {1, o / wg_size_y - 1},
                // {0, 0},
                // {n / wg_size_x - 1, 0},
                // {0, o / wg_size_y - 1},
                {n / wg_size_x - 1, o / wg_size_y - 1}};

            for (auto wg : wgs)
            {
                // Test wg in the upper left corner
                int wg_num_x = wg[0];
                int wg_num_y = wg[1];

                { // Apply stencil for a grid point not touching any border of the current yz-plane and not in a ghosted x-plane
                    int p = 0;

                    // Load some planes first
                    for (int jloc = 0; jloc < wg_size_x; jloc++)
                        for (int kloc = 0; kloc < wg_size_y; kloc++)
                        {
                            int j = wg_num_x * wg_size_x + jloc;
                            int k = wg_num_y * wg_size_y + kloc;

                            load_plane(p - 1, 0, jloc, kloc, wg_size_x, wg_size_y, locmem_size_x, locmem_size_y, locmem.field1d().data(), v_cub.field1d().data(), m, n, o, mgh, ngh, ogh, j, k, gh);
                            load_plane(p, 3, jloc, kloc, wg_size_x, wg_size_y, locmem_size_x, locmem_size_y, locmem.field1d().data(), v_cub.field1d().data(), m, n, o, mgh, ngh, ogh, j, k, gh);
                            load_plane(p + 1, 2, jloc, kloc, wg_size_x, wg_size_y, locmem_size_x, locmem_size_y, locmem.field1d().data(), v_cub.field1d().data(), m, n, o, mgh, ngh, ogh, j, k, gh);
                        }

                    int jloc = 2;
                    int kloc = 2;
                    int j = wg_num_x * wg_size_x + jloc;
                    int k = wg_num_y * wg_size_y + kloc;

                    int ioff = ngh * ogh;
                    int joff = ogh;
                    int index = (p + gh) * ioff + (j + gh) * ogh + k + gh;
                    int index_localmem_plane = (jloc + 2) * locmem_size_y + kloc + 2;
                    int joff_localmem = locmem_size_y;
                    int svno = stencilValues.getNgh() * stencilValues.getOgh();
                    // // offset inside one coefficient grid that points to the coefficient for the current grid point. Must consider different amount of ghosts for v and sv.
                    int index_sv = (p + svgh) * svno + (j + svgh) * stencilValues.getOgh() + (k + svgh);

                    double s_act = apply_stencil(locmem[0][0], locmem[3][0], locmem[2][0], stencilValues.field1d().data(), svGridSize, index_sv, index_localmem_plane, joff_localmem);
                    double s_exp = apply_stencil_check(v.data(), stencilValues.field1d().data(), svGridSize, index_sv, index, joff, ioff);

                    CAPTURE(p, j, k, jloc, kloc, index, index_localmem_plane, index_sv, s_exp, s_act);
                    REQUIRE(s_act == s_exp);
                }

                { // Apply stencil for a grid point not touching any border of the current yz-plane but in a ghosted x-plane:
                    // If the first real x-plane has index p=0, apply stencil in x-plane p = -1 = m-1.
                    int p = m - 1;

                    // Load some planes first
                    for (int jloc = 0; jloc < wg_size_x; jloc++)
                        for (int kloc = 0; kloc < wg_size_y; kloc++)
                        {
                            int j = wg_num_x * wg_size_x + jloc;
                            int k = wg_num_y * wg_size_y + kloc;

                            load_plane(p - 1, 0, jloc, kloc, wg_size_x, wg_size_y, locmem_size_x, locmem_size_y, locmem.field1d().data(), v_cub.field1d().data(), m, n, o, mgh, ngh, ogh, j, k, gh);
                            load_plane(p, 3, jloc, kloc, wg_size_x, wg_size_y, locmem_size_x, locmem_size_y, locmem.field1d().data(), v_cub.field1d().data(), m, n, o, mgh, ngh, ogh, j, k, gh);
                            load_plane(p + 1, 2, jloc, kloc, wg_size_x, wg_size_y, locmem_size_x, locmem_size_y, locmem.field1d().data(), v_cub.field1d().data(), m, n, o, mgh, ngh, ogh, j, k, gh);
                        }

                    int jloc = 2;
                    int kloc = 2;
                    int j = wg_num_x * wg_size_x + jloc;
                    int k = wg_num_y * wg_size_y + kloc;

                    int ioff = ngh * ogh;
                    int joff = ogh;
                    int index = (p + gh) * ioff + (j + gh) * ogh + k + gh;
                    int index_localmem_plane = (jloc + 2) * locmem_size_y + kloc + 2;
                    int joff_localmem = locmem_size_y;
                    int svno = stencilValues.getNgh() * stencilValues.getOgh();
                    // // offset inside one coefficient grid that points to the coefficient for the current grid point. Must consider different amount of ghosts for v and sv.
                    int index_sv = (p + svgh) * svno + (j + svgh) * stencilValues.getOgh() + (k + svgh);

                    double s_act = apply_stencil(locmem[0][0], locmem[3][0], locmem[2][0], stencilValues.field1d().data(), svGridSize, index_sv, index_localmem_plane, joff_localmem);
                    double s_exp = apply_stencil_check(v.data(), stencilValues.field1d().data(), svGridSize, index_sv, index, joff, ioff);

                    CAPTURE(p, j, k, jloc, kloc, index, index_localmem_plane, index_sv, s_exp, s_act);
                    REQUIRE(s_act == s_exp);
                }

                { // Apply stencil for a grid point touching a border of the current yz-plane
                    int p = 0;

                    // Load some planes first
                    for (int jloc = 0; jloc < wg_size_x; jloc++)
                        for (int kloc = 0; kloc < wg_size_y; kloc++)
                        {
                            int j = wg_num_x * wg_size_x + jloc;
                            int k = wg_num_y * wg_size_y + kloc;

                            load_plane(p - 1, 0, jloc, kloc, wg_size_x, wg_size_y, locmem_size_x, locmem_size_y, locmem.field1d().data(), v_cub.field1d().data(), m, n, o, mgh, ngh, ogh, j, k, gh);
                            load_plane(p, 3, jloc, kloc, wg_size_x, wg_size_y, locmem_size_x, locmem_size_y, locmem.field1d().data(), v_cub.field1d().data(), m, n, o, mgh, ngh, ogh, j, k, gh);
                            load_plane(p + 1, 2, jloc, kloc, wg_size_x, wg_size_y, locmem_size_x, locmem_size_y, locmem.field1d().data(), v_cub.field1d().data(), m, n, o, mgh, ngh, ogh, j, k, gh);
                        }

                    int jloc = 0;
                    int kloc = 0;
                    int j = wg_num_x * wg_size_x + jloc;
                    int k = wg_num_y * wg_size_y + kloc;

                    int ioff = ngh * ogh;
                    int joff = ogh;
                    int index = (p + gh) * ioff + (j + gh) * ogh + k + gh;
                    int index_localmem_plane = (jloc + 2) * locmem_size_y + kloc + 2;
                    int joff_localmem = locmem_size_y;
                    int svno = stencilValues.getNgh() * stencilValues.getOgh();
                    // // offset inside one coefficient grid that points to the coefficient for the current grid point. Must consider different amount of ghosts for v and sv.
                    int index_sv = (p + svgh) * svno + (j + svgh) * stencilValues.getOgh() + (k + svgh);

                    double s_act = apply_stencil(locmem[0][0], locmem[3][0], locmem[2][0], stencilValues.field1d().data(), svGridSize, index_sv, index_localmem_plane, joff_localmem);
                    double s_exp = apply_stencil_check(v.data(), stencilValues.field1d().data(), svGridSize, index_sv, index, joff, ioff);

                    CAPTURE(p, j, k, jloc, kloc, index, index_localmem_plane, index_sv, s_exp, s_act, wg_num_x, wg_num_y, wg_size_x, wg_size_y);
                    REQUIRE(s_act == s_exp);
                }
            }
        }
    }

    // since we're not really running in parallel, we need to load a plane by iterating over all local indices.
    auto load_entire_plane = [&](int pglob, int ploc,
                                 //  int jloc, int kloc,
                                 int wg_size_x, int wg_size_y,
                                 int locmem_size_x, int locmem_size_y,
                                 double* locmem, double* v,
                                 int m, int n, int o,
                                 int mgh, int ngh, int ogh,
                                 //  int j, int k,
                                 int vgh,
                                 int wg_num_x, int wg_num_y)
    {
        for (int jloc = 0; jloc < wg_size_x; jloc++)
            for (int kloc = 0; kloc < wg_size_y; kloc++)
            {
                int j = wg_num_x * wg_size_x + jloc;
                int k = wg_num_y * wg_size_y + kloc;
                load_plane(pglob, ploc,
                           jloc, kloc,
                           wg_size_x, wg_size_y,
                           locmem_size_x, locmem_size_y,
                           locmem, v,
                           m, n, o,
                           mgh, ngh, ogh,
                           j, k,
                           vgh);
            }
    };

    auto dump_locmem = [](int locmem_size_x, int locmem_size_y, int locmem_size_xy, double* locmem)
    {
        for (int c = 0; c < 5; c++)
            for (int a = 0; a < locmem_size_x; a++)
                for (int b = 0; b < locmem_size_y; b++)
                {
                    std::cout << c << "\t" << a << "\t" << b << "\t" << std::scientific << std::setprecision(17) << locmem[c * locmem_size_xy + a * locmem_size_y + b] << std::endl;
                }
    };

    auto dump_globmem = [](int mgh, int ngh, int ogh, double* globmem)
    {
        for (int i = 0; i < mgh; i++)
            for (int j = 0; j < ngh; j++)
                for (int k = 0; k < ogh; k++)
                {
                    std::cout << i << "\t" << j << "\t" << k << "\t" << std::scientific << std::setprecision(17) << globmem[i * ngh * ogh + j * ogh + k] << std::endl;
                }
    };

    // Simulates the opencl kernel implementing 2d stream temporal tiling Jacobi
    auto jacobi_2d_stream_temp_tiling_2iters = [&](
                                                   //    int jloc,
                                                   //    int kloc,
                                                   //    int j,
                                                   //    int k,
                                                   int wg_size_x,
                                                   int wg_size_y,
                                                   int locmem_size_x,
                                                   int locmem_size_y,
                                                   int locmem_size_xy,
                                                   int mgh,
                                                   int ngh,
                                                   int ogh,
                                                   int m,
                                                   int n,
                                                   int o,
                                                   int gh,
                                                   const int svmgh, const int svngh, const int svogh,
                                                   const int svgh,
                                                   int wg_num_x,
                                                   int wg_num_y,
                                                   double* v_in,
                                                   double* v_out,
                                                   double* r,
                                                   double* f,
                                                   double* locmem,
                                                   double* stencilValues,
                                                   int svGridSize,
                                                   //    int index_sv,
                                                   //    int index,
                                                   //    int joff,
                                                   int p_start,
                                                   int p_end)
    {
        int ioff = ngh * ogh;
        int svno = svngh * svogh;
        // int index = (jloc + 2) * locmem_size_y + kloc + 2;
        // int index_sv = (p + svgh) * svno + (j + svgh) * svogh + (k + svgh);
        // int index_localmem_plane = (jloc + 2) * locmem_size_y + kloc + 2;
        int joff = locmem_size_y;
        // double stencilsum;
        //
        // TODO apply_stencil also for inner ghost layer!

        // only for testing
        std::vector<double> stencilsums_tmp(locmem_size_xy);

        // Helper function that calls a function for every wi of a wg. Needed because we don't actually parallelize here
        // and don't synch with a barrier.
        auto for_all_wis = [&](int p, std::function<void(int, int, int, int)> func)
        {
            for (int jloc = 0; jloc < wg_size_x; jloc++)
                for (int kloc = 0; kloc < wg_size_y; kloc++)
                {
                    int j = wg_num_x * wg_size_x + jloc;
                    int k = wg_num_y * wg_size_y + kloc;
                    int index = (jloc + 2) * locmem_size_y + kloc + 2;
                    int index_sv = (p + svgh) * svno + (j + svgh) * svogh + (k + svgh);
                    func(jloc, kloc, index, index_sv);
                }
        };

        // applies a stencil on the entire plane including inner ghost layer.
        // This is a helper for this test.
        auto apply_stencil_entire_plane = [&](
                                              double* front, double* center, double* back, double* stencilValues, int svGridSize, int joff_localmem,
                                              int joff_globalmem,
                                              int loc_size_x, int loc_size_y,
                                              double* locmem_store_base, double* v_glob, int p)
        {
            // apply on real grid points
            for_all_wis(p, [&](int jloc, int kloc, int index, int index_sv)
                        {
                            // apply on real grid point
                            stencilsums_tmp[(jloc + 2) * locmem_size_y + kloc + 2] = center[index] + 0.8 * (1.0 / stencilValues[index_sv + (9 + 3 + 1) * svGridSize]) * (f[index_sv] - apply_stencil(front, center, back, stencilValues, svGridSize, index_sv, index, joff_localmem));

                            // apply_stencil_check(v_in, stencilValues, svGridSize, index_sv, index_sv, joff_globalmem, ngh * ogh);

                            int index_tmp = index;
                            int index_sv_tmp = index_sv;

                            // apply in z dim
                            if (kloc < 1)
                            {
                                index_tmp = index - 1;
                                index_sv_tmp = index_sv - 1;
                                stencilsums_tmp[(jloc + 2) * locmem_size_y + kloc + 1] = center[index_tmp] + 0.8 * (1.0 / stencilValues[index_sv_tmp + (9 + 3 + 1) * svGridSize]) * (f[index_sv_tmp] - apply_stencil(front, center, back, stencilValues, svGridSize, index_sv_tmp, index_tmp, joff_localmem));
                            }
                            else if (kloc >= wg_size_y - 1)
                            {
                                index_tmp = index + 1;
                                index_sv_tmp = index_sv + 1;
                                stencilsums_tmp[(jloc + 2) * locmem_size_y + kloc + 3] = center[index_tmp] + 0.8 * (1.0 / stencilValues[index_sv_tmp + (9 + 3 + 1) * svGridSize]) * (f[index_sv_tmp] - apply_stencil(front, center, back, stencilValues, svGridSize, index_sv_tmp, index_tmp, joff_localmem));
                            }

                            // apply in y dim
                            if (jloc < 1)
                            {
                                index_tmp = index - joff;
                                index_sv_tmp = index_sv - joff_globalmem;
                                stencilsums_tmp[(jloc + 1) * locmem_size_y + kloc + 2] = center[index_tmp] + 0.8 * (1.0 / stencilValues[index_sv_tmp + (9 + 3 + 1) * svGridSize]) * (f[index_sv_tmp] - apply_stencil(front, center, back, stencilValues, svGridSize, index_sv_tmp, index_tmp, joff_localmem));
                            }
                            else if (jloc >= wg_size_x - 1)
                            {
                                index_tmp = index + joff;
                                index_sv_tmp = index_sv + joff_globalmem;
                                stencilsums_tmp[(jloc + 3) * locmem_size_y + kloc + 2] = center[index_tmp] + 0.8 * (1.0 / stencilValues[index_sv_tmp + (9 + 3 + 1) * svGridSize]) * (f[index_sv_tmp] - apply_stencil(front, center, back, stencilValues, svGridSize, index_sv_tmp, index_tmp, joff_localmem));
                            }

                            // apply in corners
                            if (kloc < 1 && jloc < 1)
                            {
                                index_tmp = index + (-joff - 1);
                                index_sv_tmp = index_sv + (-joff_globalmem - 1);
                                stencilsums_tmp[(jloc + 1) * locmem_size_y + kloc + 1] = center[index_tmp] + 0.8 * (1.0 / stencilValues[index_sv_tmp + (9 + 3 + 1) * svGridSize]) * (f[index_sv_tmp] - apply_stencil(front, center, back, stencilValues, svGridSize, index_sv_tmp, index_tmp, joff_localmem));

                                // apply_stencil_check(v_in, stencilValues, svGridSize, index_sv_tmp, index_sv_tmp, joff_globalmem, ngh * ogh);
                            }
                            else if (kloc >= wg_size_y - 1 && jloc < 1)
                            {
                                index_tmp = index + (-joff + 1);
                                index_sv_tmp = index_sv + (-joff_globalmem + 1);
                                stencilsums_tmp[(jloc + 1) * locmem_size_y + kloc + 3] = center[index_tmp] + 0.8 * (1.0 / stencilValues[index_sv_tmp + (9 + 3 + 1) * svGridSize]) * (f[index_sv_tmp] - apply_stencil(front, center, back, stencilValues, svGridSize, index_sv_tmp, index_tmp, joff_localmem));
                            }
                            else if (kloc < 1 && jloc >= wg_size_x - 1)
                            {
                                index_tmp = index + (joff - 1);
                                index_sv_tmp = index_sv + (joff_globalmem - 1);
                                stencilsums_tmp[(jloc + 3) * locmem_size_y + kloc + 1] = center[index_tmp] + 0.8 * (1.0 / stencilValues[index_sv_tmp + (9 + 3 + 1) * svGridSize]) * (f[index_sv_tmp] - apply_stencil(front, center, back, stencilValues, svGridSize, index_sv_tmp, index_tmp, joff_localmem));
                            }
                            else if (kloc >= wg_size_y - 1 && jloc >= wg_size_x - 1)
                            {
                                index_tmp = index + (joff + 1);
                                index_sv_tmp = index_sv + (joff_globalmem + 1);
                                stencilsums_tmp[(jloc + 3) * locmem_size_y + kloc + 3] = center[index_tmp] + 0.8 * (1.0 / stencilValues[index_sv_tmp + (9 + 3 + 1) * svGridSize]) * (f[index_sv_tmp] - apply_stencil(front, center, back, stencilValues, svGridSize, index_sv_tmp, index_tmp, joff_localmem));
                            }

                            //
                        });

            // store in locmem, i.e. do Jacobi step
            if (locmem_store_base != nullptr)
            {
                for_all_wis(p, [&](int jloc, int kloc, int index, int index_sv)
                            {
                                // barrier(CLK_LOCAL_MEM_FENCE);
                                // locmem_store_base[jloc * locmem_size_y + kloc] = stencilsums_tmp[jloc * locmem_size_y + kloc];

                                // store real grid point
                                locmem_store_base[(jloc + 2) * locmem_size_y + kloc + 2] = stencilsums_tmp[(jloc + 2) * locmem_size_y + kloc + 2];

                                int index_tmp = index;
                                int index_sv_tmp = index_sv;

                                // apply in z dim
                                if (kloc < 1)
                                {
                                    index_tmp = index - 1;
                                    index_sv_tmp = index_sv - 1;
                                    locmem_store_base[(jloc + 2) * locmem_size_y + kloc + 1] = stencilsums_tmp[(jloc + 2) * locmem_size_y + kloc + 1];
                                }
                                else if (kloc >= wg_size_y - 1)
                                {
                                    index_tmp = index + 1;
                                    index_sv_tmp = index_sv + 1;
                                    locmem_store_base[(jloc + 2) * locmem_size_y + kloc + 3] = stencilsums_tmp[(jloc + 2) * locmem_size_y + kloc + 3];
                                }

                                // apply in y dim
                                if (jloc < 1)
                                {
                                    index_tmp = index - joff;
                                    index_sv_tmp = index_sv - joff_globalmem;
                                    locmem_store_base[(jloc + 1) * locmem_size_y + kloc + 2] = stencilsums_tmp[(jloc + 1) * locmem_size_y + kloc + 2];
                                }
                                else if (jloc >= wg_size_x - 1)
                                {
                                    index_tmp = index + joff;
                                    index_sv_tmp = index_sv + joff_globalmem;
                                    locmem_store_base[(jloc + 3) * locmem_size_y + kloc + 2] = stencilsums_tmp[(jloc + 3) * locmem_size_y + kloc + 2];
                                }

                                // apply in corners
                                if (kloc < 1 && jloc < 1)
                                {
                                    index_tmp = index + (-joff - 1);
                                    index_sv_tmp = index_sv + (-joff_globalmem - 1);
                                    locmem_store_base[(jloc + 1) * locmem_size_y + kloc + 1] = stencilsums_tmp[(jloc + 1) * locmem_size_y + kloc + 1];
                                }
                                else if (kloc >= wg_size_y - 1 && jloc < 1)
                                {
                                    index_tmp = index + (-joff + 1);
                                    index_sv_tmp = index_sv + (-joff_globalmem + 1);
                                    locmem_store_base[(jloc + 1) * locmem_size_y + kloc + 3] = stencilsums_tmp[(jloc + 1) * locmem_size_y + kloc + 3];
                                }
                                else if (kloc < 1 && jloc >= wg_size_x - 1)
                                {
                                    index_tmp = index + (joff - 1);
                                    index_sv_tmp = index_sv + (joff_globalmem - 1);
                                    locmem_store_base[(jloc + 3) * locmem_size_y + kloc + 1] = stencilsums_tmp[(jloc + 3) * locmem_size_y + kloc + 1];
                                }
                                else if (kloc >= wg_size_y - 1 && jloc >= wg_size_x - 1)
                                {
                                    index_tmp = index + (joff + 1);
                                    index_sv_tmp = index_sv + (joff_globalmem + 1);
                                    locmem_store_base[(jloc + 3) * locmem_size_y + kloc + 3] = stencilsums_tmp[(jloc + 3) * locmem_size_y + kloc + 3];
                                }

                                //
                            });
            }
            else
            {
                // store in global memory. Only real cells.
                // TODO index_sv only works if ghosts of v and stencilValues are equal. Need global index for v!
                for_all_wis(p, [&](int jloc, int kloc, int index, int index_sv)
                            { v_glob[index_sv] = stencilsums_tmp[(jloc + 2) * locmem_size_y + kloc + 2]; });
            }
        };

        // TODO incorporate p_start in prologue already

        // set any idx_ps ("index_plane_start"), i.e. the plane index that the wg maps to
        // fetch v plane i=idx_ps-2 and store in locmem[0] (need to get from back, if idx_ps<2, be careful about ghosts)
        load_entire_plane(m - 2, 0, wg_size_x, wg_size_y, locmem_size_x, locmem_size_y, locmem, v_in, m, n, o, mgh, ngh, ogh, gh, wg_num_x, wg_num_y);
        // fetch v plane i=idx_ps-1 and store in locmem[1] (need to get from back, if idx_ps<1, be careful about ghosts)
        load_entire_plane(m - 1, 1, wg_size_x, wg_size_y, locmem_size_x, locmem_size_y, locmem, v_in, m, n, o, mgh, ngh, ogh, gh, wg_num_x, wg_num_y);
        // fetch v plane i=idx_ps and store in locmem[2]
        load_entire_plane(0, 2, wg_size_x, wg_size_y, locmem_size_x, locmem_size_y, locmem, v_in, m, n, o, mgh, ngh, ogh, gh, wg_num_x, wg_num_y);
        // fetch v plane i=idx_ps+1 and store in locmem[3]
        load_entire_plane(1, 3, wg_size_x, wg_size_y, locmem_size_x, locmem_size_y, locmem, v_in, m, n, o, mgh, ngh, ogh, gh, wg_num_x, wg_num_y);
        // fetch v plane i=idx_ps+2 and store in locmem[4]
        load_entire_plane(2, 4, wg_size_x, wg_size_y, locmem_size_x, locmem_size_y, locmem, v_in, m, n, o, mgh, ngh, ogh, gh, wg_num_x, wg_num_y);

        // TODO apply_stencil must be mocked for all local indices, too! :(
        int p_outer = m - 1;

        // apply stencils in locmem[1] for grid points except outer border and store in locmem[0] (t=0)
        apply_stencil_entire_plane(&locmem[0], &locmem[1 * locmem_size_xy], &locmem[2 * locmem_size_xy],
                                   stencilValues, svGridSize, joff, ogh, locmem_size_x, locmem_size_y,
                                   &locmem[0], nullptr, p_outer);
        // for_all_wis([&](int jloc, int kloc, int index, int index_sv)
        //             { stencilsums_tmp[jloc * locmem_size_y + kloc] = apply_stencil(&locmem[0], &locmem[1 * locmem_size_xy], &locmem[2 * locmem_size_xy], stencilValues, svGridSize, index_sv, index, joff); });
        // for_all_wis([&](int jloc, int kloc, int index, int index_sv)
        //             {
        //                 // barrier(CLK_LOCAL_MEM_FENCE);
        //                 locmem[jloc * locmem_size_y + kloc] = stencilsums_tmp[jloc * locmem_size_y + kloc]; });
        // double stencilsums_tmp[jloc * locmem_size_y + kloc] = apply_stencil(&locmem[0], &locmem[1 * locmem_size_xy], &locmem[2 * locmem_size_xy], stencilValues, svGridSize, index_sv, index, joff);
        // // barrier(CLK_LOCAL_MEM_FENCE);
        // locmem[jloc * locmem_size_y + kloc] = stencilsums_tmp[jloc * locmem_size_y + kloc];

        // apply stencils in locmem[2] for grid points except outer border and store in locmem[1] (t=0)
        p_outer = 0;
        // index = (p + gh) * ioff + (j + gh) * ogh + k + gh;
        // index_sv = (p + svgh) * svno + (j + svgh) * svogh + (k + svgh);
        apply_stencil_entire_plane(&locmem[1 * locmem_size_xy], &locmem[2 * locmem_size_xy], &locmem[3 * locmem_size_xy],
                                   stencilValues, svGridSize, joff, ogh, locmem_size_x, locmem_size_y,
                                   &locmem[1 * locmem_size_xy], nullptr, p_outer);
        // for_all_wis([&](int jloc, int kloc, int index, int index_sv)
        //             { stencilsums_tmp[jloc * locmem_size_y + kloc] = apply_stencil(&locmem[1 * locmem_size_xy], &locmem[2 * locmem_size_xy], &locmem[3 * locmem_size_xy], stencilValues, svGridSize, index_sv, index, joff); });
        // for_all_wis([&](int jloc, int kloc, int index, int index_sv)
        //             {
        // // barrier(CLK_LOCAL_MEM_FENCE);
        // locmem[1 * locmem_size_xy + jloc * locmem_size_y + kloc] = stencilsums_tmp[jloc * locmem_size_y + kloc]; });

        // apply stencils in &locmem[3] for grid points except outer border and store in &locmem[2] (t=0)
        p_outer = 1;
        apply_stencil_entire_plane(&locmem[2 * locmem_size_xy], &locmem[3 * locmem_size_xy], &locmem[4 * locmem_size_xy],
                                   stencilValues, svGridSize, joff, ogh, locmem_size_x, locmem_size_y,
                                   &locmem[2 * locmem_size_xy], nullptr, p_outer);
        // index = (p + gh) * ioff + (j + gh) * ogh + k + gh;
        // index_sv = (p + svgh) * svno + (j + svgh) * svogh + (k + svgh);
        // for_all_wis([&](int jloc, int kloc, int index, int index_sv)
        //             { stencilsums_tmp[jloc * locmem_size_y + kloc] = apply_stencil(&locmem[2 * locmem_size_xy], &locmem[3 * locmem_size_xy], &locmem[4 * locmem_size_xy], stencilValues, svGridSize, index_sv, index, joff); });
        // for_all_wis([&](int jloc, int kloc, int index, int index_sv)
        //             {
        // // barrier(CLK_LOCAL_MEM_FENCE);
        // locmem[2 * locmem_size_xy + jloc * locmem_size_y + kloc] = stencilsums_tmp[jloc * locmem_size_y + kloc]; });
        // barrier(CLK_LOCAL_MEM_FENCE);

        // apply stencils in &locmem[1] for grid points except outer 2 borders and store in global memory (t=1)
        p_outer = 0;
        apply_stencil_entire_plane(&locmem[0 * locmem_size_xy], &locmem[1 * locmem_size_xy], &locmem[2 * locmem_size_xy],
                                   stencilValues, svGridSize, joff, ogh, locmem_size_x, locmem_size_y,
                                   nullptr, v_out, p_outer);
        // index = (p + gh) * ioff + (j + gh) * ogh + k + gh;
        // index_sv = (p + svgh) * svno + (j + svgh) * svogh + (k + svgh);
        // for_all_wis([&](int jloc, int kloc, int index, int index_sv)
        //             { stencilsums_tmp[jloc * locmem_size_y + kloc] = apply_stencil(&locmem[0], &locmem[1 * locmem_size_xy], &locmem[2 * locmem_size_xy], stencilValues, svGridSize, index_sv, index, joff); });
        // for_all_wis([&](int jloc, int kloc, int index, int index_sv)
        //             { v_in[index] = stencilsums_tmp[jloc * locmem_size_y + kloc]; });

        // TODO *********************** check content of locmem here against Jacobi result for 1 iter
        // CONTENT OK!
        // dump_locmem(locmem_size_x, locmem_size_y, locmem_size_xy, locmem);
        // TODO check content of v_in, i.e. the first result in global memory after 2 iters
        // OK!
        // dump_globmem(mgh, ngh, ogh, v_in);

        // ****** Prologue end ******

        int next_buf = 0;
        // for (int p = p_start; p < p_end; p++)
        // p_pouter in this loop is the plane index, for which the first iteration's stencil is at its center
        for (p_outer = 2; p_outer <= m + gh; p_outer++)
        {
            // load next entire plane in &locmem[next_buf] (with 2 ghost layers)
            load_entire_plane((p_outer + 1 >= m) ? (p_outer + 1) - m : p_outer + 1, next_buf, wg_size_x, wg_size_y, locmem_size_x, locmem_size_y, locmem, v_in, m, n, o, mgh, ngh, ogh, gh, wg_num_x, wg_num_y);
            // barrier(CLK_LOCAL_MEM_FENCE);

            //   apply stencils in &locmem[(next_buf-1) % 5] for grid points except outer border and store in &locmem[(next_buf-2) % 5] (t=0)
            // index = (p - 1 + gh) * ioff + (j + gh) * ogh + k + gh;
            // index_sv = (p - 1 + svgh) * svno + (j + svgh) * svogh + (k + svgh);
            apply_stencil_entire_plane(&locmem[((next_buf - 2 + 5) % 5) * locmem_size_xy], &locmem[((next_buf - 1 + 5) % 5) * locmem_size_xy], &locmem[next_buf * locmem_size_xy],
                                       stencilValues, svGridSize, joff, ogh, locmem_size_x, locmem_size_y,
                                       &locmem[((next_buf - 2 + 5) % 5) * locmem_size_xy], nullptr, p_outer);
            // for_all_wis([&](int jloc, int kloc, int index, int index_sv)
            //             { stencilsums_tmp[jloc * locmem_size_y + kloc] = apply_stencil(&locmem[next_buf * locmem_size_xy], &locmem[((next_buf - 1 + 5) % 5) * locmem_size_xy], &locmem[((next_buf - 2 + 5) % 5) * locmem_size_xy], stencilValues, svGridSize, index_sv, index, joff); });
            // // barrier(CLK_LOCAL_MEM_FENCE);
            // for_all_wis([&](int jloc, int kloc, int index, int index_sv)
            //             {
            //                 locmem[((next_buf - 2 + 5) % 5) * locmem_size_xy + jloc * locmem_size_y + kloc] = stencilsums_tmp[jloc * locmem_size_y + kloc];
            //                 // barrier(CLK_LOCAL_MEM_FENCE);
            //             });

            //   apply stencils in &locmem[(next_buf-3) % 5] for grid points except outer border and store in global memory (t=1)
            // p = p_inner - 1; // Hack for this test, as the global index is built using p, which is not passed as an argument.
            apply_stencil_entire_plane(&locmem[((next_buf - 4 + 5) % 5) * locmem_size_xy], &locmem[((next_buf - 3 + 5) % 5) * locmem_size_xy], &locmem[((next_buf - 2 + 5) % 5) * locmem_size_xy],
                                       stencilValues, svGridSize, joff, ogh, locmem_size_x, locmem_size_y,
                                       nullptr, v_out, p_outer - 1);
            // p++;
            // for_all_wis([&](int jloc, int kloc, int index, int index_sv)
            //             { stencilsums_tmp[jloc * locmem_size_y + kloc] = apply_stencil(&locmem[((next_buf - 4 + 5) % 5) * locmem_size_xy], &locmem[((next_buf - 4 + 5) % 5) * locmem_size_xy], &locmem[((next_buf - 2 + 5) % 5) * locmem_size_xy], stencilValues, svGridSize, index_sv, index, joff); });
            // for_all_wis([&](int jloc, int kloc, int index, int index_sv)
            //             { v_in[index] = stencilsums_tmp[jloc * locmem_size_y + kloc]; });

            next_buf = (next_buf + 1) % 5;
        }
    };

    // Tests the whole algorithm, i.e. gluing together load_plane and apply_stencil and advancing through x-planes.
    SECTION("algorithm_2iters")
    {
        int m = 16;
        int n = 16;
        int o = 16;
        int gh = 1;
        int mgh = m + 2 * gh;
        int ngh = n + 2 * gh;
        int ogh = o + 2 * gh;

        double omega = 0.8;
        double h2 = 1.0 / (double)(m * m);
        mgcl::MGCL_RESIDUAL_NORM resnorm = mgcl::MGCL_L2;
        mgcl::MGCL_STENCIL stencilType = mgcl::MGCL_VARYING;

        // int wg_size = 32;
        int wg_size_x = GENERATE(4, 8);
        int wg_size_y = 4;
        int grid_size = mgh * ngh * ogh;

        int locmem_size_x = (wg_size_x + 4);
        int locmem_size_y = (wg_size_y + 4);
        int locmem_size_xy = locmem_size_x * locmem_size_y;

        // std::vector<double> shmem((wg_size_x + 4) * (wg_size_y + 4) * 5); // need 5 planes in buffer
        mgcl::Cuboid locmem(5, locmem_size_x, locmem_size_y); // need 5 planes in buffer
        mgcl::Cuboid v_in_cub(m, n, o, gh, gh, gh);
        v_in_cub.fill1dIndex(false);
        mgcl::MultigridEngine::updateGhostsSeq(v_in_cub, nullptr, true, true);
        auto& v_in = v_in_cub.field1d();
        mgcl::Cuboid v_out_cub(m, n, o, gh, gh, gh);
        v_out_cub.fill1dIndex(false);
        mgcl::MultigridEngine::updateGhostsSeq(v_out_cub, nullptr, true, true);
        auto& v_out = v_out_cub.field1d();

        mgcl::Cuboid f_act(m, n, o, gh, gh, gh);
        mgcl::Cuboid r_act(m, n, o, gh, gh, gh);

        mgcl::Cuboid f_exp(m, n, o, gh, gh, gh);
        mgcl::Cuboid r_exp(m, n, o, gh, gh, gh);
        mgcl::Cuboid v_exp(m, n, o, gh, gh, gh);
        v_exp.fill1dIndex(false);
        mgcl::MultigridEngine::updateGhostsSeq(v_exp, nullptr, true, true);

        // v_exp.dumpToFile("v_input.txt");

        // v_exp.dumpToFile("v_exp.txt");

        int svgh = 1;
        mgcl::VaryingStencil stencilValues(m, n, o, 3, svgh, svgh, svgh);
        stencilValues.fill1dIndex(false);
        // stencilValues.fill(1.0, false); // TODO varying
        stencilValues.updateGhosts();

        int svGridSize = stencilValues.getMgh() * stencilValues.getNgh() * stencilValues.getOgh();

        // Load expected result using regular Jacobi
        mgcl::MultigridEngine::jacobiSeq(v_exp, f_exp, r_exp, omega, h2, 2, resnorm, stencilType, 1.0, &stencilValues, nullptr, true, true, true);

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

            for (auto wg : wgs)
            {
                v_in_cub.fill1dIndex(false);
                mgcl::MultigridEngine::updateGhostsSeq(v_in_cub, nullptr, true, true);

                // Test wg in the upper left corner
                int wg_num_x = wg[0];
                int wg_num_y = wg[1];

                {
                    // Load some planes first
                    // for (int jloc = 0; jloc < wg_size_x; jloc++)
                    //     for (int kloc = 0; kloc < wg_size_y; kloc++)
                    //     {
                    // int j = wg_num_x * wg_size_x + jloc;
                    // int k = wg_num_y * wg_size_y + kloc;

                    // int ioff = ngh * ogh;
                    // int joff = ogh;
                    // int index = (p + gh) * ioff + (j + gh) * ogh + k + gh;
                    // int index_localmem_plane = (jloc + 2) * locmem_size_y + kloc + 2;
                    // int joff_localmem = locmem_size_y;
                    // int svno = stencilValues.getNgh() * stencilValues.getOgh();
                    // // // offset inside one coefficient grid that points to the coefficient for the current grid point. Must consider different amount of ghosts for v and sv.
                    // int index_sv = (p + svgh) * svno + (j + svgh) * stencilValues.getOgh() + (k + svgh);

                    jacobi_2d_stream_temp_tiling_2iters(
                        // jloc, kloc, j, k,
                        wg_size_x, wg_size_y, locmem_size_x, locmem_size_y, locmem_size_xy, mgh, ngh, ogh, m, n, o, gh,
                        stencilValues.getMgh(), stencilValues.getNgh(), stencilValues.getOgh(), stencilValues.getGhostsM(),
                        wg_num_x, wg_num_y, v_in_cub.field1d().data(), v_out_cub.field1d().data(),
                        r_act.field1d().data(), f_act.field1d().data(), locmem.field1d().data(), stencilValues.field1d().data(), svGridSize,
                        0, 0);
                    // CAPTURE(p, j, k, jloc, kloc, index, index_localmem_plane, index_sv);
                    // }

                    // v_exp.dumpToFile("v_exp.txt");
                    // v_in_cub.dumpToFile("v_act.txt");

                    for (int i = gh; i < m + gh; i++)
                        for (int j = wg_num_x * wg_size_x + gh; j < (wg_num_x + 1) * wg_size_x + gh; j++)
                            for (int k = wg_num_y * wg_size_y + gh; k < (wg_num_y + 1) * wg_size_y + gh; k++)
                            {
                                CAPTURE(i, j, k, wg_num_x, wg_num_y, wg_size_x, wg_size_y);
                                REQUIRE_THAT(v_out_cub[i][j][k], Catch::Matchers::WithinRel(v_exp[i][j][k], 1e-6));
                            }
                }
            }
        }
    }
}

TEST_CASE("ocl_temporal_tiling_localmem_2d_stream", "[.]")
{
    using size_t2 = struct
    {
        size_t x, y;
    };

    struct JacobiTBArgs
    {
        bool return_residual;
        int m;
        int n;
        int o;
        int mgh;
        int ngh;
        int ogh;
        double h2;
        int ghosts;
        double omega;
        int num_x_planes;

        cl_program program;
        cl_command_queue commands;
        size_t2 wgsize;

        mgcl::CuboidGpu& c_dVIn;
        mgcl::CuboidGpu& c_dVOut;
        mgcl::CuboidGpu& c_dF;
        mgcl::CuboidGpu& c_dR;
        mgcl::VaryingStencilGpu& c_stencilValues;

        mgcl::ProfilingData* pd;

        int moff;
        int noff;
        int ooff;
    };

    auto jacobi_ocl_tb_2iters = [](JacobiTBArgs& args)
    {
        int err;
        int m = args.m;
        int n = args.n;
        int o = args.o;
        int mgh = args.mgh;
        int ngh = args.ngh;
        int ogh = args.ogh;
        int store_res = 0;
        double res = -1;
        int idx_start = 0;

        cl_event ev;

        double h2 = 1.0 / static_cast<double>(m * m);
        double dinv = h2 / 6.0;

        // Create the compute kernel from the program
        const char* kernelName = "jacobi_iter_27point_varying_stencil_2d_local_mem_2iters";

        cl_kernel kernel = clCreateKernel(args.program, kernelName, &err);
        mgcl::mgclCheckError(err, "Creating kernel");

        cl_mem dVIn = args.c_dVIn.getBuffer();
        cl_mem dVOut = args.c_dVOut.getBuffer();
        cl_mem dF = args.c_dF.getBuffer();
        cl_mem dR = args.c_dR.getBuffer();

        // assign kernel arguments
        int pos = 0;
        int pos_idxstart = -1;
        int pos_storeres = -1;

        auto svbuf = args.c_stencilValues.getBuf();
        int svgh = args.c_stencilValues.getGh();
        int svmgh = args.c_stencilValues.getMgh();
        int svngh = args.c_stencilValues.getNgh();
        int svogh = args.c_stencilValues.getOgh();
        int svGridSize = svmgh * svngh * svogh;
        int locmem_size = (args.wgsize.x + 4) * (args.wgsize.y + 4) * 5; // 5 planes in local memory

        err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &dVIn);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dVOut);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dF);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dR);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &svbuf);
        err |= clSetKernelArg(kernel, ++pos, sizeof(double) * locmem_size, NULL);
        err |= clSetKernelArg(kernel, ++pos, sizeof(double), &args.omega);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &m);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &n);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &o);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &mgh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ngh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ogh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svmgh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svngh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svogh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &args.ghosts);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svgh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svGridSize);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &idx_start);
        pos_idxstart = pos;
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &store_res);
        pos_storeres = pos;
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &args.num_x_planes);
        mgcl::mgclCheckError(err, "Setting kernel arguments");

        // One work-item per real grid point in a yz-plane
        // TODO adjust when not streaming along whole x-dim
        size_t num_dim_x_wis = m / args.num_x_planes;
        size_t global[3] = {num_dim_x_wis, static_cast<size_t>(n), static_cast<size_t>(o)};
        size_t local[3] = {1, args.wgsize.x, args.wgsize.y};

        // No padding of work-items. Instead, sum of wgs must match global grid size
        assert((n / args.wgsize.x) * args.wgsize.x == n && "n is not divisible by wgsize.x!");
        assert((o / args.wgsize.y) * args.wgsize.y == o && "o is not divisible by wgsize.y!");
        assert((m / args.num_x_planes) * args.num_x_planes == m && "m is not divisible by num_x_planes!");

        err = clEnqueueNDRangeKernel(args.commands, kernel, 3, NULL, global, local, 0, NULL, &ev);
        mgcl::mgclCheckError(err, "Enqueueing kernel");

        if (args.pd)
        {
            args.pd->addMeasurement(args.commands, ev, kernelName,
                                    {global[0], global[1], global[2]},
                                    {local[0], local[1], local[2]});
        }
        mgcl::mgclCheckError(clReleaseEvent(ev), "clReleaseEvent");

        clReleaseKernel(kernel);
    };

    int m = 16;
    int n = 16;
    int o = 16;
    int gh = 1;
    int mgh = m + 2 * gh;
    int ngh = n + 2 * gh;
    int ogh = o + 2 * gh;

    double omega = 0.8;
    double h2 = 1.0 / (double)(m * m);
    mgcl::MGCL_RESIDUAL_NORM resnorm = mgcl::MGCL_L2;
    mgcl::MGCL_STENCIL stencilType = mgcl::MGCL_VARYING;

    // int wg_size = 32;
    int wg_size_x = GENERATE(4, 8);
    int wg_size_y = 4;
    int grid_size = mgh * ngh * ogh;

    int num_x_planes = GENERATE_COPY(m, m / 2, m / 4);
    // int num_x_planes = m / 2;
    // int num_x_planes = m; // m / 2;

    int locmem_size_x = (wg_size_x + 4);
    int locmem_size_y = (wg_size_y + 4);
    int locmem_size_xy = locmem_size_x * locmem_size_y;

    mgcl::Cuboid v_in_cub(m, n, o, gh, gh, gh);
    v_in_cub.fill1dIndex(false);
    mgcl::MultigridEngine::updateGhostsSeq(v_in_cub, nullptr, true, true);
    auto& v_in = v_in_cub.field1d();
    mgcl::Cuboid v_out_cub(m, n, o, gh, gh, gh);
    v_out_cub.fill1dIndex(false);
    mgcl::MultigridEngine::updateGhostsSeq(v_out_cub, nullptr, true, true);
    auto& v_out = v_out_cub.field1d();

    mgcl::Cuboid f_act(m, n, o, gh, gh, gh);
    mgcl::Cuboid r_act(m, n, o, gh, gh, gh);

    mgcl::Cuboid f_exp(m, n, o, gh, gh, gh);
    mgcl::Cuboid r_exp(m, n, o, gh, gh, gh);
    mgcl::Cuboid v_exp(m, n, o, gh, gh, gh);
    v_exp.fill1dIndex(false);
    mgcl::MultigridEngine::updateGhostsSeq(v_exp, nullptr, true, true);

    // v_exp.dumpToFile("v_input.txt");

    // v_exp.dumpToFile("v_exp.txt");

    int svgh = 1;
    mgcl::VaryingStencil stencilValues(m, n, o, 3, svgh, svgh, svgh);
    // stencilValues.fill1dIndex(false);
    stencilValues.fill(1.0, false); // TODO varying
    stencilValues.updateGhosts();

    int svGridSize = stencilValues.getMgh() * stencilValues.getNgh() * stencilValues.getOgh();

    // Load expected result using regular Jacobi
    mgcl::MultigridEngine::jacobiSeq(v_exp, f_exp, r_exp, omega, h2, 2, resnorm, stencilType, 1.0, &stencilValues, nullptr, true, true, true);

    // Dummy problem for initializing OpenCL environment
    auto v_dummy = std::make_shared<mgcl::Cuboid>(1, 1, 1);
    auto f_dummy = std::make_shared<mgcl::Cuboid>(1, 1, 1);
    mgcl::Problem p(m, n, o, f_dummy, v_dummy);
    p.setDeviceType(CL_DEVICE_TYPE_GPU);
    p.setUseOpencl(true);
    p.setKernelFile("kernel_optimizations.cl");
    p.setPrintKernelLog(true);
    p.setProfilingEnabled(true);
    p.getOpenCLHelper().init();

    // Create gpu buffers
    mgcl::CuboidGpu d_vin(p.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, v_in_cub);
    mgcl::CuboidGpu d_vout(p.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, v_out_cub);
    mgcl::CuboidGpu d_f(p.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, f_act);
    mgcl::CuboidGpu d_r(p.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, r_act);
    mgcl::VaryingStencilGpu d_sv(m, n, o, 3, gh, p.getContext(), p.getCommands(), p.getProgram());
    d_sv.fill(stencilValues, p.getCommands(), true);

    JacobiTBArgs args{
        false,
        m,
        n,
        o,
        mgh,
        ngh,
        ogh,
        h2,
        gh,
        omega,
        num_x_planes,
        p.getProgram(),
        p.getCommands(),
        {4, 4},
        d_vin,
        d_vout,
        d_f,
        d_r,
        d_sv,
        p.getProfilingData(),
        0,
        0,
        0};

    jacobi_ocl_tb_2iters(args);

    if (p.isProfilingEnabled())
    {
        p.getProfilingData()->printBestTimingsPerKernel();
    }

    // Read result
    d_vout.read(p.getCommands(), &v_out_cub, true);
    mgcl::MultigridEngine::updateGhostsSeq(v_out_cub, nullptr, true, true);

    v_exp.dumpToFile("v_exp.txt");
    v_out_cub.dumpToFile("v_act.txt");

    REQUIRE(v_out_cub.isEqual(v_exp));
}
