#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>

#include "../src/mgcl/cuboid.hpp"
#include "../src/mgcl/level.hpp"
#include "../src/mgcl/multigrid_engine.hpp"
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
                                                      nullptr, true, true, true);

        // REQUIRE_THAT(res, Catch::Matchers::WithinAbs(4.02895897954478714e+04, 1e-7));
        CHECK(c_in_v->isEqual(*c_expected_out_v));
        // CHECK(c_in_r->isEqual(*c_expected_out_r));
    }

    // TODO adjust expected results for expected r and norm of r?
    SECTION("OpenCL GPU L2-norm 7point periodic")
    {
        auto deviceType = CL_DEVICE_TYPE_GPU; // GENERATE(CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU);

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
        auto deviceType = CL_DEVICE_TYPE_GPU; // GENERATE(CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU);

        if (!mgcl_test::TestUtility::deviceAvailable("", deviceType))
        {
            std::string typeName = deviceType == CL_DEVICE_TYPE_GPU ? "CL_DEVICE_TYPE_GPU" : "CL_DEVICE_TYPE_CPU";
            std::cout << "Skipping non-available device type '" << typeName << "'" << std::endl;
            return;
        }

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
                                                          stencilFactor, nullptr, true, true, true);

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
        auto deviceType = CL_DEVICE_TYPE_GPU; // GENERATE(CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU);

        if (!mgcl_test::TestUtility::deviceAvailable("", deviceType))
        {
            std::string typeName = deviceType == CL_DEVICE_TYPE_GPU ? "CL_DEVICE_TYPE_GPU" : "CL_DEVICE_TYPE_CPU";
            std::cout << "Skipping non-available device type '" << typeName << "'" << std::endl;
            return;
        }

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
                                                          true, true, true);

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
TEST_CASE("jacobi OpenCL L2-norm 7point localMemory", "[.]")
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
    int ghosts = 3;

    auto p = std::make_shared<mgcl::Problem>(m, n, o);
    p->setResidualNorm(mgcl::MGCL_L2);
    p->setStencilType(stencil);
    p->setGhosts(ghosts);
    p->setOmega(omega);
    p->setUseLocalMemory(true);
    p->setJacobiWgSizeX(8);
    p->setJacobiWgSizeY(8);

    mgcl_test::TestUtility* tu_tmp = new mgcl_test::TestUtility();
    if (tu_tmp->deviceAvailable("Quadro", p->getDeviceType()))
        p->setDeviceName("Quadro");
    delete tu_tmp;

    // create input Cuboids with different ghost cell count
    mgcl::Cuboid c_in_f_gh3(c_in_f->getM(), c_in_f->getN(), c_in_f->getO(), ghosts, ghosts, ghosts);
    mgcl::Cuboid c_in_v_gh3(c_in_v->getM(), c_in_v->getN(), c_in_v->getO(), ghosts, ghosts, ghosts);
    mgcl::Cuboid c_in_r_gh3(c_in_r->getM(), c_in_r->getN(), c_in_r->getO(), ghosts, ghosts, ghosts);
    for (int i = 0; i < c_in_f->getM(); i++)
        for (int j = 0; j < c_in_f->getN(); j++)
            for (int k = 0; k < c_in_f->getO(); k++)
            {
                c_in_f_gh3[i + ghosts][j + ghosts][k + ghosts] = (*c_in_f)[i + 1][j + 1][k + 1];
                c_in_v_gh3[i + ghosts][j + ghosts][k + ghosts] = (*c_in_v)[i + 1][j + 1][k + 1];
                c_in_r_gh3[i + ghosts][j + ghosts][k + ghosts] = (*c_in_r)[i + 1][j + 1][k + 1];
            }
    mgcl::MultigridEngine::updateGhostsSeq(c_in_f_gh3, nullptr, true, true);
    mgcl::MultigridEngine::updateGhostsSeq(c_in_v_gh3, nullptr, true, true);
    mgcl::MultigridEngine::updateGhostsSeq(c_in_r_gh3, nullptr, true, true);

    mgcl_test::TestUtility tu(p);
    auto d_in_f = std::make_shared<mgcl::CuboidGpu>(tu.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, c_in_f_gh3);
    auto d_in_v = std::make_shared<mgcl::CuboidGpu>(tu.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, c_in_v_gh3);
    auto d_in_v_out = std::make_shared<mgcl::CuboidGpu>(tu.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, c_in_v_gh3);
    auto d_in_r = std::make_shared<mgcl::CuboidGpu>(tu.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, c_in_r_gh3);

    mgcl::Level level(p.get(), 0);
    level.setDF(d_in_f);
    level.setDVIn(d_in_v);
    level.setDVOut(d_in_v_out);
    level.setDR(d_in_r);

    double res = mgcl::MultigridEngine::jacobi(*p, level, maxiter, 1);
    tu.finish();

    // read back from device and copy to Cuboid with ghosts = 1
    auto c_r_out = d_in_r->read(p->getCommands(), nullptr, true);
    auto c_v_out = d_in_v->read(p->getCommands(), nullptr, true);
    for (int i = 0; i < c_in_f->getM(); i++)
        for (int j = 0; j < c_in_f->getN(); j++)
            for (int k = 0; k < c_in_f->getO(); k++)
            {
                (*c_in_v)[i + 1][j + 1][k + 1] = (*c_v_out)[i + ghosts][j + ghosts][k + ghosts];
                (*c_in_r)[i + 1][j + 1][k + 1] = (*c_r_out)[i + ghosts][j + ghosts][k + ghosts];
            }

    // c_in_r->dumpToFile("c_in_r->txt");
    // c_expected_out_r->dumpToFile("c_expected_out_r.txt");

    REQUIRE_THAT(res, Catch::Matchers::WithinAbs(4.02895897954478714e+04, 1e-7));
    // CHECK(fabs(res - 4.02895897954478714e+04) < 1e-7);
    CHECK(c_in_v->isEqual(*c_expected_out_v));
    CHECK(c_in_r->isEqual(*c_expected_out_r));
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
    double stencilFactor = 1.0 / (30.0 * h * h);
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
                                                            stencilFactor, nullptr, false, true, true, stepsPerIter));

            // ghosts of f too small
            REQUIRE_THROWS(mgcl::MultigridEngine::jacobiSeq(v_gh, f, r_gh, omega, h * h, iters, resnorm, stencilType,
                                                            stencilFactor, nullptr, false, true, true, stepsPerIter));

            // ghosts of r too small
            REQUIRE_THROWS(mgcl::MultigridEngine::jacobiSeq(v_gh, f_gh, r, omega, h * h, iters, resnorm, stencilType,
                                                            stencilFactor, nullptr, false, true, true, stepsPerIter));
        }

        SECTION("throws when stencilValues null and stencilType varying")
        {
            REQUIRE_THROWS(mgcl::MultigridEngine::jacobiSeq(v_gh, f_gh, r_gh, omega, h * h, iters, resnorm, mgcl::MGCL_VARYING,
                                                            stencilFactor, nullptr, false, true, true, stepsPerIter));
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

    double stencilFactor = 1.0 / (30.0 * h * h);
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
        double res_exp = mgcl::MultigridEngine::jacobiSeq(v_in, f_in, r_in, omega, h * h, iters, resnorm, stencilType, stencilFactor, nullptr, true, true, 1);

        // Now calculate with gh > 1
        double res_act = mgcl::MultigridEngine::jacobiSeq(v_in_gh, f_in_gh, r_in_gh, omega, h * h, iters, resnorm, stencilType, stencilFactor, nullptr, true, true, stepsPerIter);

        REQUIRE_THAT(res_exp, Catch::Matchers::WithinAbs(res_act, 1e-7));
        REQUIRE(v_in.isEqual(v_in_gh));
        REQUIRE(r_in.isEqual(r_in_gh));
    }

    SECTION("Dirichlet 27p-Laplace")
    {
        mgcl::MGCL_STENCIL stencilType = mgcl::MGCL_LAPLACE_27POINT;

        // First calculate exptected result with regular ghost updates between iterations
        double res_exp = mgcl::MultigridEngine::jacobiSeq(v_in, f_in, r_in, omega, h * h, iters, resnorm, stencilType, stencilFactor, nullptr, true, false, 1);

        // Now calculate with gh > 1
        double res_act = mgcl::MultigridEngine::jacobiSeq(v_in_gh, f_in_gh, r_in_gh, omega, h * h, iters, resnorm, stencilType, stencilFactor, nullptr, true, false, stepsPerIter);

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
                                                          stencilType, stencilFactor, &stencilValuesExp, true, true, 1);

        // Now calculate with gh > 1
        double res_act = mgcl::MultigridEngine::jacobiSeq(v_in_gh, f_in_gh, r_in_gh, omega, h * h, iters, resnorm,
                                                          stencilType, stencilFactor, &stencilValuesAct,
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
    double stencilFactor = 1.0 / (30.0 * h * h);
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

    if (mgcl_test::TestUtility::deviceAvailable("", CL_DEVICE_TYPE_GPU))
    {
        // init problem with ghosts = 1
        auto p_exp = std::make_shared<mgcl::Problem>(m, n, o);
        p_exp->setResidualNorm(mgcl::MGCL_L2);
        p_exp->setGhosts(1);
        p_exp->setDeviceType(CL_DEVICE_TYPE_GPU);
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

            REQUIRE(sv_exp->getGhostsM() == 2);
            REQUIRE(sv_exp->getGhostsN() == 2);
            REQUIRE(sv_exp->getGhostsO() == 2);
            REQUIRE(sv_act->getGhostsM() == std::max(2, stepsPerIter));
            REQUIRE(sv_act->getGhostsN() == std::max(2, stepsPerIter));
            REQUIRE(sv_act->getGhostsO() == std::max(2, stepsPerIter));

            auto d_sv_exp = std::make_shared<mgcl::VaryingStencilGpu>(sv_exp->getM(), sv_exp->getN(), sv_exp->getO(), 3,
                                                                      sv_exp->getGhostsM(),
                                                                      tu_exp.getContext(), tu_exp.getCommands());
            auto d_sv_act = std::make_shared<mgcl::VaryingStencilGpu>(sv_act->getM(), sv_act->getN(), sv_act->getO(), 3,
                                                                      sv_act->getGhostsM(),
                                                                      tu_act.getContext(), tu_act.getCommands());

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
