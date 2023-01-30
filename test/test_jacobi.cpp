#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cmath>
#include <iostream>

#include "../cuboid.hpp"
#include "../multigrid_engine.hpp"
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

    SECTION("seq L2-norm 7point")
    {
        auto stencilValues = std::make_unique<mgcl::VaryingStencil3x3x3>(1, 1, 1, 0, 0, 0); // just a dummy
        double res = mgcl::MultigridEngine::jacobiSeq(*c_in_v, *c_in_f, *c_in_r, omega, maxiter,
                                                      mgcl::MGCL_L2, mgcl::MGCL_LAPLACE_7POINT, stencilFactor, *stencilValues, true);

        CHECK(fabs(res - 4.02895897954478714e+04) < 1e-7);
        CHECK(c_in_v->isEqual(*c_expected_out_v));
        CHECK(c_in_r->isEqual(*c_expected_out_r));
    }

    SECTION("OpenCL GPU L2-norm 7point")
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
        p->setOmega(omega);
        p->setDeviceType(deviceType);

        mgcl_test::TestUtility tu(p);
        cl_mem d_in_f = tu.createOpenCLBuffer(*c_in_f);
        cl_mem d_in_v = tu.createOpenCLBuffer(*c_in_v);
        cl_mem d_in_v_out = tu.createOpenCLBuffer(*c_in_v);
        cl_mem d_in_r = tu.createOpenCLBuffer(*c_in_r);

        mgcl::Level level(p.get(), 0);
        level.setDF(d_in_f);
        level.setDVIn(d_in_v);
        level.setDVOut(d_in_v_out);
        level.setDR(d_in_r);

        double res = mgcl::MultigridEngine::jacobi(*p, level, maxiter, 1);
        tu.finish();

        auto c_r_out = tu.readOpenCLBuffer(d_in_r, m, n, o, ghosts_m, ghosts_n, ghosts_o);
        auto c_v_out = tu.readOpenCLBuffer(d_in_v, m, n, o, ghosts_m, ghosts_n, ghosts_o);

        CHECK(fabs(res - 4.02895897954478714e+04) < 1e-7);
        CHECK(c_v_out->isEqual(*c_expected_out_v));
        CHECK(c_r_out->isEqual(*c_expected_out_r));
    }
}


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

    mgcl_test::TestUtility *tu_tmp = new mgcl_test::TestUtility();
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
    mgcl::MultigridEngine::updateGhostsSeq(c_in_f_gh3);
    mgcl::MultigridEngine::updateGhostsSeq(c_in_v_gh3);
    mgcl::MultigridEngine::updateGhostsSeq(c_in_r_gh3);

    mgcl_test::TestUtility tu(p);
    cl_mem d_in_f = tu.createOpenCLBuffer(c_in_f_gh3);
    cl_mem d_in_v = tu.createOpenCLBuffer(c_in_v_gh3);
    cl_mem d_in_v_out = tu.createOpenCLBuffer(c_in_v_gh3);
    cl_mem d_in_r = tu.createOpenCLBuffer(c_in_r_gh3);

    mgcl::Level level(p.get(), 0);
    level.setDF(d_in_f);
    level.setDVIn(d_in_v);
    level.setDVOut(d_in_v_out);
    level.setDR(d_in_r);

    double res = mgcl::MultigridEngine::jacobi(*p, level, maxiter, 1);
    tu.finish();

    // read back from device and copy to Cuboid with ghosts = 1
    auto c_r_out = tu.readOpenCLBuffer(d_in_r, m, n, o, ghosts, ghosts, ghosts);
    auto c_v_out = tu.readOpenCLBuffer(d_in_v, m, n, o, ghosts, ghosts, ghosts);
    for (int i = 0; i < c_in_f->getM(); i++)
        for (int j = 0; j < c_in_f->getN(); j++)
            for (int k = 0; k < c_in_f->getO(); k++)
            {
                (*c_in_v)[i + 1][j + 1][k + 1] = (*c_v_out)[i + ghosts][j + ghosts][k + ghosts];
                (*c_in_r)[i + 1][j + 1][k + 1] = (*c_r_out)[i + ghosts][j + ghosts][k + ghosts];
            }

    c_in_r->dumpToFile("c_in_r->txt");
    c_expected_out_r->dumpToFile("c_expected_out_r.txt");

    CHECK(fabs(res - 4.02895897954478714e+04) < 1e-7);
    CHECK(c_in_v->isEqual(*c_expected_out_v));
    CHECK(c_in_r->isEqual(*c_expected_out_r));
}
