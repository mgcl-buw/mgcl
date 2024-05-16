#include "catch2/catch_test_macros.hpp"

#include "../src/mgcl/multigrid_engine.hpp"
#include "../src/mgcl/problem.hpp"

#include <iostream>
#include <memory>

TEST_CASE("correct_error")
{
    int m, n, o;
    m = n = o = 4;
    auto v = std::make_shared<mgcl::Cuboid>(m, n, o);
    auto f = std::make_shared<mgcl::Cuboid>(m, n, o);
    mgcl::Problem p(m, n, o, f, v);

    p.setUseOpencl(true);
    p.setDeviceType(CL_DEVICE_TYPE_GPU);
    p.init();

    auto& lv = p.getLevelAt(0);

    // create test data on lv 0 and correct error
    mgcl::Cuboid lv_v_h(p.getM(), p.getN(), p.getO(), p.getGhosts(), p.getGhosts(), p.getGhosts());
    mgcl::Cuboid lv_r_h(p.getM(), p.getN(), p.getO(), p.getGhosts(), p.getGhosts(), p.getGhosts());
    lv_v_h.fillRandom();
    lv_r_h.fillRandom();

    auto lv_v_d = std::make_shared<mgcl::CuboidGpu>(p.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, lv_v_h);
    auto lv_r_d = std::make_shared<mgcl::CuboidGpu>(p.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, lv_r_h);

    lv.setDVIn(lv_v_d);
    lv.setDR(lv_r_d);

    mgcl::MultigridEngine::correctError(lv);
    p.getOpenCLHelper().finish();

    // calculate on host manually
    for (int i = 0; i < lv_v_h.field1d().size(); i++)
    {
        lv_v_h.field1d()[i] += lv_r_h.field1d()[i];
    }

    // read gpu buffers and check results
    auto lv_v_act = lv_v_d->read(p.getCommands(), nullptr, true);
    auto lv_r_act = lv_r_d->read(p.getCommands(), nullptr, true);

    REQUIRE(lv_v_act->isEqual(lv_v_h, 1e-7, true));
    REQUIRE(lv_r_act->isEqual(lv_r_h));
}