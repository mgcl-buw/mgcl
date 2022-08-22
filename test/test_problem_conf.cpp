#include <catch2/catch_test_macros.hpp>

#include "../cuboid.hpp"
#include "../problem.hpp"

/**
 * @brief Tests if config parameters of Problem are working correctly, including setting things up.
 *
 */
TEST_CASE("Problem conf")
{
    auto v = mgcl::Cuboid(8, 8, 8);
    auto f = mgcl::Cuboid(8, 8, 8);
    mgcl::Problem p(8, 8, 8, v.getData(), f.getData());

    SECTION("default values")
    {
        // REQUIRE(p.getV() == v);
        // REQUIRE(p.getF() == f);
        REQUIRE(p.getM() == 8);
        REQUIRE(p.getN() == 8);
        REQUIRE(p.getO() == 8);
        REQUIRE(p.getGhosts() == 1);
        REQUIRE(p.getGhostsIn() == 0);
        REQUIRE(p.getNu1() == 2);
        REQUIRE(p.getNu2() == 2);
        REQUIRE(p.getOmega() == 0.8);
        REQUIRE(p.getMaxiterVcycles() == 5);
        REQUIRE(p.getTol() == 1e-7);
        REQUIRE(p.getMaxlevel() == -1);
        REQUIRE(p.getResidualNorm() == mgcl::MGCL_L2);
        REQUIRE(p.getStencil() == mgcl::MGCL_7POINT);

        REQUIRE(p.getOpenCLHelper() == nullptr);
        // REQUIRE(p.getDeviceType() == CL_DEVICE_TYPE_DEFAULT);
        // REQUIRE(p.getKernelDir() == "./");
        // REQUIRE(p.getDeviceName() == "");
        // REQUIRE(p.getDeviceId() == nullptr);
        // REQUIRE(p.getCommands() == nullptr);
        // REQUIRE(p.getContext() == nullptr);
        REQUIRE(p.getDV() == nullptr);
        REQUIRE(p.getDF() == nullptr);
        REQUIRE(p.getUseOpencl() == false);
        REQUIRE(p.getReuseOpenclBuffers() == false);
        REQUIRE(p.getCopyBufferData() == false);
        REQUIRE(p.getReadResults() == false);
        REQUIRE(p.getUseLocalMemory() == false);

        REQUIRE(p.getJacobiWgSizeX() == 16);
        REQUIRE(p.getJacobiWgSizeY() == 16);
        REQUIRE(p.getJacobiIterationsPerKernel() == 3);

        REQUIRE(p.getStencilSizeMultiplier() == 1);
        REQUIRE(p.getStencilValues() == nullptr);
        REQUIRE(p.getDStencilValues() == nullptr);
        REQUIRE(p.getRestrictProlongateStencil() == true);
    }
}

TEST_CASE("checkParameters")
{
    mgcl::Cuboid v(8, 8, 8);
    mgcl::Cuboid f(8, 8, 8);

    SECTION("m,n,o wrong")
    {
        mgcl::Problem pm(0, 1, 1, v.getData(), f.getData());
        mgcl::Problem pn(0, 1, 1, v.getData(), f.getData());
        mgcl::Problem po(0, 1, 1, v.getData(), f.getData());

        REQUIRE(pm.checkParameters() == false);
        REQUIRE(pn.checkParameters() == false);
        REQUIRE(po.checkParameters() == false);
    }

    SECTION("v nullptr")
    {
        mgcl::Problem pm(8, 8, 8, nullptr, f.getData());
        REQUIRE(pm.checkParameters() == false);
    }

    SECTION("f nullptr")
    {
        mgcl::Problem pm(8, 8, 8, v.getData(), nullptr);
        REQUIRE(pm.checkParameters() == false);
    }

    SECTION("ghosts 0")
    {
        mgcl::Problem pm(8, 8, 8, v.getData(), f.getData());
        pm.setGhosts(0);
        REQUIRE(pm.checkParameters() == false);
    }

    SECTION("ghosts_in -1")
    {
        mgcl::Problem pm(8, 8, 8, v.getData(), f.getData());
        pm.setGhostsIn(-1);
        REQUIRE(pm.checkParameters() == false);
    }

    SECTION("stencil_values not set")
    {
        mgcl::Problem pm(8, 8, 8, v.getData(), f.getData());
        pm.setStencil(mgcl::MGCL_STENCIL::MGCL_19POINT_VARSYM);
        REQUIRE(pm.checkParameters() == false);
    }

    SECTION("all good")
    {
        mgcl::Problem pm(8, 8, 8, v.getData(), f.getData());
        REQUIRE(pm.checkParameters() == true);
    }
}

TEST_CASE("calculateAndSetMaxLevel")
{
    double ***f = nullptr;
    double ***v = nullptr;

    SECTION("m min")
    {
        mgcl::Problem p(4, 8, 8, f, v);

        REQUIRE(p.getMaxlevel() == -1);
        p.calculateAndSetMaxLevel();
        REQUIRE(p.getMaxlevel() == 3);
    }

    SECTION("n min")
    {
        mgcl::Problem p(16, 8, 16, f, v);

        REQUIRE(p.getMaxlevel() == -1);
        p.calculateAndSetMaxLevel();
        REQUIRE(p.getMaxlevel() == 4);
    }

    SECTION("o min")
    {
        mgcl::Problem p(32, 32, 16, f, v);

        REQUIRE(p.getMaxlevel() == -1);
        p.calculateAndSetMaxLevel();
        REQUIRE(p.getMaxlevel() == 5);
    }

    SECTION("user set valid")
    {
        mgcl::Problem p(32, 32, 32, f, v);
        p.setMaxlevel(2);

        p.calculateAndSetMaxLevel();
        REQUIRE(p.getMaxlevel() == 2);
    }

    SECTION("user set invalid")
    {
        mgcl::Problem p(4, 4, 4, f, v);
        p.setMaxlevel(8);

        p.calculateAndSetMaxLevel();
        REQUIRE(p.getMaxlevel() == 3);
    }
}
