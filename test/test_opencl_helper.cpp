#include <catch2/catch_test_macros.hpp>

#include "../cuboid.hpp"
#include "../opencl_helper.hpp"
#include "../problem.hpp"

TEST_CASE("OpenCLHelper")
{
    mgcl::Cuboid v(4, 4, 4);
    mgcl::Cuboid f(4, 4, 4);
    mgcl::Problem p(4, 4, 4, f, v);
    mgcl::OpenCLHelper openCLHelper(&p);

    SECTION("init default conf")
    {
        int err = openCLHelper.init();

        REQUIRE(err == CL_SUCCESS);
        REQUIRE(openCLHelper.getCommands() != nullptr);
        REQUIRE(openCLHelper.getContext() != nullptr);
        REQUIRE(openCLHelper.getDeviceId() != nullptr);
        REQUIRE(openCLHelper.getDeviceName() == "");
        REQUIRE(openCLHelper.getDeviceType() == CL_DEVICE_TYPE_DEFAULT);
        REQUIRE(openCLHelper.getKernelDir() == "./");
        REQUIRE(openCLHelper.getProgram() != nullptr);
    }
}