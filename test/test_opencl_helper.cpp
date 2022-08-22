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

    SECTION("init kernelDir")
    {
        std::string filePath = __FILE__;
        std::string dirPath = filePath.substr(0, filePath.rfind("/"));
        openCLHelper.setKernelDir(dirPath.append("/../build/"));

        int err = openCLHelper.init();
        REQUIRE(err == CL_SUCCESS);
        REQUIRE(openCLHelper.getProgram() != nullptr);
    }

    SECTION("reusing OpenCL platform")
    {
        int err = openCLHelper.init();
        REQUIRE(err == CL_SUCCESS);

        mgcl::Cuboid v2(3, 3, 3);
        mgcl::Cuboid f2(3, 3, 3);
        mgcl::Problem p2(3, 3, 3, f2, v2);
        p2.setReuseOpenclBuffers(true);
        mgcl::OpenCLHelper openCLHelper2(&p2);

        openCLHelper2.setContext(openCLHelper.getContext());
        openCLHelper2.setCommands(openCLHelper.getCommands());
        openCLHelper2.setDeviceId(openCLHelper.getDeviceId());

        err = openCLHelper2.init();
        REQUIRE(err == CL_SUCCESS);

        REQUIRE(openCLHelper2.getContext() == openCLHelper.getContext());
        REQUIRE(openCLHelper2.getCommands() == openCLHelper.getCommands());
        REQUIRE(openCLHelper2.getDeviceId() == openCLHelper.getDeviceId());
    }
}
