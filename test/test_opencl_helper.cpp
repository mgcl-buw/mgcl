#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <iostream>

#include "../cuboid.hpp"
#include "../opencl_helper.hpp"
#include "../problem.hpp"
#include "test_utility.hpp"

TEST_CASE("OpenCLHelper")
{
    auto v = std::make_shared<mgcl::Cuboid>(4, 4, 4);
    auto f = std::make_shared<mgcl::Cuboid>(4, 4, 4);
    auto p = std::make_shared<mgcl::Problem>(4, 4, 4, f, v);
    mgcl::OpenCLHelper openCLHelper(p.get());

    SECTION("init default conf")
    {
        REQUIRE_FALSE(openCLHelper.isInitialized());
        int err = openCLHelper.init();
        REQUIRE(openCLHelper.isInitialized());

        REQUIRE(err == CL_SUCCESS);
        REQUIRE(openCLHelper.getCommands() != nullptr);
        REQUIRE(openCLHelper.getContext() != nullptr);
        REQUIRE(openCLHelper.getDeviceId() != nullptr);
        REQUIRE(openCLHelper.getDeviceName() == "");
        REQUIRE(openCLHelper.getKernelDir() == "./");
        REQUIRE(openCLHelper.getProgram() != nullptr);
    }

    SECTION("init kernelDir")
    {
        std::string filePath = __FILE__;
        std::string dirPath = filePath.substr(0, filePath.rfind("/"));
        openCLHelper.setKernelDir(dirPath.append("/../build/"));

        REQUIRE_FALSE(openCLHelper.isInitialized());
        int err = openCLHelper.init();
        REQUIRE(openCLHelper.isInitialized());
        REQUIRE(err == CL_SUCCESS);
        REQUIRE(openCLHelper.getProgram() != nullptr);
    }

    SECTION("reusing OpenCL platform")
    {
        auto deviceType = GENERATE(CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU);

        if (!mgcl_test::TestUtility::deviceAvailable("", deviceType))
        {
            std::string typeName = deviceType == CL_DEVICE_TYPE_GPU ? "CL_DEVICE_TYPE_GPU" : "CL_DEVICE_TYPE_CPU";
            std::cout << "Skipping non-available device type '" << typeName << "'" << std::endl;
            return;
        }

        openCLHelper.setDeviceType(deviceType);

        REQUIRE_FALSE(openCLHelper.isInitialized());
        int err = openCLHelper.init();
        REQUIRE(openCLHelper.isInitialized());
        REQUIRE(err == CL_SUCCESS);

        auto v2 = std::make_shared<mgcl::Cuboid>(3, 3, 3);
        auto f2 = std::make_shared<mgcl::Cuboid>(3, 3, 3);
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

    SECTION("isInitialized")
    {
        REQUIRE(!openCLHelper.isInitialized());
        openCLHelper.init();
        REQUIRE(openCLHelper.isInitialized());
    }
}
