#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <iostream>

#include "../src/mgcl/cuboid.hpp"
#include "../src/mgcl/opencl_helper.hpp"
#include "../src/mgcl/problem.hpp"
#include "test_utility.hpp"

TEST_CASE("OpenCLHelper")
{
    auto v = std::make_shared<mgcl::Cuboid>(4, 4, 4);
    auto f = std::make_shared<mgcl::Cuboid>(4, 4, 4);
    auto p = std::make_shared<mgcl::Problem>(4, 4, 4, f, v);
    p->setDeviceType(CL_DEVICE_TYPE_GPU);
    mgcl::OpenCLHelper openCLHelper(p.get());

    SECTION("init default conf")
    {
        REQUIRE_FALSE(openCLHelper.isInitialized());
        REQUIRE_NOTHROW(openCLHelper.init());
        REQUIRE(openCLHelper.isInitialized());

        REQUIRE(openCLHelper.getCommands() != nullptr);
        REQUIRE(openCLHelper.getContext() != nullptr);
        REQUIRE(openCLHelper.getDeviceId() != nullptr);
        REQUIRE(openCLHelper.getDeviceName() == "");
        REQUIRE(openCLHelper.getKernelFile() == "./mgcl.cl");
        REQUIRE(openCLHelper.getProgram() != nullptr);
    }

    SECTION("init kernelDir")
    {
        REQUIRE_FALSE(openCLHelper.isInitialized());
        REQUIRE_NOTHROW(openCLHelper.init());
        REQUIRE(openCLHelper.isInitialized());
        REQUIRE(openCLHelper.getProgram() != nullptr);
    }

    SECTION("reusing OpenCL platform")
    {
        auto deviceType = CL_DEVICE_TYPE_GPU; // GENERATE(CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU);

        if (!mgcl_test::TestUtility::deviceAvailable("", deviceType))
        {
            std::string typeName = deviceType == CL_DEVICE_TYPE_GPU ? "CL_DEVICE_TYPE_GPU" : "CL_DEVICE_TYPE_CPU";
            std::cout << "Skipping non-available device type '" << typeName << "'" << std::endl;
            return;
        }

        openCLHelper.setDeviceType(deviceType);

        REQUIRE_FALSE(openCLHelper.isInitialized());
        REQUIRE_NOTHROW(openCLHelper.init());
        REQUIRE(openCLHelper.isInitialized());

        auto v2 = std::make_shared<mgcl::Cuboid>(3, 3, 3);
        auto f2 = std::make_shared<mgcl::Cuboid>(3, 3, 3);
        mgcl::Problem p2(3, 3, 3, f2, v2);
        p2.setReuseOpenclBuffers(true);
        mgcl::OpenCLHelper openCLHelper2(&p2);

        openCLHelper2.setContext(openCLHelper.getContext());
        openCLHelper2.setCommands(openCLHelper.getCommands());
        openCLHelper2.setDeviceId(openCLHelper.getDeviceId());

        REQUIRE_NOTHROW(openCLHelper2.init());

        REQUIRE(openCLHelper2.getContext() == openCLHelper.getContext());
        REQUIRE(openCLHelper2.getCommands() == openCLHelper.getCommands());
        REQUIRE(openCLHelper2.getDeviceId() == openCLHelper.getDeviceId());
    }

    SECTION("isInitialized")
    {
        REQUIRE(!openCLHelper.isInitialized());
        REQUIRE_NOTHROW(openCLHelper.init());
        REQUIRE(openCLHelper.isInitialized());
    }

    SECTION("setKernelFile throwing")
    {
        REQUIRE_THROWS(openCLHelper.setKernelFile("kjhnfkasdf"));
    }
}
