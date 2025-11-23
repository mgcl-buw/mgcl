#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <fstream>
#include <iostream>

#include "../src/mgcl/cuboid.hpp"
#include "../src/mgcl/opencl_helper.hpp"
#include "../src/mgcl/problem.hpp"
#include "cli_args.hpp"
#include "device_type_generator.hpp"
#include "test_utility.hpp"

TEST_CASE("OpenCLHelper")
{
    auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));

    auto v = std::make_shared<mgcl::Cuboid>(4, 4, 4);
    auto f = std::make_shared<mgcl::Cuboid>(4, 4, 4);
    auto p = std::make_shared<mgcl::Problem>(4, 4, 4, f, v);
    p->setUseOpencl(true);
    p->setDeviceType(deviceType);
    auto& openCLHelper = p->getOpenCLHelper();

    SECTION("init default conf")
    {
        REQUIRE(openCLHelper.getDeviceType() == deviceType);

        REQUIRE_FALSE(openCLHelper.isInitialized());
        REQUIRE_NOTHROW(openCLHelper.init());
        REQUIRE(openCLHelper.isInitialized());

        REQUIRE(openCLHelper.getCommands() != nullptr);
        REQUIRE(openCLHelper.getContext() != nullptr);
        REQUIRE(openCLHelper.getDeviceId() != nullptr);
        // REQUIRE(openCLHelper.getDeviceType() == CL_DEVICE_TYPE_GPU);
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
        openCLHelper.setDeviceType(deviceType);

        REQUIRE_FALSE(openCLHelper.isInitialized());
        REQUIRE_NOTHROW(openCLHelper.init());
        REQUIRE(openCLHelper.isInitialized());

        auto v2 = std::make_shared<mgcl::Cuboid>(3, 3, 3);
        auto f2 = std::make_shared<mgcl::Cuboid>(3, 3, 3);
        mgcl::Problem p2(3, 3, 3, f2, v2);
        p2.setDeviceType(deviceType);
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

    SECTION("usingBinaryFile")
    {
        // Create random binary file name
        std::string binaryFileName = "binary_" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count());

        // Remove the binary file if it already exists
        std::remove(binaryFileName.c_str());

        REQUIRE_NOTHROW(openCLHelper.setBinaryFile(binaryFileName));
        REQUIRE(openCLHelper.getBinaryFile() == binaryFileName);

        // On first init, the program should be compiled from source, as the binary does not exist
        openCLHelper.init();
        p->solve();

        // Check that the binary file exists
        std::ifstream f(binaryFileName.c_str());
        REQUIRE(f.good());

        // On second init, the program should be built from binary. We can't actually check it, but it should not
        // fail...
        openCLHelper.init();
        p->solve();

        // Remove the binary file to clean up
        std::remove(binaryFileName.c_str());
    }

    SECTION("rebuildProgram")
    {
        SECTION("success program object initialized")
        {
            openCLHelper.init();
            openCLHelper.setPreprocessorConstant("BLOCKSIZE", std::to_string(2));
            REQUIRE_NOTHROW(openCLHelper.rebuildProgram());
        }

        SECTION("success force rebuild program object")
        {
            openCLHelper.init();
            openCLHelper.setPreprocessorConstant("BLOCKSIZE", std::to_string(2));
            REQUIRE_NOTHROW(openCLHelper.rebuildProgram(true));
        }

        SECTION("failure program object not initialized")
        {
            openCLHelper.setPreprocessorConstant("BLOCKSIZE", std::to_string(2));
            REQUIRE_THROWS(openCLHelper.rebuildProgram());
        }

        SECTION("failure program object has kernels")
        {
            openCLHelper.init();

            int err;
            const char* kernelName = "update_ghosts_periodic";
            cl_kernel kernel = clCreateKernel(openCLHelper.getProgram(), kernelName, &err);
            mgcl::mgclCheckError(err, "clCreateKernel");

            openCLHelper.setPreprocessorConstant("BLOCKSIZE", std::to_string(2));
            REQUIRE_THROWS(openCLHelper.rebuildProgram());
        }

        SECTION("success program object has released kernels")
        {
            openCLHelper.init();

            int err;
            const char* kernelName = "update_ghosts_periodic";
            cl_kernel kernel = clCreateKernel(openCLHelper.getProgram(), kernelName, &err);
            mgcl::mgclCheckError(err, "clCreateKernel");
            mgcl::mgclCheckError(clReleaseKernel(kernel), "clReleaseKernel");

            openCLHelper.setPreprocessorConstant("BLOCKSIZE", std::to_string(2));
            REQUIRE_NOTHROW(openCLHelper.rebuildProgram());
        }
    }

    SECTION("queryComputeUnitCount")
    {
        openCLHelper.init();

        REQUIRE(openCLHelper.queryComputeUnitCount(openCLHelper.getDeviceId()) > 0);

        auto s = openCLHelper.queryDeviceName(openCLHelper.getDeviceId());
        // std::cout << "s: " << s << std::endl;
        if (s == "Quadro T2000 with Max-Q Design")
            REQUIRE(openCLHelper.queryComputeUnitCount(openCLHelper.getDeviceId()) == 16);
    }

    SECTION("queryDeviceName")
    {
        openCLHelper.init();
        auto s = openCLHelper.queryDeviceName(openCLHelper.getDeviceId());
        REQUIRE(s.length() > 0);
    }

    SECTION("queryMaxWgSize")
    {
        openCLHelper.init();

        REQUIRE(openCLHelper.queryMaxWgSize(openCLHelper.getDeviceId()) > 0);

        auto s = openCLHelper.queryDeviceName(openCLHelper.getDeviceId());
        // std::cout << "s: " << s << std::endl;
        if (s == "Quadro T2000 with Max-Q Design")
            REQUIRE(openCLHelper.queryMaxWgSize(openCLHelper.getDeviceId()) == 1024);
    }
}
