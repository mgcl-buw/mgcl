#include <catch2/catch_test_macros.hpp>

#include "../cuboid.hpp"
#include "test_utility.hpp"

TEST_CASE("TestUtility setup")
{
    mgcl_test::TestUtility tu;
    REQUIRE(tu.getProblem().getOpenCLHelper().isInitialized());
}

TEST_CASE("TestUtility setup reusing Problem")
{
    auto p = std::make_shared<mgcl::Problem>(4, 4, 4);
    mgcl_test::TestUtility tu(p);
    REQUIRE(tu.getProblem().getOpenCLHelper().isInitialized());
}

TEST_CASE("TestUtility createOpenCLBuffer + readOpenCLBuffer")
{
    mgcl_test::TestUtility tu;
    REQUIRE(tu.getProblem().getOpenCLHelper().isInitialized());

    mgcl::Cuboid c(4, 4, 4, 3.0);
    auto buf = tu.createOpenCLBuffer(c);
    REQUIRE(buf);

    auto &c2 = *tu.readOpenCLBuffer(buf, 4, 4, 4);
    for (int i = 0; i < c2.field1d().size(); i++)
        REQUIRE(c2.field1d()[i] == c.field1d()[i]);
}

TEST_CASE("TestUtility readOpenCLBuffer nullptr")
{
    mgcl_test::TestUtility tu;
    REQUIRE(tu.getProblem().getOpenCLHelper().isInitialized());
    REQUIRE_THROWS_AS(tu.readOpenCLBuffer(nullptr, 4, 4, 4), std::invalid_argument);
}

TEST_CASE("TestUtility deviceAvailable")
{
    mgcl_test::TestUtility tu;
    int err;
    cl_char device_name[1024] = {0};
    cl_device_type device_type;

    err = clGetDeviceInfo(tu.getDeviceId(), CL_DEVICE_NAME, sizeof(device_name), &device_name, NULL);
    mgcl::mgclCheckError(err, "clGetDeviceInfo(CL_DEVICE_NAME)");

    err = clGetDeviceInfo(tu.getDeviceId(), CL_DEVICE_TYPE, sizeof(device_type), &device_type, NULL);
    mgcl::mgclCheckError(err, "clGetDeviceInfo(CL_DEVICE_TYPE)");

    REQUIRE(mgcl_test::TestUtility::deviceAvailable(std::string((char *)device_name), device_type));
    REQUIRE_FALSE(mgcl_test::TestUtility::deviceAvailable("akljshnfklfha", device_type));
}

TEST_CASE("TestUtility setup deviceName")
{
    if (mgcl_test::TestUtility::deviceAvailable("Quadro", CL_DEVICE_TYPE_ALL))
    {
        REQUIRE_NOTHROW(mgcl_test::TestUtility("Quadro"));
    }
    REQUIRE_THROWS(mgcl_test::TestUtility("aksnhjisagnfsif"));
}

TEST_CASE("TestUtility setup deviceType")
{
    if (mgcl_test::TestUtility::deviceAvailable("", CL_DEVICE_TYPE_GPU))
        REQUIRE_NOTHROW(mgcl_test::TestUtility(CL_DEVICE_TYPE_GPU));
    else
        REQUIRE_THROWS(mgcl_test::TestUtility(CL_DEVICE_TYPE_GPU));

    if (mgcl_test::TestUtility::deviceAvailable("", CL_DEVICE_TYPE_CPU))
        REQUIRE_NOTHROW(mgcl_test::TestUtility(CL_DEVICE_TYPE_CPU));
    else
        REQUIRE_THROWS(mgcl_test::TestUtility(CL_DEVICE_TYPE_CPU));
}
