#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <memory>

#include "../cuboid.hpp"
#include "../multigrid_engine.hpp"
#include "../opencl_helper.hpp"
#include "../stencil.hpp"

#include "matrix2d.hpp"
#include "test_utility.hpp"

TEST_CASE("VaryingStencilGpu ctor+dtor")
{
    auto deviceType = GENERATE(CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU);

    if (!mgcl_test::TestUtility::deviceAvailable("", deviceType))
    {
        std::string typeName = deviceType == CL_DEVICE_TYPE_GPU ? "CL_DEVICE_TYPE_GPU" : "CL_DEVICE_TYPE_CPU";
        std::cout << "Skipping non-available device type '" << typeName << "'" << std::endl;
        return;
    }

    mgcl_test::TestUtility t(deviceType);

    int m = 4;
    int n = 4;
    int o = 4;
    int width = 3;
    int gh = 2;
    auto s = std::make_unique<mgcl::VaryingStencilGpu>(m, n, o, width, gh, t.getContext(), t.getCommands());
    t.finish();

    REQUIRE(s->getBuf());

    // check size of buffer
    size_t bufsize;
    int sizeNeeded = (m + 2 * gh) * (n + 2 * gh) * (o + 2 * gh) * width * width * width;
    int err = clGetMemObjectInfo(s->getBuf(), CL_MEM_SIZE, sizeof(size_t), &bufsize, nullptr);
    mgcl::mgclCheckError(err, "Querying buffer size");
    REQUIRE(bufsize == sizeof(double) * sizeNeeded);

    // check reference count, should be 1
    cl_uint refCount;

    err = clGetMemObjectInfo(s->getBuf(), CL_MEM_REFERENCE_COUNT, sizeof(cl_uint), &refCount, nullptr);
    mgcl::mgclCheckError(err, "clGetMemObjectInfo(s, CL_MEM_REFERENCE_COUNT)");
    REQUIRE(refCount == 1);

    // check values are 0
    double tmp[sizeNeeded];
    err = clEnqueueReadBuffer(t.getCommands(), s->getBuf(), CL_TRUE, 0,
                              sizeof(double) * sizeNeeded, tmp, 0, NULL, NULL);
    mgcl::mgclCheckError(err, "clEnqueueReadBuffer");

    for (int i = 0; i < sizeNeeded; i++)
        REQUIRE(tmp[i] == 0);

    // delete s
    s.reset();
}

TEST_CASE("VaryingStencilGpu::fill")
{
    auto deviceType = GENERATE(CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU);

    if (!mgcl_test::TestUtility::deviceAvailable("", deviceType))
    {
        std::string typeName = deviceType == CL_DEVICE_TYPE_GPU ? "CL_DEVICE_TYPE_GPU" : "CL_DEVICE_TYPE_CPU";
        std::cout << "Skipping non-available device type '" << typeName << "'" << std::endl;
        return;
    }

    mgcl_test::TestUtility t(deviceType);

    int m = 4;
    int n = 4;
    int o = 4;
    int gh = 2;

    SECTION("VaryingStencil3x3x3")
    {
        int width = 3;
        auto s = std::make_unique<mgcl::VaryingStencilGpu>(m, n, o, width, gh, t.getContext(), t.getCommands());
        t.finish();

        int sizeNeeded = (m + 2 * gh) * (n + 2 * gh) * (o + 2 * gh) * width * width * width;

        // create VaryingStencil, fill with random values and copy to gpu buffer
        mgcl::VaryingStencil3x3x3 s3(m, n, o, gh, gh, gh);
        s3.fillRandomInt();
        s->fill(s3, t.getCommands());

        // check results
        double tmp[sizeNeeded];
        int err = clEnqueueReadBuffer(t.getCommands(), s->getBuf(), CL_TRUE, 0,
                                      sizeof(double) * sizeNeeded, tmp, 0, NULL, NULL);
        mgcl::mgclCheckError(err, "clEnqueueReadBuffer");

        for (int i = 0; i < sizeNeeded; i++)
            REQUIRE(tmp[i] == s3[0][0][0][0][0][i]);
    }

    SECTION("VaryingStencil5x5x5")
    {
        int width = 5;
        auto s = std::make_unique<mgcl::VaryingStencilGpu>(m, n, o, width, gh, t.getContext(), t.getCommands());
        t.finish();

        int sizeNeeded = (m + 2 * gh) * (n + 2 * gh) * (o + 2 * gh) * width * width * width;

        // create VaryingStencil, fill with random values and copy to gpu buffer
        mgcl::VaryingStencil5x5x5 s3(m, n, o, gh, gh, gh);
        s3.fillRandomInt();
        s->fill(s3, t.getCommands());

        // check results
        double tmp[sizeNeeded];
        int err = clEnqueueReadBuffer(t.getCommands(), s->getBuf(), CL_TRUE, 0,
                                      sizeof(double) * sizeNeeded, tmp, 0, NULL, NULL);
        mgcl::mgclCheckError(err, "clEnqueueReadBuffer");

        for (int i = 0; i < sizeNeeded; i++)
            REQUIRE(tmp[i] == s3[0][0][0][0][0][i]);
    }

    SECTION("throwing")
    {
        int width = 3;
        auto s = std::make_unique<mgcl::VaryingStencilGpu>(m, n, o, width, gh, t.getContext(), t.getCommands());
        t.finish();

        // widths do not match
        mgcl::VaryingStencil5x5x5 s3(m, n, o, gh, gh, gh);
        REQUIRE_THROWS(s->fill(s3, t.getCommands()));

        // grid sizes do not match
        mgcl::VaryingStencil5x5x5 s35(m * 2, n * 3, o * 4, gh, gh, gh);
        REQUIRE_THROWS(s->fill(s35, t.getCommands()));
    }
}

TEST_CASE("VaryingStencilGpu::read")
{
    auto deviceType = GENERATE(CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU);

    if (!mgcl_test::TestUtility::deviceAvailable("", deviceType))
    {
        std::string typeName = deviceType == CL_DEVICE_TYPE_GPU ? "CL_DEVICE_TYPE_GPU" : "CL_DEVICE_TYPE_CPU";
        std::cout << "Skipping non-available device type '" << typeName << "'" << std::endl;
        return;
    }

    mgcl_test::TestUtility t(deviceType);

    int m = 4;
    int n = 4;
    int o = 4;
    int gh = 2;

    SECTION("VaryingStencil3x3x3")
    {
        int width = 3;
        auto s = std::make_unique<mgcl::VaryingStencilGpu>(m, n, o, width, gh, t.getContext(), t.getCommands());
        t.finish();

        int sizeNeeded = (m + 2 * gh) * (n + 2 * gh) * (o + 2 * gh) * width * width * width;

        // create VaryingStencil, fill with random values and copy to gpu buffer
        mgcl::VaryingStencil3x3x3 s3(m, n, o, gh, gh, gh);
        s3.fillRandomInt();
        s->fill(s3, t.getCommands());

        // read buffer
        auto ret = s->read<3>(t.getCommands());
        t.finish();

        // check results
        for (int i = 0; i < ret.field1d().size(); i++)
            REQUIRE(ret.field1d()[i] == s3.field1d()[i]);
    }

    SECTION("VaryingStencil5x5x5")
    {
        int width = 5;
        auto s = std::make_unique<mgcl::VaryingStencilGpu>(m, n, o, width, gh, t.getContext(), t.getCommands());
        t.finish();

        int sizeNeeded = (m + 2 * gh) * (n + 2 * gh) * (o + 2 * gh) * width * width * width;

        // create VaryingStencil, fill with random values and copy to gpu buffer
        mgcl::VaryingStencil5x5x5 s3(m, n, o, gh, gh, gh);
        s3.fillRandomInt();
        s->fill(s3, t.getCommands());

        // read buffer
        auto ret = s->read<5>(t.getCommands());
        t.finish();

        // check results
        for (int i = 0; i < ret.field1d().size(); i++)
            REQUIRE(ret.field1d()[i] == s3.field1d()[i]);
    }

    SECTION("throwing")
    {
        int width = 3;
        auto s = std::make_unique<mgcl::VaryingStencilGpu>(m, n, o, width, gh, t.getContext(), t.getCommands());
        t.finish();

        REQUIRE_THROWS(s->read<5>(t.getCommands()));
    }
}
TEST_CASE("FixedStencilGpu ctor+dtor")
{
    auto deviceType = GENERATE(CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU);

    if (!mgcl_test::TestUtility::deviceAvailable("", deviceType))
    {
        std::string typeName = deviceType == CL_DEVICE_TYPE_GPU ? "CL_DEVICE_TYPE_GPU" : "CL_DEVICE_TYPE_CPU";
        std::cout << "Skipping non-available device type '" << typeName << "'" << std::endl;
        return;
    }

    mgcl_test::TestUtility t(deviceType);

    int width = 3;
    int gh = 2;
    auto s = std::make_unique<mgcl::FixedStencilGpu>(width, t.getContext(), t.getCommands());
    t.finish();

    REQUIRE(s->getBuf());

    // check size of buffer
    size_t bufsize;
    int sizeNeeded = width * width * width;
    int err = clGetMemObjectInfo(s->getBuf(), CL_MEM_SIZE, sizeof(size_t), &bufsize, nullptr);
    mgcl::mgclCheckError(err, "Querying buffer size");
    REQUIRE(bufsize == sizeof(double) * sizeNeeded);

    // check reference count, should be 1
    cl_uint refCount;

    err = clGetMemObjectInfo(s->getBuf(), CL_MEM_REFERENCE_COUNT, sizeof(cl_uint), &refCount, nullptr);
    mgcl::mgclCheckError(err, "clGetMemObjectInfo(s, CL_MEM_REFERENCE_COUNT)");
    REQUIRE(refCount == 1);

    // check values are 0
    double tmp[sizeNeeded];
    err = clEnqueueReadBuffer(t.getCommands(), s->getBuf(), CL_TRUE, 0,
                              sizeof(double) * sizeNeeded, tmp, 0, NULL, NULL);
    mgcl::mgclCheckError(err, "clEnqueueReadBuffer");

    for (int i = 0; i < sizeNeeded; i++)
        REQUIRE(tmp[i] == 0);

    // delete s
    s.reset();
}

TEST_CASE("FixedStencilGpu::fill")
{
    auto deviceType = GENERATE(CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU);

    if (!mgcl_test::TestUtility::deviceAvailable("", deviceType))
    {
        std::string typeName = deviceType == CL_DEVICE_TYPE_GPU ? "CL_DEVICE_TYPE_GPU" : "CL_DEVICE_TYPE_CPU";
        std::cout << "Skipping non-available device type '" << typeName << "'" << std::endl;
        return;
    }

    mgcl_test::TestUtility t(deviceType);

    SECTION("FixedStencil3x3x3")
    {
        int width = 3;
        auto s = std::make_unique<mgcl::FixedStencilGpu>(width, t.getContext(), t.getCommands());
        t.finish();

        int sizeNeeded = width * width * width;

        // create VaryingStencil, fill with random values and copy to gpu buffer
        mgcl::FixedStencil3x3x3 s3;
        s3.fillRandom();
        s->fill(s3, t.getCommands());

        // check results
        double tmp[sizeNeeded];
        int err = clEnqueueReadBuffer(t.getCommands(), s->getBuf(), CL_TRUE, 0,
                                      sizeof(double) * sizeNeeded, tmp, 0, NULL, NULL);
        mgcl::mgclCheckError(err, "clEnqueueReadBuffer");

        for (int i = 0; i < sizeNeeded; i++)
            REQUIRE(tmp[i] == s3[0][0][i]);
    }

    SECTION("FixedStencil5x5x5")
    {
        int width = 5;
        auto s = std::make_unique<mgcl::FixedStencilGpu>(width, t.getContext(), t.getCommands());
        t.finish();

        int sizeNeeded = width * width * width;

        // create VaryingStencil, fill with random values and copy to gpu buffer
        mgcl::FixedStencil<5> s3;
        s3.fillRandom();
        s->fill(s3, t.getCommands());

        // check results
        double tmp[sizeNeeded];
        int err = clEnqueueReadBuffer(t.getCommands(), s->getBuf(), CL_TRUE, 0,
                                      sizeof(double) * sizeNeeded, tmp, 0, NULL, NULL);
        mgcl::mgclCheckError(err, "clEnqueueReadBuffer");

        for (int i = 0; i < sizeNeeded; i++)
            REQUIRE(tmp[i] == s3[0][0][i]);
    }

    SECTION("throwing")
    {
        int width = 3;
        auto s = std::make_unique<mgcl::FixedStencilGpu>(width, t.getContext(), t.getCommands());
        t.finish();

        // widths do not match
        mgcl::FixedStencil<5> s3;
        REQUIRE_THROWS(s->fill(s3, t.getCommands()));
    }
}
