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
    auto s = std::make_unique<mgcl::VaryingStencilGpu>(m, n, o, width, gh, t.getContext());

    REQUIRE(s->getBuf());

    // check size of buffer
    size_t bufsize;
    int sizeNeeded = sizeof(double) * (m + 2 * gh) * (n + 2 * gh) * (o + 2 * gh) * width * width * width;
    int err = clGetMemObjectInfo(s->getBuf(), CL_MEM_SIZE, sizeof(size_t), &bufsize, nullptr);
    mgcl::mgclCheckError(err, "Querying buffer size");
    REQUIRE(bufsize == sizeNeeded);

    // check reference count, should be 1
    cl_uint refCount;

    err = clGetMemObjectInfo(s->getBuf(), CL_MEM_REFERENCE_COUNT, sizeof(cl_uint), &refCount, nullptr);
    mgcl::mgclCheckError(err, "clGetMemObjectInfo(s, CL_MEM_REFERENCE_COUNT)");
    REQUIRE(refCount == 1);

    // delete s
    s.reset();
}
