
#include <CL/cl.h>
#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "../src/mgcl/buffer_gpu.hpp"
#include "test_utility.hpp"

TEST_CASE("BufferGpu::ctor_size")
{
    mgcl_test::TestUtility tu(CL_DEVICE_TYPE_GPU);

    SECTION("success")
    {
        mgcl::BufferGpu b(tu.getContext(), CL_MEM_READ_WRITE, 10);
        REQUIRE(b.getBuf());
        REQUIRE(b.getSize() == 10);
    }

    SECTION("throwing")
    {
        // Check that flags contains one and only one of CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY or CL_MEM_READ_ONLY
        REQUIRE_THROWS(mgcl::BufferGpu(tu.getContext(), CL_MEM_READ_WRITE | CL_MEM_WRITE_ONLY, 10));
        REQUIRE_THROWS(mgcl::BufferGpu(tu.getContext(), CL_MEM_READ_WRITE | CL_MEM_READ_ONLY, 10));
        REQUIRE_THROWS(mgcl::BufferGpu(tu.getContext(), CL_MEM_READ_ONLY | CL_MEM_WRITE_ONLY, 10));

        // Check that flags does not contain one of CL_MEM_COPY_HOST_PTR, CL_MEM_USE_HOST_PTR or CL_MEM_ALLOC_HOST_PTR
        REQUIRE_THROWS(mgcl::BufferGpu(tu.getContext(), CL_MEM_COPY_HOST_PTR, 10));
        REQUIRE_THROWS(mgcl::BufferGpu(tu.getContext(), CL_MEM_USE_HOST_PTR, 10));
        REQUIRE_THROWS(mgcl::BufferGpu(tu.getContext(), CL_MEM_ALLOC_HOST_PTR, 10));
    }
}

TEST_CASE("BufferGpu::ctor_h_data")
{
    mgcl_test::TestUtility tu(CL_DEVICE_TYPE_GPU);
    std::vector<double> h_data(10, 1);

    SECTION("success")
    {
        mgcl::BufferGpu b(tu.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, h_data);
        REQUIRE(b.getBuf());
        REQUIRE(b.getSize() == 10);

        auto ret = b.read(tu.getCommands(), nullptr, true);
        REQUIRE(ret->size() == h_data.size());
        for (size_t i = 0; i < h_data.size(); i++)
        {
            REQUIRE((*ret)[i] == h_data[i]);
        }
    }

    SECTION("throwing")
    {
        // Check that flags contains one and only one of CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY or CL_MEM_READ_ONLY
        REQUIRE_THROWS(mgcl::BufferGpu(tu.getContext(), CL_MEM_READ_WRITE | CL_MEM_WRITE_ONLY, h_data));
        REQUIRE_THROWS(mgcl::BufferGpu(tu.getContext(), CL_MEM_READ_WRITE | CL_MEM_READ_ONLY, h_data));
        REQUIRE_THROWS(mgcl::BufferGpu(tu.getContext(), CL_MEM_READ_ONLY | CL_MEM_WRITE_ONLY, h_data));

        // Check that flags contains one and only one of CL_MEM_COPY_HOST_PTR, CL_MEM_USE_HOST_PTR or CL_MEM_ALLOC_HOST_PTR
        REQUIRE_THROWS(mgcl::BufferGpu(tu.getContext(), CL_MEM_USE_HOST_PTR | CL_MEM_COPY_HOST_PTR, h_data));
        REQUIRE_THROWS(mgcl::BufferGpu(tu.getContext(), CL_MEM_ALLOC_HOST_PTR | CL_MEM_USE_HOST_PTR, h_data));
        REQUIRE_THROWS(mgcl::BufferGpu(tu.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR, h_data));
    }
}

TEST_CASE("BufferGpu::write_BufferGpu::read")
{
    mgcl_test::TestUtility tu(CL_DEVICE_TYPE_GPU);

    mgcl::BufferGpu b(tu.getContext(), CL_MEM_READ_WRITE, 10);
    b.write(tu.getCommands(), {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, true);

    std::vector<double> h_target(10);

    auto ret = b.read(tu.getCommands(), nullptr, true);
    b.read(tu.getCommands(), h_target.data(), true);

    for (size_t i = 0; i < 10; i++)
    {
        REQUIRE((*ret)[i] == i + 1);
        REQUIRE(h_target[i] == i + 1);
    }
}

TEST_CASE("BufferGpu::fill")
{
    mgcl_test::TestUtility tu(CL_DEVICE_TYPE_GPU);

    mgcl::BufferGpu b(tu.getContext(), CL_MEM_READ_WRITE, 10);

    b.fill(tu.getProgram(), tu.getCommands(), 42, true, nullptr, nullptr);
    auto ret = b.read(tu.getCommands(), nullptr, true);

    for (size_t i = 0; i < 10; i++)
    {
        REQUIRE((*ret)[i] == 42);
    }
}