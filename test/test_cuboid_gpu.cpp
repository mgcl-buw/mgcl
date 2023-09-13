#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <cstdio>
#include <fstream>

#include "../src/cuboid.hpp"
#include "../src/cuboid_gpu.hpp"
#include "test_utility.hpp"

// Check if CuboidGpu gets initialized correctly with host_ptr being null.
TEST_CASE("CuboidGpu ctor host_ptr null")
{
    mgcl_test::TestUtility tu;
    mgcl::CuboidGpu c(tu.getContext(), CL_MEM_READ_WRITE, 1, 2, 3, 2, 3, 4);

    REQUIRE(c.getM() == 1);
    REQUIRE(c.getN() == 2);
    REQUIRE(c.getO() == 3);
    REQUIRE(c.getGhostsM() == 2);
    REQUIRE(c.getGhostsN() == 3);
    REQUIRE(c.getGhostsO() == 4);
    REQUIRE(c.getMgh() == 1 + 2 * 2);
    REQUIRE(c.getNgh() == 2 + 3 * 2);
    REQUIRE(c.getOgh() == 3 + 4 * 2);
    REQUIRE(c.getSize() == c.getMgh() * c.getNgh() * c.getOgh());
}

// Check if CuboidGpu gets initialized correctly with host_ptr not being null.
TEST_CASE("CuboidGpu ctor host_ptr not null")
{
    mgcl::Cuboid ch(1, 2, 3, 2, 3, 4);
    ch.fillRandom();

    mgcl_test::TestUtility tu;
    mgcl::CuboidGpu c(tu.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 1, 2, 3, 2, 3, 4, &ch);

    REQUIRE(c.getM() == 1);
    REQUIRE(c.getN() == 2);
    REQUIRE(c.getO() == 3);
    REQUIRE(c.getGhostsM() == 2);
    REQUIRE(c.getGhostsN() == 3);
    REQUIRE(c.getGhostsO() == 4);
    REQUIRE(c.getMgh() == 1 + 2 * 2);
    REQUIRE(c.getNgh() == 2 + 3 * 2);
    REQUIRE(c.getOgh() == 3 + 4 * 2);
    REQUIRE(c.getSize() == c.getMgh() * c.getNgh() * c.getOgh());

    auto res = c.read(tu.getCommands(), nullptr, true);
    REQUIRE(res->isEqualAllCells(ch));
}

// Check if CuboidGpu gets initialized correctly retaining an existing buffer.
TEST_CASE("CuboidGpu ctor retaining buffer")
{
    int refCount;
    int err;

    mgcl_test::TestUtility tu;
    mgcl::Cuboid ch(1, 2, 3, 2, 3, 4);
    cl_mem buf = tu.createOpenCLBuffer(ch);

    err = clGetMemObjectInfo(buf, CL_MEM_REFERENCE_COUNT, sizeof(cl_uint), &refCount, nullptr);
    mgcl::mgclCheckError(err, "clGetMemObjectInfo(d_v, CL_MEM_REFERENCE_COUNT)");
    REQUIRE(refCount == 1);

    mgcl::CuboidGpu* c = new mgcl::CuboidGpu(tu.getContext(), 1, 2, 3, 2, 3, 4, buf);

    err = clGetMemObjectInfo(buf, CL_MEM_REFERENCE_COUNT, sizeof(cl_uint), &refCount, nullptr);
    mgcl::mgclCheckError(err, "clGetMemObjectInfo(d_v, CL_MEM_REFERENCE_COUNT)");
    REQUIRE(refCount == 2);

    delete c;

    err = clGetMemObjectInfo(buf, CL_MEM_REFERENCE_COUNT, sizeof(cl_uint), &refCount, nullptr);
    mgcl::mgclCheckError(err, "clGetMemObjectInfo(d_v, CL_MEM_REFERENCE_COUNT)");
    REQUIRE(refCount == 1);
}

// Check if CuboidGpu's constructor throws an exception if the dimensions do not match.
TEST_CASE("CuboidGpu ctor throwing")
{
    SECTION("Invalid dimensions")
    {
        mgcl_test::TestUtility tu;
        REQUIRE_THROWS(mgcl::CuboidGpu(tu.getContext(), CL_MEM_READ_WRITE, 0, 2, 3, 2, 3, 4, nullptr));
        REQUIRE_THROWS(mgcl::CuboidGpu(tu.getContext(), CL_MEM_READ_WRITE, 1, 0, 3, 2, 3, 4, nullptr));
        REQUIRE_THROWS(mgcl::CuboidGpu(tu.getContext(), CL_MEM_READ_WRITE, 1, 2, 0, 2, 3, 4, nullptr));
        REQUIRE_THROWS(mgcl::CuboidGpu(tu.getContext(), CL_MEM_READ_WRITE, 1, 2, 3, -1, 3, 4, nullptr));
        REQUIRE_THROWS(mgcl::CuboidGpu(tu.getContext(), CL_MEM_READ_WRITE, 1, 2, 3, 2, -1, 4, nullptr));
        REQUIRE_THROWS(mgcl::CuboidGpu(tu.getContext(), CL_MEM_READ_WRITE, 1, 2, 3, 2, 3, -1, nullptr));
    }

    SECTION("Dimensions do not match")
    {
        mgcl::Cuboid ch(1, 2, 3, 2, 3, 4);

        mgcl_test::TestUtility tu;
        REQUIRE_THROWS(mgcl::CuboidGpu(tu.getContext(), CL_MEM_READ_WRITE, 9, 2, 3, 2, 3, 4, &ch));
        REQUIRE_THROWS(mgcl::CuboidGpu(tu.getContext(), CL_MEM_READ_WRITE, 1, 9, 3, 2, 3, 4, &ch));
        REQUIRE_THROWS(mgcl::CuboidGpu(tu.getContext(), CL_MEM_READ_WRITE, 1, 2, 9, 2, 3, 4, &ch));
        REQUIRE_THROWS(mgcl::CuboidGpu(tu.getContext(), CL_MEM_READ_WRITE, 1, 2, 3, 9, 3, 4, &ch));
        REQUIRE_THROWS(mgcl::CuboidGpu(tu.getContext(), CL_MEM_READ_WRITE, 1, 2, 3, 2, 9, 4, &ch));
        REQUIRE_THROWS(mgcl::CuboidGpu(tu.getContext(), CL_MEM_READ_WRITE, 1, 2, 3, 2, 3, 9, &ch));
    }

    SECTION("invalid flags")
    {
        mgcl::Cuboid ch(1, 2, 3, 2, 3, 4);
        mgcl_test::TestUtility tu;

        // flags must contain one of CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY or CL_MEM_READ_ONLY.
        REQUIRE_THROWS(mgcl::CuboidGpu(tu.getContext(), CL_MEM_COPY_HOST_PTR, 1, 2, 3, 2, 3, 4, nullptr));
        REQUIRE_THROWS(mgcl::CuboidGpu(tu.getContext(), CL_MEM_USE_HOST_PTR, 1, 2, 3, 2, 3, 4, nullptr));
        REQUIRE_THROWS(mgcl::CuboidGpu(tu.getContext(), CL_MEM_ALLOC_HOST_PTR, 1, 2, 3, 2, 3, 4, nullptr));

        // flags must contain one of CL_MEM_COPY_HOST_PTR, CL_MEM_USE_HOST_PTR or CL_MEM_ALLOC_HOST_PTR
        // if host_ptr is given.
        REQUIRE_THROWS(mgcl::CuboidGpu(tu.getContext(), CL_MEM_WRITE_ONLY, 1, 2, 3, 2, 3, 4, &ch));
        REQUIRE_THROWS(mgcl::CuboidGpu(tu.getContext(), CL_MEM_READ_ONLY, 1, 2, 3, 2, 3, 4, &ch));
        REQUIRE_THROWS(mgcl::CuboidGpu(tu.getContext(), CL_MEM_READ_WRITE, 1, 2, 3, 2, 3, 4, &ch));

        // flags must not not contain CL_MEM_COPY_HOST_PTR, CL_MEM_USE_HOST_PTR or CL_MEM_ALLOC_HOST_PTR
        // if host_ptr is not given.
        REQUIRE_THROWS(mgcl::CuboidGpu(tu.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, 1, 2, 3, 2, 3, 4, nullptr));
        REQUIRE_THROWS(mgcl::CuboidGpu(tu.getContext(), CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, 1, 2, 3, 2, 3, 4, nullptr));
        REQUIRE_THROWS(mgcl::CuboidGpu(tu.getContext(), CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, 1, 2, 3, 2, 3, 4, nullptr));

        // flags must not contain multiple of CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY or CL_MEM_WRITE_ONLY
        REQUIRE_THROWS(mgcl::CuboidGpu(tu.getContext(), CL_MEM_READ_WRITE | CL_MEM_WRITE_ONLY, 1, 2, 3, 2, 3, 4, nullptr));
        REQUIRE_THROWS(mgcl::CuboidGpu(tu.getContext(), CL_MEM_READ_WRITE | CL_MEM_READ_ONLY, 1, 2, 3, 2, 3, 4, nullptr));
        REQUIRE_THROWS(mgcl::CuboidGpu(tu.getContext(), CL_MEM_WRITE_ONLY | CL_MEM_READ_ONLY, 1, 2, 3, 2, 3, 4, nullptr));

        // flags must not contain multiple of CL_MEM_COPY_HOST_PTR, CL_MEM_USE_HOST_PTR or CL_MEM_ALLOC_HOST_PTR if
        // host_ptr is given.
        REQUIRE_THROWS(mgcl::CuboidGpu(tu.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_USE_HOST_PTR, 1, 2, 3, 2, 3, 4, &ch));
        REQUIRE_THROWS(mgcl::CuboidGpu(tu.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR, 1, 2, 3, 2, 3, 4, &ch));
        REQUIRE_THROWS(mgcl::CuboidGpu(tu.getContext(), CL_MEM_USE_HOST_PTR | CL_MEM_ALLOC_HOST_PTR, 1, 2, 3, 2, 3, 4, &ch));
    }
}

TEST_CASE("CuboidGpu::read into new")
{
    mgcl::Cuboid ch(1, 2, 3, 2, 3, 4);
    ch.fillRandom();

    mgcl_test::TestUtility tu;
    mgcl::CuboidGpu c(tu.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 1, 2, 3, 2, 3, 4, &ch);
    auto ret = c.read(tu.getCommands(), nullptr, true);

    REQUIRE(ret->isEqualAllCells(ch));
}

TEST_CASE("CuboidGpu::read into existing")
{
    mgcl::Cuboid ch(1, 2, 3, 2, 3, 4);
    ch.fillRandom();

    mgcl_test::TestUtility tu;
    mgcl::CuboidGpu c(tu.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 1, 2, 3, 2, 3, 4, &ch);

    mgcl::Cuboid ch_act(1, 2, 3, 2, 3, 4);
    c.read(tu.getCommands(), &ch_act, true);

    REQUIRE(ch_act.isEqualAllCells(ch));
}

TEST_CASE("CuboidGpu::write")
{
    mgcl::Cuboid ch(1, 2, 3, 2, 3, 4);
    ch.fillRandom();

    mgcl_test::TestUtility tu;
    mgcl::CuboidGpu c(tu.getContext(), CL_MEM_READ_WRITE, 1, 2, 3, 2, 3, 4);
    c.write(tu.getCommands(), ch, true);

    auto ret = c.read(tu.getCommands(), nullptr, true);

    REQUIRE(ret->isEqualAllCells(ch));
}

// Tests if CuboidGpu::copyTo works correctly.
TEST_CASE("CuboidGpu::copyTo")
{
    mgcl_test::TestUtility tu;

    SECTION("success")
    {
        mgcl::Cuboid ch(1, 2, 3, 2, 3, 4);
        ch.fillRandom();

        mgcl::CuboidGpu c(tu.getContext(), CL_MEM_READ_WRITE, 1, 2, 3, 2, 3, 4);
        c.write(tu.getCommands(), ch, true);

        mgcl::CuboidGpu c2(tu.getContext(), CL_MEM_READ_WRITE, 1, 2, 3, 2, 3, 4);
        c.copyTo(tu.getCommands(), c2);

        auto ch2_act = c2.read(tu.getCommands(), nullptr, true);

        REQUIRE(ch2_act->isEqualAllCells(ch));
    }

    SECTION("throwing")
    {
        mgcl::Cuboid ch(1, 2, 3, 2, 3, 4);
        ch.fillRandom();

        mgcl::CuboidGpu c(tu.getContext(), CL_MEM_READ_WRITE, 1, 2, 3, 2, 3, 4);
        c.write(tu.getCommands(), ch, true);

        mgcl::CuboidGpu c2(tu.getContext(), CL_MEM_READ_WRITE, 3, 2, 3, 2, 3, 4);
        REQUIRE_THROWS(c.copyTo(tu.getCommands(), c2));
    }
}

// Tests if CuboidGpu::swap works correctly.
TEST_CASE("CuboidGpu::swap")
{
    mgcl_test::TestUtility tu;

    SECTION("success")
    {
        mgcl::Cuboid ch1(1, 1, 1, 0, 0, 0);
        mgcl::Cuboid ch2(1, 1, 1, 0, 0, 0);
        ch1[0][0][0] = 1.0;
        ch2[0][0][0] = 2.0;

        mgcl::CuboidGpu c1(tu.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 1, 1, 1, 0, 0, 0, &ch1);
        mgcl::CuboidGpu c2(tu.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 1, 1, 1, 0, 0, 0, &ch2);

        auto c_act1 = c1.read(tu.getCommands(), nullptr, false);
        auto c_act2 = c2.read(tu.getCommands(), nullptr, true);

        REQUIRE(c_act1->isEqual(ch1));
        REQUIRE(c_act2->isEqual(ch2));

        mgcl::CuboidGpu::swap(c1, c2);

        c_act1 = c1.read(tu.getCommands(), nullptr, false);
        c_act2 = c2.read(tu.getCommands(), nullptr, true);

        REQUIRE(c_act1->isEqual(ch2));
        REQUIRE(c_act2->isEqual(ch1));
    }

    SECTION("throwing")
    {
        mgcl::CuboidGpu c1(tu.getContext(), CL_MEM_READ_WRITE, 3, 1, 1, 0, 0, 0, nullptr);
        mgcl::CuboidGpu c2(tu.getContext(), CL_MEM_READ_WRITE, 1, 1, 1, 0, 0, 0, nullptr);
        REQUIRE_THROWS(mgcl::CuboidGpu::swap(c1, c2));
    }
}