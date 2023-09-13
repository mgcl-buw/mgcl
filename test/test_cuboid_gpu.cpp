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
    mgcl::CuboidGpu c(tu.getContext(), CL_MEM_READ_WRITE, 1, 2, 3, 2, 3, 4, &ch);

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
        mgcl_test::TestUtility tu;
        REQUIRE_THROWS(mgcl::CuboidGpu(tu.getContext(), CL_MEM_COPY_HOST_PTR, 1, 2, 3, 2, 3, 4, nullptr));
        REQUIRE_THROWS(mgcl::CuboidGpu(tu.getContext(), CL_MEM_USE_HOST_PTR, 1, 2, 3, 2, 3, 4, nullptr));
        REQUIRE_THROWS(mgcl::CuboidGpu(tu.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 1, 2, 3, 2, 3, 4, nullptr));
    }
}

TEST_CASE("CuboidGpu::read into new")
{
    mgcl::Cuboid ch(1, 2, 3, 2, 3, 4);
    ch.fillRandom();

    mgcl_test::TestUtility tu;
    mgcl::CuboidGpu c(tu.getContext(), CL_MEM_READ_WRITE, 1, 2, 3, 2, 3, 4, &ch);
    auto ret = c.read(tu.getCommands(), nullptr, true);

    REQUIRE(ret->isEqualAllCells(ch));
}

TEST_CASE("CuboidGpu::read into existing")
{
    mgcl::Cuboid ch(1, 2, 3, 2, 3, 4);
    ch.fillRandom();

    mgcl_test::TestUtility tu;
    mgcl::CuboidGpu c(tu.getContext(), CL_MEM_READ_WRITE, 1, 2, 3, 2, 3, 4, &ch);

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