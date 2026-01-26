#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cmath>
#include <iostream>
#include <memory>

#include "../src/mgcl/cuboid.hpp"
#include "../src/mgcl/level.hpp"
#include "../src/mgcl/mpi_global_data.hpp"
#include "../src/mgcl/problem.hpp"
#include "cli_args.hpp"
#include "device_type_generator.hpp"
#include "test_utility.hpp"

/**
 * @brief Tests if config parameters of Problem are working correctly, including setting things up.
 *
 */
TEST_CASE("Level constructor")
{
    SECTION("not using OpenCL")
    {
        mgcl::Problem p(4, 4, 4);
        p.setGhosts(2);
        mgcl::Level level(&p, 0);

        CHECK(level.getM() == 4);
        CHECK(level.getN() == 4);
        CHECK(level.getO() == 4);
        CHECK(level.getMgh() == 4 + 2 * p.getGhosts());
        CHECK(level.getNgh() == 4 + 2 * p.getGhosts());
        CHECK(level.getOgh() == 4 + 2 * p.getGhosts());
        CHECK(level.getStencilType() == p.getStencilType());
    }
}

TEST_CASE("Level::initOpenCLBuffers")
{
    auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));

    if (!mgcl_test::TestUtility::deviceAvailable("", deviceType))
        return;

    auto v = std::make_shared<mgcl::Cuboid>(4, 4, 4, 1, 1, 1);
    auto f = std::make_shared<mgcl::Cuboid>(4, 4, 4, 1, 1, 1);
    v->fillRandom();
    f->fillRandom();

    SECTION("default conf")
    {
        auto p = std::make_shared<mgcl::Problem>(4, 4, 4, f, v);
        p->setUseOpencl(true);
        p->setGhostsIn(1);
        p->setDeviceType(deviceType);
        REQUIRE(p->init());

        int levelNum = GENERATE(0, 1, 2);
        mgcl::Level level0(p.get(), levelNum);
        REQUIRE(level0.init());
        // REQUIRE(level0.initOpenCLBuffers() == CL_SUCCESS);

        // Check if buffers were created
        REQUIRE(level0.getDVInPtr());
        REQUIRE(level0.getDVOutPtr());
        REQUIRE(level0.getDFPtr());
        REQUIRE(level0.getDRPtr());
        REQUIRE(level0.getDRsqPtr());

        // Check if size of buffers is correct
        int sizeNeeded = (level0.getMgh()) * (level0.getNgh()) * (level0.getOgh());
        REQUIRE(sizeNeeded == level0.getDVIn().getSize());
        REQUIRE(sizeNeeded == level0.getDVOut().getSize());
        REQUIRE(sizeNeeded == level0.getDF().getSize());
        REQUIRE(sizeNeeded == level0.getDR().getSize());
        REQUIRE(level0.getM() * level0.getN() * level0.getO() == level0.getDRsq().getSize());

        // Check if content of dVIn and dV is correct if levelNum = 0
        if (levelNum == 0)
        {
            auto dvin = level0.getDVIn().read(p->getCommands(), nullptr, true);
            auto df = level0.getDF().read(p->getCommands(), nullptr, true);

            CHECK(v->isEqual(*dvin));
            CHECK(f->isEqual(*df));
        }
    }

    SECTION("reuse_opencl_buffers")
    {
        mgcl_test::TestUtility tu(deviceType);
        auto d_v = std::make_shared<mgcl::CuboidGpu>(tu.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, *v);
        auto d_f = std::make_shared<mgcl::CuboidGpu>(tu.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, *f);

        // check ref count of buffers
        cl_uint refCount;
        int err;

        REQUIRE(d_v->refCount() == 1);
        REQUIRE(d_f->refCount() == 1);

        auto p = std::make_shared<mgcl::Problem>(4, 4, 4, d_f, d_v);
        p->setUseOpencl(true);
        p->setReuseOpenclBuffers(true);
        p->reuseOpenCL(tu.getContext(), tu.getCommands(), tu.getDeviceId());

        REQUIRE(d_v->refCount() == 1);
        REQUIRE(d_f->refCount() == 1);

        // Test for levels 0, 1 and 2
        int levelNum = GENERATE(0, 1, 2);
        mgcl::Level* level0 = new mgcl::Level(p.get(), levelNum);
        REQUIRE(level0->init());

        // Check if buffers were created
        REQUIRE(level0->getDVInPtr());
        REQUIRE(level0->getDVOutPtr());
        REQUIRE(level0->getDFPtr());
        REQUIRE(level0->getDRPtr());

        // Check if size of buffers is correct
        int sizeNeeded = (level0->getMgh()) * (level0->getNgh()) * (level0->getOgh());
        REQUIRE(sizeNeeded == level0->getDVIn().getSize());
        REQUIRE(sizeNeeded == level0->getDVOut().getSize());
        REQUIRE(sizeNeeded == level0->getDF().getSize());
        REQUIRE(sizeNeeded == level0->getDR().getSize());

        // Check if content of dVIn and dV is correct if levelNum = 0
        if (levelNum == 0)
        {
            CHECK(d_v.get() == level0->getDVInPtr());
            CHECK(d_f.get() == level0->getDFPtr());
        }

        // TODO maybe enable this test again
        // // Check ref count again, buffers should've been retained
        // REQUIRE(d_v->refCount() == (levelNum == 0 ? 2 : 1));
        // REQUIRE(d_f->refCount() == (levelNum == 0 ? 2 : 1));

        delete level0;

        // Check ref count again, buffers should've been released in Level dtor
        REQUIRE(d_v->refCount() == 1);
        REQUIRE(d_f->refCount() == 1);
    }

    SECTION("VaryingStencils")
    {
        auto p = std::make_shared<mgcl::Problem>(4, 4, 4, f, v);
        p->setUseOpencl(true);
        p->setGhostsIn(1);
        p->setDeviceType(deviceType);

        p->setStencilType(mgcl::MGCL_VARYING);
        auto& sv = p->createStencilValues();
        sv->fillRandomInt();

        REQUIRE(p->init());

        for (int levelNum = 0; levelNum < 3; levelNum++)
        {
            auto& level0 = p->getLevelAt(levelNum);

            REQUIRE(level0.getStencilValuesGpu());
            if (levelNum > 0)
                REQUIRE(!level0.getStencilValues());
        }
    }
}

TEST_CASE("Level::init")
{
    auto v = std::make_shared<mgcl::Cuboid>(4, 4, 4, 1, 1, 1);
    auto f = std::make_shared<mgcl::Cuboid>(4, 4, 4, 1, 1, 1);
    v->fillRandom();
    f->fillRandom();

    auto p = std::make_shared<mgcl::Problem>(4, 4, 4, f, v);
    p->setGhostsIn(1);
    p->setStencilType(mgcl::MGCL_VARYING);
    p->createStencilValues()->fillRandom();
    // REQUIRE(p->init());

    for (int levelNum = 0; levelNum < 3; levelNum++)
    {
        mgcl::Level level(p.get(), levelNum);
        level.init();

        if (levelNum == 0)
        {
            CHECK(level.getV().isEqual(*v));
            CHECK(level.getF().isEqual(*f));
        }

        CHECK(level.getV().getM() == p->getM() >> levelNum);
        CHECK(level.getV().getN() == p->getN() >> levelNum);
        CHECK(level.getV().getO() == p->getO() >> levelNum);

        CHECK(level.getF().getM() == p->getM() >> levelNum);
        CHECK(level.getF().getN() == p->getN() >> levelNum);
        CHECK(level.getF().getO() == p->getO() >> levelNum);

        CHECK(level.getR().getM() == p->getM() >> levelNum);
        CHECK(level.getR().getN() == p->getN() >> levelNum);
        CHECK(level.getR().getO() == p->getO() >> levelNum);
    }
}

TEST_CASE("Level::init_maxLevelUsingOcl")
{
    int N = 8;
    int maxLevelUsingOcl = GENERATE(0, 1, 2, -1);
    int maxLevel = std::log2(N);
    int gh_in = 1;

    auto v = std::make_shared<mgcl::Cuboid>(N, N, N, gh_in, gh_in, gh_in);
    auto f = std::make_shared<mgcl::Cuboid>(N, N, N, gh_in, gh_in, gh_in);
    v->fillRandom();
    f->fillRandom();

    auto p = std::make_shared<mgcl::Problem>(N, N, N, f, v);
    p->setGhostsIn(gh_in);
    p->setUseOpencl(true);
    p->setMaxlevel(maxLevel);
    p->setMaxLevelUsingOcl(maxLevelUsingOcl);
    p->setStencilType(mgcl::MGCL_VARYING);
    p->createStencilValues()->fillRandom();

    p->init();

    for (int levelNum = 0; levelNum < maxLevel; levelNum++)
    {
        auto& level = p->getLevelAt(levelNum);

        CAPTURE(maxLevelUsingOcl, levelNum);

        if (maxLevelUsingOcl < 0 || levelNum <= maxLevelUsingOcl)
        {
            REQUIRE(level.getUseOpencl());

            REQUIRE(level.getDVInPtr());
            REQUIRE(level.getDVOutPtr());
            REQUIRE(level.getDRPtr());
            REQUIRE(level.getDFPtr());
            REQUIRE(level.getStencilValuesGpu());

            if (levelNum == 0)
            {
                REQUIRE(level.getVPtr());
                REQUIRE(level.getFPtr());
                REQUIRE(level.getStencilValues());
            }
            else
            {
                REQUIRE(!level.getVPtr());
                REQUIRE(!level.getFPtr());
                REQUIRE(!level.getStencilValues());
            }
            REQUIRE(!level.getRPtr());
        }
        else
        {
            REQUIRE(!level.getUseOpencl());

            REQUIRE(!level.getDVInPtr());
            REQUIRE(!level.getDVOutPtr());
            REQUIRE(!level.getDRPtr());
            REQUIRE(!level.getDFPtr());
            REQUIRE(!level.getStencilValuesGpu());

            REQUIRE(level.getVPtr());
            REQUIRE(level.getRPtr());
            REQUIRE(level.getFPtr());
            REQUIRE(level.getStencilValues());
        }
    }
}