#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <iostream>
#include <memory>

#include "../cuboid.hpp"
#include "../problem.hpp"
#include "test_utility.hpp"

/**
 * @brief Tests if config parameters of Problem are working correctly, including setting things up.
 *
 */
TEST_CASE("Level ctor and dtor")
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
    }

    // SECTION("using OpenCL")
    // {
    //     mgcl::Problem p(4, 4, 4);
    //     p.setUseOpencl(true);
    //     p.setGhosts(2);
    //     p.init();
    //     mgcl::Level level(&p, 0, 4, 4, 4);

    //     CHECK(level.getM() == 4);
    //     CHECK(level.getN() == 4);
    //     CHECK(level.getO() == 4);
    //     CHECK(level.getMgh() == 4 + 2 * p.getGhosts());
    //     CHECK(level.getNgh() == 4 + 2 * p.getGhosts());
    //     CHECK(level.getOgh() == 4 + 2 * p.getGhosts());
    // }
}

TEST_CASE("Level::initOpenCLBuffers")
{
    auto v = std::make_shared<mgcl::Cuboid>(4, 4, 4);
    auto f = std::make_shared<mgcl::Cuboid>(4, 4, 4);
    v->fillRandom();
    f->fillRandom();

    auto p = std::make_shared<mgcl::Problem>(4, 4, 4, f, v);
    p->setUseOpencl(true);
    REQUIRE(p->init());

    SECTION("default conf")
    {
        int levelNum = GENERATE(0, 1, 2);
        mgcl::Level level0(p.get(), levelNum);
        REQUIRE(level0.init());
        // REQUIRE(level0.initOpenCLBuffers() == CL_SUCCESS);

        // Check if buffers were created
        REQUIRE(level0.getDVIn());
        REQUIRE(level0.getDVOut());
        REQUIRE(level0.getDF());
        REQUIRE(level0.getDR());

        // Check if size of buffers is correct
        size_t bufsize;
        int sizeNeeded = sizeof(double) * (level0.getMgh()) * (level0.getNgh()) * (level0.getOgh());
        int err;

        err = clGetMemObjectInfo(level0.getDVIn(), CL_MEM_SIZE, sizeof(size_t), &bufsize, nullptr);
        mgcl::mgclCheckError(err, "clGetMemObjectInfo(level0.dVIn)");
        REQUIRE(err == CL_SUCCESS);
        CHECK(bufsize == sizeNeeded);

        err = clGetMemObjectInfo(level0.getDVOut(), CL_MEM_SIZE, sizeof(size_t), &bufsize, nullptr);
        mgcl::mgclCheckError(err, "clGetMemObjectInfo(level0.dVOut)");
        REQUIRE(err == CL_SUCCESS);
        CHECK(bufsize == sizeNeeded);

        err = clGetMemObjectInfo(level0.getDF(), CL_MEM_SIZE, sizeof(size_t), &bufsize, nullptr);
        mgcl::mgclCheckError(err, "clGetMemObjectInfo(level0.dF)");
        REQUIRE(err == CL_SUCCESS);
        CHECK(bufsize == sizeNeeded);

        err = clGetMemObjectInfo(level0.getDR(), CL_MEM_SIZE, sizeof(size_t), &bufsize, nullptr);
        mgcl::mgclCheckError(err, "clGetMemObjectInfo(level0.dR)");
        REQUIRE(err == CL_SUCCESS);
        CHECK(bufsize == sizeNeeded);

        // Check if content of dVIn and dV is correct if levelNum = 0
        if (levelNum == 0)
        {
            mgcl_test::TestUtility tu(p);
            auto dvin = tu.readOpenCLBuffer(level0.getDVIn(), level0.getM(), level0.getN(), level0.getO(), p->getGhosts(), p->getGhosts(), p->getGhosts());
            auto df = tu.readOpenCLBuffer(level0.getDF(), level0.getM(), level0.getN(), level0.getO(), p->getGhosts(), p->getGhosts(), p->getGhosts());

            CHECK(v->isEqual(dvin));
            CHECK(f->isEqual(df));
        }
    }
}
