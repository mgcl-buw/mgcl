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
    }
}

TEST_CASE("Level::initOpenCLBuffers")
{
    auto deviceType = GENERATE(CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU);

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

            CHECK(v->isEqual(*dvin));
            CHECK(f->isEqual(*df));
        }
    }

    SECTION("reuse_opencl_buffers")
    {
        mgcl_test::TestUtility tu(deviceType);
        auto d_v = tu.createOpenCLBuffer(*v);
        auto d_f = tu.createOpenCLBuffer(*f);

        // check ref count of buffers
        cl_uint refCount;
        int err;

        err = clGetMemObjectInfo(d_v, CL_MEM_REFERENCE_COUNT, sizeof(cl_uint), &refCount, nullptr);
        mgcl::mgclCheckError(err, "clGetMemObjectInfo(d_v, CL_MEM_REFERENCE_COUNT)");
        REQUIRE(err == CL_SUCCESS);
        CHECK(refCount == 1);

        err = clGetMemObjectInfo(d_f, CL_MEM_REFERENCE_COUNT, sizeof(cl_uint), &refCount, nullptr);
        mgcl::mgclCheckError(err, "clGetMemObjectInfo(d_f, CL_MEM_REFERENCE_COUNT)");
        REQUIRE(err == CL_SUCCESS);
        CHECK(refCount == 1);

        auto p = std::make_shared<mgcl::Problem>(4, 4, 4, d_f, d_v);
        p->setUseOpencl(true);
        p->setReuseOpenclBuffers(true);
        p->reuseOpenCL(tu.getContext(), tu.getCommands(), tu.getDeviceId());

        err = clGetMemObjectInfo(d_v, CL_MEM_REFERENCE_COUNT, sizeof(cl_uint), &refCount, nullptr);
        mgcl::mgclCheckError(err, "clGetMemObjectInfo(d_v, CL_MEM_REFERENCE_COUNT)");
        REQUIRE(err == CL_SUCCESS);
        CHECK(refCount == 1);

        err = clGetMemObjectInfo(d_f, CL_MEM_REFERENCE_COUNT, sizeof(cl_uint), &refCount, nullptr);
        mgcl::mgclCheckError(err, "clGetMemObjectInfo(d_f, CL_MEM_REFERENCE_COUNT)");
        REQUIRE(err == CL_SUCCESS);
        CHECK(refCount == 1);

        // Test for levels 0, 1 and 2
        int levelNum = GENERATE(0, 1, 2);
        mgcl::Level *level0 = new mgcl::Level(p.get(), levelNum);
        REQUIRE(level0->init());

        // Check if buffers were created
        REQUIRE(level0->getDVIn());
        REQUIRE(level0->getDVOut());
        REQUIRE(level0->getDF());
        REQUIRE(level0->getDR());

        // Check if size of buffers is correct
        size_t bufsize;
        int sizeNeeded = sizeof(double) * (level0->getMgh()) * (level0->getNgh()) * (level0->getOgh());

        err = clGetMemObjectInfo(level0->getDVIn(), CL_MEM_SIZE, sizeof(size_t), &bufsize, nullptr);
        mgcl::mgclCheckError(err, "clGetMemObjectInfo(level0->dVIn)");
        REQUIRE(err == CL_SUCCESS);
        CHECK(bufsize == sizeNeeded);

        err = clGetMemObjectInfo(level0->getDVOut(), CL_MEM_SIZE, sizeof(size_t), &bufsize, nullptr);
        mgcl::mgclCheckError(err, "clGetMemObjectInfo(level0->dVOut)");
        REQUIRE(err == CL_SUCCESS);
        CHECK(bufsize == sizeNeeded);

        err = clGetMemObjectInfo(level0->getDF(), CL_MEM_SIZE, sizeof(size_t), &bufsize, nullptr);
        mgcl::mgclCheckError(err, "clGetMemObjectInfo(level0->dF)");
        REQUIRE(err == CL_SUCCESS);
        CHECK(bufsize == sizeNeeded);

        err = clGetMemObjectInfo(level0->getDR(), CL_MEM_SIZE, sizeof(size_t), &bufsize, nullptr);
        mgcl::mgclCheckError(err, "clGetMemObjectInfo(level0->dR)");
        REQUIRE(err == CL_SUCCESS);
        CHECK(bufsize == sizeNeeded);

        // Check if content of dVIn and dV is correct if levelNum = 0
        if (levelNum == 0)
        {
            CHECK(d_v == level0->getDVIn());
            CHECK(d_f == level0->getDF());
        }

        // Check ref count again, buffers should've been retained
        err = clGetMemObjectInfo(d_v, CL_MEM_REFERENCE_COUNT, sizeof(cl_uint), &refCount, nullptr);
        mgcl::mgclCheckError(err, "clGetMemObjectInfo(d_v, CL_MEM_REFERENCE_COUNT)");
        REQUIRE(err == CL_SUCCESS);
        CHECK(refCount == (levelNum == 0 ? 2 : 1));

        err = clGetMemObjectInfo(d_f, CL_MEM_REFERENCE_COUNT, sizeof(cl_uint), &refCount, nullptr);
        mgcl::mgclCheckError(err, "clGetMemObjectInfo(d_f, CL_MEM_REFERENCE_COUNT)");
        REQUIRE(err == CL_SUCCESS);
        CHECK(refCount == (levelNum == 0 ? 2 : 1));

        delete level0;

        // Check ref count again, buffers should've been released in Level dtor
        err = clGetMemObjectInfo(d_v, CL_MEM_REFERENCE_COUNT, sizeof(cl_uint), &refCount, nullptr);
        mgcl::mgclCheckError(err, "clGetMemObjectInfo(d_v, CL_MEM_REFERENCE_COUNT)");
        REQUIRE(err == CL_SUCCESS);
        CHECK(refCount == 1);

        err = clGetMemObjectInfo(d_f, CL_MEM_REFERENCE_COUNT, sizeof(cl_uint), &refCount, nullptr);
        mgcl::mgclCheckError(err, "clGetMemObjectInfo(d_f, CL_MEM_REFERENCE_COUNT)");
        REQUIRE(err == CL_SUCCESS);
        CHECK(refCount == 1);
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
    // REQUIRE(p->init());

    int levelNum = GENERATE(0, 1, 2);
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
