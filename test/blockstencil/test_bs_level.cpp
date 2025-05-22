#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <iostream>
#include <memory>

#include "../../src/mgcl/cuboid.hpp"
#include "../../src/mgcl/level.hpp"
#include "../../src/mgcl/mpi_global_data.hpp"
#include "../../src/mgcl/problem.hpp"
#include "../cli_args.hpp"
#include "../device_type_generator.hpp"
#include "../test_utility.hpp"

TEST_CASE("Level::initBlockstencil")
{
    int blocksize = 2;
    auto v = std::make_shared<mgcl::CuboidBS>(4, 4, 4, 1, 1, 1, blocksize);
    auto f = std::make_shared<mgcl::CuboidBS>(4, 4, 4, 1, 1, 1, blocksize);
    v->fillRandom();
    f->fillRandom();

    auto p = std::make_shared<mgcl::Problem>(4, 4, 4, f, v);
    p->setGhostsIn(1);
    p->setStencilType(mgcl::MGCL_BLOCKSTENCIL);
    p->getBlockstencil()->fillRandom();
    p->getRestrictionBlockstencil()->fillRandom();
    p->getProlongationBlockstencil()->fillRandom();
    // REQUIRE(p->init());

    for (int levelNum = 0; levelNum < 3; levelNum++)
    {
        CAPTURE(levelNum);
        mgcl::Level level(p.get(), levelNum);
        level.init();

        if (levelNum == 0)
        {
            CHECK(level.getVBS().isEqual(*v));
            CHECK(level.getFBS().isEqual(*f));
        }

        CHECK(level.getVBS().getM() == p->getM() >> levelNum);
        CHECK(level.getVBS().getN() == p->getN() >> levelNum);
        CHECK(level.getVBS().getO() == p->getO() >> levelNum);

        CHECK(level.getFBS().getM() == p->getM() >> levelNum);
        CHECK(level.getFBS().getN() == p->getN() >> levelNum);
        CHECK(level.getFBS().getO() == p->getO() >> levelNum);

        CHECK(level.getRBS().getM() == p->getM() >> levelNum);
        CHECK(level.getRBS().getN() == p->getN() >> levelNum);
        CHECK(level.getRBS().getO() == p->getO() >> levelNum);

        // gets filled with galerkin on coarser levels, which is not called here
        if (levelNum == 0)
        {
            REQUIRE(level.getBlockstencil());
            REQUIRE(level.getBlockstencil()->getM() == p->getM() >> levelNum);
            REQUIRE(level.getBlockstencil()->getN() == p->getN() >> levelNum);
            REQUIRE(level.getBlockstencil()->getO() == p->getO() >> levelNum);
        }
    }
}

TEST_CASE("Level::initOpenCLBuffersBlockstencil")
{
    auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));
    int blocksize = 2;
    auto v = std::make_shared<mgcl::CuboidBS>(4, 4, 4, 1, 1, 1, blocksize);
    auto f = std::make_shared<mgcl::CuboidBS>(4, 4, 4, 1, 1, 1, blocksize);
    v->fillRandom();
    f->fillRandom();

    auto p = std::make_shared<mgcl::Problem>(4, 4, 4, f, v);
    p->setGhostsIn(1);
    p->setStencilType(mgcl::MGCL_BLOCKSTENCIL);
    p->setUseOpencl(true);
    p->getBlockstencil()->fillRandom();
    p->getRestrictionBlockstencil()->fillRandom();
    p->getProlongationBlockstencil()->fillRandom();
    p->setSmootherType(mgcl::MGCL_JACOBI_SCALAR);
    // REQUIRE(p->init());
    p->initOpenCL();

    for (int levelNum = 0; levelNum < 3; levelNum++)
    {
        mgcl::Level level(p.get(), levelNum);
        level.init();

        REQUIRE(level.getDVBSInPtr());
        REQUIRE(level.getDVBSOutPtr());
        REQUIRE(level.getDFBSPtr());
        REQUIRE(level.getDRBSPtr());
        REQUIRE(level.getDRsqBSPtr());
        if (levelNum > 0)
            REQUIRE_THROWS(level.getBlockstencilInvScalar()); // is only initialized after galerkin
        else
            REQUIRE(level.getBlockstencilGpuInvScalar()); // for lv 0 it is initialized right away

        if (levelNum == 0)
        {
            CHECK(level.getVBS().isEqual(*v));
            CHECK(level.getFBS().isEqual(*f));

            REQUIRE(level.getBlockstencilGpu());
            REQUIRE(level.getBlockstencil()->getM() == p->getM() >> levelNum);
            REQUIRE(level.getBlockstencil()->getN() == p->getN() >> levelNum);
            REQUIRE(level.getBlockstencil()->getO() == p->getO() >> levelNum);
        }

        CHECK(level.getDVBSIn().getM() == p->getM() >> levelNum);
        CHECK(level.getDVBSIn().getN() == p->getN() >> levelNum);
        CHECK(level.getDVBSIn().getO() == p->getO() >> levelNum);

        CHECK(level.getDVBSOut().getM() == p->getM() >> levelNum);
        CHECK(level.getDVBSOut().getN() == p->getN() >> levelNum);
        CHECK(level.getDVBSOut().getO() == p->getO() >> levelNum);

        CHECK(level.getDFBS().getM() == p->getM() >> levelNum);
        CHECK(level.getDFBS().getN() == p->getN() >> levelNum);
        CHECK(level.getDFBS().getO() == p->getO() >> levelNum);

        CHECK(level.getDRBS().getM() == p->getM() >> levelNum);
        CHECK(level.getDRBS().getN() == p->getN() >> levelNum);
        CHECK(level.getDRBS().getO() == p->getO() >> levelNum);
    }
}