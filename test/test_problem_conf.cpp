#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <iostream>

#include "../cuboid.hpp"
#include "../opencl_helper.hpp"
#include "../problem.hpp"
#include "test_utility.hpp"

/**
 * @brief Tests if config parameters of Problem are working correctly, including setting things up.
 *
 */
TEST_CASE("Problem conf")
{
    auto v = std::make_shared<mgcl::Cuboid>(8, 8, 8);
    auto f = std::make_shared<mgcl::Cuboid>(8, 8, 8);
    mgcl::Problem p(8, 8, 8, v, f);

    SECTION("default values")
    {
        // REQUIRE(p.getV() == v);
        // REQUIRE(p.getF() == f);
        REQUIRE(p.getM() == 8);
        REQUIRE(p.getN() == 8);
        REQUIRE(p.getO() == 8);
        REQUIRE(p.getGhosts() == 1);
        REQUIRE(p.getGhostsIn() == 0);
        REQUIRE(p.getNu1() == 2);
        REQUIRE(p.getNu2() == 2);
        REQUIRE(p.getOmega() == 0.8);
        REQUIRE(p.getMaxiterVcycles() == 5);
        REQUIRE(p.getTol() == 1e-7);
        REQUIRE(p.getMaxlevel() == -1);
        REQUIRE(p.getResidualNorm() == mgcl::MGCL_L2);
        REQUIRE(p.getStencil() == mgcl::MGCL_7POINT);

        REQUIRE(!p.getOpenCLHelper().isInitialized());
        REQUIRE(p.getDeviceType() == CL_DEVICE_TYPE_DEFAULT);
        REQUIRE(p.getKernelDir() == "./");
        REQUIRE(p.getDeviceName() == "");
        REQUIRE(p.getDV() == nullptr);
        REQUIRE(p.getDF() == nullptr);
        REQUIRE(p.getUseOpencl() == false);
        REQUIRE(p.getReuseOpenclBuffers() == false);
        REQUIRE(p.getCopyBufferData() == false);
        REQUIRE(p.getReadResults() == false);
        REQUIRE(p.getUseLocalMemory() == false);

        REQUIRE(p.getJacobiWgSizeX() == 16);
        REQUIRE(p.getJacobiWgSizeY() == 16);
        REQUIRE(p.getJacobiIterationsPerKernel() == 3);

        REQUIRE(p.getStencilSizeMultiplier() == 1);
        REQUIRE(p.getStencilValuesPtr() == nullptr);
        REQUIRE(p.getDStencilValues() == nullptr);
        REQUIRE(p.getRestrictProlongateStencil() == true);
    }
}

TEST_CASE("Problem::checkParameters")
{
    auto v = std::make_shared<mgcl::Cuboid>(8, 8, 8);
    auto f = std::make_shared<mgcl::Cuboid>(8, 8, 8);

    SECTION("m,n,o wrong")
    {
        mgcl::Problem pm(0, 1, 1, v, f);
        mgcl::Problem pn(0, 1, 1, v, f);
        mgcl::Problem po(0, 1, 1, v, f);

        REQUIRE(pm.checkParameters() == false);
        REQUIRE(pn.checkParameters() == false);
        REQUIRE(po.checkParameters() == false);
    }

    SECTION("ghosts 0")
    {
        mgcl::Problem pm(8, 8, 8, v, f);
        pm.setGhosts(0);
        REQUIRE(pm.checkParameters() == false);
    }

    SECTION("ghosts_in -1")
    {
        mgcl::Problem pm(8, 8, 8, v, f);
        pm.setGhostsIn(-1);
        REQUIRE(pm.checkParameters() == false);
    }

    SECTION("stencil_values not set")
    {
        mgcl::Problem pm(8, 8, 8, v, f);
        pm.setStencil(mgcl::MGCL_STENCIL::MGCL_19POINT_VARSYM);
        REQUIRE(pm.checkParameters() == false);
    }

    SECTION("all good")
    {
        mgcl::Problem pm(8, 8, 8, v, f);
        REQUIRE(pm.checkParameters() == true);
    }
}

TEST_CASE("Problem::calculateAndSetMaxLevel")
{
    SECTION("m min")
    {
        mgcl::Problem p(4, 8, 8);

        REQUIRE(p.getMaxlevel() == -1);
        p.calculateAndSetMaxLevel();
        REQUIRE(p.getMaxlevel() == 3);
    }

    SECTION("n min")
    {
        mgcl::Problem p(16, 8, 16);

        REQUIRE(p.getMaxlevel() == -1);
        p.calculateAndSetMaxLevel();
        REQUIRE(p.getMaxlevel() == 4);
    }

    SECTION("o min")
    {
        mgcl::Problem p(32, 32, 16);

        REQUIRE(p.getMaxlevel() == -1);
        p.calculateAndSetMaxLevel();
        REQUIRE(p.getMaxlevel() == 5);
    }

    SECTION("user set valid")
    {
        mgcl::Problem p(32, 32, 32);
        p.setMaxlevel(2);

        p.calculateAndSetMaxLevel();
        REQUIRE(p.getMaxlevel() == 2);
    }

    SECTION("user set invalid")
    {
        mgcl::Problem p(4, 4, 4);
        p.setMaxlevel(8);

        p.calculateAndSetMaxLevel();
        REQUIRE(p.getMaxlevel() == 3);
    }
}

TEST_CASE("Problem::initOpenCL, not reusing environment")
{
    mgcl::Problem p(4, 4, 4);
    REQUIRE(!p.getOpenCLHelper().isInitialized());
    REQUIRE(p.getUseOpencl() == false);

    p.setUseOpencl(true);
    p.initOpenCL();
    REQUIRE(p.getUseOpencl() == true);
    REQUIRE(p.getOpenCLHelper().isInitialized());
    REQUIRE(p.getCommands() != nullptr);
    REQUIRE(p.getContext() != nullptr);
    REQUIRE(p.getDeviceId() != nullptr);
}

TEST_CASE("Problem::initOpenCL, reusing environment")
{
    mgcl::Problem p(4, 4, 4);
    REQUIRE(!p.getOpenCLHelper().isInitialized());
    REQUIRE(p.getUseOpencl() == false);

    mgcl::OpenCLHelper openCLHelper(&p);
    openCLHelper.init();

    p.reuseOpenCL(openCLHelper.getContext(), openCLHelper.getCommands(), openCLHelper.getDeviceId());
    REQUIRE(p.getUseOpencl() == true);
    REQUIRE(p.getOpenCLHelper().isInitialized());
    REQUIRE(p.getCommands() != nullptr);
    REQUIRE(p.getContext() != nullptr);
    REQUIRE(p.getDeviceId() != nullptr);
}

TEST_CASE("set OpenCLHelper values")
{
    mgcl::Problem p(4, 4, 4);

    SECTION("getters, openCLHandler nullptr")
    {
        REQUIRE(p.getCommands() == nullptr);
        REQUIRE(p.getContext() == nullptr);
        REQUIRE(p.getDeviceId() == nullptr);
        REQUIRE(p.getDeviceType() == CL_DEVICE_TYPE_DEFAULT);
        REQUIRE(p.getKernelDir() == "./");
        REQUIRE(p.getDeviceName() == "");
    }

    /**
     * @brief Using setters before OpenCLHelper was initialized should set the values appropriately.
     *
     */
    SECTION("setters, openCLHandler not initialized")
    {
        REQUIRE(!p.getOpenCLHelper().isInitialized());

        p.setDeviceType(CL_DEVICE_TYPE_GPU);
        p.setKernelDir("test/");
        p.setDeviceName("Quadro");

        REQUIRE(p.getDeviceType() == CL_DEVICE_TYPE_GPU);
        REQUIRE(p.getKernelDir() == "test/");
        REQUIRE(p.getDeviceName() == "Quadro");
    }

    /**
     * @brief Using setters after OpenCLHelper was initialized should not have an effect.
     *
     */
    SECTION("setters, openCLHandler initialized")
    {
        REQUIRE(!p.getOpenCLHelper().isInitialized());
        p.initOpenCL();
        REQUIRE(p.getOpenCLHelper().isInitialized());

        p.setDeviceType(CL_DEVICE_TYPE_GPU);
        p.setKernelDir("test/");
        p.setDeviceName("Quadro");

        REQUIRE(p.getDeviceType() == CL_DEVICE_TYPE_DEFAULT);
        REQUIRE(p.getKernelDir() == "./");
        REQUIRE(p.getDeviceName() == "");
    }
}

/**
 * @brief Should set maxlevel, create buffers and set h for each level
 *
 */
TEST_CASE("Problem::init")
{
    int m = 4;
    int n = 4;
    int o = 4;
    auto v = std::make_shared<mgcl::Cuboid>(m, n, o);
    auto f = std::make_shared<mgcl::Cuboid>(m, n, o);
    mgcl::Problem p(m, n, o, f, v);
    v->fillRandom();
    f->fillRandom();

    SECTION("default conf, v and f nullptr")
    {
        mgcl::Problem p2(8, 8, 8);
        REQUIRE(!p2.init());
    }

    /**
     * @brief Should create level buffers on host only.
     *
     */
    SECTION("default conf")
    {
        REQUIRE(p.init());
        REQUIRE(p.getLevels().size() == p.getMaxlevel());

        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < o; k++)
                {
                    REQUIRE((*v)[i][j][k] == p.getLevels()[0]->getV()[i + 1][j + 1][k + 1]);
                    REQUIRE((*f)[i][j][k] == p.getLevels()[0]->getF()[i + 1][j + 1][k + 1]);
                }

        for (int lv = 0; lv < p.getMaxlevel(); lv++)
        {
            std::cout << "Checking level " + lv << std::endl;

            REQUIRE(p.getLevels()[lv]->getM() == m >> lv);
            REQUIRE(p.getLevels()[lv]->getN() == n >> lv);
            REQUIRE(p.getLevels()[lv]->getO() == o >> lv);

            REQUIRE(p.getLevels()[lv]->getMgh() == p.getLevels()[lv]->getM() + 2 * p.getGhosts());
            REQUIRE(p.getLevels()[lv]->getNgh() == p.getLevels()[lv]->getN() + 2 * p.getGhosts());
            REQUIRE(p.getLevels()[lv]->getOgh() == p.getLevels()[lv]->getO() + 2 * p.getGhosts());

            REQUIRE(p.getLevels()[lv]->getRPtr());

            // check if cuboids have correct dimensions
            CHECK(p.getLevels()[lv]->getV().getM() == p.getLevels()[lv]->getM());
            CHECK(p.getLevels()[lv]->getV().getN() == p.getLevels()[lv]->getN());
            CHECK(p.getLevels()[lv]->getV().getO() == p.getLevels()[lv]->getO());
            CHECK(p.getLevels()[lv]->getV().getMgh() == p.getLevels()[lv]->getMgh());
            CHECK(p.getLevels()[lv]->getV().getNgh() == p.getLevels()[lv]->getNgh());
            CHECK(p.getLevels()[lv]->getV().getOgh() == p.getLevels()[lv]->getOgh());
            CHECK(p.getLevels()[lv]->getV().getGhostsM() == p.getGhosts());
            CHECK(p.getLevels()[lv]->getV().getGhostsN() == p.getGhosts());
            CHECK(p.getLevels()[lv]->getV().getGhostsO() == p.getGhosts());

            CHECK(p.getLevels()[lv]->getF().getM() == p.getLevels()[lv]->getM());
            CHECK(p.getLevels()[lv]->getF().getN() == p.getLevels()[lv]->getN());
            CHECK(p.getLevels()[lv]->getF().getO() == p.getLevels()[lv]->getO());
            CHECK(p.getLevels()[lv]->getF().getMgh() == p.getLevels()[lv]->getMgh());
            CHECK(p.getLevels()[lv]->getF().getNgh() == p.getLevels()[lv]->getNgh());
            CHECK(p.getLevels()[lv]->getF().getOgh() == p.getLevels()[lv]->getOgh());
            CHECK(p.getLevels()[lv]->getF().getGhostsM() == p.getGhosts());
            CHECK(p.getLevels()[lv]->getF().getGhostsN() == p.getGhosts());
            CHECK(p.getLevels()[lv]->getF().getGhostsO() == p.getGhosts());

            CHECK(p.getLevels()[lv]->getR().getM() == p.getLevels()[lv]->getM());
            CHECK(p.getLevels()[lv]->getR().getN() == p.getLevels()[lv]->getN());
            CHECK(p.getLevels()[lv]->getR().getO() == p.getLevels()[lv]->getO());
            CHECK(p.getLevels()[lv]->getR().getMgh() == p.getLevels()[lv]->getMgh());
            CHECK(p.getLevels()[lv]->getR().getNgh() == p.getLevels()[lv]->getNgh());
            CHECK(p.getLevels()[lv]->getR().getOgh() == p.getLevels()[lv]->getOgh());
            CHECK(p.getLevels()[lv]->getR().getGhostsM() == p.getGhosts());
            CHECK(p.getLevels()[lv]->getR().getGhostsN() == p.getGhosts());
            CHECK(p.getLevels()[lv]->getR().getGhostsO() == p.getGhosts());

            // TODO check h when used
        }
    }

    SECTION("setting ghosts_in")
    {
        // run this section for ghosts_in = 0, 1 and 2
        auto ghosts_in = GENERATE(0, 1, 2);
        int mgh = m + 2 * ghosts_in;
        int ngh = n + 2 * ghosts_in;
        int ogh = o + 2 * ghosts_in;

        auto v2 = std::make_shared<mgcl::Cuboid>(mgh, ngh, ogh);
        auto f2 = std::make_shared<mgcl::Cuboid>(mgh, ngh, ogh);
        mgcl::Problem p2(m, n, o, v2, f2);

        p2.setGhostsIn(ghosts_in);
        REQUIRE(p2.init());
        REQUIRE(p2.getLevels().size() == p2.getMaxlevel());

        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < o; k++)
                {
                    REQUIRE((*v2)[i + ghosts_in][j + ghosts_in][k + ghosts_in] == p2.getLevels()[0]->getV()[i + 1][j + 1][k + 1]);
                    REQUIRE((*f2)[i + ghosts_in][j + ghosts_in][k + ghosts_in] == p2.getLevels()[0]->getF()[i + 1][j + 1][k + 1]);
                }

        for (int lv = 0; lv < p2.getMaxlevel(); lv++)
        {
            REQUIRE(p2.getLevels()[lv]->getM() == m >> lv);
            REQUIRE(p2.getLevels()[lv]->getN() == n >> lv);
            REQUIRE(p2.getLevels()[lv]->getO() == o >> lv);

            REQUIRE(p2.getLevels()[lv]->getMgh() == p2.getLevels()[lv]->getM() + 2 * p2.getGhosts());
            REQUIRE(p2.getLevels()[lv]->getNgh() == p2.getLevels()[lv]->getN() + 2 * p2.getGhosts());
            REQUIRE(p2.getLevels()[lv]->getOgh() == p2.getLevels()[lv]->getO() + 2 * p2.getGhosts());

            REQUIRE(p2.getLevels()[lv]->getRPtr());

            // TODO check h when used
        }
    }

    SECTION("setting ghosts")
    {
        // run this section for ghosts = 1, 2 and 3
        auto ghosts = GENERATE(1, 2, 3);
        int mgh = m + 2 * ghosts;
        int ngh = n + 2 * ghosts;
        int ogh = o + 2 * ghosts;

        p.setGhosts(ghosts);
        REQUIRE(p.init());
        REQUIRE(p.getLevels().size() == p.getMaxlevel());

        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < o; k++)
                {
                    REQUIRE((*v)[i][j][k] == p.getLevels()[0]->getV()[i + ghosts][j + ghosts][k + ghosts]);
                    REQUIRE((*f)[i][j][k] == p.getLevels()[0]->getF()[i + ghosts][j + ghosts][k + ghosts]);
                }

        for (int lv = 0; lv < p.getMaxlevel(); lv++)
        {
            REQUIRE(p.getLevels()[lv]->getM() == m >> lv);
            REQUIRE(p.getLevels()[lv]->getN() == n >> lv);
            REQUIRE(p.getLevels()[lv]->getO() == o >> lv);

            REQUIRE(p.getLevels()[lv]->getMgh() == p.getLevels()[lv]->getM() + 2 * p.getGhosts());
            REQUIRE(p.getLevels()[lv]->getNgh() == p.getLevels()[lv]->getN() + 2 * p.getGhosts());
            REQUIRE(p.getLevels()[lv]->getOgh() == p.getLevels()[lv]->getO() + 2 * p.getGhosts());

            REQUIRE(p.getLevels()[lv]->getRPtr());

            // TODO check h when used
        }
    }

    SECTION("reuse OpenCL buffers")
    {
        int ghosts = 1;
        mgcl::Cuboid vgh(m + 2 * ghosts, n + 2 * ghosts, o + 2 * ghosts);
        mgcl::Cuboid fgh(m + 2 * ghosts, n + 2 * ghosts, o + 2 * ghosts);

        mgcl_test::TestUtility tu;
        cl_mem d_v = tu.createOpenCLBuffer(vgh);
        cl_mem d_f = tu.createOpenCLBuffer(fgh);

        mgcl::Problem p2(m, n, o, d_f, d_v);
        p2.setGhostsIn(ghosts);
        p2.setReuseOpenclBuffers(true);
        REQUIRE(p2.getDF() == d_f);
        REQUIRE(p2.getDV() == d_v);
        REQUIRE(p2.getOpenCLHelper().getProblem() == &p2);

        REQUIRE(p2.reuseOpenCL(tu.getContext(), tu.getCommands(), tu.getDeviceId()) == CL_SUCCESS);
        REQUIRE(p2.init());
        REQUIRE(p2.getLevels().size() == p2.getMaxlevel());

        // no host data is created
        REQUIRE(!p2.getVPtr());
        REQUIRE(!p2.getFPtr());
        for (int lv = 0; lv < p2.getMaxlevel(); lv++)
        {
            REQUIRE(!p2.getLevels()[lv]->getVPtr());
            REQUIRE(!p2.getLevels()[lv]->getFPtr());
            REQUIRE(!p2.getLevels()[lv]->getRPtr());
        }

        // buffers are created
        REQUIRE(p2.getDV() == d_v);
        REQUIRE(p2.getDF() == d_f);
        REQUIRE(p2.getLevels()[0]);
        REQUIRE(p2.getLevels()[0]->getDVIn() == d_v);
        REQUIRE(p2.getLevels()[0]->getDF() == d_f);
        for (int lv = 0; lv < p2.getMaxlevel(); lv++)
        {
            // check if buffers are not nullptr
            REQUIRE(p2.getLevels()[lv]);
            REQUIRE(p2.getLevels()[lv]->getDVIn());
            REQUIRE(p2.getLevels()[lv]->getDVOut());
            REQUIRE(p2.getLevels()[lv]->getDF());
            REQUIRE(p2.getLevels()[lv]->getDR());

            // check sizes of buffers
            int err;
            size_t bufsize;
            int sizeNeeded = sizeof(double) * p2.getLevels()[lv]->getMgh() * p2.getLevels()[lv]->getNgh() * p2.getLevels()[lv]->getOgh();

            err = clGetMemObjectInfo(p2.getLevels()[lv]->getDVIn(), CL_MEM_SIZE, sizeof(size_t), &bufsize, nullptr);
            mgcl::mgclCheckError(err, "clGetMemObjectInfo");
            REQUIRE(sizeNeeded == bufsize);

            err = clGetMemObjectInfo(p2.getLevels()[lv]->getDVOut(), CL_MEM_SIZE, sizeof(size_t), &bufsize, nullptr);
            mgcl::mgclCheckError(err, "clGetMemObjectInfo");
            REQUIRE(sizeNeeded == bufsize);

            err = clGetMemObjectInfo(p2.getLevels()[lv]->getDF(), CL_MEM_SIZE, sizeof(size_t), &bufsize, nullptr);
            mgcl::mgclCheckError(err, "clGetMemObjectInfo");
            REQUIRE(sizeNeeded == bufsize);

            err = clGetMemObjectInfo(p2.getLevels()[lv]->getDR(), CL_MEM_SIZE, sizeof(size_t), &bufsize, nullptr);
            mgcl::mgclCheckError(err, "clGetMemObjectInfo");
            REQUIRE(sizeNeeded == bufsize);
        }
    }

    SECTION("copy OpenCL buffers")
    {
        int ghosts = 1;
        mgcl::Cuboid vgh(m + 2 * ghosts, n + 2 * ghosts, o + 2 * ghosts);
        mgcl::Cuboid fgh(m + 2 * ghosts, n + 2 * ghosts, o + 2 * ghosts);

        mgcl_test::TestUtility tu;
        cl_mem d_v = tu.createOpenCLBuffer(vgh);
        cl_mem d_f = tu.createOpenCLBuffer(fgh);

        mgcl::Problem p2(m, n, o, d_f, d_v);
        p2.setGhostsIn(ghosts);
        p2.setCopyBufferData(true);
        REQUIRE(p2.getDF() == d_f);
        REQUIRE(p2.getDV() == d_v);
        REQUIRE(p2.getOpenCLHelper().getProblem() == &p2);

        REQUIRE(p2.reuseOpenCL(tu.getContext(), tu.getCommands(), tu.getDeviceId()) == CL_SUCCESS);
        REQUIRE(p2.init());
        REQUIRE(p2.getLevels().size() == p2.getMaxlevel());

        // no host data is created
        REQUIRE(!p2.getVPtr());
        REQUIRE(!p2.getFPtr());
        for (int lv = 0; lv < p2.getMaxlevel(); lv++)
        {
            REQUIRE(!p2.getLevels()[lv]->getVPtr());
            REQUIRE(!p2.getLevels()[lv]->getFPtr());
            REQUIRE(!p2.getLevels()[lv]->getRPtr());
        }

        // buffers are created
        REQUIRE(p2.getDV() == d_v);
        REQUIRE(p2.getDF() == d_f);
        REQUIRE(p2.getLevels()[0]);
        REQUIRE(p2.getLevels()[0]->getDVIn() != d_v);
        REQUIRE(p2.getLevels()[0]->getDF() != d_f);
        for (int lv = 0; lv < p2.getMaxlevel(); lv++)
        {
            REQUIRE(p2.getLevels()[lv]);
            REQUIRE(p2.getLevels()[lv]->getDVIn());
            REQUIRE(p2.getLevels()[lv]->getDVOut());
            REQUIRE(p2.getLevels()[lv]->getDF());
            REQUIRE(p2.getLevels()[lv]->getDR());

            // check sizes of buffers
            int err;
            size_t bufsize;
            int sizeNeeded = sizeof(double) * p2.getLevels()[lv]->getMgh() * p2.getLevels()[lv]->getNgh() * p2.getLevels()[lv]->getOgh();

            err = clGetMemObjectInfo(p2.getLevels()[lv]->getDVIn(), CL_MEM_SIZE, sizeof(size_t), &bufsize, nullptr);
            mgcl::mgclCheckError(err, "clGetMemObjectInfo");
            REQUIRE(sizeNeeded == bufsize);

            err = clGetMemObjectInfo(p2.getLevels()[lv]->getDVOut(), CL_MEM_SIZE, sizeof(size_t), &bufsize, nullptr);
            mgcl::mgclCheckError(err, "clGetMemObjectInfo");
            REQUIRE(sizeNeeded == bufsize);

            err = clGetMemObjectInfo(p2.getLevels()[lv]->getDF(), CL_MEM_SIZE, sizeof(size_t), &bufsize, nullptr);
            mgcl::mgclCheckError(err, "clGetMemObjectInfo");
            REQUIRE(sizeNeeded == bufsize);

            err = clGetMemObjectInfo(p2.getLevels()[lv]->getDR(), CL_MEM_SIZE, sizeof(size_t), &bufsize, nullptr);
            mgcl::mgclCheckError(err, "clGetMemObjectInfo");
            REQUIRE(sizeNeeded == bufsize);
        }

        // contents of copied buffer and input buffers are equal
        auto lv0d = tu.readOpenCLBuffer(p2.getLevels()[0]->getDVIn(), p2.getLevels()[0]->getMgh(), p2.getLevels()[0]->getNgh(), p2.getLevels()[0]->getOgh());
        auto lv0f = tu.readOpenCLBuffer(p2.getLevels()[0]->getDF(), p2.getLevels()[0]->getMgh(), p2.getLevels()[0]->getNgh(), p2.getLevels()[0]->getOgh());

        REQUIRE(vgh.isEqual(lv0d));
        REQUIRE(fgh.isEqual(lv0f));
    }
}
