#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <iostream>
#include <memory>

#include "../src/cuboid.hpp"
#include "../src/opencl_helper.hpp"
#include "../src/problem.hpp"
#include "../src/stencil.hpp"
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
        REQUIRE(p.getMaxlevel() == 3);
        REQUIRE(p.getResidualNorm() == mgcl::MGCL_L2);
        REQUIRE(p.getStencilType() == mgcl::MGCL_LAPLACE_7POINT);

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
        REQUIRE(p.getJacobiIterationsPerKernel() == 1);
    }
}

TEST_CASE("Problem::checkParameters")
{
    auto v = std::make_shared<mgcl::Cuboid>(8, 8, 8);
    auto f = std::make_shared<mgcl::Cuboid>(8, 8, 8);

    SECTION("m,n,o wrong")
    {
        REQUIRE_THROWS(std::make_unique<mgcl::Problem>(0, 1, 1, v, f));
        REQUIRE_THROWS(std::make_unique<mgcl::Problem>(1, 0, 1, v, f));
        REQUIRE_THROWS(std::make_unique<mgcl::Problem>(1, 1, 0, v, f));

        // REQUIRE_THROWS(pm.checkParameters());
        // REQUIRE_THROWS(pn.checkParameters());
        // REQUIRE_THROWS(po.checkParameters());
    }

    SECTION("ghosts 0")
    {
        mgcl::Problem pm(8, 8, 8, v, f);
        pm.setGhosts(0);
        REQUIRE_THROWS(pm.checkParameters());
    }

    SECTION("ghosts_in -1")
    {
        mgcl::Problem pm(8, 8, 8, v, f);
        pm.setGhostsIn(-1);
        REQUIRE_THROWS(pm.checkParameters());
    }

    SECTION("ghosts_in != ghosts of v or f")
    {
        mgcl::Problem pm(8, 8, 8, v, f);
        pm.setGhostsIn(1);
        REQUIRE_THROWS(pm.checkParameters());

        auto v2 = std::make_shared<mgcl::Cuboid>(8, 8, 8, 1, 1, 1);
        mgcl::Problem pm2(8, 8, 8, v2, f);
        pm2.setGhostsIn(0);
        REQUIRE_THROWS(pm2.checkParameters());
    }

    SECTION("all good")
    {
        mgcl::Problem pm(8, 8, 8, v, f);
        REQUIRE_NOTHROW(pm.checkParameters() == true);
    }
}

TEST_CASE("Problem::checkGpuSizes")
{
    auto deviceType = CL_DEVICE_TYPE_GPU;

    if (!mgcl_test::TestUtility::deviceAvailable("", CL_DEVICE_TYPE_GPU))
    {
        deviceType = CL_DEVICE_TYPE_CPU;
        if (!mgcl_test::TestUtility::deviceAvailable("", CL_DEVICE_TYPE_CPU))
        {
            std::cout << "Neither GPU nor CPU OpenCL device available! Skipping test." << std::endl;
            return;
        }
    }

    std::string oclDeviceType = deviceType == CL_DEVICE_TYPE_GPU ? "GPU" : "CPU";

    SECTION("enough space available")
    {
        // 4x4x4 should be no problem
        mgcl::Problem p(4, 4, 4);
        p.setUseOpencl(true);
        p.setDeviceType(deviceType);

        REQUIRE(p.checkGpuSizes());
    }

    SECTION("not enough space available")
    {
        // We can be quiet sure no device can handle such a big grid...
        int N = 32768;

        mgcl::Problem p(N, N, N);
        p.setUseOpencl(true);
        p.setDeviceType(deviceType);

        REQUIRE_THROWS(p.checkGpuSizes());
    }

    SECTION("not enough space available (on Quadro, varying stencil)")
    {
        // Quadro can't handle 128^3 AND varying stencil
        int N = 128;

        if (!mgcl_test::TestUtility::deviceAvailable("Quadro", CL_DEVICE_TYPE_GPU))
        {
            std::cout << "Quadro not available!" << std::endl;
            return;
        }

        mgcl::Problem p(N, N, N);
        p.setUseOpencl(true);
        p.setDeviceType(deviceType);
        p.setStencilType(mgcl::MGCL_VARYING);

        REQUIRE_THROWS(p.checkGpuSizes());
    }
}

// Global dimensions must be a whole multiple of local dims
TEST_CASE("Problem::checkGlobalDimensions")
{
    SECTION("throwing when invalid")
    {
        REQUIRE_THROWS(mgcl::Problem(4, 4, 4, 3, 3, 3));
        REQUIRE_THROWS(mgcl::Problem(4, 4, 4, 2, 2, 2));
    }

    SECTION("not throwing when valid")
    {
        REQUIRE_NOTHROW(mgcl::Problem(4, 4, 4));
        REQUIRE_NOTHROW(mgcl::Problem(2, 2, 2, 4, 4, 4));
        REQUIRE_NOTHROW(mgcl::Problem(1, 1, 1, 4, 4, 4));
        REQUIRE_NOTHROW(mgcl::Problem(1, 1, 1, 1, 1, 1));
    }
}

TEST_CASE("Problem::calculateAndSetMaxLevel")
{
    SECTION("m min")
    {
        mgcl::Problem p(4, 8, 8);

        REQUIRE(p.getMaxlevel() == 2);
        p.calculateAndSetMaxLevel();
        REQUIRE(p.getMaxlevel() == 2);
    }

    SECTION("n min")
    {
        mgcl::Problem p(16, 8, 16);

        REQUIRE(p.getMaxlevel() == 3);
        p.calculateAndSetMaxLevel();
        REQUIRE(p.getMaxlevel() == 3);
    }

    SECTION("o min")
    {
        mgcl::Problem p(32, 32, 16);

        REQUIRE(p.getMaxlevel() == 4);
        p.calculateAndSetMaxLevel();
        REQUIRE(p.getMaxlevel() == 4);
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
        REQUIRE(p.getMaxlevel() == 2);
    }
}

TEST_CASE("Problem::initOpenCL, not reusing environment")
{
    auto deviceType = GENERATE(CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU);

    if (!mgcl_test::TestUtility::deviceAvailable("", deviceType))
    {
        std::string typeName = deviceType == CL_DEVICE_TYPE_GPU ? "CL_DEVICE_TYPE_GPU" : "CL_DEVICE_TYPE_CPU";
        std::cout << "Skipping non-available device type '" << typeName << "'" << std::endl;
        return;
    }

    mgcl::Problem p(4, 4, 4);
    REQUIRE(!p.getOpenCLHelper().isInitialized());
    REQUIRE(p.getUseOpencl() == false);

    p.setUseOpencl(true);
    p.setDeviceType(deviceType);
    p.initOpenCL();
    REQUIRE(p.getUseOpencl() == true);
    REQUIRE(p.getOpenCLHelper().isInitialized());
    REQUIRE(p.getCommands() != nullptr);
    REQUIRE(p.getContext() != nullptr);
    REQUIRE(p.getDeviceId() != nullptr);
}

TEST_CASE("Problem::initOpenCL, reusing environment")
{
    auto deviceType = GENERATE(CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU);

    if (!mgcl_test::TestUtility::deviceAvailable("", deviceType))
    {
        std::string typeName = deviceType == CL_DEVICE_TYPE_GPU ? "CL_DEVICE_TYPE_GPU" : "CL_DEVICE_TYPE_CPU";
        std::cout << "Skipping non-available device type '" << typeName << "'" << std::endl;
        return;
    }

    mgcl::Problem p(4, 4, 4);
    REQUIRE(!p.getOpenCLHelper().isInitialized());
    REQUIRE(p.getUseOpencl() == false);

    mgcl::OpenCLHelper openCLHelper(&p);
    openCLHelper.setDeviceType(deviceType);
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
        p.setDeviceType(CL_DEVICE_TYPE_GPU);
        p.initOpenCL();
        REQUIRE(p.getOpenCLHelper().isInitialized());

        p.setDeviceType(CL_DEVICE_TYPE_CPU);
        p.setKernelDir("test/");
        p.setDeviceName("Quadro");

        REQUIRE(p.getDeviceType() == CL_DEVICE_TYPE_GPU);
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
    // TODO test galerkin
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
        REQUIRE_THROWS(p2.init());
    }

    /**
     * @brief Should create level buffers on host only.
     *
     */
    SECTION("default conf")
    {
        REQUIRE(p.init());
        REQUIRE(p.getLevelsSize() == p.getMaxlevel() + 1);

        // check that default stencil is 7p Laplace stencil.
        REQUIRE(p.getStencilType() == mgcl::MGCL_LAPLACE_7POINT);

        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < o; k++)
                {
                    REQUIRE((*v)[i][j][k] == p.getLevelAt(0).getV()[i + 1][j + 1][k + 1]);
                    REQUIRE((*f)[i][j][k] == p.getLevelAt(0).getF()[i + 1][j + 1][k + 1]);
                }

        for (int lv = 0; lv <= p.getMaxlevel(); lv++)
        {
            std::cout << "Checking level " << lv << std::endl;

            REQUIRE(p.getLevelAt(lv).getM() == m >> lv);
            REQUIRE(p.getLevelAt(lv).getN() == n >> lv);
            REQUIRE(p.getLevelAt(lv).getO() == o >> lv);

            REQUIRE(p.getLevelAt(lv).getMgh() == p.getLevelAt(lv).getM() + 2 * p.getGhosts());
            REQUIRE(p.getLevelAt(lv).getNgh() == p.getLevelAt(lv).getN() + 2 * p.getGhosts());
            REQUIRE(p.getLevelAt(lv).getOgh() == p.getLevelAt(lv).getO() + 2 * p.getGhosts());

            REQUIRE(p.getLevelAt(lv).getRPtr());

            // check if cuboids have correct dimensions
            CHECK(p.getLevelAt(lv).getV().getM() == p.getLevelAt(lv).getM());
            CHECK(p.getLevelAt(lv).getV().getN() == p.getLevelAt(lv).getN());
            CHECK(p.getLevelAt(lv).getV().getO() == p.getLevelAt(lv).getO());
            CHECK(p.getLevelAt(lv).getV().getMgh() == p.getLevelAt(lv).getMgh());
            CHECK(p.getLevelAt(lv).getV().getNgh() == p.getLevelAt(lv).getNgh());
            CHECK(p.getLevelAt(lv).getV().getOgh() == p.getLevelAt(lv).getOgh());
            CHECK(p.getLevelAt(lv).getV().getGhostsM() == p.getGhosts());
            CHECK(p.getLevelAt(lv).getV().getGhostsN() == p.getGhosts());
            CHECK(p.getLevelAt(lv).getV().getGhostsO() == p.getGhosts());

            CHECK(p.getLevelAt(lv).getF().getM() == p.getLevelAt(lv).getM());
            CHECK(p.getLevelAt(lv).getF().getN() == p.getLevelAt(lv).getN());
            CHECK(p.getLevelAt(lv).getF().getO() == p.getLevelAt(lv).getO());
            CHECK(p.getLevelAt(lv).getF().getMgh() == p.getLevelAt(lv).getMgh());
            CHECK(p.getLevelAt(lv).getF().getNgh() == p.getLevelAt(lv).getNgh());
            CHECK(p.getLevelAt(lv).getF().getOgh() == p.getLevelAt(lv).getOgh());
            CHECK(p.getLevelAt(lv).getF().getGhostsM() == p.getGhosts());
            CHECK(p.getLevelAt(lv).getF().getGhostsN() == p.getGhosts());
            CHECK(p.getLevelAt(lv).getF().getGhostsO() == p.getGhosts());

            CHECK(p.getLevelAt(lv).getR().getM() == p.getLevelAt(lv).getM());
            CHECK(p.getLevelAt(lv).getR().getN() == p.getLevelAt(lv).getN());
            CHECK(p.getLevelAt(lv).getR().getO() == p.getLevelAt(lv).getO());
            CHECK(p.getLevelAt(lv).getR().getMgh() == p.getLevelAt(lv).getMgh());
            CHECK(p.getLevelAt(lv).getR().getNgh() == p.getLevelAt(lv).getNgh());
            CHECK(p.getLevelAt(lv).getR().getOgh() == p.getLevelAt(lv).getOgh());
            CHECK(p.getLevelAt(lv).getR().getGhostsM() == p.getGhosts());
            CHECK(p.getLevelAt(lv).getR().getGhostsN() == p.getGhosts());
            CHECK(p.getLevelAt(lv).getR().getGhostsO() == p.getGhosts());

            // TODO check h when used
        }
    }

    SECTION("setting ghosts_in")
    {
        // run this section for ghosts_in = 0, 1 and 2
        auto ghosts_in = GENERATE(0, 1, 2);

        auto v2 = std::make_shared<mgcl::Cuboid>(m, n, o, ghosts_in, ghosts_in, ghosts_in);
        auto f2 = std::make_shared<mgcl::Cuboid>(m, n, o, ghosts_in, ghosts_in, ghosts_in);
        mgcl::Problem p2(m, n, o, v2, f2);

        p2.setGhostsIn(ghosts_in);
        REQUIRE(p2.init());
        REQUIRE(p2.getLevelsSize() == p2.getMaxlevel() + 1);

        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < o; k++)
                {
                    REQUIRE((*v2)[i + ghosts_in][j + ghosts_in][k + ghosts_in] == p2.getLevelAt(0).getV()[i + 1][j + 1][k + 1]);
                    REQUIRE((*f2)[i + ghosts_in][j + ghosts_in][k + ghosts_in] == p2.getLevelAt(0).getF()[i + 1][j + 1][k + 1]);
                }

        for (int lv = 0; lv <= p2.getMaxlevel(); lv++)
        {
            REQUIRE(p2.getLevelAt(lv).getM() == m >> lv);
            REQUIRE(p2.getLevelAt(lv).getN() == n >> lv);
            REQUIRE(p2.getLevelAt(lv).getO() == o >> lv);

            REQUIRE(p2.getLevelAt(lv).getMgh() == p2.getLevelAt(lv).getM() + 2 * p2.getGhosts());
            REQUIRE(p2.getLevelAt(lv).getNgh() == p2.getLevelAt(lv).getN() + 2 * p2.getGhosts());
            REQUIRE(p2.getLevelAt(lv).getOgh() == p2.getLevelAt(lv).getO() + 2 * p2.getGhosts());

            REQUIRE(p2.getLevelAt(lv).getRPtr());

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
        REQUIRE(p.getLevelsSize() == p.getMaxlevel() + 1);

        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < o; k++)
                {
                    REQUIRE((*v)[i][j][k] == p.getLevelAt(0).getV()[i + ghosts][j + ghosts][k + ghosts]);
                    REQUIRE((*f)[i][j][k] == p.getLevelAt(0).getF()[i + ghosts][j + ghosts][k + ghosts]);
                }

        for (int lv = 0; lv <= p.getMaxlevel(); lv++)
        {
            REQUIRE(p.getLevelAt(lv).getM() == m >> lv);
            REQUIRE(p.getLevelAt(lv).getN() == n >> lv);
            REQUIRE(p.getLevelAt(lv).getO() == o >> lv);

            REQUIRE(p.getLevelAt(lv).getMgh() == p.getLevelAt(lv).getM() + 2 * p.getGhosts());
            REQUIRE(p.getLevelAt(lv).getNgh() == p.getLevelAt(lv).getN() + 2 * p.getGhosts());
            REQUIRE(p.getLevelAt(lv).getOgh() == p.getLevelAt(lv).getO() + 2 * p.getGhosts());

            REQUIRE(p.getLevelAt(lv).getRPtr());

            // TODO check h when used
        }
    }

    SECTION("reuse OpenCL buffers")
    {
        auto deviceType = GENERATE(CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU);

        if (!mgcl_test::TestUtility::deviceAvailable("", deviceType))
        {
            std::string typeName = deviceType == CL_DEVICE_TYPE_GPU ? "CL_DEVICE_TYPE_GPU" : "CL_DEVICE_TYPE_CPU";
            std::cout << "Skipping non-available device type '" << typeName << "'" << std::endl;
            return;
        }

        int ghosts = 1;
        mgcl::Cuboid vgh(m + 2 * ghosts, n + 2 * ghosts, o + 2 * ghosts);
        mgcl::Cuboid fgh(m + 2 * ghosts, n + 2 * ghosts, o + 2 * ghosts);

        mgcl_test::TestUtility tu(deviceType);
        cl_mem d_v = tu.createOpenCLBuffer(vgh);
        cl_mem d_f = tu.createOpenCLBuffer(fgh);

        mgcl::Problem p2(m, n, o, d_f, d_v);
        p2.setGhostsIn(ghosts);
        p2.setReuseOpenclBuffers(true);
        p2.getOpenCLHelper().setKernelDir("./");
        REQUIRE(p2.getDF() == d_f);
        REQUIRE(p2.getDV() == d_v);
        REQUIRE(p2.getOpenCLHelper().getProblem() == &p2);

        REQUIRE(p2.reuseOpenCL(tu.getContext(), tu.getCommands(), tu.getDeviceId()) == CL_SUCCESS);
        REQUIRE(p2.init());
        REQUIRE(p2.getDeviceType() == deviceType);
        REQUIRE(p2.getLevelsSize() == p2.getMaxlevel() + 1);

        // no host data is created
        REQUIRE(!p2.getVPtr());
        REQUIRE(!p2.getFPtr());
        for (int lv = 0; lv <= p2.getMaxlevel(); lv++)
        {
            REQUIRE(!p2.getLevelAt(lv).getVPtr());
            REQUIRE(!p2.getLevelAt(lv).getFPtr());
            REQUIRE(!p2.getLevelAt(lv).getRPtr());
        }

        // buffers are created
        REQUIRE(p2.getDV() == d_v);
        REQUIRE(p2.getDF() == d_f);
        REQUIRE(p2.getLevelAt(0).getDVIn() == d_v);
        REQUIRE(p2.getLevelAt(0).getDF() == d_f);
        for (int lv = 0; lv <= p2.getMaxlevel(); lv++)
        {
            // check if buffers are not nullptr
            REQUIRE(p2.getLevelAt(lv).getDVIn());
            REQUIRE(p2.getLevelAt(lv).getDVOut());
            REQUIRE(p2.getLevelAt(lv).getDF());
            REQUIRE(p2.getLevelAt(lv).getDR());

            // check sizes of buffers
            int err;
            size_t bufsize;
            int sizeNeeded = sizeof(double) * p2.getLevelAt(lv).getMgh() * p2.getLevelAt(lv).getNgh() * p2.getLevelAt(lv).getOgh();

            err = clGetMemObjectInfo(p2.getLevelAt(lv).getDVIn(), CL_MEM_SIZE, sizeof(size_t), &bufsize, nullptr);
            mgcl::mgclCheckError(err, "clGetMemObjectInfo");
            REQUIRE(sizeNeeded == bufsize);

            err = clGetMemObjectInfo(p2.getLevelAt(lv).getDVOut(), CL_MEM_SIZE, sizeof(size_t), &bufsize, nullptr);
            mgcl::mgclCheckError(err, "clGetMemObjectInfo");
            REQUIRE(sizeNeeded == bufsize);

            err = clGetMemObjectInfo(p2.getLevelAt(lv).getDF(), CL_MEM_SIZE, sizeof(size_t), &bufsize, nullptr);
            mgcl::mgclCheckError(err, "clGetMemObjectInfo");
            REQUIRE(sizeNeeded == bufsize);

            err = clGetMemObjectInfo(p2.getLevelAt(lv).getDR(), CL_MEM_SIZE, sizeof(size_t), &bufsize, nullptr);
            mgcl::mgclCheckError(err, "clGetMemObjectInfo");
            REQUIRE(sizeNeeded == bufsize);
        }
    }

    SECTION("copy OpenCL buffers")
    {
        auto deviceType = GENERATE(CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU);

        if (!mgcl_test::TestUtility::deviceAvailable("", deviceType))
        {
            std::string typeName = deviceType == CL_DEVICE_TYPE_GPU ? "CL_DEVICE_TYPE_GPU" : "CL_DEVICE_TYPE_CPU";
            std::cout << "Skipping non-available device type '" << typeName << "'" << std::endl;
            return;
        }

        int ghosts = 1;
        mgcl::Cuboid vgh(m + 2 * ghosts, n + 2 * ghosts, o + 2 * ghosts);
        mgcl::Cuboid fgh(m + 2 * ghosts, n + 2 * ghosts, o + 2 * ghosts);

        mgcl_test::TestUtility tu(deviceType);
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
        REQUIRE(p2.getDeviceType() == deviceType);
        REQUIRE(p2.getLevelsSize() == p2.getMaxlevel() + 1);

        // no host data is created
        REQUIRE(!p2.getVPtr());
        REQUIRE(!p2.getFPtr());
        for (int lv = 0; lv <= p2.getMaxlevel(); lv++)
        {
            REQUIRE(!p2.getLevelAt(lv).getVPtr());
            REQUIRE(!p2.getLevelAt(lv).getFPtr());
            REQUIRE(!p2.getLevelAt(lv).getRPtr());
        }

        // buffers are created
        REQUIRE(p2.getDV() == d_v);
        REQUIRE(p2.getDF() == d_f);
        REQUIRE(p2.getLevelAt(0).getDVIn() != d_v);
        REQUIRE(p2.getLevelAt(0).getDF() != d_f);
        for (int lv = 0; lv <= p2.getMaxlevel(); lv++)
        {
            REQUIRE(p2.getLevelAt(lv).getDVIn());
            REQUIRE(p2.getLevelAt(lv).getDVOut());
            REQUIRE(p2.getLevelAt(lv).getDF());
            REQUIRE(p2.getLevelAt(lv).getDR());

            // check sizes of buffers
            int err;
            size_t bufsize;
            int sizeNeeded = sizeof(double) * p2.getLevelAt(lv).getMgh() * p2.getLevelAt(lv).getNgh() * p2.getLevelAt(lv).getOgh();

            err = clGetMemObjectInfo(p2.getLevelAt(lv).getDVIn(), CL_MEM_SIZE, sizeof(size_t), &bufsize, nullptr);
            mgcl::mgclCheckError(err, "clGetMemObjectInfo");
            REQUIRE(sizeNeeded == bufsize);

            err = clGetMemObjectInfo(p2.getLevelAt(lv).getDVOut(), CL_MEM_SIZE, sizeof(size_t), &bufsize, nullptr);
            mgcl::mgclCheckError(err, "clGetMemObjectInfo");
            REQUIRE(sizeNeeded == bufsize);

            err = clGetMemObjectInfo(p2.getLevelAt(lv).getDF(), CL_MEM_SIZE, sizeof(size_t), &bufsize, nullptr);
            mgcl::mgclCheckError(err, "clGetMemObjectInfo");
            REQUIRE(sizeNeeded == bufsize);

            err = clGetMemObjectInfo(p2.getLevelAt(lv).getDR(), CL_MEM_SIZE, sizeof(size_t), &bufsize, nullptr);
            mgcl::mgclCheckError(err, "clGetMemObjectInfo");
            REQUIRE(sizeNeeded == bufsize);
        }

        // contents of copied buffer and input buffers are equal
        auto lv0d = tu.readOpenCLBuffer(p2.getLevelAt(0).getDVIn(), p2.getLevelAt(0).getMgh(), p2.getLevelAt(0).getNgh(), p2.getLevelAt(0).getOgh());
        auto lv0f = tu.readOpenCLBuffer(p2.getLevelAt(0).getDF(), p2.getLevelAt(0).getMgh(), p2.getLevelAt(0).getNgh(), p2.getLevelAt(0).getOgh());

        REQUIRE(vgh.isEqual(*lv0d));
        REQUIRE(fgh.isEqual(*lv0f));
    }

    SECTION("galerkin")
    {
        // Checks if varying stencils are properly set for each level. Does not check actual values for validity, this
        // is done in an own galerkin test.

        p.setStencilType(mgcl::MGCL_VARYING);
        auto& s = *p.getStencilValues();

        REQUIRE(s.getDim1() == m);
        REQUIRE(s.getDim2() == n);
        REQUIRE(s.getDim3() == o);
        REQUIRE(s.getDim4() == 3);
        REQUIRE(s.getDim5() == 3);
        REQUIRE(s.getDim6() == 3);

        // fill with 7-point Laplace
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < o; k++)
                {
                    // 7-point Laplace
                    s[i][j][k][0][1][1] = 1;
                    s[i][j][k][1][0][1] = 1;
                    s[i][j][k][1][1][0] = 1;
                    s[i][j][k][1][1][1] = -6;
                    s[i][j][k][1][1][2] = 1;
                    s[i][j][k][1][2][1] = 1;
                    s[i][j][k][2][1][1] = 1;
                }

        p.init();

        // stencilValues defined in Problem is copied to level 0
        REQUIRE(p.getStencilValues().get() == &s);
        REQUIRE(p.getStencilValues().get() == p.getLevelAt(0).getStencilValues().get());

        for (int lv = 0; lv <= p.getMaxlevel(); lv++)
        {
            auto& level = p.getLevelAt(lv);

            REQUIRE(level.getStencilValues());
            auto& sv = *level.getStencilValues();

            CHECK(sv.getDim1() == level.getM());
            CHECK(sv.getDim2() == level.getN());
            CHECK(sv.getDim3() == level.getO());
            CHECK(sv.getDim4() == 3);
            CHECK(sv.getDim5() == 3);
            CHECK(sv.getDim6() == 3);
        }
    }
}

TEST_CASE("Problem::readResults")
{
    auto deviceType = GENERATE(CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU);

    if (!mgcl_test::TestUtility::deviceAvailable("", deviceType))
    {
        std::string typeName = deviceType == CL_DEVICE_TYPE_GPU ? "CL_DEVICE_TYPE_GPU" : "CL_DEVICE_TYPE_CPU";
        std::cout << "Skipping non-available device type '" << typeName << "'" << std::endl;
        return;
    }

    auto v = std::make_shared<mgcl::Cuboid>(4, 5, 6, 2, 2, 2);
    auto f = std::make_shared<mgcl::Cuboid>(4, 5, 6, 2, 2, 2);
    v->fillRandom();
    f->fillRandom();

    mgcl::Problem p(4, 5, 6, f, v);
    p.setGhostsIn(2);
    p.setUseOpencl(true);
    p.setDeviceType(deviceType);

    SECTION("default conf")
    {
        p.init();
        REQUIRE(p.getVPtr() == v);

        // alter values of dVIn on lowest level
        cl_double one = 1.0;
        int err = clEnqueueFillBuffer(p.getOpenCLHelper().getCommands(), p.getLevelAt(0).getDVIn(), &one,
                                      sizeof(cl_double), 0,
                                      sizeof(double) * p.getLevelAt(0).getMgh() * p.getLevelAt(0).getNgh() * p.getLevelAt(0).getOgh(),
                                      0, NULL, NULL);
        mgcl::mgclCheckError(err, "clEnqueueFillBuffer");
        REQUIRE(err == CL_SUCCESS);

        // read back and check if values were copied successfully
        REQUIRE(p.readResults() == CL_SUCCESS);
        REQUIRE(p.getVPtr() == v);
        for (int i = 0; i < v->getM(); i++)
            for (int j = 0; j < v->getN(); j++)
                for (int k = 0; k < v->getO(); k++)
                {
                    CHECK((*v)[i + v->getGhostsM()][j + v->getGhostsN()][k + v->getGhostsO()] == 1);
                }
    }
}

/**
 * @brief If stencilType is changed to a varying stencil, stencilValues should be created accordingly.
 *
 */
TEST_CASE("Problem::setStencilType")
{
    mgcl::Problem p(2, 2, 2);
    CHECK(p.getStencilValues() == nullptr);

    p.setStencilType(mgcl::MGCL_LAPLACE_19POINT);
    CHECK(p.getStencilValues() == nullptr);

    p.setStencilType(mgcl::MGCL_LAPLACE_27POINT);
    CHECK(p.getStencilValues() == nullptr);

    p.setStencilType(mgcl::MGCL_LAPLACE_7POINT);
    CHECK(p.getStencilValues() == nullptr);

    p.setStencilType(mgcl::MGCL_VARYING);
    REQUIRE(p.getStencilValues() != nullptr);
    CHECK(p.getStencilValues()->getDim1() == p.getM());
    CHECK(p.getStencilValues()->getDim2() == p.getN());
    CHECK(p.getStencilValues()->getDim3() == p.getO());
    CHECK(p.getStencilValues()->getDim1gh() == p.getM() + 4);
    CHECK(p.getStencilValues()->getDim2gh() == p.getN() + 4);
    CHECK(p.getStencilValues()->getDim3gh() == p.getO() + 4);

    p.setStencilType(mgcl::MGCL_VARYING);
    REQUIRE(p.getStencilValues() != nullptr);
    CHECK(p.getStencilValues()->getDim1() == p.getM());
    CHECK(p.getStencilValues()->getDim2() == p.getN());
    CHECK(p.getStencilValues()->getDim3() == p.getO());
    CHECK(p.getStencilValues()->getDim1gh() == p.getM() + 4);
    CHECK(p.getStencilValues()->getDim2gh() == p.getN() + 4);
    CHECK(p.getStencilValues()->getDim3gh() == p.getO() + 4);

    p.setStencilType(mgcl::MGCL_VARYING);
    REQUIRE(p.getStencilValues() != nullptr);
    CHECK(p.getStencilValues()->getDim1() == p.getM());
    CHECK(p.getStencilValues()->getDim2() == p.getN());
    CHECK(p.getStencilValues()->getDim3() == p.getO());
    CHECK(p.getStencilValues()->getDim1gh() == p.getM() + 4);
    CHECK(p.getStencilValues()->getDim2gh() == p.getN() + 4);
    CHECK(p.getStencilValues()->getDim3gh() == p.getO() + 4);
}
