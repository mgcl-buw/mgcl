#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <iostream>
#include <memory>

#include "../../src/mgcl/cuboid_bs.hpp"
#include "../../src/mgcl/level.hpp"
#include "../../src/mgcl/opencl_helper.hpp"
#include "../../src/mgcl/problem.hpp"
#include "../../src/mgcl/stencil.hpp"
#include "../cli_args.hpp"
#include "../device_type_generator.hpp"
#include "../test_utility.hpp"

/**
 * @brief Tests if config parameters of Problem are working correctly, including setting things up.
 *
 */
TEST_CASE("Problem conf Blockstencil")
{
    int blocksize = 2;
    auto v = std::make_shared<mgcl::CuboidBS>(8, 8, 8, blocksize);
    auto f = std::make_shared<mgcl::CuboidBS>(8, 8, 8, blocksize);
    mgcl::Problem p(8, 8, 8, v, f);

    SECTION("default values")
    {
        // REQUIRE(p.getVBS() == v);
        // REQUIRE(p.getFBS() == f);
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
        REQUIRE(p.getStencilType() == mgcl::MGCL_BLOCKSTENCIL);

        REQUIRE(!p.getOpenCLHelper().isInitialized());
        REQUIRE(p.getDeviceType() == CL_DEVICE_TYPE_DEFAULT);
        REQUIRE(p.getKernelFile() == "./mgcl.cl");
        REQUIRE(p.getDeviceName() == "");
        REQUIRE(p.getDVPtr() == nullptr);
        REQUIRE(p.getDFPtr() == nullptr);
        REQUIRE(p.getUseOpencl() == false);
        REQUIRE(p.getReuseOpenclBuffers() == false);
        REQUIRE(p.getCopyBufferData() == false);
        REQUIRE(p.getReadResults() == false);

        REQUIRE(p.getJacobiIterationsPerKernel() == 1);
    }
}

TEST_CASE("Problem::checkParametersBlockstencil")
{
    int blocksize = 2;
    auto v = std::make_shared<mgcl::CuboidBS>(8, 8, 8, blocksize);
    auto f = std::make_shared<mgcl::CuboidBS>(8, 8, 8, blocksize);

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

    SECTION("stencilType wrong")
    {
        mgcl::Problem pm(8, 8, 8, v, f);
        pm.setStencilType(mgcl::MGCL_LAPLACE_7POINT);
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

        auto v2 = std::make_shared<mgcl::CuboidBS>(8, 8, 8, 1, 1, 1, blocksize);
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

/**
 * @brief Should set maxlevel, create buffers and set h for each level
 *
 */
TEST_CASE("Problem::initBlockstencil")
{
    int m = 4;
    int n = 4;
    int o = 4;
    int blocksize = 2;
    auto v = std::make_shared<mgcl::CuboidBS>(m, n, o, blocksize);
    auto f = std::make_shared<mgcl::CuboidBS>(m, n, o, blocksize);
    mgcl::Problem p(m, n, o, f, v);
    v->fillRandom();
    f->fillRandom();

    auto r_fs_tmp = mgcl::create3dFullWeightRestrictionStencil();
    auto p_fs_tmp = mgcl::create3dBilinearProlongationStencil();
    auto& rbs = *p.getRestrictionBlockstencil();
    auto& pbs = *p.getProlongationBlockstencil();
    for (size_t bi = 0; bi < blocksize; bi++)
        for (size_t ii = 0; ii < 3; ii++)
            for (size_t jj = 0; jj < 3; jj++)
                for (size_t kk = 0; kk < 3; kk++)
                {
                    rbs[bi][bi][ii][jj][kk] = r_fs_tmp[ii][jj][kk];
                    pbs[bi][bi][ii][jj][kk] = p_fs_tmp[ii][jj][kk];
                }

    auto& s = *p.getBlockstencil();
    REQUIRE(s.getM() == m);
    REQUIRE(s.getN() == n);
    REQUIRE(s.getO() == o);
    REQUIRE(s.getWidth() == 3);
    REQUIRE(s.getBlocksize() == blocksize);

    mgcl_test::fill7pLaplace(s, 1.0 / (double)m, false);

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

        // check that default stencil is Blockstencil
        REQUIRE(p.getStencilType() == mgcl::MGCL_BLOCKSTENCIL);
        REQUIRE(p.getLevelAt(0).getVBSPtr());
        REQUIRE(p.getLevelAt(0).getFBSPtr());
        REQUIRE(p.getLevelAt(0).getRBSPtr());

        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < o; k++)
                    for (size_t b = 0; b < blocksize; b++)
                    {
                        REQUIRE((*v)[b][i][j][k] == p.getLevelAt(0).getVBS()[b][i + 1][j + 1][k + 1]);
                        REQUIRE((*f)[b][i][j][k] == p.getLevelAt(0).getFBS()[b][i + 1][j + 1][k + 1]);
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

            REQUIRE(p.getLevelAt(lv).getRBSPtr());
            REQUIRE(p.getLevelAt(lv).getVBSPtr());
            REQUIRE(p.getLevelAt(lv).getFBSPtr());

            // check if cuboids have correct dimensions
            CHECK(p.getLevelAt(lv).getVBS().getM() == p.getLevelAt(lv).getM());
            CHECK(p.getLevelAt(lv).getVBS().getN() == p.getLevelAt(lv).getN());
            CHECK(p.getLevelAt(lv).getVBS().getO() == p.getLevelAt(lv).getO());
            CHECK(p.getLevelAt(lv).getVBS().getMgh() == p.getLevelAt(lv).getMgh());
            CHECK(p.getLevelAt(lv).getVBS().getNgh() == p.getLevelAt(lv).getNgh());
            CHECK(p.getLevelAt(lv).getVBS().getOgh() == p.getLevelAt(lv).getOgh());
            CHECK(p.getLevelAt(lv).getVBS().getGhostsM() == p.getGhosts());
            CHECK(p.getLevelAt(lv).getVBS().getGhostsN() == p.getGhosts());
            CHECK(p.getLevelAt(lv).getVBS().getGhostsO() == p.getGhosts());

            CHECK(p.getLevelAt(lv).getFBS().getM() == p.getLevelAt(lv).getM());
            CHECK(p.getLevelAt(lv).getFBS().getN() == p.getLevelAt(lv).getN());
            CHECK(p.getLevelAt(lv).getFBS().getO() == p.getLevelAt(lv).getO());
            CHECK(p.getLevelAt(lv).getFBS().getMgh() == p.getLevelAt(lv).getMgh());
            CHECK(p.getLevelAt(lv).getFBS().getNgh() == p.getLevelAt(lv).getNgh());
            CHECK(p.getLevelAt(lv).getFBS().getOgh() == p.getLevelAt(lv).getOgh());
            CHECK(p.getLevelAt(lv).getFBS().getGhostsM() == p.getGhosts());
            CHECK(p.getLevelAt(lv).getFBS().getGhostsN() == p.getGhosts());
            CHECK(p.getLevelAt(lv).getFBS().getGhostsO() == p.getGhosts());

            CHECK(p.getLevelAt(lv).getRBS().getM() == p.getLevelAt(lv).getM());
            CHECK(p.getLevelAt(lv).getRBS().getN() == p.getLevelAt(lv).getN());
            CHECK(p.getLevelAt(lv).getRBS().getO() == p.getLevelAt(lv).getO());
            CHECK(p.getLevelAt(lv).getRBS().getMgh() == p.getLevelAt(lv).getMgh());
            CHECK(p.getLevelAt(lv).getRBS().getNgh() == p.getLevelAt(lv).getNgh());
            CHECK(p.getLevelAt(lv).getRBS().getOgh() == p.getLevelAt(lv).getOgh());
            CHECK(p.getLevelAt(lv).getRBS().getGhostsM() == p.getGhosts());
            CHECK(p.getLevelAt(lv).getRBS().getGhostsN() == p.getGhosts());
            CHECK(p.getLevelAt(lv).getRBS().getGhostsO() == p.getGhosts());

            // TODO check h when used
        }
    }

    SECTION("setting ghosts_in")
    {
        // run this section for ghosts_in = 0, 1 and 2
        auto ghosts_in = GENERATE(0, 1, 2);
        CAPTURE(ghosts_in);

        auto v2 = std::make_shared<mgcl::CuboidBS>(m, n, o, ghosts_in, ghosts_in, ghosts_in, blocksize);
        auto f2 = std::make_shared<mgcl::CuboidBS>(m, n, o, ghosts_in, ghosts_in, ghosts_in, blocksize);
        mgcl::Problem p2(m, n, o, v2, f2);

        p2.setGhostsIn(ghosts_in);
        auto& s2 = *p2.getBlockstencil();

        auto& rbs2 = *p2.getRestrictionBlockstencil();
        auto& pbs2 = *p2.getProlongationBlockstencil();
        for (size_t bi = 0; bi < blocksize; bi++)
            for (size_t ii = 0; ii < 3; ii++)
                for (size_t jj = 0; jj < 3; jj++)
                    for (size_t kk = 0; kk < 3; kk++)
                    {
                        rbs2[bi][bi][ii][jj][kk] = r_fs_tmp[ii][jj][kk];
                        pbs2[bi][bi][ii][jj][kk] = p_fs_tmp[ii][jj][kk];
                    }

        mgcl_test::fill7pLaplace(s2, 1.0 / (double)m, false);

        REQUIRE(p2.init());
        REQUIRE(p2.getLevelsSize() == p2.getMaxlevel() + 1);

        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < o; k++)
                    for (size_t b = 0; b < blocksize; b++)
                    {
                        REQUIRE((*v2)[b][i + ghosts_in][j + ghosts_in][k + ghosts_in] == p2.getLevelAt(0).getVBS()[b][i + 1][j + 1][k + 1]);
                        REQUIRE((*f2)[b][i + ghosts_in][j + ghosts_in][k + ghosts_in] == p2.getLevelAt(0).getFBS()[b][i + 1][j + 1][k + 1]);
                    }

        for (int lv = 0; lv <= p2.getMaxlevel(); lv++)
        {
            REQUIRE(p2.getLevelAt(lv).getM() == m >> lv);
            REQUIRE(p2.getLevelAt(lv).getN() == n >> lv);
            REQUIRE(p2.getLevelAt(lv).getO() == o >> lv);

            REQUIRE(p2.getLevelAt(lv).getMgh() == p2.getLevelAt(lv).getM() + 2 * p2.getGhosts());
            REQUIRE(p2.getLevelAt(lv).getNgh() == p2.getLevelAt(lv).getN() + 2 * p2.getGhosts());
            REQUIRE(p2.getLevelAt(lv).getOgh() == p2.getLevelAt(lv).getO() + 2 * p2.getGhosts());

            REQUIRE(p2.getLevelAt(lv).getRBSPtr());

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
        CAPTURE(ghosts);

        p.setGhosts(ghosts);
        p.setStencilType(mgcl::MGCL_BLOCKSTENCIL);
        auto& s = *p.getBlockstencil();
        mgcl_test::fill7pLaplace(s, 1.0 / (double)m, false);

        auto& rbs = *p.getRestrictionBlockstencil();
        auto& pbs = *p.getProlongationBlockstencil();
        for (size_t bi = 0; bi < blocksize; bi++)
            for (size_t ii = 0; ii < 3; ii++)
                for (size_t jj = 0; jj < 3; jj++)
                    for (size_t kk = 0; kk < 3; kk++)
                    {
                        rbs[bi][bi][ii][jj][kk] = r_fs_tmp[ii][jj][kk];
                        pbs[bi][bi][ii][jj][kk] = p_fs_tmp[ii][jj][kk];
                    }

        REQUIRE(p.init());
        REQUIRE(p.getLevelsSize() == p.getMaxlevel() + 1);

        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < o; k++)
                    for (size_t b = 0; b < blocksize; b++)
                    {
                        REQUIRE((*v)[b][i][j][k] == p.getLevelAt(0).getVBS()[b][i + ghosts][j + ghosts][k + ghosts]);
                        REQUIRE((*f)[b][i][j][k] == p.getLevelAt(0).getFBS()[b][i + ghosts][j + ghosts][k + ghosts]);
                    }

        for (int lv = 0; lv <= p.getMaxlevel(); lv++)
        {
            REQUIRE(p.getLevelAt(lv).getM() == m >> lv);
            REQUIRE(p.getLevelAt(lv).getN() == n >> lv);
            REQUIRE(p.getLevelAt(lv).getO() == o >> lv);

            REQUIRE(p.getLevelAt(lv).getMgh() == p.getLevelAt(lv).getM() + 2 * p.getGhosts());
            REQUIRE(p.getLevelAt(lv).getNgh() == p.getLevelAt(lv).getN() + 2 * p.getGhosts());
            REQUIRE(p.getLevelAt(lv).getOgh() == p.getLevelAt(lv).getO() + 2 * p.getGhosts());

            REQUIRE(p.getLevelAt(lv).getRBSPtr());

            // TODO check h when used
        }
    }

    // SECTION("reuse OpenCL buffers")
    // {
    //     auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));

    //     int ghosts = 1;
    //     mgcl::CuboidBS vgh(m + 2 * ghosts, n + 2 * ghosts, o + 2 * ghosts, blocksize);
    //     mgcl::CuboidBS fgh(m + 2 * ghosts, n + 2 * ghosts, o + 2 * ghosts, blocksize);

    //     mgcl_test::TestUtility tu(deviceType);
    //     auto d_v = std::make_shared<mgcl::CuboidBSGpu>(tu.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, vgh);
    //     auto d_f = std::make_shared<mgcl::CuboidBSGpu>(tu.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, fgh);

    //     mgcl::Problem p2(m, n, o, d_f, d_v);
    //     p2.setGhostsIn(ghosts);
    //     p2.setReuseOpenclBuffers(true);
    //     REQUIRE(p2.getDFPtr() == d_f);
    //     REQUIRE(p2.getDVPtr() == d_v);
    //     REQUIRE(p2.getOpenCLHelper().getProblem() == &p2);

    //     REQUIRE_NOTHROW(p2.reuseOpenCL(tu.getContext(), tu.getCommands(), tu.getDeviceId()));
    //     REQUIRE(p2.init());
    //     REQUIRE(p2.getDeviceType() == deviceType);
    //     REQUIRE(p2.getLevelsSize() == p2.getMaxlevel() + 1);

    //     // no host data is created
    //     REQUIRE(!p2.getVPtr());
    //     REQUIRE(!p2.getFPtr());
    //     for (int lv = 0; lv <= p2.getMaxlevel(); lv++)
    //     {
    //         REQUIRE(!p2.getLevelAt(lv).getVPtr());
    //         REQUIRE(!p2.getLevelAt(lv).getFPtr());
    //         REQUIRE(!p2.getLevelAt(lv).getRPtr());
    //     }

    //     // buffers are created
    //     REQUIRE(p2.getDVPtr() == d_v);
    //     REQUIRE(p2.getDFPtr() == d_f);
    //     REQUIRE(p2.getLevelAt(0).getDVInPtr() == d_v.get());
    //     REQUIRE(p2.getLevelAt(0).getDFPtr() == d_f.get());
    //     for (int lv = 0; lv <= p2.getMaxlevel(); lv++)
    //     {
    //         // check if buffers are not nullptr
    //         REQUIRE(p2.getLevelAt(lv).getDVInPtr());
    //         REQUIRE(p2.getLevelAt(lv).getDVOutPtr());
    //         REQUIRE(p2.getLevelAt(lv).getDFPtr());
    //         REQUIRE(p2.getLevelAt(lv).getDRPtr());

    //         // check sizes of buffers
    //         int sizeNeeded = p2.getLevelAt(lv).getMgh() * p2.getLevelAt(lv).getNgh() * p2.getLevelAt(lv).getOgh();
    //         REQUIRE(sizeNeeded == p2.getLevelAt(lv).getDVIn().getSize());
    //         REQUIRE(sizeNeeded == p2.getLevelAt(lv).getDVOut().getSize());
    //         REQUIRE(sizeNeeded == p2.getLevelAt(lv).getDF().getSize());
    //         REQUIRE(sizeNeeded == p2.getLevelAt(lv).getDR().getSize());
    //     }
    // }

    // SECTION("copy OpenCL buffers")
    // {
    //     auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));

    //     int ghosts = 1;
    //     mgcl::CuboidBS vgh(m, n, o, ghosts, ghosts, ghosts);
    //     mgcl::CuboidBS fgh(m, n, o, ghosts, ghosts, ghosts);

    //     mgcl_test::TestUtility tu(deviceType);
    //     auto d_v = std::make_shared<mgcl::CuboidBSGpu>(tu.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, vgh);
    //     auto d_f = std::make_shared<mgcl::CuboidBSGpu>(tu.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, fgh);

    //     mgcl::Problem p2(m, n, o, d_f, d_v);
    //     p2.setGhostsIn(ghosts);
    //     p2.setCopyBufferData(true);
    //     REQUIRE(p2.getDFPtr() == d_f);
    //     REQUIRE(p2.getDVPtr() == d_v);
    //     REQUIRE(p2.getOpenCLHelper().getProblem() == &p2);

    //     REQUIRE_NOTHROW(p2.reuseOpenCL(tu.getContext(), tu.getCommands(), tu.getDeviceId()));
    //     REQUIRE(p2.init());
    //     REQUIRE(p2.getDeviceType() == deviceType);
    //     REQUIRE(p2.getLevelsSize() == p2.getMaxlevel() + 1);

    //     // no host data is created
    //     REQUIRE(!p2.getVPtr());
    //     REQUIRE(!p2.getFPtr());
    //     for (int lv = 0; lv <= p2.getMaxlevel(); lv++)
    //     {
    //         REQUIRE(!p2.getLevelAt(lv).getVPtr());
    //         REQUIRE(!p2.getLevelAt(lv).getFPtr());
    //         REQUIRE(!p2.getLevelAt(lv).getRPtr());
    //     }

    //     // buffers are created
    //     REQUIRE(p2.getDVPtr() == d_v);
    //     REQUIRE(p2.getDFPtr() == d_f);
    //     REQUIRE(p2.getLevelAt(0).getDVInPtr() != d_v.get());
    //     REQUIRE(p2.getLevelAt(0).getDFPtr() != d_f.get());
    //     for (int lv = 0; lv <= p2.getMaxlevel(); lv++)
    //     {
    //         REQUIRE(p2.getLevelAt(lv).getDVInPtr());
    //         REQUIRE(p2.getLevelAt(lv).getDVOutPtr());
    //         REQUIRE(p2.getLevelAt(lv).getDFPtr());
    //         REQUIRE(p2.getLevelAt(lv).getDRPtr());

    //         // check sizes of buffers
    //         int sizeNeeded = p2.getLevelAt(lv).getMgh() * p2.getLevelAt(lv).getNgh() * p2.getLevelAt(lv).getOgh();
    //         REQUIRE(sizeNeeded == p2.getLevelAt(lv).getDVIn().getSize());
    //         REQUIRE(sizeNeeded == p2.getLevelAt(lv).getDVOut().getSize());
    //         REQUIRE(sizeNeeded == p2.getLevelAt(lv).getDF().getSize());
    //         REQUIRE(sizeNeeded == p2.getLevelAt(lv).getDR().getSize());
    //     }

    //     // contents of copied buffer and input buffers are equal
    //     auto lv0d = p2.getLevelAt(0).getDVIn().read(tu.getCommands(), nullptr, true);
    //     auto lv0f = p2.getLevelAt(0).getDF().read(tu.getCommands(), nullptr, true);

    //     REQUIRE(vgh.isEqual(*lv0d));
    //     REQUIRE(fgh.isEqual(*lv0f));
    // }

    SECTION("galerkin")
    {
        // Checks if varying stencils are properly set for each level. Does not check actual values for validity, this
        // is done in an own galerkin test.
        mgcl::MGCL_SMOOTHER smootherType = GENERATE(mgcl::MGCL_JACOBI_SCALAR, mgcl::MGCL_JACOBI_BLOCK);

        p.setSmootherType(smootherType);
        p.init();

        // stencilValues defined in Problem is copied to level 0
        REQUIRE(p.getBlockstencil().get() == &s);
        REQUIRE(p.getBlockstencil().get() == p.getLevelAt(0).getBlockstencil().get());

        for (int lv = 0; lv <= p.getMaxlevel(); lv++)
        {
            CAPTURE(lv);
            auto& level = p.getLevelAt(lv);

            REQUIRE(level.getBlockstencil());
            auto& sv = *level.getBlockstencil();

            REQUIRE(sv.getM() == level.getM());
            REQUIRE(sv.getN() == level.getN());
            REQUIRE(sv.getO() == level.getO());
            REQUIRE(sv.getWidth() == 3);
            REQUIRE(sv.getBlocksize() == blocksize);

            if (smootherType == mgcl::MGCL_JACOBI_BLOCK)
            {
                REQUIRE(level.getBlockstencilInvBlock());
                auto& sv_inv = *level.getBlockstencilInvBlock();

                REQUIRE(sv_inv.getM() == level.getM());
                REQUIRE(sv_inv.getN() == level.getN());
                REQUIRE(sv_inv.getO() == level.getO());
                REQUIRE(sv_inv.getWidth() == 1);
                REQUIRE(sv_inv.getBlocksize() == blocksize);
            }
            else
            {
                REQUIRE(level.getBlockstencilInvScalar());
                auto& sv_inv = *level.getBlockstencilInvScalar();

                REQUIRE(sv_inv.getM() == level.getM());
                REQUIRE(sv_inv.getN() == level.getN());
                REQUIRE(sv_inv.getO() == level.getO());
                REQUIRE(sv_inv.getBlocksize() == blocksize);
            }
        }
    }
}

/**
 * @brief Should set maxlevel, create device buffers and apply galerkin
 *
 */
TEST_CASE("Problem::init_oclBlockstencil")
{
    int m = 4;
    int n = 4;
    int o = 4;
    int blocksize = 2;
    auto v = std::make_shared<mgcl::CuboidBS>(m, n, o, blocksize);
    auto f = std::make_shared<mgcl::CuboidBS>(m, n, o, blocksize);
    mgcl::Problem p(m, n, o, f, v);
    v->fillRandom();
    f->fillRandom();
    p.setUseOpencl(true);
    p.getOpenCLHelper().setPreprocessorConstant("BLOCKSIZE", std::to_string(blocksize));

    auto& s = *p.getBlockstencil();

    REQUIRE(s.getM() == m);
    REQUIRE(s.getN() == n);
    REQUIRE(s.getO() == o);
    REQUIRE(s.getWidth() == 3);
    REQUIRE(s.getBlocksize() == blocksize);

    auto r_fs_tmp = mgcl::create3dFullWeightRestrictionStencil();
    auto p_fs_tmp = mgcl::create3dBilinearProlongationStencil();
    auto& rbs = *p.getRestrictionBlockstencil();
    auto& pbs = *p.getProlongationBlockstencil();
    for (size_t bi = 0; bi < blocksize; bi++)
        for (size_t ii = 0; ii < 3; ii++)
            for (size_t jj = 0; jj < 3; jj++)
                for (size_t kk = 0; kk < 3; kk++)
                {
                    rbs[bi][bi][ii][jj][kk] = r_fs_tmp[ii][jj][kk];
                    pbs[bi][bi][ii][jj][kk] = p_fs_tmp[ii][jj][kk];
                }

    mgcl_test::fill7pLaplace(s, 1.0 / (double)m, false);

    SECTION("default conf")
    {
        REQUIRE(p.init());
        REQUIRE(p.getLevelsSize() == p.getMaxlevel() + 1);

        // check that default stencil is Blockstencil
        REQUIRE(p.getStencilType() == mgcl::MGCL_BLOCKSTENCIL);

        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < o; k++)
                    for (size_t b = 0; b < blocksize; b++)
                    {
                        REQUIRE((*v)[b][i][j][k] == p.getLevelAt(0).getVBS()[b][i + 1][j + 1][k + 1]);
                        REQUIRE((*f)[b][i][j][k] == p.getLevelAt(0).getFBS()[b][i + 1][j + 1][k + 1]);
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

            REQUIRE(p.getLevelAt(lv).getDRBSPtr());

            // check if cuboids have correct dimensions
            CHECK(p.getLevelAt(lv).getDVBSIn().getM() == p.getLevelAt(lv).getM());
            CHECK(p.getLevelAt(lv).getDVBSIn().getN() == p.getLevelAt(lv).getN());
            CHECK(p.getLevelAt(lv).getDVBSIn().getO() == p.getLevelAt(lv).getO());
            CHECK(p.getLevelAt(lv).getDVBSIn().getMgh() == p.getLevelAt(lv).getMgh());
            CHECK(p.getLevelAt(lv).getDVBSIn().getNgh() == p.getLevelAt(lv).getNgh());
            CHECK(p.getLevelAt(lv).getDVBSIn().getOgh() == p.getLevelAt(lv).getOgh());
            CHECK(p.getLevelAt(lv).getDVBSIn().getGhostsM() == p.getGhosts());
            CHECK(p.getLevelAt(lv).getDVBSIn().getGhostsN() == p.getGhosts());
            CHECK(p.getLevelAt(lv).getDVBSIn().getGhostsO() == p.getGhosts());

            CHECK(p.getLevelAt(lv).getDVBSOut().getM() == p.getLevelAt(lv).getM());
            CHECK(p.getLevelAt(lv).getDVBSOut().getN() == p.getLevelAt(lv).getN());
            CHECK(p.getLevelAt(lv).getDVBSOut().getO() == p.getLevelAt(lv).getO());
            CHECK(p.getLevelAt(lv).getDVBSOut().getMgh() == p.getLevelAt(lv).getMgh());
            CHECK(p.getLevelAt(lv).getDVBSOut().getNgh() == p.getLevelAt(lv).getNgh());
            CHECK(p.getLevelAt(lv).getDVBSOut().getOgh() == p.getLevelAt(lv).getOgh());
            CHECK(p.getLevelAt(lv).getDVBSOut().getGhostsM() == p.getGhosts());
            CHECK(p.getLevelAt(lv).getDVBSOut().getGhostsN() == p.getGhosts());
            CHECK(p.getLevelAt(lv).getDVBSOut().getGhostsO() == p.getGhosts());

            CHECK(p.getLevelAt(lv).getDFBS().getM() == p.getLevelAt(lv).getM());
            CHECK(p.getLevelAt(lv).getDFBS().getN() == p.getLevelAt(lv).getN());
            CHECK(p.getLevelAt(lv).getDFBS().getO() == p.getLevelAt(lv).getO());
            CHECK(p.getLevelAt(lv).getDFBS().getMgh() == p.getLevelAt(lv).getMgh());
            CHECK(p.getLevelAt(lv).getDFBS().getNgh() == p.getLevelAt(lv).getNgh());
            CHECK(p.getLevelAt(lv).getDFBS().getOgh() == p.getLevelAt(lv).getOgh());
            CHECK(p.getLevelAt(lv).getDFBS().getGhostsM() == p.getGhosts());
            CHECK(p.getLevelAt(lv).getDFBS().getGhostsN() == p.getGhosts());
            CHECK(p.getLevelAt(lv).getDFBS().getGhostsO() == p.getGhosts());

            CHECK(p.getLevelAt(lv).getDRBS().getM() == p.getLevelAt(lv).getM());
            CHECK(p.getLevelAt(lv).getDRBS().getN() == p.getLevelAt(lv).getN());
            CHECK(p.getLevelAt(lv).getDRBS().getO() == p.getLevelAt(lv).getO());
            CHECK(p.getLevelAt(lv).getDRBS().getMgh() == p.getLevelAt(lv).getMgh());
            CHECK(p.getLevelAt(lv).getDRBS().getNgh() == p.getLevelAt(lv).getNgh());
            CHECK(p.getLevelAt(lv).getDRBS().getOgh() == p.getLevelAt(lv).getOgh());
            CHECK(p.getLevelAt(lv).getDRBS().getGhostsM() == p.getGhosts());
            CHECK(p.getLevelAt(lv).getDRBS().getGhostsN() == p.getGhosts());
            CHECK(p.getLevelAt(lv).getDRBS().getGhostsO() == p.getGhosts());

            // TODO check h when used
        }
    }

    // SECTION("reuse OpenCL buffers")
    // {
    //     auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));

    //     int ghosts = 1;
    //     mgcl::CuboidBS vgh(m + 2 * ghosts, n + 2 * ghosts, o + 2 * ghosts, blocksize);
    //     mgcl::CuboidBS fgh(m + 2 * ghosts, n + 2 * ghosts, o + 2 * ghosts, blocksize);

    //     mgcl_test::TestUtility tu(deviceType);
    //     auto d_v = std::make_shared<mgcl::CuboidBSGpu>(tu.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, vgh);
    //     auto d_f = std::make_shared<mgcl::CuboidBSGpu>(tu.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, fgh);

    //     mgcl::Problem p2(m, n, o, d_f, d_v);
    //     p2.setGhostsIn(ghosts);
    //     p2.setReuseOpenclBuffers(true);
    //     REQUIRE(p2.getDFPtr() == d_f);
    //     REQUIRE(p2.getDVPtr() == d_v);
    //     REQUIRE(p2.getOpenCLHelper().getProblem() == &p2);

    //     REQUIRE_NOTHROW(p2.reuseOpenCL(tu.getContext(), tu.getCommands(), tu.getDeviceId()));
    //     REQUIRE(p2.init());
    //     REQUIRE(p2.getDeviceType() == deviceType);
    //     REQUIRE(p2.getLevelsSize() == p2.getMaxlevel() + 1);

    //     // no host data is created
    //     REQUIRE(!p2.getVPtr());
    //     REQUIRE(!p2.getFPtr());
    //     for (int lv = 0; lv <= p2.getMaxlevel(); lv++)
    //     {
    //         REQUIRE(!p2.getLevelAt(lv).getVPtr());
    //         REQUIRE(!p2.getLevelAt(lv).getFPtr());
    //         REQUIRE(!p2.getLevelAt(lv).getRPtr());
    //     }

    //     // buffers are created
    //     REQUIRE(p2.getDVPtr() == d_v);
    //     REQUIRE(p2.getDFPtr() == d_f);
    //     REQUIRE(p2.getLevelAt(0).getDVInPtr() == d_v.get());
    //     REQUIRE(p2.getLevelAt(0).getDFPtr() == d_f.get());
    //     for (int lv = 0; lv <= p2.getMaxlevel(); lv++)
    //     {
    //         // check if buffers are not nullptr
    //         REQUIRE(p2.getLevelAt(lv).getDVInPtr());
    //         REQUIRE(p2.getLevelAt(lv).getDVOutPtr());
    //         REQUIRE(p2.getLevelAt(lv).getDFPtr());
    //         REQUIRE(p2.getLevelAt(lv).getDRPtr());

    //         // check sizes of buffers
    //         int sizeNeeded = p2.getLevelAt(lv).getMgh() * p2.getLevelAt(lv).getNgh() * p2.getLevelAt(lv).getOgh();
    //         REQUIRE(sizeNeeded == p2.getLevelAt(lv).getDVIn().getSize());
    //         REQUIRE(sizeNeeded == p2.getLevelAt(lv).getDVOut().getSize());
    //         REQUIRE(sizeNeeded == p2.getLevelAt(lv).getDF().getSize());
    //         REQUIRE(sizeNeeded == p2.getLevelAt(lv).getDR().getSize());
    //     }
    // }

    // SECTION("copy OpenCL buffers")
    // {
    //     auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));

    //     int ghosts = 1;
    //     mgcl::CuboidBS vgh(m, n, o, ghosts, ghosts, ghosts);
    //     mgcl::CuboidBS fgh(m, n, o, ghosts, ghosts, ghosts);

    //     mgcl_test::TestUtility tu(deviceType);
    //     auto d_v = std::make_shared<mgcl::CuboidBSGpu>(tu.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, vgh);
    //     auto d_f = std::make_shared<mgcl::CuboidBSGpu>(tu.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, fgh);

    //     mgcl::Problem p2(m, n, o, d_f, d_v);
    //     p2.setGhostsIn(ghosts);
    //     p2.setCopyBufferData(true);
    //     REQUIRE(p2.getDFPtr() == d_f);
    //     REQUIRE(p2.getDVPtr() == d_v);
    //     REQUIRE(p2.getOpenCLHelper().getProblem() == &p2);

    //     REQUIRE_NOTHROW(p2.reuseOpenCL(tu.getContext(), tu.getCommands(), tu.getDeviceId()));
    //     REQUIRE(p2.init());
    //     REQUIRE(p2.getDeviceType() == deviceType);
    //     REQUIRE(p2.getLevelsSize() == p2.getMaxlevel() + 1);

    //     // no host data is created
    //     REQUIRE(!p2.getVPtr());
    //     REQUIRE(!p2.getFPtr());
    //     for (int lv = 0; lv <= p2.getMaxlevel(); lv++)
    //     {
    //         REQUIRE(!p2.getLevelAt(lv).getVPtr());
    //         REQUIRE(!p2.getLevelAt(lv).getFPtr());
    //         REQUIRE(!p2.getLevelAt(lv).getRPtr());
    //     }

    //     // buffers are created
    //     REQUIRE(p2.getDVPtr() == d_v);
    //     REQUIRE(p2.getDFPtr() == d_f);
    //     REQUIRE(p2.getLevelAt(0).getDVInPtr() != d_v.get());
    //     REQUIRE(p2.getLevelAt(0).getDFPtr() != d_f.get());
    //     for (int lv = 0; lv <= p2.getMaxlevel(); lv++)
    //     {
    //         REQUIRE(p2.getLevelAt(lv).getDVInPtr());
    //         REQUIRE(p2.getLevelAt(lv).getDVOutPtr());
    //         REQUIRE(p2.getLevelAt(lv).getDFPtr());
    //         REQUIRE(p2.getLevelAt(lv).getDRPtr());

    //         // check sizes of buffers
    //         int sizeNeeded = p2.getLevelAt(lv).getMgh() * p2.getLevelAt(lv).getNgh() * p2.getLevelAt(lv).getOgh();
    //         REQUIRE(sizeNeeded == p2.getLevelAt(lv).getDVIn().getSize());
    //         REQUIRE(sizeNeeded == p2.getLevelAt(lv).getDVOut().getSize());
    //         REQUIRE(sizeNeeded == p2.getLevelAt(lv).getDF().getSize());
    //         REQUIRE(sizeNeeded == p2.getLevelAt(lv).getDR().getSize());
    //     }

    //     // contents of copied buffer and input buffers are equal
    //     auto lv0d = p2.getLevelAt(0).getDVIn().read(tu.getCommands(), nullptr, true);
    //     auto lv0f = p2.getLevelAt(0).getDF().read(tu.getCommands(), nullptr, true);

    //     REQUIRE(vgh.isEqual(*lv0d));
    //     REQUIRE(fgh.isEqual(*lv0f));
    // }

    SECTION("galerkin")
    {
        // Checks if varying stencils are properly set for each level. Does not check actual values for validity, this
        // is done in an own galerkin test.
        mgcl::MGCL_SMOOTHER smootherType = GENERATE(mgcl::MGCL_JACOBI_SCALAR, mgcl::MGCL_JACOBI_BLOCK);

        p.setSmootherType(smootherType);
        p.init();

        // stencilValues defined in Problem is copied to level 0
        REQUIRE(p.getBlockstencil().get() == &s);
        REQUIRE(p.getBlockstencil().get() == p.getLevelAt(0).getBlockstencil().get());

        for (int lv = 0; lv <= p.getMaxlevel(); lv++)
        {
            auto& level = p.getLevelAt(lv);

            REQUIRE(level.getBlockstencilGpu());
            auto& sv = *level.getBlockstencilGpu();

            REQUIRE(sv.getM() == level.getM());
            REQUIRE(sv.getN() == level.getN());
            REQUIRE(sv.getO() == level.getO());
            REQUIRE(sv.getWidth() == 3);
            REQUIRE(sv.getBlocksize() == blocksize);

            if (smootherType == mgcl::MGCL_JACOBI_BLOCK)
            {
                REQUIRE(level.getBlockstencilGpuInvBlock());
                auto& sv_inv = *level.getBlockstencilGpuInvBlock();

                REQUIRE(sv_inv.getM() == level.getM());
                REQUIRE(sv_inv.getN() == level.getN());
                REQUIRE(sv_inv.getO() == level.getO());
                REQUIRE(sv_inv.getWidth() == 1);
                REQUIRE(sv_inv.getBlocksize() == blocksize);
            }
            else
            {
                REQUIRE(level.getBlockstencilGpuInvScalar());
                auto& sv_inv = *level.getBlockstencilGpuInvScalar();

                REQUIRE(sv_inv.getM() == level.getM());
                REQUIRE(sv_inv.getN() == level.getN());
                REQUIRE(sv_inv.getO() == level.getO());
                REQUIRE(sv_inv.getBlocksize() == blocksize);
            }
        }
    }
}

TEST_CASE("Problem::readResultsBlockstencil")
{
    auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));

    int blocksize = 2;
    auto v = std::make_shared<mgcl::CuboidBS>(4, 5, 6, 2, 2, 2, blocksize);
    auto f = std::make_shared<mgcl::CuboidBS>(4, 5, 6, 2, 2, 2, blocksize);
    v->fillRandom();
    f->fillRandom();

    mgcl::Problem p(4, 5, 6, f, v);
    p.setGhostsIn(2);
    p.setUseOpencl(true);
    p.setDeviceType(deviceType);
    p.getOpenCLHelper().setPreprocessorConstant("BLOCKSIZE", std::to_string(blocksize));

    auto& s = *p.getBlockstencil();

    auto r_fs_tmp = mgcl::create3dFullWeightRestrictionStencil();
    auto p_fs_tmp = mgcl::create3dBilinearProlongationStencil();
    auto& rbs = *p.getRestrictionBlockstencil();
    auto& pbs = *p.getProlongationBlockstencil();
    for (size_t bi = 0; bi < blocksize; bi++)
        for (size_t ii = 0; ii < 3; ii++)
            for (size_t jj = 0; jj < 3; jj++)
                for (size_t kk = 0; kk < 3; kk++)
                {
                    rbs[bi][bi][ii][jj][kk] = r_fs_tmp[ii][jj][kk];
                    pbs[bi][bi][ii][jj][kk] = p_fs_tmp[ii][jj][kk];
                }

    mgcl_test::fill7pLaplace(s, 1.0 / 4.0, false);

    SECTION("default conf")
    {
        p.init();
        REQUIRE(p.getVBSPtr() == v);

        // alter values of dVIn on lowest level
        p.getLevelAt(0).getDVBSIn().fill(p.getProgram(), p.getCommands(), 1.0, false, &p.getKernelConfig(), p.getProfilingData());

        // read back and check if values were copied successfully
        REQUIRE(p.readResults() == CL_SUCCESS);
        REQUIRE(p.getVBSPtr() == v);
        for (int i = 0; i < v->getM(); i++)
            for (int j = 0; j < v->getN(); j++)
                for (int k = 0; k < v->getO(); k++)
                    for (size_t b = 0; b < blocksize; b++)
                    {
                        CHECK((*v)[b][i + v->getGhostsM()][j + v->getGhostsN()][k + v->getGhostsO()] == 1);
                    }
    }
}

/**
 * @brief If stencilType is changed to a varying stencil, stencilValues should be created accordingly.
 *
 */
TEST_CASE("Problem::setStencilTypeBlockstencil")
{
    int blocksize = 2;
    mgcl::Problem p(2, 2, 2);
    REQUIRE_THROWS(p.getStencilValues());
    REQUIRE_THROWS(p.getFixedStencil());
    REQUIRE_THROWS(p.getBlockstencil());

    p.setStencilType(mgcl::MGCL_LAPLACE_19POINT);
    REQUIRE_THROWS(p.getStencilValues());
    REQUIRE_THROWS(p.getFixedStencil());
    REQUIRE_THROWS(p.getBlockstencil());

    p.setStencilType(mgcl::MGCL_LAPLACE_27POINT);
    REQUIRE_THROWS(p.getStencilValues());
    REQUIRE_THROWS(p.getFixedStencil());
    REQUIRE_THROWS(p.getBlockstencil());

    p.setStencilType(mgcl::MGCL_LAPLACE_7POINT);
    REQUIRE_THROWS(p.getStencilValues());
    REQUIRE_THROWS(p.getFixedStencil());
    REQUIRE_THROWS(p.getBlockstencil());

    p.setStencilType(mgcl::MGCL_VARYING);
    REQUIRE(p.getStencilValues() != nullptr);
    REQUIRE_THROWS(p.getFixedStencil());
    REQUIRE_THROWS(p.getBlockstencil());
    CHECK(p.getStencilValues()->getM() == p.getM());
    CHECK(p.getStencilValues()->getN() == p.getN());
    CHECK(p.getStencilValues()->getO() == p.getO());
    CHECK(p.getStencilValues()->getMgh() == p.getM() + 2);
    CHECK(p.getStencilValues()->getNgh() == p.getN() + 2);
    CHECK(p.getStencilValues()->getOgh() == p.getO() + 2);

    p.setStencilType(mgcl::MGCL_FIXED);
    REQUIRE(p.getFixedStencil() != nullptr);
    REQUIRE_THROWS(p.getStencilValues());
    REQUIRE_THROWS(p.getBlockstencil());

    // p.setStencilType(mgcl::MGCL_BLOCKSTENCIL);
    // REQUIRE(p.getBlockstencil());
    // REQUIRE_THROWS(p.getFixedStencil());
    // REQUIRE_THROWS(p.getStencilValues());
    // CHECK(p.getBlockstencil()->getM() == p.getM());
    // CHECK(p.getBlockstencil()->getN() == p.getN());
    // CHECK(p.getBlockstencil()->getO() == p.getO());
    // CHECK(p.getBlockstencil()->getMgh() == p.getM() + 2);
    // CHECK(p.getBlockstencil()->getNgh() == p.getN() + 2);
    // CHECK(p.getBlockstencil()->getOgh() == p.getO() + 2);
    // CHECK(p.getBlockstencil()->getBlocksize() == blocksize);
}
