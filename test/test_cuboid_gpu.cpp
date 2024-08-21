#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>

#include "../src/mgcl/cuboid.hpp"
#include "../src/mgcl/cuboid_gpu.hpp"
#include "../src/mgcl/multigrid_engine.hpp"
#include "cli_args.hpp"
#include "device_type_generator.hpp"
#include "test_utility.hpp"

// Check if CuboidGpu gets initialized correctly with host_ptr being null.
TEST_CASE("CuboidGpu ctor no host_data", "[ocl]")
{
    auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));
    SECTION("success")
    {
        mgcl_test::TestUtility tu(deviceType);
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

    SECTION("Invalid dimensions")
    {
        mgcl_test::TestUtility tu(deviceType);
        REQUIRE_THROWS(mgcl::CuboidGpu(tu.getContext(), CL_MEM_READ_WRITE, 0, 2, 3, 2, 3, 4));
        REQUIRE_THROWS(mgcl::CuboidGpu(tu.getContext(), CL_MEM_READ_WRITE, 1, 0, 3, 2, 3, 4));
        REQUIRE_THROWS(mgcl::CuboidGpu(tu.getContext(), CL_MEM_READ_WRITE, 1, 2, 0, 2, 3, 4));
        REQUIRE_THROWS(mgcl::CuboidGpu(tu.getContext(), CL_MEM_READ_WRITE, 1, 2, 3, -1, 3, 4));
        REQUIRE_THROWS(mgcl::CuboidGpu(tu.getContext(), CL_MEM_READ_WRITE, 1, 2, 3, 2, -1, 4));
        REQUIRE_THROWS(mgcl::CuboidGpu(tu.getContext(), CL_MEM_READ_WRITE, 1, 2, 3, 2, 3, -1));
    }

    SECTION("invalid flags")
    {
        mgcl::Cuboid ch(1, 2, 3, 2, 3, 4);
        mgcl_test::TestUtility tu(deviceType);

        // flags must contain one of CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY or CL_MEM_READ_ONLY.
        REQUIRE_THROWS(mgcl::CuboidGpu(tu.getContext(), CL_MEM_COPY_HOST_PTR, 1, 2, 3, 2, 3, 4));
        REQUIRE_THROWS(mgcl::CuboidGpu(tu.getContext(), CL_MEM_USE_HOST_PTR, 1, 2, 3, 2, 3, 4));
        REQUIRE_THROWS(mgcl::CuboidGpu(tu.getContext(), CL_MEM_ALLOC_HOST_PTR, 1, 2, 3, 2, 3, 4));

        // flags must not not contain CL_MEM_COPY_HOST_PTR, CL_MEM_USE_HOST_PTR or CL_MEM_ALLOC_HOST_PTR
        REQUIRE_THROWS(mgcl::CuboidGpu(tu.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, 1, 2, 3, 2, 3, 4));
        REQUIRE_THROWS(mgcl::CuboidGpu(tu.getContext(), CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, 1, 2, 3, 2, 3, 4));
        REQUIRE_THROWS(mgcl::CuboidGpu(tu.getContext(), CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, 1, 2, 3, 2, 3, 4));

        // flags must not contain multiple of CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY or CL_MEM_WRITE_ONLY
        REQUIRE_THROWS(mgcl::CuboidGpu(tu.getContext(), CL_MEM_READ_WRITE | CL_MEM_WRITE_ONLY, 1, 2, 3, 2, 3, 4));
        REQUIRE_THROWS(mgcl::CuboidGpu(tu.getContext(), CL_MEM_READ_WRITE | CL_MEM_READ_ONLY, 1, 2, 3, 2, 3, 4));
        REQUIRE_THROWS(mgcl::CuboidGpu(tu.getContext(), CL_MEM_WRITE_ONLY | CL_MEM_READ_ONLY, 1, 2, 3, 2, 3, 4));
    }
}

// Check if CuboidGpu gets initialized correctly with host_ptr not being null.
TEST_CASE("CuboidGpu ctor host_data given", "[ocl]")
{
    auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));
    SECTION("sucess")
    {
        mgcl::Cuboid ch(1, 2, 3, 2, 3, 4);
        ch.fillRandom();

        mgcl_test::TestUtility tu(deviceType);
        mgcl::CuboidGpu c(tu.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, ch);

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

    SECTION("invalid flags")
    {
        mgcl::Cuboid ch(1, 2, 3, 2, 3, 4);
        mgcl_test::TestUtility tu(deviceType);

        // flags must contain one of CL_MEM_COPY_HOST_PTR, CL_MEM_USE_HOST_PTR or CL_MEM_ALLOC_HOST_PTR
        REQUIRE_THROWS(mgcl::CuboidGpu(tu.getContext(), CL_MEM_WRITE_ONLY, ch));
        REQUIRE_THROWS(mgcl::CuboidGpu(tu.getContext(), CL_MEM_READ_ONLY, ch));
        REQUIRE_THROWS(mgcl::CuboidGpu(tu.getContext(), CL_MEM_READ_WRITE, ch));

        // flags must not contain multiple of CL_MEM_COPY_HOST_PTR, CL_MEM_USE_HOST_PTR or CL_MEM_ALLOC_HOST_PTR if
        REQUIRE_THROWS(mgcl::CuboidGpu(tu.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_USE_HOST_PTR, ch));
        REQUIRE_THROWS(mgcl::CuboidGpu(tu.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR, ch));
        REQUIRE_THROWS(mgcl::CuboidGpu(tu.getContext(), CL_MEM_USE_HOST_PTR | CL_MEM_ALLOC_HOST_PTR, ch));
    }
}

// Check if CuboidGpu gets initialized correctly retaining an existing buffer.
TEST_CASE("CuboidGpu ctor retaining buffer", "[ocl]")
{
    auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));
    SECTION("success")
    {
        int refCount;
        int err;

        mgcl_test::TestUtility tu(deviceType);
        mgcl::Cuboid ch(1, 2, 3, 2, 3, 4);
        cl_mem buf = tu.createOpenCLBuffer(ch);

        err = clGetMemObjectInfo(buf, CL_MEM_REFERENCE_COUNT, sizeof(cl_uint), &refCount, nullptr);
        mgcl::mgclCheckError(err, "clGetMemObjectInfo(d_v, CL_MEM_REFERENCE_COUNT)");
        REQUIRE(refCount == 1);

        mgcl::CuboidGpu* c = new mgcl::CuboidGpu(tu.getContext(), 0, 1, 2, 3, 2, 3, 4, buf);

        err = clGetMemObjectInfo(buf, CL_MEM_REFERENCE_COUNT, sizeof(cl_uint), &refCount, nullptr);
        mgcl::mgclCheckError(err, "clGetMemObjectInfo(d_v, CL_MEM_REFERENCE_COUNT)");
        REQUIRE(refCount == 2);

        delete c;

        err = clGetMemObjectInfo(buf, CL_MEM_REFERENCE_COUNT, sizeof(cl_uint), &refCount, nullptr);
        mgcl::mgclCheckError(err, "clGetMemObjectInfo(d_v, CL_MEM_REFERENCE_COUNT)");
        REQUIRE(refCount == 1);
    }

    SECTION("Invalid dimensions")
    {
        mgcl_test::TestUtility tu(deviceType);
        REQUIRE_THROWS(mgcl::CuboidGpu(tu.getContext(), CL_MEM_READ_WRITE, 0, 2, 3, 2, 3, 4));
        REQUIRE_THROWS(mgcl::CuboidGpu(tu.getContext(), CL_MEM_READ_WRITE, 1, 0, 3, 2, 3, 4));
        REQUIRE_THROWS(mgcl::CuboidGpu(tu.getContext(), CL_MEM_READ_WRITE, 1, 2, 0, 2, 3, 4));
        REQUIRE_THROWS(mgcl::CuboidGpu(tu.getContext(), CL_MEM_READ_WRITE, 1, 2, 3, -1, 3, 4));
        REQUIRE_THROWS(mgcl::CuboidGpu(tu.getContext(), CL_MEM_READ_WRITE, 1, 2, 3, 2, -1, 4));
        REQUIRE_THROWS(mgcl::CuboidGpu(tu.getContext(), CL_MEM_READ_WRITE, 1, 2, 3, 2, 3, -1));
    }
}

TEST_CASE("CuboidGpu::read into new", "[ocl]")
{
    auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));

    mgcl::Cuboid ch(1, 2, 3, 2, 3, 4);
    ch.fillRandom();

    mgcl_test::TestUtility tu(deviceType);
    mgcl::CuboidGpu c(tu.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, ch);
    auto ret = c.read(tu.getCommands(), nullptr, true);

    REQUIRE(ret->isEqualAllCells(ch));
}

TEST_CASE("CuboidGpu::read into existing", "[ocl]")
{
    auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));

    mgcl::Cuboid ch(1, 2, 3, 2, 3, 4);
    ch.fillRandom();

    mgcl_test::TestUtility tu(deviceType);
    mgcl::CuboidGpu c(tu.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, ch);

    mgcl::Cuboid ch_act(1, 2, 3, 2, 3, 4);
    c.read(tu.getCommands(), &ch_act, true);

    REQUIRE(ch_act.isEqualAllCells(ch));
}

TEST_CASE("CuboidGpu::write", "[ocl]")
{
    auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));

    mgcl::Cuboid ch(1, 2, 3, 2, 3, 4);
    ch.fillRandom();

    mgcl_test::TestUtility tu(deviceType);
    mgcl::CuboidGpu c(tu.getContext(), CL_MEM_READ_WRITE, 1, 2, 3, 2, 3, 4);
    c.write(tu.getCommands(), ch, true);

    auto ret = c.read(tu.getCommands(), nullptr, true);

    REQUIRE(ret->isEqualAllCells(ch));
}

TEST_CASE("CuboidGpu::fill", "[ocl]")
{
    auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));

    mgcl::Cuboid ch(1, 2, 3, 2, 3, 4);
    ch.fillRandom();

    mgcl_test::TestUtility tu(deviceType);
    mgcl::CuboidGpu c(tu.getContext(), CL_MEM_READ_WRITE, 1, 2, 3, 2, 3, 4);
    c.write(tu.getCommands(), ch, true);

    c.fill(tu.getProgram(), tu.getCommands(), 0.0, true, nullptr, nullptr);
    auto ret = c.read(tu.getCommands(), nullptr, true);

    ch.fill(0.0, false);
    REQUIRE(ret->isEqualAllCells(ch));
}

// Tests if CuboidGpu::copyTo works correctly.
TEST_CASE("CuboidGpu::copyTo", "[ocl]")
{
    auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));

    mgcl_test::TestUtility tu(deviceType);

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
TEST_CASE("CuboidGpu::swap", "[ocl]")
{
    auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));

    mgcl_test::TestUtility tu(deviceType);

    SECTION("success")
    {
        mgcl::Cuboid ch1(1, 1, 1, 0, 0, 0);
        mgcl::Cuboid ch2(1, 1, 1, 0, 0, 0);
        ch1[0][0][0] = 1.0;
        ch2[0][0][0] = 2.0;

        mgcl::CuboidGpu c1(tu.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, ch1);
        mgcl::CuboidGpu c2(tu.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, ch2);

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
        mgcl::CuboidGpu c1(tu.getContext(), CL_MEM_READ_WRITE, 3, 1, 1, 0, 0, 0);
        mgcl::CuboidGpu c2(tu.getContext(), CL_MEM_READ_WRITE, 1, 1, 1, 0, 0, 0);
        REQUIRE_THROWS(mgcl::CuboidGpu::swap(c1, c2));
    }
}

// Tests if CuboidGpu::cloneShallow works correctly.
TEST_CASE("CuboidGpu::copyShallow", "[ocl]")
{
    auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));

    mgcl_test::TestUtility tu(deviceType);

    SECTION("success")
    {
        mgcl::Cuboid ch1(1, 2, 3, 0, 1, 2);
        ch1[0][0][0] = 2.0;

        mgcl::CuboidGpu c1(tu.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, ch1);
        auto c2ptr = c1.copyShallow();
        auto& c2 = *c2ptr;

        auto c_act1 = c1.read(tu.getCommands(), nullptr, false);
        auto c_act2 = c2.read(tu.getCommands(), nullptr, true);

        REQUIRE(c_act1->isEqual(ch1));
        REQUIRE(c1.getM() == c2.getM());
        REQUIRE(c1.getN() == c2.getN());
        REQUIRE(c1.getO() == c2.getO());
        REQUIRE(c1.getGhostsM() == c2.getGhostsM());
        REQUIRE(c1.getGhostsN() == c2.getGhostsN());
        REQUIRE(c1.getGhostsO() == c2.getGhostsO());
        REQUIRE(c1.getBuffer() != c2.getBuffer());
        REQUIRE(c2.getFlags() == CL_MEM_READ_WRITE);

        // also check read-only and write-only buffers
        mgcl::CuboidGpu c_ro(tu.getContext(), CL_MEM_READ_ONLY, 1, 2, 3, 1, 2, 3);
        mgcl::CuboidGpu c_wo(tu.getContext(), CL_MEM_WRITE_ONLY, 1, 2, 3, 1, 2, 3);
        auto c2_ro_ptr = c_ro.copyShallow();
        auto& c2_ro = *c2_ro_ptr;
        auto c2_wo_ptr = c_wo.copyShallow();
        auto& c2_wo = *c2_wo_ptr;

        REQUIRE(c_ro.getM() == c2_ro.getM());
        REQUIRE(c_ro.getN() == c2_ro.getN());
        REQUIRE(c_ro.getO() == c2_ro.getO());
        REQUIRE(c_ro.getGhostsM() == c2_ro.getGhostsM());
        REQUIRE(c_ro.getGhostsN() == c2_ro.getGhostsN());
        REQUIRE(c_ro.getGhostsO() == c2_ro.getGhostsO());
        REQUIRE(c_ro.getBuffer() != c2_ro.getBuffer());
        REQUIRE(c2_ro.getFlags() == CL_MEM_READ_ONLY);

        REQUIRE(c_wo.getM() == c2_wo.getM());
        REQUIRE(c_wo.getN() == c2_wo.getN());
        REQUIRE(c_wo.getO() == c2_wo.getO());
        REQUIRE(c_wo.getGhostsM() == c2_wo.getGhostsM());
        REQUIRE(c_wo.getGhostsN() == c2_wo.getGhostsN());
        REQUIRE(c_wo.getGhostsO() == c2_wo.getGhostsO());
        REQUIRE(c_wo.getBuffer() != c2_wo.getBuffer());
        REQUIRE(c2_wo.getFlags() == CL_MEM_WRITE_ONLY);
    }
}

TEST_CASE("CuboidGpu::extract_border_planes", "[ocl]")
{
    auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));

    SECTION("indices")
    {
        int m = 3;
        int n = 5;
        int o = 7;
        int ghosts_m = 1;
        int ghosts_n = 2;
        int ghosts_o = 3;
        int mgh = m + 2 * ghosts_m;
        int ngh = n + 2 * ghosts_n;
        int ogh = o + 2 * ghosts_o;
        int yz = ngh * ogh;
        int xz = mgh * ogh;
        int xy = mgh * ngh;
        int ressize = 2 * yz * ghosts_m + 2 * xz * ghosts_n + 2 * xy * ghosts_o;

        mgcl::Cuboid h_cuboid(m, n, o, ghosts_m, ghosts_n, ghosts_o);
        h_cuboid.fill1dIndex(false);
        const double* buf_cuboid = h_cuboid.field1d().data();

        mgcl::Cuboid h_res(1, 1, ressize);
        double* buf_res = h_res.field1d().data();
        h_res.fill(-1, false);

        for (int cnt = 0; cnt < ressize; cnt++)
        {
            int idx = cnt;
            // plane sizes
            // int yz = ngh * ogh;
            // int xz = mgh * ogh;
            // int xy = mgh * ngh;

            // Front planes
            if (idx < ghosts_m * yz)
            {
                REQUIRE(idx == cnt);

                int i = idx / yz + ghosts_m;
                int j = (idx - (i - ghosts_m) * yz) / ogh;
                int k = idx % ogh;
                buf_res[idx] = buf_cuboid[i * ngh * ogh + j * ogh + k];
            }
            // Back planes
            else if (idx < 2 * ghosts_m * yz)
            {
                idx -= ghosts_m * yz; // reset to 0 for index calculation

                if (cnt == yz)
                    REQUIRE(idx == 0);

                int i = idx / yz + m;
                int j = (idx - (i - m) * yz) / ogh;
                int k = idx % ogh;

                REQUIRE(idx + ghosts_m * yz == cnt);

                buf_res[idx + ghosts_m * yz] = buf_cuboid[i * ngh * ogh + j * ogh + k];
            }
            // Top planes
            else if (idx < 2 * ghosts_m * yz + ghosts_n * xz)
            {
                idx -= 2 * ghosts_m * yz; // reset to 0 for index calculation

                if (cnt == 2 * ghosts_m * yz)
                    REQUIRE(idx == 0);

                int j = idx / xz + ghosts_n;
                int i = (idx - (j - ghosts_n) * xz) / ogh;
                int k = idx % ogh;

                REQUIRE(idx + 2 * ghosts_m * yz == cnt);

                buf_res[idx + 2 * ghosts_m * yz] = buf_cuboid[i * ngh * ogh + j * ogh + k];
            }
            // Bottom planes
            else if (idx < 2 * ghosts_m * yz + 2 * ghosts_n * xz)
            {
                idx -= 2 * ghosts_m * yz + ghosts_n * xz; // reset to 0 for index calculation

                if (cnt == 2 * ghosts_m * yz + ghosts_n * xz)
                    REQUIRE(idx == 0);

                int j = idx / xz + n;
                int i = (idx - (j - n) * xz) / ogh;
                int k = idx % ogh;

                REQUIRE(idx + 2 * ghosts_m * yz + ghosts_n * xz == cnt);

                buf_res[idx + 2 * ghosts_m * yz + ghosts_n * xz] = buf_cuboid[i * ngh * ogh + j * ogh + k];
            }
            // Left planes
            else if (idx < 2 * ghosts_m * yz + 2 * ghosts_n * xz + ghosts_o * xy)
            {
                idx -= 2 * ghosts_m * yz + 2 * ghosts_n * xz; // reset to 0 for index calculation

                if (cnt == 2 * ghosts_m * yz + 2 * ghosts_n * xz)
                    REQUIRE(idx == 0);

                int k = idx / xy + ghosts_o;
                int i = (idx - (k - ghosts_o) * xy) / ngh;
                int j = idx % ngh;

                REQUIRE(idx + 2 * ghosts_m * yz + 2 * ghosts_n * xz == cnt);

                buf_res[idx + 2 * ghosts_m * yz + 2 * ghosts_n * xz] = buf_cuboid[i * ngh * ogh + j * ogh + k];
            }
            // Right planes
            else if (idx < 2 * ghosts_m * yz + 2 * ghosts_n * xz + 2 * ghosts_o * xy)
            {
                idx -= 2 * ghosts_m * yz + 2 * ghosts_n * xz + ghosts_o * xy; // reset to 0 for index calculation

                if (cnt == 2 * ghosts_m * yz + 2 * ghosts_n * xz + ghosts_o * xy)
                    REQUIRE(idx == 0);

                int k = idx / xy + o;
                int i = (idx - (k - o) * xy) / ngh;
                int j = idx % ngh;

                REQUIRE(idx + 2 * ghosts_m * yz + 2 * ghosts_n * xz + ghosts_o * xy == cnt);

                buf_res[idx + 2 * ghosts_m * yz + 2 * ghosts_n * xz + ghosts_o * xy] = buf_cuboid[i * ngh * ogh + j * ogh + k];
            }
        }

        // Check that every index was written to
        for (auto val : h_res.field1d())
            REQUIRE(val >= 0);

        int cnt = 0;
        // front planes (yz)
        for (int i = ghosts_m; i < 2 * ghosts_m; i++) // ghm real planes in the front
            for (int j = 0; j < ngh; j++)             // all real cells in y-dir
                for (int k = 0; k < ogh; k++)         // all real cells in z-dir
                    REQUIRE(buf_res[cnt++] == h_cuboid[i][j][k]);

        // back planes (yz)
        for (int i = m; i < m + ghosts_m; i++) // ghosts_m real planes in the back
            for (int j = 0; j < ngh; j++)      // all real cells in y-dir
                for (int k = 0; k < ogh; k++)  // all real cells in z-dir
                    REQUIRE(buf_res[cnt++] == h_cuboid[i][j][k]);

        // top planes (xz)
        for (int j = ghosts_n; j < 2 * ghosts_n; j++)
            for (int i = 0; i < mgh; i++)
                for (int k = 0; k < ogh; k++)
                    REQUIRE(buf_res[cnt++] == h_cuboid[i][j][k]);

        // bottom planes (xz)
        for (int j = n; j < n + ghosts_n; j++)
            for (int i = 0; i < mgh; i++)
                for (int k = 0; k < ogh; k++)
                    REQUIRE(buf_res[cnt++] == h_cuboid[i][j][k]);

        // left planes (xy)
        for (int k = ghosts_o; k < 2 * ghosts_o; k++)
            for (int i = 0; i < mgh; i++)
                for (int j = 0; j < ngh; j++)
                    REQUIRE(buf_res[cnt++] == h_cuboid[i][j][k]);

        // right planes (xy)
        for (int k = o; k < o + ghosts_o; k++)
            for (int i = 0; i < mgh; i++)
                for (int j = 0; j < ngh; j++)
                    REQUIRE(buf_res[cnt++] == h_cuboid[i][j][k]);
    }

    SECTION("throwing")
    {
        // Create dummy problem
        auto v = std::make_shared<mgcl::Cuboid>(1, 1, 1);
        auto f = std::make_shared<mgcl::Cuboid>(1, 1, 1);
        mgcl::Problem p(1, 1, 1, f, v);
        p.setUseOpencl(true);
        p.setDeviceType(deviceType);
        p.init();

        {
            mgcl::CuboidGpu c(p.getContext(), CL_MEM_READ_WRITE, 1, 1, 1, 2, 1, 1);
            REQUIRE_THROWS(c.extractBorderPlanes(p.getCommands(), p.getProgram(), nullptr, nullptr, nullptr, nullptr));
        }
        {
            mgcl::CuboidGpu c(p.getContext(), CL_MEM_READ_WRITE, 1, 1, 1, 1, 2, 1);
            REQUIRE_THROWS(c.extractBorderPlanes(p.getCommands(), p.getProgram(), nullptr, nullptr, nullptr, nullptr));
        }
        {
            mgcl::CuboidGpu c(p.getContext(), CL_MEM_READ_WRITE, 1, 1, 1, 1, 1, 2);
            REQUIRE_THROWS(c.extractBorderPlanes(p.getCommands(), p.getProgram(), nullptr, nullptr, nullptr, nullptr));
        }
    }

    SECTION("success")
    {
        // Create dummy problem
        auto v = std::make_shared<mgcl::Cuboid>(1, 1, 1);
        auto f = std::make_shared<mgcl::Cuboid>(1, 1, 1);
        mgcl::Problem p(1, 1, 1, f, v);
        p.setUseOpencl(true);
        p.setDeviceType(deviceType);
        p.init();

        int m = 3;
        int n = 5;
        int o = 7;
        int ghosts_m = 1;
        int ghosts_n = 2;
        int ghosts_o = 3;
        int mgh = m + 2 * ghosts_m;
        int ngh = n + 2 * ghosts_n;
        int ogh = o + 2 * ghosts_o;
        int yz = ngh * ogh;
        int xz = mgh * ogh;
        int xy = mgh * ngh;
        int ressize = 2 * yz * ghosts_m + 2 * xz * ghosts_n + 2 * xy * ghosts_o;

        mgcl::Cuboid h_cuboid(m, n, o, ghosts_m, ghosts_n, ghosts_o);
        h_cuboid.fill1dIndex(false);

        mgcl::CuboidGpu c(p.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, h_cuboid);

        auto check = [&](std::vector<double>& extractedBorders) { //
            const double* data_borders = extractedBorders.data();

            int cnt = 0;
            // front planes (yz)
            for (int i = ghosts_m; i < 2 * ghosts_m; i++) // ghm real planes in the front
                for (int j = 0; j < ngh; j++)             // all cells in y-dir
                    for (int k = 0; k < ogh; k++)         // all cells in z-dir
                    {
                        CAPTURE(i, j, k, cnt);
                        REQUIRE(data_borders[cnt++] == h_cuboid[i][j][k]);
                    }

            // back planes (yz)
            for (int i = m; i < m + ghosts_m; i++) // ghosts_m real planes in the back
                for (int j = 0; j < ngh; j++)      // all cells in y-dir
                    for (int k = 0; k < ogh; k++)  // all cells in z-dir
                        REQUIRE(data_borders[cnt++] == h_cuboid[i][j][k]);

            // top planes (xz)
            for (int j = ghosts_n; j < 2 * ghosts_n; j++)
                for (int i = 0; i < mgh; i++)
                    for (int k = 0; k < ogh; k++)
                        REQUIRE(data_borders[cnt++] == h_cuboid[i][j][k]);

            // bottom planes (xz)
            for (int j = n; j < n + ghosts_n; j++)
                for (int i = 0; i < mgh; i++)
                    for (int k = 0; k < ogh; k++)
                        REQUIRE(data_borders[cnt++] == h_cuboid[i][j][k]);

            // left planes (xy)
            for (int k = ghosts_o; k < 2 * ghosts_o; k++)
                for (int i = 0; i < mgh; i++)
                    for (int j = 0; j < ngh; j++)
                        REQUIRE(data_borders[cnt++] == h_cuboid[i][j][k]);

            // right planes (xy)
            for (int k = o; k < o + ghosts_o; k++)
                for (int i = 0; i < mgh; i++)
                    for (int j = 0; j < ngh; j++)
                        REQUIRE(data_borders[cnt++] == h_cuboid[i][j][k]);
        };

        SECTION("no_reuse")
        {
            auto ret = c.extractBorderPlanes(p.getCommands(), p.getProgram(), nullptr, nullptr, nullptr, nullptr);
            REQUIRE(ret != nullptr);
            check(*ret);
        }

        SECTION("reuse_both")
        {
            std::vector<double> h_ret(ressize);
            std::fill(h_ret.begin(), h_ret.end(), -1);
            mgcl::BufferGpu d_tmp(p.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, h_ret);

            c.extractBorderPlanes(p.getCommands(), p.getProgram(), &d_tmp, &h_ret, nullptr, nullptr);
            check(h_ret);
        }

        SECTION("reuse_return_buffer")
        {
            std::vector<double> h_ret(ressize);
            std::fill(h_ret.begin(), h_ret.end(), -1);

            auto ret = c.extractBorderPlanes(p.getCommands(), p.getProgram(), nullptr, &h_ret, nullptr, nullptr);
            REQUIRE(ret == nullptr);
            check(h_ret);
        }

        SECTION("reuse_device_buffer")
        {
            mgcl::BufferGpu d_tmp(p.getContext(), CL_MEM_READ_WRITE, ressize);
            d_tmp.fill(p.getProgram(), p.getCommands(), -1, true, &p.getKernelConfig(), p.getProfilingData());

            auto ret = c.extractBorderPlanes(p.getCommands(), p.getProgram(), &d_tmp, nullptr, nullptr, nullptr);
            check(*ret);
        }
    }
}

TEST_CASE("CuboidGpu::pasteGhostsFromBorderPlanes", "[ocl]")
{
    auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));

    SECTION("indices")
    {
        int m = 3;
        int n = 5;
        int o = 7;
        int ghosts_m = 1;
        int ghosts_n = 2;
        int ghosts_o = 3;
        int mgh = m + 2 * ghosts_m;
        int ngh = n + 2 * ghosts_n;
        int ogh = o + 2 * ghosts_o;
        int yz = ngh * ogh;
        int xz = mgh * ogh;
        int xy = mgh * ngh;
        int ressize = 2 * yz * ghosts_m + 2 * xz * ghosts_n + 2 * xy * ghosts_o;

        mgcl::Cuboid h_cuboid(m, n, o, ghosts_m, ghosts_n, ghosts_o);
        h_cuboid.fill(-1, false);
        double* buf_cuboid = h_cuboid.field1d().data();

        mgcl::Cuboid h_planes(1, 1, ressize);
        h_planes.fill1dIndex(false);
        double* buf_planes = h_planes.field1d().data();

        // Simulate kernel call
        for (int cnt = 0; cnt < ressize; cnt++)
        {
            int idx = cnt;
            // plane sizes
            // int yz = ngh * ogh;
            // int xz = mgh * ogh;
            // int xy = mgh * ngh;

            // Front planes (back ghosts)
            if (idx < ghosts_m * yz)
            {
                int i = idx / yz + m + ghosts_m;
                int j = (idx - (i - (m + ghosts_m)) * yz) / ogh;
                int k = idx % ogh;

                REQUIRE(i >= m + ghosts_m);

                // No corners or edges, only ghosts directly adjacent to real back face
                if (j >= ghosts_n && j < n + ghosts_n && k >= ghosts_o && k < o + ghosts_o)
                    buf_cuboid[i * ngh * ogh + j * ogh + k] = buf_planes[idx];
            }
            // Back planes (front ghosts)
            else if (idx < 2 * ghosts_m * yz)
            {
                idx -= ghosts_m * yz; // reset to 0 for index calculation
                int i = idx / yz;
                int j = (idx - i * yz) / ogh;
                int k = idx % ogh;

                REQUIRE(i < ghosts_m);

                // No corners or edges, only ghosts directly adjacent to real front face
                if (j >= ghosts_n && j < n + ghosts_n && k >= ghosts_o && k < o + ghosts_o)
                    buf_cuboid[i * ngh * ogh + j * ogh + k] = buf_planes[idx + ghosts_m * yz];
            }
            // Top planes (bottom ghosts)
            else if (idx < 2 * ghosts_m * yz + ghosts_n * xz)
            {
                idx -= 2 * ghosts_m * yz; // reset to 0 for index calculation
                int j = idx / xz + n + ghosts_n;
                int i = (idx - (j - (n + ghosts_n)) * xz) / ogh;
                int k = idx % ogh;

                // Ignore left and right ghost cells, but include front and back ghosts
                if (k >= ghosts_o && k < o + ghosts_o)
                    buf_cuboid[i * ngh * ogh + j * ogh + k] = buf_planes[idx + 2 * ghosts_m * yz];
            }
            // Bottom planes (top ghosts)
            else if (idx < 2 * ghosts_m * yz + 2 * ghosts_n * xz)
            {
                idx -= 2 * ghosts_m * yz + ghosts_n * xz; // reset to 0 for index calculation
                int j = idx / xz;
                int i = (idx - j * xz) / ogh;
                int k = idx % ogh;

                // Ignore left and right ghost cells, but include front and back ghosts
                if (k >= ghosts_o && k < o + ghosts_o)
                    buf_cuboid[i * ngh * ogh + j * ogh + k] = buf_planes[idx + 2 * ghosts_m * yz + ghosts_n * xz];
            }
            // Left planes (right ghosts)
            else if (idx < 2 * ghosts_m * yz + 2 * ghosts_n * xz + ghosts_o * xy)
            {
                idx -= 2 * ghosts_m * yz + 2 * ghosts_n * xz; // reset to 0 for index calculation
                int k = idx / xy + o + ghosts_o;
                int i = (idx - (k - (o + ghosts_o)) * xy) / ngh;
                int j = idx % ngh;
                buf_cuboid[i * ngh * ogh + j * ogh + k] = buf_planes[idx + 2 * ghosts_m * yz + 2 * ghosts_n * xz];
            }
            // Right planes (left ghosts)
            else if (idx < 2 * ghosts_m * yz + 2 * ghosts_n * xz + 2 * ghosts_o * xy)
            {
                idx -= 2 * ghosts_m * yz + 2 * ghosts_n * xz + ghosts_o * xy; // reset to 0 for index calculation
                int k = idx / xy;
                int i = (idx - k * xy) / ngh;
                int j = idx % ngh;
                buf_cuboid[i * ngh * ogh + j * ogh + k] = buf_planes[idx + 2 * ghosts_m * yz + 2 * ghosts_n * xz + ghosts_o * xy];
            }
        }
        // End kernel

        // Check that all real cells were left untouched
        for (int i = ghosts_m; i < m + ghosts_m; i++)
            for (int j = ghosts_n; j < n + ghosts_n; j++)
                for (int k = ghosts_o; k < o + ghosts_o; k++)
                {
                    CAPTURE(i, j, k);
                    REQUIRE(h_cuboid[i][j][k] == -1);
                }

        // Check that all ghost cells were filled with any value
        for (int i = 0; i < mgh; i++)
            for (int j = 0; j < ngh; j++)
                for (int k = 0; k < ogh; k++)
                {
                    if ((i < ghosts_m || i >= m + ghosts_m) && (j < ghosts_n || j >= n + ghosts_n) && (k < ghosts_o || k >= o + ghosts_o))
                    {
                        CAPTURE(i, j, k);
                        REQUIRE(h_cuboid[i][j][k] >= 0);
                    }
                }

        // Check that ghost slices are equal to the planes
        auto slice_front = h_cuboid.sliceIncGhosts(0, ghosts_m - 1, 0, ngh - 1, 0, ogh - 1);
        auto slice_back = h_cuboid.sliceIncGhosts(ghosts_m + m, mgh - 1, 0, ngh - 1, 0, ogh - 1);
        auto slice_top = h_cuboid.sliceIncGhosts(0, mgh - 1, 0, ghosts_n - 1, 0, ogh - 1);
        auto slice_bottom = h_cuboid.sliceIncGhosts(0, mgh - 1, ghosts_n + n, ngh - 1, 0, ogh - 1);
        auto slice_left = h_cuboid.sliceIncGhosts(0, mgh - 1, 0, ngh - 1, 0, ghosts_o - 1);
        auto slice_right = h_cuboid.sliceIncGhosts(0, mgh - 1, 0, ngh - 1, o + ghosts_o, ogh - 1);

        int idx = 0;
        // Back ghosts
        for (int i = 0; i < slice_back->getMgh(); i++)
            for (int j = 0; j < slice_back->getNgh(); j++)
                for (int k = 0; k < slice_back->getOgh(); k++)
                {
                    // No corners or edges, only ghosts directly adjacent to real back face
                    if (j >= ghosts_n && j < n + ghosts_n && k >= ghosts_o && k < o + ghosts_o)
                    {
                        CAPTURE(i, j, k);
                        REQUIRE((*slice_back)[i][j][k] == buf_planes[idx]);
                    }
                    idx++;
                }
        // Front ghosts
        for (int i = 0; i < slice_front->getMgh(); i++)
            for (int j = 0; j < slice_front->getNgh(); j++)
                for (int k = 0; k < slice_front->getOgh(); k++)
                {
                    // No corners or edges, only ghosts directly adjacent to real back face
                    if (j >= ghosts_n && j < n + ghosts_n && k >= ghosts_o && k < o + ghosts_o)
                    {
                        CAPTURE(i, j, k);
                        REQUIRE((*slice_front)[i][j][k] == buf_planes[idx]);
                    }
                    idx++;
                }
        // Bottom ghosts
        for (int j = 0; j < slice_bottom->getNgh(); j++)
            for (int i = 0; i < slice_bottom->getMgh(); i++)
                for (int k = 0; k < slice_bottom->getOgh(); k++)
                {
                    // Ignore left and right ghost cells, but include front and back ghosts
                    if (k >= ghosts_o && k < o + ghosts_o)
                    {
                        CAPTURE(i, j, k);
                        REQUIRE((*slice_bottom)[i][j][k] == buf_planes[idx]);
                    }
                    idx++;
                }
        // Top ghosts
        for (int j = 0; j < slice_top->getNgh(); j++)
            for (int i = 0; i < slice_top->getMgh(); i++)
                for (int k = 0; k < slice_top->getOgh(); k++)
                {
                    // Ignore left and right ghost cells, but include front and back ghosts
                    if (k >= ghosts_o && k < o + ghosts_o)
                    {
                        CAPTURE(i, j, k);
                        REQUIRE((*slice_top)[i][j][k] == buf_planes[idx]);
                    }
                    idx++;
                }
        // Right ghosts
        for (int k = 0; k < slice_right->getOgh(); k++)
            for (int i = 0; i < slice_right->getMgh(); i++)
                for (int j = 0; j < slice_right->getNgh(); j++)
                {
                    CAPTURE(i, j, k);
                    REQUIRE((*slice_right)[i][j][k] == buf_planes[idx]);
                    idx++;
                }
        // Left ghosts
        for (int k = 0; k < slice_left->getOgh(); k++)
            for (int i = 0; i < slice_left->getMgh(); i++)
                for (int j = 0; j < slice_left->getNgh(); j++)
                {
                    CAPTURE(i, j, k);
                    REQUIRE((*slice_left)[i][j][k] == buf_planes[idx]);
                    idx++;
                }
    }

    SECTION("success")
    {
        // Create dummy problem
        auto v = std::make_shared<mgcl::Cuboid>(1, 1, 1);
        auto f = std::make_shared<mgcl::Cuboid>(1, 1, 1);
        mgcl::Problem p(1, 1, 1, f, v);
        p.setUseOpencl(true);
        p.setDeviceType(deviceType);
        p.init();

        int m = 3;
        int n = 5;
        int o = 7;
        int ghosts_m = 1;
        int ghosts_n = 2;
        int ghosts_o = 3;
        int mgh = m + 2 * ghosts_m;
        int ngh = n + 2 * ghosts_n;
        int ogh = o + 2 * ghosts_o;
        int yz = ngh * ogh;
        int xz = mgh * ogh;
        int xy = mgh * ngh;
        int ressize = 2 * yz * ghosts_m + 2 * xz * ghosts_n + 2 * xy * ghosts_o;

        mgcl::Cuboid h_cuboid(m, n, o, ghosts_m, ghosts_n, ghosts_o);
        h_cuboid.fill(-1, false);

        std::vector<double> h_planes(ressize);
        for (size_t i = 0; i < h_planes.size(); i++)
        {
            h_planes[i] = i;
        }
        double* buf_planes = h_planes.data();

        mgcl::CuboidGpu d_cuboid(p.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, h_cuboid);
        mgcl::BufferGpu d_planes(p.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, h_planes);

        // paste planes
        d_cuboid.pasteGhostsFromBorderPlanes(p.getContext(), p.getCommands(), p.getProgram(), &d_planes, nullptr, nullptr, nullptr);

        // read result
        d_cuboid.read(p.getCommands(), &h_cuboid, true);

        // Check that ghost slices are equal to the planes
        auto slice_front = h_cuboid.sliceIncGhosts(0, ghosts_m - 1, 0, ngh - 1, 0, ogh - 1);
        auto slice_back = h_cuboid.sliceIncGhosts(ghosts_m + m, mgh - 1, 0, ngh - 1, 0, ogh - 1);
        auto slice_top = h_cuboid.sliceIncGhosts(0, mgh - 1, 0, ghosts_n - 1, 0, ogh - 1);
        auto slice_bottom = h_cuboid.sliceIncGhosts(0, mgh - 1, ghosts_n + n, ngh - 1, 0, ogh - 1);
        auto slice_left = h_cuboid.sliceIncGhosts(0, mgh - 1, 0, ngh - 1, 0, ghosts_o - 1);
        auto slice_right = h_cuboid.sliceIncGhosts(0, mgh - 1, 0, ngh - 1, o + ghosts_o, ogh - 1);

        int idx = 0;
        // Back ghosts
        for (int i = 0; i < slice_back->getMgh(); i++)
            for (int j = 0; j < slice_back->getNgh(); j++)
                for (int k = 0; k < slice_back->getOgh(); k++)
                {
                    // No corners or edges, only ghosts directly adjacent to real back face
                    if (j >= ghosts_n && j < n + ghosts_n && k >= ghosts_o && k < o + ghosts_o)
                    {
                        CAPTURE(i, j, k);
                        REQUIRE((*slice_back)[i][j][k] == buf_planes[idx]);
                    }
                    idx++;
                }
        // Front ghosts
        for (int i = 0; i < slice_front->getMgh(); i++)
            for (int j = 0; j < slice_front->getNgh(); j++)
                for (int k = 0; k < slice_front->getOgh(); k++)
                {
                    // No corners or edges, only ghosts directly adjacent to real back face
                    if (j >= ghosts_n && j < n + ghosts_n && k >= ghosts_o && k < o + ghosts_o)
                    {
                        CAPTURE(i, j, k);
                        REQUIRE((*slice_front)[i][j][k] == buf_planes[idx]);
                    }
                    idx++;
                }
        // Bottom ghosts
        for (int j = 0; j < slice_bottom->getNgh(); j++)
            for (int i = 0; i < slice_bottom->getMgh(); i++)
                for (int k = 0; k < slice_bottom->getOgh(); k++)
                {
                    // Ignore left and right ghost cells, but include front and back ghosts
                    if (k >= ghosts_o && k < o + ghosts_o)
                    {
                        CAPTURE(i, j, k);
                        REQUIRE((*slice_bottom)[i][j][k] == buf_planes[idx]);
                    }
                    idx++;
                }
        // Top ghosts
        for (int j = 0; j < slice_top->getNgh(); j++)
            for (int i = 0; i < slice_top->getMgh(); i++)
                for (int k = 0; k < slice_top->getOgh(); k++)
                {
                    // Ignore left and right ghost cells, but include front and back ghosts
                    if (k >= ghosts_o && k < o + ghosts_o)
                    {
                        CAPTURE(i, j, k);
                        REQUIRE((*slice_top)[i][j][k] == buf_planes[idx]);
                    }
                    idx++;
                }
        // Right ghosts
        for (int k = 0; k < slice_right->getOgh(); k++)
            for (int i = 0; i < slice_right->getMgh(); i++)
                for (int j = 0; j < slice_right->getNgh(); j++)
                {
                    CAPTURE(i, j, k);
                    REQUIRE((*slice_right)[i][j][k] == buf_planes[idx]);
                    idx++;
                }
        // Left ghosts
        for (int k = 0; k < slice_left->getOgh(); k++)
            for (int i = 0; i < slice_left->getMgh(); i++)
                for (int j = 0; j < slice_left->getNgh(); j++)
                {
                    CAPTURE(i, j, k);
                    REQUIRE((*slice_left)[i][j][k] == buf_planes[idx]);
                    idx++;
                }
    }
}