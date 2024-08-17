#include <CL/cl.h>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <ctgmath>
#include <iostream>
#include <memory>
#include <vector>

#include "../src/mgcl/cuboid.hpp"
#include "../src/mgcl/multigrid_engine.hpp"
#include "../src/mgcl/opencl_helper.hpp"
#include "../src/mgcl/stencil.hpp"

#include "test_utility.hpp"

using std::min;

TEST_CASE("VaryingStencilGpu ctor+dtor")
{
    auto deviceType = CL_DEVICE_TYPE_GPU; // GENERATE(CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU);

    if (!mgcl_test::TestUtility::deviceAvailable("", deviceType))
    {
        std::string typeName = deviceType == CL_DEVICE_TYPE_GPU ? "CL_DEVICE_TYPE_GPU" : "CL_DEVICE_TYPE_CPU";
        std::cout << "Skipping non-available device type '" << typeName << "'" << std::endl;
        return;
    }

    mgcl_test::TestUtility t(deviceType);

    int m = 4;
    int n = 4;
    int o = 4;
    int width = 3;
    int gh = 2;
    auto s = std::make_unique<mgcl::VaryingStencilGpu>(m, n, o, width, gh, t.getContext(), t.getCommands(), t.getProgram());
    t.finish();

    REQUIRE(s->getBuf());

    // check size of buffer
    size_t bufsize;
    int sizeNeeded = (m + 2 * gh) * (n + 2 * gh) * (o + 2 * gh) * width * width * width;
    int err = clGetMemObjectInfo(s->getBuf(), CL_MEM_SIZE, sizeof(size_t), &bufsize, nullptr);
    mgcl::mgclCheckError(err, "Querying buffer size");
    REQUIRE(bufsize == sizeof(double) * sizeNeeded);

    // check reference count, should be 1
    cl_uint refCount;

    err = clGetMemObjectInfo(s->getBuf(), CL_MEM_REFERENCE_COUNT, sizeof(cl_uint), &refCount, nullptr);
    mgcl::mgclCheckError(err, "clGetMemObjectInfo(s, CL_MEM_REFERENCE_COUNT)");
    REQUIRE(refCount == 1);

    // check values are 0
    double tmp[sizeNeeded];
    err = clEnqueueReadBuffer(t.getCommands(), s->getBuf(), CL_TRUE, 0,
                              sizeof(double) * sizeNeeded, tmp, 0, NULL, NULL);
    mgcl::mgclCheckError(err, "clEnqueueReadBuffer");

    for (int i = 0; i < sizeNeeded; i++)
        REQUIRE(tmp[i] == 0);

    // delete s
    s.reset();
}

TEST_CASE("VaryingStencilGpu move ctor")
{
    auto deviceType = CL_DEVICE_TYPE_GPU; // GENERATE(CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU);

    if (!mgcl_test::TestUtility::deviceAvailable("", deviceType))
    {
        std::string typeName = deviceType == CL_DEVICE_TYPE_GPU ? "CL_DEVICE_TYPE_GPU" : "CL_DEVICE_TYPE_CPU";
        std::cout << "Skipping non-available device type '" << typeName << "'" << std::endl;
        return;
    }

    mgcl_test::TestUtility t(deviceType);

    int n = GENERATE(1, 2, 3);
    int m = GENERATE(1, 2, 3);
    int o = GENERATE(1, 2, 3);

    mgcl::VaryingStencil h(m, n, o, 3, 0, 0, 0);
    h.fillRandom();
    auto hgpu = std::make_unique<mgcl::VaryingStencilGpu>(m, n, o, 3, 0, t.getContext(), t.getCommands(), t.getProgram());
    hgpu->fill(h, t.getCommands(), true);

    // copy manually for checking results
    mgcl::VaryingStencil h_check(m, n, o, 3, 0, 0, 0);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < o; k++)
                for (int ii = 0; ii < 3; ii++)
                    for (int jj = 0; jj < 3; jj++)
                        for (int kk = 0; kk < 3; kk++)
                        {
                            h_check[ii][jj][kk][i][j][k] = h[ii][jj][kk][i][j][k];
                        }

    // check move ctor
    auto h2gpu = std::make_unique<mgcl::VaryingStencilGpu>(std::move(*hgpu));
    // auto h2gpu(std::move(*hgpu));

    REQUIRE(hgpu->getM() == 0);
    REQUIRE(hgpu->getN() == 0);
    REQUIRE(hgpu->getO() == 0);
    REQUIRE(hgpu->getGh() == 0);
    REQUIRE(hgpu->getWidth() == 0);
    auto h2 = h2gpu->read(t.getCommands(), true);
    t.finish();
    REQUIRE(h2.isEqual(h_check));

    // check if buffer has count 2 (since hgpu is not deleted yet)
    int err;
    cl_uint refCount = 0;
    err = clGetMemObjectInfo(hgpu->getBuf(), CL_MEM_REFERENCE_COUNT, sizeof(cl_uint), &refCount, nullptr);
    mgcl::mgclCheckError(err, "clGetMemObjectInfo(hgpu->getBuf(), CL_MEM_REFERENCE_COUNT)");
    REQUIRE(err == CL_SUCCESS);
    REQUIRE(refCount == 2);

    // now delete object, ref count should be 1
    hgpu.reset();
    t.finish();
    err = clGetMemObjectInfo(h2gpu->getBuf(), CL_MEM_REFERENCE_COUNT, sizeof(cl_uint), &refCount, nullptr);
    mgcl::mgclCheckError(err, "clGetMemObjectInfo(hgpu->getBuf(), CL_MEM_REFERENCE_COUNT)");
    REQUIRE(err == CL_SUCCESS);
    REQUIRE(refCount == 1);

    // check move assignment
    auto h3gpu = std::move(*h2gpu);

    REQUIRE(h2gpu->getM() == 0);
    REQUIRE(h2gpu->getN() == 0);
    REQUIRE(h2gpu->getO() == 0);
    REQUIRE(h2gpu->getGh() == 0);
    REQUIRE(h2gpu->getWidth() == 0);
    auto h3 = h3gpu.read(t.getCommands(), true);
    t.finish();
    REQUIRE(h3.isEqual(h_check));

    // check if buffer has count 2 (since hgpu is not deleted yet)
    err = clGetMemObjectInfo(h2gpu->getBuf(), CL_MEM_REFERENCE_COUNT, sizeof(cl_uint), &refCount, nullptr);
    mgcl::mgclCheckError(err, "clGetMemObjectInfo(hgpu->getBuf(), CL_MEM_REFERENCE_COUNT)");
    REQUIRE(err == CL_SUCCESS);
    REQUIRE(refCount == 2);

    // now delete object, ref count should be 1
    h2gpu.reset();
    t.finish();
    err = clGetMemObjectInfo(h3gpu.getBuf(), CL_MEM_REFERENCE_COUNT, sizeof(cl_uint), &refCount, nullptr);
    mgcl::mgclCheckError(err, "clGetMemObjectInfo(hgpu->getBuf(), CL_MEM_REFERENCE_COUNT)");
    REQUIRE(err == CL_SUCCESS);
    REQUIRE(refCount == 1);
}

TEST_CASE("VaryingStencilGpu::fill")
{
    auto deviceType = CL_DEVICE_TYPE_GPU; // GENERATE(CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU);

    if (!mgcl_test::TestUtility::deviceAvailable("", deviceType))
    {
        std::string typeName = deviceType == CL_DEVICE_TYPE_GPU ? "CL_DEVICE_TYPE_GPU" : "CL_DEVICE_TYPE_CPU";
        std::cout << "Skipping non-available device type '" << typeName << "'" << std::endl;
        return;
    }

    mgcl_test::TestUtility t(deviceType);

    int m = 4;
    int n = 4;
    int o = 4;
    int gh = 2;

    SECTION("VaryingStencil3x3x3")
    {
        int width = 3;
        auto s = std::make_unique<mgcl::VaryingStencilGpu>(m, n, o, width, gh, t.getContext(), t.getCommands(), t.getProgram());
        t.finish();

        int sizeNeeded = (m + 2 * gh) * (n + 2 * gh) * (o + 2 * gh) * width * width * width;

        // create VaryingStencil, fill with random values and copy to gpu buffer
        mgcl::VaryingStencil s3(m, n, o, 3, gh, gh, gh);
        s3.fillRandomInt();
        s->fill(s3, t.getCommands(), true);

        // check results
        double tmp[sizeNeeded];
        int err = clEnqueueReadBuffer(t.getCommands(), s->getBuf(), CL_TRUE, 0,
                                      sizeof(double) * sizeNeeded, tmp, 0, NULL, NULL);
        mgcl::mgclCheckError(err, "clEnqueueReadBuffer");

        for (int i = 0; i < sizeNeeded; i++)
            REQUIRE(tmp[i] == s3[0][0][0][0][0][i]);
    }

    SECTION("VaryingStencil5x5x5")
    {
        int width = 5;
        auto s = std::make_unique<mgcl::VaryingStencilGpu>(m, n, o, width, gh, t.getContext(), t.getCommands(), t.getProgram());
        t.finish();

        int sizeNeeded = (m + 2 * gh) * (n + 2 * gh) * (o + 2 * gh) * width * width * width;

        // create VaryingStencil, fill with random values and copy to gpu buffer
        mgcl::VaryingStencil s3(m, n, o, 5, gh, gh, gh);
        s3.fillRandomInt();
        s->fill(s3, t.getCommands(), true);

        // check results
        double tmp[sizeNeeded];
        int err = clEnqueueReadBuffer(t.getCommands(), s->getBuf(), CL_TRUE, 0,
                                      sizeof(double) * sizeNeeded, tmp, 0, NULL, NULL);
        mgcl::mgclCheckError(err, "clEnqueueReadBuffer");

        for (int i = 0; i < sizeNeeded; i++)
            REQUIRE(tmp[i] == s3[0][0][0][0][0][i]);
    }

    SECTION("throwing")
    {
        int width = 3;
        auto s = std::make_unique<mgcl::VaryingStencilGpu>(m, n, o, width, gh, t.getContext(), t.getCommands(), t.getProgram());
        t.finish();

        // widths do not match
        mgcl::VaryingStencil s3(m, n, o, 5, gh, gh, gh);
        REQUIRE_THROWS(s->fill(s3, t.getCommands(), true));

        // grid sizes do not match
        mgcl::VaryingStencil s35(m * 2, n * 3, o * 4, 5, gh, gh, gh);
        REQUIRE_THROWS(s->fill(s35, t.getCommands(), true));
    }
}

TEST_CASE("VaryingStencilGpu::read")
{
    auto deviceType = CL_DEVICE_TYPE_GPU; // GENERATE(CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU);

    if (!mgcl_test::TestUtility::deviceAvailable("", deviceType))
    {
        std::string typeName = deviceType == CL_DEVICE_TYPE_GPU ? "CL_DEVICE_TYPE_GPU" : "CL_DEVICE_TYPE_CPU";
        std::cout << "Skipping non-available device type '" << typeName << "'" << std::endl;
        return;
    }

    mgcl_test::TestUtility t(deviceType);

    int m = 4;
    int n = 4;
    int o = 4;
    int gh = 2;

    SECTION("VaryingStencil3x3x3")
    {
        int width = 3;
        auto s = std::make_unique<mgcl::VaryingStencilGpu>(m, n, o, width, gh, t.getContext(), t.getCommands(), t.getProgram());
        t.finish();

        // create VaryingStencil, fill with random values and copy to gpu buffer
        mgcl::VaryingStencil s3(m, n, o, 3, gh, gh, gh);
        s3.fillRandomInt();
        s->fill(s3, t.getCommands(), true);

        // read buffer
        auto ret = s->read(t.getCommands(), true);
        t.finish();

        // check results
        for (int i = 0; i < ret.field1d().size(); i++)
            REQUIRE(ret.field1d()[i] == s3.field1d()[i]);
    }

    SECTION("VaryingStencil5x5x5")
    {
        int width = 5;
        auto s = std::make_unique<mgcl::VaryingStencilGpu>(m, n, o, width, gh, t.getContext(), t.getCommands(), t.getProgram());
        t.finish();

        // create VaryingStencil, fill with random values and copy to gpu buffer
        mgcl::VaryingStencil s3(m, n, o, 5, gh, gh, gh);
        s3.fillRandomInt();
        s->fill(s3, t.getCommands(), true);

        // read buffer
        auto ret = s->read(t.getCommands(), true);
        t.finish();

        // check results
        for (int i = 0; i < ret.field1d().size(); i++)
            REQUIRE(ret.field1d()[i] == s3.field1d()[i]);
    }
}

TEST_CASE("VaryingStencilGpu::updateGhosts")
{
    auto deviceType = CL_DEVICE_TYPE_GPU; // GENERATE(CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU);

    if (!mgcl_test::TestUtility::deviceAvailable("", deviceType))
    {
        std::string typeName = deviceType == CL_DEVICE_TYPE_GPU ? "CL_DEVICE_TYPE_GPU" : "CL_DEVICE_TYPE_CPU";
        std::cout << "Skipping non-available device type '" << typeName << "'" << std::endl;
        return;
    }

    mgcl_test::TestUtility t(deviceType);

    SECTION("checking indices (1d), gh > m")
    {
        int gh = 8;
        int m = 3;
        std::vector<int> ireal_expected{9, 10, 8, 9, 10, 8, 9, 10, 8, 9, 10, 8, 9, 10, 8, 9};
        std::vector<int> factors_expected{3, 3, 2, 2, 2, 1, 1, 1, -1, -1, -1, -2, -2, -2, -3, -3};

        for (int globalId = 0; globalId < 2 * gh; globalId++)
        {
            int i = globalId >= gh ? globalId + m : globalId;
            int factor = floor(((double)(gh - 1 - i)) / m + 1);
            int ireal = i + factor * m;

            REQUIRE(factor == factors_expected[globalId]);
            REQUIRE(ireal == ireal_expected[globalId]);
        }
    }

    SECTION("checking indices (1d), gh < m")
    {
        int gh = 2;
        int m = 5;
        std::vector<int> ireal_expected{5, 6, 2, 3};
        std::vector<int> factors_expected{1, 1, -1, -1};

        for (int globalId = 0; globalId < 2 * gh; globalId++)
        {
            int i = globalId >= gh ? globalId + m : globalId;
            int factor = floor(((double)(gh - 1 - i)) / m + 1);
            int ireal = i + factor * m;

            REQUIRE(factor == factors_expected[globalId]);
            REQUIRE(ireal == ireal_expected[globalId]);
        }
    }

    // This test simulates the opencl kernel and checks if every ghost cell is written to exactly once.
    SECTION("writing exactly once? m > gh")
    {
        int m = 4;
        int n = 4;
        int o = 4;
        int gh = 2;
        int width = 3;

        // 1d indices
        int widthPow2 = width * width;
        int widthPow3 = widthPow2 * width;
        int widthPow3o = widthPow3 * (o + 2 * gh);
        int widthPow3on = widthPow3o * (n + 2 * gh);

        mgcl::VaryingStencil s3(m, n, o, 3, gh, gh, gh);
        for (int i = 0; i < m + 2 * gh; i++)
            for (int j = 0; j < n + 2 * gh; j++)
                for (int k = 0; k < o + 2 * gh; k++)
                {
                    // if cell is ghost cell
                    if (i < gh || j < gh || k < gh ||
                        i >= gh + m || j >= gh + n || k >= gh + o)
                    {
                        int ireal = i + floor(((double)(gh - 1 - i)) / m + 1) * m;
                        int jreal = j + floor(((double)(gh - 1 - j)) / n + 1) * n;
                        int kreal = k + floor(((double)(gh - 1 - k)) / o + 1) * o;

                        int gridsize = (m + 2 * gh) * (n + 2 * gh) * (o + 2 * gh);
                        int idx_gh_cell = i * (n + 2 * gh) * (o + 2 * gh) + j * (o + 2 * gh) + k;
                        int idx_real_cell = ireal * (n + 2 * gh) * (o + 2 * gh) + jreal * (o + 2 * gh) + kreal;

                        // Iterate over every coefficient for the grid point this work-item maps to.
                        // for (int s = 0; s < width * width * width; s++)
                        // clang-format off
                        for (int ii = 0; ii < width; ii++)
                        for (int jj = 0; jj < width; jj++)
                        for (int kk = 0; kk < width; kk++)
                        {
                            s3.field1d()[idx_gh_cell]++;

                            // check that ghost cell is written to once
                            REQUIRE(s3[ii][jj][kk][i][j][k] == 1);

                            // check that 1d index coincides with 6d index
                            REQUIRE(s3[ii][jj][kk][i][j][k] == s3.field1d()[idx_gh_cell]);

                            // check that calculated indices of real cell is actually a real cell
                            REQUIRE(ireal >= gh);
                            REQUIRE(ireal < m + gh);
                            REQUIRE(jreal >= gh);
                            REQUIRE(jreal < n + gh);
                            REQUIRE(kreal >= gh);
                            REQUIRE(kreal < o + gh);

                            idx_gh_cell += gridsize;
                            idx_real_cell += gridsize;
                        }
                        // clang-format on
                    }
                }

        // Now check if all ghost cells are 1 and all real cells are 0
        for (int i = 0; i < m + 2 * gh; i++)
            for (int j = 0; j < n + 2 * gh; j++)
                for (int k = 0; k < o + 2 * gh; k++)
                    for (int ii = 0; ii < width; ii++)
                        for (int jj = 0; jj < width; jj++)
                            for (int kk = 0; kk < width; kk++)
                            {
                                if (i < gh || j < gh || k < gh ||
                                    i >= gh + m || j >= gh + n || k >= gh + o)
                                    REQUIRE(s3[ii][jj][kk][i][j][k] == 1);
                                else
                                    REQUIRE(s3[ii][jj][kk][i][j][k] == 0);
                            }
    }

    // This test simulates the opencl kernel and checks if every ghost cell is written to exactly once.
    SECTION("writing exactly once? m < gh")
    {
        int m = 2;
        int n = 2;
        int o = 6;
        int gh = 5;
        int width = 3;

        // 1d indices
        int widthPow2 = width * width;
        int widthPow3 = widthPow2 * width;
        int widthPow3o = widthPow3 * (o + 2 * gh);
        int widthPow3on = widthPow3o * (n + 2 * gh);

        mgcl::VaryingStencil s3(m, n, o, 3, gh, gh, gh);
        for (int i = 0; i < m + 2 * gh; i++)
            for (int j = 0; j < n + 2 * gh; j++)
                for (int k = 0; k < o + 2 * gh; k++)
                {
                    // if cell is ghost cell
                    if (i < gh || j < gh || k < gh ||
                        i >= gh + m || j >= gh + n || k >= gh + o)
                    {
                        int ireal = i + floor(((double)(gh - 1 - i)) / m + 1) * m;
                        int jreal = j + floor(((double)(gh - 1 - j)) / n + 1) * n;
                        int kreal = k + floor(((double)(gh - 1 - k)) / o + 1) * o;

                        int gridsize = (m + 2 * gh) * (n + 2 * gh) * (o + 2 * gh);
                        int idx_gh_cell = i * (n + 2 * gh) * (o + 2 * gh) + j * (o + 2 * gh) + k;
                        int idx_real_cell = ireal * (n + 2 * gh) * (o + 2 * gh) + jreal * (o + 2 * gh) + kreal;

                        // Iterate over every coefficient for the grid point this work-item maps to.
                        // for (int s = 0; s < width * width * width; s++)
                        // clang-format off
                        for (int ii = 0; ii < width; ii++)
                        for (int jj = 0; jj < width; jj++)
                        for (int kk = 0; kk < width; kk++)
                        {
                            s3.field1d()[idx_gh_cell]++;

                            // check that ghost cell is written to once
                            REQUIRE(s3[ii][jj][kk][i][j][k] == 1);

                            // check that 1d index coincides with 6d index
                            REQUIRE(s3[ii][jj][kk][i][j][k] == s3.field1d()[idx_gh_cell]);

                            // check that calculated indices of real cell is actually a real cell
                            REQUIRE(ireal >= gh);
                            REQUIRE(ireal < m + gh);
                            REQUIRE(jreal >= gh);
                            REQUIRE(jreal < n + gh);
                            REQUIRE(kreal >= gh);
                            REQUIRE(kreal < o + gh);

                            idx_gh_cell += gridsize;
                            idx_real_cell += gridsize;
                        }
                        // clang-format on
                    }
                }

        // Now check if all ghost cells are 1 and all real cells are 0
        for (int i = 0; i < m + 2 * gh; i++)
            for (int j = 0; j < n + 2 * gh; j++)
                for (int k = 0; k < o + 2 * gh; k++)
                    for (int ii = 0; ii < width; ii++)
                        for (int jj = 0; jj < width; jj++)
                            for (int kk = 0; kk < width; kk++)
                            {
                                if (i < gh || j < gh || k < gh ||
                                    i >= gh + m || j >= gh + n || k >= gh + o)
                                    REQUIRE(s3[ii][jj][kk][i][j][k] == 1);
                                else
                                    REQUIRE(s3[ii][jj][kk][i][j][k] == 0);
                            }
    }

    // int m = GENERATE(2, 3, 4);
    // int n = GENERATE(2, 3, 4);
    // int o = GENERATE(2, 3, 4);
    // int gh = GENERATE(2, 3, 4);
    int m = 2;
    int n = 3;
    int o = 4;
    int gh = GENERATE(2, 3);

    SECTION("VaryingStencil3x3x3")
    {
        int width = 3;
        auto s = std::make_unique<mgcl::VaryingStencilGpu>(m, n, o, width, gh, t.getContext(), t.getCommands(), t.getProgram());
        t.finish();

        // create VaryingStencil, fill with random values and copy to gpu buffer
        mgcl::VaryingStencil s3(m, n, o, 3, gh, gh, gh);
        s3.fillRandomInt();
        s->fill(s3, t.getCommands(), true);

        // update ghosts of both, host and device stencils
        s3.updateGhosts();
        s->updateGhosts(t.getProgram(), t.getCommands(), nullptr, nullptr);
        t.finish();

        // read buffer
        auto ret = s->read(t.getCommands(), true);
        t.finish();

        // check results
        for (int i = 0; i < ret.field1d().size(); i++)
            REQUIRE(ret.field1d()[i] == s3.field1d()[i]);
    }

    SECTION("VaryingStencil5x5x5")
    {
        int width = 5;
        auto s = std::make_unique<mgcl::VaryingStencilGpu>(m, n, o, width, gh, t.getContext(), t.getCommands(), t.getProgram());
        t.finish();

        // create VaryingStencil, fill with random values and copy to gpu buffer
        mgcl::VaryingStencil s3(m, n, o, 5, gh, gh, gh);
        s3.fillRandomInt();
        s->fill(s3, t.getCommands(), true);

        // update ghosts of both, host and device stencils
        s3.updateGhosts();
        s->updateGhosts(t.getProgram(), t.getCommands(), nullptr, nullptr);
        t.finish();

        // read buffer
        auto ret = s->read(t.getCommands(), true);
        t.finish();

        // check results
        for (int i = 0; i < ret.field1d().size(); i++)
            REQUIRE(ret.field1d()[i] == s3.field1d()[i]);
    }
}

TEST_CASE("VaryingStencilGpu::multiply(var)")
{
    auto deviceType = CL_DEVICE_TYPE_GPU; // GENERATE(CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU);

    if (!mgcl_test::TestUtility::deviceAvailable("", deviceType))
    {
        std::string typeName = deviceType == CL_DEVICE_TYPE_GPU ? "CL_DEVICE_TYPE_GPU" : "CL_DEVICE_TYPE_CPU";
        std::cout << "Skipping non-available device type '" << typeName << "'" << std::endl;
        return;
    }

    mgcl_test::TestUtility t(deviceType);

    int m = 4;
    int n = 4;
    int o = 4;
    int gh = 2;

    SECTION("checking indices")
    {
        mgcl::VaryingStencil a(m, n, o, 3, gh, gh, gh);
        mgcl::VaryingStencil b(m, n, o, 3, gh, gh, gh);
        mgcl::VaryingStencil c(m, n, o, 5, gh, gh, gh);

        // fill with 1d cell index
        for (int i = 0; i < a.field1d().size(); i++)
            a.field1d()[i] = i;
        for (int i = 0; i < b.field1d().size(); i++)
            b.field1d()[i] = i;
        for (int i = 0; i < c.field1d().size(); i++)
            c.field1d()[i] = i;

        int wa = 3;
        int wb = 3;
        // int wc = 5;
        int gha = gh;
        int ghb = gh;
        int ghc = gh;

        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < o; k++)
                {
                    int wa2 = wa >> 1;
                    int wc = wa + wb - 1;

                    int gridsize_a = (m + 2 * gha) * (n + 2 * gha) * (o + 2 * gha);
                    int gridsize_b = (m + 2 * ghb) * (n + 2 * ghb) * (o + 2 * ghb);
                    int gridsize_c = (m + 2 * ghc) * (n + 2 * ghc) * (o + 2 * ghc);

                    // 1d grid point index offset from the start of the coefficient list
                    int wcPow2 = wc * wc;
                    int wcPow3 = wcPow2 * wc;
                    int cell_c = (i + ghc) * (n + 2 * ghc) * (o + 2 * ghc) + (j + ghc) * (o + 2 * ghc) + (k + ghc);

                    int waPow2 = wa * wa;
                    int waPow3 = waPow2 * wa;
                    int cell_a = (i + gha) * (n + 2 * gha) * (o + 2 * gha) + (j + gha) * (o + 2 * gha) + (k + gha);

                    int wbPow2 = wb * wb;
                    int wbPow3 = wbPow2 * wb;

                    if (i < m && j < n && k < o)
                    {
                        // clang-format off
                        for (int ci = 0; ci < wc; ci++)
                        for (int cj = 0; cj < wc; cj++)
                        for (int ck = 0; ck < wc; ck++)
                        {
                            double csum = 0;
                            for (int a_i = ci - (min(ci, wb - 1)), b_i = min(ci, wb - 1);
                                a_i <= min(ci, wa - 1) && b_i >= ci - min(ci, wa - 1);
                                a_i++, b_i--)
                            for (int a_j = cj - (min(cj, wb - 1)), b_j = min(cj, wb - 1);
                                    a_j <= min(cj, wa - 1) && b_j >= cj - min(cj, wa - 1);
                                    a_j++, b_j--)
                            for (int a_k = ck - (min(ck, wb - 1)), b_k = min(ck, wb - 1);
                                    a_k <= min(ck, wa - 1) && b_k >= ck - min(ck, wa - 1);
                                    a_k++, b_k--)
                            {
                                int gpi = i + a_i - wa2 + ghb;
                                int gpj = j + a_j - wa2 + ghb;
                                int gpk = k + a_k - wa2 + ghb;
                                
                                int cell_b = gpi * (n + 2 * ghb) * (o + 2 * ghb) + gpj * (o + 2 * ghb) + gpk;

                                // index of a coefficient is equal to the index of the coefficient block plus grid point offset.
                                int idx_a = cell_a + (a_i * waPow2 + a_j * wa + a_k) * gridsize_a;
                                int idx_b = cell_b + (b_i * wbPow2 + b_j * wb + b_k) * gridsize_b;

                                REQUIRE(c.field1d()[cell_c + (ci * wcPow2 + cj * wc + ck) * gridsize_c] == c[ci][cj][ck][i + ghc][j + ghc][k + ghc]);
                                REQUIRE(a.field1d()[idx_a] == a[a_i][a_j][a_k][i + gha][j + gha][k + gha]);
                                REQUIRE(b.field1d()[idx_b] == b[b_i][b_j][b_k][gpi][gpj][gpk]);
                            }
                        }
                        // clang-format on
                    }
                }
    }

    SECTION("widths 3 * 3")
    {
        auto a = std::make_unique<mgcl::VaryingStencilGpu>(m, n, o, 3, gh, t.getContext(), t.getCommands(), t.getProgram());
        auto b = std::make_unique<mgcl::VaryingStencilGpu>(m, n, o, 3, gh, t.getContext(), t.getCommands(), t.getProgram());
        t.finish();

        // create VaryingStencil, fill with random values and copy to gpu buffer
        mgcl::VaryingStencil a_h(m, n, o, 3, gh, gh, gh);
        a_h.fillRandomInt();
        a->fill(a_h, t.getCommands(), true);

        mgcl::VaryingStencil b_h(m, n, o, 3, gh, gh, gh);
        b_h.fillRandomInt();
        b->fill(b_h, t.getCommands(), true);
        t.finish();
        b_h.updateGhosts();
        b->updateGhosts(t.getProgram(), t.getCommands(), nullptr, nullptr);

        auto c_h = a_h.multiply(b_h, 2, nullptr, true, true);
        auto c = a->multiply(*b, 2, t.getProgram(), t.getCommands(), t.getContext(), nullptr, true, true, nullptr, nullptr);
        t.finish();

        auto ret = c.read(t.getCommands(), true);
        t.finish();

        // check results
        REQUIRE(c.getM() == c_h.getM());
        REQUIRE(c.getN() == c_h.getN());
        REQUIRE(c.getO() == c_h.getO());
        REQUIRE(c.getWidth() == c_h.getWidth());
        REQUIRE(c.getWidth() == c_h.getWidth());
        REQUIRE(c.getWidth() == c_h.getWidth());
        REQUIRE(c.getGh() == c_h.getGhostsM());
        REQUIRE(c.getGh() == c_h.getGhostsN());
        REQUIRE(c.getGh() == c_h.getGhostsO());

        // clang-format off
        for (int i = 0; i < c_h.getMgh(); i++)
        for (int j = 0; j < c_h.getNgh(); j++)
        for (int k = 0; k < c_h.getOgh(); k++)
            for (int ii = 0; ii < c_h.getWidth(); ii++)
            for (int jj = 0; jj < c_h.getWidth(); jj++)
            for (int kk = 0; kk < c_h.getWidth(); kk++)
            {
                REQUIRE(c_h[ii][jj][kk][i][j][k] == ret[ii][jj][kk][i][j][k]);
            }
        // clang-format on
    }

    SECTION("widths 3 * 5")
    {
        auto a = std::make_unique<mgcl::VaryingStencilGpu>(m, n, o, 3, gh, t.getContext(), t.getCommands(), t.getProgram());
        auto b = std::make_unique<mgcl::VaryingStencilGpu>(m, n, o, 5, gh, t.getContext(), t.getCommands(), t.getProgram());
        t.finish();

        // create VaryingStencil, fill with random values and copy to gpu buffer
        mgcl::VaryingStencil a_h(m, n, o, 3, gh, gh, gh);
        a_h.fillRandomInt();
        a->fill(a_h, t.getCommands(), true);

        mgcl::VaryingStencil b_h(m, n, o, 5, gh, gh, gh);
        b_h.fillRandomInt();
        b->fill(b_h, t.getCommands(), true);
        t.finish();
        b_h.updateGhosts();
        b->updateGhosts(t.getProgram(), t.getCommands(), nullptr, nullptr);

        auto c_h = a_h.multiply(b_h, 2, nullptr, true, true);
        auto c = a->multiply(*b, 2, t.getProgram(), t.getCommands(), t.getContext(), nullptr, true, true, nullptr, nullptr);
        t.finish();

        auto ret = c.read(t.getCommands(), true);
        t.finish();

        // check results
        REQUIRE(c.getM() == c_h.getM());
        REQUIRE(c.getN() == c_h.getN());
        REQUIRE(c.getO() == c_h.getO());
        REQUIRE(c.getWidth() == c_h.getWidth());
        REQUIRE(c.getWidth() == c_h.getWidth());
        REQUIRE(c.getWidth() == c_h.getWidth());
        REQUIRE(c.getGh() == c_h.getGhostsM());
        REQUIRE(c.getGh() == c_h.getGhostsN());
        REQUIRE(c.getGh() == c_h.getGhostsO());

        // clang-format off
        for (int i = 0; i < c_h.getMgh(); i++)
        for (int j = 0; j < c_h.getNgh(); j++)
        for (int k = 0; k < c_h.getOgh(); k++)
            for (int ii = 0; ii < c_h.getWidth(); ii++)
            for (int jj = 0; jj < c_h.getWidth(); jj++)
            for (int kk = 0; kk < c_h.getWidth(); kk++)
            {
                REQUIRE(c_h[ii][jj][kk][i][j][k] == ret[ii][jj][kk][i][j][k]);
            }
        // clang-format on
    }

    SECTION("widths 5 * 3")
    {
        auto a = std::make_unique<mgcl::VaryingStencilGpu>(m, n, o, 5, gh, t.getContext(), t.getCommands(), t.getProgram());
        auto b = std::make_unique<mgcl::VaryingStencilGpu>(m, n, o, 3, gh, t.getContext(), t.getCommands(), t.getProgram());
        t.finish();

        // create VaryingStencil, fill with random values and copy to gpu buffer
        mgcl::VaryingStencil a_h(m, n, o, 5, gh, gh, gh);
        a_h.fillRandomInt();
        a->fill(a_h, t.getCommands(), true);

        mgcl::VaryingStencil b_h(m, n, o, 3, gh, gh, gh);
        b_h.fillRandomInt();
        b->fill(b_h, t.getCommands(), true);
        t.finish();
        b_h.updateGhosts();
        b->updateGhosts(t.getProgram(), t.getCommands(), nullptr, nullptr);

        auto c_h = a_h.multiply(b_h, 2, nullptr, true, true);
        auto c = a->multiply(*b, 2, t.getProgram(), t.getCommands(), t.getContext(), nullptr, true, true, nullptr, nullptr);
        t.finish();

        auto ret = c.read(t.getCommands(), true);
        t.finish();

        // check results
        REQUIRE(c.getM() == c_h.getM());
        REQUIRE(c.getN() == c_h.getN());
        REQUIRE(c.getO() == c_h.getO());
        REQUIRE(c.getWidth() == c_h.getWidth());
        REQUIRE(c.getWidth() == c_h.getWidth());
        REQUIRE(c.getWidth() == c_h.getWidth());
        REQUIRE(c.getGh() == c_h.getGhostsM());
        REQUIRE(c.getGh() == c_h.getGhostsN());
        REQUIRE(c.getGh() == c_h.getGhostsO());

        // clang-format off
        for (int i = 0; i < c_h.getMgh(); i++)
        for (int j = 0; j < c_h.getNgh(); j++)
        for (int k = 0; k < c_h.getOgh(); k++)
            for (int ii = 0; ii < c_h.getWidth(); ii++)
            for (int jj = 0; jj < c_h.getWidth(); jj++)
            for (int kk = 0; kk < c_h.getWidth(); kk++)
            {
                REQUIRE(c_h[ii][jj][kk][i][j][k] == ret[ii][jj][kk][i][j][k]);
            }
        // clang-format on
    }

    SECTION("widths 5 * 5")
    {
        auto a = std::make_unique<mgcl::VaryingStencilGpu>(m, n, o, 5, gh, t.getContext(), t.getCommands(), t.getProgram());
        auto b = std::make_unique<mgcl::VaryingStencilGpu>(m, n, o, 5, gh, t.getContext(), t.getCommands(), t.getProgram());
        t.finish();

        // create VaryingStencil, fill with random values and copy to gpu buffer
        mgcl::VaryingStencil a_h(m, n, o, 5, gh, gh, gh);
        a_h.fillRandomInt();
        a->fill(a_h, t.getCommands(), true);

        mgcl::VaryingStencil b_h(m, n, o, 5, gh, gh, gh);
        b_h.fillRandomInt();
        b->fill(b_h, t.getCommands(), true);
        t.finish();
        b_h.updateGhosts();
        b->updateGhosts(t.getProgram(), t.getCommands(), nullptr, nullptr);

        auto c_h = a_h.multiply(b_h, 2, nullptr, true, true);
        auto c = a->multiply(*b, 2, t.getProgram(), t.getCommands(), t.getContext(), nullptr, true, true, nullptr, nullptr);
        t.finish();

        auto ret = c.read(t.getCommands(), true);
        t.finish();

        // check results
        REQUIRE(c.getM() == c_h.getM());
        REQUIRE(c.getN() == c_h.getN());
        REQUIRE(c.getO() == c_h.getO());
        REQUIRE(c.getWidth() == c_h.getWidth());
        REQUIRE(c.getWidth() == c_h.getWidth());
        REQUIRE(c.getWidth() == c_h.getWidth());
        REQUIRE(c.getGh() == c_h.getGhostsM());
        REQUIRE(c.getGh() == c_h.getGhostsN());
        REQUIRE(c.getGh() == c_h.getGhostsO());

        // clang-format off
        for (int i = 0; i < c_h.getMgh(); i++)
        for (int j = 0; j < c_h.getNgh(); j++)
        for (int k = 0; k < c_h.getOgh(); k++)
            for (int ii = 0; ii < c_h.getWidth(); ii++)
            for (int jj = 0; jj < c_h.getWidth(); jj++)
            for (int kk = 0; kk < c_h.getWidth(); kk++)
            {
                REQUIRE(c_h[ii][jj][kk][i][j][k] == ret[ii][jj][kk][i][j][k]);
            }
        // clang-format on
    }
}

TEST_CASE("VaryingStencilGpu::multiply(fix)")
{
    auto deviceType = CL_DEVICE_TYPE_GPU; // GENERATE(CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU);

    if (!mgcl_test::TestUtility::deviceAvailable("", deviceType))
    {
        std::string typeName = deviceType == CL_DEVICE_TYPE_GPU ? "CL_DEVICE_TYPE_GPU" : "CL_DEVICE_TYPE_CPU";
        std::cout << "Skipping non-available device type '" << typeName << "'" << std::endl;
        return;
    }

    mgcl_test::TestUtility t(deviceType);

    int m = 4;
    int n = 4;
    int o = 4;
    int gh = 2;

    SECTION("widths 3 * 3")
    {
        auto a = std::make_unique<mgcl::VaryingStencilGpu>(m, n, o, 3, gh, t.getContext(), t.getCommands(), t.getProgram());
        auto b = std::make_unique<mgcl::FixedStencilGpu>(3, t.getContext(), t.getCommands(), t.getProgram());
        t.finish();

        // create VaryingStencil, fill with random values and copy to gpu buffer
        mgcl::VaryingStencil a_h(m, n, o, 3, gh, gh, gh);
        a_h.fillRandomInt();
        a->fill(a_h, t.getCommands(), true);

        mgcl::FixedStencil b_h(3);
        b_h.fillRandomInt();
        b->fill(b_h, t.getCommands(), true);
        t.finish();

        auto c_h = a_h.multiply(b_h, 2, nullptr, true, true);
        auto c = a->multiply(*b, 2, t.getProgram(), t.getCommands(), t.getContext(), nullptr, true, true, nullptr, nullptr);
        t.finish();

        auto ret = c.read(t.getCommands(), true);
        t.finish();

        // check results
        REQUIRE(c.getM() == c_h.getM());
        REQUIRE(c.getN() == c_h.getN());
        REQUIRE(c.getO() == c_h.getO());
        REQUIRE(c.getWidth() == c_h.getWidth());
        REQUIRE(c.getWidth() == c_h.getWidth());
        REQUIRE(c.getWidth() == c_h.getWidth());
        REQUIRE(c.getGh() == c_h.getGhostsM());
        REQUIRE(c.getGh() == c_h.getGhostsN());
        REQUIRE(c.getGh() == c_h.getGhostsO());

        // clang-format off
        for (int i = 0; i < c_h.getMgh(); i++)
        for (int j = 0; j < c_h.getNgh(); j++)
        for (int k = 0; k < c_h.getOgh(); k++)
            for (int ii = 0; ii < c_h.getWidth(); ii++)
            for (int jj = 0; jj < c_h.getWidth(); jj++)
            for (int kk = 0; kk < c_h.getWidth(); kk++)
            {
                REQUIRE(c_h[ii][jj][kk][i][j][k] == ret[ii][jj][kk][i][j][k]);
            }
        // clang-format on
    }

    SECTION("widths 3 * 5")
    {
        auto a = std::make_unique<mgcl::VaryingStencilGpu>(m, n, o, 3, gh, t.getContext(), t.getCommands(), t.getProgram());
        auto b = std::make_unique<mgcl::FixedStencilGpu>(5, t.getContext(), t.getCommands(), t.getProgram());
        t.finish();

        // create VaryingStencil, fill with random values and copy to gpu buffer
        mgcl::VaryingStencil a_h(m, n, o, 3, gh, gh, gh);
        a_h.fillRandomInt();
        a->fill(a_h, t.getCommands(), true);

        mgcl::FixedStencil b_h(5);
        b_h.fillRandomInt();
        b->fill(b_h, t.getCommands(), true);
        t.finish();

        auto c_h = a_h.multiply(b_h, 2, nullptr, true, true);
        auto c = a->multiply(*b, 2, t.getProgram(), t.getCommands(), t.getContext(), nullptr, true, true, nullptr, nullptr);
        t.finish();

        auto ret = c.read(t.getCommands(), true);
        t.finish();

        // check results
        REQUIRE(c.getM() == c_h.getM());
        REQUIRE(c.getN() == c_h.getN());
        REQUIRE(c.getO() == c_h.getO());
        REQUIRE(c.getWidth() == c_h.getWidth());
        REQUIRE(c.getWidth() == c_h.getWidth());
        REQUIRE(c.getWidth() == c_h.getWidth());
        REQUIRE(c.getGh() == c_h.getGhostsM());
        REQUIRE(c.getGh() == c_h.getGhostsN());
        REQUIRE(c.getGh() == c_h.getGhostsO());

        // clang-format off
        for (int i = 0; i < c_h.getMgh(); i++)
        for (int j = 0; j < c_h.getNgh(); j++)
        for (int k = 0; k < c_h.getOgh(); k++)
            for (int ii = 0; ii < c_h.getWidth(); ii++)
            for (int jj = 0; jj < c_h.getWidth(); jj++)
            for (int kk = 0; kk < c_h.getWidth(); kk++)
            {
                REQUIRE(c_h[ii][jj][kk][i][j][k] == ret[ii][jj][kk][i][j][k]);
            }
        // clang-format on
    }

    SECTION("widths 5 * 3")
    {
        auto a = std::make_unique<mgcl::VaryingStencilGpu>(m, n, o, 5, gh, t.getContext(), t.getCommands(), t.getProgram());
        auto b = std::make_unique<mgcl::FixedStencilGpu>(3, t.getContext(), t.getCommands(), t.getProgram());
        t.finish();

        // create VaryingStencil, fill with random values and copy to gpu buffer
        mgcl::VaryingStencil a_h(m, n, o, 5, gh, gh, gh);
        a_h.fillRandomInt();
        a->fill(a_h, t.getCommands(), true);

        mgcl::FixedStencil b_h(3);
        b_h.fillRandomInt();
        b->fill(b_h, t.getCommands(), true);
        t.finish();

        auto c_h = a_h.multiply(b_h, 2, nullptr, true, true);
        auto c = a->multiply(*b, 2, t.getProgram(), t.getCommands(), t.getContext(), nullptr, true, true, nullptr, nullptr);
        t.finish();

        auto ret = c.read(t.getCommands(), true);
        t.finish();

        // check results
        REQUIRE(c.getM() == c_h.getM());
        REQUIRE(c.getN() == c_h.getN());
        REQUIRE(c.getO() == c_h.getO());
        REQUIRE(c.getWidth() == c_h.getWidth());
        REQUIRE(c.getWidth() == c_h.getWidth());
        REQUIRE(c.getWidth() == c_h.getWidth());
        REQUIRE(c.getGh() == c_h.getGhostsM());
        REQUIRE(c.getGh() == c_h.getGhostsN());
        REQUIRE(c.getGh() == c_h.getGhostsO());

        // clang-format off
        for (int i = 0; i < c_h.getMgh(); i++)
        for (int j = 0; j < c_h.getNgh(); j++)
        for (int k = 0; k < c_h.getOgh(); k++)
            for (int ii = 0; ii < c_h.getWidth(); ii++)
            for (int jj = 0; jj < c_h.getWidth(); jj++)
            for (int kk = 0; kk < c_h.getWidth(); kk++)
            {
                REQUIRE(c_h[ii][jj][kk][i][j][k] == ret[ii][jj][kk][i][j][k]);
            }
        // clang-format on
    }

    SECTION("widths 5 * 5")
    {
        auto a = std::make_unique<mgcl::VaryingStencilGpu>(m, n, o, 5, gh, t.getContext(), t.getCommands(), t.getProgram());
        auto b = std::make_unique<mgcl::FixedStencilGpu>(5, t.getContext(), t.getCommands(), t.getProgram());
        t.finish();

        // create VaryingStencil, fill with random values and copy to gpu buffer
        mgcl::VaryingStencil a_h(m, n, o, 5, gh, gh, gh);
        a_h.fillRandomInt();
        a->fill(a_h, t.getCommands(), true);

        mgcl::FixedStencil b_h(5);
        b_h.fillRandomInt();
        b->fill(b_h, t.getCommands(), true);
        t.finish();

        auto c_h = a_h.multiply(b_h, 2, nullptr, true, true);
        auto c = a->multiply(*b, 2, t.getProgram(), t.getCommands(), t.getContext(), nullptr, true, true, nullptr, nullptr);
        t.finish();

        auto ret = c.read(t.getCommands(), true);
        t.finish();

        // check results
        REQUIRE(c.getM() == c_h.getM());
        REQUIRE(c.getN() == c_h.getN());
        REQUIRE(c.getO() == c_h.getO());
        REQUIRE(c.getWidth() == c_h.getWidth());
        REQUIRE(c.getWidth() == c_h.getWidth());
        REQUIRE(c.getWidth() == c_h.getWidth());
        REQUIRE(c.getGh() == c_h.getGhostsM());
        REQUIRE(c.getGh() == c_h.getGhostsN());
        REQUIRE(c.getGh() == c_h.getGhostsO());

        // clang-format off
        for (int i = 0; i < c_h.getMgh(); i++)
        for (int j = 0; j < c_h.getNgh(); j++)
        for (int k = 0; k < c_h.getOgh(); k++)
            for (int ii = 0; ii < c_h.getWidth(); ii++)
            for (int jj = 0; jj < c_h.getWidth(); jj++)
            for (int kk = 0; kk < c_h.getWidth(); kk++)
            {
                REQUIRE(c_h[ii][jj][kk][i][j][k] == ret[ii][jj][kk][i][j][k]);
            }
        // clang-format on
    }
}

// Tests if indices are still correct for reordered loops (which make possible to write to global c only once).
// The test is done only for one cell, since it's independent of the actual cell.
TEST_CASE("VaryingStencilGpu::multiply loop index reordering", "[stencilIndexReordering]")
{

    int wa = GENERATE(3, 5);
    int wb = GENERATE(3, 5);
    int wc = wa + wb - 1;

    // Test results are stored in 2d arrays where one row is one iteration having the columns:
    // ai aj ak bi bj bk ai+bi aj+bj ak+bk 1didx
    std::vector<std::vector<int>> indices_original;
    std::vector<std::vector<int>> indices_new;
    std::vector<std::vector<int>> indices_new_minfn;

    // original
    // clang-format off
    for (int a_i = 0; a_i < wa; a_i++)
    for (int a_j = 0; a_j < wa; a_j++)
    for (int a_k = 0; a_k < wa; a_k++)
        for (int b_i = 0; b_i < wb; b_i++)
        for (int b_j = 0; b_j < wb; b_j++)
        for (int b_k = 0; b_k < wb; b_k++)
        {
            int ci = a_i + b_i;
            int cj = a_j + b_j;
            int ck = a_k + b_k;
            int idx1d = ci * wc * wc + cj * wc + ck;

            indices_original.push_back(std::vector<int> {
                a_i, a_j, a_k,
                b_i, b_j, b_k,
                ci, cj, ck,
                idx1d
            });
        }
    // clang-format on

    // reordered
    // clang-format off
    for (int ci = 0; ci < wc; ci++)
    for (int cj = 0; cj < wc; cj++)
    for (int ck = 0; ck < wc; ck++)
        for (int a_i = ci - (ci < (wb - 1) ? ci : (wb - 1)), b_i = (ci < (wb - 1) ? ci : (wb - 1));
             a_i <= (ci < (wa - 1) ? ci : (wa - 1)) && b_i >= ci - (ci < (wa - 1) ? ci : (wa - 1)); 
             a_i++, b_i--)
        for (int a_j = cj - (cj < (wb - 1) ? cj : (wb - 1)), b_j = (cj < (wb - 1) ? cj : (wb - 1));
             a_j <= (cj < (wa - 1) ? cj : (wa - 1)) && b_j >= cj - (cj < (wa - 1) ? cj : (wa - 1)); 
             a_j++, b_j--)
        for (int a_k = ck - (ck < (wb - 1) ? ck : (wb - 1)), b_k = (ck < (wb - 1) ? ck : (wb - 1));
             a_k <= (ck < (wa - 1) ? ck : (wa - 1)) && b_k >= ck - (ck < (wa - 1) ? ck : (wa - 1)); 
             a_k++, b_k--)
        {
            int idx1d = ci * wc * wc + cj * wc + ck;

            indices_new.push_back(std::vector<int> {
                a_i, a_j, a_k,
                b_i, b_j, b_k,
                ci, cj, ck,
                idx1d
            });

            // printf("%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n", a_i, a_j, a_k, b_i, b_j, b_k, ci, cj, ck,idx1d);
        }
    // clang-format on

    // reordered using min function
    // clang-format off
    for (int ci = 0; ci < wc; ci++)
    for (int cj = 0; cj < wc; cj++)
    for (int ck = 0; ck < wc; ck++)
        for (int a_i = ci - (min(ci, wb - 1)), b_i = min(ci, wb - 1);
                a_i <= min(ci, wa - 1) && b_i >= ci - min(ci, wa - 1);
                a_i++, b_i--)
        for (int a_j = cj - (min(cj, wb - 1)), b_j = min(cj, wb - 1);
                a_j <= min(cj, wa - 1) && b_j >= cj - min(cj, wa - 1);
                a_j++, b_j--)
        for (int a_k = ck - (min(ck, wb - 1)), b_k = min(ck, wb - 1);
                a_k <= min(ck, wa - 1) && b_k >= ck - min(ck, wa - 1);
                a_k++, b_k--)
        {
            int idx1d = ci * wc * wc + cj * wc + ck;

            indices_new_minfn.push_back(std::vector<int> {
                a_i, a_j, a_k,
                b_i, b_j, b_k,
                ci, cj, ck,
                idx1d
            });

            printf("%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n", a_i, a_j, a_k, b_i, b_j, b_k, ci, cj, ck,idx1d);
        }
    // clang-format on

    REQUIRE(indices_original.size() == indices_new.size());
    REQUIRE(indices_original.size() == indices_new_minfn.size());

    // sort both lists to make comparison easier
    // clang-format off
    std::stable_sort(std::begin(indices_original), std::end(indices_original), [](const auto &u, const auto &v){ return u[0] < v[0]; });
    std::stable_sort(std::begin(indices_original), std::end(indices_original), [](const auto &u, const auto &v){ return u[1] < v[1]; });
    std::stable_sort(std::begin(indices_original), std::end(indices_original), [](const auto &u, const auto &v){ return u[2] < v[2]; });
    std::stable_sort(std::begin(indices_original), std::end(indices_original), [](const auto &u, const auto &v){ return u[3] < v[3]; });
    std::stable_sort(std::begin(indices_original), std::end(indices_original), [](const auto &u, const auto &v){ return u[4] < v[4]; });
    std::stable_sort(std::begin(indices_original), std::end(indices_original), [](const auto &u, const auto &v){ return u[5] < v[5]; });
    std::stable_sort(std::begin(indices_new), std::end(indices_new), [](const auto &u, const auto &v){ return u[0] < v[0]; });
    std::stable_sort(std::begin(indices_new), std::end(indices_new), [](const auto &u, const auto &v){ return u[1] < v[1]; });
    std::stable_sort(std::begin(indices_new), std::end(indices_new), [](const auto &u, const auto &v){ return u[2] < v[2]; });
    std::stable_sort(std::begin(indices_new), std::end(indices_new), [](const auto &u, const auto &v){ return u[3] < v[3]; });
    std::stable_sort(std::begin(indices_new), std::end(indices_new), [](const auto &u, const auto &v){ return u[4] < v[4]; });
    std::stable_sort(std::begin(indices_new), std::end(indices_new), [](const auto &u, const auto &v){ return u[5] < v[5]; });
    std::stable_sort(std::begin(indices_new_minfn), std::end(indices_new_minfn), [](const auto &u, const auto &v){ return u[0] < v[0]; });
    std::stable_sort(std::begin(indices_new_minfn), std::end(indices_new_minfn), [](const auto &u, const auto &v){ return u[1] < v[1]; });
    std::stable_sort(std::begin(indices_new_minfn), std::end(indices_new_minfn), [](const auto &u, const auto &v){ return u[2] < v[2]; });
    std::stable_sort(std::begin(indices_new_minfn), std::end(indices_new_minfn), [](const auto &u, const auto &v){ return u[3] < v[3]; });
    std::stable_sort(std::begin(indices_new_minfn), std::end(indices_new_minfn), [](const auto &u, const auto &v){ return u[4] < v[4]; });
    std::stable_sort(std::begin(indices_new_minfn), std::end(indices_new_minfn), [](const auto &u, const auto &v){ return u[5] < v[5]; });
    // clang-format on

    for (int i = 0; i < indices_original.size(); i++)
    {
        REQUIRE(indices_original[i].size() == indices_new[i].size());
        REQUIRE(indices_original[i].size() == indices_new_minfn[i].size());
        for (int j = 0; j < indices_original[0].size(); j++)
        {
            REQUIRE(indices_original[i][j] == indices_new[i][j]);
            REQUIRE(indices_original[i][j] == indices_new_minfn[i][j]);
        }
    }
}

TEST_CASE("VaryingStencilGpu::cutFromW7ToW3")
{
    auto deviceType = CL_DEVICE_TYPE_GPU; // GENERATE(CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU);

    if (!mgcl_test::TestUtility::deviceAvailable("", deviceType))
    {
        std::string typeName = deviceType == CL_DEVICE_TYPE_GPU ? "CL_DEVICE_TYPE_GPU" : "CL_DEVICE_TYPE_CPU";
        std::cout << "Skipping non-available device type '" << typeName << "'" << std::endl;
        return;
    }

    mgcl_test::TestUtility t(deviceType);

    int m = 8;
    int n = 8;
    int o = 8;
    int gh = 2;

    // This section checks if the indices inside the kernel are correctly reduced to 1d.
    // Here, a_2h has indeed half the grid size of a_h
    SECTION("indices_regular")
    {
        // create test stencils and fill with unique values
        mgcl::VaryingStencil a_h(2 * m, 2 * n, 2 * o, 7, 0, 0, 0);
        mgcl::VaryingStencil a_2h(m, n, o, 3, gh, gh, gh);
        for (int i = 0; i < a_h.field1d().size(); i++)
            a_h.field1d()[i] = i;
        for (int i = 0; i < a_2h.field1d().size(); i++)
            a_2h.field1d()[i] = i;

        int ghin = 0;
        int ghout = 2;

        int nghout = (n + 2 * ghout);
        int oghout = (o + 2 * ghout);

        // clang-format off
        // simulate call with one work-item per cell using these 3 for loops
        for (int i = 2; i < a_2h.getM() + 2; i++)
        for (int j = 2; j < a_2h.getN() + 2; j++)
        for (int k = 2; k < a_2h.getO() + 2; k++)
        {
            int i2 = (i - 2) * 2 + 1;
            int j2 = (j - 2) * 2 + 1;
            int k2 = (k - 2) * 2 + 1;

            int gridsize_fine = (2 * m + 2 * ghin) * (2 * n + 2 * ghin) * (2 * o + 2 * ghin);
            int gridsize_coarse = (m + 2 * ghout) * nghout * oghout;

            // grid point offsets
            int cell_fine = i2 * (2 * n + 2 * ghin) * (2 * o + 2 * ghin) + j2 * (2 * o + 2 * ghin) + k2;
            int cell_coarse = i * nghout * oghout + j * oghout + k;

            // starting index for coarse grid
            int idx_coarse = cell_coarse;

            for (int ii = 0, ii2 = 1; ii < 3; ii++, ii2 += 2)
            for (int jj = 0, jj2 = 1; jj < 3; jj++, jj2 += 2)
            for (int kk = 0, kk2 = 1; kk < 3; kk++, kk2 += 2)
            {
                CAPTURE(i, j, k, ii, jj, kk, gridsize_fine, gridsize_coarse, cell_fine, cell_coarse, idx_coarse);
                // REQUIRE(cell_fine + ( ii2 * 49 + jj2 * 7 + kk2 ) * gridsize_fine == idx_fine);
                REQUIRE(cell_coarse + ( ii * 9 + jj * 3 + kk ) * gridsize_coarse == idx_coarse);
                REQUIRE(a_2h[ii][jj][kk][i][j][k] == a_2h.field1d()[idx_coarse]);
                REQUIRE(a_h[ii2][jj2][kk2][i2][j2][k2] == a_h.field1d()[cell_fine + (ii2 * 49 + jj2 * 7 + kk2) * gridsize_fine]);
                // idx_fine += gridsize_fine * 2;
                idx_coarse += gridsize_coarse;
            }
        }
        // clang-format on
    }

    // This section checks if the actual calculation is correct by checking results vs the sequential version.
    SECTION("vs seq")
    {
        mgcl::VaryingStencilGpu a_gpu(m, n, o, 7, gh, t.getContext(), t.getCommands(), t.getProgram());

        // create VaryingStencil, fill with random values and copy to gpu buffer
        mgcl::VaryingStencil a_h(m, n, o, 7, gh, gh, gh);
        a_h.fillRandomInt();
        a_gpu.fill(a_h, t.getCommands(), true);
        t.finish();

        // Cut stencil from 7x7x7 down to 3x3x3, i.e. copy only selected values to new stencil, skipping ghosts.
        mgcl::VaryingStencil a_2h(a_h.getM() >> 1, a_h.getN() >> 1, a_h.getO() >> 1, 3, 2, 2, 2);
        // clang-format off
        for (int i = 2, i2 = 1; i < a_2h.getM() + 2; i++, i2 += 2)
        for (int j = 2, j2 = 1; j < a_2h.getN() + 2; j++, j2 += 2)
        for (int k = 2, k2 = 1; k < a_2h.getO() + 2; k++, k2 += 2)
            for (int ii = 0, ii2 = 1; ii < 3; ii++, ii2 += 2)
            for (int jj = 0, jj2 = 1; jj < 3; jj++, jj2 += 2)
            for (int kk = 0, kk2 = 1; kk < 3; kk++, kk2 += 2)
            {
                a_2h[ii][jj][kk][i][j][k] = a_h[ii2][jj2][kk2][i2][j2][k2];
            }
        // clang-format on

        // cut on gpu and read back result
        auto a_2h_gpu = a_gpu.cutFromW7ToW3(t.getProgram(), t.getCommands(), t.getContext(), 2, nullptr, nullptr);
        auto ret = a_2h_gpu.read(t.getCommands(), true);
        t.finish();

        REQUIRE(ret.getM() == a_2h.getM());
        REQUIRE(ret.getN() == a_2h.getN());
        REQUIRE(ret.getO() == a_2h.getO());
        REQUIRE(ret.getWidth() == a_2h.getWidth());
        REQUIRE(ret.getWidth() == a_2h.getWidth());
        REQUIRE(ret.getWidth() == a_2h.getWidth());
        REQUIRE(ret.getMgh() == a_2h.getMgh());
        REQUIRE(ret.getNgh() == a_2h.getNgh());
        REQUIRE(ret.getOgh() == a_2h.getOgh());

        REQUIRE(ret.isEqual(a_2h));
    }

    // This section checks if the actual calculation is correct by checking results vs the sequential version.
    // Special case when using mpi, where the resulting stencil should *not* be half of the fine stencil (happens when
    // the level is just above the threshold on root).
    SECTION("vs_seq_special_case_mpi")
    {
        int resm = m;
        int resn = n;
        int reso = o;

        mgcl::VaryingStencilGpu a_gpu(m, n, o, 7, gh, t.getContext(), t.getCommands(), t.getProgram());

        // create VaryingStencil, fill with random values and copy to gpu buffer
        mgcl::VaryingStencil a_h(m, n, o, 7, gh, gh, gh);
        a_h.fillRandomInt();
        a_gpu.fill(a_h, t.getCommands(), true);
        t.finish();

        // Cut stencil from 7x7x7 down to 3x3x3, i.e. copy only selected values to new stencil, skipping ghosts.
        mgcl::VaryingStencil a_2h(resm, resn, reso, 3, 2, 2, 2);
        // clang-format off
        for (int i = 2, i2 = 1; i < (a_h.getM() >> 1) + 2; i++, i2 += 2)
        for (int j = 2, j2 = 1; j < (a_h.getN() >> 1) + 2; j++, j2 += 2)
        for (int k = 2, k2 = 1; k < (a_h.getO() >> 1) + 2; k++, k2 += 2)
            for (int ii = 0, ii2 = 1; ii < 3; ii++, ii2 += 2)
            for (int jj = 0, jj2 = 1; jj < 3; jj++, jj2 += 2)
            for (int kk = 0, kk2 = 1; kk < 3; kk++, kk2 += 2)
            {
                a_2h[ii][jj][kk][i][j][k] = a_h[ii2][jj2][kk2][i2][j2][k2];
            }
        // clang-format on

        // cut on gpu and read back result
        auto a_2h_gpu = a_gpu.cutFromW7ToW3(t.getProgram(), t.getCommands(), t.getContext(), 2, nullptr, nullptr, resm, resn, reso);
        auto ret = a_2h_gpu.read(t.getCommands(), true);
        t.finish();

        REQUIRE(ret.getM() == a_2h.getM());
        REQUIRE(ret.getN() == a_2h.getN());
        REQUIRE(ret.getO() == a_2h.getO());
        REQUIRE(ret.getWidth() == a_2h.getWidth());
        REQUIRE(ret.getWidth() == a_2h.getWidth());
        REQUIRE(ret.getWidth() == a_2h.getWidth());
        REQUIRE(ret.getMgh() == a_2h.getMgh());
        REQUIRE(ret.getNgh() == a_2h.getNgh());
        REQUIRE(ret.getOgh() == a_2h.getOgh());

        REQUIRE(ret.isEqual(a_2h));
    }
}

TEST_CASE("FixedStencilGpu ctor+dtor")
{
    auto deviceType = CL_DEVICE_TYPE_GPU; // GENERATE(CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU);

    if (!mgcl_test::TestUtility::deviceAvailable("", deviceType))
    {
        std::string typeName = deviceType == CL_DEVICE_TYPE_GPU ? "CL_DEVICE_TYPE_GPU" : "CL_DEVICE_TYPE_CPU";
        std::cout << "Skipping non-available device type '" << typeName << "'" << std::endl;
        return;
    }

    mgcl_test::TestUtility t(deviceType);

    int width = 3;
    int gh = 2;
    auto s = std::make_unique<mgcl::FixedStencilGpu>(width, t.getContext(), t.getCommands(), t.getProgram());
    t.finish();

    REQUIRE(s->getBuf());

    // check size of buffer
    size_t bufsize;
    int sizeNeeded = width * width * width;
    int err = clGetMemObjectInfo(s->getBuf(), CL_MEM_SIZE, sizeof(size_t), &bufsize, nullptr);
    mgcl::mgclCheckError(err, "Querying buffer size");
    REQUIRE(bufsize == sizeof(double) * sizeNeeded);

    // check reference count, should be 1
    cl_uint refCount;

    err = clGetMemObjectInfo(s->getBuf(), CL_MEM_REFERENCE_COUNT, sizeof(cl_uint), &refCount, nullptr);
    mgcl::mgclCheckError(err, "clGetMemObjectInfo(s, CL_MEM_REFERENCE_COUNT)");
    REQUIRE(refCount == 1);

    // check values are 0
    double tmp[sizeNeeded];
    err = clEnqueueReadBuffer(t.getCommands(), s->getBuf(), CL_TRUE, 0,
                              sizeof(double) * sizeNeeded, tmp, 0, NULL, NULL);
    mgcl::mgclCheckError(err, "clEnqueueReadBuffer");

    for (int i = 0; i < sizeNeeded; i++)
        REQUIRE(tmp[i] == 0);

    // delete s
    s.reset();
}

TEST_CASE("FixedStencilGpu::multiply(var)")
{
    auto deviceType = CL_DEVICE_TYPE_GPU; // GENERATE(CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU);

    if (!mgcl_test::TestUtility::deviceAvailable("", deviceType))
    {
        std::string typeName = deviceType == CL_DEVICE_TYPE_GPU ? "CL_DEVICE_TYPE_GPU" : "CL_DEVICE_TYPE_CPU";
        std::cout << "Skipping non-available device type '" << typeName << "'" << std::endl;
        return;
    }

    mgcl_test::TestUtility t(deviceType);

    int m = 4;
    int n = 4;
    int o = 4;
    int gh = 2;

    SECTION("widths 3 * 3")
    {
        auto a = std::make_unique<mgcl::FixedStencilGpu>(3, t.getContext(), t.getCommands(), t.getProgram());
        auto b = std::make_unique<mgcl::VaryingStencilGpu>(m, n, o, 3, gh, t.getContext(), t.getCommands(), t.getProgram());
        t.finish();

        // create VaryingStencil, fill with random values and copy to gpu buffer
        mgcl::FixedStencil a_h(3);
        a_h.fillRandomInt();
        a->fill(a_h, t.getCommands(), true);

        mgcl::VaryingStencil b_h(m, n, o, 3, gh, gh, gh);
        b_h.fillRandomInt();
        b->fill(b_h, t.getCommands(), true);
        t.finish();
        b_h.updateGhosts();
        b->updateGhosts(t.getProgram(), t.getCommands(), nullptr, nullptr);

        auto c_h = a_h.multiply(b_h, 2, nullptr, true, true);
        auto c = a->multiply(*b, 2, t.getProgram(), t.getCommands(), t.getContext(), nullptr, true, true, nullptr, nullptr);
        t.finish();

        auto ret = c.read(t.getCommands(), true);
        t.finish();

        // check results
        REQUIRE(c.getM() == c_h.getM());
        REQUIRE(c.getN() == c_h.getN());
        REQUIRE(c.getO() == c_h.getO());
        REQUIRE(c.getWidth() == c_h.getWidth());
        REQUIRE(c.getWidth() == c_h.getWidth());
        REQUIRE(c.getWidth() == c_h.getWidth());
        REQUIRE(c.getGh() == c_h.getGhostsM());
        REQUIRE(c.getGh() == c_h.getGhostsN());
        REQUIRE(c.getGh() == c_h.getGhostsO());

        // clang-format off
        for (int i = 0; i < c_h.getMgh(); i++)
        for (int j = 0; j < c_h.getNgh(); j++)
        for (int k = 0; k < c_h.getOgh(); k++)
            for (int ii = 0; ii < c_h.getWidth(); ii++)
            for (int jj = 0; jj < c_h.getWidth(); jj++)
            for (int kk = 0; kk < c_h.getWidth(); kk++)
            {
                REQUIRE(c_h[ii][jj][kk][i][j][k] == ret[ii][jj][kk][i][j][k]);
            }
        // clang-format on
    }

    SECTION("widths 3 * 5")
    {
        auto a = std::make_unique<mgcl::FixedStencilGpu>(3, t.getContext(), t.getCommands(), t.getProgram());
        auto b = std::make_unique<mgcl::VaryingStencilGpu>(m, n, o, 5, gh, t.getContext(), t.getCommands(), t.getProgram());
        t.finish();

        // create VaryingStencil, fill with random values and copy to gpu buffer
        mgcl::FixedStencil a_h(3);
        a_h.fillRandomInt();
        a->fill(a_h, t.getCommands(), true);

        mgcl::VaryingStencil b_h(m, n, o, 5, gh, gh, gh);
        b_h.fillRandomInt();
        b->fill(b_h, t.getCommands(), true);
        t.finish();
        b_h.updateGhosts();
        b->updateGhosts(t.getProgram(), t.getCommands(), nullptr, nullptr);

        auto c_h = a_h.multiply(b_h, 2, nullptr, true, true);
        auto c = a->multiply(*b, 2, t.getProgram(), t.getCommands(), t.getContext(), nullptr, true, true, nullptr, nullptr);
        t.finish();

        auto ret = c.read(t.getCommands(), true);
        t.finish();

        // check results
        REQUIRE(c.getM() == c_h.getM());
        REQUIRE(c.getN() == c_h.getN());
        REQUIRE(c.getO() == c_h.getO());
        REQUIRE(c.getWidth() == c_h.getWidth());
        REQUIRE(c.getWidth() == c_h.getWidth());
        REQUIRE(c.getWidth() == c_h.getWidth());
        REQUIRE(c.getGh() == c_h.getGhostsM());
        REQUIRE(c.getGh() == c_h.getGhostsN());
        REQUIRE(c.getGh() == c_h.getGhostsO());

        // clang-format off
        for (int i = 0; i < c_h.getMgh(); i++)
        for (int j = 0; j < c_h.getNgh(); j++)
        for (int k = 0; k < c_h.getOgh(); k++)
            for (int ii = 0; ii < c_h.getWidth(); ii++)
            for (int jj = 0; jj < c_h.getWidth(); jj++)
            for (int kk = 0; kk < c_h.getWidth(); kk++)
            {
                REQUIRE(c_h[ii][jj][kk][i][j][k] == ret[ii][jj][kk][i][j][k]);
            }
        // clang-format on
    }

    SECTION("widths 5 * 3")
    {
        auto a = std::make_unique<mgcl::FixedStencilGpu>(5, t.getContext(), t.getCommands(), t.getProgram());
        auto b = std::make_unique<mgcl::VaryingStencilGpu>(m, n, o, 3, gh, t.getContext(), t.getCommands(), t.getProgram());
        t.finish();

        // create VaryingStencil, fill with random values and copy to gpu buffer
        mgcl::FixedStencil a_h(5);
        a_h.fillRandomInt();
        a->fill(a_h, t.getCommands(), true);

        mgcl::VaryingStencil b_h(m, n, o, 3, gh, gh, gh);
        b_h.fillRandomInt();
        b->fill(b_h, t.getCommands(), true);
        t.finish();
        b_h.updateGhosts();
        b->updateGhosts(t.getProgram(), t.getCommands(), nullptr, nullptr);

        auto c_h = a_h.multiply(b_h, 2, nullptr, true, true);
        auto c = a->multiply(*b, 2, t.getProgram(), t.getCommands(), t.getContext(), nullptr, true, true, nullptr, nullptr);
        t.finish();

        auto ret = c.read(t.getCommands(), true);
        t.finish();

        // check results
        REQUIRE(c.getM() == c_h.getM());
        REQUIRE(c.getN() == c_h.getN());
        REQUIRE(c.getO() == c_h.getO());
        REQUIRE(c.getWidth() == c_h.getWidth());
        REQUIRE(c.getWidth() == c_h.getWidth());
        REQUIRE(c.getWidth() == c_h.getWidth());
        REQUIRE(c.getGh() == c_h.getGhostsM());
        REQUIRE(c.getGh() == c_h.getGhostsN());
        REQUIRE(c.getGh() == c_h.getGhostsO());

        // clang-format off
        for (int i = 0; i < c_h.getMgh(); i++)
        for (int j = 0; j < c_h.getNgh(); j++)
        for (int k = 0; k < c_h.getOgh(); k++)
            for (int ii = 0; ii < c_h.getWidth(); ii++)
            for (int jj = 0; jj < c_h.getWidth(); jj++)
            for (int kk = 0; kk < c_h.getWidth(); kk++)
            {
                REQUIRE(c_h[ii][jj][kk][i][j][k] == ret[ii][jj][kk][i][j][k]);
            }
        // clang-format on
    }

    SECTION("widths 5 * 5")
    {
        auto a = std::make_unique<mgcl::FixedStencilGpu>(5, t.getContext(), t.getCommands(), t.getProgram());
        auto b = std::make_unique<mgcl::VaryingStencilGpu>(m, n, o, 5, gh, t.getContext(), t.getCommands(), t.getProgram());
        t.finish();

        // create VaryingStencil, fill with random values and copy to gpu buffer
        mgcl::FixedStencil a_h(5);
        a_h.fillRandomInt();
        a->fill(a_h, t.getCommands(), true);

        mgcl::VaryingStencil b_h(m, n, o, 5, gh, gh, gh);
        b_h.fillRandomInt();
        b->fill(b_h, t.getCommands(), true);
        t.finish();
        b_h.updateGhosts();
        b->updateGhosts(t.getProgram(), t.getCommands(), nullptr, nullptr);

        auto c_h = a_h.multiply(b_h, 2, nullptr, true, true);
        auto c = a->multiply(*b, 2, t.getProgram(), t.getCommands(), t.getContext(), nullptr, true, true, nullptr, nullptr);
        t.finish();

        auto ret = c.read(t.getCommands(), true);
        t.finish();

        // check results
        REQUIRE(c.getM() == c_h.getM());
        REQUIRE(c.getN() == c_h.getN());
        REQUIRE(c.getO() == c_h.getO());
        REQUIRE(c.getWidth() == c_h.getWidth());
        REQUIRE(c.getWidth() == c_h.getWidth());
        REQUIRE(c.getWidth() == c_h.getWidth());
        REQUIRE(c.getGh() == c_h.getGhostsM());
        REQUIRE(c.getGh() == c_h.getGhostsN());
        REQUIRE(c.getGh() == c_h.getGhostsO());

        // clang-format off
        for (int i = 0; i < c_h.getMgh(); i++)
        for (int j = 0; j < c_h.getNgh(); j++)
        for (int k = 0; k < c_h.getOgh(); k++)
            for (int ii = 0; ii < c_h.getWidth(); ii++)
            for (int jj = 0; jj < c_h.getWidth(); jj++)
            for (int kk = 0; kk < c_h.getWidth(); kk++)
            {
                REQUIRE(c_h[ii][jj][kk][i][j][k] == ret[ii][jj][kk][i][j][k]);
            }
        // clang-format on
    }
}

TEST_CASE("FixedStencilGpu::fill")
{
    auto deviceType = CL_DEVICE_TYPE_GPU; // GENERATE(CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU);

    if (!mgcl_test::TestUtility::deviceAvailable("", deviceType))
    {
        std::string typeName = deviceType == CL_DEVICE_TYPE_GPU ? "CL_DEVICE_TYPE_GPU" : "CL_DEVICE_TYPE_CPU";
        std::cout << "Skipping non-available device type '" << typeName << "'" << std::endl;
        return;
    }

    mgcl_test::TestUtility t(deviceType);

    SECTION("FixedStencil3x3x3")
    {
        int width = 3;
        auto s = std::make_unique<mgcl::FixedStencilGpu>(width, t.getContext(), t.getCommands(), t.getProgram());
        t.finish();

        int sizeNeeded = width * width * width;

        // create VaryingStencil, fill with random values and copy to gpu buffer
        mgcl::FixedStencil s3(3);
        s3.fillRandom();
        s->fill(s3, t.getCommands(), true);

        // check results
        double tmp[sizeNeeded];
        int err = clEnqueueReadBuffer(t.getCommands(), s->getBuf(), CL_TRUE, 0,
                                      sizeof(double) * sizeNeeded, tmp, 0, NULL, NULL);
        mgcl::mgclCheckError(err, "clEnqueueReadBuffer");

        for (int i = 0; i < sizeNeeded; i++)
            REQUIRE(tmp[i] == s3[0][0][i]);
    }

    SECTION("FixedStencil5x5x5")
    {
        int width = 5;
        auto s = std::make_unique<mgcl::FixedStencilGpu>(width, t.getContext(), t.getCommands(), t.getProgram());
        t.finish();

        int sizeNeeded = width * width * width;

        // create VaryingStencil, fill with random values and copy to gpu buffer
        mgcl::FixedStencil s3(5);
        s3.fillRandom();
        s->fill(s3, t.getCommands(), true);

        // check results
        double tmp[sizeNeeded];
        int err = clEnqueueReadBuffer(t.getCommands(), s->getBuf(), CL_TRUE, 0,
                                      sizeof(double) * sizeNeeded, tmp, 0, NULL, NULL);
        mgcl::mgclCheckError(err, "clEnqueueReadBuffer");

        for (int i = 0; i < sizeNeeded; i++)
            REQUIRE(tmp[i] == s3[0][0][i]);
    }

    SECTION("throwing")
    {
        int width = 3;
        auto s = std::make_unique<mgcl::FixedStencilGpu>(width, t.getContext(), t.getCommands(), t.getProgram());
        t.finish();

        // widths do not match
        mgcl::FixedStencil s3(5);
        REQUIRE_THROWS(s->fill(s3, t.getCommands(), true));
    }
}

TEST_CASE("FixedStencilGpu::read")
{
    auto deviceType = CL_DEVICE_TYPE_GPU; // GENERATE(CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU);

    if (!mgcl_test::TestUtility::deviceAvailable("", deviceType))
    {
        std::string typeName = deviceType == CL_DEVICE_TYPE_GPU ? "CL_DEVICE_TYPE_GPU" : "CL_DEVICE_TYPE_CPU";
        std::cout << "Skipping non-available device type '" << typeName << "'" << std::endl;
        return;
    }

    mgcl_test::TestUtility t(deviceType);

    SECTION("FixedStencil3x3x3")
    {
        int width = 3;
        auto s = std::make_unique<mgcl::FixedStencilGpu>(width, t.getContext(), t.getCommands(), t.getProgram());
        t.finish();

        // create FixedStencil, fill with random values and copy to gpu buffer
        mgcl::FixedStencil s3(3);
        s3.fillRandom();
        s->fill(s3, t.getCommands(), true);
        t.finish();

        // read buffer
        auto ret = s->read(t.getCommands(), true);
        t.finish();

        // check results
        REQUIRE(ret.field1d().size() == s3.field1d().size());

        for (int i = 0; i < ret.field1d().size(); i++)
            REQUIRE(ret.field1d()[i] == s3.field1d()[i]);

        t.finish();
    }

    SECTION("FixedStencil5x5x5")
    {
        int width = 5;
        mgcl::FixedStencilGpu s(width, t.getContext(), t.getCommands(), t.getProgram());
        t.finish();

        // create FixedStencil, fill with random values and copy to gpu buffer
        mgcl::FixedStencil s3(5);
        s3.fillRandom();
        s.fill(s3, t.getCommands(), true);
        t.finish();

        // read buffer
        auto ret = s.read(t.getCommands(), true);
        t.finish();

        // check results
        REQUIRE(ret.field1d().size() == s3.field1d().size());

        for (int i = 0; i < ret.field1d().size(); i++)
            REQUIRE(ret.field1d()[i] == s3.field1d()[i]);

        t.finish();
    }
}

TEST_CASE("FixedStencilGpu::create3dFullWeightRestrictionGpu")
{
    auto deviceType = CL_DEVICE_TYPE_GPU; // GENERATE(CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU);

    if (!mgcl_test::TestUtility::deviceAvailable("", deviceType))
    {
        std::string typeName = deviceType == CL_DEVICE_TYPE_GPU ? "CL_DEVICE_TYPE_GPU" : "CL_DEVICE_TYPE_CPU";
        std::cout << "Skipping non-available device type '" << typeName << "'" << std::endl;
        return;
    }

    mgcl_test::TestUtility t(deviceType);

    auto rgpu = mgcl::create3dFullWeightRestrictionStencilGpu(t.getContext(), t.getCommands(), t.getProgram());
    auto r = rgpu.read(t.getCommands(), true);
    t.finish();

    double factor = 1.0 / 64.0;

    // full-weight restriction
    REQUIRE(r[0][0][0] == 1 * factor);
    REQUIRE(r[0][0][1] == 2 * factor);
    REQUIRE(r[0][0][2] == 1 * factor);
    REQUIRE(r[0][1][0] == 2 * factor);
    REQUIRE(r[0][1][1] == 4 * factor);
    REQUIRE(r[0][1][2] == 2 * factor);
    REQUIRE(r[0][2][0] == 1 * factor);
    REQUIRE(r[0][2][1] == 2 * factor);
    REQUIRE(r[0][2][2] == 1 * factor);
    REQUIRE(r[1][0][0] == 2 * factor);
    REQUIRE(r[1][0][1] == 4 * factor);
    REQUIRE(r[1][0][2] == 2 * factor);
    REQUIRE(r[1][1][0] == 4 * factor);
    REQUIRE(r[1][1][1] == 8 * factor);
    REQUIRE(r[1][1][2] == 4 * factor);
    REQUIRE(r[1][2][0] == 2 * factor);
    REQUIRE(r[1][2][1] == 4 * factor);
    REQUIRE(r[1][2][2] == 2 * factor);
    REQUIRE(r[2][0][0] == 1 * factor);
    REQUIRE(r[2][0][1] == 2 * factor);
    REQUIRE(r[2][0][2] == 1 * factor);
    REQUIRE(r[2][1][0] == 2 * factor);
    REQUIRE(r[2][1][1] == 4 * factor);
    REQUIRE(r[2][1][2] == 2 * factor);
    REQUIRE(r[2][2][0] == 1 * factor);
    REQUIRE(r[2][2][1] == 2 * factor);
    REQUIRE(r[2][2][2] == 1 * factor);
}

TEST_CASE("FixedStencilGpu::create3dBilinearProlongationStencilGpu")
{
    auto deviceType = CL_DEVICE_TYPE_GPU; // GENERATE(CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU);

    if (!mgcl_test::TestUtility::deviceAvailable("", deviceType))
    {
        std::string typeName = deviceType == CL_DEVICE_TYPE_GPU ? "CL_DEVICE_TYPE_GPU" : "CL_DEVICE_TYPE_CPU";
        std::cout << "Skipping non-available device type '" << typeName << "'" << std::endl;
        return;
    }

    mgcl_test::TestUtility t(deviceType);

    auto rgpu = mgcl::create3dBilinearProlongationStencilGpu(t.getContext(), t.getCommands(), t.getProgram());
    auto r = rgpu.read(t.getCommands(), true);
    t.finish();

    double factor = 1.0 / 8.0;

    // bilinear prolongation
    REQUIRE(r[0][0][0] == 1 * factor);
    REQUIRE(r[0][0][1] == 2 * factor);
    REQUIRE(r[0][0][2] == 1 * factor);
    REQUIRE(r[0][1][0] == 2 * factor);
    REQUIRE(r[0][1][1] == 4 * factor);
    REQUIRE(r[0][1][2] == 2 * factor);
    REQUIRE(r[0][2][0] == 1 * factor);
    REQUIRE(r[0][2][1] == 2 * factor);
    REQUIRE(r[0][2][2] == 1 * factor);
    REQUIRE(r[1][0][0] == 2 * factor);
    REQUIRE(r[1][0][1] == 4 * factor);
    REQUIRE(r[1][0][2] == 2 * factor);
    REQUIRE(r[1][1][0] == 4 * factor);
    REQUIRE(r[1][1][1] == 8 * factor);
    REQUIRE(r[1][1][2] == 4 * factor);
    REQUIRE(r[1][2][0] == 2 * factor);
    REQUIRE(r[1][2][1] == 4 * factor);
    REQUIRE(r[1][2][2] == 2 * factor);
    REQUIRE(r[2][0][0] == 1 * factor);
    REQUIRE(r[2][0][1] == 2 * factor);
    REQUIRE(r[2][0][2] == 1 * factor);
    REQUIRE(r[2][1][0] == 2 * factor);
    REQUIRE(r[2][1][1] == 4 * factor);
    REQUIRE(r[2][1][2] == 2 * factor);
    REQUIRE(r[2][2][0] == 1 * factor);
    REQUIRE(r[2][2][1] == 2 * factor);
    REQUIRE(r[2][2][2] == 1 * factor);
}

TEST_CASE("VaryingStencilGpu::extract_border_planes")
{
    SECTION("indices")
    {
        int m = 3;
        int n = 5;
        int o = 7;
        int ghosts_m = 1;
        int ghosts_n = 1; // 2;
        int ghosts_o = 1; // 3;
        int mgh = m + 2 * ghosts_m;
        int ngh = n + 2 * ghosts_n;
        int ogh = o + 2 * ghosts_o;
        int yz = ngh * ogh;
        int xz = mgh * ogh;
        int xy = mgh * ngh;
        int gridsize = mgh * ngh * ogh;
        int ressize = (2 * yz * ghosts_m + 2 * xz * ghosts_n + 2 * xy * ghosts_o) * 27;

        mgcl::VaryingStencil h_stencil(m, n, o, 3, ghosts_m, ghosts_n, ghosts_o);
        h_stencil.fill1dIndex(false);
        const double* buf_stencil = h_stencil.field1d().data();

        // 1d result buffer
        double buf_res[ressize];
        for (size_t i = 0; i < ressize; i++)
        {
            buf_res[i] = -1;
        }

        for (int cnt = 0; cnt < ressize; cnt++)
        {
            int idx = cnt; // global output buffer index
            // plane sizes
            // int yz = 27 * ngh * ogh;
            // int xz = 27 * mgh * ogh;
            // int xy = 27 * mgh * ngh;

            // Front planes (for each coefficient)
            if (idx < ghosts_m * yz * 27)
            {
                REQUIRE(idx == cnt);

                int idx_coeff = idx / (ghosts_m * yz);            // 1d index of the current coefficient
                int idx_grid = idx - idx_coeff * (ghosts_m * yz); // local index of the grid point inside the grid of one coefficient

                int i = idx_grid / yz + ghosts_m;
                int j = (idx_grid - (i - ghosts_m) * yz) / ogh;
                int k = idx_grid % ogh;
                buf_res[idx] = buf_stencil[idx_coeff * gridsize + i * ngh * ogh + j * ogh + k];
            }
            // Back planes
            else if (idx < 2 * ghosts_m * yz * 27)
            {
                int idx_resbuf = idx;
                idx -= ghosts_m * yz * 27;                        // reset to 0 for index calculation
                int idx_coeff = idx / (ghosts_m * yz);            // 1d index of the current coefficient
                int idx_grid = idx - idx_coeff * (ghosts_m * yz); // local index of the grid point inside the grid of one coefficient

                if (cnt == ghosts_m * yz * 27)
                    REQUIRE(idx == 0);

                int i = idx_grid / yz + m;
                int j = (idx_grid - (i - m) * yz) / ogh;
                int k = idx_grid % ogh;

                REQUIRE(idx_resbuf == cnt);

                buf_res[idx_resbuf] = buf_stencil[idx_coeff * gridsize + i * ngh * ogh + j * ogh + k];
            }
            // Top planes
            else if (idx < (2 * ghosts_m * yz + ghosts_n * xz) * 27)
            {
                int idx_resbuf = idx;
                idx -= 2 * ghosts_m * yz * 27;                    // reset to 0 for index calculation
                int idx_coeff = idx / (ghosts_n * xz);            // 1d index of the current coefficient
                int idx_grid = idx - idx_coeff * (ghosts_n * xz); // local index of the grid point inside the grid of one coefficient

                if (cnt == 2 * ghosts_m * yz * 27)
                    REQUIRE(idx == 0);

                int j = idx_grid / xz + ghosts_n;
                int i = (idx_grid - (j - ghosts_n) * xz) / ogh;
                int k = idx_grid % ogh;

                REQUIRE(idx_resbuf == cnt);

                buf_res[idx_resbuf] = buf_stencil[idx_coeff * gridsize + i * ngh * ogh + j * ogh + k];
            }
            // Bottom planes
            else if (idx < (2 * ghosts_m * yz + 2 * ghosts_n * xz) * 27)
            {
                int idx_resbuf = idx;
                idx -= (2 * ghosts_m * yz + ghosts_n * xz) * 27;  // reset to 0 for index calculation
                int idx_coeff = idx / (ghosts_n * xz);            // 1d index of the current coefficient
                int idx_grid = idx - idx_coeff * (ghosts_n * xz); // local index of the grid point inside the grid of one coefficient

                if (cnt == (2 * ghosts_m * yz + ghosts_n * xz) * 27)
                    REQUIRE(idx == 0);

                int j = idx_grid / xz + n;
                int i = (idx_grid - (j - n) * xz) / ogh;
                int k = idx_grid % ogh;

                REQUIRE(idx_resbuf == cnt);

                buf_res[idx_resbuf] = buf_stencil[idx_coeff * gridsize + i * ngh * ogh + j * ogh + k];
            }
            // Left planes
            else if (idx < (2 * ghosts_m * yz + 2 * ghosts_n * xz + ghosts_o * xy) * 27)
            {
                int idx_resbuf = idx;
                idx -= (2 * ghosts_m * yz + 2 * ghosts_n * xz) * 27; // reset to 0 for index calculation
                int idx_coeff = idx / (ghosts_o * xy);               // 1d index of the current coefficient
                int idx_grid = idx - idx_coeff * (ghosts_o * xy);    // local index of the grid point inside the grid of one coefficient

                if (cnt == (2 * ghosts_m * yz + 2 * ghosts_n * xz) * 27)
                    REQUIRE(idx == 0);

                int k = idx_grid / xy + ghosts_o;
                int i = (idx_grid - (k - ghosts_o) * xy) / ngh;
                int j = idx_grid % ngh;

                REQUIRE(idx_resbuf == cnt);

                buf_res[idx_resbuf] = buf_stencil[idx_coeff * gridsize + i * ngh * ogh + j * ogh + k];
            }
            // Right planes
            else if (idx < (2 * ghosts_m * yz + 2 * ghosts_n * xz + 2 * ghosts_o * xy) * 27)
            {
                int idx_resbuf = idx;
                idx -= (2 * ghosts_m * yz + 2 * ghosts_n * xz + ghosts_o * xy) * 27; // reset to 0 for index calculation
                int idx_coeff = idx / (ghosts_o * xy);                               // 1d index of the current coefficient
                int idx_grid = idx - idx_coeff * (ghosts_o * xy);                    // local index of the grid point inside the grid of one coefficient

                if (cnt == (2 * ghosts_m * yz + 2 * ghosts_n * xz + ghosts_o * xy) * 27)
                    REQUIRE(idx == 0);

                int k = idx_grid / xy + o;
                int i = (idx_grid - (k - o) * xy) / ngh;
                int j = idx_grid % ngh;

                REQUIRE(idx_resbuf == cnt);

                buf_res[idx_resbuf] = buf_stencil[idx_coeff * gridsize + i * ngh * ogh + j * ogh + k];
            }
        }

        // Check that every index was written to
        for (size_t i = 0; i < ressize; i++)
        {
            CAPTURE(i, ressize);
            REQUIRE(buf_res[i] >= 0);
        }

        int cnt = 0;
        // front planes (yz)
        for (int ii = 0; ii < 3; ii++)
            for (int jj = 0; jj < 3; jj++)
                for (int kk = 0; kk < 3; kk++)
                    for (int i = ghosts_m; i < 2 * ghosts_m; i++) // ghm real planes in the front
                        for (int j = 0; j < ngh; j++)             // all real cells in y-dir
                            for (int k = 0; k < ogh; k++)         // all real cells in z-dir
                            {
                                CAPTURE(cnt, ii, jj, kk, i, j, k);
                                REQUIRE(buf_res[cnt++] == h_stencil[ii][jj][kk][i][j][k]);
                            }

        // back planes (yz)
        for (int ii = 0; ii < 3; ii++)
            for (int jj = 0; jj < 3; jj++)
                for (int kk = 0; kk < 3; kk++)
                    for (int i = m; i < m + ghosts_m; i++) // ghosts_m real planes in the back
                        for (int j = 0; j < ngh; j++)      // all real cells in y-dir
                            for (int k = 0; k < ogh; k++)  // all real cells in z-dir
                            {
                                CAPTURE(cnt, ii, jj, kk, i, j, k);
                                REQUIRE(buf_res[cnt++] == h_stencil[ii][jj][kk][i][j][k]);
                            }

        // top planes (xz)
        for (int ii = 0; ii < 3; ii++)
            for (int jj = 0; jj < 3; jj++)
                for (int kk = 0; kk < 3; kk++)
                    for (int j = ghosts_n; j < 2 * ghosts_n; j++)
                        for (int i = 0; i < mgh; i++)
                            for (int k = 0; k < ogh; k++)
                            {
                                CAPTURE(cnt, ii, jj, kk, i, j, k);
                                REQUIRE(buf_res[cnt++] == h_stencil[ii][jj][kk][i][j][k]);
                            }

        // bottom planes (xz)
        for (int ii = 0; ii < 3; ii++)
            for (int jj = 0; jj < 3; jj++)
                for (int kk = 0; kk < 3; kk++)
                    for (int j = n; j < n + ghosts_n; j++)
                        for (int i = 0; i < mgh; i++)
                            for (int k = 0; k < ogh; k++)
                            {
                                CAPTURE(cnt, ii, jj, kk, i, j, k);
                                REQUIRE(buf_res[cnt++] == h_stencil[ii][jj][kk][i][j][k]);
                            }

        // left planes (xy)
        for (int ii = 0; ii < 3; ii++)
            for (int jj = 0; jj < 3; jj++)
                for (int kk = 0; kk < 3; kk++)
                    for (int k = ghosts_o; k < 2 * ghosts_o; k++)
                        for (int i = 0; i < mgh; i++)
                            for (int j = 0; j < ngh; j++)
                            {
                                CAPTURE(cnt, ii, jj, kk, i, j, k);
                                REQUIRE(buf_res[cnt++] == h_stencil[ii][jj][kk][i][j][k]);
                            }

        // right planes (xy)
        for (int ii = 0; ii < 3; ii++)
            for (int jj = 0; jj < 3; jj++)
                for (int kk = 0; kk < 3; kk++)
                    for (int k = o; k < o + ghosts_o; k++)
                        for (int i = 0; i < mgh; i++)
                            for (int j = 0; j < ngh; j++)
                            {
                                CAPTURE(cnt, ii, jj, kk, i, j, k);
                                REQUIRE(buf_res[cnt++] == h_stencil[ii][jj][kk][i][j][k]);
                            }
    }

    SECTION("success")
    {
        // Create dummy problem
        auto v = std::make_shared<mgcl::Cuboid>(1, 1, 1);
        auto f = std::make_shared<mgcl::Cuboid>(1, 1, 1);
        mgcl::Problem p(1, 1, 1, f, v);
        p.setUseOpencl(true);
        p.setDeviceType(CL_DEVICE_TYPE_GPU);
        p.setProfilingEnabled(true);
        p.init();

        int m = 3;
        int n = 5;
        int o = 7;
        int ghosts_m = 1;
        int ghosts_n = 1; // VaryingStencilGpu currently does not support different ghosts per dimension
        int ghosts_o = 1;
        int mgh = m + 2 * ghosts_m;
        int ngh = n + 2 * ghosts_n;
        int ogh = o + 2 * ghosts_o;
        int yz = ngh * ogh;
        int xz = mgh * ogh;
        int xy = mgh * ngh;
        int ressize = (2 * yz * ghosts_m + 2 * xz * ghosts_n + 2 * xy * ghosts_o) * 27;

        mgcl::VaryingStencil h_stencil(m, n, o, 3, ghosts_m, ghosts_n, ghosts_o);
        h_stencil.fill1dIndex(false);
        const double* buf_stencil = h_stencil.field1d().data();

        mgcl::VaryingStencilGpu d_stencil(m, n, o, 3, ghosts_m, p.getContext(), p.getCommands(), p.getProgram());
        d_stencil.fill(h_stencil, p.getCommands(), true);

        std::vector<double> h_ret(ressize, -1);
        mgcl::BufferGpu d_tmp(p.getContext(), CL_MEM_READ_WRITE, ressize);

        d_stencil.extractBorderPlanes(p.getCommands(), p.getProgram(), d_tmp, h_ret, &p.getKernelConfig(), p.getProfilingData());
        p.finish();

        // Check that every index was written to
        for (size_t i = 0; i < ressize; i++)
        {
            CAPTURE(i, ressize);
            REQUIRE(h_ret[i] >= 0);
        }

        int cnt = 0;
        // front planes (yz)
        for (int ii = 0; ii < 3; ii++)
            for (int jj = 0; jj < 3; jj++)
                for (int kk = 0; kk < 3; kk++)
                    for (int i = ghosts_m; i < 2 * ghosts_m; i++) // ghm real planes in the front
                        for (int j = 0; j < ngh; j++)             // all cells in y-dir
                            for (int k = 0; k < ogh; k++)         // all cells in z-dir
                            {
                                CAPTURE(i, j, k, ii, jj, kk, cnt);
                                REQUIRE(h_ret[cnt++] == h_stencil[ii][jj][kk][i][j][k]);
                            }

        // back planes (yz)
        for (int ii = 0; ii < 3; ii++)
            for (int jj = 0; jj < 3; jj++)
                for (int kk = 0; kk < 3; kk++)
                    for (int i = m; i < m + ghosts_m; i++) // ghosts_m real planes in the back
                        for (int j = 0; j < ngh; j++)      // all cells in y-dir
                            for (int k = 0; k < ogh; k++)  // all cells in z-dir
                                REQUIRE(h_ret[cnt++] == h_stencil[ii][jj][kk][i][j][k]);

        // top planes (xz)
        for (int ii = 0; ii < 3; ii++)
            for (int jj = 0; jj < 3; jj++)
                for (int kk = 0; kk < 3; kk++)
                    for (int j = ghosts_n; j < 2 * ghosts_n; j++)
                        for (int i = 0; i < mgh; i++)
                            for (int k = 0; k < ogh; k++)
                                REQUIRE(h_ret[cnt++] == h_stencil[ii][jj][kk][i][j][k]);

        // bottom planes (xz)
        for (int ii = 0; ii < 3; ii++)
            for (int jj = 0; jj < 3; jj++)
                for (int kk = 0; kk < 3; kk++)
                    for (int j = n; j < n + ghosts_n; j++)
                        for (int i = 0; i < mgh; i++)
                            for (int k = 0; k < ogh; k++)
                                REQUIRE(h_ret[cnt++] == h_stencil[ii][jj][kk][i][j][k]);

        // left planes (xy)
        for (int ii = 0; ii < 3; ii++)
            for (int jj = 0; jj < 3; jj++)
                for (int kk = 0; kk < 3; kk++)
                    for (int k = ghosts_o; k < 2 * ghosts_o; k++)
                        for (int i = 0; i < mgh; i++)
                            for (int j = 0; j < ngh; j++)
                                REQUIRE(h_ret[cnt++] == h_stencil[ii][jj][kk][i][j][k]);

        // right planes (xy)
        for (int ii = 0; ii < 3; ii++)
            for (int jj = 0; jj < 3; jj++)
                for (int kk = 0; kk < 3; kk++)
                    for (int k = o; k < o + ghosts_o; k++)
                        for (int i = 0; i < mgh; i++)
                            for (int j = 0; j < ngh; j++)
                                REQUIRE(h_ret[cnt++] == h_stencil[ii][jj][kk][i][j][k]);
    }
}

TEST_CASE("VaryingStencilGpu::pasteGhostsFromBorderPlanes")
{
    SECTION("indices")
    {
        int m = 3;
        int n = 5;
        int o = 7;
        int ghosts_m = 1;
        int ghosts_n = 1; // 2;
        int ghosts_o = 1; // 3;
        int mgh = m + 2 * ghosts_m;
        int ngh = n + 2 * ghosts_n;
        int ogh = o + 2 * ghosts_o;
        int yz = ngh * ogh;
        int xz = mgh * ogh;
        int xy = mgh * ngh;
        int gridsize = mgh * ngh * ogh;
        int ressize = (2 * yz * ghosts_m + 2 * xz * ghosts_n + 2 * xy * ghosts_o) * 27;

        mgcl::VaryingStencil h_stencil(m, n, o, 3, ghosts_m, ghosts_n, ghosts_o);
        h_stencil.fill(-1, false);
        double* buf_stencil = h_stencil.field1d().data();

        // 1d ghosts buffer, filled with 1d index
        double buf_ghosts[ressize];
        for (size_t i = 0; i < ressize; i++)
        {
            buf_ghosts[i] = i;
        }

        // Simulate kernel call
        for (int cnt = 0; cnt < ressize; cnt++)
        {
            int idx = cnt;
            // plane sizes
            // int yz = ngh * ogh;
            // int xz = mgh * ogh;
            // int xy = mgh * ngh;

            // Front planes (back ghosts)
            if (idx < ghosts_m * yz * 27)
            {
                int idx_coeff = idx / (ghosts_m * yz);            // 1d index of the current coefficient
                int idx_grid = idx - idx_coeff * (ghosts_m * yz); // local index of the grid point inside the grid of one coefficient

                int i = idx_grid / yz + m + ghosts_m;
                int j = (idx_grid - (i - (m + ghosts_m)) * yz) / ogh;
                int k = idx_grid % ogh;

                REQUIRE(i >= m + ghosts_m);

                // No corners or edges, only ghosts directly adjacent to real back face
                if (j >= ghosts_n && j < n + ghosts_n && k >= ghosts_o && k < o + ghosts_o)
                    buf_stencil[idx_coeff * gridsize + i * ngh * ogh + j * ogh + k] = buf_ghosts[idx];
            }
            // Back planes (front ghosts)
            else if (idx < 2 * ghosts_m * yz * 27)
            {
                idx -= ghosts_m * yz * 27;                        // reset to 0 for index calculation
                int idx_coeff = idx / (ghosts_m * yz);            // 1d index of the current coefficient
                int idx_grid = idx - idx_coeff * (ghosts_m * yz); // local index of the grid point inside the grid of one coefficient

                int i = idx_grid / yz;
                int j = (idx_grid - i * yz) / ogh;
                int k = idx_grid % ogh;

                CAPTURE(idx, idx_coeff, idx_grid, i, j, k, ghosts_m);
                REQUIRE(i < ghosts_m); // TODO

                // No corners or edges, only ghosts directly adjacent to real front face
                if (j >= ghosts_n && j < n + ghosts_n && k >= ghosts_o && k < o + ghosts_o)
                    buf_stencil[idx_coeff * gridsize + i * ngh * ogh + j * ogh + k] = buf_ghosts[idx + ghosts_m * yz * 27];
            }
            // Top planes (bottom ghosts)
            else if (idx < (2 * ghosts_m * yz + ghosts_n * xz) * 27)
            {
                idx -= 2 * ghosts_m * yz * 27;                    // reset to 0 for index calculation
                int idx_coeff = idx / (ghosts_n * xz);            // 1d index of the current coefficient
                int idx_grid = idx - idx_coeff * (ghosts_n * xz); // local index of the grid point inside the grid of one coefficient

                int j = idx_grid / xz + n + ghosts_n;
                int i = (idx_grid - (j - (n + ghosts_n)) * xz) / ogh;
                int k = idx_grid % ogh;

                // Ignore left and right ghost cells, but include front and back ghosts
                if (k >= ghosts_o && k < o + ghosts_o)
                    buf_stencil[idx_coeff * gridsize + i * ngh * ogh + j * ogh + k] = buf_ghosts[idx + 2 * ghosts_m * yz * 27];
            }
            // Bottom planes (top ghosts)
            else if (idx < (2 * ghosts_m * yz + 2 * ghosts_n * xz) * 27)
            {
                idx -= (2 * ghosts_m * yz + ghosts_n * xz) * 27;  // reset to 0 for index calculation
                int idx_coeff = idx / (ghosts_n * xz);            // 1d index of the current coefficient
                int idx_grid = idx - idx_coeff * (ghosts_n * xz); // local index of the grid point inside the grid of one coefficient

                int j = idx_grid / xz;
                int i = (idx_grid - j * xz) / ogh;
                int k = idx_grid % ogh;

                // Ignore left and right ghost cells, but include front and back ghosts
                if (k >= ghosts_o && k < o + ghosts_o)
                    buf_stencil[idx_coeff * gridsize + i * ngh * ogh + j * ogh + k] = buf_ghosts[idx + (2 * ghosts_m * yz + ghosts_n * xz) * 27];
            }
            // Left planes (right ghosts)
            else if (idx < (2 * ghosts_m * yz + 2 * ghosts_n * xz + ghosts_o * xy) * 27)
            {
                idx -= (2 * ghosts_m * yz + 2 * ghosts_n * xz) * 27; // reset to 0 for index calculation
                int idx_coeff = idx / (ghosts_o * xy);               // 1d index of the current coefficient
                int idx_grid = idx - idx_coeff * (ghosts_o * xy);    // local index of the grid point inside the grid of one coefficient

                int k = idx_grid / xy + o + ghosts_o;
                int i = (idx_grid - (k - (o + ghosts_o)) * xy) / ngh;
                int j = idx_grid % ngh;
                buf_stencil[idx_coeff * gridsize + i * ngh * ogh + j * ogh + k] = buf_ghosts[idx + (2 * ghosts_m * yz + 2 * ghosts_n * xz) * 27];
            }
            // Right planes (left ghosts)
            else if (idx < (2 * ghosts_m * yz + 2 * ghosts_n * xz + 2 * ghosts_o * xy) * 27)
            {
                idx -= (2 * ghosts_m * yz + 2 * ghosts_n * xz + ghosts_o * xy) * 27; // reset to 0 for index calculation
                int idx_coeff = idx / (ghosts_o * xy);                               // 1d index of the current coefficient
                int idx_grid = idx - idx_coeff * (ghosts_o * xy);                    // local index of the grid point inside the grid of one coefficient

                int k = idx_grid / xy;
                int i = (idx_grid - k * xy) / ngh;
                int j = idx_grid % ngh;
                buf_stencil[idx_coeff * gridsize + i * ngh * ogh + j * ogh + k] = buf_ghosts[idx + (2 * ghosts_m * yz + 2 * ghosts_n * xz + ghosts_o * xy) * 27];
            }
        }
        // End kernel

        // Check that all real cells were left untouched
        for (int ii = 0; ii < 3; ii++)
            for (int jj = 0; jj < 3; jj++)
                for (int kk = 0; kk < 3; kk++)
                    for (int i = ghosts_m; i < m + ghosts_m; i++)
                        for (int j = ghosts_n; j < n + ghosts_n; j++)
                            for (int k = ghosts_o; k < o + ghosts_o; k++)
                            {
                                CAPTURE(i, j, k);
                                REQUIRE(h_stencil[ii][jj][kk][i][j][k] == -1);
                            }

        // Check that all ghost cells were filled with any value
        for (int ii = 0; ii < 3; ii++)
            for (int jj = 0; jj < 3; jj++)
                for (int kk = 0; kk < 3; kk++)
                    for (int i = 0; i < mgh; i++)
                        for (int j = 0; j < ngh; j++)
                            for (int k = 0; k < ogh; k++)
                            {
                                if ((i < ghosts_m || i >= m + ghosts_m) && (j < ghosts_n || j >= n + ghosts_n) && (k < ghosts_o || k >= o + ghosts_o))
                                {
                                    CAPTURE(i, j, k);
                                    REQUIRE(h_stencil[ii][jj][kk][i][j][k] >= 0);
                                }
                            }

        int cnt = 0;
        // front planes (yz)
        for (int ii = 0; ii < 3; ii++)
            for (int jj = 0; jj < 3; jj++)
                for (int kk = 0; kk < 3; kk++)
                    for (int i = m + ghosts_m; i < mgh; i++) // ghosts_m real planes in the back
                        for (int j = 0; j < ngh; j++)        // all cells in y-dir
                            for (int k = 0; k < ogh; k++)    // all cells in z-dir
                            {
                                // No corners or edges, only ghosts directly adjacent to real back face
                                if (j >= ghosts_n && j < n + ghosts_n && k >= ghosts_o && k < o + ghosts_o)
                                {
                                    CAPTURE(i, j, k, ii, jj, kk, cnt);
                                    REQUIRE(buf_ghosts[cnt] == h_stencil[ii][jj][kk][i][j][k]);
                                }
                                cnt++;
                            }

        // back planes (yz)
        for (int ii = 0; ii < 3; ii++)
            for (int jj = 0; jj < 3; jj++)
                for (int kk = 0; kk < 3; kk++)
                    for (int i = 0; i < ghosts_m; i++)    // ghm ghost planes in the front
                        for (int j = 0; j < ngh; j++)     // all cells in y-dir
                            for (int k = 0; k < ogh; k++) // all cells in z-dir
                            {                             // No corners or edges, only ghosts directly adjacent to real back face
                                if (j >= ghosts_n && j < n + ghosts_n && k >= ghosts_o && k < o + ghosts_o)
                                {
                                    CAPTURE(i, j, k, ii, jj, kk, cnt);
                                    REQUIRE(buf_ghosts[cnt] == h_stencil[ii][jj][kk][i][j][k]);
                                }
                                cnt++;
                            }

        // top planes (xz)
        for (int ii = 0; ii < 3; ii++)
            for (int jj = 0; jj < 3; jj++)
                for (int kk = 0; kk < 3; kk++)
                    for (int j = n + ghosts_n; j < ngh; j++)
                        for (int i = 0; i < mgh; i++)
                            for (int k = 0; k < ogh; k++)
                            {
                                // Ignore left and right ghost cells, but include front and back ghosts
                                if (k >= ghosts_o && k < o + ghosts_o)
                                {
                                    CAPTURE(i, j, k, ii, jj, kk, cnt);
                                    REQUIRE(buf_ghosts[cnt] == h_stencil[ii][jj][kk][i][j][k]);
                                }
                                cnt++;
                            }

        // bottom planes (xz)
        for (int ii = 0; ii < 3; ii++)
            for (int jj = 0; jj < 3; jj++)
                for (int kk = 0; kk < 3; kk++)
                    for (int j = 0; j < ghosts_n; j++)
                        for (int i = 0; i < mgh; i++)
                            for (int k = 0; k < ogh; k++)
                            {
                                // Ignore left and right ghost cells, but include front and back ghosts
                                if (k >= ghosts_o && k < o + ghosts_o)
                                {
                                    CAPTURE(i, j, k, ii, jj, kk, cnt);
                                    REQUIRE(buf_ghosts[cnt] == h_stencil[ii][jj][kk][i][j][k]);
                                }
                                cnt++;
                            }

        // left planes (xy)
        for (int ii = 0; ii < 3; ii++)
            for (int jj = 0; jj < 3; jj++)
                for (int kk = 0; kk < 3; kk++)
                    for (int k = o + ghosts_o; k < ogh; k++)
                        for (int i = 0; i < mgh; i++)
                            for (int j = 0; j < ngh; j++)
                            {
                                CAPTURE(i, j, k, ii, jj, kk, cnt);
                                REQUIRE(buf_ghosts[cnt++] == h_stencil[ii][jj][kk][i][j][k]);
                            }

        // right planes (xy)
        for (int ii = 0; ii < 3; ii++)
            for (int jj = 0; jj < 3; jj++)
                for (int kk = 0; kk < 3; kk++)
                    for (int k = 0; k < ghosts_o; k++)
                        for (int i = 0; i < mgh; i++)
                            for (int j = 0; j < ngh; j++)
                            {
                                CAPTURE(i, j, k, ii, jj, kk, cnt);
                                REQUIRE(buf_ghosts[cnt++] == h_stencil[ii][jj][kk][i][j][k]);
                            }
    }

    // SECTION("success")
    // {
    //     // Create dummy problem
    //     auto v = std::make_shared<mgcl::Cuboid>(1, 1, 1);
    //     auto f = std::make_shared<mgcl::Cuboid>(1, 1, 1);
    //     mgcl::Problem p(1, 1, 1, f, v);
    //     p.setUseOpencl(true);
    //     p.setDeviceType(CL_DEVICE_TYPE_GPU);
    //     p.init();

    //     int m = 3;
    //     int n = 5;
    //     int o = 7;
    //     int ghosts_m = 1;
    //     int ghosts_n = 2;
    //     int ghosts_o = 3;
    //     int mgh = m + 2 * ghosts_m;
    //     int ngh = n + 2 * ghosts_n;
    //     int ogh = o + 2 * ghosts_o;
    //     int yz = ngh * ogh;
    //     int xz = mgh * ogh;
    //     int xy = mgh * ngh;
    //     int ressize = 2 * yz * ghosts_m + 2 * xz * ghosts_n + 2 * xy * ghosts_o;

    //     mgcl::Cuboid h_cuboid(m, n, o, ghosts_m, ghosts_n, ghosts_o);
    //     h_cuboid.fill(-1, false);

    //     mgcl::Cuboid h_planes(1, 1, ressize);
    //     h_planes.fill1dIndex(false);
    //     double* buf_planes = h_planes.field1d().data();

    //     mgcl::CuboidGpu d_cuboid(p.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, h_cuboid);
    //     mgcl::CuboidGpu d_planes(p.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, h_planes);

    //     // paste planes
    //     d_cuboid.pasteGhostsFromBorderPlanes(p.getContext(), p.getCommands(), p.getProgram(), &d_planes, nullptr, nullptr, nullptr);

    //     // read result
    //     d_cuboid.read(p.getCommands(), &h_cuboid, true);

    //     // Check that ghost slices are equal to the planes
    //     auto slice_front = h_cuboid.sliceIncGhosts(0, ghosts_m - 1, 0, ngh - 1, 0, ogh - 1);
    //     auto slice_back = h_cuboid.sliceIncGhosts(ghosts_m + m, mgh - 1, 0, ngh - 1, 0, ogh - 1);
    //     auto slice_top = h_cuboid.sliceIncGhosts(0, mgh - 1, 0, ghosts_n - 1, 0, ogh - 1);
    //     auto slice_bottom = h_cuboid.sliceIncGhosts(0, mgh - 1, ghosts_n + n, ngh - 1, 0, ogh - 1);
    //     auto slice_left = h_cuboid.sliceIncGhosts(0, mgh - 1, 0, ngh - 1, 0, ghosts_o - 1);
    //     auto slice_right = h_cuboid.sliceIncGhosts(0, mgh - 1, 0, ngh - 1, o + ghosts_o, ogh - 1);

    //     int idx = 0;
    //     // Back ghosts
    //     for (int i = 0; i < slice_back->getMgh(); i++)
    //         for (int j = 0; j < slice_back->getNgh(); j++)
    //             for (int k = 0; k < slice_back->getOgh(); k++)
    //             {
    //                 // No corners or edges, only ghosts directly adjacent to real back face
    //                 if (j >= ghosts_n && j < n + ghosts_n && k >= ghosts_o && k < o + ghosts_o)
    //                 {
    //                     CAPTURE(i, j, k);
    //                     REQUIRE((*slice_back)[i][j][k] == buf_planes[idx]);
    //                 }
    //                 idx++;
    //             }
    //     // Front ghosts
    //     for (int i = 0; i < slice_front->getMgh(); i++)
    //         for (int j = 0; j < slice_front->getNgh(); j++)
    //             for (int k = 0; k < slice_front->getOgh(); k++)
    //             {
    //                 // No corners or edges, only ghosts directly adjacent to real back face
    //                 if (j >= ghosts_n && j < n + ghosts_n && k >= ghosts_o && k < o + ghosts_o)
    //                 {
    //                     CAPTURE(i, j, k);
    //                     REQUIRE((*slice_front)[i][j][k] == buf_planes[idx]);
    //                 }
    //                 idx++;
    //             }
    //     // Bottom ghosts
    //     for (int j = 0; j < slice_bottom->getNgh(); j++)
    //         for (int i = 0; i < slice_bottom->getMgh(); i++)
    //             for (int k = 0; k < slice_bottom->getOgh(); k++)
    //             {
    //                 // Ignore left and right ghost cells, but include front and back ghosts
    //                 if (k >= ghosts_o && k < o + ghosts_o)
    //                 {
    //                     CAPTURE(i, j, k);
    //                     REQUIRE((*slice_bottom)[i][j][k] == buf_planes[idx]);
    //                 }
    //                 idx++;
    //             }
    //     // Top ghosts
    //     for (int j = 0; j < slice_top->getNgh(); j++)
    //         for (int i = 0; i < slice_top->getMgh(); i++)
    //             for (int k = 0; k < slice_top->getOgh(); k++)
    //             {
    //                 // Ignore left and right ghost cells, but include front and back ghosts
    //                 if (k >= ghosts_o && k < o + ghosts_o)
    //                 {
    //                     CAPTURE(i, j, k);
    //                     REQUIRE((*slice_top)[i][j][k] == buf_planes[idx]);
    //                 }
    //                 idx++;
    //             }
    //     // Right ghosts
    //     for (int k = 0; k < slice_right->getOgh(); k++)
    //         for (int i = 0; i < slice_right->getMgh(); i++)
    //             for (int j = 0; j < slice_right->getNgh(); j++)
    //             {
    //                 CAPTURE(i, j, k);
    //                 REQUIRE((*slice_right)[i][j][k] == buf_planes[idx]);
    //                 idx++;
    //             }
    //     // Left ghosts
    //     for (int k = 0; k < slice_left->getOgh(); k++)
    //         for (int i = 0; i < slice_left->getMgh(); i++)
    //             for (int j = 0; j < slice_left->getNgh(); j++)
    //             {
    //                 CAPTURE(i, j, k);
    //                 REQUIRE((*slice_left)[i][j][k] == buf_planes[idx]);
    //                 idx++;
    //             }

    //     // mgcl::Cuboid h_cuboid(m, n, o, ghosts_m, ghosts_n, ghosts_o);
    //     // h_cuboid.fill1dIndex(false);

    //     // // Create gpu buffer equal to h_cuboid, without up-to-date ghosts
    //     // mgcl::CuboidGpu c(p.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, h_cuboid);

    //     // // Update ghosts of h_cuboid
    //     // mgcl::MultigridEngine::updateGhostsSeq(h_cuboid, nullptr, true, true);

    //     // // Extract border planes from h_cuboid with now up-to-date ghosts
    //     // auto h_extractedBorders = c.extractBorderPlanes(p.getCommands(), p.getProgram(), nullptr, nullptr);

    //     // // Paste ghosts from extracted Borders to device cuboid c
    //     // c.pasteGhostsFromBorderPlanes(p.getContext(), p.getCommands(), p.getProgram(), nullptr, h_extractedBorders.get());

    //     // // Cuboids should now be equal
    //     // auto h_act = c.read(p.getCommands(), nullptr, true);

    //     // REQUIRE(h_act->isEqualAllCells(h_cuboid));

    //     // auto check = [&](mgcl::Cuboid& extractedBorders) { //
    //     //     const double* data_borders = extractedBorders.field1d().data();

    //     //     int cnt = 0;
    //     //     // front planes (yz)
    //     //     for (int i = ghosts_m; i < 2 * ghosts_m; i++) // ghm real planes in the front
    //     //         for (int j = 0; j < ngh; j++)             // all cells in y-dir
    //     //             for (int k = 0; k < ogh; k++)         // all cells in z-dir
    //     //             {
    //     //                 CAPTURE(i, j, k, cnt);
    //     //                 REQUIRE(data_borders[cnt++] == h_cuboid[i][j][k]);
    //     //             }

    //     //     // back planes (yz)
    //     //     for (int i = m; i < m + ghosts_m; i++) // ghosts_m real planes in the back
    //     //         for (int j = 0; j < ngh; j++)      // all cells in y-dir
    //     //             for (int k = 0; k < ogh; k++)  // all cells in z-dir
    //     //                 REQUIRE(data_borders[cnt++] == h_cuboid[i][j][k]);

    //     //     // top planes (xz)
    //     //     for (int j = ghosts_n; j < 2 * ghosts_n; j++)
    //     //         for (int i = 0; i < mgh; i++)
    //     //             for (int k = 0; k < ogh; k++)
    //     //                 REQUIRE(data_borders[cnt++] == h_cuboid[i][j][k]);

    //     //     // bottom planes (xz)
    //     //     for (int j = n; j < n + ghosts_n; j++)
    //     //         for (int i = 0; i < mgh; i++)
    //     //             for (int k = 0; k < ogh; k++)
    //     //                 REQUIRE(data_borders[cnt++] == h_cuboid[i][j][k]);

    //     //     // left planes (xy)
    //     //     for (int k = ghosts_o; k < 2 * ghosts_o; k++)
    //     //         for (int i = 0; i < mgh; i++)
    //     //             for (int j = 0; j < ngh; j++)
    //     //                 REQUIRE(data_borders[cnt++] == h_cuboid[i][j][k]);

    //     //     // right planes (xy)
    //     //     for (int k = o; k < o + ghosts_o; k++)
    //     //         for (int i = 0; i < mgh; i++)
    //     //             for (int j = 0; j < ngh; j++)
    //     //                 REQUIRE(data_borders[cnt++] == h_cuboid[i][j][k]);
    //     // };

    //     // SECTION("no_reuse")
    //     // {
    //     //     auto ret = c.extractBorderPlanes(p.getCommands(), p.getProgram(), nullptr, nullptr);
    //     //     REQUIRE(ret != nullptr);
    //     //     check(*ret);
    //     // }

    //     // SECTION("reuse_both")
    //     // {
    //     //     mgcl::Cuboid h_ret(1, 1, ressize, 0, 0, 0);
    //     //     h_ret.fill(-1, false);
    //     //     mgcl::CuboidGpu d_tmp(p.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, h_ret);

    //     //     c.extractBorderPlanes(p.getCommands(), p.getProgram(), &d_tmp, &h_ret);
    //     //     check(h_ret);
    //     // }

    //     // SECTION("reuse_return_buffer")
    //     // {
    //     //     mgcl::Cuboid h_ret(1, 1, ressize, 0, 0, 0);
    //     //     h_ret.fill(-1, false);

    //     //     auto ret = c.extractBorderPlanes(p.getCommands(), p.getProgram(), nullptr, &h_ret);
    //     //     REQUIRE(ret == nullptr);
    //     //     check(h_ret);
    //     // }

    //     // SECTION("reuse_device_buffer")
    //     // {
    //     //     mgcl::CuboidGpu d_tmp(p.getContext(), CL_MEM_READ_WRITE, 1, 1, ressize, 0, 0, 0);
    //     //     d_tmp.fill(p.getCommands(), -1, true);

    //     //     auto ret = c.extractBorderPlanes(p.getCommands(), p.getProgram(), &d_tmp, nullptr);
    //     //     check(*ret);
    //     // }
    // }
}