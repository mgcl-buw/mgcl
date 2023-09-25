#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <ctgmath>
#include <memory>

#include "../src/cuboid.hpp"
#include "../src/multigrid_engine.hpp"
#include "../src/opencl_helper.hpp"
#include "../src/stencil.hpp"

#include "matrix2d.hpp"
#include "test_utility.hpp"

using std::min;

TEST_CASE("VaryingStencilGpu ctor+dtor")
{
    auto deviceType = GENERATE(CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU);

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
    auto s = std::make_unique<mgcl::VaryingStencilGpu>(m, n, o, width, gh, t.getContext(), t.getCommands());
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
    auto deviceType = GENERATE(CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU);

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
    auto hgpu = std::make_unique<mgcl::VaryingStencilGpu>(m, n, o, 3, 0, t.getContext(), t.getCommands());
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
                            h_check[i][j][k][ii][jj][kk] = h[i][j][k][ii][jj][kk];
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
    auto deviceType = GENERATE(CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU);

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
        auto s = std::make_unique<mgcl::VaryingStencilGpu>(m, n, o, width, gh, t.getContext(), t.getCommands());
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
        auto s = std::make_unique<mgcl::VaryingStencilGpu>(m, n, o, width, gh, t.getContext(), t.getCommands());
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
        auto s = std::make_unique<mgcl::VaryingStencilGpu>(m, n, o, width, gh, t.getContext(), t.getCommands());
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
    auto deviceType = GENERATE(CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU);

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
        auto s = std::make_unique<mgcl::VaryingStencilGpu>(m, n, o, width, gh, t.getContext(), t.getCommands());
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
        auto s = std::make_unique<mgcl::VaryingStencilGpu>(m, n, o, width, gh, t.getContext(), t.getCommands());
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
    auto deviceType = GENERATE(CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU);

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

                        int idx_gh_cell = i * widthPow3on + j * widthPow3o + k * widthPow3;
                        int idx_real_cell = ireal * widthPow3on + jreal * widthPow3o + kreal * widthPow3;

                        // clang-format off
                        // update every stencil entry of current cell
                        for (int ii = 0; ii < width; ii++)
                        for (int jj = 0; jj < width; jj++)
                        for (int kk = 0; kk < width; kk++)
                        {
                            s3.field1d()[idx_gh_cell + ii * widthPow2 + jj * width + kk]++;

                            // check that ghost cell is written to once
                            REQUIRE(s3[i][j][k][ii][jj][kk] == 1);

                            // check that 1d index coincides with 6d index
                            REQUIRE(s3[i][j][k][ii][jj][kk] == s3.field1d()[idx_gh_cell + ii * widthPow2 + jj * width + kk]);

                            // check that calculated indices of real cell is actually a real cell
                            REQUIRE(ireal >= gh);
                            REQUIRE(ireal < m + gh);
                            REQUIRE(jreal >= gh);
                            REQUIRE(jreal < n + gh);
                            REQUIRE(kreal >= gh);
                            REQUIRE(kreal < o + gh);
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
                                    REQUIRE(s3[i][j][k][ii][jj][kk] == 1);
                                else
                                    REQUIRE(s3[i][j][k][ii][jj][kk] == 0);
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

                        int idx_gh_cell = i * widthPow3on + j * widthPow3o + k * widthPow3;
                        int idx_real_cell = ireal * widthPow3on + jreal * widthPow3o + kreal * widthPow3;

                        // clang-format off
                        // update every stencil entry of current cell
                        for (int ii = 0; ii < width; ii++)
                        for (int jj = 0; jj < width; jj++)
                        for (int kk = 0; kk < width; kk++)
                        {
                            s3.field1d()[idx_gh_cell + ii * widthPow2 + jj * width + kk]++;

                            // check that ghost cell is written to once
                            REQUIRE(s3[i][j][k][ii][jj][kk] == 1);

                            // check that 1d index coincides with 6d index
                            REQUIRE(s3[i][j][k][ii][jj][kk] == s3.field1d()[idx_gh_cell + ii * widthPow2 + jj * width + kk]);

                            // check that calculated indices of real cell is actually a real cell
                            REQUIRE(ireal >= gh);
                            REQUIRE(ireal < m + gh);
                            REQUIRE(jreal >= gh);
                            REQUIRE(jreal < n + gh);
                            REQUIRE(kreal >= gh);
                            REQUIRE(kreal < o + gh);
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
                                    REQUIRE(s3[i][j][k][ii][jj][kk] == 1);
                                else
                                    REQUIRE(s3[i][j][k][ii][jj][kk] == 0);
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
        auto s = std::make_unique<mgcl::VaryingStencilGpu>(m, n, o, width, gh, t.getContext(), t.getCommands());
        t.finish();

        // create VaryingStencil, fill with random values and copy to gpu buffer
        mgcl::VaryingStencil s3(m, n, o, 3, gh, gh, gh);
        s3.fillRandomInt();
        s->fill(s3, t.getCommands(), true);

        // update ghosts of both, host and device stencils
        s3.updateGhosts();
        s->updateGhosts(t.getProgram(), t.getCommands());
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
        auto s = std::make_unique<mgcl::VaryingStencilGpu>(m, n, o, width, gh, t.getContext(), t.getCommands());
        t.finish();

        // create VaryingStencil, fill with random values and copy to gpu buffer
        mgcl::VaryingStencil s3(m, n, o, 5, gh, gh, gh);
        s3.fillRandomInt();
        s->fill(s3, t.getCommands(), true);

        // update ghosts of both, host and device stencils
        s3.updateGhosts();
        s->updateGhosts(t.getProgram(), t.getCommands());
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
    auto deviceType = GENERATE(CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU);

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
        int wc = 5;
        int gha = gh;
        int ghb = gh;
        int ghc = gh;

        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < o; k++)
                {
                    int wa2 = wa >> 1;
                    int wc = wa + wb - 1;

                    // 1d indices
                    int wcPow2 = wc * wc;
                    int wcPow3 = wcPow2 * wc;
                    int cell_c = (i + ghc) * (n + 2 * ghc) * (o + 2 * ghc) * wcPow3 + (j + ghc) * (o + 2 * ghc) * wcPow3 + (k + ghc) * wcPow3;

                    int waPow2 = wa * wa;
                    int waPow3 = waPow2 * wa;
                    int cell_a = (i + gha) * (n + 2 * gha) * (o + 2 * gha) * waPow3 + (j + gha) * (o + 2 * gha) * waPow3 + (k + gha) * waPow3;

                    int wbPow2 = wb * wb;
                    int wbPow3 = wbPow2 * wb;

                    // clang-format off
                    for (int a_i = 0; a_i < wa; a_i++)
                    for (int a_j = 0; a_j < wa; a_j++)
                    for (int a_k = 0; a_k < wa; a_k++)
                        for (int b_i = 0; b_i < wb; b_i++)
                        for (int b_j = 0; b_j < wb; b_j++)
                        for (int b_k = 0; b_k < wb; b_k++)
                        {
                            int gpi = i + a_i - wa2 + ghb;
                            int gpj = j + a_j - wa2 + ghb;
                            int gpk = k + a_k - wa2 + ghb;

                            int cell_b = gpi * (n + 2 * ghb) * (o + 2 * ghb) * wbPow3 + gpj * (o + 2 * ghb) * wbPow3 + gpk * wbPow3;

                            int ci = a_i + b_i;
                            int cj = a_j + b_j;
                            int ck = a_k + b_k;

                            if (ci >= 0 && ci < wc &&
                                cj >= 0 && cj < wc &&
                                ck >= 0 && ck < wc)
                            {
                                REQUIRE(c.field1d()[cell_c + ci * wcPow2 + cj * wc + ck] == c[i + ghc][j + ghc][k + ghc][a_i + b_i][a_j + b_j][a_k + b_k]);
                                REQUIRE(a.field1d()[cell_a + a_i * waPow2 + a_j * wa + a_k] == a[i + gha][j + gha][k + gha][a_i][a_j][a_k]);
                                REQUIRE(b.field1d()[cell_b + b_i * wbPow2 + b_j * wb + b_k] == b[gpi][gpj][gpk][b_i][b_j][b_k]);
                                // c[cell_c + ci * wcPow2 + cj * wc + ck] +=
                                //     a[cell_a + a_i * waPow2 + a_j * wa + a_k] *
                                //     b[cell_b + b_i * wbPow2 + b_j * wb + b_k];
                                // c[i + ghc][j + ghc][k + ghc][a_i + b_i][a_j + b_j][a_k + b_k] +=
                                //     a[i + gha][j + gha][k + gha][a_i][a_j][a_k] *
                                //     b[b_i][b_j][b_k];
                            }
                        }
                    // clang-format on
                }
    }

    SECTION("widths 3 * 3")
    {
        auto a = std::make_unique<mgcl::VaryingStencilGpu>(m, n, o, 3, gh, t.getContext(), t.getCommands());
        auto b = std::make_unique<mgcl::VaryingStencilGpu>(m, n, o, 3, gh, t.getContext(), t.getCommands());
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
        b->updateGhosts(t.getProgram(), t.getCommands());

        auto c_h = a_h.multiply(b_h, 2, nullptr, true, true);
        auto c = a->multiply(*b, 2, t.getProgram(), t.getCommands(), t.getContext(), nullptr, true, true);
        t.finish();

        auto ret = c.read(t.getCommands(), true);
        t.finish();

        // check results
        REQUIRE(c.getM() == c_h.getDim1());
        REQUIRE(c.getN() == c_h.getDim2());
        REQUIRE(c.getO() == c_h.getDim3());
        REQUIRE(c.getWidth() == c_h.getDim4());
        REQUIRE(c.getWidth() == c_h.getDim5());
        REQUIRE(c.getWidth() == c_h.getDim6());
        REQUIRE(c.getGh() == c_h.getGhostsDim1());
        REQUIRE(c.getGh() == c_h.getGhostsDim2());
        REQUIRE(c.getGh() == c_h.getGhostsDim3());

        // clang-format off
        for (int i = 0; i < c_h.getDim1gh(); i++)
        for (int j = 0; j < c_h.getDim2gh(); j++)
        for (int k = 0; k < c_h.getDim3gh(); k++)
            for (int ii = 0; ii < c_h.getDim4(); ii++)
            for (int jj = 0; jj < c_h.getDim5(); jj++)
            for (int kk = 0; kk < c_h.getDim6(); kk++)
            {
                REQUIRE(c_h[i][j][k][ii][jj][kk] == ret[i][j][k][ii][jj][kk]);
            }
        // clang-format on
    }

    SECTION("widths 3 * 5")
    {
        auto a = std::make_unique<mgcl::VaryingStencilGpu>(m, n, o, 3, gh, t.getContext(), t.getCommands());
        auto b = std::make_unique<mgcl::VaryingStencilGpu>(m, n, o, 5, gh, t.getContext(), t.getCommands());
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
        b->updateGhosts(t.getProgram(), t.getCommands());

        auto c_h = a_h.multiply(b_h, 2, nullptr, true, true);
        auto c = a->multiply(*b, 2, t.getProgram(), t.getCommands(), t.getContext(), nullptr, true, true);
        t.finish();

        auto ret = c.read(t.getCommands(), true);
        t.finish();

        // check results
        REQUIRE(c.getM() == c_h.getDim1());
        REQUIRE(c.getN() == c_h.getDim2());
        REQUIRE(c.getO() == c_h.getDim3());
        REQUIRE(c.getWidth() == c_h.getDim4());
        REQUIRE(c.getWidth() == c_h.getDim5());
        REQUIRE(c.getWidth() == c_h.getDim6());
        REQUIRE(c.getGh() == c_h.getGhostsDim1());
        REQUIRE(c.getGh() == c_h.getGhostsDim2());
        REQUIRE(c.getGh() == c_h.getGhostsDim3());

        // clang-format off
        for (int i = 0; i < c_h.getDim1gh(); i++)
        for (int j = 0; j < c_h.getDim2gh(); j++)
        for (int k = 0; k < c_h.getDim3gh(); k++)
            for (int ii = 0; ii < c_h.getDim4(); ii++)
            for (int jj = 0; jj < c_h.getDim5(); jj++)
            for (int kk = 0; kk < c_h.getDim6(); kk++)
            {
                REQUIRE(c_h[i][j][k][ii][jj][kk] == ret[i][j][k][ii][jj][kk]);
            }
        // clang-format on
    }

    SECTION("widths 5 * 3")
    {
        auto a = std::make_unique<mgcl::VaryingStencilGpu>(m, n, o, 5, gh, t.getContext(), t.getCommands());
        auto b = std::make_unique<mgcl::VaryingStencilGpu>(m, n, o, 3, gh, t.getContext(), t.getCommands());
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
        b->updateGhosts(t.getProgram(), t.getCommands());

        auto c_h = a_h.multiply(b_h, 2, nullptr, true, true);
        auto c = a->multiply(*b, 2, t.getProgram(), t.getCommands(), t.getContext(), nullptr, true, true);
        t.finish();

        auto ret = c.read(t.getCommands(), true);
        t.finish();

        // check results
        REQUIRE(c.getM() == c_h.getDim1());
        REQUIRE(c.getN() == c_h.getDim2());
        REQUIRE(c.getO() == c_h.getDim3());
        REQUIRE(c.getWidth() == c_h.getDim4());
        REQUIRE(c.getWidth() == c_h.getDim5());
        REQUIRE(c.getWidth() == c_h.getDim6());
        REQUIRE(c.getGh() == c_h.getGhostsDim1());
        REQUIRE(c.getGh() == c_h.getGhostsDim2());
        REQUIRE(c.getGh() == c_h.getGhostsDim3());

        // clang-format off
        for (int i = 0; i < c_h.getDim1gh(); i++)
        for (int j = 0; j < c_h.getDim2gh(); j++)
        for (int k = 0; k < c_h.getDim3gh(); k++)
            for (int ii = 0; ii < c_h.getDim4(); ii++)
            for (int jj = 0; jj < c_h.getDim5(); jj++)
            for (int kk = 0; kk < c_h.getDim6(); kk++)
            {
                REQUIRE(c_h[i][j][k][ii][jj][kk] == ret[i][j][k][ii][jj][kk]);
            }
        // clang-format on
    }

    SECTION("widths 5 * 5")
    {
        auto a = std::make_unique<mgcl::VaryingStencilGpu>(m, n, o, 5, gh, t.getContext(), t.getCommands());
        auto b = std::make_unique<mgcl::VaryingStencilGpu>(m, n, o, 5, gh, t.getContext(), t.getCommands());
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
        b->updateGhosts(t.getProgram(), t.getCommands());

        auto c_h = a_h.multiply(b_h, 2, nullptr, true, true);
        auto c = a->multiply(*b, 2, t.getProgram(), t.getCommands(), t.getContext(), nullptr, true, true);
        t.finish();

        auto ret = c.read(t.getCommands(), true);
        t.finish();

        // check results
        REQUIRE(c.getM() == c_h.getDim1());
        REQUIRE(c.getN() == c_h.getDim2());
        REQUIRE(c.getO() == c_h.getDim3());
        REQUIRE(c.getWidth() == c_h.getDim4());
        REQUIRE(c.getWidth() == c_h.getDim5());
        REQUIRE(c.getWidth() == c_h.getDim6());
        REQUIRE(c.getGh() == c_h.getGhostsDim1());
        REQUIRE(c.getGh() == c_h.getGhostsDim2());
        REQUIRE(c.getGh() == c_h.getGhostsDim3());

        // clang-format off
        for (int i = 0; i < c_h.getDim1gh(); i++)
        for (int j = 0; j < c_h.getDim2gh(); j++)
        for (int k = 0; k < c_h.getDim3gh(); k++)
            for (int ii = 0; ii < c_h.getDim4(); ii++)
            for (int jj = 0; jj < c_h.getDim5(); jj++)
            for (int kk = 0; kk < c_h.getDim6(); kk++)
            {
                REQUIRE(c_h[i][j][k][ii][jj][kk] == ret[i][j][k][ii][jj][kk]);
            }
        // clang-format on
    }
}

TEST_CASE("VaryingStencilGpu::multiply(fix)")
{
    auto deviceType = GENERATE(CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU);

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
        auto a = std::make_unique<mgcl::VaryingStencilGpu>(m, n, o, 3, gh, t.getContext(), t.getCommands());
        auto b = std::make_unique<mgcl::FixedStencilGpu>(3, t.getContext(), t.getCommands());
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
        auto c = a->multiply(*b, 2, t.getProgram(), t.getCommands(), t.getContext(), nullptr, true, true);
        t.finish();

        auto ret = c.read(t.getCommands(), true);
        t.finish();

        // check results
        REQUIRE(c.getM() == c_h.getDim1());
        REQUIRE(c.getN() == c_h.getDim2());
        REQUIRE(c.getO() == c_h.getDim3());
        REQUIRE(c.getWidth() == c_h.getDim4());
        REQUIRE(c.getWidth() == c_h.getDim5());
        REQUIRE(c.getWidth() == c_h.getDim6());
        REQUIRE(c.getGh() == c_h.getGhostsDim1());
        REQUIRE(c.getGh() == c_h.getGhostsDim2());
        REQUIRE(c.getGh() == c_h.getGhostsDim3());

        // clang-format off
        for (int i = 0; i < c_h.getDim1gh(); i++)
        for (int j = 0; j < c_h.getDim2gh(); j++)
        for (int k = 0; k < c_h.getDim3gh(); k++)
            for (int ii = 0; ii < c_h.getDim4(); ii++)
            for (int jj = 0; jj < c_h.getDim5(); jj++)
            for (int kk = 0; kk < c_h.getDim6(); kk++)
            {
                REQUIRE(c_h[i][j][k][ii][jj][kk] == ret[i][j][k][ii][jj][kk]);
            }
        // clang-format on
    }

    SECTION("widths 3 * 5")
    {
        auto a = std::make_unique<mgcl::VaryingStencilGpu>(m, n, o, 3, gh, t.getContext(), t.getCommands());
        auto b = std::make_unique<mgcl::FixedStencilGpu>(5, t.getContext(), t.getCommands());
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
        auto c = a->multiply(*b, 2, t.getProgram(), t.getCommands(), t.getContext(), nullptr, true, true);
        t.finish();

        auto ret = c.read(t.getCommands(), true);
        t.finish();

        // check results
        REQUIRE(c.getM() == c_h.getDim1());
        REQUIRE(c.getN() == c_h.getDim2());
        REQUIRE(c.getO() == c_h.getDim3());
        REQUIRE(c.getWidth() == c_h.getDim4());
        REQUIRE(c.getWidth() == c_h.getDim5());
        REQUIRE(c.getWidth() == c_h.getDim6());
        REQUIRE(c.getGh() == c_h.getGhostsDim1());
        REQUIRE(c.getGh() == c_h.getGhostsDim2());
        REQUIRE(c.getGh() == c_h.getGhostsDim3());

        // clang-format off
        for (int i = 0; i < c_h.getDim1gh(); i++)
        for (int j = 0; j < c_h.getDim2gh(); j++)
        for (int k = 0; k < c_h.getDim3gh(); k++)
            for (int ii = 0; ii < c_h.getDim4(); ii++)
            for (int jj = 0; jj < c_h.getDim5(); jj++)
            for (int kk = 0; kk < c_h.getDim6(); kk++)
            {
                REQUIRE(c_h[i][j][k][ii][jj][kk] == ret[i][j][k][ii][jj][kk]);
            }
        // clang-format on
    }

    SECTION("widths 5 * 3")
    {
        auto a = std::make_unique<mgcl::VaryingStencilGpu>(m, n, o, 5, gh, t.getContext(), t.getCommands());
        auto b = std::make_unique<mgcl::FixedStencilGpu>(3, t.getContext(), t.getCommands());
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
        auto c = a->multiply(*b, 2, t.getProgram(), t.getCommands(), t.getContext(), nullptr, true, true);
        t.finish();

        auto ret = c.read(t.getCommands(), true);
        t.finish();

        // check results
        REQUIRE(c.getM() == c_h.getDim1());
        REQUIRE(c.getN() == c_h.getDim2());
        REQUIRE(c.getO() == c_h.getDim3());
        REQUIRE(c.getWidth() == c_h.getDim4());
        REQUIRE(c.getWidth() == c_h.getDim5());
        REQUIRE(c.getWidth() == c_h.getDim6());
        REQUIRE(c.getGh() == c_h.getGhostsDim1());
        REQUIRE(c.getGh() == c_h.getGhostsDim2());
        REQUIRE(c.getGh() == c_h.getGhostsDim3());

        // clang-format off
        for (int i = 0; i < c_h.getDim1gh(); i++)
        for (int j = 0; j < c_h.getDim2gh(); j++)
        for (int k = 0; k < c_h.getDim3gh(); k++)
            for (int ii = 0; ii < c_h.getDim4(); ii++)
            for (int jj = 0; jj < c_h.getDim5(); jj++)
            for (int kk = 0; kk < c_h.getDim6(); kk++)
            {
                REQUIRE(c_h[i][j][k][ii][jj][kk] == ret[i][j][k][ii][jj][kk]);
            }
        // clang-format on
    }

    SECTION("widths 5 * 5")
    {
        auto a = std::make_unique<mgcl::VaryingStencilGpu>(m, n, o, 5, gh, t.getContext(), t.getCommands());
        auto b = std::make_unique<mgcl::FixedStencilGpu>(5, t.getContext(), t.getCommands());
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
        auto c = a->multiply(*b, 2, t.getProgram(), t.getCommands(), t.getContext(), nullptr, true, true);
        t.finish();

        auto ret = c.read(t.getCommands(), true);
        t.finish();

        // check results
        REQUIRE(c.getM() == c_h.getDim1());
        REQUIRE(c.getN() == c_h.getDim2());
        REQUIRE(c.getO() == c_h.getDim3());
        REQUIRE(c.getWidth() == c_h.getDim4());
        REQUIRE(c.getWidth() == c_h.getDim5());
        REQUIRE(c.getWidth() == c_h.getDim6());
        REQUIRE(c.getGh() == c_h.getGhostsDim1());
        REQUIRE(c.getGh() == c_h.getGhostsDim2());
        REQUIRE(c.getGh() == c_h.getGhostsDim3());

        // clang-format off
        for (int i = 0; i < c_h.getDim1gh(); i++)
        for (int j = 0; j < c_h.getDim2gh(); j++)
        for (int k = 0; k < c_h.getDim3gh(); k++)
            for (int ii = 0; ii < c_h.getDim4(); ii++)
            for (int jj = 0; jj < c_h.getDim5(); jj++)
            for (int kk = 0; kk < c_h.getDim6(); kk++)
            {
                REQUIRE(c_h[i][j][k][ii][jj][kk] == ret[i][j][k][ii][jj][kk]);
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
            int ci = a_i + b_i;
            int cj = a_j + b_j;
            int ck = a_k + b_k;
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
            int ci = a_i + b_i;
            int cj = a_j + b_j;
            int ck = a_k + b_k;
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
    auto deviceType = GENERATE(CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU);

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
    SECTION("indices")
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

        // clang-format off
        // simulate call with one work-item per cell using these 3 for loops
        for (int i = 2; i < a_2h.getDim1() + 2; i++)
        for (int j = 2; j < a_2h.getDim2() + 2; j++)
        for (int k = 2; k < a_2h.getDim3() + 2; k++)
        {
            int i2 = (i - 2) * 2 + 1;
            int j2 = (j - 2) * 2 + 1;
            int k2 = (k - 2) * 2 + 1;

            // 7^3 = 343
            int cell_h = i2 * (2 * n + 2 * ghin) * (2 * o + 2 * ghin) * 343 + j2 * (2 * o + 2 * ghin) * 343 + k2 * 343;

            // 3^3 = 27
            int cell_h2 = i * (n + 2 * ghout) * (o + 2 * ghout) * 27 + j * (o + 2 * ghout) * 27 + k * 27;

            for (int ii = 0, ii2 = 1; ii < 3; ii++, ii2 += 2)
            for (int jj = 0, jj2 = 1; jj < 3; jj++, jj2 += 2)
            for (int kk = 0, kk2 = 1; kk < 3; kk++, kk2 += 2)
            {
                REQUIRE(a_2h[i][j][k][ii][jj][kk] == a_2h.field1d()[cell_h2 + ii * 9 + jj * 3 + kk]);
                REQUIRE(a_h[i2][j2][k2][ii2][jj2][kk2] == a_h.field1d()[cell_h + ii2 * 49 + jj2 * 7 + kk2]);
            }
        }
        // clang-format on
    }

    // This section checks if the actual calculation is correct by checking results vs the sequential version.
    SECTION("vs seq")
    {
        mgcl::VaryingStencilGpu a_gpu(m, n, o, 7, gh, t.getContext(), t.getCommands());

        // create VaryingStencil, fill with random values and copy to gpu buffer
        mgcl::VaryingStencil a_h(m, n, o, 7, gh, gh, gh);
        a_h.fillRandomInt();
        a_gpu.fill(a_h, t.getCommands(), true);
        t.finish();

        // Cut stencil from 7x7x7 down to 3x3x3, i.e. copy only selected values to new stencil, skipping ghosts.
        mgcl::VaryingStencil a_2h(a_h.getDim1() >> 1, a_h.getDim2() >> 1, a_h.getDim3() >> 1, 3, 2, 2, 2);
        // clang-format off
        for (int i = 2, i2 = 1; i < a_2h.getDim1() + 2; i++, i2 += 2)
        for (int j = 2, j2 = 1; j < a_2h.getDim2() + 2; j++, j2 += 2)
        for (int k = 2, k2 = 1; k < a_2h.getDim3() + 2; k++, k2 += 2)
            for (int ii = 0, ii2 = 1; ii < 3; ii++, ii2 += 2)
            for (int jj = 0, jj2 = 1; jj < 3; jj++, jj2 += 2)
            for (int kk = 0, kk2 = 1; kk < 3; kk++, kk2 += 2)
            {
                a_2h[i][j][k][ii][jj][kk] = a_h[i2][j2][k2][ii2][jj2][kk2];
            }
        // clang-format on

        // cut on gpu and read back result
        auto a_2h_gpu = a_gpu.cutFromW7ToW3(t.getProgram(), t.getCommands(), t.getContext(), 2);
        auto ret = a_2h_gpu.read(t.getCommands(), true);
        t.finish();

        REQUIRE(ret.getDim1() == a_2h.getDim1());
        REQUIRE(ret.getDim2() == a_2h.getDim2());
        REQUIRE(ret.getDim3() == a_2h.getDim3());
        REQUIRE(ret.getDim4() == a_2h.getDim4());
        REQUIRE(ret.getDim5() == a_2h.getDim5());
        REQUIRE(ret.getDim6() == a_2h.getDim6());
        REQUIRE(ret.getDim1gh() == a_2h.getDim1gh());
        REQUIRE(ret.getDim2gh() == a_2h.getDim2gh());
        REQUIRE(ret.getDim3gh() == a_2h.getDim3gh());
        REQUIRE(ret.getDim4gh() == a_2h.getDim4gh());
        REQUIRE(ret.getDim5gh() == a_2h.getDim5gh());
        REQUIRE(ret.getDim6gh() == a_2h.getDim6gh());

        REQUIRE(ret.isEqual(a_2h));
    }
}

TEST_CASE("FixedStencilGpu ctor+dtor")
{
    auto deviceType = GENERATE(CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU);

    if (!mgcl_test::TestUtility::deviceAvailable("", deviceType))
    {
        std::string typeName = deviceType == CL_DEVICE_TYPE_GPU ? "CL_DEVICE_TYPE_GPU" : "CL_DEVICE_TYPE_CPU";
        std::cout << "Skipping non-available device type '" << typeName << "'" << std::endl;
        return;
    }

    mgcl_test::TestUtility t(deviceType);

    int width = 3;
    int gh = 2;
    auto s = std::make_unique<mgcl::FixedStencilGpu>(width, t.getContext(), t.getCommands());
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
    auto deviceType = GENERATE(CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU);

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
        auto a = std::make_unique<mgcl::FixedStencilGpu>(3, t.getContext(), t.getCommands());
        auto b = std::make_unique<mgcl::VaryingStencilGpu>(m, n, o, 3, gh, t.getContext(), t.getCommands());
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
        b->updateGhosts(t.getProgram(), t.getCommands());

        auto c_h = a_h.multiply(b_h, 2, nullptr, true, true);
        auto c = a->multiply(*b, 2, t.getProgram(), t.getCommands(), t.getContext(), nullptr, true, true);
        t.finish();

        auto ret = c.read(t.getCommands(), true);
        t.finish();

        // check results
        REQUIRE(c.getM() == c_h.getDim1());
        REQUIRE(c.getN() == c_h.getDim2());
        REQUIRE(c.getO() == c_h.getDim3());
        REQUIRE(c.getWidth() == c_h.getDim4());
        REQUIRE(c.getWidth() == c_h.getDim5());
        REQUIRE(c.getWidth() == c_h.getDim6());
        REQUIRE(c.getGh() == c_h.getGhostsDim1());
        REQUIRE(c.getGh() == c_h.getGhostsDim2());
        REQUIRE(c.getGh() == c_h.getGhostsDim3());

        // clang-format off
        for (int i = 0; i < c_h.getDim1gh(); i++)
        for (int j = 0; j < c_h.getDim2gh(); j++)
        for (int k = 0; k < c_h.getDim3gh(); k++)
            for (int ii = 0; ii < c_h.getDim4(); ii++)
            for (int jj = 0; jj < c_h.getDim5(); jj++)
            for (int kk = 0; kk < c_h.getDim6(); kk++)
            {
                REQUIRE(c_h[i][j][k][ii][jj][kk] == ret[i][j][k][ii][jj][kk]);
            }
        // clang-format on
    }

    SECTION("widths 3 * 5")
    {
        auto a = std::make_unique<mgcl::FixedStencilGpu>(3, t.getContext(), t.getCommands());
        auto b = std::make_unique<mgcl::VaryingStencilGpu>(m, n, o, 5, gh, t.getContext(), t.getCommands());
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
        b->updateGhosts(t.getProgram(), t.getCommands());

        auto c_h = a_h.multiply(b_h, 2, nullptr, true, true);
        auto c = a->multiply(*b, 2, t.getProgram(), t.getCommands(), t.getContext(), nullptr, true, true);
        t.finish();

        auto ret = c.read(t.getCommands(), true);
        t.finish();

        // check results
        REQUIRE(c.getM() == c_h.getDim1());
        REQUIRE(c.getN() == c_h.getDim2());
        REQUIRE(c.getO() == c_h.getDim3());
        REQUIRE(c.getWidth() == c_h.getDim4());
        REQUIRE(c.getWidth() == c_h.getDim5());
        REQUIRE(c.getWidth() == c_h.getDim6());
        REQUIRE(c.getGh() == c_h.getGhostsDim1());
        REQUIRE(c.getGh() == c_h.getGhostsDim2());
        REQUIRE(c.getGh() == c_h.getGhostsDim3());

        // clang-format off
        for (int i = 0; i < c_h.getDim1gh(); i++)
        for (int j = 0; j < c_h.getDim2gh(); j++)
        for (int k = 0; k < c_h.getDim3gh(); k++)
            for (int ii = 0; ii < c_h.getDim4(); ii++)
            for (int jj = 0; jj < c_h.getDim5(); jj++)
            for (int kk = 0; kk < c_h.getDim6(); kk++)
            {
                REQUIRE(c_h[i][j][k][ii][jj][kk] == ret[i][j][k][ii][jj][kk]);
            }
        // clang-format on
    }

    SECTION("widths 5 * 3")
    {
        auto a = std::make_unique<mgcl::FixedStencilGpu>(5, t.getContext(), t.getCommands());
        auto b = std::make_unique<mgcl::VaryingStencilGpu>(m, n, o, 3, gh, t.getContext(), t.getCommands());
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
        b->updateGhosts(t.getProgram(), t.getCommands());

        auto c_h = a_h.multiply(b_h, 2, nullptr, true, true);
        auto c = a->multiply(*b, 2, t.getProgram(), t.getCommands(), t.getContext(), nullptr, true, true);
        t.finish();

        auto ret = c.read(t.getCommands(), true);
        t.finish();

        // check results
        REQUIRE(c.getM() == c_h.getDim1());
        REQUIRE(c.getN() == c_h.getDim2());
        REQUIRE(c.getO() == c_h.getDim3());
        REQUIRE(c.getWidth() == c_h.getDim4());
        REQUIRE(c.getWidth() == c_h.getDim5());
        REQUIRE(c.getWidth() == c_h.getDim6());
        REQUIRE(c.getGh() == c_h.getGhostsDim1());
        REQUIRE(c.getGh() == c_h.getGhostsDim2());
        REQUIRE(c.getGh() == c_h.getGhostsDim3());

        // clang-format off
        for (int i = 0; i < c_h.getDim1gh(); i++)
        for (int j = 0; j < c_h.getDim2gh(); j++)
        for (int k = 0; k < c_h.getDim3gh(); k++)
            for (int ii = 0; ii < c_h.getDim4(); ii++)
            for (int jj = 0; jj < c_h.getDim5(); jj++)
            for (int kk = 0; kk < c_h.getDim6(); kk++)
            {
                REQUIRE(c_h[i][j][k][ii][jj][kk] == ret[i][j][k][ii][jj][kk]);
            }
        // clang-format on
    }

    SECTION("widths 5 * 5")
    {
        auto a = std::make_unique<mgcl::FixedStencilGpu>(5, t.getContext(), t.getCommands());
        auto b = std::make_unique<mgcl::VaryingStencilGpu>(m, n, o, 5, gh, t.getContext(), t.getCommands());
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
        b->updateGhosts(t.getProgram(), t.getCommands());

        auto c_h = a_h.multiply(b_h, 2, nullptr, true, true);
        auto c = a->multiply(*b, 2, t.getProgram(), t.getCommands(), t.getContext(), nullptr, true, true);
        t.finish();

        auto ret = c.read(t.getCommands(), true);
        t.finish();

        // check results
        REQUIRE(c.getM() == c_h.getDim1());
        REQUIRE(c.getN() == c_h.getDim2());
        REQUIRE(c.getO() == c_h.getDim3());
        REQUIRE(c.getWidth() == c_h.getDim4());
        REQUIRE(c.getWidth() == c_h.getDim5());
        REQUIRE(c.getWidth() == c_h.getDim6());
        REQUIRE(c.getGh() == c_h.getGhostsDim1());
        REQUIRE(c.getGh() == c_h.getGhostsDim2());
        REQUIRE(c.getGh() == c_h.getGhostsDim3());

        // clang-format off
        for (int i = 0; i < c_h.getDim1gh(); i++)
        for (int j = 0; j < c_h.getDim2gh(); j++)
        for (int k = 0; k < c_h.getDim3gh(); k++)
            for (int ii = 0; ii < c_h.getDim4(); ii++)
            for (int jj = 0; jj < c_h.getDim5(); jj++)
            for (int kk = 0; kk < c_h.getDim6(); kk++)
            {
                REQUIRE(c_h[i][j][k][ii][jj][kk] == ret[i][j][k][ii][jj][kk]);
            }
        // clang-format on
    }
}

TEST_CASE("FixedStencilGpu::fill")
{
    auto deviceType = GENERATE(CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU);

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
        auto s = std::make_unique<mgcl::FixedStencilGpu>(width, t.getContext(), t.getCommands());
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
        auto s = std::make_unique<mgcl::FixedStencilGpu>(width, t.getContext(), t.getCommands());
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
        auto s = std::make_unique<mgcl::FixedStencilGpu>(width, t.getContext(), t.getCommands());
        t.finish();

        // widths do not match
        mgcl::FixedStencil s3(5);
        REQUIRE_THROWS(s->fill(s3, t.getCommands(), true));
    }
}

TEST_CASE("FixedStencilGpu::read")
{
    auto deviceType = GENERATE(CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU);

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
        auto s = std::make_unique<mgcl::FixedStencilGpu>(width, t.getContext(), t.getCommands());
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
        mgcl::FixedStencilGpu s(width, t.getContext(), t.getCommands());
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
    auto deviceType = GENERATE(CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU);

    if (!mgcl_test::TestUtility::deviceAvailable("", deviceType))
    {
        std::string typeName = deviceType == CL_DEVICE_TYPE_GPU ? "CL_DEVICE_TYPE_GPU" : "CL_DEVICE_TYPE_CPU";
        std::cout << "Skipping non-available device type '" << typeName << "'" << std::endl;
        return;
    }

    mgcl_test::TestUtility t(deviceType);

    auto rgpu = mgcl::create3dFullWeightRestrictionStencilGpu(t.getContext(), t.getCommands());
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
    auto deviceType = GENERATE(CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU);

    if (!mgcl_test::TestUtility::deviceAvailable("", deviceType))
    {
        std::string typeName = deviceType == CL_DEVICE_TYPE_GPU ? "CL_DEVICE_TYPE_GPU" : "CL_DEVICE_TYPE_CPU";
        std::cout << "Skipping non-available device type '" << typeName << "'" << std::endl;
        return;
    }

    mgcl_test::TestUtility t(deviceType);

    auto rgpu = mgcl::create3dBilinearProlongationStencilGpu(t.getContext(), t.getCommands());
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
