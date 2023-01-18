#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <ctgmath>
#include <memory>

#include "../cuboid.hpp"
#include "../multigrid_engine.hpp"
#include "../opencl_helper.hpp"
#include "../stencil.hpp"

#include "matrix2d.hpp"
#include "test_utility.hpp"

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
        mgcl::VaryingStencil3x3x3 s3(m, n, o, gh, gh, gh);
        s3.fillRandomInt();
        s->fill(s3, t.getCommands());

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
        mgcl::VaryingStencil5x5x5 s3(m, n, o, gh, gh, gh);
        s3.fillRandomInt();
        s->fill(s3, t.getCommands());

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
        mgcl::VaryingStencil5x5x5 s3(m, n, o, gh, gh, gh);
        REQUIRE_THROWS(s->fill(s3, t.getCommands()));

        // grid sizes do not match
        mgcl::VaryingStencil5x5x5 s35(m * 2, n * 3, o * 4, gh, gh, gh);
        REQUIRE_THROWS(s->fill(s35, t.getCommands()));
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
        mgcl::VaryingStencil3x3x3 s3(m, n, o, gh, gh, gh);
        s3.fillRandomInt();
        s->fill(s3, t.getCommands());

        // read buffer
        auto ret = s->read<3>(t.getCommands());
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
        mgcl::VaryingStencil5x5x5 s3(m, n, o, gh, gh, gh);
        s3.fillRandomInt();
        s->fill(s3, t.getCommands());

        // read buffer
        auto ret = s->read<5>(t.getCommands());
        t.finish();

        // check results
        for (int i = 0; i < ret.field1d().size(); i++)
            REQUIRE(ret.field1d()[i] == s3.field1d()[i]);
    }

    SECTION("throwing")
    {
        int width = 3;
        auto s = std::make_unique<mgcl::VaryingStencilGpu>(m, n, o, width, gh, t.getContext(), t.getCommands());
        t.finish();

        REQUIRE_THROWS(s->read<5>(t.getCommands()));
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

        mgcl::VaryingStencil3x3x3 s3(m, n, o, gh, gh, gh);
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

        mgcl::VaryingStencil3x3x3 s3(m, n, o, gh, gh, gh);
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

    int m = GENERATE(2, 3, 4);
    int n = GENERATE(2, 3, 4);
    int o = GENERATE(2, 3, 4);
    int gh = GENERATE(2, 3, 4);

    SECTION("VaryingStencil3x3x3")
    {
        int width = 3;
        auto s = std::make_unique<mgcl::VaryingStencilGpu>(m, n, o, width, gh, t.getContext(), t.getCommands());
        t.finish();

        // create VaryingStencil, fill with random values and copy to gpu buffer
        mgcl::VaryingStencil3x3x3 s3(m, n, o, gh, gh, gh);
        s3.fillRandomInt();
        s->fill(s3, t.getCommands());

        // update ghosts of both, host and device stencils
        s3.updateGhosts();
        s->updateGhosts(t.getProgram(), t.getCommands(), t.getContext());
        t.finish();

        // read buffer
        auto ret = s->read<3>(t.getCommands());
        t.finish();

        s3.dumpToFile("gh_s3.csv");
        ret.dumpToFile("gh_ret.csv");

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
        mgcl::VaryingStencil5x5x5 s3(m, n, o, gh, gh, gh);
        s3.fillRandomInt();
        s->fill(s3, t.getCommands());

        // update ghosts of both, host and device stencils
        s3.updateGhosts();
        s->updateGhosts(t.getProgram(), t.getCommands(), t.getContext());
        t.finish();

        // read buffer
        auto ret = s->read<5>(t.getCommands());
        t.finish();

        // check results
        for (int i = 0; i < ret.field1d().size(); i++)
            REQUIRE(ret.field1d()[i] == s3.field1d()[i]);
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
        mgcl::FixedStencil3x3x3 s3;
        s3.fillRandom();
        s->fill(s3, t.getCommands());

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
        mgcl::FixedStencil<5> s3;
        s3.fillRandom();
        s->fill(s3, t.getCommands());

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
        mgcl::FixedStencil<5> s3;
        REQUIRE_THROWS(s->fill(s3, t.getCommands()));
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
        mgcl::FixedStencil3x3x3 s3;
        s3.fillRandom();
        s->fill(s3, t.getCommands());
        t.finish();

        // read buffer
        auto ret = s->read<3>(t.getCommands());
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
        mgcl::FixedStencil<5> s3;
        s3.fillRandom();
        s.fill(s3, t.getCommands());
        t.finish();

        // read buffer
        auto ret = s.read<5>(t.getCommands());
        t.finish();

        // check results
        REQUIRE(ret.field1d().size() == s3.field1d().size());

        for (int i = 0; i < ret.field1d().size(); i++)
            REQUIRE(ret.field1d()[i] == s3.field1d()[i]);

        t.finish();
    }

    SECTION("throwing")
    {
        int width = 3;
        auto s = std::make_unique<mgcl::FixedStencilGpu>(width, t.getContext(), t.getCommands());
        t.finish();

        REQUIRE_THROWS(s->read<5>(t.getCommands()));

        t.finish();
    }
}
