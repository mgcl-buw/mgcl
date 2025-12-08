#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "../src/mgcl/cuboid.hpp"
#include "cli_args.hpp"
#include "device_type_generator.hpp"
#include "test_utility.hpp"

TEST_CASE("TestUtility setup")
{
    auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));

    mgcl_test::TestUtility tu(deviceType);
    REQUIRE(tu.getProblem().getOpenCLHelper().isInitialized());
}

TEST_CASE("TestUtility setup reusing Problem")
{
    auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));

    auto p = std::make_shared<mgcl::Problem>(4, 4, 4);
    p->setDeviceType(deviceType);
    mgcl_test::TestUtility tu(p);
    REQUIRE(tu.getProblem().getOpenCLHelper().isInitialized());
}

TEST_CASE("TestUtility createOpenCLBuffer + readOpenCLBuffer")
{
    auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));

    mgcl_test::TestUtility tu(deviceType);
    REQUIRE(tu.getProblem().getOpenCLHelper().isInitialized());

    mgcl::Cuboid c(4, 4, 4, 3.0);
    auto buf = tu.createOpenCLBuffer(c);
    REQUIRE(buf);

    auto c2 = tu.readOpenCLBuffer(buf, 4, 4, 4);
    for (int i = 0; i < c2->field1d().size(); i++)
        REQUIRE(c2->field1d()[i] == c.field1d()[i]);
}

TEST_CASE("TestUtility readOpenCLBuffer nullptr")
{
    auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));

    mgcl_test::TestUtility tu(deviceType);
    REQUIRE(tu.getProblem().getOpenCLHelper().isInitialized());
    REQUIRE_THROWS_AS(tu.readOpenCLBuffer(nullptr, 4, 4, 4), std::invalid_argument);
}

TEST_CASE("TestUtility deviceAvailable")
{
    auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));

    mgcl_test::TestUtility tu(deviceType);
    int err;
    cl_char device_name[1024] = {0};
    cl_device_type device_type;

    err = clGetDeviceInfo(tu.getDeviceId(), CL_DEVICE_NAME, sizeof(device_name), &device_name, NULL);
    mgcl::mgclCheckError(err, "clGetDeviceInfo(CL_DEVICE_NAME)");

    err = clGetDeviceInfo(tu.getDeviceId(), CL_DEVICE_TYPE, sizeof(device_type), &device_type, NULL);
    mgcl::mgclCheckError(err, "clGetDeviceInfo(CL_DEVICE_TYPE)");

    REQUIRE(mgcl_test::TestUtility::deviceAvailable(std::string((char*)device_name), device_type));
    REQUIRE_FALSE(mgcl_test::TestUtility::deviceAvailable("akljshnfklfha", device_type));
}

TEST_CASE("TestUtility setup deviceName")
{
    if (mgcl_test::TestUtility::deviceAvailable("Quadro", CL_DEVICE_TYPE_ALL))
    {
        REQUIRE_NOTHROW(mgcl_test::TestUtility("Quadro"));
    }
    REQUIRE_THROWS(mgcl_test::TestUtility("aksnhjisagnfsif"));
}

TEST_CASE("TestUtility setup deviceType")
{
    if (mgcl_test::TestUtility::deviceAvailable("", CL_DEVICE_TYPE_GPU))
        REQUIRE_NOTHROW(mgcl_test::TestUtility(CL_DEVICE_TYPE_GPU));
    else
        REQUIRE_THROWS(mgcl_test::TestUtility(CL_DEVICE_TYPE_GPU));

    if (mgcl_test::TestUtility::deviceAvailable("", CL_DEVICE_TYPE_CPU))
        REQUIRE_NOTHROW(mgcl_test::TestUtility(CL_DEVICE_TYPE_CPU));
    else
        REQUIRE_THROWS(mgcl_test::TestUtility(CL_DEVICE_TYPE_CPU));
}

TEST_CASE("mgcl_test::copyCuboidToCuboidBS")
{
    int m = 4;
    int n = 4;
    int o = 4;
    int blocksize = 8;

    mgcl::Cuboid src(m, n, o);
    src.fillRandom();

    SECTION("scaling 2")
    {
        int scale = 2;
        mgcl::CuboidBS dst_act(m / scale, n / scale, o / scale, 8);
        mgcl::CuboidBS dst_exp(m / scale, n / scale, o / scale, 8);

        for (int i = dst_exp.getGhostsM(), i2 = src.getGhostsM(); i < dst_exp.getM() + dst_exp.getGhostsM(); i++, i2 += scale)
            for (int j = dst_exp.getGhostsN(), j2 = src.getGhostsN(); j < dst_exp.getN() + dst_exp.getGhostsN(); j++, j2 += scale)
                for (int k = dst_exp.getGhostsO(), k2 = src.getGhostsO(); k < dst_exp.getO() + dst_exp.getGhostsO(); k++, k2 += scale)
                {
                    dst_exp[0][i][j][k] = src[i2][j2][k2];
                    dst_exp[1][i][j][k] = src[i2][j2][k2 + 1];
                    dst_exp[2][i][j][k] = src[i2][j2 + 1][k2];
                    dst_exp[3][i][j][k] = src[i2][j2 + 1][k2 + 1];
                    dst_exp[4][i][j][k] = src[i2 + 1][j2][k2];
                    dst_exp[5][i][j][k] = src[i2 + 1][j2][k2 + 1];
                    dst_exp[6][i][j][k] = src[i2 + 1][j2 + 1][k2];
                    dst_exp[7][i][j][k] = src[i2 + 1][j2 + 1][k2 + 1];
                }

        mgcl_test::copyCuboidToCuboidBS(src, dst_act, scale, scale, scale);

        REQUIRE(dst_act.isEqual(dst_exp));
    }

    SECTION("scaling 1")
    {
        int scale = 1;
        mgcl::CuboidBS dst_act(m / scale, n / scale, o / scale, 8);
        mgcl::CuboidBS dst_exp(m / scale, n / scale, o / scale, 8);

        for (int i = dst_exp.getGhostsM(), i2 = src.getGhostsM(); i < dst_exp.getM() + dst_exp.getGhostsM(); i++, i2 += scale)
            for (int j = dst_exp.getGhostsN(), j2 = src.getGhostsN(); j < dst_exp.getN() + dst_exp.getGhostsN(); j++, j2 += scale)
                for (int k = dst_exp.getGhostsO(), k2 = src.getGhostsO(); k < dst_exp.getO() + dst_exp.getGhostsO(); k++, k2 += scale)
                {
                    dst_exp[0][i][j][k] = src[i2][j2][k2];
                }

        mgcl_test::copyCuboidToCuboidBS(src, dst_act, scale, scale, scale);

        REQUIRE(dst_act.isEqual(dst_exp));
    }

    SECTION("different scalings")
    {
        src.fill1dIndex(true);

        int scalem = 2;
        int scalen = 1;
        int scaleo = 4;
        mgcl::CuboidBS dst_act(m / scalem, n / scalen, o / scaleo, scalem * scalen * scaleo);
        mgcl::CuboidBS dst_exp(m / scalem, n / scalen, o / scaleo, scalem * scalen * scaleo);

        dst_exp[0][0][0][0] = 0;
        dst_exp[1][0][0][0] = 1;
        dst_exp[2][0][0][0] = 2;
        dst_exp[3][0][0][0] = 3;
        dst_exp[4][0][0][0] = 16;
        dst_exp[5][0][0][0] = 17;
        dst_exp[6][0][0][0] = 18;
        dst_exp[7][0][0][0] = 19;

        dst_exp[0][0][1][0] = 4;
        dst_exp[1][0][1][0] = 5;
        dst_exp[2][0][1][0] = 6;
        dst_exp[3][0][1][0] = 7;
        dst_exp[4][0][1][0] = 20;
        dst_exp[5][0][1][0] = 21;
        dst_exp[6][0][1][0] = 22;
        dst_exp[7][0][1][0] = 23;

        dst_exp[0][0][2][0] = 8;
        dst_exp[1][0][2][0] = 9;
        dst_exp[2][0][2][0] = 10;
        dst_exp[3][0][2][0] = 11;
        dst_exp[4][0][2][0] = 24;
        dst_exp[5][0][2][0] = 25;
        dst_exp[6][0][2][0] = 26;
        dst_exp[7][0][2][0] = 27;

        dst_exp[0][0][3][0] = 12;
        dst_exp[1][0][3][0] = 13;
        dst_exp[2][0][3][0] = 14;
        dst_exp[3][0][3][0] = 15;
        dst_exp[4][0][3][0] = 28;
        dst_exp[5][0][3][0] = 29;
        dst_exp[6][0][3][0] = 30;
        dst_exp[7][0][3][0] = 31;

        dst_exp[0][1][0][0] = 32;
        dst_exp[1][1][0][0] = 33;
        dst_exp[2][1][0][0] = 34;
        dst_exp[3][1][0][0] = 35;
        dst_exp[4][1][0][0] = 48;
        dst_exp[5][1][0][0] = 49;
        dst_exp[6][1][0][0] = 50;
        dst_exp[7][1][0][0] = 51;

        dst_exp[0][1][1][0] = 36;
        dst_exp[1][1][1][0] = 37;
        dst_exp[2][1][1][0] = 38;
        dst_exp[3][1][1][0] = 39;
        dst_exp[4][1][1][0] = 52;
        dst_exp[5][1][1][0] = 53;
        dst_exp[6][1][1][0] = 54;
        dst_exp[7][1][1][0] = 55;

        dst_exp[0][1][2][0] = 40;
        dst_exp[1][1][2][0] = 41;
        dst_exp[2][1][2][0] = 42;
        dst_exp[3][1][2][0] = 43;
        dst_exp[4][1][2][0] = 56;
        dst_exp[5][1][2][0] = 57;
        dst_exp[6][1][2][0] = 58;
        dst_exp[7][1][2][0] = 59;

        dst_exp[0][1][3][0] = 44;
        dst_exp[1][1][3][0] = 45;
        dst_exp[2][1][3][0] = 46;
        dst_exp[3][1][3][0] = 47;
        dst_exp[4][1][3][0] = 60;
        dst_exp[5][1][3][0] = 61;
        dst_exp[6][1][3][0] = 62;
        dst_exp[7][1][3][0] = 63;

        mgcl_test::copyCuboidToCuboidBS(src, dst_act, scalem, scalen, scaleo);

        REQUIRE(dst_act.isEqual(dst_exp));
    }
}

// We know that these tests are ok for copyCuboidToCuboidBS, so we just run them and go backwards again, which should
// then yield the original values.
TEST_CASE("mgcl_test::copyCuboidBSToCuboid")
{
    int m = 4;
    int n = 4;
    int o = 4;
    int blocksize = 8;

    mgcl::Cuboid src(m, n, o);
    mgcl::Cuboid exp(m, n, o);
    src.fillRandom();

    int scalem = GENERATE(1, 2);
    int scalen = GENERATE(1, 2, 4);
    int scaleo = GENERATE(2, 4);

    mgcl::CuboidBS dst_act(m / scalem, n / scalen, o / scaleo, scalem * scalen * scaleo);

    mgcl_test::copyCuboidToCuboidBS(src, dst_act, scalem, scalen, scaleo);
    mgcl_test::copyCuboidBSToCuboid(dst_act, exp, scalem, scalen, scaleo);

    REQUIRE(exp.isEqual(src));
}
