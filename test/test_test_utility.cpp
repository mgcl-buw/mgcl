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
                    dst_exp[i][j][k][0] = src[i2][j2][k2];
                    dst_exp[i][j][k][1] = src[i2][j2][k2 + 1];
                    dst_exp[i][j][k][2] = src[i2][j2 + 1][k2];
                    dst_exp[i][j][k][3] = src[i2][j2 + 1][k2 + 1];
                    dst_exp[i][j][k][4] = src[i2 + 1][j2][k2];
                    dst_exp[i][j][k][5] = src[i2 + 1][j2][k2 + 1];
                    dst_exp[i][j][k][6] = src[i2 + 1][j2 + 1][k2];
                    dst_exp[i][j][k][7] = src[i2 + 1][j2 + 1][k2 + 1];
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
                    dst_exp[i][j][k][0] = src[i2][j2][k2];
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
        dst_exp[0][0][0][1] = 1;
        dst_exp[0][0][0][2] = 2;
        dst_exp[0][0][0][3] = 3;
        dst_exp[0][0][0][4] = 16;
        dst_exp[0][0][0][5] = 17;
        dst_exp[0][0][0][6] = 18;
        dst_exp[0][0][0][7] = 19;

        dst_exp[0][1][0][0] = 4;
        dst_exp[0][1][0][1] = 5;
        dst_exp[0][1][0][2] = 6;
        dst_exp[0][1][0][3] = 7;
        dst_exp[0][1][0][4] = 20;
        dst_exp[0][1][0][5] = 21;
        dst_exp[0][1][0][6] = 22;
        dst_exp[0][1][0][7] = 23;

        dst_exp[0][2][0][0] = 8;
        dst_exp[0][2][0][1] = 9;
        dst_exp[0][2][0][2] = 10;
        dst_exp[0][2][0][3] = 11;
        dst_exp[0][2][0][4] = 24;
        dst_exp[0][2][0][5] = 25;
        dst_exp[0][2][0][6] = 26;
        dst_exp[0][2][0][7] = 27;

        dst_exp[0][3][0][0] = 12;
        dst_exp[0][3][0][1] = 13;
        dst_exp[0][3][0][2] = 14;
        dst_exp[0][3][0][3] = 15;
        dst_exp[0][3][0][4] = 28;
        dst_exp[0][3][0][5] = 29;
        dst_exp[0][3][0][6] = 30;
        dst_exp[0][3][0][7] = 31;

        dst_exp[1][0][0][0] = 32;
        dst_exp[1][0][0][1] = 33;
        dst_exp[1][0][0][2] = 34;
        dst_exp[1][0][0][3] = 35;
        dst_exp[1][0][0][4] = 48;
        dst_exp[1][0][0][5] = 49;
        dst_exp[1][0][0][6] = 50;
        dst_exp[1][0][0][7] = 51;

        dst_exp[1][1][0][0] = 36;
        dst_exp[1][1][0][1] = 37;
        dst_exp[1][1][0][2] = 38;
        dst_exp[1][1][0][3] = 39;
        dst_exp[1][1][0][4] = 52;
        dst_exp[1][1][0][5] = 53;
        dst_exp[1][1][0][6] = 54;
        dst_exp[1][1][0][7] = 55;

        dst_exp[1][2][0][0] = 40;
        dst_exp[1][2][0][1] = 41;
        dst_exp[1][2][0][2] = 42;
        dst_exp[1][2][0][3] = 43;
        dst_exp[1][2][0][4] = 56;
        dst_exp[1][2][0][5] = 57;
        dst_exp[1][2][0][6] = 58;
        dst_exp[1][2][0][7] = 59;

        dst_exp[1][3][0][0] = 44;
        dst_exp[1][3][0][1] = 45;
        dst_exp[1][3][0][2] = 46;
        dst_exp[1][3][0][3] = 47;
        dst_exp[1][3][0][4] = 60;
        dst_exp[1][3][0][5] = 61;
        dst_exp[1][3][0][6] = 62;
        dst_exp[1][3][0][7] = 63;

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
