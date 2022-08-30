#include <catch2/catch_test_macros.hpp>

#include "../cuboid.hpp"
#include "test_utility.hpp"

TEST_CASE("TestUtility setup")
{
    mgcl_test::TestUtility tu;
    REQUIRE(tu.getProblem().getOpenCLHelper().isInitialized());
}

TEST_CASE("TestUtility setup reusing Problem")
{
    auto p = std::make_shared<mgcl::Problem>(4, 4, 4);
    mgcl_test::TestUtility tu(p);
    REQUIRE(tu.getProblem().getOpenCLHelper().isInitialized());
}

TEST_CASE("TestUtility createOpenCLBuffer + readOpenCLBuffer")
{
    mgcl_test::TestUtility tu;
    REQUIRE(tu.getProblem().getOpenCLHelper().isInitialized());

    mgcl::Cuboid c(4, 4, 4, 3.0);
    auto buf = tu.createOpenCLBuffer(c);
    REQUIRE(buf);

    auto c2 = tu.readOpenCLBuffer(buf, 4, 4, 4);
    for (int i = 0; i < c2.field1d().size(); i++)
        REQUIRE(c2.field1d()[i] == c.field1d()[i]);
}

TEST_CASE("TestUtility readOpenCLBuffer nullptr")
{
    mgcl_test::TestUtility tu;
    REQUIRE(tu.getProblem().getOpenCLHelper().isInitialized());
    REQUIRE_THROWS_AS(tu.readOpenCLBuffer(nullptr, 4, 4, 4), std::invalid_argument);
}
