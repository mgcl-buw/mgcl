#include <catch2/catch_test_macros.hpp>

#include "../cuboid.hpp"

TEST_CASE("cuboid alloc + free")
{
    SECTION("cuboid_alloc")
    {
        auto c = mgcl::cuboid_alloc(3, 3, 3);
        REQUIRE(c != NULL);
        mgcl::cuboid_free(c, 3, 3, 3);
    }

    SECTION("cube_alloc")
    {
        auto c = mgcl::cube_alloc(3);
        REQUIRE(c != NULL);
        mgcl::cube_free(c, 3);
    }
}

TEST_CASE("cuboid class")
{

    mgcl::Cuboid c(1, 2, 3);

    SECTION("dimensions")
    {
        REQUIRE(c.getM() == 1);
        REQUIRE(c.getN() == 2);
        REQUIRE(c.getO() == 3);
    }

    SECTION("values")
    {
        c[0][1][1] = 3.0;
        REQUIRE(c[0][0][0] == 0.0);
        REQUIRE(c[0][1][1] == 3.0);
        REQUIRE(c[0][0][0 + 1 * 3 + 1] == 3.0);
    }

    SECTION("copy constructor")
    {
        mgcl::Cuboid c2(c);
        REQUIRE(c2.getM() == c.getM());
        REQUIRE(c2.getN() == c.getN());
        REQUIRE(c2.getO() == c.getO());
        REQUIRE(c2.getData() != c.getData());
        REQUIRE(c2[0][0][0] == c[0][0][0]);
    }
}
