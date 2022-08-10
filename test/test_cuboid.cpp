#include <catch2/catch_test_macros.hpp>

#include "../cuboid.hpp"

TEST_CASE("alloc + free")
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
