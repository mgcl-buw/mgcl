#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

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

    SECTION("default value")
    {
        for (auto v : c.field1d())
            REQUIRE(v == 0.0);

        mgcl::Cuboid c2(2, 2, 2, 5);
        for (auto v : c2.field1d())
            REQUIRE(v == 5.0);
    }

    SECTION("fillRandom")
    {
        c.fillRandom(3.5, 5.5);
        for (auto v : c.field1d())
        {
            REQUIRE(v >= 3.5);
            REQUIRE(v <= 5.5);
        }
    }

    SECTION("isEqual false")
    {
        mgcl::Cuboid c2(c.getM(), c.getN(), c.getO());

        c[0][0][0] = 3;
        c2[0][0][0] = 5;

        c[0][0][2] = 7;
        c2[0][0][2] = 8;

        REQUIRE(!c.isEqual(c2));
        REQUIRE(!c2.isEqual(c));
    }

    SECTION("isEqual true")
    {
        mgcl::Cuboid c2(c.getM(), c.getN(), c.getO());

        c[0][0][0] = 3;
        c2[0][0][0] = 3;

        c[0][0][2] = 7;
        c2[0][0][2] = 7;

        REQUIRE(c.isEqual(c2));
        REQUIRE(c2.isEqual(c));
    }

    SECTION("isEqual ghosts")
    {
        int ghosts_m_c = GENERATE(0, 1, 2);
        int ghosts_n_c = GENERATE(0, 1, 2);
        int ghosts_o_c = GENERATE(0, 1, 2);
        int ghosts_m_c2 = GENERATE(0, 1, 2);
        int ghosts_n_c2 = GENERATE(0, 1, 2);
        int ghosts_o_c2 = GENERATE(0, 1, 2);

        mgcl::Cuboid c2(8, 8, 8, ghosts_m_c, ghosts_n_c, ghosts_o_c);
        mgcl::Cuboid c3(8, 8, 8, ghosts_m_c2, ghosts_n_c2, ghosts_o_c2);
        mgcl::Cuboid c4(16, 16, 16);

        REQUIRE(c2.isEqual(c3));
        REQUIRE(c3.isEqual(c2));
        REQUIRE_THROWS_AS(c4.isEqual(c2), std::invalid_argument);
    }

    SECTION("ghosts > 0")
    {
        int m = 3;
        int n = 4;
        int o = 5;
        int ghosts_m = GENERATE(0, 1, 2);
        int ghosts_n = GENERATE(0, 1, 2);
        int ghosts_o = GENERATE(0, 1, 2);
        mgcl::Cuboid c2(m, n, o, ghosts_m, ghosts_n, ghosts_o);

        REQUIRE(c2.getM() == m);
        REQUIRE(c2.getN() == n);
        REQUIRE(c2.getO() == o);
        REQUIRE(c2.getGhostsM() == ghosts_m);
        REQUIRE(c2.getGhostsN() == ghosts_n);
        REQUIRE(c2.getGhostsO() == ghosts_o);
        REQUIRE(c2.getMgh() == m + 2 * ghosts_m);
        REQUIRE(c2.getNgh() == n + 2 * ghosts_n);
        REQUIRE(c2.getOgh() == o + 2 * ghosts_o);
    }
}
