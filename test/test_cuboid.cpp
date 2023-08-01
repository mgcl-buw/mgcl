#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <cstdio>
#include <fstream>

#include "../src/cuboid.hpp"

TEST_CASE("cuboid class")
{

    mgcl::Cuboid c(1, 2, 3);

    SECTION("move ctor")
    {
        mgcl::Cuboid c_check(1, 2, 3);
        c.fillRandom();

        for (int i = 0; i < c.getM(); i++)
            for (int j = 0; j < c.getN(); j++)
                for (int k = 0; k < c.getO(); k++)
                {
                    c_check[i][j][k] = c[i][j][k];
                }

        mgcl::Cuboid c2(std::move(c));

        REQUIRE(c.getData() == nullptr);
        REQUIRE(c.getM() == 0);
        REQUIRE(c.getN() == 0);
        REQUIRE(c.getO() == 0);

        REQUIRE(c2.isEqual(c_check));
    }

    SECTION("move assignment ctor")
    {
        mgcl::Cuboid c_check(1, 2, 3);
        c.fillRandom();

        for (int i = 0; i < c.getM(); i++)
            for (int j = 0; j < c.getN(); j++)
                for (int k = 0; k < c.getO(); k++)
                {
                    c_check[i][j][k] = c[i][j][k];
                }

        mgcl::Cuboid c2 = std::move(c);

        REQUIRE(c.getData() == nullptr);
        REQUIRE(c.getM() == 0);
        REQUIRE(c.getN() == 0);
        REQUIRE(c.getO() == 0);

        REQUIRE(c2.isEqual(c_check));
    }

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

    SECTION("fill")
    {
        mgcl::Cuboid c2(4, 8, 16, 1, 2, 3, 0);

        for (auto v : c2.field1d())
            CHECK(v == 0.0);

        // fill real cells only first
        c2.fill(5.0, true);
        for (int i = c2.getGhostsM(); i < c2.getM() + c2.getGhostsM(); i++)
            for (int j = c2.getGhostsN(); j < c2.getN() + c2.getGhostsN(); j++)
                for (int k = c2.getGhostsO(); k < c2.getO() + c2.getGhostsO(); k++)
                {
                    CHECK(c2[i][j][k] == 5.0);
                }

        for (int i = 0; i < c2.getGhostsM(); i++)
            for (int j = 0; j < c2.getGhostsN(); j++)
                for (int k = 0; k < c2.getGhostsO(); k++)
                {
                    CHECK(c2[i][j][k] == 0.0);
                }

        for (int i = c2.getM() + c2.getGhostsM(); i < c2.getMgh(); i++)
            for (int j = c2.getN() + c2.getGhostsN(); j < c2.getNgh(); j++)
                for (int k = c2.getO() + c2.getGhostsO(); k < c2.getOgh(); k++)
                {
                    CHECK(c2[i][j][k] == 0.0);
                }

        // fill ghosted cells, too
        c2.fill(7.0);
        for (auto v : c2.field1d())
            CHECK(v == 7.0);
    }

    SECTION("dumpToFile")
    {
        mgcl::Cuboid c2(1, 2, 3, 0, 1, 2, 5.0);
        std::string path = "./test.txt";
        c2.dumpToFile(path);
        std::ifstream f(path.c_str());
        REQUIRE(f.good());

        auto lineCount = std::count(std::istreambuf_iterator<char>(f),
                                    std::istreambuf_iterator<char>(), '\n');
        CHECK(lineCount == c2.getMgh() * c2.getNgh() * c2.getOgh());

        f.close();
        CHECK(remove(path.c_str()) == 0);
    }
}

TEST_CASE("Cuboid::slice")
{
    int m = 4;
    int n = 4;
    int o = 4;

    mgcl::Cuboid cb(m, n, o, 1, 1, 1);
    cb.fillRandom();

    auto cs = cb.slice(0, 1, 0, 2, 2, 3, 3, 2);

    REQUIRE(cs->getM() == 2);
    REQUIRE(cs->getN() == 3);
    REQUIRE(cs->getO() == 2);
    REQUIRE(cs->getGhostsM() == 3);
    REQUIRE(cs->getGhostsN() == 2);
    REQUIRE(cs->getGhostsO() == cb.getGhostsO());

    for (int i = 0; i < cs->getM(); i++)
        for (int j = 0; j < cs->getN(); j++)
            for (int k = 0; k < cs->getO(); k++)
            {
                REQUIRE(cs->getData()[i + cs->getGhostsM()][j + cs->getGhostsN()][k + cs->getGhostsO()] ==
                        cb[i + cb.getGhostsM()][j + cb.getGhostsN()][k + cb.getGhostsO() + 2]);
            }
}