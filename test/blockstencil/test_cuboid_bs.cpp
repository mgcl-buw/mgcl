#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <cstdio>
#include <fstream>

#include "../../src/mgcl/cuboid_bs.hpp"

TEST_CASE("CuboidBS")
{
    int blocksize = 2;
    int m = 1;
    int n = 2;
    int o = 3;
    mgcl::CuboidBS c(m, n, o, blocksize);

    SECTION("to1dIndex")
    {
        REQUIRE(c.to1dIndex(0, 0, 0, 0) == 0);
        REQUIRE(c.to1dIndex(0, 0, 0, 1) == 1);
        REQUIRE(c.to1dIndex(0, 0, 1, 0) == 2);
        REQUIRE(c.to1dIndex(0, 0, 1, 1) == 3);
        REQUIRE(c.to1dIndex(0, 0, 2, 0) == 4);
        REQUIRE(c.to1dIndex(0, 0, 2, 1) == 5);
        REQUIRE(c.to1dIndex(0, 1, 0, 0) == 6);
        REQUIRE(c.to1dIndex(0, 1, 0, 1) == 7);
        REQUIRE(c.to1dIndex(0, 1, 1, 0) == 8);
        REQUIRE(c.to1dIndex(0, 1, 1, 1) == 9);
        REQUIRE(c.to1dIndex(0, 1, 2, 0) == 10);
        REQUIRE(c.to1dIndex(0, 1, 2, 1) == 11);
        REQUIRE(c.to1dIndex(1, 0, 0, 0) == 12);
    }

    SECTION("move ctor")
    {
        mgcl::CuboidBS c_check(m, n, o, blocksize);
        c.fillRandom();

        for (int i = 0; i < c.getM(); i++)
            for (int j = 0; j < c.getN(); j++)
                for (int k = 0; k < c.getO(); k++)
                    for (int b = 0; b < blocksize; b++)
                    {
                        c_check[i][j][k][b] = c[i][j][k][b];
                    }

        mgcl::CuboidBS c2(std::move(c));

        REQUIRE(c.getData() == nullptr);
        REQUIRE(c.getM() == 0);
        REQUIRE(c.getN() == 0);
        REQUIRE(c.getO() == 0);
        REQUIRE(c.getBlocksize() == 0);

        REQUIRE(c2.isEqual(c_check));
    }

    SECTION("move assignment ctor")
    {
        mgcl::CuboidBS c_check(m, n, o, blocksize);
        c.fillRandom();

        for (int i = 0; i < c.getM(); i++)
            for (int j = 0; j < c.getN(); j++)
                for (int k = 0; k < c.getO(); k++)
                    for (int b = 0; b < blocksize; b++)
                    {
                        c_check[i][j][k][b] = c[i][j][k][b];
                    }

        mgcl::CuboidBS c2 = std::move(c);

        REQUIRE(c.getData() == nullptr);
        REQUIRE(c.getM() == 0);
        REQUIRE(c.getN() == 0);
        REQUIRE(c.getO() == 0);
        REQUIRE(c.getBlocksize() == 0);

        REQUIRE(c2.isEqual(c_check));
    }

    SECTION("dimensions")
    {
        REQUIRE(c.getM() == 1);
        REQUIRE(c.getN() == 2);
        REQUIRE(c.getO() == 3);
        REQUIRE(c.getBlocksize() == blocksize);
    }

    SECTION("values")
    {
        c[0][1][1][0] = 3.0;
        c[0][1][1][1] = 5.0;
        REQUIRE(c[0][0][0][0] == 0.0);
        REQUIRE(c[0][0][0][1] == 0.0);
        REQUIRE(c[0][0][0][0] == 0.0);
        REQUIRE(c[0][1][1][0] == 3.0);
        REQUIRE(c[0][1][1][1] == 5.0);
        REQUIRE(c[0][0][0][0 + 1 * blocksize * o + 1 * blocksize + 0] == 3.0);
        REQUIRE(c[0][0][0][0 + 1 * blocksize * o + 1 * blocksize + 1] == 5.0);
    }

    SECTION("default value")
    {
        for (auto v : c.field1d())
            REQUIRE(v == 0.0);

        mgcl::CuboidBS c2(2, 2, 2, blocksize, 5);
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
        mgcl::CuboidBS c2(c.getM(), c.getN(), c.getO(), blocksize);

        c[0][0][0][0] = 3;
        c2[0][0][0][0] = 5;

        c[0][0][2][1] = 7;
        c2[0][0][2][1] = 8;

        REQUIRE(!c.isEqual(c2));
        REQUIRE(!c2.isEqual(c));
    }

    SECTION("isEqual true")
    {
        mgcl::CuboidBS c2(c.getM(), c.getN(), c.getO(), blocksize);

        c[0][0][0][0] = 3;
        c2[0][0][0][0] = 3;

        c[0][0][2][1] = 7;
        c2[0][0][2][1] = 7;

        REQUIRE(c.isEqual(c2));
        REQUIRE(c2.isEqual(c));
    }

    SECTION("isEqual ghosts")
    {
        int ghosts_m_c = GENERATE(0, 1, 2);
        int ghosts_n_c = GENERATE(1, 2);
        int ghosts_o_c = 2;
        int ghosts_m_c2 = GENERATE(0, 1, 2);
        int ghosts_n_c2 = GENERATE(1, 2);
        int ghosts_o_c2 = 2;

        mgcl::CuboidBS c2(8, 8, 8, ghosts_m_c, ghosts_n_c, ghosts_o_c, blocksize);
        mgcl::CuboidBS c3(8, 8, 8, ghosts_m_c2, ghosts_n_c2, ghosts_o_c2, blocksize);
        mgcl::CuboidBS c4(16, 16, 16, blocksize);

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
        mgcl::CuboidBS c2(m, n, o, ghosts_m, ghosts_n, ghosts_o, blocksize);

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
        mgcl::CuboidBS c2(4, 8, 16, 1, 2, 3, blocksize, 0);

        for (auto v : c2.field1d())
            CHECK(v == 0.0);

        // fill real cells only first
        c2.fill(5.0, true);
        for (int i = c2.getGhostsM(); i < c2.getM() + c2.getGhostsM(); i++)
            for (int j = c2.getGhostsN(); j < c2.getN() + c2.getGhostsN(); j++)
                for (int k = c2.getGhostsO(); k < c2.getO() + c2.getGhostsO(); k++)
                    for (int b = 0; b < blocksize; b++)
                    {
                        CHECK(c2[i][j][k][b] == 5.0);
                    }

        for (int i = 0; i < c2.getGhostsM(); i++)
            for (int j = 0; j < c2.getGhostsN(); j++)
                for (int k = 0; k < c2.getGhostsO(); k++)
                    for (int b = 0; b < blocksize; b++)
                    {
                        CHECK(c2[i][j][k][b] == 0.0);
                    }

        for (int i = c2.getM() + c2.getGhostsM(); i < c2.getMgh(); i++)
            for (int j = c2.getN() + c2.getGhostsN(); j < c2.getNgh(); j++)
                for (int k = c2.getO() + c2.getGhostsO(); k < c2.getOgh(); k++)
                    for (int b = 0; b < blocksize; b++)
                    {
                        CHECK(c2[i][j][k][b] == 0.0);
                    }

        // fill ghosted cells, too
        c2.fill(7.0);
        for (auto v : c2.field1d())
            CHECK(v == 7.0);
    }

    SECTION("dumpToFile")
    {
        mgcl::CuboidBS c2(1, 2, 3, 0, 1, 2, 2, 5.0);
        std::string path = "./test.txt";
        c2.dumpToFile(path);
        std::ifstream f(path.c_str());
        REQUIRE(f.good());

        auto lineCount = std::count(std::istreambuf_iterator<char>(f),
                                    std::istreambuf_iterator<char>(), '\n');
        CHECK(lineCount == c2.getBlocksize() * c2.getMgh() * c2.getNgh() * c2.getOgh());

        f.close();
        CHECK(remove(path.c_str()) == 0);
    }
}

TEST_CASE("CuboidBS::slice")
{
    int m = 4;
    int n = 4;
    int o = 4;
    int blocksize = 2;

    mgcl::CuboidBS cb(m, n, o, 1, 1, 1, blocksize);
    cb.fillRandom();

    SECTION("throwing")
    {
        REQUIRE_THROWS(cb.slice(-1, 0, 0, 0, 0, 0));
        REQUIRE_THROWS(cb.slice(0, 0, -1, 0, 0, 0));
        REQUIRE_THROWS(cb.slice(0, 0, 0, 0, -1, 0));

        REQUIRE_THROWS(cb.slice(0, m + 1, 0, 0, 0, 0));
        REQUIRE_THROWS(cb.slice(0, 0, 0, n + 1, 0, 0));
        REQUIRE_THROWS(cb.slice(0, 0, 0, 0, 0, n + 1));
    }

    SECTION("success")
    {
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
                    for (int b = 0; b < blocksize; b++)
                    {
                        REQUIRE(cs->getData()[i + cs->getGhostsM()][j + cs->getGhostsN()][k + cs->getGhostsO()][b] ==
                                cb[i + cb.getGhostsM()][j + cb.getGhostsN()][k + cb.getGhostsO() + 2][b]);
                    }
    }
}

TEST_CASE("CuboidBS::sliceIncGhosts")
{
    int m = 4;
    int n = 4;
    int o = 4;
    int blocksize = 2;

    mgcl::CuboidBS cb(m, n, o, 1, 1, 1, blocksize);
    cb.fillRandom();

    SECTION("throwing")
    {
        REQUIRE_THROWS(cb.sliceIncGhosts(-1, 0, 0, 0, 0, 0));
        REQUIRE_THROWS(cb.sliceIncGhosts(0, 0, -1, 0, 0, 0));
        REQUIRE_THROWS(cb.sliceIncGhosts(0, 0, 0, 0, -1, 0));

        REQUIRE_THROWS(cb.sliceIncGhosts(0, m + 3, 0, 0, 0, 0));
        REQUIRE_THROWS(cb.sliceIncGhosts(0, 0, 0, n + 3, 0, 0));
        REQUIRE_THROWS(cb.sliceIncGhosts(0, 0, 0, 0, 0, n + 3));
    }

    SECTION("success")
    {
        auto cs = cb.sliceIncGhosts(0, 1, 0, 2, 2, 3);

        REQUIRE(cs->getM() == 2);
        REQUIRE(cs->getN() == 3);
        REQUIRE(cs->getO() == 2);
        REQUIRE(cs->getGhostsM() == 0);
        REQUIRE(cs->getGhostsN() == 0);
        REQUIRE(cs->getGhostsO() == 0);
        REQUIRE(cs->getBlocksize() == blocksize);

        for (int i = 0; i < cs->getM(); i++)
            for (int j = 0; j < cs->getN(); j++)
                for (int k = 0; k < cs->getO(); k++)
                    for (int b = 0; b < blocksize; b++)
                    {
                        REQUIRE(cs->getData()[i][j][k][b] == cb[i][j][k + 2][b]);
                    }
    }
}

// Test if CuboidBS gets filled with 1d index.
TEST_CASE("CuboidBS::fill1dindex")
{
    int m = 1;
    int n = 2;
    int o = 3;
    int ghm = 0;
    int ghn = 1;
    int gho = 2;
    int blocksize = 2;

    mgcl::CuboidBS c_real(m, n, o, ghm, ghn, gho, blocksize);
    c_real.fill1dIndex(true);
    mgcl::CuboidBS c_gh(m, n, o, ghm, ghn, gho, blocksize);
    c_gh.fill1dIndex(false);

    int cnt = 0;
    for (int i = 0; i < c_real.getMgh(); i++)
        for (int j = 0; j < c_real.getNgh(); j++)
            for (int k = 0; k < c_real.getOgh(); k++)
                for (int b = 0; b < blocksize; b++)
                {
                    if (i > ghm && i < m + ghm && j > ghn && j < n + ghn && k > gho && k < o + gho)
                        REQUIRE(c_real[i][j][k][b] == cnt);

                    REQUIRE(c_gh[i][j][k][b] == cnt);
                    cnt++;
                }
}
