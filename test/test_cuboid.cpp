#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <cstdio>
#include <fstream>

#include "../src/mgcl/cuboid.hpp"

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
                {
                    REQUIRE(cs->getData()[i + cs->getGhostsM()][j + cs->getGhostsN()][k + cs->getGhostsO()] ==
                            cb[i + cb.getGhostsM()][j + cb.getGhostsN()][k + cb.getGhostsO() + 2]);
                }
    }
}

TEST_CASE("Cuboid::sliceIncGhosts")
{
    int m = 4;
    int n = 4;
    int o = 4;

    mgcl::Cuboid cb(m, n, o, 1, 1, 1);
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

        for (int i = 0; i < cs->getM(); i++)
            for (int j = 0; j < cs->getN(); j++)
                for (int k = 0; k < cs->getO(); k++)
                {
                    REQUIRE(cs->getData()[i][j][k] == cb[i][j][k + 2]);
                }
    }
}

// Test if Cuboid gets filled with 1d index.
TEST_CASE("Cuboid::fill1dindex")
{
    int m = 1;
    int n = 2;
    int o = 3;
    int ghm = 0;
    int ghn = 1;
    int gho = 2;

    mgcl::Cuboid c_real(m, n, o, ghm, ghn, gho);
    c_real.fill1dIndex(true);
    mgcl::Cuboid c_gh(m, n, o, ghm, ghn, gho);
    c_gh.fill1dIndex(false);

    int cnt = 0;
    for (int i = 0; i < c_real.getMgh(); i++)
        for (int j = 0; j < c_real.getNgh(); j++)
            for (int k = 0; k < c_real.getOgh(); k++)
            {
                if (i > ghm && i < m + ghm && j > ghn && j < n + ghn && k > gho && k < o + gho)
                    REQUIRE(c_real[i][j][k] == cnt);

                REQUIRE(c_gh[i][j][k] == cnt);
                cnt++;
            }
}

// TEST_CASE("Cuboid::extract_ghosts")
// {
//     int m = 1;
//     int n = 2;
//     int o = 3;
//     int ghm = 0;
//     int ghn = 1;
//     int gho = 2;

//     mgcl::Cuboid c(m, n, o, ghm, ghn, gho);
//     c.fill1dIndex(false);
// }

TEST_CASE("Cuboid::extract_border_planes")
{
    int m = 3;
    int n = 4;
    int o = 5;
    int ghm = 1;
    int ghn = 2;
    int gho = 3;
    int ngh = n + 2 * ghn;
    int ogh = o + 2 * gho;

    mgcl::Cuboid c(m, n, o, ghm, ghn, gho);
    c.fill1dIndex(false);

    // plane sizes
    int yz = n * o; //* ghm;
    int xz = m * o; //* ghn;
    int xy = m * n; //* gho;

    int ressize = 2 * yz * ghm + 2 * xz * ghn + 2 * xy * gho;
    double* res = new double[ressize];

    // if (e < yz)
    // {
    int cnt = 0;
    // __kernel void extract_front_planes(__global const double* c, __global double* res, const int yz, const int xz, const int xy)
    // {
    //     int gid = get_global_id(0);
    //     int i = gid / yz;
    //     int j = gid % (yz / n) + ghn;
    //     int k = gid % xy + gho;
    //     res[gid] = c[i * yz * xz + j * xz + k];
    // }

    // res[e] = c[0][e / yz][e % yz];
    // }

    // front planes (yz)
    for (int i = ghm; i < 2 * ghm; i++)         // ghm real planes in the front
        for (int j = ghn; j < ghn + n; j++)     // all real cells in y-dir
            for (int k = gho; k < gho + o; k++) // all real cells in z-dir
            {
                res[cnt++] = c[i][j][k];

                int gid = cnt - 1; // -1 just becauce there was cnt++. Don't use it in kernel.
                int i2 = gid / yz + ghm;
                int j2 = (gid - (i2 - ghm) * yz) / o + ghn;
                int k2 = gid % o + gho;

                CAPTURE(i, j, k, i2, j2, k2, gid);
                REQUIRE(i == i2);
                REQUIRE(j == j2);
                REQUIRE(k == k2);
                REQUIRE(c[i][j][k] == c[0][0][i2 * ngh * ogh + j2 * ogh + k2]);

                // write into cnt
            }

    // back planes (yz)
    for (int i = m; i < m + ghm; i++)           // ghm real planes in the back
        for (int j = ghn; j < ghn + n; j++)     // all real cells in y-dir
            for (int k = gho; k < gho + o; k++) // all real cells in z-dir
            {
                res[cnt++] = c[i][j][k];

                int gid = cnt - ghm * yz - 1; // -1 just becauce there was cnt++. Don't use it in kernel.
                int i2 = gid / yz + m;
                int j2 = (gid - (i2 - m) * yz) / o + ghn;
                int k2 = gid % o + gho;

                CAPTURE(i, j, k, i2, j2, k2, gid);
                REQUIRE(i == i2);
                REQUIRE(j == j2);
                REQUIRE(k == k2);
                REQUIRE(c[i][j][k] == c[0][0][i2 * ngh * ogh + j2 * ogh + k2]);

                // write into cnt + ghm*yz
            }

    // top planes (xz)
    for (int j = ghn; j < 2 * ghn; j++)
        for (int i = ghm; i < ghm + m; i++)
            for (int k = gho; k < gho + o; k++)
            {
                res[cnt++] = c[i][j][k];

                int gid = cnt - 2 * ghm * yz - 1; // -1 just becauce there was cnt++. Don't use it in kernel.
                int j2 = gid / xz + ghn;
                int i2 = (gid - (j2 - ghn) * xz) / o + ghm;
                int k2 = gid % o + gho;

                CAPTURE(i, j, k, i2, j2, k2, gid, yz, xz, xy);
                REQUIRE(i == i2);
                REQUIRE(j == j2);
                REQUIRE(k == k2);
                REQUIRE(c[i][j][k] == c[0][0][i2 * ngh * ogh + j2 * ogh + k2]);

                // write into cnt + 2*ghm*yz
            }

    // bottom planes (xz)
    for (int j = n; j < n + ghn; j++)
        for (int i = ghm; i < ghm + m; i++)
            for (int k = gho; k < gho + o; k++)
            {
                res[cnt++] = c[i][j][k];

                int gid = cnt - 2 * ghm * yz - ghn * xz - 1; // -1 just becauce there was cnt++. Don't use it in kernel.
                int j2 = gid / xz + n;
                int i2 = (gid - (j2 - n) * xz) / o + ghm;
                int k2 = gid % o + gho;

                CAPTURE(i, j, k, i2, j2, k2, gid, yz, xz, xy);
                REQUIRE(i == i2);
                REQUIRE(j == j2);
                REQUIRE(k == k2);
                REQUIRE(c[i][j][k] == c[0][0][i2 * ngh * ogh + j2 * ogh + k2]);

                // write into cnt + 2*ghm*yz + ghn*xz
            }

    // left planes (xy)
    for (int k = gho; k < 2 * gho; k++)
        for (int i = ghm; i < ghm + m; i++)
            for (int j = ghn; j < ghn + n; j++)
            {
                res[cnt++] = c[i][j][k];

                int gid = cnt - 2 * ghm * yz - 2 * ghn * xz - 1; // -1 just becauce there was cnt++. Don't use it in kernel.
                int k2 = gid / xy + gho;
                int i2 = (gid - (k2 - gho) * xy) / n + ghm;
                int j2 = gid % n + ghn;

                CAPTURE(i, j, k, i2, j2, k2, gid, yz, xz, xy);
                REQUIRE(i == i2);
                REQUIRE(j == j2);
                REQUIRE(k == k2);
                REQUIRE(c[i][j][k] == c[0][0][i2 * ngh * ogh + j2 * ogh + k2]);

                // write into cnt + 2*ghm*yz + 2*ghn*xz
            }

    // right planes (xy)
    for (int k = o; k < o + gho; k++)
        for (int i = ghm; i < ghm + m; i++)
            for (int j = ghn; j < ghn + n; j++)
            {
                res[cnt++] = c[i][j][k];

                int gid = cnt - 2 * ghm * yz - 2 * ghn * xz - gho * xy - 1; // -1 just becauce there was cnt++. Don't use it in kernel.
                int k2 = gid / xy + o;
                int i2 = (gid - (k2 - o) * xy) / n + ghm;
                int j2 = gid % n + ghn;

                CAPTURE(i, j, k, i2, j2, k2, gid, yz, xz, xy);
                REQUIRE(i == i2);
                REQUIRE(j == j2);
                REQUIRE(k == k2);
                REQUIRE(c[i][j][k] == c[0][0][i2 * ngh * ogh + j2 * ogh + k2]);

                // write into cnt + 2*ghm*yz + 2*ghn*xz + gho*xy
            }

    REQUIRE(ressize == cnt);

    delete[] res;
}