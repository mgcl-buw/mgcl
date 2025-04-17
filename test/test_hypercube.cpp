#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <cstdio>
#include <fstream>

#include "../src/mgcl/hypercube.hpp"

TEST_CASE("Hypercube4d")
{

    mgcl::Hypercube4d c(1, 2, 3, 4);

    SECTION("dimensions")
    {
        REQUIRE(c.getDim1() == 1);
        REQUIRE(c.getDim2() == 2);
        REQUIRE(c.getDim3() == 3);
        REQUIRE(c.getDim4() == 4);
    }

    SECTION("values")
    {
        c[0][1][2][3] = 3.0;
        REQUIRE(c[0][0][0][0] == 0.0);
        REQUIRE(c[0][1][2][3] == 3.0);
        REQUIRE(c[0][0][0][0 + 1 * 3 * 4 + 2 * 4 + 3] == 3.0);
    }

    SECTION("default value")
    {
        for (auto v : c.field1d())
            REQUIRE(v == 0.0);

        mgcl::Hypercube4d c2(2, 2, 2, 2, 5.0);
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
        mgcl::Hypercube4d c2(c.getDim1(), c.getDim2(), c.getDim3(), c.getDim4());

        c[0][0][0][0] = 3;
        c2[0][0][0][0] = 5;

        c[0][0][0][2] = 7;
        c2[0][0][0][2] = 8;

        REQUIRE(!c.isEqual(c2));
        REQUIRE(!c2.isEqual(c));
    }

    SECTION("isEqual true")
    {
        mgcl::Hypercube4d c2(c.getDim1(), c.getDim2(), c.getDim3(), c.getDim4());

        c[0][0][0][0] = 3;
        c2[0][0][0][0] = 3;

        c[0][0][0][2] = 7;
        c2[0][0][0][2] = 7;

        REQUIRE(c.isEqual(c2));
        REQUIRE(c2.isEqual(c));
    }

    SECTION("isEqual ghosts")
    {
        int ghosts_dim1_c = GENERATE(0, 1, 2);
        int ghosts_dim2_c = GENERATE(0, 2);
        int ghosts_dim3_c = GENERATE(2);
        int ghosts_dim4_c = GENERATE(0, 1);
        int ghosts_dim1_c2 = GENERATE(0, 1, 2);
        int ghosts_dim2_c2 = GENERATE(2);
        int ghosts_dim3_c2 = GENERATE(1, 2);
        int ghosts_dim4_c2 = GENERATE(0, 2);

        mgcl::Hypercube4d c2(8, 8, 8, 8, ghosts_dim1_c, ghosts_dim2_c, ghosts_dim3_c, ghosts_dim4_c);
        mgcl::Hypercube4d c3(8, 8, 8, 8, ghosts_dim1_c2, ghosts_dim2_c2, ghosts_dim3_c2, ghosts_dim4_c2);
        mgcl::Hypercube4d c4(16, 16, 16, 16);

        REQUIRE(c2.isEqual(c3));
        REQUIRE(c3.isEqual(c2));
        REQUIRE_THROWS_AS(c4.isEqual(c2), std::invalid_argument);
    }

    SECTION("ghosts > 0")
    {
        int dim1 = 1;
        int dim2 = 2;
        int dim3 = 3;
        int dim4 = 4;
        int ghosts_dim1 = GENERATE(0, 2);
        int ghosts_dim2 = GENERATE(0, 1);
        int ghosts_dim3 = GENERATE(1, 2);
        int ghosts_dim4 = GENERATE(0, 1, 2);
        mgcl::Hypercube4d c2(dim1,
                             dim2,
                             dim3,
                             dim4,
                             ghosts_dim1,
                             ghosts_dim2,
                             ghosts_dim3,
                             ghosts_dim4);

        REQUIRE(c2.getDim1() == dim1);
        REQUIRE(c2.getDim2() == dim2);
        REQUIRE(c2.getDim3() == dim3);
        REQUIRE(c2.getDim4() == dim4);
        REQUIRE(c2.getGhostsDim1() == ghosts_dim1);
        REQUIRE(c2.getGhostsDim2() == ghosts_dim2);
        REQUIRE(c2.getGhostsDim3() == ghosts_dim3);
        REQUIRE(c2.getGhostsDim4() == ghosts_dim4);
        REQUIRE(c2.getDim1gh() == dim1 + 2 * ghosts_dim1);
        REQUIRE(c2.getDim2gh() == dim2 + 2 * ghosts_dim2);
        REQUIRE(c2.getDim3gh() == dim3 + 2 * ghosts_dim3);
        REQUIRE(c2.getDim4gh() == dim4 + 2 * ghosts_dim4);
    }

    SECTION("fill")
    {
        mgcl::Hypercube4d c2(4, 8, 16, 32, 1, 2, 3, 0);

        for (auto v : c2.field1d())
            CHECK(v == 0.0);

        // fill real cells only first
        c2.fill(5.0, true);
        for (int i = c2.getGhostsDim1(); i < c2.getDim1() + c2.getGhostsDim1(); i++)
            for (int j = c2.getGhostsDim2(); j < c2.getDim2() + c2.getGhostsDim2(); j++)
                for (int k = c2.getGhostsDim3(); k < c2.getDim3() + c2.getGhostsDim3(); k++)
                    for (int l = c2.getGhostsDim4(); l < c2.getDim4() + c2.getGhostsDim4(); l++)
                    {
                        CHECK(c2[i][j][k][l] == 5.0);
                    }

        for (int i = 0; i < c2.getGhostsDim1(); i++)
            for (int j = 0; j < c2.getGhostsDim2(); j++)
                for (int k = 0; k < c2.getGhostsDim3(); k++)
                    for (int l = 0; l < c2.getGhostsDim4(); l++)
                    {
                        CHECK(c2[i][j][k][l] == 0.0);
                    }

        for (int i = c2.getDim1() + c2.getGhostsDim1(); i < c2.getDim1gh(); i++)
            for (int j = c2.getDim2() + c2.getGhostsDim2(); j < c2.getDim2gh(); j++)
                for (int k = c2.getDim3() + c2.getGhostsDim3(); k < c2.getDim3gh(); k++)
                    for (int l = c2.getDim4() + c2.getGhostsDim4(); l < c2.getDim4gh(); l++)
                    {
                        CHECK(c2[i][j][k][l] == 0.0);
                    }

        // fill ghosted cells, too
        c2.fill(7.0);
        for (auto v : c2.field1d())
            CHECK(v == 7.0);
    }

    SECTION("dumpToFile")
    {
        mgcl::Hypercube4d c2(1, 2, 3, 4, 0, 1, 2, 5.0);
        std::string path = "./test.txt";
        c2.dumpToFile(path);
        std::ifstream f(path.c_str());
        REQUIRE(f.good());

        auto lineCount = std::count(std::istreambuf_iterator<char>(f),
                                    std::istreambuf_iterator<char>(), '\n');
        CHECK(lineCount == c2.getDim1gh() * c2.getDim2gh() * c2.getDim3gh() * c2.getDim4gh());

        f.close();
        CHECK(remove(path.c_str()) == 0);
    }
}

TEST_CASE("Hypercube6d")
{
    SECTION("move ctor")
    {
        int n = GENERATE(1, 2, 3);
        int m = GENERATE(1, 2, 3);
        int o = GENERATE(1, 2, 3);
        int d4to6 = GENERATE(3, 5, 7);

        mgcl::Hypercube6d h(m, n, o, d4to6, d4to6, d4to6);
        h.fillRandom();

        // copy manually for checking results
        mgcl::Hypercube6d h_check(m, n, o, d4to6, d4to6, d4to6);
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < o; k++)
                    for (int ii = 0; ii < d4to6; ii++)
                        for (int jj = 0; jj < d4to6; jj++)
                            for (int kk = 0; kk < d4to6; kk++)
                            {
                                h_check[i][j][k][ii][jj][kk] = h[i][j][k][ii][jj][kk];
                            }

        // check move ctor
        auto h2(std::move(h));

        CHECK(h.getDim1() == 0);
        CHECK(h.getDim2() == 0);
        CHECK(h.getDim3() == 0);
        CHECK(h.getDim4() == 0);
        CHECK(h.getDim5() == 0);
        CHECK(h.getDim6() == 0);
        CHECK(h.getData() == nullptr);
        CHECK(h2.isEqual(h_check));

        // check move assignment
        auto h3 = std::move(h2);

        CHECK(h2.getDim1() == 0);
        CHECK(h2.getDim2() == 0);
        CHECK(h2.getDim3() == 0);
        CHECK(h2.getDim4() == 0);
        CHECK(h2.getDim5() == 0);
        CHECK(h2.getDim6() == 0);
        CHECK(h2.getData() == nullptr);
        CHECK(h3.isEqual(h_check));
    }
}

// Test if Hypercube6d gets filled with 1d index.
TEST_CASE("Hypercube6d::fill1dindex")
{
    int dim1 = 1;
    int dim2 = 2;
    int dim3 = 3;
    int dim4 = 4;
    int dim5 = 5;
    int dim6 = 6;
    int ghdim1 = 0;
    int ghdim2 = 1;
    int ghdim3 = 2;
    int ghdim4 = 0;
    int ghdim5 = 1;
    int ghdim6 = 2;

    mgcl::Hypercube6d c_real(dim1, dim2, dim3, dim4, dim5, dim6, ghdim1, ghdim2, ghdim3, ghdim4, ghdim5, ghdim6);
    c_real.fill1dIndex(true);
    mgcl::Hypercube6d c_gh(dim1, dim2, dim3, dim4, dim5, dim6, ghdim1, ghdim2, ghdim3, ghdim4, ghdim5, ghdim6);
    c_gh.fill1dIndex(false);

    int cnt = 0;
    for (int d1 = 0; d1 < c_gh.getDim1gh(); d1++)
        for (int d2 = 0; d2 < c_gh.getDim2gh(); d2++)
            for (int d3 = 0; d3 < c_gh.getDim3gh(); d3++)
                for (int d4 = 0; d4 < c_gh.getDim4gh(); d4++)
                    for (int d5 = 0; d5 < c_gh.getDim5gh(); d5++)
                        for (int d6 = 0; d6 < c_gh.getDim6gh(); d6++)
                        {
                            if (d1 > ghdim1 && d1 < dim1 + ghdim1 &&
                                d2 > ghdim2 && d2 < dim2 + ghdim2 &&
                                d3 > ghdim3 && d3 < dim3 + ghdim3 &&
                                d4 > ghdim4 && d4 < dim4 + ghdim4 &&
                                d5 > ghdim5 && d5 < dim5 + ghdim5 &&
                                d6 > ghdim6 && d6 < dim6 + ghdim6)
                                REQUIRE(c_real[d1][d2][d3][d4][d5][d6] == cnt);

                            REQUIRE(c_gh[d1][d2][d3][d4][d5][d6] == cnt);
                            cnt++;
                        }
}

TEST_CASE("Hypercube8d")
{
    SECTION("move ctor")
    {
        int n = GENERATE(1, 2, 3);
        int m = GENERATE(1, 3);
        int o = GENERATE(2, 3);
        int d4to8 = GENERATE(1, 3);

        mgcl::Hypercube8d h(m, n, o, d4to8, d4to8, d4to8, d4to8, d4to8);
        h.fillRandom();

        // copy manually for checking results
        mgcl::Hypercube8d h_check(m, n, o, d4to8, d4to8, d4to8, d4to8, d4to8);
        for (int dim1 = 0; dim1 < m; dim1++)
            for (int dim2 = 0; dim2 < n; dim2++)
                for (int dim3 = 0; dim3 < o; dim3++)
                    for (int dim4 = 0; dim4 < d4to8; dim4++)
                        for (int dim5 = 0; dim5 < d4to8; dim5++)
                            for (int dim6 = 0; dim6 < d4to8; dim6++)
                                for (int dim7 = 0; dim7 < d4to8; dim7++)
                                    for (int dim8 = 0; dim8 < d4to8; dim8++)
                                    {
                                        h_check[dim1][dim2][dim3][dim4][dim5][dim6][dim7][dim8] = h[dim1][dim2][dim3][dim4][dim5][dim6][dim7][dim8];
                                    }

        // check size
        CHECK(h.getSize() == m * n * o * d4to8 * d4to8 * d4to8 * d4to8 * d4to8);

        // check move ctor
        auto h2(std::move(h));

        CHECK(h.getDim1() == 0);
        CHECK(h.getDim2() == 0);
        CHECK(h.getDim3() == 0);
        CHECK(h.getDim4() == 0);
        CHECK(h.getDim5() == 0);
        CHECK(h.getDim6() == 0);
        CHECK(h.getDim7() == 0);
        CHECK(h.getDim8() == 0);
        CHECK(h.getData() == nullptr);
        CHECK(h2.isEqual(h_check));

        // check move assignment
        auto h3 = std::move(h2);

        CHECK(h2.getDim1() == 0);
        CHECK(h2.getDim2() == 0);
        CHECK(h2.getDim3() == 0);
        CHECK(h2.getDim4() == 0);
        CHECK(h2.getDim5() == 0);
        CHECK(h2.getDim6() == 0);
        CHECK(h2.getDim7() == 0);
        CHECK(h2.getDim8() == 0);
        CHECK(h2.getData() == nullptr);
        CHECK(h3.isEqual(h_check));
    }
}

// Test if Hypercube8d gets filled with 1d index.
TEST_CASE("Hypercube8d::fill1dindex")
{
    int dim1 = 1;
    int dim2 = 2;
    int dim3 = 3;
    int dim4 = 4;
    int dim5 = 5;
    int dim6 = 6;
    int dim7 = 7;
    int dim8 = 8;
    int ghdim1 = 0;
    int ghdim2 = 1;
    int ghdim3 = 2;
    int ghdim4 = 0;
    int ghdim5 = 1;
    int ghdim6 = 2;
    int ghdim7 = 0;
    int ghdim8 = 2;

    mgcl::Hypercube8d c_real(dim1, dim2, dim3, dim4, dim5, dim6, dim7, dim8, ghdim1, ghdim2, ghdim3, ghdim4, ghdim5, ghdim6, ghdim7, ghdim8);
    c_real.fill1dIndex(true);
    mgcl::Hypercube8d c_gh(dim1, dim2, dim3, dim4, dim5, dim6, dim7, dim8, ghdim1, ghdim2, ghdim3, ghdim4, ghdim5, ghdim6, ghdim7, ghdim8);
    c_gh.fill1dIndex(false);

    int cnt = 0;
    for (int d1 = 0; d1 < c_gh.getDim1gh(); d1++)
        for (int d2 = 0; d2 < c_gh.getDim2gh(); d2++)
            for (int d3 = 0; d3 < c_gh.getDim3gh(); d3++)
                for (int d4 = 0; d4 < c_gh.getDim4gh(); d4++)
                    for (int d5 = 0; d5 < c_gh.getDim5gh(); d5++)
                        for (int d6 = 0; d6 < c_gh.getDim6gh(); d6++)
                            for (int d7 = 0; d7 < c_gh.getDim7gh(); d7++)
                                for (int d8 = 0; d8 < c_gh.getDim8gh(); d8++)
                                {
                                    if (d1 > ghdim1 && d1 < dim1 + ghdim1 &&
                                        d2 > ghdim2 && d2 < dim2 + ghdim2 &&
                                        d3 > ghdim3 && d3 < dim3 + ghdim3 &&
                                        d4 > ghdim4 && d4 < dim4 + ghdim4 &&
                                        d5 > ghdim5 && d5 < dim5 + ghdim5 &&
                                        d6 > ghdim6 && d6 < dim6 + ghdim6 &&
                                        d7 > ghdim7 && d7 < dim7 + ghdim7 &&
                                        d8 > ghdim8 && d8 < dim8 + ghdim8)
                                        REQUIRE(c_real[d1][d2][d3][d4][d5][d6][d7][d8] == cnt);

                                    REQUIRE(c_gh[d1][d2][d3][d4][d5][d6][d7][d8] == cnt);
                                    cnt++;
                                }
}

TEST_CASE("Hypercube5d")
{
    SECTION("move ctor")
    {
        int n = GENERATE(1, 2, 3);
        int m = GENERATE(1, 3);
        int o = GENERATE(2, 3);
        int d4to5 = GENERATE(1, 3);

        mgcl::Hypercube5d h(m, n, o, d4to5, d4to5);
        h.fillRandom();

        // copy manually for checking results
        mgcl::Hypercube5d h_check(m, n, o, d4to5, d4to5);
        for (int dim1 = 0; dim1 < m; dim1++)
            for (int dim2 = 0; dim2 < n; dim2++)
                for (int dim3 = 0; dim3 < o; dim3++)
                    for (int dim4 = 0; dim4 < d4to5; dim4++)
                        for (int dim5 = 0; dim5 < d4to5; dim5++)
                        {
                            h_check[dim1][dim2][dim3][dim4][dim5] = h[dim1][dim2][dim3][dim4][dim5];
                        }

        // check size
        CHECK(h.getSize() == m * n * o * d4to5 * d4to5);

        // check move ctor
        auto h2(std::move(h));

        CHECK(h.getDim1() == 0);
        CHECK(h.getDim2() == 0);
        CHECK(h.getDim3() == 0);
        CHECK(h.getDim4() == 0);
        CHECK(h.getDim5() == 0);
        CHECK(h.getData() == nullptr);
        CHECK(h2.isEqual(h_check));

        // check move assignment
        auto h3 = std::move(h2);

        CHECK(h2.getDim1() == 0);
        CHECK(h2.getDim2() == 0);
        CHECK(h2.getDim3() == 0);
        CHECK(h2.getDim4() == 0);
        CHECK(h2.getDim5() == 0);
        CHECK(h2.getData() == nullptr);
        CHECK(h3.isEqual(h_check));
    }
}

// Test if Hypercube5d gets filled with 1d index.
TEST_CASE("Hypercube5d::fill1dindex")
{
    int dim1 = 1;
    int dim2 = 2;
    int dim3 = 3;
    int dim4 = 4;
    int dim5 = 5;
    int ghdim1 = 0;
    int ghdim2 = 1;
    int ghdim3 = 2;
    int ghdim4 = 0;
    int ghdim5 = 1;

    mgcl::Hypercube5d c_real(dim1, dim2, dim3, dim4, dim5, ghdim1, ghdim2, ghdim3, ghdim4, ghdim5);
    c_real.fill1dIndex(true);
    mgcl::Hypercube5d c_gh(dim1, dim2, dim3, dim4, dim5, ghdim1, ghdim2, ghdim3, ghdim4, ghdim5);
    c_gh.fill1dIndex(false);

    int cnt = 0;
    for (int d1 = 0; d1 < c_gh.getDim1gh(); d1++)
        for (int d2 = 0; d2 < c_gh.getDim2gh(); d2++)
            for (int d3 = 0; d3 < c_gh.getDim3gh(); d3++)
                for (int d4 = 0; d4 < c_gh.getDim4gh(); d4++)
                    for (int d5 = 0; d5 < c_gh.getDim5gh(); d5++)
                    {
                        if (d1 > ghdim1 && d1 < dim1 + ghdim1 &&
                            d2 > ghdim2 && d2 < dim2 + ghdim2 &&
                            d3 > ghdim3 && d3 < dim3 + ghdim3 &&
                            d4 > ghdim4 && d4 < dim4 + ghdim4 &&
                            d5 > ghdim5 && d5 < dim5 + ghdim5)
                            REQUIRE(c_real[d1][d2][d3][d4][d5] == cnt);

                        REQUIRE(c_gh[d1][d2][d3][d4][d5] == cnt);
                        cnt++;
                    }
}
