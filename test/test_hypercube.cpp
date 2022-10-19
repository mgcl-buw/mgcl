#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <cstdio>
#include <fstream>

#include "../hypercube.hpp"

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
        int ghosts_dim2_c = GENERATE(0, 1, 2);
        int ghosts_dim3_c = GENERATE(0, 1, 2);
        int ghosts_dim4_c = GENERATE(0, 1, 2);
        int ghosts_dim1_c2 = GENERATE(0, 1, 2);
        int ghosts_dim2_c2 = GENERATE(0, 1, 2);
        int ghosts_dim3_c2 = GENERATE(0, 1, 2);
        int ghosts_dim4_c2 = GENERATE(0, 1, 2);

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
        int ghosts_dim1 = GENERATE(0, 1, 2);
        int ghosts_dim2 = GENERATE(0, 1, 2);
        int ghosts_dim3 = GENERATE(0, 1, 2);
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