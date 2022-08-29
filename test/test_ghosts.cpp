#include <catch2/catch_test_macros.hpp>

#include <cmath>

#include "../cuboid.hpp"
#include "../multigrid_engine.hpp"
#include "test_utility.hpp"

TEST_CASE("updating ghosts")
{
    int m = 16;
    int n = 8;
    int o = 4;
    int ghosts_m = 2;
    int ghosts_n = 1;
    int ghosts_o = 0;
    int mgh = m + 2 * ghosts_m;
    int ngh = n + 2 * ghosts_n;
    int ogh = o + 2 * ghosts_o;

    mgcl::Cuboid c1(m, n, o, ghosts_m, ghosts_n, ghosts_o);
    c1.fillRandom();

    SECTION("updateGhostsSeq")
    {
        mgcl::MultigridEngine::updateGhostsSeq(c1);
        for (int i = 0; i < ghosts_m; i++)
            for (int j = 0; j < ghosts_n; j++)
                for (int k = 0; k < ghosts_o; k++)
                {
                    REQUIRE(c1[i][j][k] == c1[i + m][j + n][k + o]);
                    REQUIRE(c1[i + ghosts_m][j + ghosts_n][k + ghosts_o] == c1[i + m + ghosts_m][j + n + ghosts_n][k + o + ghosts_o]);
                }
    }

    SECTION("updateGhosts OpenCL")
    {
        mgcl_test::TestUtility tu;
        cl_mem d_c1 = tu.createOpenCLBuffer(c1);

        mgcl::MultigridEngine::updateGhosts(tu.getProblem(), d_c1, mgh, ngh, ogh, ghosts_m, ghosts_n, ghosts_o);
        tu.finish();

        auto c2 = tu.readOpenCLBuffer(d_c1, mgh, ngh, ogh);

        double tol = 1e-7;
        for (int i = 0; i < ghosts_m; i++)
            for (int j = 0; j < ghosts_n; j++)
                for (int k = 0; k < ghosts_o; k++)
                {
                    REQUIRE(fabs(c2[i][j][k] - c2[i + m][j + n][k + o]) < tol);
                    REQUIRE(fabs(c2[i + ghosts_m][j + ghosts_n][k + ghosts_o] - c2[i + m + ghosts_m][j + n + ghosts_n][k + o + ghosts_o]) < tol);
                }
    }
}
