#include <catch2/catch_test_macros.hpp>

#include <cmath>

#include "../src/cuboid.hpp"
#include "../src/multigrid_engine.hpp"
#include "test_utility.hpp"

TEST_CASE("updateGhosts")
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

    SECTION("periodic")
    {
        SECTION("seq")
        {
            mgcl::MultigridEngine::updateGhostsSeq(c1);

            // check in z-direction
            for (int i = 0; i < ghosts_m; i++)
                for (int j = 0; j < n + 2 * ghosts_n; j++)
                    for (int k = 0; k < o + 2 * ghosts_o; k++)
                    {
                        CHECK(c1[i][j][k] == c1[i + m][j][k]);
                        CHECK(c1[i + ghosts_m][j][k] == c1[i + ghosts_m + m][j][k]);
                    }

            // check in y-direction
            for (int i = 0; i < m + 2 * ghosts_m; i++)
                for (int j = 0; j < ghosts_n; j++)
                    for (int k = 0; k < o + 2 * ghosts_o; k++)
                    {
                        CHECK(c1[i][j][k] == c1[i][j + n][k]);
                        CHECK(c1[i][j + ghosts_n][k] == c1[i][j + ghosts_n + n][k]);
                    }

            // check in x-direction
            for (int i = 0; i < m + 2 * ghosts_m; i++)
                for (int j = 0; j < n + 2 * ghosts_n; j++)
                    for (int k = 0; k < ghosts_o; k++)
                    {
                        CHECK(c1[i][j][k] == c1[i][j][k + o]);
                        CHECK(c1[i][j][k + ghosts_o] == c1[i][j][k + ghosts_o + o]);
                    }
        }

        if (mgcl_test::TestUtility::deviceAvailable("", CL_DEVICE_TYPE_GPU))
        {
            SECTION("openclgpu")
            {
                mgcl_test::TestUtility tu(CL_DEVICE_TYPE_GPU);
                cl_mem d_c1 = tu.createOpenCLBuffer(c1);

                mgcl::MultigridEngine::updateGhosts(tu.getProblem(), d_c1, mgh, ngh, ogh, ghosts_m, ghosts_n, ghosts_o);
                tu.finish();

                auto c2 = tu.readOpenCLBuffer(d_c1, mgh, ngh, ogh);

                double tol = 1e-7;
                for (int i = 0; i < ghosts_m; i++)
                    for (int j = 0; j < ghosts_n; j++)
                        for (int k = 0; k < ghosts_o; k++)
                        {
                            REQUIRE(fabs((*c2)[i][j][k] - (*c2)[i + m][j + n][k + o]) < tol);
                            REQUIRE(fabs((*c2)[i + ghosts_m][j + ghosts_n][k + ghosts_o] - (*c2)[i + m + ghosts_m][j + n + ghosts_n][k + o + ghosts_o]) < tol);
                        }
            }
        }

        if (mgcl_test::TestUtility::deviceAvailable("", CL_DEVICE_TYPE_CPU))
        {
            SECTION("openclcpu")
            {
                mgcl_test::TestUtility tu(CL_DEVICE_TYPE_CPU);
                cl_mem d_c1 = tu.createOpenCLBuffer(c1);

                mgcl::MultigridEngine::updateGhosts(tu.getProblem(), d_c1, mgh, ngh, ogh, ghosts_m, ghosts_n, ghosts_o);
                tu.finish();

                auto c2 = tu.readOpenCLBuffer(d_c1, mgh, ngh, ogh);

                double tol = 1e-7;
                for (int i = 0; i < ghosts_m; i++)
                    for (int j = 0; j < ghosts_n; j++)
                        for (int k = 0; k < ghosts_o; k++)
                        {
                            REQUIRE(fabs((*c2)[i][j][k] - (*c2)[i + m][j + n][k + o]) < tol);
                            REQUIRE(fabs((*c2)[i + ghosts_m][j + ghosts_n][k + ghosts_o] - (*c2)[i + m + ghosts_m][j + n + ghosts_n][k + o + ghosts_o]) < tol);
                        }
            }
        }
    }

    // SECTION("non-periodic")
    // {
    //     // reset outermost border to zero
    //     // TODO
    //     for (int i = 0; i < 1; i++)
    //         for (int j = 0; j < 1; j++)
    //             for (int k = 0; k < 1; k++)
    //             {
    //                 c1[i][j][k] = 0;
    //             }

    //     SECTION("seq")
    //     {
    //         mgcl::MultigridEngine::updateGhostsSeq(c1);
    //         for (int i = 0; i < ghosts_m; i++)
    //             for (int j = 0; j < ghosts_n; j++)
    //                 for (int k = 0; k < ghosts_o; k++)
    //                 {
    //                     REQUIRE(c1[i][j][k] == c1[i + m][j + n][k + o]);
    //                     REQUIRE(c1[i + ghosts_m][j + ghosts_n][k + ghosts_o] == c1[i + m + ghosts_m][j + n + ghosts_n][k + o + ghosts_o]);
    //                 }
    //     }

    //     // TODO ocl
    // }
}
