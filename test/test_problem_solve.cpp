#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "../cuboid.hpp"
#include "../problem.hpp"

/**
 * @brief Tests if solving works correctly.
 *
 */
TEST_CASE("Problem solving")
{
    int N = 64;
    double h = 1.0 / (double)N;

    SECTION("periodic 4th order")
    {
        auto vseq = mgcl::Cuboid(N, N, N);
        auto fseq = mgcl::Cuboid(N, N, N);
        auto vocl = mgcl::Cuboid(N, N, N);
        auto focl = mgcl::Cuboid(N, N, N);
        auto solution = mgcl::Cuboid(N, N, N);

        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                for (int k = 0; k < N; k++)
                {
                    double zs = i * h;
                    double ys = j * h;
                    double xs = k * h;
                    double xs2 = xs * xs;
                    double ys2 = ys * ys;
                    double zs2 = zs * zs;
                    double xsm1_2 = (xs - 1) * (xs - 1);
                    double ysm1_2 = (ys - 1) * (ys - 1);
                    double zsm1_2 = (zs - 1) * (zs - 1);
                    double xs3 = xs * xs * xs;
                    double ys3 = ys * ys * ys;
                    double zs3 = zs * zs * zs;
                    double xsm1_3 = (xs - 1) * (xs - 1) * (xs - 1);
                    double ysm1_3 = (ys - 1) * (ys - 1) * (ys - 1);
                    double zsm1_3 = (zs - 1) * (zs - 1) * (zs - 1);
                    double xs4 = xs * xs * xs * xs;
                    double ys4 = ys * ys * ys * ys;
                    double zs4 = zs * zs * zs * zs;
                    double xsm1_4 = (xs - 1) * (xs - 1) * (xs - 1) * (xs - 1);
                    double ysm1_4 = (ys - 1) * (ys - 1) * (ys - 1) * (ys - 1);
                    double zsm1_4 = (zs - 1) * (zs - 1) * (zs - 1) * (zs - 1);
                    vseq[i][j][k] = vocl[i][j][k] = 0;
                    solution[i][j][k] = 1000000 * (xs * (xs - 1)) * (xs * (xs - 1)) * (xs * (xs - 1)) * (xs * (xs - 1)) *
                                        (ys * (ys - 1)) * (ys * (ys - 1)) * (ys * (ys - 1)) * (ys * (ys - 1)) *
                                        (zs * (zs - 1)) * (zs * (zs - 1)) * (zs * (zs - 1)) * (zs * (zs - 1));
                    fseq[i][j][k] = focl[i][j][k] =
                        -1000000 *
                        (12 * xs4 * ys4 * zs4 * xsm1_4 * ysm1_4 * zsm1_2 + 12 * xs4 * ys4 * zs4 * xsm1_4 * ysm1_2 * zsm1_4 +
                         12 * xs4 * ys4 * zs4 * xsm1_2 * ysm1_4 * zsm1_4 + 32 * xs4 * ys4 * zs3 * xsm1_4 * ysm1_4 * zsm1_3 +
                         12 * xs4 * ys4 * zs2 * xsm1_4 * ysm1_4 * zsm1_4 + 32 * xs4 * ys3 * zs4 * xsm1_4 * ysm1_3 * zsm1_4 +
                         12 * xs4 * ys2 * zs4 * xsm1_4 * ysm1_4 * zsm1_4 + 32 * xs3 * ys4 * zs4 * xsm1_3 * ysm1_4 * zsm1_4 +
                         12 * xs2 * ys4 * zs4 * xsm1_4 * ysm1_4 * zsm1_4);
                }

        mgcl::Problem pseq(N, N, N, vseq, fseq);
        pseq.solveSeq();

        REQUIRE(solution.isEqual(vseq));

        mgcl::Problem pocl(N, N, N, vocl, focl);
        // pocl.setDeviceType(CL_DEVICE_TYPE_GPU);
        pocl.solve();

        REQUIRE(solution.isEqual(vocl));
    }
}
