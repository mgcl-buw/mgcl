/**
 * @file bench_ignore_tol.cpp
 * @brief Compares solving with and without ignoring the tolerance, while having a fixed number of
 * iterations and a tolerance set that is never reached.
 * @date 2024-03-01
 *
 */

#include "nanobench.h"

#include "catch2/catch_test_macros.hpp"

#include <chrono>
#include <iostream>
#include <vector>
using namespace std::chrono_literals;

#include "../src/mgcl/cuboid.hpp"
#include "../src/mgcl/problem.hpp"

#include "cli_args.hpp"

TEST_CASE("benchIgnoreTol")
{
    if (CLI_ARGS::grids.size() == 0 && (CLI_ARGS::gridsMin.size() == 0 || CLI_ARGS::gridsMax.size() == 0))
        throw "Need to specify at least one local grid size, e.g. using --grids 4,8,16 or --gridsMin 4,4,4 AND --gridsMax 32,32,32";

    // build grids to be tested from CLI args
    std::vector<std::vector<int>> gridsTBT;
    for (auto N : CLI_ARGS::grids)
        gridsTBT.push_back({N, N, N});
    if (CLI_ARGS::gridsMin.size() > 0 && CLI_ARGS::gridsMax.size() > 0)
        for (int m = CLI_ARGS::gridsMin[0]; m <= CLI_ARGS::gridsMax[0]; m *= 2)
            for (int n = CLI_ARGS::gridsMin[1]; n <= CLI_ARGS::gridsMax[1]; n *= 2)
                for (int o = CLI_ARGS::gridsMin[2]; o <= CLI_ARGS::gridsMax[2]; o *= 2)
                    gridsTBT.push_back({m, n, o});

    std::cout << "Testing the following grid sizes" << std::endl;
    for (auto gr : gridsTBT)
    {
        int m = gr[0];
        int n = gr[1];
        int o = gr[2];
        std::cout << "  " << m << "," << n << "," << o << std::endl;
    }

    double omega = 0.8;
    double tol = 1e-7;
    int nu1 = CLI_ARGS::nu1;
    int nu2 = CLI_ARGS::nu2;
    int maxIterVCycles = CLI_ARGS::vCycleIterations;

    std::cout << "Problem parameters:" << std::endl
              << "  tol: " << tol << std::endl
              << "  nu1: " << nu1 << std::endl
              << "  nu2: " << nu2 << std::endl
              << "  omega: " << omega << std::endl
              << "  v-cycle iterations: " << maxIterVCycles << std::endl;

    // for each grid
    for (auto gr : gridsTBT)
    {
        int m = gr[0];
        int n = gr[1];
        int o = gr[2];

        auto f = std::make_shared<mgcl::Cuboid>(m, n, o);
        auto v = std::make_shared<mgcl::Cuboid>(m, n, o);
        v->fillRandom();

        // periodic function 4th order on right hand side
        double hm = 1.0 / static_cast<double>(m);
        double hn = 1.0 / static_cast<double>(n);
        double ho = 1.0 / static_cast<double>(o);
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < o; k++)
                {
                    double zs = i * ho;
                    double ys = j * hn;
                    double xs = k * hm;
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
                    (*f)[i][j][k] =
                        -1000000 *
                        (12 * xs4 * ys4 * zs4 * xsm1_4 * ysm1_4 * zsm1_2 + 12 * xs4 * ys4 * zs4 * xsm1_4 * ysm1_2 * zsm1_4 +
                         12 * xs4 * ys4 * zs4 * xsm1_2 * ysm1_4 * zsm1_4 + 32 * xs4 * ys4 * zs3 * xsm1_4 * ysm1_4 * zsm1_3 +
                         12 * xs4 * ys4 * zs2 * xsm1_4 * ysm1_4 * zsm1_4 + 32 * xs4 * ys3 * zs4 * xsm1_4 * ysm1_3 * zsm1_4 +
                         12 * xs4 * ys2 * zs4 * xsm1_4 * ysm1_4 * zsm1_4 + 32 * xs3 * ys4 * zs4 * xsm1_3 * ysm1_4 * zsm1_4 +
                         12 * xs2 * ys4 * zs4 * xsm1_4 * ysm1_4 * zsm1_4);
                }

        mgcl::Problem p(m, n, o, f, v);
        p.setUseOpencl(true);
        // p.setProfilingEnabled(true);
        p.setStencilType(mgcl::MGCL_VARYING);
        p.setSilent(true);
        p.setNu1(nu1);
        p.setNu2(nu2);
        p.setOmega(omega);
        p.setMaxiterVcycles(maxIterVCycles);
        p.setTol(tol);

        auto svptr = p.getStencilValues();
        auto& s = *svptr;
        for (int i = 0; i < s.getMgh(); i++)
            for (int j = 0; j < s.getNgh(); j++)
                for (int k = 0; k < s.getOgh(); k++)
                {
                    // 7-point Laplace
                    s[0][1][1][i][j][k] = (1.0 / (hm * hm)) * -1.0;
                    s[1][0][1][i][j][k] = (1.0 / (hm * hm)) * -1.0;
                    s[1][1][0][i][j][k] = (1.0 / (hm * hm)) * -1.0;
                    s[1][1][1][i][j][k] = (1.0 / (hm * hm)) * 6.0;
                    s[1][1][2][i][j][k] = (1.0 / (hm * hm)) * -1.0;
                    s[1][2][1][i][j][k] = (1.0 / (hm * hm)) * -1.0;
                    s[2][1][1][i][j][k] = (1.0 / (hm * hm)) * -1.0;
                }

        p.init();

        ankerl::nanobench::Bench b;
        b.timeUnit(1ms, "ms")
            // .epochs(1)
            // .epochIterations(1)
            .minEpochTime(100ms)
            .maxEpochTime(5s)
            .relative(true);
        std::string name;

        p.setIgnoreTol(false);
        name = std::string("ignoreFalse_")
                   .append(std::to_string(m))
                   .append("x")
                   .append(std::to_string(n))
                   .append("x")
                   .append(std::to_string(o));
        b.run(name.c_str(), [&] { //
            p.solve(true);
            p.getOpenCLHelper().finish();
        });

        p.setIgnoreTol(true);
        name = std::string("ignoreTrue__")
                   .append(std::to_string(m))
                   .append("x")
                   .append(std::to_string(n))
                   .append("x")
                   .append(std::to_string(o));
        b.run(name.c_str(), [&] { //
            p.solve(true);
            p.getOpenCLHelper().finish();
        });
    }
}