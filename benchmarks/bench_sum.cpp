#include "nanobench.h"

#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
using namespace std::chrono_literals;

#include "../src/cuboid.hpp"
#include "../src/problem.hpp"
#include "../src/util.hpp"
#include "../test/test_utility.hpp"
#include "bench_render_templates.hpp"
#include "pmg_utility.hpp"

#include "../thirdparty/mgcl_c/mgcl.hpp"
#include "../thirdparty/pmg/mg.h"

TEST_CASE("mgcl bench util::sum", "[!benchmark][sum][seqVsOcl]")
{
    std::vector grids{4, 8, 16, 32, 64, 128, 256, 512};
    // std::vector<size_t> locals{16, 32, 64};
    size_t local = 32;

    // for (auto local : locals)
    for (auto N : grids)
    {
        // int N = 16;
        int m = N;
        int n = N;
        int o = N;

        mgcl_test::TestUtility tu(CL_DEVICE_TYPE_GPU);

        mgcl::Cuboid data(m, n, o);
        data.fillRandom(-10, 10);

        cl_mem dData = tu.createOpenCLBuffer(data);

        ankerl::nanobench::Bench b;
        b.timeUnit(1ns, "ns")
            // .epochs(1)
            // .epochIterations(1)
            .minEpochTime(100ms)
            .maxEpochTime(5s)
            .relative(true);

        b.run(std::string("seq, N: ").append(std::to_string(N)).append(", wg: ").append(std::to_string(local)).c_str(), [&]
              {
                  double sum = 0;
                  for (int i = 0; i < m; i++)
                      for (int j = 0; j < n; j++)
                          for (int k = 0; k < o; k++)
                          {
                              sum += data[i][j][k];
                          }
                  ankerl::nanobench::doNotOptimizeAway(sum);
                  //
              });

        b.run(std::string("ocl, N: ").append(std::to_string(N)).append(", wg: ").append(std::to_string(local)).c_str(), [&]
              {
                  ankerl::nanobench::doNotOptimizeAway(
                      mgcl::util::sum(dData, data.field1d().size(), tu.getContext(), tu.getProgram(),
                                      tu.getCommands(), true, local)); //
              });
    }
}

TEST_CASE("mgcl bench util::sum", "[!benchmark][sum][locals]")
{
    // Almost not effect for small grids (plus they are fast anyway).
    std::vector grids{32, 64, 128, 256, 512};
    std::vector<size_t> locals{16, 32, 64, 128, 192, 256, 384, 512, 768, 1024};

    for (auto N : grids)
    {
        // int N = 16;
        int m = N;
        int n = N;
        int o = N;

        mgcl_test::TestUtility tu(CL_DEVICE_TYPE_GPU);

        mgcl::Cuboid data(m, n, o);
        data.fillRandom(-10, 10);

        cl_mem dData = tu.createOpenCLBuffer(data);

        ankerl::nanobench::Bench b;
        b.timeUnit(1ns, "ns")
            // .epochs(1)
            // .epochIterations(1)
            .minEpochTime(100ms)
            .maxEpochTime(5s)
            .relative(true);

        for (auto local : locals)
        {
            b.run(std::string("ocl, N: ").append(std::to_string(N)).append(", wg: ").append(std::to_string(local)).c_str(), [&]
                  {
                      ankerl::nanobench::doNotOptimizeAway(
                          mgcl::util::sum(dData, data.field1d().size(), tu.getContext(), tu.getProgram(),
                                          tu.getCommands(), true, local)); //
                  });
        }
        std::cout << "=============" << std::endl;
    }
}
