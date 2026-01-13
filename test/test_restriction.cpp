#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cmath>
#include <iostream>
#include <memory>

#include "../src/mgcl/cuboid.hpp"
#include "../src/mgcl/level.hpp"
#include "../src/mgcl/multigrid_engine.hpp"
#include "../src/mgcl/stencil.hpp"
#include "cli_args.hpp"
#include "device_type_generator.hpp"
#include "test_results.hpp"
#include "test_utility.hpp"

std::shared_ptr<mgcl::Cuboid> restrictionTestInputFine();
std::shared_ptr<mgcl::Cuboid> restrictionTestInputCoarse();
std::shared_ptr<mgcl::Cuboid> restrictionTestOutputFine();
std::shared_ptr<mgcl::Cuboid> restrictionTestOutputCoarse();

TEST_CASE("restriction")
{
    int m = 16;
    int n = 16;
    int o = 16;
    int ghosts_m = 1;
    int ghosts_n = 1;
    int ghosts_o = 1;
    int mgh = m + 2 * ghosts_m;
    int ngh = n + 2 * ghosts_n;
    int ogh = o + 2 * ghosts_o;

    auto c_fine = mgcl_test::test_restriction::inputFine16();
    auto c_coarse = mgcl_test::test_restriction::inputCoarse8();
    auto c_expected_fine = mgcl_test::test_restriction::outputFine16();
    auto c_expected_coarse = mgcl_test::test_restriction::outputCoarse8();

    auto p = std::make_shared<mgcl::Problem>(m, n, o);
    mgcl::Level lv_fine(p.get(), 0);
    mgcl::Level lv_coarse(p.get(), 1);

    SECTION("restrictSeq")
    {
        mgcl::MultigridEngine::restrictSeq(lv_fine, lv_coarse, *c_fine, *c_coarse);

        REQUIRE(c_fine->isEqual(*c_expected_fine));
        REQUIRE(c_coarse->isEqual(*c_expected_coarse));
    }

    SECTION("restrict OpenCL")
    {
        auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));

        p->setDeviceType(deviceType);

        mgcl_test::TestUtility tu(p);
        mgcl::CuboidGpu d_c_fine(tu.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, *c_fine);
        mgcl::CuboidGpu d_c_coarse(tu.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, *c_coarse);

        mgcl::MultigridEngine::restrict(lv_fine, lv_coarse, d_c_fine, d_c_coarse);
        tu.finish();

        auto c_fine_out = d_c_fine.read(tu.getCommands(), nullptr, true);
        auto c_coarse_out = d_c_coarse.read(tu.getCommands(), nullptr, true);

        REQUIRE(c_fine_out->isEqual(*c_expected_fine));
        REQUIRE(c_coarse_out->isEqual(*c_expected_coarse));
    }
}

TEST_CASE("restriction1gpDirichlet")
{
    int m = 2;
    int n = 2;
    int o = 2;
    int mc = m / 2;
    int nc = n / 2;
    int oc = o / 2;
    int ghosts_m = 1;
    int ghosts_n = 1;
    int ghosts_o = 1;
    int mgh = m + 2 * ghosts_m;
    int ngh = n + 2 * ghosts_n;
    int ogh = o + 2 * ghosts_o;

    auto c_fine = std::make_shared<mgcl::Cuboid>(m, n, o, ghosts_m, ghosts_n, ghosts_o);
    auto c_coarse = std::make_shared<mgcl::Cuboid>(mc, nc, oc, ghosts_m, ghosts_n, ghosts_o);
    c_fine->fill(0);

    (*c_fine)[ghosts_m + 1][ghosts_n + 1][ghosts_o + 1] = 1; // self
    // direct neighbors
    (*c_fine)[ghosts_m + 1][ghosts_n + 1][ghosts_o + 1 - 1] = 1;
    (*c_fine)[ghosts_m + 1][ghosts_n + 1][ghosts_o + 1 + 1] = 2;
    (*c_fine)[ghosts_m + 1][ghosts_n + 1 - 1][ghosts_o + 1] = 3;
    (*c_fine)[ghosts_m + 1][ghosts_n + 1 + 1][ghosts_o + 1] = 4;
    (*c_fine)[ghosts_m + 1 - 1][ghosts_n + 1][ghosts_o + 1] = 5;
    (*c_fine)[ghosts_m + 1 + 1][ghosts_n + 1][ghosts_o + 1] = 6;
    // edge midpoints xy-plane
    (*c_fine)[ghosts_m + 1 - 1][ghosts_n + 1 + 1][ghosts_o + 1] = 7;
    (*c_fine)[ghosts_m + 1 - 1][ghosts_n + 1 - 1][ghosts_o + 1] = 8;
    (*c_fine)[ghosts_m + 1 + 1][ghosts_n + 1 + 1][ghosts_o + 1] = 9;
    (*c_fine)[ghosts_m + 1 + 1][ghosts_n + 1 - 1][ghosts_o + 1] = 10;
    // edge midpoints xz-plane
    (*c_fine)[ghosts_m + 1 - 1][ghosts_n + 1][ghosts_o + 1 + 1] = 11;
    (*c_fine)[ghosts_m + 1 - 1][ghosts_n + 1][ghosts_o + 1 - 1] = 12;
    (*c_fine)[ghosts_m + 1 + 1][ghosts_n + 1][ghosts_o + 1 + 1] = 13;
    (*c_fine)[ghosts_m + 1 + 1][ghosts_n + 1][ghosts_o + 1 - 1] = 14;
    // edge midpoints yz-plane
    (*c_fine)[ghosts_m + 1][ghosts_n + 1 - 1][ghosts_o + 1 + 1] = 15;
    (*c_fine)[ghosts_m + 1][ghosts_n + 1 - 1][ghosts_o + 1 - 1] = 16;
    (*c_fine)[ghosts_m + 1][ghosts_n + 1 + 1][ghosts_o + 1 + 1] = 17;
    (*c_fine)[ghosts_m + 1][ghosts_n + 1 + 1][ghosts_o + 1 - 1] = 18;
    // corners
    (*c_fine)[ghosts_m + 1 - 1][ghosts_n + 1 - 1][ghosts_o + 1 - 1] = 19;
    (*c_fine)[ghosts_m + 1 - 1][ghosts_n + 1 - 1][ghosts_o + 1 + 1] = 20;
    (*c_fine)[ghosts_m + 1 - 1][ghosts_n + 1 + 1][ghosts_o + 1 - 1] = 21;
    (*c_fine)[ghosts_m + 1 - 1][ghosts_n + 1 + 1][ghosts_o + 1 + 1] = 22;
    (*c_fine)[ghosts_m + 1 + 1][ghosts_n + 1 - 1][ghosts_o + 1 - 1] = 23;
    (*c_fine)[ghosts_m + 1 + 1][ghosts_n + 1 - 1][ghosts_o + 1 + 1] = 24;
    (*c_fine)[ghosts_m + 1 + 1][ghosts_n + 1 + 1][ghosts_o + 1 - 1] = 25;
    (*c_fine)[ghosts_m + 1 + 1][ghosts_n + 1 + 1][ghosts_o + 1 + 1] = 26;
    c_fine->dumpToFile("cfine", false);

    // clang-format off
    double expected = 0.125 * 1
    + 0.0625 * 1
    + 0.0625 * 2
    + 0.0625 * 3
    + 0.0625 * 4
    + 0.0625 * 5
    + 0.0625 * 6
    + 0.03125 * 7
    + 0.03125 * 8
    + 0.03125 * 9
    + 0.03125 * 10
    + 0.03125 * 11
    + 0.03125 * 12
    + 0.03125 * 13
    + 0.03125 * 14
    + 0.03125 * 15
    + 0.03125 * 16
    + 0.03125 * 17
    + 0.03125 * 18
    + 0.015625 * 19
    + 0.015625 * 20
    + 0.015625 * 21
    + 0.015625 * 22
    + 0.015625 * 23
    + 0.015625 * 24
    + 0.015625 * 25
    + 0.015625 * 26;
    // clang-format on

    auto p = std::make_shared<mgcl::Problem>(m, n, o);
    p->setBc(mgcl::BC::DIRICHLET);
    mgcl::Level lv_fine(p.get(), 0);
    mgcl::Level lv_coarse(p.get(), 1);

    SECTION("restrictSeq")
    {
        mgcl::MultigridEngine::restrictSeq(lv_fine, lv_coarse, *c_fine, *c_coarse);

        REQUIRE((*c_coarse)[ghosts_m][ghosts_n][ghosts_o] == expected);
    }

    SECTION("restrict OpenCL")
    {
        auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));

        p->setDeviceType(deviceType);

        mgcl_test::TestUtility tu(p);
        mgcl::CuboidGpu d_c_fine(tu.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, *c_fine);
        mgcl::CuboidGpu d_c_coarse(tu.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, *c_coarse);

        mgcl::MultigridEngine::restrict(lv_fine, lv_coarse, d_c_fine, d_c_coarse);
        tu.finish();

        auto c_coarse_out = d_c_coarse.read(tu.getCommands(), nullptr, true);

        REQUIRE((*c_coarse_out)[ghosts_m][ghosts_n][ghosts_o] == expected);
    }
}

// TEST_CASE("restriction: fixed vs. varying stencil")
// {
//     // size of fine grid
//     int m = 16;
//     int n = 16;
//     int o = 16;

//     auto c_fine_fixed = mgcl::Cuboid(m, n, o);
//     c_fine_fixed.fillRandom();
//     auto c_fine_varying = mgcl::Cuboid::copyFrom(c_fine_fixed);
//     auto c_coarse_fixed = mgcl::Cuboid(m / 2, n / 2, o / 2);
//     auto c_coarse_varying = mgcl::Cuboid(m / 2, n / 2, o / 2);

//     auto s = mgcl::create3dFullWeightRestrictionStencil(m, n, o, 0, 0, 0);
//     int ghm = 0;
//     int ghn = 0;
//     int gho = 0;
//     int ghmsv = 0;
//     int ghnsv = 0;
//     int ghosv = 0;

//     // apply stencil
//     for (int i = ghm, isv = ghmsv; i < m + ghm; i++, isv++)
//         for (int j = ghn, jsv = ghnsv; j < n + ghn; j++, jsv++)
//             for (int k = gho, ksv = ghosv; k < o + gho; k++, ksv++)
//             {
//                 // clang-format off
//                 stencilsum = stencilValues[isv][jsv][ksv][1][1][1]  * vraw[i][j][k]
//                     + stencilValues[isv][jsv][ksv][1][1][0]         * vraw[i][j][k - 1]
//                     + stencilValues[isv][jsv][ksv][1][1][2]         * vraw[i][j][k + 1]
//                     + stencilValues[isv][jsv][ksv][1][0][1]         * vraw[i][j - 1][k]
//                     + stencilValues[isv][jsv][ksv][1][2][1]         * vraw[i][j + 1][k]
//                     + stencilValues[isv][jsv][ksv][0][1][1]         * vraw[i - 1][j][k]
//                     + stencilValues[isv][jsv][ksv][2][1][1]         * vraw[i + 1][j][k];
//                 // clang-format on
//             }

//     // create dummy problem and levels
//     auto p = std::make_shared<mgcl::Problem>(m, n, o);
//     mgcl::Level lv_fine(p.get(), 0);
//     mgcl::Level lv_coarse(p.get(), 1);

//     mgcl::MultigridEngine::restrictSeq(lv_fine, lv_coarse, c_fine_fixed, c_coarse_fixed);
// }
