#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cmath>
#include <iostream>

#include "../src/cuboid.hpp"
#include "../src/multigrid_engine.hpp"
#include "../src/stencil.hpp"
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
        auto deviceType = GENERATE(CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU);

        if (!mgcl_test::TestUtility::deviceAvailable("", deviceType))
        {
            std::string typeName = deviceType == CL_DEVICE_TYPE_GPU ? "CL_DEVICE_TYPE_GPU" : "CL_DEVICE_TYPE_CPU";
            std::cout << "Skipping non-available device type '" << typeName << "'" << std::endl;
            return;
        }

        p->setDeviceType(deviceType);

        mgcl_test::TestUtility tu(p);
        cl_mem d_c_fine = tu.createOpenCLBuffer(*c_fine);
        cl_mem d_c_coarse = tu.createOpenCLBuffer(*c_coarse);

        mgcl::MultigridEngine::restrict(lv_fine, lv_coarse, d_c_fine, d_c_coarse);
        tu.finish();

        auto c_fine_out = tu.readOpenCLBuffer(d_c_fine, m, n, o, ghosts_m, ghosts_n, ghosts_o);
        auto c_coarse_out = tu.readOpenCLBuffer(d_c_coarse, c_expected_coarse->getM(), c_expected_coarse->getN(),
                                                c_expected_coarse->getO(), ghosts_m, ghosts_n, ghosts_o);

        REQUIRE(c_fine_out->isEqual(*c_expected_fine));
        REQUIRE(c_coarse_out->isEqual(*c_expected_coarse));
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
