#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cmath>
#include <iostream>

#include "../cuboid.hpp"
#include "../multigrid_engine.hpp"
#include "test_results.hpp"
#include "test_utility.hpp"

std::shared_ptr<mgcl::Cuboid> residualTestInputF();
std::shared_ptr<mgcl::Cuboid> residualTestInputV();
std::shared_ptr<mgcl::Cuboid> residualTestOutputR();

TEST_CASE("residual")
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

    auto c_in_f = mgcl_test::test_residual::inputF16();
    auto c_in_v = mgcl_test::test_residual::inputV16();
    auto c_in_r = std::make_unique<mgcl::Cuboid>(m, n, o, ghosts_m, ghosts_n, ghosts_o);
    auto c_expected_out_r = mgcl_test::test_residual::outputR16();

    auto stencil = mgcl::MGCL_LAPLACE_7POINT;
    double h = 1.0 / (double)m;
    double stencilFactor = 1.0 / (h * h);

    SECTION("residualSeq L2-norm 7point")
    {
        auto stencilValues = std::make_unique<mgcl::VaryingStencil3x3x3>(1, 1, 1, 0, 0, 0); // just a dummy
        double res = mgcl::MultigridEngine::residualSeq(*c_in_f, *c_in_v, *c_in_r, mgcl::MGCL_L2,
                                                        mgcl::MGCL_LAPLACE_7POINT, stencilFactor, *stencilValues, true);

        CHECK(fabs(res - 3.00209960095333271e+07) < 1e-7);
        REQUIRE(c_in_r->isEqual(*c_expected_out_r));
    }

    SECTION("residual OpenCL L2-norm 7point")
    {
        auto deviceType = GENERATE(CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU);

        if (!mgcl_test::TestUtility::deviceAvailable("", deviceType))
        {
            std::string typeName = deviceType == CL_DEVICE_TYPE_GPU ? "CL_DEVICE_TYPE_GPU" : "CL_DEVICE_TYPE_CPU";
            std::cout << "Skipping non-available device type '" << typeName << "'" << std::endl;
            return;
        }

        auto p = std::make_shared<mgcl::Problem>(m, n, o);
        p->setResidualNorm(mgcl::MGCL_L2);
        p->setStencilType(stencil);
        p->setGhosts(1);
        p->setDeviceType(deviceType);

        mgcl_test::TestUtility tu(p);
        cl_mem d_in_f = tu.createOpenCLBuffer(*c_in_f);
        cl_mem d_in_v = tu.createOpenCLBuffer(*c_in_v);
        cl_mem d_in_r = tu.createOpenCLBuffer(*c_in_r);

        mgcl::Level level(p.get(), 0);
        level.setDF(d_in_f);
        level.setDVIn(d_in_v);
        level.setDR(d_in_r);

        double res = mgcl::MultigridEngine::residual(*p, level, 1);
        tu.finish();

        auto c_r_out = tu.readOpenCLBuffer(d_in_r, m, n, o, ghosts_m, ghosts_n, ghosts_o);

        CHECK(fabs(res - 3.00209960095333271e+07) < 1e-7);
        REQUIRE(c_r_out->isEqual(*c_expected_out_r));
    }

    SECTION("residualSeq L2-norm 7point varying stencil")
    {
        int n = 16;
        auto vals = mgcl::VaryingStencil3x3x3(n, n, n, 1, 1, 1);

        double h2inv = static_cast<double>(n * n);
        for (int i = 0; i < vals.getDim1gh(); i++)
            for (int j = 0; j < vals.getDim2gh(); j++)
                for (int k = 0; k < vals.getDim3gh(); k++)
                {
                    vals[i][j][k][1][1][1] = 6.0 * h2inv;
                    vals[i][j][k][1][1][0] = -1.0 * h2inv;
                    vals[i][j][k][1][1][2] = -1.0 * h2inv;
                    vals[i][j][k][1][0][1] = -1.0 * h2inv;
                    vals[i][j][k][1][2][1] = -1.0 * h2inv;
                    vals[i][j][k][0][1][1] = -1.0 * h2inv;
                    vals[i][j][k][2][1][1] = -1.0 * h2inv;
                }

        double res = mgcl::MultigridEngine::residualSeq(*c_in_f, *c_in_v, *c_in_r, mgcl::MGCL_L2,
                                                        mgcl::MGCL_VARYING_7POINT, stencilFactor, vals, true);

        CHECK(fabs(res - 3.00209960095333271e+07) < 1e-7);
        REQUIRE(c_in_r->isEqual(*c_expected_out_r));
    }
}

// TODO maybe this test doesn't make sense after all
// TEST_CASE("residual galerkin")
// {
//     // creates 7p Laplace stencil on fine grid, applies galerkin operator and calculates residual on coarse grid
//     SECTION("residualSeq L2-norm 7point galerkin")
//     {
//         int n = 16;
//         double h = 1.0 / (double)n;
//         double stencilFactor = 1.0 / (h * h);

//         mgcl::Cuboid c_in_f(n, n, n, 2, 2, 2);
//         mgcl::Cuboid c_in_v(n, n, n, 2, 2, 2);
//         mgcl::Cuboid c_r_galerkin(n, n, n, 2, 2, 2, 0);
//         mgcl::Cuboid c_r_explicit(n, n, n, 2, 2, 2, 0);
//         auto vals_fine = mgcl::VaryingStencil3x3x3(2 * n, 2 * n, 2 * n, 2, 2, 2);

//         c_in_f.fillRandom();
//         c_in_v.fillRandom();

//         // fill fine grid stencil with 7p Laplace on which galerkin will be applied
//         double h2inv_fine = static_cast<double>(2 * n * 2 * n);
//         for (int i = 0; i < vals_fine.getDim1gh(); i++)
//             for (int j = 0; j < vals_fine.getDim2gh(); j++)
//                 for (int k = 0; k < vals_fine.getDim3gh(); k++)
//                 {
//                     vals_fine[i][j][k][1][1][1] = 6.0 * h2inv_fine;
//                     vals_fine[i][j][k][1][1][0] = -1.0 * h2inv_fine;
//                     vals_fine[i][j][k][1][1][2] = -1.0 * h2inv_fine;
//                     vals_fine[i][j][k][1][0][1] = -1.0 * h2inv_fine;
//                     vals_fine[i][j][k][1][2][1] = -1.0 * h2inv_fine;
//                     vals_fine[i][j][k][0][1][1] = -1.0 * h2inv_fine;
//                     vals_fine[i][j][k][2][1][1] = -1.0 * h2inv_fine;
//                 }

//         auto vals_coarse = mgcl::MultigridEngine::galerkin(vals_fine);

//         double res_galerkin = mgcl::MultigridEngine::residualSeq(c_in_f, c_in_v, c_r_galerkin, mgcl::MGCL_L2,
//                                                                  mgcl::MGCL_VARYING_7POINT, stencilFactor, vals_coarse, true);

//         double res_explicit = mgcl::MultigridEngine::residualSeq(c_in_f, c_in_v, c_r_explicit, mgcl::MGCL_L2,
//                                                                  mgcl::MGCL_LAPLACE_7POINT, stencilFactor, vals_coarse, true);

//         CHECK(fabs(res_galerkin - res_explicit) < 1e-7);
//         REQUIRE(c_r_galerkin.isEqual(c_r_explicit));
//     }
// }
