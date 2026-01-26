#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>

#include "../../src/mgcl/cuboid.hpp"
#include "../../src/mgcl/problem.hpp"
#include "../cli_args.hpp"
#include "../device_type_generator.hpp"
#include "../test_utility.hpp"

/**
 * @brief Tests if solving works correctly for u = x^4 * (x-1)^4, where u is a vector with only one component.
 *
 */
TEST_CASE("solve_bs_periodic_blocksize1")
{
    int N = 16;
    double h = 1.0 / (double)N;

    // Problem parameters
    double tol = 1e-14;
    int nu1 = 2;
    int nu2 = 2;
    double omega = 0.8;
    int maxIterVCycles = 5;
    int maxlevel = 10;

    auto vsc = std::make_shared<mgcl::Cuboid>(N, N, N);
    auto fsc = std::make_shared<mgcl::Cuboid>(N, N, N);
    auto solutionsc = mgcl::Cuboid(N, N, N);

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
                (*vsc)[i][j][k] = 0;
                solutionsc[i][j][k] = 1000000 * (xs * (xs - 1)) * (xs * (xs - 1)) * (xs * (xs - 1)) * (xs * (xs - 1)) *
                                      (ys * (ys - 1)) * (ys * (ys - 1)) * (ys * (ys - 1)) * (ys * (ys - 1)) *
                                      (zs * (zs - 1)) * (zs * (zs - 1)) * (zs * (zs - 1)) * (zs * (zs - 1));
                (*fsc)[i][j][k] =
                    -1000000 *
                    (12 * xs4 * ys4 * zs4 * xsm1_4 * ysm1_4 * zsm1_2 + 12 * xs4 * ys4 * zs4 * xsm1_4 * ysm1_2 * zsm1_4 +
                     12 * xs4 * ys4 * zs4 * xsm1_2 * ysm1_4 * zsm1_4 + 32 * xs4 * ys4 * zs3 * xsm1_4 * ysm1_4 * zsm1_3 +
                     12 * xs4 * ys4 * zs2 * xsm1_4 * ysm1_4 * zsm1_4 + 32 * xs4 * ys3 * zs4 * xsm1_4 * ysm1_3 * zsm1_4 +
                     12 * xs4 * ys2 * zs4 * xsm1_4 * ysm1_4 * zsm1_4 + 32 * xs3 * ys4 * zs4 * xsm1_3 * ysm1_4 * zsm1_4 +
                     12 * xs2 * ys4 * zs4 * xsm1_4 * ysm1_4 * zsm1_4);
            }

    SECTION("Sequential")
    {
        SECTION("Galerkin_27p")
        {
            mgcl::Problem psc(N, N, N, fsc, vsc);
            psc.setMaxiterVcycles(maxIterVCycles);
            psc.setTol(tol);
            psc.setNu1(nu1);
            psc.setNu2(nu2);
            psc.setOmega(omega);
            psc.setMaxlevel(maxlevel);

            psc.setStencilType(mgcl::MGCL_VARYING);
            auto& s = *psc.createStencilValues();
            mgcl_test::fill27pLaplace(s, h, false);

            psc.solveSeq();

            // check if input v is equal to the v stored in Problem instance
            REQUIRE(vsc.get() == psc.getVPtr().get());
            REQUIRE(vsc->isEqual(psc.getV()));

            // check if solution is good
            auto err = mgcl_test::calculateError(solutionsc, *vsc);
            auto errNorm = mgcl_test::calculateErrorNorm(1.0 / (double)N, *err);
            auto errMax = mgcl_test::calculateMaxError(*err);

            // solution.dumpToFile("out_solution.txt");
            // (*v).dumpToFile("out_v.txt");

            std::cout
                << "seq Galerkin 27p" << std::endl
                << std::scientific << std::setprecision(17) << "  ||e||_2 = " << errNorm << std::endl
                << std::scientific << std::setprecision(17) << "  e_max = " << errMax << std::endl;

            CHECK(errNorm < 1e-2);
            CHECK(errMax < 1e-2);
        }

        SECTION("block")
        {
            int blocksize = 1;
            std::shared_ptr<mgcl::CuboidBS> vbs = std::make_shared<mgcl::CuboidBS>(N, N, N, 0, 0, 0, blocksize);
            std::shared_ptr<mgcl::CuboidBS> fbs = std::make_shared<mgcl::CuboidBS>(N, N, N, 0, 0, 0, blocksize);
            // mgcl::Blockstencil bs(Nblockstencil, Nblockstencil, Nblockstencil, 3, blocksize, 0, 0, 0);

            // bs_inv.dumpToFile("bs_inv.txt");
            // bs.dumpToFile("bs.txt");

            // fill v with values of v1 and v2, vice versa for f
            mgcl_test::copyCuboidToCuboidBS(*vsc, *vbs, 1, 1, 1);
            mgcl_test::copyCuboidToCuboidBS(*fsc, *fbs, 1, 1, 1);

            mgcl::Problem pbs(N, N, N, fbs, vbs);
            // pbs.setIgnoreTol(true);
            pbs.setMaxiterVcycles(maxIterVCycles);
            pbs.setTol(tol);
            pbs.setNu1(nu1);
            pbs.setNu2(nu2);
            pbs.setOmega(omega);
            pbs.setMaxlevel(maxlevel);
            pbs.setStencilType(mgcl::MGCL_BLOCKSTENCIL);

            auto bs = pbs.createBlockstencil();
            mgcl_test::fill27pLaplace(*bs, h, false);
            // bs->dumpToFile("bs.txt");
            auto r = pbs.getRestrictionBlockstencil();
            auto p = pbs.getProlongationBlockstencil();
            mgcl_test::fill3dFullWeightRestrictionBlockstencil(*r);
            mgcl_test::fill3dBilinearProlongationBlockstencil(*p);

            SECTION("Blockstencil_Jacobiscalar")
            {
                pbs.setSmootherType(mgcl::MGCL_JACOBI_SCALAR);

                pbs.solveSeq();

                mgcl_test::copyCuboidBSToCuboid(*vbs, *vsc, 1, 1, 1);

                // check if input v is equal to the v stored in Problem instance
                REQUIRE(vbs.get() == pbs.getVBSPtr().get());
                REQUIRE(vbs->isEqual(pbs.getVBS()));

                // check if solution is good
                auto err = mgcl_test::calculateError(solutionsc, *vsc);
                auto errNorm = mgcl_test::calculateErrorNorm(1.0 / (double)N, *err);
                auto errMax = mgcl_test::calculateMaxError(*err);

                // solution.dumpToFile("out_solution.txt");
                // (*v).dumpToFile("out_v.txt");

                std::cout
                    << "seq Blockstencil scalar Jacobi" << std::endl
                    << std::scientific << std::setprecision(17) << "  ||e||_2 = " << errNorm << std::endl
                    << std::scientific << std::setprecision(17) << "  e_max = " << errMax << std::endl;

                CHECK(errNorm < 1e-2);
                CHECK(errMax < 1e-2);
            }

            SECTION("Blockstencil_Jacobiblock")
            {
                pbs.setSmootherType(mgcl::MGCL_JACOBI_BLOCK);

                pbs.solveSeq();

                mgcl_test::copyCuboidBSToCuboid(*vbs, *vsc, 1, 1, 1);

                // check if input v is equal to the v stored in Problem instance
                REQUIRE(vbs.get() == pbs.getVBSPtr().get());
                REQUIRE(vbs->isEqual(pbs.getVBS()));

                // check if solution is good
                auto err = mgcl_test::calculateError(solutionsc, *vsc);
                auto errNorm = mgcl_test::calculateErrorNorm(1.0 / (double)N, *err);
                auto errMax = mgcl_test::calculateMaxError(*err);

                // solution.dumpToFile("out_solution.txt");
                // (*v).dumpToFile("out_v.txt");

                std::cout
                    << "seq Blockstencil block Jacobi" << std::endl
                    << std::scientific << std::setprecision(17) << "  ||e||_2 = " << errNorm << std::endl
                    << std::scientific << std::setprecision(17) << "  e_max = " << errMax << std::endl;

                CHECK(errNorm < 1e-2);
                CHECK(errMax < 1e-2);
            }
        }
    }

    SECTION("OpenCL")
    {
        auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));

        std::string oclDeviceType = deviceType == CL_DEVICE_TYPE_GPU ? "GPU" : "CPU";

        SECTION("Galerkin_27p")
        {
            mgcl::Problem psc(N, N, N, fsc, vsc);
            psc.setMaxiterVcycles(maxIterVCycles);
            psc.setTol(tol);
            psc.setNu1(nu1);
            psc.setNu2(nu2);
            psc.setOmega(omega);
            psc.setMaxlevel(maxlevel);
            psc.setUseOpencl(true);
            psc.setReadResults(true);
            psc.setDeviceType(deviceType);
            // p.setDeviceName("Quadro");

            psc.setStencilType(mgcl::MGCL_VARYING);
            auto& s = *psc.createStencilValues();
            mgcl_test::fill27pLaplace(s, h, false);

            psc.solve();

            // check if solution is good
            auto err = mgcl_test::calculateError(solutionsc, *vsc);
            auto errNorm = mgcl_test::calculateErrorNorm(1.0 / (double)N, *err);
            auto errMax = mgcl_test::calculateMaxError(*err);

            // solution.dumpToFile("out_solution.txt");
            // (*v).dumpToFile("out_v.txt");

            std::cout
                << "ocl Galerkin 27p" << std::endl
                << std::scientific << std::setprecision(17) << "  ||e||_2 = " << errNorm << std::endl
                << std::scientific << std::setprecision(17) << "  e_max = " << errMax << std::endl;

            CHECK(errNorm < 1e-2);
            CHECK(errMax < 1e-2);
        }

        SECTION("block")
        {
            int blocksize = 1;
            std::shared_ptr<mgcl::CuboidBS> vbs = std::make_shared<mgcl::CuboidBS>(N, N, N, 0, 0, 0, blocksize);
            std::shared_ptr<mgcl::CuboidBS> fbs = std::make_shared<mgcl::CuboidBS>(N, N, N, 0, 0, 0, blocksize);
            // mgcl::Blockstencil bs(Nblockstencil, Nblockstencil, Nblockstencil, 3, blocksize, 0, 0, 0);
            mgcl::FixedStencil fs(3);

            mgcl_test::fill27pLaplace(fs, h, false);

            // bs_inv.dumpToFile("bs_inv.txt");
            // bs.dumpToFile("bs.txt");

            // fill v with values of v1 and v2, vice versa for f
            mgcl_test::copyCuboidToCuboidBS(*vsc, *vbs, 1, 1, 1);
            mgcl_test::copyCuboidToCuboidBS(*fsc, *fbs, 1, 1, 1);

            mgcl::Problem pbs(N, N, N, fbs, vbs);
            // pbs.setIgnoreTol(true);
            pbs.setMaxiterVcycles(maxIterVCycles);
            pbs.setTol(tol);
            pbs.setNu1(nu1);
            pbs.setNu2(nu2);
            pbs.setOmega(omega);
            pbs.setMaxlevel(maxlevel);
            pbs.setStencilType(mgcl::MGCL_BLOCKSTENCIL);
            pbs.setUseOpencl(true);
            pbs.setReadResults(true);
            pbs.setDeviceType(deviceType);
            // pbs.setDeviceName("Quadro");

            auto bs = pbs.createBlockstencil();
            mgcl_test::fill27pLaplace(*bs, h, false);
            // bs->dumpToFile("bs.txt");
            auto r = pbs.getRestrictionBlockstencil();
            auto p = pbs.getProlongationBlockstencil();
            mgcl_test::fill3dFullWeightRestrictionBlockstencil(*r);
            mgcl_test::fill3dBilinearProlongationBlockstencil(*p);

            SECTION("Blockstencil_Jacobiscalar")
            {
                pbs.setSmootherType(mgcl::MGCL_JACOBI_SCALAR);

                pbs.solve();

                mgcl_test::copyCuboidBSToCuboid(*vbs, *vsc, 1, 1, 1);

                // check if solution is good
                auto err = mgcl_test::calculateError(solutionsc, *vsc);
                auto errNorm = mgcl_test::calculateErrorNorm(1.0 / (double)N, *err);
                auto errMax = mgcl_test::calculateMaxError(*err);

                // solution.dumpToFile("out_solution.txt");
                // (*v).dumpToFile("out_v.txt");

                std::cout
                    << "ocl Blockstencil scalar Jacobi" << std::endl
                    << std::scientific << std::setprecision(17) << "  ||e||_2 = " << errNorm << std::endl
                    << std::scientific << std::setprecision(17) << "  e_max = " << errMax << std::endl;

                CHECK(errNorm < 1e-2);
                CHECK(errMax < 1e-2);
            }

            SECTION("Blockstencil_Jacobiblock")
            {
                pbs.setSmootherType(mgcl::MGCL_JACOBI_BLOCK);

                pbs.solve();

                mgcl_test::copyCuboidBSToCuboid(*vbs, *vsc, 1, 1, 1);

                // check if solution is good
                auto err = mgcl_test::calculateError(solutionsc, *vsc);
                auto errNorm = mgcl_test::calculateErrorNorm(1.0 / (double)N, *err);
                auto errMax = mgcl_test::calculateMaxError(*err);

                // solution.dumpToFile("out_solution.txt");
                // (*v).dumpToFile("out_v.txt");

                std::cout
                    << "ocl Blockstencil block Jacobi" << std::endl
                    << std::scientific << std::setprecision(17) << "  ||e||_2 = " << errNorm << std::endl
                    << std::scientific << std::setprecision(17) << "  e_max = " << errMax << std::endl;

                CHECK(errNorm < 1e-2);
                CHECK(errMax < 1e-2);
            }
        }
    }
}

/**
 * @brief Tests if solving works correctly for u = x^4 * (x-1)^4, where u is a vector of independent quantities.
 *
 */
TEST_CASE("solve_bs_periodic_independent_quantities")
{
    int N = 16;
    double h = 1.0 / (double)N;

    // Problem parameters
    double tol = 1e-14;
    int nu1 = 2;
    int nu2 = 2;
    double omega = 0.8;
    int maxIterVCycles = 5;
    int maxlevel = 10;

    auto vsc = std::make_shared<mgcl::Cuboid>(N, N, N);
    auto fsc = std::make_shared<mgcl::Cuboid>(N, N, N);
    auto solutionsc = mgcl::Cuboid(N, N, N);

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
                (*vsc)[i][j][k] = 0;
                solutionsc[i][j][k] = 1000000 * (xs * (xs - 1)) * (xs * (xs - 1)) * (xs * (xs - 1)) * (xs * (xs - 1)) *
                                      (ys * (ys - 1)) * (ys * (ys - 1)) * (ys * (ys - 1)) * (ys * (ys - 1)) *
                                      (zs * (zs - 1)) * (zs * (zs - 1)) * (zs * (zs - 1)) * (zs * (zs - 1));
                (*fsc)[i][j][k] =
                    -1000000 *
                    (12 * xs4 * ys4 * zs4 * xsm1_4 * ysm1_4 * zsm1_2 + 12 * xs4 * ys4 * zs4 * xsm1_4 * ysm1_2 * zsm1_4 +
                     12 * xs4 * ys4 * zs4 * xsm1_2 * ysm1_4 * zsm1_4 + 32 * xs4 * ys4 * zs3 * xsm1_4 * ysm1_4 * zsm1_3 +
                     12 * xs4 * ys4 * zs2 * xsm1_4 * ysm1_4 * zsm1_4 + 32 * xs4 * ys3 * zs4 * xsm1_4 * ysm1_3 * zsm1_4 +
                     12 * xs4 * ys2 * zs4 * xsm1_4 * ysm1_4 * zsm1_4 + 32 * xs3 * ys4 * zs4 * xsm1_3 * ysm1_4 * zsm1_4 +
                     12 * xs2 * ys4 * zs4 * xsm1_4 * ysm1_4 * zsm1_4);
            }

    mgcl::Problem psc(N, N, N, fsc, vsc);
    psc.setMaxiterVcycles(maxIterVCycles);
    psc.setTol(tol);
    psc.setNu1(nu1);
    psc.setNu2(nu2);
    psc.setOmega(omega);
    psc.setMaxlevel(maxlevel);

    SECTION("Sequential")
    {
        SECTION("Galerkin_27p vs blockstencil")
        {
            // *** setup of block problem ***
            int blocksize = 2;
            std::shared_ptr<mgcl::CuboidBS> vbs = std::make_shared<mgcl::CuboidBS>(N, N, N, 0, 0, 0, blocksize);
            std::shared_ptr<mgcl::CuboidBS> fbs = std::make_shared<mgcl::CuboidBS>(N, N, N, 0, 0, 0, blocksize);
            // mgcl::Blockstencil bs(Nblockstencil, Nblockstencil, Nblockstencil, 3, blocksize, 0, 0, 0);
            mgcl::FixedStencil fs(3);

            mgcl_test::fill27pLaplace(fs, h, false);

            // bs_inv.dumpToFile("bs_inv.txt");
            // bs.dumpToFile("bs.txt");

            // copy scalar Cuboid into both CuboidBS vector slots, for both v and f
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    for (int k = 0; k < N; k++)
                    {
                        (*vbs)[0][i][j][k] = (*vsc)[i][j][k];
                        (*fbs)[0][i][j][k] = (*fsc)[i][j][k];
                        (*vbs)[1][i][j][k] = (*vsc)[i][j][k];
                        (*fbs)[1][i][j][k] = (*fsc)[i][j][k];
                    }

            mgcl::Problem pbs(N, N, N, fbs, vbs);
            // pbs.setIgnoreTol(true);
            pbs.setMaxiterVcycles(maxIterVCycles);
            pbs.setTol(tol);
            pbs.setNu1(nu1);
            pbs.setNu2(nu2);
            pbs.setOmega(omega);
            pbs.setMaxlevel(maxlevel);
            pbs.setStencilType(mgcl::MGCL_BLOCKSTENCIL);

            auto bs = pbs.createBlockstencil();
            mgcl_test::fill27pLaplace(*bs, h, false);
            // bs->dumpToFile("bs.txt");
            auto r = pbs.getRestrictionBlockstencil();
            auto p = pbs.getProlongationBlockstencil();
            mgcl_test::fill3dFullWeightRestrictionBlockstencil(*r);
            mgcl_test::fill3dBilinearProlongationBlockstencil(*p);

            // *** setup of scalar problem ***
            psc.setStencilType(mgcl::MGCL_VARYING);
            auto& s = *psc.createStencilValues();

            double h = 1.0 / static_cast<double>(N);
            mgcl_test::fill27pLaplace(s, h, false);

            SECTION("Jacobi scalar")
            {
                std::cout << std::endl
                          << "**** seq Galerkin27p vs blockstencil scalar Jacobi ****" << std::endl;
                pbs.setSmootherType(mgcl::MGCL_JACOBI_SCALAR);
            }

            SECTION("Jacobi block")
            {
                std::cout << std::endl
                          << "**** seq Galerkin27p vs blockstencil block Jacobi ****" << std::endl;
                pbs.setSmootherType(mgcl::MGCL_JACOBI_BLOCK);
            }

            psc.solveSeq();
            pbs.solveSeq();

            // check if input v is equal to the v stored in Problem instance
            REQUIRE(vsc.get() == psc.getVPtr().get());
            REQUIRE(vsc->isEqual(psc.getV()));
            REQUIRE(vbs.get() == pbs.getVBSPtr().get());
            REQUIRE(vbs->isEqual(pbs.getVBS()));

            // check if solution is good
            auto err = mgcl_test::calculateError(solutionsc, *vsc);
            auto errNorm = mgcl_test::calculateErrorNorm(1.0 / (double)N, *err);
            auto errMax = mgcl_test::calculateMaxError(*err);

            // solution.dumpToFile("out_solution.txt");
            // (*v).dumpToFile("out_v.txt");

            std::cout
                << "seq Galerkin 27p" << std::endl
                << std::scientific << std::setprecision(17) << "  ||e||_2 = " << errNorm << std::endl
                << std::scientific << std::setprecision(17) << "  e_max = " << errMax << std::endl;

            CHECK(errNorm < 1e-2);
            CHECK(errMax < 1e-2);

            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    for (int k = 0; k < N; k++)
                    {
                        REQUIRE_THAT((*vbs)[0][i][j][k], Catch::Matchers::WithinAbs((*vsc)[i][j][k], 1e-14));
                        REQUIRE_THAT((*vbs)[1][i][j][k], Catch::Matchers::WithinAbs((*vsc)[i][j][k], 1e-14));
                    }
        }
    }

    SECTION("OpenCL")
    {
        auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));

        std::string oclDeviceType = deviceType == CL_DEVICE_TYPE_GPU ? "GPU" : "CPU";

        SECTION("Galerkin_27p vs blockstencil")
        {
            // *** setup of block problem ***
            int blocksize = 2;
            std::shared_ptr<mgcl::CuboidBS> vbs = std::make_shared<mgcl::CuboidBS>(N, N, N, 0, 0, 0, blocksize);
            std::shared_ptr<mgcl::CuboidBS> fbs = std::make_shared<mgcl::CuboidBS>(N, N, N, 0, 0, 0, blocksize);
            // mgcl::Blockstencil bs(Nblockstencil, Nblockstencil, Nblockstencil, 3, blocksize, 0, 0, 0);
            mgcl::FixedStencil fs(3);

            mgcl_test::fill27pLaplace(fs, h, false);

            // bs_inv.dumpToFile("bs_inv.txt");
            // bs.dumpToFile("bs.txt");

            // copy scalar Cuboid into both CuboidBS vector slots, for both v and f
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    for (int k = 0; k < N; k++)
                    {
                        (*vbs)[0][i][j][k] = (*vsc)[i][j][k];
                        (*fbs)[0][i][j][k] = (*fsc)[i][j][k];
                        (*vbs)[1][i][j][k] = (*vsc)[i][j][k];
                        (*fbs)[1][i][j][k] = (*fsc)[i][j][k];
                    }

            mgcl::Problem pbs(N, N, N, fbs, vbs);
            // pbs.setIgnoreTol(true);
            pbs.setMaxiterVcycles(maxIterVCycles);
            pbs.setTol(tol);
            pbs.setNu1(nu1);
            pbs.setNu2(nu2);
            pbs.setOmega(omega);
            pbs.setMaxlevel(maxlevel);
            pbs.setStencilType(mgcl::MGCL_BLOCKSTENCIL);
            pbs.setUseOpencl(true);
            pbs.setReadResults(true);
            pbs.setDeviceType(deviceType);

            psc.setUseOpencl(true);
            psc.setReadResults(true);
            psc.setDeviceType(deviceType);
            // p.setDeviceName("Quadro");

            auto bs = pbs.createBlockstencil();
            mgcl_test::fill27pLaplace(*bs, h, false);
            // bs->dumpToFile("bs.txt");
            auto r = pbs.getRestrictionBlockstencil();
            auto p = pbs.getProlongationBlockstencil();
            mgcl_test::fill3dFullWeightRestrictionBlockstencil(*r);
            mgcl_test::fill3dBilinearProlongationBlockstencil(*p);

            // *** setup of scalar problem ***
            psc.setStencilType(mgcl::MGCL_VARYING);
            auto& s = *psc.createStencilValues();

            double h = 1.0 / static_cast<double>(N);
            mgcl_test::fill27pLaplace(s, h, false);

            SECTION("Jacobi scalar")
            {
                std::cout << std::endl
                          << "**** ocl Galerkin27p vs blockstencil scalar Jacobi ****" << std::endl;
                pbs.setSmootherType(mgcl::MGCL_JACOBI_SCALAR);
            }

            SECTION("Jacobi block")
            {
                std::cout << std::endl
                          << "**** ocl Galerkin27p vs blockstencil block Jacobi ****" << std::endl;
                pbs.setSmootherType(mgcl::MGCL_JACOBI_BLOCK);
            }

            psc.solve();
            pbs.solve();

            // check if input v is equal to the v stored in Problem instance
            REQUIRE(vsc.get() == psc.getVPtr().get());
            REQUIRE(vsc->isEqual(psc.getV()));
            REQUIRE(vbs.get() == pbs.getVBSPtr().get());
            REQUIRE(vbs->isEqual(pbs.getVBS()));

            // check if solution is good
            auto err = mgcl_test::calculateError(solutionsc, *vsc);
            auto errNorm = mgcl_test::calculateErrorNorm(1.0 / (double)N, *err);
            auto errMax = mgcl_test::calculateMaxError(*err);

            // solution.dumpToFile("out_solution.txt");
            // (*v).dumpToFile("out_v.txt");

            std::cout
                << "ocl Galerkin 27p" << std::endl
                << std::scientific << std::setprecision(17) << "  ||e||_2 = " << errNorm << std::endl
                << std::scientific << std::setprecision(17) << "  e_max = " << errMax << std::endl;

            CHECK(errNorm < 1e-2);
            CHECK(errMax < 1e-2);

            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    for (int k = 0; k < N; k++)
                    {
                        REQUIRE_THAT((*vbs)[0][i][j][k], Catch::Matchers::WithinAbs((*vsc)[i][j][k], 1e-14));
                        REQUIRE_THAT((*vbs)[1][i][j][k], Catch::Matchers::WithinAbs((*vsc)[i][j][k], 1e-14));
                    }
        }
    }
}

// TEST_CASE("solve_dirichlet")
// {
//     int N = 16;
//     double h = 1.0 / (double)N;

//     // Problem parameters
//     double tol = 1e-7;
//     int nu1 = 2;
//     int nu2 = 2;
//     double omega = 0.8;
//     int maxIterVCycles = 20;
//     int maxlevel = 10;

//     mgcl::BC bc = mgcl::BC::DIRICHLET;
//     int ghin = 1;

//     auto v = std::make_shared<mgcl::Cuboid>(N, N, N, ghin, ghin, ghin);
//     auto f = std::make_shared<mgcl::Cuboid>(N, N, N, ghin, ghin, ghin);
//     auto solution = mgcl::Cuboid(N, N, N);

//     // Set boundaries to 0.
//     for (int i = 0; i < ghin; i++)
//         for (int j = 0; j < ghin; j++)
//             for (int k = 0; k < ghin; k++)
//             {
//                 (*v)[i][j][k] = 0.0;
//                 (*f)[i][j][k] = 0.0;

//                 (*v)[i + N][j + N][k + N] = 0.0;
//                 (*f)[i + N][j + N][k + N] = 0.0;
//             }

//     for (int i = ghin; i < N + ghin; i++)
//         for (int j = ghin; j < N + ghin; j++)
//             for (int k = ghin; k < N + ghin; k++)
//             {
//                 double zs = i * h;
//                 double ys = j * h;
//                 double xs = k * h;
//                 double xs2 = xs * xs;
//                 double ys2 = ys * ys;
//                 double zs2 = zs * zs;
//                 double xsm1_2 = (xs - 1) * (xs - 1);
//                 double ysm1_2 = (ys - 1) * (ys - 1);
//                 double zsm1_2 = (zs - 1) * (zs - 1);
//                 double xs3 = xs * xs * xs;
//                 double ys3 = ys * ys * ys;
//                 double zs3 = zs * zs * zs;
//                 double xsm1_3 = (xs - 1) * (xs - 1) * (xs - 1);
//                 double ysm1_3 = (ys - 1) * (ys - 1) * (ys - 1);
//                 double zsm1_3 = (zs - 1) * (zs - 1) * (zs - 1);
//                 double xs4 = xs * xs * xs * xs;
//                 double ys4 = ys * ys * ys * ys;
//                 double zs4 = zs * zs * zs * zs;
//                 double xsm1_4 = (xs - 1) * (xs - 1) * (xs - 1) * (xs - 1);
//                 double ysm1_4 = (ys - 1) * (ys - 1) * (ys - 1) * (ys - 1);
//                 double zsm1_4 = (zs - 1) * (zs - 1) * (zs - 1) * (zs - 1);
//                 (*v)[i][j][k] = 0;
//                 solution[i - ghin][j - ghin][k - ghin] = 1000000 * (xs * (xs - 1)) * (xs * (xs - 1)) * (xs * (xs - 1)) * (xs * (xs - 1)) *
//                                                          (ys * (ys - 1)) * (ys * (ys - 1)) * (ys * (ys - 1)) * (ys * (ys - 1)) *
//                                                          (zs * (zs - 1)) * (zs * (zs - 1)) * (zs * (zs - 1)) * (zs * (zs - 1));
//                 (*f)[i][j][k] =
//                     -1000000 *
//                     (12 * xs4 * ys4 * zs4 * xsm1_4 * ysm1_4 * zsm1_2 + 12 * xs4 * ys4 * zs4 * xsm1_4 * ysm1_2 * zsm1_4 +
//                      12 * xs4 * ys4 * zs4 * xsm1_2 * ysm1_4 * zsm1_4 + 32 * xs4 * ys4 * zs3 * xsm1_4 * ysm1_4 * zsm1_3 +
//                      12 * xs4 * ys4 * zs2 * xsm1_4 * ysm1_4 * zsm1_4 + 32 * xs4 * ys3 * zs4 * xsm1_4 * ysm1_3 * zsm1_4 +
//                      12 * xs4 * ys2 * zs4 * xsm1_4 * ysm1_4 * zsm1_4 + 32 * xs3 * ys4 * zs4 * xsm1_3 * ysm1_4 * zsm1_4 +
//                      12 * xs2 * ys4 * zs4 * xsm1_4 * ysm1_4 * zsm1_4);
//             }

//     mgcl::Problem p(N, N, N, f, v);
//     p.setMaxiterVcycles(maxIterVCycles);
//     p.setTol(tol);
//     p.setNu1(nu1);
//     p.setNu2(nu2);
//     p.setOmega(omega);
//     p.setMaxlevel(maxlevel);
//     p.setBc(bc);
//     p.setGhostsIn(ghin);

//     SECTION("Sequential")
//     {
//         SECTION("Laplace_7p")
//         {
//             p.solveSeq();

//             // check if input v is equal to the v stored in Problem instance
//             REQUIRE(v.get() == p.getVPtr().get());
//             REQUIRE(v->isEqual(p.getV()));

//             // check if solution is good
//             auto err = mgcl_test::calculateError(solution, *v);
//             auto errNorm = mgcl_test::calculateErrorNorm(1.0 / (double)N, *err);
//             auto errMax = mgcl_test::calculateMaxError(*err);

//             // solution.dumpToFile("out_solution.txt");
//             // (*v).dumpToFile("out_v.txt");

//             std::cout
//                 << "seq Laplace 7p" << std::endl
//                 << std::scientific << std::setprecision(17) << "  ||e||_2 = " << errNorm << std::endl
//                 << std::scientific << std::setprecision(17) << "  e_max = " << errMax << std::endl;

//             CHECK(errNorm < 1e-2);
//             CHECK(errMax < 1e-2);
//         }

//         SECTION("Galerkin_7p")
//         {
//             p.setStencilType(mgcl::MGCL_VARYING);
//             auto& s = *p.createStencilValues();
//             double h2inv = N * N; // h = 1/N -> 1/h = N

//             double h = 1.0 / static_cast<double>(N);
//             mgcl_test::fill7pLaplace(s, h, false);

//             p.solveSeq();

//             // check if input v is equal to the v stored in Problem instance
//             REQUIRE(v.get() == p.getVPtr().get());
//             REQUIRE(v->isEqual(p.getV()));

//             // check if solution is good
//             auto err = mgcl_test::calculateError(solution, *v);
//             auto errNorm = mgcl_test::calculateErrorNorm(1.0 / (double)N, *err);
//             auto errMax = mgcl_test::calculateMaxError(*err);

//             // solution.dumpToFile("out_solution.txt");
//             // (*v).dumpToFile("out_v.txt");

//             std::cout
//                 << "seq Galerkin 7p" << std::endl
//                 << std::scientific << std::setprecision(17) << "  ||e||_2 = " << errNorm << std::endl
//                 << std::scientific << std::setprecision(17) << "  e_max = " << errMax << std::endl;

//             CHECK(errNorm < 1e-2);
//             CHECK(errMax < 1e-2);
//         }

//         SECTION("Laplace_27p")
//         {
//             p.setStencilType(mgcl::MGCL_LAPLACE_27POINT);
//             p.solveSeq();

//             // check if input v is equal to the v stored in Problem instance
//             REQUIRE(v.get() == p.getVPtr().get());
//             REQUIRE(v->isEqual(p.getV()));

//             // check if solution is good
//             auto err = mgcl_test::calculateError(solution, *v);
//             auto errNorm = mgcl_test::calculateErrorNorm(1.0 / (double)N, *err);
//             auto errMax = mgcl_test::calculateMaxError(*err);

//             // solution.dumpToFile("out_solution.txt");
//             // (*v).dumpToFile("out_v.txt");

//             std::cout
//                 << "seq Laplace 27p" << std::endl
//                 << std::scientific << std::setprecision(17) << "  ||e||_2 = " << errNorm << std::endl
//                 << std::scientific << std::setprecision(17) << "  e_max = " << errMax << std::endl;

//             CHECK(errNorm < 1e-2);
//             CHECK(errMax < 1e-2);
//         }

//         SECTION("Galerkin_27p")
//         {
//             p.setStencilType(mgcl::MGCL_VARYING);
//             auto& s = *p.createStencilValues();
//             double h2inv = N * N; // h = 1/N -> 1/h = N

//             double h = 1.0 / static_cast<double>(N);
//             mgcl_test::fill27pLaplace(s, h, false);

//             p.solveSeq();

//             // check if input v is equal to the v stored in Problem instance
//             REQUIRE(v.get() == p.getVPtr().get());
//             REQUIRE(v->isEqual(p.getV()));

//             // check if solution is good
//             auto err = mgcl_test::calculateError(solution, *v);
//             auto errNorm = mgcl_test::calculateErrorNorm(1.0 / (double)N, *err);
//             auto errMax = mgcl_test::calculateMaxError(*err);

//             // solution.dumpToFile("out_solution.txt");
//             // (*v).dumpToFile("out_v.txt");

//             std::cout
//                 << "seq Galerkin 27p" << std::endl
//                 << std::scientific << std::setprecision(17) << "  ||e||_2 = " << errNorm << std::endl
//                 << std::scientific << std::setprecision(17) << "  e_max = " << errMax << std::endl;

//             CHECK(errNorm < 1e-2);
//             CHECK(errMax < 1e-2);
//         }
//     }

//     SECTION("OpenCL")
//     {
//         auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));

//         std::string oclDeviceType = deviceType == CL_DEVICE_TYPE_GPU ? "GPU" : "CPU";

//         p.setUseOpencl(true);
//         p.setDeviceType(CL_DEVICE_TYPE_GPU);
//         p.setReadResults(true);
//         p.setDeviceType(deviceType);
//         // p.setDeviceName("Quadro");

//         SECTION("Laplace_7p")
//         {
//             p.solve();

//             // check if solution is good
//             auto err = mgcl_test::calculateError(solution, *v);
//             auto errNorm = mgcl_test::calculateErrorNorm(1.0 / (double)N, *err);
//             auto errMax = mgcl_test::calculateMaxError(*err);

//             std::cout
//                 << "ocl " << oclDeviceType << " Laplace 7p" << std::endl
//                 << std::scientific << "  ||e||_2 = " << errNorm << std::endl
//                 << std::scientific << "  e_max = " << errMax << std::endl;

//             CHECK(errNorm < 1e-2);
//             CHECK(errMax < 1e-2);
//         }

//         SECTION("Galerkin_7p")
//         {
//             p.setStencilType(mgcl::MGCL_VARYING);
//             auto& s = *p.createStencilValues();
//             double h2inv = N * N; // h = 1/N -> 1/h = N

//             double h = 1.0 / static_cast<double>(N);
//             mgcl_test::fill7pLaplace(s, h, false);

//             p.solve();

//             // check if solution is good
//             auto err = mgcl_test::calculateError(solution, *v);
//             auto errNorm = mgcl_test::calculateErrorNorm(1.0 / (double)N, *err);
//             auto errMax = mgcl_test::calculateMaxError(*err);

//             // solution.dumpToFile("out_solution.txt");
//             // (*v).dumpToFile("out_v.txt");

//             std::cout
//                 << "ocl " << oclDeviceType << " Galerkin 7p" << std::endl
//                 << std::scientific << std::setprecision(17) << "  ||e||_2 = " << errNorm << std::endl
//                 << std::scientific << std::setprecision(17) << "  e_max = " << errMax << std::endl;

//             CHECK(errNorm < 1e-2);
//             CHECK(errMax < 1e-2);
//         }

//         SECTION("Laplace_27p")
//         {
//             p.setStencilType(mgcl::MGCL_LAPLACE_27POINT);
//             p.solve();

//             // check if solution is good
//             auto err = mgcl_test::calculateError(solution, *v);
//             auto errNorm = mgcl_test::calculateErrorNorm(1.0 / (double)N, *err);
//             auto errMax = mgcl_test::calculateMaxError(*err);

//             std::cout
//                 << "ocl " << oclDeviceType << " Laplace 27p" << std::endl
//                 << std::scientific << "  ||e||_2 = " << errNorm << std::endl
//                 << std::scientific << "  e_max = " << errMax << std::endl;

//             CHECK(errNorm < 1e-2);
//             CHECK(errMax < 1e-2);
//         }

//         SECTION("Galerkin_27p")
//         {
//             p.setStencilType(mgcl::MGCL_VARYING);
//             auto& s = *p.createStencilValues();
//             double h2inv = N * N; // h = 1/N -> 1/h = N

//             double h = 1.0 / static_cast<double>(N);
//             mgcl_test::fill27pLaplace(s, h, false);

//             p.solve();

//             // check if solution is good
//             auto err = mgcl_test::calculateError(solution, *v);
//             auto errNorm = mgcl_test::calculateErrorNorm(1.0 / (double)N, *err);
//             auto errMax = mgcl_test::calculateMaxError(*err);

//             // solution.dumpToFile("out_solution.txt");
//             // (*v).dumpToFile("out_v.txt");

//             std::cout
//                 << "ocl " << oclDeviceType << " Galerkin 27p" << std::endl
//                 << std::scientific << std::setprecision(17) << "  ||e||_2 = " << errNorm << std::endl
//                 << std::scientific << std::setprecision(17) << "  e_max = " << errMax << std::endl;

//             CHECK(errNorm < 1e-2);
//             CHECK(errMax < 1e-2);
//         }
//     }

// }

// /**
//  * @brief Tests if solving works correctly for u = x^4 * (x-1)^4.
//  * Tests different amounts of jacobi iterations without ghost update in-between.
//  *
//  */
// TEST_CASE("Problem_solving:_periodic_4th_order_Jacobi_iters")
// {
//     int N = 16;
//     double h = 1.0 / (double)N;

//     // Problem parameters
//     double tol = 1e-7;
//     int nu1 = 2;
//     int nu2 = 2;
//     double omega = 0.8;
//     // int maxIterVCycles = 1;
//     int maxIterVCycles = 20;
//     int maxlevel = 10;

//     int jacobiIters = GENERATE(1, 2, 3);
//     int gh = jacobiIters;
//     CAPTURE(jacobiIters);
//     std::cerr << "Testing with jacobiIters = " << jacobiIters << std::endl;

//     auto v = std::make_shared<mgcl::Cuboid>(N, N, N);
//     auto f = std::make_shared<mgcl::Cuboid>(N, N, N);
//     auto solution = mgcl::Cuboid(N, N, N);

//     for (int i = 0; i < N; i++)
//         for (int j = 0; j < N; j++)
//             for (int k = 0; k < N; k++)
//             {
//                 double zs = i * h;
//                 double ys = j * h;
//                 double xs = k * h;
//                 double xs2 = xs * xs;
//                 double ys2 = ys * ys;
//                 double zs2 = zs * zs;
//                 double xsm1_2 = (xs - 1) * (xs - 1);
//                 double ysm1_2 = (ys - 1) * (ys - 1);
//                 double zsm1_2 = (zs - 1) * (zs - 1);
//                 double xs3 = xs * xs * xs;
//                 double ys3 = ys * ys * ys;
//                 double zs3 = zs * zs * zs;
//                 double xsm1_3 = (xs - 1) * (xs - 1) * (xs - 1);
//                 double ysm1_3 = (ys - 1) * (ys - 1) * (ys - 1);
//                 double zsm1_3 = (zs - 1) * (zs - 1) * (zs - 1);
//                 double xs4 = xs * xs * xs * xs;
//                 double ys4 = ys * ys * ys * ys;
//                 double zs4 = zs * zs * zs * zs;
//                 double xsm1_4 = (xs - 1) * (xs - 1) * (xs - 1) * (xs - 1);
//                 double ysm1_4 = (ys - 1) * (ys - 1) * (ys - 1) * (ys - 1);
//                 double zsm1_4 = (zs - 1) * (zs - 1) * (zs - 1) * (zs - 1);
//                 (*v)[i][j][k] = 0;
//                 solution[i][j][k] = 1000000 * (xs * (xs - 1)) * (xs * (xs - 1)) * (xs * (xs - 1)) * (xs * (xs - 1)) *
//                                     (ys * (ys - 1)) * (ys * (ys - 1)) * (ys * (ys - 1)) * (ys * (ys - 1)) *
//                                     (zs * (zs - 1)) * (zs * (zs - 1)) * (zs * (zs - 1)) * (zs * (zs - 1));
//                 (*f)[i][j][k] =
//                     -1000000 *
//                     (12 * xs4 * ys4 * zs4 * xsm1_4 * ysm1_4 * zsm1_2 + 12 * xs4 * ys4 * zs4 * xsm1_4 * ysm1_2 * zsm1_4 +
//                      12 * xs4 * ys4 * zs4 * xsm1_2 * ysm1_4 * zsm1_4 + 32 * xs4 * ys4 * zs3 * xsm1_4 * ysm1_4 * zsm1_3 +
//                      12 * xs4 * ys4 * zs2 * xsm1_4 * ysm1_4 * zsm1_4 + 32 * xs4 * ys3 * zs4 * xsm1_4 * ysm1_3 * zsm1_4 +
//                      12 * xs4 * ys2 * zs4 * xsm1_4 * ysm1_4 * zsm1_4 + 32 * xs3 * ys4 * zs4 * xsm1_3 * ysm1_4 * zsm1_4 +
//                      12 * xs2 * ys4 * zs4 * xsm1_4 * ysm1_4 * zsm1_4);
//             }

//     mgcl::Problem p(N, N, N, f, v);
//     p.setMaxiterVcycles(maxIterVCycles);
//     p.setTol(tol);
//     p.setNu1(nu1);
//     p.setNu2(nu2);
//     p.setOmega(omega);
//     p.setMaxlevel(maxlevel);
//     p.setGhosts(gh);
//     p.setJacobiIterationsPerKernel(jacobiIters);

//     SECTION("Sequential")
//     {
//         SECTION("Laplace")
//         {
//             p.solveSeq();

//             // check if input v is equal to the v stored in Problem instance
//             REQUIRE(v.get() == p.getVPtr().get());
//             REQUIRE(v->isEqual(p.getV()));

//             // check if solution is good
//             auto err = mgcl_test::calculateError(solution, *v);
//             auto errNorm = mgcl_test::calculateErrorNorm(1.0 / (double)N, *err);
//             auto errMax = mgcl_test::calculateMaxError(*err);

//             // solution.dumpToFile("out_solution.txt");
//             // (*v).dumpToFile("out_v.txt");

//             std::cout
//                 << "seq Laplace" << std::endl
//                 << std::scientific << std::setprecision(17) << "  ||e||_2 = " << errNorm << std::endl
//                 << std::scientific << std::setprecision(17) << "  e_max = " << errMax << std::endl;

//             CHECK(errNorm < 1e-2);
//             CHECK(errMax < 1e-2);

//             // check if error is equal to old mgcl implementation (problem params must match)
//             if (p.getMaxiterVcycles() == 10 && N == 32 && p.getTol() == 1e-14 &&
//                 p.getNu1() == 2 && p.getNu2() == 2 && p.getOmega() == 0.8 &&
//                 p.getStencilType() == mgcl::MGCL_LAPLACE_7POINT)
//             {
//                 CHECK(fabs(errNorm - 3.93115528889639940e-03) < 1e-14);
//                 CHECK(fabs(errMax - 3.95723982871564600e-03) < 1e-14);
//             }
//         }

//         SECTION("Galerkin")
//         {
//             p.setStencilType(mgcl::MGCL_VARYING);
//             auto& s = *p.createStencilValues();
//             double h2inv = N * N; // h = 1/N -> 1/h = N

//             double h = 1.0 / static_cast<double>(N);
//             mgcl_test::fill7pLaplace(s, h, false);

//             p.solveSeq();

//             // check if input v is equal to the v stored in Problem instance
//             REQUIRE(v.get() == p.getVPtr().get());
//             REQUIRE(v->isEqual(p.getV()));

//             // check if solution is good
//             auto err = mgcl_test::calculateError(solution, *v);
//             auto errNorm = mgcl_test::calculateErrorNorm(1.0 / (double)N, *err);
//             auto errMax = mgcl_test::calculateMaxError(*err);

//             // solution.dumpToFile("out_solution.txt");
//             // (*v).dumpToFile("out_v.txt");

//             std::cout
//                 << "seq Galerkin" << std::endl
//                 << std::scientific << std::setprecision(17) << "  ||e||_2 = " << errNorm << std::endl
//                 << std::scientific << std::setprecision(17) << "  e_max = " << errMax << std::endl;

//             CHECK(errNorm < 1e-2);
//             CHECK(errMax < 1e-2);
//         }
//     }

//     SECTION("OpenCL")
//     {
//         auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));

//         std::string oclDeviceType = deviceType == CL_DEVICE_TYPE_GPU ? "GPU" : "CPU";

//         p.setUseOpencl(true);
//         p.setDeviceType(CL_DEVICE_TYPE_GPU);
//         p.setReadResults(true);
//         p.setDeviceType(deviceType);
//         // p.setDeviceName("Quadro");

//         SECTION("Laplace")
//         {
//             p.solve();

//             // check if solution is good
//             auto err = mgcl_test::calculateError(solution, *v);
//             auto errNorm = mgcl_test::calculateErrorNorm(1.0 / (double)N, *err);
//             auto errMax = mgcl_test::calculateMaxError(*err);

//             std::cout
//                 << "ocl " << oclDeviceType << " Laplace" << std::endl
//                 << std::scientific << "  ||e||_2 = " << errNorm << std::endl
//                 << std::scientific << "  e_max = " << errMax << std::endl;

//             CHECK(errNorm < 1e-2);
//             CHECK(errMax < 1e-2);

//             // check if error is equal to old mgcl implementation (problem params must match)
//             if (p.getMaxiterVcycles() == 10 && N == 32 && p.getTol() == 1e-14 &&
//                 p.getNu1() == 2 && p.getNu2() == 2 && p.getOmega() == 0.8 &&
//                 p.getDeviceName() == "Quadro" && p.getDeviceType() == CL_DEVICE_TYPE_GPU)
//             {
//                 CHECK(fabs(errNorm - 3.93115528889612358e-03) < 1e-14);
//                 CHECK(fabs(errMax - 3.95723982871536324e-03) < 1e-14);
//             }
//         }

//         SECTION("Galerkin")
//         {
//             p.setStencilType(mgcl::MGCL_VARYING);
//             auto& s = *p.createStencilValues();
//             double h2inv = N * N; // h = 1/N -> 1/h = N

//             double h = 1.0 / static_cast<double>(N);
//             mgcl_test::fill7pLaplace(s, h, false);

//             p.solve();

//             // check if solution is good
//             auto err = mgcl_test::calculateError(solution, *v);
//             auto errNorm = mgcl_test::calculateErrorNorm(1.0 / (double)N, *err);
//             auto errMax = mgcl_test::calculateMaxError(*err);

//             // solution.dumpToFile("out_solution.txt");
//             // (*v).dumpToFile("out_v.txt");

//             std::cout
//                 << "ocl " << oclDeviceType << " Galerkin" << std::endl
//                 << std::scientific << std::setprecision(17) << "  ||e||_2 = " << errNorm << std::endl
//                 << std::scientific << std::setprecision(17) << "  e_max = " << errMax << std::endl;

//             CHECK(errNorm < 1e-2);
//             CHECK(errMax < 1e-2);
//         }
//     }
// }

// /**
//  * @brief Tests if tolerance is ignored if ignoreTol is true, i.e. maximum amount of v-cycle iterations
//  * is done although the tolerance is low.
//  *
//  */
// TEST_CASE("Problem_ignore_tol")
// {
//     int N = 16;
//     double h = 1.0 / (double)N;

//     // Problem parameters
//     double tol = 1e-1; // will be reached really quick
//     int nu1 = 2;
//     int nu2 = 2;
//     double omega = 0.8;
//     int maxIterVCycles = 10;
//     int maxlevel = 10;

//     auto v = std::make_shared<mgcl::Cuboid>(N, N, N);
//     auto f = std::make_shared<mgcl::Cuboid>(N, N, N);
//     auto solution = mgcl::Cuboid(N, N, N);

//     for (int i = 0; i < N; i++)
//         for (int j = 0; j < N; j++)
//             for (int k = 0; k < N; k++)
//             {
//                 double zs = i * h;
//                 double ys = j * h;
//                 double xs = k * h;
//                 double xs2 = xs * xs;
//                 double ys2 = ys * ys;
//                 double zs2 = zs * zs;
//                 double xsm1_2 = (xs - 1) * (xs - 1);
//                 double ysm1_2 = (ys - 1) * (ys - 1);
//                 double zsm1_2 = (zs - 1) * (zs - 1);
//                 double xs3 = xs * xs * xs;
//                 double ys3 = ys * ys * ys;
//                 double zs3 = zs * zs * zs;
//                 double xsm1_3 = (xs - 1) * (xs - 1) * (xs - 1);
//                 double ysm1_3 = (ys - 1) * (ys - 1) * (ys - 1);
//                 double zsm1_3 = (zs - 1) * (zs - 1) * (zs - 1);
//                 double xs4 = xs * xs * xs * xs;
//                 double ys4 = ys * ys * ys * ys;
//                 double zs4 = zs * zs * zs * zs;
//                 double xsm1_4 = (xs - 1) * (xs - 1) * (xs - 1) * (xs - 1);
//                 double ysm1_4 = (ys - 1) * (ys - 1) * (ys - 1) * (ys - 1);
//                 double zsm1_4 = (zs - 1) * (zs - 1) * (zs - 1) * (zs - 1);
//                 (*v)[i][j][k] = 0;
//                 solution[i][j][k] = 1000000 * (xs * (xs - 1)) * (xs * (xs - 1)) * (xs * (xs - 1)) * (xs * (xs - 1)) *
//                                     (ys * (ys - 1)) * (ys * (ys - 1)) * (ys * (ys - 1)) * (ys * (ys - 1)) *
//                                     (zs * (zs - 1)) * (zs * (zs - 1)) * (zs * (zs - 1)) * (zs * (zs - 1));
//                 (*f)[i][j][k] =
//                     -1000000 *
//                     (12 * xs4 * ys4 * zs4 * xsm1_4 * ysm1_4 * zsm1_2 + 12 * xs4 * ys4 * zs4 * xsm1_4 * ysm1_2 * zsm1_4 +
//                      12 * xs4 * ys4 * zs4 * xsm1_2 * ysm1_4 * zsm1_4 + 32 * xs4 * ys4 * zs3 * xsm1_4 * ysm1_4 * zsm1_3 +
//                      12 * xs4 * ys4 * zs2 * xsm1_4 * ysm1_4 * zsm1_4 + 32 * xs4 * ys3 * zs4 * xsm1_4 * ysm1_3 * zsm1_4 +
//                      12 * xs4 * ys2 * zs4 * xsm1_4 * ysm1_4 * zsm1_4 + 32 * xs3 * ys4 * zs4 * xsm1_3 * ysm1_4 * zsm1_4 +
//                      12 * xs2 * ys4 * zs4 * xsm1_4 * ysm1_4 * zsm1_4);
//             }

//     mgcl::Problem p(N, N, N, f, v);
//     p.setMaxiterVcycles(maxIterVCycles);
//     p.setTol(tol);
//     p.setNu1(nu1);
//     p.setNu2(nu2);
//     p.setOmega(omega);
//     p.setMaxlevel(maxlevel);

//     SECTION("Sequential")
//     {
//         SECTION("ignoreTolFalse")
//         {
//             p.setIgnoreTol(false);
//             p.solveSeq();

//             REQUIRE(p.getElapsedIterations() < maxIterVCycles);
//         }

//         SECTION("ignoreTolTrue")
//         {
//             p.setIgnoreTol(true);
//             p.solveSeq();

//             REQUIRE(p.getElapsedIterations() == maxIterVCycles);
//         }
//     }

//     SECTION("OpenCL")
//     {
//         auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));

//         std::string oclDeviceType = deviceType == CL_DEVICE_TYPE_GPU ? "GPU" : "CPU";

//         p.setUseOpencl(true);
//         p.setDeviceType(CL_DEVICE_TYPE_GPU);
//         p.setDeviceType(deviceType);

//         SECTION("ignoreTolFalse")
//         {
//             p.setIgnoreTol(false);
//             p.solve();

//             REQUIRE(p.getElapsedIterations() < maxIterVCycles);
//         }

//         SECTION("ignoreTolTrue")
//         {
//             p.setIgnoreTol(true);
//             p.solve();

//             REQUIRE(p.getElapsedIterations() == maxIterVCycles);
//         }
//     }
// }

// /**
//  * @brief Checks whether using FixedStencil yields the same solution as using VaryingStencil with values of FixedStencil
//  *
//  */
// TEST_CASE("solve_fixed_vs_varying_stencil")
// {
//     int N = 16;
//     double h = 1.0 / (double)N;

//     // Problem parameters
//     double tol = 1e-1; // will be reached really quick
//     int nu1 = 2;
//     int nu2 = 2;
//     double omega = 0.8;
//     int maxIterVCycles = 10;
//     int maxlevel = 10;

//     auto v_fixed = std::make_shared<mgcl::Cuboid>(N, N, N);
//     auto v_varying = std::make_shared<mgcl::Cuboid>(N, N, N);
//     auto f = std::make_shared<mgcl::Cuboid>(N, N, N);
//     auto solution = mgcl::Cuboid(N, N, N);

//     for (int i = 0; i < N; i++)
//         for (int j = 0; j < N; j++)
//             for (int k = 0; k < N; k++)
//             {
//                 double zs = i * h;
//                 double ys = j * h;
//                 double xs = k * h;
//                 double xs2 = xs * xs;
//                 double ys2 = ys * ys;
//                 double zs2 = zs * zs;
//                 double xsm1_2 = (xs - 1) * (xs - 1);
//                 double ysm1_2 = (ys - 1) * (ys - 1);
//                 double zsm1_2 = (zs - 1) * (zs - 1);
//                 double xs3 = xs * xs * xs;
//                 double ys3 = ys * ys * ys;
//                 double zs3 = zs * zs * zs;
//                 double xsm1_3 = (xs - 1) * (xs - 1) * (xs - 1);
//                 double ysm1_3 = (ys - 1) * (ys - 1) * (ys - 1);
//                 double zsm1_3 = (zs - 1) * (zs - 1) * (zs - 1);
//                 double xs4 = xs * xs * xs * xs;
//                 double ys4 = ys * ys * ys * ys;
//                 double zs4 = zs * zs * zs * zs;
//                 double xsm1_4 = (xs - 1) * (xs - 1) * (xs - 1) * (xs - 1);
//                 double ysm1_4 = (ys - 1) * (ys - 1) * (ys - 1) * (ys - 1);
//                 double zsm1_4 = (zs - 1) * (zs - 1) * (zs - 1) * (zs - 1);
//                 (*v_fixed)[i][j][k] = 0;
//                 (*v_varying)[i][j][k] = 0;
//                 solution[i][j][k] = 1000000 * (xs * (xs - 1)) * (xs * (xs - 1)) * (xs * (xs - 1)) * (xs * (xs - 1)) *
//                                     (ys * (ys - 1)) * (ys * (ys - 1)) * (ys * (ys - 1)) * (ys * (ys - 1)) *
//                                     (zs * (zs - 1)) * (zs * (zs - 1)) * (zs * (zs - 1)) * (zs * (zs - 1));
//                 (*f)[i][j][k] =
//                     -1000000 *
//                     (12 * xs4 * ys4 * zs4 * xsm1_4 * ysm1_4 * zsm1_2 + 12 * xs4 * ys4 * zs4 * xsm1_4 * ysm1_2 * zsm1_4 +
//                      12 * xs4 * ys4 * zs4 * xsm1_2 * ysm1_4 * zsm1_4 + 32 * xs4 * ys4 * zs3 * xsm1_4 * ysm1_4 * zsm1_3 +
//                      12 * xs4 * ys4 * zs2 * xsm1_4 * ysm1_4 * zsm1_4 + 32 * xs4 * ys3 * zs4 * xsm1_4 * ysm1_3 * zsm1_4 +
//                      12 * xs4 * ys2 * zs4 * xsm1_4 * ysm1_4 * zsm1_4 + 32 * xs3 * ys4 * zs4 * xsm1_3 * ysm1_4 * zsm1_4 +
//                      12 * xs2 * ys4 * zs4 * xsm1_4 * ysm1_4 * zsm1_4);
//             }

//     mgcl::Problem p_fixed(N, N, N, f, v_fixed);
//     p_fixed.setMaxiterVcycles(maxIterVCycles);
//     p_fixed.setTol(tol);
//     p_fixed.setNu1(nu1);
//     p_fixed.setNu2(nu2);
//     p_fixed.setOmega(omega);
//     p_fixed.setMaxlevel(maxlevel);

//     p_fixed.setStencilType(mgcl::MGCL_FIXED);
//     auto& fixedStencil = p_fixed.createFixedStencil();
//     fixedStencil->fillRandom();
//     (*fixedStencil)[1][1][1] = 1.0; // make sure the stencil does not produce nan

//     mgcl::Problem p_varying(N, N, N, f, v_varying);
//     p_varying.setMaxiterVcycles(maxIterVCycles);
//     p_varying.setTol(tol);
//     p_varying.setNu1(nu1);
//     p_varying.setNu2(nu2);
//     p_varying.setOmega(omega);
//     p_varying.setMaxlevel(maxlevel);

//     p_varying.setStencilType(mgcl::MGCL_VARYING);
//     auto& sv = p_varying.createStencilValues();
//     // copy from fixed into varying
//     // clang-format off
//     for (int i = 0; i < sv->getMgh(); i++)
//     for (int j = 0; j < sv->getNgh(); j++)
//     for (int k = 0; k < sv->getOgh(); k++)
//         for (int ii = 0; ii < 3; ii++)
//         for (int jj = 0; jj < 3; jj++)
//         for (int kk = 0; kk < 3; kk++)
//         {
//             (*sv)[ii][jj][kk][i][j][k] = (*fixedStencil)[ii][jj][kk];
//         }
//     // clang-format on

//     SECTION("Sequential")
//     {
//         p_fixed.solve();
//         p_varying.solve();

//         REQUIRE(v_fixed->isEqual(*v_varying));
//     }

//     SECTION("OpenCL")
//     {
//         auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));

//         std::string oclDeviceType = deviceType == CL_DEVICE_TYPE_GPU ? "GPU" : "CPU";

//         p_fixed.setUseOpencl(true);
//         p_fixed.setDeviceType(CL_DEVICE_TYPE_GPU);
//         p_fixed.setDeviceType(deviceType);
//         p_varying.setUseOpencl(true);
//         p_varying.setDeviceType(CL_DEVICE_TYPE_GPU);
//         p_varying.setDeviceType(deviceType);

//         p_fixed.solve();
//         p_varying.solve();

//         REQUIRE(v_fixed->isEqual(*v_varying));
//     }
// }
