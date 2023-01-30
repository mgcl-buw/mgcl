/**
 * @file test_results.hpp
 * @author Simon Hoffmann (shoffmann@uni-wuppertal.de)
 * @brief This file contains hard coded test results, calculated by an old implementation, with matlab or by hand.
 * @version 0.1
 * @date 2022-10-11
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef TEST_RESULTS_HPP
#define TEST_RESULTS_HPP

#include <memory>

#include "../stencil.hpp"
#include "matrix2d.hpp"

namespace mgcl_test
{
    namespace Matrix2d_fromVaryingStencil
    {
        std::unique_ptr<mgcl::VaryingStencil5x5x5> varyingStencilLaplace4x4x4();
        mgcl_test::Matrix2d matrix2dLaplace64x64();
        std::unique_ptr<mgcl::VaryingStencil3x3x3> varyingStencil2x3x4RandomPeriodic();
        mgcl_test::Matrix2d matrix2d24x24RandomPeriodic();
        mgcl_test::Matrix2d matrix2d24x24RandomNotPeriodic();
    }

    namespace test_residual
    {
        std::unique_ptr<mgcl::Cuboid> inputF16();
        std::unique_ptr<mgcl::Cuboid> inputV16();
        std::unique_ptr<mgcl::Cuboid> outputR16();
    }

    namespace test_restriction
    {
        std::unique_ptr<mgcl::Cuboid> inputFine16();
        std::unique_ptr<mgcl::Cuboid> inputCoarse8();
        std::unique_ptr<mgcl::Cuboid> outputFine16();
        std::unique_ptr<mgcl::Cuboid> outputCoarse8();
    }

    namespace test_jacobi
    {
        std::shared_ptr<mgcl::Cuboid> inputV16();
        std::shared_ptr<mgcl::Cuboid> inputR16();
        std::shared_ptr<mgcl::Cuboid> inputF16();
        std::shared_ptr<mgcl::Cuboid> outputV16();
        std::shared_ptr<mgcl::Cuboid> outputR16();
    }
}

#endif // TEST_RESULTS_HPP
