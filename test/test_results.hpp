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
        std::unique_ptr<mgcl::VaryingStencil5x5x5> varyingStencil4x4x4();
        mgcl_test::Matrix2d matrix2d64x64();
    }
}

#endif // TEST_RESULTS_HPP
