#pragma once

#include <memory>
#include <stdexcept>
#include <vector>

#include "cuboid.hpp"
#include "hypercube.hpp"
#include "mgcl.hpp"

namespace mgcl
{
    /**
     * @brief Fixed 3x3x3 stencil (same stencil entries for each grid point).
     */
    class FixedStencil : Cuboid
    {
    public:
        FixedStencil() : Cuboid(3, 3, 3, 0, 0, 0) {}
    };

    /**
     * @brief Class for 3x3x3 varying stencils, i.e. stencil can differ for each grid point.
     *
     */
    class VaryingStencil : Hypercube6d
    {
    public:
        VaryingStencil(int m, int n, int o, int ghosts_m, int ghosts_n, int ghosts_o)
            : Hypercube6d(m, n, o, 3, 3, 3, ghosts_m, ghosts_n, ghosts_o, 0, 0, 0) {}

        std::unique_ptr<VaryingStencil> multiply(VaryingStencil &b);

        enum Pos
        {
            // clang-format off
            // 7p
            SELF,   // [i][j][k]
            FRONT,  // [i][j][k - 1]
            BACK,   // [i][j][k + 1]
            TOP,    // [i][j - 1][k]
            BOTTOM, // [i][j + 1][k]
            LEFT,   // [i - 1][j][k]
            RIGHT,  // [i + 1][j][k]

            // 19p
            FRONT_TOP,    // [i][j - 1][k - 1]
            BACK_TOP,     // [i][j - 1][k + 1]
            FRONT_BOTTOM, // [i][j + 1][k - 1]
            BACK_BOTTOM,  // [i][j + 1][k + 1]
            FRONT_LEFT,   // [i - 1][j][k - 1]
            BACK_LEFT,    // [i - 1][j][k + 1]
            FRONT_RIGHT,  // [i + 1][j][k - 1]
            BACK_RIGHT,   // [i + 1][j][k + 1]
            LEFT_TOP,     // [i - 1][j - 1][k]
            LEFT_BOTTOM,  // [i - 1][j + 1][k]
            RIGHT_TOP,    // [i + 1][j - 1][k]
            RIGHT_BOTTOM, // [i + 1][j + 1][k]

            // 27p
            FRONT_TOP_LEFT,     // [i - 1][j - 1][k - 1]
            BACK_TOP_LEFT,      // [i - 1][j - 1][k + 1]
            FRONT_BOTTOM_LEFT,  // [i - 1][j + 1][k - 1]
            BACK_BOTTOM_LEFT,   // [i - 1][j + 1][k + 1]
            FRONT_TOP_RIGHT,    // [i + 1][j - 1][k - 1]
            BACK_TOP_RIGHT,     // [i + 1][j - 1][k + 1]
            FRONT_BOTTOM_RIGHT, // [i + 1][j + 1][k - 1]
            BACK_BOTTOM_RIGHT   // [i + 1][j + 1][k + 1]

            // clang-format on
        };
    };
}