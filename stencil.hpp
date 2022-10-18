#ifndef MGCL_STENCIL_HPP
#define MGCL_STENCIL_HPP

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
    class FixedStencil : public Cuboid
    {
    public:
        FixedStencil() : Cuboid(3, 3, 3, 0, 0, 0) {}
    };

    /**
     * @brief Class for NxNxN varying stencils, i.e. stencil can differ for each grid point.
     * Choose N = 3 to include only direct neighbors, N = 5 to include 2 nearest neighbors, etc.
     * If two stencils of size NxNxN and NBxNBxNB get multiplied with each other, the resulting stencil has size
     * (max(N, NB)+2)^3.
     *
     */
    template <int N>
    class VaryingStencil : public Hypercube6d
    {
    public:
        VaryingStencil(int m, int n, int o, int ghosts_m, int ghosts_n, int ghosts_o)
            : Hypercube6d(m, n, o, N, N, N, ghosts_m, ghosts_n, ghosts_o, 0, 0, 0) {}

        template <int NB>
        std::unique_ptr<VaryingStencil<(N > NB ? N : NB) + 2>> multiply(VaryingStencil<NB> &b) const;
        template <int NB>
        std::unique_ptr<VaryingStencil<(N > NB ? N : NB) + 2>> operator*(VaryingStencil<NB> &b) const;
    };

    typedef VaryingStencil<3> VaryingStencil3x3x3;
    typedef VaryingStencil<5> VaryingStencil5x5x5;
    typedef VaryingStencil<7> VaryingStencil7x7x7;

    enum StencilPos
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
}

#endif // MGCL_STENCIL_HPP
