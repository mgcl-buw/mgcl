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
            : Hypercube6d(m, n, o, N, N, N, ghosts_m, ghosts_n, ghosts_o, 0, 0, 0)
        {
            static_assert(N % 2 == 1, "VaryingStencil<N> is only defined for odd N!");
        }
        VaryingStencil(VaryingStencil &) = delete;
        VaryingStencil &operator=(const VaryingStencil &) = delete;
        VaryingStencil(VaryingStencil &&o) : Hypercube6d(std::move(o)) {}
        VaryingStencil &operator=(const VaryingStencil &&o)
        {
            Hypercube6d::operator=(std::move(o));
            return *this;
        }

        /**
         * @brief Multiplies this stencil with another one. Dimensions must match. Resulting stencil is two cells wider
         * in each direction.
         *
         * @tparam NA Width of this stencil
         * @tparam NB Width of the other stencil
         * @param b Other stencil
         * @throws If dimensions do not match.
         * @return std::unique_ptr<VaryingStencil<N + 2>> Resulting stencil (two cells wider).
         */
        template <int NB>
        VaryingStencil<(N > NB ? N : NB) + 2> multiply(VaryingStencil<NB> &b) const
        {
            if (dim1gh != b.getDim1gh() ||
                dim2gh != b.getDim2gh() ||
                dim3gh != b.getDim3gh())
                throw "Stencils map to different amount of grid cells!";

            int m = dim1;
            int n = dim2;
            int o = dim3;
            int ghm = ghostsDim1;
            int ghn = ghostsDim2;
            int gho = ghostsDim3;
            int N2 = N >> 1;
            int NB2 = NB >> 1;

            auto c = VaryingStencil<(N > NB ? N : NB) + 2>(m, n, o, ghm, ghn, gho);

            // clang-format off
            for (int x = 0; x < m; x++)
            for (int y = 0; y < n; y++)
            for (int z = 0; z < o; z++)
                for (int a_i = 0; a_i < N; a_i++)
                for (int a_j = 0; a_j < N; a_j++)
                for (int a_k = 0; a_k < N; a_k++)
                    for (int b_i = 0; b_i < NB; b_i++)
                    for (int b_j = 0; b_j < NB; b_j++)
                    for (int b_k = 0; b_k < NB; b_k++)
                        if ((x + a_i) >= N2 && (x + a_i) <= m + N2 - 1 &&
                            (y + a_j) >= N2 && (y + a_j) <= n + N2 - 1 &&
                            (z + a_k) >= N2 && (z + a_k) <= o + N2 - 1 &&
                            (x + a_i - N2 + b_i) >= NB2 && (x + a_i - N2 + b_i) <= m + NB2 - 1 &&
                            (y + a_j - N2 + b_j) >= NB2 && (y + a_j - N2 + b_j) <= n + NB2 - 1 &&
                            (z + a_k - N2 + b_k) >= NB2 && (z + a_k - N2 + b_k) <= o + NB2 - 1)
                        {
                            c[x + ghm][y + ghn][z + gho][a_i + b_i][a_j + b_j][a_k + b_k] +=
                                field_6d[x + ghostsDim1][y + ghostsDim2][z + ghostsDim3][a_i][a_j][a_k] *
                                b[x + a_i - N2 + b.getGhostsDim1()][y + a_j - N2 + b.getGhostsDim2()][z + a_k - N2 + b.getGhostsDim3()][b_i][b_j][b_k];
                        }
            // clang-format on

            return c;
        }

        // Overload with lvalue-reference to enable chaining, i.e. a.multiply(b.multiply(c))
        template <int NB>
        VaryingStencil<(N > NB ? N : NB) + 2> multiply(VaryingStencil<NB> &&b) const
        {
            return multiply(b);
        }

        // Just an alias for multiply
        template <int NB>
        VaryingStencil<(N > NB ? N : NB) + 2> operator*(VaryingStencil<NB> &&b) const
        {
            return multiply(b);
        }

        template <int NB>
        VaryingStencil<(N > NB ? N : NB) + 2> operator*(VaryingStencil<NB> &b) const
        {
            return multiply(b);
        }
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
