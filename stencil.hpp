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
            static_assert(N % 2 == 1 || N < 3, "VaryingStencil<N> is only defined for odd N >= 3!");
        }
        VaryingStencil(VaryingStencil &) = delete;
        VaryingStencil &operator=(const VaryingStencil &) = delete;
        VaryingStencil(VaryingStencil &&o) : Hypercube6d(std::move(o)) {}
        VaryingStencil &operator=(const VaryingStencil &&o)
        {
            Hypercube6d::operator=(std::move(o));
            return *this;
        }

        // updates ghost cells, respects periodic ghosts, i.e. when gh > m
        void updateGhosts()
        {
            int m = dim1;
            int n = dim2;
            int o = dim3;
            int ghosts_m = ghostsDim1;
            int ghosts_n = ghostsDim2;
            int ghosts_o = ghostsDim3;

            auto &c = *this;

            int ghm_start_right = ghosts_m + m;
            int ghn_start_right = ghosts_n + n;
            int gho_start_right = ghosts_o + o;

            // clang-format off
            // sending data in z-direction           
            for (int i = 0; i < ghosts_m; i++)
            {
                int factor_left = (ghosts_m - 1 - i) / m + 1;
                int factor_right = (ghm_start_right + i - ghosts_m) / m;

                for (int j = 0; j < n + 2 * ghosts_n; j++)
                for (int k = 0; k < o + 2 * ghosts_o; k++)
                    for (int ii = 0; ii < N; ii++)
                    for (int jj = 0; jj < N; jj++)
                    for (int kk = 0; kk < N; kk++)
                    {
                        
                        c[i][j][k][ii][jj][kk] = c[i + factor_left * m][j][k][ii][jj][kk]; // left ghost cell = right real cell
                        c[ghm_start_right + i][j][k][ii][jj][kk] = c[ghm_start_right + i - factor_right * m][j][k][ii][jj][kk]; // right ghost cell = left real cell
                    }
            }

            // sending data in y-direction           
            for (int i = 0; i < ghosts_n; i++)
            {
                int factor_left = (ghosts_n - 1 - i) / n + 1;
                int factor_right = (ghn_start_right + i - ghosts_n) / n;

                for (int j = 0; j < m + 2 * ghosts_m; j++)
                for (int k = 0; k < o + 2 * ghosts_o; k++)
                    for (int ii = 0; ii < N; ii++)
                    for (int jj = 0; jj < N; jj++)
                    for (int kk = 0; kk < N; kk++)
                    {
                        
                        c[j][i][k][ii][jj][kk] = c[j][i + factor_left * n][k][ii][jj][kk]; // left ghost cell = right real cell
                        c[j][ghn_start_right + i][k][ii][jj][kk] = c[j][ghn_start_right + i - factor_right * n][k][ii][jj][kk]; // right ghost cell = left real cell
                    }
            }

            // sending data in x-direction           
            for (int i = 0; i < ghosts_o; i++)
            {
                int factor_left = (ghosts_o - 1 - i) / o + 1;
                int factor_right = (gho_start_right + i - ghosts_o) / o;

                for (int j = 0; j < m + 2 * ghosts_m; j++)
                for (int k = 0; k < n + 2 * ghosts_n; k++)
                    for (int ii = 0; ii < N; ii++)
                    for (int jj = 0; jj < N; jj++)
                    for (int kk = 0; kk < N; kk++)
                    {
                        
                        c[j][k][i][ii][jj][kk] = c[j][k][i + factor_left * o][ii][jj][kk]; // left ghost cell = right real cell
                        c[j][k][gho_start_right + i][ii][jj][kk] = c[j][k][gho_start_right + i - factor_right * o][ii][jj][kk]; // right ghost cell = left real cell
                    }
            }
            // clang-format on
        }

        /**
         * @brief Multiplies this stencil with another one. Dimensions must match. Resulting stencil is two cells wider
         * in each direction. The right-hand side stencil must have ghosts equal to (width_a - 1) / 2 at each border.
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
            if (dim1 != b.getDim1() ||
                dim2 != b.getDim2() ||
                dim3 != b.getDim3())
                throw "Stencils map to different amount of grid cells!";

            if (b.getGhostsDim1() != b.getGhostsDim2() ||
                b.getGhostsDim1() != b.getGhostsDim3())
                throw "Ghosts of b must be equal in each dimension!";

            if (b.getGhostsDim1() != N >> 1)
                throw std::string("Ghosts of b be must be of size ").append(std::to_string(N >> 1));

            int m = dim1;
            int n = dim2;
            int o = dim3;
            int N2 = N >> 1;
            int NB2 = NB >> 1;
            int gh = b.getGhostsDim1();

            // TODO ghosts of c?
            auto c = VaryingStencil<(N > NB ? N : NB) + 2>(dim1, dim2, dim3, 0, 0, 0);
            int width_c = c.getDim4();

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
                    {
                        int gpi = x + a_i - N2 + gh;
                        int gpj = y + a_j - N2 + gh;
                        int gpk = z + a_k - N2 + gh;

                        int ci = a_i + b_i;
                        int cj = a_j + b_j;
                        int ck = a_k + b_k;

                        if (ci >= 0 && ci < width_c &&
                            cj >= 0 && cj < width_c &&
                            ck >= 0 && ck < width_c)
                        {
                            c[x][y][z][a_i + b_i][a_j + b_j][a_k + b_k] +=
                                field_6d[x][y][z][a_i][a_j][a_k] *
                                b[gpi][gpj][gpk][b_i][b_j][b_k];
                        }
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

    /**
     * @brief Creates and returns a 3d full-weight restriction stencil for a grid of size mxnxo.
     *
     * @param m real grid size dim 1
     * @param n real grid size dim 2
     * @param o real grid size dim 3
     * @param ghm ghosts in dim 1
     * @param ghn ghosts in dim 2
     * @param gho ghosts in dim 3
     * @return std::unique_ptr<VaryingStencil<3>>
     */
    static std::unique_ptr<VaryingStencil3x3x3> create3dFullWeightRestrictionStencil(int m, int n, int o, int ghm, int ghn, int gho)
    {
        auto bptr = std::make_unique<VaryingStencil3x3x3>(m, n, o, ghm, ghn, gho);
        auto &b = *bptr;
        double factor1 = 1.0 / 64.0;
        double factor2 = 2.0 / 64.0;
        double factor4 = 4.0 / 64.0;
        double factor8 = 8.0 / 64.0;

        int mgh = m + 2 * ghm;
        int ngh = n + 2 * ghn;
        int ogh = o + 2 * gho;

        for (int i = 0; i < mgh; i++)
            for (int j = 0; j < ngh; j++)
                for (int k = 0; k < ogh; k++)
                {
                    b[i][j][k][0][0][0] = factor1;
                    b[i][j][k][0][0][1] = factor2;
                    b[i][j][k][0][0][2] = factor1;
                    b[i][j][k][0][1][0] = factor2;
                    b[i][j][k][0][1][1] = factor4;
                    b[i][j][k][0][1][2] = factor2;
                    b[i][j][k][0][2][0] = factor1;
                    b[i][j][k][0][2][1] = factor2;
                    b[i][j][k][0][2][2] = factor1;
                    b[i][j][k][1][0][0] = factor2;
                    b[i][j][k][1][0][1] = factor4;
                    b[i][j][k][1][0][2] = factor2;
                    b[i][j][k][1][1][0] = factor4;
                    b[i][j][k][1][1][1] = factor8;
                    b[i][j][k][1][1][2] = factor4;
                    b[i][j][k][1][2][0] = factor2;
                    b[i][j][k][1][2][1] = factor4;
                    b[i][j][k][1][2][2] = factor2;
                    b[i][j][k][2][0][0] = factor1;
                    b[i][j][k][2][0][1] = factor2;
                    b[i][j][k][2][0][2] = factor1;
                    b[i][j][k][2][1][0] = factor2;
                    b[i][j][k][2][1][1] = factor4;
                    b[i][j][k][2][1][2] = factor2;
                    b[i][j][k][2][2][0] = factor1;
                    b[i][j][k][2][2][1] = factor2;
                    b[i][j][k][2][2][2] = factor1;
                }

        return bptr;
    }
}

#endif // MGCL_STENCIL_HPP
