#include "stencil.hpp"

namespace mgcl
{
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
    template <int N>
    template <int NB>
    std::unique_ptr<VaryingStencil<(N > NB ? N : NB) + 2>> VaryingStencil<N>::multiply(VaryingStencil<NB> &b) const
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

        auto c = std::make_unique<VaryingStencil<(N > NB ? N : NB) + 2>>(m, n, o, ghm, ghn, gho);

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
                        (x + a_i - 1 + b_i) >= NB2 && (x + a_i - 1 + b_i) <= m + NB2 - 1 &&
                        (y + a_j - 1 + b_j) >= NB2 && (y + a_j - 1 + b_j) <= n + NB2 - 1 &&
                        (z + a_k - 1 + b_k) >= NB2 && (z + a_k - 1 + b_k) <= o + NB2 - 1)
                    {
                        (*c)[x + ghm][y + ghn][z + gho][a_i + b_i][a_j + b_j][a_k + b_k] +=
                            field_6d[x + ghostsDim1][y + ghostsDim2][z + ghostsDim3][a_i][a_j][a_k] *
                            b[x + a_i - 1 + b.getGhostsDim1()][y + a_j - 1 + b.getGhostsDim2()][z + a_k - 1 + b.getGhostsDim3()][b_i][b_j][b_k];
                    }
        // clang-format on

        return c;
    }

    template <int N>
    template <int NB>
    std::unique_ptr<VaryingStencil<(N > NB ? N : NB) + 2>> VaryingStencil<N>::operator*(VaryingStencil<NB> &b) const
    {
        return multiply(b);
    }

    // instantiate template classes to avoid linker errors
    template class VaryingStencil<3>;
    template class VaryingStencil<5>;
    template class VaryingStencil<7>;
}
