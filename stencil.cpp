#include "stencil.hpp"

namespace mgcl
{
    std::unique_ptr<VaryingStencil> VaryingStencil::multiply(VaryingStencil &b)
    {
        int m = dim1;
        int n = dim2;
        int o = dim3;
        int ghm = ghostsDim1;
        int ghn = ghostsDim2;
        int gho = ghostsDim3;

        auto c = std::make_unique<VaryingStencil>(m, n, o, ghm, ghn, gho);

        // clang-format off
        for (int x = 0; x < m; x++)
        for (int y = 0; y < n; y++)
        for (int z = 0; z < o; z++)
            for (int a_i = 0; a_i < 3; a_i++)
            for (int a_j = 0; a_j < 3; a_j++)
            for (int a_k = 0; a_k < 3; a_k++)
                for (int b_i = 0; b_i < 3; b_i++)
                for (int b_j = 0; b_j < 3; b_j++)
                for (int b_k = 0; b_k < 3; b_k++)
                    if ((x + a_i) >= 1 && (x + a_i) <= m &&
                        (y + a_j) >= 1 && (y + a_j) <= n &&
                        (z + a_k) >= 1 && (z + a_k) <= o &&
                        (x + a_i - 1 + b_i) >= 1 && (x + a_i - 1 + b_i) <= m &&
                        (y + a_j - 1 + b_j) >= 1 && (y + a_j - 1 + b_j) <= n &&
                        (z + a_k - 1 + b_k) >= 1 && (z + a_k - 1 + b_k) <= o)
                    {
                        (*c)[x][y][z][a_i + b_i][a_j + b_j][a_k + b_k] +=
                            field_6d[x][y][z][a_i][a_j][a_k] * b[x + a_i - 1][y + a_j - 1][z + a_k - 1][b_i][b_j][b_k];
                    }
        // clang-format on

        return c;
    }
}