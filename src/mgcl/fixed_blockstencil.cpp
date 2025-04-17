#include "fixed_blockstencil.hpp"
#include "mgcl.hpp"
#include "mpi_level_data.hpp"
#include "mpi_util.hpp"

#include <cassert>
#include <iostream>

namespace mgcl
{
    FixedBlockstencil::FixedBlockstencil(int _width, int blocksize)
        : Hypercube5d(blocksize, blocksize, _width, _width, _width)
    {
        if (_width % 2 == 0 || _width < 3)
            error("FixedBlockstencil is only defined for odd width >= 3!");

        if (blocksize < 1)
            error("FixedBlockstencil is only defined for blocksize >= 1!");
    }

    // /**
    //  * @brief Multiplies this stencil with another one. Dimensions must match. Resulting stencil is two cells wider
    //  * in each direction. The right-hand side stencil must have ghosts equal to (width_a - 1) / 2 at each border.
    //  *
    //  * @tparam NA Width of this stencil
    //  * @tparam NB Width of the other stencil
    //  * @param b Other stencil
    //  * @throws If dimensions do not match.
    //  * @return std::unique_ptr<FixedBlockstencil<N + 2>> Resulting stencil (two cells wider).
    //  */
    // FixedBlockstencil FixedBlockstencil::multiply(FixedBlockstencil& b, int ghc,
    //                                     MPILevelData* mpiData, bool periodic, bool forceLocal) const
    // {
    //     assert(false && "FixedBlockstencil::multiply: Not yet implemented");

    //     if (getM() != b.getM() ||
    //         getN() != b.getN() ||
    //         getO() != b.getO())
    //         error("Stencils map to different amount of grid cells!");

    //     if (b.getGhostsM() != b.getGhostsN() ||
    //         b.getGhostsM() != b.getGhostsO())
    //         error("Ghosts of b must be equal in each dimension!");

    //     if (getGhostsM() != getGhostsN() ||
    //         getGhostsM() != getGhostsO())
    //         error("Ghosts of a must be equal in each dimension!");

    //     if (getBlocksize() != b.getBlocksize())
    //         error("Blocksize of a and b must be equal!");

    //     int N = getWidth();

    //     if (b.getGhostsM() < N >> 1)
    //         error(std::string("Ghosts of b be must be >= ").append(std::to_string(N >> 1)));

    //     int NB = b.getWidth();

    //     int m = getM();
    //     int n = getN();
    //     int o = getO();
    //     int N2 = N >> 1;
    //     int gha = getGhostsM();
    //     int ghb = b.getGhostsM();

    //     int wc = N + NB - 1;
    //     auto c = FixedBlockstencil(getM(), getN(), getO(), wc, getBlocksize(), ghc, ghc, ghc);

    //     // // clang-format off
    //     // for (int x = 0; x < m; x++)
    //     // for (int y = 0; y < n; y++)
    //     // for (int z = 0; z < o; z++)
    //     //     for (int a_i = 0; a_i < N; a_i++)
    //     //     for (int a_j = 0; a_j < N; a_j++)
    //     //     for (int a_k = 0; a_k < N; a_k++)
    //     //         for (int b_i = 0; b_i < NB; b_i++)
    //     //         for (int b_j = 0; b_j < NB; b_j++)
    //     //         for (int b_k = 0; b_k < NB; b_k++)
    //     //         {
    //     //             int gpi = x + a_i - N2 + ghb;
    //     //             int gpj = y + a_j - N2 + ghb;
    //     //             int gpk = z + a_k - N2 + ghb;

    //     //             int ci = a_i + b_i;
    //     //             int cj = a_j + b_j;
    //     //             int ck = a_k + b_k;

    //     //             if (ci >= 0 && ci < wc &&
    //     //                 cj >= 0 && cj < wc &&
    //     //                 ck >= 0 && ck < wc)
    //     //             {
    //     //                 // TODO
    //     //                 c[ci][cj][ck][x + ghc][y + ghc][z + ghc] +=
    //     //                     field_8d[a_i][a_j][a_k][x + gha][y + gha][z + gha] *
    //     //                     b[b_i][b_j][b_k][gpi][gpj][gpk];
    //     //             }
    //     //         }
    //     // // clang-format on

    //     // if (ghc > 0)
    //     //     updateGhostsStencilMpi(c, mpiData, periodic, forceLocal);

    //     return c;
    // }

    // /**
    //  * @brief Multiplies this stencil with a constant factor.
    //  *
    //  * @param factor
    //  * @return FixedBlockstencil&
    //  */
    // FixedBlockstencil& FixedBlockstencil::operator*(double factor)
    // {
    //     for (int i = 0; i < dim1; i++)
    //         for (int j = 0; j < dim2; j++)
    //             for (int k = 0; k < dim3; k++)
    //                 for (int ii = 0; ii < dim4; ii++)
    //                     for (int jj = 0; jj < dim5; jj++)
    //                         for (int kk = 0; kk < dim6; kk++)
    //                             for (int ll = 0; ll < dim7; ll++)
    //                                 for (int mm = 0; mm < dim8; mm++)
    //                                 {
    //                                     (*this)[i][j][k][ii][jj][kk][ll][mm] *= factor;
    //                                 }

    //     return *this;
    // }

    /**
     * @brief Returns a copy of this FixedBlockstencil but without values, i.e. only with the same sizes.
     *
     * @return std::unique_ptr<FixedBlockstencil<N>>
     */
    std::unique_ptr<FixedBlockstencil> FixedBlockstencil::copyShallow()
    {
        return std::make_unique<FixedBlockstencil>(getWidth(), getBlocksize());
    }

    std::ostream& operator<<(std::ostream& os, const FixedBlockstencil& v)
    {
        os << "FixedBlockstencil: " << std::endl
           << " width: " << v.getWidth() << std::endl
           << " blocksize: " << v.getBlocksize() << std::endl;
        return os;
    }

}
