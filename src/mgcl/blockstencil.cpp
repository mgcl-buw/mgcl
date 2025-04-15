#include "blockstencil.hpp"
#include "mgcl.hpp"
#include "mpi_level_data.hpp"

#include <cassert>
#include <iostream>

namespace mgcl
{
    Blockstencil::Blockstencil(int m, int n, int o, int _width, int blocksize, int ghosts_m, int ghosts_n, int ghosts_o)
        : Hypercube8d(blocksize, blocksize, _width, _width, _width, m, n, o, 0, 0, 0, 0, 0, ghosts_m, ghosts_n, ghosts_o)
    {
        if (_width % 2 == 0 || _width < 3)
            error("Blockstencil is only defined for odd width >= 3!");

        if (blocksize < 1)
            error("Blockstencil is only defined for blocksize >= 1!");
    }

    // updates ghost cells, respects periodic ghosts, i.e. when gh > m
    void Blockstencil::updateGhostsLocally()
    {
        int m = getM();
        int n = getN();
        int o = getO();
        int ghosts_m = getGhostsM();
        int ghosts_n = getGhostsN();
        int ghosts_o = getGhostsO();

        auto& c = *this;

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
                for (int ii = 0; ii < getWidth(); ii++)
                for (int jj = 0; jj < getWidth(); jj++)
                for (int kk = 0; kk < getWidth(); kk++)
                    for (int bi = 0; bi < getBlocksize(); bi++)
                    for (int bj = 0; bj < getBlocksize(); bj++)
                    {
                        c[bi][bj][ii][jj][kk][i][j][k] = c[bi][bj][ii][jj][kk][i + factor_left * m][j][k]; // left ghost cell = right real cell
                        c[bi][bj][ii][jj][kk][ghm_start_right + i][j][k] = c[bi][bj][ii][jj][kk][ghm_start_right + i - factor_right * m][j][k]; // right ghost cell = left real cell
                    }
        }

        // sending data in y-direction           
        for (int i = 0; i < ghosts_n; i++)
        {
            int factor_left = (ghosts_n - 1 - i) / n + 1;
            int factor_right = (ghn_start_right + i - ghosts_n) / n;

            for (int j = 0; j < m + 2 * ghosts_m; j++)
            for (int k = 0; k < o + 2 * ghosts_o; k++)
                for (int ii = 0; ii < getWidth(); ii++)
                for (int jj = 0; jj < getWidth(); jj++)
                for (int kk = 0; kk < getWidth(); kk++)
                    for (int bi = 0; bi < getBlocksize(); bi++)
                    for (int bj = 0; bj < getBlocksize(); bj++)
                    {
                        c[bi][bj][ii][jj][kk][j][i][k] = c[bi][bj][ii][jj][kk][j][i + factor_left * n][k]; // left ghost cell = right real cell
                        c[bi][bj][ii][jj][kk][j][ghn_start_right + i][k] = c[bi][bj][ii][jj][kk][j][ghn_start_right + i - factor_right * n][k]; // right ghost cell = left real cell
                    }
        }

        // sending data in x-direction           
        for (int i = 0; i < ghosts_o; i++)
        {
            int factor_left = (ghosts_o - 1 - i) / o + 1;
            int factor_right = (gho_start_right + i - ghosts_o) / o;

            for (int j = 0; j < m + 2 * ghosts_m; j++)
            for (int k = 0; k < n + 2 * ghosts_n; k++)
                for (int ii = 0; ii < getWidth(); ii++)
                for (int jj = 0; jj < getWidth(); jj++)
                for (int kk = 0; kk < getWidth(); kk++)
                    for (int bi = 0; bi < getBlocksize(); bi++)
                    for (int bj = 0; bj < getBlocksize(); bj++)
                    {
                        c[bi][bj][ii][jj][kk][j][k][i] = c[bi][bj][ii][jj][kk][j][k][i + factor_left * o]; // left ghost cell = right real cell
                        c[bi][bj][ii][jj][kk][j][k][gho_start_right + i] = c[bi][bj][ii][jj][kk][j][k][gho_start_right + i - factor_right * o]; // right ghost cell = left real cell
                    }
        }
        // clang-format on
    }

    // /**
    //  * @brief Multiplies this stencil with another one. Dimensions must match. Resulting stencil is two cells wider
    //  * in each direction. The right-hand side stencil must have ghosts equal to (width_a - 1) / 2 at each border.
    //  *
    //  * @tparam NA Width of this stencil
    //  * @tparam NB Width of the other stencil
    //  * @param b Other stencil
    //  * @throws If dimensions do not match.
    //  * @return std::unique_ptr<Blockstencil<N + 2>> Resulting stencil (two cells wider).
    //  */
    // Blockstencil Blockstencil::multiply(Blockstencil& b, int ghc,
    //                                     MPILevelData* mpiData, bool periodic, bool forceLocal) const
    // {
    //     assert(false && "Blockstencil::multiply: Not yet implemented");

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
    //     auto c = Blockstencil(getM(), getN(), getO(), wc, getBlocksize(), ghc, ghc, ghc);

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
    //  * @return Blockstencil&
    //  */
    // Blockstencil& Blockstencil::operator*(double factor)
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
     * @brief Creates and returns a slice of this Blockstencil<N> and returns it as a new Blockstencil<N>.
     *   Boundaries must fit, else an exception is thrown. Blockstencil<N> cells are not included,
     *   i.e. m_end < this->m must hold.
     * By default the new Blockstencil<N> will have the same ghost cell amount as the original one.
     * Boundaries are 0-based, i.e. both start and end will be included.
     * Only real cells are copied, while e.g. m_start = 0 denotes the first real cell.
     *
     * @return std::unique_ptr<Blockstencil>
     */
    std::unique_ptr<Blockstencil> Blockstencil::slice(int m_start, int m_end, int n_start, int n_end,
                                                      int o_start, int o_end,
                                                      int ghm, int ghn, int gho)
    {
        if (m_start < 0 || n_start < 0 || o_start < 0 ||
            m_end >= getM() || n_end >= getN() || o_end >= getO())
            error("Boundaries out of range!");

        if (ghm < 0)
            ghm = getGhostsM();
        if (ghn < 0)
            ghn = getGhostsN();
        if (gho < 0)
            gho = getGhostsO();

        auto ret = std::make_unique<Blockstencil>((m_end - m_start) + 1, (n_end - n_start) + 1,
                                                  (o_end - o_start) + 1,
                                                  getWidth(), getBlocksize(),
                                                  ghm, ghn, gho);
        for (int i = m_start, is = ghm, ib = i + getGhostsM(); i <= m_end; i++, is++, ib++)
            for (int j = n_start, js = ghn, jb = j + getGhostsN(); j <= n_end; j++, js++, jb++)
                for (int k = o_start, ks = gho, kb = k + getGhostsO(); k <= o_end; k++, ks++, kb++)
                    for (int ii = 0; ii < getWidth(); ii++)
                        for (int jj = 0; jj < getWidth(); jj++)
                            for (int kk = 0; kk < getWidth(); kk++)
                                for (int bi = 0; bi < getBlocksize(); bi++)
                                    for (int bj = 0; bj < getBlocksize(); bj++)
                                    {
                                        ret->getData()[bi][bj][ii][jj][kk][is][js][ks] = getData()[bi][bj][ii][jj][kk][ib][jb][kb];
                                    }

        return ret;
    }

    /**
     * @brief Creates and returns a slice of this Blockstencil<N> and returns it as a new Blockstencil<N>.
     * Boundaries must fit, else an exception is thrown. Ghost cells are included, i.e. m_end < this->mgh must hold.
     * The returned Blockstencil<N> has no ghosts cells.
     * Boundaries are 0-based, i.e. both start and end will be included.
     * This behaves just like Cuboid::sliceIncGhosts.
     * The stencil can only be sliced for grid points, i.e. getM(), getN() and getO().
     *
     * @return std::unique_ptr<Blockstencil<N>>
     */
    std::unique_ptr<Blockstencil> Blockstencil::sliceIncGhosts(int m_start, int m_end, int n_start, int n_end,
                                                               int o_start, int o_end)
    {
        if (m_start < 0 || n_start < 0 || o_start < 0 ||
            m_end >= getMgh() || n_end >= getNgh() || o_end >= getOgh())
            error("Boundaries out of range!");

        auto ret = std::make_unique<Blockstencil>((m_end - m_start) + 1, (n_end - n_start) + 1,
                                                  (o_end - o_start) + 1,
                                                  getWidth(), getBlocksize(),
                                                  0, 0, 0);

        for (int i = m_start, is = i - m_start; i <= m_end; i++, is++)
            for (int j = n_start, js = j - n_start; j <= n_end; j++, js++)
                for (int k = o_start, ks = k - o_start; k <= o_end; k++, ks++)
                    for (int ii = 0; ii < getWidth(); ii++)
                        for (int jj = 0; jj < getWidth(); jj++)
                            for (int kk = 0; kk < getWidth(); kk++)
                                for (int bi = 0; bi < getBlocksize(); bi++)
                                    for (int bj = 0; bj < getBlocksize(); bj++)
                                    {
                                        ret->getData()[bi][bj][ii][jj][kk][is][js][ks] = getData()[bi][bj][ii][jj][kk][i][j][k];
                                    }

        return ret;
    }

    /**
     * @brief Returns a copy of this Blockstencil but without values, i.e. only with the same sizes.
     *
     * @return std::unique_ptr<Blockstencil<N>>
     */
    std::unique_ptr<Blockstencil> Blockstencil::copyShallow()
    {
        return std::make_unique<Blockstencil>(getM(), getN(), getO(), getWidth(), getBlocksize(), getGhostsM(), getGhostsN(), getGhostsO());
    }

    std::ostream& operator<<(std::ostream& os, const Blockstencil& v)
    {
        os << "Blockstencil: " << std::endl
           << " m,n,o: " << v.getM() << "," << v.getN() << "," << v.getO() << std::endl
           << " width: " << v.getWidth() << std::endl
           << " blocksize: " << v.getBlocksize() << std::endl
           << " ghm,ghn,gho: " << v.getGhostsM() << "," << v.getGhostsN() << "," << v.getGhostsO() << std::endl;
        return os;
    }

}
