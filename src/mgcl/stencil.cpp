#include "stencil.hpp"
#include "kernel_config.hpp"
#include "mpi_stencil.hpp"
#include "opencl_helper.hpp"
#include "profiling_data.hpp"

#include <algorithm>
#include <utility> // for exchange

#ifdef __APPLE__
#include <OpenCL/cl_platform.h>
#else
#include <CL/cl_platform.h>
#endif

namespace mgcl
{
    // forward declarations
    // void updateGhostsStencilMpi(VaryingStencil& s, MPILevelData* mpiData, bool periodic, bool forceLocal);

    using std::min;

    /*****************************************************
     * FixedStencil
     *****************************************************/

    FixedStencil::FixedStencil(int _width) : Cuboid(_width, _width, _width, 0, 0, 0)
    {
        if (_width % 2 == 0 || _width < 3)
            throw "FixedStencil is only defined for odd width >= 3!";
    }

    /**
     * @brief Multiplies this stencil with a VaryingStencil on the right-hand side. Resulting stencil is two cells
     * wider in each direction. The right-hand side stencil must have ghosts equal to (width_a - 1) / 2 at each border.
     *
     * @tparam NA Width of this stencil
     * @tparam NB Width of the other stencil
     * @param b Other stencil
     * @param ghc Ghosts of resulting stencil at one border.
     * @throws If dimensions do not match.
     * @return std::unique_ptr<VaryingStencil<N + 2>> Resulting stencil (two cells wider).
     */
    VaryingStencil FixedStencil::multiply(VaryingStencil& b, int ghc,
                                          MPILevelData* mpiData, bool periodic, bool forceLocal) const
    {
        if (b.getGhostsM() != b.getGhostsN() ||
            b.getGhostsM() != b.getGhostsO())
            throw "Ghosts of b must be equal in each dimension!";

        int N = getWidth();

        if (b.getGhostsM() < N >> 1)
            throw std::string("Ghosts of b be must be >= ").append(std::to_string(N >> 1));

        int NB = b.getWidth();
        int N2 = N >> 1;
        int ghb = b.getGhostsM();

        // TODO ghosts of c?
        int wc = N + NB - 1;
        auto c = VaryingStencil(b.getM(), b.getN(), b.getO(), wc, ghc, ghc, ghc);

        // clang-format off
        for (int x = 0; x < b.getM(); x++)
        for (int y = 0; y < b.getN(); y++)
        for (int z = 0; z < b.getO(); z++)
            for (int ci = 0; ci < wc; ci++)
            for (int cj = 0; cj < wc; cj++)
            for (int ck = 0; ck < wc; ck++)
            {
                double csum = 0;
                for (int a_i = ci - (min(ci, NB - 1)), b_i = min(ci, NB - 1);
                    a_i <= min(ci, N - 1) && b_i >= ci - min(ci, N - 1);
                    a_i++, b_i--)
                for (int a_j = cj - (min(cj, NB - 1)), b_j = min(cj, NB - 1);
                        a_j <= min(cj, N - 1) && b_j >= cj - min(cj, N - 1);
                        a_j++, b_j--)
                for (int a_k = ck - (min(ck, NB - 1)), b_k = min(ck, NB - 1);
                        a_k <= min(ck, N - 1) && b_k >= ck - min(ck, N - 1);
                        a_k++, b_k--)
                {
                    int gpi = x + a_i - N2 + ghb;
                    int gpj = y + a_j - N2 + ghb;
                    int gpk = z + a_k - N2 + ghb;

                    csum +=
                        field_3d[a_i][a_j][a_k] *
                        b[b_i][b_j][b_k][gpi][gpj][gpk];
                }

                c[ci][cj][ck][x + ghc][y + ghc][z + ghc] = csum;
            }
        // clang-format on

        if (ghc > 0)
            updateGhostsStencilMpi(c, mpiData, periodic, forceLocal);

        return c;
    }

    /*****************************************************
     * VaryingStencil
     *****************************************************/

    VaryingStencil::VaryingStencil(int m, int n, int o, int _width, int ghosts_m, int ghosts_n, int ghosts_o)
        : Hypercube6d(_width, _width, _width, m, n, o, 0, 0, 0, ghosts_m, ghosts_n, ghosts_o)
    {
        if (_width % 2 == 0 || _width < 3)
            throw "VaryingStencil is only defined for odd width >= 3!";
    }

    // updates ghost cells, respects periodic ghosts, i.e. when gh > m
    void VaryingStencil::updateGhosts()
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
                {
                    c[ii][jj][kk][i][j][k] = c[ii][jj][kk][i + factor_left * m][j][k]; // left ghost cell = right real cell
                    c[ii][jj][kk][ghm_start_right + i][j][k] = c[ii][jj][kk][ghm_start_right + i - factor_right * m][j][k]; // right ghost cell = left real cell
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
                {
                    c[ii][jj][kk][j][i][k] = c[ii][jj][kk][j][i + factor_left * n][k]; // left ghost cell = right real cell
                    c[ii][jj][kk][j][ghn_start_right + i][k] = c[ii][jj][kk][j][ghn_start_right + i - factor_right * n][k]; // right ghost cell = left real cell
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
                {
                    c[ii][jj][kk][j][k][i] = c[ii][jj][kk][j][k][i + factor_left * o]; // left ghost cell = right real cell
                    c[ii][jj][kk][j][k][gho_start_right + i] = c[ii][jj][kk][j][k][gho_start_right + i - factor_right * o]; // right ghost cell = left real cell
                }
        }
        // clang-format on
    }

    /**
     * @brief Multiplies this stencil with a FixedStencil on the rhs. Resulting stencil is two cells wider
     * in each direction. The right-hand side stencil must have ghosts equal to (width_a - 1) / 2 at each border.
     *
     * @tparam NA Width of this stencil
     * @tparam NB Width of the other stencil
     * @param b Other stencil
     * @param ghc Ghosts of resulting stencil at one border.
     * @throws If dimensions do not match.
     * @return std::unique_ptr<VaryingStencil<N + 2>> Resulting stencil (two cells wider).
     */
    VaryingStencil VaryingStencil::multiply(FixedStencil& b, int ghc,
                                            MPILevelData* mpiData, bool periodic, bool forceLocal) const
    {
        if (getGhostsM() != getGhostsN() ||
            getGhostsM() != getGhostsO())
            throw "Ghosts of a must be equal in each dimension!";

        int m = getM();
        int n = getN();
        int o = getO();
        int gha = getGhostsM();

        int N = getWidth();
        int NB = b.getWidth();

        int wc = N + NB - 1;
        auto c = VaryingStencil(getM(), getN(), getO(), wc, ghc, ghc, ghc);

        // clang-format off
            for (int x = 0; x < m; x++)
            for (int y = 0; y < n; y++)
            for (int z = 0; z < o; z++)
                for (int ci = 0; ci < wc; ci++)
                for (int cj = 0; cj < wc; cj++)
                for (int ck = 0; ck < wc; ck++)
                {
                    double csum = 0;
                    for (int a_i = ci - (min(ci, NB - 1)), b_i = min(ci, NB - 1);
                        a_i <= min(ci, N - 1) && b_i >= ci - min(ci, N - 1);
                        a_i++, b_i--)
                    for (int a_j = cj - (min(cj, NB - 1)), b_j = min(cj, NB - 1);
                            a_j <= min(cj, N - 1) && b_j >= cj - min(cj, N - 1);
                            a_j++, b_j--)
                    for (int a_k = ck - (min(ck, NB - 1)), b_k = min(ck, NB - 1);
                            a_k <= min(ck, N - 1) && b_k >= ck - min(ck, N - 1);
                            a_k++, b_k--)
                    {
                        csum +=
                            field_6d[a_i][a_j][a_k][x + gha][y + gha][z + gha] *
                            b[b_i][b_j][b_k];
                    }                              

                    c[ci][cj][ck][x + ghc][y + ghc][z + ghc] = csum;
                }
        // clang-format on

        if (ghc > 0)
            updateGhostsStencilMpi(c, mpiData, periodic, forceLocal);

        return c;
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
    VaryingStencil VaryingStencil::multiply(VaryingStencil& b, int ghc,
                                            MPILevelData* mpiData, bool periodic, bool forceLocal) const
    {
        if (getM() != b.getM() ||
            getN() != b.getN() ||
            getO() != b.getO())
            throw "Stencils map to different amount of grid cells!";

        if (b.getGhostsM() != b.getGhostsN() ||
            b.getGhostsM() != b.getGhostsO())
            throw "Ghosts of b must be equal in each dimension!";

        if (getGhostsM() != getGhostsN() ||
            getGhostsM() != getGhostsO())
            throw "Ghosts of a must be equal in each dimension!";

        int N = getWidth();

        if (b.getGhostsM() < N >> 1)
            throw std::string("Ghosts of b be must be >= ").append(std::to_string(N >> 1));

        int NB = b.getWidth();

        int m = getM();
        int n = getN();
        int o = getO();
        int N2 = N >> 1;
        int gha = getGhostsM();
        int ghb = b.getGhostsM();

        int wc = N + NB - 1;
        auto c = VaryingStencil(getM(), getN(), getO(), wc, ghc, ghc, ghc);

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
                    int gpi = x + a_i - N2 + ghb;
                    int gpj = y + a_j - N2 + ghb;
                    int gpk = z + a_k - N2 + ghb;

                    int ci = a_i + b_i;
                    int cj = a_j + b_j;
                    int ck = a_k + b_k;

                    if (ci >= 0 && ci < wc &&
                        cj >= 0 && cj < wc &&
                        ck >= 0 && ck < wc)
                    {
                        c[ci][cj][ck][x + ghc][y + ghc][z + ghc] +=
                            field_6d[a_i][a_j][a_k][x + gha][y + gha][z + gha] *
                            b[b_i][b_j][b_k][gpi][gpj][gpk];
                    }
                }
        // clang-format on

        if (ghc > 0)
            updateGhostsStencilMpi(c, mpiData, periodic, forceLocal);

        return c;
    }

    /**
     * @brief Multiplies this stencil with a constant factor.
     *
     * @param factor
     * @return VaryingStencil&
     */
    VaryingStencil& VaryingStencil::operator*(double factor)
    {
        for (int i = 0; i < dim1; i++)
            for (int j = 0; j < dim2; j++)
                for (int k = 0; k < dim3; k++)
                    for (int ii = 0; ii < dim4; ii++)
                        for (int jj = 0; jj < dim5; jj++)
                            for (int kk = 0; kk < dim6; kk++)
                            {
                                (*this)[i][j][k][ii][jj][kk] *= factor;
                            }

        return *this;
    }

    /**
     * @brief Creates and returns a slice of this VaryingStencil<N> and returns it as a new VaryingStencil<N>.
     *   Boundaries must fit, else an exception is thrown. VaryingStencil<N> cells are not included,
     *   i.e. m_end < this->m must hold.
     * By default the new VaryingStencil<N> will have the same ghost cell amount as the original one.
     * Boundaries are 0-based, i.e. both start and end will be included.
     * Only real cells are copied, while e.g. m_start = 0 denotes the first real cell.
     *
     * @return std::unique_ptr<VaryingStencil>
     */
    std::unique_ptr<VaryingStencil> VaryingStencil::slice(int m_start, int m_end, int n_start, int n_end,
                                                          int o_start, int o_end,
                                                          int ghm, int ghn, int gho)
    {
        if (m_start < 0 || n_start < 0 || o_start < 0 ||
            m_end >= getM() || n_end >= getN() || o_end >= getO())
            throw "Boundaries out of range!";

        if (ghm < 0)
            ghm = getGhostsM();
        if (ghn < 0)
            ghn = getGhostsN();
        if (gho < 0)
            gho = getGhostsO();

        auto ret = std::make_unique<VaryingStencil>((m_end - m_start) + 1, (n_end - n_start) + 1,
                                                    (o_end - o_start) + 1, getWidth(), ghm, ghn, gho);
        for (int i = m_start, is = ghm, ib = i + getGhostsM(); i <= m_end; i++, is++, ib++)
            for (int j = n_start, js = ghn, jb = j + getGhostsN(); j <= n_end; j++, js++, jb++)
                for (int k = o_start, ks = gho, kb = k + getGhostsO(); k <= o_end; k++, ks++, kb++)
                    for (int ii = 0; ii < getWidth(); ii++)
                        for (int jj = 0; jj < getWidth(); jj++)
                            for (int kk = 0; kk < getWidth(); kk++)
                            {
                                ret->getData()[ii][jj][kk][is][js][ks] = getData()[ii][jj][kk][ib][jb][kb];
                            }

        return ret;
    }

    /**
     * @brief Creates and returns a slice of this VaryingStencil<N> and returns it as a new VaryingStencil<N>.
     * Boundaries must fit, else an exception is thrown. Ghost cells are included, i.e. m_end < this->mgh must hold.
     * The returned VaryingStencil<N> has no ghosts cells.
     * Boundaries are 0-based, i.e. both start and end will be included.
     * This behaves just like Cuboid::sliceIncGhosts.
     * The stencil can only be sliced for grid points, i.e. getM(), getN() and getO().
     *
     * @return std::unique_ptr<VaryingStencil<N>>
     */
    std::unique_ptr<VaryingStencil> VaryingStencil::sliceIncGhosts(int m_start, int m_end, int n_start, int n_end,
                                                                   int o_start, int o_end)
    {
        if (m_start < 0 || n_start < 0 || o_start < 0 ||
            m_end >= getMgh() || n_end >= getNgh() || o_end >= getOgh())
            throw "Boundaries out of range!";

        auto ret = std::make_unique<VaryingStencil>((m_end - m_start) + 1, (n_end - n_start) + 1,
                                                    (o_end - o_start) + 1, getWidth(), 0, 0, 0);

        for (int i = m_start, is = i - m_start; i <= m_end; i++, is++)
            for (int j = n_start, js = j - n_start; j <= n_end; j++, js++)
                for (int k = o_start, ks = k - o_start; k <= o_end; k++, ks++)
                    for (int ii = 0; ii < getWidth(); ii++)
                        for (int jj = 0; jj < getWidth(); jj++)
                            for (int kk = 0; kk < getWidth(); kk++)
                            {
                                ret->getData()[ii][jj][kk][is][js][ks] = getData()[ii][jj][kk][i][j][k];
                            }

        return ret;
    }

    /**
     * @brief Returns a copy of this VaryingStencil but without values, i.e. only with the same sizes.
     *
     * @return std::unique_ptr<VaryingStencil<N>>
     */
    std::unique_ptr<VaryingStencil> VaryingStencil::copyShallow()
    {
        return std::make_unique<VaryingStencil>(getM(), getN(), getO(), getWidth(), getGhostsM(), getGhostsN(), getGhostsO());
    }

    std::ostream& operator<<(std::ostream& os, const VaryingStencil& v)
    {
        os << "VaryingStencil: " << std::endl
           << " m,n,o: " << v.getM() << "," << v.getN() << "," << v.getO() << std::endl
           << " width: " << v.getWidth() << std::endl
           << " ghm,ghn,gho: " << v.getGhostsM() << "," << v.getGhostsN() << "," << v.getGhostsO() << std::endl;
        return os;
    }

    /*****************************************************
     * Static methods
     *****************************************************/

    /**
     * @brief Creates and returns a fixed 3d full-weight restriction stencil (27p).
     *
     * @return std::unique_ptr<FixedStencil>
     */
    FixedStencil create3dFullWeightRestrictionStencil()
    {
        FixedStencil b(3);
        double factor1 = 1.0 / 64.0; // corner
        double factor2 = 2.0 / 64.0; // diagonally off
        double factor4 = 4.0 / 64.0; // adjacent
        double factor8 = 8.0 / 64.0; // center

        b[0][0][0] = factor1;
        b[0][0][1] = factor2;
        b[0][0][2] = factor1;
        b[0][1][0] = factor2;
        b[0][1][1] = factor4;
        b[0][1][2] = factor2;
        b[0][2][0] = factor1;
        b[0][2][1] = factor2;
        b[0][2][2] = factor1;
        b[1][0][0] = factor2;
        b[1][0][1] = factor4;
        b[1][0][2] = factor2;
        b[1][1][0] = factor4;
        b[1][1][1] = factor8;
        b[1][1][2] = factor4;
        b[1][2][0] = factor2;
        b[1][2][1] = factor4;
        b[1][2][2] = factor2;
        b[2][0][0] = factor1;
        b[2][0][1] = factor2;
        b[2][0][2] = factor1;
        b[2][1][0] = factor2;
        b[2][1][1] = factor4;
        b[2][1][2] = factor2;
        b[2][2][0] = factor1;
        b[2][2][1] = factor2;
        b[2][2][2] = factor1;

        return b;
    }

    /**
     * @brief Creates and returns a fixed 3d bilinear prolongation stencil (27p).
     *
     * @return std::unique_ptr<FixedStencil>
     */
    FixedStencil create3dBilinearProlongationStencil()
    {
        FixedStencil b(3);
        double factor1 = 1.0 / 8.0; // corner
        double factor2 = 1.0 / 4.0; // diagonally off
        double factor4 = 1.0 / 2.0; // adjacent
        double factor8 = 1.0;       // center

        b[0][0][0] = factor1;
        b[0][0][1] = factor2;
        b[0][0][2] = factor1;
        b[0][1][0] = factor2;
        b[0][1][1] = factor4;
        b[0][1][2] = factor2;
        b[0][2][0] = factor1;
        b[0][2][1] = factor2;
        b[0][2][2] = factor1;
        b[1][0][0] = factor2;
        b[1][0][1] = factor4;
        b[1][0][2] = factor2;
        b[1][1][0] = factor4;
        b[1][1][1] = factor8;
        b[1][1][2] = factor4;
        b[1][2][0] = factor2;
        b[1][2][1] = factor4;
        b[1][2][2] = factor2;
        b[2][0][0] = factor1;
        b[2][0][1] = factor2;
        b[2][0][2] = factor1;
        b[2][1][0] = factor2;
        b[2][1][1] = factor4;
        b[2][1][2] = factor2;
        b[2][2][0] = factor1;
        b[2][2][1] = factor2;
        b[2][2][2] = factor1;

        return b;
    }

    /*****************************************************
     * VaryingStencilGpu
     *****************************************************/

    VaryingStencilGpu::VaryingStencilGpu(int m_, int n_, int o_, int width_, int gh_, cl_context context, cl_command_queue queue)
        : m(m_), n(n_), o(o_), width(width_), gh(gh_)
    {
        int err;
        buf = clCreateBuffer(context, CL_MEM_READ_WRITE,
                             sizeof(double) * (m + 2 * gh) * (n + 2 * gh) * (o + 2 * gh) * width * width * width,
                             NULL, &err);
        mgclCheckError(err, "clCreateBuffer");

        cl_double zero = 0;
        err = clEnqueueFillBuffer(queue, buf, &zero, sizeof(cl_double), 0,
                                  sizeof(double) * (m + 2 * gh) * (n + 2 * gh) * (o + 2 * gh) * width * width * width,
                                  0, NULL, NULL);
        mgclCheckError(err, "clEnqueueFillBuffer");
    }

    VaryingStencilGpu::VaryingStencilGpu(VaryingStencilGpu&& s)
        : m(std::exchange(s.m, 0)),
          n(std::exchange(s.n, 0)),
          o(std::exchange(s.o, 0)),
          width(std::exchange(s.width, 0)),
          gh(std::exchange(s.gh, 0)),
          buf(s.buf) // don't set buf to nullptr since it gets released in dtor
    {
        // retain buffers (i.e. increase internal reference count so they won't be released by accident in dtor)
        if (buf)
        {
            int err = clRetainMemObject(buf);
            mgclCheckError(err, "clRetainMemObject(buf)");
        }
    }

    VaryingStencilGpu& VaryingStencilGpu::operator=(VaryingStencilGpu&& s)
    {
        m = std::exchange(s.m, 0);
        n = std::exchange(s.n, 0);
        o = std::exchange(s.o, 0);
        width = std::exchange(s.width, 0);
        gh = std::exchange(s.gh, 0);
        buf = s.buf;

        // retain buffers (i.e. increase internal reference count so they won't be released by accident in dtor)
        if (buf)
        {
            int err = clRetainMemObject(buf);
            mgclCheckError(err, "clRetainMemObject(buf)");
        }

        return *this;
    }

    VaryingStencilGpu::~VaryingStencilGpu()
    {
        if (buf)
        {
            int err = clReleaseMemObject(buf);
            mgclCheckError(err, "clReleaseMemObject");
        }
    }

    /**
     * Fills the gpu buffer with values from a VaryingStencil.
     */
    void VaryingStencilGpu::fill(VaryingStencil& f, cl_command_queue queue, bool blocking)
    {
        if (f.getWidth() != width)
            throw "Widths are not equal!";

        if (m != f.getM() || n != f.getN() || o != f.getO() ||
            gh != f.getGhostsM() || gh != f.getGhostsN() || gh != f.getGhostsO())
            throw "VaryingStencilGpu::fill: Dimensions are not equal. this.m,n,o = " +
                std::to_string(m) + "," + std::to_string(n) + "," + std::to_string(o) +
                ", f.m,n,o = " + std::to_string(f.getM()) + "," + std::to_string(f.getN()) +
                "," + std::to_string(f.getO());

        int err = clEnqueueWriteBuffer(queue, buf, blocking ? CL_TRUE : CL_FALSE, 0,
                                       sizeof(double) * (m + 2 * gh) * (n + 2 * gh) * (o + 2 * gh) * width * width * width,
                                       f[0][0][0][0][0], 0, NULL, NULL);
        mgclCheckError(err, "clEnqueueWriteBuffer");
    }

    /**
     * Reads the gpu buffer into a new VaryingStencil. The template parameter N must match the width of the gpu
     * stencil.
     */
    VaryingStencil VaryingStencilGpu::read(cl_command_queue queue, bool blocking)
    {
        VaryingStencil ret(m, n, o, width, gh, gh, gh);
        int err = clEnqueueReadBuffer(queue, buf, blocking ? CL_TRUE : CL_FALSE, 0,
                                      sizeof(double) * (m + 2 * gh) * (n + 2 * gh) * (o + 2 * gh) * width * width * width,
                                      ret.field1d().data(), 0, NULL, NULL);
        mgclCheckError(err, "clEnqueueReadBuffer");

        return ret;
    }

    /**
     * @brief Updates ghost cells, respects periodic ghosts, i.e. when gh > m
     *
     * @param program
     * @param queue
     * @param conf Kernel Config, i.e. determines the work-group size. If null, a default value is used.
     */
    void VaryingStencilGpu::updateGhosts(
        cl_program program, cl_command_queue queue,
        conf::KernelConfig* conf, ProfilingData* pd)
    {
        int err;

        // Create the compute kernel from the program
        const char* kernelName = "update_ghosts_varying_stencil";
        cl_kernel kernel = clCreateKernel(program, kernelName, &err);
        mgclCheckError(err, "clCreateKernel");

        // assign kernel arguments
        int pos = 0;
        err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &buf);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &m);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &n);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &o);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &width);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &gh);
        mgclCheckError(err, "Setting kernel arguments");

        // one work-item per ghost cell (excluding real cells). Pad global sizes to fit to local sizes
        int mgh = m + 2 * gh;
        int ngh = n + 2 * gh;
        int ogh = o + 2 * gh;
        size_t global[3] = {static_cast<size_t>(mgh), static_cast<size_t>(ngh), static_cast<size_t>(ogh)};
        size_t local[3] = {
            static_cast<size_t>(mgh > 4 ? 4 : mgh),
            static_cast<size_t>(ngh > 4 ? 4 : ngh),
            static_cast<size_t>(ogh > 4 ? 4 : ogh)};

        // Apply kernel config, if available
        if (conf)
        {
            const auto& c = conf::getWorkGroupSizeForKernelAndWiCount(*conf, kernelName, mgh * ngh * ogh);
            local[0] = static_cast<size_t>(mgh > c[0] ? c[0] : mgh);
            local[1] = static_cast<size_t>(ngh > c[1] ? c[1] : ngh);
            local[2] = static_cast<size_t>(ogh > c[2] ? c[2] : ogh);
        }

        for (int i = 0; i < 3; i++)
            if (global[i] % local[i] != 0)
                global[i] += local[i] - (global[i] % local[i]);

        cl_event ev;

        // enqueue kernel
        err = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global, local, 0, NULL, &ev);
        mgclCheckError(err, "Enqueueing update ghosts of varying stencil kernel");

        if (pd != nullptr)
        {
            pd->addMeasurement(queue, ev, kernelName,
                               {global[0], global[1], global[2]},
                               {local[0], local[1], local[2]});
        }
        mgclCheckError(clReleaseEvent(ev), "clReleaseEvent");

        err = clReleaseKernel(kernel);
        mgclCheckError(err, "Releasing update ghosts of varying stencil kernel");
    }

    /**
     * @brief Multiplies two varying stencils on the gpu and creates a new gpu buffer which will be returned.
     *
     * @param b
     * @param ghc Amount of ghost cells for the result.
     * @param program
     * @param queue
     * @param context
     * @param mpiData Pointer to MPI data, which is required for the ghost update, if MPI is in use. Set to nullptr if
     * MPI is not in use.
     * @param periodic Forwarded to the ghost update.
     * @param forceLocal Forwarded to the ghost update. If true, ghost cells are updated without the use of MPI
     * routines, even when mpiData is not null.
     * @param conf Kernel Config, i.e. determines the work-group size. If null, a default value is used.
     * @return VaryingStencilGpu
     */
    VaryingStencilGpu VaryingStencilGpu::multiply(
        VaryingStencilGpu& b, int ghc,
        cl_program program, cl_command_queue queue, cl_context context,
        MPILevelData* mpiData, bool periodic, bool forceLocal,
        conf::KernelConfig* conf, ProfilingData* pd)
    {
        int err;

        // Create the compute kernel from the program
        const char* kernelName = "mult_stencils_var_var";
        cl_kernel kernel = clCreateKernel(program, kernelName, &err);
        mgclCheckError(err, "clCreateKernel");

        // create output buffer c
        VaryingStencilGpu c(m, n, o, getWidth() + b.getWidth() - 1, ghc, context, queue);

        auto bbuf = b.getBuf();
        auto cbuf = c.getBuf();
        auto wb = b.getWidth();
        auto ghb = b.getGh();

        // assign kernel arguments
        int pos = 0;
        err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &buf);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &bbuf);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &cbuf);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &m);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &n);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &o);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &width);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &wb);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &gh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghb);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghc);
        mgclCheckError(err, "Setting kernel arguments");

        // one work-item per cell (excluding ghost cells). Pad global sizes to fit to local sizes
        size_t global[3] = {static_cast<size_t>(m), static_cast<size_t>(n), static_cast<size_t>(o)};
        size_t local[3] = {static_cast<size_t>(m > 4 ? 4 : m),
                           static_cast<size_t>(n > 4 ? 4 : n),
                           static_cast<size_t>(o > 4 ? 4 : o)};

        // Apply kernel config, if available
        if (conf)
        {
            const auto& c = conf::getWorkGroupSizeForKernelAndWiCount(*conf, kernelName, m * n * o);
            local[0] = static_cast<size_t>(m > c[0] ? c[0] : m);
            local[1] = static_cast<size_t>(n > c[1] ? c[1] : n);
            local[2] = static_cast<size_t>(o > c[2] ? c[2] : o);
        }

        for (int i = 0; i < 3; i++)
            if (global[i] % local[i] != 0)
                global[i] += local[i] - (global[i] % local[i]);

        // update ghosts of b first (maybe not needed if done earlier)
        // b.updateGhosts(program, queue);

        cl_event ev;

        // enqueue multiplication kernel
        err = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global, local, 0, NULL, &ev);
        mgclCheckError(err, "Enqueueing stencil multiplication kernel");

        if (pd != nullptr)
        {
            pd->addMeasurement(queue, ev, kernelName,
                               {global[0], global[1], global[2]},
                               {local[0], local[1], local[2]});
        }
        mgclCheckError(clReleaseEvent(ev), "clReleaseEvent");

        // update ghosts of c
        if (ghc > 0)
            updateGhostsStencilOclMpi(queue, program, c, mpiData, periodic, forceLocal, conf, pd);

        clReleaseKernel(kernel);
        return c;
    }

    /**
     * @brief Multiplies a varying stencil a with a fixed stencil b on the gpu and creates a new gpu buffer c which will
     * be returned, i.e. a * b = c
     *
     * @param b
     * @param ghc Amount of ghost cells for the result.
     * @param program
     * @param queue
     * @param context
     * @param mpiData Pointer to MPI data, which is required for the ghost update, if MPI is in use. Set to nullptr if
     * MPI is not in use.
     * @param periodic Forwarded to the ghost update.
     * @param forceLocal Forwarded to the ghost update. If true, ghost cells are updated without the use of MPI
     * routines, even when mpiData is not null.
     * @param conf Kernel Config, i.e. determines the work-group size. If null, a default value is used.
     * @return VaryingStencilGpu
     */
    VaryingStencilGpu VaryingStencilGpu::multiply(
        FixedStencilGpu& b, int ghc,
        cl_program program, cl_command_queue queue, cl_context context,
        MPILevelData* mpiData, bool periodic, bool forceLocal,
        conf::KernelConfig* conf, ProfilingData* pd)
    {
        int err;

        // Create the compute kernel from the program
        const char* kernelName = "mult_stencils_var_fix";
        cl_kernel kernel = clCreateKernel(program, kernelName, &err);
        mgclCheckError(err, "clCreateKernel");

        // create output buffer c
        VaryingStencilGpu c(m, n, o, getWidth() + b.getWidth() - 1, ghc, context, queue);

        auto bbuf = b.getBuf();
        auto cbuf = c.getBuf();
        auto wb = b.getWidth();

        // assign kernel arguments
        int pos = 0;
        err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &buf);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &bbuf);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &cbuf);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &m);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &n);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &o);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &width);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &wb);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &gh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghc);
        mgclCheckError(err, "Setting kernel arguments");

        int wc = c.getWidth();
        // one work-item per cell (excluding ghost cells). Pad global sizes to fit to local sizes
        size_t global[3] = {static_cast<size_t>(m), static_cast<size_t>(n), static_cast<size_t>(o * wc * wc * wc)};
        size_t local[3] = {static_cast<size_t>(m > 4 ? 4 : m),
                           static_cast<size_t>(n > 4 ? 4 : n),
                           static_cast<size_t>(o > 4 ? 4 : o)};

        // Apply kernel config, if available
        if (conf)
        {
            const auto& c = conf::getWorkGroupSizeForKernelAndWiCount(*conf, kernelName, m * n * o * wc * wc * wc);
            local[0] = static_cast<size_t>(m > c[0] ? c[0] : m);
            local[1] = static_cast<size_t>(n > c[1] ? c[1] : n);
            local[2] = static_cast<size_t>(o > c[2] ? c[2] : o);
        }

        for (int i = 0; i < 3; i++)
            if (global[i] % local[i] != 0)
                global[i] += local[i] - (global[i] % local[i]);

        cl_event ev;

        // enqueue multiplication kernel
        err = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global, local, 0, NULL, &ev);
        mgclCheckError(err, "Enqueueing stencil multiplication kernel");

        if (pd != nullptr)
        {
            pd->addMeasurement(queue, ev, kernelName,
                               {global[0], global[1], global[2]},
                               {local[0], local[1], local[2]});
        }
        mgclCheckError(clReleaseEvent(ev), "clReleaseEvent");

        // update ghosts of c
        if (ghc > 0)
            updateGhostsStencilOclMpi(queue, program, c, mpiData, periodic, forceLocal, conf, pd);

        clReleaseKernel(kernel);
        return c;
    }

    /**
     * @brief Cut stencil from 7x7x7 down to 3x3x3, i.e. copy only selected values to new stencil, skipping ghosts.
     * Ghosts of returning stencil is hard-coded to be 2!
     * @param program OpenCL program
     * @param queue OpenCL command queue
     * @param context OpenCL context
     * @param ghout Number of ghost cells of the output stencil.
     * @param conf Kernel Config, i.e. determines the work-group size. If null, a default value is used.
     * @param resm Size of resulting stencil's grid. Per default halve of this's size.
     * @param resn Size of resulting stencil's grid. Per default halve of this's size.
     * @param reso Size of resulting stencil's grid. Per default halve of this's size.
     * @return VaryingStencilGpu
     */

    VaryingStencilGpu VaryingStencilGpu::cutFromW7ToW3(
        cl_program program, cl_command_queue queue, cl_context context,
        int ghout, mgcl::conf::KernelConfig* conf, ProfilingData* pd,
        int resm, int resn, int reso)
    {
        int err;

        if (width != 7)
            throw "Width is not 7!";

        int m2 = m >> 1;
        int n2 = n >> 1;
        int o2 = o >> 1;
        resm = resm == 0 ? m2 : resm;
        resn = resn == 0 ? n2 : resn;
        reso = reso == 0 ? o2 : reso;

        if (m2 == 0 || n2 == 0 || o2 == 0)
            throw "Cannot cut down stencil of grid size 1!";

        VaryingStencilGpu a_2h(resm, resn, reso, 3, ghout, context, queue);

        // Create the compute kernel from the program
        const char* kernelName = "cut_stencils_w7_to_w3";
        cl_kernel kernel = clCreateKernel(program, kernelName, &err);
        mgclCheckError(err, "clCreateKernel");

        auto outbuf = a_2h.getBuf();
        int nghout = resn + 2 * ghout;
        int oghout = reso + 2 * ghout;

        // assign kernel arguments
        int pos = 0;
        err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &buf);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &outbuf);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &m);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &n);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &o);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &resm);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &resn);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &reso);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &gh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghout);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &nghout);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &oghout);
        mgclCheckError(err, "Setting kernel arguments");

        // one work-item per cell (excluding ghost cells). Pad global sizes to fit to local sizes
        size_t global[3] = {static_cast<size_t>(m2), static_cast<size_t>(n2), static_cast<size_t>(o2)};
        size_t local[3] = {static_cast<size_t>(m2 > 4 ? 4 : m2), static_cast<size_t>(n2 > 4 ? 4 : n2),
                           static_cast<size_t>(o2 > 4 ? 4 : o2)};

        // Apply kernel config, if available
        if (conf)
        {
            const auto& c = conf::getWorkGroupSizeForKernelAndWiCount(*conf, kernelName, m2 * n2 * o2);
            local[0] = static_cast<size_t>(m > c[0] ? c[0] : m);
            local[1] = static_cast<size_t>(n > c[1] ? c[1] : n);
            local[2] = static_cast<size_t>(o > c[2] ? c[2] : o);
        }

        for (int i = 0; i < 3; i++)
            if (global[i] % local[i] != 0)
            {
                // printf("padding global size %d from %ld to ", i, global[i]);
                global[i] += local[i] - (global[i] % local[i]);
                // printf("%ld (multiple of %ld)\n", global[i], local[i]);
            }

        cl_event ev;

        // enqueue kernel
        err = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global, local, 0, NULL, &ev);
        mgclCheckError(err, "Enqueueing stencil cut kernel");

        if (pd != nullptr)
        {
            pd->addMeasurement(queue, ev, kernelName,
                               {global[0], global[1], global[2]},
                               {local[0], local[1], local[2]});
        }
        mgclCheckError(clReleaseEvent(ev), "clReleaseEvent");

        clReleaseKernel(kernel);

        return a_2h;
    }

    int VaryingStencilGpu::getM() const
    {
        return m;
    }

    int VaryingStencilGpu::getN() const
    {
        return n;
    }

    int VaryingStencilGpu::getO() const
    {
        return o;
    }

    int VaryingStencilGpu::getMgh() const
    {
        return m + 2 * gh;
    }

    int VaryingStencilGpu::getNgh() const
    {
        return n + 2 * gh;
    }

    int VaryingStencilGpu::getOgh() const
    {
        return o + 2 * gh;
    }

    cl_mem VaryingStencilGpu::getBuf() const
    {
        return buf;
    }

    int VaryingStencilGpu::getGh() const
    {
        return gh;
    }

    int VaryingStencilGpu::getWidth() const
    {
        return width;
    }

    std::ostream& operator<<(std::ostream& os, const VaryingStencilGpu& v)
    {
        os << "VaryingStencilGpu: " << std::endl
           << " m,n,o: " << v.m << "," << v.n << "," << v.o << std::endl
           << " width: " << v.width << std::endl
           << " gh: " << v.gh << std::endl
           << " buf: " << v.buf << std::endl;
        return os;
    }

    /**
     * *********************************************
     * FixedStencilGpu below
     * *********************************************
     */

    FixedStencilGpu::FixedStencilGpu(int width_, cl_context context, cl_command_queue queue)
        : width(width_)
    {
        int err;
        buf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double) * width * width * width, NULL, &err);
        mgclCheckError(err, "clCreateBuffer");

        cl_double zero = 0;
        err = clEnqueueFillBuffer(queue, buf, &zero, sizeof(cl_double), 0, sizeof(double) * width * width * width,
                                  0, NULL, NULL);
        mgclCheckError(err, "clEnqueueFillBuffer");
    }

    FixedStencilGpu::FixedStencilGpu(FixedStencilGpu&& s)
        : width(std::exchange(s.width, 0)),
          buf(s.buf) // don't set buf to nullptr since it gets released in dtor
    {
        // retain buffers (i.e. increase internal reference count so they won't be released by accident in dtor)
        if (buf)
        {
            int err = clRetainMemObject(buf);
            mgclCheckError(err, "clRetainMemObject(buf)");
        }
    }

    FixedStencilGpu& FixedStencilGpu::operator=(FixedStencilGpu&& s)
    {
        width = std::exchange(s.width, 0);
        buf = s.buf;

        // retain buffers (i.e. increase internal reference count so they won't be released by accident in dtor)
        if (buf)
        {
            int err = clRetainMemObject(buf);
            mgclCheckError(err, "clRetainMemObject(buf)");
        }

        return *this;
    }

    FixedStencilGpu::~FixedStencilGpu()
    {
        if (buf)
        {
            int err = clReleaseMemObject(buf);
            mgclCheckError(err, "clReleaseMemObject");
        }
    }

    void FixedStencilGpu::fill(FixedStencil& f, cl_command_queue queue, bool blocking)
    {
        if (f.getWidth() != width)
            throw "Widths are not equal!";

        int err = clEnqueueWriteBuffer(queue, buf, blocking ? CL_TRUE : CL_FALSE, 0,
                                       sizeof(double) * width * width * width,
                                       f[0][0], 0, NULL, NULL);
        mgclCheckError(err, "clEnqueueFillBuffer");
    }

    /**
     * Reads the gpu buffer into a new FixedStencil. The template parameter N must match the width of the gpu
     * stencil.
     */
    FixedStencil FixedStencilGpu::read(cl_command_queue queue, bool blocking)
    {
        FixedStencil ret(width);
        int err = clEnqueueReadBuffer(queue, buf, blocking ? CL_TRUE : CL_FALSE, 0,
                                      sizeof(double) * width * width * width,
                                      ret.field1d().data(), 0, NULL, NULL);
        mgclCheckError(err, "clEnqueueReadBuffer");

        return ret;
    }

    /**
     * @brief Multiplies a fixed stencil a with a varying stencil b on the gpu and creates a new gpu buffer which will
     * be returned, i.e. a * b = c.
     *
     * @param b
     * @param ghc Amount of ghost cells for the result.
     * @param program
     * @param queue
     * @param context
     * @param mpiData Pointer to MPI data, which is required for the ghost update, if MPI is in use. Set to nullptr if
     * MPI is not in use.
     * @param periodic Forwarded to the ghost update.
     * @param forceLocal Forwarded to the ghost update. If true, ghost cells are updated without the use of MPI
     * routines, even when mpiData is not null.
     * @param conf Kernel Config, i.e. determines the work-group size. If null, a default value is used.
     * @return VaryingStencilGpu
     */
    VaryingStencilGpu FixedStencilGpu::multiply(VaryingStencilGpu& b, int ghc,
                                                cl_program program, cl_command_queue queue, cl_context context,
                                                MPILevelData* mpiData, bool periodic, bool forceLocal,
                                                conf::KernelConfig* conf, ProfilingData* pd)
    {
        int err;

        // Create the compute kernel from the program
        const char* kernelName = "mult_stencils_fix_var";
        cl_kernel kernel = clCreateKernel(program, kernelName, &err);
        mgclCheckError(err, "clCreateKernel");

        int m = b.getM();
        int n = b.getN();
        int o = b.getO();

        // create output buffer c
        VaryingStencilGpu c(m, n, o, getWidth() + b.getWidth() - 1, ghc, context, queue);

        auto bbuf = b.getBuf();
        auto cbuf = c.getBuf();
        auto wb = b.getWidth();
        auto ghb = b.getGh();

        // assign kernel arguments
        int pos = 0;
        err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &buf);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &bbuf);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &cbuf);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &m);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &n);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &o);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &width);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &wb);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghb);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghc);
        mgclCheckError(err, "Setting kernel arguments");

        int wc = c.getWidth();
        // one work-item per cell (excluding ghost cells). Pad global sizes to fit to local sizes
        size_t global[3] = {static_cast<size_t>(m), static_cast<size_t>(n), static_cast<size_t>(o * wc * wc * wc)};
        size_t local[3] = {static_cast<size_t>(m > 4 ? 4 : m),
                           static_cast<size_t>(n > 4 ? 4 : n),
                           static_cast<size_t>(o > 4 ? 4 : o)};

        // Apply kernel config, if available
        if (conf)
        {
            const auto& c = conf::getWorkGroupSizeForKernelAndWiCount(*conf, kernelName, m * n * o * wc * wc * wc);
            local[0] = static_cast<size_t>(m > c[0] ? c[0] : m);
            local[1] = static_cast<size_t>(n > c[1] ? c[1] : n);
            local[2] = static_cast<size_t>(o > c[2] ? c[2] : o);
        }

        for (int i = 0; i < 3; i++)
            if (global[i] % local[i] != 0)
                global[i] += local[i] - (global[i] % local[i]);

        // update ghosts of b first (maybe not needed if done earlier)
        // b.updateGhosts(program, queue);

        cl_event ev;

        // enqueue multiplication kernel
        err = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global, local, 0, NULL, &ev);
        mgclCheckError(err, "Enqueueing stencil multiplication kernel");

        if (pd != nullptr)
        {
            pd->addMeasurement(queue, ev, kernelName,
                               {global[0], global[1], global[2]},
                               {local[0], local[1], local[2]});
        }
        mgclCheckError(clReleaseEvent(ev), "clReleaseEvent");

        // update ghosts of c
        if (ghc > 0)
            updateGhostsStencilOclMpi(queue, program, c, mpiData, periodic, forceLocal, conf, pd);

        clReleaseKernel(kernel);
        return c;
    }

    int FixedStencilGpu::getWidth() const
    {
        return width;
    }

    cl_mem FixedStencilGpu::getBuf() const
    {
        return buf;
    }

    FixedStencilGpu create3dFullWeightRestrictionStencilGpu(cl_context context, cl_command_queue queue)
    {
        FixedStencilGpu ret(3, context, queue);

        auto s = create3dFullWeightRestrictionStencil();
        ret.fill(s, queue, true);

        return ret;
    }

    FixedStencilGpu create3dBilinearProlongationStencilGpu(cl_context context, cl_command_queue queue)
    {
        FixedStencilGpu ret(3, context, queue);

        auto s = create3dBilinearProlongationStencil();
        ret.fill(s, queue, true);

        return ret;
    }
}
