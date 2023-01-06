#ifndef MGCL_STENCIL_HPP
#define MGCL_STENCIL_HPP

#include <memory>
#include <stdexcept>
#include <vector>

#ifndef CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#endif // CL_USE_DEPRECATED_OPENCL_1_2_APIS

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif // CL_TARGET_OPENCL_VERSION

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "cuboid.hpp"
#include "hypercube.hpp"
#include "mgcl.hpp"

namespace mgcl
{
    // forward declarations
    template <int N>
    class VaryingStencil;
    class FixedStencilGpu;
    class VaryingStencilGpu;

    /**
     * @brief Fixed 3x3x3 stencil (same stencil entries for each grid point).
     */
    template <int N>
    class FixedStencil : public Cuboid
    {
    public:
        FixedStencil() : Cuboid(3, 3, 3, 0, 0, 0)
        {
            static_assert(N % 2 == 1 || N < 3, "FixedStencil<N> is only defined for odd N >= 3!");
        }

        /**
         * @brief Multiplies this stencil with a VaryingStencil on the right-hand side. Resulting stencil is two cells
         * wider in each direction. The right-hand side stencil must have ghosts equal to (width_a - 1) / 2 at each border.
         *
         * @tparam NA Width of this stencil
         * @tparam NB Width of the other stencil
         * @param b Other stencil
         * @param ghc Ghosts of resulting stencil at one border. Defaults to 2.
         * @throws If dimensions do not match.
         * @return std::unique_ptr<VaryingStencil<N + 2>> Resulting stencil (two cells wider).
         */
        template <int NB>
        VaryingStencil<(N + NB - 1)> multiply(VaryingStencil<NB> &b, int ghc = 2) const
        {
            if (b.getGhostsDim1() != b.getGhostsDim2() ||
                b.getGhostsDim1() != b.getGhostsDim3())
                throw "Ghosts of b must be equal in each dimension!";

            if (b.getGhostsDim1() < N >> 1)
                throw std::string("Ghosts of b be must be >= ").append(std::to_string(N >> 1));

            int N2 = N >> 1;
            int NB2 = NB >> 1;
            int ghb = b.getGhostsDim1();

            // TODO ghosts of c?
            auto c = VaryingStencil<(N + NB - 1)>(b.getDim1(), b.getDim2(), b.getDim3(), ghc, ghc, ghc);
            int width_c = c.getDim4();

            // clang-format off
            for (int x = 0; x < b.getDim1(); x++)
            for (int y = 0; y < b.getDim2(); y++)
            for (int z = 0; z < b.getDim3(); z++)
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

                        if (ci >= 0 && ci < width_c &&
                            cj >= 0 && cj < width_c &&
                            ck >= 0 && ck < width_c)
                        {
                            c[x + ghc][y + ghc][z + ghc][a_i + b_i][a_j + b_j][a_k + b_k] +=
                                field_3d[a_i][a_j][a_k] *
                                b[gpi][gpj][gpk][b_i][b_j][b_k];
                        }
                    }
            // clang-format on

            if (ghc > 0)
                c.updateGhosts();

            return c;
        }
    };

    typedef FixedStencil<3> FixedStencil3x3x3;

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
         * @brief Multiplies this stencil with a FixedStencil on the rhs. Resulting stencil is two cells wider
         * in each direction. The right-hand side stencil must have ghosts equal to (width_a - 1) / 2 at each border.
         *
         * @tparam NA Width of this stencil
         * @tparam NB Width of the other stencil
         * @param b Other stencil
         * @param ghc Ghosts of resulting stencil at one border. Defaults to 2.
         * @throws If dimensions do not match.
         * @return std::unique_ptr<VaryingStencil<N + 2>> Resulting stencil (two cells wider).
         */
        template <int NB>
        VaryingStencil<(N + NB - 1)> multiply(FixedStencil<NB> &b, int ghc = 2) const
        {
            if (getGhostsDim1() != getGhostsDim2() ||
                getGhostsDim1() != getGhostsDim3())
                throw "Ghosts of a must be equal in each dimension!";

            int m = dim1;
            int n = dim2;
            int o = dim3;
            int N2 = N >> 1;
            int NB2 = NB >> 1;
            int gha = getGhostsDim1();

            auto c = VaryingStencil<(N + NB - 1)>(dim1, dim2, dim3, ghc, ghc, ghc);
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
                        int ci = a_i + b_i;
                        int cj = a_j + b_j;
                        int ck = a_k + b_k;

                        if (ci >= 0 && ci < width_c &&
                            cj >= 0 && cj < width_c &&
                            ck >= 0 && ck < width_c)
                        {
                            c[x + ghc][y + ghc][z + ghc][a_i + b_i][a_j + b_j][a_k + b_k] +=
                                field_6d[x + gha][y + gha][z + gha][a_i][a_j][a_k] *
                                b[b_i][b_j][b_k];
                        }
                    }
            // clang-format on

            if (ghc > 0)
                c.updateGhosts();

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
        template <int NB>
        VaryingStencil<(N + NB - 1)> multiply(VaryingStencil<NB> &b, int ghc = 2) const
        {
            if (dim1 != b.getDim1() ||
                dim2 != b.getDim2() ||
                dim3 != b.getDim3())
                throw "Stencils map to different amount of grid cells!";

            if (b.getGhostsDim1() != b.getGhostsDim2() ||
                b.getGhostsDim1() != b.getGhostsDim3())
                throw "Ghosts of b must be equal in each dimension!";

            if (getGhostsDim1() != getGhostsDim2() ||
                getGhostsDim1() != getGhostsDim3())
                throw "Ghosts of a must be equal in each dimension!";

            if (b.getGhostsDim1() < N >> 1)
                throw std::string("Ghosts of b be must be >= ").append(std::to_string(N >> 1));

            int m = dim1;
            int n = dim2;
            int o = dim3;
            int N2 = N >> 1;
            int NB2 = NB >> 1;
            int gha = getGhostsDim1();
            int ghb = b.getGhostsDim1();

            // TODO ghosts of c?
            auto c = VaryingStencil<(N + NB - 1)>(dim1, dim2, dim3, ghc, ghc, ghc);
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
                        int gpi = x + a_i - N2 + ghb;
                        int gpj = y + a_j - N2 + ghb;
                        int gpk = z + a_k - N2 + ghb;

                        int ci = a_i + b_i;
                        int cj = a_j + b_j;
                        int ck = a_k + b_k;

                        if (ci >= 0 && ci < width_c &&
                            cj >= 0 && cj < width_c &&
                            ck >= 0 && ck < width_c)
                        {
                            c[x + ghc][y + ghc][z + ghc][a_i + b_i][a_j + b_j][a_k + b_k] +=
                                field_6d[x + gha][y + gha][z + gha][a_i][a_j][a_k] *
                                b[gpi][gpj][gpk][b_i][b_j][b_k];
                        }
                    }
            // clang-format on

            if (ghc > 0)
                c.updateGhosts();

            return c;
        }

        /**
         * @brief Multiplies this stencil with a constant factor.
         *
         * @param factor
         * @return VaryingStencil<N>&
         */
        VaryingStencil<N> &operator*(double factor)
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
    };

    typedef VaryingStencil<3> VaryingStencil3x3x3;
    typedef VaryingStencil<5> VaryingStencil5x5x5;
    typedef VaryingStencil<7> VaryingStencil7x7x7;

    /**
     * @brief Creates and returns a fixed 3d full-weight restriction stencil (27p).
     *
     * @return std::unique_ptr<FixedStencil3x3x3>
     */
    static FixedStencil3x3x3 create3dFullWeightRestrictionStencil()
    {
        FixedStencil3x3x3 b;
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
     * @return std::unique_ptr<FixedStencil3x3x3>
     */
    static FixedStencil3x3x3 create3dBilinearProlongationStencil()
    {
        FixedStencil3x3x3 b;
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

    /**
     * @brief Wrapper class for a varying stencil gpu buffer. Stores additional information like width of the grid,
     * width of the stencil and amount of ghost cells. The device buffer gets automatically created in the constructor
     * and gets released in the destructor.
     *
     */
    class VaryingStencilGpu
    {
    private:
        int m;
        int n;
        int o;
        int width;
        int gh;
        cl_mem buf = nullptr;

    public:
        VaryingStencilGpu(int m_, int n_, int o_, int width_, int gh_, cl_context context);
        ~VaryingStencilGpu();

        std::unique_ptr<VaryingStencilGpu> multiply(VaryingStencilGpu &b, int ghc,
                                                    cl_program program, cl_command_queue queue, cl_context context);
        std::unique_ptr<VaryingStencilGpu> multiply(FixedStencilGpu &b, int ghc,
                                                    cl_program program, cl_command_queue queue, cl_context context);

        int getM() const;
        int getN() const;
        int getO() const;
        int getWidth() const;
        int getGh() const;
        cl_mem getBuf() const;
    };

    /**
     * @brief Wrapper class for a fixed stencil gpu buffer. Stores additional information like width of the stencil
     *  and amount of ghost cells.
     *
     */
    class FixedStencilGpu
    {
    private:
        int width;
        int gh;
        cl_mem buf;

    public:
        FixedStencilGpu(int width_, int gh_, cl_context context);
        std::unique_ptr<VaryingStencilGpu> multiply(VaryingStencilGpu &b, int ghc,
                                                    cl_program program, cl_command_queue queue, cl_context context);
    };
}

#endif // MGCL_STENCIL_HPP
