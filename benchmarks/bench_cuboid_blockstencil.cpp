/**
 * @date 17.12.2024
 * This file contains code for benchmarking different implementations for enabling vector-valued cuboids.
 *
 * One approach would be to always use 4d arrays instead of 3d arrays and when using scalar values, setting one dimension
 * to 1.
 *
 */

#include "bench_util.hpp"
#include "nanobench.h"

#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"

#include <CL/cl.h>
#include <catch2/catch_message.hpp>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>
using namespace std::chrono_literals;

#include "../src/mgcl/cuboid.hpp"
#include "../src/mgcl/problem.hpp"
#include "cli_args.hpp"

namespace mgcl_bench
{
    // Cuboid variant that stores 4d arrays, Can set the outermost dim to 1 to use as 3d array.
    class Cuboid4d
    {
    protected:
        int m;
        int n;
        int o;
        int w;
        int mgh;
        int ngh;
        int ogh;
        int ghostsM = 0;
        int ghostsN = 0;
        int ghostsO = 0;
        std::vector<double> field_1d;
        double**** field_4d;

    public:
        explicit Cuboid4d(int m_, int n_, int o_, int w_, int ghostsM_, int ghostsN_, int ghostsO_)
            : m(m_),
              n(n_),
              o(o_),
              w(w_),
              mgh(m_ + 2 * ghostsM_),
              ngh(n_ + 2 * ghostsN_),
              ogh(o_ + 2 * ghostsO_),
              ghostsM(ghostsM_),
              ghostsN(ghostsN_),
              ghostsO(ghostsO_)
        {
            int i, j, k;

            field_1d.resize(mgh * ngh * ogh * w);
            for (i = 0; i < field_1d.size(); i++)
                field_1d[i] = 0.0;

            field_4d = new double***[w];
            for (i = 0; i < w; i++)
            {
                field_4d[i] = new double**[mgh];
                for (j = 0; j < mgh; j++)
                {
                    field_4d[i][j] = new double*[ngh];
                    for (k = 0; k < ngh; k++)
                    {
                        field_4d[i][j][k] = &field_1d[i * mgh * ngh * ogh + j * ngh * ogh + k * ogh];
                    }
                }
            }
        }

        ~Cuboid4d()
        {
            for (int l = 0; l < w; l++)
            {
                for (int i = 0; i < mgh; i++)
                {
                    delete[] field_4d[l][i];
                }
                delete[] field_4d[l];
            }
            delete[] field_4d;
            field_4d = nullptr;
        }

        int getM() const { return m; }
        int getN() const { return n; }
        int getO() const { return o; }
        int getW() const { return w; }
        int getMgh() const { return mgh; }
        int getNgh() const { return ngh; }
        int getOgh() const { return ogh; }
        int getGhostsM() const { return ghostsM; }
        int getGhostsN() const { return ghostsN; }
        int getGhostsO() const { return ghostsO; }
        double**** getData() const { return field_4d; }
        inline double*** operator[](int index) { return field_4d[index]; }

        /**
         * @brief Fills this Cuboid with the 1d index of the corresponding cell, including ghost cells.
         *
         * @param realCellsOnly If true, only real cells get filled.
         */
        void fill1dIndex(bool realCellsOnly)
        {
            if (realCellsOnly)
                for (int l = 0; l < w; l++)
                    for (int i = ghostsM; i < m + ghostsM; i++)
                        for (int j = ghostsN; j < n + ghostsN; j++)
                            for (int k = ghostsO; k < o + ghostsO; k++)
                            {
                                (*this)[l][i][j][k] = l * mgh * ngh * ogh + i * ngh * ogh + j * ogh + k;
                            }
            else
                for (int l = 0; l < w; l++)
                    for (int i = 0; i < mgh; i++)
                        for (int j = 0; j < ngh; j++)
                            for (int k = 0; k < ogh; k++)
                            {
                                (*this)[l][i][j][k] = l * mgh * ngh * ogh + i * ngh * ogh + j * ogh + k;
                            }
        }

        /* @brief Dumps content to file fiven by path overwriting existing files.
         *
         * @param path Path to file, overwrites existing one.
         * @param realCellsOnly If true, only real cells will be written. Defaults to false.
         * @throws runtime_error When file could not be opened.
         */
        void dumpToFile(std::string path, bool realCellsOnly)
        {
            std::ofstream myfile;
            myfile.open(path, std::ios::out | std::ios::trunc);

            if (myfile.is_open())
            {
                if (realCellsOnly)
                {
                    for (int i = ghostsM; i < mgh - ghostsM; i++)
                        for (int j = ghostsN; j < ngh - ghostsN; j++)
                            for (int k = ghostsO; k < ogh - ghostsO; k++)
                            {
                                myfile << i - ghostsM << "\t" << j - ghostsN << "\t" << k - ghostsO << "\t"
                                       << std::scientific << std::setprecision(17) << field_4d[0][i][j][k] << std::endl;
                            }
                }
                else
                {
                    for (int i = 0; i < mgh; i++)
                        for (int j = 0; j < ngh; j++)
                            for (int k = 0; k < ogh; k++)
                            {
                                myfile << i << "\t" << j << "\t" << k << "\t"
                                       << std::scientific << std::setprecision(17) << field_4d[0][i][j][k] << std::endl;
                            }
                }
                myfile.close();
            }
            else
            {
                error(std::runtime_error("Couldn't open file for writing given by: " + path));
            }
        }
    };

    // Templated Cuboid variant that stores 4d arrays. Can set the outermost dim to 1 to use as 3d array.
    template <int w>
    class Cuboid4dTempl
    {
    protected:
        int m;
        int n;
        int o;
        int mgh;
        int ngh;
        int ogh;
        int ghostsM = 0;
        int ghostsN = 0;
        int ghostsO = 0;
        std::vector<double> field_1d;
        double**** field_4d;

    public:
        explicit Cuboid4dTempl(int m_, int n_, int o_, int ghostsM_, int ghostsN_, int ghostsO_)
            : m(m_),
              n(n_),
              o(o_),
              mgh(m_ + 2 * ghostsM_),
              ngh(n_ + 2 * ghostsN_),
              ogh(o_ + 2 * ghostsO_),
              ghostsM(ghostsM_),
              ghostsN(ghostsN_),
              ghostsO(ghostsO_)
        {
            int i, j, k;

            field_1d.resize(mgh * ngh * ogh * w);
            for (i = 0; i < field_1d.size(); i++)
                field_1d[i] = 0.0;

            field_4d = new double***[w];
            for (i = 0; i < w; i++)
            {
                field_4d[i] = new double**[mgh];
                for (j = 0; j < mgh; j++)
                {
                    field_4d[i][j] = new double*[ngh];
                    for (k = 0; k < ngh; k++)
                    {
                        field_4d[i][j][k] = &field_1d[i * mgh * ngh * ogh + j * ngh * ogh + k * ogh];
                    }
                }
            }
        }

        ~Cuboid4dTempl()
        {
            for (int l = 0; l < w; l++)
            {
                for (int i = 0; i < mgh; i++)
                {
                    delete[] field_4d[l][i];
                }
                delete[] field_4d[l];
            }
            delete[] field_4d;
            field_4d = nullptr;
        }

        int getM() const { return m; }
        int getN() const { return n; }
        int getO() const { return o; }
        int getW() const { return w; }
        int getMgh() const { return mgh; }
        int getNgh() const { return ngh; }
        int getOgh() const { return ogh; }
        int getGhostsM() const { return ghostsM; }
        int getGhostsN() const { return ghostsN; }
        int getGhostsO() const { return ghostsO; }
        double**** getData() const { return field_4d; }
        inline double*** operator[](int index) { return field_4d[index]; }

        /**
         * @brief Fills this Cuboid with the 1d index of the corresponding cell, including ghost cells.
         *
         * @param realCellsOnly If true, only real cells get filled.
         */
        void fill1dIndex(bool realCellsOnly)
        {
            if (realCellsOnly)
                for (int l = 0; l < w; l++)
                    for (int i = ghostsM; i < m + ghostsM; i++)
                        for (int j = ghostsN; j < n + ghostsN; j++)
                            for (int k = ghostsO; k < o + ghostsO; k++)
                            {
                                (*this)[l][i][j][k] = l * mgh * ngh * ogh + i * ngh * ogh + j * ogh + k;
                            }
            else
                for (int l = 0; l < w; l++)
                    for (int i = 0; i < mgh; i++)
                        for (int j = 0; j < ngh; j++)
                            for (int k = 0; k < ogh; k++)
                            {
                                (*this)[l][i][j][k] = l * mgh * ngh * ogh + i * ngh * ogh + j * ogh + k;
                            }
        }

        /* @brief Dumps content to file fiven by path overwriting existing files.
         *
         * @param path Path to file, overwrites existing one.
         * @param realCellsOnly If true, only real cells will be written. Defaults to false.
         * @throws runtime_error When file could not be opened.
         */
        void dumpToFile(std::string path, bool realCellsOnly)
        {
            std::ofstream myfile;
            myfile.open(path, std::ios::out | std::ios::trunc);

            if (myfile.is_open())
            {
                if (realCellsOnly)
                {
                    for (int i = ghostsM; i < mgh - ghostsM; i++)
                        for (int j = ghostsN; j < ngh - ghostsN; j++)
                            for (int k = ghostsO; k < ogh - ghostsO; k++)
                            {
                                myfile << i - ghostsM << "\t" << j - ghostsN << "\t" << k - ghostsO << "\t"
                                       << std::scientific << std::setprecision(17) << field_4d[0][i][j][k] << std::endl;
                            }
                }
                else
                {
                    for (int i = 0; i < mgh; i++)
                        for (int j = 0; j < ngh; j++)
                            for (int k = 0; k < ogh; k++)
                            {
                                myfile << i << "\t" << j << "\t" << k << "\t"
                                       << std::scientific << std::setprecision(17) << field_4d[0][i][j][k] << std::endl;
                            }
                }
                myfile.close();
            }
            else
            {
                error(std::runtime_error("Couldn't open file for writing given by: " + path));
            }
        }
    };

    // TODO templated Cuboid4d with fixed size w

    /* Calculates r = f - A*v using 7-point, 19-point or 27-point stencil of 3D laplacian or a varying stencil.
     * m,n,o is the size of the real grid.
     * v needs to have updated ghost cells if the problem is periodic!
     * moff, noff and ooff can be used to change the size of the grid that the residual shall be calculated for.
     *   Per default only real cells are considered (moff = 0), but with e.g. moff = -1, the first ghost cell border is
     *   considered, too. Analogously, with moff = 1 the outermost set of real cells is ignored. The calculation
     *   of the boundaries is e.g. istart = v.ghosts_m + moff.
     * Based on MultigridEngine::residualSeq as of 17.12.2024.
     */
    double residualSeq(mgcl::Cuboid& f, mgcl::Cuboid& v, mgcl::Cuboid& r, mgcl::MGCL_RESIDUAL_NORM resnorm,
                       mgcl::MGCL_STENCIL stencilType, double stencilFactor,
                       mgcl::VaryingStencil* stencilValuesCuboid, mgcl::FixedStencil* fixedStencil,
                       bool returnResidualNorm,
                       bool periodic, bool updateGhostsLocally, int moff, int noff, int ooff)
    {
        double res = 0.0;
        double stencilsum = 0;
        double****** stencilValues;
        double*** vraw = v.getData();
        double*** fsRaw;

        // check if off is too small (i.e. start < 0)
        if (moff <= -v.getGhostsM() || noff <= -v.getGhostsN() || ooff <= -v.getGhostsO())
            error("moff, noff and ooff must not be <= -ghosts");

        // check if off is too large (i.e. start > end)
        if (moff * 2 >= v.getM() || noff * 2 >= v.getN() || ooff * 2 >= v.getO())
            error("2*moff, 2*noff and 2*ooff must not be >= m, n or o");

        // check that stencilValues is not null if stencil type is varying
        if (stencilType == mgcl::MGCL_VARYING && stencilValuesCuboid == nullptr)
            error("stencilType is varying but stencilValues is null!");

        if (stencilType == mgcl::MGCL_FIXED && fixedStencil == nullptr)
        {
            error("stencilType is fixed but fixedStencil is null!");
        }

        if (stencilType == mgcl::MGCL_VARYING)
            stencilValues = stencilValuesCuboid->getData();

        if (stencilType == mgcl::MGCL_FIXED)
        {
            fsRaw = fixedStencil->getData();
        }

        int istart_v = v.getGhostsM() + moff;
        int jstart_v = v.getGhostsN() + noff;
        int kstart_v = v.getGhostsO() + ooff;
        int iend_v = v.getMgh() - v.getGhostsM() - moff;
        int jend_v = v.getNgh() - v.getGhostsN() - noff;
        int kend_v = v.getOgh() - v.getGhostsO() - ooff;
        int istart_r = r.getGhostsM() + moff;
        int jstart_r = r.getGhostsN() + noff;
        int kstart_r = r.getGhostsO() + ooff;
        int istart_f = f.getGhostsM() + moff;
        int jstart_f = f.getGhostsN() + noff;
        int kstart_f = f.getGhostsO() + ooff;
        int istart_sv = stencilValuesCuboid ? stencilValuesCuboid->getGhostsM() + moff : 0;
        int jstart_sv = stencilValuesCuboid ? stencilValuesCuboid->getGhostsN() + noff : 0;
        int kstart_sv = stencilValuesCuboid ? stencilValuesCuboid->getGhostsO() + ooff : 0;

        for (int iv = istart_v, ir = istart_r, fi = istart_f, isv = istart_sv; iv < iend_v; iv++, ir++, fi++, isv++)
            for (int jv = jstart_v, jr = jstart_r, fj = jstart_f, jsv = jstart_sv; jv < jend_v; jv++, jr++, fj++, jsv++)
                for (int kv = kstart_v, kr = kstart_r, fk = kstart_f, ksv = kstart_sv; kv < kend_v; kv++, kr++, fk++, ksv++)
                {
                    // A*v
                    if (stencilType == mgcl::MGCL_LAPLACE_7POINT)
                    {
                        // clang-format off
                        stencilsum = (6.0 * vraw[iv][jv][kv]
                            - vraw[iv][jv][kv - 1] - vraw[iv][jv][kv + 1]
                            - vraw[iv][jv - 1][kv] - vraw[iv][jv + 1][kv]
                            - vraw[iv - 1][jv][kv] - vraw[iv + 1][jv][kv]
                            ) * stencilFactor;
                        // clang-format on
                    }
                    else if (stencilType == mgcl::MGCL_LAPLACE_19POINT)
                    {
                        // clang-format off
                        stencilsum = (24.0 * vraw[iv][jv][kv]
                                - 2.0 * vraw[iv][jv][kv - 1] - 2.0 * vraw[iv][jv][kv + 1]
                                - 2.0 * vraw[iv][jv - 1][kv] - 2.0 * vraw[iv][jv + 1][kv]
                                - 2.0 * vraw[iv - 1][jv][kv] - 2.0 * vraw[iv + 1][jv][kv]
                                
                                - vraw[iv][jv - 1][kv - 1] - vraw[iv][jv - 1][kv + 1]
                                - vraw[iv][jv + 1][kv - 1] - vraw[iv][jv + 1][kv + 1]
                                - vraw[iv - 1][jv][kv - 1] - vraw[iv - 1][jv][kv + 1]
                                - vraw[iv + 1][jv][kv - 1] - vraw[iv + 1][jv][kv + 1]
                                - vraw[iv - 1][jv - 1][kv] - vraw[iv - 1][jv + 1][kv]
                                - vraw[iv + 1][jv - 1][kv] - vraw[iv + 1][jv + 1][kv]
                                ) * stencilFactor;
                        // clang-format on
                    }
                    else if (stencilType == mgcl::MGCL_LAPLACE_27POINT)
                    {
                        // clang-format off
                        stencilsum = (88.0 * vraw[iv][jv][kv]
                                - 6.0 * vraw[iv][jv][kv - 1] - 6.0 * vraw[iv][jv][kv + 1]
                                - 6.0 * vraw[iv][jv - 1][kv] - 6.0 * vraw[iv][jv + 1][kv]
                                - 6.0 * vraw[iv - 1][jv][kv] - 6.0 * vraw[iv + 1][jv][kv]

                                - 3.0 * vraw[iv][jv - 1][kv - 1] - 3.0 * vraw[iv][jv - 1][kv + 1]
                                - 3.0 * vraw[iv][jv + 1][kv - 1] - 3.0 * vraw[iv][jv + 1][kv + 1]
                                - 3.0 * vraw[iv - 1][jv][kv - 1] - 3.0 * vraw[iv - 1][jv][kv + 1]
                                - 3.0 * vraw[iv + 1][jv][kv - 1] - 3.0 * vraw[iv + 1][jv][kv + 1]
                                - 3.0 * vraw[iv - 1][jv - 1][kv] - 3.0 * vraw[iv - 1][jv + 1][kv]
                                - 3.0 * vraw[iv + 1][jv - 1][kv] - 3.0 * vraw[iv + 1][jv + 1][kv]

                                - 2.0 * vraw[iv - 1][jv - 1][kv - 1] - 2.0 * vraw[iv - 1][jv - 1][kv + 1]
                                - 2.0 * vraw[iv - 1][jv + 1][kv - 1] - 2.0 * vraw[iv - 1][jv + 1][kv + 1]
                                - 2.0 * vraw[iv + 1][jv - 1][kv - 1] - 2.0 * vraw[iv + 1][jv - 1][kv + 1]
                                - 2.0 * vraw[iv + 1][jv + 1][kv - 1] - 2.0 * vraw[iv + 1][jv + 1][kv + 1]
                                ) * stencilFactor;
                        // clang-format on
                    }
                    else if (stencilType == mgcl::MGCL_FIXED)
                    {
                        // clang-format off
                        stencilsum = fsRaw[1][1][1]  * vraw[iv][jv][kv]
                            + fsRaw[1][1][0] * vraw[ iv ][ jv ][kv-1]
                            + fsRaw[1][1][2] * vraw[ iv ][ jv ][kv+1]
                            + fsRaw[1][0][1] * vraw[ iv ][jv-1][ kv ]
                            + fsRaw[1][2][1] * vraw[ iv ][jv+1][ kv ]
                            + fsRaw[0][1][1] * vraw[iv-1][ jv ][ kv ]
                            + fsRaw[2][1][1] * vraw[iv+1][ jv ][ kv ]
                            
                            + fsRaw[1][0][0] * vraw[ iv ][jv-1][kv-1]
                            + fsRaw[1][0][2] * vraw[ iv ][jv-1][kv+1]
                            + fsRaw[1][2][0] * vraw[ iv ][jv+1][kv-1]
                            + fsRaw[1][2][2] * vraw[ iv ][jv+1][kv+1]
                            + fsRaw[0][1][0] * vraw[iv-1][ jv ][kv-1]
                            + fsRaw[0][1][2] * vraw[iv-1][ jv ][kv+1]
                            + fsRaw[2][1][0] * vraw[iv+1][ jv ][kv-1]
                            + fsRaw[2][1][2] * vraw[iv+1][ jv ][kv+1]
                            + fsRaw[0][0][1] * vraw[iv-1][jv-1][ kv ]
                            + fsRaw[0][2][1] * vraw[iv-1][jv+1][ kv ]
                            + fsRaw[2][0][1] * vraw[iv+1][jv-1][ kv ]
                            + fsRaw[2][2][1] * vraw[iv+1][jv+1][ kv ]
                            
                            + fsRaw[0][0][0] * vraw[iv-1][jv-1][kv-1]
                            + fsRaw[0][0][2] * vraw[iv-1][jv-1][kv+1]
                            + fsRaw[0][2][0] * vraw[iv-1][jv+1][kv-1]
                            + fsRaw[0][2][2] * vraw[iv-1][jv+1][kv+1]
                            + fsRaw[2][0][0] * vraw[iv+1][jv-1][kv-1]
                            + fsRaw[2][0][2] * vraw[iv+1][jv-1][kv+1]
                            + fsRaw[2][2][0] * vraw[iv+1][jv+1][kv-1]
                            + fsRaw[2][2][2] * vraw[iv+1][jv+1][kv+1];
                        // clang-format on
                    }
                    else if (stencilType == mgcl::MGCL_VARYING)
                    {
                        // clang-format off
                        stencilsum = stencilValues[1][1][1][isv][jsv][ksv]  * vraw[iv][jv][kv]
                            + stencilValues[1][1][0][isv][jsv][ksv] * vraw[ iv ][ jv ][kv-1]
                            + stencilValues[1][1][2][isv][jsv][ksv] * vraw[ iv ][ jv ][kv+1]
                            + stencilValues[1][0][1][isv][jsv][ksv] * vraw[ iv ][jv-1][ kv ]
                            + stencilValues[1][2][1][isv][jsv][ksv] * vraw[ iv ][jv+1][ kv ]
                            + stencilValues[0][1][1][isv][jsv][ksv] * vraw[iv-1][ jv ][ kv ]
                            + stencilValues[2][1][1][isv][jsv][ksv] * vraw[iv+1][ jv ][ kv ]
                            
                            + stencilValues[1][0][0][isv][jsv][ksv] * vraw[ iv ][jv-1][kv-1]
                            + stencilValues[1][0][2][isv][jsv][ksv] * vraw[ iv ][jv-1][kv+1]
                            + stencilValues[1][2][0][isv][jsv][ksv] * vraw[ iv ][jv+1][kv-1]
                            + stencilValues[1][2][2][isv][jsv][ksv] * vraw[ iv ][jv+1][kv+1]
                            + stencilValues[0][1][0][isv][jsv][ksv] * vraw[iv-1][ jv ][kv-1]
                            + stencilValues[0][1][2][isv][jsv][ksv] * vraw[iv-1][ jv ][kv+1]
                            + stencilValues[2][1][0][isv][jsv][ksv] * vraw[iv+1][ jv ][kv-1]
                            + stencilValues[2][1][2][isv][jsv][ksv] * vraw[iv+1][ jv ][kv+1]
                            + stencilValues[0][0][1][isv][jsv][ksv] * vraw[iv-1][jv-1][ kv ]
                            + stencilValues[0][2][1][isv][jsv][ksv] * vraw[iv-1][jv+1][ kv ]
                            + stencilValues[2][0][1][isv][jsv][ksv] * vraw[iv+1][jv-1][ kv ]
                            + stencilValues[2][2][1][isv][jsv][ksv] * vraw[iv+1][jv+1][ kv ]
                            
                            + stencilValues[0][0][0][isv][jsv][ksv] * vraw[iv-1][jv-1][kv-1]
                            + stencilValues[0][0][2][isv][jsv][ksv] * vraw[iv-1][jv-1][kv+1]
                            + stencilValues[0][2][0][isv][jsv][ksv] * vraw[iv-1][jv+1][kv-1]
                            + stencilValues[0][2][2][isv][jsv][ksv] * vraw[iv-1][jv+1][kv+1]
                            + stencilValues[2][0][0][isv][jsv][ksv] * vraw[iv+1][jv-1][kv-1]
                            + stencilValues[2][0][2][isv][jsv][ksv] * vraw[iv+1][jv-1][kv+1]
                            + stencilValues[2][2][0][isv][jsv][ksv] * vraw[iv+1][jv+1][kv-1]
                            + stencilValues[2][2][2][isv][jsv][ksv] * vraw[iv+1][jv+1][kv+1];
                        // clang-format on

                        // if (j == 2 && k == 2 && i == 2)
                        // {
                        //     printf("seq stencilsum = %e\n", stencilsum);
                        //     print27point_sv(v, i, j, k, stencilValuesCuboid, isv, jsv, ksv);
                        // }
                    }

                    // r = f - A*v
                    r[ir][jr][kr] = f[fi][fj][fk] - stencilsum;

                    if (returnResidualNorm)
                    {
                        if (resnorm == mgcl::MGCL_L2)
                            res += r[ir][jr][kr] * r[ir][jr][kr];
                        else if (fabs(r[ir][jr][kr]) > res)
                            res = fabs(r[ir][jr][kr]);
                    }
                }

        return (returnResidualNorm && resnorm == mgcl::MGCL_L2) ? sqrt(res) : res;
    }

    /* Calculates r = f - A*v using 7-point, 19-point or 27-point stencil of 3D laplacian or a varying stencil.
     * m,n,o is the size of the real grid.
     * v needs to have updated ghost cells if the problem is periodic!
     * moff, noff and ooff can be used to change the size of the grid that the residual shall be calculated for.
     *   Per default only real cells are considered (moff = 0), but with e.g. moff = -1, the first ghost cell border is
     *   considered, too. Analogously, with moff = 1 the outermost set of real cells is ignored. The calculation
     *   of the boundaries is e.g. istart = v.ghosts_m + moff.
     * Based on MultigridEngine::residualSeq as of 17.12.2024.
     */
    double residualSeq(mgcl_bench::Cuboid4d& f, mgcl_bench::Cuboid4d& v, mgcl_bench::Cuboid4d& r, mgcl::MGCL_RESIDUAL_NORM resnorm,
                       mgcl::MGCL_STENCIL stencilType, double stencilFactor,
                       mgcl::VaryingStencil* stencilValuesCuboid, mgcl::FixedStencil* fixedStencil,
                       bool returnResidualNorm,
                       bool periodic, bool updateGhostsLocally, int moff, int noff, int ooff)
    {
        double res = 0.0;
        double stencilsum = 0;
        double****** stencilValues;
        double**** vraw = v.getData();
        double*** fsRaw;

        // check if off is too small (i.e. start < 0)
        if (moff <= -v.getGhostsM() || noff <= -v.getGhostsN() || ooff <= -v.getGhostsO())
            error("moff, noff and ooff must not be <= -ghosts");

        // check if off is too large (i.e. start > end)
        if (moff * 2 >= v.getM() || noff * 2 >= v.getN() || ooff * 2 >= v.getO())
            error("2*moff, 2*noff and 2*ooff must not be >= m, n or o");

        // check that stencilValues is not null if stencil type is varying
        if (stencilType == mgcl::MGCL_VARYING && stencilValuesCuboid == nullptr)
            error("stencilType is varying but stencilValues is null!");

        if (stencilType == mgcl::MGCL_FIXED && fixedStencil == nullptr)
        {
            error("stencilType is fixed but fixedStencil is null!");
        }

        if (stencilType == mgcl::MGCL_VARYING)
            stencilValues = stencilValuesCuboid->getData();

        if (stencilType == mgcl::MGCL_FIXED)
        {
            fsRaw = fixedStencil->getData();
        }

        int istart_v = v.getGhostsM() + moff;
        int jstart_v = v.getGhostsN() + noff;
        int kstart_v = v.getGhostsO() + ooff;
        int iend_v = v.getMgh() - v.getGhostsM() - moff;
        int jend_v = v.getNgh() - v.getGhostsN() - noff;
        int kend_v = v.getOgh() - v.getGhostsO() - ooff;
        int istart_r = r.getGhostsM() + moff;
        int jstart_r = r.getGhostsN() + noff;
        int kstart_r = r.getGhostsO() + ooff;
        int istart_f = f.getGhostsM() + moff;
        int jstart_f = f.getGhostsN() + noff;
        int kstart_f = f.getGhostsO() + ooff;
        int istart_sv = stencilValuesCuboid ? stencilValuesCuboid->getGhostsM() + moff : 0;
        int jstart_sv = stencilValuesCuboid ? stencilValuesCuboid->getGhostsN() + noff : 0;
        int kstart_sv = stencilValuesCuboid ? stencilValuesCuboid->getGhostsO() + ooff : 0;

        for (int iv = istart_v, ir = istart_r, fi = istart_f, isv = istart_sv; iv < iend_v; iv++, ir++, fi++, isv++)
            for (int jv = jstart_v, jr = jstart_r, fj = jstart_f, jsv = jstart_sv; jv < jend_v; jv++, jr++, fj++, jsv++)
                for (int kv = kstart_v, kr = kstart_r, fk = kstart_f, ksv = kstart_sv; kv < kend_v; kv++, kr++, fk++, ksv++)
                {
                    // A*v
                    if (stencilType == mgcl::MGCL_LAPLACE_7POINT)
                    {
                        // clang-format off
                        stencilsum = (6.0 * vraw[0][iv][jv][kv]
                            - vraw[0][iv][jv][kv - 1] - vraw[0][iv][jv][kv + 1]
                            - vraw[0][iv][jv - 1][kv] - vraw[0][iv][jv + 1][kv]
                            - vraw[0][iv - 1][jv][kv] - vraw[0][iv + 1][jv][kv]
                            ) * stencilFactor;
                        // clang-format on
                    }
                    else if (stencilType == mgcl::MGCL_LAPLACE_19POINT)
                    {
                        // clang-format off
                        stencilsum = (24.0 * vraw[0][iv][jv][kv]
                                - 2.0 * vraw[0][iv][jv][kv - 1] - 2.0 * vraw[0][iv][jv][kv + 1]
                                - 2.0 * vraw[0][iv][jv - 1][kv] - 2.0 * vraw[0][iv][jv + 1][kv]
                                - 2.0 * vraw[0][iv - 1][jv][kv] - 2.0 * vraw[0][iv + 1][jv][kv]
                                
                                - vraw[0][iv][jv - 1][kv - 1] - vraw[0][iv][jv - 1][kv + 1]
                                - vraw[0][iv][jv + 1][kv - 1] - vraw[0][iv][jv + 1][kv + 1]
                                - vraw[0][iv - 1][jv][kv - 1] - vraw[0][iv - 1][jv][kv + 1]
                                - vraw[0][iv + 1][jv][kv - 1] - vraw[0][iv + 1][jv][kv + 1]
                                - vraw[0][iv - 1][jv - 1][kv] - vraw[0][iv - 1][jv + 1][kv]
                                - vraw[0][iv + 1][jv - 1][kv] - vraw[0][iv + 1][jv + 1][kv]
                                ) * stencilFactor;
                        // clang-format on
                    }
                    else if (stencilType == mgcl::MGCL_LAPLACE_27POINT)
                    {
                        // clang-format off
                        stencilsum = (88.0 * vraw[0][iv][jv][kv]
                                - 6.0 * vraw[0][iv][jv][kv - 1] - 6.0 * vraw[0][iv][jv][kv + 1]
                                - 6.0 * vraw[0][iv][jv - 1][kv] - 6.0 * vraw[0][iv][jv + 1][kv]
                                - 6.0 * vraw[0][iv - 1][jv][kv] - 6.0 * vraw[0][iv + 1][jv][kv]

                                - 3.0 * vraw[0][iv][jv - 1][kv - 1] - 3.0 * vraw[0][iv][jv - 1][kv + 1]
                                - 3.0 * vraw[0][iv][jv + 1][kv - 1] - 3.0 * vraw[0][iv][jv + 1][kv + 1]
                                - 3.0 * vraw[0][iv - 1][jv][kv - 1] - 3.0 * vraw[0][iv - 1][jv][kv + 1]
                                - 3.0 * vraw[0][iv + 1][jv][kv - 1] - 3.0 * vraw[0][iv + 1][jv][kv + 1]
                                - 3.0 * vraw[0][iv - 1][jv - 1][kv] - 3.0 * vraw[0][iv - 1][jv + 1][kv]
                                - 3.0 * vraw[0][iv + 1][jv - 1][kv] - 3.0 * vraw[0][iv + 1][jv + 1][kv]

                                - 2.0 * vraw[0][iv - 1][jv - 1][kv - 1] - 2.0 * vraw[0][iv - 1][jv - 1][kv + 1]
                                - 2.0 * vraw[0][iv - 1][jv + 1][kv - 1] - 2.0 * vraw[0][iv - 1][jv + 1][kv + 1]
                                - 2.0 * vraw[0][iv + 1][jv - 1][kv - 1] - 2.0 * vraw[0][iv + 1][jv - 1][kv + 1]
                                - 2.0 * vraw[0][iv + 1][jv + 1][kv - 1] - 2.0 * vraw[0][iv + 1][jv + 1][kv + 1]
                                ) * stencilFactor;
                        // clang-format on
                    }
                    else if (stencilType == mgcl::MGCL_FIXED)
                    {
                        // clang-format off
                        stencilsum = fsRaw[1][1][1]  * vraw[0][iv][jv][kv]
                            + fsRaw[1][1][0] * vraw[0][ iv ][ jv ][kv-1]
                            + fsRaw[1][1][2] * vraw[0][ iv ][ jv ][kv+1]
                            + fsRaw[1][0][1] * vraw[0][ iv ][jv-1][ kv ]
                            + fsRaw[1][2][1] * vraw[0][ iv ][jv+1][ kv ]
                            + fsRaw[0][1][1] * vraw[0][iv-1][ jv ][ kv ]
                            + fsRaw[2][1][1] * vraw[0][iv+1][ jv ][ kv ]
                            
                            + fsRaw[1][0][0] * vraw[0][ iv ][jv-1][kv-1]
                            + fsRaw[1][0][2] * vraw[0][ iv ][jv-1][kv+1]
                            + fsRaw[1][2][0] * vraw[0][ iv ][jv+1][kv-1]
                            + fsRaw[1][2][2] * vraw[0][ iv ][jv+1][kv+1]
                            + fsRaw[0][1][0] * vraw[0][iv-1][ jv ][kv-1]
                            + fsRaw[0][1][2] * vraw[0][iv-1][ jv ][kv+1]
                            + fsRaw[2][1][0] * vraw[0][iv+1][ jv ][kv-1]
                            + fsRaw[2][1][2] * vraw[0][iv+1][ jv ][kv+1]
                            + fsRaw[0][0][1] * vraw[0][iv-1][jv-1][ kv ]
                            + fsRaw[0][2][1] * vraw[0][iv-1][jv+1][ kv ]
                            + fsRaw[2][0][1] * vraw[0][iv+1][jv-1][ kv ]
                            + fsRaw[2][2][1] * vraw[0][iv+1][jv+1][ kv ]
                            
                            + fsRaw[0][0][0] * vraw[0][iv-1][jv-1][kv-1]
                            + fsRaw[0][0][2] * vraw[0][iv-1][jv-1][kv+1]
                            + fsRaw[0][2][0] * vraw[0][iv-1][jv+1][kv-1]
                            + fsRaw[0][2][2] * vraw[0][iv-1][jv+1][kv+1]
                            + fsRaw[2][0][0] * vraw[0][iv+1][jv-1][kv-1]
                            + fsRaw[2][0][2] * vraw[0][iv+1][jv-1][kv+1]
                            + fsRaw[2][2][0] * vraw[0][iv+1][jv+1][kv-1]
                            + fsRaw[2][2][2] * vraw[0][iv+1][jv+1][kv+1];
                        // clang-format on
                    }
                    else if (stencilType == mgcl::MGCL_VARYING)
                    {
                        // clang-format off
                        stencilsum = stencilValues[1][1][1][isv][jsv][ksv]  * vraw[0][iv][jv][kv]
                            + stencilValues[1][1][0][isv][jsv][ksv] * vraw[0][ iv ][ jv ][kv-1]
                            + stencilValues[1][1][2][isv][jsv][ksv] * vraw[0][ iv ][ jv ][kv+1]
                            + stencilValues[1][0][1][isv][jsv][ksv] * vraw[0][ iv ][jv-1][ kv ]
                            + stencilValues[1][2][1][isv][jsv][ksv] * vraw[0][ iv ][jv+1][ kv ]
                            + stencilValues[0][1][1][isv][jsv][ksv] * vraw[0][iv-1][ jv ][ kv ]
                            + stencilValues[2][1][1][isv][jsv][ksv] * vraw[0][iv+1][ jv ][ kv ]
                            
                            + stencilValues[1][0][0][isv][jsv][ksv] * vraw[0][ iv ][jv-1][kv-1]
                            + stencilValues[1][0][2][isv][jsv][ksv] * vraw[0][ iv ][jv-1][kv+1]
                            + stencilValues[1][2][0][isv][jsv][ksv] * vraw[0][ iv ][jv+1][kv-1]
                            + stencilValues[1][2][2][isv][jsv][ksv] * vraw[0][ iv ][jv+1][kv+1]
                            + stencilValues[0][1][0][isv][jsv][ksv] * vraw[0][iv-1][ jv ][kv-1]
                            + stencilValues[0][1][2][isv][jsv][ksv] * vraw[0][iv-1][ jv ][kv+1]
                            + stencilValues[2][1][0][isv][jsv][ksv] * vraw[0][iv+1][ jv ][kv-1]
                            + stencilValues[2][1][2][isv][jsv][ksv] * vraw[0][iv+1][ jv ][kv+1]
                            + stencilValues[0][0][1][isv][jsv][ksv] * vraw[0][iv-1][jv-1][ kv ]
                            + stencilValues[0][2][1][isv][jsv][ksv] * vraw[0][iv-1][jv+1][ kv ]
                            + stencilValues[2][0][1][isv][jsv][ksv] * vraw[0][iv+1][jv-1][ kv ]
                            + stencilValues[2][2][1][isv][jsv][ksv] * vraw[0][iv+1][jv+1][ kv ]
                            
                            + stencilValues[0][0][0][isv][jsv][ksv] * vraw[0][iv-1][jv-1][kv-1]
                            + stencilValues[0][0][2][isv][jsv][ksv] * vraw[0][iv-1][jv-1][kv+1]
                            + stencilValues[0][2][0][isv][jsv][ksv] * vraw[0][iv-1][jv+1][kv-1]
                            + stencilValues[0][2][2][isv][jsv][ksv] * vraw[0][iv-1][jv+1][kv+1]
                            + stencilValues[2][0][0][isv][jsv][ksv] * vraw[0][iv+1][jv-1][kv-1]
                            + stencilValues[2][0][2][isv][jsv][ksv] * vraw[0][iv+1][jv-1][kv+1]
                            + stencilValues[2][2][0][isv][jsv][ksv] * vraw[0][iv+1][jv+1][kv-1]
                            + stencilValues[2][2][2][isv][jsv][ksv] * vraw[0][iv+1][jv+1][kv+1];
                        // clang-format on

                        // if (j == 2 && k == 2 && i == 2)
                        // {
                        //     printf("seq stencilsum = %e\n", stencilsum);
                        //     print27point_sv(v, i, j, k, stencilValuesCuboid, isv, jsv, ksv);
                        // }
                    }

                    // r = f - A*v
                    r[0][ir][jr][kr] = f[0][fi][fj][fk] - stencilsum;

                    if (returnResidualNorm)
                    {
                        if (resnorm == mgcl::MGCL_L2)
                            res += r[0][ir][jr][kr] * r[0][ir][jr][kr];
                        else if (fabs(r[0][ir][jr][kr]) > res)
                            res = fabs(r[0][ir][jr][kr]);
                    }
                }

        return (returnResidualNorm && resnorm == mgcl::MGCL_L2) ? sqrt(res) : res;
    }

    /* Calculates r = f - A*v using 7-point, 19-point or 27-point stencil of 3D laplacian or a varying stencil.
     * m,n,o is the size of the real grid.
     * v needs to have updated ghost cells if the problem is periodic!
     * moff, noff and ooff can be used to change the size of the grid that the residual shall be calculated for.
     *   Per default only real cells are considered (moff = 0), but with e.g. moff = -1, the first ghost cell border is
     *   considered, too. Analogously, with moff = 1 the outermost set of real cells is ignored. The calculation
     *   of the boundaries is e.g. istart = v.ghosts_m + moff.
     * Based on MultigridEngine::residualSeq as of 17.12.2024.
     */
    template <int w>
    double residualSeq(mgcl_bench::Cuboid4dTempl<w>& f, mgcl_bench::Cuboid4dTempl<w>& v, mgcl_bench::Cuboid4dTempl<w>& r, mgcl::MGCL_RESIDUAL_NORM resnorm,
                       mgcl::MGCL_STENCIL stencilType, double stencilFactor,
                       mgcl::VaryingStencil* stencilValuesCuboid, mgcl::FixedStencil* fixedStencil,
                       bool returnResidualNorm,
                       bool periodic, bool updateGhostsLocally, int moff, int noff, int ooff)
    {
        double res = 0.0;
        double stencilsum = 0;
        double****** stencilValues;
        double**** vraw = v.getData();
        double*** fsRaw;

        // check if off is too small (i.e. start < 0)
        if (moff <= -v.getGhostsM() || noff <= -v.getGhostsN() || ooff <= -v.getGhostsO())
            error("moff, noff and ooff must not be <= -ghosts");

        // check if off is too large (i.e. start > end)
        if (moff * 2 >= v.getM() || noff * 2 >= v.getN() || ooff * 2 >= v.getO())
            error("2*moff, 2*noff and 2*ooff must not be >= m, n or o");

        // check that stencilValues is not null if stencil type is varying
        if (stencilType == mgcl::MGCL_VARYING && stencilValuesCuboid == nullptr)
            error("stencilType is varying but stencilValues is null!");

        if (stencilType == mgcl::MGCL_FIXED && fixedStencil == nullptr)
        {
            error("stencilType is fixed but fixedStencil is null!");
        }

        if (stencilType == mgcl::MGCL_VARYING)
            stencilValues = stencilValuesCuboid->getData();

        if (stencilType == mgcl::MGCL_FIXED)
        {
            fsRaw = fixedStencil->getData();
        }

        int istart_v = v.getGhostsM() + moff;
        int jstart_v = v.getGhostsN() + noff;
        int kstart_v = v.getGhostsO() + ooff;
        int iend_v = v.getMgh() - v.getGhostsM() - moff;
        int jend_v = v.getNgh() - v.getGhostsN() - noff;
        int kend_v = v.getOgh() - v.getGhostsO() - ooff;
        int istart_r = r.getGhostsM() + moff;
        int jstart_r = r.getGhostsN() + noff;
        int kstart_r = r.getGhostsO() + ooff;
        int istart_f = f.getGhostsM() + moff;
        int jstart_f = f.getGhostsN() + noff;
        int kstart_f = f.getGhostsO() + ooff;
        int istart_sv = stencilValuesCuboid ? stencilValuesCuboid->getGhostsM() + moff : 0;
        int jstart_sv = stencilValuesCuboid ? stencilValuesCuboid->getGhostsN() + noff : 0;
        int kstart_sv = stencilValuesCuboid ? stencilValuesCuboid->getGhostsO() + ooff : 0;

        for (int iv = istart_v, ir = istart_r, fi = istart_f, isv = istart_sv; iv < iend_v; iv++, ir++, fi++, isv++)
            for (int jv = jstart_v, jr = jstart_r, fj = jstart_f, jsv = jstart_sv; jv < jend_v; jv++, jr++, fj++, jsv++)
                for (int kv = kstart_v, kr = kstart_r, fk = kstart_f, ksv = kstart_sv; kv < kend_v; kv++, kr++, fk++, ksv++)
                {
                    // A*v
                    if (stencilType == mgcl::MGCL_LAPLACE_7POINT)
                    {
                        // clang-format off
                        stencilsum = (6.0 * vraw[0][iv][jv][kv]
                            - vraw[0][iv][jv][kv - 1] - vraw[0][iv][jv][kv + 1]
                            - vraw[0][iv][jv - 1][kv] - vraw[0][iv][jv + 1][kv]
                            - vraw[0][iv - 1][jv][kv] - vraw[0][iv + 1][jv][kv]
                            ) * stencilFactor;
                        // clang-format on
                    }
                    else if (stencilType == mgcl::MGCL_LAPLACE_19POINT)
                    {
                        // clang-format off
                        stencilsum = (24.0 * vraw[0][iv][jv][kv]
                                - 2.0 * vraw[0][iv][jv][kv - 1] - 2.0 * vraw[0][iv][jv][kv + 1]
                                - 2.0 * vraw[0][iv][jv - 1][kv] - 2.0 * vraw[0][iv][jv + 1][kv]
                                - 2.0 * vraw[0][iv - 1][jv][kv] - 2.0 * vraw[0][iv + 1][jv][kv]
                                
                                - vraw[0][iv][jv - 1][kv - 1] - vraw[0][iv][jv - 1][kv + 1]
                                - vraw[0][iv][jv + 1][kv - 1] - vraw[0][iv][jv + 1][kv + 1]
                                - vraw[0][iv - 1][jv][kv - 1] - vraw[0][iv - 1][jv][kv + 1]
                                - vraw[0][iv + 1][jv][kv - 1] - vraw[0][iv + 1][jv][kv + 1]
                                - vraw[0][iv - 1][jv - 1][kv] - vraw[0][iv - 1][jv + 1][kv]
                                - vraw[0][iv + 1][jv - 1][kv] - vraw[0][iv + 1][jv + 1][kv]
                                ) * stencilFactor;
                        // clang-format on
                    }
                    else if (stencilType == mgcl::MGCL_LAPLACE_27POINT)
                    {
                        // clang-format off
                        stencilsum = (88.0 * vraw[0][iv][jv][kv]
                                - 6.0 * vraw[0][iv][jv][kv - 1] - 6.0 * vraw[0][iv][jv][kv + 1]
                                - 6.0 * vraw[0][iv][jv - 1][kv] - 6.0 * vraw[0][iv][jv + 1][kv]
                                - 6.0 * vraw[0][iv - 1][jv][kv] - 6.0 * vraw[0][iv + 1][jv][kv]

                                - 3.0 * vraw[0][iv][jv - 1][kv - 1] - 3.0 * vraw[0][iv][jv - 1][kv + 1]
                                - 3.0 * vraw[0][iv][jv + 1][kv - 1] - 3.0 * vraw[0][iv][jv + 1][kv + 1]
                                - 3.0 * vraw[0][iv - 1][jv][kv - 1] - 3.0 * vraw[0][iv - 1][jv][kv + 1]
                                - 3.0 * vraw[0][iv + 1][jv][kv - 1] - 3.0 * vraw[0][iv + 1][jv][kv + 1]
                                - 3.0 * vraw[0][iv - 1][jv - 1][kv] - 3.0 * vraw[0][iv - 1][jv + 1][kv]
                                - 3.0 * vraw[0][iv + 1][jv - 1][kv] - 3.0 * vraw[0][iv + 1][jv + 1][kv]

                                - 2.0 * vraw[0][iv - 1][jv - 1][kv - 1] - 2.0 * vraw[0][iv - 1][jv - 1][kv + 1]
                                - 2.0 * vraw[0][iv - 1][jv + 1][kv - 1] - 2.0 * vraw[0][iv - 1][jv + 1][kv + 1]
                                - 2.0 * vraw[0][iv + 1][jv - 1][kv - 1] - 2.0 * vraw[0][iv + 1][jv - 1][kv + 1]
                                - 2.0 * vraw[0][iv + 1][jv + 1][kv - 1] - 2.0 * vraw[0][iv + 1][jv + 1][kv + 1]
                                ) * stencilFactor;
                        // clang-format on
                    }
                    else if (stencilType == mgcl::MGCL_FIXED)
                    {
                        // clang-format off
                        stencilsum = fsRaw[1][1][1]  * vraw[0][iv][jv][kv]
                            + fsRaw[1][1][0] * vraw[0][ iv ][ jv ][kv-1]
                            + fsRaw[1][1][2] * vraw[0][ iv ][ jv ][kv+1]
                            + fsRaw[1][0][1] * vraw[0][ iv ][jv-1][ kv ]
                            + fsRaw[1][2][1] * vraw[0][ iv ][jv+1][ kv ]
                            + fsRaw[0][1][1] * vraw[0][iv-1][ jv ][ kv ]
                            + fsRaw[2][1][1] * vraw[0][iv+1][ jv ][ kv ]
                            
                            + fsRaw[1][0][0] * vraw[0][ iv ][jv-1][kv-1]
                            + fsRaw[1][0][2] * vraw[0][ iv ][jv-1][kv+1]
                            + fsRaw[1][2][0] * vraw[0][ iv ][jv+1][kv-1]
                            + fsRaw[1][2][2] * vraw[0][ iv ][jv+1][kv+1]
                            + fsRaw[0][1][0] * vraw[0][iv-1][ jv ][kv-1]
                            + fsRaw[0][1][2] * vraw[0][iv-1][ jv ][kv+1]
                            + fsRaw[2][1][0] * vraw[0][iv+1][ jv ][kv-1]
                            + fsRaw[2][1][2] * vraw[0][iv+1][ jv ][kv+1]
                            + fsRaw[0][0][1] * vraw[0][iv-1][jv-1][ kv ]
                            + fsRaw[0][2][1] * vraw[0][iv-1][jv+1][ kv ]
                            + fsRaw[2][0][1] * vraw[0][iv+1][jv-1][ kv ]
                            + fsRaw[2][2][1] * vraw[0][iv+1][jv+1][ kv ]
                            
                            + fsRaw[0][0][0] * vraw[0][iv-1][jv-1][kv-1]
                            + fsRaw[0][0][2] * vraw[0][iv-1][jv-1][kv+1]
                            + fsRaw[0][2][0] * vraw[0][iv-1][jv+1][kv-1]
                            + fsRaw[0][2][2] * vraw[0][iv-1][jv+1][kv+1]
                            + fsRaw[2][0][0] * vraw[0][iv+1][jv-1][kv-1]
                            + fsRaw[2][0][2] * vraw[0][iv+1][jv-1][kv+1]
                            + fsRaw[2][2][0] * vraw[0][iv+1][jv+1][kv-1]
                            + fsRaw[2][2][2] * vraw[0][iv+1][jv+1][kv+1];
                        // clang-format on
                    }
                    else if (stencilType == mgcl::MGCL_VARYING)
                    {
                        // clang-format off
                        stencilsum = stencilValues[1][1][1][isv][jsv][ksv]  * vraw[0][iv][jv][kv]
                            + stencilValues[1][1][0][isv][jsv][ksv] * vraw[0][ iv ][ jv ][kv-1]
                            + stencilValues[1][1][2][isv][jsv][ksv] * vraw[0][ iv ][ jv ][kv+1]
                            + stencilValues[1][0][1][isv][jsv][ksv] * vraw[0][ iv ][jv-1][ kv ]
                            + stencilValues[1][2][1][isv][jsv][ksv] * vraw[0][ iv ][jv+1][ kv ]
                            + stencilValues[0][1][1][isv][jsv][ksv] * vraw[0][iv-1][ jv ][ kv ]
                            + stencilValues[2][1][1][isv][jsv][ksv] * vraw[0][iv+1][ jv ][ kv ]
                            
                            + stencilValues[1][0][0][isv][jsv][ksv] * vraw[0][ iv ][jv-1][kv-1]
                            + stencilValues[1][0][2][isv][jsv][ksv] * vraw[0][ iv ][jv-1][kv+1]
                            + stencilValues[1][2][0][isv][jsv][ksv] * vraw[0][ iv ][jv+1][kv-1]
                            + stencilValues[1][2][2][isv][jsv][ksv] * vraw[0][ iv ][jv+1][kv+1]
                            + stencilValues[0][1][0][isv][jsv][ksv] * vraw[0][iv-1][ jv ][kv-1]
                            + stencilValues[0][1][2][isv][jsv][ksv] * vraw[0][iv-1][ jv ][kv+1]
                            + stencilValues[2][1][0][isv][jsv][ksv] * vraw[0][iv+1][ jv ][kv-1]
                            + stencilValues[2][1][2][isv][jsv][ksv] * vraw[0][iv+1][ jv ][kv+1]
                            + stencilValues[0][0][1][isv][jsv][ksv] * vraw[0][iv-1][jv-1][ kv ]
                            + stencilValues[0][2][1][isv][jsv][ksv] * vraw[0][iv-1][jv+1][ kv ]
                            + stencilValues[2][0][1][isv][jsv][ksv] * vraw[0][iv+1][jv-1][ kv ]
                            + stencilValues[2][2][1][isv][jsv][ksv] * vraw[0][iv+1][jv+1][ kv ]
                            
                            + stencilValues[0][0][0][isv][jsv][ksv] * vraw[0][iv-1][jv-1][kv-1]
                            + stencilValues[0][0][2][isv][jsv][ksv] * vraw[0][iv-1][jv-1][kv+1]
                            + stencilValues[0][2][0][isv][jsv][ksv] * vraw[0][iv-1][jv+1][kv-1]
                            + stencilValues[0][2][2][isv][jsv][ksv] * vraw[0][iv-1][jv+1][kv+1]
                            + stencilValues[2][0][0][isv][jsv][ksv] * vraw[0][iv+1][jv-1][kv-1]
                            + stencilValues[2][0][2][isv][jsv][ksv] * vraw[0][iv+1][jv-1][kv+1]
                            + stencilValues[2][2][0][isv][jsv][ksv] * vraw[0][iv+1][jv+1][kv-1]
                            + stencilValues[2][2][2][isv][jsv][ksv] * vraw[0][iv+1][jv+1][kv+1];
                        // clang-format on

                        // if (j == 2 && k == 2 && i == 2)
                        // {
                        //     printf("seq stencilsum = %e\n", stencilsum);
                        //     print27point_sv(v, i, j, k, stencilValuesCuboid, isv, jsv, ksv);
                        // }
                    }

                    // r = f - A*v
                    r[0][ir][jr][kr] = f[0][fi][fj][fk] - stencilsum;

                    if (returnResidualNorm)
                    {
                        if (resnorm == mgcl::MGCL_L2)
                            res += r[0][ir][jr][kr] * r[0][ir][jr][kr];
                        else if (fabs(r[0][ir][jr][kr]) > res)
                            res = fabs(r[0][ir][jr][kr]);
                    }
                }

        return (returnResidualNorm && resnorm == mgcl::MGCL_L2) ? sqrt(res) : res;
    }
}

// Bench residual for Cuboid vs. Cuboid4d having outermost dim set to 1.
// Goal is to check whether there are performance impacts when using Cuboid4d for 3d calculations.
TEST_CASE("residualCuboidVsCuboid4d")
{
    using std::min;

    if (CLI_ARGS::grids.size() == 0 && (CLI_ARGS::gridsMin.size() == 0 || CLI_ARGS::gridsMax.size() == 0))
        throw "Need to specify at least one local grid size, e.g. using --grids 4,8,16 or --gridsMin 4,4,4 AND --gridsMax 32,32,32";

    // build grids to be tested from CLI args
    std::vector<std::vector<int>> gridsTBT;
    for (auto N : CLI_ARGS::grids)
        gridsTBT.push_back({N, N, N});
    if (CLI_ARGS::gridsMin.size() > 0 && CLI_ARGS::gridsMax.size() > 0)
        for (int m = CLI_ARGS::gridsMin[0]; m <= CLI_ARGS::gridsMax[0]; m *= 2)
            for (int n = CLI_ARGS::gridsMin[1]; n <= CLI_ARGS::gridsMax[1]; n *= 2)
                for (int o = CLI_ARGS::gridsMin[2]; o <= CLI_ARGS::gridsMax[2]; o *= 2)
                    gridsTBT.push_back({m, n, o});

    std::vector<bench_util::Result> results;

    mgcl::MGCL_RESIDUAL_NORM resnorm = mgcl::MGCL_L2;
    mgcl::MGCL_STENCIL stencilType = mgcl::MGCL_VARYING;

    int ghosts = 1;
    int moff = 0;
    int noff = 0;
    int ooff = 0;

    for (auto gr : gridsTBT)
    {
        int m = gr[0];
        int n = gr[1];
        int o = gr[2];

        auto sv = std::make_unique<mgcl::VaryingStencil>(m, n, o, 3, ghosts, ghosts, ghosts);
        sv->fillRandom();

        ankerl::nanobench::Bench bench;
        bench.timeUnit(1ms, "ms")
            .epochs(CLI_ARGS::bench_epochs)
            .epochIterations(CLI_ARGS::bench_iterations)
            .relative(false);

        auto r_cuboid3d = std::make_shared<mgcl::Cuboid>(m, n, o, ghosts, ghosts, ghosts);
        auto r_cuboid4dw1 = std::make_shared<mgcl_bench::Cuboid4d>(m, n, o, 1, ghosts, ghosts, ghosts);
        auto r_cuboid4dw1templ = std::make_shared<mgcl_bench::Cuboid4dTempl<1>>(m, n, o, ghosts, ghosts, ghosts);
        if (CLI_ARGS::checkResults)
        {
            bench.epochs(1).epochIterations(1);
        }

        {
            std::string name = std::string("residual_cuboid3d_")
                                   .append(std::to_string(m))
                                   .append("_")
                                   .append(std::to_string(n))
                                   .append("_")
                                   .append(std::to_string(o));

            auto v_in = std::make_shared<mgcl::Cuboid>(m, n, o, ghosts, ghosts, ghosts);
            auto f_in = std::make_shared<mgcl::Cuboid>(m, n, o, ghosts, ghosts, ghosts);
            v_in->fill1dIndex(false);
            f_in->fill1dIndex(false);
            // v_in->fillRandom();
            // f_in->fillRandom();
            // v_in->dumpToFile("v1.txt", false);
            // f_in->dumpToFile("f1.txt", false);

            bench.run(std::string(name).c_str(), [&] { //
                mgcl_bench::residualSeq(*f_in, *v_in, *r_cuboid3d, resnorm, stencilType, 0, sv.get(), nullptr, false, false, true, moff, noff, ooff);
            });

            bench_util::Result res;
            res.name = name;
            res.minTime = bench_util::getMinTime(bench, name);
            res.medianTime = bench_util::getMedianTime(bench, name);
            res.avgTime = bench_util::getAvgTime(bench, name);
            res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
            res.m = m;
            res.n = n;
            res.o = o;
            results.push_back(res);
        }

        {
            std::string name = std::string("residual_cuboid4d_w1_")
                                   .append(std::to_string(m))
                                   .append("_")
                                   .append(std::to_string(n))
                                   .append("_")
                                   .append(std::to_string(o));

            auto v_in = std::make_shared<mgcl_bench::Cuboid4d>(m, n, o, 1, ghosts, ghosts, ghosts);
            auto f_in = std::make_shared<mgcl_bench::Cuboid4d>(m, n, o, 1, ghosts, ghosts, ghosts);
            v_in->fill1dIndex(false);
            f_in->fill1dIndex(false);
            // v_in->fillRandom();
            // f_in->fillRandom();
            // v_in->dumpToFile("v2.txt", false);
            // f_in->dumpToFile("f2.txt", false);

            bench.run(std::string(name).c_str(), [&] { //
                mgcl_bench::residualSeq(*f_in, *v_in, *r_cuboid4dw1, resnorm, stencilType, 0, sv.get(), nullptr, false, false, true, moff, noff, ooff);
            });

            bench_util::Result res;
            res.name = name;
            res.minTime = bench_util::getMinTime(bench, name);
            res.medianTime = bench_util::getMedianTime(bench, name);
            res.avgTime = bench_util::getAvgTime(bench, name);
            res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
            res.m = m;
            res.n = n;
            res.o = o;
            results.push_back(res);
        }

        {
            std::string name = std::string("residual_cuboid4d_w1_templ_")
                                   .append(std::to_string(m))
                                   .append("_")
                                   .append(std::to_string(n))
                                   .append("_")
                                   .append(std::to_string(o));

            auto v_in = std::make_shared<mgcl_bench::Cuboid4dTempl<1>>(m, n, o, ghosts, ghosts, ghosts);
            auto f_in = std::make_shared<mgcl_bench::Cuboid4dTempl<1>>(m, n, o, ghosts, ghosts, ghosts);
            v_in->fill1dIndex(false);
            f_in->fill1dIndex(false);
            // v_in->fillRandom();
            // f_in->fillRandom();
            // v_in->dumpToFile("v2.txt", false);
            // f_in->dumpToFile("f2.txt", false);

            bench.run(std::string(name).c_str(), [&] { //
                mgcl_bench::residualSeq<1>(*f_in, *v_in, *r_cuboid4dw1templ, resnorm, stencilType, 0, sv.get(), nullptr, false, false, true, moff, noff, ooff);
            });

            bench_util::Result res;
            res.name = name;
            res.minTime = bench_util::getMinTime(bench, name);
            res.medianTime = bench_util::getMedianTime(bench, name);
            res.avgTime = bench_util::getAvgTime(bench, name);
            res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
            res.m = m;
            res.n = n;
            res.o = o;
            results.push_back(res);
        }

        if (CLI_ARGS::checkResults)
        {
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    for (int k = 0; k < o; k++)
                    {
                        REQUIRE((*r_cuboid3d)[i][j][k] == (*r_cuboid4dw1)[0][i][j][k]);
                        REQUIRE((*r_cuboid3d)[i][j][k] == (*r_cuboid4dw1templ)[0][i][j][k]);
                    }
        }
    }

    bench_util::printCsvFormat(results);
}
