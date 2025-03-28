#include "cuboid_bs.hpp"

#include "mgcl.hpp"

#include <algorithm> // for fill
#include <cmath>     // for fabs
#include <fstream>   // IWYU pragma: keep
#include <iomanip>   // for operator<<, setw, setprecision
#include <iostream>  // for basic_ostream::operator<<, basic_ostream, opera...
#include <random>    // for mt19937, default_random_engine, uniform_int_dis...
#include <stdexcept> // for invalid_argument, runtime_error
#include <utility>   // for move

namespace mgcl
{
    /**
     * @brief Construct a new CuboidBS object having ghosts = 0 and default value 0.
     *
     * @param m_ Dimension of real grid.
     * @param n_ Dimension of real grid.
     * @param o_ Dimension of real grid.
     * @param value Initial value, defaults to 0.
     */
    CuboidBS::CuboidBS(int m_, int n_, int o_, int blocksize)
        : CuboidBS(m_, n_, o_, 0, 0, 0, blocksize, 0.0)
    {
    }

    /**
     * @brief Construct a new CuboidBS object having ghosts = 0 and the given initial value.
     *
     * @param m_ Dimension of real grid.
     * @param n_ Dimension of real grid.
     * @param o_ Dimension of real grid.
     * @param value Initial value, defaults to 0.
     */
    CuboidBS::CuboidBS(int m_, int n_, int o_, int blocksize, double value)
        : CuboidBS(m_, n_, o_, 0, 0, 0, blocksize, value)
    {
    }

    /**
     * @brief Construct a new CuboidBS object. m, n and o must be size of real grid. ghostsX is amount of ghost cells
     * in one direction. Default value 0.
     *
     * @param m_ Dimension of real grid.
     * @param n_ Dimension of real grid.
     * @param o_ Dimension of real grid.
     * @param ghostsM_ Amount of ghost cells in one direction.
     * @param ghostsN_ Amount of ghost cells in one direction.
     * @param ghostsO_ Amount of ghost cells in one direction.
     * @param value initial value, defaults to 0.
     */
    CuboidBS::CuboidBS(int m_, int n_, int o_, int ghostsM_, int ghostsN_, int ghostsO_, int blocksize)
        : CuboidBS(m_, n_, o_, ghostsM_, ghostsN_, ghostsO_, blocksize, 0.0)
    {
    }

    /**
     * @brief Construct a new CuboidBS object. m, n and o must be size of real grid. ghostsX is amount of ghost cells
     * in one direction. value is optionally initial value, defaults to 0.
     *
     * @param m_ Dimension of real grid.
     * @param n_ Dimension of real grid.
     * @param o_ Dimension of real grid.
     * @param ghostsM_ Amount of ghost cells in one direction.
     * @param ghostsN_ Amount of ghost cells in one direction.
     * @param ghostsO_ Amount of ghost cells in one direction.
     * @param value initial value, defaults to 0.
     */
    CuboidBS::CuboidBS(int m_, int n_, int o_, int ghostsM_, int ghostsN_, int ghostsO_, int blocksize, double value)
        : m(m_),
          n(n_),
          o(o_),
          mgh(m_ + 2 * ghostsM_),
          ngh(n_ + 2 * ghostsN_),
          ogh(o_ + 2 * ghostsO_),
          ghostsM(ghostsM_),
          ghostsN(ghostsN_),
          ghostsO(ghostsO_),
          blocksize(blocksize)
    {
        size_t i, j, k;

        field_1d.resize(blocksize * mgh * ngh * ogh);
        for (i = 0; i < field_1d.size(); i++)
            field_1d[i] = value;

        field_4d = new double***[mgh];
        for (i = 0; i < mgh; i++)
        {
            field_4d[i] = new double**[ngh];
            for (j = 0; j < ngh; j++)
            {
                field_4d[i][j] = new double*[ogh];
                for (k = 0; k < ogh; k++)
                {
                    field_4d[i][j][k] = &field_1d[i * ngh * ogh * blocksize + j * ogh * blocksize + k * blocksize];
                }
            }
        }
    }

    CuboidBS::CuboidBS(CuboidBS&& c)
        : m(c.m),
          n(c.n),
          o(c.o),
          mgh(c.mgh),
          ngh(c.ngh),
          ogh(c.ogh),
          ghostsM(c.ghostsM),
          ghostsN(c.ghostsN),
          ghostsO(c.ghostsO),
          blocksize(c.blocksize),
          field_1d(std::move(c.field_1d)),
          field_4d(c.field_4d)
    {
        c.m = 0;
        c.n = 0;
        c.o = 0;
        c.mgh = 0;
        c.ngh = 0;
        c.ogh = 0;
        c.ghostsM = 0;
        c.ghostsN = 0;
        c.ghostsO = 0;
        c.blocksize = 0;
        c.field_4d = nullptr;
    }

    CuboidBS& CuboidBS::operator=(CuboidBS&& c)
    {
        m = c.m;
        n = c.n;
        o = c.o;
        mgh = c.mgh;
        ngh = c.ngh;
        ogh = c.ogh;
        ghostsM = c.ghostsM;
        ghostsN = c.ghostsN;
        ghostsO = c.ghostsO;
        blocksize = c.blocksize;
        field_1d = std::move(c.field_1d);
        field_4d = c.field_4d;

        c.m = 0;
        c.n = 0;
        c.o = 0;
        c.mgh = 0;
        c.ngh = 0;
        c.ogh = 0;
        c.ghostsM = 0;
        c.ghostsN = 0;
        c.ghostsO = 0;
        c.blocksize = 0;
        c.field_4d = nullptr;
        return *this;
    }

    CuboidBS::~CuboidBS()
    {
        for (int i = 0; i < mgh; i++)
        {
            for (int j = 0; j < ngh; j++)
            {
                delete[] field_4d[i][j];
            }
            delete[] field_4d[i];
        }
        delete[] field_4d;
        field_4d = nullptr;
    }

    /**
     * @brief Fills CuboidBS with random values between low and high, which default to 0 and 1.
     *
     */
    void CuboidBS::fillRandom(double low, double high, bool realCellsOnly)
    {
        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_real_distribution<double> dist(low, high);

        if (realCellsOnly)
        {
            for (int d1 = ghostsM; d1 < m + ghostsM; d1++)
                for (int d2 = ghostsN; d2 < n + ghostsN; d2++)
                    for (int d3 = ghostsM; d3 < o + ghostsO; d3++)
                        for (int d4 = 0; d4 < blocksize; d4++)
                        {
                            field_4d[d1][d2][d3][d4] = dist(rng);
                        }
        }
        else
        {
            for (int i = 0; i < field_1d.size(); i++)
                field_1d[i] = dist(rng);
        }
    }

    /**
     * @brief Fills CuboidBS with random int values between low and high, which default to 1 and 10.
     * @param realCellsOnly if true, only real cells will be filled.
     */
    void CuboidBS::fillRandomInt(int low, int high, bool realCellsOnly)
    {
        // use fixed seed to get same results every run
        std::default_random_engine rng(123);
        std::uniform_int_distribution<int> dist(low, high);

        if (realCellsOnly)
        {
            for (int d1 = ghostsM; d1 < m + ghostsM; d1++)
                for (int d2 = ghostsN; d2 < n + ghostsN; d2++)
                    for (int d3 = ghostsM; d3 < o + ghostsO; d3++)
                        for (int d4 = 0; d4 < blocksize; d4++)
                        {
                            field_4d[d1][d2][d3][d4] = dist(rng);
                        }
        }
        else
        {
            for (int i = 0; i < field_1d.size(); i++)
                field_1d[i] = dist(rng);
        }
    }

    /**
     * @brief Fills CuboidBS with given value
     *
     * @param value Value to fill
     * @param realCellsOnly If true, only real cells will be set to value. Defaults to false.
     */
    void CuboidBS::fill(double value, bool realCellsOnly)
    {
        if (realCellsOnly)
        {
            for (int i = ghostsM; i < m + ghostsM; i++)
                for (int j = ghostsN; j < n + ghostsN; j++)
                    for (int k = ghostsO; k < o + ghostsO; k++)
                        for (int b = 0; b < blocksize; b++)
                        {
                            field_4d[i][j][k][b] = value;
                        }
        }
        else
        {
            std::fill(field_1d.begin(), field_1d.end(), value);
        }
    }

    /**
     * @brief Fills this CuboidBS with the 1d index of the corresponding cell, including ghost cells.
     *
     * @param realCellsOnly If true, only real cells get filled.
     */
    void CuboidBS::fill1dIndex(bool realCellsOnly)
    {
        if (realCellsOnly)
            for (int i = ghostsM; i < m + ghostsM; i++)
                for (int j = ghostsN; j < n + ghostsN; j++)
                    for (int k = ghostsO; k < o + ghostsO; k++)
                        for (int b = 0; b < blocksize; b++)
                        {
                            (*this)[i][j][k][b] = to1dIndex(i, j, k, b);
                        }
        else
            for (int i = 0; i < mgh; i++)
                for (int j = 0; j < ngh; j++)
                    for (int k = 0; k < ogh; k++)
                        for (int b = 0; b < blocksize; b++)
                        {
                            (*this)[i][j][k][b] = to1dIndex(i, j, k, b);
                        }
    }

    /**
     * @brief Returns true if real cells contents of this CuboidBS is equal to the one of another CuboidBS c within a
     * given tolerance tol, respecting ghost cell amount. Dimensions of real cell amount of this CuboidBS and c
     * must be equal (without ghost cells).
     *
     * @param c Other CuboidBS
     * @param tol tolerance that is used for checking equality. Defaults to 1e-7.
     * @return true CuboidBSs equal.
     * @return false CuboidBSs not equal.
     * @throws invalid_argument When dimensions of CuboidBSs don't match.
     */
    bool CuboidBS::isEqual(CuboidBS& c, double tol)
    {
        if (m != c.getM() ||
            n != c.getN() ||
            o != c.getO() ||
            blocksize != c.getBlocksize())
        {
            error(std::invalid_argument("Cannot check equality for CuboidBSs. Dimensions differ."));
        }

        double diff;

        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < o; k++)
                    for (int b = 0; b < blocksize; b++)
                    {
                        diff = fabs(field_4d[i + ghostsM][j + ghostsN][k + ghostsO][b] -
                                    c[i + c.getGhostsM()][j + c.getGhostsN()][k + c.getGhostsO()][b]);
                        if (diff > tol)
                        {
                            return false;
                        }
                    }

        return true;
    }

    /**
     * @brief Returns true if all cells contents of this CuboidBS are equal to the one of another CuboidBS c within a
     * given tolerance tol, not respecting ghost cell amount. Dimensions of ghosted cell amount of this CuboidBS and c
     * must be equal (ghost cell amounts can differ as long as sums are equal).
     *
     * @param c Other CuboidBS
     * @param tol tolerance that is used for checking equality. Defaults to 1e-7.
     * @return true CuboidBSs equal.
     * @return false CuboidBSs not equal.
     * @throws invalid_argument When dimensions of CuboidBSs don't match.
     */
    bool CuboidBS::isEqualAllCells(CuboidBS& c, double tol)
    {
        if (mgh != c.getMgh() ||
            ngh != c.getNgh() ||
            ogh != c.getOgh() ||
            blocksize != c.getBlocksize())
        {
            error(std::invalid_argument("Cannot check equality for CuboidBSs. Dimensions differ."));
        }

        double diff;

        for (int i = 0; i < mgh; i++)
            for (int j = 0; j < ngh; j++)
                for (int k = 0; k < ogh; k++)
                    for (int b = 0; b < blocksize; b++)
                    {
                        diff = fabs(field_4d[i][j][k][b] - c[i][j][k][b]);
                        if (diff > tol)
                        {
                            return false;
                        }
                    }

        return true;
    }

    /**
     * @brief Dumps content to file fiven by path overwriting existing files.
     *
     * @param path Path to file, overwrites existing one.
     * @param realCellsOnly If true, only real cells will be written. Defaults to false.
     * @throws runtime_error When file could not be opened.
     */
    void CuboidBS::dumpToFile(std::string path, bool realCellsOnly)
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
                            for (int b = 0; b < blocksize; b++)
                            {
                                myfile << i - ghostsM << "\t" << j - ghostsN << "\t" << k - ghostsO << "\t" << b << "\t"
                                       << std::scientific << std::setprecision(17) << field_4d[i][j][k][b] << std::endl;
                            }
            }
            else
            {
                for (int i = 0; i < mgh; i++)
                    for (int j = 0; j < ngh; j++)
                        for (int k = 0; k < ogh; k++)
                            for (int b = 0; b < blocksize; b++)
                            {
                                myfile << i << "\t" << j << "\t" << k << "\t" << b << "\t"
                                       << std::scientific << std::setprecision(17) << field_4d[i][j][k][b] << std::endl;
                            }
            }
            myfile.close();
        }
        else
        {
            error(std::runtime_error("Couldn't open file for writing given by: " + path));
        }
    }

    /**
     * @brief Creates a copy of the given CuboidBS and returns it.
     *
     * @param c
     * @return CuboidBS
     */
    CuboidBS CuboidBS::copyFrom(CuboidBS& c)
    {
        CuboidBS ret(c.getM(), c.getN(), c.getO(), c.getGhostsM(), c.getGhostsN(), c.getGhostsO(), c.getBlocksize(), 0);

        for (int i = 0; i < c.getMgh(); i++)
            for (int j = 0; j < c.getNgh(); j++)
                for (int k = 0; k < c.getOgh(); k++)
                    for (int b = 0; b < c.getBlocksize(); b++)
                    {
                        ret[i][j][k][b] = c[i][j][k][b];
                    }

        return ret;
    }

    /**
     * @brief Fills real cells from CuboidBS c. Dimensions of real cells must match. Ghost cells are left untouched.
     *
     * @param c
     */
    void CuboidBS::fillRealFrom(CuboidBS& c)
    {
        if (m != c.getM() || n != c.getN() || o != c.getO())
            error("Dimensions do not match!");

        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < o; k++)
                    for (int b = 0; b < blocksize; b++)
                    {
                        field_4d[i + ghostsM][j + ghostsN][k + ghostsO][b] = c[i + c.getGhostsM()][j + c.getGhostsN()][k + c.getGhostsO()][b];
                    }
    }

    /**
     * @brief Fills all cells from CuboidBS c. Ghosted dimensions of real cells must match. It doesn't matter if ghost
     * cell amount varies as long as the sum fits.
     *
     * @param c
     */
    void CuboidBS::fillAllFrom(CuboidBS& c)
    {
        if (mgh != c.getMgh() || ngh != c.getNgh() || ogh != c.getOgh())
            error("Dimensions do not match!");

        for (int i = 0; i < mgh; i++)
            for (int j = 0; j < ngh; j++)
                for (int k = 0; k < ogh; k++)
                    for (int b = 0; b < blocksize; b++)
                    {
                        field_4d[i][j][k][b] = c[i][j][k][b];
                    }
    }

    /**
     * @brief Creates and returns a slice of this CuboidBS and returns it as a new CuboidBS. Boundaries must fit, else an
     * exception is thrown. Ghost cells are not included, i.e. m_end < this->m must hold.
     * By default the new CuboidBS will have the same ghost cell amount as the original one.
     * Boundaries are 0-based, i.e. both start and end will be included.
     *
     * @return std::unique_ptr<CuboidBS>
     */
    std::unique_ptr<CuboidBS> CuboidBS::slice(int m_start, int m_end, int n_start, int n_end, int o_start, int o_end,
                                              int ghm, int ghn, int gho)
    {
        if (m_start < 0 || n_start < 0 || o_start < 0 ||
            m_end >= m || n_end >= n || o_end >= o)
            error("Boundaries out of range!");

        if (ghm < 0)
            ghm = ghostsM;
        if (ghn < 0)
            ghn = ghostsN;
        if (gho < 0)
            gho = ghostsO;

        auto ret = std::make_unique<CuboidBS>((m_end - m_start) + 1, (n_end - n_start) + 1, (o_end - o_start) + 1,
                                              ghm, ghn, gho, blocksize);
        for (int i = m_start, is = ghm, ib = i + ghostsM; i <= m_end; i++, is++, ib++)
            for (int j = n_start, js = ghn, jb = j + ghostsN; j <= n_end; j++, js++, jb++)
                for (int k = o_start, ks = gho, kb = k + ghostsO; k <= o_end; k++, ks++, kb++)
                    for (int b = 0; b < blocksize; b++)
                    {
                        ret->getData()[is][js][ks][b] = getData()[ib][jb][kb][b];
                    }

        return ret;
    }

    /**
     * @brief Creates and returns a slice of this CuboidBS and returns it as a new CuboidBS. Boundaries must fit, else an
     * exception is thrown. Ghost cells are included, i.e. m_end < this->mgh must hold.
     * The returned CuboidBS has no ghosts cells.
     * Boundaries are 0-based, i.e. both start and end will be included.
     *
     * @return std::unique_ptr<CuboidBS>
     */
    std::unique_ptr<CuboidBS> CuboidBS::sliceIncGhosts(int m_start, int m_end, int n_start, int n_end,
                                                       int o_start, int o_end)
    {
        if (m_start < 0 || n_start < 0 || o_start < 0 ||
            m_end >= mgh || n_end >= ngh || o_end >= ogh)
            error("Boundaries out of range!");

        auto ret = std::make_unique<CuboidBS>((m_end - m_start) + 1, (n_end - n_start) + 1, (o_end - o_start) + 1,
                                              0, 0, 0, blocksize);

        for (int i = m_start, is = i - m_start; i <= m_end; i++, is++)
            for (int j = n_start, js = j - n_start; j <= n_end; j++, js++)
                for (int k = o_start, ks = k - o_start; k <= o_end; k++, ks++)
                    for (int b = 0; b < blocksize; b++)
                    {
                        ret->getData()[is][js][ks][b] = getData()[i][j][k][b];
                    }

        return ret;
    }

    /**
     * @brief Returns a new CuboidBS with the same dimensions as this CuboidBS but without data.
     *
     * @return std::unique_ptr<CuboidBS>
     */
    std::unique_ptr<CuboidBS> CuboidBS::copyShallow()
    {
        return std::make_unique<CuboidBS>(m, n, o, ghostsM, ghostsN, ghostsO, blocksize);
    }

}