#include "cuboid.hpp"

#include <algorithm> // for fill
#include <cmath>     // for fabs
#include <fstream>   // IWYU pragma: keep
#include <iomanip>   // for operator<<, setw, setprecision
#include <iostream>  // for basic_ostream::operator<<, basic_ostream, opera...
#include <random>    // for mt19937, default_random_engine, uniform_int_dis...
#include <stdexcept> // for invalid_argument, runtime_error
#include <tuple>     // for tuple, get, make_tuple
#include <utility>   // for move

namespace mgcl
{
    /**
     * @brief Construct a new Cuboid object having ghosts = 0 and optionally a value which defaults to 0.
     *
     * @param m_ Dimension of real grid.
     * @param n_ Dimension of real grid.
     * @param o_ Dimension of real grid.
     * @param value Initial value, defaults to 0.
     */
    Cuboid::Cuboid(int m_, int n_, int o_, double value)
        : Cuboid(m_, n_, o_, 0, 0, 0, value)
    {
    }

    /**
     * @brief Construct a new Cuboid object. m, n and o must be size of real grid. ghostsX is amount of ghost cells
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
    Cuboid::Cuboid(int m_, int n_, int o_, int ghostsM_, int ghostsN_, int ghostsO_, double value)
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
        int i, j;

        field_1d.resize(mgh * ngh * ogh);
        for (i = 0; i < field_1d.size(); i++)
            field_1d[i] = value;

        field_3d = new double **[mgh];
        for (i = 0; i < mgh; i++)
        {
            field_3d[i] = new double *[ngh];
            for (j = 0; j < ngh; j++)
            {
                field_3d[i][j] = &field_1d[i * ngh * ogh + j * ogh];
            }
        }
    }

    Cuboid::Cuboid(Cuboid &&c)
        : m(c.m),
          n(c.n),
          o(c.o),
          mgh(c.mgh),
          ngh(c.ngh),
          ogh(c.ogh),
          ghostsM(c.ghostsM),
          ghostsN(c.ghostsN),
          ghostsO(c.ghostsO),
          field_1d(std::move(c.field_1d)),
          field_3d(c.field_3d)
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
        c.field_3d = nullptr;
    }

    Cuboid &Cuboid::operator=(Cuboid &&c)
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
        field_1d = std::move(c.field_1d);
        field_3d = c.field_3d;

        c.m = 0;
        c.n = 0;
        c.o = 0;
        c.mgh = 0;
        c.ngh = 0;
        c.ogh = 0;
        c.ghostsM = 0;
        c.ghostsN = 0;
        c.ghostsO = 0;
        c.field_3d = nullptr;
        return *this;
    }

    Cuboid::~Cuboid()
    {
        for (int i = 0; i < mgh; i++)
        {
            delete[] field_3d[i];
        }
        delete[] field_3d;
        field_3d = nullptr;
    }

    int Cuboid::getO() const
    {
        return o;
    }

    int Cuboid::getN() const
    {
        return n;
    }

    double ***Cuboid::getData() const
    {
        return field_3d;
    }

    int Cuboid::getM() const
    {
        return m;
    }

    int Cuboid::getGhostsO() const
    {
        return ghostsO;
    }
    int Cuboid::getGhostsN() const
    {
        return ghostsN;
    }
    int Cuboid::getGhostsM() const
    {
        return ghostsM;
    }
    int Cuboid::getOgh() const
    {
        return ogh;
    }
    int Cuboid::getNgh() const
    {
        return ngh;
    }
    int Cuboid::getMgh() const
    {
        return mgh;
    }

    /**
     * @brief Fills Cuboid with random values between low and high, which default to 0 and 1.
     *
     */
    void Cuboid::fillRandom(double low, double high)
    {
        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_real_distribution<double> dist(low, high);

        for (int i = 0; i < field_1d.size(); i++)
            field_1d[i] = dist(rng);
    }

    /**
     * @brief Fills Cuboid with random int values between low and high, which default to 1 and 10.
     * @param realCellsOnly if true, only real cells will be filled.
     */
    void Cuboid::fillRandomInt(int low, int high, bool realCellsOnly)
    {
        // use fixed seed to get same results every run
        std::default_random_engine rng(123);
        std::uniform_int_distribution<int> dist(low, high);

        if (realCellsOnly)
        {
            for (int d1 = ghostsM; d1 < m + ghostsM; d1++)
                for (int d2 = ghostsN; d2 < n + ghostsN; d2++)
                    for (int d3 = ghostsM; d3 < o + ghostsO; d3++)
                    {
                        field_3d[d1][d2][d3] = dist(rng);
                    }
        }
        else
        {
            for (int i = 0; i < field_1d.size(); i++)
                field_1d[i] = dist(rng);
        }
    }

    std::vector<double> &Cuboid::field1d()
    {
        return field_1d;
    }

    /**
     * @brief Fills Cuboid with given value
     *
     * @param value Value to fill
     * @param realCellsOnly If true, only real cells will be set to value. Defaults to false.
     */
    void Cuboid::fill(double value, bool realCellsOnly)
    {
        if (realCellsOnly)
        {
            for (int i = ghostsM; i < m + ghostsM; i++)
                for (int j = ghostsN; j < n + ghostsN; j++)
                    for (int k = ghostsO; k < o + ghostsO; k++)
                    {
                        field_3d[i][j][k] = value;
                    }
        }
        else
        {
            std::fill(field_1d.begin(), field_1d.end(), value);
        }
    }

    /**
     * @brief Returns true if real cells contents of this Cuboid is equal to the one of another Cuboid c within a
     * given tolerance tol, respecting ghost cell amount. Dimensions of real cell amount of this Cuboid and c
     * must be equal (without ghost cells).
     *
     * @param c Other Cuboid
     * @param tol tolerance that is used for checking equality. Defaults to 1e-7.
     * @param printDiffs If true, differences will be printed to standard output. Defaults to true.
     * @return true Cuboids equal.
     * @return false Cuboids not equal.
     * @throws invalid_argument When dimensions of Cuboids don't match.
     */
    bool Cuboid::isEqual(Cuboid &c, double tol, bool printDiffs)
    {
        if (m != c.getM() ||
            n != c.getN() ||
            o != c.getO())
        {
            throw std::invalid_argument("Cannot check equality for Cuboids. Dimensions differ.");
        }

        std::vector<std::tuple<int, int, int, double, double, double>> diffs;
        bool ret = true;
        double diff = 0;

        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < o; k++)
                {
                    diff = fabs(field_3d[i + ghostsM][j + ghostsN][k + ghostsO] -
                                c[i + c.getGhostsM()][j + c.getGhostsN()][k + c.getGhostsO()]);
                    if (diff > tol)
                    {
                        ret = false;
                        diffs.push_back(std::make_tuple(i, j, k,
                                                        field_3d[i + ghostsM][j + ghostsN][k + ghostsO],
                                                        c[i + c.getGhostsM()][j + c.getGhostsN()][k + c.getGhostsO()],
                                                        diff));
                    }
                }

        if (printDiffs && !diffs.empty())
        {
            std::cout << "Cuboids not equal. Differences: " << std::endl
                      << "   i   j   k       this      other   difference" << std::endl;
            for (auto d : diffs)
            {
                std::cout << std::setw(4) << std::get<0>(d)
                          << std::setw(4) << std::get<1>(d)
                          << std::setw(4) << std::get<2>(d)
                          << std::scientific << std::setprecision(3) << std::setw(11) << std::get<3>(d)
                          << std::scientific << std::setprecision(3) << std::setw(11) << std::get<4>(d)
                          << std::scientific << std::setprecision(3) << std::setw(13) << std::get<5>(d)
                          << std::endl;
            }
        }

        return ret;
    }

    /**
     * @brief Dumps content to file fiven by path overwriting existing files
     *
     * @param path Path to file, overwrites existing one.
     * @throws runtime_error When file could not be opened.
     */
    void Cuboid::dumpToFile(std::string path)
    {
        std::ofstream myfile;
        myfile.open(path, std::ios::out | std::ios::trunc);

        if (myfile.is_open())
        {
            for (int i = 0; i < mgh; i++)
                for (int j = 0; j < ngh; j++)
                    for (int k = 0; k < ogh; k++)
                    {
                        myfile << i << "\t" << j << "\t" << k << "\t"
                               << std::scientific << std::setprecision(17) << field_3d[i][j][k] << std::endl;
                    }
            myfile.close();
        }
        else
        {
            throw std::runtime_error("Couldn't open file for writing given by: " + path);
        }
    }

    /**
     * @brief Creates a copy of the given Cuboid and returns it.
     *
     * @param c
     * @return Cuboid
     */
    Cuboid Cuboid::copyFrom(Cuboid &c)
    {
        Cuboid ret(c.getM(), c.getN(), c.getO(), c.getGhostsM(), c.getGhostsN(), c.getGhostsO(), 0);

        for (int i = 0; i < c.getMgh(); i++)
            for (int j = 0; j < c.getNgh(); j++)
                for (int k = 0; k < c.getOgh(); k++)
                {
                    ret[i][j][k] = c[i][j][k];
                }

        return ret;
    }
}
