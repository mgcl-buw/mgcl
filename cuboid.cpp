#include "cuboid.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <tuple>

namespace mgcl
{

    double ***cuboid_alloc(int m, int n, int o)
    {
        double *field;
        double ***cuboid;
        int i, j;

        field = (double *)calloc(sizeof(double), m * n * o);
        cuboid = (double ***)malloc(sizeof(double **) * m);
        for (i = 0; i < m; i++)
        {
            cuboid[i] = (double **)malloc(sizeof(double *) * n);
            for (j = 0; j < n; j++)
            {
                cuboid[i][j] = &field[i * n * o + j * o];
            }
        }

        return cuboid;
    }

    double ***cube_alloc(int n)
    {
        return cuboid_alloc(n, n, n);
    }

    void cuboid_free(double ***cuboid, int m, int n, int o)
    {
        if (cuboid == NULL)
            return;

        int i;

        free(cuboid[0][0]);
        for (i = 0; i < m; i++)
        {
            free(cuboid[i]);
        }
        free(cuboid);
        cuboid = NULL;

        return;
    }

    void cube_free(double ***cube, int n)
    {
        cuboid_free(cube, n, n, n);

        return;
    }

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

    /**
     * @brief Construct a new Cuboid object allocating memory for field_3d.
     *
     * @param c Cuboid to be cloned from.
     */
    Cuboid::Cuboid(const Cuboid &c)
        : m(c.m), n(c.n), o(c.o), mgh(c.mgh), ngh(c.ngh), ogh(c.ogh), field_1d(c.field_1d)
    {
        field_3d = new double **[mgh];
        for (int i = 0; i < mgh; i++)
        {
            field_3d[i] = new double *[ngh];
            for (int j = 0; j < ngh; j++)
            {
                field_3d[i][j] = &field_1d[i * ngh * ogh + j * ogh];
            }
        }
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

    double **Cuboid::operator[](int index)
    {
        return field_3d[index];
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

    std::vector<double> &Cuboid::field1d()
    {
        return field_1d;
    }

    /**
     * @brief Fills Cuboid with given value
     *
     * @param value Value to fill
     * @param realCellsOnly If true, only real cells will be set to value.
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
     * @brief Returns true if this Cuboid is equal to another Cuboid c within a given tolerance tol, respecting ghost
     * cell amount. Dimensions of real cell amount of this Cuboid and c must be equal (without ghost cells).
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
}
