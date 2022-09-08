#include "hypercube.hpp"

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
    /**
     * @brief Construct a new Hypercube4d object having ghosts = 0 and optionally a value which defaults to 0.
     *
     * @param dim1_ Dimension of real grid.
     * @param dim2_ Dimension of real grid.
     * @param dim3_ Dimension of real grid.
     * @param dim4_ Dimension of real grid.
     * @param value Initial value, defaults to 0.
     */
    Hypercube4d::Hypercube4d(int dim1_, int dim2_, int dim3_, int dim4_, double value)
        : Hypercube4d(dim1_, dim2_, dim3_, dim4_, 0, 0, 0, 0, value) {}

    /**
     * @brief Construct a new ghosted Hypercube4d object, optionally having a value which defaults to 0.
     *
     * @param dim1_ Dimension of real grid.
     * @param dim2_ Dimension of real grid.
     * @param dim3_ Dimension of real grid.
     * @param dim4_ Dimension of real grid.
     * @param ghostsDim1_ Amount of ghost cells in one direction.
     * @param ghostsDim2_ Amount of ghost cells in one direction.
     * @param ghostsDim3_ Amount of ghost cells in one direction.
     * @param ghostsDim4_ Amount of ghost cells in one direction.
     * @param value Initial value, defaults to 0.
     */
    Hypercube4d::Hypercube4d(int dim1_, int dim2_, int dim3_, int dim4_, int ghostsDim1_, int ghostsDim2_, int ghostsDim3_, int ghostsDim4_, double value)
        : dim1(dim1_),
          dim2(dim2_),
          dim3(dim3_),
          dim4(dim4_),
          dim1gh(dim1_ + 2 * ghostsDim1_),
          dim2gh(dim2_ + 2 * ghostsDim2_),
          dim3gh(dim3_ + 2 * ghostsDim3_),
          dim4gh(dim4_ + 2 * ghostsDim4_),
          ghostsDim1(ghostsDim1_),
          ghostsDim2(ghostsDim2_),
          ghostsDim3(ghostsDim3_),
          ghostsDim4(ghostsDim4_)
    {
        field_1d.resize(dim1gh * dim2gh * dim3gh * dim4gh);
        for (int i = 0; i < field_1d.size(); i++)
            field_1d[i] = value;

        field_4d = new double ***[dim1gh];
        for (int i = 0; i < dim1gh; i++)
        {
            field_4d[i] = new double **[dim2gh];
            for (int j = 0; j < dim2gh; j++)
            {
                field_4d[i][j] = new double *[dim3gh];
                for (int k = 0; k < dim3gh; k++)
                {
                    field_4d[i][j][k] = &field_1d[i * dim2gh * dim3gh * dim4gh + j * dim3gh * dim4gh + k * dim4gh];
                }
            }
        }
    }

    /**
     * @brief Copies a Hypercube4d object.
     *
     * @param c Hypercube to clone.
     */
    Hypercube4d::Hypercube4d(const Hypercube4d &c)
        : dim1(c.dim1),
          dim2(c.dim2),
          dim3(c.dim3),
          dim4(c.dim4),
          dim1gh(c.dim1gh),
          dim2gh(c.dim2gh),
          dim3gh(c.dim3gh),
          dim4gh(c.dim4gh),
          ghostsDim1(c.ghostsDim1),
          ghostsDim2(c.ghostsDim2),
          ghostsDim3(c.ghostsDim3),
          ghostsDim4(c.ghostsDim4),
          field_1d(c.field_1d)
    {
        field_4d = new double ***[dim1gh];
        for (int i = 0; i < dim1gh; i++)
        {
            field_4d[i] = new double **[dim2gh];
            for (int j = 0; j < dim2gh; j++)
            {
                field_4d[i][j] = new double *[dim3gh];
                for (int k = 0; k < dim3gh; k++)
                {
                    field_4d[i][j][k] = &field_1d[i * dim2gh * dim3gh * dim4gh + j * dim3gh * dim4gh + k * dim4gh];
                }
            }
        }
    }

    /**
     * @brief Destroy the Hypercube4d object, freeing memory.
     */
    Hypercube4d::~Hypercube4d()
    {
        for (int i = 0; i < dim1gh; i++)
        {
            for (int j = 0; j < dim2gh; j++)
            {
                delete[] field_4d[i][j];
            }
            delete[] field_4d[i];
        }
        delete[] field_4d;
        field_4d = nullptr;
    }

    double ****Hypercube4d::getData() const
    {
        return field_4d;
    }

    double ***Hypercube4d::operator[](int index)
    {
        return field_4d[index];
    }

    /**
     * @brief Fills Hypercube4d with random values between low and high, which default to 0 and 1.
     *
     */
    void Hypercube4d::fillRandom(double low, double high)
    {
        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_real_distribution<double> dist(low, high);

        for (int i = 0; i < field_1d.size(); i++)
            field_1d[i] = dist(rng);
    }

    /**
     * @brief Fills Hypercube4d with given value
     *
     * @param value Value to fill
     * @param realCellsOnly If true, only real cells will be set to value. Defaults to false.
     */
    void Hypercube4d::fill(double value, bool realCellsOnly)
    {
        if (realCellsOnly)
        {
            for (int i = ghostsDim1; i < dim1 + ghostsDim1; i++)
                for (int j = ghostsDim2; j < dim2 + ghostsDim2; j++)
                    for (int k = ghostsDim3; k < dim3 + ghostsDim3; k++)
                        for (int l = ghostsDim4; l < dim4 + ghostsDim4; l++)
                        {
                            field_4d[i][j][k][l] = value;
                        }
        }
        else
        {
            std::fill(field_1d.begin(), field_1d.end(), value);
        }
    }

    std::vector<double> &Hypercube4d::field1d()
    {
        return field_1d;
    }

    /**
     * @brief Returns true if real cells contents of this Hypercube4d is equal to the one of another Hypercube4d c
     * within a given tolerance tol, respecting ghost cell amount. Dimensions of real cell amount of this Hypercube4d
     *  and c must be equal (without ghost cells).
     *
     * @param c Other Hypercube4d
     * @param tol tolerance that is used for checking equality. Defaults to 1e-7.
     * @param printDiffs If true, differences will be printed to standard output. Defaults to true.
     * @return true Cuboids equal.
     * @return false Cuboids not equal.
     * @throws invalid_argument When dimensions of Cuboids don't match.
     */
    bool Hypercube4d::isEqual(Hypercube4d &c, double tol, bool printDiffs)
    {
        if (dim1 != c.getDim1() ||
            dim2 != c.getDim2() ||
            dim3 != c.getDim3() ||
            dim4 != c.getDim4())
        {
            throw std::invalid_argument("Cannot check equality for Hypercube4d. Dimensions differ.");
        }

        std::vector<std::tuple<int, int, int, int, double, double, double>> diffs;
        bool ret = true;
        double diff = 0;

        for (int i = 0; i < dim1; i++)
            for (int j = 0; j < dim2; j++)
                for (int k = 0; k < dim3; k++)
                    for (int l = 0; l < dim4; l++)
                    {
                        diff = fabs(field_4d[i + ghostsDim1][j + ghostsDim2][k + ghostsDim3][l + ghostsDim4] -
                                    c[i + c.getGhostsDim1()][j + c.getGhostsDim2()][k + c.getGhostsDim3()][l + c.getGhostsDim4()]);
                        if (diff > tol)
                        {
                            ret = false;
                            diffs.push_back(std::make_tuple(i, j, k, l,
                                                            field_4d[i + ghostsDim1][j + ghostsDim2][k + ghostsDim3][l + ghostsDim4],
                                                            c[i + c.getGhostsDim1()][j + c.getGhostsDim2()][k + c.getGhostsDim3()][l + c.getGhostsDim4()],
                                                            diff));
                        }
                    }

        if (printDiffs && !diffs.empty())
        {
            std::cout << "Hypercube4ds not equal. Differences: " << std::endl
                      << "   i   j   k   l       this      other   difference" << std::endl;
            for (auto d : diffs)
            {
                std::cout << std::setw(4) << std::get<0>(d)
                          << std::setw(4) << std::get<1>(d)
                          << std::setw(4) << std::get<2>(d)
                          << std::setw(4) << std::get<3>(d)
                          << std::scientific << std::setprecision(3) << std::setw(11) << std::get<4>(d)
                          << std::scientific << std::setprecision(3) << std::setw(11) << std::get<5>(d)
                          << std::scientific << std::setprecision(3) << std::setw(13) << std::get<6>(d)
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
    void Hypercube4d::dumpToFile(std::string path)
    {
        std::ofstream myfile;
        myfile.open(path, std::ios::out | std::ios::trunc);

        if (myfile.is_open())
        {
            for (int i = 0; i < dim1gh; i++)
                for (int j = 0; j < dim2gh; j++)
                    for (int k = 0; k < dim3gh; k++)
                        for (int l = 0; l < dim4gh; l++)
                        {
                            myfile << i << "\t" << j << "\t" << k << "\t" << l << "\t"
                                   << std::scientific << std::setprecision(17) << field_4d[i][j][k][l] << std::endl;
                        }
            myfile.close();
        }
        else
        {
            throw std::runtime_error("Couldn't open file for writing given by: " + path);
        }
    }

    int Hypercube4d::getDim1() const
    {
        return dim1;
    }
    int Hypercube4d::getDim4() const
    {
        return dim4;
    }
    int Hypercube4d::getDim3gh() const
    {
        return dim3gh;
    }
    int Hypercube4d::getGhostsDim2() const
    {
        return ghostsDim2;
    }

    int Hypercube4d::getDim3() const
    {
        return dim3;
    }

    int Hypercube4d::getDim2gh() const
    {
        return dim2gh;
    }

    int Hypercube4d::getGhostsDim1() const
    {
        return ghostsDim1;
    }

    int Hypercube4d::getGhostsDim4() const
    {
        return ghostsDim4;
    }

    int Hypercube4d::getDim2() const
    {
        return dim2;
    }

    int Hypercube4d::getDim1gh() const
    {
        return dim1gh;
    }

    int Hypercube4d::getDim4gh() const
    {
        return dim4gh;
    }

    int Hypercube4d::getGhostsDim3() const
    {
        return ghostsDim3;
    }

}
