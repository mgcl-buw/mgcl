#include "matrix.hpp"
#include "mgcl.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>

namespace mgcl
{

    /**
     * @brief Construct a new Matrix object having ghosts = 0 and optionally a value which defaults to 0.
     *
     * @param dim1_ Dimension of real grid.
     * @param dim2_ Dimension of real grid.
     * @param value Initial value, defaults to 0.
     */
    Matrix::Matrix(int dim1_, int dim2_, double value)
        : Matrix(dim1_, dim2_, 0, 0, value) {}

    /**
     * @brief Construct a new ghosted Matrix object, optionally having a value which defaults to 0.
     *
     * @param dim1_ Dimension of real grid.
     * @param dim2_ Dimension of real grid.
     * @param ghostsDim1_ Amount of ghost cells in one direction.
     * @param ghostsDim2_ Amount of ghost cells in one direction.
     * @param value Initial value, defaults to 0.
     */
    Matrix::Matrix(int dim1_, int dim2_, int ghostsDim1_, int ghostsDim2_, double value)
        : dim1(dim1_),
          dim2(dim2_),
          dim1gh(dim1_ + 2 * ghostsDim1_),
          dim2gh(dim2_ + 2 * ghostsDim2_),
          ghostsDim1(ghostsDim1_),
          ghostsDim2(ghostsDim2_),
          size(dim1gh * dim2gh)
    {
        field_1d.resize(size);
        for (int i = 0; i < field_1d.size(); i++)
            field_1d[i] = value;

        field_2d = new double*[dim1gh];
        for (int d1 = 0; d1 < dim1gh; d1++)
        {
            field_2d[d1] = &field_1d[d1 * dim2gh];
        }
    }
    Matrix::Matrix(Matrix&& o)
        : dim1(o.dim1),
          dim2(o.dim2),
          dim1gh(o.dim1gh),
          dim2gh(o.dim2gh),
          ghostsDim1(o.ghostsDim1),
          ghostsDim2(o.ghostsDim2),
          field_1d(std::move(o.field_1d)),
          field_2d(o.field_2d),
          size(o.size)
    {
        o.dim1 = 0;
        o.dim2 = 0;
        o.dim1gh = 0;
        o.dim2gh = 0;
        o.ghostsDim1 = 0;
        o.ghostsDim2 = 0;
        o.field_2d = nullptr;
        o.size = 0;
    }

    Matrix& Matrix::operator=(Matrix&& o)
    {
        dim1 = o.dim1;
        dim2 = o.dim2;
        dim1gh = o.dim1gh;
        dim2gh = o.dim2gh;
        ghostsDim1 = o.ghostsDim1;
        ghostsDim2 = o.ghostsDim2;
        field_1d = std::move(o.field_1d);
        field_2d = o.field_2d;
        size = o.size;

        o.dim1 = 0;
        o.dim2 = 0;
        o.dim1gh = 0;
        o.dim2gh = 0;
        o.ghostsDim1 = 0;
        o.ghostsDim2 = 0;
        o.field_2d = nullptr;
        o.size = 0;
        return *this;
    }

    /**
     * @brief Destroy the Matrix object, freeing memory.
     */
    Matrix::~Matrix()
    {
        if (field_2d)
        {
            delete[] field_2d;
            field_2d = nullptr;
        }
    }

    /**
     * @brief Fills Matrix with random values between low and high, which default to 0 and 1.
     *
     */
    void Matrix::fillRandom(double low, double high)
    {
        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_real_distribution<double> dist(low, high);

        for (int i = 0; i < field_1d.size(); i++)
            field_1d[i] = dist(rng);
    }

    void Matrix::fillRandomInt(int low, int high, bool realCellsOnly)
    {
        // use fixed seed to get same results every run
        std::default_random_engine rng(123);
        std::uniform_int_distribution<int> dist(low, high);

        if (realCellsOnly)
        {
            for (int d1 = ghostsDim1; d1 < dim1 + ghostsDim1; d1++)
                for (int d2 = ghostsDim2; d2 < dim2 + ghostsDim2; d2++)
                {
                    field_2d[d1][d2] = dist(rng);
                }
        }
        else
        {
            for (int i = 0; i < field_1d.size(); i++)
                field_1d[i] = dist(rng);
        }
    }

    /**
     * @brief Fills Matrix with given value
     *
     * @param value Value to fill
     * @param realCellsOnly If true, only real cells will be set to value. Defaults to false.
     */
    void Matrix::fill(double value, bool realCellsOnly)
    {
        if (realCellsOnly)
        {
            for (int d1 = ghostsDim1; d1 < dim1 + ghostsDim1; d1++)
                for (int d2 = ghostsDim2; d2 < dim2 + ghostsDim2; d2++)
                {
                    field_2d[d1][d2] = value;
                }
        }
        else
        {
            std::fill(field_1d.begin(), field_1d.end(), value);
        }
    }

    /**
     * @brief Fills this Matrix with the 1d index of the corresponding cell, including ghost cells.
     *
     * @param realCellsOnly If true, only real cells get filled.
     */
    void Matrix::fill1dIndex(bool realCellsOnly)
    {
        if (realCellsOnly)
            for (int d1 = ghostsDim1; d1 < dim1 + ghostsDim1; d1++)
                for (int d2 = ghostsDim2; d2 < dim2 + ghostsDim2; d2++)
                {
                    (*this)[d1][d2] = d1 * dim2gh + d2;
                }
        else
            for (int d1 = 0; d1 < dim1gh; d1++)
                for (int d2 = 0; d2 < dim2gh; d2++)
                {
                    (*this)[d1][d2] = d1 * dim2gh + d2;
                }
    }

    /**
     * @brief Returns true if real cells contents of this Matrix is equal to the one of another Matrix c
     * within a given tolerance tol, respecting ghost cell amount. Dimensions of real cell amount of this Matrix
     *  and c must be equal (without ghost cells).
     *
     * @param c Other Matrix
     * @param tol tolerance that is used for checking equality. Defaults to 1e-7.
     * @param printDiffs If true, differences will be printed to standard output. Defaults to true.
     * @return true Cuboids equal.
     * @return false Cuboids not equal.
     * @throws invalid_argument When dimensions of Cuboids don't match.
     */
    bool Matrix::isEqual(Matrix& c, double tol)
    {
        if (dim1 != c.getDim1() ||
            dim2 != c.getDim2())
        {
            error("Dimensions of Matrices don't match.");
        }

        double diff;
        for (int d1 = 0; d1 < dim1; d1++)
            for (int d2 = 0; d2 < dim2; d2++)
            {
                diff = fabs(field_2d[d1 + ghostsDim1][d2 + ghostsDim2] -
                            c[d1 + c.getGhostsDim1()][d2 + c.getGhostsDim2()]);
                if (diff > tol)
                {
                    return false;
                }
            }

        return true;
    }

    /**
     * @brief Returns true if real cells contents of this Matrix is equal to the one of another Matrix c
     * within a given tolerance tol, respecting ghost cell amount. Dimensions of real cell amount of this Matrix
     *  and c must be equal (without ghost cells).
     *
     * @param c Other Matrix
     * @param tol tolerance that is used for checking equality. Defaults to 1e-7.
     * @param printDiffs If true, differences will be printed to standard output. Defaults to true.
     * @return true Cuboids equal.
     * @return false Cuboids not equal.
     * @throws invalid_argument When dimensions of Cuboids don't match.
     */
    bool Matrix::isEqualIncGhosts(Matrix& c, double tol)
    {
        if (dim1gh != c.getDim1gh() ||
            dim2gh != c.getDim2gh())
        {
            error("Dimensions of Matrices don't match.");
        }

        double diff;
        for (int d1 = 0; d1 < dim1gh; d1++)
            for (int d2 = 0; d2 < dim2gh; d2++)
            {
                diff = fabs(field_2d[d1][d2] - c[d1][d2]);
                if (diff > tol)
                {
                    return false;
                }
            }

        return true;
    }

    /**
     * @brief Dumps content to file fiven by path overwriting existing files
     *
     * @param path Path to file, overwrites existing one.
     * @throws runtime_error When file could not be opened.
     */
    void Matrix::dumpToFile(std::string path, bool realCellsOnly)
    {
        std::ofstream myfile;
        myfile.open(path, std::ios::out | std::ios::trunc);

        if (myfile.is_open())
        {
            if (realCellsOnly)
            {
                for (int d1 = ghostsDim1; d1 < dim1 + ghostsDim1; d1++)
                    for (int d2 = ghostsDim2; d2 < dim2 + ghostsDim2; d2++)
                    {
                        myfile << d1 << "\t" << d2 << "\t" << std::scientific << std::setprecision(17) << field_2d[d1][d2] << std::endl;
                    }
            }
            else
            {
                for (int d1 = 0; d1 < dim1gh; d1++)
                    for (int d2 = 0; d2 < dim2gh; d2++)
                    {
                        myfile << d1 << "\t" << d2 << "\t" << std::scientific << std::setprecision(17) << field_2d[d1][d2] << std::endl;
                    }
            }
            myfile.close();
        }
        else
        {
            error(std::runtime_error("Couldn't open file for writing given by: " + path));
        }
    }

    void Matrix::dumpToFileMatlab(std::string path, std::string varname, bool realCellsOnly)
    {
        std::ofstream myfile;
        myfile.open(path, std::ios::out | std::ios::trunc);

        if (myfile.is_open())
        {
            if (realCellsOnly)
            {
                myfile << varname << " = zeros(" << dim1 << ","
                       << dim2 << ");" << std::endl;
                for (int d1 = ghostsDim1; d1 < dim1 + ghostsDim1; d1++)
                    for (int d2 = ghostsDim2; d2 < dim2 + ghostsDim2; d2++)
                    {
                        myfile << varname << "(" << d1 + 1 << "," << d2 + 1
                               << ")="
                               << std::scientific << std::setprecision(17) << field_2d[d1][d2] << ";" << std::endl;
                    }
            }
            else
            {
                myfile << varname << " = zeros(" << dim1gh << ","
                       << dim2gh << ");" << std::endl;
                for (int d1 = 0; d1 < dim1gh; d1++)
                    for (int d2 = 0; d2 < dim2gh; d2++)
                    {
                        myfile << varname << "(" << d1 + 1 << "," << d2 + 1
                               << ")="
                               << std::scientific << std::setprecision(17) << field_2d[d1][d2] << ";" << std::endl;
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
     * @brief Copies contents of real cells of another Matrix o to real cells of this one.
     *
     * @param o
     */
    void Matrix::copyRealFrom(Matrix& o)
    {
        // TODO check ranges
        if (dim1 != o.dim1 || dim2 != o.dim2)
            error("Dimensions must match!");

        // clang-format off
        for (int i = ghostsDim1, id = o.ghostsDim1; i < dim1 + ghostsDim1; i++, id++)
        for (int j = ghostsDim2, jd = o.ghostsDim2; j < dim2 + ghostsDim2; j++, jd++)
            {
                (*this)[i][j] = o[id][jd];
            }
        // clang-format on
    }
}