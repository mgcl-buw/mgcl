#include "hypercube.hpp"

#include "mgcl.hpp"

#include <cmath>
#include <fstream> // IWYU pragma: keep
#include <iomanip>
#include <iostream>
#include <random>
#include <tuple>

#include <algorithm> // for fill
#include <stdexcept> // for runtime_error, invalid_argument
#include <utility>   // for move

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

        field_4d = new double***[dim1gh];
        for (int i = 0; i < dim1gh; i++)
        {
            field_4d[i] = new double**[dim2gh];
            for (int j = 0; j < dim2gh; j++)
            {
                field_4d[i][j] = new double*[dim3gh];
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
    // Hypercube4d::Hypercube4d(const Hypercube4d &c)
    //     : dim1(c.dim1),
    //       dim2(c.dim2),
    //       dim3(c.dim3),
    //       dim4(c.dim4),
    //       dim1gh(c.dim1gh),
    //       dim2gh(c.dim2gh),
    //       dim3gh(c.dim3gh),
    //       dim4gh(c.dim4gh),
    //       ghostsDim1(c.ghostsDim1),
    //       ghostsDim2(c.ghostsDim2),
    //       ghostsDim3(c.ghostsDim3),
    //       ghostsDim4(c.ghostsDim4),
    //       field_1d(c.field_1d)
    // {
    //     field_4d = new double ***[dim1gh];
    //     for (int i = 0; i < dim1gh; i++)
    //     {
    //         field_4d[i] = new double **[dim2gh];
    //         for (int j = 0; j < dim2gh; j++)
    //         {
    //             field_4d[i][j] = new double *[dim3gh];
    //             for (int k = 0; k < dim3gh; k++)
    //             {
    //                 field_4d[i][j][k] = &field_1d[i * dim2gh * dim3gh * dim4gh + j * dim3gh * dim4gh + k * dim4gh];
    //             }
    //         }
    //     }
    // }

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

    double**** Hypercube4d::getData() const
    {
        return field_4d;
    }

    double*** Hypercube4d::operator[](int index)
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

    std::vector<double>& Hypercube4d::field1d()
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
    bool Hypercube4d::isEqual(Hypercube4d& c, double tol, bool printDiffs)
    {
        if (dim1 != c.getDim1() ||
            dim2 != c.getDim2() ||
            dim3 != c.getDim3() ||
            dim4 != c.getDim4())
        {
            error(std::invalid_argument("Cannot check equality for Hypercube4d. Dimensions differ."));
        }

        std::vector<std::tuple<int, int, int, int, double, double, double>> diffs;
        bool ret = true;
        double diff;

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
            error(std::runtime_error("Couldn't open file for writing given by: " + path));
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

    /****************************************************************
     * Hypercube6d
     * **************************************************************/

    /**
     * @brief Construct a new Hypercube6d object having ghosts = 0 and optionally a value which defaults to 0.
     *
     * @param dim1_ Dimension of real grid.
     * @param dim2_ Dimension of real grid.
     * @param dim3_ Dimension of real grid.
     * @param dim4_ Dimension of real grid.
     * @param value Initial value, defaults to 0.
     */
    Hypercube6d::Hypercube6d(int dim1_, int dim2_, int dim3_, int dim4_, int dim5_, int dim6_, double value)
        : Hypercube6d(dim1_, dim2_, dim3_, dim4_, dim5_, dim6_, 0, 0, 0, 0, 0, 0, value) {}

    /**
     * @brief Construct a new ghosted Hypercube6d object, optionally having a value which defaults to 0.
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
    Hypercube6d::Hypercube6d(int dim1_, int dim2_, int dim3_, int dim4_, int dim5_, int dim6_,
                             int ghostsDim1_, int ghostsDim2_, int ghostsDim3_, int ghostsDim4_, int ghostsDim5_, int ghostsDim6_, double value)
        : dim1(dim1_),
          dim2(dim2_),
          dim3(dim3_),
          dim4(dim4_),
          dim5(dim5_),
          dim6(dim6_),
          dim1gh(dim1_ + 2 * ghostsDim1_),
          dim2gh(dim2_ + 2 * ghostsDim2_),
          dim3gh(dim3_ + 2 * ghostsDim3_),
          dim4gh(dim4_ + 2 * ghostsDim4_),
          dim5gh(dim5_ + 2 * ghostsDim5_),
          dim6gh(dim6_ + 2 * ghostsDim6_),
          ghostsDim1(ghostsDim1_),
          ghostsDim2(ghostsDim2_),
          ghostsDim3(ghostsDim3_),
          ghostsDim4(ghostsDim4_),
          ghostsDim5(ghostsDim5_),
          ghostsDim6(ghostsDim6_)
    {
        field_1d.resize(dim1gh * dim2gh * dim3gh * dim4gh * dim5gh * dim6gh);
        for (int i = 0; i < field_1d.size(); i++)
            field_1d[i] = value;

        field_6d = new double*****[dim1gh];
        int size5gh = dim6gh;
        int size4gh = size5gh * dim5gh;
        int size3gh = size4gh * dim4gh;
        int size2gh = size3gh * dim3gh;
        int size1gh = size2gh * dim2gh;
        for (int d1 = 0; d1 < dim1gh; d1++)
        {
            field_6d[d1] = new double****[dim2gh];
            for (int d2 = 0; d2 < dim2gh; d2++)
            {
                field_6d[d1][d2] = new double***[dim3gh];
                for (int d3 = 0; d3 < dim3gh; d3++)
                {
                    field_6d[d1][d2][d3] = new double**[dim4gh];
                    for (int d4 = 0; d4 < dim4gh; d4++)
                    {
                        field_6d[d1][d2][d3][d4] = new double*[dim5gh];
                        for (int d5 = 0; d5 < dim5gh; d5++)
                        {
                            field_6d[d1][d2][d3][d4][d5] = &field_1d[d1 * size1gh + d2 * size2gh + d3 * size3gh + d4 * size4gh + d5 * size5gh];
                        }
                    }
                }
            }
        }
    }

    Hypercube6d::Hypercube6d(Hypercube6d&& o)
        : dim1(o.dim1),
          dim2(o.dim2),
          dim3(o.dim3),
          dim4(o.dim4),
          dim5(o.dim5),
          dim6(o.dim6),
          dim1gh(o.dim1gh),
          dim2gh(o.dim2gh),
          dim3gh(o.dim3gh),
          dim4gh(o.dim4gh),
          dim5gh(o.dim5gh),
          dim6gh(o.dim6gh),
          ghostsDim1(o.ghostsDim1),
          ghostsDim2(o.ghostsDim2),
          ghostsDim3(o.ghostsDim3),
          ghostsDim4(o.ghostsDim4),
          ghostsDim5(o.ghostsDim5),
          ghostsDim6(o.ghostsDim6),
          field_1d(std::move(o.field_1d)),
          field_6d(o.field_6d)
    {
        o.dim1 = 0;
        o.dim2 = 0;
        o.dim3 = 0;
        o.dim4 = 0;
        o.dim5 = 0;
        o.dim6 = 0;
        o.dim1gh = 0;
        o.dim2gh = 0;
        o.dim3gh = 0;
        o.dim4gh = 0;
        o.dim5gh = 0;
        o.dim6gh = 0;
        o.ghostsDim1 = 0;
        o.ghostsDim2 = 0;
        o.ghostsDim3 = 0;
        o.ghostsDim4 = 0;
        o.ghostsDim5 = 0;
        o.ghostsDim6 = 0;
        o.field_6d = nullptr;
    }

    Hypercube6d& Hypercube6d::operator=(Hypercube6d&& o)
    {
        dim1 = o.dim1;
        dim2 = o.dim2;
        dim3 = o.dim3;
        dim4 = o.dim4;
        dim5 = o.dim5;
        dim6 = o.dim6;
        dim1gh = o.dim1gh;
        dim2gh = o.dim2gh;
        dim3gh = o.dim3gh;
        dim4gh = o.dim4gh;
        dim5gh = o.dim5gh;
        dim6gh = o.dim6gh;
        ghostsDim1 = o.ghostsDim1;
        ghostsDim2 = o.ghostsDim2;
        ghostsDim3 = o.ghostsDim3;
        ghostsDim4 = o.ghostsDim4;
        ghostsDim5 = o.ghostsDim5;
        ghostsDim6 = o.ghostsDim6;
        field_1d = std::move(o.field_1d);
        field_6d = o.field_6d;

        o.dim1 = 0;
        o.dim2 = 0;
        o.dim3 = 0;
        o.dim4 = 0;
        o.dim5 = 0;
        o.dim6 = 0;
        o.dim1gh = 0;
        o.dim2gh = 0;
        o.dim3gh = 0;
        o.dim4gh = 0;
        o.dim5gh = 0;
        o.dim6gh = 0;
        o.ghostsDim1 = 0;
        o.ghostsDim2 = 0;
        o.ghostsDim3 = 0;
        o.ghostsDim4 = 0;
        o.ghostsDim5 = 0;
        o.ghostsDim6 = 0;
        o.field_6d = nullptr;
        return *this;
    }

    /**
     * @brief Destroy the Hypercube6d object, freeing memory.
     */
    Hypercube6d::~Hypercube6d()
    {
        if (field_6d)
        {
            for (int d1 = 0; d1 < dim1gh; d1++)
            {
                for (int d2 = 0; d2 < dim2gh; d2++)
                {
                    for (int d3 = 0; d3 < dim3gh; d3++)
                    {
                        for (int d4 = 0; d4 < dim4gh; d4++)
                        {
                            delete[] field_6d[d1][d2][d3][d4];
                        }
                        delete[] field_6d[d1][d2][d3];
                    }
                    delete[] field_6d[d1][d2];
                }
                delete[] field_6d[d1];
            }
            delete[] field_6d;
            field_6d = nullptr;
        }
    }

    double****** Hypercube6d::getData() const
    {
        return field_6d;
    }

    double***** Hypercube6d::operator[](int index) const
    {
        return field_6d[index];
    }

    /**
     * @brief Fills Hypercube6d with random values between low and high, which default to 0 and 1.
     *
     */
    void Hypercube6d::fillRandom(double low, double high)
    {
        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_real_distribution<double> dist(low, high);

        for (int i = 0; i < field_1d.size(); i++)
            field_1d[i] = dist(rng);
    }

    void Hypercube6d::fillRandomInt(int low, int high, bool realCellsOnly)
    {
        // use fixed seed to get same results every run
        std::default_random_engine rng(123);
        std::uniform_int_distribution<int> dist(low, high);

        if (realCellsOnly)
        {
            for (int d1 = ghostsDim1; d1 < dim1 + ghostsDim1; d1++)
                for (int d2 = ghostsDim2; d2 < dim2 + ghostsDim2; d2++)
                    for (int d3 = ghostsDim3; d3 < dim3 + ghostsDim3; d3++)
                        for (int d4 = ghostsDim4; d4 < dim4 + ghostsDim4; d4++)
                            for (int d5 = ghostsDim5; d5 < dim5 + ghostsDim5; d5++)
                                for (int d6 = ghostsDim6; d6 < dim6 + ghostsDim6; d6++)
                                {
                                    field_6d[d1][d2][d3][d4][d5][d6] = dist(rng);
                                }
        }
        else
        {
            for (int i = 0; i < field_1d.size(); i++)
                field_1d[i] = dist(rng);
        }
    }

    /**
     * @brief Fills Hypercube6d with given value
     *
     * @param value Value to fill
     * @param realCellsOnly If true, only real cells will be set to value. Defaults to false.
     */
    void Hypercube6d::fill(double value, bool realCellsOnly)
    {
        if (realCellsOnly)
        {
            for (int d1 = ghostsDim1; d1 < dim1 + ghostsDim1; d1++)
                for (int d2 = ghostsDim2; d2 < dim2 + ghostsDim2; d2++)
                    for (int d3 = ghostsDim3; d3 < dim3 + ghostsDim3; d3++)
                        for (int d4 = ghostsDim4; d4 < dim4 + ghostsDim4; d4++)
                            for (int d5 = ghostsDim5; d5 < dim5 + ghostsDim5; d5++)
                                for (int d6 = ghostsDim6; d6 < dim6 + ghostsDim6; d6++)
                                {
                                    field_6d[d1][d2][d3][d4][d5][d6] = value;
                                }
        }
        else
        {
            std::fill(field_1d.begin(), field_1d.end(), value);
        }
    }

    /**
     * @brief Fills this Hypercube6d with the 1d index of the corresponding cell, including ghost cells.
     *
     * @param realCellsOnly If true, only real cells get filled.
     */
    void Hypercube6d::fill1dIndex(bool realCellsOnly)
    {
        if (realCellsOnly)
            for (int d1 = ghostsDim1; d1 < dim1 + ghostsDim1; d1++)
                for (int d2 = ghostsDim2; d2 < dim2 + ghostsDim2; d2++)
                    for (int d3 = ghostsDim3; d3 < dim3 + ghostsDim3; d3++)
                        for (int d4 = ghostsDim4; d4 < dim4 + ghostsDim4; d4++)
                            for (int d5 = ghostsDim5; d5 < dim5 + ghostsDim5; d5++)
                                for (int d6 = ghostsDim6; d6 < dim6 + ghostsDim6; d6++)
                                {
                                    (*this)[d1][d2][d3][d4][d5][d6] =
                                        d1 * dim2gh * dim3gh * dim4gh * dim5gh * dim6gh +
                                        d2 * dim3gh * dim4gh * dim5gh * dim6gh +
                                        d3 * dim4gh * dim5gh * dim6gh +
                                        d4 * dim5gh * dim6gh + d5 * dim6gh + d6;
                                }
        else
            for (int d1 = 0; d1 < dim1gh; d1++)
                for (int d2 = 0; d2 < dim2gh; d2++)
                    for (int d3 = 0; d3 < dim3gh; d3++)
                        for (int d4 = 0; d4 < dim4gh; d4++)
                            for (int d5 = 0; d5 < dim5gh; d5++)
                                for (int d6 = 0; d6 < dim6gh; d6++)
                                {
                                    (*this)[d1][d2][d3][d4][d5][d6] =
                                        d1 * dim2gh * dim3gh * dim4gh * dim5gh * dim6gh +
                                        d2 * dim3gh * dim4gh * dim5gh * dim6gh +
                                        d3 * dim4gh * dim5gh * dim6gh +
                                        d4 * dim5gh * dim6gh + d5 * dim6gh + d6;
                                }
    }

    std::vector<double>& Hypercube6d::field1d()
    {
        return field_1d;
    }

    /**
     * @brief Returns true if real cells contents of this Hypercube6d is equal to the one of another Hypercube6d c
     * within a given tolerance tol, respecting ghost cell amount. Dimensions of real cell amount of this Hypercube6d
     *  and c must be equal (without ghost cells).
     *
     * @param c Other Hypercube6d
     * @param tol tolerance that is used for checking equality. Defaults to 1e-7.
     * @param printDiffs If true, differences will be printed to standard output. Defaults to true.
     * @return true Cuboids equal.
     * @return false Cuboids not equal.
     * @throws invalid_argument When dimensions of Cuboids don't match.
     */
    bool Hypercube6d::isEqual(Hypercube6d& c, double tol, bool printDiffs)
    {
        if (dim1 != c.getDim1() ||
            dim2 != c.getDim2() ||
            dim3 != c.getDim3() ||
            dim4 != c.getDim4() ||
            dim5 != c.getDim5() ||
            dim6 != c.getDim6())
        {
            error(std::invalid_argument("Cannot check equality for Hypercube6d. Dimensions differ."));
        }

        std::vector<std::tuple<int, int, int, int, int, int, double, double, double>> diffs;
        bool ret = true;
        double diff;

        for (int d1 = 0; d1 < dim1; d1++)
            for (int d2 = 0; d2 < dim2; d2++)
                for (int d3 = 0; d3 < dim3; d3++)
                    for (int d4 = 0; d4 < dim4; d4++)
                        for (int d5 = 0; d5 < dim5; d5++)
                            for (int d6 = 0; d6 < dim6; d6++)
                            {
                                diff = fabs(field_6d[d1 + ghostsDim1][d2 + ghostsDim2][d3 + ghostsDim3][d4 + ghostsDim4][d5 + ghostsDim5][d6 + ghostsDim6] -
                                            c[d1 + c.getGhostsDim1()][d2 + c.getGhostsDim2()][d3 + c.getGhostsDim3()][d4 + c.getGhostsDim4()][d5 + c.getGhostsDim5()][d6 + c.getGhostsDim6()]);
                                if (diff > tol)
                                {
                                    ret = false;
                                    diffs.push_back(std::make_tuple(d1, d2, d3, d4, d5, d6,
                                                                    field_6d[d1 + ghostsDim1][d2 + ghostsDim2][d3 + ghostsDim3][d4 + ghostsDim4][d5 + ghostsDim5][d6 + ghostsDim6],
                                                                    c[d1 + c.getGhostsDim1()][d2 + c.getGhostsDim2()][d3 + c.getGhostsDim3()][d4 + c.getGhostsDim4()][d5 + c.getGhostsDim5()][d6 + c.getGhostsDim6()],
                                                                    diff));
                                }
                            }

        if (printDiffs && !diffs.empty())
        {
            std::cout << "Hypercube4ds not equal. Differences: " << std::endl
                      << "   d1  d2  d3  d4  d5  d6       this      other   difference" << std::endl;
            for (auto d : diffs)
            {
                std::cout << std::setw(4) << std::get<0>(d)
                          << std::setw(4) << std::get<1>(d)
                          << std::setw(4) << std::get<2>(d)
                          << std::setw(4) << std::get<3>(d)
                          << std::setw(4) << std::get<4>(d)
                          << std::setw(4) << std::get<5>(d)
                          << std::scientific << std::setprecision(3) << std::setw(11) << std::get<6>(d)
                          << std::scientific << std::setprecision(3) << std::setw(11) << std::get<7>(d)
                          << std::scientific << std::setprecision(3) << std::setw(13) << std::get<8>(d)
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
    void Hypercube6d::dumpToFile(std::string path, bool realCellsOnly)
    {
        std::ofstream myfile;
        myfile.open(path, std::ios::out | std::ios::trunc);

        if (myfile.is_open())
        {
            if (realCellsOnly)
            {
                for (int d1 = ghostsDim1; d1 < dim1 + ghostsDim1; d1++)
                    for (int d2 = ghostsDim2; d2 < dim2 + ghostsDim2; d2++)
                        for (int d3 = ghostsDim3; d3 < dim3 + ghostsDim3; d3++)
                            for (int d4 = ghostsDim4; d4 < dim4 + ghostsDim4; d4++)
                                for (int d5 = ghostsDim5; d5 < dim5 + ghostsDim5; d5++)
                                    for (int d6 = ghostsDim6; d6 < dim6 + ghostsDim6; d6++)
                                    {
                                        myfile << d1 << "\t" << d2 << "\t" << d3 << "\t" << d4 << "\t" << d5 << "\t" << d6 << "\t"
                                               << std::scientific << std::setprecision(17) << field_6d[d1][d2][d3][d4][d5][d6] << std::endl;
                                    }
            }
            else
            {
                for (int d1 = 0; d1 < dim1gh; d1++)
                    for (int d2 = 0; d2 < dim2gh; d2++)
                        for (int d3 = 0; d3 < dim3gh; d3++)
                            for (int d4 = 0; d4 < dim4gh; d4++)
                                for (int d5 = 0; d5 < dim5gh; d5++)
                                    for (int d6 = 0; d6 < dim6gh; d6++)
                                    {
                                        myfile << d1 << "\t" << d2 << "\t" << d3 << "\t" << d4 << "\t" << d5 << "\t" << d6 << "\t"
                                               << std::scientific << std::setprecision(17) << field_6d[d1][d2][d3][d4][d5][d6] << std::endl;
                                    }
            }
            myfile.close();
        }
        else
        {
            error(std::runtime_error("Couldn't open file for writing given by: " + path));
        }
    }

    void Hypercube6d::dumpToFileMatlab(std::string path, std::string varname, bool realCellsOnly)
    {
        std::ofstream myfile;
        myfile.open(path, std::ios::out | std::ios::trunc);

        if (myfile.is_open())
        {
            if (realCellsOnly)
            {
                myfile << varname << " = zeros(" << dim1 << ","
                       << dim2 << ","
                       << dim3 << ","
                       << dim4 << ","
                       << dim5 << ","
                       << dim6 << ");" << std::endl;
                for (int d1 = ghostsDim1; d1 < dim1 + ghostsDim1; d1++)
                    for (int d2 = ghostsDim2; d2 < dim2 + ghostsDim2; d2++)
                        for (int d3 = ghostsDim3; d3 < dim3 + ghostsDim3; d3++)
                            for (int d4 = ghostsDim4; d4 < dim4 + ghostsDim4; d4++)
                                for (int d5 = ghostsDim5; d5 < dim5 + ghostsDim5; d5++)
                                    for (int d6 = ghostsDim6; d6 < dim6 + ghostsDim6; d6++)
                                    {
                                        myfile << varname << "(" << d1 + 1 << "," << d2 + 1 << "," << d3 + 1 << "," << d4 + 1 << "," << d5 + 1 << "," << d6 + 1 << ")="
                                               << std::scientific << std::setprecision(17) << field_6d[d1][d2][d3][d4][d5][d6] << ";" << std::endl;
                                    }
            }
            else
            {
                myfile << varname << " = zeros(" << dim1gh << ","
                       << dim2gh << ","
                       << dim3gh << ","
                       << dim4gh << ","
                       << dim5gh << ","
                       << dim6gh << ");" << std::endl;
                for (int d1 = 0; d1 < dim1gh; d1++)
                    for (int d2 = 0; d2 < dim2gh; d2++)
                        for (int d3 = 0; d3 < dim3gh; d3++)
                            for (int d4 = 0; d4 < dim4gh; d4++)
                                for (int d5 = 0; d5 < dim5gh; d5++)
                                    for (int d6 = 0; d6 < dim6gh; d6++)
                                    {
                                        myfile << varname << "(" << d1 + 1 << "," << d2 + 1 << "," << d3 + 1 << "," << d4 + 1 << "," << d5 + 1 << "," << d6 + 1 << ")="
                                               << std::scientific << std::setprecision(17) << field_6d[d1][d2][d3][d4][d5][d6] << ";" << std::endl;
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
     * @brief Copies contents of real cells of another Hypercube6d o to real cells of this one.
     *
     * @param o
     */
    void Hypercube6d::copyRealFrom(Hypercube6d& o)
    {
        // TODO check ranges
        if (dim1 != o.dim1 || dim2 != o.dim2 || dim3 != o.dim3 ||
            dim4 != o.dim4 || dim5 != o.dim5 || dim6 != o.dim6)
            error("Dimensions must match!");

        // clang-format off
        for (int i = ghostsDim1, id = o.ghostsDim1; i < dim1 + ghostsDim1; i++, id++)
        for (int j = ghostsDim2, jd = o.ghostsDim2; j < dim2 + ghostsDim2; j++, jd++)
        for (int k = ghostsDim3, kd = o.ghostsDim3; k < dim3 + ghostsDim3; k++, kd++)
            for (int ii = ghostsDim4, iid = o.ghostsDim4; ii < dim4 + ghostsDim4; ii++, iid++)
            for (int jj = ghostsDim5, jjd = o.ghostsDim5; jj < dim5 + ghostsDim5; jj++, jjd++)
            for (int kk = ghostsDim6, kkd = o.ghostsDim6; kk < dim6 + ghostsDim6; kk++, kkd++)
            {
                (*this)[i][j][k][ii][jj][kk] = o[id][jd][kd][iid][jjd][kkd];
            }
        // clang-format on
    }

    int Hypercube6d::getDim1() const
    {
        return dim1;
    }
    int Hypercube6d::getDim4() const
    {
        return dim4;
    }
    int Hypercube6d::getDim3gh() const
    {
        return dim3gh;
    }
    int Hypercube6d::getGhostsDim2() const
    {
        return ghostsDim2;
    }

    int Hypercube6d::getDim5() const
    {
        return dim5;
    }

    int Hypercube6d::getDim6gh() const
    {
        return dim6gh;
    }

    int Hypercube6d::getDim3() const
    {
        return dim3;
    }

    int Hypercube6d::getDim2gh() const
    {
        return dim2gh;
    }

    int Hypercube6d::getGhostsDim1() const
    {
        return ghostsDim1;
    }

    int Hypercube6d::getGhostsDim4() const
    {
        return ghostsDim4;
    }

    int Hypercube6d::getDim5gh() const
    {
        return dim5gh;
    }

    int Hypercube6d::getGhostsDim6() const
    {
        return ghostsDim6;
    }

    int Hypercube6d::getDim2() const
    {
        return dim2;
    }

    int Hypercube6d::getDim1gh() const
    {
        return dim1gh;
    }

    int Hypercube6d::getDim4gh() const
    {
        return dim4gh;
    }

    int Hypercube6d::getGhostsDim3() const
    {
        return ghostsDim3;
    }

    int Hypercube6d::getDim6() const
    {
        return dim6;
    }

    int Hypercube6d::getGhostsDim5() const
    {
        return ghostsDim5;
    }

    /****************************************************************
     * Hypercube8d
     * **************************************************************/

    /**
     * @brief Construct a new Hypercube8d object having ghosts = 0 and optionally a value which defaults to 0.
     *
     * @param dim1_ Dimension of real grid.
     * @param dim2_ Dimension of real grid.
     * @param dim3_ Dimension of real grid.
     * @param dim4_ Dimension of real grid.
     * @param dim5_ Dimension of real grid.
     * @param dim6_ Dimension of real grid.
     * @param dim7_ Dimension of real grid.
     * @param dim8_ Dimension of real grid.
     * @param value Initial value, defaults to 0.
     */
    Hypercube8d::Hypercube8d(int dim1_, int dim2_, int dim3_, int dim4_, int dim5_, int dim6_, int dim7_, int dim8_, double value)
        : Hypercube8d(dim1_, dim2_, dim3_, dim4_, dim5_, dim6_, dim7_, dim8_, 0, 0, 0, 0, 0, 0, 0, 0, value) {}

    /**
     * @brief Construct a new ghosted Hypercube8d object, optionally having a value which defaults to 0.
     *
     * @param dim1_ Dimension of real grid.
     * @param dim2_ Dimension of real grid.
     * @param dim3_ Dimension of real grid.
     * @param dim4_ Dimension of real grid.
     * @param dim5_ Dimension of real grid.
     * @param dim6_ Dimension of real grid.
     * @param dim7_ Dimension of real grid.
     * @param dim8_ Dimension of real grid.
     * @param ghostsDim1_ Amount of ghost cells in one direction.
     * @param ghostsDim2_ Amount of ghost cells in one direction.
     * @param ghostsDim3_ Amount of ghost cells in one direction.
     * @param ghostsDim4_ Amount of ghost cells in one direction.
     * @param ghostsDim5_ Amount of ghost cells in one direction.
     * @param ghostsDim6_ Amount of ghost cells in one direction.
     * @param ghostsDim7_ Amount of ghost cells in one direction.
     * @param ghostsDim8_ Amount of ghost cells in one direction.
     * @param value Initial value, defaults to 0.
     */
    Hypercube8d::Hypercube8d(int dim1_, int dim2_, int dim3_, int dim4_, int dim5_, int dim6_, int dim7_, int dim8_,
                             int ghostsDim1_, int ghostsDim2_, int ghostsDim3_, int ghostsDim4_, int ghostsDim5_, int ghostsDim6_, int ghostsDim7_, int ghostsDim8_, double value)
        : dim1(dim1_),
          dim2(dim2_),
          dim3(dim3_),
          dim4(dim4_),
          dim5(dim5_),
          dim6(dim6_),
          dim7(dim7_),
          dim8(dim8_),
          dim1gh(dim1_ + 2 * ghostsDim1_),
          dim2gh(dim2_ + 2 * ghostsDim2_),
          dim3gh(dim3_ + 2 * ghostsDim3_),
          dim4gh(dim4_ + 2 * ghostsDim4_),
          dim5gh(dim5_ + 2 * ghostsDim5_),
          dim6gh(dim6_ + 2 * ghostsDim6_),
          dim7gh(dim7_ + 2 * ghostsDim7_),
          dim8gh(dim8_ + 2 * ghostsDim8_),
          ghostsDim1(ghostsDim1_),
          ghostsDim2(ghostsDim2_),
          ghostsDim3(ghostsDim3_),
          ghostsDim4(ghostsDim4_),
          ghostsDim5(ghostsDim5_),
          ghostsDim6(ghostsDim6_),
          ghostsDim7(ghostsDim7_),
          ghostsDim8(ghostsDim8_),
          size(dim1gh * dim2gh * dim3gh * dim4gh * dim5gh * dim6gh * dim7gh * dim8gh)
    {
        field_1d.resize(size);
        for (int i = 0; i < field_1d.size(); i++)
            field_1d[i] = value;

        field_8d = new double*******[dim1gh];
        int size7gh = dim8gh;
        int size6gh = size7gh * dim7gh;
        int size5gh = size6gh * dim6gh;
        int size4gh = size5gh * dim5gh;
        int size3gh = size4gh * dim4gh;
        int size2gh = size3gh * dim3gh;
        int size1gh = size2gh * dim2gh;
        for (int d1 = 0; d1 < dim1gh; d1++)
        {
            field_8d[d1] = new double******[dim2gh];
            for (int d2 = 0; d2 < dim2gh; d2++)
            {
                field_8d[d1][d2] = new double*****[dim3gh];
                for (int d3 = 0; d3 < dim3gh; d3++)
                {
                    field_8d[d1][d2][d3] = new double****[dim4gh];
                    for (int d4 = 0; d4 < dim4gh; d4++)
                    {
                        field_8d[d1][d2][d3][d4] = new double***[dim5gh];
                        for (int d5 = 0; d5 < dim5gh; d5++)
                        {
                            field_8d[d1][d2][d3][d4][d5] = new double**[dim6gh];
                            for (int d6 = 0; d6 < dim6gh; d6++)
                            {
                                field_8d[d1][d2][d3][d4][d5][d6] = new double*[dim7gh];
                                for (int d7 = 0; d7 < dim7gh; d7++)
                                {
                                    field_8d[d1][d2][d3][d4][d5][d6][d7] = &field_1d[d1 * size1gh + d2 * size2gh + d3 * size3gh + d4 * size4gh + d5 * size5gh + d6 * size6gh + d7 * size7gh];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Hypercube8d::Hypercube8d(Hypercube8d&& o)
        : dim1(o.dim1),
          dim2(o.dim2),
          dim3(o.dim3),
          dim4(o.dim4),
          dim5(o.dim5),
          dim6(o.dim6),
          dim7(o.dim7),
          dim8(o.dim8),
          dim1gh(o.dim1gh),
          dim2gh(o.dim2gh),
          dim3gh(o.dim3gh),
          dim4gh(o.dim4gh),
          dim5gh(o.dim5gh),
          dim6gh(o.dim6gh),
          dim7gh(o.dim7gh),
          dim8gh(o.dim8gh),
          ghostsDim1(o.ghostsDim1),
          ghostsDim2(o.ghostsDim2),
          ghostsDim3(o.ghostsDim3),
          ghostsDim4(o.ghostsDim4),
          ghostsDim5(o.ghostsDim5),
          ghostsDim6(o.ghostsDim6),
          ghostsDim7(o.ghostsDim7),
          ghostsDim8(o.ghostsDim8),
          field_1d(std::move(o.field_1d)),
          field_8d(o.field_8d),
          size(o.size)
    {
        o.dim1 = 0;
        o.dim2 = 0;
        o.dim3 = 0;
        o.dim4 = 0;
        o.dim5 = 0;
        o.dim6 = 0;
        o.dim7 = 0;
        o.dim8 = 0;
        o.dim1gh = 0;
        o.dim2gh = 0;
        o.dim3gh = 0;
        o.dim4gh = 0;
        o.dim5gh = 0;
        o.dim6gh = 0;
        o.dim7gh = 0;
        o.dim8gh = 0;
        o.ghostsDim1 = 0;
        o.ghostsDim2 = 0;
        o.ghostsDim3 = 0;
        o.ghostsDim4 = 0;
        o.ghostsDim5 = 0;
        o.ghostsDim6 = 0;
        o.ghostsDim7 = 0;
        o.ghostsDim8 = 0;
        o.field_8d = nullptr;
        o.size = 0;
    }

    Hypercube8d& Hypercube8d::operator=(Hypercube8d&& o)
    {
        dim1 = o.dim1;
        dim2 = o.dim2;
        dim3 = o.dim3;
        dim4 = o.dim4;
        dim5 = o.dim5;
        dim6 = o.dim6;
        dim7 = o.dim7;
        dim8 = o.dim8;
        dim1gh = o.dim1gh;
        dim2gh = o.dim2gh;
        dim3gh = o.dim3gh;
        dim4gh = o.dim4gh;
        dim5gh = o.dim5gh;
        dim6gh = o.dim6gh;
        dim7gh = o.dim7gh;
        dim8gh = o.dim8gh;
        ghostsDim1 = o.ghostsDim1;
        ghostsDim2 = o.ghostsDim2;
        ghostsDim3 = o.ghostsDim3;
        ghostsDim4 = o.ghostsDim4;
        ghostsDim5 = o.ghostsDim5;
        ghostsDim6 = o.ghostsDim6;
        ghostsDim7 = o.ghostsDim7;
        ghostsDim8 = o.ghostsDim8;
        field_1d = std::move(o.field_1d);
        field_8d = o.field_8d;
        size = o.size;

        o.dim1 = 0;
        o.dim2 = 0;
        o.dim3 = 0;
        o.dim4 = 0;
        o.dim5 = 0;
        o.dim6 = 0;
        o.dim7 = 0;
        o.dim8 = 0;
        o.dim1gh = 0;
        o.dim2gh = 0;
        o.dim3gh = 0;
        o.dim4gh = 0;
        o.dim5gh = 0;
        o.dim6gh = 0;
        o.dim7gh = 0;
        o.dim8gh = 0;
        o.ghostsDim1 = 0;
        o.ghostsDim2 = 0;
        o.ghostsDim3 = 0;
        o.ghostsDim4 = 0;
        o.ghostsDim5 = 0;
        o.ghostsDim6 = 0;
        o.ghostsDim7 = 0;
        o.ghostsDim8 = 0;
        o.field_8d = nullptr;
        o.size = 0;
        return *this;
    }

    /**
     * @brief Destroy the Hypercube8d object, freeing memory.
     */
    Hypercube8d::~Hypercube8d()
    {
        if (field_8d)
        {
            for (int d1 = 0; d1 < dim1gh; d1++)
            {
                for (int d2 = 0; d2 < dim2gh; d2++)
                {
                    for (int d3 = 0; d3 < dim3gh; d3++)
                    {
                        for (int d4 = 0; d4 < dim4gh; d4++)
                        {
                            for (int d5 = 0; d5 < dim5gh; d5++)
                            {
                                for (int d6 = 0; d6 < dim6gh; d6++)
                                {
                                    delete[] field_8d[d1][d2][d3][d4][d5][d6];
                                }
                                delete[] field_8d[d1][d2][d3][d4][d5];
                            }
                            delete[] field_8d[d1][d2][d3][d4];
                        }
                        delete[] field_8d[d1][d2][d3];
                    }
                    delete[] field_8d[d1][d2];
                }
                delete[] field_8d[d1];
            }
            delete[] field_8d;
            field_8d = nullptr;
        }
    }

    double******** Hypercube8d::getData() const
    {
        return field_8d;
    }

    double******* Hypercube8d::operator[](int index) const
    {
        return field_8d[index];
    }

    /**
     * @brief Fills Hypercube8d with random values between low and high, which default to 0 and 1.
     *
     */
    void Hypercube8d::fillRandom(double low, double high)
    {
        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_real_distribution<double> dist(low, high);

        for (int i = 0; i < field_1d.size(); i++)
            field_1d[i] = dist(rng);
    }

    void Hypercube8d::fillRandomInt(int low, int high, bool realCellsOnly)
    {
        // use fixed seed to get same results every run
        std::default_random_engine rng(123);
        std::uniform_int_distribution<int> dist(low, high);

        if (realCellsOnly)
        {
            for (int d1 = ghostsDim1; d1 < dim1 + ghostsDim1; d1++)
                for (int d2 = ghostsDim2; d2 < dim2 + ghostsDim2; d2++)
                    for (int d3 = ghostsDim3; d3 < dim3 + ghostsDim3; d3++)
                        for (int d4 = ghostsDim4; d4 < dim4 + ghostsDim4; d4++)
                            for (int d5 = ghostsDim5; d5 < dim5 + ghostsDim5; d5++)
                                for (int d6 = ghostsDim6; d6 < dim6 + ghostsDim6; d6++)
                                    for (int d7 = ghostsDim7; d7 < dim7 + ghostsDim7; d7++)
                                        for (int d8 = ghostsDim8; d8 < dim8 + ghostsDim8; d8++)
                                        {
                                            field_8d[d1][d2][d3][d4][d5][d6][d7][d8] = dist(rng);
                                        }
        }
        else
        {
            for (int i = 0; i < field_1d.size(); i++)
                field_1d[i] = dist(rng);
        }
    }

    /**
     * @brief Fills Hypercube8d with given value
     *
     * @param value Value to fill
     * @param realCellsOnly If true, only real cells will be set to value. Defaults to false.
     */
    void Hypercube8d::fill(double value, bool realCellsOnly)
    {
        if (realCellsOnly)
        {
            for (int d1 = ghostsDim1; d1 < dim1 + ghostsDim1; d1++)
                for (int d2 = ghostsDim2; d2 < dim2 + ghostsDim2; d2++)
                    for (int d3 = ghostsDim3; d3 < dim3 + ghostsDim3; d3++)
                        for (int d4 = ghostsDim4; d4 < dim4 + ghostsDim4; d4++)
                            for (int d5 = ghostsDim5; d5 < dim5 + ghostsDim5; d5++)
                                for (int d6 = ghostsDim6; d6 < dim6 + ghostsDim6; d6++)
                                    for (int d7 = ghostsDim7; d7 < dim7 + ghostsDim7; d7++)
                                        for (int d8 = ghostsDim8; d8 < dim8 + ghostsDim8; d8++)
                                        {
                                            field_8d[d1][d2][d3][d4][d5][d6][d7][d8] = value;
                                        }
        }
        else
        {
            std::fill(field_1d.begin(), field_1d.end(), value);
        }
    }

    /**
     * @brief Fills this Hypercube8d with the 1d index of the corresponding cell, including ghost cells.
     *
     * @param realCellsOnly If true, only real cells get filled.
     */
    void Hypercube8d::fill1dIndex(bool realCellsOnly)
    {
        if (realCellsOnly)
            for (int d1 = ghostsDim1; d1 < dim1 + ghostsDim1; d1++)
                for (int d2 = ghostsDim2; d2 < dim2 + ghostsDim2; d2++)
                    for (int d3 = ghostsDim3; d3 < dim3 + ghostsDim3; d3++)
                        for (int d4 = ghostsDim4; d4 < dim4 + ghostsDim4; d4++)
                            for (int d5 = ghostsDim5; d5 < dim5 + ghostsDim5; d5++)
                                for (int d6 = ghostsDim6; d6 < dim6 + ghostsDim6; d6++)
                                    for (int d7 = ghostsDim7; d7 < dim7 + ghostsDim7; d7++)
                                        for (int d8 = ghostsDim8; d8 < dim8 + ghostsDim8; d8++)
                                        {
                                            (*this)[d1][d2][d3][d4][d5][d6][d7][d8] =
                                                d1 * dim2gh * dim3gh * dim4gh * dim5gh * dim6gh * dim7gh * dim8gh +
                                                d2 * dim3gh * dim4gh * dim5gh * dim6gh * dim7gh * dim8gh +
                                                d3 * dim4gh * dim5gh * dim6gh * dim7gh * dim8gh +
                                                d4 * dim5gh * dim6gh * dim7gh * dim8gh +
                                                d5 * dim6gh * dim7gh * dim8gh +
                                                d6 * dim7gh * dim8gh +
                                                d7 * dim8gh +
                                                d8;
                                        }
        else
            for (int d1 = 0; d1 < dim1gh; d1++)
                for (int d2 = 0; d2 < dim2gh; d2++)
                    for (int d3 = 0; d3 < dim3gh; d3++)
                        for (int d4 = 0; d4 < dim4gh; d4++)
                            for (int d5 = 0; d5 < dim5gh; d5++)
                                for (int d6 = 0; d6 < dim6gh; d6++)
                                    for (int d7 = 0; d7 < dim7gh; d7++)
                                        for (int d8 = 0; d8 < dim8gh; d8++)
                                        {
                                            (*this)[d1][d2][d3][d4][d5][d6][d7][d8] =
                                                d1 * dim2gh * dim3gh * dim4gh * dim5gh * dim6gh * dim7gh * dim8gh +
                                                d2 * dim3gh * dim4gh * dim5gh * dim6gh * dim7gh * dim8gh +
                                                d3 * dim4gh * dim5gh * dim6gh * dim7gh * dim8gh +
                                                d4 * dim5gh * dim6gh * dim7gh * dim8gh +
                                                d5 * dim6gh * dim7gh * dim8gh +
                                                d6 * dim7gh * dim8gh +
                                                d7 * dim8gh +
                                                d8;
                                        }
    }

    std::vector<double>& Hypercube8d::field1d()
    {
        return field_1d;
    }

    /**
     * @brief Returns true if real cells contents of this Hypercube8d is equal to the one of another Hypercube8d c
     * within a given tolerance tol, respecting ghost cell amount. Dimensions of real cell amount of this Hypercube8d
     *  and c must be equal (without ghost cells).
     *
     * @param c Other Hypercube8d
     * @param tol tolerance that is used for checking equality. Defaults to 1e-7.
     * @param printDiffs If true, differences will be printed to standard output. Defaults to true.
     * @return true Cuboids equal.
     * @return false Cuboids not equal.
     * @throws invalid_argument When dimensions of Cuboids don't match.
     */
    bool Hypercube8d::isEqual(Hypercube8d& c, double tol, bool printDiffs)
    {
        if (dim1 != c.getDim1() ||
            dim2 != c.getDim2() ||
            dim3 != c.getDim3() ||
            dim4 != c.getDim4() ||
            dim5 != c.getDim5() ||
            dim6 != c.getDim6() ||
            dim7 != c.getDim7() ||
            dim8 != c.getDim8())
        {
            error(std::invalid_argument("Cannot check equality for Hypercube8d. Dimensions differ."));
        }

        double diff;
        for (int d1 = 0; d1 < dim1; d1++)
            for (int d2 = 0; d2 < dim2; d2++)
                for (int d3 = 0; d3 < dim3; d3++)
                    for (int d4 = 0; d4 < dim4; d4++)
                        for (int d5 = 0; d5 < dim5; d5++)
                            for (int d6 = 0; d6 < dim6; d6++)
                                for (int d7 = 0; d7 < dim7; d7++)
                                    for (int d8 = 0; d8 < dim8; d8++)
                                    {
                                        diff = fabs(field_8d[d1 + ghostsDim1][d2 + ghostsDim2][d3 + ghostsDim3][d4 + ghostsDim4][d5 + ghostsDim5][d6 + ghostsDim6][d7 + ghostsDim7][d8 + ghostsDim8] -
                                                    c[d1 + c.getGhostsDim1()][d2 + c.getGhostsDim2()][d3 + c.getGhostsDim3()][d4 + c.getGhostsDim4()][d5 + c.getGhostsDim5()][d6 + c.getGhostsDim6()][d7 + c.getGhostsDim7()][d8 + c.getGhostsDim8()]);
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
    void Hypercube8d::dumpToFile(std::string path, bool realCellsOnly)
    {
        std::ofstream myfile;
        myfile.open(path, std::ios::out | std::ios::trunc);

        if (myfile.is_open())
        {
            if (realCellsOnly)
            {
                for (int d1 = ghostsDim1; d1 < dim1 + ghostsDim1; d1++)
                    for (int d2 = ghostsDim2; d2 < dim2 + ghostsDim2; d2++)
                        for (int d3 = ghostsDim3; d3 < dim3 + ghostsDim3; d3++)
                            for (int d4 = ghostsDim4; d4 < dim4 + ghostsDim4; d4++)
                                for (int d5 = ghostsDim5; d5 < dim5 + ghostsDim5; d5++)
                                    for (int d6 = ghostsDim6; d6 < dim6 + ghostsDim6; d6++)
                                        for (int d7 = ghostsDim7; d7 < dim7 + ghostsDim7; d7++)
                                            for (int d8 = ghostsDim8; d8 < dim8 + ghostsDim8; d8++)
                                            {
                                                myfile << d1 << "\t" << d2 << "\t" << d3 << "\t" << d4 << "\t" << d5 << "\t" << d6 << "\t" << d7 << "\t" << d8 << "\t"
                                                       << std::scientific << std::setprecision(17) << field_8d[d1][d2][d3][d4][d5][d6][d7][d8] << std::endl;
                                            }
            }
            else
            {
                for (int d1 = 0; d1 < dim1gh; d1++)
                    for (int d2 = 0; d2 < dim2gh; d2++)
                        for (int d3 = 0; d3 < dim3gh; d3++)
                            for (int d4 = 0; d4 < dim4gh; d4++)
                                for (int d5 = 0; d5 < dim5gh; d5++)
                                    for (int d6 = 0; d6 < dim6gh; d6++)
                                        for (int d7 = 0; d7 < dim7gh; d7++)
                                            for (int d8 = 0; d8 < dim8gh; d8++)
                                            {
                                                myfile << d1 << "\t" << d2 << "\t" << d3 << "\t" << d4 << "\t" << d5 << "\t" << d6 << "\t" << d7 << "\t" << d8 << "\t"
                                                       << std::scientific << std::setprecision(17) << field_8d[d1][d2][d3][d4][d5][d6][d7][d8] << std::endl;
                                            }
            }
            myfile.close();
        }
        else
        {
            error(std::runtime_error("Couldn't open file for writing given by: " + path));
        }
    }

    void Hypercube8d::dumpToFileMatlab(std::string path, std::string varname, bool realCellsOnly)
    {
        std::ofstream myfile;
        myfile.open(path, std::ios::out | std::ios::trunc);

        if (myfile.is_open())
        {
            if (realCellsOnly)
            {
                myfile << varname << " = zeros(" << dim1 << ","
                       << dim2 << ","
                       << dim3 << ","
                       << dim4 << ","
                       << dim5 << ","
                       << dim6 << ","
                       << dim7 << ","
                       << dim8 << ");" << std::endl;
                for (int d1 = ghostsDim1; d1 < dim1 + ghostsDim1; d1++)
                    for (int d2 = ghostsDim2; d2 < dim2 + ghostsDim2; d2++)
                        for (int d3 = ghostsDim3; d3 < dim3 + ghostsDim3; d3++)
                            for (int d4 = ghostsDim4; d4 < dim4 + ghostsDim4; d4++)
                                for (int d5 = ghostsDim5; d5 < dim5 + ghostsDim5; d5++)
                                    for (int d6 = ghostsDim6; d6 < dim6 + ghostsDim6; d6++)
                                        for (int d7 = ghostsDim7; d7 < dim7 + ghostsDim7; d7++)
                                            for (int d8 = ghostsDim8; d8 < dim8 + ghostsDim8; d8++)
                                            {
                                                myfile << varname << "(" << d1 + 1 << "," << d2 + 1 << "," << d3 + 1 << "," << d4 + 1 << "," << d5 + 1 << "," << d6 + 1 << "," << d7 + 1 << "," << d8 + 1 << ")="
                                                       << std::scientific << std::setprecision(17) << field_8d[d1][d2][d3][d4][d5][d6][d7][d8] << ";" << std::endl;
                                            }
            }
            else
            {
                myfile << varname << " = zeros(" << dim1gh << ","
                       << dim2gh << ","
                       << dim3gh << ","
                       << dim4gh << ","
                       << dim5gh << ","
                       << dim6gh << ","
                       << dim7gh << ","
                       << dim8gh << ");" << std::endl;
                for (int d1 = 0; d1 < dim1gh; d1++)
                    for (int d2 = 0; d2 < dim2gh; d2++)
                        for (int d3 = 0; d3 < dim3gh; d3++)
                            for (int d4 = 0; d4 < dim4gh; d4++)
                                for (int d5 = 0; d5 < dim5gh; d5++)
                                    for (int d6 = 0; d6 < dim6gh; d6++)
                                        for (int d7 = 0; d7 < dim7gh; d7++)
                                            for (int d8 = 0; d8 < dim8gh; d8++)
                                            {
                                                myfile << varname << "(" << d1 + 1 << "," << d2 + 1 << "," << d3 + 1 << "," << d4 + 1 << "," << d5 + 1 << "," << d6 + 1 << "," << d7 + 1 << "," << d8 + 1 << ")="
                                                       << std::scientific << std::setprecision(17) << field_8d[d1][d2][d3][d4][d5][d6][d7][d8] << ";" << std::endl;
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
     * @brief Copies contents of real cells of another Hypercube8d o to real cells of this one.
     *
     * @param o
     */
    void Hypercube8d::copyRealFrom(Hypercube8d& o)
    {
        // TODO check ranges
        if (dim1 != o.dim1 || dim2 != o.dim2 || dim3 != o.dim3 ||
            dim4 != o.dim4 || dim5 != o.dim5 || dim6 != o.dim6 ||
            dim7 != o.dim7 || dim8 != o.dim8)
            error("Dimensions must match!");

        // clang-format off
        for (int i = ghostsDim1, id = o.ghostsDim1; i < dim1 + ghostsDim1; i++, id++)
        for (int j = ghostsDim2, jd = o.ghostsDim2; j < dim2 + ghostsDim2; j++, jd++)
        for (int k = ghostsDim3, kd = o.ghostsDim3; k < dim3 + ghostsDim3; k++, kd++)
            for (int ii = ghostsDim4, iid = o.ghostsDim4; ii < dim4 + ghostsDim4; ii++, iid++)
            for (int jj = ghostsDim5, jjd = o.ghostsDim5; jj < dim5 + ghostsDim5; jj++, jjd++)
            for (int kk = ghostsDim6, kkd = o.ghostsDim6; kk < dim6 + ghostsDim6; kk++, kkd++)
            for (int l = ghostsDim7, lld = o.ghostsDim7; l < dim7 + ghostsDim7; l++, lld++)
            for (int m = ghostsDim8, mmd = o.ghostsDim8; m < dim8 + ghostsDim8; m++, mmd++)
            {
                (*this)[i][j][k][ii][jj][kk][l][m] = o[id][jd][kd][iid][jjd][kkd][lld][mmd];
            }
        // clang-format on
    }
}
