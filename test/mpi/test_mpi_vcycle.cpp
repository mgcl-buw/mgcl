#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>

#include "../../src/cuboid.hpp"
#include "../../src/mpi_data.hpp"
#include "../../src/mpi_util.hpp"
#include "../../src/problem.hpp"
#include "../test_utility.hpp"

#include "mpi.h"

std::shared_ptr<mgcl::Cuboid> calculateError(mgcl::Cuboid& solution, mgcl::Cuboid& approximation);
double calculateMaxError(mgcl::Cuboid& error);
double calculateErrorNorm(double h, mgcl::Cuboid& error);

// Tests if vcycle is correct for multiple processes but everything is actually done on one process, i.e.
// gathering and scattering happens on level 0.
// With 1 process mgcl should detect that MPI is actually not used, thus this test should behave like the solve tests.
TEST_CASE("MPI_vcycle_immediate_gather_scatter")
{
    using std::min;

    // global grid sizes
    int m = 8;
    int n = 8;
    int o = 8;
    int periodic = 1;
    int gh = 1;

    // check if mpi is initialized
    int isInitialized = 0;
    MPI_Initialized(&isInitialized);
    REQUIRE(isInitialized);

    MPI_Comm mpi_comm = MPI_COMM_WORLD;

    // check number of processes
    int mpi_size = -1;
    MPI_Comm_size(mpi_comm, &mpi_size);
    // REQUIRE(mpi_size == 8);

    /* MPI variables */
    int mpi_rank;
    int mpi_dims[3] = {0, 0, 0};
    int mpi_periods[3] = {periodic, periodic, periodic};
    int mpi_coords[3];

    /* Initialize cartesian process grid */
    MPI_Comm_size(mpi_comm, &mpi_size);
    MPI_Dims_create(mpi_size, 3, mpi_dims);
    MPI_Cart_create(mpi_comm, 3, mpi_dims, mpi_periods, 1, &mpi_comm);
    MPI_Comm_rank(mpi_comm, &mpi_rank);
    MPI_Cart_coords(mpi_comm, mpi_rank, 3, mpi_coords);

    /* Initialize start and end for local grid */
    int m_start = (m / mpi_dims[0]) * mpi_coords[0] + min(mpi_coords[0], (m % mpi_dims[0]));
    int m_end = (m / mpi_dims[0]) * (mpi_coords[0] + 1) + min(mpi_coords[0] + 1, (m % mpi_dims[0])) - 1;
    int n_start = (n / mpi_dims[1]) * mpi_coords[1] + min(mpi_coords[1], (n % mpi_dims[1]));
    int n_end = (n / mpi_dims[1]) * (mpi_coords[1] + 1) + min(mpi_coords[1] + 1, (n % mpi_dims[1])) - 1;
    int o_start = (o / mpi_dims[2]) * mpi_coords[2] + min(mpi_coords[2], (o % mpi_dims[2]));
    int o_end = (o / mpi_dims[2]) * (mpi_coords[2] + 1) + min(mpi_coords[2] + 1, (o % mpi_dims[2])) - 1;

    int ml = (m_end - m_start) + 1;
    int nl = (n_end - n_start) + 1;
    int ol = (o_end - o_start) + 1;

    CAPTURE(ml, nl, ol);

    // print coords and boundaries per rank
    // if (mpi_rank == 0)
    //     std::cout << "rank;coords[0];coords[1];coords[2];ms;me;ns;ne;os;oe" << std::endl;

    // for (int i = 0; i < mpi_size; i++)
    // {
    //     MPI_Barrier(mpi_comm);
    //     if (mpi_rank == i)
    //     {
    //         std::cout << mpi_rank << ";" << mpi_coords[0] << ";" << mpi_coords[1] << ";" << mpi_coords[2] << ";"
    //                   << m_start << ";" << m_end << ";"
    //                   << n_start << ";" << n_end << ";"
    //                   << o_start << ";" << o_end << std::endl;
    //     }
    // }

    REQUIRE(ml > 0);
    REQUIRE(ml <= m);
    REQUIRE(nl > 0);
    REQUIRE(nl <= n);
    REQUIRE(ol > 0);
    REQUIRE(ol <= o);

    // Set up 4th order periodic problem
    auto v = std::make_shared<mgcl::Cuboid>(m, n, o, 0, 0, 0);
    auto f = std::make_shared<mgcl::Cuboid>(m, n, o, 0, 0, 0);
    auto solution = std::make_shared<mgcl::Cuboid>(m, n, o, 0, 0, 0);
    mgcl_test::create4hOrderPeriodicProblem(*v, *f, *solution);

    // Create local slices
    std::shared_ptr<mgcl::Cuboid> vloc(v->slice(m_start, m_end, n_start, n_end, o_start, o_end));
    std::shared_ptr<mgcl::Cuboid> floc(f->slice(m_start, m_end, n_start, n_end, o_start, o_end));
    std::shared_ptr<mgcl::Cuboid> solutionloc(solution->slice(m_start, m_end, n_start, n_end, o_start, o_end));

    // Create local problem
    mgcl::Problem p(ml, nl, ol, floc, vloc, m, n, o);
    p.setGhosts(gh);
    p.setMpiMinGridPoints(m);

    p.solveSeq();

    // vloc->dumpToFile("vloc.txt");
    // floc->dumpToFile("floc.txt");
    // solutionloc->dumpToFile("solloc.txt");
    // v->dumpToFile("v.txt");
    // f->dumpToFile("f.txt");
    // solution->dumpToFile("sol.txt");

    // check if solution is good
    auto err = calculateError(*solutionloc, *vloc);
    auto errNorm = calculateErrorNorm(1.0 / (double)m, *err);
    auto errMax = calculateMaxError(*err);

    for (int i = 0; i < mpi_size; i++)
    {
        MPI_Barrier(mpi_comm);
        if (i == mpi_rank)
        {
            if (i == 0)
                std::cout << "seq MPI Laplace" << std::endl;
            std::cout << "rank " << i << ": " << std::endl;
            std::cout
                << std::scientific << std::setprecision(17) << "  ||e||_2 = " << errNorm << std::endl
                << std::scientific << std::setprecision(17) << "  e_max = " << errMax << std::endl;
        }
    }

    REQUIRE(errNorm < 1e-2);
    REQUIRE(errMax < 1e-2);

    // REQUIRE(vloc->isEqual(*solutionloc, 1e-7, true));
}

/**
 * @brief Calculates error for each cell, e.g. difference between solution and approximation. Dimensions must match.
 *
 * @param solution
 * @param approximation
 * @return std::shared_ptr<mgcl::Cuboid>
 */
std::shared_ptr<mgcl::Cuboid> calculateError(mgcl::Cuboid& solution, mgcl::Cuboid& approximation)
{
    if (solution.getM() != approximation.getM() ||
        solution.getN() != approximation.getN() ||
        solution.getO() != approximation.getO())
        throw std::invalid_argument("Dimensions do not match.");

    auto ret = std::make_shared<mgcl::Cuboid>(solution.getM(), solution.getN(), solution.getO());
    for (int i = 0, is = solution.getGhostsM(), ia = approximation.getGhostsM(); is < solution.getMgh(); i++, is++, ia++)
        for (int j = 0, js = solution.getGhostsN(), ja = approximation.getGhostsN(); js < solution.getNgh(); j++, js++, ja++)
            for (int k = 0, ks = solution.getGhostsO(), ka = approximation.getGhostsO(); ks < solution.getOgh(); k++, ks++, ka++)
            {
                (*ret)[i][j][k] = fabs(solution[is][js][ks] - approximation[ia][ja][ka]);
            }

    return ret;
}

/**
 * @brief Returns the maximum absolute error. calculateError should have been called first.
 *
 * @param error
 * @return double
 */
double calculateMaxError(mgcl::Cuboid& error)
{
    double max = 0;
    for (int i = 0; i < error.getM(); i++)
        for (int j = 0; j < error.getN(); j++)
            for (int k = 0; k < error.getO(); k++)
            {
                if (max < error[i][j][k])
                    max = error[i][j][k];
            }
    return max;
}

/**
 * @brief Returns the 2-norm of the given error. calculateError should have been called first.
 *
 * @param h width of one cell
 * @param error precalculated error per cell
 * @return double Error norm of form ||e||_2 * h^3
 */
double calculateErrorNorm(double h, mgcl::Cuboid& error)
{
    double sum = 0;

    for (int i = 0; i < error.getM(); i++)
        for (int j = 0; j < error.getN(); j++)
            for (int k = 0; k < error.getO(); k++)
            {
                sum += error[i][j][k] * error[i][j][k];
            }

    return sqrt(sum * h * h * h);
}