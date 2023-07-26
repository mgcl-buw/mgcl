#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <iostream>
#include <memory>

#include "../src/cuboid.hpp"
#include "../src/problem.hpp"
// #include "test_utility.hpp"

#include "mpi.h"

// checks if neighbours are initialized correctly for each level for 1 process
TEST_CASE("Level::initMpiData (1 process)", "[mpi1]")
{
    int N = 8;
    int periodic = 1;

    // check if mpi is initialized
    int isInitialized = 0;
    MPI_Initialized(&isInitialized);
    REQUIRE(isInitialized);

    MPI_Comm mpi_comm = MPI_COMM_WORLD;

    // check number of processes
    int mpi_size = -1;
    MPI_Comm_size(mpi_comm, &mpi_size);
    REQUIRE(mpi_size == 1);

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

    // Init some random data
    auto v = std::make_shared<mgcl::Cuboid>(N, N, N);
    auto f = std::make_shared<mgcl::Cuboid>(N, N, N);
    v->fillRandom();
    f->fillRandom();

    mgcl::Problem p(N, N, N, v, f);
    p.setMpiComm(mpi_comm);
    p.init();

    for (int i = 0; i < p.getMaxlevel() - 1; i++)
    {
        auto &lv = p.getLevelAt(i);
        auto &mpiData = lv.getMpiData();

        REQUIRE(mpiData.rank == 0);
        REQUIRE(mpiData.left == 0);
        REQUIRE(mpiData.right == 0);
        REQUIRE(mpiData.up == 0);
        REQUIRE(mpiData.down == 0);
        REQUIRE(mpiData.front == 0);
        REQUIRE(mpiData.back == 0);
    }
}