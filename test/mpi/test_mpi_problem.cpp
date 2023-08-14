#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <iostream>
#include <memory>

#include "../src/cuboid.hpp"
#include "../src/problem.hpp"
// #include "test_utility.hpp"

#include "mpi.h"

// Checks that an exception is thrown if the communicator has no cartesian topology attached.
TEST_CASE("MPI Problem::setMpiComm")
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

    // Check with valid topology first
    mgcl::Problem p(N, N, N);
    REQUIRE_NOTHROW(p.setMpiComm(mpi_comm));

    // Now check with no topology
    MPI_Group group;
    MPI_Comm_group(mpi_comm, &group);
    MPI_Comm_create(mpi_comm, group, &mpi_comm);

    mgcl::Problem p2(N, N, N);
    REQUIRE_THROWS(p.setMpiComm(mpi_comm));
}

// Max level should be calculated using global size, not local size of grid.
// For one process, m local equals m global.
// Run with: mpiexec -n 1 tests_mpi [mpi1]
TEST_CASE("MPI Problem::calculateAndSetMaxLevel (1 process)", "[mpi1]")
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

    mgcl::Problem p(N, N, N);
    REQUIRE(p.getMaxlevel() == 3);

    mgcl::Problem p2(N, N, N, N, N, N);
    REQUIRE(p2.getMaxlevel() == 3);
}

// Max level should be calculated using global size, not local size of grid.
// For 4 processes, m local = 1/4 * m global.
// Run with: mpiexec -n 4 tests_mpi [mpi4]
TEST_CASE("MPI Problem::calculateAndSetMaxLevel (4 processes)", "[mpi4]")
{
    int N = 4;
    int Ng = N * 4;
    int periodic = 1;

    // check if mpi is initialized
    int isInitialized = 0;
    MPI_Initialized(&isInitialized);
    REQUIRE(isInitialized);

    MPI_Comm mpi_comm = MPI_COMM_WORLD;

    // check number of processes
    int mpi_size = -1;
    MPI_Comm_size(mpi_comm, &mpi_size);
    REQUIRE(mpi_size == 4);

    mgcl::Problem p(N, N, N, Ng, Ng, Ng);
    REQUIRE(p.getMaxlevel() == 4);
}

// mpiLevelThreshold should be calculated s.t. there are at least 8 grid points per process per direction.
// This test is done on one process only.
TEST_CASE("MPI Problem::calculateAndSetMpiLevelThreshold valid (1 process)", "[mpi1]")
{
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

    // Set MPI cartesion topology dimenions and appropriate local sizes.
    // clang-format off
    std::vector<std::vector<int>> dims = {
        {4,4,4}, {2,2,2}, {4,4,4}, {1,2,4},
        {1,2,4}, {1,2,4}, {1,2,2}, {2,2,1}
    };

    std::vector<std::vector<int>> local = {
        {4,4,4}, {8,8,8}, {8,8,8}, {4,4,2},
        {4,4,4}, {16,4,2}, {16,16,16}, {64,64,128}
    };

    std::vector<int> expected = {0,1,1,0,0,0,2,4};

    REQUIRE(dims.size() == local.size());
    REQUIRE(dims.size() == expected.size());
    // clang-format on

    // Check for valid
    for (int i = 0; i < dims.size(); i++)
    {

        // Check with valid topology first
        mgcl::Problem p(local[i][0], local[i][1], local[i][2],
                        local[i][0] * dims[i][0], local[i][1] * dims[i][1], local[i][2] * dims[i][2]);
        // p.setMpiComm(mpi_comm);
        p.calculateAndSetMpiLevelThreshold();

        REQUIRE(p.getMpiLevelThreshold() == expected[i]);
    }
}

// calculateAndSetMpiLevelThreshold should be throw if mpiLevelThreshold is set too high by user.
// This test is done on one process only.
TEST_CASE("MPI Problem::calculateAndSetMpiLevelThreshold throwing (1 process)", "[mpi1]")
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

    // Check with valid topology first
    mgcl::Problem p(N, N, N);
    REQUIRE_NOTHROW(p.setMpiComm(mpi_comm));
    p.setMpiLevelThreshold(1000);

    REQUIRE_THROWS(p.calculateAndSetMpiLevelThreshold());
}
