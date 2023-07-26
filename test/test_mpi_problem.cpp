#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <iostream>
#include <memory>

#include "../src/cuboid.hpp"
#include "../src/problem.hpp"
// #include "test_utility.hpp"

#include "mpi.h"

// Checks that an exception is thrown if the communicator has no cartesian topology attached.
TEST_CASE("Problem::setMpiComm")
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
