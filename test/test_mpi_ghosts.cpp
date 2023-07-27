#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <iostream>
#include <memory>

#include "../src/cuboid.hpp"
#include "../src/multigrid_engine.hpp"
#include "../src/problem.hpp"
// #include "test_utility.hpp"

#include "mpi.h"

// Checks ghost update for 1 process.
// Run with: mpiexec -n 1 tests_mpi [mpi1]
// TODO differentiate for gh>m and non-periodic
TEST_CASE("MPI updateGhostsSeq (1 process)", "[mpi1]")
{
    int N = 4;
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

    SECTION("gh = 1 < m,n,o")
    {
        int gh = 1;
        int m = N;
        int n = N;
        int o = N;

        // Init Problem to create all needed structures
        mgcl::Problem p(N, N, N, v, f);
        p.setGhosts(1);
        p.setMpiComm(mpi_comm);
        p.init();

        // Check on level 0
        auto &lv = p.getLevelAt(0);
        auto &mpiData = lv.getMpiData();

        // Create test data
        mgcl::Cuboid c(N, N, N);
        c.fillRandom(-10, 10, true);

        // Update ghosts of test data
        mgcl::MultigridEngine::updateGhostsSeq(c, &mpiData, true);

        // Check result
        // check in z-direction
        for (int i = 0; i < gh; i++)
            for (int j = 0; j < n + 2 * gh; j++)
                for (int k = 0; k < o + 2 * gh; k++)
                {
                    REQUIRE(c[i][j][k] == c[i + m][j][k]);
                    REQUIRE(c[i + gh][j][k] == c[i + gh + m][j][k]);
                }

        // check in y-direction
        for (int i = 0; i < m + 2 * gh; i++)
            for (int j = 0; j < gh; j++)
                for (int k = 0; k < o + 2 * gh; k++)
                {
                    REQUIRE(c[i][j][k] == c[i][j + n][k]);
                    REQUIRE(c[i][j + gh][k] == c[i][j + gh + n][k]);
                }

        // check in x-direction
        for (int i = 0; i < m + 2 * gh; i++)
            for (int j = 0; j < n + 2 * gh; j++)
                for (int k = 0; k < gh; k++)
                {
                    REQUIRE(c[i][j][k] == c[i][j][k + o]);
                    REQUIRE(c[i][j][k + gh] == c[i][j][k + gh + o]);
                }
    }
}

// Checks if neighbours are initialized correctly for each level for 2 processes.
// Run with: mpiexec -n 2 tests_mpi [mpi2]
// TEST_CASE("Level::initMpiData (2 processes)", "[mpi2]")
// {
//     int N = 8;
//     int periodic = 1;

//     // check if mpi is initialized
//     int isInitialized = 0;
//     MPI_Initialized(&isInitialized);
//     REQUIRE(isInitialized);

//     MPI_Comm mpi_comm = MPI_COMM_WORLD;

//     // check number of processes
//     int mpi_size = -1;
//     MPI_Comm_size(mpi_comm, &mpi_size);
//     REQUIRE(mpi_size == 2);

//     /* MPI variables */
//     int mpi_rank;
//     int mpi_dims[3] = {0, 0, 0};
//     int mpi_periods[3] = {periodic, periodic, periodic};
//     int mpi_coords[3];

//     /* Initialize cartesian process grid */
//     MPI_Comm_size(mpi_comm, &mpi_size);
//     MPI_Dims_create(mpi_size, 3, mpi_dims);
//     MPI_Cart_create(mpi_comm, 3, mpi_dims, mpi_periods, 1, &mpi_comm);
//     MPI_Comm_rank(mpi_comm, &mpi_rank);
//     MPI_Cart_coords(mpi_comm, mpi_rank, 3, mpi_coords);

//     // Init some random data
//     auto v = std::make_shared<mgcl::Cuboid>(N, N, N);
//     auto f = std::make_shared<mgcl::Cuboid>(N, N, N);
//     v->fillRandom();
//     f->fillRandom();

//     mgcl::Problem p(N, N, N, v, f);
//     p.setMpiComm(mpi_comm);
//     p.init();

//     int other_rank = mpi_rank == 0 ? 1 : 0;
//     int maxlv = p.getMaxlevel() - 1;

//     for (int i = 0; i < maxlv; i++)
//     {
//         auto &lv = p.getLevelAt(i);
//         auto &mpiData = lv.getMpiData();

//         REQUIRE(mpiData.rank == mpi_rank);

//         // If there are two processes in a direction, the neighbour must be the other process. Else or if no
//         // grid points are left on this level, the neighbour is set to self.
//         if (lv.getM() > 0 && mpi_dims[0] > 1)
//         {
//             REQUIRE(mpiData.left == other_rank);
//             REQUIRE(mpiData.right == other_rank);
//         }
//         else
//         {
//             REQUIRE(mpiData.left == mpi_rank);
//             REQUIRE(mpiData.right == mpi_rank);
//         }

//         if (lv.getN() > 0 && mpi_dims[1] > 1)
//         {
//             REQUIRE(mpiData.up == other_rank);
//             REQUIRE(mpiData.down == other_rank);
//         }
//         else
//         {
//             REQUIRE(mpiData.up == mpi_rank);
//             REQUIRE(mpiData.down == mpi_rank);
//         }

//         if (lv.getO() > 0 && mpi_dims[2] > 1)
//         {
//             REQUIRE(mpiData.front == other_rank);
//             REQUIRE(mpiData.back == other_rank);
//         }
//         else
//         {
//             REQUIRE(mpiData.front == mpi_rank);
//             REQUIRE(mpiData.back == mpi_rank);
//         }
//     }
// }
