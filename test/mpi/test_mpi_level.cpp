#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <iostream>
#include <memory>

#include "../../src/mgcl/cuboid.hpp"
#include "../../src/mgcl/problem.hpp"
// #include "test_utility.hpp"

#include "mpi.h"

// Checks if neighbours are initialized correctly for each level for 1 process.
// Run with: mpiexec -n 1 tests_mpi "Level::initMpiData (1 process)"
TEST_CASE("Level::initMpiData (1 process)")
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
    auto v = std::make_shared<mgcl::Cuboid>(N, N, N, 1, 1, 1);
    auto f = std::make_shared<mgcl::Cuboid>(N, N, N, 1, 1, 1);
    v->fillRandom();
    f->fillRandom();

    mgcl::Problem p(N, N, N, v, f);
    p.setMpiComm(mpi_comm);
    p.setGhostsIn(1);
    p.init();

    for (int i = 0; i < p.getMaxlevel(); i++)
    {
        auto& lv = p.getLevelAt(i);
        REQUIRE(lv.getMpiDataPtr() == nullptr);
        REQUIRE_THROWS(lv.getMpiData());
        REQUIRE(lv.isCalculatedLocally());

        // Each level must equal the local size of the problem divided by 2^num.
        REQUIRE((p.getM() >> i) == lv.getM());
        REQUIRE((p.getN() >> i) == lv.getN());
        REQUIRE((p.getO() >> i) == lv.getO());
    }
}

// Checks if neighbours are initialized correctly for each level for 2 processes.
// Run with: mpiexec -n 2 tests_mpi "Level::initMpiData (2 processes)"
TEST_CASE("Level::initMpiData (2 processes)")
{
    using std::min;

    // global grid size
    int m = 8;
    int n = 8;
    int o = 8;
    int periodic = 1;

    // check if mpi is initialized
    int isInitialized = 0;
    MPI_Initialized(&isInitialized);
    REQUIRE(isInitialized);

    MPI_Comm mpi_comm = MPI_COMM_WORLD;

    // check number of processes
    int mpi_size = -1;
    MPI_Comm_size(mpi_comm, &mpi_size);
    REQUIRE(mpi_size == 2);

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

    // Init some random data
    auto v = std::make_shared<mgcl::Cuboid>(ml, nl, ol, 1, 1, 1);
    auto f = std::make_shared<mgcl::Cuboid>(ml, nl, ol, 1, 1, 1);
    v->fillRandom();
    f->fillRandom();

    mgcl::Problem p(ml, nl, ol, f, v, m, n, o);
    p.setMpiComm(mpi_comm);
    p.setGhostsIn(1);
    p.init();

    int other_rank = mpi_rank == 0 ? 1 : 0;

    for (int i = 0; i < p.getMaxlevel(); i++)
    {
        auto& lv = p.getLevelAt(i);

        if (p.getMpiLevelThreshold() > i)
        {
            auto& mpiData = lv.getMpiData();
            REQUIRE(!lv.isCalculatedLocally());

            // Each level below the threshold must equal the local size of the problem divided by 2^num.
            REQUIRE((p.getM() >> i) == lv.getM());
            REQUIRE((p.getN() >> i) == lv.getN());
            REQUIRE((p.getO() >> i) == lv.getO());

            REQUIRE(mpiData.rank == mpi_rank);

            // If there are two processes in a direction, the neighbour must be the other process. Else or if no
            // grid points are left on this level, the neighbour is set to self.
            if (lv.getM() > 0 && mpi_dims[2] > 1)
            {
                REQUIRE(mpiData.left == other_rank);
                REQUIRE(mpiData.right == other_rank);
            }
            else
            {
                REQUIRE(mpiData.left == mpi_rank);
                REQUIRE(mpiData.right == mpi_rank);
            }

            if (lv.getN() > 0 && mpi_dims[1] > 1)
            {
                REQUIRE(mpiData.up == other_rank);
                REQUIRE(mpiData.down == other_rank);
            }
            else
            {
                REQUIRE(mpiData.up == mpi_rank);
                REQUIRE(mpiData.down == mpi_rank);
            }

            if (lv.getO() > 0 && mpi_dims[0] > 1)
            {
                REQUIRE(mpiData.front == other_rank);
                REQUIRE(mpiData.back == other_rank);
            }
            else
            {
                REQUIRE(mpiData.front == mpi_rank);
                REQUIRE(mpiData.back == mpi_rank);
            }
        }
        else
        {
            REQUIRE(lv.isCalculatedLocally());
            REQUIRE(lv.getMpiDataPtr() != nullptr);
            // REQUIRE(lv.getMpiDataPtr() == nullptr);
            // REQUIRE_THROWS(lv.getMpiData());

            // Each level on rank 0 at or above the threshold must equal the global size of the problem divided by 2^num.
            // v, f and r must not be null.
            if (mpi_rank == 0)
            {
                REQUIRE((p.getMGlobal() >> i) == lv.getM());
                REQUIRE((p.getNGlobal() >> i) == lv.getN());
                REQUIRE((p.getOGlobal() >> i) == lv.getO());
            }
            // On other ranks, size is based on the local size of the problem and v, f and r are null.
            else
            {
                REQUIRE((p.getM() >> i) == lv.getM());
                REQUIRE((p.getN() >> i) == lv.getN());
                REQUIRE((p.getO() >> i) == lv.getO());
                // REQUIRE(lv.getVPtr() == nullptr);
                // REQUIRE(lv.getFPtr() == nullptr);
                // REQUIRE(lv.getRPtr() == nullptr);
            }
        }
    }
}

// Checks if neighbours are initialized correctly for each level for 2 processes when level treshold is at 0, i.e.
// actually everything is done on proc 0. But data must be initialized for level 0 on every process.
// Run with: mpiexec -n 2 tests_mpi Level::initMpiData-2procs-levelThreshold0
TEST_CASE("Level::initMpiData-2procs-levelThreshold0")
{
    using std::min;

    // global grid size
    int m = 8;
    int n = 8;
    int o = 8;
    int periodic = 1;

    // check if mpi is initialized
    int isInitialized = 0;
    MPI_Initialized(&isInitialized);
    REQUIRE(isInitialized);

    MPI_Comm mpi_comm = MPI_COMM_WORLD;

    // check number of processes
    int mpi_size = -1;
    MPI_Comm_size(mpi_comm, &mpi_size);
    REQUIRE(mpi_size == 2);

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

    // Init some random data
    auto v = std::make_shared<mgcl::Cuboid>(ml, nl, ol, 1, 1, 1);
    auto f = std::make_shared<mgcl::Cuboid>(ml, nl, ol, 1, 1, 1);
    v->fillRandom();
    f->fillRandom();

    mgcl::Problem p(ml, nl, ol, f, v, m, n, o);
    p.setMpiComm(mpi_comm);
    p.setGhostsIn(1);
    p.setMpiMinGridPoints(m); // Ensures mpiLevelThreshold to be 0.
    p.init();

    REQUIRE(p.getMpiLevelThreshold() == 0);

    // Check level 0: Data must be initialized on every process. Neighbours won't be initialized.
    {
        auto& lv = p.getLevelAt(0);

        REQUIRE(lv.isCalculatedLocally());
        REQUIRE(lv.getMpiDataPtr() != nullptr);
        // REQUIRE(lv.getMpiDataPtr() == nullptr);
        // REQUIRE_THROWS(lv.getMpiData());

        // Each level on rank 0 at or above the threshold must equal the global size of the problem divided by 2^num.
        // v, f and r must not be null.
        if (mpi_rank == 0)
        {
            REQUIRE(p.getMGlobal() == lv.getM());
            REQUIRE(p.getNGlobal() == lv.getN());
            REQUIRE(p.getOGlobal() == lv.getO());
        }
        // On other ranks, size is based on the local size of the problem. v, f and r must not be null for lv 0.
        else
        {
            REQUIRE(p.getM() == lv.getM());
            REQUIRE(p.getN() == lv.getN());
            REQUIRE(p.getO() == lv.getO());
        }

        REQUIRE(lv.getVPtr() != nullptr);
        REQUIRE(lv.getFPtr() != nullptr);
        REQUIRE(lv.getRPtr() != nullptr);

        auto& mpiData = lv.getMpiData();
        REQUIRE(mpiData.rank == mpi_rank);
    }

    // On level 0, each rank has data allocated.
    // On level >= 1 only rank 0 has data allocated.
    // Neighbours don't care, since work is done locally on rank 0.
    for (int i = 0; i < p.getMaxlevel(); i++)
    {
        auto& lv = p.getLevelAt(i);

        REQUIRE(lv.isCalculatedLocally());
        REQUIRE(lv.getMpiDataPtr() != nullptr);

        // Each level on rank 0 at or above the threshold must equal the global size of the problem divided by 2^num.
        // v, f and r must not be null.
        if (mpi_rank == 0)
        {
            REQUIRE((p.getMGlobal() >> i) == lv.getM());
            REQUIRE((p.getNGlobal() >> i) == lv.getN());
            REQUIRE((p.getOGlobal() >> i) == lv.getO());
        }
        // On other ranks, size is based on the local size of the problem. v, f and r are only null for lv > 0.
        else
        {
            REQUIRE((p.getM() >> i) == lv.getM());
            REQUIRE((p.getN() >> i) == lv.getN());
            REQUIRE((p.getO() >> i) == lv.getO());

            // if (i == 0)
            // {
            REQUIRE(lv.getVPtr() != nullptr);
            REQUIRE(lv.getFPtr() != nullptr);
            REQUIRE(lv.getRPtr() != nullptr);
            // }
            // else
            // {
            //     REQUIRE(lv.getVPtr() == nullptr);
            //     REQUIRE(lv.getFPtr() == nullptr);
            //     REQUIRE(lv.getRPtr() == nullptr);
            // }
        }

        auto& mpiData = lv.getMpiData();
        REQUIRE(mpiData.rank == mpi_rank);
    }
}

// Checks if neighbours are initialized correctly for each level for 8 processes.
// Run with: mpiexec -n 8 tests_mpi "Level::initMpiData (8 processes)"
TEST_CASE("Level::initMpiData (8 processes)")
{
    int N = 4;      // local size of grid in one direction
    int Ng = N * 2; // global size of grid in one direction
    int periodic = 1;

    // check if mpi is initialized
    int isInitialized = 0;
    MPI_Initialized(&isInitialized);
    REQUIRE(isInitialized);

    MPI_Comm mpi_comm = MPI_COMM_WORLD;

    // check number of processes
    int mpi_size = -1;
    MPI_Comm_size(mpi_comm, &mpi_size);
    REQUIRE(mpi_size == 8);

    /* MPI variables */
    int mpi_rank;
    int mpi_dims[3] = {2, 2, 2};
    int mpi_periods[3] = {periodic, periodic, periodic};
    int mpi_coords[3];

    /* Initialize cartesian process grid */
    // No need for MPI_Dims_create, since dims are given explicitely to MPI_Cart_create (2x2x2).
    // Disable reordering in MPI_Cart_create, s.t. a process with given rank will always have the same coordinates.
    MPI_Cart_create(mpi_comm, 3, mpi_dims, mpi_periods, 0, &mpi_comm);
    MPI_Comm_rank(mpi_comm, &mpi_rank);
    MPI_Cart_coords(mpi_comm, mpi_rank, 3, mpi_coords);

    // if (mpi_rank == 0)
    //     std::cout << "rank;coords" << std::endl;

    // MPI_Barrier(mpi_comm);
    // std::cout << mpi_rank << ";" << mpi_coords[0] << ";" << mpi_coords[1] << ";" << mpi_coords[2] << std::endl;
    // MPI_Barrier(mpi_comm);

    // Check that coordinates are correct for our test setup, i.e.
    // rank 0: 0,0,0,
    // rank 1: 0,0,1,
    // rank 2: 0,1,0 etc.
    REQUIRE(mpi_rank == ((mpi_coords[0] << 2) + (mpi_coords[1] << 1) + mpi_coords[2]));

    // Init some random data
    auto v = std::make_shared<mgcl::Cuboid>(N, N, N, 1, 1, 1);
    auto f = std::make_shared<mgcl::Cuboid>(N, N, N, 1, 1, 1);
    v->fillRandom();
    f->fillRandom();

    mgcl::Problem p(N, N, N, f, v, Ng, Ng, Ng);
    p.setMpiComm(mpi_comm);
    p.setGhostsIn(1);
    p.setMpiMinGridPoints(2);
    p.init();

    // Cartesian topology layout:
    //       z=0        z=1
    //    +---+---+  +---+---+
    // y  + 0 + 1 +  + 4 + 5 +
    // |  +---+---+  +---+---+
    // v  + 2 + 3 +  + 6 + 7 +
    //    +---+---+  +---+---+
    //       x->        x->

    // Grid points per process and level in one direction:
    // -----+-- p1 ---+-- p2 ---+
    // lv 0 | * * * * | * * * * |
    // lv 1 |   *   * |   *   * |
    // lv 2 |       * |       * |
    // lv 3 |         |       * |

    // print neighbours per rank
    // for (int i = 0; i < mpi_size; i++)
    // {
    //     MPI_Barrier(mpi_comm);
    //     if (i == mpi_rank)
    //         std::cout << mpi_rank << ": " << mpiData0.left << "," << mpiData0.right << ","
    //                   << mpiData0.up << "," << mpiData0.down << ","
    //                   << mpiData0.back << "," << mpiData0.front << std::endl;
    // }

    // ==================================

    // Check levels for which mpi is used
    for (int i = 0; i < p.getMpiLevelThreshold(); i++)
    {
        auto& lv = p.getLevelAt(i);
        auto& mpiData0 = lv.getMpiData();
        REQUIRE(mpiData0.mpiSize() == 8);
        REQUIRE(!lv.isCalculatedLocally());

        // Each level below the threshold must equal the local size of the problem divided by 2^num.
        REQUIRE((p.getM() >> i) == lv.getM());
        REQUIRE((p.getN() >> i) == lv.getN());
        REQUIRE((p.getO() >> i) == lv.getO());

        // clang-format off
    std::vector<std::vector<int>> neighbours0 = {
        {1, 2, 4}, {0, 3, 5}, {3, 0, 6}, {2, 1, 7},
        {5, 6, 0}, {4, 7, 1}, {7, 4, 2}, {6, 5, 3}
    };
        // clang-format on

        REQUIRE(mpiData0.left == neighbours0[mpi_rank][0]);
        REQUIRE(mpiData0.right == neighbours0[mpi_rank][0]);
        REQUIRE(mpiData0.up == neighbours0[mpi_rank][1]);
        REQUIRE(mpiData0.down == neighbours0[mpi_rank][1]);
        REQUIRE(mpiData0.front == neighbours0[mpi_rank][2]);
        REQUIRE(mpiData0.back == neighbours0[mpi_rank][2]);
    }

    // ==================================

    // Check levels for which mpi is not used
    for (int i = p.getMpiLevelThreshold(); i <= p.getMaxlevel(); i++)
    {
        auto& lv = p.getLevelAt(i);
        REQUIRE(lv.isCalculatedLocally());
        REQUIRE(lv.getMpiDataPtr() != nullptr);
        // REQUIRE_THROWS(lv.getMpiData());

        // Each level on rank 0 at or above the threshold must equal the global size of the problem divided by 2^num.
        // v, f and r must not be null.
        if (mpi_rank == 0)
        {
            REQUIRE((p.getMGlobal() >> i) == lv.getM());
            REQUIRE((p.getNGlobal() >> i) == lv.getN());
            REQUIRE((p.getOGlobal() >> i) == lv.getO());
        }
        // On other ranks, size is based on the local size of the problem and v, f and r are null.
        else
        {
            REQUIRE((p.getM() >> i) == lv.getM());
            REQUIRE((p.getN() >> i) == lv.getN());
            REQUIRE((p.getO() >> i) == lv.getO());
            // REQUIRE(lv.getVPtr() == nullptr);
            // REQUIRE(lv.getFPtr() == nullptr);
            // REQUIRE(lv.getRPtr() == nullptr);
        }
    }
}

// Checks if neighbours are initialized correctly for each level for 24 processes and a non-uniform domain.
// Run with: mpiexec --oversubscribe -n 24 tests_mpi "Level::initMpiData (24 processes)"
TEST_CASE("Level::initMpiData (24 processes)")
{
    int m = 8; // local size of grid in one direction
    int n = 4;
    int o = 4;
    int mg = m * 2; // global size of grid in one direction
    int ng = n * 2; // global size of grid in one direction
    int og = o * 2; // global size of grid in one direction
    int periodic = 1;

    // check if mpi is initialized
    int isInitialized = 0;
    MPI_Initialized(&isInitialized);
    REQUIRE(isInitialized);

    MPI_Comm mpi_comm = MPI_COMM_WORLD;

    // check number of processes
    int mpi_size = -1;
    MPI_Comm_size(mpi_comm, &mpi_size);
    REQUIRE(mpi_size == 24);

    /* MPI variables */
    int mpi_rank;
    int mpi_dims[3] = {4, 3, 2};
    int mpi_periods[3] = {periodic, periodic, periodic};
    int mpi_coords[3];

    /* Initialize cartesian process grid */
    // No need for MPI_Dims_create, since dims are given explicitely to MPI_Cart_create (2x2x2).
    // Disable reordering in MPI_Cart_create, s.t. a process with given rank will always have the same coordinates.
    MPI_Cart_create(mpi_comm, 3, mpi_dims, mpi_periods, 0, &mpi_comm);
    MPI_Comm_rank(mpi_comm, &mpi_rank);
    MPI_Cart_coords(mpi_comm, mpi_rank, 3, mpi_coords);

    // if (mpi_rank == 0)
    //     std::cout << "rank;coords" << std::endl;

    // for (int i = 0; i < mpi_size; i++)
    // {
    //     MPI_Barrier(mpi_comm);
    //     if (i == mpi_rank)
    //         std::cout << mpi_rank << ";" << mpi_coords[0] << ";" << mpi_coords[1] << ";" << mpi_coords[2] << std::endl;
    // }

    // Check that coordinates are correct for our test setup, i.e.
    // rank 0: 0,0,0,
    // rank 1: 0,0,1,
    // rank 2: 0,1,0 etc.
    REQUIRE(mpi_rank == ((mpi_coords[0] * mpi_dims[1] * mpi_dims[2]) + (mpi_coords[1] * mpi_dims[2]) + mpi_coords[2]));

    // Init some random data
    auto v = std::make_shared<mgcl::Cuboid>(m, n, o, 1, 1, 1);
    auto f = std::make_shared<mgcl::Cuboid>(m, n, o, 1, 1, 1);
    v->fillRandom();
    f->fillRandom();

    mgcl::Problem p(m, n, o, f, v, mg, ng, og);
    p.setMpiComm(mpi_comm);
    p.setGhostsIn(1);
    p.setMpiMinGridPoints(2);
    p.init();

    // Cartesian topology layout:
    //       z=0        z=1
    //    +---+---+  +---+---+
    // y  + 0 + 1 +  + 4 + 5 +
    // |  +---+---+  +---+---+
    // v  + 2 + 3 +  + 6 + 7 +
    //    +---+---+  +---+---+
    //       x->        x->

    // Grid points per process and level in one direction:
    // -----+-- p1 ---+-- p2 ---+
    // lv 0 | * * * * | * * * * |
    // lv 1 |   *   * |   *   * |
    // lv 2 |       * |       * |
    // lv 3 |         |       * |

    // print neighbours per rank
    // for (int i = 0; i < mpi_size; i++)
    // {
    //     MPI_Barrier(mpi_comm);
    //     if (i == mpi_rank)
    //         std::cout << mpi_rank << ": " << mpiData0.left << "," << mpiData0.right << ","
    //                   << mpiData0.up << "," << mpiData0.down << ","
    //                   << mpiData0.back << "," << mpiData0.front << std::endl;
    // }

    // ==================================

    // Check levels for which mpi is used
    for (int i = 0; i < p.getMpiLevelThreshold(); i++)
    {
        auto& lv = p.getLevelAt(i);
        auto& mpiData0 = lv.getMpiData();
        REQUIRE(mpiData0.mpiSize() == 24);
        REQUIRE(!lv.isCalculatedLocally());

        // Each level below the threshold must equal the local size of the problem divided by 2^num.
        REQUIRE((p.getM() >> i) == lv.getM());
        REQUIRE((p.getN() >> i) == lv.getN());
        REQUIRE((p.getO() >> i) == lv.getO());

        // print neighbours per rank
        // if (0 == mpi_rank)
        //     std::cout << "rank;left;right;up;down;back;front" << std::endl;
        // for (int i = 0; i < mpi_size; i++)
        // {
        //     MPI_Barrier(mpi_comm);
        //     if (i == mpi_rank)
        //         std::cout << mpi_rank << ": " << mpiData0.left << "," << mpiData0.right << ","
        //                   << mpiData0.up << "," << mpiData0.down << ","
        //                   << mpiData0.back << "," << mpiData0.front << std::endl;
        // }

        // clang-format off
        std::vector<std::vector<int>> neighbours0 = {
            {1,1,2,4,6,18}, {0,0,3,5,7,19}, {3,3,4,0,8,20}, {2,2,5,1,9,21},
            {5,5,0,2,10,22}, {4,4,1,3,11,23}, {7,7,8,10,12,0}, {6,6,9,11,13,1},
            {9,9,10,6,14,2}, {8,8,11,7,15,3}, {11,11,6,8,16,4}, {10,10,7,9,17,5},
            {13,13,14,16,18,6}, {12,12,15,17,19,7}, {15,15,16,12,20,8}, {14,14,17,13,21,9},
            {17,17,12,14,22,10}, {16,16,13,15,23,11}, {19,19,20,22,0,12}, {18,18,21,23,1,13},
            {21,21,22,18,2,14}, {20,20,23,19,3,15}, {23,23,18,20,4,16}, {22,22,19,21,5,17}
        };
        // clang-format on

        REQUIRE(mpiData0.left == neighbours0[mpi_rank][0]);
        REQUIRE(mpiData0.right == neighbours0[mpi_rank][1]);
        REQUIRE(mpiData0.up == neighbours0[mpi_rank][2]);
        REQUIRE(mpiData0.down == neighbours0[mpi_rank][3]);
        REQUIRE(mpiData0.back == neighbours0[mpi_rank][4]);
        REQUIRE(mpiData0.front == neighbours0[mpi_rank][5]);
    }

    // ==================================

    // Check levels for which mpi is not used
    for (int i = p.getMpiLevelThreshold(); i <= p.getMaxlevel(); i++)
    {
        auto& lv = p.getLevelAt(i);
        REQUIRE(lv.isCalculatedLocally());
        REQUIRE(lv.getMpiDataPtr() != nullptr);
        // REQUIRE_THROWS(lv.getMpiData());

        // Each level on rank 0 at or above the threshold must equal the global size of the problem divided by 2^num.
        // v, f and r must not be null.
        if (mpi_rank == 0)
        {
            REQUIRE((p.getMGlobal() >> i) == lv.getM());
            REQUIRE((p.getNGlobal() >> i) == lv.getN());
            REQUIRE((p.getOGlobal() >> i) == lv.getO());
        }
        // On other ranks, size is based on the local size of the problem and v, f and r are null.
        else
        {
            REQUIRE((p.getM() >> i) == lv.getM());
            REQUIRE((p.getN() >> i) == lv.getN());
            REQUIRE((p.getO() >> i) == lv.getO());
            // REQUIRE(lv.getVPtr() == nullptr);
            // REQUIRE(lv.getFPtr() == nullptr);
            // REQUIRE(lv.getRPtr() == nullptr);
        }
    }
}
