#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <iostream>
#include <memory>

#include "../../src/cuboid.hpp"
#include "../../src/multigrid_engine.hpp"
#include "../../src/problem.hpp"
// #include "test_utility.hpp"

#include "mpi.h"

// Checks that an exception is thrown if the communicator has no cartesian topology attached.
TEST_CASE("MPI_Problem::setMpiComm")
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
// Run with: mpiexec -n 1 tests_mpi MPI_Problem::calculateAndSetMaxLevel_1proc
TEST_CASE("MPI_Problem::calculateAndSetMaxLevel_1proc")
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
// Run with: mpiexec -n 4 tests_mpi MPI_Problem::calculateAndSetMaxLevel_4procs
TEST_CASE("MPI_Problem::calculateAndSetMaxLevel_4procs")
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
// This test must be run on multiple processes.
// Run with e.g. mpiexec -n 2 tests_mpi MPI_Problem::calculateAndSetMpiLevelThreshold_valid_2procs
TEST_CASE("MPI_Problem::calculateAndSetMpiLevelThreshold_valid_2procs")
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
    REQUIRE(mpi_size > 1);

    // Set MPI cartesion topology dimenions and appropriate local sizes.
    // clang-format off
    std::vector<std::vector<int>> dims = {
        {4,4,4}, {2,2,2}, {4,4,4}, 
        {1,2,4},{1,2,2}, {2,2,1}
    };

    std::vector<std::vector<int>> local = {
        {4,4,4}, {8,8,8}, {8,8,8}, 
        {4,4,4}, {16,16,16}, {64,64,128}
    };

    std::vector<int> expected = {1,2,2,1,3,5};

    REQUIRE(dims.size() == local.size());
    REQUIRE(dims.size() == expected.size());
    // clang-format on

    // Check for valid with default minGridPoints = 4
    for (int i = 0; i < dims.size(); i++)
    {
        CAPTURE(i);
        // Check with valid topology first
        mgcl::Problem p(local[i][0], local[i][1], local[i][2],
                        local[i][0] * dims[i][0], local[i][1] * dims[i][1], local[i][2] * dims[i][2]);
        // p.setMpiComm(mpi_comm);
        p.calculateAndSetMpiLevelThreshold();

        REQUIRE(p.getMpiLevelThreshold() == expected[i]);
    }

    // Now check with different minGridPoints set by user
    {
        mgcl::Problem p(8, 8, 8, 16, 16, 16);
        p.setMpiMinGridPoints(8);
        p.calculateAndSetMpiLevelThreshold();
        REQUIRE(p.getMpiLevelThreshold() == 1);
    }

    {
        mgcl::Problem p(8, 8, 8, 16, 16, 16);
        p.setMpiMinGridPoints(16);
        p.calculateAndSetMpiLevelThreshold();
        REQUIRE(p.getMpiLevelThreshold() == 0);
    }
}

// calculateAndSetMpiLevelThreshold should be throw if mpiLevelThreshold is set too high by user.
// This test must be run on multiple processes.
// run with e.g. tests_mpi MPI_Problem::calculateAndSetMpiLevelThreshold_throwing_2procs
TEST_CASE("MPI_Problem::calculateAndSetMpiLevelThreshold_throwing_2procs")
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
    REQUIRE(mpi_size > 1);

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

    {
        mgcl::Problem p(N, N, N);
        REQUIRE_NOTHROW(p.setMpiComm(mpi_comm));
        REQUIRE_THROWS(p.setMpiMinGridPoints(1)); // calls calculateAndSetMpiLevelThreshold
    }

    {
        mgcl::Problem p(8, 8, 8, 16, 16, 16);
        REQUIRE_THROWS(p.setMpiMinGridPoints(32)); // calls calculateAndSetMpiLevelThreshold
    }
}

// Input data shall be copied to level 0 on each processor, regardless of mpiLevelThreshold.
// Can be run with any number of processes, e.g.
// mpiexec -n 8 tests_mpi MPI_Problem::init
TEST_CASE("MPI_Problem::init")
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
    // REQUIRE(mpi_size == 2);

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

    // Init data (same for each process)
    auto v = std::make_shared<mgcl::Cuboid>(m, n, o, 1, 1, 1);
    auto f = std::make_shared<mgcl::Cuboid>(m, n, o, 1, 1, 1);
    v->fill1dIndex(true);
    f->fill1dIndex(true);
    auto vl = std::shared_ptr<mgcl::Cuboid>(v->slice(m_start, m_end, n_start, n_end, o_start, o_end));
    auto fl = std::shared_ptr<mgcl::Cuboid>(f->slice(m_start, m_end, n_start, n_end, o_start, o_end));

    mgcl::Problem p(ml, nl, ol, vl, fl, m, n, o);
    p.setMpiComm(mpi_comm);
    p.setMpiLevelThreshold(0);
    p.setGhostsIn(1);
    p.init();

    REQUIRE(p.getMpiLevelThreshold() == 0);

    if (mpi_rank == 0)
    {
        // On rank 0, sizes are global sizes for mpiLevelThreshold = 0.
        REQUIRE(p.getLevelAt(0).getV().getM() == p.getMGlobal());
        REQUIRE(p.getLevelAt(0).getV().getN() == p.getNGlobal());
        REQUIRE(p.getLevelAt(0).getV().getO() == p.getOGlobal());
        REQUIRE(p.getLevelAt(0).getF().getM() == p.getMGlobal());
        REQUIRE(p.getLevelAt(0).getF().getN() == p.getNGlobal());
        REQUIRE(p.getLevelAt(0).getF().getO() == p.getOGlobal());
        REQUIRE(p.getLevelAt(0).getR().getM() == p.getMGlobal());
        REQUIRE(p.getLevelAt(0).getR().getN() == p.getNGlobal());
        REQUIRE(p.getLevelAt(0).getR().getO() == p.getOGlobal());

        // Compare only the local slices for input data that Problem holds.
        auto vloc = p.getLevelAt(0).getV().slice(m_start, m_end, n_start, n_end, o_start, o_end);
        auto floc = p.getLevelAt(0).getF().slice(m_start, m_end, n_start, n_end, o_start, o_end);
        REQUIRE(vloc->isEqual(p.getV()));
        REQUIRE(floc->isEqual(p.getF()));

        // Compare global data, that level 0 should hold (gathering happens in Level::init).
        REQUIRE(p.getLevelAt(0).getV().isEqual(*v));
        REQUIRE(p.getLevelAt(0).getF().isEqual(*f));

        // Ghosts of F should be up-to-date, too.
        mgcl::MultigridEngine::updateGhostsSeq(*f, nullptr, true, true);
        REQUIRE(p.getLevelAt(0).getF().isEqualAllCells(*f));
    }
    else
    {
        // On other processes, sizes are local sizes for regardless of mpiLevelThreshold.
        REQUIRE(p.getLevelAt(0).getV().getM() == p.getM());
        REQUIRE(p.getLevelAt(0).getV().getN() == p.getN());
        REQUIRE(p.getLevelAt(0).getV().getO() == p.getO());
        REQUIRE(p.getLevelAt(0).getF().getM() == p.getM());
        REQUIRE(p.getLevelAt(0).getF().getN() == p.getN());
        REQUIRE(p.getLevelAt(0).getF().getO() == p.getO());
        REQUIRE(p.getLevelAt(0).getR().getM() == p.getM());
        REQUIRE(p.getLevelAt(0).getR().getN() == p.getN());
        REQUIRE(p.getLevelAt(0).getR().getO() == p.getO());

        // For every other process, input data gets copied to level 0 1-to-1.
        REQUIRE(p.getV().isEqual(p.getLevelAt(0).getV()));
        REQUIRE(p.getF().isEqual(p.getLevelAt(0).getF()));
    }
}
