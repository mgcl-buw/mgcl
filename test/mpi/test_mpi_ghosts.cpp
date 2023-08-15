#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <iostream>
#include <memory>

#include "../../src/cuboid.hpp"
#include "../../src/multigrid_engine.hpp"
#include "../../src/problem.hpp"
#include "../test_utility.hpp"

#include "mpi.h"

// Checks ghost update for 1 process which is actually done locally without MPI calls.
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

        // Create test data
        mgcl::Cuboid c(N, N, N, gh, gh, gh);
        c.fillRandom(-10, 10, true);

        // Update ghosts of test data
        mgcl::MultigridEngine::updateGhostsSeq(c, nullptr, true);

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

// Checks ghost update for any number of processes that is allowed by mgcl, e.g. 1, 2, 4, 8, 24.
// Run with: mpiexec -n 8 tests_mpi [mpiN]
// TODO differentiate for gh>m and non-periodic
TEST_CASE("MPI updateGhostsSeq (n processes)", "[mpiN]")
{
    using std::min;

    // global grid sizes
    int m = 16;
    int n = 16;
    int o = 16;
    int periodic = 1;

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
    int n_start = (m / mpi_dims[1]) * mpi_coords[1] + min(mpi_coords[1], (n % mpi_dims[1]));
    int n_end = (m / mpi_dims[1]) * (mpi_coords[1] + 1) + min(mpi_coords[1] + 1, (n % mpi_dims[1])) - 1;
    int o_start = (m / mpi_dims[2]) * mpi_coords[2] + min(mpi_coords[2], (o % mpi_dims[2]));
    int o_end = (m / mpi_dims[2]) * (mpi_coords[2] + 1) + min(mpi_coords[2] + 1, (o % mpi_dims[2])) - 1;

    int ml = (m_end - m_start) + 1;
    int nl = (n_end - n_start) + 1;
    int ol = (o_end - o_start) + 1;

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

    // Init some random data (will be unused)
    auto v = std::make_shared<mgcl::Cuboid>(ml, nl, ol);
    auto f = std::make_shared<mgcl::Cuboid>(ml, nl, ol);
    v->fillRandom();
    f->fillRandom();

    SECTION("gh = 1 < m,n,o")
    {
        int gh = 1;

        // Init Problem to create all needed structures
        mgcl::Problem p(ml, nl, ol, v, f, m, n, o);
        p.setGhosts(1);
        p.setMpiComm(mpi_comm);
        p.init();

        // Check on level 0
        auto &lv = p.getLevelAt(0);
        auto mpiData = lv.getMpiDataPtr();

        // print neighbours per rank
        // for (int i = 0; i < mpi_size; i++)
        // {
        //     MPI_Barrier(mpi_comm);
        //     if (i == mpi_rank)
        //         std::cout << mpi_rank << ": " << mpiData->left << "," << mpiData->right << ","
        //                   << mpiData->up << "," << mpiData->down << ","
        //                   << mpiData->back << "," << mpiData->front << std::endl;
        // }

        // Create global test data. No random data so values will be the same for all processes. Fill with 1d index.
        mgcl::Cuboid cg(m, n, o, gh, gh, gh);
        int cnt = 0;
        for (int i = gh; i < m + gh; i++)
            for (int j = gh; j < n + gh; j++)
                for (int k = gh; k < o + gh; k++)
                {
                    cg[i][j][k] = cnt++;
                }

        // Update ghosts of expected result locally, i.e. not using MPI routines.
        mgcl::MultigridEngine::updateGhostsSeq(cg, nullptr, true);

        // Create local slice of global data
        auto clptr = cg.slice(m_start, m_end, n_start, n_end, o_start, o_end);
        auto &cl = *clptr;

        // Update ghosts of test data
        mgcl::MultigridEngine::updateGhostsSeq(cl, mpiData, true);

        // cl.dumpToFile("cl" + std::to_string(mpi_rank) + ".txt");

        // if (mpi_rank == 0)
        // {
        //     cg.dumpToFile("cg.txt");

        // Check result
        // check in z-direction
        for (int i = 0; i < gh; i++)
            for (int j = 0; j < nl + 2 * gh; j++)
                for (int k = 0; k < ol + 2 * gh; k++)
                {
                    if (mpi_coords[0] > 0) // not the first process
                        REQUIRE(cl[i][j][k] == cg[m_start + i][j + n_start][k + o_start]);
                    else
                        REQUIRE(cl[i][j][k] == cg[i + m][j + n_start][k + o_start]);

                    if (mpi_coords[0] < mpi_dims[0]) // not the last process
                        REQUIRE(cl[i + gh + ml][j][k] == cg[m_end + gh + 1 + i][j + n_start][k + o_start]);
                    else
                        REQUIRE(cl[i + gh + ml][j][k] == cg[i + gh][j + n_start][k + o_start]);
                }

        // check in y-direction
        for (int i = 0; i < ml + 2 * gh; i++)
            for (int j = 0; j < gh; j++)
                for (int k = 0; k < ol + 2 * gh; k++)
                {
                    if (mpi_coords[1] > 0) // not the first process
                        REQUIRE(cl[i][j][k] == cg[i + m_start][n_start + j][k + o_start]);
                    else
                        REQUIRE(cl[i][j][k] == cg[i + m_start][j + n][k + o_start]);

                    if (mpi_coords[1] < mpi_dims[1]) // not the last process
                        REQUIRE(cl[i][j + gh + nl][k] == cg[i + m_start][n_end + gh + 1 + j][k + o_start]);
                    else
                        REQUIRE(cl[i][j + gh + nl][k] == cg[i + m_start][j + gh][k + o_start]);
                }

        // check in x-direction
        for (int i = 0; i < ml + 2 * gh; i++)
            for (int j = 0; j < nl + 2 * gh; j++)
                for (int k = 0; k < gh; k++)
                {
                    if (mpi_coords[2] > 0) // not the first process
                        REQUIRE(cl[i][j][k] == cg[i + m_start][j + n_start][o_start + k]);
                    else
                        REQUIRE(cl[i][j][k] == cg[i + m_start][j + n_start][k + o]);

                    if (mpi_coords[2] < mpi_dims[2]) // not the last process
                        REQUIRE(cl[i][j][k + gh + ol] == cg[i + m_start][j + n_start][o_end + gh + 1 + k]);
                    else
                        REQUIRE(cl[i][j][k + gh + ol] == cg[i + m_start][j + n_start][k + gh]);
                }
        // }
    }
}

// Checks ghost update for any number of processes that is allowed by mgcl, e.g. 1, 2, 4, 8, 24.
// Run with: mpiexec -n 8 tests_mpi "MPI updateGhosts ocl (n processes)"
// TODO differentiate for gh>m and non-periodic
TEST_CASE("MPI updateGhosts ocl (n processes)", "[mpiN]")
{
    using std::min;

    // global grid sizes
    int m = 16;
    int n = 16;
    int o = 16;
    int periodic = 1;

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
    int n_start = (m / mpi_dims[1]) * mpi_coords[1] + min(mpi_coords[1], (n % mpi_dims[1]));
    int n_end = (m / mpi_dims[1]) * (mpi_coords[1] + 1) + min(mpi_coords[1] + 1, (n % mpi_dims[1])) - 1;
    int o_start = (m / mpi_dims[2]) * mpi_coords[2] + min(mpi_coords[2], (o % mpi_dims[2]));
    int o_end = (m / mpi_dims[2]) * (mpi_coords[2] + 1) + min(mpi_coords[2] + 1, (o % mpi_dims[2])) - 1;

    int ml = (m_end - m_start) + 1;
    int nl = (n_end - n_start) + 1;
    int ol = (o_end - o_start) + 1;

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

    // Init some random data (will be unused)
    auto v = std::make_shared<mgcl::Cuboid>(ml, nl, ol);
    auto f = std::make_shared<mgcl::Cuboid>(ml, nl, ol);
    v->fillRandom();
    f->fillRandom();

    int gh = 1;

    // Init Problem to create all needed structures
    auto pptr = std::make_shared<mgcl::Problem>(ml, nl, ol, v, f, m, n, o);
    auto &p = *pptr;
    p.setGhosts(1);
    p.setMpiComm(mpi_comm);
    p.init();

    // Check on level 0
    auto &lv = p.getLevelAt(0);
    auto mpiData = lv.getMpiDataPtr();

    // print neighbours per rank
    // for (int i = 0; i < mpi_size; i++)
    // {
    //     MPI_Barrier(mpi_comm);
    //     if (i == mpi_rank)
    //         std::cout << mpi_rank << ": " << mpiData->left << "," << mpiData->right << ","
    //                   << mpiData->up << "," << mpiData->down << ","
    //                   << mpiData->back << "," << mpiData->front << std::endl;
    // }

    // Create global test data. No random data so values will be the same for all processes. Fill with 1d index.
    mgcl::Cuboid cg(m, n, o, gh, gh, gh);
    int cnt = 0;
    for (int i = gh; i < m + gh; i++)
        for (int j = gh; j < n + gh; j++)
            for (int k = gh; k < o + gh; k++)
            {
                cg[i][j][k] = cnt++;
            }

    // Update ghosts of expected result locally, i.e. not using MPI routines.
    mgcl::MultigridEngine::updateGhostsSeq(cg, nullptr, true);

    // Create local slice of global data
    auto clptr = cg.slice(m_start, m_end, n_start, n_end, o_start, o_end);
    auto &cl = *clptr;

    // Create ocl buffer
    mgcl_test::TestUtility tu(pptr);
    auto d_cl = tu.createOpenCLBuffer(cl);

    // Update ghosts of test data
    mgcl::MultigridEngine::updateGhosts(p, d_cl, ml + 2 * gh, nl + 2 * gh, ol + 2 * gh, gh, gh, gh, mpiData);
    tu.finish();

    auto cl_res_ptr = tu.readOpenCLBuffer(d_cl, ml, nl, ol, gh, gh, gh);
    auto &cl_res = *cl_res_ptr;

    // cl.dumpToFile("cl" + std::to_string(mpi_rank) + ".txt");

    // if (mpi_rank == 0)
    // {
    //     cg.dumpToFile("cg.txt");

    // Check result
    // check in z-direction
    for (int i = 0; i < gh; i++)
        for (int j = 0; j < nl + 2 * gh; j++)
            for (int k = 0; k < ol + 2 * gh; k++)
            {
                if (mpi_coords[0] > 0) // not the first process
                    REQUIRE(cl_res[i][j][k] == cg[m_start + i][j + n_start][k + o_start]);
                else
                    REQUIRE(cl_res[i][j][k] == cg[i + m][j + n_start][k + o_start]);

                if (mpi_coords[0] < mpi_dims[0]) // not the last process
                    REQUIRE(cl_res[i + gh + ml][j][k] == cg[m_end + gh + 1 + i][j + n_start][k + o_start]);
                else
                    REQUIRE(cl_res[i + gh + ml][j][k] == cg[i + gh][j + n_start][k + o_start]);
            }

    // check in y-direction
    for (int i = 0; i < ml + 2 * gh; i++)
        for (int j = 0; j < gh; j++)
            for (int k = 0; k < ol + 2 * gh; k++)
            {
                if (mpi_coords[1] > 0) // not the first process
                    REQUIRE(cl_res[i][j][k] == cg[i + m_start][n_start + j][k + o_start]);
                else
                    REQUIRE(cl_res[i][j][k] == cg[i + m_start][j + n][k + o_start]);

                if (mpi_coords[1] < mpi_dims[1]) // not the last process
                    REQUIRE(cl_res[i][j + gh + nl][k] == cg[i + m_start][n_end + gh + 1 + j][k + o_start]);
                else
                    REQUIRE(cl_res[i][j + gh + nl][k] == cg[i + m_start][j + gh][k + o_start]);
            }

    // check in x-direction
    for (int i = 0; i < ml + 2 * gh; i++)
        for (int j = 0; j < nl + 2 * gh; j++)
            for (int k = 0; k < gh; k++)
            {
                if (mpi_coords[2] > 0) // not the first process
                    REQUIRE(cl_res[i][j][k] == cg[i + m_start][j + n_start][o_start + k]);
                else
                    REQUIRE(cl_res[i][j][k] == cg[i + m_start][j + n_start][k + o]);

                if (mpi_coords[2] < mpi_dims[2]) // not the last process
                    REQUIRE(cl_res[i][j][k + gh + ol] == cg[i + m_start][j + n_start][o_end + gh + 1 + k]);
                else
                    REQUIRE(cl_res[i][j][k + gh + ol] == cg[i + m_start][j + n_start][k + gh]);
            }
    // }
}
