#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <iostream>
#include <memory>

#include "../../src/cuboid.hpp"
#include "../../src/mpi_stencil.hpp"
#include "../../src/multigrid_engine.hpp"
#include "../../src/problem.hpp"
#include "../test_utility.hpp"

#include "mpi.h"

// Checks stencil ghost update for 1 process which is actually done locally without MPI calls.
// Run with: mpiexec -n 1 tests_mpi MPI-stencil-updateGhostsSeq-1proc
// TODO differentiate for gh>m and non-periodic
TEST_CASE("MPI-stencil-updateGhostsSeq-1proc")
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

        // p.setStencilType(mgcl::MGCL_VARYING);
        // auto& s = *p.getStencilValues();
        // s.fill1dIndex(true);

        p.init();

        // Check on level 0
        auto& lv = p.getLevelAt(0);

        // Create test data
        mgcl::VaryingStencil3x3x3 s3(N, N, N, gh, gh, gh);
        s3.fillRandomInt(-10, 10, true);

        // Update ghosts of test data
        mgcl::updateGhostsStencilMpi(s3, nullptr, true, false);

        // check in z-direction
        for (int i = 0; i < gh; i++)
            for (int j = 0; j < n + 2 * gh; j++)
                for (int k = 0; k < o + 2 * gh; k++)
                    for (int ii = 0; ii < 3; ii++)
                        for (int jj = 0; jj < 3; jj++)
                            for (int kk = 0; kk < 3; kk++)
                            {
                                CHECK(s3[i][j][k][ii][jj][kk] == s3[i + m][j][k][ii][jj][kk]);
                                CHECK(s3[i + gh][j][k][ii][jj][kk] == s3[i + gh + m][j][k][ii][jj][kk]);
                            }

        // check in y-direction
        for (int i = 0; i < m + 2 * gh; i++)
            for (int j = 0; j < gh; j++)
                for (int k = 0; k < o + 2 * gh; k++)
                    for (int ii = 0; ii < 3; ii++)
                        for (int jj = 0; jj < 3; jj++)
                            for (int kk = 0; kk < 3; kk++)
                            {
                                CHECK(s3[i][j][k][ii][jj][kk] == s3[i][j + n][k][ii][jj][kk]);
                                CHECK(s3[i][j + gh][k][ii][jj][kk] == s3[i][j + gh + n][k][ii][jj][kk]);
                            }

        // check in x-direction
        for (int i = 0; i < m + 2 * gh; i++)
            for (int j = 0; j < n + 2 * gh; j++)
                for (int k = 0; k < gh; k++)
                    for (int ii = 0; ii < 3; ii++)
                        for (int jj = 0; jj < 3; jj++)
                            for (int kk = 0; kk < 3; kk++)
                            {
                                CHECK(s3[i][j][k][ii][jj][kk] == s3[i][j][k + o][ii][jj][kk]);
                                CHECK(s3[i][j][k + gh][ii][jj][kk] == s3[i][j][k + gh + o][ii][jj][kk]);
                            }
    }
}

// Checks ghost update for any number of processes that is allowed by mgcl, e.g. 1, 2, 4, 8, 24.
// Run with: mpiexec -n 8 tests_mpi MPI-stencil-updateGhostsSeq-nprocs
// TODO differentiate for gh>m and non-periodic
TEST_CASE("MPI-stencil-updateGhostsSeq-nprocs")
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
    int n_start = (n / mpi_dims[1]) * mpi_coords[1] + min(mpi_coords[1], (n % mpi_dims[1]));
    int n_end = (n / mpi_dims[1]) * (mpi_coords[1] + 1) + min(mpi_coords[1] + 1, (n % mpi_dims[1])) - 1;
    int o_start = (o / mpi_dims[2]) * mpi_coords[2] + min(mpi_coords[2], (o % mpi_dims[2]));
    int o_end = (o / mpi_dims[2]) * (mpi_coords[2] + 1) + min(mpi_coords[2] + 1, (o % mpi_dims[2])) - 1;

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

    int gh = GENERATE(1, 2);

    // Init Problem to create all needed structures
    mgcl::Problem p(ml, nl, ol, v, f, m, n, o);
    p.setGhosts(1);
    p.setMpiComm(mpi_comm);
    p.init();

    // Check on level 0
    auto& lv = p.getLevelAt(0);
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
    mgcl::VaryingStencil3x3x3 cg(m, n, o, gh, gh, gh);
    cg.fill1dIndex(true);

    // Update ghosts of expected result locally, i.e. not using MPI routines.
    cg.updateGhosts();

    // Create local slice of global data
    auto clptr = cg.slice(m_start, m_end, n_start, n_end, o_start, o_end);
    auto& cl = *clptr;

    // Update ghosts of test data
    mgcl::updateGhostsStencilMpi(cl, mpiData, true, false);

    // cl.dumpToFile("cl" + std::to_string(mpi_rank) + ".txt");

    // if (mpi_rank == 0)
    // {
    //     cg.dumpToFile("cg.txt");

    // Check result
    // check in z-direction
    for (int i = 0; i < gh; i++)
        for (int j = 0; j < nl + 2 * gh; j++)
            for (int k = 0; k < ol + 2 * gh; k++)
                for (int ii = 0; ii < 3; ii++)
                    for (int jj = 0; jj < 3; jj++)
                        for (int kk = 0; kk < 3; kk++)
                        {
                            if (mpi_coords[0] > 0) // not the first process
                                REQUIRE(cl[i][j][k][ii][jj][kk] == cg[m_start + i][j + n_start][k + o_start][ii][jj][kk]);
                            else
                                REQUIRE(cl[i][j][k][ii][jj][kk] == cg[i + m][j + n_start][k + o_start][ii][jj][kk]);

                            if (mpi_coords[0] < mpi_dims[0]) // not the last process
                                REQUIRE(cl[i + gh + ml][j][k][ii][jj][kk] == cg[m_end + gh + 1 + i][j + n_start][k + o_start][ii][jj][kk]);
                            else
                                REQUIRE(cl[i + gh + ml][j][k][ii][jj][kk] == cg[i + gh][j + n_start][k + o_start][ii][jj][kk]);
                        }

    // check in y-direction
    for (int i = 0; i < ml + 2 * gh; i++)
        for (int j = 0; j < gh; j++)
            for (int k = 0; k < ol + 2 * gh; k++)
                for (int ii = 0; ii < 3; ii++)
                    for (int jj = 0; jj < 3; jj++)
                        for (int kk = 0; kk < 3; kk++)
                        {
                            if (mpi_coords[1] > 0) // not the first process
                                REQUIRE(cl[i][j][k][ii][jj][kk] == cg[i + m_start][n_start + j][k + o_start][ii][jj][kk]);
                            else
                                REQUIRE(cl[i][j][k][ii][jj][kk] == cg[i + m_start][j + n][k + o_start][ii][jj][kk]);

                            if (mpi_coords[1] < mpi_dims[1]) // not the last process
                                REQUIRE(cl[i][j + gh + nl][k][ii][jj][kk] == cg[i + m_start][n_end + gh + 1 + j][k + o_start][ii][jj][kk]);
                            else
                                REQUIRE(cl[i][j + gh + nl][k][ii][jj][kk] == cg[i + m_start][j + gh][k + o_start][ii][jj][kk]);
                        }

    // check in x-direction
    for (int i = 0; i < ml + 2 * gh; i++)
        for (int j = 0; j < nl + 2 * gh; j++)
            for (int k = 0; k < gh; k++)
                for (int ii = 0; ii < 3; ii++)
                    for (int jj = 0; jj < 3; jj++)
                        for (int kk = 0; kk < 3; kk++)
                        {
                            if (mpi_coords[2] > 0) // not the first process
                                REQUIRE(cl[i][j][k][ii][jj][kk] == cg[i + m_start][j + n_start][o_start + k][ii][jj][kk]);
                            else
                                REQUIRE(cl[i][j][k][ii][jj][kk] == cg[i + m_start][j + n_start][k + o][ii][jj][kk]);

                            if (mpi_coords[2] < mpi_dims[2]) // not the last process
                                REQUIRE(cl[i][j][k + gh + ol][ii][jj][kk] == cg[i + m_start][j + n_start][o_end + gh + 1 + k][ii][jj][kk]);
                            else
                                REQUIRE(cl[i][j][k + gh + ol][ii][jj][kk] == cg[i + m_start][j + n_start][k + gh][ii][jj][kk]);
                        }
}

// Checks ghost update for any number of processes that is allowed by mgcl, e.g. 1, 2, 4, 8, 24.
// Run with: mpiexec -n 8 tests_mpi "MPI-updateGhostsStencilOclMpi-nprocs"
// TODO differentiate for gh>m and non-periodic
TEST_CASE("MPI-updateGhostsStencilOclMpi-nprocs")
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
    int n_start = (n / mpi_dims[1]) * mpi_coords[1] + min(mpi_coords[1], (n % mpi_dims[1]));
    int n_end = (n / mpi_dims[1]) * (mpi_coords[1] + 1) + min(mpi_coords[1] + 1, (n % mpi_dims[1])) - 1;
    int o_start = (o / mpi_dims[2]) * mpi_coords[2] + min(mpi_coords[2], (o % mpi_dims[2]));
    int o_end = (o / mpi_dims[2]) * (mpi_coords[2] + 1) + min(mpi_coords[2] + 1, (o % mpi_dims[2])) - 1;

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
    auto& p = *pptr;
    p.setGhosts(1);
    p.setMpiComm(mpi_comm);
    p.init();

    // Check on level 0
    auto& lv = p.getLevelAt(0);
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

    mgcl_test::TestUtility tu(pptr);

    // Create global test data. No random data so values will be the same for all processes. Fill with 1d index.
    mgcl::VaryingStencil3x3x3 sglob(m, n, o, gh, gh, gh);
    sglob.fill1dIndex(true);
    mgcl::VaryingStencilGpu sgpu(m, n, o, 3, gh, tu.getContext(), tu.getCommands());
    sgpu.fill<3>(sglob, tu.getCommands());
    tu.finish();

    // Update ghosts of expected result locally, i.e. not using MPI routines.
    sgpu.updateGhosts(tu.getProgram(), tu.getCommands());

    // Create local slice of global data
    auto clptr = sglob.slice(m_start, m_end, n_start, n_end, o_start, o_end);
    auto& sloc = *clptr;
    mgcl::VaryingStencilGpu sgpuLocal(sloc.getDim1(), sloc.getDim2(), sloc.getDim3(), 3, gh, tu.getContext(), tu.getCommands());
    sgpuLocal.fill<3>(sloc, tu.getCommands());
    tu.finish();

    // Update ghosts of test data
    mgcl::updateGhostsStencilOclMpi(tu.getProgram(), tu.getCommands(), sgpuLocal, *mpiData, true, false);

    // Read results
    auto cg = sgpu.read<3>(tu.getCommands());
    auto cl = sgpuLocal.read<3>(tu.getCommands());
    tu.finish();

    // cl.dumpToFile("cl" + std::to_string(mpi_rank) + ".txt");

    // if (mpi_rank == 0)
    // {
    //     cg.dumpToFile("cg.txt");

    // Check result
    // check in z-direction
    for (int i = 0; i < gh; i++)
        for (int j = 0; j < nl + 2 * gh; j++)
            for (int k = 0; k < ol + 2 * gh; k++)
                for (int ii = 0; ii < 3; ii++)
                    for (int jj = 0; jj < 3; jj++)
                        for (int kk = 0; kk < 3; kk++)
                        {
                            if (mpi_coords[0] > 0) // not the first process
                                REQUIRE(cl[i][j][k][ii][jj][kk] == cg[m_start + i][j + n_start][k + o_start][ii][jj][kk]);
                            else
                                REQUIRE(cl[i][j][k][ii][jj][kk] == cg[i + m][j + n_start][k + o_start][ii][jj][kk]);

                            if (mpi_coords[0] < mpi_dims[0]) // not the last process
                                REQUIRE(cl[i + gh + ml][j][k][ii][jj][kk] == cg[m_end + gh + 1 + i][j + n_start][k + o_start][ii][jj][kk]);
                            else
                                REQUIRE(cl[i + gh + ml][j][k][ii][jj][kk] == cg[i + gh][j + n_start][k + o_start][ii][jj][kk]);
                        }

    // check in y-direction
    for (int i = 0; i < ml + 2 * gh; i++)
        for (int j = 0; j < gh; j++)
            for (int k = 0; k < ol + 2 * gh; k++)
                for (int ii = 0; ii < 3; ii++)
                    for (int jj = 0; jj < 3; jj++)
                        for (int kk = 0; kk < 3; kk++)
                        {
                            if (mpi_coords[1] > 0) // not the first process
                                REQUIRE(cl[i][j][k][ii][jj][kk] == cg[i + m_start][n_start + j][k + o_start][ii][jj][kk]);
                            else
                                REQUIRE(cl[i][j][k][ii][jj][kk] == cg[i + m_start][j + n][k + o_start][ii][jj][kk]);

                            if (mpi_coords[1] < mpi_dims[1]) // not the last process
                                REQUIRE(cl[i][j + gh + nl][k][ii][jj][kk] == cg[i + m_start][n_end + gh + 1 + j][k + o_start][ii][jj][kk]);
                            else
                                REQUIRE(cl[i][j + gh + nl][k][ii][jj][kk] == cg[i + m_start][j + gh][k + o_start][ii][jj][kk]);
                        }

    // check in x-direction
    for (int i = 0; i < ml + 2 * gh; i++)
        for (int j = 0; j < nl + 2 * gh; j++)
            for (int k = 0; k < gh; k++)
                for (int ii = 0; ii < 3; ii++)
                    for (int jj = 0; jj < 3; jj++)
                        for (int kk = 0; kk < 3; kk++)
                        {
                            if (mpi_coords[2] > 0) // not the first process
                                REQUIRE(cl[i][j][k][ii][jj][kk] == cg[i + m_start][j + n_start][o_start + k][ii][jj][kk]);
                            else
                                REQUIRE(cl[i][j][k][ii][jj][kk] == cg[i + m_start][j + n_start][k + o][ii][jj][kk]);

                            if (mpi_coords[2] < mpi_dims[2]) // not the last process
                                REQUIRE(cl[i][j][k + gh + ol][ii][jj][kk] == cg[i + m_start][j + n_start][o_end + gh + 1 + k][ii][jj][kk]);
                            else
                                REQUIRE(cl[i][j][k + gh + ol][ii][jj][kk] == cg[i + m_start][j + n_start][k + gh][ii][jj][kk]);
                        }
}

// Checks if galerkin works for multiple processes but threshold 0, i.e. everything is done locally.
// Run with: mpiexec -n 8 tests_mpi "MPI-galerkin-threshold0"
TEST_CASE("MPI-galerkin-threshold0")
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
    int n_start = (n / mpi_dims[1]) * mpi_coords[1] + min(mpi_coords[1], (n % mpi_dims[1]));
    int n_end = (n / mpi_dims[1]) * (mpi_coords[1] + 1) + min(mpi_coords[1] + 1, (n % mpi_dims[1])) - 1;
    int o_start = (o / mpi_dims[2]) * mpi_coords[2] + min(mpi_coords[2], (o % mpi_dims[2]));
    int o_end = (o / mpi_dims[2]) * (mpi_coords[2] + 1) + min(mpi_coords[2] + 1, (o % mpi_dims[2])) - 1;

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
    auto& p = *pptr;
    p.setGhosts(gh);
    p.setMpiComm(mpi_comm);
    p.setMpiMinGridPoints(m); // ensure threshold level is 0

    p.setStencilType(mgcl::MGCL_VARYING);
    auto& sv = p.getStencilValues();
    sv->fillRandomInt();

    p.init();

    for (int i = 0; i <= p.getMaxlevel(); i++)
    {
        auto& lv = p.getLevelAt(i);
        auto& sv = *lv.getStencilValues();

        REQUIRE(sv.getDim1() == m >> i);
        REQUIRE(sv.getDim2() == n >> i);
        REQUIRE(sv.getDim3() == o >> i);
    }
}