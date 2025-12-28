#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <iostream>
#include <memory>

#include "../../src/mgcl/blockstencil.hpp"
#include "../../src/mgcl/blockstencil_gpu.hpp"
#include "../../src/mgcl/cuboid.hpp"
#include "../../src/mgcl/cuboid_bs.hpp"
#include "../../src/mgcl/cuboid_bs_gpu.hpp"
#include "../../src/mgcl/cuboid_gpu.hpp"
#include "../../src/mgcl/mpi_util.hpp"
#include "../../src/mgcl/multigrid_engine.hpp"
#include "../../src/mgcl/problem.hpp"
#include "../cli_args.hpp"
#include "../device_type_generator.hpp"
#include "../test_utility.hpp"

#include "mpi.h"

// Checks ghost update for 1 process which is actually done locally without MPI calls.
// Run with: mpiexec -n 1 tests_mpi [mpi1]
// TODO differentiate for gh>m and non-periodic
TEST_CASE("MPIupdateGhostsSeq1process", "[mpi1]")
{
    int N = 4;
    int periodic = GENERATE(1, 0);
    mgcl::BC bc = periodic ? mgcl::BC::PERIODIC : mgcl::BC::DIRICHLET;

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

    int gh = 1;

    // Init some random data
    auto v = std::make_shared<mgcl::Cuboid>(N, N, N, gh, gh, gh);
    auto f = std::make_shared<mgcl::Cuboid>(N, N, N, gh, gh, gh);
    v->fillRandom();
    f->fillRandom();

    SECTION("gh = 1 < m,n,o")
    {
        int m = N;
        int n = N;
        int o = N;

        // Init Problem to create all needed structures
        mgcl::Problem p(N, N, N, v, f);
        p.setGhosts(1);
        p.setMpiComm(mpi_comm);
        p.setBc(bc);
        p.setGhostsIn(gh);
        p.init();

        // Check on level 0
        auto& lv = p.getLevelAt(0);

        // Create test data
        mgcl::Cuboid c(N, N, N, gh, gh, gh);
        c.fillRandom(-10, 10, true);

        // Update ghosts of test data
        mgcl::MultigridEngine::updateGhostsSeq(c, nullptr, periodic, false);

        if (periodic)
        { // Check result
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
        else
        {
            // check in z-direction
            for (int i = 0; i < gh; i++)
                for (int j = 0; j < n + 2 * gh; j++)
                    for (int k = 0; k < o + 2 * gh; k++)
                    {
                        REQUIRE(c[i][j][k] == 0);
                        REQUIRE(c[i + gh + m][j][k] == 0);
                    }

            // check in y-direction
            for (int i = 0; i < m + 2 * gh; i++)
                for (int j = 0; j < gh; j++)
                    for (int k = 0; k < o + 2 * gh; k++)
                    {
                        REQUIRE(c[i][j][k] == 0);
                        REQUIRE(c[i][j + gh + n][k] == 0);
                    }

            // check in x-direction
            for (int i = 0; i < m + 2 * gh; i++)
                for (int j = 0; j < n + 2 * gh; j++)
                    for (int k = 0; k < gh; k++)
                    {
                        REQUIRE(c[i][j][k] == 0);
                        REQUIRE(c[i][j][k + gh + o] == 0);
                    }
        }
    }
}

// Checks ghost update for any number of processes that is allowed by mgcl, e.g. 1, 2, 4, 8, 24.
// Run with: mpiexec -n 8 tests_mpi [mpiN]
// TODO differentiate for gh>m and non-periodic
TEST_CASE("MPIupdateGhostsSeqNProcesses")
{
    using std::min;

    // global grid sizes
    int m = 16;
    int n = 16;
    int o = 16;
    int periodic = GENERATE(1, 0);
    mgcl::BC bc = periodic ? mgcl::BC::PERIODIC : mgcl::BC::DIRICHLET;

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

    // // print coords and boundaries per rank
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

    REQUIRE(ml > 1);
    REQUIRE(ml <= m);
    REQUIRE(nl > 1);
    REQUIRE(nl <= n);
    REQUIRE(ol > 1);
    REQUIRE(ol <= o);

    int gh = 1;

    // Init some random data (will be unused)
    auto v = std::make_shared<mgcl::Cuboid>(ml, nl, ol, gh, gh, gh);
    auto f = std::make_shared<mgcl::Cuboid>(ml, nl, ol, gh, gh, gh);
    v->fillRandom();
    f->fillRandom();

    SECTION("gh = 1 < m,n,o")
    {

        // Init Problem to create all needed structures
        mgcl::Problem p(ml, nl, ol, f, v, m, n, o);
        p.setGhosts(1);
        p.setMpiComm(mpi_comm);
        p.setBc(bc);
        p.setGhostsIn(gh);
        p.init();

        // Check on level 0
        auto& lv = p.getLevelAt(0);
        auto mpiData = lv.getMpiDataPtr();

        REQUIRE((mpi_size == 1 || !lv.isCalculatedLocally()));

        // // print neighbours per rank
        // for (int i = 0; i < mpi_size; i++)
        // {
        //     MPI_Barrier(mpi_comm);
        //     if (i == mpi_rank)
        //         std::cout << mpi_rank << ": " << mpiData->left[0] << "," << mpiData->right[0] << ","
        //                   << mpiData->up[0] << "," << mpiData->down[0] << ","
        //                   << mpiData->back[0] << "," << mpiData->front[0] << std::endl;
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
        mgcl::MultigridEngine::updateGhostsSeq(cg, nullptr, periodic, true);

        // Create local slice of global data
        auto clptr = cg.slice(m_start, m_end, n_start, n_end, o_start, o_end);
        auto& cl = *clptr;

        // Update ghosts of test data
        mgcl::MultigridEngine::updateGhostsSeq(cl, mpiData, periodic, false);

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
                    CAPTURE(i, j, k, mpi_rank);
                    if (mpi_coords[0] > 0) // not the first process
                        REQUIRE(cl[i][j][k] == cg[m_start + i][j + n_start][k + o_start]);
                    else if (periodic)
                        REQUIRE(cl[i][j][k] == cg[i + m][j + n_start][k + o_start]);

                    if (mpi_coords[0] < mpi_dims[0]) // not the last process
                        REQUIRE(cl[i + gh + ml][j][k] == cg[m_end + gh + 1 + i][j + n_start][k + o_start]);
                    else if (periodic)
                        REQUIRE(cl[i + gh + ml][j][k] == cg[i + gh][j + n_start][k + o_start]);
                }

        // check in y-direction
        for (int i = 0; i < ml + 2 * gh; i++)
            for (int j = 0; j < gh; j++)
                for (int k = 0; k < ol + 2 * gh; k++)
                {
                    if (mpi_coords[1] > 0) // not the first process
                        REQUIRE(cl[i][j][k] == cg[i + m_start][n_start + j][k + o_start]);
                    else if (periodic)
                        REQUIRE(cl[i][j][k] == cg[i + m_start][j + n][k + o_start]);

                    if (mpi_coords[1] < mpi_dims[1]) // not the last process
                        REQUIRE(cl[i][j + gh + nl][k] == cg[i + m_start][n_end + gh + 1 + j][k + o_start]);
                    else if (periodic)
                        REQUIRE(cl[i][j + gh + nl][k] == cg[i + m_start][j + gh][k + o_start]);
                }

        // check in x-direction
        for (int i = 0; i < ml + 2 * gh; i++)
            for (int j = 0; j < nl + 2 * gh; j++)
                for (int k = 0; k < gh; k++)
                {
                    if (mpi_coords[2] > 0) // not the first process
                        REQUIRE(cl[i][j][k] == cg[i + m_start][j + n_start][o_start + k]);
                    else if (periodic)
                        REQUIRE(cl[i][j][k] == cg[i + m_start][j + n_start][k + o]);

                    if (mpi_coords[2] < mpi_dims[2]) // not the last process
                        REQUIRE(cl[i][j][k + gh + ol] == cg[i + m_start][j + n_start][o_end + gh + 1 + k]);
                    else if (periodic)
                        REQUIRE(cl[i][j][k + gh + ol] == cg[i + m_start][j + n_start][k + gh]);
                }
        // }
    }
}

// Checks ghost update for any number of processes that is allowed by mgcl, e.g. 1, 2, 4, 8, 24.
// Run with: mpiexec -n 8 tests_mpi "MPI updateGhosts ocl (n processes)"
// TODO differentiate for gh>m and non-periodic
TEST_CASE("MPIupdateGhostsOclNprocesses")
{
    auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));

    using std::min;

    // global grid sizes
    int m = 8;
    int n = 8;
    int o = 8;
    int periodic = GENERATE(1, 0);
    mgcl::BC bc = periodic ? mgcl::BC::PERIODIC : mgcl::BC::DIRICHLET;

    // yz front: 0..323
    // yz back: 324..647
    // xz left: 648..827
    // xz right: 828..1007
    // xy top: 1008..1277
    // xy bottom: 1278..1547

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

    // // print coords and boundaries per rank if (mpi_rank == 0)
    // std::cout << "rank;coords[0];coords[1];coords[2];ms;me;ns;ne;os;oe" << std::endl;
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

    REQUIRE(ml > 1);
    REQUIRE(ml <= m);
    REQUIRE(nl > 1);
    REQUIRE(nl <= n);
    REQUIRE(ol > 1);
    REQUIRE(ol <= o);

    int gh = 1;

    // Init some random data (will be unused)
    auto v = std::make_shared<mgcl::Cuboid>(ml, nl, ol, gh, gh, gh);
    auto f = std::make_shared<mgcl::Cuboid>(ml, nl, ol, gh, gh, gh);
    v->fillRandom();
    f->fillRandom();

    // Init Problem to create all needed structures
    auto pptr = std::make_shared<mgcl::Problem>(ml, nl, ol, f, v, m, n, o);
    auto& p = *pptr;
    p.setGhosts(1);
    p.setMpiComm(mpi_comm);
    p.setUseOpencl(true);
    p.setDeviceType(deviceType);
    p.setBc(bc);
    p.setGhostsIn(gh);
    p.setMpiMinGridPoints(2);
    p.init();

    // Check on level 0
    auto& lv = p.getLevelAt(0);
    auto mpiData = lv.getMpiDataPtr();

    REQUIRE((mpi_size == 1 || !lv.isCalculatedLocally()));

    // // print neighbours per rank
    // for (int i = 0; i < mpi_size; i++)
    // {
    //     MPI_Barrier(mpi_comm);
    //     if (i == mpi_rank)
    //         std::cout << mpi_rank << ": " << mpiData->left[0] << "," << mpiData->right[0] << ","
    //                   << mpiData->up[0] << "," << mpiData->down[0] << ","
    //                   << mpiData->back[0] << "," << mpiData->front[0] << std::endl;
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
    mgcl::MultigridEngine::updateGhostsSeq(cg, nullptr, periodic, true);

    // Create local slice of global data
    auto clptr = cg.slice(m_start, m_end, n_start, n_end, o_start, o_end);
    auto& cl = *clptr;

    // Create ocl buffer
    auto d_cl = std::make_shared<mgcl::CuboidGpu>(p.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, cl);

    // Update ghosts of test data
    mgcl::MultigridEngine::updateGhosts(p, *d_cl, mpiData, false);
    p.getOpenCLHelper().finish();

    auto cl_res_ptr = d_cl->read(p.getCommands(), nullptr, true);
    auto& cl_res = *cl_res_ptr;

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
                CAPTURE(i, j, k, mpi_rank);
                if (mpi_coords[0] > 0) // not the first process
                    REQUIRE(cl_res[i][j][k] == cg[m_start + i][j + n_start][k + o_start]);
                else if (periodic)
                    REQUIRE(cl_res[i][j][k] == cg[i + m][j + n_start][k + o_start]);

                if (mpi_coords[0] < mpi_dims[0]) // not the last process
                    REQUIRE(cl_res[i + gh + ml][j][k] == cg[m_end + gh + 1 + i][j + n_start][k + o_start]);
                else if (periodic)
                    REQUIRE(cl_res[i + gh + ml][j][k] == cg[i + gh][j + n_start][k + o_start]);
            }

    // check in y-direction
    for (int i = 0; i < ml + 2 * gh; i++)
        for (int j = 0; j < gh; j++)
            for (int k = 0; k < ol + 2 * gh; k++)
            {
                if (mpi_coords[1] > 0) // not the first process
                    REQUIRE(cl_res[i][j][k] == cg[i + m_start][n_start + j][k + o_start]);
                else if (periodic)
                    REQUIRE(cl_res[i][j][k] == cg[i + m_start][j + n][k + o_start]);

                if (mpi_coords[1] < mpi_dims[1]) // not the last process
                    REQUIRE(cl_res[i][j + gh + nl][k] == cg[i + m_start][n_end + gh + 1 + j][k + o_start]);
                else if (periodic)
                    REQUIRE(cl_res[i][j + gh + nl][k] == cg[i + m_start][j + gh][k + o_start]);
            }

    // check in x-direction
    for (int i = 0; i < ml + 2 * gh; i++)
        for (int j = 0; j < nl + 2 * gh; j++)
            for (int k = 0; k < gh; k++)
            {
                if (mpi_coords[2] > 0) // not the first process
                    REQUIRE(cl_res[i][j][k] == cg[i + m_start][j + n_start][o_start + k]);
                else if (periodic)
                    REQUIRE(cl_res[i][j][k] == cg[i + m_start][j + n_start][k + o]);

                if (mpi_coords[2] < mpi_dims[2]) // not the last process
                    REQUIRE(cl_res[i][j][k + gh + ol] == cg[i + m_start][j + n_start][o_end + gh + 1 + k]);
                else if (periodic)
                    REQUIRE(cl_res[i][j][k + gh + ol] == cg[i + m_start][j + n_start][k + gh]);
            }
}

// Checks ghost update for any number of processes that is allowed by mgcl, e.g. 1, 2, 4, 8, 24.
// Run with: mpiexec -n 8 tests_mpi [mpiN]
// TODO differentiate for gh>m and non-periodic
TEST_CASE("MPIupdateGhostsSeqCuboidBSNprocesses")
{
    using std::min;

    // global grid sizes
    int m = 16;
    int n = 16;
    int o = 16;
    int blocksize = 2;
    int periodic = GENERATE(1, 0);
    mgcl::BC bc = periodic ? mgcl::BC::PERIODIC : mgcl::BC::DIRICHLET;

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

    REQUIRE(ml > 1);
    REQUIRE(ml <= m);
    REQUIRE(nl > 1);
    REQUIRE(nl <= n);
    REQUIRE(ol > 1);
    REQUIRE(ol <= o);

    int gh = 1;

    // Init some random data (will be unused)
    auto v = std::make_shared<mgcl::Cuboid>(ml, nl, ol, gh, gh, gh);
    auto f = std::make_shared<mgcl::Cuboid>(ml, nl, ol, gh, gh, gh);
    v->fillRandom();
    f->fillRandom();

    SECTION("gh = 1 < m,n,o")
    {

        // Init Problem to create all needed structures
        mgcl::Problem p(ml, nl, ol, f, v, m, n, o);
        p.setGhosts(1);
        p.setMpiComm(mpi_comm);
        p.setBc(bc);
        p.setGhostsIn(gh);
        p.init();

        // Check on level 0
        auto& lv = p.getLevelAt(0);
        auto mpiData = lv.getMpiDataPtr();

        REQUIRE((mpi_size == 1 || !lv.isCalculatedLocally()));

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
        mgcl::CuboidBS cg(m, n, o, gh, gh, gh, blocksize);
        cg.fill1dIndex(true);

        // Update ghosts of expected result locally, i.e. not using MPI routines.
        cg.updateGhosts(nullptr, true, periodic);

        // Create local slice of global data
        auto clptr = cg.slice(m_start, m_end, n_start, n_end, o_start, o_end);
        auto& cl = *clptr;

        // Update ghosts of test data
        cl.updateGhosts(mpiData, false, periodic);

        // cl.dumpToFile("cl" + std::to_string(mpi_rank) + ".txt");

        // if (mpi_rank == 0)
        // {
        //     cg.dumpToFile("cg.txt");

        // Check result
        // check in z-direction
        for (int i = 0; i < gh; i++)
            for (int j = 0; j < nl + 2 * gh; j++)
                for (int k = 0; k < ol + 2 * gh; k++)
                    for (int b = 0; b < blocksize; b++)
                    {
                        if (mpi_coords[0] > 0) // not the first process
                            REQUIRE(cl[b][i][j][k] == cg[b][m_start + i][j + n_start][k + o_start]);
                        else if (periodic)
                            REQUIRE(cl[b][i][j][k] == cg[b][i + m][j + n_start][k + o_start]);

                        if (mpi_coords[0] < mpi_dims[0]) // not the last process
                            REQUIRE(cl[b][i + gh + ml][j][k] == cg[b][m_end + gh + 1 + i][j + n_start][k + o_start]);
                        else if (periodic)
                            REQUIRE(cl[b][i + gh + ml][j][k] == cg[b][i + gh][j + n_start][k + o_start]);
                    }

        // check in y-direction
        for (int i = 0; i < ml + 2 * gh; i++)
            for (int j = 0; j < gh; j++)
                for (int k = 0; k < ol + 2 * gh; k++)
                    for (int b = 0; b < blocksize; b++)
                    {
                        if (mpi_coords[1] > 0) // not the first process
                            REQUIRE(cl[b][i][j][k] == cg[b][i + m_start][n_start + j][k + o_start]);
                        else if (periodic)
                            REQUIRE(cl[b][i][j][k] == cg[b][i + m_start][j + n][k + o_start]);

                        if (mpi_coords[1] < mpi_dims[1]) // not the last process
                            REQUIRE(cl[b][i][j + gh + nl][k] == cg[b][i + m_start][n_end + gh + 1 + j][k + o_start]);
                        else if (periodic)
                            REQUIRE(cl[b][i][j + gh + nl][k] == cg[b][i + m_start][j + gh][k + o_start]);
                    }

        // check in x-direction
        for (int i = 0; i < ml + 2 * gh; i++)
            for (int j = 0; j < nl + 2 * gh; j++)
                for (int k = 0; k < gh; k++)
                    for (int b = 0; b < blocksize; b++)
                    {
                        if (mpi_coords[2] > 0) // not the first process
                            REQUIRE(cl[b][i][j][k] == cg[b][i + m_start][j + n_start][o_start + k]);
                        else if (periodic)
                            REQUIRE(cl[b][i][j][k] == cg[b][i + m_start][j + n_start][k + o]);

                        if (mpi_coords[2] < mpi_dims[2]) // not the last process
                            REQUIRE(cl[b][i][j][k + gh + ol] == cg[b][i + m_start][j + n_start][o_end + gh + 1 + k]);
                        else if (periodic)
                            REQUIRE(cl[b][i][j][k + gh + ol] == cg[b][i + m_start][j + n_start][k + gh]);
                    }
        // }
    }
}

// Checks ghost update for any number of processes that is allowed by mgcl, e.g. 1, 2, 4, 8, 24.
// Run with: mpiexec -n 8 tests_mpi "MPI_updateGhosts_ocl_CuboidBS_(n_processes)"
// TODO differentiate for gh>m and non-periodic
TEST_CASE("MPIupdateGhostsOclCuboidBSNprocesses")
{
    auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));

    using std::min;

    // global grid sizes
    // TODO test for e.g. m=4
    int m = 8;
    int n = 8;
    int o = 8;
    int blocksize = 2;
    int periodic = GENERATE(1, 0);
    mgcl::BC bc = periodic ? mgcl::BC::PERIODIC : mgcl::BC::DIRICHLET;

    // yz front: 0..647
    // yz back: 648..1295
    // xz left: 1296..1655
    // xz right: 1656..2015
    // xy top: 2016..2375
    // xy bottom: 2376..2735

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

    REQUIRE(ml > 1);
    REQUIRE(ml <= m);
    REQUIRE(nl > 1);
    REQUIRE(nl <= n);
    REQUIRE(ol > 1);
    REQUIRE(ol <= o);

    int gh = 1;

    // Init some random data (will be unused)
    auto v = std::make_shared<mgcl::Cuboid>(ml, nl, ol, gh, gh, gh);
    auto f = std::make_shared<mgcl::Cuboid>(ml, nl, ol, gh, gh, gh);
    v->fillRandom();
    f->fillRandom();

    // Init Problem to create all needed structures
    auto pptr = std::make_shared<mgcl::Problem>(ml, nl, ol, f, v, m, n, o);
    auto& p = *pptr;
    p.setGhosts(1);
    p.setMpiComm(mpi_comm);
    p.setUseOpencl(true);
    p.setDeviceType(deviceType);
    p.getOpenCLHelper().setPreprocessorConstant("BLOCKSIZE", std::to_string(blocksize));
    p.setGhostsIn(gh);
    p.setBc(bc);
    p.setMpiMinGridPoints(2);
    p.init();

    // Check on level 0
    auto& lv = p.getLevelAt(0);
    auto mpiData = lv.getMpiDataPtr();

    REQUIRE((mpi_size == 1 || !lv.isCalculatedLocally()));

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
    mgcl::CuboidBS cg(m, n, o, gh, gh, gh, blocksize);
    cg.fill1dIndex(true);

    // // filling with halve the 1d index makes it possible to check against scalar Cuboid (manually)
    // double cnt = 0;
    // for (int i = gh; i < m + gh; i++)
    //     for (int j = gh; j < n + gh; j++)
    //         for (int k = gh; k < o + gh; k++)
    //             for (int b = 0; b < blocksize; b++)
    //             {
    //                 cg[b][i][j][k] = cnt / 2.0;
    //                 cnt++;
    //             }

    // Update ghosts of expected result locally, i.e. not using MPI routines.
    cg.updateGhosts(nullptr, true, periodic);

    // Create local slice of global data
    auto clptr = cg.slice(m_start, m_end, n_start, n_end, o_start, o_end);
    auto& cl = *clptr;
    // std::cout << "cl m,n,o: " << cl.getM() << "," << cl.getN() << "," << cl.getO() << std::endl;

    // Create CuboidBSGpu buffer, planes buf and send and recv bufs
    // TODO update in Problem init when using blockstencils
    auto d_cl = std::make_shared<mgcl::CuboidBSGpu>(p.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, cl);
    int yz = d_cl->getNgh() * d_cl->getOgh(); // 324
    int xz = d_cl->getMgh() * d_cl->getOgh(); // 180
    int xy = d_cl->getMgh() * d_cl->getNgh(); // 180
    int ressize = (2 * yz * d_cl->getGhostsM() + 2 * xz * d_cl->getGhostsN() + 2 * xy * d_cl->getGhostsO()) * blocksize;
    mgcl::BufferGpu dPlanesBuf(p.getContext(), CL_MEM_READ_WRITE, ressize);
    auto hPlanesBufSend = std::make_shared<std::vector<double>>(ressize);
    auto hPlanesBufRecv = std::make_shared<std::vector<double>>(ressize);

    // Update ghosts of test data using MPI
    d_cl->updateGhostsOclMpi(p.getProgram(), p.getCommands(),
                             &dPlanesBuf, hPlanesBufSend.get(), hPlanesBufRecv.get(),
                             mpiData, false, periodic,
                             &p.getKernelConfig(), p.getProfilingData());
    p.finish();

    auto cl_res_ptr = d_cl->read(p.getCommands(), nullptr, true);
    auto& cl_res = *cl_res_ptr;

    // cl.dumpToFile("cl" + std::to_string(mpi_rank) + ".txt");
    // cl_res.dumpToFile("cl_res" + std::to_string(mpi_rank) + ".txt");

    // if (mpi_rank == 0)
    // {
    //     cg.dumpToFile("cg.txt");

    // Check result
    // check in z-direction
    for (int i = 0; i < gh; i++)
        for (int j = 0; j < nl + 2 * gh; j++)
            for (int k = 0; k < ol + 2 * gh; k++)
                for (int b = 0; b < blocksize; b++)
                {
                    CAPTURE(i, j, k, b, mpi_rank);
                    if (mpi_coords[0] > 0) // not the first process
                        REQUIRE(cl_res[b][i][j][k] == cg[b][m_start + i][j + n_start][k + o_start]);
                    else if (periodic)
                        REQUIRE(cl_res[b][i][j][k] == cg[b][i + m][j + n_start][k + o_start]);

                    if (mpi_coords[0] < mpi_dims[0]) // not the last process
                        REQUIRE(cl_res[b][i + gh + ml][j][k] == cg[b][m_end + gh + 1 + i][j + n_start][k + o_start]);
                    else if (periodic)
                        REQUIRE(cl_res[b][i + gh + ml][j][k] == cg[b][i + gh][j + n_start][k + o_start]);
                }

    // check in y-direction
    for (int i = 0; i < ml + 2 * gh; i++)
        for (int j = 0; j < gh; j++)
            for (int k = 0; k < ol + 2 * gh; k++)
                for (int b = 0; b < blocksize; b++)
                {
                    if (mpi_coords[1] > 0) // not the first process
                        REQUIRE(cl_res[b][i][j][k] == cg[b][i + m_start][n_start + j][k + o_start]);
                    else if (periodic)
                        REQUIRE(cl_res[b][i][j][k] == cg[b][i + m_start][j + n][k + o_start]);

                    if (mpi_coords[1] < mpi_dims[1]) // not the last process
                        REQUIRE(cl_res[b][i][j + gh + nl][k] == cg[b][i + m_start][n_end + gh + 1 + j][k + o_start]);
                    else if (periodic)
                        REQUIRE(cl_res[b][i][j + gh + nl][k] == cg[b][i + m_start][j + gh][k + o_start]);
                }

    // check in x-direction
    for (int i = 0; i < ml + 2 * gh; i++)
        for (int j = 0; j < nl + 2 * gh; j++)
            for (int k = 0; k < gh; k++)
                for (int b = 0; b < blocksize; b++)
                {
                    if (mpi_coords[2] > 0) // not the first process
                        REQUIRE(cl_res[b][i][j][k] == cg[b][i + m_start][j + n_start][o_start + k]);
                    else if (periodic)
                        REQUIRE(cl_res[b][i][j][k] == cg[b][i + m_start][j + n_start][k + o]);

                    if (mpi_coords[2] < mpi_dims[2]) // not the last process
                        REQUIRE(cl_res[b][i][j][k + gh + ol] == cg[b][i + m_start][j + n_start][o_end + gh + 1 + k]);
                    else if (periodic)
                        REQUIRE(cl_res[b][i][j][k + gh + ol] == cg[b][i + m_start][j + n_start][k + gh]);
                }
}

// Checks ghost update for any number of processes that is allowed by mgcl, e.g. 1, 2, 4, 8, 24.
// Run with: mpiexec -n 8 tests_mpi "MPIupdateGhostsSeqBlockstencilNprocesses"
// TODO differentiate for gh>m and non-periodic
TEST_CASE("MPIupdateGhostsSeqBlockstencilNprocesses")
{
    using std::min;

    // global grid sizes. Don't go lower than mpiLevelThreshold allows!
    int m = 8;
    int n = 8;
    int o = 8;
    int blocksize = 2;
    int width = 3;
    int periodic = GENERATE(1, 0);
    mgcl::BC bc = periodic ? mgcl::BC::PERIODIC : mgcl::BC::DIRICHLET;

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

    int err;

    /* Initialize cartesian process grid */
    mgcl::mpi_util::mgclCheckMpiError(mpi_comm, MPI_Comm_size(mpi_comm, &mpi_size), "MPI_Comm_size");
    mgcl::mpi_util::mgclCheckMpiError(mpi_comm, MPI_Dims_create(mpi_size, 3, mpi_dims), "MPI_Dims_create");
    mgcl::mpi_util::mgclCheckMpiError(mpi_comm, MPI_Cart_create(mpi_comm, 3, mpi_dims, mpi_periods, 1, &mpi_comm), "MPI_Cart_create");
    mgcl::mpi_util::mgclCheckMpiError(mpi_comm, MPI_Comm_rank(mpi_comm, &mpi_rank), "MPI_Comm_rank");
    mgcl::mpi_util::mgclCheckMpiError(mpi_comm, MPI_Cart_coords(mpi_comm, mpi_rank, 3, mpi_coords), "MPI_Cart_coords");

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

    // // print coords and boundaries per rank
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

    REQUIRE(ml > 1);
    REQUIRE(ml <= m);
    REQUIRE(nl > 1);
    REQUIRE(nl <= n);
    REQUIRE(ol > 1);
    REQUIRE(ol <= o);

    int gh = 1;

    // Init some random data (will be unused)
    auto v = std::make_shared<mgcl::Cuboid>(ml, nl, ol, gh, gh, gh);
    auto f = std::make_shared<mgcl::Cuboid>(ml, nl, ol, gh, gh, gh);
    v->fillRandom();
    f->fillRandom();

    SECTION("gh = 1 < m,n,o")
    {

        // Init Problem to create all needed structures
        mgcl::Problem p(ml, nl, ol, f, v, m, n, o);
        p.setGhosts(1);
        p.setMpiComm(mpi_comm);
        p.setMpiMinGridPoints(2);
        p.setGhostsIn(gh);
        p.setBc(bc);
        p.init();

        // Check on level 0
        auto& lv = p.getLevelAt(0);
        auto mpiData = lv.getMpiDataPtr();

        REQUIRE((mpi_size == 1 || !lv.isCalculatedLocally()));

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
        mgcl::Blockstencil cg(m, n, o, width, blocksize, gh, gh, gh);
        cg.fill1dIndex(true);

        // Update ghosts of expected result locally, i.e. not using MPI routines.
        cg.updateGhosts(nullptr, true, periodic);

        // Create local slice of global data
        auto clptr = cg.slice(m_start, m_end, n_start, n_end, o_start, o_end);
        auto& cl = *clptr;

        // Update ghosts of test data
        cl.updateGhosts(mpiData, false, periodic);

        // cl.dumpToFile("cl" + std::to_string(mpi_rank) + ".txt");

        // if (mpi_rank == 0)
        // {
        //     cg.dumpToFile("cg.txt");

        // Check result
        // check in z-direction
        for (int bi = 0; bi < blocksize; bi++)
            for (int bj = 0; bj < blocksize; bj++)
                for (int ii = 0; ii < width; ii++)
                    for (int jj = 0; jj < width; jj++)
                        for (int kk = 0; kk < width; kk++)
                            for (int i = 0; i < gh; i++)
                                for (int j = 0; j < nl + 2 * gh; j++)
                                    for (int k = 0; k < ol + 2 * gh; k++)
                                    {
                                        CAPTURE(i, j, k, ii, jj, kk, bi, bj, mpi_rank);
                                        if (mpi_coords[0] > 0) // not the first process
                                            REQUIRE(cl[bi][bj][ii][jj][kk][i][j][k] == cg[bi][bj][ii][jj][kk][m_start + i][j + n_start][k + o_start]);
                                        else if (periodic)
                                            REQUIRE(cl[bi][bj][ii][jj][kk][i][j][k] == cg[bi][bj][ii][jj][kk][i + m][j + n_start][k + o_start]);

                                        if (mpi_coords[0] < mpi_dims[0]) // not the last process
                                            REQUIRE(cl[bi][bj][ii][jj][kk][i + gh + ml][j][k] == cg[bi][bj][ii][jj][kk][m_end + gh + 1 + i][j + n_start][k + o_start]);
                                        else if (periodic)
                                            REQUIRE(cl[bi][bj][ii][jj][kk][i + gh + ml][j][k] == cg[bi][bj][ii][jj][kk][i + gh][j + n_start][k + o_start]);
                                    }

        // check in y-direction
        for (int bi = 0; bi < blocksize; bi++)
            for (int bj = 0; bj < blocksize; bj++)
                for (int ii = 0; ii < width; ii++)
                    for (int jj = 0; jj < width; jj++)
                        for (int kk = 0; kk < width; kk++)
                            for (int i = 0; i < ml + 2 * gh; i++)
                                for (int j = 0; j < gh; j++)
                                    for (int k = 0; k < ol + 2 * gh; k++)
                                    {
                                        CAPTURE(i, j, k, ii, jj, kk, bi, bj, mpi_rank);
                                        if (mpi_coords[1] > 0) // not the first process
                                            REQUIRE(cl[bi][bj][ii][jj][kk][i][j][k] == cg[bi][bj][ii][jj][kk][i + m_start][n_start + j][k + o_start]);
                                        else if (periodic)
                                            REQUIRE(cl[bi][bj][ii][jj][kk][i][j][k] == cg[bi][bj][ii][jj][kk][i + m_start][j + n][k + o_start]);

                                        if (mpi_coords[1] < mpi_dims[1]) // not the last process
                                            REQUIRE(cl[bi][bj][ii][jj][kk][i][j + gh + nl][k] == cg[bi][bj][ii][jj][kk][i + m_start][n_end + gh + 1 + j][k + o_start]);
                                        else if (periodic)
                                            REQUIRE(cl[bi][bj][ii][jj][kk][i][j + gh + nl][k] == cg[bi][bj][ii][jj][kk][i + m_start][j + gh][k + o_start]);
                                    }

        // check in x-direction
        for (int bi = 0; bi < blocksize; bi++)
            for (int bj = 0; bj < blocksize; bj++)
                for (int ii = 0; ii < width; ii++)
                    for (int jj = 0; jj < width; jj++)
                        for (int kk = 0; kk < width; kk++)
                            for (int i = 0; i < ml + 2 * gh; i++)
                                for (int j = 0; j < nl + 2 * gh; j++)
                                    for (int k = 0; k < gh; k++)
                                    {
                                        CAPTURE(i, j, k, ii, jj, kk, bi, bj, mpi_rank);
                                        if (mpi_coords[2] > 0) // not the first process
                                            REQUIRE(cl[bi][bj][ii][jj][kk][i][j][k] == cg[bi][bj][ii][jj][kk][i + m_start][j + n_start][o_start + k]);
                                        else if (periodic)
                                            REQUIRE(cl[bi][bj][ii][jj][kk][i][j][k] == cg[bi][bj][ii][jj][kk][i + m_start][j + n_start][k + o]);

                                        if (mpi_coords[2] < mpi_dims[2]) // not the last process
                                            REQUIRE(cl[bi][bj][ii][jj][kk][i][j][k + gh + ol] == cg[bi][bj][ii][jj][kk][i + m_start][j + n_start][o_end + gh + 1 + k]);
                                        else if (periodic)
                                            REQUIRE(cl[bi][bj][ii][jj][kk][i][j][k + gh + ol] == cg[bi][bj][ii][jj][kk][i + m_start][j + n_start][k + gh]);
                                    }
    }
}

// Checks ghost update for any number of processes that is allowed by mgcl, e.g. 1, 2, 4, 8, 24.
// Run with: mpiexec -n 8 tests_mpi "MPIupdateGhostsOclBlockstencilNprocesses"
// TODO differentiate for gh>m and non-periodic
TEST_CASE("MPIupdateGhostsOclBlockstencilNprocesses")
{
    auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));

    using std::min;

    // global grid sizes
    int m = 8;
    int n = 8;
    int o = 8;
    int blocksize = 2;
    int blocksize2 = blocksize * blocksize;
    int width = 3;
    int stencilSize = width * width * width;
    int periodic = GENERATE(1, 0);
    mgcl::BC bc = periodic ? mgcl::BC::PERIODIC : mgcl::BC::DIRICHLET;

    // yz front: 0..647
    // yz back: 648..1295
    // xz left: 1296..1655
    // xz right: 1656..2015
    // xy top: 2016..2375
    // xy bottom: 2376..2735

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

    REQUIRE(ml > 1);
    REQUIRE(ml <= m);
    REQUIRE(nl > 1);
    REQUIRE(nl <= n);
    REQUIRE(ol > 1);
    REQUIRE(ol <= o);

    int gh = 1;

    // Init some random data (will be unused)
    auto v = std::make_shared<mgcl::Cuboid>(ml, nl, ol, gh, gh, gh);
    auto f = std::make_shared<mgcl::Cuboid>(ml, nl, ol, gh, gh, gh);
    v->fillRandom();
    f->fillRandom();

    // Init Problem to create all needed structures
    auto pptr = std::make_shared<mgcl::Problem>(ml, nl, ol, f, v, m, n, o);
    auto& p = *pptr;
    p.setGhosts(1);
    p.setMpiComm(mpi_comm);
    p.setUseOpencl(true);
    p.setDeviceType(deviceType);
    p.getOpenCLHelper().setPreprocessorConstant("BLOCKSIZE", std::to_string(blocksize));
    p.setBc(bc);
    p.setGhostsIn(gh);
    p.setMpiMinGridPoints(2);
    p.init();

    // Check on level 0
    auto& lv = p.getLevelAt(0);
    auto mpiData = lv.getMpiDataPtr();

    REQUIRE((mpi_size == 1 || !lv.isCalculatedLocally()));

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
    mgcl::Blockstencil cg(m, n, o, width, blocksize, gh, gh, gh);
    cg.fill1dIndex(true);

    // // filling with halve the 1d index makes it possible to check against scalar Cuboid (manually)
    // double cnt = 0;
    // for (int i = gh; i < m + gh; i++)
    //     for (int j = gh; j < n + gh; j++)
    //         for (int k = gh; k < o + gh; k++)
    //             for (int b = 0; b < blocksize; b++)
    //             {
    //                 cg[i][j][k][b] = cnt / 2.0;
    //                 cnt++;
    //             }

    // Update ghosts of expected result locally, i.e. not using MPI routines.
    if (periodic)
        cg.updateGhostsLocally();

    // Create local slice of global data
    auto clptr = cg.slice(m_start, m_end, n_start, n_end, o_start, o_end);
    auto& cl = *clptr;
    // std::cout << "cl m,n,o: " << cl.getM() << "," << cl.getN() << "," << cl.getO() << std::endl;

    // if (mpi_size == 2)
    // {
    //     for (int bi = 0; bi < blocksize; bi++)
    //         for (int bj = 0; bj < blocksize; bj++)
    //             for (int ii = 0; ii < width; ii++)
    //                 for (int jj = 0; jj < width; jj++)
    //                     for (int kk = 0; kk < width; kk++)
    //                         for (int i = m_start + gh; i <= m_end - gh; i++)
    //                             for (int j = n_start + gh; j <= n_end - gh; j++)
    //                                 for (int k = o_start + gh; k <= o_end - gh; k++)
    //                                 {
    //                                     CAPTURE(i, j, k, ii, jj, kk, bi, bj, mpi_rank);
    //                                     REQUIRE(cg[bi][bj][ii][jj][kk][i][j][k] == cl[bi][bj][ii][jj][kk][i - mpi_coords[0] * (m_end - m_start + 1)][j - mpi_coords[1] * (n_end - n_start + 1)][k - mpi_coords[2] * (o_end - o_start + 1)]);
    //                                 }
    // }

    // Create CuboidBSGpu buffer, planes buf and send and recv bufs
    // TODO update in Problem init when using blockstencils
    auto d_cl = std::make_shared<mgcl::BlockstencilGpu>(cl, p.getContext(), p.getCommands(), p.getProgram());
    int yz = d_cl->getNgh() * d_cl->getOgh(); // 324
    int xz = d_cl->getMgh() * d_cl->getOgh(); // 180
    int xy = d_cl->getMgh() * d_cl->getNgh(); // 180
    int ressize = (2 * yz * d_cl->getGh() + 2 * xz * d_cl->getGh() + 2 * xy * d_cl->getGh()) * blocksize2 * stencilSize;
    mgcl::BufferGpu dPlanesBuf(p.getContext(), CL_MEM_READ_WRITE, ressize);
    auto hPlanesBufSend = std::make_shared<std::vector<double>>(ressize);
    auto hPlanesBufRecv = std::make_shared<std::vector<double>>(ressize);

    // if (mpi_rank == 0)
    // {
    //     int yzgh = yz * d_cl->getGh();
    //     int xzgh = xz * d_cl->getGh();
    //     int xygh = xy * d_cl->getGh();
    //     std::cout << "yzgh: " << yzgh << ", xzgh: " << xzgh << ", xygh: " << xygh << std::endl;

    //     std::cout << "starting indices:" << std::endl;
    //     std::cout << "  front: " << 0 << std::endl
    //               << "  back: " << yzgh * blocksize2 * stencilSize << std::endl
    //               << "  top: " << 2 * yzgh * blocksize2 * stencilSize << std::endl
    //               << "  bottom: " << 2 * yzgh * blocksize2 * stencilSize + xzgh * blocksize2 * stencilSize << std::endl
    //               << "  left: " << 2 * yzgh * blocksize2 * stencilSize + 2 * xzgh * blocksize2 * stencilSize << std::endl
    //               << "  right: " << 2 * yzgh * blocksize2 * stencilSize + 2 * xzgh * blocksize2 * stencilSize + xygh * blocksize2 * stencilSize << std::endl;
    //     // yzgh: 100, xzgh: 60, xygh: 60
    //     // starting indices:
    //     //   front: 0
    //     //   back: 10800
    //     //   top: 21600
    //     //   bottom: 28080
    //     //   left: 34560
    //     //   right: 41040
    // }
    // MPI_Barrier(mpi_comm);

    // Update ghosts of test data using MPI
    d_cl->updateGhostsOclMpi(p.getProgram(), p.getCommands(),
                             dPlanesBuf, *hPlanesBufSend, *hPlanesBufRecv,
                             *mpiData, false, periodic,
                             &p.getKernelConfig(), p.getProfilingData());
    p.finish();

    auto cl_res = d_cl->read(p.getCommands(), true);

    // cl.dumpToFile("cl" + std::to_string(mpi_rank) + ".txt");
    // cl_res.dumpToFile("cl_res" + std::to_string(mpi_rank) + ".txt");

    // if (mpi_rank == 0)
    // {
    //     cg.dumpToFile("cg.txt");
    // }

    // Check result
    // check in z-direction
    for (int bi = 0; bi < blocksize; bi++)
        for (int bj = 0; bj < blocksize; bj++)
            for (int ii = 0; ii < width; ii++)
                for (int jj = 0; jj < width; jj++)
                    for (int kk = 0; kk < width; kk++)
                        for (int i = 0; i < gh; i++)
                            for (int j = 0; j < nl + 2 * gh; j++)
                                for (int k = 0; k < ol + 2 * gh; k++)
                                {
                                    CAPTURE(i, j, k, ii, jj, kk, bi, bj, mpi_rank);
                                    if (mpi_coords[0] > 0) // not the first process
                                        REQUIRE(cl_res[bi][bj][ii][jj][kk][i][j][k] == cg[bi][bj][ii][jj][kk][m_start + i][j + n_start][k + o_start]);
                                    else if (periodic)
                                        REQUIRE(cl_res[bi][bj][ii][jj][kk][i][j][k] == cg[bi][bj][ii][jj][kk][i + m][j + n_start][k + o_start]);

                                    if (mpi_coords[0] < mpi_dims[0]) // not the last process
                                        REQUIRE(cl_res[bi][bj][ii][jj][kk][i + gh + ml][j][k] == cg[bi][bj][ii][jj][kk][m_end + gh + 1 + i][j + n_start][k + o_start]);
                                    else if (periodic)
                                        REQUIRE(cl_res[bi][bj][ii][jj][kk][i + gh + ml][j][k] == cg[bi][bj][ii][jj][kk][i + gh][j + n_start][k + o_start]);
                                }

    // check in y-direction
    for (int bi = 0; bi < blocksize; bi++)
        for (int bj = 0; bj < blocksize; bj++)
            for (int ii = 0; ii < width; ii++)
                for (int jj = 0; jj < width; jj++)
                    for (int kk = 0; kk < width; kk++)
                        for (int i = 0; i < ml + 2 * gh; i++)
                            for (int j = 0; j < gh; j++)
                                for (int k = 0; k < ol + 2 * gh; k++)
                                {
                                    CAPTURE(i, j, k, ii, jj, kk, bi, bj, mpi_rank);
                                    if (mpi_coords[1] > 0) // not the first process
                                        REQUIRE(cl_res[bi][bj][ii][jj][kk][i][j][k] == cg[bi][bj][ii][jj][kk][i + m_start][n_start + j][k + o_start]);
                                    else if (periodic)
                                        REQUIRE(cl_res[bi][bj][ii][jj][kk][i][j][k] == cg[bi][bj][ii][jj][kk][i + m_start][j + n][k + o_start]);

                                    if (mpi_coords[1] < mpi_dims[1]) // not the last process
                                        REQUIRE(cl_res[bi][bj][ii][jj][kk][i][j + gh + nl][k] == cg[bi][bj][ii][jj][kk][i + m_start][n_end + gh + 1 + j][k + o_start]);
                                    else if (periodic)
                                        REQUIRE(cl_res[bi][bj][ii][jj][kk][i][j + gh + nl][k] == cg[bi][bj][ii][jj][kk][i + m_start][j + gh][k + o_start]);
                                }

    // check in x-direction
    for (int bi = 0; bi < blocksize; bi++)
        for (int bj = 0; bj < blocksize; bj++)
            for (int ii = 0; ii < width; ii++)
                for (int jj = 0; jj < width; jj++)
                    for (int kk = 0; kk < width; kk++)
                        for (int i = 0; i < ml + 2 * gh; i++)
                            for (int j = 0; j < nl + 2 * gh; j++)
                                for (int k = 0; k < gh; k++)
                                {
                                    CAPTURE(i, j, k, ii, jj, kk, bi, bj, mpi_rank);
                                    if (mpi_coords[2] > 0) // not the first process
                                        REQUIRE(cl_res[bi][bj][ii][jj][kk][i][j][k] == cg[bi][bj][ii][jj][kk][i + m_start][j + n_start][o_start + k]);
                                    else if (periodic)
                                        REQUIRE(cl_res[bi][bj][ii][jj][kk][i][j][k] == cg[bi][bj][ii][jj][kk][i + m_start][j + n_start][k + o]);

                                    if (mpi_coords[2] < mpi_dims[2]) // not the last process
                                        REQUIRE(cl_res[bi][bj][ii][jj][kk][i][j][k + gh + ol] == cg[bi][bj][ii][jj][kk][i + m_start][j + n_start][o_end + gh + 1 + k]);
                                    else if (periodic)
                                        REQUIRE(cl_res[bi][bj][ii][jj][kk][i][j][k + gh + ol] == cg[bi][bj][ii][jj][kk][i + m_start][j + n_start][k + gh]);
                                }
}