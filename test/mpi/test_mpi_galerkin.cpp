#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
// #include <iostream>
#include <memory>

#include "../../src/mgcl/cuboid.hpp"
#include "../../src/mgcl/mpi_stencil.hpp"
#include "../../src/mgcl/mpi_util.hpp"
#include "../../src/mgcl/multigrid_engine.hpp"
#include "../../src/mgcl/problem.hpp"

#include "mpi.h"

// Extracted from Level::initMpiData
int initMpiDataLevel(mgcl::MPILevelData* mpiData, mgcl::MPILevelData* mpiDataAbove, mgcl::Problem* problem, int num,
                     int m, int n, int o)
{
    int ret = 0;

    // if (!isCalculatedLocally())
    {
        // MPI variables
        MPI_Comm mpi_comm = problem->getMpiComm();
        bool periodic = problem->isPeriodic();

        if (mpiData->mpiSize() == 1)
        {
            mpiData->left = mpiData->rank;
            mpiData->right = mpiData->rank;
            mpiData->down = mpiData->rank;
            mpiData->up = mpiData->rank;
            mpiData->back = mpiData->rank;
            mpiData->front = mpiData->rank;
            return MPI_SUCCESS;
        }

        if (num == 0)
        {
            /* Calculating neighbours */
            ret = MPI_Cart_shift(mpiData->comm, 2, 1, &mpiData->left, &mpiData->right);
            mgcl::mpi_util::mgclCheckMpiError(mpiData->comm, ret, "MPI_Cart_shift x-direction");
            ret = MPI_Cart_shift(mpiData->comm, 1, 1, &mpiData->down, &mpiData->up);
            mgcl::mpi_util::mgclCheckMpiError(mpiData->comm, ret, "MPI_Cart_shift y-direction");
            ret = MPI_Cart_shift(mpiData->comm, 0, 1, &mpiData->front, &mpiData->back);
            mgcl::mpi_util::mgclCheckMpiError(mpiData->comm, ret, "MPI_Cart_shift z-direction");
        }
        else
        {
            // Send and get ranks of neighbours that still have nodes (i.e. m,n,o > 0).

            /* Temporary buffer (for receiving unused messages) */
            int tmpbuf;

            /* MPI variables */
            int myid;
            MPI_Request reqs[2];
            MPI_Status stats[2];

            // mgcl::Level& levelAbove = problem->getLevelAt(num - 1);

            /* Getting local rank */
            MPI_Comm_rank(mpi_comm, &myid);

            if (periodic)
            {
                /* Initializing neighbours */
                mpiData->left = myid;
                mpiData->right = myid;
                mpiData->down = myid;
                mpiData->up = myid;
                mpiData->back = myid;
                mpiData->front = myid;
            }
            else
            {
                /* Initializing neighbours */
                mpiData->left = MPI_PROC_NULL;
                mpiData->right = MPI_PROC_NULL;
                mpiData->down = MPI_PROC_NULL;
                mpiData->up = MPI_PROC_NULL;
                mpiData->back = MPI_PROC_NULL;
                mpiData->front = MPI_PROC_NULL;
            }

            if (m > 0)
            {
                /* Sending data to left */
                reqs[0] = MPI_REQUEST_NULL;
                reqs[1] = MPI_REQUEST_NULL;

                if ((myid != mpiDataAbove->left) && (MPI_PROC_NULL != mpiDataAbove->left))
                    MPI_Isend((void*)&myid, 1, MPI_INT, mpiDataAbove->left, 0, mpi_comm, &reqs[0]);

                if ((myid != mpiDataAbove->right) && (MPI_PROC_NULL != mpiDataAbove->right))
                    MPI_Irecv((void*)&mpiData->right, 1, MPI_INT, mpiDataAbove->right, 0, mpi_comm, &reqs[1]);

                MPI_Waitall(2, reqs, stats);

                /* Sending data to right */
                reqs[0] = MPI_REQUEST_NULL;
                reqs[1] = MPI_REQUEST_NULL;
                if ((myid != mpiDataAbove->right) && (MPI_PROC_NULL != mpiDataAbove->right))
                    MPI_Isend((void*)&myid, 1, MPI_INT, mpiDataAbove->right, 0, mpi_comm,
                              &reqs[0]);
                if ((myid != mpiDataAbove->left) && (MPI_PROC_NULL != mpiDataAbove->left))
                    MPI_Irecv((void*)&mpiData->left, 1, MPI_INT, mpiDataAbove->left, 0,
                              mpi_comm, &reqs[1]);
                MPI_Waitall(2, reqs, stats);
            }
            else
            {
                /* Sending data to left */
                reqs[0] = MPI_REQUEST_NULL;
                reqs[1] = MPI_REQUEST_NULL;
                if ((myid != mpiDataAbove->left) && (MPI_PROC_NULL != mpiDataAbove->left))
                    MPI_Isend((void*)&mpiDataAbove->right, 1, MPI_INT, mpiDataAbove->left,
                              0, mpi_comm, &reqs[0]);
                if ((myid != mpiDataAbove->right) && (MPI_PROC_NULL != mpiDataAbove->right))
                    MPI_Irecv((void*)&tmpbuf, 1, MPI_INT, mpiDataAbove->right,
                              0, mpi_comm, &reqs[1]);
                MPI_Waitall(2, reqs, stats);

                /* Sending data to right */
                reqs[0] = MPI_REQUEST_NULL;
                reqs[1] = MPI_REQUEST_NULL;
                if ((myid != mpiDataAbove->right) && (MPI_PROC_NULL != mpiDataAbove->right))
                    MPI_Isend((void*)&mpiDataAbove->left, 1, MPI_INT, mpiDataAbove->right,
                              0, mpi_comm, &reqs[0]);
                if ((myid != mpiDataAbove->left) && (MPI_PROC_NULL != mpiDataAbove->left))
                    MPI_Irecv((void*)&tmpbuf, 1, MPI_INT, mpiDataAbove->left, 0,
                              mpi_comm, &reqs[1]);
                MPI_Waitall(2, reqs, stats);
            }

            if (n > 0)
            {
                /* Sending data downwards */
                reqs[0] = MPI_REQUEST_NULL;
                reqs[1] = MPI_REQUEST_NULL;
                if ((myid != mpiDataAbove->down) && (MPI_PROC_NULL != mpiDataAbove->down))
                    MPI_Isend((void*)&myid, 1, MPI_INT, mpiDataAbove->down, 0, mpi_comm,
                              &reqs[0]);
                if ((myid != mpiDataAbove->up) && (MPI_PROC_NULL != mpiDataAbove->up))
                    MPI_Irecv((void*)&mpiData->up, 1, MPI_INT, mpiDataAbove->up,
                              0, mpi_comm, &reqs[1]);
                MPI_Waitall(2, reqs, stats);

                /* Sending data upwards */
                reqs[0] = MPI_REQUEST_NULL;
                reqs[1] = MPI_REQUEST_NULL;
                if ((myid != mpiDataAbove->up) && (MPI_PROC_NULL != mpiDataAbove->up))
                    MPI_Isend((void*)&myid, 1, MPI_INT, mpiDataAbove->up, 0, mpi_comm,
                              &reqs[0]);
                if ((myid != mpiDataAbove->down) && (MPI_PROC_NULL != mpiDataAbove->down))
                    MPI_Irecv((void*)&mpiData->down, 1, MPI_INT, mpiDataAbove->down,
                              0, mpi_comm, &reqs[1]);
                MPI_Waitall(2, reqs, stats);
            }
            else
            {
                /* Sending data downwards */
                reqs[0] = MPI_REQUEST_NULL;
                reqs[1] = MPI_REQUEST_NULL;
                if ((myid != mpiDataAbove->down) && (MPI_PROC_NULL != mpiDataAbove->down))
                    MPI_Isend((void*)&mpiDataAbove->up, 1, MPI_INT,
                              mpiDataAbove->down, 0, mpi_comm, &reqs[0]);
                if ((myid != mpiDataAbove->up) && (MPI_PROC_NULL != mpiDataAbove->up))
                    MPI_Irecv((void*)&tmpbuf, 1, MPI_INT, mpiDataAbove->up,
                              0, mpi_comm, &reqs[1]);
                MPI_Waitall(2, reqs, stats);

                /* Sending data upwards */
                reqs[0] = MPI_REQUEST_NULL;
                reqs[1] = MPI_REQUEST_NULL;
                if ((myid != mpiDataAbove->up) && (MPI_PROC_NULL != mpiDataAbove->up))
                    MPI_Isend((void*)&mpiDataAbove->down, 1, MPI_INT,
                              mpiDataAbove->up, 0, mpi_comm, &reqs[0]);
                if ((myid != mpiDataAbove->down) && (MPI_PROC_NULL != mpiDataAbove->down))
                    MPI_Irecv((void*)&tmpbuf, 1, MPI_INT, mpiDataAbove->down,
                              0, mpi_comm, &reqs[1]);
                MPI_Waitall(2, reqs, stats);
            }

            if (o > 0)
            {
                /* Sending data to back */
                reqs[0] = MPI_REQUEST_NULL;
                reqs[1] = MPI_REQUEST_NULL;
                if ((myid != mpiDataAbove->back) && (MPI_PROC_NULL != mpiDataAbove->back))
                    MPI_Isend((void*)&myid, 1, MPI_INT, mpiDataAbove->back, 0, mpi_comm,
                              &reqs[0]);
                if ((myid != mpiDataAbove->front) && (MPI_PROC_NULL != mpiDataAbove->front))
                    MPI_Irecv((void*)&mpiData->front, 1, MPI_INT, mpiDataAbove->front,
                              0, mpi_comm, &reqs[1]);
                MPI_Waitall(2, reqs, stats);

                /* Sending data to front */
                reqs[0] = MPI_REQUEST_NULL;
                reqs[1] = MPI_REQUEST_NULL;
                if ((myid != mpiDataAbove->front) && (MPI_PROC_NULL != mpiDataAbove->front))
                    MPI_Isend((void*)&myid, 1, MPI_INT, mpiDataAbove->front, 0, mpi_comm,
                              &reqs[0]);
                if ((myid != mpiDataAbove->back) && (MPI_PROC_NULL != mpiDataAbove->back))
                    MPI_Irecv((void*)&mpiData->back, 1, MPI_INT, mpiDataAbove->back,
                              0, mpi_comm, &reqs[1]);
                MPI_Waitall(2, reqs, stats);
            }
            else
            {
                /* Sending data to back */
                reqs[0] = MPI_REQUEST_NULL;
                reqs[1] = MPI_REQUEST_NULL;
                if ((myid != mpiDataAbove->back) && (MPI_PROC_NULL != mpiDataAbove->back))
                    MPI_Isend((void*)&mpiDataAbove->front, 1, MPI_INT,
                              mpiDataAbove->back, 0, mpi_comm, &reqs[0]);
                if ((myid != mpiDataAbove->front) && (MPI_PROC_NULL != mpiDataAbove->front))
                    MPI_Irecv((void*)&tmpbuf, 1, MPI_INT, mpiDataAbove->front,
                              0, mpi_comm, &reqs[1]);
                MPI_Waitall(2, reqs, stats);

                /* Sending data to front */
                reqs[0] = MPI_REQUEST_NULL;
                reqs[1] = MPI_REQUEST_NULL;
                if ((myid != mpiDataAbove->front) && (MPI_PROC_NULL != mpiDataAbove->front))
                    MPI_Isend((void*)&mpiDataAbove->back, 1, MPI_INT,
                              mpiDataAbove->front, 0, mpi_comm, &reqs[0]);
                if ((myid != mpiDataAbove->back) && (MPI_PROC_NULL != mpiDataAbove->back))
                    MPI_Irecv((void*)&tmpbuf, 1, MPI_INT, mpiDataAbove->back,
                              0, mpi_comm, &reqs[1]);
                MPI_Waitall(2, reqs, stats);
            }

            if (m <= 0 || n <= 0 || o <= 0)
            {
                mpiData->left = myid;
                mpiData->right = myid;
                mpiData->down = myid;
                mpiData->up = myid;
                mpiData->back = myid;
                mpiData->front = myid;
            }
        }
    }

    return ret;
}

// Checks if sequential optimized galerkin calculation works distributively using multiple MPI processes.
TEST_CASE("MPI_seq_galerkinOptimized_nprocs")
{
    using std::min;

    // global grid sizes
    int m = 16;
    int n = 16;
    int o = 16;
    int periodic = 1;
    int gh = 1;

    // Problem parameters
    double tol = 1e-7;
    int nu1 = 2;
    int nu2 = 2;
    double omega = 0.8;
    int maxIterVCycles = 5;

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

    // Init some random data
    auto vloc = std::make_shared<mgcl::Cuboid>(ml, nl, ol);
    auto floc = std::make_shared<mgcl::Cuboid>(ml, nl, ol);
    vloc->fillRandom();
    floc->fillRandom();

    // Create Problem to init all the MPI stuff
    mgcl::Problem p(ml, nl, ol, floc, vloc, m, n, o);
    p.setMpiLevelThreshold(3);
    p.setMpiComm(mpi_comm);

    // p.setStencilType(mgcl::MGCL_VARYING);
    // auto& s = *p.getStencilValues();
    // s.fill1dIndex(true);

    // Initialize Problem partially, without calculating the Galerkin operator.
    // from Problem::init
    // Create cartesian process grid if none was set and more than one processes are used.
    p.getMPIGlobalData().createCartGrid(periodic);
    p.calculateAndSetMpiLevelThreshold();

    mgcl::MPILevelData mpiDataFine(mpi_comm);
    initMpiDataLevel(&mpiDataFine, nullptr, &p, 0, ml, nl, ol);
    mgcl::MPILevelData mpiDataCoarse(mpi_comm);
    initMpiDataLevel(&mpiDataCoarse, &mpiDataFine, &p, 1, ml >> 1, nl >> 1, ol >> 1);

    // Create locally sized varying stencil on each MPI process
    mgcl::VaryingStencil a_h_loc(ml, nl, ol, 3, 1, 1, 1);
    // fill with 1d global index, depending on this process' coordinates, in order to get the same values as in the
    // global stencil values
    // clang-format off
    for (int i = a_h_loc.getGhostsM(); i < a_h_loc.getGhostsM() + a_h_loc.getM(); i++)
    for (int j = a_h_loc.getGhostsN(); j < a_h_loc.getGhostsN() + a_h_loc.getN(); j++)
    for (int k = a_h_loc.getGhostsO(); k < a_h_loc.getGhostsO() + a_h_loc.getO(); k++)
        for (int ii = 0; ii < 3; ii++)
        for (int jj = 0; jj < 3; jj++)
        for (int kk = 0; kk < 3; kk++)
        {
            int mgh = m + 2 * a_h_loc.getGhostsM();
            int ngh = n + 2 * a_h_loc.getGhostsN();
            int ogh = o + 2 * a_h_loc.getGhostsO();
            int itarget = ml * mpi_coords[0] + i;
            int jtarget = nl * mpi_coords[1] + j;
            int ktarget = ol * mpi_coords[2] + k;
            a_h_loc[ii][jj][kk][i][j][k] = ktarget + jtarget * ogh + itarget * ogh * ngh + kk * ogh * ngh * mgh + jj * ogh * ngh * mgh * 3 + ii * ogh * ngh * mgh * 3 * 3;
        }
    // clang-format on
    mgcl::updateGhostsStencilMpi(a_h_loc, &mpiDataFine, periodic, false);

    auto a_2h_loc = mgcl::MultigridEngine::galerkinOptimized(
        a_h_loc, 1, a_h_loc.getM() >> 1, a_h_loc.getN() >> 1, a_h_loc.getO() >> 1);

    // a_2h_loc->dumpToFile(std::to_string(mpi_rank) + "_a_2h_loc.txt");

    MPI_Barrier(mpi_comm);

    // Create globally sized varying stencil only on root and calculate Galerkin operator locally
    std::unique_ptr<mgcl::VaryingStencil> a_2h_glob;
    if (mpi_rank == 0)
    {
        mgcl::VaryingStencil a_h_glob(m, n, o, 3, 1, 1, 1);
        a_h_glob.fill1dIndex(true);
        a_h_glob.updateGhosts();

        // Check that local and global stencil values on fine grid (i.e. the input) are the same (at least for root)
        // clang-format off
        for (int i = a_h_loc.getGhostsM(); i < a_h_loc.getGhostsM() + a_h_loc.getM(); i++)
        for (int j = a_h_loc.getGhostsN(); j < a_h_loc.getGhostsN() + a_h_loc.getN(); j++)
        for (int k = a_h_loc.getGhostsO(); k < a_h_loc.getGhostsO() + a_h_loc.getO(); k++)
            for (int ii = 0; ii < 3; ii++)
            for (int jj = 0; jj < 3; jj++)
            for (int kk = 0; kk < 3; kk++)
                {
                    CAPTURE(mpi_rank, i,j,k,ii,jj,kk, mpi_coords[0], mpi_coords[1], mpi_coords[2]);                
                    REQUIRE(a_h_loc[ii][jj][kk][i][j][k] == a_h_glob[ii][jj][kk][i][j][k]);
                }
        // clang-format on

        a_2h_glob = mgcl::MultigridEngine::galerkinOptimized(
            a_h_glob, 1, a_h_glob.getM() >> 1, a_h_glob.getN() >> 1, a_h_glob.getO() >> 1);

        // a_2h_glob->dumpToFile(std::to_string(mpi_rank) + "_a_2h_glob.txt");
    }

    // Gather local approximations on rank 0 for checking.
    if (mpi_rank > 0)
        mgcl::mpi_util::gather(p.getMpiComm(), *a_2h_loc);
    else
    {
        mgcl::VaryingStencil a_2h_glob_check(m >> 1, n >> 1, o >> 1, 3, 1, 1, 1);

        // copy from a_2h_loc to a_2h_glob first
        for (int i = a_2h_loc->getGhostsM(); i < a_2h_loc->getM() + a_2h_loc->getGhostsM(); i++)
            for (int j = a_2h_loc->getGhostsN(); j < a_2h_loc->getN() + a_2h_loc->getGhostsN(); j++)
                for (int k = a_2h_loc->getGhostsO(); k < a_2h_loc->getO() + a_2h_loc->getGhostsO(); k++)
                    for (int ii = 0; ii < a_2h_loc->getWidth(); ii++)
                        for (int jj = 0; jj < a_2h_loc->getWidth(); jj++)
                            for (int kk = 0; kk < a_2h_loc->getWidth(); kk++)
                            {
                                CAPTURE(i, j, k, ii, jj, kk);
                                a_2h_glob_check[ii][jj][kk][i][j][k] = (*a_2h_loc)[ii][jj][kk][i][j][k];
                            }

        // Gather into a_h_glob_check from other processes
        if (mpi_size > 1)
            mgcl::mpi_util::gather(p.getMpiComm(), a_2h_glob_check); // TODO check with different ghost amounts

        REQUIRE(a_2h_glob_check.isEqual(*a_2h_glob));
    }
}

// Test that Galerkin yields the same results for different threshold levels.
// run with e.g. mpiexec -n 8 tests_mpi MPI_galerkin_different_thresholds
TEST_CASE("MPI_seq_galerkin_different_thresholds")
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

    // Init Problem with threshold level 0
    auto pptr_th0 = std::make_shared<mgcl::Problem>(ml, nl, ol, f, v, m, n, o);
    auto& p_th0 = *pptr_th0;
    p_th0.setMpiComm(mpi_comm);
    p_th0.setMpiLevelThreshold(0);

    p_th0.setStencilType(mgcl::MGCL_VARYING);
    auto& sv0 = p_th0.getStencilValues();
    sv0->fill1dIndex(false);

    p_th0.init();

    // Init Problem with threshold level 1
    auto pptr_th1 = std::make_shared<mgcl::Problem>(ml, nl, ol, f, v, m, n, o);
    auto& p_th1 = *pptr_th1;
    p_th1.setMpiComm(mpi_comm);
    p_th1.setMpiLevelThreshold(0);

    p_th1.setStencilType(mgcl::MGCL_VARYING);
    auto& sv = p_th1.getStencilValues();
    sv->fill1dIndex(false);

    p_th1.init();

    // Init Problem with threshold level 2
    auto pptr_th2 = std::make_shared<mgcl::Problem>(ml, nl, ol, f, v, m, n, o);
    auto& p_th2 = *pptr_th2;
    p_th2.setMpiComm(mpi_comm);
    p_th2.setMpiLevelThreshold(0);

    p_th2.setStencilType(mgcl::MGCL_VARYING);
    sv = p_th2.getStencilValues();
    // copy data manually since for threshold 2 stencil values has local size.
    for (int i = 0; i < sv->getMgh(); i++)
        for (int j = 0; j < sv->getNgh(); j++)
            for (int k = 0; k < sv->getOgh(); k++)
                for (int ii = 0; ii < sv->getWidth(); ii++)
                    for (int jj = 0; jj < sv->getWidth(); jj++)
                        for (int kk = 0; kk < sv->getWidth(); kk++)
                            sv->getData()[ii][jj][kk][i][j][k] = sv0->getData()[ii][jj][kk][i][j][k];

    p_th2.init();

    if (mpi_rank == 0)
    {
        // Stencil values for problems with thresholds 0 and 1 should be the same for each level.
        for (int lv = 0; lv < p_th0.getMaxlevel(); lv++)
            REQUIRE(p_th0.getLevelAt(lv).getStencilValues()->isEqual(*p_th1.getLevelAt(lv).getStencilValues()));

        // Stencil values for problems with thresholds 0 and 2 should be the same for each level starting with 1.
        for (int lv = 1; lv < p_th0.getMaxlevel(); lv++)
            REQUIRE(p_th0.getLevelAt(lv).getStencilValues()->isEqual(*p_th2.getLevelAt(lv).getStencilValues()));
    }
}