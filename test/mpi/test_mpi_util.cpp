#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <csetjmp>
#include <iostream>
#include <memory>
#include <string>

#include "../../src/mgcl/cuboid.hpp"
#include "../../src/mgcl/cuboid_gpu.hpp"
#include "../../src/mgcl/mpi_level_data.hpp"
#include "../../src/mgcl/mpi_util.hpp"
#include "../../src/mgcl/multigrid_engine.hpp"
#include "../../src/mgcl/problem.hpp"
#include "../../src/mgcl/stencil.hpp"
#include "../cli_args.hpp"
#include "../device_type_generator.hpp"
#include "../test_utility.hpp"

#include "mpi.h"

// Checks that gathering is correct having each proess send their local grid and rank 0
// receiving into big global grid.
// Does NOT use mpi_util::gather.
// Run with e.g. mpiexec -n 8 tests_mpi "mpi_util::gather-src-dest-different"
TEST_CASE("mpi_util::gather-src-dest-different")
{
    using std::min;
    int err;

    int mloc = 8;
    int nloc = 8;
    int oloc = 8;
    int gh = 1;
    int mlocgh = mloc + 2 * gh;
    int nlocgh = nloc + 2 * gh;
    int olocgh = oloc + 2 * gh;
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

    // Calculate global sizes
    int mglob = mloc * mpi_dims[0];
    int nglob = nloc * mpi_dims[1];
    int oglob = oloc * mpi_dims[2];
    int mglobgh = mglob + 2 * gh;
    int nglobgh = nglob + 2 * gh;
    int oglobgh = oglob + 2 * gh;

    /* Initialize start and end for local grid */
    int m_start = mloc * mpi_coords[0] + min(mpi_coords[0], (mglob % mpi_dims[0]));
    int m_end = mloc * (mpi_coords[0] + 1) + min(mpi_coords[0] + 1, (mglob % mpi_dims[0])) - 1;
    int n_start = nloc * mpi_coords[1] + min(mpi_coords[1], (nglob % mpi_dims[1]));
    int n_end = nloc * (mpi_coords[1] + 1) + min(mpi_coords[1] + 1, (nglob % mpi_dims[1])) - 1;
    int o_start = oloc * mpi_coords[2] + min(mpi_coords[2], (oglob % mpi_dims[2]));
    int o_end = oloc * (mpi_coords[2] + 1) + min(mpi_coords[2] + 1, (oglob % mpi_dims[2])) - 1;

    // for (int i = 0; i < mpi_size; i++)
    // {
    //     MPI_Barrier(mpi_comm);
    //     if (i == mpi_rank)
    //     {
    //         std::cout << "rank,ms,me,ns,ne,os,oe: "
    //                   << mpi_rank << ","
    //                   << m_start << "," << m_end << ","
    //                   << n_start << "," << n_end << ","
    //                   << o_start << "," << o_end << std::endl;
    //         std::cout << "coords: " << mpi_coords[0] << "," << mpi_coords[1] << "," << mpi_coords[2] << std::endl;
    //     }
    // }

    // Create test data
    mgcl::Cuboid cglob(mglob, nglob, oglob, gh, gh, gh);
    cglob.fill1dIndex(true);

    auto cloc = cglob.slice(m_start, m_end, n_start, n_end, o_start, o_end);

    // Gather
    mgcl::Cuboid cglob_recv(mglob, nglob, oglob, gh, gh, gh);

    // Create subarray type for the send buffer
    MPI_Datatype subarraySend;
    {
        int sizes[3] = {mlocgh, nlocgh, olocgh};
        int subsizes[3] = {mloc, nloc, oloc};
        int starts[3] = {gh, gh, gh};
        err = MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &subarraySend);
        mgcl::mpi_util::mgclCheckMpiError(mpi_comm, err, "MPI_Type_create_subarray");
        err = MPI_Type_commit(&subarraySend);
        mgcl::mpi_util::mgclCheckMpiError(mpi_comm, err, "MPI_Type_commit");
    }

    // Create subarray type for the receive buffer
    MPI_Datatype subarrayRecv;
    MPI_Datatype subarrayRecvResized;
    {
        int sizes[3] = {mglobgh, nglobgh, oglobgh};
        int subsizes[3] = {mloc, nloc, oloc};
        int starts[3] = {gh, gh, gh};
        // int starts[3] = {0, 0, 0};
        err = MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &subarrayRecv);
        mgcl::mpi_util::mgclCheckMpiError(mpi_comm, err, "MPI_Type_create_subarray");
        err = MPI_Type_commit(&subarrayRecv);
        mgcl::mpi_util::mgclCheckMpiError(mpi_comm, err, "MPI_Type_commit");

        // Resize recv data to avoid overlapping resulting from ghosts. Enabling explicitely stating start and extent
        // later in MPI_Gatherv (counts and displ). One unit is the size of a local grid.
        err = MPI_Type_create_resized(subarrayRecv, 0, 1 * sizeof(double), &subarrayRecvResized);
        // err = MPI_Type_create_resized(subarrayRecv, 0, oglobgh * sizeof(double), &subarrayRecvResized);
        mgcl::mpi_util::mgclCheckMpiError(mpi_comm, err, "MPI_Type_create_resized");
        err = MPI_Type_commit(&subarrayRecvResized);
        mgcl::mpi_util::mgclCheckMpiError(mpi_comm, err, "MPI_Type_commit");
    }

    MPI_Barrier(mpi_comm);

    // Calculate displacements for each processor.
    int counts[mpi_size];
    int displ[mpi_size];
    for (int i = 0; i < mpi_size; i++)
    {
        int coords[3] = {0, 0, 0};
        err = MPI_Cart_coords(mpi_comm, i, 3, coords);
        mgcl::mpi_util::mgclCheckMpiError(mpi_comm, err, "MPI_Cart_coords");

        counts[i] = 1;
        displ[i] = coords[0] * mloc * nglobgh * oglobgh + coords[1] * nloc * oglobgh + coords[2] * oloc;
    }
    MPI_Barrier(mpi_comm);

    // int counts[4] = {1, 1, 1, 1};
    // int displ[4] = {0, nloc, mloc * nglobgh, mloc * nglobgh + nloc};
    // int counts[2] = {1, 1};
    // int displ[2] = {0, mloc * nglobgh * oglobgh};

    // Alternative with
    // int starts[3] = {0, 0, 0}; and
    // MPI_Type_create_resized(subarrayRecv, 0, 1 * sizeof(double), &subarrayRecvResized);
    //
    // int counts[4] = {1, 1, 1, 1};
    // int displ[4] = {nglobgh * oglobgh + oglobgh + gh,
    //                 nglobgh * oglobgh + (nloc + gh) * oglobgh + gh,
    //                 (mloc + gh) * nglobgh * oglobgh + oglobgh + gh,
    //                 (mloc + gh) * nglobgh * oglobgh + (nloc + gh) * oglobgh + gh};
    // int counts[2] = {1, 1};
    // int displ[2] = {nglobgh * oglobgh + oglobgh + gh, (mloc + gh) * nglobgh * oglobgh + oglobgh + gh};
    // CAPTURE(displ);

    MPI_Gatherv(cloc->field1d().data(), 1, subarraySend,
                cglob_recv.field1d().data(), counts, displ, subarrayRecvResized, 0, mpi_comm);
    mgcl::mpi_util::mgclCheckMpiError(mpi_comm, err, "MPI_Gather");

    mgcl::mpi_util::mgclCheckMpiError(mpi_comm, MPI_Type_free(&subarraySend), "MPI_Type_free");
    mgcl::mpi_util::mgclCheckMpiError(mpi_comm, MPI_Type_free(&subarrayRecv), "MPI_Type_free");
    mgcl::mpi_util::mgclCheckMpiError(mpi_comm, MPI_Type_free(&subarrayRecvResized), "MPI_Type_free");

    MPI_Barrier(mpi_comm);
    if (mpi_rank == 0)
    {
        CAPTURE(cglob_recv.getM(), cglob_recv.getN(), cglob_recv.getO());
        // cglob_recv.dumpToFile("cglob_recv.txt");
        // cglob.dumpToFile("cglob.txt");

        // Check result
        REQUIRE(cglob.isEqual(cglob_recv));
    }
    MPI_Barrier(mpi_comm);
}

// Checks that gathering is correct while rank 0 has the same globally sized buffer for sending and receiving,
// while all other processes only send local grids.
// Uses mpi_util::gather.
// Run with e.g. mpiexec -n 8 tests_mpi "mpi_util::gather-src-dest-same"
TEST_CASE("mpi_util::gather-src-dest-same")
{
    using std::min;

    int mloc = 8;
    int nloc = 8;
    int oloc = 8;
    int gh = 1;
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

    // Calculate global sizes
    int mglob = mloc * mpi_dims[0];
    int nglob = nloc * mpi_dims[1];
    int oglob = oloc * mpi_dims[2];

    /* Initialize start and end for local grid */
    int m_start = mloc * mpi_coords[0] + min(mpi_coords[0], (mglob % mpi_dims[0]));
    int m_end = mloc * (mpi_coords[0] + 1) + min(mpi_coords[0] + 1, (mglob % mpi_dims[0])) - 1;
    int n_start = nloc * mpi_coords[1] + min(mpi_coords[1], (nglob % mpi_dims[1]));
    int n_end = nloc * (mpi_coords[1] + 1) + min(mpi_coords[1] + 1, (nglob % mpi_dims[1])) - 1;
    int o_start = oloc * mpi_coords[2] + min(mpi_coords[2], (oglob % mpi_dims[2]));
    int o_end = oloc * (mpi_coords[2] + 1) + min(mpi_coords[2] + 1, (oglob % mpi_dims[2])) - 1;

    // for (int i = 0; i < mpi_size; i++)
    // {
    //     MPI_Barrier(mpi_comm);
    //     if (i == mpi_rank)
    //     {
    //         std::cout << "rank,ms,me,ns,ne,os,oe: "
    //                   << mpi_rank << ","
    //                   << m_start << "," << m_end << ","
    //                   << n_start << "," << n_end << ","
    //                   << o_start << "," << o_end << std::endl;
    //         std::cout << "coords: " << mpi_coords[0] << "," << mpi_coords[1] << "," << mpi_coords[2] << std::endl;
    //     }
    // }

    // Create test data
    mgcl::Cuboid cglob(mglob, nglob, oglob, gh, gh, gh);
    cglob.fill1dIndex(true);

    // Recv buffer, partially filled. Has global size on rank 0, local size else.
    std::unique_ptr<mgcl::Cuboid> cglob_recv;
    if (mpi_rank == 0)
    {
        cglob_recv = std::make_unique<mgcl::Cuboid>(mglob, nglob, oglob, gh, gh, gh);
        // Copy local slice into test recv buffer
        for (int i = gh; i < mloc + gh; i++)
            for (int j = gh; j < nloc + gh; j++)
                for (int k = gh; k < oloc + gh; k++)
                {
                    (*cglob_recv)[i][j][k] = cglob[i][j][k];
                }
    }
    else
    {
        // Local slice of data
        auto cloc = cglob.slice(m_start, m_end, n_start, n_end, o_start, o_end);
        cglob_recv = std::move(cloc);
    }

    mgcl::mpi_util::gather(mpi_comm, *cglob_recv);

    MPI_Barrier(mpi_comm);
    if (mpi_rank == 0)
    {
        CAPTURE(cglob_recv->getM(), cglob_recv->getN(), cglob_recv->getO());
        // cglob_recv.dumpToFile("cglob_recv.txt");
        // cglob.dumpToFile("cglob.txt");

        // Check result
        REQUIRE(cglob.isEqual(*cglob_recv));
    }
    MPI_Barrier(mpi_comm);
}

// Checks that gathering is correct while rank 0 has the same globally sized buffer for sending and receiving,
// while all other processes only send local grids.
// In this test, send buffers have different amount of ghost cells than receive buffers.
// Uses mpi_util::gather.
// Run with e.g. mpiexec -n 8 tests_mpi mpi_util::gather-src-dest-same-different-gh
TEST_CASE("mpi_util::gather-src-dest-same-different-gh")
{
    using std::min;

    int mloc = 8;
    int nloc = 8;
    int oloc = 8;
    int ghsend = GENERATE(0, 1);
    int ghrecv = GENERATE(0, 1);
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

    // Calculate global sizes
    int mglob = mloc * mpi_dims[0];
    int nglob = nloc * mpi_dims[1];
    int oglob = oloc * mpi_dims[2];

    /* Initialize start and end for local grid */
    int m_start = mloc * mpi_coords[0] + min(mpi_coords[0], (mglob % mpi_dims[0]));
    int m_end = mloc * (mpi_coords[0] + 1) + min(mpi_coords[0] + 1, (mglob % mpi_dims[0])) - 1;
    int n_start = nloc * mpi_coords[1] + min(mpi_coords[1], (nglob % mpi_dims[1]));
    int n_end = nloc * (mpi_coords[1] + 1) + min(mpi_coords[1] + 1, (nglob % mpi_dims[1])) - 1;
    int o_start = oloc * mpi_coords[2] + min(mpi_coords[2], (oglob % mpi_dims[2]));
    int o_end = oloc * (mpi_coords[2] + 1) + min(mpi_coords[2] + 1, (oglob % mpi_dims[2])) - 1;

    // for (int i = 0; i < mpi_size; i++)
    // {
    //     MPI_Barrier(mpi_comm);
    //     if (i == mpi_rank)
    //     {
    //         std::cout << "rank,ms,me,ns,ne,os,oe: "
    //                   << mpi_rank << ","
    //                   << m_start << "," << m_end << ","
    //                   << n_start << "," << n_end << ","
    //                   << o_start << "," << o_end << std::endl;
    //         std::cout << "coords: " << mpi_coords[0] << "," << mpi_coords[1] << "," << mpi_coords[2] << std::endl;
    //     }
    // }

    // Create test data
    mgcl::Cuboid cglob(mglob, nglob, oglob, ghrecv, ghrecv, ghrecv);
    cglob.fill1dIndex(true);

    std::unique_ptr<mgcl::Cuboid> cglob_recv;
    if (mpi_rank == 0)
    {
        // Send and recv buffer with global size
        cglob_recv = std::make_unique<mgcl::Cuboid>(mglob, nglob, oglob, ghrecv, ghrecv, ghrecv);
        // Copy local slice into test recv buffer
        for (int i = ghrecv; i < mloc + ghrecv; i++)
            for (int j = ghrecv; j < nloc + ghrecv; j++)
                for (int k = ghrecv; k < oloc + ghrecv; k++)
                {
                    (*cglob_recv)[i][j][k] = cglob[i][j][k];
                }
    }
    else
    {
        // Send buffer, local slice
        auto cloc = cglob.slice(m_start, m_end, n_start, n_end, o_start, o_end, ghsend, ghsend, ghsend);
        cglob_recv = std::move(cloc);
    }

    mgcl::mpi_util::gather(mpi_comm, *cglob_recv);

    MPI_Barrier(mpi_comm);
    if (mpi_rank == 0)
    {
        CAPTURE(cglob_recv->getM(), cglob_recv->getN(), cglob_recv->getO());
        // cglob_recv.dumpToFile("cglob_recv.txt");
        // cglob.dumpToFile("cglob.txt");

        // Check result
        REQUIRE(cglob.isEqual(*cglob_recv));
    }
    MPI_Barrier(mpi_comm);
}

// Checks that gathering is correct while rank 0 has the same globally sized buffer for sending and receiving,
// while all other processes only send local grids.
// In this test, send buffers have different amount of ghost cells than receive buffers.
// Uses mpi_util::gather.
// Run with e.g. mpiexec -n 8 tests_mpi mpi_util::gather-GPU-src-dest-same-different-gh
TEST_CASE("mpi_util::gather-GPU-src-dest-same-different-gh")
{
    auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));

    using std::min;

    int mloc = 8;
    int nloc = 8;
    int oloc = 8;
    int ghsend = GENERATE(0, 1);
    int ghrecv = GENERATE(0, 1);
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

    // Calculate global sizes
    int mglob = mloc * mpi_dims[0];
    int nglob = nloc * mpi_dims[1];
    int oglob = oloc * mpi_dims[2];

    /* Initialize start and end for local grid */
    int m_start = mloc * mpi_coords[0] + min(mpi_coords[0], (mglob % mpi_dims[0]));
    int m_end = mloc * (mpi_coords[0] + 1) + min(mpi_coords[0] + 1, (mglob % mpi_dims[0])) - 1;
    int n_start = nloc * mpi_coords[1] + min(mpi_coords[1], (nglob % mpi_dims[1]));
    int n_end = nloc * (mpi_coords[1] + 1) + min(mpi_coords[1] + 1, (nglob % mpi_dims[1])) - 1;
    int o_start = oloc * mpi_coords[2] + min(mpi_coords[2], (oglob % mpi_dims[2]));
    int o_end = oloc * (mpi_coords[2] + 1) + min(mpi_coords[2] + 1, (oglob % mpi_dims[2])) - 1;

    // for (int i = 0; i < mpi_size; i++)
    // {
    //     MPI_Barrier(mpi_comm);
    //     if (i == mpi_rank)
    //     {
    //         std::cout << "rank,ms,me,ns,ne,os,oe: "
    //                   << mpi_rank << ","
    //                   << m_start << "," << m_end << ","
    //                   << n_start << "," << n_end << ","
    //                   << o_start << "," << o_end << std::endl;
    //         std::cout << "coords: " << mpi_coords[0] << "," << mpi_coords[1] << "," << mpi_coords[2] << std::endl;
    //     }
    // }

    // Create test data
    mgcl::Cuboid cglob(mglob, nglob, oglob, ghrecv, ghrecv, ghrecv);
    cglob.fill1dIndex(true);

    std::unique_ptr<mgcl::Cuboid> cglob_recv;
    if (mpi_rank == 0)
    {
        // Send and recv buffer with global size
        cglob_recv = std::make_unique<mgcl::Cuboid>(mglob, nglob, oglob, ghrecv, ghrecv, ghrecv);
        // Copy local slice into test recv buffer
        for (int i = ghrecv; i < mloc + ghrecv; i++)
            for (int j = ghrecv; j < nloc + ghrecv; j++)
                for (int k = ghrecv; k < oloc + ghrecv; k++)
                {
                    (*cglob_recv)[i][j][k] = cglob[i][j][k];
                }
    }
    else
    {
        // Send buffer, local slice
        auto cloc = cglob.slice(m_start, m_end, n_start, n_end, o_start, o_end, ghsend, ghsend, ghsend);
        cglob_recv = std::move(cloc);
    }

    // Create gpu buffer from cglob_recv.
    mgcl_test::TestUtility tu(deviceType);
    mgcl::CuboidGpu d_cglob_recv(tu.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, *cglob_recv);

    mgcl::mpi_util::gather(mpi_comm, tu.getCommands(), d_cglob_recv);

    // Read result into cglob_recv.
    d_cglob_recv.read(tu.getCommands(), cglob_recv.get(), true);

    MPI_Barrier(mpi_comm);
    if (mpi_rank == 0)
    {
        CAPTURE(cglob_recv->getM(), cglob_recv->getN(), cglob_recv->getO());
        // cglob_recv.dumpToFile("cglob_recv.txt");
        // cglob.dumpToFile("cglob.txt");

        // Check result
        REQUIRE(cglob.isEqual(*cglob_recv));
    }
    MPI_Barrier(mpi_comm);
}

// Checks that scattering is correct while rank 0 has the a globally sized buffer for sending and locally sized buffer
// for receiving, while all other processes only receive local grids.
// Uses mpi_util::scatter.
// Run with e.g. mpiexec -n 8 tests_mpi "mpi_util::scatter-src-dest-different"
TEST_CASE("mpi_util::scatter-src-dest-different")
{
    using std::min;

    int mloc = 8;
    int nloc = 8;
    int oloc = 8;
    int gh = 1;
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

    // Calculate global sizes
    int mglob = mloc * mpi_dims[0];
    int nglob = nloc * mpi_dims[1];
    int oglob = oloc * mpi_dims[2];

    /* Initialize start and end for local grid */
    int m_start = mloc * mpi_coords[0] + min(mpi_coords[0], (mglob % mpi_dims[0]));
    int m_end = mloc * (mpi_coords[0] + 1) + min(mpi_coords[0] + 1, (mglob % mpi_dims[0])) - 1;
    int n_start = nloc * mpi_coords[1] + min(mpi_coords[1], (nglob % mpi_dims[1]));
    int n_end = nloc * (mpi_coords[1] + 1) + min(mpi_coords[1] + 1, (nglob % mpi_dims[1])) - 1;
    int o_start = oloc * mpi_coords[2] + min(mpi_coords[2], (oglob % mpi_dims[2]));
    int o_end = oloc * (mpi_coords[2] + 1) + min(mpi_coords[2] + 1, (oglob % mpi_dims[2])) - 1;

    // for (int i = 0; i < mpi_size; i++)
    // {
    //     MPI_Barrier(mpi_comm);
    //     if (i == mpi_rank)
    //     {
    //         std::cout << "rank,ms,me,ns,ne,os,oe: "
    //                   << mpi_rank << ","
    //                   << m_start << "," << m_end << ","
    //                   << n_start << "," << n_end << ","
    //                   << o_start << "," << o_end << std::endl;
    //         std::cout << "coords: " << mpi_coords[0] << "," << mpi_coords[1] << "," << mpi_coords[2] << std::endl;
    //     }
    // }

    // Create test data (used on rank 0)
    mgcl::Cuboid cglob(mglob, nglob, oglob, gh, gh, gh);
    cglob.fill1dIndex(true);

    // Local slice of data, holds expected result
    auto cloc_exp = cglob.slice(m_start, m_end, n_start, n_end, o_start, o_end);

    // Local slice for actual result, reset with 0.
    auto cloc_act = cglob.slice(m_start, m_end, n_start, n_end, o_start, o_end);
    cloc_act->fill(0);

    if (mpi_rank == 0)
        mgcl::mpi_util::scatter(mpi_comm, &cglob, *cloc_act);
    else
        mgcl::mpi_util::scatter(mpi_comm, nullptr, *cloc_act);

    // MPI_Barrier(mpi_comm);
    // if (mpi_rank == 0)
    // {
    //     // CAPTURE(cglob_recv->getM(), cglob_recv->getN(), cglob_recv->getO());
    //     // cglob_recv.dumpToFile("cglob_recv.txt");
    //     // cglob.dumpToFile("cglob.txt");
    // }

    // Check result. On rank 0 the grid must be unchanged (at least the local portion of it).
    // On other processes the local grid must be filled accordingly to global test data.
    if (mpi_rank == 0)
        REQUIRE(cglob.isEqual(cglob));
    else
        REQUIRE(cloc_act->isEqual(*cloc_exp));

    MPI_Barrier(mpi_comm);
}

// Checks that scattering is correct while rank 0 has the same globally sized buffer for sending and receiving,
// while all other processes only receive local grids.
// Uses mpi_util::scatter_inplace.
// Run with e.g. mpiexec -n 8 tests_mpi "mpi_util::scatter-src-dest-same"
TEST_CASE("mpi_util::scatter-src-dest-same")
{
    using std::min;

    int mloc = 8;
    int nloc = 8;
    int oloc = 8;
    int gh = 1;
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

    // Calculate global sizes
    int mglob = mloc * mpi_dims[0];
    int nglob = nloc * mpi_dims[1];
    int oglob = oloc * mpi_dims[2];

    /* Initialize start and end for local grid */
    int m_start = mloc * mpi_coords[0] + min(mpi_coords[0], (mglob % mpi_dims[0]));
    int m_end = mloc * (mpi_coords[0] + 1) + min(mpi_coords[0] + 1, (mglob % mpi_dims[0])) - 1;
    int n_start = nloc * mpi_coords[1] + min(mpi_coords[1], (nglob % mpi_dims[1]));
    int n_end = nloc * (mpi_coords[1] + 1) + min(mpi_coords[1] + 1, (nglob % mpi_dims[1])) - 1;
    int o_start = oloc * mpi_coords[2] + min(mpi_coords[2], (oglob % mpi_dims[2]));
    int o_end = oloc * (mpi_coords[2] + 1) + min(mpi_coords[2] + 1, (oglob % mpi_dims[2])) - 1;

    // for (int i = 0; i < mpi_size; i++)
    // {
    //     MPI_Barrier(mpi_comm);
    //     if (i == mpi_rank)
    //     {
    //         std::cout << "rank,ms,me,ns,ne,os,oe: "
    //                   << mpi_rank << ","
    //                   << m_start << "," << m_end << ","
    //                   << n_start << "," << n_end << ","
    //                   << o_start << "," << o_end << std::endl;
    //         std::cout << "coords: " << mpi_coords[0] << "," << mpi_coords[1] << "," << mpi_coords[2] << std::endl;
    //     }
    // }

    // Create test data (used on rank 0)
    mgcl::Cuboid cglob(mglob, nglob, oglob, gh, gh, gh);
    cglob.fill1dIndex(true);
    auto cglob_exp = mgcl::Cuboid::copyFrom(cglob);

    // Local slice of data, holds expected result
    auto cloc_exp = cglob.slice(m_start, m_end, n_start, n_end, o_start, o_end);

    // Local slice for actual result, reset with 0.
    auto cloc_act = cglob.slice(m_start, m_end, n_start, n_end, o_start, o_end);
    cloc_act->fill(0);

    if (mpi_rank == 0)
        mgcl::mpi_util::scatter_inplace(mpi_comm, cglob);
    else
        mgcl::mpi_util::scatter_inplace(mpi_comm, *cloc_act);

    // MPI_Barrier(mpi_comm);
    // if (mpi_rank == 0)
    // {
    //     // CAPTURE(cglob_recv->getM(), cglob_recv->getN(), cglob_recv->getO());
    //     // cglob_recv.dumpToFile("cglob_recv.txt");
    //     // cglob.dumpToFile("cglob.txt");
    // }

    // Check result. On rank 0 the grid must be unchanged (at least the local portion of it).
    // On other processes the local grid must be filled accordingly to global test data.
    if (mpi_rank == 0)
        REQUIRE(cglob.isEqual(cglob_exp));
    else
        REQUIRE(cloc_act->isEqual(*cloc_exp));

    MPI_Barrier(mpi_comm);
}

// Checks that scattering is correct while rank 0 has the same globally sized buffer for sending and receiving,
// while all other processes only receive local grids. Includes ghosts.
// Uses mpi_util::scatter_inplace_wgh.
// Run with e.g. mpiexec -n 8 tests_mpi "mpi_util::scatter-src-dest-same-with-ghosts"
TEST_CASE("mpi_util::scatter-src-dest-same-with-ghosts")
{
    using std::min;

    int mloc = 8;
    int nloc = 8;
    int oloc = 8;
    int gh = 1;
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

    // Calculate global sizes
    int mglob = mloc * mpi_dims[0];
    int nglob = nloc * mpi_dims[1];
    int oglob = oloc * mpi_dims[2];
    int mglobgh = mglob + 2 * gh;
    int nglobgh = nglob + 2 * gh;
    int oglobgh = oglob + 2 * gh;

    /* Initialize start and end for local grid */
    int m_start = mloc * mpi_coords[0] + min(mpi_coords[0], (mglob % mpi_dims[0]));
    int m_end = mloc * (mpi_coords[0] + 1) + min(mpi_coords[0] + 1, (mglob % mpi_dims[0])) - 1;
    int n_start = nloc * mpi_coords[1] + min(mpi_coords[1], (nglob % mpi_dims[1]));
    int n_end = nloc * (mpi_coords[1] + 1) + min(mpi_coords[1] + 1, (nglob % mpi_dims[1])) - 1;
    int o_start = oloc * mpi_coords[2] + min(mpi_coords[2], (oglob % mpi_dims[2]));
    int o_end = oloc * (mpi_coords[2] + 1) + min(mpi_coords[2] + 1, (oglob % mpi_dims[2])) - 1;

    int ml = (m_end - m_start) + 1;
    int nl = (n_end - n_start) + 1;
    int ol = (o_end - o_start) + 1;
    int mlgh = ml + 2 * gh;
    int nlgh = nl + 2 * gh;
    int olgh = ol + 2 * gh;

    // for (int i = 0; i < mpi_size; i++)
    // {
    //     MPI_Barrier(mpi_comm);
    //     if (i == mpi_rank)
    //     {
    //         std::cout << "rank,ms,me,ns,ne,os,oe: "
    //                   << mpi_rank << ","
    //                   << m_start << "," << m_end << ","
    //                   << n_start << "," << n_end << ","
    //                   << o_start << "," << o_end << std::endl;
    //         std::cout << "coords: " << mpi_coords[0] << "," << mpi_coords[1] << "," << mpi_coords[2] << std::endl;
    //     }
    // }

    // Create test data (used on rank 0)
    mgcl::Cuboid cglob(mglob, nglob, oglob, gh, gh, gh);
    cglob.fill1dIndex(false);
    auto cglob_exp = mgcl::Cuboid::copyFrom(cglob);

    // Local slice of data including ghosts, holds expected result
    mgcl::Cuboid cloc_exp(ml, nl, ol, gh, gh, gh);
    for (int i = 0; i < mlgh; i++)
        for (int j = 0; j < nlgh; j++)
            for (int k = 0; k < olgh; k++)
            {
                cloc_exp[i][j][k] = cglob[i + m_start][j + n_start][k + o_start];
            }

    // Local slice for actual result, reset with 0.
    auto cloc_act = mgcl::Cuboid::copyFrom(cloc_exp);
    cloc_act.fill(0);

    if (mpi_rank == 0)
        mgcl::mpi_util::scatter_inplace_wgh(mpi_comm, cglob);
    else
        mgcl::mpi_util::scatter_inplace_wgh(mpi_comm, cloc_act);

    // MPI_Barrier(mpi_comm);
    // if (mpi_rank == 0)
    // {
    //     // CAPTURE(cglob_recv->getM(), cglob_recv->getN(), cglob_recv->getO());
    //     // cglob_recv.dumpToFile("cglob_recv.txt");
    //     // cglob.dumpToFile("cglob.txt");
    // }

    // cloc_act.dumpToFile(std::to_string(mpi_rank) + "cloc_act.txt");
    // cloc_exp.dumpToFile(std::to_string(mpi_rank) + "cloc_exp.txt");

    // Check result. On rank 0 the grid must be unchanged (at least the local portion of it).
    // On other processes the local grid must be filled accordingly to global test data.
    if (mpi_rank == 0)
        REQUIRE(cglob.isEqualAllCells(cglob_exp));
    else
        REQUIRE(cloc_act.isEqualAllCells(cloc_exp));

    MPI_Barrier(mpi_comm);
}

// Checks that scattering is correct while rank 0 has the same globally sized buffer for sending and receiving,
// while all other processes only receive local grids. Includes ghosts.
// Uses mpi_util::scatter_inplace_wgh.
// Run with e.g. mpiexec -n 8 tests_mpi "mpi_util::scatter-GPU-src-dest-same-with-ghosts"
TEST_CASE("mpi_util::scatter-GPU-src-dest-same-with-ghosts")
{
    auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));

    using std::min;

    int mloc = 8;
    int nloc = 8;
    int oloc = 8;
    int gh = 1;
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

    // Calculate global sizes
    int mglob = mloc * mpi_dims[0];
    int nglob = nloc * mpi_dims[1];
    int oglob = oloc * mpi_dims[2];
    int mglobgh = mglob + 2 * gh;
    int nglobgh = nglob + 2 * gh;
    int oglobgh = oglob + 2 * gh;

    /* Initialize start and end for local grid */
    int m_start = mloc * mpi_coords[0] + min(mpi_coords[0], (mglob % mpi_dims[0]));
    int m_end = mloc * (mpi_coords[0] + 1) + min(mpi_coords[0] + 1, (mglob % mpi_dims[0])) - 1;
    int n_start = nloc * mpi_coords[1] + min(mpi_coords[1], (nglob % mpi_dims[1]));
    int n_end = nloc * (mpi_coords[1] + 1) + min(mpi_coords[1] + 1, (nglob % mpi_dims[1])) - 1;
    int o_start = oloc * mpi_coords[2] + min(mpi_coords[2], (oglob % mpi_dims[2]));
    int o_end = oloc * (mpi_coords[2] + 1) + min(mpi_coords[2] + 1, (oglob % mpi_dims[2])) - 1;

    int ml = (m_end - m_start) + 1;
    int nl = (n_end - n_start) + 1;
    int ol = (o_end - o_start) + 1;
    int mlgh = ml + 2 * gh;
    int nlgh = nl + 2 * gh;
    int olgh = ol + 2 * gh;

    // for (int i = 0; i < mpi_size; i++)
    // {
    //     MPI_Barrier(mpi_comm);
    //     if (i == mpi_rank)
    //     {
    //         std::cout << "rank,ms,me,ns,ne,os,oe: "
    //                   << mpi_rank << ","
    //                   << m_start << "," << m_end << ","
    //                   << n_start << "," << n_end << ","
    //                   << o_start << "," << o_end << std::endl;
    //         std::cout << "coords: " << mpi_coords[0] << "," << mpi_coords[1] << "," << mpi_coords[2] << std::endl;
    //     }
    // }

    // Create test data (used on rank 0)
    mgcl::Cuboid cglob(mglob, nglob, oglob, gh, gh, gh);
    cglob.fill1dIndex(false);
    auto cglob_exp = mgcl::Cuboid::copyFrom(cglob);

    // Local slice of data including ghosts, holds expected result
    mgcl::Cuboid cloc_exp(ml, nl, ol, gh, gh, gh);
    for (int i = 0; i < mlgh; i++)
        for (int j = 0; j < nlgh; j++)
            for (int k = 0; k < olgh; k++)
            {
                cloc_exp[i][j][k] = cglob[i + m_start][j + n_start][k + o_start];
            }

    // Local slice for actual result, reset with 0.
    auto cloc_act = mgcl::Cuboid::copyFrom(cloc_exp);
    cloc_act.fill(0);

    mgcl_test::TestUtility tu(deviceType);

    if (mpi_rank == 0)
    {
        mgcl::CuboidGpu d_c(tu.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, cglob);
        mgcl::mpi_util::scatter_inplace_wgh(mpi_comm, tu.getCommands(), d_c);
        d_c.read(tu.getCommands(), &cglob, true);
    }
    else
    {
        mgcl::CuboidGpu d_c(tu.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, cloc_act);
        mgcl::mpi_util::scatter_inplace_wgh(mpi_comm, tu.getCommands(), d_c);
        d_c.read(tu.getCommands(), &cloc_act, true);
    }

    // MPI_Barrier(mpi_comm);
    // if (mpi_rank == 0)
    // {
    //     // CAPTURE(cglob_recv->getM(), cglob_recv->getN(), cglob_recv->getO());
    //     // cglob_recv.dumpToFile("cglob_recv.txt");
    //     // cglob.dumpToFile("cglob.txt");
    // }

    // cloc_act.dumpToFile(std::to_string(mpi_rank) + "cloc_act.txt");
    // cloc_exp.dumpToFile(std::to_string(mpi_rank) + "cloc_exp.txt");

    // Check result. On rank 0 the grid must be unchanged (at least the local portion of it).
    // On other processes the local grid must be filled accordingly to global test data.
    if (mpi_rank == 0)
        REQUIRE(cglob.isEqualAllCells(cglob_exp));
    else
        REQUIRE(cloc_act.isEqualAllCells(cloc_exp));
}

// Checks that gathering is correct while rank 0 has the same globally sized buffer for sending and receiving,
// while all other processes only send local grids.
// Uses mpi_util::gather with a stencil as argument.
// Run with e.g. mpiexec -n 8 tests_mpi "mpi_util::gather-src-dest-same-stencil"
TEST_CASE("mpi_util::gather-src-dest-same-stencil")
{
    using std::min;

    int mloc = 4;
    int nloc = 8;
    int oloc = 8;
    int gh = 2;
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

    // Calculate global sizes
    int mglob = mloc * mpi_dims[0];
    int nglob = nloc * mpi_dims[1];
    int oglob = oloc * mpi_dims[2];

    /* Initialize start and end for local grid */
    int m_start = mloc * mpi_coords[0] + min(mpi_coords[0], (mglob % mpi_dims[0]));
    int m_end = mloc * (mpi_coords[0] + 1) + min(mpi_coords[0] + 1, (mglob % mpi_dims[0])) - 1;
    int n_start = nloc * mpi_coords[1] + min(mpi_coords[1], (nglob % mpi_dims[1]));
    int n_end = nloc * (mpi_coords[1] + 1) + min(mpi_coords[1] + 1, (nglob % mpi_dims[1])) - 1;
    int o_start = oloc * mpi_coords[2] + min(mpi_coords[2], (oglob % mpi_dims[2]));
    int o_end = oloc * (mpi_coords[2] + 1) + min(mpi_coords[2] + 1, (oglob % mpi_dims[2])) - 1;

    // for (int i = 0; i < mpi_size; i++)
    // {
    //     MPI_Barrier(mpi_comm);
    //     if (i == mpi_rank)
    //     {
    //         std::cout << "rank,ms,me,ns,ne,os,oe: "
    //                   << mpi_rank << ","
    //                   << m_start << "," << m_end << ","
    //                   << n_start << "," << n_end << ","
    //                   << o_start << "," << o_end << std::endl;
    //         std::cout << "coords: " << mpi_coords[0] << "," << mpi_coords[1] << "," << mpi_coords[2] << std::endl;
    //     }
    // }

    // Create test data
    mgcl::VaryingStencil cglob(mglob, nglob, oglob, 3, gh, gh, gh);
    cglob.fill1dIndex(true);

    // Recv buffer, partially filled. Has global size on rank 0, local size else.
    std::unique_ptr<mgcl::VaryingStencil> cglob_recv;
    if (mpi_rank == 0)
    {
        cglob_recv = std::make_unique<mgcl::VaryingStencil>(mglob, nglob, oglob, 3, gh, gh, gh);
        // Copy local slice into test recv buffer
        for (int i = gh; i < mloc + gh; i++)
            for (int j = gh; j < nloc + gh; j++)
                for (int k = gh; k < oloc + gh; k++)
                    for (int ii = 0; ii < 3; ii++)
                        for (int jj = 0; jj < 3; jj++)
                            for (int kk = 0; kk < 3; kk++)
                            {
                                (*cglob_recv)[ii][jj][kk][i][j][k] = cglob[ii][jj][kk][i][j][k];
                            }
    }
    else
    {
        // Local slice of data
        auto cloc = cglob.slice(m_start, m_end, n_start, n_end, o_start, o_end);
        cglob_recv = std::move(cloc);
    }

    mgcl::mpi_util::gather(mpi_comm, *cglob_recv);

    MPI_Barrier(mpi_comm);
    if (mpi_rank == 0)
    {
        // CAPTURE(cglob_recv->getM(), cglob_recv->getN(), cglob_recv->getO());
        // cglob_recv->dumpToFile("cglob_recv.txt");
        // cglob.dumpToFile("cglob.txt");

        // Check result
        REQUIRE(cglob.isEqual(*cglob_recv));
    }
    MPI_Barrier(mpi_comm);
}

// Checks that gathering is correct while rank 0 has the same globally sized buffer for sending and receiving,
// while all other processes only send local grids.
// Uses mpi_util::gather with a stencil as argument.
// Run with e.g. mpiexec -n 8 tests_mpi "mpi_util::gather-GPU-src-dest-same-stencil"
TEST_CASE("mpi_util::gather-GPU-src-dest-same-stencil")
{
    auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));

    using std::min;

    int mloc = 8;
    int nloc = 8;
    int oloc = 8;
    int gh = 2;
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

    // Calculate global sizes
    int mglob = mloc * mpi_dims[0];
    int nglob = nloc * mpi_dims[1];
    int oglob = oloc * mpi_dims[2];

    /* Initialize start and end for local grid */
    int m_start = mloc * mpi_coords[0] + min(mpi_coords[0], (mglob % mpi_dims[0]));
    int m_end = mloc * (mpi_coords[0] + 1) + min(mpi_coords[0] + 1, (mglob % mpi_dims[0])) - 1;
    int n_start = nloc * mpi_coords[1] + min(mpi_coords[1], (nglob % mpi_dims[1]));
    int n_end = nloc * (mpi_coords[1] + 1) + min(mpi_coords[1] + 1, (nglob % mpi_dims[1])) - 1;
    int o_start = oloc * mpi_coords[2] + min(mpi_coords[2], (oglob % mpi_dims[2]));
    int o_end = oloc * (mpi_coords[2] + 1) + min(mpi_coords[2] + 1, (oglob % mpi_dims[2])) - 1;

    // for (int i = 0; i < mpi_size; i++)
    // {
    //     MPI_Barrier(mpi_comm);
    //     if (i == mpi_rank)
    //     {
    //         std::cout << "rank,ms,me,ns,ne,os,oe: "
    //                   << mpi_rank << ","
    //                   << m_start << "," << m_end << ","
    //                   << n_start << "," << n_end << ","
    //                   << o_start << "," << o_end << std::endl;
    //         std::cout << "coords: " << mpi_coords[0] << "," << mpi_coords[1] << "," << mpi_coords[2] << std::endl;
    //     }
    // }

    // Create test data
    mgcl::VaryingStencil cglob(mglob, nglob, oglob, 3, gh, gh, gh);
    cglob.fill1dIndex(true);

    // Recv buffer, partially filled. Has global size on rank 0, local size else.
    std::unique_ptr<mgcl::VaryingStencil> cglob_recv;
    if (mpi_rank == 0)
    {
        cglob_recv = std::make_unique<mgcl::VaryingStencil>(mglob, nglob, oglob, 3, gh, gh, gh);
        // Copy local slice into test recv buffer
        for (int i = gh; i < mloc + gh; i++)
            for (int j = gh; j < nloc + gh; j++)
                for (int k = gh; k < oloc + gh; k++)
                    for (int ii = 0; ii < 3; ii++)
                        for (int jj = 0; jj < 3; jj++)
                            for (int kk = 0; kk < 3; kk++)
                            {
                                (*cglob_recv)[ii][jj][kk][i][j][k] = cglob[ii][jj][kk][i][j][k];
                            }
    }
    else
    {
        // Local slice of data
        auto cloc = cglob.slice(m_start, m_end, n_start, n_end, o_start, o_end);
        cglob_recv = std::move(cloc);
    }

    // Create gpu buffer from cglob_recv.
    mgcl_test::TestUtility tu(deviceType);
    mgcl::VaryingStencilGpu d_cglob_recv(cglob_recv->getM(), cglob_recv->getN(), cglob_recv->getO(),
                                         3, gh, tu.getContext(), tu.getCommands(), tu.getProgram());
    d_cglob_recv.fill(*cglob_recv, tu.getCommands(), true);

    mgcl::mpi_util::gather(mpi_comm, tu.getCommands(), d_cglob_recv);

    // Read result into cglob_recv.
    auto result = d_cglob_recv.read(tu.getCommands(), true);

    if (mpi_rank == 0)
    {
        // CAPTURE(cglob_recv->getM(), cglob_recv->getN(), cglob_recv->getO());
        // cglob_recv->dumpToFile("cglob_recv.txt");
        // cglob.dumpToFile("cglob.txt");

        // Check result
        REQUIRE(cglob.isEqual(result));
    }
}

// Checks that sending border planes of a cuboid is correct.
// Run with e.g. mpiexec -n 8 tests_mpi "mpi_util::sendBorderPlanes_cuboid"
TEST_CASE("mpi_util::sendBorderPlanes_cuboid")
{
    using std::min;

    int m = 8;
    int n = 8;
    int o = 8;
    int ghosts_m = 1;
    int ghosts_n = 2;
    int ghosts_o = 3;
    int mgh = m + 2 * ghosts_m;
    int ngh = n + 2 * ghosts_n;
    int ogh = o + 2 * ghosts_o;
    int periodic = 1; // GENERATE(0,1);

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

    // Calculate global sizes
    int mglob = m * mpi_dims[0];
    int nglob = n * mpi_dims[1];
    int oglob = o * mpi_dims[2];

    // Create dummy problem
    auto v = std::make_shared<mgcl::Cuboid>(m, n, o);
    auto f = std::make_shared<mgcl::Cuboid>(m, n, o);
    mgcl::Problem p(m, n, o, f, v, mglob, nglob, oglob);
    p.setMpiComm(mpi_comm);
    p.init();

    auto& mpiData = p.getLevelAt(0).getMpiData();

    // for (int i = 0; i < mpi_size; i++)
    // {
    //     MPI_Barrier(mpi_comm);
    //     if (i == mpi_rank)
    //     {
    //         std::cerr << i << ":" << std::endl;
    //         std::cerr << "  coords: " << mpi_coords[0] << "," << mpi_coords[1] << "," << mpi_coords[2] << std::endl;
    //         std::cerr << "  front: " << mpiData.front << std::endl;
    //         std::cerr << "   back: " << mpiData.back << std::endl;
    //         std::cerr << "     up: " << mpiData.up << std::endl;
    //         std::cerr << "   down: " << mpiData.down << std::endl;
    //         std::cerr << "   left: " << mpiData.left << std::endl;
    //         std::cerr << "  right: " << mpiData.right << std::endl;
    //     }
    // }

    int yz = ngh * ogh;
    int xz = mgh * ogh;
    int xy = mgh * ngh;
    int ressize = 2 * ghosts_m * yz + 2 * ghosts_n * xz + 2 * ghosts_o * xy;

    // int base_yz_front = 0;
    int base_yz_back = ghosts_m * yz;
    int base_xz_top = 2 * ghosts_m * yz;
    int base_xz_bottom = 2 * ghosts_m * yz + ghosts_n * xz;
    int base_xy_left = 2 * ghosts_m * yz + 2 * ghosts_n * xz;
    int base_xy_right = 2 * ghosts_m * yz + 2 * ghosts_n * xz + ghosts_o * xy;

    std::vector<double> sbuf(ressize);
    std::vector<double> rbuf(ressize);

    mgcl::Cuboid c(m, n, o, ghosts_m, ghosts_n, ghosts_o);
    c.fill1dIndex(false);

    // fill planes with 1d index from cuboid
    for (int i = 0; i < ghosts_m; i++)
        for (int j = 0; j < ngh; j++)
            for (int k = 0; k < ogh; k++)
            {
                // front
                sbuf[i * yz + j * ogh + k] = c[i + ghosts_m][j][k];

                // back
                sbuf[base_yz_back + i * yz + j * ogh + k] = c[i + m][j][k];
            }
    for (int i = 0; i < mgh; i++)
        for (int j = 0; j < ghosts_n; j++)
            for (int k = 0; k < ogh; k++)
            {
                // top
                sbuf[base_xz_top + j * xz + i * ogh + k] = c[i][j + ghosts_n][k];

                // bottom
                sbuf[base_xz_bottom + j * xz + i * ogh + k] = c[i][j + n][k];
            }
    for (int i = 0; i < mgh; i++)
        for (int j = 0; j < ngh; j++)
            for (int k = 0; k < ghosts_o; k++)
            {
                // left
                sbuf[base_xy_left + k * xy + i * ngh + j] = c[i][j][k + ghosts_o];

                // right
                sbuf[base_xy_right + k * xy + i * ngh + j] = c[i][j][k + o];
            }

    for (size_t i = 0; i < rbuf.size(); i++)
    {
        rbuf[i] = -1;
    }

    mgcl::mpi_util::sendBorderPlanes(mgh, ngh, ogh, ghosts_m, ghosts_n, ghosts_o, 1,
                                     sbuf, rbuf, mpiData);

    mgcl::MultigridEngine::updateGhostsSeq(c, nullptr, true, true);

    // Check against cuboid with updated ghosts
    // Edges in send buffers for top and down after sending to front and back
    for (int i = 0; i < ghosts_m; i++)
        for (int j = 0; j < ghosts_n; j++)
            for (int k = ghosts_o; k < ghosts_o + o; k++)
            {
                CAPTURE(i, j, k);
                // top (bottom ghosts)
                REQUIRE(rbuf[base_xz_top + j * xz + i * ogh + k] == c[i][j + ghosts_n + n][k]);

                // bottom (top ghosts)
                REQUIRE(rbuf[base_xz_bottom + j * xz + i * ogh + k] == c[i][j][k]);
            }
    // Toruses in send buffers for left and right after sending to top and bottom
    for (int i = 0; i < mgh; i++)
        for (int j = 0; j < ngh; j++)
            for (int k = 0; k < ghosts_o; k++)
            {
                // left (right ghosts)
                REQUIRE(rbuf[base_xy_left + k * xy + i * ngh + j] == c[i][j][k + ghosts_o + o]);

                // right (left ghosts)
                REQUIRE(rbuf[base_xy_right + k * xy + i * ngh + j] == c[i][j][k]);
            }
}

// Checks that sending border planes of a varying stencil is correct.
// Run with e.g. mpiexec -n 8 tests_mpi "mpi_util::sendBorderPlanes_stencil"
TEST_CASE("mpi_util::sendBorderPlanes_stencil")
{
    using std::min;

    int m = 8;
    int n = 8;
    int o = 8;
    int ghosts_m = 1;
    int ghosts_n = 2;
    int ghosts_o = 3;
    int mgh = m + 2 * ghosts_m;
    int ngh = n + 2 * ghosts_n;
    int ogh = o + 2 * ghosts_o;
    int periodic = 1;     // GENERATE(0,1);
    int stencilWidth = 3; // GENERATE(3, 5);

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

    // Calculate global sizes
    int mglob = m * mpi_dims[0];
    int nglob = n * mpi_dims[1];
    int oglob = o * mpi_dims[2];

    // Create dummy problem
    auto v = std::make_shared<mgcl::Cuboid>(m, n, o);
    auto f = std::make_shared<mgcl::Cuboid>(m, n, o);
    mgcl::Problem p(m, n, o, f, v, mglob, nglob, oglob);
    p.setMpiComm(mpi_comm);
    p.init();

    auto& mpiData = p.getLevelAt(0).getMpiData();

    // for (int i = 0; i < mpi_size; i++)
    // {
    //     MPI_Barrier(mpi_comm);
    //     if (i == mpi_rank)
    //     {
    //         std::cerr << i << ":" << std::endl;
    //         std::cerr << "  coords: " << mpi_coords[0] << "," << mpi_coords[1] << "," << mpi_coords[2] << std::endl;
    //         std::cerr << "  front: " << mpiData.front << std::endl;
    //         std::cerr << "   back: " << mpiData.back << std::endl;
    //         std::cerr << "     up: " << mpiData.up << std::endl;
    //         std::cerr << "   down: " << mpiData.down << std::endl;
    //         std::cerr << "   left: " << mpiData.left << std::endl;
    //         std::cerr << "  right: " << mpiData.right << std::endl;
    //     }
    // }

    // Sizes of planes
    int yz = ngh * ogh;
    int xz = mgh * ogh;
    int xy = mgh * ngh;

    // Size of planes times amount of ghosts in that direction, i.e. number of grid points that are sent in that
    // direction
    int yzgh = yz * ghosts_m;
    int xzgh = xz * ghosts_n;
    int xygh = xy * ghosts_o;

    int stencilSize = stencilWidth * stencilWidth * stencilWidth;
    int ressize = (2 * yzgh + 2 * xzgh + 2 * xygh) * stencilSize;

    // int base_yz_front = 0;
    int base_yz_back = (yzgh)*stencilSize;
    int base_xz_top = (2 * yzgh) * stencilSize;
    int base_xz_bottom = (2 * yzgh + xzgh) * stencilSize;
    int base_xy_left = (2 * yzgh + 2 * xzgh) * stencilSize;
    int base_xy_right = (2 * yzgh + 2 * xzgh + xygh) * stencilSize;

    std::vector<double> sbuf(ressize);
    std::vector<double> rbuf(ressize);

    mgcl::VaryingStencil c(m, n, o, stencilWidth, ghosts_m, ghosts_n, ghosts_o);
    c.fill1dIndex(false);

    // fill planes with 1d index from cuboid
    for (int ii = 0; ii < stencilWidth; ii++)
        for (int jj = 0; jj < stencilWidth; jj++)
            for (int kk = 0; kk < stencilWidth; kk++)
                for (int i = 0; i < ghosts_m; i++)
                    for (int j = 0; j < ngh; j++)
                        for (int k = 0; k < ogh; k++)
                        {
                            // front
                            sbuf[(ii * stencilWidth * stencilWidth + jj * stencilWidth + kk) * yzgh + i * yz + j * ogh + k] = c[ii][jj][kk][i + ghosts_m][j][k];

                            // back
                            sbuf[(ii * stencilWidth * stencilWidth + jj * stencilWidth + kk) * yzgh + base_yz_back + i * yz + j * ogh + k] = c[ii][jj][kk][i + m][j][k];
                        }
    for (int ii = 0; ii < stencilWidth; ii++)
        for (int jj = 0; jj < stencilWidth; jj++)
            for (int kk = 0; kk < stencilWidth; kk++)
                for (int i = 0; i < mgh; i++)
                    for (int j = 0; j < ghosts_n; j++)
                        for (int k = 0; k < ogh; k++)
                        {
                            // top
                            sbuf[(ii * stencilWidth * stencilWidth + jj * stencilWidth + kk) * xzgh + base_xz_top + j * xz + i * ogh + k] = c[ii][jj][kk][i][j + ghosts_n][k];

                            // bottom
                            sbuf[(ii * stencilWidth * stencilWidth + jj * stencilWidth + kk) * xzgh + base_xz_bottom + j * xz + i * ogh + k] = c[ii][jj][kk][i][j + n][k];
                        }
    for (int ii = 0; ii < stencilWidth; ii++)
        for (int jj = 0; jj < stencilWidth; jj++)
            for (int kk = 0; kk < stencilWidth; kk++)
                for (int i = 0; i < mgh; i++)
                    for (int j = 0; j < ngh; j++)
                        for (int k = 0; k < ghosts_o; k++)
                        {
                            // left
                            sbuf[(ii * stencilWidth * stencilWidth + jj * stencilWidth + kk) * xygh + base_xy_left + k * xy + i * ngh + j] = c[ii][jj][kk][i][j][k + ghosts_o];

                            // right
                            sbuf[(ii * stencilWidth * stencilWidth + jj * stencilWidth + kk) * xygh + base_xy_right + k * xy + i * ngh + j] = c[ii][jj][kk][i][j][k + o];
                        }

    for (size_t i = 0; i < rbuf.size(); i++)
    {
        rbuf[i] = -1;
    }

    mgcl::mpi_util::sendBorderPlanes(mgh, ngh, ogh, ghosts_m, ghosts_n, ghosts_o, stencilWidth,
                                     sbuf, rbuf, mpiData);

    c.updateGhosts();

    // Check against cuboid with updated ghosts
    // Edges in send buffers for top and down after sending to front and back
    for (int ii = 0; ii < stencilWidth; ii++)
        for (int jj = 0; jj < stencilWidth; jj++)
            for (int kk = 0; kk < stencilWidth; kk++)
                for (int i = 0; i < ghosts_m; i++)
                    for (int j = 0; j < ghosts_n; j++)
                        for (int k = ghosts_o; k < ghosts_o + o; k++)
                        {
                            CAPTURE(i, j, k);
                            // top (bottom ghosts)
                            REQUIRE(rbuf[(ii * stencilWidth * stencilWidth + jj * stencilWidth + kk) * xzgh + base_xz_top + j * xz + i * ogh + k] == c[ii][jj][kk][i][j + ghosts_n + n][k]);

                            // bottom (top ghosts)
                            REQUIRE(rbuf[(ii * stencilWidth * stencilWidth + jj * stencilWidth + kk) * xzgh + base_xz_bottom + j * xz + i * ogh + k] == c[ii][jj][kk][i][j][k]);
                        }
    // Toruses in send buffers for left and right after sending to top and bottom
    for (int ii = 0; ii < stencilWidth; ii++)
        for (int jj = 0; jj < stencilWidth; jj++)
            for (int kk = 0; kk < stencilWidth; kk++)
                for (int i = 0; i < mgh; i++)
                    for (int j = 0; j < ngh; j++)
                        for (int k = 0; k < ghosts_o; k++)
                        {
                            // left (right ghosts)
                            REQUIRE(rbuf[(ii * stencilWidth * stencilWidth + jj * stencilWidth + kk) * xygh + base_xy_left + k * xy + i * ngh + j] == c[ii][jj][kk][i][j][k + ghosts_o + o]);

                            // right (left ghosts)
                            REQUIRE(rbuf[(ii * stencilWidth * stencilWidth + jj * stencilWidth + kk) * xygh + base_xy_right + k * xy + i * ngh + j] == c[ii][jj][kk][i][j][k]);
                        }
}

// Checks that sending border planes of a CuboidBS is correct.
// Run with e.g. mpiexec -n 8 tests_mpi "mpi_util::sendBorderPlanes_cuboidbs_blockstencil"
TEST_CASE("mpi_util::sendBorderPlanes_cuboidbs")
{
    using std::min;

    int m = 8;
    int n = 8;
    int o = 8;
    int ghosts_m = 1;
    int ghosts_n = 2;
    int ghosts_o = 3;
    int mgh = m + 2 * ghosts_m;
    int ngh = n + 2 * ghosts_n;
    int ogh = o + 2 * ghosts_o;
    int blocksize = 2;
    int periodic = 1; // GENERATE(0,1);

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

    // Calculate global sizes
    int mglob = m * mpi_dims[0];
    int nglob = n * mpi_dims[1];
    int oglob = o * mpi_dims[2];

    // Create dummy problem
    auto v = std::make_shared<mgcl::Cuboid>(m, n, o);
    auto f = std::make_shared<mgcl::Cuboid>(m, n, o);
    mgcl::Problem p(m, n, o, f, v, mglob, nglob, oglob);
    p.setMpiComm(mpi_comm);
    p.init();

    auto& mpiData = p.getLevelAt(0).getMpiData();

    // for (int i = 0; i < mpi_size; i++)
    // {
    //     MPI_Barrier(mpi_comm);
    //     if (i == mpi_rank)
    //     {
    //         std::cerr << i << ":" << std::endl;
    //         std::cerr << "  coords: " << mpi_coords[0] << "," << mpi_coords[1] << "," << mpi_coords[2] << std::endl;
    //         std::cerr << "  front: " << mpiData.front << std::endl;
    //         std::cerr << "   back: " << mpiData.back << std::endl;
    //         std::cerr << "     up: " << mpiData.up << std::endl;
    //         std::cerr << "   down: " << mpiData.down << std::endl;
    //         std::cerr << "   left: " << mpiData.left << std::endl;
    //         std::cerr << "  right: " << mpiData.right << std::endl;
    //     }
    // }

    int yz = ngh * ogh;
    int xz = mgh * ogh;
    int xy = mgh * ngh;
    int ressize = (2 * ghosts_m * yz + 2 * ghosts_n * xz + 2 * ghosts_o * xy) * blocksize;

    // int base_yz_front = 0;
    int base_yz_back = ghosts_m * yz * blocksize;
    int base_xz_top = 2 * ghosts_m * yz * blocksize;
    int base_xz_bottom = (2 * ghosts_m * yz + ghosts_n * xz) * blocksize;
    int base_xy_left = (2 * ghosts_m * yz + 2 * ghosts_n * xz) * blocksize;
    int base_xy_right = (2 * ghosts_m * yz + 2 * ghosts_n * xz + ghosts_o * xy) * blocksize;

    std::vector<double> sbuf(ressize);
    std::vector<double> rbuf(ressize);

    mgcl::CuboidBS c(m, n, o, ghosts_m, ghosts_n, ghosts_o, blocksize);
    c.fill1dIndex(false);

    // fill planes with 1d index from cuboid
    for (int i = 0; i < ghosts_m; i++)
        for (int j = 0; j < ngh; j++)
            for (int k = 0; k < ogh; k++)
                for (int b = 0; b < blocksize; b++)
                {
                    // front
                    sbuf[(i * yz + j * ogh + k) * blocksize + b] = c[i + ghosts_m][j][k][b];

                    // back
                    sbuf[base_yz_back + (i * yz + j * ogh + k) * blocksize + b] = c[i + m][j][k][b];
                }
    for (int i = 0; i < mgh; i++)
        for (int j = 0; j < ghosts_n; j++)
            for (int k = 0; k < ogh; k++)
                for (int b = 0; b < blocksize; b++)
                {
                    // top
                    sbuf[base_xz_top + (j * xz + i * ogh + k) * blocksize + b] = c[i][j + ghosts_n][k][b];

                    // bottom
                    sbuf[base_xz_bottom + (j * xz + i * ogh + k) * blocksize + b] = c[i][j + n][k][b];
                }
    for (int i = 0; i < mgh; i++)
        for (int j = 0; j < ngh; j++)
            for (int k = 0; k < ghosts_o; k++)
                for (int b = 0; b < blocksize; b++)
                {
                    // left
                    sbuf[base_xy_left + (k * xy + i * ngh + j) * blocksize + b] = c[i][j][k + ghosts_o][b];

                    // right
                    sbuf[base_xy_right + (k * xy + i * ngh + j) * blocksize + b] = c[i][j][k + o][b];
                }

    for (size_t i = 0; i < rbuf.size(); i++)
    {
        rbuf[i] = -1;
    }

    mgcl::mpi_util::sendBorderPlanesCuboidBS(mgh, ngh, ogh, ghosts_m, ghosts_n, ghosts_o, blocksize,
                                             sbuf, rbuf, mpiData);

    c.updateGhosts(nullptr, true);

    // Check against cuboid with updated ghosts
    // Edges in send buffers for top and down after sending to front and back
    for (int i = 0; i < ghosts_m; i++)
        for (int j = 0; j < ghosts_n; j++)
            for (int k = ghosts_o; k < ghosts_o + o; k++)
                for (int b = 0; b < blocksize; b++)
                {
                    CAPTURE(i, j, k);
                    // top (bottom ghosts)
                    REQUIRE(rbuf[base_xz_top + (j * xz + i * ogh + k) * blocksize + b] == c[i][j + ghosts_n + n][k][b]);

                    // bottom (top ghosts)
                    REQUIRE(rbuf[base_xz_bottom + (j * xz + i * ogh + k) * blocksize + b] == c[i][j][k][b]);
                }
    // Toruses in send buffers for left and right after sending to top and bottom
    for (int i = 0; i < mgh; i++)
        for (int j = 0; j < ngh; j++)
            for (int k = 0; k < ghosts_o; k++)
                for (int b = 0; b < blocksize; b++)
                {
                    // left (right ghosts)
                    REQUIRE(rbuf[base_xy_left + (k * xy + i * ngh + j) * blocksize + b] == c[i][j][k + ghosts_o + o][b]);

                    // right (left ghosts)
                    REQUIRE(rbuf[base_xy_right + (k * xy + i * ngh + j) * blocksize + b] == c[i][j][k][b]);
                }
}

// Checks that sending border planes of a Blockstencil is correct.
// Run with e.g. mpiexec -n 8 tests_mpi "mpi_util::sendBorderPlanes_blockstencil"
TEST_CASE("mpi_util::sendBorderPlanes_blockstencil")
{
    using std::min;

    int m = 8;
    int n = 8;
    int o = 8;
    int ghosts_m = 1;
    int ghosts_n = 2;
    int ghosts_o = 3;
    int mgh = m + 2 * ghosts_m;
    int ngh = n + 2 * ghosts_n;
    int ogh = o + 2 * ghosts_o;
    int stencilWidth = 3;
    int stencilSize = stencilWidth * stencilWidth * stencilWidth;
    int blocksize = 2;
    int blocksize2 = blocksize * blocksize;
    int periodic = 1; // GENERATE(0,1);

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

    // Calculate global sizes
    int mglob = m * mpi_dims[0];
    int nglob = n * mpi_dims[1];
    int oglob = o * mpi_dims[2];

    // Create dummy problem
    auto v = std::make_shared<mgcl::Cuboid>(m, n, o);
    auto f = std::make_shared<mgcl::Cuboid>(m, n, o);
    mgcl::Problem p(m, n, o, f, v, mglob, nglob, oglob);
    p.setMpiComm(mpi_comm);
    p.init();

    auto& mpiData = p.getLevelAt(0).getMpiData();

    // for (int i = 0; i < mpi_size; i++)
    // {
    //     MPI_Barrier(mpi_comm);
    //     if (i == mpi_rank)
    //     {
    //         std::cerr << i << ":" << std::endl;
    //         std::cerr << "  coords: " << mpi_coords[0] << "," << mpi_coords[1] << "," << mpi_coords[2] << std::endl;
    //         std::cerr << "  front: " << mpiData.front << std::endl;
    //         std::cerr << "   back: " << mpiData.back << std::endl;
    //         std::cerr << "     up: " << mpiData.up << std::endl;
    //         std::cerr << "   down: " << mpiData.down << std::endl;
    //         std::cerr << "   left: " << mpiData.left << std::endl;
    //         std::cerr << "  right: " << mpiData.right << std::endl;
    //     }
    // }

    int yz = ngh * ogh;
    int xz = mgh * ogh;
    int xy = mgh * ngh;
    int yzgh = yz * ghosts_m;
    int xzgh = xz * ghosts_n;
    int xygh = xy * ghosts_o;
    int ressize = (2 * ghosts_m * yz + 2 * ghosts_n * xz + 2 * ghosts_o * xy) * blocksize2 * stencilSize;

    // int base_yz_front = 0;
    int base_yz_back = ghosts_m * yz * blocksize2 * stencilSize;
    int base_xz_top = 2 * ghosts_m * yz * blocksize2 * stencilSize;
    int base_xz_bottom = (2 * ghosts_m * yz + ghosts_n * xz) * blocksize2 * stencilSize;
    int base_xy_left = (2 * ghosts_m * yz + 2 * ghosts_n * xz) * blocksize2 * stencilSize;
    int base_xy_right = (2 * ghosts_m * yz + 2 * ghosts_n * xz + ghosts_o * xy) * blocksize2 * stencilSize;

    // size of all planes for all coeffs, i.e. content of one matrix entry
    int ressizeOneCoeff = (2 * yz * ghosts_m + 2 * xz * ghosts_n + 2 * xy * ghosts_o);
    int ressizeOneMatrixEntry = (2 * yz * ghosts_m + 2 * xz * ghosts_n + 2 * xy * ghosts_o) * stencilSize;
    int yzRessizeOneMatrixEntry = yzgh * 27;
    int xzRessizeOneMatrixEntry = xzgh * 27;
    int xyRessizeOneMatrixEntry = xygh * 27;

    std::vector<double> sbuf(ressize);
    std::vector<double> rbuf(ressize);

    mgcl::Blockstencil c(m, n, o, stencilWidth, blocksize, ghosts_m, ghosts_n, ghosts_o);
    c.fill1dIndex(false);

    // fill sbuf with -1
    for (size_t i = 0; i < sbuf.size(); i++)
    {
        sbuf[i] = -1;
    }

    auto yz1dindex = [&blocksize, &yzRessizeOneMatrixEntry, &stencilWidth, &yzgh, &yz, &ogh](int bi, int bj, int ii, int jj, int kk, int i, int k, int j)
    {
        return (bi * blocksize + bj) * yzRessizeOneMatrixEntry + (ii * stencilWidth * stencilWidth + jj * stencilWidth + kk) * yzgh + (i * yz + j * ogh + k);
    };
    auto xz1dindex = [&blocksize, &xzRessizeOneMatrixEntry, &stencilWidth, &xzgh, &xz, &ogh](int bi, int bj, int ii, int jj, int kk, int i, int k, int j)
    {
        return (bi * blocksize + bj) * xzRessizeOneMatrixEntry + (ii * stencilWidth * stencilWidth + jj * stencilWidth + kk) * xzgh + (j * xz + i * ogh + k);
    };
    auto xy1dindex = [&blocksize, &xyRessizeOneMatrixEntry, &stencilWidth, &xygh, &xy, &ngh](int bi, int bj, int ii, int jj, int kk, int i, int k, int j)
    {
        return (bi * blocksize + bj) * xyRessizeOneMatrixEntry + (ii * stencilWidth * stencilWidth + jj * stencilWidth + kk) * xygh + (k * xy + i * ngh + j);
    };

    // fill planes with 1d index from blockstencil
    for (int bi = 0; bi < blocksize; bi++)
        for (int bj = 0; bj < blocksize; bj++)
            for (int ii = 0; ii < stencilWidth; ii++)
                for (int jj = 0; jj < stencilWidth; jj++)
                    for (int kk = 0; kk < stencilWidth; kk++)
                    {
                        for (int i = 0; i < ghosts_m; i++)
                            for (int j = 0; j < ngh; j++)
                                for (int k = 0; k < ogh; k++)
                                {
                                    // front
                                    // sbuf[(bi * blocksize + bj) * ressizeOneMatrixEntry + (ii * stencilWidth * stencilWidth + jj * stencilWidth + kk) * ressizeOneCoeff + (i * yz + j * ogh + k)] = c[bi][bj][ii][jj][kk][i + ghosts_m][j][k];
                                    sbuf[yz1dindex(bi, bj, ii, jj, kk, i, k, j)] = c[bi][bj][ii][jj][kk][i + ghosts_m][j][k];

                                    // back
                                    sbuf[base_yz_back + yz1dindex(bi, bj, ii, jj, kk, i, k, j)] = c[bi][bj][ii][jj][kk][i + m][j][k];
                                }
                        for (int i = 0; i < mgh; i++)
                            for (int j = 0; j < ghosts_n; j++)
                                for (int k = 0; k < ogh; k++)
                                {
                                    // top
                                    sbuf[base_xz_top + xz1dindex(bi, bj, ii, jj, kk, i, k, j)] = c[bi][bj][ii][jj][kk][i][j + ghosts_n][k];

                                    // bottom
                                    sbuf[base_xz_bottom + xz1dindex(bi, bj, ii, jj, kk, i, k, j)] = c[bi][bj][ii][jj][kk][i][j + n][k];
                                }
                        for (int i = 0; i < mgh; i++)
                            for (int j = 0; j < ngh; j++)
                                for (int k = 0; k < ghosts_o; k++)
                                {
                                    // left
                                    sbuf[base_xy_left + xy1dindex(bi, bj, ii, jj, kk, i, k, j)] = c[bi][bj][ii][jj][kk][i][j][k + ghosts_o];

                                    // right
                                    sbuf[base_xy_right + xy1dindex(bi, bj, ii, jj, kk, i, k, j)] = c[bi][bj][ii][jj][kk][i][j][k + o];
                                }
                    }

    // fill rbuf with -1 and make sure, sbuf is completely filled
    for (size_t i = 0; i < rbuf.size(); i++)
    {
        rbuf[i] = -1;
        REQUIRE(sbuf[i] != -1);
    }

    mgcl::mpi_util::sendBorderPlanesBlockstencil(mgh, ngh, ogh, ghosts_m, ghosts_n, ghosts_o, stencilWidth, blocksize,
                                                 sbuf, rbuf, mpiData);

    c.updateGhostsLocally();

    // Check against cuboid with updated ghosts
    // Edges in send buffers for top and down after sending to front and back
    for (int bi = 0; bi < blocksize; bi++)
        for (int bj = 0; bj < blocksize; bj++)
            for (int ii = 0; ii < stencilWidth; ii++)
                for (int jj = 0; jj < stencilWidth; jj++)
                    for (int kk = 0; kk < stencilWidth; kk++)
                    {
                        for (int i = 0; i < ghosts_m; i++)
                            for (int j = 0; j < ghosts_n; j++)
                                for (int k = ghosts_o; k < ghosts_o + o; k++)
                                {
                                    CAPTURE(i, j, k, ii, jj, kk, bi, bj, mpi_rank);
                                    // top (bottom ghosts)
                                    REQUIRE(rbuf[base_xz_top + xz1dindex(bi, bj, ii, jj, kk, i, k, j)] == c[bi][bj][ii][jj][kk][i][j + ghosts_n + n][k]);

                                    // bottom (top ghosts)
                                    REQUIRE(rbuf[base_xz_bottom + xz1dindex(bi, bj, ii, jj, kk, i, k, j)] == c[bi][bj][ii][jj][kk][i][j][k]);
                                }
                        // Toruses in send buffers for left and right after sending to top and bottom
                        for (int i = 0; i < mgh; i++)
                            for (int j = 0; j < ngh; j++)
                                for (int k = 0; k < ghosts_o; k++)
                                {
                                    // left (right ghosts)
                                    REQUIRE(rbuf[base_xy_left + xy1dindex(bi, bj, ii, jj, kk, i, k, j)] == c[bi][bj][ii][jj][kk][i][j][k + ghosts_o + o]);

                                    // right (left ghosts)
                                    REQUIRE(rbuf[base_xy_right + xy1dindex(bi, bj, ii, jj, kk, i, k, j)] == c[bi][bj][ii][jj][kk][i][j][k]);
                                }
                    }
}