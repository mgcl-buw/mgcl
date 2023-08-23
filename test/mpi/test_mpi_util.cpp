#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <iostream>
#include <memory>

#include "../../src/cuboid.hpp"
#include "../../src/mpi_data.hpp"
#include "../../src/mpi_util.hpp"
#include "../../src/problem.hpp"
// #include "test_utility.hpp"

#include "mpi.h"

// Checks that gathering is correct (for usage in vcycle).
TEST_CASE("MPI_Gatherv")
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
    int cnt = 0;
    for (int i = gh; i < mglob + gh; i++)
        for (int j = gh; j < nglob + gh; j++)
            for (int k = gh; k < oglob + gh; k++)
            {
                cglob[i][j][k] = cnt++;
            }

    auto cloc = cglob.slice(m_start, m_end, n_start, n_end, o_start, o_end);

    // Gather
    mgcl::Cuboid cglob_recv(mglob, nglob, oglob, gh, gh, gh);

    // // Create subarray type for the send buffer
    // MPI_Datatype subarraySend;
    // {
    //     int sizes[3] = {mlocgh, nlocgh, olocgh};
    //     int subsizes[3] = {mloc, nloc, oloc};
    //     int starts[3] = {gh, gh, gh};
    //     err = MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &subarraySend);
    //     mgcl::mgclCheckMpiError(mpi_comm, err, "MPI_Type_create_subarray");
    //     err = MPI_Type_commit(&subarraySend);
    //     mgcl::mgclCheckMpiError(mpi_comm, err, "MPI_Type_commit");
    // }

    // // Create subarray type for the receive buffer
    // MPI_Datatype subarrayRecv;
    // MPI_Datatype subarrayRecvResized;
    // {
    //     int sizes[3] = {mglobgh, nglobgh, oglobgh};
    //     int subsizes[3] = {mloc, nloc, oloc};
    //     int starts[3] = {gh, gh, gh};
    //     // int starts[3] = {0, 0, 0};
    //     err = MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &subarrayRecv);
    //     mgcl::mgclCheckMpiError(mpi_comm, err, "MPI_Type_create_subarray");
    //     err = MPI_Type_commit(&subarrayRecv);
    //     mgcl::mgclCheckMpiError(mpi_comm, err, "MPI_Type_commit");

    //     // Resize recv data to avoid overlapping resulting from ghosts. Enabling explicitely stating start and extent
    //     // later in MPI_Gatherv (counts and displ). One unit is the size of a local grid.
    //     err = MPI_Type_create_resized(subarrayRecv, 0, 1 * sizeof(double), &subarrayRecvResized);
    //     // err = MPI_Type_create_resized(subarrayRecv, 0, oglobgh * sizeof(double), &subarrayRecvResized);
    //     mgcl::mgclCheckMpiError(mpi_comm, err, "MPI_Type_create_resized");
    //     err = MPI_Type_commit(&subarrayRecvResized);
    //     mgcl::mgclCheckMpiError(mpi_comm, err, "MPI_Type_commit");
    // }

    // MPI_Barrier(mpi_comm);

    // // Calculate displacements for each processor.
    // int counts[mpi_size];
    // int displ[mpi_size];
    // for (int i = 0; i < mpi_size; i++)
    // {
    //     int coords[3] = {0, 0, 0};
    //     err = MPI_Cart_coords(mpi_comm, i, 3, coords);
    //     mgcl::mgclCheckMpiError(mpi_comm, err, "MPI_Cart_coords");

    //     counts[i] = 1;
    //     displ[i] = coords[0] * mloc * nglobgh * oglobgh + coords[1] * nloc * oglobgh + coords[2] * oloc;
    // }
    // MPI_Barrier(mpi_comm);

    // // int counts[4] = {1, 1, 1, 1};
    // // int displ[4] = {0, nloc, mloc * nglobgh, mloc * nglobgh + nloc};
    // // int counts[2] = {1, 1};
    // // int displ[2] = {0, mloc * nglobgh * oglobgh};

    // // Alternative with
    // // int starts[3] = {0, 0, 0}; and
    // // MPI_Type_create_resized(subarrayRecv, 0, 1 * sizeof(double), &subarrayRecvResized);
    // //
    // // int counts[4] = {1, 1, 1, 1};
    // // int displ[4] = {nglobgh * oglobgh + oglobgh + gh,
    // //                 nglobgh * oglobgh + (nloc + gh) * oglobgh + gh,
    // //                 (mloc + gh) * nglobgh * oglobgh + oglobgh + gh,
    // //                 (mloc + gh) * nglobgh * oglobgh + (nloc + gh) * oglobgh + gh};
    // // int counts[2] = {1, 1};
    // // int displ[2] = {nglobgh * oglobgh + oglobgh + gh, (mloc + gh) * nglobgh * oglobgh + oglobgh + gh};
    // // CAPTURE(displ);

    // // MPI_Gatherv(cloc->field1d().data(), 1, subarraySend,
    // //             cglob_recv.field1d().data(), counts, displ, subarrayRecvResized, 0, mpi_comm);
    // // err = MPI_Gather(cloc->field1d().data(), 1, subarraySend,
    // //                  cglob_recv.field1d().data(), 1, subarrayRecv, 0, mpi_comm);
    // // mgcl::mgclCheckMpiError(mpi_comm, err, "MPI_Gather");

    // mgcl::mgclCheckMpiError(mpi_comm, MPI_Type_free(&subarraySend), "MPI_Type_free");
    // mgcl::mgclCheckMpiError(mpi_comm, MPI_Type_free(&subarrayRecv), "MPI_Type_free");
    // mgcl::mgclCheckMpiError(mpi_comm, MPI_Type_free(&subarrayRecvResized), "MPI_Type_free");

    mgcl::mpi_util::gather(mpi_comm, *cloc, &cglob_recv);

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
