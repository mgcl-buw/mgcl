#include "mpi_util.hpp"

#include "mpi_data.hpp"

#include <iostream>

namespace mgcl::mpi_util
{
    /**
     * @brief Gathers all local grids into one big grid. Must be called from each process. On rank 0 c must have the
     *   size of the global grid, all other processes just need to send their local grid.
     *
     * @param comm
     * @param c Global grid for root process (rank 0), local grids for all other processes.
     */
    void gather(MPI_Comm comm, Cuboid &c)
    {
        int err;
        int rank;
        err = MPI_Comm_rank(comm, &rank);
        mgclCheckMpiError(comm, err, "MPI_Comm_rank");

        int mpi_size;
        err = MPI_Comm_size(comm, &mpi_size);
        mgclCheckMpiError(comm, err, "MPI_Comm_size");

        if (mpi_size <= 1)
            throw "Gather is only supported for at least 2 processes.";

        // Get local grid size from another process
        MPI_Status stats[2];
        MPI_Request reqs[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
        int locsizes[6] = {c.getM(), c.getN(), c.getO(), c.getGhostsM(), c.getGhostsN(), c.getGhostsO()};
        if (rank == 1)
        {
            err = MPI_Isend(locsizes, 6, MPI_INT, 0, 0, comm, &reqs[0]);
            mgclCheckMpiError(comm, err, "MPI_Isend");
        }
        else if (rank == 0)
        {
            err = MPI_Irecv(locsizes, 6, MPI_INT, 1, 0, comm, &reqs[1]);
            mgclCheckMpiError(comm, err, "MPI_Irecv");
        }
        MPI_Waitall(2, reqs, stats);

        int mloc = locsizes[0];
        int nloc = locsizes[1];
        int oloc = locsizes[2];
        int mlocgh = mloc + 2 * locsizes[3];
        int nlocgh = nloc + 2 * locsizes[4];
        int olocgh = oloc + 2 * locsizes[5];
        int mglobgh = c.getMgh();
        int nglobgh = c.getNgh();
        int oglobgh = c.getOgh();

        // Create subarray type for the send buffer
        MPI_Datatype subarraySend;
        if (rank != 0)
        {
            int sizes[3] = {mlocgh, nlocgh, olocgh};
            int subsizes[3] = {mloc, nloc, oloc};
            int starts[3] = {locsizes[3], locsizes[4], locsizes[5]};
            err = MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &subarraySend);
            mgclCheckMpiError(comm, err, "MPI_Type_create_subarray");
            err = MPI_Type_commit(&subarraySend);
            mgclCheckMpiError(comm, err, "MPI_Type_commit");
        }

        // Create subarray type for the receive buffer
        MPI_Datatype subarrayRecv;
        MPI_Datatype subarrayRecvResized;
        if (rank == 0)
        {
            int sizes[3] = {mglobgh, nglobgh, oglobgh};
            int subsizes[3] = {mloc, nloc, oloc};
            int starts[3] = {c.getGhostsM(), c.getGhostsN(), c.getGhostsO()};
            // int starts[3] = {0, 0, 0};
            err = MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &subarrayRecv);
            mgclCheckMpiError(comm, err, "MPI_Type_create_subarray");
            err = MPI_Type_commit(&subarrayRecv);
            mgclCheckMpiError(comm, err, "MPI_Type_commit");

            // Resize recv data to avoid overlapping resulting from ghosts. Enabling explicitely stating start and extent
            // later in MPI_Gatherv (counts and displ). One unit is the size of a local grid.
            err = MPI_Type_create_resized(subarrayRecv, 0, 1 * sizeof(double), &subarrayRecvResized);
            // err = MPI_Type_create_resized(subarrayRecv, 0, oglobgh * sizeof(double), &subarrayRecvResized);
            mgclCheckMpiError(comm, err, "MPI_Type_create_resized");
            err = MPI_Type_commit(&subarrayRecvResized);
            mgclCheckMpiError(comm, err, "MPI_Type_commit");
        }

        int counts[mpi_size];
        int displ[mpi_size];
        if (rank == 0)
            for (int i = 0; i < mpi_size; i++)
            {
                int coords[3] = {0, 0, 0};
                err = MPI_Cart_coords(comm, i, 3, coords);
                mgclCheckMpiError(comm, err, "MPI_Cart_coords");

                counts[i] = 1;
                displ[i] = coords[0] * mloc * nglobgh * oglobgh + coords[1] * nloc * oglobgh + coords[2] * oloc;
            }

        if (rank == 0)
            MPI_Gatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                        c.field1d().data(), counts, displ, subarrayRecvResized, 0, comm);
        else
            MPI_Gatherv(c.field1d().data(), 1, subarraySend,
                        nullptr, nullptr, nullptr, MPI_DATATYPE_NULL, 0, comm);
        mgclCheckMpiError(comm, err, "MPI_Gather");

        if (rank != 0)
            mgclCheckMpiError(comm, MPI_Type_free(&subarraySend), "MPI_Type_free");
        if (rank == 0)
        {
            mgclCheckMpiError(comm, MPI_Type_free(&subarrayRecv), "MPI_Type_free");
            mgclCheckMpiError(comm, MPI_Type_free(&subarrayRecvResized), "MPI_Type_free");
        }
    }

    /**
     * @brief Checks the return code of a MPI call and prints it if not MPI_SUCCESS.
     */
    void mgcl_check_mpi_error(MPI_Comm comm, int err, const char *operation, const char *filename, int line)
    {
        if (err != MPI_SUCCESS)
        {
            fprintf(stderr, "Error during operation '%s', ", operation);
            fprintf(stderr, "in '%s' on line %d\n", filename, line);
            fprintf(stderr, "Error code was %d\n", err);
            MPI_Abort(comm, err);
        }
    }

} // namespace mpi_util
