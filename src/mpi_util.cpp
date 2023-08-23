#include "mpi_util.hpp"

#include "mpi_data.hpp"

namespace mgcl::mpi_util
{
    /**
     * @brief Gathers all local grids (send) into one big grid (recv). Must be called from each process. recv_ptr is
     * ignored on each process that is not the root process (i.e. does not have rank 0) and thus may be null.
     *
     * @param comm
     * @param send Send buffer, local to each processor (incl. root).
     * @param recv Receive buffer, must have the correct size (multiple of send buffer).
     */
    void gather(MPI_Comm comm, Cuboid &send, Cuboid *recv_ptr)
    {
        int err;
        int rank;
        err = MPI_Comm_rank(comm, &rank);
        mgcl::mgclCheckMpiError(comm, err, "MPI_Comm_rank");

        if (rank == 0 && recv_ptr == nullptr)
            throw "recv_ptr must not be null on root process!";

        int mloc = send.getM();
        int nloc = send.getN();
        int oloc = send.getO();
        int mlocgh = send.getMgh();
        int nlocgh = send.getNgh();
        int olocgh = send.getOgh();
        int mglob = recv_ptr->getM();
        int nglob = recv_ptr->getN();
        int oglob = recv_ptr->getO();
        int mglobgh = recv_ptr->getMgh();
        int nglobgh = recv_ptr->getNgh();
        int oglobgh = recv_ptr->getOgh();

        // Create subarray type for the send buffer
        MPI_Datatype subarraySend;
        {
            int sizes[3] = {mlocgh, nlocgh, olocgh};
            int subsizes[3] = {mloc, nloc, oloc};
            int starts[3] = {send.getGhostsM(), send.getGhostsN(), send.getGhostsO()};
            err = MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &subarraySend);
            mgcl::mgclCheckMpiError(comm, err, "MPI_Type_create_subarray");
            err = MPI_Type_commit(&subarraySend);
            mgcl::mgclCheckMpiError(comm, err, "MPI_Type_commit");
        }

        // Create subarray type for the receive buffer
        MPI_Datatype subarrayRecv;
        MPI_Datatype subarrayRecvResized;
        {
            int sizes[3] = {mglobgh, nglobgh, oglobgh};
            int subsizes[3] = {mloc, nloc, oloc};
            int starts[3] = {recv_ptr->getGhostsM(), recv_ptr->getGhostsN(), recv_ptr->getGhostsO()};
            // int starts[3] = {0, 0, 0};
            err = MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &subarrayRecv);
            mgcl::mgclCheckMpiError(comm, err, "MPI_Type_create_subarray");
            err = MPI_Type_commit(&subarrayRecv);
            mgcl::mgclCheckMpiError(comm, err, "MPI_Type_commit");

            // Resize recv data to avoid overlapping resulting from ghosts. Enabling explicitely stating start and extent
            // later in MPI_Gatherv (counts and displ). One unit is the size of a local grid.
            err = MPI_Type_create_resized(subarrayRecv, 0, 1 * sizeof(double), &subarrayRecvResized);
            // err = MPI_Type_create_resized(subarrayRecv, 0, oglobgh * sizeof(double), &subarrayRecvResized);
            mgcl::mgclCheckMpiError(comm, err, "MPI_Type_create_resized");
            err = MPI_Type_commit(&subarrayRecvResized);
            mgcl::mgclCheckMpiError(comm, err, "MPI_Type_commit");
        }

        // Calculate displacements for each processor.
        int mpi_size;
        err = MPI_Comm_size(comm, &mpi_size);
        mgcl::mgclCheckMpiError(comm, err, "MPI_Comm_size");

        int counts[mpi_size];
        int displ[mpi_size];
        for (int i = 0; i < mpi_size; i++)
        {
            int coords[3] = {0, 0, 0};
            err = MPI_Cart_coords(comm, i, 3, coords);
            mgcl::mgclCheckMpiError(comm, err, "MPI_Cart_coords");

            counts[i] = 1;
            displ[i] = coords[0] * mloc * nglobgh * oglobgh + coords[1] * nloc * oglobgh + coords[2] * oloc;
        }

        MPI_Gatherv(send.field1d().data(), 1, subarraySend,
                    recv_ptr->field1d().data(), counts, displ, subarrayRecvResized, 0, comm);
        mgcl::mgclCheckMpiError(comm, err, "MPI_Gather");

        mgcl::mgclCheckMpiError(comm, MPI_Type_free(&subarraySend), "MPI_Type_free");
        mgcl::mgclCheckMpiError(comm, MPI_Type_free(&subarrayRecv), "MPI_Type_free");
        mgcl::mgclCheckMpiError(comm, MPI_Type_free(&subarrayRecvResized), "MPI_Type_free");
    }

} // namespace mpi_util
