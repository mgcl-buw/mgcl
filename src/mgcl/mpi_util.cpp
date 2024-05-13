#include "mpi_util.hpp"

#include "cuboid.hpp"
#include "mpi_level_data.hpp"
#include "stencil.hpp"

#include <iostream>

// #include <cpptrace/cpptrace.hpp>

namespace mgcl::mpi_util
{
    /**
     * @brief Gathers all local grids into one big grid. Must be called from each process. On rank 0 c must have the
     *   size of the global grid, all other processes just need to send their local grid.
     *
     * @param comm
     * @param c Global grid for root process (rank 0), local grids for all other processes.
     */
    void gather(MPI_Comm comm, Cuboid& c)
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

        // MPI_Gatherv signature:
        // int MPI_Gatherv(const void* buffer_send,
        //                 int count_send,
        //                 MPI_Datatype datatype_send,
        //                 void* buffer_recv,
        //                 const int* counts_recv,
        //                 const int* displacements,
        //                 MPI_Datatype datatype_recv,
        //                 int root,
        //                 MPI_Comm communicator);

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
     * @brief Gathers all local grid stencils into one big grid stencil. Must be called from each process.
     * On rank 0 c must have the size of the global grid, all other processes just need to send their local grid.
     *
     * @param comm
     * @param c Global varying stencil for root process (rank 0), local varying stencil for all other processes.
     */
    void gather(MPI_Comm comm, VaryingStencil& c)
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
        int width = c.getWidth();

        // Create subarray type for the send buffer
        MPI_Datatype subarraySend;
        if (rank != 0)
        {
            int sizes[6] = {width, width, width, mlocgh, nlocgh, olocgh};
            int subsizes[6] = {width, width, width, mloc, nloc, oloc};
            int starts[6] = {0, 0, 0, locsizes[3], locsizes[4], locsizes[5]};
            err = MPI_Type_create_subarray(6, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &subarraySend);
            mgclCheckMpiError(comm, err, "MPI_Type_create_subarray");
            err = MPI_Type_commit(&subarraySend);
            mgclCheckMpiError(comm, err, "MPI_Type_commit");
        }

        // Create subarray type for the receive buffer
        MPI_Datatype subarrayRecv;
        MPI_Datatype subarrayRecvResized;
        if (rank == 0)
        {
            int sizes[6] = {width, width, width, mglobgh, nglobgh, oglobgh};
            int subsizes[6] = {width, width, width, mloc, nloc, oloc};
            int starts[6] = {0, 0, 0, c.getGhostsM(), c.getGhostsN(), c.getGhostsO()};
            // int starts[3] = {0, 0, 0};
            err = MPI_Type_create_subarray(6, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &subarrayRecv);
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
        int displ[mpi_size]; // displacements, 1d coordinate in the global receive buffer for each process
        // int width3 = width * width * width; // size of one stencil
        if (rank == 0)
            for (int i = 0; i < mpi_size; i++)
            {
                int coords[3] = {0, 0, 0};
                err = MPI_Cart_coords(comm, i, 3, coords);
                mgclCheckMpiError(comm, err, "MPI_Cart_coords");

                counts[i] = 1;
                displ[i] = coords[0] * mloc * nglobgh * oglobgh +
                           coords[1] * nloc * oglobgh +
                           coords[2] * oloc;
            }

        // MPI_Gatherv signature:
        // int MPI_Gatherv(const void* buffer_send,
        //                 int count_send,
        //                 MPI_Datatype datatype_send,
        //                 void* buffer_recv,
        //                 const int* counts_recv,
        //                 const int* displacements,
        //                 MPI_Datatype datatype_recv,
        //                 int root,
        //                 MPI_Comm communicator);

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
     * @brief Gathers all local grids into one big grid. Must be called from each process. On rank 0 c must have the
     *   size of the global grid, all other processes just need to send their local grid.
     *
     * @param comm MPI communicator.
     * @param commands OpenCL command queue.
     * @param c Global grid for root process (rank 0), local grids for all other processes.
     */
    void gather(MPI_Comm comm, cl_command_queue commands, CuboidGpu& c)
    {
        auto tmp = c.read(commands, nullptr, true);
        gather(comm, *tmp);
        c.write(commands, *tmp, true);
    }

    /**
     * @brief Gathers all local grid stencils into one big grid stencil. Must be called from each process.
     * On rank 0 c must have the size of the global grid, all other processes just need to send their local grid.
     *
     * @param comm MPI communicator.
     * @param commands OpenCL command queue.
     * @param c Global grid for root process (rank 0), local grids for all other processes.
     */
    void gather(MPI_Comm comm, cl_command_queue commands, VaryingStencilGpu& c)
    {
        auto tmp = c.read(commands, true);
        gather(comm, tmp);
        c.fill(tmp, commands, true);
    }

    /**
     * @brief Scatters from rank 0 to all other processes. Must be called from each process. On rank 0 src must have the
     *   size of the global grid, all other processes give in nullptr. dest is the local grid that is to be filled.
     *
     * @param comm
     * @param src Global grid for root process (rank 0), nullptr for other processes
     * @param dest Local grid into which data is scattered
     */
    void scatter(MPI_Comm comm, Cuboid* src, Cuboid& dest)
    {
        int err;
        int rank;
        err = MPI_Comm_rank(comm, &rank);
        mgclCheckMpiError(comm, err, "MPI_Comm_rank");

        int mpi_size;
        err = MPI_Comm_size(comm, &mpi_size);
        mgclCheckMpiError(comm, err, "MPI_Comm_size");

        if (mpi_size <= 1)
            throw "Scatter is only supported for at least 2 processes.";

        if (rank == 0 && src == nullptr)
            throw "src must not be null for root process!";

        int mloc = dest.getM();
        int nloc = dest.getN();
        int oloc = dest.getO();
        int mglobgh = src ? src->getMgh() : 0;
        int nglobgh = src ? src->getNgh() : 0;
        int oglobgh = src ? src->getOgh() : 0;

        // Create subarray type for the receive buffer
        MPI_Datatype subarrayRecv;
        int sizes[3] = {dest.getMgh(), dest.getNgh(), dest.getOgh()};
        int subsizes[3] = {mloc, nloc, oloc};
        int starts[3] = {dest.getGhostsM(), dest.getGhostsN(), dest.getGhostsM()};
        err = MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &subarrayRecv);
        mgclCheckMpiError(comm, err, "MPI_Type_create_subarray");
        err = MPI_Type_commit(&subarrayRecv);
        mgclCheckMpiError(comm, err, "MPI_Type_commit");

        // Create subarray type for the send buffer
        MPI_Datatype subarraySend;
        MPI_Datatype subarraySendResized;
        if (rank == 0)
        {
            int sizes[3] = {mglobgh, nglobgh, oglobgh};
            int subsizes[3] = {mloc, nloc, oloc};
            int starts[3] = {src->getGhostsM(), src->getGhostsN(), src->getGhostsO()};
            // int starts[3] = {0, 0, 0};
            err = MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &subarraySend);
            mgclCheckMpiError(comm, err, "MPI_Type_create_subarray");
            err = MPI_Type_commit(&subarraySend);
            mgclCheckMpiError(comm, err, "MPI_Type_commit");

            // Resize send data to avoid overlapping resulting from ghosts. Enabling explicitely stating start and extent
            // later in MPI_Scatterv (counts and displ). One unit is the size of a local grid.
            err = MPI_Type_create_resized(subarraySend, 0, 1 * sizeof(double), &subarraySendResized);
            // err = MPI_Type_create_resized(subarrayRecv, 0, oglobgh * sizeof(double), &subarrayRecvResized);
            mgclCheckMpiError(comm, err, "MPI_Type_create_resized");
            err = MPI_Type_commit(&subarraySendResized);
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
            MPI_Scatterv(src->field1d().data(), counts, displ, subarraySendResized,
                         dest.field1d().data(), 1, subarrayRecv, 0, comm);
        else
            MPI_Scatterv(nullptr, nullptr, nullptr, MPI_DATATYPE_NULL,
                         dest.field1d().data(), 1, subarrayRecv, 0, comm);
        mgclCheckMpiError(comm, err, "MPI_Scatterv");

        mgclCheckMpiError(comm, MPI_Type_free(&subarrayRecv), "MPI_Type_free");
        if (rank == 0)
        {
            mgclCheckMpiError(comm, MPI_Type_free(&subarraySend), "MPI_Type_free");
            mgclCheckMpiError(comm, MPI_Type_free(&subarraySendResized), "MPI_Type_free");
        }
    }

    /**
     * @brief Scatters from rank 0 to all other processes. Must be called from each process. On rank 0 c must have the
     *   size of the global grid, local grid size on all other processes. Rank 0 scatters in-place, i.e. should not
     *   be altered at all.
     *
     * @param comm
     * @param c Global grid for root process (rank 0), local grid for other processes
     */
    void scatter_inplace(MPI_Comm comm, Cuboid& c)
    {
        int err;
        int rank;
        err = MPI_Comm_rank(comm, &rank);
        mgclCheckMpiError(comm, err, "MPI_Comm_rank");

        int mpi_size;
        err = MPI_Comm_size(comm, &mpi_size);
        mgclCheckMpiError(comm, err, "MPI_Comm_size");

        if (mpi_size <= 1)
            throw "Scatter is only supported for at least 2 processes.";

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

        // Create subarray type for the receive buffer
        MPI_Datatype subarrayRecv;
        if (rank > 0)
        {
            int sizes[3] = {mlocgh, nlocgh, olocgh};
            int subsizes[3] = {mloc, nloc, oloc};
            int starts[3] = {locsizes[3], locsizes[4], locsizes[5]};
            err = MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &subarrayRecv);
            mgclCheckMpiError(comm, err, "MPI_Type_create_subarray");
            err = MPI_Type_commit(&subarrayRecv);
            mgclCheckMpiError(comm, err, "MPI_Type_commit");
        }

        // Create subarray type for the send buffer
        MPI_Datatype subarraySend;
        MPI_Datatype subarraySendResized;
        if (rank == 0)
        {
            int sizes[3] = {mglobgh, nglobgh, oglobgh};
            int subsizes[3] = {mloc, nloc, oloc};
            int starts[3] = {c.getGhostsM(), c.getGhostsN(), c.getGhostsO()};
            // int starts[3] = {0, 0, 0};
            err = MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &subarraySend);
            mgclCheckMpiError(comm, err, "MPI_Type_create_subarray");
            err = MPI_Type_commit(&subarraySend);
            mgclCheckMpiError(comm, err, "MPI_Type_commit");

            // Resize send data to avoid overlapping resulting from ghosts. Enabling explicitely stating start and extent
            // later in MPI_Scatterv (counts and displ). One unit is the size of a local grid.
            err = MPI_Type_create_resized(subarraySend, 0, 1 * sizeof(double), &subarraySendResized);
            // err = MPI_Type_create_resized(subarrayRecv, 0, oglobgh * sizeof(double), &subarrayRecvResized);
            mgclCheckMpiError(comm, err, "MPI_Type_create_resized");
            err = MPI_Type_commit(&subarraySendResized);
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
            MPI_Scatterv(c.field1d().data(), counts, displ, subarraySendResized,
                         MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, 0, comm);
        else
            MPI_Scatterv(nullptr, nullptr, nullptr, MPI_DATATYPE_NULL,
                         c.field1d().data(), 1, subarrayRecv, 0, comm);
        mgclCheckMpiError(comm, err, "MPI_Scatterv");

        if (rank > 0)
            mgclCheckMpiError(comm, MPI_Type_free(&subarrayRecv), "MPI_Type_free");
        if (rank == 0)
        {
            mgclCheckMpiError(comm, MPI_Type_free(&subarraySend), "MPI_Type_free");
            mgclCheckMpiError(comm, MPI_Type_free(&subarraySendResized), "MPI_Type_free");
        }
    }

    /**
     * @brief Scatters from rank 0 to all other processes. Must be called from each process. On rank 0 c must have the
     *   size of the global grid, local grid size on all other processes. Rank 0 scatters in-place, i.e. should not
     *   be altered at all. Includes ghost cells.
     *
     * @param comm
     * @param c Global grid for root process (rank 0), local grid for other processes
     */
    void scatter_inplace_wgh(MPI_Comm comm, Cuboid& c)
    {
        int err;
        int rank;
        err = MPI_Comm_rank(comm, &rank);
        mgclCheckMpiError(comm, err, "MPI_Comm_rank");

        int mpi_size;
        err = MPI_Comm_size(comm, &mpi_size);
        mgclCheckMpiError(comm, err, "MPI_Comm_size");

        if (mpi_size <= 1)
            throw "Scatter is only supported for at least 2 processes.";

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

        // Create subarray type for the receive buffer
        MPI_Datatype subarrayRecv;
        if (rank > 0)
        {
            int sizes[3] = {mlocgh, nlocgh, olocgh};
            int subsizes[3] = {mlocgh, nlocgh, olocgh};
            int starts[3] = {0, 0, 0};
            err = MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &subarrayRecv);
            mgclCheckMpiError(comm, err, "MPI_Type_create_subarray");
            err = MPI_Type_commit(&subarrayRecv);
            mgclCheckMpiError(comm, err, "MPI_Type_commit");
        }

        // Create subarray type for the send buffer
        MPI_Datatype subarraySend;
        MPI_Datatype subarraySendResized;
        if (rank == 0)
        {
            int sizes[3] = {mglobgh, nglobgh, oglobgh};
            int subsizes[3] = {mlocgh, nlocgh, olocgh};
            int starts[3] = {0, 0, 0};
            // int starts[3] = {0, 0, 0};
            err = MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &subarraySend);
            mgclCheckMpiError(comm, err, "MPI_Type_create_subarray");
            err = MPI_Type_commit(&subarraySend);
            mgclCheckMpiError(comm, err, "MPI_Type_commit");

            // Resize send data to avoid overlapping resulting from ghosts. Enabling explicitely stating start and extent
            // later in MPI_Scatterv (counts and displ). One unit is the size of a local grid.
            err = MPI_Type_create_resized(subarraySend, 0, 1 * sizeof(double), &subarraySendResized);
            // err = MPI_Type_create_resized(subarrayRecv, 0, oglobgh * sizeof(double), &subarrayRecvResized);
            mgclCheckMpiError(comm, err, "MPI_Type_create_resized");
            err = MPI_Type_commit(&subarraySendResized);
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
            MPI_Scatterv(c.field1d().data(), counts, displ, subarraySendResized,
                         MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, 0, comm);
        else
            MPI_Scatterv(nullptr, nullptr, nullptr, MPI_DATATYPE_NULL,
                         c.field1d().data(), 1, subarrayRecv, 0, comm);
        mgclCheckMpiError(comm, err, "MPI_Scatterv");

        if (rank > 0)
            mgclCheckMpiError(comm, MPI_Type_free(&subarrayRecv), "MPI_Type_free");
        if (rank == 0)
        {
            mgclCheckMpiError(comm, MPI_Type_free(&subarraySend), "MPI_Type_free");
            mgclCheckMpiError(comm, MPI_Type_free(&subarraySendResized), "MPI_Type_free");
        }
    }

    /**
     * @brief Scatters from rank 0 to all other processes. Must be called from each process. On rank 0 c must have the
     *   size of the global grid, local grid size on all other processes. Rank 0 scatters in-place, i.e. should not
     *   be altered at all. Includes ghost cells.
     *
     * @param comm MPI communicator.
     * @param commands OpenCL command queue.
     * @param c Global grid for root process (rank 0), local grid for other processes
     */
    void scatter_inplace_wgh(MPI_Comm comm, cl_command_queue commands, CuboidGpu& c)
    {
        auto tmp = c.read(commands, nullptr, true);
        scatter_inplace_wgh(comm, *tmp);
        c.write(commands, *tmp, true);
    }

    /**
     * @brief Checks the return code of a MPI call and prints it if not MPI_SUCCESS.
     */
    void mgcl_check_mpi_error(MPI_Comm comm, int err, const char* operation, const char* filename, int line)
    {
        if (err != MPI_SUCCESS)
        {
            int length;
            char message[MPI_MAX_ERROR_STRING];
            MPI_Error_string(err, message, &length);

            fprintf(stderr, "Error during operation '%s', ", operation);
            fprintf(stderr, "in '%s' on line %d\n", filename, line);
            fprintf(stderr, "Error code %d: message was %s\n", err, message);
            // cpptrace::generate_trace().print();
            MPI_Abort(comm, err);
        }
    }

    /**
     * @brief Sends border planes to neighbouring processes.
     * TODO maybe use MPI_Sendrecv_inplace instead
     *
     * @param mgh Extend of cuboid, that the planes belong to, in z-direction
     * @param ngh Extend of cuboid, that the planes belong to, in y-direction
     * @param ogh Extend of cuboid, that the planes belong to, in x-direction
     * @param ghosts_m Ghosts of cuboid, that the planes belong to, at one border in z-direction
     * @param ghosts_n Ghosts of cuboid, that the planes belong to, at one border in y-direction
     * @param ghosts_o Ghosts of cuboid, that the planes belong to, at one border in x-direction
     * @param sbuf Send buffer, must contain planes in the same order that CuboidGpu::extractBorderPlanes() returns.
     * @param rbuf Temporary receive buffer
     * @param mpiData Contains neighbouring processes' info
     */
    void sendBorderPlanes(int mgh, int ngh, int ogh, int ghosts_m, int ghosts_n, int ghosts_o,
                          Cuboid& sbuf, Cuboid& rbuf, MPILevelData& mpiData)
    {
        int yz = ngh * ogh;
        int xz = mgh * ogh;
        int xy = mgh * ngh;

        int m = mgh - 2 * ghosts_m;
        int n = ngh - 2 * ghosts_n;

        // int base_yz_front = 0;
        int base_yz_back = ghosts_m * yz;
        int base_xz_top = 2 * ghosts_m * yz;
        int base_xz_bottom = 2 * ghosts_m * yz + ghosts_n * xz;
        int base_xy_left = 2 * ghosts_m * yz + 2 * ghosts_n * xz;
        int base_xy_right = 2 * ghosts_m * yz + 2 * ghosts_n * xz + ghosts_o * xy;

        // Send planes to neighbors
        int myid, err;
        MPI_Comm_rank(mpiData.comm, &myid);

        // Send front planes to the back
        err = MPI_Sendrecv(static_cast<void*>(sbuf[0][0]), ghosts_m * yz, MPI_DOUBLE, mpiData.back, 0,
                           static_cast<void*>(rbuf[0][0]), ghosts_m * yz, MPI_DOUBLE, mpiData.front, 0,
                           mpiData.comm, MPI_STATUS_IGNORE);
        mgcl::mpi_util::mgclCheckMpiError(mpiData.comm, err, "MPI_Sendrecv");

        // Send back planes to the front
        err = MPI_Sendrecv(static_cast<void*>(&(sbuf[0][0][base_yz_back])), ghosts_m * yz, MPI_DOUBLE, mpiData.front, 0,
                           static_cast<void*>(&(rbuf[0][0][base_yz_back])), ghosts_m * yz, MPI_DOUBLE, mpiData.back, 0,
                           mpiData.comm, MPI_STATUS_IGNORE);
        mgcl::mpi_util::mgclCheckMpiError(mpiData.comm, err, "MPI_Sendrecv");

        // Write received edges of cuboid to send buffer
        // i0: i index of recv buffers for yz plane (always all i indices)
        // i1: i index of send buffers xz back ghosts
        // j0: j index of send buffers for xz plane (always all j indices)
        // j1: j index of recv buffer yz top edge
        // j2: j index of recv buffer yz bottom edge
        for (int i0 = 0, i1 = m + ghosts_m;
             i0 < ghosts_m;
             i0++, i1++)
            for (int j0 = 0, j1 = ghosts_n, j2 = n;
                 j0 < ghosts_n;
                 j0++, j1++, j2++)
                for (int k = 0; k < ogh; k++)
                {
                    // Upper front edge - Write ghosts in the front (from back recv buffer) to xz top send buffer
                    sbuf[0][0][base_xz_top + j0 * xz + i0 * ogh + k] = rbuf[0][0][base_yz_back + i0 * yz + j1 * ogh + k];

                    // Lower front edge - Write ghosts in the front (from back recv buffer) to xz bottom send buffer
                    sbuf[0][0][base_xz_bottom + j0 * xz + i0 * ogh + k] = rbuf[0][0][base_yz_back + i0 * yz + j2 * ogh + k];

                    // Upper back edge - Write ghosts in the back (from front recv buffer, base 0) to xz top send buffer
                    sbuf[0][0][base_xz_top + j0 * xz + i1 * ogh + k] = rbuf[0][0][i0 * yz + j1 * ogh + k];

                    // Lower back edge - Write ghosts in the back (from front recv buffer, base 0) to xz bottom send buffer
                    sbuf[0][0][base_xz_bottom + j0 * xz + i1 * ogh + k] = rbuf[0][0][i0 * yz + j2 * ogh + k];
                }

        // Send top planes to the bottom
        err = MPI_Sendrecv(static_cast<void*>(&(sbuf[0][0][base_xz_top])), ghosts_n * xz, MPI_DOUBLE, mpiData.down, 0,
                           static_cast<void*>(&(rbuf[0][0][base_xz_top])), ghosts_n * xz, MPI_DOUBLE, mpiData.up, 0,
                           mpiData.comm, MPI_STATUS_IGNORE);
        mgcl::mpi_util::mgclCheckMpiError(mpiData.comm, err, "MPI_Sendrecv");

        // Send bottom planes to the top
        err = MPI_Sendrecv(static_cast<void*>(&(sbuf[0][0][base_xz_bottom])), ghosts_n * xz, MPI_DOUBLE, mpiData.up, 0,
                           static_cast<void*>(&(rbuf[0][0][base_xz_bottom])), ghosts_n * xz, MPI_DOUBLE, mpiData.down, 0,
                           mpiData.comm, MPI_STATUS_IGNORE);
        mgcl::mpi_util::mgclCheckMpiError(mpiData.comm, err, "MPI_Sendrecv");

        // Send left planes to the right
        err = MPI_Sendrecv(static_cast<void*>(&(sbuf[0][0][base_xy_left])), ghosts_o * xy, MPI_DOUBLE, mpiData.right, 0,
                           static_cast<void*>(&(rbuf[0][0][base_xy_left])), ghosts_o * xy, MPI_DOUBLE, mpiData.left, 0,
                           mpiData.comm, MPI_STATUS_IGNORE);
        mgcl::mpi_util::mgclCheckMpiError(mpiData.comm, err, "MPI_Sendrecv");

        // Send right planes to the left
        err = MPI_Sendrecv(static_cast<void*>(&(sbuf[0][0][base_xy_right])), ghosts_o * xy, MPI_DOUBLE, mpiData.left, 0,
                           static_cast<void*>(&(rbuf[0][0][base_xy_right])), ghosts_o * xy, MPI_DOUBLE, mpiData.right, 0,
                           mpiData.comm, MPI_STATUS_IGNORE);
        mgcl::mpi_util::mgclCheckMpiError(mpiData.comm, err, "MPI_Sendrecv");
    }

} // namespace mpi_util
