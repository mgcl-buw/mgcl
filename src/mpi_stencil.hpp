#ifndef MGCL_MPI_STENCIL_HPP
#define MGCL_MPI_STENCIL_HPP

#include "mpi_level_data.hpp"
#include "stencil.hpp"

namespace mgcl
{
    /* Updates ghost cells of a varying stencil. This is essentially equal to updateGhostsSeq in ghostscl.cpp.
     * If periodic is true, every ghost cell will be updated. Otherwise the outermost ghost cells will be excluded. It
     *   also affects the update using MPI where nodes are wrapped around in the periodic case.
     * mpiData parameter is optional (i.e. nullable) and is only used when MPI is used. If mgcl is called with
     *   only one MPI process, updateGhostsSeqLocally will be used instead.
     * forceLocal: If true, ghosts will be updated locally (maybe giving wrong results, if m_local < m_global).
     *   This is used e.g. for levels above the mpiLevelThreshold. */
    template <int N>
    void updateGhostsStencilMpi(VaryingStencil<N>& s, MPILevelData* mpiData, bool periodic, bool forceLocal)
    {
        // TODO adjust for ghosts > 1
        if (forceLocal || mpiData == nullptr || mpiData->mpiSize() == 1)
        {
            s.updateGhosts();
            return;
        }

        int m = s.getDim1();
        int n = s.getDim2();
        int o = s.getDim3();
        int ghosts_m = s.getGhostsDim1();
        int ghosts_n = s.getGhostsDim2();
        int ghosts_o = s.getGhostsDim3();
        int mgh = s.getDim1gh();
        int ngh = s.getDim2gh();
        int ogh = s.getDim3gh();

        if (m < ghosts_m || n < ghosts_n || o < ghosts_o)
            throw "Ghost-update using MPI is only allowed for gh <= m,n,o.";

        // // Create subarray type for the ghost slices for each direction
        // MPI_Datatype ghostsSliceX;
        // int sizesX[3] = {mgh, ngh, ogh};
        // int subsizesX[3] = {ghosts_m, ngh, ogh};
        // int startsX[3] = {0, 0, 0};
        // err = MPI_Type_create_subarray(2, sizesX, subsizesX, startsX, MPI_ORDER_C, MPI_DOUBLE, &ghostsSliceX);
        // mgclCheckMpiError(comm, err, "MPI_Type_create_subarray");
        // err = MPI_Type_commit(&ghostsSliceX);
        // mgclCheckMpiError(comm, err, "MPI_Type_commit");

        // MPI_Datatype ghostsSliceY;
        // int sizesY[3] = {mgh, ngh, ogh};
        // int subsizesY[3] = {mgh, ghosts_n, ogh};
        // int startsY[3] = {0, 0, 0};
        // err = MPI_Type_create_subarray(2, sizesY, subsizesY, startsY, MPI_ORDER_C, MPI_DOUBLE, &ghostsSliceY);
        // mgclCheckMpiError(comm, err, "MPI_Type_create_subarray");
        // err = MPI_Type_commit(&ghostsSliceY);
        // mgclCheckMpiError(comm, err, "MPI_Type_commit");

        // MPI_Datatype ghostsSliceZ;
        // int sizesZ[3] = {mgh, ngh, ogh};
        // int subsizesZ[3] = {mgh, ngh, ghosts_o};
        // int startsZ[3] = {0, 0, 0};
        // err = MPI_Type_create_subarray(2, sizesZ, subsizesZ, startsZ, MPI_ORDER_C, MPI_DOUBLE, &ghostsSliceZ);
        // mgclCheckMpiError(comm, err, "MPI_Type_create_subarray");
        // err = MPI_Type_commit(&ghostsSliceZ);
        // mgclCheckMpiError(comm, err, "MPI_Type_commit");

        // // Exchange data in x-direction
        // // Send data left
        // MPI_Sendrecv(&(c[ghosts_m][0][0]), 1, ghostsSliceX, mpiData->left, 0,
        //              &(c[0][0][0]), 1, ghostsSliceX, mpiData->right, 0, comm, MPI_STATUS_IGNORE);
        // // Send data right
        // MPI_Sendrecv(&(c[0][0][0]), 1, ghostsSliceX, mpiData->right, 0,
        //              &(c[ghosts_m + m][0][0]), 1, ghostsSliceX, mpiData->left, 0, comm, MPI_STATUS_IGNORE);

        // // Exchange data in y-direction
        // // Send data upwards
        // MPI_Sendrecv(&(c[0][ghosts_n][0]), 1, ghostsSliceY, mpiData->up, 0,
        //              &(c[0][0][0]), 1, ghostsSliceY, mpiData->down, 0, comm, MPI_STATUS_IGNORE);
        // // Send data downwards
        // MPI_Sendrecv(&(c[0][0][0]), 1, ghostsSliceY, mpiData->down, 0,
        //              &(c[0][ghosts_n + n][0]), 1, ghostsSliceY, mpiData->up, 0, comm, MPI_STATUS_IGNORE);

        // // Exchange data in z-direction
        // // Send data to the back
        // MPI_Sendrecv(&(c[0][0][ghosts_o]), 1, ghostsSliceZ, mpiData->back, 0,
        //              &(c[0][0][0]), 1, ghostsSliceZ, mpiData->front, 0, comm, MPI_STATUS_IGNORE);
        // // Send data to the front
        // MPI_Sendrecv(&(c[0][0][0]), 1, ghostsSliceZ, mpiData->front, 0,
        //              &(c[0][0][ghosts_o + o]), 1, ghostsSliceZ, mpiData->back, 0, comm, MPI_STATUS_IGNORE);

        // // Free types
        // MPI_Type_free(&ghostsSliceX);
        // MPI_Type_free(&ghostsSliceY);
        // MPI_Type_free(&ghostsSliceZ);

        // /* Loop variables */
        // int i, j, k;

        // /* Getting local rank */
        // int myid;
        // MPI_Comm_rank(mpiData->comm, &myid);

        // /* Sending data to the front */
        // auto sbufyz_ptr = c.sliceIncGhosts(ghosts_m, 2 * ghosts_m - 1, 0, ngh - 1, 0, ogh - 1); // TODO max when gh > m
        // auto sbufyz = sbufyz_ptr->getData();
        // auto rbufyz_ptr = std::make_unique<Cuboid>(sbufyz_ptr->getM(), sbufyz_ptr->getN(), sbufyz_ptr->getO(), 0, 0, 0);
        // auto rbufyz = rbufyz_ptr->getData();

        // MPI_Sendrecv((void*)sbufyz[0][0], ghosts_m * ngh * ogh, MPI_DOUBLE, mpiData->front, 0,
        //              (void*)rbufyz[0][0], ghosts_m * ngh * ogh, MPI_DOUBLE, mpiData->back, 0,
        //              mpiData->comm, MPI_STATUS_IGNORE);

        // if (MPI_PROC_NULL != mpiData->back)
        //     for (i = 0; i < ghosts_m; i++)
        //         for (j = 0; j < ngh; j++)
        //             for (k = 0; k < ogh; k++)
        //                 c[mgh - ghosts_m + i][j][k] = rbufyz[i][j][k];

        // /* Sending data to the back */
        // sbufyz_ptr = c.sliceIncGhosts(m, m + ghosts_m - 1, 0, ngh - 1, 0, ogh - 1); // TODO max when gh > m
        // sbufyz = sbufyz_ptr->getData();
        // rbufyz_ptr = std::make_unique<Cuboid>(sbufyz_ptr->getM(), sbufyz_ptr->getN(), sbufyz_ptr->getO(), 0, 0, 0);
        // rbufyz = rbufyz_ptr->getData();

        // MPI_Sendrecv((void*)sbufyz[0][0], ghosts_m * ngh * ogh, MPI_DOUBLE, mpiData->back, 0,
        //              (void*)rbufyz[0][0], ghosts_m * ngh * ogh, MPI_DOUBLE, mpiData->front, 0,
        //              mpiData->comm, MPI_STATUS_IGNORE);

        // if (MPI_PROC_NULL != mpiData->front)
        //     for (i = 0; i < ghosts_m; i++)
        //         for (j = 0; j < ngh; j++)
        //             for (k = 0; k < ogh; k++)
        //                 c[i][j][k] = rbufyz[i][j][k];

        // /* Sending data downwards */
        // auto sbufxz_ptr = c.sliceIncGhosts(0, mgh - 1, ghosts_m, 2 * ghosts_n - 1, 0, ogh - 1); // TODO max when gh > m
        // auto sbufxz = sbufxz_ptr->getData();
        // auto rbufxz_ptr = std::make_unique<Cuboid>(sbufxz_ptr->getM(), sbufxz_ptr->getN(), sbufxz_ptr->getO(), 0, 0, 0);
        // auto rbufxz = rbufxz_ptr->getData();

        // MPI_Sendrecv((void*)sbufxz[0][0], mgh * ghosts_n * ogh, MPI_DOUBLE, mpiData->down, 0,
        //              (void*)rbufxz[0][0], mgh * ghosts_n * ogh, MPI_DOUBLE, mpiData->up, 0,
        //              mpiData->comm, MPI_STATUS_IGNORE);

        // if (MPI_PROC_NULL != mpiData->up)
        //     for (i = 0; i < mgh; i++)
        //         for (j = 0; j < ghosts_n; j++)
        //             for (k = 0; k < ogh; k++)
        //                 c[i][ngh - ghosts_n + j][k] = rbufxz[i][j][k];

        // /* Sending data upwards */
        // sbufxz_ptr = c.sliceIncGhosts(0, mgh - 1, n, n + ghosts_n - 1, 0, ogh - 1); // TODO max when gh > m
        // sbufxz = sbufxz_ptr->getData();
        // rbufxz_ptr = std::make_unique<Cuboid>(sbufxz_ptr->getM(), sbufxz_ptr->getN(), sbufxz_ptr->getO(), 0, 0, 0);
        // rbufxz = rbufxz_ptr->getData();

        // MPI_Sendrecv((void*)sbufxz[0][0], mgh * ghosts_n * ogh, MPI_DOUBLE, mpiData->up, 0,
        //              (void*)rbufxz[0][0], mgh * ghosts_n * ogh, MPI_DOUBLE, mpiData->down, 0,
        //              mpiData->comm, MPI_STATUS_IGNORE);

        // if (MPI_PROC_NULL != mpiData->down)
        //     for (i = 0; i < mgh; i++)
        //         for (j = 0; j < ghosts_n; j++)
        //             for (k = 0; k < ogh; k++)
        //                 c[i][j][k] = rbufxz[i][j][k];

        // /* Sending data to the left */
        // sbufxz_ptr = c.sliceIncGhosts(0, mgh - 1, 0, ngh - 1, ghosts_o, 2 * ghosts_o - 1); // TODO max when gh > m
        // sbufxz = sbufxz_ptr->getData();
        // rbufxz_ptr = std::make_unique<Cuboid>(sbufxz_ptr->getM(), sbufxz_ptr->getN(), sbufxz_ptr->getO(), 0, 0, 0);
        // rbufxz = rbufxz_ptr->getData();

        // // std::cout << myid << "," << mpiData->left << std::endl;
        // // MPI_Barrier(comm);
        // // sbufxz_ptr->dumpToFile("sbufyz_ptr_left" + std::to_string(myid) + ".txt");

        // MPI_Sendrecv((void*)sbufxz[0][0], mgh * ngh * ghosts_o, MPI_DOUBLE, mpiData->left, 0,
        //              (void*)rbufxz[0][0], mgh * ngh * ghosts_o, MPI_DOUBLE, mpiData->right, 0,
        //              mpiData->comm, MPI_STATUS_IGNORE);
        // if (MPI_PROC_NULL != mpiData->right)
        //     for (i = 0; i < mgh; i++)
        //         for (j = 0; j < ngh; j++)
        //             for (k = 0; k < ghosts_o; k++)
        //                 c[i][j][ogh - ghosts_o + k] = rbufxz[i][j][k];

        // /* Sending data to the right */
        // sbufxz_ptr = c.sliceIncGhosts(0, mgh - 1, 0, ngh - 1, o, o + ghosts_o - 1); // TODO max when gh > m
        // sbufxz = sbufxz_ptr->getData();
        // rbufxz_ptr = std::make_unique<Cuboid>(sbufxz_ptr->getM(), sbufxz_ptr->getN(), sbufxz_ptr->getO(), 0, 0, 0);
        // rbufxz = rbufxz_ptr->getData();

        // MPI_Sendrecv((void*)sbufxz[0][0], mgh * ngh * ghosts_o, MPI_DOUBLE, mpiData->right, 0,
        //              (void*)rbufxz[0][0], mgh * ngh * ghosts_o, MPI_DOUBLE, mpiData->left, 0,
        //              mpiData->comm, MPI_STATUS_IGNORE);

        // if (MPI_PROC_NULL != mpiData->left)
        //     for (i = 0; i < mgh; i++)
        //         for (j = 0; j < ngh; j++)
        //             for (k = 0; k < ghosts_o; k++)
        //                 c[i][j][k] = rbufxz[i][j][k];
    }
}

#endif // MGCL_MPI_STENCIL_HPP
