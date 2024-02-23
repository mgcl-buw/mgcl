#include "cuboid.hpp" // for Cuboid
#include "kernel_config.hpp"
#include "mgcl.hpp" // for BC
#include "mpi_util.hpp"
#include "multigrid_engine.hpp" // for Problem, MultigridEngine
#include "opencl_helper.hpp"    // for mgclCheckError, OpenCLHelper

#include <cassert>
#include <cstddef> // for size_t, NULL
#include <iostream>

#include "mpi_level_data.hpp"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

namespace mgcl
{
    using std::size_t;

    // Private helper function for updating ghosts without MPI.
    void updateGhostsSeqLocally(Cuboid& c, bool periodic)
    {
        int m = c.getM();
        int n = c.getN();
        int o = c.getO();
        int ghosts_m = c.getGhostsM();
        int ghosts_n = c.getGhostsN();
        int ghosts_o = c.getGhostsO();

        int ghm_start_right = ghosts_m + m;
        int ghn_start_right = ghosts_n + n;
        int gho_start_right = ghosts_o + o;

        // clang-format off
        // sending data in z-direction           
        for (int i = 0; i < ghosts_m; i++)
        {
            int factor_left = (ghosts_m - 1 - i) / m + 1;
            int factor_right = (ghm_start_right + i - ghosts_m) / m;

            for (int j = 0; j < n + 2 * ghosts_n; j++)
            for (int k = 0; k < o + 2 * ghosts_o; k++)
                {
                    
                    c[i][j][k] = c[i + factor_left * m][j][k]; // left ghost cell = right real cell
                    c[ghm_start_right + i][j][k] = c[ghm_start_right + i - factor_right * m][j][k]; // right ghost cell = left real cell
                }
        }

        // sending data in y-direction           
        for (int i = 0; i < ghosts_n; i++)
        {
            int factor_left = (ghosts_n - 1 - i) / n + 1;
            int factor_right = (ghn_start_right + i - ghosts_n) / n;

            for (int j = 0; j < m + 2 * ghosts_m; j++)
            for (int k = 0; k < o + 2 * ghosts_o; k++)
                {
                    
                    c[j][i][k] = c[j][i + factor_left * n][k]; // left ghost cell = right real cell
                    c[j][ghn_start_right + i][k] = c[j][ghn_start_right + i - factor_right * n][k]; // right ghost cell = left real cell
                }
        }

        // sending data in x-direction           
        for (int i = 0; i < ghosts_o; i++)
        {
            int factor_left = (ghosts_o - 1 - i) / o + 1;
            int factor_right = (gho_start_right + i - ghosts_o) / o;

            for (int j = 0; j < m + 2 * ghosts_m; j++)
            for (int k = 0; k < n + 2 * ghosts_n; k++)
                {
                    
                    c[j][k][i] = c[j][k][i + factor_left * o]; // left ghost cell = right real cell
                    c[j][k][gho_start_right + i] = c[j][k][gho_start_right + i - factor_right * o]; // right ghost cell = left real cell
                }
        }
        // clang-format on
    }

    /* Updates ghost cells.
     * If periodic is true, every ghost cell will be updated. Otherwise the outermost ghost cells will be excluded. It
     *   also affects the update using MPI where nodes are wrapped around in the periodic case.
     * mpiData parameter is optional (i.e. nullable) and is only used when MPI is used. If mgcl is called with
     *   only one MPI process, updateGhostsSeqLocally will be used instead.
     * forceLocal: If true, ghosts will be updated locally (maybe giving wrong results, if m_local < m_global).
     *   This is used e.g. for levels above the mpiLevelThreshold. */
    void MultigridEngine::updateGhostsSeq(Cuboid& c, MPILevelData* mpiData, bool periodic, bool forceLocal)
    {
        // TODO adjust for ghosts > 1
        if (forceLocal || mpiData == nullptr || mpiData->mpiSize() == 1)
        {
            updateGhostsSeqLocally(c, periodic);
            return;
        }

        int m = c.getM();
        int n = c.getN();
        int o = c.getO();
        int ghosts_m = c.getGhostsM();
        int ghosts_n = c.getGhostsN();
        int ghosts_o = c.getGhostsO();
        int mgh = c.getMgh();
        int ngh = c.getNgh();
        int ogh = c.getOgh();

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

        /* Loop variables */
        int i, j, k;

        // int ghm_start_right = ghosts_m + m;
        // int ghn_start_right = ghosts_n + n;
        // int gho_start_right = ghosts_o + o;

        // // clang-format off
        // // sending data in z-direction
        // for (int i = 0; i < ghosts_m; i++)
        // {
        //     int factor_left = (ghosts_m - 1 - i) / m + 1;
        //     int factor_right = (ghm_start_right + i - ghosts_m) / m;

        //     for (int j = 0; j < n + 2 * ghosts_n; j++)
        //     for (int k = 0; k < o + 2 * ghosts_o; k++)
        //         {

        //             c[i][j][k] = c[i + factor_left * m][j][k]; // left ghost cell = right real cell
        //             c[ghm_start_right + i][j][k] = c[ghm_start_right + i - factor_right * m][j][k]; // right ghost cell = left real cell
        //         }
        // }
        // // clang-format on

        /* Getting local rank */
        int myid;
        MPI_Comm_rank(mpiData->comm, &myid);

        int err;

        /* Sending data to the front */
        auto sbuf_ptr = c.sliceIncGhosts(ghosts_m, 2 * ghosts_m - 1, 0, ngh - 1, 0, ogh - 1); // TODO max when gh > m
        auto sbuf = sbuf_ptr->getData();
        auto rbuf_ptr = std::make_unique<Cuboid>(sbuf_ptr->getM(), sbuf_ptr->getN(), sbuf_ptr->getO(), 0, 0, 0);
        auto rbuf = rbuf_ptr->getData();

        err = MPI_Sendrecv(static_cast<void*>(sbuf[0][0]), ghosts_m * ngh * ogh, MPI_DOUBLE, mpiData->front, 0,
                           static_cast<void*>(rbuf[0][0]), ghosts_m * ngh * ogh, MPI_DOUBLE, mpiData->back, 0,
                           mpiData->comm, MPI_STATUS_IGNORE);
        mgcl::mpi_util::mgclCheckMpiError(mpiData->comm, err, "MPI_Sendrecv");

        if (MPI_PROC_NULL != mpiData->back)
            for (i = 0; i < ghosts_m; i++)
                for (j = 0; j < ngh; j++)
                    for (k = 0; k < ogh; k++)
                        c[mgh - ghosts_m + i][j][k] = rbuf[i][j][k];

        /* Sending data to the back */
        sbuf_ptr = c.sliceIncGhosts(m, m + ghosts_m - 1, 0, ngh - 1, 0, ogh - 1); // TODO max when gh > m
        sbuf = sbuf_ptr->getData();
        rbuf_ptr = std::make_unique<Cuboid>(sbuf_ptr->getM(), sbuf_ptr->getN(), sbuf_ptr->getO(), 0, 0, 0);
        rbuf = rbuf_ptr->getData();

        err = MPI_Sendrecv(static_cast<void*>(sbuf[0][0]), ghosts_m * ngh * ogh, MPI_DOUBLE, mpiData->back, 0,
                           static_cast<void*>(rbuf[0][0]), ghosts_m * ngh * ogh, MPI_DOUBLE, mpiData->front, 0,
                           mpiData->comm, MPI_STATUS_IGNORE);
        mgcl::mpi_util::mgclCheckMpiError(mpiData->comm, err, "MPI_Sendrecv");

        if (MPI_PROC_NULL != mpiData->front)
            for (i = 0; i < ghosts_m; i++)
                for (j = 0; j < ngh; j++)
                    for (k = 0; k < ogh; k++)
                        c[i][j][k] = rbuf[i][j][k];

        /* Sending data downwards */
        sbuf_ptr = c.sliceIncGhosts(0, mgh - 1, ghosts_m, 2 * ghosts_n - 1, 0, ogh - 1); // TODO max when gh > m
        sbuf = sbuf_ptr->getData();
        rbuf_ptr = std::make_unique<Cuboid>(sbuf_ptr->getM(), sbuf_ptr->getN(), sbuf_ptr->getO(), 0, 0, 0);
        rbuf = rbuf_ptr->getData();

        err = MPI_Sendrecv(static_cast<void*>(sbuf[0][0]), mgh * ghosts_n * ogh, MPI_DOUBLE, mpiData->down, 0,
                           static_cast<void*>(rbuf[0][0]), mgh * ghosts_n * ogh, MPI_DOUBLE, mpiData->up, 0,
                           mpiData->comm, MPI_STATUS_IGNORE);
        mgcl::mpi_util::mgclCheckMpiError(mpiData->comm, err, "MPI_Sendrecv");

        if (MPI_PROC_NULL != mpiData->up)
            for (i = 0; i < mgh; i++)
                for (j = 0; j < ghosts_n; j++)
                    for (k = 0; k < ogh; k++)
                        c[i][ngh - ghosts_n + j][k] = rbuf[i][j][k];

        /* Sending data upwards */
        sbuf_ptr = c.sliceIncGhosts(0, mgh - 1, n, n + ghosts_n - 1, 0, ogh - 1); // TODO max when gh > m
        sbuf = sbuf_ptr->getData();
        rbuf_ptr = std::make_unique<Cuboid>(sbuf_ptr->getM(), sbuf_ptr->getN(), sbuf_ptr->getO(), 0, 0, 0);
        rbuf = rbuf_ptr->getData();

        err = MPI_Sendrecv(static_cast<void*>(sbuf[0][0]), mgh * ghosts_n * ogh, MPI_DOUBLE, mpiData->up, 0,
                           static_cast<void*>(rbuf[0][0]), mgh * ghosts_n * ogh, MPI_DOUBLE, mpiData->down, 0,
                           mpiData->comm, MPI_STATUS_IGNORE);
        mgcl::mpi_util::mgclCheckMpiError(mpiData->comm, err, "MPI_Sendrecv");

        if (MPI_PROC_NULL != mpiData->down)
            for (i = 0; i < mgh; i++)
                for (j = 0; j < ghosts_n; j++)
                    for (k = 0; k < ogh; k++)
                        c[i][j][k] = rbuf[i][j][k];

        /* Sending data to the left */
        sbuf_ptr = c.sliceIncGhosts(0, mgh - 1, 0, ngh - 1, ghosts_o, 2 * ghosts_o - 1); // TODO max when gh > m
        sbuf = sbuf_ptr->getData();
        rbuf_ptr = std::make_unique<Cuboid>(sbuf_ptr->getM(), sbuf_ptr->getN(), sbuf_ptr->getO(), 0, 0, 0);
        rbuf = rbuf_ptr->getData();

        // std::cout << myid << "," << mpiData->left << std::endl;
        // MPI_Barrier(comm);
        // sbuf_ptr->dumpToFile("sbuf_ptr_left" + std::to_string(myid) + ".txt");

        err = MPI_Sendrecv(static_cast<void*>(sbuf[0][0]), mgh * ngh * ghosts_o, MPI_DOUBLE, mpiData->left, 0,
                           static_cast<void*>(rbuf[0][0]), mgh * ngh * ghosts_o, MPI_DOUBLE, mpiData->right, 0,
                           mpiData->comm, MPI_STATUS_IGNORE);
        mgcl::mpi_util::mgclCheckMpiError(mpiData->comm, err, "MPI_Sendrecv");

        if (MPI_PROC_NULL != mpiData->right)
            for (i = 0; i < mgh; i++)
                for (j = 0; j < ngh; j++)
                    for (k = 0; k < ghosts_o; k++)
                        c[i][j][ogh - ghosts_o + k] = rbuf[i][j][k];

        /* Sending data to the right */
        sbuf_ptr = c.sliceIncGhosts(0, mgh - 1, 0, ngh - 1, o, o + ghosts_o - 1); // TODO max when gh > m
        sbuf = sbuf_ptr->getData();
        rbuf_ptr = std::make_unique<Cuboid>(sbuf_ptr->getM(), sbuf_ptr->getN(), sbuf_ptr->getO(), 0, 0, 0);
        rbuf = rbuf_ptr->getData();

        err = MPI_Sendrecv(static_cast<void*>(sbuf[0][0]), mgh * ngh * ghosts_o, MPI_DOUBLE, mpiData->right, 0,
                           static_cast<void*>(rbuf[0][0]), mgh * ngh * ghosts_o, MPI_DOUBLE, mpiData->left, 0,
                           mpiData->comm, MPI_STATUS_IGNORE);
        mgcl::mpi_util::mgclCheckMpiError(mpiData->comm, err, "MPI_Sendrecv");

        if (MPI_PROC_NULL != mpiData->left)
            for (i = 0; i < mgh; i++)
                for (j = 0; j < ngh; j++)
                    for (k = 0; k < ghosts_o; k++)
                        c[i][j][k] = rbuf[i][j][k];
    }

    /* updates ghost cells on opencl device.
     * m,n,o must be size of ghosted grid.
     * Only enqueues the kernel. Neither waits for kernel to finish nor reads back results */
    int MultigridEngine::updateGhosts(Problem& problem, CuboidGpu& dBuffer,
                                      MPILevelData* mpiData, bool forceLocal)
    {
        // TODO actually request these as arguments
        int m = dBuffer.getM();
        int n = dBuffer.getN();
        int o = dBuffer.getO();
        int mgh = dBuffer.getMgh();
        int ngh = dBuffer.getNgh();
        int ogh = dBuffer.getOgh();
        int ghosts_m = dBuffer.getGhostsM();
        int ghosts_n = dBuffer.getGhostsN();
        int ghosts_o = dBuffer.getGhostsO();

        if (!forceLocal && problem.useMpi() && mpiData)
        {
            updateGhostsOclMpi(problem.getCommands(), dBuffer, *mpiData, problem.isPeriodic(), forceLocal);
            return CL_SUCCESS;
        }

        if (problem.useMpi() && !mpiData)
            throw "Problem uses MPI but mpiData is null!";

        if (!problem.isPeriodic())
            return CL_SUCCESS;

        int err;

        // Create the compute kernel from the program
        const char* kernelName = "update_ghosts_periodic";
        cl_kernel kernel = clCreateKernel(problem.getOpenCLHelper().getProgram(), kernelName, &err);
        mgclCheckError(err, "clCreateKernel");

        // assign kernel arguments
        int pos = 0;
        err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &dBuffer);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &m);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &n);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &o);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghosts_m);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghosts_n);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghosts_o);
        mgclCheckError(err, "Setting kernel arguments");

        // one work-item per ghost cell (excluding real cells). Pad global sizes to fit to local sizes
        // int mgh = m + 2 * gh;
        // int ngh = n + 2 * gh;
        // int ogh = o + 2 * gh;
        size_t global[3] = {static_cast<size_t>(mgh), static_cast<size_t>(ngh), static_cast<size_t>(ogh)};
        const auto& c = conf::getWorkGroupSizeForKernelAndWiCount(problem.getKernelConfig(), kernelName, 1);
        const size_t local[3] = {
            static_cast<size_t>(mgh > c[0] ? c[0] : mgh),
            static_cast<size_t>(ngh > c[1] ? c[1] : ngh),
            static_cast<size_t>(ogh > c[2] ? c[2] : ogh)};

        for (int i = 0; i < 3; i++)
            if (global[i] % local[i] != 0)
            {
                // printf("padding global size %d from %ld to ", i, global[i]);
                global[i] += local[i] - (global[i] % local[i]);
                // printf("%ld (multiple of %ld)\n", global[i], local[i]);
            }

        // enqueue kernel
        err = clEnqueueNDRangeKernel(problem.getOpenCLHelper().getCommands(), kernel, 3, NULL, global, local, 0, NULL, NULL);
        mgclCheckError(err, "Enqueueing update_ghosts_periodic kernel");

        err = clReleaseKernel(kernel);
        mgclCheckError(err, "Releasing update_ghosts_periodic kernel");

        return err;
    }

    /**
     * @brief Updates ghosts of an OpenCL buffer respecting MPI usage. That is, the buffer is sent to host, ghosts
     * are updated using MPI routines, and the updated buffer is sent back to the device.
     * Waits for previous commands to finish before reading the buffer.
     *
     * @param commands
     * @param d_buf
     * @param mpiData
     * @param m Real grid's size in 1st dim
     * @param n Real grid's size in 2nd dim
     * @param o Real grid's size in 3rd dim
     * @param ghosts_m
     * @param ghosts_n
     * @param ghosts_o
     * @param periodic
     * @param forceLocal
     */
    void MultigridEngine::updateGhostsOclMpi(cl_command_queue commands, CuboidGpu& d_buf, MPILevelData& mpiData,
                                             bool periodic, bool forceLocal)
    {
        // Read back from GPU and update ghosts on host in order to update neighbouring nodes, too.
        auto tmp = d_buf.read(commands, nullptr, true);
        MultigridEngine::updateGhostsSeq(*tmp, &mpiData, periodic, forceLocal);
        d_buf.write(commands, *tmp, true);
    }
}
