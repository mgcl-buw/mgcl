#ifndef MGCL_MPI_STENCIL_HPP
#define MGCL_MPI_STENCIL_HPP

#include "mpi_level_data.hpp"
#include "opencl_helper.hpp"
#include "stencil.hpp"

#ifdef __APPLE__
#include <OpenCL/cl_platform.h>
#else
#include <CL/cl_platform.h>
#endif

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

        /* Loop variables */
        int i, j, k;

        // Size of one stencil
        int ssize = s.getDim4gh() * s.getDim5gh() * s.getDim6gh();

        /* Getting local rank */
        int myid;
        MPI_Comm_rank(mpiData->comm, &myid);

        /* Sending data to the front */
        auto sbuf_ptr = s.sliceIncGhosts(ghosts_m, 2 * ghosts_m - 1, 0, ngh - 1, 0, ogh - 1); // TODO max when gh > m
        auto sbuf = sbuf_ptr->getData();
        auto rbuf_ptr = sbuf_ptr->copyShallow();
        auto rbuf = rbuf_ptr->getData();

        MPI_Sendrecv(static_cast<void*>(sbuf[0][0][0][0][0]), ghosts_m * ngh * ogh * ssize, MPI_DOUBLE, mpiData->front, 0,
                     static_cast<void*>(rbuf[0][0][0][0][0]), ghosts_m * ngh * ogh * ssize, MPI_DOUBLE, mpiData->back, 0,
                     mpiData->comm, MPI_STATUS_IGNORE);

        if (MPI_PROC_NULL != mpiData->back)
            for (i = 0; i < ghosts_m; i++)
                for (j = 0; j < ngh; j++)
                    for (k = 0; k < ogh; k++)
                        for (int ii = 0; ii < s.getDim4(); ii++)
                            for (int jj = 0; jj < s.getDim5(); jj++)
                                for (int kk = 0; kk < s.getDim6(); kk++)
                                    s[mgh - ghosts_m + i][j][k][ii][jj][kk] = rbuf[i][j][k][ii][jj][kk];

        /* Sending data to the back */
        sbuf_ptr = s.sliceIncGhosts(m, m + ghosts_m - 1, 0, ngh - 1, 0, ogh - 1); // TODO max when gh > m
        sbuf = sbuf_ptr->getData();
        rbuf_ptr = sbuf_ptr->copyShallow();
        rbuf = rbuf_ptr->getData();

        MPI_Sendrecv(static_cast<void*>(sbuf[0][0][0][0][0]), ghosts_m * ngh * ogh * ssize, MPI_DOUBLE, mpiData->back, 0,
                     static_cast<void*>(rbuf[0][0][0][0][0]), ghosts_m * ngh * ogh * ssize, MPI_DOUBLE, mpiData->front, 0,
                     mpiData->comm, MPI_STATUS_IGNORE);

        if (MPI_PROC_NULL != mpiData->front)
            for (i = 0; i < ghosts_m; i++)
                for (j = 0; j < ngh; j++)
                    for (k = 0; k < ogh; k++)
                        for (int ii = 0; ii < s.getDim4(); ii++)
                            for (int jj = 0; jj < s.getDim5(); jj++)
                                for (int kk = 0; kk < s.getDim6(); kk++)
                                    s[i][j][k][ii][jj][kk] = rbuf[i][j][k][ii][jj][kk];

        /* Sending data downwards */
        sbuf_ptr = s.sliceIncGhosts(0, mgh - 1, ghosts_m, 2 * ghosts_n - 1, 0, ogh - 1); // TODO max when gh > m
        sbuf = sbuf_ptr->getData();
        rbuf_ptr = sbuf_ptr->copyShallow();
        rbuf = rbuf_ptr->getData();

        MPI_Sendrecv(static_cast<void*>(sbuf[0][0][0][0][0]), mgh * ghosts_n * ogh * ssize, MPI_DOUBLE, mpiData->down, 0,
                     static_cast<void*>(rbuf[0][0][0][0][0]), mgh * ghosts_n * ogh * ssize, MPI_DOUBLE, mpiData->up, 0,
                     mpiData->comm, MPI_STATUS_IGNORE);

        if (MPI_PROC_NULL != mpiData->up)
            for (i = 0; i < mgh; i++)
                for (j = 0; j < ghosts_n; j++)
                    for (k = 0; k < ogh; k++)
                        for (int ii = 0; ii < s.getDim4(); ii++)
                            for (int jj = 0; jj < s.getDim5(); jj++)
                                for (int kk = 0; kk < s.getDim6(); kk++)
                                    s[i][ngh - ghosts_n + j][k][ii][jj][kk] = rbuf[i][j][k][ii][jj][kk];

        /* Sending data upwards */
        sbuf_ptr = s.sliceIncGhosts(0, mgh - 1, n, n + ghosts_n - 1, 0, ogh - 1); // TODO max when gh > m
        sbuf = sbuf_ptr->getData();
        rbuf_ptr = sbuf_ptr->copyShallow();
        rbuf = rbuf_ptr->getData();

        MPI_Sendrecv(static_cast<void*>(sbuf[0][0][0][0][0]), mgh * ghosts_n * ogh * ssize, MPI_DOUBLE, mpiData->up, 0,
                     static_cast<void*>(rbuf[0][0][0][0][0]), mgh * ghosts_n * ogh * ssize, MPI_DOUBLE, mpiData->down, 0,
                     mpiData->comm, MPI_STATUS_IGNORE);

        if (MPI_PROC_NULL != mpiData->down)
            for (i = 0; i < mgh; i++)
                for (j = 0; j < ghosts_n; j++)
                    for (k = 0; k < ogh; k++)
                        for (int ii = 0; ii < s.getDim4(); ii++)
                            for (int jj = 0; jj < s.getDim5(); jj++)
                                for (int kk = 0; kk < s.getDim6(); kk++)
                                    s[i][j][k][ii][jj][kk] = rbuf[i][j][k][ii][jj][kk];

        /* Sending data to the left */
        sbuf_ptr = s.sliceIncGhosts(0, mgh - 1, 0, ngh - 1, ghosts_o, 2 * ghosts_o - 1); // TODO max when gh > m
        sbuf = sbuf_ptr->getData();
        rbuf_ptr = sbuf_ptr->copyShallow();
        rbuf = rbuf_ptr->getData();

        // std::cout << myid << "," << mpiData->left << std::endl;
        // MPI_Barrier(comm);
        // sbuf_ptr->dumpToFile("sbuf_ptr_left" + std::to_string(myid) + ".txt");

        MPI_Sendrecv(static_cast<void*>(sbuf[0][0][0][0][0]), mgh * ngh * ghosts_o * ssize, MPI_DOUBLE, mpiData->left, 0,
                     static_cast<void*>(rbuf[0][0][0][0][0]), mgh * ngh * ghosts_o * ssize, MPI_DOUBLE, mpiData->right, 0,
                     mpiData->comm, MPI_STATUS_IGNORE);
        if (MPI_PROC_NULL != mpiData->right)
            for (i = 0; i < mgh; i++)
                for (j = 0; j < ngh; j++)
                    for (k = 0; k < ghosts_o; k++)
                        for (int ii = 0; ii < s.getDim4(); ii++)
                            for (int jj = 0; jj < s.getDim5(); jj++)
                                for (int kk = 0; kk < s.getDim6(); kk++)
                                    s[i][j][ogh - ghosts_o + k][ii][jj][kk] = rbuf[i][j][k][ii][jj][kk];

        /* Sending data to the right */
        sbuf_ptr = s.sliceIncGhosts(0, mgh - 1, 0, ngh - 1, o, o + ghosts_o - 1); // TODO max when gh > m
        sbuf = sbuf_ptr->getData();
        rbuf_ptr = sbuf_ptr->copyShallow();
        rbuf = rbuf_ptr->getData();

        MPI_Sendrecv(static_cast<void*>(sbuf[0][0][0][0][0]), mgh * ngh * ghosts_o * ssize, MPI_DOUBLE, mpiData->right, 0,
                     static_cast<void*>(rbuf[0][0][0][0][0]), mgh * ngh * ghosts_o * ssize, MPI_DOUBLE, mpiData->left, 0,
                     mpiData->comm, MPI_STATUS_IGNORE);

        if (MPI_PROC_NULL != mpiData->left)
            for (i = 0; i < mgh; i++)
                for (j = 0; j < ngh; j++)
                    for (k = 0; k < ghosts_o; k++)
                        for (int ii = 0; ii < s.getDim4(); ii++)
                            for (int jj = 0; jj < s.getDim5(); jj++)
                                for (int kk = 0; kk < s.getDim6(); kk++)
                                    s[i][j][k][ii][jj][kk] = rbuf[i][j][k][ii][jj][kk];
    }

    /**
     * @brief Updates ghosts of an VaryingStencilGpu respecting MPI usage. That is, the buffer is sent to host, ghosts
     * are updated using MPI routines, and the updated buffer is sent back to the device.
     * Waits for previous commands to finish before reading the buffer.
     * Currently only stencil widths of 3 and 5 are supported.
     *
     * @param commands
     * @param periodic
     * @param forceLocal
     */
    void updateGhostsStencilOclMpi(cl_program program, cl_command_queue commands, VaryingStencilGpu& s,
                                   MPILevelData& mpiData, bool periodic, bool forceLocal)
    {
        if (s.getWidth() == 3)
        {
            auto tmp = s.read<3>(commands);
            mgclCheckError(clFinish(commands), "clFinish");
            updateGhostsStencilMpi<3>(tmp, &mpiData, periodic, forceLocal);
            s.fill<3>(tmp, commands);
            mgclCheckError(clFinish(commands), "clFinish");
        }
        else if (s.getWidth() == 5)
        {
            auto tmp = s.read<5>(commands);
            mgclCheckError(clFinish(commands), "clFinish");
            updateGhostsStencilMpi<5>(tmp, &mpiData, periodic, forceLocal);
            s.fill<5>(tmp, commands);
            mgclCheckError(clFinish(commands), "clFinish");
        }
        else
            throw "updateGhostsStencilOclMpi is only supported for stencils widths 3 and 5!";
    }
}

#endif // MGCL_MPI_STENCIL_HPP
