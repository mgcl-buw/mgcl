#ifndef MGCL_MPI_STENCIL_HPP
#define MGCL_MPI_STENCIL_HPP

#include "mpi_stencil.hpp"
#include "kernel_config.hpp"
#include "mgcl.hpp"
#include "mpi_level_data.hpp"
#include "mpi_util.hpp"
#include "opencl_helper.hpp"
#include "profiling_data.hpp"
#include "stencil.hpp"
#include <CL/cl.h>

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
     *   This is used e.g. for levels above the mpiLevelThreshold.
     * The VaryingStencil must have the same sizes for each process. However this is not checked here. */
    void updateGhostsStencilMpi(VaryingStencil& s, MPILevelData* mpiData, bool periodic, bool forceLocal)
    {
        // do nothing if single-gpu and Dirichlet bc's
        if (!periodic && (mpiData == nullptr || mpiData->mpiSize() == 1 || forceLocal))
            return;

        // TODO adjust for ghosts > 1
        if (forceLocal || mpiData == nullptr || mpiData->mpiSize() == 1)
        {
            if (periodic)
                s.updateGhosts();
            return;
        }

        int m = s.getM();
        int n = s.getN();
        int o = s.getO();
        int ghosts_m = s.getGhostsM();
        int ghosts_n = s.getGhostsN();
        int ghosts_o = s.getGhostsO();
        int mgh = s.getMgh();
        int ngh = s.getNgh();
        int ogh = s.getOgh();

        if (m < ghosts_m || n < ghosts_n || o < ghosts_o)
            error("Ghost-update using MPI is only allowed for gh <= m,n,o.");

        /* Loop variables */
        int i, j, k;
        int err;

        // Size of one stencil
        int ssize = s.getWidth() * s.getWidth() * s.getWidth();

        /* Getting local rank */
        int myid;
        MPI_Comm_rank(mpiData->comm, &myid);

        /* Sending data to the front */
        auto sbuf_ptr = s.sliceIncGhosts(ghosts_m, 2 * ghosts_m - 1, 0, ngh - 1, 0, ogh - 1); // TODO max when gh > m
        auto sbuf = sbuf_ptr->getData();
        auto rbuf_ptr = sbuf_ptr->copyShallow();
        auto rbuf = rbuf_ptr->getData();

        err = MPI_Sendrecv(static_cast<void*>(sbuf[0][0][0][0][0]), ghosts_m * ngh * ogh * ssize, MPI_DOUBLE, mpiData->front[0], 0,
                           static_cast<void*>(rbuf[0][0][0][0][0]), ghosts_m * ngh * ogh * ssize, MPI_DOUBLE, mpiData->back[0], 0,
                           mpiData->comm, MPI_STATUS_IGNORE);
        mgcl::mpi_util::mgclCheckMpiError(mpiData->comm, err, "MPI_Sendrecv");

        if (MPI_PROC_NULL != mpiData->back[0])
            for (i = 0; i < ghosts_m; i++)
                for (j = 0; j < ngh; j++)
                    for (k = 0; k < ogh; k++)
                        for (int ii = 0; ii < s.getWidth(); ii++)
                            for (int jj = 0; jj < s.getWidth(); jj++)
                                for (int kk = 0; kk < s.getWidth(); kk++)
                                    s[ii][jj][kk][mgh - ghosts_m + i][j][k] = rbuf[ii][jj][kk][i][j][k];

        /* Sending data to the back */
        sbuf_ptr = s.sliceIncGhosts(m, m + ghosts_m - 1, 0, ngh - 1, 0, ogh - 1); // TODO max when gh > m
        sbuf = sbuf_ptr->getData();
        rbuf_ptr = sbuf_ptr->copyShallow();
        rbuf = rbuf_ptr->getData();

        err = MPI_Sendrecv(static_cast<void*>(sbuf[0][0][0][0][0]), ghosts_m * ngh * ogh * ssize, MPI_DOUBLE, mpiData->back[0], 0,
                           static_cast<void*>(rbuf[0][0][0][0][0]), ghosts_m * ngh * ogh * ssize, MPI_DOUBLE, mpiData->front[0], 0,
                           mpiData->comm, MPI_STATUS_IGNORE);
        mgcl::mpi_util::mgclCheckMpiError(mpiData->comm, err, "MPI_Sendrecv");

        if (MPI_PROC_NULL != mpiData->front[0])
            for (i = 0; i < ghosts_m; i++)
                for (j = 0; j < ngh; j++)
                    for (k = 0; k < ogh; k++)
                        for (int ii = 0; ii < s.getWidth(); ii++)
                            for (int jj = 0; jj < s.getWidth(); jj++)
                                for (int kk = 0; kk < s.getWidth(); kk++)
                                    s[ii][jj][kk][i][j][k] = rbuf[ii][jj][kk][i][j][k];

        /* Sending data upwards */
        sbuf_ptr = s.sliceIncGhosts(0, mgh - 1, ghosts_m, 2 * ghosts_n - 1, 0, ogh - 1); // TODO max when gh > m
        sbuf = sbuf_ptr->getData();
        rbuf_ptr = sbuf_ptr->copyShallow();
        rbuf = rbuf_ptr->getData();

        err = MPI_Sendrecv(static_cast<void*>(sbuf[0][0][0][0][0]), mgh * ghosts_n * ogh * ssize, MPI_DOUBLE, mpiData->up[0], 0,
                           static_cast<void*>(rbuf[0][0][0][0][0]), mgh * ghosts_n * ogh * ssize, MPI_DOUBLE, mpiData->down[0], 0,
                           mpiData->comm, MPI_STATUS_IGNORE);
        mgcl::mpi_util::mgclCheckMpiError(mpiData->comm, err, "MPI_Sendrecv");

        if (MPI_PROC_NULL != mpiData->down[0])
            for (i = 0; i < mgh; i++)
                for (j = 0; j < ghosts_n; j++)
                    for (k = 0; k < ogh; k++)
                        for (int ii = 0; ii < s.getWidth(); ii++)
                            for (int jj = 0; jj < s.getWidth(); jj++)
                                for (int kk = 0; kk < s.getWidth(); kk++)
                                    s[ii][jj][kk][i][ngh - ghosts_n + j][k] = rbuf[ii][jj][kk][i][j][k];

        /* Sending data downwards */
        sbuf_ptr = s.sliceIncGhosts(0, mgh - 1, n, n + ghosts_n - 1, 0, ogh - 1); // TODO max when gh > m
        sbuf = sbuf_ptr->getData();
        rbuf_ptr = sbuf_ptr->copyShallow();
        rbuf = rbuf_ptr->getData();

        err = MPI_Sendrecv(static_cast<void*>(sbuf[0][0][0][0][0]), mgh * ghosts_n * ogh * ssize, MPI_DOUBLE, mpiData->down[0], 0,
                           static_cast<void*>(rbuf[0][0][0][0][0]), mgh * ghosts_n * ogh * ssize, MPI_DOUBLE, mpiData->up[0], 0,
                           mpiData->comm, MPI_STATUS_IGNORE);
        mgcl::mpi_util::mgclCheckMpiError(mpiData->comm, err, "MPI_Sendrecv");

        if (MPI_PROC_NULL != mpiData->up[0])
            for (i = 0; i < mgh; i++)
                for (j = 0; j < ghosts_n; j++)
                    for (k = 0; k < ogh; k++)
                        for (int ii = 0; ii < s.getWidth(); ii++)
                            for (int jj = 0; jj < s.getWidth(); jj++)
                                for (int kk = 0; kk < s.getWidth(); kk++)
                                    s[ii][jj][kk][i][j][k] = rbuf[ii][jj][kk][i][j][k];

        /* Sending data to the left */
        sbuf_ptr = s.sliceIncGhosts(0, mgh - 1, 0, ngh - 1, ghosts_o, 2 * ghosts_o - 1); // TODO max when gh > m
        sbuf = sbuf_ptr->getData();
        rbuf_ptr = sbuf_ptr->copyShallow();
        rbuf = rbuf_ptr->getData();

        // std::cout << myid << "," << mpiData->left[0] << std::endl;
        // MPI_Barrier(comm);
        // sbuf_ptr->dumpToFile("sbuf_ptr_left" + std::to_string(myid) + ".txt");

        err = MPI_Sendrecv(static_cast<void*>(sbuf[0][0][0][0][0]), mgh * ngh * ghosts_o * ssize, MPI_DOUBLE, mpiData->left[0], 0,
                           static_cast<void*>(rbuf[0][0][0][0][0]), mgh * ngh * ghosts_o * ssize, MPI_DOUBLE, mpiData->right[0], 0,
                           mpiData->comm, MPI_STATUS_IGNORE);
        mgcl::mpi_util::mgclCheckMpiError(mpiData->comm, err, "MPI_Sendrecv");
        if (MPI_PROC_NULL != mpiData->right[0])
            for (i = 0; i < mgh; i++)
                for (j = 0; j < ngh; j++)
                    for (k = 0; k < ghosts_o; k++)
                        for (int ii = 0; ii < s.getWidth(); ii++)
                            for (int jj = 0; jj < s.getWidth(); jj++)
                                for (int kk = 0; kk < s.getWidth(); kk++)
                                    s[ii][jj][kk][i][j][ogh - ghosts_o + k] = rbuf[ii][jj][kk][i][j][k];

        /* Sending data to the right */
        sbuf_ptr = s.sliceIncGhosts(0, mgh - 1, 0, ngh - 1, o, o + ghosts_o - 1); // TODO max when gh > m
        sbuf = sbuf_ptr->getData();
        rbuf_ptr = sbuf_ptr->copyShallow();
        rbuf = rbuf_ptr->getData();

        err = MPI_Sendrecv(static_cast<void*>(sbuf[0][0][0][0][0]), mgh * ngh * ghosts_o * ssize, MPI_DOUBLE, mpiData->right[0], 0,
                           static_cast<void*>(rbuf[0][0][0][0][0]), mgh * ngh * ghosts_o * ssize, MPI_DOUBLE, mpiData->left[0], 0,
                           mpiData->comm, MPI_STATUS_IGNORE);
        mgcl::mpi_util::mgclCheckMpiError(mpiData->comm, err, "MPI_Sendrecv");

        if (MPI_PROC_NULL != mpiData->left[0])
            for (i = 0; i < mgh; i++)
                for (j = 0; j < ngh; j++)
                    for (k = 0; k < ghosts_o; k++)
                        for (int ii = 0; ii < s.getWidth(); ii++)
                            for (int jj = 0; jj < s.getWidth(); jj++)
                                for (int kk = 0; kk < s.getWidth(); kk++)
                                    s[ii][jj][kk][i][j][k] = rbuf[ii][jj][kk][i][j][k];
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
     * @param conf Optional kernel config, if local ghost update is done.
     */
    void updateGhostsStencilOclMpi(
        cl_command_queue commands, cl_program program,
        VaryingStencilGpu& s,
        BufferGpu& d_planes_buf,
        std::vector<double>& sbuf, std::vector<double>& rbuf,
        MPILevelData* mpiData, bool forceLocal, bool periodic,
        conf::KernelConfig* conf, ProfilingData* pd)
    {
        // do nothing if single-gpu and Dirichlet bc's
        if (!periodic && (mpiData == nullptr || mpiData->mpiSize() == 1 || forceLocal))
            return;

        // TODO adjust for ghosts > 1
        if (forceLocal || mpiData == nullptr || mpiData->mpiSize() == 1)
        {
            if (periodic)
                s.updateGhosts(program, commands, conf, pd);
            return;
        }

        // // TODO optimize, like for cuboids
        // auto tmp = s.read(commands, true);
        // updateGhostsStencilMpi(tmp, mpiData, true, forceLocal);
        // s.fill(tmp, commands, true);
        // return;

        // Use temporary buffer for extracting and pasting planes. Check if it's large enough beforehand.
        // TODO maybe disable check in UNSAFE mode
        int yz = s.getNgh() * s.getOgh();
        int xz = s.getMgh() * s.getOgh();
        int xy = s.getMgh() * s.getNgh();
        size_t ressize = (2 * yz * s.getGh() + 2 * xz * s.getGh() + 2 * xy * s.getGh()) * s.getWidth() * s.getWidth() * s.getWidth();

        if (d_planes_buf.getSize() < ressize)
            error("MultigridEngine::updateGhostsOclMpi: d_planes_buf is too small. Need at least " + std::to_string(ressize) + ", but is " + std::to_string(d_planes_buf.getSize()));

        if (sbuf.size() < ressize || rbuf.size() < ressize)
            throw "MultigridEngine::updateGhostsOclMpi: sbuf or rbuf is too small. Need at least " +
                std::to_string(ressize) + ", but is " + std::to_string(sbuf.size()) +
                " (send) and " + std::to_string(rbuf.size()) + " (recv)";

        // std::fill(sbuf.begin(), sbuf.end(), 0.0);
        // std::fill(rbuf.begin(), rbuf.end(), 0.0);

        // Extract border planes from the buffer
        s.extractBorderPlanes(commands, program,
                              d_planes_buf, sbuf,
                              conf, pd);
        mgclCheckError(clFinish(commands), "clFinish");

        // Send our planes to neighbours and receive their planes
        mpi_util::sendBorderPlanes(s.getMgh(), s.getNgh(), s.getOgh(),
                                   s.getGh(), s.getGh(), s.getGh(), s.getWidth(),
                                   sbuf, rbuf, *mpiData);

        // Paste planes back into the buffer.
        d_planes_buf.write(commands, rbuf, true, ressize, pd);
        s.pasteGhostsFromBorderPlanes(commands, program,
                                      d_planes_buf,
                                      conf, pd);
    }
}

#endif // MGCL_MPI_STENCIL_HPP
