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
    void updateGhostsStencilMpi(VaryingStencil& s, MPILevelData* mpiData, bool periodic, bool forceLocal);

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
    void updateGhostsStencilOclMpi(
        cl_command_queue commands, cl_program program,
        VaryingStencilGpu& s,
        BufferGpu& d_planes_buf,
        std::vector<double>& h_buf,
        MPILevelData* mpiData,
        bool forceLocal,
        conf::KernelConfig* conf, ProfilingData* pd);
}

#endif // MGCL_MPI_STENCIL_HPP
