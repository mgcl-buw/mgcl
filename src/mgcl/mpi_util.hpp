#ifndef MGCL_MPIUTIL_HPP
#define MGCL_MPIUTIL_HPP

#include <memory>

#include "mpi.h"

#include "cuboid.hpp"
#include "cuboid_gpu.hpp"
#include "stencil.hpp"

namespace mgcl::mpi_util
{
    void gather(MPI_Comm comm, Cuboid& c);
    void gather(MPI_Comm comm, cl_command_queue commands, CuboidGpu& c);
    void gather(MPI_Comm comm, VaryingStencil& c);
    void gather(MPI_Comm comm, cl_command_queue commands, VaryingStencilGpu& c);
    void scatter(MPI_Comm comm, Cuboid* src, Cuboid& dest);
    void scatter_inplace(MPI_Comm comm, Cuboid& c);
    void scatter_inplace_wgh(MPI_Comm comm, Cuboid& c);
    void scatter_inplace_wgh(MPI_Comm comm, cl_command_queue commands, CuboidGpu& c);

    void sendBorderPlanes(int mgh, int ngh, int ogh, int ghosts_m, int ghosts_n, int ghosts_o,
                          Cuboid& sbuf, Cuboid& rbuf, MPILevelData& mpiData);

    void mgcl_check_mpi_error(MPI_Comm comm, int err, const char* operation, const char* filename, int line);
}

#define mgclCheckMpiError(C, E, S) mgcl_check_mpi_error(C, E, S, __FILE__, __LINE__);

#endif // MGCL_MPIUTIL_HPP
