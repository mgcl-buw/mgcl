#ifndef MGCL_MPIUTIL_HPP
#define MGCL_MPIUTIL_HPP

#include <memory>

#include "mpi.h"

#include "cuboid.hpp"

namespace mgcl::mpi_util
{
    void gather(MPI_Comm comm, Cuboid& c);
    void scatter(MPI_Comm comm, Cuboid* src, Cuboid& dest);
    void scatter_inplace(MPI_Comm comm, Cuboid& c);

    void mgcl_check_mpi_error(MPI_Comm comm, int err, const char* operation, const char* filename, int line);
}

#define mgclCheckMpiError(C, E, S) mgcl_check_mpi_error(C, E, S, __FILE__, __LINE__);

#endif // MGCL_MPIUTIL_HPP
