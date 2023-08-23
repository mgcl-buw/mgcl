#ifndef MGCL_MPIUTIL_HPP
#define MGCL_MPIUTIL_HPP

#include <memory>

#include "mpi.h"

#include "cuboid.hpp"

namespace mgcl::mpi_util
{
    void gather(MPI_Comm comm, Cuboid &send, Cuboid *recv_ptr);
}
#endif // MGCL_MPIUTIL_HPP
