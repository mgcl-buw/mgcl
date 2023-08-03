#ifndef MGCL_MPIDATA_HPP
#define MGCL_MPIDATA_HPP

#include <memory>

#include "cuboid.hpp"

#ifdef MGCL_USE_MPI
#include "mpi.h"
#endif // MGCL_USE_MPI

namespace mgcl
{
#ifdef MGCL_USE_MPI
    class MPIData
    {
    public:
        MPI_Comm comm;

        // rank of this process
        int rank;

        // IDs of neighbouring processes
        int left;
        int right;
        int up;
        int down;
        int front;
        int back;

        // Starting values for this process (end can be calculated using attached level).
        int xstart;
        int ystart;
        int zstart;

        MPIData(MPI_Comm _comm);
        MPIData(const MPIData &) = delete;
        MPIData &operator=(const MPIData &) = delete;
        MPIData(const MPIData &&) = delete;
        MPIData &operator=(MPIData &&) = delete;
        ~MPIData() {}

        // utility functions
        int mpiSize();

        static void mgcl_check_mpi_error(MPI_Comm comm, int err, const char *operation, const char *filename, int line);
    };

#define mgclCheckMpiError(C, E, S) MPIData::mgcl_check_mpi_error(C, E, S, __FILE__, __LINE__);
#else
    class MPIData
    {
    }; // Just a stub
#endif
}
#endif // MGCL_MPIDATA_HPP
