#ifndef MGCL_MPIDATA_HPP
#define MGCL_MPIDATA_HPP

#include <memory>

#include "mpi.h"

namespace mgcl
{
    // MPI relevant data for each level
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

        MPIData(MPI_Comm _comm);
        MPIData(const MPIData &) = delete;
        MPIData &operator=(const MPIData &) = delete;
        MPIData(const MPIData &&) = delete;
        MPIData &operator=(MPIData &&) = delete;
        ~MPIData() {}

        // utility functions
        int mpiSize();
    };
}
#endif // MGCL_MPIDATA_HPP
