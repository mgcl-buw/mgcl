#ifndef MGCL_MPILEVELDATA_HPP
#define MGCL_MPILEVELDATA_HPP

#include <memory>

#include "mpi.h"

namespace mgcl
{
    // MPI relevant data for each level
    class MPILevelData
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

        MPILevelData(MPI_Comm _comm);
        MPILevelData(const MPILevelData&) = delete;
        MPILevelData& operator=(const MPILevelData&) = delete;
        MPILevelData(const MPILevelData&&) = delete;
        MPILevelData& operator=(MPILevelData&&) = delete;
        ~MPILevelData() {}

        // utility functions
        int mpiSize();
    };
}
#endif // MGCL_MPILEVELDATA_HPP
