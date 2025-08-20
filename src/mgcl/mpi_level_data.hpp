#ifndef MGCL_MPILEVELDATA_HPP
#define MGCL_MPILEVELDATA_HPP

#include <cassert>
#include <memory>
#include <vector>

#include "mpi.h"

namespace mgcl
{
    // MPI relevant data for each level
    class MPILevelData
    {
    public:
        const MPI_Comm comm;

        // rank of this process
        int rank;

        // number of neighbors in each direction
        const int ghosts;

        // Ranks of neighbouring processes. Closest one will be at index 0.
        std::vector<int> left;
        std::vector<int> right;
        std::vector<int> up;
        std::vector<int> down;
        std::vector<int> front;
        std::vector<int> back;

        MPILevelData(MPI_Comm _comm, int ghosts);
        MPILevelData(const MPILevelData&) = delete;
        MPILevelData& operator=(const MPILevelData&) = delete;
        MPILevelData(const MPILevelData&&) = delete;
        MPILevelData& operator=(MPILevelData&&) = delete;
        ~MPILevelData() {}

        // utility functions
        int mpiSize();
        void printNeighbours();
    };
}
#endif // MGCL_MPILEVELDATA_HPP
