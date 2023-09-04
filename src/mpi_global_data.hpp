#ifndef MGCL_MPIGLOBALDATA_HPP
#define MGCL_MPIGLOBALDATA_HPP

#include <memory>

#include "mpi.h"

namespace mgcl
{
    // Global MPI relevant data, e.g. comm size or comm rank.
    class MPIGlobalData
    {
    public:
        MPI_Comm comm;

        MPIGlobalData();
        MPIGlobalData(const MPIGlobalData&) = delete;
        MPIGlobalData& operator=(const MPIGlobalData&) = delete;
        MPIGlobalData(const MPIGlobalData&&) = delete;
        MPIGlobalData& operator=(MPIGlobalData&&) = delete;
        ~MPIGlobalData() {}

        // utility functions
        int mpiRank();
        int mpiSize();
        void createCartGrid(bool periodic);

        MPI_Comm getComm() const;
        void setComm(const MPI_Comm& comm_);
    };
}
#endif // MGCL_MPIGLOBALDATA_HPP
