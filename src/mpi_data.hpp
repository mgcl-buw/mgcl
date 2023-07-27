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
    private:
        // Rectangle buffers for sending and receiving data (Rectangle = Cuboid with m = 1).
        std::unique_ptr<Cuboid> _sbufxy = nullptr;
        std::unique_ptr<Cuboid> _sbufxz = nullptr;
        std::unique_ptr<Cuboid> _sbufyz = nullptr;
        std::unique_ptr<Cuboid> _rbufxy = nullptr;
        std::unique_ptr<Cuboid> _rbufxz = nullptr;
        std::unique_ptr<Cuboid> _rbufyz = nullptr;

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

        MPIData(MPI_Comm _comm, int mgh, int ngh, int ogh);
        MPIData(const MPIData &) = delete;
        MPIData &operator=(const MPIData &) = delete;
        MPIData(const MPIData &&) = delete;
        MPIData &operator=(MPIData &&) = delete;
        ~MPIData() {}

        // Access to Rectangle buffers (2d view of Cuboid)
        inline double **sbufxy() { return (*_sbufxy)[0]; };
        inline double **sbufxz() { return (*_sbufxz)[0]; };
        inline double **sbufyz() { return (*_sbufyz)[0]; };
        inline double **rbufxy() { return (*_rbufxy)[0]; };
        inline double **rbufxz() { return (*_rbufxz)[0]; };
        inline double **rbufyz() { return (*_rbufyz)[0]; };

        // utility functions
        bool mpiSize();

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
