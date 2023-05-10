#ifndef MGCL_MPIDATA_HPP
#define MGCL_MPIDATA_HPP

#ifdef MGCL_USE_MPI
#include "mpi.h"
#endif // MGCL_USE_MPI

namespace mgcl
{
    class MPIData
    {
    public:
#ifdef MGCL_USE_MPI
        static MPI_Comm comm;
#endif

        int id;
        int left;
        int right;
        int top;
        int down;
        int front;
        int back;

        int xstart;
        int ystart;
        int zstart;
    };
}
#endif // MGCL_MPIDATA_HPP
