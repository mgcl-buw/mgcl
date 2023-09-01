#ifndef PMG_UTILITY_HPP
#define PMG_UTILITY_HPP

#include "mpi.h"

inline MPI_Comm* init_mpi_for_pmg()
{
    // setup MPI
    int mpi_size;
    int mpi_rank;
    int mpi_dims[3] = {1, 1, 1};
    int mpi_periods[3] = {1, 1, 1};
    int mpi_coords[3];
    MPI_Comm* mpi_comm_cart = (MPI_Comm*)malloc(sizeof(MPI_Comm));

    // init only once
    int initialized;
    MPI_Initialized(&initialized);

    if (!initialized)
        MPI_Init(NULL, NULL);

    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Dims_create(mpi_size, 3, mpi_dims);
    MPI_Cart_create(MPI_COMM_WORLD, 3, mpi_dims, mpi_periods, 1, mpi_comm_cart);
    MPI_Cart_coords(*mpi_comm_cart, mpi_rank, 3, mpi_coords);

    return mpi_comm_cart;
}

#endif // PMG_UTILITY_HPP
