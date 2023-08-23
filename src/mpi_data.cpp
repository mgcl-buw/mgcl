#include "mpi_data.hpp"

#include "mpi_util.hpp"

/**
 * @brief Construct a new mgcl::MPIData object
 *
 * @param _comm MPI communicator
 */
mgcl::MPIData::MPIData(MPI_Comm _comm)
    : comm(_comm)
{
    MPI_Comm_rank(_comm, &rank);
}

int mgcl::MPIData::mpiSize()
{
    int mpi_size;
    int err = MPI_Comm_size(comm, &mpi_size);
    mpi_util::mgclCheckMpiError(comm, err, "MPI_Comm_size");
    return mpi_size;
}
