#include "mpi_data.hpp"

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
    mgcl::mgclCheckMpiError(comm, err, "MPI_Comm_size");
    return mpi_size;
}

void mgcl::MPIData::mgcl_check_mpi_error(MPI_Comm comm, int err, const char *operation, const char *filename, int line)
{
    if (err != MPI_SUCCESS)
    {
        fprintf(stderr, "Error during operation '%s', ", operation);
        fprintf(stderr, "in '%s' on line %d\n", filename, line);
        fprintf(stderr, "Error code was %d\n", err);
        MPI_Abort(comm, err);
    }
}
