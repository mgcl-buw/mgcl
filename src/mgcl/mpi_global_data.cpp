#include "mpi_global_data.hpp"

#include "mpi_util.hpp"

/**
 * @brief Construct a new mgcl::MPIGlobalData object with MPI_COMM_WORLD as communicator.
 */
mgcl::MPIGlobalData::MPIGlobalData() : comm(MPI_COMM_WORLD) {}

/**
 * @brief Returns the process rank.
 *
 * @return int
 */
int mgcl::MPIGlobalData::mpiRank()
{
    int ret;
    int err = MPI_Comm_rank(comm, &ret);
    mpi_util::mgclCheckMpiError(comm, err, "MPI_Comm_rank");
    return ret;
}

/**
 * @brief Returns the communicator size, i.e. number of processes attached to the given communicator.
 *
 * @return int
 */
int mgcl::MPIGlobalData::mpiSize()
{
    int ret;
    int err = MPI_Comm_size(comm, &ret);
    mpi_util::mgclCheckMpiError(comm, err, "MPI_Comm_size");
    return ret;
}

/**
 * @brief Create cartesian process grid if none was set and more than one processes are used.
 *
 * @param periodic Whether the grid shall be periodic.
 */
void mgcl::MPIGlobalData::createCartGrid(bool periodic)
{
    int type;
    int mpi_dims[3] = {0, 0, 0};
    int mpi_periods[3] = {periodic, periodic, periodic};

    int err = MPI_Topo_test(comm, &type);
    mpi_util::mgclCheckMpiError(comm, err, "MPI_Topo_test");
    if (mpiSize() > 1 && type != MPI_CART)
    {
        err = MPI_Dims_create(mpiSize(), 3, mpi_dims);
        mpi_util::mgclCheckMpiError(comm, err, "MPI_Dims_create");
        err = MPI_Cart_create(comm, 3, mpi_dims, mpi_periods, 1, &comm);
        mpi_util::mgclCheckMpiError(comm, err, "MPI_Cart_create");
    }
}

MPI_Comm mgcl::MPIGlobalData::getComm() const
{
    return comm;
}

void mgcl::MPIGlobalData::setComm(const MPI_Comm& comm_)
{
    comm = comm_;

    int type;
    int err = MPI_Topo_test(comm_, &type);
    mpi_util::mgclCheckMpiError(comm_, err, "MPI_Topo_test");

    if (type != MPI_CART)
        throw "MPI Comm must have a cartesian topology attached!";
}
