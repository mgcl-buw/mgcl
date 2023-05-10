#include "mpi_data.hpp"

/**
 * @brief Construct a new mgcl::MPIData object
 *
 * @param _comm MPI communicator
 * @param m Dimension m of Cuboids for this process including ghosts.
 * @param n Dimension n of Cuboids for this process including ghosts.
 * @param o Dimension o of Cuboids for this process including ghosts.
 */
mgcl::MPIData::MPIData(MPI_Comm _comm, int m, int n, int o)
    : comm(_comm)
{
    _sbufxy = std::make_unique<Cuboid>(1, m, n);
    _sbufxz = std::make_unique<Cuboid>(1, m, o);
    _sbufyz = std::make_unique<Cuboid>(1, n, o);
    _rbufxy = std::make_unique<Cuboid>(1, m, n);
    _rbufxz = std::make_unique<Cuboid>(1, m, o);
    _rbufyz = std::make_unique<Cuboid>(1, n, o);
}
