#include "mpi_data.hpp"

/**
 * @brief Construct a new mgcl::MPIData object
 *
 * @param _comm MPI communicator
 * @param mgh Dimension mgh of Cuboids for this process including ghosts.
 * @param ngh Dimension ngh of Cuboids for this process including ghosts.
 * @param ogh Dimension ogh of Cuboids for this process including ghosts.
 */
mgcl::MPIData::MPIData(MPI_Comm _comm, int mgh, int ngh, int ogh)
    : comm(_comm)
{
    _sbufxy = std::make_unique<Cuboid>(1, mgh, ngh);
    _sbufxz = std::make_unique<Cuboid>(1, mgh, ogh);
    _sbufyz = std::make_unique<Cuboid>(1, ngh, ogh);
    _rbufxy = std::make_unique<Cuboid>(1, mgh, ngh);
    _rbufxz = std::make_unique<Cuboid>(1, mgh, ogh);
    _rbufyz = std::make_unique<Cuboid>(1, ngh, ogh);
}
