#include "mpi_level_data.hpp"

#include "mpi_util.hpp"
#include <cassert>
#include <iostream>
#include <iterator>
#include <mpi.h>
#include <ostream>
#include <sstream>

namespace mgcl
{
    /**
     * @brief Construct a new mgcl::MPILevelData object
     *
     * @param _comm MPI communicator
     */
    MPILevelData::MPILevelData(MPI_Comm _comm, int ghosts)
        : comm(_comm), ghosts(ghosts)
    {
        MPI_Comm_rank(_comm, &rank);

        left = std::vector<int>(ghosts, MPI_PROC_NULL);
        right = std::vector<int>(ghosts, MPI_PROC_NULL);
        up = std::vector<int>(ghosts, MPI_PROC_NULL);
        down = std::vector<int>(ghosts, MPI_PROC_NULL);
        front = std::vector<int>(ghosts, MPI_PROC_NULL);
        back = std::vector<int>(ghosts, MPI_PROC_NULL);
    }

    int MPILevelData::mpiSize()
    {
        int mpi_size;
        int err = MPI_Comm_size(comm, &mpi_size);
        mpi_util::mgclCheckMpiError(comm, err, "MPI_Comm_size");
        return mpi_size;
    }

    void MPILevelData::printNeighbours()
    {
        auto arjoin = [](const std::vector<int>& vec, std::string delim)
        {
            std::ostringstream oss;
            std::copy(vec.begin(), vec.end(), std::ostream_iterator<int>(oss, delim.c_str()));
            return oss.str();
        };

        std::cout << rank << ": " << std::endl;
        std::cout << "   left: " << arjoin(left, ",") << std::endl;
        std::cout << "  right: " << arjoin(right, ",") << std::endl;
        std::cout << "     up: " << arjoin(up, ",") << std::endl;
        std::cout << "   down: " << arjoin(down, ",") << std::endl;
        std::cout << "  front: " << arjoin(front, ",") << std::endl;
        std::cout << "   back: " << arjoin(back, ",") << std::endl;
    }
}