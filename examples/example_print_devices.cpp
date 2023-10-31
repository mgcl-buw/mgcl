
#include <iostream>
#include <memory>

#include "mpi.h"

#include "../src/mgcl/cuboid.hpp"
#include "../src/mgcl/problem.hpp"

/*
 * This program creates a problem using OpenCL, which then prints the used devices.
 * This is mainly useful for checking which process uses which device in a multi-device setup.
 */
int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int mpi_size;
    int mpi_rank;

    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    int N = 16;

    auto f = std::make_shared<mgcl::Cuboid>(N, N, N);
    auto v = std::make_shared<mgcl::Cuboid>(N, N, N);
    mgcl::Problem p(N, N, N, f, v);
    p.setUseOpencl(true);

    for (int i = 0; i < mpi_size; i++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if (i == mpi_rank)
        {
            std::cout << "on rank " << i << ": ";
            p.getOpenCLHelper().init();
            std::cout << std::flush;
        }
    }

    MPI_Finalize();

    return 0;
}
