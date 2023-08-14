#include <catch2/catch_session.hpp>

#include "mpi.h"

// Initializes MPI, runs Catch2 tests and finalizes MPI.
int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int result = Catch::Session().run(argc, argv);

    MPI_Finalize();

    return result;
}
