#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include "../../src/cuboid.hpp"
#include "../../src/multigrid_engine.hpp"
#include "../../src/opencl_helper.hpp"
#include "../../src/problem.hpp"
#include "../ocl_wrapper.hpp"
#include "../test_utility.hpp"

// There was a bug with this benchmark, so this is a regression test. Results are not checked directly. If there is no
// runtime error, like a segfault or mpi error, this test will pass.
// Benchmarks different threshold levels for the vcycle using MPI seq vs opencl.
// Only rank 0 will print the timings.
// Run with e.g.: mpiexec -n 4 tests_mpi benchmark_vcycle_MPI_OCL_galerkin_thresholds
TEST_CASE("benchmark_vcycle_MPI_OCL_galerkin_thresholds")
{
    using std::min;

    // Check if mpi is initialized
    int isInitialized = 0;
    MPI_Initialized(&isInitialized);
    REQUIRE(isInitialized);

    MPI_Comm mpi_comm = MPI_COMM_WORLD;
    MPI_Comm_set_errhandler(mpi_comm, MPI_ERRORS_RETURN);

    // Check number of processes
    int mpi_size = -1;
    MPI_Comm_size(mpi_comm, &mpi_size);
    // REQUIRE(mpi_size == 1);

    int periodic = 1;

    /* MPI variables */
    int mpi_rank;
    int mpi_dims[3] = {0, 0, 0};
    int mpi_periods[3] = {periodic, periodic, periodic};
    int mpi_coords[3];

    /* Initialize cartesian process grid */
    MPI_Comm_size(mpi_comm, &mpi_size);
    MPI_Dims_create(mpi_size, 3, mpi_dims);
    MPI_Cart_create(mpi_comm, 3, mpi_dims, mpi_periods, 1, &mpi_comm);
    MPI_Comm_rank(mpi_comm, &mpi_rank);
    MPI_Cart_coords(mpi_comm, mpi_rank, 3, mpi_coords);

    double omega = 0.8;
    int nu1 = 3;
    int nu2 = 3;
    int vCycleIterations = 10;
    mgcl::MGCL_RESIDUAL_NORM resnorm = mgcl::MGCL_L2;
    mgcl::MGCL_STENCIL stencilType = mgcl::MGCL_VARYING;

    if (mpi_rank == 0)
        std::cout << "Problem parameters:" << std::endl
                  << "  Omega: " << omega << std::endl
                  << "  Nu1: " << nu1 << std::endl
                  << "  Nu2: " << nu2 << std::endl
                  << "  VCycle iterations: " << vCycleIterations << std::endl;

    std::vector<int> minGridPoints{2, 4, 8, 16};
    std::vector<int> grids = {4};
    for (auto N : grids)
    {
        int mglob = N * mpi_dims[0];
        int nglob = N * mpi_dims[1];
        int oglob = N * mpi_dims[2];

        for (auto mgp : minGridPoints)
        {
            if (mgp > N)
                continue;

            CAPTURE(N, mgp);

            int ghin = 0;
            auto v = std::make_shared<mgcl::Cuboid>(N, N, N, ghin, ghin, ghin);
            auto f = std::make_shared<mgcl::Cuboid>(N, N, N, ghin, ghin, ghin);
            v->fillRandom();
            f->fillRandom();

            mgcl::Problem pocl(N, N, N, v, f, mglob, nglob, oglob);
            // pocl.setSilent(true);
            pocl.setMpiMinGridPoints(mgp);
            pocl.setOmega(omega);
            pocl.setNu1(nu1);
            pocl.setNu2(nu2);
            pocl.setMaxiterVcycles(vCycleIterations);
            pocl.setGhostsIn(ghin);
            pocl.setStencilType(stencilType);
            pocl.setResidualNorm(resnorm);
            pocl.setMpiComm(mpi_comm);
            pocl.setUseOpencl(true);
            pocl.setDeviceType(CL_DEVICE_TYPE_GPU);
            pocl.setReadResults(true);

            auto& sv = pocl.getStencilValues();
            sv->fill1dIndex(true);

            pocl.solve();
            // tu.finish(); //
            MPI_Barrier(mpi_comm);
        }
    }
}
