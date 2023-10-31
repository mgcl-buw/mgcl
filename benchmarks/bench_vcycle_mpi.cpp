#include "nanobench.h"

#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>
using namespace std::chrono_literals;

#include "../src/mgcl/cuboid.hpp"
#include "../src/mgcl/multigrid_engine.hpp"
#include "../src/mgcl/opencl_helper.hpp"
#include "../src/mgcl/problem.hpp"
#include "../test/ocl_wrapper.hpp"
#include "../test/test_utility.hpp"
#include "bench_render_templates.hpp"
#include "cli_args.hpp"

// Benchmarks the vcycle using MPI seq.
// Only rank 0 will print the timings.
// Run with e.g.: mpiexec -n 64 benchmarks benchmark_vcycle_MPI_Seq_only_galerkin
TEST_CASE("benchmark_vcycle_MPI_Seq_only_galerkin")
{
    using std::min;

    if (CLI_ARGS::grids.size() == 0)
        throw "Need to specify at least one local grid size, e.g. using --grids 4,8,16";

    // Check if mpi is initialized
    int isInitialized = 0;
    MPI_Initialized(&isInitialized);
    REQUIRE(isInitialized);

    MPI_Comm mpi_comm = MPI_COMM_WORLD;

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

    if (mpi_rank == 0)
        std::cout << "Problem parameters:" << std::endl
                  << "  Omega: " << omega << std::endl
                  << "  Nu1: " << nu1 << std::endl
                  << "  Nu2: " << nu2 << std::endl
                  << "  VCycle iterations: " << CLI_ARGS::vCycleIterations << std::endl
                  << "  #procs: " << mpi_size << std::endl;

    if (mpi_rank == 0)
    {
        std::cout << "Testing the following grid sizes (N local; m,n,o global)" << std::endl;
        for (auto N : CLI_ARGS::grids)
        {
            std::cout << "  local size: " << N << ", global sizes: "
                      << N * mpi_dims[0] << "," << N * mpi_dims[1] << "," << N * mpi_dims[2] << std::endl;
        }
    }
    MPI_Barrier(mpi_comm);

    int maxStepsPerIter = 3;
    std::stringstream ss;
    ss << "N;iters;spi;ns" << std::endl;

    // Vector to collect all minimum times per spi, in order to get avg results later.
    std::vector<std::vector<int>> mintimesPerSpi(maxStepsPerIter);

    for (auto N : CLI_ARGS::grids)
    {
        int mglob = N * mpi_dims[0];
        int nglob = N * mpi_dims[1];
        int oglob = N * mpi_dims[2];
        double h = 1.0 / (double)mglob;

        mgcl::MGCL_RESIDUAL_NORM resnorm = mgcl::MGCL_L2;
        mgcl::MGCL_STENCIL stencilType = mgcl::MGCL_VARYING;

        int ghin = 0;
        auto v = std::make_shared<mgcl::Cuboid>(N, N, N, ghin, ghin, ghin);
        auto f = std::make_shared<mgcl::Cuboid>(N, N, N, ghin, ghin, ghin);
        v->fillRandom();
        f->fillRandom();

        mgcl::Problem pseq(N, N, N, v, f, mglob, nglob, oglob);
        pseq.setSilent(true);
        pseq.setOmega(omega);
        pseq.setNu1(nu1);
        pseq.setNu2(nu2);
        pseq.setMaxiterVcycles(CLI_ARGS::vCycleIterations);
        pseq.setGhostsIn(ghin);
        pseq.setStencilType(stencilType);
        pseq.setResidualNorm(resnorm);
        pseq.setMpiComm(mpi_comm);

        auto& sv = pseq.getStencilValues();
        sv->fill1dIndex(true);

        ankerl::nanobench::Bench bench;
        bench.timeUnit(1ns, "ns")
            .epochs(11)
            .epochIterations(1)
            // .minEpochTime(100ms)
            .relative(true);

        // disable output for non-root processes
        if (mpi_rank > 0)
            bench.output(nullptr);

        std::string name = std::string("seqN").append(std::to_string(N));

        bench.run(std::string(name).c_str(), [&] { //
            pseq.solveSeq();
            // tu.finish(); //
            MPI_Barrier(mpi_comm);
        });
    }
}

// Benchmarks the vcycle using MPI seq vs opencl.
// Only rank 0 will print the timings.
// Run with e.g.: mpiexec -n 4 benchmarks benchmark_vcycle_MPI_Seq_vs_OCL_galerkin
TEST_CASE("benchmark_vcycle_MPI_Seq_vs_OCL_galerkin")
{
    using std::min;

    if (CLI_ARGS::grids.size() == 0)
        throw "Need to specify at least one local grid size, e.g. using --grids 4,8,16";

    // Check if mpi is initialized
    int isInitialized = 0;
    MPI_Initialized(&isInitialized);
    REQUIRE(isInitialized);

    MPI_Comm mpi_comm = MPI_COMM_WORLD;

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

    if (mpi_rank == 0)
        std::cout << "Problem parameters:" << std::endl
                  << "  Omega: " << omega << std::endl
                  << "  Nu1: " << nu1 << std::endl
                  << "  Nu2: " << nu2 << std::endl
                  << "  VCycle iterations: " << CLI_ARGS::vCycleIterations << std::endl
                  << "  #procs: " << mpi_size << std::endl;

    if (mpi_rank == 0)
    {
        std::cout << "Testing the following grid sizes (N local; m,n,o global)" << std::endl;
        for (auto N : CLI_ARGS::grids)
        {
            std::cout << "  local size: " << N << ", global sizes: "
                      << N * mpi_dims[0] << "," << N * mpi_dims[1] << "," << N * mpi_dims[2] << std::endl;
        }
    }
    MPI_Barrier(mpi_comm);

    int maxStepsPerIter = 3;
    std::stringstream ss;
    ss << "N;iters;spi;ns" << std::endl;

    // Vector to collect all minimum times per spi, in order to get avg results later.
    std::vector<std::vector<int>> mintimesPerSpi(maxStepsPerIter);

    for (auto N : CLI_ARGS::grids)
    {
        int mglob = N * mpi_dims[0];
        int nglob = N * mpi_dims[1];
        int oglob = N * mpi_dims[2];
        double h = 1.0 / (double)mglob;

        mgcl::MGCL_RESIDUAL_NORM resnorm = mgcl::MGCL_L2;
        mgcl::MGCL_STENCIL stencilType = mgcl::MGCL_VARYING;

        int ghin = 0;
        auto v = std::make_shared<mgcl::Cuboid>(N, N, N, ghin, ghin, ghin);
        auto f = std::make_shared<mgcl::Cuboid>(N, N, N, ghin, ghin, ghin);
        v->fillRandom();
        f->fillRandom();

        mgcl::Problem pseq(N, N, N, v, f, mglob, nglob, oglob);
        pseq.setSilent(true);
        pseq.setOmega(omega);
        pseq.setNu1(nu1);
        pseq.setNu2(nu2);
        pseq.setMaxiterVcycles(CLI_ARGS::vCycleIterations);
        pseq.setGhostsIn(ghin);
        pseq.setStencilType(stencilType);
        pseq.setResidualNorm(resnorm);
        pseq.setMpiComm(mpi_comm);

        auto& sv = pseq.getStencilValues();
        sv->fill1dIndex(true);

        ankerl::nanobench::Bench bench;
        bench.timeUnit(1ns, "ns")
            .epochs(11)
            .epochIterations(1)
            // .minEpochTime(100ms)
            .relative(true);

        // disable output for non-root processes
        if (mpi_rank > 0)
            bench.output(nullptr);

        std::string name = std::string("seqN").append(std::to_string(N));

        bench.run(std::string(name).c_str(), [&] { //
            pseq.solveSeq();
            // tu.finish(); //
            MPI_Barrier(mpi_comm);
        });

        v->fillRandom();
        f->fillRandom();
        mgcl::Problem pocl(N, N, N, v, f, mglob, nglob, oglob);
        pocl.setSilent(true);
        pocl.setOmega(omega);
        pocl.setNu1(nu1);
        pocl.setNu2(nu2);
        pocl.setGhostsIn(ghin);
        pocl.setStencilType(stencilType);
        pocl.setResidualNorm(resnorm);
        pocl.setMpiComm(mpi_comm);
        pocl.setUseOpencl(true);
        pocl.setDeviceType(CL_DEVICE_TYPE_GPU);

        sv = pocl.getStencilValues();
        sv->fill1dIndex(true);

        name = std::string("oclN").append(std::to_string(N));

        bench.run(std::string(name).c_str(), [&] { //
            pocl.solve();
            // tu.finish(); //
            MPI_Barrier(mpi_comm);
        });
    }
}

// Benchmarks different threshold levels for the vcycle using MPI seq vs opencl.
// Only rank 0 will print the timings.
// Run with e.g.: mpiexec -n 4 benchmarks benchmark_vcycle_MPI_OCL_galerkin_thresholds
TEST_CASE("benchmark_vcycle_MPI_OCL_galerkin_thresholds")
{
    using std::min;

    if (CLI_ARGS::grids.size() == 0)
        throw "Need to specify at least one local grid size, e.g. using --grids 4,8,16";

    // Check if mpi is initialized
    int isInitialized = 0;
    MPI_Initialized(&isInitialized);
    REQUIRE(isInitialized);

    MPI_Comm mpi_comm = MPI_COMM_WORLD;

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
    mgcl::MGCL_RESIDUAL_NORM resnorm = mgcl::MGCL_L2;
    mgcl::MGCL_STENCIL stencilType = mgcl::MGCL_VARYING;

    if (mpi_rank == 0)
        std::cout << "Problem parameters:" << std::endl
                  << "  Omega: " << omega << std::endl
                  << "  Nu1: " << nu1 << std::endl
                  << "  Nu2: " << nu2 << std::endl
                  << "  VCycle iterations: " << CLI_ARGS::vCycleIterations << std::endl;

    std::vector<int> minGridPoints{2, 4, 8, 16};
    for (auto N : CLI_ARGS::grids)
    {
        int mglob = N * mpi_dims[0];
        int nglob = N * mpi_dims[1];
        int oglob = N * mpi_dims[2];
        double h = 1.0 / (double)mglob;

        ankerl::nanobench::Bench bench;
        bench.timeUnit(1ns, "ns")
            .epochs(11)
            .epochIterations(1)
            // .minEpochTime(100ms)
            .relative(true);

        // disable output for non-root processes
        if (mpi_rank > 0)
            bench.output(nullptr);

        for (auto mgp : minGridPoints)
        {
            if (mgp > N)
                continue;

            int ghin = 0;
            auto v = std::make_shared<mgcl::Cuboid>(N, N, N, ghin, ghin, ghin);
            auto f = std::make_shared<mgcl::Cuboid>(N, N, N, ghin, ghin, ghin);
            v->fillRandom();
            f->fillRandom();

            mgcl::Problem pocl(N, N, N, v, f, mglob, nglob, oglob);
            pocl.setSilent(true);
            pocl.setMpiMinGridPoints(mgp);
            pocl.setOmega(omega);
            pocl.setNu1(nu1);
            pocl.setNu2(nu2);
            pocl.setMaxiterVcycles(CLI_ARGS::vCycleIterations);
            pocl.setGhostsIn(ghin);
            pocl.setStencilType(stencilType);
            pocl.setResidualNorm(resnorm);
            pocl.setMpiComm(mpi_comm);
            pocl.setUseOpencl(true);
            pocl.setDeviceType(CL_DEVICE_TYPE_GPU);
            pocl.setReadResults(true);

            auto& sv = pocl.getStencilValues();
            sv->fill1dIndex(true);

            std::string name = std::string("oclN")
                                   .append(std::to_string(N))
                                   .append("mgp")
                                   .append(std::to_string(mgp));

            bench.run(std::string(name).c_str(), [&] { //
                pocl.solve();
                // tu.finish(); //
                MPI_Barrier(mpi_comm);
            });
        }
    }
}

// Benchmarks different jacobiItersPerKernel for the vcycle using MPI and OpenCL.
// Only rank 0 will print the timings.
// Run with e.g.: mpiexec -n 4 benchmarks benchmark_vcycle_MPI_OCL_galerkin_jacobi_iters
TEST_CASE("benchmark_vcycle_MPI_OCL_galerkin_jacobi_iters")
{
    using std::min;

    if (CLI_ARGS::grids.size() == 0)
        throw "Need to specify at least one local grid size, e.g. using --grids 4,8,16";

    // Check if mpi is initialized
    int isInitialized = 0;
    MPI_Initialized(&isInitialized);
    REQUIRE(isInitialized);

    MPI_Comm mpi_comm = MPI_COMM_WORLD;

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
    mgcl::MGCL_RESIDUAL_NORM resnorm = mgcl::MGCL_L2;
    mgcl::MGCL_STENCIL stencilType = mgcl::MGCL_VARYING;

    if (mpi_rank == 0)
        std::cout << "Problem parameters:" << std::endl
                  << "  Omega: " << omega << std::endl
                  << "  VCycle iterations: " << CLI_ARGS::vCycleIterations << std::endl
                  << "  #procs: " << mpi_size << std::endl;

    std::vector<int> itersPerKernel{1, 2, 3};
    std::vector<int> nus{2, 3, 5, 10}; // nu1 and nu2

    for (auto N : CLI_ARGS::grids)
    {
        int mglob = N * mpi_dims[0];
        int nglob = N * mpi_dims[1];
        int oglob = N * mpi_dims[2];
        double h = 1.0 / (double)mglob;

        for (auto nu12 : nus)
        {
            ankerl::nanobench::Bench bench;
            bench.timeUnit(1ns, "ns")
                .epochs(11)
                .epochIterations(1)
                // .minEpochTime(100ms)
                .relative(true);

            // disable output for non-root processes
            if (mpi_rank > 0)
                bench.output(nullptr);

            for (auto ipk : itersPerKernel)
            {
                if (ipk > nu12)
                    continue;

                int ghin = 0;
                auto v = std::make_shared<mgcl::Cuboid>(N, N, N, ghin, ghin, ghin);
                auto f = std::make_shared<mgcl::Cuboid>(N, N, N, ghin, ghin, ghin);
                v->fillRandom();
                f->fillRandom();

                mgcl::Problem pocl(N, N, N, v, f, mglob, nglob, oglob);
                pocl.setSilent(true);
                // pocl.setMpiMinGridPoints(mgp);
                pocl.setOmega(omega);
                pocl.setNu1(nu12);
                pocl.setNu2(nu12);
                pocl.setMaxiterVcycles(CLI_ARGS::vCycleIterations);
                pocl.setGhostsIn(ghin);
                pocl.setJacobiIterationsPerKernel(ipk);
                pocl.setGhosts(ipk);
                pocl.setStencilType(stencilType);
                pocl.setResidualNorm(resnorm);
                pocl.setMpiComm(mpi_comm);
                pocl.setUseOpencl(true);
                pocl.setDeviceType(CL_DEVICE_TYPE_GPU);
                pocl.setReadResults(true);

                auto& sv = pocl.getStencilValues();
                sv->fill1dIndex(true);

                std::string name = std::string("oclN")
                                       .append(std::to_string(N))
                                       .append("nu")
                                       .append(std::to_string(nu12))
                                       .append("ipk")
                                       .append(std::to_string(ipk));

                bench.run(std::string(name).c_str(), [&] { //
                    pocl.solve();
                    // tu.finish(); //
                    MPI_Barrier(mpi_comm);
                });
            }
        }
    }
}
