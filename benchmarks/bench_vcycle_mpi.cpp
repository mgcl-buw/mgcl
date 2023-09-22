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

#include "../src/cuboid.hpp"
#include "../src/multigrid_engine.hpp"
#include "../src/opencl_helper.hpp"
#include "../src/problem.hpp"
#include "../test/ocl_wrapper.hpp"
#include "../test/test_utility.hpp"
#include "bench_render_templates.hpp"
#include "cli_args.hpp"

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
                  << "  VCycle iterations: " << CLI_ARGS::vCycleIterations << std::endl;

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

        // /* Initialize start and end for local grid */
        // int m_start = (mglob / mpi_dims[0]) * mpi_coords[0] + min(mpi_coords[0], (mglob % mpi_dims[0]));
        // int m_end = (mglob / mpi_dims[0]) * (mpi_coords[0] + 1) + min(mpi_coords[0] + 1, (mglob % mpi_dims[0])) - 1;
        // int n_start = (nglob / mpi_dims[1]) * mpi_coords[1] + min(mpi_coords[1], (nglob % mpi_dims[1]));
        // int n_end = (nglob / mpi_dims[1]) * (mpi_coords[1] + 1) + min(mpi_coords[1] + 1, (nglob % mpi_dims[1])) - 1;
        // int o_start = (oglob / mpi_dims[2]) * mpi_coords[2] + min(mpi_coords[2], (oglob % mpi_dims[2]));
        // int o_end = (oglob / mpi_dims[2]) * (mpi_coords[2] + 1) + min(mpi_coords[2] + 1, (oglob % mpi_dims[2])) - 1;

        // int m = (m_end - m_start) + 1;
        // int n = (n_end - n_start) + 1;
        // int o = (o_end - o_start) + 1;

        // for (auto iters : itersAll)
        // {

        //     for (int spi = 1; spi <= maxStepsPerIter; spi++)
        //     {
        //         auto v = std::make_shared<mgcl::Cuboid>(m, n, o, spi, spi, spi);
        //         auto r = std::make_shared<mgcl::Cuboid>(m, n, o, spi, spi, spi);
        //         auto f = std::make_shared<mgcl::Cuboid>(m, n, o, spi, spi, spi);
        //         v->fillRandom();
        //         f->fillRandom();

        //         auto p = std::make_shared<mgcl::Problem>(m, n, o, v, f, mglob, nglob, oglob);
        //         p->setMpiComm(mpi_comm);
        //         p->setGhosts(spi);
        //         p->setJacobiIterationsPerKernel(spi);
        //         p->setUseOpencl(true);
        //         p->setSilent(true);
        //         p->init();
        //         auto& level = p->getLevelAt(0);
        //         mgcl_test::TestUtility tu(p);

        //         std::string name = std::string("ocl, N = ")
        //                                .append(std::to_string(N))
        //                                .append(", spi = ")
        //                                .append(std::to_string(spi))
        //                                .append(", iters = ")
        //                                .append(std::to_string(iters));
        //         bench.run(std::string(name).c_str(), [&] { //
        //             mgcl::MultigridEngine::jacobi(*p, level, iters, false, spi);
        //             tu.finish(); //
        //             MPI_Barrier(mpi_comm);
        //         });

        //         // std::cout << "rank " << mpi_rank << " done" << std::endl;
        //         MPI_Barrier(mpi_comm);

        //         // Get minimum of all epochs in ns
        //         double minTime = 1000000;
        //         for (auto r : bench.results())
        //             if (r.minimum(ankerl::nanobench::Result::Measure::elapsed) < minTime)
        //                 minTime = r.minimum(ankerl::nanobench::Result::Measure::elapsed) * 1000.0 * 1000.0 * 1000.0;

        //         ss << N << ";" << iters << ";" << spi << ";" << minTime << std::endl;
        //         mintimesPerSpi[spi - 1].push_back(minTime);

        //         // std::cout << "bench.results()[0].size(): " << bench.results()[0].size() << std::endl;
        //     }

        //     // std::ofstream renderOutCsv(std::string("jacobiMultiIterOcl_")
        //     //                                .append(std::to_string(N))
        //     //                                .append("_")
        //     //                                .append(".csv"));

        //     // std::streambuf *old = std::cout.rdbuf(ss.rdbuf());
        //     // bench.render(ankerl::nanobench::templates::csv(), std::cout);
        //     // std::cout.rdbuf(old);
    }

    // std::cout << ss.str() << std::endl;

    // std::vector<double> avgs = {0, 0, 0};
    // for (int spi = 0; spi < maxStepsPerIter; spi++)
    // {
    //     for (int val : mintimesPerSpi[spi])
    //         avgs[spi] += val;

    //     avgs[spi] /= (double)mintimesPerSpi[spi].size();
    // }

    // for (int i = 0; i < mpi_size; i++)
    // {
    //     MPI_Barrier(mpi_comm);
    //     if (i == mpi_rank)
    //         std::cout << "rank " << i << " avgs:" << std::endl
    //                   << "  spi: 1: " << avgs[0] << " ns" << std::endl
    //                   << "  spi: 2: " << avgs[1] << " ns" << std::endl
    //                   << "  spi: 3: " << avgs[2] << " ns" << std::endl;
    // }
    // MPI_Barrier(mpi_comm);
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
