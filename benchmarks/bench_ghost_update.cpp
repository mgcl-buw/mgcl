#include "nanobench.h"

#include "catch2/catch_test_macros.hpp"

#include <chrono>
#include <iostream>
#include <vector>
using namespace std::chrono_literals;

#include "../src/mgcl/cuboid.hpp"
#include "../src/mgcl/multigrid_engine.hpp"
#include "../src/mgcl/opencl_helper.hpp"
#include "../src/mgcl/problem.hpp"
#include "bench_util.hpp"
#include "cli_args.hpp"

// Runs ghost update using MPI and OCL, as it happens between Jacobi iterations. I.e. when using MPI,
// the gpu buffer is first read, then ghosts are updated sequentially, then the data is copied back to the gpu.
// Timings will be collected per node and printed by rank at the end.
// Run with e.g.: mpiexec -n 4 benchmarks bench_ghost_update_mpi_ocl
TEST_CASE("bench_ghost_update_mpi_ocl")
{
    using std::min;

    if (CLI_ARGS::grids.size() == 0 && (CLI_ARGS::gridsMin.size() == 0 || CLI_ARGS::gridsMax.size() == 0))
        throw "Need to specify at least one local grid size, e.g. using --grids 4,8,16 or --gridsMin 4,4,4 AND --gridsMax 32,32,32";

    // build grids to be tested from CLI args
    std::vector<std::vector<int>> gridsTBT;
    for (auto N : CLI_ARGS::grids)
        gridsTBT.push_back({N, N, N});
    if (CLI_ARGS::gridsMin.size() > 0 && CLI_ARGS::gridsMax.size() > 0)
        for (int m = CLI_ARGS::gridsMin[0]; m <= CLI_ARGS::gridsMax[0]; m *= 2)
            for (int n = CLI_ARGS::gridsMin[1]; n <= CLI_ARGS::gridsMax[1]; n *= 2)
                for (int o = CLI_ARGS::gridsMin[2]; o <= CLI_ARGS::gridsMax[2]; o *= 2)
                    gridsTBT.push_back({m, n, o});

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

    if (mpi_rank == 0)
    {
        std::cout << "Testing the following grid sizes" << std::endl;
        for (auto gr : gridsTBT)
        {
            int m = gr[0];
            int n = gr[1];
            int o = gr[2];
            std::cout << "  local size: " << m << "," << n << "," << o << ", global size: "
                      << m * mpi_dims[0] << "," << n * mpi_dims[1] << "," << o * mpi_dims[2] << std::endl;
        }
    }
    MPI_Barrier(mpi_comm);

    int maxGhosts = 3;
    std::vector<bench_util::ResultGhostUpdateMpi> results;

    bool printedGpu = false;
    for (auto gr : gridsTBT)
    {
        int m = gr[0];
        int n = gr[1];
        int o = gr[2];
        int mglob = m * mpi_dims[0];
        int nglob = n * mpi_dims[1];
        int oglob = o * mpi_dims[2];
        double hm = 1.0 / (double)mglob;
        double hn = 1.0 / (double)nglob;
        double ho = 1.0 / (double)oglob;

        mgcl::MGCL_RESIDUAL_NORM resnorm = mgcl::MGCL_L2;
        mgcl::MGCL_STENCIL stencilType = mgcl::MGCL_LAPLACE_7POINT;

        ankerl::nanobench::Bench bench;
        bench.timeUnit(1ms, "ms")
            .epochs(CLI_ARGS::bench_epochs)
            .epochIterations(CLI_ARGS::bench_iterations)
            // .minEpochTime(100ms)
            .relative(CLI_ARGS::jacobiIters.size() > 1);

        for (int ghosts = 1; ghosts <= maxGhosts; ghosts++)
        {
            auto v = std::make_shared<mgcl::Cuboid>(m, n, o, ghosts, ghosts, ghosts);
            auto r = std::make_shared<mgcl::Cuboid>(m, n, o, ghosts, ghosts, ghosts);
            auto f = std::make_shared<mgcl::Cuboid>(m, n, o, ghosts, ghosts, ghosts);
            v->fillRandom();
            f->fillRandom();

            auto p = std::make_shared<mgcl::Problem>(m, n, o, f, v, mglob, nglob, oglob);
            p->setMpiComm(mpi_comm);
            p->setGhosts(ghosts);
            p->setGhostsIn(ghosts);
            p->setUseOpencl(true);
            p->setSilent(true);
            if (p->getStencilType() == mgcl::MGCL_VARYING)
                p->getStencilValues()->fillRandom();
            p->init();

            auto& buf = p->getLevelAt(0).getDVIn();
            auto mpiLevelData = p->getLevelAt(0).getMpiDataPtr();

            if (!printedGpu)
            {
                for (int i = 0; i < mpi_size; i++)
                {
                    MPI_Barrier(mpi_comm);
                    if (i == mpi_rank)
                    {
                        std::cout << "on rank " << mpi_rank << ", GPU info: ";
                        p->getOpenCLHelper().outputDeviceInfo(p->getOpenCLHelper().getDeviceId());
                    }
                }
                printedGpu = true;
            }

            std::string name = std::string("ocl_mpi_N")
                                   .append(std::to_string(m))
                                   .append("_")
                                   .append(std::to_string(n))
                                   .append("_")
                                   .append(std::to_string(o))
                                   .append("_gh")
                                   .append(std::to_string(ghosts));
            bench.run(std::string(name).c_str(), [&] { //
                MPI_Barrier(mpi_comm);
                mgcl::MultigridEngine::updateGhosts(*p, buf, mpiLevelData, false);
                p->getOpenCLHelper().finish();
                MPI_Barrier(mpi_comm);
            });

            // std::cout << "rank " << mpi_rank << " done" << std::endl;
            MPI_Barrier(mpi_comm);

            bench_util::ResultGhostUpdateMpi res;
            res.name = name;
            res.minTime = bench_util::getMinTime(bench, name);
            res.medianTime = bench_util::getMedianTime(bench, name);
            res.avgTime = bench_util::getAvgTime(bench, name);
            res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
            res.mloc = m;
            res.nloc = n;
            res.oloc = o;
            res.mglob = mglob;
            res.nglob = nglob;
            res.oglob = oglob;
            res.ghosts = ghosts;
            res.gpus = mpi_size;
            results.push_back(res);
        }
    }

    bench_util::printCsvFormat(results, mpi_comm, mpi_rank);

    MPI_Barrier(mpi_comm);
}

// Runs ghost update using MPI, as it happens between Jacobi iterations. I.e. when using MPI,
// the gpu buffer is first read, then ghosts are updated sequentially, then the data is copied back to the gpu.
// Timings will be collected per node and printed by rank at the end.
// Run with e.g.: mpiexec -n 4 benchmarks bench_ghost_update_mpi_ocl
TEST_CASE("bench_ghost_update_mpi_seq")
{
    using std::min;

    if (CLI_ARGS::grids.size() == 0 && (CLI_ARGS::gridsMin.size() == 0 || CLI_ARGS::gridsMax.size() == 0))
        throw "Need to specify at least one local grid size, e.g. using --grids 4,8,16 or --gridsMin 4,4,4 AND --gridsMax 32,32,32";

    // build grids to be tested from CLI args
    std::vector<std::vector<int>> gridsTBT;
    for (auto N : CLI_ARGS::grids)
        gridsTBT.push_back({N, N, N});
    if (CLI_ARGS::gridsMin.size() > 0 && CLI_ARGS::gridsMax.size() > 0)
        for (int m = CLI_ARGS::gridsMin[0]; m <= CLI_ARGS::gridsMax[0]; m *= 2)
            for (int n = CLI_ARGS::gridsMin[1]; n <= CLI_ARGS::gridsMax[1]; n *= 2)
                for (int o = CLI_ARGS::gridsMin[2]; o <= CLI_ARGS::gridsMax[2]; o *= 2)
                    gridsTBT.push_back({m, n, o});

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

    if (mpi_rank == 0)
    {
        std::cout << "Testing the following grid sizes" << std::endl;
        for (auto gr : gridsTBT)
        {
            int m = gr[0];
            int n = gr[1];
            int o = gr[2];
            std::cout << "  local size: " << m << "," << n << "," << o << ", global size: "
                      << m * mpi_dims[0] << "," << n * mpi_dims[1] << "," << o * mpi_dims[2] << std::endl;
        }
    }
    MPI_Barrier(mpi_comm);

    int maxGhosts = 3;
    std::vector<bench_util::ResultGhostUpdateMpi> results;

    for (auto gr : gridsTBT)
    {
        int m = gr[0];
        int n = gr[1];
        int o = gr[2];
        int mglob = m * mpi_dims[0];
        int nglob = n * mpi_dims[1];
        int oglob = o * mpi_dims[2];
        double hm = 1.0 / (double)mglob;
        double hn = 1.0 / (double)nglob;
        double ho = 1.0 / (double)oglob;

        mgcl::MGCL_RESIDUAL_NORM resnorm = mgcl::MGCL_L2;
        mgcl::MGCL_STENCIL stencilType = mgcl::MGCL_LAPLACE_7POINT;

        ankerl::nanobench::Bench bench;
        bench.timeUnit(1ms, "ms")
            .epochs(CLI_ARGS::bench_epochs)
            .epochIterations(CLI_ARGS::bench_iterations)
            // .minEpochTime(100ms)
            .relative(CLI_ARGS::jacobiIters.size() > 1);

        for (int ghosts = 1; ghosts <= maxGhosts; ghosts++)
        {
            auto v = std::make_shared<mgcl::Cuboid>(m, n, o, ghosts, ghosts, ghosts);
            auto r = std::make_shared<mgcl::Cuboid>(m, n, o, ghosts, ghosts, ghosts);
            auto f = std::make_shared<mgcl::Cuboid>(m, n, o, ghosts, ghosts, ghosts);
            v->fillRandom();
            f->fillRandom();

            auto p = std::make_shared<mgcl::Problem>(m, n, o, f, v, mglob, nglob, oglob);
            p->setMpiComm(mpi_comm);
            p->setGhosts(ghosts);
            p->setGhostsIn(ghosts);
            p->setUseOpencl(false);
            p->setSilent(true);
            if (p->getStencilType() == mgcl::MGCL_VARYING)
                p->getStencilValues()->fillRandom();
            p->init();

            auto& buf = p->getLevelAt(0).getV();
            auto mpiLevelData = p->getLevelAt(0).getMpiDataPtr();

            std::string name = std::string("seq_mpi_N")
                                   .append(std::to_string(m))
                                   .append("_")
                                   .append(std::to_string(n))
                                   .append("_")
                                   .append(std::to_string(o))
                                   .append("_gh")
                                   .append(std::to_string(ghosts));
            bench.run(std::string(name).c_str(), [&] { //
                MPI_Barrier(mpi_comm);
                mgcl::MultigridEngine::updateGhostsSeq(buf, mpiLevelData, true, false);
                MPI_Barrier(mpi_comm);
            });

            // std::cout << "rank " << mpi_rank << " done" << std::endl;
            MPI_Barrier(mpi_comm);

            bench_util::ResultGhostUpdateMpi res;
            res.name = name;
            res.minTime = bench_util::getMinTime(bench, name);
            res.medianTime = bench_util::getMedianTime(bench, name);
            res.avgTime = bench_util::getAvgTime(bench, name);
            res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
            res.mloc = m;
            res.nloc = n;
            res.oloc = o;
            res.mglob = mglob;
            res.nglob = nglob;
            res.oglob = oglob;
            res.ghosts = ghosts;
            res.gpus = mpi_size;
            results.push_back(res);
        }
    }

    bench_util::printCsvFormat(results, mpi_comm, mpi_rank);

    MPI_Barrier(mpi_comm);
}

// Tests the method mgcl::Cuboid::slice for all 3 directions, which is needed when updating ghosts.
// This test can be run without MPI.
// The argument --grids ... is required.
TEST_CASE("bench_cuboid_slice")
{
    using std::min;

    if (CLI_ARGS::grids.size() == 0 && (CLI_ARGS::gridsMin.size() == 0 || CLI_ARGS::gridsMax.size() == 0))
        throw "Need to specify at least one local grid size, e.g. using --grids 4,8,16 or --gridsMin 4,4,4 AND --gridsMax 32,32,32";

    // build grids to be tested from CLI args
    std::vector<std::vector<int>> gridsTBT;
    for (auto N : CLI_ARGS::grids)
        gridsTBT.push_back({N, N, N});
    if (CLI_ARGS::gridsMin.size() > 0 && CLI_ARGS::gridsMax.size() > 0)
        for (int m = CLI_ARGS::gridsMin[0]; m <= CLI_ARGS::gridsMax[0]; m *= 2)
            for (int n = CLI_ARGS::gridsMin[1]; n <= CLI_ARGS::gridsMax[1]; n *= 2)
                for (int o = CLI_ARGS::gridsMin[2]; o <= CLI_ARGS::gridsMax[2]; o *= 2)
                    gridsTBT.push_back({m, n, o});

    // Check if mpi is initialized
    int isInitialized = 0;
    MPI_Initialized(&isInitialized);
    REQUIRE(isInitialized);

    /* MPI variables */
    int mpi_rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    if (mpi_rank == 0)
    {
        std::cout << "Testing the following grid sizes" << std::endl;
        for (auto gr : gridsTBT)
        {
            int m = gr[0];
            int n = gr[1];
            int o = gr[2];
            std::cout << "  " << m << "," << n << "," << o << std::endl;
        }
    }
    else
    {
        return;
    }

    std::vector<bench_util::Result> results;

    for (auto gr : gridsTBT)
    {
        int m = gr[0];
        int n = gr[1];
        int o = gr[2];

        mgcl::Cuboid c(m, n, o);
        int gh = 1;

        ankerl::nanobench::Bench bench;
        bench.timeUnit(1ms, "ms")
            .epochs(CLI_ARGS::bench_epochs)
            .epochIterations(CLI_ARGS::bench_iterations);

        {
            std::string name = std::string("slice_xdir_")
                                   .append(std::to_string(m))
                                   .append("_")
                                   .append(std::to_string(n))
                                   .append("_")
                                   .append(std::to_string(o));
            bench.run(std::string(name).c_str(), [&] { //
                ankerl::nanobench::doNotOptimizeAway(c.slice(gh, 2 * gh - 1, 0, n - 1, 0, o - 1));
            });

            bench_util::Result res;
            res.name = name;
            res.m = m;
            res.n = n;
            res.o = o;
            res.minTime = bench_util::getMinTime(bench, name);
            res.medianTime = bench_util::getMedianTime(bench, name);
            res.avgTime = bench_util::getAvgTime(bench, name);
            res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
            results.push_back(res);
        }

        {
            std::string name = std::string("slice_ydir_")
                                   .append(std::to_string(m))
                                   .append("_")
                                   .append(std::to_string(n))
                                   .append("_")
                                   .append(std::to_string(o));
            bench.run(std::string(name).c_str(), [&] { //
                ankerl::nanobench::doNotOptimizeAway(c.slice(0, m - 1, gh, 2 * gh - 1, 0, o - 1));
            });

            bench_util::Result res;
            res.name = name;
            res.m = m;
            res.n = n;
            res.o = o;
            res.minTime = bench_util::getMinTime(bench, name);
            res.medianTime = bench_util::getMedianTime(bench, name);
            res.avgTime = bench_util::getAvgTime(bench, name);
            res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
            results.push_back(res);
        }

        {
            std::string name = std::string("slice_zdir_")
                                   .append(std::to_string(m))
                                   .append("_")
                                   .append(std::to_string(n))
                                   .append("_")
                                   .append(std::to_string(o));
            bench.run(std::string(name).c_str(), [&] { //
                ankerl::nanobench::doNotOptimizeAway(c.slice(0, m - 1, 0, n - 1, gh, 2 * gh - 1));
            });

            bench_util::Result res;
            res.name = name;
            res.m = m;
            res.n = n;
            res.o = o;
            res.minTime = bench_util::getMinTime(bench, name);
            res.medianTime = bench_util::getMedianTime(bench, name);
            res.avgTime = bench_util::getAvgTime(bench, name);
            res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
            results.push_back(res);
        }
    }

    bench_util::printCsvFormat(results);
}