#include "nanobench.h"

#include "catch2/catch_test_macros.hpp"

#include <chrono>
#include <iostream>
#include <vector>
using namespace std::chrono_literals;

#include "../src/mgcl/cuboid.hpp"
#include "../src/mgcl/mpi_util.hpp"
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

        if (mpi_rank > 0)
            bench.output(nullptr);

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
            p->setDeviceType(CL_DEVICE_TYPE_GPU);
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
                        p->getOpenCLHelper().outputDeviceInfo();
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
// Timings will be collected per node and printed by rank at the end.
// Run with e.g.: mpiexec -n 4 benchmarks bench_ghost_update_mpi_seq
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
            p->setDeviceType(CL_DEVICE_TYPE_GPU);
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

// Benchmarks extract ghosts kernel plus data transfer between host and device without MPI.
// If called with multiple processes, only root process runs the benchmarks.
// These benchmarks seem to be very unstable.
// Run with e.g.: benchmarks bench_data_transfer_host_device
TEST_CASE("bench_ghostupdate_mpi_ocl_steps")
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
            std::cout << "  " << m << "," << n << "," << o << std::endl;
        }
    }

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
        int ghosts = 1;
        int mgh = m + 2 * ghosts;
        int ngh = n + 2 * ghosts;
        int ogh = o + 2 * ghosts;

        // Create a dummy problem
        auto v = std::make_shared<mgcl::Cuboid>(m, n, o);
        auto f = std::make_shared<mgcl::Cuboid>(m, n, o);
        mgcl::Problem p(m, n, o, f, v, mglob, nglob, oglob);
        p.setUseOpencl(true);
        p.setDeviceType(CL_DEVICE_TYPE_GPU);
        p.setGhosts(ghosts);
        p.setSilent(true);
        p.setMpiComm(mpi_comm);
        p.init();

        auto& mpiData = p.getLevelAt(0).getMpiData();

        if (!printedGpu)
        {
            for (int i = 0; i < mpi_size; i++)
            {
                MPI_Barrier(mpi_comm);
                if (i == mpi_rank)
                {
                    std::cout << "on rank " << mpi_rank << ", GPU info: ";
                    p.getOpenCLHelper().outputDeviceInfo();
                }
            }
            printedGpu = true;
        }

        int err;

        // actual test buffers
        mgcl::Cuboid& c_h = p.getLevelAt(0).getV();
        c_h.fillRandom();
        mgcl::CuboidGpu& c_d = p.getLevelAt(0).getDVIn();

        int yz = c_d.getNgh() * c_d.getOgh();
        int xz = c_d.getMgh() * c_d.getOgh();
        int xy = c_d.getMgh() * c_d.getNgh();
        int ressize = 2 * yz * c_d.getGhostsM() + 2 * xz * c_d.getGhostsN() + 2 * xy * c_d.getGhostsO();
        auto& d_planesbuf = p.getDPlanesBuf();

        ankerl::nanobench::Bench bench;
        bench.timeUnit(1ms, "ms")
            .epochs(CLI_ARGS::bench_epochs)
            .epochIterations(CLI_ARGS::bench_iterations);

        if (mpi_rank > 0)
            bench.output(nullptr);

        {
            std::string name = std::string("extractBorderPlanes_newret_")
                                   .append(std::to_string(m))
                                   .append("_")
                                   .append(std::to_string(n))
                                   .append("_")
                                   .append(std::to_string(o))
                                   .append("_gh")
                                   .append(std::to_string(ghosts));

            bench.run(std::string(name).c_str(), [&] { //
                auto sbuf_ptr = c_d.extractBorderPlanes(p.getCommands(), p.getProgram(), &d_planesbuf, nullptr, nullptr, nullptr);
                p.getOpenCLHelper().finish();
            });

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
            res.gpus = mpi_size;
            res.ghosts = ghosts;
            results.push_back(res);
        }

        {
            std::vector<double> h_planesbuf(ressize);
            std::string name = std::string("extractBorderPlanes_reuseret_")
                                   .append(std::to_string(m))
                                   .append("_")
                                   .append(std::to_string(n))
                                   .append("_")
                                   .append(std::to_string(o))
                                   .append("_gh")
                                   .append(std::to_string(ghosts));

            bench.run(std::string(name).c_str(), [&] { //
                c_d.extractBorderPlanes(p.getCommands(), p.getProgram(), &d_planesbuf, &h_planesbuf, nullptr, nullptr);
                p.getOpenCLHelper().finish();
            });

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
            res.gpus = mpi_size;
            res.ghosts = ghosts;
            results.push_back(res);
        }

        {
            std::vector<double> sbuf(ressize);
            std::vector<double> rbuf(ressize);
            std::string name = std::string("sendBorderPlanes")
                                   .append(std::to_string(m))
                                   .append("_")
                                   .append(std::to_string(n))
                                   .append("_")
                                   .append(std::to_string(o))
                                   .append("_gh")
                                   .append(std::to_string(ghosts));

            bench.run(std::string(name).c_str(), [&] { //
                mgcl::mpi_util::sendBorderPlanes(c_d.getMgh(), c_d.getNgh(), c_d.getOgh(),
                                                 c_d.getGhostsM(), c_d.getGhostsN(), c_d.getGhostsO(), 1,
                                                 sbuf, rbuf, mpiData);
            });

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
            res.gpus = mpi_size;
            res.ghosts = ghosts;
            results.push_back(res);
        }

        {
            std::vector<double> sbuf(ressize);
            std::vector<double> rbuf(ressize);
            std::string name = std::string("sendBorderPlanes_copyedges")
                                   .append(std::to_string(m))
                                   .append("_")
                                   .append(std::to_string(n))
                                   .append("_")
                                   .append(std::to_string(o))
                                   .append("_gh")
                                   .append(std::to_string(ghosts));

            // int base_yz_front = 0;
            int base_yz_back = ghosts * yz;
            int base_xz_top = 2 * ghosts * yz;
            int base_xz_bottom = 2 * ghosts * yz + ghosts * xz;
            int base_xy_left = 2 * ghosts * yz + 2 * ghosts * xz;
            int base_xy_right = 2 * ghosts * yz + 2 * ghosts * xz + ghosts * xy;

            bench.run(std::string(name).c_str(), [&] { //
                // Write received edges of cuboid to send buffer
                // i0: i index of recv buffers for yz plane (always all i indices)
                // i1: i index of send buffers xz back ghosts
                // j0: j index of send buffers for xz plane (always all j indices)
                // j1: j index of recv buffer yz top edge
                // j2: j index of recv buffer yz bottom edge
                for (int i0 = 0, i1 = m + ghosts;
                     i0 < ghosts;
                     i0++, i1++)
                    for (int j0 = 0, j1 = ghosts, j2 = n;
                         j0 < ghosts;
                         j0++, j1++, j2++)
                        for (int k = 0; k < ogh; k++)
                        {
                            // Upper front edge - Write ghosts in the front (from back recv buffer) to xz top send buffer
                            sbuf[base_xz_top + j0 * xz + i0 * ogh + k] = rbuf[base_yz_back + i0 * yz + j1 * ogh + k];

                            // Lower front edge - Write ghosts in the front (from back recv buffer) to xz bottom send buffer
                            sbuf[base_xz_bottom + j0 * xz + i0 * ogh + k] = rbuf[base_yz_back + i0 * yz + j2 * ogh + k];

                            // Upper back edge - Write ghosts in the back (from front recv buffer, base 0) to xz top send buffer
                            sbuf[base_xz_top + j0 * xz + i1 * ogh + k] = rbuf[i0 * yz + j1 * ogh + k];

                            // Lower back edge - Write ghosts in the back (from front recv buffer, base 0) to xz bottom send buffer
                            sbuf[base_xz_bottom + j0 * xz + i1 * ogh + k] = rbuf[i0 * yz + j2 * ogh + k];
                        }

                // Write received left torus of cuboid to send buffer
                // k0: k index of send buffers for xy planes (left and right)
                // k1: k index of recv buffers for copy into left send buffer
                // k2: k index of recv buffers for copy into right send buffer
                for (int k0 = 0, k1 = ghosts, k2 = o;
                     k0 < ghosts;
                     k0++, k1++, k2++)
                {

                    // Copying from yz planes (front back)
                    // i0: i index of recv buffers for yz plane (always all i indices)
                    // i1: i index of send buffers xz back ghosts
                    for (int i0 = 0, i1 = m + ghosts;
                         i0 < ghosts;
                         i0++, i1++)
                        for (int j = ghosts; j < ghosts + n; j++)
                        {
                            // Left front face - Write ghosts in the send left buffer from recv back buffer
                            sbuf[base_xy_left + k0 * xy + i0 * ngh + j] = rbuf[base_yz_back + i0 * yz + j * ogh + k1];

                            // Left back face - Write ghosts in the send left buffer from recv front buffer
                            sbuf[base_xy_left + k0 * xy + i1 * ngh + j] = rbuf[i0 * yz + j * ogh + k1];

                            // TODO right
                            // Right front face - Write ghosts in the send right buffer from recv back buffer
                            sbuf[base_xy_right + k0 * xy + i0 * ngh + j] = rbuf[base_yz_back + i0 * yz + j * ogh + k2];

                            // Right back face - Write ghosts in the send right buffer from recv front buffer
                            sbuf[base_xy_right + k0 * xy + i1 * ngh + j] = rbuf[i0 * yz + j * ogh + k2];
                        }

                    // Copying from xz planes (top bottom)
                    // j0: j index of recv buffers yz bottom and send both left and right
                    // j1: j index of send buffers xy bottom ghosts (recv top)
                    for (int i = 0; i < mgh; i++)
                        for (int j0 = 0, j1 = n + ghosts;
                             j0 < ghosts;
                             j0++, j1++)
                        {
                            // Left top edge - Write ghosts in the send left buffer from recv bottom buffer
                            sbuf[base_xy_left + k0 * xy + i * ngh + j0] = rbuf[base_xz_bottom + j0 * xz + i * ogh + k1];

                            // Left bottom edge - Write ghosts in the send left buffer from recv top buffer
                            sbuf[base_xy_left + k0 * xy + i * ngh + j1] = rbuf[base_xz_top + j0 * xz + i * ogh + k1];

                            // Right top edge - Write ghosts in the send left buffer from recv bottom buffer
                            sbuf[base_xy_right + k0 * xy + i * ngh + j0] = rbuf[base_xz_bottom + j0 * xz + i * ogh + k2];

                            // Right bottom face - Write ghosts in the send right buffer from recv top buffer
                            sbuf[base_xy_right + k0 * xy + i * ngh + j1] = rbuf[base_xz_top + j0 * xz + i * ogh + k2];
                        }
                }
            });

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
            res.gpus = mpi_size;
            res.ghosts = ghosts;
            results.push_back(res);
        }

        {
            std::vector<double> rbuf(ressize);
            std::string name = std::string("pasteGhostsFromBorderPlanes_newdevicebuf_N")
                                   .append(std::to_string(m))
                                   .append("_")
                                   .append(std::to_string(n))
                                   .append("_")
                                   .append(std::to_string(o))
                                   .append("_gh")
                                   .append(std::to_string(ghosts));

            bench.run(std::string(name).c_str(), [&] { //
                c_d.pasteGhostsFromBorderPlanes(p.getContext(), p.getCommands(), p.getProgram(), nullptr, &rbuf, nullptr, nullptr);
                p.getOpenCLHelper().finish();
            });

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
            res.gpus = mpi_size;
            res.ghosts = ghosts;
            results.push_back(res);
        }

        {
            std::string name = std::string("pasteGhostsFromBorderPlanes_reusedevicebuf_N")
                                   .append(std::to_string(m))
                                   .append("_")
                                   .append(std::to_string(n))
                                   .append("_")
                                   .append(std::to_string(o))
                                   .append("_gh")
                                   .append(std::to_string(ghosts));

            bench.run(std::string(name).c_str(), [&] { //
                c_d.pasteGhostsFromBorderPlanes(p.getContext(), p.getCommands(), p.getProgram(), &d_planesbuf, nullptr, nullptr, nullptr);
                p.getOpenCLHelper().finish();
            });

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
            res.gpus = mpi_size;
            res.ghosts = ghosts;
            results.push_back(res);
        }
    }

    bench_util::printCsvFormat(results, mpi_comm, mpi_rank);
}

namespace mgcl_bench_ghost_update_wgsizes
{
    enum class KernelVersion
    {
        THREE_D,
        ONE_D,
        THREE_D_OLD_INDEX_CALC, // e.g. int ireal = i + floor(((double)(ghm - 1 - i)) / m + 1) * m;
        SPLIT                   // split into 3 kernels, one per dimension. Also is a 3d kernel.
    };

    using size_t3 = struct
    {
        int x, y, z;
    };

    // Regular jacobi method like in production code, but with mpi stuff removed. I.e. only single gpu
    int updateGhosts(mgcl::Problem& problem, mgcl::CuboidGpu& dBuffer, KernelVersion kernelVersion, std::vector<size_t> wgsizes)
    {
        // TODO actually request these as arguments
        int m = dBuffer.getM();
        int n = dBuffer.getN();
        int o = dBuffer.getO();
        int mgh = dBuffer.getMgh();
        int ngh = dBuffer.getNgh();
        int ogh = dBuffer.getOgh();
        int ghosts_m = dBuffer.getGhostsM();
        int ghosts_n = dBuffer.getGhostsN();
        int ghosts_o = dBuffer.getGhostsO();

        if (!problem.isPeriodic())
            return CL_SUCCESS;

        int err;

        bool is3d = kernelVersion == KernelVersion::THREE_D || kernelVersion == KernelVersion::THREE_D_OLD_INDEX_CALC;

        // Create the compute kernel from the program
        std::string kernelName;
        if (kernelVersion == KernelVersion::THREE_D)
            kernelName = "update_ghosts_periodic_3d";
        else if (kernelVersion == KernelVersion::THREE_D_OLD_INDEX_CALC)
            kernelName = "update_ghosts_periodic_3d_old_index_calc";
        else if (kernelVersion == KernelVersion::ONE_D)
            kernelName = "update_ghosts_periodic_1d";
        cl_kernel kernel = clCreateKernel(problem.getOpenCLHelper().getProgram(), kernelName.c_str(), &err);
        mgcl::mgclCheckError(err, "clCreateKernel");

        // assign kernel arguments
        int pos = 0;
        err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &dBuffer);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &m);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &n);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &o);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghosts_m);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghosts_n);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghosts_o);
        mgcl::mgclCheckError(err, "Setting kernel arguments");

        // one work-item per ghost cell (excluding real cells). Pad global sizes to fit to local sizes
        // int mgh = m + 2 * gh;
        // int ngh = n + 2 * gh;
        // int ogh = o + 2 * gh;
        size_t global[3] = {static_cast<size_t>(ogh), static_cast<size_t>(ngh), static_cast<size_t>(mgh)};
        size_t const local[3] = {wgsizes[0], wgsizes[1], wgsizes[2]};

        if (!is3d)
        {
            global[0] = static_cast<size_t>(mgh * ngh * ogh);
            global[1] = 1;
            global[2] = 1;
        }

        for (int i = 0; i < (is3d ? 3 : 1); i++)
            if (global[i] % local[i] != 0)
                global[i] += local[i] - (global[i] % local[i]);

        cl_event ev;

        // enqueue kernel
        err = clEnqueueNDRangeKernel(problem.getOpenCLHelper().getCommands(), kernel, (is3d ? 3 : 1), NULL, global, local, 0, NULL, &ev);
        mgcl::mgclCheckError(err, "Enqueueing update_ghosts_periodic kernel");

        if (problem.isProfilingEnabled())
        {
            problem.getProfilingData()->addMeasurement(problem.getCommands(), ev, kernelName,
                                                       {global[0], global[1], global[2]},
                                                       {local[0], local[1], local[2]});
        }
        mgcl::mgclCheckError(clReleaseEvent(ev), "clReleaseEvent");

        err = clReleaseKernel(kernel);
        mgcl::mgclCheckError(err, "Releasing update_ghosts_periodic kernel");

        return err;
    }

    int updateGhostsSplit(mgcl::Problem& problem, mgcl::CuboidGpu& dBuffer, KernelVersion kernelVersion, std::vector<size_t> wgsizes)
    {
        // TODO actually request these as arguments
        int m = dBuffer.getM();
        int n = dBuffer.getN();
        int o = dBuffer.getO();
        int mgh = dBuffer.getMgh();
        int ngh = dBuffer.getNgh();
        int ogh = dBuffer.getOgh();
        int ghosts_m = dBuffer.getGhostsM();
        int ghosts_n = dBuffer.getGhostsN();
        int ghosts_o = dBuffer.getGhostsO();

        if (!problem.isPeriodic())
            return CL_SUCCESS;

        if (kernelVersion != KernelVersion::SPLIT)
            throw "Only KernelVersion::SPLIT is supported";

        int err;

        bool is3d = true;

        // Create the compute kernel from the program
        const char* kernelNamex = "update_ghosts_periodic_x";
        const char* kernelNamey = "update_ghosts_periodic_y";
        const char* kernelNamez = "update_ghosts_periodic_z";
        cl_kernel kernelx = clCreateKernel(problem.getOpenCLHelper().getProgram(), kernelNamex, &err);
        mgcl::mgclCheckError(err, "clCreateKernel");
        cl_kernel kernely = clCreateKernel(problem.getOpenCLHelper().getProgram(), kernelNamey, &err);
        mgcl::mgclCheckError(err, "clCreateKernel");
        cl_kernel kernelz = clCreateKernel(problem.getOpenCLHelper().getProgram(), kernelNamez, &err);
        mgcl::mgclCheckError(err, "clCreateKernel");

        // assign kernel arguments
        int pos = 0;
        err = clSetKernelArg(kernelx, pos, sizeof(cl_mem), &dBuffer);
        err |= clSetKernelArg(kernelx, ++pos, sizeof(int), &m);
        err |= clSetKernelArg(kernelx, ++pos, sizeof(int), &n);
        err |= clSetKernelArg(kernelx, ++pos, sizeof(int), &o);
        err |= clSetKernelArg(kernelx, ++pos, sizeof(int), &ghosts_m);
        err |= clSetKernelArg(kernelx, ++pos, sizeof(int), &ghosts_n);
        err |= clSetKernelArg(kernelx, ++pos, sizeof(int), &ghosts_o);
        mgcl::mgclCheckError(err, "Setting kernelx arguments");
        pos = 0;
        err = clSetKernelArg(kernely, pos, sizeof(cl_mem), &dBuffer);
        err |= clSetKernelArg(kernely, ++pos, sizeof(int), &m);
        err |= clSetKernelArg(kernely, ++pos, sizeof(int), &n);
        err |= clSetKernelArg(kernely, ++pos, sizeof(int), &o);
        err |= clSetKernelArg(kernely, ++pos, sizeof(int), &ghosts_m);
        err |= clSetKernelArg(kernely, ++pos, sizeof(int), &ghosts_n);
        err |= clSetKernelArg(kernely, ++pos, sizeof(int), &ghosts_o);
        mgcl::mgclCheckError(err, "Setting kernely arguments");
        pos = 0;
        err = clSetKernelArg(kernelz, pos, sizeof(cl_mem), &dBuffer);
        err |= clSetKernelArg(kernelz, ++pos, sizeof(int), &m);
        err |= clSetKernelArg(kernelz, ++pos, sizeof(int), &n);
        err |= clSetKernelArg(kernelz, ++pos, sizeof(int), &o);
        err |= clSetKernelArg(kernelz, ++pos, sizeof(int), &ghosts_m);
        err |= clSetKernelArg(kernelz, ++pos, sizeof(int), &ghosts_n);
        err |= clSetKernelArg(kernelz, ++pos, sizeof(int), &ghosts_o);
        mgcl::mgclCheckError(err, "Setting kernelz arguments");

        // one work-item per ghost cell (excluding real cells). Pad global sizes to fit to local sizes
        // int mgh = m + 2 * gh;
        // int ngh = n + 2 * gh;
        // int ogh = o + 2 * gh;
        size_t globalx[2] = {static_cast<size_t>(ngh), static_cast<size_t>(ogh)};
        size_t globaly[2] = {static_cast<size_t>(mgh), static_cast<size_t>(ogh)};
        size_t globalz[2] = {static_cast<size_t>(mgh), static_cast<size_t>(ngh)};
        size_t const local[3] = {wgsizes[0], wgsizes[1], wgsizes[2]};

        for (int i = 0; i < 2; i++)
        {
            if (globalx[i] % local[i] != 0)
                globalx[i] += local[i] - (globalx[i] % local[i]);
            if (globaly[i] % local[i] != 0)
                globaly[i] += local[i] - (globaly[i] % local[i]);
            if (globalz[i] % local[i] != 0)
                globalz[i] += local[i] - (globalz[i] % local[i]);
        }

        cl_event ev;

        // enqueue kernel
        err = clEnqueueNDRangeKernel(problem.getOpenCLHelper().getCommands(), kernelx, 2, NULL, globalx, local, 0, NULL, &ev);
        mgcl::mgclCheckError(err, "Enqueueing update_ghosts_periodic kernel");
        err = clEnqueueNDRangeKernel(problem.getOpenCLHelper().getCommands(), kernely, 2, NULL, globaly, local, 0, NULL, &ev);
        mgcl::mgclCheckError(err, "Enqueueing update_ghosts_periodic kernel");
        err = clEnqueueNDRangeKernel(problem.getOpenCLHelper().getCommands(), kernelz, 2, NULL, globalz, local, 0, NULL, &ev);
        mgcl::mgclCheckError(err, "Enqueueing update_ghosts_periodic kernel");

        if (problem.isProfilingEnabled())
        {
            problem.getProfilingData()->addMeasurement(problem.getCommands(), ev, kernelNamex,
                                                       {globalx[0], globalx[1], 0},
                                                       {local[0], local[1], local[2]});
            problem.getProfilingData()->addMeasurement(problem.getCommands(), ev, kernelNamey,
                                                       {globaly[0], globaly[1], 0},
                                                       {local[0], local[1], local[2]});
            problem.getProfilingData()->addMeasurement(problem.getCommands(), ev, kernelNamez,
                                                       {globalz[0], globalz[1], 0},
                                                       {local[0], local[1], local[2]});
        }
        mgcl::mgclCheckError(clReleaseEvent(ev), "clReleaseEvent");

        err = clReleaseKernel(kernelx);
        mgcl::mgclCheckError(err, "Releasing update_ghosts_periodic kernel");
        err = clReleaseKernel(kernely);
        mgcl::mgclCheckError(err, "Releasing update_ghosts_periodic kernel");
        err = clReleaseKernel(kernelz);
        mgcl::mgclCheckError(err, "Releasing update_ghosts_periodic kernel");

        return err;
    }

    // Benchs the ghost update of CuboidGpu for different workgroup sizes.
    TEST_CASE("benchGhostUpdateWgSizes")
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

        std::vector<bench_util::ResultMpi> results;

        int ghosts = 1;
        int periodic = 1;
        std::stringstream kernelProfilesStream;

        for (auto stepsPerIter : CLI_ARGS::jacobiStepsPerIter)
            if (stepsPerIter > ghosts)
                throw "stepsPerIter must be <= ghosts. Not supported yet.";

        // check if mpi is initialized
        int isInitialized = 0;
        MPI_Initialized(&isInitialized);
        REQUIRE(isInitialized);

        MPI_Comm mpi_comm = MPI_COMM_WORLD;

        // check number of processes
        int mpi_size = -1;
        MPI_Comm_size(mpi_comm, &mpi_size);
        // REQUIRE(mpi_size == 8);

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

        for (auto gr : gridsTBT)
        {
            int ml = gr[0];
            int nl = gr[1];
            int ol = gr[2];
            int mglob = ml * mpi_dims[0];
            int nglob = nl * mpi_dims[1];
            int oglob = ol * mpi_dims[2];

            CAPTURE(ml, nl, ol, mglob, nglob, oglob);

            // print coords and boundaries per rank
            // if (mpi_rank == 0)
            //     std::cout << "rank;coords[0];coords[1];coords[2];ms;me;ns;ne;os;oe" << std::endl;

            // for (int i = 0; i < mpi_size; i++)
            // {
            //     MPI_Barrier(mpi_comm);
            //     if (mpi_rank == i)
            //     {
            //         std::cout << mpi_rank << ";" << mpi_coords[0] << ";" << mpi_coords[1] << ";" << mpi_coords[2] << ";"
            //                   << m_start << ";" << m_end << ";"
            //                   << n_start << ";" << n_end << ";"
            //                   << o_start << ";" << o_end << std::endl;
            //     }
            // }

            REQUIRE(ml > 0);
            REQUIRE(ml <= mglob);
            REQUIRE(nl > 0);
            REQUIRE(nl <= nglob);
            REQUIRE(ol > 0);
            REQUIRE(ol <= oglob);

            auto v_in = std::make_shared<mgcl::Cuboid>(ml, nl, ol, 0, 0, 0);
            auto f_in = std::make_shared<mgcl::Cuboid>(ml, nl, ol, 0, 0, 0);
            v_in->fill1dIndex(true);
            f_in->fill1dIndex(true);
            // v_in->fillRandom();
            // f_in->fillRandom();

            // Create dummy problem to initialize OpenCL
            mgcl::Problem p(ml, nl, ol, f_in, v_in, mglob, nglob, oglob);
            p.setSilent(true);
            p.setKernelFile("kernels_ghost_update.cl");
            if (CLI_ARGS::useBinaryFile)
            {
                p.setBinaryFile("kernels_ghost_update.bin");
            }
            p.setUseOpencl(true);
            p.setGhosts(ghosts);
            p.setStencilType(mgcl::MGCL_LAPLACE_7POINT);
            p.setDeviceType(CL_DEVICE_TYPE_GPU);
            p.setDeviceStrategy(mgcl::OCL_DEVICE_STRATEGY::DISTRIBUTE_EVENLY);
            p.setProfilingEnabled(CLI_ARGS::enableKernelProfiling);
            p.setMpiComm(mpi_comm);

            // auto& conf = p.getKernelConfig();
            // // Jacobi kernels
            // conf["jacobi_iter_27point_varying_stencil_1d_update_step_only"] = mgcl::conf::KernelWorkgroupSizes{{1, {32, 1, 1}}};
            p.init();

            if (CLI_ARGS::enableKernelProfiling)
                p.getProfilingData()->getMeasurements().clear();

            auto& lv0 = p.getLevelAt(0);

            ankerl::nanobench::Bench bench;
            bench.timeUnit(1ms, "ms")
                .epochs(CLI_ARGS::bench_epochs)
                .epochIterations(CLI_ARGS::bench_iterations)
                .relative(false);

            if (mpi_rank > 0)
                bench.output(nullptr);

            if (CLI_ARGS::checkResults)
            {
                bench.epochs(1).epochIterations(1);
            }

            std::vector<std::vector<size_t>> wg_sizes_1d = {{4, 1, 1}, {8, 1, 1}, {32, 1, 1}, {64, 1, 1}, {128, 1, 1}, {256, 1, 1}};
            for (auto wg : wg_sizes_1d)
            {
                lv0.getDVIn().fill(p.getProgram(), p.getCommands(), 0.0, false, nullptr, nullptr);
                lv0.getDVIn().fill1dIndex(p.getProgram(), p.getCommands(), true, true, nullptr, nullptr);

                std::string name = std::string("ghost_update_1d_")
                                       .append(std::to_string(mglob))
                                       .append("_")
                                       .append(std::to_string(nglob))
                                       .append("_")
                                       .append(std::to_string(oglob))
                                       .append("_wg")
                                       .append(std::to_string(wg[0]))
                                       .append("x")
                                       .append(std::to_string(wg[1]))
                                       .append("x")
                                       .append(std::to_string(wg[2]));

                bench.run(std::string(name).c_str(), [&] { //
                    updateGhosts(p, lv0.getDVIn(), KernelVersion::ONE_D, wg);
                    p.finish();
                });

                bench_util::ResultMpi res;
                res.name = name;
                res.minTime = bench_util::getMinTime(bench, name);
                res.medianTime = bench_util::getMedianTime(bench, name);
                res.avgTime = bench_util::getAvgTime(bench, name);
                res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
                res.m = ml;
                res.n = nl;
                res.o = ol;
                res.mglob = mglob;
                res.nglob = nglob;
                res.oglob = oglob;
                res.gpus = mpi_size;
                res.LT = -1;
                results.push_back(res);

                // if (CLI_ARGS::checkResults)
                // {
                //     v_out_default = std::make_unique<mgcl::Cuboid>(ml, nl, ol, ghosts, ghosts, ghosts);
                //     lv0.getDVIn().read(p.getCommands(), v_out_default.get(), true);
                // }
            }

            std::vector<std::vector<size_t>> wg_sizes_3d = {{4, 4, 4}, {4, 4, 8}, {2, 2, 8}, {8, 8, 8}, {4, 8, 8}, {4, 4, 16}, {8, 4, 4}, {32, 1, 1}, {64, 1, 1}};
            for (auto wg : wg_sizes_3d)
            {
                {
                    lv0.getDVIn().fill(p.getProgram(), p.getCommands(), 0.0, false, nullptr, nullptr);
                    lv0.getDVIn().fill1dIndex(p.getProgram(), p.getCommands(), true, true, nullptr, nullptr);

                    std::string name = std::string("ghost_update_3d_")
                                           .append(std::to_string(mglob))
                                           .append("_")
                                           .append(std::to_string(nglob))
                                           .append("_")
                                           .append(std::to_string(oglob))
                                           .append("_wg")
                                           .append(std::to_string(wg[0]))
                                           .append("x")
                                           .append(std::to_string(wg[1]))
                                           .append("x")
                                           .append(std::to_string(wg[2]));

                    bench.run(std::string(name).c_str(), [&] { //
                        updateGhosts(p, lv0.getDVIn(), KernelVersion::THREE_D, wg);
                        p.finish();
                    });

                    bench_util::ResultMpi res;
                    res.name = name;
                    res.minTime = bench_util::getMinTime(bench, name);
                    res.medianTime = bench_util::getMedianTime(bench, name);
                    res.avgTime = bench_util::getAvgTime(bench, name);
                    res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
                    res.m = ml;
                    res.n = nl;
                    res.o = ol;
                    res.mglob = mglob;
                    res.nglob = nglob;
                    res.oglob = oglob;
                    res.gpus = mpi_size;
                    res.LT = -1;
                    results.push_back(res);

                    // if (CLI_ARGS::checkResults)
                    // {
                    //     v_out_default = std::make_unique<mgcl::Cuboid>(ml, nl, ol, ghosts, ghosts, ghosts);
                    //     lv0.getDVIn().read(p.getCommands(), v_out_default.get(), true);
                    // } }
                }

                {
                    lv0.getDVIn().fill(p.getProgram(), p.getCommands(), 0.0, false, nullptr, nullptr);
                    lv0.getDVIn().fill1dIndex(p.getProgram(), p.getCommands(), true, true, nullptr, nullptr);

                    std::string name = std::string("ghost_update_3d_old_index_calc_")
                                           .append(std::to_string(mglob))
                                           .append("_")
                                           .append(std::to_string(nglob))
                                           .append("_")
                                           .append(std::to_string(oglob))
                                           .append("_wg")
                                           .append(std::to_string(wg[0]))
                                           .append("x")
                                           .append(std::to_string(wg[1]))
                                           .append("x")
                                           .append(std::to_string(wg[2]));

                    bench.run(std::string(name).c_str(), [&] { //
                        updateGhosts(p, lv0.getDVIn(), KernelVersion::THREE_D_OLD_INDEX_CALC, wg);
                        p.finish();
                    });

                    bench_util::ResultMpi res;
                    res.name = name;
                    res.minTime = bench_util::getMinTime(bench, name);
                    res.medianTime = bench_util::getMedianTime(bench, name);
                    res.avgTime = bench_util::getAvgTime(bench, name);
                    res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
                    res.m = ml;
                    res.n = nl;
                    res.o = ol;
                    res.mglob = mglob;
                    res.nglob = nglob;
                    res.oglob = oglob;
                    res.gpus = mpi_size;
                    res.LT = -1;
                    results.push_back(res);

                    // if (CLI_ARGS::checkResults)
                    // {
                    //     v_out_default = std::make_unique<mgcl::Cuboid>(ml, nl, ol, ghosts, ghosts, ghosts);
                    //     lv0.getDVIn().read(p.getCommands(), v_out_default.get(), true);
                    // } }
                }

                if (false)
                {
                    lv0.getDVIn().fill(p.getProgram(), p.getCommands(), 0.0, false, nullptr, nullptr);
                    lv0.getDVIn().fill1dIndex(p.getProgram(), p.getCommands(), true, true, nullptr, nullptr);

                    std::string name = std::string("ghost_update_3d_")
                                           .append(std::to_string(mglob))
                                           .append("_")
                                           .append(std::to_string(nglob))
                                           .append("_")
                                           .append(std::to_string(oglob))
                                           .append("_wg")
                                           .append(std::to_string(wg[0]))
                                           .append("x")
                                           .append(std::to_string(wg[1]))
                                           .append("x")
                                           .append(std::to_string(wg[2]));

                    bench.run(std::string(name).c_str(), [&] { //
                        updateGhostsSplit(p, lv0.getDVIn(), KernelVersion::SPLIT, wg);
                        p.finish();
                    });

                    bench_util::ResultMpi res;
                    res.name = name;
                    res.minTime = bench_util::getMinTime(bench, name);
                    res.medianTime = bench_util::getMedianTime(bench, name);
                    res.avgTime = bench_util::getAvgTime(bench, name);
                    res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
                    res.m = ml;
                    res.n = nl;
                    res.o = ol;
                    res.mglob = mglob;
                    res.nglob = nglob;
                    res.oglob = oglob;
                    res.gpus = mpi_size;
                    res.LT = -1;
                    results.push_back(res);

                    // if (CLI_ARGS::checkResults)
                    // {
                    //     v_out_default = std::make_unique<mgcl::Cuboid>(ml, nl, ol, ghosts, ghosts, ghosts);
                    //     lv0.getDVIn().read(p.getCommands(), v_out_default.get(), true);
                    // } }
                }
            }

            // call regular ghpst update that is in production code once for kernel timing comparison
            auto& conf = p.getKernelConfig();
            // Jacobi kernels
            conf["update_ghosts_periodic"] = mgcl::conf::KernelWorkgroupSizes{{1, {4, 4, 8}}};
            mgcl::MultigridEngine::updateGhosts(p, lv0.getDVIn(), nullptr, true);

            if (CLI_ARGS::enableKernelProfiling)
            {
                p.getProfilingData()->printBestTimingsPerKernel(kernelProfilesStream);
            }

            MPI_Barrier(mpi_comm);
            bench_util::printCsvFormat(results, mpi_comm, mpi_rank);
            MPI_Barrier(mpi_comm);

            if (CLI_ARGS::enableKernelProfiling)
            {
                kernelProfilesStream << "rank: " << mpi_rank << std::endl;
                std::cout << kernelProfilesStream.str() << std::endl;
            }
            MPI_Barrier(mpi_comm);
        }
    }

    // Benchs the ghost update of CuboidGpu for different ghost layer sizes, i.e. ghost amounts.
    TEST_CASE("benchGhostUpdateLayerSize")
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

        std::vector<bench_util::ResultGhostUpdateMpi> results;

        int ghosts = 1;
        int periodic = 1;
        std::stringstream kernelProfilesStream;

        for (auto stepsPerIter : CLI_ARGS::jacobiStepsPerIter)
            if (stepsPerIter > ghosts)
                throw "stepsPerIter must be <= ghosts. Not supported yet.";

        // check if mpi is initialized
        int isInitialized = 0;
        MPI_Initialized(&isInitialized);
        REQUIRE(isInitialized);

        MPI_Comm mpi_comm = MPI_COMM_WORLD;

        // check number of processes
        int mpi_size = -1;
        MPI_Comm_size(mpi_comm, &mpi_size);
        // REQUIRE(mpi_size == 8);

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

        for (auto gr : gridsTBT)
        {
            int ml = gr[0];
            int nl = gr[1];
            int ol = gr[2];
            int mglob = ml * mpi_dims[0];
            int nglob = nl * mpi_dims[1];
            int oglob = ol * mpi_dims[2];

            CAPTURE(ml, nl, ol, mglob, nglob, oglob);

            // print coords and boundaries per rank
            // if (mpi_rank == 0)
            //     std::cout << "rank;coords[0];coords[1];coords[2];ms;me;ns;ne;os;oe" << std::endl;

            // for (int i = 0; i < mpi_size; i++)
            // {
            //     MPI_Barrier(mpi_comm);
            //     if (mpi_rank == i)
            //     {
            //         std::cout << mpi_rank << ";" << mpi_coords[0] << ";" << mpi_coords[1] << ";" << mpi_coords[2] << ";"
            //                   << m_start << ";" << m_end << ";"
            //                   << n_start << ";" << n_end << ";"
            //                   << o_start << ";" << o_end << std::endl;
            //     }
            // }

            REQUIRE(ml > 0);
            REQUIRE(ml <= mglob);
            REQUIRE(nl > 0);
            REQUIRE(nl <= nglob);
            REQUIRE(ol > 0);
            REQUIRE(ol <= oglob);

            auto v_in = std::make_shared<mgcl::Cuboid>(ml, nl, ol, 0, 0, 0);
            auto f_in = std::make_shared<mgcl::Cuboid>(ml, nl, ol, 0, 0, 0);
            v_in->fill1dIndex(true);
            f_in->fill1dIndex(true);
            // v_in->fillRandom();
            // f_in->fillRandom();

            // Create dummy problem to initialize OpenCL
            mgcl::Problem p(ml, nl, ol, f_in, v_in, mglob, nglob, oglob);
            p.setSilent(true);
            p.setKernelFile("kernels_ghost_update.cl");
            if (CLI_ARGS::useBinaryFile)
            {
                p.setBinaryFile("kernels_ghost_update.bin");
            }
            p.setUseOpencl(true);
            p.setGhosts(ghosts);
            p.setStencilType(mgcl::MGCL_LAPLACE_7POINT);
            p.setDeviceType(CL_DEVICE_TYPE_GPU);
            p.setDeviceStrategy(mgcl::OCL_DEVICE_STRATEGY::DISTRIBUTE_EVENLY);
            p.setProfilingEnabled(CLI_ARGS::enableKernelProfiling);
            p.setMpiComm(mpi_comm);

            // auto& conf = p.getKernelConfig();
            // // Jacobi kernels
            // conf["jacobi_iter_27point_varying_stencil_1d_update_step_only"] = mgcl::conf::KernelWorkgroupSizes{{1, {32, 1, 1}}};
            p.init();

            if (CLI_ARGS::enableKernelProfiling)
                p.getProfilingData()->getMeasurements().clear();

            auto& lv0 = p.getLevelAt(0);

            ankerl::nanobench::Bench bench;
            bench.timeUnit(1ms, "ms")
                .epochs(CLI_ARGS::bench_epochs)
                .epochIterations(CLI_ARGS::bench_iterations)
                .relative(false);

            if (mpi_rank > 0)
                bench.output(nullptr);

            if (CLI_ARGS::checkResults)
            {
                bench.epochs(1).epochIterations(1);
            }

            std::vector<size_t> gh_counts = {1, 2, 3, 4, 5};
            std::vector<std::vector<size_t>> wg_sizes_3d = {{4, 4, 4}, {32, 1, 1}};
            for (auto gh : gh_counts)
            {
                mgcl::CuboidGpu c(p.getContext(), CL_MEM_READ_WRITE, ml, nl, ol, gh, gh, gh);
                for (auto wg : wg_sizes_3d)
                {
                    c.fill(p.getProgram(), p.getCommands(), 0.0, false, nullptr, nullptr);
                    c.fill1dIndex(p.getProgram(), p.getCommands(), true, true, nullptr, nullptr);

                    std::string name = std::string("ghost_update_3d_")
                                           .append(std::to_string(mglob))
                                           .append("_")
                                           .append(std::to_string(nglob))
                                           .append("_")
                                           .append(std::to_string(oglob))
                                           .append("_wg")
                                           .append(std::to_string(wg[0]))
                                           .append("x")
                                           .append(std::to_string(wg[1]))
                                           .append("x")
                                           .append(std::to_string(wg[2]));

                    bench.run(std::string(name).c_str(), [&] { //
                        updateGhosts(p, c, KernelVersion::THREE_D, wg);
                        p.finish();
                    });

                    bench_util::ResultGhostUpdateMpi res;

                    res.name = name;
                    res.minTime = bench_util::getMinTime(bench, name);
                    res.medianTime = bench_util::getMedianTime(bench, name);
                    res.avgTime = bench_util::getAvgTime(bench, name);
                    res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
                    res.mloc = ml;
                    res.nloc = nl;
                    res.oloc = ol;
                    res.mglob = mglob;
                    res.nglob = nglob;
                    res.oglob = oglob;
                    res.ghosts = gh;     // amount of ghost cells in one direction
                    res.gpus = mpi_size; // GPU-count, equals mpi proc count
                    results.push_back(res);

                    // if (CLI_ARGS::checkResults)
                    // {
                    //     v_out_default = std::make_unique<mgcl::Cuboid>(ml, nl, ol, ghosts, ghosts, ghosts);
                    //     lv0.getDVIn().read(p.getCommands(), v_out_default.get(), true);
                    // }
                }
            }

            if (CLI_ARGS::enableKernelProfiling)
            {
                p.getProfilingData()->printBestTimingsPerKernel(kernelProfilesStream);
            }
        }

        MPI_Barrier(mpi_comm);
        bench_util::printCsvFormat(results, mpi_comm, mpi_rank);
        MPI_Barrier(mpi_comm);

        if (CLI_ARGS::enableKernelProfiling)
        {
            kernelProfilesStream << "rank: " << mpi_rank << std::endl;
            std::cout << kernelProfilesStream.str() << std::endl;
        }
        MPI_Barrier(mpi_comm);
    }
}