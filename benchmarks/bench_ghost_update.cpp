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
                    p.getOpenCLHelper().outputDeviceInfo(p.getOpenCLHelper().getDeviceId());
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
            mgcl::Cuboid h_planesbuf(1, 1, ressize);
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
            mgcl::Cuboid sbuf(1, 1, ressize);
            mgcl::Cuboid rbuf(1, 1, ressize);
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
                                                 c_d.getGhostsM(), c_d.getGhostsN(), c_d.getGhostsO(),
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
            mgcl::Cuboid sbuf(1, 1, ressize);
            mgcl::Cuboid rbuf(1, 1, ressize);
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
                            sbuf[0][0][base_xz_top + j0 * xz + i0 * ogh + k] = rbuf[0][0][base_yz_back + i0 * yz + j1 * ogh + k];

                            // Lower front edge - Write ghosts in the front (from back recv buffer) to xz bottom send buffer
                            sbuf[0][0][base_xz_bottom + j0 * xz + i0 * ogh + k] = rbuf[0][0][base_yz_back + i0 * yz + j2 * ogh + k];

                            // Upper back edge - Write ghosts in the back (from front recv buffer, base 0) to xz top send buffer
                            sbuf[0][0][base_xz_top + j0 * xz + i1 * ogh + k] = rbuf[0][0][i0 * yz + j1 * ogh + k];

                            // Lower back edge - Write ghosts in the back (from front recv buffer, base 0) to xz bottom send buffer
                            sbuf[0][0][base_xz_bottom + j0 * xz + i1 * ogh + k] = rbuf[0][0][i0 * yz + j2 * ogh + k];
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
                            sbuf[0][0][base_xy_left + k0 * xy + i0 * ngh + j] = rbuf[0][0][base_yz_back + i0 * yz + j * ogh + k1];

                            // Left back face - Write ghosts in the send left buffer from recv front buffer
                            sbuf[0][0][base_xy_left + k0 * xy + i1 * ngh + j] = rbuf[0][0][i0 * yz + j * ogh + k1];

                            // TODO right
                            // Right front face - Write ghosts in the send right buffer from recv back buffer
                            sbuf[0][0][base_xy_right + k0 * xy + i0 * ngh + j] = rbuf[0][0][base_yz_back + i0 * yz + j * ogh + k2];

                            // Right back face - Write ghosts in the send right buffer from recv front buffer
                            sbuf[0][0][base_xy_right + k0 * xy + i1 * ngh + j] = rbuf[0][0][i0 * yz + j * ogh + k2];
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
                            sbuf[0][0][base_xy_left + k0 * xy + i * ngh + j0] = rbuf[0][0][base_xz_bottom + j0 * xz + i * ogh + k1];

                            // Left bottom edge - Write ghosts in the send left buffer from recv top buffer
                            sbuf[0][0][base_xy_left + k0 * xy + i * ngh + j1] = rbuf[0][0][base_xz_top + j0 * xz + i * ogh + k1];

                            // Right top edge - Write ghosts in the send left buffer from recv bottom buffer
                            sbuf[0][0][base_xy_right + k0 * xy + i * ngh + j0] = rbuf[0][0][base_xz_bottom + j0 * xz + i * ogh + k2];

                            // Right bottom face - Write ghosts in the send right buffer from recv top buffer
                            sbuf[0][0][base_xy_right + k0 * xy + i * ngh + j1] = rbuf[0][0][base_xz_top + j0 * xz + i * ogh + k2];
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
            mgcl::Cuboid rbuf(1, 1, ressize);
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
