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

// The benchmarks in this file are aimig to measure the performance of the data transfer between the
// gpu and the host, as well as the time for MPI communication between processes.

// Benchmark data transfer between host and device witout MPI.
// If called with multiple processes, only root process runs the benchmarks.
// The following tests are done:
// 1. host_to_device_reusing_els<elements>: data transfer from host to device reusing device buffer
// 2. host_to_device_newbuf_els<elements>: data transfer from host to device creating a device new buffer
// 3. device_to_host_reusing_sl<elements>: data transfer from device to host reusing host buffer
// 4. device_to_host_newbuf_els<elements>: data transfer from device to host creating a host new buffer
// 5. device_to_host_to_device_els_<elements>: 4. + 1.
// Run with e.g.: benchmarks bench_data_transfer_host_device
TEST_CASE("bench_data_transfer_host_device")
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

    /* Initialize cartesian process grid */
    MPI_Comm_size(mpi_comm, &mpi_size);
    MPI_Comm_rank(mpi_comm, &mpi_rank);

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

    std::vector<bench_util::ResultDataTransferMpi> results;

    bool printedGpu = false;
    for (auto gr : gridsTBT)
    {
        int m = gr[0];
        int n = gr[1];
        int o = gr[2];
        int elements = m * n * o;

        // Create a dummy problem
        auto v = std::make_shared<mgcl::Cuboid>(1, 1, 1);
        auto f = std::make_shared<mgcl::Cuboid>(1, 1, 1);
        mgcl::Problem p(1, 1, 1, f, v);
        p.setUseOpencl(true);
        p.setSilent(true);
        // p.setMpiComm(mpi_comm);
        p.init();

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

        // actual test buffers
        mgcl::Cuboid c_h(m, n, o);
        c_h.fillRandom();
        mgcl::CuboidGpu c_d(p.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, c_h);

        ankerl::nanobench::Bench bench;
        bench.timeUnit(1ms, "ms")
            .epochs(CLI_ARGS::bench_epochs)
            .epochIterations(CLI_ARGS::bench_iterations);

        {
            std::string name = std::string("host_to_device_reusing_els")
                                   .append(std::to_string(elements));
            bench.run(std::string(name).c_str(), [&] { //
                c_d.write(p.getCommands(), c_h, true);
                p.getOpenCLHelper().finish();
            });

            bench_util::ResultDataTransferMpi res;
            res.name = name;
            res.minTime = bench_util::getMinTime(bench, name);
            res.medianTime = bench_util::getMedianTime(bench, name);
            res.avgTime = bench_util::getAvgTime(bench, name);
            res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
            res.elements = elements;
            res.gpus = mpi_size;
            results.push_back(res);
        }

        {
            std::string name2 = std::string("device_to_host_reusing_els")
                                    .append(std::to_string(elements));
            bench.run(std::string(name2).c_str(), [&] { //
                c_d.read(p.getCommands(), &c_h, true);
                p.getOpenCLHelper().finish();
            });

            bench_util::ResultDataTransferMpi res2;
            res2.name = name2;
            res2.minTime = bench_util::getMinTime(bench, name2);
            res2.medianTime = bench_util::getMedianTime(bench, name2);
            res2.avgTime = bench_util::getAvgTime(bench, name2);
            res2.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name2);
            res2.elements = elements;
            res2.gpus = mpi_size;
            results.push_back(res2);
        }

        {
            std::string name = std::string("host_to_device_newbuf_els")
                                   .append(std::to_string(elements));
            bench.run(std::string(name).c_str(), [&] { //
                ankerl::nanobench::doNotOptimizeAway(
                    mgcl::CuboidGpu(p.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, c_h));
                p.getOpenCLHelper().finish();
            });

            bench_util::ResultDataTransferMpi res;
            res.name = name;
            res.minTime = bench_util::getMinTime(bench, name);
            res.medianTime = bench_util::getMedianTime(bench, name);
            res.avgTime = bench_util::getAvgTime(bench, name);
            res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
            res.elements = elements;
            res.gpus = mpi_size;
            results.push_back(res);
        }

        {
            std::string name = std::string("device_to_host_newbuf_els")
                                   .append(std::to_string(elements));
            bench.run(std::string(name).c_str(), [&] { //
                ankerl::nanobench::doNotOptimizeAway(c_d.read(p.getCommands(), nullptr, true));
                p.getOpenCLHelper().finish();
            });

            bench_util::ResultDataTransferMpi res;
            res.name = name;
            res.minTime = bench_util::getMinTime(bench, name);
            res.medianTime = bench_util::getMedianTime(bench, name);
            res.avgTime = bench_util::getAvgTime(bench, name);
            res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
            res.elements = elements;
            res.gpus = mpi_size;
            results.push_back(res);
        }

        {
            std::string name = std::string("device_to_host_to_device_els")
                                   .append(std::to_string(elements));
            bench.run(std::string(name).c_str(), [&] { //
                auto tmp = c_d.read(p.getCommands(), nullptr, true);
                c_d.write(p.getCommands(), *tmp, true);
                p.getOpenCLHelper().finish();
            });

            bench_util::ResultDataTransferMpi res;
            res.name = name;
            res.minTime = bench_util::getMinTime(bench, name);
            res.medianTime = bench_util::getMedianTime(bench, name);
            res.avgTime = bench_util::getAvgTime(bench, name);
            res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
            res.elements = elements;
            res.gpus = mpi_size;
            results.push_back(res);
        }
    }

    bench_util::printCsvFormat(results, mpi_comm, mpi_rank);
}