#include "nanobench.h"

#include "catch2/catch_test_macros.hpp"

#include <chrono>
#include <iostream>
#include <mpi.h>
#include <vector>
using namespace std::chrono_literals;

#include "../src/mgcl/cuboid.hpp"
#include "../src/mgcl/mpi_util.hpp"
#include "../src/mgcl/multigrid_engine.hpp"
#include "../src/mgcl/opencl_helper.hpp"
#include "../src/mgcl/problem.hpp"
#include "bench_util.hpp"
#include "cli_args.hpp"

// The benchmarks in this file are aimig to measure the performance of the data transfer between the
// gpu and the host, as well as the time for MPI communication between processes.

// Benchmark data transfer between host and device without MPI.
// If called with multiple processes, only root process runs the benchmarks.
// The following tests are done:
// 1. host_to_device_reusing_els<elements>: data transfer from host to device reusing device buffer
// 2. host_to_device_newbuf_els<elements>: data transfer from host to device creating a device new buffer
// 3. device_to_host_reusing_sl<elements>: data transfer from device to host reusing host buffer
// 4. device_to_host_newbuf_els<elements>: data transfer from device to host creating a host new buffer
// 5. device_to_host_to_device_els_<elements>: 4. + 1.
// Run with e.g.: benchmarks bench_data_transfer_host_device
// E.g. for the grid 128^3 and gh=1, ghost planes are 6*130^2 = 101,400 elements
TEST_CASE("bench_data_transfer_host_device")
{
    using std::min;

    if (CLI_ARGS::elements.size() == 0)
        throw "Need to specify at least one element count, e.g. using --elements 16384,32768";

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
        std::cout << "Testing the following element counts" << std::endl
                  << "  ";
        for (auto e : CLI_ARGS::elements)
        {
            std::cout << e << ", ";
        }
        std::cout << std::endl;
    }
    else
    {
        std::cout << "Doing nothing on rank " << mpi_rank << std::endl;
        return;
    }

    std::vector<bench_util::ResultDataTransferMpi> results;
    int err;
    bool printedGpu = false;
    for (auto el : CLI_ARGS::elements)
    {
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
        mgcl::Cuboid c_h(1, 1, el);
        c_h.fillRandom();
        mgcl::CuboidGpu c_d(p.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, c_h);

        ankerl::nanobench::Bench bench;
        bench.timeUnit(1ms, "ms")
            .epochs(CLI_ARGS::bench_epochs)
            .epochIterations(CLI_ARGS::bench_iterations);

        {
            std::string name = std::string("host_to_device_reusing_els")
                                   .append(std::to_string(el));
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
            res.elements = el;
            res.gpus = mpi_size;
            results.push_back(res);
        }

        {
            std::string name2 = std::string("device_to_host_reusing_els")
                                    .append(std::to_string(el));
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
            res2.elements = el;
            res2.gpus = mpi_size;
            results.push_back(res2);
        }

        {
            std::string name = std::string("host_to_device_newbuf_els")
                                   .append(std::to_string(el));
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
            res.elements = el;
            res.gpus = mpi_size;
            results.push_back(res);
        }

        {
            std::string name = std::string("device_to_host_newbuf_els")
                                   .append(std::to_string(el));
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
            res.elements = el;
            res.gpus = mpi_size;
            results.push_back(res);
        }

        {
            std::string name = std::string("device_to_host_to_device_els")
                                   .append(std::to_string(el));
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
            res.elements = el;
            res.gpus = mpi_size;
            results.push_back(res);
        }
    }

    bench_util::printCsvFormat(results, mpi_comm, mpi_rank);
}

// This test tests the transfer rate between at least 3 MPI processes as done in ghost update.
// 3 processes is the minimum requirement, so MPI_Sendrecv can send and receive to/from different processes.
// TODO: Check MPI_Ssend vs MPI_Bsend
// TODO: MPI_Sendrecv vs. MPI_Sendrecv_replace
TEST_CASE("bench_data_transfer_MPI")
{
    using std::min;

    if (CLI_ARGS::elements.size() == 0)
        throw "Need to specify at least one element count, e.g. using --elements 16384,32768";

    // Check if mpi is initialized
    int isInitialized = 0;
    MPI_Initialized(&isInitialized);
    REQUIRE(isInitialized);

    MPI_Comm mpi_comm = MPI_COMM_WORLD;

    // Check number of processes
    int mpi_size = -1;
    MPI_Comm_size(mpi_comm, &mpi_size);
    REQUIRE(mpi_size == 3);

    int periodic = 1;

    /* MPI variables */
    int mpi_rank;

    /* Initialize cartesian process grid */
    MPI_Comm_size(mpi_comm, &mpi_size);
    MPI_Comm_rank(mpi_comm, &mpi_rank);

    if (mpi_rank == 0)
    {
        std::cout << "Testing the following element counts" << std::endl
                  << "  ";
        for (auto e : CLI_ARGS::elements)
        {
            std::cout << e << ", ";
        }
        std::cout << std::endl;
    }

    std::vector<bench_util::ResultDataTransferMpi> results;
    int err;
    for (auto el : CLI_ARGS::elements)
    {
        // actual test buffers
        double* sbuf = new double[el];
        double* rbuf = new double[el];

        ankerl::nanobench::Bench bench;
        bench.timeUnit(1ms, "ms")
            .epochs(CLI_ARGS::bench_epochs)
            .epochIterations(CLI_ARGS::bench_iterations);

        // output only on root process
        if (mpi_rank > 0)
            bench.output(nullptr);

        MPI_Barrier(MPI_COMM_WORLD);
        if (mpi_rank <= 1)
        { // uses only two processes
            std::string name = std::string("MPI_Send_")
                                   .append(std::to_string(el));

            if (mpi_rank == 0)
            {
                bench.run(std::string(name).c_str(), [&] { //
                    err = MPI_Send(static_cast<void*>(sbuf), el, MPI_DOUBLE, 1, 0,
                                   MPI_COMM_WORLD);
                    mgcl::mpi_util::mgclCheckMpiError(MPI_COMM_WORLD, err, "MPI_Send");
                });
            }
            else if (mpi_rank == 1)
            {
                bench.run(std::string(name).c_str(), [&] { //
                    err = MPI_Recv(static_cast<void*>(rbuf), el, MPI_DOUBLE, 0, 0,
                                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    mgcl::mpi_util::mgclCheckMpiError(MPI_COMM_WORLD, err, "MPI_Recv");
                });
            }

            bench_util::ResultDataTransferMpi res;
            res.name = name;
            res.minTime = bench_util::getMinTime(bench, name);
            res.medianTime = bench_util::getMedianTime(bench, name);
            res.avgTime = bench_util::getAvgTime(bench, name);
            res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
            res.elements = el;
            res.gpus = mpi_size;
            results.push_back(res);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        { // uses three processes
            int idx_recipients[3] = {1, 2, 0};
            int idx_senders[3] = {2, 0, 1};
            std::string name = std::string("MPI_Sendrecv_")
                                   .append(std::to_string(el));

            bench.run(std::string(name).c_str(), [&] { //
                err = MPI_Sendrecv(static_cast<void*>(sbuf), el, MPI_DOUBLE, idx_recipients[mpi_rank], 0,
                                   static_cast<void*>(rbuf), el, MPI_DOUBLE, idx_senders[mpi_rank], 0,
                                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                mgcl::mpi_util::mgclCheckMpiError(MPI_COMM_WORLD, err, "MPI_Sendrecv");
            });

            bench_util::ResultDataTransferMpi res;
            res.name = name;
            res.minTime = bench_util::getMinTime(bench, name);
            res.medianTime = bench_util::getMedianTime(bench, name);
            res.avgTime = bench_util::getAvgTime(bench, name);
            res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
            res.elements = el;
            res.gpus = mpi_size;
            results.push_back(res);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    bench_util::printCsvFormat(results, mpi_comm, mpi_rank);
}
