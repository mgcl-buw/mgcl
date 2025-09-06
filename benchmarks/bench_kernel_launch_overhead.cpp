#include "nanobench.h"

#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
using namespace std::chrono_literals;

#include "../src/mgcl/cuboid.hpp"
#include "../src/mgcl/multigrid_engine.hpp"
#include "../src/mgcl/opencl_helper.hpp"
#include "../src/mgcl/problem.hpp"
#include "../test/test_utility.hpp"
#include "bench_util.hpp"
#include "cli_args.hpp"

namespace mgcl_bench_kernel_launch_overhead
{
    bool PRINT_PARAMS = true;

    using EmptyKernelArgs = struct
    {
        std::array<int, 3> global;
        std::array<int, 3> local;
        std::vector<mgcl::BufferGpu>& kernelArgs;
        int argCount;

        cl_program program;
        cl_kernel kernel;
        cl_command_queue queue;
        cl_context context;

        mgcl::ProfilingData* pd;
    };

    using RegisterPressureKernelsArgs = struct
    {
        std::array<int, 3> global;
        std::array<int, 3> local;
        int registersPerThread;
        mgcl::BufferGpu& out;

        cl_program program;
        cl_kernel kernel;
        cl_command_queue queue;
        cl_context context;

        mgcl::ProfilingData* pd;
    };

    /**
     * @brief Launches a kernel that does nothing but has between 0 and 8 cl_mem arguments.
     *
     * @param args
     */
    void prepareAndLaunchEmptyKernel(const EmptyKernelArgs& args)
    {
        int err;

        if (args.argCount < 0 || args.argCount > 8)
            throw "args.argCount must be between 0 and 8";

        // Create the compute kernel from the program
        const std::string kernelName = std::string("empty_kernel_args0" + std::to_string(args.argCount));
        cl_event ev;

        // Create the compute kernel from the program
        cl_kernel kernel = clCreateKernel(args.program, kernelName.c_str(), &err);
        mgcl::mgclCheckError(err, std::string("Creating kernel " + kernelName));

        // assign kernel arguments
        int pos = 0;
        for (pos = 0; pos < args.argCount; pos++)
        {
            cl_mem buf = args.kernelArgs[pos].getBuf();
            err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &buf);
            mgcl::mgclCheckError(err, std::string("Setting kernel argument " + std::to_string(pos)));
        }

        // one work-item per cell (including ghost cells). Pad global sizes to fit to local sizes
        size_t global = args.global[0] * args.global[1] * args.global[2];
        // const auto& c = mgcl::conf::getWorkGroupSizeForKernelAndWiCount(problem.getKernelConfig(), kernelName, global);
        size_t local = args.local[0]; // c[0];

        if (global % local != 0)
            global += local - (global % local);

        err = clEnqueueNDRangeKernel(args.queue, kernel, 1, NULL, &global, &local, 0, NULL, &ev);
        mgcl::mgclCheckError(err, "Enqueueing kernel " + kernelName);

        if (args.pd)
        {
            args.pd->addMeasurement(args.queue, ev, kernelName,
                                    {global, 0, 0},
                                    {local, 1, 1});
        }
        mgcl::mgclCheckError(clReleaseEvent(ev), "clReleaseEvent");

        err = clReleaseKernel(kernel); // TODO maybe clFinish before release?
        mgcl::mgclCheckError(err, "clReleaseKernel " + kernelName);
    }

    /**
     * @brief Launches a kernel that does nothing but has between 0 and 8 cl_mem arguments.
     *
     * @param args
     */
    void prepareAndLaunchRegisterPressureKernel(const RegisterPressureKernelsArgs& args)
    {
        int err;

        if (args.registersPerThread % 16 != 0)
        {
            throw "args.registersPerThread must be a multiple of 16";
        }

        // Create the compute kernel from the program
        const std::string kernelName = std::string("reg" + std::to_string(args.registersPerThread));
        cl_event ev;

        // Create the compute kernel from the program
        cl_kernel kernel = clCreateKernel(args.program, kernelName.c_str(), &err);
        mgcl::mgclCheckError(err, std::string("Creating kernel " + kernelName));

        // assign kernel arguments
        int pos = 0;
        cl_mem buf = args.out.getBuf();
        err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &buf);
        mgcl::mgclCheckError(err, std::string("Setting kernel argument " + std::to_string(pos)));

        // one work-item per cell (including ghost cells). Pad global sizes to fit to local sizes
        size_t global = args.global[0] * args.global[1] * args.global[2];
        // const auto& c = mgcl::conf::getWorkGroupSizeForKernelAndWiCount(problem.getKernelConfig(), kernelName, global);
        size_t local = args.local[0]; // c[0];

        if (global % local != 0)
            global += local - (global % local);

        err = clEnqueueNDRangeKernel(args.queue, kernel, 1, NULL, &global, &local, 0, NULL, &ev);
        mgcl::mgclCheckError(err, "Enqueueing kernel " + kernelName);

        if (args.pd)
        {
            args.pd->addMeasurement(args.queue, ev, kernelName,
                                    {global, 0, 0},
                                    {local, 1, 1});
        }
        mgcl::mgclCheckError(clReleaseEvent(ev), "clReleaseEvent");

        err = clReleaseKernel(kernel); // TODO maybe clFinish before release?
        mgcl::mgclCheckError(err, "clReleaseKernel " + kernelName);
    }
}

// Launches empty kernel with 0 to 8 cl_mem arguments and prints profiling data.
// Size of ndrange can be controlled by --grids N. It is launched as 1d kernel with N*N*N work-items.
// Run with e.g.: mpiexec -n 1 benchmarks "bench_kernel_launch_overhead_mpi" --grids 8
TEST_CASE("bench_kernel_launch_overhead_mpi_register_pressure")
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

    if (mpi_rank == 0 && mgcl_bench_kernel_launch_overhead::PRINT_PARAMS)
    {
        mgcl_bench_kernel_launch_overhead::PRINT_PARAMS = false;

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

    std::stringstream ss;
    std::vector<bench_util::ResultMpi> results;
    bool printedGpu = false;
    for (auto gr : gridsTBT)
    {
        int m = gr[0];
        int n = gr[1];
        int o = gr[2];
        int mglob = m * mpi_dims[0];
        int nglob = n * mpi_dims[1];
        int oglob = o * mpi_dims[2];

        ankerl::nanobench::Bench bench;
        bench.timeUnit(1ms, "ms")
            .epochs(CLI_ARGS::bench_epochs)
            .epochIterations(CLI_ARGS::bench_iterations)
            // .minEpochTime(100ms)
            .relative(true);

        auto v = std::make_shared<mgcl::Cuboid>(1, 1, 1);
        auto f = std::make_shared<mgcl::Cuboid>(1, 1, 1);

        auto p = std::make_shared<mgcl::Problem>(m, n, o, f, v, mglob, nglob, oglob);
        p->setMpiComm(mpi_comm);
        p->setUseOpencl(true);
        p->setProfilingEnabled(CLI_ARGS::enableKernelProfiling);
        p->setDeviceType(CL_DEVICE_TYPE_GPU);
        p->setSilent(true);
        p->setKernelFile("kernel_launch_overhead.cl");
        p->init();

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

        mgcl::BufferGpu out(p->getContext(), CL_MEM_READ_WRITE, m * n * o);

        for (int regs = 16; regs <= 256; regs += 16)
        {
            std::string name = std::string("ocl_mpi_N")
                                   .append(std::to_string(m))
                                   .append("_")
                                   .append(std::to_string(n))
                                   .append("_")
                                   .append(std::to_string(o))
                                   .append("_regs")
                                   .append(std::to_string(regs));

            mgcl_bench_kernel_launch_overhead::RegisterPressureKernelsArgs args{
                {m, n, o},
                {64, 1, 1},
                regs,
                out,
                p->getOpenCLHelper().getProgram(),
                nullptr,
                p->getOpenCLHelper().getCommands(),
                p->getOpenCLHelper().getContext(),
                p->getProfilingData()};

            bench.run(std::string(name).c_str(), [&] { //
                MPI_Barrier(mpi_comm);
                mgcl_bench_kernel_launch_overhead::prepareAndLaunchRegisterPressureKernel(args);
                MPI_Barrier(mpi_comm);
            });

            // std::cout << "rank " << mpi_rank << " done" << std::endl;
            MPI_Barrier(mpi_comm);

            bench_util::ResultMpi res;
            res.name = name;
            res.minTime = bench_util::getMinTime(bench, name);
            res.medianTime = bench_util::getMedianTime(bench, name);
            res.avgTime = bench_util::getAvgTime(bench, name);
            res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
            res.m = m;
            res.n = n;
            res.o = o;
            res.mglob = mglob;
            res.nglob = nglob;
            res.oglob = oglob;
            res.gpus = mpi_size;
            res.LT = -1;
            results.push_back(res);
        }

        MPI_Barrier(mpi_comm);
        if (mpi_rank == 0 && CLI_ARGS::enableKernelProfiling)
        {
            p->getProfilingData()->printBestTimingsPerKernelAsCsv(ss);
            // p->getProfilingData()->printMeasurementsAsCsv(ss);
        }
        MPI_Barrier(mpi_comm);
    }

    bench_util::printCsvFormat(results, mpi_comm, mpi_rank);

    MPI_Barrier(mpi_comm);
    if (mpi_rank == 0)
    {
        std::cout << ss.str() << std::endl;
    }

    MPI_Barrier(mpi_comm);
}