/**
 * @file bench_various_ndim_kernels.cpp
 * @brief Compares different kernel versions for various parts of mgcl, e.g. update ghosts.
 * @date 2024-03-04
 *
 */
#include "nanobench.h"

#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"

#include "../src/mgcl/opencl_helper.hpp"
#include "../src/mgcl/problem.hpp"
#include "../test/ocl_wrapper.hpp"
#include "../test/test_utility.hpp"
#include "cli_args.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
using namespace std::chrono_literals;

/**
 * @brief Compares different kernel version for ghost update of mgcl::Cuboid. Only for ghosts = 1 atm.
 *
 */
TEST_CASE("benchUpdateGhostsKernelVersions")
{
    // set to true if results of different kernel version shall be checked.
    bool checkResults = true;

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

    std::cout << "Testing the following grid sizes" << std::endl;
    for (auto gr : gridsTBT)
    {
        int m = gr[0];
        int n = gr[1];
        int o = gr[2];
        std::cout << "  " << m << "," << n << "," << o << std::endl;
    }

    for (auto Ns : gridsTBT)
    {
        int m = Ns[0];
        int n = Ns[1];
        int o = Ns[2];
        int ghosts_m = 1;
        int ghosts_n = 1;
        int ghosts_o = 1;

        ankerl::nanobench::Bench bench;
        bench.timeUnit(1ms, "ms")
            // .epochs(1)
            // .epochIterations(1)
            .minEpochTime(100ms)
            .relative(false);

        // do only one iteration if results shall be checked
        if (checkResults)
        {
            bench.epochs(1).epochIterations(1);
        }

        // if (N >= 32)
        //     bench.epochs(1).epochIterations(2);

        // if (N >= 64)
        //     bench.epochs(1).epochIterations(1);

        // OCLWrapper oclw(CL_DEVICE_TYPE_GPU,
        //                 mgcl_test::TestUtility::deviceAvailable("Quadro", CL_DEVICE_TYPE_GPU) ? "Quadro" : "",
        //                 "",
        //                 "./kernel_optimizations.cl");

        auto v = std::make_shared<mgcl::Cuboid>(1, 1, 1);
        auto f = std::make_shared<mgcl::Cuboid>(1, 1, 1);
        mgcl::Problem p(1, 1, 1, f, v);
        p.setUseOpencl(true);
        p.setKernelFile("./kernel_optimizations.cl");
        p.getOpenCLHelper().setReadKernelFromFile(true);
        p.initOpenCL();
        // p.init();

        cl_context context = p.getContext();
        cl_program program = p.getProgram();
        cl_command_queue commands = p.getCommands();

        // Host buffer defining the same input for all tests
        mgcl::Cuboid hbuf(m, n, o, ghosts_m, ghosts_n, ghosts_o);
        hbuf.fillRandomInt(-100, 100, true);

        std::vector<std::unique_ptr<mgcl::Cuboid>> results;

        // 3d kernel (as of 2024-03-04)
        {
            mgcl::CuboidGpu dBuffer(context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, hbuf);

            // TODO actually request these as arguments
            int mgh = dBuffer.getMgh();
            int ngh = dBuffer.getNgh();
            int ogh = dBuffer.getOgh();

            int err;

            // Create the compute kernel from the program
            const char* kernelName = "update_ghosts_periodic_3d";
            cl_kernel kernel = clCreateKernel(program, kernelName, &err);
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
            size_t global[3] = {static_cast<size_t>(mgh), static_cast<size_t>(ngh), static_cast<size_t>(ogh)};
            const size_t local[3] = {
                static_cast<size_t>(mgh > 4 ? 4 : mgh),
                static_cast<size_t>(ngh > 4 ? 4 : ngh),
                static_cast<size_t>(ogh > 4 ? 4 : ogh)};

            for (int i = 0; i < 3; i++)
                if (global[i] % local[i] != 0)
                    global[i] += local[i] - (global[i] % local[i]);

            // cl_event ev;
            // enqueue kernel
            // err = clEnqueueNDRangeKernel(oclw.commands, kernel, 3, NULL, global, local, 0, NULL, &ev);
            // mgcl::mgclCheckError(err, "Enqueueing update_ghosts_periodic kernel");

            // if (problem.isProfilingEnabled())
            // {
            //     problem.getProfilingData()->addMeasurement(problem.getCommands(), ev, kernelName,
            //                                                {global[0], global[1], global[2]},
            //                                                {local[0], local[1], local[2]});
            // }
            // mgcl::mgclCheckError(clReleaseEvent(ev), "clReleaseEvent");

            std::string name;
            name = std::string(kernelName).append("_").append(std::to_string(m)).append("x").append(std::to_string(n)).append("x").append(std::to_string(o));

            bench.run(name, [&]
                      {
                          clEnqueueNDRangeKernel(commands, kernel, 3, NULL, global, local, 0, NULL, NULL);
                          clFinish(commands); //
                      });

            err = clReleaseKernel(kernel);
            mgcl::mgclCheckError(err, "Releasing update_ghosts_periodic kernel");

            results.push_back(dBuffer.read(commands, nullptr, true));
        }

        // 1d kernel (as of 2024-03-04)
        {
            mgcl::CuboidGpu dBuffer(context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, hbuf);

            // TODO actually request these as arguments
            int mgh = dBuffer.getMgh();
            int ngh = dBuffer.getNgh();
            int ogh = dBuffer.getOgh();

            int err;

            // Create the compute kernel from the program
            const char* kernelName = "update_ghosts_periodic_1d";
            cl_kernel kernel = clCreateKernel(program, kernelName, &err);
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
            size_t global = mgh * ngh * ogh;
            const size_t local = 32;

            if (global % local != 0)
                global += local - (global % local);

            // cl_event ev;
            // enqueue kernel
            // err = clEnqueueNDRangeKernel(oclw.commands, kernel, 3, NULL, global, local, 0, NULL, &ev);
            // mgcl::mgclCheckError(err, "Enqueueing update_ghosts_periodic kernel");

            // if (problem.isProfilingEnabled())
            // {
            //     problem.getProfilingData()->addMeasurement(problem.getCommands(), ev, kernelName,
            //                                                {global[0], global[1], global[2]},
            //                                                {local[0], local[1], local[2]});
            // }
            // mgcl::mgclCheckError(clReleaseEvent(ev), "clReleaseEvent");

            std::string name;
            name = std::string(kernelName).append("_").append(std::to_string(m)).append("x").append(std::to_string(n)).append("x").append(std::to_string(o));

            bench.run(name, [&]
                      {
                          clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
                          clFinish(commands); //
                      });

            err = clReleaseKernel(kernel);
            mgcl::mgclCheckError(err, "Releasing update_ghosts_periodic kernel");

            results.push_back(dBuffer.read(commands, nullptr, true));
        }

        // 1d kernel, executed for only ghost cells
        {
            mgcl::CuboidGpu dBuffer(context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, hbuf);

            // TODO actually request these as arguments
            int mgh = dBuffer.getMgh();
            int ngh = dBuffer.getNgh();
            int ogh = dBuffer.getOgh();

            int err;

            // Create the compute kernel from the program
            const char* kernelName = "update_ghosts_periodic_1d_ghosts_cells_only";
            cl_kernel kernel = clCreateKernel(program, kernelName, &err);
            mgcl::mgclCheckError(err, "clCreateKernel");

            // assign kernel arguments
            int pos = 0;
            err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &dBuffer);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &m);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &n);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &o);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &mgh);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ngh);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ogh);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghosts_m);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghosts_n);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghosts_o);
            mgcl::mgclCheckError(err, "Setting kernel arguments");

            // one work-item per ghost cell (excluding real cells). Pad global sizes to fit to local sizes
            size_t global = mgh * ngh * ogh - m * n * o;
            const size_t local = 32;

            if (global % local != 0)
                global += local - (global % local);

            // cl_event ev;
            // enqueue kernel
            // err = clEnqueueNDRangeKernel(oclw.commands, kernel, 3, NULL, global, local, 0, NULL, &ev);
            // mgcl::mgclCheckError(err, "Enqueueing update_ghosts_periodic kernel");

            // if (problem.isProfilingEnabled())
            // {
            //     problem.getProfilingData()->addMeasurement(problem.getCommands(), ev, kernelName,
            //                                                {global[0], global[1], global[2]},
            //                                                {local[0], local[1], local[2]});
            // }
            // mgcl::mgclCheckError(clReleaseEvent(ev), "clReleaseEvent");

            std::string name;
            name = std::string(kernelName).append("_").append(std::to_string(m)).append("x").append(std::to_string(n)).append("x").append(std::to_string(o));

            bench.run(name, [&]
                      {
                          clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
                          clFinish(commands); //
                      });

            err = clReleaseKernel(kernel);
            mgcl::mgclCheckError(err, "Releasing update_ghosts_periodic kernel");

            results.push_back(dBuffer.read(commands, nullptr, true));
        }

        if (checkResults)
        {
            // assumes that first result is correct
            for (int i = 1; i < results.size(); i++)
            {
                REQUIRE(results[0]->isEqualAllCells(*results[i]));
            }
            std::cout << "All results ok." << std::endl;
        }
    }
}