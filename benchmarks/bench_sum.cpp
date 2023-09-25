#include "nanobench.h"

#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
using namespace std::chrono_literals;

#include "../src/cuboid.hpp"
#include "../src/problem.hpp"
#include "../src/util.hpp"
#include "../test/ocl_wrapper.hpp"
#include "../test/test_utility.hpp"
#include "bench_render_templates.hpp"
#include "pmg_utility.hpp"

#include "../thirdparty/mgcl_c/mgcl.hpp"
#include "../thirdparty/pmg/mg.h"

double sum(cl_mem buf, size_t num_elements, cl_context context, cl_program program, cl_command_queue commands,
           bool return_sum, size_t localSize, std::string kernelName, size_t globalSize, int fractions = 1);
double sum_finish_on_cpu(cl_mem buf, size_t num_elements, cl_context context, cl_program program, cl_command_queue commands,
                         bool return_sum, size_t localSize, std::string kernelName, size_t global, int fractions);

// Checks sum sequentially vs opencl
TEST_CASE("mgcl bench util::sum", "[!benchmark][sum][seqVsOcl]")
{
    std::vector grids{4, 8, 16, 32, 64, 128, 256, 512};
    // std::vector<size_t> locals{16, 32, 64};
    size_t local = 512;

    // for (auto local : locals)
    for (auto N : grids)
    {
        // int N = 16;
        int m = N;
        int n = N;
        int o = N;

        mgcl_test::TestUtility tu(CL_DEVICE_TYPE_GPU);

        mgcl::Cuboid data(m, n, o);
        data.fillRandom(-10, 10);

        auto dData = std::make_shared<mgcl::CuboidGpu>(tu.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, data);

        ankerl::nanobench::Bench b;
        b.timeUnit(1ns, "ns")
            // .epochs(1)
            // .epochIterations(1)
            .minEpochTime(100ms)
            .maxEpochTime(5s)
            .relative(true);

        b.run(std::string("seq, N: ").append(std::to_string(N)).append(", wg: ").append(std::to_string(local)).c_str(), [&]
              {
                  double sum = 0;
                  for (int i = 0; i < m; i++)
                      for (int j = 0; j < n; j++)
                          for (int k = 0; k < o; k++)
                          {
                              sum += data[i][j][k];
                          }
                  ankerl::nanobench::doNotOptimizeAway(sum);
                  //
              });

        b.run(std::string("ocl, N: ").append(std::to_string(N)).append(", wg: ").append(std::to_string(local)).c_str(), [&]
              {
                  ankerl::nanobench::doNotOptimizeAway(
                      mgcl::util::sum(*dData, tu.getProgram(), tu.getCommands(), true, local)); //
              });
    }
}

// Bench timings of different parts, i.e. actual kernel runtime, copying data, etc.
// NOTE: createAndReleaseBuffers somehow takes longer than everything else when no kernel is enqueued, so it
// was excluded from the benchmark.
TEST_CASE("mgcl bench util::sum", "[!benchmark][sum][parts]")
{
    std::vector grids{4, 8, 16, 32, 64, 128, 256, 512};
    // std::vector<size_t> locals{16, 32, 64};
    size_t local = 512;

    // for (auto local : locals)
    for (auto N : grids)
    {
        // int N = 16;
        int m = N;
        int n = N;
        int o = N;

        mgcl_test::TestUtility tu(CL_DEVICE_TYPE_GPU);
        OCLWrapper oclw(CL_DEVICE_TYPE_GPU,
                        mgcl_test::TestUtility::deviceAvailable("Quadro", CL_DEVICE_TYPE_GPU) ? "Quadro" : "",
                        "", "./kernel_optimizations.cl", tu.getContext());

        mgcl::Cuboid data(m, n, o);
        data.fillRandom(-10, 10);

        cl_mem buf = tu.createOpenCLBuffer(data);

        ankerl::nanobench::Bench b;
        b.timeUnit(1ns, "ns")
            // .epochs(1)
            // .epochIterations(1)
            .minEpochTime(100ms)
            .maxEpochTime(5s)
            .relative(true);

        // cl_mem buf;
        size_t num_elements = data.field1d().size();
        cl_context context = oclw.context;
        cl_program program = oclw.program;
        cl_command_queue commands = oclw.commands;
        bool return_sum = true;
        size_t localSize = local;
        std::string kernelName = "sum_partial_global_eq_x_num_elements";
        int fractions = 256;
        size_t global = ceil(static_cast<double>(num_elements) / fractions);

        int err;

        // One work-item per element in buf
        // size_t global = num_elements;

        // should not happen, but for e.g. global = o/2, o = 1
        if (global <= 0)
            global = 2;

        // make localSize even so reduction in kernel works properly
        if (localSize % 2 != 0)
            localSize++;

        // Pad global work-item count to fit wg-size
        if (global % localSize != 0)
            global += localSize - (global % localSize);

        // number of partial sums = num of work-groups
        int num_partials = global / localSize;

        // auto createAndReleaseBuffers = [&context, &commands, num_partials]()
        // {
        //     int err;

        //     cl_mem dPartialSums = clCreateBuffer(context, CL_MEM_READ_WRITE,
        //                                          sizeof(double) * num_partials, nullptr, &err);
        //     mgcl::mgclCheckError(err, "Creating dPartialSums buffer");

        //     // fill buffer with zeros
        //     double zero = 0;
        //     err = clEnqueueFillBuffer(commands, dPartialSums, &zero, sizeof(double), 0,
        //                               sizeof(double) * num_partials, 0, NULL, NULL);
        //     mgcl::mgclCheckError(err, "setting dPartialSums to 0");

        //     cl_mem dTotalSum = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double), nullptr, &err);
        //     mgcl::mgclCheckError(err, "Creating dTotalSum buffer");

        //     err = clReleaseMemObject(dPartialSums);
        //     mgcl::mgclCheckError(err, "clReleaseMemObject dPartialSums");

        //     err = clReleaseMemObject(dTotalSum);
        //     mgcl::mgclCheckError(err, "clReleaseMemObject dTotalSum");
        // };

        auto prepareAndEnqueueKernels = [&context, &commands, &program, &kernelName, &buf,
                                         localSize, fractions, global, num_elements, num_partials]()
        {
            int err;

            cl_mem dPartialSums = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                                 sizeof(double) * num_partials, nullptr, &err);
            mgcl::mgclCheckError(err, "Creating dPartialSums buffer");

            // fill buffer with zeros
            double zero = 0;
            err = clEnqueueFillBuffer(commands, dPartialSums, &zero, sizeof(double), 0,
                                      sizeof(double) * num_partials, 0, NULL, NULL);
            mgcl::mgclCheckError(err, "setting dPartialSums to 0");

            // Create the compute kernel from the program
            cl_kernel kernel_sum_partial = clCreateKernel(program, kernelName.c_str(), &err);
            mgcl::mgclCheckError(err, std::string("Creating kernel ").append(kernelName).c_str());

            int pos = 0;
            err = clSetKernelArg(kernel_sum_partial, pos, sizeof(cl_mem), &buf);
            err |= clSetKernelArg(kernel_sum_partial, ++pos, sizeof(cl_mem), &dPartialSums);
            err |= clSetKernelArg(kernel_sum_partial, ++pos, localSize * sizeof(double), nullptr);
            err |= clSetKernelArg(kernel_sum_partial, ++pos, sizeof(int), &num_elements);
            if (kernelName == "sum_partial_global_eq_x_num_elements")
                err |= clSetKernelArg(kernel_sum_partial, ++pos, sizeof(int), &fractions);
            mgcl::mgclCheckError(err, "Setting kernel sum_partial arguments");

            err = clEnqueueNDRangeKernel(commands, kernel_sum_partial, 1, NULL, &global, &localSize, 0, NULL, NULL);
            mgcl::mgclCheckError(err, "Enqueueing kernel sum_partial");

            // Create the compute kernel from the program
            cl_kernel kernel_sum_finish = clCreateKernel(program, "sum_finish", &err);
            mgcl::mgclCheckError(err, "Creating sum_finish kernel");

            cl_mem dTotalSum = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double), nullptr, &err);
            mgcl::mgclCheckError(err, "Creating dTotalSum buffer");

            pos = 0;
            err = clSetKernelArg(kernel_sum_finish, pos, sizeof(cl_mem), &dPartialSums);
            err |= clSetKernelArg(kernel_sum_finish, ++pos, sizeof(cl_mem), &dTotalSum);
            err |= clSetKernelArg(kernel_sum_finish, ++pos, sizeof(int), &num_partials);
            mgcl::mgclCheckError(err, "Setting sum_finish kernel arguments");

            size_t one = 1;
            err = clEnqueueNDRangeKernel(commands, kernel_sum_finish, 1, NULL, &one, &one, 0, NULL, NULL);
            mgcl::mgclCheckError(err, "Enqueueing kernel sum_finish");

            err = clReleaseMemObject(dPartialSums);
            mgcl::mgclCheckError(err, "clReleaseMemObject dPartialSums");

            err = clReleaseMemObject(dTotalSum);
            mgcl::mgclCheckError(err, "clReleaseMemObject dTotalSum");
        };

        auto runKernels = [&context, &commands, &program, &kernelName, &buf,
                           localSize, fractions, global, num_elements, num_partials]()
        {
            int err;

            cl_mem dPartialSums = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                                 sizeof(double) * num_partials, nullptr, &err);
            mgcl::mgclCheckError(err, "Creating dPartialSums buffer");

            // fill buffer with zeros
            double zero = 0;
            err = clEnqueueFillBuffer(commands, dPartialSums, &zero, sizeof(double), 0,
                                      sizeof(double) * num_partials, 0, NULL, NULL);
            mgcl::mgclCheckError(err, "setting dPartialSums to 0");

            // Create the compute kernel from the program
            cl_kernel kernel_sum_partial = clCreateKernel(program, kernelName.c_str(), &err);
            mgcl::mgclCheckError(err, std::string("Creating kernel ").append(kernelName).c_str());

            int pos = 0;
            err = clSetKernelArg(kernel_sum_partial, pos, sizeof(cl_mem), &buf);
            err |= clSetKernelArg(kernel_sum_partial, ++pos, sizeof(cl_mem), &dPartialSums);
            err |= clSetKernelArg(kernel_sum_partial, ++pos, localSize * sizeof(double), nullptr);
            err |= clSetKernelArg(kernel_sum_partial, ++pos, sizeof(int), &num_elements);
            if (kernelName == "sum_partial_global_eq_x_num_elements")
                err |= clSetKernelArg(kernel_sum_partial, ++pos, sizeof(int), &fractions);
            mgcl::mgclCheckError(err, "Setting kernel sum_partial arguments");

            err = clEnqueueNDRangeKernel(commands, kernel_sum_partial, 1, NULL, &global, &localSize, 0, NULL, NULL);
            mgcl::mgclCheckError(err, "Enqueueing kernel sum_partial");

            // Create the compute kernel from the program
            cl_kernel kernel_sum_finish = clCreateKernel(program, "sum_finish", &err);
            mgcl::mgclCheckError(err, "Creating sum_finish kernel");

            cl_mem dTotalSum = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double), nullptr, &err);
            mgcl::mgclCheckError(err, "Creating dTotalSum buffer");

            pos = 0;
            err = clSetKernelArg(kernel_sum_finish, pos, sizeof(cl_mem), &dPartialSums);
            err |= clSetKernelArg(kernel_sum_finish, ++pos, sizeof(cl_mem), &dTotalSum);
            err |= clSetKernelArg(kernel_sum_finish, ++pos, sizeof(int), &num_partials);
            mgcl::mgclCheckError(err, "Setting sum_finish kernel arguments");

            size_t one = 1;
            err = clEnqueueNDRangeKernel(commands, kernel_sum_finish, 1, NULL, &one, &one, 0, NULL, NULL);
            mgcl::mgclCheckError(err, "Enqueueing kernel sum_finish");

            err = clFinish(commands);
            mgcl::mgclCheckError(err, "Finishing sum kernels");

            err = clReleaseMemObject(dPartialSums);
            mgcl::mgclCheckError(err, "clReleaseMemObject dPartialSums");

            err = clReleaseMemObject(dTotalSum);
            mgcl::mgclCheckError(err, "clReleaseMemObject dTotalSum");
        };

        auto readResults = [&context, &commands, &program, &kernelName, &buf,
                            localSize, fractions, global, num_elements, num_partials]()
        {
            int err;

            cl_mem dPartialSums = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                                 sizeof(double) * num_partials, nullptr, &err);
            mgcl::mgclCheckError(err, "Creating dPartialSums buffer");

            // fill buffer with zeros
            double zero = 0;
            err = clEnqueueFillBuffer(commands, dPartialSums, &zero, sizeof(double), 0,
                                      sizeof(double) * num_partials, 0, NULL, NULL);
            mgcl::mgclCheckError(err, "setting dPartialSums to 0");

            // Create the compute kernel from the program
            cl_kernel kernel_sum_partial = clCreateKernel(program, kernelName.c_str(), &err);
            mgcl::mgclCheckError(err, std::string("Creating kernel ").append(kernelName).c_str());

            int pos = 0;
            err = clSetKernelArg(kernel_sum_partial, pos, sizeof(cl_mem), &buf);
            err |= clSetKernelArg(kernel_sum_partial, ++pos, sizeof(cl_mem), &dPartialSums);
            err |= clSetKernelArg(kernel_sum_partial, ++pos, localSize * sizeof(double), nullptr);
            err |= clSetKernelArg(kernel_sum_partial, ++pos, sizeof(int), &num_elements);
            if (kernelName == "sum_partial_global_eq_x_num_elements")
                err |= clSetKernelArg(kernel_sum_partial, ++pos, sizeof(int), &fractions);
            mgcl::mgclCheckError(err, "Setting kernel sum_partial arguments");

            err = clEnqueueNDRangeKernel(commands, kernel_sum_partial, 1, NULL, &global, &localSize, 0, NULL, NULL);
            mgcl::mgclCheckError(err, "Enqueueing kernel sum_partial");

            // Create the compute kernel from the program
            cl_kernel kernel_sum_finish = clCreateKernel(program, "sum_finish", &err);
            mgcl::mgclCheckError(err, "Creating sum_finish kernel");

            cl_mem dTotalSum = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double), nullptr, &err);
            mgcl::mgclCheckError(err, "Creating dTotalSum buffer");

            pos = 0;
            err = clSetKernelArg(kernel_sum_finish, pos, sizeof(cl_mem), &dPartialSums);
            err |= clSetKernelArg(kernel_sum_finish, ++pos, sizeof(cl_mem), &dTotalSum);
            err |= clSetKernelArg(kernel_sum_finish, ++pos, sizeof(int), &num_partials);
            mgcl::mgclCheckError(err, "Setting sum_finish kernel arguments");

            size_t one = 1;
            err = clEnqueueNDRangeKernel(commands, kernel_sum_finish, 1, NULL, &one, &one, 0, NULL, NULL);
            mgcl::mgclCheckError(err, "Enqueueing kernel sum_finish");

            err = clFinish(commands);
            mgcl::mgclCheckError(err, "Finishing sum kernels");

            double ret;
            err = clEnqueueReadBuffer(commands, dTotalSum, CL_TRUE, 0, sizeof(double),
                                      &ret, 0, NULL, NULL);
            mgcl::mgclCheckError(err, "Error: Failed to read dTotalSum from device!");

            err = clReleaseMemObject(dPartialSums);
            mgcl::mgclCheckError(err, "clReleaseMemObject dPartialSums");

            err = clReleaseMemObject(dTotalSum);
            mgcl::mgclCheckError(err, "clReleaseMemObject dTotalSum");
        };

        // b.run(std::string("createAndReleaseBuffers, N: ").append(std::to_string(N)).append(", wg: ").append(std::to_string(local)).c_str(), [&]
        //       {
        //           createAndReleaseBuffers(); //
        //       });

        // err = clFinish(commands);
        // mgcl::mgclCheckError(err, "Finishing sum kernels");

        b.run(std::string("prepareAndEnqueueKernels, N: ").append(std::to_string(N)).append(", wg: ").append(std::to_string(local)).c_str(), [&]
              {
                  prepareAndEnqueueKernels(); //
              });

        err = clFinish(commands);
        mgcl::mgclCheckError(err, "Finishing sum kernels");

        b.run(std::string("runKernels, N: ").append(std::to_string(N)).append(", wg: ").append(std::to_string(local)).c_str(), [&]
              {
                  runKernels(); //
              });

        err = clFinish(commands);
        mgcl::mgclCheckError(err, "Finishing sum kernels");

        b.run(std::string("readResults, N: ").append(std::to_string(N)).append(", wg: ").append(std::to_string(local)).c_str(), [&]
              {
                  readResults(); //
              });

        err = clFinish(commands);
        mgcl::mgclCheckError(err, "Finishing sum kernels");

        auto el = ankerl::nanobench::Result::Measure::elapsed;
        // std::cout << "aggregated results:" << std::scientific << std::endl
        //           << "  1   createAndReleaseBuffers: " << b.results()[0].minimum(el) << std::endl
        //           << "2-1  prepareAndEnqueueKernels: " << b.results()[1].minimum(el) - b.results()[0].minimum(el) << std::endl
        //           << "3-2                runKernels: " << b.results()[2].minimum(el) - b.results()[1].minimum(el) << std::endl
        //           << "4-3               readResults: " << b.results()[3].minimum(el) - b.results()[2].minimum(el) << std::endl;

        std::cout << "aggregated results (min. time):" << std::scientific << std::endl
                  << "  1 prepareAndEnqueueKernels: " << b.results()[0].minimum(el) << std::endl
                  << "2-1               runKernels: " << b.results()[1].minimum(el) - b.results()[0].minimum(el) << std::endl
                  << "3-2              readResults: " << b.results()[2].minimum(el) - b.results()[1].minimum(el) << std::endl;

        std::cout << "==============" << std::endl;
    }
}

// Checks different work-group sizes for different grid sizes
TEST_CASE("mgcl bench util::sum", "[!benchmark][sum][locals]")
{
    // Almost not effect for small grids (plus they are fast anyway).
    std::vector grids{32, 64, 128, 256, 512};
    std::vector<size_t> locals{16, 32, 64, 128, 192, 256, 384, 512, 768, 1024};

    for (auto N : grids)
    {
        // int N = 16;
        int m = N;
        int n = N;
        int o = N;

        mgcl_test::TestUtility tu(CL_DEVICE_TYPE_GPU);

        mgcl::Cuboid data(m, n, o);
        data.fillRandom(-10, 10);

        auto dData = std::make_shared<mgcl::CuboidGpu>(tu.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, data);

        ankerl::nanobench::Bench b;
        b.timeUnit(1ns, "ns")
            // .epochs(1)
            // .epochIterations(1)
            .minEpochTime(100ms)
            .maxEpochTime(5s)
            .relative(true);

        for (auto local : locals)
        {
            b.run(std::string("ocl, N: ").append(std::to_string(N)).append(", wg: ").append(std::to_string(local)).c_str(), [&]
                  {
                      ankerl::nanobench::doNotOptimizeAway(
                          mgcl::util::sum(*dData, tu.getProgram(), tu.getCommands(), true, local)); //
                  });
        }
        std::cout << "=============" << std::endl;
    }
}

// Checks different kernel versions
TEST_CASE("mgcl bench util::sum", "[!benchmark][sum][kernelVersions]")
{
    // Check if results are equal.
    SECTION("acceptance")
    {
        mgcl_test::TestUtility tu(CL_DEVICE_TYPE_GPU);
        OCLWrapper oclw(CL_DEVICE_TYPE_GPU,
                        mgcl_test::TestUtility::deviceAvailable("Quadro", CL_DEVICE_TYPE_GPU) ? "Quadro" : "",
                        "", "./kernel_optimizations.cl", tu.getContext());

        int m = 1;
        int n = 1;
        int o = GENERATE(1, 2, 3, 33, 64, 256);
        size_t local = GENERATE(32, 33);

        std::cout << "checking o: " << o << ", local: " << local << std::endl;

        mgcl::Cuboid data(m, n, o);
        // data.fillRandom(-10, 10);
        for (int i = 0; i < o; i++)
        {
            // data[0][0][i] = (i + 1) * (1.0 / 4.0);
            data[0][0][i] = (1.0 / 4.0);
        }

        cl_mem dData = tu.createOpenCLBuffer(data);

        std::vector<double> sums;

        sums.push_back(sum(dData, data.field1d().size(), oclw.context, oclw.program,
                           oclw.commands, true, local, "sum_partial_global_eq_num_elements", o));

        sums.push_back(sum(dData, data.field1d().size(), oclw.context, oclw.program,
                           oclw.commands, true, local, "sum_partial_global_eq_half_num_elements", ceil(o / 2.0)));

        sums.push_back(sum(dData, data.field1d().size(), oclw.context, oclw.program,
                           oclw.commands, true, local, "sum_partial_global_eq_quarter_num_elements", ceil(o / 4.0)));

        sums.push_back(sum(dData, data.field1d().size(), oclw.context, oclw.program,
                           oclw.commands, true, local, "sum_partial_global_eq_x_num_elements", ceil(o / 8.0), 8));

        sums.push_back(sum(dData, data.field1d().size(), oclw.context, oclw.program,
                           oclw.commands, true, local, "sum_partial_global_eq_x_num_elements", ceil(o / 16.0), 16));

        sums.push_back(sum(dData, data.field1d().size(), oclw.context, oclw.program,
                           oclw.commands, true, local, "sum_partial_global_eq_x_num_elements", ceil(o / 32.0), 32));

        sums.push_back(sum(dData, data.field1d().size(), oclw.context, oclw.program,
                           oclw.commands, true, local, "sum_partial_global_eq_x_num_elements", ceil(o / 64.0), 64));

        sums.push_back(sum(dData, data.field1d().size(), oclw.context, oclw.program,
                           oclw.commands, true, local, "sum_partial_global_eq_x_num_elements", ceil(o / 128.0), 128));

        sums.push_back(sum(dData, data.field1d().size(), oclw.context, oclw.program,
                           oclw.commands, true, local, "sum_partial_global_eq_x_num_elements", ceil(o / 256.0), 256));

        sums.push_back(sum_finish_on_cpu(dData, data.field1d().size(), oclw.context, oclw.program,
                                         oclw.commands, true, local, "sum_partial_global_eq_x_num_elements",
                                         ceil(o / 256.0), 256));

        // for (auto s : sums)
        //     std::cout << "  sum: " << s << std::endl;

        for (auto s : sums)
            REQUIRE_THAT(sums[0], Catch::Matchers::WithinAbs(s, 1e-14));
    }

    // Run the actual benchmark.
    SECTION("benchmark")
    {
        // Almost not effect for small grids (plus they are fast anyway).
        std::vector grids{32, 64, 128, 256, 512};
        // std::vector<size_t> locals{16, 32, 64, 128, 192, 256, 384, 512, 768, 1024};
        std::vector<size_t> locals{512};

        for (auto N : grids)
        {
            // int N = 16;
            int m = N;
            int n = N;
            int o = N;

            mgcl_test::TestUtility tu(CL_DEVICE_TYPE_GPU);
            OCLWrapper oclw(CL_DEVICE_TYPE_GPU,
                            mgcl_test::TestUtility::deviceAvailable("Quadro", CL_DEVICE_TYPE_GPU) ? "Quadro" : "",
                            "", "./kernel_optimizations.cl", tu.getContext());

            mgcl::Cuboid data(m, n, o);
            data.fillRandom(-10, 10);

            cl_mem dData = tu.createOpenCLBuffer(data);

            ankerl::nanobench::Bench b;
            b.timeUnit(1ns, "ns")
                // .epochs(1)
                // .epochIterations(1)
                .minEpochTime(100ms)
                .maxEpochTime(5s)
                .relative(true);

            int num_elements = data.field1d().size();

            for (auto local : locals)
            {
                b.run(std::string("sum_partial_global_eq_num_elements, N: ").append(std::to_string(N)).append(", wg: ").append(std::to_string(local)).c_str(), [&]
                      {
                          ankerl::nanobench::doNotOptimizeAway(
                              sum(dData, num_elements, oclw.context, oclw.program,
                                  oclw.commands, true, local, "sum_partial_global_eq_num_elements", num_elements)); //
                      });

                b.run(std::string("sum_partial_global_eq_half_num_elements, N: ").append(std::to_string(N)).append(", wg: ").append(std::to_string(local)).c_str(), [&]
                      {
                          ankerl::nanobench::doNotOptimizeAway(
                              sum(dData, num_elements, oclw.context, oclw.program,
                                  oclw.commands, true, local, "sum_partial_global_eq_half_num_elements", ceil(num_elements / 2.0))); //
                      });

                b.run(std::string("sum_partial_global_eq_quarter_num_elements, N: ").append(std::to_string(N)).append(", wg: ").append(std::to_string(local)).c_str(), [&]
                      {
                          ankerl::nanobench::doNotOptimizeAway(
                              sum(dData, num_elements, oclw.context, oclw.program,
                                  oclw.commands, true, local, "sum_partial_global_eq_quarter_num_elements", ceil(num_elements / 4.0))); //
                      });

                b.run(std::string("sum_partial_global_eq_1/8_num_elements, N: ").append(std::to_string(N)).append(", wg: ").append(std::to_string(local)).c_str(), [&]
                      {
                          ankerl::nanobench::doNotOptimizeAway(
                              sum(dData, num_elements, oclw.context, oclw.program,
                                  oclw.commands, true, local, "sum_partial_global_eq_x_num_elements", ceil(num_elements / 8.0), 8)); //
                      });

                b.run(std::string("sum_partial_global_eq_1/16_num_elements, N: ").append(std::to_string(N)).append(", wg: ").append(std::to_string(local)).c_str(), [&]
                      {
                          ankerl::nanobench::doNotOptimizeAway(
                              sum(dData, num_elements, oclw.context, oclw.program,
                                  oclw.commands, true, local, "sum_partial_global_eq_x_num_elements", ceil(num_elements / 16.0), 16)); //
                      });

                b.run(std::string("sum_partial_global_eq_1/32_num_elements, N: ").append(std::to_string(N)).append(", wg: ").append(std::to_string(local)).c_str(), [&]
                      {
                          ankerl::nanobench::doNotOptimizeAway(
                              sum(dData, num_elements, oclw.context, oclw.program,
                                  oclw.commands, true, local, "sum_partial_global_eq_x_num_elements", ceil(num_elements / 32.0), 32)); //
                      });

                b.run(std::string("sum_partial_global_eq_1/64_num_elements, N: ").append(std::to_string(N)).append(", wg: ").append(std::to_string(local)).c_str(), [&]
                      {
                          ankerl::nanobench::doNotOptimizeAway(
                              sum(dData, num_elements, oclw.context, oclw.program,
                                  oclw.commands, true, local, "sum_partial_global_eq_x_num_elements", ceil(num_elements / 64.0), 64)); //
                      });

                b.run(std::string("sum_partial_global_eq_1/128_num_elements, N: ").append(std::to_string(N)).append(", wg: ").append(std::to_string(local)).c_str(), [&]
                      {
                          ankerl::nanobench::doNotOptimizeAway(
                              sum(dData, num_elements, oclw.context, oclw.program,
                                  oclw.commands, true, local, "sum_partial_global_eq_x_num_elements", ceil(num_elements / 128.0), 128)); //
                      });

                b.run(std::string("sum_partial_global_eq_1/256_num_elements, N: ").append(std::to_string(N)).append(", wg: ").append(std::to_string(local)).c_str(), [&]
                      {
                          ankerl::nanobench::doNotOptimizeAway(
                              sum(dData, num_elements, oclw.context, oclw.program,
                                  oclw.commands, true, local, "sum_partial_global_eq_x_num_elements", ceil(num_elements / 256.0), 256)); //
                      });

                b.run(std::string("sum_partial_global_eq_1/512_num_elements, N: ").append(std::to_string(N)).append(", wg: ").append(std::to_string(local)).c_str(), [&]
                      {
                          ankerl::nanobench::doNotOptimizeAway(
                              sum(dData, num_elements, oclw.context, oclw.program,
                                  oclw.commands, true, local, "sum_partial_global_eq_x_num_elements", ceil(num_elements / 512.0), 512)); //
                      });

                b.run(std::string("sum_partial_global_eq_1/512_num_elements + sum_finish on cpu, N: ").append(std::to_string(N)).append(", wg: ").append(std::to_string(local)).c_str(), [&]
                      {
                          ankerl::nanobench::doNotOptimizeAway(
                              sum_finish_on_cpu(dData, num_elements, oclw.context, oclw.program,
                                                oclw.commands, true, local, "sum_partial_global_eq_x_num_elements",
                                                ceil(num_elements / 512.0), 512)); //
                      });
            }
            std::cout << "=============" << std::endl;
        }
    }
}

// Copied and slightly modified for different inputs from mgcl::util::sum
double sum(cl_mem buf, size_t num_elements, cl_context context, cl_program program, cl_command_queue commands,
           bool return_sum, size_t localSize, std::string kernelName, size_t global, int fractions)
{
    int err;

    // One work-item per element in buf
    // size_t global = num_elements;

    // should not happen, but for e.g. global = o/2, o = 1
    if (global <= 0)
        global = 2;

    // make localSize even so reduction in kernel works properly
    if (localSize % 2 != 0)
        localSize++;

    // Pad global work-item count to fit wg-size
    if (global % localSize != 0)
        global += localSize - (global % localSize);

    // number of partial sums = num of work-groups
    int num_partials = global / localSize;

    // int pointer_flag = problem.getOpenCLHelper().getDeviceType() == CL_DEVICE_TYPE_GPU ? CL_MEM_COPY_HOST_PTR : CL_MEM_USE_HOST_PTR;
    cl_mem dPartialSums = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                         sizeof(double) * num_partials, nullptr, &err);
    mgcl::mgclCheckError(err, "Creating dPartialSums buffer");

    // fill buffer with zeros
    double zero = 0;
    err = clEnqueueFillBuffer(commands, dPartialSums, &zero, sizeof(double), 0,
                              sizeof(double) * num_partials, 0, NULL, NULL);
    mgcl::mgclCheckError(err, "setting dPartialSums to 0");

    // Create the compute kernel from the program
    cl_kernel kernel_sum_partial = clCreateKernel(program, kernelName.c_str(), &err);
    mgcl::mgclCheckError(err, std::string("Creating kernel ").append(kernelName).c_str());

    int pos = 0;
    err = clSetKernelArg(kernel_sum_partial, pos, sizeof(cl_mem), &buf);
    err |= clSetKernelArg(kernel_sum_partial, ++pos, sizeof(cl_mem), &dPartialSums);
    err |= clSetKernelArg(kernel_sum_partial, ++pos, localSize * sizeof(double), nullptr);
    err |= clSetKernelArg(kernel_sum_partial, ++pos, sizeof(int), &num_elements);
    if (kernelName == "sum_partial_global_eq_x_num_elements")
        err |= clSetKernelArg(kernel_sum_partial, ++pos, sizeof(int), &fractions);
    mgcl::mgclCheckError(err, "Setting kernel sum_partial arguments");

    err = clEnqueueNDRangeKernel(commands, kernel_sum_partial, 1, NULL, &global, &localSize, 0, NULL, NULL);
    mgcl::mgclCheckError(err, "Enqueueing kernel sum_partial");

    // Create the compute kernel from the program
    cl_kernel kernel_sum_finish = clCreateKernel(program, "sum_finish", &err);
    mgcl::mgclCheckError(err, "Creating sum_finish kernel");

    cl_mem dTotalSum = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double), nullptr, &err);
    mgcl::mgclCheckError(err, "Creating dTotalSum buffer");

    pos = 0;
    err = clSetKernelArg(kernel_sum_finish, pos, sizeof(cl_mem), &dPartialSums);
    err |= clSetKernelArg(kernel_sum_finish, ++pos, sizeof(cl_mem), &dTotalSum);
    err |= clSetKernelArg(kernel_sum_finish, ++pos, sizeof(int), &num_partials);
    mgcl::mgclCheckError(err, "Setting sum_finish kernel arguments");

    size_t one = 1;
    err = clEnqueueNDRangeKernel(commands, kernel_sum_finish, 1, NULL, &one, &one, 0, NULL, NULL);
    mgcl::mgclCheckError(err, "Enqueueing kernel sum_finish");

    double ret = 0;
    if (return_sum)
    {
        clFinish(commands);

        err = clEnqueueReadBuffer(commands, dTotalSum, CL_TRUE, 0, sizeof(double),
                                  &ret, 0, NULL, NULL);
        mgcl::mgclCheckError(err, "Error: Failed to read dTotalSum from device!");
    }

    err = clReleaseMemObject(dPartialSums);
    mgcl::mgclCheckError(err, "clReleaseMemObject dPartialSums");

    err = clReleaseMemObject(dTotalSum);
    mgcl::mgclCheckError(err, "clReleaseMemObject dTotalSum");

    return ret;
}

// Copied and slightly modified for different inputs from mgcl::util::sum.
// Sums up partial sums sequentially on cpu.
double sum_finish_on_cpu(cl_mem buf, size_t num_elements, cl_context context, cl_program program, cl_command_queue commands,
                         bool return_sum, size_t localSize, std::string kernelName, size_t global, int fractions)
{
    int err;

    // One work-item per element in buf
    // size_t global = num_elements;

    // should not happen, but for e.g. global = o/2, o = 1
    if (global <= 0)
        global = 2;

    // make localSize even so reduction in kernel works properly
    if (localSize % 2 != 0)
        localSize++;

    // Pad global work-item count to fit wg-size
    if (global % localSize != 0)
        global += localSize - (global % localSize);

    // number of partial sums = num of work-groups
    int num_partials = global / localSize;

    // int pointer_flag = problem.getOpenCLHelper().getDeviceType() == CL_DEVICE_TYPE_GPU ? CL_MEM_COPY_HOST_PTR : CL_MEM_USE_HOST_PTR;
    cl_mem dPartialSums = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                         sizeof(double) * num_partials, nullptr, &err);
    mgcl::mgclCheckError(err, "Creating dPartialSums buffer");

    // fill buffer with zeros
    double zero = 0;
    err = clEnqueueFillBuffer(commands, dPartialSums, &zero, sizeof(double), 0,
                              sizeof(double) * num_partials, 0, NULL, NULL);
    mgcl::mgclCheckError(err, "setting dPartialSums to 0");

    // Create the compute kernel from the program
    cl_kernel kernel_sum_partial = clCreateKernel(program, kernelName.c_str(), &err);
    mgcl::mgclCheckError(err, std::string("Creating kernel ").append(kernelName).c_str());

    int pos = 0;
    err = clSetKernelArg(kernel_sum_partial, pos, sizeof(cl_mem), &buf);
    err |= clSetKernelArg(kernel_sum_partial, ++pos, sizeof(cl_mem), &dPartialSums);
    err |= clSetKernelArg(kernel_sum_partial, ++pos, localSize * sizeof(double), nullptr);
    err |= clSetKernelArg(kernel_sum_partial, ++pos, sizeof(int), &num_elements);
    if (kernelName == "sum_partial_global_eq_x_num_elements")
        err |= clSetKernelArg(kernel_sum_partial, ++pos, sizeof(int), &fractions);
    mgcl::mgclCheckError(err, "Setting kernel sum_partial arguments");

    err = clEnqueueNDRangeKernel(commands, kernel_sum_partial, 1, NULL, &global, &localSize, 0, NULL, NULL);
    mgcl::mgclCheckError(err, "Enqueueing kernel sum_partial");

    double ret = 0;
    if (return_sum)
    {
        clFinish(commands);
        mgcl::mgclCheckError(err, "clFinish");

        double tmp[num_partials];
        err = clEnqueueReadBuffer(commands, dPartialSums, CL_TRUE, 0, sizeof(double), tmp, 0, NULL, NULL);
        mgcl::mgclCheckError(err, "Error: Failed to read dPartialSums from device!");

        for (int i = 0; i < num_partials; i++)
            ret += tmp[i];
    }

    err = clReleaseMemObject(dPartialSums);
    mgcl::mgclCheckError(err, "clReleaseMemObject dPartialSums");

    return ret;
}
