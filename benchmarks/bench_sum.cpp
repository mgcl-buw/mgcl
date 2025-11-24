#include "bench_util.hpp"
#include "cli_args.hpp"
#include "nanobench.h"

#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include <catch2/catch_message.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>
using namespace std::chrono_literals;

#include "../src/mgcl/cuboid.hpp"
#include "../src/mgcl/problem.hpp"
#include "../src/mgcl/util.hpp"
#include "../test/ocl_wrapper.hpp"
#include "../test/test_utility.hpp"
#include "bench_render_templates.hpp"

double sum(cl_mem buf, size_t num_elements, cl_context context, cl_program program, cl_command_queue commands,
           bool return_sum, size_t localSize, std::string kernelName, size_t globalSize, int batchSize,
           mgcl::ProfilingData* pd, int cuCount, int maxWgSize);
double sum_finish_on_cpu(cl_mem buf, size_t num_elements, cl_context context, cl_program program, cl_command_queue commands,
                         bool return_sum, size_t localSize, std::string kernelName, size_t global, int batchSize,
                         mgcl::ProfilingData* pd, int cuCount, int maxWgSize, int maxKernelCalls, int& out_numKernelCalls, int& out_elementsOnCpu);
double sum_finish_use_same_kernel(cl_mem buf, size_t num_elements, cl_context context, cl_program program, cl_command_queue commands,
                                  bool return_sum, size_t localSize, std::string kernelName, size_t global, int batchSize,
                                  mgcl::ProfilingData* pd, int cuCount, int maxWgSizebool, bool isUnrolled, int& out_numKernelCalls);

int nextPowerOfTwo(int x)
{
    if (x <= 1)
        return 1;

    // If x is already a power of two, return it.
    // (x & (x - 1)) == 0 means power of two.
    if ((x & (x - 1)) == 0)
        return x;

    // Spread the highest set bit rightwards
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;

    return x + 1;
}

int padGlobal(size_t global, size_t local)
{
    return global + (local - (global % local));
}

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

        size_t maxWgSize;
        int err = clGetDeviceInfo(tu.getDeviceId(), CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWgSize, nullptr);
        mgcl::mgclCheckError(err, "clGetDeviceInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE)");

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
                      mgcl::util::sum(*dData, tu.getProgram(), tu.getCommands(), true, maxWgSize, nullptr, nullptr)); //
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
        int batchSize = 256;
        size_t global = ceil(static_cast<double>(num_elements) / batchSize);

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
                                         localSize, batchSize, global, num_elements, num_partials]()
        {
            int err;

            cl_mem dPartialSums = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                                 sizeof(double) * num_partials, nullptr, &err);
            mgcl::mgclCheckError(err, "Creating dPartialSums buffer");

            // fill buffer with zeros
            mgcl::util::fill(program, commands, dPartialSums, 0, num_partials, false, nullptr, nullptr);

            // Create the compute kernel from the program
            cl_kernel kernel_sum_partial = clCreateKernel(program, kernelName.c_str(), &err);
            mgcl::mgclCheckError(err, std::string("Creating kernel ").append(kernelName).c_str());

            int pos = 0;
            err = clSetKernelArg(kernel_sum_partial, pos, sizeof(cl_mem), &buf);
            err |= clSetKernelArg(kernel_sum_partial, ++pos, sizeof(cl_mem), &dPartialSums);
            err |= clSetKernelArg(kernel_sum_partial, ++pos, localSize * sizeof(double), nullptr);
            err |= clSetKernelArg(kernel_sum_partial, ++pos, sizeof(int), &num_elements);
            if (kernelName == "sum_partial_global_eq_x_num_elements")
                err |= clSetKernelArg(kernel_sum_partial, ++pos, sizeof(int), &batchSize);
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
                           localSize, batchSize, global, num_elements, num_partials]()
        {
            int err;

            cl_mem dPartialSums = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                                 sizeof(double) * num_partials, nullptr, &err);
            mgcl::mgclCheckError(err, "Creating dPartialSums buffer");

            // fill buffer with zeros
            mgcl::util::fill(program, commands, dPartialSums, 0, num_partials, false, nullptr, nullptr);

            // Create the compute kernel from the program
            cl_kernel kernel_sum_partial = clCreateKernel(program, kernelName.c_str(), &err);
            mgcl::mgclCheckError(err, std::string("Creating kernel ").append(kernelName).c_str());

            int pos = 0;
            err = clSetKernelArg(kernel_sum_partial, pos, sizeof(cl_mem), &buf);
            err |= clSetKernelArg(kernel_sum_partial, ++pos, sizeof(cl_mem), &dPartialSums);
            err |= clSetKernelArg(kernel_sum_partial, ++pos, localSize * sizeof(double), nullptr);
            err |= clSetKernelArg(kernel_sum_partial, ++pos, sizeof(int), &num_elements);
            if (kernelName == "sum_partial_global_eq_x_num_elements")
                err |= clSetKernelArg(kernel_sum_partial, ++pos, sizeof(int), &batchSize);
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
                            localSize, batchSize, global, num_elements, num_partials]()
        {
            int err;

            cl_mem dPartialSums = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                                 sizeof(double) * num_partials, nullptr, &err);
            mgcl::mgclCheckError(err, "Creating dPartialSums buffer");

            // fill buffer with zeros
            mgcl::util::fill(program, commands, dPartialSums, 0, num_partials, false, nullptr, nullptr);

            // Create the compute kernel from the program
            cl_kernel kernel_sum_partial = clCreateKernel(program, kernelName.c_str(), &err);
            mgcl::mgclCheckError(err, std::string("Creating kernel ").append(kernelName).c_str());

            int pos = 0;
            err = clSetKernelArg(kernel_sum_partial, pos, sizeof(cl_mem), &buf);
            err |= clSetKernelArg(kernel_sum_partial, ++pos, sizeof(cl_mem), &dPartialSums);
            err |= clSetKernelArg(kernel_sum_partial, ++pos, localSize * sizeof(double), nullptr);
            err |= clSetKernelArg(kernel_sum_partial, ++pos, sizeof(int), &num_elements);
            if (kernelName == "sum_partial_global_eq_x_num_elements")
                err |= clSetKernelArg(kernel_sum_partial, ++pos, sizeof(int), &batchSize);
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

        size_t maxWgSize;
        int err = clGetDeviceInfo(tu.getDeviceId(), CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWgSize, nullptr);
        mgcl::mgclCheckError(err, "clGetDeviceInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE)");

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
                          mgcl::util::sum(*dData, tu.getProgram(), tu.getCommands(), true, maxWgSize, nullptr, nullptr)); //
                  });
        }
        std::cout << "=============" << std::endl;
    }
}

// Checks different kernel versions
TEST_CASE("benchSumReductionVersions")
{

    using std::min;

    // Check if results are equal.
    if (CLI_ARGS::checkResults)
    {
        // dummy problem
        auto vdummy = std::make_shared<mgcl::Cuboid>(1, 1, 1);
        auto fdummy = std::make_shared<mgcl::Cuboid>(1, 1, 1);
        mgcl::Problem p(1, 1, 1, fdummy, vdummy);
        p.setUseOpencl(true);
        p.setDeviceType(CL_DEVICE_TYPE_GPU);
        p.setSilent(true);
        p.setKernelFile("kernel_optimizations.cl");
        p.setProfilingEnabled(CLI_ARGS::enableKernelProfiling);
        p.getOpenCLHelper().setPreprocessorConstant("SUM_WG_SIZE", std::to_string(128));
        p.init();

        int cuCount = p.getOpenCLHelper().queryComputeUnitCount(p.getOpenCLHelper().getDeviceId());
        int maxWgSize = p.getOpenCLHelper().queryMaxWgSize(p.getOpenCLHelper().getDeviceId());

        int m = 1;
        int n = 1;

        std::vector<int> o_sizes = {1, 2, 3, 33, 256, 257, 2048};
        std::vector<int> locals = {4, 32, 128, 512};
        // std::vector<int> o_sizes = {256};
        // std::vector<int> locals = {4};
        std::vector<int> batchSizes = {4, 5, 32, 128};
        std::vector<std::string> testNames;
        int minElementsForCpu = 1024;
        for (int local : locals)
        {
            p.getOpenCLHelper().setPreprocessorConstant("SUM_WG_SIZE", std::to_string(local));
            p.getOpenCLHelper().rebuildProgram(true);
            for (int o : o_sizes)
            {
                std::cout << "checking o: " << o << ", local: " << local << std::endl;

                double checksum = 0;

                mgcl::Cuboid data(m, n, o);
                // data.fillRandom(-10, 10);
                for (int i = 0; i < o; i++)
                {
                    // data[0][0][i] = (i + 1) * (1.0 / 4.0);
                    data[0][0][i] = (1.0 / 4.0);
                    checksum += data[0][0][i];
                }

                mgcl::CuboidGpu dDataGpu(p.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, data);
                cl_mem dData = dDataGpu.getBuffer();

                auto context = p.getContext();
                auto program = p.getProgram();
                auto commands = p.getCommands();

                std::vector<double> sums;
                int cntKernelCalls;
                int elementsOnCpu;

                testNames.push_back("sum_partial_global_eq_num_elements");
                sums.push_back(sum(dData, data.field1d().size(), context, program,
                                   commands, true, local, "sum_partial_global_eq_num_elements", o, 1, nullptr, cuCount, maxWgSize));

                for (auto batchSize : batchSizes)
                {
                    testNames.push_back("sum_partial_global_eq_x_num_elements, finish kernel, batchSize: " + std::to_string(batchSize));
                    CAPTURE(batchSize);
                    sums.push_back(sum(dData, data.field1d().size(), context, program,
                                       commands, true, local, "sum_partial_global_eq_x_num_elements", ceil(o / static_cast<double>(batchSize)), batchSize, nullptr, cuCount, maxWgSize));
                }

                for (auto batchSize : batchSizes)
                {
                    testNames.push_back("sum_partial_global_eq_x_num_elements, finish on cpu, batchSize: " + std::to_string(batchSize));
                    CAPTURE(batchSize);
                    sums.push_back(sum_finish_on_cpu(dData, data.field1d().size(), context, program,
                                                     commands, true, local, "sum_partial_global_eq_x_num_elements_same_kernel_finish", ceil(o / static_cast<double>(batchSize)), batchSize, nullptr, cuCount, maxWgSize, minElementsForCpu, cntKernelCalls, elementsOnCpu));
                }

                for (auto batchSize : batchSizes)
                {
                    testNames.push_back("sum_partial_global_eq_x_num_elements, same kernel, batchSize: " + std::to_string(batchSize));
                    CAPTURE(batchSize);
                    sums.push_back(sum_finish_use_same_kernel(dData, data.field1d().size(), context, program,
                                                              commands, true, local, "sum_partial_global_eq_x_num_elements_same_kernel_finish", ceil(o / static_cast<double>(batchSize)), batchSize, nullptr, cuCount, maxWgSize, false, cntKernelCalls));
                }

                // for (auto batchSize : batchSizes)
                int batchSize = 4;
                {
                    testNames.push_back("sum_partial_global_eq_x_num_elements, same kernel unrolled, batchSize: " + std::to_string(batchSize));
                    CAPTURE(batchSize);
                    sums.push_back(sum_finish_use_same_kernel(dData, data.field1d().size(), context, program,
                                                              commands, true, local, "sum_partial_global_eq_x_num_elements_same_kernel_finish_unrolled", ceil(o / static_cast<double>(batchSize)), batchSize, nullptr, cuCount, maxWgSize, true, cntKernelCalls));
                }

                // sums.push_back(sum_finish_use_same_kernel(dData, data.field1d().size(), context, program,
                //                                           commands, true, local, "",
                //                                           ceil(o / 32.0), 32, nullptr, cuCount, maxWgSize));
                // sums.push_back(sum_finish_use_same_kernel_unrolled(dData, data.field1d().size(), context, program,
                //                                                    commands, true, local, "",
                //                                                    ceil(o / 32.0), 32, nullptr, cuCount, maxWgSize));

                // for (auto s : sums)
                //     std::cout << "  sum: " << s << std::endl;

                int cnt = 0;
                for (auto s : sums)
                {
                    CAPTURE(cnt, o, local, testNames[cnt]);
                    cnt++;
                    REQUIRE_THAT(checksum, Catch::Matchers::WithinAbs(s, 1e-14));
                }
            }
        }
    }
    else
    {

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

        std::vector<bench_util::ResultSumReduction> results;

        std::stringstream kernelProfilesStream;

        // Almost not effect for small grids (plus they are fast anyway).
        // std::vector grids{32, 64, 128, 256, 512};
        // std::vector<size_t> locals{16, 32, 64, 128, 192, 256, 384, 512, 768, 1024};
        // std::vector<size_t> locals{32, 64, 128, 256, 512};
        std::vector<size_t> locals{256};
        // std::vector<int> maxKernelCalls = {1, 2, 3}; // max kernel calls before sum is finished on CPU
        std::vector<int> maxKernelCalls = {2}; // max kernel calls before sum is finished on CPU

        for (auto grid : gridsTBT)
        {
            // int N = 16;
            int m = grid[0];
            int n = grid[1];
            int o = grid[2];
            int N = m * n * o;

            int brentP = N / std::log2(N);      // Number of work-items we should use according to Brent
            int targetBatchSize = std::log2(N); // Target batch size according to Brent

            double batchRatioLower = 0.2; // ratio to test around brentP
            double batchRatioUpper = 1.0; // ratio to test around brentP
            int targetBatchSizeLower = ceil(targetBatchSize - targetBatchSize * batchRatioLower);
            int targetBatchSizeUpper = floor(targetBatchSize + targetBatchSize * batchRatioUpper);
            int batchStepsize = 4;
            // std::vector<int> batchSizes = {1, 2, 3, 4, 5, 1 << 7, 1 << 9, 1 << 12, 1 << 13, 1 << 14, 1 << 15, 1 << 16, 1 << 17, 1 << 18};
            std::vector<int> batchSizes = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
            std::cout << "> brentP: " << brentP << std::endl;
            std::cout << "> batchRatioLower: " << batchRatioLower << std::endl;
            std::cout << "> batchRatioUpper: " << batchRatioUpper << std::endl;
            std::cout << "> batchStepsize: " << batchStepsize << std::endl;
            std::cout << "> batchSizes: [" << targetBatchSizeLower << "," << targetBatchSizeUpper << "]" << std::endl;

            // for (int r = targetBatchSizeLower; r < targetBatchSizeUpper; r += batchStepsize)
            // {
            //     batchSizes.push_back(r);
            // }

            // dummy problem
            auto vdummy = std::make_shared<mgcl::Cuboid>(1, 1, 1);
            auto fdummy = std::make_shared<mgcl::Cuboid>(1, 1, 1);
            mgcl::Problem p(1, 1, 1, fdummy, vdummy);
            p.setUseOpencl(true);
            p.setDeviceType(CL_DEVICE_TYPE_GPU);
            p.setSilent(true);
            p.setKernelFile("kernel_optimizations.cl");
            p.setProfilingEnabled(CLI_ARGS::enableKernelProfiling);
            p.getOpenCLHelper().setPreprocessorConstant("SUM_WG_SIZE", std::to_string(512));
            // TODO preprocessor values
            p.init();

            int cuCount = p.getOpenCLHelper().queryComputeUnitCount(p.getOpenCLHelper().getDeviceId());
            int maxWgSize = p.getOpenCLHelper().queryMaxWgSize(p.getOpenCLHelper().getDeviceId());
            std::cout << "cuCount: " << cuCount << std::endl;
            std::cout << "maxWgSize: " << maxWgSize << std::endl;

            mgcl::Cuboid data(m, n, o);
            data.fillRandom(-10, 10);

            mgcl::CuboidGpu dDataGpu(p.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, data);
            cl_mem dData = dDataGpu.getBuffer();

            ankerl::nanobench::Bench b;
            b.timeUnit(1ns, "ns")
                .epochs(CLI_ARGS::bench_epochs)
                .epochIterations(CLI_ARGS::bench_iterations)
                // .minEpochTime(100ms)
                // .maxEpochTime(5s)
                .relative(false);

            int num_elements = data.field1d().size();

            auto pd = p.getProfilingData();

            for (auto local : locals)
            {
                p.getOpenCLHelper().setPreprocessorConstant("SUM_WG_SIZE", std::to_string(local));
                p.getOpenCLHelper().rebuildProgram(true);
                auto context = p.getContext();
                auto program = p.getProgram();
                auto commands = p.getCommands();

                // {
                //     std::string name = std::string("sum_partial_global_eq_x_num_elements_N")
                //                            .append(std::to_string(N))
                //                            .append("_wg")
                //                            .append(std::to_string(local));
                //     b.run(name.c_str(), [&]
                //           {
                //               ankerl::nanobench::doNotOptimizeAway(
                //                   sum(dData, num_elements, context, program,
                //                       commands, true, local, "sum_partial_global_eq_num_elements", num_elements, 1, pd, cuCount, maxWgSize)); //
                //           });

                //     bench_util::ResultSumReduction res;
                //     res.name = name;
                //     res.minTime = bench_util::getMinTime(b, name);
                //     res.medianTime = bench_util::getMedianTime(b, name);
                //     res.avgTime = bench_util::getAvgTime(b, name);
                //     res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(b, name);
                //     res.N = N;
                //     res.batchSize = 1;
                //     res.wgx = local;
                //     res.ndrange = padGlobal(num_elements, local);
                //     res.kernelCalls = 1;
                //     results.push_back(res);
                // }

                // std::vector<int> batchSize_finishOnGpu{8, 16, 32, 64, 128, 256, 512, 1024};
                // for (int fr : batchSize_finishOnGpu)
                // for (int bs : batchSizes)
                // {
                //     int cntKernelCalls = -1;
                //     int glob = ceil(num_elements / static_cast<double>(bs));
                //     std::string name = std::string("sum_partial_global_eq_1/")
                //                            .append(std::to_string(bs))
                //                            .append("_N")
                //                            .append(std::to_string(N))
                //                            .append("_wg")
                //                            .append(std::to_string(local))
                //                            .append("_finishOnGPU");
                //     b.run(name.c_str(), [&]
                //           {
                //               ankerl::nanobench::doNotOptimizeAway(
                //                   sum(dData, num_elements, context, program,
                //                       commands, true, local, "sum_partial_global_eq_x_num_elements",
                //                       glob, bs, pd, cuCount, maxWgSize)); //
                //           });

                //     bench_util::ResultSumReduction res;
                //     res.name = name;
                //     res.minTime = bench_util::getMinTime(b, name);
                //     res.medianTime = bench_util::getMedianTime(b, name);
                //     res.avgTime = bench_util::getAvgTime(b, name);
                //     res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(b, name);
                //     res.N = N;
                //     res.batchSize = bs;
                //     res.wgx = local;
                //     res.ndrange = padGlobal(glob, local);
                //     res.kernelCalls = cntKernelCalls;
                //     results.push_back(res);
                // }

                // for (int fr : batchSize_finishOnGpu)
                for (int bs : batchSizes)
                {
                    int cntKernelCalls = -1;
                    int glob = ceil(num_elements / static_cast<double>(bs));
                    std::string name = std::string("sum_partial_global_eq_1/")
                                           .append(std::to_string(bs))
                                           .append("_N")
                                           .append(std::to_string(N))
                                           .append("_wg")
                                           .append(std::to_string(local))
                                           .append("_finishOnGPU_sameKernel");
                    b.run(name.c_str(), [&]
                          {
                              ankerl::nanobench::doNotOptimizeAway(
                                  sum_finish_use_same_kernel(dData, num_elements, context, program,
                                                             commands, true, local, "sum_partial_global_eq_x_num_elements_same_kernel_finish",
                                                             glob, bs, pd, cuCount, maxWgSize, false, cntKernelCalls)); //
                          });

                    bench_util::ResultSumReduction res;
                    res.name = "gpuOnly";
                    res.minTime = bench_util::getMinTime(b, name);
                    res.medianTime = bench_util::getMedianTime(b, name);
                    res.avgTime = bench_util::getAvgTime(b, name);
                    res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(b, name);
                    res.N = N;
                    res.batchSize = bs;
                    res.wgx = local;
                    res.ndrange = padGlobal(glob, local);
                    res.kernelCalls = cntKernelCalls;
                    res.elementsOnCpu = 0;
                    results.push_back(res);
                }

                // std::vector<int> batchSize_finishOnGpu_unrolled{8, 16, 32, 64, 128, 256, 512, 1024};
                // for (int fr : batchSize_finishOnGpu_unrolled)
                for (int bs : batchSizes)
                {
                    int cntKernelCalls = -1;
                    int glob = ceil(num_elements / static_cast<double>(bs));
                    std::string name = std::string("sum_partial_global_eq_1/")
                                           .append(std::to_string(bs))
                                           .append("_N")
                                           .append(std::to_string(N))
                                           .append("_wg")
                                           .append(std::to_string(local))
                                           .append("_finishOnGPU_sameKernel_unrolled");
                    b.run(name.c_str(), [&]
                          {
                              ankerl::nanobench::doNotOptimizeAway(
                                  sum_finish_use_same_kernel(dData, num_elements, context, program,
                                                             commands, true, local, "sum_partial_global_eq_x_num_elements_same_kernel_finish_unrolled",
                                                             glob, bs, pd, cuCount, maxWgSize, true, cntKernelCalls)); //
                          });

                    bench_util::ResultSumReduction res;
                    res.name = "gpuOnlyUnrolled";
                    res.minTime = bench_util::getMinTime(b, name);
                    res.medianTime = bench_util::getMedianTime(b, name);
                    res.avgTime = bench_util::getAvgTime(b, name);
                    res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(b, name);
                    res.N = N;
                    res.batchSize = bs;
                    res.wgx = local;
                    res.ndrange = padGlobal(glob, local);
                    res.kernelCalls = cntKernelCalls;
                    res.elementsOnCpu = 0;
                    results.push_back(res);
                }

                // std::vector<int> batchSize_finishOnCpu{128, 256, 512};
                // for (int fr : batchSize_finishOnCpu)
                for (int bs : batchSizes)
                    for (int me : maxKernelCalls)
                    {
                        // if log_wgsize(ndrange) < maxKernelCalls, skip
                        int maxlv = ceil(log2(ceil(num_elements / static_cast<double>(bs))) / log2(local));
                        if (maxlv < me)
                            continue;

                        int cntKernelCalls = -1;
                        int elementsOnCpu;
                        int glob = ceil(num_elements / static_cast<double>(bs));
                        std::string name = std::string("sum_partial_global_eq_1/")
                                               .append(std::to_string(bs))
                                               .append("_N")
                                               .append(std::to_string(N))
                                               .append("_wg")
                                               .append(std::to_string(local))
                                               .append("_finishOnCPU_levelsOnGpu")
                                               .append(std::to_string(me))
                                               .append("_maxlv")
                                               .append(std::to_string(maxlv));
                        b.run(name.c_str(), [&]
                              {
                                  ankerl::nanobench::doNotOptimizeAway(
                                      sum_finish_on_cpu(dData, num_elements, context, program,
                                                        commands, true, local, "sum_partial_global_eq_x_num_elements_same_kernel_finish",
                                                        glob, bs, pd, cuCount, maxWgSize, me, cntKernelCalls, elementsOnCpu)); //
                              });

                        bench_util::ResultSumReduction res;
                        res.name = "finishOnCpu";
                        res.minTime = bench_util::getMinTime(b, name);
                        res.medianTime = bench_util::getMedianTime(b, name);
                        res.avgTime = bench_util::getAvgTime(b, name);
                        res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(b, name);
                        res.N = N;
                        res.batchSize = bs;
                        res.wgx = local;
                        res.ndrange = padGlobal(glob, local);
                        res.kernelCalls = cntKernelCalls;
                        res.elementsOnCpu = elementsOnCpu;
                        results.push_back(res);
                    }
            }
            std::cout << "=============" << std::endl;

            if (CLI_ARGS::enableKernelProfiling)
            {
                // p.getProfilingData()->printBestTimingsPerKernel(kernelProfilesStream);
                p.getProfilingData()->printBestTimingsPerKernelAsCsv(kernelProfilesStream);
            }
        } // end for all N

        // MPI_Barrier(mpi_comm);
        bench_util::printCsvFormat(results);
        // MPI_Barrier(mpi_comm);

        if (CLI_ARGS::enableKernelProfiling)
        {
            // kernelProfilesStream << "rank: " << mpi_rank << std::endl;
            std::cout << kernelProfilesStream.str() << std::endl;
        }
        // MPI_Barrier(mpi_comm);
    }
}

// Copied and slightly modified for different inputs from mgcl::util::sum
double sum(cl_mem buf, size_t num_elements, cl_context context, cl_program program, cl_command_queue commands,
           bool return_sum, size_t localSize, std::string kernelName, size_t global, int batchSize,
           mgcl::ProfilingData* pd, int cuCount, int maxWgSize)
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
    mgcl::util::fill(program, commands, dPartialSums, 0, num_partials, false, nullptr, nullptr);

    // Create the compute kernel from the program
    cl_kernel kernel_sum_partial = clCreateKernel(program, kernelName.c_str(), &err);
    mgcl::mgclCheckError(err, std::string("Creating kernel ").append(kernelName).c_str());

    int pos = 0;
    err = clSetKernelArg(kernel_sum_partial, pos, sizeof(cl_mem), &buf);
    err |= clSetKernelArg(kernel_sum_partial, ++pos, sizeof(cl_mem), &dPartialSums);
    err |= clSetKernelArg(kernel_sum_partial, ++pos, localSize * sizeof(double), nullptr);
    err |= clSetKernelArg(kernel_sum_partial, ++pos, sizeof(int), &num_elements);
    if (kernelName == "sum_partial_global_eq_x_num_elements")
        err |= clSetKernelArg(kernel_sum_partial, ++pos, sizeof(int), &batchSize);
    mgcl::mgclCheckError(err, "Setting kernel sum_partial arguments");

    cl_event ev;
    err = clEnqueueNDRangeKernel(commands, kernel_sum_partial, 1, NULL, &global, &localSize, 0, NULL, &ev);
    mgcl::mgclCheckError(err, "Enqueueing kernel sum_partial");

    if (pd != nullptr)
    {
        pd->addMeasurement(commands, ev, kernelName,
                           {global, 0, 0},
                           {localSize, 1, 1});
    }
    mgcl::mgclCheckError(clReleaseEvent(ev), "clReleaseEvent");

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

    cl_event ev2;
    size_t one = 1;
    err = clEnqueueNDRangeKernel(commands, kernel_sum_finish, 1, NULL, &one, &one, 0, NULL, &ev2);
    mgcl::mgclCheckError(err, "Enqueueing kernel sum_finish");

    if (pd != nullptr)
    {
        pd->addMeasurement(commands, ev2, "sum_finish",
                           {one, 0, 0},
                           {one, 1, 1});
    }
    mgcl::mgclCheckError(clReleaseEvent(ev2), "clReleaseEvent");

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

    mgcl::mgclCheckError(clReleaseKernel(kernel_sum_partial), "clReleaseKernel(kernel_sum_partial)");
    mgcl::mgclCheckError(clReleaseKernel(kernel_sum_finish), "clReleaseKernel(kernel_sum_finish)");

    return ret;
}

double sum_finish_use_same_kernel(cl_mem buf, size_t num_elements, cl_context context, cl_program program, cl_command_queue commands,
                                  bool return_sum, size_t localSize, std::string kernelName, size_t global, int batchSize,
                                  mgcl::ProfilingData* pd, int cuCount, int maxWgSize, bool isUnrolled, int& out_numKernelCalls)
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

    // kernelName = "sum_partial_global_eq_x_num_elements_same_kernel_finish_unrolled";

    // number of partial sums = num of work-groups
    int num_partials = ceil(global / static_cast<double>(localSize));

    // if (num_partials > 512)
    // {
    //     throw "num_partials must be less than max block size of 512 for sum_partial_global_eq_x_num_elements_same_kernel_finish. num_partials: " +
    //         std::to_string(num_partials) + ", localSize: " + std::to_string(localSize) + ", global: " + std::to_string(global);
    // }

    // int pointer_flag = problem.getOpenCLHelper().getDeviceType() == CL_DEVICE_TYPE_GPU ? CL_MEM_COPY_HOST_PTR : CL_MEM_USE_HOST_PTR;
    cl_mem dPartialSums = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                         sizeof(double) * num_partials, nullptr, &err);
    mgcl::mgclCheckError(err, "Creating dPartialSums buffer");

    // fill buffer with zeros
    mgcl::util::fill(program, commands, dPartialSums, 0, num_partials, false, nullptr, nullptr);

    // Create the compute kernel from the program
    cl_kernel kernel_sum_partial = clCreateKernel(program, kernelName.c_str(), &err);
    mgcl::mgclCheckError(err, std::string("Creating kernel ").append(kernelName).c_str());

    // repeatedly run kernel until last partial sums can be reduced in a single work-group
    int cntKernelCalls = 0;
    cl_mem src = buf;
    do // while num_partials > 1
    {
        // std::cout << "cntKernelCalls: " << cntKernelCalls << std::endl;
        // std::cout << "  localSize: " << localSize << std::endl;
        // std::cout << "  batchSize: " << batchSize << std::endl;
        // std::cout << "  num_partials: " << num_partials << std::endl;
        // std::cout << "  global: " << global << std::endl;
        // std::cout << "  num_elements: " << num_elements << std::endl;

        int pos = 0;
        err = clSetKernelArg(kernel_sum_partial, pos, sizeof(cl_mem), &src);
        err |= clSetKernelArg(kernel_sum_partial, ++pos, sizeof(cl_mem), &dPartialSums);
        err |= clSetKernelArg(kernel_sum_partial, ++pos, localSize * sizeof(double), nullptr);
        err |= clSetKernelArg(kernel_sum_partial, ++pos, sizeof(int), &num_elements);
        err |= clSetKernelArg(kernel_sum_partial, ++pos, sizeof(int), &batchSize);
        mgcl::mgclCheckError(err, "Setting kernel sum_partial arguments");

        cl_event ev;
        err = clEnqueueNDRangeKernel(commands, kernel_sum_partial, 1, NULL, &global, &localSize, 0, NULL, &ev);
        mgcl::mgclCheckError(err, "Enqueueing kernel sum_partial");

        if (pd != nullptr)
        {
            pd->addMeasurement(commands, ev, kernelName,
                               {global, 0, 0},
                               {localSize, 1, 1});
        }
        mgcl::mgclCheckError(clReleaseEvent(ev), "clReleaseEvent");

        // launch as many wi's in next as there were results in the last kernel call.
        global = num_partials;
        num_elements = num_partials;
        src = dPartialSums;

        // if num_partials < maximum wg size, set localSize to maximum wg size, s.t. we only need one final kernel call.
        // num_partials is input size of next kernel call.
        if (global < maxWgSize && !isUnrolled)
        {
            int tmp = nextPowerOfTwo(global);
            localSize = tmp > maxWgSize ? maxWgSize : tmp;
        }

        // Pad work-item count to a multiple of wg size.
        if (global % localSize != 0)
            global += localSize - (global % localSize);

        // number of partial sums = num of work-groups
        num_partials = ceil(global / static_cast<double>(localSize));

        // if (num_partials > 1 && num_partials % 2 != 0)
        // {
        //     throw "num_partials must be even for sum_partial_global_eq_x_num_elements_same_kernel_finish_unrolled. num_partials: " +
        //         std::to_string(num_partials) + ", localSize: " + std::to_string(localSize) + ", cntKernelCalls: " + std::to_string(cntKernelCalls);
        // }

        // recalculate batchSize according to Brent's theorem (N/log2(N)), but use at least #CU work-groups.
        // batchSize = static_cast<int>(ceil(static_cast<double>(num_partials) / log2(static_cast<double>(num_partials))));
        // int maxWis = localSize * cuCount;
        // if (num_elements > maxWis)
        // {
        //     batchSize = batchSize > maxWis ? maxWis : batchSize;
        // }
        batchSize = 1;
        cntKernelCalls++;
    } while (num_elements > 1);

    out_numKernelCalls = cntKernelCalls;

    double ret = 0;
    if (return_sum)
    {
        err = clFinish(commands);
        mgcl::mgclCheckError(err, "Error: clFinish failed!");

        err = clEnqueueReadBuffer(commands, dPartialSums, CL_TRUE, 0, sizeof(double),
                                  &ret, 0, NULL, NULL);
        mgcl::mgclCheckError(err, "Error: Failed to read dTotalSum from device!");
    }

    err = clReleaseMemObject(dPartialSums);
    mgcl::mgclCheckError(err, "clReleaseMemObject dPartialSums");

    mgcl::mgclCheckError(clReleaseKernel(kernel_sum_partial), "clReleaseKernel(kernel_sum_partial)");

    return ret;
}

double sum_finish_on_cpu(cl_mem buf, size_t num_elements, cl_context context, cl_program program, cl_command_queue commands,
                         bool return_sum, size_t localSize, std::string kernelName, size_t global, int batchSize,
                         mgcl::ProfilingData* pd, int cuCount, int maxWgSize, int maxKernelCalls, int& out_numKernelCalls, int& out_elementsOnCpu)
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

    // kernelName = "sum_partial_global_eq_x_num_elements_same_kernel_finish";

    // number of partial sums = num of work-groups
    int num_partials = ceil(global / static_cast<double>(localSize));

    // if (num_partials > 512)
    // {
    //     throw "num_partials must be less than max block size of 512 for sum_partial_global_eq_x_num_elements_same_kernel_finish. num_partials: " +
    //         std::to_string(num_partials) + ", localSize: " + std::to_string(localSize) + ", global: " + std::to_string(global);
    // }

    // int pointer_flag = problem.getOpenCLHelper().getDeviceType() == CL_DEVICE_TYPE_GPU ? CL_MEM_COPY_HOST_PTR : CL_MEM_USE_HOST_PTR;
    cl_mem dPartialSums = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                         sizeof(double) * num_partials, nullptr, &err);
    mgcl::mgclCheckError(err, "Creating dPartialSums buffer");

    // fill buffer with zeros
    mgcl::util::fill(program, commands, dPartialSums, 0, num_partials, false, nullptr, nullptr);

    // Create the compute kernel from the program
    cl_kernel kernel_sum_partial = clCreateKernel(program, kernelName.c_str(), &err);
    mgcl::mgclCheckError(err, std::string("Creating kernel ").append(kernelName).c_str());

    // repeatedly run kernel until last partial sums can be reduced in a single work-group
    int cntKernelCalls = 0;
    cl_mem src = buf;
    do // while num_partials > 1
    {
        // std::cout << "cntKernelCalls: " << cntKernelCalls << std::endl;
        // std::cout << "  localSize: " << localSize << std::endl;
        // std::cout << "  batchSize: " << batchSize << std::endl;
        // std::cout << "  num_partials: " << num_partials << std::endl;
        // std::cout << "  global: " << global << std::endl;
        // std::cout << "  num_elements: " << num_elements << std::endl;

        int pos = 0;
        err = clSetKernelArg(kernel_sum_partial, pos, sizeof(cl_mem), &src);
        err |= clSetKernelArg(kernel_sum_partial, ++pos, sizeof(cl_mem), &dPartialSums);
        err |= clSetKernelArg(kernel_sum_partial, ++pos, localSize * sizeof(double), nullptr);
        err |= clSetKernelArg(kernel_sum_partial, ++pos, sizeof(int), &num_elements);
        err |= clSetKernelArg(kernel_sum_partial, ++pos, sizeof(int), &batchSize);
        mgcl::mgclCheckError(err, "Setting kernel sum_partial arguments");

        cl_event ev;
        err = clEnqueueNDRangeKernel(commands, kernel_sum_partial, 1, NULL, &global, &localSize, 0, NULL, &ev);
        mgcl::mgclCheckError(err, "Enqueueing kernel sum_partial");

        if (pd != nullptr)
        {
            pd->addMeasurement(commands, ev, kernelName,
                               {global, 0, 0},
                               {localSize, 1, 1});
        }
        mgcl::mgclCheckError(clReleaseEvent(ev), "clReleaseEvent");

        // launch as many wi's in next as there were results in the last kernel call.
        global = num_partials;
        num_elements = num_partials;
        src = dPartialSums;

        // if num_partials < maximum wg size, set localSize to maximum wg size, s.t. we only need one final kernel call.
        // num_partials is input size of next kernel call.
        if (global < maxWgSize)
        {
            int tmp = nextPowerOfTwo(global);
            localSize = tmp > maxWgSize ? maxWgSize : tmp;
        }

        // Pad work-item count to a multiple of wg size.
        if (global % localSize != 0)
            global += localSize - (global % localSize);

        // number of partial sums = num of work-groups
        num_partials = ceil(global / static_cast<double>(localSize));

        // if (num_partials > 1 && num_partials % 2 != 0)
        // {
        //     throw "num_partials must be even for sum_partial_global_eq_x_num_elements_same_kernel_finish_unrolled. num_partials: " +
        //         std::to_string(num_partials) + ", localSize: " + std::to_string(localSize) + ", cntKernelCalls: " + std::to_string(cntKernelCalls);
        // }

        // recalculate batchSize according to Brent's theorem (N/log2(N)), but use at least #CU work-groups.
        // batchSize = static_cast<int>(ceil(static_cast<double>(num_partials) / log2(static_cast<double>(num_partials))));
        // int maxWis = localSize * cuCount;
        // if (num_elements > maxWis)
        // {
        //     batchSize = batchSize > maxWis ? maxWis : batchSize;
        // }
        batchSize = 1;
        cntKernelCalls++;
    } while (num_elements > 1 && cntKernelCalls < maxKernelCalls);

    out_numKernelCalls = cntKernelCalls;
    out_elementsOnCpu = num_elements;

    double ret = 0;
    if (return_sum)
    {
        err = clFinish(commands);
        mgcl::mgclCheckError(err, "Error: clFinish failed!");

        auto tmp = std::make_unique<double[]>(num_elements);
        err = clEnqueueReadBuffer(commands, dPartialSums, CL_TRUE, 0, num_elements * sizeof(double), tmp.get(), 0, NULL, NULL);
        mgcl::mgclCheckError(err, "Error: Failed to read dTotalSum from device!");

        for (int i = 0; i < num_elements; i++)
            ret += tmp.get()[i];
    }

    err = clReleaseMemObject(dPartialSums);
    mgcl::mgclCheckError(err, "clReleaseMemObject dPartialSums");

    mgcl::mgclCheckError(clReleaseKernel(kernel_sum_partial), "clReleaseKernel(kernel_sum_partial)");

    return ret;
}