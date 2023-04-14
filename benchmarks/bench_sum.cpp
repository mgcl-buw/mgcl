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
           bool return_sum, size_t localSize, std::string kernelName, size_t globalSize);

// Checks sum sequentially vs opencl
TEST_CASE("mgcl bench util::sum", "[!benchmark][sum][seqVsOcl]")
{
    std::vector grids{4, 8, 16, 32, 64, 128, 256, 512};
    // std::vector<size_t> locals{16, 32, 64};
    size_t local = 32;

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

        cl_mem dData = tu.createOpenCLBuffer(data);

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
                      mgcl::util::sum(dData, data.field1d().size(), tu.getContext(), tu.getProgram(),
                                      tu.getCommands(), true, local)); //
              });
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

        cl_mem dData = tu.createOpenCLBuffer(data);

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
                          mgcl::util::sum(dData, data.field1d().size(), tu.getContext(), tu.getProgram(),
                                          tu.getCommands(), true, local)); //
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

        double s1 = sum(dData, data.field1d().size(), oclw.context, oclw.program,
                        oclw.commands, true, local, "sum_partial_global_eq_num_elements", o);

        double s2 = sum(dData, data.field1d().size(), oclw.context, oclw.program,
                        oclw.commands, true, local, "sum_partial_global_eq_half_num_elements", ceil(o / 2.0));

        double s3 = sum(dData, data.field1d().size(), oclw.context, oclw.program,
                        oclw.commands, true, local, "sum_partial_global_eq_quarter_num_elements", ceil(o / 4.0));

        std::cout << "  s1: " << s1 << std::endl
                  << "  s2: " << s2 << std::endl
                  << "  s3: " << s3 << std::endl;
        REQUIRE_THAT(s1, Catch::Matchers::WithinAbs(s2, 1e-14));
        REQUIRE_THAT(s1, Catch::Matchers::WithinAbs(s3, 1e-14));
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
            }
            std::cout << "=============" << std::endl;
        }
    }
}

// Copied and slightly modified for different inputs from mgcl::util::sum
double sum(cl_mem buf, size_t num_elements, cl_context context, cl_program program, cl_command_queue commands,
           bool return_sum, size_t localSize, std::string kernelName, size_t global)
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
