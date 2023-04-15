#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "ocl_wrapper.hpp"
#include "test_utility.hpp"

#include "../src/cuboid.hpp"
#include "../src/opencl_helper.hpp"
#include "../src/util.hpp"

#ifdef __APPLE__
#include <OpenCL/cl.h> // for clSetKernelArg, _cl_mem, cl_mem, clE...
#else
#include <CL/cl.h> // for clSetKernelArg, _cl_mem, cl_mem, clE...
#endif

#include <cmath>
#include <iomanip>
#include <iostream>

double sum_ocl_naive(cl_mem buf, int num_elements, cl_context context);

// Test if the sum reduction kernel yields correct results
TEST_CASE("util::sum")
{
    // Copied kernel code to run on cpu to find bugs more easily.
    SECTION("seq kernel")
    {
        int num_elements = 4400;
        int localSize = 32;

        // One work-item per element in buf
        size_t global = num_elements;

        // Pad global work-item count to fit wg-size
        if (global % localSize != 0)
            global += localSize - (global % localSize);

        // number of partial sums = num of work-groups
        int num_partials = global / localSize;

        std::vector<double> buf(num_elements);
        std::vector<double> partial_sums(num_partials);
        std::vector<double> buf_local(localSize);

        // fill with data
        double sum_host = 0;
        for (int i = 0; i < num_elements; i++)
        {
            buf[i] = static_cast<double>(i) * (1.0 / 5013.0);
            sum_host += buf[i];
        }

        auto sum_partial = [num_elements, global, localSize, &buf_local, &partial_sums, &buf]()
        {
            int buf_size = num_elements;
            for (int i = 0; i < global; i += localSize)
            {
                // int i = get_global_id(0);
                int wg_size = localSize;
                int iloc = i % wg_size;

                if (i < buf_size)
                {
                    // copy buf of this work-item into local storage
                    // buf_local[iloc] = buf[i];
                    for (int j = 0; j < wg_size; j++)
                    {
                        buf_local[iloc + j] = buf[i + j];
                    }

                    // sum up buf using parallel sum reduction, "a >> 1" == "a / 2" for int
                    // TODO: ensure that stride is even (or handle odd strides)

                    for (int stride = wg_size >> 1; stride > 0; stride >>= 1)
                    {
                        // synchronize local memory
                        // barrier(CLK_LOCAL_MEM_FENCE);

                        // fold upper half onto lower half
                        for (int j = 0; j < stride; j++)
                        {
                            if (iloc + j < stride && iloc + j + stride < wg_size)
                                buf_local[iloc + j] += buf_local[iloc + j + stride];
                        }
                    }

                    // write into output partial_sums
                    if (iloc == 0)
                        partial_sums[i / wg_size] = buf_local[iloc];
                }
            }
        };

        std::vector<double> buf_partial_sums;
        std::vector<double> buf_sum(1);
        int partial_sums_count = num_partials;

        auto sum_finish = [&buf_partial_sums, &buf_sum, partial_sums_count]()
        {
            double sum = 0;

            for (int p = 0; p < partial_sums_count; p++)
                sum += buf_partial_sums[p];

            buf_sum[0] = sum;
        };

        sum_partial();
        buf_partial_sums = partial_sums;
        sum_finish();

        REQUIRE_THAT(sum_host, Catch::Matchers::WithinAbs(buf_sum[0], 1e-12));
    }

    SECTION("sum_ocl_naive")
    {
        int m = 1;
        int n = 1;
        int o = 800000;

        mgcl::Cuboid data(m, n, o);

        double sum_host = 0; //(o / 2) / 512.0;
        double cnt = -o / 2;
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < o; k++)
                {
                    data[i][j][k] = cnt / 5013.0;
                    sum_host += data[i][j][k];
                    cnt++;
                    if (cnt == 0)
                        cnt++;
                    // cnt += 1e-1;
                }

        REQUIRE_THAT(sum_host, Catch::Matchers::WithinAbs(0, 1e-8));

        mgcl_test::TestUtility tu;
        cl_mem dBuf = tu.createOpenCLBuffer(data);

        double sum_ocl = sum_ocl_naive(dBuf, m * n * o, tu.getContext());

        REQUIRE_THAT(sum_host, Catch::Matchers::WithinAbs(sum_ocl, 1e-8));
    }

    SECTION("opencl kernel")
    {
        SECTION("sum to 0")
        {
            mgcl_test::TestUtility tu(CL_DEVICE_TYPE_GPU);

            // size_t local = GENERATE(8, 32, 43);
            size_t local = 32;
            int m = 1;
            int n = 1;
            // int o = GENERATE(1, 2, range(5, 100, 7), 84848); // 1056; // GENERATE(1, 2, 32, 47);
            int o = 80000;

            // std::cout << std::to_string(m) << "," << std::to_string(n) << "," << std::to_string(o) << "..." << std::endl;

            mgcl::Cuboid data(m, n, o);

            double sum_host = 0; //(o / 2) / 512.0;
            double cnt = -o / 2;
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    for (int k = 0; k < o; k++)
                    {
                        data[i][j][k] = cnt / 5013.0;
                        sum_host += data[i][j][k];
                        cnt++;
                        if (cnt == 0)
                            cnt++;
                        // cnt += 1e-1;
                    }

            REQUIRE_THAT(sum_host, Catch::Matchers::WithinAbs(0, 1e-8));

            cl_mem dData = tu.createOpenCLBuffer(data);
            double sum_device = mgcl::util::sum(dData, data.field1d().size(), tu.getContext(), tu.getProgram(),
                                                tu.getCommands(), true, local);

            // std::cout << std::scientific << std::setprecision(17) << "  host: " << sum_host << std::endl
            //           << "device: " << sum_device << std::endl;
            REQUIRE_THAT(sum_host, Catch::Matchers::WithinAbs(sum_device, 1e-8));
            // std::cout << std::to_string(m) << "," << std::to_string(n) << "," << std::to_string(o) << "...OK" << std::endl;
        }

        SECTION("arbitrary values")
        {

            mgcl_test::TestUtility tu(CL_DEVICE_TYPE_GPU);

            // size_t local = GENERATE(8, 32, 43);
            size_t local = 32;
            int m = 1;
            int n = 1;
            int o = GENERATE(1, 2, range(5, 100, 7), 8484); // 1056; // GENERATE(1, 2, 32, 47);
            // int o = 80484;

            // std::cout << std::to_string(m) << "," << std::to_string(n) << "," << std::to_string(o) << "..." << std::endl;

            mgcl::Cuboid data(m, n, o);

            double sum_host = 0;
            double cnt = 0;
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    for (int k = 0; k < o; k++)
                    {
                        data[i][j][k] = cnt;
                        sum_host += data[i][j][k];
                        cnt += 1.0 / 5013.0;
                        // cnt += 1e-1;
                    }

            cl_mem dData = tu.createOpenCLBuffer(data);
            double sum_device = mgcl::util::sum(dData, data.field1d().size(), tu.getContext(), tu.getProgram(),
                                                tu.getCommands(), true, local);

            // std::cout << std::scientific << std::setprecision(17) << "  host: " << sum_host << std::endl
            //           << "device: " << sum_device << std::endl;
            REQUIRE_THAT(sum_host, Catch::Matchers::WithinAbs(sum_device, 1e-8));
            // std::cout << std::to_string(m) << "," << std::to_string(n) << "," << std::to_string(o) << "...OK" << std::endl;
        }
    }
}

// Test if the sum reduction kernel yields correct results
TEST_CASE("util::max")
{
    mgcl_test::TestUtility tu(CL_DEVICE_TYPE_GPU);

    // size_t local = GENERATE(8, 32, 43);
    size_t local = 32;
    int m = 1;
    int n = 1;
    int o = GENERATE(1, 2, range(5, 100, 7), 8484); // 1056; // GENERATE(1, 2, 32, 47);
    // int o = 80484;

    // std::cout << std::to_string(m) << "," << std::to_string(n) << "," << std::to_string(o) << "..." << std::endl;

    mgcl::Cuboid data(m, n, o);
    data.fillRandom(-10, 10);

    double max_host = -100000;
    // double cnt = 0;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < o; k++)
            {
                // data[i][j][k] = cnt;
                if (data[i][j][k] > max_host)
                    max_host = data[i][j][k];
                // cnt += 1.0 / 5013.0;
                // cnt++;
                // cnt += 1e-1;
            }

    cl_mem dData = tu.createOpenCLBuffer(data);
    double max_device = mgcl::util::max(dData, data.field1d().size(), tu.getContext(), tu.getProgram(),
                                        tu.getCommands(), true, local);

    // std::cout << std::scientific << std::setprecision(17) << "  host: " << sum_host << std::endl
    //           << "device: " << sum_device << std::endl;
    REQUIRE_THAT(max_host, Catch::Matchers::WithinAbs(max_device, 1e-8));
    // std::cout << std::to_string(m) << "," << std::to_string(n) << "," << std::to_string(o) << "...OK" << std::endl;
}

// Test if the sum reduction kernel yields correct results
TEST_CASE("util::max_abs")
{
    mgcl_test::TestUtility tu(CL_DEVICE_TYPE_GPU);

    // size_t local = GENERATE(8, 32, 43);
    size_t local = 32;
    int m = 1;
    int n = 1;
    int o = GENERATE(1, 2, range(5, 100, 7), 8484); // 1056; // GENERATE(1, 2, 32, 47);
    // int o = 80484;

    // std::cout << std::to_string(m) << "," << std::to_string(n) << "," << std::to_string(o) << "..." << std::endl;

    mgcl::Cuboid data(m, n, o);
    data.fillRandom(-10, 10);
    data[0][0][0] = -10000;
    double max_abs_host = fabs(data[0][0][0]);

    cl_mem dData = tu.createOpenCLBuffer(data);
    double max_abs_device = mgcl::util::max_abs(dData, data.field1d().size(), tu.getContext(), tu.getProgram(),
                                                tu.getCommands(), true, local);

    // std::cout << std::scientific << std::setprecision(17) << "  host: " << sum_host << std::endl
    //           << "device: " << sum_device << std::endl;
    REQUIRE_THAT(max_abs_host, Catch::Matchers::WithinAbs(max_abs_device, 1e-8));
    // std::cout << std::to_string(m) << "," << std::to_string(n) << "," << std::to_string(o) << "...OK" << std::endl;
}

// Builds sum on device using a naive kernel (only 1 work-item iterating over all elements).
double sum_ocl_naive(cl_mem buf, int num_elements, cl_context context)
{
    std::string kernelSrc = R"DELIM(
        __kernel void sum_naive(
            __global double *restrict buf,
            __global double *restrict out,
            int num_elements
        )
        {
            double sum = 0;

            for (int p = 0; p < num_elements; p++)
                sum += buf[p];

            out[0] = sum;
        }
    )DELIM";

    OCLWrapper w(CL_DEVICE_TYPE_GPU, "", kernelSrc, "", context);

    int err;
    cl_kernel sum_naive_kernel = clCreateKernel(w.program, "sum_naive", &err);
    mgcl::mgclCheckError(err, "Creating kernel sum_naive");

    size_t one = 1;
    cl_mem dOut = clCreateBuffer(w.context, CL_MEM_WRITE_ONLY, sizeof(double), nullptr, &err);
    mgcl::mgclCheckError(err, "Creating dPartialSums buffer");

    int pos = 0;
    err = clSetKernelArg(sum_naive_kernel, pos, sizeof(cl_mem), &buf);
    err |= clSetKernelArg(sum_naive_kernel, ++pos, sizeof(cl_mem), &dOut);
    err |= clSetKernelArg(sum_naive_kernel, ++pos, sizeof(int), &num_elements);
    mgcl::mgclCheckError(err, "Setting kernel sum_naive arguments");

    size_t global = 1;
    size_t local = 1;
    err = clEnqueueNDRangeKernel(w.commands, sum_naive_kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    mgcl::mgclCheckError(err, "Enqueueing kernel sum_naive");

    err = clFinish(w.commands);
    mgcl::mgclCheckError(err, "Finishing kernel sum_naive");

    double tmp;
    err = clEnqueueReadBuffer(w.commands, dOut, CL_TRUE, 0, sizeof(double), &tmp, 0, NULL, NULL);
    mgcl::mgclCheckError(err, "Error: Failed to read dOut from device!");

    return tmp;
}
