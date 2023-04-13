#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "test_utility.hpp"

#include "../src/cuboid.hpp"
#include "../src/util.hpp"

#ifdef __APPLE__
#include <OpenCL/cl.h> // for clSetKernelArg, _cl_mem, cl_mem, clE...
#else
#include <CL/cl.h> // for clSetKernelArg, _cl_mem, cl_mem, clE...
#endif

#include <iomanip>
#include <iostream>

double sum_ocl_naive(std::vector<double> input);

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
            int o = 4400;

            // std::cout << std::to_string(m) << "," << std::to_string(n) << "," << std::to_string(o) << "..." << std::endl;

            mgcl::Cuboid data(m, n, o);

            double sum_host = 0; //(o / 2) / 512.0;
            double cnt = -o / 2;
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    for (int k = 0; k < o; k++)
                    {
                        data[i][j][k] = cnt / 512.0;
                        sum_host += data[i][j][k];
                        cnt++;
                        if (cnt == 0)
                            cnt++;
                        // cnt += 1e-1;
                    }

            REQUIRE(sum_host == 0);

            cl_mem dData = tu.createOpenCLBuffer(data);
            double sum_device = mgcl::util::sum(dData, data.field1d().size(), tu.getContext(), tu.getProgram(),
                                                tu.getCommands(), true, local);

            // std::cout << std::scientific << std::setprecision(17) << "  host: " << sum_host << std::endl
            //           << "device: " << sum_device << std::endl;
            REQUIRE_THAT(sum_host, Catch::Matchers::WithinAbs(sum_device, 1e-12));
            // std::cout << std::to_string(m) << "," << std::to_string(n) << "," << std::to_string(o) << "...OK" << std::endl;
        }

        SECTION("arbitrary values")
        {

            mgcl_test::TestUtility tu(CL_DEVICE_TYPE_GPU);

            // size_t local = GENERATE(8, 32, 43);
            size_t local = 32;
            int m = 1;
            int n = 1;
            // int o = GENERATE(1, 2, range(5, 100, 7), 8484); // 1056; // GENERATE(1, 2, 32, 47);
            int o = 8484;

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
            REQUIRE_THAT(sum_host, Catch::Matchers::WithinAbs(sum_device, 1e-11));
            // std::cout << std::to_string(m) << "," << std::to_string(n) << "," << std::to_string(o) << "...OK" << std::endl;
        }
    }
}

// Builds sum on device using a naive kernel (only 1 work-item iterating over all elements).
double sum_ocl_naive(std::vector<double> input)
{
}
