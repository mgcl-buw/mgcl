#include "util.hpp"

#include "opencl_helper.hpp"

#include <cmath>

namespace mgcl::util
{

    /**
     * @brief Builds the sum of a buffer on device and returns it if return_sum is true.
     * First build partial sums, then build global sum so work-groups get synchronized.
     *
     * @param buf Buffer to build the sum of.
     * @param num_elements
     * @param context
     * @param program
     * @param commands
     */
    double sum(cl_mem buf, size_t num_elements, cl_context context, cl_program program, cl_command_queue commands,
               bool return_sum, size_t localSize)
    {
        int err;

        // Determine number of work-items worked out in benchmarks manually.
        int fractions = 4;
        if (num_elements > 16e6) // > 256^3
            fractions = 512;
        else if (num_elements > 2e6) // > 128^3
            fractions = 128;
        else if (num_elements > 250000) // > 64^3
            fractions = 32;

        // One work-item per element in buf
        size_t global = ceil((1.0 / fractions) * num_elements);

        // Pad global work-item count to fit wg-size
        if (global % localSize != 0)
            global += localSize - (global % localSize);

        // number of partial sums = num of work-groups
        int num_partials = global / localSize;

        // int pointer_flag = problem.getOpenCLHelper().getDeviceType() == CL_DEVICE_TYPE_GPU ? CL_MEM_COPY_HOST_PTR : CL_MEM_USE_HOST_PTR;
        cl_mem dPartialSums = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                             sizeof(double) * num_partials, nullptr, &err);
        mgclCheckError(err, "Creating dPartialSums buffer");

        // fill buffer with zeros
        double zero = 0;
        err = clEnqueueFillBuffer(commands, dPartialSums, &zero, sizeof(double), 0,
                                  sizeof(double) * num_partials, 0, NULL, NULL);
        mgclCheckError(err, "setting dPartialSums to 0");

        // Create the compute kernel from the program
        cl_kernel kernel_sum_partial = clCreateKernel(program, "sum_partial_global_eq_x_num_elements", &err);
        mgclCheckError(err, "Creating kernel sum_partial_global_eq_x_num_elements");

        int pos = 0;
        err = clSetKernelArg(kernel_sum_partial, pos, sizeof(cl_mem), &buf);
        err |= clSetKernelArg(kernel_sum_partial, ++pos, sizeof(cl_mem), &dPartialSums);
        err |= clSetKernelArg(kernel_sum_partial, ++pos, localSize * sizeof(double), nullptr);
        err |= clSetKernelArg(kernel_sum_partial, ++pos, sizeof(int), &num_elements);
        err |= clSetKernelArg(kernel_sum_partial, ++pos, sizeof(int), &fractions);
        mgclCheckError(err, "Setting kernel sum_partial arguments");

        err = clEnqueueNDRangeKernel(commands, kernel_sum_partial, 1, NULL, &global, &localSize, 0, NULL, NULL);
        mgclCheckError(err, "Enqueueing kernel sum_partial");

        // Create the compute kernel from the program
        cl_kernel kernel_sum_finish = clCreateKernel(program, "sum_finish", &err);
        mgclCheckError(err, "Creating sum_finish kernel");

        cl_mem dTotalSum = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double), nullptr, &err);
        mgclCheckError(err, "Creating dTotalSum buffer");

        pos = 0;
        err = clSetKernelArg(kernel_sum_finish, pos, sizeof(cl_mem), &dPartialSums);
        err |= clSetKernelArg(kernel_sum_finish, ++pos, sizeof(cl_mem), &dTotalSum);
        err |= clSetKernelArg(kernel_sum_finish, ++pos, sizeof(int), &num_partials);
        mgclCheckError(err, "Setting sum_finish kernel arguments");

        size_t one = 1;
        err = clEnqueueNDRangeKernel(commands, kernel_sum_finish, 1, NULL, &one, &one, 0, NULL, NULL);
        mgclCheckError(err, "Enqueueing kernel sum_finish");

        double ret = 0;
        if (return_sum)
        {
            clFinish(commands);

            err = clEnqueueReadBuffer(commands, dTotalSum, CL_TRUE, 0, sizeof(double),
                                      &ret, 0, NULL, NULL);
            mgclCheckError(err, "Error: Failed to read dTotalSum from device!");
        }

        err = clReleaseMemObject(dPartialSums);
        mgclCheckError(err, "clReleaseMemObject dPartialSums");

        err = clReleaseMemObject(dTotalSum);
        mgclCheckError(err, "clReleaseMemObject dTotalSum");

        mgclCheckError(clReleaseKernel(kernel_sum_partial), "clReleaseKernel(kernel_sum_partial)");
        mgclCheckError(clReleaseKernel(kernel_sum_finish), "clReleaseKernel(kernel_sum_partial)");

        return ret;
    }

    /**
     * @brief Finds the maximum value of a buffer on device and returns it if return_max is true.
     * First build partial maxima, then build global maximum so work-groups get synchronized.
     *
     * @param buf Buffer to find the maximum in.
     * @param num_elements
     * @param context
     * @param program
     * @param commands
     */
    double max(cl_mem buf, size_t num_elements, cl_context context, cl_program program, cl_command_queue commands,
               bool return_max, size_t localSize)
    {
        int err;

        // Determine number of work-items worked out in benchmarks manually.
        int fractions = 4;
        if (num_elements > 16e6) // > 256^3
            fractions = 512;
        else if (num_elements > 2e6) // > 128^3
            fractions = 128;
        else if (num_elements > 250000) // > 64^3
            fractions = 32;

        // One work-item per element in buf
        size_t global = ceil((1.0 / fractions) * num_elements);

        // Pad global work-item count to fit wg-size
        if (global % localSize != 0)
            global += localSize - (global % localSize);

        // number of partial maxima = num of work-groups
        int num_partials = global / localSize;

        // int pointer_flag = problem.getOpenCLHelper().getDeviceType() == CL_DEVICE_TYPE_GPU ? CL_MEM_COPY_HOST_PTR : CL_MEM_USE_HOST_PTR;
        cl_mem dPartialMax = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                            sizeof(double) * num_partials, nullptr, &err);
        mgclCheckError(err, "Creating dPartialMax buffer");

        // fill buffer with zeros
        double zero = 0;
        err = clEnqueueFillBuffer(commands, dPartialMax, &zero, sizeof(double), 0,
                                  sizeof(double) * num_partials, 0, NULL, NULL);
        mgclCheckError(err, "setting dPartialMax to 0");

        // Create the compute kernel from the program
        cl_kernel kernel_max_partial = clCreateKernel(program, "max_partial_global_eq_x_num_elements", &err);
        mgclCheckError(err, "Creating kernel max_partial_global_eq_x_num_elements");

        int pos = 0;
        err = clSetKernelArg(kernel_max_partial, pos, sizeof(cl_mem), &buf);
        err |= clSetKernelArg(kernel_max_partial, ++pos, sizeof(cl_mem), &dPartialMax);
        err |= clSetKernelArg(kernel_max_partial, ++pos, localSize * sizeof(double), nullptr);
        err |= clSetKernelArg(kernel_max_partial, ++pos, sizeof(int), &num_elements);
        err |= clSetKernelArg(kernel_max_partial, ++pos, sizeof(int), &fractions);
        mgclCheckError(err, "Setting kernel max_partial arguments");

        err = clEnqueueNDRangeKernel(commands, kernel_max_partial, 1, NULL, &global, &localSize, 0, NULL, NULL);
        mgclCheckError(err, "Enqueueing kernel max_partial");

        // Create the compute kernel from the program
        cl_kernel kernel_max_finish = clCreateKernel(program, "max_finish", &err);
        mgclCheckError(err, "Creating max_finish kernel");

        cl_mem dTotalMax = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double), nullptr, &err);
        mgclCheckError(err, "Creating dTotalMax buffer");

        pos = 0;
        err = clSetKernelArg(kernel_max_finish, pos, sizeof(cl_mem), &dPartialMax);
        err |= clSetKernelArg(kernel_max_finish, ++pos, sizeof(cl_mem), &dTotalMax);
        err |= clSetKernelArg(kernel_max_finish, ++pos, sizeof(int), &num_partials);
        mgclCheckError(err, "Setting max_finish kernel arguments");

        size_t one = 1;
        err = clEnqueueNDRangeKernel(commands, kernel_max_finish, 1, NULL, &one, &one, 0, NULL, NULL);
        mgclCheckError(err, "Enqueueing kernel max_finish");

        double ret = 0;
        if (return_max)
        {
            clFinish(commands);

            err = clEnqueueReadBuffer(commands, dTotalMax, CL_TRUE, 0, sizeof(double),
                                      &ret, 0, NULL, NULL);
            mgclCheckError(err, "Error: Failed to read dTotalMax from device!");
        }

        err = clReleaseMemObject(dPartialMax);
        mgclCheckError(err, "clReleaseMemObject dPartialMax");

        err = clReleaseMemObject(dTotalMax);
        mgclCheckError(err, "clReleaseMemObject dTotalMax");

        mgclCheckError(clReleaseKernel(kernel_max_partial), "clReleaseKernel(kernel_max_partial)");
        mgclCheckError(clReleaseKernel(kernel_max_finish), "clReleaseKernel(kernel_max_finish)");

        return ret;
    }

    /**
     * @brief Finds the maximum absolute value of a buffer on device and returns it if return_max is true.
     * First build partial maxima, then build global maximum so work-groups get synchronized.
     *
     * @param buf Buffer to find the maximum in.
     * @param num_elements
     * @param context
     * @param program
     * @param commands
     */
    double max_abs(cl_mem buf, size_t num_elements, cl_context context, cl_program program, cl_command_queue commands,
                   bool return_max, size_t localSize)
    {
        int err;

        // Determine number of work-items worked out in benchmarks manually.
        int fractions = 4;
        if (num_elements > 16e6) // > 256^3
            fractions = 512;
        else if (num_elements > 2e6) // > 128^3
            fractions = 128;
        else if (num_elements > 250000) // > 64^3
            fractions = 32;

        // One work-item per element in buf
        size_t global = ceil((1.0 / fractions) * num_elements);

        // Pad global work-item count to fit wg-size
        if (global % localSize != 0)
            global += localSize - (global % localSize);

        // number of partial maxima = num of work-groups
        int num_partials = global / localSize;

        // int pointer_flag = problem.getOpenCLHelper().getDeviceType() == CL_DEVICE_TYPE_GPU ? CL_MEM_COPY_HOST_PTR : CL_MEM_USE_HOST_PTR;
        cl_mem dPartialMax = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                            sizeof(double) * num_partials, nullptr, &err);
        mgclCheckError(err, "Creating dPartialMax buffer");

        // fill buffer with zeros
        double zero = 0;
        err = clEnqueueFillBuffer(commands, dPartialMax, &zero, sizeof(double), 0,
                                  sizeof(double) * num_partials, 0, NULL, NULL);
        mgclCheckError(err, "setting dPartialMax to 0");

        // Create the compute kernel from the program
        cl_kernel kernel_max_partial = clCreateKernel(program, "max_abs_partial_global_eq_x_num_elements", &err);
        mgclCheckError(err, "Creating kernel max_abs_partial_global_eq_x_num_elements");

        int pos = 0;
        err = clSetKernelArg(kernel_max_partial, pos, sizeof(cl_mem), &buf);
        err |= clSetKernelArg(kernel_max_partial, ++pos, sizeof(cl_mem), &dPartialMax);
        err |= clSetKernelArg(kernel_max_partial, ++pos, localSize * sizeof(double), nullptr);
        err |= clSetKernelArg(kernel_max_partial, ++pos, sizeof(int), &num_elements);
        err |= clSetKernelArg(kernel_max_partial, ++pos, sizeof(int), &fractions);
        mgclCheckError(err, "Setting kernel max_partial arguments");

        err = clEnqueueNDRangeKernel(commands, kernel_max_partial, 1, NULL, &global, &localSize, 0, NULL, NULL);
        mgclCheckError(err, "Enqueueing kernel max_partial");

        // Create the compute kernel from the program
        cl_kernel kernel_max_finish = clCreateKernel(program, "max_finish", &err);
        mgclCheckError(err, "Creating max_finish kernel");

        cl_mem dTotalMax = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double), nullptr, &err);
        mgclCheckError(err, "Creating dTotalMax buffer");

        pos = 0;
        err = clSetKernelArg(kernel_max_finish, pos, sizeof(cl_mem), &dPartialMax);
        err |= clSetKernelArg(kernel_max_finish, ++pos, sizeof(cl_mem), &dTotalMax);
        err |= clSetKernelArg(kernel_max_finish, ++pos, sizeof(int), &num_partials);
        mgclCheckError(err, "Setting max_finish kernel arguments");

        size_t one = 1;
        err = clEnqueueNDRangeKernel(commands, kernel_max_finish, 1, NULL, &one, &one, 0, NULL, NULL);
        mgclCheckError(err, "Enqueueing kernel max_finish");

        double ret = 0;
        if (return_max)
        {
            clFinish(commands);

            err = clEnqueueReadBuffer(commands, dTotalMax, CL_TRUE, 0, sizeof(double),
                                      &ret, 0, NULL, NULL);
            mgclCheckError(err, "Error: Failed to read dTotalMax from device!");
        }

        err = clReleaseMemObject(dPartialMax);
        mgclCheckError(err, "clReleaseMemObject dPartialMax");

        err = clReleaseMemObject(dTotalMax);
        mgclCheckError(err, "clReleaseMemObject dTotalMax");

        mgclCheckError(clReleaseKernel(kernel_max_partial), "clReleaseKernel(kernel_max_partial)");
        mgclCheckError(clReleaseKernel(kernel_max_finish), "clReleaseKernel(kernel_max_finish)");

        return ret;
    }

    // Returns minimum of a, b and c
    int seq::min3(int a, int b, int c)
    {
        if (a < b && a < c)
            return a;
        else if (b < a && b < c)
            return b;
        return c;
    }

    // Returns minimum of a, b and c
    double seq::min3(double a, double b, double c)
    {
        if (a < b && a < c)
            return a;
        else if (b < a && b < c)
            return b;
        return c;
    }

} // namespace util
