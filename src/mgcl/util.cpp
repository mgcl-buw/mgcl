#include "util.hpp"

#include "kernel_config.hpp"
#include "opencl_helper.hpp"
#include "profiling_data.hpp"

#include <cmath>

namespace mgcl::util
{

    double sum(CuboidGpu& buf, cl_program program, cl_command_queue commands,
               bool return_sum, mgcl::conf::KernelConfig* conf, mgcl::ProfilingData* pd)
    {
        return sum(buf.getBuffer(), buf.getSize(), program, commands, buf.getContext(), return_sum, conf, pd);
    }

    /**
     * @brief Builds the sum of a buffer on device and returns it if return_sum is true.
     * First build partial sums, then build global sum so work-groups get synchronized.
     *
     * @param buf Buffer to build the sum of.
     * @param program
     * @param commands
     * @param return_sum
     * @param conf
     * @param pd
     */
    double sum(cl_mem buf, size_t size, cl_program program, cl_command_queue commands, cl_context context,
               bool return_sum, mgcl::conf::KernelConfig* conf, mgcl::ProfilingData* pd)
    {
        int err;
        size_t num_elements = size;

        const char* kernelNamePartials = "sum_partial_global_eq_x_num_elements";

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
        size_t localSize = 256;

        // Apply kernel config, if available
        if (conf)
        {
            const auto& c = conf::getWorkGroupSizeForKernelAndWiCount(*conf, kernelNamePartials, num_elements);
            localSize = c[0];
        }

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
        fill(program, commands, dPartialSums, 0.0, num_partials, false, conf, pd);

        // Create the compute kernel from the program
        cl_kernel kernel_sum_partial = clCreateKernel(program, kernelNamePartials, &err);
        mgclCheckError(err, "Creating kernel sum_partial_global_eq_x_num_elements");

        cl_mem rawbuf = buf;
        int pos = 0;
        err = clSetKernelArg(kernel_sum_partial, pos, sizeof(cl_mem), &rawbuf);
        err |= clSetKernelArg(kernel_sum_partial, ++pos, sizeof(cl_mem), &dPartialSums);
        err |= clSetKernelArg(kernel_sum_partial, ++pos, localSize * sizeof(double), nullptr);
        err |= clSetKernelArg(kernel_sum_partial, ++pos, sizeof(int), &num_elements);
        err |= clSetKernelArg(kernel_sum_partial, ++pos, sizeof(int), &fractions);
        mgclCheckError(err, "Setting kernel sum_partial arguments");

        cl_event ev;
        err = clEnqueueNDRangeKernel(commands, kernel_sum_partial, 1, NULL, &global, &localSize, 0, NULL, &ev);
        mgclCheckError(err, "Enqueueing kernel sum_partial");

        if (pd != nullptr)
        {
            pd->addMeasurement(commands, ev, kernelNamePartials,
                               {global, 0, 0},
                               {localSize, 1, 1});
        }
        mgclCheckError(clReleaseEvent(ev), "clReleaseEvent");

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

    double max(CuboidGpu& buf, cl_program program, cl_command_queue commands,
               bool return_max, mgcl::conf::KernelConfig* conf, mgcl::ProfilingData* pd)
    {
        return max(buf.getBuffer(), buf.getSize(), program, commands, buf.getContext(), return_max, conf, pd);
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
    double max(cl_mem buf, size_t size, cl_program program, cl_command_queue commands, cl_context context,
               bool return_max, mgcl::conf::KernelConfig* conf, mgcl::ProfilingData* pd)
    {
        int err;
        size_t num_elements = size;

        const char* kernelNamePartials = "max_partial_global_eq_x_num_elements";

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
        size_t localSize = 256;

        // Apply kernel config, if available
        if (conf)
        {
            const auto& c = conf::getWorkGroupSizeForKernelAndWiCount(*conf, kernelNamePartials, num_elements);
            localSize = c[0];
        }

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
        fill(program, commands, dPartialMax, 0.0, num_partials, false, conf, pd);

        // Create the compute kernel from the program
        cl_kernel kernel_max_partial = clCreateKernel(program, kernelNamePartials, &err);
        mgclCheckError(err, "Creating kernel max_partial_global_eq_x_num_elements");

        cl_mem rawbuf = buf;
        int pos = 0;
        err = clSetKernelArg(kernel_max_partial, pos, sizeof(cl_mem), &rawbuf);
        err |= clSetKernelArg(kernel_max_partial, ++pos, sizeof(cl_mem), &dPartialMax);
        err |= clSetKernelArg(kernel_max_partial, ++pos, localSize * sizeof(double), nullptr);
        err |= clSetKernelArg(kernel_max_partial, ++pos, sizeof(int), &num_elements);
        err |= clSetKernelArg(kernel_max_partial, ++pos, sizeof(int), &fractions);
        mgclCheckError(err, "Setting kernel max_partial arguments");

        cl_event ev;
        err = clEnqueueNDRangeKernel(commands, kernel_max_partial, 1, NULL, &global, &localSize, 0, NULL, &ev);
        mgclCheckError(err, "Enqueueing kernel max_partial");

        if (pd != nullptr)
        {
            pd->addMeasurement(commands, ev, kernelNamePartials,
                               {global, 0, 0},
                               {localSize, 1, 1});
        }
        mgclCheckError(clReleaseEvent(ev), "clReleaseEvent");

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

    double max_abs(CuboidGpu& buf, cl_program program, cl_command_queue commands,
                   bool return_max, mgcl::conf::KernelConfig* conf, mgcl::ProfilingData* pd)
    {
        return max_abs(buf.getBuffer(), buf.getSize(), program, commands, buf.getContext(), return_max, conf, pd);
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
    double max_abs(cl_mem buf, size_t size, cl_program program, cl_command_queue commands, cl_context context,
                   bool return_max, mgcl::conf::KernelConfig* conf, mgcl::ProfilingData* pd)
    {
        int err;
        size_t num_elements = size;

        const char* kernelNamePartials = "max_abs_partial_global_eq_x_num_elements";

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
        size_t localSize = 256;

        // Apply kernel config, if available
        if (conf)
        {
            const auto& c = conf::getWorkGroupSizeForKernelAndWiCount(*conf, kernelNamePartials, num_elements);
            localSize = c[0];
        }

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
        fill(program, commands, dPartialMax, 0.0, num_partials, false, conf, pd);

        // Create the compute kernel from the program
        cl_kernel kernel_max_partial = clCreateKernel(program, kernelNamePartials, &err);
        mgclCheckError(err, "Creating kernel max_abs_partial_global_eq_x_num_elements");

        cl_mem rawbuf = buf;
        int pos = 0;
        err = clSetKernelArg(kernel_max_partial, pos, sizeof(cl_mem), &rawbuf);
        err |= clSetKernelArg(kernel_max_partial, ++pos, sizeof(cl_mem), &dPartialMax);
        err |= clSetKernelArg(kernel_max_partial, ++pos, localSize * sizeof(double), nullptr);
        err |= clSetKernelArg(kernel_max_partial, ++pos, sizeof(int), &num_elements);
        err |= clSetKernelArg(kernel_max_partial, ++pos, sizeof(int), &fractions);
        mgclCheckError(err, "Setting kernel max_partial arguments");

        cl_event ev;
        err = clEnqueueNDRangeKernel(commands, kernel_max_partial, 1, NULL, &global, &localSize, 0, NULL, &ev);
        mgclCheckError(err, "Enqueueing kernel max_partial");

        if (pd != nullptr)
        {
            pd->addMeasurement(commands, ev, kernelNamePartials,
                               {global, 0, 0},
                               {localSize, 1, 1});
        }
        mgclCheckError(clReleaseEvent(ev), "clReleaseEvent");

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
     * @brief Fills the device buffer with a given value.
     *
     * @param program
     * @param commands
     * @param buffer Device buffer to be filled
     * @param value Value to fill the buffer with
     * @param size Number of elements in the buffer
     * @param blocking If true, the operation is blocking
     * @param conf
     * @param pd
     */
    void fill(cl_program program, cl_command_queue commands,
              cl_mem buffer, double value, int size, bool blocking,
              mgcl::conf::KernelConfig* conf, mgcl::ProfilingData* pd)
    {
        // Create the compute kernel from the program
        int err;
        const char* kernelName = "fill_buffer";
        cl_kernel kernel = clCreateKernel(program, kernelName, &err);
        mgcl::mgclCheckError(err, "clCreateKernel");

        // assign kernel arguments
        int pos = 0;
        err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &buffer);
        err |= clSetKernelArg(kernel, ++pos, sizeof(double), &value);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &size);
        mgcl::mgclCheckError(err, "Setting kernel arguments");

        // one work-item per ghost cell (excluding real cells). Pad global sizes to fit to local sizes
        size_t global = size;
        size_t local = 64;
        // Apply kernel config, if available
        if (conf)
        {
            const auto& c = conf::getWorkGroupSizeForKernelAndWiCount(*conf, kernelName, global);
            local = c[0];
        }

        if (global % local != 0)
            global += local - (global % local);

        cl_event ev;

        // enqueue kernel
        err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, &ev);
        mgcl::mgclCheckError(err, "Enqueueing fill_buffer kernel");

        if (pd != nullptr)
        {
            pd->addMeasurement(commands, ev, kernelName,
                               {global, 0, 0},
                               {local, 1, 1});
        }
        mgclCheckError(clReleaseEvent(ev), "clReleaseEvent");

        if (blocking)
            mgcl::mgclCheckError(clFinish(commands), "clFinish");

        err = clReleaseKernel(kernel);
        mgcl::mgclCheckError(err, "Releasing fill_buffer kernel");
    }

    // Returns minimum of a, b and c
    int seq::min3(int a, int b, int c)
    {
        if (a <= b && a <= c)
            return a;
        else if (b <= a && b <= c)
            return b;
        return c;
    }

    // Returns minimum of a, b and c
    double seq::min3(double a, double b, double c)
    {
        if (a <= b && a <= c)
            return a;
        else if (b <= a && b <= c)
            return b;
        return c;
    }

} // namespace util
