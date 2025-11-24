#include "util.hpp"

#include "kernel_config.hpp"
#include "opencl_helper.hpp"
#include "profiling_data.hpp"

#include <cmath>

namespace mgcl::util
{
    size_t DEFAULT_REDUCTION_MAX_WG_SIZE = 1024;

    size_t nextPowerOfTwo(int x)
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

    double sum(CuboidGpu& buf, cl_program program, cl_command_queue commands,
               bool return_sum, size_t maxWgSize, mgcl::conf::KernelConfig* conf, mgcl::ProfilingData* pd)
    {
        return sum(buf.getBuffer(), buf.getSize(), program, commands, buf.getContext(), return_sum, maxWgSize, conf, pd);
    }

    double sum(cl_mem buf, size_t size, cl_program program, cl_command_queue commands, cl_context context,
               bool return_sum, size_t maxWgSize, mgcl::conf::KernelConfig* conf, mgcl::ProfilingData* pd)
    {
        return reduce(buf, size, program, commands, context, return_sum, maxWgSize, "sum_partial_global_eq_x_num_elements", conf, pd);
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
    double reduce(cl_mem buf, size_t size, cl_program program, cl_command_queue commands, cl_context context,
                  bool return_sum, size_t maxWgSize, std::string kernelName,
                  mgcl::conf::KernelConfig* conf, mgcl::ProfilingData* pd)
    {
        int err;
        size_t num_elements = size;

        // const char* kernelName = "sum_partial_global_eq_x_num_elements";

        // Set batchSize according to Brent's theorem (each work-item does log2(N) work)
        int batchSize = num_elements <= 1 ? 1 : std::log2(num_elements);

        size_t global = ceil((1.0 / batchSize) * num_elements);
        size_t localSize = 256;

        // Apply kernel config, if available
        if (conf)
        {
            const auto& c = conf::getWorkGroupSizeForKernelAndWiCount(*conf, kernelName, num_elements);
            localSize = c[0];
        }

        // Pad global work-item count to fit wg-size
        if (global % localSize != 0)
            global += localSize - (global % localSize);

        // number of partial sums = num of work-groups
        int num_partials = ceil(global / static_cast<double>(localSize));

        // int pointer_flag = problem.getOpenCLHelper().getDeviceType() == CL_DEVICE_TYPE_GPU ? CL_MEM_COPY_HOST_PTR : CL_MEM_USE_HOST_PTR;
        cl_mem dPartialSums = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                             sizeof(double) * num_partials, nullptr, &err);
        mgclCheckError(err, "Creating dPartialSums buffer");

        // fill buffer with zeros
        fill(program, commands, dPartialSums, 0.0, num_partials, false, conf, pd);

        // Create the compute kernel from the program
        cl_kernel kernel_sum_partial = clCreateKernel(program, kernelName.c_str(), &err);
        mgclCheckError(err, "Creating kernel " + kernelName);

        cl_mem src = buf;
        do // while num_elements > 1
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
            mgcl::mgclCheckError(err, "Setting kernel " + kernelName + " arguments");

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
                size_t tmp = nextPowerOfTwo(global);
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
        } while (num_elements > 1);

        double ret = 0;
        if (return_sum)
        {
            clFinish(commands);

            err = clEnqueueReadBuffer(commands, dPartialSums, CL_TRUE, 0, sizeof(double),
                                      &ret, 0, NULL, NULL);
            mgclCheckError(err, "Error: Failed to read dPartialSums from device!");
        }

        err = clReleaseMemObject(dPartialSums);
        mgclCheckError(err, "clReleaseMemObject dPartialSums");

        mgclCheckError(clReleaseKernel(kernel_sum_partial), "clReleaseKernel(" + kernelName + ")");

        return ret;
    }

    double max(CuboidGpu& buf, cl_program program, cl_command_queue commands,
               bool return_max, size_t maxWgSize, mgcl::conf::KernelConfig* conf, mgcl::ProfilingData* pd)
    {
        return max(buf.getBuffer(), buf.getSize(), program, commands, buf.getContext(), return_max, maxWgSize, conf, pd);
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
               bool return_max, size_t maxWgSize, mgcl::conf::KernelConfig* conf, mgcl::ProfilingData* pd)
    {
        return reduce(buf, size, program, commands, context, return_max, maxWgSize, "max_partial_global_eq_x_num_elements", conf, pd);
    }

    double max_abs(CuboidGpu& buf, cl_program program, cl_command_queue commands,
                   bool return_max, size_t maxWgSize, mgcl::conf::KernelConfig* conf, mgcl::ProfilingData* pd)
    {
        return max_abs(buf.getBuffer(), buf.getSize(), program, commands, buf.getContext(), return_max, maxWgSize, conf, pd);
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
                   bool return_max, size_t maxWgSize, mgcl::conf::KernelConfig* conf, mgcl::ProfilingData* pd)
    {
        return reduce(buf, size, program, commands, context, return_max, maxWgSize, "max_abs_partial_global_eq_x_num_elements", conf, pd);
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
