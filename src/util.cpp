#include "util.hpp"

#include "opencl_helper.hpp"

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

        // One work-item per element in buf
        size_t global = num_elements;

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
        cl_kernel kernel_sum_partial = clCreateKernel(program, "sum_partial", &err);
        mgclCheckError(err, "Creating kernel sum_partial");

        int pos = 0;
        err = clSetKernelArg(kernel_sum_partial, pos, sizeof(cl_mem), &buf);
        err |= clSetKernelArg(kernel_sum_partial, ++pos, sizeof(cl_mem), &dPartialSums);
        err |= clSetKernelArg(kernel_sum_partial, ++pos, localSize * sizeof(double), nullptr);
        err |= clSetKernelArg(kernel_sum_partial, ++pos, sizeof(int), &num_elements);
        mgclCheckError(err, "Setting kernel sum_partial arguments");

        err = clEnqueueNDRangeKernel(commands, kernel_sum_partial, 1, NULL, &global, &localSize, 0, NULL, NULL);
        mgclCheckError(err, "Enqueueing kernel sum_partial");

        // double ret2 = 0;
        // if (return_sum)
        // {
        //     clFinish(commands);

        //     double tmp[num_partials];
        //     err = clEnqueueReadBuffer(commands, dPartialSums, CL_TRUE, 0, sizeof(double) * num_partials,
        //                               &tmp, 0, NULL, NULL);
        //     mgclCheckError(err, "Error: Failed to read dTotalSum from device!");

        //     for (int i = 0; i < num_partials; i++)
        //     {
        //         ret2 += tmp[i];
        //     }
        // }

        // return ret2;

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

        return ret;
    }

} // namespace util
