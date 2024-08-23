#include "buffer_gpu.hpp"

#include "mgcl.hpp"
#include "opencl_helper.hpp"
#include <CL/cl.h>
#include <cstddef>
#include <memory>
#include <vector>

namespace mgcl
{

    BufferGpu::BufferGpu(cl_context context, cl_mem_flags flags, size_t size)
        : context(context), _size(size)
    {
        bool containsReadWrite = (flags & CL_MEM_READ_WRITE) == CL_MEM_READ_WRITE;
        bool containsWriteOnly = (flags & CL_MEM_WRITE_ONLY) == CL_MEM_WRITE_ONLY;
        bool containsReadOnly = (flags & CL_MEM_READ_ONLY) == CL_MEM_READ_ONLY;

        // Check that flags contains one and only one of CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY or CL_MEM_READ_ONLY
        if (!containsReadWrite && !containsWriteOnly && !containsReadOnly)
            error("flags must contain one of CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY or CL_MEM_READ_ONLY.");

        if ((containsReadWrite && containsReadOnly) || (containsReadWrite && containsWriteOnly) || (containsReadOnly && containsWriteOnly))
            error("flags must contain one and only one of CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY or CL_MEM_READ_WRITE.");

        bool containsCopyHostPtr = (flags & CL_MEM_COPY_HOST_PTR) == CL_MEM_COPY_HOST_PTR;
        bool containsUseHostPtr = (flags & CL_MEM_USE_HOST_PTR) == CL_MEM_USE_HOST_PTR;
        bool containsAllocHostPtr = (flags & CL_MEM_ALLOC_HOST_PTR) == CL_MEM_ALLOC_HOST_PTR;

        // Check that flags does not contain one of CL_MEM_COPY_HOST_PTR, CL_MEM_USE_HOST_PTR or CL_MEM_ALLOC_HOST_PTR
        if ((containsAllocHostPtr || containsUseHostPtr || containsCopyHostPtr))
            error("flags contain CL_MEM_ALLOC_HOST_PTR, CL_MEM_USE_HOST_PTR or CL_MEM_COPY_HOST_PTR, which is not allowed when creating a new buffer.");

        cl_int err;
        buf = clCreateBuffer(context, flags, sizeof(double) * size, nullptr, &err);
        mgclCheckError(err, "clCreateBuffer");
    }

    /**
     * @brief Creates a device buffer and copies data from h_data.
     *
     * @param context
     * @param flags
     * @param size
     */
    BufferGpu::BufferGpu(cl_context context, cl_mem_flags flags, std::vector<double>& h_data)
        : context(context), _size(h_data.size())
    {
        bool containsReadWrite = (flags & CL_MEM_READ_WRITE) == CL_MEM_READ_WRITE;
        bool containsWriteOnly = (flags & CL_MEM_WRITE_ONLY) == CL_MEM_WRITE_ONLY;
        bool containsReadOnly = (flags & CL_MEM_READ_ONLY) == CL_MEM_READ_ONLY;

        // Check that flags contains one and only one of CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY or CL_MEM_READ_ONLY
        if (!containsReadWrite && !containsWriteOnly && !containsReadOnly)
            error("flags must contain one of CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY or CL_MEM_READ_ONLY.");

        if ((containsReadWrite && containsReadOnly) || (containsReadWrite && containsWriteOnly) || (containsReadOnly && containsWriteOnly))
            error("flags must contain one and only one of CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY or CL_MEM_READ_WRITE.");

        bool containsCopyHostPtr = (flags & CL_MEM_COPY_HOST_PTR) == CL_MEM_COPY_HOST_PTR;
        bool containsUseHostPtr = (flags & CL_MEM_USE_HOST_PTR) == CL_MEM_USE_HOST_PTR;
        bool containsAllocHostPtr = (flags & CL_MEM_ALLOC_HOST_PTR) == CL_MEM_ALLOC_HOST_PTR;

        if ((containsAllocHostPtr && containsUseHostPtr) || (containsAllocHostPtr && containsCopyHostPtr) || (containsUseHostPtr && containsCopyHostPtr))
            error("flags must contain one and only one of CL_MEM_ALLOC_HOST_PTR, CL_MEM_USE_HOST_PTR or CL_MEM_COPY_HOST_PTR.");

        cl_int err;
        buf = clCreateBuffer(context, flags, sizeof(double) * _size, h_data.data(), &err);
        mgclCheckError(err, "clCreateBuffer");
    }

    BufferGpu::~BufferGpu()
    {
        if (buf != nullptr)
            mgclCheckError(clReleaseMemObject(buf), "clReleaseMemObject");
    }

    /**
     * @brief Reads and returns the whole device buffer.
     *
     * @param queue
     * @param h_target
     * @param blocking
     * @return std::unique_ptr<std::vector<double>>
     */
    std::unique_ptr<std::vector<double>> BufferGpu::read(cl_command_queue queue, double* h_target, bool blocking) const
    {
        return read(queue, h_target, blocking, _size);
    }

    /**
     * @brief Reads and returns 'size' elements from the device buffer.
     *
     * @param queue
     * @param h_target
     * @param blocking
     * @param _size
     * @return std::unique_ptr<std::vector<double>>
     */
    std::unique_ptr<std::vector<double>> BufferGpu::read(cl_command_queue queue, double* h_target, bool blocking, size_t _size) const
    {
        std::unique_ptr<std::vector<double>> ret = nullptr;
        int err;
        if (h_target)
        {
            err = clEnqueueReadBuffer(queue, buf, blocking ? CL_TRUE : CL_FALSE, 0, sizeof(double) * _size, h_target, 0, nullptr, nullptr);
        }
        else
        {
            ret = std::make_unique<std::vector<double>>(_size);
            err = clEnqueueReadBuffer(queue, buf, blocking ? CL_TRUE : CL_FALSE, 0, sizeof(double) * _size, ret->data(), 0, nullptr, nullptr);
        }
        mgclCheckError(err, "clEnqueueReadBuffer");
        return ret;
    }

    /**
     * @brief Writes whole data into the device buffer.
     *
     * @param queue
     * @param host_data
     * @param blocking
     */
    void BufferGpu::write(cl_command_queue queue, const std::vector<double>& host_data, bool blocking)
    {
        write(queue, host_data, blocking, _size);
    }

    /**
     * @brief Writes '_size' elements into the device buffer.
     *
     * @param queue
     * @param host_data
     * @param blocking
     * @param _size
     */
    void BufferGpu::write(cl_command_queue queue, const std::vector<double>& host_data, bool blocking, size_t _size)
    {
        if (host_data.size() < _size)
            error("CuboidGpu::write: host_data is not big enough!");

        int err = clEnqueueWriteBuffer(queue, buf, blocking ? CL_TRUE : CL_FALSE, 0, sizeof(double) * _size, host_data.data(), 0, nullptr, nullptr);
        mgclCheckError(err, "clEnqueueWriteBuffer");
    }

    void BufferGpu::fill(cl_program program, cl_command_queue queue, double value, bool blocking,
                         conf::KernelConfig* conf, mgcl::ProfilingData* pd)
    {
        // Create the compute kernel from the program
        int err;
        const char* kernelName = "fill_buffer";
        cl_kernel kernel = clCreateKernel(program, kernelName, &err);
        mgcl::mgclCheckError(err, "clCreateKernel");

        // assign kernel arguments
        int pos = 0;
        err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &buf);
        err |= clSetKernelArg(kernel, ++pos, sizeof(double), &value);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &_size);
        mgcl::mgclCheckError(err, "Setting kernel arguments");

        // one work-item per ghost cell (excluding real cells). Pad global sizes to fit to local sizes
        size_t global = _size;
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
        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0, NULL, &ev);
        mgcl::mgclCheckError(err, "Enqueueing fill_buffer kernel");

        if (pd != nullptr)
        {
            pd->addMeasurement(queue, ev, kernelName,
                               {global, 0, 0},
                               {local, 1, 1});
        }
        mgclCheckError(clReleaseEvent(ev), "clReleaseEvent");

        if (blocking)
            mgcl::mgclCheckError(clFinish(queue), "clFinish");

        err = clReleaseKernel(kernel);
        mgcl::mgclCheckError(err, "Releasing fill_buffer kernel");
    }

} // namespace mgcl
