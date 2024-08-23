#ifndef MGCL_BUFFER_GPU_HPP
#define MGCL_BUFFER_GPU_HPP

#include "kernel_config.hpp"
#include "profiling_data.hpp"

#include <cstddef>
#include <memory>
#include <vector>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

namespace mgcl
{
    /**
     * @brief Simple class for a 1d OpenCL buffer holding double values.
     *
     */
    class BufferGpu
    {
    private:
        size_t _size;
        cl_mem buf;
        cl_context context;

    public:
        BufferGpu(cl_context context, cl_mem_flags flags, size_t size);
        BufferGpu(cl_context context, cl_mem_flags flags, std::vector<double>& h_data);

        BufferGpu(const BufferGpu& s) = delete;
        BufferGpu(BufferGpu&& s) = delete;
        BufferGpu& operator=(const BufferGpu& s) = delete;
        BufferGpu& operator=(BufferGpu&& s) = delete;
        ~BufferGpu();

        // TODO return type?
        std::unique_ptr<std::vector<double>> read(cl_command_queue queue, double* h_target, bool blocking) const;
        std::unique_ptr<std::vector<double>> read(cl_command_queue queue, double* h_target, bool blocking, size_t _size) const;
        void write(cl_command_queue queue, const std::vector<double>& host_data, bool blocking);
        void write(cl_command_queue queue, const std::vector<double>& host_data, bool blocking, size_t _size);
        void fill(cl_program program, cl_command_queue queue, double value, bool blocking,
                  conf::KernelConfig* conf, mgcl::ProfilingData* pd);

        inline size_t getSize() const { return _size; }
        inline cl_mem getBuf() const { return buf; }
        inline cl_context getContext() const { return context; }
    };
}
#endif // MGCL_BUFFER_GPU_HPP