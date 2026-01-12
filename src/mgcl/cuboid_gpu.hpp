#ifndef MGCL_CUBOID_GPU_HPP
#define MGCL_CUBOID_GPU_HPP

#include "buffer_gpu.hpp"
#include "profiling_data.hpp"
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <memory>

#include "cuboid.hpp"
#include "kernel_config.hpp"

namespace mgcl
{
    /**
     * @brief This class is a wrapper for a 3d OpenCL buffer, following RAII.
     *
     */
    class CuboidGpu
    {
    private:
        cl_mem buffer;
        cl_context context;
        cl_mem_flags flags;
        int m;
        int n;
        int o;
        int ghosts_m;
        int ghosts_n;
        int ghosts_o;
        int mgh;
        int ngh;
        int ogh;
        int size;

    public:
        CuboidGpu(cl_context context, cl_mem_flags flags,
                  int m, int n, int o, int ghosts_m, int ghosts_n, int ghosts_o,
                  const cl_mem buf = nullptr);
        CuboidGpu(cl_context context, cl_mem_flags flags, const Cuboid& host_data);

        CuboidGpu(const CuboidGpu&) = delete;
        CuboidGpu(CuboidGpu&&) = delete;
        CuboidGpu& operator=(const CuboidGpu&) = delete;
        CuboidGpu& operator=(CuboidGpu&&) = delete;
        ~CuboidGpu();

        std::unique_ptr<Cuboid> read(cl_command_queue commands, Cuboid* const host_ptr, bool blocking) const;
        std::unique_ptr<Cuboid> read1d(cl_command_queue commands, int size, Cuboid* const host_ptr, bool blocking) const;
        void write(cl_command_queue commands, const Cuboid& host_data, bool blocking);
        void write1d(cl_command_queue commands, int _size, const Cuboid& host_data, bool blocking);
        void fill(cl_program program, cl_command_queue commands,
                  double value, bool blocking,
                  conf::KernelConfig* conf, mgcl::ProfilingData* pd);
        void fill1dIndex(cl_program program, cl_command_queue commands,
                         bool blocking, bool realCellsOnly,
                         conf::KernelConfig* conf, mgcl::ProfilingData* pd);

        void retain();
        int refCount();
        void copyTo(cl_command_queue commands, CuboidGpu& target);

        std::unique_ptr<CuboidGpu> copyShallow();

        std::unique_ptr<std::vector<double>> extractBorderPlanes(cl_command_queue commands, cl_program program,
                                                                 BufferGpu* d_target, std::vector<double>* h_target,
                                                                 mgcl::conf::KernelConfig* conf, mgcl::ProfilingData* pd,
                                                                 bool readResult);
        void pasteGhostsFromBorderPlanes(cl_context context, cl_command_queue commands, cl_program program,
                                         BufferGpu* d_source, std::vector<double>* h_source,
                                         mgcl::conf::KernelConfig* conf, mgcl::ProfilingData* pd);

        void dumpToFile(cl_command_queue commands, const std::string& path, bool realCellsOnly = false) const;

        double l2norm(cl_program program, cl_command_queue commands,
                      CuboidGpu* tmpSquared,
                      mgcl::conf::KernelConfig* conf, mgcl::ProfilingData* pd);

        cl_mem getBuffer() const;
        cl_mem_flags getFlags() const { return flags; }
        int getM() const;
        int getN() const;
        int getO() const;
        int getGhostsM() const;
        int getGhostsN() const;
        int getGhostsO() const;
        int getMgh() const;
        int getNgh() const;
        int getOgh() const;
        int getSize() const;
        cl_context getContext() const;

        static void swap(CuboidGpu& a, CuboidGpu& b);
    };
}

#endif // MGCL_CUBOID_GPU_HPP