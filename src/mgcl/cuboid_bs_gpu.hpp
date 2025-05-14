#ifndef MGCL_CUBOID_BS_GPU_HPP
#define MGCL_CUBOID_BS_GPU_HPP

#include "buffer_gpu.hpp"
#include "profiling_data.hpp"
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <memory>

#include "cuboid_bs.hpp"
#include "kernel_config.hpp"

namespace mgcl
{
    /**
     * @brief This class is a wrapper for a 3d OpenCL buffer, following RAII.
     *
     */
    class CuboidBSGpu
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
        int blocksize;

    public:
        CuboidBSGpu(cl_context context, cl_mem_flags flags,
                    int m, int n, int o, int ghosts_m, int ghosts_n, int ghosts_o, int blocksize,
                    const cl_mem buf = nullptr);
        CuboidBSGpu(cl_context context, cl_mem_flags flags, const CuboidBS& host_data);

        CuboidBSGpu(const CuboidBSGpu&) = delete;
        CuboidBSGpu(CuboidBSGpu&&) = delete;
        CuboidBSGpu& operator=(const CuboidBSGpu&) = delete;
        CuboidBSGpu& operator=(CuboidBSGpu&&) = delete;
        ~CuboidBSGpu();

        std::unique_ptr<CuboidBS> read(cl_command_queue commands, CuboidBS* const host_ptr, bool blocking) const;
        std::unique_ptr<CuboidBS> read1d(cl_command_queue commands, int size, CuboidBS* const host_ptr, bool blocking) const;
        void write(cl_command_queue commands, CuboidBS& host_data, bool blocking);
        void write1d(cl_command_queue commands, int _size, CuboidBS& host_data, bool blocking);
        void fill(cl_program program, cl_command_queue commands,
                  double value, bool blocking,
                  conf::KernelConfig* conf, mgcl::ProfilingData* pd);

        void retain();
        int refCount();
        void copyTo(cl_command_queue commands, CuboidBSGpu& target);

        std::unique_ptr<CuboidBSGpu> copyShallow();

        void updateGhostsLocally(cl_program program, cl_command_queue commands,
                                 conf::KernelConfig* conf, mgcl::ProfilingData* pd);
        void updateGhostsOclMpi(cl_program program, cl_command_queue commands,
                                BufferGpu* dPlanesBuf,
                                std::vector<double>* hPlanesBufSend, std::vector<double>* hPlanesBufRecv,
                                MPILevelData* mpiData, bool forceLocal,
                                conf::KernelConfig* conf, mgcl::ProfilingData* pd);
        std::unique_ptr<std::vector<double>> extractBorderPlanes(cl_command_queue commands, cl_program program,
                                                                 BufferGpu* d_target, std::vector<double>* h_target,
                                                                 mgcl::conf::KernelConfig* conf, mgcl::ProfilingData* pd);
        void pasteGhostsFromBorderPlanes(cl_context context, cl_command_queue commands, cl_program program,
                                         BufferGpu* d_source, std::vector<double>* h_source,
                                         mgcl::conf::KernelConfig* conf, mgcl::ProfilingData* pd);

        void dumpToFile(cl_command_queue commands, const std::string& path, bool realCellsOnly = false) const;

        inline cl_mem getBuffer() const { return buffer; }
        inline cl_mem_flags getFlags() const { return flags; }
        inline int getM() const { return m; }
        inline int getN() const { return n; }
        inline int getO() const { return o; }
        inline int getGhostsM() const { return ghosts_m; }
        inline int getGhostsN() const { return ghosts_n; }
        inline int getGhostsO() const { return ghosts_o; }
        inline int getMgh() const { return mgh; }
        inline int getNgh() const { return ngh; }
        inline int getOgh() const { return ogh; }
        inline int getSize() const { return size; }
        inline cl_context getContext() const { return context; }
        inline int getBlocksize() const { return blocksize; }

        static void swap(CuboidBSGpu& a, CuboidBSGpu& b);
    };
}

#endif // MGCL_CUBOID_BS_GPU_HPP