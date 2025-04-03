#ifndef MGCL_BLOCKSTENCIL_HPP
#define MGCL_BLOCKSTENCIL_HPP

#include "buffer_gpu.hpp"
#include "kernel_config.hpp"
#include "profiling_data.hpp"
#include <ostream>

#ifndef CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#endif // CL_USE_DEPRECATED_OPENCL_1_2_APIS

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif // CL_TARGET_OPENCL_VERSION

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

namespace mgcl
{

    /**
     * @brief Wrapper class for a varying stencil gpu buffer. Stores additional information like width of the grid,
     * width of the stencil and amount of ghost cells. The device buffer gets automatically created in the constructor
     * and gets released in the destructor.
     *
     */
    class VaryingStencilGpu
    {
    private:
        int m;
        int n;
        int o;
        int width;
        int gh;
        cl_mem buf = nullptr;

    public:
        VaryingStencilGpu(int m_, int n_, int o_, int width_, int gh_,
                          cl_context context, cl_command_queue queue, cl_program program);
        VaryingStencilGpu(VaryingStencilGpu&&);
        VaryingStencilGpu& operator=(VaryingStencilGpu&&);
        ~VaryingStencilGpu();

        VaryingStencilGpu(const VaryingStencilGpu&) = delete;
        VaryingStencilGpu& operator=(const VaryingStencilGpu&) = delete;

        void fill(VaryingStencil& f, cl_command_queue queue, bool blocking);
        VaryingStencil read(cl_command_queue queue, bool blocking);
        void read(cl_command_queue queue, bool blocking, VaryingStencil& h_stencil);
        void write(cl_command_queue queue, bool blocking, VaryingStencil& h_stencil);

        void updateGhosts(
            cl_program program, cl_command_queue queue,
            conf::KernelConfig* conf, ProfilingData* pd);

        VaryingStencilGpu multiply(
            FixedStencilGpu& b, int ghc,
            BufferGpu* d_planes_buf,
            std::vector<double>* sbuf, std::vector<double>* rbuf,
            cl_program program, cl_command_queue queue, cl_context context,
            MPILevelData* mpiData, bool forceLocal,
            conf::KernelConfig* conf, ProfilingData* pd);

        VaryingStencilGpu multiply(
            VaryingStencilGpu& b, int ghc,
            BufferGpu* d_planes_buf,
            std::vector<double>* sbuf, std::vector<double>* rbuf,
            cl_program program, cl_command_queue queue, cl_context context,
            MPILevelData* mpiData, bool forceLocal,
            conf::KernelConfig* conf, ProfilingData* pd);

        VaryingStencilGpu cutFromW7ToW3(
            cl_program program, cl_command_queue queue, cl_context context,
            int ghout, mgcl::conf::KernelConfig* conf, ProfilingData* pd,
            int resm = 0, int resn = 0, int reso = 0);

        void extractBorderPlanes(cl_command_queue commands, cl_program program,
                                 BufferGpu& d_target, std::vector<double>& h_target,
                                 mgcl::conf::KernelConfig* conf, mgcl::ProfilingData* pd);
        void pasteGhostsFromBorderPlanes(cl_command_queue commands, cl_program program,
                                         BufferGpu& d_ghosts,
                                         mgcl::conf::KernelConfig* conf, mgcl::ProfilingData* pd);

        int getM() const;
        int getN() const;
        int getO() const;
        int getMgh() const;
        int getNgh() const;
        int getOgh() const;
        int getWidth() const;
        int getGh() const;
        cl_mem getBuf() const;

        friend std::ostream& operator<<(std::ostream& os, const VaryingStencilGpu& v);
    };
}

#endif // MGCL_BLOCKSTENCIL_HPP
