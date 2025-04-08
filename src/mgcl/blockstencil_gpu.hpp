#ifndef MGCL_BLOCKSTENCIL_GPU_HPP
#define MGCL_BLOCKSTENCIL_GPU_HPP

#include "blockstencil.hpp"
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
     * @brief Wrapper class for blockstencil gpu buffer. Stores additional information like width of the grid,
     * width of the stencil and amount of ghost cells. The device buffer gets automatically created in the constructor
     * and gets released in the destructor.
     *
     * Memory layout is [bi][bj][ci][cj][ck][x][y][z], where
     * - bi and bj are matrix indices,
     * - ci, cj and ck are coefficient indices and
     * - x, y and z are grid point indices
     *
     *
     */
    class BlockstencilGpu
    {
    private:
        int m;
        int n;
        int o;
        int width;
        int blocksize;
        int gh;
        cl_mem buf = nullptr;

    public:
        BlockstencilGpu(int m_, int n_, int o_, int width_, int blocksize_, int gh_,
                        cl_context context, cl_command_queue queue, cl_program program);
        BlockstencilGpu(Blockstencil& bs, cl_context context, cl_command_queue queue, cl_program program);
        BlockstencilGpu(BlockstencilGpu&&);
        BlockstencilGpu& operator=(BlockstencilGpu&&);
        ~BlockstencilGpu();

        BlockstencilGpu(const BlockstencilGpu&) = delete;
        BlockstencilGpu& operator=(const BlockstencilGpu&) = delete;

        void fill(Blockstencil& f, cl_command_queue queue, bool blocking);
        Blockstencil read(cl_command_queue queue, bool blocking);
        void read(cl_command_queue queue, bool blocking, Blockstencil& h_stencil);
        void write(cl_command_queue queue, bool blocking, Blockstencil& h_stencil);

        bool isEqual(cl_command_queue queue, Blockstencil& bs, double tol = 1e-7);
        bool isEqualIncGhosts(cl_command_queue queue, Blockstencil& bs, double tol = 1e-7);

        void updateGhosts(
            cl_program program, cl_command_queue queue,
            conf::KernelConfig* conf, ProfilingData* pd);

        void extractBorderPlanes(cl_command_queue commands, cl_program program,
                                 BufferGpu& d_target, std::vector<double>& h_target,
                                 mgcl::conf::KernelConfig* conf, mgcl::ProfilingData* pd);
        void pasteGhostsFromBorderPlanes(cl_command_queue commands, cl_program program,
                                         BufferGpu& d_ghosts,
                                         mgcl::conf::KernelConfig* conf, mgcl::ProfilingData* pd);

        void dumpToFile(cl_command_queue commands, std::string path, bool realCellsOnly);

        inline int getM() const { return m; }
        inline int getN() const { return n; }
        inline int getO() const { return o; }
        inline int getMgh() const { return m + 2 * gh; }
        inline int getNgh() const { return n + 2 * gh; }
        inline int getOgh() const { return o + 2 * gh; }
        inline int getWidth() const { return width; }
        inline int getBlocksize() const { return blocksize; }
        inline int getGh() const { return gh; }
        inline int getSize() const { return blocksize * blocksize * (m + 2 * gh) * (n + 2 * gh) * (o + 2 * gh) * width * width * width; }
        inline cl_mem getBuf() const { return buf; }

        friend std::ostream& operator<<(std::ostream& os, const BlockstencilGpu& v);
    };
}

#endif // MGCL_BLOCKSTENCIL_GPU_HPP
