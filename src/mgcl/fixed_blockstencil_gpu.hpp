#ifndef MGCL_FIXED_BLOCKSTENCIL_GPU_HPP
#define MGCL_FIXED_BLOCKSTENCIL_GPU_HPP

#include "buffer_gpu.hpp"
#include "fixed_blockstencil.hpp"
#include "hypercube.hpp"
#include "kernel_config.hpp"
#include "mpi_level_data.hpp"
#include "profiling_data.hpp"
#include <CL/cl.h>

namespace mgcl
{
    /**
     * @brief Class for storing a blockstencil of varying width and blocksize. Each coefficient is a matrix of size
     * blocksize x blocksize. The stencil has the same width x width x width coefficients for all grid points.
     *
     * Memory layout is [bi][bj][ci][cj][ck], where
     * - bi and bj are matrix indices and
     * - ci, cj and ck are coefficient indices
     *
     */
    class FixedBlockstencilGpu
    {
    private:
        int width;
        int blocksize;
        BufferGpu buf;

    public:
        FixedBlockstencilGpu(int width_, int blocksize_, cl_context context);
        FixedBlockstencilGpu(FixedBlockstencil& fs, cl_context context, cl_command_queue queue);
        FixedBlockstencilGpu(const FixedBlockstencilGpu& s) = delete;
        FixedBlockstencilGpu(FixedBlockstencilGpu&& s) = delete;
        FixedBlockstencilGpu& operator=(const FixedBlockstencilGpu& s) = delete;
        FixedBlockstencilGpu& operator=(FixedBlockstencilGpu&& s) = delete;
        // ~FixedBlockstencilGpu();

        void fill(double value, cl_command_queue queue, cl_program program, bool blocking, mgcl::conf::KernelConfig* conf = nullptr, mgcl::ProfilingData* pd = nullptr);
        void fill(FixedBlockstencil& f, cl_command_queue queue, bool blocking);
        FixedBlockstencil read(cl_command_queue queue, bool blocking);

        bool isEqual(cl_command_queue queue, FixedBlockstencil& bs, double tol = 1e-7);
        void dumpToFile(cl_command_queue commands, std::string path, bool realCellsOnly);

        inline cl_context getContext() { return buf.getContext(); }
        inline int getBlocksize() { return blocksize; }
        inline int getWidth() const { return width; }
        inline int getSize() const { return buf.getSize(); }
        inline BufferGpu& getBuf() { return buf; }
    };

}

#endif // MGCL_FIXED_BLOCKSTENCIL_GPU_HPP
