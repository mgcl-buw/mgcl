
#include "fixed_blockstencil_gpu.hpp"
#include "buffer_gpu.hpp"
#include "fixed_blockstencil.hpp"
#include <CL/cl.h>
#include <cassert>

namespace mgcl
{
    FixedBlockstencilGpu::FixedBlockstencilGpu(int width_, int blocksize_, cl_context context)
        : width(width_), blocksize(blocksize_), buf(BufferGpu(context, CL_MEM_READ_WRITE, width_ * width_ * width_ * blocksize_ * blocksize_))
    {
    }

    FixedBlockstencilGpu::FixedBlockstencilGpu(FixedBlockstencil& fs, cl_context context, cl_command_queue queue)
        : width(fs.getWidth()), blocksize(fs.getBlocksize()), buf(BufferGpu(context, CL_MEM_READ_WRITE, fs.getSize()))
    {
        fill(fs, queue, true);
    }

    void FixedBlockstencilGpu::fill(double value, cl_command_queue queue, cl_program program, bool blocking, mgcl::conf::KernelConfig* conf, mgcl::ProfilingData* pd)
    {
        buf.fill(program, queue, value, blocking, conf, pd);
    }

    void FixedBlockstencilGpu::fill(FixedBlockstencil& f, cl_command_queue queue, bool blocking)
    {
        buf.write(queue, f.field1d(), blocking);
    }

    FixedBlockstencil FixedBlockstencilGpu::read(cl_command_queue queue, bool blocking)
    {
        FixedBlockstencil ret(width, blocksize);
        assert(ret.getSize() == buf.getSize() && "FixedBlockstencilGpu::read(): FixedBlockstencil and GPU buffer differ in size!");

        auto tmp = buf.read(queue, nullptr, blocking);

        for (size_t i = 0; i < tmp->size(); i++)
        {
            ret.field1d()[i] = (*tmp)[i];
        }

        return ret;
    }

    bool FixedBlockstencilGpu::isEqual(cl_command_queue queue, FixedBlockstencil& bs, double tol)
    {
        auto tmp = read(queue, true);
        return bs.isEqual(tmp, tol);
    }

    void FixedBlockstencilGpu::dumpToFile(cl_command_queue commands, std::string path, bool realCellsOnly)
    {
        auto tmp = read(commands, true);
        tmp.dumpToFile(path, realCellsOnly);
    }
}
