#include "blockstencil_gpu.hpp"
#include "blockstencil.hpp"
#include "mgcl.hpp"
#include "mpi_util.hpp"
#include "opencl_helper.hpp"
#include "util.hpp"
#include <CL/cl.h>
#include <cassert>
#include <iostream>
#include <utility>

namespace mgcl
{

    BlockstencilGpu::BlockstencilGpu(int m_, int n_, int o_, int width_, int blocksize_, int gh_,
                                     cl_context context, cl_command_queue queue, cl_program program)
        : m(m_), n(n_), o(o_), width(width_), blocksize(blocksize_), gh(gh_)
    {
        int err;
        buf = clCreateBuffer(context, CL_MEM_READ_WRITE,
                             sizeof(double) * getSize(),
                             NULL, &err);
        mgclCheckError(err, "clCreateBuffer");

        mgcl::util::fill(program, queue, buf, 0,
                         getSize(),
                         false, nullptr, nullptr);
    }

    BlockstencilGpu::BlockstencilGpu(Blockstencil& bs, cl_context context, cl_command_queue queue, cl_program program)
        : BlockstencilGpu(bs.getM(), bs.getN(), bs.getO(), bs.getWidth(), bs.getBlocksize(), bs.getGhostsM(), context, queue, program)
    {
        write(queue, true, bs);
    }

    BlockstencilGpu::BlockstencilGpu(BlockstencilGpu&& s)
        : m(std::exchange(s.m, 0)),
          n(std::exchange(s.n, 0)),
          o(std::exchange(s.o, 0)),
          width(std::exchange(s.width, 0)),
          gh(std::exchange(s.gh, 0)),
          blocksize(std::exchange(s.blocksize, 0)),
          buf(s.buf) // don't set buf to nullptr since it gets released in dtor
    {
        // retain buffers (i.e. increase internal reference count so they won't be released by accident in dtor)
        if (buf)
        {
            int err = clRetainMemObject(buf);
            mgclCheckError(err, "clRetainMemObject(buf)");
        }
    }

    BlockstencilGpu& BlockstencilGpu::operator=(BlockstencilGpu&& s)
    {
        m = std::exchange(s.m, 0);
        n = std::exchange(s.n, 0);
        o = std::exchange(s.o, 0);
        width = std::exchange(s.width, 0);
        gh = std::exchange(s.gh, 0);
        blocksize = std::exchange(s.blocksize, 0);
        buf = s.buf;

        // retain buffers (i.e. increase internal reference count so they won't be released by accident in dtor)
        if (buf)
        {
            int err = clRetainMemObject(buf);
            mgclCheckError(err, "clRetainMemObject(buf)");
        }

        return *this;
    }

    BlockstencilGpu::~BlockstencilGpu()
    {
        if (buf)
        {
            int err = clReleaseMemObject(buf);
            mgclCheckError(err, "clReleaseMemObject");
        }
    }

    /**
     * Fills the gpu buffer with values from a Blockstencil.
     */
    void BlockstencilGpu::fill(Blockstencil& f, cl_command_queue queue, bool blocking)
    {
        if (f.getWidth() != width)
            error("Widths are not equal!");

        if (f.getBlocksize() != blocksize)
            error("Blocksizes are not equal!");

        if (m != f.getM() || n != f.getN() || o != f.getO() ||
            gh != f.getGhostsM() || gh != f.getGhostsN() || gh != f.getGhostsO())
        {
            error("BlockstencilGpu::fill: Dimensions are not equal. this.m,n,o = " +
                      std::to_string(m) + "," + std::to_string(n) + "," + std::to_string(o) +
                      ", f.m,n,o = " + std::to_string(f.getM()) + "," + std::to_string(f.getN()) +
                      "," + std::to_string(f.getO()););
        }

        int err = clEnqueueWriteBuffer(queue, buf, blocking ? CL_TRUE : CL_FALSE, 0,
                                       sizeof(double) * getSize(),
                                       f.field1d().data(), 0, NULL, NULL);
        mgclCheckError(err, "clEnqueueWriteBuffer");
    }

    /**
     * Reads the gpu buffer into a new Blockstencil. The template parameter N must match the width of the gpu
     * stencil.
     */
    Blockstencil BlockstencilGpu::read(cl_command_queue queue, bool blocking)
    {
        Blockstencil ret(m, n, o, width, blocksize, gh, gh, gh);
        int err = clEnqueueReadBuffer(queue, buf, blocking ? CL_TRUE : CL_FALSE, 0,
                                      sizeof(double) * getSize(),
                                      ret.field1d().data(), 0, NULL, NULL);
        mgclCheckError(err, "clEnqueueReadBuffer");

        return ret;
    }

    std::unique_ptr<Blockstencil> BlockstencilGpu::read_ptr(cl_command_queue queue, bool blocking)
    {
        return std::make_unique<Blockstencil>(read(queue, blocking));
    }

    /**
     * Reads the gpu buffer into the supplied Blockstencil h_stencil.
     */
    void BlockstencilGpu::read(cl_command_queue queue, bool blocking, Blockstencil& h_stencil)
    {
        if (h_stencil.field1d().size() != static_cast<size_t>(getSize()))
            error("Size of h_stencil does not match gpu stencil size");

        int err = clEnqueueReadBuffer(queue, buf, blocking ? CL_TRUE : CL_FALSE, 0,
                                      h_stencil.field1d().size() * sizeof(double),
                                      h_stencil.field1d().data(), 0, NULL, NULL);
        mgclCheckError(err, "clEnqueueReadBuffer");
    }

    /**
     * Writes to the gpu buffer from the supplied Blockstencil h_stencil.
     */
    void BlockstencilGpu::write(cl_command_queue queue, bool blocking, Blockstencil& h_stencil)
    {
        if (h_stencil.field1d().size() != static_cast<size_t>(getSize()))
            error("Size of h_stencil does not match gpu stencil size");

        int err = clEnqueueWriteBuffer(queue, buf, blocking ? CL_TRUE : CL_FALSE, 0,
                                       h_stencil.field1d().size() * sizeof(double),
                                       h_stencil.field1d().data(), 0, NULL, NULL);
        mgclCheckError(err, "clEnqueueWriteBuffer");
    }

    /**
     * @brief Reads device buffer into temporary host buffer and returns true, if real grid points'
     * content is equal to other Blockstencil.
     *
     * @param queue
     * @param bs
     * @param tol
     * @return true
     * @return false
     */
    bool BlockstencilGpu::isEqual(cl_command_queue queue, Blockstencil& bs, double tol)
    {
        auto tmp = this->read(queue, true);
        return tmp.isEqual(bs, tol);
    }

    /**
     * @brief Reads device buffer into temporary host buffer and returns true, if all grid points'
     * content is equal to other Blockstencil.
     *
     * @param queue
     * @param bs
     * @param tol
     * @return true
     * @return false
     */
    bool BlockstencilGpu::isEqualIncGhosts(cl_command_queue queue, Blockstencil& bs, double tol)
    {
        auto tmp = this->read(queue, true);
        return tmp.isEqualIncGhosts(bs, tol);
    }

    /**
     * @brief Updates ghost cells, respects periodic ghosts, i.e. when gh > m
     *
     * @param program
     * @param queue
     * @param conf Kernel Config, i.e. determines the work-group size. If null, a default value is used.
     */
    void BlockstencilGpu::updateGhostsLocally(
        cl_program program, cl_command_queue queue,
        conf::KernelConfig* conf, ProfilingData* pd)
    {
        int err;

        // Create the compute kernel from the program
        const char* kernelName = "update_ghosts_blockstencil";
        cl_kernel kernel = clCreateKernel(program, kernelName, &err);
        mgclCheckError(err, "clCreateKernel");

        // assign kernel arguments
        int pos = 0;
        err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &buf);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &m);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &n);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &o);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &width);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &gh);
        mgclCheckError(err, "Setting kernel arguments");

        // one work-item per ghost cell (excluding real cells). Pad global sizes to fit to local sizes
        int mgh = m + 2 * gh;
        int ngh = n + 2 * gh;
        int ogh = o + 2 * gh;
        size_t global[3] = {static_cast<size_t>(mgh), static_cast<size_t>(ngh), static_cast<size_t>(ogh)};
        size_t local[3] = {
            static_cast<size_t>(mgh > 4 ? 4 : mgh),
            static_cast<size_t>(ngh > 4 ? 4 : ngh),
            static_cast<size_t>(ogh > 4 ? 4 : ogh)};

        // Apply kernel config, if available
        if (conf)
        {
            const auto& c = conf::getWorkGroupSizeForKernelAndWiCount(*conf, kernelName, mgh * ngh * ogh);
            local[0] = static_cast<size_t>(mgh > c[0] ? c[0] : mgh);
            local[1] = static_cast<size_t>(ngh > c[1] ? c[1] : ngh);
            local[2] = static_cast<size_t>(ogh > c[2] ? c[2] : ogh);
        }

        for (int i = 0; i < 3; i++)
            if (global[i] % local[i] != 0)
                global[i] += local[i] - (global[i] % local[i]);

        cl_event ev;

        // enqueue kernel
        err = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global, local, 0, NULL, &ev);
        mgclCheckError(err, "Enqueueing kernel update_ghosts_blockstencil");

        if (pd != nullptr)
        {
            pd->addMeasurement(queue, ev, kernelName,
                               {global[0], global[1], global[2]},
                               {local[0], local[1], local[2]});
        }
        mgclCheckError(clReleaseEvent(ev), "clReleaseEvent");

        err = clReleaseKernel(kernel);
        mgclCheckError(err, "Releasing kernel update_ghosts_blockstencil");
    }

    /**
     * @brief Updates ghosts of an OpenCL buffer respecting MPI usage. That is, the buffer is sent to host, ghosts
     * are updated using MPI routines, and the updated buffer is sent back to the device.
     * Waits for previous commands to finish before reading the buffer.
     *
     * @param program OpenCL program
     * @param commands OpenCL command queue
     * @param dPlanesBuf Device buffer for storing planes. Must be at least of size
     *   (2 * yz * getGhostsM() + 2 * xz * getGhostsN() + 2 * xy * getGhostsO()) * blocksize^2 * stencilWidth^3
     * @param hPlanesBufSend Host buffer for sending planes.
     * @param hPlanesBufRecv Host buffer for receiving planes.
     * @param mpiData Contains MPI topology data for the current level.
     * @param forceLocal If true, ghosts will be updated locally without MPI routines.
     * @param conf OpenCL Kernel launch config
     * @param pd OpenCL profiling data
     */
    void BlockstencilGpu::updateGhostsOclMpi(cl_program program, cl_command_queue commands,
                                             BufferGpu& dPlanesBuf,
                                             std::vector<double>& hPlanesBufSend, std::vector<double>& hPlanesBufRecv,
                                             MPILevelData& mpiData, bool forceLocal, bool periodic,
                                             conf::KernelConfig* conf, mgcl::ProfilingData* pd)
    {
        // do nothing if single-gpu and Dirichlet bc's
        if (!periodic && mpiData.mpiSize() == 1)
            return;

        if (forceLocal)
        {
            updateGhostsLocally(program, commands, conf, pd);
            return;
        }

        // Use temporary buffer for extracting and pasting planes. Check if it's large enough beforehand.
        // TODO maybe disable check in UNSAFE mode
        int yz = getNgh() * getOgh();
        int xz = getMgh() * getOgh();
        int xy = getMgh() * getNgh();
        int ressize = (2 * yz * getGh() + 2 * xz * getGh() + 2 * xy * getGh()) * blocksize * blocksize * width * width * width;

        if (dPlanesBuf.getSize() < ressize)
            error("BlockstencilGpu::updateGhostsOclMpi: dPlanesBuf is too small. Need at least " + std::to_string(ressize) + ", but is " + std::to_string(dPlanesBuf.getSize()));

        if (hPlanesBufSend.size() < ressize || hPlanesBufRecv.size() < ressize)
            throw "BlockstencilGpu::updateGhostsOclMpi: hPlanesBufSend or hPlanesBufRecv is too small. Need at least " +
                std::to_string(ressize) + ", but is " + std::to_string(hPlanesBufSend.size()) +
                " (send) and " + std::to_string(hPlanesBufRecv.size()) + " (recv)";

        // Extract border planes from the buffer
        extractBorderPlanes(commands, program,
                            dPlanesBuf, hPlanesBufSend,
                            conf, pd);

        // Send our planes to neighbours and receive their planes
        mpi_util::sendBorderPlanesBlockstencil(getMgh(), getNgh(), getOgh(),
                                               getGh(), getGh(), getGh(), width, blocksize,
                                               hPlanesBufSend, hPlanesBufRecv, mpiData);

        // Paste planes back into the buffer.
        dPlanesBuf.write(commands, hPlanesBufRecv, false, ressize);
        pasteGhostsFromBorderPlanes(commands, program,
                                    dPlanesBuf,
                                    conf, pd);
    }

    /**
     * @brief Extracts the border planes of the blockstencil.
     *
     * @param commands OpenCL command queue
     * @param program OpenCL program
     * @param d_target BlockstencilGpu that data gets extracted into.
     * @param h_target std::vector<double> that data gets extracted into.
     */
    void BlockstencilGpu::extractBorderPlanes(cl_command_queue commands, cl_program program,
                                              BufferGpu& d_target, std::vector<double>& h_target,
                                              mgcl::conf::KernelConfig* conf, mgcl::ProfilingData* pd)
    {
        // Plane sizes
        int yz = getNgh() * getOgh();
        int xz = getMgh() * getOgh();
        int xy = getMgh() * getNgh();
        int ghosts_m = getGh();
        int ghosts_n = getGh();
        int ghosts_o = getGh();
        int wicount = (2 * yz * ghosts_m + 2 * xz * ghosts_n + 2 * xy * ghosts_o) * 27;

        if (ghosts_m > m || ghosts_n > n || ghosts_o > o)
            error("BlockstencilGpu::extractBorderPlanes: Only defined for ghosts <= m, n, o");

        int err;

        // Create the compute kernel from the program
        const char* kernelName = "extract_border_planes_blockstencil";
        cl_kernel kernel = clCreateKernel(program, kernelName, &err);
        mgcl::mgclCheckError(err, "clCreateKernel");

        int mgh = getMgh();
        int ngh = getNgh();
        int ogh = getOgh();

        // assign kernel arguments
        cl_mem d_target_buffer = d_target.getBuf();
        int pos = 0;
        err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &buf);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &d_target_buffer);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &m);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &n);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &o);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &mgh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ngh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ogh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghosts_m);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghosts_n);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghosts_o);
        mgcl::mgclCheckError(err, "Setting kernel arguments");

        // one work-item per ghost cell (excluding real cells). Pad global sizes to fit to local sizes
        size_t global = wicount;
        size_t local = 32;
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
        mgcl::mgclCheckError(err, "Enqueueing extract_border_planes_blockstencil kernel");

        if (pd != nullptr)
        {
            pd->addMeasurement(commands, ev, kernelName,
                               {global, 0, 0},
                               {local, 1, 1});
        }
        mgclCheckError(clReleaseEvent(ev), "clReleaseEvent");

        err = clReleaseKernel(kernel);
        mgcl::mgclCheckError(err, "Releasing extract_border_planes_blockstencil kernel");

        // Read into h_target
        d_target.read(commands, h_target.data(), true);
    }

    /**
     * @brief Pastes ghost data into the border planes of the varying stencil.
     *
     * @param commands OpenCL command queue
     * @param program OpenCL program
     * @param d_ghosts BufferGpu containing ghost data from neighboring processes
     */
    void BlockstencilGpu::pasteGhostsFromBorderPlanes(cl_command_queue commands, cl_program program,
                                                      BufferGpu& d_ghosts,
                                                      mgcl::conf::KernelConfig* conf, mgcl::ProfilingData* pd)
    {
        // Plane sizes
        int yz = getNgh() * getOgh();
        int xz = getMgh() * getOgh();
        int xy = getMgh() * getNgh();
        int ghosts_m = getGh();
        int ghosts_n = getGh();
        int ghosts_o = getGh();
        int wicount = (2 * yz * ghosts_m + 2 * xz * ghosts_n + 2 * xy * ghosts_o) * 27;

        if (ghosts_m > m || ghosts_n > n || ghosts_o > o)
            error("BlockstencilGpu::pasteGhostsFromBorderPlanes: Only defined for ghosts <= m, n, o");

        int err;

        // Create the compute kernel from the program
        const char* kernelName = "paste_ghosts_from_border_planes_blockstencil";
        cl_kernel kernel = clCreateKernel(program, kernelName, &err);
        mgcl::mgclCheckError(err, "clCreateKernel");

        int mgh = getMgh();
        int ngh = getNgh();
        int ogh = getOgh();

        // assign kernel arguments
        cl_mem d_ghosts_buffer = d_ghosts.getBuf();
        int pos = 0;
        err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &buf);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &d_ghosts_buffer);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &m);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &n);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &o);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &mgh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ngh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ogh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghosts_m);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghosts_n);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghosts_o);
        mgcl::mgclCheckError(err, "Setting kernel arguments");

        // one work-item per ghost cell (excluding real cells). Pad global sizes to fit to local sizes
        size_t global = wicount;
        size_t local = 32;
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
        mgcl::mgclCheckError(err, "Enqueueing paste_ghosts_from_border_planes_blockstencil kernel");

        if (pd != nullptr)
        {
            pd->addMeasurement(commands, ev, kernelName,
                               {global, 0, 0},
                               {local, 1, 1});
        }
        mgclCheckError(clReleaseEvent(ev), "clReleaseEvent");

        err = clReleaseKernel(kernel);
        mgcl::mgclCheckError(err, "Releasing paste_ghosts_from_border_planes_blockstencil kernel");
    }

    std::unique_ptr<CuboidBSGpu> BlockstencilGpu::invertDiagonal(cl_context context, cl_command_queue queue)
    {
        // TODO create native gpu implementation?
        auto tmp = read(queue, true);
        auto inv = tmp.invertDiagonal();
        if (!inv)
            error("BlockstencilGpu::invertDiagonal: At least one entry on the diagonal is 0.");

        return std::make_unique<CuboidBSGpu>(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, *inv);
    }

    std::unique_ptr<BlockstencilGpu> BlockstencilGpu::invertCenterMatrices(cl_context context, cl_command_queue queue, cl_program program)
    {
        // TODO create native gpu implementation?
        auto tmp = read(queue, true);
        auto inv = tmp.invertCenterMatrices();
        if (!inv)
            error("BlockstencilGpu::invertCenterMatrices: At least one blockstencil is singular.");

        return std::make_unique<BlockstencilGpu>(*inv, context, queue, program);
    }

    std::ostream& operator<<(std::ostream& os, const BlockstencilGpu& v)
    {
        os << "BlockstencilGpu: " << std::endl
           << " m,n,o: " << v.m << "," << v.n << "," << v.o << std::endl
           << " width: " << v.width << std::endl
           << " blocksize: " << v.blocksize << std::endl
           << " gh: " << v.gh << std::endl
           << " buf: " << v.buf << std::endl;
        return os;
    }

    void BlockstencilGpu::dumpToFile(cl_command_queue commands, std::string path, bool realCellsOnly)
    {

        auto tmp = read(commands, true);
        tmp.dumpToFile(path, realCellsOnly);
    }

}