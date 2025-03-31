#include "cuboid_bs_gpu.hpp"

#include "kernel_config.hpp"
#include "opencl_helper.hpp"
#include "profiling_data.hpp"

#include "mgcl.hpp"

#include <cassert>
#include <fstream>
#include <iomanip>
#include <memory>
#include <vector>

namespace mgcl
{
    /**
     * @brief Construct a new CuboidBSGpu object with given sizes. A CuboidBSGpu object is only valid in its context.
     *   flags must be valid, see parameter specification.
     *
     * @param context OpenCL context this buffer is valid in
     * @param flags Must contain one of CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY or CL_MEM_READ_ONLY. Furthermore if
     *   host_ptr is given, it must contain one of CL_MEM_COPY_HOST_PTR, CL_MEM_USE_HOST_PTR or CL_MEM_ALLOC_HOST_PTR.
     *   Example using a gpu with given host_ptr: CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR
     * @param m extend
     * @param n extend
     * @param o extend
     * @param ghosts_m amount of ghost cells at one border
     * @param ghosts_n amount of ghost cells at one border
     * @param ghosts_o amount of ghost cells at one border
     * @param buf buffer that shall be retained. If given, flags will be ignored.
     * @throws string if dimensions are not strictly positive or ghosts are not positive or 0
     * @throws string if flags are not valid
     */
    CuboidBSGpu::CuboidBSGpu(cl_context context, cl_mem_flags flags,
                             int m, int n, int o,
                             int ghosts_m, int ghosts_n, int ghosts_o,
                             int blocksize,
                             const cl_mem buf)
        : context(context), buffer(buf), flags(flags), m(m), n(n), o(o),
          ghosts_m(ghosts_m), ghosts_n(ghosts_n), ghosts_o(ghosts_o),
          mgh(m + 2 * ghosts_m), ngh(n + 2 * ghosts_n), ogh(o + 2 * ghosts_o),
          size(mgh * ngh * ogh * blocksize), blocksize(blocksize)
    {
        if (m <= 0 || n <= 0 || o <= 0 || blocksize <= 0)
            error("m, n, o and blocksize must be > 0.");

        if (ghosts_m < 0 || ghosts_n < 0 || ghosts_o < 0)
            error("ghosts must be >= 0.");

        if (buf)
            retain();
        else
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
                error("host_ptr is null, but flags contains CL_MEM_ALLOC_HOST_PTR, CL_MEM_USE_HOST_PTR or CL_MEM_COPY_HOST_PTR.");

            cl_int err;
            buffer = clCreateBuffer(context, flags, sizeof(double) * size, nullptr, &err);
            mgclCheckError(err, "clCreateBuffer");
        }
    }

    /**
     * @brief Construct a new CuboidBSGpu object with the same size and content as of the given CuboidBS.
     *   A CuboidBSGpu object is only valid in its context. flags must be valid, see parameter specification.
     *
     * @param context OpenCL context this buffer is valid in
     * @param flags Must contain one and only one of CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY or CL_MEM_READ_ONLY as well as
     *   one and only one of CL_MEM_COPY_HOST_PTR, CL_MEM_USE_HOST_PTR or CL_MEM_ALLOC_HOST_PTR.
     *   Example using a gpu with given host_ptr: CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR
     * @param host_data CuboidBS that gets copied from initially.
     * @throws string if flags are not valid
     */
    CuboidBSGpu::CuboidBSGpu(cl_context context, cl_mem_flags flags,
                             const CuboidBS& host_data)
        : context(context), buffer(nullptr), flags(flags), m(host_data.getM()),
          n(host_data.getN()), o(host_data.getO()),
          ghosts_m(host_data.getGhostsM()), ghosts_n(host_data.getGhostsN()), ghosts_o(host_data.getGhostsO()),
          mgh(m + 2 * ghosts_m), ngh(n + 2 * ghosts_n), ogh(o + 2 * ghosts_o),
          blocksize(host_data.getBlocksize()),
          size(mgh * ngh * ogh * host_data.getBlocksize())
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

        // Check that flags contains one of CL_MEM_COPY_HOST_PTR, CL_MEM_USE_HOST_PTR or CL_MEM_ALLOC_HOST_PTR if
        // host_ptr is given.
        if (!(containsAllocHostPtr || containsCopyHostPtr || containsUseHostPtr))
            error("host_ptr not null, but flags does not contain CL_MEM_ALLOC_HOST_PTR, CL_MEM_USE_HOST_PTR or CL_MEM_COPY_HOST_PTR.");

        // Check that flags contains one and only one of CL_MEM_COPY_HOST_PTR, CL_MEM_USE_HOST_PTR or CL_MEM_ALLOC_HOST_PTR.
        if ((containsCopyHostPtr && containsAllocHostPtr) || (containsCopyHostPtr && containsUseHostPtr) || (containsUseHostPtr && containsAllocHostPtr))
            error("flags must contain one and only one of CL_MEM_COPY_HOST_PTR, CL_MEM_ALLOC_HOST_PTR, or CL_MEM_USE_HOST_PTR");

        cl_int err;
        buffer = clCreateBuffer(context, flags, sizeof(double) * size, host_data.getData()[0][0][0], &err);
        mgclCheckError(err, "clCreateBuffer");
    }

    CuboidBSGpu::~CuboidBSGpu()
    {
        if (buffer != nullptr)
            mgclCheckError(clReleaseMemObject(buffer), "clReleaseMemObject");
    }

    /**
     * @brief Reads the buffer into host_ptr, if not null, or a new CuboidBS otherwise. Dimensions of host_ptr must
     *   match.
     *
     * @param host_ptr CuboidBS that data gets read into.
     * @param blocking Blocking read, if true.
     * @return std::unique_ptr<CuboidBS> New CuboidBS that data gets read into, if host_ptr is null. If host_ptr is not
     *   null, a nullptr will be returned.
     * @throws string If Dimensions do not match.
     */
    std::unique_ptr<CuboidBS> CuboidBSGpu::read(cl_command_queue commands, CuboidBS* const host_ptr, bool blocking) const
    {
        if (host_ptr)
        {
            if (host_ptr->getM() != m || host_ptr->getN() != n || host_ptr->getO() != o ||
                host_ptr->getGhostsM() != ghosts_m ||
                host_ptr->getGhostsN() != ghosts_n ||
                host_ptr->getGhostsO() != ghosts_o ||
                host_ptr->getBlocksize() != blocksize)
                error("Dimensions do not match!");
        }

        auto ret = host_ptr ? nullptr : std::make_unique<CuboidBS>(m, n, o, ghosts_m, ghosts_n, ghosts_o, blocksize);

        auto ptr = host_ptr ? host_ptr : ret.get();
        int err = clEnqueueReadBuffer(commands, buffer, blocking ? CL_TRUE : CL_FALSE, 0, sizeof(double) * size,
                                      ptr->getData()[0][0][0], 0, NULL, NULL);
        mgcl::mgclCheckError(err, "clEnqueueReadBuffer");

        return ret;
    }

    /**
     * @brief Reads the buffer into host_ptr, if not null, or a new CuboidBS otherwise. Dimensions of host_ptr must
     *   *not* match, instead size elements are read but which must be less than the target size.
     *   If a new cuboid is created, it has the size 1x1xsize and ghosts=0.
     *
     * @param size Number of elements to read
     * @param host_ptr CuboidBS that data gets read into.
     * @param blocking Blocking read, if true.
     * @return std::unique_ptr<CuboidBS> New CuboidBS that data gets read into, if host_ptr is null. If host_ptr is not
     *   null, a nullptr will be returned.
     * @throws string If Dimensions do not match.
     */
    std::unique_ptr<CuboidBS> CuboidBSGpu::read1d(cl_command_queue commands, int _size, CuboidBS* const host_ptr, bool blocking) const
    {
        if (host_ptr)
        {
            if (host_ptr->getSize() < _size)
                error("CuboidBSGpu::read1d: Target CuboidBS is too small!");
        }

        if (size < _size)
            error("CuboidBSGpu::read1d: Source CuboidBS is smaller than requested read size!");

        auto ret = host_ptr ? nullptr : std::make_unique<CuboidBS>(1, 1, 1, _size);

        auto ptr = host_ptr ? host_ptr : ret.get();
        int err = clEnqueueReadBuffer(commands, buffer, blocking ? CL_TRUE : CL_FALSE, 0, sizeof(double) * _size,
                                      ptr->getData()[0][0][0], 0, NULL, NULL);
        mgcl::mgclCheckError(err, "clEnqueueReadBuffer");

        return ret;
    }

    /**
     * @brief Writes data from host_data into the gpu buffer.
     *
     * @param commands OpenCL command queue
     * @param host_data CuboidBS that the data gets written from
     * @param blocking Blocking Write, if true.
     * @throws string If Dimensions do not match.
     */
    void CuboidBSGpu::write(cl_command_queue commands, const CuboidBS& host_data, bool blocking)
    {
        if (host_data.getM() != m || host_data.getN() != n || host_data.getO() != o ||
            host_data.getGhostsM() != ghosts_m ||
            host_data.getGhostsN() != ghosts_n ||
            host_data.getGhostsO() != ghosts_o ||
            host_data.getBlocksize() != blocksize)
            error("Dimensions do not match!");

        int err = clEnqueueWriteBuffer(commands, buffer, blocking ? CL_TRUE : CL_FALSE, 0, sizeof(double) * size,
                                       host_data.getData()[0][0][0], 0, NULL, NULL);
        mgcl::mgclCheckError(err, "clEnqueueReadBuffer");
    }

    /**
     * @brief Writes data from host_data into the gpu buffer.
     *
     * @param commands OpenCL command queue
     * @param host_data CuboidBS that the data gets written from
     * @param blocking Blocking Write, if true.
     * @throws string If Dimensions do not match.
     */
    void CuboidBSGpu::write1d(cl_command_queue commands, int _size, const CuboidBS& host_data, bool blocking)
    {
        if (host_data.getSize() < _size)
            error("CuboidBSGpu::write1d: Source host cuboid is too small!");

        if (size < _size)
            error("CuboidBSGpu::write1d: Target device cuboid is too small!");

        int err = clEnqueueWriteBuffer(commands, buffer, blocking ? CL_TRUE : CL_FALSE, 0, sizeof(double) * _size,
                                       host_data.getData()[0][0][0], 0, NULL, NULL);
        mgcl::mgclCheckError(err, "clEnqueueReadBuffer");
    }

    /**
     * @brief Fills the buffer with the given value.
     *
     * @param commands OpenCL command queue
     * @param value Value to fill the buffer with.
     * @param blocking If true, the write will be blocking.
     */
    void CuboidBSGpu::fill(cl_program program, cl_command_queue commands,
                           double value, bool blocking,
                           conf::KernelConfig* conf, mgcl::ProfilingData* pd)
    {
        // int err = clEnqueueFillBuffer(commands, buffer, &value, sizeof(double), 0, sizeof(double) * size,
        //                               0, nullptr, nullptr);
        // mgcl::mgclCheckError(err, "clEnqueueFillBuffer");
        // if (blocking)
        //     mgcl::mgclCheckError(clFinish(commands), "clFinish");

        // Create the compute kernel from the program
        int err;
        const char* kernelName = "fill_buffer";
        cl_kernel kernel = clCreateKernel(program, kernelName, &err);
        mgcl::mgclCheckError(err, "clCreateKernel");

        // assign kernel arguments
        int pos = 0;
        err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &buffer);
        err |= clSetKernelArg(kernel, ++pos, sizeof(double), &value);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &size);
        mgcl::mgclCheckError(err, "Setting kernel arguments");

        // one work-item per ghost cell (excluding real cells). Pad global sizes to fit to local sizes
        size_t global = size;
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
        err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, &ev);
        mgcl::mgclCheckError(err, "Enqueueing fill_buffer kernel");

        if (pd != nullptr)
        {
            pd->addMeasurement(commands, ev, kernelName,
                               {global, 0, 0},
                               {local, 1, 1});
        }
        mgclCheckError(clReleaseEvent(ev), "clReleaseEvent");

        if (blocking)
            mgcl::mgclCheckError(clFinish(commands), "clFinish");

        err = clReleaseKernel(kernel);
        mgcl::mgclCheckError(err, "Releasing fill_buffer kernel");
    }

    /**
     * @brief Retains the buffer using clRetainMemObject.
     */
    void CuboidBSGpu::retain()
    {
        mgclCheckError(clRetainMemObject(buffer), "clRetainMemObject(buffer)");
    }

    /**
     * @brief Returns the reference count of the buffer.
     *
     * @return int reference count
     */
    int CuboidBSGpu::refCount()
    {
        cl_uint refCount;
        int err = clGetMemObjectInfo(buffer, CL_MEM_REFERENCE_COUNT, sizeof(cl_uint), &refCount, nullptr);
        mgcl::mgclCheckError(err, "clGetMemObjectInfo(CL_MEM_REFERENCE_COUNT)");
        return refCount;
    }

    /**
     * @brief Asynchronously copies values from this buffer to the other buffer. Dimensions must match.
     *
     * @param commands command queue
     * @param target buffer to copy to
     * @throws string If Dimensions do not match.
     */
    void CuboidBSGpu::copyTo(cl_command_queue commands, CuboidBSGpu& target)
    {
        if (size != target.getSize())
            error("Sizes do not match!");

        int err = clEnqueueCopyBuffer(commands, buffer, target.getBuffer(), 0, 0,
                                      sizeof(double) * size, 0, NULL, NULL);
        mgclCheckError(err, "clEnqueueCopyBuffer");
    }

    /**
     * @brief Returns a shallow copy of this CuboidBSGpu instance, i.e. creating a new CuboidBSGpu with the same dimensions
     * but without copying the data. The copy has the same read-write flags as the original.
     *
     * @return CuboidBSGpu
     */
    std::unique_ptr<CuboidBSGpu> CuboidBSGpu::copyShallow()
    {
        // filter flags to only forward r/w access to the copy
        cl_mem_flags f = flags & (CL_MEM_READ_WRITE | CL_MEM_READ_ONLY | CL_MEM_WRITE_ONLY);
        return std::make_unique<CuboidBSGpu>(context, f, m, n, o, ghosts_m, ghosts_n, ghosts_o, blocksize);
    }

    void CuboidBSGpu::dumpToFile(cl_command_queue commands, const std::string& path, bool realCellsOnly) const
    {
        auto tmp = read(commands, nullptr, true);
        std::ofstream myfile;
        myfile.open(path, std::ios::out | std::ios::trunc);

        if (myfile.is_open())
        {
            if (realCellsOnly)
            {
                for (int i = ghosts_m; i < mgh - ghosts_m; i++)
                    for (int j = ghosts_n; j < ngh - ghosts_n; j++)
                        for (int k = ghosts_o; k < ogh - ghosts_o; k++)
                            for (int b = 0; b < blocksize; b++)
                            {
                                myfile << i - ghosts_m << "\t" << j - ghosts_n << "\t" << k - ghosts_o << "\t" << b << "\t"
                                       << std::scientific << std::setprecision(17) << tmp->getData()[i][j][k][b] << std::endl;
                            }
            }
            else
            {
                for (int i = 0; i < mgh; i++)
                    for (int j = 0; j < ngh; j++)
                        for (int k = 0; k < ogh; k++)
                            for (int b = 0; b < blocksize; b++)
                            {
                                myfile << i << "\t" << j << "\t" << k << "\t" << b << "\t"
                                       << std::scientific << std::setprecision(17) << tmp->getData()[i][j][k][b] << std::endl;
                            }
            }
            myfile.close();
        }
        else
        {
            error("Couldn't open file for writing given by: " + path);
        }
    }

    /**
     * @brief Swaps internal buffers of a and b.
     *
     * @param a
     * @param b
     * @throws string If Dimensions do not match.
     */
    void CuboidBSGpu::swap(CuboidBSGpu& a, CuboidBSGpu& b)
    {
        if (a.getM() != b.getM() || a.getN() != b.getN() || a.getO() != b.getO() ||
            a.getGhostsM() != b.getGhostsM() || a.getGhostsN() != b.getGhostsN() || a.getGhostsO() != b.getGhostsO() ||
            a.getBlocksize() != b.getBlocksize())
            error("Dimensions do not match!");

        std::swap(a.buffer, b.buffer);
    }

    /**
     * @brief Extracts the border planes of the cuboid. If target is nullptr, a new CuboidBS is created and returned.
     *
     * @param commands OpenCL command queue
     * @param program OpenCL program
     * @param d_target CuboidBSGpu that data gets extracted into. If nullptr, a new CuboidBSGpu is created temporarily.
     * @param h_target CuboidBS that data gets extracted into. If target is nullptr, a new CuboidBS is created and returned.
     * @return std::unique_ptr<CuboidBS>
     */
    std::unique_ptr<std::vector<double>> CuboidBSGpu::extractBorderPlanes(cl_command_queue commands, cl_program program,
                                                                          BufferGpu* d_target, std::vector<double>* h_target,
                                                                          mgcl::conf::KernelConfig* conf, mgcl::ProfilingData* pd)
    {
        // Plane sizes
        int yz = ngh * ogh;
        int xz = mgh * ogh;
        int xy = mgh * ngh;
        int ressize = (2 * yz * ghosts_m + 2 * xz * ghosts_n + 2 * xy * ghosts_o) * blocksize;

        if (ghosts_m > m || ghosts_n > n || ghosts_o > o)
            error("CuboidBSGpu::extractBorderPlanes: Only defined for ghosts <= m, n, o");

        // Create return buffer, if not provided
        std::unique_ptr<std::vector<double>> ret = nullptr;
        std::vector<double>* retraw = h_target;
        if (h_target == nullptr)
        {
            ret = std::make_unique<std::vector<double>>(ressize);
            retraw = ret.get();
        }

        // Create device target buffer, if not provided
        bool createdDTarget = false;
        if (d_target == nullptr)
        {
            d_target = new BufferGpu(context, CL_MEM_READ_WRITE, ressize);
            d_target->write(commands, *retraw, true);
            createdDTarget = true;
        }

        int err;

        // Create the compute kernel from the program
        const char* kernelName = "extract_border_planes_cuboidbs";
        cl_kernel kernel = clCreateKernel(program, kernelName, &err);
        mgcl::mgclCheckError(err, "clCreateKernel");

        // assign kernel arguments
        cl_mem d_target_buffer = d_target->getBuf();
        int pos = 0;
        err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &buffer);
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
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &blocksize);
        mgcl::mgclCheckError(err, "Setting kernel arguments");

        // one work-item per ghost cell (excluding real cells). Pad global sizes to fit to local sizes
        size_t global = ressize;
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
        mgcl::mgclCheckError(err, "Enqueueing extract_border_planes_cuboidbs kernel");

        if (pd != nullptr)
        {
            pd->addMeasurement(commands, ev, kernelName,
                               {global, 0, 0},
                               {local, 1, 1});
        }
        mgclCheckError(clReleaseEvent(ev), "clReleaseEvent");

        mgcl::mgclCheckError(clFinish(commands), "clFinish");

        err = clReleaseKernel(kernel);
        mgcl::mgclCheckError(err, "Releasing extract_border_planes_cuboidbs kernel");

        // Read into h_target
        d_target->read(commands, retraw->data(), true, ressize);

        if (createdDTarget)
            delete d_target;

        return ret;
    }

    /**
     * @brief Pastes into ghost cells of this cuboid from provided border planes (d_source or h_source). At least one source
     *  buffer must be given.
     *
     * @param commands
     * @param program
     * @param d_source Device buffer containing border planes in the same order as they get extracted by extractBorderPlanes.
     *   if null, h_source is used instead.
     * @param h_source Host buffer containing border planes in the same order as they get extracted by extractBorderPlanes.
     *   Ignored if d_source is not null. If d_source is null, a temporary device buffer is created.
     */
    void CuboidBSGpu::pasteGhostsFromBorderPlanes(cl_context context, cl_command_queue commands, cl_program program,
                                                  BufferGpu* d_source, std::vector<double>* h_source,
                                                  mgcl::conf::KernelConfig* conf, mgcl::ProfilingData* pd)
    {
        // TODO implement
        assert(false && "Not implemented yet");

        if (d_source == nullptr && h_source == nullptr)
            error("CuboidBSGpu::pasteGhostsFromBorderPlanes: At least one source buffer must be given.");

        // Plane sizes
        int yz = ngh * ogh;
        int xz = mgh * ogh;
        int xy = mgh * ngh;
        int ressize = 2 * yz * ghosts_m + 2 * xz * ghosts_n + 2 * xy * ghosts_o;

        BufferGpu* d_tmp = d_source;
        bool createdDTmp = false;
        if (d_tmp == nullptr)
        {
            d_tmp = new BufferGpu(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, ressize);
            d_tmp->write(commands, *h_source, true);
            createdDTmp = true;
        }

        int err;

        // Create the compute kernel from the program
        const char* kernelName = "paste_ghosts_from_border_planes";
        cl_kernel kernel = clCreateKernel(program, kernelName, &err);
        mgcl::mgclCheckError(err, "clCreateKernel");

        // assign kernel arguments
        cl_mem d_target_buffer = d_tmp->getBuf();
        int pos = 0;
        err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &buffer);
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

        // one work-item for each border cell. Pad global sizes to fit to local sizes
        size_t global = ressize;
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
        mgcl::mgclCheckError(err, "Enqueueing pasteGhostsFromBorderPlanes kernel");

        if (pd != nullptr)
        {
            pd->addMeasurement(commands, ev, kernelName,
                               {global, 0, 0},
                               {local, 1, 1});
        }
        mgclCheckError(clReleaseEvent(ev), "clReleaseEvent");

        mgcl::mgclCheckError(clFinish(commands), "clFinish");

        err = clReleaseKernel(kernel);
        mgcl::mgclCheckError(err, "Releasing pasteGhostsFromBorderPlanes kernel");

        if (createdDTmp)
            delete d_tmp;
    }

    void CuboidBSGpu::updateGhostsLocally(cl_program program, cl_command_queue commands,
                                          conf::KernelConfig* conf, mgcl::ProfilingData* pd)
    {
        int err;

        // Create the compute kernel from the program
        const char* kernelName = "update_ghosts_cuboidbs_periodic_blockstencil";
        cl_kernel kernel = clCreateKernel(program, kernelName, &err);
        mgclCheckError(err, "clCreateKernel");

        // assign kernel arguments
        int pos = 0;
        err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &buffer);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &m);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &n);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &o);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghosts_m);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghosts_n);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghosts_o);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &blocksize);
        mgclCheckError(err, "Setting kernel arguments");

        // one work-item per ghost cell (excluding real cells). Pad global sizes to fit to local sizes
        // int mgh = m + 2 * gh;
        // int ngh = n + 2 * gh;
        // int ogh = o + 2 * gh;
        size_t global[3] = {static_cast<size_t>(mgh), static_cast<size_t>(ngh), static_cast<size_t>(ogh)};
        std::array<size_t, 3> c{4, 4, 4};
        if (conf)
        {
            c = conf::getWorkGroupSizeForKernelAndWiCount(*conf, kernelName, 1);
        }

        const size_t local[3] = {
            static_cast<size_t>(mgh > c[0] ? c[0] : mgh),
            static_cast<size_t>(ngh > c[1] ? c[1] : ngh),
            static_cast<size_t>(ogh > c[2] ? c[2] : ogh)};

        for (int i = 0; i < 3; i++)
            if (global[i] % local[i] != 0)
                global[i] += local[i] - (global[i] % local[i]);

        cl_event ev;

        // enqueue kernel
        err = clEnqueueNDRangeKernel(commands, kernel, 3, NULL, global, local, 0, NULL, &ev);
        mgclCheckError(err, "Enqueueing update_ghosts_cuboidbs_periodic_blockstencil kernel");

        if (pd)
        {
            pd->addMeasurement(commands, ev, kernelName,
                               {global[0], global[1], global[2]},
                               {local[0], local[1], local[2]});
        }
        mgclCheckError(clReleaseEvent(ev), "clReleaseEvent");

        err = clReleaseKernel(kernel);
        mgclCheckError(err, "Releasing update_ghosts_cuboidbs_periodic_blockstencil kernel");
    }

    // /* updates ghost cells on opencl device.
    //  * m,n,o must be size of ghosted grid.
    //  * Only enqueues the kernel. Neither waits for kernel to finish nor reads back results */
    // void CuboidBSGpu::updateGhosts(MPILevelData* mpiData, bool forceLocal)
    // {
    //     if (forceLocal || mpiData == nullptr || mpiData->mpiSize() == 1)
    //     {
    //         updateGhostsLocally();
    //         return;
    //     }

    //     if (problem.useMpi() && !mpiData)
    //         error("Problem uses MPI but mpiData is null!");

    //     if (!problem.isPeriodic())
    //         return CL_SUCCESS;

    //     int err;

    //     // Create the compute kernel from the program
    //     const char* kernelName = "update_ghosts_periodic";
    //     cl_kernel kernel = clCreateKernel(problem.getOpenCLHelper().getProgram(), kernelName, &err);
    //     mgclCheckError(err, "clCreateKernel");

    //     // assign kernel arguments
    //     int pos = 0;
    //     err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &dBuffer);
    //     err |= clSetKernelArg(kernel, ++pos, sizeof(int), &m);
    //     err |= clSetKernelArg(kernel, ++pos, sizeof(int), &n);
    //     err |= clSetKernelArg(kernel, ++pos, sizeof(int), &o);
    //     err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghosts_m);
    //     err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghosts_n);
    //     err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghosts_o);
    //     mgclCheckError(err, "Setting kernel arguments");

    //     // one work-item per ghost cell (excluding real cells). Pad global sizes to fit to local sizes
    //     // int mgh = m + 2 * gh;
    //     // int ngh = n + 2 * gh;
    //     // int ogh = o + 2 * gh;
    //     size_t global[3] = {static_cast<size_t>(mgh), static_cast<size_t>(ngh), static_cast<size_t>(ogh)};
    //     const auto& c = conf::getWorkGroupSizeForKernelAndWiCount(problem.getKernelConfig(), kernelName, 1);
    //     const size_t local[3] = {
    //         static_cast<size_t>(mgh > c[0] ? c[0] : mgh),
    //         static_cast<size_t>(ngh > c[1] ? c[1] : ngh),
    //         static_cast<size_t>(ogh > c[2] ? c[2] : ogh)};

    //     for (int i = 0; i < 3; i++)
    //         if (global[i] % local[i] != 0)
    //             global[i] += local[i] - (global[i] % local[i]);

    //     cl_event ev;

    //     // enqueue kernel
    //     err = clEnqueueNDRangeKernel(problem.getOpenCLHelper().getCommands(), kernel, 3, NULL, global, local, 0, NULL, &ev);
    //     mgclCheckError(err, "Enqueueing update_ghosts_periodic kernel");

    //     if (problem.isProfilingEnabled())
    //     {
    //         problem.getProfilingData()->addMeasurement(problem.getCommands(), ev, kernelName,
    //                                                    {global[0], global[1], global[2]},
    //                                                    {local[0], local[1], local[2]});
    //     }
    //     mgclCheckError(clReleaseEvent(ev), "clReleaseEvent");

    //     err = clReleaseKernel(kernel);
    //     mgclCheckError(err, "Releasing update_ghosts_periodic kernel");

    //     return err;
    // }

    // /**
    //  * @brief Updates ghosts of an OpenCL buffer respecting MPI usage. That is, the buffer is sent to host, ghosts
    //  * are updated using MPI routines, and the updated buffer is sent back to the device.
    //  * Waits for previous commands to finish before reading the buffer.
    //  *
    //  * @param commands
    //  * @param d_buf
    //  * @param mpiData
    //  * @param m Real grid's size in 1st dim
    //  * @param n Real grid's size in 2nd dim
    //  * @param o Real grid's size in 3rd dim
    //  * @param ghosts_m
    //  * @param ghosts_n
    //  * @param ghosts_o
    //  * @param periodic
    //  * @param forceLocal
    //  */
    // void CuboidBSGpu::updateGhostsOclMpi(Problem& p, CuboidGpu& d_buf, MPILevelData& mpiData,
    //                                      bool periodic, bool forceLocal)
    // {
    //     // Read back from GPU and update ghosts on host in order to update neighbouring nodes, too.
    //     // auto tmp = d_buf.read(commands, nullptr, true);
    //     // MultigridEngine::updateGhostsSeq(*tmp, &mpiData, periodic, forceLocal);
    //     // d_buf.write(commands, *tmp, true);

    //     if (forceLocal)
    //     {
    //         MultigridEngine::updateGhosts(p, d_buf, nullptr, true);
    //         return;
    //     }

    //     if (p.getDPlanesBufPtr() == nullptr)
    //         error("MultigridEngine::updateGhostsOclMpi: dPlanesBufPtr is null");

    //     // Use temporary buffer for extracting and pasting planes. Check if it's large enough beforehand.
    //     // TODO maybe disable check in UNSAFE mode
    //     int yz = d_buf.getNgh() * d_buf.getOgh();
    //     int xz = d_buf.getMgh() * d_buf.getOgh();
    //     int xy = d_buf.getMgh() * d_buf.getNgh();
    //     int ressize = 2 * yz * d_buf.getGhostsM() + 2 * xz * d_buf.getGhostsN() + 2 * xy * d_buf.getGhostsO();

    //     auto dPlanesBuf = p.getDPlanesBufPtr();
    //     if (dPlanesBuf->getSize() < ressize)
    //         error("MultigridEngine::updateGhostsOclMpi: dPlanesBuf is too small. Need at least " + std::to_string(ressize) + ", but is " + std::to_string(dPlanesBuf->getSize()));

    //     auto hPlanesBufSend = p.getHPlanesBufSendPtr();
    //     auto hPlanesBufRecv = p.getHPlanesBufRecvPtr();
    //     if (hPlanesBufSend->size() < ressize || hPlanesBufRecv->size() < ressize)
    //         throw "MultigridEngine::updateGhostsOclMpi: hPlanesBufSend or hPlanesBufRecv is too small. Need at least " +
    //             std::to_string(ressize) + ", but is " + std::to_string(hPlanesBufSend->size()) +
    //             " (send) and " + std::to_string(hPlanesBufRecv->size()) + " (recv)";

    //     // Extract border planes from the buffer
    //     d_buf.extractBorderPlanes(p.getCommands(), p.getProgram(),
    //                               dPlanesBuf, hPlanesBufSend,
    //                               &p.getKernelConfig(), p.getProfilingData());
    //     auto& sbuf = *hPlanesBufSend;
    //     auto& rbuf = *hPlanesBufRecv;

    //     // Send our planes to neighbours and receive their planes
    //     mpi_util::sendBorderPlanes(d_buf.getMgh(), d_buf.getNgh(), d_buf.getOgh(),
    //                                d_buf.getGhostsM(), d_buf.getGhostsN(), d_buf.getGhostsO(), 1,
    //                                sbuf, rbuf, mpiData);

    //     // Paste planes back into the buffer.
    //     dPlanesBuf->write(p.getCommands(), rbuf, false, ressize);
    //     d_buf.pasteGhostsFromBorderPlanes(p.getContext(), p.getCommands(), p.getProgram(),
    //                                       dPlanesBuf, nullptr,
    //                                       &p.getKernelConfig(), p.getProfilingData());
    // }

} // namespace mgcl