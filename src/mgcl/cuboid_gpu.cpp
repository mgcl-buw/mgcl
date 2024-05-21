#include "cuboid_gpu.hpp"

#include "kernel_config.hpp"
#include "opencl_helper.hpp"

#include <fstream>
#include <iomanip>
#include <memory>

namespace mgcl
{
    /**
     * @brief Construct a new CuboidGpu object with given sizes. A CuboidGpu object is only valid in its context.
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
    CuboidGpu::CuboidGpu(cl_context context, cl_mem_flags flags,
                         int m, int n, int o,
                         int ghosts_m, int ghosts_n, int ghosts_o,
                         const cl_mem buf)
        : context(context), buffer(buf), flags(flags), m(m), n(n), o(o),
          ghosts_m(ghosts_m), ghosts_n(ghosts_n), ghosts_o(ghosts_o),
          mgh(m + 2 * ghosts_m), ngh(n + 2 * ghosts_n), ogh(o + 2 * ghosts_o),
          size(mgh * ngh * ogh)
    {
        if (m <= 0 || n <= 0 || o <= 0)
            throw "m, n and o must be > 0.";

        if (ghosts_m < 0 || ghosts_n < 0 || ghosts_o < 0)
            throw "ghosts must be >= 0.";

        if (buf)
            retain();
        else
        {
            bool containsReadWrite = (flags & CL_MEM_READ_WRITE) == CL_MEM_READ_WRITE;
            bool containsWriteOnly = (flags & CL_MEM_WRITE_ONLY) == CL_MEM_WRITE_ONLY;
            bool containsReadOnly = (flags & CL_MEM_READ_ONLY) == CL_MEM_READ_ONLY;

            // Check that flags contains one and only one of CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY or CL_MEM_READ_ONLY
            if (!containsReadWrite && !containsWriteOnly && !containsReadOnly)
                throw "flags must contain one of CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY or CL_MEM_READ_ONLY.";

            if ((containsReadWrite && containsReadOnly) || (containsReadWrite && containsWriteOnly) || (containsReadOnly && containsWriteOnly))
                throw "flags must contain one and only one of CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY or CL_MEM_READ_WRITE.";

            bool containsCopyHostPtr = (flags & CL_MEM_COPY_HOST_PTR) == CL_MEM_COPY_HOST_PTR;
            bool containsUseHostPtr = (flags & CL_MEM_USE_HOST_PTR) == CL_MEM_USE_HOST_PTR;
            bool containsAllocHostPtr = (flags & CL_MEM_ALLOC_HOST_PTR) == CL_MEM_ALLOC_HOST_PTR;

            // Check that flags does not contain one of CL_MEM_COPY_HOST_PTR, CL_MEM_USE_HOST_PTR or CL_MEM_ALLOC_HOST_PTR
            if ((containsAllocHostPtr || containsUseHostPtr || containsCopyHostPtr))
                throw "host_ptr is null, but flags contains CL_MEM_ALLOC_HOST_PTR, CL_MEM_USE_HOST_PTR or CL_MEM_COPY_HOST_PTR.";

            cl_int err;
            buffer = clCreateBuffer(context, flags, sizeof(double) * size, nullptr, &err);
            mgclCheckError(err, "clCreateBuffer");
        }
    }

    /**
     * @brief Construct a new CuboidGpu object with the same size and content as of the given Cuboid.
     *   A CuboidGpu object is only valid in its context. flags must be valid, see parameter specification.
     *
     * @param context OpenCL context this buffer is valid in
     * @param flags Must contain one and only one of CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY or CL_MEM_READ_ONLY as well as
     *   one and only one of CL_MEM_COPY_HOST_PTR, CL_MEM_USE_HOST_PTR or CL_MEM_ALLOC_HOST_PTR.
     *   Example using a gpu with given host_ptr: CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR
     * @param host_data Cuboid that gets copied from initially.
     * @throws string if flags are not valid
     */
    CuboidGpu::CuboidGpu(cl_context context, cl_mem_flags flags,
                         const Cuboid& host_data)
        : context(context), buffer(nullptr), flags(flags), m(host_data.getM()),
          n(host_data.getN()), o(host_data.getO()),
          ghosts_m(host_data.getGhostsM()), ghosts_n(host_data.getGhostsN()), ghosts_o(host_data.getGhostsO()),
          mgh(m + 2 * ghosts_m), ngh(n + 2 * ghosts_n), ogh(o + 2 * ghosts_o),
          size(mgh * ngh * ogh)
    {
        bool containsReadWrite = (flags & CL_MEM_READ_WRITE) == CL_MEM_READ_WRITE;
        bool containsWriteOnly = (flags & CL_MEM_WRITE_ONLY) == CL_MEM_WRITE_ONLY;
        bool containsReadOnly = (flags & CL_MEM_READ_ONLY) == CL_MEM_READ_ONLY;

        // Check that flags contains one and only one of CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY or CL_MEM_READ_ONLY
        if (!containsReadWrite && !containsWriteOnly && !containsReadOnly)
            throw "flags must contain one of CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY or CL_MEM_READ_ONLY.";

        if ((containsReadWrite && containsReadOnly) || (containsReadWrite && containsWriteOnly) || (containsReadOnly && containsWriteOnly))
            throw "flags must contain one and only one of CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY or CL_MEM_READ_WRITE.";

        bool containsCopyHostPtr = (flags & CL_MEM_COPY_HOST_PTR) == CL_MEM_COPY_HOST_PTR;
        bool containsUseHostPtr = (flags & CL_MEM_USE_HOST_PTR) == CL_MEM_USE_HOST_PTR;
        bool containsAllocHostPtr = (flags & CL_MEM_ALLOC_HOST_PTR) == CL_MEM_ALLOC_HOST_PTR;

        // Check that flags contains one of CL_MEM_COPY_HOST_PTR, CL_MEM_USE_HOST_PTR or CL_MEM_ALLOC_HOST_PTR if
        // host_ptr is given.
        if (!(containsAllocHostPtr || containsCopyHostPtr || containsUseHostPtr))
            throw "host_ptr not null, but flags does not contain CL_MEM_ALLOC_HOST_PTR, CL_MEM_USE_HOST_PTR or CL_MEM_COPY_HOST_PTR.";

        // Check that flags contains one and only one of CL_MEM_COPY_HOST_PTR, CL_MEM_USE_HOST_PTR or CL_MEM_ALLOC_HOST_PTR.
        if ((containsCopyHostPtr && containsAllocHostPtr) || (containsCopyHostPtr && containsUseHostPtr) || (containsUseHostPtr && containsAllocHostPtr))
            throw "flags must contain one and only one of CL_MEM_COPY_HOST_PTR, CL_MEM_ALLOC_HOST_PTR, or CL_MEM_USE_HOST_PTR";

        cl_int err;
        buffer = clCreateBuffer(context, flags, sizeof(double) * size, host_data.getData()[0][0], &err);
        mgclCheckError(err, "clCreateBuffer");
    }

    CuboidGpu::~CuboidGpu()
    {
        if (buffer != nullptr)
            mgclCheckError(clReleaseMemObject(buffer), "clReleaseMemObject");
    }

    /**
     * @brief Reads the buffer into host_ptr, if not null, or a new Cuboid otherwise. Dimensions of host_ptr must
     *   match.
     *
     * @param host_ptr Cuboid that data gets read into.
     * @param blocking Blocking read, if true.
     * @return std::unique_ptr<Cuboid> New Cuboid that data gets read into, if host_ptr is null. If host_ptr is not
     *   null, a nullptr will be returned.
     * @throws string If Dimensions do not match.
     */
    std::unique_ptr<Cuboid> CuboidGpu::read(cl_command_queue commands, Cuboid* const host_ptr, bool blocking) const
    {
        if (host_ptr)
        {
            if (host_ptr->getM() != m || host_ptr->getN() != n || host_ptr->getO() != o ||
                host_ptr->getGhostsM() != ghosts_m ||
                host_ptr->getGhostsN() != ghosts_n ||
                host_ptr->getGhostsO() != ghosts_o)
                throw "Dimensions do not match!";
        }

        auto ret = host_ptr ? nullptr : std::make_unique<Cuboid>(m, n, o, ghosts_m, ghosts_n, ghosts_o);

        auto ptr = host_ptr ? host_ptr : ret.get();
        int err = clEnqueueReadBuffer(commands, buffer, blocking ? CL_TRUE : CL_FALSE, 0, sizeof(double) * size,
                                      ptr->getData()[0][0], 0, NULL, NULL);
        mgcl::mgclCheckError(err, "clEnqueueReadBuffer");

        return ret;
    }

    /**
     * @brief Reads the buffer into host_ptr, if not null, or a new Cuboid otherwise. Dimensions of host_ptr must
     *   *not* match, instead size elements are read but which must be less than the target size.
     *   If a new cuboid is created, it has the size 1x1xsize and ghosts=0.
     *
     * @param size Number of elements to read
     * @param host_ptr Cuboid that data gets read into.
     * @param blocking Blocking read, if true.
     * @return std::unique_ptr<Cuboid> New Cuboid that data gets read into, if host_ptr is null. If host_ptr is not
     *   null, a nullptr will be returned.
     * @throws string If Dimensions do not match.
     */
    std::unique_ptr<Cuboid> CuboidGpu::read1d(cl_command_queue commands, int _size, Cuboid* const host_ptr, bool blocking) const
    {
        if (host_ptr)
        {
            if (host_ptr->getSize() < _size)
                throw "CuboidGpu::read1d: Target Cuboid is too small!";
        }

        if (size < _size)
            throw "CuboidGpu::read1d: Source Cuboid is smaller than requested read size!";

        auto ret = host_ptr ? nullptr : std::make_unique<Cuboid>(1, 1, _size);

        auto ptr = host_ptr ? host_ptr : ret.get();
        int err = clEnqueueReadBuffer(commands, buffer, blocking ? CL_TRUE : CL_FALSE, 0, sizeof(double) * _size,
                                      ptr->getData()[0][0], 0, NULL, NULL);
        mgcl::mgclCheckError(err, "clEnqueueReadBuffer");

        return ret;
    }

    /**
     * @brief Writes data from host_data into the gpu buffer.
     *
     * @param commands OpenCL command queue
     * @param host_data Cuboid that the data gets written from
     * @param blocking Blocking Write, if true.
     * @throws string If Dimensions do not match.
     */
    void CuboidGpu::write(cl_command_queue commands, const Cuboid& host_data, bool blocking)
    {
        if (host_data.getM() != m || host_data.getN() != n || host_data.getO() != o ||
            host_data.getGhostsM() != ghosts_m ||
            host_data.getGhostsN() != ghosts_n ||
            host_data.getGhostsO() != ghosts_o)
            throw "Dimensions do not match!";

        int err = clEnqueueWriteBuffer(commands, buffer, blocking ? CL_TRUE : CL_FALSE, 0, sizeof(double) * size,
                                       host_data.getData()[0][0], 0, NULL, NULL);
        mgcl::mgclCheckError(err, "clEnqueueReadBuffer");
    }

    /**
     * @brief Writes data from host_data into the gpu buffer.
     *
     * @param commands OpenCL command queue
     * @param host_data Cuboid that the data gets written from
     * @param blocking Blocking Write, if true.
     * @throws string If Dimensions do not match.
     */
    void CuboidGpu::write1d(cl_command_queue commands, int _size, const Cuboid& host_data, bool blocking)
    {
        if (host_data.getSize() < _size)
            throw "CuboidGpu::write1d: Source host cuboid is too small!";

        if (size < _size)
            throw "CuboidGpu::write1d: Target device cuboid is too small!";

        int err = clEnqueueWriteBuffer(commands, buffer, blocking ? CL_TRUE : CL_FALSE, 0, sizeof(double) * _size,
                                       host_data.getData()[0][0], 0, NULL, NULL);
        mgcl::mgclCheckError(err, "clEnqueueReadBuffer");
    }

    /**
     * @brief Fills the buffer with the given value.
     *
     * @param commands OpenCL command queue
     * @param value Value to fill the buffer with.
     * @param blocking If true, the write will be blocking.
     */
    void CuboidGpu::fill(cl_command_queue commands, double value, bool blocking)
    {
        int err = clEnqueueFillBuffer(commands, buffer, &value, sizeof(double), 0, sizeof(double) * size,
                                      0, nullptr, nullptr);
        mgcl::mgclCheckError(err, "clEnqueueFillBuffer");
        if (blocking)
            mgcl::mgclCheckError(clFinish(commands), "clFinish");
    }

    /**
     * @brief Retains the buffer using clRetainMemObject.
     */
    void CuboidGpu::retain()
    {
        mgclCheckError(clRetainMemObject(buffer), "clRetainMemObject(buffer)");
    }

    /**
     * @brief Returns the reference count of the buffer.
     *
     * @return int reference count
     */
    int CuboidGpu::refCount()
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
    void CuboidGpu::copyTo(cl_command_queue commands, CuboidGpu& target)
    {
        if (mgh != target.getMgh() || ngh != target.getNgh() || ogh != target.getOgh())
            throw "Dimensions do not match!";

        int err = clEnqueueCopyBuffer(commands, buffer, target.getBuffer(), 0, 0,
                                      sizeof(double) * mgh * ngh * ogh, 0, NULL, NULL);
        mgclCheckError(err, "clEnqueueCopyBuffer");
    }

    /**
     * @brief Returns a shallow copy of this CuboidGpu instance, i.e. creating a new CuboidGpu with the same dimensions
     * but without copying the data. The copy has the same read-write flags as the original.
     *
     * @return CuboidGpu
     */
    std::unique_ptr<CuboidGpu> CuboidGpu::copyShallow()
    {
        // filter flags to only forward r/w access to the copy
        cl_mem_flags f = flags & (CL_MEM_READ_WRITE | CL_MEM_READ_ONLY | CL_MEM_WRITE_ONLY);
        return std::make_unique<CuboidGpu>(context, f, m, n, o, ghosts_m, ghosts_n, ghosts_o);
    }

    void CuboidGpu::dumpToFile(cl_command_queue commands, const std::string& path, bool realCellsOnly) const
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
                        {
                            myfile << i - ghosts_m << "\t" << j - ghosts_n << "\t" << k - ghosts_o << "\t"
                                   << std::scientific << std::setprecision(17) << tmp->getData()[i][j][k] << std::endl;
                        }
            }
            else
            {
                for (int i = 0; i < mgh; i++)
                    for (int j = 0; j < ngh; j++)
                        for (int k = 0; k < ogh; k++)
                        {
                            myfile << i << "\t" << j << "\t" << k << "\t"
                                   << std::scientific << std::setprecision(17) << tmp->getData()[i][j][k] << std::endl;
                        }
            }
            myfile.close();
        }
        else
        {
            throw "Couldn't open file for writing given by: " + path;
        }
    }

    int CuboidGpu::getM() const
    {
        return m;
    }

    int CuboidGpu::getGhostsM() const
    {
        return ghosts_m;
    }

    int CuboidGpu::getMgh() const
    {
        return mgh;
    }

    int CuboidGpu::getSize() const
    {
        return size;
    }

    cl_mem CuboidGpu::getBuffer() const
    {
        return buffer;
    }

    int CuboidGpu::getN() const
    {
        return n;
    }

    int CuboidGpu::getGhostsN() const
    {
        return ghosts_n;
    }

    int CuboidGpu::getNgh() const
    {
        return ngh;
    }

    int CuboidGpu::getO() const
    {
        return o;
    }

    int CuboidGpu::getGhostsO() const
    {
        return ghosts_o;
    }

    int CuboidGpu::getOgh() const
    {
        return ogh;
    }

    cl_context CuboidGpu::getContext() const
    {
        return context;
    }

    /**
     * @brief Swaps internal buffers of a and b.
     *
     * @param a
     * @param b
     * @throws string If Dimensions do not match.
     */
    void CuboidGpu::swap(CuboidGpu& a, CuboidGpu& b)
    {
        if (a.getM() != b.getM() || a.getN() != b.getN() || a.getO() != b.getO() ||
            a.getGhostsM() != b.getGhostsM() || a.getGhostsN() != b.getGhostsN() || a.getGhostsO() != b.getGhostsO())
            throw "Dimensions do not match!";

        std::swap(a.buffer, b.buffer);
    }

    /**
     * @brief Extracts the border planes of the cuboid. If target is nullptr, a new Cuboid is created and returned.
     *
     * @param commands OpenCL command queue
     * @param program OpenCL program
     * @param d_target CuboidGpu that data gets extracted into. If nullptr, a new CuboidGpu is created temporarily.
     * @param h_target Cuboid that data gets extracted into. If target is nullptr, a new Cuboid is created and returned.
     * @return std::unique_ptr<Cuboid>
     */
    std::unique_ptr<Cuboid> CuboidGpu::extractBorderPlanes(cl_command_queue commands, cl_program program,
                                                           CuboidGpu* d_target, Cuboid* h_target,
                                                           mgcl::conf::KernelConfig* conf)
    {
        // Plane sizes
        int yz = ngh * ogh;
        int xz = mgh * ogh;
        int xy = mgh * ngh;
        int ressize = 2 * yz * ghosts_m + 2 * xz * ghosts_n + 2 * xy * ghosts_o;

        if (ghosts_m > m || ghosts_n > n || ghosts_o > o)
            throw "CuboidGpu::extractBorderPlanes: Only defined for ghosts <= m, n, o";

        // Create return buffer, if not provided
        std::unique_ptr<Cuboid> ret = nullptr;
        Cuboid* retraw = h_target;
        if (h_target == nullptr)
        {
            ret = std::make_unique<Cuboid>(1, 1, ressize);
            retraw = ret.get();
        }

        // Create device target buffer, if not provided
        bool createdDTarget = false;
        if (d_target == nullptr)
        {
            d_target = new CuboidGpu(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, *retraw);
            createdDTarget = true;
        }

        int err;

        // Create the compute kernel from the program
        const char* kernelName = "extract_border_planes";
        cl_kernel kernel = clCreateKernel(program, kernelName, &err);
        mgcl::mgclCheckError(err, "clCreateKernel");

        // assign kernel arguments
        cl_mem d_target_buffer = d_target->getBuffer();
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

        // cl_event ev;

        // enqueue kernel
        err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
        // err = clEnqueueNDRangeKernel(p.getOpenCLHelper().getCommands(), kernel, 3, NULL, global, local, 0, NULL, &ev);
        mgcl::mgclCheckError(err, "Enqueueing extract_border_planes kernel");

        // if (p.isProfilingEnabled())
        // {
        //     p.getProfilingData()->addMeasurement(p.getCommands(), ev, kernelName,
        //                                          {global[0], global[1], global[2]},
        //                                          {local[0], local[1], local[2]});
        // }
        // mgcl::mgclCheckError(clReleaseEvent(ev), "clReleaseEvent");

        mgcl::mgclCheckError(clFinish(commands), "clFinish");

        err = clReleaseKernel(kernel);
        mgcl::mgclCheckError(err, "Releasing extract_border_planes kernel");

        // Read into h_target
        d_target->read1d(commands, ressize, retraw, true);

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
    void CuboidGpu::pasteGhostsFromBorderPlanes(cl_context context, cl_command_queue commands, cl_program program,
                                                CuboidGpu* d_source, Cuboid* h_source,
                                                mgcl::conf::KernelConfig* conf)
    {
        if (d_source == nullptr && h_source == nullptr)
            throw "CuboidGpu::pasteGhostsFromBorderPlanes: At least one source buffer must be given.";

        // Plane sizes
        int yz = ngh * ogh;
        int xz = mgh * ogh;
        int xy = mgh * ngh;
        int ressize = 2 * yz * ghosts_m + 2 * xz * ghosts_n + 2 * xy * ghosts_o;

        CuboidGpu* d_tmp = d_source;
        bool createdDTmp = false;
        if (d_tmp == nullptr)
        {
            d_tmp = new CuboidGpu(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, *h_source);
            createdDTmp = true;
        }

        int err;

        // Create the compute kernel from the program
        const char* kernelName = "paste_ghosts_from_border_planes";
        cl_kernel kernel = clCreateKernel(program, kernelName, &err);
        mgcl::mgclCheckError(err, "clCreateKernel");

        // assign kernel arguments
        cl_mem d_target_buffer = d_tmp->getBuffer();
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

        // cl_event ev;

        // enqueue kernel
        err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
        // err = clEnqueueNDRangeKernel(p.getOpenCLHelper().getCommands(), kernel, 3, NULL, global, local, 0, NULL, &ev);
        mgcl::mgclCheckError(err, "Enqueueing pasteGhostsFromBorderPlanes kernel");

        // if (p.isProfilingEnabled())
        // {
        //     p.getProfilingData()->addMeasurement(p.getCommands(), ev, kernelName,
        //                                          {global[0], global[1], global[2]},
        //                                          {local[0], local[1], local[2]});
        // }
        // mgcl::mgclCheckError(clReleaseEvent(ev), "clReleaseEvent");

        mgcl::mgclCheckError(clFinish(commands), "clFinish");

        err = clReleaseKernel(kernel);
        mgcl::mgclCheckError(err, "Releasing pasteGhostsFromBorderPlanes kernel");

        if (createdDTmp)
            delete d_tmp;
    }

} // namespace mgcl