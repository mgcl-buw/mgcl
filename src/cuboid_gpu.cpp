#include "cuboid_gpu.hpp"

#include "opencl_helper.hpp"

namespace mgcl
{
    /**
     * @brief Construct a new CuboidGpu object with given sizes. Copies data from a Cuboid, if host_ptr
     *   is given. A CuboidGpu object is only valid in its context.
     * Currently only GPU devices are supported, i.e. the flag for creating the buffer is always CL_MEM_COPY_HOST_PTR.
     *   However, one may add e.g. CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY or CL_MEM_READ_ONLY.
     *
     * @param context OpenCL context this buffer is valid in
     * @param flags OpenCL flags
     * @param m extend
     * @param n extend
     * @param o extend
     * @param ghosts_m amount of ghost cells at one border
     * @param ghosts_n amount of ghost cells at one border
     * @param ghosts_o amount of ghost cells at one border
     * @param host_ptr Cuboid that gets copied from initially.
     */
    CuboidGpu::CuboidGpu(cl_context context, cl_mem_flags flags,
                         int m, int n, int o,
                         int ghosts_m, int ghosts_n, int ghosts_o,
                         const Cuboid* const host_ptr)
        : context(context), buffer(nullptr), m(m), n(n), o(o),
          ghosts_m(ghosts_m), ghosts_n(ghosts_n), ghosts_o(ghosts_o),
          mgh(m + 2 * ghosts_m), ngh(n + 2 * ghosts_n), ogh(o + 2 * ghosts_o),
          size(mgh * ngh * ogh)
    {
        if (m <= 0 || n <= 0 || o <= 0)
            throw "m, n and o must be > 0.";

        if (ghosts_m < 0 || ghosts_n < 0 || ghosts_o < 0)
            throw "ghosts must be >= 0.";

        if (flags != CL_MEM_READ_WRITE && flags != CL_MEM_WRITE_ONLY && flags != CL_MEM_READ_ONLY)
            throw "flags must be CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY or CL_MEM_WRITE_ONLY";

        cl_int err;
        if (host_ptr)
        {
            if (host_ptr->getM() != m || host_ptr->getN() != n || host_ptr->getO() != o ||
                host_ptr->getMgh() != mgh || host_ptr->getNgh() != ngh || host_ptr->getOgh() != ogh)
                throw "Dimension of host_ptr and CuboidGpu must match.";

            // TODO adjust flags for CPU
            buffer = clCreateBuffer(context, flags | CL_MEM_COPY_HOST_PTR, sizeof(double) * size,
                                    host_ptr->getData()[0][0], &err);
        }
        else
            buffer = clCreateBuffer(context, flags, sizeof(double) * size, nullptr, &err);
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

} // namespace mgcl