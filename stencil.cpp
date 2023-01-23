#include "stencil.hpp"
#include "opencl_helper.hpp"

namespace mgcl
{

    VaryingStencilGpu::VaryingStencilGpu(int m_, int n_, int o_, int width_, int gh_, cl_context context, cl_command_queue queue)
        : m(m_), n(n_), o(o_), width(width_), gh(gh_)
    {
        int err;
        buf = clCreateBuffer(context, CL_MEM_READ_WRITE,
                             sizeof(double) * (m + 2 * gh) * (n + 2 * gh) * (o + 2 * gh) * width * width * width,
                             NULL, &err);
        mgclCheckError(err, "clCreateBuffer");

        cl_double zero = 0;
        err = clEnqueueFillBuffer(queue, buf, &zero, sizeof(cl_double), 0,
                                  sizeof(double) * (m + 2 * gh) * (n + 2 * gh) * (o + 2 * gh) * width * width * width,
                                  0, NULL, NULL);
        mgclCheckError(err, "clEnqueueFillBuffer");
    }

    VaryingStencilGpu::~VaryingStencilGpu()
    {
        int err = clReleaseMemObject(buf);
        mgclCheckError(err, "clReleaseMemObject");
        buf = nullptr;
    }

    /**
     * Updates ghost cells, respects periodic ghosts, i.e. when gh > m
     */
    void VaryingStencilGpu::updateGhosts(cl_program program, cl_command_queue queue)
    {
        int err;

        // Create the compute kernel from the program
        cl_kernel kernel = clCreateKernel(program, "update_ghosts_varying_stencil", &err);
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
        const size_t local[3] = {
            static_cast<size_t>(mgh > 4 ? 4 : mgh),
            static_cast<size_t>(ngh > 4 ? 4 : ngh),
            static_cast<size_t>(ogh > 4 ? 4 : ogh)};

        for (int i = 0; i < 3; i++)
            if (global[i] % local[i] != 0)
            {
                // printf("padding global size %d from %ld to ", i, global[i]);
                global[i] += local[i] - (global[i] % local[i]);
                // printf("%ld (multiple of %ld)\n", global[i], local[i]);
            }

        // enqueue kernel
        err = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global, local, 0, NULL, NULL);
        mgclCheckError(err, "Enqueueing update ghosts of varying stencil kernel");

        err = clReleaseKernel(kernel);
        mgclCheckError(err, "Releasing update ghosts of varying stencil kernel");
    }

    /**
     * @brief Multiplies two varying stencils on the gpu and creates a new gpu buffer which will be returned.
     *
     * @param b
     * @param ghc
     * @param program
     * @param queue
     * @param context
     * @return std::unique_ptr<VaryingStencilGpu>
     */
    std::unique_ptr<VaryingStencilGpu> VaryingStencilGpu::multiply(VaryingStencilGpu &b, int ghc,
                                                                   cl_program program, cl_command_queue queue, cl_context context)
    {
        int err;

        // Create the compute kernel from the program
        cl_kernel kernel = clCreateKernel(program, "mult_stencils_var_var", &err);
        mgclCheckError(err, "clCreateKernel");

        // create output buffer c
        auto c = std::make_unique<VaryingStencilGpu>(m, n, o, width + b.getWidth() - 1, ghc, context, queue);

        auto bbuf = b.getBuf();
        auto cbuf = c->getBuf();
        auto wb = b.getWidth();
        auto ghb = b.getGh();

        // assign kernel arguments
        int pos = 0;
        err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &buf);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &bbuf);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &cbuf);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &m);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &n);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &o);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &width);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &wb);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &gh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghb);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghc);
        mgclCheckError(err, "Setting kernel arguments");

        // one work-item per cell (excluding ghost cells). Pad global sizes to fit to local sizes
        size_t global[3] = {static_cast<size_t>(m), static_cast<size_t>(n), static_cast<size_t>(o)};
        const size_t local[3] = {static_cast<size_t>(m > 4 ? 4 : m), static_cast<size_t>(n > 4 ? 4 : n),
                                 static_cast<size_t>(o > 4 ? 4 : o)};

        for (int i = 0; i < 3; i++)
            if (global[i] % local[i] != 0)
            {
                // printf("padding global size %d from %ld to ", i, global[i]);
                global[i] += local[i] - (global[i] % local[i]);
                // printf("%ld (multiple of %ld)\n", global[i], local[i]);
            }

        // update ghosts of b first (maybe not needed if done earlier)
        // b.updateGhosts(program, queue);

        // enqueue multiplication kernel
        err = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global, local, 0, NULL, NULL);
        mgclCheckError(err, "Enqueueing stencil multiplication kernel");

        // update ghosts of c
        if (ghc > 0)
            c->updateGhosts(program, queue);

        clReleaseKernel(kernel);
        return c;
    }

    /**
     * @brief Multiplies a varying stencil a with a fixed stencil b on the gpu and creates a new gpu buffer c which will
     * be returned, i.e. a * b = c
     *
     * @param b
     * @param ghc
     * @param program
     * @param queue
     * @param context
     * @return std::unique_ptr<VaryingStencilGpu>
     */
    std::unique_ptr<VaryingStencilGpu> VaryingStencilGpu::multiply(FixedStencilGpu &b, int ghc,
                                                                   cl_program program, cl_command_queue queue, cl_context context)
    {
        int err;

        // Create the compute kernel from the program
        cl_kernel kernel = clCreateKernel(program, "mult_stencils_var_fix", &err);
        mgclCheckError(err, "clCreateKernel");

        // create output buffer c
        auto c = std::make_unique<VaryingStencilGpu>(m, n, o, width + b.getWidth() - 1, ghc, context, queue);

        auto bbuf = b.getBuf();
        auto cbuf = c->getBuf();
        auto wb = b.getWidth();

        // assign kernel arguments
        int pos = 0;
        err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &buf);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &bbuf);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &cbuf);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &m);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &n);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &o);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &width);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &wb);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &gh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghc);
        mgclCheckError(err, "Setting kernel arguments");

        // one work-item per cell (excluding ghost cells). Pad global sizes to fit to local sizes
        size_t global[3] = {static_cast<size_t>(m), static_cast<size_t>(n), static_cast<size_t>(o)};
        const size_t local[3] = {static_cast<size_t>(m > 4 ? 4 : m), static_cast<size_t>(n > 4 ? 4 : n),
                                 static_cast<size_t>(o > 4 ? 4 : o)};

        for (int i = 0; i < 3; i++)
            if (global[i] % local[i] != 0)
            {
                // printf("padding global size %d from %ld to ", i, global[i]);
                global[i] += local[i] - (global[i] % local[i]);
                // printf("%ld (multiple of %ld)\n", global[i], local[i]);
            }

        // enqueue multiplication kernel
        err = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global, local, 0, NULL, NULL);
        mgclCheckError(err, "Enqueueing stencil multiplication kernel");

        // update ghosts of c
        if (ghc > 0)
            c->updateGhosts(program, queue);

        clReleaseKernel(kernel);
        return c;
    }

    int VaryingStencilGpu::getO() const
    {
        return o;
    }

    cl_mem VaryingStencilGpu::getBuf() const
    {
        return buf;
    }

    int VaryingStencilGpu::getN() const
    {
        return n;
    }

    int VaryingStencilGpu::getGh() const
    {
        return gh;
    }

    int VaryingStencilGpu::getM() const
    {
        return m;
    }

    int VaryingStencilGpu::getWidth() const
    {
        return width;
    }

    /**
     * *********************************************
     * FixedStencilGpu below
     * *********************************************
     */

    FixedStencilGpu::FixedStencilGpu(int width_, cl_context context, cl_command_queue queue)
        : width(width_)
    {
        int err;
        buf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double) * width * width * width, NULL, &err);
        mgclCheckError(err, "clCreateBuffer");

        cl_double zero = 0;
        err = clEnqueueFillBuffer(queue, buf, &zero, sizeof(cl_double), 0, sizeof(double) * width * width * width,
                                  0, NULL, NULL);
        mgclCheckError(err, "clEnqueueFillBuffer");
    }

    FixedStencilGpu::~FixedStencilGpu()
    {
        int err = clReleaseMemObject(buf);
        mgclCheckError(err, "clReleaseMemObject");
        buf = nullptr;
    }

    /**
     * @brief Multiplies a fixed stencil a with a varying stencil b on the gpu and creates a new gpu buffer which will
     * be returned, i.e. a * b = c.
     *
     * @param b
     * @param ghc
     * @param program
     * @param queue
     * @param context
     * @return std::unique_ptr<VaryingStencilGpu>
     */
    std::unique_ptr<VaryingStencilGpu> FixedStencilGpu::multiply(VaryingStencilGpu &b, int ghc,
                                                                 cl_program program, cl_command_queue queue, cl_context context)
    {
        int err;

        // Create the compute kernel from the program
        cl_kernel kernel = clCreateKernel(program, "mult_stencils_fix_var", &err);
        mgclCheckError(err, "clCreateKernel");

        int m = b.getM();
        int n = b.getN();
        int o = b.getO();

        // create output buffer c
        auto c = std::make_unique<VaryingStencilGpu>(m, n, o, width + b.getWidth() - 1, ghc, context, queue);

        auto bbuf = b.getBuf();
        auto cbuf = c->getBuf();
        auto wb = b.getWidth();
        auto ghb = b.getGh();

        // assign kernel arguments
        int pos = 0;
        err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &buf);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &bbuf);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &cbuf);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &m);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &n);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &o);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &width);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &wb);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghb);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghc);
        mgclCheckError(err, "Setting kernel arguments");

        // one work-item per cell (excluding ghost cells). Pad global sizes to fit to local sizes
        size_t global[3] = {static_cast<size_t>(m), static_cast<size_t>(n), static_cast<size_t>(o)};
        const size_t local[3] = {static_cast<size_t>(m > 4 ? 4 : m), static_cast<size_t>(n > 4 ? 4 : n),
                                 static_cast<size_t>(o > 4 ? 4 : o)};

        for (int i = 0; i < 3; i++)
            if (global[i] % local[i] != 0)
            {
                // printf("padding global size %d from %ld to ", i, global[i]);
                global[i] += local[i] - (global[i] % local[i]);
                // printf("%ld (multiple of %ld)\n", global[i], local[i]);
            }

        // update ghosts of b first (maybe not needed if done earlier)
        // b.updateGhosts(program, queue);

        // enqueue multiplication kernel
        err = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global, local, 0, NULL, NULL);
        mgclCheckError(err, "Enqueueing stencil multiplication kernel");

        // update ghosts of c
        if (ghc > 0)
            c->updateGhosts(program, queue);

        clReleaseKernel(kernel);
        return c;
    }

    int FixedStencilGpu::getWidth() const
    {
        return width;
    }

    cl_mem FixedStencilGpu::getBuf() const
    {
        return buf;
    }
}
