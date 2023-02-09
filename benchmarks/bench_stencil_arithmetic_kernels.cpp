#include "nanobench.h"

#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
using namespace std::chrono_literals;

#include "../cuboid.hpp"
#include "../opencl_helper.hpp"
#include "../problem.hpp"
#include "../stencil.hpp"
#include "../test/test_utility.hpp"
#include "bench_render_templates.hpp"

// Small wrapper that initializes the OpenCL platform and compiles a given kernel file.
class OCLWrapper
{
public:
    OCLWrapper(cl_device_type deviceType, std::string deviceName, std::string kernelFilePath);
    OCLWrapper(const OCLWrapper &) = delete;
    OCLWrapper &operator=(const OCLWrapper &) = delete;
    OCLWrapper(const OCLWrapper &&) = delete;
    OCLWrapper &operator=(OCLWrapper &&) = delete;
    ~OCLWrapper();

    std::string kernelDir = "./";
    std::string deviceName = "";
    cl_device_type deviceType = CL_DEVICE_TYPE_DEFAULT;
    cl_device_id deviceId = nullptr;
    cl_context context = nullptr;
    cl_command_queue commands = nullptr;
    cl_program program = nullptr;

    int err;
};

TEST_CASE("benchmark var*var stencils optimizations", "[console][varvarkerneloptimizations]")
{
    int N = GENERATE(8, 16, 32, 64);
    // int N = 16;
    int m = N;
    int n = N;
    int o = N;

    int wa = 3;
    int wb = 3;
    int gha = 2;
    int ghb = 2;
    int ghc = 2;

    ankerl::nanobench::Bench bench;
    bench.timeUnit(1ms, "ms")
        // .epochs(1)
        // .epochIterations(1)
        .minEpochTime(100ms)
        .relative(false);

    // if (N >= 32)
    //     bench.epochs(1).epochIterations(2);

    // if (N >= 64)
    //     bench.epochs(1).epochIterations(1);

    mgcl_test::TestUtility tu;
    OCLWrapper oclw(CL_DEVICE_TYPE_GPU,
                    tu.deviceAvailable("Quadro", CL_DEVICE_TYPE_GPU) ? "Quadro" : "",
                    "./kernel_optimizations.cl");
    // bool gpuAvailable = tu.deviceAvailable("Quadro", CL_DEVICE_TYPE_GPU);
    // bool cpuAvailable = tu.deviceAvailable("", CL_DEVICE_TYPE_CPU);

    // mgcl::VaryingStencilGpu a(m, n, o, wa, gha, oclw.context, oclw.commands);
    // mgcl::VaryingStencilGpu b(m, n, o, wb, ghb, oclw.context, oclw.commands);
    // mgcl::VaryingStencilGpu c(m, n, o, wa + wb - 1, ghc, oclw.context, oclw.commands);

    // naive
    {
        int err;

        // Create the compute kernel from the program
        cl_kernel kernel = clCreateKernel(oclw.program, "mult_stencils_var_var", &err);
        mgcl::mgclCheckError(err, "clCreateKernel");

        // create output buffer c
        // mgcl::VaryingStencilGpu c(m, n, o, width + b.getWidth() - 1, ghc, context, queue);

        mgcl::VaryingStencilGpu a(m, n, o, wa, gha, oclw.context, oclw.commands);
        mgcl::VaryingStencilGpu b(m, n, o, wb, ghb, oclw.context, oclw.commands);
        mgcl::VaryingStencilGpu c(m, n, o, wa + wb - 1, ghc, oclw.context, oclw.commands);

        auto abuf = a.getBuf();
        auto bbuf = b.getBuf();
        auto cbuf = c.getBuf();
        // auto wb = b.getWidth();
        // auto ghb = b.getGh();

        // assign kernel arguments
        int pos = 0;
        err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &abuf);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &bbuf);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &cbuf);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &m);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &n);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &o);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &wa);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &wb);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &gha);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghb);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghc);
        mgcl::mgclCheckError(err, "Setting kernel arguments");

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
        std::string name = std::string("naive var*var, N = ").append(std::to_string(N));
        bench.run(std::string(name).c_str(), [&]
                  { 
            err = clEnqueueNDRangeKernel(oclw.commands, kernel, 3, NULL, global, local, 0, NULL, NULL);
            mgcl::mgclCheckError(err, "Enqueueing stencil multiplication kernel");
            clFinish(oclw.commands); });

        // update ghosts of c
        // if (ghc > 0)
        //     c.updateGhosts(oclw.program, oclw.commands);

        clReleaseKernel(kernel);
    }

    // naive branchless
    {
        int err;

        // Create the compute kernel from the program
        cl_kernel kernel = clCreateKernel(oclw.program, "mult_stencils_var_var_branchless", &err);
        mgcl::mgclCheckError(err, "clCreateKernel");

        // create output buffer c
        // mgcl::VaryingStencilGpu c(m, n, o, width + b.getWidth() - 1, ghc, context, queue);

        mgcl::VaryingStencilGpu a(m, n, o, wa, gha, oclw.context, oclw.commands);
        mgcl::VaryingStencilGpu b(m, n, o, wb, ghb, oclw.context, oclw.commands);
        mgcl::VaryingStencilGpu c(m, n, o, wa + wb - 1, ghc, oclw.context, oclw.commands);

        auto abuf = a.getBuf();
        auto bbuf = b.getBuf();
        auto cbuf = c.getBuf();
        // auto wb = b.getWidth();
        // auto ghb = b.getGh();

        // assign kernel arguments
        int pos = 0;
        err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &abuf);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &bbuf);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &cbuf);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &m);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &n);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &o);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &wa);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &wb);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &gha);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghb);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghc);
        mgcl::mgclCheckError(err, "Setting kernel arguments");

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
        std::string name = std::string("naive branchless var*var, N = ").append(std::to_string(N));
        bench.run(std::string(name).c_str(), [&]
                  { 
            err = clEnqueueNDRangeKernel(oclw.commands, kernel, 3, NULL, global, local, 0, NULL, NULL);
            mgcl::mgclCheckError(err, "Enqueueing stencil multiplication kernel");
            clFinish(oclw.commands); });

        // update ghosts of c
        // if (ghc > 0)
        //     c.updateGhosts(oclw.program, oclw.commands);

        clReleaseKernel(kernel);
    }

    // reordered for loops (only 1 write to global buffer c)
    {
        int err;

        // Create the compute kernel from the program
        cl_kernel kernel = clCreateKernel(oclw.program, "mult_stencils_var_var_reordered", &err);
        mgcl::mgclCheckError(err, "clCreateKernel");

        // create output buffer c
        // mgcl::VaryingStencilGpu c(m, n, o, width + b.getWidth() - 1, ghc, context, queue);

        mgcl::VaryingStencilGpu a(m, n, o, wa, gha, oclw.context, oclw.commands);
        mgcl::VaryingStencilGpu b(m, n, o, wb, ghb, oclw.context, oclw.commands);
        mgcl::VaryingStencilGpu c(m, n, o, wa + wb - 1, ghc, oclw.context, oclw.commands);

        auto abuf = a.getBuf();
        auto bbuf = b.getBuf();
        auto cbuf = c.getBuf();
        // auto wb = b.getWidth();
        // auto ghb = b.getGh();

        // assign kernel arguments
        int pos = 0;
        err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &abuf);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &bbuf);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &cbuf);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &m);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &n);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &o);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &wa);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &wb);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &gha);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghb);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghc);
        mgcl::mgclCheckError(err, "Setting kernel arguments");

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
        std::string name = std::string("reordered var*var, N = ").append(std::to_string(N));
        bench.run(std::string(name).c_str(), [&]
                  { 
            err = clEnqueueNDRangeKernel(oclw.commands, kernel, 3, NULL, global, local, 0, NULL, NULL);
            mgcl::mgclCheckError(err, "Enqueueing stencil multiplication kernel");
            clFinish(oclw.commands); });

        // update ghosts of c
        // if (ghc > 0)
        //     c.updateGhosts(oclw.program, oclw.commands);

        clReleaseKernel(kernel);
    }

    // reordered for loops (only 1 write to global buffer c) using min function instead of ternary operator
    {
        int err;

        // Create the compute kernel from the program
        cl_kernel kernel = clCreateKernel(oclw.program, "mult_stencils_var_var_reordered_minfn", &err);
        mgcl::mgclCheckError(err, "clCreateKernel");

        // create output buffer c
        // mgcl::VaryingStencilGpu c(m, n, o, width + b.getWidth() - 1, ghc, context, queue);

        mgcl::VaryingStencilGpu a(m, n, o, wa, gha, oclw.context, oclw.commands);
        mgcl::VaryingStencilGpu b(m, n, o, wb, ghb, oclw.context, oclw.commands);
        mgcl::VaryingStencilGpu c(m, n, o, wa + wb - 1, ghc, oclw.context, oclw.commands);

        auto abuf = a.getBuf();
        auto bbuf = b.getBuf();
        auto cbuf = c.getBuf();
        // auto wb = b.getWidth();
        // auto ghb = b.getGh();

        // assign kernel arguments
        int pos = 0;
        err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &abuf);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &bbuf);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &cbuf);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &m);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &n);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &o);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &wa);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &wb);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &gha);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghb);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghc);
        mgcl::mgclCheckError(err, "Setting kernel arguments");

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
        std::string name = std::string("reordered minfn var*var, N = ").append(std::to_string(N));
        bench.run(std::string(name).c_str(), [&]
                  { 
            err = clEnqueueNDRangeKernel(oclw.commands, kernel, 3, NULL, global, local, 0, NULL, NULL);
            mgcl::mgclCheckError(err, "Enqueueing stencil multiplication kernel");
            clFinish(oclw.commands); });

        // update ghosts of c
        // if (ghc > 0)
        //     c.updateGhosts(oclw.program, oclw.commands);

        clReleaseKernel(kernel);
    }
}

OCLWrapper::OCLWrapper(cl_device_type deviceType, std::string deviceName, std::string kernelFilePath)
{
    // OCLWrapper w;

    int i;
    cl_uint numPlatforms;
    cl_device_id device_id_;

    // Find number of platforms
    err = clGetPlatformIDs(0, nullptr, &numPlatforms);
    mgcl::mgclCheckError(err, "Finding platforms");
    if (numPlatforms == 0)
    {
        printf("Found 0 platforms!\n");
        // return w;
    }

    // Get all platforms
    cl_platform_id Platform[numPlatforms];
    err = clGetPlatformIDs(numPlatforms, Platform, nullptr);
    mgcl::mgclCheckError(err, "Getting platforms");

    cl_char device_name_available[1024] = {0}; // string to hold name of compute device

    // take first device that conforms given device_type and name
    for (i = 0; i < numPlatforms; i++)
    {
        err = clGetDeviceIDs(Platform[i], deviceType, 1, &device_id_, nullptr);
        if (err == CL_SUCCESS)
        {
            if (deviceName != "" && deviceName != "default")
            {
                err = clGetDeviceInfo(device_id_, CL_DEVICE_NAME, sizeof(device_name_available),
                                      &device_name_available, nullptr);
                mgcl::mgclCheckError(err, "clGetDeviceInfo(CL_DEVICE_NAME)");

                // continue to next device if name doesn't fit
                if (std::string((char *)device_name_available).find(deviceName) == std::string::npos)
                    continue;
            }

            deviceId = device_id_;
            break;
        }
    }

    if (deviceId == nullptr)
        mgcl::mgclCheckError(-1, "Finding a device");

    err = mgcl::OpenCLHelper::outputDeviceInfo(deviceId);
    mgcl::mgclCheckError(err, "Printing device output");

    // Create a compute context
    context = clCreateContext(0, 1, &deviceId, nullptr, nullptr, &err);
    mgcl::mgclCheckError(err, "Creating context");

    // Create a command queue
    commands = clCreateCommandQueue(context, deviceId, 0, &err);
    mgcl::mgclCheckError(err, "Creating command queue");

    // Update device type that is in use
    err = clGetDeviceInfo(deviceId, CL_DEVICE_TYPE, sizeof(deviceType), &deviceType, nullptr);
    mgcl::mgclCheckError(err, "clGetDeviceInfo(CL_DEVICE_NAME)");

    // read kernel source
    const char *KernelSource = mgcl::OpenCLHelper::loadKernelSource(kernelFilePath.c_str());
    if (KernelSource == nullptr)
        // return w;
        throw "kernel source is null!";

    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1, &KernelSource, nullptr, &err);
    mgcl::mgclCheckError(err, "Creating program");

    // Build the program
    err = clBuildProgram(program, 0, nullptr, "-cl-fast-relaxed-math", nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        // Determine the size of the log
        size_t log_size;
        clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);

        // Allocate memory for the log
        char *log = (char *)malloc(log_size);

        // Get the log
        clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG, log_size, log, nullptr);

        // Print the log
        printf("%s\n", log);

        free(log);
        // return w;
    }
}

OCLWrapper::~OCLWrapper()
{
    int err;

    if (program)
    {
        err = clReleaseProgram(program);
        mgcl::mgclCheckError(err, "clReleaseProgram");
        program = nullptr;
    }

    if (context)
    {
        err = clReleaseContext(context);
        mgcl::mgclCheckError(err, "clReleaseContext");
        context = nullptr;
    }

    if (commands)
    {
        err = clReleaseCommandQueue(commands);
        mgcl::mgclCheckError(err, "clReleaseCommandQueue");
        commands = nullptr;
    }

    if (deviceId)
    {
        err = clReleaseDevice(deviceId);
        mgcl::mgclCheckError(err, "clReleaseDevice");
        deviceId = nullptr;
    }
}