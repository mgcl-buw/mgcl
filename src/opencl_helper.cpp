#include "opencl_helper.hpp"
#include "cuboid.hpp"  // for
#include "level.hpp"   // for Level
#include "problem.hpp" // for Problem

#include <cassert>
#include <fstream>
#include <sstream>

#include <cstdio>  // for printf, size_t, NULL, fprintf, fclose, fopen
#include <cstdlib> // for malloc, exit, free, EXIT_FAILURE
#include <iostream>

#include <cpptrace/cpptrace.hpp>

namespace mgcl
{
    OpenCLHelper::~OpenCLHelper()
    {
        release();
    }

    /**
     * @brief Initializes or retains OpenCL Environment. If a new environment shall be created subsequently, release
     * must be called first.
     * @throws string If no platform was found.
     */
    void OpenCLHelper::init()
    {
        int err;
        cl_uint numPlatforms;
        cl_device_id device_id_;

        // initialize opencl stuff if not done yet and if buffers should not be reused
        if (!isInitialized() && !problem->getReuseOpenclBuffers() && !problem->getCopyBufferData())
        {
            // Find number of platforms
            err = clGetPlatformIDs(0, nullptr, &numPlatforms);
            mgclCheckError(err, "Finding platforms");
            if (numPlatforms == 0)
            {
                if (!problem->silent)
                    printf("Found 0 platforms!\n");
                throw "Found 0 platforms!";
            }

            // Get all platforms
            cl_platform_id Platform[numPlatforms];
            err = clGetPlatformIDs(numPlatforms, Platform, nullptr);
            mgclCheckError(err, "Getting platforms");

            char device_name_available[1024] = {0}; // string to hold name of compute device

            // take first device that conforms given device_type and name
            for (cl_uint i = 0; i < numPlatforms; i++)
            {
                err = clGetDeviceIDs(Platform[i], deviceType, 1, &device_id_, nullptr);
                if (err == CL_SUCCESS)
                {
                    if (deviceName != "" && deviceName != "default")
                    {
                        err = clGetDeviceInfo(device_id_, CL_DEVICE_NAME, sizeof(device_name_available),
                                              &device_name_available, nullptr);
                        mgclCheckError(err, "clGetDeviceInfo(CL_DEVICE_NAME)");

                        // continue to next device if name doesn't fit
                        if (std::string(device_name_available).find(deviceName) == std::string::npos)
                            continue;
                    }

                    deviceId = device_id_;
                    break;
                }
            }

            if (deviceId == nullptr)
                mgclCheckError(-1, "Finding a device");

            if (!problem->silent)
            {
                err = outputDeviceInfo(deviceId);
                mgclCheckError(err, "Printing device output");
            }

            // Create a compute context
            context = clCreateContext(0, 1, &deviceId, nullptr, nullptr, &err);
            mgclCheckError(err, "Creating context");

            // Create a command queue
            commands = clCreateCommandQueue(context, deviceId, 0, &err);
            mgclCheckError(err, "Creating command queue");
        }
        else
        {
            // retain OpenCL environment, i.e. make sure, everything is valid while reusing it in mgcl
            err = clRetainContext(context);
            mgclCheckError(err, "clRetainContext");

            err = clRetainCommandQueue(commands);
            mgclCheckError(err, "clRetainCommandQueue");

            err = clRetainDevice(deviceId);
            mgclCheckError(err, "clRetainDevice");
        }

        // Update device type that is in use
        err = clGetDeviceInfo(deviceId, CL_DEVICE_TYPE, sizeof(deviceType), &deviceType, nullptr);
        mgclCheckError(err, "clGetDeviceInfo(CL_DEVICE_NAME)");

        // read kernel source
        std::string filename = kernelDir + "mgcl.cl";
        std::string kernelSource = loadKernelSource(filename);
        const char* ksc = kernelSource.c_str();

        // Create the compute program from the source buffer
        program = clCreateProgramWithSource(context, 1, &ksc, nullptr, &err);
        mgclCheckError(err, "Creating program");

        // Build the program
        err = clBuildProgram(program, 0, nullptr, "-cl-fast-relaxed-math", nullptr, nullptr);
        if (err != CL_SUCCESS)
        {
            // Determine the size of the log
            size_t log_size;
            clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);

            // Allocate memory for the log
            char* log = static_cast<char*>(malloc(log_size));

            // Get the log
            clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG, log_size, log, nullptr);

            // Print the log
            if (!problem->silent)
                printf("%s\n", log);

            free(log);

            assert(err == CL_SUCCESS && "Building the kernel failed.");
        }
    }

    /**
     * @brief Releases OpenCL environment, i.e. program, context, commandQueue and deviceId.
     *
     * @return int OpenCL error code.
     */
    void OpenCLHelper::release()
    {
        int err;

        if (program)
        {
            err = clReleaseProgram(program);
            mgclCheckError(err, "clReleaseProgram");
            program = nullptr;
        }

        if (context)
        {
            err = clReleaseContext(context);
            mgclCheckError(err, "clReleaseContext");
            context = nullptr;
        }

        if (commands)
        {
            err = clReleaseCommandQueue(commands);
            mgclCheckError(err, "clReleaseCommandQueue");
            commands = nullptr;
        }

        if (deviceId)
        {
            err = clReleaseDevice(deviceId);
            mgclCheckError(err, "clReleaseDevice");
            deviceId = nullptr;
        }
    }

    /**
     * @brief Returns true if OpenCL platform is initialized, false otherwise.
     *
     */
    bool OpenCLHelper::isInitialized()
    {
        return context && commands && deviceId;
    }

    /**
     * @brief Checks if OpenCL-Parameters are valid (only useful if reuse_opencl_buffers || copy_buffer_data)
     *
     * @return true All good.
     * @return false Somethings nullptr or buffers d_v or d_f have wrong size if reuse_opencl_buffers.
     */
    bool OpenCLHelper::checkParameters()
    {
        if (problem->getReuseOpenclBuffers() || problem->getCopyBufferData())
        {
            if (problem->getDVPtr() == nullptr || problem->getDFPtr() == nullptr)
            {
                if (!problem->silent)
                    printf("OpenCL buffers d_v and d_f not set but reuse_opencl_buffers or copy_buffer_data specified. "
                           "Aborting.\n");
                return false;
            }

            if (deviceId == nullptr)
            {
                if (!problem->silent)
                    printf("reuse_opencl_buffers or copy_buffer_data specified but device ID (mgcl_config.device_id) not set. "
                           "Aborting.\n");
                return false;
            }

            if (commands == nullptr)
            {
                if (!problem->silent)
                    printf("reuse_opencl_buffers or copy_buffer_data specified but command queue (mgcl_config.commands) not "
                           "set. Aborting.\n");
                return false;
            }

            if (context == nullptr)
            {
                if (!problem->silent)
                    printf("reuse_opencl_buffers or copy_buffer_data specified but context (mgcl_config.context) not set. "
                           "Aborting.\n");
                return false;
            }
        }

        // check size of buffers
        if (problem->getReuseOpenclBuffers())
        {
            int m = problem->getM();
            int n = problem->getN();
            int o = problem->getO();
            int ghosts = problem->getGhosts();

            int sizeNeeded = (m + 2 * ghosts) * (n + 2 * ghosts) * (o + 2 * ghosts);
            if (problem->getDV().getSize() != sizeNeeded)
            {
                if (!problem->silent)
                    std::cout << "OpenCL buffer d_v has wrong size (" << problem->getDV().getSize() << " but need "
                              << sizeNeeded << ")" << std::endl;
                return false;
            }

            if (problem->getDF().getSize() != sizeNeeded)
            {
                if (!problem->silent)
                    std::cout << "OpenCL buffer d_f has wrong size (" << problem->getDF().getSize() << " but need "
                              << sizeNeeded << ")" << std::endl;
                return false;
            }
        }

        return true;
    }

    /**
     * @brief Copies given input buffers to newly created buffers. Will be used if Problem::copy_buffer_data is true.
     *
     * @return int OpenCL error code.
     */
    int OpenCLHelper::copyInputBuffers()
    {
        // TODO respect ghosts?
        int err;
        auto& level0 = problem->getLevelAt(0);
        int m = level0.getMgh();
        int n = level0.getNgh();
        int o = level0.getOgh();
        int ghosts_in = problem->getGhostsIn();

        // Create the compute kernel from the program
        cl_kernel kernel = clCreateKernel(program, "copy_input_data", &err);
        mgclCheckError(err, "Creating copy input data kernel");

        cl_mem d_pv = problem->getDV().getBuffer();
        cl_mem d_lv0v = level0.getDVIn().getBuffer();
        cl_mem d_pf = problem->getDF().getBuffer();
        cl_mem d_lv0f = level0.getDF().getBuffer();

        // assign kernel arguments
        int pos = 0;
        err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &d_pv);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &d_lv0v);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &d_pf);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &d_lv0f);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &m);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &n);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &o);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghosts_in);
        mgclCheckError(err, "Setting copy input data kernel arguments");

        // one work-item per cell (including ghost cells). Pad global sizes to fit to local sizes
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

        err = clEnqueueNDRangeKernel(commands, kernel, 3, NULL, global, local, 0, NULL, NULL);
        mgclCheckError(err, "Enqueueing copy input data kernel");

        clReleaseKernel(kernel);
        return err;
    }

    /**
     * @brief Copies buffers dV and dF from level 0 to given input buffers. Will be used if Problem::copy_buffer_data
     * is true.
     *
     * @return int OpenCL error code.
     */
    int OpenCLHelper::copyOutputBuffers()
    {
        int err;
        auto& level0 = problem->getLevelAt(0);
        int m = level0.m;
        int n = level0.n;
        int o = level0.o;
        int ghosts_in = problem->ghosts_in;

        // Create the compute kernel from the program
        cl_kernel kernel = clCreateKernel(program, "copy_output_data", &err);
        mgclCheckError(err, "Creating copy output data kernel");

        // assign kernel arguments
        int pos = 0;
        err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &problem->dV);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &level0.dVIn);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &m);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &n);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &o);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghosts_in);
        mgclCheckError(err, "Setting copy output data kernel arguments");

        // one work-item per cell (including ghost cells). Pad global sizes to fit to local sizes
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

        err = clEnqueueNDRangeKernel(commands, kernel, 3, NULL, global, local, 0, NULL, NULL);
        mgclCheckError(err, "Enqueueing copy output data kernel");

        clReleaseKernel(kernel);
        return err;
    }

    /**
     * @brief Reads buffer from device to host. Creates new buffer if out is nullptr.
     *
     * @param out Buffer that gets filled with device data. Might be nullptr.
     * @param d_buf Device buffer to be read from.
     * @param m Size of buffer.
     * @param n Size of buffer.
     * @param o Size of buffer.
     * @return int OpenCL error code.
     */
    Cuboid OpenCLHelper::readBuffer(cl_mem d_buf, int m, int n, int o)
    {
        int err;
        err = clFinish(commands);
        mgclCheckError(err, "Waiting for kernel to finish");

        Cuboid ret(m, n, o);
        double*** tmp = ret.getData();

        err = clEnqueueReadBuffer(commands, d_buf, CL_TRUE, 0, sizeof(double) * m * n * o, tmp[0][0], 0, NULL, NULL);
        mgclCheckError(err, "clEnqueueReadBuffer");

        return ret;
    }

    /**
     * @brief Reads buffer from device to host and prints it.
     *
     * @param d_buf Device buffer to be printed.
     * @param m Buffer size.
     * @param n Buffer size.
     * @param o Buffer size.
     */
    void OpenCLHelper::printBuffer(cl_mem d_buf, int m, int n, int o)
    {
        printf("printing buffer with size %d,%d,%d\n", m, n, o);
        auto c = readBuffer(d_buf, m, n, o);

        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < o; k++)
                {
                    printf("i,j,k, %d,%d,%d val = %f\n", i, j, k, c[i][j][k]);
                }
    }

    /**
     * @brief Calls clFinish on the command queue.
     */
    void OpenCLHelper::finish()
    {
        if (commands)
            mgclCheckError(clFinish(commands), "clFinish");
        else
            throw "Command queue is not initialized!";
    }

    /**
     * @brief Loads kernel source from file into std::string.
     *
     * @param file File to be read from.
     * @return std::string File contents.
     */
    std::string OpenCLHelper::loadKernelSource(std::string file)
    {
        std::ifstream t(file);
        std::stringstream buffer;
        buffer << t.rdbuf();
        return buffer.str();
    }

    /**
     * @brief Prints various OpenCL device information.
     *
     * @param device_id ID of device.
     * @return int OpenCL error code.
     */
    int OpenCLHelper::outputDeviceInfo(cl_device_id device_id)
    {
        int err;                         // error code returned from OpenCL calls
        cl_device_type device_type;      // Parameter defining the type of the compute device
        cl_uint comp_units;              // the max number of compute units on a device
        cl_char vendor_name[1024] = {0}; // string to hold vendor name for compute device
        cl_char device_name[1024] = {0}; // string to hold name of compute device

        err = clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_name), &device_name, NULL);
        mgclCheckError(err, "clGetDeviceInfo(CL_DEVICE_NAME)");
        printf("Using OpenCL device %s ", device_name);

        err = clGetDeviceInfo(device_id, CL_DEVICE_TYPE, sizeof(device_type), &device_type, NULL);
        mgclCheckError(err, "clGetDeviceInfo(CL_DEVICE_TYPE)");

        if (device_type == CL_DEVICE_TYPE_GPU)
            printf("GPU from ");

        else if (device_type == CL_DEVICE_TYPE_CPU)
            printf("\n CPU from ");

        else
            printf("\n non CPU or GPU processor from ");

        err = clGetDeviceInfo(device_id, CL_DEVICE_VENDOR, sizeof(vendor_name), &vendor_name, NULL);
        mgclCheckError(err, "clGetDeviceInfo(CL_DEVICE_VENDOR)");
        printf("%s", vendor_name);

        err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &comp_units, NULL);
        mgclCheckError(err, "clGetDeviceInfo(CL_DEVICE_MAX_COMPUTE_UNITS)");
        printf(" with a max of %d compute units \n", comp_units);

        return err;
    }

    const char* OpenCLHelper::mgcl_err_code(cl_int err_in)
    {
        switch (err_in)
        {
        case CL_SUCCESS:
            return (char*)"CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND:
            return (char*)"CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE:
            return (char*)"CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE:
            return (char*)"CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
            return (char*)"CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES:
            return (char*)"CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY:
            return (char*)"CL_OUT_OF_HOST_MEMORY";
        case CL_PROFILING_INFO_NOT_AVAILABLE:
            return (char*)"CL_PROFILING_INFO_NOT_AVAILABLE";
        case CL_MEM_COPY_OVERLAP:
            return (char*)"CL_MEM_COPY_OVERLAP";
        case CL_IMAGE_FORMAT_MISMATCH:
            return (char*)"CL_IMAGE_FORMAT_MISMATCH";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:
            return (char*)"CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case CL_BUILD_PROGRAM_FAILURE:
            return (char*)"CL_BUILD_PROGRAM_FAILURE";
        case CL_MAP_FAILURE:
            return (char*)"CL_MAP_FAILURE";
        case CL_MISALIGNED_SUB_BUFFER_OFFSET:
            return (char*)"CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
            return (char*)"CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case CL_INVALID_VALUE:
            return (char*)"CL_INVALID_VALUE";
        case CL_INVALID_DEVICE_TYPE:
            return (char*)"CL_INVALID_DEVICE_TYPE";
        case CL_INVALID_PLATFORM:
            return (char*)"CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE:
            return (char*)"CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT:
            return (char*)"CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES:
            return (char*)"CL_INVALID_QUEUE_PROPERTIES";
        case CL_INVALID_COMMAND_QUEUE:
            return (char*)"CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_HOST_PTR:
            return (char*)"CL_INVALID_HOST_PTR";
        case CL_INVALID_MEM_OBJECT:
            return (char*)"CL_INVALID_MEM_OBJECT";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
            return (char*)"CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case CL_INVALID_IMAGE_SIZE:
            return (char*)"CL_INVALID_IMAGE_SIZE";
        case CL_INVALID_SAMPLER:
            return (char*)"CL_INVALID_SAMPLER";
        case CL_INVALID_BINARY:
            return (char*)"CL_INVALID_BINARY";
        case CL_INVALID_BUILD_OPTIONS:
            return (char*)"CL_INVALID_BUILD_OPTIONS";
        case CL_INVALID_PROGRAM:
            return (char*)"CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE:
            return (char*)"CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME:
            return (char*)"CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL_DEFINITION:
            return (char*)"CL_INVALID_KERNEL_DEFINITION";
        case CL_INVALID_KERNEL:
            return (char*)"CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX:
            return (char*)"CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE:
            return (char*)"CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE:
            return (char*)"CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS:
            return (char*)"CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION:
            return (char*)"CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE:
            return (char*)"CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE:
            return (char*)"CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_GLOBAL_OFFSET:
            return (char*)"CL_INVALID_GLOBAL_OFFSET";
        case CL_INVALID_EVENT_WAIT_LIST:
            return (char*)"CL_INVALID_EVENT_WAIT_LIST";
        case CL_INVALID_EVENT:
            return (char*)"CL_INVALID_EVENT";
        case CL_INVALID_OPERATION:
            return (char*)"CL_INVALID_OPERATION";
        case CL_INVALID_GL_OBJECT:
            return (char*)"CL_INVALID_GL_OBJECT";
        case CL_INVALID_BUFFER_SIZE:
            return (char*)"CL_INVALID_BUFFER_SIZE";
        case CL_INVALID_MIP_LEVEL:
            return (char*)"CL_INVALID_MIP_LEVEL";
        case CL_INVALID_GLOBAL_WORK_SIZE:
            return (char*)"CL_INVALID_GLOBAL_WORK_SIZE";
        case CL_INVALID_PROPERTY:
            return (char*)"CL_INVALID_PROPERTY";

        default:
            return (char*)"UNKNOWN ERROR";
        }
    }

    void OpenCLHelper::mgcl_check_error(cl_int err, const char* operation, const char* filename, int line)
    {
        if (err != CL_SUCCESS)
        {
            fprintf(stderr, "Error during operation '%s', ", operation);
            fprintf(stderr, "in '%s' on line %d\n", filename, line);
            fprintf(stderr, "Error code was \"%s\" (%d)\n", mgcl_err_code(err), err);
            cpptrace::generate_trace().print();
            exit(EXIT_FAILURE);
        }
    }

    Problem* OpenCLHelper::getProblem() const
    {
        return problem;
    }

    cl_command_queue OpenCLHelper::getCommands() const
    {
        return commands;
    }

    void OpenCLHelper::setCommands(const cl_command_queue& commands_)
    {
        commands = commands_;
    }

    std::string OpenCLHelper::getDeviceName() const
    {
        return deviceName;
    }

    void OpenCLHelper::setDeviceName(const std::string& deviceName_)
    {
        deviceName = deviceName_;
    }

    std::string OpenCLHelper::getKernelDir() const
    {
        return kernelDir;
    }

    void OpenCLHelper::setKernelDir(const std::string& kernelDir_)
    {
        // TODO add trailing slash and test
        kernelDir = kernelDir_;
    }

    cl_device_type OpenCLHelper::getDeviceType() const
    {
        return deviceType;
    }

    void OpenCLHelper::setDeviceType(const cl_device_type& deviceType_)
    {
        deviceType = deviceType_;
    }

    cl_program OpenCLHelper::getProgram() const
    {
        return program;
    }

    void OpenCLHelper::setProgram(const cl_program& program_)
    {
        program = program_;
    }

    cl_context OpenCLHelper::getContext() const
    {
        return context;
    }

    void OpenCLHelper::setContext(const cl_context& context_)
    {
        context = context_;
    }

    cl_device_id OpenCLHelper::getDeviceId() const
    {
        return deviceId;
    }

    void OpenCLHelper::setDeviceId(const cl_device_id& deviceId_)
    {
        deviceId = deviceId_;
    }
}