#include "opencl_helper.hpp"
#include "cuboid.hpp" // for
#include "level.hpp"  // for Level
#include "mgcl.hpp"
#include "mgcl_kernel.hpp"
#include "problem.hpp" // for Problem

#include <CL/cl.h>
#include <cassert>
#include <fstream>
#include <iomanip>
#include <sstream>

#include <cstdio>  // for printf, size_t, NULL, fprintf, fclose, fopen
#include <cstdlib> // for malloc, exit, free, EXIT_FAILURE
#include <iostream>

#ifdef __APPLE__
#include <OpenCL/cl_ext.h>
#else
#include <CL/cl_ext.h>
#endif

// #include <cpptrace/cpptrace.hpp>

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
        cl_uint numDevices;
        cl_device_id* device_ids;
        cl_device_type _device_type = deviceType;

        // initialize opencl stuff if not done yet and if buffers should not be reused
        if (!isInitialized() && !problem->getReuseOpenclBuffers() && !problem->getCopyBufferData())
        {
            // Find number of platforms
            err = clGetPlatformIDs(0, nullptr, &numPlatforms);
            mgclCheckError(err, "Finding platforms");
            if (numPlatforms == 0)
            {
                error("No OpenCL platform was found!");
            }

            // Get all platforms
            cl_platform_id platforms[numPlatforms];
            err = clGetPlatformIDs(numPlatforms, platforms, nullptr);
            mgclCheckError(err, "Getting platforms");

            char device_name_available[1024] = {0}; // string to hold name of compute device

            // take first device that conforms given device_type and name
            for (cl_uint i = 0; i < numPlatforms; i++)
            {
                // Find number of devices for a platform
                err = clGetDeviceIDs(platforms[i], _device_type, 0, nullptr, &numDevices);
                if (err == CL_DEVICE_NOT_FOUND)
                {
                    continue; // no device with given type found in current platform
                }
                mgcl::mgclCheckError(err, "Finding devices using clGetDeviceIDs");

                device_ids = new cl_device_id[numDevices];
                err = clGetDeviceIDs(platforms[i], _device_type, numDevices, device_ids, nullptr);
                mgclCheckError(err, "clGetDeviceIDs");

                // Loop over all devices for a given plattform
                for (cl_uint j{0}; j < numDevices; ++j)
                {
                    // get device extensions and check for double precision
                    size_t numExtensions;
                    err = clGetDeviceInfo(device_ids[j], CL_DEVICE_EXTENSIONS, 0, nullptr, &numExtensions);
                    mgclCheckError(err, "Finding number of extensions using clGetDeviceInfo(CL_DEVICE_EXTENSIONS)");

                    // char* extensions = new char[numExtensions];
                    std::unique_ptr<char[]> extensions(new char[numExtensions]);
                    err = clGetDeviceInfo(device_ids[j], CL_DEVICE_EXTENSIONS, numExtensions, extensions.get(), nullptr);
                    mgclCheckError(err, "Finding extensions using clGetDeviceInfo(CL_DEVICE_EXTENSIONS)");

                    // if device does not support double precision, continue to next device
                    if (std::string(extensions.get()).find("cl_khr_fp64") == std::string::npos)
                        continue;

                    if (deviceName != "" && deviceName != "default")
                    {
                        err = clGetDeviceInfo(device_ids[j], CL_DEVICE_NAME,
                                              sizeof(device_name_available), &device_name_available,
                                              nullptr);
                        mgclCheckError(err, "Finding device name using clGetDeviceInfo(CL_DEVICE_NAME)");

                        // if device name does not match, continue to next device
                        if (std::string(device_name_available).find(deviceName) == std::string::npos)
                            continue;
                    }

                    deviceId = device_ids[j];
                    platformId = platforms[i];
                    deviceType = _device_type;
                    break;
                }

                delete[] device_ids;
            }

            if (deviceId == nullptr)
                mgclCheckError(-1, "Finding a device");

            if (!problem->silent)
            {
                err = outputDeviceInfo();
                mgclCheckError(err, "Printing device output");
            }

            // Create a compute context
            context = clCreateContext(0, 1, &deviceId, nullptr, nullptr, &err);
            mgclCheckError(err, "Creating context");

            // Create a command queue
            cl_command_queue_properties props = problem->isProfilingEnabled() ? CL_QUEUE_PROFILING_ENABLE : 0;
            commands = clCreateCommandQueue(context, deviceId, props, &err);
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
        mgclCheckError(err, "clGetDeviceInfo(CL_DEVICE_TYPE)");

        // if binaryPath is set, check if it exists and use it instead of recompiling
        std::ifstream fbin(binaryFile, std::ios::binary | std::ios::in);
        if (binaryFile != "" && fbin)
        {
            if (!problem->silent)
            {
                std::cout << "mgcl: Found and using binary file: " << binaryFile << std::endl;
            }

            // Create the compute program from the binary
            size_t binarySize;
            fbin.seekg(0, std::ios::end);
            binarySize = fbin.tellg();
            fbin.seekg(0, std::ios::beg);

            unsigned char* binary = new unsigned char[binarySize];
            fbin.read(reinterpret_cast<char*>(binary), binarySize);

            program = clCreateProgramWithBinary(context, 1, &deviceId, &binarySize,
                                                const_cast<const unsigned char**>(&binary), nullptr, &err);
            delete[] binary;
            mgclCheckError(err, "clCreateProgramWithBinary");
        }
        else
        {
            if (!problem->silent)
            {
                std::cout << "mgcl: No binary file given or not found. Building from source." << std::endl;

                if (!readKernelFromFile && kernelFile != "")
                {
                    std::cout << "mgcl: Warning: kernelFile was set, but readKernelFromFile is false." << std::endl;
                }
            }

            // read kernel source
            std::string kernelSource;
            const char* ksc;
            if (readKernelFromFile)
            {
                if (!problem->silent)
                    std::cout << "mgcl: Info: Building kernel from file: " << kernelFile << std::endl;
                kernelSource = loadKernelSource(kernelFile);
                ksc = kernelSource.c_str();
            }
            else
            {
                ksc = MGCL_KERNEL_SOURCE.c_str();
            }

            // Create the compute program from the source buffer
            program = clCreateProgramWithSource(context, 1, &ksc, nullptr, &err);
            mgclCheckError(err, "Creating program");
        }

        std::string options = "-cl-fast-relaxed-math " + preprocessorConstantsToString();
        if (!problem->silent)
        {
            std::cout << "mgcl: Building program with options: " << options << std::endl;
        }

        // Build the program
        err = clBuildProgram(program, 1, &deviceId, options.c_str(), nullptr, nullptr);
        // err = clBuildProgram(program, 1, &deviceId, "-cl-fast-relaxed-math -cl-nv-arch sm_75", nullptr, nullptr);
        if (err != CL_SUCCESS || problem->isPrintKernelLog())
        {
            // Determine the size of the log
            size_t log_size;
            clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);

            std::cout << "mgcl: Printing build log (" << log_size << " Bytes):" << std::endl;

            // Allocate memory for the log
            char* log = static_cast<char*>(malloc(log_size));

            // Get the log
            clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG, log_size, log, nullptr);

            // Print the log
            printf("%s\n", log);

            free(log);

            if (err != CL_SUCCESS)
            {
                assert(err == CL_SUCCESS && "Building the kernel failed.");
            }
        }

        // Save the program binary if binaryFile is not empty and binaryFile does not exist yet
        if (binaryFile != "" && !fbin && (!problem->useMpi() || problem->mpiRank() == 0))
        {
            if (!problem->silent)
            {
                std::cout << "mgcl: Saving binary file to: " << binaryFile << std::endl;
            }

            // Save the program binary to "test.bin"
            size_t binarySize;
            err = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &binarySize, nullptr);
            mgclCheckError(err, "clGetProgramInfo(CL_PROGRAM_BINARY_SIZES)");

            unsigned char* binary = new unsigned char[binarySize];
            err = clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(unsigned char*), &binary, nullptr);
            mgclCheckError(err, "clGetProgramInfo(CL_PROGRAM_BINARIES)");

            std::ofstream binaryFileOut(binaryFile, std::ios::out | std::ios::binary);
            binaryFileOut.write(reinterpret_cast<char*>(binary), binarySize);
            binaryFileOut.close();
            delete[] binary;
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
        const char* kernelName = "copy_input_data";
        cl_kernel kernel = clCreateKernel(program, kernelName, &err);
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
        const auto& c = conf::getWorkGroupSizeForKernelAndWiCount(problem->getKernelConfig(), kernelName, 1);
        const size_t local[3] = {
            static_cast<size_t>(m > c[0] ? c[0] : m),
            static_cast<size_t>(n > c[1] ? c[1] : n),
            static_cast<size_t>(o > c[2] ? c[2] : o)};

        for (int i = 0; i < 3; i++)
            if (global[i] % local[i] != 0)
                global[i] += local[i] - (global[i] % local[i]);

        cl_event ev;

        err = clEnqueueNDRangeKernel(commands, kernel, 3, NULL, global, local, 0, NULL, &ev);
        mgclCheckError(err, "Enqueueing copy input data kernel");

        if (problem->isProfilingEnabled())
        {
            problem->getProfilingData()->addMeasurement(problem->getCommands(), ev, kernelName,
                                                        {global[0], global[1], global[2]},
                                                        {local[0], local[1], local[2]});
        }
        mgclCheckError(clReleaseEvent(ev), "clReleaseEvent");

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
        const char* kernelName = "copy_output_data";
        cl_kernel kernel = clCreateKernel(program, kernelName, &err);
        mgclCheckError(err, "Creating copy output data kernel");

        cl_mem pdv = problem->dV->getBuffer();
        cl_mem lv0dv = level0.dVIn->getBuffer();

        // assign kernel arguments
        int pos = 0;
        err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &pdv);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &lv0dv);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &m);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &n);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &o);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghosts_in);
        mgclCheckError(err, "Setting copy output data kernel arguments");

        // one work-item per cell (including ghost cells). Pad global sizes to fit to local sizes
        size_t global[3] = {static_cast<size_t>(m), static_cast<size_t>(n), static_cast<size_t>(o)};
        const auto& c = conf::getWorkGroupSizeForKernelAndWiCount(problem->getKernelConfig(), kernelName, 1);
        const size_t local[3] = {
            static_cast<size_t>(m > c[0] ? c[0] : m),
            static_cast<size_t>(n > c[1] ? c[1] : n),
            static_cast<size_t>(o > c[2] ? c[2] : o)};

        for (int i = 0; i < 3; i++)
            if (global[i] % local[i] != 0)
                global[i] += local[i] - (global[i] % local[i]);

        cl_event ev;

        err = clEnqueueNDRangeKernel(commands, kernel, 3, NULL, global, local, 0, NULL, &ev);
        mgclCheckError(err, "Enqueueing copy output data kernel");

        if (problem->isProfilingEnabled())
        {
            problem->getProfilingData()->addMeasurement(problem->getCommands(), ev, kernelName,
                                                        {global[0], global[1], global[2]},
                                                        {local[0], local[1], local[2]});
        }
        mgclCheckError(clReleaseEvent(ev), "clReleaseEvent");

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
            error("Command queue is not initialized!");
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
     * @brief Prints details of the selected OpenCL platform and device.
     *
     * @return int OpenCL error code.
     */
    int OpenCLHelper::outputDeviceInfo()
    {
        int err;                           // error code returned from OpenCL calls
        cl_device_type device_type;        // Parameter defining the type of the compute device
        cl_uint comp_units;                // the max number of compute units on a device
        cl_char vendor_name[1024] = {0};   // string to hold vendor name for compute device
        cl_char device_name[1024] = {0};   // string to hold name of compute device
        cl_char platform_name[1024] = {0}; // string to hold name of compute device

        err = clGetPlatformInfo(platformId, CL_PLATFORM_NAME, sizeof(platform_name), &platform_name, NULL);
        mgclCheckError(err, "clGetPlatformInfo(CL_PLATFORM_NAME)");

        err = clGetDeviceInfo(deviceId, CL_DEVICE_NAME, sizeof(device_name), &device_name, NULL);
        mgclCheckError(err, "clGetDeviceInfo(CL_DEVICE_NAME)");
        printf("Using OpenCL platform %s with device %s ", platform_name, device_name);

        err = clGetDeviceInfo(deviceId, CL_DEVICE_TYPE, sizeof(device_type), &device_type, NULL);
        mgclCheckError(err, "clGetDeviceInfo(CL_DEVICE_TYPE)");

        if (device_type == CL_DEVICE_TYPE_GPU)
            printf("GPU from ");

        else if (device_type == CL_DEVICE_TYPE_CPU)
            printf("\n CPU from ");

        else
            printf("\n non CPU or GPU processor from ");

        err = clGetDeviceInfo(deviceId, CL_DEVICE_VENDOR, sizeof(vendor_name), &vendor_name, NULL);
        mgclCheckError(err, "clGetDeviceInfo(CL_DEVICE_VENDOR)");
        printf("%s", vendor_name);

        err = clGetDeviceInfo(deviceId, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &comp_units, NULL);
        mgclCheckError(err, "clGetDeviceInfo(CL_DEVICE_MAX_COMPUTE_UNITS)");
        printf(" with a max of %d compute units", comp_units);

#ifdef CL_DEVICE_UUID_KHR
        cl_uchar uuid[CL_UUID_SIZE_KHR];
        bool uuid_available = false;
        err = clGetDeviceInfo(deviceId, CL_DEVICE_UUID_KHR, sizeof(cl_uchar) * CL_UUID_SIZE_KHR,
                              &uuid, nullptr);
        uuid_available = err == CL_SUCCESS;
        if (uuid_available)
        {
            printf(", uuid: %02x%02x%02x%02x-"
                   "%02x%02x-"
                   "%02x%02x-"
                   "%02x%02x-"
                   "%02x%02x%02x%02x%02x%02x\n",
                   uuid[0], uuid[1], uuid[2], uuid[3], uuid[4],
                   uuid[5], uuid[6],
                   uuid[7], uuid[8],
                   uuid[9], uuid[10],
                   uuid[11], uuid[12], uuid[13], uuid[14], uuid[15]);
        }
        else
        {
            err = CL_SUCCESS;
            printf("\n");
        }
#else
        printf("\n");
#endif

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
            // cpptrace::generate_trace().print();
            // error("Error");
            exit(EXIT_FAILURE);
        }
    }

    Problem* OpenCLHelper::getProblem() const
    {
        return problem;
    }

    std::string OpenCLHelper::getBinaryFile() const
    {
        return binaryFile;
    }

    void OpenCLHelper::setBinaryFile(const std::string& binaryFile_)
    {
        binaryFile = binaryFile_;
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

    std::string OpenCLHelper::getKernelFile() const
    {
        return kernelFile;
    }

    /**
     * @brief Sets kernel file. Throws if file does not exist. Check is done by trying to open the file.
     */
    void OpenCLHelper::setKernelFile(const std::string& kernelFile_)
    {
        // Check if file actually exists by trying to open it
        std::ifstream f;
        f.open(kernelFile_);
        if (f.fail())
            error(std::string("Kernel file '").append(kernelFile_).append("' does not exist."));
        f.close();

        kernelFile = kernelFile_;
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

    /**
     * @brief Returns set preprocessor constants as a string, e.g. "-DBLOCKSIZE=2"
     */
    std::string OpenCLHelper::preprocessorConstantsToString()
    {
        std::string result = "";
        for (auto it = preprocessorConstants.begin(); it != preprocessorConstants.end(); it++)
        {
            result += " -D" + it->first + "=" + it->second;
        }
        return result;
    }

    std::string OpenCLHelper::deviceTypeToString(cl_device_type dt)
    {
        if (dt == CL_DEVICE_TYPE_GPU)
            return "CL_DEVICE_TYPE_GPU";
        else if (dt == CL_DEVICE_TYPE_CPU)
            return "CL_DEVICE_TYPE_CPU";
        else if (dt == CL_DEVICE_TYPE_ACCELERATOR)
            return "CL_DEVICE_TYPE_ACCELERATOR";
        else if (dt == CL_DEVICE_TYPE_DEFAULT)
            return "CL_DEVICE_TYPE_DEFAULT";
        else if (dt == CL_DEVICE_TYPE_ALL)
            return "CL_DEVICE_TYPE_ALL";
        else
            return "UNKNOWN DEVICE TYPE";
    }

    /**
     * @brief Searches all platforms and devices and prints them
     *
     */
    std::string OpenCLHelper::availableDevicesInfo()
    {
        int err;
        cl_uint numPlatforms;
        cl_uint numDevices;
        cl_device_id* device_ids;
        cl_device_type _device_type = CL_DEVICE_TYPE_ALL;

        // Find number of platforms
        err = clGetPlatformIDs(0, nullptr, &numPlatforms);
        mgclCheckError(err, "Finding platforms");
        if (numPlatforms == 0)
        {
            std::cout << "No OpenCL platform found." << std::endl;
            return "";
        }

        std::stringstream ss;

        // Get all platforms
        cl_platform_id platforms[numPlatforms];
        err = clGetPlatformIDs(numPlatforms, platforms, nullptr);
        mgclCheckError(err, "Getting platforms");

        char device_name_available[1024] = {0}; // string to hold name of compute device

        // take first device that conforms given device_type and name
        for (cl_uint i = 0; i < numPlatforms; i++)
        {
            // Get platform name
            char platform_name[1024] = {0};
            err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platform_name), platform_name, nullptr);
            mgclCheckError(err, "Getting platform name");
            ss << "Platform " << i << ": " << platform_name << std::endl;

            // Find number of devices for a platform
            err = clGetDeviceIDs(platforms[i], _device_type, 0, nullptr, &numDevices);
            if (err == CL_DEVICE_NOT_FOUND)
            {
                continue; // no device with given type found in current platform
            }
            mgcl::mgclCheckError(err, "Finding devices using clGetDeviceIDs");

            device_ids = new cl_device_id[numDevices];
            err = clGetDeviceIDs(platforms[i], _device_type, numDevices, device_ids, nullptr);
            mgclCheckError(err, "clGetDeviceIDs");

            // Loop over all devices for a given plattform
            for (cl_uint j{0}; j < numDevices; ++j)
            {

                err = clGetDeviceInfo(device_ids[j], CL_DEVICE_NAME,
                                      sizeof(device_name_available), &device_name_available,
                                      nullptr);
                mgclCheckError(err, "Finding device name using clGetDeviceInfo(CL_DEVICE_NAME)");

                cl_device_type dt;
                err = clGetDeviceInfo(device_ids[j], CL_DEVICE_TYPE, sizeof(cl_device_type), &dt, nullptr);
                mgclCheckError(err, "Finding device type using clGetDeviceInfo(CL_DEVICE_TYPE)");

#ifdef CL_DEVICE_UUID_KHR
                cl_uchar uuid[CL_UUID_SIZE_KHR];
                bool uuid_available = false;
                err = clGetDeviceInfo(deviceId, CL_DEVICE_UUID_KHR, sizeof(cl_uchar) * CL_UUID_SIZE_KHR,
                                      &uuid, nullptr);
                uuid_available = err == CL_SUCCESS;
                std::string uuid_str = "uuid: ";
                if (uuid_available)
                {
                    std::stringstream ss2;
                    ss2 << std::hex << std::setfill('0');
                    ss2 << std::setw(2) << static_cast<int>(uuid[0])
                        << std::setw(2) << static_cast<int>(uuid[1])
                        << std::setw(2) << static_cast<int>(uuid[2])
                        << std::setw(2) << static_cast<int>(uuid[3]) << '-'
                        << std::setw(2) << static_cast<int>(uuid[4])
                        << std::setw(2) << static_cast<int>(uuid[5]) << '-'
                        << std::setw(2) << static_cast<int>(uuid[6])
                        << std::setw(2) << static_cast<int>(uuid[7]) << '-'
                        << std::setw(2) << static_cast<int>(uuid[8])
                        << std::setw(2) << static_cast<int>(uuid[9]) << '-'
                        << std::setw(2) << static_cast<int>(uuid[10])
                        << std::setw(2) << static_cast<int>(uuid[11])
                        << std::setw(2) << static_cast<int>(uuid[12])
                        << std::setw(2) << static_cast<int>(uuid[13])
                        << std::setw(2) << static_cast<int>(uuid[14])
                        << std::setw(2) << static_cast<int>(uuid[15]);

                    uuid_str += ss2.str(); // result holds the formatted UUID string
                }
                else
                {
                    uuid_str += "N/A";
                }
#else
                std::string uuid_str;
#endif

                ss << "- Device id " << j << ": " << device_name_available << ", " << deviceTypeToString(dt) << ", " << uuid_str << std::endl;
            }

            delete[] device_ids;
        }

        return ss.str();
    }
}
