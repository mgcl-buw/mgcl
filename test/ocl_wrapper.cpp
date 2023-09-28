#include "ocl_wrapper.hpp"
#include "../src/mgcl/opencl_helper.hpp"

OCLWrapper::OCLWrapper(cl_device_type deviceType, std::string deviceName, std::string kernelString,
                       std::string kernelFilePath, cl_context _context)
    : context(_context)
{
    int i;
    cl_uint numPlatforms;
    cl_device_id device_id_;

    // Query device id of given context, or find a device and create a context if none is given.
    if (context)
    {
        err = clRetainContext(context);
        mgcl::mgclCheckError(err, "clRetainContext");

        cl_uint numDevices;
        err = clGetContextInfo(context, CL_CONTEXT_NUM_DEVICES, sizeof(cl_uint), &numDevices, nullptr);
        mgcl::mgclCheckError(err, "CL_CONTEXT_NUM_DEVICES");

        cl_device_id devs[numDevices];
        err = clGetContextInfo(context, CL_CONTEXT_DEVICES, sizeof(cl_device_id) * numDevices, devs, nullptr);
        mgcl::mgclCheckError(err, "CL_CONTEXT_DEVICES");

        deviceId = devs[0];
    }
    else
    {
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
                    if (std::string((char*)device_name_available).find(deviceName) == std::string::npos)
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
    }

    // Create a command queue
    commands = clCreateCommandQueue(context, deviceId, 0, &err);
    mgcl::mgclCheckError(err, "Creating command queue");

    // Update device type that is in use
    err = clGetDeviceInfo(deviceId, CL_DEVICE_TYPE, sizeof(deviceType), &deviceType, nullptr);
    mgcl::mgclCheckError(err, "clGetDeviceInfo(CL_DEVICE_NAME)");

    // Read kernel source, if kernelString is not given
    if (kernelString == "")
        kernelString = mgcl::OpenCLHelper::loadKernelSource(kernelFilePath);

    const char* kernelSource = kernelString.c_str();

    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, &err);
    mgcl::mgclCheckError(err, "Creating program");

    // Build the program
    err = clBuildProgram(program, 0, nullptr, "-cl-fast-relaxed-math", nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        // Determine the size of the log
        size_t log_size;
        clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);

        // Allocate memory for the log
        char* log = (char*)malloc(log_size);

        // Get the log
        clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG, log_size, log, nullptr);

        // Print the log
        printf("%s\n", log);

        free(log);
        // return w;
    }
}

OCLWrapper::OCLWrapper(cl_device_type deviceType, std::string deviceName, std::string kernelString,
                       std::string kernelFilePath)
{
    OCLWrapper(deviceType, deviceName, kernelString, kernelFilePath, nullptr);
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
