#include "test_utility.hpp"

#include "../mgcl.hpp"

#include <iostream>
#include <stdexcept>

mgcl_test::TestUtility::TestUtility()
{
    problem = std::make_shared<mgcl::Problem>(2, 2, 2);
    problem->setSilent(true);
    problem->initOpenCL();
}

mgcl_test::TestUtility::TestUtility(std::string deviceName)
{
    if (!deviceAvailable(deviceName, CL_DEVICE_TYPE_ALL))
        throw std::runtime_error("OpenCL device is not available. deviceName: " + deviceName);

    problem = std::make_shared<mgcl::Problem>(2, 2, 2);
    problem->setSilent(true);
    problem->setDeviceName(deviceName);
    problem->initOpenCL();
}

mgcl_test::TestUtility::TestUtility(cl_device_type deviceType)
{
    if (!deviceAvailable("", deviceType))
    {
        std::string typeName = "CL_DEVICE_TYPE_DEFAULT";
        if (deviceType == CL_DEVICE_TYPE_CPU)
            typeName = "CL_DEVICE_TYPE_CPU";
        else if (deviceType == CL_DEVICE_TYPE_GPU)
            typeName = "CL_DEVICE_TYPE_GPU";
        else if (deviceType == CL_DEVICE_TYPE_ACCELERATOR)
            typeName = "CL_DEVICE_TYPE_ACCELERATOR";
        throw std::runtime_error("OpenCL device is not available. deviceType: " + typeName);
    }

    problem = std::make_shared<mgcl::Problem>(2, 2, 2);
    problem->setSilent(true);
    problem->setDeviceType(deviceType);
    problem->initOpenCL();
}

mgcl_test::TestUtility::TestUtility(std::shared_ptr<mgcl::Problem> problem_)
    : problem(std::shared_ptr<mgcl::Problem>(problem_))
{
    problem->setSilent(true);
    problem->initOpenCL();
}

mgcl_test::TestUtility::~TestUtility()
{
    for (auto buf : openclBuffers)
        clReleaseMemObject(buf);
}

/**
 * @brief Creates an OpenCL buffer with content of c.
 *
 * @param c
 * @return cl_mem
 */
cl_mem mgcl_test::TestUtility::createOpenCLBuffer(mgcl::Cuboid &c)
{
    int err;
    cl_mem buf = clCreateBuffer(problem->getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                sizeof(double) * c.getMgh() * c.getNgh() * c.getOgh(), c.field1d().data(), &err);
    mgcl::mgclCheckError(err, "clCreateBuffer");
    openclBuffers.push_back(buf);
    return buf;
}

/**
 * @brief Waits for command queue to finish and reads a buffer into a Cuboid.
 *
 * @param buf
 * @return mgcl::Cuboid
 */

/**
 * @brief Waits for command queue to finish and reads a buffer into a Cuboid.
 *
 * @param buf OpenCL buffer to be read from.
 * @param m Amount of real grid cells in x-direction.
 * @param n Amount of real grid cells in y-direction.
 * @param o Amount of real grid cells in z-direction.
 * @param ghosts_m Amount of ghost cells in one x-direction.
 * @param ghosts_n Amount of ghost cells in one y-direction.
 * @param ghosts_o Amount of ghost cells in one z-direction.
 * @return mgcl::Cuboid
 */
mgcl::Cuboid mgcl_test::TestUtility::readOpenCLBuffer(cl_mem buf, int m, int n, int o, int ghosts_m, int ghosts_n, int ghosts_o)
{
    if (!buf)
        throw std::invalid_argument("Can't read OpenCL buffer, buf is null!");

    finish();

    mgcl::Cuboid c(m, n, o, ghosts_m, ghosts_n, ghosts_o);
    int size = c.getMgh() * c.getNgh() * c.getOgh();
    double *tmp = c.field1d().data();

    int err;
    err = clEnqueueReadBuffer(problem->getCommands(), buf, CL_TRUE, 0,
                              sizeof(double) * size, tmp, 0, NULL, NULL);
    mgcl::mgclCheckError(err, "clEnqueueReadBuffer");

    return c;
}

int mgcl_test::TestUtility::finish()
{
    int err = clFinish(problem->getCommands());
    mgcl::mgclCheckError(err, "clFinish");
    return err;
}

/**
 * @brief Returns true if a device whose name does contain deviceName is available on this machine. Code taken from
 * mgcl::OpenCLHelper.
 *
 */
bool mgcl_test::TestUtility::deviceAvailable(std::string deviceName, cl_device_type deviceType)
{
    int err, i;
    cl_uint numPlatforms;
    cl_device_id device_id_;

    // Find number of platforms
    err = clGetPlatformIDs(0, nullptr, &numPlatforms);
    mgcl::mgclCheckError(err, "Finding platforms");
    if (numPlatforms == 0)
    {
        printf("Found 0 platforms!\n");
        return false;
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

            err = clGetDeviceInfo(device_id_, CL_DEVICE_NAME, sizeof(device_name_available),
                                  &device_name_available, nullptr);
            if (err != CL_SUCCESS)
            {
                printf("Error: Failed to access device name!\n");
                return false;
            }

            // return true if a device was found
            if (std::string((char *)device_name_available).find(deviceName) != std::string::npos)
                return true;
        }
    }

    return false;
}

cl_context mgcl_test::TestUtility::getContext()
{
    return problem->getContext();
}

mgcl::Problem &mgcl_test::TestUtility::getProblem()
{
    return *problem;
}

cl_command_queue mgcl_test::TestUtility::getCommands()
{
    return problem->getCommands();
}

cl_device_id mgcl_test::TestUtility::getDeviceId()
{
    return problem->getDeviceId();
}
