#include "test_utility.hpp"

#include "../src/mgcl/mgcl.hpp"

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
        mgcl::mgclCheckError(clReleaseMemObject(buf), "clReleaseMemObject");
}

/**
 * @brief Creates an OpenCL buffer with content of c.
 *
 * @param c
 * @return cl_mem
 */
cl_mem mgcl_test::TestUtility::createOpenCLBuffer(mgcl::Cuboid& c)
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
std::unique_ptr<mgcl::Cuboid> mgcl_test::TestUtility::readOpenCLBuffer(cl_mem buf, int m, int n, int o, int ghosts_m, int ghosts_n, int ghosts_o)
{
    if (!buf)
        throw std::invalid_argument("Can't read OpenCL buffer, buf is null!");

    finish();

    auto c = std::make_unique<mgcl::Cuboid>(m, n, o, ghosts_m, ghosts_n, ghosts_o);
    int size = c->getMgh() * c->getNgh() * c->getOgh();
    double* tmp = c->field1d().data();

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
 * @brief Releases all buffers created by this TestUtility instance and clears buffer vector.
 *
 */
void mgcl_test::TestUtility::releaseBuffers()
{
    for (auto buf : openclBuffers)
        mgcl::mgclCheckError(clReleaseMemObject(buf), "clReleaseMemObject");
    openclBuffers.clear();
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
    cl_uint numDevices;
    cl_device_id* device_ids;

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
    	// Find number of devices for a platform
    	err = clGetDeviceIDs(Platform[i], deviceType, 0, nullptr, &numDevices);
    	mgcl::mgclCheckError(err, "Finding devices");
    	device_ids = new cl_device_id[numDevices];
        err = clGetDeviceIDs(Platform[i], deviceType, numDevices, device_ids, nullptr);
        if (err == CL_SUCCESS) for (int i{0}; i<numDevices; ++i)
        {
            err = clGetDeviceInfo(device_ids[i], CL_DEVICE_NAME, sizeof(device_name_available),
                                  &device_name_available, nullptr);
            if (err != CL_SUCCESS)
            {
                printf("Error: Failed to access device name!\n");
                return false;
            }

            // return true if a device was found
            if (std::string((char*)device_name_available).find(deviceName) != std::string::npos)
                return true;
        }
        delete[] device_ids;
    }

    return false;
}

cl_context mgcl_test::TestUtility::getContext()
{
    return problem->getContext();
}

mgcl::Problem& mgcl_test::TestUtility::getProblem()
{
    return *problem;
}

cl_command_queue mgcl_test::TestUtility::getCommands()
{
    return problem->getCommands();
}

cl_program mgcl_test::TestUtility::getProgram()
{
    return problem->getProgram();
}

cl_device_id mgcl_test::TestUtility::getDeviceId()
{
    return problem->getDeviceId();
}

/**************************************************************
 * End TestUtility class
 **************************************************************/

/**
 * @brief Fills cuboids with a 4th order periodic problem. Cuboids sizes must match.
 *
 * @param v
 * @param f
 * @param solution
 */
void mgcl_test::create4hOrderPeriodicProblem(mgcl::Cuboid& v, mgcl::Cuboid& f, mgcl::Cuboid& solution)
{
    if (v.getM() != f.getM() || v.getN() != f.getN() || v.getO() != f.getO() ||
        v.getM() != solution.getM() || v.getN() != solution.getN() || v.getO() != solution.getO())
        throw "Dimensions must match!";

    double h = 1.0 / (double)v.getM();
    for (int i = 0, i_v = v.getGhostsM(), i_f = f.getGhostsM(), i_s = solution.getGhostsM(); i_v < v.getM(); i++, i_v++, i_f++, i_s++)
        for (int j = 0, j_v = v.getGhostsN(), j_f = f.getGhostsN(), j_s = solution.getGhostsN(); j_v < v.getN(); j++, j_v++, j_f++, j_s++)
            for (int k = 0, k_v = v.getGhostsO(), k_f = f.getGhostsO(), k_s = solution.getGhostsO(); k_v < v.getO(); k++, k_v++, k_f++, k_s++)
            {
                double zs = i * h;
                double ys = j * h;
                double xs = k * h;
                double xs2 = xs * xs;
                double ys2 = ys * ys;
                double zs2 = zs * zs;
                double xsm1_2 = (xs - 1) * (xs - 1);
                double ysm1_2 = (ys - 1) * (ys - 1);
                double zsm1_2 = (zs - 1) * (zs - 1);
                double xs3 = xs * xs * xs;
                double ys3 = ys * ys * ys;
                double zs3 = zs * zs * zs;
                double xsm1_3 = (xs - 1) * (xs - 1) * (xs - 1);
                double ysm1_3 = (ys - 1) * (ys - 1) * (ys - 1);
                double zsm1_3 = (zs - 1) * (zs - 1) * (zs - 1);
                double xs4 = xs * xs * xs * xs;
                double ys4 = ys * ys * ys * ys;
                double zs4 = zs * zs * zs * zs;
                double xsm1_4 = (xs - 1) * (xs - 1) * (xs - 1) * (xs - 1);
                double ysm1_4 = (ys - 1) * (ys - 1) * (ys - 1) * (ys - 1);
                double zsm1_4 = (zs - 1) * (zs - 1) * (zs - 1) * (zs - 1);
                v[i_v][j_v][k_v] = 0;
                solution[i_s][j_s][k_s] = 1000000 * (xs * (xs - 1)) * (xs * (xs - 1)) * (xs * (xs - 1)) * (xs * (xs - 1)) *
                                          (ys * (ys - 1)) * (ys * (ys - 1)) * (ys * (ys - 1)) * (ys * (ys - 1)) *
                                          (zs * (zs - 1)) * (zs * (zs - 1)) * (zs * (zs - 1)) * (zs * (zs - 1));
                f[i_f][j_f][k_f] =
                    -1000000 *
                    (12 * xs4 * ys4 * zs4 * xsm1_4 * ysm1_4 * zsm1_2 + 12 * xs4 * ys4 * zs4 * xsm1_4 * ysm1_2 * zsm1_4 +
                     12 * xs4 * ys4 * zs4 * xsm1_2 * ysm1_4 * zsm1_4 + 32 * xs4 * ys4 * zs3 * xsm1_4 * ysm1_4 * zsm1_3 +
                     12 * xs4 * ys4 * zs2 * xsm1_4 * ysm1_4 * zsm1_4 + 32 * xs4 * ys3 * zs4 * xsm1_4 * ysm1_3 * zsm1_4 +
                     12 * xs4 * ys2 * zs4 * xsm1_4 * ysm1_4 * zsm1_4 + 32 * xs3 * ys4 * zs4 * xsm1_3 * ysm1_4 * zsm1_4 +
                     12 * xs2 * ys4 * zs4 * xsm1_4 * ysm1_4 * zsm1_4);
            }
}

void mgcl_test::fill7pLaplace(mgcl::VaryingStencil& v, double h, bool negativeCenter)
{
    double f = negativeCenter ? -1.0 : 1.0;
    double h2inv = f * (1.0 / (h * h));
    for (int i = 0; i < v.getMgh(); i++)
        for (int j = 0; j < v.getNgh(); j++)
            for (int k = 0; k < v.getOgh(); k++)
            {
                // 7-point Laplace
                v[0][1][1][i][j][k] = h2inv * -1.0;
                v[1][0][1][i][j][k] = h2inv * -1.0;
                v[1][1][0][i][j][k] = h2inv * -1.0;
                v[1][1][1][i][j][k] = h2inv * 6.0;
                v[1][1][2][i][j][k] = h2inv * -1.0;
                v[1][2][1][i][j][k] = h2inv * -1.0;
                v[2][1][1][i][j][k] = h2inv * -1.0;
            }
}
