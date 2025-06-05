#include "test_utility.hpp"

#include "../src/mgcl/mgcl.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

mgcl_test::TestUtility::TestUtility()
{
    problem = std::make_shared<mgcl::Problem>(2, 2, 2);
    problem->setSilent(true);
    problem->initOpenCL();
    problem->printDeviceInfo();
}

mgcl_test::TestUtility::TestUtility(std::string deviceName)
{
    if (!deviceAvailable(deviceName, CL_DEVICE_TYPE_ALL))
        throw std::runtime_error("OpenCL device is not available. deviceName: " + deviceName);

    problem = std::make_shared<mgcl::Problem>(2, 2, 2);
    problem->setSilent(true);
    problem->setDeviceName(deviceName);
    problem->initOpenCL();
    problem->printDeviceInfo();
}

mgcl_test::TestUtility::TestUtility(cl_device_type deviceType) : mgcl_test::TestUtility(deviceType, false) {}

mgcl_test::TestUtility::TestUtility(cl_device_type deviceType, bool profilingEnabled)
    : profilingEnabled(profilingEnabled)
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
    problem->setProfilingEnabled(profilingEnabled);
    problem->initOpenCL();
    problem->printDeviceInfo();
}

mgcl_test::TestUtility::TestUtility(std::shared_ptr<mgcl::Problem> problem_)
    : problem(std::shared_ptr<mgcl::Problem>(problem_))
{
    problem->setSilent(true);
    problem->initOpenCL();
    problem->printDeviceInfo();
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
 * @brief Creates an OpenCL buffer with content of c.
 *
 * @param c
 * @return cl_mem
 */
cl_mem mgcl_test::TestUtility::createOpenCLBuffer(mgcl::CuboidBS& c)
{
    int err;
    cl_mem buf = clCreateBuffer(problem->getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                sizeof(double) * c.getSize(), c.field1d().data(), &err);
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
    mgcl::mgclCheckError(err, "clGetPlatformIDs numPlatforms");
    if (numPlatforms == 0)
    {
        printf("Found 0 platforms!\n");
        return false;
    }

    // Get all platforms
    cl_platform_id Platform[numPlatforms];
    err = clGetPlatformIDs(numPlatforms, Platform, nullptr);
    mgcl::mgclCheckError(err, "clGetPlatformIDs platforms");

    cl_char device_name_available[1024] = {0}; // string to hold name of compute device

    // take first device that conforms given device_type and name
    for (i = 0; i < numPlatforms; i++)
    {
        // Find number of devices for a platform
        err = clGetDeviceIDs(Platform[i], deviceType, 0, nullptr, &numDevices);
        if (err == CL_DEVICE_NOT_FOUND)
        {
            continue; // no device with given type found in current platform
        }
        mgcl::mgclCheckError(err, "clgetdeviceids numdevices for devicetype");
        device_ids = new cl_device_id[numDevices];
        err = clGetDeviceIDs(Platform[i], deviceType, numDevices, device_ids, nullptr);
        mgcl::mgclCheckError(err, "clGetDeviceIDs device_ids for deviceType");
        if (err == CL_SUCCESS)
        {
            for (int j = 0; j < numDevices; j++)
            {
                err = clGetDeviceInfo(device_ids[j], CL_DEVICE_NAME, sizeof(device_name_available),
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

void mgcl_test::fill7pLaplace(mgcl::Blockstencil& v, double h, bool negativeCenter)
{
    double f = negativeCenter ? -1.0 : 1.0;
    double h2inv = f * (1.0 / (h * h));
    for (size_t b = 0; b < v.getBlocksize(); b++)
        for (int i = 0; i < v.getMgh(); i++)
            for (int j = 0; j < v.getNgh(); j++)
                for (int k = 0; k < v.getOgh(); k++)
                {
                    // 7-point Laplace
                    v[b][b][0][1][1][i][j][k] = h2inv * -1.0;
                    v[b][b][1][0][1][i][j][k] = h2inv * -1.0;
                    v[b][b][1][1][0][i][j][k] = h2inv * -1.0;
                    v[b][b][1][1][1][i][j][k] = h2inv * 6.0;
                    v[b][b][1][1][2][i][j][k] = h2inv * -1.0;
                    v[b][b][1][2][1][i][j][k] = h2inv * -1.0;
                    v[b][b][2][1][1][i][j][k] = h2inv * -1.0;
                }
}

// 27p Laplace stencil taken from "A Family of Large-Stencil Discrete Laplacian Approximations in
// Three Dimensions", O'Reilly and Beck, 2006
void mgcl_test::fill27pLaplace(mgcl::Blockstencil& v, double h, bool negativeCenter)
{
    double factor = 1.0 / (26.0 * h * h); // TODO use actual hs
    if (negativeCenter)
        factor *= -1.0;

    for (size_t b = 0; b < v.getBlocksize(); b++)
        for (int i = 0; i < v.getMgh(); i++)
            for (int j = 0; j < v.getNgh(); j++)
                for (int k = 0; k < v.getOgh(); k++)
                {
                    // 27-point Laplace
                    // center
                    v[b][b][1][1][1][i][j][k] = factor * 88.0;

                    // adjacent to center
                    v[b][b][0][1][1][i][j][k] = factor * -6.0;
                    v[b][b][1][0][1][i][j][k] = factor * -6.0;
                    v[b][b][1][1][0][i][j][k] = factor * -6.0;
                    v[b][b][1][1][2][i][j][k] = factor * -6.0;
                    v[b][b][1][2][1][i][j][k] = factor * -6.0;
                    v[b][b][2][1][1][i][j][k] = factor * -6.0;

                    // diagonally adjacent to center
                    v[b][b][1][0][0][i][j][k] = -3.0 * factor;
                    v[b][b][1][0][2][i][j][k] = -3.0 * factor;
                    v[b][b][1][2][0][i][j][k] = -3.0 * factor;
                    v[b][b][1][2][2][i][j][k] = -3.0 * factor;
                    v[b][b][0][1][0][i][j][k] = -3.0 * factor;
                    v[b][b][0][1][2][i][j][k] = -3.0 * factor;
                    v[b][b][2][1][0][i][j][k] = -3.0 * factor;
                    v[b][b][2][1][2][i][j][k] = -3.0 * factor;
                    v[b][b][0][0][1][i][j][k] = -3.0 * factor;
                    v[b][b][0][2][1][i][j][k] = -3.0 * factor;
                    v[b][b][2][0][1][i][j][k] = -3.0 * factor;
                    v[b][b][2][2][1][i][j][k] = -3.0 * factor;

                    // corners
                    v[b][b][0][0][0][i][j][k] = -2.0 * factor;
                    v[b][b][0][0][2][i][j][k] = -2.0 * factor;
                    v[b][b][0][2][0][i][j][k] = -2.0 * factor;
                    v[b][b][0][2][2][i][j][k] = -2.0 * factor;
                    v[b][b][2][0][0][i][j][k] = -2.0 * factor;
                    v[b][b][2][0][2][i][j][k] = -2.0 * factor;
                    v[b][b][2][2][0][i][j][k] = -2.0 * factor;
                    v[b][b][2][2][2][i][j][k] = -2.0 * factor;
                }
}

void mgcl_test::fill7pLaplace(mgcl::FixedStencil& v, double h, bool negativeCenter)
{
    double f = negativeCenter ? -1.0 : 1.0;
    double h2inv = f * (1.0 / (h * h));
    // 7-point Laplace
    v[0][1][1] = h2inv * -1.0;
    v[1][0][1] = h2inv * -1.0;
    v[1][1][0] = h2inv * -1.0;
    v[1][1][1] = h2inv * 6.0;
    v[1][1][2] = h2inv * -1.0;
    v[1][2][1] = h2inv * -1.0;
    v[2][1][1] = h2inv * -1.0;
}

void mgcl_test::fill27pLaplace(mgcl::FixedStencil& v, double h, bool negativeCenter)
{
    double factor = 1.0 / (26.0 * h * h); // TODO use actual hs
    if (negativeCenter)
        factor *= -1.0;

    // 27-point Laplace
    // center
    v[1][1][1] = factor * 88.0;

    // adjacent to center
    v[0][1][1] = factor * -6.0;
    v[1][0][1] = factor * -6.0;
    v[1][1][0] = factor * -6.0;
    v[1][1][2] = factor * -6.0;
    v[1][2][1] = factor * -6.0;
    v[2][1][1] = factor * -6.0;

    // diagonally adjacent to center
    v[1][0][0] = -3.0 * factor;
    v[1][0][2] = -3.0 * factor;
    v[1][2][0] = -3.0 * factor;
    v[1][2][2] = -3.0 * factor;
    v[0][1][0] = -3.0 * factor;
    v[0][1][2] = -3.0 * factor;
    v[2][1][0] = -3.0 * factor;
    v[2][1][2] = -3.0 * factor;
    v[0][0][1] = -3.0 * factor;
    v[0][2][1] = -3.0 * factor;
    v[2][0][1] = -3.0 * factor;
    v[2][2][1] = -3.0 * factor;

    // corners
    v[0][0][0] = -2.0 * factor;
    v[0][0][2] = -2.0 * factor;
    v[0][2][0] = -2.0 * factor;
    v[0][2][2] = -2.0 * factor;
    v[2][0][0] = -2.0 * factor;
    v[2][0][2] = -2.0 * factor;
    v[2][2][0] = -2.0 * factor;
    v[2][2][2] = -2.0 * factor;
}

void mgcl_test::fill19pLaplace(mgcl::VaryingStencil& v, double h, bool negativeCenter)
{
    double factor = 1.0 / (6.0 * h * h); // TODO use actual hs
    if (negativeCenter)
        factor *= -1.0;

    for (int i = 0; i < v.getMgh(); i++)
        for (int j = 0; j < v.getNgh(); j++)
            for (int k = 0; k < v.getOgh(); k++)
            {
                // 27-point Laplace
                // center
                v[1][1][1][i][j][k] = factor * 24.0;

                // adjacent to center
                v[0][1][1][i][j][k] = factor * -2.0;
                v[1][0][1][i][j][k] = factor * -2.0;
                v[1][1][0][i][j][k] = factor * -2.0;
                v[1][1][2][i][j][k] = factor * -2.0;
                v[1][2][1][i][j][k] = factor * -2.0;
                v[2][1][1][i][j][k] = factor * -2.0;

                // diagonally adjacent to center
                v[1][0][0][i][j][k] = -factor;
                v[1][0][2][i][j][k] = -factor;
                v[1][2][0][i][j][k] = -factor;
                v[1][2][2][i][j][k] = -factor;
                v[0][1][0][i][j][k] = -factor;
                v[0][1][2][i][j][k] = -factor;
                v[2][1][0][i][j][k] = -factor;
                v[2][1][2][i][j][k] = -factor;
                v[0][0][1][i][j][k] = -factor;
                v[0][2][1][i][j][k] = -factor;
                v[2][0][1][i][j][k] = -factor;
                v[2][2][1][i][j][k] = -factor;
            }
}

// 27p Laplace stencil taken from "A Family of Large-Stencil Discrete Laplacian Approximations in
// Three Dimensions", O'Reilly and Beck, 2006
void mgcl_test::fill27pLaplace(mgcl::VaryingStencil& v, double h, bool negativeCenter)
{
    double factor = 1.0 / (26.0 * h * h); // TODO use actual hs
    if (negativeCenter)
        factor *= -1.0;

    for (int i = 0; i < v.getMgh(); i++)
        for (int j = 0; j < v.getNgh(); j++)
            for (int k = 0; k < v.getOgh(); k++)
            {
                // 27-point Laplace
                // center
                v[1][1][1][i][j][k] = factor * 88.0;

                // adjacent to center
                v[0][1][1][i][j][k] = factor * -6.0;
                v[1][0][1][i][j][k] = factor * -6.0;
                v[1][1][0][i][j][k] = factor * -6.0;
                v[1][1][2][i][j][k] = factor * -6.0;
                v[1][2][1][i][j][k] = factor * -6.0;
                v[2][1][1][i][j][k] = factor * -6.0;

                // diagonally adjacent to center
                v[1][0][0][i][j][k] = -3.0 * factor;
                v[1][0][2][i][j][k] = -3.0 * factor;
                v[1][2][0][i][j][k] = -3.0 * factor;
                v[1][2][2][i][j][k] = -3.0 * factor;
                v[0][1][0][i][j][k] = -3.0 * factor;
                v[0][1][2][i][j][k] = -3.0 * factor;
                v[2][1][0][i][j][k] = -3.0 * factor;
                v[2][1][2][i][j][k] = -3.0 * factor;
                v[0][0][1][i][j][k] = -3.0 * factor;
                v[0][2][1][i][j][k] = -3.0 * factor;
                v[2][0][1][i][j][k] = -3.0 * factor;
                v[2][2][1][i][j][k] = -3.0 * factor;

                // corners
                v[0][0][0][i][j][k] = -2.0 * factor;
                v[0][0][2][i][j][k] = -2.0 * factor;
                v[0][2][0][i][j][k] = -2.0 * factor;
                v[0][2][2][i][j][k] = -2.0 * factor;
                v[2][0][0][i][j][k] = -2.0 * factor;
                v[2][0][2][i][j][k] = -2.0 * factor;
                v[2][2][0][i][j][k] = -2.0 * factor;
                v[2][2][2][i][j][k] = -2.0 * factor;
            }
}

/**
 * @brief Calculates error for each cell, e.g. difference between solution and approximation. Dimensions must match.
 *
 * @param solution
 * @param approximation
 * @return std::shared_ptr<mgcl::Cuboid>
 */
std::shared_ptr<mgcl::Cuboid> mgcl_test::calculateError(mgcl::Cuboid& solution, mgcl::Cuboid& approximation)
{
    if (solution.getM() != approximation.getM() ||
        solution.getN() != approximation.getN() ||
        solution.getO() != approximation.getO())
        throw std::invalid_argument("Dimensions do not match.");

    auto ret = std::make_shared<mgcl::Cuboid>(solution.getM(), solution.getN(), solution.getO());
    for (int i = 0, is = solution.getGhostsM(), ia = approximation.getGhostsM(); is < solution.getMgh(); i++, is++, ia++)
        for (int j = 0, js = solution.getGhostsN(), ja = approximation.getGhostsN(); js < solution.getNgh(); j++, js++, ja++)
            for (int k = 0, ks = solution.getGhostsO(), ka = approximation.getGhostsO(); ks < solution.getOgh(); k++, ks++, ka++)
            {
                (*ret)[i][j][k] = fabs(solution[is][js][ks] - approximation[ia][ja][ka]);
            }

    return ret;
}

/**
 * @brief Returns the maximum absolute error. calculateError should have been called first.
 *
 * @param error
 * @return double
 */
double mgcl_test::calculateMaxError(mgcl::Cuboid& error)
{
    double max = 0;
    for (int i = 0; i < error.getM(); i++)
        for (int j = 0; j < error.getN(); j++)
            for (int k = 0; k < error.getO(); k++)
            {
                if (max < error[i][j][k])
                    max = error[i][j][k];
            }
    return max;
}

/**
 * @brief Returns the 2-norm of the given error. calculateError should have been called first.
 *
 * @param h width of one cell
 * @param error precalculated error per cell
 * @return double Error norm of form ||e||_2 * h^3
 */
double mgcl_test::calculateErrorNorm(double h, mgcl::Cuboid& error)
{
    double sum = 0;

    for (int i = 0; i < error.getM(); i++)
        for (int j = 0; j < error.getN(); j++)
            for (int k = 0; k < error.getO(); k++)
            {
                sum += error[i][j][k] * error[i][j][k];
            }

    return sqrt(sum * h * h * h);
}

/**
 * @brief Creates and returns a fixed 3d full-weight restriction stencil (27p).
 *
 * @return std::unique_ptr<FixedStencil>
 */
void mgcl_test::fill3dFullWeightRestrictionBlockstencil(mgcl::FixedBlockstencil& bs)
{
    double factor1 = 1.0 / 64.0; // corner
    double factor2 = 2.0 / 64.0; // diagonally off
    double factor4 = 4.0 / 64.0; // adjacent
    double factor8 = 8.0 / 64.0; // center

    for (size_t b = 0; b < bs.getBlocksize(); b++)
    {
        bs[b][b][0][0][0] = factor1;
        bs[b][b][0][0][1] = factor2;
        bs[b][b][0][0][2] = factor1;
        bs[b][b][0][1][0] = factor2;
        bs[b][b][0][1][1] = factor4;
        bs[b][b][0][1][2] = factor2;
        bs[b][b][0][2][0] = factor1;
        bs[b][b][0][2][1] = factor2;
        bs[b][b][0][2][2] = factor1;
        bs[b][b][1][0][0] = factor2;
        bs[b][b][1][0][1] = factor4;
        bs[b][b][1][0][2] = factor2;
        bs[b][b][1][1][0] = factor4;
        bs[b][b][1][1][1] = factor8;
        bs[b][b][1][1][2] = factor4;
        bs[b][b][1][2][0] = factor2;
        bs[b][b][1][2][1] = factor4;
        bs[b][b][1][2][2] = factor2;
        bs[b][b][2][0][0] = factor1;
        bs[b][b][2][0][1] = factor2;
        bs[b][b][2][0][2] = factor1;
        bs[b][b][2][1][0] = factor2;
        bs[b][b][2][1][1] = factor4;
        bs[b][b][2][1][2] = factor2;
        bs[b][b][2][2][0] = factor1;
        bs[b][b][2][2][1] = factor2;
        bs[b][b][2][2][2] = factor1;
    }
}

/**
 * @brief Creates and returns a fixed 3d bilinear prolongation stencil (27p).
 *
 * @return std::unique_ptr<FixedStencil>
 */
void mgcl_test::fill3dBilinearProlongationBlockstencil(mgcl::FixedBlockstencil& bs)
{
    double factor1 = 1.0 / 8.0; // corner
    double factor2 = 1.0 / 4.0; // diagonally off
    double factor4 = 1.0 / 2.0; // adjacent
    double factor8 = 1.0;       // center

    for (size_t b = 0; b < bs.getBlocksize(); b++)
    {
        bs[b][b][0][0][0] = factor1;
        bs[b][b][0][0][1] = factor2;
        bs[b][b][0][0][2] = factor1;
        bs[b][b][0][1][0] = factor2;
        bs[b][b][0][1][1] = factor4;
        bs[b][b][0][1][2] = factor2;
        bs[b][b][0][2][0] = factor1;
        bs[b][b][0][2][1] = factor2;
        bs[b][b][0][2][2] = factor1;
        bs[b][b][1][0][0] = factor2;
        bs[b][b][1][0][1] = factor4;
        bs[b][b][1][0][2] = factor2;
        bs[b][b][1][1][0] = factor4;
        bs[b][b][1][1][1] = factor8;
        bs[b][b][1][1][2] = factor4;
        bs[b][b][1][2][0] = factor2;
        bs[b][b][1][2][1] = factor4;
        bs[b][b][1][2][2] = factor2;
        bs[b][b][2][0][0] = factor1;
        bs[b][b][2][0][1] = factor2;
        bs[b][b][2][0][2] = factor1;
        bs[b][b][2][1][0] = factor2;
        bs[b][b][2][1][1] = factor4;
        bs[b][b][2][1][2] = factor2;
        bs[b][b][2][2][0] = factor1;
        bs[b][b][2][2][1] = factor2;
        bs[b][b][2][2][2] = factor1;
    }
}

/**
 * @brief Copies values of a Cuboid into a CuboidBS, where the scaling factors determine the scaling between both inputs.
 * E.g.
 * scale{m,n,o} = 1: Cuboid m,n,o = CuboidBS m,n,o (blocksize 1)
 * scale{m,n,o} = 2: Cuboid m,n,o = 1/2 CuboidBS m,n,o (blocksize 8)
 * scale{m,n,o} = 4: Cuboid m,n,o = 1/4 CuboidBS m,n,o (blocksize 64)
 * scalem = 2, scalen = 1, scaleo = 4: Cuboid m = 1/2 CuboidBS m; Cuboid n = CuboidBS n; Cuboid o = 1/4 CuboidBS o; blocksize = 2*1*4=8
 *
 * Fills only real cells.
 *
 * @param src
 * @param dst
 * @param scalem
 * @param scalen
 * @param scaleo
 */
void mgcl_test::copyCuboidToCuboidBS(mgcl::Cuboid& src, mgcl::CuboidBS& dst, int scalem, int scalen, int scaleo)
{
    if (src.getM() != dst.getM() * scalem || src.getN() != dst.getN() * scalen || src.getO() != dst.getO() * scaleo)
    {
        throw "mgcl_test::copyCuboidToCuboidBS(): Cuboid dimensions do not match!\n  src m,n,o: " + std::to_string(src.getM()) + "," + std::to_string(src.getN()) + "," + std::to_string(src.getO()) + "\n  dst m,n,o: " + std::to_string(dst.getM()) + "," + std::to_string(dst.getN()) + "," + std::to_string(dst.getO());
    }

    // fill v with values of v1 and v2, vice versa for f
    for (int i = dst.getGhostsM(), i2 = src.getGhostsM(); i < dst.getM() + dst.getGhostsM(); i++, i2 += scalem)
        for (int j = dst.getGhostsN(), j2 = src.getGhostsN(); j < dst.getN() + dst.getGhostsN(); j++, j2 += scalen)
            for (int k = dst.getGhostsO(), k2 = src.getGhostsO(); k < dst.getO() + dst.getGhostsO(); k++, k2 += scaleo)
            {
                int b = 0;
                for (int ii = 0; ii < scalem; ii++)
                    for (int jj = 0; jj < scalen; jj++)
                        for (int kk = 0; kk < scaleo; kk++)
                        {
                            dst[i][j][k][b++] = src[i2 + ii][j2 + jj][k2 + kk];
                        }
            }
}

/**
 * @brief Copies values of a CuboidBS into a Cuboid, where the scaling factors determine the scaling between both inputs.
 * E.g.
 * scale{m,n,o} = 1: Cuboid m,n,o = CuboidBS m,n,o (blocksize 1)
 * scale{m,n,o} = 2: Cuboid m,n,o = 1/2 CuboidBS m,n,o (blocksize 8)
 * scale{m,n,o} = 4: Cuboid m,n,o = 1/4 CuboidBS m,n,o (blocksize 64)
 * scalem = 2, scalen = 1, scaleo = 4: Cuboid m = 1/2 CuboidBS m; Cuboid n = CuboidBS n; Cuboid o = 1/4 CuboidBS o; blocksize = 2*1*4=8
 *
 * Fills only real cells.
 *
 * @param src
 * @param dst
 * @param scalem
 * @param scalen
 * @param scaleo
 */
void mgcl_test::copyCuboidBSToCuboid(mgcl::CuboidBS& src, mgcl::Cuboid& dst, int scalem, int scalen, int scaleo)
{
    if (src.getM() * scalem != dst.getM() || src.getN() * scalen != dst.getN() || src.getO() * scaleo != dst.getO())
    {
        throw "mgcl_test::copyCuboidBSToCuboid(): Cuboid dimensions do not match!\n  src m,n,o: " + std::to_string(src.getM()) + "," + std::to_string(src.getN()) + "," + std::to_string(src.getO()) + "\n  dst m,n,o: " + std::to_string(dst.getM()) + "," + std::to_string(dst.getN()) + "," + std::to_string(dst.getO());
    }

    // fill v with values of v1 and v2, vice versa for f
    for (int i = src.getGhostsM(), i2 = dst.getGhostsM(); i < src.getM() + src.getGhostsM(); i++, i2 += scalem)
        for (int j = src.getGhostsN(), j2 = dst.getGhostsN(); j < src.getN() + src.getGhostsN(); j++, j2 += scalen)
            for (int k = src.getGhostsO(), k2 = dst.getGhostsO(); k < src.getO() + src.getGhostsO(); k++, k2 += scaleo)
            {
                int b = 0;
                for (int ii = 0; ii < scalem; ii++)
                    for (int jj = 0; jj < scalen; jj++)
                        for (int kk = 0; kk < scaleo; kk++)
                        {
                            dst[i2 + ii][j2 + jj][k2 + kk] = src[i][j][k][b++];
                        }
            }
}

void mgcl_test::fillVaryingStencilFromFixedStencil(mgcl::VaryingStencil& bs, mgcl::FixedStencil& fs)
{
    for (int i = bs.getGhostsM(); i < bs.getM() + bs.getGhostsM(); i++)
        for (int j = bs.getGhostsN(); j < bs.getN() + bs.getGhostsN(); j++)
            for (int k = bs.getGhostsO(); k < bs.getO() + bs.getGhostsO(); k++)
                for (int ii = 0; ii < fs.getWidth(); ii++)
                    for (int jj = 0; jj < fs.getWidth(); jj++)
                        for (int kk = 0; kk < fs.getWidth(); kk++)
                        {
                            bs[ii][jj][kk][i][j][k] = fs[ii][jj][kk];
                        }
    {
    }
}

void mgcl_test::fillBlockstencilFromFixedStencil(mgcl::Blockstencil& bs, mgcl::FixedStencil& fs)
{
    double ftl = fs[0][0][0];
    double ftc = fs[0][0][1];
    double ftr = fs[0][0][2];
    double fcl = fs[0][1][0];
    double fcc = fs[0][1][1];
    double fcr = fs[0][1][2];
    double fbl = fs[0][2][0];
    double fbc = fs[0][2][1];
    double fbr = fs[0][2][2];
    double ctl = fs[1][0][0];
    double ctc = fs[1][0][1];
    double ctr = fs[1][0][2];
    double ccl = fs[1][1][0];
    double ccc = fs[1][1][1];
    double ccr = fs[1][1][2];
    double cbl = fs[1][2][0];
    double cbc = fs[1][2][1];
    double cbr = fs[1][2][2];
    double btl = fs[2][0][0];
    double btc = fs[2][0][1];
    double btr = fs[2][0][2];
    double bcl = fs[2][1][0];
    double bcc = fs[2][1][1];
    double bcr = fs[2][1][2];
    double bbl = fs[2][2][0];
    double bbc = fs[2][2][1];
    double bbr = fs[2][2][2];

    // fill blockstencil with values from fs1
    for (int i = bs.getGhostsM(); i < bs.getM() + bs.getGhostsM(); i++)
        for (int j = bs.getGhostsN(); j < bs.getN() + bs.getGhostsN(); j++)
            for (int k = bs.getGhostsO(); k < bs.getO() + bs.getGhostsO(); k++)
            {
                // ***** front *****
                // b_ftl
                bs[0][7][0][0][0][i][j][k] = ftl;

                // b_ftc
                bs[0][6][0][0][1][i][j][k] = ftc;
                bs[0][7][0][0][1][i][j][k] = ftr;
                bs[1][6][0][0][1][i][j][k] = ftl;
                bs[1][7][0][0][1][i][j][k] = ftc;

                // b_ftr
                bs[1][6][0][0][2][i][j][k] = ftr;

                // b_fcl
                bs[0][5][0][1][0][i][j][k] = fcl;
                bs[0][7][0][1][0][i][j][k] = fbl;
                bs[2][5][0][1][0][i][j][k] = ftl;
                bs[2][7][0][1][0][i][j][k] = fcl;

                // b_fcc
                bs[0][4][0][1][1][i][j][k] = fcc;
                bs[0][5][0][1][1][i][j][k] = fcr;
                bs[0][6][0][1][1][i][j][k] = fbc;
                bs[0][7][0][1][1][i][j][k] = fbr;
                bs[1][4][0][1][1][i][j][k] = fcl;
                bs[1][5][0][1][1][i][j][k] = fcc;
                bs[1][6][0][1][1][i][j][k] = fbl;
                bs[1][7][0][1][1][i][j][k] = fbc;
                bs[2][4][0][1][1][i][j][k] = ftc;
                bs[2][5][0][1][1][i][j][k] = ftr;
                bs[2][6][0][1][1][i][j][k] = fcc;
                bs[2][7][0][1][1][i][j][k] = fcr;
                bs[3][4][0][1][1][i][j][k] = ftl;
                bs[3][5][0][1][1][i][j][k] = ftc;
                bs[3][6][0][1][1][i][j][k] = fcl;
                bs[3][7][0][1][1][i][j][k] = fcc;

                // b_fcr
                bs[1][4][0][1][2][i][j][k] = fcr;
                bs[1][6][0][1][2][i][j][k] = fbr;
                bs[3][4][0][1][2][i][j][k] = ftr;
                bs[3][6][0][1][2][i][j][k] = fcr;

                // b_fbl
                bs[2][5][0][2][0][i][j][k] = fbl;

                // b_fbc
                bs[2][4][0][2][1][i][j][k] = fbc;
                bs[2][5][0][2][1][i][j][k] = fbr;
                bs[3][4][0][2][1][i][j][k] = fbl;
                bs[3][5][0][2][1][i][j][k] = fbc;

                // b_fbr
                bs[3][4][0][2][2][i][j][k] = fbr;

                // ***** center *****
                // b_ctl
                bs[0][3][1][0][0][i][j][k] = ctl;
                bs[0][7][1][0][0][i][j][k] = btl;
                bs[4][3][1][0][0][i][j][k] = ftl;
                bs[4][7][1][0][0][i][j][k] = ctl;

                // b_ctc
                bs[0][2][1][0][1][i][j][k] = ctc;
                bs[0][3][1][0][1][i][j][k] = ctr;
                bs[0][6][1][0][1][i][j][k] = btc;
                bs[0][7][1][0][1][i][j][k] = btr;
                bs[1][2][1][0][1][i][j][k] = ctl;
                bs[1][3][1][0][1][i][j][k] = ctc;
                bs[1][6][1][0][1][i][j][k] = btl;
                bs[1][7][1][0][1][i][j][k] = btc;
                bs[4][2][1][0][1][i][j][k] = ftc;
                bs[4][3][1][0][1][i][j][k] = ftr;
                bs[4][6][1][0][1][i][j][k] = ctc;
                bs[4][7][1][0][1][i][j][k] = ctr;
                bs[5][2][1][0][1][i][j][k] = ftl;
                bs[5][3][1][0][1][i][j][k] = ftc;
                bs[5][6][1][0][1][i][j][k] = ctl;
                bs[5][7][1][0][1][i][j][k] = ctc;

                // b_ctr
                bs[1][2][1][0][2][i][j][k] = ctr;
                bs[1][6][1][0][2][i][j][k] = btr;
                bs[5][2][1][0][2][i][j][k] = ftr;
                bs[5][6][1][0][2][i][j][k] = ctr;

                // b_ccl
                bs[0][1][1][1][0][i][j][k] = ccl;
                bs[0][3][1][1][0][i][j][k] = cbl;
                bs[0][5][1][1][0][i][j][k] = bcl;
                bs[0][7][1][1][0][i][j][k] = bbl;
                bs[2][1][1][1][0][i][j][k] = ctl;
                bs[2][3][1][1][0][i][j][k] = ccl;
                bs[2][5][1][1][0][i][j][k] = btl;
                bs[2][7][1][1][0][i][j][k] = bcl;
                bs[4][1][1][1][0][i][j][k] = fcl;
                bs[4][3][1][1][0][i][j][k] = fbl;
                bs[4][5][1][1][0][i][j][k] = ccl;
                bs[4][7][1][1][0][i][j][k] = cbl;
                bs[6][1][1][1][0][i][j][k] = ftl;
                bs[6][3][1][1][0][i][j][k] = fcl;
                bs[6][5][1][1][0][i][j][k] = ctl;
                bs[6][7][1][1][0][i][j][k] = ccl;

                // b_ccc
                bs[0][0][1][1][1][i][j][k] = ccc;
                bs[0][1][1][1][1][i][j][k] = ccr;
                bs[0][2][1][1][1][i][j][k] = cbc;
                bs[0][3][1][1][1][i][j][k] = cbr;
                bs[0][4][1][1][1][i][j][k] = bcc;
                bs[0][5][1][1][1][i][j][k] = bcr;
                bs[0][6][1][1][1][i][j][k] = bbc;
                bs[0][7][1][1][1][i][j][k] = bbr;
                bs[1][0][1][1][1][i][j][k] = ccl;
                bs[1][1][1][1][1][i][j][k] = ccc;
                bs[1][2][1][1][1][i][j][k] = cbl;
                bs[1][3][1][1][1][i][j][k] = cbc;
                bs[1][4][1][1][1][i][j][k] = bcl;
                bs[1][5][1][1][1][i][j][k] = bcc;
                bs[1][6][1][1][1][i][j][k] = bbl;
                bs[1][7][1][1][1][i][j][k] = bbc;
                bs[2][0][1][1][1][i][j][k] = ctc;
                bs[2][1][1][1][1][i][j][k] = ctr;
                bs[2][2][1][1][1][i][j][k] = ccc;
                bs[2][3][1][1][1][i][j][k] = ccr;
                bs[2][4][1][1][1][i][j][k] = btc;
                bs[2][5][1][1][1][i][j][k] = btr;
                bs[2][6][1][1][1][i][j][k] = bcc;
                bs[2][7][1][1][1][i][j][k] = bcr;
                bs[3][0][1][1][1][i][j][k] = ctl;
                bs[3][1][1][1][1][i][j][k] = ctc;
                bs[3][2][1][1][1][i][j][k] = ccl;
                bs[3][3][1][1][1][i][j][k] = ccc;
                bs[3][4][1][1][1][i][j][k] = btl;
                bs[3][5][1][1][1][i][j][k] = btc;
                bs[3][6][1][1][1][i][j][k] = bcl;
                bs[3][7][1][1][1][i][j][k] = bcc;
                bs[4][0][1][1][1][i][j][k] = fcc;
                bs[4][1][1][1][1][i][j][k] = fcr;
                bs[4][2][1][1][1][i][j][k] = fbc;
                bs[4][3][1][1][1][i][j][k] = fbr;
                bs[4][4][1][1][1][i][j][k] = ccc;
                bs[4][5][1][1][1][i][j][k] = ccr;
                bs[4][6][1][1][1][i][j][k] = cbc;
                bs[4][7][1][1][1][i][j][k] = cbr;
                bs[5][0][1][1][1][i][j][k] = fcl;
                bs[5][1][1][1][1][i][j][k] = fcc;
                bs[5][2][1][1][1][i][j][k] = fbl;
                bs[5][3][1][1][1][i][j][k] = fbc;
                bs[5][4][1][1][1][i][j][k] = ccl;
                bs[5][5][1][1][1][i][j][k] = ccc;
                bs[5][6][1][1][1][i][j][k] = cbl;
                bs[5][7][1][1][1][i][j][k] = cbc;
                bs[6][0][1][1][1][i][j][k] = ftc;
                bs[6][1][1][1][1][i][j][k] = ftr;
                bs[6][2][1][1][1][i][j][k] = fcc;
                bs[6][3][1][1][1][i][j][k] = fcr;
                bs[6][4][1][1][1][i][j][k] = ctc;
                bs[6][5][1][1][1][i][j][k] = ctr;
                bs[6][6][1][1][1][i][j][k] = ccc;
                bs[6][7][1][1][1][i][j][k] = ccr;
                bs[7][0][1][1][1][i][j][k] = ftl;
                bs[7][1][1][1][1][i][j][k] = ftc;
                bs[7][2][1][1][1][i][j][k] = fcl;
                bs[7][3][1][1][1][i][j][k] = fcc;
                bs[7][4][1][1][1][i][j][k] = ctl;
                bs[7][5][1][1][1][i][j][k] = ctc;
                bs[7][6][1][1][1][i][j][k] = ccl;
                bs[7][7][1][1][1][i][j][k] = ccc;

                // b_ccr
                bs[1][0][1][1][2][i][j][k] = ccr;
                bs[3][0][1][1][2][i][j][k] = ctr;
                bs[5][0][1][1][2][i][j][k] = fcr;
                bs[7][0][1][1][2][i][j][k] = ftr;
                bs[1][2][1][1][2][i][j][k] = cbr;
                bs[3][2][1][1][2][i][j][k] = ccr;
                bs[5][2][1][1][2][i][j][k] = fbr;
                bs[7][2][1][1][2][i][j][k] = fcr;
                bs[1][4][1][1][2][i][j][k] = bcr;
                bs[3][4][1][1][2][i][j][k] = btr;
                bs[5][4][1][1][2][i][j][k] = ccr;
                bs[7][4][1][1][2][i][j][k] = ctr;
                bs[1][6][1][1][2][i][j][k] = bbr;
                bs[3][6][1][1][2][i][j][k] = bcr;
                bs[5][6][1][1][2][i][j][k] = cbr;
                bs[7][6][1][1][2][i][j][k] = ccr;

                // b_cbl
                bs[2][1][1][2][0][i][j][k] = cbl;
                bs[2][5][1][2][0][i][j][k] = bbl;
                bs[6][1][1][2][0][i][j][k] = fbl;
                bs[6][5][1][2][0][i][j][k] = cbl;

                // b_cbc
                bs[2][0][1][2][1][i][j][k] = cbc;
                bs[2][1][1][2][1][i][j][k] = cbr;
                bs[2][4][1][2][1][i][j][k] = bbc;
                bs[2][5][1][2][1][i][j][k] = bbr;
                bs[3][0][1][2][1][i][j][k] = cbr;
                bs[3][1][1][2][1][i][j][k] = cbc;
                bs[3][4][1][2][1][i][j][k] = bbr;
                bs[3][5][1][2][1][i][j][k] = bbc;
                bs[6][0][1][2][1][i][j][k] = fbc;
                bs[6][1][1][2][1][i][j][k] = fbr;
                bs[6][4][1][2][1][i][j][k] = cbc;
                bs[6][5][1][2][1][i][j][k] = cbr;
                bs[7][0][1][2][1][i][j][k] = fbr;
                bs[7][1][1][2][1][i][j][k] = fbc;
                bs[7][4][1][2][1][i][j][k] = cbr;
                bs[7][5][1][2][1][i][j][k] = cbc;

                // b_cbr
                bs[3][0][1][2][2][i][j][k] = cbr;
                bs[3][4][1][2][2][i][j][k] = bbr;
                bs[7][0][1][2][2][i][j][k] = fbr;
                bs[7][4][1][2][2][i][j][k] = cbr;

                // **** back ****
                // b_btl
                bs[4][3][2][0][0][i][j][k] = btl;

                // b_btc
                bs[4][2][2][0][1][i][j][k] = btc;
                bs[4][3][2][0][1][i][j][k] = btr;
                bs[5][2][2][0][1][i][j][k] = btl;
                bs[5][3][2][0][1][i][j][k] = btc;

                // b_btr
                bs[5][2][2][0][2][i][j][k] = btr;

                // b_bcl
                bs[4][1][2][1][0][i][j][k] = bcl;
                bs[4][3][2][1][0][i][j][k] = bbl;
                bs[6][1][2][1][0][i][j][k] = btl;
                bs[6][3][2][1][0][i][j][k] = bcl;

                // b_bcc
                bs[4][0][2][1][1][i][j][k] = bcc;
                bs[4][1][2][1][1][i][j][k] = bcr;
                bs[4][2][2][1][1][i][j][k] = bbc;
                bs[4][3][2][1][1][i][j][k] = bbr;
                bs[5][0][2][1][1][i][j][k] = bcl;
                bs[5][1][2][1][1][i][j][k] = bcc;
                bs[5][2][2][1][1][i][j][k] = bbl;
                bs[5][3][2][1][1][i][j][k] = bbc;
                bs[6][0][2][1][1][i][j][k] = btc;
                bs[6][1][2][1][1][i][j][k] = btr;
                bs[6][2][2][1][1][i][j][k] = bcc;
                bs[6][3][2][1][1][i][j][k] = bcr;
                bs[7][0][2][1][1][i][j][k] = btl;
                bs[7][1][2][1][1][i][j][k] = btc;
                bs[7][2][2][1][1][i][j][k] = bcl;
                bs[7][3][2][1][1][i][j][k] = bcc;

                // b_bcr
                bs[5][0][2][1][2][i][j][k] = bcr;
                bs[5][2][2][1][2][i][j][k] = bbr;
                bs[7][0][2][1][2][i][j][k] = btr;
                bs[7][2][2][1][2][i][j][k] = bcr;

                // b_bbr
                bs[6][1][2][2][0][i][j][k] = bbl;

                // b_bbc
                bs[6][0][2][2][1][i][j][k] = bbc;
                bs[6][1][2][2][1][i][j][k] = bbr;
                bs[7][0][2][2][1][i][j][k] = bbl;
                bs[7][1][2][2][1][i][j][k] = bbc;

                // b_bbr
                bs[7][0][2][2][2][i][j][k] = bbr;
            }

    bs.updateGhostsLocally();
}

void mgcl_test::fillFixedBlockstencilFromFixedStencil(mgcl::FixedBlockstencil& bs, mgcl::FixedStencil& fs)
{
    double ftl = fs[0][0][0];
    double ftc = fs[0][0][1];
    double ftr = fs[0][0][2];
    double fcl = fs[0][1][0];
    double fcc = fs[0][1][1];
    double fcr = fs[0][1][2];
    double fbl = fs[0][2][0];
    double fbc = fs[0][2][1];
    double fbr = fs[0][2][2];
    double ctl = fs[1][0][0];
    double ctc = fs[1][0][1];
    double ctr = fs[1][0][2];
    double ccl = fs[1][1][0];
    double ccc = fs[1][1][1];
    double ccr = fs[1][1][2];
    double cbl = fs[1][2][0];
    double cbc = fs[1][2][1];
    double cbr = fs[1][2][2];
    double btl = fs[2][0][0];
    double btc = fs[2][0][1];
    double btr = fs[2][0][2];
    double bcl = fs[2][1][0];
    double bcc = fs[2][1][1];
    double bcr = fs[2][1][2];
    double bbl = fs[2][2][0];
    double bbc = fs[2][2][1];
    double bbr = fs[2][2][2];

    // fill blockstencil with values from fs1
    // ***** front *****
    // b_ftl
    bs[0][7][0][0][0] = ftl;

    // b_ftc
    bs[0][6][0][0][1] = ftc;
    bs[0][7][0][0][1] = ftr;
    bs[1][6][0][0][1] = ftl;
    bs[1][7][0][0][1] = ftc;

    // b_ftr
    bs[1][6][0][0][2] = ftr;

    // b_fcl
    bs[0][5][0][1][0] = fcl;
    bs[0][7][0][1][0] = fbl;
    bs[2][5][0][1][0] = ftl;
    bs[2][7][0][1][0] = fcl;

    // b_fcc
    bs[0][4][0][1][1] = fcc;
    bs[0][5][0][1][1] = fcr;
    bs[0][6][0][1][1] = fbc;
    bs[0][7][0][1][1] = fbr;
    bs[1][4][0][1][1] = fcl;
    bs[1][5][0][1][1] = fcc;
    bs[1][6][0][1][1] = fbl;
    bs[1][7][0][1][1] = fbc;
    bs[2][4][0][1][1] = ftc;
    bs[2][5][0][1][1] = ftr;
    bs[2][6][0][1][1] = fcc;
    bs[2][7][0][1][1] = fcr;
    bs[3][4][0][1][1] = ftl;
    bs[3][5][0][1][1] = ftc;
    bs[3][6][0][1][1] = fcl;
    bs[3][7][0][1][1] = fcc;

    // b_fcr
    bs[1][4][0][1][2] = fcr;
    bs[1][6][0][1][2] = fbr;
    bs[3][4][0][1][2] = ftr;
    bs[3][6][0][1][2] = fcr;

    // b_fbl
    bs[2][5][0][2][0] = fbl;

    // b_fbc
    bs[2][4][0][2][1] = fbc;
    bs[2][5][0][2][1] = fbr;
    bs[3][4][0][2][1] = fbl;
    bs[3][5][0][2][1] = fbc;

    // b_fbr
    bs[3][4][0][2][2] = fbr;

    // ***** center *****
    // b_ctl
    bs[0][3][1][0][0] = ctl;
    bs[0][7][1][0][0] = btl;
    bs[4][3][1][0][0] = ftl;
    bs[4][7][1][0][0] = ctl;

    // b_ctc
    bs[0][2][1][0][1] = ctc;
    bs[0][3][1][0][1] = ctr;
    bs[0][6][1][0][1] = btc;
    bs[0][7][1][0][1] = btr;
    bs[1][2][1][0][1] = ctl;
    bs[1][3][1][0][1] = ctc;
    bs[1][6][1][0][1] = btl;
    bs[1][7][1][0][1] = btc;
    bs[4][2][1][0][1] = ftc;
    bs[4][3][1][0][1] = ftr;
    bs[4][6][1][0][1] = ctc;
    bs[4][7][1][0][1] = ctr;
    bs[5][2][1][0][1] = ftl;
    bs[5][3][1][0][1] = ftc;
    bs[5][6][1][0][1] = ctl;
    bs[5][7][1][0][1] = ctc;

    // b_ctr
    bs[1][2][1][0][2] = ctr;
    bs[1][6][1][0][2] = btr;
    bs[5][2][1][0][2] = ftr;
    bs[5][6][1][0][2] = ctr;

    // b_ccl
    bs[0][1][1][1][0] = ccl;
    bs[0][3][1][1][0] = cbl;
    bs[0][5][1][1][0] = bcl;
    bs[0][7][1][1][0] = bbl;
    bs[2][1][1][1][0] = ctl;
    bs[2][3][1][1][0] = ccl;
    bs[2][5][1][1][0] = btl;
    bs[2][7][1][1][0] = bcl;
    bs[4][1][1][1][0] = fcl;
    bs[4][3][1][1][0] = fbl;
    bs[4][5][1][1][0] = ccl;
    bs[4][7][1][1][0] = cbl;
    bs[6][1][1][1][0] = ftl;
    bs[6][3][1][1][0] = fcl;
    bs[6][5][1][1][0] = ctl;
    bs[6][7][1][1][0] = ccl;

    // b_ccc
    bs[0][0][1][1][1] = ccc;
    bs[0][1][1][1][1] = ccr;
    bs[0][2][1][1][1] = cbc;
    bs[0][3][1][1][1] = cbr;
    bs[0][4][1][1][1] = bcc;
    bs[0][5][1][1][1] = bcr;
    bs[0][6][1][1][1] = bbc;
    bs[0][7][1][1][1] = bbr;
    bs[1][0][1][1][1] = ccl;
    bs[1][1][1][1][1] = ccc;
    bs[1][2][1][1][1] = cbl;
    bs[1][3][1][1][1] = cbc;
    bs[1][4][1][1][1] = bcl;
    bs[1][5][1][1][1] = bcc;
    bs[1][6][1][1][1] = bbl;
    bs[1][7][1][1][1] = bbc;
    bs[2][0][1][1][1] = ctc;
    bs[2][1][1][1][1] = ctr;
    bs[2][2][1][1][1] = ccc;
    bs[2][3][1][1][1] = ccr;
    bs[2][4][1][1][1] = btc;
    bs[2][5][1][1][1] = btr;
    bs[2][6][1][1][1] = bcc;
    bs[2][7][1][1][1] = bcr;
    bs[3][0][1][1][1] = ctl;
    bs[3][1][1][1][1] = ctc;
    bs[3][2][1][1][1] = ccl;
    bs[3][3][1][1][1] = ccc;
    bs[3][4][1][1][1] = btl;
    bs[3][5][1][1][1] = btc;
    bs[3][6][1][1][1] = bcl;
    bs[3][7][1][1][1] = bcc;
    bs[4][0][1][1][1] = fcc;
    bs[4][1][1][1][1] = fcr;
    bs[4][2][1][1][1] = fbc;
    bs[4][3][1][1][1] = fbr;
    bs[4][4][1][1][1] = ccc;
    bs[4][5][1][1][1] = ccr;
    bs[4][6][1][1][1] = cbc;
    bs[4][7][1][1][1] = cbr;
    bs[5][0][1][1][1] = fcl;
    bs[5][1][1][1][1] = fcc;
    bs[5][2][1][1][1] = fbl;
    bs[5][3][1][1][1] = fbc;
    bs[5][4][1][1][1] = ccl;
    bs[5][5][1][1][1] = ccc;
    bs[5][6][1][1][1] = cbl;
    bs[5][7][1][1][1] = cbc;
    bs[6][0][1][1][1] = ftc;
    bs[6][1][1][1][1] = ftr;
    bs[6][2][1][1][1] = fcc;
    bs[6][3][1][1][1] = fcr;
    bs[6][4][1][1][1] = ctc;
    bs[6][5][1][1][1] = ctr;
    bs[6][6][1][1][1] = ccc;
    bs[6][7][1][1][1] = ccr;
    bs[7][0][1][1][1] = ftl;
    bs[7][1][1][1][1] = ftc;
    bs[7][2][1][1][1] = fcl;
    bs[7][3][1][1][1] = fcc;
    bs[7][4][1][1][1] = ctl;
    bs[7][5][1][1][1] = ctc;
    bs[7][6][1][1][1] = ccl;
    bs[7][7][1][1][1] = ccc;

    // b_ccr
    bs[1][0][1][1][2] = ccr;
    bs[3][0][1][1][2] = ctr;
    bs[5][0][1][1][2] = fcr;
    bs[7][0][1][1][2] = ftr;
    bs[1][2][1][1][2] = cbr;
    bs[3][2][1][1][2] = ccr;
    bs[5][2][1][1][2] = fbr;
    bs[7][2][1][1][2] = fcr;
    bs[1][4][1][1][2] = bcr;
    bs[3][4][1][1][2] = btr;
    bs[5][4][1][1][2] = ccr;
    bs[7][4][1][1][2] = ctr;
    bs[1][6][1][1][2] = bbr;
    bs[3][6][1][1][2] = bcr;
    bs[5][6][1][1][2] = cbr;
    bs[7][6][1][1][2] = ccr;

    // b_cbl
    bs[2][1][1][2][0] = cbl;
    bs[2][5][1][2][0] = bbl;
    bs[6][1][1][2][0] = fbl;
    bs[6][5][1][2][0] = cbl;

    // b_cbc
    bs[2][0][1][2][1] = cbc;
    bs[2][1][1][2][1] = cbr;
    bs[2][4][1][2][1] = bbc;
    bs[2][5][1][2][1] = bbr;
    bs[3][0][1][2][1] = cbr;
    bs[3][1][1][2][1] = cbc;
    bs[3][4][1][2][1] = bbr;
    bs[3][5][1][2][1] = bbc;
    bs[6][0][1][2][1] = fbc;
    bs[6][1][1][2][1] = fbr;
    bs[6][4][1][2][1] = cbc;
    bs[6][5][1][2][1] = cbr;
    bs[7][0][1][2][1] = fbr;
    bs[7][1][1][2][1] = fbc;
    bs[7][4][1][2][1] = cbr;
    bs[7][5][1][2][1] = cbc;

    // b_cbr
    bs[3][0][1][2][2] = cbr;
    bs[3][4][1][2][2] = bbr;
    bs[7][0][1][2][2] = fbr;
    bs[7][4][1][2][2] = cbr;

    // **** back ****
    // b_btl
    bs[4][3][2][0][0] = btl;

    // b_btc
    bs[4][2][2][0][1] = btc;
    bs[4][3][2][0][1] = btr;
    bs[5][2][2][0][1] = btl;
    bs[5][3][2][0][1] = btc;

    // b_btr
    bs[5][2][2][0][2] = btr;

    // b_bcl
    bs[4][1][2][1][0] = bcl;
    bs[4][3][2][1][0] = bbl;
    bs[6][1][2][1][0] = btl;
    bs[6][3][2][1][0] = bcl;

    // b_bcc
    bs[4][0][2][1][1] = bcc;
    bs[4][1][2][1][1] = bcr;
    bs[4][2][2][1][1] = bbc;
    bs[4][3][2][1][1] = bbr;
    bs[5][0][2][1][1] = bcl;
    bs[5][1][2][1][1] = bcc;
    bs[5][2][2][1][1] = bbl;
    bs[5][3][2][1][1] = bbc;
    bs[6][0][2][1][1] = btc;
    bs[6][1][2][1][1] = btr;
    bs[6][2][2][1][1] = bcc;
    bs[6][3][2][1][1] = bcr;
    bs[7][0][2][1][1] = btl;
    bs[7][1][2][1][1] = btc;
    bs[7][2][2][1][1] = bcl;
    bs[7][3][2][1][1] = bcc;

    // b_bcr
    bs[5][0][2][1][2] = bcr;
    bs[5][2][2][1][2] = bbr;
    bs[7][0][2][1][2] = btr;
    bs[7][2][2][1][2] = bcr;

    // b_bbr
    bs[6][1][2][2][0] = bbl;

    // b_bbc
    bs[6][0][2][2][1] = bbc;
    bs[6][1][2][2][1] = bbr;
    bs[7][0][2][2][1] = bbl;
    bs[7][1][2][2][1] = bbc;

    // b_bbr
    bs[7][0][2][2][2] = bbr;
}

void mgcl_test::fillFixedStencilFromBlockstencil(mgcl::Blockstencil& bs, mgcl::FixedStencil& fs)
{
    double* ftl = &fs[0][0][0];
    double* ftc = &fs[0][0][1];
    double* ftr = &fs[0][0][2];
    double* fcl = &fs[0][1][0];
    double* fcc = &fs[0][1][1];
    double* fcr = &fs[0][1][2];
    double* fbl = &fs[0][2][0];
    double* fbc = &fs[0][2][1];
    double* fbr = &fs[0][2][2];
    double* ctl = &fs[1][0][0];
    double* ctc = &fs[1][0][1];
    double* ctr = &fs[1][0][2];
    double* ccl = &fs[1][1][0];
    double* ccc = &fs[1][1][1];
    double* ccr = &fs[1][1][2];
    double* cbl = &fs[1][2][0];
    double* cbc = &fs[1][2][1];
    double* cbr = &fs[1][2][2];
    double* btl = &fs[2][0][0];
    double* btc = &fs[2][0][1];
    double* btr = &fs[2][0][2];
    double* bcl = &fs[2][1][0];
    double* bcc = &fs[2][1][1];
    double* bcr = &fs[2][1][2];
    double* bbl = &fs[2][2][0];
    double* bbc = &fs[2][2][1];
    double* bbr = &fs[2][2][2];

    // fill blockstencil with values from fs1
    int i = bs.getGhostsM();
    int j = bs.getGhostsN();
    int k = bs.getGhostsO();

    // ***** front *****
    // b_ftl
    *ftl = bs[0][7][0][0][0][i][j][k];

    // b_ftc
    *ftc = bs[0][6][0][0][1][i][j][k];
    *ftr = bs[0][7][0][0][1][i][j][k];
    *ftl = bs[1][6][0][0][1][i][j][k];
    *ftc = bs[1][7][0][0][1][i][j][k];

    // b_ftr
    *ftr = bs[1][6][0][0][2][i][j][k];

    // b_fcl
    *fcl = bs[0][5][0][1][0][i][j][k];
    *fbl = bs[0][7][0][1][0][i][j][k];
    *ftl = bs[2][5][0][1][0][i][j][k];
    *fcl = bs[2][7][0][1][0][i][j][k];

    // b_fcc
    *fcc = bs[0][4][0][1][1][i][j][k];
    *fcr = bs[0][5][0][1][1][i][j][k];
    *fbc = bs[0][6][0][1][1][i][j][k];
    *fbr = bs[0][7][0][1][1][i][j][k];
    *fcl = bs[1][4][0][1][1][i][j][k];
    *fcc = bs[1][5][0][1][1][i][j][k];
    *fbl = bs[1][6][0][1][1][i][j][k];
    *fbc = bs[1][7][0][1][1][i][j][k];
    *ftc = bs[2][4][0][1][1][i][j][k];
    *ftr = bs[2][5][0][1][1][i][j][k];
    *fcc = bs[2][6][0][1][1][i][j][k];
    *fcr = bs[2][7][0][1][1][i][j][k];
    *ftl = bs[3][4][0][1][1][i][j][k];
    *ftc = bs[3][5][0][1][1][i][j][k];
    *fcl = bs[3][6][0][1][1][i][j][k];
    *fcc = bs[3][7][0][1][1][i][j][k];

    // b_fcr
    *fcr = bs[1][4][0][1][2][i][j][k];
    *fbr = bs[1][6][0][1][2][i][j][k];
    *ftr = bs[3][4][0][1][2][i][j][k];
    *fcr = bs[3][6][0][1][2][i][j][k];

    // b_fbl
    *fbl = bs[2][5][0][2][0][i][j][k];

    // b_fbc
    *fbc = bs[2][4][0][2][1][i][j][k];
    *fbr = bs[2][5][0][2][1][i][j][k];
    *fbl = bs[3][4][0][2][1][i][j][k];
    *fbc = bs[3][5][0][2][1][i][j][k];

    // b_fbr
    *fbr = bs[3][4][0][2][2][i][j][k];

    // ***** center *****
    // b_ctl
    *ctl = bs[0][3][1][0][0][i][j][k];
    *btl = bs[0][7][1][0][0][i][j][k];
    *ftl = bs[4][3][1][0][0][i][j][k];
    *ctl = bs[4][7][1][0][0][i][j][k];

    // b_ctc
    *ctc = bs[0][2][1][0][1][i][j][k];
    *ctr = bs[0][3][1][0][1][i][j][k];
    *btc = bs[0][6][1][0][1][i][j][k];
    *btr = bs[0][7][1][0][1][i][j][k];
    *ctl = bs[1][2][1][0][1][i][j][k];
    *ctc = bs[1][3][1][0][1][i][j][k];
    *btl = bs[1][6][1][0][1][i][j][k];
    *btc = bs[1][7][1][0][1][i][j][k];
    *ftc = bs[4][2][1][0][1][i][j][k];
    *ftr = bs[4][3][1][0][1][i][j][k];
    *ctc = bs[4][6][1][0][1][i][j][k];
    *ctr = bs[4][7][1][0][1][i][j][k];
    *ftl = bs[5][2][1][0][1][i][j][k];
    *ftc = bs[5][3][1][0][1][i][j][k];
    *ctl = bs[5][6][1][0][1][i][j][k];
    *ctc = bs[5][7][1][0][1][i][j][k];

    // b_ctr
    *ctr = bs[1][2][1][0][2][i][j][k];
    *btr = bs[1][6][1][0][2][i][j][k];
    *ftr = bs[5][2][1][0][2][i][j][k];
    *ctr = bs[5][6][1][0][2][i][j][k];

    // b_ccl
    *ccl = bs[0][1][1][1][0][i][j][k];
    *cbl = bs[0][3][1][1][0][i][j][k];
    *bcl = bs[0][5][1][1][0][i][j][k];
    *bbl = bs[0][7][1][1][0][i][j][k];
    *ctl = bs[2][1][1][1][0][i][j][k];
    *ccl = bs[2][3][1][1][0][i][j][k];
    *btl = bs[2][5][1][1][0][i][j][k];
    *bcl = bs[2][7][1][1][0][i][j][k];
    *fcl = bs[4][1][1][1][0][i][j][k];
    *fbl = bs[4][3][1][1][0][i][j][k];
    *ccl = bs[4][5][1][1][0][i][j][k];
    *cbl = bs[4][7][1][1][0][i][j][k];
    *ftl = bs[6][1][1][1][0][i][j][k];
    *fcl = bs[6][3][1][1][0][i][j][k];
    *ctl = bs[6][5][1][1][0][i][j][k];
    *ccl = bs[6][7][1][1][0][i][j][k];

    // b_ccc
    *ccc = bs[0][0][1][1][1][i][j][k];
    *ccr = bs[0][1][1][1][1][i][j][k];
    *cbc = bs[0][2][1][1][1][i][j][k];
    *cbr = bs[0][3][1][1][1][i][j][k];
    *bcc = bs[0][4][1][1][1][i][j][k];
    *bcr = bs[0][5][1][1][1][i][j][k];
    *bbc = bs[0][6][1][1][1][i][j][k];
    *bbr = bs[0][7][1][1][1][i][j][k];
    *ccl = bs[1][0][1][1][1][i][j][k];
    *ccc = bs[1][1][1][1][1][i][j][k];
    *cbl = bs[1][2][1][1][1][i][j][k];
    *cbc = bs[1][3][1][1][1][i][j][k];
    *bcl = bs[1][4][1][1][1][i][j][k];
    *bcc = bs[1][5][1][1][1][i][j][k];
    *bbl = bs[1][6][1][1][1][i][j][k];
    *bbc = bs[1][7][1][1][1][i][j][k];
    *ctc = bs[2][0][1][1][1][i][j][k];
    *ctr = bs[2][1][1][1][1][i][j][k];
    *ccc = bs[2][2][1][1][1][i][j][k];
    *ccr = bs[2][3][1][1][1][i][j][k];
    *btc = bs[2][4][1][1][1][i][j][k];
    *btr = bs[2][5][1][1][1][i][j][k];
    *bcc = bs[2][6][1][1][1][i][j][k];
    *bcr = bs[2][7][1][1][1][i][j][k];
    *ctl = bs[3][0][1][1][1][i][j][k];
    *ctc = bs[3][1][1][1][1][i][j][k];
    *ccl = bs[3][2][1][1][1][i][j][k];
    *ccc = bs[3][3][1][1][1][i][j][k];
    *btl = bs[3][4][1][1][1][i][j][k];
    *btc = bs[3][5][1][1][1][i][j][k];
    *bcl = bs[3][6][1][1][1][i][j][k];
    *bcc = bs[3][7][1][1][1][i][j][k];
    *fcc = bs[4][0][1][1][1][i][j][k];
    *fcr = bs[4][1][1][1][1][i][j][k];
    *fbc = bs[4][2][1][1][1][i][j][k];
    *fbr = bs[4][3][1][1][1][i][j][k];
    *ccc = bs[4][4][1][1][1][i][j][k];
    *ccr = bs[4][5][1][1][1][i][j][k];
    *cbc = bs[4][6][1][1][1][i][j][k];
    *cbr = bs[4][7][1][1][1][i][j][k];
    *fcl = bs[5][0][1][1][1][i][j][k];
    *fcc = bs[5][1][1][1][1][i][j][k];
    *fbl = bs[5][2][1][1][1][i][j][k];
    *fbc = bs[5][3][1][1][1][i][j][k];
    *ccl = bs[5][4][1][1][1][i][j][k];
    *ccc = bs[5][5][1][1][1][i][j][k];
    *cbl = bs[5][6][1][1][1][i][j][k];
    *cbc = bs[5][7][1][1][1][i][j][k];
    *ftc = bs[6][0][1][1][1][i][j][k];
    *ftr = bs[6][1][1][1][1][i][j][k];
    *fcc = bs[6][2][1][1][1][i][j][k];
    *fcr = bs[6][3][1][1][1][i][j][k];
    *ctc = bs[6][4][1][1][1][i][j][k];
    *ctr = bs[6][5][1][1][1][i][j][k];
    *ccc = bs[6][6][1][1][1][i][j][k];
    *ccr = bs[6][7][1][1][1][i][j][k];
    *ftl = bs[7][0][1][1][1][i][j][k];
    *ftc = bs[7][1][1][1][1][i][j][k];
    *fcl = bs[7][2][1][1][1][i][j][k];
    *fcc = bs[7][3][1][1][1][i][j][k];
    *ctl = bs[7][4][1][1][1][i][j][k];
    *ctc = bs[7][5][1][1][1][i][j][k];
    *ccl = bs[7][6][1][1][1][i][j][k];
    *ccc = bs[7][7][1][1][1][i][j][k];

    // b_ccr
    *ccr = bs[1][0][1][1][2][i][j][k];
    *ctr = bs[3][0][1][1][2][i][j][k];
    *fcr = bs[5][0][1][1][2][i][j][k];
    *ftr = bs[7][0][1][1][2][i][j][k];
    *cbr = bs[1][2][1][1][2][i][j][k];
    *ccr = bs[3][2][1][1][2][i][j][k];
    *fbr = bs[5][2][1][1][2][i][j][k];
    *fcr = bs[7][2][1][1][2][i][j][k];
    *bcr = bs[1][4][1][1][2][i][j][k];
    *btr = bs[3][4][1][1][2][i][j][k];
    *ccr = bs[5][4][1][1][2][i][j][k];
    *ctr = bs[7][4][1][1][2][i][j][k];
    *bbr = bs[1][6][1][1][2][i][j][k];
    *bcr = bs[3][6][1][1][2][i][j][k];
    *cbr = bs[5][6][1][1][2][i][j][k];
    *ccr = bs[7][6][1][1][2][i][j][k];

    // b_cbl
    *cbl = bs[2][1][1][2][0][i][j][k];
    *bbl = bs[2][5][1][2][0][i][j][k];
    *fbl = bs[6][1][1][2][0][i][j][k];
    *cbl = bs[6][5][1][2][0][i][j][k];

    // b_cbc
    *cbc = bs[2][0][1][2][1][i][j][k];
    *cbr = bs[2][1][1][2][1][i][j][k];
    *bbc = bs[2][4][1][2][1][i][j][k];
    *bbr = bs[2][5][1][2][1][i][j][k];
    *cbr = bs[3][0][1][2][1][i][j][k];
    *cbc = bs[3][1][1][2][1][i][j][k];
    *bbr = bs[3][4][1][2][1][i][j][k];
    *bbc = bs[3][5][1][2][1][i][j][k];
    *fbc = bs[6][0][1][2][1][i][j][k];
    *fbr = bs[6][1][1][2][1][i][j][k];
    *cbc = bs[6][4][1][2][1][i][j][k];
    *cbr = bs[6][5][1][2][1][i][j][k];
    *fbr = bs[7][0][1][2][1][i][j][k];
    *fbc = bs[7][1][1][2][1][i][j][k];
    *cbr = bs[7][4][1][2][1][i][j][k];
    *cbc = bs[7][5][1][2][1][i][j][k];

    // b_cbr
    *cbr = bs[3][0][1][2][2][i][j][k];
    *bbr = bs[3][4][1][2][2][i][j][k];
    *fbr = bs[7][0][1][2][2][i][j][k];
    *cbr = bs[7][4][1][2][2][i][j][k];

    // **** back ****
    // b_btl
    *btl = bs[4][3][2][0][0][i][j][k];

    // b_btc
    *btc = bs[4][2][2][0][1][i][j][k];
    *btr = bs[4][3][2][0][1][i][j][k];
    *btl = bs[5][2][2][0][1][i][j][k];
    *btc = bs[5][3][2][0][1][i][j][k];

    // b_btr
    *btr = bs[5][2][2][0][2][i][j][k];

    // b_bcl
    *bcl = bs[4][1][2][1][0][i][j][k];
    *bbl = bs[4][3][2][1][0][i][j][k];
    *btl = bs[6][1][2][1][0][i][j][k];
    *bcl = bs[6][3][2][1][0][i][j][k];

    // b_bcc
    *bcc = bs[4][0][2][1][1][i][j][k];
    *bcr = bs[4][1][2][1][1][i][j][k];
    *bbc = bs[4][2][2][1][1][i][j][k];
    *bbr = bs[4][3][2][1][1][i][j][k];
    *bcl = bs[5][0][2][1][1][i][j][k];
    *bcc = bs[5][1][2][1][1][i][j][k];
    *bbl = bs[5][2][2][1][1][i][j][k];
    *bbc = bs[5][3][2][1][1][i][j][k];
    *btc = bs[6][0][2][1][1][i][j][k];
    *btr = bs[6][1][2][1][1][i][j][k];
    *bcc = bs[6][2][2][1][1][i][j][k];
    *bcr = bs[6][3][2][1][1][i][j][k];
    *btl = bs[7][0][2][1][1][i][j][k];
    *btc = bs[7][1][2][1][1][i][j][k];
    *bcl = bs[7][2][2][1][1][i][j][k];
    *bcc = bs[7][3][2][1][1][i][j][k];

    // b_bcr
    *bcr = bs[5][0][2][1][2][i][j][k];
    *bbr = bs[5][2][2][1][2][i][j][k];
    *btr = bs[7][0][2][1][2][i][j][k];
    *bcr = bs[7][2][2][1][2][i][j][k];

    // b_bbr
    *bbl = bs[6][1][2][2][0][i][j][k];

    // b_bbc
    *bbc = bs[6][0][2][2][1][i][j][k];
    *bbr = bs[6][1][2][2][1][i][j][k];
    *bbl = bs[7][0][2][2][1][i][j][k];
    *bbc = bs[7][1][2][2][1][i][j][k];

    // b_bbr
    *bbr = bs[7][0][2][2][2][i][j][k];
}