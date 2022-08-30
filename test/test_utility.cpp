#include "test_utility.hpp"

#include "../mgcl.hpp"

#include <iostream>

mgcl_test::TestUtility::TestUtility()
{
    problem = std::make_shared<mgcl::Problem>(2, 2, 2);
    problem->initOpenCL();
}

mgcl_test::TestUtility::TestUtility(std::shared_ptr<mgcl::Problem> problem_)
    : problem(std::shared_ptr<mgcl::Problem>(problem_))
{
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
