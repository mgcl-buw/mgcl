#include "test_utility.hpp"

#include "../mgcl.hpp"

#include <iostream>

mgcl_test::TestUtility::TestUtility()
    : problem(2, 2, 2)
{
    problem.initOpenCL();

    // int err = clRetainContext(openclHelper.getContext());
    // mgcl::mgclCheckError(err, "clRetainContext");

    // err = clRetainCommandQueue(openclHelper.getCommands());
    // mgcl::mgclCheckError(err, "clRetainCommandQueue");

    // err = clRetainDevice(openclHelper.getDeviceId());
    // mgcl::mgclCheckError(err, "clRetainDevice");
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
    cl_mem buf = clCreateBuffer(problem.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
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
mgcl::Cuboid mgcl_test::TestUtility::readOpenCLBuffer(cl_mem buf, int m, int n, int o)
{
    finish();

    mgcl::Cuboid c(m, n, o);
    double *tmp = new double[m * n * o];
    if (buf)
    {
        int err;
        err = clEnqueueReadBuffer(problem.getCommands(), buf, CL_TRUE, 0,
                                  sizeof(double) * m * n * o, tmp, 0, NULL, NULL);
        mgcl::mgclCheckError(err, "clEnqueueReadBuffer");
    }

    // copy into cuboid
    for (int i = 0; i < m * n * o; i++)
        c.field1d()[i] = tmp[i];
    delete[] tmp;
    return c;
}

int mgcl_test::TestUtility::finish()
{
    int err = clFinish(problem.getCommands());
    mgcl::mgclCheckError(err, "clFinish");
    return err;
}

cl_context mgcl_test::TestUtility::getContext()
{
    return problem.getContext();
}

mgcl::Problem &mgcl_test::TestUtility::getProblem()
{
    return problem;
}

cl_command_queue mgcl_test::TestUtility::getCommands()
{
    return problem.getCommands();
}

cl_device_id mgcl_test::TestUtility::getDeviceId()
{
    return problem.getDeviceId();
}
