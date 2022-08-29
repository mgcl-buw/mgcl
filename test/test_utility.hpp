#pragma once

#include <vector>

#ifndef CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#endif // CL_USE_DEPRECATED_OPENCL_1_2_APIS

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif // CL_TARGET_OPENCL_VERSION

#include <CL/cl.h>

#include "../cuboid.hpp"
#include "../opencl_helper.hpp"
#include "../problem.hpp"

namespace mgcl_test
{
    class TestUtility
    {
    private:
        mgcl::Problem problem;
        std::vector<cl_mem> openclBuffers;

    public:
        TestUtility();
        ~TestUtility();
        cl_mem createOpenCLBuffer(mgcl::Cuboid &c);
        mgcl::Cuboid readOpenCLBuffer(cl_mem buf, int m, int n, int o);
        int finish();

        cl_context getContext();
        cl_command_queue getCommands();
        cl_device_id getDeviceId();
        mgcl::Problem &getProblem();
    };
}