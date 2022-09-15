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
        std::shared_ptr<mgcl::Problem> problem;
        std::vector<cl_mem> openclBuffers;

    public:
        TestUtility();
        TestUtility(std::string deviceName);
        TestUtility(cl_device_type deviceType);
        TestUtility(std::shared_ptr<mgcl::Problem> problem);
        ~TestUtility();
        cl_mem createOpenCLBuffer(mgcl::Cuboid &c);
        std::shared_ptr<mgcl::Cuboid> readOpenCLBuffer(cl_mem buf, int m, int n, int o, int ghosts_m = 0, int ghosts_n = 0, int ghosts_o = 0);
        int finish();

        static bool deviceAvailable(std::string deviceName, cl_device_type deviceType);

        cl_context getContext();
        cl_command_queue getCommands();
        cl_device_id getDeviceId();
        mgcl::Problem &getProblem();
    };
}