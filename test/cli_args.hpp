#ifndef MGCL_CLI_ARGS_HPP
#define MGCL_CLI_ARGS_HPP

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <vector>

namespace CLI_ARGS
{
    extern std::vector<cl_device_type> deviceTypes;
}

#endif // MGCL_CLI_ARGS_HPP
