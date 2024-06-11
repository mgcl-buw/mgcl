/*
 * @Author: Simon Hoffmann
 * @Email: shoffmann@uni-wuppertal.de
 * @Date: 2023-04-12 12:45:59
 * @Last Modified by: Simon Hoffmann
 * @Last Modified time: 2024-02-23 09:56:40
 * @Description: Utility kernels for various functions, i.e. building the sum of a buffer.
 */

#ifndef MGCL_UTIL_HPP
#define MGCL_UTIL_HPP

#include <stddef.h> // for size_t

#include "cuboid_gpu.hpp"
#include "kernel_config.hpp"
#include "profiling_data.hpp"

#ifdef __APPLE__
#include <OpenCL/cl.h> // for clSetKernelArg, _cl_mem, cl_mem, clE...
#else
#include <CL/cl.h> // for clSetKernelArg, _cl_mem, cl_mem, clE...
#endif

namespace mgcl::util
{
    double sum(CuboidGpu& buf, cl_program program, cl_command_queue commands,
               bool return_sum, mgcl::conf::KernelConfig* conf, mgcl::ProfilingData* pd);

    double max(CuboidGpu& buf, cl_program program, cl_command_queue commands,
               bool return_sum, mgcl::conf::KernelConfig* conf, mgcl::ProfilingData* pd);

    double max_abs(CuboidGpu& buf, cl_program program, cl_command_queue commands,
                   bool return_sum, mgcl::conf::KernelConfig* conf, mgcl::ProfilingData* pd);

    void fill(cl_program program, cl_command_queue commands,
              cl_mem buffer, double value, int size, mgcl::conf::KernelConfig* conf, mgcl::ProfilingData* pd);

    namespace seq
    {
        int min3(int a, int b, int c);
        double min3(double a, double b, double c);
    }
}

#endif // MGCL_UTIL_HPP
