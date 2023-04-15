/*
 * @Author: Simon Hoffmann
 * @Email: shoffmann@uni-wuppertal.de
 * @Date: 2023-04-12 12:45:59
 * @Last Modified by: Simon Hoffmann
 * @Last Modified time: 2023-04-12 14:14:44
 * @Description: Utility kernels for various functions, i.e. building the sum of a buffer.
 */

#ifndef MGCL_UTIL_HPP
#define MGCL_UTIL_HPP

#include <stddef.h> // for size_t

#ifdef __APPLE__
#include <OpenCL/cl.h> // for clSetKernelArg, _cl_mem, cl_mem, clE...
#else
#include <CL/cl.h> // for clSetKernelArg, _cl_mem, cl_mem, clE...
#endif

namespace mgcl::util
{
    double sum(cl_mem buf, size_t num_elements, cl_context context, cl_program program, cl_command_queue commands,
               bool return_sum, size_t localSize = 512);
    double max(cl_mem buf, size_t num_elements, cl_context context, cl_program program, cl_command_queue commands,
               bool return_sum, size_t localSize = 512);
    double max_abs(cl_mem buf, size_t num_elements, cl_context context, cl_program program, cl_command_queue commands,
                   bool return_sum, size_t localSize = 512);
}

#endif // MGCL_UTIL_HPP
