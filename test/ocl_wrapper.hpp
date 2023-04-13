/*
 * @Author: Simon Hoffmann
 * @Email: shoffmann@uni-wuppertal.de
 * @Date: 2023-04-13 14:30:16
 * @Last Modified by: Simon Hoffmann
 * @Last Modified time: 2023-04-13 14:53:24
 * @Description: Wrapper for OpenCL Environment
 */

#ifndef MGCL_OCL_WRAPPER_HPP
#define MGCL_OCL_WRAPPER_HPP

#ifdef __APPLE__
#include <OpenCL/cl.h> // for clSetKernelArg, _cl_mem, cl_mem, clE...
#else
#include <CL/cl.h> // for clSetKernelArg, _cl_mem, cl_mem, clE...
#endif

#include <string>

// Small wrapper that initializes the OpenCL platform and compiles a given kernel file or a kernel string.
class OCLWrapper
{
public:
    OCLWrapper(cl_device_type deviceType, std::string deviceName, std::string kernelString, std::string kernelFilePath);
    OCLWrapper(cl_device_type deviceType, std::string deviceName, std::string kernelString, std::string kernelFilePath,
               cl_context _context);
    OCLWrapper(const OCLWrapper &) = delete;
    OCLWrapper &operator=(const OCLWrapper &) = delete;
    OCLWrapper(const OCLWrapper &&) = delete;
    OCLWrapper &operator=(OCLWrapper &&) = delete;
    ~OCLWrapper();

    std::string kernelString = "";
    std::string kernelDir = "./";
    std::string deviceName = "";
    cl_device_type deviceType = CL_DEVICE_TYPE_DEFAULT;
    cl_device_id deviceId = nullptr;
    cl_context context = nullptr;
    cl_command_queue commands = nullptr;
    cl_program program = nullptr;

    int err;
};

#endif // MGCL_OCL_WRAPPER_HPP
