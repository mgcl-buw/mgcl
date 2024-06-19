#pragma once

#include "cuboid.hpp"
#include <string>

#ifndef CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#endif // CL_USE_DEPRECATED_OPENCL_1_2_APIS

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif // CL_TARGET_OPENCL_VERSION

#ifdef __APPLE__
#include <OpenCL/cl_platform.h>
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#include <CL/cl_platform.h>
#endif

namespace mgcl
{
    // forward declarations
    class Problem;

    class OpenCLHelper
    {
    private:
        Problem* problem;

        std::string kernelFile = "./mgcl.cl";
        std::string deviceName = "";                        /* Use first found device if not set */
        cl_device_type deviceType = CL_DEVICE_TYPE_DEFAULT; /* Defaults to CL_DEVICE_TYPE_DEFAULT */
        cl_device_id deviceId = nullptr;                    /* must be set if a specific device should be reused */
        cl_context context = nullptr;                       /* must be set if a specific context/device/buffers should be reused */
        cl_command_queue commands = nullptr;                /* must be set if a specific context/device/buffers should be reused */
        cl_program program = nullptr;                       /* compute program, only for internal purposes */
        cl_platform_id platformId = nullptr;                /* Cannot be set from outside, just to print platform name */

        /* If true, kernel code is read from separate file, which is denoted by kernelFile. This allows
         * us to use a different kernel file for e.g. benchmarking. */
        bool readKernelFromFile = false;

        friend class Problem;

    public:
        explicit OpenCLHelper(Problem* problem_) : problem(problem_) {}
        OpenCLHelper(const OpenCLHelper&) = delete;
        OpenCLHelper& operator=(const OpenCLHelper&) = delete;
        OpenCLHelper(const OpenCLHelper&&) = delete;
        OpenCLHelper& operator=(OpenCLHelper&&) = delete;
        ~OpenCLHelper();

        void init();
        void release();
        bool isInitialized();
        bool checkParameters();
        int copyInputBuffers();
        int copyOutputBuffers();
        Cuboid readBuffer(cl_mem d_buf, int m, int n, int o);
        void printBuffer(cl_mem d_buf, int m, int n, int o);

        void finish();

        static std::string loadKernelSource(std::string file);
        int outputDeviceInfo();

        std::string getKernelFile() const;
        void setKernelFile(const std::string& kernelFile_);

        std::string getDeviceName() const;
        void setDeviceName(const std::string& deviceName_);

        cl_device_type getDeviceType() const;
        void setDeviceType(const cl_device_type& deviceType_);

        cl_program getProgram() const;
        void setProgram(const cl_program& program_);

        cl_command_queue getCommands() const;
        void setCommands(const cl_command_queue& commands_);

        cl_context getContext() const;
        void setContext(const cl_context& context_);

        static const char* mgcl_err_code(cl_int err_in);
        static void mgcl_check_error(cl_int err, const char* operation, const char* filename, int line);

        cl_device_id getDeviceId() const;
        void setDeviceId(const cl_device_id& deviceId_);

        Problem* getProblem() const;

        bool getReadKernelFromFile() const { return readKernelFromFile; }
        void setReadKernelFromFile(bool readKernelFromFile_) { readKernelFromFile = readKernelFromFile_; }
    };

#define mgclCheckError(E, S) OpenCLHelper::mgcl_check_error(E, S, __FILE__, __LINE__)
}
