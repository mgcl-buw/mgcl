#pragma once

#include <vector>

#ifndef CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#endif // CL_USE_DEPRECATED_OPENCL_1_2_APIS

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif // CL_TARGET_OPENCL_VERSION

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "../src/mgcl/cuboid.hpp"
#include "../src/mgcl/cuboid_bs.hpp"
#include "../src/mgcl/opencl_helper.hpp"
#include "../src/mgcl/problem.hpp"

namespace mgcl_test
{
    class TestUtility
    {
    private:
        std::shared_ptr<mgcl::Problem> problem;
        std::vector<cl_mem> openclBuffers;
        bool profilingEnabled = false;

    public:
        TestUtility();
        TestUtility(std::string deviceName);
        TestUtility(cl_device_type deviceType);
        TestUtility(cl_device_type deviceType, bool profilingEnabled);
        TestUtility(std::shared_ptr<mgcl::Problem> problem);
        ~TestUtility();
        cl_mem createOpenCLBuffer(mgcl::Cuboid& c);
        cl_mem createOpenCLBuffer(mgcl::CuboidBS& c);
        std::unique_ptr<mgcl::Cuboid> readOpenCLBuffer(cl_mem buf, int m, int n, int o, int ghosts_m = 0, int ghosts_n = 0, int ghosts_o = 0);
        int finish();
        void releaseBuffers();

        static bool deviceAvailable(std::string deviceName, cl_device_type deviceType);

        cl_context getContext();
        cl_command_queue getCommands();
        cl_program getProgram();
        cl_device_id getDeviceId();
        mgcl::Problem& getProblem();
    };

    void create4hOrderPeriodicProblem(mgcl::Cuboid& v, mgcl::Cuboid& f, mgcl::Cuboid& solution);
    void fill7pLaplace(mgcl::VaryingStencil& v, double h, bool negativeCenter);
    void fill19pLaplace(mgcl::VaryingStencil& v, double h, bool negativeCenter);
    void fill27pLaplace(mgcl::VaryingStencil& v, double h, bool negativeCenter);
    void fill7pLaplace(mgcl::FixedStencil& v, double h, bool negativeCenter);
    void fill27pLaplace(mgcl::FixedStencil& v, double h, bool negativeCenter);
    void fill7pLaplace(mgcl::Blockstencil& v, double h, bool negativeCenter);
    void fill27pLaplace(mgcl::Blockstencil& v, double h, bool negativeCenter);

    std::shared_ptr<mgcl::Cuboid> calculateError(mgcl::Cuboid& solution, mgcl::Cuboid& approximation);
    double calculateMaxError(mgcl::Cuboid& error);
    double calculateErrorNorm(double h, mgcl::Cuboid& error);

    void fill3dFullWeightRestrictionBlockstencil(mgcl::FixedBlockstencil& bs);
    void fill3dBilinearProlongationBlockstencil(mgcl::FixedBlockstencil& bs);

    void fillVaryingStencilFromFixedStencil(mgcl::VaryingStencil& bs, mgcl::FixedStencil& fs);
    void fillBlockstencilFromFixedStencil(mgcl::Blockstencil& bs, mgcl::FixedStencil& fs);
    void fillFixedBlockstencilFromFixedStencil(mgcl::FixedBlockstencil& bs, mgcl::FixedStencil& fs);
    void fillFixedStencilFromBlockstencil(mgcl::Blockstencil& bs, mgcl::FixedStencil& fs);
    void copyCuboidToCuboidBS(mgcl::Cuboid& src, mgcl::CuboidBS& dst, int scalem, int scalen, int scaleo);
    void copyCuboidBSToCuboid(mgcl::CuboidBS& src, mgcl::Cuboid& dst, int scalem, int scalen, int scaleo);
}
