#pragma once

#ifndef CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#endif // CL_USE_DEPRECATED_OPENCL_1_2_APIS

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif // CL_TARGET_OPENCL_VERSION

#include <CL/cl.h>
#include <memory>
#include <string>
#include <vector>

#include "level.hpp"
#include "mgcl.hpp"

namespace mgcl
{
    // main interface class for using mgcl. Defines all the problem variables.
    class Problem
    {
    private:
        /* initial solution and right hand side vectors */
        double ***v = nullptr; /* can be ommited if device buffers are supplied */
        double ***f = nullptr; /* can be ommited if device buffers are supplied */

        /* Buffers for v and f. Only need to be set if buffers already exist on device and should be reused */
        cl_mem d_v = nullptr;
        cl_mem d_f = nullptr;
        cl_mem d_stencil_values = nullptr;

        /* grid dimensions */
        int m;
        int n;
        int o;

        /* Holds level-dependent data for each level */
        std::vector<std::unique_ptr<Level>> levels;

        // TODO check what happens when reuse_buffer and ghosts != ghosts_in
        /* Amount of ghost cells surrounding v and f. If optimized jacobi shall be used it must be greater or equal than
         * max(nu1, nu2). Must fit to buffers ghosts size if reuse_opencl_buffers is set. Defaults to 1. */
        int ghosts = 1;

        /* Amount of ghost cells of input data. Defaults to 0. Only relevant if buffers are not reused. */
        int ghosts_in = 0;

        /* maximum grid level */
        int maxlevel = -1;

        /* maximum v-cycle iteration count */
        int maxiter_vcycles = 5;

        /* pre and post smoothing steps */
        int nu1 = 2;
        int nu2 = 2;

        /* damping factor */
        double omega = 0.8;

        /* tolerance */
        double tol = 1e-7;

        /* Type of norm of the residual which will be used as termination criterium */
        MGCL_RESIDUAL_NORM residual_norm = MGCL_L2;

        /* Type of stencil that will be used in jacobi's method */
        MGCL_STENCIL stencil = MGCL_7POINT;

        /* Stencil values per grid point. Must be given by the user if a varying symmetric stencil shall be used.
         * This array is only used as input, mgcl_level_data::stencil_values will be used internally for each level.
         * Size depends on stencil type:
         * -  7-point varsym: size = m*n*o*4
         * - 19-point varsym: size = m*n*o*7
         * - 27-point varsym: size = m*n*o*8 */
        double ***stencil_values = nullptr;

        /* Size multiplier for non-fixed stencils. Used internally only. */
        int stencil_size_multiplier = 1;

        /* Wether to restrict and prolongate varsym stencil or just keep the values as is on each level. Defaults to true.
         */
        bool restrict_prolongate_stencil = true;

        /* Whether to use opencl or not. Defaults to 0 (not using opencl) */
        // TODO needed?
        bool use_opencl = false;

        /* Whether to reuse opencl buffers, context and commands or not. Defaults to 0 (not reusing opencl buffers).
         * Provided buffers must have size of ghosted grid, e.g. (m+2*ghosts) * (n+2*ghosts) * (o+2*ghosts) */
        bool reuse_opencl_buffers = false;

        /* If true input data from d_v and d_f will be copied to newly created buffers, respecting the nearfield ghost cell
         * amount. Defaults to 0. */
        bool copy_buffer_data = false;

        /* If true, resulting v is read from device to host when mgcl has finished. Defaults to 0 (not reading results). */
        bool read_results = false;

        /* If true, optimized jacobi version is used which calculates multiple iterations in one kernel call using local
         * memory. ghosts must be equal to iteration count per kernel call which is limited by local memory size of the
         * device. Defaults to false. */
        bool use_local_memory = false;

        /* Preferred work-group size in x-dimension for jacobi smoother. Defaults to 16. */
        int jacobi_wg_size_x = 16;

        /* Preferred work-group size in y-dimension for jacobi smoother. Defaults to 16. */
        int jacobi_wg_size_y = 16;

        /* Preferred iterations per jacobi kernel call. Is only used when use_local_memory is true. Defaults to 3.
         * Gets automatically decreased if nu1, nu2 or ghosts are smaller. */
        int jacobi_iterations_per_kernel = 3;

        /* OpenCL stuff */
        std::string kernel_dir = "./";
        std::string device_name = "";                        /* Use first found device if not set */
        cl_device_type device_type = CL_DEVICE_TYPE_DEFAULT; /* Defaults to CL_DEVICE_TYPE_DEFAULT */
        cl_device_id device_id = nullptr;                    /* must be set if a specific device should be reused */
        cl_context context = nullptr;                        /* must be set if a specific context/device/buffers should be reused */
        cl_command_queue commands = nullptr;                 /* must be set if a specific context/device/buffers should be reused */
        cl_program program = nullptr;                        /* compute program, only for internal purposes */

    public:
        Problem(int _m, int _n, int _o, Cuboid _f, Cuboid _v);
        Problem(int _m, int _n, int _o, double ***_f, double ***_v);
        Problem(int _m, int _n, int _o, cl_mem _d_f, cl_mem _d_v);
        Problem(const Problem &) = delete;
        Problem &operator=(const Problem &) = delete;
        ~Problem() = default;

        // TODO implement
        bool checkParameters();
        bool checkOpenCLParameters();
        int calculateAndSetMaxLevel();
        bool init();
        void restrict_seq(Level &fine, Level &coarse);
        void restrict_test(Level &fine, Level &coarse);
        void restrict(Level &fine, Level &coarse);
        void prolongate_seq(Level &fine, Level &coarse);
        void prolongate_test(Level &fine, Level &coarse);
        void prolongate(Level &fine, Level &coarse);
        void finish();
        int correct_error(cl_mem d_v, cl_mem d_r, int m, int n, int o);
        int readBackResults();
        void mgcl();
        void mgcl_seq();
        double vcycle_seq(Level &level);
        double vcycle(Level &level);
        void test_read(Level &level);

        /********************************
         * Getters and Setters
         ********************************/

        double ***getV() const;
        void setV(double ***v_);

        double ***getF() const;
        void setF(double ***f_);

        cl_mem dV() const;
        void setDV(const cl_mem &dV);

        cl_mem dF() const;
        void setDF(const cl_mem &dF);

        int getM() const;

        int getN() const;

        int getO() const;

        int getGhosts() const;
        void setGhosts(int ghosts_);

        int getGhostsIn() const;
        void setGhostsIn(int ghostsIn);

        int getMaxlevel() const;
        void setMaxlevel(int maxlevel_);

        int getMaxiterVcycles() const;
        void setMaxiterVcycles(int maxiterVcycles);

        int getNu1() const;
        void setNu1(int nu1_);

        int getNu2() const;
        void setNu2(int nu2_);

        double getOmega() const;
        void setOmega(double omega_);

        double getTol() const;
        void setTol(double tol_);

        MGCL_RESIDUAL_NORM getResidualNorm() const;
        void setResidualNorm(const MGCL_RESIDUAL_NORM &residualNorm);

        MGCL_STENCIL getStencil() const;
        void setStencil(const MGCL_STENCIL &stencil_);

        double ***stencilValues() const;
        void setStencilValues(double ***stencilValues);

        int stencilSizeMultiplier() const;
        void setStencilSizeMultiplier(int stencilSizeMultiplier);

        bool restrictProlongateStencil() const;
        void setRestrictProlongateStencil(bool restrictProlongateStencil);

        bool reuseOpenclBuffers() const;
        void setReuseOpenclBuffers(bool reuseOpenclBuffers);

        bool copyBufferData() const;
        void setCopyBufferData(bool copyBufferData);

        bool getReadResults() const;
        void setReadResults(bool readResults);

        bool useLocalMemory() const;
        void setUseLocalMemory(bool useLocalMemory);

        int jacobiWgSizeX() const;
        void setJacobiWgSizeX(int jacobiWgSizeX);

        int jacobiWgSizeY() const;
        void setJacobiWgSizeY(int jacobiWgSizeY);

        int jacobiIterationsPerKernel() const;
        void setJacobiIterationsPerKernel(int jacobiIterationsPerKernel);

        std::string getKernelDir() const;
        void setKernelDir(const std::string &kernelDir);

        std::string getDeviceName() const;
        void setDeviceName(const std::string &deviceName);

        cl_device_type getDeviceType() const;
        void setDeviceType(const cl_device_type &deviceType);

        cl_device_id getDeviceId() const;
        void setDeviceId(const cl_device_id &deviceId);

        cl_context getContext() const;
        void setContext(const cl_context &context_);

        cl_command_queue getCommands() const;
        void setCommands(const cl_command_queue &commands_);

        bool useOpencl() const;
        void setUseOpencl(bool useOpencl);

        cl_mem dStencilValues() const;
        void setDStencilValues(const cl_mem &dStencilValues);
    };
}