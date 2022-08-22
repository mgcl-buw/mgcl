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

#include "cuboid.hpp"
#include "mgcl.hpp"

namespace mgcl
{
    // forward declarations
    class Level;
    class OpenCLHelper;
    class MultigridEngine;

    // main interface class for using mgcl. Defines all the problem variables.
    class Problem
    {
    private:
        /* initial solution and right hand side vectors */
        double ***v = nullptr; /* can be ommited if device buffers are supplied */
        double ***f = nullptr; /* can be ommited if device buffers are supplied */

        /* Buffers for v and f. Only need to be set if buffers already exist on device and should be reused */
        cl_mem dV = nullptr;
        cl_mem dF = nullptr;
        cl_mem dStencilValues = nullptr;

        /* grid dimensions */
        int m;
        int n;
        int o;

        /* Holds level-dependent data for each level */
        std::vector<std::shared_ptr<Level>> levels;

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

        /* Manages OpenCL stuff */
        std::shared_ptr<OpenCLHelper> openCLHelper = nullptr;

        friend class OpenCLHelper;
        friend class Level;
        friend class MultigridEngine;

    public:
        Problem(int _m, int _n, int _o, Cuboid _f, Cuboid _v);
        Problem(int _m, int _n, int _o, double ***_f, double ***_v);
        Problem(int _m, int _n, int _o, cl_mem _d_f, cl_mem _d_v);
        Problem(const Problem &) = delete;
        Problem &operator=(const Problem &) = delete;
        ~Problem() = default;

        // TODO implement
        bool checkParameters();
        int calculateAndSetMaxLevel();
        bool init();
        int readResults();

        void solve();
        void solveSeq();

        /********************************
         * Getters and Setters
         ********************************/

        double ***getV() const;
        void setV(double ***v_);

        double ***getF() const;
        void setF(double ***f_);

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

        double ***getStencilValues() const;
        void setStencilValues(double ***stencilValues);

        int getStencilSizeMultiplier() const;
        void setStencilSizeMultiplier(int stencilSizeMultiplier);

        bool getRestrictProlongateStencil() const;
        void setRestrictProlongateStencil(bool restrictProlongateStencil);

        bool getReuseOpenclBuffers() const;
        void setReuseOpenclBuffers(bool reuseOpenclBuffers);

        bool getCopyBufferData() const;
        void setCopyBufferData(bool copyBufferData);

        bool getReadResults() const;
        void setReadResults(bool readResults);

        bool getUseLocalMemory() const;
        void setUseLocalMemory(bool useLocalMemory);

        int getJacobiWgSizeX() const;
        void setJacobiWgSizeX(int jacobiWgSizeX);

        int getJacobiWgSizeY() const;
        void setJacobiWgSizeY(int jacobiWgSizeY);

        int getJacobiIterationsPerKernel() const;
        void setJacobiIterationsPerKernel(int jacobiIterationsPerKernel);

        bool getUseOpencl() const;
        void setUseOpencl(bool useOpencl);

        cl_mem getDStencilValues() const;
        void setDStencilValues(const cl_mem &dStencilValues);

        std::vector<std::shared_ptr<Level>> getLevels() const;

        cl_mem getDV() const;
        void setDV(const cl_mem &dV_);

        cl_mem getDF() const;
        void setDF(const cl_mem &dF_);

        std::shared_ptr<OpenCLHelper> getOpenCLHelper() const;
        void setOpenCLHelper(const std::shared_ptr<OpenCLHelper> &openCLHelper_);
    };
}