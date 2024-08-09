#pragma once

#include "profiling_data.hpp"
#ifndef CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#endif // CL_USE_DEPRECATED_OPENCL_1_2_APIS

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif // CL_TARGET_OPENCL_VERSION

#include "kernel_config.hpp"
#include "level.hpp"
#include "mgcl.hpp" // for MGCL_RESIDUAL_NORM, MGCL_STENCIL, MGCL_L2
#include "mpi_global_data.hpp"
#include "opencl_helper.hpp" // for OpenCLHelper
#include "stencil.hpp"       // for VaryingStencil3x3x3

#include <memory> // for shared_ptr, unique_ptr
#include <ostream>
#include <string> // for string
#include <vector> // for vector

#include "mpi.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h> // for cl_mem, _cl_mem, cl_command_queue, cl_c...
#endif

namespace mgcl
{
    // forward declarations
    class Cuboid;
    class CuboidGpu;

    /**
     * @brief Main interface class for using mgcl. Defines all the parameters which can be set using the appropriate
     * setters. The problem is then solved using solve (using OpenCL if use_opencl is true, otherwise sequentially).
     * If an OpenCLEnvironment shall be reused, reuseOpenCL can be used.
     *
     */
    class Problem
    {
    private:
        /* initial solution and right hand side vectors */
        std::shared_ptr<Cuboid> v = nullptr; /* can be ommited if device buffers are supplied */
        std::shared_ptr<Cuboid> f = nullptr; /* can be ommited if device buffers are supplied */

        /* Buffers for v and f. Only need to be set if buffers already exist on device and should be reused */
        std::shared_ptr<CuboidGpu> dV = nullptr;
        std::shared_ptr<CuboidGpu> dF = nullptr;
        std::shared_ptr<CuboidGpu> dStencilValues = nullptr;

        // Temporary buffer for storing cuboid planes for ghost update. Must be greater or equal to the amount
        // of ghost cells of the cuboid that is being updated. It is initialized based on dVIn of level 0 and
        // thus should be big enough for all cuboids.
        std::shared_ptr<CuboidGpu> dPlanesBuf = nullptr;
        std::shared_ptr<Cuboid> hPlanesBufSend = nullptr;
        std::shared_ptr<Cuboid> hPlanesBufRecv = nullptr;

        /* local grid dimensions (on one MPI process if MPI is used) */
        int m;
        int n;
        int o;

        /* global grid dimensions (is equal to local grid dimensions if MPI is not used) */
        int mGlobal;
        int nGlobal;
        int oGlobal;

        /* Holds level-dependent data for each level */
        std::vector<std::unique_ptr<Level>> levels;

        // Vector for holding the relative residuals for each iteration. Can be queried after solve.
        std::vector<double> residuals;

        // TODO check what happens when reuse_buffer and ghosts != ghosts_in
        /* Amount of ghost cells surrounding v and f. If optimized jacobi shall be used it must be greater or equal than
         * max(nu1, nu2). Must fit to buffers ghosts size if reuse_opencl_buffers is set. Defaults to 1. */
        int ghosts = 1;

        /* Amount of ghost cells of input data. Defaults to 0. Only relevant if buffers are not reused or bc is not
           periodic. */
        int ghosts_in = 0; // TODO remove, use ghosts attribute from cuboids?

        /* maximum grid level, starting from 0 as the finest level */
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

        /* If true, tolerance will be ignored, thus maxiter_vcycles v-cycle iterations will be done for sure.
         * Defaults to false.
         * Set this to true if you are using OpenCL and don't need to know the relative residual each v-cycle iteration. */
        bool ignoreTol = false;

        /* Type of norm of the residual which will be used as termination criterium */
        MGCL_RESIDUAL_NORM residual_norm = MGCL_L2;

        /* Stencil that will be used in Jacobi's method */
        MGCL_STENCIL stencilType = MGCL_LAPLACE_7POINT;
        std::shared_ptr<VaryingStencil> stencilValues = nullptr;

        /* Boundary condition that shall be used. Only affects whether ghosts are updated. Values need to be set
           in input v.
           If not using periodic, ghosts_in must be set appropriately, see readme. */
        BC bc = BC::PERIODIC;

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

        /* Preferred iterations per jacobi kernel call. Is only used when use_local_memory is true. Defaults to 1.
         * Gets automatically decreased if nu1, nu2 or ghosts are smaller. */
        int jacobi_iterations_per_kernel = 1;

        /* If true, no output will be generated by mgcl. */
        bool silent = false;

        /* Elapsed V-Cycle iterations. Can be queried after solve. */
        int elapsedIterations = 0;

        /* Manages OpenCL stuff */
        OpenCLHelper openCLHelper;

        /* If true, kernel timings will be measured. */
        bool profilingEnabled = false;
        std::unique_ptr<ProfilingData> profilingData = nullptr;

        /* Kernel config, can be customized to set inidividual work-group sizes per kernel, dependend on number
         * of work-items. */
        conf::KernelConfig kernelConfig = conf::createDefaultKernelConfig();

        /* MPI relevant data. */
        std::unique_ptr<MPIGlobalData> mpiGlobalData = std::make_unique<MPIGlobalData>();

        /* First coarse level for which MPI is not used anymore. Only for internal purposes. */
        int mpiLevelThreshold = -1;

        /* Minimum amount of grid points for which MPI is used. Coarser levels will be run on one process. */
        int mpiMinGridPoints = 4;

        /* If true, all MPI routines are ignored, even if the program is started with more than one MPI process. */
        bool ignoreMpi = false;

        void checkGlobalDimensions();
        bool useMpi();

        friend class OpenCLHelper;
        friend class Level;
        friend class MultigridEngine;

    public:
        Problem(int m_, int n_, int o_, int m_global_ = -1, int n_global_ = -1, int o_global_ = -1);
        Problem(int m_, int n_, int o_, Cuboid* f_, Cuboid* v_, int m_global_ = -1, int n_global_ = -1, int o_global_ = -1);
        Problem(int m_, int n_, int o_, std::shared_ptr<Cuboid> f_, std::shared_ptr<Cuboid> v_,
                int m_global_ = -1, int n_global_ = -1, int o_global_ = -1);
        Problem(int m_, int n_, int o_, std::shared_ptr<CuboidGpu> d_f_, std::shared_ptr<CuboidGpu> d_v_,
                int m_global_ = -1, int n_global_ = -1, int o_global_ = -1);
        Problem(const Problem&) = delete;
        Problem& operator=(const Problem&) = delete;
        Problem(const Problem&&) = delete;
        Problem& operator=(const Problem&&) = delete;
        ~Problem() = default;

        bool checkParameters();
        void checkGpuSizes();
        int calculateAndSetMaxLevel();
        bool init();

        void reuseOpenCL(cl_context context, cl_command_queue commandQueue, cl_device_id deviceId);
        void initOpenCL();
        int readResults();
        void finish();

        void solve();
        void solve(bool skipInit);
        void solveSeq();
        void solveSeq(bool skipInit);

        Level& getLevelAt(int index) const;
        int getLevelsSize() const;

        inline bool isPeriodic() const { return bc == BC::PERIODIC; }

        int mpiSize();

        void printDeviceInfo();

        /********************************
         * Getters and Setters
         ********************************/

        Cuboid& getV() const;
        std::shared_ptr<Cuboid> getVPtr() const;
        void setV(std::shared_ptr<Cuboid> v_);

        Cuboid& getF() const;
        std::shared_ptr<Cuboid> getFPtr() const;
        void setF(std::shared_ptr<Cuboid> f_);

        CuboidGpu* getDPlanesBufPtr() const;
        CuboidGpu& getDPlanesBuf() const;
        void setDPlanesBuf(const std::shared_ptr<CuboidGpu> dPlanesBuf_);

        Cuboid* getHPlanesBufSendPtr() const;
        Cuboid& getHPlanesBufSend() const;
        void setHPlanesBufSend(const std::shared_ptr<Cuboid> hPlanesBuf_);
        Cuboid* getHPlanesBufRecvPtr() const;
        Cuboid& getHPlanesBufRecv() const;
        void setHPlanesBufRecv(const std::shared_ptr<Cuboid> hPlanesBuf_);

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
        void setResidualNorm(const MGCL_RESIDUAL_NORM& residualNorm);

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

        CuboidGpu& getDStencilValues() const;
        std::shared_ptr<CuboidGpu> getDStencilValuesPtr() const;
        void setDStencilValues(std::shared_ptr<CuboidGpu> dStencilValues);

        CuboidGpu& getDV() const;
        std::shared_ptr<CuboidGpu> getDVPtr() const;
        void setDV(std::shared_ptr<CuboidGpu> dV_);

        CuboidGpu& getDF() const;
        std::shared_ptr<CuboidGpu> getDFPtr() const;
        void setDF(std::shared_ptr<CuboidGpu> dF_);

        OpenCLHelper& getOpenCLHelper();

        std::string getKernelFile() const;
        void setKernelFile(const std::string& kernelFile_);

        std::string getDeviceName() const;
        void setDeviceName(const std::string& deviceName_);

        cl_device_type getDeviceType() const;
        void setDeviceType(const cl_device_type& deviceType_);

        cl_device_id getDeviceId() const;

        cl_context getContext() const;

        cl_command_queue getCommands() const;

        cl_program getProgram() const;

        bool getSilent() const;
        void setSilent(bool silent_);

        bool getIgnoreTol() const;
        void setIgnoreTol(bool ignoreTol_);

        MGCL_STENCIL getStencilType() const;
        void setStencilType(const MGCL_STENCIL& stencilType_);

        std::shared_ptr<VaryingStencil>& getStencilValues();

        BC getBc() const;
        void setBc(const BC& bc_);

        void setMpiComm(MPI_Comm _comm);
        MPI_Comm getMpiComm();

        int getMpiLevelThreshold();
        void setMpiLevelThreshold(int mpiLevelThreshold_);

        int getMpiMinGridPoints() const;
        void setMpiMinGridPoints(int mpiMinGridPoints_);

        void calculateAndSetMpiLevelThreshold();

        int mpiRank() const;

        int getMGlobal() const;
        int getNGlobal() const;
        int getOGlobal() const;

        bool getIgnoreMpi() const { return ignoreMpi; }
        void setIgnoreMpi(bool ignoreMpi_) { ignoreMpi = ignoreMpi_; }

        MPIGlobalData& getMPIGlobalData() { return *mpiGlobalData; }

        bool isProfilingEnabled() const { return profilingEnabled; }
        void setProfilingEnabled(bool profilingEnabled_);

        ProfilingData* getProfilingData() { return profilingData.get(); }

        conf::KernelConfig& getKernelConfig() { return kernelConfig; }

        inline int getElapsedIterations() const { return elapsedIterations; }

        std::vector<double>& getResiduals();

        friend std::ostream& operator<<(std::ostream& os, const Problem& lv);
    };
}
