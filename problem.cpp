#include <ctgmath>
#include <exception>
#include <iostream>
#include <string>

#include "cuboid.hpp"
#include "multigrid_engine.hpp"
#include "problem.hpp"

namespace mgcl
{
    Problem::Problem(int m_, int n_, int o_)
        : m(m_), n(n_), o(o_),
          openCLHelper(this),
          stencil(std::make_shared<StencilLaplace7p>(1.0 / (double)m))
    {
        calculateAndSetMaxLevel();
    }

    Problem::Problem(int m_, int n_, int o_, Cuboid *f_, Cuboid *v_)
        : m(m_), n(n_), o(o_), f(f_), v(v_),
          openCLHelper(this),
          stencil(std::make_shared<StencilLaplace7p>(1.0 / (double)m))
    {
        calculateAndSetMaxLevel();
    }

    Problem::Problem(int m_, int n_, int o_, std::shared_ptr<Cuboid> f_, std::shared_ptr<Cuboid> v_)
        : m(m_), n(n_), o(o_), f(f_), v(v_),
          openCLHelper(this),
          stencil(std::make_shared<StencilLaplace7p>(1.0 / (double)m))
    {
        calculateAndSetMaxLevel();
    }

    Problem::Problem(int m_, int n_, int o_, cl_mem d_f_, cl_mem d_v_)
        : m(m_), n(n_), o(o_), dF(d_f_), dV(d_v_),
          openCLHelper(this),
          stencil(std::make_shared<StencilLaplace7p>(1.0 / (double)m))
    {
        calculateAndSetMaxLevel();
    }

    /**
     * @brief Checks mandatory configuration fields: m, n, o, v, f, d_v, d_f, ghosts and ghosts_in.
     *
     * @return true All good.
     * @return false Something's wrong.
     */
    bool Problem::checkParameters()
    {
        //  check mandatory config fields
        if ((v == nullptr || f == nullptr) && (dV == nullptr || dF == nullptr))
        {
            if (!silent)
                printf("mgcl: supplied v or f and d_v or d_f is nullptr. Aborting.\n");
            return false;
        }

        if (m < 1 || n < 1 || o < 1)
        {
            if (!silent)
                printf("mgcl: m, n or o not supplied, zero or negative. Aborting.\n");
            return false;
        }

        if (ghosts < 1)
        {
            if (!silent)
                printf("mgcl: ghosts must be >= 1. Aborting.\n");
            return false;
        }

        if (ghosts_in < 0)
        {
            if (!silent)
                printf("mgcl: ghosts_in must be >= 0. Aborting.\n");
            return false;
        }

        return true;
    }

    /**
     * @brief Calculates 0-based max level using the minimum of real grid dimensions or uses user set max level if
     * it's valid.
     *
     * @return int 0-based max level, i.e. log2(min(m,n,o)) or valid user set maxlevel
     */
    int Problem::calculateAndSetMaxLevel()
    {
        // find max level or use user specified one
        int minsize = m < n ? m : n;
        minsize = minsize < o ? minsize : o;
        int maxlv = log2(minsize);

        if (maxlevel >= 0) // user has specified a maxlevel
        {
            if (maxlv < maxlevel) // user specified maxlevel is too high
            {
                if (!silent)
                    printf("user specified maxlevel of %d is too high! Using %d instead.\n", maxlevel, maxlv);
                maxlevel = maxlv;
            }
        }
        else
            maxlevel = maxlv; // use calculated maxlevel

        return maxlevel;
    }

    /**
     * @brief Creates Level objects, allocates memory.
     *
     * @return true All good.
     * @return false There was an error somewhere.
     */
    bool Problem::init()
    {
        if (!checkParameters())
            return false;

        // create opencl environment with default parameters if not done yet
        if (use_opencl)
        {
            if (initOpenCL() != CL_SUCCESS)
                return false;
        }

        // check opencl components if device buffers should be reused
        if (reuse_opencl_buffers || copy_buffer_data)
        {
            if (!openCLHelper.checkParameters())
                return false;
        }

        calculateAndSetMaxLevel();
        if (!silent)
            printf("maxlevel = %d\n", maxlevel);

        // initialize levels
        for (int level = 0; level <= maxlevel; level++)
        {
            auto lv = std::make_unique<Level>(this, level);
            levels.push_back(std::move(lv));
            levels.back()->init();
        }

        return true;
    }

    /**
     * @brief Releases current OpenCL environment, sets use_opencl to true, assigns OpenCL variables and initializes
     * or retains new environment.
     *
     * @param context OpenCL context to be reused.
     * @param commandQueue OpenCL command_queue to be reused.
     * @param deviceId OpenCL device_id to be reused.
     */
    int Problem::reuseOpenCL(cl_context context, cl_command_queue commandQueue, cl_device_id deviceId)
    {
        int err;
        err = openCLHelper.release();
        mgclCheckError(err, "openCLHelper.release");

        openCLHelper.setContext(context);
        openCLHelper.setCommands(commandQueue);
        openCLHelper.setDeviceId(deviceId);
        setUseOpencl(true);

        err = openCLHelper.init();
        mgclCheckError(err, "openCLHelper.init");
        return err;
    }

    /**
     * @brief Lazyly initializes OpenCL environment.
     *
     * @return int OpenCL error code
     */
    int Problem::initOpenCL()
    {
        int err = CL_SUCCESS;
        if (!openCLHelper.isInitialized())
        {
            err = openCLHelper.init();
            mgclCheckError(err, "openCLHelper.init");
        }
        return err;
    }

    /* Waits for all running OpenCL kernels to finish and reads back results from device. Creates arrays on host if none
     * were specified */
    int Problem::readResults()
    {
        int err = clFinish(openCLHelper.getCommands());
        mgclCheckError(err, "Waiting for kernels to finish");

        if (reuse_opencl_buffers || copy_buffer_data)
        {
            levels[0]->setV(std::make_shared<Cuboid>(levels[0]->m, levels[0]->n, levels[0]->o, ghosts, ghosts, ghosts));
            if (v == NULL)
                v = std::make_shared<Cuboid>(m, n, o, ghosts_in, ghosts_in, ghosts_in);
        }

        // read back results TODO: only for testing purposes, maybe define TESTING?
        err = clEnqueueReadBuffer(openCLHelper.getCommands(), levels[0]->dVIn, CL_TRUE, 0,
                                  sizeof(double) * levels[0]->mgh * levels[0]->ngh * levels[0]->ogh, levels[0]->getV()[0][0], 0, NULL, NULL);
        mgclCheckError(err, "Error: Failed to read output arrays from device!");

        // copy result to initial v vector
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < o; k++)
                {
                    (*v)[i + ghosts_in][j + ghosts_in][k + ghosts_in] =
                        levels[0]->getV()[i + ghosts][j + ghosts][k + ghosts];
                }

        return err;
    }

    /**
     * @brief Solves problem using multigrid method on OpenCL, if use_opencl is true. Otherwise solveSeq is called.
     *
     */
    void Problem::solve()
    {
        // run mgcl_seq if use_opencl is not set
        if (!use_opencl)
        {
            if (!silent)
                printf("Not using OpenCL (not specified in conf). Running mgcl sequentially.\n");
            solveSeq();
            return;
        }
        if (!silent)
            printf("Starting mgcl using OpenCL\n");

        // set up data for each level TODO reuse device buffers in final code
        if (!init())
            throw std::runtime_error("Failed to initialize mgcl data structures.");

        // calculate initial residual
        double initres = MultigridEngine::residual(*this, *levels[0], !ignoreTol);
        if (!silent && !ignoreTol)
            printf("Starting mgcl with initres = %e\n", initres);

        // run vcycle maxiter_vcycles times
        double res, relres;
        for (int i = 0; i < maxiter_vcycles; i++)
        {
            auto tstart = std::chrono::steady_clock::now();
            res = MultigridEngine::vcycle(*this, *levels[0]);
            auto tend = mgcl_since(tstart).count();

            if (!ignoreTol)
                relres = initres == 0 ? 0 : res / initres;

            if (!silent)
                if (ignoreTol)
                    printf("iter = %d, elapsed time = %ld ms\n", i, tend);
                else
                    printf("iter = %d, elapsed time = %ld ms, rel. res = %e\n", i, tend, relres);

            if (!ignoreTol && relres < tol)
                break;
        }

        // copy resulting v to d_v on device
        if (copy_buffer_data)
            openCLHelper.copyOutputBuffers();

        // write result into v on host
        if (read_results)
            readResults();
    }

    /**
     * @brief Solves problem sequentially using multigrid method.
     *
     * @throws runtime_error When initializing data structures failed.
     */
    void Problem::solveSeq()
    {
        // set up data for each level
        if (!init())
            throw std::runtime_error("Failed to initialize mgcl data structures.");

        // calculate initial residual (different from pmg's initres bc ghosts are not updated in pmg first)
        MultigridEngine::updateGhostsSeq(levels[0]->getV());
        double initres = MultigridEngine::residualSeq(levels[0]->getF(), levels[0]->getV(), levels[0]->getR(), residual_norm, *stencil, !ignoreTol);
        if (!silent && !ignoreTol)
            printf("Starting mgcl with initres = %e\n", initres);

        // run vcycle maxiter_vcycles times
        double res, relres;
        for (int i = 0; i < maxiter_vcycles; i++)
        {
            auto tstart = std::chrono::steady_clock::now();
            res = MultigridEngine::vcycleSeq(*this, *levels[0]);
            auto tend = mgcl_since(tstart).count();

            if (!ignoreTol)
                relres = initres == 0 ? 0 : res / initres;

            if (!silent)
                if (ignoreTol)
                    printf("iter = %d, elapsed time = %ld ms\n", i, tend);
                else
                    printf("iter = %d, elapsed time = %ld ms, rel. res = %e\n", i, tend, relres);

            if (!ignoreTol && relres < tol)
                break;
        }

        // write data to output
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < o; k++)
                {
                    (*v)[i + ghosts_in][j + ghosts_in][k + ghosts_in] =
                        levels[0]->getV()[i + ghosts][j + ghosts][k + ghosts];
                }
    }

    Level &Problem::getLevelAt(int index) const
    {
        return *levels[index];
    }

    int Problem::getLevelsSize() const
    {
        return levels.size();
    }

    /********************************
     * Getters and Setters
     ********************************/

    Cuboid &Problem::getF() const
    {
        return *f;
    }

    std::shared_ptr<Cuboid> Problem::getFPtr() const
    {
        return f;
    }

    void Problem::setF(std::shared_ptr<Cuboid> f_)
    {
        f = f_;
    }

    int Problem::getO() const
    {
        return o;
    }

    int Problem::getGhostsIn() const
    {
        return ghosts_in;
    }

    void Problem::setGhostsIn(int ghostsIn)
    {
        ghosts_in = ghostsIn;
    }

    int Problem::getMaxiterVcycles() const
    {
        return maxiter_vcycles;
    }

    void Problem::setMaxiterVcycles(int maxiterVcycles)
    {
        maxiter_vcycles = maxiterVcycles;
    }

    int Problem::getNu2() const
    {
        return nu2;
    }

    void Problem::setNu2(int nu2_)
    {
        nu2 = nu2_;
    }

    double Problem::getTol() const
    {
        return tol;
    }

    void Problem::setTol(double tol_)
    {
        tol = tol_;
    }

    std::shared_ptr<Stencil> Problem::getStencil() const
    {
        return stencil;
    }

    void Problem::setStencil(std::shared_ptr<Stencil> stencil_)
    {
        stencil = stencil_;
    }

    bool Problem::getReuseOpenclBuffers() const
    {
        return reuse_opencl_buffers;
    }

    void Problem::setReuseOpenclBuffers(bool reuseOpenclBuffers)
    {
        reuse_opencl_buffers = reuseOpenclBuffers;
    }

    bool Problem::getReadResults() const
    {
        return read_results;
    }

    void Problem::setReadResults(bool readResults)
    {
        read_results = readResults;
    }

    int Problem::getJacobiWgSizeX() const
    {
        return jacobi_wg_size_x;
    }

    void Problem::setJacobiWgSizeX(int jacobiWgSizeX)
    {
        jacobi_wg_size_x = jacobiWgSizeX;
    }

    int Problem::getJacobiIterationsPerKernel() const
    {
        return jacobi_iterations_per_kernel;
    }

    void Problem::setJacobiIterationsPerKernel(int jacobiIterationsPerKernel)
    {
        jacobi_iterations_per_kernel = jacobiIterationsPerKernel;
    }

    cl_mem Problem::getDStencilValues() const
    {
        return dStencilValues;
    }

    cl_mem Problem::getDV() const
    {
        return dV;
    }

    void Problem::setDV(const cl_mem &dV_)
    {
        dV = dV_;
    }

    void Problem::setDStencilValues(const cl_mem &dStencilValues_)
    {
        dStencilValues = dStencilValues_;
    }

    int Problem::getN() const
    {
        return n;
    }

    int Problem::getM() const
    {
        return m;
    }

    int Problem::getGhosts() const
    {
        return ghosts;
    }

    void Problem::setGhosts(int ghosts_)
    {
        ghosts = ghosts_;
    }

    int Problem::getMaxlevel() const
    {
        return maxlevel;
    }

    void Problem::setMaxlevel(int maxlevel_)
    {
        maxlevel = maxlevel_;
        calculateAndSetMaxLevel();
    }

    int Problem::getNu1() const
    {
        return nu1;
    }

    void Problem::setNu1(int nu1_)
    {
        nu1 = nu1_;
    }

    double Problem::getOmega() const
    {
        return omega;
    }

    void Problem::setOmega(double omega_)
    {
        omega = omega_;
    }

    MGCL_RESIDUAL_NORM Problem::getResidualNorm() const
    {
        return residual_norm;
    }

    void Problem::setResidualNorm(const MGCL_RESIDUAL_NORM &residualNorm)
    {
        residual_norm = residualNorm;
    }

    bool Problem::getCopyBufferData() const
    {
        return copy_buffer_data;
    }

    void Problem::setCopyBufferData(bool copyBufferData)
    {
        copy_buffer_data = copyBufferData;
    }

    bool Problem::getUseLocalMemory() const
    {
        return use_local_memory;
    }

    void Problem::setUseLocalMemory(bool useLocalMemory)
    {
        use_local_memory = useLocalMemory;
    }

    int Problem::getJacobiWgSizeY() const
    {
        return jacobi_wg_size_y;
    }

    void Problem::setJacobiWgSizeY(int jacobiWgSizeY)
    {
        jacobi_wg_size_y = jacobiWgSizeY;
    }

    bool Problem::getUseOpencl() const
    {
        return use_opencl;
    }

    void Problem::setUseOpencl(bool useOpencl)
    {
        use_opencl = useOpencl;
    }

    cl_mem Problem::getDF() const
    {
        return dF;
    }

    void Problem::setDF(const cl_mem &dF_)
    {
        dF = dF_;
    }

    OpenCLHelper &Problem::getOpenCLHelper()
    {
        return openCLHelper;
    }

    std::string Problem::getDeviceName() const
    {
        return openCLHelper.deviceName;
    }

    void Problem::setDeviceName(const std::string &deviceName_)
    {
        if (!openCLHelper.isInitialized())
            openCLHelper.deviceName = deviceName_;
    }

    cl_device_id Problem::getDeviceId() const
    {
        return openCLHelper.deviceId;
    }

    cl_command_queue Problem::getCommands() const
    {
        return openCLHelper.commands;
    }

    std::string Problem::getKernelDir() const
    {
        return openCLHelper.kernelDir;
    }

    void Problem::setKernelDir(const std::string &kernelDir_)
    {
        if (!openCLHelper.isInitialized())
            openCLHelper.kernelDir = kernelDir_;
    }

    cl_device_type Problem::getDeviceType() const
    {
        return openCLHelper.deviceType;
    }

    void Problem::setDeviceType(const cl_device_type &deviceType_)
    {
        if (!openCLHelper.isInitialized())
            openCLHelper.deviceType = deviceType_;
    }

    cl_context Problem::getContext() const
    {
        return openCLHelper.context;
    }

    bool Problem::getSilent() const
    {
        return silent;
    }

    void Problem::setSilent(bool silent_)
    {
        silent = silent_;
    }

    cl_program Problem::getProgram() const
    {
        return openCLHelper.program;
    }

    bool Problem::getIgnoreTol() const
    {
        return ignoreTol;
    }

    void Problem::setIgnoreTol(bool ignoreTol_)
    {
        ignoreTol = ignoreTol_;
    }

    Cuboid &Problem::getV() const
    {
        return *v;
    }

    std::shared_ptr<Cuboid> Problem::getVPtr() const
    {
        return v;
    }

    void Problem::setV(std::shared_ptr<Cuboid> v_)
    {
        v = v_;
    }
}