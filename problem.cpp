#include <ctgmath>
#include <exception>
#include <string>

#include "cuboid.hpp"
#include "multigrid_engine.hpp"
#include "problem.hpp"

namespace mgcl
{
    Problem::Problem(int m_, int n_, int o_)
        : m(m_), n(n_), o(o_)
    {
    }

    Problem::Problem(int m_, int n_, int o_, Cuboid *f_, Cuboid *v_)
        : m(m_), n(n_), o(o_), f(std::make_shared<Cuboid>(*f_)), v(std::make_shared<Cuboid>(*v_))
    {
    }

    Problem::Problem(int m_, int n_, int o_, std::shared_ptr<Cuboid> f_, std::shared_ptr<Cuboid> v_)
        : m(m_), n(n_), o(o_), f(f_), v(v_)
    {
    }

    Problem::Problem(int m_, int n_, int o_, cl_mem d_f_, cl_mem d_v_)
        : m(m_), n(n_), o(o_), dF(d_f_), dV(d_v_)
    {
    }

    /**
     * @brief Checks mandatory configuration fields: m, n, o, v, f, d_v, d_f, stencil_values, ghosts and ghosts_in.
     *
     * @return true All good.
     * @return false Something's wrong.
     */
    bool Problem::checkParameters()
    {
        // TODO opencl
        //  check mandatory config fields
        if ((v == nullptr || f == nullptr) && (dV == nullptr || dF == nullptr))
        {
            printf("mgcl: supplied v or f and d_v or d_f is nullptr. Aborting.\n");
            return false;
        }

        if (m < 1 || n < 1 || o < 1)
        {
            printf("mgcl: m, n or o not supplied, zero or negative. Aborting.\n");
            return false;
        }

        if (ghosts < 1)
        {
            printf("mgcl: ghosts must be >= 1. Aborting.\n");
            return false;
        }

        if (ghosts_in < 0)
        {
            printf("mgcl: ghosts_in must be >= 0. Aborting.\n");
            return false;
        }

        if (((stencil_values == nullptr && !use_opencl) ||
             (dStencilValues == nullptr && reuse_opencl_buffers)) &&
            (stencil == MGCL_7POINT_VARSYM || stencil == MGCL_19POINT_VARSYM ||
             stencil == MGCL_27POINT_VARSYM))
        {
            printf("stencil is set to be varying symmetric but stencil_values is nullptr! Aborting.\n");
            return false;
        }

        return true;
    }

    /**
     * @brief Calculates max level using grid dimensions
     *
     * @return int max level
     */
    int Problem::calculateAndSetMaxLevel()
    {
        // find max level or use user specified one
        int minsize = m < n ? m : n;
        minsize = minsize < o ? minsize : o;
        int maxlv = log2(minsize) + 1;

        if (maxlevel >= 0) // user has specified a maxlevel
        {
            if (maxlv < maxlevel) // user specified maxlevel is too high
            {
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

        // set stencil size if stencil is set to a varying symmetric one
        if (stencil == MGCL_7POINT_VARSYM)
            stencil_size_multiplier = 4;
        else if (stencil == MGCL_19POINT_VARSYM)
            stencil_size_multiplier = 7;
        else if (stencil == MGCL_27POINT_VARSYM)
            stencil_size_multiplier = 8;

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
        printf("maxlevel = %d\n", maxlevel);

        for (int level = 0; level < maxlevel; level++)
        {
            if (level == 0)
            {
                int mg = m + 2 * ghosts;
                int ng = n + 2 * ghosts;
                int og = o + 2 * ghosts;

                auto lv = std::make_shared<Level>(this, level, m, n, o);

                // create ghosted arrays for v and f on host if device buffer should not be reused
                if (!reuse_opencl_buffers && !copy_buffer_data)
                {
                    lv->setV(std::make_shared<Cuboid>(m, n, o, ghosts, ghosts, ghosts));
                    lv->setF(std::make_shared<Cuboid>(m, n, o, ghosts, ghosts, ghosts));

                    // copy initial input data from conf into mgcl data struct
                    for (int i = 0; i < m; i++)
                        for (int j = 0; j < n; j++)
                            for (int k = 0; k < o; k++)
                            {
                                lv->getV()[i + ghosts][j + ghosts][k + ghosts] =
                                    (*v)[i + ghosts_in][j + ghosts_in][k + ghosts_in];
                                lv->getF()[i + ghosts][j + ghosts][k + ghosts] =
                                    (*f)[i + ghosts_in][j + ghosts_in][k + ghosts_in];
                            }

                    MultigridEngine::updateGhostsSeq(lv->getF());

                    // allocate initial stencil_values, including ghost cells, if varying symmetric stencil shall be used
                    if (stencil_values || dStencilValues)
                    {
                        lv->setStencilValues(std::make_shared<Cuboid>(m, n, o * stencil_size_multiplier, ghosts, ghosts, ghosts));

                        // copy initial input stencil data from conf into mgcl data struct
                        for (int i = 0; i < m; i++)
                            for (int j = 0; j < n; j++)
                                for (int k = 0; k < o * stencil_size_multiplier; k++)
                                {
                                    lv->getStencilValues()[i + ghosts][j + ghosts]
                                                          [k + ghosts * stencil_size_multiplier] =
                                        (*stencil_values)[i + ghosts_in][j + ghosts_in]
                                                         [k + ghosts_in * stencil_size_multiplier];
                                }

                        MultigridEngine::updateGhostsSeq(lv->getStencilValues());
                    }
                }

                // r on host is only needed if opencl should not be used
                if (!use_opencl)
                {
                    lv->setR(std::make_shared<Cuboid>(m, n, o, ghosts, ghosts, ghosts));
                }

                levels.push_back(std::move(lv));
            }
            else
            {
                // ghosted sizes of current level's grid
                int mg = levels[level - 1]->getM() / 2 + 2 * ghosts;
                int ng = levels[level - 1]->getN() / 2 + 2 * ghosts;
                int og = levels[level - 1]->getO() / 2 + 2 * ghosts;

                auto lv = std::make_shared<Level>(this, level,
                                                  levels[level - 1]->getM() / 2,
                                                  levels[level - 1]->getN() / 2,
                                                  levels[level - 1]->getO() / 2);

                if (!use_opencl)
                {
                    lv->setV(std::make_shared<Cuboid>(lv->getM(), lv->getN(), lv->getO(), ghosts, ghosts, ghosts));
                    lv->setF(std::make_shared<Cuboid>(lv->getM(), lv->getN(), lv->getO(), ghosts, ghosts, ghosts));
                    lv->setR(std::make_shared<Cuboid>(lv->getM(), lv->getN(), lv->getO(), ghosts, ghosts, ghosts));

                    if (stencil_values != nullptr)
                    {
                        if (restrict_prolongate_stencil)
                            lv->setStencilValues(std::make_shared<Cuboid>(m, n, o * stencil_size_multiplier, ghosts, ghosts, ghosts));
                        else
                            lv->setStencilValues(levels[0]->getStencilValuesPtr());
                    }
                }

                levels.push_back(std::move(lv));
            }

            levels.back()->initOpenCLBuffers();
            levels.back()->setH(1.0 / (m * m));
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
                                  sizeof(double) * levels[0]->m * levels[0]->n * levels[0]->o, levels[0]->getV()[0][0], 0, NULL, NULL);
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
            printf("Not using OpenCL (not specified in conf). Running mgcl sequentially.\n");
            solveSeq();
            return;
        }
        printf("Starting mgcl using OpenCL\n");

        // set up data for each level TODO reuse device buffers in final code
        if (!init())
            throw std::runtime_error("Failed to initialize mgcl data structures.");

        // calculate initial residual
        double initres;
        if (stencil_values)
            initres = MultigridEngine::stencilResidual(*this, *levels[0], 1);
        else
            initres = MultigridEngine::residual(*this, *levels[0], 1);
        printf("Starting mgcl with initres = %e\n", initres);

        // run vcycle maxiter_vcycles times
        double res, relres;
        for (int i = 0; i < maxiter_vcycles; i++)
        {
            auto tstart = std::chrono::steady_clock::now();
            res = MultigridEngine::vcycle(*this, *levels[0]);
            auto tend = mgcl_since(tstart).count();
            relres = initres == 0 ? 0 : res / initres;
            printf("iter = %d, elapsed time = %ld ms, rel. res = %e\n", i, tend, relres);

            if (relres < tol)
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
        // update_ghosts_seq(data[0].f, m, n, o);
        double initres;
        if (stencil_values)
            initres =
                MultigridEngine::stencilResidualSeq(levels[0]->getF(), levels[0]->getV(), levels[0]->getR(), m, n, o, ghosts,
                                                    residual_norm, stencil, *stencil_values, stencil_size_multiplier);
        else
            initres = MultigridEngine::residualSeq(levels[0]->getF(), levels[0]->getV(), levels[0]->getR(), residual_norm, stencil);
        printf("Starting mgcl with initres = %e\n", initres);

        // run vcycle maxiter_vcycles times
        double res, relres;
        for (int i = 0; i < maxiter_vcycles; i++)
        {
            auto tstart = std::chrono::steady_clock::now();
            res = MultigridEngine::vcycleSeq(*this, *levels[0]);
            auto tend = mgcl_since(tstart).count();
            relres = initres == 0 ? 0 : res / initres;
            printf("iter = %d, elapsed time = %ld ms, rel. res = %e\n", i, tend, relres);

            if (relres < tol)
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

    MGCL_STENCIL Problem::getStencil() const
    {
        return stencil;
    }

    void Problem::setStencil(const MGCL_STENCIL &stencil_)
    {
        stencil = stencil_;
    }

    int Problem::getStencilSizeMultiplier() const
    {
        return stencil_size_multiplier;
    }

    void Problem::setStencilSizeMultiplier(int stencilSizeMultiplier)
    {
        stencil_size_multiplier = stencilSizeMultiplier;
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

    Cuboid &Problem::getStencilValues() const
    {
        return *stencil_values;
    }

    std::shared_ptr<Cuboid> Problem::getStencilValuesPtr() const
    {
        return stencil_values;
    }

    void Problem::setStencilValues(std::shared_ptr<Cuboid> stencilValues)
    {
        stencil_values = stencilValues;
    }

    bool Problem::getRestrictProlongateStencil() const
    {
        return restrict_prolongate_stencil;
    }

    void Problem::setRestrictProlongateStencil(bool restrictProlongateStencil)
    {
        restrict_prolongate_stencil = restrictProlongateStencil;
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

    std::vector<std::shared_ptr<Level>> Problem::getLevels() const
    {
        return levels;
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

    cl_program Problem::getProgram() const
    {
        return openCLHelper.program;
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