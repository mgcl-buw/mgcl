#include "problem.hpp"
#include "cuboid.hpp" // for Cuboid
#include "level.hpp"  // for Level
#include "mpi_level_data.hpp"
#include "mpi_util.hpp"
#include "multigrid_engine.hpp" // for Problem, MultigridEngine
#include "util.hpp"

#include <CL/cl_platform.h> // for cl_ulong
#include <algorithm>        // for max
#include <chrono>           // for __enable_if_is_duration, steady_clock
#include <cmath>            // for log2
#include <cstdio>           // for printf, NULL
#include <functional>       // for function
#include <iostream>
#include <stdexcept>   // for runtime_error
#include <string>      // for to_string, basic_string, string
#include <sys/types.h> // for ulong
#include <utility>     // for move

namespace mgcl
{

    Problem::Problem(int m_, int n_, int o_, int m_global_, int n_global_, int o_global_)
        : m(m_), n(n_), o(o_),
          mGlobal(m_global_ == -1 ? m_ : m_global_),
          nGlobal(n_global_ == -1 ? n_ : n_global_),
          oGlobal(o_global_ == -1 ? o_ : o_global_),
          openCLHelper(this)
    {
        checkGlobalDimensions();
        calculateAndSetMaxLevel();
    }

    Problem::Problem(int m_, int n_, int o_, Cuboid* f_, Cuboid* v_, int m_global_, int n_global_, int o_global_)
        : m(m_), n(n_), o(o_), f(f_), v(v_),
          mGlobal(m_global_ == -1 ? m_ : m_global_),
          nGlobal(n_global_ == -1 ? n_ : n_global_),
          oGlobal(o_global_ == -1 ? o_ : o_global_),
          openCLHelper(this)
    {
        checkGlobalDimensions();
        calculateAndSetMaxLevel();
    }

    Problem::Problem(int m_, int n_, int o_, std::shared_ptr<Cuboid> f_, std::shared_ptr<Cuboid> v_,
                     int m_global_, int n_global_, int o_global_)
        : m(m_), n(n_), o(o_), f(f_), v(v_),
          mGlobal(m_global_ == -1 ? m_ : m_global_),
          nGlobal(n_global_ == -1 ? n_ : n_global_),
          oGlobal(o_global_ == -1 ? o_ : o_global_),
          openCLHelper(this)
    {
        checkGlobalDimensions();
        calculateAndSetMaxLevel();
    }

    Problem::Problem(int m_, int n_, int o_, cl_mem d_f_, cl_mem d_v_,
                     int m_global_, int n_global_, int o_global_)
        : m(m_), n(n_), o(o_), dF(d_f_), dV(d_v_),
          mGlobal(m_global_ == -1 ? m_ : m_global_),
          nGlobal(n_global_ == -1 ? n_ : n_global_),
          oGlobal(o_global_ == -1 ? o_ : o_global_),
          openCLHelper(this)
    {
        checkGlobalDimensions();
        calculateAndSetMaxLevel();
    }

    // throws an exception if global dimensions are not a multiple of local dims.
    void Problem::checkGlobalDimensions()
    {
        if (mGlobal <= 0 || mGlobal % m != 0)
            throw "mGlobal must be a multiple of m!";

        if (nGlobal <= 0 || nGlobal % n != 0)
            throw "nGlobal must be a multiple of n!";

        if (oGlobal <= 0 || oGlobal % o != 0)
            throw "oGlobal must be a multiple of o!";
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
            throw "mgcl: supplied v or f and d_v or d_f is nullptr. Aborting.\n";
        }

        if (m < 1 || n < 1 || o < 1)
        {
            throw "mgcl: m, n or o not supplied, zero or negative. Aborting.\n";
        }

        if (ghosts < 1)
        {
            throw "mgcl: ghosts must be >= 1. Aborting.\n";
        }

        if (ghosts_in < 0)
        {
            throw "mgcl: ghosts_in must be >= 0. Aborting.\n";
        }

        if (ghosts_in <= 0 && bc != BC::PERIODIC)
        {
            throw "mgcl: ghosts_in must be > 0 if boundary conditions are not periodic. Aborting.\n";
        }

        // clang-format off
        if (v && f && (
            ghosts_in != v->getGhostsM() || ghosts_in != v->getGhostsN() || ghosts_in != v->getGhostsO() ||
            ghosts_in != f->getGhostsM() || ghosts_in != f->getGhostsN() || ghosts_in != f->getGhostsO()
            ))
            // clang-format on
            throw "ghosts_in is different than ghosts of v and/or f!";

        return true;
    }

    /**
     * @brief Checks if there is enough space available on the OpenCL device s.t. every buffer that is needed
     * can be created. Sets maxlevel and initialized OpenCL environment if not done yet.
     *
     * @return true Enough space available.
     * @return false Otherwise.
     */
    bool Problem::checkGpuSizes()
    {
        // Set maxlevel if not done yet.
        if (maxlevel == -1)
            calculateAndSetMaxLevel();

        // Init OpenCL if not done yet.
        if (!openCLHelper.isInitialized())
            initOpenCL();

        // Check which size can be allocated.
        cl_ulong sizeAvailable;
        cl_ulong sizeAllocablePerBuffer;
        int err = clGetDeviceInfo(getDeviceId(), CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong),
                                  &sizeAvailable, nullptr);
        mgcl::mgclCheckError(err, "clGetDeviceInfo(CL_DEVICE_GLOBAL_MEM_SIZE)");
        err = clGetDeviceInfo(getDeviceId(), CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong),
                              &sizeAllocablePerBuffer, nullptr);
        mgcl::mgclCheckError(err, "clGetDeviceInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE)");

        // Buffers needed for each level, dependent on settings, are:
        // Permanent: dVIn, dVOut, dF, dR, stencilValuesGpu
        // Temporary galerkin results: sr, sp, sr * a_h, sas = sr * a_h * sp
        ulong sizeNeeded = 0;
        ulong maxBufferSizeNeeded = 0;

        // updates sizeNeeded and maxBufferSizeNeeded
        std::function<void(ulong)> upd = [&sizeNeeded, &maxBufferSizeNeeded](ulong inc)
        {
            sizeNeeded += inc;
            maxBufferSizeNeeded = std::max(maxBufferSizeNeeded, inc);
        };

        for (int l = 0; l <= maxlevel; l++)
        {
            int ml = (m >> l);
            int nl = (n >> l);
            int ol = (o >> l);
            int mgh = ml + 2 * ghosts;
            int ngh = nl + 2 * ghosts;
            int ogh = ol + 2 * ghosts;

            if ((l == 0 && !reuse_opencl_buffers) || l > 0)
            {
                upd(sizeof(double) * mgh * ngh * ogh); // dVIn
                upd(sizeof(double) * mgh * ngh * ogh); // dF
            }

            upd(sizeof(double) * mgh * ngh * ogh); // dVOut
            upd(sizeof(double) * mgh * ngh * ogh); // dR

            if (stencilType == MGCL_VARYING)
            {
                // Ghost cell amount per border of varying stencil is 2 for each level (required for galerkin)
                int gh = 2;
                upd(sizeof(double) * (ml + 2 * gh) * (nl + 2 * gh) * (ol + 2 * gh) * 3 * 3 * 3); // stencilValues

                // Temporary buffers created in galerkin
                upd(sizeof(double) * 3 * 3 * 3); // full-weight restriction stencil
                upd(sizeof(double) * 3 * 3 * 3); // bilinear prolongation stencil

                // intermediate result of sr * a_h, gh = 2
                upd(sizeof(double) * (m + 2 * gh) * (n + 2 * gh) * (o + 2 * gh) * 5 * 5 * 5);

                // intermediate result of sas = sr * a_h * sp, gh = 0
                upd(sizeof(double) * ml * nl * ol * 7 * 7 * 7);
            }
        }

        if (maxBufferSizeNeeded > sizeAllocablePerBuffer || sizeNeeded > sizeAvailable)
        {
            std::string msg("Not enough memory available on device for a grid of size ");
            msg.append(std::to_string(m))
                .append("x")
                .append(std::to_string(n))
                .append("x")
                .append(std::to_string(o))
                .append("!\n")
                .append("  Total global memory available: ")
                .append(std::to_string(sizeAvailable / 1024 / 1024))
                .append(" MiB\n      Max. allocable per buffer: ")
                .append(std::to_string(sizeAllocablePerBuffer / 1024 / 1024))
                .append(" MiB\n     Total global memory needed: ")
                .append(std::to_string(sizeNeeded / 1024 / 1024))
                .append(" MiB\n        Max. buffer size needed: ")
                .append(std::to_string(maxBufferSizeNeeded / 1024 / 1024))
                .append(" MiB");

            throw msg;
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
        int minsize = mGlobal < nGlobal ? mGlobal : nGlobal;
        minsize = minsize < oGlobal ? minsize : oGlobal;
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
     * @brief Sets the threshold for levels that will be calculated on multiple MPI processes.
     * All levels below the threshold (exclusively) will be calculated on multiple MPI processes. All levels at or
     *   above the threshold (inclusively) will be calculated locally on only one process. I.e. for threshold = 0,
     *   all levels will be calculated on only one process.
     * Level indices are 0-based, i.e. the finest level has index 0, the coarsest one has index log2(min(mg,ng,og)),
     *   where mg,ng,og are the global sizes of the domain.
     * The threshold must not exceed the level on which min(ml,nl,ol) <= ghosts, e.g. for 8 processes, global size of
     *   8x8x8 and ghosts = 2, the maximum level is 3 and the maximum valid threshold is 1, since starting with level
     *   2, ml <= ghosts. This constraint exists because the update of ghost cells is too expensive for ml <= ghosts.
     *   It is not checked in this function but later in checkParameters(), which is called when solve()
     *   is called.
     *
     */
    void Problem::calculateAndSetMpiLevelThreshold()
    {
        if (mpiMinGridPoints <= 1)
            throw "mpiMinGridPoints must be at least 2!";

        mpiLevelThreshold = static_cast<int>(log2(util::seq::min3(m, n, o))) -
                            (static_cast<int>(log2(mpiMinGridPoints)) - 1);

        if (mpiLevelThreshold < 0)
            throw "mpiMinGridPoints is too high! It must be less than or equal to local min(m,n,o).";

        if (!silent)
            std::cout << "mpiLevelThreshold automatically set to " << mpiLevelThreshold << std::endl;
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

        calculateAndSetMaxLevel();
        if (!silent)
            printf("maxlevel = %d\n", maxlevel);

        if (useMpi())
        {
            // Create cartesian process grid if none was set and more than one processes are used.
            mpiGlobalData->createCartGrid(isPeriodic());
            calculateAndSetMpiLevelThreshold();
        }

        // create opencl environment with default parameters if not done yet
        if (use_opencl)
        {
            if (initOpenCL() != CL_SUCCESS)
                return false;

            if (!checkGpuSizes())
                return false;
        }

        // check opencl components if device buffers should be reused
        if (reuse_opencl_buffers || copy_buffer_data)
        {
            if (!openCLHelper.checkParameters())
                return false;
        }

        // initialize levels
        for (int level = 0; level <= maxlevel; level++)
        {
            auto lv = std::make_unique<Level>(this, level);
            levels.push_back(std::move(lv));
            levels.back()->init();

            // Apply Galerkin operator if stencil is varying and we're not on level 0.
            if (levels[0]->getStencilValues() && level >= 1)
            {
                if (!use_opencl)
                {
                    // Gather
                    if (useMpi() && getMpiLevelThreshold() == levels[level - 1]->getNum())
                    {
                    }

                    levels.back()->stencilValues = std::make_shared<VaryingStencil3x3x3>(
                        MultigridEngine::galerkin(*levels[level - 1]->getStencilValues()));
                }
                else
                {
                    // Gather
                    if (useMpi() && getMpiLevelThreshold() == levels[level - 1]->getNum())
                    {
                    }

                    levels.back()->stencilValuesGpu = std::make_shared<VaryingStencilGpu>(
                        MultigridEngine::galerkin(
                            *levels[level - 1]->getStencilValuesGpu(),
                            getProgram(), getCommands(), getContext()));
                }
            }
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
            {
                if (ignoreTol)
                    printf("iter = %d, elapsed time = %ld ms\n", i, tend);
                else
                    printf("iter = %d, elapsed time = %ld ms, rel. res = %e\n", i, tend, relres);
            }

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

        // Edge case: Do nothing if mpi is used but level threshold is 0 (i.e. all work is done on proc 0).
        if (!(useMpi() && getMpiLevelThreshold() <= 0 && mpiRank() > 0))
        {

            // calculate initial residual (different from pmg's initres bc ghosts are not updated in pmg first)
            if (isPeriodic())
                MultigridEngine::updateGhostsSeq(levels[0]->getV(), levels[0]->getMpiDataPtr(), isPeriodic(),
                                                 levels[0]->isCalculatedLocally());

            double initres = MultigridEngine::residualSeq(levels[0]->getF(), levels[0]->getV(), levels[0]->getR(),
                                                          residual_norm, stencilType, levels[0]->stencilFactor,
                                                          levels[0]->stencilValues.get(), !ignoreTol, isPeriodic(),
                                                          levels[0]->isCalculatedLocally(),
                                                          0, 0, 0, getLevelAt(0).getMpiDataPtr());
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
                {
                    if (ignoreTol)
                        printf("iter = %d, elapsed time = %ld ms\n", i, tend);
                    else
                        printf("iter = %d, elapsed time = %ld ms, rel. res = %e\n", i, tend, relres);
                }

                if (!ignoreTol && relres < tol)
                    break;
            }
        }
        else if (!silent)
        {
            std::cout << "mgcl: Rank " << mpiRank()
                      << " does nothing (mpiLevelThreshold <= 0). Waiting for results from rank 0 ..." << std::endl;
        }

        // TODO scatter from proc 0 after solving if useMpi() && getMpiLevelThreshold() == 0
        // TODO check mpiSize > 1 maybe
        if (useMpi() && getMpiLevelThreshold() == 0)
        {
            mpi_util::scatter_inplace(mpiGlobalData->getComm(), getLevelAt(0).getV());
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

    Level& Problem::getLevelAt(int index) const
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

    Cuboid& Problem::getF() const
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

    void Problem::setDV(const cl_mem& dV_)
    {
        dV = dV_;
    }

    void Problem::setDStencilValues(const cl_mem& dStencilValues_)
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

    void Problem::setResidualNorm(const MGCL_RESIDUAL_NORM& residualNorm)
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

    void Problem::setDF(const cl_mem& dF_)
    {
        dF = dF_;
    }

    OpenCLHelper& Problem::getOpenCLHelper()
    {
        return openCLHelper;
    }

    std::string Problem::getDeviceName() const
    {
        return openCLHelper.deviceName;
    }

    void Problem::setDeviceName(const std::string& deviceName_)
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

    void Problem::setKernelDir(const std::string& kernelDir_)
    {
        if (!openCLHelper.isInitialized())
            openCLHelper.kernelDir = kernelDir_;
    }

    cl_device_type Problem::getDeviceType() const
    {
        return openCLHelper.deviceType;
    }

    void Problem::setDeviceType(const cl_device_type& deviceType_)
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

    MGCL_STENCIL Problem::getStencilType() const
    {
        return stencilType;
    }

    BC Problem::getBc() const
    {
        return bc;
    }

    void Problem::setBc(const BC& bc_)
    {
        bc = bc_;
    }

    cl_program Problem::getProgram() const
    {
        return openCLHelper.program;
    }

    bool Problem::getIgnoreTol() const
    {
        return ignoreTol;
    }

    void Problem::setStencilType(const MGCL_STENCIL& stencilType_)
    {
        stencilType = stencilType_;
        // TODO move to getStencilValues?
        if (stencilType == MGCL_VARYING)
        {
            int gh = std::max(2, jacobi_iterations_per_kernel);
            stencilValues = std::make_shared<VaryingStencil3x3x3>(m, n, o, gh, gh, gh);
        }
        else
            stencilValues = nullptr;
    }

    std::shared_ptr<VaryingStencil3x3x3>& Problem::getStencilValues()
    {
        return stencilValues;
    }

    void Problem::setIgnoreTol(bool ignoreTol_)
    {
        ignoreTol = ignoreTol_;
    }

    Cuboid& Problem::getV() const
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

    // Sets MPI communicator and checks that cartesian topology is attached to it.
    void Problem::setMpiComm(MPI_Comm _comm)
    {
        mpiGlobalData->setComm(_comm);
    }

    MPI_Comm Problem::getMpiComm()
    {
        return mpiGlobalData->getComm();
    }

    int Problem::getMpiLevelThreshold()
    {
        return mpiLevelThreshold;
    }

    int Problem::getMpiMinGridPoints() const
    {
        return mpiMinGridPoints;
    }

    void Problem::setMpiMinGridPoints(int mpiMinGridPoints_)
    {
        mpiMinGridPoints = mpiMinGridPoints_;
    }

    /**
     * @brief Returns true, if the program is run with more than one MPI processes. Only then the MPI routines
     * will be used internally.
     */
    bool Problem::useMpi() // TODO maybe as attribute and setter/getter
    {
        return mpiSize() > 1;
    }

    int Problem::mpiRank() const
    {
        return mpiGlobalData->mpiRank();
    }

    /**
     * @brief Returns the communicator size, i.e. number of processes attached to this communicator.
     */

    int Problem::mpiSize()
    {
        return mpiGlobalData->mpiSize();
    }

    int Problem::getMGlobal() const
    {
        return mGlobal;
    }

    int Problem::getNGlobal() const
    {
        return nGlobal;
    }

    int Problem::getOGlobal() const
    {
        return oGlobal;
    }

    std::ostream& operator<<(std::ostream& os, const Problem& p)
    {
        os << "Problem: " << std::endl
           << " m,n,o: " << p.m << "," << p.n << "," << p.o << std::endl
           << " mGlobal,nGlobal,oGlobal: " << p.mGlobal << "," << p.nGlobal << "," << p.oGlobal << std::endl
           << " ghosts: " << p.ghosts << std::endl
           << " ghosts_in: " << p.ghosts_in << std::endl
           << " maxlevel: " << p.maxlevel << std::endl
           << " maxiter_vcycles: " << p.maxiter_vcycles << std::endl
           << " nu1,nu2: " << p.nu1 << "," << p.nu2 << std::endl
           << " omega: " << p.omega << std::endl
           << " tol: " << p.tol << std::endl
           << " ignoreTol: " << p.ignoreTol << std::endl
           << " residual_norm: " << p.residual_norm << std::endl
           << " stencilType: " << p.stencilType << std::endl
           << " use_opencl: " << p.use_opencl << std::endl
           << " reuse_opencl_buffers: " << p.reuse_opencl_buffers << std::endl
           << " jacobi_iterations_per_kernel: " << p.jacobi_iterations_per_kernel << std::endl
           << " silent: " << p.silent << std::endl
           << " mpi_rank: " << p.mpiGlobalData->mpiRank() << std::endl
           << " mpiLevelThreshold: " << p.mpiLevelThreshold << std::endl
           << " mpiMinGridPoints: " << p.mpiMinGridPoints << std::endl;

        return os;
    }
}