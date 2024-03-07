#include "problem.hpp"
#include "cuboid.hpp" // for Cuboid
#include "level.hpp"  // for Level
#include "mpi_global_data.hpp"
#include "mpi_stencil.hpp"
#include "mpi_util.hpp"
#include "multigrid_engine.hpp" // for Problem, MultigridEngine
#include "profiling_data.hpp"
#include "util.hpp"
#include <memory>

#ifdef __APPLE__
#include <OpenCL/cl_platform.h> // for cl_ulong
#define ulong unsigned long
#else
#include <CL/cl_platform.h> // for cl_ulong
#endif
#include <algorithm>  // for max
#include <chrono>     // for __enable_if_is_duration, steady_clock
#include <cmath>      // for log2
#include <cstdio>     // for printf, NULL
#include <functional> // for function
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

    Problem::Problem(int m_, int n_, int o_, std::shared_ptr<CuboidGpu> d_f_, std::shared_ptr<CuboidGpu> d_v_,
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
     * @brief Checks mandatory configuration fields: m, n, o, f, v, d_v, d_f, ghosts and ghosts_in.
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

        if (mpiRank() == 0 && getMpiLevelThreshold() == 0 && stencilValues &&
            (stencilValues->getM() < mGlobal || stencilValues->getN() < nGlobal || stencilValues->getO() < oGlobal))
            throw "Mpi threshold level is 0 but stencilValues has local size. Please use setMpiMinGridPoints before setStencilType!";

        if (mpiRank() == 0 && getMpiLevelThreshold() > 1 && stencilValues &&
            (stencilValues->getM() > m || stencilValues->getN() > n || stencilValues->getO() > o))
            throw "Mpi threshold level is not 0 but stencilValues has global size. Please use setMpiMinGridPoints before setStencilType!";

        if (stencilValues && (stencilValues->getGhostsM() < ghosts ||
                              stencilValues->getGhostsN() < ghosts ||
                              stencilValues->getGhostsO() < ghosts))
            throw "Ghosts of stencilValues must be >= ghosts. Make sure to call setGhosts and setJacobiIterationsPerKernel before setStencilType!";

        return true;
    }

    /**
     * @brief Checks if there is enough space available on the OpenCL device s.t. every buffer that is needed
     * can be created. Sets maxlevel and initialized OpenCL environment if not done yet.
     * @throws string If not enough space is available.
     */
    void Problem::checkGpuSizes()
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
        // Permanent: dVIn, dVOut, dF, dR, dRsq, stencilValuesGpu
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
            upd(sizeof(double) * ml * nl * ol);    // dRsq

            if (stencilType == MGCL_VARYING)
            {
                // Ghost cell amount per border of varying stencil is 2 for each level (required for galerkin)
                int gh = 2;
                upd(sizeof(double) * (ml + 2 * gh) * (nl + 2 * gh) * (ol + 2 * gh) * 3 * 3 * 3); // stencilValues

                // Temporary buffers created in galerkin
                upd(sizeof(double) * 3 * 3 * 3); // full-weight restriction stencil
                upd(sizeof(double) * 3 * 3 * 3); // bilinear prolongation stencil

                // intermediate result of sr * a_h, gh = 2
                upd(sizeof(double) * (ml + 2 * gh) * (nl + 2 * gh) * (ol + 2 * gh) * 5 * 5 * 5);

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
        if (!useMpi())
            return;

        if (mpiMinGridPoints <= 1)
            throw "mpiMinGridPoints must be at least 2!";

        // Threshold was already set (automatically or by user).
        if (mpiLevelThreshold >= 0)
            return;

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
            initOpenCL();
            checkGpuSizes();
        }

        // check opencl components if device buffers should be reused
        if (reuse_opencl_buffers || copy_buffer_data)
        {
            if (!openCLHelper.checkParameters())
                return false;
        }

        // initialize levels
        // gathered flag needed for enforcing local ghost update after stencil values were gathered. If threshold is 0,
        // no gathering happens at all.
        bool gathered = getMpiLevelThreshold() == 0;
        int gh_sv = stencilValues ? stencilValues->getGhostsM() : 0;
        for (int level = 0; level <= maxlevel; level++)
        {
            {
                auto lv = std::make_unique<Level>(this, level);
                levels.push_back(std::move(lv)); // lv is invalid after this line, thus restrict the visibility
                levels.back()->init();
            }

            // Apply Galerkin operator if stencil is varying and we're not on level 0.
            if (levels[0]->getStencilValues() && level >= 1 &&
                levels.back()->getM() > 0 && levels.back()->getN() > 0 && levels.back()->getO() > 0)
            {
                auto& lvFine = *levels[level - 1];
                auto& lvCoarse = *levels[level];
                bool isJustAboveThreshold = level == getMpiLevelThreshold() - 1;
                bool updateGhostsCoarse = !useMpi() || mpiSize() == 1 || !isJustAboveThreshold;

                // stencilValues of this level must be of global size on rank 0, if this level is just above
                // the threshold, since it is getting gathered into.
                int svm = (mpiRank() == 0 && isJustAboveThreshold ? (mGlobal >> level) : lvCoarse.getM());
                int svn = (mpiRank() == 0 && isJustAboveThreshold ? (nGlobal >> level) : lvCoarse.getN());
                int svo = (mpiRank() == 0 && isJustAboveThreshold ? (oGlobal >> level) : lvCoarse.getO());

                if (!use_opencl)
                {
                    // Gather partial stencil values of the previous level
                    if (useMpi() && getMpiLevelThreshold() == lvCoarse.getNum())
                    {

                        mpi_util::gather(getMpiComm(), *lvFine.getStencilValues());
                        gathered = true;
                        updateGhostsStencilMpi(*lvFine.getStencilValues(), lvFine.getMpiDataPtr(), isPeriodic(), true);
                    }

                    // Only calculate galerkin if
                    // 1. MPI is not used at all, or
                    // 2. this level is calculated distributively, or
                    // 3. this level is calculated locally and rank is 0.
                    // Otherwise we would run into neighbour issues when trying to update ghosts.
                    if (!useMpi() || !lvCoarse.isCalculatedLocally() || mpiRank() == 0)
                        lvCoarse.stencilValues = std::make_shared<VaryingStencil>(
                            MultigridEngine::galerkin(*lvFine.getStencilValues(), gh_sv,
                                                      lvFine.getMpiDataPtr(), lvCoarse.getMpiDataPtr(),
                                                      isPeriodic(), gathered,
                                                      lvCoarse.isCalculatedLocally(), !updateGhostsCoarse,
                                                      svm, svn, svo));
                }
                else
                {
                    // Gather partial stencil values of the previous level
                    if (useMpi() && getMpiLevelThreshold() == lvCoarse.getNum())
                    {
                        mpi_util::gather(getMpiComm(), getCommands(), *lvFine.getStencilValuesGpu());
                        gathered = true;
                        updateGhostsStencilOclMpi(getCommands(), getProgram(), *lvFine.getStencilValuesGpu(),
                                                  lvFine.getMpiDataPtr(), isPeriodic(), true, &getKernelConfig(),
                                                  getProfilingData());
                    }

                    // Only calculate galerkin if
                    // 1. MPI is not used at all, or
                    // 2. this level is calculated distributively, or
                    // 3. this level is calculated locally and rank is 0.
                    // Otherwise we would run into neighbour issues when trying to update ghosts.
                    if (!useMpi() || !lvCoarse.isCalculatedLocally() || mpiRank() == 0)
                        levels.back()->stencilValuesGpu = std::make_shared<VaryingStencilGpu>(
                            MultigridEngine::galerkin(
                                *lvFine.getStencilValuesGpu(), gh_sv,
                                getProgram(), getCommands(), getContext(),
                                lvFine.getMpiDataPtr(), lvCoarse.getMpiDataPtr(),
                                isPeriodic(), gathered,
                                lvCoarse.isCalculatedLocally(), !updateGhostsCoarse,
                                &getKernelConfig(), getProfilingData(), svm, svn, svo));
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
    void Problem::reuseOpenCL(cl_context context, cl_command_queue commandQueue, cl_device_id deviceId)
    {
        openCLHelper.release();

        openCLHelper.setContext(context);
        openCLHelper.setCommands(commandQueue);
        openCLHelper.setDeviceId(deviceId);
        setUseOpencl(true);

        openCLHelper.init();
    }

    /**
     * @brief Lazyly initializes OpenCL environment.
     *
     * @return int OpenCL error code
     */
    void Problem::initOpenCL()
    {
        if (!openCLHelper.isInitialized())
            openCLHelper.init();
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
        err = clEnqueueReadBuffer(openCLHelper.getCommands(), levels[0]->getDVIn().getBuffer(), CL_TRUE, 0,
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
     * @brief Calls solve(false), i.e. calls Problem::init first.
     *
     */
    void Problem::solve()
    {
        solve(false);
    }

    /**
     * @brief Solves problem using multigrid method on OpenCL, if use_opencl is true. Otherwise solveSeq is called.
     *
     * @param skipInit If true, skips initialization. Requires Problem::init to be called first. Mainly used for
     * testing and benchmarking.
     */
    void Problem::solve(bool skipInit)
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
        if (!skipInit && !init())
            throw std::runtime_error("Failed to initialize mgcl data structures.");

        // Edge case: Do nothing if mpi is used but level threshold is 0 (i.e. all work is done on proc 0).
        if (!(useMpi() && getMpiLevelThreshold() <= 0 && mpiRank() > 0))
        {

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
            mpi_util::scatter_inplace_wgh(mpiGlobalData->getComm(), getCommands(), getLevelAt(0).getDVIn());
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
        if (stencilValues && (stencilValues->getGhostsM() < jacobiIterationsPerKernel ||
                              stencilValues->getGhostsN() < jacobiIterationsPerKernel ||
                              stencilValues->getGhostsO() < jacobiIterationsPerKernel))
            throw "Ghosts of stencilValues must be >= ghosts. Make sure to call setGhosts and setJacobiIterationsPerKernel before setStencilType!";

        jacobi_iterations_per_kernel = jacobiIterationsPerKernel;
    }

    CuboidGpu& Problem::getDStencilValues() const
    {
        if (!dStencilValues)
            throw "dStencilValues is null!";
        return *dStencilValues;
    }

    std::shared_ptr<CuboidGpu> Problem::getDStencilValuesPtr() const
    {
        return dStencilValues;
    }

    void Problem::setDStencilValues(std::shared_ptr<CuboidGpu> dStencilValues_)
    {
        dStencilValues = dStencilValues_;
    }

    CuboidGpu& Problem::getDV() const
    {
        if (!dV)
            throw "dStencilValues is null!";
        return *dV;
    }

    std::shared_ptr<CuboidGpu> Problem::getDVPtr() const
    {
        return dV;
    }

    void Problem::setDV(std::shared_ptr<CuboidGpu> dV_)
    {
        dV = dV_;
    }

    CuboidGpu& Problem::getDF() const
    {
        if (!dF)
            throw "dF is null!";
        return *dF;
    }

    std::shared_ptr<CuboidGpu> Problem::getDFPtr() const
    {
        return dF;
    }

    void Problem::setDF(std::shared_ptr<CuboidGpu> dF_)
    {
        dF = dF_;
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
        if (stencilValues && (stencilValues->getGhostsM() < ghosts_ ||
                              stencilValues->getGhostsN() < ghosts_ ||
                              stencilValues->getGhostsO() < ghosts_))
            throw "Ghosts of stencilValues must be >= ghosts. Make sure to call setGhosts and setJacobiIterationsPerKernel before setStencilType!";

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

    std::string Problem::getKernelFile() const
    {
        return openCLHelper.getKernelFile();
    }

    void Problem::setKernelFile(const std::string& kernelFile_)
    {
        if (!openCLHelper.isInitialized())
            openCLHelper.setKernelFile(kernelFile_);
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
            calculateAndSetMpiLevelThreshold();
            int gh = std::max(2, jacobi_iterations_per_kernel);
            if (getMpiLevelThreshold() <= 1 && mpiRank() == 0)
                stencilValues = std::make_shared<VaryingStencil>(mGlobal, nGlobal, oGlobal, 3, gh, gh, gh);
            else
                stencilValues = std::make_shared<VaryingStencil>(m, n, o, 3, gh, gh, gh);
        }
        else
            stencilValues = nullptr;
    }

    std::shared_ptr<VaryingStencil>& Problem::getStencilValues()
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

    /**
     * @brief Toggles collection of GPU kernel profiling data. Sets or resets profilingData.
     *
     * @param profilingEnabled_
     */
    void Problem::setProfilingEnabled(bool profilingEnabled_)
    {
        profilingEnabled = profilingEnabled_;
        if (profilingEnabled)
            profilingData = std::make_unique<mgcl::ProfilingData>();
        else
            profilingData = nullptr;
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

    /**
     * @brief
     * All levels below the threshold (exclusively) will be calculated on multiple MPI processes. All levels at or
     *   above the threshold (inclusively) will be calculated locally on only one process. I.e. for threshold = 0,
     *   all levels will be calculated on only one process.
     */
    int Problem::getMpiLevelThreshold()
    {
        return mpiLevelThreshold;
    }

    /**
     * @brief Must be between 0 and maxlevel, which is log2(min(mGlobal, nGlobal, oGlobal)) or the maxlevel set by user.
     * Also updates mpiMinGridPoints.
     *
     * @param mpiLevelThreshold_ Number of level between 0 and maxlevel
     */
    void Problem::setMpiLevelThreshold(int mpiLevelThreshold_)
    {
        if (mpiLevelThreshold_ < 0)
            throw "MpiLevelThreshold cannot be negative";
        if (mpiLevelThreshold_ > maxlevel)
            throw "MpiLevelThreshold cannot be larger than maxlevel (" + std::to_string(maxlevel) + ")";
        mpiLevelThreshold = mpiLevelThreshold_;

        // update mpiMinGridPoints
        mpiMinGridPoints = std::pow(2, maxlevel - mpiLevelThreshold);
    }

    int Problem::getMpiMinGridPoints() const
    {
        return mpiMinGridPoints;
    }

    void Problem::setMpiMinGridPoints(int mpiMinGridPoints_)
    {
        mpiMinGridPoints = mpiMinGridPoints_;
        calculateAndSetMpiLevelThreshold();
    }

    /**
     * @brief Returns true, if the program is run with more than one MPI processes and ignoreMpi is false.
     * Only then the MPI routines will be used internally.
     */
    bool Problem::useMpi() // TODO maybe as attribute and setter/getter
    {
        return !getIgnoreMpi() && mpiSize() > 1;
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
