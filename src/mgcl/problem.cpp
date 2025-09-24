#include "problem.hpp"
#include "blockstencil.hpp"
#include "cuboid.hpp" // for Cuboid
#include "fixed_blockstencil.hpp"
#include "level.hpp" // for Level
#include "mgcl.hpp"
#include "mpi_global_data.hpp"
#include "mpi_stencil.hpp"
#include "mpi_util.hpp"
#include "multigrid_engine.hpp" // for Problem, MultigridEngine
#include "opencl_helper.hpp"
#include "profiling_data.hpp"
#include "stencil.hpp"
#include "util.hpp"
#include <CL/cl.h>
#include <cassert>
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

    Problem::Problem(int m_, int n_, int o_,
                     std::shared_ptr<CuboidBS> f_, std::shared_ptr<CuboidBS> v_,
                     int m_global_, int n_global_, int o_global_)
        : m(m_), n(n_), o(o_), f_bs(f_), v_bs(v_), blocksize(v_->getBlocksize()),
          mGlobal(m_global_ == -1 ? m_ : m_global_),
          nGlobal(n_global_ == -1 ? n_ : n_global_),
          oGlobal(o_global_ == -1 ? o_ : o_global_),
          openCLHelper(this)
    {
        if (v_bs->getBlocksize() != f_bs->getBlocksize())
        {
            error("v and f must have the same blocksize!");
        }

        if (blocksize < 1)
        {
            error("blocksize must be > 0!");
        }

        setStencilType(MGCL_BLOCKSTENCIL);

        checkGlobalDimensions();
        calculateAndSetMaxLevel();
    }

    // throws an exception if global dimensions are not a multiple of local dims.
    void Problem::checkGlobalDimensions()
    {
        if (mGlobal <= 0 || mGlobal % m != 0)
            error("mGlobal must be a multiple of m!");

        if (nGlobal <= 0 || nGlobal % n != 0)
            error("nGlobal must be a multiple of n!");

        if (oGlobal <= 0 || oGlobal % o != 0)
            error("oGlobal must be a multiple of o!");
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
        if ((v == nullptr || f == nullptr) && (dV == nullptr || dF == nullptr) && (v_bs == nullptr || f_bs == nullptr))
        {
            error("mgcl: supplied v or f and d_v or d_f and v_bs or f_bs is nullptr. Aborting.\n");
        }

        if (v_bs && f_bs && !v && !f && !dV && !dF && stencilType != MGCL_BLOCKSTENCIL)
        {
            error("mgcl: v_bs and f_bs are supplied but stencil is not blockstencil\n");
        }

        if (m < 1 || n < 1 || o < 1)
        {
            error("mgcl: m, n or o not supplied, zero or negative. Aborting.\n");
        }

        if (ghosts < 1)
        {
            error("mgcl: ghosts must be >= 1. Aborting.\n");
        }

        if (ghosts_in < 0)
        {
            error("mgcl: ghosts_in must be >= 0. Aborting.\n");
        }

        if (ghosts_in <= 0 && bc != BC::PERIODIC)
        {
            error("mgcl: ghosts_in must be > 0 if boundary conditions are not periodic. Aborting.\n");
        }

        // clang-format off
        if (v && f && (
            ghosts_in != v->getGhostsM() || ghosts_in != v->getGhostsN() || ghosts_in != v->getGhostsO() ||
            ghosts_in != f->getGhostsM() || ghosts_in != f->getGhostsN() || ghosts_in != f->getGhostsO()
            ))
            // clang-format on
            error("ghosts_in is different than ghosts of v and/or f!");

        // clang-format off
        if (v_bs && f_bs && (
            ghosts_in != v_bs->getGhostsM() || ghosts_in != v_bs->getGhostsN() || ghosts_in != v_bs->getGhostsO() ||
            ghosts_in != f_bs->getGhostsM() || ghosts_in != f_bs->getGhostsN() || ghosts_in != f_bs->getGhostsO()
            ))
            // clang-format on
            error("ghosts_in is different than ghosts of v_bs and/or f_bs!");

        if (mpiRank() == 0 && getMpiLevelThreshold() == 0 && stencilValues &&
            (stencilValues->getM() < mGlobal || stencilValues->getN() < nGlobal || stencilValues->getO() < oGlobal))
            error("Mpi threshold level is 0 but stencilValues has local size. Please use setMpiMinGridPoints before setStencilType!");

        if (mpiRank() == 0 && getMpiLevelThreshold() > 1 && stencilValues &&
            (stencilValues->getM() > m || stencilValues->getN() > n || stencilValues->getO() > o))
            error("Mpi threshold level is not 0 but stencilValues has global size. Please use setMpiMinGridPoints before setStencilType!");

        if (stencilValues && (stencilValues->getGhostsM() < ghosts ||
                              stencilValues->getGhostsN() < ghosts ||
                              stencilValues->getGhostsO() < ghosts))
            error("Ghosts of stencilValues must be >= ghosts. Make sure to call setGhosts and setJacobiIterationsPerKernel before setStencilType!");

        if (mpiRank() == 0 && getMpiLevelThreshold() == 0 && blockstencil &&
            (blockstencil->getM() < mGlobal || blockstencil->getN() < nGlobal || blockstencil->getO() < oGlobal))
            error("Mpi threshold level is 0 but blockstencil has local size. Please use setMpiMinGridPoints before setStencilType!");

        if (mpiRank() == 0 && getMpiLevelThreshold() > 1 && blockstencil &&
            (blockstencil->getM() > m || blockstencil->getN() > n || blockstencil->getO() > o))
            error("Mpi threshold level is not 0 but blockstencil has global size. Please use setMpiMinGridPoints before setStencilType!");

        if (smootherType == MGCL_JACOBI_BLOCK && stencilType != MGCL_BLOCKSTENCIL)
        {
            error("smootherType is set to MGCL_JACOBI_BLOCK but stencilType is not MGCL_BLOCKSTENCIL!");
        }

        if (stencilType == MGCL_BLOCKSTENCIL && (!restrictionBlockstencilAccessed || !prolongationBlockstencilAccessed))
        {
            warning("stencilType is set to MGCL_BLOCKSTENCIL but restrictionBlockstencil or prolongationBlockstencil do not seem to be set!");
        }

        // TODO Is this test needed?
        // if (blockstencil && (blockstencil->getGhostsM() < ghosts ||
        //                      blockstencil->getGhostsN() < ghosts ||
        //                      blockstencil->getGhostsO() < ghosts))
        //     error("Ghosts of blockstencil must be >= ghosts. Make sure to call setGhosts and setJacobiIterationsPerKernel before setStencilType!");

        // error if there is a zero coefficient on the diagonal, since then the Jacobi result will be NaN. For blockstencil,
        // this will be checked for each level after inverting.
        if (stencilValues)
        {
            for (int i = ghosts; i < m + ghosts; i++)
                for (int j = ghosts; j < n + ghosts; j++)
                    for (int k = ghosts; k < o + ghosts; k++)
                    {
                        if ((*stencilValues)[1][1][1][i][j][k] == 0)
                        {
                            error("stencilValues having at least one zero on the diagonal. Did you call Problem::init() before setting the coefficients?");
                        }
                    }
        }
        else if (fixedStencil)
        {
            if ((*fixedStencil)[1][1][1] == 0)
            {
                error("fixedStencil having a zero on the diagonal. Did you call Problem::init() before setting the coefficients?");
            }
        }

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
        // Permanent: dVIn, dVOut, dF, dR, dRsq, stencilValuesGpu, dPlanesBuf (if MPI is in use)
        // Temporary galerkin stencils: sr, sp
        ulong sizeNeeded = 0;
        ulong maxBufferSizeNeeded = 0;

        // updates sizeNeeded and maxBufferSizeNeeded
        std::function<void(ulong)> upd = [&sizeNeeded, &maxBufferSizeNeeded](ulong inc)
        {
            sizeNeeded += inc;
            maxBufferSizeNeeded = std::max(maxBufferSizeNeeded, inc);
        };

        // dPlanesBuf if MPI is in use
        if (useMpi() && mpiSize() > 1)
        {
            int mgh = m + 2 * ghosts;
            int ngh = n + 2 * ghosts;
            int ogh = o + 2 * ghosts;
            int yz = ngh * ogh;
            int xz = mgh * ogh;
            int xy = mgh * ngh;

            int ressize = (2 * yz * ghosts + 2 * xz * ghosts + 2 * xy * ghosts);
            if (stencilType == MGCL_VARYING)
            {
                ressize *= stencilValues->getWidth() * stencilValues->getWidth() * stencilValues->getWidth();
            }
            else if (stencilType == MGCL_BLOCKSTENCIL)
            {
                ressize *= blockstencil->getWidth() * blockstencil->getWidth() * blockstencil->getWidth() * blocksize * blocksize;
            }

            upd(sizeof(double) * ressize);
        }

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
                upd(sizeof(double) * mgh * ngh * ogh * (stencilType == MGCL_BLOCKSTENCIL ? blocksize : 1)); // dVIn
                upd(sizeof(double) * mgh * ngh * ogh * (stencilType == MGCL_BLOCKSTENCIL ? blocksize : 1)); // dF
            }

            upd(sizeof(double) * mgh * ngh * ogh * (stencilType == MGCL_BLOCKSTENCIL ? blocksize : 1)); // dVOut
            upd(sizeof(double) * mgh * ngh * ogh * (stencilType == MGCL_BLOCKSTENCIL ? blocksize : 1)); // dR
            upd(sizeof(double) * ml * nl * ol * (stencilType == MGCL_BLOCKSTENCIL ? blocksize : 1));    // dRsq

            if (stencilType == MGCL_VARYING)
            {
                // Ghost cell amount per border of varying stencil is 1 for each level
                int gh = 1;
                upd(sizeof(double) * (ml + 2 * gh) * (nl + 2 * gh) * (ol + 2 * gh) * 3 * 3 * 3); // stencilValues

                // Temporary buffers created in galerkin
                upd(sizeof(double) * 3 * 3 * 3); // full-weight restriction stencil
                upd(sizeof(double) * 3 * 3 * 3); // bilinear prolongation stencil
            }
            else if (stencilType == MGCL_BLOCKSTENCIL)
            {
                // Ghost cell amount per border of blockstencil is 1 for each level
                int gh = 1;
                upd(sizeof(double) * blocksize * blocksize * (ml + 2 * gh) * (nl + 2 * gh) * (ol + 2 * gh) * 3 * 3 * 3); // blockstencil
                if (smootherType == MGCL_JACOBI_BLOCK)
                    upd(sizeof(double) * blocksize * blocksize * ml * nl * ol); // blockstencil_inv
                else
                    upd(sizeof(double) * blocksize * ml * nl * ol); // blockstencil_inv

                // Temporary buffers created in galerkin
                upd(sizeof(double) * blocksize * blocksize * 3 * 3 * 3); // full-weight restriction stencil
                upd(sizeof(double) * blocksize * blocksize * 3 * 3 * 3); // bilinear prolongation stencil
            }

            else if (stencilType == MGCL_FIXED)
            {
                upd(sizeof(double) * 3 * 3 * 3); // fixedStencil

                // Temporary buffers created in galerkin
                upd(sizeof(double) * 3 * 3 * 3); // full-weight restriction stencil
                upd(sizeof(double) * 3 * 3 * 3); // bilinear prolongation stencil
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

            error(msg);
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
            error("mpiMinGridPoints must be at least 2!");

        // Threshold was already set (automatically or by user).
        if (mpiLevelThreshold >= 0)
            return;

        mpiLevelThreshold = static_cast<int>(log2(util::seq::min3(m, n, o))) -
                            (static_cast<int>(log2(mpiMinGridPoints)) - 1);

        if (mpiLevelThreshold < 0)
            error("mpiMinGridPoints is too high! It must be less than or equal to local min(m,n,o).");

        if (!silent)
            std::cout
                << "mpiLevelThreshold automatically set to " << mpiLevelThreshold << std::endl;
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

            // If using MPI and OCL, create temporary planes buffer for ghost updates once and reuse on each level
            // TODO change ghosts to ghosts_m etc?
            if (useMpi() && mpiSize() > 1)
            {
                int mgh = m + 2 * ghosts;
                int ngh = n + 2 * ghosts;
                int ogh = o + 2 * ghosts;

                int yz = ngh * ogh;
                int xz = mgh * ogh;
                int xy = mgh * ngh;
                int ressize = (2 * yz * ghosts + 2 * xz * ghosts + 2 * xy * ghosts);
                if (stencilType == MGCL_VARYING)
                {
                    ressize *= stencilValues->getWidth() * stencilValues->getWidth() * stencilValues->getWidth();
                }
                else if (stencilType == MGCL_BLOCKSTENCIL)
                {
                    ressize *= blockstencil->getWidth() * blockstencil->getWidth() * blockstencil->getWidth() * blocksize * blocksize;
                }

                dPlanesBuf = std::make_shared<BufferGpu>(getContext(), CL_MEM_READ_WRITE, ressize);

                // TODO not needing this much memory for these?
                hPlanesBufSend = std::make_shared<std::vector<double>>(ressize);
                hPlanesBufRecv = std::make_shared<std::vector<double>>(ressize);
            }

            if (stencilType == MGCL_BLOCKSTENCIL)
            {
                restrictionBlockstencilGpu = std::make_shared<FixedBlockstencilGpu>(*restrictionBlockstencil, getContext(), getCommands());
                prolongationBlockstencilGpu = std::make_shared<FixedBlockstencilGpu>(*prolongationBlockstencil, getContext(), getCommands());
            }
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
        for (int level = 0; level <= maxlevel; level++)
        {
            {
                auto lv = std::make_unique<Level>(this, level);
                levels.push_back(std::move(lv)); // lv is invalid after this line, thus restrict the visibility
                levels.back()->init();
            }

            // Apply Galerkin operator if we're not on level 0, depending on stencil type.
            if (levels.back()->getM() > 0 && levels.back()->getN() > 0 && levels.back()->getO() > 0)
            {
                int gh_sv = stencilValues ? stencilValues->getGhostsM() : 0;
                if (level >= 1 && getStencilType() == MGCL_VARYING)
                {
                    auto& lvFine = *levels[level - 1];
                    auto& lvCoarse = *levels[level];

                    // stencilValues of this level must be of global size on rank 0, if this level is at
                    // the threshold, since it is getting gathered into.
                    int svm = (mpiRank() == 0 && lvCoarse.getNum() >= getMpiLevelThreshold() ? (mGlobal >> level) : lvCoarse.getM());
                    int svn = (mpiRank() == 0 && lvCoarse.getNum() >= getMpiLevelThreshold() ? (nGlobal >> level) : lvCoarse.getN());
                    int svo = (mpiRank() == 0 && lvCoarse.getNum() >= getMpiLevelThreshold() ? (oGlobal >> level) : lvCoarse.getO());

                    bool updateGhostsLocally = !useMpi() || lvCoarse.getNum() >= getMpiLevelThreshold();

                    if (!lvFine.getUseOpencl())
                    {
                        // Call Galerkin on each rank, if not above threshold, or else only on root.
                        if (!useMpi() || lvCoarse.getNum() <= getMpiLevelThreshold() || mpiRank() == 0)
                            lvCoarse.stencilValues = MultigridEngine::galerkinHandcrafted(
                                *lvFine.getStencilValues(), gh_sv,
                                svm, svn, svo);

                        // Gather stencil values onto root if threshold is reached
                        if (useMpi() && lvCoarse.getNum() == getMpiLevelThreshold())
                            mpi_util::gather(getMpiComm(), *lvCoarse.getStencilValues());

                        // update ghosts of stencil values depending on threshold is reached or not
                        if (!useMpi() || mpiRank() == 0 || (mpiRank() > 0 && lvCoarse.getNum() < getMpiLevelThreshold()))
                        {
                            if (updateGhostsLocally)
                                lvCoarse.getStencilValues()->updateGhosts();
                            else
                                updateGhostsStencilMpi(*lvCoarse.getStencilValues(), lvCoarse.getMpiDataPtr(), isPeriodic(), false);
                        }
                    }
                    else
                    {
                        std::shared_ptr<VaryingStencilGpu> sv_coarse;
                        // Call Galerkin on each rank, if not above threshold, or else only on root.
                        if (!useMpi() || lvCoarse.getNum() <= getMpiLevelThreshold() || mpiRank() == 0)
                            sv_coarse = MultigridEngine::galerkinHandcrafted(
                                *lvFine.getStencilValuesGpu(), gh_sv,
                                svm, svn, svo,
                                getProgram(), getCommands(), getContext(),
                                &getKernelConfig(), getProfilingData());

                        if (sv_coarse)
                        {
                            if (lvCoarse.getUseOpencl())
                            {
                                // Is true, if maxLevelUsingOpencl is not reached yet
                                lvCoarse.stencilValuesGpu = sv_coarse;

                                // Gather stencil values onto root if threshold is reached
                                if (useMpi() && lvCoarse.getNum() == getMpiLevelThreshold())
                                    mpi_util::gather(getMpiComm(), getCommands(), *lvCoarse.getStencilValuesGpu());

                                // update ghosts of stencil values depending on threshold is reached or not
                                if (!useMpi() || mpiRank() == 0 || (mpiRank() > 0 && lvCoarse.getNum() < getMpiLevelThreshold()))
                                {
                                    if (updateGhostsLocally)
                                        lvCoarse.getStencilValuesGpu()->updateGhosts(getProgram(), getCommands(), &getKernelConfig(), getProfilingData());
                                    else
                                        updateGhostsStencilOclMpi(getCommands(), getProgram(), *lvCoarse.getStencilValuesGpu(),
                                                                  getDPlanesBuf(), getHPlanesBufSend(), getHPlanesBufRecv(),
                                                                  lvCoarse.getMpiDataPtr(), false,
                                                                  &getKernelConfig(), getProfilingData());
                                }
                            }
                            else
                            {
                                // TODO sv_coarse might be null here

                                // If fine level uses opencl, but coarse shall not, write stencil values for coarse level from device to host
                                lvCoarse.stencilValues = std::make_shared<mgcl::VaryingStencil>(sv_coarse->read(getCommands(), true));

                                // Gather stencil values onto root if threshold is reached
                                if (useMpi() && lvCoarse.getNum() == getMpiLevelThreshold())
                                    mpi_util::gather(getMpiComm(), *lvCoarse.getStencilValues());

                                // update ghosts of stencil values depending on threshold is reached or not
                                if (!useMpi() || mpiRank() == 0 || (mpiRank() > 0 && lvCoarse.getNum() < getMpiLevelThreshold()))
                                {
                                    if (updateGhostsLocally)
                                        lvCoarse.getStencilValues()->updateGhosts();
                                    else
                                        updateGhostsStencilMpi(*lvCoarse.getStencilValues(), lvCoarse.getMpiDataPtr(), isPeriodic(), false);
                                }
                            }
                        }
                    }
                }

                else if (level >= 1 && getStencilType() == MGCL_BLOCKSTENCIL)
                {
                    int gh_sv = blockstencil ? blockstencil->getGhostsM() : 0;
                    auto& lvFine = *levels[level - 1];
                    auto& lvCoarse = *levels[level];

                    // stencilValues of this level must be of global size on rank 0, if this level is at
                    // the threshold, since it is getting gathered into.
                    int svm = (mpiRank() == 0 && lvCoarse.getNum() >= getMpiLevelThreshold() ? (mGlobal >> level) : lvCoarse.getM());
                    int svn = (mpiRank() == 0 && lvCoarse.getNum() >= getMpiLevelThreshold() ? (nGlobal >> level) : lvCoarse.getN());
                    int svo = (mpiRank() == 0 && lvCoarse.getNum() >= getMpiLevelThreshold() ? (oGlobal >> level) : lvCoarse.getO());

                    bool updateGhostsLocally = !useMpi() || lvCoarse.getNum() >= getMpiLevelThreshold();

                    if (!use_opencl)
                    {
                        // Call Galerkin on each rank, if not above threshold, or else only on root.
                        if (!useMpi() || lvCoarse.getNum() <= getMpiLevelThreshold() || mpiRank() == 0)
                        {
                            assert(lvFine.getBlockstencil() && "Blockstencil on fine level not set");
                            assert(getRestrictionBlockstencil() && "Restriction blockstencil not set");
                            assert(getProlongationBlockstencil() && "Prolongation blockstencil not set");

                            lvCoarse.blockstencil = MultigridEngine::galerkinOptimized(
                                *lvFine.getBlockstencil(),
                                *getRestrictionBlockstencil(),
                                *getProlongationBlockstencil(),
                                gh_sv,
                                svm, svn, svo);
                        }
                        // Gather stencil values onto root if threshold is reached
                        if (useMpi() && lvCoarse.getNum() == getMpiLevelThreshold())
                            mpi_util::gather(getMpiComm(), *lvCoarse.getBlockstencil());

                        // update ghosts of stencil values depending on threshold is reached or not
                        if (!useMpi() || mpiRank() == 0 || (mpiRank() > 0 && lvCoarse.getNum() < getMpiLevelThreshold()))
                        {
                            if (updateGhostsLocally)
                                lvCoarse.getBlockstencil()->updateGhostsLocally();
                            else
                                lvCoarse.getBlockstencil()->updateGhosts(lvCoarse.getMpiDataPtr(), false);

                            // lvCoarse.getBlockstencil()->dumpToFile("bs_level_" + std::to_string(lvCoarse.getNum()) + ".txt");
                            lvCoarse.createInverseOfBlockstencilSeq();
                        }
                    }
                    else
                    {
                        // Call Galerkin on each rank, if not above threshold, or else only on root.
                        if (!useMpi() || lvCoarse.getNum() <= getMpiLevelThreshold() || mpiRank() == 0)
                        {
                            assert(lvFine.getBlockstencilGpu() && "BlockstencilGpu on fine level not set");
                            assert(getRestrictionBlockstencilGpu() && "Restriction blockstencil GPU not set");
                            assert(getProlongationBlockstencilGpu() && "Prolongation blockstencil GPU not set");

                            lvCoarse.blockstencilGpu = MultigridEngine::galerkinOptimized(
                                *lvFine.getBlockstencilGpu(),
                                *getRestrictionBlockstencilGpu(),
                                *getProlongationBlockstencilGpu(),
                                gh_sv,
                                svm, svn, svo,
                                getProgram(), getCommands(), getContext(),
                                &getKernelConfig(), getProfilingData());
                        }

                        // Gather stencil values onto root if threshold is reached
                        if (useMpi() && lvCoarse.getNum() == getMpiLevelThreshold())
                            mpi_util::gather(getMpiComm(), getCommands(), *lvCoarse.getBlockstencilGpu());

                        // update ghosts of stencil values depending on threshold is reached or not
                        if (!useMpi() || mpiRank() == 0 || (mpiRank() > 0 && lvCoarse.getNum() < getMpiLevelThreshold()))
                        {
                            if (updateGhostsLocally)
                                lvCoarse.getBlockstencilGpu()->updateGhostsLocally(getProgram(), getCommands(), &getKernelConfig(), getProfilingData());
                            else
                                lvCoarse.getBlockstencilGpu()->updateGhostsOclMpi(
                                    getProgram(), getCommands(),
                                    getDPlanesBuf(), getHPlanesBufSend(), getHPlanesBufRecv(),
                                    lvCoarse.getMpiData(), false,
                                    &getKernelConfig(), getProfilingData());

                            lvCoarse.createInverseOfBlockstencilGpu();
                        }
                    }
                }

                // Apply Galerkin operator if stencil is fixed and we're not on level 0.
                // No need for gathering required as for VaryingStencil, since coefficients do not differ per grid point.
                // No need to differentiate between GPU and CPU, as FixedStencil is stored on Host only.
                else if (level >= 1 && getStencilType() == MGCL_FIXED)
                {
                    auto& lvFine = *levels[level - 1];
                    auto& lvCoarse = *levels[level];

                    // Call Galerkin on each rank, if not above threshold, or else only on root.
                    if (!useMpi() || lvCoarse.getNum() <= getMpiLevelThreshold() || mpiRank() == 0)
                        lvCoarse.fixedStencil = MultigridEngine::galerkinOptimized(*lvFine.getFixedStencil());
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

        openCLHelper.init(useMpi() ? mpiRank() : 0);
    }

    /**
     * @brief Lazyly initializes OpenCL environment.
     *
     * @return int OpenCL error code
     */
    void Problem::initOpenCL()
    {
        if (!openCLHelper.isInitialized())
        {
            if (getVBSPtr())
            {
                openCLHelper.setPreprocessorConstant("BLOCKSIZE", std::to_string(getVBS().getBlocksize()));
            }
            openCLHelper.init(useMpi() ? mpiRank() : 0);
        }
    }

    /* Waits for all running OpenCL kernels to finish and reads back results from device. Creates arrays on host if none
     * were specified */
    int Problem::readResults()
    {
        if (stencilType == MGCL_BLOCKSTENCIL)
        {
            readResultsBlockstencil();
            return CL_SUCCESS;
        }

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

    void Problem::readResultsBlockstencil()
    {
        int err = clFinish(openCLHelper.getCommands());
        mgclCheckError(err, "Waiting for kernels to finish");

        // if (reuse_opencl_buffers || copy_buffer_data)
        // {
        //     levels[0]->setV(std::make_shared<Cuboid>(levels[0]->m, levels[0]->n, levels[0]->o, ghosts, ghosts, ghosts));
        //     if (v == NULL)
        //         v = std::make_shared<Cuboid>(m, n, o, ghosts_in, ghosts_in, ghosts_in);
        // }

        // read back results TODO: only for testing purposes, maybe define TESTING?
        err = clEnqueueReadBuffer(openCLHelper.getCommands(), levels[0]->getDVBSIn().getBuffer(), CL_TRUE, 0,
                                  sizeof(double) * levels[0]->getDVBSIn().getSize(), levels[0]->getVBS().field1d().data(), 0, NULL, NULL);
        mgclCheckError(err, "Error: Failed to read output arrays from device!");

        // copy result to initial v vector
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < o; k++)
                    for (size_t b = 0; b < blocksize; b++)
                    {
                        (*v_bs)[i + ghosts_in][j + ghosts_in][k + ghosts_in][b] =
                            levels[0]->getVBS()[i + ghosts][j + ghosts][k + ghosts][b];
                    }
    }

    /**
     * @brief Waits for all running OpenCL kernels to finish.
     */
    void Problem::finish()
    {
        if (openCLHelper.isInitialized())
            mgclCheckError(clFinish(getCommands()), "clFinish");
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
            error(std::runtime_error("Failed to initialize mgcl data structures."));

        // Edge case: Do nothing if mpi is used but level threshold is 0 (i.e. all work is done on proc 0).
        if (!(useMpi() && getMpiLevelThreshold() <= 0 && mpiRank() > 0))
        {
            residuals.clear();

            double initres;
            auto& lv0 = *levels[0];

            // calculate initial residual
            if (getVPtr())
            {
                initres = MultigridEngine::residual(*this, lv0, !ignoreTol);
            }
            else
            {
                args::ResidualBSOclArgs residual_args{
                    lv0.getDFBS(), lv0.getDVBSIn(), lv0.getDRBS(),
                    residual_norm,
                    *lv0.blockstencilGpu,
                    lv0.getDRsqBSPtr().get(),
                    true, isPeriodic(),
                    lv0.isCalculatedLocally(),
                    getDPlanesBufPtr(), getHPlanesBufSendPtr(), getHPlanesBufRecvPtr(),
                    getProgram(), getCommands(), getContext(),
                    0, 0, 0,
                    lv0.getMpiDataPtr(),
                    &getKernelConfig(), getProfilingData()};
                initres = MultigridEngine::residual(residual_args);
            }

            if (!silent && !ignoreTol)
                printf("Starting mgcl with initres = %e\n", initres);

            // run vcycle maxiter_vcycles times
            double res, relres;
            elapsedIterations = 0;
            for (int i = 0; i < maxiter_vcycles; i++)
            {
                elapsedIterations++;
                auto tstart = std::chrono::steady_clock::now();
                if (getVPtr())
                {
                    res = MultigridEngine::vcycle(*this, lv0);
                }
                else
                {
                    res = MultigridEngine::vcycleOclBlockstencil(*this, lv0);
                }
                auto tend = mgcl_since(tstart).count();

                if (!ignoreTol)
                {
                    relres = initres == 0 ? 0 : res / initres;

                    // If mpi is in use, calculate global relres first
                    if (useMpi() && getMpiLevelThreshold() > 0)
                    {
                        if (residual_norm == MGCL_RESIDUAL_NORM::MGCL_L2)
                        {
                            relres = relres * relres;
                            MPI_Allreduce(&relres, &relres, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                            relres = sqrt(relres);
                        }
                        else
                        {
                            MPI_Allreduce(&relres, &relres, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                        }
                    }

                    residuals.push_back(relres);
                }

                if (!silent)
                {
                    if (ignoreTol)
                        printf("iter = %d, elapsed time = %ld ms\n", i, tend);
                    else
                    {
                        printf("iter = %d, elapsed time = %ld ms, rel. res = %e\n", i, tend, relres);
                    }
                }

                if (!ignoreTol && relres <= tol)
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
            if (getVPtr())
            {
                mpi_util::scatter_inplace_wgh(mpiGlobalData->getComm(), getCommands(), getLevelAt(0).getDVIn());
            }
            else
            {
                mpi_util::scatter_inplace_wgh(mpiGlobalData->getComm(), getCommands(), getLevelAt(0).getDVBSIn());
            }
        }

        // copy resulting v to d_v on device
        if (copy_buffer_data && getVPtr())
            openCLHelper.copyOutputBuffers();

        // write result into v on host
        if (read_results)
        {
            if (getVPtr())
            {
                readResults();
            }
            else
            {
                readResultsBlockstencil();
            }
        }
    }

    /**
     * @brief Solves problem sequentially using multigrid method.
     *
     * @throws runtime_error When initializing data structures failed.
     */
    void Problem::solveSeq()
    {
        solveSeq(false);
    }

    /**
     * @brief Solves problem sequentially using multigrid method.
     *
     * @param skipInit If true, skips initialization. Requires Problem::init to be called first. Mainly used for
     * testing and benchmarking.
     * @throws runtime_error When initializing data structures failed.
     */
    void Problem::solveSeq(bool skipInit)
    {
        // set up data for each level
        if (!skipInit && !init())
            error(std::runtime_error("Failed to initialize mgcl data structures."));

        // Edge case: Do nothing if mpi is used but level threshold is 0 (i.e. all work is done on proc 0).
        if (!(useMpi() && getMpiLevelThreshold() <= 0 && mpiRank() > 0))
        {
            auto& lv0 = *levels[0];
            double initres;

            if (getVPtr())
            {
                // calculate initial residual
                if (isPeriodic())
                    MultigridEngine::updateGhostsSeq(lv0.getV(), lv0.getMpiDataPtr(), isPeriodic(),
                                                     levels[0]->isCalculatedLocally());

                initres = MultigridEngine::residualSeq(lv0.getF(), lv0.getV(), lv0.getR(),
                                                       residual_norm, stencilType, lv0.stencilFactor,
                                                       lv0.stencilValues.get(),
                                                       lv0.fixedStencil.get(),
                                                       !ignoreTol, isPeriodic(),
                                                       lv0.isCalculatedLocally(),
                                                       0, 0, 0, lv0.getMpiDataPtr());
            }
            else
            {
                // calculate initial residual
                if (isPeriodic())
                {
                    lv0.getVBS().updateGhosts(lv0.getMpiDataPtr(), lv0.isCalculatedLocally());
                }

                args::ResidualBSSeqArgs residual_args{
                    lv0.getFBS(), lv0.getVBS(), lv0.getRBS(),
                    residual_norm,
                    *lv0.getBlockstencil(),
                    true, isPeriodic(),
                    lv0.isCalculatedLocally(),
                    0, 0, 0,
                    lv0.getMpiDataPtr()};
                initres = MultigridEngine::residualSeq(residual_args);
            }

            if (!silent && !ignoreTol)
                printf("Starting mgcl with initres = %e\n", initres);

            // run vcycle maxiter_vcycles times
            double res, relres;
            elapsedIterations = 0;
            residuals.clear();
            for (int i = 0; i < maxiter_vcycles; i++)
            {
                elapsedIterations++;
                auto tstart = std::chrono::steady_clock::now();
                if (getVPtr())
                    res = MultigridEngine::vcycleSeq(*this, lv0);
                else
                    res = MultigridEngine::vcycleSeqBlockstencil(*this, lv0);
                auto tend = mgcl_since(tstart).count();

                if (!ignoreTol)
                {
                    relres = initres == 0 ? 0 : res / initres;

                    // If mpi is in use, calculate global relres first
                    if (useMpi() && getMpiLevelThreshold() > 0)
                    {
                        if (residual_norm == MGCL_RESIDUAL_NORM::MGCL_L2)
                        {
                            relres = relres * relres;
                            MPI_Allreduce(&relres, &relres, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                            relres = sqrt(relres);
                        }
                        else
                        {
                            MPI_Allreduce(&relres, &relres, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                        }
                    }

                    residuals.push_back(res);
                }

                if (!silent)
                {
                    if (ignoreTol)
                        printf("iter = %d, elapsed time = %ld ms\n", i, tend);
                    else
                    {
                        printf("iter = %d, elapsed time = %ld ms, rel. res = %e\n", i, tend, relres);
                    }
                }

                if (!ignoreTol && relres <= tol)
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
            if (getVPtr())
                mpi_util::scatter_inplace_wgh(mpiGlobalData->getComm(), getLevelAt(0).getV());
            else
                mpi_util::scatter_inplace_wgh(mpiGlobalData->getComm(), getLevelAt(0).getVBS());
        }

        // write data to output
        if (getVPtr())
        {
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    for (int k = 0; k < o; k++)
                    {
                        (*v)[i + ghosts_in][j + ghosts_in][k + ghosts_in] =
                            levels[0]->getV()[i + ghosts][j + ghosts][k + ghosts];
                    }
        }
        else
        {
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    for (int k = 0; k < o; k++)
                        for (size_t b = 0; b < blocksize; b++)
                        {
                            (*v_bs)[i + ghosts_in][j + ghosts_in][k + ghosts_in][b] =
                                levels[0]->getVBS()[i + ghosts][j + ghosts][k + ghosts][b];
                        }
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

    void Problem::printDeviceInfo()
    {
        if (openCLHelper.isInitialized())
        {
            openCLHelper.outputDeviceInfo();
        }
        else
        {
            std::cout << "No Device information available. Initialize OpenCL Helper first by calling Problem::init()" << std::endl;
        }
    }

    /********************************
     * Getters and Setters
     ********************************/

    Cuboid& Problem::getF() const
    {
        if (!f)
            error("f is null.");
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

    BufferGpu& Problem::getDPlanesBuf() const
    {
        if (!dPlanesBuf)
            error("dPlanesBuf is null.");
        return *dPlanesBuf;
    }

    BufferGpu* Problem::getDPlanesBufPtr() const
    {
        return dPlanesBuf.get();
    }

    void Problem::setDPlanesBuf(const std::shared_ptr<BufferGpu> dPlanesBuf_)
    {
        dPlanesBuf = dPlanesBuf_;
    }

    std::vector<double>& Problem::getHPlanesBufSend() const
    {
        if (!hPlanesBufSend)
            error("hPlanesBuf is null.");
        return *hPlanesBufSend;
    }

    std::vector<double>* Problem::getHPlanesBufSendPtr() const
    {
        return hPlanesBufSend.get();
    }

    void Problem::setHPlanesBufSend(std::shared_ptr<std::vector<double>> hPlanesBuf_)
    {
        hPlanesBufSend = hPlanesBuf_;
    }

    std::vector<double>& Problem::getHPlanesBufRecv() const
    {
        if (!hPlanesBufRecv)
            error("hPlanesBuf is null.");
        return *hPlanesBufRecv;
    }

    std::vector<double>* Problem::getHPlanesBufRecvPtr() const
    {
        return hPlanesBufRecv.get();
    }

    void Problem::setHPlanesBufRecv(const std::shared_ptr<std::vector<double>> hPlanesBuf_)
    {
        hPlanesBufRecv = hPlanesBuf_;
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
        if (stencilType == MGCL_BLOCKSTENCIL)
            error("resuing opencl buffers not yet supported for blockstencil");
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

    int Problem::getJacobiIterationsPerKernel() const
    {
        return jacobi_iterations_per_kernel;
    }

    /**
     * @brief Sets the amount of jacobi iterations per kernel call, i.e. without ghost update in-between. > 1 not recommended
     * for multi-gpu case.
     *
     * @param jacobiIterationsPerKernel
     */
    void Problem::setJacobiIterationsPerKernel(int jacobiIterationsPerKernel)
    {
        // TODO change this like for blockstencil
        if (stencilValues && (stencilValues->getGhostsM() < jacobiIterationsPerKernel ||
                              stencilValues->getGhostsN() < jacobiIterationsPerKernel ||
                              stencilValues->getGhostsO() < jacobiIterationsPerKernel))
            error("Ghosts of stencilValues must be >= ghosts. Make sure to call setGhosts and setJacobiIterationsPerKernel before setStencilType!");

        jacobi_iterations_per_kernel = jacobiIterationsPerKernel;
        ghosts = std::max(ghosts, jacobi_iterations_per_kernel);

        if (stencilType == MGCL_BLOCKSTENCIL)
        {
            // recreate blockstencil with appropriate ghost amount
            setStencilType(MGCL_BLOCKSTENCIL);
        }
    }

    CuboidGpu& Problem::getDV() const
    {
        if (!dV)
            error("dV is null!");
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
            error("dF is null!");
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
            error("Ghosts of stencilValues must be >= ghosts. Make sure to call setGhosts and setJacobiIterationsPerKernel before setStencilType!");

        // if (blockstencil && (blockstencil->getGhostsM() < ghosts_ ||
        //                      blockstencil->getGhostsN() < ghosts_ ||
        //                      blockstencil->getGhostsO() < ghosts_))
        //     error("Ghosts of blockstencil must be >= ghosts. Make sure to call setGhosts and setJacobiIterationsPerKernel before setStencilType!");

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
        if (stencilType == MGCL_BLOCKSTENCIL)
            error("copy opencl buffers not yet supported for blockstencil");
        copy_buffer_data = copyBufferData;
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
        assert(!openCLHelper.isInitialized() && "OpenCLHelper is already initialized!");
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

    cl_command_queue Problem::getCommands2() const
    {
        return openCLHelper.getCommands2();
    }

    std::string Problem::getKernelFile() const
    {
        return openCLHelper.getKernelFile();
    }

    void Problem::setKernelFile(const std::string& kernelFile_)
    {
        assert(!openCLHelper.isInitialized() && "OpenCLHelper is already initialized!");
        openCLHelper.setKernelFile(kernelFile_);
        if (!kernelFile_.empty())
        {
            openCLHelper.setReadKernelFromFile(true);
        }
        else
        {
            openCLHelper.setReadKernelFromFile(false);
        }
    }

    cl_device_type Problem::getDeviceType() const
    {
        return openCLHelper.deviceType;
    }

    void Problem::setDeviceType(const cl_device_type& deviceType_)
    {
        assert(!openCLHelper.isInitialized() && "OpenCLHelper is already initialized!");
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

    void Problem::setBinaryFile(const std::string& binaryFile_)
    {
        openCLHelper.setBinaryFile(binaryFile_);
    }

    std::string Problem::getBinaryFile() const
    {
        return openCLHelper.getBinaryFile();
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
            fixedStencil = nullptr;
            blockstencil = nullptr;
            prolongationBlockstencil = nullptr;
            restrictionBlockstencil = nullptr;
            calculateAndSetMpiLevelThreshold();
            int gh = std::max(1, jacobi_iterations_per_kernel);
            if (useMpi() && getMpiLevelThreshold() == 0 && mpiRank() == 0) // TODO check
                stencilValues = std::make_shared<VaryingStencil>(mGlobal, nGlobal, oGlobal, 3, gh, gh, gh);
            else
                stencilValues = std::make_shared<VaryingStencil>(m, n, o, 3, gh, gh, gh);
        }
        else if (stencilType == MGCL_FIXED)
        {
            // No need for ghosts for fixed stencil as the coefficients would be the same for the ghost points
            stencilValues = nullptr;
            blockstencil = nullptr;
            prolongationBlockstencil = nullptr;
            restrictionBlockstencil = nullptr;
            fixedStencil = std::make_shared<FixedStencil>(3);
        }
        else if (stencilType == MGCL_BLOCKSTENCIL)
        {
            fixedStencil = nullptr;
            stencilValues = nullptr;
            calculateAndSetMpiLevelThreshold();
            int gh = std::max(1, jacobi_iterations_per_kernel);
            if (useMpi() && getMpiLevelThreshold() == 0 && mpiRank() == 0) // TODO check
            {
                blockstencil = std::make_shared<Blockstencil>(mGlobal, nGlobal, oGlobal, 3, blocksize, gh, gh, gh);
            }
            else
            {
                blockstencil = std::make_shared<Blockstencil>(m, n, o, 3, blocksize, gh, gh, gh);
            }
            restrictionBlockstencil = std::make_shared<FixedBlockstencil>(3, blocksize);
            prolongationBlockstencil = std::make_shared<FixedBlockstencil>(3, blocksize);
        }
        else
        {
            stencilValues = nullptr;
            fixedStencil = nullptr;
        }
    }

    std::shared_ptr<VaryingStencil>& Problem::getStencilValues()
    {
        if (stencilType != MGCL_VARYING)
        {
            error("Problem::getStencilValues: stencilType is not MGCL_VARYING. Use Problem::setStencilType(MGCL_VARYING) first.");
        }
        return stencilValues;
    }

    std::shared_ptr<FixedStencil>& Problem::getFixedStencil()
    {
        if (stencilType != MGCL_FIXED)
        {
            error("Problem::getFixedStencil: stencilType is not MGCL_FIXED. Use Problem::setStencilType(MGCL_FIXED) first.");
        }
        return fixedStencil;
    }

    std::shared_ptr<Blockstencil>& Problem::getBlockstencil()
    {
        if (stencilType != MGCL_BLOCKSTENCIL)
        {
            error("Problem::getBlockstencil: stencilType is not MGCL_BLOCKSTENCIL. Use Problem::setStencilType(MGCL_BLOCKSTENCIL) first.");
        }
        return blockstencil;
    }

    std::shared_ptr<FixedBlockstencil>& Problem::getRestrictionBlockstencil()
    {
        if (stencilType != MGCL_BLOCKSTENCIL)
        {
            error("Problem::getRestrictionBlockstencil: stencilType is not MGCL_BLOCKSTENCIL. Use Problem::setStencilType(MGCL_BLOCKSTENCIL) first.");
        }
        restrictionBlockstencilAccessed = true;
        return restrictionBlockstencil;
    }

    std::shared_ptr<FixedBlockstencil>& Problem::getProlongationBlockstencil()
    {
        if (stencilType != MGCL_BLOCKSTENCIL)
        {
            error("Problem::getProlongationBlockstencil: stencilType is not MGCL_BLOCKSTENCIL. Use Problem::setStencilType(MGCL_BLOCKSTENCIL) first.");
        }
        prolongationBlockstencilAccessed = true;
        return prolongationBlockstencil;
    }

    std::shared_ptr<FixedBlockstencilGpu>& Problem::getRestrictionBlockstencilGpu()
    {
        // TODO sanity check?
        return restrictionBlockstencilGpu;
    }

    std::shared_ptr<FixedBlockstencilGpu>& Problem::getProlongationBlockstencilGpu()
    {
        // TODO sanity check?
        return prolongationBlockstencilGpu;
    }

    void Problem::setIgnoreTol(bool ignoreTol_)
    {
        ignoreTol = ignoreTol_;
    }

    Cuboid& Problem::getV() const
    {
        if (!v)
            error("v is null!");
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

    CuboidBS& Problem::getVBS() const
    {
        if (!v_bs)
            error("v_bs is null!");
        return *v_bs;
    }

    std::shared_ptr<CuboidBS> Problem::getVBSPtr() const
    {
        return v_bs;
    }

    void Problem::setVBS(std::shared_ptr<CuboidBS> v_)
    {
        v_bs = v_;
    }

    CuboidBS& Problem::getFBS() const
    {
        if (!f_bs)
            error("f_bs is null!");
        return *f_bs;
    }

    std::shared_ptr<CuboidBS> Problem::getFBSPtr() const
    {
        return f_bs;
    }

    void Problem::setFBS(std::shared_ptr<CuboidBS> f_)
    {
        f_bs = f_;
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

    std::vector<double>& Problem::getResiduals()
    {
        return residuals;
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
            error("MpiLevelThreshold cannot be negative");
        if (mpiLevelThreshold_ > maxlevel)
            error("MpiLevelThreshold cannot be larger than maxlevel (" + std::to_string(maxlevel) + ")");
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

    OCL_DEVICE_STRATEGY Problem::getDeviceStrategy() const
    {
        return openCLHelper.getDeviceStrategy();
    }

    void Problem::setDeviceStrategy(const OCL_DEVICE_STRATEGY deviceStrategy)
    {
        assert(!openCLHelper.isInitialized() && "OpenCLHelper is already initialized!");
        openCLHelper.setDeviceStrategy(deviceStrategy);
    }
}
