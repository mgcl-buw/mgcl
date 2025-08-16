#include "level.hpp"
#include "blockstencil.hpp"
#include "cuboid.hpp" // for Cuboid
#include "cuboid_bs.hpp"
#include "mgcl.hpp"
#include "mpi_level_data.hpp"
#include "mpi_stencil.hpp"
#include "mpi_util.hpp"
#include "multigrid_engine.hpp" // for Problem, MultigridEngine
#include "problem.hpp"
#include "stencil.hpp"

#include <cassert>
#include <cstddef> // for NULL
#include <iostream>
#include <memory>
#include <stdexcept> // for invalid_argument
#include <string>    // for to_string, allocator, basic_string

#ifdef __APPLE__
#include <OpenCL/cl_platform.h>
#else
#include <CL/cl_platform.h> // for cl_double
#endif

namespace mgcl
{
    /**
     * @brief Construct a new Level object. m, n and o must be dims of real grid.
     *
     * @param problem_ Problem this Level belongs to.
     * @param num_ Number of level in the Problem (finest grid is level 0)
     * @param stencilType_ Type of stencil that shall be used on this level
     * @throws invalid_argument When num_ is invalid, i.e. < 0 or > Problem.maxlevel
     */
    Level::Level(Problem* problem_, int num_)
        : problem(problem_),
          num(num_),
          m(((problem_->getMpiLevelThreshold() <= num && problem_->mpiRank() == 0) ? problem_->getMGlobal() : problem_->getM()) >> num_),
          n(((problem_->getMpiLevelThreshold() <= num && problem_->mpiRank() == 0) ? problem_->getNGlobal() : problem_->getN()) >> num_),
          o(((problem_->getMpiLevelThreshold() <= num && problem_->mpiRank() == 0) ? problem_->getOGlobal() : problem_->getO()) >> num_),
          mgh(m + 2 * problem->getGhosts()),
          ngh(n + 2 * problem->getGhosts()),
          ogh(o + 2 * problem->getGhosts()),
          h(1.0 / static_cast<double>(problem->getMGlobal() >> num_)), // TODO differentiate for non-cube-like domains
          stencilType(problem_->stencilType)
    {
        if (stencilType == MGCL_LAPLACE_7POINT)
            stencilFactor = 1.0 / (h * h);
        else if (stencilType == MGCL_LAPLACE_19POINT)
            stencilFactor = 1.0 / (6.0 * h * h);
        else if (stencilType == MGCL_LAPLACE_27POINT)
            stencilFactor = 1.0 / (26.0 * h * h);

        if (num_ < 0 || num_ > problem->getMaxlevel())
            throw std::invalid_argument(std::string("num is invalid! num: ")
                                            .append(std::to_string(num_))
                                            .append(", problem.maxlevel: ")
                                            .append(std::to_string(problem->getMaxlevel())));

        // if (useMpi)
        if (problem->useMpi())
            mpiData = std::make_shared<MPILevelData>(problem->getMpiComm());
    }

    /**
     * @brief Initializes data for this level.
     *
     * @return true All good.
     * @return false Something went wrong.
     */
    bool Level::init()
    {
        if (m <= 0 || n <= 0 || o <= 0)
            return true;

        if (stencilType == MGCL_BLOCKSTENCIL)
            return initBlockstencil();

        // First init MPI data to get information about neighbours and be able to update ghosts.
        initMpiData();

        // Always allocate data on level 0 so input data is copied and not left uninitialized.
        if (num == 0)
        {
            // copy stencilsValues pointer from Problem to first Level and update ghosts
            if (stencilType == MGCL_VARYING)
            {
                stencilValues = problem->stencilValues;

                // TODO refactor updateGhosts local?
                updateGhostsStencilMpi(*stencilValues, mpiData.get(), problem->isPeriodic(),
                                       isCalculatedLocally()); // TODO test
            }

            // copy fixedStencil pointer from Problem to first Level
            if (stencilType == MGCL_FIXED)
            {
                fixedStencil = problem->getFixedStencil();
            }

            // create ghosted arrays for v and f on host if device buffer should not be reused
            if (!problem->reuse_opencl_buffers && !problem->copy_buffer_data)
            {
                v = std::make_shared<Cuboid>(m, n, o, problem->ghosts, problem->ghosts, problem->ghosts);
                f = std::make_shared<Cuboid>(m, n, o, problem->ghosts, problem->ghosts, problem->ghosts);

                // copy initial input data from conf into mgcl data struct
                for (int i = 0; i < problem->getV().getM(); i++)
                    for (int j = 0; j < problem->getV().getN(); j++)
                        for (int k = 0; k < problem->getV().getO(); k++)
                        {
                            getV()[i + problem->ghosts][j + problem->ghosts][k + problem->ghosts] =
                                problem->getV()[i + problem->ghosts_in][j + problem->ghosts_in][k + problem->ghosts_in];
                            getF()[i + problem->ghosts][j + problem->ghosts][k + problem->ghosts] =
                                problem->getF()[i + problem->ghosts_in][j + problem->ghosts_in][k + problem->ghosts_in];
                        }

                // If mgcl is run with multiple processes, but already level 0 shall be calculated on only one process
                // (i.e. mpiMinGridPoints > min(m_local,n_local,o_local)), gather data on rank 0.
                if (problem->useMpi() && problem->mpiSize() > 1 && problem->getMpiLevelThreshold() <= 0)
                {
                    mpi_util::gather(problem->getMpiComm(), getV());
                    mpi_util::gather(problem->getMpiComm(), getF());
                    // TODO what to do when reuse_opencl_buffers?
                }

                if (problem->isPeriodic())
                    MultigridEngine::updateGhostsSeq(getF(), mpiData.get(), problem->isPeriodic(), isCalculatedLocally());
            }

            // r on host is only needed if opencl should not be used
            if (!problem->use_opencl)
            {
                r = std::make_shared<Cuboid>(m, n, o, problem->ghosts, problem->ghosts, problem->ghosts);
            }

            // TODO check this with mpi
            if (initOpenCLBuffers() != CL_SUCCESS)
                return false;
        }
        // TODO revisit, maybe not worth the effort since coarse levels are small anyways
        // else if (!problem->useMpi() || !isCalculatedLocally() || mpiData->rank == 0)
        else
        {
            // Only allocate data on coarser levels, if
            // 1. mgcl is run without MPI at all, or
            // 2. mgcl is run with MPI and this level is below the level threshold (i.e. enough grid points), or
            // 3. mgcl is run with MPI, this level is equal to or above the level threshold but the rank is 0.
            if (!problem->use_opencl)
            {
                v = std::make_shared<Cuboid>(m, n, o, problem->ghosts, problem->ghosts, problem->ghosts);
                f = std::make_shared<Cuboid>(m, n, o, problem->ghosts, problem->ghosts, problem->ghosts);
                r = std::make_shared<Cuboid>(m, n, o, problem->ghosts, problem->ghosts, problem->ghosts);
            }

            if (initOpenCLBuffers() != CL_SUCCESS)
                return false;
        }

        return true;
    }

    /**
     * @brief Initializes data for this level if stenciltype is blockstencil.
     *
     * @return true All good.
     * @return false Something went wrong.
     */
    bool Level::initBlockstencil()
    {
        assert(stencilType == MGCL_BLOCKSTENCIL && "Level::initBlockstencil(): stencilType is not MGCL_BLOCKSTENCIL.");

        if (m <= 0 || n <= 0 || o <= 0)
            return true;

        // First init MPI data to get information about neighbours and be able to update ghosts.
        initMpiData();

        // Always allocate data on level 0 so input data is copied and not left uninitialized.
        if (num == 0)
        {
            // copy stencilsValues pointer from Problem to first Level and update ghosts
            blockstencil = problem->blockstencil;

            // TODO refactor updateGhosts local?
            blockstencil->updateGhosts(mpiData.get(), isCalculatedLocally());

            // Create inverse for level 0. Coarser levels' inverse will be created after galerkin
            createInverseOfBlockstencilSeq();

            // create ghosted arrays for v and f on host if device buffer should not be reused
            if (!problem->reuse_opencl_buffers && !problem->copy_buffer_data)
            {
                v_bs = std::make_shared<CuboidBS>(m, n, o, problem->ghosts, problem->ghosts, problem->ghosts, problem->getBlocksize());
                f_bs = std::make_shared<CuboidBS>(m, n, o, problem->ghosts, problem->ghosts, problem->ghosts, problem->getBlocksize());

                // copy initial input data from conf into mgcl data struct
                for (int i = 0; i < problem->getVBS().getM(); i++)
                    for (int j = 0; j < problem->getVBS().getN(); j++)
                        for (int k = 0; k < problem->getVBS().getO(); k++)
                            for (size_t b = 0; b < problem->getBlocksize(); b++)
                            {
                                getVBS()[i + problem->ghosts][j + problem->ghosts][k + problem->ghosts][b] =
                                    problem->getVBS()[i + problem->ghosts_in][j + problem->ghosts_in][k + problem->ghosts_in][b];
                                getFBS()[i + problem->ghosts][j + problem->ghosts][k + problem->ghosts][b] =
                                    problem->getFBS()[i + problem->ghosts_in][j + problem->ghosts_in][k + problem->ghosts_in][b];
                            }

                // If mgcl is run with multiple processes, but already level 0 shall be calculated on only one process
                // (i.e. mpiMinGridPoints > min(m_local,n_local,o_local)), gather data on rank 0.
                if (problem->useMpi() && problem->mpiSize() > 1 && problem->getMpiLevelThreshold() <= 0)
                {
                    mpi_util::gather(problem->getMpiComm(), getVBS());
                    mpi_util::gather(problem->getMpiComm(), getFBS());
                    // TODO what to do when reuse_opencl_buffers?
                }

                if (problem->isPeriodic())
                {
                    getFBS().updateGhosts(mpiData.get(), isCalculatedLocally());
                }
            }

            // r on host is only needed if opencl should not be used
            if (!problem->use_opencl)
            {
                r_bs = std::make_shared<CuboidBS>(m, n, o, problem->ghosts, problem->ghosts, problem->ghosts, problem->getBlocksize());
            }

            // TODO check this with mpi
            if (initOpenCLBuffersBlockstencil() != CL_SUCCESS)
                return false;
        }
        // TODO revisit, maybe not worth the effort since coarse levels are small anyways
        // else if (!problem->useMpi() || !isCalculatedLocally() || mpiData->rank == 0)
        else
        {
            // Only allocate data on coarser levels, if
            // 1. mgcl is run without MPI at all, or
            // 2. mgcl is run with MPI and this level is below the level threshold (i.e. enough grid points), or
            // 3. mgcl is run with MPI, this level is equal to or above the level threshold but the rank is 0.
            if (!problem->use_opencl)
            {
                v_bs = std::make_shared<CuboidBS>(m, n, o, problem->ghosts, problem->ghosts, problem->ghosts, problem->getBlocksize());
                f_bs = std::make_shared<CuboidBS>(m, n, o, problem->ghosts, problem->ghosts, problem->ghosts, problem->getBlocksize());
                r_bs = std::make_shared<CuboidBS>(m, n, o, problem->ghosts, problem->ghosts, problem->ghosts, problem->getBlocksize());
            }

            if (initOpenCLBuffersBlockstencil() != CL_SUCCESS)
                return false;
        }

        return true;
    }

    /**
     * @brief Initializes OpenCL buffers for this level based on settings. Returns immediately if use_opencl is false.
     *
     * @return int error code from OpenCL calls.
     */
    int Level::initOpenCLBuffers()
    {
        if (!problem->getUseOpencl())
            return CL_SUCCESS;

        int err;
        auto context = problem->getOpenCLHelper().getContext();
        auto deviceType = problem->getOpenCLHelper().getDeviceType();

        // create d_v_in and d_f buffers on level zero and copy data to it only if buffers should not be reused
        if (num == 0)
        {
            // create gpu buffer for varying stencil if needed
            if (stencilType == MGCL_VARYING)
            {
                // If level threshold is 0, stencilValues must have global sizes.
                if (problem->getMpiLevelThreshold() == 0 && problem->mpiRank() == 0)
                {
                    stencilValuesGpu = std::make_shared<VaryingStencilGpu>(
                        problem->mGlobal, problem->nGlobal, problem->oGlobal, 3,
                        std::max(1, problem->getJacobiIterationsPerKernel()),
                        problem->getContext(), problem->getCommands(), problem->getProgram());
                }
                else
                {
                    stencilValuesGpu = std::make_shared<VaryingStencilGpu>(
                        m, n, o, 3, std::max(1, problem->getJacobiIterationsPerKernel()),
                        problem->getContext(), problem->getCommands(), problem->getProgram());
                }

                // Fill stencil values on gpu on level 0 from input stencil
                stencilValuesGpu->fill(*stencilValues, problem->getCommands(), true);
            }

            if (problem->getReuseOpenclBuffers())
            {
                dVIn = problem->getDVPtr();
                dF = problem->getDFPtr();

                // TODO check with CuboidGpu
                // // retain buffers (i.e. increase internal reference count so they won't be released by accident)
                // err = clRetainMemObject(dVIn);
                // mgclCheckError(err, "clRetainMemObject(dVIn)");
                // err = clRetainMemObject(dF);
                // mgclCheckError(err, "clRetainMemObject(dF)");
            }
            else if (problem->getCopyBufferData())
            {
                dVIn = std::make_shared<CuboidGpu>(context, CL_MEM_READ_WRITE, m, n, o,
                                                   problem->getGhosts(), problem->getGhosts(), problem->getGhosts());
                dF = std::make_shared<CuboidGpu>(context, CL_MEM_READ_WRITE, m, n, o,
                                                 problem->getGhosts(), problem->getGhosts(), problem->getGhosts());
                problem->getOpenCLHelper().copyInputBuffers();
            }
            else
            {
                int pointer_flag = deviceType == CL_DEVICE_TYPE_GPU ? CL_MEM_COPY_HOST_PTR : CL_MEM_USE_HOST_PTR;
                dVIn = std::make_shared<CuboidGpu>(context, pointer_flag | CL_MEM_READ_WRITE, *v);
                dF = std::make_shared<CuboidGpu>(context, pointer_flag | CL_MEM_READ_WRITE, *f);
            }
        }
        else
        {
            dVIn = std::make_shared<CuboidGpu>(context, CL_MEM_READ_WRITE, m, n, o,
                                               problem->getGhosts(), problem->getGhosts(), problem->getGhosts());
            dF = std::make_shared<CuboidGpu>(context, CL_MEM_READ_WRITE, m, n, o,
                                             problem->getGhosts(), problem->getGhosts(), problem->getGhosts());
        }

        dVOut = std::make_shared<CuboidGpu>(context, CL_MEM_READ_WRITE, m, n, o,
                                            problem->getGhosts(), problem->getGhosts(), problem->getGhosts());
        dR = std::make_shared<CuboidGpu>(context, CL_MEM_READ_WRITE, m, n, o,
                                         problem->getGhosts(), problem->getGhosts(), problem->getGhosts());
        dRsq = std::make_shared<CuboidGpu>(context, CL_MEM_WRITE_ONLY, m, n, o, 0, 0, 0);

        dVOut->fill(problem->getProgram(), problem->getCommands(), 0.0, false, &problem->getKernelConfig(), problem->getProfilingData());
        dR->fill(problem->getProgram(), problem->getCommands(), 0.0, false, &problem->getKernelConfig(), problem->getProfilingData());
        dRsq->fill(problem->getProgram(), problem->getCommands(), 0.0, false, &problem->getKernelConfig(), problem->getProfilingData());

        err = MultigridEngine::updateGhosts(*problem, *dF, mpiData.get(), isCalculatedLocally());
        mgclCheckError(err, "Updating ghosts of d_f");

        return CL_SUCCESS;
    }

    /**
     * @brief Initializes OpenCL buffers for this level based on settings. Returns immediately if use_opencl is false.
     *
     * @return int error code from OpenCL calls.
     */
    int Level::initOpenCLBuffersBlockstencil()
    {
        assert(stencilType == MGCL_BLOCKSTENCIL && "Level::initOpenCLBuffersBlockstencil(): stencilType is not MGCL_BLOCKSTENCIL.");

        if (!problem->getUseOpencl())
            return CL_SUCCESS;

        auto context = problem->getOpenCLHelper().getContext();
        auto deviceType = problem->getOpenCLHelper().getDeviceType();

        // create d_v_in and d_f buffers on level zero and copy data to it only if buffers should not be reused
        if (num == 0)
        {
            // create gpu buffer for blockstencil if needed
            // If level threshold is 0, blockstencil must have global sizes.
            if (problem->getMpiLevelThreshold() == 0 && problem->mpiRank() == 0)
            {
                blockstencilGpu = std::make_shared<BlockstencilGpu>(
                    problem->mGlobal, problem->nGlobal, problem->oGlobal, 3, blockstencil->getBlocksize(),
                    std::max(1, problem->getJacobiIterationsPerKernel()),
                    problem->getContext(), problem->getCommands(), problem->getProgram());
            }
            else
            {
                blockstencilGpu = std::make_shared<BlockstencilGpu>(
                    m, n, o, 3, blockstencil->getBlocksize(),
                    std::max(1, problem->getJacobiIterationsPerKernel()),
                    problem->getContext(), problem->getCommands(), problem->getProgram());
            }

            // Fill stencil values on gpu on level 0 from input stencil
            blockstencilGpu->fill(*blockstencil, problem->getCommands(), true);

            // Create inverse for level 0. Coarser levels' inverse will be created after galerkin
            createInverseOfBlockstencilGpu();

            // if (problem->getReuseOpenclBuffers())
            // {
            //     dVIn = problem->getDVPtr();
            //     dF = problem->getDFPtr();

            //     // TODO check with CuboidGpu
            //     // // retain buffers (i.e. increase internal reference count so they won't be released by accident)
            //     // err = clRetainMemObject(dVIn);
            //     // mgclCheckError(err, "clRetainMemObject(dVIn)");
            //     // err = clRetainMemObject(dF);
            //     // mgclCheckError(err, "clRetainMemObject(dF)");
            // }
            // else if (problem->getCopyBufferData())
            // {
            //     dVIn = std::make_shared<CuboidGpu>(context, CL_MEM_READ_WRITE, m, n, o,
            //                                        problem->getGhosts(), problem->getGhosts(), problem->getGhosts());
            //     dF = std::make_shared<CuboidGpu>(context, CL_MEM_READ_WRITE, m, n, o,
            //                                      problem->getGhosts(), problem->getGhosts(), problem->getGhosts());
            //     problem->getOpenCLHelper().copyInputBuffers();
            // }
            // else
            // {
            int pointer_flag = deviceType == CL_DEVICE_TYPE_GPU ? CL_MEM_COPY_HOST_PTR : CL_MEM_USE_HOST_PTR;
            dVIn_bs = std::make_shared<CuboidBSGpu>(context, pointer_flag | CL_MEM_READ_WRITE, *v_bs);
            dF_bs = std::make_shared<CuboidBSGpu>(context, pointer_flag | CL_MEM_READ_WRITE, *f_bs);
            // }
        }
        else
        {
            dVIn_bs = std::make_shared<CuboidBSGpu>(context, CL_MEM_READ_WRITE, m, n, o,
                                                    problem->getGhosts(), problem->getGhosts(), problem->getGhosts(),
                                                    problem->getBlocksize());
            dF_bs = std::make_shared<CuboidBSGpu>(context, CL_MEM_READ_WRITE, m, n, o,
                                                  problem->getGhosts(), problem->getGhosts(), problem->getGhosts(),
                                                  problem->getBlocksize());
        }

        dVOut_bs = std::make_shared<CuboidBSGpu>(context, CL_MEM_READ_WRITE, m, n, o,
                                                 problem->getGhosts(), problem->getGhosts(), problem->getGhosts(),
                                                 problem->getBlocksize());
        dR_bs = std::make_shared<CuboidBSGpu>(context, CL_MEM_READ_WRITE, m, n, o,
                                              problem->getGhosts(), problem->getGhosts(), problem->getGhosts(),
                                              problem->getBlocksize());
        dRsq_bs = std::make_shared<CuboidBSGpu>(context, CL_MEM_WRITE_ONLY, m, n, o, 0, 0, 0, problem->getBlocksize());

        dVOut_bs->fill(problem->getProgram(), problem->getCommands(), 0.0, false, &problem->getKernelConfig(), problem->getProfilingData());
        dR_bs->fill(problem->getProgram(), problem->getCommands(), 0.0, false, &problem->getKernelConfig(), problem->getProfilingData());
        dRsq_bs->fill(problem->getProgram(), problem->getCommands(), 0.0, false, &problem->getKernelConfig(), problem->getProfilingData());

        dF_bs->updateGhostsOclMpi(
            problem->getProgram(), problem->getCommands(),
            problem->getDPlanesBufPtr(), problem->getHPlanesBufSendPtr(), problem->getHPlanesBufRecvPtr(),
            mpiData.get(), isCalculatedLocally(),
            &problem->getKernelConfig(), problem->getProfilingData());

        return CL_SUCCESS;
    }

    /**
     * @brief Initializes MPI data for the current level and sets dimensions.
     * Mainly taken from pmg from Matthias Bolten.
     * If mgcl is used on one MPI process only, no intialization is needed and this functions returns early.
     *
     * @return int
     */
    int Level::initMpiData()
    {
        int ret = 0;

        if (!isCalculatedLocally())
        {
            // MPI variables
            MPI_Comm mpi_comm = problem->getMpiComm();
            bool periodic = problem->isPeriodic();

            if (mpiData->mpiSize() == 1)
            {
                mpiData->left = mpiData;
                mpiData->right = mpiData;
                mpiData->down = mpiData;
                mpiData->up = mpiData;
                mpiData->back = mpiData;
                mpiData->front = mpiData;
                return MPI_SUCCESS;
            }

            mpiData->left = std::make_shared<MPILevelData>(mpiData->comm);
            mpiData->right = std::make_shared<MPILevelData>(mpiData->comm);
            mpiData->down = std::make_shared<MPILevelData>(mpiData->comm);
            mpiData->up = std::make_shared<MPILevelData>(mpiData->comm);
            mpiData->back = std::make_shared<MPILevelData>(mpiData->comm);
            mpiData->front = std::make_shared<MPILevelData>(mpiData->comm);

            if (num == 0)
            {
                /* Calculating neighbours */
                ret = MPI_Cart_shift(mpiData->comm, 2, 1, &mpiData->left->rank, &mpiData->right->rank);
                mpi_util::mgclCheckMpiError(mpiData->comm, ret, "MPI_Cart_shift x-direction");
                ret = MPI_Cart_shift(mpiData->comm, 1, 1, &mpiData->down->rank, &mpiData->up->rank);
                mpi_util::mgclCheckMpiError(mpiData->comm, ret, "MPI_Cart_shift y-direction");
                ret = MPI_Cart_shift(mpiData->comm, 0, 1, &mpiData->front->rank, &mpiData->back->rank);
                mpi_util::mgclCheckMpiError(mpiData->comm, ret, "MPI_Cart_shift z-direction");
            }
            else
            {
                // Send and get ranks of neighbours that still have nodes (i.e. m,n,o > 0).

                /* Temporary buffer (for receiving unused messages) */
                int tmpbuf;

                /* MPI variables */
                int myid;
                MPI_Request reqs[2];
                MPI_Status stats[2];

                Level& levelAbove = problem->getLevelAt(num - 1);

                /* Getting local rank */
                MPI_Comm_rank(mpi_comm, &myid);

                if (periodic)
                {
                    /* Initializing neighbours */
                    mpiData->left->rank = myid;
                    mpiData->right->rank = myid;
                    mpiData->down->rank = myid;
                    mpiData->up->rank = myid;
                    mpiData->back->rank = myid;
                    mpiData->front->rank = myid;
                }
                else
                {
                    /* Initializing neighbours */
                    mpiData->left->rank = MPI_PROC_NULL;
                    mpiData->right->rank = MPI_PROC_NULL;
                    mpiData->down->rank = MPI_PROC_NULL;
                    mpiData->up->rank = MPI_PROC_NULL;
                    mpiData->back->rank = MPI_PROC_NULL;
                    mpiData->front->rank = MPI_PROC_NULL;
                }

                if (m > 0)
                {
                    /* Sending data to left */
                    reqs[0] = MPI_REQUEST_NULL;
                    reqs[1] = MPI_REQUEST_NULL;

                    if ((myid != levelAbove.mpiData->left->rank) && (MPI_PROC_NULL != levelAbove.mpiData->left->rank))
                        MPI_Isend((void*)&myid, 1, MPI_INT, levelAbove.mpiData->left->rank, 0, mpi_comm, &reqs[0]);

                    if ((myid != levelAbove.mpiData->right->rank) && (MPI_PROC_NULL != levelAbove.mpiData->right->rank))
                        MPI_Irecv((void*)&mpiData->right->rank, 1, MPI_INT, levelAbove.mpiData->right->rank, 0, mpi_comm, &reqs[1]);

                    MPI_Waitall(2, reqs, stats);

                    /* Sending data to right */
                    reqs[0] = MPI_REQUEST_NULL;
                    reqs[1] = MPI_REQUEST_NULL;
                    if ((myid != levelAbove.mpiData->right->rank) && (MPI_PROC_NULL != levelAbove.mpiData->right->rank))
                        MPI_Isend((void*)&myid, 1, MPI_INT, levelAbove.mpiData->right->rank, 0, mpi_comm,
                                  &reqs[0]);
                    if ((myid != levelAbove.mpiData->left->rank) && (MPI_PROC_NULL != levelAbove.mpiData->left->rank))
                        MPI_Irecv((void*)&mpiData->left->rank, 1, MPI_INT, levelAbove.mpiData->left->rank, 0,
                                  mpi_comm, &reqs[1]);
                    MPI_Waitall(2, reqs, stats);
                }
                else
                {
                    /* Sending data to left */
                    reqs[0] = MPI_REQUEST_NULL;
                    reqs[1] = MPI_REQUEST_NULL;
                    if ((myid != levelAbove.mpiData->left->rank) && (MPI_PROC_NULL != levelAbove.mpiData->left->rank))
                        MPI_Isend((void*)&levelAbove.mpiData->right->rank, 1, MPI_INT, levelAbove.mpiData->left->rank,
                                  0, mpi_comm, &reqs[0]);
                    if ((myid != levelAbove.mpiData->right->rank) && (MPI_PROC_NULL != levelAbove.mpiData->right->rank))
                        MPI_Irecv((void*)&tmpbuf, 1, MPI_INT, levelAbove.mpiData->right->rank,
                                  0, mpi_comm, &reqs[1]);
                    MPI_Waitall(2, reqs, stats);

                    /* Sending data to right */
                    reqs[0] = MPI_REQUEST_NULL;
                    reqs[1] = MPI_REQUEST_NULL;
                    if ((myid != levelAbove.mpiData->right->rank) && (MPI_PROC_NULL != levelAbove.mpiData->right->rank))
                        MPI_Isend((void*)&levelAbove.mpiData->left->rank, 1, MPI_INT, levelAbove.mpiData->right->rank,
                                  0, mpi_comm, &reqs[0]);
                    if ((myid != levelAbove.mpiData->left->rank) && (MPI_PROC_NULL != levelAbove.mpiData->left->rank))
                        MPI_Irecv((void*)&tmpbuf, 1, MPI_INT, levelAbove.mpiData->left->rank, 0,
                                  mpi_comm, &reqs[1]);
                    MPI_Waitall(2, reqs, stats);
                }

                if (n > 0)
                {
                    /* Sending data downwards */
                    reqs[0] = MPI_REQUEST_NULL;
                    reqs[1] = MPI_REQUEST_NULL;
                    if ((myid != levelAbove.mpiData->down->rank) && (MPI_PROC_NULL != levelAbove.mpiData->down->rank))
                        MPI_Isend((void*)&myid, 1, MPI_INT, levelAbove.mpiData->down->rank, 0, mpi_comm,
                                  &reqs[0]);
                    if ((myid != levelAbove.mpiData->up->rank) && (MPI_PROC_NULL != levelAbove.mpiData->up->rank))
                        MPI_Irecv((void*)&mpiData->up->rank, 1, MPI_INT, levelAbove.mpiData->up->rank,
                                  0, mpi_comm, &reqs[1]);
                    MPI_Waitall(2, reqs, stats);

                    /* Sending data upwards */
                    reqs[0] = MPI_REQUEST_NULL;
                    reqs[1] = MPI_REQUEST_NULL;
                    if ((myid != levelAbove.mpiData->up->rank) && (MPI_PROC_NULL != levelAbove.mpiData->up->rank))
                        MPI_Isend((void*)&myid, 1, MPI_INT, levelAbove.mpiData->up->rank, 0, mpi_comm,
                                  &reqs[0]);
                    if ((myid != levelAbove.mpiData->down->rank) && (MPI_PROC_NULL != levelAbove.mpiData->down->rank))
                        MPI_Irecv((void*)&mpiData->down->rank, 1, MPI_INT, levelAbove.mpiData->down->rank,
                                  0, mpi_comm, &reqs[1]);
                    MPI_Waitall(2, reqs, stats);
                }
                else
                {
                    /* Sending data downwards */
                    reqs[0] = MPI_REQUEST_NULL;
                    reqs[1] = MPI_REQUEST_NULL;
                    if ((myid != levelAbove.mpiData->down->rank) && (MPI_PROC_NULL != levelAbove.mpiData->down->rank))
                        MPI_Isend((void*)&levelAbove.mpiData->up->rank, 1, MPI_INT,
                                  levelAbove.mpiData->down->rank, 0, mpi_comm, &reqs[0]);
                    if ((myid != levelAbove.mpiData->up->rank) && (MPI_PROC_NULL != levelAbove.mpiData->up->rank))
                        MPI_Irecv((void*)&tmpbuf, 1, MPI_INT, levelAbove.mpiData->up->rank,
                                  0, mpi_comm, &reqs[1]);
                    MPI_Waitall(2, reqs, stats);

                    /* Sending data upwards */
                    reqs[0] = MPI_REQUEST_NULL;
                    reqs[1] = MPI_REQUEST_NULL;
                    if ((myid != levelAbove.mpiData->up->rank) && (MPI_PROC_NULL != levelAbove.mpiData->up->rank))
                        MPI_Isend((void*)&levelAbove.mpiData->down->rank, 1, MPI_INT,
                                  levelAbove.mpiData->up->rank, 0, mpi_comm, &reqs[0]);
                    if ((myid != levelAbove.mpiData->down->rank) && (MPI_PROC_NULL != levelAbove.mpiData->down->rank))
                        MPI_Irecv((void*)&tmpbuf, 1, MPI_INT, levelAbove.mpiData->down->rank,
                                  0, mpi_comm, &reqs[1]);
                    MPI_Waitall(2, reqs, stats);
                }

                if (o > 0)
                {
                    /* Sending data to back */
                    reqs[0] = MPI_REQUEST_NULL;
                    reqs[1] = MPI_REQUEST_NULL;
                    if ((myid != levelAbove.mpiData->back->rank) && (MPI_PROC_NULL != levelAbove.mpiData->back->rank))
                        MPI_Isend((void*)&myid, 1, MPI_INT, levelAbove.mpiData->back->rank, 0, mpi_comm,
                                  &reqs[0]);
                    if ((myid != levelAbove.mpiData->front->rank) && (MPI_PROC_NULL != levelAbove.mpiData->front->rank))
                        MPI_Irecv((void*)&mpiData->front->rank, 1, MPI_INT, levelAbove.mpiData->front->rank,
                                  0, mpi_comm, &reqs[1]);
                    MPI_Waitall(2, reqs, stats);

                    /* Sending data to front */
                    reqs[0] = MPI_REQUEST_NULL;
                    reqs[1] = MPI_REQUEST_NULL;
                    if ((myid != levelAbove.mpiData->front->rank) && (MPI_PROC_NULL != levelAbove.mpiData->front->rank))
                        MPI_Isend((void*)&myid, 1, MPI_INT, levelAbove.mpiData->front->rank, 0, mpi_comm,
                                  &reqs[0]);
                    if ((myid != levelAbove.mpiData->back->rank) && (MPI_PROC_NULL != levelAbove.mpiData->back->rank))
                        MPI_Irecv((void*)&mpiData->back->rank, 1, MPI_INT, levelAbove.mpiData->back->rank,
                                  0, mpi_comm, &reqs[1]);
                    MPI_Waitall(2, reqs, stats);
                }
                else
                {
                    /* Sending data to back */
                    reqs[0] = MPI_REQUEST_NULL;
                    reqs[1] = MPI_REQUEST_NULL;
                    if ((myid != levelAbove.mpiData->back->rank) && (MPI_PROC_NULL != levelAbove.mpiData->back->rank))
                        MPI_Isend((void*)&levelAbove.mpiData->front->rank, 1, MPI_INT,
                                  levelAbove.mpiData->back->rank, 0, mpi_comm, &reqs[0]);
                    if ((myid != levelAbove.mpiData->front->rank) && (MPI_PROC_NULL != levelAbove.mpiData->front->rank))
                        MPI_Irecv((void*)&tmpbuf, 1, MPI_INT, levelAbove.mpiData->front->rank,
                                  0, mpi_comm, &reqs[1]);
                    MPI_Waitall(2, reqs, stats);

                    /* Sending data to front */
                    reqs[0] = MPI_REQUEST_NULL;
                    reqs[1] = MPI_REQUEST_NULL;
                    if ((myid != levelAbove.mpiData->front->rank) && (MPI_PROC_NULL != levelAbove.mpiData->front->rank))
                        MPI_Isend((void*)&levelAbove.mpiData->back->rank, 1, MPI_INT,
                                  levelAbove.mpiData->front->rank, 0, mpi_comm, &reqs[0]);
                    if ((myid != levelAbove.mpiData->back->rank) && (MPI_PROC_NULL != levelAbove.mpiData->back->rank))
                        MPI_Irecv((void*)&tmpbuf, 1, MPI_INT, levelAbove.mpiData->back->rank,
                                  0, mpi_comm, &reqs[1]);
                    MPI_Waitall(2, reqs, stats);
                }

                if (m <= 0 || n <= 0 || o <= 0)
                {
                    mpiData->left->rank = myid;
                    mpiData->right->rank = myid;
                    mpiData->down->rank = myid;
                    mpiData->up->rank = myid;
                    mpiData->back->rank = myid;
                    mpiData->front->rank = myid;
                }
            }
        }

        return ret;
    }

    std::ostream& operator<<(std::ostream& os, const Level& lv)
    {
        os << "Level: " << std::endl
           << " num: " << lv.num << std::endl
           << " m,n,o: " << lv.m << "," << lv.n << "," << lv.o << std::endl
           << " mgh,ngh,ogh: " << lv.mgh << "," << lv.ngh << "," << lv.ogh << std::endl
           << " h: " << lv.h << std::endl
           << " stencilType: " << lv.stencilType << std::endl
           << " stencilFactor: " << lv.stencilFactor << std::endl
           << " isCalculatedLocally: " << lv.isCalculatedLocally() << std::endl;
        return os;
    }

    void Level::setV(const std::shared_ptr<Cuboid>& v_)
    {
        v = v_;
    }

    int Level::getMgh() const
    {
        return mgh;
    }

    MGCL_STENCIL Level::getStencilType() const
    {
        return stencilType;
    }

    std::shared_ptr<VaryingStencilGpu>& Level::getStencilValuesGpu()
    {
        return stencilValuesGpu;
    }

    void Level::setStencilValuesGpu(std::shared_ptr<VaryingStencilGpu> sv)
    {
        stencilValuesGpu = sv;
    }

    int Level::getNgh() const
    {
        return ngh;
    }

    std::shared_ptr<VaryingStencil>& Level::getStencilValues()
    {
        return stencilValues;
    }

    std::shared_ptr<FixedStencil>& Level::getFixedStencil()
    {
        return fixedStencil;
    }

    int Level::getOgh() const
    {
        return ogh;
    }

    double Level::getStencilFactor() const
    {
        return stencilFactor;
    }

    int Level::getNum() const
    {
        return num;
    }

    Cuboid& Level::getV() const
    {
        assert(v && "v is null");
        return *v;
    }

    std::shared_ptr<Cuboid> Level::getVPtr() const
    {
        return v;
    }

    void Level::setF(const std::shared_ptr<Cuboid>& f_)
    {
        f = f_;
    }

    Cuboid& Level::getF() const
    {
        assert(f && "f is null");
        return *f;
    }

    std::shared_ptr<Cuboid> Level::getFPtr() const
    {
        return f;
    }

    void Level::setR(const std::shared_ptr<Cuboid>& r_)
    {
        r = r_;
    }

    Cuboid& Level::getR() const
    {
        assert(r && "r is null");
        return *r;
    }

    std::shared_ptr<Cuboid> Level::getRPtr() const
    {
        return r;
    }

    int Level::getN() const
    {
        return n;
    }

    int Level::getO() const
    {
        return o;
    }

    CuboidGpu& Level::getDVIn() const
    {
        if (!dVIn)
            error("dVIn is null.");
        return *dVIn;
    }

    CuboidGpu* Level::getDVInPtr() const
    {
        return dVIn.get();
    }

    void Level::setDVIn(std::shared_ptr<CuboidGpu> dVIn_)
    {
        dVIn = dVIn_;
        // if (dVIn)
        //     dVIn->retain();
    }

    CuboidGpu& Level::getDF() const
    {
        if (!dF)
            error("dF is null.");
        return *dF;
    }

    CuboidGpu* Level::getDFPtr() const
    {
        return dF.get();
    }

    void Level::setDF(std::shared_ptr<CuboidGpu> dF_)
    {
        dF = dF_;
        // if (dF)
        //     dF->retain();
    }

    int Level::getM() const
    {
        return m;
    }

    double Level::getH() const
    {
        return h;
    }

    void Level::setH(double h_)
    {
        h = h_;
    }

    CuboidGpu& Level::getDVOut() const
    {
        if (!dVOut)
            error("dVOut is null.");
        return *dVOut;
    }

    CuboidGpu* Level::getDVOutPtr() const
    {
        return dVOut.get();
    }

    void Level::setDVOut(std::shared_ptr<CuboidGpu> dVOut_)
    {
        dVOut = dVOut_;
        // if (dVOut)
        //     dVOut->retain();
    }

    CuboidGpu& Level::getDR() const
    {
        if (!dR)
            error("dR is null.");
        return *dR;
    }

    CuboidGpu* Level::getDRPtr() const
    {
        return dR.get();
    }

    void Level::setDR(std::shared_ptr<CuboidGpu> dR_)
    {
        dR = dR_;
        // if (dR)
        //     dR->retain();
    }

    CuboidGpu& Level::getDRsq() const
    {
        if (!dRsq)
            error("dRsq is null.");
        return *dRsq;
    }

    CuboidGpu* Level::getDRsqPtr() const
    {
        return dRsq.get();
    }

    void Level::setDRsq(std::shared_ptr<CuboidGpu> dR_)
    {
        dRsq = dR_;
    }

    MPILevelData* Level::getMpiDataPtr()
    {
        return mpiData.get();
    }

    MPILevelData& Level::getMpiData()
    {
        if (mpiData != nullptr)
            return *mpiData;
        else
            error("mpiData is null.");
    }

    CuboidBS& Level::getVBS() const
    {
        if (!v_bs)
            error("v_bs is null!");
        return *v_bs;
    }

    std::shared_ptr<CuboidBS> Level::getVBSPtr() const
    {
        return v_bs;
    }

    void Level::setVBS(const std::shared_ptr<CuboidBS>& v_)
    {
        v_bs = v_;
    }

    CuboidBS& Level::getFBS() const
    {
        if (!f_bs)
            error("f_bs is null!");
        return *f_bs;
    }

    std::shared_ptr<CuboidBS> Level::getFBSPtr() const
    {
        return f_bs;
    }

    void Level::setFBS(const std::shared_ptr<CuboidBS>& f_)
    {
        f_bs = f_;
    }

    CuboidBS& Level::getRBS() const
    {
        if (!r_bs)
            error("r_bs is null!");
        return *r_bs;
    }

    std::shared_ptr<CuboidBS> Level::getRBSPtr() const
    {
        return r_bs;
    }

    void Level::setRBS(const std::shared_ptr<CuboidBS>& r_)
    {
        r_bs = r_;
    }

    CuboidBSGpu& Level::getDVBSIn() const
    {
        if (!dVIn_bs)
            error("dVBS is null.");
        return *dVIn_bs;
    }

    std::shared_ptr<CuboidBSGpu> Level::getDVBSInPtr() const
    {
        return dVIn_bs;
    }

    void Level::setDVBSIn(const std::shared_ptr<CuboidBSGpu>& v_)
    {
        dVIn_bs = v_;
    }

    CuboidBSGpu& Level::getDVBSOut() const
    {
        if (!dVOut_bs)
            error("dVBS is null.");
        return *dVOut_bs;
    }

    std::shared_ptr<CuboidBSGpu> Level::getDVBSOutPtr() const
    {
        return dVOut_bs;
    }

    void Level::setDVBSOut(const std::shared_ptr<CuboidBSGpu>& v_)
    {
        dVOut_bs = v_;
    }

    CuboidBSGpu& Level::getDFBS() const
    {
        if (!dF_bs)
            error("dFBS is null.");
        return *dF_bs;
    }

    std::shared_ptr<CuboidBSGpu> Level::getDFBSPtr() const
    {
        return dF_bs;
    }

    void Level::setDFBS(const std::shared_ptr<CuboidBSGpu>& f_)
    {
        dF_bs = f_;
    }

    CuboidBSGpu& Level::getDRBS() const
    {
        if (!dR_bs)
            error("dRBS is null.");
        return *dR_bs;
    }

    std::shared_ptr<CuboidBSGpu> Level::getDRBSPtr() const
    {
        return dR_bs;
    }

    void Level::setDRBS(const std::shared_ptr<CuboidBSGpu>& r_)
    {
        dR_bs = r_;
    }

    CuboidBSGpu& Level::getDRsqBS() const
    {
        if (!dRsq_bs)
            error("dRsqBS is null.");
        return *dRsq_bs;
    }

    std::shared_ptr<CuboidBSGpu> Level::getDRsqBSPtr() const
    {
        return dRsq_bs;
    }

    void Level::setDRsqBS(const std::shared_ptr<CuboidBSGpu> dR_)
    {
        dRsq_bs = dR_;
    }

    /**
     * @brief Returns true, if the calculations on this levels is done locally. If MPI is not used, this function
     * always returns true. Otherwise it checks whether this level is equal to or above the mpiLevelThreshold.
     */
    bool Level::isCalculatedLocally() const
    {
        return !problem->useMpi() || problem->mpiSize() == 1 || num >= problem->getMpiLevelThreshold();
    }

    /**
     * @brief Creates the inverse of the blockstencil depending on smoother type.
     *
     */
    void Level::createInverseOfBlockstencilSeq()
    {
        if (stencilType != MGCL_BLOCKSTENCIL)
        {
            error("Problem::createInverseOfBlockstencilSeq: stencilType is not MGCL_BLOCKSTENCIL. Use Problem::setStencilType(MGCL_BLOCKSTENCIL) first.");
        }

        // Create inverse for level 0. Coarser levels' inverse will be created after galerkin
        if (problem->getSmootherType() == MGCL_JACOBI_SCALAR)
        {
            blockstencilInv = blockstencil->invertDiagonal();
        }
        else
        {
            blockstencilInv = blockstencil->invertCenterMatrices();
        }

        if (auto bsi = std::get_if<std::shared_ptr<Blockstencil>>(&blockstencilInv))
        {
            bsi->get()->updateGhosts(mpiData.get(), isCalculatedLocally());
        }
        else if (auto bsi = std::get_if<std::shared_ptr<CuboidBS>>(&blockstencilInv))
        {
            bsi->get()->updateGhosts(mpiData.get(), isCalculatedLocally());
        }
        // else if (auto bsi = std::get_if<std::shared_ptr<BlockstencilGpu>>(&blockstencilInv))
        // {
        //     bsi->get()->updateGhosts(mpiData.get(), isCalculatedLocally());
        // }
        // else if (auto bsi = std::get_if<std::shared_ptr<CuboidBSGpu>>(&blockstencilInv))
        // {
        //     bsi->get()->updateGhosts(mpiData.get(), isCalculatedLocally());
        // }
        // TODO gpu variants
    }

    /**
     * @brief Creates the inverse of the blockstencil for Gpu depending on smoother type.
     *
     */
    void Level::createInverseOfBlockstencilGpu()
    {
        if (stencilType != MGCL_BLOCKSTENCIL)
        {
            error("Problem::createInverseOfBlockstencilGpu: stencilType is not MGCL_BLOCKSTENCIL. Use Problem::setStencilType(MGCL_BLOCKSTENCIL) first.");
        }

        if (!problem->getUseOpencl())
        {
            error("Problem::createInverseOfBlockstencilGpu: OpenCL is not enabled. Use Problem::setUseOpencl(true) first.");
        }

        // Create inverse for level 0. Coarser levels' inverse will be created after galerkin
        if (problem->getSmootherType() == MGCL_JACOBI_SCALAR)
        {
            blockstencilInv = blockstencilGpu->invertDiagonal(problem->getContext(), problem->getCommands());
        }
        else
        {
            blockstencilInv = blockstencilGpu->invertCenterMatrices(problem->getContext(), problem->getCommands(), problem->getProgram());
        }
    }
}
