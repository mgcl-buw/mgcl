#include "level.hpp"
#include "cuboid.hpp" // for Cuboid
#include "mpi_stencil.hpp"
#include "mpi_util.hpp"
#include "multigrid_engine.hpp" // for Problem, MultigridEngine
#include "problem.hpp"

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
            stencilFactor = 1.0 / (30.0 * h * h);

        if (num_ < 0 || num_ > problem->getMaxlevel())
            throw std::invalid_argument(std::string("num is invalid! num: ")
                                            .append(std::to_string(num_))
                                            .append(", problem.maxlevel: ")
                                            .append(std::to_string(problem->getMaxlevel())));

        // if (useMpi)
        if (problem->useMpi())
            mpiData = std::make_unique<MPILevelData>(problem->getMpiComm());
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

        // First init MPI data to get information about neighbours and be able to update ghosts.
        initMpiData();

        // Always allocate data on level 0 so input data is copied and not left uninitialized.
        if (num == 0)
        {
            // copy stencilsValues pointer from Problem to first Level and update ghosts
            if (stencilType == MGCL_VARYING)
            {
                stencilValues = problem->stencilValues;

                // Only update ghosts of stencilValues if the threshold is not 1, because when applying Galerkin,
                // the values of level 0 are gathered anyways. Plus the sizes are different, so this would result
                // in a MPI error when trying to update ghosts.
                // Update ghosts locally if the threshold is 0, i.e. everything is done locally.
                if (problem->getMpiLevelThreshold() != 1)
                    updateGhostsStencilMpi(*stencilValues, mpiData.get(), problem->isPeriodic(),
                                           problem->getMpiLevelThreshold() == 0);
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
                // If level threshold is 0 or 1, stencilValues must have global sizes.
                if (problem->getMpiLevelThreshold() <= 1 && problem->mpiRank() == 0)
                    stencilValuesGpu = std::make_shared<VaryingStencilGpu>(
                        problem->mGlobal, problem->nGlobal, problem->oGlobal, 3,
                        std::max(2, problem->getJacobiIterationsPerKernel()),
                        problem->getContext(), problem->getCommands());
                else
                    stencilValuesGpu = std::make_shared<VaryingStencilGpu>(
                        m, n, o, 3, std::max(2, problem->getJacobiIterationsPerKernel()),
                        problem->getContext(), problem->getCommands());

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
                mpiData->left = mpiData->rank;
                mpiData->right = mpiData->rank;
                mpiData->down = mpiData->rank;
                mpiData->up = mpiData->rank;
                mpiData->back = mpiData->rank;
                mpiData->front = mpiData->rank;
                return MPI_SUCCESS;
            }

            if (num == 0)
            {
                /* Calculating neighbours */
                ret = MPI_Cart_shift(mpiData->comm, 2, 1, &mpiData->left, &mpiData->right);
                mpi_util::mgclCheckMpiError(mpiData->comm, ret, "MPI_Cart_shift x-direction");
                ret = MPI_Cart_shift(mpiData->comm, 1, 1, &mpiData->down, &mpiData->up);
                mpi_util::mgclCheckMpiError(mpiData->comm, ret, "MPI_Cart_shift y-direction");
                ret = MPI_Cart_shift(mpiData->comm, 0, 1, &mpiData->front, &mpiData->back);
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
                    mpiData->left = myid;
                    mpiData->right = myid;
                    mpiData->down = myid;
                    mpiData->up = myid;
                    mpiData->back = myid;
                    mpiData->front = myid;
                }
                else
                {
                    /* Initializing neighbours */
                    mpiData->left = MPI_PROC_NULL;
                    mpiData->right = MPI_PROC_NULL;
                    mpiData->down = MPI_PROC_NULL;
                    mpiData->up = MPI_PROC_NULL;
                    mpiData->back = MPI_PROC_NULL;
                    mpiData->front = MPI_PROC_NULL;
                }

                if (m > 0)
                {
                    /* Sending data to left */
                    reqs[0] = MPI_REQUEST_NULL;
                    reqs[1] = MPI_REQUEST_NULL;

                    if ((myid != levelAbove.mpiData->left) && (MPI_PROC_NULL != levelAbove.mpiData->left))
                        MPI_Isend((void*)&myid, 1, MPI_INT, levelAbove.mpiData->left, 0, mpi_comm, &reqs[0]);

                    if ((myid != levelAbove.mpiData->right) && (MPI_PROC_NULL != levelAbove.mpiData->right))
                        MPI_Irecv((void*)&mpiData->right, 1, MPI_INT, levelAbove.mpiData->right, 0, mpi_comm, &reqs[1]);

                    MPI_Waitall(2, reqs, stats);

                    /* Sending data to right */
                    reqs[0] = MPI_REQUEST_NULL;
                    reqs[1] = MPI_REQUEST_NULL;
                    if ((myid != levelAbove.mpiData->right) && (MPI_PROC_NULL != levelAbove.mpiData->right))
                        MPI_Isend((void*)&myid, 1, MPI_INT, levelAbove.mpiData->right, 0, mpi_comm,
                                  &reqs[0]);
                    if ((myid != levelAbove.mpiData->left) && (MPI_PROC_NULL != levelAbove.mpiData->left))
                        MPI_Irecv((void*)&mpiData->left, 1, MPI_INT, levelAbove.mpiData->left, 0,
                                  mpi_comm, &reqs[1]);
                    MPI_Waitall(2, reqs, stats);
                }
                else
                {
                    /* Sending data to left */
                    reqs[0] = MPI_REQUEST_NULL;
                    reqs[1] = MPI_REQUEST_NULL;
                    if ((myid != levelAbove.mpiData->left) && (MPI_PROC_NULL != levelAbove.mpiData->left))
                        MPI_Isend((void*)&levelAbove.mpiData->right, 1, MPI_INT, levelAbove.mpiData->left,
                                  0, mpi_comm, &reqs[0]);
                    if ((myid != levelAbove.mpiData->right) && (MPI_PROC_NULL != levelAbove.mpiData->right))
                        MPI_Irecv((void*)&tmpbuf, 1, MPI_INT, levelAbove.mpiData->right,
                                  0, mpi_comm, &reqs[1]);
                    MPI_Waitall(2, reqs, stats);

                    /* Sending data to right */
                    reqs[0] = MPI_REQUEST_NULL;
                    reqs[1] = MPI_REQUEST_NULL;
                    if ((myid != levelAbove.mpiData->right) && (MPI_PROC_NULL != levelAbove.mpiData->right))
                        MPI_Isend((void*)&levelAbove.mpiData->left, 1, MPI_INT, levelAbove.mpiData->right,
                                  0, mpi_comm, &reqs[0]);
                    if ((myid != levelAbove.mpiData->left) && (MPI_PROC_NULL != levelAbove.mpiData->left))
                        MPI_Irecv((void*)&tmpbuf, 1, MPI_INT, levelAbove.mpiData->left, 0,
                                  mpi_comm, &reqs[1]);
                    MPI_Waitall(2, reqs, stats);
                }

                if (n > 0)
                {
                    /* Sending data downwards */
                    reqs[0] = MPI_REQUEST_NULL;
                    reqs[1] = MPI_REQUEST_NULL;
                    if ((myid != levelAbove.mpiData->down) && (MPI_PROC_NULL != levelAbove.mpiData->down))
                        MPI_Isend((void*)&myid, 1, MPI_INT, levelAbove.mpiData->down, 0, mpi_comm,
                                  &reqs[0]);
                    if ((myid != levelAbove.mpiData->up) && (MPI_PROC_NULL != levelAbove.mpiData->up))
                        MPI_Irecv((void*)&mpiData->up, 1, MPI_INT, levelAbove.mpiData->up,
                                  0, mpi_comm, &reqs[1]);
                    MPI_Waitall(2, reqs, stats);

                    /* Sending data upwards */
                    reqs[0] = MPI_REQUEST_NULL;
                    reqs[1] = MPI_REQUEST_NULL;
                    if ((myid != levelAbove.mpiData->up) && (MPI_PROC_NULL != levelAbove.mpiData->up))
                        MPI_Isend((void*)&myid, 1, MPI_INT, levelAbove.mpiData->up, 0, mpi_comm,
                                  &reqs[0]);
                    if ((myid != levelAbove.mpiData->down) && (MPI_PROC_NULL != levelAbove.mpiData->down))
                        MPI_Irecv((void*)&mpiData->down, 1, MPI_INT, levelAbove.mpiData->down,
                                  0, mpi_comm, &reqs[1]);
                    MPI_Waitall(2, reqs, stats);
                }
                else
                {
                    /* Sending data downwards */
                    reqs[0] = MPI_REQUEST_NULL;
                    reqs[1] = MPI_REQUEST_NULL;
                    if ((myid != levelAbove.mpiData->down) && (MPI_PROC_NULL != levelAbove.mpiData->down))
                        MPI_Isend((void*)&levelAbove.mpiData->up, 1, MPI_INT,
                                  levelAbove.mpiData->down, 0, mpi_comm, &reqs[0]);
                    if ((myid != levelAbove.mpiData->up) && (MPI_PROC_NULL != levelAbove.mpiData->up))
                        MPI_Irecv((void*)&tmpbuf, 1, MPI_INT, levelAbove.mpiData->up,
                                  0, mpi_comm, &reqs[1]);
                    MPI_Waitall(2, reqs, stats);

                    /* Sending data upwards */
                    reqs[0] = MPI_REQUEST_NULL;
                    reqs[1] = MPI_REQUEST_NULL;
                    if ((myid != levelAbove.mpiData->up) && (MPI_PROC_NULL != levelAbove.mpiData->up))
                        MPI_Isend((void*)&levelAbove.mpiData->down, 1, MPI_INT,
                                  levelAbove.mpiData->up, 0, mpi_comm, &reqs[0]);
                    if ((myid != levelAbove.mpiData->down) && (MPI_PROC_NULL != levelAbove.mpiData->down))
                        MPI_Irecv((void*)&tmpbuf, 1, MPI_INT, levelAbove.mpiData->down,
                                  0, mpi_comm, &reqs[1]);
                    MPI_Waitall(2, reqs, stats);
                }

                if (o > 0)
                {
                    /* Sending data to back */
                    reqs[0] = MPI_REQUEST_NULL;
                    reqs[1] = MPI_REQUEST_NULL;
                    if ((myid != levelAbove.mpiData->back) && (MPI_PROC_NULL != levelAbove.mpiData->back))
                        MPI_Isend((void*)&myid, 1, MPI_INT, levelAbove.mpiData->back, 0, mpi_comm,
                                  &reqs[0]);
                    if ((myid != levelAbove.mpiData->front) && (MPI_PROC_NULL != levelAbove.mpiData->front))
                        MPI_Irecv((void*)&mpiData->front, 1, MPI_INT, levelAbove.mpiData->front,
                                  0, mpi_comm, &reqs[1]);
                    MPI_Waitall(2, reqs, stats);

                    /* Sending data to front */
                    reqs[0] = MPI_REQUEST_NULL;
                    reqs[1] = MPI_REQUEST_NULL;
                    if ((myid != levelAbove.mpiData->front) && (MPI_PROC_NULL != levelAbove.mpiData->front))
                        MPI_Isend((void*)&myid, 1, MPI_INT, levelAbove.mpiData->front, 0, mpi_comm,
                                  &reqs[0]);
                    if ((myid != levelAbove.mpiData->back) && (MPI_PROC_NULL != levelAbove.mpiData->back))
                        MPI_Irecv((void*)&mpiData->back, 1, MPI_INT, levelAbove.mpiData->back,
                                  0, mpi_comm, &reqs[1]);
                    MPI_Waitall(2, reqs, stats);
                }
                else
                {
                    /* Sending data to back */
                    reqs[0] = MPI_REQUEST_NULL;
                    reqs[1] = MPI_REQUEST_NULL;
                    if ((myid != levelAbove.mpiData->back) && (MPI_PROC_NULL != levelAbove.mpiData->back))
                        MPI_Isend((void*)&levelAbove.mpiData->front, 1, MPI_INT,
                                  levelAbove.mpiData->back, 0, mpi_comm, &reqs[0]);
                    if ((myid != levelAbove.mpiData->front) && (MPI_PROC_NULL != levelAbove.mpiData->front))
                        MPI_Irecv((void*)&tmpbuf, 1, MPI_INT, levelAbove.mpiData->front,
                                  0, mpi_comm, &reqs[1]);
                    MPI_Waitall(2, reqs, stats);

                    /* Sending data to front */
                    reqs[0] = MPI_REQUEST_NULL;
                    reqs[1] = MPI_REQUEST_NULL;
                    if ((myid != levelAbove.mpiData->front) && (MPI_PROC_NULL != levelAbove.mpiData->front))
                        MPI_Isend((void*)&levelAbove.mpiData->back, 1, MPI_INT,
                                  levelAbove.mpiData->front, 0, mpi_comm, &reqs[0]);
                    if ((myid != levelAbove.mpiData->back) && (MPI_PROC_NULL != levelAbove.mpiData->back))
                        MPI_Irecv((void*)&tmpbuf, 1, MPI_INT, levelAbove.mpiData->back,
                                  0, mpi_comm, &reqs[1]);
                    MPI_Waitall(2, reqs, stats);
                }

                if (m <= 0 || n <= 0 || o <= 0)
                {
                    mpiData->left = myid;
                    mpiData->right = myid;
                    mpiData->down = myid;
                    mpiData->up = myid;
                    mpiData->back = myid;
                    mpiData->front = myid;
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
            throw "dVIn is null.";
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
            throw "dF is null.";
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
            throw "dVOut is null.";
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
            throw "dR is null.";
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
            throw "dRsq is null.";
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
            throw "mpiData is null.";
    }

    /**
     * @brief Returns true, if the calculations on this levels is done locally. If MPI is not used, this function
     * always returns true. Otherwise it checks whether this level is equal to or above the mpiLevelThreshold.
     */
    bool Level::isCalculatedLocally() const
    {
        return !problem->useMpi() || problem->mpiSize() == 1 || num >= problem->getMpiLevelThreshold();
    }
}
