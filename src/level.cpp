#include "level.hpp"
#include "cuboid.hpp"           // for Cuboid
#include "multigrid_engine.hpp" // for Problem, MultigridEngine
#include "problem.hpp"

#include <cstddef>   // for NULL
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
    Level::Level(Problem *problem_, int num_)
        : problem(problem_),
          num(num_),
          m(problem_->getM() >> num_),
          n(problem_->getN() >> num_),
          o(problem_->getO() >> num_),
          mgh(m + 2 * problem->getGhosts()),
          ngh(n + 2 * problem->getGhosts()),
          ogh(o + 2 * problem->getGhosts()),
          h(1.0 / (double)problem->m_global), // TODO differentiate for non-cube-like domains
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

        if (problem->useMpi())
            mpiData = std::make_unique<MPIData>(problem->getMpiComm());
    }

    Level::~Level()
    {
        int err;
        if (dVIn)
        {
            err = clReleaseMemObject(dVIn);
            mgclCheckError(err, "clReleaseMemObject(dVIn)");
        }

        if (dF)
        {
            err = clReleaseMemObject(dF);
            mgclCheckError(err, "clReleaseMemObject(dF)");
        }

        if (dVOut)
        {
            err = clReleaseMemObject(dVOut);
            mgclCheckError(err, "clReleaseMemObject(dVOut)");
        }

        if (dR)
        {
            err = clReleaseMemObject(dR);
            mgclCheckError(err, "clReleaseMemObject(dR)");
        }
    }

    /**
     * @brief Initializes data for this level.
     *
     * @return true All good.
     * @return false Something went wrong.
     */
    bool Level::init()
    {
        // First init MPI data, so dimensions of Cuboids is updated.
        initMpiData();

        // TODO F does not need ghosts
        // TODO R needs same amount of ghosts as V
        if (num == 0)
        {
            // move stencilsValues pointer from Problem to first Level
            if (stencilType == MGCL_VARYING)
                stencilValues = problem->stencilValues;

            // create ghosted arrays for v and f on host if device buffer should not be reused
            if (!problem->reuse_opencl_buffers && !problem->copy_buffer_data)
            {
                v = std::make_shared<Cuboid>(m, n, o, problem->ghosts, problem->ghosts, problem->ghosts);
                f = std::make_shared<Cuboid>(m, n, o, problem->ghosts, problem->ghosts, problem->ghosts);

                // copy initial input data from conf into mgcl data struct
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                        for (int k = 0; k < o; k++)
                        {
                            getV()[i + problem->ghosts][j + problem->ghosts][k + problem->ghosts] =
                                problem->getV()[i + problem->ghosts_in][j + problem->ghosts_in][k + problem->ghosts_in];
                            getF()[i + problem->ghosts][j + problem->ghosts][k + problem->ghosts] =
                                problem->getF()[i + problem->ghosts_in][j + problem->ghosts_in][k + problem->ghosts_in];
                        }

                if (problem->bc == BC::PERIODIC)
                    MultigridEngine::updateGhostsSeq(getF(), mpiData.get());
            }

            // r on host is only needed if opencl should not be used
            if (!problem->use_opencl)
            {
                r = std::make_shared<Cuboid>(m, n, o, problem->ghosts, problem->ghosts, problem->ghosts);
            }
        }
        else
        {
            if (!problem->use_opencl)
            {
                v = std::make_shared<Cuboid>(m, n, o, problem->ghosts, problem->ghosts, problem->ghosts);
                f = std::make_shared<Cuboid>(m, n, o, problem->ghosts, problem->ghosts, problem->ghosts);
                r = std::make_shared<Cuboid>(m, n, o, problem->ghosts, problem->ghosts, problem->ghosts);
            }
        }

        if (initOpenCLBuffers() != CL_SUCCESS)
            return false;

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

        // create gpu buffer for varying stencil if needed
        if (stencilType == MGCL_VARYING)
        {
            stencilValuesGpu = std::make_shared<VaryingStencilGpu>(
                m, n, o, 3, std::max(2, problem->getJacobiIterationsPerKernel()),
                problem->getContext(), problem->getCommands());

            if (num == 0)
                // Fill stencil values on gpu on level 0 from input stencil
                stencilValuesGpu->fill(*stencilValues, problem->getCommands());
        }

        // create d_v_in and d_f buffers on level zero and copy data to it only if buffers should not be reused
        if (num == 0)
        {
            if (problem->getReuseOpenclBuffers())
            {
                dVIn = problem->getDV();
                dF = problem->getDF();

                // retain buffers (i.e. increase internal reference count so they won't be released by accident)
                err = clRetainMemObject(dVIn);
                mgclCheckError(err, "clRetainMemObject(dVIn)");
                err = clRetainMemObject(dF);
                mgclCheckError(err, "clRetainMemObject(dF)");
            }
            else if (problem->getCopyBufferData())
            {
                dVIn = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                      sizeof(double) * mgh * ngh * ogh, NULL, &err);
                mgclCheckError(err, "clCreateBuffer");
                dF = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    sizeof(double) * mgh * ngh * ogh, NULL, &err);
                mgclCheckError(err, "clCreateBuffer");
                problem->getOpenCLHelper().copyInputBuffers();
            }
            else
            {
                int pointer_flag = deviceType == CL_DEVICE_TYPE_GPU ? CL_MEM_COPY_HOST_PTR : CL_MEM_USE_HOST_PTR;
                dVIn = clCreateBuffer(context, CL_MEM_READ_WRITE | pointer_flag,
                                      sizeof(double) * mgh * ngh * ogh, (*v)[0][0], &err);
                mgclCheckError(err, "clCreateBuffer");
                dF = clCreateBuffer(context, CL_MEM_READ_WRITE | pointer_flag,
                                    sizeof(double) * mgh * ngh * ogh, (*f)[0][0], &err);
                mgclCheckError(err, "clCreateBuffer");
            }
        }
        else
        {
            dVIn = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                  sizeof(double) * mgh * ngh * ogh, NULL, &err);
            mgclCheckError(err, "clCreateBuffer");
            dF = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                sizeof(double) * mgh * ngh * ogh, NULL, &err);
            mgclCheckError(err, "clCreateBuffer");
        }

        dVOut = clCreateBuffer(context, CL_MEM_READ_WRITE,
                               sizeof(double) * mgh * ngh * ogh, NULL, &err);
        mgclCheckError(err, "clCreateBuffer");
        dR = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double) * mgh * ngh * ogh,
                            NULL, &err);
        mgclCheckError(err, "clCreateBuffer");

        // Init dVOut and dR to zero.
        double zero = 0.0;
        err = clEnqueueFillBuffer(problem->getCommands(), dVOut, &zero, sizeof(cl_double), 0,
                                  sizeof(double) * mgh * ngh * ogh, 0, NULL, NULL);
        mgclCheckError(err, "initializing dVOut to 0");

        err = clEnqueueFillBuffer(problem->getCommands(), dR, &zero, sizeof(cl_double), 0,
                                  sizeof(double) * mgh * ngh * ogh, 0, NULL, NULL);
        mgclCheckError(err, "initializing dR to 0");

        err = MultigridEngine::updateGhosts(*problem, dF, mgh, ngh, ogh, problem->ghosts, problem->ghosts,
                                            problem->ghosts, mpiData.get());
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

        if (problem->useMpi())
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
                mgcl::mgclCheckMpiError(mpiData->comm, ret, "MPI_Cart_shift x-direction");
                ret = MPI_Cart_shift(mpiData->comm, 1, 1, &mpiData->down, &mpiData->up);
                mgcl::mgclCheckMpiError(mpiData->comm, ret, "MPI_Cart_shift y-direction");
                ret = MPI_Cart_shift(mpiData->comm, 0, 1, &mpiData->front, &mpiData->back);
                mgcl::mgclCheckMpiError(mpiData->comm, ret, "MPI_Cart_shift z-direction");
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

                Level &levelAbove = problem->getLevelAt(num - 1);

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
                        MPI_Isend((void *)&myid, 1, MPI_INT, levelAbove.mpiData->left, 0, mpi_comm, &reqs[0]);

                    if ((myid != levelAbove.mpiData->right) && (MPI_PROC_NULL != levelAbove.mpiData->right))
                        MPI_Irecv((void *)&mpiData->right, 1, MPI_INT, levelAbove.mpiData->right, 0, mpi_comm, &reqs[1]);

                    MPI_Waitall(2, reqs, stats);

                    /* Sending data to right */
                    reqs[0] = MPI_REQUEST_NULL;
                    reqs[1] = MPI_REQUEST_NULL;
                    if ((myid != levelAbove.mpiData->right) && (MPI_PROC_NULL != levelAbove.mpiData->right))
                        MPI_Isend((void *)&myid, 1, MPI_INT, levelAbove.mpiData->right, 0, mpi_comm,
                                  &reqs[0]);
                    if ((myid != levelAbove.mpiData->left) && (MPI_PROC_NULL != levelAbove.mpiData->left))
                        MPI_Irecv((void *)&mpiData->left, 1, MPI_INT, levelAbove.mpiData->left, 0,
                                  mpi_comm, &reqs[1]);
                    MPI_Waitall(2, reqs, stats);
                }
                else
                {
                    /* Sending data to left */
                    reqs[0] = MPI_REQUEST_NULL;
                    reqs[1] = MPI_REQUEST_NULL;
                    if ((myid != levelAbove.mpiData->left) && (MPI_PROC_NULL != levelAbove.mpiData->left))
                        MPI_Isend((void *)&levelAbove.mpiData->right, 1, MPI_INT, levelAbove.mpiData->left,
                                  0, mpi_comm, &reqs[0]);
                    if ((myid != levelAbove.mpiData->right) && (MPI_PROC_NULL != levelAbove.mpiData->right))
                        MPI_Irecv((void *)&tmpbuf, 1, MPI_INT, levelAbove.mpiData->right,
                                  0, mpi_comm, &reqs[1]);
                    MPI_Waitall(2, reqs, stats);

                    /* Sending data to right */
                    reqs[0] = MPI_REQUEST_NULL;
                    reqs[1] = MPI_REQUEST_NULL;
                    if ((myid != levelAbove.mpiData->right) && (MPI_PROC_NULL != levelAbove.mpiData->right))
                        MPI_Isend((void *)&levelAbove.mpiData->left, 1, MPI_INT, levelAbove.mpiData->right,
                                  0, mpi_comm, &reqs[0]);
                    if ((myid != levelAbove.mpiData->left) && (MPI_PROC_NULL != levelAbove.mpiData->left))
                        MPI_Irecv((void *)&tmpbuf, 1, MPI_INT, levelAbove.mpiData->left, 0,
                                  mpi_comm, &reqs[1]);
                    MPI_Waitall(2, reqs, stats);
                }

                if (n > 0)
                {
                    /* Sending data downwards */
                    reqs[0] = MPI_REQUEST_NULL;
                    reqs[1] = MPI_REQUEST_NULL;
                    if ((myid != levelAbove.mpiData->down) && (MPI_PROC_NULL != levelAbove.mpiData->down))
                        MPI_Isend((void *)&myid, 1, MPI_INT, levelAbove.mpiData->down, 0, mpi_comm,
                                  &reqs[0]);
                    if ((myid != levelAbove.mpiData->up) && (MPI_PROC_NULL != levelAbove.mpiData->up))
                        MPI_Irecv((void *)&mpiData->up, 1, MPI_INT, levelAbove.mpiData->up,
                                  0, mpi_comm, &reqs[1]);
                    MPI_Waitall(2, reqs, stats);

                    /* Sending data upwards */
                    reqs[0] = MPI_REQUEST_NULL;
                    reqs[1] = MPI_REQUEST_NULL;
                    if ((myid != levelAbove.mpiData->up) && (MPI_PROC_NULL != levelAbove.mpiData->up))
                        MPI_Isend((void *)&myid, 1, MPI_INT, levelAbove.mpiData->up, 0, mpi_comm,
                                  &reqs[0]);
                    if ((myid != levelAbove.mpiData->down) && (MPI_PROC_NULL != levelAbove.mpiData->down))
                        MPI_Irecv((void *)&mpiData->down, 1, MPI_INT, levelAbove.mpiData->down,
                                  0, mpi_comm, &reqs[1]);
                    MPI_Waitall(2, reqs, stats);
                }
                else
                {
                    /* Sending data downwards */
                    reqs[0] = MPI_REQUEST_NULL;
                    reqs[1] = MPI_REQUEST_NULL;
                    if ((myid != levelAbove.mpiData->down) && (MPI_PROC_NULL != levelAbove.mpiData->down))
                        MPI_Isend((void *)&levelAbove.mpiData->up, 1, MPI_INT,
                                  levelAbove.mpiData->down, 0, mpi_comm, &reqs[0]);
                    if ((myid != levelAbove.mpiData->up) && (MPI_PROC_NULL != levelAbove.mpiData->up))
                        MPI_Irecv((void *)&tmpbuf, 1, MPI_INT, levelAbove.mpiData->up,
                                  0, mpi_comm, &reqs[1]);
                    MPI_Waitall(2, reqs, stats);

                    /* Sending data upwards */
                    reqs[0] = MPI_REQUEST_NULL;
                    reqs[1] = MPI_REQUEST_NULL;
                    if ((myid != levelAbove.mpiData->up) && (MPI_PROC_NULL != levelAbove.mpiData->up))
                        MPI_Isend((void *)&levelAbove.mpiData->down, 1, MPI_INT,
                                  levelAbove.mpiData->up, 0, mpi_comm, &reqs[0]);
                    if ((myid != levelAbove.mpiData->down) && (MPI_PROC_NULL != levelAbove.mpiData->down))
                        MPI_Irecv((void *)&tmpbuf, 1, MPI_INT, levelAbove.mpiData->down,
                                  0, mpi_comm, &reqs[1]);
                    MPI_Waitall(2, reqs, stats);
                }

                if (o > 0)
                {
                    /* Sending data to back */
                    reqs[0] = MPI_REQUEST_NULL;
                    reqs[1] = MPI_REQUEST_NULL;
                    if ((myid != levelAbove.mpiData->back) && (MPI_PROC_NULL != levelAbove.mpiData->back))
                        MPI_Isend((void *)&myid, 1, MPI_INT, levelAbove.mpiData->back, 0, mpi_comm,
                                  &reqs[0]);
                    if ((myid != levelAbove.mpiData->front) && (MPI_PROC_NULL != levelAbove.mpiData->front))
                        MPI_Irecv((void *)&mpiData->front, 1, MPI_INT, levelAbove.mpiData->front,
                                  0, mpi_comm, &reqs[1]);
                    MPI_Waitall(2, reqs, stats);

                    /* Sending data to front */
                    reqs[0] = MPI_REQUEST_NULL;
                    reqs[1] = MPI_REQUEST_NULL;
                    if ((myid != levelAbove.mpiData->front) && (MPI_PROC_NULL != levelAbove.mpiData->front))
                        MPI_Isend((void *)&myid, 1, MPI_INT, levelAbove.mpiData->front, 0, mpi_comm,
                                  &reqs[0]);
                    if ((myid != levelAbove.mpiData->back) && (MPI_PROC_NULL != levelAbove.mpiData->back))
                        MPI_Irecv((void *)&mpiData->back, 1, MPI_INT, levelAbove.mpiData->back,
                                  0, mpi_comm, &reqs[1]);
                    MPI_Waitall(2, reqs, stats);
                }
                else
                {
                    /* Sending data to back */
                    reqs[0] = MPI_REQUEST_NULL;
                    reqs[1] = MPI_REQUEST_NULL;
                    if ((myid != levelAbove.mpiData->back) && (MPI_PROC_NULL != levelAbove.mpiData->back))
                        MPI_Isend((void *)&levelAbove.mpiData->front, 1, MPI_INT,
                                  levelAbove.mpiData->back, 0, mpi_comm, &reqs[0]);
                    if ((myid != levelAbove.mpiData->front) && (MPI_PROC_NULL != levelAbove.mpiData->front))
                        MPI_Irecv((void *)&tmpbuf, 1, MPI_INT, levelAbove.mpiData->front,
                                  0, mpi_comm, &reqs[1]);
                    MPI_Waitall(2, reqs, stats);

                    /* Sending data to front */
                    reqs[0] = MPI_REQUEST_NULL;
                    reqs[1] = MPI_REQUEST_NULL;
                    if ((myid != levelAbove.mpiData->front) && (MPI_PROC_NULL != levelAbove.mpiData->front))
                        MPI_Isend((void *)&levelAbove.mpiData->back, 1, MPI_INT,
                                  levelAbove.mpiData->front, 0, mpi_comm, &reqs[0]);
                    if ((myid != levelAbove.mpiData->back) && (MPI_PROC_NULL != levelAbove.mpiData->back))
                        MPI_Irecv((void *)&tmpbuf, 1, MPI_INT, levelAbove.mpiData->back,
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

    void Level::setV(const std::shared_ptr<Cuboid> &v_)
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

    std::shared_ptr<VaryingStencilGpu> &Level::getStencilValuesGpu()
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

    std::shared_ptr<VaryingStencil3x3x3> &Level::getStencilValues()
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

    Cuboid &Level::getV() const
    {
        return *v;
    }

    std::shared_ptr<Cuboid> Level::getVPtr() const
    {
        return v;
    }

    void Level::setF(const std::shared_ptr<Cuboid> &f_)
    {
        f = f_;
    }

    Cuboid &Level::getF() const
    {
        return *f;
    }

    std::shared_ptr<Cuboid> Level::getFPtr() const
    {
        return f;
    }

    void Level::setR(const std::shared_ptr<Cuboid> &r_)
    {
        r = r_;
    }

    Cuboid &Level::getR() const
    {
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

    cl_mem Level::getDVIn() const
    {
        return dVIn;
    }

    void Level::setDVIn(const cl_mem dVIn_)
    {
        dVIn = dVIn_;
        if (dVIn_)
            mgclCheckError(clRetainMemObject(dVIn_), "clRetainMemObject(dVIn)");
    }

    cl_mem Level::getDF() const
    {
        return dF;
    }

    void Level::setDF(const cl_mem dF_)
    {
        dF = dF_;
        if (dF_)
            mgclCheckError(clRetainMemObject(dF_), "clRetainMemObject(dF)");
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

    cl_mem Level::getDVOut() const
    {
        return dVOut;
    }

    void Level::setDVOut(const cl_mem dVOut_)
    {
        dVOut = dVOut_;
        if (dVOut_)
            mgclCheckError(clRetainMemObject(dVOut_), "clRetainMemObject(dVOut)");
    }

    cl_mem Level::getDR() const
    {
        return dR;
    }

    void Level::setDR(const cl_mem dR_)
    {
        dR = dR_;
        if (dR_)
            mgclCheckError(clRetainMemObject(dR_), "clRetainMemObject(dR)");
    }

    MPIData *Level::getMpiDataPtr()
    {
        return mpiData.get();
    }

    MPIData &Level::getMpiData()
    {
        if (mpiData != nullptr)
            return *mpiData;
        else
            throw "mpiData is null.";
    }
}
