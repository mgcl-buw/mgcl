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
     * @brief Construct a new Level:: Level object. m, n and o must be dims of real grid.
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
          h(1.0 / (double)m),
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

        // TODO init mpiData
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
                    MultigridEngine::updateGhostsSeq(getF());
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
                m, n, o, 3, 2, problem->getContext(), problem->getCommands());

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
                                            problem->ghosts);
        mgclCheckError(err, "Updating ghosts of d_f");

        return CL_SUCCESS;
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

    void Level::setDVIn(const cl_mem &dVIn_)
    {
        dVIn = dVIn_;
        if (dVIn_)
            mgclCheckError(clRetainMemObject(dVIn_), "clRetainMemObject(dVIn)");
    }

    cl_mem Level::getDF() const
    {
        return dF;
    }

    void Level::setDF(const cl_mem &dF_)
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

    void Level::setDVOut(const cl_mem &dVOut_)
    {
        dVOut = dVOut_;
        if (dVOut_)
            mgclCheckError(clRetainMemObject(dVOut_), "clRetainMemObject(dVOut)");
    }

    cl_mem Level::getDR() const
    {
        return dR;
    }

    void Level::setDR(const cl_mem &dR_)
    {
        dR = dR_;
        if (dR_)
            mgclCheckError(clRetainMemObject(dR_), "clRetainMemObject(dR)");
    }
}
