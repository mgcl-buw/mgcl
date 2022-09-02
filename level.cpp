#include "level.hpp"

namespace mgcl
{
    /**
     * @brief Construct a new Level:: Level object. m, n and o must be dims of real grid.
     *
     * @param problem_ Problem this Level belongs to.
     * @param num_ Number of level in the Problem (finest grid is level 0)
     * @param m_ Amount of real grid cells in x-direction.
     * @param n_ Amount of real grid cells in y-direction.
     * @param o_ Amount of real grid cells in z-direction.
     */
    Level::Level(Problem *problem_, int num_, int m_, int n_, int o_)
        : problem(problem_),
          num(num_),
          m(m_),
          n(n_),
          o(o_)
    {
        mgh = m + 2 * problem->getGhosts();
        ngh = n + 2 * problem->getGhosts();
        ogh = o + 2 * problem->getGhosts();
    }

    Level::~Level()
    {
        // release buffer of v_in and f only if it was not reused
        if ((!problem->getReuseOpenclBuffers() && num == 0) || num > 0)
        {
            if (dVIn)
                clReleaseMemObject(dVIn);

            if (dF)
                clReleaseMemObject(dF);
        }

        if (dVOut)
            clReleaseMemObject(dVOut);

        if (dR)
            clReleaseMemObject(dR);
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

        setH(1.0 / (m * m));

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
            if (problem->getReuseOpenclBuffers())
            {
                dVIn = problem->getDV();
                dF = problem->getDF();
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

    int Level::getNgh() const
    {
        return ngh;
    }

    int Level::getOgh() const
    {
        return ogh;
    }

    int Level::getNum() const
    {
        return num;
    }

    Cuboid &Level::getV()
    {
        return *v;
    }

    std::shared_ptr<Cuboid> Level::getVPtr()
    {
        return v;
    }

    void Level::setF(const std::shared_ptr<Cuboid> &f_)
    {
        f = f_;
    }

    Cuboid &Level::getF()
    {
        return *f;
    }

    std::shared_ptr<Cuboid> Level::getFPtr()
    {
        return f;
    }

    void Level::setR(const std::shared_ptr<Cuboid> &r_)
    {
        r = r_;
    }

    Cuboid &Level::getR()
    {
        return *r;
    }

    std::shared_ptr<Cuboid> Level::getRPtr()
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
    }

    cl_mem Level::getDF() const
    {
        return dF;
    }

    void Level::setDF(const cl_mem &dF_)
    {
        dF = dF_;
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
    }

    cl_mem Level::getDR() const
    {
        return dR;
    }

    void Level::setDR(const cl_mem &dR_)
    {
        dR = dR_;
    }
}
