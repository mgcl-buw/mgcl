#include "level.hpp"

namespace mgcl
{
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

            if (dStencilValues && (problem->getStencilValues() || problem->getDStencilValues()))
                clReleaseMemObject(dStencilValues);
        }

        if (dVOut)
            clReleaseMemObject(dVOut);

        if (dR)
            clReleaseMemObject(dR);
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
                dStencilValues = problem->getDStencilValues();
            }
            else if (problem->getCopyBufferData())
            {
                dVIn = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                      sizeof(double) * m * n * o, NULL, &err);
                dF = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    sizeof(double) * m * n * o, NULL, &err);
                // TODO stencil_values
                problem->getOpenCLHelper().copyInputBuffers();
            }
            else
            {
                int pointer_flag = deviceType == CL_DEVICE_TYPE_GPU ? CL_MEM_COPY_HOST_PTR : CL_MEM_USE_HOST_PTR;
                dVIn = clCreateBuffer(context, CL_MEM_READ_WRITE | pointer_flag,
                                      sizeof(double) * m * n * o, v[0][0], &err);
                dF = clCreateBuffer(context, CL_MEM_READ_WRITE | pointer_flag,
                                    sizeof(double) * m * n * o, f[0][0], &err);

                // create buffers for stencil values if no fixed stencil shall be used
                if (problem->getStencilValues())
                    dStencilValues =
                        clCreateBuffer(context, CL_MEM_READ_WRITE | pointer_flag,
                                       sizeof(double) * m * n * o * problem->getStencilSizeMultiplier(),
                                       stencil_values[0][0], &err);
            }
        }
        else
        {
            dVIn = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                  sizeof(double) * m * n * o, NULL, &err);
            dF = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                sizeof(double) * m * n * o, NULL, &err);

            if (problem->getStencilValues())
            {
                if (problem->getRestrictProlongateStencil())
                    dStencilValues = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                                    sizeof(double) * m * n *
                                                        o * problem->getStencilSizeMultiplier(),
                                                    NULL, &err);
                else
                    dStencilValues = problem->getLevels()[0]->getDStencilValues();
            }
        }

        dVOut = clCreateBuffer(context, CL_MEM_READ_WRITE,
                               sizeof(double) * m * n * o, NULL, &err);
        dR = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double) * m * n * o,
                            NULL, &err);
        mgclCheckError(err, "Creating device buffers");

        err = MultigridEngine::updateGhosts(*problem, dF, m, n, o, problem->ghosts, problem->ghosts,
                                            problem->ghosts);
        mgclCheckError(err, "Updating ghosts of d_f");

        if (dStencilValues)
        {
            err = MultigridEngine::updateGhosts(*problem, dStencilValues, m, n,
                                                o * problem->getStencilSizeMultiplier(), problem->getGhosts(), problem->getGhosts(),
                                                problem->getGhosts() * problem->getStencilSizeMultiplier());
            mgclCheckError(err, "Updating ghosts of d_stencil_values");
        }

        return CL_SUCCESS;
    }
}
