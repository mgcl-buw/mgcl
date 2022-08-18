#include <ctgmath>

#include "clutil.hpp"
#include "cuboid.hpp"
#include "ghostscl.hpp"
#include "problem.hpp"

namespace mgcl
{
    Problem::Problem(int _m, int _n, int _o, Cuboid _f, Cuboid _v)
        : Problem(_m, _n, _o, _f.getData(), _v.getData())
    {
    }

    Problem::Problem(int _m, int _n, int _o, double ***_f, double ***_v)
        : m(_m), n(_n), o(_o), f(_f), v(_v)
    {
    }

    Problem::Problem(int _m, int _n, int _o, cl_mem _d_f, cl_mem _d_v)
        : m(_m), n(_n), o(_o), d_f(_d_f), d_v(_d_v)
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
        // check mandatory config fields
        if ((v == nullptr || f == nullptr) && (d_v == nullptr || d_f == nullptr))
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
             (d_stencil_values == nullptr && reuse_opencl_buffers)) &&
            (stencil == MGCL_7POINT_VARSYM || stencil == MGCL_19POINT_VARSYM ||
             stencil == MGCL_27POINT_VARSYM))
        {
            printf("stencil is set to be varying symmetric but stencil_values is nullptr! Aborting.\n");
            return false;
        }

        return true;
    }

    /**
     * @brief Checks if OpenCL-Parameters are valid (only useful if reuse_opencl_buffers || copy_buffer_data)
     *
     * @return true All good.
     * @return false Somethings nullptr or buffers d_v or d_f have wrong size if reuse_opencl_buffers.
     */
    bool Problem::checkOpenCLParameters()
    {
        if (d_v == nullptr || d_f == nullptr)
        {
            printf("OpenCL buffers d_v and d_f not set but reuse_opencl_buffers or copy_buffer_data specified. "
                   "Aborting.\n");
            return false;
        }

        if (device_id == nullptr)
        {
            printf("reuse_opencl_buffers or copy_buffer_data specified but device ID (mgcl_config.device_id) not set. "
                   "Aborting.\n");
            return false;
        }

        if (commands == nullptr)
        {
            printf("reuse_opencl_buffers or copy_buffer_data specified but command queue (mgcl_config.commands) not "
                   "set. Aborting.\n");
            return false;
        }

        if (context == nullptr)
        {
            printf("reuse_opencl_buffers or copy_buffer_data specified but context (mgcl_config.context) not set. "
                   "Aborting.\n");
            return false;
        }

        // check size of buffers
        if (reuse_opencl_buffers)
        {
            size_t bufsize;
            int sizeNeeded = sizeof(double) * (m + 2 * ghosts) * (n + 2 * ghosts) * (o + 2 * ghosts);
            int err = clGetMemObjectInfo(d_v, CL_MEM_SIZE, sizeof(size_t), &bufsize, nullptr);
            mgclCheckError(err, "Querying buffer size of d_v\n");
            if (bufsize != sizeNeeded)
            {
                printf("OpenCL buffer d_v has wrong size (%ld but need %d)\n", bufsize, sizeNeeded);
                return false;
            }

            err = clGetMemObjectInfo(d_f, CL_MEM_SIZE, sizeof(size_t), &bufsize, nullptr);
            mgclCheckError(err, "Querying buffer size of d_f\n");
            if (bufsize != sizeNeeded)
            {
                printf("OpenCL buffer d_f has wrong size (%ld but need %d)\n", bufsize, sizeNeeded);
                return false;
            }
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

        // check opencl components if device buffers should be reused
        if (reuse_opencl_buffers || copy_buffer_data)
        {
            if (!checkOpenCLParameters())
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

                auto lv = std::make_unique<Level>(level, mg, ng, og);

                // create ghosted arrays for v and f on host if device buffer should not be reused
                if (!reuse_opencl_buffers && !copy_buffer_data)
                {
                    lv->setV(cuboid_alloc(mg, ng, og));
                    lv->setF(cuboid_alloc(mg, ng, og));

                    // copy initial input data from conf into mgcl data struct
                    for (int i = 0; i < m; i++)
                        for (int j = 0; j < n; j++)
                            for (int k = 0; k < o; k++)
                            {
                                lv->getV()[i + ghosts][j + ghosts][k + ghosts] =
                                    v[i + ghosts_in][j + ghosts_in][k + ghosts_in];
                                lv->getF()[i + ghosts][j + ghosts][k + ghosts] =
                                    f[i + ghosts_in][j + ghosts_in][k + ghosts_in];
                            }

                    update_ghosts_seq(lv->getF(), m, n, o, ghosts, ghosts, ghosts);

                    // allocate initial stencil_values, including ghost cells, if varying symmetric stencil shall be used
                    if (stencil_values || d_stencil_values)
                    {
                        lv->setStencilValues(cuboid_alloc(mg, ng, og * stencil_size_multiplier));

                        // copy initial input stencil data from conf into mgcl data struct
                        for (int i = 0; i < m; i++)
                            for (int j = 0; j < n; j++)
                                for (int k = 0; k < o * stencil_size_multiplier; k++)
                                {
                                    lv->getStencilValues()[i + ghosts][j + ghosts]
                                                          [k + ghosts * stencil_size_multiplier] =
                                        stencil_values[i + ghosts_in][j + ghosts_in]
                                                      [k + ghosts_in * stencil_size_multiplier];
                                }

                        update_ghosts_seq(lv->getStencilValues(), m, n, o * stencil_size_multiplier, ghosts,
                                          ghosts, ghosts * stencil_size_multiplier);
                    }
                }

                // r on host is only needed if opencl should not be used
                if (!use_opencl)
                {
                    lv->setR(cuboid_alloc(mg, ng, og));
                }

                levels.push_back(std::move(lv));
            }
            else
            {
                // ghosted sizes of current level's grid
                int mg = (levels.at(level - 1)->getM() - 2 * ghosts) / 2 + 2 * ghosts;
                int ng = (levels.at(level - 1)->getN() - 2 * ghosts) / 2 + 2 * ghosts;
                int og = (levels.at(level - 1)->getO() - 2 * ghosts) / 2 + 2 * ghosts;

                auto lv = std::make_unique<Level>(level, mg, ng, og);

                if (!use_opencl)
                {
                    lv->setV(cuboid_alloc(mg, ng, og));
                    lv->setF(cuboid_alloc(mg, ng, og));
                    lv->setR(cuboid_alloc(mg, ng, og));

                    if (stencil_values != nullptr)
                    {
                        if (restrict_prolongate_stencil)
                            lv->setStencilValues(cuboid_alloc(mg, ng, og * stencil_size_multiplier));
                        else
                            lv->setStencilValues(levels[0]->getStencilValues());
                    }
                }

                levels.push_back(std::move(lv));
            }

            levels.back()->setH(1.0 / (m * m));
        }
        return true;
    }

    /********************************
     * Getters and Setters
     ********************************/

    double ***Problem::getF() const
    {
        return f;
    }

    void Problem::setF(double ***f_)
    {
        f = f_;
    }

    cl_mem Problem::dF() const
    {
        return d_f;
    }

    void Problem::setDF(const cl_mem &dF)
    {
        d_f = dF;
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

    int Problem::stencilSizeMultiplier() const
    {
        return stencil_size_multiplier;
    }

    void Problem::setStencilSizeMultiplier(int stencilSizeMultiplier)
    {
        stencil_size_multiplier = stencilSizeMultiplier;
    }

    bool Problem::reuseOpenclBuffers() const
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

    int Problem::jacobiWgSizeX() const
    {
        return jacobi_wg_size_x;
    }

    void Problem::setJacobiWgSizeX(int jacobiWgSizeX)
    {
        jacobi_wg_size_x = jacobiWgSizeX;
    }

    int Problem::jacobiIterationsPerKernel() const
    {
        return jacobi_iterations_per_kernel;
    }

    void Problem::setJacobiIterationsPerKernel(int jacobiIterationsPerKernel)
    {
        jacobi_iterations_per_kernel = jacobiIterationsPerKernel;
    }

    std::string Problem::getDeviceName() const
    {
        return device_name;
    }

    void Problem::setDeviceName(const std::string &deviceName)
    {
        device_name = deviceName;
    }

    cl_device_id Problem::getDeviceId() const
    {
        return device_id;
    }

    void Problem::setDeviceId(const cl_device_id &deviceId)
    {
        device_id = deviceId;
    }

    cl_command_queue Problem::getCommands() const
    {
        return commands;
    }

    void Problem::setCommands(const cl_command_queue &commands_)
    {
        commands = commands_;
    }

    cl_mem Problem::dStencilValues() const
    {
        return d_stencil_values;
    }

    void Problem::setDStencilValues(const cl_mem &dStencilValues)
    {
        d_stencil_values = dStencilValues;
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

    double ***Problem::stencilValues() const
    {
        return stencil_values;
    }

    void Problem::setStencilValues(double ***stencilValues)
    {
        stencil_values = stencilValues;
    }

    bool Problem::restrictProlongateStencil() const
    {
        return restrict_prolongate_stencil;
    }

    void Problem::setRestrictProlongateStencil(bool restrictProlongateStencil)
    {
        restrict_prolongate_stencil = restrictProlongateStencil;
    }

    bool Problem::copyBufferData() const
    {
        return copy_buffer_data;
    }

    void Problem::setCopyBufferData(bool copyBufferData)
    {
        copy_buffer_data = copyBufferData;
    }

    bool Problem::useLocalMemory() const
    {
        return use_local_memory;
    }

    void Problem::setUseLocalMemory(bool useLocalMemory)
    {
        use_local_memory = useLocalMemory;
    }

    int Problem::jacobiWgSizeY() const
    {
        return jacobi_wg_size_y;
    }

    void Problem::setJacobiWgSizeY(int jacobiWgSizeY)
    {
        jacobi_wg_size_y = jacobiWgSizeY;
    }

    std::string Problem::getKernelDir() const
    {
        return kernel_dir;
    }

    void Problem::setKernelDir(const std::string &kernelDir)
    {
        kernel_dir = kernelDir;
    }

    cl_device_type Problem::getDeviceType() const
    {
        return device_type;
    }

    void Problem::setDeviceType(const cl_device_type &deviceType)
    {
        device_type = deviceType;
    }

    cl_context Problem::getContext() const
    {
        return context;
    }

    void Problem::setContext(const cl_context &context_)
    {
        context = context_;
    }

    bool Problem::useOpencl() const
    {
        return use_opencl;
    }

    void Problem::setUseOpencl(bool useOpencl)
    {
        use_opencl = useOpencl;
    }

    double ***Problem::getV() const
    {
        return v;
    }

    void Problem::setV(double ***v_)
    {
        v = v_;
    }

    cl_mem Problem::dV() const
    {
        return d_v;
    }

    void Problem::setDV(const cl_mem &dV)
    {
        d_v = dV;
    }
}