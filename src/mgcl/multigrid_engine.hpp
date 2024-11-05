#ifndef MGCL_MULTIGRID_ENGINE_HPP
#define MGCL_MULTIGRID_ENGINE_HPP

#include "kernel_config.hpp"
#ifndef CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#endif // CL_USE_DEPRECATED_OPENCL_1_2_APIS

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif // CL_TARGET_OPENCL_VERSION

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "cuboid_gpu.hpp"
#include "mgcl.hpp"
#include "problem.hpp"
#include "profiling_data.hpp"
#include "stencil.hpp"

namespace mgcl
{
    // forward declaration
    class Problem;
    class Level;
    class Cuboid;
    class MPILevelData;

    /**
     * @brief Encapsulates all relevant methods that execute the logic of the multigrid method.
     *
     */
    class MultigridEngine
    {
    public:
        static double vcycleSeq(Problem& problem, Level& level);
        static double vcycle(Problem& problem, Level& level);
        static int correctError(Level& level);

        static void restrictSeq(Level& fine, Level& coarse, Cuboid& fineVals, Cuboid& coarseVals);
        static void restrict(Level& fine, Level& coarse, CuboidGpu& d_fine_values, CuboidGpu& d_coarse_values);

        static void prolongateSeq(Level& fine, Level& coarse, Cuboid& fineVals, Cuboid& coarseVals);
        static void prolongate(Level& fine, Level& coarse, CuboidGpu& d_fine_values, CuboidGpu& d_coarse_values);

        static void updateGhostsSeq(Cuboid& c, MPILevelData* mpiData, bool periodic, bool forceLocal);
        static int updateGhosts(Problem& problem, CuboidGpu& dBuffer, MPILevelData* mpiData, bool forceLocal);
        static void updateGhostsOclMpi(Problem& p, CuboidGpu& d_buf, MPILevelData& mpiData,
                                       bool periodic, bool forceLocal);

        static double residual(Problem& problem, Level& level, bool returnResidual,
                               int moff = 0, int noff = 0, int ooff = 0);
        static double residualSeq(Cuboid& f, Cuboid& v, Cuboid& r, MGCL_RESIDUAL_NORM resnorm,
                                  MGCL_STENCIL stencilType, double stencilFactor, VaryingStencil* stencilValues,
                                  bool returnResidualNorm, bool periodic, bool updateGhostsLocally,
                                  int moff = 0, int noff = 0, int ooff = 0, MPILevelData* mpiData = nullptr);

        static double jacobiSeq(Cuboid& v, Cuboid& f, Cuboid& r, double omega, double h2,
                                int maxiter, MGCL_RESIDUAL_NORM resnorm, MGCL_STENCIL stencilType, double stencilFactor,
                                VaryingStencil* stencilValuesCuboid, bool returnResidualNorm, bool periodic, bool updateGhostsLocally,
                                int stepsPerIter = 1, MPILevelData* mpiData = nullptr);
        static double jacobi(Problem& problem, Level& level, int maxiter, bool returnResidual, int stepsPerIter = 1);
        static double jacobiLocalMem(Problem& problem, Level& level, int maxiter, int returnResidual);

        static std::unique_ptr<VaryingStencil> galerkinOptimized(VaryingStencil& a_h, int gh_a2h,
                                                                 int resm, int resn, int reso);
        static std::unique_ptr<VaryingStencil> galerkinHandcrafted(VaryingStencil& a_h, int gh_a2h,
                                                                   int resm, int resn, int reso);
        static std::unique_ptr<VaryingStencilGpu> galerkinOptimized(VaryingStencilGpu& a_h, int gh_a2h,
                                                                    int resm, int resn, int reso,
                                                                    cl_program program, cl_command_queue queue, cl_context context,
                                                                    conf::KernelConfig* kernelConfig, ProfilingData* pd);

        static void print7point(Cuboid& v, int i, int j, int k);
        static void print19point(Cuboid& v, int i, int j, int k);
        static void print27point(Cuboid& v, int i, int j, int k);
        static void print27point_sv(Cuboid& v, int i, int j, int k,
                                    VaryingStencil& sv, int i_sv, int j_sv, int k_sv);
    };
}

#endif // !MGCL_MULTIGRID_ENGINE_HPP
