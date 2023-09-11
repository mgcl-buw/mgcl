#ifndef MGCL_MULTIGRID_ENGINE_HPP
#define MGCL_MULTIGRID_ENGINE_HPP

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

#include "mgcl.hpp"
#include "problem.hpp"
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
        static int correctError(Problem& problem, cl_mem d_v, cl_mem d_r, int m, int n, int o);

        static void restrictSeq(Level& fine, Level& coarse, Cuboid& fineVals, Cuboid& coarseVals);
        static void restrict(Level& fine, Level& coarse, cl_mem d_fine_values, cl_mem d_coarse_values);

        static void prolongateSeq(Level& fine, Level& coarse, Cuboid& fineVals, Cuboid& coarseVals);
        static void prolongate(Level& fine, Level& coarse, cl_mem d_fine_values, cl_mem d_coarse_values);

        static void updateGhostsSeq(Cuboid& c, MPILevelData* mpiData, bool periodic, bool forceLocal);
        static int updateGhosts(Problem& problem, cl_mem dBuffer, int m, int n, int o,
                                int ghostsM, int ghostsN, int ghostsO, MPILevelData* mpiData, bool forceLocal);
        static void updateGhostsOclMpi(cl_command_queue commands, cl_mem d_buf, MPILevelData& mpiData,
                                       int m, int n, int o, int ghosts_m, int ghosts_n, int ghosts_o,
                                       bool periodic, bool forceLocal);

        static double residual(Problem& problem, Level& level, bool returnResidual,
                               int moff = 0, int noff = 0, int ooff = 0);
        static double residualSeq(Cuboid& f, Cuboid& v, Cuboid& r, MGCL_RESIDUAL_NORM resnorm,
                                  MGCL_STENCIL stencilType, double stencilFactor, VaryingStencil3x3x3* stencilValues,
                                  bool returnResidualNorm, bool periodic, bool updateGhostsLocally,
                                  int moff = 0, int noff = 0, int ooff = 0, MPILevelData* mpiData = nullptr);

        static double jacobiSeq(Cuboid& v, Cuboid& f, Cuboid& r, double omega, double h2,
                                int maxiter, MGCL_RESIDUAL_NORM resnorm, MGCL_STENCIL stencilType, double stencilFactor,
                                VaryingStencil3x3x3* stencilValuesCuboid, bool returnResidualNorm, bool periodic, bool updateGhostsLocally,
                                int stepsPerIter = 1, MPILevelData* mpiData = nullptr);
        static double jacobi(Problem& problem, Level& level, int maxiter, bool returnResidual, int stepsPerIter = 1);
        static double jacobiLocalMem(Problem& problem, Level& level, int maxiter, int returnResidual);

        static VaryingStencil3x3x3 galerkin(VaryingStencil3x3x3& a_h,
                                            MPILevelData* mpiDataFine, MPILevelData* mpiDataCoarse,
                                            bool periodic, bool forceLocalFine, bool forceLocalCoarse,
                                            int resm = 0, int resn = 0, int reso = 0);
        static VaryingStencilGpu galerkin(VaryingStencilGpu& a_h, cl_program program, cl_command_queue queue, cl_context context);

        static void print7point(Cuboid& v, int i, int j, int k);
        static void print19point(Cuboid& v, int i, int j, int k);
        static void print27point(Cuboid& v, int i, int j, int k);
        static void print27point_sv(Cuboid& v, int i, int j, int k,
                                    VaryingStencil3x3x3& sv, int i_sv, int j_sv, int k_sv);
    };
}

#endif // !MGCL_MULTIGRID_ENGINE_HPP
