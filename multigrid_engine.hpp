#pragma once

#include "level.hpp"
#include "problem.hpp"

namespace mgcl
{

    /**
     * @brief Encapsulates all relevant methods that execute the logic of the multigrid method.
     *
     */
    class MultigridEngine
    {
    public:
        static void restrictSeq(Level &fine, Level &coarse, double ***fineVals, double ***coarseVals);
        static void restrictTest(Level &fine, Level &coarse);
        static void restrict(Level &fine, Level &coarse);

        static void prolongateSeq(Level &fine, Level &coarse, double ***fineVals, double ***coarseVals);
        static void prolongateTest(Level &fine, Level &coarse);
        static void prolongate(Level &fine, Level &coarse);

        static void updateGhostsSeq(double ***v, int m, int n, int o, int ghostsM, int ghostsN, int ghostsO);
        static int updateGhosts(Problem &problem, cl_mem dBuffer, int m, int n, int o, int ghostsM, int ghostsN, int ghostsO);
        static int updateGhostsTest(Problem &problem, double ***v, int m, int n, int o, int ghostsM, int ghostsN,
                                    int ghostsO);

        static double jacobiSeq(double ***v, double ***f, double ***r, int m, int n, int o, int ghosts, double omega,
                                int maxiter, MGCL_RESIDUAL_NORM resnorm, MGCL_STENCIL stencil);
        static double residual(Problem &problem, Level &level, int returnResidual);
        static double residualSeq(double ***f, double ***v, double ***r, int m, int n, int o, int ghosts, MGCL_RESIDUAL_NORM resnorm,
                                  MGCL_STENCIL stencil);
        static double residualTest(Problem &problem, double ***v, double ***f, double ***r, int m, int n, int o,
                                   int returnResidual);
        static void jacobiTest(Problem &problem, double ***v, double ***f, double ***r, int m, int n, int o, int maxiter,
                               int readResults);
        static double jacobi(Problem &problem, Level &level, int maxiter, int returnResidual);
        static double jacobiLocalMem(Problem &problem, Level &level, int maxiter, int returnResidual);
        static void print7point(double ***v, int i, int j, int k);
        static void print19point(double ***v, int i, int j, int k);

        static void stencilRestrictSeq(double ***fine, double ***coarse, int m, int n, int o, int ghosts,
                                       int stencilSizeMultiplier);
        static void stencilRestrictTest(Problem &problem, double ***fine, double ***coarse, int m, int n, int o, int ghosts,
                                        int stencilSizeMultiplier);
        static void stencilRestrict(Problem &problem, Level &fine, Level &coarse);
        static void stencilProlongateSeq(double ***fine, double ***coarse, int m, int n, int o, int ghosts,
                                         int stencilSizeMultiplier);
        static void stencilProlongateTest(Problem &problem, double ***fine, double ***coarse, int m, int n, int o, int ghosts,
                                          int stencilSizeMultiplier);
        static void stencilProlongate(Problem &problem, Level &fine, Level &coarse);
        static double stencilJacobiSeq(double ***v, double ***f, double ***r, int m, int n, int o, int ghosts, double omega,
                                       int maxiter, MGCL_RESIDUAL_NORM resnorm, MGCL_STENCIL stencil, double ***stencilValues,
                                       int stencilSizeMultiplier);
        static double stencilResidual(double ***f, double ***v, double ***r, int m, int n, int o, int ghosts,
                                      MGCL_RESIDUAL_NORM resnorm, MGCL_STENCIL stencil, double ***stencilValues,
                                      int stencilSizeMultiplier);
        static double stencilJacobi(Problem &problem, Level &level, int maxiter, int returnResidual);
        static double stencilJacobiLocalMem(Problem &problem, Level &level, int maxiter, int returnResidual);
        static void stencilJacobiTest(Problem &problem, Level &level, double ***v, double ***r, int m, int n, int o,
                                      int maxiter);
        static double stencilResidual(Problem &problem, Level &level, int returnResidual);
    };
}