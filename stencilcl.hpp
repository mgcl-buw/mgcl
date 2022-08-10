#pragma once

#include "mgcl.hpp"

namespace mgcl
{
    void mgcl_stencil_restrict_seq(double ***fine, double ***coarse, int m, int n, int o, int ghosts,
                                   int stencil_size_multiplier);
    void mgcl_stencil_restrict_test(mgcl_config *conf, double ***fine, double ***coarse, int m, int n, int o, int ghosts,
                                    int stencil_size_multiplier);
    void mgcl_stencil_restrict(mgcl_config *conf, mgcl_level_data *fine, mgcl_level_data *coarse);
    void mgcl_stencil_prolongate_seq(double ***fine, double ***coarse, int m, int n, int o, int ghosts,
                                     int stencil_size_multiplier);
    void mgcl_stencil_prolongate_test(mgcl_config *conf, double ***fine, double ***coarse, int m, int n, int o, int ghosts,
                                      int stencil_size_multiplier);
    void mgcl_stencil_prolongate(mgcl_config *conf, mgcl_level_data *fine, mgcl_level_data *coarse);
    double mgcl_stencil_jacobi_seq(double ***v, double ***f, double ***r, int m, int n, int o, int ghosts, double omega,
                                   int maxiter, MGCL_RESIDUAL_NORM resnorm, MGCL_STENCIL stencil, double ***stencil_values,
                                   int stencil_size_multiplier);
    double stencil_residual(double ***f, double ***v, double ***r, int m, int n, int o, int ghosts,
                            MGCL_RESIDUAL_NORM resnorm, MGCL_STENCIL stencil, double ***stencil_values,
                            int stencil_size_multiplier);
    double mgcl_stencil_jacobi(mgcl_config *conf, mgcl_level_data *data, int maxiter, int return_residual);
    double mgcl_stencil_jacobi_local_mem(mgcl_config *conf, mgcl_level_data *data, int maxiter, int return_residual);
    void mgcl_stencil_jacobi_test(mgcl_config *conf, mgcl_level_data *data, double ***v, double ***r, int m, int n, int o,
                                  int maxiter);
    double mgcl_stencil_residual(mgcl_config *conf, mgcl_level_data *data, int return_residual);
}
