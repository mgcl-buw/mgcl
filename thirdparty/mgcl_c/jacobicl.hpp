#pragma once

#include "mgcl.hpp"
#include <CL/cl.h>

double mgcl_jacobi_seq(double ***v, double ***f, double ***r, int m, int n, int o, int ghosts, double omega,
                       int maxiter, MGCL_RESIDUAL_NORM resnorm, MGCL_STENCIL stencil);
double mgcl_residual(mgcl_config *conf, mgcl_level_data *data, int return_residual);
double residual(double ***f, double ***v, double ***r, int m, int n, int o, int ghosts, MGCL_RESIDUAL_NORM resnorm,
                MGCL_STENCIL stencil);
double mgcl_residual_test(mgcl_config *conf, double ***v, double ***f, double ***r, int m, int n, int o,
                          int return_residual);
void mgcl_jacobi_test(mgcl_config *conf, double ***v, double ***f, double ***r, int m, int n, int o, int maxiter,
                      int read_results);
double mgcl_jacobi(mgcl_config *conf, mgcl_level_data *data, int maxiter, int return_residual);
double mgcl_jacobi_local_mem(mgcl_config *conf, mgcl_level_data *data, int maxiter, int return_residual);
void print_7point(double ***v, int i, int j, int k);
void print_19point(double ***v, int i, int j, int k);
