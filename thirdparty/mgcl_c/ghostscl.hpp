#pragma once

#include "mgcl.hpp"

void update_ghosts_seq(double ***v, int m, int n, int o, int ghosts_m, int ghosts_n, int ghosts_o);
int mgcl_update_ghosts(mgcl_config *conf, cl_mem d_v, int m, int n, int o, int ghosts_m, int ghosts_n, int ghosts_o);
int mgcl_update_ghosts_test(mgcl_config *conf, double ***v, int m, int n, int o, int ghosts_m, int ghosts_n,
                            int ghosts_o);
