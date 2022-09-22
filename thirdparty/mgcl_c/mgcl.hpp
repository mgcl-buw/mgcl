#pragma once

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <string>

// define mgcl_debug for debugging output. Compile with -D MGCL_DEBUG to enable, e.g. run: make CPPFLAGS="-D MGCL_DEBUG"
// taken from https://stackoverflow.com/questions/1644868/define-macro-for-debug-printing-in-c
#ifdef MGCL_DEBUG
#define mgcl_debug(fmt, ...) printf("%s:%d:%s(): " fmt, __FILE__, __LINE__, __func__, ##__VA_ARGS__)
#else
#define mgcl_debug(fmt, ...) \
    do                       \
    {                        \
    } while (0)
#endif

typedef enum
{
    MGCL_L2,
    MGCL_INF
} MGCL_RESIDUAL_NORM;

typedef enum
{
    MGCL_7POINT,
    MGCL_19POINT,
    MGCL_27POINT, // fixed laplacian stencils
    MGCL_7POINT_VARSYM,
    MGCL_19POINT_VARSYM,
    MGCL_27POINT_VARSYM // varying symmetric stencils
} MGCL_STENCIL;

/* level-dependent mgcl data */
typedef struct
{
    /* solution, right hand side and residual vectors */
    double ***v;
    double ***f;
    double ***r;

    /* grid dimensions of ghosted grid */
    int m, n, o;

    /* spacing of real grid on current level */
    double h;

    /* Stencil values per grid point. Must be given by the user if a varying symmetric stencil shall be used.
     * While mgcl_conf::stencil_values should contain only values for real grid points, mgcl_level_data::stencil_values
     * contains values of ghosted points per level, too.
     * Size depends on stencil type:
     * -  7-point varsym: size = (m + 2*ghosts)*(n + 2*ghosts)*(o + 2*ghosts) * 4
     * - 19-point varsym: size = (m + 2*ghosts)*(n + 2*ghosts)*(o + 2*ghosts) * 7
     * - 27-point varsym: size = (m + 2*ghosts)*(n + 2*ghosts)*(o + 2*ghosts) * 8 */
    double ***stencil_values;

    /* opencl buffers */
    cl_mem d_v_in;
    cl_mem d_v_out;
    cl_mem d_f;
    cl_mem d_r;
    cl_mem d_stencil_values;
} mgcl_level_data;

/* Config that needs to get supplemented when using mgcl */
typedef struct
{
    /* initial solution and right hand side vectors */
    double ***v; /* can be ommited if device buffers are supplied */
    double ***f; /* can be ommited if device buffers are supplied */

    /* Buffers for v and f. Only need to be set if buffers already exist on device and should be reused */
    cl_mem d_v;
    cl_mem d_f;
    cl_mem d_stencil_values;

    /* grid dimensions */
    int m, n, o;

    // TODO check what happens when reuse_buffer and ghosts != ghosts_in
    /* Amount of ghost cells surrounding v and f. If optimized jacobi shall be used it must be greater or equal than
     * max(nu1, nu2). Must fit to buffers ghosts size if reuse_opencl_buffers is set. Defaults to 1. */
    int ghosts;

    /* Amount of ghost cells of input data. Defaults to 0. Only relevant if buffers are not reused. */
    int ghosts_in;

    /* maximum grid level */
    int maxlevel;

    /* maximum v-cycle iteration count */
    int maxiter_vcycles;

    /* pre and post smoothing steps */
    int nu1, nu2;

    /* damping factor */
    double omega;

    /* tolerance */
    double tol;

    /* Type of norm of the residual which will be used as termination criterium */
    MGCL_RESIDUAL_NORM residual_norm;

    /* Type of stencil that will be used in jacobi's method */
    MGCL_STENCIL stencil;

    /* Stencil values per grid point. Must be given by the user if a varying symmetric stencil shall be used.
     * This array is only used as input, mgcl_level_data::stencil_values will be used internally for each level.
     * Size depends on stencil type:
     * -  7-point varsym: size = m*n*o*4
     * - 19-point varsym: size = m*n*o*7
     * - 27-point varsym: size = m*n*o*8 */
    double ***stencil_values;

    /* Size multiplier for non-fixed stencils. Used internally only. */
    int stencil_size_multiplier;

    /* Wether to restrict and prolongate varsym stencil or just keep the values as is on each level. Defaults to true.
     */
    int restrict_prolongate_stencil;

    /* Whether to use opencl or not. Defaults to 0 (not using opencl) */
    int use_opencl;

    /* Whether to reuse opencl buffers, context and commands or not. Defaults to 0 (not reusing opencl buffers).
     * Provided buffers must have size of ghosted grid, e.g. (m+2*ghosts) * (n+2*ghosts) * (o+2*ghosts) */
    int reuse_opencl_buffers;

    /* If true input data from d_v and d_f will be copied to newly created buffers, respecting the nearfield ghost cell
     * amount. Defaults to 0. */
    int copy_buffer_data;

    /* If true, resulting v is read from device to host when mgcl has finished. Defaults to 0 (not reading results). */
    int read_results;

    /* If true, optimized jacobi version is used which calculates multiple iterations in one kernel call using local
     * memory. ghosts must be equal to iteration count per kernel call which is limited by local memory size of the
     * device. Defaults to false. */
    int use_local_memory;

    /* Preferred work-group size in x-dimension for jacobi smoother. Defaults to 16. */
    int jacobi_wg_size_x;

    /* Preferred work-group size in y-dimension for jacobi smoother. Defaults to 16. */
    int jacobi_wg_size_y;

    /* Preferred iterations per jacobi kernel call. Is only used when use_local_memory is true. Defaults to 3.
     * Gets automatically decreased if nu1, nu2 or ghosts are smaller. */
    int jacobi_iterations_per_kernel;

    // if true, no output will be printed to stdout. Defaults to false.
    int silent;

    // if true, tol will be ignored thus all iterations of maxVcycleIters will be done. Defaults to false.
    int ignoreTol;

    /* OpenCL stuff */
    std::string kernel_dir;
    std::string device_name;    /* Use first found device if not set */
    cl_device_type device_type; /* Defaults to CL_DEVICE_TYPE_DEFAULT */
    cl_device_id device_id;     /* must be set if a specific device should be reused */
    cl_context context;         /* must be set if a specific context/device/buffers should be reused */
    cl_command_queue commands;  /* must be set if a specific context/device/buffers should be reused */
    cl_program program;         /* compute program, only for internal purposes */
} mgcl_config;

void mgcl_generate_config(mgcl_config **outconf);
int mgcl_init(mgcl_config *conf, mgcl_level_data **data);
void mgcl_restrict_seq(double ***r_fine, double ***r_coarse, int m, int n, int o, int ghosts);
void mgcl_restrict_test(mgcl_config *conf, double ***fine, double ***coarse, int m, int n, int o, int ghosts);
void mgcl_restrict(mgcl_config *conf, mgcl_level_data *fine, mgcl_level_data *coarse);
void mgcl_prolongate_seq(double ***fine, double ***coarse, int m, int n, int o, int ghosts);
void mgcl_prolongate_test(mgcl_config *conf, double ***fine, double ***coarse, int m, int n, int o, int ghosts);
void mgcl_prolongate(mgcl_config *conf, mgcl_level_data *fine, mgcl_level_data *coarse);
void mgcl_finish(mgcl_config *conf, mgcl_level_data *data);
int mgcl_correct_error(mgcl_config *conf, cl_mem d_v, cl_mem d_r, int m, int n, int o);
int mgcl_read_results(mgcl_config *conf, mgcl_level_data *data);
void mgcl_c_mgcl(mgcl_config *conf);
void mgcl_c_mgcl_seq(mgcl_config *conf);
double mgcl_vcycle_seq(mgcl_config *conf, mgcl_level_data *data, int level);
double mgcl_vcycle(mgcl_config *conf, mgcl_level_data *data, int level);
void mgcl_test_read(mgcl_config *conf, mgcl_level_data *data, int level);
