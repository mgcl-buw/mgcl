/**
 * This file contains the variable for the kernel source, which gets filled
 * by cmake at configure time every time the content of mgcl.cl changes.
 * This file itself is not modified but rather serves as a template for inserting
 * the kernel code. The generated file is placed alongside
 * and named 'mgcl_kernel.hpp'.
 *
 */

#include <string>

namespace mgcl
{
    const std::string MGCL_KERNEL_SOURCE = R"DELIM(#ifndef NULL
#define NULL 0
#endif

/* Prints components of 7-point laplacian stencil for debugging purposes */
void print_7point(__global double* A, int index, int ioff, int joff, int koff)
{
    printf("7point stencil at %d:\n", index);
    printf("v[self] = %.17e\n", A[index]);
    printf(" v[k-1] = %.17e\n", A[index - koff]);
    printf(" v[k+1] = %.17e\n", A[index + koff]);
    printf(" v[j-1] = %.17e\n", A[index - joff]);
    printf(" v[j+1] = %.17e\n", A[index + joff]);
    printf(" v[i-1] = %.17e\n", A[index - ioff]);
    printf(" v[i+1] = %.17e\n", A[index + ioff]);
}

void print_7point_local(__local double* A, __local double* rm, __local double* rp, int index, int ioff, int joff,
                        int koff)
{
    printf("7point stencil at %d:\n", index);
    printf("v[self] = %e\n", A[index]);
    printf(" v[k-1] = %e\n", A[index - koff]);
    printf(" v[k+1] = %e\n", A[index + koff]);
    printf(" v[j-1] = %e\n", A[index - joff]);
    printf(" v[j+1] = %e\n", A[index + joff]);
    printf(" v[i-1] = %e\n", rm == NULL ? A[index - ioff] : rm[index]);
    printf(" v[i+1] = %e\n", rp == NULL ? A[index + ioff] : rp[index]);
}

void print_19point_local(__local double* A, __local double* rm, __local double* rp, int index, int ioff, int joff,
                         int koff)
{
    printf("19point stencil at %d:\n", index);
    printf("v[self] = %e\n", A[index]);
    printf(" v[ i ][ j ][k-1] = %e\n", A[index - koff]);
    printf(" v[ i ][ j ][k+1] = %e\n", A[index + koff]);
    printf(" v[ i ][j-1][ k ] = %e\n", A[index - joff]);
    printf(" v[ i ][j+1][ k ] = %e\n", A[index + joff]);
    printf(" v[i-1][ j ][ k ] = %e\n", rm == NULL ? A[index - ioff] : rm[index]);
    printf(" v[i+1][ j ][ k ] = %e\n", rp == NULL ? A[index + ioff] : rp[index]);
    printf(" v[ i ][j-1][k-1] = %e\n", A[index - joff - koff]);
    printf(" v[ i ][j-1][k+1] = %e\n", A[index - joff + koff]);
    printf(" v[ i ][j+1][k-1] = %e\n", A[index + joff - koff]);
    printf(" v[ i ][j+1][k+1] = %e\n", A[index + joff + koff]);
    printf(" v[i-1][ j ][k-1] = %e\n", rm == NULL ? A[index - ioff - 1] : rm[index - 1]);
    printf(" v[i-1][ j ][k+1] = %e\n", rm == NULL ? A[index - ioff + 1] : rm[index + 1]);
    printf(" v[i+1][ j ][k-1] = %e\n", rp == NULL ? A[index + ioff - 1] : rp[index - 1]);
    printf(" v[i+1][ j ][k+1] = %e\n", rp == NULL ? A[index + ioff + 1] : rp[index + 1]);
    printf(" v[i-1][j-1][ k ] = %e\n", rm == NULL ? A[index - ioff - joff] : rm[index - joff]);
    printf(" v[i-1][j+1][ k ] = %e\n", rm == NULL ? A[index - ioff + joff] : rm[index + joff]);
    printf(" v[i+1][j-1][ k ] = %e\n", rp == NULL ? A[index + ioff - joff] : rp[index - joff]);
    printf(" v[i+1][j+1][ k ] = %e\n", rp == NULL ? A[index + ioff + joff] : rp[index + joff]);
}

void print27point(__global double* v, int index, int ioff, int joff, int koff)
{
    printf("27point stencil at %d:\n", index);
    printf("v[self] = %e\n", v[index]);
    printf(" v[ i ][ j ][k-1] = %e\n", v[index - koff]);
    printf(" v[ i ][ j ][k+1] = %e\n", v[index + koff]);
    printf(" v[ i ][j-1][ k ] = %e\n", v[index - joff]);
    printf(" v[ i ][j+1][ k ] = %e\n", v[index + joff]);
    printf(" v[i-1][ j ][ k ] = %e\n", v[index - ioff]);
    printf(" v[i+1][ j ][ k ] = %e\n", v[index + ioff]);
    printf(" v[ i ][j-1][k-1] = %e\n", v[index - joff - koff]);
    printf(" v[ i ][j-1][k+1] = %e\n", v[index - joff + koff]);
    printf(" v[ i ][j+1][k-1] = %e\n", v[index + joff - koff]);
    printf(" v[ i ][j+1][k+1] = %e\n", v[index + joff + koff]);
    printf(" v[i-1][ j ][k-1] = %e\n", v[index - ioff - koff]);
    printf(" v[i-1][ j ][k+1] = %e\n", v[index - ioff + koff]);
    printf(" v[i+1][ j ][k-1] = %e\n", v[index + ioff - koff]);
    printf(" v[i+1][ j ][k+1] = %e\n", v[index + ioff + koff]);
    printf(" v[i-1][j-1][ k ] = %e\n", v[index - ioff - joff]);
    printf(" v[i-1][j+1][ k ] = %e\n", v[index - ioff + joff]);
    printf(" v[i+1][j-1][ k ] = %e\n", v[index + ioff - joff]);
    printf(" v[i+1][j+1][ k ] = %e\n", v[index + ioff + joff]);
    printf(" v[i-1][j-1][k-1] = %e\n", v[index - ioff - joff - koff]);
    printf(" v[i-1][j-1][k+1] = %e\n", v[index - ioff - joff + koff]);
    printf(" v[i-1][j+1][k-1] = %e\n", v[index - ioff + joff - koff]);
    printf(" v[i-1][j+1][k+1] = %e\n", v[index - ioff + joff + koff]);
    printf(" v[i+1][j-1][k-1] = %e\n", v[index + ioff - joff - koff]);
    printf(" v[i+1][j-1][k+1] = %e\n", v[index + ioff - joff + koff]);
    printf(" v[i+1][j+1][k-1] = %e\n", v[index + ioff + joff - koff]);
    printf(" v[i+1][j+1][k+1] = %e\n", v[index + ioff + joff + koff]);
}

void print27point_sv(__global double* v, int index, int ioff, int joff, int koff,
                     __global double* sv, int index_sv)
{
    printf("27point stencil at %d,%d:\n", index_sv, index);
    printf(" sv * v[    self     ] = %e * %e\n", sv[index_sv + 9 + 3 + 1], v[index]);
    printf(" sv * v[ i ][ j ][k-1] = %e * %e\n", sv[index_sv + 9 + 3], v[index - koff]);
    printf(" sv * v[ i ][ j ][k+1] = %e * %e\n", sv[index_sv + 9 + 3 + 2], v[index + koff]);
    printf(" sv * v[ i ][j-1][ k ] = %e * %e\n", sv[index_sv + 9 + 1], v[index - joff]);
    printf(" sv * v[ i ][j+1][ k ] = %e * %e\n", sv[index_sv + 9 + 6 + 1], v[index + joff]);
    printf(" sv * v[i-1][ j ][ k ] = %e * %e\n", sv[index_sv + 3 + 1], v[index - ioff]);
    printf(" sv * v[i+1][ j ][ k ] = %e * %e\n", sv[index_sv + 18 + 3 + 1], v[index + ioff]);
    printf(" sv * v[ i ][j-1][k-1] = %e * %e\n", sv[index_sv + 9], v[index - joff - koff]);
    printf(" sv * v[ i ][j-1][k+1] = %e * %e\n", sv[index_sv + 9 + 2], v[index - joff + koff]);
    printf(" sv * v[ i ][j+1][k-1] = %e * %e\n", sv[index_sv + 9 + 6], v[index + joff - koff]);
    printf(" sv * v[ i ][j+1][k+1] = %e * %e\n", sv[index_sv + 9 + 6 + 2], v[index + joff + koff]);
    printf(" sv * v[i-1][ j ][k-1] = %e * %e\n", sv[index_sv + 3], v[index - ioff - koff]);
    printf(" sv * v[i-1][ j ][k+1] = %e * %e\n", sv[index_sv + 3 + 2], v[index - ioff + koff]);
    printf(" sv * v[i+1][ j ][k-1] = %e * %e\n", sv[index_sv + 18 + 3], v[index + ioff - koff]);
    printf(" sv * v[i+1][ j ][k+1] = %e * %e\n", sv[index_sv + 18 + 3 + 2], v[index + ioff + koff]);
    printf(" sv * v[i-1][j-1][ k ] = %e * %e\n", sv[index_sv + 1], v[index - ioff - joff]);
    printf(" sv * v[i-1][j+1][ k ] = %e * %e\n", sv[index_sv + 6 + 1], v[index - ioff + joff]);
    printf(" sv * v[i+1][j-1][ k ] = %e * %e\n", sv[index_sv + 18 + 1], v[index + ioff - joff]);
    printf(" sv * v[i+1][j+1][ k ] = %e * %e\n", sv[index_sv + 18 + 6 + 1], v[index + ioff + joff]);
    printf(" sv * v[i-1][j-1][k-1] = %e * %e\n", sv[index_sv], v[index - ioff - joff - koff]);
    printf(" sv * v[i-1][j-1][k+1] = %e * %e\n", sv[index_sv + 2], v[index - ioff - joff + koff]);
    printf(" sv * v[i-1][j+1][k-1] = %e * %e\n", sv[index_sv + 6], v[index - ioff + joff - koff]);
    printf(" sv * v[i-1][j+1][k+1] = %e * %e\n", sv[index_sv + 6 + 2], v[index - ioff + joff + koff]);
    printf(" sv * v[i+1][j-1][k-1] = %e * %e\n", sv[index_sv + 18], v[index + ioff - joff - koff]);
    printf(" sv * v[i+1][j-1][k+1] = %e * %e\n", sv[index_sv + 18 + 2], v[index + ioff - joff + koff]);
    printf(" sv * v[i+1][j+1][k-1] = %e * %e\n", sv[index_sv + 18 + 6], v[index + ioff + joff - koff]);
    printf(" sv * v[i+1][j+1][k+1] = %e * %e\n", sv[index_sv + 18 + 6 + 2], v[index + ioff + joff + koff]);
}

/**
 * Updates ghosts of a cuboid, respecting small grids, e.g. gh > m.
 * Needs to be called with one work-item per cell of ghosted grid.
 * Work-items that map to a real cell simply do nothing (optimization potential here!).
 * m,n,o are sizes of real grid.
 * ghm, ghn, gho are amount of ghosts at one border.
 */
__kernel void update_ghosts_periodic(
    __global double* restrict c,
    int m, int n, int o,
    int ghm, int ghn, int gho)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);

    int mgh = m + 2 * ghm;
    int ngh = n + 2 * ghn;
    int ogh = o + 2 * gho;

    if ((i < ghm || j < ghn || k < gho ||
         i >= ghm + m || j >= ghn + n || k >= gho + o) &&
        (i < mgh && j < ngh && k < ogh))
    {
        int ireal = i + floor(((double)(ghm - 1 - i)) / m + 1) * m;
        int jreal = j + floor(((double)(ghn - 1 - j)) / n + 1) * n;
        int kreal = k + floor(((double)(gho - 1 - k)) / o + 1) * o;

        // 1d indices
        int idx_gh_cell = i * ngh * ogh + j * ogh + k;
        int idx_real_cell = ireal * ngh * ogh + jreal * ogh + kreal;

        // update ghost cell
        c[idx_gh_cell] = c[idx_real_cell];
    }
}

/* Copies data from v_input to v_in and from f_input to f, respecting nearfield ghost cell count.
 * m, n and o are dimensions of mgcl's ghosted grid, thus sizes of v_in and f.
 * ghosts_in is ghost cell count in one direction of nearfield. */
__kernel void copy_input_data(__global double* v_input, __global double* v_in, __global double* f_input,
                              __global double* f, const int m, const int n, const int o, const int ghosts_in)
{
    // index of ghosted mgcl grid
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);
    int index = i * n * o + j * o + k;

    // index of input grid cell
    int index_input = (i + ghosts_in - 1) * (n - 2 + 2 * ghosts_in) * (o - 2 + 2 * ghosts_in) +
                      (j + ghosts_in - 1) * (o - 2 + 2 * ghosts_in) + (k + ghosts_in - 1);

    if (i < m && j < n && k < o)
    {
        v_in[index] = v_input[index_input];
        f[index] = f_input[index_input];
    }
}

/* Copies data from v_in to v_output, respecting nearfield ghost cell count.
 * m, n and o are dimensions of mgcl's ghosted grid, thus size of v_in.
 * ghosts_in is ghost cell count in one direction of nearfield. */
__kernel void copy_output_data(__global double* v_output, __global double* v_in, const int m, const int n, const int o,
                               const int ghosts_in)
{
    // index of ghosted mgcl grid
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);
    int index = i * n * o + j * o + k;

    // index of input grid cell
    int index_output = (i + ghosts_in - 1) * (n - 2 + 2 * ghosts_in) * (o - 2 + 2 * ghosts_in) +
                       (j + ghosts_in - 1) * (o - 2 + 2 * ghosts_in) + (k + ghosts_in - 1);

    if (i < m && j < n && k < o)
    {
        v_output[index_output] = v_in[index];
    }
}

/* Corrects error by adding e to v, whereas both have the same size.
 * 3d kernel that needs to be started with m x n x o work-items.
 * m, n and o are dimensions of ghosted grid. */
__kernel void correct_error(
    __global double* restrict v,
    __global double* restrict e,
    const int m, const int n, const int o,
    const int ghosts)
{
    int i = get_global_id(0) + ghosts;
    int j = get_global_id(1) + ghosts;
    int k = get_global_id(2) + ghosts;
    int idx = i * n * o + j * o + k;

    // only for real cells
    if (i >= ghosts && j >= ghosts && k >= ghosts && i < m - ghosts && j < n - ghosts && k < o - ghosts)
    {
        v[idx] += e[idx];
    }
}

/* Calculates residual without dinv */
__kernel void residual_7point(
    __global double* restrict v_in, // needed s.t. every work-item can read surrounding cell values
    __global double* restrict f,
    __global double* restrict r,
    const double h2inv, const int mgh, const int ngh, const int ogh,
    const int ghosts, const int moff, const int noff, const int ooff)
{
    int idx = get_global_id(0);
    int no = ngh * ogh;
    int i = idx / no;
    int j = (idx - i * no) / ogh;
    int k = idx % ogh;

    // loop boundaries
    // TODO maybe refactor to use v_ghm, etc.?
    int istart_v = ghosts + moff;
    int jstart_v = ghosts + noff;
    int kstart_v = ghosts + ooff;
    int iend_v = mgh - ghosts - moff;
    int jend_v = ngh - ghosts - noff;
    int kend_v = ogh - ghosts - ooff;

    // calculate residual only for relevant cells (off = 0: only real cells)
    if (i >= istart_v && j >= jstart_v && k >= kstart_v && i < iend_v && j < jend_v && k < kend_v)
    {
        int index = i * no + j * ogh + k;

        // TODO refactor like in 27point
        // A*v
        double stencilsum = (6.0 * v_in[index] - v_in[index - 1] - v_in[index + 1] - v_in[i * no + (j - 1) * ogh + k] -
                             v_in[i * no + (j + 1) * ogh + k] - v_in[(i - 1) * no + j * ogh + k] -
                             v_in[(i + 1) * no + j * ogh + k]) *
                            h2inv;

        // if (i == 1 && j == 1 && k == 2)
        //     printf("stencilsum = %e\n", stencilsum);
        // if (i == 1 && j == 1 && k == 2 && m > 4)
        //     print_7point(v_in, m, n, o);
        // r = f - A*v
        r[index] = f[index] - stencilsum;
    }
}

/* Calculates residual without dinv */
__kernel void residual_19point(
    __global double* restrict v_in, // needed s.t. every work-item can read surrounding cell values
    __global double* restrict f,
    __global double* restrict r,
    const double h2inv, const int mgh, const int ngh, const int ogh,
    const int ghosts, const int moff, const int noff, const int ooff)
{
    int idx = get_global_id(0);
    int no = ngh * ogh;
    int i = idx / no;
    int j = (idx - i * no) / ogh;
    int k = idx % ogh;

    // loop boundaries
    // TODO maybe refactor to use v_ghm, etc.?
    int istart_v = ghosts + moff;
    int jstart_v = ghosts + noff;
    int kstart_v = ghosts + ooff;
    int iend_v = mgh - ghosts - moff;
    int jend_v = ngh - ghosts - noff;
    int kend_v = ogh - ghosts - ooff;

    // calculate residual only for relevant cells (off = 0: only real cells)
    if (i >= istart_v && j >= jstart_v && k >= kstart_v && i < iend_v && j < jend_v && k < kend_v)
    {
        int index = i * no + j * ogh + k;

        // TODO refactor like in 27point
        // A*v
        double stencilsum = (24.0 * v_in[index] - 2.0 * v_in[index - 1] - 2.0 * v_in[index + 1] -
                             2.0 * v_in[i * no + (j - 1) * ogh + k] - 2.0 * v_in[i * no + (j + 1) * ogh + k] -
                             2.0 * v_in[(i - 1) * no + j * ogh + k] - 2.0 * v_in[(i + 1) * no + j * ogh + k] -
                             v_in[i * no + (j - 1) * ogh + k - 1] - v_in[i * no + (j - 1) * ogh + k + 1] -
                             v_in[i * no + (j + 1) * ogh + k - 1] - v_in[i * no + (j + 1) * ogh + k + 1] -
                             v_in[(i - 1) * no + j * ogh + k - 1] - v_in[(i - 1) * no + j * ogh + k + 1] -
                             v_in[(i + 1) * no + j * ogh + k - 1] - v_in[(i + 1) * no + j * ogh + k + 1] -
                             v_in[(i - 1) * no + (j - 1) * ogh + k] - v_in[(i - 1) * no + (j + 1) * ogh + k] -
                             v_in[(i + 1) * no + (j - 1) * ogh + k] - v_in[(i + 1) * no + (j + 1) * ogh + k]) *
                            h2inv;

        // if (i == 1 && j == 1 && k == 2)
        //     printf("stencilsum = %e\n", stencilsum);
        // if (i == 1 && j == 1 && k == 2 && m > 4)
        //     print_7point(v_in, m, n, o);
        // r = f - A*v
        r[index] = f[index] - stencilsum;
    }
}

/* Calculates residual without dinv */
__kernel void residual_27point(
    __global double* restrict v_in, // needed s.t. every work-item can read surrounding cell values
    __global double* restrict f,
    __global double* restrict r,
    const double h2inv, const int mgh, const int ngh, const int ogh,
    const int ghosts, const int moff, const int noff, const int ooff)
{
    int idx = get_global_id(0);
    int no = ngh * ogh;
    int i = idx / no;
    int j = (idx - i * no) / ogh;
    int k = idx % ogh;

    // loop boundaries
    // TODO maybe refactor to use v_ghm, etc.?
    int istart_v = ghosts + moff;
    int jstart_v = ghosts + noff;
    int kstart_v = ghosts + ooff;
    int iend_v = mgh - ghosts - moff;
    int jend_v = ngh - ghosts - noff;
    int kend_v = ogh - ghosts - ooff;
    // int istart_r = r.getGhostsM() + moff;
    // int jstart_r = r.getGhostsN() + noff;
    // int kstart_r = r.getGhostsO() + ooff;
    // int istart_f = f.getGhostsM() + moff;
    // int jstart_f = f.getGhostsN() + noff;
    // int kstart_f = f.getGhostsO() + ooff;

    // calculate residual only for relevant cells (off = 0: only real cells)
    if (i >= istart_v && j >= jstart_v && k >= kstart_v && i < iend_v && j < jend_v && k < kend_v)
    {
        int index = i * ngh * ogh + j * ogh + k;
        int ioff = ngh * ogh;
        int joff = ogh;
        int koff = 1;

        // A*v
        double stencilsum =
            (128.0 * v_in[index] - 14.0 * v_in[index - koff] - 14.0 * v_in[index + koff] - 14.0 * v_in[index - joff] -
             14.0 * v_in[index + joff] - 14.0 * v_in[index - ioff] - 14.0 * v_in[index + ioff]

             - 3.0 * v_in[index - joff - koff] - 3.0 * v_in[index - joff + koff] - 3.0 * v_in[index + joff - koff] -
             3.0 * v_in[index + joff + koff] - 3.0 * v_in[index - ioff - koff] - 3.0 * v_in[index - ioff + koff] -
             3.0 * v_in[index + ioff - koff] - 3.0 * v_in[index + ioff + koff] - 3.0 * v_in[index - ioff - joff] -
             3.0 * v_in[index - ioff + joff] - 3.0 * v_in[index + ioff - joff] - 3.0 * v_in[index + ioff + joff]

             - v_in[index - ioff - joff - koff] - v_in[index - ioff - joff + koff] - v_in[index - ioff + joff - koff] -
             v_in[index - ioff + joff + koff] - v_in[index + ioff - joff - koff] - v_in[index + ioff - joff + koff] -
             v_in[index + ioff + joff - koff] - v_in[index + ioff + joff + koff]) *
            h2inv;

        // if (i == 1 && j == 1 && k == 2)
        //     printf("stencilsum = %e\n", stencilsum);
        // if (i == 1 && j == 1 && k == 2 && m > 4)
        //     print_7point(v_in, m, n, o);
        // r = f - A*v
        r[index] = f[index] - stencilsum;
    }
}

/* Calculates residual without dinv */
__kernel void residual_27point_varying_stencil(
    __global double* restrict v_in, // needed s.t. every work-item can read surrounding cell values
    __global double* restrict f,
    __global double* restrict r,
    __global double* restrict stencilValues,
    const int mgh, const int ngh, const int ogh,
    const int svmgh, const int svngh, const int svogh,
    const int ghosts, const int ghosts_sv,
    const int svGridSize,
    const int moff, const int noff, const int ooff)

{
    int idx = get_global_id(0);
    int no = ngh * ogh;
    int i = idx / no;
    int j = (idx - i * no) / ogh;
    int k = idx % ogh;

    // loop boundaries
    // TODO maybe refactor to use v_ghm, etc.?
    int istart_v = ghosts + moff;
    int jstart_v = ghosts + noff;
    int kstart_v = ghosts + ooff;
    int iend_v = mgh - ghosts - moff;
    int jend_v = ngh - ghosts - noff;
    int kend_v = ogh - ghosts - ooff;

    // calculate residual only for relevant cells (off = 0: only real cells)
    if (i >= istart_v && j >= jstart_v && k >= kstart_v && i < iend_v && j < jend_v && k < kend_v)
    {
        int ioff = ngh * ogh;
        int joff = ogh;
        int koff = 1;
        int index = i * ioff + j * ogh + k;

        int svno = svngh * svogh;
        // offset inside one coefficient grid that points to the coefficient for the current grid point. Must consider different amount of ghosts for v and sv.
        int index_sv = (i - ghosts + ghosts_sv) * svno + (j - ghosts + ghosts_sv) * svogh + (k - ghosts + ghosts_sv);

        // if (i == 2 && j == 2 && k == 2)
        // {
        //     printf("i,j,k,mgh,ngh,ogh,gh,gh_sv,index_sv,gridsize: %d,%d,%d,%d,%d,%d,%d,%d,%d,%d\ngh", i, j, k, mgh, ngh, ogh, ghosts, ghosts_sv, index_sv, gridsize);
        // }

        // A*v
        // clang-format off
        double stencilsum = stencilValues[index_sv + (9 + 3 + 1) * svGridSize] * v_in[index]
            + stencilValues[index_sv + (9 + 3) * svGridSize]      * v_in[index - 1]
            + stencilValues[index_sv + (9 + 3 + 2) * svGridSize]  * v_in[index + 1]
            + stencilValues[index_sv + (9 + 1) * svGridSize]      * v_in[index - joff]
            + stencilValues[index_sv + (9 + 6 + 1) * svGridSize]  * v_in[index + joff]
            + stencilValues[index_sv + (3 + 1) * svGridSize]      * v_in[index - ioff]
            + stencilValues[index_sv + (18 + 3 + 1) * svGridSize] * v_in[index + ioff]
            
            + stencilValues[index_sv + (9) * svGridSize]          * v_in[index - joff - koff]
            + stencilValues[index_sv + (9 + 2) * svGridSize]      * v_in[index - joff + koff]
            + stencilValues[index_sv + (9 + 6) * svGridSize]      * v_in[index + joff - koff]
            + stencilValues[index_sv + (9 + 6 + 2) * svGridSize]  * v_in[index + joff + koff]
            + stencilValues[svGridSize * 3 + index_sv]            * v_in[index - ioff - koff]
            + stencilValues[index_sv + (3 + 2) * svGridSize]      * v_in[index - ioff + koff]
            + stencilValues[index_sv + (18 + 3) * svGridSize]     * v_in[index + ioff - koff]
            + stencilValues[index_sv + (18 + 3 + 2) * svGridSize] * v_in[index + ioff + koff]
            + stencilValues[svGridSize + index_sv]                * v_in[index - ioff - joff]
            + stencilValues[index_sv + (6 + 1) * svGridSize]      * v_in[index - ioff + joff]
            + stencilValues[index_sv + (18 + 1) * svGridSize]     * v_in[index + ioff - joff]
            + stencilValues[index_sv + (18 + 6 + 1) * svGridSize] * v_in[index + ioff + joff]

            + stencilValues[index_sv]                           * v_in[index - ioff - joff - koff]
            + stencilValues[svGridSize * 2 + index_sv]            * v_in[index - ioff - joff + koff]
            + stencilValues[index_sv + (6) * svGridSize]          * v_in[index - ioff + joff - koff]
            + stencilValues[index_sv + (6 + 2) * svGridSize]      * v_in[index - ioff + joff + koff]
            + stencilValues[index_sv + (18) * svGridSize]         * v_in[index + ioff - joff - koff]
            + stencilValues[index_sv + (18 + 2) * svGridSize]     * v_in[index + ioff - joff + koff]
            + stencilValues[index_sv + (18 + 6) * svGridSize]     * v_in[index + ioff + joff - koff]
            + stencilValues[index_sv + (18 + 6 + 2) * svGridSize] * v_in[index + ioff + joff + koff];
        // clang-format on

        // if (i == 2 && j == 2 && k == 2)
        // {
        //     printf("ocl stencilsum = %e\n", stencilsum);
        //     print27point_sv(v_in, index, ioff, joff, koff, stencilValues, index_sv);
        // }

        // r = f - A*v
        r[index] = f[index] - stencilsum;
    }
}

/* Calculates the squares of the residual.
 * r is ghosted, rsquares must be only real cells.
 * m, n and o must be real grid size.
 * ghosts is amount of ghost cells at one border for r.
 * Kernel must be called with m x n x o work-items.
 */
__kernel void residual_squared(
    __global double* restrict r,
    __global double* restrict rsquares,
    const int mgh, const int ngh, const int ogh, const int ghosts)
{
    int idx = get_global_id(0);
    int no = ngh * ogh;
    int i = idx / no;
    int j = (idx - i * no) / ogh;
    int k = idx % ogh;

    // account for padding
    if (i < mgh && j < ngh && k < ogh)
    {
        int index = i * (ngh + 2 * ghosts) * (ogh + 2 * ghosts) + j * (ogh + 2 * ghosts) + k;
        int index_sq = i * ngh * ogh + j * ogh + k;
        double ridx = r[index];
        rsquares[index_sq] = ridx * ridx;
    }
}

/* runs one iteration of jacobi's method using one work-item per row.
 * uses a 2D kernel which loops over cells in x-direction. y and z is parallelized.
 * global size must be of ghosted grid.
 * m, n and o must be dimensions of ghosted grid, too.
 * h2 is grid spacing to the power of 2
 * dinv is h2/A(i,i), e.g. h2/6.0 for 3D laplacian stencil
 * if store_residual is true, the residual will be stored into global field r
 * idx_start determines which cells shall be calculated, which is relevant for running
 *   Jacobi with multiple iterations without ghost cell update in-between. I.e. when
 *   stepsPerIter = 1: idx_start = ghosts.
 */
__kernel void jacobi_iter_7point(
    __global double* restrict v_in, // needed s.t. every work-item can read surrounding cell values
    __global double* restrict v_out,
    __global double* restrict f,
    __global double* restrict r,
    const double h2inv,
    const double dinv, const double omega,
    const int mgh, const int ngh, const int ogh, const int ghosts,
    const int idx_start, const int store_residual)
{
    int j = get_global_id(0);
    int k = get_global_id(1);

    // calculate residual for real cells plus some ghost cells if stepsPerIter > 1.
    if (j >= idx_start && k >= idx_start && j < ngh - idx_start && k < ogh - idx_start)
    {
        int ioff = ngh * ogh;
        int joff = ogh;
        int koff = 1;
        int index = idx_start * ioff + j * ogh + k;

        for (int i = idx_start; i < mgh - idx_start; i++)
        {
            double res;
            double v_in_index = v_in[index];

            // A*v
            double stencilsum = (6.0 * v_in_index - v_in[index - koff] - v_in[index + koff] - v_in[index - joff] -
                                 v_in[index + joff] - v_in[index - ioff] - v_in[index + ioff]) *
                                h2inv;

            // r = f - A*v
            res = f[index] - stencilsum;

            // u_(m+1) = u_(m) + omega * (D^-1) * r_(m)
            v_out[index] = v_in_index + omega * dinv * res;

            // if (j == ghosts && k == ghosts && i >= ghosts && i <= ghosts)
            // {
            //     printf("x,y,z = %d,%d,%d, f = %.17e, stencilsum = %.17e, res = %.17e, v_out = %.17e\n", i, j, k, f[index], stencilsum, res, v_out[index]);
            //     print_7point(v_in, index, ioff, joff, koff);
            // }

            if (store_residual)
                r[index] = res;

            index += ioff;
        }
    }
}

/* runs one iteration of jacobi's method using one work-item per cell.
 * global size must be of ghosted grid.
 * m, n and o must be dimensions of ghosted grid, too.
 * h2 is grid spacing to the power of 2
 * dinv is h2/A(i,i), e.g. h2/6.0 for 3D laplacian stencil
 * if store_residual is true, the residual will be stored into global field r
 * idx_start determines which cells shall be calculated, which is relevant for running
 *   Jacobi with multiple iterations without ghost cell update in-between. I.e. when
 *   stepsPerIter = 1: idx_start = ghosts.
 */
__kernel void jacobi_iter_19point(
    __global double* restrict v_in, // needed s.t. every work-item can read surrounding cell values
    __global double* restrict v_out,
    __global double* restrict f,
    __global double* restrict r,
    const double h2inv,
    const double dinv, const double omega,
    const int mgh, const int ngh, const int ogh, const int ghosts,
    const int idx_start, const int store_residual)
{
    int j = get_global_id(0);
    int k = get_global_id(1);

    // calculate residual for real cells plus some ghost cells if stepsPerIter > 1.
    if (j >= idx_start && k >= idx_start && j < ngh - idx_start && k < ogh - idx_start)
    {
        int ioff = ngh * ogh;
        int joff = ogh;
        int koff = 1;
        int index = idx_start * ioff + j * ogh + k;

        for (int i = idx_start; i < mgh - idx_start; i++)
        {
            double res;
            double v_in_index = v_in[index];

            // A*v 19-point laplacian stencil
            double stencilsum =
                (24.0 * v_in_index - 2.0 * v_in[index - koff] - 2.0 * v_in[index + koff] - 2.0 * v_in[index - joff] -
                 2.0 * v_in[index + joff] - 2.0 * v_in[index - ioff] - 2.0 * v_in[index + ioff] -
                 v_in[index - joff - koff] - v_in[index - joff + koff] - v_in[index + joff - koff] -
                 v_in[index + joff + koff] - v_in[index - ioff - koff] - v_in[index - ioff + koff] -
                 v_in[index + ioff - koff] - v_in[index + ioff + koff] - v_in[index - ioff - joff] -
                 v_in[index - ioff + joff] - v_in[index + ioff - joff] - v_in[index + ioff + joff]) *
                h2inv; // h2inv = 1 / (6 * h2)

            // r = f - A*v
            res = f[index] - stencilsum;

            // u_(m+1) = u_(m) + omega * (D^-1) * r_(m)
            v_out[index] = v_in_index + omega * res * dinv;

            if (store_residual)
                r[index] = res;

            index += ioff;
        }
    }
}

/* runs one iteration of jacobi's method using 27-point stencil and one work-item per cell.
 * global size must be of ghosted grid.
 * m, n and o must be dimensions of ghosted grid, too.
 * h2 is grid spacing to the power of 2.
 * dinv is h2/A(i,i), e.g. h2/6.0 for 3D laplacian stencil.
 * If store_residual is true, the residual will be stored into global field r.
 * idx_start determines which cells shall be calculated, which is relevant for running
 *   Jacobi with multiple iterations without ghost cell update in-between. I.e. when
 *   stepsPerIter = 1: idx_start = ghosts.
 */
__kernel void jacobi_iter_27point(
    __global double* restrict v_in, // needed s.t. every work-item can read surrounding cell values
    __global double* restrict v_out,
    __global double* restrict f,
    __global double* restrict r,
    const double h2inv, const double dinv, const double omega,
    const int mgh, const int ngh, const int ogh, const int ghosts,
    const int idx_start, const int store_residual)
{
    int j = get_global_id(0);
    int k = get_global_id(1);

    // calculate residual for real cells plus some ghost cells if stepsPerIter > 1.
    if (j >= idx_start && k >= idx_start && j < ngh - idx_start && k < ogh - idx_start)
    {
        int ioff = ngh * ogh;
        int joff = ogh;
        int koff = 1;
        int index = idx_start * ioff + j * ogh + k;

        for (int i = idx_start; i < mgh - idx_start; i++)
        {
            double res;
            double v_in_index = v_in[index];

            // A*v 19-point laplacian stencil
            double stencilsum =
                (128.0 * v_in_index - 14.0 * v_in[index - koff] - 14.0 * v_in[index + koff] -
                 14.0 * v_in[index - joff] - 14.0 * v_in[index + joff] - 14.0 * v_in[index - ioff] -
                 14.0 * v_in[index + ioff]

                 - 3.0 * v_in[index - joff - koff] - 3.0 * v_in[index - joff + koff] - 3.0 * v_in[index + joff - koff] -
                 3.0 * v_in[index + joff + koff] - 3.0 * v_in[index - ioff - koff] - 3.0 * v_in[index - ioff + koff] -
                 3.0 * v_in[index + ioff - koff] - 3.0 * v_in[index + ioff + koff] - 3.0 * v_in[index - ioff - joff] -
                 3.0 * v_in[index - ioff + joff] - 3.0 * v_in[index + ioff - joff] - 3.0 * v_in[index + ioff + joff]

                 - v_in[index - ioff - joff - koff] - v_in[index - ioff - joff + koff] -
                 v_in[index - ioff + joff - koff] - v_in[index - ioff + joff + koff] -
                 v_in[index + ioff - joff - koff] - v_in[index + ioff - joff + koff] -
                 v_in[index + ioff + joff - koff] - v_in[index + ioff + joff + koff]) *
                h2inv; // h2inv = 1 / (30 * h2)

            // r = f - A*v
            res = f[index] - stencilsum;

            // u_(m+1) = u_(m) + omega * (D^-1) * r_(m)
            v_out[index] = v_in_index + omega * res * dinv;

            if (store_residual)
                r[index] = res;

            index += ioff;
        }
    }
}

/* runs one iteration of jacobi's method using one work-item per row.
 * uses a 2D kernel which loops over cells in x-direction. y and z is parallelized.
 * global size must be of ghosted grid.
 * mgh, ngh and ogh must be dimensions of local ghosted grid, too.
 * svmgh, svngh and svogh are ghosted grid sizes of stencilValues (might differ from mgh,ngh,ogh when using MPI).
 * h2 is grid spacing to the power of 2
 * dinv is h2/A(i,i), e.g. h2/6.0 for 3D laplacian stencil
 * if store_residual is true, the residual will be stored into global field r.
 * stencilValues is a VaryingStencilGpu having width 3 (i.e. a 6d array).
 * ghosts is the amount of ghost cells of v, f and r.
 * ghosts_sv is the amount of ghost cells of stencilValues.
 * idx_start determines which cells shall be calculated, which is relevant for running
 *   Jacobi with multiple iterations without ghost cell update in-between. I.e. when
 *   stepsPerIter = 1: idx_start = ghosts.
 */
__kernel void jacobi_iter_27point_varying_stencil(
    __global double* restrict v_in, // needed s.t. every work-item can read surrounding cell values
    __global double* restrict v_out,
    __global double* restrict f,
    __global double* restrict r,
    __global double* restrict stencilValues,
    const double omega,
    const int mgh, const int ngh, const int ogh,
    const int svmgh, const int svngh, const int svogh,
    const int ghosts, const int ghosts_sv,
    const int svGridSize,
    const int idx_start, const int store_residual)
{
    int j = get_global_id(0);
    int k = get_global_id(1);

    // calculate residual for real cells plus some ghost cells if stepsPerIter > 1.
    if (j >= idx_start && k >= idx_start && j < ngh - idx_start && k < ogh - idx_start)
    {
        int ioff = ngh * ogh;
        int joff = ogh;
        int koff = 1;
        int index = idx_start * ioff + j * ogh + k;

        int svno = svngh * svogh;
        // offset inside one coefficient grid that points to the coefficient for the current grid point. Must consider different amount of ghosts for v and sv.
        int index_sv = (idx_start - ghosts + ghosts_sv) * svno + (j - ghosts + ghosts_sv) * svogh + (k - ghosts + ghosts_sv);

        for (int i = idx_start; i < mgh - idx_start; i++)
        {
            double res;
            double v_in_index = v_in[index];
            double sv_self = stencilValues[index_sv + (9 + 3 + 1) * svGridSize];

            // A*v
            // clang-format off
            double stencilsum = stencilValues[index_sv + (9 + 3 + 1) * svGridSize] * v_in[index]
                + stencilValues[index_sv + (9 + 3) * svGridSize]      * v_in[index - 1]
                + stencilValues[index_sv + (9 + 3 + 2) * svGridSize]  * v_in[index + 1]
                + stencilValues[index_sv + (9 + 1) * svGridSize]      * v_in[index - joff]
                + stencilValues[index_sv + (9 + 6 + 1) * svGridSize]  * v_in[index + joff]
                + stencilValues[index_sv + (3 + 1) * svGridSize]      * v_in[index - ioff]
                + stencilValues[index_sv + (18 + 3 + 1) * svGridSize] * v_in[index + ioff]
                
                + stencilValues[index_sv + (9) * svGridSize]          * v_in[index - joff - koff]
                + stencilValues[index_sv + (9 + 2) * svGridSize]      * v_in[index - joff + koff]
                + stencilValues[index_sv + (9 + 6) * svGridSize]      * v_in[index + joff - koff]
                + stencilValues[index_sv + (9 + 6 + 2) * svGridSize]  * v_in[index + joff + koff]
                + stencilValues[svGridSize * 3 + index_sv]            * v_in[index - ioff - koff]
                + stencilValues[index_sv + (3 + 2) * svGridSize]      * v_in[index - ioff + koff]
                + stencilValues[index_sv + (18 + 3) * svGridSize]     * v_in[index + ioff - koff]
                + stencilValues[index_sv + (18 + 3 + 2) * svGridSize] * v_in[index + ioff + koff]
                + stencilValues[svGridSize + index_sv]                * v_in[index - ioff - joff]
                + stencilValues[index_sv + (6 + 1) * svGridSize]      * v_in[index - ioff + joff]
                + stencilValues[index_sv + (18 + 1) * svGridSize]     * v_in[index + ioff - joff]
                + stencilValues[index_sv + (18 + 6 + 1) * svGridSize] * v_in[index + ioff + joff]

                + stencilValues[index_sv]                           * v_in[index - ioff - joff - koff]
                + stencilValues[svGridSize * 2 + index_sv]            * v_in[index - ioff - joff + koff]
                + stencilValues[index_sv + (6) * svGridSize]          * v_in[index - ioff + joff - koff]
                + stencilValues[index_sv + (6 + 2) * svGridSize]      * v_in[index - ioff + joff + koff]
                + stencilValues[index_sv + (18) * svGridSize]         * v_in[index + ioff - joff - koff]
                + stencilValues[index_sv + (18 + 2) * svGridSize]     * v_in[index + ioff - joff + koff]
                + stencilValues[index_sv + (18 + 6) * svGridSize]     * v_in[index + ioff + joff - koff]
                + stencilValues[index_sv + (18 + 6 + 2) * svGridSize] * v_in[index + ioff + joff + koff];
            // clang-format on

            // r = f - A*v
            res = f[index] - stencilsum;

            // u_(m+1) = u_(m) + omega * (D^-1) * r_(m)
            v_out[index] = v_in_index + omega * (1.0 / sv_self) * res;

            if (store_residual)
                r[index] = res;

            index += ioff;
            index_sv += svno;
        }
    }
}

/* runs one iteration of jacobi's method using one work-item per grid node.
 * uses a 1d kernel, which parallelizes all three loop in x,y and z directions.
 * global size must be of ghosted grid.
 * mgh, ngh and ogh must be dimensions of local ghosted grid, too.
 * svmgh, svngh and svogh are ghosted grid sizes of stencilValues (might differ from mgh,ngh,ogh when using MPI).
 * h2 is grid spacing to the power of 2
 * dinv is h2/A(i,i), e.g. h2/6.0 for 3D laplacian stencil
 * if store_residual is true, the residual will be stored into global field r.
 * stencilValues is a VaryingStencilGpu having width 3 (i.e. a 6d array).
 * ghosts is the amount of ghost cells of v, f and r.
 * ghosts_sv is the amount of ghost cells of stencilValues.
 * idx_start determines which cells shall be calculated, which is relevant for running
 *   Jacobi with multiple iterations without ghost cell update in-between. I.e. when
 *   stepsPerIter = 1: idx_start = ghosts.
 */
__kernel void jacobi_iter_27point_varying_stencil_1d(
    __global double* restrict v_in, // needed s.t. every work-item can read surrounding cell values
    __global double* restrict v_out,
    __global double* restrict f,
    __global double* restrict r,
    __global double* restrict stencilValues,
    const double omega,
    const int mgh, const int ngh, const int ogh,
    const int svmgh, const int svngh, const int svogh,
    const int ghosts, const int ghosts_sv,
    const int svGridSize,
    const int idx_start, const int store_residual)
{
    int idx = get_global_id(0);
    int no = ngh * ogh;
    int i = idx / no;
    int j = (idx - i * no) / ogh;
    int k = idx % ogh;

    // calculate residual for real cells plus some ghost cells if stepsPerIter > 1.
    if (i >= idx_start && j >= idx_start && k >= idx_start && i < mgh - idx_start && j < ngh - idx_start && k < ogh - idx_start)
    {
        int ioff = ngh * ogh;
        int joff = ogh;
        int koff = 1;
        int index = i * ioff + j * ogh + k;

        int svno = svngh * svogh;
        // offset inside one coefficient grid that points to the coefficient for the current grid point. Must consider different amount of ghosts for v and sv.
        int index_sv = (i - ghosts + ghosts_sv) * svno + (j - ghosts + ghosts_sv) * svogh + (k - ghosts + ghosts_sv);

        double res;
        double v_in_index = v_in[index];
        double sv_self = stencilValues[index_sv + (9 + 3 + 1) * svGridSize];

        // A*v
        // clang-format off
        double stencilsum = sv_self * v_in[index]
            + stencilValues[index_sv + (9 + 3) * svGridSize]      * v_in[index - 1]
            + stencilValues[index_sv + (9 + 3 + 2) * svGridSize]  * v_in[index + 1]
            + stencilValues[index_sv + (9 + 1) * svGridSize]      * v_in[index - joff]
            + stencilValues[index_sv + (9 + 6 + 1) * svGridSize]  * v_in[index + joff]
            + stencilValues[index_sv + (3 + 1) * svGridSize]      * v_in[index - ioff]
            + stencilValues[index_sv + (18 + 3 + 1) * svGridSize] * v_in[index + ioff]
            
            + stencilValues[index_sv + (9) * svGridSize]          * v_in[index - joff - koff]
            + stencilValues[index_sv + (9 + 2) * svGridSize]      * v_in[index - joff + koff]
            + stencilValues[index_sv + (9 + 6) * svGridSize]      * v_in[index + joff - koff]
            + stencilValues[index_sv + (9 + 6 + 2) * svGridSize]  * v_in[index + joff + koff]
            + stencilValues[svGridSize * 3 + index_sv]            * v_in[index - ioff - koff]
            + stencilValues[index_sv + (3 + 2) * svGridSize]      * v_in[index - ioff + koff]
            + stencilValues[index_sv + (18 + 3) * svGridSize]     * v_in[index + ioff - koff]
            + stencilValues[index_sv + (18 + 3 + 2) * svGridSize] * v_in[index + ioff + koff]
            + stencilValues[svGridSize + index_sv]                * v_in[index - ioff - joff]
            + stencilValues[index_sv + (6 + 1) * svGridSize]      * v_in[index - ioff + joff]
            + stencilValues[index_sv + (18 + 1) * svGridSize]     * v_in[index + ioff - joff]
            + stencilValues[index_sv + (18 + 6 + 1) * svGridSize] * v_in[index + ioff + joff]

            + stencilValues[index_sv]                           * v_in[index - ioff - joff - koff]
            + stencilValues[svGridSize * 2 + index_sv]            * v_in[index - ioff - joff + koff]
            + stencilValues[index_sv + (6) * svGridSize]          * v_in[index - ioff + joff - koff]
            + stencilValues[index_sv + (6 + 2) * svGridSize]      * v_in[index - ioff + joff + koff]
            + stencilValues[index_sv + (18) * svGridSize]         * v_in[index + ioff - joff - koff]
            + stencilValues[index_sv + (18 + 2) * svGridSize]     * v_in[index + ioff - joff + koff]
            + stencilValues[index_sv + (18 + 6) * svGridSize]     * v_in[index + ioff + joff - koff]
            + stencilValues[index_sv + (18 + 6 + 2) * svGridSize] * v_in[index + ioff + joff + koff];
        // clang-format on

        // r = f - A*v
        res = f[index] - stencilsum;

        // u_(m+1) = u_(m) + omega * (D^-1) * r_(m)
        v_out[index] = v_in_index + omega * (1.0 / sv_self) * res;

        if (store_residual)
            r[index] = res;
    }
}

/* Restricts from fine to coarse grid.
 * Needs to get called with m*n*o work-items.
 * m,n,o is size of ghosted coarse grid.
 * fine and coarse must be of sizes of ghosted grids.
 * gh_vals_coarse must be sizes of coarse grid's cuboids. Most of the time its equal to m,n,o but for
 * the threshold-level when using MPI, the cuboid is bigger than the actual level would be. */
__kernel void restrict_to_coarse(
    __global double* restrict fine,
    __global double* restrict coarse,
    const int m, const int n, const int o, const int ghosts,
    const int ngh_vals_coarse, const int ogh_vals_coarse)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);
    int g2 = 2 * ghosts;

    if (i < m && j < n && k < o)
    {
        const int index = (i + ghosts) * ngh_vals_coarse * ogh_vals_coarse + (j + ghosts) * ogh_vals_coarse + (k + ghosts);
        const int nf = (n - g2) * 2 + g2, of = (o - g2) * 2 + g2;
        const int i2 = i * 2 + ghosts + 1, j2 = j * 2 + ghosts + 1, k2 = k * 2 + ghosts + 1;
        coarse[index] =
            0.125 * fine[i2 * nf * of + j2 * of + k2] // self
            // direct neighbours
            + 0.0625 * fine[(i2 - 1) * nf * of + j2 * of + k2] + 0.0625 * fine[(i2 + 1) * nf * of + j2 * of + k2] +
            0.0625 * fine[i2 * nf * of + (j2 - 1) * of + k2] + 0.0625 * fine[i2 * nf * of + (j2 + 1) * of + k2] +
            0.0625 * fine[i2 * nf * of + j2 * of + k2 - 1] +
            0.0625 * fine[i2 * nf * of + j2 * of + k2 + 1]
            // edge midpoints xy-plane
            + 0.03125 * fine[(i2 - 1) * nf * of + (j2 - 1) * of + k2] +
            0.03125 * fine[(i2 - 1) * nf * of + (j2 + 1) * of + k2] +
            0.03125 * fine[(i2 + 1) * nf * of + (j2 - 1) * of + k2] +
            0.03125 * fine[(i2 + 1) * nf * of + (j2 + 1) * of + k2]
            // edge midpoints xz-plane
            + 0.03125 * fine[(i2 - 1) * nf * of + j2 * of + k2 - 1] +
            0.03125 * fine[(i2 - 1) * nf * of + j2 * of + k2 + 1] +
            0.03125 * fine[(i2 + 1) * nf * of + j2 * of + k2 - 1] +
            0.03125 * fine[(i2 + 1) * nf * of + j2 * of + k2 + 1]
            // edge midpoints yz-plane
            + 0.03125 * fine[i2 * nf * of + (j2 - 1) * of + k2 - 1] +
            0.03125 * fine[i2 * nf * of + (j2 - 1) * of + k2 + 1] +
            0.03125 * fine[i2 * nf * of + (j2 + 1) * of + k2 - 1] +
            0.03125 * fine[i2 * nf * of + (j2 + 1) * of + k2 + 1]
            // corners
            + 0.015625 * fine[(i2 - 1) * nf * of + (j2 - 1) * of + k2 - 1] +
            0.015625 * fine[(i2 - 1) * nf * of + (j2 - 1) * of + k2 + 1] +
            0.015625 * fine[(i2 - 1) * nf * of + (j2 + 1) * of + k2 - 1] +
            0.015625 * fine[(i2 - 1) * nf * of + (j2 + 1) * of + k2 + 1] +
            0.015625 * fine[(i2 + 1) * nf * of + (j2 - 1) * of + k2 - 1] +
            0.015625 * fine[(i2 + 1) * nf * of + (j2 - 1) * of + k2 + 1] +
            0.015625 * fine[(i2 + 1) * nf * of + (j2 + 1) * of + k2 - 1] +
            0.015625 * fine[(i2 + 1) * nf * of + (j2 + 1) * of + k2 + 1];
    }
}

/* Prolongates from coarse to fine grid.
 * Needs to get called with #work-items = #nodes of coarse grid.
 * m,n,o is size of ghosted fine grid.
 * fine and coarse must be of sizes of ghosted grids.
 * gh_vals_coarse must be sizes of coarse grid's cuboids. Most of the time its equal to m/2,n/2,o/2 but for
 * the threshold-level when using MPI, the cuboid is bigger than the actual level would be. */
__kernel void prolongate_to_fine(
    __global double* restrict fine,
    __global double* restrict coarse,
    const int m, const int n, const int o, const int ghosts,
    const int ngh_vals_coarse, const int ogh_vals_coarse)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);
    int g2 = 2 * ghosts;

    const int mc = (m - g2) / 2 + g2, nc = (n - g2) / 2 + g2, oc = (o - g2) / 2 + g2;

    if (i > ghosts - 1 && i < mc - ghosts && j > ghosts - 1 && j < nc - ghosts && k > ghosts - 1 && k < oc - ghosts)
    {
        const int index_coarse = i * ngh_vals_coarse * ogh_vals_coarse + j * ogh_vals_coarse + k;
        const int i2 = i * 2 - (ghosts - 1), j2 = j * 2 - (ghosts - 1), k2 = k * 2 - (ghosts - 1);

        fine[i2 * n * o + j2 * o + k2] = coarse[index_coarse];

        fine[i2 * n * o + j2 * o + k2 - 1] = 0.5 * (coarse[index_coarse] + coarse[index_coarse - 1]);
        fine[i2 * n * o + (j2 - 1) * o + k2] = 0.5 * (coarse[index_coarse] + coarse[i * ngh_vals_coarse * ogh_vals_coarse + (j - 1) * ogh_vals_coarse + k]);
        fine[(i2 - 1) * n * o + j2 * o + k2] = 0.5 * (coarse[index_coarse] + coarse[(i - 1) * ngh_vals_coarse * ogh_vals_coarse + j * ogh_vals_coarse + k]);

        fine[i2 * n * o + (j2 - 1) * o + k2 - 1] =
            0.25 * (coarse[index_coarse] + coarse[index_coarse - 1] + coarse[i * ngh_vals_coarse * ogh_vals_coarse + (j - 1) * ogh_vals_coarse + k] +
                    coarse[i * ngh_vals_coarse * ogh_vals_coarse + (j - 1) * ogh_vals_coarse + k - 1]);
        fine[(i2 - 1) * n * o + j2 * o + k2 - 1] =
            0.25 * (coarse[index_coarse] + coarse[index_coarse - 1] + coarse[(i - 1) * ngh_vals_coarse * ogh_vals_coarse + j * ogh_vals_coarse + k] +
                    coarse[(i - 1) * ngh_vals_coarse * ogh_vals_coarse + j * ogh_vals_coarse + k - 1]);
        fine[(i2 - 1) * n * o + (j2 - 1) * o + k2] =
            0.25 * (coarse[index_coarse] + coarse[i * ngh_vals_coarse * ogh_vals_coarse + (j - 1) * ogh_vals_coarse + k] +
                    coarse[(i - 1) * ngh_vals_coarse * ogh_vals_coarse + j * ogh_vals_coarse + k] + coarse[(i - 1) * ngh_vals_coarse * ogh_vals_coarse + (j - 1) * ogh_vals_coarse + k]);

        fine[(i2 - 1) * n * o + (j2 - 1) * o + k2 - 1] =
            0.125 * (coarse[index_coarse] + coarse[index_coarse - 1] + coarse[i * ngh_vals_coarse * ogh_vals_coarse + (j - 1) * ogh_vals_coarse + k] +
                     coarse[i * ngh_vals_coarse * ogh_vals_coarse + (j - 1) * ogh_vals_coarse + k - 1] + coarse[(i - 1) * ngh_vals_coarse * ogh_vals_coarse + j * ogh_vals_coarse + k] +
                     coarse[(i - 1) * ngh_vals_coarse * ogh_vals_coarse + j * ogh_vals_coarse + k - 1] + coarse[(i - 1) * ngh_vals_coarse * ogh_vals_coarse + (j - 1) * ogh_vals_coarse + k] +
                     coarse[(i - 1) * ngh_vals_coarse * ogh_vals_coarse + (j - 1) * ogh_vals_coarse + k - 1]);
    }
}

/* Loads a plane from global buffer src into target buffer dest.
 * Also loads outmost ghosted border.
 * joff is the offset in j direction, e.g. plane's x-size */
void load_plane(__global double* restrict src, __local double* restrict dest, int index_src, int index_dest, int joff,
                int joff_global)
{
    // if (blockIdx.x == 0 && blockIdx.y == 0 && get_local_id(0) == 0 && get_local_id(1) == 0)
    //     printf("load plane: index_src = %d, index_dest = %d, joff = %d\n", index_src, index_dest, joff);

    // initialize local storage + registers
    // self
    dest[index_dest] = src[index_src];

    // outmost ghosted border per block (non-diagonal)
    if (get_local_id(1) == 0)
        dest[index_dest - 1] = src[index_src - 1];
    else if (get_local_id(1) == get_local_size(1) - 1)
        dest[index_dest + 1] = src[index_src + 1];

    if (get_local_id(0) == 0)
        dest[index_dest - joff] = src[index_src - joff_global];
    else if (get_local_id(0) == get_local_size(0) - 1)
        dest[index_dest + joff] = src[index_src + joff_global];

    barrier(CLK_GLOBAL_MEM_FENCE);
}

/* Loads a plane from global buffer src into target buffer dest.
 * Also loads outmost ghosted border including diagonal corners.
 * joff is the offset in j direction, e.g. plane's x-size */
void load_plane_diag(__global double* restrict src, __local double* restrict dest, int index_src, int index_dest,
                     int joff, int joff_global)
{
    // if (blockIdx.x == 0 && blockIdx.y == 0 && get_local_id(0) == 0 && get_local_id(1) == 0)
    //     printf("load plane: index_src = %d, index_dest = %d, joff = %d value = %e (global)\n", index_src, index_dest,
    //     joff, src[index_src]);

    // initialize local storage + registers
    // self
    dest[index_dest] = src[index_src];

    // outmost ghosted border per block (non-diagonal)
    if (get_local_id(1) == 0)
        dest[index_dest - 1] = src[index_src - 1];
    else if (get_local_id(1) == get_local_size(1) - 1)
        dest[index_dest + 1] = src[index_src + 1];

    if (get_local_id(0) == 0)
        dest[index_dest - joff] = src[index_src - joff_global];
    else if (get_local_id(0) == get_local_size(0) - 1)
        dest[index_dest + joff] = src[index_src + joff_global];

    // load corners
    if (get_local_id(0) == 0 && get_local_id(1) == 0) // top left
        dest[index_dest - joff - 1] = src[index_src - joff_global - 1];
    else if (get_local_id(0) == get_local_size(0) - 1 && get_local_id(1) == 0) // top right
        dest[index_dest + joff - 1] = src[index_src + joff_global - 1];
    else if (get_local_id(0) == 0 && get_local_id(1) == get_local_size(1) - 1) // bottom left
        dest[index_dest - joff + 1] = src[index_src - joff_global + 1];
    else if (get_local_id(0) == get_local_size(0) - 1 && get_local_id(1) == get_local_size(1) - 1) // bottom right
        dest[index_dest + joff + 1] = src[index_src + joff_global + 1];

    barrier(CLK_GLOBAL_MEM_FENCE);
    // if (blockIdx.x == 0 && blockIdx.y == 0 && get_local_id(0) == 0 && get_local_id(1) == 0)
    //     printf("load plane: index_src = %d, index_dest = %d, joff = %d value = %e (local)\n", index_src, index_dest,
    //     joff, dest[index_dest]);
}

/* runs 3 jacobi iterations on a 3D grid using shared memory.
 * uses algorithm of stencilgen.
 * x-plane of v_in needs to have 3 ghosted cells at each side to enable temporal tiling.
 * m,n,o must be size of v_in, including ghost cells.
 * uses shared memory only instead of registers.
 * debug_out has the same dimensions as v_in.
 * ghosts is the amount of ghost cells of v_in in one direction */
__kernel void jacobi_stream_shmem_7point(
    __global double* restrict v_in, // needed s.t. every work-item can read surrounding cell values
    __global double* restrict v_out, __global double* restrict f, __global double* restrict r, __local double* vlocal,
    const double h2inv, const double dinv, const double omega, const int m, const int n, const int o, const int ghosts,
    const int iterations, const int store_residual)
{
    // extended block dimensions (field stored in shared memory has one more row and column)
    int blockDimExX = get_local_size(0) + 2;
    int blockDimExY = get_local_size(1) + 2;
    int blockSizeEx = blockDimExX * blockDimExY;

    // index of real cell having block_ghosts = iterations-1. offset by +1 because input v has one more ghost row than
    // block
    int jreal = get_group_id(0) * get_local_size(0) - get_group_id(0) * (iterations - 1) * 2 + get_local_id(0) + 1;
    int kreal = get_group_id(1) * get_local_size(1) - get_group_id(1) * (iterations - 1) * 2 + get_local_id(1) + 1;

    // account for padding
    if (jreal < n && kreal < o)
    {
        int ioff = n * o;
        int ioff_out = (n - ghosts * 2) * (o - ghosts * 2); // size in x-direction without ghost cells
        int joff = blockDimExX;
        int koff = 1;
        int index = jreal * o + kreal; // global index of current cell (!= thread) in x-plane (for reading only)
        int index_out = (jreal - ghosts) * (o - ghosts * 2) + kreal - ghosts;
        int index_f = index; // seperate index for accessing F which is changing per time step
        // ghosted field has one more row than block, thus an offset must be added
        int index_loc = (get_local_id(0) + 1) * blockDimExX + get_local_id(1) +
                        1; // index of cell relative to current thread block (wg) + 2
        double stencilsum;
        double res;
        int tidx;
        int tidx_old;
        double omega_dinv = omega * dinv;
        int cnt = iterations == 1 ? 3 : 0; // counter for checking how many times plane buffers are shifted

        // store values of x-1, current x and x+1-planes in local memory, e.g. for t = 3, wg size = 12
        // size = (time steps + 1) * (block (wg) size + 2) = 4*14*14
        __local double* A = vlocal;
        __local double* rp = vlocal + blockSizeEx * iterations;
        __local double* rm =
            vlocal + blockSizeEx * 2 * iterations; // (time steps + 1) * (block (wg) size + 2) = 4*14*14
        __local double* tmp;

        // initialize local storage + registers
        load_plane(v_in, rm, index, index_loc, joff, o);
        index += ioff;
        load_plane(v_in, A, index, index_loc, joff, o);

        // cell index of next time step in shared memory
        tidx_old = index_loc;
        tidx = blockSizeEx + index_loc;

        // iterate over prologue planes
        for (int x = 1; x < iterations * 2 - 1; x++)
        {
            load_plane(v_in, rp, index + ioff, index_loc, joff, o);
            barrier(CLK_LOCAL_MEM_FENCE);

            // calculate higher iterations just like in the streaming part later but
            // stop earlier when for the current plane the current timestep does not exist
            tidx_old = index_loc;
            tidx = blockSizeEx + index_loc;
            index_f = index;
            for (int t = 0; x - t >= t + 1; t++)
            {
                // calculate and store stencil
                stencilsum = (6.0 * A[tidx_old] - A[tidx_old - koff] - A[tidx_old + koff] - A[tidx_old - joff] -
                              A[tidx_old + joff] - rm[tidx_old] - rp[tidx_old]) *
                             h2inv;

                res = f[index_f] - stencilsum;
                rm[tidx] = A[tidx_old] + omega_dinv * res;
                barrier(CLK_LOCAL_MEM_FENCE);

                // if (get_group_id(0) == 0 && get_group_id(1) == 0 && get_local_id(0) == 0 && get_local_id(1) == 0 && x
                // == 1)
                // {
                //     printf("x = %d, res = %e, v_out = %e\n", x, res, rm[tidx]);
                //     print_7point(A, rm, rp, tidx_old, ioff, joff, koff);
                // }
                // if (get_group_id(0) == 0 && get_group_id(1) == 0 && get_local_id(0) == 2 && get_local_id(1) == 2)
                // printf("x = %d, t = %d -> writing to x = %d\n", x, t, x-t);

                // if ((get_local_id(0) >= (iterations-1) && get_local_id(0) < get_local_size(0) - (iterations-1) &&
                //     get_local_id(1) >= (iterations-1) && get_local_id(1) < get_local_size(1) - (iterations-1)) ||
                //     (get_group_id(0) == 0 && get_local_id(0) < get_local_size(0) - (iterations-1)) ||
                //     (get_group_id(1) == 0 && get_local_id(1) < get_local_size(1) - (iterations-1)) ||
                //     (get_group_id(0) == gridDim.x - 1 && get_local_id(0) >= (iterations-1)) ||
                //     (get_group_id(1) == gridDim.y - 1 && get_local_id(1) >= (iterations-1)))
                //     debug_out[index] = rm[tidx];

                // if (t == 1 && get_local_id(0) >= ghosts-1 && get_local_id(0) < get_local_size(0) - (ghosts-1) &&
                //     get_local_id(1) >= ghosts-1 && get_local_id(1) < get_local_size(1) - (ghosts-1))
                //     debug_out[index_f] = rm[tidx];

                // shift data
                tmp = rm;
                rm = A;
                A = rp;
                rp = tmp;
                cnt++;

                // update indices for next time step
                tidx_old = tidx;
                tidx += blockSizeEx;
                index_f -= ioff;
            }

            // buffers must be shifted backwards a multiple of 3 times plus 1 in total for the next plane, e.g.
            // 1,4,7,...
            for (cnt = (cnt - 1) % 3; cnt > 0; cnt--)
            {
                tmp = rp;
                rp = A;
                A = rm;
                rm = tmp;
            }

            index += ioff; // update index to next plane
        }

        // now stream for each x-plane that is left
        for (int x = iterations * 2 - 1; x < m - 1; x++)
        {
            load_plane(v_in, rp, index + ioff, index_loc, joff, o);
            barrier(CLK_LOCAL_MEM_FENCE);

            // calculate every time step but the last one and store result in shared memory
            tidx_old = index_loc;
            tidx = blockSizeEx + index_loc;
            index_f = index;
            for (int t = 0; t < iterations - 1; t++)
            {
                // calculate and store stencil
                stencilsum = (6.0 * A[tidx_old] - A[tidx_old - koff] - A[tidx_old + koff] - A[tidx_old - joff] -
                              A[tidx_old + joff] - rm[tidx_old] - rp[tidx_old]) *
                             h2inv;

                res = f[index_f] - stencilsum;
                rm[tidx] = A[tidx_old] + omega_dinv * res;
                barrier(CLK_LOCAL_MEM_FENCE);

                // if (get_group_id(0) == 0 && get_group_id(1) == 0 && get_local_id(0) == 2 && get_local_id(1) == 2 && x
                // == 5)
                // {
                //     printf("x = %d, res = %e, v_out = %e\n", x, res, rm[tidx]);
                //     print_7point(A, rm, rp, tidx_old, ioff, joff, koff);
                // }

                // if (t == 1 && get_local_id(0) >= ghosts-1 && get_local_id(0) < get_local_size(0) - (ghosts-1) &&
                //     get_local_id(1) >= ghosts-1 && get_local_id(1) < get_local_size(1) - (ghosts-1))
                //     debug_out[index_f] = rm[tidx];

                // shift data
                tmp = rm;
                rm = A;
                A = rp;
                rp = tmp;
                cnt++;

                // update indices for next time step
                tidx_old = tidx;
                tidx += blockSizeEx;
                index_f -= ioff;
            }

            // calculate last time step and store in global memory
            stencilsum = (6.0 * A[tidx_old] - A[tidx_old - koff] - A[tidx_old + koff] - A[tidx_old - joff] -
                          A[tidx_old + joff] - rm[tidx_old] - rp[tidx_old]) *
                         h2inv;

            res = f[index_f] - stencilsum;

            // if (jreal == ghosts && kreal == ghosts && x >= ghosts && x <= ghosts+2)
            // {
            //     printf("get_num_groups = %d,%d\n", (int)get_num_groups(0), (int)get_num_groups(1));
            //     printf("jreal,kreal = %d,%d; global_id = %d,%d; local_id = %d,%d; group_id = %d,%d\n", jreal, kreal,
            //     (int)get_global_id(0), (int)get_global_id(1), (int)get_local_id(0), (int)get_local_id(1),
            //     (int)get_group_id(0), (int)get_group_id(1)); printf("index_f = %d\n", index_f); printf("x = %d, res =
            //     %e, v_out = %e\n", x, res, A[tidx_old] + omega_dinv * res); print_7point_local(A, rm, rp, tidx_old,
            //     ioff, joff, koff); if (get_local_id(0) >= iterations-1 && get_local_id(0) < get_local_size(0) -
            //     (iterations-1) && get_local_id(1) >= iterations-1 && get_local_id(1) < get_local_size(1) -
            //     (iterations-1))
            //         printf("will store result\n");
            // }

            // store result
            if (get_local_id(0) >= iterations - 1 && get_local_id(0) < get_local_size(0) - (iterations - 1) &&
                get_local_id(1) >= iterations - 1 && get_local_id(1) < get_local_size(1) - (iterations - 1))
                v_out[index_f] = A[tidx_old] + omega_dinv * res;

            barrier(CLK_LOCAL_MEM_FENCE);

            if (store_residual && get_local_id(0) >= ghosts - 1 && get_local_id(0) < get_local_size(0) - (ghosts - 1) &&
                get_local_id(1) >= ghosts - 1 && get_local_id(1) < get_local_size(1) - (ghosts - 1))
                r[index_out] = res;

            // buffers must be shifted backwards a multiple of 3 times plus 1 in total for the next plane, e.g.
            // 1,4,7,...
            for (cnt = (cnt - 1) % 3; cnt > 0; cnt--)
            {
                tmp = rp;
                rp = A;
                A = rm;
                rm = tmp;
            }

            index += ioff; // update index to next plane
            index_out += ioff_out;
            cnt = iterations == 1 ? 3 : 0; // TODO find better solution for shifting planes with special case t==1
        }
    }
}

/* runs iterations jacobi iterations on a 3D grid using shared memory using 19 point stencil.
 * uses algorithm of stencilgen.
 * x-plane of v_in needs to have 3 ghosted cells at each side to enable temporal tiling.
 * m,n,o must be size of v_in, including ghost cells.
 * uses shared memory only instead of registers.
 * debug_out has the same dimensions as v_in.
 * ghosts is the amount of ghost cells of v_in in one direction */
__kernel void jacobi_stream_shmem_19point(
    __global double* restrict v_in, // needed s.t. every work-item can read surrounding cell values
    __global double* restrict v_out, __global double* restrict f, __global double* restrict r, __local double* vlocal,
    const double h2inv, const double dinv, const double omega, const int m, const int n, const int o, const int ghosts,
    const int iterations, const int store_residual)
{
    // extended block dimensions (field stored in shared memory has one more row and column)
    int blockDimExX = get_local_size(0) + 2;
    int blockDimExY = get_local_size(1) + 2;
    int blockSizeEx = blockDimExX * blockDimExY;

    // index of real cell having block_ghosts = iterations-1. offset by +1 because input v has one more ghost row than
    // block
    int jreal = get_group_id(0) * get_local_size(0) - get_group_id(0) * (ghosts - 1) * 2 + get_local_id(0) + 1;
    int kreal = get_group_id(1) * get_local_size(1) - get_group_id(1) * (ghosts - 1) * 2 + get_local_id(1) + 1;

    // account for padding
    if (jreal < n && kreal < o)
    {
        int ioff = n * o;
        int ioff_out = (n - ghosts * 2) * (o - ghosts * 2); // size in x-direction without ghost cells
        int joff = blockDimExX;
        int koff = 1;
        int index = jreal * o + kreal; // global index of current cell (!= thread) in x-plane (for reading only)
        int index_out = (jreal - ghosts) * (o - ghosts * 2) + kreal - ghosts;
        int index_f = index; // seperate index for accessing F which is changing per time step
        // ghosted field has one more row than block, thus an offset must be added
        int index_loc = (get_local_id(0) + 1) * blockDimExX + get_local_id(1) +
                        1; // index of cell relative to current thread block (wg) + 2
        double stencilsum;
        double res;
        int tidx;
        int tidx_old;
        double omega_dinv = omega * dinv;
        int cnt = iterations == 1 ? 3 : 0; // counter for checking how many times plane buffers are shifted

        // store values of x-1, current x and x+1-planes in local memory, e.g. for t = 3, wg size = 12
        // size = (time steps + 1) * (block (wg) size + 2) = 4*14*14
        __local double* A = vlocal;
        __local double* rp = vlocal + blockSizeEx * iterations;
        __local double* rm =
            vlocal + blockSizeEx * 2 * iterations; // (time steps + 1) * (block (wg) size + 2) = 4*14*14
        __local double* tmp;

        // initialize local storage + registers
        load_plane_diag(v_in, rm, index, index_loc, joff, o);
        index += ioff;
        load_plane_diag(v_in, A, index, index_loc, joff, o);

        // cell index of next time step in shared memory
        tidx_old = index_loc;
        tidx = blockSizeEx + index_loc;

        // iterate over prologue planes
        for (int x = 1; x < iterations * 2 - 1; x++)
        {
            load_plane_diag(v_in, rp, index + ioff, index_loc, joff, o);

            // calculate higher iterations just like in the streaming part later but
            // stop earlier when for the current plane the current timestep does not exist
            tidx_old = index_loc;
            tidx = blockSizeEx + index_loc;
            index_f = index;
            for (int t = 0; x - t >= t + 1; t++)
            {
                // calculate and store stencil
                stencilsum = (24.0 * A[tidx_old] - 2.0 * A[tidx_old - koff] - 2.0 * A[tidx_old + koff] -
                              2.0 * A[tidx_old - joff] - 2.0 * A[tidx_old + joff] - 2.0 * rm[tidx_old] -
                              2.0 * rp[tidx_old] - A[tidx_old - joff - koff] - A[tidx_old - joff + koff] -
                              A[tidx_old + joff - koff] - A[tidx_old + joff + koff] - rm[tidx_old - koff] -
                              rm[tidx_old + koff] - rp[tidx_old - koff] - rp[tidx_old + koff] - rm[tidx_old - joff] -
                              rm[tidx_old + joff] - rp[tidx_old - joff] - rp[tidx_old + joff]) *
                             h2inv;

                res = f[index_f] - stencilsum;
                rm[tidx] = A[tidx_old] + omega_dinv * res;
                barrier(CLK_LOCAL_MEM_FENCE);

                // if (get_group_id(0) == 0 && get_group_id(1) == 0 && get_local_id(0) == 0 && get_local_id(1) == 0 && x
                // == 1)
                // {
                //     printf("x = %d, res = %e, v_out = %e\n", x, res, rm[tidx]);
                //     print_7point(A, rm, rp, tidx_old, ioff, joff, koff);
                // }
                // if (get_group_id(0) == 0 && get_group_id(1) == 0 && get_local_id(0) == 2 && get_local_id(1) == 2)
                // printf("x = %d, t = %d -> writing to x = %d\n", x, t, x-t);

                // if ((get_local_id(0) >= (iterations-1) && get_local_id(0) < get_local_size(0) - (iterations-1) &&
                //     get_local_id(1) >= (iterations-1) && get_local_id(1) < get_local_size(1) - (iterations-1)) ||
                //     (get_group_id(0) == 0 && get_local_id(0) < get_local_size(0) - (iterations-1)) ||
                //     (get_group_id(1) == 0 && get_local_id(1) < get_local_size(1) - (iterations-1)) ||
                //     (get_group_id(0) == gridDim.x - 1 && get_local_id(0) >= (iterations-1)) ||
                //     (get_group_id(1) == gridDim.y - 1 && get_local_id(1) >= (iterations-1)))
                //     debug_out[index] = rm[tidx];

                // if (t == 1 && get_local_id(0) >= ghosts-1 && get_local_id(0) < get_local_size(0) - (ghosts-1) &&
                //     get_local_id(1) >= ghosts-1 && get_local_id(1) < get_local_size(1) - (ghosts-1))
                //     debug_out[index_f] = rm[tidx];

                // shift data
                tmp = rm;
                rm = A;
                A = rp;
                rp = tmp;
                cnt++;

                // update indices for next time step
                tidx_old = tidx;
                tidx += blockSizeEx;
                index_f -= ioff;
            }

            // buffers must be shifted backwards a multiple of 3 times plus 1 in total for the next plane, e.g.
            // 1,4,7,...
            for (cnt = (cnt - 1) % 3; cnt > 0; cnt--)
            {
                tmp = rp;
                rp = A;
                A = rm;
                rm = tmp;
            }

            index += ioff; // update index to next plane
        }

        // now stream for each x-plane that is left
        for (int x = iterations * 2 - 1; x < m - 1; x++)
        {
            load_plane_diag(v_in, rp, index + ioff, index_loc, joff, o);

            // calculate every time step but the last one and store result in shared memory
            tidx_old = index_loc;
            tidx = blockSizeEx + index_loc;
            index_f = index;
            for (int t = 0; t < iterations - 1; t++)
            {
                // calculate and store stencil
                stencilsum = (24.0 * A[tidx_old] - 2.0 * A[tidx_old - koff] - 2.0 * A[tidx_old + koff] -
                              2.0 * A[tidx_old - joff] - 2.0 * A[tidx_old + joff] - 2.0 * rm[tidx_old] -
                              2.0 * rp[tidx_old] - A[tidx_old - joff - koff] - A[tidx_old - joff + koff] -
                              A[tidx_old + joff - koff] - A[tidx_old + joff + koff] - rm[tidx_old - koff] -
                              rm[tidx_old + koff] - rp[tidx_old - koff] - rp[tidx_old + koff] - rm[tidx_old - joff] -
                              rm[tidx_old + joff] - rp[tidx_old - joff] - rp[tidx_old + joff]) *
                             h2inv;

                res = f[index_f] - stencilsum;
                rm[tidx] = A[tidx_old] + omega_dinv * res;
                barrier(CLK_LOCAL_MEM_FENCE);

                // if (get_group_id(0) == 0 && get_group_id(1) == 0 && get_local_id(0) == 2 && get_local_id(1) == 2 && x
                // == 5)
                // {
                //     printf("x = %d, res = %e, v_out = %e\n", x, res, rm[tidx]);
                //     print_7point(A, rm, rp, tidx_old, ioff, joff, koff);
                // }

                // if (t == 1 && get_local_id(0) >= ghosts-1 && get_local_id(0) < get_local_size(0) - (ghosts-1) &&
                //     get_local_id(1) >= ghosts-1 && get_local_id(1) < get_local_size(1) - (ghosts-1))
                //     debug_out[index_f] = rm[tidx];

                // shift data
                tmp = rm;
                rm = A;
                A = rp;
                rp = tmp;
                cnt++;

                // update indices for next time step
                tidx_old = tidx;
                tidx += blockSizeEx;
                index_f -= ioff;
            }

            // calculate last time step and store in global memory
            stencilsum =
                (24.0 * A[tidx_old] - 2.0 * A[tidx_old - koff] - 2.0 * A[tidx_old + koff] - 2.0 * A[tidx_old - joff] -
                 2.0 * A[tidx_old + joff] - 2.0 * rm[tidx_old] - 2.0 * rp[tidx_old] - A[tidx_old - joff - koff] -
                 A[tidx_old - joff + koff] - A[tidx_old + joff - koff] - A[tidx_old + joff + koff] -
                 rm[tidx_old - koff] - rm[tidx_old + koff] - rp[tidx_old - koff] - rp[tidx_old + koff] -
                 rm[tidx_old - joff] - rm[tidx_old + joff] - rp[tidx_old - joff] - rp[tidx_old + joff]) *
                h2inv;

            res = f[index_f] - stencilsum;

            // if (jreal == 1 && kreal == 1 && x >= 0 && x <= 6)
            // {
            //     printf("x = %d\tidx = %d\tres = %e\tv_out = %e\n", x, index_f, res, A[tidx_old] + omega_dinv * res);
            //     print_19point_local(A, rm, rp, tidx_old, ioff, joff, 1);
            // }

            // store result
            if (get_local_id(0) >= ghosts - 1 && get_local_id(0) < get_local_size(0) - (ghosts - 1) &&
                get_local_id(1) >= ghosts - 1 && get_local_id(1) < get_local_size(1) - (ghosts - 1))
                v_out[index_f] = A[tidx_old] + omega_dinv * res;

            barrier(CLK_LOCAL_MEM_FENCE);

            if (store_residual && get_local_id(0) >= ghosts - 1 && get_local_id(0) < get_local_size(0) - (ghosts - 1) &&
                get_local_id(1) >= ghosts - 1 && get_local_id(1) < get_local_size(1) - (ghosts - 1))
                r[index_out] = res;

            // buffers must be shifted backwards a multiple of 3 times plus 1 in total for the next plane, e.g.
            // 1,4,7,...
            for (cnt = (cnt - 1) % 3; cnt > 0; cnt--)
            {
                tmp = rp;
                rp = A;
                A = rm;
                rm = tmp;
            }

            index += ioff; // update index to next plane
            index_out += ioff_out;
            cnt = iterations == 1 ? 3 : 0; // TODO find better solution for shifting planes with special case t==1
        }
    }
}

/* runs iterations jacobi iterations on a 3D grid using shared memory using 27 point stencil.
 * uses algorithm of stencilgen.
 * x-plane of v_in needs to have 3 ghosted cells at each side to enable temporal tiling.
 * m,n,o must be size of v_in, including ghost cells.
 * uses shared memory only instead of registers.
 * debug_out has the same dimensions as v_in.
 * ghosts is the amount of ghost cells of v_in in one direction */
__kernel void jacobi_stream_shmem_27point(
    __global double* restrict v_in, // needed s.t. every work-item can read surrounding cell values
    __global double* restrict v_out, __global double* restrict f, __global double* restrict r, __local double* vlocal,
    const double h2inv, const double dinv, const double omega, const int m, const int n, const int o, const int ghosts,
    const int iterations, const int store_residual)
{
    // extended block dimensions (field stored in shared memory has one more row and column)
    int blockDimExX = get_local_size(0) + 2;
    int blockDimExY = get_local_size(1) + 2;
    int blockSizeEx = blockDimExX * blockDimExY;

    // index of real cell having block_ghosts = iterations-1. offset by +1 because input v has one more ghost row than
    // block
    int jreal = get_group_id(0) * get_local_size(0) - get_group_id(0) * (ghosts - 1) * 2 + get_local_id(0) + 1;
    int kreal = get_group_id(1) * get_local_size(1) - get_group_id(1) * (ghosts - 1) * 2 + get_local_id(1) + 1;

    // account for padding
    if (jreal < n && kreal < o)
    {
        int ioff = n * o;
        int ioff_out = (n - ghosts * 2) * (o - ghosts * 2); // size in x-direction without ghost cells
        int joff = blockDimExX;
        int koff = 1;
        int index = jreal * o + kreal; // global index of current cell (!= thread) in x-plane (for reading only)
        int index_out = (jreal - ghosts) * (o - ghosts * 2) + kreal - ghosts;
        int index_f = index; // seperate index for accessing F which is changing per time step
        // ghosted field has one more row than block, thus an offset must be added
        int index_loc = (get_local_id(0) + 1) * blockDimExX + get_local_id(1) +
                        1; // index of cell relative to current thread block (wg) + 2
        double stencilsum;
        double res;
        int tidx;
        int tidx_old;
        double omega_dinv = omega * dinv;
        int cnt = iterations == 1 ? 3 : 0; // counter for checking how many times plane buffers are shifted

        // store values of x-1, current x and x+1-planes in local memory, e.g. for t = 3, wg size = 12
        // size = (time steps + 1) * (block (wg) size + 2) = 4*14*14
        __local double* A = vlocal;
        __local double* rp = vlocal + blockSizeEx * iterations;
        __local double* rm =
            vlocal + blockSizeEx * 2 * iterations; // (time steps + 1) * (block (wg) size + 2) = 4*14*14
        __local double* tmp;

        // initialize local storage + registers
        load_plane_diag(v_in, rm, index, index_loc, joff, o);
        index += ioff;
        load_plane_diag(v_in, A, index, index_loc, joff, o);

        // cell index of next time step in shared memory
        tidx_old = index_loc;
        tidx = blockSizeEx + index_loc;

        // iterate over prologue planes
        for (int x = 1; x < iterations * 2 - 1; x++)
        {
            load_plane_diag(v_in, rp, index + ioff, index_loc, joff, o);

            // calculate higher iterations just like in the streaming part later but
            // stop earlier when for the current plane the current timestep does not exist
            tidx_old = index_loc;
            tidx = blockSizeEx + index_loc;
            index_f = index;
            for (int t = 0; x - t >= t + 1; t++)
            {
                // calculate and store stencil
                stencilsum =
                    (128.0 * A[tidx_old] - 14.0 * A[tidx_old - 1] - 14.0 * A[tidx_old + 1] - 14.0 * A[tidx_old - joff] -
                     14.0 * A[tidx_old + joff] - 14.0 * rm[tidx_old] - 14.0 * rp[tidx_old]

                     - 3.0 * A[tidx_old - joff - koff] - 3.0 * A[tidx_old - joff + koff] -
                     3.0 * A[tidx_old + joff - koff] - 3.0 * A[tidx_old + joff + koff] - 3.0 * rm[tidx_old - koff] -
                     3.0 * rm[tidx_old + koff] - 3.0 * rp[tidx_old - koff] - 3.0 * rp[tidx_old + koff] -
                     3.0 * rm[tidx_old - joff] - 3.0 * rm[tidx_old + joff] - 3.0 * rp[tidx_old - joff] -
                     3.0 * rp[tidx_old + joff]

                     - rm[tidx_old - joff - 1] - rm[tidx_old - joff + 1] - rm[tidx_old + joff - 1] -
                     rm[tidx_old + joff + 1] - rp[tidx_old - joff - 1] - rp[tidx_old - joff + 1] -
                     rp[tidx_old + joff - 1] - rp[tidx_old + joff + 1]) *
                    h2inv;

                res = f[index_f] - stencilsum;
                rm[tidx] = A[tidx_old] + omega_dinv * res;
                barrier(CLK_LOCAL_MEM_FENCE);

                // if (get_group_id(0) == 0 && get_group_id(1) == 0 && get_local_id(0) == 0 && get_local_id(1) == 0 && x
                // == 1)
                // {
                //     printf("x = %d, res = %e, v_out = %e\n", x, res, rm[tidx]);
                //     print_7point(A, rm, rp, tidx_old, ioff, joff, koff);
                // }
                // if (get_group_id(0) == 0 && get_group_id(1) == 0 && get_local_id(0) == 2 && get_local_id(1) == 2)
                // printf("x = %d, t = %d -> writing to x = %d\n", x, t, x-t);

                // if ((get_local_id(0) >= (iterations-1) && get_local_id(0) < get_local_size(0) - (iterations-1) &&
                //     get_local_id(1) >= (iterations-1) && get_local_id(1) < get_local_size(1) - (iterations-1)) ||
                //     (get_group_id(0) == 0 && get_local_id(0) < get_local_size(0) - (iterations-1)) ||
                //     (get_group_id(1) == 0 && get_local_id(1) < get_local_size(1) - (iterations-1)) ||
                //     (get_group_id(0) == gridDim.x - 1 && get_local_id(0) >= (iterations-1)) ||
                //     (get_group_id(1) == gridDim.y - 1 && get_local_id(1) >= (iterations-1)))
                //     debug_out[index] = rm[tidx];

                // if (t == 1 && get_local_id(0) >= ghosts-1 && get_local_id(0) < get_local_size(0) - (ghosts-1) &&
                //     get_local_id(1) >= ghosts-1 && get_local_id(1) < get_local_size(1) - (ghosts-1))
                //     debug_out[index_f] = rm[tidx];

                // shift data
                tmp = rm;
                rm = A;
                A = rp;
                rp = tmp;
                cnt++;

                // update indices for next time step
                tidx_old = tidx;
                tidx += blockSizeEx;
                index_f -= ioff;
            }

            // buffers must be shifted backwards a multiple of 3 times plus 1 in total for the next plane, e.g.
            // 1,4,7,...
            for (cnt = (cnt - 1) % 3; cnt > 0; cnt--)
            {
                tmp = rp;
                rp = A;
                A = rm;
                rm = tmp;
            }

            index += ioff; // update index to next plane
        }

        // now stream for each x-plane that is left
        for (int x = iterations * 2 - 1; x < m - 1; x++)
        {
            load_plane_diag(v_in, rp, index + ioff, index_loc, joff, o);

            // calculate every time step but the last one and store result in shared memory
            tidx_old = index_loc;
            tidx = blockSizeEx + index_loc;
            index_f = index;
            for (int t = 0; t < iterations - 1; t++)
            {
                // calculate and store stencil
                stencilsum =
                    (128.0 * A[tidx_old] - 14.0 * A[tidx_old - 1] - 14.0 * A[tidx_old + 1] - 14.0 * A[tidx_old - joff] -
                     14.0 * A[tidx_old + joff] - 14.0 * rm[tidx_old] - 14.0 * rp[tidx_old]

                     - 3.0 * A[tidx_old - joff - koff] - 3.0 * A[tidx_old - joff + koff] -
                     3.0 * A[tidx_old + joff - koff] - 3.0 * A[tidx_old + joff + koff] - 3.0 * rm[tidx_old - koff] -
                     3.0 * rm[tidx_old + koff] - 3.0 * rp[tidx_old - koff] - 3.0 * rp[tidx_old + koff] -
                     3.0 * rm[tidx_old - joff] - 3.0 * rm[tidx_old + joff] - 3.0 * rp[tidx_old - joff] -
                     3.0 * rp[tidx_old + joff]

                     - rm[tidx_old - joff - 1] - rm[tidx_old - joff + 1] - rm[tidx_old + joff - 1] -
                     rm[tidx_old + joff + 1] - rp[tidx_old - joff - 1] - rp[tidx_old - joff + 1] -
                     rp[tidx_old + joff - 1] - rp[tidx_old + joff + 1]) *
                    h2inv;

                res = f[index_f] - stencilsum;
                rm[tidx] = A[tidx_old] + omega_dinv * res;
                barrier(CLK_LOCAL_MEM_FENCE);

                // if (get_group_id(0) == 0 && get_group_id(1) == 0 && get_local_id(0) == 2 && get_local_id(1) == 2 && x
                // == 5)
                // {
                //     printf("x = %d, res = %e, v_out = %e\n", x, res, rm[tidx]);
                //     print_7point(A, rm, rp, tidx_old, ioff, joff, koff);
                // }

                // if (t == 1 && get_local_id(0) >= ghosts-1 && get_local_id(0) < get_local_size(0) - (ghosts-1) &&
                //     get_local_id(1) >= ghosts-1 && get_local_id(1) < get_local_size(1) - (ghosts-1))
                //     debug_out[index_f] = rm[tidx];

                // shift data
                tmp = rm;
                rm = A;
                A = rp;
                rp = tmp;
                cnt++;

                // update indices for next time step
                tidx_old = tidx;
                tidx += blockSizeEx;
                index_f -= ioff;
            }

            // calculate last time step and store in global memory
            stencilsum =
                (128.0 * A[tidx_old] - 14.0 * A[tidx_old - 1] - 14.0 * A[tidx_old + 1] - 14.0 * A[tidx_old - joff] -
                 14.0 * A[tidx_old + joff] - 14.0 * rm[tidx_old] - 14.0 * rp[tidx_old]

                 - 3.0 * A[tidx_old - joff - koff] - 3.0 * A[tidx_old - joff + koff] - 3.0 * A[tidx_old + joff - koff] -
                 3.0 * A[tidx_old + joff + koff] - 3.0 * rm[tidx_old - koff] - 3.0 * rm[tidx_old + koff] -
                 3.0 * rp[tidx_old - koff] - 3.0 * rp[tidx_old + koff] - 3.0 * rm[tidx_old - joff] -
                 3.0 * rm[tidx_old + joff] - 3.0 * rp[tidx_old - joff] - 3.0 * rp[tidx_old + joff]

                 - rm[tidx_old - joff - 1] - rm[tidx_old - joff + 1] - rm[tidx_old + joff - 1] -
                 rm[tidx_old + joff + 1] - rp[tidx_old - joff - 1] - rp[tidx_old - joff + 1] - rp[tidx_old + joff - 1] -
                 rp[tidx_old + joff + 1]) *
                h2inv;

            res = f[index_f] - stencilsum;

            // if (get_group_id(0) == 0 && get_group_id(1) == 0 && get_local_id(0) == 0 && get_local_id(1) == 5 && x ==
            // 1)
            // {
            //     printf("x = %d\tidx = %d\tres = %e\tv_out = %e\n", x, index_f, res, A[tidx_old] + omega_dinv * res);
            //     print_19point(A, rm, rp, tidx_old, ioff, joff, 1);
            // }

            // store result
            if (get_local_id(0) >= ghosts - 1 && get_local_id(0) < get_local_size(0) - (ghosts - 1) &&
                get_local_id(1) >= ghosts - 1 && get_local_id(1) < get_local_size(1) - (ghosts - 1))
                v_out[index_f] = A[tidx_old] + omega_dinv * res;

            barrier(CLK_LOCAL_MEM_FENCE);

            if (store_residual && get_local_id(0) >= ghosts - 1 && get_local_id(0) < get_local_size(0) - (ghosts - 1) &&
                get_local_id(1) >= ghosts - 1 && get_local_id(1) < get_local_size(1) - (ghosts - 1))
                r[index_out] = res;

            // buffers must be shifted backwards a multiple of 3 times plus 1 in total for the next plane, e.g.
            // 1,4,7,...
            for (cnt = (cnt - 1) % 3; cnt > 0; cnt--)
            {
                tmp = rp;
                rp = A;
                A = rm;
                rm = tmp;
            }

            index += ioff; // update index to next plane
            index_out += ioff_out;
            cnt = iterations == 1 ? 3 : 0; // TODO find better solution for shifting planes with special case t==1
        }
    }
}

/**
 * Updates ghosts of a varying stencil, respecting small grids, e.g. gh > m.
 * Needs to be called with one work-item per cell of ghosted grid.
 * Work-items that map to a real cell simply do nothing (optimization potential here!).
 */
__kernel void update_ghosts_varying_stencil(
    __global double* restrict c,
    int m, int n, int o,
    int width, int gh)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);

    if ((i < gh || j < gh || k < gh ||
         i >= gh + m || j >= gh + n || k >= gh + o) &&
        (i < m + 2 * gh && j < n + 2 * gh && k < o + 2 * gh))
    {
        int ireal = i + floor(((double)(gh - 1 - i)) / m + 1) * m;
        int jreal = j + floor(((double)(gh - 1 - j)) / n + 1) * n;
        int kreal = k + floor(((double)(gh - 1 - k)) / o + 1) * o;

        int gridsize = (m + 2 * gh) * (n + 2 * gh) * (o + 2 * gh);
        int idx_gh_cell = i * (n + 2 * gh) * (o + 2 * gh) + j * (o + 2 * gh) + k;
        int idx_real_cell = ireal * (n + 2 * gh) * (o + 2 * gh) + jreal * (o + 2 * gh) + kreal;

        // Iterate over every coefficient for the grid point this work-item maps to.
        for (int s = 0; s < width * width * width; s++)
        {
            c[idx_gh_cell] = c[idx_real_cell];
            idx_gh_cell += gridsize;
            idx_real_cell += gridsize;
        }
    }
}

/**
 * Multiplies two varying stencils c = a * b.
 * m, n and o are dimensions of the grid.
 * wa and wb are the widths of the stencils. Must be odd and >= 3.
 * gha, ghb and ghc are ghosts of the varying stencils at one border. ghb must be >= floor(wa / 2)
 * These restrictions are not checked in the kernel but shall be checked beforehand!
 * This kernel is supposed to be launched with one work-item per grid cell.
 * Remember to update ghosts of c if ghc > 0 afterwards.
 */
__kernel void mult_stencils_var_var(
    __global double* restrict a,
    __global double* restrict b,
    __global double* restrict c,
    int m, int n, int o,
    int wa, int wb,
    int gha, int ghb, int ghc)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);

    int wa2 = wa >> 1;
    int wc = wa + wb - 1;

    int gridsize_a = (m + 2 * gha) * (n + 2 * gha) * (o + 2 * gha);
    int gridsize_b = (m + 2 * ghb) * (n + 2 * ghb) * (o + 2 * ghb);
    int gridsize_c = (m + 2 * ghc) * (n + 2 * ghc) * (o + 2 * ghc);

    // 1d grid point index offset from the start of the coefficient list
    int wcPow2 = wc * wc;
    int wcPow3 = wcPow2 * wc;
    int cell_c = (i + ghc) * (n + 2 * ghc) * (o + 2 * ghc) + (j + ghc) * (o + 2 * ghc) + (k + ghc);

    int waPow2 = wa * wa;
    int waPow3 = waPow2 * wa;
    int cell_a = (i + gha) * (n + 2 * gha) * (o + 2 * gha) + (j + gha) * (o + 2 * gha) + (k + gha);

    int wbPow2 = wb * wb;
    int wbPow3 = wbPow2 * wb;

    if (i < m && j < n && k < o)
    {
        // clang-format off
        for (int ci = 0; ci < wc; ci++)
        for (int cj = 0; cj < wc; cj++)
        for (int ck = 0; ck < wc; ck++)
        {
            double csum = 0;
            for (int a_i = ci - (min(ci, wb - 1)), b_i = min(ci, wb - 1);
                a_i <= min(ci, wa - 1) && b_i >= ci - min(ci, wa - 1);
                a_i++, b_i--)
            for (int a_j = cj - (min(cj, wb - 1)), b_j = min(cj, wb - 1);
                    a_j <= min(cj, wa - 1) && b_j >= cj - min(cj, wa - 1);
                    a_j++, b_j--)
            for (int a_k = ck - (min(ck, wb - 1)), b_k = min(ck, wb - 1);
                    a_k <= min(ck, wa - 1) && b_k >= ck - min(ck, wa - 1);
                    a_k++, b_k--)
            {
                int gpi = i + a_i - wa2 + ghb;
                int gpj = j + a_j - wa2 + ghb;
                int gpk = k + a_k - wa2 + ghb;
                
                int cell_b = gpi * (n + 2 * ghb) * (o + 2 * ghb) + gpj * (o + 2 * ghb) + gpk;

                // index of a coefficient is equal to the index of the coefficient block plus grid point offset.
                int idx_a = cell_a + (a_i * waPow2 + a_j * wa + a_k) * gridsize_a;
                int idx_b = cell_b + (b_i * wbPow2 + b_j * wb + b_k) * gridsize_b;

                csum +=
                    a[idx_a] *
                    b[idx_b];
            }

            c[cell_c + (ci * wcPow2 + cj * wc + ck) * gridsize_c] = csum;
        }
        // clang-format on
    }
}

/**
 * Multiplies a varying stencil a with a fixed stencil b, i.e. c = a * b.
 * m, n and o are dimensions of the grid.
 * wa and wb are the widths of the stencils. Must be odd and >= 3.
 * gha and ghc are ghosts of the stencils at one border.
 * These restrictions are not checked in the kernel but shall be checked beforehand!
 * This kernel is supposed to be launched with m x n x o*wc*wc*wc work-items.
 * Remember to update ghosts of c if ghc > 0 afterwards.
 */
__kernel void mult_stencils_var_fix(
    __global double* restrict a,
    __global double* restrict b,
    __global double* restrict c,
    int m, int n, int o,
    int wa, int wb,
    int gha, int ghc)
{
    int wa2 = wa >> 1;
    int wc = wa + wb - 1;
    int wcPow2 = wc * wc;
    int wcPow3 = wcPow2 * wc;

    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2) / wcPow3;
    int ci = (get_global_id(2) / wcPow2) % wc;
    int cj = (get_global_id(2) / wc) % wc;
    int ck = get_global_id(2) % wc;

    int gridsize_a = (m + 2 * gha) * (n + 2 * gha) * (o + 2 * gha);
    int gridsize_c = (m + 2 * ghc) * (n + 2 * ghc) * (o + 2 * ghc);

    // 1d indices
    int cell_c = (i + ghc) * (n + 2 * ghc) * (o + 2 * ghc) + (j + ghc) * (o + 2 * ghc) + (k + ghc);

    int waPow2 = wa * wa;
    int waPow3 = waPow2 * wa;
    int cell_a = (i + gha) * (n + 2 * gha) * (o + 2 * gha) + (j + gha) * (o + 2 * gha) + (k + gha);

    int wbPow2 = wb * wb;
    int wbPow3 = wbPow2 * wb;

    if (i < m && j < n && k < o)
    {
        // clang-format off
        double csum = 0;
        for (int a_i = ci - (min(ci, wb - 1)), b_i = min(ci, wb - 1);
            a_i <= min(ci, wa - 1) && b_i >= ci - min(ci, wa - 1);
            a_i++, b_i--)
        for (int a_j = cj - (min(cj, wb - 1)), b_j = min(cj, wb - 1);
                a_j <= min(cj, wa - 1) && b_j >= cj - min(cj, wa - 1);
                a_j++, b_j--)
        for (int a_k = ck - (min(ck, wb - 1)), b_k = min(ck, wb - 1);
                a_k <= min(ck, wa - 1) && b_k >= ck - min(ck, wa - 1);
                a_k++, b_k--)
        {
            int idx_a = cell_a + (a_i * waPow2 + a_j * wa + a_k) * gridsize_a;

            csum +=
                a[idx_a] *
                b[b_i * wbPow2 + b_j * wb + b_k];
        }

        c[cell_c + (ci * wcPow2 + cj * wc + ck) * gridsize_c] = csum;
        // clang-format on
    }
}

/**
 * Multiplies a fixed stencil a with a varying stencilb, i.e. c = a * b.
 * m, n and o are dimensions of the grid.
 * wa and wb are the widths of the stencils. Must be odd and >= 3.
 * ghb and ghc are ghosts of the varying stencils at one border. ghb must be >= floor(wa / 2)
 * These restrictions are not checked in the kernel but shall be checked beforehand!
 * This kernel is supposed to be launched with m x n x o*wc*wc*wc work-items.
 * Remember to update ghosts of c if ghc > 0 afterwards.
 */
__kernel void mult_stencils_fix_var(
    __global double* restrict a,
    __global double* restrict b,
    __global double* restrict c,
    int m, int n, int o,
    int wa, int wb,
    int ghb, int ghc)
{
    int wa2 = wa >> 1;
    int wc = wa + wb - 1;
    int wcPow2 = wc * wc;
    int wcPow3 = wcPow2 * wc;

    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2) / wcPow3;
    int ci = (get_global_id(2) / wcPow2) % wc;
    int cj = (get_global_id(2) / wc) % wc;
    int ck = get_global_id(2) % wc;

    int gridsize_b = (m + 2 * ghb) * (n + 2 * ghb) * (o + 2 * ghb);
    int gridsize_c = (m + 2 * ghc) * (n + 2 * ghc) * (o + 2 * ghc);

    // 1d indices
    int cell_c = (i + ghc) * (n + 2 * ghc) * (o + 2 * ghc) + (j + ghc) * (o + 2 * ghc) + (k + ghc);

    int waPow2 = wa * wa;

    int wbPow2 = wb * wb;
    int wbPow3 = wbPow2 * wb;

    if (i < m && j < n && k < o)
    {
        // clang-format off
        double csum = 0;
        for (int a_i = ci - (min(ci, wb - 1)), b_i = min(ci, wb - 1);
            a_i <= min(ci, wa - 1) && b_i >= ci - min(ci, wa - 1);
            a_i++, b_i--)
        for (int a_j = cj - (min(cj, wb - 1)), b_j = min(cj, wb - 1);
                a_j <= min(cj, wa - 1) && b_j >= cj - min(cj, wa - 1);
                a_j++, b_j--)
        for (int a_k = ck - (min(ck, wb - 1)), b_k = min(ck, wb - 1);
                a_k <= min(ck, wa - 1) && b_k >= ck - min(ck, wa - 1);
                a_k++, b_k--)
        {
            int gpi = i + a_i - wa2 + ghb;
            int gpj = j + a_j - wa2 + ghb;
            int gpk = k + a_k - wa2 + ghb;

            int cell_b = gpi * (n + 2 * ghb) * (o + 2 * ghb) + gpj * (o + 2 * ghb) + gpk;
            int idx_b = cell_b + (b_i * wbPow2 + b_j * wb + b_k) * gridsize_b;

            csum +=
                a[a_i * waPow2 + a_j * wa + a_k] *
                b[idx_b];
        }

        c[cell_c + (ci * wcPow2 + cj * wc + ck) * gridsize_c] = csum;
        // clang-format on
    }
}

/**
 * Copies values from in which is of width 7 on a grid 2m x 2n x 2o to stencil out of width
 * 3 on grid m x n x o.
 * a_2h might have bigger sizes than the grid it represents, thus mout, nout and oout are needed.
 * Parameters:
 * in: 7x7x7 stencil on fine grid 2m x 2n x 2o
 * out: 3x3x3 stencil on coarse grid m x n x o
 * m, n, o: Extends of the coarse grid without ghosts.
 * ghin: Ghosts of the stencil on the fine grid.
 * ghout: Ghosts of the stencil on the coarse grid.
 * nghout, oghout: Extends of the stencil on the coarse grid including ghosts.
 */
__kernel void cut_stencils_w7_to_w3(
    __global double* restrict in,
    __global double* restrict out,
    const int m_fine, const int n_fine, const int o_fine,
    const int m_coarse, const int n_coarse, const int o_coarse,
    const int ghin, const int ghout,
    const int nghout, const int oghout)
{
    int i = get_global_id(0) + ghout;
    int j = get_global_id(1) + ghout;
    int k = get_global_id(2) + ghout;

    int i2 = get_global_id(0) * 2 + 1;
    int j2 = get_global_id(1) * 2 + 1;
    int k2 = get_global_id(2) * 2 + 1;

    int gridsize_fine = (m_fine + 2 * ghin) * (n_fine + 2 * ghin) * (o_fine + 2 * ghin);
    int gridsize_coarse = (m_coarse + 2 * ghout) * nghout * oghout;

    // grid point offsets
    int cell_fine = i2 * (n_fine + 2 * ghin) * (o_fine + 2 * ghin) + j2 * (o_fine + 2 * ghin) + k2;
    int cell_coarse = i * nghout * oghout + j * oghout + k;

    if (i < m_coarse + ghout && j < n_coarse + ghout && k < o_coarse + ghout)
    {
        // clang-format off
        for (int ii = 0, ii2 = 1; ii < 3; ii++, ii2 += 2)
        for (int jj = 0, jj2 = 1; jj < 3; jj++, jj2 += 2)
        for (int kk = 0, kk2 = 1; kk < 3; kk++, kk2 += 2)
        {
            out[cell_coarse] = in[cell_fine + (ii2 * 49 + jj2 * 7 + kk2) * gridsize_fine];
            cell_coarse += gridsize_coarse;
        }
        // clang-format on
    }
}

/*********************************************************
 ******************** Utility kernels ********************
 *********************************************************/

// Form partial sum of buf per work-group and write result into buf_local.
// For full sum of buf sum_finish must be enqueued after this kernel (so global memory gets synchronized between work-groups).
// Not that using barrier(CLK_GLOBAL_MEM_FENCE) does not work for this as it does not synchronize work-groups.
// Must be called with a 1-D kernel range with #work-items = 1/fractions * #elements in buf.
// num_elements must be half of #elements in buf.
// partial_sums's size must be equal to number of work-groups.
// buf_local's size must be equal to work-group size.
// fractions determines the size of mapping of work-items to num_elements, i.e. 4 means #wi = 1/4 * #elements
__kernel void sum_partial_global_eq_x_num_elements(
    __global double* restrict buf,
    __global double* restrict partial_sums,
    __local double* buf_local,
    int num_elements,
    int fractions)
{
    int i = get_global_id(0);
    int wg_size = get_local_size(0);
    int iloc = get_local_id(0);

    if (i < num_elements)
    {
        // copy buf of this work-item into local storage. Two values since #wi = num_elements / 2 + padding
        buf_local[iloc] = buf[i];

        for (int f = 1; f < fractions; f++)
            if (i + f * get_global_size(0) < num_elements)
                buf_local[iloc] += buf[i + f * get_global_size(0)];

        // sum up buf using parallel sum reduction, "a >> 1" == "a / 2" for int
        // TODO: ensure that stride is even (or handle odd strides)
        for (int stride = wg_size >> 1; stride > 0; stride >>= 1)
        {
            // synchronize local memory
            barrier(CLK_LOCAL_MEM_FENCE);

            // fold upper half onto lower half
            if (iloc < stride && iloc + stride < wg_size && iloc + stride < num_elements)
            {
                buf_local[iloc] += buf_local[iloc + stride];
            }
        }

        // write into output partial_sums
        if (iloc == 0)
        {
            partial_sums[get_group_id(0)] = buf_local[iloc];
        }
    }
}

// Sums buf_partial_sums and writes result into buf_sum. buf_partial_sums needs to be filled using sum_partial before using this kernel.
// Must be called with only one work-item which iterates over the partial sums.
__kernel void sum_finish(
    __global double* restrict buf_partial_sums,
    __global double* restrict buf_sum,
    int partial_sums_count)
{
    double sum = 0;

    for (int p = 0; p < partial_sums_count; p++)
        sum += buf_partial_sums[p];

    buf_sum[0] = sum;
}

// Form partial maximum of buf per work-group and write result into buf_local.
// For full maximum of buf max_finish must be enqueued after this kernel (so global memory gets synchronized between work-groups).
// Not that using barrier(CLK_GLOBAL_MEM_FENCE) does not work for this as it does not synchronize work-groups.
// Must be called with a 1-D kernel range with #work-items = 1/fractions * #elements in buf.
// num_elements must be half of #elements in buf.
// partial_max's size must be equal to number of work-groups.
// buf_local's size must be equal to work-group size.
// fractions determines the size of mapping of work-items to num_elements, i.e. 4 means #wi = 1/4 * #elements.
// Work-group size must be even or the reduction will not work, i.e. miss the last element of each wg.
__kernel void max_partial_global_eq_x_num_elements(
    __global double* restrict buf,
    __global double* restrict partial_max,
    __local double* buf_local,
    int num_elements,
    int fractions)
{
    int i = get_global_id(0);
    int wg_size = get_local_size(0);
    int iloc = get_local_id(0);

    if (i < num_elements)
    {
        // copy buf of this work-item into local storage. Two values since #wi = num_elements / 2 + padding
        buf_local[iloc] = buf[i];

        for (int f = 1; f < fractions; f++)
            if (i + f * get_global_size(0) < num_elements)
            {
                double next = buf[i + f * get_global_size(0)];
                if (next > buf_local[iloc])
                    buf_local[iloc] = next;
            }

        // find maximum using parallel reduction, "a >> 1" == "a / 2" for int
        // TODO: ensure that stride is even (or handle odd strides)
        for (int stride = wg_size >> 1; stride > 0; stride >>= 1)
        {
            // synchronize local memory
            barrier(CLK_LOCAL_MEM_FENCE);

            // fold upper half onto lower half
            if (iloc < stride && iloc + stride < wg_size && iloc + stride < num_elements)
            {
                double next = buf_local[iloc + stride];
                if (next > buf_local[iloc])
                    buf_local[iloc] = next;
            }
        }

        // write into output
        if (iloc == 0)
        {
            partial_max[get_group_id(0)] = buf_local[iloc];
        }
    }
}

// Finds maximum in buf_partial_max and writes result into buf_max. buf_partial_max needs to be filled using max_partial before using this kernel.
// Must be called with only one work-item which iterates over the partial maxima.
__kernel void max_finish(
    __global double* restrict buf_partial_max,
    __global double* restrict buf_max,
    int partial_max_count)
{
    double max = buf_partial_max[0];

    for (int p = 1; p < partial_max_count; p++)
    {
        double next = buf_partial_max[p];
        if (next > max)
            max = next;
    }

    buf_max[0] = max;
}

// Form partial maximum of buf per work-group and write result into buf_local.
// For full maximum of buf max_finish must be enqueued after this kernel (so global memory gets synchronized between work-groups).
// Not that using barrier(CLK_GLOBAL_MEM_FENCE) does not work for this as it does not synchronize work-groups.
// Must be called with a 1-D kernel range with #work-items = 1/fractions * #elements in buf.
// num_elements must be half of #elements in buf.
// partial_max's size must be equal to number of work-groups.
// buf_local's size must be equal to work-group size.
// fractions determines the size of mapping of work-items to num_elements, i.e. 4 means #wi = 1/4 * #elements.
// Work-group size must be even or the reduction will not work, i.e. miss the last element of each wg.
__kernel void max_abs_partial_global_eq_x_num_elements(
    __global double* restrict buf,
    __global double* restrict partial_max,
    __local double* buf_local,
    int num_elements,
    int fractions)
{
    int i = get_global_id(0);
    int wg_size = get_local_size(0);
    int iloc = get_local_id(0);

    if (i < num_elements)
    {
        // copy buf of this work-item into local storage. Two values since #wi = num_elements / 2 + padding
        buf_local[iloc] = fabs(buf[i]);

        for (int f = 1; f < fractions; f++)
            if (i + f * get_global_size(0) < num_elements)
            {
                double next = fabs(buf[i + f * get_global_size(0)]);
                if (next > buf_local[iloc])
                    buf_local[iloc] = next;
            }

        // find maximum using parallel reduction, "a >> 1" == "a / 2" for int
        // TODO: ensure that stride is even (or handle odd strides)
        for (int stride = wg_size >> 1; stride > 0; stride >>= 1)
        {
            // synchronize local memory
            barrier(CLK_LOCAL_MEM_FENCE);

            // fold upper half onto lower half
            if (iloc < stride && iloc + stride < wg_size && iloc + stride < num_elements)
            {
                double next = fabs(buf_local[iloc + stride]);
                if (next > buf_local[iloc])
                    buf_local[iloc] = next;
            }
        }

        // write into output
        if (iloc == 0)
        {
            partial_max[get_group_id(0)] = buf_local[iloc];
        }
    }
}
)DELIM";
}