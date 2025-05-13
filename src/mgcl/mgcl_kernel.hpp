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

// Blocksize for blockstencils, defaults to 2
#ifndef BLOCKSIZE
#define BLOCKSIZE 2
#endif

#ifndef BLOCKSIZE_POW_2
#define BLOCKSIZE_POW_2 (BLOCKSIZE * BLOCKSIZE)
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

/**
 * Updates ghosts of a CuboidBS, respecting small grids, e.g. gh > m.
 * Needs to be called with one work-item per cell of ghosted grid.
 * Work-items that map to a real cell simply do nothing (optimization potential here!).
 * m,n,o are sizes of real grid.
 * ghm, ghn, gho are amount of ghosts at one border.
 */
__kernel void update_ghosts_cuboidbs_periodic_blockstencil(
    __global double* restrict c,
    const int m, const int n, const int o,
    const int ghm, const int ghn, const int gho,
    const int blocksize)
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
        int idx_gh_cell = i * ngh * ogh * blocksize + j * ogh * blocksize + k * blocksize;
        int idx_real_cell = ireal * ngh * ogh * blocksize + jreal * ogh * blocksize + kreal * blocksize;

        // update ghost cell
        for (int b = 0; b < blocksize; b++)
            c[idx_gh_cell + b] = c[idx_real_cell + b];
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
            (88.0 * v_in[index] - 6.0 * v_in[index - koff] - 6.0 * v_in[index + koff] - 6.0 * v_in[index - joff] -
             6.0 * v_in[index + joff] - 6.0 * v_in[index - ioff] - 6.0 * v_in[index + ioff]

             - 3.0 * v_in[index - joff - koff] - 3.0 * v_in[index - joff + koff] - 3.0 * v_in[index + joff - koff] -
             3.0 * v_in[index + joff + koff] - 3.0 * v_in[index - ioff - koff] - 3.0 * v_in[index - ioff + koff] -
             3.0 * v_in[index + ioff - koff] - 3.0 * v_in[index + ioff + koff] - 3.0 * v_in[index - ioff - joff] -
             3.0 * v_in[index - ioff + joff] - 3.0 * v_in[index + ioff - joff] - 3.0 * v_in[index + ioff + joff]

             - 2.0 * v_in[index - ioff - joff - koff] - 2.0 * v_in[index - ioff - joff + koff] - 2.0 * v_in[index - ioff + joff - koff] -
             2.0 * v_in[index - ioff + joff + koff] - 2.0 * v_in[index + ioff - joff - koff] - 2.0 * v_in[index + ioff - joff + koff] -
             2.0 * v_in[index + ioff + joff - koff] - 2.0 * v_in[index + ioff + joff + koff]) *
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

/**
 * Calculates the residual r = f - A*v for a 27-point stencil.
 * Must be called with one wi per grid point that the residual shall be calculated for (only real grid points, if {m,n,o}off = 0)
 * Arguments:
 * - v_in: v
 * - f: f
 * - r: r
 * - mgh, ngh, ogh: ghosted grid dimensions
 * - ghosts: number of ghosts
 * - moff, noff, ooff: offsets for relevant cells. Can be used to enlarge or shrink the grid cells that the residual is calculated for.
 *     E.g. if real m=n=o=4 and gh=2, with moff=0, the residual is calculated for grid points m=2,...,5. With moff=-1, the residual is calculated for grid points m=1,...,6.
 * - c000 ... c222: coefficients for the 27-point stencil with respective index
 */
__kernel void residual_27point_fixed_stencil(
    __global double* restrict v_in, // needed s.t. every work-item can read surrounding cell values
    __global double* restrict f,
    __global double* restrict r,
    const int mgh, const int ngh, const int ogh,
    const int ghosts,
    const int moff, const int noff, const int ooff,
    const double c000,
    const double c001,
    const double c002,
    const double c010,
    const double c011,
    const double c012,
    const double c020,
    const double c021,
    const double c022,
    const double c100,
    const double c101,
    const double c102,
    const double c110,
    const double c111,
    const double c112,
    const double c120,
    const double c121,
    const double c122,
    const double c200,
    const double c201,
    const double c202,
    const double c210,
    const double c211,
    const double c212,
    const double c220,
    const double c221,
    const double c222)
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

        // A*v
        // clang-format off
        double stencilsum = c111 * v_in[index]
            + c110 * v_in[index - 1]
            + c112 * v_in[index + 1]
            + c101 * v_in[index - joff]
            + c121 * v_in[index + joff]
            + c011 * v_in[index - ioff]
            + c211 * v_in[index + ioff]
            
            + c100 * v_in[index - joff - koff]
            + c102 * v_in[index - joff + koff]
            + c120 * v_in[index + joff - koff]
            + c122 * v_in[index + joff + koff]
            + c010 * v_in[index - ioff - koff]
            + c012 * v_in[index - ioff + koff]
            + c210 * v_in[index + ioff - koff]
            + c212 * v_in[index + ioff + koff]
            + c001 * v_in[index - ioff - joff]
            + c021 * v_in[index - ioff + joff]
            + c201 * v_in[index + ioff - joff]
            + c221 * v_in[index + ioff + joff]

            + c000 * v_in[index - ioff - joff - koff]
            + c002 * v_in[index - ioff - joff + koff]
            + c020 * v_in[index - ioff + joff - koff]
            + c022 * v_in[index - ioff + joff + koff]
            + c200 * v_in[index + ioff - joff - koff]
            + c202 * v_in[index + ioff - joff + koff]
            + c220 * v_in[index + ioff + joff - koff]
            + c222 * v_in[index + ioff + joff + koff];
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

/* Calculates residual using a blockstencil.
 * Layout: [mx][my][cx][cy][cz][gpx][gpy][gpz] for coeffs, [gpx][gpy][gpz][m] for v, f, r
 *
 * One wi per grid point.
 *
 * svGridSize = sv_mgh * sv_ngh * sv_ogh
 * svGridSizeCoeffs = 27 * svGridSize
 */
__kernel void residual_27point_blockstencil_block_first_v_gp_first(
    __global double* restrict v_in, // needed s.t. every work-item can read surrounding cell values
    __global double* restrict f,
    __global double* restrict r,
    __global double* restrict stencilValues,
    const int mgh, const int ngh, const int ogh,
    const int svmgh, const int svngh, const int svogh,
    const int ghosts, const int ghosts_sv,
    const int svGridSize, const int svGridSizeCoeffs,
    const int blocksize,
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
        int ioff = blocksize * ngh * ogh;
        int joff = blocksize * ogh;
        int koff = blocksize;
        int index = i * ioff + j * joff + k * koff;
        int gridsize = mgh * ngh * ogh;

        int svno = svngh * svogh;
        // offset inside one coefficient grid that points to the coefficient for the current grid point. Must consider different amount of ghosts for v and sv.
        int index_sv_gp = (i - ghosts + ghosts_sv) * svno + (j - ghosts + ghosts_sv) * svogh + (k - ghosts + ghosts_sv);

        // if (i == 2 && j == 2 && k == 2)
        // {
        //     printf("i,j,k,mgh,ngh,ogh,gh,gh_sv,index_sv_gp,gridsize: %d,%d,%d,%d,%d,%d,%d,%d,%d,%d\ngh", i, j, k, mgh, ngh, ogh, ghosts, ghosts_sv, index_sv_gp, gridsize);
        // }

        // Layout: [cx][cy][cz][mx][my][gpx][gpy][gpz] for coeffs, [gpx][gpy][gpz][m] for v, f, r
        int idx_block = 0;
        for (int bi = 0; bi < blocksize; bi++)
        {
            double stencilsum = 0;

            for (int bj = 0; bj < blocksize; bj++)
            {
                // A*v
                // clang-format off
                stencilsum += stencilValues[index_sv_gp + (9 + 3 + 1) * svGridSize + idx_block] * v_in[index + bj]
                    + stencilValues[index_sv_gp + (9 + 3) * svGridSize + idx_block]      * v_in[index - koff + bj]
                    + stencilValues[index_sv_gp + (9 + 3 + 2) * svGridSize + idx_block]  * v_in[index + koff + bj]
                    + stencilValues[index_sv_gp + (9 + 1) * svGridSize + idx_block]      * v_in[index - joff + bj]
                    + stencilValues[index_sv_gp + (9 + 6 + 1) * svGridSize + idx_block]  * v_in[index + joff + bj]
                    + stencilValues[index_sv_gp + (3 + 1) * svGridSize + idx_block]      * v_in[index - ioff + bj]
                    + stencilValues[index_sv_gp + (18 + 3 + 1) * svGridSize + idx_block] * v_in[index + ioff + bj]
                    
                    + stencilValues[index_sv_gp + (9) * svGridSize + idx_block]          * v_in[index - joff - koff + bj]
                    + stencilValues[index_sv_gp + (9 + 2) * svGridSize + idx_block]      * v_in[index - joff + koff + bj]
                    + stencilValues[index_sv_gp + (9 + 6) * svGridSize + idx_block]      * v_in[index + joff - koff + bj]
                    + stencilValues[index_sv_gp + (9 + 6 + 2) * svGridSize + idx_block]  * v_in[index + joff + koff + bj]
                    + stencilValues[index_sv_gp + svGridSize * 3 + idx_block]            * v_in[index - ioff - koff + bj]
                    + stencilValues[index_sv_gp + (3 + 2) * svGridSize + idx_block]      * v_in[index - ioff + koff + bj]
                    + stencilValues[index_sv_gp + (18 + 3) * svGridSize + idx_block]     * v_in[index + ioff - koff + bj]
                    + stencilValues[index_sv_gp + (18 + 3 + 2) * svGridSize + idx_block] * v_in[index + ioff + koff + bj]
                    + stencilValues[index_sv_gp + svGridSize + idx_block]                * v_in[index - ioff - joff + bj]
                    + stencilValues[index_sv_gp + (6 + 1) * svGridSize + idx_block]      * v_in[index - ioff + joff + bj]
                    + stencilValues[index_sv_gp + (18 + 1) * svGridSize + idx_block]     * v_in[index + ioff - joff + bj]
                    + stencilValues[index_sv_gp + (18 + 6 + 1) * svGridSize + idx_block] * v_in[index + ioff + joff + bj]

                    + stencilValues[index_sv_gp + idx_block]                                  * v_in[index - ioff - joff - koff + bj]
                    + stencilValues[index_sv_gp + svGridSize * 2 + idx_block]            * v_in[index - ioff - joff + koff + bj]
                    + stencilValues[index_sv_gp + (6) * svGridSize + idx_block]          * v_in[index - ioff + joff - koff + bj]
                    + stencilValues[index_sv_gp + (6 + 2) * svGridSize + idx_block]      * v_in[index - ioff + joff + koff + bj]
                    + stencilValues[index_sv_gp + (18) * svGridSize + idx_block]         * v_in[index + ioff - joff - koff + bj]
                    + stencilValues[index_sv_gp + (18 + 2) * svGridSize + idx_block]     * v_in[index + ioff - joff + koff + bj]
                    + stencilValues[index_sv_gp + (18 + 6) * svGridSize + idx_block]     * v_in[index + ioff + joff - koff + bj]
                    + stencilValues[index_sv_gp + (18 + 6 + 2) * svGridSize + idx_block] * v_in[index + ioff + joff + koff + bj];
                // clang-format on

                idx_block += svGridSizeCoeffs; // increase by gridsize to get to next matrix entry
            }

            // r = f - A*v
            // printf("id, stencilsum, f[index + bi]: %i, %lf, %lf\n", idx, stencilsum, f[index + bi]);
            r[index + bi] = f[index + bi] - stencilsum;
            // printf("id, r: %i, %lf\n", idx, r[index + bi]);
        }

        // if (i == 2 && j == 2 && k == 2)
        // {
        //     printf("ocl stencilsum = %e\n", stencilsum);
        //     print27point_sv(v_in, index, ioff, joff, koff, stencilValues, index_sv_gp);
        // }
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

            // A*v 27-point laplacian stencil
            double stencilsum =
                (88.0 * v_in_index - 6.0 * v_in[index - koff] - 6.0 * v_in[index + koff] -
                 6.0 * v_in[index - joff] - 6.0 * v_in[index + joff] - 6.0 * v_in[index - ioff] -
                 6.0 * v_in[index + ioff]

                 - 3.0 * v_in[index - joff - koff] - 3.0 * v_in[index - joff + koff] - 3.0 * v_in[index + joff - koff] -
                 3.0 * v_in[index + joff + koff] - 3.0 * v_in[index - ioff - koff] - 3.0 * v_in[index - ioff + koff] -
                 3.0 * v_in[index + ioff - koff] - 3.0 * v_in[index + ioff + koff] - 3.0 * v_in[index - ioff - joff] -
                 3.0 * v_in[index - ioff + joff] - 3.0 * v_in[index + ioff - joff] - 3.0 * v_in[index + ioff + joff]

                 - 2.0 * v_in[index - ioff - joff - koff] - 2.0 * v_in[index - ioff - joff + koff] -
                 2.0 * v_in[index - ioff + joff - koff] - 2.0 * v_in[index - ioff + joff + koff] -
                 2.0 * v_in[index + ioff - joff - koff] - 2.0 * v_in[index + ioff - joff + koff] -
                 2.0 * v_in[index + ioff + joff - koff] - 2.0 * v_in[index + ioff + joff + koff]) *
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

/* runs one iteration of jacobi's method using one work-item per grid node for a fixed stencil.
 * uses a 1d kernel, which parallelizes all three loop in x,y and z directions.
 * global size must be of ghosted grid.
 * Arguments:
 * - v_in: input v, only read from
 * - v_out: output v, only written to
 * - f: rhs, only read from
 * - r: residual, only written to if store_residual is true. Else unused
 * - omega: relaxation parameter
 * - mgh, ngh,ogh: Dimensions of local ghosted grid
 * - store_residual: If true, the residual will be stored into global field r.
 * - ghosts: Amount of ghost cells of v, f and r.
 * - idx_start: Determines which cells shall be calculated, which is relevant for running
 *     Jacobi with multiple iterations without ghost cell update in-between. I.e. when
 *     stepsPerIter = 1: idx_start = ghosts.
 * - c000 ... c222: coefficients for the 27-point stencil with respective index
 */
__kernel void jacobi_iter_27point_fixed_stencil_1d(
    __global double* restrict v_in, // needed s.t. every work-item can read surrounding cell values
    __global double* restrict v_out,
    __global double* restrict f,
    __global double* restrict r,
    const double omega,
    const int mgh, const int ngh, const int ogh,
    const int ghosts,
    const int idx_start, const int store_residual,
    const double c000,
    const double c001,
    const double c002,
    const double c010,
    const double c011,
    const double c012,
    const double c020,
    const double c021,
    const double c022,
    const double c100,
    const double c101,
    const double c102,
    const double c110,
    const double c111,
    const double c112,
    const double c120,
    const double c121,
    const double c122,
    const double c200,
    const double c201,
    const double c202,
    const double c210,
    const double c211,
    const double c212,
    const double c220,
    const double c221,
    const double c222)
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

        double res;
        double v_in_index = v_in[index];

        // A*v
        // clang-format off
        double stencilsum = c111 * v_in_index
            + c110 * v_in[index - 1]
            + c112 * v_in[index + 1]
            + c101 * v_in[index - joff]
            + c121 * v_in[index + joff]
            + c011 * v_in[index - ioff]
            + c211 * v_in[index + ioff]
            
            + c100 * v_in[index - joff - koff]
            + c102 * v_in[index - joff + koff]
            + c120 * v_in[index + joff - koff]
            + c122 * v_in[index + joff + koff]
            + c010 * v_in[index - ioff - koff]
            + c012 * v_in[index - ioff + koff]
            + c210 * v_in[index + ioff - koff]
            + c212 * v_in[index + ioff + koff]
            + c001 * v_in[index - ioff - joff]
            + c021 * v_in[index - ioff + joff]
            + c201 * v_in[index + ioff - joff]
            + c221 * v_in[index + ioff + joff]

            + c000 * v_in[index - ioff - joff - koff]
            + c002 * v_in[index - ioff - joff + koff]
            + c020 * v_in[index - ioff + joff - koff]
            + c022 * v_in[index - ioff + joff + koff]
            + c200 * v_in[index + ioff - joff - koff]
            + c202 * v_in[index + ioff - joff + koff]
            + c220 * v_in[index + ioff + joff - koff]
            + c222 * v_in[index + ioff + joff + koff];
        // clang-format on

        // r = f - A*v
        res = f[index] - stencilsum;

        // u_(m+1) = u_(m) + omega * (D^-1) * r_(m)
        v_out[index] = v_in_index + omega * (1.0 / c111) * res;

        if (store_residual)
            r[index] = res;
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
 *
 * Layout: [mx][my][cx][cy][cz][gpx][gpy][gpz] for coeffs, [gpx][gpy][gpz][m] for v, f, r
 *
 * svGridSize = sv_mgh * sv_ngh * sv_ogh
 * svGridSizeCoeffs = 27 * svGridSize
 */
__kernel void jacobi_iter_27point_blockstencil_block_first_v_gp_first(
    __global double* restrict v_in, // needed s.t. every work-item can read surrounding cell values
    __global double* restrict v_out,
    __global double* restrict f,
    __global double* restrict r,
    __global double* restrict stencilValues,
    __global double* restrict bs_inv,
    const double omega,
    const int mgh, const int ngh, const int ogh,
    const int svmgh, const int svngh, const int svogh,
    const int ghosts, const int ghosts_sv,
    const int svGridSize, const int svGridSizeCoeffs,
    const int idx_start, const int store_residual,
    const int blocksize)
{
    int idx = get_global_id(0);
    int no = ngh * ogh;
    int i = idx / no;
    int j = (idx - i * no) / ogh;
    int k = idx % ogh;

    // calculate residual for real cells plus some ghost cells if stepsPerIter > 1.
    if (i >= idx_start && j >= idx_start && k >= idx_start && i < mgh - idx_start && j < ngh - idx_start && k < ogh - idx_start)
    {
        int ioff = blocksize * ngh * ogh;
        int joff = blocksize * ogh;
        int koff = blocksize;
        int index = i * ioff + j * joff + k * koff;
        int gridsize = mgh * ngh * ogh;

        int svno = svngh * svogh;
        // offset inside one coefficient grid that points to the coefficient for the current grid point. Must consider different amount of ghosts for v and sv.
        int index_sv_gp = (i - ghosts + ghosts_sv) * svno + (j - ghosts + ghosts_sv) * svogh + (k - ghosts + ghosts_sv);

        double res[4]; // TODO make it variable

        // Layout: [cx][cy][cz][mx][my][gpx][gpy][gpz] for coeffs, [gpx][gpy][gpz][m] for v, f, r
        int idx_block = 0;
        for (int bi = 0; bi < blocksize; bi++)
        {
            double stencilsum = 0;

            for (int bj = 0; bj < blocksize; bj++)
            {
                // A*v
                // clang-format off
                stencilsum += stencilValues[index_sv_gp + (9 + 3 + 1) * svGridSize + idx_block] * v_in[index + bj]
                    + stencilValues[index_sv_gp + (9 + 3) * svGridSize + idx_block]      * v_in[index - koff + bj]
                    + stencilValues[index_sv_gp + (9 + 3 + 2) * svGridSize + idx_block]  * v_in[index + koff + bj]
                    + stencilValues[index_sv_gp + (9 + 1) * svGridSize + idx_block]      * v_in[index - joff + bj]
                    + stencilValues[index_sv_gp + (9 + 6 + 1) * svGridSize + idx_block]  * v_in[index + joff + bj]
                    + stencilValues[index_sv_gp + (3 + 1) * svGridSize + idx_block]      * v_in[index - ioff + bj]
                    + stencilValues[index_sv_gp + (18 + 3 + 1) * svGridSize + idx_block] * v_in[index + ioff + bj]

                    + stencilValues[index_sv_gp + (9) * svGridSize + idx_block]          * v_in[index - joff - koff + bj]
                    + stencilValues[index_sv_gp + (9 + 2) * svGridSize + idx_block]      * v_in[index - joff + koff + bj]
                    + stencilValues[index_sv_gp + (9 + 6) * svGridSize + idx_block]      * v_in[index + joff - koff + bj]
                    + stencilValues[index_sv_gp + (9 + 6 + 2) * svGridSize + idx_block]  * v_in[index + joff + koff + bj]
                    + stencilValues[index_sv_gp + svGridSize * 3 + idx_block]            * v_in[index - ioff - koff + bj]
                    + stencilValues[index_sv_gp + (3 + 2) * svGridSize + idx_block]      * v_in[index - ioff + koff + bj]
                    + stencilValues[index_sv_gp + (18 + 3) * svGridSize + idx_block]     * v_in[index + ioff - koff + bj]
                    + stencilValues[index_sv_gp + (18 + 3 + 2) * svGridSize + idx_block] * v_in[index + ioff + koff + bj]
                    + stencilValues[index_sv_gp + svGridSize + idx_block]                * v_in[index - ioff - joff + bj]
                    + stencilValues[index_sv_gp + (6 + 1) * svGridSize + idx_block]      * v_in[index - ioff + joff + bj]
                    + stencilValues[index_sv_gp + (18 + 1) * svGridSize + idx_block]     * v_in[index + ioff - joff + bj]
                    + stencilValues[index_sv_gp + (18 + 6 + 1) * svGridSize + idx_block] * v_in[index + ioff + joff + bj]

                    + stencilValues[index_sv_gp + idx_block]                                  * v_in[index - ioff - joff - koff + bj]
                    + stencilValues[index_sv_gp + svGridSize * 2 + idx_block]            * v_in[index - ioff - joff + koff + bj]
                    + stencilValues[index_sv_gp + (6) * svGridSize + idx_block]          * v_in[index - ioff + joff - koff + bj]
                    + stencilValues[index_sv_gp + (6 + 2) * svGridSize + idx_block]      * v_in[index - ioff + joff + koff + bj]
                    + stencilValues[index_sv_gp + (18) * svGridSize + idx_block]         * v_in[index + ioff - joff - koff + bj]
                    + stencilValues[index_sv_gp + (18 + 2) * svGridSize + idx_block]     * v_in[index + ioff - joff + koff + bj]
                    + stencilValues[index_sv_gp + (18 + 6) * svGridSize + idx_block]     * v_in[index + ioff + joff - koff + bj]
                    + stencilValues[index_sv_gp + (18 + 6 + 2) * svGridSize + idx_block] * v_in[index + ioff + joff + koff + bj];
                // clang-format on

                idx_block += svGridSizeCoeffs; // increase by gridsize to get to next matrix entry
            }

            // r = f - A*v
            res[bi] = f[index + bi] - stencilsum;

            if (store_residual)
                r[index + bi] = res[bi];
        }

        // TODO fuse loops?
        int m = mgh - 2 * ghosts;
        int n = ngh - 2 * ghosts;
        int o = ogh - 2 * ghosts;
        // Assumption for index of bs_inv: No ghosts, width = 1
        int gridSizeBsInv = m * n * o;
        int idx_bs_inv = (i - ghosts) * n * o + (j - ghosts) * o + k - ghosts;
        for (int bi = 0; bi < blocksize; bi++)
        {
            double sum = 0;

            // calculate bs_inv * r
            for (int bj = 0; bj < blocksize; bj++)
            {
                sum += bs_inv[idx_bs_inv] * res[bj];

                idx_bs_inv += gridSizeBsInv;
            }

            // u_(m+1) = u_(m) + omega * (D^-1) * r_(m)
            v_out[index + bi] = v_in[index + bi] + omega * sum;
        }
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

/* Restricts from fine to coarse grid using blockstencils.
 * Needs to get called with m*n*o work-items.
 *
 * Layout: [mx][my][cx][cy][cz] for fbs, [gpx][gpy][gpz][m] for fine, coarse
 *
 * Arguments:
 * * m,n,o is size of ghosted coarse grid.
 * * fine and coarse must be of sizes of ghosted grids.
 * * gh_vals_coarse must be sizes of coarse grid's cuboids. Most of the time its equal to m,n,o but for
 *   the threshold-level when using MPI, the cuboid is bigger than the actual level would be.
 * * fbs: Fixed restriction blockstencil
 */
__kernel void restrict_to_coarse_blockstencil(
    __global double* restrict fine,
    __global double* restrict coarse,
    __global double* restrict fbs,
    const int m, const int n, const int o, const int ghosts,
    const int ngh_vals_coarse, const int ogh_vals_coarse,
    const int blocksize)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);
    int g2 = 2 * ghosts;

    if (i < m && j < n && k < o)
    {
        const int idx_c = (i + ghosts) * ngh_vals_coarse * ogh_vals_coarse * blocksize + (j + ghosts) * ogh_vals_coarse * blocksize + (k + ghosts) * blocksize;
        const int nf = (n - g2) * 2 + g2, of = (o - g2) * 2 + g2;
        const int i2 = i * 2 + ghosts + 1, j2 = j * 2 + ghosts + 1, k2 = k * 2 + ghosts + 1;
        const int idx_f_self = i2 * nf * of * blocksize + j2 * of * blocksize + k2 * blocksize;
        const int ioff_f = nf * of * blocksize;
        const int joff_f = of * blocksize;
        const int koff_f = blocksize;

        for (int bi = 0; bi < blocksize; bi++)
        {
            double sum = 0;

            for (int bj = 0; bj < blocksize; bj++)
            {
                int fbs_idx_self = bi * blocksize * 27 + bj * 27 + 1 * 9 + 1 * 3 + 1;
                sum +=
                    // 0.125 * fine[idx_f_self + bj] // self
                    fbs[fbs_idx_self] * fine[idx_f_self + bj] + // self
                    // direct neighbours
                    fbs[fbs_idx_self - 9] * fine[idx_f_self - ioff_f + bj] +
                    fbs[fbs_idx_self + 9] * fine[idx_f_self + ioff_f + bj] +
                    fbs[fbs_idx_self - 3] * fine[idx_f_self - joff_f + bj] +
                    fbs[fbs_idx_self + 3] * fine[idx_f_self + joff_f + bj] +
                    fbs[fbs_idx_self - 1] * fine[idx_f_self - koff_f + bj] +
                    fbs[fbs_idx_self + 1] * fine[idx_f_self + koff_f + bj] +
                    // edge midpoints xy-plane
                    fbs[fbs_idx_self - 9 - 3] * fine[idx_f_self - ioff_f - joff_f + bj] +
                    fbs[fbs_idx_self - 9 + 3] * fine[idx_f_self - ioff_f + joff_f + bj] +
                    fbs[fbs_idx_self + 9 - 3] * fine[idx_f_self + ioff_f - joff_f + bj] +
                    fbs[fbs_idx_self + 9 + 3] * fine[idx_f_self + ioff_f + joff_f + bj] +
                    // edge midpoints xz-plane
                    fbs[fbs_idx_self - 9 - 1] * fine[idx_f_self - ioff_f - koff_f + bj] +
                    fbs[fbs_idx_self - 9 + 1] * fine[idx_f_self - ioff_f + koff_f + bj] +
                    fbs[fbs_idx_self + 9 - 1] * fine[idx_f_self + ioff_f - koff_f + bj] +
                    fbs[fbs_idx_self + 9 + 1] * fine[idx_f_self + ioff_f + koff_f + bj] +
                    // edge midpoints yz-plane
                    fbs[fbs_idx_self - 3 - 1] * fine[idx_f_self - joff_f - koff_f + bj] +
                    fbs[fbs_idx_self - 3 + 1] * fine[idx_f_self - joff_f + koff_f + bj] +
                    fbs[fbs_idx_self + 3 - 1] * fine[idx_f_self + joff_f - koff_f + bj] +
                    fbs[fbs_idx_self + 3 + 1] * fine[idx_f_self + joff_f + koff_f + bj] +
                    // corners
                    fbs[fbs_idx_self - 9 - 3 - 1] * fine[idx_f_self - ioff_f - joff_f - koff_f + bj] +
                    fbs[fbs_idx_self - 9 - 3 + 1] * fine[idx_f_self - ioff_f - joff_f + koff_f + bj] +
                    fbs[fbs_idx_self - 9 + 3 - 1] * fine[idx_f_self - ioff_f + joff_f - koff_f + bj] +
                    fbs[fbs_idx_self - 9 + 3 + 1] * fine[idx_f_self - ioff_f + joff_f + koff_f + bj] +
                    fbs[fbs_idx_self + 9 - 3 - 1] * fine[idx_f_self + ioff_f - joff_f - koff_f + bj] +
                    fbs[fbs_idx_self + 9 - 3 + 1] * fine[idx_f_self + ioff_f - joff_f + koff_f + bj] +
                    fbs[fbs_idx_self + 9 + 3 - 1] * fine[idx_f_self + ioff_f + joff_f - koff_f + bj] +
                    fbs[fbs_idx_self + 9 + 3 + 1] * fine[idx_f_self + ioff_f + joff_f + koff_f + bj];
            }
            coarse[idx_c + bi] = sum;
        }
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

/* Prolongates from coarse to fine grid.
 * Needs to get called with #work-items = #nodes of coarse grid.
 * m,n,o is size of ghosted fine grid.
 * fine and coarse must be of sizes of ghosted grids.
 * gh_vals_coarse must be sizes of coarse grid's cuboids. Most of the time its equal to m/2,n/2,o/2 but for
 * the threshold-level when using MPI, the cuboid is bigger than the actual level would be. */
__kernel void prolongate_to_fine_blockstencil(
    __global double* restrict fine,
    __global double* restrict coarse,
    __global double* restrict fbs,
    const int mf, const int nf, const int of, const int ghosts,
    const int ngh_vals_coarse, const int ogh_vals_coarse,
    const int blocksize)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);
    int g2 = 2 * ghosts;

    const int mc = (mf - g2) / 2 + g2;
    const int nc = (nf - g2) / 2 + g2;
    const int oc = (of - g2) / 2 + g2;

    if (i > ghosts - 1 && i < mc - ghosts && j > ghosts - 1 && j < nc - ghosts && k > ghosts - 1 && k < oc - ghosts)
    {
        const int index_coarse = (i * ngh_vals_coarse * ogh_vals_coarse + j * ogh_vals_coarse + k) * blocksize;
        const int i2 = i * 2 - (ghosts - 1), j2 = j * 2 - (ghosts - 1), k2 = k * 2 - (ghosts - 1);

        const int ioff_f = nf * of * blocksize;
        const int joff_f = of * blocksize;
        const int koff_f = blocksize;
        const int ioff_c = nc * oc * blocksize;
        const int joff_c = oc * blocksize;
        const int koff_c = blocksize;
        const int index_fine_self = i2 * ioff_f + j2 * joff_f + k2 * koff_f;

        for (int bi = 0; bi < blocksize; bi++)
        {
            double sums[8] = {0};
            for (int bj = 0; bj < blocksize; bj++)
            {
                int fbs_idx_self = bi * blocksize * 27 + bj * 27 + 1 * 9 + 1 * 3 + 1;

                sums[0] += fbs[fbs_idx_self] * coarse[index_coarse + bj];

                sums[1] += fbs[fbs_idx_self - 1] * coarse[index_coarse + bj] + fbs[fbs_idx_self + 1] * coarse[index_coarse - koff_c + bj];

                sums[2] += fbs[fbs_idx_self - 3] * coarse[index_coarse + bj] + fbs[fbs_idx_self + 3] * coarse[index_coarse - joff_c + bj];

                sums[3] += fbs[fbs_idx_self - 9] * coarse[index_coarse + bj] + fbs[fbs_idx_self + 9] * coarse[index_coarse - ioff_c + bj];

                sums[4] += fbs[fbs_idx_self - 3 - 1] * coarse[index_coarse + bj] + fbs[fbs_idx_self - 3 + 1] * coarse[index_coarse - koff_c + bj] + fbs[fbs_idx_self + 3 - 1] * coarse[index_coarse - joff_c + bj] + fbs[fbs_idx_self + 3 + 1] * coarse[index_coarse - joff_c - koff_c + bj];

                sums[5] += fbs[fbs_idx_self - 9 - 1] * coarse[index_coarse + bj] + fbs[fbs_idx_self - 9 + 1] * coarse[index_coarse - koff_c + bj] + fbs[fbs_idx_self + 9 - 1] * coarse[index_coarse - ioff_c + bj] + fbs[fbs_idx_self + 9 + 1] * coarse[index_coarse - ioff_c - koff_c + bj];

                sums[6] += fbs[fbs_idx_self - 9 - 3] * coarse[index_coarse + bj] + fbs[fbs_idx_self - 9 + 3] * coarse[index_coarse - joff_c + bj] + fbs[fbs_idx_self + 9 - 3] * coarse[index_coarse - ioff_c + bj] + fbs[fbs_idx_self + 9 + 3] * coarse[index_coarse - ioff_c - joff_c + bj];

                sums[7] += fbs[fbs_idx_self - 9 - 3 - 1] * coarse[index_coarse + bj] +
                           fbs[fbs_idx_self - 9 - 3 + 1] * coarse[index_coarse - koff_c + bj] +
                           fbs[fbs_idx_self - 9 + 3 - 1] * coarse[index_coarse - joff_c + bj] +
                           fbs[fbs_idx_self - 9 + 3 + 1] * coarse[index_coarse - joff_c - koff_c + bj] +
                           fbs[fbs_idx_self + 9 - 3 - 1] * coarse[index_coarse - ioff_c + bj] +
                           fbs[fbs_idx_self + 9 - 3 + 1] * coarse[index_coarse - ioff_c - koff_c + bj] +
                           fbs[fbs_idx_self + 9 + 3 - 1] * coarse[index_coarse - ioff_c - joff_c + bj] +
                           fbs[fbs_idx_self + 9 + 3 + 1] * coarse[index_coarse - ioff_c - joff_c - koff_c + bj];
            }
            fine[index_fine_self + bi] = sums[0];
            fine[index_fine_self - koff_f + bi] = sums[1];
            fine[index_fine_self - joff_f + bi] = sums[2];
            fine[index_fine_self - ioff_f + bi] = sums[3];
            fine[index_fine_self - joff_f - koff_f + bi] = sums[4];
            fine[index_fine_self - ioff_f - koff_f + bi] = sums[5];
            fine[index_fine_self - ioff_f - joff_f + bi] = sums[6];
            fine[index_fine_self - ioff_f - joff_f - koff_f + bi] = sums[7];
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
 * Updates ghosts of a blockstencil, respecting small grids, e.g. gh > m.
 * Needs to be called with one work-item per cell of ghosted grid.
 * Work-items that map to a real cell simply do nothing (optimization potential here!).
 */
__kernel void update_ghosts_blockstencil(
    __global double* restrict c,
    const int m, const int n, const int o,
    const int width, const int blocksize, const int gh)
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

        // Iterate over every matrix entry and coefficient for the grid point this work-item maps to.
        for (int b = 0; b < blocksize * blocksize; b++)
        {
            for (int s = 0; s < width * width * width; s++)
            {
                c[idx_gh_cell] = c[idx_real_cell];
                idx_gh_cell += gridsize;
                idx_real_cell += gridsize;
            }
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

/*******************************************
 * Helper structs and functions for galerkin
 *******************************************/

// Helper struct for galerkin. Defines an interval with integer start and end.
typedef struct Interval
{
    int start;
    int end;
} Interval;

// Returns the intersection of two intervals or [-1,-1] if they don't overlap
Interval intersect(Interval a, Interval b)
{
    Interval ret;
    // Check if intervals overlap
    if (a.start <= b.end && b.start <= a.end)
    {
        // Calculate start and end points of intersection
        ret.start = (a.start > b.start) ? a.start : b.start;
        ret.end = (a.end < b.end) ? a.end : b.end;
        return ret;
    }
    else
    {
        // Intervals do not overlap
        ret.start = -1;
        ret.end = -1;
        return ret;
    }
};

// Helper struct for galerkin. Defines a grid point with integer 3d coordinates.
typedef struct Point
{
    int x;
    int y;
    int z;
} Point;

// Returns the stencil entry indices of the stencil sitting at locationOfStencil that maps to mapsTo.
// No check is done, if the mapping is possible, i.e. the returned value might be outside of range [0,2].
// The result is just the difference of the indices plus one, since the stencil entry indices start at 0
// and not at -1.
Point stencilEntryThatMapsTo(Point locationOfStencil, Point mapsTo)
{
    Point ret;
    ret.x = mapsTo.x - locationOfStencil.x + 1;
    ret.y = mapsTo.y - locationOfStencil.y + 1;
    ret.z = mapsTo.z - locationOfStencil.z + 1;
    return ret;
};

// Returns the grid point indices that is mapped to by the stencil entry of another point.
// stencilEntry must be 0-based, hence the substraction by 1.
Point pointMappedToByStencilEntry(Point locationOfStencil, Point stencilEntry)
{
    Point ret;
    ret.x = locationOfStencil.x + (stencilEntry.x - 1);
    ret.y = locationOfStencil.y + (stencilEntry.y - 1);
    ret.z = locationOfStencil.z + (stencilEntry.z - 1);
    return ret;
};

// Returns the point on the fine grid that is related to the coarse grid point, respecting ghost cells.
Point coarseToFine(Point p, int ghc, int ghf)
{
    Point ret;
    ret.x = (p.x - ghc) * 2 + 1 + ghf;
    ret.y = (p.y - ghc) * 2 + 1 + ghf;
    ret.z = (p.z - ghc) * 2 + 1 + ghf;
    return ret;
};

/**
 * Applies the Galerkin operator, calculating the stencils a_2h for the coarser grid, based on the stencils a_h on the fine grid.
 * Optimized version that does not need full stencil multiplication, but instead directly writes to the resulting stencils.
 *
 * Parallelizes the outermost 6 loops (each coarse grid point and each stencil entry), thus must be called as 1d kernel with
 *   (a_h_m / 2) * (a_h_n / 2) * (a_h_o / 2) * 27 work-items (real grid size).
 *
 * Parameters:
 * a_h: Varying 3x3x3 stencil on fine grid. Has size a_h_m * a_h_n * a_h_o.
 * a_2h: Varying 3x3x3 stencil on coarse grid. Size depends on level and mpiLevelThreshold (equals resm, resn, reso in host code).
 * r: Fixed 3x3x3 restriction stencil.
 * p: Fixed 3x3x3 prolongation stencil.
 * mgh_f, ngh_f, ogh_f: Extends of the fine grid with ghosts.
 * m_c_loc, n_c_loc, o_c_loc: Extends of the local coarse grid without ghosts, i.e. the points that get written into.
 * m_c_buf, n_c_buf, o_c_buf: Extends of the coarse grid buffer without ghosts. Only for calculation of the indices.
 *   This is only different from m_c_loc etc. when using MPI and on rank 0 and on the threshold level.
 * gh_f: Amount of ghosts of the stencil on the fine grid in one direction.
 * gh_c: Amount of ghosts of the stencil on the coarse grid in one direction.
 */
__kernel void galerkin(
    __global double* restrict a_h,
    __global double* restrict a_2h,
    __global double* restrict r,
    __global double* restrict p,
    const int mgh_f, const int ngh_f, const int ogh_f,
    const int m_c_loc, const int n_c_loc, const int o_c_loc,
    const int m_c_buf, const int n_c_buf, const int o_c_buf,
    const int gh_f, const int gh_c)
{
    int gridsize = m_c_loc * n_c_loc * o_c_loc;

    int idx = get_global_id(0) % gridsize;
    int no = n_c_loc * o_c_loc;
    int i = idx / no;
    int j = (idx - i * no) / o_c_loc;
    int k = idx % o_c_loc;

    int stencil_idx = get_global_id(0) / gridsize;
    int ii = stencil_idx / 9;
    int jj = (stencil_idx - ii * 9) / 3;
    int kk = stencil_idx % 3;

    // only for real cells of coarse grid
    i += gh_c;
    j += gh_c;
    k += gh_c;

    // plane and grid size of ghosted fine grid
    int nogh_f = ngh_f * ogh_f;
    int mnogh_f = mgh_f * nogh_f;

    // plane and grid size of ghosted coarse grid
    int nogh_c = (n_c_buf + 2 * gh_c) * (o_c_buf + 2 * gh_c);
    int mnogh_c = (m_c_buf + 2 * gh_c) * nogh_c;

    // Calculate only for real cells of coarse grid
    if (i < m_c_loc + gh_c && j < n_c_loc + gh_c && k < o_c_loc + gh_c && stencil_idx < 27)
    {
        // for (int i = a_2h->getGhostsM(); i < (a_h.getM() >> 1) + a_2h->getGhostsM(); i++)
        //     for (int j = a_2h->getGhostsN(); j < (a_h.getN() >> 1) + a_2h->getGhostsN(); j++)
        //         for (int k = a_2h->getGhostsO(); k < (a_h.getO() >> 1) + a_2h->getGhostsO(); k++)
        // for each stencil entry of the coarse grid point this work-item maps to
        {
            // calculate fine grid point indices
            Point gp_c = {i, j, k};
            Point gp_f = coarseToFine(gp_c, gh_c, gh_f);
            Point entry_gpc = {ii, jj, kk};
            Point entry_gpf = coarseToFine(
                pointMappedToByStencilEntry(gp_c, entry_gpc),
                gh_c, gh_f);

            // find intersection S_P of neighbouring points for entry_gpf with reach=1 and gp_f with reach=2
            Interval xa = {.start = gp_f.x - 2, .end = gp_f.x + 2};
            Interval ya = {.start = gp_f.y - 2, .end = gp_f.y + 2};
            Interval za = {.start = gp_f.z - 2, .end = gp_f.z + 2};
            Interval xb = {.start = entry_gpf.x - 1, .end = entry_gpf.x + 1};
            Interval yb = {.start = entry_gpf.y - 1, .end = entry_gpf.y + 1};
            Interval zb = {.start = entry_gpf.z - 1, .end = entry_gpf.z + 1};

            Interval S_P[3] = {
                intersect(xa, xb),
                intersect(ya, yb),
                intersect(za, zb),
            };

            // Start calc (R*A)*P
            double res = 0;

            // for each fine grid point gp_sp in S_P:
            for (int spi = S_P[0].start; spi <= S_P[0].end; spi++)
                for (int spj = S_P[1].start; spj <= S_P[1].end; spj++)
                    for (int spk = S_P[2].start; spk <= S_P[2].end; spk++)
                    {
                        Point gp_sp = {spi, spj, spk};
                        // tmp_p <- in stencil P located at gp_sp: Find stencil entry entry_p that maps to entry_gpf. Since
                        // gp_sp is in S_P, it is ensured that the stencil has a stencil entry that maps to entry_gpf.
                        Point tmp_p_indices = stencilEntryThatMapsTo(gp_sp, entry_gpf);
                        double tmp_p = p[tmp_p_indices.x * 9 + tmp_p_indices.y * 3 + tmp_p_indices.z];

                        // Start calc R*A
                        // find intersection S_R of neighbouring points for gp_f and gp_sp, both with reach=1
                        xa.start = gp_f.x - 1;
                        xa.end = gp_f.x + 1;
                        ya.start = gp_f.y - 1;
                        ya.end = gp_f.y + 1;
                        za.start = gp_f.z - 1;
                        za.end = gp_f.z + 1;
                        xb.start = spi - 1;
                        xb.end = spi + 1;
                        yb.start = spj - 1;
                        yb.end = spj + 1;
                        zb.start = spk - 1;
                        zb.end = spk + 1;

                        Interval S_R[3] = {
                            intersect(xa, xb),
                            intersect(ya, yb),
                            intersect(za, zb),
                        };

                        double sum = 0;
                        // for each fine grid point gp_sr in S_R:
                        for (int sri = S_R[0].start; sri <= S_R[0].end; sri++)
                            for (int srj = S_R[1].start; srj <= S_R[1].end; srj++)
                                for (int srk = S_R[2].start; srk <= S_R[2].end; srk++)
                                {
                                    Point gp_sr = {sri, srj, srk};
                                    // tmp_r <- in stencil R located at gp_f: Find stencil entry entry_r that maps to gp_sr
                                    Point tmp_r_indices = stencilEntryThatMapsTo(gp_f, gp_sr);
                                    double tmp_r = r[tmp_r_indices.x * 9 + tmp_r_indices.y * 3 + tmp_r_indices.z];
                                    // tmp_a <- in stencil A located at gp_sr: Find stencil entry that maps to gp_sp
                                    Point tmp_a_indices = stencilEntryThatMapsTo(gp_sr, gp_sp);

                                    // double tmp_a = a_h[tmp_a_indices.x][tmp_a_indices.y][tmp_a_indices.z][gp_sr.x][gp_sr.y][gp_sr.z];
                                    double tmp_a = a_h[tmp_a_indices.x * 9 * mnogh_f + tmp_a_indices.y * 3 * mnogh_f + tmp_a_indices.z * mnogh_f + gp_sr.x * nogh_f + gp_sr.y * ogh_f + gp_sr.z];
                                    // sum <- sum + tmp_r * tmp_a
                                    sum += tmp_r * tmp_a;
                                    // End calc R*A
                                }

                        //   res <- res + sum * tmp_p
                        res += sum * tmp_p;
                        // End calc (R*A)*P
                    }

            // store res in rap
            // (*a_2h)[ii][jj][kk][i][j][k] = res;
            a_2h[ii * 9 * mnogh_c + jj * 3 * mnogh_c + kk * mnogh_c + i * nogh_c + j * (o_c_buf + 2 * gh_c) + k] = res;
        }
    }
}

/**
 * Applies the Galerkin operator for FixedStencil
 *
 * Parallelizes the outermost 3 loops (each stencil entry), thus must be called as 1d kernel with
 *   27 work-items.
 *
 * Parameters:
 * a_h: Fixed 3x3x3 stencil on fine grid. Has size 27.
 * a_2h: Fixed 3x3x3 stencil on coarse grid. Has size 27.
 * r: Fixed 3x3x3 restriction stencil.
 * p: Fixed 3x3x3 prolongation stencil.
 */
__kernel void galerkin_fixed_stencil(
    __global double* restrict a_h,
    __global double* restrict a_2h,
    __global double* restrict r,
    __global double* restrict p)
{
    // TODO optimize, maybe get rid of intersection calcs
    int i = 1;
    int j = 1;
    int k = 1;

    int stencil_idx = get_global_id(0);
    int ii = stencil_idx / 9;
    int jj = (stencil_idx - ii * 9) / 3;
    int kk = stencil_idx % 3;

    // Calculate only for coefficients
    if (stencil_idx < 27)
    {
        // for each stencil entry of the coarse grid point this work-item maps to
        {
            // calculate fine grid point indices
            Point gp_c = {i, j, k};
            Point gp_f = coarseToFine(gp_c, 0, 0);
            Point entry_gpc = {ii, jj, kk};
            Point entry_gpf = coarseToFine(
                pointMappedToByStencilEntry(gp_c, entry_gpc),
                0, 0);

            // find intersection S_P of neighbouring points for entry_gpf with reach=1 and gp_f with reach=2
            Interval xa = {.start = gp_f.x - 2, .end = gp_f.x + 2};
            Interval ya = {.start = gp_f.y - 2, .end = gp_f.y + 2};
            Interval za = {.start = gp_f.z - 2, .end = gp_f.z + 2};
            Interval xb = {.start = entry_gpf.x - 1, .end = entry_gpf.x + 1};
            Interval yb = {.start = entry_gpf.y - 1, .end = entry_gpf.y + 1};
            Interval zb = {.start = entry_gpf.z - 1, .end = entry_gpf.z + 1};

            Interval S_P[3] = {
                intersect(xa, xb),
                intersect(ya, yb),
                intersect(za, zb),
            };

            // Start calc (R*A)*P
            double res = 0;

            // for each fine grid point gp_sp in S_P:
            for (int spi = S_P[0].start; spi <= S_P[0].end; spi++)
                for (int spj = S_P[1].start; spj <= S_P[1].end; spj++)
                    for (int spk = S_P[2].start; spk <= S_P[2].end; spk++)
                    {
                        Point gp_sp = {spi, spj, spk};
                        // tmp_p <- in stencil P located at gp_sp: Find stencil entry entry_p that maps to entry_gpf. Since
                        // gp_sp is in S_P, it is ensured that the stencil has a stencil entry that maps to entry_gpf.
                        Point tmp_p_indices = stencilEntryThatMapsTo(gp_sp, entry_gpf);
                        double tmp_p = p[tmp_p_indices.x * 9 + tmp_p_indices.y * 3 + tmp_p_indices.z];

                        // Start calc R*A
                        // find intersection S_R of neighbouring points for gp_f and gp_sp, both with reach=1
                        xa.start = gp_f.x - 1;
                        xa.end = gp_f.x + 1;
                        ya.start = gp_f.y - 1;
                        ya.end = gp_f.y + 1;
                        za.start = gp_f.z - 1;
                        za.end = gp_f.z + 1;
                        xb.start = spi - 1;
                        xb.end = spi + 1;
                        yb.start = spj - 1;
                        yb.end = spj + 1;
                        zb.start = spk - 1;
                        zb.end = spk + 1;

                        Interval S_R[3] = {
                            intersect(xa, xb),
                            intersect(ya, yb),
                            intersect(za, zb),
                        };

                        double sum = 0;
                        // for each fine grid point gp_sr in S_R:
                        for (int sri = S_R[0].start; sri <= S_R[0].end; sri++)
                            for (int srj = S_R[1].start; srj <= S_R[1].end; srj++)
                                for (int srk = S_R[2].start; srk <= S_R[2].end; srk++)
                                {
                                    Point gp_sr = {sri, srj, srk};
                                    // tmp_r <- in stencil R located at gp_f: Find stencil entry entry_r that maps to gp_sr
                                    Point tmp_r_indices = stencilEntryThatMapsTo(gp_f, gp_sr);
                                    double tmp_r = r[tmp_r_indices.x * 9 + tmp_r_indices.y * 3 + tmp_r_indices.z];
                                    // tmp_a <- in stencil A located at gp_sr: Find stencil entry that maps to gp_sp
                                    Point tmp_a_indices = stencilEntryThatMapsTo(gp_sr, gp_sp);

                                    // double tmp_a = a_h[tmp_a_indices.x][tmp_a_indices.y][tmp_a_indices.z][gp_sr.x][gp_sr.y][gp_sr.z];
                                    double tmp_a = a_h[tmp_a_indices.x * 9 + tmp_a_indices.y * 3 + tmp_a_indices.z];
                                    // sum <- sum + tmp_r * tmp_a
                                    sum += tmp_r * tmp_a;
                                    // End calc R*A
                                }

                        //   res <- res + sum * tmp_p
                        res += sum * tmp_p;
                        // End calc (R*A)*P
                    }

            // store res in rap
            // (*a_2h)[ii][jj][kk][i][j][k] = res;
            a_2h[ii * 9 + jj * 3 + kk] = res;
        }
    }
}

/**
 * Applies the Galerkin operator, calculating the stencils a_2h for the coarser grid, based on the stencils a_h on the fine grid.
 * Optimized version that does not need full stencil multiplication, but instead directly writes to the resulting stencils.
 *
 * Parallelizes the outermost 6 loops (each coarse grid point and each stencil entry), thus must be called as 1d kernel with
 *   (a_h_m / 2) * (a_h_n / 2) * (a_h_o / 2) * 27 work-items (real grid size).
 *
 * Parameters:
 * a_h: Varying 3x3x3 stencil on fine grid. Has size a_h_m * a_h_n * a_h_o.
 * a_2h: Varying 3x3x3 stencil on coarse grid. Size depends on level and mpiLevelThreshold (equals resm, resn, reso in host code).
 * r: Fixed 3x3x3 restriction stencil.
 * p: Fixed 3x3x3 prolongation stencil.
 * mgh_f, ngh_f, ogh_f: Extends of the fine grid with ghosts.
 * m_c_loc, n_c_loc, o_c_loc: Extends of the local coarse grid without ghosts, i.e. the points that get written into.
 * m_c_buf, n_c_buf, o_c_buf: Extends of the coarse grid buffer without ghosts. Only for calculation of the indices.
 *   This is only different from m_c_loc etc. when using MPI and on rank 0 and on the threshold level.
 * gh_f: Amount of ghosts of the stencil on the fine grid in one direction.
 * gh_c: Amount of ghosts of the stencil on the coarse grid in one direction.
 */
__kernel void galerkin_blockstencil(
    __global double* restrict a_h,
    __global double* restrict a_2h,
    __global double* restrict r,
    __global double* restrict p,
    const int mgh_f, const int ngh_f, const int ogh_f,
    const int m_c_loc, const int n_c_loc, const int o_c_loc,
    const int m_c_buf, const int n_c_buf, const int o_c_buf,
    const int gh_f, const int gh_c)
{
    int gridsize = m_c_loc * n_c_loc * o_c_loc;

    int idx = get_global_id(0) % gridsize;
    int no = n_c_loc * o_c_loc;
    int i = idx / no;
    int j = (idx - i * no) / o_c_loc;
    int k = idx % o_c_loc;

    int stencil_idx = get_global_id(0) / gridsize;
    int ii = stencil_idx / 9;
    int jj = (stencil_idx - ii * 9) / 3;
    int kk = stencil_idx % 3;

    // only for real cells of coarse grid
    i += gh_c;
    j += gh_c;
    k += gh_c;

    // plane and grid size of ghosted fine grid
    int nogh_f = ngh_f * ogh_f;
    int mnogh_f = mgh_f * nogh_f;

    // plane and grid size of ghosted coarse grid
    int nogh_c = (n_c_buf + 2 * gh_c) * (o_c_buf + 2 * gh_c);
    int mnogh_c = (m_c_buf + 2 * gh_c) * nogh_c;

    // Calculate only for real cells of coarse grid
    if (i < m_c_loc + gh_c && j < n_c_loc + gh_c && k < o_c_loc + gh_c && stencil_idx < 27)
    {
        // for (int i = a_2h->getGhostsM(); i < (a_h.getM() >> 1) + a_2h->getGhostsM(); i++)
        //     for (int j = a_2h->getGhostsN(); j < (a_h.getN() >> 1) + a_2h->getGhostsN(); j++)
        //         for (int k = a_2h->getGhostsO(); k < (a_h.getO() >> 1) + a_2h->getGhostsO(); k++)
        // for each stencil entry of the coarse grid point this work-item maps to
        {
            // calculate fine grid point indices
            Point gp_c = {i, j, k};
            Point gp_f = coarseToFine(gp_c, gh_c, gh_f);
            Point entry_gpc = {ii, jj, kk};
            Point entry_gpf = coarseToFine(
                pointMappedToByStencilEntry(gp_c, entry_gpc),
                gh_c, gh_f);

            // find intersection S_P of neighbouring points for entry_gpf with reach=1 and gp_f with reach=2
            Interval xa = {.start = gp_f.x - 2, .end = gp_f.x + 2};
            Interval ya = {.start = gp_f.y - 2, .end = gp_f.y + 2};
            Interval za = {.start = gp_f.z - 2, .end = gp_f.z + 2};
            Interval xb = {.start = entry_gpf.x - 1, .end = entry_gpf.x + 1};
            Interval yb = {.start = entry_gpf.y - 1, .end = entry_gpf.y + 1};
            Interval zb = {.start = entry_gpf.z - 1, .end = entry_gpf.z + 1};

            Interval S_P[3] = {
                intersect(xa, xb),
                intersect(ya, yb),
                intersect(za, zb),
            };

            // Start calc (R*A)*P
            double res[BLOCKSIZE_POW_2] = {0};

            // for each fine grid point gp_sp in S_P:
            for (int spi = S_P[0].start; spi <= S_P[0].end; spi++)
                for (int spj = S_P[1].start; spj <= S_P[1].end; spj++)
                    for (int spk = S_P[2].start; spk <= S_P[2].end; spk++)
                    {
                        Point gp_sp = {spi, spj, spk};
                        // tmp_p <- in stencil P located at gp_sp: Find stencil entry entry_p that maps to entry_gpf. Since
                        // gp_sp is in S_P, it is ensured that the stencil has a stencil entry that maps to entry_gpf.
                        Point tmp_p_indices = stencilEntryThatMapsTo(gp_sp, entry_gpf);
                        double tmp_p = p[tmp_p_indices.x * 9 + tmp_p_indices.y * 3 + tmp_p_indices.z];

                        // Start calc R*A
                        // find intersection S_R of neighbouring points for gp_f and gp_sp, both with reach=1
                        xa.start = gp_f.x - 1;
                        xa.end = gp_f.x + 1;
                        ya.start = gp_f.y - 1;
                        ya.end = gp_f.y + 1;
                        za.start = gp_f.z - 1;
                        za.end = gp_f.z + 1;
                        xb.start = spi - 1;
                        xb.end = spi + 1;
                        yb.start = spj - 1;
                        yb.end = spj + 1;
                        zb.start = spk - 1;
                        zb.end = spk + 1;

                        Interval S_R[3] = {
                            intersect(xa, xb),
                            intersect(ya, yb),
                            intersect(za, zb),
                        };

                        double sum[BLOCKSIZE_POW_2] = {0};
                        // for each fine grid point gp_sr in S_R:
                        for (int sri = S_R[0].start; sri <= S_R[0].end; sri++)
                            for (int srj = S_R[1].start; srj <= S_R[1].end; srj++)
                                for (int srk = S_R[2].start; srk <= S_R[2].end; srk++)
                                {
                                    Point gp_sr = {sri, srj, srk};
                                    // tmp_r <- in stencil R located at gp_f: Find stencil entry entry_r that maps to gp_sr
                                    Point tmp_r_indices = stencilEntryThatMapsTo(gp_f, gp_sr);
                                    // tmp_a <- in stencil A located at gp_sr: Find stencil entry that maps to gp_sp
                                    Point tmp_a_indices = stencilEntryThatMapsTo(gp_sr, gp_sp);

                                    //  calculate r * a first
                                    double ra[BLOCKSIZE_POW_2] = {0};
                                    for (size_t bi = 0; bi < BLOCKSIZE; bi++)
                                        for (size_t bj = 0; bj < BLOCKSIZE; bj++)
                                            for (size_t bk = 0; bk < BLOCKSIZE; bk++)
                                            {
                                                ra[bi * BLOCKSIZE + bj] +=
                                                    r[bi * 27 * BLOCKSIZE + bk * 27 + tmp_r_indices.x * 9 + tmp_r_indices.y * 3 + tmp_r_indices.z] *
                                                    a_h[bk * BLOCKSIZE * 27 * mnogh_f + bj * 27 * mnogh_f + tmp_a_indices.x * 9 * mnogh_f + tmp_a_indices.y * 3 * mnogh_f + tmp_a_indices.z * mnogh_f + bi * nogh_f + bj * ogh_f + bk];
                                            }

                                    // calc sum += ra
                                    for (size_t bi = 0; bi < BLOCKSIZE; bi++)
                                        for (size_t bj = 0; bj < BLOCKSIZE; bj++)
                                        {
                                            sum[bi * BLOCKSIZE + bj] += ra[bi * BLOCKSIZE + bj];
                                        }
                                    // End calc R*A
                                }

                        //   res <- res + sum * tmp_p
                        // Calculate sum * tmp_p first
                        double sum_mult_p[BLOCKSIZE_POW_2] = {0};
                        for (size_t bi = 0; bi < BLOCKSIZE; bi++)
                            for (size_t bj = 0; bj < BLOCKSIZE; bj++)
                                for (size_t bk = 0; bk < BLOCKSIZE; bk++)
                                {
                                    sum_mult_p[bi * BLOCKSIZE + bj] +=
                                        sum[bi * BLOCKSIZE + bk] *
                                        p[bk * 27 * BLOCKSIZE + bj * 27 + tmp_p_indices.x * 9 + tmp_p_indices.y * 3 + tmp_p_indices.z];
                                }

                        for (size_t bi = 0; bi < BLOCKSIZE; bi++)
                            for (size_t bj = 0; bj < BLOCKSIZE; bj++)
                            {
                                res[bi * BLOCKSIZE + bj] += sum_mult_p[bi * BLOCKSIZE + bj];
                            }
                        // End calc (R*A)*P
                    }

            // store res in rap
            for (size_t bi = 0; bi < BLOCKSIZE; bi++)
                for (size_t bj = 0; bj < BLOCKSIZE; bj++)
                {
                    a_2h[bi * BLOCKSIZE * 27 * mnogh_c + bj * 27 * mnogh_c + ii * 9 * mnogh_c + jj * 3 * mnogh_c + kk * mnogh_c + i * nogh_c + j * (o_c_buf + 2 * gh_c) + k] = res[bi * BLOCKSIZE + bj];
                }
        }
    }
}

/**
 * Applies the Galerkin operator, calculating the stencils a_2h for the coarser grid, based on the stencils a_h on the fine grid.
 * Handcrafted version derived from matrix multiplication trace. Only works for 3x3x3 stencils and fixed R and P. This is
 * somewhat analogous to what Hypre does.
 *
 * Parallelizes the loop over the coarse grid points, thus must be called as 1d kernel with
 *   (a_h_m / 2) * (a_h_n / 2) * (a_h_o / 2) work-items (real grid size).
 *
 * Parameters:
 * a_h: Varying 3x3x3 stencil on fine grid. Has size a_h_m * a_h_n * a_h_o.
 * a_2h: Varying 3x3x3 stencil on coarse grid. Size depends on level and mpiLevelThreshold (equals resm, resn, reso in host code).
 * r: Fixed 3x3x3 restriction stencil.
 * p: Fixed 3x3x3 prolongation stencil.
 * mgh_f, ngh_f, ogh_f: Extends of the fine grid with ghosts.
 * m_c_loc, n_c_loc, o_c_loc: Extends of the local coarse grid without ghosts, i.e. the points that get written into.
 * m_c_buf, n_c_buf, o_c_buf: Extends of the coarse grid buffer without ghosts. Only for calculation of the indices.
 *   This is only different from m_c_loc etc. when using MPI and on rank 0 and on the threshold level.
 * gh_f: Amount of ghosts of the stencil on the fine grid in one direction.
 * gh_c: Amount of ghosts of the stencil on the coarse grid in one direction.
 */
__kernel void galerkin_handcrafted(
    __global double* restrict a_h,
    __global double* restrict a_2h,
    __global double* restrict r,
    __global double* restrict p,
    const int mgh_f, const int ngh_f, const int ogh_f,
    const int m_c_loc, const int n_c_loc, const int o_c_loc,
    const int m_c_buf, const int n_c_buf, const int o_c_buf,
    const int gh_f, const int gh_c)
{
    int idx = get_global_id(0);
    int no = n_c_loc * o_c_loc;
    int ci = idx / no;
    int cj = (idx - ci * no) / o_c_loc;
    int ck = idx % o_c_loc;

    // fine grid point that corresponds to this wi's coarse grid point
    int fi = ci * 2 + 1;
    int fj = cj * 2 + 1;
    int fk = ck * 2 + 1;

    // only for real cells of coarse grid
    ci += gh_c;
    cj += gh_c;
    ck += gh_c;
    fi += gh_f;
    fj += gh_f;
    fk += gh_f;

    // plane and grid size of ghosted fine grid
    int nogh_f = ngh_f * ogh_f;
    int mnogh_f = mgh_f * nogh_f;

    // plane and grid size of ghosted coarse grid
    int nogh_c = (n_c_buf + 2 * gh_c) * (o_c_buf + 2 * gh_c);
    int mnogh_c = (m_c_buf + 2 * gh_c) * nogh_c;

    // Calculate only for real cells of coarse grid
    if (ci < m_c_loc + gh_c && cj < n_c_loc + gh_c && ck < o_c_loc + gh_c)
    {
        double ra[125];
        ra[0 * 25 + 0 * 5 + 0] = r[0 * 9 + 0 * 3 + 0] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + -1];
        ra[0 * 25 + 0 * 5 + 1] = r[0 * 9 + 0 * 3 + 0] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[0 * 9 + 0 * 3 + 1] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + 0];
        ra[0 * 25 + 0 * 5 + 2] = r[0 * 9 + 0 * 3 + 0] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[0 * 9 + 0 * 3 + 1] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[0 * 9 + 0 * 3 + 2] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + 1];
        ra[0 * 25 + 0 * 5 + 3] = r[0 * 9 + 0 * 3 + 1] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[0 * 9 + 0 * 3 + 2] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + 1];
        ra[0 * 25 + 0 * 5 + 4] = r[0 * 9 + 0 * 3 + 2] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + 1];
        ra[0 * 25 + 1 * 5 + 0] = r[0 * 9 + 0 * 3 + 0] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[0 * 9 + 1 * 3 + 0] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + -1];
        ra[0 * 25 + 1 * 5 + 1] = r[0 * 9 + 0 * 3 + 0] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[0 * 9 + 0 * 3 + 1] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[0 * 9 + 1 * 3 + 0] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[0 * 9 + 1 * 3 + 1] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + 0];
        ra[0 * 25 + 1 * 5 + 2] = r[0 * 9 + 0 * 3 + 0] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[0 * 9 + 0 * 3 + 1] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[0 * 9 + 0 * 3 + 2] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[0 * 9 + 1 * 3 + 0] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[0 * 9 + 1 * 3 + 1] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[0 * 9 + 1 * 3 + 2] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + 1];
        ra[0 * 25 + 1 * 5 + 3] = r[0 * 9 + 0 * 3 + 1] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[0 * 9 + 0 * 3 + 2] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[0 * 9 + 1 * 3 + 1] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[0 * 9 + 1 * 3 + 2] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + 1];
        ra[0 * 25 + 1 * 5 + 4] = r[0 * 9 + 0 * 3 + 2] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[0 * 9 + 1 * 3 + 2] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + 1];
        ra[0 * 25 + 2 * 5 + 0] = r[0 * 9 + 0 * 3 + 0] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[0 * 9 + 1 * 3 + 0] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[0 * 9 + 2 * 3 + 0] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + -1];
        ra[0 * 25 + 2 * 5 + 1] = r[0 * 9 + 0 * 3 + 0] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[0 * 9 + 0 * 3 + 1] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[0 * 9 + 1 * 3 + 0] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[0 * 9 + 1 * 3 + 1] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[0 * 9 + 2 * 3 + 0] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[0 * 9 + 2 * 3 + 1] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + 0];
        ra[0 * 25 + 2 * 5 + 2] = r[0 * 9 + 0 * 3 + 0] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[0 * 9 + 0 * 3 + 1] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[0 * 9 + 0 * 3 + 2] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[0 * 9 + 1 * 3 + 0] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[0 * 9 + 1 * 3 + 1] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[0 * 9 + 1 * 3 + 2] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[0 * 9 + 2 * 3 + 0] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[0 * 9 + 2 * 3 + 1] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[0 * 9 + 2 * 3 + 2] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + 1];
        ra[0 * 25 + 2 * 5 + 3] = r[0 * 9 + 0 * 3 + 1] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[0 * 9 + 0 * 3 + 2] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[0 * 9 + 1 * 3 + 1] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[0 * 9 + 1 * 3 + 2] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[0 * 9 + 2 * 3 + 1] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[0 * 9 + 2 * 3 + 2] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + 1];
        ra[0 * 25 + 2 * 5 + 4] = r[0 * 9 + 0 * 3 + 2] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[0 * 9 + 1 * 3 + 2] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[0 * 9 + 2 * 3 + 2] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + 1];
        ra[0 * 25 + 3 * 5 + 0] = r[0 * 9 + 1 * 3 + 0] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[0 * 9 + 2 * 3 + 0] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + -1];
        ra[0 * 25 + 3 * 5 + 1] = r[0 * 9 + 1 * 3 + 0] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[0 * 9 + 1 * 3 + 1] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[0 * 9 + 2 * 3 + 0] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[0 * 9 + 2 * 3 + 1] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + 0];
        ra[0 * 25 + 3 * 5 + 2] = r[0 * 9 + 1 * 3 + 0] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[0 * 9 + 1 * 3 + 1] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[0 * 9 + 1 * 3 + 2] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[0 * 9 + 2 * 3 + 0] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[0 * 9 + 2 * 3 + 1] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[0 * 9 + 2 * 3 + 2] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + 1];
        ra[0 * 25 + 3 * 5 + 3] = r[0 * 9 + 1 * 3 + 1] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[0 * 9 + 1 * 3 + 2] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[0 * 9 + 2 * 3 + 1] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[0 * 9 + 2 * 3 + 2] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + 1];
        ra[0 * 25 + 3 * 5 + 4] = r[0 * 9 + 1 * 3 + 2] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[0 * 9 + 2 * 3 + 2] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + 1];
        ra[0 * 25 + 4 * 5 + 0] = r[0 * 9 + 2 * 3 + 0] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + -1];
        ra[0 * 25 + 4 * 5 + 1] = r[0 * 9 + 2 * 3 + 0] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[0 * 9 + 2 * 3 + 1] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + 0];
        ra[0 * 25 + 4 * 5 + 2] = r[0 * 9 + 2 * 3 + 0] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[0 * 9 + 2 * 3 + 1] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[0 * 9 + 2 * 3 + 2] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + 1];
        ra[0 * 25 + 4 * 5 + 3] = r[0 * 9 + 2 * 3 + 1] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[0 * 9 + 2 * 3 + 2] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + 1];
        ra[0 * 25 + 4 * 5 + 4] = r[0 * 9 + 2 * 3 + 2] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + 1];
        ra[1 * 25 + 0 * 5 + 0] = r[0 * 9 + 0 * 3 + 0] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[1 * 9 + 0 * 3 + 0] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + -1];
        ra[1 * 25 + 0 * 5 + 1] = r[0 * 9 + 0 * 3 + 0] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[0 * 9 + 0 * 3 + 1] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[1 * 9 + 0 * 3 + 0] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[1 * 9 + 0 * 3 + 1] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + 0];
        ra[1 * 25 + 0 * 5 + 2] = r[0 * 9 + 0 * 3 + 0] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[0 * 9 + 0 * 3 + 1] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[0 * 9 + 0 * 3 + 2] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[1 * 9 + 0 * 3 + 0] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[1 * 9 + 0 * 3 + 1] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[1 * 9 + 0 * 3 + 2] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + 1];
        ra[1 * 25 + 0 * 5 + 3] = r[0 * 9 + 0 * 3 + 1] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[0 * 9 + 0 * 3 + 2] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[1 * 9 + 0 * 3 + 1] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[1 * 9 + 0 * 3 + 2] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + 1];
        ra[1 * 25 + 0 * 5 + 4] = r[0 * 9 + 0 * 3 + 2] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[1 * 9 + 0 * 3 + 2] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + 1];
        ra[1 * 25 + 1 * 5 + 0] = r[0 * 9 + 0 * 3 + 0] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[0 * 9 + 1 * 3 + 0] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[1 * 9 + 0 * 3 + 0] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[1 * 9 + 1 * 3 + 0] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + -1];
        ra[1 * 25 + 1 * 5 + 1] = r[0 * 9 + 0 * 3 + 0] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[0 * 9 + 0 * 3 + 1] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[0 * 9 + 1 * 3 + 0] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[0 * 9 + 1 * 3 + 1] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[1 * 9 + 0 * 3 + 0] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[1 * 9 + 0 * 3 + 1] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[1 * 9 + 1 * 3 + 0] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[1 * 9 + 1 * 3 + 1] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + 0];
        ra[1 * 25 + 1 * 5 + 2] = r[0 * 9 + 0 * 3 + 0] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[0 * 9 + 0 * 3 + 1] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[0 * 9 + 0 * 3 + 2] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[0 * 9 + 1 * 3 + 0] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[0 * 9 + 1 * 3 + 1] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[0 * 9 + 1 * 3 + 2] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[1 * 9 + 0 * 3 + 0] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[1 * 9 + 0 * 3 + 1] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[1 * 9 + 0 * 3 + 2] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[1 * 9 + 1 * 3 + 0] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[1 * 9 + 1 * 3 + 1] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[1 * 9 + 1 * 3 + 2] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + 1];
        ra[1 * 25 + 1 * 5 + 3] = r[0 * 9 + 0 * 3 + 1] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[0 * 9 + 0 * 3 + 2] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[0 * 9 + 1 * 3 + 1] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[0 * 9 + 1 * 3 + 2] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[1 * 9 + 0 * 3 + 1] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[1 * 9 + 0 * 3 + 2] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[1 * 9 + 1 * 3 + 1] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[1 * 9 + 1 * 3 + 2] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + 1];
        ra[1 * 25 + 1 * 5 + 4] = r[0 * 9 + 0 * 3 + 2] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[0 * 9 + 1 * 3 + 2] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[1 * 9 + 0 * 3 + 2] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[1 * 9 + 1 * 3 + 2] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + 1];
        ra[1 * 25 + 2 * 5 + 0] = r[0 * 9 + 0 * 3 + 0] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[0 * 9 + 1 * 3 + 0] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[0 * 9 + 2 * 3 + 0] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[1 * 9 + 0 * 3 + 0] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[1 * 9 + 1 * 3 + 0] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[1 * 9 + 2 * 3 + 0] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + -1];
        ra[1 * 25 + 2 * 5 + 1] = r[0 * 9 + 0 * 3 + 0] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[0 * 9 + 0 * 3 + 1] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[0 * 9 + 1 * 3 + 0] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[0 * 9 + 1 * 3 + 1] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[0 * 9 + 2 * 3 + 0] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[0 * 9 + 2 * 3 + 1] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[1 * 9 + 0 * 3 + 0] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[1 * 9 + 0 * 3 + 1] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[1 * 9 + 1 * 3 + 0] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[1 * 9 + 1 * 3 + 1] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[1 * 9 + 2 * 3 + 0] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[1 * 9 + 2 * 3 + 1] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + 0];
        ra[1 * 25 + 2 * 5 + 2] = r[0 * 9 + 0 * 3 + 0] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[0 * 9 + 0 * 3 + 1] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[0 * 9 + 0 * 3 + 2] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[0 * 9 + 1 * 3 + 0] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[0 * 9 + 1 * 3 + 1] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[0 * 9 + 1 * 3 + 2] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[0 * 9 + 2 * 3 + 0] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[0 * 9 + 2 * 3 + 1] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[0 * 9 + 2 * 3 + 2] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + 1] + r[1 * 9 + 0 * 3 + 0] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[1 * 9 + 0 * 3 + 1] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[1 * 9 + 0 * 3 + 2] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[1 * 9 + 1 * 3 + 0] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[1 * 9 + 1 * 3 + 1] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[1 * 9 + 1 * 3 + 2] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[1 * 9 + 2 * 3 + 0] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[1 * 9 + 2 * 3 + 1] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[1 * 9 + 2 * 3 + 2] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + 1];
        ra[1 * 25 + 2 * 5 + 3] = r[0 * 9 + 0 * 3 + 1] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[0 * 9 + 0 * 3 + 2] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[0 * 9 + 1 * 3 + 1] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[0 * 9 + 1 * 3 + 2] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[0 * 9 + 2 * 3 + 1] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[0 * 9 + 2 * 3 + 2] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + 1] + r[1 * 9 + 0 * 3 + 1] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[1 * 9 + 0 * 3 + 2] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[1 * 9 + 1 * 3 + 1] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[1 * 9 + 1 * 3 + 2] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[1 * 9 + 2 * 3 + 1] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[1 * 9 + 2 * 3 + 2] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + 1];
        ra[1 * 25 + 2 * 5 + 4] = r[0 * 9 + 0 * 3 + 2] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[0 * 9 + 1 * 3 + 2] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[0 * 9 + 2 * 3 + 2] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + 1] + r[1 * 9 + 0 * 3 + 2] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[1 * 9 + 1 * 3 + 2] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[1 * 9 + 2 * 3 + 2] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + 1];
        ra[1 * 25 + 3 * 5 + 0] = r[0 * 9 + 1 * 3 + 0] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[0 * 9 + 2 * 3 + 0] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[1 * 9 + 1 * 3 + 0] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[1 * 9 + 2 * 3 + 0] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + -1];
        ra[1 * 25 + 3 * 5 + 1] = r[0 * 9 + 1 * 3 + 0] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[0 * 9 + 1 * 3 + 1] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[0 * 9 + 2 * 3 + 0] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[0 * 9 + 2 * 3 + 1] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[1 * 9 + 1 * 3 + 0] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[1 * 9 + 1 * 3 + 1] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[1 * 9 + 2 * 3 + 0] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[1 * 9 + 2 * 3 + 1] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + 0];
        ra[1 * 25 + 3 * 5 + 2] = r[0 * 9 + 1 * 3 + 0] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[0 * 9 + 1 * 3 + 1] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[0 * 9 + 1 * 3 + 2] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[0 * 9 + 2 * 3 + 0] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[0 * 9 + 2 * 3 + 1] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[0 * 9 + 2 * 3 + 2] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + 1] + r[1 * 9 + 1 * 3 + 0] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[1 * 9 + 1 * 3 + 1] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[1 * 9 + 1 * 3 + 2] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[1 * 9 + 2 * 3 + 0] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[1 * 9 + 2 * 3 + 1] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[1 * 9 + 2 * 3 + 2] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + 1];
        ra[1 * 25 + 3 * 5 + 3] = r[0 * 9 + 1 * 3 + 1] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[0 * 9 + 1 * 3 + 2] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[0 * 9 + 2 * 3 + 1] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[0 * 9 + 2 * 3 + 2] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + 1] + r[1 * 9 + 1 * 3 + 1] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[1 * 9 + 1 * 3 + 2] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[1 * 9 + 2 * 3 + 1] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[1 * 9 + 2 * 3 + 2] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + 1];
        ra[1 * 25 + 3 * 5 + 4] = r[0 * 9 + 1 * 3 + 2] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[0 * 9 + 2 * 3 + 2] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + 1] + r[1 * 9 + 1 * 3 + 2] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[1 * 9 + 2 * 3 + 2] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + 1];
        ra[1 * 25 + 4 * 5 + 0] = r[0 * 9 + 2 * 3 + 0] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[1 * 9 + 2 * 3 + 0] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + -1];
        ra[1 * 25 + 4 * 5 + 1] = r[0 * 9 + 2 * 3 + 0] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[0 * 9 + 2 * 3 + 1] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[1 * 9 + 2 * 3 + 0] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[1 * 9 + 2 * 3 + 1] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + 0];
        ra[1 * 25 + 4 * 5 + 2] = r[0 * 9 + 2 * 3 + 0] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[0 * 9 + 2 * 3 + 1] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[0 * 9 + 2 * 3 + 2] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + 1] + r[1 * 9 + 2 * 3 + 0] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[1 * 9 + 2 * 3 + 1] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[1 * 9 + 2 * 3 + 2] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + 1];
        ra[1 * 25 + 4 * 5 + 3] = r[0 * 9 + 2 * 3 + 1] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[0 * 9 + 2 * 3 + 2] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + 1] + r[1 * 9 + 2 * 3 + 1] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[1 * 9 + 2 * 3 + 2] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + 1];
        ra[1 * 25 + 4 * 5 + 4] = r[0 * 9 + 2 * 3 + 2] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + 1] + r[1 * 9 + 2 * 3 + 2] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + 1];
        ra[2 * 25 + 0 * 5 + 0] = r[0 * 9 + 0 * 3 + 0] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[1 * 9 + 0 * 3 + 0] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[2 * 9 + 0 * 3 + 0] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + -1];
        ra[2 * 25 + 0 * 5 + 1] = r[0 * 9 + 0 * 3 + 0] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[0 * 9 + 0 * 3 + 1] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[1 * 9 + 0 * 3 + 0] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[1 * 9 + 0 * 3 + 1] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[2 * 9 + 0 * 3 + 0] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[2 * 9 + 0 * 3 + 1] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + 0];
        ra[2 * 25 + 0 * 5 + 2] = r[0 * 9 + 0 * 3 + 0] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[0 * 9 + 0 * 3 + 1] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[0 * 9 + 0 * 3 + 2] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[1 * 9 + 0 * 3 + 0] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[1 * 9 + 0 * 3 + 1] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[1 * 9 + 0 * 3 + 2] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[2 * 9 + 0 * 3 + 0] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[2 * 9 + 0 * 3 + 1] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[2 * 9 + 0 * 3 + 2] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + 1];
        ra[2 * 25 + 0 * 5 + 3] = r[0 * 9 + 0 * 3 + 1] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[0 * 9 + 0 * 3 + 2] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[1 * 9 + 0 * 3 + 1] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[1 * 9 + 0 * 3 + 2] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[2 * 9 + 0 * 3 + 1] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[2 * 9 + 0 * 3 + 2] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + 1];
        ra[2 * 25 + 0 * 5 + 4] = r[0 * 9 + 0 * 3 + 2] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[1 * 9 + 0 * 3 + 2] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[2 * 9 + 0 * 3 + 2] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + 1];
        ra[2 * 25 + 1 * 5 + 0] = r[0 * 9 + 0 * 3 + 0] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[0 * 9 + 1 * 3 + 0] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[1 * 9 + 0 * 3 + 0] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[1 * 9 + 1 * 3 + 0] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[2 * 9 + 0 * 3 + 0] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[2 * 9 + 1 * 3 + 0] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + -1];
        ra[2 * 25 + 1 * 5 + 1] = r[0 * 9 + 0 * 3 + 0] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[0 * 9 + 0 * 3 + 1] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[0 * 9 + 1 * 3 + 0] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[0 * 9 + 1 * 3 + 1] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[1 * 9 + 0 * 3 + 0] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[1 * 9 + 0 * 3 + 1] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[1 * 9 + 1 * 3 + 0] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[1 * 9 + 1 * 3 + 1] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[2 * 9 + 0 * 3 + 0] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[2 * 9 + 0 * 3 + 1] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[2 * 9 + 1 * 3 + 0] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[2 * 9 + 1 * 3 + 1] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + 0];
        ra[2 * 25 + 1 * 5 + 2] = r[0 * 9 + 0 * 3 + 0] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[0 * 9 + 0 * 3 + 1] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[0 * 9 + 0 * 3 + 2] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[0 * 9 + 1 * 3 + 0] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[0 * 9 + 1 * 3 + 1] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[0 * 9 + 1 * 3 + 2] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[1 * 9 + 0 * 3 + 0] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[1 * 9 + 0 * 3 + 1] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[1 * 9 + 0 * 3 + 2] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[1 * 9 + 1 * 3 + 0] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[1 * 9 + 1 * 3 + 1] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[1 * 9 + 1 * 3 + 2] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[2 * 9 + 0 * 3 + 0] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[2 * 9 + 0 * 3 + 1] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[2 * 9 + 0 * 3 + 2] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[2 * 9 + 1 * 3 + 0] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[2 * 9 + 1 * 3 + 1] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[2 * 9 + 1 * 3 + 2] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + 1];
        ra[2 * 25 + 1 * 5 + 3] = r[0 * 9 + 0 * 3 + 1] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[0 * 9 + 0 * 3 + 2] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[0 * 9 + 1 * 3 + 1] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[0 * 9 + 1 * 3 + 2] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[1 * 9 + 0 * 3 + 1] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[1 * 9 + 0 * 3 + 2] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[1 * 9 + 1 * 3 + 1] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[1 * 9 + 1 * 3 + 2] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[2 * 9 + 0 * 3 + 1] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[2 * 9 + 0 * 3 + 2] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[2 * 9 + 1 * 3 + 1] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[2 * 9 + 1 * 3 + 2] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + 1];
        ra[2 * 25 + 1 * 5 + 4] = r[0 * 9 + 0 * 3 + 2] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[0 * 9 + 1 * 3 + 2] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[1 * 9 + 0 * 3 + 2] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[1 * 9 + 1 * 3 + 2] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[2 * 9 + 0 * 3 + 2] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[2 * 9 + 1 * 3 + 2] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + 1];
        ra[2 * 25 + 2 * 5 + 0] = r[0 * 9 + 0 * 3 + 0] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[0 * 9 + 1 * 3 + 0] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[0 * 9 + 2 * 3 + 0] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[1 * 9 + 0 * 3 + 0] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[1 * 9 + 1 * 3 + 0] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[1 * 9 + 2 * 3 + 0] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[2 * 9 + 0 * 3 + 0] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[2 * 9 + 1 * 3 + 0] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[2 * 9 + 2 * 3 + 0] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + -1];
        ra[2 * 25 + 2 * 5 + 1] = r[0 * 9 + 0 * 3 + 0] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[0 * 9 + 0 * 3 + 1] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[0 * 9 + 1 * 3 + 0] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[0 * 9 + 1 * 3 + 1] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[0 * 9 + 2 * 3 + 0] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[0 * 9 + 2 * 3 + 1] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[1 * 9 + 0 * 3 + 0] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[1 * 9 + 0 * 3 + 1] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[1 * 9 + 1 * 3 + 0] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[1 * 9 + 1 * 3 + 1] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[1 * 9 + 2 * 3 + 0] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[1 * 9 + 2 * 3 + 1] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[2 * 9 + 0 * 3 + 0] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[2 * 9 + 0 * 3 + 1] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[2 * 9 + 1 * 3 + 0] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[2 * 9 + 1 * 3 + 1] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[2 * 9 + 2 * 3 + 0] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[2 * 9 + 2 * 3 + 1] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + 0];
        ra[2 * 25 + 2 * 5 + 2] = r[0 * 9 + 0 * 3 + 0] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[0 * 9 + 0 * 3 + 1] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[0 * 9 + 0 * 3 + 2] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[0 * 9 + 1 * 3 + 0] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[0 * 9 + 1 * 3 + 1] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[0 * 9 + 1 * 3 + 2] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[0 * 9 + 2 * 3 + 0] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[0 * 9 + 2 * 3 + 1] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[0 * 9 + 2 * 3 + 2] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + 1] + r[1 * 9 + 0 * 3 + 0] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[1 * 9 + 0 * 3 + 1] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[1 * 9 + 0 * 3 + 2] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[1 * 9 + 1 * 3 + 0] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[1 * 9 + 1 * 3 + 1] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[1 * 9 + 1 * 3 + 2] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[1 * 9 + 2 * 3 + 0] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[1 * 9 + 2 * 3 + 1] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[1 * 9 + 2 * 3 + 2] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + 1] + r[2 * 9 + 0 * 3 + 0] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[2 * 9 + 0 * 3 + 1] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[2 * 9 + 0 * 3 + 2] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[2 * 9 + 1 * 3 + 0] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[2 * 9 + 1 * 3 + 1] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[2 * 9 + 1 * 3 + 2] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[2 * 9 + 2 * 3 + 0] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[2 * 9 + 2 * 3 + 1] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[2 * 9 + 2 * 3 + 2] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + 1];
        ra[2 * 25 + 2 * 5 + 3] = r[0 * 9 + 0 * 3 + 1] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[0 * 9 + 0 * 3 + 2] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[0 * 9 + 1 * 3 + 1] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[0 * 9 + 1 * 3 + 2] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[0 * 9 + 2 * 3 + 1] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[0 * 9 + 2 * 3 + 2] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + 1] + r[1 * 9 + 0 * 3 + 1] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[1 * 9 + 0 * 3 + 2] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[1 * 9 + 1 * 3 + 1] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[1 * 9 + 1 * 3 + 2] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[1 * 9 + 2 * 3 + 1] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[1 * 9 + 2 * 3 + 2] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + 1] + r[2 * 9 + 0 * 3 + 1] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[2 * 9 + 0 * 3 + 2] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[2 * 9 + 1 * 3 + 1] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[2 * 9 + 1 * 3 + 2] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[2 * 9 + 2 * 3 + 1] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[2 * 9 + 2 * 3 + 2] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + 1];
        ra[2 * 25 + 2 * 5 + 4] = r[0 * 9 + 0 * 3 + 2] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[0 * 9 + 1 * 3 + 2] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[0 * 9 + 2 * 3 + 2] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + 1] + r[1 * 9 + 0 * 3 + 2] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[1 * 9 + 1 * 3 + 2] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[1 * 9 + 2 * 3 + 2] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + 1] + r[2 * 9 + 0 * 3 + 2] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[2 * 9 + 1 * 3 + 2] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[2 * 9 + 2 * 3 + 2] * a_h[0 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + 1];
        ra[2 * 25 + 3 * 5 + 0] = r[0 * 9 + 1 * 3 + 0] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[0 * 9 + 2 * 3 + 0] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[1 * 9 + 1 * 3 + 0] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[1 * 9 + 2 * 3 + 0] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[2 * 9 + 1 * 3 + 0] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[2 * 9 + 2 * 3 + 0] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + -1];
        ra[2 * 25 + 3 * 5 + 1] = r[0 * 9 + 1 * 3 + 0] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[0 * 9 + 1 * 3 + 1] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[0 * 9 + 2 * 3 + 0] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[0 * 9 + 2 * 3 + 1] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[1 * 9 + 1 * 3 + 0] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[1 * 9 + 1 * 3 + 1] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[1 * 9 + 2 * 3 + 0] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[1 * 9 + 2 * 3 + 1] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[2 * 9 + 1 * 3 + 0] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[2 * 9 + 1 * 3 + 1] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[2 * 9 + 2 * 3 + 0] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[2 * 9 + 2 * 3 + 1] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + 0];
        ra[2 * 25 + 3 * 5 + 2] = r[0 * 9 + 1 * 3 + 0] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[0 * 9 + 1 * 3 + 1] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[0 * 9 + 1 * 3 + 2] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[0 * 9 + 2 * 3 + 0] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[0 * 9 + 2 * 3 + 1] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[0 * 9 + 2 * 3 + 2] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + 1] + r[1 * 9 + 1 * 3 + 0] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[1 * 9 + 1 * 3 + 1] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[1 * 9 + 1 * 3 + 2] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[1 * 9 + 2 * 3 + 0] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[1 * 9 + 2 * 3 + 1] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[1 * 9 + 2 * 3 + 2] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + 1] + r[2 * 9 + 1 * 3 + 0] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[2 * 9 + 1 * 3 + 1] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[2 * 9 + 1 * 3 + 2] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[2 * 9 + 2 * 3 + 0] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[2 * 9 + 2 * 3 + 1] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[2 * 9 + 2 * 3 + 2] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + 1];
        ra[2 * 25 + 3 * 5 + 3] = r[0 * 9 + 1 * 3 + 1] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[0 * 9 + 1 * 3 + 2] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[0 * 9 + 2 * 3 + 1] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[0 * 9 + 2 * 3 + 2] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + 1] + r[1 * 9 + 1 * 3 + 1] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[1 * 9 + 1 * 3 + 2] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[1 * 9 + 2 * 3 + 1] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[1 * 9 + 2 * 3 + 2] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + 1] + r[2 * 9 + 1 * 3 + 1] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[2 * 9 + 1 * 3 + 2] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[2 * 9 + 2 * 3 + 1] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[2 * 9 + 2 * 3 + 2] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + 1];
        ra[2 * 25 + 3 * 5 + 4] = r[0 * 9 + 1 * 3 + 2] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[0 * 9 + 2 * 3 + 2] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + 1] + r[1 * 9 + 1 * 3 + 2] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[1 * 9 + 2 * 3 + 2] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + 1] + r[2 * 9 + 1 * 3 + 2] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[2 * 9 + 2 * 3 + 2] * a_h[0 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + 1];
        ra[2 * 25 + 4 * 5 + 0] = r[0 * 9 + 2 * 3 + 0] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[1 * 9 + 2 * 3 + 0] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[2 * 9 + 2 * 3 + 0] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + -1];
        ra[2 * 25 + 4 * 5 + 1] = r[0 * 9 + 2 * 3 + 0] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[0 * 9 + 2 * 3 + 1] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[1 * 9 + 2 * 3 + 0] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[1 * 9 + 2 * 3 + 1] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[2 * 9 + 2 * 3 + 0] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[2 * 9 + 2 * 3 + 1] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + 0];
        ra[2 * 25 + 4 * 5 + 2] = r[0 * 9 + 2 * 3 + 0] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[0 * 9 + 2 * 3 + 1] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[0 * 9 + 2 * 3 + 2] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + 1] + r[1 * 9 + 2 * 3 + 0] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[1 * 9 + 2 * 3 + 1] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[1 * 9 + 2 * 3 + 2] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + 1] + r[2 * 9 + 2 * 3 + 0] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[2 * 9 + 2 * 3 + 1] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[2 * 9 + 2 * 3 + 2] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + 1];
        ra[2 * 25 + 4 * 5 + 3] = r[0 * 9 + 2 * 3 + 1] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[0 * 9 + 2 * 3 + 2] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + 1] + r[1 * 9 + 2 * 3 + 1] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[1 * 9 + 2 * 3 + 2] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + 1] + r[2 * 9 + 2 * 3 + 1] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[2 * 9 + 2 * 3 + 2] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + 1];
        ra[2 * 25 + 4 * 5 + 4] = r[0 * 9 + 2 * 3 + 2] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + -1) * nogh_f + (fj + 1) * ogh_f + fk + 1] + r[1 * 9 + 2 * 3 + 2] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + 1] + r[2 * 9 + 2 * 3 + 2] * a_h[0 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + 1];
        ra[3 * 25 + 0 * 5 + 0] = r[1 * 9 + 0 * 3 + 0] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[2 * 9 + 0 * 3 + 0] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + -1];
        ra[3 * 25 + 0 * 5 + 1] = r[1 * 9 + 0 * 3 + 0] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[1 * 9 + 0 * 3 + 1] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[2 * 9 + 0 * 3 + 0] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[2 * 9 + 0 * 3 + 1] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + 0];
        ra[3 * 25 + 0 * 5 + 2] = r[1 * 9 + 0 * 3 + 0] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[1 * 9 + 0 * 3 + 1] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[1 * 9 + 0 * 3 + 2] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[2 * 9 + 0 * 3 + 0] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[2 * 9 + 0 * 3 + 1] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[2 * 9 + 0 * 3 + 2] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + 1];
        ra[3 * 25 + 0 * 5 + 3] = r[1 * 9 + 0 * 3 + 1] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[1 * 9 + 0 * 3 + 2] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[2 * 9 + 0 * 3 + 1] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[2 * 9 + 0 * 3 + 2] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + 1];
        ra[3 * 25 + 0 * 5 + 4] = r[1 * 9 + 0 * 3 + 2] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[2 * 9 + 0 * 3 + 2] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + 1];
        ra[3 * 25 + 1 * 5 + 0] = r[1 * 9 + 0 * 3 + 0] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[1 * 9 + 1 * 3 + 0] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[2 * 9 + 0 * 3 + 0] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[2 * 9 + 1 * 3 + 0] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + -1];
        ra[3 * 25 + 1 * 5 + 1] = r[1 * 9 + 0 * 3 + 0] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[1 * 9 + 0 * 3 + 1] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[1 * 9 + 1 * 3 + 0] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[1 * 9 + 1 * 3 + 1] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[2 * 9 + 0 * 3 + 0] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[2 * 9 + 0 * 3 + 1] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[2 * 9 + 1 * 3 + 0] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[2 * 9 + 1 * 3 + 1] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + 0];
        ra[3 * 25 + 1 * 5 + 2] = r[1 * 9 + 0 * 3 + 0] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[1 * 9 + 0 * 3 + 1] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[1 * 9 + 0 * 3 + 2] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[1 * 9 + 1 * 3 + 0] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[1 * 9 + 1 * 3 + 1] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[1 * 9 + 1 * 3 + 2] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[2 * 9 + 0 * 3 + 0] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[2 * 9 + 0 * 3 + 1] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[2 * 9 + 0 * 3 + 2] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[2 * 9 + 1 * 3 + 0] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[2 * 9 + 1 * 3 + 1] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[2 * 9 + 1 * 3 + 2] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + 1];
        ra[3 * 25 + 1 * 5 + 3] = r[1 * 9 + 0 * 3 + 1] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[1 * 9 + 0 * 3 + 2] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[1 * 9 + 1 * 3 + 1] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[1 * 9 + 1 * 3 + 2] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[2 * 9 + 0 * 3 + 1] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[2 * 9 + 0 * 3 + 2] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[2 * 9 + 1 * 3 + 1] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[2 * 9 + 1 * 3 + 2] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + 1];
        ra[3 * 25 + 1 * 5 + 4] = r[1 * 9 + 0 * 3 + 2] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[1 * 9 + 1 * 3 + 2] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[2 * 9 + 0 * 3 + 2] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[2 * 9 + 1 * 3 + 2] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + 1];
        ra[3 * 25 + 2 * 5 + 0] = r[1 * 9 + 0 * 3 + 0] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[1 * 9 + 1 * 3 + 0] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[1 * 9 + 2 * 3 + 0] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[2 * 9 + 0 * 3 + 0] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[2 * 9 + 1 * 3 + 0] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[2 * 9 + 2 * 3 + 0] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + -1];
        ra[3 * 25 + 2 * 5 + 1] = r[1 * 9 + 0 * 3 + 0] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[1 * 9 + 0 * 3 + 1] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[1 * 9 + 1 * 3 + 0] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[1 * 9 + 1 * 3 + 1] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[1 * 9 + 2 * 3 + 0] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[1 * 9 + 2 * 3 + 1] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[2 * 9 + 0 * 3 + 0] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[2 * 9 + 0 * 3 + 1] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[2 * 9 + 1 * 3 + 0] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[2 * 9 + 1 * 3 + 1] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[2 * 9 + 2 * 3 + 0] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[2 * 9 + 2 * 3 + 1] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + 0];
        ra[3 * 25 + 2 * 5 + 2] = r[1 * 9 + 0 * 3 + 0] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[1 * 9 + 0 * 3 + 1] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[1 * 9 + 0 * 3 + 2] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[1 * 9 + 1 * 3 + 0] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[1 * 9 + 1 * 3 + 1] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[1 * 9 + 1 * 3 + 2] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[1 * 9 + 2 * 3 + 0] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[1 * 9 + 2 * 3 + 1] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[1 * 9 + 2 * 3 + 2] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + 1] + r[2 * 9 + 0 * 3 + 0] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[2 * 9 + 0 * 3 + 1] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[2 * 9 + 0 * 3 + 2] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[2 * 9 + 1 * 3 + 0] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[2 * 9 + 1 * 3 + 1] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[2 * 9 + 1 * 3 + 2] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[2 * 9 + 2 * 3 + 0] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[2 * 9 + 2 * 3 + 1] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[2 * 9 + 2 * 3 + 2] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + 1];
        ra[3 * 25 + 2 * 5 + 3] = r[1 * 9 + 0 * 3 + 1] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[1 * 9 + 0 * 3 + 2] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[1 * 9 + 1 * 3 + 1] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[1 * 9 + 1 * 3 + 2] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[1 * 9 + 2 * 3 + 1] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[1 * 9 + 2 * 3 + 2] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + 1] + r[2 * 9 + 0 * 3 + 1] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[2 * 9 + 0 * 3 + 2] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[2 * 9 + 1 * 3 + 1] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[2 * 9 + 1 * 3 + 2] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[2 * 9 + 2 * 3 + 1] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[2 * 9 + 2 * 3 + 2] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + 1];
        ra[3 * 25 + 2 * 5 + 4] = r[1 * 9 + 0 * 3 + 2] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[1 * 9 + 1 * 3 + 2] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[1 * 9 + 2 * 3 + 2] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + 1] + r[2 * 9 + 0 * 3 + 2] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[2 * 9 + 1 * 3 + 2] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[2 * 9 + 2 * 3 + 2] * a_h[1 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + 1];
        ra[3 * 25 + 3 * 5 + 0] = r[1 * 9 + 1 * 3 + 0] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[1 * 9 + 2 * 3 + 0] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[2 * 9 + 1 * 3 + 0] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[2 * 9 + 2 * 3 + 0] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + -1];
        ra[3 * 25 + 3 * 5 + 1] = r[1 * 9 + 1 * 3 + 0] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[1 * 9 + 1 * 3 + 1] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[1 * 9 + 2 * 3 + 0] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[1 * 9 + 2 * 3 + 1] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[2 * 9 + 1 * 3 + 0] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[2 * 9 + 1 * 3 + 1] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[2 * 9 + 2 * 3 + 0] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[2 * 9 + 2 * 3 + 1] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + 0];
        ra[3 * 25 + 3 * 5 + 2] = r[1 * 9 + 1 * 3 + 0] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[1 * 9 + 1 * 3 + 1] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[1 * 9 + 1 * 3 + 2] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[1 * 9 + 2 * 3 + 0] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[1 * 9 + 2 * 3 + 1] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[1 * 9 + 2 * 3 + 2] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + 1] + r[2 * 9 + 1 * 3 + 0] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[2 * 9 + 1 * 3 + 1] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[2 * 9 + 1 * 3 + 2] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[2 * 9 + 2 * 3 + 0] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[2 * 9 + 2 * 3 + 1] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[2 * 9 + 2 * 3 + 2] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + 1];
        ra[3 * 25 + 3 * 5 + 3] = r[1 * 9 + 1 * 3 + 1] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[1 * 9 + 1 * 3 + 2] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[1 * 9 + 2 * 3 + 1] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[1 * 9 + 2 * 3 + 2] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + 1] + r[2 * 9 + 1 * 3 + 1] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[2 * 9 + 1 * 3 + 2] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[2 * 9 + 2 * 3 + 1] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[2 * 9 + 2 * 3 + 2] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + 1];
        ra[3 * 25 + 3 * 5 + 4] = r[1 * 9 + 1 * 3 + 2] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[1 * 9 + 2 * 3 + 2] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + 1] + r[2 * 9 + 1 * 3 + 2] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[2 * 9 + 2 * 3 + 2] * a_h[1 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + 1];
        ra[3 * 25 + 4 * 5 + 0] = r[1 * 9 + 2 * 3 + 0] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[2 * 9 + 2 * 3 + 0] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + -1];
        ra[3 * 25 + 4 * 5 + 1] = r[1 * 9 + 2 * 3 + 0] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[1 * 9 + 2 * 3 + 1] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[2 * 9 + 2 * 3 + 0] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[2 * 9 + 2 * 3 + 1] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + 0];
        ra[3 * 25 + 4 * 5 + 2] = r[1 * 9 + 2 * 3 + 0] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[1 * 9 + 2 * 3 + 1] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[1 * 9 + 2 * 3 + 2] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + 1] + r[2 * 9 + 2 * 3 + 0] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[2 * 9 + 2 * 3 + 1] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[2 * 9 + 2 * 3 + 2] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + 1];
        ra[3 * 25 + 4 * 5 + 3] = r[1 * 9 + 2 * 3 + 1] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[1 * 9 + 2 * 3 + 2] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + 1] + r[2 * 9 + 2 * 3 + 1] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[2 * 9 + 2 * 3 + 2] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + 1];
        ra[3 * 25 + 4 * 5 + 4] = r[1 * 9 + 2 * 3 + 2] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + 0) * nogh_f + (fj + 1) * ogh_f + fk + 1] + r[2 * 9 + 2 * 3 + 2] * a_h[1 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + 1];
        ra[4 * 25 + 0 * 5 + 0] = r[2 * 9 + 0 * 3 + 0] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + -1];
        ra[4 * 25 + 0 * 5 + 1] = r[2 * 9 + 0 * 3 + 0] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[2 * 9 + 0 * 3 + 1] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + 0];
        ra[4 * 25 + 0 * 5 + 2] = r[2 * 9 + 0 * 3 + 0] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[2 * 9 + 0 * 3 + 1] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[2 * 9 + 0 * 3 + 2] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + 1];
        ra[4 * 25 + 0 * 5 + 3] = r[2 * 9 + 0 * 3 + 1] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[2 * 9 + 0 * 3 + 2] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + 1];
        ra[4 * 25 + 0 * 5 + 4] = r[2 * 9 + 0 * 3 + 2] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + 1];
        ra[4 * 25 + 1 * 5 + 0] = r[2 * 9 + 0 * 3 + 0] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[2 * 9 + 1 * 3 + 0] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + -1];
        ra[4 * 25 + 1 * 5 + 1] = r[2 * 9 + 0 * 3 + 0] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[2 * 9 + 0 * 3 + 1] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[2 * 9 + 1 * 3 + 0] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[2 * 9 + 1 * 3 + 1] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + 0];
        ra[4 * 25 + 1 * 5 + 2] = r[2 * 9 + 0 * 3 + 0] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[2 * 9 + 0 * 3 + 1] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[2 * 9 + 0 * 3 + 2] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[2 * 9 + 1 * 3 + 0] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[2 * 9 + 1 * 3 + 1] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[2 * 9 + 1 * 3 + 2] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + 1];
        ra[4 * 25 + 1 * 5 + 3] = r[2 * 9 + 0 * 3 + 1] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[2 * 9 + 0 * 3 + 2] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[2 * 9 + 1 * 3 + 1] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[2 * 9 + 1 * 3 + 2] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + 1];
        ra[4 * 25 + 1 * 5 + 4] = r[2 * 9 + 0 * 3 + 2] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[2 * 9 + 1 * 3 + 2] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + 1];
        ra[4 * 25 + 2 * 5 + 0] = r[2 * 9 + 0 * 3 + 0] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[2 * 9 + 1 * 3 + 0] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[2 * 9 + 2 * 3 + 0] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + -1];
        ra[4 * 25 + 2 * 5 + 1] = r[2 * 9 + 0 * 3 + 0] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[2 * 9 + 0 * 3 + 1] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[2 * 9 + 1 * 3 + 0] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[2 * 9 + 1 * 3 + 1] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[2 * 9 + 2 * 3 + 0] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[2 * 9 + 2 * 3 + 1] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + 0];
        ra[4 * 25 + 2 * 5 + 2] = r[2 * 9 + 0 * 3 + 0] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + -1] + r[2 * 9 + 0 * 3 + 1] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[2 * 9 + 0 * 3 + 2] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[2 * 9 + 1 * 3 + 0] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[2 * 9 + 1 * 3 + 1] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[2 * 9 + 1 * 3 + 2] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[2 * 9 + 2 * 3 + 0] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[2 * 9 + 2 * 3 + 1] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[2 * 9 + 2 * 3 + 2] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + 1];
        ra[4 * 25 + 2 * 5 + 3] = r[2 * 9 + 0 * 3 + 1] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + 0] + r[2 * 9 + 0 * 3 + 2] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[2 * 9 + 1 * 3 + 1] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[2 * 9 + 1 * 3 + 2] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[2 * 9 + 2 * 3 + 1] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[2 * 9 + 2 * 3 + 2] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + 1];
        ra[4 * 25 + 2 * 5 + 4] = r[2 * 9 + 0 * 3 + 2] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + -1) * ogh_f + fk + 1] + r[2 * 9 + 1 * 3 + 2] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[2 * 9 + 2 * 3 + 2] * a_h[2 * 9 * mnogh_f + 0 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + 1];
        ra[4 * 25 + 3 * 5 + 0] = r[2 * 9 + 1 * 3 + 0] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[2 * 9 + 2 * 3 + 0] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + -1];
        ra[4 * 25 + 3 * 5 + 1] = r[2 * 9 + 1 * 3 + 0] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[2 * 9 + 1 * 3 + 1] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[2 * 9 + 2 * 3 + 0] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[2 * 9 + 2 * 3 + 1] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + 0];
        ra[4 * 25 + 3 * 5 + 2] = r[2 * 9 + 1 * 3 + 0] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + -1] + r[2 * 9 + 1 * 3 + 1] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[2 * 9 + 1 * 3 + 2] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[2 * 9 + 2 * 3 + 0] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[2 * 9 + 2 * 3 + 1] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[2 * 9 + 2 * 3 + 2] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + 1];
        ra[4 * 25 + 3 * 5 + 3] = r[2 * 9 + 1 * 3 + 1] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + 0] + r[2 * 9 + 1 * 3 + 2] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[2 * 9 + 2 * 3 + 1] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[2 * 9 + 2 * 3 + 2] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + 1];
        ra[4 * 25 + 3 * 5 + 4] = r[2 * 9 + 1 * 3 + 2] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + 0) * ogh_f + fk + 1] + r[2 * 9 + 2 * 3 + 2] * a_h[2 * 9 * mnogh_f + 1 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + 1];
        ra[4 * 25 + 4 * 5 + 0] = r[2 * 9 + 2 * 3 + 0] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + -1];
        ra[4 * 25 + 4 * 5 + 1] = r[2 * 9 + 2 * 3 + 0] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[2 * 9 + 2 * 3 + 1] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + 0];
        ra[4 * 25 + 4 * 5 + 2] = r[2 * 9 + 2 * 3 + 0] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + -1] + r[2 * 9 + 2 * 3 + 1] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[2 * 9 + 2 * 3 + 2] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 0 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + 1];
        ra[4 * 25 + 4 * 5 + 3] = r[2 * 9 + 2 * 3 + 1] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + 0] + r[2 * 9 + 2 * 3 + 2] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 1 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + 1];
        ra[4 * 25 + 4 * 5 + 4] = r[2 * 9 + 2 * 3 + 2] * a_h[2 * 9 * mnogh_f + 2 * 3 * mnogh_f + 2 * mnogh_f + (fi + 1) * nogh_f + (fj + 1) * ogh_f + fk + 1];

        a_2h[0 * 9 * mnogh_c + 0 * 3 * mnogh_c + 0 * mnogh_c + ci * nogh_c + cj * (o_c_buf + 2 * gh_c) + ck] = ra[0 * 25 + 0 * 5 + 0] * p[1 * 9 + 1 * 3 + 1] + ra[0 * 25 + 0 * 5 + 1] * p[1 * 9 + 1 * 3 + 0] + ra[0 * 25 + 1 * 5 + 0] * p[1 * 9 + 0 * 3 + 1] + ra[0 * 25 + 1 * 5 + 1] * p[1 * 9 + 0 * 3 + 0] + ra[1 * 25 + 0 * 5 + 0] * p[0 * 9 + 1 * 3 + 1] + ra[1 * 25 + 0 * 5 + 1] * p[0 * 9 + 1 * 3 + 0] + ra[1 * 25 + 1 * 5 + 0] * p[0 * 9 + 0 * 3 + 1] + ra[1 * 25 + 1 * 5 + 1] * p[0 * 9 + 0 * 3 + 0];
        a_2h[0 * 9 * mnogh_c + 0 * 3 * mnogh_c + 1 * mnogh_c + ci * nogh_c + cj * (o_c_buf + 2 * gh_c) + ck] = ra[0 * 25 + 0 * 5 + 1] * p[1 * 9 + 1 * 3 + 2] + ra[0 * 25 + 0 * 5 + 2] * p[1 * 9 + 1 * 3 + 1] + ra[0 * 25 + 0 * 5 + 3] * p[1 * 9 + 1 * 3 + 0] + ra[0 * 25 + 1 * 5 + 1] * p[1 * 9 + 0 * 3 + 2] + ra[0 * 25 + 1 * 5 + 2] * p[1 * 9 + 0 * 3 + 1] + ra[0 * 25 + 1 * 5 + 3] * p[1 * 9 + 0 * 3 + 0] + ra[1 * 25 + 0 * 5 + 1] * p[0 * 9 + 1 * 3 + 2] + ra[1 * 25 + 0 * 5 + 2] * p[0 * 9 + 1 * 3 + 1] + ra[1 * 25 + 0 * 5 + 3] * p[0 * 9 + 1 * 3 + 0] + ra[1 * 25 + 1 * 5 + 1] * p[0 * 9 + 0 * 3 + 2] + ra[1 * 25 + 1 * 5 + 2] * p[0 * 9 + 0 * 3 + 1] + ra[1 * 25 + 1 * 5 + 3] * p[0 * 9 + 0 * 3 + 0];
        a_2h[0 * 9 * mnogh_c + 0 * 3 * mnogh_c + 2 * mnogh_c + ci * nogh_c + cj * (o_c_buf + 2 * gh_c) + ck] = ra[0 * 25 + 0 * 5 + 3] * p[1 * 9 + 1 * 3 + 2] + ra[0 * 25 + 0 * 5 + 4] * p[1 * 9 + 1 * 3 + 1] + ra[0 * 25 + 1 * 5 + 3] * p[1 * 9 + 0 * 3 + 2] + ra[0 * 25 + 1 * 5 + 4] * p[1 * 9 + 0 * 3 + 1] + ra[1 * 25 + 0 * 5 + 3] * p[0 * 9 + 1 * 3 + 2] + ra[1 * 25 + 0 * 5 + 4] * p[0 * 9 + 1 * 3 + 1] + ra[1 * 25 + 1 * 5 + 3] * p[0 * 9 + 0 * 3 + 2] + ra[1 * 25 + 1 * 5 + 4] * p[0 * 9 + 0 * 3 + 1];
        a_2h[0 * 9 * mnogh_c + 1 * 3 * mnogh_c + 0 * mnogh_c + ci * nogh_c + cj * (o_c_buf + 2 * gh_c) + ck] = ra[0 * 25 + 1 * 5 + 0] * p[1 * 9 + 2 * 3 + 1] + ra[0 * 25 + 1 * 5 + 1] * p[1 * 9 + 2 * 3 + 0] + ra[0 * 25 + 2 * 5 + 0] * p[1 * 9 + 1 * 3 + 1] + ra[0 * 25 + 2 * 5 + 1] * p[1 * 9 + 1 * 3 + 0] + ra[0 * 25 + 3 * 5 + 0] * p[1 * 9 + 0 * 3 + 1] + ra[0 * 25 + 3 * 5 + 1] * p[1 * 9 + 0 * 3 + 0] + ra[1 * 25 + 1 * 5 + 0] * p[0 * 9 + 2 * 3 + 1] + ra[1 * 25 + 1 * 5 + 1] * p[0 * 9 + 2 * 3 + 0] + ra[1 * 25 + 2 * 5 + 0] * p[0 * 9 + 1 * 3 + 1] + ra[1 * 25 + 2 * 5 + 1] * p[0 * 9 + 1 * 3 + 0] + ra[1 * 25 + 3 * 5 + 0] * p[0 * 9 + 0 * 3 + 1] + ra[1 * 25 + 3 * 5 + 1] * p[0 * 9 + 0 * 3 + 0];
        a_2h[0 * 9 * mnogh_c + 1 * 3 * mnogh_c + 1 * mnogh_c + ci * nogh_c + cj * (o_c_buf + 2 * gh_c) + ck] = ra[0 * 25 + 1 * 5 + 1] * p[1 * 9 + 2 * 3 + 2] + ra[0 * 25 + 1 * 5 + 2] * p[1 * 9 + 2 * 3 + 1] + ra[0 * 25 + 1 * 5 + 3] * p[1 * 9 + 2 * 3 + 0] + ra[0 * 25 + 2 * 5 + 1] * p[1 * 9 + 1 * 3 + 2] + ra[0 * 25 + 2 * 5 + 2] * p[1 * 9 + 1 * 3 + 1] + ra[0 * 25 + 2 * 5 + 3] * p[1 * 9 + 1 * 3 + 0] + ra[0 * 25 + 3 * 5 + 1] * p[1 * 9 + 0 * 3 + 2] + ra[0 * 25 + 3 * 5 + 2] * p[1 * 9 + 0 * 3 + 1] + ra[0 * 25 + 3 * 5 + 3] * p[1 * 9 + 0 * 3 + 0] + ra[1 * 25 + 1 * 5 + 1] * p[0 * 9 + 2 * 3 + 2] + ra[1 * 25 + 1 * 5 + 2] * p[0 * 9 + 2 * 3 + 1] + ra[1 * 25 + 1 * 5 + 3] * p[0 * 9 + 2 * 3 + 0] + ra[1 * 25 + 2 * 5 + 1] * p[0 * 9 + 1 * 3 + 2] + ra[1 * 25 + 2 * 5 + 2] * p[0 * 9 + 1 * 3 + 1] + ra[1 * 25 + 2 * 5 + 3] * p[0 * 9 + 1 * 3 + 0] + ra[1 * 25 + 3 * 5 + 1] * p[0 * 9 + 0 * 3 + 2] + ra[1 * 25 + 3 * 5 + 2] * p[0 * 9 + 0 * 3 + 1] + ra[1 * 25 + 3 * 5 + 3] * p[0 * 9 + 0 * 3 + 0];
        a_2h[0 * 9 * mnogh_c + 1 * 3 * mnogh_c + 2 * mnogh_c + ci * nogh_c + cj * (o_c_buf + 2 * gh_c) + ck] = ra[0 * 25 + 1 * 5 + 3] * p[1 * 9 + 2 * 3 + 2] + ra[0 * 25 + 1 * 5 + 4] * p[1 * 9 + 2 * 3 + 1] + ra[0 * 25 + 2 * 5 + 3] * p[1 * 9 + 1 * 3 + 2] + ra[0 * 25 + 2 * 5 + 4] * p[1 * 9 + 1 * 3 + 1] + ra[0 * 25 + 3 * 5 + 3] * p[1 * 9 + 0 * 3 + 2] + ra[0 * 25 + 3 * 5 + 4] * p[1 * 9 + 0 * 3 + 1] + ra[1 * 25 + 1 * 5 + 3] * p[0 * 9 + 2 * 3 + 2] + ra[1 * 25 + 1 * 5 + 4] * p[0 * 9 + 2 * 3 + 1] + ra[1 * 25 + 2 * 5 + 3] * p[0 * 9 + 1 * 3 + 2] + ra[1 * 25 + 2 * 5 + 4] * p[0 * 9 + 1 * 3 + 1] + ra[1 * 25 + 3 * 5 + 3] * p[0 * 9 + 0 * 3 + 2] + ra[1 * 25 + 3 * 5 + 4] * p[0 * 9 + 0 * 3 + 1];
        a_2h[0 * 9 * mnogh_c + 2 * 3 * mnogh_c + 0 * mnogh_c + ci * nogh_c + cj * (o_c_buf + 2 * gh_c) + ck] = ra[0 * 25 + 3 * 5 + 0] * p[1 * 9 + 2 * 3 + 1] + ra[0 * 25 + 3 * 5 + 1] * p[1 * 9 + 2 * 3 + 0] + ra[0 * 25 + 4 * 5 + 0] * p[1 * 9 + 1 * 3 + 1] + ra[0 * 25 + 4 * 5 + 1] * p[1 * 9 + 1 * 3 + 0] + ra[1 * 25 + 3 * 5 + 0] * p[0 * 9 + 2 * 3 + 1] + ra[1 * 25 + 3 * 5 + 1] * p[0 * 9 + 2 * 3 + 0] + ra[1 * 25 + 4 * 5 + 0] * p[0 * 9 + 1 * 3 + 1] + ra[1 * 25 + 4 * 5 + 1] * p[0 * 9 + 1 * 3 + 0];
        a_2h[0 * 9 * mnogh_c + 2 * 3 * mnogh_c + 1 * mnogh_c + ci * nogh_c + cj * (o_c_buf + 2 * gh_c) + ck] = ra[0 * 25 + 3 * 5 + 1] * p[1 * 9 + 2 * 3 + 2] + ra[0 * 25 + 3 * 5 + 2] * p[1 * 9 + 2 * 3 + 1] + ra[0 * 25 + 3 * 5 + 3] * p[1 * 9 + 2 * 3 + 0] + ra[0 * 25 + 4 * 5 + 1] * p[1 * 9 + 1 * 3 + 2] + ra[0 * 25 + 4 * 5 + 2] * p[1 * 9 + 1 * 3 + 1] + ra[0 * 25 + 4 * 5 + 3] * p[1 * 9 + 1 * 3 + 0] + ra[1 * 25 + 3 * 5 + 1] * p[0 * 9 + 2 * 3 + 2] + ra[1 * 25 + 3 * 5 + 2] * p[0 * 9 + 2 * 3 + 1] + ra[1 * 25 + 3 * 5 + 3] * p[0 * 9 + 2 * 3 + 0] + ra[1 * 25 + 4 * 5 + 1] * p[0 * 9 + 1 * 3 + 2] + ra[1 * 25 + 4 * 5 + 2] * p[0 * 9 + 1 * 3 + 1] + ra[1 * 25 + 4 * 5 + 3] * p[0 * 9 + 1 * 3 + 0];
        a_2h[0 * 9 * mnogh_c + 2 * 3 * mnogh_c + 2 * mnogh_c + ci * nogh_c + cj * (o_c_buf + 2 * gh_c) + ck] = ra[0 * 25 + 3 * 5 + 3] * p[1 * 9 + 2 * 3 + 2] + ra[0 * 25 + 3 * 5 + 4] * p[1 * 9 + 2 * 3 + 1] + ra[0 * 25 + 4 * 5 + 3] * p[1 * 9 + 1 * 3 + 2] + ra[0 * 25 + 4 * 5 + 4] * p[1 * 9 + 1 * 3 + 1] + ra[1 * 25 + 3 * 5 + 3] * p[0 * 9 + 2 * 3 + 2] + ra[1 * 25 + 3 * 5 + 4] * p[0 * 9 + 2 * 3 + 1] + ra[1 * 25 + 4 * 5 + 3] * p[0 * 9 + 1 * 3 + 2] + ra[1 * 25 + 4 * 5 + 4] * p[0 * 9 + 1 * 3 + 1];
        a_2h[1 * 9 * mnogh_c + 0 * 3 * mnogh_c + 0 * mnogh_c + ci * nogh_c + cj * (o_c_buf + 2 * gh_c) + ck] = ra[1 * 25 + 0 * 5 + 0] * p[2 * 9 + 1 * 3 + 1] + ra[1 * 25 + 0 * 5 + 1] * p[2 * 9 + 1 * 3 + 0] + ra[1 * 25 + 1 * 5 + 0] * p[2 * 9 + 0 * 3 + 1] + ra[1 * 25 + 1 * 5 + 1] * p[2 * 9 + 0 * 3 + 0] + ra[2 * 25 + 0 * 5 + 0] * p[1 * 9 + 1 * 3 + 1] + ra[2 * 25 + 0 * 5 + 1] * p[1 * 9 + 1 * 3 + 0] + ra[2 * 25 + 1 * 5 + 0] * p[1 * 9 + 0 * 3 + 1] + ra[2 * 25 + 1 * 5 + 1] * p[1 * 9 + 0 * 3 + 0] + ra[3 * 25 + 0 * 5 + 0] * p[0 * 9 + 1 * 3 + 1] + ra[3 * 25 + 0 * 5 + 1] * p[0 * 9 + 1 * 3 + 0] + ra[3 * 25 + 1 * 5 + 0] * p[0 * 9 + 0 * 3 + 1] + ra[3 * 25 + 1 * 5 + 1] * p[0 * 9 + 0 * 3 + 0];
        a_2h[1 * 9 * mnogh_c + 0 * 3 * mnogh_c + 1 * mnogh_c + ci * nogh_c + cj * (o_c_buf + 2 * gh_c) + ck] = ra[1 * 25 + 0 * 5 + 1] * p[2 * 9 + 1 * 3 + 2] + ra[1 * 25 + 0 * 5 + 2] * p[2 * 9 + 1 * 3 + 1] + ra[1 * 25 + 0 * 5 + 3] * p[2 * 9 + 1 * 3 + 0] + ra[1 * 25 + 1 * 5 + 1] * p[2 * 9 + 0 * 3 + 2] + ra[1 * 25 + 1 * 5 + 2] * p[2 * 9 + 0 * 3 + 1] + ra[1 * 25 + 1 * 5 + 3] * p[2 * 9 + 0 * 3 + 0] + ra[2 * 25 + 0 * 5 + 1] * p[1 * 9 + 1 * 3 + 2] + ra[2 * 25 + 0 * 5 + 2] * p[1 * 9 + 1 * 3 + 1] + ra[2 * 25 + 0 * 5 + 3] * p[1 * 9 + 1 * 3 + 0] + ra[2 * 25 + 1 * 5 + 1] * p[1 * 9 + 0 * 3 + 2] + ra[2 * 25 + 1 * 5 + 2] * p[1 * 9 + 0 * 3 + 1] + ra[2 * 25 + 1 * 5 + 3] * p[1 * 9 + 0 * 3 + 0] + ra[3 * 25 + 0 * 5 + 1] * p[0 * 9 + 1 * 3 + 2] + ra[3 * 25 + 0 * 5 + 2] * p[0 * 9 + 1 * 3 + 1] + ra[3 * 25 + 0 * 5 + 3] * p[0 * 9 + 1 * 3 + 0] + ra[3 * 25 + 1 * 5 + 1] * p[0 * 9 + 0 * 3 + 2] + ra[3 * 25 + 1 * 5 + 2] * p[0 * 9 + 0 * 3 + 1] + ra[3 * 25 + 1 * 5 + 3] * p[0 * 9 + 0 * 3 + 0];
        a_2h[1 * 9 * mnogh_c + 0 * 3 * mnogh_c + 2 * mnogh_c + ci * nogh_c + cj * (o_c_buf + 2 * gh_c) + ck] = ra[1 * 25 + 0 * 5 + 3] * p[2 * 9 + 1 * 3 + 2] + ra[1 * 25 + 0 * 5 + 4] * p[2 * 9 + 1 * 3 + 1] + ra[1 * 25 + 1 * 5 + 3] * p[2 * 9 + 0 * 3 + 2] + ra[1 * 25 + 1 * 5 + 4] * p[2 * 9 + 0 * 3 + 1] + ra[2 * 25 + 0 * 5 + 3] * p[1 * 9 + 1 * 3 + 2] + ra[2 * 25 + 0 * 5 + 4] * p[1 * 9 + 1 * 3 + 1] + ra[2 * 25 + 1 * 5 + 3] * p[1 * 9 + 0 * 3 + 2] + ra[2 * 25 + 1 * 5 + 4] * p[1 * 9 + 0 * 3 + 1] + ra[3 * 25 + 0 * 5 + 3] * p[0 * 9 + 1 * 3 + 2] + ra[3 * 25 + 0 * 5 + 4] * p[0 * 9 + 1 * 3 + 1] + ra[3 * 25 + 1 * 5 + 3] * p[0 * 9 + 0 * 3 + 2] + ra[3 * 25 + 1 * 5 + 4] * p[0 * 9 + 0 * 3 + 1];
        a_2h[1 * 9 * mnogh_c + 1 * 3 * mnogh_c + 0 * mnogh_c + ci * nogh_c + cj * (o_c_buf + 2 * gh_c) + ck] = ra[1 * 25 + 1 * 5 + 0] * p[2 * 9 + 2 * 3 + 1] + ra[1 * 25 + 1 * 5 + 1] * p[2 * 9 + 2 * 3 + 0] + ra[1 * 25 + 2 * 5 + 0] * p[2 * 9 + 1 * 3 + 1] + ra[1 * 25 + 2 * 5 + 1] * p[2 * 9 + 1 * 3 + 0] + ra[1 * 25 + 3 * 5 + 0] * p[2 * 9 + 0 * 3 + 1] + ra[1 * 25 + 3 * 5 + 1] * p[2 * 9 + 0 * 3 + 0] + ra[2 * 25 + 1 * 5 + 0] * p[1 * 9 + 2 * 3 + 1] + ra[2 * 25 + 1 * 5 + 1] * p[1 * 9 + 2 * 3 + 0] + ra[2 * 25 + 2 * 5 + 0] * p[1 * 9 + 1 * 3 + 1] + ra[2 * 25 + 2 * 5 + 1] * p[1 * 9 + 1 * 3 + 0] + ra[2 * 25 + 3 * 5 + 0] * p[1 * 9 + 0 * 3 + 1] + ra[2 * 25 + 3 * 5 + 1] * p[1 * 9 + 0 * 3 + 0] + ra[3 * 25 + 1 * 5 + 0] * p[0 * 9 + 2 * 3 + 1] + ra[3 * 25 + 1 * 5 + 1] * p[0 * 9 + 2 * 3 + 0] + ra[3 * 25 + 2 * 5 + 0] * p[0 * 9 + 1 * 3 + 1] + ra[3 * 25 + 2 * 5 + 1] * p[0 * 9 + 1 * 3 + 0] + ra[3 * 25 + 3 * 5 + 0] * p[0 * 9 + 0 * 3 + 1] + ra[3 * 25 + 3 * 5 + 1] * p[0 * 9 + 0 * 3 + 0];
        a_2h[1 * 9 * mnogh_c + 1 * 3 * mnogh_c + 1 * mnogh_c + ci * nogh_c + cj * (o_c_buf + 2 * gh_c) + ck] = ra[1 * 25 + 1 * 5 + 1] * p[2 * 9 + 2 * 3 + 2] + ra[1 * 25 + 1 * 5 + 2] * p[2 * 9 + 2 * 3 + 1] + ra[1 * 25 + 1 * 5 + 3] * p[2 * 9 + 2 * 3 + 0] + ra[1 * 25 + 2 * 5 + 1] * p[2 * 9 + 1 * 3 + 2] + ra[1 * 25 + 2 * 5 + 2] * p[2 * 9 + 1 * 3 + 1] + ra[1 * 25 + 2 * 5 + 3] * p[2 * 9 + 1 * 3 + 0] + ra[1 * 25 + 3 * 5 + 1] * p[2 * 9 + 0 * 3 + 2] + ra[1 * 25 + 3 * 5 + 2] * p[2 * 9 + 0 * 3 + 1] + ra[1 * 25 + 3 * 5 + 3] * p[2 * 9 + 0 * 3 + 0] + ra[2 * 25 + 1 * 5 + 1] * p[1 * 9 + 2 * 3 + 2] + ra[2 * 25 + 1 * 5 + 2] * p[1 * 9 + 2 * 3 + 1] + ra[2 * 25 + 1 * 5 + 3] * p[1 * 9 + 2 * 3 + 0] + ra[2 * 25 + 2 * 5 + 1] * p[1 * 9 + 1 * 3 + 2] + ra[2 * 25 + 2 * 5 + 2] * p[1 * 9 + 1 * 3 + 1] + ra[2 * 25 + 2 * 5 + 3] * p[1 * 9 + 1 * 3 + 0] + ra[2 * 25 + 3 * 5 + 1] * p[1 * 9 + 0 * 3 + 2] + ra[2 * 25 + 3 * 5 + 2] * p[1 * 9 + 0 * 3 + 1] + ra[2 * 25 + 3 * 5 + 3] * p[1 * 9 + 0 * 3 + 0] + ra[3 * 25 + 1 * 5 + 1] * p[0 * 9 + 2 * 3 + 2] + ra[3 * 25 + 1 * 5 + 2] * p[0 * 9 + 2 * 3 + 1] + ra[3 * 25 + 1 * 5 + 3] * p[0 * 9 + 2 * 3 + 0] + ra[3 * 25 + 2 * 5 + 1] * p[0 * 9 + 1 * 3 + 2] + ra[3 * 25 + 2 * 5 + 2] * p[0 * 9 + 1 * 3 + 1] + ra[3 * 25 + 2 * 5 + 3] * p[0 * 9 + 1 * 3 + 0] + ra[3 * 25 + 3 * 5 + 1] * p[0 * 9 + 0 * 3 + 2] + ra[3 * 25 + 3 * 5 + 2] * p[0 * 9 + 0 * 3 + 1] + ra[3 * 25 + 3 * 5 + 3] * p[0 * 9 + 0 * 3 + 0];
        a_2h[1 * 9 * mnogh_c + 1 * 3 * mnogh_c + 2 * mnogh_c + ci * nogh_c + cj * (o_c_buf + 2 * gh_c) + ck] = ra[1 * 25 + 1 * 5 + 3] * p[2 * 9 + 2 * 3 + 2] + ra[1 * 25 + 1 * 5 + 4] * p[2 * 9 + 2 * 3 + 1] + ra[1 * 25 + 2 * 5 + 3] * p[2 * 9 + 1 * 3 + 2] + ra[1 * 25 + 2 * 5 + 4] * p[2 * 9 + 1 * 3 + 1] + ra[1 * 25 + 3 * 5 + 3] * p[2 * 9 + 0 * 3 + 2] + ra[1 * 25 + 3 * 5 + 4] * p[2 * 9 + 0 * 3 + 1] + ra[2 * 25 + 1 * 5 + 3] * p[1 * 9 + 2 * 3 + 2] + ra[2 * 25 + 1 * 5 + 4] * p[1 * 9 + 2 * 3 + 1] + ra[2 * 25 + 2 * 5 + 3] * p[1 * 9 + 1 * 3 + 2] + ra[2 * 25 + 2 * 5 + 4] * p[1 * 9 + 1 * 3 + 1] + ra[2 * 25 + 3 * 5 + 3] * p[1 * 9 + 0 * 3 + 2] + ra[2 * 25 + 3 * 5 + 4] * p[1 * 9 + 0 * 3 + 1] + ra[3 * 25 + 1 * 5 + 3] * p[0 * 9 + 2 * 3 + 2] + ra[3 * 25 + 1 * 5 + 4] * p[0 * 9 + 2 * 3 + 1] + ra[3 * 25 + 2 * 5 + 3] * p[0 * 9 + 1 * 3 + 2] + ra[3 * 25 + 2 * 5 + 4] * p[0 * 9 + 1 * 3 + 1] + ra[3 * 25 + 3 * 5 + 3] * p[0 * 9 + 0 * 3 + 2] + ra[3 * 25 + 3 * 5 + 4] * p[0 * 9 + 0 * 3 + 1];
        a_2h[1 * 9 * mnogh_c + 2 * 3 * mnogh_c + 0 * mnogh_c + ci * nogh_c + cj * (o_c_buf + 2 * gh_c) + ck] = ra[1 * 25 + 3 * 5 + 0] * p[2 * 9 + 2 * 3 + 1] + ra[1 * 25 + 3 * 5 + 1] * p[2 * 9 + 2 * 3 + 0] + ra[1 * 25 + 4 * 5 + 0] * p[2 * 9 + 1 * 3 + 1] + ra[1 * 25 + 4 * 5 + 1] * p[2 * 9 + 1 * 3 + 0] + ra[2 * 25 + 3 * 5 + 0] * p[1 * 9 + 2 * 3 + 1] + ra[2 * 25 + 3 * 5 + 1] * p[1 * 9 + 2 * 3 + 0] + ra[2 * 25 + 4 * 5 + 0] * p[1 * 9 + 1 * 3 + 1] + ra[2 * 25 + 4 * 5 + 1] * p[1 * 9 + 1 * 3 + 0] + ra[3 * 25 + 3 * 5 + 0] * p[0 * 9 + 2 * 3 + 1] + ra[3 * 25 + 3 * 5 + 1] * p[0 * 9 + 2 * 3 + 0] + ra[3 * 25 + 4 * 5 + 0] * p[0 * 9 + 1 * 3 + 1] + ra[3 * 25 + 4 * 5 + 1] * p[0 * 9 + 1 * 3 + 0];
        a_2h[1 * 9 * mnogh_c + 2 * 3 * mnogh_c + 1 * mnogh_c + ci * nogh_c + cj * (o_c_buf + 2 * gh_c) + ck] = ra[1 * 25 + 3 * 5 + 1] * p[2 * 9 + 2 * 3 + 2] + ra[1 * 25 + 3 * 5 + 2] * p[2 * 9 + 2 * 3 + 1] + ra[1 * 25 + 3 * 5 + 3] * p[2 * 9 + 2 * 3 + 0] + ra[1 * 25 + 4 * 5 + 1] * p[2 * 9 + 1 * 3 + 2] + ra[1 * 25 + 4 * 5 + 2] * p[2 * 9 + 1 * 3 + 1] + ra[1 * 25 + 4 * 5 + 3] * p[2 * 9 + 1 * 3 + 0] + ra[2 * 25 + 3 * 5 + 1] * p[1 * 9 + 2 * 3 + 2] + ra[2 * 25 + 3 * 5 + 2] * p[1 * 9 + 2 * 3 + 1] + ra[2 * 25 + 3 * 5 + 3] * p[1 * 9 + 2 * 3 + 0] + ra[2 * 25 + 4 * 5 + 1] * p[1 * 9 + 1 * 3 + 2] + ra[2 * 25 + 4 * 5 + 2] * p[1 * 9 + 1 * 3 + 1] + ra[2 * 25 + 4 * 5 + 3] * p[1 * 9 + 1 * 3 + 0] + ra[3 * 25 + 3 * 5 + 1] * p[0 * 9 + 2 * 3 + 2] + ra[3 * 25 + 3 * 5 + 2] * p[0 * 9 + 2 * 3 + 1] + ra[3 * 25 + 3 * 5 + 3] * p[0 * 9 + 2 * 3 + 0] + ra[3 * 25 + 4 * 5 + 1] * p[0 * 9 + 1 * 3 + 2] + ra[3 * 25 + 4 * 5 + 2] * p[0 * 9 + 1 * 3 + 1] + ra[3 * 25 + 4 * 5 + 3] * p[0 * 9 + 1 * 3 + 0];
        a_2h[1 * 9 * mnogh_c + 2 * 3 * mnogh_c + 2 * mnogh_c + ci * nogh_c + cj * (o_c_buf + 2 * gh_c) + ck] = ra[1 * 25 + 3 * 5 + 3] * p[2 * 9 + 2 * 3 + 2] + ra[1 * 25 + 3 * 5 + 4] * p[2 * 9 + 2 * 3 + 1] + ra[1 * 25 + 4 * 5 + 3] * p[2 * 9 + 1 * 3 + 2] + ra[1 * 25 + 4 * 5 + 4] * p[2 * 9 + 1 * 3 + 1] + ra[2 * 25 + 3 * 5 + 3] * p[1 * 9 + 2 * 3 + 2] + ra[2 * 25 + 3 * 5 + 4] * p[1 * 9 + 2 * 3 + 1] + ra[2 * 25 + 4 * 5 + 3] * p[1 * 9 + 1 * 3 + 2] + ra[2 * 25 + 4 * 5 + 4] * p[1 * 9 + 1 * 3 + 1] + ra[3 * 25 + 3 * 5 + 3] * p[0 * 9 + 2 * 3 + 2] + ra[3 * 25 + 3 * 5 + 4] * p[0 * 9 + 2 * 3 + 1] + ra[3 * 25 + 4 * 5 + 3] * p[0 * 9 + 1 * 3 + 2] + ra[3 * 25 + 4 * 5 + 4] * p[0 * 9 + 1 * 3 + 1];
        a_2h[2 * 9 * mnogh_c + 0 * 3 * mnogh_c + 0 * mnogh_c + ci * nogh_c + cj * (o_c_buf + 2 * gh_c) + ck] = ra[3 * 25 + 0 * 5 + 0] * p[2 * 9 + 1 * 3 + 1] + ra[3 * 25 + 0 * 5 + 1] * p[2 * 9 + 1 * 3 + 0] + ra[3 * 25 + 1 * 5 + 0] * p[2 * 9 + 0 * 3 + 1] + ra[3 * 25 + 1 * 5 + 1] * p[2 * 9 + 0 * 3 + 0] + ra[4 * 25 + 0 * 5 + 0] * p[1 * 9 + 1 * 3 + 1] + ra[4 * 25 + 0 * 5 + 1] * p[1 * 9 + 1 * 3 + 0] + ra[4 * 25 + 1 * 5 + 0] * p[1 * 9 + 0 * 3 + 1] + ra[4 * 25 + 1 * 5 + 1] * p[1 * 9 + 0 * 3 + 0];
        a_2h[2 * 9 * mnogh_c + 0 * 3 * mnogh_c + 1 * mnogh_c + ci * nogh_c + cj * (o_c_buf + 2 * gh_c) + ck] = ra[3 * 25 + 0 * 5 + 1] * p[2 * 9 + 1 * 3 + 2] + ra[3 * 25 + 0 * 5 + 2] * p[2 * 9 + 1 * 3 + 1] + ra[3 * 25 + 0 * 5 + 3] * p[2 * 9 + 1 * 3 + 0] + ra[3 * 25 + 1 * 5 + 1] * p[2 * 9 + 0 * 3 + 2] + ra[3 * 25 + 1 * 5 + 2] * p[2 * 9 + 0 * 3 + 1] + ra[3 * 25 + 1 * 5 + 3] * p[2 * 9 + 0 * 3 + 0] + ra[4 * 25 + 0 * 5 + 1] * p[1 * 9 + 1 * 3 + 2] + ra[4 * 25 + 0 * 5 + 2] * p[1 * 9 + 1 * 3 + 1] + ra[4 * 25 + 0 * 5 + 3] * p[1 * 9 + 1 * 3 + 0] + ra[4 * 25 + 1 * 5 + 1] * p[1 * 9 + 0 * 3 + 2] + ra[4 * 25 + 1 * 5 + 2] * p[1 * 9 + 0 * 3 + 1] + ra[4 * 25 + 1 * 5 + 3] * p[1 * 9 + 0 * 3 + 0];
        a_2h[2 * 9 * mnogh_c + 0 * 3 * mnogh_c + 2 * mnogh_c + ci * nogh_c + cj * (o_c_buf + 2 * gh_c) + ck] = ra[3 * 25 + 0 * 5 + 3] * p[2 * 9 + 1 * 3 + 2] + ra[3 * 25 + 0 * 5 + 4] * p[2 * 9 + 1 * 3 + 1] + ra[3 * 25 + 1 * 5 + 3] * p[2 * 9 + 0 * 3 + 2] + ra[3 * 25 + 1 * 5 + 4] * p[2 * 9 + 0 * 3 + 1] + ra[4 * 25 + 0 * 5 + 3] * p[1 * 9 + 1 * 3 + 2] + ra[4 * 25 + 0 * 5 + 4] * p[1 * 9 + 1 * 3 + 1] + ra[4 * 25 + 1 * 5 + 3] * p[1 * 9 + 0 * 3 + 2] + ra[4 * 25 + 1 * 5 + 4] * p[1 * 9 + 0 * 3 + 1];
        a_2h[2 * 9 * mnogh_c + 1 * 3 * mnogh_c + 0 * mnogh_c + ci * nogh_c + cj * (o_c_buf + 2 * gh_c) + ck] = ra[3 * 25 + 1 * 5 + 0] * p[2 * 9 + 2 * 3 + 1] + ra[3 * 25 + 1 * 5 + 1] * p[2 * 9 + 2 * 3 + 0] + ra[3 * 25 + 2 * 5 + 0] * p[2 * 9 + 1 * 3 + 1] + ra[3 * 25 + 2 * 5 + 1] * p[2 * 9 + 1 * 3 + 0] + ra[3 * 25 + 3 * 5 + 0] * p[2 * 9 + 0 * 3 + 1] + ra[3 * 25 + 3 * 5 + 1] * p[2 * 9 + 0 * 3 + 0] + ra[4 * 25 + 1 * 5 + 0] * p[1 * 9 + 2 * 3 + 1] + ra[4 * 25 + 1 * 5 + 1] * p[1 * 9 + 2 * 3 + 0] + ra[4 * 25 + 2 * 5 + 0] * p[1 * 9 + 1 * 3 + 1] + ra[4 * 25 + 2 * 5 + 1] * p[1 * 9 + 1 * 3 + 0] + ra[4 * 25 + 3 * 5 + 0] * p[1 * 9 + 0 * 3 + 1] + ra[4 * 25 + 3 * 5 + 1] * p[1 * 9 + 0 * 3 + 0];
        a_2h[2 * 9 * mnogh_c + 1 * 3 * mnogh_c + 1 * mnogh_c + ci * nogh_c + cj * (o_c_buf + 2 * gh_c) + ck] = ra[3 * 25 + 1 * 5 + 1] * p[2 * 9 + 2 * 3 + 2] + ra[3 * 25 + 1 * 5 + 2] * p[2 * 9 + 2 * 3 + 1] + ra[3 * 25 + 1 * 5 + 3] * p[2 * 9 + 2 * 3 + 0] + ra[3 * 25 + 2 * 5 + 1] * p[2 * 9 + 1 * 3 + 2] + ra[3 * 25 + 2 * 5 + 2] * p[2 * 9 + 1 * 3 + 1] + ra[3 * 25 + 2 * 5 + 3] * p[2 * 9 + 1 * 3 + 0] + ra[3 * 25 + 3 * 5 + 1] * p[2 * 9 + 0 * 3 + 2] + ra[3 * 25 + 3 * 5 + 2] * p[2 * 9 + 0 * 3 + 1] + ra[3 * 25 + 3 * 5 + 3] * p[2 * 9 + 0 * 3 + 0] + ra[4 * 25 + 1 * 5 + 1] * p[1 * 9 + 2 * 3 + 2] + ra[4 * 25 + 1 * 5 + 2] * p[1 * 9 + 2 * 3 + 1] + ra[4 * 25 + 1 * 5 + 3] * p[1 * 9 + 2 * 3 + 0] + ra[4 * 25 + 2 * 5 + 1] * p[1 * 9 + 1 * 3 + 2] + ra[4 * 25 + 2 * 5 + 2] * p[1 * 9 + 1 * 3 + 1] + ra[4 * 25 + 2 * 5 + 3] * p[1 * 9 + 1 * 3 + 0] + ra[4 * 25 + 3 * 5 + 1] * p[1 * 9 + 0 * 3 + 2] + ra[4 * 25 + 3 * 5 + 2] * p[1 * 9 + 0 * 3 + 1] + ra[4 * 25 + 3 * 5 + 3] * p[1 * 9 + 0 * 3 + 0];
        a_2h[2 * 9 * mnogh_c + 1 * 3 * mnogh_c + 2 * mnogh_c + ci * nogh_c + cj * (o_c_buf + 2 * gh_c) + ck] = ra[3 * 25 + 1 * 5 + 3] * p[2 * 9 + 2 * 3 + 2] + ra[3 * 25 + 1 * 5 + 4] * p[2 * 9 + 2 * 3 + 1] + ra[3 * 25 + 2 * 5 + 3] * p[2 * 9 + 1 * 3 + 2] + ra[3 * 25 + 2 * 5 + 4] * p[2 * 9 + 1 * 3 + 1] + ra[3 * 25 + 3 * 5 + 3] * p[2 * 9 + 0 * 3 + 2] + ra[3 * 25 + 3 * 5 + 4] * p[2 * 9 + 0 * 3 + 1] + ra[4 * 25 + 1 * 5 + 3] * p[1 * 9 + 2 * 3 + 2] + ra[4 * 25 + 1 * 5 + 4] * p[1 * 9 + 2 * 3 + 1] + ra[4 * 25 + 2 * 5 + 3] * p[1 * 9 + 1 * 3 + 2] + ra[4 * 25 + 2 * 5 + 4] * p[1 * 9 + 1 * 3 + 1] + ra[4 * 25 + 3 * 5 + 3] * p[1 * 9 + 0 * 3 + 2] + ra[4 * 25 + 3 * 5 + 4] * p[1 * 9 + 0 * 3 + 1];
        a_2h[2 * 9 * mnogh_c + 2 * 3 * mnogh_c + 0 * mnogh_c + ci * nogh_c + cj * (o_c_buf + 2 * gh_c) + ck] = ra[3 * 25 + 3 * 5 + 0] * p[2 * 9 + 2 * 3 + 1] + ra[3 * 25 + 3 * 5 + 1] * p[2 * 9 + 2 * 3 + 0] + ra[3 * 25 + 4 * 5 + 0] * p[2 * 9 + 1 * 3 + 1] + ra[3 * 25 + 4 * 5 + 1] * p[2 * 9 + 1 * 3 + 0] + ra[4 * 25 + 3 * 5 + 0] * p[1 * 9 + 2 * 3 + 1] + ra[4 * 25 + 3 * 5 + 1] * p[1 * 9 + 2 * 3 + 0] + ra[4 * 25 + 4 * 5 + 0] * p[1 * 9 + 1 * 3 + 1] + ra[4 * 25 + 4 * 5 + 1] * p[1 * 9 + 1 * 3 + 0];
        a_2h[2 * 9 * mnogh_c + 2 * 3 * mnogh_c + 1 * mnogh_c + ci * nogh_c + cj * (o_c_buf + 2 * gh_c) + ck] = ra[3 * 25 + 3 * 5 + 1] * p[2 * 9 + 2 * 3 + 2] + ra[3 * 25 + 3 * 5 + 2] * p[2 * 9 + 2 * 3 + 1] + ra[3 * 25 + 3 * 5 + 3] * p[2 * 9 + 2 * 3 + 0] + ra[3 * 25 + 4 * 5 + 1] * p[2 * 9 + 1 * 3 + 2] + ra[3 * 25 + 4 * 5 + 2] * p[2 * 9 + 1 * 3 + 1] + ra[3 * 25 + 4 * 5 + 3] * p[2 * 9 + 1 * 3 + 0] + ra[4 * 25 + 3 * 5 + 1] * p[1 * 9 + 2 * 3 + 2] + ra[4 * 25 + 3 * 5 + 2] * p[1 * 9 + 2 * 3 + 1] + ra[4 * 25 + 3 * 5 + 3] * p[1 * 9 + 2 * 3 + 0] + ra[4 * 25 + 4 * 5 + 1] * p[1 * 9 + 1 * 3 + 2] + ra[4 * 25 + 4 * 5 + 2] * p[1 * 9 + 1 * 3 + 1] + ra[4 * 25 + 4 * 5 + 3] * p[1 * 9 + 1 * 3 + 0];
        a_2h[2 * 9 * mnogh_c + 2 * 3 * mnogh_c + 2 * mnogh_c + ci * nogh_c + cj * (o_c_buf + 2 * gh_c) + ck] = ra[3 * 25 + 3 * 5 + 3] * p[2 * 9 + 2 * 3 + 2] + ra[3 * 25 + 3 * 5 + 4] * p[2 * 9 + 2 * 3 + 1] + ra[3 * 25 + 4 * 5 + 3] * p[2 * 9 + 1 * 3 + 2] + ra[3 * 25 + 4 * 5 + 4] * p[2 * 9 + 1 * 3 + 1] + ra[4 * 25 + 3 * 5 + 3] * p[1 * 9 + 2 * 3 + 2] + ra[4 * 25 + 3 * 5 + 4] * p[1 * 9 + 2 * 3 + 1] + ra[4 * 25 + 4 * 5 + 3] * p[1 * 9 + 1 * 3 + 2] + ra[4 * 25 + 4 * 5 + 4] * p[1 * 9 + 1 * 3 + 1];
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

/**
 * Extracts border planes from buf_cuboid and writes result into buf_res.
 * The planes are stored in the following order:
 *   front (yz), back (yz), top (xz), bottom (xz), left (xy), right (xy)
 * Hence, the borders of the planes are stored multiple times.
 * The planes itself are stored as follows:
 * - yz: j-major, i.e. forall j { forall k { ... } }
 * - xz: i-major, i.e. forall i { forall k { ... } }
 * - xy: i-major, i.e. forall i { forall k { ... } }
 *
 * This kernel must be called as a 1d kernel with
 *   #borderCells = ghosts_m*n*o * ghosts_n*m*o * ghosts_o*n*m
 * work-items.
 * Arguments:
 * * buf_cuboid: CuboidGPU::buffer of size mgh*ngh*ogh
 * * buf_res: CuboidGPU::buffer of size 1*1*#borderCells
 * * m, n, o: Extents of buf_cuboid excluding ghost cells
 * * mgh, ngh, ogh: Extents of buf_cuboid including ghost cells
 * * ghosts_m, ghosts_n, ghosts_o: Ghost cell amount of buf_cuboid
 */
__kernel void extract_border_planes(
    __global double* buf_cuboid,
    __global double* buf_res,
    int m, int n, int o,
    int mgh, int ngh, int ogh,
    int ghosts_m, int ghosts_n, int ghosts_o)
{
    // plane sizes
    int yz = ngh * ogh;
    int xz = mgh * ogh;
    int xy = mgh * ngh;

    // Get buf_cuboid's 1d and 3d indices
    int idx = get_global_id(0);

    // Front planes
    if (idx < ghosts_m * yz)
    {
        int i = idx / yz + ghosts_m;
        int j = (idx - (i - ghosts_m) * yz) / ogh;
        int k = idx % ogh;
        buf_res[idx] = buf_cuboid[i * ngh * ogh + j * ogh + k];
    }
    // Back planes
    else if (idx < 2 * ghosts_m * yz)
    {
        idx -= ghosts_m * yz; // reset to 0 for index calculation
        int i = idx / yz + m;
        int j = (idx - (i - m) * yz) / ogh;
        int k = idx % ogh;
        buf_res[idx + ghosts_m * yz] = buf_cuboid[i * ngh * ogh + j * ogh + k];
    }
    // Top planes
    else if (idx < 2 * ghosts_m * yz + ghosts_n * xz)
    {
        idx -= 2 * ghosts_m * yz; // reset to 0 for index calculation
        int j = idx / xz + ghosts_n;
        int i = (idx - (j - ghosts_n) * xz) / ogh;
        int k = idx % ogh;
        buf_res[idx + 2 * ghosts_m * yz] = buf_cuboid[i * ngh * ogh + j * ogh + k];
    }
    // Bottom planes
    else if (idx < 2 * ghosts_m * yz + 2 * ghosts_n * xz)
    {
        idx -= 2 * ghosts_m * yz + ghosts_n * xz; // reset to 0 for index calculation
        int j = idx / xz + n;
        int i = (idx - (j - n) * xz) / ogh;
        int k = idx % ogh;
        buf_res[idx + 2 * ghosts_m * yz + ghosts_n * xz] = buf_cuboid[i * ngh * ogh + j * ogh + k];
    }
    // Left planes
    else if (idx < 2 * ghosts_m * yz + 2 * ghosts_n * xz + ghosts_o * xy)
    {
        idx -= 2 * ghosts_m * yz + 2 * ghosts_n * xz; // reset to 0 for index calculation
        int k = idx / xy + ghosts_o;
        int i = (idx - (k - ghosts_o) * xy) / ngh;
        int j = idx % ngh;
        buf_res[idx + 2 * ghosts_m * yz + 2 * ghosts_n * xz] = buf_cuboid[i * ngh * ogh + j * ogh + k];
    }
    // Right planes
    else if (idx < 2 * ghosts_m * yz + 2 * ghosts_n * xz + 2 * ghosts_o * xy)
    {
        idx -= 2 * ghosts_m * yz + 2 * ghosts_n * xz + ghosts_o * xy; // reset to 0 for index calculation
        int k = idx / xy + o;
        int i = (idx - (k - o) * xy) / ngh;
        int j = idx % ngh;
        buf_res[idx + 2 * ghosts_m * yz + 2 * ghosts_n * xz + ghosts_o * xy] = buf_cuboid[i * ngh * ogh + j * ogh + k];
    }
}

/**
 * Paste ghosts from border planes into buf_cuboid.
 * Since corners and edges are not up-to-date for alle planes, some
 *   cells at the borders are ignored. Every cuboid ghost cell should be written
 *   exactly once. TODO some wi's do nothing as result, so maybe optimize.
 * The planes are stored in the following order:
 *   front (yz), back (yz), top (xz), bottom (xz), left (xy), right (xy)
 * Hence, the borders of the planes are stored multiple times.
 * The planes itself are stored as follows:
 * - yz: j-major, i.e. forall j { forall k { ... } }
 * - xz: i-major, i.e. forall i { forall k { ... } }
 * - xy: i-major, i.e. forall i { forall k { ... } }
 *
 * This kernel must be called as a 1d kernel with
 *   #borderCells = ghosts_m*n*o * ghosts_n*m*o * ghosts_o*n*m
 * work-items.
 * Arguments:
 * * buf_cuboid: CuboidGPU::buffer of size mgh*ngh*ogh
 * * buf_planes: CuboidGPU::buffer of size 1*1*#borderCells
 * * m, n, o: Extents of buf_cuboid excluding ghost cells
 * * mgh, ngh, ogh: Extents of buf_cuboid including ghost cells
 * * ghosts_m, ghosts_n, ghosts_o: Ghost cell amount of buf_cuboid
 */
__kernel void paste_ghosts_from_border_planes(
    __global double* buf_cuboid,
    __global double* buf_planes,
    int m, int n, int o,
    int mgh, int ngh, int ogh,
    int ghosts_m, int ghosts_n, int ghosts_o)
{
    // plane sizes
    int yz = ngh * ogh;
    int xz = mgh * ogh;
    int xy = mgh * ngh;

    // Get buf_cuboid's 1d and 3d indices
    int idx = get_global_id(0);

    // Front planes (back ghosts)
    if (idx < ghosts_m * yz)
    {
        int i = idx / yz + m + ghosts_m;
        int j = (idx - (i - (m + ghosts_m)) * yz) / ogh;
        int k = idx % ogh;

        // No corners or edges, only ghosts directly adjacent to real back face
        if (j >= ghosts_n && j < n + ghosts_n && k >= ghosts_o && k < o + ghosts_o)
            buf_cuboid[i * ngh * ogh + j * ogh + k] = buf_planes[idx];
    }
    // Back planes (front ghosts)
    else if (idx < 2 * ghosts_m * yz)
    {
        idx -= ghosts_m * yz; // reset to 0 for index calculation
        int i = idx / yz;
        int j = (idx - i * yz) / ogh;
        int k = idx % ogh;

        // No corners or edges, only ghosts directly adjacent to real front face
        if (j >= ghosts_n && j < n + ghosts_n && k >= ghosts_o && k < o + ghosts_o)
            buf_cuboid[i * ngh * ogh + j * ogh + k] = buf_planes[idx + ghosts_m * yz];
    }
    // Top planes (bottom ghosts)
    else if (idx < 2 * ghosts_m * yz + ghosts_n * xz)
    {
        idx -= 2 * ghosts_m * yz; // reset to 0 for index calculation
        int j = idx / xz + n + ghosts_n;
        int i = (idx - (j - (n + ghosts_n)) * xz) / ogh;
        int k = idx % ogh;

        // Ignore left and right ghost cells, but include front and back ghosts
        if (k >= ghosts_o && k < o + ghosts_o)
            buf_cuboid[i * ngh * ogh + j * ogh + k] = buf_planes[idx + 2 * ghosts_m * yz];
    }
    // // Bottom planes (top ghosts)
    else if (idx < 2 * ghosts_m * yz + 2 * ghosts_n * xz)
    {
        idx -= 2 * ghosts_m * yz + ghosts_n * xz; // reset to 0 for index calculation
        int j = idx / xz;
        int i = (idx - j * xz) / ogh;
        int k = idx % ogh;

        // Ignore left and right ghost cells, but include front and back ghosts
        if (k >= ghosts_o && k < o + ghosts_o)
            buf_cuboid[i * ngh * ogh + j * ogh + k] = buf_planes[idx + 2 * ghosts_m * yz + ghosts_n * xz];
    }
    // Left planes (right ghosts)
    else if (idx < 2 * ghosts_m * yz + 2 * ghosts_n * xz + ghosts_o * xy)
    {
        idx -= 2 * ghosts_m * yz + 2 * ghosts_n * xz; // reset to 0 for index calculation
        int k = idx / xy + o + ghosts_o;
        int i = (idx - (k - (o + ghosts_o)) * xy) / ngh;
        int j = idx % ngh;
        buf_cuboid[i * ngh * ogh + j * ogh + k] = buf_planes[idx + 2 * ghosts_m * yz + 2 * ghosts_n * xz];
    }
    // Right planes (left ghosts)
    else if (idx < 2 * ghosts_m * yz + 2 * ghosts_n * xz + 2 * ghosts_o * xy)
    {
        idx -= 2 * ghosts_m * yz + 2 * ghosts_n * xz + ghosts_o * xy; // reset to 0 for index calculation
        int k = idx / xy;
        int i = (idx - k * xy) / ngh;
        int j = idx % ngh;
        buf_cuboid[i * ngh * ogh + j * ogh + k] = buf_planes[idx + 2 * ghosts_m * yz + 2 * ghosts_n * xz + ghosts_o * xy];
    }
}

/**
 * Extracts border planes from buf_stencil and writes result into buf_res.
 * The planes are stored in the following order, each for all coefficients:
 *   front (yz), back (yz), top (xz), bottom (xz), left (xy), right (xy)
 * Hence, the borders of the planes are stored multiple times.
 * The planes itself are stored as follows:
 * - yz: j-major, i.e. forall j { forall k { ... } }
 * - xz: i-major, i.e. forall i { forall k { ... } }
 * - xy: i-major, i.e. forall i { forall k { ... } }
 *
 * This kernel must be called as a 1d kernel with
 *   #borderCells = (ghosts_m*n*o * ghosts_n*m*o * ghosts_o*n*m) * 27
 * work-items.
 * Hardcoded for a 27p stencil.
 * Arguments:
 * * buf_stencil: VaryingStencilGpu of size mgh*ngh*ogh
 * * buf_res: std::vector of size #borderCells
 * * m, n, o: Extents of buf_stencil excluding ghost cells
 * * mgh, ngh, ogh: Extents of buf_stencil including ghost cells
 * * ghosts_m, ghosts_n, ghosts_o: Ghost cell amount of buf_stencil
 */
__kernel void extract_border_planes_varying_stencil(
    __global double* buf_stencil,
    __global double* buf_res,
    int m, int n, int o,
    int mgh, int ngh, int ogh,
    int ghosts_m, int ghosts_n, int ghosts_o)
{
    // plane sizes
    int yz = ngh * ogh;
    int xz = mgh * ogh;
    int xy = mgh * ngh;

    // size of the ghosted grid
    int gridsize = mgh * ngh * ogh;

    // wi-index = index in output buffer buf_res
    int idx = get_global_id(0);

    // Front planes (for each coefficient)
    if (idx < ghosts_m * yz * 27)
    {
        int idx_coeff = idx / (ghosts_m * yz);            // 1d index of the current coefficient
        int idx_grid = idx - idx_coeff * (ghosts_m * yz); // local index of the grid point inside the grid of one coefficient

        int i = idx_grid / yz + ghosts_m;
        int j = (idx_grid - (i - ghosts_m) * yz) / ogh;
        int k = idx_grid % ogh;
        buf_res[idx] = buf_stencil[idx_coeff * gridsize + i * ngh * ogh + j * ogh + k];
    }
    // Back planes
    else if (idx < 2 * ghosts_m * yz * 27)
    {
        int idx_resbuf = idx;
        idx -= ghosts_m * yz * 27;                        // reset to 0 for index calculation
        int idx_coeff = idx / (ghosts_m * yz);            // 1d index of the current coefficient
        int idx_grid = idx - idx_coeff * (ghosts_m * yz); // local index of the grid point inside the grid of one coefficient

        int i = idx_grid / yz + m;
        int j = (idx_grid - (i - m) * yz) / ogh;
        int k = idx_grid % ogh;

        buf_res[idx_resbuf] = buf_stencil[idx_coeff * gridsize + i * ngh * ogh + j * ogh + k];
    }
    // Top planes
    else if (idx < (2 * ghosts_m * yz + ghosts_n * xz) * 27)
    {
        int idx_resbuf = idx;
        idx -= 2 * ghosts_m * yz * 27;                    // reset to 0 for index calculation
        int idx_coeff = idx / (ghosts_n * xz);            // 1d index of the current coefficient
        int idx_grid = idx - idx_coeff * (ghosts_n * xz); // local index of the grid point inside the grid of one coefficient

        int j = idx_grid / xz + ghosts_n;
        int i = (idx_grid - (j - ghosts_n) * xz) / ogh;
        int k = idx_grid % ogh;

        buf_res[idx_resbuf] = buf_stencil[idx_coeff * gridsize + i * ngh * ogh + j * ogh + k];
    }
    // Bottom planes
    else if (idx < (2 * ghosts_m * yz + 2 * ghosts_n * xz) * 27)
    {
        int idx_resbuf = idx;
        idx -= (2 * ghosts_m * yz + ghosts_n * xz) * 27;  // reset to 0 for index calculation
        int idx_coeff = idx / (ghosts_n * xz);            // 1d index of the current coefficient
        int idx_grid = idx - idx_coeff * (ghosts_n * xz); // local index of the grid point inside the grid of one coefficient

        int j = idx_grid / xz + n;
        int i = (idx_grid - (j - n) * xz) / ogh;
        int k = idx_grid % ogh;

        buf_res[idx_resbuf] = buf_stencil[idx_coeff * gridsize + i * ngh * ogh + j * ogh + k];
    }
    // Left planes
    else if (idx < (2 * ghosts_m * yz + 2 * ghosts_n * xz + ghosts_o * xy) * 27)
    {
        int idx_resbuf = idx;
        idx -= (2 * ghosts_m * yz + 2 * ghosts_n * xz) * 27; // reset to 0 for index calculation
        int idx_coeff = idx / (ghosts_o * xy);               // 1d index of the current coefficient
        int idx_grid = idx - idx_coeff * (ghosts_o * xy);    // local index of the grid point inside the grid of one coefficient

        int k = idx_grid / xy + ghosts_o;
        int i = (idx_grid - (k - ghosts_o) * xy) / ngh;
        int j = idx_grid % ngh;

        buf_res[idx_resbuf] = buf_stencil[idx_coeff * gridsize + i * ngh * ogh + j * ogh + k];
    }
    // Right planes
    else if (idx < (2 * ghosts_m * yz + 2 * ghosts_n * xz + 2 * ghosts_o * xy) * 27)
    {
        int idx_resbuf = idx;
        idx -= (2 * ghosts_m * yz + 2 * ghosts_n * xz + ghosts_o * xy) * 27; // reset to 0 for index calculation
        int idx_coeff = idx / (ghosts_o * xy);                               // 1d index of the current coefficient
        int idx_grid = idx - idx_coeff * (ghosts_o * xy);                    // local index of the grid point inside the grid of one coefficient

        int k = idx_grid / xy + o;
        int i = (idx_grid - (k - o) * xy) / ngh;
        int j = idx_grid % ngh;

        buf_res[idx_resbuf] = buf_stencil[idx_coeff * gridsize + i * ngh * ogh + j * ogh + k];
    }
}

/**
 * Paste ghosts from border planes "buf_ghosts" into buf_stencil.
 * Since corners and edges are not up-to-date for alle planes, some
 *   cells at the borders are ignored. Every cuboid ghost cell should be written
 *   exactly once. TODO some wi's do nothing as result, so maybe optimize.
 * The planes are stored in the following order:
 *   front (yz), back (yz), top (xz), bottom (xz), left (xy), right (xy)
 * Hence, the borders of the planes are stored multiple times.
 * The planes itself are stored as follows:
 * - yz: j-major, i.e. forall j { forall k { ... } }
 * - xz: i-major, i.e. forall i { forall k { ... } }
 * - xy: i-major, i.e. forall i { forall k { ... } }
 *
 * This kernel must be called as a 1d kernel with
 *   #borderCells = ghosts_m*n*o * ghosts_n*m*o * ghosts_o*n*m * 27
 * work-items.
 * Arguments:
 * * buf_cuboid: VaryingStencilGpu::buffer of size mgh*ngh*ogh*27
 * * buf_ghosts BufferGpu::buffer of size 1*1*#borderCells
 * * m, n, o: Extents of buf_cuboid excluding ghost cells
 * * mgh, ngh, ogh: Extents of buf_cuboid including ghost cells
 * * ghosts_m, ghosts_n, ghosts_o: Ghost cell amount of buf_cuboid
 */
__kernel void paste_ghosts_from_border_planes_varying_stencil(
    __global double* buf_stencil,
    __global double* buf_ghosts,
    int m, int n, int o,
    int mgh, int ngh, int ogh,
    int ghosts_m, int ghosts_n, int ghosts_o)
{
    // plane sizes
    int yz = ngh * ogh;
    int xz = mgh * ogh;
    int xy = mgh * ngh;

    // size of the ghosted grid
    int gridsize = mgh * ngh * ogh;

    // wi-index = index in input buffer buf_ghosts
    int idx = get_global_id(0);

    // Front planes (back ghosts)
    if (idx < ghosts_m * yz * 27)
    {
        int idx_coeff = idx / (ghosts_m * yz);            // 1d index of the current coefficient
        int idx_grid = idx - idx_coeff * (ghosts_m * yz); // local index of the grid point inside the grid of one coefficient

        int i = idx_grid / yz + m + ghosts_m;
        int j = (idx_grid - (i - (m + ghosts_m)) * yz) / ogh;
        int k = idx_grid % ogh;

        // No corners or edges, only ghosts directly adjacent to real back face
        if (j >= ghosts_n && j < n + ghosts_n && k >= ghosts_o && k < o + ghosts_o)
            buf_stencil[idx_coeff * gridsize + i * ngh * ogh + j * ogh + k] = buf_ghosts[idx];
    }
    // Back planes (front ghosts)
    else if (idx < 2 * ghosts_m * yz * 27)
    {
        idx -= ghosts_m * yz * 27;                        // reset to 0 for index calculation
        int idx_coeff = idx / (ghosts_m * yz);            // 1d index of the current coefficient
        int idx_grid = idx - idx_coeff * (ghosts_m * yz); // local index of the grid point inside the grid of one coefficient

        int i = idx_grid / yz;
        int j = (idx_grid - i * yz) / ogh;
        int k = idx_grid % ogh;

        // No corners or edges, only ghosts directly adjacent to real front face
        if (j >= ghosts_n && j < n + ghosts_n && k >= ghosts_o && k < o + ghosts_o)
            buf_stencil[idx_coeff * gridsize + i * ngh * ogh + j * ogh + k] = buf_ghosts[idx + ghosts_m * yz * 27];
    }
    // Top planes (bottom ghosts)
    else if (idx < (2 * ghosts_m * yz + ghosts_n * xz) * 27)
    {
        idx -= 2 * ghosts_m * yz * 27;                    // reset to 0 for index calculation
        int idx_coeff = idx / (ghosts_n * xz);            // 1d index of the current coefficient
        int idx_grid = idx - idx_coeff * (ghosts_n * xz); // local index of the grid point inside the grid of one coefficient

        int j = idx_grid / xz + n + ghosts_n;
        int i = (idx_grid - (j - (n + ghosts_n)) * xz) / ogh;
        int k = idx_grid % ogh;

        // Ignore left and right ghost cells, but include front and back ghosts
        if (k >= ghosts_o && k < o + ghosts_o)
            buf_stencil[idx_coeff * gridsize + i * ngh * ogh + j * ogh + k] = buf_ghosts[idx + 2 * ghosts_m * yz * 27];
    }
    // Bottom planes (top ghosts)
    else if (idx < (2 * ghosts_m * yz + 2 * ghosts_n * xz) * 27)
    {
        idx -= (2 * ghosts_m * yz + ghosts_n * xz) * 27;  // reset to 0 for index calculation
        int idx_coeff = idx / (ghosts_n * xz);            // 1d index of the current coefficient
        int idx_grid = idx - idx_coeff * (ghosts_n * xz); // local index of the grid point inside the grid of one coefficient

        int j = idx_grid / xz;
        int i = (idx_grid - j * xz) / ogh;
        int k = idx_grid % ogh;

        // Ignore left and right ghost cells, but include front and back ghosts
        if (k >= ghosts_o && k < o + ghosts_o)
            buf_stencil[idx_coeff * gridsize + i * ngh * ogh + j * ogh + k] = buf_ghosts[idx + (2 * ghosts_m * yz + ghosts_n * xz) * 27];
    }
    // Left planes (right ghosts)
    else if (idx < (2 * ghosts_m * yz + 2 * ghosts_n * xz + ghosts_o * xy) * 27)
    {
        idx -= (2 * ghosts_m * yz + 2 * ghosts_n * xz) * 27; // reset to 0 for index calculation
        int idx_coeff = idx / (ghosts_o * xy);               // 1d index of the current coefficient
        int idx_grid = idx - idx_coeff * (ghosts_o * xy);    // local index of the grid point inside the grid of one coefficient

        int k = idx_grid / xy + o + ghosts_o;
        int i = (idx_grid - (k - (o + ghosts_o)) * xy) / ngh;
        int j = idx_grid % ngh;
        buf_stencil[idx_coeff * gridsize + i * ngh * ogh + j * ogh + k] = buf_ghosts[idx + (2 * ghosts_m * yz + 2 * ghosts_n * xz) * 27];
    }
    // Right planes (left ghosts)
    else if (idx < (2 * ghosts_m * yz + 2 * ghosts_n * xz + 2 * ghosts_o * xy) * 27)
    {
        idx -= (2 * ghosts_m * yz + 2 * ghosts_n * xz + ghosts_o * xy) * 27; // reset to 0 for index calculation
        int idx_coeff = idx / (ghosts_o * xy);                               // 1d index of the current coefficient
        int idx_grid = idx - idx_coeff * (ghosts_o * xy);                    // local index of the grid point inside the grid of one coefficient

        int k = idx_grid / xy;
        int i = (idx_grid - k * xy) / ngh;
        int j = idx_grid % ngh;
        buf_stencil[idx_coeff * gridsize + i * ngh * ogh + j * ogh + k] = buf_ghosts[idx + (2 * ghosts_m * yz + 2 * ghosts_n * xz + ghosts_o * xy) * 27];
    }
}

/**
 * Extracts border planes from buf_cuboid and writes result into buf_res.
 * The planes are stored in the following order:
 *   front (yz), back (yz), top (xz), bottom (xz), left (xy), right (xy)
 * Hence, the borders of the planes are stored multiple times.
 * The planes itself are stored as follows:
 * - yz: j-major, i.e. forall j { forall k { ... } }
 * - xz: i-major, i.e. forall i { forall k { ... } }
 * - xy: i-major, i.e. forall i { forall k { ... } }
 *
 * This kernel must be called as a 1d kernel with
 *   #borderCells = ghosts_m*n*o * ghosts_n*m*o * ghosts_o*n*m
 * work-items.
 * Block entries for one grid point lie in memory consecutively, i.e. gp000_m0, gp000_m1, gp001_m0, gp001_m1, etc.
 * Arguments:
 * * buf_cuboid: CuboidBSGPU::buffer of size mgh*ngh*ogh
 * * buf_res: CuboidGPU::buffer of size 1*1*#borderCells
 * * m, n, o: Extents of buf_cuboid excluding ghost cells
 * * mgh, ngh, ogh: Extents of buf_cuboid including ghost cells
 * * ghosts_m, ghosts_n, ghosts_o: Ghost cell amount of buf_cuboid
 * * blocksize: Number of vector components
 */
__kernel void extract_border_planes_cuboidbs(
    __global double* buf_cuboid,
    __global double* buf_res,
    int m, int n, int o,
    int mgh, int ngh, int ogh,
    int ghosts_m, int ghosts_n, int ghosts_o,
    const int blocksize)
{
    // plane sizes
    int yz = ngh * ogh;
    int xz = mgh * ogh;
    int xy = mgh * ngh;

    // Get buf_cuboid's 1d index for the first matrix entry
    int idx = get_global_id(0);

    // Front planes
    if (idx < ghosts_m * yz)
    {
        int i = idx / yz + ghosts_m;
        int j = (idx - (i - ghosts_m) * yz) / ogh;
        int k = idx % ogh;
        int gp_base_idx = (i * ngh * ogh + j * ogh + k) * blocksize; // gp base index in source cuboid
        for (int bi = 0; bi < blocksize; bi++)
        {
            buf_res[idx * blocksize + bi] = buf_cuboid[gp_base_idx + bi]; // store block entries for each grid point consecutively
        }
    }
    // Back planes
    else if (idx < 2 * ghosts_m * yz)
    {
        idx -= ghosts_m * yz; // reset to 0 for index calculation
        int i = idx / yz + m;
        int j = (idx - (i - m) * yz) / ogh;
        int k = idx % ogh;
        int gp_base_idx = (i * ngh * ogh + j * ogh + k) * blocksize; // gp base index in source cuboid
        for (int bi = 0; bi < blocksize; bi++)
        {
            buf_res[(idx + ghosts_m * yz) * blocksize + bi] = buf_cuboid[gp_base_idx + bi];
        }
    }
    // Top planes
    else if (idx < 2 * ghosts_m * yz + ghosts_n * xz)
    {
        idx -= 2 * ghosts_m * yz; // reset to 0 for index calculation
        int j = idx / xz + ghosts_n;
        int i = (idx - (j - ghosts_n) * xz) / ogh;
        int k = idx % ogh;
        int gp_base_idx = (i * ngh * ogh + j * ogh + k) * blocksize; // gp base index in source cuboid
        for (int bi = 0; bi < blocksize; bi++)
        {
            buf_res[(idx + 2 * ghosts_m * yz) * blocksize + bi] = buf_cuboid[gp_base_idx + bi];
        }
    }
    // Bottom planes
    else if (idx < 2 * ghosts_m * yz + 2 * ghosts_n * xz)
    {
        idx -= 2 * ghosts_m * yz + ghosts_n * xz; // reset to 0 for index calculation
        int j = idx / xz + n;
        int i = (idx - (j - n) * xz) / ogh;
        int k = idx % ogh;
        int gp_base_idx = (i * ngh * ogh + j * ogh + k) * blocksize; // gp base index in source cuboid
        for (int bi = 0; bi < blocksize; bi++)
        {
            buf_res[(idx + 2 * ghosts_m * yz + ghosts_n * xz) * blocksize + bi] = buf_cuboid[gp_base_idx + bi];
        }
    }
    // Left planes
    else if (idx < 2 * ghosts_m * yz + 2 * ghosts_n * xz + ghosts_o * xy)
    {
        idx -= 2 * ghosts_m * yz + 2 * ghosts_n * xz; // reset to 0 for index calculation
        int k = idx / xy + ghosts_o;
        int i = (idx - (k - ghosts_o) * xy) / ngh;
        int j = idx % ngh;
        int gp_base_idx = (i * ngh * ogh + j * ogh + k) * blocksize; // gp base index in source cuboid
        for (int bi = 0; bi < blocksize; bi++)
        {
            buf_res[(idx + 2 * ghosts_m * yz + 2 * ghosts_n * xz) * blocksize + bi] = buf_cuboid[gp_base_idx + bi];
        }
    }
    // Right planes
    else if (idx < 2 * ghosts_m * yz + 2 * ghosts_n * xz + 2 * ghosts_o * xy)
    {
        idx -= 2 * ghosts_m * yz + 2 * ghosts_n * xz + ghosts_o * xy; // reset to 0 for index calculation
        int k = idx / xy + o;
        int i = (idx - (k - o) * xy) / ngh;
        int j = idx % ngh;
        int gp_base_idx = (i * ngh * ogh + j * ogh + k) * blocksize; // gp base index in source cuboid
        for (int bi = 0; bi < blocksize; bi++)
        {
            buf_res[(idx + 2 * ghosts_m * yz + 2 * ghosts_n * xz + ghosts_o * xy) * blocksize + bi] = buf_cuboid[gp_base_idx + bi];
        }
    }
}

/**
 * Paste ghosts from border planes into buf_cuboid.
 * Since corners and edges are not up-to-date for alle planes, some
 *   cells at the borders are ignored. Every cuboid ghost cell should be written
 *   exactly once. TODO some wi's do nothing as result, so maybe optimize.
 * The planes are stored in the following order:
 *   front (yz), back (yz), top (xz), bottom (xz), left (xy), right (xy)
 * Hence, the borders of the planes are stored multiple times.
 * The planes itself are stored as follows:
 * - yz: j-major, i.e. forall j { forall k { ... } }
 * - xz: i-major, i.e. forall i { forall k { ... } }
 * - xy: i-major, i.e. forall i { forall k { ... } }
 *
 * This kernel must be called as a 1d kernel with
 *   #borderCells = ghosts_m*n*o * ghosts_n*m*o * ghosts_o*n*m
 * work-items.
 * Arguments:
 * * buf_cuboid: CuboidBSGPU::buffer of size mgh*ngh*ogh
 * * buf_planes: CuboidGPU::buffer of size 1*1*#borderCells
 * * m, n, o: Extents of buf_cuboid excluding ghost cells
 * * mgh, ngh, ogh: Extents of buf_cuboid including ghost cells
 * * ghosts_m, ghosts_n, ghosts_o: Ghost cell amount of buf_cuboid
 */
__kernel void paste_ghosts_from_border_planes_cuboidbs(
    __global double* buf_cuboid,
    __global double* buf_planes,
    const int m, const int n, const int o,
    const int mgh, const int ngh, const int ogh,
    const int ghosts_m, const int ghosts_n, const int ghosts_o,
    const int blocksize)
{
    // plane sizes
    int yz = ngh * ogh;
    int xz = mgh * ogh;
    int xy = mgh * ngh;

    // Get buf_cuboid's 1d and 3d indices
    int idx = get_global_id(0);

    // Front planes (back ghosts)
    if (idx < ghosts_m * yz)
    {
        int i = idx / yz + m + ghosts_m;
        int j = (idx - (i - (m + ghosts_m)) * yz) / ogh;
        int k = idx % ogh;
        int gp_base_idx = (i * ngh * ogh + j * ogh + k) * blocksize; // gp base index in source cuboid

        // No corners or edges, only ghosts directly adjacent to real back face
        if (j >= ghosts_n && j < n + ghosts_n && k >= ghosts_o && k < o + ghosts_o)
        {
            for (int bi = 0; bi < blocksize; bi++)
            {
                buf_cuboid[gp_base_idx + bi] = buf_planes[idx * blocksize + bi];
            }
        }
    }
    // Back planes (front ghosts)
    else if (idx < 2 * ghosts_m * yz)
    {
        idx -= ghosts_m * yz; // reset to 0 for index calculation
        int i = idx / yz;
        int j = (idx - i * yz) / ogh;
        int k = idx % ogh;
        int gp_base_idx = (i * ngh * ogh + j * ogh + k) * blocksize; // gp base index in source cuboid

        // No corners or edges, only ghosts directly adjacent to real front face
        if (j >= ghosts_n && j < n + ghosts_n && k >= ghosts_o && k < o + ghosts_o)
        {
            for (int bi = 0; bi < blocksize; bi++)
            {
                buf_cuboid[gp_base_idx + bi] = buf_planes[(idx + ghosts_m * yz) * blocksize + bi];
            }
        }
    }
    // Top planes (bottom ghosts)
    else if (idx < 2 * ghosts_m * yz + ghosts_n * xz)
    {
        idx -= 2 * ghosts_m * yz; // reset to 0 for index calculation
        int j = idx / xz + n + ghosts_n;
        int i = (idx - (j - (n + ghosts_n)) * xz) / ogh;
        int k = idx % ogh;
        int gp_base_idx = (i * ngh * ogh + j * ogh + k) * blocksize; // gp base index in source cuboid

        // Ignore left and right ghost cells, but include front and back ghosts
        if (k >= ghosts_o && k < o + ghosts_o)
        {
            for (int bi = 0; bi < blocksize; bi++)
            {
                buf_cuboid[gp_base_idx + bi] = buf_planes[(idx + 2 * ghosts_m * yz) * blocksize + bi];
            }
        }
    }
    // // Bottom planes (top ghosts)
    else if (idx < 2 * ghosts_m * yz + 2 * ghosts_n * xz)
    {
        idx -= 2 * ghosts_m * yz + ghosts_n * xz; // reset to 0 for index calculation
        int j = idx / xz;
        int i = (idx - j * xz) / ogh;
        int k = idx % ogh;
        int gp_base_idx = (i * ngh * ogh + j * ogh + k) * blocksize; // gp base index in source cuboid

        // Ignore left and right ghost cells, but include front and back ghosts
        if (k >= ghosts_o && k < o + ghosts_o)
        {
            for (int bi = 0; bi < blocksize; bi++)
            {
                buf_cuboid[gp_base_idx + bi] = buf_planes[(idx + 2 * ghosts_m * yz + ghosts_n * xz) * blocksize + bi];
            }
        }
    }
    // Left planes (right ghosts)
    else if (idx < 2 * ghosts_m * yz + 2 * ghosts_n * xz + ghosts_o * xy)
    {
        idx -= 2 * ghosts_m * yz + 2 * ghosts_n * xz; // reset to 0 for index calculation
        int k = idx / xy + o + ghosts_o;
        int i = (idx - (k - (o + ghosts_o)) * xy) / ngh;
        int j = idx % ngh;
        int gp_base_idx = (i * ngh * ogh + j * ogh + k) * blocksize; // gp base index in source cuboid

        for (int bi = 0; bi < blocksize; bi++)
        {
            buf_cuboid[gp_base_idx + bi] = buf_planes[(idx + 2 * ghosts_m * yz + 2 * ghosts_n * xz) * blocksize + bi];
        }
    }
    // Right planes (left ghosts)
    else if (idx < 2 * ghosts_m * yz + 2 * ghosts_n * xz + 2 * ghosts_o * xy)
    {
        idx -= 2 * ghosts_m * yz + 2 * ghosts_n * xz + ghosts_o * xy; // reset to 0 for index calculation
        int k = idx / xy;
        int i = (idx - k * xy) / ngh;
        int j = idx % ngh;
        int gp_base_idx = (i * ngh * ogh + j * ogh + k) * blocksize; // gp base index in source cuboid

        for (int bi = 0; bi < blocksize; bi++)
        {
            buf_cuboid[gp_base_idx + bi] = buf_planes[(idx + 2 * ghosts_m * yz + 2 * ghosts_n * xz + ghosts_o * xy) * blocksize + bi];
        }
    }
}

/**
 * Extracts border planes from buf_stencil and writes result into buf_res.
 * The planes are stored in the following order, each for all coefficients:
 *   front (yz), back (yz), top (xz), bottom (xz), left (xy), right (xy)
 * Hence, the borders of the planes are stored multiple times.
 * The planes itself are stored as follows:
 * - yz: j-major, i.e. forall j { forall k { ... } }
 * - xz: i-major, i.e. forall i { forall k { ... } }
 * - xy: i-major, i.e. forall i { forall k { ... } }
 * For each matrix entry, all coefficients with all planes are stored.
 * bi=0, bj=0: all coeffs, all 6 planes, then
 * bi=0, bj=1: all coeffs, all 6 planes and so on.
 *
 * This kernel must be called as a 1d kernel with
 *   #wis = (ghosts_m*n*o * ghosts_n*m*o * ghosts_o*n*m) * 27
 * work-items.
 * Hardcoded for a 27p blockstencil.
 * Memory layout blockstencil: [bi][bj][ci][cj][ck][x][y][z]
 * Arguments:
 * * buf_stencil: BlockstencilGpu of size mgh*ngh*ogh*27*blocksize^2
 * * buf_res: std::vector of size (ghosts_m*n*o * ghosts_n*m*o * ghosts_o*n*m) * 27 * blocksize^2
 * * m, n, o: Extents of buf_stencil excluding ghost cells
 * * mgh, ngh, ogh: Extents of buf_stencil including ghost cells
 * * ghosts_m, ghosts_n, ghosts_o: Ghost cell amount of buf_stencil
 * * blocksize: Size of block in one dimension
 */
__kernel void extract_border_planes_blockstencil(
    __global double* buf_stencil,
    __global double* buf_res,
    const int m, const int n, const int o,
    const int mgh, const int ngh, const int ogh,
    const int ghosts_m, const int ghosts_n, const int ghosts_o,
    const int blocksize)
{
    // plane sizes
    int yz = ngh * ogh;
    int xz = mgh * ogh;
    int xy = mgh * ngh;

    // size of the ghosted grid
    int gridsize = mgh * ngh * ogh;

    // size of 27pt stencil including ghosted grid
    int gridsizeStencil = gridsize * 27;
    int blocksize2 = blocksize * blocksize;

    int yzgh = yz * ghosts_m;
    int xzgh = xz * ghosts_n;
    int xygh = xy * ghosts_o;

    // size of one matrix entry (coeffs and gps) for each plane
    int yzRessizeOneMatrixEntry = yzgh * 27;
    int xzRessizeOneMatrixEntry = xzgh * 27;
    int xyRessizeOneMatrixEntry = xygh * 27;

    // wi-index = index in output buffer buf_res
    int idx = get_global_id(0);

    // Front planes (for each coefficient)
    if (idx < ghosts_m * yz * 27)
    {
        int idx_coeff = idx / (ghosts_m * yz);            // 1d index of the current coefficient
        int idx_grid = idx - idx_coeff * (ghosts_m * yz); // local index of the grid point inside the grid of one coefficient

        int i = idx_grid / yz + ghosts_m;
        int j = (idx_grid - (i - ghosts_m) * yz) / ogh;
        int k = idx_grid % ogh;
        for (int b = 0; b < blocksize2; b++)
        {
            // if (idx == 0)
            // {
            //     printf("b: %i, idx_coeff: %i, idx_grid: %i, i: %i, j: %i, k: %i\n", b, idx_coeff, idx_grid, i, j, k);
            //     printf("idx + b * yzRessizeOneMatrixEntry: %i\n", idx + b * yzRessizeOneMatrixEntry);
            //     printf("b * gridsizeStencil + idx_coeff * gridsize + i * ngh * ogh + j * ogh + k: %i\n", b * gridsizeStencil + idx_coeff * gridsize + i * ngh * ogh + j * ogh + k);
            // }
            buf_res[idx + b * yzRessizeOneMatrixEntry] = buf_stencil[b * gridsizeStencil + idx_coeff * gridsize + i * ngh * ogh + j * ogh + k];
        }
    }
    // Back planes
    else if (idx < 2 * ghosts_m * yz * 27)
    {
        idx -= ghosts_m * yz * 27;                              // reset to 0 for index calculation
        int idx_resbuf = ghosts_m * yz * 27 * blocksize2 + idx; // add resbuf offset to idx
        int idx_coeff = idx / (ghosts_m * yz);                  // 1d index of the current coefficient
        int idx_grid = idx - idx_coeff * (ghosts_m * yz);       // local index of the grid point inside the grid of one coefficient

        int i = idx_grid / yz + m;
        int j = (idx_grid - (i - m) * yz) / ogh;
        int k = idx_grid % ogh;

        for (int b = 0; b < blocksize2; b++)
        {
            buf_res[idx_resbuf + b * yzRessizeOneMatrixEntry] = buf_stencil[b * gridsizeStencil + idx_coeff * gridsize + i * ngh * ogh + j * ogh + k];
        }
    }
    // Top planes
    else if (idx < (2 * ghosts_m * yz + ghosts_n * xz) * 27)
    {
        idx -= 2 * ghosts_m * yz * 27;                                // reset to 0 for index calculation
        int idx_resbuf = (2 * ghosts_m * yz * 27) * blocksize2 + idx; // add resbuf offset to idx
        int idx_coeff = idx / (ghosts_n * xz);                        // 1d index of the current coefficient
        int idx_grid = idx - idx_coeff * (ghosts_n * xz);             // local index of the grid point inside the grid of one coefficient

        int j = idx_grid / xz + ghosts_n;
        int i = (idx_grid - (j - ghosts_n) * xz) / ogh;
        int k = idx_grid % ogh;

        for (int b = 0; b < blocksize2; b++)
        {
            buf_res[idx_resbuf + b * xzRessizeOneMatrixEntry] = buf_stencil[b * gridsizeStencil + idx_coeff * gridsize + i * ngh * ogh + j * ogh + k];
        }
    }
    // Bottom planes
    else if (idx < (2 * ghosts_m * yz + 2 * ghosts_n * xz) * 27)
    {
        idx -= (2 * ghosts_m * yz + ghosts_n * xz) * 27;                                // reset to 0 for index calculation
        int idx_resbuf = ((2 * ghosts_m * yz + ghosts_n * xz) * 27 * blocksize2) + idx; // add resbuf offset to idx
        int idx_coeff = idx / (ghosts_n * xz);                                          // 1d index of the current coefficient
        int idx_grid = idx - idx_coeff * (ghosts_n * xz);                               // local index of the grid point inside the grid of one coefficient

        int j = idx_grid / xz + n;
        int i = (idx_grid - (j - n) * xz) / ogh;
        int k = idx_grid % ogh;

        for (int b = 0; b < blocksize2; b++)
        {
            buf_res[idx_resbuf + b * xzRessizeOneMatrixEntry] = buf_stencil[b * gridsizeStencil + idx_coeff * gridsize + i * ngh * ogh + j * ogh + k];
        }
    }
    // Left planes
    else if (idx < (2 * ghosts_m * yz + 2 * ghosts_n * xz + ghosts_o * xy) * 27)
    {
        idx -= (2 * ghosts_m * yz + 2 * ghosts_n * xz) * 27;                                // reset to 0 for index calculation
        int idx_resbuf = ((2 * ghosts_m * yz + 2 * ghosts_n * xz) * 27 * blocksize2) + idx; // add resbuf offset to idx
        int idx_coeff = idx / (ghosts_o * xy);                                              // 1d index of the current coefficient
        int idx_grid = idx - idx_coeff * (ghosts_o * xy);                                   // local index of the grid point inside the grid of one coefficient

        int k = idx_grid / xy + ghosts_o;
        int i = (idx_grid - (k - ghosts_o) * xy) / ngh;
        int j = idx_grid % ngh;

        for (int b = 0; b < blocksize2; b++)
        {
            buf_res[idx_resbuf + b * xyRessizeOneMatrixEntry] = buf_stencil[b * gridsizeStencil + idx_coeff * gridsize + i * ngh * ogh + j * ogh + k];
        }
    }
    // Right planes
    else if (idx < (2 * ghosts_m * yz + 2 * ghosts_n * xz + 2 * ghosts_o * xy) * 27)
    {
        idx -= (2 * ghosts_m * yz + 2 * ghosts_n * xz + ghosts_o * xy) * 27;                                // reset to 0 for index calculation
        int idx_resbuf = ((2 * ghosts_m * yz + 2 * ghosts_n * xz + ghosts_o * xy) * 27 * blocksize2) + idx; // add resbuf offset to idx
        int idx_coeff = idx / (ghosts_o * xy);                                                              // 1d index of the current coefficient
        int idx_grid = idx - idx_coeff * (ghosts_o * xy);                                                   // local index of the grid point inside the grid of one coefficient

        int k = idx_grid / xy + o;
        int i = (idx_grid - (k - o) * xy) / ngh;
        int j = idx_grid % ngh;

        for (int b = 0; b < blocksize2; b++)
        {
            buf_res[idx_resbuf + b * xyRessizeOneMatrixEntry] = buf_stencil[b * gridsizeStencil + idx_coeff * gridsize + i * ngh * ogh + j * ogh + k];
        }
    }
}

/**
 * Paste ghosts from border planes "buf_ghosts" into buf_stencil.
 * Since corners and edges are not up-to-date for alle planes, some
 *   cells at the borders are ignored. Every cuboid ghost cell should be written
 *   exactly once. TODO some wi's do nothing as result, so maybe optimize.
 * The planes are stored in the following order:
 *   front (yz), back (yz), top (xz), bottom (xz), left (xy), right (xy)
 * Hence, the borders of the planes are stored multiple times.
 * The planes itself are stored as follows:
 * - yz: j-major, i.e. forall j { forall k { ... } }
 * - xz: i-major, i.e. forall i { forall k { ... } }
 * - xy: i-major, i.e. forall i { forall k { ... } }
 *
 * This kernel must be called as a 1d kernel with
 *   #wis = (ghosts_m*n*o * ghosts_n*m*o * ghosts_o*n*m) * 27
 * work-items.
 * Memory layout: [bi][bj][ci][cj][ck][x][y][z]
 * Arguments:
 * * buf_cuboid: VaryingStencilGpu::buffer of size mgh*ngh*ogh*27*blocksize^2
 * * buf_ghosts BufferGpu::buffer of size (ghosts_m*n*o * ghosts_n*m*o * ghosts_o*n*m) * 27 * blocksize^2
 * * m, n, o: Extents of buf_cuboid excluding ghost cells
 * * mgh, ngh, ogh: Extents of buf_cuboid including ghost cells
 * * ghosts_m, ghosts_n, ghosts_o: Ghost cell amount of buf_cuboid
 * * blocksize: Size of block in one dimension
 */
__kernel void paste_ghosts_from_border_planes_blockstencil(
    __global double* buf_stencil,
    __global double* buf_ghosts,
    const int m, const int n, const int o,
    const int mgh, const int ngh, const int ogh,
    const int ghosts_m, const int ghosts_n, const int ghosts_o,
    const int blocksize)
{
    // plane sizes
    int yz = ngh * ogh;
    int xz = mgh * ogh;
    int xy = mgh * ngh;

    // size of the ghosted grid
    int gridsize = mgh * ngh * ogh;

    // size of 27pt stencil including ghosted grid
    int gridsizeStencil = gridsize * 27;
    int blocksize2 = blocksize * blocksize;

    int yzgh = yz * ghosts_m;
    int xzgh = xz * ghosts_n;
    int xygh = xy * ghosts_o;

    // size of one matrix entry (coeffs and gps) for each plane
    int yzRessizeOneMatrixEntry = yzgh * 27;
    int xzRessizeOneMatrixEntry = xzgh * 27;
    int xyRessizeOneMatrixEntry = xygh * 27;

    // wi-index = index in input buffer buf_ghosts
    int idx = get_global_id(0);

    // Front planes (back ghosts)
    if (idx < ghosts_m * yz * 27)
    {
        int idx_coeff = idx / (ghosts_m * yz);            // 1d index of the current coefficient
        int idx_grid = idx - idx_coeff * (ghosts_m * yz); // local index of the grid point inside the grid of one coefficient

        int i = idx_grid / yz + m + ghosts_m;
        int j = (idx_grid - (i - (m + ghosts_m)) * yz) / ogh;
        int k = idx_grid % ogh;

        // No corners or edges, only ghosts directly adjacent to real back face
        if (j >= ghosts_n && j < n + ghosts_n && k >= ghosts_o && k < o + ghosts_o)
            for (int b = 0; b < blocksize2; b++)
            {
                buf_stencil[b * gridsizeStencil + idx_coeff * gridsize + i * ngh * ogh + j * ogh + k] = buf_ghosts[b * yzRessizeOneMatrixEntry + idx];
            }
    }
    // Back planes (front ghosts)
    else if (idx < 2 * ghosts_m * yz * 27)
    {
        idx -= ghosts_m * yz * 27;                        // reset to 0 for index calculation
        int idx_coeff = idx / (ghosts_m * yz);            // 1d index of the current coefficient
        int idx_grid = idx - idx_coeff * (ghosts_m * yz); // local index of the grid point inside the grid of one coefficient

        int i = idx_grid / yz;
        int j = (idx_grid - i * yz) / ogh;
        int k = idx_grid % ogh;

        // No corners or edges, only ghosts directly adjacent to real front face
        if (j >= ghosts_n && j < n + ghosts_n && k >= ghosts_o && k < o + ghosts_o)
            for (int b = 0; b < blocksize2; b++)
            {
                buf_stencil[b * gridsizeStencil + idx_coeff * gridsize + i * ngh * ogh + j * ogh + k] = buf_ghosts[b * yzRessizeOneMatrixEntry + idx + ghosts_m * yz * 27 * blocksize2];
            }
    }
    // Top planes (bottom ghosts)
    else if (idx < (2 * ghosts_m * yz + ghosts_n * xz) * 27)
    {
        idx -= 2 * ghosts_m * yz * 27;                    // reset to 0 for index calculation
        int idx_coeff = idx / (ghosts_n * xz);            // 1d index of the current coefficient
        int idx_grid = idx - idx_coeff * (ghosts_n * xz); // local index of the grid point inside the grid of one coefficient

        int j = idx_grid / xz + n + ghosts_n;
        int i = (idx_grid - (j - (n + ghosts_n)) * xz) / ogh;
        int k = idx_grid % ogh;

        // Ignore left and right ghost cells, but include front and back ghosts
        if (k >= ghosts_o && k < o + ghosts_o)
            for (int b = 0; b < blocksize2; b++)
            {
                buf_stencil[b * gridsizeStencil + idx_coeff * gridsize + i * ngh * ogh + j * ogh + k] = buf_ghosts[b * xzRessizeOneMatrixEntry + idx + 2 * ghosts_m * yz * 27 * blocksize2];
            }
    }
    // Bottom planes (top ghosts)
    else if (idx < (2 * ghosts_m * yz + 2 * ghosts_n * xz) * 27)
    {
        idx -= (2 * ghosts_m * yz + ghosts_n * xz) * 27;  // reset to 0 for index calculation
        int idx_coeff = idx / (ghosts_n * xz);            // 1d index of the current coefficient
        int idx_grid = idx - idx_coeff * (ghosts_n * xz); // local index of the grid point inside the grid of one coefficient

        int j = idx_grid / xz;
        int i = (idx_grid - j * xz) / ogh;
        int k = idx_grid % ogh;

        // Ignore left and right ghost cells, but include front and back ghosts
        if (k >= ghosts_o && k < o + ghosts_o)
            for (int b = 0; b < blocksize2; b++)
            {
                buf_stencil[b * gridsizeStencil + idx_coeff * gridsize + i * ngh * ogh + j * ogh + k] = buf_ghosts[b * xzRessizeOneMatrixEntry + idx + (2 * ghosts_m * yz + ghosts_n * xz) * 27 * blocksize2];
            }
    }
    // Left planes (right ghosts)
    else if (idx < (2 * ghosts_m * yz + 2 * ghosts_n * xz + ghosts_o * xy) * 27)
    {
        idx -= (2 * ghosts_m * yz + 2 * ghosts_n * xz) * 27; // reset to 0 for index calculation
        int idx_coeff = idx / (ghosts_o * xy);               // 1d index of the current coefficient
        int idx_grid = idx - idx_coeff * (ghosts_o * xy);    // local index of the grid point inside the grid of one coefficient

        int k = idx_grid / xy + o + ghosts_o;
        int i = (idx_grid - (k - (o + ghosts_o)) * xy) / ngh;
        int j = idx_grid % ngh;
        for (int b = 0; b < blocksize2; b++)
        {
            buf_stencil[b * gridsizeStencil + idx_coeff * gridsize + i * ngh * ogh + j * ogh + k] = buf_ghosts[b * xyRessizeOneMatrixEntry + idx + (2 * ghosts_m * yz + 2 * ghosts_n * xz) * 27 * blocksize2];
        }
    }
    // Right planes (left ghosts)
    else if (idx < (2 * ghosts_m * yz + 2 * ghosts_n * xz + 2 * ghosts_o * xy) * 27)
    {
        idx -= (2 * ghosts_m * yz + 2 * ghosts_n * xz + ghosts_o * xy) * 27; // reset to 0 for index calculation
        int idx_coeff = idx / (ghosts_o * xy);                               // 1d index of the current coefficient
        int idx_grid = idx - idx_coeff * (ghosts_o * xy);                    // local index of the grid point inside the grid of one coefficient

        int k = idx_grid / xy;
        int i = (idx_grid - k * xy) / ngh;
        int j = idx_grid % ngh;
        for (int b = 0; b < blocksize2; b++)
        {
            buf_stencil[b * gridsizeStencil + idx_coeff * gridsize + i * ngh * ogh + j * ogh + k] = buf_ghosts[b * xyRessizeOneMatrixEntry + idx + (2 * ghosts_m * yz + 2 * ghosts_n * xz + ghosts_o * xy) * 27 * blocksize2];
        }
    }
}

/**
 * Fill buffer with value, equivalent to clEnqueueFillBuffer.
 * size is the number of elements in the buffer.
 */
__kernel void fill_buffer(
    __global double* buf,
    double value,
    int size)
{
    int idx = get_global_id(0);
    if (idx < size)
        buf[idx] = value;
}
)DELIM";
}