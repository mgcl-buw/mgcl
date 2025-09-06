#ifndef NULL
#define NULL 0
#endif

__kernel void empty_kernel_args00()
{
    int idx = get_global_id(0);
}

__kernel void empty_kernel_args01(
    __global float* a1)
{
    int idx = get_global_id(0);
}

__kernel void empty_kernel_args02(
    __global float* a1,
    __global float* a2)
{
    int idx = get_global_id(0);
}

__kernel void empty_kernel_args03(
    __global float* a1,
    __global float* a2,
    __global float* a3)
{
    int idx = get_global_id(0);
}

__kernel void empty_kernel_args04(
    __global float* a1,
    __global float* a2,
    __global float* a3,
    __global float* a4)
{
    int idx = get_global_id(0);
}

__kernel void empty_kernel_args05(
    __global float* a1,
    __global float* a2,
    __global float* a3,
    __global float* a4,
    __global float* a5)
{
    int idx = get_global_id(0);
}

__kernel void empty_kernel_args06(
    __global float* a1,
    __global float* a2,
    __global float* a3,
    __global float* a4,
    __global float* a5,
    __global float* a6)
{
    int idx = get_global_id(0);
}

__kernel void empty_kernel_args07(
    __global float* a1,
    __global float* a2,
    __global float* a3,
    __global float* a4,
    __global float* a5,
    __global float* a6,
    __global float* a7)
{
    int idx = get_global_id(0);
}

__kernel void empty_kernel_args08(
    __global float* a1,
    __global float* a2,
    __global float* a3,
    __global float* a4,
    __global float* a5,
    __global float* a6,
    __global float* a7,
    __global float* a8)
{
    int idx = get_global_id(0);
}

/**
 * Fill buffer with value, equivalent to clEnqueueFillBuffer.
 * Arguments:
 *   buf: Buffer to fill.
 *   value: Value to fill the buffer with.
 *   size is the number of elements in the buffer.
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

/**
 * Fill buffer with its 1d index as value.
 * Arguments:
 *   buf: Buffer to fill.
 *   size: Number of elements in the buffer.
 *   mgh, ngh, ogh: Extents of the buffer including ghost cells.
 *   ghosts_m, ghosts_n, ghosts_o: Ghost cell amount of the buffer.
 *   realCellsOnly: If true, only fill the real cells, i.e. skip ghost cells.
 */
__kernel void fill_1d_index(
    __global double* buf,
    int size,
    int mgh, int ngh, int ogh,
    int ghosts_m, int ghosts_n, int ghosts_o,
    int realCellsOnly)
{
    int idx = get_global_id(0);
    int no = ngh * ogh;
    int i = idx / no;
    int j = (idx - i * no) / ogh;
    int k = idx % ogh;

    if (idx < size)
    {
        if (!realCellsOnly || (realCellsOnly && i >= ghosts_m && i < mgh - ghosts_m && j >= ghosts_n && j < ngh - ghosts_n && k >= ghosts_o && k < ogh - ghosts_o))
            buf[idx] = idx;
    }
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