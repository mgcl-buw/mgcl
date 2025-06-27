#ifndef NULL
#define NULL 0
#endif

/**
 * As in currently productive code.
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