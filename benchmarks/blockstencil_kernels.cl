#ifndef NULL
#define NULL 0
#endif

/*****************************************/
/************ block stencil start ********/
/*****************************************/

/* Calculates residual without dinv.
 * Layout: [cx][cy][cz][mx][my][gpx][gpy][gpz] for coeffs, [m][gpx][gpy][gpz] for v, f, r
 *
 * svGridSize = sv_mgh * sv_ngh * sv_ogh
 * svGridSizeBlock = blocksize^2 * svGridSize
 */
__kernel void residual_27point_blockstencil_coeffs_first_v_block_first(
    __global double* restrict v_in, // needed s.t. every work-item can read surrounding cell values
    __global double* restrict f,
    __global double* restrict r,
    __global double* restrict stencilValues,
    const int mgh, const int ngh, const int ogh,
    const int svmgh, const int svngh, const int svogh,
    const int ghosts, const int ghosts_sv,
    const int svGridSize, const int svGridSizeBlock,
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
        int ioff = ngh * ogh;
        int joff = ogh;
        int koff = 1;
        int index = i * ioff + j * ogh + k;
        int gridsize = mgh * ngh * ogh;

        int svno = svngh * svogh;
        // offset inside one coefficient grid that points to the coefficient for the current grid point. Must consider different amount of ghosts for v and sv.
        int index_sv_gp = (i - ghosts + ghosts_sv) * svno + (j - ghosts + ghosts_sv) * svogh + (k - ghosts + ghosts_sv);

        // if (i == 2 && j == 2 && k == 2)
        // {
        //     printf("i,j,k,mgh,ngh,ogh,gh,gh_sv,index_sv_gp,gridsize: %d,%d,%d,%d,%d,%d,%d,%d,%d,%d\ngh", i, j, k, mgh, ngh, ogh, ghosts, ghosts_sv, index_sv_gp, gridsize);
        // }
        double stencilsums[3]; // assumption: blocksize <= 3

        // Layout: [cx][cy][cz][mx][my][gpx][gpy][gpz] for coeffs, [m][gpx][gpy][gpz] for v, f, r
        int idx_block = 0;
        for (int bi = 0; bi < blocksize; bi++)
        {
            for (int bj = 0; bj < blocksize; bj++)
            {
                // A*v
                // clang-format off
                stencilsums[bi] += stencilValues[index_sv_gp + (9 + 3 + 1) * svGridSizeBlock + idx_block] * v_in[index + bj * gridsize]
                    + stencilValues[index_sv_gp + (9 + 3) * svGridSizeBlock + idx_block]      * v_in[index - koff + bj * gridsize]
                    + stencilValues[index_sv_gp + (9 + 3 + 2) * svGridSizeBlock + idx_block]  * v_in[index + koff + bj * gridsize]
                    + stencilValues[index_sv_gp + (9 + 1) * svGridSizeBlock + idx_block]      * v_in[index - joff + bj * gridsize]
                    + stencilValues[index_sv_gp + (9 + 6 + 1) * svGridSizeBlock + idx_block]  * v_in[index + joff + bj * gridsize]
                    + stencilValues[index_sv_gp + (3 + 1) * svGridSizeBlock + idx_block]      * v_in[index - ioff + bj * gridsize]
                    + stencilValues[index_sv_gp + (18 + 3 + 1) * svGridSizeBlock + idx_block] * v_in[index + ioff + bj * gridsize]
                    
                    + stencilValues[index_sv_gp + (9) * svGridSizeBlock + idx_block]          * v_in[index - joff - koff + bj * gridsize]
                    + stencilValues[index_sv_gp + (9 + 2) * svGridSizeBlock + idx_block]      * v_in[index - joff + koff + bj * gridsize]
                    + stencilValues[index_sv_gp + (9 + 6) * svGridSizeBlock + idx_block]      * v_in[index + joff - koff + bj * gridsize]
                    + stencilValues[index_sv_gp + (9 + 6 + 2) * svGridSizeBlock + idx_block]  * v_in[index + joff + koff + bj * gridsize]
                    + stencilValues[index_sv_gp + svGridSizeBlock * 3 + idx_block]            * v_in[index - ioff - koff + bj * gridsize]
                    + stencilValues[index_sv_gp + (3 + 2) * svGridSizeBlock + idx_block]      * v_in[index - ioff + koff + bj * gridsize]
                    + stencilValues[index_sv_gp + (18 + 3) * svGridSizeBlock + idx_block]     * v_in[index + ioff - koff + bj * gridsize]
                    + stencilValues[index_sv_gp + (18 + 3 + 2) * svGridSizeBlock + idx_block] * v_in[index + ioff + koff + bj * gridsize]
                    + stencilValues[index_sv_gp + svGridSizeBlock + idx_block]                * v_in[index - ioff - joff + bj * gridsize]
                    + stencilValues[index_sv_gp + (6 + 1) * svGridSizeBlock + idx_block]      * v_in[index - ioff + joff + bj * gridsize]
                    + stencilValues[index_sv_gp + (18 + 1) * svGridSizeBlock + idx_block]     * v_in[index + ioff - joff + bj * gridsize]
                    + stencilValues[index_sv_gp + (18 + 6 + 1) * svGridSizeBlock + idx_block] * v_in[index + ioff + joff + bj * gridsize]

                    + stencilValues[index_sv_gp + idx_block]                                  * v_in[index - ioff - joff - koff + bj * gridsize]
                    + stencilValues[index_sv_gp + svGridSizeBlock * 2 + idx_block]            * v_in[index - ioff - joff + koff + bj * gridsize]
                    + stencilValues[index_sv_gp + (6) * svGridSizeBlock + idx_block]          * v_in[index - ioff + joff - koff + bj * gridsize]
                    + stencilValues[index_sv_gp + (6 + 2) * svGridSizeBlock + idx_block]      * v_in[index - ioff + joff + koff + bj * gridsize]
                    + stencilValues[index_sv_gp + (18) * svGridSizeBlock + idx_block]         * v_in[index + ioff - joff - koff + bj * gridsize]
                    + stencilValues[index_sv_gp + (18 + 2) * svGridSizeBlock + idx_block]     * v_in[index + ioff - joff + koff + bj * gridsize]
                    + stencilValues[index_sv_gp + (18 + 6) * svGridSizeBlock + idx_block]     * v_in[index + ioff + joff - koff + bj * gridsize]
                    + stencilValues[index_sv_gp + (18 + 6 + 2) * svGridSizeBlock + idx_block] * v_in[index + ioff + joff + koff + bj * gridsize];
                // clang-format on

                idx_block += svGridSize; // increase by gridsize to get to next matrix entry
            }
        }

        for (int bi = 0; bi < blocksize; bi++)
        {
            // r = f - A*v
            r[index + bi * gridsize] = f[index + bi * gridsize] - stencilsums[bi];
        }

        // if (i == 2 && j == 2 && k == 2)
        // {
        //     printf("ocl stencilsum = %e\n", stencilsum);
        //     print27point_sv(v_in, index, ioff, joff, koff, stencilValues, index_sv_gp);
        // }
    }
}

/* Calculates residual without dinv.
 * Layout: [cx][cy][cz][mx][my][gpx][gpy][gpz] for coeffs, [gpx][gpy][gpz][m] for v, f, r
 *
 * svGridSize = sv_mgh * sv_ngh * sv_ogh
 * svGridSizeBlock = blocksize^2 * svGridSize
 */
__kernel void residual_27point_blockstencil_coeffs_first_v_gp_first(
    __global double* restrict v_in, // needed s.t. every work-item can read surrounding cell values
    __global double* restrict f,
    __global double* restrict r,
    __global double* restrict stencilValues,
    const int mgh, const int ngh, const int ogh,
    const int svmgh, const int svngh, const int svogh,
    const int ghosts, const int ghosts_sv,
    const int svGridSize, const int svGridSizeBlock,
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
        int index = i * ngh * ogh + j * ogh + k;
        int gridsize = mgh * ngh * ogh;

        int svno = svngh * svogh;
        // offset inside one coefficient grid that points to the coefficient for the current grid point. Must consider different amount of ghosts for v and sv.
        int index_sv_gp = (i - ghosts + ghosts_sv) * svno + (j - ghosts + ghosts_sv) * svogh + (k - ghosts + ghosts_sv);

        // if (i == 2 && j == 2 && k == 2)
        // {
        //     printf("i,j,k,mgh,ngh,ogh,gh,gh_sv,index_sv_gp,gridsize: %d,%d,%d,%d,%d,%d,%d,%d,%d,%d\ngh", i, j, k, mgh, ngh, ogh, ghosts, ghosts_sv, index_sv_gp, gridsize);
        // }
        double stencilsums[3]; // assumption: blocksize <= 3

        // Layout: [cx][cy][cz][mx][my][gpx][gpy][gpz] for coeffs, [gpx][gpy][gpz][m] for v, f, r
        int idx_block = 0;
        for (int bi = 0; bi < blocksize; bi++)
        {
            for (int bj = 0; bj < blocksize; bj++)
            {
                // A*v
                // clang-format off
                stencilsums[bi] += stencilValues[index_sv_gp + (9 + 3 + 1) * svGridSizeBlock + idx_block] * v_in[index + bj]
                    + stencilValues[index_sv_gp + (9 + 3) * svGridSizeBlock + idx_block]      * v_in[index - koff + bj]
                    + stencilValues[index_sv_gp + (9 + 3 + 2) * svGridSizeBlock + idx_block]  * v_in[index + koff + bj]
                    + stencilValues[index_sv_gp + (9 + 1) * svGridSizeBlock + idx_block]      * v_in[index - joff + bj]
                    + stencilValues[index_sv_gp + (9 + 6 + 1) * svGridSizeBlock + idx_block]  * v_in[index + joff + bj]
                    + stencilValues[index_sv_gp + (3 + 1) * svGridSizeBlock + idx_block]      * v_in[index - ioff + bj]
                    + stencilValues[index_sv_gp + (18 + 3 + 1) * svGridSizeBlock + idx_block] * v_in[index + ioff + bj]
                    
                    + stencilValues[index_sv_gp + (9) * svGridSizeBlock + idx_block]          * v_in[index - joff - koff + bj]
                    + stencilValues[index_sv_gp + (9 + 2) * svGridSizeBlock + idx_block]      * v_in[index - joff + koff + bj]
                    + stencilValues[index_sv_gp + (9 + 6) * svGridSizeBlock + idx_block]      * v_in[index + joff - koff + bj]
                    + stencilValues[index_sv_gp + (9 + 6 + 2) * svGridSizeBlock + idx_block]  * v_in[index + joff + koff + bj]
                    + stencilValues[index_sv_gp + svGridSizeBlock * 3 + idx_block]            * v_in[index - ioff - koff + bj]
                    + stencilValues[index_sv_gp + (3 + 2) * svGridSizeBlock + idx_block]      * v_in[index - ioff + koff + bj]
                    + stencilValues[index_sv_gp + (18 + 3) * svGridSizeBlock + idx_block]     * v_in[index + ioff - koff + bj]
                    + stencilValues[index_sv_gp + (18 + 3 + 2) * svGridSizeBlock + idx_block] * v_in[index + ioff + koff + bj]
                    + stencilValues[index_sv_gp + svGridSizeBlock + idx_block]                * v_in[index - ioff - joff + bj]
                    + stencilValues[index_sv_gp + (6 + 1) * svGridSizeBlock + idx_block]      * v_in[index - ioff + joff + bj]
                    + stencilValues[index_sv_gp + (18 + 1) * svGridSizeBlock + idx_block]     * v_in[index + ioff - joff + bj]
                    + stencilValues[index_sv_gp + (18 + 6 + 1) * svGridSizeBlock + idx_block] * v_in[index + ioff + joff + bj]

                    + stencilValues[index_sv_gp + idx_block]                                  * v_in[index - ioff - joff - koff + bj]
                    + stencilValues[index_sv_gp + svGridSizeBlock * 2 + idx_block]            * v_in[index - ioff - joff + koff + bj]
                    + stencilValues[index_sv_gp + (6) * svGridSizeBlock + idx_block]          * v_in[index - ioff + joff - koff + bj]
                    + stencilValues[index_sv_gp + (6 + 2) * svGridSizeBlock + idx_block]      * v_in[index - ioff + joff + koff + bj]
                    + stencilValues[index_sv_gp + (18) * svGridSizeBlock + idx_block]         * v_in[index + ioff - joff - koff + bj]
                    + stencilValues[index_sv_gp + (18 + 2) * svGridSizeBlock + idx_block]     * v_in[index + ioff - joff + koff + bj]
                    + stencilValues[index_sv_gp + (18 + 6) * svGridSizeBlock + idx_block]     * v_in[index + ioff + joff - koff + bj]
                    + stencilValues[index_sv_gp + (18 + 6 + 2) * svGridSizeBlock + idx_block] * v_in[index + ioff + joff + koff + bj];
                // clang-format on

                idx_block += svGridSize; // increase by gridsize to get to next matrix entry
            }
        }

        for (int bi = 0; bi < blocksize; bi++)
        {
            // r = f - A*v
            r[index + bi] = f[index + bi] - stencilsums[bi];
        }

        // if (i == 2 && j == 2 && k == 2)
        // {
        //     printf("ocl stencilsum = %e\n", stencilsum);
        //     print27point_sv(v_in, index, ioff, joff, koff, stencilValues, index_sv_gp);
        // }
    }
}

/* Calculates residual without dinv.
 * Layout: [mx][my][cx][cy][cz][gpx][gpy][gpz] for coeffs, [m][gpx][gpy][gpz] for v, f, r
 *
 * svGridSize = sv_mgh * sv_ngh * sv_ogh
 * svGridSizeCoeffs = 27 * svGridSize
 */
__kernel void residual_27point_blockstencil_block_first_v_block_first(
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
        int ioff = ngh * ogh;
        int joff = ogh;
        int koff = 1;
        int index = i * ioff + j * ogh + k;
        int gridsize = mgh * ngh * ogh;

        int svno = svngh * svogh;
        // offset inside one coefficient grid that points to the coefficient for the current grid point. Must consider different amount of ghosts for v and sv.
        int index_sv_gp = (i - ghosts + ghosts_sv) * svno + (j - ghosts + ghosts_sv) * svogh + (k - ghosts + ghosts_sv);

        // if (i == 2 && j == 2 && k == 2)
        // {
        //     printf("i,j,k,mgh,ngh,ogh,gh,gh_sv,index_sv_gp,gridsize: %d,%d,%d,%d,%d,%d,%d,%d,%d,%d\ngh", i, j, k, mgh, ngh, ogh, ghosts, ghosts_sv, index_sv_gp, gridsize);
        // }
        double stencilsums[3]; // assumption: blocksize <= 3

        // Layout: [cx][cy][cz][mx][my][gpx][gpy][gpz] for coeffs, [m][gpx][gpy][gpz] for v, f, r
        int idx_block = 0;
        for (int bi = 0; bi < blocksize; bi++)
        {
            for (int bj = 0; bj < blocksize; bj++)
            {
                // A*v
                // clang-format off
                stencilsums[bi] += stencilValues[index_sv_gp + (9 + 3 + 1) * svGridSize + idx_block] * v_in[index + bj * gridsize]
                    + stencilValues[index_sv_gp + (9 + 3) * svGridSize + idx_block]      * v_in[index - koff + bj * gridsize]
                    + stencilValues[index_sv_gp + (9 + 3 + 2) * svGridSize + idx_block]  * v_in[index + koff + bj * gridsize]
                    + stencilValues[index_sv_gp + (9 + 1) * svGridSize + idx_block]      * v_in[index - joff + bj * gridsize]
                    + stencilValues[index_sv_gp + (9 + 6 + 1) * svGridSize + idx_block]  * v_in[index + joff + bj * gridsize]
                    + stencilValues[index_sv_gp + (3 + 1) * svGridSize + idx_block]      * v_in[index - ioff + bj * gridsize]
                    + stencilValues[index_sv_gp + (18 + 3 + 1) * svGridSize + idx_block] * v_in[index + ioff + bj * gridsize]
                    
                    + stencilValues[index_sv_gp + (9) * svGridSize + idx_block]          * v_in[index - joff - koff + bj * gridsize]
                    + stencilValues[index_sv_gp + (9 + 2) * svGridSize + idx_block]      * v_in[index - joff + koff + bj * gridsize]
                    + stencilValues[index_sv_gp + (9 + 6) * svGridSize + idx_block]      * v_in[index + joff - koff + bj * gridsize]
                    + stencilValues[index_sv_gp + (9 + 6 + 2) * svGridSize + idx_block]  * v_in[index + joff + koff + bj * gridsize]
                    + stencilValues[index_sv_gp + svGridSize * 3 + idx_block]            * v_in[index - ioff - koff + bj * gridsize]
                    + stencilValues[index_sv_gp + (3 + 2) * svGridSize + idx_block]      * v_in[index - ioff + koff + bj * gridsize]
                    + stencilValues[index_sv_gp + (18 + 3) * svGridSize + idx_block]     * v_in[index + ioff - koff + bj * gridsize]
                    + stencilValues[index_sv_gp + (18 + 3 + 2) * svGridSize + idx_block] * v_in[index + ioff + koff + bj * gridsize]
                    + stencilValues[index_sv_gp + svGridSize + idx_block]                * v_in[index - ioff - joff + bj * gridsize]
                    + stencilValues[index_sv_gp + (6 + 1) * svGridSize + idx_block]      * v_in[index - ioff + joff + bj * gridsize]
                    + stencilValues[index_sv_gp + (18 + 1) * svGridSize + idx_block]     * v_in[index + ioff - joff + bj * gridsize]
                    + stencilValues[index_sv_gp + (18 + 6 + 1) * svGridSize + idx_block] * v_in[index + ioff + joff + bj * gridsize]

                    + stencilValues[index_sv_gp + idx_block]                                  * v_in[index - ioff - joff - koff + bj * gridsize]
                    + stencilValues[index_sv_gp + svGridSize * 2 + idx_block]            * v_in[index - ioff - joff + koff + bj * gridsize]
                    + stencilValues[index_sv_gp + (6) * svGridSize + idx_block]          * v_in[index - ioff + joff - koff + bj * gridsize]
                    + stencilValues[index_sv_gp + (6 + 2) * svGridSize + idx_block]      * v_in[index - ioff + joff + koff + bj * gridsize]
                    + stencilValues[index_sv_gp + (18) * svGridSize + idx_block]         * v_in[index + ioff - joff - koff + bj * gridsize]
                    + stencilValues[index_sv_gp + (18 + 2) * svGridSize + idx_block]     * v_in[index + ioff - joff + koff + bj * gridsize]
                    + stencilValues[index_sv_gp + (18 + 6) * svGridSize + idx_block]     * v_in[index + ioff + joff - koff + bj * gridsize]
                    + stencilValues[index_sv_gp + (18 + 6 + 2) * svGridSize + idx_block] * v_in[index + ioff + joff + koff + bj * gridsize];
                // clang-format on

                idx_block += svGridSizeCoeffs; // increase by gridsize * 27 to get to next matrix entry for same grid point
            }
        }

        for (int bi = 0; bi < blocksize; bi++)
        {
            // r = f - A*v
            r[index + bi * gridsize] = f[index + bi * gridsize] - stencilsums[bi];
        }

        // if (i == 2 && j == 2 && k == 2)
        // {
        //     printf("ocl stencilsum = %e\n", stencilsum);
        //     print27point_sv(v_in, index, ioff, joff, koff, stencilValues, index_sv_gp);
        // }
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