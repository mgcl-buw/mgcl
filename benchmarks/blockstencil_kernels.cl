#ifndef NULL
#define NULL 0
#endif

/**********************************************************/
/************ block stencil residual kernels start ********/
/**********************************************************/

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

        // Layout: [cx][cy][cz][mx][my][gpx][gpy][gpz] for coeffs, [m][gpx][gpy][gpz] for v, f, r
        int idx_block = 0;
        for (int bi = 0; bi < blocksize; bi++)
        {
            double stencilsum = 0;
            for (int bj = 0; bj < blocksize; bj++)
            {
                // A*v
                // clang-format off
                stencilsum += stencilValues[index_sv_gp + (9 + 3 + 1) * svGridSizeBlock + idx_block] * v_in[index + bj * gridsize]
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

            // r = f - A*v
            r[index + bi * gridsize] = f[index + bi * gridsize] - stencilsum;
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

        // Layout: [cx][cy][cz][mx][my][gpx][gpy][gpz] for coeffs, [gpx][gpy][gpz][m] for v, f, r
        int idx_block = 0;
        for (int bi = 0; bi < blocksize; bi++)
        {
            double stencilsum = 0;
            for (int bj = 0; bj < blocksize; bj++)
            {
                // A*v
                // clang-format off
                stencilsum += stencilValues[index_sv_gp + (9 + 3 + 1) * svGridSizeBlock + idx_block] * v_in[index + bj]
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

            // r = f - A*v
            r[index + bi] = f[index + bi] - stencilsum;
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

        // Layout: [cx][cy][cz][mx][my][gpx][gpy][gpz] for coeffs, [m][gpx][gpy][gpz] for v, f, r
        int idx_block = 0;
        for (int bi = 0; bi < blocksize; bi++)
        {
            double stencilsum = 0;
            for (int bj = 0; bj < blocksize; bj++)
            {
                // A*v
                // clang-format off
                stencilsum += stencilValues[index_sv_gp + (9 + 3 + 1) * svGridSize + idx_block] * v_in[index + bj * gridsize]
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

            // r = f - A*v
            r[index + bi * gridsize] = f[index + bi * gridsize] - stencilsum;
        }

        // if (i == 2 && j == 2 && k == 2)
        // {
        //     printf("ocl stencilsum = %e\n", stencilsum);
        //     print27point_sv(v_in, index, ioff, joff, koff, stencilValues, index_sv_gp);
        // }
    }
}

/* Calculates residual without dinv.
 * Layout: [mx][my][cx][cy][cz][gpx][gpy][gpz] for coeffs, [gpx][gpy][gpz][m] for v, f, r
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
        int index = i * ngh * ogh + j * ogh + k;
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
            r[index + bi] = f[index + bi] - stencilsum;
        }

        // if (i == 2 && j == 2 && k == 2)
        // {
        //     printf("ocl stencilsum = %e\n", stencilsum);
        //     print27point_sv(v_in, index, ioff, joff, koff, stencilValues, index_sv_gp);
        // }
    }
}

/**********************************************************/
/************ block stencil residual kernels end **********/
/**********************************************************/

/**********************************************************/
/************ block stencil border planes kernels start **********/
/**********************************************************/

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
 * Block entries for one grid points lay conescutively in memory, i.e. gp000_m0, gp000_m1, gp001_m0, gp001_m1, etc.
 * Arguments:
 * * buf_cuboid: CuboidGPU::buffer of size mgh*ngh*ogh
 * * buf_res: CuboidGPU::buffer of size 1*1*#borderCells
 * * m, n, o: Extents of buf_cuboid excluding ghost cells
 * * mgh, ngh, ogh: Extents of buf_cuboid including ghost cells
 * * ghosts_m, ghosts_n, ghosts_o: Ghost cell amount of buf_cuboid
 * * blocksize: Number of vector components
 */
__kernel void extract_border_planes_gp_first(
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
 * Grid points for one block entry lay conescutively in memory, i.e. gp000_m0, gp001_m0, gp002_m0, ..., gp000_m1, gp001_m1 etc.
 * Arguments:
 * * buf_cuboid: CuboidGPU::buffer of size mgh*ngh*ogh
 * * buf_res: CuboidGPU::buffer of size 1*1*#borderCells
 * * m, n, o: Extents of buf_cuboid excluding ghost cells
 * * mgh, ngh, ogh: Extents of buf_cuboid including ghost cells
 * * ghosts_m, ghosts_n, ghosts_o: Ghost cell amount of buf_cuboid
 * * blocksize: Number of vector components
 */
__kernel void extract_border_planes_block_first(
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
    int gridsize = mgh * ngh * ogh;

    // Get buf_cuboid's 1d index for the first matrix entry
    int idx = get_global_id(0);

    // Front planes
    if (idx < ghosts_m * yz)
    {
        int i = idx / yz + ghosts_m;
        int j = (idx - (i - ghosts_m) * yz) / ogh;
        int k = idx % ogh;
        int gp_base_idx = i * ngh * ogh + j * ogh + k; // gp base index in source cuboid
        for (int bi = 0; bi < blocksize; bi++)
        {
            buf_res[idx + bi * gridsize] = buf_cuboid[gp_base_idx + bi * gridsize]; // store block entries for each grid point consecutively
        }
    }
    // Back planes
    else if (idx < 2 * ghosts_m * yz)
    {
        idx -= ghosts_m * yz; // reset to 0 for index calculation
        int i = idx / yz + m;
        int j = (idx - (i - m) * yz) / ogh;
        int k = idx % ogh;
        int gp_base_idx = i * ngh * ogh + j * ogh + k; // gp base index in source cuboid
        for (int bi = 0; bi < blocksize; bi++)
        {
            buf_res[idx + (ghosts_m * yz) * blocksize + bi * gridsize] = buf_cuboid[gp_base_idx + bi * gridsize];
        }
    }
    // Top planes
    else if (idx < 2 * ghosts_m * yz + ghosts_n * xz)
    {
        idx -= 2 * ghosts_m * yz; // reset to 0 for index calculation
        int j = idx / xz + ghosts_n;
        int i = (idx - (j - ghosts_n) * xz) / ogh;
        int k = idx % ogh;
        int gp_base_idx = i * ngh * ogh + j * ogh + k; // gp base index in source cuboid
        for (int bi = 0; bi < blocksize; bi++)
        {
            buf_res[idx + (2 * ghosts_m * yz) * blocksize + bi * gridsize] = buf_cuboid[gp_base_idx + bi * gridsize];
        }
    }
    // Bottom planes
    else if (idx < 2 * ghosts_m * yz + 2 * ghosts_n * xz)
    {
        idx -= 2 * ghosts_m * yz + ghosts_n * xz; // reset to 0 for index calculation
        int j = idx / xz + n;
        int i = (idx - (j - n) * xz) / ogh;
        int k = idx % ogh;
        int gp_base_idx = i * ngh * ogh + j * ogh + k; // gp base index in source cuboid
        for (int bi = 0; bi < blocksize; bi++)
        {
            buf_res[idx + (2 * ghosts_m * yz + ghosts_n * xz) * blocksize + bi * gridsize] = buf_cuboid[gp_base_idx + bi * gridsize];
        }
    }
    // Left planes
    else if (idx < 2 * ghosts_m * yz + 2 * ghosts_n * xz + ghosts_o * xy)
    {
        idx -= 2 * ghosts_m * yz + 2 * ghosts_n * xz; // reset to 0 for index calculation
        int k = idx / xy + ghosts_o;
        int i = (idx - (k - ghosts_o) * xy) / ngh;
        int j = idx % ngh;
        int gp_base_idx = i * ngh * ogh + j * ogh + k; // gp base index in source cuboid
        for (int bi = 0; bi < blocksize; bi++)
        {
            buf_res[idx + (2 * ghosts_m * yz + 2 * ghosts_n * xz) * blocksize + bi * gridsize] = buf_cuboid[gp_base_idx + bi * gridsize];
        }
    }
    // Right planes
    else if (idx < 2 * ghosts_m * yz + 2 * ghosts_n * xz + 2 * ghosts_o * xy)
    {
        idx -= 2 * ghosts_m * yz + 2 * ghosts_n * xz + ghosts_o * xy; // reset to 0 for index calculation
        int k = idx / xy + o;
        int i = (idx - (k - o) * xy) / ngh;
        int j = idx % ngh;
        int gp_base_idx = i * ngh * ogh + j * ogh + k; // gp base index in source cuboid
        for (int bi = 0; bi < blocksize; bi++)
        {
            buf_res[idx + (2 * ghosts_m * yz + 2 * ghosts_n * xz + ghosts_o * xy) * blocksize + bi * gridsize] = buf_cuboid[gp_base_idx + bi * gridsize];
        }
    }
}

/**********************************************************/
/************ block stencil border planes kernels end **********/
/**********************************************************/

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