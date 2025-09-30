#ifndef NULL
#define NULL 0
#endif

/* Calculates residual without dinv
 * Uses layout [3x3x3][m,n,o] for stencilValues.
 */
__kernel void residual_27point_varying_stencil_coeffs_first_1d(
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

/* Calculates residual without dinv.
 * Started as 3d kernel. Outer dimension (m) is global index 0. */
__kernel void residual_27point_varying_stencil_coeffs_first_3d_m0(
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
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);

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

/* Calculates residual without dinv.
 * Started as 3d kernel. Inner dimension (o) is global index 0. */
__kernel void residual_27point_varying_stencil_coeffs_first_3d_o0(
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
    int i = get_global_id(2);
    int j = get_global_id(1);
    int k = get_global_id(0);

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
 * Uses layout [m,n,o][3x3x3] for stencilValues.
 */
__kernel void residual_27point_varying_stencil_gps_first_1d(
    __global double* v_in,
    __global double* f,
    __global double* r,
    __global double* stencilValues,
    const int mgh, const int ngh, const int ogh,
    const int svmgh, const int svngh, const int svogh, // not needed but for sake of simplicity
    const int ghosts, const int ghosts_sv,
    const int svGridSize, // not needed but for sake of simplicity
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

        int koff_sv = 27;
        int joff_sv = ((ogh - 2 * ghosts) + 2 * ghosts_sv) * koff_sv;
        int ioff_sv = ((ngh - 2 * ghosts) + 2 * ghosts_sv) * joff_sv;
        int index_sv = (i + (ghosts_sv - ghosts)) * ioff_sv + (j + (ghosts_sv - ghosts)) * joff_sv + (k + (ghosts_sv - ghosts)) * koff_sv;

        // A*v
        // clang-format off
        double stencilsum = stencilValues[index_sv + 9 + 3 + 1] * v_in[index]
            + stencilValues[index_sv + 9 + 3]      * v_in[index - 1]
            + stencilValues[index_sv + 9 + 3 + 2]  * v_in[index + 1]
            + stencilValues[index_sv + 9 + 1]      * v_in[index - joff]
            + stencilValues[index_sv + 9 + 6 + 1]  * v_in[index + joff]
            + stencilValues[index_sv + 3 + 1]      * v_in[index - ioff]
            + stencilValues[index_sv + 18 + 3 + 1] * v_in[index + ioff]
            
            + stencilValues[index_sv + 9]          * v_in[index - joff - koff]
            + stencilValues[index_sv + 9 + 2]      * v_in[index - joff + koff]
            + stencilValues[index_sv + 9 + 6]      * v_in[index + joff - koff]
            + stencilValues[index_sv + 9 + 6 + 2]  * v_in[index + joff + koff]
            + stencilValues[index_sv + 3]          * v_in[index - ioff - koff]
            + stencilValues[index_sv + 3 + 2]      * v_in[index - ioff + koff]
            + stencilValues[index_sv + 18 + 3]     * v_in[index + ioff - koff]
            + stencilValues[index_sv + 18 + 3 + 2] * v_in[index + ioff + koff]
            + stencilValues[index_sv + 1]          * v_in[index - ioff - joff]
            + stencilValues[index_sv + 6 + 1]      * v_in[index - ioff + joff]
            + stencilValues[index_sv + 18 + 1]     * v_in[index + ioff - joff]
            + stencilValues[index_sv + 18 + 6 + 1] * v_in[index + ioff + joff]

            + stencilValues[index_sv]              * v_in[index - ioff - joff - koff]
            + stencilValues[index_sv + 2]          * v_in[index - ioff - joff + koff]
            + stencilValues[index_sv + 6]          * v_in[index - ioff + joff - koff]
            + stencilValues[index_sv + 6 + 2]      * v_in[index - ioff + joff + koff]
            + stencilValues[index_sv + 18]         * v_in[index + ioff - joff - koff]
            + stencilValues[index_sv + 18 + 2]     * v_in[index + ioff - joff + koff]
            + stencilValues[index_sv + 18 + 6]     * v_in[index + ioff + joff - koff]
            + stencilValues[index_sv + 18 + 6 + 2] * v_in[index + ioff + joff + koff];
        // clang-format on

        // r = f - A*v
        r[index] = f[index] - stencilsum;
    }
}

/**
 * Uses layout [m,n,o][3x3x3] for stencilValues.
 */
__kernel void residual_27point_varying_stencil_gps_first_3d_o0(
    __global double* v_in,
    __global double* f,
    __global double* r,
    __global double* stencilValues,
    const int mgh, const int ngh, const int ogh,
    const int svmgh, const int svngh, const int svogh, // not needed but for sake of simplicity
    const int ghosts, const int ghosts_sv,
    const int svGridSize, // not needed but for sake of simplicity
    const int moff, const int noff, const int ooff)
{
    int i = get_global_id(2);
    int j = get_global_id(1);
    int k = get_global_id(0);

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

        int koff_sv = 27;
        int joff_sv = ((ogh - 2 * ghosts) + 2 * ghosts_sv) * koff_sv;
        int ioff_sv = ((ngh - 2 * ghosts) + 2 * ghosts_sv) * joff_sv;
        int index_sv = (i + (ghosts_sv - ghosts)) * ioff_sv + (j + (ghosts_sv - ghosts)) * joff_sv + (k + (ghosts_sv - ghosts)) * koff_sv;

        // A*v
        // clang-format off
        double stencilsum = stencilValues[index_sv + 9 + 3 + 1] * v_in[index]
            + stencilValues[index_sv + 9 + 3]      * v_in[index - 1]
            + stencilValues[index_sv + 9 + 3 + 2]  * v_in[index + 1]
            + stencilValues[index_sv + 9 + 1]      * v_in[index - joff]
            + stencilValues[index_sv + 9 + 6 + 1]  * v_in[index + joff]
            + stencilValues[index_sv + 3 + 1]      * v_in[index - ioff]
            + stencilValues[index_sv + 18 + 3 + 1] * v_in[index + ioff]
            
            + stencilValues[index_sv + 9]          * v_in[index - joff - koff]
            + stencilValues[index_sv + 9 + 2]      * v_in[index - joff + koff]
            + stencilValues[index_sv + 9 + 6]      * v_in[index + joff - koff]
            + stencilValues[index_sv + 9 + 6 + 2]  * v_in[index + joff + koff]
            + stencilValues[index_sv + 3]          * v_in[index - ioff - koff]
            + stencilValues[index_sv + 3 + 2]      * v_in[index - ioff + koff]
            + stencilValues[index_sv + 18 + 3]     * v_in[index + ioff - koff]
            + stencilValues[index_sv + 18 + 3 + 2] * v_in[index + ioff + koff]
            + stencilValues[index_sv + 1]          * v_in[index - ioff - joff]
            + stencilValues[index_sv + 6 + 1]      * v_in[index - ioff + joff]
            + stencilValues[index_sv + 18 + 1]     * v_in[index + ioff - joff]
            + stencilValues[index_sv + 18 + 6 + 1] * v_in[index + ioff + joff]

            + stencilValues[index_sv]              * v_in[index - ioff - joff - koff]
            + stencilValues[index_sv + 2]          * v_in[index - ioff - joff + koff]
            + stencilValues[index_sv + 6]          * v_in[index - ioff + joff - koff]
            + stencilValues[index_sv + 6 + 2]      * v_in[index - ioff + joff + koff]
            + stencilValues[index_sv + 18]         * v_in[index + ioff - joff - koff]
            + stencilValues[index_sv + 18 + 2]     * v_in[index + ioff - joff + koff]
            + stencilValues[index_sv + 18 + 6]     * v_in[index + ioff + joff - koff]
            + stencilValues[index_sv + 18 + 6 + 2] * v_in[index + ioff + joff + koff];
        // clang-format on

        // r = f - A*v
        r[index] = f[index] - stencilsum;
    }
}

/* Calculates residual without dinv.
   Indices of coeffs are precalculated for a fixed grid size and amount of ghosts. */
__kernel void residual_27point_varying_stencil_coeffs_first_indices_precalc_64(
    __global double* restrict v_in, // needed s.t. every work-item can read surrounding cell values
    __global double* restrict f,
    __global double* restrict r,
    __global double* restrict stencilValues,
    const int mgh, const int ngh, const int ogh,
    const int svmgh, const int svngh, const int svogh,
    const int ghosts, const int ghosts_sv,
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
        int index = i * ngh * ogh + j * ogh + k;

        int svno = svngh * svogh;
        // offset inside one coefficient grid that points to the coefficient for the current grid point. Must consider different amount of ghosts for v and sv.
        int index_sv = (i - ghosts + ghosts_sv) * svno + (j - ghosts + ghosts_sv) * svogh + (k - ghosts + ghosts_sv);

        // if (i == 2 && j == 2 && k == 2)
        // {
        //     printf("i,j,k,mgh,ngh,ogh,gh,gh_sv,index_sv,gridsize: %d,%d,%d,%d,%d,%d,%d,%d,%d,%d\ngh", i, j, k, mgh, ngh, ogh, ghosts, ghosts_sv, index_sv, gridsize);
        // }

        // A*v
        // clang-format off
        double stencilsum = stencilValues[index_sv + 3737448] * v_in[index]
            + stencilValues[index_sv + 3449952]      * v_in[index - 1]
            + stencilValues[index_sv + 4024944]  * v_in[index + 1]
            + stencilValues[index_sv + 2874960]      * v_in[index - 66]
            + stencilValues[index_sv + 4599936]  * v_in[index + 66]
            + stencilValues[index_sv + 1149984]      * v_in[index - 4356]
            + stencilValues[index_sv + 6324912] * v_in[index + 4356]
            
            + stencilValues[index_sv + 2587464]          * v_in[index - 67]
            + stencilValues[index_sv + 3162456]      * v_in[index - 65]
            + stencilValues[index_sv + 4312440]      * v_in[index + 65]
            + stencilValues[index_sv + 4887432]  * v_in[index + 67]
            + stencilValues[index_sv + 862488]            * v_in[index - 4357]
            + stencilValues[index_sv + 1437480]      * v_in[index - 4355]
            + stencilValues[index_sv + 6037416]     * v_in[index + 4355]
            + stencilValues[index_sv + 6612408] * v_in[index + 4357]
            + stencilValues[index_sv + 287496]                * v_in[index - 4422]
            + stencilValues[index_sv + 2012472]      * v_in[index - 4290]
            + stencilValues[index_sv + 5462424]     * v_in[index + 4290]
            + stencilValues[index_sv + 7187400] * v_in[index + 4422]

            + stencilValues[index_sv]                           * v_in[index - 4423]
            + stencilValues[index_sv + 574992]            * v_in[index - 4421]
            + stencilValues[index_sv + 1724976]          * v_in[index + (-4291)]
            + stencilValues[index_sv + 2299968]      * v_in[index + (-4289)]
            + stencilValues[index_sv + 5174928]         * v_in[index + (4289)]
            + stencilValues[index_sv + 5749920]     * v_in[index + (4291)]
            + stencilValues[index_sv + 6899904]     * v_in[index + (4421)]
            + stencilValues[index_sv + 7474896] * v_in[index + (4423)];
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

/* Calculates residual with a 27p varying stencil.
 * 1d kernel, must be launched with mgh*ngh*ogh work-items, i.e. #work-items == amount of ghosted grid cells.
 * Work-group size can be arbitrary chosen. 32 seems optimal for my laptop gpu.
 * Arguments:
 *   v_in: current guess, same size of ghosted grid. Only gets read in this kernel.
 *      f: rhs, same size of ghosted grids. Only gets read in this kernel.
 *      r: residual, same size of ghosted grid. Only gets written in this kernel.
 * stencilValues: 27p varying stencil per grid cell. Size = mgh*ngh*ogh * 27.
 *      m: ghosted grid size in dim 1
 *      n: ghosted grid size in dim 2
 *      o: ghosted grid size in dim 3
 * ghosts: Amount of ghosts of v_in, f and r
 * ghosts_sv: Amount of ghosts of stencilValues
 *   moff: Amount of ghost cells which shall be calculated in dim 1.
 *   noff: Amount of ghost cells which shall be calculated in dim 2.
 *   ooff: Amount of ghost cells which shall be calculated in dim 3.
 *   I.e. moff = -1 means that the first layer of ghost cells will be updated, too. That would require ghosts >= 2 however.
 *
 * Restructured calculation order of stencil (memory accessing etc)
 */
// residual_27point_varying_stencil_1d_one_wi_per_cell_restructured
__kernel void residual_27point_varying_stencil_1d_gps_first_restructured(
    __global double* v_in,
    __global double* f,
    __global double* r,
    __global double* stencilValues,
    const int mgh, const int ngh, const int ogh,
    const int svmgh, const int svngh, const int svogh, // not needed but for sake of simplicity
    const int ghosts, const int ghosts_sv,
    const int svGridSize, // not needed but for sake of simplicity
    const int moff, const int noff, const int ooff)
{
    int idx = get_global_id(0);
    int no = ngh * ogh;
    int i = idx / no;
    int j = (idx - i * no) / ogh;
    int k = idx % ogh;

    // loop boundaries
    // TODO maybe refactor to use v_ghm, etc.?
    // int istart_v = ghosts + moff;
    // int jstart_v = ghosts + noff;
    // int kstart_v = ghosts + ooff;
    // int iend_v = mgh - ghosts - moff;
    // int jend_v = ngh - ghosts - noff;
    // int kend_v = ogh - ghosts - ooff;

    // calculate residual only for relevant cells (off = 0: only real cells)
    if (i >= ghosts + moff && j >= ghosts + noff && k >= ghosts + ooff && i < mgh - ghosts - moff && j < ngh - ghosts - noff && k < ogh - ghosts - ooff)
    {
        int ioff = ngh * ogh;
        // int joff = ogh;
        // int koff = 1;
        int index = i * ioff + j * ogh + k;

        // int koff_sv = 27;
        int joff_sv = ((ogh - 2 * ghosts) + 2 * ghosts_sv) * 27;
        // int ioff_sv = ((ngh - 2 * ghosts) + 2 * ghosts_sv) * joff_sv;
        int index_sv = (i + (ghosts_sv - ghosts)) * ((ngh - 2 * ghosts) + 2 * ghosts_sv) * joff_sv + (j + (ghosts_sv - ghosts)) * joff_sv + (k + (ghosts_sv - ghosts)) * 27;

        // A*v
        // clang-format off
        double stencilsum = 0.0;
        double sv[8];
        double vtmp[8];
        {
            sv[0] = stencilValues[index_sv + 9 + 3 + 1];
            sv[1] = stencilValues[index_sv + 9 + 3];
            sv[2] = stencilValues[index_sv + 9 + 3 + 2];
            sv[3] = stencilValues[index_sv + 9 + 1];
            sv[4] = stencilValues[index_sv + 9 + 6 + 1];
            sv[5] = stencilValues[index_sv + 3 + 1];
            sv[6] = stencilValues[index_sv + 18 + 3 + 1];
            sv[7] = stencilValues[index_sv + 9];
            vtmp[0] = v_in[index];
            vtmp[1] = v_in[index - 1];
            vtmp[2] = v_in[index + 1];
            vtmp[3] = v_in[index - ogh];
            vtmp[4] = v_in[index + ogh];
            vtmp[5] = v_in[index - ioff];
            vtmp[6] = v_in[index + ioff];
            vtmp[7] = v_in[index - ogh - 1];;

            stencilsum += 
                  sv[0] * vtmp[0]
                + sv[1] * vtmp[1]
                + sv[2] * vtmp[2]
                + sv[3] * vtmp[3]
                + sv[4] * vtmp[4]
                + sv[5] * vtmp[5]
                + sv[6] * vtmp[6]
                + sv[7] * vtmp[7];
        }
        
        {
            sv[0] = stencilValues[index_sv + 9 + 2];
            sv[1] = stencilValues[index_sv + 9 + 6];
            sv[2] = stencilValues[index_sv + 9 + 6 + 2];
            sv[3] = stencilValues[index_sv + 3];
            sv[4] = stencilValues[index_sv + 3 + 2];
            sv[5] = stencilValues[index_sv + 18 + 3];
            sv[6] = stencilValues[index_sv + 18 + 3 + 2];
            sv[7] = stencilValues[index_sv + 1];
            vtmp[0] = v_in[index - ogh + 1];
            vtmp[1] = v_in[index + ogh - 1];
            vtmp[2] = v_in[index + ogh + 1];
            vtmp[3] = v_in[index - ioff - 1];
            vtmp[4] = v_in[index - ioff + 1];
            vtmp[5] = v_in[index + ioff - 1];
            vtmp[6] = v_in[index + ioff + 1];
            vtmp[7] = v_in[index - ioff - ogh];
            
            stencilsum +=
                  sv[0] * vtmp[0]
                + sv[1] * vtmp[1]
                + sv[2] * vtmp[2]
                + sv[3] * vtmp[3]
                + sv[4] * vtmp[4]
                + sv[5] * vtmp[5]
                + sv[6] * vtmp[6]
                + sv[7] * vtmp[7];
        }
        
        {
            sv[0] = stencilValues[index_sv + 6 + 1];
            sv[1] = stencilValues[index_sv + 18 + 1];
            sv[2] = stencilValues[index_sv + 18 + 6 + 1];
            sv[3] = stencilValues[index_sv];
            sv[4] = stencilValues[index_sv + 2];
            sv[5] = stencilValues[index_sv + 6];
            sv[6] = stencilValues[index_sv + 6 + 2];
            sv[7] = stencilValues[index_sv + 18];
            vtmp[0] = v_in[index - ioff + ogh];
            vtmp[1] = v_in[index + ioff - ogh];
            vtmp[2] = v_in[index + ioff + ogh];
            vtmp[3] = v_in[index - ioff - ogh - 1];
            vtmp[4] = v_in[index - ioff - ogh + 1];
            vtmp[5] = v_in[index - ioff + ogh - 1];
            vtmp[6] = v_in[index - ioff + ogh + 1];
            vtmp[7] = v_in[index + ioff - ogh - 1];
            
            stencilsum +=
                  sv[0] * vtmp[0]
                + sv[1] * vtmp[1]
                + sv[2] * vtmp[2]
                + sv[3] * vtmp[3]
                + sv[4] * vtmp[4]
                + sv[5] * vtmp[5]
                + sv[6] * vtmp[6]
                + sv[7] * vtmp[7];
        }
        
        {
            sv[0] = stencilValues[index_sv + 18 + 2];
            sv[1] = stencilValues[index_sv + 18 + 6];
            sv[2] = stencilValues[index_sv + 18 + 6 + 2];
            vtmp[0] = v_in[index + ioff - ogh + 1];
            vtmp[1] = v_in[index + ioff + ogh - 1];
            vtmp[2] = v_in[index + ioff + ogh + 1];;
            
            stencilsum +=
                  sv[0] * vtmp[0]
                + sv[1] * vtmp[1]
                + sv[2] * vtmp[2];
        }
        // clang-format on

        // r = f - A*v
        r[index] = f[index] - stencilsum;
    }
}

/* Calculates residual with a 27p varying stencil.
 * 1d kernel, must be launched with m*n*o work-items, i.e. #work-items == amount of real grid cells.
 * Work-group size can be arbitrary chosen. 32 seems optimal for my laptop gpu.
 * Arguments:
 *   v_in: current guess, same size of ghosted grid. Only gets read in this kernel.
 *      f: rhs, same size of ghosted grids. Only gets read in this kernel.
 *      r: residual, same size of ghosted grid. Only gets written in this kernel.
 * stencilValues: 27p varying stencil per grid cell. Size = mgh*ngh*ogh * 27.
 *      m: real grid size in dim 1
 *      n: real grid size in dim 2
 *      o: real grid size in dim 3
 * ghosts: Amount of ghosts of v_in, f and r
 * ghosts_sv: Amount of ghosts of stencilValues
 *   moff: Amount of ghost cells which shall be calculated in dim 1.
 *   noff: Amount of ghost cells which shall be calculated in dim 2.
 *   ooff: Amount of ghost cells which shall be calculated in dim 3.
 *   I.e. moff = -1 means that the first layer of ghost cells will be updated, too. That would require ghosts >= 2 however.
 */
// residual_27point_varying_stencil_1d_one_wi_per_cell_real_only
__kernel void residual_27point_varying_stencil_1d_gps_first_real_only(
    __global double* v_in,
    __global double* f,
    __global double* r,
    __global double* stencilValues,
    const int m, const int n, const int o,
    const int svmgh, const int svngh, const int svogh, // not needed but for sake of simplicity
    const int ghosts, const int ghosts_sv,
    const int svGridSize, // not needed but for sake of simplicity
    const int moff, const int noff, const int ooff)
{
    int idx = get_global_id(0);
    int no = n * o;
    int i = idx / no;
    int j = (idx - i * no) / o;
    int k = idx % o;

    // loop boundaries
    // TODO maybe refactor to use v_ghm, etc.?
    int istart_v = moff;
    int jstart_v = noff;
    int kstart_v = ooff;
    int iend_v = m - moff;
    int jend_v = n - noff;
    int kend_v = o - ooff;

    // calculate residual only for relevant cells (off = 0: only real cells)
    if (i >= istart_v && j >= jstart_v && k >= kstart_v && i < iend_v && j < jend_v && k < kend_v)
    {
        int ioff = (n + 2 * ghosts) * (o + 2 * ghosts);
        int joff = (o + 2 * ghosts);
        int koff = 1;
        int index = (i + ghosts) * ioff + (j + ghosts) * joff + k + ghosts;
        // int index = 200000;
        // if (i == 0 && j == 0 && k == 0)
        //     printf("i,j,k: %d, %d, %d, ioff,joff,koff: %d, %d, %d, idx: %d\n", i, j, k, ioff, joff, koff, index);
        // return;

        int koff_sv = 27;
        int joff_sv = (o + 2 * ghosts_sv) * koff_sv;
        int ioff_sv = (n + 2 * ghosts_sv) * joff_sv;
        int index_sv = (i + ghosts_sv) * ioff_sv + (j + ghosts_sv) * joff_sv + (k + ghosts_sv) * koff_sv;
        // int index_sv = 200000;

        // A*v
        // clang-format off
        double stencilsum = stencilValues[index_sv + 9 + 3 + 1] * v_in[index]
            + stencilValues[index_sv + 9 + 3]      * v_in[index - 1]
            + stencilValues[index_sv + 9 + 3 + 2]  * v_in[index + 1]
            + stencilValues[index_sv + 9 + 1]      * v_in[index - joff]
            + stencilValues[index_sv + 9 + 6 + 1]  * v_in[index + joff]
            + stencilValues[index_sv + 3 + 1]      * v_in[index - ioff]
            + stencilValues[index_sv + 18 + 3 + 1] * v_in[index + ioff]
            
            + stencilValues[index_sv + 9]          * v_in[index - joff - koff]
            + stencilValues[index_sv + 9 + 2]      * v_in[index - joff + koff]
            + stencilValues[index_sv + 9 + 6]      * v_in[index + joff - koff]
            + stencilValues[index_sv + 9 + 6 + 2]  * v_in[index + joff + koff]
            + stencilValues[index_sv + 3]          * v_in[index - ioff - koff]
            + stencilValues[index_sv + 3 + 2]      * v_in[index - ioff + koff]
            + stencilValues[index_sv + 18 + 3]     * v_in[index + ioff - koff]
            + stencilValues[index_sv + 18 + 3 + 2] * v_in[index + ioff + koff]
            + stencilValues[index_sv + 1]          * v_in[index - ioff - joff]
            + stencilValues[index_sv + 6 + 1]      * v_in[index - ioff + joff]
            + stencilValues[index_sv + 18 + 1]     * v_in[index + ioff - joff]
            + stencilValues[index_sv + 18 + 6 + 1] * v_in[index + ioff + joff]

            + stencilValues[index_sv]              * v_in[index - ioff - joff - koff]
            + stencilValues[index_sv + 2]          * v_in[index - ioff - joff + koff]
            + stencilValues[index_sv + 6]          * v_in[index - ioff + joff - koff]
            + stencilValues[index_sv + 6 + 2]      * v_in[index - ioff + joff + koff]
            + stencilValues[index_sv + 18]         * v_in[index + ioff - joff - koff]
            + stencilValues[index_sv + 18 + 2]     * v_in[index + ioff - joff + koff]
            + stencilValues[index_sv + 18 + 6]     * v_in[index + ioff + joff - koff]
            + stencilValues[index_sv + 18 + 6 + 2] * v_in[index + ioff + joff + koff];
        // clang-format on

        // r = f - A*v
        r[index] = f[index] - stencilsum;
    }
}

/* Calculates residual without dinv.
 * Calculates 4 grid points per work-item to increase ILP.
 */
__kernel void residual_27point_varying_stencil_coeffs_first_4_gps_per_thread(
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

    // 2nd gp
    idx += get_global_size(0);
    i = idx / no;
    j = (idx - i * no) / ogh;
    k = idx % ogh;
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

    // 3rd gp
    idx += get_global_size(0);
    i = idx / no;
    j = (idx - i * no) / ogh;
    k = idx % ogh;
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

    // 4th gp
    idx += get_global_size(0);
    i = idx / no;
    j = (idx - i * no) / ogh;
    k = idx % ogh;
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

/*
 * Calculates residual with a 27p varying stencil.
 * 1d kernel, must be launched with mgh*ngh*ogh*wiPerGridPoint work-items.
 * Multiple work-items calculate the entry for one grid point. The amount of work-items mapped to one grid point
 *   is defined by wiPerGridPoint.
 * Work-group size must be a multiple of wiPerGridPoint and optimally also a multiple of wrap size, e.g.
 *   for wiPerGridPoint = 2, wg size = 32 might be a good choice.
 * Arguments:
 *   v_in: current guess, same size of ghosted grid. Only gets read in this kernel.
 *      f: rhs, same size of ghosted grids. Only gets read in this kernel.
 *      r: residual, same size of ghosted grid. Only gets written in this kernel.
 * stencilValues: 27p varying stencil per grid cell. Size = mgh*ngh*ogh * 27.
 *      m: ghosted grid size in dim 1
 *      n: ghosted grid size in dim 2
 *      o: ghosted grid size in dim 3
 * ghosts: Amount of ghosts of v_in, f and r
 * ghosts_sv: Amount of ghosts of stencilValues
 *   moff: Amount of ghost cells which shall be calculated in dim 1.
 *   noff: Amount of ghost cells which shall be calculated in dim 2.
 *   ooff: Amount of ghost cells which shall be calculated in dim 3.
 *   I.e. moff = -1 means that the first layer of ghost cells will be updated, too. That would require ghosts >= 2 however.
 */
__kernel void residual_27point_varying_stencil_1d_mult_wi_per_cell_2(
    __global double* v_in,
    __global double* f,
    __global double* r,
    __global double* stencilValues,
    __local double* partials, // size = wg-size / 2
    const int m, const int n, const int o,
    const int svmgh, const int svngh, const int svogh, // not needed but for sake of simplicity
    const int ghosts, const int ghosts_sv,
    const int svGridSize, // not needed but for sake of simplicity
    const int moff, const int noff, const int ooff, const int wiPerGridPoint)
{
    int idx = get_global_id(0);
    // int idx = blockIdx.x * blockDim.x + get_local_id(0); // 1d work-item index
    int idx_gp = idx / wiPerGridPoint; // 1d grid point index
    int no = n * o;
    int i = idx_gp / no;
    int j = (idx_gp - i * no) / o;
    int k = idx_gp % o;
    // if (i == ghosts && j == ghosts && k == ghosts)
    //     printf("idx %d, idx_gp %d\n", idx, idx_gp);

    // calculate residual only for relevant cells (off = 0: only real cells)
    if (i >= ghosts + moff && j >= ghosts + noff && k >= ghosts + ooff &&
        i < m - ghosts - moff && j < n - ghosts - noff && k < o - ghosts - ooff)
    {
        int ioff = n * o;
        int joff = o;
        int koff = 1;
        int index = i * ioff + j * o + k;

        int koff_sv = 27;
        int joff_sv = ((o - 2 * ghosts) + 2 * ghosts_sv) * koff_sv;
        int ioff_sv = ((n - 2 * ghosts) + 2 * ghosts_sv) * joff_sv;
        int index_sv = (i + (ghosts_sv - ghosts)) * ioff_sv + (j + (ghosts_sv - ghosts)) * joff_sv + (k + (ghosts_sv - ghosts)) * koff_sv;

        // A*v
        // clang-format off
        double stencilsum = 0.0;
        // __shared__ double partials[blockDim.x / 2];
        // __shared__ double partials[64]; // hard-coded for block size 64
        if (get_local_id(0) % 2 == 0) {
    // if (i == ghosts && j == ghosts && k == ghosts)
    //     printf("get_local_id(0) %d\n", get_local_id(0));
            stencilsum = 
                  stencilValues[index_sv + 9 + 3 + 1]  * v_in[index]
                + stencilValues[index_sv + 9 + 3]      * v_in[index - 1]
                + stencilValues[index_sv + 9 + 3 + 2]  * v_in[index + 1]
                + stencilValues[index_sv + 9 + 1]      * v_in[index - joff]
                + stencilValues[index_sv + 9 + 6 + 1]  * v_in[index + joff]
                + stencilValues[index_sv + 3 + 1]      * v_in[index - ioff]
                + stencilValues[index_sv + 18 + 3 + 1] * v_in[index + ioff]
                
                + stencilValues[index_sv + 9]          * v_in[index - joff - koff]
                + stencilValues[index_sv + 9 + 2]      * v_in[index - joff + koff]
                + stencilValues[index_sv + 9 + 6]      * v_in[index + joff - koff]
                + stencilValues[index_sv + 9 + 6 + 2]  * v_in[index + joff + koff]
                + stencilValues[index_sv + 3]          * v_in[index - ioff - koff]
                + stencilValues[index_sv + 3 + 2]      * v_in[index - ioff + koff];
        } else {
    // if (i == ghosts && j == ghosts && k == ghosts)
    //     printf("get_local_id(0) %d\n", get_local_id(0));
            stencilsum = 
                  stencilValues[index_sv + 18 + 3]     * v_in[index + ioff - koff]
                + stencilValues[index_sv + 18 + 3 + 2] * v_in[index + ioff + koff]
                + stencilValues[index_sv + 1]          * v_in[index - ioff - joff]
                + stencilValues[index_sv + 6 + 1]      * v_in[index - ioff + joff]
                + stencilValues[index_sv + 18 + 1]     * v_in[index + ioff - joff]
                + stencilValues[index_sv + 18 + 6 + 1] * v_in[index + ioff + joff]

                + stencilValues[index_sv]              * v_in[index - ioff - joff - koff]
                + stencilValues[index_sv + 2]          * v_in[index - ioff - joff + koff]
                + stencilValues[index_sv + 6]          * v_in[index - ioff + joff - koff]
                + stencilValues[index_sv + 6 + 2]      * v_in[index - ioff + joff + koff]
                + stencilValues[index_sv + 18]         * v_in[index + ioff - joff - koff]
                + stencilValues[index_sv + 18 + 2]     * v_in[index + ioff - joff + koff]
                + stencilValues[index_sv + 18 + 6]     * v_in[index + ioff + joff - koff]
                + stencilValues[index_sv + 18 + 6 + 2] * v_in[index + ioff + joff + koff];
            // store result from upper half of warp in shared memory
            partials[get_local_id(0) - 1] = stencilsum;

    // if (i == ghosts && j == ghosts && k == ghosts)threadIdx.x
    //     printf("get_local_id(0) %d storing into %d\n", get_local_id(0), get_local_id(0) - (blockDim.x >> 1));
        }
        // clang-format on

        // wait for warp to finish and add results from left neighbour
        barrier(CLK_LOCAL_MEM_FENCE);
        if (get_local_id(0) % 2 == 0)
        {
            stencilsum += partials[get_local_id(0)];
            // if (i == ghosts && j == ghosts && k == ghosts)
            //     printf("get_local_id(0) %d reading from %d\n", get_local_id(0), get_local_id(0));

            // r = f - A*v
            r[index] = f[index] - stencilsum;
        }

        /*
        int ioff = no;
        // int joff = o;
        // int koff = 1;
        int index = (i - 1) * ioff + (j - 1) * o + k - 1; // start in upper left corner in front (stencil entry idx 0)

        int koff_sv = 27;
        int joff_sv = ((o - 2 * ghosts) + 2 * ghosts_sv) * koff_sv;
        int ioff_sv = ((n - 2 * ghosts) + 2 * ghosts_sv) * joff_sv;
        int index_sv = (i + (ghosts_sv - ghosts)) * ioff_sv + (j + (ghosts_sv - ghosts)) * joff_sv + (k + (ghosts_sv - ghosts)) * koff_sv;

        // A*v
        // clang-format off
        double stencilsum = 0.0;
        for (int ii = 0; ii < 3; ii++)
        for (int jj = 0; jj < 3; jj++)
        for (int kk = 0; kk < 3; kk++)
        {
            stencilsum += stencilValues[index_sv + ii * 9 + jj * 3 + kk] * v_in[index + ii * no + jj * o + kk];
        }
        // clang-format on
        */
    }
}

__kernel void residual_27point_varying_stencil_1d_mult_wi_per_cell_4(
    __global double* v_in,
    __global double* f,
    __global double* r,
    __global double* stencilValues,
    __local double* partials, // size = (wg-size / 4) * 3
    const int m, const int n, const int o,
    const int svmgh, const int svngh, const int svogh, // not needed but for sake of simplicity
    const int ghosts, const int ghosts_sv,
    const int svGridSize, // not needed but for sake of simplicity
    const int moff, const int noff, const int ooff, const int wiPerGridPoint)
{
    int idx = get_global_id(0);
    // int idx = blockIdx.x * blockDim.x + get_local_id(0); // 1d work-item index
    int idx_gp = idx / wiPerGridPoint; // 1d grid point index
    int no = n * o;
    int i = idx_gp / no;
    int j = (idx_gp - i * no) / o;
    int k = idx_gp % o;
    // if (i == ghosts && j == ghosts && k == ghosts)
    //     printf("idx %d, idx_gp %d\n", idx, idx_gp);

    // calculate residual only for relevant cells (off = 0: only real cells)
    if (i >= ghosts + moff && j >= ghosts + noff && k >= ghosts + ooff &&
        i < m - ghosts - moff && j < n - ghosts - noff && k < o - ghosts - ooff)
    {
        int ioff = n * o;
        int joff = o;
        int koff = 1;
        int index = i * ioff + j * o + k;

        int koff_sv = 27;
        int joff_sv = ((o - 2 * ghosts) + 2 * ghosts_sv) * koff_sv;
        int ioff_sv = ((n - 2 * ghosts) + 2 * ghosts_sv) * joff_sv;
        int index_sv = (i + (ghosts_sv - ghosts)) * ioff_sv + (j + (ghosts_sv - ghosts)) * joff_sv + (k + (ghosts_sv - ghosts)) * koff_sv;

        // A*v
        // clang-format off
        double stencilsum = 0.0;
        // __shared__ double partials[blockDim.x / 2];
        // __shared__ double partials[32]; // hard-coded for block size 64
        if (get_local_id(0) % 4 == 0) {
    // if (i == ghosts && j == ghosts && k == ghosts)
    //     printf("get_local_id(0) %d\n", get_local_id(0));
            stencilsum = 
                  stencilValues[index_sv + 9 + 3 + 1]  * v_in[index]
                + stencilValues[index_sv + 9 + 3]      * v_in[index - 1]
                + stencilValues[index_sv + 9 + 3 + 2]  * v_in[index + 1]
                + stencilValues[index_sv + 9 + 1]      * v_in[index - joff]
                + stencilValues[index_sv + 9 + 6 + 1]  * v_in[index + joff]
                + stencilValues[index_sv + 3 + 1]      * v_in[index - ioff]
                + stencilValues[index_sv + 18 + 3 + 1] * v_in[index + ioff];
        } else if (get_local_id(0) % 4 == 1) {
            stencilsum = 
                  stencilValues[index_sv + 9]          * v_in[index - joff - koff]
                + stencilValues[index_sv + 9 + 2]      * v_in[index - joff + koff]
                + stencilValues[index_sv + 9 + 6]      * v_in[index + joff - koff]
                + stencilValues[index_sv + 9 + 6 + 2]  * v_in[index + joff + koff]
                + stencilValues[index_sv + 3]          * v_in[index - ioff - koff]
                + stencilValues[index_sv + 3 + 2]      * v_in[index - ioff + koff];
            partials[get_local_id(0)] = stencilsum;
        } else if (get_local_id(0) % 4 == 2) {
            stencilsum = 
                  stencilValues[index_sv + 18 + 3]     * v_in[index + ioff - koff]
                + stencilValues[index_sv + 18 + 3 + 2] * v_in[index + ioff + koff]
                + stencilValues[index_sv + 1]          * v_in[index - ioff - joff]
                + stencilValues[index_sv + 6 + 1]      * v_in[index - ioff + joff]
                + stencilValues[index_sv + 18 + 1]     * v_in[index + ioff - joff]
                + stencilValues[index_sv + 18 + 6 + 1] * v_in[index + ioff + joff]
                + stencilValues[index_sv]              * v_in[index - ioff - joff - koff];
            partials[get_local_id(0)] = stencilsum;
        } else {
    // if (i == ghosts && j == ghosts && k == ghosts)
    //     printf("get_local_id(0) %d\n", get_local_id(0));
            stencilsum = 
                  stencilValues[index_sv + 2]          * v_in[index - ioff - joff + koff]
                + stencilValues[index_sv + 6]          * v_in[index - ioff + joff - koff]
                + stencilValues[index_sv + 6 + 2]      * v_in[index - ioff + joff + koff]
                + stencilValues[index_sv + 18]         * v_in[index + ioff - joff - koff]
                + stencilValues[index_sv + 18 + 2]     * v_in[index + ioff - joff + koff]
                + stencilValues[index_sv + 18 + 6]     * v_in[index + ioff + joff - koff]
                + stencilValues[index_sv + 18 + 6 + 2] * v_in[index + ioff + joff + koff];
            // store result from upper half of warp in shared memory
            partials[get_local_id(0)] = stencilsum;

    // if (i == ghosts && j == ghosts && k == ghosts)
    //     printf("get_local_id(0) %d storing into %d\n", get_local_id(0), get_local_id(0) - (blockDim.x >> 1));
        }
        // clang-format on

        // wait for warp to finish and add results from left neighbour
        barrier(CLK_LOCAL_MEM_FENCE);
        if (get_local_id(0) % 4 == 0)
        {
            stencilsum += partials[get_local_id(0) + 1];
            stencilsum += partials[get_local_id(0) + 2];
            stencilsum += partials[get_local_id(0) + 3];
            // if (i == ghosts && j == ghosts && k == ghosts)
            //     printf("get_local_id(0) %d reading from %d\n", get_local_id(0), get_local_id(0));

            // r = f - A*v
            r[index] = f[index] - stencilsum;
        }

        /*
        int ioff = no;
        // int joff = o;
        // int koff = 1;
        int index = (i - 1) * ioff + (j - 1) * o + k - 1; // start in upper left corner in front (stencil entry idx 0)

        int koff_sv = 27;
        int joff_sv = ((o - 2 * ghosts) + 2 * ghosts_sv) * koff_sv;
        int ioff_sv = ((n - 2 * ghosts) + 2 * ghosts_sv) * joff_sv;
        int index_sv = (i + (ghosts_sv - ghosts)) * ioff_sv + (j + (ghosts_sv - ghosts)) * joff_sv + (k + (ghosts_sv - ghosts)) * koff_sv;

        // A*v
        // clang-format off
        double stencilsum = 0.0;
        for (int ii = 0; ii < 3; ii++)
        for (int jj = 0; jj < 3; jj++)
        for (int kk = 0; kk < 3; kk++)
        {
            stencilsum += stencilValues[index_sv + ii * 9 + jj * 3 + kk] * v_in[index + ii * no + jj * o + kk];
        }
        // clang-format on
        */
    }
}

/*
 * This kernel must be called with 4 wi per grid node. Each wi applies a part of the stencil and
 * stores the result in shared memory. The first wi of the 4 will build the sum in the end.
 * Stencil values are spread out, see other sv spread kernel for more information.
 */
__kernel void residual_27point_varying_stencil_1d_mult_wi_per_cell_4_sv_spread(
    __global double* v_in,
    __global double* f,
    __global double* r,
    __global double* stencilValues,
    __local double* partials, // size = (wg-size / 4) * 3
    const int m, const int n, const int o,
    const int svmgh, const int svngh, const int svogh, // not needed but for sake of simplicity
    const int ghosts, const int ghosts_sv,
    const int svGridSize, // not needed but for sake of simplicity
    const int moff, const int noff, const int ooff, const int wiPerGridPoint)
{
    int idx = get_global_id(0);
    // int idx = blockIdx.x * blockDim.x + get_local_id(0); // 1d work-item index
    int idx_gp = idx / wiPerGridPoint; // 1d grid point index
    int no = n * o;
    int i = idx_gp / no;
    int j = (idx_gp - i * no) / o;
    int k = idx_gp % o;
    // if (i == ghosts && j == ghosts && k == ghosts)
    //     printf("idx %d, idx_gp %d\n", idx, idx_gp);

    // calculate residual only for relevant cells (off = 0: only real cells)
    if (i >= ghosts + moff && j >= ghosts + noff && k >= ghosts + ooff &&
        i < m - ghosts - moff && j < n - ghosts - noff && k < o - ghosts - ooff)
    {
        int ioff = n * o;
        int joff = o;
        int koff = 1;
        int index = i * ioff + j * o + k;
        // int gridsize = m * n * o;

        // int koff_sv = 27;
        // int joff_sv = ((o - 2 * ghosts) + 2 * ghosts_sv) * koff_sv;
        // int ioff_sv = ((n - 2 * ghosts) + 2 * ghosts_sv) * joff_sv;
        // int index_sv = (i + (ghosts_sv - ghosts)) * ioff_sv + (j + (ghosts_sv - ghosts)) * joff_sv + (k + (ghosts_sv - ghosts)) * koff_sv;
        int svno = svngh * svogh;
        // offset inside one coefficient grid that points to the coefficient for the current grid point. Must consider different amount of ghosts for v and sv.
        int index_sv = (i - ghosts + ghosts_sv) * svno + (j - ghosts + ghosts_sv) * svogh + (k - ghosts + ghosts_sv);

        // A*v
        // clang-format off
        double stencilsum = 0.0;
        // __shared__ double partials[blockDim.x / 2];
        // __shared__ double partials[64]; // hard-coded for block size 64
        if (get_local_id(0) % 4 == 0) {
    // if (i == ghosts && j == ghosts && k == ghosts)
    //     printf("get_local_id(0) %d\n", get_local_id(0));
            stencilsum = 
                stencilValues[index_sv + (9 + 3 + 1) * svGridSize] * v_in[index]
                + stencilValues[index_sv + (9 + 3) * svGridSize]      * v_in[index - 1]
                + stencilValues[index_sv + (9 + 3 + 2) * svGridSize]  * v_in[index + 1]
                + stencilValues[index_sv + (9 + 1) * svGridSize]      * v_in[index - joff]
                + stencilValues[index_sv + (9 + 6 + 1) * svGridSize]  * v_in[index + joff]
                + stencilValues[index_sv + (3 + 1) * svGridSize]      * v_in[index - ioff]
                + stencilValues[index_sv + (18 + 3 + 1) * svGridSize] * v_in[index + ioff];
        } else if (get_local_id(0) % 4 == 1) {
            stencilsum = 
                + stencilValues[index_sv + (9) * svGridSize]          * v_in[index - joff - koff]
                + stencilValues[index_sv + (9 + 2) * svGridSize]      * v_in[index - joff + koff]
                + stencilValues[index_sv + (9 + 6) * svGridSize]      * v_in[index + joff - koff]
                + stencilValues[index_sv + (9 + 6 + 2) * svGridSize]  * v_in[index + joff + koff]
                + stencilValues[svGridSize * 3 + index]          * v_in[index - ioff - koff]
                + stencilValues[index_sv + (3 + 2) * svGridSize]      * v_in[index - ioff + koff]
                + stencilValues[index_sv + (18 + 3) * svGridSize]     * v_in[index + ioff - koff];
            partials[get_local_id(0)] = stencilsum;
        } else if (get_local_id(0) % 4 == 2) {
            stencilsum = 
                + stencilValues[index_sv + (18 + 3 + 2) * svGridSize] * v_in[index + ioff + koff]
                + stencilValues[svGridSize + index]          * v_in[index - ioff - joff]
                + stencilValues[index_sv + (6 + 1) * svGridSize]      * v_in[index - ioff + joff]
                + stencilValues[index_sv + (18 + 1) * svGridSize]     * v_in[index + ioff - joff]
                + stencilValues[index_sv + (18 + 6 + 1) * svGridSize] * v_in[index + ioff + joff]
                + stencilValues[index_sv]              * v_in[index - ioff - joff - koff]
                + stencilValues[svGridSize * 2 + index]          * v_in[index - ioff - joff + koff];
            partials[get_local_id(0)] = stencilsum;
        } else {
    // if (i == ghosts && j == ghosts && k == ghosts)
    //     printf("get_local_id(0) %d\n", get_local_id(0));
            stencilsum = 
                + stencilValues[index_sv + (6) * svGridSize]          * v_in[index - ioff + joff - koff]
                + stencilValues[index_sv + (6 + 2) * svGridSize]      * v_in[index - ioff + joff + koff]
                + stencilValues[index_sv + (18) * svGridSize]         * v_in[index + ioff - joff - koff]
                + stencilValues[index_sv + (18 + 2) * svGridSize]     * v_in[index + ioff - joff + koff]
                + stencilValues[index_sv + (18 + 6) * svGridSize]     * v_in[index + ioff + joff - koff]
                + stencilValues[index_sv + (18 + 6 + 2) * svGridSize] * v_in[index + ioff + joff + koff];
            // store result from upper half of warp in shared memory
            partials[get_local_id(0)] = stencilsum;

    // if (i == ghosts && j == ghosts && k == ghosts)
    //     printf("get_local_id(0) %d storing into %d\n", get_local_id(0), get_local_id(0) - (blockDim.x >> 1));
        }
        // clang-format on

        // wait for warp to finish and add results from left neighbour
        barrier(CLK_LOCAL_MEM_FENCE);
        if (get_local_id(0) % 4 == 0)
        {
            stencilsum += partials[get_local_id(0) + 1];
            stencilsum += partials[get_local_id(0) + 2];
            stencilsum += partials[get_local_id(0) + 3];
            // if (i == ghosts && j == ghosts && k == ghosts)
            //     printf("get_local_id(0) %d reading from %d\n", get_local_id(0), get_local_id(0));

            // r = f - A*v
            r[index] = f[index] - stencilsum;
        }
    }
}

/*
 * This kernel must be called with 4 wi per grid node. Each wi applies a part of the stencil and
 * stores the result in shared memory. The first wi of the 4 will build the sum in the end.
 * Stencil values are spread out, see other sv spread kernel for more information.
 * The wi associated with one grid point are spread out evenly in the whole block. E.g. when
 *   block=32 and wiPerGridPoint=4, every 8th wi will calculate data regarding grid point 0. Thus,
 *   if the block size is large enough, one warp won't suffer from branch divergence.
 */
__kernel void residual_27point_varying_stencil_1d_mult_wi_per_cell_4_sv_spread_shmem_spread(
    __global double* v_in,
    __global double* f,
    __global double* r,
    __global double* stencilValues,
    __local double* partials, // size = (wg-size / 4) * 3
    const int m, const int n, const int o,
    const int svmgh, const int svngh, const int svogh, // not needed but for sake of simplicity
    const int ghosts, const int ghosts_sv,
    const int moff, const int noff, const int ooff,
    const int svGridSize, // not needed but for sake of simplicity
    const int wiPerGridPoint
    // ,const int gridPointsPerBlock, const int gridsize
)
{
    int idx = get_global_id(0);
    // int idx = blockIdx.x * blockDim.x + get_local_id(0); // 1d work-item index
    int gridPointsPerBlock = get_local_size(0) / wiPerGridPoint;
    int idx_gp = (idx % gridPointsPerBlock) + get_group_id(0) * gridPointsPerBlock; // 1d grid point index
    int no = n * o;
    int i = idx_gp / no;
    int j = (idx_gp - i * no) / o;
    int k = idx_gp % o;
    // if (i == ghosts && j == ghosts && k == ghosts)
    //     printf("idx %d, idx_gp %d\n", idx, idx_gp);

    // calculate residual only for relevant cells (off = 0: only real cells)
    if (i >= ghosts + moff && j >= ghosts + noff && k >= ghosts + ooff &&
        i < m - ghosts - moff && j < n - ghosts - noff && k < o - ghosts - ooff)
    {
        int ioff = n * o;
        int joff = o;
        int koff = 1;
        int index = i * ioff + j * o + k;
        // int gridsize = m * n * o;

        // int koff_sv = 27;
        // int joff_sv = ((o - 2 * ghosts) + 2 * ghosts_sv) * koff_sv;
        // int ioff_sv = ((n - 2 * ghosts) + 2 * ghosts_sv) * joff_sv;
        // int index_sv = (i + (ghosts_sv - ghosts)) * ioff_sv + (j + (ghosts_sv - ghosts)) * joff_sv + (k + (ghosts_sv - ghosts)) * koff_sv;
        int svno = svngh * svogh;
        // offset inside one coefficient grid that points to the coefficient for the current grid point. Must consider different amount of ghosts for v and sv.
        int index_sv = (i - ghosts + ghosts_sv) * svno + (j - ghosts + ghosts_sv) * svogh + (k - ghosts + ghosts_sv);

        // A*v
        // clang-format off
        double stencilsum = 0.0;
        // __shared__ double partials[blockDim.x / 2];
        // __shared__ double partials[256]; // hard-coded for block size 256
        if (get_local_id(0) < gridPointsPerBlock) {
    // if (i == ghosts && j == ghosts && k == ghosts)
    //     printf("get_local_id(0) %d\n", get_local_id(0));
            stencilsum = 
                stencilValues[index_sv + (9 + 3 + 1) * svGridSize] * v_in[index]
                + stencilValues[index_sv + (9 + 3) * svGridSize]      * v_in[index - 1]
                + stencilValues[index_sv + (9 + 3 + 2) * svGridSize]  * v_in[index + 1]
                + stencilValues[index_sv + (9 + 1) * svGridSize]      * v_in[index - joff]
                + stencilValues[index_sv + (9 + 6 + 1) * svGridSize]  * v_in[index + joff]
                + stencilValues[index_sv + (3 + 1) * svGridSize]      * v_in[index - ioff]
                + stencilValues[index_sv + (18 + 3 + 1) * svGridSize] * v_in[index + ioff];
        } else if (get_local_id(0) < gridPointsPerBlock * 2) {
            stencilsum = 
                + stencilValues[index_sv + (9) * svGridSize]          * v_in[index - joff - koff]
                + stencilValues[index_sv + (9 + 2) * svGridSize]      * v_in[index - joff + koff]
                + stencilValues[index_sv + (9 + 6) * svGridSize]      * v_in[index + joff - koff]
                + stencilValues[index_sv + (9 + 6 + 2) * svGridSize]  * v_in[index + joff + koff]
                + stencilValues[svGridSize * 3 + index]          * v_in[index - ioff - koff]
                + stencilValues[index_sv + (3 + 2) * svGridSize]      * v_in[index - ioff + koff]
                + stencilValues[index_sv + (18 + 3) * svGridSize]     * v_in[index + ioff - koff];
            partials[get_local_id(0)] = stencilsum;
        } else if (get_local_id(0) < gridPointsPerBlock * 3) {
            stencilsum = 
                + stencilValues[index_sv + (18 + 3 + 2) * svGridSize] * v_in[index + ioff + koff]
                + stencilValues[svGridSize + index]          * v_in[index - ioff - joff]
                + stencilValues[index_sv + (6 + 1) * svGridSize]      * v_in[index - ioff + joff]
                + stencilValues[index_sv + (18 + 1) * svGridSize]     * v_in[index + ioff - joff]
                + stencilValues[index_sv + (18 + 6 + 1) * svGridSize] * v_in[index + ioff + joff]
                + stencilValues[index_sv]              * v_in[index - ioff - joff - koff]
                + stencilValues[svGridSize * 2 + index]          * v_in[index - ioff - joff + koff];
            partials[get_local_id(0)] = stencilsum;
        } else {
    // if (i == ghosts && j == ghosts && k == ghosts)
    //     printf("get_local_id(0) %d\n", get_local_id(0));
            stencilsum = 
                + stencilValues[index_sv + (6) * svGridSize]          * v_in[index - ioff + joff - koff]
                + stencilValues[index_sv + (6 + 2) * svGridSize]      * v_in[index - ioff + joff + koff]
                + stencilValues[index_sv + (18) * svGridSize]         * v_in[index + ioff - joff - koff]
                + stencilValues[index_sv + (18 + 2) * svGridSize]     * v_in[index + ioff - joff + koff]
                + stencilValues[index_sv + (18 + 6) * svGridSize]     * v_in[index + ioff + joff - koff]
                + stencilValues[index_sv + (18 + 6 + 2) * svGridSize] * v_in[index + ioff + joff + koff];
            // store result from upper half of warp in shared memory
            partials[get_local_id(0)] = stencilsum;

    // if (i == ghosts && j == ghosts && k == ghosts)
    //     printf("get_local_id(0) %d storing into %d\n", get_local_id(0), get_local_id(0) - (blockDim.x >> 1));
        }
        // clang-format on

        // wait for warp to finish and add results from other work-items calculating for this grid point
        barrier(CLK_LOCAL_MEM_FENCE);
        if (get_local_id(0) < gridPointsPerBlock)
        {
            stencilsum += partials[get_local_id(0) + gridPointsPerBlock];
            stencilsum += partials[get_local_id(0) + gridPointsPerBlock * 2];
            stencilsum += partials[get_local_id(0) + gridPointsPerBlock * 3];
            // if (i == ghosts && j == ghosts && k == ghosts)
            //     printf("get_local_id(0) %d reading from %d\n", get_local_id(0), get_local_id(0));

            // r = f - A*v
            r[index] = f[index] - stencilsum;
        }
    }
}

/*
 * This kernel must be called with 2 wi per grid node. Each wi applies a part of the stencil and
 * stores the result in shared memory. The first wi of the 2 will build the sum in the end.
 * Stencil values are spread out, see other sv spread kernel for more information.
 * The wi associated with one grid point are spread out evenly in the whole block. E.g. when
 *   block=32 and wiPerGridPoint=2, every 16th wi will calculate data regarding grid point 0. Thus,
 *   if the block size is large enough, one warp won't suffer from branch divergence.
 */
__kernel void residual_27point_varying_stencil_1d_mult_wi_per_cell_2_sv_spread_shmem_spread(
    __global double* v_in,
    __global double* f,
    __global double* r,
    __global double* stencilValues,
    __local double* partials, // size = wg-size / 2
    const int m, const int n, const int o,
    const int svmgh, const int svngh, const int svogh, // not needed but for sake of simplicity
    const int ghosts, const int ghosts_sv,
    const int moff, const int noff, const int ooff,
    const int svGridSize, // not needed but for sake of simplicity
    const int wiPerGridPoint
    // , const int gridPointsPerBlock, const int gridsize
)
{
    int idx = get_global_id(0);
    // int idx = blockIdx.x * blockDim.x + get_local_id(0); // 1d work-item index
    // int gridPointsPerBlock = blockDim.x / wiPerGridPoint;
    int gridPointsPerBlock = get_local_size(0) / wiPerGridPoint;
    int idx_gp = (idx % gridPointsPerBlock) + get_group_id(0) * gridPointsPerBlock; // 1d grid point index
    int no = n * o;
    int i = idx_gp / no;
    int j = (idx_gp - i * no) / o;
    int k = idx_gp % o;
    // if (i == ghosts && j == ghosts && k == ghosts)
    //     printf("idx %d, idx_gp %d\n", idx, idx_gp);

    // calculate residual only for relevant cells (off = 0: only real cells)
    if (i >= ghosts + moff && j >= ghosts + noff && k >= ghosts + ooff &&
        i < m - ghosts - moff && j < n - ghosts - noff && k < o - ghosts - ooff)
    {
        int ioff = n * o;
        int joff = o;
        int koff = 1;
        int index = i * ioff + j * o + k;
        int gridsize = m * n * o;

        // int koff_sv = 27;
        // int joff_sv = ((o - 2 * ghosts) + 2 * ghosts_sv) * koff_sv;
        // int ioff_sv = ((n - 2 * ghosts) + 2 * ghosts_sv) * joff_sv;
        // int index_sv = (i + (ghosts_sv - ghosts)) * ioff_sv + (j + (ghosts_sv - ghosts)) * joff_sv + (k + (ghosts_sv - ghosts)) * koff_sv;
        int svno = svngh * svogh;
        // offset inside one coefficient grid that points to the coefficient for the current grid point. Must consider different amount of ghosts for v and sv.
        int index_sv = (i - ghosts + ghosts_sv) * svno + (j - ghosts + ghosts_sv) * svogh + (k - ghosts + ghosts_sv);

        // A*v
        // clang-format off
        double stencilsum = 0.0;
        // __shared__ double partials[blockDim.x / 2];
        // __shared__ double partials[128]; // hard-coded for block size 128
        if (get_local_id(0) < gridPointsPerBlock) {
    // if (i == ghosts && j == ghosts && k == ghosts)
    //     printf("get_local_id(0) %d\n", get_local_id(0));
            stencilsum = 
                stencilValues[index_sv + (9 + 3 + 1) * svGridSize] * v_in[index]
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
                + stencilValues[svGridSize * 3 + index]          * v_in[index - ioff - koff]
                + stencilValues[index_sv + (3 + 2) * svGridSize]      * v_in[index - ioff + koff]
                + stencilValues[index_sv + (18 + 3) * svGridSize]     * v_in[index + ioff - koff];
        } else if (get_local_id(0) < gridPointsPerBlock * 2) {
            stencilsum = 
                + stencilValues[index_sv + (18 + 3 + 2) * svGridSize] * v_in[index + ioff + koff]
                + stencilValues[svGridSize + index]          * v_in[index - ioff - joff]
                + stencilValues[index_sv + (6 + 1) * svGridSize]      * v_in[index - ioff + joff]
                + stencilValues[index_sv + (18 + 1) * svGridSize]     * v_in[index + ioff - joff]
                + stencilValues[index_sv + (18 + 6 + 1) * svGridSize] * v_in[index + ioff + joff]
                + stencilValues[index_sv]              * v_in[index - ioff - joff - koff]
                + stencilValues[svGridSize * 2 + index]          * v_in[index - ioff - joff + koff]
                + stencilValues[index_sv + (6) * svGridSize]          * v_in[index - ioff + joff - koff]
                + stencilValues[index_sv + (6 + 2) * svGridSize]      * v_in[index - ioff + joff + koff]
                + stencilValues[index_sv + (18) * svGridSize]         * v_in[index + ioff - joff - koff]
                + stencilValues[index_sv + (18 + 2) * svGridSize]     * v_in[index + ioff - joff + koff]
                + stencilValues[index_sv + (18 + 6) * svGridSize]     * v_in[index + ioff + joff - koff]
                + stencilValues[index_sv + (18 + 6 + 2) * svGridSize] * v_in[index + ioff + joff + koff];
            partials[get_local_id(0)] = stencilsum;
        }
        // clang-format on

        // wait for warp to finish and add results from other work-items calculating for this grid point
        barrier(CLK_LOCAL_MEM_FENCE);
        if (get_local_id(0) < gridPointsPerBlock)
        {
            stencilsum += partials[get_local_id(0) + gridPointsPerBlock];

            // r = f - A*v
            r[index] = f[index] - stencilsum;
        }
    }
}

// This version shall test the impact of removing the necessity to load v (the grid points itself).
// We expect only a marginal impact at best, since v is only 1/27th the amound of data opposed to coefficients.
__kernel void residual_27point_varying_stencil_no_v(
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
        double stencilsum = stencilValues[index_sv + (9 + 3 + 1) * svGridSize] * 2.0
            + stencilValues[index_sv + (9 + 3) * svGridSize]      * 2.0
            + stencilValues[index_sv + (9 + 3 + 2) * svGridSize]  * 2.0
            + stencilValues[index_sv + (9 + 1) * svGridSize]      * 2.0
            + stencilValues[index_sv + (9 + 6 + 1) * svGridSize]  * 2.0
            + stencilValues[index_sv + (3 + 1) * svGridSize]      * 2.0
            + stencilValues[index_sv + (18 + 3 + 1) * svGridSize] * 2.0
            
            + stencilValues[index_sv + (9) * svGridSize]          * 2.0
            + stencilValues[index_sv + (9 + 2) * svGridSize]      * 2.0
            + stencilValues[index_sv + (9 + 6) * svGridSize]      * 2.0
            + stencilValues[index_sv + (9 + 6 + 2) * svGridSize]  * 2.0
            + stencilValues[svGridSize * 3 + index_sv]            * 2.0
            + stencilValues[index_sv + (3 + 2) * svGridSize]      * 2.0
            + stencilValues[index_sv + (18 + 3) * svGridSize]     * 2.0
            + stencilValues[index_sv + (18 + 3 + 2) * svGridSize] * 2.0
            + stencilValues[svGridSize + index_sv]                * 2.0
            + stencilValues[index_sv + (6 + 1) * svGridSize]      * 2.0
            + stencilValues[index_sv + (18 + 1) * svGridSize]     * 2.0
            + stencilValues[index_sv + (18 + 6 + 1) * svGridSize] * 2.0

            + stencilValues[index_sv]                           * 2.0
            + stencilValues[svGridSize * 2 + index_sv]            * 2.0
            + stencilValues[index_sv + (6) * svGridSize]          * 2.0
            + stencilValues[index_sv + (6 + 2) * svGridSize]      * 2.0
            + stencilValues[index_sv + (18) * svGridSize]         * 2.0
            + stencilValues[index_sv + (18 + 2) * svGridSize]     * 2.0
            + stencilValues[index_sv + (18 + 6) * svGridSize]     * 2.0
            + stencilValues[index_sv + (18 + 6 + 2) * svGridSize] * 2.0;
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

/****************
 * Kernels from mgcl below
 **************** */

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

/* Calculates the squares of the residual.
 * r is ghosted, rsquares must be only real cells.
 * m, n and o must be real grid size.
 * ghosts is amount of ghost cells at one border for r.
 * Kernel must be called with m x n x o work-items.
 */
__kernel void residual_squared(
    __global double* restrict r,
    __global double* restrict rsquares,
    const int m, const int n, const int o, const int ghosts)
{
    int idx = get_global_id(0);
    int no = n * o;
    int i = idx / no;
    int j = (idx - i * no) / o;
    int k = idx % o;

    // account for padding
    if (i < m && j < n && k < o)
    {
        int index = i * (n + 2 * ghosts) * (o + 2 * ghosts) + j * (o + 2 * ghosts) + k;
        int index_sq = i * n * o + j * o + k;
        double ridx = r[index];
        rsquares[index_sq] = ridx * ridx;
    }
}

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
