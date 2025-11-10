#ifndef NULL
#define NULL 0
#endif

/**
 * This is the default residual kernel as in production code.
 */
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
 * Same as in production code.
 */
__kernel void jacobi_iter_27point_varying_stencil_1d_update_step_only(
    __global double* restrict v, // needed s.t. every work-item can read surrounding cell values
    __global double* restrict f,
    __global double* restrict r,
    __global double* restrict stencilValues,
    const double omega,
    const int mgh, const int ngh, const int ogh,
    const int svmgh, const int svngh, const int svogh,
    const int ghosts, const int ghosts_sv,
    const int svGridSize,
    const int idx_start)
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
        int index = i * ioff + j * ogh + k;

        int svno = svngh * svogh;
        // offset inside one coefficient grid that points to the coefficient for the current grid point. Must consider different amount of ghosts for v and sv.
        int index_sv = (i - ghosts + ghosts_sv) * svno + (j - ghosts + ghosts_sv) * svogh + (k - ghosts + ghosts_sv);

        double sv_self = stencilValues[index_sv + (9 + 3 + 1) * svGridSize];

        // u_(m+1) = u_(m) + omega * (D^-1) * r_(m)
        v[index] = v[index] + omega * (1.0 / sv_self) * r[index];
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
 * Same as in production code.
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
 *
 * Same as in production code.
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

/**************************************************/
/*                                                */
/* Usual kernels from here on                     */
/*                                                */
/**************************************************/

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
