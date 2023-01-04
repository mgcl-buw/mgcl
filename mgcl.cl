#define NULL 0

/* Prints components of 7-point laplacian stencil for debugging purposes */
void print_7point(__global double *A, int index, int ioff, int joff, int koff)
{
    printf("7point stencil at %d:\n", index);
    printf("v[self] = %e\n", A[index]);
    printf(" v[k-1] = %e\n", A[index - koff]);
    printf(" v[k+1] = %e\n", A[index + koff]);
    printf(" v[j-1] = %e\n", A[index - joff]);
    printf(" v[j+1] = %e\n", A[index + joff]);
    printf(" v[i-1] = %e\n", A[index - ioff]);
    printf(" v[i+1] = %e\n", A[index + ioff]);
}

void print_7point_local(__local double *A, __local double *rm, __local double *rp, int index, int ioff, int joff,
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

void print_19point_local(__local double *A, __local double *rm, __local double *rp, int index, int ioff, int joff,
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

/* Updates ghost cells. m,n,o must be size of ghosted grid
 * One cell per work-item */
__kernel void update_ghosts(__global double *restrict v, const int m, const int n, const int o)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);

    if (i < m && j < n && k < o)
    {
        int index = i * n * o + j * o + k;
        int ghosts = 1;

        // indices of right real cells per left ghost cell
        int mr = i + (m - 2 * ghosts);
        int nr = j + (n - 2 * ghosts);
        int or = k + (o - 2 * ghosts);

        // indices of left real cells per right ghost cell
        int ml = i - (m - 2 * ghosts);
        int nl = j - (n - 2 * ghosts);
        int ol = k - (o - 2 * ghosts);
        // update ghost cells for periodic boundary condition
        // make sure that every cell gets written only once to avoid race condition by using if-statements.
        // cells are divided by xy-planes (behind, self and in front of grid)
        // same xy-plane as grid
        if (k >= ghosts && k < o - ghosts)
        {
            if (i < ghosts && j >= ghosts && j < n - ghosts) // middle left
                v[index] = v[mr * n * o + j * o + k];
            if (i >= m - ghosts && j >= ghosts && j < n - ghosts) // middle right
                v[index] = v[ml * n * o + j * o + k];
            if (j < ghosts && i >= ghosts && i < m - ghosts) // top middle
                v[index] = v[i * n * o + nr * o + k];
            if (j >= n - ghosts && i >= ghosts && i < m - ghosts) // bottom middle
                v[index] = v[i * n * o + nl * o + k];
            if (i < ghosts && j < ghosts) // top left
                v[index] = v[mr * n * o + nr * o + k];
            if (i >= m - ghosts && j < ghosts) // top right
                v[index] = v[ml * n * o + nr * o + k];
            if (i < ghosts && j >= n - ghosts) // bottom left
                v[index] = v[mr * n * o + nl * o + k];
            if (i >= m - ghosts && j >= n - ghosts) // bottom right
                v[index] = v[ml * n * o + nl * o + k];
        }

        // xy-plane in front of grid
        if (k < ghosts)
        {
            if (i < ghosts && j < ghosts) // top left
                v[index] = v[mr * n * o + nr * o + or ];
            if (i < ghosts && j >= ghosts && j < n - ghosts) // middle left
                v[index] = v[mr * n * o + j * o + or ];
            if (i < ghosts && j >= n - ghosts) // bottom left
                v[index] = v[mr * n * o + nl * o + or ];

            if (i >= ghosts && i < m - ghosts && j >= ghosts && j < n - ghosts)
                v[index] = v[i * n * o + j * o + or ];       // center middle
            if (i >= ghosts && i < m - ghosts && j < ghosts) // top middle
                v[index] = v[i * n * o + nr * o + or ];
            if (i >= ghosts && i < m - ghosts && j >= n - ghosts) // bottom middle
                v[index] = v[i * n * o + nl * o + or ];

            if (i >= m - ghosts && j < ghosts) // top right
                v[index] = v[ml * n * o + nr * o + or ];
            if (i >= m - ghosts && j >= ghosts && j < n - ghosts) // middle right
                v[index] = v[ml * n * o + j * o + or ];
            if (i >= m - ghosts && j >= n - ghosts) // bottom right
                v[index] = v[ml * n * o + nl * o + or ];
        }

        // xy-plane behind grid
        if (k >= o - ghosts)
        {
            if (i < ghosts && j < ghosts) // top left
                v[index] = v[mr * n * o + nr * o + ol];
            if (i < ghosts && j >= ghosts && j < n - ghosts) // middle left
                v[index] = v[mr * n * o + j * o + ol];
            if (i < ghosts && j >= n - ghosts) // bottom left
                v[index] = v[mr * n * o + nl * o + ol];

            if (i >= ghosts && i < m - ghosts && j >= ghosts && j < n - ghosts)
                v[index] = v[i * n * o + j * o + ol];        // center middle
            if (i >= ghosts && i < n - ghosts && j < ghosts) // top middle
                v[index] = v[i * n * o + nr * o + ol];
            if (i >= ghosts && i < n - ghosts && j >= n - ghosts) // bottom middle
                v[index] = v[i * n * o + nl * o + ol];

            if (i >= m - ghosts && j < ghosts) // top right
                v[index] = v[ml * n * o + nr * o + ol];
            if (i >= m - ghosts && j >= ghosts && j < n - ghosts) // middle right
                v[index] = v[ml * n * o + j * o + ol];
            if (i >= m - ghosts && j >= n - ghosts) // bottom right
                v[index] = v[ml * n * o + nl * o + ol];
        }
    }
}

/* Updates ghost cells. m,n,o must be size of ghosted grid
 * One cell per work-item */
__kernel void update_ghosts_2d(global double *v, const int m, const int n, const int o, const int ghosts_m,
                               const int ghosts_n, const int ghosts_o)
{
    int j = get_global_id(0);
    int k = get_global_id(1);
    // int k = blockIdx.z*blockDim.z + threadIdx.z;
    // printf("i,j,k = %d %d %d\n", i,j,k);

    if (j < n && k < o)
    {
        int indexghosts = j * o + k;
        int ioff = o * n;

        int wm = m - 2 * ghosts_m;

        int nr = j + (n - 2 * ghosts_n);
        int orr = k + (o - 2 * ghosts_o);

        int nl = j - (n - 2 * ghosts_n);
        int ol = k - (o - 2 * ghosts_o);

        for (int i = 0; i < m; i++)
        {

            // indices of right real cells per left ghost cell
            int mr = i + wm;

            // indices of left real cells per right ghost cell
            int ml = i - wm;

            // update ghost cells for periodic boundary condition
            // make sure that every cell gets written only once to avoid race condition by using if-statements.
            // cells are divided by xy-planes (behind, self and in front of grid)
            // same xy-plane as grid
            if (k >= ghosts_o && k < o - ghosts_o)
            {
                if (i < ghosts_m && j >= ghosts_n && j < n - ghosts_n) // middle left
                    v[indexghosts] = v[mr * n * o + j * o + k];
                if (i >= m - ghosts_m && j >= ghosts_n && j < n - ghosts_n) // middle right
                    v[indexghosts] = v[ml * n * o + j * o + k];
                if (j < ghosts_n && i >= ghosts_m && i < m - ghosts_m) // top middle
                    v[indexghosts] = v[i * n * o + nr * o + k];
                if (j >= n - ghosts_n && i >= ghosts_m && i < m - ghosts_m) // bottom middle
                    v[indexghosts] = v[i * n * o + nl * o + k];
                if (i < ghosts_m && j < ghosts_n) // top left
                    v[indexghosts] = v[mr * n * o + nr * o + k];
                if (i >= m - ghosts_m && j < ghosts_n) // top right
                    v[indexghosts] = v[ml * n * o + nr * o + k];
                if (i < ghosts_m && j >= n - ghosts_n) // bottom left
                    v[indexghosts] = v[mr * n * o + nl * o + k];
                if (i >= m - ghosts_m && j >= n - ghosts_n) // bottom right
                    v[indexghosts] = v[ml * n * o + nl * o + k];
            }

            // xy-plane in front of grid
            if (k < ghosts_o)
            {
                if (i < ghosts_m && j < ghosts_n) // top left
                    v[indexghosts] = v[mr * n * o + nr * o + orr];
                if (i < ghosts_m && j >= ghosts_n && j < n - ghosts_n) // middle left
                    v[indexghosts] = v[mr * n * o + j * o + orr];
                if (i < ghosts_m && j >= n - ghosts_n) // bottom left
                    v[indexghosts] = v[mr * n * o + nl * o + orr];

                if (i >= ghosts_m && i < m - ghosts_m && j >= ghosts_n && j < n - ghosts_n)
                    v[indexghosts] = v[i * n * o + j * o + orr];       // center middle
                if (i >= ghosts_m && i < m - ghosts_m && j < ghosts_n) // top middle
                    v[indexghosts] = v[i * n * o + nr * o + orr];
                if (i >= ghosts_m && i < m - ghosts_m && j >= n - ghosts_n) // bottom middle
                    v[indexghosts] = v[i * n * o + nl * o + orr];

                if (i >= m - ghosts_m && j < ghosts_n) // top right
                    v[indexghosts] = v[ml * n * o + nr * o + orr];
                if (i >= m - ghosts_m && j >= ghosts_n && j < n - ghosts_n) // middle right
                    v[indexghosts] = v[ml * n * o + j * o + orr];
                if (i >= m - ghosts_m && j >= n - ghosts_n) // bottom right
                    v[indexghosts] = v[ml * n * o + nl * o + orr];
            }

            // xy-plane behind grid
            if (k >= o - ghosts_o)
            {
                if (i < ghosts_m && j < ghosts_n) // top left
                    v[indexghosts] = v[mr * n * o + nr * o + ol];
                if (i < ghosts_m && j >= ghosts_n && j < n - ghosts_n) // middle left
                    v[indexghosts] = v[mr * n * o + j * o + ol];
                if (i < ghosts_m && j >= n - ghosts_n) // bottom left
                    v[indexghosts] = v[mr * n * o + nl * o + ol];

                if (i >= ghosts_m && i < m - ghosts_m && j >= ghosts_n && j < n - ghosts_n)
                    v[indexghosts] = v[i * n * o + j * o + ol];        // center middle
                if (i >= ghosts_m && i < n - ghosts_m && j < ghosts_n) // top middle
                    v[indexghosts] = v[i * n * o + nr * o + ol];
                if (i >= ghosts_m && i < n - ghosts_m && j >= n - ghosts_n) // bottom middle
                    v[indexghosts] = v[i * n * o + nl * o + ol];

                if (i >= m - ghosts_m && j < ghosts_n) // top right
                    v[indexghosts] = v[ml * n * o + nr * o + ol];
                if (i >= m - ghosts_m && j >= ghosts_n && j < n - ghosts_n) // middle right
                    v[indexghosts] = v[ml * n * o + j * o + ol];
                if (i >= m - ghosts_m && j >= n - ghosts_n) // bottom right
                    v[indexghosts] = v[ml * n * o + nl * o + ol];
            }

            indexghosts += ioff;
        }
    }
}

/* Copies data from v_input to v_in and from f_input to f, respecting nearfield ghost cell count.
 * m, n and o are dimensions of mgcl's ghosted grid, thus sizes of v_in and f.
 * ghosts_in is ghost cell count in one direction of nearfield. */
__kernel void copy_input_data(__global double *v_input, __global double *v_in, __global double *f_input,
                              __global double *f, const int m, const int n, const int o, const int ghosts_in)
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
__kernel void copy_output_data(__global double *v_output, __global double *v_in, const int m, const int n, const int o,
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

/* Corrects error by adding e to v */
__kernel void correct_error(__global double *restrict v, __global double *restrict e, const int m, const int n,
                            const int o, const int ghosts)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);
    int idx = i * n * o + j * o + k;

    // only for real cells
    if (i > ghosts - 1 && j > ghosts - 1 && k > ghosts - 1 && i < m - ghosts && j < n - ghosts && k < o - ghosts)
    {
        v[idx] += e[idx];
    }
}

/* Calculates residual without dinv */
__kernel void residual_7point(
    __global double *restrict v_in, // needed s.t. every work-item can read surrounding cell values
    __global double *restrict f, __global double *restrict r, const double h2inv, const int m, const int n, const int o,
    const int ghosts)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);

    // calculate residual only for real cells since ghost cells do not have further ghost cells for themselves
    if (i > ghosts - 1 && j > ghosts - 1 && k > ghosts - 1 && i < m - ghosts && j < n - ghosts && k < o - ghosts)
    {
        int index = i * n * o + j * o + k;

        // A*v
        double stencilsum = (6.0 * v_in[index] - v_in[index - 1] - v_in[index + 1] - v_in[i * n * o + (j - 1) * o + k] -
                             v_in[i * n * o + (j + 1) * o + k] - v_in[(i - 1) * n * o + j * o + k] -
                             v_in[(i + 1) * n * o + j * o + k]) *
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
    __global double *restrict v_in, // needed s.t. every work-item can read surrounding cell values
    __global double *restrict f, __global double *restrict r, const double h2inv, const int m, const int n, const int o,
    const int ghosts)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);

    // calculate residual only for real cells since ghost cells do not have further ghost cells for themselves
    if (i > ghosts - 1 && j > ghosts - 1 && k > ghosts - 1 && i < m - ghosts && j < n - ghosts && k < o - ghosts)
    {
        int index = i * n * o + j * o + k;

        // A*v
        double stencilsum = (24.0 * v_in[index] - 2.0 * v_in[index - 1] - 2.0 * v_in[index + 1] -
                             2.0 * v_in[i * n * o + (j - 1) * o + k] - 2.0 * v_in[i * n * o + (j + 1) * o + k] -
                             2.0 * v_in[(i - 1) * n * o + j * o + k] - 2.0 * v_in[(i + 1) * n * o + j * o + k] -
                             v_in[i * n * o + (j - 1) * o + k - 1] - v_in[i * n * o + (j - 1) * o + k + 1] -
                             v_in[i * n * o + (j + 1) * o + k - 1] - v_in[i * n * o + (j + 1) * o + k + 1] -
                             v_in[(i - 1) * n * o + j * o + k - 1] - v_in[(i - 1) * n * o + j * o + k + 1] -
                             v_in[(i + 1) * n * o + j * o + k - 1] - v_in[(i + 1) * n * o + j * o + k + 1] -
                             v_in[(i - 1) * n * o + (j - 1) * o + k] - v_in[(i - 1) * n * o + (j + 1) * o + k] -
                             v_in[(i + 1) * n * o + (j - 1) * o + k] - v_in[(i + 1) * n * o + (j + 1) * o + k]) *
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
    __global double *restrict v_in, // needed s.t. every work-item can read surrounding cell values
    __global double *restrict f, __global double *restrict r, const double h2inv, const int m, const int n, const int o,
    const int ghosts)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);

    // calculate residual only for real cells since ghost cells do not have further ghost cells for themselves
    if (i > ghosts - 1 && j > ghosts - 1 && k > ghosts - 1 && i < m - ghosts && j < n - ghosts && k < o - ghosts)
    {
        int index = i * n * o + j * o + k;
        int ioff = n * o;
        int joff = o;
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

/* Calculates the squares of the residual.
 * TODO use local memory and do sum reduction */
__kernel void residual_squared(__global double *restrict r, __global double *restrict rsquares, const int m,
                               const int n, const int o, const int ghosts)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);

    // only for real cells
    if (i > ghosts - 1 && j > ghosts - 1 && k > ghosts - 1 && i < m - ghosts && j < n - ghosts && k < o - ghosts)
    {
        int index = i * n * o + j * o + k;
        double ridx = r[index];
        rsquares[index] = ridx * ridx;
    }
}

/* runs one iteration of jacobi's method using one work-item per row.
 * uses a 2D kernel which loops over cells in x-direction. y and z is parallelized.
 * global size must be of ghosted grid.
 * m, n and o must be dimensions of ghosted grid, too.
 * h2 is grid spacing to the power of 2
 * dinv is h2/A(i,i), e.g. h2/6.0 for 3D laplacian stencil
 * if store_residual is true, the residual will be stored into global field r */
__kernel void jacobi_iter_7point(
    __global double *restrict v_in, // needed s.t. every work-item can read surrounding cell values
    __global double *restrict v_out, __global double *restrict f, __global double *restrict r, const double h2inv,
    const double dinv, const double omega, const int m, const int n, const int o, const int ghosts,
    const int store_residual)
{
    int j = get_global_id(0);
    int k = get_global_id(1);

    // calculate residual only for real cells since ghost cells do not have further ghost cells for themselves
    if (j > ghosts - 1 && k > ghosts - 1 && j < n - ghosts && k < o - ghosts)
    {
        int ioff = n * o;
        int joff = o;
        int koff = 1;
        int index = ghosts * ioff + j * o + k;

        for (int i = ghosts; i < m - ghosts; i++)
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

            // if (get_global_id(0) == ghosts && get_global_id(1) == ghosts && i >= ghosts && i <= ghosts+2)
            // {
            //     printf("x = %d, res = %e, v_out = %e\n", i, res, v_out[index]);
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
 * if store_residual is true, the residual will be stored into global field r */
__kernel void jacobi_iter_19point(
    __global double *restrict v_in, // needed s.t. every work-item can read surrounding cell values
    __global double *restrict v_out, __global double *restrict f, __global double *restrict r, const double h2inv,
    const double dinv, const double omega, const int m, const int n, const int o, const int ghosts,
    const int store_residual)
{
    int j = get_global_id(0);
    int k = get_global_id(1);

    // calculate residual only for real cells since ghost cells do not have further ghost cells for themselves
    if (j > ghosts - 1 && k > ghosts - 1 && j < n - ghosts && k < o - ghosts)
    {
        int ioff = n * o;
        int joff = o;
        int koff = 1;
        int index = ghosts * ioff + j * o + k;

        for (int i = ghosts; i < m - ghosts; i++)
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
 * h2 is grid spacing to the power of 2
 * dinv is h2/A(i,i), e.g. h2/6.0 for 3D laplacian stencil
 * if store_residual is true, the residual will be stored into global field r */
__kernel void jacobi_iter_27point(
    __global double *restrict v_in, // needed s.t. every work-item can read surrounding cell values
    __global double *restrict v_out, __global double *restrict f, __global double *restrict r, const double h2inv,
    const double dinv, const double omega, const int m, const int n, const int o, const int ghosts,
    const int store_residual)
{
    int j = get_global_id(0);
    int k = get_global_id(1);

    // calculate residual only for real cells since ghost cells do not have further ghost cells for themselves
    if (j > ghosts - 1 && k > ghosts - 1 && j < n - ghosts && k < o - ghosts)
    {
        int ioff = n * o;
        int joff = o;
        int koff = 1;
        int index = ghosts * ioff + j * o + k;

        for (int i = ghosts; i < m - ghosts; i++)
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

/* Restricts from fine to coarse grid.
 * Needs to get called with m*n*o work-items.
 * m,n,o is size of ghosted coarse grid.
 * fine and coarse must be of sizes of ghosted grids */
__kernel void restrict_to_coarse(__global double *restrict fine, __global double *restrict coarse, const int m,
                                 const int n, const int o, const int ghosts)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);
    int g2 = 2 * ghosts;

    if (i < m && j < n && k < o)
    {
        const int index = (i + ghosts) * n * o + (j + ghosts) * o + (k + ghosts);
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
 * fine and coarse must be of sizes of ghosted grids */
__kernel void prolongate_to_fine(__global double *restrict fine, __global double *restrict coarse, const int m,
                                 const int n, const int o, const int ghosts)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);
    int g2 = 2 * ghosts;

    const int mc = (m - g2) / 2 + g2, nc = (n - g2) / 2 + g2, oc = (o - g2) / 2 + g2;

    if (i > ghosts - 1 && i < mc - ghosts && j > ghosts - 1 && j < nc - ghosts && k > ghosts - 1 && k < oc - ghosts)
    {
        const int index_coarse = i * nc * oc + j * oc + k;
        const int i2 = i * 2 - (ghosts - 1), j2 = j * 2 - (ghosts - 1), k2 = k * 2 - (ghosts - 1);

        fine[i2 * n * o + j2 * o + k2] = coarse[index_coarse];

        fine[i2 * n * o + j2 * o + k2 - 1] = 0.5 * (coarse[index_coarse] + coarse[index_coarse - 1]);
        fine[i2 * n * o + (j2 - 1) * o + k2] = 0.5 * (coarse[index_coarse] + coarse[i * nc * oc + (j - 1) * oc + k]);
        fine[(i2 - 1) * n * o + j2 * o + k2] = 0.5 * (coarse[index_coarse] + coarse[(i - 1) * nc * oc + j * oc + k]);

        fine[i2 * n * o + (j2 - 1) * o + k2 - 1] =
            0.25 * (coarse[index_coarse] + coarse[index_coarse - 1] + coarse[i * nc * oc + (j - 1) * oc + k] +
                    coarse[i * nc * oc + (j - 1) * oc + k - 1]);
        fine[(i2 - 1) * n * o + j2 * o + k2 - 1] =
            0.25 * (coarse[index_coarse] + coarse[index_coarse - 1] + coarse[(i - 1) * nc * oc + j * oc + k] +
                    coarse[(i - 1) * nc * oc + j * oc + k - 1]);
        fine[(i2 - 1) * n * o + (j2 - 1) * o + k2] =
            0.25 * (coarse[index_coarse] + coarse[i * nc * oc + (j - 1) * oc + k] +
                    coarse[(i - 1) * nc * oc + j * oc + k] + coarse[(i - 1) * nc * oc + (j - 1) * oc + k]);

        fine[(i2 - 1) * n * o + (j2 - 1) * o + k2 - 1] =
            0.125 * (coarse[index_coarse] + coarse[index_coarse - 1] + coarse[i * nc * oc + (j - 1) * oc + k] +
                     coarse[i * nc * oc + (j - 1) * oc + k - 1] + coarse[(i - 1) * nc * oc + j * oc + k] +
                     coarse[(i - 1) * nc * oc + j * oc + k - 1] + coarse[(i - 1) * nc * oc + (j - 1) * oc + k] +
                     coarse[(i - 1) * nc * oc + (j - 1) * oc + k - 1]);
    }
}

/* Loads a plane from global buffer src into target buffer dest.
 * Also loads outmost ghosted border.
 * joff is the offset in j direction, e.g. plane's x-size */
void load_plane(__global double *restrict src, __local double *restrict dest, int index_src, int index_dest, int joff,
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
void load_plane_diag(__global double *restrict src, __local double *restrict dest, int index_src, int index_dest,
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
    __global double *restrict v_in, // needed s.t. every work-item can read surrounding cell values
    __global double *restrict v_out, __global double *restrict f, __global double *restrict r, __local double *vlocal,
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
        __local double *A = vlocal;
        __local double *rp = vlocal + blockSizeEx * iterations;
        __local double *rm =
            vlocal + blockSizeEx * 2 * iterations; // (time steps + 1) * (block (wg) size + 2) = 4*14*14
        __local double *tmp;

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
    __global double *restrict v_in, // needed s.t. every work-item can read surrounding cell values
    __global double *restrict v_out, __global double *restrict f, __global double *restrict r, __local double *vlocal,
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
        __local double *A = vlocal;
        __local double *rp = vlocal + blockSizeEx * iterations;
        __local double *rm =
            vlocal + blockSizeEx * 2 * iterations; // (time steps + 1) * (block (wg) size + 2) = 4*14*14
        __local double *tmp;

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
    __global double *restrict v_in, // needed s.t. every work-item can read surrounding cell values
    __global double *restrict v_out, __global double *restrict f, __global double *restrict r, __local double *vlocal,
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
        __local double *A = vlocal;
        __local double *rp = vlocal + blockSizeEx * iterations;
        __local double *rm =
            vlocal + blockSizeEx * 2 * iterations; // (time steps + 1) * (block (wg) size + 2) = 4*14*14
        __local double *tmp;

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
