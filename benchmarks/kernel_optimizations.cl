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
    __global double *restrict a,
    __global double *restrict b,
    __global double *restrict c,
    int m, int n, int o,
    int wa, int wb,
    int gha, int ghb, int ghc)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);

    int wa2 = wa >> 1;
    int wc = wa + wb - 1;

    // 1d indices
    int wcPow2 = wc * wc;
    int wcPow3 = wcPow2 * wc;
    int cell_c = (i + ghc) * (n + 2 * ghc) * (o + 2 * ghc) * wcPow3 + (j + ghc) * (o + 2 * ghc) * wcPow3 + (k + ghc) * wcPow3;

    int waPow2 = wa * wa;
    int waPow3 = waPow2 * wa;
    int cell_a = (i + gha) * (n + 2 * gha) * (o + 2 * gha) * waPow3 + (j + gha) * (o + 2 * gha) * waPow3 + (k + gha) * waPow3;

    int wbPow2 = wb * wb;
    int wbPow3 = wbPow2 * wb;

    if (i < m && j < n && k < o)
    {
        // clang-format off
        for (int a_i = 0; a_i < wa; a_i++)
        for (int a_j = 0; a_j < wa; a_j++)
        for (int a_k = 0; a_k < wa; a_k++)
            for (int b_i = 0; b_i < wb; b_i++)
            for (int b_j = 0; b_j < wb; b_j++)
            for (int b_k = 0; b_k < wb; b_k++)
            {
                int gpi = i + a_i - wa2 + ghb;
                int gpj = j + a_j - wa2 + ghb;
                int gpk = k + a_k - wa2 + ghb;

                int cell_b = gpi * (n + 2 * ghb) * (o + 2 * ghb) * wbPow3 + gpj * (o + 2 * ghb) * wbPow3 + gpk * wbPow3;

                int ci = a_i + b_i;
                int cj = a_j + b_j;
                int ck = a_k + b_k;

                if (ci >= 0 && ci < wc &&
                    cj >= 0 && cj < wc &&
                    ck >= 0 && ck < wc)
                {
                    c[cell_c + ci * wcPow2 + cj * wc + ck] +=
                        a[cell_a + a_i * waPow2 + a_j * wa + a_k] *
                        b[cell_b + b_i * wbPow2 + b_j * wb + b_k];

                    // c[i + ghc][j + ghc][k + ghc][a_i + b_i][a_j + b_j][a_k + b_k] +=
                    //     a[i + gha][j + gha][k + gha][a_i][a_j][a_k] *
                    //     b[gpi][gpj][gpk][b_i][b_j][b_k];
                }
            }
        // clang-format on
    }
}

// this version replaces the if-statement inside the loop by just multiplying the boolean result with the inner calculation.
__kernel void mult_stencils_var_var_branchless(
    __global double *restrict a,
    __global double *restrict b,
    __global double *restrict c,
    int m, int n, int o,
    int wa, int wb,
    int gha, int ghb, int ghc)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);

    int wa2 = wa >> 1;
    int wc = wa + wb - 1;

    // 1d indices
    int wcPow2 = wc * wc;
    int wcPow3 = wcPow2 * wc;
    int cell_c = (i + ghc) * (n + 2 * ghc) * (o + 2 * ghc) * wcPow3 + (j + ghc) * (o + 2 * ghc) * wcPow3 + (k + ghc) * wcPow3;

    int waPow2 = wa * wa;
    int waPow3 = waPow2 * wa;
    int cell_a = (i + gha) * (n + 2 * gha) * (o + 2 * gha) * waPow3 + (j + gha) * (o + 2 * gha) * waPow3 + (k + gha) * waPow3;

    int wbPow2 = wb * wb;
    int wbPow3 = wbPow2 * wb;

    if (i < m && j < n && k < o)
    {
        // clang-format off
        for (int a_i = 0; a_i < wa; a_i++)
        for (int a_j = 0; a_j < wa; a_j++)
        for (int a_k = 0; a_k < wa; a_k++)
            for (int b_i = 0; b_i < wb; b_i++)
            for (int b_j = 0; b_j < wb; b_j++)
            for (int b_k = 0; b_k < wb; b_k++)
            {
                int gpi = i + a_i - wa2 + ghb;
                int gpj = j + a_j - wa2 + ghb;
                int gpk = k + a_k - wa2 + ghb;

                int cell_b = gpi * (n + 2 * ghb) * (o + 2 * ghb) * wbPow3 + gpj * (o + 2 * ghb) * wbPow3 + gpk * wbPow3;

                int ci = a_i + b_i;
                int cj = a_j + b_j;
                int ck = a_k + b_k;

                c[cell_c + ci * wcPow2 + cj * wc + ck] += 
                    (ci >= 0 && ci < wc &&
                    cj >= 0 && cj < wc &&
                    ck >= 0 && ck < wc) *
                    a[cell_a + a_i * waPow2 + a_j * wa + a_k] *
                    b[cell_b + b_i * wbPow2 + b_j * wb + b_k];

                // c[i + ghc][j + ghc][k + ghc][a_i + b_i][a_j + b_j][a_k + b_k] +=
                //     a[i + gha][j + gha][k + gha][a_i][a_j][a_k] *
                //     b[gpi][gpj][gpk][b_i][b_j][b_k];
            }
        // clang-format on
    }
}

// This version has a reordered foor loop structure s.t. intermediate results for c can be stored in a private variable and written
// to the global buffer later only once.
__kernel void mult_stencils_var_var_reordered(
    __global double *restrict a,
    __global double *restrict b,
    __global double *restrict c,
    int m, int n, int o,
    int wa, int wb,
    int gha, int ghb, int ghc)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);

    int wa2 = wa >> 1;
    int wc = wa + wb - 1;

    // 1d indices
    int wcPow2 = wc * wc;
    int wcPow3 = wcPow2 * wc;
    int cell_c = (i + ghc) * (n + 2 * ghc) * (o + 2 * ghc) * wcPow3 + (j + ghc) * (o + 2 * ghc) * wcPow3 + (k + ghc) * wcPow3;

    int waPow2 = wa * wa;
    int waPow3 = waPow2 * wa;
    int cell_a = (i + gha) * (n + 2 * gha) * (o + 2 * gha) * waPow3 + (j + gha) * (o + 2 * gha) * waPow3 + (k + gha) * waPow3;

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
            for (int a_i = ci - (ci < (wb - 1) ? ci : (wb - 1)), b_i = (ci < (wb - 1) ? ci : (wb - 1));
             a_i <= (ci < (wa - 1) ? ci : (wa - 1)) && b_i >= ci - (ci < (wa - 1) ? ci : (wa - 1)); 
             a_i++, b_i--)
            for (int a_j = cj - (cj < (wb - 1) ? cj : (wb - 1)), b_j = (cj < (wb - 1) ? cj : (wb - 1));
                a_j <= (cj < (wa - 1) ? cj : (wa - 1)) && b_j >= cj - (cj < (wa - 1) ? cj : (wa - 1)); 
                a_j++, b_j--)
            for (int a_k = ck - (ck < (wb - 1) ? ck : (wb - 1)), b_k = (ck < (wb - 1) ? ck : (wb - 1));
                a_k <= (ck < (wa - 1) ? ck : (wa - 1)) && b_k >= ck - (ck < (wa - 1) ? ck : (wa - 1)); 
                a_k++, b_k--)
            {
                int gpi = i + a_i - wa2 + ghb;
                int gpj = j + a_j - wa2 + ghb;
                int gpk = k + a_k - wa2 + ghb;

                int cell_b = gpi * (n + 2 * ghb) * (o + 2 * ghb) * wbPow3 + gpj * (o + 2 * ghb) * wbPow3 + gpk * wbPow3;

                csum +=
                    a[cell_a + a_i * waPow2 + a_j * wa + a_k] *
                    b[cell_b + b_i * wbPow2 + b_j * wb + b_k];
            }

            c[cell_c + ci * wcPow2 + cj * wc + ck] = csum;
        }
        // clang-format on
    }
}

// Same as mult_stencils_var_var_reordered but using min function instead of ternary operators.
__kernel void mult_stencils_var_var_reordered_minfn(
    __global double *restrict a,
    __global double *restrict b,
    __global double *restrict c,
    int m, int n, int o,
    int wa, int wb,
    int gha, int ghb, int ghc)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);

    int wa2 = wa >> 1;
    int wc = wa + wb - 1;

    // 1d indices
    int wcPow2 = wc * wc;
    int wcPow3 = wcPow2 * wc;
    int cell_c = (i + ghc) * (n + 2 * ghc) * (o + 2 * ghc) * wcPow3 + (j + ghc) * (o + 2 * ghc) * wcPow3 + (k + ghc) * wcPow3;

    int waPow2 = wa * wa;
    int waPow3 = waPow2 * wa;
    int cell_a = (i + gha) * (n + 2 * gha) * (o + 2 * gha) * waPow3 + (j + gha) * (o + 2 * gha) * waPow3 + (k + gha) * waPow3;

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

                int cell_b = gpi * (n + 2 * ghb) * (o + 2 * ghb) * wbPow3 + gpj * (o + 2 * ghb) * wbPow3 + gpk * wbPow3;

                csum +=
                    a[cell_a + a_i * waPow2 + a_j * wa + a_k] *
                    b[cell_b + b_i * wbPow2 + b_j * wb + b_k];
            }

            c[cell_c + ci * wcPow2 + cj * wc + ck] = csum;
        }
        // clang-format on
    }
}

// Same as mult_stencils_var_var_reordered_minfn but using smaller data types, too.
__kernel void mult_stencils_var_var_reordered_minfn_small_types(
    __global double *restrict a,
    __global double *restrict b,
    __global double *restrict c,
    ushort m, ushort n, ushort o,
    uchar wa, uchar wb,
    uchar gha, uchar ghb, uchar ghc)
{
    ushort i = get_global_id(0);
    ushort j = get_global_id(1);
    ushort k = get_global_id(2);

    uchar wa2 = wa >> 1;
    uchar wc = wa + wb - 1;

    // 1d indices
    ushort wcPow2 = wc * wc;
    ushort wcPow3 = wcPow2 * wc;
    int cell_c = (i + ghc) * (n + 2 * ghc) * (o + 2 * ghc) * wcPow3 + (j + ghc) * (o + 2 * ghc) * wcPow3 + (k + ghc) * wcPow3;

    ushort waPow2 = wa * wa;
    ushort waPow3 = waPow2 * wa;
    int cell_a = (i + gha) * (n + 2 * gha) * (o + 2 * gha) * waPow3 + (j + gha) * (o + 2 * gha) * waPow3 + (k + gha) * waPow3;

    ushort wbPow2 = wb * wb;
    ushort wbPow3 = wbPow2 * wb;

    if (i < m && j < n && k < o)
    {
        // clang-format off
        for (int ci = 0; ci < wc; ci++)
        for (int cj = 0; cj < wc; cj++)
        for (int ck = 0; ck < wc; ck++)
        {
            double csum = 0;
            for (uchar a_i = ci - (min(ci, wb - 1)), b_i = min(ci, wb - 1);
                a_i <= min(ci, wa - 1) && b_i >= ci - min(ci, wa - 1);
                a_i++, b_i--)
            for (uchar a_j = cj - (min(cj, wb - 1)), b_j = min(cj, wb - 1);
                    a_j <= min(cj, wa - 1) && b_j >= cj - min(cj, wa - 1);
                    a_j++, b_j--)
            for (uchar a_k = ck - (min(ck, wb - 1)), b_k = min(ck, wb - 1);
                    a_k <= min(ck, wa - 1) && b_k >= ck - min(ck, wa - 1);
                    a_k++, b_k--)
            {
                ushort gpi = i + a_i - wa2 + ghb;
                ushort gpj = j + a_j - wa2 + ghb;
                ushort gpk = k + a_k - wa2 + ghb;

                int cell_b = gpi * (n + 2 * ghb) * (o + 2 * ghb) * wbPow3 + gpj * (o + 2 * ghb) * wbPow3 + gpk * wbPow3;

                csum +=
                    a[cell_a + a_i * waPow2 + a_j * wa + a_k] *
                    b[cell_b + b_i * wbPow2 + b_j * wb + b_k];
            }

            c[cell_c + ci * wcPow2 + cj * wc + ck] = csum;
        }
        // clang-format on
    }
}

// Same as mult_stencils_var_var_reordered_minfn but with const specifiers.
__kernel void mult_stencils_var_var_reordered_minfn_consts(
    __global double *restrict a,
    __global double *restrict b,
    __global double *restrict c,
    const int m, const int n, const int o,
    const int wa, const int wb,
    const int gha, const int ghb, const int ghc)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);
    const int k = get_global_id(2);

    const int wa2 = wa >> 1;
    const int wc = wa + wb - 1;

    // 1d indices
    const int wcPow2 = wc * wc;
    const int wcPow3 = wcPow2 * wc;
    const int cell_c = (i + ghc) * (n + 2 * ghc) * (o + 2 * ghc) * wcPow3 + (j + ghc) * (o + 2 * ghc) * wcPow3 + (k + ghc) * wcPow3;

    const int waPow2 = wa * wa;
    const int waPow3 = waPow2 * wa;
    const int cell_a = (i + gha) * (n + 2 * gha) * (o + 2 * gha) * waPow3 + (j + gha) * (o + 2 * gha) * waPow3 + (k + gha) * waPow3;

    const int wbPow2 = wb * wb;
    const int wbPow3 = wbPow2 * wb;

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

                int cell_b = gpi * (n + 2 * ghb) * (o + 2 * ghb) * wbPow3 + gpj * (o + 2 * ghb) * wbPow3 + gpk * wbPow3;

                csum +=
                    a[cell_a + a_i * waPow2 + a_j * wa + a_k] *
                    b[cell_b + b_i * wbPow2 + b_j * wb + b_k];
            }

            c[cell_c + ci * wcPow2 + cj * wc + ck] = csum;
        }
        // clang-format on
    }
}

// Same as mult_stencils_var_var_reordered_minfn but started as a 1d kernel.
__kernel void mult_stencils_var_var_reordered_minfn_1d(
    __global double *restrict a,
    __global double *restrict b,
    __global double *restrict c,
    int m, int n, int o,
    int wa, int wb,
    int gha, int ghb, int ghc)
{

    const int winum = get_global_id(0);
    int i = winum / (n * o);
    int j = (winum / o) % n;
    int k = winum % o;

    int wa2 = wa >> 1;
    int wc = wa + wb - 1;

    // 1d indices
    int wcPow2 = wc * wc;
    int wcPow3 = wcPow2 * wc;
    int cell_c = (i + ghc) * (n + 2 * ghc) * (o + 2 * ghc) * wcPow3 + (j + ghc) * (o + 2 * ghc) * wcPow3 + (k + ghc) * wcPow3;

    int waPow2 = wa * wa;
    int waPow3 = waPow2 * wa;
    int cell_a = (i + gha) * (n + 2 * gha) * (o + 2 * gha) * waPow3 + (j + gha) * (o + 2 * gha) * waPow3 + (k + gha) * waPow3;

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

                int cell_b = gpi * (n + 2 * ghb) * (o + 2 * ghb) * wbPow3 + gpj * (o + 2 * ghb) * wbPow3 + gpk * wbPow3;

                csum +=
                    a[cell_a + a_i * waPow2 + a_j * wa + a_k] *
                    b[cell_b + b_i * wbPow2 + b_j * wb + b_k];
            }

            c[cell_c + ci * wcPow2 + cj * wc + ck] = csum;
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
 * This kernel is supposed to be launched with one work-item per grid cell.
 * Remember to update ghosts of c if ghc > 0 afterwards.
 */
__kernel void mult_stencils_var_fix(
    __global double *restrict a,
    __global double *restrict b,
    __global double *restrict c,
    int m, int n, int o,
    int wa, int wb,
    int gha, int ghc)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);

    int wa2 = wa >> 1;
    int wc = wa + wb - 1;

    // 1d indices
    int wcPow2 = wc * wc;
    int wcPow3 = wcPow2 * wc;
    int cell_c = (i + ghc) * (n + 2 * ghc) * (o + 2 * ghc) * wcPow3 + (j + ghc) * (o + 2 * ghc) * wcPow3 + (k + ghc) * wcPow3;

    int waPow2 = wa * wa;
    int waPow3 = waPow2 * wa;
    int cell_a = (i + gha) * (n + 2 * gha) * (o + 2 * gha) * waPow3 + (j + gha) * (o + 2 * gha) * waPow3 + (k + gha) * waPow3;

    int wbPow2 = wb * wb;
    int wbPow3 = wbPow2 * wb;

    if (i < m && j < n && k < o)
    {
        // TODO temporary values in private array

        // clang-format off
        for (int a_i = 0; a_i < wa; a_i++)
        for (int a_j = 0; a_j < wa; a_j++)
        for (int a_k = 0; a_k < wa; a_k++)
            for (int b_i = 0; b_i < wb; b_i++)
            for (int b_j = 0; b_j < wb; b_j++)
            for (int b_k = 0; b_k < wb; b_k++)
            {
                int ci = a_i + b_i;
                int cj = a_j + b_j;
                int ck = a_k + b_k;

                if (ci >= 0 && ci < wc &&
                    cj >= 0 && cj < wc &&
                    ck >= 0 && ck < wc)
                {
                    c[cell_c + ci * wcPow2 + cj * wc + ck] +=
                        a[cell_a + a_i * waPow2 + a_j * wa + a_k] *
                        b[b_i * wbPow2 + b_j * wb + b_k];

                    // c[i + ghc][j + ghc][k + ghc][a_i + b_i][a_j + b_j][a_k + b_k] +=
                    //     a[i + gha][j + gha][k + gha][a_i][a_j][a_k] *
                    //     b[gpi][gpj][gpk][b_i][b_j][b_k];
                }
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
 * This kernel is supposed to be launched with one work-item per grid cell.
 * Remember to update ghosts of c if ghc > 0 afterwards.
 */
__kernel void mult_stencils_var_fix_reordered(
    __global double *restrict a,
    __global double *restrict b,
    __global double *restrict c,
    int m, int n, int o,
    int wa, int wb,
    int gha, int ghc)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);

    int wa2 = wa >> 1;
    int wc = wa + wb - 1;

    // 1d indices
    int wcPow2 = wc * wc;
    int wcPow3 = wcPow2 * wc;
    int cell_c = (i + ghc) * (n + 2 * ghc) * (o + 2 * ghc) * wcPow3 + (j + ghc) * (o + 2 * ghc) * wcPow3 + (k + ghc) * wcPow3;

    int waPow2 = wa * wa;
    int waPow3 = waPow2 * wa;
    int cell_a = (i + gha) * (n + 2 * gha) * (o + 2 * gha) * waPow3 + (j + gha) * (o + 2 * gha) * waPow3 + (k + gha) * waPow3;

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
                csum +=
                    a[cell_a + a_i * waPow2 + a_j * wa + a_k] *
                    b[b_i * wbPow2 + b_j * wb + b_k];
            }

            c[cell_c + ci * wcPow2 + cj * wc + ck] = csum;
        }
        // clang-format on
    }
}

__kernel void mult_stencils_var_fix_reordered_constb(
    __global double *restrict a,
    __constant double *restrict b,
    __global double *restrict c,
    int m, int n, int o,
    int wa, int wb,
    int gha, int ghc)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);

    int wa2 = wa >> 1;
    int wc = wa + wb - 1;

    // 1d indices
    int wcPow2 = wc * wc;
    int wcPow3 = wcPow2 * wc;
    int cell_c = (i + ghc) * (n + 2 * ghc) * (o + 2 * ghc) * wcPow3 + (j + ghc) * (o + 2 * ghc) * wcPow3 + (k + ghc) * wcPow3;

    int waPow2 = wa * wa;
    int waPow3 = waPow2 * wa;
    int cell_a = (i + gha) * (n + 2 * gha) * (o + 2 * gha) * waPow3 + (j + gha) * (o + 2 * gha) * waPow3 + (k + gha) * waPow3;

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
                csum +=
                    a[cell_a + a_i * waPow2 + a_j * wa + a_k] *
                    b[b_i * wbPow2 + b_j * wb + b_k];
            }

            c[cell_c + ci * wcPow2 + cj * wc + ck] = csum;
        }
        // clang-format on
    }
}

// CAUTION: Probably does not yield correct results for small grids, only for experimental purposes!
__kernel void mult_stencils_var_fix_reordered_localb(
    __global double *restrict a,
    __global double *restrict b,
    __global double *restrict c,
    int m, int n, int o,
    int wa, int wb,
    int gha, int ghc)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);

    int wa2 = wa >> 1;
    int wc = wa + wb - 1;

    // 1d indices
    int wcPow2 = wc * wc;
    int wcPow3 = wcPow2 * wc;
    int cell_c = (i + ghc) * (n + 2 * ghc) * (o + 2 * ghc) * wcPow3 + (j + ghc) * (o + 2 * ghc) * wcPow3 + (k + ghc) * wcPow3;

    int waPow2 = wa * wa;
    int waPow3 = waPow2 * wa;
    int cell_a = (i + gha) * (n + 2 * gha) * (o + 2 * gha) * waPow3 + (j + gha) * (o + 2 * gha) * waPow3 + (k + gha) * waPow3;

    int wbPow2 = wb * wb;
    int wbPow3 = wbPow2 * wb;

    __local double b_loc[27];
    int locId = get_local_id(2);
    if (get_local_size(1) > 1)
        locId = get_local_id(1) * get_local_size(2) + get_local_id(2);

    if (locId < 27)
        b_loc[locId] = b[locId];

    barrier(CLK_LOCAL_MEM_FENCE);

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
                csum +=
                    a[cell_a + a_i * waPow2 + a_j * wa + a_k] *
                    b_loc[b_i * wbPow2 + b_j * wb + b_k];
            }

            c[cell_c + ci * wcPow2 + cj * wc + ck] = csum;
        }
        // clang-format on
    }
}

// this version has the widths of the stencils as inline numbers (e.g. replaced via string manipulation and recompiled)
// only for wa = wb = 3 for now
__kernel void mult_stencils_var_fix_reordered_widths_inline(
    __global double *restrict a,
    __global double *restrict b,
    __global double *restrict c,
    int m, int n, int o,
    int gha, int ghc)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);

    // 1d indices
    int cell_c = (i + ghc) * (n + 2 * ghc) * (o + 2 * ghc) * 125 + (j + ghc) * (o + 2 * ghc) * 125 + (k + ghc) * 125;
    int cell_a = (i + gha) * (n + 2 * gha) * (o + 2 * gha) * 27 + (j + gha) * (o + 2 * gha) * 27 + (k + gha) * 27;

    if (i < m && j < n && k < o)
    {
        // clang-format off
        for (int ci = 0; ci < 5; ci++)
        for (int cj = 0; cj < 5; cj++)
        for (int ck = 0; ck < 5; ck++)
        {
            double csum = 0;
            for (int a_i = ci - (min(ci, 2)), b_i = min(ci, 2);
                a_i <= min(ci, 2) && b_i >= ci - min(ci, 2);
                a_i++, b_i--)
            for (int a_j = cj - (min(cj, 2)), b_j = min(cj, 2);
                    a_j <= min(cj, 2) && b_j >= cj - min(cj, 2);
                    a_j++, b_j--)
            for (int a_k = ck - (min(ck, 2)), b_k = min(ck, 2);
                    a_k <= min(ck, 2) && b_k >= ck - min(ck, 2);
                    a_k++, b_k--)
            {
                csum +=
                    a[cell_a + a_i * 9 + a_j * 3 + a_k] *
                    b[b_i * 9 + b_j * 3 + b_k];
            }

            c[cell_c + ci * 25 + cj * 5 + ck] = csum;
        }
        // clang-format on
    }
}

// stores bounds of innermost loop in variables to reduce calls to min
__kernel void mult_stencils_var_fix_reordered_loop_bounds_preserved(
    __global double *restrict a,
    __global double *restrict b,
    __global double *restrict c,
    int m, int n, int o,
    int wa, int wb,
    int gha, int ghc)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);

    int wa2 = wa >> 1;
    int wc = wa + wb - 1;

    // 1d indices
    int wcPow2 = wc * wc;
    int wcPow3 = wcPow2 * wc;
    int cell_c = (i + ghc) * (n + 2 * ghc) * (o + 2 * ghc) * wcPow3 + (j + ghc) * (o + 2 * ghc) * wcPow3 + (k + ghc) * wcPow3;

    int waPow2 = wa * wa;
    int waPow3 = waPow2 * wa;
    int cell_a = (i + gha) * (n + 2 * gha) * (o + 2 * gha) * waPow3 + (j + gha) * (o + 2 * gha) * waPow3 + (k + gha) * waPow3;

    int wbPow2 = wb * wb;
    int wbPow3 = wbPow2 * wb;

    if (i < m && j < n && k < o)
    {
        // clang-format off
        for (int ci = 0; ci < wc; ci++)
        {
            int start_ai = ci - (min(ci, wb - 1));
            int start_bi = min(ci, wb - 1);
            int end_ai = min(ci, wa - 1);
            int end_bi = ci - min(ci, wa - 1);

            for (int cj = 0; cj < wc; cj++)
            {
                int start_aj = cj - (min(cj, wb - 1));
                int start_bj = min(cj, wb - 1);
                int end_aj = min(cj, wa - 1);
                int end_bj = cj - min(cj, wa - 1);

                for (int ck = 0; ck < wc; ck++)
                {
                    int start_ak = ck - (min(ck, wb - 1));
                    int start_bk = min(ck, wb - 1);
                    int end_ak = min(ck, wa - 1);
                    int end_bk = ck - min(ck, wa - 1);

                    double csum = 0;
                    for (int a_i = start_ai, b_i = start_bi;
                        a_i <= end_ai && b_i >= end_bi;
                        a_i++, b_i--)
                    for (int a_j = start_aj, b_j = start_bj;
                            a_j <= end_aj && b_j >= end_bj;
                            a_j++, b_j--)
                    for (int a_k = start_ak, b_k = start_bk;
                            a_k <= end_ak && b_k >= end_bk;
                            a_k++, b_k--)
                    {
                        csum +=
                            a[cell_a + a_i * waPow2 + a_j * wa + a_k] *
                            b[b_i * wbPow2 + b_j * wb + b_k];
                    }

                    c[cell_c + ci * wcPow2 + cj * wc + ck] = csum;
                }
            }
        }
        // clang-format on
    }
}

// reordered + parallel c loop. Must be called with m x n x o*wc*wc*wc work-items
__kernel void mult_stencils_var_fix_reordered_parallel_c(
    __global double *restrict a,
    __global double *restrict b,
    __global double *restrict c,
    int m, int n, int o,
    int wa, int wb,
    int gha, int ghc)
{
    int wc = wa + wb - 1;
    int wcPow2 = wc * wc;
    int wcPow3 = wcPow2 * wc;

    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2) / wcPow3;
    int ci = (get_global_id(2) / wcPow2) % wc;
    int cj = (get_global_id(2) / wc) % wc;
    int ck = get_global_id(2) % wc;

    // 1d indices
    int cell_c = (i + ghc) * (n + 2 * ghc) * (o + 2 * ghc) * wcPow3 + (j + ghc) * (o + 2 * ghc) * wcPow3 + (k + ghc) * wcPow3;

    int waPow2 = wa * wa;
    int waPow3 = waPow2 * wa;
    int cell_a = (i + gha) * (n + 2 * gha) * (o + 2 * gha) * waPow3 + (j + gha) * (o + 2 * gha) * waPow3 + (k + gha) * waPow3;

    int wbPow2 = wb * wb;

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
            csum +=
                a[cell_a + a_i * waPow2 + a_j * wa + a_k] *
                b[b_i * wbPow2 + b_j * wb + b_k];
        }

        c[cell_c + ci * wcPow2 + cj * wc + ck] = csum;
        // clang-format on
    }
}

/**
 * Multiplies a fixed stencil a with a varying stencilb, i.e. c = a * b.
 * m, n and o are dimensions of the grid.
 * wa and wb are the widths of the stencils. Must be odd and >= 3.
 * ghb and ghc are ghosts of the varying stencils at one border. ghb must be >= floor(wa / 2)
 * These restrictions are not checked in the kernel but shall be checked beforehand!
 * This kernel is supposed to be launched with one work-item per grid cell.
 * Remember to update ghosts of c if ghc > 0 afterwards.
 */
__kernel void mult_stencils_fix_var(
    __global double *restrict a,
    __global double *restrict b,
    __global double *restrict c,
    int m, int n, int o,
    int wa, int wb,
    int ghb, int ghc)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);

    int wa2 = wa >> 1;
    int wc = wa + wb - 1;

    // 1d indices
    int wcPow2 = wc * wc;
    int wcPow3 = wcPow2 * wc;
    int cell_c = (i + ghc) * (n + 2 * ghc) * (o + 2 * ghc) * wcPow3 + (j + ghc) * (o + 2 * ghc) * wcPow3 + (k + ghc) * wcPow3;

    int waPow2 = wa * wa;

    int wbPow2 = wb * wb;
    int wbPow3 = wbPow2 * wb;

    if (i < m && j < n && k < o)
    {
        // TODO temporary values in private array

        // clang-format off
        for (int a_i = 0; a_i < wa; a_i++)
        for (int a_j = 0; a_j < wa; a_j++)
        for (int a_k = 0; a_k < wa; a_k++)
            for (int b_i = 0; b_i < wb; b_i++)
            for (int b_j = 0; b_j < wb; b_j++)
            for (int b_k = 0; b_k < wb; b_k++)
            {
                int gpi = i + a_i - wa2 + ghb;
                int gpj = j + a_j - wa2 + ghb;
                int gpk = k + a_k - wa2 + ghb;

                int cell_b = gpi * (n + 2 * ghb) * (o + 2 * ghb) * wbPow3 + gpj * (o + 2 * ghb) * wbPow3 + gpk * wbPow3;

                int ci = a_i + b_i;
                int cj = a_j + b_j;
                int ck = a_k + b_k;

                if (ci >= 0 && ci < wc &&
                    cj >= 0 && cj < wc &&
                    ck >= 0 && ck < wc)
                {
                    c[cell_c + ci * wcPow2 + cj * wc + ck] +=
                        a[a_i * waPow2 + a_j * wa + a_k] *
                        b[cell_b + b_i * wbPow2 + b_j * wb + b_k];

                    // c[i + ghc][j + ghc][k + ghc][a_i + b_i][a_j + b_j][a_k + b_k] +=
                    //     a[i + gha][j + gha][k + gha][a_i][a_j][a_k] *
                    //     b[gpi][gpj][gpk][b_i][b_j][b_k];
                }
            }
        // clang-format on
    }
}

// reordered for loops
__kernel void mult_stencils_fix_var_reordered(
    __global double *restrict a,
    __global double *restrict b,
    __global double *restrict c,
    int m, int n, int o,
    int wa, int wb,
    int ghb, int ghc)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);

    int wa2 = wa >> 1;
    int wc = wa + wb - 1;

    // 1d indices
    int wcPow2 = wc * wc;
    int wcPow3 = wcPow2 * wc;
    int cell_c = (i + ghc) * (n + 2 * ghc) * (o + 2 * ghc) * wcPow3 + (j + ghc) * (o + 2 * ghc) * wcPow3 + (k + ghc) * wcPow3;

    int waPow2 = wa * wa;

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

                int cell_b = gpi * (n + 2 * ghb) * (o + 2 * ghb) * wbPow3 + gpj * (o + 2 * ghb) * wbPow3 + gpk * wbPow3;

                csum +=
                    a[a_i * waPow2 + a_j * wa + a_k] *
                    b[cell_b + b_i * wbPow2 + b_j * wb + b_k];
            }

            c[cell_c + ci * wcPow2 + cj * wc + ck] = csum;
        }
        // clang-format on
    }
}

// reordered for loops + parallel c loops. Must be called with m x n x o*wc*wc*wc work-items
__kernel void mult_stencils_fix_var_reordered_parallel_c(
    __global double *restrict a,
    __global double *restrict b,
    __global double *restrict c,
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

    // 1d indices
    int cell_c = (i + ghc) * (n + 2 * ghc) * (o + 2 * ghc) * wcPow3 + (j + ghc) * (o + 2 * ghc) * wcPow3 + (k + ghc) * wcPow3;

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

            int cell_b = gpi * (n + 2 * ghb) * (o + 2 * ghb) * wbPow3 + gpj * (o + 2 * ghb) * wbPow3 + gpk * wbPow3;

            csum +=
                a[a_i * waPow2 + a_j * wa + a_k] *
                b[cell_b + b_i * wbPow2 + b_j * wb + b_k];
        }

        c[cell_c + ci * wcPow2 + cj * wc + ck] = csum;
        // clang-format on
    }
}

/*********************************************************
 ******************** Utility kernels ********************
 *********************************************************/

// Form partial sum of buf per work-group and write result into buf_local.
// For full sum of buf sum_finish must be enqueued after this kernel (so global memory gets synchronized between work-groups).
// Not that using barrier(CLK_GLOBAL_MEM_FENCE) does not work for this as it does not synchronize work-groups.
// Must be called with a 1-D kernel range with #work-items = #elements in buf.
// num_elements must be #elements in buf.
// partial_sums's size must be equal to number of work-groups.
// buf_local's size must be equal to work-group size.
__kernel void sum_partial_global_eq_num_elements(
    __global double *restrict buf,
    __global double *restrict partial_sums,
    __local double *buf_local,
    int num_elements)
{
    int i = get_global_id(0);
    int wg_size = get_local_size(0);
    int iloc = get_local_id(0);

    if (i < num_elements)
    {
        // copy buf of this work-item into local storage
        buf_local[iloc] = buf[i];

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

// Form partial sum of buf per work-group and write result into buf_local.
// For full sum of buf sum_finish must be enqueued after this kernel (so global memory gets synchronized between work-groups).
// Not that using barrier(CLK_GLOBAL_MEM_FENCE) does not work for this as it does not synchronize work-groups.
// Must be called with a 1-D kernel range with #work-items = 0.5 * #elements in buf.
// num_elements must be half of #elements in buf.
// partial_sums's size must be equal to number of work-groups.
// buf_local's size must be equal to work-group size.
__kernel void sum_partial_global_eq_half_num_elements(
    __global double *restrict buf,
    __global double *restrict partial_sums,
    __local double *buf_local,
    int num_elements)
{
    int i = get_global_id(0);
    int wg_size = get_local_size(0);
    int iloc = get_local_id(0);

    if (i < num_elements)
    {
        // copy buf of this work-item into local storage. Two values since #wi = num_elements / 2 + padding
        buf_local[iloc] = buf[i];
        if (i + get_global_size(0) < num_elements)
            buf_local[iloc] += buf[i + get_global_size(0)];

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

// Form partial sum of buf per work-group and write result into buf_local.
// For full sum of buf sum_finish must be enqueued after this kernel (so global memory gets synchronized between work-groups).
// Not that using barrier(CLK_GLOBAL_MEM_FENCE) does not work for this as it does not synchronize work-groups.
// Must be called with a 1-D kernel range with #work-items = 0.5 * #elements in buf.
// num_elements must be half of #elements in buf.
// partial_sums's size must be equal to number of work-groups.
// buf_local's size must be equal to work-group size.
__kernel void sum_partial_global_eq_quarter_num_elements(
    __global double *restrict buf,
    __global double *restrict partial_sums,
    __local double *buf_local,
    int num_elements)
{
    int i = get_global_id(0);
    int wg_size = get_local_size(0);
    int iloc = get_local_id(0);

    if (i < num_elements)
    {
        // copy buf of this work-item into local storage. Two values since #wi = num_elements / 2 + padding
        buf_local[iloc] = buf[i];
        if (i + get_global_size(0) < num_elements)
            buf_local[iloc] += buf[i + get_global_size(0)];
        if (i + 2 * get_global_size(0) < num_elements)
            buf_local[iloc] += buf[i + 2 * get_global_size(0)];
        if (i + 3 * get_global_size(0) < num_elements)
            buf_local[iloc] += buf[i + 3 * get_global_size(0)];

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
    __global double *restrict buf_partial_sums,
    __global double *restrict buf_sum,
    int partial_sums_count)
{
    double sum = 0;

    for (int p = 0; p < partial_sums_count; p++)
        sum += buf_partial_sums[p];

    buf_sum[0] = sum;
}
