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
    __global double* restrict a,
    __global double* restrict b,
    __global double* restrict c,
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
    __global double* restrict a,
    __global double* restrict b,
    __global double* restrict c,
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
    __global double* restrict a,
    __global double* restrict b,
    __global double* restrict c,
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
    __global double* restrict a,
    __global double* restrict b,
    __global double* restrict c,
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
    __global double* restrict a,
    __global double* restrict b,
    __global double* restrict c,
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
    __global double* restrict a,
    __constant double* restrict b,
    __global double* restrict c,
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
    __global double* restrict a,
    __global double* restrict b,
    __global double* restrict c,
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
    __global double* restrict a,
    __global double* restrict b,
    __global double* restrict c,
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
    __global double* restrict a,
    __global double* restrict b,
    __global double* restrict c,
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
    __global double* restrict a,
    __global double* restrict b,
    __global double* restrict c,
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
    __global double* restrict a,
    __global double* restrict b,
    __global double* restrict c,
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
    __global double* restrict a,
    __global double* restrict b,
    __global double* restrict c,
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
 * Parallelizes the outermost 3 loops, thus must be called as 1d kernel with
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
    int idx = get_global_id(0);
    int no = n_c_loc * o_c_loc;
    int i = idx / no;
    int j = (idx - i * no) / o_c_loc;
    int k = idx % o_c_loc;

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
    if (i < m_c_loc + gh_c && n_c_loc + gh_c && o_c_loc + gh_c)
    {
        // for (int i = a_2h->getGhostsM(); i < (a_h.getM() >> 1) + a_2h->getGhostsM(); i++)
        //     for (int j = a_2h->getGhostsN(); j < (a_h.getN() >> 1) + a_2h->getGhostsN(); j++)
        //         for (int k = a_2h->getGhostsO(); k < (a_h.getO() >> 1) + a_2h->getGhostsO(); k++)
        // for each stencil entry of the coarse grid poiint this work-item maps to
        for (int ii = 0; ii < 3; ii++)
            for (int jj = 0; jj < 3; jj++)
                for (int kk = 0; kk < 3; kk++)
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

/**************************************************************
 * Helper structs and functions for galerkin, pointer versions
 **************************************************************/

// Returns the intersection of two intervals or [-1,-1] if they don't overlap
void intersect_ptr(Interval* a, Interval* b, Interval* ret)
{
    // Check if intervals overlap
    if (a->start <= b->end && b->start <= a->end)
    {
        // Calculate start and end points of intersection
        ret->start = (a->start > b->start) ? a->start : b->start;
        ret->end = (a->end < b->end) ? a->end : b->end;
    }
    else
    {
        // Intervals do not overlap
        ret->start = -1;
        ret->end = -1;
    }
};

// Returns the stencil entry indices of the stencil sitting at locationOfStencil that maps to mapsTo.
// No check is done, if the mapping is possible, i.e. the returned value might be outside of range [0,2].
// The result is just the difference of the indices plus one, since the stencil entry indices start at 0
// and not at -1.
void stencilEntryThatMapsTo_ptr(Point* locationOfStencil, Point* mapsTo, Point* ret)
{
    ret->x = mapsTo->x - locationOfStencil->x + 1;
    ret->y = mapsTo->y - locationOfStencil->y + 1;
    ret->z = mapsTo->z - locationOfStencil->z + 1;
};

// Returns the grid point indices that is mapped to by the stencil entry of another point.
// stencilEntry must be 0-based, hence the substraction by 1.
void pointMappedToByStencilEntry_ptr(Point* locationOfStencil, Point* stencilEntry, Point* ret)
{
    ret->x = locationOfStencil->x + (stencilEntry->x - 1);
    ret->y = locationOfStencil->y + (stencilEntry->y - 1);
    ret->z = locationOfStencil->z + (stencilEntry->z - 1);
};

// Returns the point on the fine grid that is related to the coarse grid point, respecting ghost cells.
void coarseToFine_ptr(Point* p, int ghc, int ghf, Point* ret)
{
    ret->x = (p->x - ghc) * 2 + 1 + ghf;
    ret->y = (p->y - ghc) * 2 + 1 + ghf;
    ret->z = (p->z - ghc) * 2 + 1 + ghf;
};

/**
 * Applies the Galerkin operator, calculating the stencils a_2h for the coarser grid, based on the stencils a_h on the fine grid.
 * Optimized version that does not need full stencil multiplication, but instead directly writes to the resulting stencils.
 *
 * Parallelizes the outermost 3 loops, thus must be called as 1d kernel with
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
__kernel void galerkin_ptr(
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
    int i = idx / no;
    int j = (idx - i * no) / o_c_loc;
    int k = idx % o_c_loc;

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
    if (i < m_c_loc + gh_c && n_c_loc + gh_c && o_c_loc + gh_c)
    {
        // for (int i = a_2h->getGhostsM(); i < (a_h.getM() >> 1) + a_2h->getGhostsM(); i++)
        //     for (int j = a_2h->getGhostsN(); j < (a_h.getN() >> 1) + a_2h->getGhostsN(); j++)
        //         for (int k = a_2h->getGhostsO(); k < (a_h.getO() >> 1) + a_2h->getGhostsO(); k++)
        // for each stencil entry of the coarse grid poiint this work-item maps to
        for (int ii = 0; ii < 3; ii++)
            for (int jj = 0; jj < 3; jj++)
                for (int kk = 0; kk < 3; kk++)
                {
                    // calculate fine grid point indices
                    Point gp_c = {i, j, k};
                    Point gp_f;
                    coarseToFine_ptr(&gp_c, gh_c, gh_f, &gp_f);
                    Point entry_gpc = {ii, jj, kk};
                    Point tmp;
                    pointMappedToByStencilEntry_ptr(&gp_c, &entry_gpc, &tmp);
                    Point entry_gpf;
                    coarseToFine_ptr(&tmp, gh_c, gh_f, &entry_gpf);

                    // find intersection S_P of neighbouring points for entry_gpf with reach=1 and gp_f with reach=2
                    Interval xa = {.start = gp_f.x - 2, .end = gp_f.x + 2};
                    Interval ya = {.start = gp_f.y - 2, .end = gp_f.y + 2};
                    Interval za = {.start = gp_f.z - 2, .end = gp_f.z + 2};
                    Interval xb = {.start = entry_gpf.x - 1, .end = entry_gpf.x + 1};
                    Interval yb = {.start = entry_gpf.y - 1, .end = entry_gpf.y + 1};
                    Interval zb = {.start = entry_gpf.z - 1, .end = entry_gpf.z + 1};

                    Interval S_P_x;
                    Interval S_P_y;
                    Interval S_P_z;
                    intersect_ptr(&xa, &xb, &S_P_x);
                    intersect_ptr(&ya, &yb, &S_P_y);
                    intersect_ptr(&za, &zb, &S_P_z);

                    // Start calc (R*A)*P
                    double res = 0;

                    // for each fine grid point gp_sp in S_P:
                    for (int spi = S_P_x.start; spi <= S_P_x.end; spi++)
                        for (int spj = S_P_y.start; spj <= S_P_y.end; spj++)
                            for (int spk = S_P_z.start; spk <= S_P_z.end; spk++)
                            {
                                Point gp_sp = {spi, spj, spk};
                                // tmp_p <- in stencil P located at gp_sp: Find stencil entry entry_p that maps to entry_gpf. Since
                                // gp_sp is in S_P, it is ensured that the stencil has a stencil entry that maps to entry_gpf.
                                Point tmp_p_indices;
                                stencilEntryThatMapsTo_ptr(&gp_sp, &entry_gpf, &tmp_p_indices);
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

                                Interval S_R_x;
                                Interval S_R_y;
                                Interval S_R_z;
                                intersect_ptr(&xa, &xb, &S_R_x);
                                intersect_ptr(&ya, &yb, &S_R_y);
                                intersect_ptr(&za, &zb, &S_R_z);

                                double sum = 0;
                                // for each fine grid point gp_sr in S_R:
                                for (int sri = S_R_x.start; sri <= S_R_x.end; sri++)
                                    for (int srj = S_R_y.start; srj <= S_R_y.end; srj++)
                                        for (int srk = S_R_z.start; srk <= S_R_z.end; srk++)
                                        {
                                            Point gp_sr = {sri, srj, srk};
                                            // tmp_r <- in stencil R located at gp_f: Find stencil entry entry_r that maps to gp_sr
                                            Point tmp_r_indices;
                                            stencilEntryThatMapsTo_ptr(&gp_f, &gp_sr, &tmp_r_indices);
                                            double tmp_r = r[tmp_r_indices.x * 9 + tmp_r_indices.y * 3 + tmp_r_indices.z];
                                            // tmp_a <- in stencil A located at gp_sr: Find stencil entry that maps to gp_sp
                                            Point tmp_a_indices;
                                            stencilEntryThatMapsTo_ptr(&gp_sr, &gp_sp, &tmp_a_indices);

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

/*********************************************************
 *                    Utility kernels                    *
 *********************************************************/

// Form partial sum of buf per work-group and write result into buf_local.
// For full sum of buf sum_finish must be enqueued after this kernel (so global memory gets synchronized between work-groups).
// Not that using barrier(CLK_GLOBAL_MEM_FENCE) does not work for this as it does not synchronize work-groups.
// Must be called with a 1-D kernel range with #work-items = #elements in buf.
// num_elements must be #elements in buf.
// partial_sums's size must be equal to number of work-groups.
// buf_local's size must be equal to work-group size.
__kernel void sum_partial_global_eq_num_elements(
    __global double* restrict buf,
    __global double* restrict partial_sums,
    __local double* buf_local,
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
    __global double* restrict buf,
    __global double* restrict partial_sums,
    __local double* buf_local,
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
// Must be called with a 1-D kernel range with #work-items = 0.25 * #elements in buf.
// num_elements must be quarter of #elements in buf.
// partial_sums's size must be equal to number of work-groups.
// buf_local's size must be equal to work-group size.
__kernel void sum_partial_global_eq_quarter_num_elements(
    __global double* restrict buf,
    __global double* restrict partial_sums,
    __local double* buf_local,
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

// Form partial sum of buf per work-group and write result into buf_local.
// For full sum of buf sum_finish must be enqueued after this kernel (so global memory gets synchronized between work-groups).
// Not that using barrier(CLK_GLOBAL_MEM_FENCE) does not work for this as it does not synchronize work-groups.
// Must be called with a 1-D kernel range with #work-items = 1/fractions * #elements in buf.
// num_elements must be 1/fraction * #elements in buf.
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
        // copy buf of this work-item into local storage.
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

/**************************************
 ***                                ***
 *          Jacobi Kernels            *
 ***                                ***
 **************************************/

/* runs one iteration of jacobi's method using one work-item per grid node.
 * uses a 3D kernel, which parallelizes all three loop in x,y and z directions.
 * global size must be of ghosted grid.
 * m, n and o must be dimensions of ghosted grid, too.
 * h2 is grid spacing to the power of 2
 * dinv is h2/A(i,i), e.g. h2/6.0 for 3D laplacian stencil
 * if store_residual is true, the residual will be stored into global field r.
 * stencilValues is a VaryingStencilGpu having width 3 (i.e. a 6d array).
 * ghosts_sv is the amount of ghost cells of stencilValues.
 * idx_start determines which cells shall be calculated, which is relevant for running
 *   Jacobi with multiple iterations without ghost cell update in-between. I.e. when
 *   stepsPerIter = 1: idx_start = ghosts.
 */
__kernel void jacobi_iter_27point_varying_stencil_3d(
    __global double* restrict v_in, // needed s.t. every work-item can read surrounding cell values
    __global double* restrict v_out,
    __global double* restrict f,
    __global double* restrict r,
    __global double* restrict stencilValues,
    const double omega,
    const int mgh, const int ngh, const int ogh,
    const int ghosts, const int ghosts_sv,
    const int idx_start, const int store_residual)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);

    // calculate residual for real cells plus some ghost cells if stepsPerIter > 1.
    if (i >= idx_start && j >= idx_start && k >= idx_start && i < mgh - idx_start && j < ngh - idx_start && k < ogh - idx_start)
    {
        int ioff = ngh * ogh;
        int joff = ogh;
        int koff = 1;
        int index = i * ioff + j * ogh + k;

        int koff_sv = 27;
        int joff_sv = ((ogh - 2 * ghosts) + 2 * ghosts_sv) * koff_sv;
        int ioff_sv = ((ngh - 2 * ghosts) + 2 * ghosts_sv) * joff_sv;
        int index_sv = (i - ghosts + ghosts_sv) * ioff_sv + (j + (ghosts_sv - ghosts)) * joff_sv + (k + (ghosts_sv - ghosts)) * koff_sv;

        double res;
        double v_in_index = v_in[index];
        double sv_self = stencilValues[index_sv + 9 + 3 + 1];

        // A*v
        // clang-format off
            double stencilsum = sv_self * v_in_index
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

        // clang-format off
            // double stencilsum = sv_self * v_in_index
            //     + stencilValues[index_sv + 1 * 9 + 1 * 3 + 0] * v_in[index - 1]
            //     + stencilValues[index_sv + 1 * 9 + 1 * 3 + 2] * v_in[index + 1]
            //     + stencilValues[index_sv + 1 * 9 + 0 * 3 + 1] * v_in[index - joff]
            //     + stencilValues[index_sv + 1 * 9 + 2 * 3 + 1] * v_in[index + joff]
            //     + stencilValues[index_sv + 0 * 9 + 1 * 3 + 1] * v_in[index - ioff]
            //     + stencilValues[index_sv + 2 * 9 + 1 * 3 + 1] * v_in[index + ioff]
            //     + stencilValues[index_sv + 1 * 9 + 0 * 3 + 0] * v_in[index - joff - koff]
            //     + stencilValues[index_sv + 1 * 9 + 0 * 3 + 2] * v_in[index - joff + koff]
            //     + stencilValues[index_sv + 1 * 9 + 2 * 3 + 0] * v_in[index + joff - koff]
            //     + stencilValues[index_sv + 1 * 9 + 2 * 3 + 2] * v_in[index + joff + koff]
            //     + stencilValues[index_sv + 0 * 9 + 1 * 3 + 0] * v_in[index - ioff - koff]
            //     + stencilValues[index_sv + 0 * 9 + 1 * 3 + 2] * v_in[index - ioff + koff]
            //     + stencilValues[index_sv + 2 * 9 + 1 * 3 + 0] * v_in[index + ioff - koff]
            //     + stencilValues[index_sv + 2 * 9 + 1 * 3 + 2] * v_in[index + ioff + koff]
            //     + stencilValues[index_sv + 0 * 9 + 0 * 3 + 1] * v_in[index - ioff - joff]
            //     + stencilValues[index_sv + 0 * 9 + 2 * 3 + 1] * v_in[index - ioff + joff]
            //     + stencilValues[index_sv + 2 * 9 + 0 * 3 + 1] * v_in[index + ioff - joff]
            //     + stencilValues[index_sv + 2 * 9 + 2 * 3 + 1] * v_in[index + ioff + joff]
            //     + stencilValues[index_sv + 0 * 9 + 0 * 3 + 0] * v_in[index - ioff - joff - koff]
            //     + stencilValues[index_sv + 0 * 9 + 0 * 3 + 2] * v_in[index - ioff - joff + koff]
            //     + stencilValues[index_sv + 0 * 9 + 2 * 3 + 0] * v_in[index - ioff + joff - koff]
            //     + stencilValues[index_sv + 0 * 9 + 2 * 3 + 2] * v_in[index - ioff + joff + koff]
            //     + stencilValues[index_sv + 2 * 9 + 0 * 3 + 0] * v_in[index + ioff - joff - koff]
            //     + stencilValues[index_sv + 2 * 9 + 0 * 3 + 2] * v_in[index + ioff - joff + koff]
            //     + stencilValues[index_sv + 2 * 9 + 2 * 3 + 0] * v_in[index + ioff + joff - koff]
            //     + stencilValues[index_sv + 2 * 9 + 2 * 3 + 2] * v_in[index + ioff + joff + koff];
        // clang-format on

        // clang-format off
            // stencilsum = stencilValues[isv][jsv][ksv][1][1][1]  * vraw[i][j][k]
            //     + stencilValues[isv][jsv][ksv][1][1][0]         * vraw[i][j][k - 1]
            //     + stencilValues[isv][jsv][ksv][1][1][2]         * vraw[i][j][k + 1]
            //     + stencilValues[isv][jsv][ksv][1][0][1]         * vraw[i][j - 1][k]
            //     + stencilValues[isv][jsv][ksv][1][2][1]         * vraw[i][j + 1][k]
            //     + stencilValues[isv][jsv][ksv][0][1][1]         * vraw[i - 1][j][k]
            //     + stencilValues[isv][jsv][ksv][2][1][1]         * vraw[i + 1][j][k]
               
            //     + stencilValues[isv][jsv][ksv][1][0][0] * vraw[i][j - 1][k - 1]
            //     + stencilValues[isv][jsv][ksv][1][0][2] * vraw[i][j - 1][k + 1]
            //     + stencilValues[isv][jsv][ksv][1][2][0] * vraw[i][j + 1][k - 1]
            //     + stencilValues[isv][jsv][ksv][1][2][2] * vraw[i][j + 1][k + 1]
            //     + stencilValues[isv][jsv][ksv][0][1][0] * vraw[i - 1][j][k - 1]
            //     + stencilValues[isv][jsv][ksv][0][1][2] * vraw[i - 1][j][k + 1]
            //     + stencilValues[isv][jsv][ksv][2][1][0] * vraw[i + 1][j][k - 1]
            //     + stencilValues[isv][jsv][ksv][2][1][2] * vraw[i + 1][j][k + 1]
            //     + stencilValues[isv][jsv][ksv][0][0][1] * vraw[i - 1][j - 1][k]
            //     + stencilValues[isv][jsv][ksv][0][2][1] * vraw[i - 1][j + 1][k]
            //     + stencilValues[isv][jsv][ksv][2][0][1] * vraw[i + 1][j - 1][k]
            //     + stencilValues[isv][jsv][ksv][2][2][1] * vraw[i + 1][j + 1][k]
                
            //     + stencilValues[isv][jsv][ksv][0][0][0] * vraw[i - 1][j - 1][k - 1]
            //     + stencilValues[isv][jsv][ksv][0][0][2] * vraw[i - 1][j - 1][k + 1]
            //     + stencilValues[isv][jsv][ksv][0][2][0] * vraw[i - 1][j + 1][k - 1]
            //     + stencilValues[isv][jsv][ksv][0][2][2] * vraw[i - 1][j + 1][k + 1]
            //     + stencilValues[isv][jsv][ksv][2][0][0] * vraw[i + 1][j - 1][k - 1]
            //     + stencilValues[isv][jsv][ksv][2][0][2] * vraw[i + 1][j - 1][k + 1]
            //     + stencilValues[isv][jsv][ksv][2][2][0] * vraw[i + 1][j + 1][k - 1]
            //     + stencilValues[isv][jsv][ksv][2][2][2] * vraw[i + 1][j + 1][k + 1];
        // clang-format on

        // r = f - A*v
        res = f[index] - stencilsum;

        // barrier(CLK_GLOBAL_MEM_FENCE);
        // if (get_global_id(0) == 2 && get_global_id(1) == 5 && i == 1)
        // {
        //     // printf("ocl x = %d, omega = %e, stencilsum = %e, res = %e,  v_in = %e, sv_self = %e\n", i, omega, stencilsum, res, v_in[index], sv_self);
        //     // printf("ocl omega * (1.0 / sv_self) * res = %e\n", omega * (1.0 / sv_self) * res);
        //     // printf("ocl omega = %e, (1.0 / sv_self) = %e, res = %e\n", omega, (1.0 / sv_self), res);
        //     printf("ocl stencilsum = %e\n", stencilsum);
        //     // print27point(v_in, index, ioff, joff, koff);
        //     print27point_sv(v_in, index, ioff, joff, koff, stencilValues, index_sv);
        //     // print_7point(v_in, index, ioff, joff, koff);
        // }
        // barrier(CLK_GLOBAL_MEM_FENCE);

        // u_(m+1) = u_(m) + omega * (D^-1) * r_(m)
        v_out[index] = v_in_index + omega * (1.0 / sv_self) * res;

        // barrier(CLK_GLOBAL_MEM_FENCE);
        // if (get_global_id(0) == 2 && get_global_id(1) == 5 && i == 1)
        // {
        //     printf("ocl x = %d, omega = %e, res = %e, v_out = %e, sv_self = %e\n", i, omega, res, v_out[index], sv_self);
        //     print27point(v_out, index, ioff, joff, koff);
        //     // print_7point(v_in, index, ioff, joff, koff);
        // }

        if (store_residual)
            r[index] = res;
    }
}

/**************************************
 ***                                ***
 *          Residual Kernels          *
 ***                                ***
 **************************************/

/*
 * Calculates residual with a 27p varying stencil.
 * 3d kernel, must be launched with ghosted mxnxo work-items, i.e. #work-items == amount of ghosted grid cells.
 * Work-group size can be arbitrary chosen. 1x1x32 seems optimal for my laptop gpu.
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
__kernel void residual_27point_varying_stencil_3d_one_wi_per_cell(
    __global double* restrict v_in,
    __global double* restrict f,
    __global double* restrict r,
    __global double* restrict stencilValues,
    const int m, const int n, const int o,
    const int ghosts, const int ghosts_sv,
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
    int iend_v = m - ghosts - moff;
    int jend_v = n - ghosts - noff;
    int kend_v = o - ghosts - ooff;

    // calculate residual only for relevant cells (off = 0: only real cells)
    if (i >= istart_v && j >= jstart_v && k >= kstart_v && i < iend_v && j < jend_v && k < kend_v)
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
 */
__kernel void residual_27point_varying_stencil_1d_one_wi_per_cell(
    __global double* restrict v_in,
    __global double* restrict f,
    __global double* restrict r,
    __global double* restrict stencilValues,
    const int m, const int n, const int o,
    const int ghosts, const int ghosts_sv,
    const int moff, const int noff, const int ooff)
{
    int idx = get_global_id(0);
    int no = n * o;
    int i = idx / no;
    int j = (idx - i * no) / o;
    int k = idx % o;

    // loop boundaries
    // TODO maybe refactor to use v_ghm, etc.?
    int istart_v = ghosts + moff;
    int jstart_v = ghosts + noff;
    int kstart_v = ghosts + ooff;
    int iend_v = m - ghosts - moff;
    int jend_v = n - ghosts - noff;
    int kend_v = o - ghosts - ooff;

    // calculate residual only for relevant cells (off = 0: only real cells)
    if (i >= istart_v && j >= jstart_v && k >= kstart_v && i < iend_v && j < jend_v && k < kend_v)
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

        // if (i == 2 && j == 2 && k == 2)
        // {
        //     printf("ocl stencilsum = %e\n", stencilsum);
        //     print27point_sv(v_in, index, ioff, joff, koff, stencilValues, index_sv);
        // }

        // r = f - A*v
        r[index] = f[index] - stencilsum;
    }
}

/******************************************
 ***                                    ***
 *          Ghost Update Kernels          *
 ***                                    ***
 ******************************************/

/**
 * Updates ghosts of a cuboid, respecting small grids, e.g. gh > m.
 * Needs to be called with one work-item per cell of ghosted grid.
 * Work-items that map to a real cell simply do nothing (optimization potential here!).
 * m,n,o are sizes of real grid.
 * ghm, ghn, gho are amount of ghosts at one border.
 */
__kernel void update_ghosts_periodic_3d(
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
 * Updates ghosts of a cuboid, respecting small grids, e.g. gh > m.
 * Needs to be called with one work-item per cell of ghosted grid.
 * Work-items that map to a real cell simply do nothing (optimization potential here!).
 * m,n,o are sizes of real grid.
 * ghm, ghn, gho are amount of ghosts at one border.
 */
__kernel void update_ghosts_periodic_1d(
    __global double* restrict c,
    int m, int n, int o,
    int ghm, int ghn, int gho)
{
    int idx = get_global_id(0);
    int mgh = m + 2 * ghm;
    int ngh = n + 2 * ghn;
    int ogh = o + 2 * gho;
    int no = ngh * ogh;
    int i = idx / no;
    int j = (idx - i * no) / ogh;
    int k = idx % ogh;

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
 * Updates ghosts of a cuboid, respecting small grids, e.g. gh > m.
 * Needs to be called with one work-item per ghost cell, i.e. 2*ghm + 2*ghn + 2*gho.
 * Work-items that map to a real cell simply do nothing (optimization potential here!).
 * m,n,o are sizes of real grid.
 * ghm, ghn, gho are amount of ghosts at one border.
 */
__kernel void update_ghosts_periodic_1d_ghosts_cells_only(
    __global double* restrict c,
    int m, int n, int o,
    int mgh, int ngh, int ogh,
    int ghm, int ghn, int gho)
{
    int idx = get_global_id(0);
    int no = ngh * ogh;
    // TOOD this does not work since ghost cells is not a completely filled cube. How to calculate from 1d idx?
    int i = idx / no;
    int j = (idx - i * no) / ogh;
    int k = idx % ogh;

    if (i < 2 * ghm && j < 2 * ghn && k < 2 * gho)
    {
        // TOOD this does not work since ghost cells is not a completely filled cube. How to calculate from 1d idx?
        i = (i < ghm) ? i : i + m;
        j = (j < ghn) ? j : j + n;
        k = (k < gho) ? k : k + o;
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

/******************************************************************
 ***                                                            ***
 *         Original content of mgcl.cl as of 2024-03-04           *
 * Added s.t. methods without benchmark kernel variants           *
 * can be executed still.                                         *
 ***                                                            ***
 ******************************************************************/
#ifndef NULL
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