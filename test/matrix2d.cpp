#include "matrix2d.hpp"

#include <iostream>

namespace mgcl_test
{

    Matrix2d Matrix2d::kronecker(const Matrix2d &b)
    {
        Matrix2d c(m * b.getM(), n * b.getN());

        for (int a1 = 0; a1 < m; a1++)
            for (int a2 = 0; a2 < n; a2++)
                for (int b1 = 0; b1 < b.getM(); b1++)
                    for (int b2 = 0; b2 < b.getN(); b2++)
                    {
                        c[a1 * b.getM() + b1][a2 * b.getN() + b2] = values[a1][a2] * b[b1][b2];
                    }

        return c;
    }

    Matrix2d Matrix2d::operator+(const Matrix2d &b)
    {
        if (m != b.getM() || n != b.getN())
            throw "Matrix dimensions do not match!";

        Matrix2d c(m, n);

        for (int a1 = 0; a1 < m; a1++)
            for (int a2 = 0; a2 < n; a2++)
            {
                c[a1][a2] = values[a1][a2] + b[a1][a2];
            }

        return c;
    }

    void Matrix2d::operator+=(const Matrix2d &b)
    {
        if (m != b.getM() || n != b.getN())
            throw "Matrix dimensions do not match!";

        for (int a1 = 0; a1 < m; a1++)
            for (int a2 = 0; a2 < n; a2++)
            {
                values[a1][a2] += b[a1][a2];
            }
    }

    bool Matrix2d::operator==(const Matrix2d &b) const
    {
        if (m != b.getM() || n != b.getN())
            return false;

        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                if (values[i][j] != b[i][j])
                    return false;
            }

        return true;
    }

    bool Matrix2d::operator!=(const Matrix2d &b) const
    {
        return !operator==(b);
    }

    Matrix2d Matrix2d::operator*(const Matrix2d &b) const
    {
        if (n != b.getM())
            throw "Dimensions do not fit!";

        Matrix2d c(m, b.getN());
        auto &a = *this;

        for (int ci = 0; ci < c.getM(); ci++)
            for (int cj = 0; cj < c.getN(); cj++)
            {
                double sum = 0;
                for (int idx = 0; idx < n; idx++)
                    sum += a[ci][idx] * b[idx][cj];

                c[ci][cj] = sum;
            }

        return c;
    }

    std::vector<double> &Matrix2d::operator[](int index)
    {
        return values[index];
    }

    const std::vector<double> &Matrix2d::operator[](int index) const
    {
        return values[index];
    }

    std::vector<double> &Matrix2d::at(int index)
    {
        return values.at(index);
    }

    const std::vector<double> &Matrix2d::at(int index) const
    {
        return values.at(index);
    }

    int Matrix2d::getM() const
    {
        return m;
    }

    int Matrix2d::getN() const
    {
        return n;
    }

    /**
     * @brief Shortcut for Matrix2d::eye(m, m)
     *
     * @param m
     * @return Matrix2d
     */
    Matrix2d Matrix2d::eye(int m)
    {
        return Matrix2d::eye(m, m);
    }

    /**
     * @brief Creates and returns identity matrix of size nxn.
     *
     * @param n
     * @return Matrix2d
     */
    Matrix2d Matrix2d::eye(int m, int n)
    {
        Matrix2d ret(m, n);
        int minidx = m < n ? m : n;
        for (int i = 0; i < minidx; i++)
        {
            ret[i][i] = 1;
        }
        return ret;
    }

    /**
     * @brief Creates a diagonal matrix of size nxn with values offset from the diagonal. Negative offset is to the left,
     * positive offset to the right.
     *
     * @param valuesAndOffsets Values for the diagonals that are offset by the second value from the main diagonal.
     * @param m 1st dim of resulting Matrix
     * @param n 2nd dim of resulting Matrix
     * @return Matrix2d
     */
    Matrix2d Matrix2d::diag(std::vector<std::tuple<double, int>> valuesAndOffsets, int m, int n)
    {
        if (m <= 0 || n <= 0)
            throw "m and n must be greater than zero!";

        Matrix2d ret(m, n);

        int minidx = m > n ? n : m;

        for (auto valAndOff : valuesAndOffsets)
        {
            double val = std::get<0>(valAndOff);
            int off = std::get<1>(valAndOff);

            for (int row = 0; row < m; row++)
                if (row + off >= 0 && row + off < n)
                    ret[row][row + off] = val;
        }

        return ret;
    }

    /**
     * @brief Generates 3d Laplace matrix.
     *
     * @param m
     * @param n
     * @param o
     * @return Matrix2d
     */
    Matrix2d Matrix2d::laplace7p3d(int m, int n, int o)
    {
        std::vector<std::tuple<double, int>> vals{{-2, 0}, {1, -1}, {1, 1}};
        auto Dxx = diag(vals, m, m);
        auto Dyy = diag(vals, n, n);
        auto Dzz = diag(vals, o, o);

        return Dxx.kronecker(eye(o)).kronecker(eye(n)) +
               eye(m).kronecker(Dyy.kronecker(eye(o))) +
               eye(n).kronecker(eye(m).kronecker(Dzz));
    }

    /**
     * @brief Generates full-weight restriction matrix scaled by 64.
     *
     * @param m
     * @param n
     * @param o
     * @return Matrix2d
     */
    Matrix2d Matrix2d::restrictionFullWeight(int m, int n, int o)
    {
        std::vector<std::tuple<double, int>> vals{{2, 0}, {1, -1}, {1, 1}};
        auto Dxx = diag(vals, m, m);
        auto Dyy = diag(vals, n, n);
        auto Dzz = diag(vals, o, o);

        return Dxx.kronecker(Dyy.kronecker(Dzz));
    }

    template <int N>
    Matrix2d Matrix2d::fromVaryingStencil(mgcl::VaryingStencil<N> &s)
    {
        int m = s.getDim1();
        int n = s.getDim2();
        int o = s.getDim3();
        Matrix2d c(m * n * o, m * n * o);
        int N2 = N >> 1;

        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < o; k++)
                    for (int ii = 0; ii < N; ii++)
                        for (int jj = 0; jj < N; jj++)
                            for (int kk = 0; kk < N; kk++)
                                // check if current stencil entry maps to a real grid point
                                if (i + ii >= N2 &&
                                    i + ii <= m + N2 - 1 &&
                                    j + jj >= N2 &&
                                    j + jj <= n + N2 - 1 &&
                                    k + kk >= N2 &&
                                    k + kk <= o + N2 - 1)
                                    // row is equal to grid point number, column is equal to grid point number plus a shift of half of stencil's size
                                    c[i * n * o + j * o + k][i * n * o + j * o + k + (ii - N2) * n * o + (jj - N2) * o + (kk - N2)] =
                                        s[i + s.getGhostsDim1()][j + s.getGhostsDim2()][k + s.getGhostsDim3()][ii][jj][kk];

        return c;
    }

    // instantiate template functions to avoid linker errors
    template Matrix2d Matrix2d::fromVaryingStencil(mgcl::VaryingStencil3x3x3 &s);
    template Matrix2d Matrix2d::fromVaryingStencil(mgcl::VaryingStencil5x5x5 &s);

    std::ostream &operator<<(std::ostream &os, Matrix2d const &value)
    {
        {
            os << std::scientific;
            for (int i = 0; i < value.getM(); i++)
            {
                for (int j = 0; j < value.getN(); j++)
                {
                    os << value[i][j] << "  ";
                }
                os << "\n";
            }

            return os;
        }
    }

} // namespace mgcl
