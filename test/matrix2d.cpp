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

} // namespace mgcl
