#include "matrix2d.hpp"

namespace mgcl_test
{

    Matrix2d Matrix2d::kronecker(Matrix2d &b)
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

    Matrix2d Matrix2d::operator+(Matrix2d &b)
    {
        Matrix2d c(m, n);

        for (int a1 = 0; a1 < m; a1++)
            for (int a2 = 0; a2 < n; a2++)
            {
                c[a1][a2] = values[a1][a2] + b[a1][a2];
            }

        return c;
    }

    void Matrix2d::operator+=(Matrix2d &b)
    {
        for (int a1 = 0; a1 < m; a1++)
            for (int a2 = 0; a2 < n; a2++)
            {
                values[a1][a2] += b[a1][a2];
            }
    }

    std::vector<double> &Matrix2d::operator[](int index)
    {
        return values[index];
    }

    std::vector<double> &Matrix2d::at(int index)
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
     * @brief Creates and returns identity matrix of size nxn.
     * TODO not only square matrices
     *
     * @param n
     * @return Matrix2d
     */
    Matrix2d Matrix2d::eye(int n)
    {
        Matrix2d ret(n, n);
        for (int i = 0; i < n; i++)
        {
            ret[i][i] = 1;
        }
        return ret;
    }

    /**
     * @brief Creates a diagonal matrix of size nxn with values offset from the diagonal. Negative offset is to the left,
     * positive offset to the right.
     * TODO implement + not only square matrices
     *
     * @param value
     * @param n
     * @param off
     * @return Matrix2d
     */
    Matrix2d Matrix2d::diag(double value, int n, int off)
    {
        Matrix2d ret(n, n);

        for (int i = 0; i < n; i++)
        {
            ret[i][i] = 1;
        }

        return ret;
    }

} // namespace mgcl
