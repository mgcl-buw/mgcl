#include "matrix2d.hpp"

#include <fstream>
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

    /**
     * @brief Creates and returns the transposed of this matrix
     *
     * @return Matrix2d
     */
    Matrix2d Matrix2d::transposed()
    {
        Matrix2d t(n, m);

        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                t[j][i] = values[i][j];
            }

        return t;
    }

    /**
     * @brief Returns true if matrices are equal within the given tolerance.
     */
    bool Matrix2d::isEqual(const Matrix2d &b, double tol) const
    {
        if (m != b.getM() || n != b.getN())
            return false;

        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                double diff = values[i][j] - b[i][j];
                if (diff < 0)
                    diff *= -1;

                if (diff > tol)
                    return false;
            }

        return true;
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

    Matrix2d &Matrix2d::operator*(double b)
    {
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                values[i][j] *= b;
            }
        return *this;
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

    void Matrix2d::dumpToFile(std::string path) const
    {
        std::ofstream myfile;
        myfile.open(path, std::ios::out | std::ios::trunc);

        if (myfile.is_open())
        {
            for (int d1 = 0; d1 < m; d1++)
            {
                for (int d2 = 0; d2 < n; d2++)
                {
                    myfile << std::scientific << std::setprecision(17) << values[d1][d2] << "\t";
                }
                myfile << std::endl;
            }
            myfile.close();
        }
        else
        {
            throw "Couldn't open file for writing given by: " + path;
        }
    }

    void Matrix2d::dumpToFileWithIndices(std::string path) const
    {
        std::ofstream myfile;
        myfile.open(path, std::ios::out | std::ios::trunc);

        if (myfile.is_open())
        {
            for (int d1 = 0; d1 < m; d1++)
                for (int d2 = 0; d2 < n; d2++)
                {
                    myfile << d1 << "\t" << d2 << "\t" << std::scientific << std::setprecision(17)
                           << values[d1][d2] << std::endl;
                }
            myfile.close();
        }
        else
        {
            throw "Couldn't open file for writing given by: " + path;
        }
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
     * @param periodic Default is true.
     * @return Matrix2d
     */
    Matrix2d Matrix2d::laplace7p3d(int m, int n, int o, bool periodic)
    {
        std::vector<std::tuple<double, int>> vals{{-2, 0}, {1, -1}, {1, 1}};
        auto Dxx = diag(vals, m, m);
        auto Dyy = diag(vals, n, n);
        auto Dzz = diag(vals, o, o);

        if (periodic)
        {
            Dxx[0][m - 1] += 1;
            Dxx[m - 1][0] += 1;
            Dyy[0][n - 1] += 1;
            Dyy[n - 1][0] += 1;
            Dzz[0][o - 1] += 1;
            Dzz[o - 1][0] += 1;
        }

        return Dxx.kronecker(eye(o)).kronecker(eye(n)) +
               eye(m).kronecker(Dyy.kronecker(eye(o))) +
               eye(n).kronecker(eye(m).kronecker(Dzz));
    }

    /**
     * @brief Generates 3d full-weight restriction matrix.
     *
     * @param m
     * @param n
     * @param o
     * @return Matrix2d
     */
    Matrix2d Matrix2d::restrictionFullWeight(int m, int n, int o, bool periodic)
    {
        std::vector<std::tuple<double, int>> vals{{0.5, 0}, {0.25, -1}, {0.25, 1}};
        auto Dxx = diag(vals, m, m);
        auto Dyy = diag(vals, n, n);
        auto Dzz = diag(vals, o, o);

        if (periodic)
        {
            Dxx[0][m - 1] += 0.25;
            Dxx[m - 1][0] += 0.25;
            Dyy[0][n - 1] += 0.25;
            Dyy[n - 1][0] += 0.25;
            Dzz[0][o - 1] += 0.25;
            Dzz[o - 1][0] += 0.25;
        }

        return cuttingMatrix3d(m / 2, n / 2, o / 2) * Dxx.kronecker(Dyy.kronecker(Dzz));
    }

    /**
     * @brief Generates 3d bilinear prolongation matrix.
     *
     * @param m
     * @param n
     * @param o
     * @return Matrix2d
     */
    Matrix2d Matrix2d::prolongationBilinear(int m, int n, int o, bool periodic)
    {
        return 8 * restrictionFullWeight(m, n, o, periodic).transposed();
    }

    /**
     * @brief Creates a 1d cutting matrix K for cutting out rows and columns of another matrix A using K * A.
     *
     * @param m Size of rows. Matrix will be of size m x 2*m
     * @return Matrix2d
     */
    Matrix2d Matrix2d::cuttingMatrix1d(int m)
    {
        Matrix2d x(m, m << 1);

        for (int i = 0, j = 1; i < x.getM(); i++, j += 2)
            x[i][j] = 1;

        return x;
    }

    /**
     * @brief Creates a 3d cutting matrix K for cutting out rows and columns of another matrix A using K * A.
     *
     * @param m Coarse grid size in x-direction
     * @param n Coarse grid size in y-direction
     * @param o Coarse grid size in z-direction
     * @return Matrix2d
     */
    Matrix2d Matrix2d::cuttingMatrix3d(int m, int n, int o)
    {
        Matrix2d x = cuttingMatrix1d(m);
        Matrix2d y = cuttingMatrix1d(n);
        Matrix2d z = cuttingMatrix1d(o);

        return x.kronecker(y.kronecker(z));
    }

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
