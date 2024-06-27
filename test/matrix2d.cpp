#include "matrix2d.hpp"

#include <fstream>
#include <iostream>

namespace mgcl_test
{

    Matrix2d Matrix2d::kronecker(const Matrix2d& b)
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
    bool Matrix2d::isEqual(const Matrix2d& b, double tol) const
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

    Matrix2d Matrix2d::operator+(const Matrix2d& b)
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

    void Matrix2d::operator+=(const Matrix2d& b)
    {
        if (m != b.getM() || n != b.getN())
            throw "Matrix dimensions do not match!";

        for (int a1 = 0; a1 < m; a1++)
            for (int a2 = 0; a2 < n; a2++)
            {
                values[a1][a2] += b[a1][a2];
            }
    }

    bool Matrix2d::operator==(const Matrix2d& b) const
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

    bool Matrix2d::operator!=(const Matrix2d& b) const
    {
        return !operator==(b);
    }

    Matrix2d Matrix2d::operator*(const Matrix2d& b) const
    {
        if (n != b.getM())
            throw "Dimensions do not fit!";

        Matrix2d c(m, b.getN());
        auto& a = *this;

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

    Matrix2d& Matrix2d::operator*(double b)
    {
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                values[i][j] *= b;
            }
        return *this;
    }

    std::vector<double>& Matrix2d::operator[](int index)
    {
        return values[index];
    }

    const std::vector<double>& Matrix2d::operator[](int index) const
    {
        return values[index];
    }

    std::vector<double>& Matrix2d::at(int index)
    {
        return values.at(index);
    }

    const std::vector<double>& Matrix2d::at(int index) const
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

    void Matrix2d::dumpToFileDecimalFormat(std::string path, int precision) const

    {
        std::ofstream myfile;
        myfile.open(path, std::ios::out | std::ios::trunc);

        if (myfile.is_open())
        {
            for (int d1 = 0; d1 < m; d1++)
            {
                for (int d2 = 0; d2 < n; d2++)
                {
                    myfile << std::fixed << std::setprecision(precision) << values[d1][d2] << "\t";
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

    int Matrix2d::getM() const
    {
        return m;
    }

    int Matrix2d::getN() const
    {
        return n;
    }

    /**
     * @brief Calculates the equivalent indices for the varying stencil for a specific entry. Returns -1 if
     *   the matrix entry does not map to a stencil entry. Only works for square matrices.
     *
     * @param row Index of the row of the entry, i.e. the grid point the stencil belongs to.
     * @param col Index of the column of the entry, i.e. the grid point the stencil entry is applied to.
     * @param m Size of the grid in the x-direction
     * @param n Size of the grid in the y-direction
     * @param o Size of the grid in the z-direction
     * @param periodic Whether the matrix (and the grid) is periodic
     * @return std::array<int, 6> Indices of the stencil of the form {i, j, k, ii, jj, kk}, where i,j,k is the index
     *   of the grid point and ii,jj,kk is the index of the stencil entry.
     */
    std::array<int, 6> Matrix2d::getStencilIndicesForEntry(int row, int col,
                                                           int m, int n, int o,
                                                           int stencilWidth, bool periodic) const
    {
        if (m * n * o != getM())
            throw "Matrix2d::getStencilIndicesForEntry: m * n * o != getM(). Only works for square matrices";
        if (m * n * o != getN())
            throw "Matrix2d::getStencilIndicesForEntry: m * n * o != getN(). Only works for square matrices";

        // Grid point mapped to by current matrix row (row number equals 1d index of grid point)
        int no_rows = n * o;
        int i = row / no_rows;
        int j = (row - i * no_rows) / o;
        int k = row % o;

        // Grid point the current stencil entry maps to (column number equals 1d index of grid point)
        int no = n * o;
        int gpi = col / no;
        int gpj = (col - gpi * no) / o;
        int gpk = col % o;

        if (periodic)
        {
            gpi = gpi % m;
            gpj = gpj % n;
            gpk = gpk % o;

            // shift mod result into positive range
            if (gpi < 0)
                gpi += m;

            if (gpj < 0)
                gpj += n;

            if (gpk < 0)
                gpk += o;
        }

        // Stencil entry
        int ii = gpi + (stencilWidth / 2) - i;
        int jj = gpj + (stencilWidth / 2) - j;
        int kk = gpk + (stencilWidth / 2) - k;

        // check if matrix entry is actually part of a stencil
        if (ii < 0 || ii >= stencilWidth || jj < 0 || jj >= stencilWidth || kk < 0 || kk >= stencilWidth)
            return {-1, -1, -1, -1, -1, -1};
        else
            return {i, j, k, ii, jj, kk};
    }

    /**
     * @brief Calculates the stencil entry indices for a given matrix entry. Returns -1 if
     *   the matrix entry does not map to a stencil entry. Specialized for restriction matrix, e.g. of size 8x64.
     *
     * The restriction matrix has e.g. the size 8x64, where the rows denote the coarse grid points while
     * the columns denote the fine grid points. Thus we need some special handling regarding the row index compared
     * to the index acquisition of a square matrix.
     * It is assumed, that m,n,o is the size of the fine grid, i.e. m*n*o = cols, and m/2,n/2,o/2 is the size of the
     * coarse grid, i.e. m/2*n/2*o/2 = rows.
     *
     * @param row Index of the row of the entry, i.e. the grid point the stencil belongs to.
     * @param col Index of the column of the entry, i.e. the grid point the stencil entry is applied to.
     * @param m Size of the fine grid in the x-direction
     * @param n Size of the fine grid in the y-direction
     * @param o Size of the fine grid in the z-direction
     * @param periodic Whether the matrix (and the grid) is periodic
     * @return std::array<int, 3> Indices of the stencil of the form {ii, jj, kk}, where ii,jj,kk is
     *   the index of the stencil entry.
     */
    std::array<int, 3> Matrix2d::getStencilEntryOfRestrictionMatrix(int row, int col,
                                                                    int m, int n, int o,
                                                                    int stencilWidth, bool periodic) const
    {
        int m2 = m >> 1;
        int n2 = n >> 1;
        int o2 = o >> 1;

        if (m2 * n2 * o2 != getM())
            throw "Matrix2d::getStencilIndicesForEntry: m/2 * n/2 * o/2 != getM().";
        if (m * n * o != getN())
            throw "Matrix2d::getStencilIndicesForEntry: m * n * o != getN().";

        // Coarse grid point mapped to by current matrix row.
        int no_rows = n2 * o2;
        int i = row / no_rows;
        int j = (row - i * no_rows) / o2;
        int k = row % o2;

        // Correct for fine grid
        i = i * 2 + 1;
        j = j * 2 + 1;
        k = k * 2 + 1;

        // Grid point the current stencil entry maps to (column number equals 1d index of grid point)
        int no_cols = n * o;
        int gpi = col / no_cols;
        int gpj = (col - gpi * no_cols) / o;
        int gpk = col % o;

        if (periodic)
        {
            gpi = gpi % m;
            gpj = gpj % n;
            gpk = gpk % o;

            // shift mod result into positive range
            if (gpi < 0)
                gpi += m;

            if (gpj < 0)
                gpj += n;

            if (gpk < 0)
                gpk += o;
        }

        // Stencil entry
        int ii = gpi + (stencilWidth / 2) - i;
        int jj = gpj + (stencilWidth / 2) - j;
        int kk = gpk + (stencilWidth / 2) - k;

        // check if matrix entry is actually part of a stencil
        if (ii < 0 || ii >= stencilWidth || jj < 0 || jj >= stencilWidth || kk < 0 || kk >= stencilWidth)
            return {-1, -1, -1};
        else
            return {ii, jj, kk};
    }

    /**
     * @brief Calculates the stencil entry indices for a given matrix entry. Returns -1 if
     *   the matrix entry does not map to a stencil entry. Specialized for prolongation matrix, e.g. of size 64x8.
     *
     * The prolongation matrix has e.g. the size 64x8, where the rows denote the fine grid points while
     * the columns denote the coarse grid points. Thus we need some special handling regarding the column index compared
     * to the index acquisition of a square matrix.
     * It is assumed, that m,n,o is the size of the fine grid, i.e. m*n*o = rows, and m/2,n/2,o/2 is the size of the
     * coarse grid, i.e. m/2*n/2*o/2 = cols.
     *
     * @param row Index of the row of the entry, i.e. the grid point the stencil belongs to.
     * @param col Index of the column of the entry, i.e. the grid point the stencil entry is applied to.
     * @param m Size of the fine grid in the x-direction
     * @param n Size of the fine grid in the y-direction
     * @param o Size of the fine grid in the z-direction
     * @param periodic Whether the matrix (and the grid) is periodic
     * @return std::array<int, 3> Indices of the stencil of the form {ii, jj, kk}, where ii,jj,kk is
     *   the index of the stencil entry.
     */
    std::array<int, 3> Matrix2d::getStencilEntryOfProlongationMatrix(int row, int col,
                                                                     int m, int n, int o,
                                                                     int stencilWidth, bool periodic) const
    {
        int m2 = m >> 1;
        int n2 = n >> 1;
        int o2 = o >> 1;

        if (m2 * n2 * o2 != getN())
            throw "Matrix2d::getStencilIndicesForEntry: m/2 * n/2 * o/2 != getN().";
        if (m * n * o != getM())
            throw "Matrix2d::getStencilIndicesForEntry: m * n * o != getM().";

        // Coarse grid point mapped to by current matrix row.
        int no_rows = n * o;
        int i = row / no_rows;
        int j = (row - i * no_rows) / o;
        int k = row % o;

        // Coarse grid point the current stencil entry maps to
        int no_cols = n2 * o2;
        int gpi = col / no_cols;
        int gpj = (col - gpi * no_cols) / o2;
        int gpk = col % o2;

        // Correct for fine grid
        gpi = gpi * 2 + 1;
        gpj = gpj * 2 + 1;
        gpk = gpk * 2 + 1;

        if (periodic)
        {
            gpi = gpi % m;
            gpj = gpj % n;
            gpk = gpk % o;

            // shift mod result into positive range
            if (gpi < 0)
                gpi += m;

            if (gpj < 0)
                gpj += n;

            if (gpk < 0)
                gpk += o;
        }

        // Stencil entry
        int ii = gpi + (stencilWidth / 2) - i;
        int jj = gpj + (stencilWidth / 2) - j;
        int kk = gpk + (stencilWidth / 2) - k;

        // check if matrix entry is actually part of a stencil
        if (ii < 0 || ii >= stencilWidth || jj < 0 || jj >= stencilWidth || kk < 0 || kk >= stencilWidth)
            return {-1, -1, -1};
        else
            return {ii, jj, kk};
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
     * @brief Generates 3d full-weight matrix that is not multiplied with the cutting matrix.
     *
     * @param m
     * @param n
     * @param o
     * @return Matrix2d
     */
    Matrix2d Matrix2d::fullWeightNonCut(int m, int n, int o, bool periodic)
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

        return Dxx.kronecker(Dyy.kronecker(Dzz));
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

    std::ostream& operator<<(std::ostream& os, Matrix2d const& value)
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
