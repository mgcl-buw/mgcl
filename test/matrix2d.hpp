#ifndef MATRIX2D_HPP
#define MATRIX2D_HPP

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>

#include "../stencil.hpp"

namespace mgcl_test
{
    class Matrix2d
    {
    private:
        int m = 0;
        int n = 0;
        std::vector<std::vector<double>> values;

    public:
        Matrix2d(int m_, int n_) : m(m_), n(n_)
        {
            values.resize(m);
            for (auto &col : values)
            {
                col.resize(n);
                std::fill(col.begin(), col.end(), 0);
            }
        }

        Matrix2d(std::vector<std::vector<double>> &&vec)
            : m(vec.size()), n(vec.at(0).size()), values(std::move(vec)) {}

        Matrix2d kronecker(const Matrix2d &b);
        Matrix2d operator+(const Matrix2d &b);
        void operator+=(const Matrix2d &b);
        bool operator==(const Matrix2d &b) const;
        bool operator!=(const Matrix2d &b) const;
        Matrix2d operator*(const Matrix2d &b) const;

        std::vector<double> &operator[](int index);
        const std::vector<double> &operator[](int index) const;
        std::vector<double> &at(int index);
        const std::vector<double> &at(int index) const;

        int getM() const;
        int getN() const;

        static Matrix2d eye(int m);
        static Matrix2d eye(int m, int n);
        static Matrix2d diag(std::vector<std::tuple<double, int>> valuesAndOffsets, int m, int n);
        static Matrix2d laplace7p3d(int m, int n, int o);
        static Matrix2d restrictionFullWeight(int m, int n, int o);
        static Matrix2d cuttingMatrix1d(int m);
        static Matrix2d cuttingMatrix3d(int m, int n, int o);

        template <int N>
        static Matrix2d fromVaryingStencil(mgcl::VaryingStencil<N> &s)
        {
            int m = s.getDim1();
            int n = s.getDim2();
            int o = s.getDim3();
            Matrix2d c(m * n * o, m * n * o);
            int N2 = N >> 1;

            // clang-format off
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

            // clang-format on
            return c;
        }
    };

    std::ostream &operator<<(std::ostream &os, Matrix2d const &value);

}

#endif // MATRIX2D_HPP
