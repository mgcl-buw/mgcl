#ifndef MATRIX2D_HPP
#define MATRIX2D_HPP

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>

#include "../src/stencil.hpp"

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
            for (auto& col : values)
            {
                col.resize(n);
                std::fill(col.begin(), col.end(), 0);
            }
        }

        Matrix2d(std::vector<std::vector<double>>&& vec)
            : m(vec.size()), n(vec.at(0).size()), values(std::move(vec)) {}

        Matrix2d kronecker(const Matrix2d& b);
        Matrix2d transposed();

        bool isEqual(const Matrix2d& b, double tol) const;

        Matrix2d operator+(const Matrix2d& b);
        void operator+=(const Matrix2d& b);
        bool operator==(const Matrix2d& b) const;
        bool operator!=(const Matrix2d& b) const;
        Matrix2d operator*(const Matrix2d& b) const;
        Matrix2d& operator*(double b);
        Matrix2d& operator*(int b) { return operator*(static_cast<double>(b)); }
        friend Matrix2d& operator*(double b, Matrix2d& m) { return m * b; }
        friend Matrix2d& operator*(double b, Matrix2d&& m) { return m * b; }
        friend Matrix2d& operator*(int b, Matrix2d& m) { return m * b; }
        friend Matrix2d& operator*(int b, Matrix2d&& m) { return m * b; }

        std::vector<double>& operator[](int index);
        const std::vector<double>& operator[](int index) const;
        std::vector<double>& at(int index);
        const std::vector<double>& at(int index) const;

        void dumpToFile(std::string path) const;
        void dumpToFileWithIndices(std::string path) const;

        int getM() const;
        int getN() const;

        static Matrix2d eye(int m);
        static Matrix2d eye(int m, int n);
        static Matrix2d diag(std::vector<std::tuple<double, int>> valuesAndOffsets, int m, int n);
        static Matrix2d laplace7p3d(int m, int n, int o, bool periodic = true);
        static Matrix2d fullWeightNonCut(int m, int n, int o, bool periodic = true);
        static Matrix2d restrictionFullWeight(int m, int n, int o, bool periodic = true);
        static Matrix2d prolongationBilinear(int m, int n, int o, bool periodic = true);
        static Matrix2d cuttingMatrix1d(int m);
        static Matrix2d cuttingMatrix3d(int m, int n, int o);

        /**
         * @brief Creates a matrix from a VaryingStencil of size NxNxN.
         *
         * @tparam N
         * @param s
         * @param periodic
         * @return Matrix2d
         */
        template <int N>
        static Matrix2d fromVaryingStencil(mgcl::VaryingStencil<N>& s, bool periodic)
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
                {
                    // Column index of the resulting matrix is equal to the grid point
                    // the current stencil entry maps to. Therefore wrap grid point
                    // indices around to account boundary conditions.
                    int gpi = i + (ii - N2); // grid point index mapped to by stencil entry in x-direction
                    int gpj = j + (jj - N2); // grid point index mapped to by stencil entry in y-direction
                    int gpk = k + (kk - N2); // grid point index mapped to by stencil entry in z-direction

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

                    if (gpi >= 0 && gpi < m &&
                        gpj >= 0 && gpj < n &&
                        gpk >= 0 && gpk < o)
                    {
                        // row: current grid point
                        // column: grid point the current stencil entry maps to (dependent on current grid point)
                        c[i * n * o + j * o + k][gpi * n * o + gpj * o + gpk] +=
                            s[i + s.getGhostsDim1()][j + s.getGhostsDim2()][k + s.getGhostsDim3()][ii][jj][kk];
                    }
                }
            // clang-format on
            return c;
        }
    };

    std::ostream& operator<<(std::ostream& os, Matrix2d const& value);

}

#endif // MATRIX2D_HPP
