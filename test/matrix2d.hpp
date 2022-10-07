#ifndef MATRIX2D_HPP
#define MATRIX2D_HPP

#include <algorithm>
#include <vector>

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

        std::vector<double> &operator[](int index);
        const std::vector<double> &operator[](int index) const;
        std::vector<double> &at(int index);
        const std::vector<double> &at(int index) const;

        int getM() const;
        int getN() const;

        static Matrix2d eye(int m);
        static Matrix2d eye(int m, int n);
        static Matrix2d diag(std::vector<std::tuple<double, int>> valuesAndOffsets, int m, int n);
    };
}

#endif // MATRIX2D_HPP
