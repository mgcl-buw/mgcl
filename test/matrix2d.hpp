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

        Matrix2d kronecker(Matrix2d &b);
        Matrix2d operator+(Matrix2d &b);
        void operator+=(Matrix2d &b);

        std::vector<double> &operator[](int index);
        std::vector<double> &at(int index);

        int getM() const;
        int getN() const;

        static Matrix2d eye(int n);
        static Matrix2d diag(double value, int n, int off);
    };
}

#endif // MATRIX2D_HPP
