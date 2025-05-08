#ifndef MGCL_MATRIX_HPP
#define MGCL_MATRIX_HPP

#include <cstddef>
#include <string>
#include <vector>
namespace mgcl
{

    /**
     * @brief Class for a 2d matrix that has an underlying 1d double vector
     *
     */
    class Matrix
    {
    protected:
        int dim1;
        int dim2;
        int dim1gh;
        int dim2gh;
        int ghostsDim1 = 0;
        int ghostsDim2 = 0;
        std::vector<double> field_1d;
        double** field_2d;
        size_t size;

    public:
        Matrix(int dim1_, int dim2_, double value = 0);
        Matrix(int dim1_, int dim2_, int ghostsDim1_, int ghostsDim2_, double value = 0);
        Matrix(int dim1_, int dim2_, double* values);
        Matrix(const Matrix& c) = delete;
        Matrix& operator=(const Matrix&) = delete;
        Matrix(Matrix&&);
        Matrix& operator=(Matrix&&);
        ~Matrix();

        void fillRandom(double low = 0, double high = 1);
        void fillRandomInt(int low = 1, int high = 10, bool realCellsOnly = false);
        void fill(double value, bool realCellsOnly = false);
        void fill1dIndex(bool realCellsOnly);
        bool isEqual(Matrix& c, double tol = 1e-7);
        bool isEqualIncGhosts(Matrix& c, double tol = 1e-7);
        void dumpToFile(std::string path, bool realCellsOnly = false);
        void dumpToFileMatlab(std::string path, std::string varname, bool realCellsOnly = false);
        void copyRealFrom(Matrix& o);
        void swapRows(int i, int j);

        inline std::vector<double>& field1d() { return field_1d; }
        inline double** getData() const { return field_2d; }
        inline double* operator[](int index) const { return field_2d[index]; }
        inline int getDim1() const { return dim1; }
        inline int getDim2() const { return dim2; }
        inline int getDim1gh() const { return dim1gh; }
        inline int getDim2gh() const { return dim2gh; }
        inline int getGhostsDim1() const { return ghostsDim1; }
        inline int getGhostsDim2() const { return ghostsDim2; }
        inline size_t getSize() const { return size; }
    };
}

#endif // MGCL_MATRIX_HPP
