#pragma once

#include <string>
#include <vector>

namespace mgcl
{
    class Hypercube4d
    {
    private:
        int dim1;
        int dim2;
        int dim3;
        int dim4;
        int dim1gh;
        int dim2gh;
        int dim3gh;
        int dim4gh;
        int ghostsDim1 = 0;
        int ghostsDim2 = 0;
        int ghostsDim3 = 0;
        int ghostsDim4 = 0;
        std::vector<double> field_1d;
        double ****field_4d;

    public:
        Hypercube4d(int dim1_, int dim2_, int dim3_, int dim4_, double value = 0);
        Hypercube4d(int dim1_, int dim2_, int dim3_, int dim4_, int ghostsDim1_, int ghostsDim2_, int ghostsDim3_, int ghostsDim4_, double value = 0);
        Hypercube4d(const Hypercube4d &c);
        ~Hypercube4d();

        double ****getData() const;
        double ***operator[](int index);
        void fillRandom(double low = 0, double high = 1);
        void fill(double value, bool realCellsOnly = false);
        std::vector<double> &field1d();
        bool isEqual(Hypercube4d &c, double tol = 1e-7, bool printDiffs = true);
        void dumpToFile(std::string path);

        int getDim1() const;
        int getDim2() const;
        int getDim3() const;
        int getDim4() const;
        int getDim1gh() const;
        int getDim2gh() const;
        int getDim3gh() const;
        int getDim4gh() const;
        int getGhostsDim1() const;
        int getGhostsDim2() const;
        int getGhostsDim3() const;
        int getGhostsDim4() const;
    };
}
