#ifndef MGCL_HYPERCUBE_HPP
#define MGCL_HYPERCUBE_HPP

#include <string>
#include <vector>

namespace mgcl
{
    /**
     * @brief Class for a 4d hypercube that has an underlying 1d double vector
     *
     */
    class Hypercube4d
    {
    protected:
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
        Hypercube4d(const Hypercube4d &c) = delete;
        Hypercube4d &operator=(const Hypercube4d &) = delete;
        Hypercube4d(const Hypercube4d &&) = delete;
        Hypercube4d &operator=(const Hypercube4d &&) = delete;
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

    /**
     * @brief Class for a 6d hypercube that has an underlying 1d double vector
     *
     */
    class Hypercube6d
    {
    protected:
        int dim1;
        int dim2;
        int dim3;
        int dim4;
        int dim5;
        int dim6;
        int dim1gh;
        int dim2gh;
        int dim3gh;
        int dim4gh;
        int dim5gh;
        int dim6gh;
        int ghostsDim1 = 0;
        int ghostsDim2 = 0;
        int ghostsDim3 = 0;
        int ghostsDim4 = 0;
        int ghostsDim5 = 0;
        int ghostsDim6 = 0;
        std::vector<double> field_1d;
        double ******field_6d;

    public:
        Hypercube6d(int dim1_, int dim2_, int dim3_, int dim4_, int dim5_, int dim6_, double value = 0);
        Hypercube6d(int dim1_, int dim2_, int dim3_, int dim4_, int dim5_, int dim6_,
                    int ghostsDim1_, int ghostsDim2_, int ghostsDim3_, int ghostsDim4_,
                    int ghostsDim5_, int ghostsDim6_, double value = 0);
        Hypercube6d(const Hypercube6d &c) = delete;
        Hypercube6d &operator=(const Hypercube6d &) = delete;
        Hypercube6d(Hypercube6d &&);
        Hypercube6d &operator=(Hypercube6d &&);
        ~Hypercube6d();

        double ******getData() const;
        double *****operator[](int index);
        void fillRandom(double low = 0, double high = 1);
        void fillRandomInt(int low = 1, int high = 10, bool realCellsOnly = false);
        void fill(double value, bool realCellsOnly = false);
        std::vector<double> &field1d();
        bool isEqual(Hypercube6d &c, double tol = 1e-7, bool printDiffs = true);
        void dumpToFile(std::string path, bool realCellsOnly = false);
        void dumpToFileMatlab(std::string path, std::string varname, bool realCellsOnly = false);

        int getDim1() const;
        int getDim2() const;
        int getDim3() const;
        int getDim4() const;
        int getDim5() const;
        int getDim6() const;
        int getDim1gh() const;
        int getDim2gh() const;
        int getDim3gh() const;
        int getDim4gh() const;
        int getDim5gh() const;
        int getDim6gh() const;
        int getGhostsDim1() const;
        int getGhostsDim2() const;
        int getGhostsDim3() const;
        int getGhostsDim4() const;
        int getGhostsDim5() const;
        int getGhostsDim6() const;
    };
}

#endif // MGCL_HYPERCUBE_HPP
