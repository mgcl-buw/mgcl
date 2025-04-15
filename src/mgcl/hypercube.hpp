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
        double**** field_4d;

    public:
        Hypercube4d(int dim1_, int dim2_, int dim3_, int dim4_, double value = 0);
        Hypercube4d(int dim1_, int dim2_, int dim3_, int dim4_, int ghostsDim1_, int ghostsDim2_, int ghostsDim3_, int ghostsDim4_, double value = 0);
        Hypercube4d(const Hypercube4d& c) = delete;
        Hypercube4d& operator=(const Hypercube4d&) = delete;
        Hypercube4d(const Hypercube4d&&) = delete;
        Hypercube4d& operator=(const Hypercube4d&&) = delete;
        ~Hypercube4d();

        double**** getData() const;
        double*** operator[](int index);
        void fillRandom(double low = 0, double high = 1);
        void fill(double value, bool realCellsOnly = false);
        std::vector<double>& field1d();
        bool isEqual(Hypercube4d& c, double tol = 1e-7, bool printDiffs = true);
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
        double****** field_6d;

    public:
        Hypercube6d(int dim1_, int dim2_, int dim3_, int dim4_, int dim5_, int dim6_, double value = 0);
        Hypercube6d(int dim1_, int dim2_, int dim3_, int dim4_, int dim5_, int dim6_,
                    int ghostsDim1_, int ghostsDim2_, int ghostsDim3_, int ghostsDim4_,
                    int ghostsDim5_, int ghostsDim6_, double value = 0);
        Hypercube6d(const Hypercube6d& c) = delete;
        Hypercube6d& operator=(const Hypercube6d&) = delete;
        Hypercube6d(Hypercube6d&&);
        Hypercube6d& operator=(Hypercube6d&&);
        ~Hypercube6d();

        double****** getData() const;
        double***** operator[](int index) const;
        void fillRandom(double low = 0, double high = 1);
        void fillRandomInt(int low = 1, int high = 10, bool realCellsOnly = false);
        void fill(double value, bool realCellsOnly = false);
        void fill1dIndex(bool realCellsOnly);
        std::vector<double>& field1d();
        bool isEqual(Hypercube6d& c, double tol = 1e-7, bool printDiffs = false);
        void dumpToFile(std::string path, bool realCellsOnly = false);
        void dumpToFileMatlab(std::string path, std::string varname, bool realCellsOnly = false);
        void copyRealFrom(Hypercube6d& o);

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
        inline size_t getSize() const { return dim1gh * dim2gh * dim3gh * dim4gh * dim5gh * dim6gh; }
    };

    /**
     * @brief Class for a 6d hypercube that has an underlying 1d double vector
     *
     */
    class Hypercube8d
    {
    protected:
        int dim1;
        int dim2;
        int dim3;
        int dim4;
        int dim5;
        int dim6;
        int dim7;
        int dim8;
        int dim1gh;
        int dim2gh;
        int dim3gh;
        int dim4gh;
        int dim5gh;
        int dim6gh;
        int dim7gh;
        int dim8gh;
        int ghostsDim1 = 0;
        int ghostsDim2 = 0;
        int ghostsDim3 = 0;
        int ghostsDim4 = 0;
        int ghostsDim5 = 0;
        int ghostsDim6 = 0;
        int ghostsDim7 = 0;
        int ghostsDim8 = 0;
        std::vector<double> field_1d;
        double******** field_8d;
        size_t size;

    public:
        Hypercube8d(int dim1_, int dim2_, int dim3_, int dim4_, int dim5_, int dim6_, int dim7_, int dim8_, double value = 0);
        Hypercube8d(int dim1_, int dim2_, int dim3_, int dim4_, int dim5_, int dim6_, int dim7_, int dim8_,
                    int ghostsDim1_, int ghostsDim2_, int ghostsDim3_, int ghostsDim4_,
                    int ghostsDim5_, int ghostsDim6_, int ghostsDim7_, int ghostsDim8_, double value = 0);
        Hypercube8d(const Hypercube8d& c) = delete;
        Hypercube8d& operator=(const Hypercube8d&) = delete;
        Hypercube8d(Hypercube8d&&);
        Hypercube8d& operator=(Hypercube8d&&);
        ~Hypercube8d();

        double******** getData() const;
        double******* operator[](int index) const;
        void fillRandom(double low = 0, double high = 1);
        void fillRandomInt(int low = 1, int high = 10, bool realCellsOnly = false);
        void fill(double value, bool realCellsOnly = false);
        void fill1dIndex(bool realCellsOnly);
        std::vector<double>& field1d();
        bool isEqual(Hypercube8d& c, double tol = 1e-7);
        bool isEqualIncGhosts(Hypercube8d& c, double tol = 1e-7);
        void dumpToFile(std::string path, bool realCellsOnly = false);
        void dumpToFileMatlab(std::string path, std::string varname, bool realCellsOnly = false);
        void copyRealFrom(Hypercube8d& o);

        inline int getDim1() const { return dim1; }
        inline int getDim2() const { return dim2; }
        inline int getDim3() const { return dim3; }
        inline int getDim4() const { return dim4; }
        inline int getDim5() const { return dim5; }
        inline int getDim6() const { return dim6; }
        inline int getDim7() const { return dim7; }
        inline int getDim8() const { return dim8; }
        inline int getDim1gh() const { return dim1gh; }
        inline int getDim2gh() const { return dim2gh; }
        inline int getDim3gh() const { return dim3gh; }
        inline int getDim4gh() const { return dim4gh; }
        inline int getDim5gh() const { return dim5gh; }
        inline int getDim6gh() const { return dim6gh; }
        inline int getDim7gh() const { return dim7gh; }
        inline int getDim8gh() const { return dim8gh; }
        inline int getGhostsDim1() const { return ghostsDim1; }
        inline int getGhostsDim2() const { return ghostsDim2; }
        inline int getGhostsDim3() const { return ghostsDim3; }
        inline int getGhostsDim4() const { return ghostsDim4; }
        inline int getGhostsDim5() const { return ghostsDim5; }
        inline int getGhostsDim6() const { return ghostsDim6; }
        inline int getGhostsDim7() const { return ghostsDim7; }
        inline int getGhostsDim8() const { return ghostsDim8; }
        inline size_t getSize() const { return size; }
    };
}

#endif // MGCL_HYPERCUBE_HPP
