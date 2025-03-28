#ifndef MGCL_CUBOID_BS__HPP
#define MGCL_CUBOID_BS__HPP

#include <memory>
#include <string>
#include <vector>

namespace mgcl
{
    /**
     * @brief Class for storing vector valued cuboids, i.e. each grid point has a vector.
     * Layout was chosen to be grid points first, i.e. [gpx][gpy][gpz][b], due to benchmark results, especially
     * for residual.
     *
     */
    class CuboidBS
    {
        struct Index4d
        {
            size_t i, j, k, b;
        };

    protected:
        int m;
        int n;
        int o;
        int mgh;
        int ngh;
        int ogh;
        int ghostsM = 0;
        int ghostsN = 0;
        int ghostsO = 0;
        int blocksize;
        std::vector<double> field_1d;
        double**** field_4d;

    public:
        CuboidBS(int m_, int n_, int o_, int blocksize);
        CuboidBS(int m_, int n_, int o_, int blocksize, double value);
        CuboidBS(int m_, int n_, int o_, int ghostsM_, int ghostsN_, int ghostsO_, int blocksize);
        CuboidBS(int m_, int n_, int o_, int ghostsM_, int ghostsN_, int ghostsO_, int blocksize, double value);
        CuboidBS(const CuboidBS&) = delete;
        CuboidBS& operator=(const CuboidBS&) = delete;
        CuboidBS(CuboidBS&&);
        CuboidBS& operator=(CuboidBS&&);
        ~CuboidBS();

        void fillRandom(double low = 0, double high = 1, bool realCellsOnly = false);
        void fillRandomInt(int low = 1, int high = 10, bool realCellsOnly = false);
        void fill(double value, bool realCellsOnly = false);
        void fill1dIndex(bool realCellsOnly);
        bool isEqual(CuboidBS& c, double tol = 1e-7);
        bool isEqualAllCells(CuboidBS& c, double tol = 1e-7);
        void dumpToFile(std::string path, bool realCellsOnly = false);
        void fillRealFrom(CuboidBS& c);
        void fillAllFrom(CuboidBS& c);
        std::unique_ptr<CuboidBS> slice(int m_start, int m_end, int n_start, int n_end, int o_start, int o_end,
                                        int ghm = -1, int ghn = -1, int gho = -1);
        std::unique_ptr<CuboidBS> sliceIncGhosts(int m_start, int m_end, int n_start, int n_end,
                                                 int o_start, int o_end);
        std::unique_ptr<CuboidBS> copyShallow();

        inline double*** operator[](int index) { return field_4d[index]; }

        inline size_t to1dIndex(int i, int j, int k, int b) { return i * blocksize * ngh * ogh + j * blocksize * ogh + k * blocksize + b; }

        inline std::vector<double>& field1d() { return field_1d; };
        inline double**** getData() const { return field_4d; };
        inline int getM() const { return m; };
        inline int getN() const { return n; };
        inline int getO() const { return o; };
        inline int getGhostsM() const { return ghostsM; };
        inline int getGhostsN() const { return ghostsN; };
        inline int getGhostsO() const { return ghostsO; };
        inline int getMgh() const { return mgh; };
        inline int getNgh() const { return ngh; };
        inline int getOgh() const { return ogh; };
        inline int getSize() const { return blocksize * mgh * ngh * ogh; };
        inline int getBlocksize() const { return blocksize; };

        static CuboidBS copyFrom(CuboidBS& c);
    };
}
#endif /* ifndef MGCL_CUBOID_BS__HPP */
