#ifndef MGCL_CUBOID__HPP
#define MGCL_CUBOID__HPP

#include <string>
#include <vector>

namespace mgcl
{
    class Cuboid
    {
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
        std::vector<double> field_1d;
        double ***field_3d;

    public:
        Cuboid(int m_, int n_, int o_, double value = 0);
        Cuboid(int m_, int n_, int o_, int ghostsM_, int ghostsN_, int ghostsO_, double value = 0);
        Cuboid(const Cuboid &) = delete;
        Cuboid &operator=(const Cuboid &) = delete;
        Cuboid(Cuboid &&);
        Cuboid &operator=(Cuboid &&);
        ~Cuboid();

        double ***getData() const;
        inline double **operator[](int index) { return field_3d[index]; }
        void fillRandom(double low = 0, double high = 1);
        void fillRandomInt(int low = 1, int high = 10, bool realCellsOnly = false);
        void fill(double value, bool realCellsOnly = false);
        std::vector<double> &field1d();
        bool isEqual(Cuboid &c, double tol = 1e-7, bool printDiffs = false);
        void dumpToFile(std::string path);
        void fillRealFrom(Cuboid &c);

        int getO() const;
        int getN() const;
        int getM() const;
        int getGhostsO() const;
        int getGhostsN() const;
        int getGhostsM() const;
        int getOgh() const;
        int getNgh() const;
        int getMgh() const;

        static Cuboid copyFrom(Cuboid &c);
    };
}
#endif /* ifndef MGCL_CUBOID__HPP */
