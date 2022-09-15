/*
 *  cuboid.h
 *  mgab
 *
 *  Created by Matthias Bolten on 08.08.06.
 *  Copyright 2006 Matthias Bolten. All rights reserved.
 *
 */

#ifndef _CUBOID__H_
#define _CUBOID__H_

#include <string>
#include <vector>

namespace mgcl
{
    class Cuboid
    {
    private:
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
        ~Cuboid();

        double ***getData() const;
        double **operator[](int index);
        void fillRandom(double low = 0, double high = 1);
        void fill(double value, bool realCellsOnly = false);
        std::vector<double> &field1d();
        bool isEqual(Cuboid &c, double tol = 1e-7, bool printDiffs = false);
        void dumpToFile(std::string path);

        int getO() const;
        int getN() const;
        int getM() const;
        int getGhostsO() const;
        int getGhostsN() const;
        int getGhostsM() const;
        int getOgh() const;
        int getNgh() const;
        int getMgh() const;
    };

    double ***cuboid_alloc(int m, int n, int o);
    double ***cube_alloc(int n);

    void cuboid_free(double ***cuboid, int m, int n, int o);
    void cube_free(double ***cuboid, int n);
}
#endif /* ifndef _CUBOID__H_ */
