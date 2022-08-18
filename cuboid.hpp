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

namespace mgcl
{
    class Cuboid
    {
    private:
        int m;
        int n;
        int o;
        double *field_1d;
        double ***field_3d;

    public:
        Cuboid(int m_, int n_, int o_);
        ~Cuboid();

        int getO() const;
        int getN() const;
        int getM() const;
        double ***getData() const;

        double **operator[](int index);
    };

    double ***cuboid_alloc(int m, int n, int o);
    double ***cube_alloc(int n);

    void cuboid_free(double ***cuboid, int m, int n, int o);
    void cube_free(double ***cuboid, int n);
}
#endif /* ifndef _CUBOID__H_ */
