/*
 *  cuboid.c
 *  mgab
 *
 *  Created by Matthias Bolten on 08.08.06.
 *  Copyright 2006 Matthias Bolten. All rights reserved.
 *
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif
#include "cuboid.hpp"
#include <stdio.h>
#include <stdlib.h>

namespace mgcl
{

    double ***cuboid_alloc(int m, int n, int o)
    {
        double *field;
        double ***cuboid;
        int i, j;

        field = (double *)calloc(sizeof(double), m * n * o);
        cuboid = (double ***)malloc(sizeof(double **) * m);
        for (i = 0; i < m; i++)
        {
            cuboid[i] = (double **)malloc(sizeof(double *) * n);
            for (j = 0; j < n; j++)
            {
                cuboid[i][j] = &field[i * n * o + j * o];
            }
        }

        return cuboid;
    }

    double ***cube_alloc(int n)
    {
        return cuboid_alloc(n, n, n);
    }

    void cuboid_free(double ***cuboid, int m, int n, int o)
    {
        if (cuboid == NULL)
            return;

        int i;

        free(cuboid[0][0]);
        for (i = 0; i < m; i++)
        {
            free(cuboid[i]);
        }
        free(cuboid);
        cuboid = NULL;

        return;
    }

    void cube_free(double ***cube, int n)
    {
        cuboid_free(cube, n, n, n);

        return;
    }

    Cuboid::Cuboid(int m_, int n_, int o_)
        : m(m_),
          n(n_),
          o(o_)
    {
        int i, j;

        field_1d = (double *)calloc(sizeof(double), m * n * o);
        field_3d = (double ***)malloc(sizeof(double **) * m);
        for (i = 0; i < m; i++)
        {
            field_3d[i] = (double **)malloc(sizeof(double *) * n);
            for (j = 0; j < n; j++)
            {
                field_3d[i][j] = &field_1d[i * n * o + j * o];
            }
        }
    }

    Cuboid::~Cuboid()
    {
        int i;

        free(field_3d[0][0]);
        for (i = 0; i < m; i++)
        {
            free(field_3d[i]);
        }
        free(field_3d);
        field_3d = nullptr;
        field_1d = nullptr;
    }

    int Cuboid::getO() const
    {
        return o;
    }

    int Cuboid::getN() const
    {
        return n;
    }

    double ***Cuboid::getData() const
    {
        return field_3d;
    }

    int Cuboid::getM() const
    {
        return m;
    }

    double **Cuboid::operator[](int index)
    {
        return field_3d[index];
    }
}
