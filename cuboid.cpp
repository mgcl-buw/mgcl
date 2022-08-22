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

        field_1d.resize(m * n * o);
        field_3d = new double **[m];
        for (i = 0; i < m; i++)
        {
            field_3d[i] = new double *[n];
            for (j = 0; j < n; j++)
            {
                field_3d[i][j] = &field_1d[i * n * o + j * o];
            }
        }
    }

    /**
     * @brief Construct a new Cuboid object allocating memory for field_3d.
     *
     * @param c Cuboid to be cloned from.
     */
    Cuboid::Cuboid(const Cuboid &c)
        : m(c.m), n(c.n), o(c.o), field_1d(c.field_1d)
    {
        field_3d = new double **[m];
        for (int i = 0; i < m; i++)
        {
            field_3d[i] = new double *[n];
            for (int j = 0; j < n; j++)
            {
                field_3d[i][j] = &field_1d[i * n * o + j * o];
            }
        }
    }

    Cuboid::~Cuboid()
    {
        for (int i = 0; i < m; i++)
        {
            delete[] field_3d[i];
        }
        delete[] field_3d;
        field_3d = nullptr;
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
