#pragma once

#include <CL/cl.h>

namespace mgcl
{
    class Level
    {
    private:
        /* Number of level in given Problem. */
        int num;

        /* solution, right hand side and residual vectors */
        double ***v = nullptr;
        double ***f = nullptr;
        double ***r = nullptr;

        /* grid dimensions of ghosted grid */
        int m;
        int n;
        int o;

        /* spacing of real grid on current level */
        double h;

        /* Stencil values per grid point. Must be given by the user if a varying symmetric stencil shall be used.
         * While mgcl_conf::stencil_values should contain only values for real grid points, mgcl_level_data::stencil_values
         * contains values of ghosted points per level, too.
         * Size depends on stencil type:
         * -  7-point varsym: size = (m + 2*ghosts)*(n + 2*ghosts)*(o + 2*ghosts) * 4
         * - 19-point varsym: size = (m + 2*ghosts)*(n + 2*ghosts)*(o + 2*ghosts) * 7
         * - 27-point varsym: size = (m + 2*ghosts)*(n + 2*ghosts)*(o + 2*ghosts) * 8 */
        double ***stencil_values = nullptr;

        /* opencl buffers */
        cl_mem d_v_in = nullptr;
        cl_mem d_v_out = nullptr;
        cl_mem d_f = nullptr;
        cl_mem d_r = nullptr;
        cl_mem d_stencil_values = nullptr;

    public:
        Level(int _num, int _m, int _n, int _o)
            : num(_num), m(_m), n(_n), o(_o) {}

        Level(int _num, int _m, int _n, int _o, double ***_v, double ***_f)
            : num(_num), m(_m), n(_n), o(_o), v(_v), f(_f) {}

        Level(int _num, int _m, int _n, int _o, double ***_v, double ***_f, double ***_stencil_values)
            : num(_num), m(_m), n(_n), o(_o), v(_v), f(_f), stencil_values(_stencil_values) {}

        Level(int _num, int _m, int _n, int _o, cl_mem _d_v, cl_mem _d_f)
            : num(_num), m(_m), n(_n), o(_o), d_v_in(_d_v), d_f(_d_f) {}

        Level(int _num, int _m, int _n, int _o, cl_mem _d_v, cl_mem _d_f, cl_mem _d_stencil_values)
            : num(_num), m(_m), n(_n), o(_o), d_v_in(_d_v), d_f(_d_f), d_stencil_values(_d_stencil_values) {}

        int getNum() const;

        double ***getV() const;
        void setV(double ***v_);

        double ***getF() const;
        void setF(double ***f_);

        double ***getR() const;
        void setR(double ***r_);

        double ***getStencilValues() const;
        void setStencilValues(double ***stencilValues);

        int getM() const;

        int getN() const;

        int getO() const;

        double getH() const;
        void setH(double h_);
    };

    inline int Level::getNum() const
    {
        return num;
    }

    inline double ***Level::getF() const
    {
        return f;
    }

    inline void Level::setF(double ***f_)
    {
        f = f_;
    }

    inline double ***Level::getStencilValues() const
    {
        return stencil_values;
    }

    inline int Level::getN() const
    {
        return n;
    }

    inline void Level::setStencilValues(double ***stencilValues)
    {
        stencil_values = stencilValues;
    }

    inline int Level::getO() const
    {
        return o;
    }

    inline double ***Level::getV() const
    {
        return v;
    }

    inline void Level::setV(double ***v_)
    {
        v = v_;
    }

    inline double ***Level::getR() const
    {
        return r;
    }

    inline void Level::setR(double ***r_)
    {
        r = r_;
    }

    inline int Level::getM() const
    {
        return m;
    }

    inline double Level::getH() const
    {
        return h;
    }

    inline void Level::setH(double h_)
    {
        h = h_;
    }
}
