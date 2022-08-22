#pragma once

#include "multigrid_engine.hpp"
#include "opencl_helper.hpp"
#include "problem.hpp"

namespace mgcl
{
    class Level
    {
    private:
        /* Problem that this Level belongs to */
        Problem *problem;

        /* Number of level in given Problem. */
        int num;

        /* solution, right hand side and residual vectors */
        double ***v = nullptr;
        double ***f = nullptr;
        double ***r = nullptr;

        /* grid dimensions of real grid */
        int m;
        int n;
        int o;

        /* grid dimensions of ghosted grid */
        int mgh;
        int ngh;
        int ogh;

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
        cl_mem dVIn = nullptr;
        cl_mem dVOut = nullptr;
        cl_mem dF = nullptr;
        cl_mem dR = nullptr;
        cl_mem dStencilValues = nullptr;

        friend class OpenCLHelper;
        friend class MultigridEngine;
        friend class Problem;

    public:
        Level(Problem *problem_, int _num, int _m, int _n, int _o);
        ~Level();

        int initOpenCLBuffers();

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

        cl_mem getDVIn() const;
        void setDVIn(const cl_mem &dVIn_);

        cl_mem getDVOut() const;
        void setDVOut(const cl_mem &dVOut_);

        cl_mem getDF() const;
        void setDF(const cl_mem &dF_);

        cl_mem getDR() const;
        void setDR(const cl_mem &dR_);

        cl_mem getDStencilValues() const;
        void setDStencilValues(const cl_mem &dStencilValues_);
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

    inline cl_mem Level::getDVIn() const
    {
        return dVIn;
    }

    inline void Level::setDVIn(const cl_mem &dVIn_)
    {
        dVIn = dVIn_;
    }

    inline cl_mem Level::getDF() const
    {
        return dF;
    }

    inline void Level::setDF(const cl_mem &dF_)
    {
        dF = dF_;
    }

    inline cl_mem Level::getDStencilValues() const
    {
        return dStencilValues;
    }

    inline void Level::setDStencilValues(const cl_mem &dStencilValues_)
    {
        dStencilValues = dStencilValues_;
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

    inline cl_mem Level::getDVOut() const
    {
        return dVOut;
    }

    inline void Level::setDVOut(const cl_mem &dVOut_)
    {
        dVOut = dVOut_;
    }

    inline cl_mem Level::getDR() const
    {
        return dR;
    }

    inline void Level::setDR(const cl_mem &dR_)
    {
        dR = dR_;
    }
}
