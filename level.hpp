#pragma once

#include "cuboid.hpp"
#include "multigrid_engine.hpp"
#include "opencl_helper.hpp"

namespace mgcl
{
    // forward declarations
    class Problem;

    class Level
    {
    private:
        /* Problem that this Level belongs to */
        Problem *problem;

        /* Number of level in given Problem. */
        int num;

        /* solution, right hand side and residual vectors */
        std::shared_ptr<Cuboid> v;
        std::shared_ptr<Cuboid> f;
        std::shared_ptr<Cuboid> r;

        /* grid dimensions of real grid */
        int m;
        int n;
        int o;

        /* grid dimensions of ghosted grid */
        int mgh;
        int ngh;
        int ogh;

        /* spacing of real grid on current level */
        // TODO not used yet
        double h;

        /* Stencil values per grid point. Must be given by the user if a varying symmetric stencil shall be used.
         * While mgcl_conf::stencil_values should contain only values for real grid points, mgcl_level_data::stencil_values
         * contains values of ghosted points per level, too.
         * Size depends on stencil type:
         * -  7-point varsym: size = (m + 2*ghosts)*(n + 2*ghosts)*(o + 2*ghosts) * 4
         * - 19-point varsym: size = (m + 2*ghosts)*(n + 2*ghosts)*(o + 2*ghosts) * 7
         * - 27-point varsym: size = (m + 2*ghosts)*(n + 2*ghosts)*(o + 2*ghosts) * 8 */
        std::shared_ptr<Cuboid> stencil_values;

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

        Cuboid &getV();
        std::shared_ptr<Cuboid> getVPtr();
        void setV(const std::shared_ptr<Cuboid> &v_);
        Cuboid &getF();
        std::shared_ptr<Cuboid> getFPtr();
        void setF(const std::shared_ptr<Cuboid> &f_);
        Cuboid &getR();
        std::shared_ptr<Cuboid> getRPtr();
        void setR(const std::shared_ptr<Cuboid> &r_);

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

        int getMgh() const;

        int getNgh() const;

        int getOgh() const;

        Cuboid &getStencilValues() const;
        std::shared_ptr<Cuboid> getStencilValuesPtr() const;
        void setStencilValues(const std::shared_ptr<Cuboid> &stencilValues);
    };
}
