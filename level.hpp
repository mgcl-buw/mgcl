#pragma once

#include "cuboid.hpp"
#include "multigrid_engine.hpp"
#include "opencl_helper.hpp"
#include "stencil.hpp"

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
        std::shared_ptr<Cuboid> v = nullptr;
        std::shared_ptr<Cuboid> f = nullptr;
        std::shared_ptr<Cuboid> r = nullptr;

        /* Stencil for this Level that will be applied on v */
        std::shared_ptr<Stencil> stencil = nullptr;

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

        /* opencl buffers */
        cl_mem dVIn = nullptr;
        cl_mem dVOut = nullptr;
        cl_mem dF = nullptr;
        cl_mem dR = nullptr;

        friend class OpenCLHelper;
        friend class MultigridEngine;
        friend class Problem;

    public:
        Level(Problem *problem_, int _num);
        Level(const Level &) = delete;
        Level &operator=(const Level &) = delete;
        Level(const Level &&) = delete;
        Level &operator=(Level &&) = delete;
        ~Level();

        bool init();
        int initOpenCLBuffers();

        int getNum() const;

        Cuboid &getV() const;
        std::shared_ptr<Cuboid> getVPtr() const;
        void setV(const std::shared_ptr<Cuboid> &v_);
        Cuboid &getF() const;
        std::shared_ptr<Cuboid> getFPtr() const;
        void setF(const std::shared_ptr<Cuboid> &f_);
        Cuboid &getR() const;
        std::shared_ptr<Cuboid> getRPtr() const;
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

        int getMgh() const;

        int getNgh() const;

        int getOgh() const;

        std::shared_ptr<Stencil> getStencil() const;
        void setStencil(const std::shared_ptr<Stencil> &stencil_);
    };
}
