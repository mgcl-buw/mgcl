#ifndef MGCL_LEVEL_HPP
#define MGCL_LEVEL_HPP

#include "mgcl.hpp" // for MGCL_STENCIL
#include "mpi_data.hpp"
#include "stencil.hpp" // for VaryingStencil3x3x3

#include <memory> // for shared_ptr
#include <ostream>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

namespace mgcl
{
    // forward declarations
    class Problem;
    class Cuboid;

    class Level
    {
    private:
        /* Problem that this Level belongs to */
        Problem* problem;

        /* Number of level in given Problem. */
        int num;

        /* solution, right hand side and residual vectors */
        std::shared_ptr<Cuboid> v = nullptr;
        std::shared_ptr<Cuboid> f = nullptr;
        std::shared_ptr<Cuboid> r = nullptr;

        /* Stencil for this Level that will be applied on v */
        MGCL_STENCIL stencilType;
        double stencilFactor = 1;
        std::shared_ptr<VaryingStencil3x3x3> stencilValues = nullptr;
        std::shared_ptr<VaryingStencilGpu> stencilValuesGpu = nullptr;

        /* grid dimensions of local real grid */
        int m;
        int n;
        int o;

        /* grid dimensions of local ghosted grid */
        int mgh;
        int ngh;
        int ogh;

        /* spacing of real grid on current level */
        double h;

        /* opencl buffers */
        cl_mem dVIn = nullptr;
        cl_mem dVOut = nullptr;
        cl_mem dF = nullptr;
        cl_mem dR = nullptr;

        /* MPI relevant data, e.g. neighbour process ranks. Null if Problem::useMpi is false. */
        std::unique_ptr<MPIData> mpiData = nullptr;

        friend class OpenCLHelper;
        friend class MultigridEngine;
        friend class Problem;

    public:
        Level(Problem* problem_, int _num);
        Level(const Level&) = delete;
        Level& operator=(const Level&) = delete;
        Level(const Level&&) = delete;
        Level& operator=(Level&&) = delete;
        ~Level();

        bool init();
        int initOpenCLBuffers();
        int initMpiData();

        int getNum() const;

        Cuboid& getV() const;
        std::shared_ptr<Cuboid> getVPtr() const;
        void setV(const std::shared_ptr<Cuboid>& v_);
        Cuboid& getF() const;
        std::shared_ptr<Cuboid> getFPtr() const;
        void setF(const std::shared_ptr<Cuboid>& f_);
        Cuboid& getR() const;
        std::shared_ptr<Cuboid> getRPtr() const;
        void setR(const std::shared_ptr<Cuboid>& r_);

        int getM() const;
        int getN() const;
        int getO() const;

        double getH() const;
        void setH(double h_);

        MPIData* getMpiDataPtr();
        MPIData& getMpiData();

        cl_mem getDVIn() const;
        void setDVIn(const cl_mem dVIn_);

        cl_mem getDVOut() const;
        void setDVOut(const cl_mem dVOut_);

        cl_mem getDF() const;
        void setDF(const cl_mem dF_);

        cl_mem getDR() const;
        void setDR(const cl_mem dR_);

        int getMgh() const;

        int getNgh() const;

        int getOgh() const;

        MGCL_STENCIL getStencilType() const;

        std::shared_ptr<VaryingStencil3x3x3>& getStencilValues();

        double getStencilFactor() const;

        std::shared_ptr<VaryingStencilGpu>& getStencilValuesGpu();
        void setStencilValuesGpu(std::shared_ptr<VaryingStencilGpu> sv);

        bool isCalculatedLocally() const;

        friend std::ostream& operator<<(std::ostream& os, const Level& lv);
    };
}

#endif // MGCL_LEVEL_HPP
