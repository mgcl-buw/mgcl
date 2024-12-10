#ifndef MGCL_LEVEL_HPP
#define MGCL_LEVEL_HPP

#include "cuboid_gpu.hpp"
#include "mgcl.hpp" // for MGCL_STENCIL
#include "mpi_level_data.hpp"
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
        std::shared_ptr<VaryingStencil> stencilValues = nullptr;
        std::shared_ptr<VaryingStencilGpu> stencilValuesGpu = nullptr;
        std::shared_ptr<FixedStencil> fixedStencil = nullptr;
        std::shared_ptr<FixedStencilGpu> fixedStencilGpu = nullptr;

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
        std::shared_ptr<CuboidGpu> dVIn = nullptr;
        std::shared_ptr<CuboidGpu> dVOut = nullptr;
        std::shared_ptr<CuboidGpu> dF = nullptr;
        std::shared_ptr<CuboidGpu> dR = nullptr;
        std::shared_ptr<CuboidGpu> dRsq = nullptr; // temporary buffer for storing the squared residual

        /* MPI relevant data, e.g. neighbour process ranks. Null if Problem::useMpi is false. */
        std::unique_ptr<MPILevelData> mpiData = nullptr;

        friend class OpenCLHelper;
        friend class MultigridEngine;
        friend class Problem;

    public:
        Level(Problem* problem_, int _num);
        Level(const Level&) = delete;
        Level& operator=(const Level&) = delete;
        Level(const Level&&) = delete;
        Level& operator=(Level&&) = delete;
        ~Level(){};

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

        MPILevelData* getMpiDataPtr();
        MPILevelData& getMpiData();

        CuboidGpu* getDVInPtr() const;
        CuboidGpu& getDVIn() const;
        void setDVIn(const std::shared_ptr<CuboidGpu> dVIn_);

        CuboidGpu* getDVOutPtr() const;
        CuboidGpu& getDVOut() const;
        void setDVOut(const std::shared_ptr<CuboidGpu> dVOut_);

        CuboidGpu* getDFPtr() const;
        CuboidGpu& getDF() const;
        void setDF(const std::shared_ptr<CuboidGpu> dF_);

        CuboidGpu* getDRPtr() const;
        CuboidGpu& getDR() const;
        void setDR(const std::shared_ptr<CuboidGpu> dR_);

        CuboidGpu* getDRsqPtr() const;
        CuboidGpu& getDRsq() const;
        void setDRsq(const std::shared_ptr<CuboidGpu> dR_);

        int getMgh() const;

        int getNgh() const;

        int getOgh() const;

        MGCL_STENCIL getStencilType() const;

        std::shared_ptr<VaryingStencil>& getStencilValues();
        std::shared_ptr<FixedStencil>& getFixedStencil();

        double getStencilFactor() const;

        std::shared_ptr<VaryingStencilGpu>& getStencilValuesGpu();
        void setStencilValuesGpu(std::shared_ptr<VaryingStencilGpu> sv);
        std::shared_ptr<FixedStencilGpu>& getFixedStencilGpu();
        void setFixedStencilGpu(std::shared_ptr<FixedStencilGpu> sv);

        bool isCalculatedLocally() const;

        friend std::ostream& operator<<(std::ostream& os, const Level& lv);
    };
}

#endif // MGCL_LEVEL_HPP
