#ifndef MGCL_LEVEL_HPP
#define MGCL_LEVEL_HPP

#include "blockstencil.hpp"
#include "blockstencil_gpu.hpp"
#include "cuboid_bs.hpp"
#include "cuboid_bs_gpu.hpp"
#include "cuboid_gpu.hpp"
#include "mgcl.hpp" // for MGCL_STENCIL
#include "mpi_level_data.hpp"
#include "stencil.hpp" // for VaryingStencil3x3x3
#include "types.hpp"

#include <cassert>
#include <memory> // for shared_ptr
#include <ostream>
#include <variant>

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
        std::shared_ptr<CuboidBS> v_bs = nullptr;
        std::shared_ptr<CuboidBS> f_bs = nullptr;
        std::shared_ptr<CuboidBS> r_bs = nullptr;

        /* Stencil for this Level that will be applied on v */
        MGCL_STENCIL stencilType;
        double stencilFactor = 1;
        std::shared_ptr<VaryingStencil> stencilValues = nullptr;
        std::shared_ptr<VaryingStencilGpu> stencilValuesGpu = nullptr;
        std::shared_ptr<FixedStencil> fixedStencil = nullptr;
        std::shared_ptr<Blockstencil> blockstencil = nullptr;
        std::shared_ptr<BlockstencilGpu> blockstencilGpu = nullptr;
        TBlockstencilInv blockstencilInv;

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
        std::shared_ptr<CuboidBSGpu> dVIn_bs = nullptr;
        std::shared_ptr<CuboidBSGpu> dVOut_bs = nullptr;
        std::shared_ptr<CuboidBSGpu> dF_bs = nullptr;
        std::shared_ptr<CuboidBSGpu> dR_bs = nullptr;
        std::shared_ptr<CuboidBSGpu> dRsq_bs = nullptr; // temporary buffer for storing the squared residual

        /* MPI relevant data, e.g. neighbour process ranks. Null if Problem::useMpi is false. */
        std::shared_ptr<MPILevelData> mpiData = nullptr;

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
        bool initBlockstencil();
        int initOpenCLBuffersBlockstencil();
        int initMpiData();

        int getNum() const;

        void createInverseOfBlockstencilSeq();
        void createInverseOfBlockstencilGpu();

        Cuboid& getV() const;
        std::shared_ptr<Cuboid> getVPtr() const;
        void setV(const std::shared_ptr<Cuboid>& v_);
        Cuboid& getF() const;
        std::shared_ptr<Cuboid> getFPtr() const;
        void setF(const std::shared_ptr<Cuboid>& f_);
        Cuboid& getR() const;
        std::shared_ptr<Cuboid> getRPtr() const;
        void setR(const std::shared_ptr<Cuboid>& r_);

        CuboidBS& getVBS() const;
        std::shared_ptr<CuboidBS> getVBSPtr() const;
        void setVBS(const std::shared_ptr<CuboidBS>& v_);
        CuboidBS& getFBS() const;
        std::shared_ptr<CuboidBS> getFBSPtr() const;
        void setFBS(const std::shared_ptr<CuboidBS>& f_);
        CuboidBS& getRBS() const;
        std::shared_ptr<CuboidBS> getRBSPtr() const;
        void setRBS(const std::shared_ptr<CuboidBS>& r_);

        CuboidBSGpu& getDVBSIn() const;
        std::shared_ptr<CuboidBSGpu> getDVBSInPtr() const;
        void setDVBSIn(const std::shared_ptr<CuboidBSGpu>& v_);
        CuboidBSGpu& getDVBSOut() const;
        std::shared_ptr<CuboidBSGpu> getDVBSOutPtr() const;
        void setDVBSOut(const std::shared_ptr<CuboidBSGpu>& v_);
        CuboidBSGpu& getDFBS() const;
        std::shared_ptr<CuboidBSGpu> getDFBSPtr() const;
        void setDFBS(const std::shared_ptr<CuboidBSGpu>& f_);
        CuboidBSGpu& getDRBS() const;
        std::shared_ptr<CuboidBSGpu> getDRBSPtr() const;
        void setDRBS(const std::shared_ptr<CuboidBSGpu>& r_);
        std::shared_ptr<CuboidBSGpu> getDRsqBSPtr() const;
        CuboidBSGpu& getDRsqBS() const;
        void setDRsqBS(const std::shared_ptr<CuboidBSGpu> dR_);

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
        std::shared_ptr<Blockstencil>& getBlockstencil() { return blockstencil; }
        TBlockstencilInv& getBlockstencilInvVariant() { return blockstencilInv; }
        std::shared_ptr<Blockstencil>& getBlockstencilInvBlock()
        {
            if (std::holds_alternative<std::shared_ptr<Blockstencil>>(blockstencilInv))
            {
                return std::get<std::shared_ptr<Blockstencil>>(blockstencilInv);
            }
            else
            {
                throw "Level::getBlockstencilInvBlock: blockstencilInv does not have type Blockstencil. Check Jacobi type, maybe it's not set to JACOBI_BLOCK?";
            }
        }

        std::shared_ptr<CuboidBS>& getBlockstencilInvScalar()
        {
            if (std::holds_alternative<std::shared_ptr<CuboidBS>>(blockstencilInv))
            {
                return std::get<std::shared_ptr<CuboidBS>>(blockstencilInv);
            }
            else
            {
                throw "Level::getBlockstencilInvScalar: blockstencilInv does not have type CuboidBS. Check Jacobi type, maybe it's not set to JACOBI_SCALAR?";
            }
        }

        double getStencilFactor() const;

        std::shared_ptr<VaryingStencilGpu>& getStencilValuesGpu();
        void setStencilValuesGpu(std::shared_ptr<VaryingStencilGpu> sv);
        inline std::shared_ptr<BlockstencilGpu>& getBlockstencilGpu() { return blockstencilGpu; }
        inline void setBlockstencilGpu(std::shared_ptr<BlockstencilGpu> sv) { blockstencilGpu = sv; }

        std::shared_ptr<BlockstencilGpu>& getBlockstencilGpuInvBlock()
        {
            if (std::holds_alternative<std::shared_ptr<BlockstencilGpu>>(blockstencilInv))
            {
                return std::get<std::shared_ptr<BlockstencilGpu>>(blockstencilInv);
            }
            else
            {
                throw "Level::getBlockstencilGpuInvBlock: blockstencilInv does not have type BlockstencilGpu. Check Jacobi type, maybe it's not set to JACOBI_BLOCK? Or OpenCL is not in use?";
            }
        }

        std::shared_ptr<CuboidBSGpu>& getBlockstencilGpuInvScalar()
        {
            if (std::holds_alternative<std::shared_ptr<CuboidBSGpu>>(blockstencilInv))
            {
                return std::get<std::shared_ptr<CuboidBSGpu>>(blockstencilInv);
            }
            else
            {
                throw "Level::getBlockstencilInvBlock: blockstencilInv does not have type CuboidBSGpu. Check Jacobi type, maybe it's not set to JACOBI_SCALAR? Or OpenCL is not in use?";
            }
        }

        inline void setBlockstencilGpuInvBlock(std::shared_ptr<BlockstencilGpu> sv) { blockstencilInv = sv; }
        inline void setBlockstencilGpuInvScalar(std::shared_ptr<CuboidBSGpu> sv) { blockstencilInv = sv; }

        bool isCalculatedLocally() const;

        friend std::ostream& operator<<(std::ostream& os, const Level& lv);
    };
}

#endif // MGCL_LEVEL_HPP
