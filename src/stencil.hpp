#ifndef MGCL_STENCIL_HPP
#define MGCL_STENCIL_HPP

#include <algorithm>
#include <cstddef>
#include <string>

#ifndef CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#endif // CL_USE_DEPRECATED_OPENCL_1_2_APIS

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif // CL_TARGET_OPENCL_VERSION

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "cuboid.hpp"
#include "hypercube.hpp"
#include "opencl_helper.hpp"

namespace mgcl
{
    // forward declarations
    class VaryingStencil;
    class FixedStencilGpu;
    class MPILevelData;

    /**
     * @brief Fixed 3x3x3 stencil (same stencil entries for each grid point).
     */
    class FixedStencil : public Cuboid
    {
    public:
        explicit FixedStencil(int _width);
        inline int getWidth() const { return m; }
        VaryingStencil multiply(VaryingStencil& b, int ghc,
                                MPILevelData* mpiData, bool periodic, bool forceLocal) const;
    };

    /**
     * @brief Class for NxNxN varying stencils, i.e. stencil can differ for each grid point.
     * Choose N = 3 to include only direct neighbors, N = 5 to include 2 nearest neighbors, etc.
     * If two stencils of size NxNxN and NBxNBxNB get multiplied with each other, the resulting stencil has size
     * (N + NB - 1)^3.
     *
     */
    class VaryingStencil : public Hypercube6d
    {
    public:
        VaryingStencil(int m, int n, int o, int _width, int ghosts_m, int ghosts_n, int ghosts_o);
        VaryingStencil(VaryingStencil&) = delete;
        VaryingStencil& operator=(const VaryingStencil&) = delete;
        VaryingStencil(VaryingStencil&& o) : Hypercube6d(std::move(o)) {}
        VaryingStencil& operator=(VaryingStencil&& o)
        {
            Hypercube6d::operator=(std::move(o));
            return *this;
        }
        inline int getWidth() const { return dim4; }
        void updateGhosts();
        VaryingStencil multiply(FixedStencil& b, int ghc,
                                MPILevelData* mpiData, bool periodic, bool forceLocal) const;
        VaryingStencil multiply(VaryingStencil& b, int ghc,
                                MPILevelData* mpiData, bool periodic, bool forceLocal) const;
        VaryingStencil& operator*(double factor);
        std::unique_ptr<VaryingStencil> slice(int m_start, int m_end, int n_start, int n_end,
                                              int o_start, int o_end,
                                              int ghm = -1, int ghn = -1, int gho = -1);
        std::unique_ptr<VaryingStencil> sliceIncGhosts(int m_start, int m_end, int n_start, int n_end,
                                                       int o_start, int o_end);
        std::unique_ptr<VaryingStencil> copyShallow();
    };

    FixedStencil create3dFullWeightRestrictionStencil();
    FixedStencil create3dBilinearProlongationStencil();

    /**
     * @brief Wrapper class for a varying stencil gpu buffer. Stores additional information like width of the grid,
     * width of the stencil and amount of ghost cells. The device buffer gets automatically created in the constructor
     * and gets released in the destructor.
     *
     */
    class VaryingStencilGpu
    {
    private:
        int m;
        int n;
        int o;
        int width;
        int gh;
        cl_mem buf = nullptr;

    public:
        VaryingStencilGpu(int m_, int n_, int o_, int width_, int gh_, cl_context context, cl_command_queue queue);
        VaryingStencilGpu(VaryingStencilGpu&&);
        VaryingStencilGpu& operator=(VaryingStencilGpu&&);
        ~VaryingStencilGpu();

        VaryingStencilGpu(const VaryingStencilGpu&) = delete;
        VaryingStencilGpu& operator=(const VaryingStencilGpu&) = delete;

        void fill(VaryingStencil& f, cl_command_queue queue);
        VaryingStencil read(cl_command_queue queue);

        void updateGhosts(cl_program program, cl_command_queue queue);
        VaryingStencilGpu multiply(VaryingStencilGpu& b, int ghc,
                                   cl_program program, cl_command_queue queue, cl_context context);
        VaryingStencilGpu multiply(FixedStencilGpu& b, int ghc,
                                   cl_program program, cl_command_queue queue, cl_context context);

        VaryingStencilGpu cutFromW7ToW3(cl_program program, cl_command_queue queue, cl_context context);

        int getM() const;
        int getN() const;
        int getO() const;
        int getWidth() const;
        int getGh() const;
        cl_mem getBuf() const;
    };

    /**
     * @brief Wrapper class for a fixed stencil gpu buffer. Stores additional information like width of the stencil
     *  and amount of ghost cells.
     *
     */
    class FixedStencilGpu
    {
    private:
        int width;
        cl_mem buf = nullptr;

    public:
        FixedStencilGpu(int width_, cl_context context, cl_command_queue queue);
        FixedStencilGpu(FixedStencilGpu&&);
        FixedStencilGpu& operator=(FixedStencilGpu&&);
        ~FixedStencilGpu();

        FixedStencilGpu(const FixedStencilGpu&) = delete;
        FixedStencilGpu& operator=(const FixedStencilGpu&) = delete;

        void fill(FixedStencil& f, cl_command_queue queue);
        FixedStencil read(cl_command_queue queue);

        VaryingStencilGpu multiply(VaryingStencilGpu& b, int ghc,
                                   cl_program program, cl_command_queue queue, cl_context context);

        int getWidth() const;
        cl_mem getBuf() const;
    };

    FixedStencilGpu create3dFullWeightRestrictionStencilGpu(cl_context context, cl_command_queue queue);
    FixedStencilGpu create3dBilinearProlongationStencilGpu(cl_context context, cl_command_queue queue);
}

#endif // MGCL_STENCIL_HPP
