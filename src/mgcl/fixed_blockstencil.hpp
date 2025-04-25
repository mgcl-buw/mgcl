#ifndef MGCL_FIXED_BLOCKSTENCIL_HPP
#define MGCL_FIXED_BLOCKSTENCIL_HPP

#include "hypercube.hpp"
#include "mpi_level_data.hpp"

namespace mgcl
{
    /**
     * @brief Class for storing a blockstencil of varying width and blocksize. Each coefficient is a matrix of size
     * blocksize x blocksize. The stencil has the same width x width x width coefficients for all grid points.
     *
     * Memory layout is [bi][bj][ci][cj][ck], where
     * - bi and bj are matrix indices and
     * - ci, cj and ck are coefficient indices
     *
     */
    class FixedBlockstencil : public Hypercube5d
    {
    public:
        FixedBlockstencil(int _width, int blocksize);
        FixedBlockstencil(FixedBlockstencil&) = delete;
        FixedBlockstencil& operator=(const FixedBlockstencil&) = delete;
        FixedBlockstencil(FixedBlockstencil&& o) : Hypercube5d(std::move(o)) {}
        FixedBlockstencil& operator=(FixedBlockstencil&& o)
        {
            Hypercube5d::operator=(std::move(o));
            return *this;
        }
        inline int getWidth() const { return dim3; }
        inline int getBlocksize() const { return dim1; }
        // FixedBlockstencil multiply(FixedBlockstencil& b, int ghc,
        //                            MPILevelData* mpiData, bool periodic, bool forceLocal) const;
        // FixedBlockstencil& operator*(double factor);
        std::unique_ptr<FixedBlockstencil> copyShallow();

        friend std::ostream& operator<<(std::ostream& os, const FixedBlockstencil& lv);
    };

}

#endif // MGCL_FIXED_BLOCKSTENCIL_HPP
