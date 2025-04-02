#ifndef MGCL_BLOCKSTENCIL_HPP
#define MGCL_BLOCKSTENCIL_HPP

#include "hypercube.hpp"
#include "mpi_level_data.hpp"

namespace mgcl
{
    /**
     * @brief Class for NxNxN varying stencils, i.e. stencil can differ for each grid point.
     * Choose N = 3 to include only direct neighbors, N = 5 to include 2 nearest neighbors, etc.
     * If two stencils of size NxNxN and NBxNBxNB get multiplied with each other, the resulting stencil has size
     * (N + NB - 1)^3.
     *
     */
    class Blockstencil : public Hypercube8d
    {
    public:
        Blockstencil(int m, int n, int o, int _width, int blocksize, int ghosts_m, int ghosts_n, int ghosts_o);
        Blockstencil(Blockstencil&) = delete;
        Blockstencil& operator=(const Blockstencil&) = delete;
        Blockstencil(Blockstencil&& o) : Hypercube8d(std::move(o)) {}
        Blockstencil& operator=(Blockstencil&& o)
        {
            Hypercube8d::operator=(std::move(o));
            return *this;
        }
        inline int getM() const { return dim6; }
        inline int getN() const { return dim7; }
        inline int getO() const { return dim8; }
        inline int getGhostsM() const { return ghostsDim6; }
        inline int getGhostsN() const { return ghostsDim7; }
        inline int getGhostsO() const { return ghostsDim8; }
        inline int getMgh() const { return dim6 + 2 * ghostsDim6; }
        inline int getNgh() const { return dim7 + 2 * ghostsDim7; }
        inline int getOgh() const { return dim8 + 2 * ghostsDim8; }
        inline int getWidth() const { return dim3; }
        inline int getBlocksize() const { return dim1; }
        void updateGhosts();
        Blockstencil multiply(Blockstencil& b, int ghc,
                              MPILevelData* mpiData, bool periodic, bool forceLocal) const;
        Blockstencil& operator*(double factor);
        std::unique_ptr<Blockstencil> slice(int m_start, int m_end, int n_start, int n_end,
                                            int o_start, int o_end,
                                            int ghm = -1, int ghn = -1, int gho = -1);
        std::unique_ptr<Blockstencil> sliceIncGhosts(int m_start, int m_end, int n_start, int n_end,
                                                     int o_start, int o_end);
        std::unique_ptr<Blockstencil> copyShallow();

        friend std::ostream& operator<<(std::ostream& os, const Blockstencil& lv);
    };

}

#endif // MGCL_BLOCKSTENCIL_HPP
