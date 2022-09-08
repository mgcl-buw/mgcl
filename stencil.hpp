#pragma once

#include <memory>
#include <stdexcept>
#include <vector>

#include "cuboid.hpp"

namespace mgcl
{
    /**
     * @brief Abstract base class for stencils. When a stencil calls apply, it is applied to one cell of the Cuboid v.
     * The stencil result is multiplied with stencilFactor which defaults to 1.
     *
     */
    class Stencil
    {
    protected:
        std::shared_ptr<Cuboid> v = nullptr;

    public:
        Stencil(std::shared_ptr<Cuboid> v_) : v(v_) {}

        virtual double apply(int i, int j, int k) = 0;
    };

    /**
     * @brief Class for fixed stencils, i.e. the same stencil for each grid point, e.g. Laplace stencil.
     *
     */
    class FixedStencil : public Stencil
    {
    protected:
        double stencilFactor = 1;

    public:
        FixedStencil(std::shared_ptr<Cuboid> v_) : Stencil(v_) {}

        double getStencilFactor() const;
    };

    /**
     * @brief Applies 7-point Laplace stencil.
     *
     */
    class StencilLaplace7p : public FixedStencil
    {
    public:
        StencilLaplace7p(std::shared_ptr<Cuboid> v_) : FixedStencil(v_)
        {
            stencilFactor = (double)(v->getM() * v->getM()); // h2inv
        }
        double apply(int i, int j, int k);
    };

    /**
     * @brief Applies 19-point Laplace stencil.
     *
     */
    class StencilLaplace19p : public FixedStencil
    {
    public:
        StencilLaplace19p(std::shared_ptr<Cuboid> v_) : FixedStencil(v_)
        {
            stencilFactor = ((double)(v->getM() * v->getM())) / 6.0;
        }
        double apply(int i, int j, int k);
    };

    /**
     * @brief Applies 27-point Laplace stencil.
     *
     */
    class StencilLaplace27p : public FixedStencil
    {
    public:
        StencilLaplace27p(std::shared_ptr<Cuboid> v_) : FixedStencil(v_)
        {
            stencilFactor = ((double)(v->getM() * v->getM())) / 30.0;
        }
        double apply(int i, int j, int k);
    };

    /**
     * @brief Class for varying stencils, i.e. stencil can differ for each grid point.
     *
     */
    class VaryingStencil : public Stencil
    {
    protected:
        std::shared_ptr<Cuboid> stencilValues;
        int stencilSizePerGridPoint = 0;

    public:
        VaryingStencil(std::shared_ptr<Cuboid> v_, std::shared_ptr<Cuboid> stencilValues_)
            : Stencil(v_), stencilValues(stencilValues_) {}

        virtual double apply(int i, int j, int k) = 0;
        void checkSizes();
    };

    /**
     * @brief Applies a varying 7-point stencil, i.e. the stencil can differ for each grid point.
     *
     */
    class StencilVarying7p : public VaryingStencil
    {
    public:
        StencilVarying7p(std::shared_ptr<Cuboid> v_, std::shared_ptr<Cuboid> stencilValues_)
            : VaryingStencil(v_, stencilValues_)
        {
            stencilSizePerGridPoint = 7;
            checkSizes();
        }

        double apply(int i, int j, int k);
    };

    /**
     * @brief Applies a varying 19-point stencil, i.e. the stencil can differ for each grid point.
     *
     */
    class StencilVarying19p : public VaryingStencil
    {
    public:
        StencilVarying19p(std::shared_ptr<Cuboid> v_, std::shared_ptr<Cuboid> stencilValues_)
            : VaryingStencil(v_, stencilValues_)
        {
            stencilSizePerGridPoint = 19;
            checkSizes();
        }

        double apply(int i, int j, int k);
    };

    /**
     * @brief Applies a varying 27-point stencil, i.e. the stencil can differ for each grid point.
     *
     */
    class StencilVarying27p : public VaryingStencil
    {
    public:
        StencilVarying27p(std::shared_ptr<Cuboid> v_, std::shared_ptr<Cuboid> stencilValues_)
            : VaryingStencil(v_, stencilValues_)
        {
            stencilSizePerGridPoint = 27;
            checkSizes();
        }

        double apply(int i, int j, int k);
    };

}