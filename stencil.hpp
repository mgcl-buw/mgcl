#pragma once

#include <memory>
#include <stdexcept>
#include <vector>

#include "cuboid.hpp"
#include "hypercube.hpp"

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
        std::shared_ptr<Hypercube4d> stencilValues = nullptr;
        int stencilSizePerGridPoint = 0;

        enum Pos
        {
            // clang-format off
            // 7p
            SELF,   // [i][j][k]
            FRONT,  // [i][j][k - 1]
            BACK,   // [i][j][k + 1]
            TOP,    // [i][j - 1][k]
            BOTTOM, // [i][j + 1][k]
            LEFT,   // [i - 1][j][k]
            RIGHT,  // [i + 1][j][k]

            // 19p
            FRONT_TOP,    // [i][j - 1][k - 1]
            BACK_TOP,     // [i][j - 1][k + 1]
            FRONT_BOTTOM, // [i][j + 1][k - 1]
            BACK_BOTTOM,  // [i][j + 1][k + 1]
            FRONT_LEFT,   // [i - 1][j][k - 1]
            BACK_LEFT,    // [i - 1][j][k + 1]
            FRONT_RIGHT,  // [i + 1][j][k - 1]
            BACK_RIGHT,   // [i + 1][j][k + 1]
            LEFT_TOP,     // [i - 1][j - 1][k]
            LEFT_BOTTOM,  // [i - 1][j + 1][k]
            RIGHT_TOP,    // [i + 1][j - 1][k]
            RIGHT_BOTTOM, // [i + 1][j + 1][k]

            // 27p
            FRONT_TOP_LEFT,     // [i - 1][j - 1][k - 1]
            BACK_TOP_LEFT,      // [i - 1][j - 1][k + 1]
            FRONT_BOTTOM_LEFT,  // [i - 1][j + 1][k - 1]
            BACK_BOTTOM_LEFT,   // [i - 1][j + 1][k + 1]
            FRONT_TOP_RIGHT,    // [i + 1][j - 1][k - 1]
            BACK_TOP_RIGHT,     // [i + 1][j - 1][k + 1]
            FRONT_BOTTOM_RIGHT, // [i + 1][j + 1][k - 1]
            BACK_BOTTOM_RIGHT   // [i + 1][j + 1][k + 1]

            // clang-format on
        };

    public:
        /**
         * @brief Construct a new VaryingStencil object for Cuboid v_. stencilValues will be created as a 4d Hypercube
         * in concrete subclasses having values of the stencil per grid point with dimensions
         * (v.m, v.n, v.o, stencilSizePerGridPoint).
         *
         * @param v_ Cuboid that this stencil shall be applied on.
         * @param stencilValues_
         */
        VaryingStencil(std::shared_ptr<Cuboid> v_)
            : Stencil(v_) {}

        virtual double apply(int i, int j, int k) = 0;
        std::shared_ptr<Hypercube4d> getStencilValues() const;
    };

    /**
     * @brief Applies a varying 7-point stencil, i.e. the stencil can differ for each grid point.
     * stencilValues is created with dimensions (v.m, v.n, v.o, 7) and can be filled using the getter.
     *
     */
    class StencilVarying7p : public VaryingStencil
    {
    public:
        StencilVarying7p(std::shared_ptr<Cuboid> v_)
            : VaryingStencil(v_)
        {
            stencilSizePerGridPoint = 7;
            stencilValues = std::make_shared<Hypercube4d>(v_->getM(), v_->getN(), v_->getO(), stencilSizePerGridPoint);
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
        StencilVarying19p(std::shared_ptr<Cuboid> v_)
            : VaryingStencil(v_)
        {
            stencilSizePerGridPoint = 19;
            stencilValues = std::make_shared<Hypercube4d>(v_->getM(), v_->getN(), v_->getO(), stencilSizePerGridPoint);
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
        StencilVarying27p(std::shared_ptr<Cuboid> v_)
            : VaryingStencil(v_)
        {
            stencilSizePerGridPoint = 27;
            stencilValues = std::make_shared<Hypercube4d>(v_->getM(), v_->getN(), v_->getO(), stencilSizePerGridPoint);
        }

        double apply(int i, int j, int k);
    };

}