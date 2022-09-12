#include "stencil.hpp"
#include "mgcl.hpp"

namespace mgcl
{
    double FixedStencil::getStencilFactor() const
    {
        return stencilFactor;
    }

    /**
     * @brief Applies 7p Laplace stencil to v at grid point i,j,k and returns stencil sum.
     *
     * @param i grid point location
     * @param j grid point location
     * @param k grid point location
     * @return double stencil sum
     */
    double StencilLaplace7p::apply(Cuboid &v, int i, int j, int k)
    {
        // clang-format off
        return (6.0 * v[i][j][k]
            - v[i][j][k - 1] - v[i][j][k + 1]
            - v[i][j - 1][k] - v[i][j + 1][k]
            - v[i - 1][j][k] - v[i + 1][j][k]
            ) * stencilFactor;
        // clang-format on
    }

    /**
     * @brief Creates a clone of this stencil with maybe another Cuboid and without the need of knowing
     * the concrete subtype.
     *
     * @param v Cuboid the new stencil shall be applied on.
     * @return std::shared_ptr<Stencil> to the clone.
     */
    std::shared_ptr<Stencil> StencilLaplace7p::clone(int m, int n, int o, double h)
    {
        return std::make_shared<StencilLaplace7p>(h);
    }

    /**
     * @brief Applies 19p Laplace stencil to v at grid point i,j,k and returns stencil sum.
     *
     * @param i grid point location
     * @param j grid point location
     * @param k grid point location
     * @return double stencil sum
     */
    double StencilLaplace19p::apply(Cuboid &v, int i, int j, int k)
    {
        // clang-format off
        return (24.0 * v[i][j][k]
                - 2.0 * v[i][j][k - 1] - 2.0 * v[i][j][k + 1]
                - 2.0 * v[i][j - 1][k] - 2.0 * v[i][j + 1][k]
                - 2.0 * v[i - 1][j][k] - 2.0 * v[i + 1][j][k]
                
                - v[i][j - 1][k - 1] - v[i][j - 1][k + 1]
                - v[i][j + 1][k - 1] - v[i][j + 1][k + 1]
                - v[i - 1][j][k - 1] - v[i - 1][j][k + 1]
                - v[i + 1][j][k - 1] - v[i + 1][j][k + 1]
                - v[i - 1][j - 1][k] - v[i - 1][j + 1][k]
                - v[i + 1][j - 1][k] - v[i + 1][j + 1][k]
                ) * stencilFactor;
        // clang-format on
    }

    /**
     * @brief Creates a clone of this stencil with maybe another Cuboid and without the need of knowing
     * the concrete subtype.
     *
     * @param v Cuboid the new stencil shall be applied on.
     * @return std::shared_ptr<Stencil> to the clone.
     */
    std::shared_ptr<Stencil> StencilLaplace19p::clone(int m, int n, int o, double h)
    {
        return std::make_shared<StencilLaplace19p>(h);
    }

    /**
     * @brief Applies 27p Laplace stencil to v at grid point i,j,k and returns stencil sum.
     *
     * @param i grid point location
     * @param j grid point location
     * @param k grid point location
     * @return double stencil sum
     */
    double StencilLaplace27p::apply(Cuboid &v, int i, int j, int k)
    {
        // clang-format off
        return (128.0 * v[i][j][k]
                - 14.0 * v[i][j][k - 1] - 14.0 * v[i][j][k + 1]
                - 14.0 * v[i][j - 1][k] - 14.0 * v[i][j + 1][k]
                - 14.0 * v[i - 1][j][k] - 14.0 * v[i + 1][j][k]

                - 3.0 * v[i][j - 1][k - 1] - 3.0 * v[i][j - 1][k + 1]
                - 3.0 * v[i][j + 1][k - 1] - 3.0 * v[i][j + 1][k + 1]
                - 3.0 * v[i - 1][j][k - 1] - 3.0 * v[i - 1][j][k + 1]
                - 3.0 * v[i + 1][j][k - 1] - 3.0 * v[i + 1][j][k + 1]
                - 3.0 * v[i - 1][j - 1][k] - 3.0 * v[i - 1][j + 1][k]
                - 3.0 * v[i + 1][j - 1][k] - 3.0 * v[i + 1][j + 1][k]

                - v[i - 1][j - 1][k - 1] - v[i - 1][j - 1][k + 1]
                - v[i - 1][j + 1][k - 1] - v[i - 1][j + 1][k + 1]
                - v[i + 1][j - 1][k - 1] - v[i + 1][j - 1][k + 1]
                - v[i + 1][j + 1][k - 1] - v[i + 1][j + 1][k + 1]
                ) * stencilFactor;
        // clang-format on
    }

    /**
     * @brief Creates a clone of this stencil with maybe another Cuboid and without the need of knowing
     * the concrete subtype.
     *
     * @param v Cuboid the new stencil shall be applied on.
     * @return std::shared_ptr<Stencil> to the clone.
     */
    std::shared_ptr<Stencil> StencilLaplace27p::clone(int m, int n, int o, double h)
    {
        return std::make_shared<StencilLaplace27p>(h);
    }

    /**
     * @brief Applies varying 7p stencil to v at grid point i,j,k and returns stencil sum.
     *
     * @param i grid point location
     * @param j grid point location
     * @param k grid point location
     * @return double stencil sum
     */
    double StencilVarying7p::apply(Cuboid &v, int i, int j, int k)
    {
        // clang-format off
        return (*stencilValues)[i][j][k][SELF]  * v[i][j][k]
            + (*stencilValues)[i][j][k][FRONT]  * v[i][j][k - 1]
            + (*stencilValues)[i][j][k][BACK]   * v[i][j][k + 1]
            + (*stencilValues)[i][j][k][TOP]    * v[i][j - 1][k]
            + (*stencilValues)[i][j][k][BOTTOM] * v[i][j + 1][k]
            + (*stencilValues)[i][j][k][LEFT]   * v[i - 1][j][k]
            + (*stencilValues)[i][j][k][RIGHT]  * v[i + 1][j][k];
        // clang-format on
    }

    /**
     * @brief Creates a clone of this stencil with maybe another Cuboid and without the need of knowing
     * the concrete subtype. stencilValues won't be set in the clone.
     *
     * @param v Cuboid the new stencil shall be applied on.
     * @return std::shared_ptr<Stencil> to the clone.
     */
    std::shared_ptr<Stencil> StencilVarying7p::clone(int m, int n, int o, double h)
    {
        return std::make_shared<StencilVarying7p>(m, n, o);
    }

    /**
     * @brief Applies varying 19p stencil to v at grid point i,j,k and returns stencil sum.
     *
     * @param i grid point location
     * @param j grid point location
     * @param k grid point location
     * @return double stencil sum
     */
    double StencilVarying19p::apply(Cuboid &v, int i, int j, int k)
    {
        // clang-format off
        return (*stencilValues)[i][j][k][SELF]  * v[i][j][k]
            + (*stencilValues)[i][j][k][FRONT]  * v[i][j][k - 1]
            + (*stencilValues)[i][j][k][BACK]   * v[i][j][k + 1]
            + (*stencilValues)[i][j][k][TOP]    * v[i][j - 1][k]
            + (*stencilValues)[i][j][k][BOTTOM] * v[i][j + 1][k]
            + (*stencilValues)[i][j][k][LEFT]   * v[i - 1][j][k]
            + (*stencilValues)[i][j][k][RIGHT]  * v[i + 1][j][k]
            
            + (*stencilValues)[i][j][k][FRONT_TOP]    * v[i][j - 1][k - 1]
            + (*stencilValues)[i][j][k][BACK_TOP]     * v[i][j - 1][k + 1]
            + (*stencilValues)[i][j][k][FRONT_BOTTOM] * v[i][j + 1][k - 1]
            + (*stencilValues)[i][j][k][BACK_BOTTOM]  * v[i][j + 1][k + 1]
            + (*stencilValues)[i][j][k][FRONT_LEFT]   * v[i - 1][j][k - 1]
            + (*stencilValues)[i][j][k][BACK_LEFT]    * v[i - 1][j][k + 1]
            + (*stencilValues)[i][j][k][FRONT_RIGHT]  * v[i + 1][j][k - 1]
            + (*stencilValues)[i][j][k][BACK_RIGHT]   * v[i + 1][j][k + 1]
            + (*stencilValues)[i][j][k][LEFT_TOP]     * v[i - 1][j - 1][k]
            + (*stencilValues)[i][j][k][LEFT_BOTTOM]  * v[i - 1][j + 1][k]
            + (*stencilValues)[i][j][k][RIGHT_TOP]    * v[i + 1][j - 1][k]
            + (*stencilValues)[i][j][k][RIGHT_BOTTOM] * v[i + 1][j + 1][k];
        // clang-format on
    }

    /**
     * @brief Creates a clone of this stencil with maybe another Cuboid and without the need of knowing
     * the concrete subtype. stencilValues won't be set in the clone.
     *
     * @param v Cuboid the new stencil shall be applied on.
     * @return std::shared_ptr<Stencil> to the clone.
     */
    std::shared_ptr<Stencil> StencilVarying19p::clone(int m, int n, int o, double h)
    {
        return std::make_shared<StencilVarying19p>(m, n, o);
    }

    /**
     * @brief Applies varying 27p stencil to v at grid point i,j,k and returns stencil sum.
     *
     * @param i grid point location
     * @param j grid point location
     * @param k grid point location
     * @return double stencil sum
     */
    double StencilVarying27p::apply(Cuboid &v, int i, int j, int k)
    {
        // clang-format off
        return (*stencilValues)[i][j][k][SELF]  * v[i][j][k]
            + (*stencilValues)[i][j][k][FRONT]  * v[i][j][k - 1]
            + (*stencilValues)[i][j][k][BACK]   * v[i][j][k + 1]
            + (*stencilValues)[i][j][k][TOP]    * v[i][j - 1][k]
            + (*stencilValues)[i][j][k][BOTTOM] * v[i][j + 1][k]
            + (*stencilValues)[i][j][k][LEFT]   * v[i - 1][j][k]
            + (*stencilValues)[i][j][k][RIGHT]  * v[i + 1][j][k]
            
            + (*stencilValues)[i][j][k][FRONT_TOP]    * v[i][j - 1][k - 1]
            + (*stencilValues)[i][j][k][BACK_TOP]     * v[i][j - 1][k + 1]
            + (*stencilValues)[i][j][k][FRONT_BOTTOM] * v[i][j + 1][k - 1]
            + (*stencilValues)[i][j][k][BACK_BOTTOM]  * v[i][j + 1][k + 1]
            + (*stencilValues)[i][j][k][FRONT_LEFT]   * v[i - 1][j][k - 1]
            + (*stencilValues)[i][j][k][BACK_LEFT]    * v[i - 1][j][k + 1]
            + (*stencilValues)[i][j][k][FRONT_RIGHT]  * v[i + 1][j][k - 1]
            + (*stencilValues)[i][j][k][BACK_RIGHT]   * v[i + 1][j][k + 1]
            + (*stencilValues)[i][j][k][LEFT_TOP]     * v[i - 1][j - 1][k]
            + (*stencilValues)[i][j][k][LEFT_BOTTOM]  * v[i - 1][j + 1][k]
            + (*stencilValues)[i][j][k][RIGHT_TOP]    * v[i + 1][j - 1][k]
            + (*stencilValues)[i][j][k][RIGHT_BOTTOM] * v[i + 1][j + 1][k]
            
            + (*stencilValues)[i][j][k][FRONT_TOP_LEFT]     * v[i - 1][j - 1][k - 1]
            + (*stencilValues)[i][j][k][BACK_TOP_LEFT]      * v[i - 1][j - 1][k + 1]
            + (*stencilValues)[i][j][k][FRONT_BOTTOM_LEFT]  * v[i - 1][j + 1][k - 1]
            + (*stencilValues)[i][j][k][BACK_BOTTOM_LEFT]   * v[i - 1][j + 1][k + 1]
            + (*stencilValues)[i][j][k][FRONT_TOP_RIGHT]    * v[i + 1][j - 1][k - 1]
            + (*stencilValues)[i][j][k][BACK_TOP_RIGHT]     * v[i + 1][j - 1][k + 1]
            + (*stencilValues)[i][j][k][FRONT_BOTTOM_RIGHT] * v[i + 1][j + 1][k - 1]
            + (*stencilValues)[i][j][k][BACK_BOTTOM_RIGHT]  * v[i + 1][j + 1][k + 1];
        // clang-format on
    }

    /**
     * @brief Creates a clone of this stencil with maybe another Cuboid and without the need of knowing
     * the concrete subtype. stencilValues won't be set in the clone.
     *
     * @param v Cuboid the new stencil shall be applied on.
     * @return std::shared_ptr<Stencil> to the clone.
     */
    std::shared_ptr<Stencil> StencilVarying27p::clone(int m, int n, int o, double h)
    {
        return std::make_shared<StencilVarying27p>(m, n, o);
    }

    std::shared_ptr<Hypercube4d> VaryingStencil::getStencilValues() const
    {
        return stencilValues;
    }

    MGCL_STENCIL Stencil::getType() const
    {
        return type;
    }
}