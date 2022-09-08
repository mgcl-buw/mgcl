#include "stencil.hpp"

namespace mgcl
{
    double StencilLaplace7p::apply(int i, int j, int k)
    {
        // clang-format off
        return (6.0 * (*v)[i][j][k]
            - (*v)[i][j][k - 1] - (*v)[i][j][k + 1]
            - (*v)[i][j - 1][k] - (*v)[i][j + 1][k]
            - (*v)[i - 1][j][k] - (*v)[i + 1][j][k]
            ) * stencilFactor;
        // clang-format on
    }

    double StencilLaplace19p::apply(int i, int j, int k)
    {
        // clang-format off
        return (24.0 * (*v)[i][j][k]
                - 2.0 * (*v)[i][j][k - 1] - 2.0 * (*v)[i][j][k + 1]
                - 2.0 * (*v)[i][j - 1][k] - 2.0 * (*v)[i][j + 1][k]
                - 2.0 * (*v)[i - 1][j][k] - 2.0 * (*v)[i + 1][j][k]
                
                - (*v)[i][j - 1][k - 1] - (*v)[i][j - 1][k + 1]
                - (*v)[i][j + 1][k - 1] - (*v)[i][j + 1][k + 1]
                - (*v)[i - 1][j][k - 1] - (*v)[i - 1][j][k + 1]
                - (*v)[i + 1][j][k - 1] - (*v)[i + 1][j][k + 1]
                - (*v)[i - 1][j - 1][k] - (*v)[i - 1][j + 1][k]
                - (*v)[i + 1][j - 1][k] - (*v)[i + 1][j + 1][k]
                ) * stencilFactor;
        // clang-format on
    }

    double StencilLaplace27p::apply(int i, int j, int k)
    {
        // clang-format off
        return (128.0 * (*v)[i][j][k]
                - 14.0 * (*v)[i][j][k - 1] - 14.0 * (*v)[i][j][k + 1]
                - 14.0 * (*v)[i][j - 1][k] - 14.0 * (*v)[i][j + 1][k]
                - 14.0 * (*v)[i - 1][j][k] - 14.0 * (*v)[i + 1][j][k]

                - 3.0 * (*v)[i][j - 1][k - 1] - 3.0 * (*v)[i][j - 1][k + 1]
                - 3.0 * (*v)[i][j + 1][k - 1] - 3.0 * (*v)[i][j + 1][k + 1]
                - 3.0 * (*v)[i - 1][j][k - 1] - 3.0 * (*v)[i - 1][j][k + 1]
                - 3.0 * (*v)[i + 1][j][k - 1] - 3.0 * (*v)[i + 1][j][k + 1]
                - 3.0 * (*v)[i - 1][j - 1][k] - 3.0 * (*v)[i - 1][j + 1][k]
                - 3.0 * (*v)[i + 1][j - 1][k] - 3.0 * (*v)[i + 1][j + 1][k]

                - (*v)[i - 1][j - 1][k - 1] - (*v)[i - 1][j - 1][k + 1]
                - (*v)[i - 1][j + 1][k - 1] - (*v)[i - 1][j + 1][k + 1]
                - (*v)[i + 1][j - 1][k - 1] - (*v)[i + 1][j - 1][k + 1]
                - (*v)[i + 1][j + 1][k - 1] - (*v)[i + 1][j + 1][k + 1]
                ) * stencilFactor;
        // clang-format on
    }

    /**
     * @brief Checks if size of stencilValues fits to the Cuboid given.
     *
     * @throws invalid_argument If sizes do not fit
     */
    void VaryingStencil::checkSizes()
    {
        if (stencilValues->field1d().size() != v->field1d().size() * stencilSizePerGridPoint)
            throw std::invalid_argument(
                std::string("stencilValues.size() != v.size() * stencilSizePerGridPoint (")
                    .append(std::to_string(stencilValues->field1d().size()))
                    .append(" != ")
                    .append(std::to_string(v->field1d().size()))
                    .append(" * ")
                    .append(std::to_string(stencilSizePerGridPoint)));
    }

    double FixedStencil::getStencilFactor() const
    {
        return stencilFactor;
    }
}