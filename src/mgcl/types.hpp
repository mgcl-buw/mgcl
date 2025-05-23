#ifndef MGCL_TYPES_HPP
#define MGCL_TYPES_HPP

#include "blockstencil.hpp"
#include "blockstencil_gpu.hpp"
#include <memory>
#include <variant>

namespace mgcl
{
    using TBlockstencilInv = std::variant<
        std::shared_ptr<Blockstencil>,
        std::shared_ptr<CuboidBS>,
        std::shared_ptr<BlockstencilGpu>,
        std::shared_ptr<CuboidBSGpu>>;
}

#endif // MGCL_TYPES_HPP
