#include "kernel_config.hpp"
#include <algorithm>

namespace mgcl
{
    /**
     * @brief Get the Work Group Size For Kernel And Wi Count object
     *
     * Throws an exception, if no configuration for the kernelName is found.
     *
     * @param kernelName
     * @param wiCount
     * @return std::array<size_t, 3>&
     */
    std::array<size_t, 3>& getWorkGroupSizeForKernelAndWiCount(KernelConfig& conf, std::string kernelName, size_t wiCount)
    {
        auto& confForKernel = conf.at(kernelName); // Will throw exception, if key is not found.

        // Sort the workgroup sizes wrt wiCount in descending order
        std::sort(confForKernel.begin(), confForKernel.end(), [](const auto& a, const auto& b)
                  { return a.first > b.first; });

        // Loop through the workgroup sizes from high wiCount to low and return the first one that is bigger
        // or equal to wiCount.
        for (auto& workgroupSize : confForKernel)
            if (wiCount >= workgroupSize.first)
                return workgroupSize.second;

        return confForKernel.back().second;
    }
}