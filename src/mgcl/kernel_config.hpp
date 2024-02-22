#ifndef __KERNEL_CONFIG_H__
#define __KERNEL_CONFIG_H__

#include <array>
#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace mgcl
{
    // Alias for the size of the workgroup for a kernel, dependend on the number of work-items launched.
    // The work-item count is a lower bound, i.e. all work-item counts between that bound and the next bigger
    // one will be launched with the given workgroup size.
    using KernelWorkgroupSizes = std::vector<std::pair<size_t, std::array<size_t, 3>>>;

    // Alias for the kernel configuration, which can be supplied to
    // a problem before calling solve.
    using KernelConfig = std::map<std::string, KernelWorkgroupSizes>;

    std::array<size_t, 3>& getWorkGroupSizeForKernelAndWiCount(KernelConfig& conf, std::string kernelName, size_t wiCount);
}
#endif // __KERNEL_CONFIG_H__