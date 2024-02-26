#include "profiling_data.hpp"

#include <iostream>

namespace mgcl
{

    /**
     * @brief Prints measurements to ostream.
     *
     */
    std::ostream& operator<<(std::ostream& os, const ProfilingData& pd)
    {
        for (const auto& entry : pd.measurements)
        {
            auto kernelName = entry.first;
            os << kernelName << std::endl;
            for (const auto& m : entry.second)
            {
                os << "* time in ns: " << m.elapsed << std::endl;
                os << "  work-items: " << m.work_items[0] << "," << m.work_items[1] << "," << m.work_items[2] << std::endl;
                os << "  work-group: " << m.work_group[0] << "," << m.work_group[1] << "," << m.work_group[2] << std::endl;
            }
        }
        return os;
    }
} // namespace mgcl
