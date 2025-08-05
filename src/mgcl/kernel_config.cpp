#include "kernel_config.hpp"
#include "mgcl.hpp"
#include <algorithm>
#include <stdexcept>

namespace mgcl::conf
{
    /**
     * @brief Create a KernelConfig object with default values. Note that this function does not set the
     * attribute kernelConfig. If a customized KernelConfig shall be used, the current kernelConfig can be
     * queried using Problem::getKernelConfig, which returns an editable reference.
     *
     * @return std::unique_ptr<KernelConfig>
     */
    KernelConfig createDefaultKernelConfig()
    {
        KernelConfig ret;

        // Jacobi kernels
        ret["jacobi_iter_27point_varying_stencil_1d"] = KernelWorkgroupSizes{{1, {128, 1, 1}}};
        ret["jacobi_iter_27point_fixed_stencil_1d"] = KernelWorkgroupSizes{{1, {128, 1, 1}}};
        ret["jacobi_iter_7point"] = KernelWorkgroupSizes{{1, {1, 64, 1}}};
        ret["jacobi_iter_19point"] = KernelWorkgroupSizes{{1, {1, 64, 1}}};
        ret["jacobi_iter_27point"] = KernelWorkgroupSizes{{1, {1, 64, 1}}};

        // Residual kernels
        ret["residual_27point_varying_stencil"] = KernelWorkgroupSizes{{1, {128, 1, 1}}};
        ret["residual_27point_fixed_stencil"] = KernelWorkgroupSizes{{1, {128, 1, 1}}};
        ret["residual_7point"] = KernelWorkgroupSizes{{1, {128, 1, 1}}};
        ret["residual_19point"] = KernelWorkgroupSizes{{1, {128, 1, 1}}};
        ret["residual_27point"] = KernelWorkgroupSizes{{1, {128, 1, 1}}};
        ret["residual_squared"] = KernelWorkgroupSizes{{1, {128, 1, 1}}};

        // Ghost update kernels
        ret["update_ghosts_periodic"] = KernelWorkgroupSizes{{1, {4, 4, 4}}};
        ret["extract_border_planes"] = KernelWorkgroupSizes{{1, {32, 1, 1}}};
        ret["paste_ghosts_from_border_planes"] = KernelWorkgroupSizes{{1, {32, 1, 1}}};
        ret["extract_border_planes_varying_stencil"] = KernelWorkgroupSizes{{1, {128, 1, 1}}};
        ret["paste_ghosts_from_border_planes_varying_stencil"] = KernelWorkgroupSizes{{1, {128, 1, 1}}};

        // Copy buffer kernels
        ret["copy_input_data"] = KernelWorkgroupSizes{{1, {4, 4, 4}}};
        ret["copy_output_data"] = KernelWorkgroupSizes{{1, {4, 4, 4}}};

        // V-cycle kernels
        ret["correct_error"] = KernelWorkgroupSizes{{1, {4, 4, 4}}};
        ret["restrict_to_coarse"] = KernelWorkgroupSizes{{1, {4, 4, 4}}};
        ret["prolongate_to_fine"] = KernelWorkgroupSizes{{1, {4, 4, 4}}};

        // Stencil kernels
        ret["update_ghosts_varying_stencil"] = KernelWorkgroupSizes{{1, {4, 4, 4}}};
        ret["mult_stencils_var_var"] = KernelWorkgroupSizes{{1, {4, 4, 4}}};
        ret["mult_stencils_var_fix"] = KernelWorkgroupSizes{{1, {4, 4, 4}}};
        ret["mult_stencils_fix_var"] = KernelWorkgroupSizes{{1, {4, 4, 4}}};
        ret["cut_stencils_w7_to_w3"] = KernelWorkgroupSizes{{1, {4, 4, 4}}};

        // Galerkin kernels
        ret["galerkin"] = KernelWorkgroupSizes{{1, {128, 1, 1}}};
        ret["galerkin_handcrafted"] = KernelWorkgroupSizes{{1, {128, 1, 1}}};
        ret["galerkin_fixed_stencil"] = KernelWorkgroupSizes{{1, {32, 1, 1}}};

        // Utility kernels
        ret["sum_partial_global_eq_x_num_elements"] = KernelWorkgroupSizes{{1, {128, 1, 1}}};
        // c["sum_finish"] = KernelWorkgroupSizes{{1, {4, 4, 4}}}; // Launches only 1 wi
        ret["max_partial_global_eq_x_num_elements"] = KernelWorkgroupSizes{{1, {128, 1, 1}}};
        // c["max_finish"] = KernelWorkgroupSizes{{1, {4, 4, 4}}}; // Launches only 1 wi
        ret["max_abs_partial_global_eq_x_num_elements"] = KernelWorkgroupSizes{{1, {128, 1, 1}}};
        ret["fill_buffer"] = KernelWorkgroupSizes{{1, {64, 1, 1}}};
        ret["fill_1d_index"] = KernelWorkgroupSizes{{1, {64, 1, 1}}};

        // blockstencil kernels
        ret["extract_border_planes_cuboidbs"] = KernelWorkgroupSizes{{1, {32, 1, 1}}};
        ret["paste_ghosts_from_border_planes_cuboidbs"] = KernelWorkgroupSizes{{1, {32, 1, 1}}};
        ret["update_ghosts_cuboidbs_periodic_blockstencil"] = KernelWorkgroupSizes{{1, {4, 4, 4}}};
        ret["update_ghosts_blockstencil"] = KernelWorkgroupSizes{{1, {4, 4, 4}}};
        ret["extract_border_planes_blockstencil"] = KernelWorkgroupSizes{{1, {32, 1, 1}}};
        ret["paste_ghosts_from_border_planes_blockstencil"] = KernelWorkgroupSizes{{1, {32, 1, 1}}};
        ret["residual_27point_blockstencil_block_first_v_gp_first"] = KernelWorkgroupSizes{{1, {128, 1, 1}}};
        ret["residual_squared_blockstencil"] = KernelWorkgroupSizes{{1, {128, 1, 1}}};
        ret["jacobi_iter_27point_blockstencil_block_first_v_gp_first_blockjacobi"] = KernelWorkgroupSizes{{1, {128, 1, 1}}};
        ret["jacobi_iter_27point_blockstencil_block_first_v_gp_first_scalarjacobi"] = KernelWorkgroupSizes{{1, {128, 1, 1}}};
        ret["restrict_to_coarse_blockstencil"] = KernelWorkgroupSizes{{1, {4, 4, 4}}};
        ret["prolongate_to_fine_blockstencil"] = KernelWorkgroupSizes{{1, {4, 4, 4}}};
        ret["galerkin_blockstencil"] = KernelWorkgroupSizes{{1, {128, 1, 1}}};
        ret["correct_error_blockstencil"] = KernelWorkgroupSizes{{1, {4, 4, 4}}};

        return ret;
    }

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
        try
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
        catch (std::out_of_range ex)
        {
            error("No launch configuration found for kernel with name: " + kernelName);
        }
    }
}