#include "device_type_generator.hpp"

namespace mgcl_test
{
    // Avoids -Wweak-vtables
    cl_device_type const& CLDeviceTypeGenerator::get() const
    {
        return device_types[current_index];
    }

    // This helper function provides a nicer UX when instantiating the generator
    // Notice that it returns an instance of GeneratorWrapper<int>, which
    // is a value-wrapper around std::unique_ptr<IGenerator<int>>.
    Catch::Generators::GeneratorWrapper<cl_device_type> deviceTypes(std::vector<cl_device_type>& device_types)
    {
        return Catch::Generators::GeneratorWrapper<cl_device_type>(
            new CLDeviceTypeGenerator(device_types)
            // Another possibility:
            // Catch::Detail::make_unique<RandomIntGenerator>(low, high)
        );
    }

} // end anonymous namespaces
