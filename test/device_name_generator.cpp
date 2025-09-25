#include "device_name_generator.hpp"

namespace mgcl_test
{
    // Avoids -Wweak-vtables
    std::string const& CLDeviceNameGenerator::get() const
    {
        return device_names[current_index];
    }

    // This helper function provides a nicer UX when instantiating the generator
    // Notice that it returns an instance of GeneratorWrapper<int>, which
    // is a value-wrapper around std::unique_ptr<IGenerator<int>>.
    Catch::Generators::GeneratorWrapper<std::string> deviceNames(std::vector<std::string>& device_names)
    {
        return Catch::Generators::GeneratorWrapper<std::string>(
            new CLDeviceNameGenerator(device_names)
            // Another possibility:
            // Catch::Detail::make_unique<RandomIntGenerator>(low, high)
        );
    }

} // end anonymous namespaces
