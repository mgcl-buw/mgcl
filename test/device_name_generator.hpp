#ifndef MGCL_DEVICE_NAME_GENERATOR_HPP
#define MGCL_DEVICE_NAME_GENERATOR_HPP

#include <iostream>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>

#include <vector>

namespace mgcl_test
{
    // This class implements a Catch2 Generator for using a std::vector<cl_device_type> as argument for
    // the GENERATE() macro. It can be used like this:
    // auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));
    // Based on this example: https://github.com/catchorg/Catch2/blob/devel/examples/300-Gen-OwnGenerator.cpp
    class CLDeviceNameGenerator final : public Catch::Generators::IGenerator<std::string>
    {
        std::vector<std::string> device_names;
        int current_index = 0;

    public:
        CLDeviceNameGenerator(std::vector<std::string>& device_names)
            : device_names(device_names) {}

        std::string const& get() const override;
        // Returns true, if there is a next element in the vector
        bool next() override
        {
            return (++current_index < device_names.size());
        }
    };

    // This helper function provides a nicer UX when instantiating the generator
    // Notice that it returns an instance of GeneratorWrapper<int>, which
    // is a value-wrapper around std::unique_ptr<IGenerator<int>>.
    Catch::Generators::GeneratorWrapper<std::string> deviceNames(std::vector<std::string>& device_names);
}

#endif // MGCL_DEVICE_NAME_GENERATOR_HPP
