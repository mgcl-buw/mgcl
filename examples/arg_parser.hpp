#ifndef MGCL_ARG_PARSER_HPP
#define MGCL_ARG_PARSER_HPP

#include <algorithm>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace mgcl_examples_helper
{
    class ArgParser
    {
    public:
        enum ArgType
        {
            FLAG,
            VALUE,
            INT_LIST,
            ENUM_VALUE
        };

        struct ArgInfo
        {
            std::string long_name;
            std::string description;
            ArgType type;
            std::vector<std::string> aliases;
            std::set<std::string> allowed_values; // For ENUM_VALUE
            bool present = false;
            std::vector<std::string> values;
        };

        ArgParser()
        {
            // Always register --help/-h automatically
            registerArg("help", "Show this help message and exit", FLAG, {"-h"});
        }

        void registerFlag(const std::string& long_name, const std::string& description,
                          const std::vector<std::string>& aliases = {})
        {
            registerArg(long_name, description, FLAG, aliases);
        }

        void registerValue(const std::string& long_name, const std::string& description,
                           const std::vector<std::string>& aliases = {})
        {
            registerArg(long_name, description, VALUE, aliases);
        }

        void registerIntList(const std::string& long_name, const std::string& description,
                             const std::vector<std::string>& aliases = {})
        {
            registerArg(long_name, description, INT_LIST, aliases);
        }

        void registerEnumValue(const std::string& long_name, const std::string& description,
                               const std::vector<std::string>& allowed,
                               const std::vector<std::string>& aliases = {})
        {
            ArgInfo arg{long_name, description, ENUM_VALUE, aliases};
            arg.allowed_values.insert(allowed.begin(), allowed.end());
            registerArg(arg);
        }

        void parse(int argc, char* argv[])
        {
            for (int i = 1; i < argc; ++i)
            {
                std::string token = argv[i];
                if (auto it = lookup.find(token); it != lookup.end())
                {
                    ArgInfo& arg = args[it->second];
                    arg.present = true;

                    if (arg.type == FLAG)
                    {
                        continue; // No value expected
                    }

                    if (i + 1 >= argc)
                    {
                        throw std::runtime_error("Missing value for argument: " + token);
                    }
                    std::string value = argv[++i];
                    handleValue(arg, value);
                }
                else
                {
                    throw std::runtime_error("Unknown argument: " + token);
                }
            }

            // Handle help automatically
            if (isPresent("--help") || isPresent("-h"))
            {
                printHelp();
                std::exit(0);
            }
        }

        bool isPresent(const std::string& name) const
        {
            if (auto it = lookup.find(name); it != lookup.end())
            {
                return args.at(it->second).present;
            }
            return false;
        }

        std::vector<int> getIntList(const std::string& name) const
        {
            std::vector<int> result;
            const auto& vals = getArg(name).values;
            for (const auto& v : vals)
                result.push_back(std::stoi(v));
            return result;
        }

        std::string getValue(const std::string& name) const
        {
            const auto& vals = getArg(name).values;
            return vals.empty() ? "" : vals.front();
        }

        void printHelp() const
        {
            std::cout << "Usage: program [options]\n\n";
            std::cout << "Available arguments:\n";
            for (const auto& arg : args)
            {
                std::cout << "  --" << arg.long_name;
                for (const auto& alias : arg.aliases)
                    std::cout << ", " << alias;
                std::cout << "\n    " << arg.description;

                if (!arg.allowed_values.empty())
                {
                    std::cout << " [Allowed: ";
                    bool first = true;
                    for (const auto& val : arg.allowed_values)
                    {
                        if (!first)
                            std::cout << "|";
                        std::cout << val;
                        first = false;
                    }
                    std::cout << "]";
                }
                std::cout << "\n";
            }
        }

    private:
        std::vector<ArgInfo> args;
        std::unordered_map<std::string, size_t> lookup;

        void registerArg(const std::string& long_name, const std::string& description,
                         ArgType type, const std::vector<std::string>& aliases = {})
        {
            ArgInfo arg{long_name, description, type, aliases};
            registerArg(arg);
        }

        void registerArg(ArgInfo& arg)
        {
            size_t index = args.size();
            args.push_back(arg);
            lookup["--" + arg.long_name] = index;
            for (const auto& alias : arg.aliases)
                lookup[alias] = index;
        }

        void handleValue(ArgInfo& arg, const std::string& value)
        {
            if (arg.type == VALUE)
            {
                arg.values.push_back(value);
            }
            else if (arg.type == INT_LIST)
            {
                std::stringstream ss(value);
                std::string token;
                while (std::getline(ss, token, ','))
                    arg.values.push_back(token);
            }
            else if (arg.type == ENUM_VALUE)
            {
                if (arg.allowed_values.count(value) == 0)
                {
                    throw std::runtime_error("Invalid value for --" + arg.long_name + ": " + value);
                }
                arg.values.push_back(value);
            }
        }

        const ArgInfo& getArg(const std::string& name) const
        {
            auto it = lookup.find(name);
            if (it == lookup.end())
                throw std::runtime_error("Argument not registered: " + name);
            return args.at(it->second);
        }
    };

    // // Example usage
    // int main(int argc, char* argv[])
    // {
    //     ArgParser parser;
    //     parser.registerFlag("non-periodic", "Disable periodic behavior", {"-np"});
    //     parser.registerIntList("N", "Specify a list of integers", {"-N"});
    //     parser.registerEnumValue("device-type", "Choose device type", {"cpu", "gpu"}, {"--dt"});

    //     try
    //     {
    //         parser.parse(argc, argv);

    //         if (parser.isPresent("--non-periodic"))
    //         {
    //             std::cout << "Non-periodic mode enabled.\n";
    //         }

    //         if (parser.isPresent("-N"))
    //         {
    //             auto values = parser.getIntList("-N");
    //             std::cout << "Parsed N values:";
    //             for (auto v : values)
    //                 std::cout << " " << v;
    //             std::cout << "\n";
    //         }

    //         if (parser.isPresent("--device-type"))
    //         {
    //             std::cout << "Device: " << parser.getValue("--device-type") << "\n";
    //         }
    //     }
    //     catch (const std::exception& e)
    //     {
    //         std::cerr << "Error: " << e.what() << "\n";
    //         parser.printHelp();
    //         return 1;
    //     }

    //     return 0;
    // }

}

#endif // MGCL_ARG_PARSER_HPP
