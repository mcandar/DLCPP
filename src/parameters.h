#include <iostream>
#include <string>
#include <unordered_map>

namespace parameters
{

    // Defaults
    int epoch = 100;
    double learning_rate = 0.0001;

    void parse_arguments(int argc, char *argv[])
    {
        if (argc > 1)
        {
            for (int i = 1; i < argc - 1; i++)
            {
                std::string arg = argv[i];
                std::string arg_next = argv[i + 1];

                if (arg == "-e" || arg == "--epoch")
                {
                    epoch = std::stoi(arg_next);
                }
                else if (arg == "-lr" || arg == "--learning_rate")
                {
                    learning_rate = std::stod(arg_next);
                }
            }
        }
    }
}