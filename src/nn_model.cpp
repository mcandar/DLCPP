#include <iostream>
#include <Eigen/Dense>
#include <ctime>
#include "data.h"
#include "parameters.h"
#include "models.h"

using namespace models;

int main(int argc, char *argv[])
{
    parameters::parse_arguments(argc, argv);
    clock_t start = std::clock();

    NeuralNetwork nn = NeuralNetwork(
        parameters::learning_rate,
        parameters::epoch);

    Dense dense1 = Dense(4, 100, "dense_input");
    Sigmoid sigm1 = Sigmoid("sigmoid_input");
    Dense dense2 = Dense(100, 40, "dense_hidden");
    Sigmoid sigm2 = Sigmoid("sigmoid_hidden");
    Dense dense3 = Dense(40, 1, "dense_output");
    Sigmoid sigm3 = Sigmoid("sigmoid_output");

    nn.add_dense(&dense1);
    nn.add_sigmoid(&sigm1);
    nn.add_dense(&dense2);
    nn.add_sigmoid(&sigm2);
    nn.add_dense(&dense3);
    nn.add_sigmoid(&sigm3);

    nn.train(data::X, data::y);

    clock_t end = std::clock();
    double elapsed_secs = double(end - start) / CLOCKS_PER_SEC;
    std::cout << "Elapsed: " << elapsed_secs << " seconds." << std::endl;

    return 0;
}