// #ifndef HEADERFILE_H
// #define HEADERFILE_H
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <unordered_map>
#include <cstdio>

namespace models
{

    Eigen::MatrixXd sigmoid(Eigen::MatrixXd X)
    {
        return 1.0 / (1.0 + (-X).array().exp());
    }

    Eigen::MatrixXd sigmoid_derivative(Eigen::MatrixXd X)
    {
        Eigen::MatrixXd temp = sigmoid(X);
        return temp.array() * (1.0 - temp.array());
    }

    double mean_squared_error(Eigen::MatrixXd a, Eigen::MatrixXd b)
    {
        return (a - b).array().pow(2.0).mean();
    }

    Eigen::MatrixXd mean_squared_error_derivative(Eigen::MatrixXd a, Eigen::MatrixXd b)
    {
        double n = a.cols(); // assuming it is a row vector
        return 2.0 * ((a - b).array() / n);
    }

    class RNN
    {
    public:
        Eigen::MatrixXd forward(Eigen::MatrixXd input)
        {
            Eigen::MatrixXd output;
            return output;
        }

        Eigen::MatrixXd backward(Eigen::MatrixXd output_gradient, double learning_rate)
        {
            Eigen::MatrixXd output;
            return output;
        }
    };

    class CONV
    {

    public:
        Eigen::MatrixXd forward(Eigen::MatrixXd input)
        {
            Eigen::MatrixXd output;
            return output;
        }

        Eigen::MatrixXd backward(Eigen::MatrixXd output_gradient, double learning_rate)
        {
            Eigen::MatrixXd output;
            return output;
        }
    };

    class Dense
    {
    public:
        int input_size;
        int output_size;
        std::string name;
        Eigen::MatrixXd weights;
        Eigen::MatrixXd bias;
        Eigen::MatrixXd input;

        Dense(int input_size, int output_size, std::string name = "dense")
        {
            this->input_size = input_size;
            this->output_size = output_size;
            this->name = name;
            this->weights = Eigen::MatrixXd::Random(input_size, output_size);
            this->bias = Eigen::MatrixXd::Random(1, output_size); // row size 1 should be changed later
        }

        Eigen::MatrixXd forward(Eigen::MatrixXd input)
        {
            this->input = input;
            return (input * weights).array() + bias.array();
        }

        Eigen::MatrixXd backward(Eigen::MatrixXd output_gradient, double learning_rate)
        {
            Eigen::MatrixXd x_grad = this->weights * output_gradient;
            Eigen::MatrixXd weights_grad = output_gradient * this->input;

            weights = weights.array() - (learning_rate * weights_grad.transpose().array()).array();
            bias = bias.array() - (learning_rate * output_gradient.transpose().array()).array();

            return x_grad;
        }
    };

    // std::ostream &operator<<(ostream &os, const Dense dense)
    // {
    //     os << "Dense.name: " << dense.name << std::endl
    //        << "Dense.weight: " << std::endl
    //        << dense.weights << std::endl
    //        << "Dense.bias: " << std::endl
    //        << dense.bias << std::endl;
    //     return os;
    // }

    class Sigmoid
    {
    public:
        std::string name;

        Sigmoid(std::string name = "sigmoid")
        {
            this->name = name;
        }

        Eigen::MatrixXd forward(Eigen::MatrixXd input)
        {
            return sigmoid(input);
        }

        Eigen::MatrixXd backward(Eigen::MatrixXd output_gradient, double learning_rate)
        {
            return sigmoid_derivative(output_gradient);
        }
    };

    // write a destructor too
    class NeuralNetwork
    {
    public:
        int n = 0;
        // TODO: use enum for declaration
        static constexpr int ENUM_DENSE = 0;
        static constexpr int ENUM_RNN = 1;
        static constexpr int ENUM_CONV = 2;
        static constexpr int ENUM_SIGMOID = 3;

        int LAYER_CODE;

        double learning_rate;
        int epoch;
        int batch_size;

        std::unordered_map<int, Dense *> dense_map;
        std::unordered_map<int, RNN *> rnn_map;
        std::unordered_map<int, CONV *> conv_map;
        std::unordered_map<int, Sigmoid *> sigmoid_map;
        std::unordered_map<int, int> layer_order;

        NeuralNetwork(double learning_rate, int epoch, int batch_size = 1)
        {
            this->learning_rate = learning_rate;
            this->epoch = epoch;
            this->batch_size = batch_size;
        }

        void add_dense(Dense *layer)
        {
            dense_map[n] = layer;
            layer_order[n] = ENUM_DENSE;
            n++;
        }

        void add_rnn(RNN *layer)
        {
            rnn_map[n] = layer;
            layer_order[n] = ENUM_RNN;
            n++;
        }

        void add_conv(CONV *layer)
        {
            conv_map[n] = layer;
            layer_order[n] = ENUM_CONV;
            n++;
        }

        void add_sigmoid(Sigmoid *layer)
        {
            sigmoid_map[n] = layer;
            layer_order[n] = ENUM_SIGMOID;
            n++;
        }

        Eigen::MatrixXd step_forward(const int i, const Eigen::MatrixXd x)
        {
            Eigen::MatrixXd output;

            const int layer_code = layer_order[i];

            switch (layer_code)
            {
            case ENUM_DENSE:
                output = dense_map[i]->forward(x);
                break;
            case ENUM_RNN:
                output = rnn_map[i]->forward(x);
                break;
            case ENUM_CONV:
                output = conv_map[i]->forward(x);
                break;
            case ENUM_SIGMOID:
                output = sigmoid_map[i]->forward(x);
                break;
            default:
                break;
            }

            return output;
        }

        Eigen::MatrixXd step_backward(const int i, const Eigen::MatrixXd x)
        {
            Eigen::MatrixXd output;

            const int layer_code = layer_order[i];

            switch (layer_code)
            {
            case ENUM_DENSE:
                output = dense_map[i]->backward(x, learning_rate);
                break;
            case ENUM_RNN:
                output = rnn_map[i]->backward(x, learning_rate);
                break;
            case ENUM_CONV:
                output = conv_map[i]->backward(x, learning_rate);
                break;
            case ENUM_SIGMOID:
                output = sigmoid_map[i]->backward(x, learning_rate);
                break;
            default:
                break;
            }

            return output;
        }

        Eigen::MatrixXd forward(const Eigen::MatrixXd x)
        {
            Eigen::MatrixXd output = x;

            for (int i = 0; i < n; ++i)
            {
                output = step_forward(i, output);
            }

            return output;
        }

        Eigen::MatrixXd backward(const Eigen::MatrixXd x)
        {
            Eigen::MatrixXd output = x;

            for (int i = n - 1; i >= 0; --i)
            {
                output = step_backward(i, output);
            }

            return output;
        }

        Eigen::MatrixXd slice_X(Eigen::MatrixXd X, int ri, int rj, int ci, int cj)
        {
            return X(Eigen::seq(ri, rj), Eigen::seq(ci, cj));
        }

        Eigen::MatrixXd slice_y(Eigen::MatrixXd y, int ri, int rj, int ci, int cj)
        {
            return y(Eigen::seq(ri, rj), Eigen::seq(ci, cj));
        }

        void train(Eigen::MatrixXd X, Eigen::MatrixXd y)
        {
            // TODO: get this info from layers
            const int input_row_size = X.rows();
            const int input_column_size = X.cols();
            const int output_row_size = y.rows();
            const int output_column_size = y.cols();
            const int sample_row_size = 1; // TODO: change this later according to input sample

            double err;
            Eigen::MatrixXd err_grad, grad, output;
            // Eigen::Matrix<double, sample_row_size, input_column_size> sample_X;
            // Eigen::Matrix<double, sample_row_size, output_column_size> sample_y;

            Eigen::MatrixXd sample_X;
            Eigen::MatrixXd sample_y;

            for (int i = 0; i < epoch; ++i)
            {
                err = 0.0;
                for (int j = 0; j < input_row_size; ++j)
                {
                    sample_X = slice_X(X, j, j, 0, input_column_size - 1);
                    sample_y = slice_y(y, j, j, 0, output_column_size - 1);

                    output = forward(sample_X);
                    err += mean_squared_error(output, sample_y);
                    err_grad = mean_squared_error_derivative(output, sample_y);
                    grad = backward(err_grad);
                }
                err = err / input_row_size;
                if (i % 10 == 0)
                {
                    std::printf("%d Error: %f \n", i, err);
                }
            }

            std::cout << output << std::endl;
        }
    };
}
// #endif