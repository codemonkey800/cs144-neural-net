#pragma once
#include "matrix.hpp"

namespace NeuralNetwork {
    template<size_t Input, size_t Hidden, size_t Output>
    class NeuralNetwork {
    public:
        NeuralNetwork(const double learningRate) : _learningRate{learningRate} {
        }

        ~NeuralNetwork() {
        }

    private:
        double _learningRate;

        Matrix::Matrix<Input, Hidden> _inputWeights;
        Matrix::Matrix<Hidden, Output> _hiddenWeights;
    };
} // NeuralNetwork

