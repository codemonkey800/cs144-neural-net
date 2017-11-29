#pragma once
#include <vector>
#include "matrix.hpp"

namespace NeuralNetwork {
    const size_t IMAGE_SIZE = 28;

    struct TrainingInput {
        int label;
        Matrix::Matrix<IMAGE_SIZE, IMAGE_SIZE> image;
    };

    using TrainingSet = std::vector<TrainingInput>;

    template<size_t Input, size_t Hidden, size_t Output>
    class NeuralNetwork {
    public:
        NeuralNetwork(const double learningRate) : _learningRate{learningRate} {
        }

        int query(const Matrix::ColumnVector<Input>& input) const {
        }

        void train(const TrainingSet& trainingSet) {
        }

    private:
        double _learningRate;

        Matrix::Matrix<Input, Hidden> _inputWeights;
        Matrix::Matrix<Hidden, Output> _hiddenWeights;
    };
} // NeuralNetwork

