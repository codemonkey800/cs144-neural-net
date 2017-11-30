#pragma once
#include <vector>
#include "math.hpp"
#include "matrix.hpp"

namespace NeuralNetwork {
    template<size_t InputSize>
    using InputVector = Matrix::Matrix<double, InputSize, 1>;

    template<size_t CurrentLayerSize, size_t PrevLayerSize>
    using Weights = Matrix::Matrix<double, CurrentLayerSize, PrevLayerSize>;

    template<size_t InputSize>
    struct TrainingLabel {
        int label;
        InputVector<InputSize> input;
    };

    template<size_t InputSize>
    using TrainingSet = std::vector<TrainingLabel<InputSize>>;

    template<size_t InputSize, size_t HiddenSize, size_t OutputSize>
    class NeuralNetwork {
    public:
        NeuralNetwork(const double learningRate) :
            _learningRate{learningRate},
            _inputWeights{Matrix::randomMatrix<HiddenSize, InputSize>()},
            _hiddenWeights{Matrix::randomMatrix<OutputSize, HiddenSize>()} {
        }

        int query(const InputVector<InputSize>& input) const {
            auto hiddenOutput = Math::sigmoid(_inputWeights * input);
            auto output = Math::sigmoid(_hiddenWeights * hiddenOutput);

            // Pick result with highest probability of happening.
            int result = 0;
            for (int i = 1; i < OutputSize; ++i) {
                if (output[i][0] <= output[result][0]) continue;
                result = i;
            }

            return result;
        }

        void train(const TrainingSet<InputSize>& trainingSet, const bool verbose = false) {
        }

    private:
        double _learningRate;

        Weights<HiddenSize, InputSize> _inputWeights;
        Weights<OutputSize, HiddenSize> _hiddenWeights;
    };
} // NeuralNetwork

