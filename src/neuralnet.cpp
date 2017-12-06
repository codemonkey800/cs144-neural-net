#pragma once
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include "activation.h"
#include "activation.cpp"
#include "neuralnet.h"
#include "matrix.cpp"

namespace NeuralNetwork {
    template<size_t InputSize, size_t HiddenSize, size_t OutputSize>
    NeuralNetwork<InputSize, HiddenSize, OutputSize>::NeuralNetwork(const double learningRate, const bool verbose) :
        _learningRate{learningRate},
        _verbose{verbose},
        _inputWeights{Matrix::randomMatrix<HiddenSize, InputSize>()},
        _hiddenWeights{Matrix::randomMatrix<OutputSize, HiddenSize>()} {
    }

    template<size_t InputSize, size_t HiddenSize, size_t OutputSize>
    size_t NeuralNetwork<InputSize, HiddenSize, OutputSize>::query(const ColumnVector<InputSize>& input) const {
        ActivationFunction activator{};
        auto hiddenOutput = activator.activate(_inputWeights * input);
        auto output = activator.activate(_hiddenWeights * hiddenOutput);

        // Pick result with highest probability of happening. It is up to
        // the caller of the API to interpret the result meaning in the
        // context of the data set.{}
        size_t result = 0;
        for (size_t i = 1; i < OutputSize; ++i) {
            if (output[i][0] <= output[result][0]) continue;
            result = i;
        }

        return result;
    }

    template<size_t InputSize, size_t HiddenSize, size_t OutputSize>
    void NeuralNetwork<InputSize, HiddenSize, OutputSize>::train(const TrainingSet<InputSize, OutputSize>& trainingSet) {
        ActivationFunction activator{};

        // These values are used for percentages when verbose output is enabled.
        size_t labelNumber = 1;
        auto trainingSetSize = trainingSet.size();

        for (const auto& trainingLabel : trainingSet) {
            // First we preprare the input to the hidden layer and its
            // output using the sigmoid function.
            auto hiddenInput = _inputWeights * trainingLabel.input;
            auto hiddenOutput = activator.activate(hiddenInput);

            // Next, we prepare the input to the output layer as well as
            // its output.
            auto outputInput = _hiddenWeights * hiddenOutput;
            auto output = activator.activate(outputInput);

            // Now, we calculate how far off we are and backpropogate those errors.
            auto outputErrors = trainingLabel.label - output;
            auto hiddenErrors = _hiddenWeights.transpose() * outputErrors;

            // Calculate the derivatives of the error functions.
            auto outputErrorsDerivative = (-outputErrors ^ activator.activate(outputInput, true)) * hiddenOutput.transpose();
            auto hiddenErrorsDerivative = (-hiddenErrors ^ activator.activate(hiddenInput, true)) * trainingLabel.input.transpose();

            // Update the weights using the derivatives from earlier.
            // Gradient descent slowly minimizes the error over time after
            // many iterations.
            _hiddenWeights = _hiddenWeights - _learningRate * outputErrorsDerivative;
            _inputWeights = _inputWeights - _learningRate * hiddenErrorsDerivative;

            // Finally, print the percentage for training the network.
            printPercentage("Training Network", labelNumber, trainingSetSize);
            labelNumber++;
        }

        endPercentage();
    }

    template<size_t InputSize, size_t HiddenSize, size_t OutputSize>
    bool NeuralNetwork<InputSize, HiddenSize, OutputSize>::dumpWeightsToFile(const std::string& file) const {
        try {
            std::ofstream stream{file};
            dumpMatrix("Dumping Input Weights to File", stream, _inputWeights);
            stream << std::endl;
            dumpMatrix("Dumping Hidden Weights to File", stream, _hiddenWeights);
        } catch (const std::ofstream::failure&) {
            printMessage("Unable to write weights to file.");
            return false;
        }

        return true;
    }

    template<size_t InputSize, size_t HiddenSize, size_t OutputSize>
    bool NeuralNetwork<InputSize, HiddenSize, OutputSize>::loadWeightsFromFile(const std::string& file) {
        try {
            std::ifstream stream{file};
            loadMatrix("Loading Input Weights from File", stream, _inputWeights);
            loadMatrix("Loading Hidden Weights from File", stream, _hiddenWeights);
        } catch (const std::ifstream::failure&) {
            printMessage("Unable to read weights from file. Restorting to full training of network.");
            return false;
        } catch (const std::invalid_argument&) {
            printMessage("Unable to parse weights from file. Restoring to full training of network.");
            return false;
        }

        return true;
    }

    template<size_t InputSize, size_t HiddenSize, size_t OutputSize>
    void NeuralNetwork<InputSize, HiddenSize, OutputSize>::printMessage(const std::string& message) const {
        if (!_verbose) return;
        std::cout << message << std::endl;
    }

    template<size_t InputSize, size_t HiddenSize, size_t OutputSize>
    void NeuralNetwork<InputSize, HiddenSize, OutputSize>::printPercentage(const std::string& title, const size_t count, const size_t total) const {
        if (!_verbose) return;
        std::printf(
            "\r%s: %ld / %ld (%.2f%%)",
            title.c_str(),
            count, total,
            static_cast<double>(count) / static_cast<double>(total) * 100.0
        );
        std::cout << std::flush;
    }

    template<size_t InputSize, size_t HiddenSize, size_t OutputSize>
    void NeuralNetwork<InputSize, HiddenSize, OutputSize>::endPercentage() const {
        if (!_verbose) return;
        std::cout << std::endl;
    }

    template<size_t InputSize, size_t HiddenSize, size_t OutputSize>
    template<size_t N, size_t M>
    void NeuralNetwork<InputSize, HiddenSize, OutputSize>::dumpMatrix(const std::string& title, std::ofstream& stream, const Weights<N, M>& weights) const {
        size_t entryNumber = 1;
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < M; ++j) {
                stream << weights[i][j] << ' ';
                printPercentage(title, entryNumber++, N * M);
            }
        }
        endPercentage();
    }

    template<size_t InputSize, size_t HiddenSize, size_t OutputSize>
    template<size_t N, size_t M>
    void NeuralNetwork<InputSize, HiddenSize, OutputSize>::loadMatrix(const std::string& title, std::ifstream& stream, Weights<N, M>& weights) {
        std::string token;
        size_t entryNumber = 1;
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < M; ++j) {
                std::getline(stream, token, ' ');
                weights[i][j] = std::stod(token);
                printPercentage(title, entryNumber++, N * M);
            }
        }
        endPercentage();
    }
} // NeuralNetwork

