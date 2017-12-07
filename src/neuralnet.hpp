#pragma once
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "math.hpp"
#include "matrix.hpp"

namespace NeuralNetwork {
    /**
     * A matrix with one column and `N` rows is essentially a column vector.
     * The column vector is useful in the training algorithm.
     *
     * @tparam N The size of the column vector.
     */
    template<size_t N>
    using ColumnVector = Matrix::Matrix<double, N, 1>;

    /**
     * A convenience type representing a Matrix of weights of size
     * `CurrentLayerSize * PrevLayerSize`.
     *
     * @tparam CurrentLayerSize The size of the current layer.
     * @tparam PrevLayerSize The size of the previous layer.
     */
    template<size_t CurrentLayerSize, size_t PrevLayerSize>
    using Weights = Matrix::Matrix<double, CurrentLayerSize, PrevLayerSize>;

    /**
     * Data structure representing an instance of a image and its corresponding
     * label.
     *
     * @tparam InputSize The size of the input layer.
     * @tparam OutputSize The size of the output layer.
     */
    template<size_t InputSize, size_t OutputSize>
    struct TrainingLabel {
        size_t value;
        ColumnVector<OutputSize> label;
        ColumnVector<InputSize> input;
    };

    /**
     * Vector of training labels, representing the a data set of training
     * labels.
     *
     * @tparam InputSize The size of the input layer.
     * @tparam OutputSize The size of the output layer.
     */
    template<size_t InputSize, size_t OutputSize>
    using TrainingSet = std::vector<TrainingLabel<InputSize, OutputSize>>;

    /**
     * Class representing a 3-layer neural network.
     *
     * @tparam InputSize The size of input layer.
     * @tparam HiddenSize The size of hidden layer.
     * @tparam OutputSize The size of output layer.
     */
    template<size_t InputSize, size_t HiddenSize, size_t OutputSize>
    class NeuralNetwork {
    public:
        /**
         * Constructs a new neural network.
         *
         * @param learningRate The learning rate of the network.
         * @param verbose Enable verbose logging. Defaults to false.
         */
        NeuralNetwork(const double learningRate, const bool verbose = false) :
            _learningRate{learningRate},
            _verbose{verbose},
            _inputWeights{Matrix::randomMatrix<HiddenSize, InputSize>()},
            _hiddenWeights{Matrix::randomMatrix<OutputSize, HiddenSize>()} {
        }

        /**
         * Queries a result `0 <= result < OutputSize` such that `result`
         * corresponds to be a result with the highest probability of
         * happening.
         *
         * @param input The input vector.
         */
        size_t query(const ColumnVector<InputSize>& input) const {
            auto hiddenOutput = Math::sigmoid(_inputWeights * input);
            auto output = Math::sigmoid(_hiddenWeights * hiddenOutput);

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

        /**
         * Uses the training set given to train the neural network using every
         * training label instance from the data set.
         *
         * @param trainingSet The training data set.
         */
        void train(const TrainingSet<InputSize, OutputSize>& trainingSet) {
            // These values are used for percentages when verbose output is enabled.
            size_t labelNumber = 1;
            auto trainingSetSize = trainingSet.size();

            for (const auto& trainingLabel : trainingSet) {
                // First we preprare the input to the hidden layer and its
                // output using the sigmoid function.
                auto hiddenInput = _inputWeights * trainingLabel.input;
                auto hiddenOutput = Math::sigmoid(hiddenInput);

                // Next, we prepare the input to the output layer as well as
                // its output.
                auto outputInput = _hiddenWeights * hiddenOutput;
                auto output = Math::sigmoid(outputInput);


                // Now, we calculate how far off we are and backpropogate those errors.
                auto outputErrors = trainingLabel.label - output;
                auto hiddenErrors = _hiddenWeights.transpose() * outputErrors;

                // Calculate the derivatives of the error functions.
                auto outputErrorsDerivative = (-outputErrors ^ Math::sigmoid(outputInput, true)) * hiddenOutput.transpose();
                auto hiddenErrorsDerivative = (-hiddenErrors ^ Math::sigmoid(hiddenInput, true)) * trainingLabel.input.transpose();

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

        /**
         * Dump the input and hidden weights to a file for later use.
         *
         * @param file The file to write the weights to.
         * @return True if the network was populated successfully, false if something failed.
         */
        bool dumpWeightsToFile(const std::string& file) const {
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

        /**
         * Load the input and hidden weights from a file in order to skip
         * training.
         *
         * @param file The file to read the weights from.
         * @return True if the network was populated successfully, false if something failed.
         */
        bool loadWeightsFromFile(const std::string& file) {
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

    private:
        double _learningRate;
        bool _verbose;

        Weights<HiddenSize, InputSize> _inputWeights;
        Weights<OutputSize, HiddenSize> _hiddenWeights;

        /**
         * Prints a message if verbose output is enabled.
         *
         * @param message The message to print.
         */
        void printMessage(const std::string& message) const {
            if (!_verbose) return;
            std::cout << message << std::endl;
        }

        /**
         * Prints the current percentage for some computation with some natural
         * sizing. First the title is printed, then the count out of total, and
         * finally the percentage. A carriage return is at the front of the
         * string to ensure that the message is shown on a single line.
         *
         * @param title The title of the computation.
         * @param count The count.
         * @param total The total.
         */
        void printPercentage(const std::string& title, const size_t count, const size_t total) const {
            if (!_verbose) return;
            std::printf(
                "\r%s: %ld / %ld (%.2f%%)",
                title.c_str(),
                count, total, Math::percentage(count, total)
            );
            std::cout << std::flush;
        }

        /**
         * Ends the percentage after using `printPercentage()` by simply
         * adding a newline at the end.
         */
        void endPercentage() const {
            if (!_verbose) return;
            std::cout << std::endl;
        }

        /**
         * Dumps a matrix to a file stream and prints the progress if verbose
         * output is enabled.
         *
         * @tparam N The number of rows.
         * @tparam M The number of columns.
         * @param title The title of computation.
         * @param stream The ouput file stream.
         * @param The weights to serialize.
         */
        template<size_t N, size_t M>
        void dumpMatrix(const std::string& title, std::ofstream& stream, const Weights<N, M>& weights) const {
            size_t entryNumber = 1;
            for (size_t i = 0; i < N; ++i) {
                for (size_t j = 0; j < M; ++j) {
                    stream << weights[i][j] << ' ';
                    printPercentage(title, entryNumber++, N * M);
                }
            }
            endPercentage();
        }

        /**
         * Loads Matrix entries from a file and stores it in the weights given.
         * This function also prints the progress if verbose output is enabled.
         *
         * @tparam N The number of rows.
         * @tparam M The number of columns.
         * @param title The title of computation.
         * @param stream The input file stream.
         * @param The weights to deserialize.
         */
        template<size_t N, size_t M>
        void loadMatrix(const std::string& title, std::ifstream& stream, Weights<N, M>& weights) {
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
    };
} // NeuralNetwork

