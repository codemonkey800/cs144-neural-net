#pragma once
#include <fstream>
#include <string>
#include <vector>
#include "matrix.h"

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
        explicit NeuralNetwork(const double learningRate, const bool verbose = false);

        /**
         * Queries a result `0 <= result < OutputSize` such that `result`
         * corresponds to be a result with the highest probability of
         * happening.
         *
         * @param input The input vector.
         */
        size_t query(const ColumnVector<InputSize>& input) const;

        /**
         * Uses the training set given to train the neural network using every
         * training label instance from the data set.
         *
         * @param trainingSet The training data set.
         */
        void train(const TrainingSet<InputSize, OutputSize>& trainingSet);

        /**
         * Dump the input and hidden weights to a file for later use.
         *
         * @param file The file to write the weights to.
         * @return True if the network was populated successfully, false if something failed.
         */
        bool dumpWeightsToFile(const std::string& file) const;

        /**
         * Load the input and hidden weights from a file in order to skip
         * training.
         *
         * @param file The file to read the weights from.
         * @return True if the network was populated successfully, false if something failed.
         */
        bool loadWeightsFromFile(const std::string& file);

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
        void printMessage(const std::string& message) const;

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
        void printPercentage(const std::string& title, const size_t count, const size_t total) const;

        /**
         * Ends the percentage after using `printPercentage()` by simply
         * adding a newline at the end.
         */
        void endPercentage() const;

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
        void dumpMatrix(const std::string& title, std::ofstream& stream, const Weights<N, M>& weights) const;

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
        void loadMatrix(const std::string& title, std::ifstream& stream, Weights<N, M>& weights);
    };
} // NeuralNetwork

