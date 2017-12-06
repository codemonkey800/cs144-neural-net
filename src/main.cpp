#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "matrix.h"
#include "matrix.cpp"
#include "neuralnet.h"
#include "neuralnet.cpp"

/**
 * Takes a pixel value in [0, 255] and "normalizes" it. This is done by dividing
 * the pixel value by 255, multiplying it by 0.99 for scale, and then
 * adding 0.01 to shift the range to [0.01, 1.0].
 *
 * @param pixel The pixel value in range [0, 255].
 * @return The pixel normalized to be in the range [0.01, 1.0].
 */
inline double normalizePixel(const int pixel) {
    return (static_cast<double>(pixel) / 255.0) * 0.99 + 0.01;
}

inline double percentage(const double count, const double total) {
    return count / total;
}

/**
 * Parses the current line as a training label. The training label contains the
 * correct value (`trainingLabel.label`) and the input data
 * (`trainingLabel.input`).
 *
 * @tparam InputSize The size of the input layer.
 * @param line The current line to parse.
 * @return The parsed training label.
 */
template<size_t InputSize, size_t OutputSize>
NeuralNetwork::TrainingLabel<InputSize, OutputSize> parseInput(const std::string& line) {
    NeuralNetwork::TrainingLabel<InputSize, OutputSize> trainingLabel{};
    std::istringstream stream{line};
    std::string token;

    // Parse correct value, or label.
    std::getline(stream, token, ',');
    trainingLabel.value = std::stoi(token);

    // Prepare label column vector. For index `i` corresponding to integers 0
    // through `OutputSize`, we assign 1 to the neuron that has the correct
    // output signal, and 0.01 to the neurons with the incorrect output signal.
    for (size_t i = 0; i < OutputSize; ++i) {
        trainingLabel.label[i][0] = trainingLabel.value == i ? 1 : 0.01;
    }

    // Parse image data into column vector.
    for (size_t i = 0; i < InputSize; ++i) {
        std::getline(stream, token, ',');
        int pixel = std::stoi(token);
        trainingLabel.input[i][0] = normalizePixel(pixel);
    }

    return trainingLabel;
}

/**
 * Parses standard input line by line and builds a training data set.
 *
 * @tparam InputSize The size of the input layer.
 * @return The training data set.
 */
template<size_t InputSize, size_t OutputSize>
NeuralNetwork::TrainingSet<InputSize, OutputSize> parseTrainingSet() {
    NeuralNetwork::TrainingSet<InputSize, OutputSize> trainingSet{};

    for (std::string line; std::getline(std::cin, line); ) {
        trainingSet.push_back(parseInput<InputSize, OutputSize>(line));
    }

    return trainingSet;
}

/**
 * Counts the number of correct neural network predictions by iterating through
 * the training data set, querying the network, and comparing the result to the
 * expected value.
 *
 * @tparam InputSize The size of the input layer.
 * @tparam HiddenSize The size of the hidden layer.
 * @tparam OutputSize The size of the output layer.
 * @param network The neural network.
 * @param trainingSet The training set.
 * @param verbose Flag to print verbose info or not.
 * @return The number of correct predictions.
 */
template<size_t InputSize, size_t HiddenSize, size_t OutputSize>
size_t countCorrectPredictions(
    const NeuralNetwork::NeuralNetwork<InputSize, HiddenSize, OutputSize>& network,
    const NeuralNetwork::TrainingSet<InputSize, OutputSize>& trainingSet,
    const bool verbose
) {
    size_t count = 0;
    size_t labelNumber = 1;
    auto trainingSetSize = trainingSet.size();

    for (const auto& trainingLabel : trainingSet) {
        auto result = network.query(trainingLabel.input);
        if (trainingLabel.value == result) count++;
        if (!verbose) continue;

        // If verbose output is enabled, print out current label number and the
        // number of matches, as well as their percentages out of the total.
        std::printf(
            "\rCounting Correct Predictions: %ld / %ld (%.2f%%), %ld matches (%.2f%%)",
            labelNumber, trainingSetSize, percentage(labelNumber, trainingSetSize),
            count, percentage(count, trainingSetSize)
        );
        // We flush the output so that the cursor stays at the end of the
        // console. If this line is missing, the cursor will constantly appear
        // to go back and forth.
        std::cout << std::flush;
        labelNumber++;
    }

    if (verbose) std::cout << std::endl;

    return count;
}

/**
 * Helper function that calculates the amount of time it takes for a function
 * `func` to run. The function `func` should contain all the logic we want to
 * time. The function will most likely be a lambda that performs specific
 * functionality that may be long-running.
 *
 * @param func The function to time.
 * @return The duration in milliseconds.
 */
std::chrono::milliseconds timeFunction(const std::function<void()>& func) {
    auto startTime = std::chrono::high_resolution_clock::now();
    func();
    auto endTime = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
}

/**
 * Prints the help message for this program. Because the executable can be run
 * in various contexts, we use the `argv[0]` value given int `main()` to print
 * out its usage.
 *
 * @param The exe for this program.
 */
void printHelp(const char* exe) {
    std::cout << "Usage: " << exe << " [-v|-d|-l] < data/mnist_test.csv"
              << std::endl << std::endl
              << "Flags:" << std::endl
              << "  -v - Enable verbose output." << std::endl
              << "  -d - Dump network weights after training." << std::endl
              << "  -l - Load network weights from previous training." << std::endl;
}

int main(const int argc, const char* argv[]) {
    // CLI boolean flag
    bool verbose = false;
    bool dumpWeights = false;
    bool loadWeights = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-v") == 0) verbose = true;
        else if (std::strcmp(argv[i], "-d") == 0) dumpWeights = true;
        else if (std::strcmp(argv[i], "-l") == 0) loadWeights = true;
        else {
            // If we received an unrecognized flag, then we print the help
            // message and exit.
            printHelp(argv[0]);
            return 0;
        }
    }

    const std::string weightsFile = "weights.data";

    // Neural network input paramters
    const size_t inputSize = 784;
    const size_t hiddenSize = 300;
    const size_t outputSize = 10;
    const double learningRate = 0.3;

    NeuralNetwork::NeuralNetwork<inputSize, hiddenSize, outputSize> network{learningRate, verbose};
    NeuralNetwork::TrainingSet<inputSize, outputSize> trainingSet;

    // Begin parsing, training, and matching. These are all long-running
    // computations, so we time their execution and print it out at the end for
    // statistics.
    auto parseTime = timeFunction([&trainingSet]() {
        trainingSet = parseTrainingSet<inputSize, outputSize>();
    });

    auto trainTime = timeFunction([&network, &trainingSet, &weightsFile, loadWeights]() {
        // If the load weights flag is passed and if the network is able to
        // load from the file, then we can skip training.
        if (loadWeights && network.loadWeightsFromFile(weightsFile)) return;
        network.train(trainingSet);
    });

    // To save weights for later use, we can dump them to a file if the dump
    // weights flag is passed.
    if (dumpWeights) network.dumpWeightsToFile(weightsFile);

    size_t matches;
    auto matchTime = timeFunction([&matches, &network, &trainingSet, verbose] {
        matches = countCorrectPredictions(network, trainingSet, verbose);
    });

    auto trainingSetSize = trainingSet.size();
    std::cout << "Neural Network Stats:" << std::endl;
    std::printf(
        "  Matches: %ld / %ld (%.2f%%)\n",
        matches, trainingSetSize, percentage(matches, trainingSetSize)
    );
    std::cout << "  Parsing time: " << parseTime.count() << "ms" << std::endl
              << "  Training time: " << trainTime.count() << "ms" << std::endl
              << "  Matching time: " << matchTime.count() << "ms" << std::endl;
}

