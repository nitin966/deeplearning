#include <iostream>
#include "neuralnet/NeuralNetwork.h"

int main() {
    // Define the neural network architecture
    int inputSize = 2;                        // Number of input neurons
    std::vector<int> hiddenSizes = {4};       // Hidden layers
    int outputSize = 1;                       // One output neuron
    double learningRate = 0.1;                // Learning rate

    // Create the neural network

    NeuralNetwork nn(inputSize, hiddenSizes, outputSize, learningRate);

    // Sample training data (XOR problem)
    std::vector<std::vector<double>> inputs = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1}
    };
    std::vector<std::vector<double>> targets = {
        {0}, {1}, {1}, {0}
    };

    // Train the network
    int epochs = 10000;
    nn.train(inputs, targets, epochs);

    // Test the network
    std::cout << "\nTesting Neural Network:\n";
    for (const auto& input : inputs) {
        std::vector<double> output = nn.predict(input);
        std::cout << "Input: (" << input[0] << ", " << input[1] << ") => Output: " << output[0] << "\n";
    }

    return 0;
}
