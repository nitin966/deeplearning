# Simple Neural Network Implementation in C++

This repository contains a basic implementation of a feedforward neural network in C++. The neural network supports multiple layers, sigmoid activation, and is trained using backpropagation with gradient descent.

## Features

- Modular design with separate classes for neurons, layers, and the full network
- Sigmoid activation function
- Backpropagation algorithm for training
- Support for arbitrary network architectures
- Weight initialization using Xavier/Glorot initialization for better convergence
- Mean squared error loss function

## Classes Overview

### Neuron

The `Neuron` class represents a single neuron in the network:

- Maintains weights and bias
- Computes activation using sigmoid function
- Provides weight update functionality during backpropagation
- Uses proper weight initialization for better training

### Layer

The `Layer` class represents a layer of neurons:

- Contains a collection of Neuron objects
- Handles forward propagation through the layer
- Provides access to individual neurons for backpropagation

### NeuralNetwork

The `NeuralNetwork` class ties everything together:

- Constructs a network with specified architecture
- Implements feedforward for predictions
- Implements backpropagation for training
- Provides training and prediction interfaces
- Calculates loss to monitor training progress

## Example Usage

Here's a simple example showing how to create and train a neural network to learn the XOR function:

```cpp
#include "NeuralNetwork.h"
#include <iostream>
#include <vector>

int main() {
    // Create XOR training data
    std::vector<std::vector<double>> inputs = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1}
    };
    std::vector<std::vector<double>> targets = {
        {0}, {1}, {1}, {0}
    };
    
    // Create neural network with 2 inputs, 1 hidden layer with 4 neurons, and 1 output
    std::vector<int> hiddenSizes = {4};
    double learningRate = 0.1;
    NeuralNetwork network(2, hiddenSizes, 1, learningRate);
    
    // Train the network
    network.train(inputs, targets, 10000);
    
    // Test the network
    std::cout << "Testing Neural Network:" << std::endl;
    for (size_t i = 0; i < inputs.size(); i++) {
        std::vector<double> output = network.predict(inputs[i]);
        std::cout << "Input: (" << inputs[i][0] << ", " << inputs[i][1] 
                  << ") => Output: " << output[0] << std::endl;
    }
    
    return 0;
}
```

## Implementation Details

### Weight Initialization

Weights are initialized using the Xavier/Glorot initialization method, which scales the random values by the square root of the number of inputs to the neuron. This helps prevent vanishing or exploding gradients during training.

### Activation Function

The implementation uses the sigmoid activation function:

```
f(x) = 1 / (1 + exp(-x))
```

The derivative used for backpropagation is:

```
f'(x) = f(x) * (1 - f(x))
```

### Backpropagation

The backpropagation algorithm:

1. Performs a forward pass to calculate all neuron activations
2. Calculates the output layer error using the target values
3. Propagates the error backward through the network
4. Updates weights proportionally to their contribution to the error

## Future Improvements

Potential enhancements to the current implementation:

- Add support for different activation functions (ReLU, tanh, etc.)
- Implement mini-batch gradient descent for faster training
- Add regularization methods (L1, L2) to prevent overfitting
- Add momentum and adaptive learning rates
- Implement dropout for better generalization
- Add serialization/deserialization for saving and loading models

## Building and Running

To build the project, use a C++ compiler that supports C++11 or later:

```bash
g++ -std=c++11 -o neural_network main.cpp NeuralNetwork.cpp Layer.cpp Neuron.cpp
```

Run the executable:

```bash
./neural_network
```
