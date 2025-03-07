#include "NeuralNetwork.h"
#include <iostream>
#include <cmath>

NeuralNetwork::NeuralNetwork(int inputSize, const std::vector<int>& hiddenSizes, int outputSize, double learningRate) 
    : learningRate(learningRate) {
    
    // Input to first hidden layer
    layers.emplace_back(hiddenSizes[0], inputSize);
    
    // Hidden layers
    for (size_t i = 1; i < hiddenSizes.size(); i++) {
        layers.emplace_back(hiddenSizes[i], hiddenSizes[i - 1]);
    }
    
    // Output layer
    layers.emplace_back(outputSize, hiddenSizes.back());
}

std::vector<double> NeuralNetwork::feedForward(const std::vector<double>& inputs) const {
    std::vector<double> activations = inputs;
    
    for (const auto& layer : layers) {
        activations = layer.feedForward(activations);
    }
    
    return activations;
}

void NeuralNetwork::backpropagate(const std::vector<double>& inputs, const std::vector<double>& targets) {
    // Store all activations at each layer
    std::vector<std::vector<double>> activationsList;
    activationsList.push_back(inputs);
    
    std::vector<double> currentActivations = inputs;
    
    // Forward pass to store all activations
    for (const auto& layer : layers) {
        currentActivations = layer.feedForward(currentActivations);
        activationsList.push_back(currentActivations);
    }
    
    // Store deltas for each layer
    std::vector<std::vector<double>> layerDeltas(layers.size());
    
    // Calculate output layer deltas
    const int outputLayerIndex = layers.size() - 1;
    layerDeltas[outputLayerIndex].resize(layers[outputLayerIndex].getNeurons().size());
    
    for (size_t j = 0; j < layers[outputLayerIndex].getNeurons().size(); j++) {
        double output = activationsList[outputLayerIndex + 1][j];
        double error = targets[j] - output; // Calculate error
        
        // Use derivative of sigmoid: output * (1 - output)
        layerDeltas[outputLayerIndex][j] = error * layers[outputLayerIndex].getNeurons()[j].activateDerivative(output);
    }
    
    // Backpropagate through hidden layers
    for (int l = outputLayerIndex - 1; l >= 0; l--) {
        layerDeltas[l].resize(layers[l].getNeurons().size());
        
        // Calculate deltas for current layer
        for (size_t j = 0; j < layers[l].getNeurons().size(); j++) {
            double output = activationsList[l + 1][j];
            
            // Compute error signal from next layer
            double error = 0.0;
            const Layer& nextLayer = layers[l + 1];
            const std::vector<double>& nextDeltas = layerDeltas[l + 1];
            
            for (size_t k = 0; k < nextLayer.getNeurons().size(); k++) {
                error += nextLayer.getNeurons()[k].getWeights()[j] * nextDeltas[k];
            }
            
            // Calculate delta = error * derivative of activation function
            layerDeltas[l][j] = error * layers[l].getNeurons()[j].activateDerivative(output);
        }
    }
    
    // Update weights using calculated deltas
    for (size_t l = 0; l < layers.size(); l++) {
        Layer& layer = layers[l];
        const std::vector<double>& prevActivations = activationsList[l];
        const std::vector<double>& deltas = layerDeltas[l];
        
        for (size_t j = 0; j < layer.getNeurons().size(); j++) {
            Neuron& neuron = layer.getNeurons()[j];
            neuron.updateWeights(prevActivations, deltas[j], learningRate);
        }
    }
}

void NeuralNetwork::train(const std::vector<std::vector<double>>& inputs, 
                          const std::vector<std::vector<double>>& targets, 
                          int epochs) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        double totalLoss = 0;
        
        for (size_t i = 0; i < inputs.size(); i++) {
            std::vector<double> output = feedForward(inputs[i]);
            double loss = computeLoss(output, targets[i]);
            totalLoss += loss;
            backpropagate(inputs[i], targets[i]);
        }
        
        if (epoch % 1000 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << totalLoss / inputs.size() << std::endl;
        }
    }
}

double NeuralNetwork::computeLoss(const std::vector<double>& output, const std::vector<double>& target) const {
    double loss = 0;
    for (size_t i = 0; i < output.size(); i++) {
        // Use mean squared error instead of absolute error
        double diff = target[i] - output[i];
        loss += diff * diff;
    }
    return loss / output.size(); // Return average loss
}

std::vector<double> NeuralNetwork::predict(const std::vector<double>& input) const {
    return feedForward(input);
}