#include "Neuron.h"
#include <cmath>

// Initialize static members
std::random_device Neuron::rd;
std::mt19937 Neuron::gen(rd());
std::normal_distribution<double> Neuron::dist(0.0, 1.0);

Neuron::Neuron(int inputSize) {
    weights.resize(inputSize);
    bias = dist(gen) / std::sqrt(inputSize);
    initializeWeights(inputSize);
}

void Neuron::initializeWeights(int inputSize) {
    double scale = 1.0 / std::sqrt(inputSize);
    for (size_t i = 0; i < weights.size(); i++) {
        weights[i] = dist(gen) * scale;
    }
}

double Neuron::activate(double sum) const {
    return 1.0 / (1.0 + std::exp(-sum)); // Sigmoid activation
}

double Neuron::activateDerivative(double value) const {
    return value * (1.0 - value); // Sigmoid derivative
}

double Neuron::computeOutput(const std::vector<double>& inputs) const {
    double sum = bias;
    for (size_t i = 0; i < inputs.size(); i++) {
        sum += inputs[i] * weights[i];
    }
    return activate(sum);
}

const std::vector<double>& Neuron::getWeights() const {
    return weights;
}

double Neuron::getBias() const {
    return bias;
}

void Neuron::updateWeights(const std::vector<double>& inputs, double delta, double learningRate) {
    for (size_t i = 0; i < weights.size(); i++) {
        weights[i] += learningRate * delta * inputs[i];
    }
    bias += learningRate * delta;
}