#include "Layer.h"

Layer::Layer(int numNeurons, int inputSize) {
    neurons.reserve(numNeurons);
    for (int i = 0; i < numNeurons; i++) {
        neurons.emplace_back(inputSize);
    }
}

std::vector<double> Layer::feedForward(const std::vector<double>& inputs) const {
    std::vector<double> outputs(neurons.size());
    for (size_t i = 0; i < neurons.size(); i++) {
        outputs[i] = neurons[i].computeOutput(inputs);
    }
    return outputs;
}

const std::vector<Neuron>& Layer::getNeurons() const {
    return neurons;
}

// Implementation of the non-const version
std::vector<Neuron>& Layer::getNeurons() {
    return neurons;
}