#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include "Neuron.h"

class Layer {
private:
    std::vector<Neuron> neurons;

public:
    Layer(int numNeurons, int inputSize);
    
    std::vector<double> feedForward(const std::vector<double>& inputs) const;
    
    const std::vector<Neuron>& getNeurons() const;
    
    // Add a non-const version to provide mutable access to neurons
    std::vector<Neuron>& getNeurons();
};

#endif // LAYER_H