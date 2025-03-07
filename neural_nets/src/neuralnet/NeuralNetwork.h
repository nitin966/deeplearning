#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include "Layer.h"

class NeuralNetwork {
private:
    std::vector<Layer> layers;
    double learningRate;
    
    double computeHiddenError(int layerIndex, int neuronIndex, const std::vector<double>& nextLayerDeltas);
    double computeLoss(const std::vector<double>& output, const std::vector<double>& target) const;

public:
    NeuralNetwork(int inputSize, const std::vector<int>& hiddenSizes, int outputSize, double learningRate);
    
    std::vector<double> feedForward(const std::vector<double>& inputs) const;
    void backpropagate(const std::vector<double>& inputs, const std::vector<double>& targets);
    
    void train(const std::vector<std::vector<double>>& inputs, 
               const std::vector<std::vector<double>>& targets, 
               int epochs);
               
    std::vector<double> predict(const std::vector<double>& input) const;
};

#endif // NEURAL_NETWORK_H