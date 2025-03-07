#ifndef NEURON_H
#define NEURON_H

#include <vector>
#include <random>

class Neuron {
private:
    std::vector<double> weights;
    double bias;
    static std::random_device rd;
    static std::mt19937 gen;
    static std::normal_distribution<double> dist;

    void initializeWeights(int inputSize);

public:
    Neuron(int inputSize);
    
    double activate(double sum) const;
    double activateDerivative(double value) const;
    double computeOutput(const std::vector<double>& inputs) const;
    
    const std::vector<double>& getWeights() const;
    double getBias() const;
    
    void updateWeights(const std::vector<double>& inputs, double delta, double learningRate);
};

#endif // NEURON_H