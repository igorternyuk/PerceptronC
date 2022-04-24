#ifndef NET_H
#define NET_H

#include <vector>
#include <cstdlib>
#include <functional>

class Net
{
public:
    Net(const std::vector<unsigned int>& topology);
    void feedForward(const std::vector<double>& inputValues);
    void backProp(std::vector<double>& targetValues);
    void getResults(std::vector<double>& resultValues) const;
    double getError() const { return m_error; }
    static double getRandom();

    struct Connection
    {
        double W;
        double gradW;
    };

    static double sigmoid(double x);
    static double sigmoidDer(double x);
    static double hypTan(double x);
    static double hypTanDer(double x);
    static double relu(double x);
    static double reluDer(double x);

    class Neuron;
    using Layer = std::vector<Neuron>;

    class Neuron
    {
    public:
        Neuron(size_t numOutputs, size_t index, std::function<double(double)> transferFunction,
               std::function<double(double)> transferFunctionDer);
        void feedForward(const Layer& prevLayer);
        void setOutputValue(double val) { m_outputVal = val;};
        double getOutputVal() const { return m_outputVal; }

        void calcOutputGradients(double targetVal);
        void calcHiddenGradients(Layer& nextLayer);
        void updateInputWeights(Layer& prevLayer);
        double sumW(const Layer& nextLayer);

    private:
        inline static double eta = 0.1;
        inline static double alfa = 0.5;
        std::vector<Connection> m_outputWeights;
        double m_outputVal;
        double m_gradient;
        size_t m_index;
        std::function<double(double)> m_transferFunction;
        std::function<double(double)> m_transferFunctionDer;
    };


private:
    void calcOverallError(const std::vector<double>& targetValues);

    std::vector<Layer> m_layers;
    double m_error = 0.0;
};


#endif // NET_H
