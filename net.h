#ifndef NET_H
#define NET_H

#include <vector>
#include <cstdlib>

class Net
{
public:
    Net(const std::vector<unsigned int>& topology);
    void feedForward(const std::vector<double>& inputValues);
    void backProp(std::vector<double>& targetValues);
    void getResults(std::vector<double>& resultValues) const;

    struct Connection
    {
        double W;
        double gradW;
    };

    class Neuron;
    using Layer = std::vector<Neuron>;

    class Neuron
    {
    public:
        Neuron(size_t numOutputs, size_t index);
        void feedForward(const Layer& prevLayer);
        void setOutputValue(double val) { m_outputVal = val;};
        double getOutputVal() const { return m_outputVal; }
        void calcOutputGradients(double targetVal);
        void calcHiddenGradients(Layer& nextLayer);
        void updateInputWeights(Layer& prevLayer);
        double sumW(const Layer& nextLayer);
        static double transferFunction(double x);
        static double transferFunctionDer(double x);
    private:
        inline static double eta = 0.2;
        inline static double alfa = 0.5;
        std::vector<Connection> m_outputWeights;
        double m_outputVal;
        double m_gradient;
        size_t m_index;
    };


private:
    void calcOverallError(const std::vector<double>& targetValues);
    static double getRandomWeight();
    std::vector<Layer> m_layers;
    double m_error = 0.0;
};


#endif // NET_H
