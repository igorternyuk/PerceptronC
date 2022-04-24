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
        double getOutputVal() { return m_outputVal; }
    private:
        std::vector<Connection> m_outputWeights;
        double m_outputVal;
        size_t m_index;
    };



private:
    static double getRandomWeight();
    std::vector<Layer> m_layers;
};


#endif // NET_H
