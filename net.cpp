#include "net.h"
#include <random>
#include <functional>

#define SIGMOID(x) (1.0 / (1.0 + exp(-(x))))
#define RELU(x) ((x) > 0 ? (x) : 0.0)

Net::Net(const std::vector<unsigned int> &topology)
{
    const unsigned int numLayer = topology.size();
    for(size_t li = 0; li < numLayer; ++li)
    {
        m_layers.push_back(Layer());
        const size_t numOutputs = li == numLayer - 1 ? 0 : topology[li + 1];
        for(size_t ni = 0; ni <= topology[ni]; ++ni)
        {
            m_layers.back().push_back(Neuron(numOutputs, ni));
        }
    }
}

void Net::feedForward(const std::vector<double> &inputValues)
{
    for(size_t ni = 0; ni < inputValues.size(); ++ni)
    {
        m_layers[0][ni].setOutputValue(inputValues[ni]);
    }

    for(size_t li = 0; li < m_layers.size(); ++li)
    {
        const Layer& prevLayer = m_layers[li-1];
        for(size_t ni = 0; ni < m_layers[li].size(); ++ni)
        {
            m_layers[li][ni].feedForward(prevLayer);
        }
    }
}

void Net::backProp(std::vector<double> &targetValues)
{

}

void Net::getResults(std::vector<double> &resultValues) const
{

}

double Net::getRandomWeight()
{
    std::mt19937::result_type seed = time(0);
    auto real_rand = std::bind(std::uniform_real_distribution<double>(0,1), std::mt19937(seed));
    return real_rand();
}

Net::Neuron::Neuron(size_t numOutputs, size_t index): m_index(index)
{
    for(size_t ni = 0; ni < numOutputs; ++ni)
    {
        Connection c;
        c.W = getRandomWeight();
        c.gradW = 0.0;
        m_outputWeights.push_back(c);
    }
}

void Net::Neuron::feedForward(const Layer &prevLayer)
{
    double total = 0.0;
    for(size_t ni = 0; ni < prevLayer.size(); ++ni)
    {
        total += prevLayer[ni].m_outputVal * prevLayer[ni].m_outputWeights[m_index].W;
    }
    m_outputVal = SIGMOID(total);
}
