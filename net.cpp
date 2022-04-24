#include "net.h"
#include <random>
#include <functional>

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
    calcOverallError(targetValues);

    //Calculate output layer gradients

    Layer& outputLayer = m_layers.back();
    for(size_t ni = 0; ni < outputLayer.size(); ++ni)
        outputLayer[ni].calcOutputGradients(targetValues[ni]);

    //Calculate gradients on hidden layers

    for(size_t li = m_layers.size() - 2; li > 0; --li)
    {
        Layer& currLayer = m_layers[li];
        Layer& nextLayer = m_layers[li + 1];

        for(size_t ni = 0; ni < currLayer.size(); ++ni)
            currLayer[ni].calcHiddenGradients(nextLayer);
    }

    //Update input weights

    for(size_t li = m_layers.size() - 1; li > 0; --li)
    {
        Layer& currLayer = m_layers[li];
        Layer& prevLayer = m_layers[li - 1];

        for(size_t ni = 0; ni < currLayer.size(); ++ni)
            currLayer[ni].updateInputWeights(prevLayer);
    }
}

void Net::getResults(std::vector<double> &resultValues) const
{
    resultValues.clear();
    const Layer& lastLayer = m_layers.back();
    for(size_t ni = 0; ni < lastLayer.size(); ++ni)
        resultValues.push_back(lastLayer.at(ni).getOutputVal());
}

void Net::calcOverallError(const std::vector<double>& targetValues)
{
    const Layer& lastLayer = m_layers.back();
    m_error = 0.0;
    for(size_t ni = 0; ni < lastLayer.size(); ++ni)
    {
        double delta = targetValues[ni] - lastLayer[ni].getOutputVal();
        m_error += delta * delta;
    }

    m_error /= lastLayer.size();
    m_error = sqrt(m_error);
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
    m_outputVal = transferFunction(total);
}

void Net::Neuron::calcOutputGradients(double targetVal)
{
    double delta = targetVal - m_outputVal;
    m_gradient = delta * transferFunctionDer(m_outputVal);

}

double Net::Neuron::sumW(const Layer &nextLayer)
{
    double sum = 0.0;

    for(size_t ni = 0; ni < nextLayer.size(); ++ni)
        sum += m_outputWeights[ni].W * nextLayer[ni].m_gradient;

    return sum;
}

void Net::Neuron::calcHiddenGradients(Layer& nextLayer)
{
    m_gradient = sumW(nextLayer) * transferFunctionDer(m_outputVal);
}

void Net::Neuron::updateInputWeights(Layer &prevLayer)
{
    for(size_t ni = 0; ni < prevLayer.size(); ++ni)
    {
        Neuron& neuron = prevLayer[ni];
        double gradW_old = neuron.m_outputWeights[m_index].gradW;
        double gradW_new = eta * neuron.getOutputVal() * m_gradient + alfa * gradW_old;
        neuron.m_outputWeights[m_index].gradW = gradW_new;
        neuron.m_outputWeights[m_index].W += gradW_new;
    }
}

double Net::Neuron::transferFunction(double x)
{
    return 1.0 * (1.0 + exp(-x));
}

double Net::Neuron::transferFunctionDer(double x)
{
    return transferFunction(x)*(1.0 - transferFunction(x));
}
