#include "net.h"
#include <random>
#include <functional>

Net::Net(const std::vector<unsigned int> &topology)
{
    const unsigned int numLayer = topology.size();
    for(size_t li = 0; li < numLayer; ++li)
    {
        m_layers.push_back(Layer());
        bool isOutputLayer = li == numLayer - 1;
        const size_t numNeurons = isOutputLayer ? topology[li] : topology[li] + 1;
        const size_t numOutputs = isOutputLayer ? 0 : topology[li + 1] + 1;

        for(size_t ni = 0; ni < numNeurons; ++ni)
        {
            //Neuron n(numOutputs, ni, isOutputLayer ? Net::relu : Net::sigmoid, isOutputLayer ? Net::reluDer : Net::sigmoidDer);
           Neuron n(numOutputs, ni, Net::hypTan, Net::hypTanDer);
            m_layers.back().push_back(n);
        }


        m_layers.back().back().setOutputValue(1.0); //bias neuron
    }
}

void Net::feedForward(const std::vector<double> &inputValues)
{
    for(size_t ni = 0; ni < inputValues.size(); ++ni)
        m_layers[0][ni].setOutputValue(inputValues[ni]);

    for(size_t li = 1; li < m_layers.size(); ++li)
    {
        const Layer& prevLayer = m_layers[li-1];
        for(size_t ni = 0; ni < m_layers[li].size(); ++ni)
            m_layers[li][ni].feedForward(prevLayer);
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
    const Layer& lastLayer = m_layers.back();
    for(size_t ni = 0; ni < lastLayer.size(); ++ni)
        resultValues.push_back(lastLayer.at(ni).getOutputVal());
}

double Net::sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

double Net::sigmoidDer(double x)
{
    return sigmoid(x) * (1.0 - sigmoid(x));
}

double Net::hypTan(double x)
{
    return tanh(x);
}

double Net::hypTanDer(double x)
{
    return 1 - x * x;
}

double Net::relu(double x)
{
    return x > 0 ? x : 0.0;
}

double Net::reluDer(double x)
{
    return x > 0 ? 1.0 : 0.0;
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

double Net::getRandom()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> urand(0, 1);
    return urand(gen);
}

Net::Neuron::Neuron(size_t numOutputs, size_t index, std::function<double(double)> transferFunction,
                    std::function<double(double)> transferFunctionDer):
    m_index(index), m_transferFunction(transferFunction), m_transferFunctionDer(transferFunctionDer)
{
    for(size_t ni = 0; ni < numOutputs; ++ni)
    {
        Connection c;
        c.W = getRandom();
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
    m_outputVal = m_transferFunction(total);
    bool stop = true;
}

void Net::Neuron::calcOutputGradients(double targetVal)
{
    double delta = targetVal - m_outputVal;
    m_gradient = delta * m_transferFunctionDer(m_outputVal);
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
    m_gradient = sumW(nextLayer) * m_transferFunctionDer(m_outputVal);
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
