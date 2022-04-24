#include <vector>

class Net
{
    public:
    Net(const std::vector<unsigned int>& topology);
    void feedForward(const std::vector<double>& inputValues);
    void backProp(std::vector<double>& targetValues);
    void getResults(std::vector<double>& resultValues) const;

    class Neuron
    {

    };

    class Layer
    {

    };

    private:

    std::vector<Layer> m_layers;
};
