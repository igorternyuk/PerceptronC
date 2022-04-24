#include "net.h"
#include <iostream>
#include <cmath>

int main()
{
    std::vector<unsigned int> topology {1, 5, 3, 5, 2, 1};
    std::vector<double> inputValues, targetValues;

    Net net(topology);

    const int N1 = 30000;
    const double angMin = -M_PI;
    const double angMax = M_PI;
    for(size_t pi = 1; pi <= N1; ++pi)
    {
        double t = Net::getRandom(); //pi - 1.0) / (N - 1.0);
        double ang = angMin * (1.0 - t) + angMax * t;
        inputValues.push_back(ang);
        targetValues.push_back(cos(ang));
    }

    std::cout << "\n----------------------------- TRAINING --------------------------------\n" << std::endl;

    for(size_t pi = 0; pi < inputValues.size(); ++pi)
    {
        std::vector<double> input;
        input.push_back(inputValues[pi]);
        std::vector<double> target;
        target.push_back(targetValues[pi]);
        net.feedForward(input);
        net.backProp(target);
        std::vector<double> res;
        net.getResults(res);
        double eps = fabs(target[0]) < 1e-12 ? res[0] : fabs((res[0] - target[0])/target[0]);
        eps = 100.0*eps;
        std::cout << "i = " << (pi + 1) << " x = " << input[0] << " Ynn = " << res[0] << " Ytarget = " << target[0] << " eps = " << eps << "% error = " << net.getError() << std::endl;
    }

    std::cout << "\n----------------------------- VALIDATION --------------------------------\n" << std::endl;

    const int N2 = 500;
    for(size_t pi = 1; pi <= N2; ++pi)
    {
        double t = (pi - 1.0) / (N2 - 1.0);
        double ang = angMin * (1.0 - t) + angMax * t;
        std::vector<double> input;
        input.push_back(ang);
        std::vector<double> target;
        target.push_back(cos(ang));
        net.feedForward(input);
        net.backProp(target);
        std::vector<double> res;
        net.getResults(res);
        double eps = fabs(target[0]) < 1e-12 ? res[0] : fabs((res[0] - target[0])/target[0]);
        eps = 100.0*eps;
        std::cout << "i = " << (pi + 1) << " x = " << input[0] << " Ynn = " << res[0] << " Ytarget = " << target[0] << " eps = " << eps << "% error = " << net.getError() << std::endl;
    }

    return 0;
}
