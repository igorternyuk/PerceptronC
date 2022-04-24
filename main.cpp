#include "NN.h"
#include <iostream>
#include <cmath>

int main()
{
    //std::cout << "Hello, world!" << std::endl;
    std::vector<unsigned int> topology {};
    std::vector<double> inputValues, targetValues, resultValues;

    Net net(topology);
    int a = 3;
    int b = 4;
    double c = sqrt(a*a+b*b);
    return 0;
    //g++ -std=c++20 -Wshadow -Wall -o main main.cpp
}
