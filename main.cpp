#include "net.h"
#include <iostream>

int main()
{
    //std::cout << "Hello, world!" << std::endl;
    std::vector<unsigned int> topology;
    std::vector<double> inputValues, targetValues, resultValues;

    Net net(topology);

    return 0;
    //g++ -std=c++20 -Wshadow -Wall -o main main.cpp
}
