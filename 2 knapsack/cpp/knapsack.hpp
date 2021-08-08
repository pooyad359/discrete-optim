#ifndef _KNAPSACK
#define _KNAPSACK
#include <vector>
#include <algorithm>
#include <iterator>
#include <string>
#include <sstream>
#include <iostream>
#include "./utils/utils.hpp"

using std::cout;
using std::endl;

class Knapsack
{
public:
    int capacity;
    int n_items;
    std::vector<int> weights;
    std::vector<int> values;
    Knapsack(std::vector<int>, std::vector<int>, int);
    std::string toString();
    // void Knapsack::print();
};

#endif