#ifndef _KNAPSACK
#define _KNAPSACK
#include <vector>
#include <algorithm>
#include <iterator>
#include <string>
#include <sstream>
#include <iostream>
#include "./utils/utils.hpp"
#include <stdlib.h>
#include <random>
#include <ctime>
using std::cout;
using std::endl;

class Knapsack
{
public:
    int capacity;
    int n_items;
    std::vector<int> weights;
    std::vector<int> values;
    std::vector<double> densities;
    std::vector<size_t> indexSortedDensities;
    Knapsack(std::vector<int>, std::vector<int>, int);
    std::vector<double> getDensities();
    void argSortDensities();
    std::string toString();
    void showItemsByDensity();
    std::vector<size_t> solveGreedy();
    bool checkSolution(std::vector<bool>);
    int totalWeight(std::vector<bool>);
    int totalValue(std::vector<bool>);
    std::vector<bool> randomSolution();
};

#endif