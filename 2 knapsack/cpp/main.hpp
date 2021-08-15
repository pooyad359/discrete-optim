#ifndef _SOLVER
#define _SOLVER
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <sstream>
#include <iterator>
#include <algorithm>
#include <chrono>
#include "./utils/utils.cpp"
#include "knapsack.cpp"
#include "solvers/greedy.cpp"
#include "solvers/dynamic_programming.hpp"

using namespace std::chrono;
using std::cout;
using std::endl;
template <typename T>
void showVector(std::vector<T>);

template <typename T>
std::string vectorToString(std::vector<T>);

template <typename T>
std::vector<size_t> argsort(const std::vector<T> &v);
#endif