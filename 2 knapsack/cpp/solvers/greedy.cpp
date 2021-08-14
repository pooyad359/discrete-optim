#include "greedy.hpp"

std::vector<bool> greedySolver(Knapsack knapsack)
{
    int spareWeight = knapsack.capacity;
    std::vector<bool> solution(knapsack.n_items);
    size_t item;
    for (int i = 0; i < knapsack.n_items; i++)
    {
        item = knapsack.indexSortedDensities[i];
        if (knapsack.weights[item] > spareWeight)
        {
            solution[item] = 0;
        }
        else
        {
            solution[item] = 1;
            spareWeight -= knapsack.weights[item];
        }
    }
    return solution;
}