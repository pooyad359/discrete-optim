#include "knapsack.hpp"

Knapsack::Knapsack(std::vector<int> w, std::vector<int> v, int cap)
{
    std::copy(v.begin(), v.end(), std::back_inserter(Knapsack::values));
    std::copy(w.begin(), w.end(), std::back_inserter(Knapsack::weights));
    Knapsack::capacity = cap;
    Knapsack::n_items = values.size();
    Knapsack::densities = Knapsack::getDensities();
}

std::string Knapsack::toString()
{
    std::stringstream output;
    output << "Items Count: " << Knapsack::n_items << endl;
    output << "Max Capacity: " << Knapsack::capacity << endl;
    output << "Items values: " << vectorToString(Knapsack::values) << endl;
    output << "Items weights: " << vectorToString(Knapsack::weights) << endl;
    return output.str();
}

std::vector<double> Knapsack::getDensities()
{
    std::vector<double> densities;
    int length;
    length = Knapsack::values.size();
    for (int i = 0; i < length; i++)
    {
        densities.push_back(Knapsack::values[i] * 1.0 / Knapsack::weights[i]);
    }
    return densities;
}

void Knapsack::argSortDensities()
{
    Knapsack::indexSortedDensities = argsort(Knapsack::densities);
}

void Knapsack::showItemsByDensity()
{
    Knapsack::argSortDensities();
    for (int idx : Knapsack::indexSortedDensities)
    {
        cout << Knapsack::values[idx] << "\t";
        cout << Knapsack::weights[idx] << "\t";
        cout << Knapsack::densities[idx] << endl;
    }
}

std::vector<bool> Knapsack::randomSolution()
{
    srand(time(0));
    std::vector<bool> solution;
    // cout << Knapsack::n_items << endl;
    for (int i = 0; i < Knapsack::n_items; i++)
    {
        solution.push_back(rand() % 2 == 0 ? 1 : 0);
    }
    return solution;
}

int Knapsack::totalWeight(std::vector<bool> solution)
{
    int totalWeight = 0;
    for (int i; i < Knapsack::n_items; i++)
    {
        totalWeight += Knapsack::weights[i] * solution[i];
    }
    return totalWeight;
}
int Knapsack::totalValue(std::vector<bool> solution)
{
    int totalValue = 0;
    for (int i; i < Knapsack::n_items; i++)
    {
        totalValue += Knapsack::values[i] * solution[i];
    }
    return totalValue;
}

// Returns true if the capacity is not exeeded
bool Knapsack::checkSolution(std::vector<bool> solution)
{
    return Knapsack::totalWeight(solution) < Knapsack::capacity;
}
