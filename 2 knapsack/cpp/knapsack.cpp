#include "knapsack.hpp"
Knapsack::Knapsack(std::vector<int> w, std::vector<int> v, int cap)
{
    std::copy(v.begin(), v.end(), std::back_inserter(Knapsack::values));
    std::copy(w.begin(), w.end(), std::back_inserter(Knapsack::weights));
    Knapsack::capacity = cap;
    Knapsack::n_items = values.size();
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
// };
