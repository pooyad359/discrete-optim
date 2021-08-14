#include "main.hpp"

int main(int argc, char **argv)
{
    long n_items, capacity, temp;
    std::string filename = argv[1];
    std::vector<int> weights, values;
    std::vector<bool> solution;
    std::string line;
    std::ifstream file;
    file.open(filename);
    if (file.is_open())
    {
        // Read First Line
        std::getline(file, line);
        std::stringstream lineStream(line);
        lineStream >> n_items;
        lineStream >> capacity;

        //Read the rest
        while (std::getline(file, line))
        {
            std::stringstream lineStream(line);
            lineStream >> temp;
            values.push_back(temp);
            lineStream >> temp;
            weights.push_back(temp);
        }
    }
    Knapsack ks(weights, values, capacity);
    cout << ks.toString() << endl;
    file.close();
    ks.argSortDensities();
    // ks.showItemsByDensity();
    solution = greedySolver(ks);
    solution = dynamicProg(ks);
    showVector(solution);
    cout << ks.totalWeight(solution) << endl;
    cout << ks.totalValue(solution) << endl;
    cout << (ks.checkSolution(solution) ? "Solution is Valid" : "Solution is invalid!") << endl;
    return 0;
}
