#include "main.hpp"

int main(int argc, char **argv)
{
    long n_items, capacity, temp;
    std::string filename = argv[1];
    std::string output = argv[2];
    std::vector<int> weights, values;
    std::vector<bool> solution;
    std::string line;
    std::ifstream fileIn;
    std::ofstream fileOut;
    fileIn.open(filename);
    cout << "\t" << argv[1] << "\t" << argv[2] << endl;
    if (fileIn.is_open())
    {
        // Read First Line
        std::getline(fileIn, line);
        std::stringstream lineStream(line);
        lineStream >> n_items;
        lineStream >> capacity;

        //Read the rest
        while (std::getline(fileIn, line))
        {
            std::stringstream lineStream(line);
            lineStream >> temp;
            values.push_back(temp);
            lineStream >> temp;
            weights.push_back(temp);
        }
    }
    cout << "Input:" << endl;
    auto start = high_resolution_clock::now();
    Knapsack ks(weights, values, capacity);
    cout << ks.toString() << endl;
    fileIn.close();
    // ks.argSortDensities();
    solution = dynamicProg(ks);
    auto stop = high_resolution_clock::now();
    cout << "Total Weight: " << ks.totalWeight(solution) << endl;
    cout << "Total Value: " << ks.totalValue(solution) << endl;
    cout << "Solution:" << endl;
    showVector(solution);
    cout << (ks.checkSolution(solution) ? "Solution is Valid" : "Solution is invalid!") << endl;
    cout << endl;
    // Write solution to file
    fileOut.open(output);
    fileOut << ks.totalValue(solution) << " " << 1 << endl;
    fileOut << vectorToString(solution);
    fileOut.close();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "C++ Elapsed Time: " << duration.count() / 1000.0 << endl;

    return 0;
}
