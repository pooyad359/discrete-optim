#include "solver.hpp"
#include ".\utils\utils.cpp"
int main(int argc, char **argv)
{
    long n_items, capacity, temp;
    std::string filename = argv[1];
    std::vector<int> weights, values;
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
    cout << "File Name: " << filename << endl;
    cout << "Items Count: " << n_items << endl;
    cout << "Max Capacity: " << capacity << endl;
    cout << "Items values: " << vectorToString(values) << endl;
    cout << "Items weights: " << vectorToString(weights) << endl;
    cout << vectorToString(argsort(values)) << endl;
    cout << "Items values: " << vectorToString(values) << endl;
    file.close();
    return 0;
}
