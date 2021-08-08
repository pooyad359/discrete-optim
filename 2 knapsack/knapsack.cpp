#include "knapsack.hpp"

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

template <typename T>
void showVector(std::vector<T> input)
{
    for (T x : input)
    {
        cout << x << "   ";
    }
    cout << endl;
}

template <typename T>
std::string vectorToString(std::vector<T> input)
{
    std::ostringstream oss;
    if (!input.empty())
    {
        // Convert all but the last element to avoid a trailing ","
        std::copy(input.begin(), input.end() - 1,
                  std::ostream_iterator<int>(oss, ", "));

        // Now add the last element with no delimiter
        oss << input.back();
    }
    return oss.str();
}

template <typename T>
std::vector<size_t> argsort(const std::vector<T> &v)
{
    std::vector<size_t> idx(v.size());
    iota(idx.begin(), idx.end(), 0);
    std::stable_sort(
        idx.begin(),
        idx.end(),
        [&v](size_t i1, size_t i2)
        {
            return v[i1] < v[i2];
        });
    return idx;
}
