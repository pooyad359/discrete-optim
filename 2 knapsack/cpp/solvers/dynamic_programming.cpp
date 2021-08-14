#include "dynamic_programming.hpp"

std::vector<bool> dynamicProg(Knapsack knapsack)
{
    int rowCount = knapsack.capacity;
    int colCount = knapsack.n_items;
    int itemCount = knapsack.n_items;
    int idx, wi, vi;
    std::vector<bool> solution(itemCount, 0);
    std::vector<std::vector<int>> mat(rowCount, std::vector<int>(colCount, 0));
    for (int row = 0; row < rowCount; row++)
    {
        for (int col = 1; col < colCount; col++)
        {
            idx = col - 1;
            wi = knapsack.weights[idx];
            vi = knapsack.values[idx];
            if (row < wi)
            {
                mat[row][col] = mat[row][col - 1];
            }
            else
            {
                mat[row][col] = std::max(mat[row][col - 1], vi + mat[row - wi][col - 1]);
            }
        }
    }
    int row = rowCount - 1;
    for (int col = itemCount; col > 0; col--)
    {
        if (mat[row][col] != mat[row][col - 1])
        {
            solution[col - 1] = 1;
            row = row - knapsack.weights[col - 1];
        }
    }
    return solution;
}