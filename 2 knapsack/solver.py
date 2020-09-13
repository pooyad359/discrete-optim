#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
import numpy as np
from numba import jit
Item = namedtuple("Item", ['index', 'value', 'weight'])

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split("\n")
    data = [line.split() for line in lines if len(line) > 0]
    n, cap = data.pop(0)
    n, cap = int(n), float(cap)
    data = np.array(data, dtype=float)
    values = data[:,0]
    weights = data[:,1]

    # Optimisation
    if n>400:
        print('### Greedy Algo ###')
        selection, weight, count, spare = greedy_dens(values, weights, cap)

    # elif n>:
    #     selection, weight, count,spare = DynP(values, weights, cap)
    else:
        print('### Dynamic Programming ###')
        selection, weight, count, spare = DynPjit(values, weights, cap)
    value = np.sum(selection * values)
    
    # prepare the solution in the specified output format
    output_data = str(value.astype(int)) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, selection.astype(int)))
    return output_data

def greedy_dens(values, weights, capacity):
    n = len(values)
    densities = values / weights
    idx = np.argsort(densities)
    ks_weight = 0
    selection = np.zeros(n)
    for i in reversed(idx):
        selection[i] = 1
        if np.sum(selection * weights) > capacity:
            selection[i] = 0
            continue
    final_weight = np.sum(selection * weights)
    spare = capacity - final_weight
    return (
        selection.astype(int),
        np.sum(selection * weights).astype(int),
        np.sum(selection).astype(int),
        spare,
    )

def DynP(values, weights, capacity):
    rows, cols = int(capacity) + 1, len(values) + 1
    mat = np.zeros((rows, cols))
    for row in range(rows):
        for col in range(1, cols):

            i = col - 1
            wi = int(weights[i])
            vi = int(values[i])
            if row < wi:
                mat[row, col] = mat[row, col - 1]
            else:
                mat[row, col] = max(mat[row, col - 1], vi + mat[row - wi, col - 1])
    selection = np.zeros(len(values), dtype=int)
    row = rows - 1
    for c in range(len(values), 0, -1):
        if mat[row, c] != mat[row, c - 1]:
            selection[c - 1] = 1
            row = row - int(weights[c - 1])

    #     return mat[-1, -1], selection
    final_weight = np.sum(selection * weights)
    final_value = np.sum(selection * values)
    if abs(final_value - mat[-1, -1]) > 1e-3:
        print(f"***Warning: {np.sum(selection * values)} =/= {mat[-1,-1]}")
    spare = capacity - final_weight
    return (
        selection.astype(int),
        np.sum(selection * weights).astype(int),
        np.sum(selection).astype(int),
        spare,
    )

@jit
def DynPjit(values, weights, capacity):
    rows, cols = int(capacity) + 1, len(values) + 1
    mat = np.zeros((rows, cols))
    for row in range(rows):
        for col in range(1, cols):

            i = col - 1
            wi = int(weights[i])
            vi = int(values[i])
            if row < wi:
                mat[row, col] = mat[row, col - 1]
            else:
                mat[row, col] = max(mat[row, col - 1], vi + mat[row - wi, col - 1])
    selection = np.zeros(len(values), dtype=int)
    row = rows - 1
    for c in range(len(values), 0, -1):
        if mat[row, c] != mat[row, c - 1]:
            selection[c - 1] = 1
            row = row - int(weights[c - 1])

    #     return mat[-1, -1], selection
    final_weight = np.sum(selection * weights)
    final_value = np.sum(selection * values)
    # if abs(final_value - mat[-1, -1]) > 1e-3:
    #     print(f"***Warning: {np.sum(selection * values)} =/= {mat[-1,-1]}")
    spare = capacity - final_weight
    return (
        selection.astype(int),
        np.sum(selection * weights).astype(int),
        np.sum(selection).astype(int),
        spare,
    )

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

