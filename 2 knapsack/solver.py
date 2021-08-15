#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
import numpy as np
from numba import jit
from tqdm.auto import tqdm
import uuid
import os
import time

Item = namedtuple("Item", ["index", "value", "weight"])
N_TRIALS = 100


def cpp_solver(input_data):
    print("*** USING C++ SOLVER ***")
    input_name = "." + uuid.uuid4().hex
    output_name = "." + uuid.uuid4().hex
    with open(input_name, "w") as fp:
        fp.write(input_data)
    output = os.popen(f".\cpp\main.exe {input_name} {output_name}")
    print(output.read())
    with open(output_name, "r") as fp:
        submission = fp.read()
    os.remove(input_name)
    os.remove(output_name)
    return submission


def solve_it(input_data, use_cpp=False):
    if use_cpp:
        return cpp_solver(input_data)

    print("*** USING PYTHON SOLVER ***")
    # parse the input
    lines = input_data.split("\n")
    data = [line.split() for line in lines if len(line) > 0]
    n, cap = data.pop(0)
    n, cap = int(n), float(cap)
    data = np.array(data, dtype=float)
    values = data[:, 0]
    weights = data[:, 1]

    # Optimisation
    if n > 1000:
        # print("### Greedy Algo ###")
        # selection, weight, count, spare = greedy_dens(values, weights, cap)
        print("### Filtering ###")
        selection, weight, count, spare = filtering(values, weights, cap)
        for i in tqdm(range(N_TRIALS)):
            selection, weight, count, spare = local_search(
                values, weights, cap, selection
            )
            # solution = output[0]
    # elif n>:
    #     selection, weight, count,spare = DynP(values, weights, cap)
    else:
        print("### Dynamic Programming ###")
        selection, weight, count, spare = DynPjit(values, weights, cap)
    value = np.sum(selection * values)

    # prepare the solution in the specified output format
    output_data = str(value.astype(int)) + " " + str(0) + "\n"
    output_data += " ".join(map(str, selection.astype(int)))
    return output_data


def filtering(values, weights, cap, rho_min=None):
    n = len(values)
    rho = values / weights
    if rho_min is None:
        rho_min = np.quantile(rho, 1 - 500 / n)
    idi = np.arange(n)
    idf = idi[rho > rho_min]
    print("Number of Filtered items:", len(idf))

    output = DynPjit(values[idf], weights[idf], cap)
    sol = output[0]
    idx = idf[sol == 1]
    solution = np.zeros(n)
    solution[idx] = 1
    print(solution.dot(values))
    idx = np.argsort(rho)
    for i in reversed(idx):
        if solution[i]:
            continue
        solution[i] = 1
        if solution.dot(weights) > cap:
            solution[i] = 0
    return (
        solution,
        solution.dot(weights),
        solution.sum(),
        cap - solution.dot(weights),
    )


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


def local_search(values, weights, capacity, solution):
    n = len(values)
    init_value = np.dot(values, solution)
    densities = values / weights
    idx = np.argsort(densities)
    idx_ones = np.nonzero(solution)[0]
    switched = np.random.choice(idx_ones)
    selection = solution.copy()
    selection[switched] = 0
    # print(switched, idx_ones)
    for i in reversed(idx):
        if i == switched or selection[i]:
            continue
        selection[i] = 1
        if np.dot(selection, weights) > capacity:
            selection[i] = 0

    final_value = np.dot(selection, values)
    if final_value < init_value:
        selection = solution.copy()
        final_value = init_value
    final_weight = np.dot(selection, weights)
    spare = capacity - final_weight
    return (
        selection,
        final_weight,
        selection.sum(),
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


@jit(
    parallel=False,
)
def DynPjit(values, weights, capacity):
    weights = np.int32(weights)
    values = np.int32(values)
    capacity = np.int32(capacity)
    nitems = np.shape(values)[0]
    rows = np.int32(capacity) + 1
    cols = np.int32(nitems + 1)
    mat = np.zeros((rows, cols))
    for row in np.arange(rows, dtype=np.int32):
        for col in np.arange(1, cols, dtype=np.int32):

            i = np.int32(col - 1)
            wi = np.int32(weights[i])
            vi = np.int32(values[i])
            if row < wi:
                mat[row, col] = mat[row, col - 1]
            else:
                mat[row, col] = max(mat[row, col - 1], vi + mat[row - wi, col - 1])
    selection = np.zeros(nitems, dtype=np.int32)
    row = rows - 1
    for c in range(nitems, 0, -1):
        if mat[row, c] != mat[row, c - 1]:
            selection[c - 1] = 1
            row = row - np.int32(weights[c - 1])

    #     return mat[-1, -1], selection
    final_weight = np.sum(selection * weights)
    final_value = np.sum(selection * values)
    # if abs(final_value - mat[-1, -1]) > 1e-3:
    #     print(f"***Warning: {np.sum(selection * values)} =/= {mat[-1,-1]}")
    spare = capacity - final_weight
    return (
        selection.astype(np.int8),
        np.dot(selection, weights),
        np.sum(selection, dtype=np.int32),
        spare,
    )


if __name__ == "__main__":
    import sys

    start = time.perf_counter_ns()
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        use_cpp = len(sys.argv) > 2
        with open(file_location, "r") as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data, use_cpp))
    else:
        print(
            "This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)"
        )
    end = time.perf_counter_ns()
    print(f"Elapsed Time (ms): {(end - start) * 1e-6:.2f}")
