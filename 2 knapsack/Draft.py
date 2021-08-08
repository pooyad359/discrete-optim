# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np
import os
from numba import jit

# %load_ext nb_black

path = './data'
files = os.listdir(path)


for i, f in enumerate(files):
    print(i, f)


def parse_data(data):
    lines = data.split("\n")
    n, cap = lines[0].split()
    data = [line.split() for line in lines if len(line) > 0]
    n, cap = data.pop(0)
    return np.int(n), np.float(cap), np.array(data, dtype=float)


# +
file = files[14]
with open(os.path.join(path, file)) as fp:
    data = fp.read()

n, K, data = parse_data(data)
values = data[:, 0]
weights = data[:, 1]
# -



# +
def check_results(selection, weights, values, cap):
    assert (
        selection * weights
    ).sum() <= cap, f"Too heavy!: The capacity is {cap} but the total weight is {(selection * weights).sum()}."


def test_method(func, file):
    with open(os.path.join(path, file)) as fp:
        data = fp.read()
    n, K, data = parse_data(data)
    values = data[:, 0]
    weights = data[:, 1]
    print(f"Total number: {n} \nCapacity: {K}")
    selection, weight, count, spare = func(values, weights, K)
    final_value = np.dot(selection, values)
    print(
        f"Final Weight: {weight} \nFinal Value: {final_value}\nNumber of Items: {count}\nSpare capacity: {spare}"
    )


# +
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
            break
    final_weight = np.sum(selection * weights)
    spare = capacity - final_weight
    return (
        selection.astype(int),
        np.sum(selection * weights).astype(int),
        np.sum(selection).astype(int),
        spare,
    )


def greedy_dens2(values, weights, capacity):
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


@jit
def DynPj(values, weights, capacity):
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
    #     if abs(final_value - mat[-1, -1]) > 1e-3:
    #         print(f"***Warning: {np.sum(selection * values)} =/= {mat[-1,-1]}")
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
    #     if abs(final_value - mat[-1, -1]) > 1e-3:
    #         print(f"***Warning: {np.sum(selection * values)} =/= {mat[-1,-1]}")
    spare = capacity - final_weight
    return (
        selection.astype(int),
        np.sum(selection * weights).astype(int),
        np.sum(selection).astype(int),
        spare,
    )


from ortools.algorithms import pywrapknapsack_solver


def ortoolopt(weights, values, cap):
    weights = [list(weights)]
    values = list(values)
    solver = pywrapknapsack_solver.KnapsackSolver(
        pywrapknapsack_solver.KnapsackSolver.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER,
        "KnapsackExample",
    )
    solver.Init(values, weights, [cap])
    computed_value = solver.Solve()
    packed_items = []
    packed_weights = []
    total_weight = 0
#     print("Total value =", computed_value)
    for i in range(len(values)):
        if solver.BestSolutionContains(i):
            packed_items.append(i)
            packed_weights.append(weights[0][i])
            total_weight += weights[0][i]
    #     print("Total weight:", total_weight)
    #     print("Packed items:", packed_items)
    #     print("Packed_weights:", packed_weights)
    selection = np.zeros(len(values))
    selection[packed_items] = 1
    return (
        selection.astype(int),
        total_weight,
        len(values),
        cap - total_weight,
    )

def LinearRelaxation(values, weights, capacity):
    n = len(values)
    densities = values / weights
    idx = np.argsort(densities)
    ks_weight = 0
    selection = np.zeros(n)
    for i in reversed(idx):
        selection[i] = 1
        if np.sum(selection * weights) > capacity:
            selection[i] = 0
            final_weight = (selection * weights).sum()
            spare = capacity - final_weight
            xi = spare / weights[i]
            final_value = (selection * values).sum() + xi * values[i]
            return final_value



# +
file = files[0]
with open(os.path.join(path, file)) as fp:
    data = fp.read()

n, K, data = parse_data(data)
values = data[:, 0]
weights = data[:, 1]
selection, weight, count, spare = ortoolopt(values, weights, K)
print(np.dot(selection, values))
print(selection)
print(weight, K)

# +
file = files[0]
with open(os.path.join(path, file)) as fp:
    data = fp.read()

n, K, data = parse_data(data)
values = data[:, 0]
weights = data[:, 1]
selection, weight, count, spare = DynPj(values, weights, K)
print(np.dot(selection, values))
print(selection)
# -

for i, f in enumerate(files):
    print(i, f)

# %%time 
test_method(ortoolopt,files[5])

# %%time
test_method(DynPj, files[7])

# %%time
test_method(DynP, files[7])

# %%time
test_method(greedy_dens2, files[11])




ortoolopt(weights, values, K)

list(range(len(values), 0, -1))

for i, f in enumerate(files):
    print(i, f)


class Node():
    def __init__(self,weight,value,level):
        self.w = weight
        self.v = value
        self.l = level
    def children(self):
        level = self.l + 1
        self.left = Node(weight[self])


class Tree:
    def __init__(self, weights, values, capacity):
        densities = values / weights
        idx = np.argsort(densities)
        self.weights = weights[idx]
        self.values = values[idx]
        self.cap = capacity
        self.active_level = 0
        self.count = len(weights)
        self.selection = np.ones(self.count) * -1
        self.best_selection = None
        self.best_value = 0
    @property
    def get_lr(self):
        weight = self.get_weight()
        return LinearRelaxation(self.values[l:], self.weights[l:], self.cap - weight)

    @property
    def get_value(self):
        idx = self.selection > -1
        return (self.values[idx] * self.selection[idx]).sum()

    @property
    def get_weight(self):
        idx = self.selection > -1
        return (self.weights[idx] * self.selection[idx]).sum()

    @property
    def get_room(self):
        return self.cap - self.get_weight

    @property
    def is_end(self):
        return self.selection[-1] == -1

    @property
    def get_next_level(self):
        return np.argmin(self.selection)

    def step(self):
        estim = self.get_lr
        room = self.cap
        value = self.get_value
        for s in [1,0]:
            i = self.get_next_level
            self.selection[i]=s
            self.
