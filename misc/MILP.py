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

# # Set up

# %load_ext nb_black

from ortools.linear_solver import pywraplp
import numpy as np
import os


# +
def main():
    # Create the mip solver with the SCIP backend.
    solver = pywraplp.Solver.CreateSolver('SCIP','SCIP')

    infinity = solver.infinity()
    # x and y are integer non-negative variables.
    x = solver.IntVar(0, infinity, 'x')
    y = solver.IntVar(0, infinity, 'y')

    print('Number of variables =', solver.NumVariables())

    # x + 7 * y <= 17.5.
    solver.Add(x + 7 * y <= 17.5)

    # x <= 3.5.
    solver.Add(x <= 3.5)

    print('Number of constraints =', solver.NumConstraints())

    # Maximize x + 10 * y.
    solver.Maximize(x + 10 * y)

    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        print('Solution:')
        print('Objective value =', solver.Objective().Value())
        print('x =', x.solution_value())
        print('y =', y.solution_value())
    else:
        print('The problem does not have an optimal solution.')

    print('\nAdvanced usage:')
    print('Problem solved in %f milliseconds' % solver.wall_time())
    print('Problem solved in %d iterations' % solver.iterations())
    print('Problem solved in %d branch-and-bound nodes' % solver.nodes())



# -

main()

# # Knapsack

path = '../2 knapsack/data'
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
file = files[0]
with open(os.path.join(path, file)) as fp:
    data = fp.read()

n, K, data = parse_data(data)
values = data[:, 0]
weights = data[:, 1]


# -

def ks_mip(weights, values, capacity):
    assert len(weights) == len(values), "Weights and Values must be of the same size."

    # Create solver
    solver = pywraplp.Solver.CreateSolver("SCIP", "SCIP")

    # Define Variables
    n = len(weights)
    choices = [solver.BoolVar(f"x_{i}") for i in range(n)]

    # Constraints
    total_weight = sum([weights[i] * choices[i] for i in range(n)])
    solver.Add(total_weight <= capacity)

    # Objective
    total_value = sum([values[i] * choices[i] for i in range(n)])
    solver.Maximize(total_value)

    # Solve
    status = solver.Solve()

    # Solution
    if status == pywraplp.Solver.OPTIMAL:
        solution = [choices[i].solution_value() for i in range(n)]
        total_weight = sum([weights[i] * solution[i] for i in range(n)])
        print("Solution:")
        print(f"Objective value = {solver.Objective().Value()}")
        print(f"Weight = {total_weight}/{capacity}")
        print(solution)
    #         print('x =', x.solution_value())
    #         print('y =', y.solution_value())
    else:
        print("The problem does not have an optimal solution.")

    print("\nAdvanced usage:")
    print("Problem solved in %f milliseconds" % solver.wall_time())
    print("Problem solved in %d iterations" % solver.iterations())
    print("Problem solved in %d branch-and-bound nodes" % solver.nodes())


ks_mip(weights, values, capacity=K)

# # Graph Coloring

path = "../3 coloring/data"
files = os.listdir(path)
[f"{i}. {f}" for i, f in enumerate(files)]

idx = 10
print(files[idx])
with open(os.path.join(path, files[idx]), "r") as fp:
    input_data = fp.read()
print(input_data[:15])

# +
lines = input_data.split("\n")

first_line = lines[0].split()
node_count = int(first_line[0])
edge_count = int(first_line[1])

edges = []
for i in range(1, edge_count + 1):
    line = lines[i]
    parts = line.split()
    edges.append((int(parts[0]), int(parts[1])))

print(node_count, edge_count)
# -

edge_array = np.array(edges)
orders = [np.sum(edge_array == i) for i in range(node_count)]
highest_order = int(np.argmax(orders))
print(highest_order, orders[highest_order])

max_color = highest_order + 1

# +
import networkx as nx
import matplotlib.pyplot as plt


def create_graph(n_points, edges):
    graph = nx.Graph()
    graph.add_nodes_from(range(n_points))
    graph.add_edges_from(edges)
    return graph


def get_max_clique(n_points, edges):
    graph = create_graph(n_points, edges)
    cliques = nx.find_cliques(graph)
    return max([len(c) for c in cliques])


def gc_mip(n_points, edges, max_color):
    n_colors = max_color
    min_colors = get_max_clique(n_points, edges)
    # Create solver
    solver = pywraplp.Solver.CreateSolver("SCIP", "SCIP")

    # Define Variables
    colors = [
        [solver.BoolVar(f"x_{i}_{c}") for c in range(n_colors)] for i in range(n_points)
    ]

    # Constraints
    # Connected edges
    for i, j in edges:
        for c in range(n_colors):
            solver.Add(colors[i][c] + colors[j][c] <= 1)

    # Only one color
    for i in range(n_points):
        solver.Add(sum(colors[i]) <= 1)
        solver.Add(sum(colors[i]) >= 1)

        #     total_weight = sum([weights[i] * choices[i] for i in range(n)])
        #     solver.Add(total_weight <= capacity)

        # Break Symmetry
#         for c in range(n_colors - 1):
#             sum1 = sum([colors[i][c] for i in range(n_points)])
#             sum2 = sum([colors[i][c + 1] for i in range(n_points)])
#             solver.Add(sum1 >= sum2)

#             # Min colors
#             if c < min_colors:
#                 solver.Add(sum1 >= 1)

    # Objective
    obj = 0
    for i in range(n_points):
        obj += sum([colors[i][c] * c for c in range(n_colors)])
    solver.Minimize(obj)

    # Solve
    status = solver.Solve()

    # Solution
    if status == pywraplp.Solver.OPTIMAL:
        #         solution = [choices[i].solution_value() for i in range(n)]
        solution = [
            [colors[i][c].solution_value() for c in range(n_colors)]
            for i in range(n_points)
        ]
        for i in range(n_points):
            print(solution[i])
        print("Solution:")
        print(f"Objective value = {solver.Objective().Value()}")

    #         print(solution)
    #         print('x =', x.solution_value())
    #         print('y =', y.solution_value())
    else:
        print("The problem does not have an optimal solution.")

    print("\nAdvanced usage:")
    print("Problem solved in %f milliseconds" % solver.wall_time())
    print("Problem solved in %d iterations" % solver.iterations())
    print("Problem solved in %d branch-and-bound nodes" % solver.nodes())


# -

gc_mip(node_count, edges, max_color//2)

# +
# gc_mip(node_count, edges, max_color // 2)
# -


