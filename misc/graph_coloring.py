#!/usr/bin/python
# -*- coding: utf-8 -*-

from ortools.constraint_solver import pywrapcp
from ortools.sat.python import cp_model
import numpy as np
from tqdm.auto import tqdm, trange
from copy import deepcopy
from collections import namedtuple, Counter
from pdb import set_trace

N_POP = 100
N_ROUNDS = 1000

GreedyGraph = namedtuple("GreedyGraph", ["colors", "order", "count"])


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split("\n")

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    edges = []
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append((int(parts[0]), int(parts[1])))
    print("\n\n----- Problem Description -----")
    print("Number of Nodes", node_count)
    print("Number of Edges:", edge_count)
    print(f"Average node degree {2*edge_count/node_count:.1f}")
    # build a trivial solution
    # every node has its own color
    # solution = range(0, node_count)

    # res = fixed_color(edges,node_count,n_colors)
    MODE = 4
    if MODE == 0:
        print("<<< USING STEP-WISE CP >>>")
        n_colors, res = stepwise_opt(edges, node_count)
    elif MODE == 1:
        print("<<< USING CP SOLVER >>>")
        res = cp_solver_test(edges, node_count)
        n_colors = np.max(res) + 1
    elif MODE == 2:
        print("<<< USING GA + GREEDY >>>")
        N_ROUNDS = 1000
        if node_count >= 200:
            N_ROUNDS = 200
        if node_count >= 500:
            N_ROUNDS = 50
        if node_count >= 1000:
            N_ROUNDS = 10
        res = ga_solver(edges, node_count, 1, N_ROUNDS)
        n_colors = np.max(res) + 1
    elif MODE == 3:
        print("<<< USING GREEDY + REORDERING >>>")
        res = reordered_greedy(edges, node_count, 1, 5000)
        n_colors = count_colors(res)

    elif MODE == 4:
        print("<<< DOUBLE TRIAL >>>")
        res1 = reordered_greedy(edges, node_count, 1, 5000)
        n_colors_1 = count_colors(res1)
        n_colors_2, res2 = stepwise_opt(edges, node_count, n_colors_1)
        if res2 is None:
            res = res1
            n_colors = n_colors_1
        elif n_colors_1 < n_colors_2:
            res = res1
            n_colors = n_colors_1
        else:
            res = res2
            n_colors = n_colors_2
    else:
        print("<<< USING GREEDY >>>")
        res = greedy_solver(edges, node_count)
        n_colors = np.max(res) + 1
    # res = all_res[-1]
    # set_trace()
    print("\n\n---- Solution ----")
    print("Number of colors:", n_colors)
    print(*sorted(Counter(res).items()), sep="\n")
    # algo2(edges,node_count,nc)

    # prepare the solution in the specified output format
    output_data = str(n_colors) + " " + str(1) + "\n"
    output_data += " ".join(map(str, res))
    # output_data = None
    return output_data


def algo(edges, n_points):
    # solver = pywrapcp.Solver('Colors')
    solver = pywrapcp.Solver("Colors")
    colors = [solver.IntVar(0, n_points - 1, f"c_{i}") for i in range(n_points)]
    for i, j in edges:
        solver.Add(colors[i] != colors[j])
    solver.Minimize(solver.Max(colors), 1)
    db = solver.Phase(colors, solver.CHOOSE_FIRST_UNBOUND, solver.ASSIGN_MIN_VALUE)
    solver.NewSearch(db)

    if solver.NextSolution():
        return [c.Value() for c in colors]
    else:
        return []


def choose_color(colors):
    color = 0
    while color in colors:
        color += 1
    return color


def get_degrees(edges, node_count):
    edge_array = np.array(edges)
    degrees = [np.sum(edge_array == i) for i in range(node_count)]
    return degrees


def greedy_solver(edges, n_points):
    degrees = get_degrees(edges, n_points)
    edge_dict = {i: set() for i in range(n_points)}
    for v1, v2 in edges:
        edge_dict[v1].add(v2)
        edge_dict[v2].add(v1)
    # get indices by degree of the vertex
    idx = np.argsort(degrees)[::-1]
    colors = -np.ones(n_points, dtype=int)
    colors[idx[0]] = 0
    for i, ind in enumerate(idx):
        connected_nodes = edge_dict[ind]
        c = choose_color(colors[list(connected_nodes)])
        colors[ind] = c
    return colors


def semi_greedy_solver(edges, n_points, degrees, edge_dict, order=None):

    # get indices by degree of the vertex
    if order is None:
        idx = np.argsort(degrees)[::-1]
    else:
        idx = np.array(order)
    colors = -np.ones(n_points, dtype=int)
    colors[idx[0]] = 0
    for i, ind in enumerate(idx):
        connected_nodes = edge_dict[ind]
        c = choose_color(colors[list(connected_nodes)])
        colors[ind] = c
    return colors


def cp_solver_test(edges, n_points):
    edge_array = np.array(edges)
    orders = [np.sum(edge_array == i) for i in range(n_points)]
    highest_order = int(np.argmax(orders))

    solver = cp_model.CpSolver()
    model = cp_model.CpModel()

    max_color = highest_order

    c = [model.NewIntVar(0, max_color, "i%i" % i) for i in range(0, n_points)]

    for i in range(len(edges)):
        model.Add(c[edges[i][0]] != c[edges[i][1]])

    model.Minimize(max([c[i] for i in range(0, n_points)]))

    status = solver.Solve(model)
    print("\t>>> Solver status: ", solver.StatusName(status))
    solution = [solver.Value(c[i]) for i in range(0, n_points)]
    # output_data = str(max(solution) + 1) + " " + str(solver.StatusName(status))
    return solution


def fixed_color(edges, n_points, n_colors):
    solver = pywrapcp.Solver("Colors")
    colors = [solver.IntVar(0, n_colors - 1, f"c_{i}") for i in range(n_points)]
    for i, j in edges:
        solver.Add(colors[i] != colors[j])
    db = solver.Phase(colors, solver.CHOOSE_FIRST_UNBOUND, solver.ASSIGN_MIN_VALUE)
    solver.NewSearch(db)

    if solver.NextSolution():
        return [c.Value() for c in colors]
    else:
        return []


def stepwise_opt(edges, node_count, n_colors=None):
    print("\n\n*** Initiating Step-wise optimisation ***")
    if n_colors is None:
        print(f"\nFinding Solution for {node_count} Nodes", flush=True)
        n_colors, colors = color_graph(edges, node_count)
        print(f"Found solution for {n_colors} colors", flush=True)
    else:
        colors = None
    while n_colors is not None:
        print(f"\tTrying {n_colors-1} colors", flush=True)
        n_colors_prev, colors_prev = n_colors, colors
        max_color = n_colors - 1
        n_colors, colors = color_graph(edges, node_count, max_color=max_color - 1)
    return n_colors_prev, colors_prev


def color_graph(edges, node_count, max_color=None, max_time=None):
    if max_color is None:
        degs = node_degree(edges)
        max_color = np.max(degs)

    if max_time is None:
        max_time = max_time_calc(edges, node_count)
    # Create model
    model = cp_model.CpModel()

    # Define Variable
    colors = [model.NewIntVar(0, int(max_color), f"c_{i}") for i in range(node_count)]

    # Define Constraints
    for n1, n2 in edges:
        model.Add(colors[n1] != colors[n2])

    # Symmetry breaking
    degs = node_degree(edges)
    idx = np.argmax(degs)
    model.Add(colors[idx] == 0)

    # Define Objective
    model.Minimize(max(*colors))

    # Create Solver
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max_time
    status = solver.Solve(model)

    if status == cp_model.FEASIBLE or status == cp_model.OPTIMAL:
        res = [solver.Value(c) for c in colors]
        n_colors = max(*res) + 1
        return n_colors, res
    else:
        return None, []


def max_time_calc(edges, node_count):
    pct = len(edges) / node_count ** 2
    t = node_count / 1000 * 20  # 10 sec base for 1000 nodes
    t = t * pct / 0.05  # scale based on percentage of complete graph
    return max(10, t * 2)


def node_degree(edges, n_nodes=None):
    occur = np.array(edges).flatten()
    if n_nodes is None:
        n_nodes = occur.max()
    return [len(occur[occur == o]) for o in range(n_nodes)]


def swap_elements(array, n=1, random_state=None):
    np.random.seed(random_state)
    x = np.array(array)
    length = len(x)
    for _ in range(n):
        i, j = np.random.choice(range(length), size=2, replace=False)
        x[i], x[j] = x[j], x[i]
    return x


def ga_solver(edges, n_points, random_state=None, n_rounds=1000):
    np.random.seed(random_state)

    # Get degree of each node
    degrees = get_degrees(edges, n_points)

    # Create a dictionary of connections for each node
    edge_dict = {i: set() for i in range(n_points)}
    for v1, v2 in edges:
        edge_dict[v1].add(v2)
        edge_dict[v2].add(v1)

    # Get initial solution using greedy
    solution = greedy_solver(edges, n_points)
    n_colors = max(solution) + 1
    idx = np.argsort(degrees)[::-1]
    n_greedy = n_colors

    # Genetic Algorithm

    # Initialize population
    population = [GreedyGraph(solution, idx, n_colors)]

    for _ in range(1, N_POP):
        n = np.random.randint(1, 6)
        new_idx = swap_elements(idx.copy(), n)
        res = semi_greedy_solver(
            edges,
            n_points,
            degrees,
            edge_dict,
            order=new_idx,
        )
        res_colors = np.max(res) + 1
        population.append(GreedyGraph(res, new_idx, res_colors))

    # Evolution
    pbar = trange(n_rounds)
    for i in pbar:
        best = np.min([g.count for g in population])
        worst = np.max([g.count for g in population])
        desc = f"{n_greedy} -> {best}/{worst}"
        pbar.set_description(desc, refresh=True)
        if best == worst:
            new_population = [
                deepcopy(g) for g in population if np.random.rand() > 0.75
            ]
        else:
            new_population = [deepcopy(g) for g in population if g.count == best]
        while len(new_population) < N_POP:
            selected = population[np.random.randint(N_POP)]
            n = np.random.randint(1, 6)
            new_idx = swap_elements(selected.order, n)
            res = semi_greedy_solver(
                edges,
                n_points,
                degrees,
                edge_dict,
                order=new_idx,
            )
            res_colors = np.max(res) + 1
            new_population.append(GreedyGraph(res, new_idx, res_colors))
        population = new_population.copy()

    id_sol = np.argmin([g.count for g in population])
    return population[id_sol].colors


def sort_by_degree(all_degrees, idx):
    degrees = np.array([all_degrees[i] for i in idx])
    argsort = np.argsort(degrees)[::-1]
    return np.array(idx)[argsort]


def shuffle_groups(degrees, groups):
    new_groups = [sort_by_degree(degrees, group) for group in groups.copy()]
    np.random.shuffle(new_groups)
    if isinstance(new_groups[0], list):
        new_idx = sum(new_groups, start=[])
    else:
        new_idx = np.concatenate(new_groups)
    return new_idx


def count_colors(colors):
    return len(set(colors))


def reordered_greedy(edges, n_points, random_state=None, trials=2000):
    # Set Seed value
    np.random.seed(random_state)

    # Get degree of each node
    degrees = get_degrees(edges, n_points)

    # Create a dictionary of connections for each node
    edge_dict = {i: set() for i in range(n_points)}
    for v1, v2 in edges:
        edge_dict[v1].add(v2)
        edge_dict[v2].add(v1)

    # Get initial solution using greedy
    solution = greedy_solver(edges, n_points)
    n_colors = count_colors(solution)
    idx = np.argsort(degrees)[::-1]
    n_greedy = n_colors

    pbar = trange(trials)
    for i in pbar:
        groups = [np.argwhere(solution == c).flatten() for c in range(n_colors)]
        new_groups = shuffle_groups(degrees, groups)
        solution = semi_greedy_solver(edges, n_points, degrees, edge_dict, new_groups)
        n_colors = count_colors(solution)
        desc = f"#{i+1:04.0f} \t{n_colors}/{n_greedy}"
        pbar.set_description(desc, refresh=True)
    return solution


import sys

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        print("*** Using file", file_location)
        with open(file_location, "r") as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print(
            "This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)"
        )
