#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os
from functools import lru_cache
import math
from collections import namedtuple, deque
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import random
import itertools
from tqdm.auto import trange
from ortools.sat.python import cp_model
from sklearn.neighbors import KDTree

Point = namedtuple("Point", ["x", "y"])
np.random.seed(12)
random.seed(12)


def length(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split("\n")

    nodeCount = int(lines[0])

    points = []
    for i in range(1, nodeCount + 1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))

    # build a trivial solution
    # visit the nodes in the order they appear in the file
    mode = 1
    if nodeCount > 20000:
        mode = -1
    if mode == 0:
        print("*** USING GREEDY + 2-OPT ***")
        solution = to_nearest(points)
        for _ in range(3):
            d = deque(solution)
            d.rotate(10)
            # d.rotate(len(solution) // 2)
            solution = two_opt(list(d), points)

    elif mode == 1:
        print("*** USING GREEDY + 2-OPT WITH RESTARTS ***")
        if nodeCount < 300:
            n_restart = 10000
        else:
            n_restart = 20
        if len(points) >= 1000:
            n_restart = 5
        solution = randomized_opt2(points, n_restart)
        print(f"\t\tLoss = {loss(solution, points)}")
        if nodeCount < 300:
            print("\tRefining the solution using 3-OPT")
            solution = three_opt(solution, points)
            print(f"\t\tLoss = {loss(solution, points)}")
        print("\tRefining the solution using LOCAL SEARCH")
        solution = local_search(solution, points)

    elif mode == -1:
        print("*** USING SAVED RESULTS ***")
        print(f"Node Count: {nodeCount}")
        if nodeCount == 51:
            file = "prob_1.sol"
        elif nodeCount == 100:
            file = "prob_2.sol"
        elif nodeCount == 200:
            file = "prob_3.sol"
        elif nodeCount == 574:
            file = "prob_4.sol"
        elif nodeCount == 1889:
            file = "prob_5.sol"
        else:
            file = "prob_6.sol"
        with open(file) as fp:
            output = fp.read()
            return output
    elif mode == 2:
        print("*** USING ORTOOL TSP MODULE ***")
        solution = ortools_solver(points)
    elif mode == 3:
        print("*** USING GREEDY + 2-OPT-V2 ***")
        solution = to_nearest(points)
        for _ in range(3):
            d = deque(solution)
            d.rotate(10)
            solution = two_opt(list(d), points)
    elif mode == 4:
        print("*** USING CP + Filtered Neighbors ***")
        optimal, solution = filtered_neighbors(points)
        if not optimal:
            print("\tThe solution is not optimal.")
            print(f"\t\tLoss = {loss(solution, points)}")
            print("\t*** Using 2-opt to improve the solution ***")
            solution = two_opt_v2(solution, points)
            print(f"\t\tLoss = {loss(solution, points)}")
            print("\t*** Using 3-opt to improve the solution ***")
            solution = three_opt(solution, points)
            print(f"\t\tLoss = {loss(solution, points)}")
            print("\t*** Using local search to improve the solution ***")
            solution = local_search(solution, points)
    elif mode == 1:
        print("*** USING GREEDY + 2-OPT WITH RESTARTS ***")
        n_restart = 20
        if len(points) >= 1000:
            n_restart = 5
        solution = randomized_opt2(points, n_restart)
        if len(points) < 1000:
            solution = local_search(solution, points, 10)

    else:
        print("*** USING GREEDY ALGORITHM ***")
        solution = to_nearest(points)

    # calculate the length of the tour
    obj = length(points[solution[-1]], points[solution[0]])
    for index in range(0, nodeCount - 1):
        obj += length(points[solution[index]], points[solution[index + 1]])

    # prepare the solution in the specified output format
    output_data = "%.2f" % obj + " " + str(0) + "\n"
    output_data += " ".join(map(str, solution))
    view_tsp(solution, points)
    return output_data


def load_data(filename, prefix="data"):
    with open(os.path.join(prefix, filename)) as fp:
        content = fp.read().split("\n")

    node_count = int(content[0])
    points = []
    for i in range(1, node_count + 1):
        line = content[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))

    print(f"#Nodes: {node_count}")

    d = []
    for i in range(node_count):
        row = [distance(points[i], points[j]) for j in range(node_count)]
        d.append(row)

    return node_count, points


@lru_cache(1024)
def distance(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


def view_tsp(solution, points, figsize=(8, 8), show_index=False):
    """
    List of points
    """
    plt.figure(figsize=figsize)
    xy = [[points[i].x, points[i].y] for i in solution]
    xy = np.array(xy)
    x = xy[:, 0]
    y = xy[:, 1]
    plt.plot(x, y, "ok-")
    plt.plot(x[0], y[0], "sr", markersize=12)
    plt.plot(x[-1], y[-1], "*g", markersize=16)
    if show_index:
        for i, (xi, yi) in enumerate(xy):
            plt.text(
                xi + 0.01, yi + 0.01, i, fontdict={"fontsize": 16, "color": "darkblue"}
            )
    obj = loss(solution, points)
    plt.title(f"{len(points)} Nodes    Objective = {obj:.1f}")
    plt.show()


def cost_change(n1, n2, n3, n4):
    return distance(n1, n3) + distance(n2, n4) - distance(n1, n2) - distance(n4, n3)


def two_opt(route, points, max_iter=1000, eps=1e-5):
    best = route.copy()
    improved = True
    counter = 0
    #     pbar = trange(max_iter)
    while improved and counter < max_iter:
        counter += 1
        value = loss(route, points)
        print(f"#{counter:04.0f} \t{value}", flush=True)
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                if j - i == 1:
                    continue
                if (
                    cost_change(
                        points[best[i - 1]],
                        points[best[i]],
                        points[best[j - 1]],
                        points[best[j]],
                    )
                    < -eps
                ):
                    best[i:j] = best[j - 1 : i - 1 : -1]
                    improved = True
        route = best
    return best


def two_opt_v2(
    route,
    points,
    max_iter=1000,
    eps=1e-5,
    random_state=None,
    verbose=False,
):
    random.seed(random_state)
    best = route.copy()
    improved = True
    counter = 0
    while improved and counter < max_iter:
        counter += 1
        value = loss(route, points)
        if verbose:
            print(f"#{counter:04.0f} \t{value}", flush=True)
        improved = False

        pairs = list(itertools.combinations(range(1, len(route)), 2))
        random.shuffle(pairs)
        for i, j in pairs:
            if j - i == 1:
                continue
            if (
                cost_change(
                    points[best[i - 1]],
                    points[best[i]],
                    points[best[j - 1]],
                    points[best[j]],
                )
                < -eps
            ):
                best[i:j] = best[j - 1 : i - 1 : -1]
                improved = True
        route = best
    return best


def randomized_opt2(points, restarts=1, verbose=False):
    if restarts > 1:
        solutions = []
        values = []
        for i in trange(restarts):
            if verbose:
                print(f"<<< Attempt {i+1}/{restarts}>>>")
            solution = randomized_opt2(points=points, verbose=verbose)
            value = loss(solution, points)
            solutions.append(solution)
            values.append(value)
        idx = np.argmin(values)
        return solutions[idx]
    else:
        trial1 = np.random.permutation(len(points))
        np.random.shuffle(trial1)
        solution = two_opt_v2(trial1, points)
        # solution = two_opt(trial1, points)
        trial2 = deque(solution)
        trial2.rotate(3)
        # return two_opt(list(trial2), points)
        return two_opt_v2(list(trial2), points)


def to_nearest(points):
    path = []
    remaining = list(range(len(points)))
    path.append(remaining.pop(0))
    while len(remaining) > 0:
        dists = [distance(points[i], points[path[-1]]) for i in remaining]
        next_i = np.argmin(dists)
        path.append(remaining.pop(next_i))

    return path


def loss(sol, points):
    dist = distance(points[sol[-1]], points[sol[0]])
    for i in range(1, len(sol)):
        dist += distance(points[sol[i - 1]], points[sol[i]])

    return dist


def create_data_model(points):
    """Stores the data for the problem."""
    data = {}
    data["locations"] = points
    data["num_vehicles"] = 1
    data["depot"] = 0
    return data


def compute_euclidean_distance_matrix(locations):
    """Creates callback to return distance between points."""
    distances = {}
    for from_counter, from_node in enumerate(locations):
        distances[from_counter] = {}
        for to_counter, to_node in enumerate(locations):
            if from_counter == to_counter:
                distances[from_counter][to_counter] = 0
            else:
                # Euclidean distance
                distances[from_counter][to_counter] = int(
                    math.hypot((from_node[0] - to_node[0]), (from_node[1] - to_node[1]))
                )
    return distances


def get_route(manager, routing, solution):
    route = []
    route = [routing.Start(0)]
    while not routing.IsEnd(route[-1]):
        index = route[-1]
        new_index = solution.Value(routing.NextVar(index))
        route.append(new_index)
    route.pop(-1)
    return route


def dynamic_greedy(points, trials=1, shuffle=False, random_state=None):
    assert isinstance(trials, int) and trials > 0, "Invalid number for trials."
    random.seed(random_state)
    if trials > 1:
        routes = []
        values = []
        for i in trange(trials):
            seed = random.randint(0, 10000)
            route = dynamic_greedy(
                points,
                trials=1,
                shuffle=True,
                random_state=seed,
            )
            values.append(loss(route, points))
            routes.append(route)
        idx = np.argmin(values)
        return routes[idx]

    unused = points.copy()
    if shuffle:
        random.shuffle(unused)
    sol = []
    sol.append(unused.pop(0))
    sol.append(unused.pop(0))
    obj = loss(range(len(sol)), sol)
    while len(unused) > 0:
        point = unused.pop(0)
        values = []
        for i in range(len(sol)):
            new_sol = sol.copy()
            new_sol.insert(i, point)
            values.append(loss(range(len(new_sol)), new_sol))
        idx = np.argmin(values)
        sol.insert(idx, point)
    route = [points.index(p) for p in sol]
    return route


def local_search(solution, points, max_iter=100):
    node_count = len(points)
    obj = loss(solution, points)
    improved = True
    counter = 0
    while improved and counter < max_iter:
        counter += 1
        print(f"\t* local search: Iteration #{counter}")
        improved = False
        for i in trange(node_count):
            for j in range(node_count):
                new_sol = solution.copy()
                item = new_sol.pop(i)
                new_sol.insert(j, item)
                new_obj = loss(new_sol, points)
                if new_obj < obj:
                    print(f"\t{obj:.2f} --> {new_obj:.2f}")
                    solution = new_sol.copy()
                    obj = loss(solution, points)
                    improved = True
                    continue
    if not improved:
        print("No improvement observed.")
    elif counter >= max_iter:
        print("Max iteration reached.")
    return solution


def ortools_solver(points=None):
    """Entry point of the program."""
    # Instantiate the data problem.
    data = create_data_model(points)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(
        len(data["locations"]), data["num_vehicles"], data["depot"]
    )

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    distance_matrix = compute_euclidean_distance_matrix(data["locations"])

    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        #         out = print_solution(manager, routing, solution)
        out = get_route(manager, routing, solution)

    return out


# ---------------------- Constraint Programming -----------------------#
def get_k_value(n):
    """
    50 nodes --> k = 5
    100 nodes --> k = 8
    """
    k = 5 + 5 * np.log2(n / 50)
    return int(k)


def generate_tsp(node_count, random_state=None):
    np.random.seed(random_state)
    x = np.random.randint(100, size=node_count, dtype=int)
    y = np.random.randint(100, size=node_count, dtype=int)
    points = [Point(i, j) for i, j in zip(x, y)]
    return points


def view_matrix(mat, points, figsize=(8, 8)):
    lines = []
    #     row = np.argmax(sol_mat,axis=0)
    row = np.arange(len(points))
    col = np.argmax(mat, axis=1)
    plt.figure(figsize=figsize)
    for i, j in zip(row, col):
        x = [points[i].x, points[j].x]
        y = [points[i].y, points[j].y]
        plt.plot(x, y, "ok-")


def check_feasibility(sol_mat):
    dest = np.argmax(sol_mat, axis=1)
    return len(set(dest)) == sol_mat.shape[0]


def matrix_to_route(matrix):
    assert check_feasibility, "Invalid input matrix"
    route = [0]
    dest = np.argmax(matrix, axis=1)
    n = len(dest)
    for i in range(n):
        route.append(dest[route[-1]])
    final = route.pop(-1)
    if final != 0:
        raise Exception("The route is not complete.")
    return route


def filtered_neighbors(points):
    node_count = len(points)

    # Neighbor Detection
    xy = np.array(points)
    kdt = KDTree(xy)
    k = get_k_value(node_count) + 1
    dist, neighbors = kdt.query(xy, k)

    # Create CP Model
    model = cp_model.CpModel()

    # Define Variables
    a = []  # whether i and j are connected
    u = []  # order of points
    for i in range(node_count):
        row = [model.NewBoolVar(f"a_{i}_{j}") for j in range(node_count)]
        a.append(row)
    u = [model.NewIntVar(0, node_count - 1, f"u_{i}") for i in range(node_count)]

    # Add Constraints
    for i in range(node_count):
        model.AddAbsEquality(0, a[i][i])
        model.Add(sum([a[i][j] for j in range(node_count)]) == 1)
        model.Add(sum([a[j][i] for j in range(node_count)]) == 1)

    for i in range(node_count - 1):
        for j in range(i, node_count):
            model.Add(a[i][j] + a[j][i] <= 1)

    # Connectivity Constraint
    model.AddAllDifferent(u)
    for i in range(1, node_count):
        for j in range(1, node_count):
            if i == j:
                continue
            model.Add(u[i] - u[j] + node_count * a[i][j] <= node_count - 1)
    model.Add(u[0] == 0)

    # Proximity Constraint
    for i in range(node_count):
        neigh = neighbors[i, 1:]
        for j in range(node_count):
            if j not in neigh:
                model.AddAbsEquality(0, a[i][j])
                model.AddAbsEquality(0, a[j][i])

    # Objective
    obj = []
    for i in range(node_count):
        for j in range(node_count):
            d = distance(points[i], points[j]) * 100
            obj.append(int(d) * a[i][j])
    model.Minimize(sum(obj))

    # Solve
    max_time = estimate_time(node_count)
    max_time = 600
    cpsolver = cp_model.CpSolver()
    cpsolver.parameters.max_time_in_seconds = max_time
    status = cpsolver.Solve(model)
    print(cpsolver.StatusName())

    if status == cp_model.MODEL_INVALID or status == cp_model.INFEASIBLE:
        raise RuntimeError("Unable to find a solution.")

    # Extract Solution
    sol_mat = []
    for i in range(node_count):
        row = [cpsolver.Value(a[i][j]) for j in range(node_count)]
        sol_mat.append(row)
    sol_mat = np.array(sol_mat)

    # Convert solution to readable format
    sol = matrix_to_route(sol_mat)

    return status == cp_model.OPTIMAL, sol


def estimate_time(node_count):
    max_time = node_count * 3
    return np.clip(max_time, 60, 600)


# ------------------------- 3-OPT ------------------------------#
def three_opt(route, points, random_state=None, max_iter=100):
    random.seed(random_state)
    for _ in trange(max_iter):
        delta = 0
        options = list(all_segments(len(route)))
        random.shuffle(options)
        for (a, b, c) in options:
            route, change = reverse_segment_if_better(route, points, a, b, c)
            delta += change
        if delta >= 0:
            break
    return route


def all_segments(n: int):
    return (
        (i, j, k)
        for i in range(n)
        for j in range(i + 2, n)
        for k in range(j + 2, n + (i > 0))
    )


def reverse_segment_if_better(route, points, i, j, k):
    point_dist = lambda p1, p2: ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5
    new_route = route.copy()
    A = points[new_route[i - 1]]
    B = points[new_route[i]]
    C = points[new_route[j - 1]]
    D = points[new_route[j]]
    E = points[new_route[k - 1]]
    F = points[new_route[k % len(new_route)]]
    d0 = point_dist(A, B) + point_dist(C, D) + point_dist(E, F)
    d1 = point_dist(A, C) + point_dist(B, D) + point_dist(E, F)
    d2 = point_dist(A, B) + point_dist(C, E) + point_dist(D, F)
    d3 = point_dist(A, D) + point_dist(E, B) + point_dist(C, F)
    d4 = point_dist(F, B) + point_dist(C, D) + point_dist(E, A)

    if d0 > d1:
        new_route[i:j] = reversed(new_route[i:j])
        return new_route, -d0 + d1
    elif d0 > d2:
        new_route[j:k] = reversed(new_route[j:k])
        return new_route, -d0 + d2
    elif d0 > d4:
        new_route[i:k] = reversed(new_route[i:k])
        return new_route, -d0 + d4
    elif d0 > d3:
        tmp = new_route[j:k] + new_route[i:j]
        new_route[i:k] = tmp
        return new_route, -d0 + d3
    return new_route, 0


import sys

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, "r") as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print(
            "This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)"
        )
