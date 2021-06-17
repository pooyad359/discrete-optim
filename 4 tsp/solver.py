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

Point = namedtuple("Point", ["x", "y"])


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
    mode = 2
    if nodeCount > 10000:
        mode = -1
    if mode == 0:
        solution = to_nearest(points)
        for _ in range(2):
            d = deque(solution)
            d.rotate(5)
            # d.rotate(len(solution) // 2)
            solution = two_opt(list(d), points)

    if mode == 1:
        n_restart = 20
        if len(points) > 1000:
            n_restart = 5
        solution = randomized_opt2(points, n_restart)
    elif mode == -1:
        with open("./prob_6_nearest.sol") as fp:
            output = fp.read()
            return output
    elif mode == 2:
        solution = ortools_solver(points)
    else:
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


@lru_cache(1024)
def distance(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


def view_tsp(solution, points, figsize=(8, 8)):
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
    # for i, (xi, yi) in enumerate(xy):
    #     plt.text(
    #         xi + 0.01, yi + 0.01, i, fontdict={"fontsize": 16, "color": "darkblue"}
    #     )
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


def randomized_opt2(points, restarts=1):
    if restarts > 1:
        solutions = []
        values = []
        for i in range(restarts):
            print(f"<<< Attempt {i+1}/{restarts}>>>")
            solution = randomized_opt2(points)
            value = loss(solution, points)
            solutions.append(solution)
            values.append(value)
        idx = np.argmin(values)
        return solutions[idx]
    else:
        trial1 = np.random.permutation(len(points))
        np.random.shuffle(trial1)
        solution = two_opt(trial1, points)
        trial2 = deque(solution)
        trial2.rotate(3)
        return two_opt(list(trial2), points)


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