#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

import os
from functools import lru_cache
import math
from collections import namedtuple

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
    solution = to_nearest(points)

    # calculate the length of the tour
    obj = length(points[solution[-1]], points[solution[0]])
    for index in range(0, nodeCount - 1):
        obj += length(points[solution[index]], points[solution[index + 1]])

    # prepare the solution in the specified output format
    output_data = "%.2f" % obj + " " + str(0) + "\n"
    output_data += " ".join(map(str, solution))

    return output_data


@lru_cache(256)
def distance(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


def view_tsp(solution, figsize=(8, 8)):
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
    for i, (xi, yi) in enumerate(xy):
        plt.text(
            xi + 0.01, yi + 0.01, i, fontdict={"fontsize": 16, "color": "darkblue"}
        )


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
