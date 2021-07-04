#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
import math
from os.path import join
import os
from calc import length, total_cost
from algorithms import greedy, ex_local_search, random_allocation, clustering

ls = os.listdir

Point = namedtuple("Point", ["x", "y"])
Facility = namedtuple("Facility", ["index", "setup_cost", "capacity", "location"])
Customer = namedtuple("Customer", ["index", "demand", "location"])


def load_data(file, prefix="data"):
    with open(join(prefix, file)) as fp:
        content = fp.read()
    return parse_input(content)


def parse_input(input_data):
    lines = input_data.split("\n")

    parts = lines[0].split()
    facility_count = int(parts[0])
    customer_count = int(parts[1])

    facilities = []
    for i in range(1, facility_count + 1):
        parts = lines[i].split()
        facilities.append(
            Facility(
                index=i - 1,
                setup_cost=float(parts[0]),
                capacity=int(parts[1]),
                location=Point(float(parts[2]), float(parts[3])),
            )
        )

    customers = []
    for i in range(facility_count + 1, facility_count + 1 + customer_count):
        parts = lines[i].split()
        customers.append(
            Customer(
                index=i - 1 - facility_count,
                demand=int(parts[0]),
                location=Point(float(parts[1]), float(parts[2])),
            )
        )

    return customers, facilities


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    customers, facilities = parse_input(input_data)
    facility_count = len(facilities)
    customer_count = len(customers)

    # Solution
    mode = 1
    if mode == 0:
        solution = greedy(customers, facilities)
        solution = ex_local_search(solution, customers, facilities, True)
    elif mode == 1:
        solution = clustering(customers, facilities)
    else:
        solution = greedy(customers, facilities)

    # calculate the cost of the solution

    obj = total_cost(solution, customers, facilities)
    # prepare the solution in the specified output format
    output_data = "%.2f" % obj + " " + str(0) + "\n"
    output_data += " ".join(map(str, solution))

    return output_data


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
            "This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/fl_16_2)"
        )
