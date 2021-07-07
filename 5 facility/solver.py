#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
import math
from os.path import join
import os
from calc import length, total_cost
from algorithms import cap_mip, greedy, ex_local_search, random_allocation
from algorithms import clustering, greedy_furthest, double_trial

ls = os.listdir

Point = namedtuple("Point", ["x", "y"])
Facility = namedtuple("Facility", ["index", "setup_cost", "capacity", "location"])
Customer = namedtuple("Customer", ["index", "demand", "location"])


def load_data(file, prefix="data"):
    with open(join(prefix, file)) as fp:
        content = fp.read()
    return parse_input(content)


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    customers, facilities = parse_input(input_data)
    facility_count = len(facilities)
    customer_count = len(customers)

    # Solution
    # Selecting the solver
    if facility_count * customer_count < 50_000:
        mode = 3
    else:
        mode = 3

    # Applying the solver
    status = "FEASIBLE"
    if mode == 0:
        solution = greedy(customers, facilities)
        solution = ex_local_search(solution, customers, facilities, True)
    elif mode == 1:
        solution = clustering(customers, facilities)
    elif mode == 2:
        solution, status = cap_mip(customers, facilities)
        print(f'MIP Solver finished with status "{status}"')
    elif mode == 3:
        print("Double trial with greedy")
        solution = double_trial(customers, facilities, greedy_furthest)
    else:
        solution = greedy(customers, facilities)
        status = "FEASIBLE"

    # calculate the cost of the solution

    obj = total_cost(solution, customers, facilities)
    # prepare the solution in the specified output format
    status_code = 1 if status == "OPTIMAL" else 0
    output_data = "%.2f" % obj + " " + str(status_code) + "\n"
    output_data += " ".join(map(str, solution))

    return output_data


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
