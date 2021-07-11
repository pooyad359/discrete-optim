import gurobipy as gp
from gurobipy import GRB
import numpy as np
from calc import distance_matrix
from itertools import product
from math import sqrt


def cap_mip_gr(customers, facilities, max_time=60):
    n_cust = len(customers)
    n_fac = len(facilities)
    dist = distance_matrix(customers, facilities)
    cartesian_prod = list(product(range(n_cust), range(n_fac)))
    setup_cost = [f.setup_cost for f in facilities]
    shipping_cost = {(c, f): dist[f, c] for c, f in cartesian_prod}
    demands = np.array([c.demand for c in customers])
    caps = np.array([f.capacity for f in facilities])
    solver = gp.Model("facility_location")

    # Define Variables
    x = solver.addVars(n_fac, vtype=GRB.BINARY, name="Select")
    y = solver.addVars(cartesian_prod, ub=1, vtype=GRB.CONTINUOUS, name="Assign")

    # Define Constraints
    solver.addConstrs((y[(c, f)] <= x[f] for c, f in cartesian_prod), name="Setup2ship")
    solver.addConstrs(
        (gp.quicksum(y[(c, f)] for f in range(n_fac)) == 1 for c in range(n_cust)),
        name="Demand",
    )
    solver.addConstrs(
        (
            gp.quicksum(y[(c, f)] * demands[c] for c in range(n_cust)) <= caps[f]
            for f in range(n_fac)
        ),
        name="Capacity",
    )
    # Set Objective
    solver.setObjective(x.prod(setup_cost) + y.prod(shipping_cost), GRB.MINIMIZE)
    solver.setParam("TimeLimit", max_time)

    # Solve
    solver.optimize()
    # Parse Outputs
    solution = np.zeros((n_fac, n_cust))
    for c in range(n_cust):
        for f in range(n_fac):
            solution[f, c] = y[(c, f)].x

    return solution.argmax(axis=0)