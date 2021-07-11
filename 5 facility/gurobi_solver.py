import gurobipy as gp
from gurobipy import GRB
import numpy as np
from calc import distance_matrix, validate
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
    y = solver.addVars(cartesian_prod, vtype=GRB.BINARY, name="Assign")

    # Define Constraints
    solver.addConstrs((y[(c, f)] <= x[f] for c, f in cartesian_prod), name="Setup2ship")
    solver.addConstrs(
        (gp.quicksum(y[(c, f)] for f in range(n_fac)) == 1 for c in range(n_cust)),
        name="Demand",
    )
    solver.addConstrs(
        (
            gp.quicksum(y[(c, f)] * demands[c] for c in range(n_cust)) <= caps[f] * x[f]
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


def get_grid_size(n_facilities, per_grid=100):
    for n in range(1, 10):
        if n_facilities / n ** 2 <= per_grid:
            return n
    else:
        return None


def is_inside(point, xrange, yrange):
    return xrange[0] < point.x < xrange[1] and yrange[0] < point.y < yrange[1]


def piecewise_mip(customers, facilities, per_grid=100, max_time=60):
    n_cust = len(customers)
    n_fac = len(facilities)
    n = get_grid_size(n_fac, per_grid)
    x = [o.location.x for o in facilities + customers]
    y = [o.location.y for o in facilities + customers]
    xmin, xmax = min(x) - 1, max(x) + 1
    ymin, ymax = min(y) - 1, max(y) + 1
    xrange = np.linspace(xmin, xmax, n + 1)
    yrange = np.linspace(ymin, ymax, n + 1)
    allocations = []
    for i in range(n):
        xl = xrange[i]
        xu = xrange[i + 1]
        for j in range(n):
            print(f"*** Row#{i+1}/{n} Column#{j+1}/{n} ***")
            yl = yrange[j]
            yu = yrange[j + 1]

            cgrid = [c for c in customers if is_inside(c.location, (xl, xu), (yl, yu))]
            fgrid = [f for f in facilities if is_inside(f.location, (xl, xu), (yl, yu))]

            solgrid = cap_mip_gr(cgrid, fgrid)
            try:
                validate(solgrid, cgrid, fgrid)
            except AssertionError:
                print("***************************")
                print("***  INVALID SOLUTION   ***")
                print("***************************")
            allocations += [
                (cgrid[c].index, fgrid[f].index) for c, f in enumerate(solgrid)
            ]
    solution = -np.ones(n_cust)
    for c, f in allocations:
        solution[c] = f
    return np.int32(solution)