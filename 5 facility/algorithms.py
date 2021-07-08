import numpy as np
from calc import total_cost, total_demand, length, est_facilities, distance_matrix
from tqdm.auto import tqdm, trange
from exceptions import *
from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
from collections import Counter
from operator import itemgetter
from ortools.linear_solver.pywraplp import Solver


def double_trial(customers, facilities, solver, **kwargs):
    # First attempt
    solution = solver(customers, facilities, **kwargs)

    # Choosing top facilities
    min_fac = est_facilities(customers, facilities)
    sorted_facs = sorted(Counter(solution).items(), key=itemgetter(1), reverse=True)
    top_facs_idx = [f for f, i in sorted_facs[:min_fac]]
    top_facs = [facilities[f] for f in top_facs_idx]

    # Second Attempt
    solution2 = greedy_furthest(customers, top_facs, **kwargs)

    # Rearrange the solution
    solution2 = [top_facs_idx[f] for f in solution2]
    return solution2


def greedy(customers, facilities, eps=1e-3):
    allocations = -np.ones(len(customers))
    demand = total_demand(customers)
    facility_unit_cost = [f.setup_cost / f.capacity for f in facilities]
    ordered_facilities = np.argsort(facility_unit_cost)
    remaining_demand = demand
    for fi in ordered_facilities:
        facility = facilities[fi]
        distances = [
            length(facility.location, customer.location) for customer in customers
        ]
        sorted_cust = np.argsort(distances)
        remaining_capacity = facility.capacity
        for ci in sorted_cust:
            customer = customers[ci]
            if allocations[ci] >= 0:
                continue
            allocations[ci] = fi
            remaining_capacity -= customer.demand
            if remaining_capacity < 0:
                allocations[ci] = -1
                remaining_capacity += customer.demand
                break
    return allocations.astype(int)


def greedy_furthest(customers, facilities, ignore_setup=True, p_skip=0):
    dist = distance_matrix(customers, facilities)
    n_cust = len(customers)
    n_fac = len(facilities)
    solution = -np.ones(n_cust)
    opened_fac = np.zeros(n_fac)
    remaining_cap = np.array([f.capacity for f in facilities])
    setup_cost = np.array([f.setup_cost for f in facilities])
    dist_ord = dist.mean(axis=0).argsort()
    for c in tqdm(reversed(dist_ord), total=n_cust):
        customer = customers[c]
        choice_cost = dist[:, c]
        if not ignore_setup:
            choice_cost += (1 - opened_fac) * setup_cost
        for f in np.argsort(choice_cost):
            if remaining_cap[f] >= customer.demand:
                if np.random.rand() < p_skip:
                    continue
                opened_fac[f] = 1
                remaining_cap[f] -= customer.demand
                solution[c] = f
                break
    return solution.astype(int)


def random_allocation(customers, facilities):
    max_iter = 10000
    n_fac = len(facilities)
    allocations = -np.ones(len(customers))
    remaining_capacity = np.array([f.capacity for f in facilities])
    for i, customer in enumerate(customers):
        for counter in range(max_iter):
            selected_facility = np.random.choice(
                range(n_fac),
                p=remaining_capacity / remaining_capacity.sum(),
            )
            if remaining_capacity[selected_facility] >= customer.demand:
                remaining_capacity[selected_facility] -= customer.demand
                allocations[i] = selected_facility
                break
        else:
            raise IterationError("Maximum number of iterations reached.")
    return allocations.astype(int)


def ex_local_search(solution, customers, facilities, njobs=1):
    allocations = solution.copy()
    n_cutomers = len(customers)
    n_facilities = len(facilities)
    old_cost = total_cost(allocations, customers, facilities)
    pbar = trange(n_cutomers)
    for i in pbar:
        customer = customers[i]
        costs = np.zeros(n_facilities)
        old_alloc = allocations[i]

        parallel = Parallel(n_jobs=njobs)
        delayed_func = delayed(eval_swap_values)
        costs = parallel(
            delayed_func(
                allocations=allocations,
                customers=customers,
                facilities=facilities,
                customer=i,
                new_facility=j,
            )
            for j in range(n_facilities)
        )
        new_alloc = np.argmin(costs)
        allocations[i] = new_alloc
        desc = "{:.1f} --> {:.1f} --> {:.1f}".format(
            old_cost,
            costs[old_alloc],
            costs[new_alloc],
        )
        pbar.set_description(desc)
    return allocations


def k_local_search(solution, customers, facilities, k=5, njobs=1):
    allocations = solution.copy()
    n_cutomers = len(customers)
    n_facilities = len(facilities)
    old_cost = total_cost(allocations, customers, facilities)
    pbar = trange(n_cutomers)
    for i in pbar:
        last_cost = total_cost(allocations, customers, facilities)
        customer = customers[i]
        costs = np.zeros(n_facilities)
        old_alloc = allocations[i]

        current_fac = solution[i]
        closest_facs = find_k_neighbors([facilities[current_fac]], facilities, k)[0]
        parallel = Parallel(n_jobs=njobs)
        delayed_func = delayed(eval_swap_values)
        costs = parallel(
            delayed_func(
                allocations=allocations,
                customers=customers,
                facilities=facilities,
                customer=i,
                new_facility=closest_facs[j],
            )
            for j in range(k)
        )
        new_alloc = closest_facs[np.argmin(costs)]
        allocations[i] = new_alloc
        desc = "{:.1f} --> {:.1f} --> {:.1f}".format(
            old_cost,
            last_cost,
            min(costs),
        )
        pbar.set_description(desc)
    return allocations


def find_k_neighbors(x, y, k=1):
    """
    Finds k nearest neighbors for x in y. x and y are either Facilities or Customers,
    and must have location property.
    """
    x = np.array([o.location for o in x])
    y = np.array([o.location for o in y])
    kdtree = KDTree(y)
    dist, neighbors = kdtree.query(x, k)
    return neighbors


def clustering(customers, facilities):
    n_cust = len(customers)
    n_fac = len(facilities)

    # Find minimum number of clusters
    min_fac = est_facilities(customers, facilities)

    # Find cluster centroids
    kmean = KMeans(min_fac)
    xy = np.array([c.location for c in customers])
    kmean.fit(xy)
    centroids = kmean.cluster_centers_

    # Find the distance between facilities and centroids
    fneigh = KDTree(np.array([f.location for f in facilities]))
    dist, locs = fneigh.query(centroids, min_fac)

    # Match facilities with centroids
    inactive_facs = set(range(n_fac))
    active_facs = []
    for i in range(min_fac):
        for j in range(min_fac):
            if locs[i, j] in inactive_facs:
                inactive_facs = inactive_facs - {locs[i, j]}
                active_facs.append(locs[i, j])
                break

    # Assign facilities to customers
    turn = 0
    remaining_cap = [facilities[f].capacity for f in active_facs]
    remaining_cust = set(range(n_cust))
    max_iter = len(active_facs) * n_cust
    sol = -np.ones(n_cust, dtype=int)
    pbar = trange(n_cust)
    for _ in range(max_iter):
        turn += 1
        turn %= len(active_facs)
        turn_fac = active_facs[turn]
        kd_cust = KDTree(
            np.array([customers[c].location for c in list(remaining_cust)])
        )
        dist, pot_cust = kd_cust.query([facilities[turn_fac].location], 1)
        pot_cust = list(remaining_cust)[pot_cust[0][0]]
        if remaining_cap[turn] < customers[pot_cust].demand:
            continue
        else:
            pbar.update()
            sol[pot_cust] = turn_fac
            remaining_cust = remaining_cust - {pot_cust}
            remaining_cap[turn] -= customers[pot_cust].demand
        if min(sol) >= 0:
            break
    else:
        IterationError("Maximum number of iteration reached.")
    return sol


def eval_swap_values(allocations, customers, facilities, customer, new_facility):
    alloc = allocations.copy()
    alloc[customer] = new_facility
    return total_cost(alloc, customers, facilities)


def uncap_mip(customers, facilities, max_time=60):
    n_fac = len(facilities)
    n_cust = len(customers)
    solver = Solver.CreateSolver("FacilityLocation", "SCIP")

    x = []
    y = []
    for f in range(n_fac):
        y.append([solver.BoolVar(f"y_{f}_{c}") for c in range(n_cust)])
        x.append(solver.BoolVar(f"x_{f}"))

    caps = [f.capacity for f in facilities]
    setup = [f.setup_cost for f in facilities]
    dist = distance_matrix(customers, facilities).astype(int)
    demands = [c.demand for c in customers]

    for f in range(n_fac):
        for c in range(n_cust):
            solver.Add(y[f][c] <= x[f])
    for c in range(n_cust):
        solver.Add(sum([y[f][c] for f in range(n_fac)]) == 1)

    obj = 0
    for f in range(n_fac):
        #     obj += setup[f]*x[f]
        obj += sum([dist[f][c] * y[f][c] for c in range(n_cust)])

    solver.Minimize(obj)

    STATUS = {
        Solver.FEASIBLE: "FEASIBLE",
        Solver.UNBOUNDED: "UNBOUNDED",
        Solver.BASIC: "BASIC",
        Solver.INFEASIBLE: "INFEASIBLE",
        Solver.NOT_SOLVED: "NOT_SOLVED",
        Solver.OPTIMAL: "OPTIMAL",
    }
    solver.SetTimeLimit(max_time * 1000)

    status = solver.Solve()
    STATUS[status]
    a = []
    for f in range(n_fac):
        a.append([y[f][c].solution_value() for c in range(n_cust)])

    sol = np.array(a).argmax(axis=0)
    return sol, STATUS[status]


def uncap_mip_iter(customers, facilities, triple=True, max_time=60):
    min_fac = est_facilities(customers, facilities)

    # Round 1
    sol, status = uncap_mip(customers, facilities, max_time)
    print(f"Initial Status: {status}")

    # Round 2
    if triple:
        cntr = list(Counter(sol).items())
        fid = sorted(cntr, key=itemgetter(1), reverse=True)[: min_fac * 2]
        fid = sorted([o[0] for o in fid])
        fac_sub = [f for i, f in enumerate(facilities) if i in fid]
        sol, status = uncap_mip(customers, fac_sub, max_time)
        sol = [fid[f] for f in sol]
        print(f"Intermediate Status: {status}")

    # Round 3
    cntr = list(Counter(sol).items())
    fid = sorted(cntr, key=itemgetter(1), reverse=True)[:min_fac]
    fid = sorted([o[0] for o in fid])
    fac_sub = [f for i, f in enumerate(facilities) if i in fid]
    sol, status = uncap_mip(customers, fac_sub, max_time)
    sol = [fid[f] for f in sol]
    print(f"Final Status: {status}")

    return sol


def cap_mip(customers, facilities, max_time=60):
    n_fac = len(facilities)
    n_cust = len(customers)
    solver = Solver.CreateSolver("FacilityLocation", "SCIP")

    x = []
    y = []
    for f in range(n_fac):
        y.append([solver.BoolVar(f"y_{f}_{c}") for c in range(n_cust)])
        x.append(solver.BoolVar(f"x_{f}"))

    caps = [f.capacity for f in facilities]
    setup = [f.setup_cost for f in facilities]
    dist = distance_matrix(customers, facilities).astype(int)
    demands = [c.demand for c in customers]

    for f in range(n_fac):
        for c in range(n_cust):
            solver.Add(y[f][c] <= x[f])
    for c in range(n_cust):
        solver.Add(sum([y[f][c] for f in range(n_fac)]) == 1)
    for f in range(n_fac):
        solver.Add(sum([y[f][c] * demands[c] for c in range(n_cust)]) <= caps[f])

    obj = 0
    for f in range(n_fac):
        obj += setup[f] * x[f]
        obj += sum([dist[f][c] * y[f][c] for c in range(n_cust)])

    solver.Minimize(obj)

    STATUS = {
        Solver.FEASIBLE: "FEASIBLE",
        Solver.UNBOUNDED: "UNBOUNDED",
        Solver.BASIC: "BASIC",
        Solver.INFEASIBLE: "INFEASIBLE",
        Solver.NOT_SOLVED: "NOT_SOLVED",
        Solver.OPTIMAL: "OPTIMAL",
    }
    solver.SetTimeLimit(max_time * 1000)

    status = solver.Solve()
    STATUS[status]
    a = []
    for f in range(n_fac):
        a.append([y[f][c].solution_value() for c in range(n_cust)])

    sol = np.array(a).argmax(axis=0)
    return sol, STATUS[status]