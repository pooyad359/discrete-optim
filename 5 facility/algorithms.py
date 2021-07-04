import numpy as np
from calc import total_cost, total_demand, length, min_facilities
from tqdm.auto import tqdm, trange
from exceptions import *
from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans


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


def ex_local_search(solution, customers, facilities, verbose=False):
    allocations = solution.copy()
    n_cutomers = len(customers)
    n_facilities = len(facilities)
    old_cost = total_cost(allocations, customers, facilities)
    pbar = trange(n_cutomers)
    for i in pbar:
        customer = customers[i]
        costs = np.zeros(n_facilities)
        old_alloc = allocations[i]
        for j in range(n_facilities):
            allocations[i] = j
            costs[j] = total_cost(allocations, customers, facilities)
        new_alloc = np.argmin(costs)
        allocations[i] = new_alloc
        if verbose:
            desc = "{:.1f} --> {:.1f} --> {:.1f}".format(
                old_cost,
                costs[old_alloc],
                costs[new_alloc],
            )
            pbar.set_description(desc)
    return allocations


def clustering(customers, facilities):
    n_cust = len(customers)
    n_fac = len(facilities)

    # Find minimum number of clusters
    min_fac = min_facilities(customers, facilities)

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