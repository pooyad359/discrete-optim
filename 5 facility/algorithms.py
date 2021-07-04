import numpy as np
from calc import total_cost, total_demand, length
from tqdm.auto import tqdm, trange
from exceptions import *


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


def local_search(solution, customers, facilities, verbose=False):
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