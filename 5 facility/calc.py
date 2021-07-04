import math
import numpy as np


def total_demand(customers):
    return sum((c.demand for c in customers))


def total_cost(allocations, customers, facilities):
    try:
        validate(allocations, customers, facilities)
    except AssertionError:
        return np.inf
    used_facilities = set(allocations)
    capital = sum([facilities[f].setup_cost for f in used_facilities])
    travelling_cost = 0
    for c, f in enumerate(allocations):
        travelling_cost += length(customers[c].location, facilities[f].location)
    return capital + travelling_cost


def validate(allocations, customers, facilities):
    # Check all customers are allocated correctly
    assert min(allocations) >= 0, "Unassigned customer was found"
    assert max(allocations) < len(facilities), "Invalid facility number detected"

    # Check facilities are not overloaded
    for i in set(allocations):
        facility = facilities[i]
        allocated_customers = np.where(np.array(allocations) == i)[0]
        demand = 0
        cap = facility.capacity
        for c in allocated_customers:
            demand += customers[c].demand
        assert demand <= cap, f"Facility#{i} is overloaded: {demand:.1f}/{cap:.1f}"


def length(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


def distance_matrix(customers, facilities):
    pf = np.array([f.location for f in facilities])
    pc = np.array([c.location for c in customers])
    diff = (pf[:, None, :] - pc[None, :, :]) ** 2
    dist = diff.sum(axis=2) ** 0.5
    return dist