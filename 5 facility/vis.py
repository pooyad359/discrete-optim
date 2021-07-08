import matplotlib.pyplot as plt
import numpy as np
from calc import min_facilities, total_cost, validate


def view_problem(customers, facilities, figsize=(8, 8)):
    loc_cust = np.array([c.location for c in customers])
    loc_fac = np.array([f.location for f in facilities])
    plt.figure(figsize=figsize)
    plt.scatter(loc_cust[:, 0], loc_cust[:, 1], marker="o", c="k", s=15)
    plt.scatter(loc_fac[:, 0], loc_fac[:, 1], marker="*", c="r", s=100)

    plt.title(f"{len(customers)} Customers    {len(facilities)} Facilities")

    # Adjust plot limits
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    dx = xmax - xmin
    dy = ymax - ymin
    diff = max(dx, dy)
    plt.xlim((xmin, xmin + diff))
    plt.ylim((ymin, ymin + diff))


def view_solution(solution, customers, facilities, figsize=(8, 8)):
    loc_cust = np.array([c.location for c in customers])
    loc_fac = np.array([f.location for f in facilities])
    plt.figure(figsize=figsize)
    plt.scatter(loc_cust[:, 0], loc_cust[:, 1], marker="o", c="k", s=5)
    plt.scatter(loc_fac[:, 0], loc_fac[:, 1], marker="*", c="r", s=5)

    cost = total_cost(solution, customers, facilities)
    s = ""
    try:
        validate(solution, customers, facilities)
    except AssertionError:
        s = "*"
    plt.title(
        f"{len(customers)} Customers    {len(facilities)} Facilities   Total Cost{s} = {cost:.1f}"
    )
    for c, f in enumerate(solution):
        x = [facilities[f].location.x, customers[c].location.x]
        y = [facilities[f].location.y, customers[c].location.y]
        plt.plot(x, y, "-k", alpha=0.5)


def problem_analysis(customers, facilities, plot=True):
    demands = [c.demand for c in customers]
    caps = np.array([f.capacity for f in facilities])
    print(f"Customers: {len(customers)}")
    print(f"Facilities: {len(facilities)}")
    print(f"Total Demand: {sum(demands)}")
    print(f"Minimum = {demands.min()} \t Maximum = {demands.max}")
    print(f"Total Capacity: {sum(caps)}")
    print(f"Minimum = {caps.min()} \t Maximum = {caps.max}")
    print(f"Minimum Facilities Required: {min_facilities(customers,facilities)}")
    if plot:
        # Capacity
        plt.figure()
        plt.hist(caps, 100)
        plt.title(f"Capacities: Average = {caps.mean():.1f} StD = {caps.std():.1f}")

        # Setup Cost
        setup = np.array([f.setup_cost for f in facilities])
        plt.figure()
        plt.hist(setup, 100)
        plt.title(f"Setup Cost: Average = {setup.mean():.1f} StD = {setup.std():.1f}")

        # Demand
        plt.figure()
        plt.hist(demands, 100)
        plt.title(f"Demand: Average = {demands.mean():.1f} StD = {demands.std():.1f}")

        # Map
        view_problem(customers, facilities)