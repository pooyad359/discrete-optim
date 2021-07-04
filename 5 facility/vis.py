import matplotlib.pyplot as plt
import numpy as np
from calc import total_cost


def view_problem(customers, facilities, figsize=(8, 8)):
    loc_cust = np.array([c.location for c in customers])
    loc_fac = np.array([f.location for f in facilities])
    plt.figure(figsize=figsize)
    plt.scatter(loc_cust[:, 0], loc_cust[:, 1], marker="o", c="k", s=5)
    plt.scatter(loc_fac[:, 0], loc_fac[:, 1], marker="*", c="r", s=5)

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
    plt.title(
        f"{len(customers)} Customers    {len(facilities)} Facilities   Total Cost = {cost:.1f}"
    )
    for c, f in enumerate(solution):
        x = [facilities[f].location.x, customers[c].location.x]
        y = [facilities[f].location.y, customers[c].location.y]
        plt.plot(x, y, "-k", alpha=0.5)
