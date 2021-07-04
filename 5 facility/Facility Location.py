# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Setup

# %load_ext autoreload
# %autoreload 2

import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import join
from solver import Point, Facility, Customer, length, load_data
from vis import view_problem

path = 'data'
ls = os.listdir

# print(*ls(path),sep = '  ,  ')
for i,file in enumerate(ls(path)):
    customers, facilities = load_data(file)
    print(f'{file:15} {len(customers):6} Customers \t{len(facilities):6} Facilities')

customers, facilities = load_data('fl_100_7')
(customers, facilities)

# # Visualisation
#

from vis import view_problem, view_solution

view_problem(customers,facilities)

sol = greedy(customers,facilities)

view_solution(sol,customers,facilities)

# # Exploration

# +
from calc import validate, total_cost

from algorithms import greedy
from exceptions import *
# -

# ## Random allocation

from pdb import set_trace


# +
from tqdm.auto import tqdm, trange


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
            raise IterationError('Maximum number of iterations reached.')
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
            desc = '{:.1f} --> {:.1f} --> {:.1f}'.format(
                old_cost,
                costs[old_alloc],
                costs[new_alloc],
            )
            pbar.set_description(desc)
    return allocations


# -

sol = random_allocation(customers,facilities)
total_cost(sol,customers,facilities)

sol2 = local_search(sol,customers,facilities,True)

sol = greedy(customers,facilities)
total_cost(sol,customers,facilities)

sol2 = local_search(sol,customers,facilities,True)

view_solution(sol2,customers,facilities)

validate(sol2,customers,facilities)

# ## Constraint Programming

# ### Uncapacitated Facilities

from ortools.sat.python.cp_model import CpModel, CpSolver



