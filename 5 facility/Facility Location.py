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

customers, facilities = load_data('fl_50_1')
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
from joblib import Parallel, delayed


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
            desc = '{:.1f} --> {:.1f} --> {:.1f}'.format(
                old_cost,
                costs[old_alloc],
                costs[new_alloc],
            )
            pbar.set_description(desc)
    return allocations


def ex_local_search_v2(solution, customers, facilities, verbose=False):
    allocations = solution.copy()
    n_cutomers = len(customers)
    n_facilities = len(facilities)
    old_cost = total_cost(allocations, customers, facilities)
    pbar = trange(n_cutomers)
    for i in pbar:
        customer = customers[i]
        costs = np.zeros(n_facilities)
        old_alloc = allocations[i]

        parallel = Parallel(n_jobs=-1)
        delayed_func = delayed(eval_swap_values)
        costs = parallel(
            delayed_func(
                allocations=allocations,
                customers=customers,
                facilities=facilities,
                customer=i,
                new_facility=j,
            ) for j in range(n_facilities))
#         for j in range(n_facilities):
#             allocations[i] = j
#             costs[j] = total_cost(allocations, customers, facilities)
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


def eval_swap_values(allocations, customers, facilities, customer,
                     new_facility):
    alloc = allocations.copy()
    alloc[customer] = new_facility
    return total_cost(alloc, customers, facilities)


# -

from algorithms import greedy

customers, facilities = load_data('fl_100_3')
len(customers),len(facilities)

# sol = random_allocation(customers,facilities)
sol = greedy(customers,facilities)
total_cost(sol,customers,facilities)

sol2 = ex_local_search_v2(sol,customers,facilities,True)

# %debug

sol = greedy(customers,facilities)
total_cost(sol,customers,facilities)

sol2 = local_search(sol,customers,facilities,True)

view_solution(sol2,customers,facilities)

validate(sol2,customers,facilities)

# ## Constraint Programming

# ### Uncapacitated Facilities

from ortools.sat.python import cp_model
from calc import distance_matrix


customers,facilities= load_data('fl_100_1')
n_cust = len(customers)
n_fac = len(facilities)


caps = [f.capacity for f in facilities]
setup = [f.setup_cost for f in facilities]
dist = distance_matrix(customers,facilities).astype(int)
demands = [c.demand for c in customers]

model = cp_model.CpModel()

a = [] # allocation matrix (facilities x customers)
for f in range(n_fac):
    a.append([model.NewBoolVar(f'a_{c}_{f}') for c in range(n_cust)])

# +
# Only one facility per customer
for c in range(n_cust):
    model.Add(sum([a[f][c] for f in range(n_fac)])==1)
    
# Capacity check
for f in range(n_fac):
    model.Add(sum([a[f][c]*demands[c] for c in range(n_cust)])<=caps[f])
# -

obj = 0
for f in range(n_fac):
    for c in range(n_cust):
        obj+=a[f][c]*dist[f,c]

model.Minimize(obj)

cpsolver = cp_model.CpSolver()
cpsolver.parameters.max_time_in_seconds = 60.0
status = cpsolver.Solve(model)

STATUS = {
    cp_model.FEASIBLE: 'FEASIBLE',
    cp_model.UNKNOWN: 'UNKNOWN',
    cp_model.MODEL_INVALID: 'MODEL_INVALID',
    cp_model.INFEASIBLE: 'INFEASIBLE',
    cp_model.OPTIMAL: 'OPTIMAL',
}
STATUS[status]

# +
values = [] # allocation matrix (facilities x customers)
for f in range(n_fac):
    values.append([cpsolver.Value(a[f][c]) for c in range(n_cust)])

values = np.array(values)

sol = values.argmax(axis=0)
# -

total_cost(sol,customers,facilities)

view_solution(sol,customers,facilities)

# ## Solve using clustering
# Ignores the facility capital cost

import math
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
from calc import total_demand, total_cost
from tqdm.auto import trange

q = [2,3,5,3,1]

np.cumsum(sorted(q))


# +
def min_facilities(customers, facilities):
    caps = [f.capacity for f in facilities]
    cumsum = np.cumsum(sorted(caps))
    demands = [c.demand for c in customers]
    total_demand = sum(demands)
    for i in range(len(facilities)):
        if cumsum[i]>=total_demand:
            print(f'Demand = {total_demand}, Capacity = {cumsum[i]}')
            return i+1
    
#     return math.ceil(total_demand/max(caps))


# -

min_facilities(customers,facilities)

customers, facilities = load_data('fl_1000_2')
n_cust = len(customers)
n_fac = len(facilities)
n_cust,n_fac

[f.setup_cost for f in facilities]


# +
# total_demand(customers)

# +
# [f.capacity for f in facilities]
# -

def clustering(customers, facilities):
    n_cust = len(customers)
    n_fac = len(facilities)
    
    # Find minimum number of clusters
    min_fac = min_facilities(customers,facilities)
    
    # Find cluster centroids
    kmean = KMeans(min_fac)
    xy = np.array([c.location for c in customers])
    kmean.fit(xy)
    centroids = kmean.cluster_centers_
    
    # Find the distance between facilities and centroids
    fneigh = KDTree(np.array([f.location for f in facilities]))
    dist, locs = fneigh.query(centroids,min_fac)
    
    # Match facilities with centroids
    inactive_facs = set(range(n_fac))
    active_facs = []
    for i in range(min_fac):
        for j in range(min_fac):
            if locs[i,j] in inactive_facs:
                inactive_facs = inactive_facs - {locs[i,j]}
                active_facs.append(locs[i,j])
                break
    
    # Assign facilities to customers
    turn = 0
    remaining_cap = [facilities[f].capacity for f in active_facs]
    remaining_cust = set(range(n_cust))
    max_iter = len(active_facs)*n_cust
    sol = -np.ones(n_cust,dtype=int)
    pbar = trange(n_cust)
    for _ in range(max_iter):
        turn +=1
        turn %=len(active_facs)
        turn_fac = active_facs[turn]
        kd_cust = KDTree(np.array([customers[c].location for c in list(remaining_cust)]))
        dist, pot_cust = kd_cust.query([facilities[turn_fac].location],1)
        pot_cust = list(remaining_cust)[pot_cust[0][0]]
        if remaining_cap[turn]<customers[pot_cust].demand:
            continue
        else:
            pbar.update()
            sol[pot_cust] = turn_fac
            remaining_cust = remaining_cust - {pot_cust}
            remaining_cap[turn]-=customers[pot_cust].demand
        if min(sol)>=0:
            break
    else:
        IterationError('Maximum number of iteration reached.')
    return sol


sol = clustering(customers,facilities)

sol

view_solution(sol,customers,facilities,(16,16))

# ## MIP

from ortools.linear_solver.pywraplp import Solver
from calc import distance_matrix,total_demand

customers, facilities = load_data('fl_100_7')
n_cust = len(customers)
n_fac = len(facilities)
n_cust,n_fac

total_demand(customers)

min_facilities(customers,facilities)

# ### Define model and variables

solver = Solver.CreateSolver('FacilityLocation','SCIP')

x = []
y = []
for f in range(n_fac):
    y.append([solver.BoolVar(f'y_{f}_{c}') for c in range(n_cust)])
    x.append(solver.BoolVar(f'x_{f}'))

# ### Constraints

facilities = facilities[:7]

view_problem(customers,facilities,(16,16))

caps = [f.capacity for f in facilities]
setup = [f.setup_cost for f in facilities]
dist = distance_matrix(customers,facilities).astype(int)
demands = [c.demand for c in customers]

n_fac=len(facilities)

# +
for f in range(n_fac):
    for c in range(n_cust):
        solver.Add(y[f][c]<=x[f])
for c in range(n_cust):
    solver.Add(sum([y[f][c] for f in range(n_fac)])==1)
    
for f in range(n_fac):
    solver.Add(sum([y[f][c]*demands[c] for c in range(n_cust)])<=caps[f])
# -

obj = 0
for f in range(n_fac):
    obj += setup[f]*x[f]
    obj += sum([dist[f][c]*y[f][c] for c in range(n_cust)])

solver.Minimize(obj)

# +

STATUS = {
    Solver.FEASIBLE: 'FEASIBLE',
    Solver.UNBOUNDED: 'UNBOUNDED',
    Solver.BASIC: 'BASIC',
    Solver.INFEASIBLE: 'INFEASIBLE',
    Solver.NOT_SOLVED: 'NOT_SOLVED',
    Solver.OPTIMAL: 'OPTIMAL',
}
solver.SetTimeLimit(120000)
# -

status = solver.Solve()
STATUS[status]

a = []
for f in range(n_fac):
    a.append([y[f][c].solution_value() for c in range(n_cust)])

sol = np.array(a).argmax(axis=0)

view_solution(sol,customers,facilities)

from calc import validate
validate(sol,customers,facilities)


