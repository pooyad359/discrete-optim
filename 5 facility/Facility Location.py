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
            ) for j in range(n_facilities))
        new_alloc = np.argmin(costs)
        allocations[i] = new_alloc
        desc = '{:.1f} --> {:.1f} --> {:.1f}'.format(
            old_cost,
            costs[old_alloc],
            costs[new_alloc],
        )
        pbar.set_description(desc)
    return allocations

def k_local_search(solution, customers, facilities, k = 5, njobs = 1):
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
        closest_facs = find_k_neighbors([facilities[current_fac]],facilities,k)[0]
        parallel = Parallel(n_jobs=njobs)
        delayed_func = delayed(eval_swap_values)
        costs = parallel(
            delayed_func(
                allocations=allocations,
                customers=customers,
                facilities=facilities,
                customer=i,
                new_facility=closest_facs[j],
            ) for j in range(k))
        new_alloc = closest_facs[np.argmin(costs)]
        allocations[i] = new_alloc
        desc = '{:.1f} --> {:.1f} --> {:.1f}'.format(
            old_cost,
            last_cost,
            min(costs),
        )
        pbar.set_description(desc)
    return allocations


def eval_swap_values(allocations, customers, facilities, customer,
                     new_facility):
    alloc = allocations.copy()
    alloc[customer] = new_facility
    return total_cost(alloc, customers, facilities)

def find_k_neighbors(x,y, k = 1):
    '''
    Finds k nearest neighbors for x in y. x and y are either Facilities or Customers,
    and must have location property.
    '''
    x = np.array([o.location for o in x])
    y = np.array([o.location for o in y])
    kdtree = KDTree(y)
    dist, neighbors = kdtree.query(x,k)
    return neighbors


# -

from algorithms import greedy
from calc import total_cost

customers, facilities = load_data('fl_1000_2')
len(customers),len(facilities)

# sol = random_allocation(customers,facilities)
sol = greedy(customers,facilities)
total_cost(sol,customers,facilities)

sol2 = k_local_search(sol,customers,facilities,12)

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
def est_facilities(customers, facilities):
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

est_facilities(customers,facilities)

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
    min_fac = est_facilities(customers,facilities)
    
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
from calc import distance_matrix, total_demand, est_facilities
from vis import view_problem, view_solution

from algorithms import uncap_mip_iter, clustering, cap_mip
from sklearn.neighbors import KDTree
from calc import diagnose, validate

customers, facilities = load_data('fl_25_2')
n_cust = len(customers)
n_fac = len(facilities)
n_cust,n_fac

total_demand(customers)

# +
# facilities=facilities[:620]
# -

view_problem(customers,facilities)

est_facilities(customers,facilities)

sol = uncap_mip(customers,facilities,120)

sol , status = sol

sol = uncap_mip_iter(customers,facilities)

view_solution(sol,customers,facilities)

diagnose(sol,customers,facilities)

sol2 = fix_unfeasibles(sol, customers, facilities, max_iter = 1000)

diagnose(sol2, customers, facilities)

view_solution(sol2,customers,facilities)

from algorithms import ex_local_search_v2
sol3 = ex_local_search_v2(sol2,customers,facilities)

view_solution(sol3,customers,facilities)

cap_mip(customers,facilities)

q = _

view_solution(q[0],customers,facilities)


def fix_unfeasibles(allocations, customers, facilities, max_iter = 100):
    allocations = np.array(allocations)
    loc_cust = np.array([c.location for c in customers])
    loc_fac = np.array([f.location for f in facilities])
    for i in range(max_iter):
        diag = diagnose(allocations, customers, facilities)
        idx = diag[:,0]
        spare = diag[:,3]
        f_worst = idx[spare.argmin()]
        f_best = idx[spare.argmax()]
        c_worst = np.where(allocations==f_worst)[0]
        kdtree = KDTree(loc_cust[c_worst,:])
        dist,target = kdtree.query([loc_fac[f_best,:]],1)
        allocations[c_worst[target]] = f_best
        try:
            validate(allocations, customers, facilities)
            break
        except AssertionError:
            continue
    else:
        print('Maximum iteration reached without finding a feasible solution.')
    return allocations


# ### Capacitated MIP

from algorithms import uncap_mip_iter, clustering, cap_mip
from solver import Facility, Customer
from calc import distance_matrix, min_facilities, est_facilities
from vis import view_problem, view_problem, problem_analysis

customers, facilities = load_data('fl_25_2')
est_facilities(customers,facilities)

min_facilities(customers,facilities)


# +
def cap_mip2(customers,
             facilities,
             max_time=60,
             min_fac=None,
             max_fac=None,
             k_neigh=None):
    n_fac = len(facilities)
    n_cust = len(customers)
    solver = Solver.CreateSolver("FacilityLocation", "SCIP")
    if min_fac is None:
        min_fac = min_facilities(customers, facilities)
    print(f'Minimum Facilities: {min_fac}')
    est_fac = est_facilities(customers, facilities)
    est_fac = max(5, min_fac)
    print('Estimated Facilities:', est_fac)
    if k_neigh is None:
        k_neigh = n_fac // est_fac * 2
        k_neigh = min(k_neigh, n_fac // 2)
    print(f'Only using {k_neigh} nearest facilities')

    # Estimate Customers per Facilitiy
    cpf = n_cust // min_fac
    cpf = np.clip(cpf, 2, n_cust // 2)

    # Define Variables
    x = []
    y = []
    for f in range(n_fac):
        y.append([solver.BoolVar(f"y_{f}_{c}") for c in range(n_cust)])
        x.append(solver.BoolVar(f"x_{f}"))

    caps = np.array([f.capacity for f in facilities])
    setup = np.array([f.setup_cost * 100 for f in facilities])
    dist = distance_matrix(customers, facilities) * 100
    dist = dist.astype(int)
    demands = np.array([c.demand for c in customers])

    # Problem Analysis
    free_setup = np.where(caps == 0)[0]
    is_fixed_setup = True if np.std(caps[caps > 0]) == 0 else False

    # Add Constraints
    print('\t Adding Constaints')
    # If facility is closed then it is not connected to any customer
    for f in range(n_fac):
        for c in range(n_cust):
            solver.Add(y[f][c] <= x[f])

    # Each customer is connected to only one facility
    for c in range(n_cust):
        solver.Add(sum([y[f][c] for f in range(n_fac)]) == 1)

    # The demand is not more than the capacity of the facility
    for f in range(n_fac):
        solver.Add(
            sum([y[f][c] * demands[c]
                 for c in range(n_cust)]) <= caps[f] * x[f])
        solver.Add(
            sum([y[f][c] * demands[c] for c in range(n_cust)]) <= caps[f])

    # Customers per facility
    for f in range(n_fac):
        solver.Add(sum([y[f][c] for c in range(n_cust)]) <= n_cust * x[f])

    # The free facilities must be open
    for f in free_setup:
        solver.Add(x[f] == 1)

    # Customer can ONLY connect to nearby facilities
    for c in range(n_cust):
        idx = np.argsort(dist[:, c])
        for f in idx[k_neigh:]:
            solver.Add(y[f][c] == 0)

    print('\t Adding Covercut Constaints')
    # Cover cut for customers which can be connected to a facility
    #     round1 = []
    #     round2 = []
    #     for f in range(n_fac):
    #         argsort = np.argsort(dist[f, :])
    #         idx = argsort[:cpf]
    #         if sum(demands[idx]) > caps[f]:
    #             round1.append(f)
    #             solver.Add(sum([y[f][c] for c in idx]) <= (cpf - 1))
    #         if cpf > 4:
    #             k = int(cpf * .9)
    #             idx = argsort[:k]
    #             if sum(demands[idx]) > caps[f]:
    #                 round2.append(f)
    #                 solver.Add(sum([y[f][c] for c in idx]) <= (k - 1))

    #     print(round1)
    #     print(round2, flush=True)

    # Maximum Facility open
    solver.Add(sum(x) >= min_fac)
    if max_fac is not None:
        solver.Add(sum(x) <= max_fac)
    #     solver.Add(sum(x)>=2)

    # Define objective
    obj = 0
    for f in range(n_fac):
        # Setup cost
        if not is_fixed_setup:
            obj += setup[f] * x[f]
        # Service cost
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

    # Solve
    print('\t Starting the Solver')
    status = solver.Solve()
    STATUS[status]

    # Retreive values
    a = []
    for f in range(n_fac):
        a.append([y[f][c].solution_value() for c in range(n_cust)])

    # Convert solution matrix to facility index
    sol = np.array(a).argmax(axis=0)

    return sol, STATUS[status]


def find_max_customers(capacity, demands):
    if capacity >= np.sum(demands):
        return len(demands)
    sorted_demands = np.array(sorted(demands))
    cumsum_demands = sorted_demands.cumsum()
    loc = np.searchsorted(cumsum_demands > capacity, .5)
    return loc - 1


# -

customers, facilities = load_data('fl_25_2')
n_cust = len(customers)
n_fac = len(facilities)
n_cust,n_fac

problem_analysis(customers,facilities,False)

est_facilities(customers,facilities)

sol, status = cap_mip2(customers[:20],facilities,120,k_neigh=20,max_fac=20)
print(status)

view_solution(sol,customers,facilities)

diagnose(sol,customers,facilities)

# ## KNN Search

from calc import distance_matrix, total_demand, est_facilities
from vis import view_problem, view_solution
from algorithms import uncap_mip_iter, clustering
from sklearn.neighbors import KDTree
from calc import diagnose, validate

customers, facilities = load_data('fl_2000_2')
n_cust = len(customers)
n_fac = len(facilities)
n_cust,n_fac

total_demand(customers)

est_facilities(customers, facilities)

view_problem(customers,facilities)

sol = find_k_neighbors(customers,facilities,1)

sol = sol[:,0]

view_solution(sol,customers,facilities)

len(set(sol.flatten()))

sol2 = k_local_search(sol,customers,facilities)

view_solution(sol2,customers,facilities)



problem_analysis(customers,facilities)

# ## Greedy Furthest Point

from calc import distance_matrix, total_demand, est_facilities,total_cost
from vis import view_problem, view_solution
from algorithms import uncap_mip_iter, clustering, greedy
from sklearn.neighbors import KDTree
from calc import diagnose, validate
from algorithms import greedy_furthest, double_trial
from tqdm.auto import tqdm

customers, facilities = load_data('fl_200_7')
n_cust = len(customers)
n_fac = len(facilities)
n_cust,n_fac

total_demand(customers)

est_facilities(customers, facilities)

dist = distance_matrix(customers,facilities)

n_fac, n_cust

dist.shape

solution = greedy_furthest(customers,facilities)

#1  fl_25_2
#2  fl_50_6
#3  fl_100_7
#4  fl_100_1
#5  fl_200_7
#6  fl_500_7
#7  fl_1000_2
#8  fl_2000_2
customers, facilities = load_data('fl_500_7')
n_cust = len(customers)
n_fac = len(facilities)
print(n_cust,n_fac)
solution = double_trial(customers,facilities,greedy_furthest,p_skip=0.2)
total_cost(solution,customers,facilities)

view_solution(solution,customers,facilities)

solution = greedy(customers,facilities)
total_cost(solution,customers,facilities)

solution = clustering(customers,facilities)
total_cost(solution,customers,facilities)

# ## Greedy Furthest on Subset

from calc import distance_matrix, total_demand, est_facilities,total_cost
from vis import view_problem, view_solution
from algorithms import uncap_mip_iter, clustering
from sklearn.neighbors import KDTree
from calc import diagnose, validate
from algorithms import greedy_furthest, double_trial
from tqdm.auto import tqdm, trange

# +
#1  fl_25_2
#2  fl_50_6
#3  fl_100_7
#4  fl_100_1
#5  fl_200_7
#6  fl_500_7
#7  fl_1000_2
#8  fl_2000_2
# -

customers, facilities = load_data('fl_100_1')
n_cust = len(customers)
n_fac = len(facilities)
print(n_cust,n_fac)

solution = greedy_furthest(customers,facilities,p_skip=0,pbar=False)
total_cost(solution,customers,facilities)

solution = double_trial(customers,facilities,greedy_furthest,p_skip=0,pbar=False)
total_cost(solution,customers,facilities)

est_facs = est_facilities(customers,facilities)
print(est_facs)
n_trials = 10000
weights = np.ones(n_fac)
q = 2e7
evap = .02
offset = 2e7

costs = []
for i in trange(n_trials):
    probs = weights/weights.sum()
    idx = np.random.choice(range(n_fac),size=est_facs,replace=False,p=probs)
    selection = [facilities[f] for f in idx ]
    solution = greedy_furthest(customers,selection,pbar=False)
    solution = [idx[f] for f in solution]
    new_weights = np.array([1 if f in solution else 0 for f in range(n_fac)])
    cost = total_cost(solution, customers,facilities)
    costs.append(cost)
    weights = weights*(1-evap) + q*new_weights/(cost-offset)

plt.plot(weights)

plt.plot(costs)

total_cost(solution, customers,facilities)

# ## Ant Colony

from collections import defaultdict


# +
def ant_simulator(customers, facilities, distances, probabilities):
    n_cust = len(customers)
    n_fac = len(facilities)
    solution = -np.ones(n_cust)
    opened_fac = np.zeros(n_fac)
    remaining_cap = np.array([f.capacity for f in facilities])
    setup_cost = np.array([f.setup_cost for f in facilities])
    dist_ord = distances.mean(axis=0).argsort()
    for c in reversed(dist_ord):
        customer = customers[c]
        choice_prob = probabilities[:, c]
        choice_prob[remaining_cap < customer.demand] = 0
        choice_prob /= choice_prob.sum()
        f = np.random.choice(range(n_fac), p=choice_prob)
        solution[c] = f
    return solution.astype(int)


def ant_colony(
    customers,
    facilities,
    q=1,
    offset=0,
    evaporation=.1,
    ants=100,
    generations=100,
):
    n_cust = len(customers)
    n_fac = len(facilities)
    dist = distance_matrix(customers, facilities)
    weights = np.ones_like(dist)
    best_sol = None
    best_cost = np.inf
    metrics = defaultdict(list)
    greedy_solution = double_trial(
        customers,
        facilities,
        greedy_furthest,
        p_skip=0,
        pbar=False,
    )
    for gen in trange(generations):
        probs = weights/dist
        updates = np.zeros_like(dist)
        costs = []
        for ant in range(ants):
            if ant==0:
                solution = greedy_solution.copy()
            else:
                solution = ant_simulator(customers, facilities, dist, probs)
            cost = total_cost(solution,customers, facilities)
            edges = allocation2matrix(solution, n_fac)
            updates += edges*q/(cost-offset)
            costs.append(cost)
            if cost<best_cost:
                best_cost = cost
                best_sol = solution
                
        weights = weights*(1-evaporation) + updates
        metrics['min_cost'].append(np.min(costs))
        metrics['max_cost'].append(np.max(costs))
        metrics['mean_cost'].append(np.mean(costs))
    return best_sol, metrics

def allocation2matrix(allocations, n_facilities):
    matrix = np.eye(n_facilities)
    return matrix[np.array(allocations)].T


# -

customers, facilities = load_data('fl_200_7')
n_cust = len(customers)
n_fac = len(facilities)
print(n_cust,n_fac)

solution = double_trial(
    customers,
    facilities,
    greedy_furthest,
    p_skip=0,
    pbar=False,
)
cost = total_cost(solution, customers, facilities)
q = cost/10
offset = cost * .75
s, m = ant_colony(
    customers,
    facilities,
    q=q,
    offset=offset,
    evaporation=.05,
    generations=200,
)

total_cost(s,customers,facilities)

plt.figure(figsize=(10, 6))
n = len(m['min_cost'])
plt.fill_between(range(n), m["min_cost"], m["max_cost"], alpha=0.2)
plt.plot(range(n), m["mean_cost"])

view_solution(s,customers,facilities)

print(f'{total_cost(s,customers,facilities):.2f}', 0)
print(' '.join([str(o) for o in s]))

dist.shape

solution

dist = distance_matrix(customers,facilities)

# ## Greedy with restarts

from calc import distance_matrix, total_demand, est_facilities,total_cost
from vis import view_problem, view_solution
from algorithms import uncap_mip_iter, clustering
from sklearn.neighbors import KDTree
from calc import diagnose, validate
from algorithms import greedy_furthest, double_trial
from tqdm.auto import tqdm, trange

# +
#1  fl_25_2
#2  fl_50_6
#3  fl_100_7
#4  fl_100_1
#5  fl_200_7
#6  fl_500_7
#7  fl_1000_2
#8  fl_2000_2
# -

customers, facilities = load_data('fl_200_7')
n_cust = len(customers)
n_fac = len(facilities)
print(n_cust,n_fac)


# +
def iter_greedy(customers,facilities,branches = 5,random_state=None):
    np.random.seed(random_state)
    solution = greedy_furthest(customers,facilities,pbar=False)
    cost = total_cost(solution, customers, facilities)
    for _ in range(len(facilities)):
        idx = np.random.choice(np.unique(solution),5)
#         new_facilities = [f for f in facilities if f.index!=dropped_facility]
        old_facilities = facilities.copy()
        new_sols = []
        new_costs = []
        for i in idx:
            facilities = old_facilities.copy()
            dropped_facility = facilities.pop(i)
            new_sols.append( greedy_furthest(customers,facilities,pbar=False))
            new_costs.append( total_cost(new_sols[-1], customers, facilities))
        if min(new_costs)<cost:
            i = np.argmin(new_costs)
            solution = new_sols[i]
            cost = new_costs[i]
        else:
            return [old_facilities[f].index for f in solution]
            
            
def greedy_restart(customers,facilities,restarts=100,branches = 5,random_state=None):
    best_sol = []
    best_cost = np.inf
    pbar = trange(restarts)
    for _ in pbar:
        pbar.set_description(f'{best_cost:.1f}')
        solution = iter_greedy(customers,facilities,branches,random_state)
        cost = total_cost(solution,customers,facilities)
        if cost<best_cost:
            best_cost = cost
            best_sol = solution
    return best_sol


# -

sol = iter_greedy(customers,facilities,20)

solution = greedy_restart(customers,facilities,100,10)

total_cost(_,customers,facilities)

problem_analysis(customers,facilities)

# ## Local Search
#

from algorithms import ex_local_search, k_local_search, greedy_furthest,double_trial

# +
#1  fl_25_2
#2  fl_50_6
#3  fl_100_7
#4  fl_100_1
#5  fl_200_7
#6  fl_500_7
#7  fl_1000_2
#8  fl_2000_2
# -

customers, facilities = load_data('fl_500_7')
n_cust = len(customers)
n_fac = len(facilities)
print(n_cust,n_fac)

from random import shuffle

costs = []
order = np.arange(n_cust)
np.random.shuffle(order)
for i in trange(1000):
    
    solution = greedy_furthest(
        customers,
        facilities,
        ignore_setup=True,
        pbar=False,
        order=order,
    )
    cost = total_cost(solution, customers, facilities)
    costs.append(cost)
    
    idx = np.argsort(solution)
    groups = defaultdict(list)
    for c, f in enumerate(solution):
        groups[f].append(c)
    groups[-1] = [f for f in range(len(facilities)) if f not in solution]
    for f in groups.keys():
        shuffle(groups[f])
    groups = list(groups.values())
    shuffle(groups)
    order = sum(groups,start=[])
print(min(costs))
plt.hist(costs,bins=20);

solution  = greedy_furthest(customers,facilities,ignore_setup=False,pbar=False)
cost = total_cost(solution,customers,facilities)
print(cost)

solution  = double_trial(customers,facilities,greedy_furthest,ignore_setup=False,pbar=False)
cost = total_cost(solution,customers,facilities)
print(cost)

solution  = double_trial(customers,facilities,greedy_furthest,ignore_setup=True,pbar=False)
cost = total_cost(solution,customers,facilities)
print(cost)

for i in range(10):
    sol2 = k_local_search(solution,customers,facilities,k=10)
    cost2 = total_cost(sol2,customers,facilities)
    if cost2==cost:
        print('No improvement was observed')
        break
    else:
        solution = sol2
        cost = cost2

# # MIP Gurobi

from itertools import product
from math import sqrt
from calc import diagnose, total_cost
import gurobipy as gp
from gurobipy import GRB
from solver import load_data



def cap_mip_gr(customers,facilities,max_time = 60):
    n_cust = len(customers)
    n_fac = len(facilities)
    dist = distance_matrix(customers, facilities)
    cartesian_prod = list(product(range(n_cust), range(n_fac)))
    setup_cost = [f.setup_cost for f in facilities]
    shipping_cost = {(c,f):dist[f,c] for c,f in cartesian_prod}
    demands = np.array([c.demand for c in customers])
    caps = np.array([f.capacity for f in facilities])
    solver = gp.Model('facility_location')
    
    # Define Variables
    x = solver.addVars(n_fac,vtype=GRB.BINARY, name='Select')
    y = solver.addVars(cartesian_prod, ub=1, vtype=GRB.CONTINUOUS, name='Assign')
    
    # Define Constraints
    solver.addConstrs((y[(c,f)] <= x[f] for c,f in cartesian_prod), name='Setup2ship')
    solver.addConstrs((gp.quicksum(y[(c,f)] for f in range(n_fac)) == 1 for c in range(n_cust)), name='Demand')
    solver.addConstrs((gp.quicksum(y[(c,f)]*demands[c] for c in range(n_cust)) <= caps[f] for f in range(n_fac)),name = 'Capacity')
    # Set Objective
    solver.setObjective(x.prod(setup_cost)+y.prod(shipping_cost), GRB.MINIMIZE)
    solver.setParam("TimeLimit",max_time)
    
    # Solve
    solver.optimize()
    # Parse Outputs
    solution = np.zeros((n_fac,n_cust))
    for c in range(n_cust):
        for f in range(n_fac):
            solution[f,c] = y[(c,f)].x
            
    return solution.argmax(axis=0)


# +
#1  fl_25_2
#2  fl_50_6
#3  fl_100_7
#4  fl_100_1
#5  fl_200_7
#6  fl_500_7
#7  fl_1000_2
#8  fl_2000_2
# -

customers, facilities = load_data('fl_200_7')
n_cust = len(customers)
n_fac = len(facilities)
print(n_cust,n_fac)

solution = cap_mip_gr(customers,facilities,120)

total_cost(solution, customers, facilities)

view_solution(solution,customers,facilities)

from algorithms import fix_allocations
fix_allocations(solution, customers,facilities)

# ## Divide and Conquer

# +
#1  fl_25_2
#2  fl_50_6
#3  fl_100_7
#4  fl_100_1
#5  fl_200_7
#6  fl_500_7
#7  fl_1000_2
#8  fl_2000_2
# -

customers, facilities = load_data('fl_100_1')
n_cust = len(customers)
n_fac = len(facilities)
print(n_cust,n_fac)
view_problem(customers,facilities)

solution = cap_mip_gr(customers[:100],facilities,120)

view_solution(solution,customers[:100],facilities)


# +
def divide_conquer(customers,facilities,batch_size = 100):
    new_facilities = facilities.copy()
    solution = []
    for i in range(0,len(customers),batch_size):
        batch = customers[i:(i+100)]
        batch_sol = cap_mip_gr(batch,new_facilities,120)
        solution += [new_facilities[f].index for f in batch_sol]
        demands = [(f,batch[c].demand) for c,f in enumerate(batch_sol)]
        new_facilities = update_facilities(new_facilities,demands)
    return solution    

def update_facilities(facilities, demands):
    pass


# -

import itertools

# ## Piecewise Solver

from gurobi_solver import cap_mip_gr


# +
def get_grid_size(n_facilities, per_grid=100):
    for n in range(1, 10):
        if n_facilities / n**2 <= per_grid:
            return n
    else:
        return None


def is_inside(point, xrange, yrange):
    return xrange[0] < point.x < xrange[1] and yrange[0] < point.y < yrange[1]


def piecewise_mip(customers, facilities, per_grid=100, max_time = 60):
    n_cust = len(customers)
    n_fac = len(facilities)
    n = get_grid_size(n_fac,per_grid)
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
            print(f'*** Row#{i+1}/{n} Column#{j+1}/{n} ***')
            yl = yrange[j]
            yu = yrange[j+1]

            cgrid = [
                c for c in customers
                if is_inside(c.location, (xl, xu), (yl, yu))
            ]
            fgrid = [
                f for f in facilities
                if is_inside(f.location, (xl, xu), (yl, yu))
            ]

            solgrid = cap_mip_gr(cgrid, fgrid)
            try:
                validate(solgrid,cgrid,fgrid)
            except AssertionError:
                print('***************************')
                print('***  INVALID SOLUTION   ***')
                print('***************************')
            allocations += [(cgrid[c].index, fgrid[f].index)
                         for c, f in enumerate(solgrid)]
    solution = -np.ones(n_cust)
    for c,f in allocations:
        solution[c] = f
    return np.int32(solution)


# +
#1  fl_25_2
#2  fl_50_6
#3  fl_100_7
#4  fl_100_1
#5  fl_200_7
#6  fl_500_7
#7  fl_1000_2
#8  fl_2000_2
# -

customers, facilities = load_data('fl_500_7')
n_cust = len(customers)
n_fac = len(facilities)
print(n_cust,n_fac)
view_problem(customers,facilities)

sol = piecewise_mip(customers, facilities, per_grid=80)

view_solution(sol,customers,facilities)






