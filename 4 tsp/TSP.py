# -*- coding: utf-8 -*-
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

# +
import numpy as np
import matplotlib.pyplot as plt
import os
from functools import lru_cache, partial
import math
from collections import namedtuple
from tqdm.auto import tqdm, trange
Point = namedtuple("Point", ['x', 'y'])

ls = os.listdir
join = os.path.join


# +
@lru_cache(256)
def distance(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def view_tsp(solution, points, figsize = (8,8)):
    '''
    List of points
    '''
    plt.figure(figsize=figsize)
    xy = [[points[i].x, points[i].y] for i in solution]
    xy = np.array(xy)
    x = xy[:,0]
    y = xy[:,1]
    plt.plot(x,y,'ok-')
    plt.plot(x[0],y[0],'sr',markersize=12)
    plt.plot(x[-1],y[-1],'*g',markersize=16)
    for i, (xi,yi) in enumerate(xy):
        plt.text(xi+.01,yi+.01,i,fontdict={'fontsize':16,'color':'darkblue'})
        
def to_nearest(points):
    path = []
    remaining = list(range(len(points)))
    path.append(remaining.pop(0))
    while len(remaining)>0:
        dists = [distance(points[i],points[path[-1]]) for i in remaining]
        next_i = np.argmin(dists)
        path.append(remaining.pop(next_i))
        
    return path
            
def loss(sol,points):
    dist = distance(points[sol[-1]],points[sol[0]])
    for i in range(1,len(sol)):
        dist += distance(points[sol[i-1]],points[sol[i]])
        
    return dist


# -

ls('./data')

with open(join('./data/','tsp_70_1')) as fp:
    content = fp.read().split('\n')


nodeCount = int(content[0])
points = []
for i in range(1, nodeCount+1):
    line = content[i]
    parts = line.split()
    points.append(Point(float(parts[0]), float(parts[1])))

sol=to_nearest(points)
print(sol)
print(loss(sol,points))
view_tsp(sol)

with open('./prob_6_nearest.sol') as fp:
    c = fp.read()

c

# # SK OPT

with open(join('./data/','tsp_70_1')) as fp:
    content = fp.read().split('\n')


# +
node_count = int(content[0])
points = []
for i in range(1, nodeCount + 1):
    line = content[i]
    parts = line.split()
    points.append(Point(float(parts[0]), float(parts[1])))

print(f'#Nodes: {node_count}')
# -

evaluate = partial(loss,points=points)

from sko import GA

optim = GA.GA_TSP(
    evaluate,
    node_count,
    size_pop=200,
    max_iter=1000,
)

solution, value = optim.run()

view_tsp(solution)

# + [markdown] heading_collapsed=true
# # 2-opt

# + hidden=true
with open(join('./data/','tsp_76_1')) as fp:
    content = fp.read().split('\n')


# + hidden=true
node_count = int(content[0])
points = []
for i in range(1, nodeCount + 1):
    line = content[i]
    parts = line.split()
    points.append(Point(float(parts[0]), float(parts[1])))

print(f'#Nodes: {node_count}')

# + hidden=true
evaluate = partial(loss,points=points)


# + hidden=true
def cost_change(n1, n2, n3, n4):
    return (distance(n1, n3) + 
            distance(n2, n4) - 
            distance(n1, n2) -
            distance(n4, n3))


def two_opt(route,points,max_iter = 1000):
    best = route.copy()
    improved = True
    counter = 0
#     pbar = trange(max_iter)
    while improved and counter<max_iter:
        counter += 1
        value = evaluate(route)
        print(f'#{counter:04.0f} \t{value}',flush = True)
        improved = False
        for i in range(1, len(route) - 2):
                
            for j in range(i + 1, len(route)):
                if j - i == 1: continue
                if cost_change(
                        points[best[i - 1]],
                        points[best[i]],
                        points[best[j - 1]],
                        points[best[j]],
                ) < 0:
                    best[i:j] = best[j - 1:i - 1:-1]
                    improved = True
        route = best
    return best


# + hidden=true
# Initial solution using greedy
sol=to_nearest(points)
evaluate(sol)

# + hidden=true
from collections import deque

# + hidden=true
sol2 = two_opt(sol)

# + hidden=true
view_tsp(sol2)
# -

# # CP

from solver import view_tsp

with open(join('./data/','tsp_70_1')) as fp:
    content = fp.read().split('\n')


# +
node_count = int(content[0])
points = []
for i in range(1, node_count + 1):
    line = content[i]
    parts = line.split()
    points.append(Point(float(parts[0]), float(parts[1])))

print(f'#Nodes: {node_count}')
# -

from ortools.sat.python import cp_model

model = cp_model.CpModel()

a = [] # whether i and j are connected
u = [] # order of oints
for i in range(node_count):
    row = [model.NewBoolVar(f'a_{i}_{j}') for j in range(node_count)]
    a.append(row)
# u = [model.NewIntVar(1,node_count-1,f'u_{i}') for i in range(node_count)]

# +
# Add Constraints
for i in range(node_count):
    model.Add(a[i][i] == 0)
    model.Add(sum([ a[i][j] for j in range(node_count)])==1)
    model.Add(sum([ a[j][i] for j in range(node_count)])==1)
    
# model.AddAllDifferent(u)

# for i in range(1,node_count):
#     for j in range(1,node_count):
#         if i==j: continue
#         model.Add(u[i]-u[j]+node_count*a[i][j]<=node_count-1)
# model.Add(u[0]==0)
# model.Add(u[-1]==node_count)
# -

# Objective
obj = []
for i in range(node_count):
    for j in range(node_count):
        d = distance(points[i],points[j])*100
        obj.append(int(d)*a[i][j])
model.Minimize(sum(obj))

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
sol_mat = []
for i in range(node_count):
    row = [cpsolver.Value(a[i][j]) for j in range(node_count)]
    sol_mat.append(row)
    print(row)
    
sol_mat = np.array(sol_mat)

# -

np.argmax(sol_mat,axis=0)


def view_matrix(mat,points,figsize = (8,8)):
    lines = []
#     row = np.argmax(sol_mat,axis=0)
    row = np.arange(len(points))
    col = np.argmax(sol_mat,axis=1)
    plt.figure(figsize=figsize)
    for i,j in zip(row,col):
        x = [points[i].x,points[j].x]
        y = [points[i].y,points[j].y]
        plt.plot(x,y,'ok-')


view_matrix(sol_mat,points)

plt.figure(figsize=(6, 6))
plt.scatter(
    x=[p.x for p in points],
    y=[p.y for p in points],
)

# # Randomized Opt-2

np.random.permutation(12)


def randomized_opt2(points,restarts = 1):
    if restarts>1:
        solutions = []
        values = []
        for i in range(restarts):
            print(f'<<< Attempt {i+1}/{restarts}>>>')
            solution = randomized_opt2(points)
            value = loss(solution,points)
            solutions.append(solution)
            values.append(value)
        idx = np.argmin(values)
        return solutions[idx]
    else:
        trial1 = np.random.permutation(len(points))
        np.random.shuffle(trial1)
        solution = two_opt(trial1,points)
        trial2 = deque(solution)
        trial2.rotate(3)
        return two_opt(list(trial2),points)


res = randomized_opt2(points,10)

view_tsp(res)

# # OR-tool routing solver

import math
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


# +
def create_data_model(points):
    """Stores the data for the problem."""
    data = {}
    data['locations'] = points
    data['num_vehicles'] = 1
    data['depot'] = 0
    return data


def compute_euclidean_distance_matrix(locations):
    """Creates callback to return distance between points."""
    distances = {}
    for from_counter, from_node in enumerate(locations):
        distances[from_counter] = {}
        for to_counter, to_node in enumerate(locations):
            if from_counter == to_counter:
                distances[from_counter][to_counter] = 0
            else:
                # Euclidean distance
                distances[from_counter][to_counter] = (int(
                    math.hypot((from_node[0] - to_node[0]),
                               (from_node[1] - to_node[1]))))
    return distances


def print_solution(manager, routing, solution):
    """Prints solution on console."""
    print('Objective: {}'.format(solution.ObjectiveValue()))
    index = routing.Start(0)
    plan_output = 'Route:\n'
    route_distance = 0
    while not routing.IsEnd(index):
        plan_output += ' {} ->'.format(manager.IndexToNode(index))
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    plan_output += ' {}\n'.format(manager.IndexToNode(index))
    print(plan_output)
    plan_output += 'Objective: {}m\n'.format(route_distance)
    return plan_output



# +
def get_route(manager, routing, solution):
    route=[]
    route =[ routing.Start(0)]
    while not routing.IsEnd(route[-1]):
        index = route[-1]
        new_index = solution.Value(routing.NextVar(index))
        route.append(new_index)
    route.pop(-1)
    return route

def main(points = None):
    """Entry point of the program."""
    # Instantiate the data problem.
    data = create_data_model(locations = points)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['locations']),
                                           data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    distance_matrix = compute_euclidean_distance_matrix(data['locations'])

    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
#         out = print_solution(manager, routing, solution)
        out = get_route(manager, routing, solution)
        
    return out
        


# -

o = main(points)

o

loss(o,points)

with open(join('./data/','tsp_70_1')) as fp:
    content = fp.read().split('\n')


# +
node_count = int(content[0])
points = []
for i in range(1, node_count + 1):
    line = content[i]
    parts = line.split()
    points.append(Point(float(parts[0]), float(parts[1])))

print(f'#Nodes: {node_count}')

# +
manager = pywrapcp.RoutingIndexManager(node_count, 1, 0)
routing = pywrapcp.RoutingModel(manager)
distance_callback = lambda i,j: distance(points[i],points[j])
transit_callback_index = routing.RegisterTransitCallback(distance_callback)
routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

search_parameters = pywrapcp.DefaultRoutingSearchParameters()
# search_parameters.first_solution_strategy = (
#     routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
search_parameters.local_search_metaheuristic = (
    routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
search_parameters.time_limit.seconds = 30
search_parameters.log_search = True

solution = routing.SolveWithParameters(search_parameters)
# -

if solution:
    out=print_solution(manager, routing, solution)



route=get_route(manager, routing, solution)

route

# # MILP

# ## Miller–Tucker–Zemlin formulation

import os
from ortools.linear_solver import pywraplp 

print(*ls('./data'),sep='\t')

with open('./data/tsp_51_1') as fp:
    content = fp.read().split('\n')


# +
node_count = int(content[0])
points = []
for i in range(1, node_count + 1):
    line = content[i]
    parts = line.split()
    points.append(Point(float(parts[0]), float(parts[1])))

print(f'#Nodes: {node_count}')

d = []
for i in range(node_count):
    row =[distance(points[i],points[j]) for j in range(node_count)]
    d.append(row)
    
np.shape(d)
# -

plt.figure(figsize=(6, 6))
plt.scatter(
    x=[p.x for p in points],
    y=[p.y for p in points],
)

solver = pywraplp.Solver.CreateSolver('TSP1','SCIP')

# ### Variables

# Edges
x = [[solver.BoolVar(f'x_{i}_{j}') for j in range(node_count)]
     for i in range(node_count)]

# +
# Order Visitied
# u = [0]+[solver.IntVar(1,node_count-1,f'u_{i}') for i in range(1,node_count)]
# -

# ### Constraints

# +
for i in range(node_count):
    # x_i_i == 0
    solver.Add(x[i][i]==0)
    
    # Row sum == 1
    solver.Add(sum([x[i][j] for j in range(node_count)])==1)
    
    # Column sum == 1
    solver.Add(sum([x[j][i] for j in range(node_count)])==1)

# for i in range(1,node_count):
#     for j in range(1,node_count):
#         if i==j: continue
#         # No subtour
#         solver.Add(u[i]-u[j]+node_count*x[i][j]<=node_count-1)
    
# -

# ### Objective

# +
obj = 0
for i in range(node_count):
    for j in range(node_count):
        obj += d[i][j]*x[i][j]

solver.Minimize(obj)
# -

# ### Solve

STATUS = {
    pywraplp.Solver.FEASIBLE: 'FEASIBLE',
    pywraplp.Solver.UNBOUNDED: 'UNBOUNDED',
    pywraplp.Solver.BASIC: 'BASIC',
    pywraplp.Solver.INFEASIBLE: 'INFEASIBLE',
    pywraplp.Solver.NOT_SOLVED: 'NOT_SOLVED',
    pywraplp.Solver.OPTIMAL: 'OPTIMAL',
}
solver.SetTimeLimit(60000)

status = solver.Solve()
STATUS[status]

for i in range(node_count):
    print([x[i][j].solution_value() for j in range(node_count)])

# # Local Search - Node Adjustment
#

import os
from solver import randomized_opt2, view_tsp,local_search,two_opt_v2

print(*ls('./data'),sep='\t')

with open('./data/tsp_100_3') as fp:
    content = fp.read().split('\n')


# +
node_count = int(content[0])
points = []
for i in range(1, node_count + 1):
    line = content[i]
    parts = line.split()
    points.append(Point(float(parts[0]), float(parts[1])))

print(f'#Nodes: {node_count}')

d = []
for i in range(node_count):
    row =[distance(points[i],points[j]) for j in range(node_count)]
    d.append(row)
    
np.shape(d)
# -

plt.figure(figsize=(6, 6))
plt.scatter(
    x=[p.x for p in points],
    y=[p.y for p in points],
)

sol = randomized_opt2(points,5)

view_tsp(sol,points)

sol2 = local_search(sol,points)

view_tsp(sol2,points,(12,12))

# # Dynamic Greedy

import random
from solver import load_data, two_opt_v2, local_search
from tqdm.auto import trange


# +
def dynamic_greedy(points,trials = 1,shuffle=False, random_state=None):
    assert isinstance(trials,int) and trials>0, 'Invalid number for trials.'
    random.seed(random_state)
    if trials>1:
        routes = []
        values = []
        for i in trange(trials):
            seed = random.randint(0,10000)
            route = dynamic_greedy(
                points,trials = 1,
                shuffle=True, 
                random_state=seed,
            )
            values.append(loss(route,points))
            routes.append(route)
        idx = np.argmin(values)
        print(*enumerate(values),sep='\n')
        return routes[idx]
    
    unused = points.copy()
    random.shuffle(unused)
    sol = []
    sol.append(unused.pop(0))
    sol.append(unused.pop(0))
    obj = loss(range(len(sol)),sol)
    while len(unused)>0:
        point = unused.pop(0)
        values = []
        for i in range(len(sol)):
            new_sol = sol.copy()
            new_sol.insert(i,point)
            values.append(loss(range(len(new_sol)),new_sol))
        idx = np.argmin(values)
        sol.insert(idx,point)
    route = [points.index(p) for p in sol]
    return route


# -

node_count,points = load_data('tsp_100_3')

sol = dynamic_greedy(points,20)

view_tsp(sol,points)

sol2 = two_opt_v2(sol,points)
view_tsp(sol2,points)

sol3 = local_search(sol2,points)
view_tsp(sol3,points)

solrnd = randomized_opt2(points,200)

view_tsp(solrnd,points)

solrnd3 = local_search(solrnd,points)
view_tsp(solrnd3,points)

# # 3-opt

from solver import load_data, two_opt_v2, local_search,randomized_opt2

node_count, points = load_data('tsp_51_1')


# +
def three_opt(route, points, random_state = None):
    random.seed(random_state)
    while True:
        delta = 0
        options = list(all_segments(len(route)))
        random.shuffle(options)
        for (a, b, c) in options:
            route, change = reverse_segment_if_better(route, points, a, b, c)
            delta += change
        if delta >= 0:
            break
    return route

def all_segments(n: int):
    return ((i, j, k)
        for i in range(n)
        for j in range(i + 2, n)
        for k in range(j + 2, n + (i > 0)))

def reverse_segment_if_better(route,points, i, j, k):
    point_dist = lambda p1,p2: ((p1.x-p2.x)**2+(p1.y-p2.y)**2)**.5
    new_route = route.copy()
    A = points[new_route[i - 1]]
    B = points[new_route[i]]
    C = points[new_route[j - 1]]
    D = points[new_route[j]]
    E = points[new_route[k - 1]]
    F = points[new_route[k % len(new_route)]]
    d0 = point_dist(A, B) + point_dist(C, D) + point_dist(E, F)
    d1 = point_dist(A, C) + point_dist(B, D) + point_dist(E, F)
    d2 = point_dist(A, B) + point_dist(C, E) + point_dist(D, F)
    d3 = point_dist(A, D) + point_dist(E, B) + point_dist(C, F)
    d4 = point_dist(F, B) + point_dist(C, D) + point_dist(E, A)

    if d0 > d1:
        new_route[i:j] = reversed(new_route[i:j])
        return new_route, -d0 + d1
    elif d0 > d2:
        new_route[j:k] = reversed(new_route[j:k])
        return new_route, -d0 + d2
    elif d0 > d4:
        new_route[i:k] = reversed(new_route[i:k])
        return new_route, -d0 + d4
    elif d0 > d3:
        tmp = new_route[j:k] + new_route[i:j]
        new_route[i:k] = tmp
        return new_route, -d0 + d3
    return new_route, 0


# -

sol = randomized_opt2(points,100)

view_tsp(sol,points)

init = np.random.choice(range(node_count),node_count,replace=False)
sol2 = three_opt(sol,points,2)
view_tsp(sol2,points)

init = np.random.choice(range(node_count),node_count,replace=False)
sol2 = three_opt(list(init),points,2)
view_tsp(sol2,points)



# # Filtered Neighbors

from sklearn.neighbors import KDTree
from solver import load_data
import numpy as np
from ortools.sat.python import cp_model


# +
def generate_tsp(node_count, random_state = None):
    np.random.seed(random_state)
    x = np.random.randint(100,size=node_count,dtype=int)
    y = np.random.randint(100,size=node_count,dtype=int)
    points = [Point(i,j) for i,j in zip(x,y)]
    return points

def view_matrix(mat,points,figsize = (8,8)):
    lines = []
#     row = np.argmax(sol_mat,axis=0)
    row = np.arange(len(points))
    col = np.argmax(sol_mat,axis=1)
    plt.figure(figsize=figsize)
    for i,j in zip(row,col):
        x = [points[i].x,points[j].x]
        y = [points[i].y,points[j].y]
        plt.plot(x,y,'ok-')
        
def check_feasibility(sol_mat):
    dest = np.argmax(sol_mat,axis=1)
    return len(set(dest))==sol_mat.shape[0]

def matrix_to_route(matrix):
    assert check_feasibility, 'Invalid input matrix'
    route = [0]
    dest = np.argmax(matrix,axis = 1)
    n = len(dest)
    for i in range(n):
        route.append(dest[route[-1]])
    final = route.pop(-1)
    if final!=0:
        raise Exception('The route is not complete.')
    return route


# +
node_count, points = load_data('tsp_51_1')


# node_count=20
# points = generate_tsp(node_count,23)

plt.figure(figsize=(6, 6))
plt.scatter(
    x=[p.x for p in points],
    y=[p.y for p in points],
)


# -

def get_k_value(n):
    '''
    50 nodes --> k = 5
    100 nodes --> k = 8
    '''
    k = 5 + 3*np.log2(n/50)
    return int(k)


# +
def filtered_neighbors(points):
    node_count = len(points)
    
    # Neighbor Detection
    xy = np.array(points)
    kdt = KDTree(xy)
    k = get_k_value(node_count) + 1
    dist, neighbors = kdt.query(xy,k)
    
    # Create CP Model
    model = cp_model.CpModel()
    
    # Define Variables
    a = [] # whether i and j are connected
    u = [] # order of points
    for i in range(node_count):
        row = [model.NewBoolVar(f'a_{i}_{j}') for j in range(node_count)]
        a.append(row)
    u = [model.NewIntVar(0,node_count-1,f'u_{i}') for i in range(node_count)]
    
    # Add Constraints
    for i in range(node_count):
        model.AddAbsEquality(0, a[i][i])
        model.Add(sum([ a[i][j] for j in range(node_count)])==1)
        model.Add(sum([ a[j][i] for j in range(node_count)])==1)
    
    for i in range(node_count-1):
        for j in range(i,node_count):
            model.Add(a[i][j]+a[j][i]<=1)
    
    # Connectivity Constraint
    model.AddAllDifferent(u)
    for i in range(1,node_count):
        for j in range(1,node_count):
            if i==j: continue
            model.Add(u[i]-u[j]+node_count*a[i][j]<=node_count-1)
    model.Add(u[0]==0)
    
    # Proximity Constraint
    for i in range(node_count):
        neigh = neighbors[i,1:]
        for j in range(node_count):
            if j not in neigh:
                model.AddAbsEquality(0,a[i][j])
                model.AddAbsEquality(0,a[j][i])
                
    # Objective
    obj = []
    for i in range(node_count):
        for j in range(node_count):
            d = distance(points[i],points[j])*100
            obj.append(int(d)*a[i][j])
    model.Minimize(sum(obj))
    
    # Solve
    cpsolver = cp_model.CpSolver()
    cpsolver.parameters.max_time_in_seconds = 60.0
    status = cpsolver.Solve(model)
    print(cpsolver.StatusName())
    
    if status == cp_model.MODEL_INVALID or cp_model.INFEASIBLE:
        raise RuntimeError('Unable to find a solution.')
    
    # Extract Solution
    sol_mat = []
    for i in range(node_count):
        row = [cpsolver.Value(a[i][j]) for j in range(node_count)]
        sol_mat.append(row)
    sol_mat = np.array(sol_mat)
    
    # Convert solution to readable format
    sol = matrix_to_route(sol_mat)
    
    return status == cp_model.OPTIMAL, sol

    
# -

filtered_neighbors(points)

# %debug

xy = np.array(points)
kdt = KDTree(xy)

dist, neighbors = kdt.query(xy,6)
neighbors[0]

model = cp_model.CpModel()

a = [] # whether i and j are connected
u = [] # order of points
for i in range(node_count):
    row = [model.NewBoolVar(f'a_{i}_{j}') for j in range(node_count)]
    a.append(row)
u = [model.NewIntVar(0,node_count-1,f'u_{i}') for i in range(node_count)]

# +
# Add Constraints
for i in range(node_count):
    model.AddAbsEquality(0, a[i][i])
    model.Add(sum([ a[i][j] for j in range(node_count)])==1)
    model.Add(sum([ a[j][i] for j in range(node_count)])==1)

for i in range(node_count-1):
    for j in range(i,node_count):
        model.Add(a[i][j]+a[j][i]<=1)

# for i in range(node_count-2):
#     for j in range(i,node_count-1):
#         for k in range(j,node_count):
#             model.Add(a[i][j]+a[j][k]+a[k][i]<=2)
#             model.Add(a[j][i]+a[i][k]+a[k][j]<=2)
            
model.AddAllDifferent(u)

for i in range(1,node_count):
    for j in range(1,node_count):
        if i==j: continue
        model.Add(u[i]-u[j]+node_count*a[i][j]<=node_count-1)
model.Add(u[0]==0)
# model.Add(u[-1]==node_count)
# -

for i in range(node_count):
    neigh = neighbors[i,1:]
    for j in range(node_count):
        if j not in neigh:
            model.AddAbsEquality(0,a[i][j])
            model.AddAbsEquality(0,a[j][i])


# Objective
obj = []
for i in range(node_count):
    for j in range(node_count):
        d = distance(points[i],points[j])*100
        obj.append(int(d)*a[i][j])
model.Minimize(sum(obj))

cpsolver = cp_model.CpSolver()
cpsolver.parameters.max_time_in_seconds = 60.0
status = cpsolver.Solve(model)

# STATUS = {
#     cp_model.FEASIBLE: 'FEASIBLE',
#     cp_model.UNKNOWN: 'UNKNOWN',
#     cp_model.MODEL_INVALID: 'MODEL_INVALID',
#     cp_model.INFEASIBLE: 'INFEASIBLE',
#     cp_model.OPTIMAL: 'OPTIMAL',
# }
# STATUS[status]
cpsolver.StatusName()

# +
sol_mat = []
for i in range(node_count):
    row = [cpsolver.Value(a[i][j]) for j in range(node_count)]
    sol_mat.append(row)
    print(row)
    
sol_mat = np.array(sol_mat)

# -

np.argmax(sol_mat,axis=0)

view_matrix(sol_mat,points)



sol = matrix_to_route(sol_mat)

view_tsp(sol,points)

sol2 = two_opt_v2(sol,points)
view_tsp(sol2,points)

np.random.randint(0,5,(3,3))

np.sum(_,axis=1)

q = 10
f'{np.log(q)=}'

# # 
#     
