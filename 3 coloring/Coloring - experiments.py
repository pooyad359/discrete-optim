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

# # Main

from ortools.constraint_solver import pywrapcp
from ortools.sat.python import cp_model
import os

files =os.listdir('./data')
[f'{i}. {f}' for i,f in enumerate(files)]

idx = 27
with open(f'./data/{files[idx]}','r') as fp:
    input_data = fp.read()
print(input_data[:15])

# +


lines = input_data.split('\n')

first_line = lines[0].split()
node_count = int(first_line[0])
edge_count = int(first_line[1])

edges = []
for i in range(1, edge_count + 1):
    line = lines[i]
    parts = line.split()
    edges.append((int(parts[0]), int(parts[1])))
    
print(node_count, edge_count)
# -

edge_array = np.array(edges)
orders = [np.sum(edge_array==i) for i in range(node_count)]
highest_order = int(np.argmax(orders))
print(highest_order,orders[highest_order] )

# # Constraint Programing
#

# +
n_points = node_count
n_colors = max(orders)+1

solver = pywrapcp.Solver('Colors')

# colors = [solver.IntVar(0,n_points-1,f'c_{i}') for i in range(n_points)]
colors = [solver.IntVar(0,highest_order-1,f'c_{i}') for i in range(n_points)]
print(n_points,)
# -



for i,j in edges:
    solver.Add(colors[i]!=colors[j])

# +
# solver.Add(colors[highest_order] < 1)
# solver.

# +
mode = 1
if mode ==0:
    
    solver.Minimize(solver.Sum(colors),1)
else:
    max_colors = solver.IntVar(0,highest_order,'max_color')
    solver.Add(colors[highest_order] < 1)
    solver.MaxEquality(colors,max_colors)
    solver.Minimize(max_colors,1)
#     solver.Minimize(solver.Max(colors),1)

# solver.TimeLimit(5000)
db = solver.Phase(colors,
                    solver.CHOOSE_FIRST_UNBOUND,
                    solver.ASSIGN_MIN_VALUE)
solver.NewSearch(db)

# -

solver.NextSolution()

# +
num_solutions = 0

if solver.NextSolution():
    res = [c.Value() for c in colors]

print(max(*res)+1)
print(*res)

# +
    
solver.EndSearch()
# -

print(len(res),res)


def algo(edges,n_points):
    # solver = pywrapcp.Solver('Colors')
    solver = pywrapcp.Solver('Colors')
    colors = [solver.IntVar(0,n_points-1,f'c_{i}') for i in range(n_points)]
    for i,j in edges:
        solver.Add(colors[i]!=colors[j])
    solver.Minimize(solver.Max(colors),1)
    db = solver.Phase(colors,
                        solver.CHOOSE_FIRST_UNBOUND,
                        solver.ASSIGN_MIN_VALUE)
    solver.NewSearch(db)

    if solver.NextSolution():
        return [c.Value() for c in colors]
    else:
        return []


# +

def algo2(edges,n_points):
    solver = pywrapcp.Solver('Colors')
    colors = [solver.IntVar(0,n_points-1,f'c_{i}') for i in range(n_points)]
    
    for i,j in edges:
        solver.Add(colors[i]!=colors[j])
    solver.Minimize(solver.Max(colors),1)
    db = solver.Phase(colors,
                        solver.CHOOSE_FIRST_UNBOUND,
                        solver.ASSIGN_MIN_VALUE)
    solver.NewSearch(db)

    if solver.NextSolution():
        return [c.Value() for c in colors]
    else:
        return []
# -





# # Linear Solver

from ortools.linear_solver import pywraplp
import numpy as np
import os

files =os.listdir('./data')
[f'{i}. {f}' for i,f in enumerate(files)]

idx = 26
print(files[idx])
with open(f'./data/{files[idx]}','r') as fp:
    input_data = fp.read()
print(input_data[:15])

# +
lines = input_data.split('\n')

first_line = lines[0].split()
node_count = int(first_line[0])
edge_count = int(first_line[1])

edges = []
for i in range(1, edge_count + 1):
    line = lines[i]
    parts = line.split()
    edges.append((int(parts[0]), int(parts[1])))
# -

solver = pywraplp.Solver.CreateSolver('simple_mip_program', 'CBC')

ccnt = np.array(edges).flatten()
n_colors = max(*[len(ccnt[ccnt==i]) for i in range(node_count)])
n_colors = 18#node_count
cmat = [[0]*n_colors]*node_count # node_count x n_colors

# Create the color matrix
for i in range(node_count):
#     for j in range(n_colors):
#         print(f'c_{i}_{j}')
    cmat[i] = [solver.IntVar(0,1,f'c_{i}_{j}') for j in range(n_colors)]

# +

# Assign one color to every node
for i in range(node_count):
    solver.Add(solver.Sum(cmat[i])==1)
#     solver.Add(solver.Sum(cmat[i])>=1)
# -

solver.Add(solver.Sum([cmat[i][0] for i in range(n_colors)])>=1)

# Neighbor nodes can't have the same color
for n1,n2 in edges:
    for c in range(n_colors):
        solver.Add(cmat[n1][c]+cmat[n2][c]<=1)

# +
col_sums = [solver.Sum([cmat[i][c] for i in range(node_count)]) for c in range(n_colors)]

for c in range(1,n_colors):
    solver.Add(col_sums[c-1]>=col_sums[c])
# -

solver.Minimize(solver.Sum([col_sums[i]*i for i in range(n_colors)]))
# solver.Maximize(solver.Sum([col_sums[i]*col_sums[i] for i in range(n_colors)]))

status = solver.Solve()

status == pywraplp.Solver.OPTIMAL

solver.Sum([solver.Sum(cmat[i])*i for i in range(node_count)]).solution_value()

for j in range(node_count):
    print(*[cmat[j][i].solution_value() for i in range(n_colors)])

# # Constraint Prog. 2

from ortools.constraint_solver import pywrapcp
from ortools.sat.python import cp_model
import os


def show_mat(l):
    for row in l:
        print(*row)


files =os.listdir('./data')
[f'{i}. {f}' for i,f in enumerate(files)]

idx = 27
with open(f'./data/{files[idx]}','r') as fp:
    input_data = fp.read()
print(input_data[:15])

# +


lines = input_data.split('\n')

first_line = lines[0].split()
node_count = int(first_line[0])
edge_count = int(first_line[1])

edges = []
for i in range(1, edge_count + 1):
    line = lines[i]
    parts = line.split()
    edges.append((int(parts[0]), int(parts[1])))
# -

n_points = node_count
n_colors = n_points-1
solver = pywrapcp.Solver('Colors')
colors = [[solver.IntVar(0,1,f'c_{i}_{j}') for j in range(n_colors)] for i in range(n_points)]
print(n_points)

for row in colors:
    solver.Add(solver.Sum(row)==1)

# Neighbor nodes can't have the same color
for n1,n2 in edges:
    for c in range(n_colors):
        solver.Add(colors[n1][c]*colors[n2][c]==0)    

col_sums = [solver.Sum([colors[i][c] for i in range(node_count)]) for c in range(n_colors)]
for c in range(1,n_colors):
    solver.Add(col_sums[c-1]>=col_sums[c])

# solver.Minimize(solver.Sum([col_sums[i]*i for i in range(n_colors)]),1)
solver.Maximize(solver.Sum([col_sums[i]*col_sums[i] for i in range(n_colors)]),1)

flat_c = [item for sublist in colors for item in sublist]
db = solver.Phase(flat_c,solver.CHOOSE_FIRST_UNBOUND,solver.ASSIGN_MIN_VALUE)
solver.NewSearch(db)


# +
num_solutions = 0

if solver.NextSolution():
    res = []
    for row in colors:
        res.append([c.Value() for c in row])
    print(f'Solution#{num_solutions+1}')
    show_mat(res)
    num_solutions+=1

    
solver.EndSearch()
# print(max(*res)+1)

# -

show_mat(res)

# # Sample problems

solver = pywrapcp.Solver('nums') 

x = [solver.IntVar(1,9,f'x_{i}') for i in range(5)]

solver.Add(solver.AllDifferent(x))
for i in range(1,len(x)):
    solver.Add(x[i]<x[i-1])

solver.Add(solver.Sum(x)==20)
solver.Minimize(x[0],1)

# +
db = solver.Phase(x,solver.CHOOSE_FIRST_UNBOUND,solver.ASSIGN_MIN_VALUE)
solver.NewSearch(db)
num_solutions = 0

if solver.NextSolution():
    res = [c.Value() for c in x]
    
solver.EndSearch()
# -

x

# # Use Cp_model

files =os.listdir('./data')
[f'{i}. {f}' for i,f in enumerate(files)]

from ortools.sat.python import cp_model
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


# +
def from_file(index):
    files = os.listdir('./data')
    with open(f'./data/{files[index]}', 'r') as fp:
        input_data = fp.read()
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    edges = []
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append((int(parts[0]), int(parts[1])))

    print(f'File: {files[index]}')
    print(f'Nodes: {node_count}')
    print(f'Edges: {edge_count}')
    return node_count, edge_count, edges


def node_degree(edges, n_nodes=None):
    occur = np.array(edges).flatten()
    if n_nodes is None:
        n_nodes = occur.max()
    return [len(occur[occur == o]) for o in range(n_nodes)]


def visualize(edges, n_nodes=None, colors=None, random_seed=1, mode=0,alpha = 0.5):
    np.random.seed(random_seed)
    plt.figure(figsize=(10, 10))
    if n_nodes is None:
        n_nodes = np.array(edges).max() + 1

    if mode == 0:
        t = np.linspace(0, 1, n_nodes + 1) * 2 * np.pi
        coords = np.c_[np.sin(t[:-1]), np.cos(t[:-1])]
    elif mode == 1:
        coords = np.random.rand(n_nodes, 2)
    else:
        raise (ValueError('Unsupported mode.'))

    if colors is not None:
        cmap = cm.get_cmap('jet', len(np.unique(colors)))
        colors = np.array(colors)
        for c in np.unique(colors):
            plt.scatter(coords[colors == c, 0],
                        coords[colors == c, 1],
                        marker='o',
                        color=cmap(c))
    else:
        plt.scatter(coords[:, 0], coords[:, 1], marker='o', c='black')
    for n1, n2 in edges:
        xs = coords[[n1, n2], 0]
        ys = coords[[n1, n2], 1]
        plt.plot(xs, ys, color='black', linewidth=1, alpha=alpha)
        
def check_results(edges,colors):
    n_err = 0
    for n1,n2 in edges:
        if colors[n1]==colors[n2]:
            nerr +=1
            print(f'Error ditected: Nodes {n1} and {n2} were assigned the same color code ({color[n1]}).')
    if n_err>0:
        print(f'[{n_err} errors were found.]')
        return False
    else:
        print('[All good!]')
        return True


# -

# ## Load file

node_count, edge_count, edges = from_file(7)

visualize(edges,alpha=.1)

degs = node_degree(edges)
print(degs)
max_color = np.max(degs)
print(max_color)

max_color

# ## Modelling

# +
model = cp_model.CpModel()


# variables
max_color = 15
colors = [model.NewIntVar(0,int(max_color),f'c_{i}') for i in range(node_count)]
# colors = [model.NewIntVar(0,int(17),f'c_{i}') for i in range(node_count)]

# c_i =/= c_j
for n1,n2 in edges:
    model.Add(colors[n1]!=colors[n2])
# -

model.Minimize(max(*colors))

solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 30.0
status = solver.Solve(model)

# +
res = [solver.Value(c) for c in colors]

print(*res)
print(max(*res)+1)
print([res.count(i) for i in np.unique(res)])
# -

check_results(edges,res)

visualize(edges,colors=res)

cp_model.UNKNOWN, cp_model.FIXED_SEARCH,cp_model.FEASIBLE, cp_model.INFEASIBLE,cp_model.OPTIMAL


# ## Final Function

# +
def stepwise_opt(edges,node_count):
    print(f'*** Finding Solution for {node_count} Nodes',flush=True)
    n_colors,colors = color_graph(edges,node_count)
    print(f'Found solution for {n_colors} colors',flush=True)
    while n_colors is not None:
        print(f'\tTrying {n_colors-1} colors',flush=True)
        n_colors_prev, colors_prev = n_colors, colors
        max_color = n_colors-1
        n_colors,colors = color_graph(edges,node_count,max_color=max_color-1)
    return n_colors_prev, colors_prev

def color_graph(edges, node_count, max_color=None,max_time = None):
    if max_color is None:
        degs = node_degree(edges)
        max_color = np.max(degs)
        
    if max_time is None:
        max_time = max_time_calc(edges,node_count)
    # Create model
    model = cp_model.CpModel()
    
    # Define Variable
    colors = [model.NewIntVar(0,int(max_color),f'c_{i}') for i in range(node_count)]
    
    # Define Constraints
    for n1,n2 in edges:
        model.Add(colors[n1]!=colors[n2])
        
    # Define Objective
    model.Minimize(max(*colors))
    
    # Create Solver
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max_time
    status = solver.Solve(model)
    
    if status == cp_model.FEASIBLE or status==cp_model.OPTIMAL:
        res = [solver.Value(c) for c in colors]
        n_colors = max(*res)+1
        return n_colors, res
    else:
        return None,[]

def max_time_calc(edges,node_count):
    pct = len(edges)/node_count**2
    t = node_count/1000*10 # using 10 sec for 1000 nodes as base
    t = t*pct/.05 # scale based on percentage of complete graph
    return max(10,t*2)


# -

node_count, edge_count, edges = from_file(11)

color_graph(edges,node_count,max_time=120)

stepwise_opt(edges,node_count)

# # Constraint Programing - matrix notation

files =os.listdir('./data')
[f'{i}. {f}' for i,f in enumerate(files)]

from ortools.sat.python import cp_model
from ortools.sat.python import 
from ortools.constraint_solver import pywrapcp
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


# +
def from_file(index):
    files = os.listdir('./data')
    with open(f'./data/{files[index]}', 'r') as fp:
        input_data = fp.read()
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    edges = []
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append((int(parts[0]), int(parts[1])))

    print(f'File: {files[index]}')
    print(f'Nodes: {node_count}')
    print(f'Edges: {edge_count}')
    return node_count, edge_count, edges


def node_degree(edges, n_nodes=None):
    occur = np.array(edges).flatten()
    if n_nodes is None:
        n_nodes = occur.max()
    return [len(occur[occur == o]) for o in range(n_nodes)]


def visualize(edges, n_nodes=None, colors=None, random_seed=1, mode=0,alpha = 0.5):
    np.random.seed(random_seed)
    plt.figure(figsize=(10, 10))
    if n_nodes is None:
        n_nodes = np.array(edges).max() + 1

    if mode == 0:
        t = np.linspace(0, 1, n_nodes + 1) * 2 * np.pi
        coords = np.c_[np.sin(t[:-1]), np.cos(t[:-1])]
    elif mode == 1:
        coords = np.random.rand(n_nodes, 2)
    else:
        raise (ValueError('Unsupported mode.'))

    if colors is not None:
        cmap = cm.get_cmap('jet', len(np.unique(colors)))
        colors = np.array(colors)
        for c in np.unique(colors):
            plt.scatter(coords[colors == c, 0],
                        coords[colors == c, 1],
                        marker='o',
                        color=cmap(c))
    else:
        plt.scatter(coords[:, 0], coords[:, 1], marker='o', c='black')
    for n1, n2 in edges:
        xs = coords[[n1, n2], 0]
        ys = coords[[n1, n2], 1]
        plt.plot(xs, ys, color='black', linewidth=1, alpha=alpha)
        
def check_results(edges,colors):
    n_err = 0
    for n1,n2 in edges:
        if colors[n1]==colors[n2]:
            nerr +=1
            print(f'Error ditected: Nodes {n1} and {n2} were assigned the same color code ({color[n1]}).')
    if n_err>0:
        print(f'[{n_err} errors were found.]')
        return False
    else:
        print('[All good!]')
        return True


# -

# ## Load file

node_count, edge_count, edges = from_file(7)

visualize(edges,alpha=.1)

degs = node_degree(edges)
print(degs)
max_color = np.max(degs)
print(max_color)

max_color

# ## Modelling



n_points = node_count
n_colors = max(degs)+1
n_colors = 16
# solver = pywrapcp.Solver('Colors')
model = cp_model.CpModel()

colors = [[model.NewBoolVar(f'x_{i}_{c}') for c in range(n_colors)] for i in range(n_points)]

# +
# Connected edges
for i,j in edges:
    for c in range(n_colors):
        model.AddLinearConstraint(colors[i][c]+colors[j][c],0,1)
        
# Only one color
for i in range(n_points):
    model.AddLinearConstraint(sum(colors[i]),1,1)
    
# Breaking Symmetry
for c in range(n_colors-1):
    s1 = sum([colors[i][c] for i in range(n_points)])
    s2 = sum([colors[i][c+1] for i in range(n_points)])
    model.Add(s1>=s2)

# +
# obj = 0
# for c in range(n_colors):
#     obj+=sum([colors[i][c]*c for i in range(n_points)])
# model.Minimize(obj)
# -

# ## Solving

solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 30.0
status = solver.Solve(model)

solver.StatusName(status)

# res = [solver.Value(c) for c in colors]
res = [[solver.Value(colors[i][c]) for c in range(n_colors)] for i in range(n_points)]
res = np.array(res)
# print(*res)
# print(max(*res)+1)
# print([res.count(i) for i in np.unique(res)])

sol = res@np.arange(res.shape[1])

s = set(sol)
print(f'{len(s)}/{n_colors}',s,sep='\n')

check_results(edges,sol)

# +
# mode = 1
# if mode ==0:
    
#     solver.Minimize(solver.Sum(colors),1)
# else:
#     max_colors = solver.IntVar(0,highest_order,'max_color')
#     solver.Add(colors[highest_order] < 1)
#     solver.MaxEquality(colors,max_colors)
#     solver.Minimize(max_colors,1)
# #     solver.Minimize(solver.Max(colors),1)
# -



# solver.TimeLimit(5000)
db = solver.Phase(sum(colors,[]))#, solver.CHOOSE_FIRST_UNBOUND)
solver.NewSearch(db)

solver.NextSolution()

# +
num_solutions = 0

if solver.NextSolution():
    res = [c.Value() for c in colors]

print(max(*res)+1)
print(*res)

# +
n_points = node_count
n_colors = max(orders)+1

solver = pywrapcp.Solver('Colors')

# colors = [solver.IntVar(0,n_points-1,f'c_{i}') for i in range(n_points)]
colors = [solver.IntVar(0,highest_order-1,f'c_{i}') for i in range(n_points)]
print(n_points,)



for i,j in edges:
    solver.Add(colors[i]!=colors[j])

# solver.Add(colors[highest_order] < 1)
# solver.

mode = 1
if mode ==0:
    
    solver.Minimize(solver.Sum(colors),1)
else:
    max_colors = solver.IntVar(0,highest_order,'max_color')
    solver.Add(colors[highest_order] < 1)
    solver.MaxEquality(colors,max_colors)
    solver.Minimize(max_colors,1)
#     solver.Minimize(solver.Max(colors),1)

# solver.TimeLimit(5000)
db = solver.Phase(colors,
                    solver.CHOOSE_FIRST_UNBOUND,
                    solver.ASSIGN_MIN_VALUE)
solver.NewSearch(db)


solver.NextSolution()

num_solutions = 0

if solver.NextSolution():
    res = [c.Value() for c in colors]

print(max(*res)+1)
print(*res)
# -







model.Minimize(max(*colors))

solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 30.0
status = solver.Solve(model)

# +
res = [solver.Value(c) for c in colors]

print(*res)
print(max(*res)+1)
print([res.count(i) for i in np.unique(res)])

# +
files =os.listdir('./data')
[f'{i}. {f}' for i,f in enumerate(files)]

from ortools.sat.python import cp_model
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

def from_file(index):
    files = os.listdir('./data')
    with open(f'./data/{files[index]}', 'r') as fp:
        input_data = fp.read()
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    edges = []
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append((int(parts[0]), int(parts[1])))

    print(f'File: {files[index]}')
    print(f'Nodes: {node_count}')
    print(f'Edges: {edge_count}')
    return node_count, edge_count, edges


def node_degree(edges, n_nodes=None):
    occur = np.array(edges).flatten()
    if n_nodes is None:
        n_nodes = occur.max()
    return [len(occur[occur == o]) for o in range(n_nodes)]


def visualize(edges, n_nodes=None, colors=None, random_seed=1, mode=0,alpha = 0.5):
    np.random.seed(random_seed)
    plt.figure(figsize=(10, 10))
    if n_nodes is None:
        n_nodes = np.array(edges).max() + 1

    if mode == 0:
        t = np.linspace(0, 1, n_nodes + 1) * 2 * np.pi
        coords = np.c_[np.sin(t[:-1]), np.cos(t[:-1])]
    elif mode == 1:
        coords = np.random.rand(n_nodes, 2)
    else:
        raise (ValueError('Unsupported mode.'))

    if colors is not None:
        cmap = cm.get_cmap('jet', len(np.unique(colors)))
        colors = np.array(colors)
        for c in np.unique(colors):
            plt.scatter(coords[colors == c, 0],
                        coords[colors == c, 1],
                        marker='o',
                        color=cmap(c))
    else:
        plt.scatter(coords[:, 0], coords[:, 1], marker='o', c='black')
    for n1, n2 in edges:
        xs = coords[[n1, n2], 0]
        ys = coords[[n1, n2], 1]
        plt.plot(xs, ys, color='black', linewidth=1, alpha=alpha)
        
def check_results(edges,colors):
    n_err = 0
    for n1,n2 in edges:
        if colors[n1]==colors[n2]:
            nerr +=1
            print(f'Error ditected: Nodes {n1} and {n2} were assigned the same color code ({color[n1]}).')
    if n_err>0:
        print(f'[{n_err} errors were found.]')
        return False
    else:
        print('[All good!]')
        return True

## Load file

node_count, edge_count, edges = from_file(28)

visualize(edges,alpha=.1)

degs = node_degree(edges)
print(degs)
max_color = np.max(degs)
print(max_color)

max_color

## Modelling

model = cp_model.CpModel()


# variables
max_color = 12
colors = [model.NewIntVar(0,int(max_color),f'c_{i}') for i in range(node_count)]
# colors = [model.NewIntVar(0,int(17),f'c_{i}') for i in range(node_count)]

# c_i =/= c_j
for n1,n2 in edges:
    model.Add(colors[n1]!=colors[n2])

model.Minimize(max(*colors))

solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 30.0
status = solver.Solve(model)

res = [solver.Value(c) for c in colors]

print(*res)
print(max(*res)+1)
print([res.count(i) for i in np.unique(res)])
# -





# # Naive Search

from ortools.sat.python import cp_model
import numpy as np
import os

files =os.listdir('./data')
[f'{i}. {f}' for i,f in enumerate(files)]

idx = 5
print(files[idx])
with open(f'./data/{files[idx]}','r') as fp:
    input_data = fp.read()
print(input_data[:15])

# +
lines = input_data.split('\n')

first_line = lines[0].split()
node_count = int(first_line[0])
edge_count = int(first_line[1])

edges = []
for i in range(1, edge_count + 1):
    line = lines[i]
    parts = line.split()
    edges.append((int(parts[0]), int(parts[1])))
    
print(node_count, edge_count)
# -

edge_array = np.array(edges)
orders = [np.sum(edge_array==i) for i in range(node_count)]
highest_order = int(np.argmax(orders))
print(highest_order,orders[highest_order] )

# +
solver = cp_model.CpSolver()
model = cp_model.CpModel()

max_color = highest_order

c = [model.NewIntVar(0, max_color, "i%i" % i) for i in range(0, node_count)]

for i in range(len(edges)):
    model.Add(c[edges[i][0]] != c[edges[i][1]])

model.Minimize(max([c[i] for i in range(0, node_count)]))  
 
status = solver.Solve(model)
print(status)
solution = [solver.Value(c[i]) for i in range(0, node_count)]

output_data = str(max(solution)+1) + ' ' + str(solver.StatusName(status)) + '\n'
output_data += ' '.join(map(str, solution))
print(output_data)
# -


