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

import numpy as np
import os
from tqdm.auto import tqdm, trange


def get_degrees(edges,node_count):
    edge_array = np.array(edges)
    degrees = [np.sum(edge_array==i) for i in range(node_count)]
    return degrees


files =os.listdir('./data')
[f'{i}. {f}' for i,f in enumerate(files)]

idx = 34
print(files[idx])
with open(f'./data/{files[idx]}','r') as fp:
    input_data = fp.read()
print(input_data[:25])

# +
lines = input_data.split('\n')

first_line = lines[0].split()
n_points = node_count = int(first_line[0])
edge_count = int(first_line[1])

edges = []
for i in range(1, edge_count + 1):
    line = lines[i]
    parts = line.split()
    edges.append((int(parts[0]), int(parts[1])))
    
print(node_count, edge_count)


# -

def choose_color(colors):
    color = 0
    while color in colors:
        color +=1
    return color


# +
def greedy_solver(edges, n_points):
    degrees = get_degrees(edges, n_points)
    edge_dict = {i: set() for i in range(n_points)}
    for v1, v2 in edges:
        edge_dict[v1].add(v2)
        edge_dict[v2].add(v1)
    # get indices by degree of the vertex
    idx = np.argsort(degrees)[::-1]
    colors = -np.ones(n_points, dtype=int)
    colors[idx[0]] = 0
    for i, ind in enumerate(idx):
        connected_nodes = edge_dict[ind]
        c = choose_color(colors[list(connected_nodes)])
        colors[ind] = c
    return colors

def semi_greedy_solver(edges,n_points,degrees, edge_dict, order = None):
        
    # get indices by degree of the vertex
    if order is None:
        idx = np.argsort(degrees)[::-1]
    else:
        idx = np.array(order)
    colors = -np.ones(n_points, dtype=int)
    colors[idx[0]] = 0
    for i, ind in enumerate(idx):
        connected_nodes = edge_dict[ind]
        c = choose_color(colors[list(connected_nodes)])
        colors[ind] = c
    return colors


# -

# # Random Selection

def swap_elements(array,n=1,random_state = None):
    np.random.seed(random_state)
    x = np.array(array)
    length = len(x)
    for _ in range(n):
        i,j = np.random.choice(range(length),size=2,replace = False)
        x[i], x[j] = x[j], x[i]
    return x


# +
N_ROUNDS = 1000

degrees = get_degrees(edges, n_points)
edge_dict = {i: set() for i in range(n_points)}
for v1, v2 in edges:
    edge_dict[v1].add(v2)
    edge_dict[v2].add(v1)

solution = greedy_solver(edges, n_points)
n_colors = max(solution) + 1
idx = np.argsort(degrees)[::-1]

pbar = trange(N_ROUNDS)
for i in pbar:
    n = np.random.randint(1, 6)
    new_idx = swap_elements(idx.copy(), n)
    res = semi_greedy_solver(
        edges,
        n_points,
        degrees,
        edge_dict,
        order=new_idx,
    )
    res_colors = np.max(res) + 1

    desc = f'#{i+1:03.0f}:  {res_colors}  / {n_colors}\n'

    pbar.set_description(desc, refresh=True)
    if res_colors <= n_colors:
        solution = res
        n_colors = res_colors
        idx = new_idx
# -

# # Genetic Algorithm

from collections import namedtuple
from copy import deepcopy

GreedyGraph = namedtuple('GreedyGraph',['colors','order','count'])

# +
degrees = get_degrees(edges, n_points)
edge_dict = {i: set() for i in range(n_points)}
for v1, v2 in edges:
    edge_dict[v1].add(v2)
    edge_dict[v2].add(v1)

solution = greedy_solver(edges, n_points)
n_colors = max(solution) + 1
idx = np.argsort(degrees)[::-1]
print(n_colors)

# +
np.random.seed(1)
# Initialize population
N_POP = 100
N_ROUNDS = 200
population = [GreedyGraph(solution,idx,n_colors)]

for _ in range(1,N_POP):
    n = np.random.randint(1, 6)
    new_idx = swap_elements(idx.copy(), n)
    res = semi_greedy_solver(
        edges,
        n_points,
        degrees,
        edge_dict,
        order=new_idx,
    )
    res_colors = np.max(res) + 1
    population.append(GreedyGraph(res,new_idx,res_colors))
# -

for i in tqdm(range(N_ROUNDS)):
    best = np.min([g.count for g in population])
    worst = np.max([g.count for g in population])
    print(f'#{i:03.0f}\tBest: {best}  \tWorst: {worst}  \t {len(population)}')
    if best==worst:
        new_population = [deepcopy(g) for g in population if np.random.rand()>.75]
    else:
        new_population = [deepcopy(g) for g in population if g.count==best]
    while len(new_population)<N_POP:#for _ in range(len(new_popultaion),N_POP):
        selected = population[np.random.randint(N_POP)]
        n = np.random.randint(1,6)
        new_idx = swap_elements(selected.order,n)
        res = semi_greedy_solver(
            edges,
            n_points,
            degrees,
            edge_dict,
            order=new_idx,
        )
        res_colors = np.max(res) + 1
        new_population.append(GreedyGraph(res,new_idx,res_colors))
    population = new_population.copy()
    

# +
N_POP = 100
N_ROUNDS = 1000

GreedyGraph = namedtuple('GreedyGraph',['colors','order','count'])

def ga_solver(edges, n_points, random_state = None):
    np.random.seed(random_state)
    
   
    # Get degree of each node
    degrees = get_degrees(edges, n_points)
    
    # Create a dictionary of connections for each node
    edge_dict = {i: set() for i in range(n_points)}
    for v1, v2 in edges:
        edge_dict[v1].add(v2)
        edge_dict[v2].add(v1)
    
    # Get initial solution using greedy
    solution = greedy_solver(edges, n_points)
    n_colors = max(solution) + 1
    idx = np.argsort(degrees)[::-1]
    n_greedy = n_colors

    # Genetic Algorithm
    
    # Initialize population
    population = [GreedyGraph(solution,idx,n_colors)]

    for _ in range(1,N_POP):
        n = np.random.randint(1, 6)
        new_idx = swap_elements(idx.copy(), n)
        res = semi_greedy_solver(
            edges,
            n_points,
            degrees,
            edge_dict,
            order=new_idx,
        )
        res_colors = np.max(res) + 1
        population.append(GreedyGraph(res,new_idx,res_colors))
    
    # Evolution
    pbar = trange(N_ROUNDS)
    for i in pbar:
        best = np.min([g.count for g in population])
        worst = np.max([g.count for g in population])
        desc = f'{n_greedy} -> {best}/{worst}'
        pbar.set_description(desc, refresh=True)
        if best==worst:
            new_population = [deepcopy(g) for g in population if np.random.rand()>.75]
        else:
            new_population = [deepcopy(g) for g in population if g.count==best]
        while len(new_population)<N_POP:
            selected = population[np.random.randint(N_POP)]
            n = np.random.randint(1,6)
            new_idx = swap_elements(selected.order,n)
            res = semi_greedy_solver(
                edges,
                n_points,
                degrees,
                edge_dict,
                order=new_idx,
            )
            res_colors = np.max(res) + 1
            new_population.append(GreedyGraph(res,new_idx,res_colors))
        population = new_population.copy()
        
    id_sol = np.argmin([g.count for g in population])
    return population[id_sol].colors
# -

N_ROUNDS=100
ga_solver(edges,n_points,1)

# # Group rearangement

files =os.listdir('./data')
[f'{i}. {f}' for i,f in enumerate(files)]

idx = 2
print(files[idx])
with open(f'./data/{files[idx]}','r') as fp:
    input_data = fp.read()
print(input_data[:25])

# +
lines = input_data.split('\n')

first_line = lines[0].split()
n_points = node_count = int(first_line[0])
edge_count = int(first_line[1])

edges = []
for i in range(1, edge_count + 1):
    line = lines[i]
    parts = line.split()
    edges.append((int(parts[0]), int(parts[1])))
    
print(node_count, edge_count)


# +
def sort_by_degree(all_degrees,idx):
    degrees = np.array([all_degrees[i] for i in idx])
    argsort = np.argsort(degrees)[::-1]
    return np.array(idx)[argsort]

def shuffle_groups(degrees, groups):
    new_groups = [sort_by_degree(degrees, group) for group in groups.copy()]
    np.random.shuffle(new_groups)
    if isinstance(new_groups[0],list):
        new_idx = sum(new_groups,start = [])
    else:
        new_idx = np.concatenate(new_groups)
    return new_idx

def count_colors(colors):
    return len(set(colors))
    

def reordered_greedy(edges, n_points, random_state = None, trials = 2000):
    # Set Seed value
    np.random.seed(random_state)
    
    # Get degree of each node
    degrees = get_degrees(edges, n_points)
    
    # Create a dictionary of connections for each node
    edge_dict = {i: set() for i in range(n_points)}
    for v1, v2 in edges:
        edge_dict[v1].add(v2)
        edge_dict[v2].add(v1)
    
    # Get initial solution using greedy
    solution = greedy_solver(edges, n_points)
    n_colors = count_colors(solution)
    idx = np.argsort(degrees)[::-1]
    n_greedy = n_colors
    
    pbar = trange(trials)
    for i in pbar:
        groups = [np.argwhere(solution==c).flatten() for c in range(n_colors)]
        new_groups = shuffle_groups(degrees,groups)
        solution = semi_greedy_solver(edges,n_points,degrees,edge_dict,new_groups)
        n_colors = count_colors(solution)
        desc = f'#{i+1:04.0f} \t{n_colors}/{n_greedy}'
        pbar.set_description(desc, refresh=True)
        
    return solution
        


# -

reordered_greedy(edges,n_points,1,4000)

# %debug

np.concatenate([np.array([1,2]),np.array([3,4,8])])


