#!/usr/bin/python
# -*- coding: utf-8 -*-

from ortools.constraint_solver import pywrapcp
from ortools.sat.python import cp_model
import numpy as np
def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    edges = []
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append((int(parts[0]), int(parts[1])))

    # build a trivial solution
    # every node has its own color
    solution = range(0, node_count)

    # if node_count==50:
    #     n_colors = 6
    # elif node_count==70:
    #     n_colors=22
    # elif node_count == 100:
    #     n_colors = 20
    # elif node_count == 250:
    #     n_colors = 99
    # elif node_count == 500:
    #     n_colors = 17
    # elif node_count == 1000:
    #     n_colors = 126

    # res = fixed_color(edges,node_count,n_colors)
    n_colors, res = stepwise_opt(edges, node_count)
    # res = all_res[-1]

    # algo2(edges,node_count,nc)

    # prepare the solution in the specified output format
    output_data = str(n_colors) + ' ' + str(1) + '\n'
    output_data += ' '.join(map(str, res))
    # output_data = None
    return output_data

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

def fixed_color(edges,n_points,n_colors):
    solver = pywrapcp.Solver('Colors')
    colors = [solver.IntVar(0,n_colors-1,f'c_{i}') for i in range(n_points)]
    for i,j in edges:
        solver.Add(colors[i]!=colors[j])
    db = solver.Phase(colors,
                        solver.CHOOSE_FIRST_UNBOUND,
                        solver.ASSIGN_MIN_VALUE)
    solver.NewSearch(db)

    if solver.NextSolution():
        return [c.Value() for c in colors]
    else:
        return []

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
    t = node_count/1000*10 # 10 sec base for 1000 nodes
    t = t*pct/.05 # scale based on percentage of complete graph
    return max(10,t*2)

def node_degree(edges, n_nodes=None):
    occur = np.array(edges).flatten()
    if n_nodes is None:
        n_nodes = occur.max()
    return [len(occur[occur == o]) for o in range(n_nodes)]

import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')

