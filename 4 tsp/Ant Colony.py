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

# %load_ext autoreload
# %autoreload 2
# %load_ext nb_black

# +
import numpy as np
import matplotlib.pyplot as plt
import os
from functools import lru_cache, partial
import math
from collections import namedtuple
from tqdm.auto import tqdm, trange
from solver import load_data, view_tsp, loss, to_nearest, distance
from collections import defaultdict
from ant_colony import (
    make_distance_matrix,
    run_ant_colony,
    random_route,
    plot_history,
)
from solver import (
    two_opt_v2,
    three_opt,
    randomized_opt2,
    two_opt,
    local_search,
)

Point = namedtuple("Point", ["x", "y"])


# def eucl_dist(self, point):
#     return ((self.x - point.x) ** 2 + (self.y - point.y) ** 2) ** 0.5


# Point.__sub__ = eucl_dist

ls = os.listdir
join = os.path.join

# +
n, points = load_data("tsp_100_3")

# n = 50
# points = generate_tsp(n, 12)

dist = make_distance_matrix(points)
phermones = None
# -

best_solution, phermones, history = run_ant_colony(
    points,
    phermones=phermones,
    trials=2000,
    q=1,
    evaporation=0.05,
    offset=0,
    alpha=1,
    beta=1,
)

route = random_route(dist, phermones, alpha=1, beta=1)
print(route)
plot_history(history)

view_tsp(best_solution, points, show_index=True)

refined = three_opt(best_solution, points)
refined = local_search(refined, points)
view_tsp(refined, points, show_index=False)

route = random_route(dist, phermones, alpha=0.1, beta=5)
view_tsp(route, points, show_index=True)


