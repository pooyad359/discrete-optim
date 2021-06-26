
import numpy as np
import matplotlib.pyplot as plt
import os
from functools import lru_cache, partial
import math
from collections import namedtuple
from tqdm.auto import tqdm, trange
from solver import load_data, view_tsp, loss
from collections import defaultdict
from joblib import Parallel, delayed

def generate_tsp(node_count, random_state=None):
    np.random.seed(random_state)
    x = np.random.randint(100, size=node_count, dtype=int)
    y = np.random.randint(100, size=node_count, dtype=int)
    points = [Point(i, j) for i, j in zip(x, y)]
    return points


def make_distance_matrix(points):
    p = np.array(points)
    dxy = (p[:, None] - p[None, :]) ** 2
    return dxy.sum(axis=2) ** 0.5

def random_route(
    distances, phermones=None, alpha=1, beta=1, starting_point=0, p_2opt=0
):
    if phermones is None:
        phermones = np.ones_like(distances)
    n = distances.shape[0]
    route = [starting_point]
    remaining_nodes = set(range(n)) - {starting_point}
    weights = phermones ** alpha / distances ** beta
    for i in range(n - 1):
        w_next = weights[route[-1], list(remaining_nodes)].copy()
        w_next /= w_next.sum()
        next_point = np.random.choice(list(remaining_nodes), p=w_next)
        remaining_nodes -= {next_point}
        route.append(next_point)
    return route


def weight_updates(routes, points, q=None, offset=0):
    n = len(routes[0])
    if q is None:
        q = n
    dw = np.zeros((n, n))
    for route in routes:
        cost = loss(route, points) - offset
        for i in range(n):
            src = route[i]
            dest = route[(i + 1) % n]
            dw[src, dest] += q / cost
    return dw


# parallel = Parallel(n_jobs=6, verbose=0)
# delayed_func = delayed(testfun)
# r = parallel(delayed_func(i) for i in range(10))


def run_ant_colony(
    points,
    phermones=None,
    ants=100,
    trials=100,
    q=None,
    evaporation=0.2,
    alpha=1,
    beta=1,
    offset=0,
    rate_2opt=0,
    random_state=None,
):
    np.random.seed(random_state)
    dist = make_distance_matrix(points)
    if phermones is None:
        phermones = np.ones_like(dist)
    history = defaultdict(list)
    pbar = trange(trials)

    best_solution = None
    best_loss = np.inf
    for generation in pbar:
        parallel = Parallel(n_jobs=6, verbose=0)
        delayed_func = delayed(random_route)
        params = {
            "distances": dist,
            "phermones": phermones,
            "alpha": 1,
            "beta": 1,
        }
        routes = parallel(delayed_func(**params) for i in range(ants))
        #         routes = [random_route(dist, phermones, alpha=1, beta=1) for _ in range(ants)]
        losses = [loss(route, points) for route in routes]
        updates = weight_updates(routes, points, q, offset)
        phermones = (1 - evaporation) * phermones + updates
        history["generation"].append(generation)
        history["min"].append(np.min(losses))
        history["max"].append(np.max(losses))
        history["mean"].append(np.mean(losses))
        pbar.set_description(f"#{generation:04.0f}\t{np.mean(losses):.2f}")
        if np.min(losses) < best_loss:
            best_solution = routes[np.argmin(losses)]
            best_loss = np.min(losses)
    return best_solution, phermones, history


def plot_history(history):
    plt.figure(figsize=(10, 6))
    plt.fill_between(history["generation"], history["min"], history["max"], alpha=0.2)
    plt.plot(history["generation"], history["mean"])
