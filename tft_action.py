"""
tft_action.py
===============
Changes from GraphGen_TfT.py:

TfT action

went from mirroring the behavior{1} to forgiving based on probability [1-epsilon] {2}

{1}: Mirroring the behavior
def tft_action(node, neighbour, g):
    #Mirror what the neighbour did last round.
    #Default to C on first encounter (optimistic TfT).
    return g.nodes[node]['memory'].get(neighbour, 'C')


{2}: forgiving based on probability [1-epsilon]
def tft_action(node, neighbour, g):
    last = g.nodes[node]['memory'].get(neighbour, 'C')
    if last == 'D':
        return 'D' if random.random() < epsilon else 'C'  # epsilon = 0.05
    return 'C'

"""

"""

stable cooperative Equilibrium
@OUTPUT:

(.venv) (base) ➜  Model1_TnE git:(master) ✗ /Users/thzaamoun/Downloads/Model1_TnE/.venv/bin/python /Users/thzaamoun/Downloads/Model1_TnE/tft_action.py
Generation  0: mean ℓ = 0.999  |  max ℓ = 1.000
Generation  1: mean ℓ = 0.999  |  max ℓ = 1.000
Generation  2: mean ℓ = 0.999  |  max ℓ = 1.000
Generation  3: mean ℓ = 0.999  |  max ℓ = 1.000
Generation  4: mean ℓ = 0.999  |  max ℓ = 1.000
Generation  5: mean ℓ = 0.999  |  max ℓ = 1.000
Generation  6: mean ℓ = 0.999  |  max ℓ = 1.000
Generation  7: mean ℓ = 0.999  |  max ℓ = 1.000
Generation  8: mean ℓ = 0.999  |  max ℓ = 1.000
Generation  9: mean ℓ = 0.999  |  max ℓ = 1.000
Generation 10: mean ℓ = 0.999  |  max ℓ = 1.000
Generation 11: mean ℓ = 0.999  |  max ℓ = 1.000
Generation 12: mean ℓ = 0.999  |  max ℓ = 1.000
Generation 13: mean ℓ = 0.999  |  max ℓ = 1.000
Generation 14: mean ℓ = 0.999  |  max ℓ = 1.000
Generation 15: mean ℓ = 0.999  |  max ℓ = 1.000
Generation 16: mean ℓ = 0.999  |  max ℓ = 1.000
Generation 17: mean ℓ = 0.999  |  max ℓ = 1.000
Generation 18: mean ℓ = 0.999  |  max ℓ = 1.000
Generation 19: mean ℓ = 0.999  |  max ℓ = 1.000
Generation 20: mean ℓ = 0.999  |  max ℓ = 1.000
Generation 21: mean ℓ = 0.999  |  max ℓ = 1.000
Generation 22: mean ℓ = 0.999  |  max ℓ = 1.000
Generation 23: mean ℓ = 0.999  |  max ℓ = 1.000
Generation 24: mean ℓ = 0.999  |  max ℓ = 1.000
Generation 25: mean ℓ = 0.999  |  max ℓ = 1.000
Generation 26: mean ℓ = 0.999  |  max ℓ = 1.000
Generation 27: mean ℓ = 0.999  |  max ℓ = 1.000
Generation 28: mean ℓ = 0.999  |  max ℓ = 1.000
Generation 29: mean ℓ = 0.999  |  max ℓ = 1.000
Generation 30: mean ℓ = 0.999  |  max ℓ = 1.000
Generation 31: mean ℓ = 0.999  |  max ℓ = 1.000
Generation 32: mean ℓ = 0.999  |  max ℓ = 1.000
Generation 33: mean ℓ = 0.999  |  max ℓ = 1.000
Generation 34: mean ℓ = 0.999  |  max ℓ = 1.000
Generation 35: mean ℓ = 0.999  |  max ℓ = 1.000
Generation 36: mean ℓ = 0.999  |  max ℓ = 1.000
Generation 37: mean ℓ = 0.999  |  max ℓ = 1.000
Generation 38: mean ℓ = 0.999  |  max ℓ = 1.000
Generation 39: mean ℓ = 0.999  |  max ℓ = 1.000
Generation 40: mean ℓ = 0.999  |  max ℓ = 1.000
Generation 41: mean ℓ = 0.999  |  max ℓ = 1.000
Generation 42: mean ℓ = 0.999  |  max ℓ = 1.000
Generation 43: mean ℓ = 0.999  |  max ℓ = 1.000
Generation 44: mean ℓ = 0.999  |  max ℓ = 1.000
Generation 45: mean ℓ = 0.999  |  max ℓ = 1.000
Generation 46: mean ℓ = 0.999  |  max ℓ = 1.000
Generation 47: mean ℓ = 0.999  |  max ℓ = 1.000
Generation 48: mean ℓ = 0.999  |  max ℓ = 1.000
Generation 49: mean ℓ = 0.999  |  max ℓ = 1.000
"""

import networkx as nx
import numpy as np
import random
import copy

# -----------------------------
# Hyperparameters
# -----------------------------
N     = 100     # agents per graph
P     = 50      # population size (number of graphs)
T     = 500     # IPD timesteps per fitness evaluation
G     = 50      # generations
mu    = 0.15    # graph mutation rate (rewiring probability) — up from 0.05
NOISE = 0.0005  # (per-edge & per-round random flip — down from 0.001
epsilon = 0.05 # forgiveness probability
# PD Payoffs — _b config
# Valid PD:  T > R > P > S  →  5 > 4 > 1 > 0  ✓
# No-alt:    2R > T + S     →  8 > 5           ✓
R, T_payoff, S, P_payoff = 4, 5, 0, 1

GRAPH_DENSITY = 0.05   # down from 0.1 — sparser = better clustering


# -----------------------------
# Graph creation
# -----------------------------
def create_random_graph():
    g = nx.gnp_random_graph(N, p=GRAPH_DENSITY)
    g.remove_edges_from(nx.selfloop_edges(g))
    return g


# -----------------------------
# Initialisation
# -----------------------------
def initialize_strategies(g):
    """Everyone starts as C with empty memory (optimistic)."""
    for node in g.nodes():
        g.nodes[node]['strategy'] = 'C'
        g.nodes[node]['payoff']   = 0
        g.nodes[node]['memory']   = {}  # {neighbour: last_action_they_took}


# -----------------------------
# TfT action
# -----------------------------
def tft_action(node, neighbour, g):
    last = g.nodes[node]['memory'].get(neighbour, 'C')
    if last == 'D':
        return 'D' if random.random() < epsilon else 'C'  # epsilon = 0.05
    return 'C'


# -----------------------------
# Single round
# -----------------------------
def play_round(g):
    """Play one round across all edges. Returns cooperation count for this round."""
    for node in g.nodes():
        g.nodes[node]['payoff'] = 0

    round_cooperation = 0
    new_memories = {node: {} for node in g.nodes()}

    for u, v in g.edges():
        a_u = tft_action(u, v, g)
        a_v = tft_action(v, u, g)

        # Noise: tiny chance of random flip
        if random.random() < NOISE:
            a_u = 'D' if a_u == 'C' else 'C'
        if random.random() < NOISE:
            a_v = 'D' if a_v == 'C' else 'C'

        # Payoffs
        if a_u == 'C' and a_v == 'C':
            g.nodes[u]['payoff'] += R
            g.nodes[v]['payoff'] += R
            round_cooperation += 1
        elif a_u == 'C' and a_v == 'D':
            g.nodes[u]['payoff'] += S
            g.nodes[v]['payoff'] += T_payoff
            round_cooperation += 0.5
        elif a_u == 'D' and a_v == 'C':
            g.nodes[u]['payoff'] += T_payoff
            g.nodes[v]['payoff'] += S
            round_cooperation += 0.5
        else:
            g.nodes[u]['payoff'] += P_payoff
            g.nodes[v]['payoff'] += P_payoff

        # Each node records what IT did, so the other can mirror it
        new_memories[u][v] = a_u
        new_memories[v][u] = a_v

    for node in g.nodes():
        g.nodes[node]['memory'] = new_memories[node]

    return round_cooperation


# -----------------------------
# Fitness evaluation
# -----------------------------
def play_ipd(g):
    """Run T rounds. Return ℓ = fraction of cooperative edge-interactions."""
    total_cooperation = 0
    for _ in range(T):
        total_cooperation += play_round(g)
    max_possible = T * g.number_of_edges()
    return total_cooperation / max_possible if max_possible > 0 else 0


# -----------------------------
# Graph mutation
# -----------------------------
def mutate_graph(g):
    """
    With probability mu:
      - 50%: remove a defector-involved edge (prune bad connections)
      - 50%: add a random new edge (exploration)
    Uses memory to target defector edges accurately.
    """
    if random.random() > mu:
        return

    nodes = list(g.nodes())

    if random.random() < 0.5 and g.number_of_edges() > 0:
        # Find edges where either endpoint defected last round
        defector_edges = [
            (u, v) for u, v in g.edges()
            if g.nodes[u]['memory'].get(v, 'C') == 'D'
            or g.nodes[v]['memory'].get(u, 'C') == 'D'
        ]
        # Fall back to any non-CC edge
        if not defector_edges:
            defector_edges = [
                (u, v) for u, v in g.edges()
                if not (g.nodes[u]['memory'].get(v, 'C') == 'C'
                        and g.nodes[v]['memory'].get(u, 'C') == 'C')
            ]
        if defector_edges:
            g.remove_edge(*random.choice(defector_edges))
    else:
        u, v = random.sample(nodes, 2)
        if not g.has_edge(u, v):
            g.add_edge(u, v)


# -----------------------------
# Evolutionary loop
# -----------------------------
population = []
for _ in range(P):
    g = create_random_graph()
    initialize_strategies(g)
    population.append(g)

for generation in range(G):

    fitnesses = []
    for graph in population:
        fitness = play_ipd(graph)
        fitnesses.append(fitness)

    mean_l = np.mean(fitnesses)
    max_l  = np.max(fitnesses)
    print(f"Generation {generation:>2}: mean ℓ = {mean_l:.3f}  |  max ℓ = {max_l:.3f}")

    # Fitness-proportional selection
    total_fitness = sum(fitnesses)
    probabilities = [f / total_fitness for f in fitnesses] if total_fitness > 0 else [1/P] * P

    new_population = []
    for _ in range(P):
        parent_index = np.random.choice(range(P), p=probabilities)
        offspring = copy.deepcopy(population[parent_index])
        mutate_graph(offspring)
        new_population.append(offspring)

    population = new_population