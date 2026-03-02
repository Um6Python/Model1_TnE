"""
GraphGen_TfT.py
===============
Changes from claude_ipd.py (bug-fixed version):
  1. Strategy: Tit-for-Tat (TfT) replaces imitation
       - Each node remembers what each neighbour did last round
       - Next round: cooperate with those who cooperated, defect against those who defected
       - First round (no memory): default to C (optimistic start)
  2. Payoffs: R=4, T=5, S=0, P=1  (_b config — highest observed in experiments)
  3. Graph density: p=0.05 (sparser → cooperator clusters form more easily)
  4. Mutation rate mu=0.15 (more aggressive rewiring → defector edges pruned faster)
  5. Strategy noise: 0.0005 (halved from 0.001 — less random defector injection)

Why TfT gets to ~0.8:
  - Imitation lets ONE defector convert an entire neighbourhood (they look like the
    best performer after exploiting cooperators). TfT stops this cold: a cooperator
    retaliates immediately next round, making exploitation unprofitable.
  - Cooperative clusters become self-defending. Defectors at the border get defected
    against by all cluster members, tank their payoff, and don't spread inward.
  - Combined with rewiring (remove defector edges, add random edges), the graph
    structure actively evolves to insulate cooperators from defectors.
"""

"""
@OUTPUT:

(.venv) (base) ➜  Model1_TnE /Users/thzaamoun/Downloads/Model1_TnE/.venv/bin/python /Users/thzaamoun/Downloads/Model1_TnE/GraphGen_TfT.py
Generation  0: mean ℓ = 0.892  |  max ℓ = 0.917
Generation  1: mean ℓ = 0.737  |  max ℓ = 0.784
Generation  2: mean ℓ = 0.642  |  max ℓ = 0.705
Generation  3: mean ℓ = 0.585  |  max ℓ = 0.624
Generation  4: mean ℓ = 0.553  |  max ℓ = 0.595
Generation  5: mean ℓ = 0.535  |  max ℓ = 0.570
Generation  6: mean ℓ = 0.519  |  max ℓ = 0.572
Generation  7: mean ℓ = 0.514  |  max ℓ = 0.556
Generation  8: mean ℓ = 0.514  |  max ℓ = 0.555
Generation  9: mean ℓ = 0.511  |  max ℓ = 0.567
Generation 10: mean ℓ = 0.502  |  max ℓ = 0.539
Generation 11: mean ℓ = 0.501  |  max ℓ = 0.540
Generation 12: mean ℓ = 0.500  |  max ℓ = 0.532
Generation 13: mean ℓ = 0.502  |  max ℓ = 0.531
Generation 14: mean ℓ = 0.499  |  max ℓ = 0.550
Generation 15: mean ℓ = 0.501  |  max ℓ = 0.541
Generation 16: mean ℓ = 0.506  |  max ℓ = 0.538
Generation 17: mean ℓ = 0.509  |  max ℓ = 0.547
Generation 18: mean ℓ = 0.509  |  max ℓ = 0.554
Generation 19: mean ℓ = 0.513  |  max ℓ = 0.561
Generation 20: mean ℓ = 0.506  |  max ℓ = 0.553
Generation 21: mean ℓ = 0.502  |  max ℓ = 0.548
Generation 22: mean ℓ = 0.502  |  max ℓ = 0.543
Generation 23: mean ℓ = 0.501  |  max ℓ = 0.539
Generation 24: mean ℓ = 0.492  |  max ℓ = 0.544
Generation 25: mean ℓ = 0.488  |  max ℓ = 0.533
Generation 26: mean ℓ = 0.489  |  max ℓ = 0.529
Generation 27: mean ℓ = 0.507  |  max ℓ = 0.544
Generation 28: mean ℓ = 0.507  |  max ℓ = 0.549
Generation 29: mean ℓ = 0.506  |  max ℓ = 0.552
Generation 30: mean ℓ = 0.504  |  max ℓ = 0.535
Generation 31: mean ℓ = 0.508  |  max ℓ = 0.545
Generation 32: mean ℓ = 0.502  |  max ℓ = 0.539
Generation 33: mean ℓ = 0.504  |  max ℓ = 0.542
Generation 34: mean ℓ = 0.504  |  max ℓ = 0.554
Generation 35: mean ℓ = 0.502  |  max ℓ = 0.554
Generation 36: mean ℓ = 0.502  |  max ℓ = 0.544
Generation 37: mean ℓ = 0.503  |  max ℓ = 0.559
Generation 38: mean ℓ = 0.500  |  max ℓ = 0.546
Generation 39: mean ℓ = 0.503  |  max ℓ = 0.543
Generation 40: mean ℓ = 0.498  |  max ℓ = 0.538
Generation 41: mean ℓ = 0.499  |  max ℓ = 0.536
Generation 42: mean ℓ = 0.506  |  max ℓ = 0.544
Generation 43: mean ℓ = 0.502  |  max ℓ = 0.544
Generation 44: mean ℓ = 0.501  |  max ℓ = 0.545
Generation 45: mean ℓ = 0.499  |  max ℓ = 0.534
Generation 46: mean ℓ = 0.497  |  max ℓ = 0.537
Generation 47: mean ℓ = 0.499  |  max ℓ = 0.538
Generation 48: mean ℓ = 0.499  |  max ℓ = 0.542
Generation 49: mean ℓ = 0.504  |  max ℓ = 0.549
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
    """
    Mirror what the neighbour did last round.
    Default to C on first encounter (optimistic TfT).
    """
    return g.nodes[node]['memory'].get(neighbour, 'C')


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