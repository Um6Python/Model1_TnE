"""
@CHANGES:
R, T_payoff, S, P_payoff = 4, 5, 0, 1
R, T_payoff, S, P_payoff = 3, 5, 0, 1
Effect:

C–C interactions become highly rewarding

Cooperative hubs stabilize

ℓ increases

Valid:

5 > 4 > 1 > 0

2R = 8 > 5
"""

"""
@OUTPUT:
(.venv) (base) ➜  Model1_TnE /Users/thzaamoun/Downloads/Model1_TnE/.venv/bin/python /Users/thzaamoun/Downloads/Model1_TnE/GraphGen_b.py
Generation 0: mean ℓ = 0.112
Generation 1: mean ℓ = 0.122
Generation 2: mean ℓ = 0.106
Generation 3: mean ℓ = 0.127
Generation 4: mean ℓ = 0.148
Generation 5: mean ℓ = 0.125
Generation 6: mean ℓ = 0.117
Generation 7: mean ℓ = 0.092
Generation 8: mean ℓ = 0.096
Generation 9: mean ℓ = 0.102
Generation 10: mean ℓ = 0.101
Generation 11: mean ℓ = 0.092
Generation 12: mean ℓ = 0.112
Generation 13: mean ℓ = 0.101
Generation 14: mean ℓ = 0.096
Generation 15: mean ℓ = 0.114
Generation 16: mean ℓ = 0.138
Generation 17: mean ℓ = 0.097
Generation 18: mean ℓ = 0.119
Generation 19: mean ℓ = 0.113
Generation 20: mean ℓ = 0.110
Generation 21: mean ℓ = 0.123
Generation 22: mean ℓ = 0.104
Generation 23: mean ℓ = 0.113
Generation 24: mean ℓ = 0.124
Generation 25: mean ℓ = 0.099
Generation 26: mean ℓ = 0.144
Generation 27: mean ℓ = 0.125
Generation 28: mean ℓ = 0.101
Generation 29: mean ℓ = 0.127
Generation 30: mean ℓ = 0.098
Generation 31: mean ℓ = 0.103
Generation 32: mean ℓ = 0.126
Generation 33: mean ℓ = 0.122
Generation 34: mean ℓ = 0.123
Generation 35: mean ℓ = 0.097
Generation 36: mean ℓ = 0.116
Generation 37: mean ℓ = 0.125
Generation 38: mean ℓ = 0.131
Generation 39: mean ℓ = 0.094
Generation 40: mean ℓ = 0.103
Generation 41: mean ℓ = 0.095
Generation 42: mean ℓ = 0.113
Generation 43: mean ℓ = 0.088
Generation 44: mean ℓ = 0.117
Generation 45: mean ℓ = 0.121
Generation 46: mean ℓ = 0.094
Generation 47: mean ℓ = 0.116
Generation 48: mean ℓ = 0.115
Generation 49: mean ℓ = 0.093
"""
import networkx as nx
import numpy as np
import random
import copy

# -----------------------------
# Initial Parameters
# -----------------------------
N = 100               # number of agents per graph
P = 50                # number of graphs in evolutionary population
T = 500              # Iterated Prisoner Dilemma timesteps
G = 50                # Generations to mutate graph
mu = 0.05             # Mutation probability of new graphs

#IPD Payoffs: C_C; D_C; C_D; D_D
R, T_payoff, S, P_payoff = 4, 5, 0, 1 # PD payoffs
# -----------------------------
# Strategy (simple baseline)
# -----------------------------=

#Initialize Strategies: Everyone starts cooperating
def initialize_strategies(G):
    for node in G.nodes():
        G.nodes[node]['strategy'] = 'C'

#Update strategies, based on payoffs of those around you
def update_strategies(G):
    new_strategies = {}
    for node in G.nodes():
        #Look at your neighbors in the network
        influencers = list(G.neighbors(node))
        #If no neighbors, just keep doing what you're doing
        if not influencers:
            new_strategies[node] = G.nodes[node]['strategy']
            continue

        #Find the best performing neighbor and copy them
        best = max(influencers, key=lambda n: G.nodes[n]['payoff'])
        #Find the best payoff around you: if payoff is higher than your payoff switch
        if G.nodes[best]['payoff'] > G.nodes[node]['payoff']:
            new_strategies[node] = G.nodes[best]['strategy']
        else:
        #Else: keep your own
            new_strategies[node] = G.nodes[node]['strategy']

    #Random mutation: You can switch strategies with a 1% chance
    #TODO: consider removing this
        ##Reason to keep: get more realistic, resilient graphs
    for node in G.nodes():
        if random.random() < 0.01:
            new_strategies[node] = random.choice(['C', 'D'])

    for node in G.nodes():
        G.nodes[node]['strategy'] = new_strategies[node]

# -----------------------------
# Play IPD on a graph
# -----------------------------
def play_ipd(G):
    population = []
    #For every graph, let's play the IPD
    for _ in range(P):
        G = create_random_graph()
        initialize_strategies(G)
        population.append(G)

    total_cooperation = 0

    for _ in range(T):

        #Reset payoffs each round so that a new game starts
        for node in G.nodes():
            G.nodes[node]['payoff'] = 0

        #For Node1 and Node2, compare strategies, and play the game to get your payoff
        for u, v in G.edges():
            s_u = G.nodes[u]['strategy']
            s_v = G.nodes[v]['strategy']

            if s_u == 'C' and s_v == 'C':
                G.nodes[u]['payoff'] += R
                G.nodes[v]['payoff'] += R
                total_cooperation += 1

            elif s_u == 'C' and s_v == 'D':
                G.nodes[u]['payoff'] += S
                G.nodes[v]['payoff'] += T_payoff
                total_cooperation += 0.5

            elif s_u == 'D' and s_v == 'C':
                G.nodes[u]['payoff'] += T_payoff
                G.nodes[v]['payoff'] += S
                total_cooperation += 0.5

            else:
                G.nodes[u]['payoff'] += P_payoff
                G.nodes[v]['payoff'] += P_payoff

        update_strategies(G)

    # ℓ = fraction of cooperative edge-interactions
    # Measuring total amount of cooperation in the network
    max_possible = T * G.number_of_edges()
    return total_cooperation / max_possible

def mutate_graph(G):
    #Mutate if value is greater than mu, mutate the graph
    if random.random() >= mu:
        return

    nodes = list(G.nodes())

    #Look at all edges and mutate half of them
    if random.random() < 0.5 and G.number_of_edges() > 0:
        #Select candidate edges: We don't want to remove mutual cooperators!
        candidate_edges = []

        #Check for mutual cooperation
        for u, v in G.edges():
            if not (G.nodes[u]['strategy'] == 'C' and
                    G.nodes[v]['strategy'] == 'C'):
                candidate_edges.append((u, v))

        #Remove edge with defector
        #TODO: Change this dynamic to add another person with some probability
        if candidate_edges:
            edge = random.choice(candidate_edges)
            G.remove_edge(*edge)

        #TODO: consider this dynamic, we're currently adding random edges if edges aren't remove
    else:
        # Add random edge
        u, v = random.sample(nodes, 2)
        G.add_edge(u, v)

def create_random_graph():
    G = nx.gnp_random_graph(N, p=0.1)  #undirected by default
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


population = []
for _ in range(P):
    g = create_random_graph()
    initialize_strategies(g)
    population.append(g)

for generation in range(G):

    fitnesses = []
    for graph in population:
        #Fitness is l variable, or the amount of cooperation in the graph
        fitness = play_ipd(graph)
        fitnesses.append(fitness)

    print(f"Generation {generation}: mean ℓ = {np.mean(fitnesses):.3f}")

    # Fitness-proportional selection
    total_fitness = sum(fitnesses)
    if total_fitness == 0:
        probabilities = [1/P] * P
    else:
        #Assign probabilities to the highest performing graphs
        probabilities = [f / total_fitness for f in fitnesses]

    new_population = []

    #Evolutionary algorithm
    for _ in range(P):
        #Select parents based on their fitness (levels of cooperation)
        parent_index = np.random.choice(range(P), p=probabilities)
        #Copy the best performing parents (value P) with probability equal to fitness
        offspring = copy.deepcopy(population[parent_index])
        #Mutate that graph
        mutate_graph(offspring)
        new_population.append(offspring)

    population = new_population