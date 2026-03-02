"""
@Change Log:
- changed mutation rate to 0.001
- changed the ipd function to just initialize strategies on the graph passed in instead of overwriting them
"""

"""
@OUTPUT
Generation 0: mean ℓ = 0.076
Generation 1: mean ℓ = 0.052
Generation 2: mean ℓ = 0.183
Generation 3: mean ℓ = 0.192
Generation 4: mean ℓ = 0.171
Generation 5: mean ℓ = 0.186
Generation 6: mean ℓ = 0.183
Generation 7: mean ℓ = 0.155
Generation 8: mean ℓ = 0.168
Generation 9: mean ℓ = 0.188
Generation 10: mean ℓ = 0.220
Generation 11: mean ℓ = 0.177
Generation 12: mean ℓ = 0.192
Generation 13: mean ℓ = 0.173
Generation 14: mean ℓ = 0.176
Generation 15: mean ℓ = 0.158
Generation 16: mean ℓ = 0.207
Generation 17: mean ℓ = 0.202
Generation 18: mean ℓ = 0.214
Generation 19: mean ℓ = 0.194
Generation 20: mean ℓ = 0.183
Generation 21: mean ℓ = 0.206
Generation 22: mean ℓ = 0.182
Generation 23: mean ℓ = 0.155
Generation 24: mean ℓ = 0.184
Generation 25: mean ℓ = 0.177
Generation 26: mean ℓ = 0.183
Generation 27: mean ℓ = 0.160
Generation 28: mean ℓ = 0.174
Generation 29: mean ℓ = 0.193
Generation 30: mean ℓ = 0.184
Generation 31: mean ℓ = 0.188
Generation 32: mean ℓ = 0.163
Generation 33: mean ℓ = 0.188
Generation 34: mean ℓ = 0.185
Generation 35: mean ℓ = 0.161
Generation 36: mean ℓ = 0.187
Generation 37: mean ℓ = 0.217
Generation 38: mean ℓ = 0.203
Generation 39: mean ℓ = 0.196
Generation 40: mean ℓ = 0.195
Generation 41: mean ℓ = 0.178
Generation 42: mean ℓ = 0.185
Generation 43: mean ℓ = 0.169
Generation 44: mean ℓ = 0.179
Generation 45: mean ℓ = 0.182
Generation 46: mean ℓ = 0.178
Generation 47: mean ℓ = 0.188
Generation 48: mean ℓ = 0.185
Generation 49: mean ℓ = 0.203

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
R, T_payoff, S, P_payoff = 3, 5, 0, 1  # PD payoffs

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
        if random.random() < 0.001:
            new_strategies[node] = random.choice(['C', 'D'])

    for node in G.nodes():
        G.nodes[node]['strategy'] = new_strategies[node]

# -----------------------------
# Play IPD on a graph
# -----------------------------

def play_ipd(G):
    # Strategies should already be initialized when graph was created
    total_cooperation = 0

    for _ in range(T):
        for node in G.nodes():
            G.nodes[node]['payoff'] = 0

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

    max_possible = T * G.number_of_edges()
    return total_cooperation / max_possible if max_possible > 0 else 0

def mutate_graph(G):
    #Mutate if value is greater than mu, mutate the graph
    if random.random() > mu:
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