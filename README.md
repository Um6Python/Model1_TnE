# Model Summary: 
Our model is a computational model of cooperation in directed networks used to model the vulnerability of terrorist cells to perturbations from outside actors, such as intelligence agencies or local authorities. Following (LITERATURE ON DIRECTED GRAPHS), we identify stable configurations of directed graphs with cycles, thought to mimic the properties of terrorist cell configurations (LITERATURE ON TERRORIST CELLS). Using the prisoner’s dilemma, a game theoretic model in which individuals are expected to forgo defecting from their neighbors to an external authority, we identify stable configurations of directed graphs, and attempt to “flip” individuals in the network to identify targets for external intervention. 

The model is an agent-based simulation in which agents, representing individuals within terrorist cells or illicit criminal networks, are configured as nodes on a directed network. Agents repeatedly interact with neighbors playing a Prisoner’s Dilemma game, accumulating payoffs from these interactions over time. Graphs are generated using an evolutionary algorithm which facilitates the emergence of cooperation on directed graphs; at each step in the graph generation process, links are dynamically rewired based on a mutation parameter (μ), with the fitness target being the overall levels of cooperation in the network. Graphs are then percolated, wherein single nodes are randomly selected and “flipped” to always use a DEFECT strategy; we then assess vulnerability of the graph to perturbations based on which targeted nodes lead to the lowest levels of cooperation in the network.

We measure the structure of our generated graphs, including the presence of cycles, hierarchical structure, and epicyclical structure; the levels of cooperation present in these graphs; and the levels of cooperation following interventions. On the agent-level, we measure overall payoffs to agents in the graphs, their betweenness centrality, levels of cooperation in the overall graph following an intervention on individual nodes, and the number of cycles affected by flipping a chosen node to DEFECT. Following intervention, on the network level, we measure the number of key critical nodes leading to equivalent outcomes and the time until a collapse equilibrium is found, giving us a measure of network-level resilience.

# Entities & State Variables:
Each model is comprised of agents assembled as nodes on a generated directed network. The principle dynamic is twofold: first involving the generation of large directed graphs which can sustain cooperative dynamics in a simple prisoner’s dilemma game and second involving perturbation to these networks by flipping agent strategies from a dynamic learning process to always DEFECT.
Agents in the model play a standard prisoner’s dilemma with payoffs for cooperating and defecting with partners with two strategies, COOPERATE AND DEFECT (Table 1). Each agent maintains a binary strategy state (s  {C,D}) and a continuous payoff accumulator, which stores the rewards derived from repeated Prisoner’s Dilemma interactions. At each step, agents assess the highest payoff from their connected neighbors, changing their strategy to whichever neighbor’s strategy receives the highest payoff. Agents play asynchronously, with individual agents learning and then updating their strategies at each timestep.
Table 1. Prisoner’s Dilemma Payoff Matrix


Defect
Cooperate
Defect
-6, -6
0, -10
Cooperate
-10, 0
-1, -1


# Model Process:

## Model 1: Graph Generation
A population of P directed graphs is initialized with N=100 agents in each graph. Each graph begins as a fully connected directed network (each agent connected to every other agent).
Agents then play an iterated prisoner’s dilemma game for 1,000 timesteps and a fitness is assigned to each network equal to overall levels of cooperation (ℓ) whole network.
Graphs are then selected for reproduction based on fitness (graphs with higher ℓ are more likely to be selected). Selected graphs are copied to form the next generation.
Following reproduction, each offspring graph is mutated: with probability μ, a randomly chosen directed edge is removed or replaced with a new randomly chosen directed edge.
The iterated prisoner’s dilemma is then played again on all graphs in the new generation, cooperation levels are reassessed, and the selection–mutation process repeats for multiple (50?) generations.


## Model 2: Targeted Intervention
For each evolved directed graph, agents first play the iterated prisoner’s dilemma game for 1,000 timesteps and the baseline level of cooperation (ℓ) is recorded.
Graphs are then percolated by randomly selecting a single node and permanently flipping that node to an always-DEFECT strategy. All other agents retain their original update rules.
The iterated prisoner’s dilemma is then played again for 1,000 timesteps and the new level of cooperation (ℓ​) is recorded.
This process is repeated by resetting the graph and targeting different nodes, one at a time (100 processes for each N=100 graph).
Data Collected
Graph Generation: Data assessed for the graph generation process will include the overall level of cooperation in each graph (ℓ), the fraction of cooperators over time (ϕ), average payoff of the overall population, and network properties including path length, number and size of cycles in the network, hierarchy, centralization, and out and in-degree distributions.

Targeted Intervention: Data assessed for perturbation on the generated graphs will include the eigenvector and betweenness centrality and in/out degrees of targeted nodes, time to reach a new equilibrium following perturbation, overall levels of cooperation (ℓ), the fraction of cooperators over time (ϕ), average payoff of the overall population, and the fraction of cycles in the network in collapse took place.


#Model iteration ideas:
Weight the edges based on interaction probability and mutate this
Seed with “psychological” profiles or” susceptibility to flip” 
