"""
Creation and processing of various graphs.
"""
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import itertools

def regular_graph(nNodes:int,D:int,maxiter:int=1000) -> nx.Graph:
    """
    Generates a `D`-regular graph with `nNodes` vertices. WORK IN PROGRESS; this algorithm might not terminate,
    which is why I have included an ugly brake that re-initializes the graph and starts again. Off the top of
    my head I just don't know how to generate a D-regular graph.
    """
    # sanity check
    if nNodes < D + 1 or (nNodes * D) % 2 == 1: raise ValueError(f"There is no {D}-regular graph with {nNodes} vertices.")

    G = nx.Graph()
    G.add_nodes_from(np.arange(nNodes))

    # adding edges
    stubs = D * [node for node in range(nNodes)]
    edges = []
    while len(stubs) > 1:
        node1 = np.random.choice(stubs)
        node2 = np.random.choice(stubs)

        i = 0
        while node1 == node2 or {node1,node2} in edges:
            node2 = np.random.choice(stubs)
            i += 1

            if i >= maxiter:
                print(f"Algorithm has not terminated after {maxiter} iterations; starting again.")
                stubs = D * [node for node in range(nNodes)]
                edges = []
                break

        if i < maxiter:
            stubs.remove(node1)
            stubs.remove(node2)

            edges += [{node1,node2},]

    G.add_edges_from(edges)

    return G

def bipartite_regular_graph(nNodes:int,D:int,maxiter:int=1000) -> nx.Graph:
    """
    Algorithm from Kirkley, 2021 ([Sci. Adv. 7, eabf1211 (2021)](https://doi.org/10.1126/sciadv.abf1211)), which
    generates a bipartite, regular graph.
    """
    blue_nodes = [(f"b{node}",{"color":"blue"}) for node in range(nNodes)]
    red_nodes = [(f"r{node}",{"color":"red"}) for node in range(nNodes)]

    blue_stubs = D * [node for node in range(nNodes)]
    red_stubs = D * [node for node in range(nNodes)]
    edges = []

    while len(blue_stubs) > 0:
        blue_node = np.random.choice(blue_stubs)
        red_node = np.random.choice(red_stubs)

        i = 0
        while {f"b{blue_node}",f"r{red_node}"} in edges:
            blue_node = np.random.choice(blue_stubs)
            red_node = np.random.choice(red_stubs)

            i += 1
            if i > maxiter:
                print(f"Algorithm has not terminated after {maxiter} iterations; starting again.")
                blue_stubs = D * [node for node in range(nNodes)]
                red_stubs = D * [node for node in range(nNodes)]
                edges = []
                break

        if i <= maxiter:
            edges += [{f"b{blue_node}",f"r{red_node}"},]
            blue_stubs.remove(blue_node)
            red_stubs.remove(red_node)

    G = nx.Graph()
    G.add_nodes_from(blue_nodes + red_nodes)
    G.add_edges_from(edges)

    return G

def short_loop_graph(nNodes:int,D:int,p:float=0) -> nx.Graph:
    """
    Algorithm from Kirkley, 2021 ([Sci. Adv. 7, eabf1211 (2021)](https://doi.org/10.1126/sciadv.abf1211)), which
    generates a network with short primitive cycles.
    """
    # sanity check
    if p > 1 or p < 0: raise ValueError("p must be a value between zero and one.")

    # initial bipartite regular graph
    biG = bipartite_regular_graph(nNodes,D)

    edges = []
    for red_node in range(nNodes):
        # projecting onto the blue nodes
        for blue1,blue2 in itertools.combinations(biG.adj[f"r{red_node}"],r=2):
            if {blue1,blue2} not in edges: edges += [{blue1,blue2},]
        
        # removing the red node
        biG.remove_node(f"r{red_node}")

    biG.add_edges_from(edges)

    # removing some of the edges randomly
    for iRemoval in range(int(p * biG.number_of_edges())):
        iEdge = np.random.randint(low=0,high=biG.number_of_edges())
        edge = list(biG.edges)[iEdge]
        biG.remove_edge(*edge)

    return biG

if __name__ == "__main__":
    loopyG = short_loop_graph(70,3,.6)
    print("Network created")

    #nx.draw(loopyG,with_labels=True,font_weight="bold")
    #plt.show()

    cycle_lengths = [len(cycle) for cycle in nx.simple_cycles(loopyG)]
    plt.hist(cycle_lengths,bins=max(cycle_lengths)-min(cycle_lengths))
    plt.show()
