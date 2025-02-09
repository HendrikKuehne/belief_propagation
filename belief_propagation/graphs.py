"""
Creation of various graphs.
"""
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import itertools

def regular_graph(nNodes:int,D:int,maxiter:int=1000,verbose:bool=False) -> nx.MultiGraph:
    """
    Generates a `D`-regular graph with `nNodes` nodes. WORK IN PROGRESS; this algorithm might not terminate,
    which is why I have included an ugly brake that re-initializes the graph and starts again. Off the top of
    my head I don't know how to generate a D-regular graph, and it is not as important for me to look this up.
    """
    # sanity check
    if nNodes < D + 1 or (nNodes * D) % 2 == 1: raise ValueError(f"There is no {D}-regular graph with {nNodes} nodes.")

    # defining edges
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
                if verbose: print(f"Algorithm has not terminated after {maxiter} iterations; starting again.")
                stubs = D * [node for node in range(nNodes)]
                edges = []
                break

        if i < maxiter:
            stubs.remove(node1)
            stubs.remove(node2)

            edges += [{node1,node2},]

    G = nx.MultiGraph(incoming_graph_data=edges)
    return G

def bipartite_regular_graph(nNodes:int,D:int,maxiter:int=1000,verbose:bool=False) -> nx.MultiGraph:
    """
    Algorithm from Kirkley, 2021 ([Sci. Adv. 7, eabf1211 (2021)](https://doi.org/10.1126/sciadv.abf1211)), which
    generates a bipartite, regular graph.
    """
    # blue nodes' labels run from 0 to nNodes, red nodes' labels run from nNodes to 2*nNodes
    blue_stubs = D * [node for node in range(nNodes)]
    red_stubs = D * [node + nNodes for node in range(nNodes)]
    edges = []

    while len(blue_stubs) > 0:
        blue_node = np.random.choice(blue_stubs)
        red_node = np.random.choice(red_stubs)

        i = 0
        while {blue_node,red_node} in edges:
            blue_node = np.random.choice(blue_stubs)
            red_node = np.random.choice(red_stubs)

            i += 1
            if i > maxiter:
                if verbose: print(f"Algorithm has not terminated after {maxiter} iterations; starting again.")
                blue_stubs = D * [node for node in range(nNodes)]
                red_stubs = D * [node + nNodes for node in range(nNodes)]
                edges = []
                break

        if i <= maxiter:
            if len({blue_node,red_node}) == 1:
                print({blue_node,red_node})
            edges += [{blue_node,red_node},]
            blue_stubs.remove(blue_node)
            red_stubs.remove(red_node)

    G = nx.MultiGraph(incoming_graph_data=edges)
    return G

def short_loop_graph(nNodes:int,D:int,p:float=0,verbose:bool=False) -> nx.MultiGraph:
    """
    Algorithm from Kirkley, 2021 ([Sci. Adv. 7, eabf1211 (2021)](https://doi.org/10.1126/sciadv.abf1211)), which
    generates a network with few short primitive cycles.
    """
    # sanity check
    if p > 1 or p < 0: raise ValueError("p must be a value between zero and one.")

    # initial bipartite regular graph
    biG = bipartite_regular_graph(nNodes,D,verbose=verbose)

    edges = []
    for red_node in np.arange(nNodes,2*nNodes):
        # projecting onto the blue nodes
        for blue1,blue2 in itertools.combinations(biG.adj[red_node],r=2):
            if {blue1,blue2} not in edges: edges += [{blue1,blue2},]
        
        # removing the red node
        biG.remove_node(red_node)

    biG.add_edges_from(edges)

    # removing some of the edges randomly
    for iRemoval in range(int(p * biG.number_of_edges())):
        iEdge = np.random.randint(low=0,high=biG.number_of_edges())
        edge = list(biG.edges)[iEdge]
        biG.remove_edge(*edge)

    # extracting the largest connected component
    largest_cc = max(nx.connected_components(biG), key=len)

    return biG.subgraph(largest_cc).copy()
    # we need to copy because this removes the freeze of the subgraph

def loop_capped_graph(nNodes:int,maxlength:int,p:float=.5,rng:np.random.Generator=None) -> nx.MultiGraph:
    """
    Generates a graph that is globally tree-like. This is achieved by constructing a tree composed of
    clusters of nodes, where each cluster obeys the cycle maximum length. The parameter `p`
    determines cluster connectivity. For `p=1`, clusters are fully connected; for `p=0`, a tree is
    returned.
    """
    if p == 0: return tree(nNodes)

    # initialization
    if rng is None: rng = np.random.default_rng()
    iNode = 0
    clusters = []

    nodes = set()
    # generating clusters
    while len(nodes) < nNodes:
        cluster = [(iNode + i,iNode + j) for i,j in itertools.combinations(range(maxlength),r=2) if rng.uniform() <= p]
        if len(cluster) == 0: continue
        nodes = nodes.union(*[set(edge) for edge in cluster])
        iNode += maxlength
        clusters += [cluster,]

    # adding all clusters to a graph and connecting them tree-like
    G = nx.MultiGraph()
    G.add_edges_from(clusters.pop(0))

    for cluster in clusters:
        old_docking_node = rng.choice(np.array(G.nodes()))
        new_docking_node = rng.choice(list(set().union(*[set(edge) for edge in cluster])))
        G.add_edges_from(cluster + [(old_docking_node,new_docking_node,)])

        # do we add a new edge or simply merge an existing node and a new one?
        if rng.uniform() < .5: G = nx.contracted_edge(G,(old_docking_node,new_docking_node),self_loops=False)

    # extracting the largest connected component
    largest_cc = max(nx.connected_components(G), key=len)

    return G.subgraph(largest_cc).copy()
    # we need to copy because this removes the freeze of the subgraph

def global_loop(global_cycle_length:int,nNodes:int,maxlength:int,p:float=.5,rng:np.random.Generator=None) -> nx.MultiGraph:
    """
    Generates a graph that is globally tree-like. This is achieved by constructing a tree composed of
    clusters of nodes, where each cluster obeys the cycle maximum length.
    """
    # initialization
    if rng is None: rng = np.random.default_rng()
    clusters = [[(i,(i+1) % global_cycle_length) for i in range(global_cycle_length)],]
    iNode = global_cycle_length

    nodes = {i for i in range(global_cycle_length)}
    # generating clusters
    while len(nodes) < nNodes + global_cycle_length:
        cluster = [(iNode + i,iNode + j) for i,j in itertools.combinations(range(maxlength),r=2) if rng.uniform() <= p]
        if len(cluster) == 0: continue
        nodes = nodes.union(*[set(edge) for edge in cluster])
        iNode += maxlength
        clusters += [cluster,]

    # adding all clusters to a graph and connecting them tree-like
    G = nx.MultiGraph()
    G.add_edges_from(clusters.pop(0))

    for cluster in clusters:
        old_docking_node = rng.choice(np.array(G.nodes()))
        new_docking_node = rng.choice(list(set().union(*[set(edge) for edge in cluster])))
        G.add_edges_from(cluster + [(old_docking_node,new_docking_node,)])

        # do we add a new edge or simply merge an existing node and a new one?
        if rng.uniform() < .5: G = nx.contracted_edge(G,(old_docking_node,new_docking_node),self_loops=False)

    # extracting the largest connected component
    largest_cc = max(nx.connected_components(G), key=len)

    return G.subgraph(largest_cc).copy()
    # we need to copy because this removes the freeze of the subgraph

def tree(nNodes:int,rng:np.random.Generator=None) -> nx.MultiGraph:
    """
    Generates a tree by appending nodes at random to the tree.
    """
    if rng is None: rng = np.random.default_rng()

    not_connected = [i for i in range(1,nNodes)]
    connected = [0]
    G = nx.MultiGraph()
    G.add_node(0)

    while len(not_connected) > 0:
        node = rng.choice(not_connected)
        neighbor = rng.choice(connected)
        G.add_edge(node,neighbor)
        connected += [node,]
        not_connected.remove(node)

    return G

def hex(m:int,n:int) -> nx.MultiGraph:
    """
    Hexagonal graph.
    """
    G = nx.hexagonal_lattice_graph(m=m,n=n,create_using=nx.MultiGraph,with_positions=False)

    # removing the pos key
    for node in G.nodes():
        try:
            del G.nodes[node]["pos"]
        except KeyError:
            continue

    # re-labeling nodes
    mapping = {label:i for i,label in enumerate(G.nodes())}
    G = nx.relabel_nodes(G,mapping)

    return G

def heavyhex(m:int,n:int) -> nx.MultiGraph:
    """
    Heavy-hex graph, as defined in [Phys. Rev. X 10, 011022 (2020)](https://doi.org/10.1103/PhysRevX.10.011022).
    """
    G = hex(m=m,n=n)
    N = G.number_of_nodes()

    edges_to_add = ()
    edges_to_remove = ()
    # adding qubits to the edges
    for node1,node2 in G.edges():
        edges_to_add += ((node1,N),(N,node2))
        edges_to_remove += ((node1,node2),)
        N += 1
    G.add_edges_from(edges_to_add)
    G.remove_edges_from(edges_to_remove)

    return G

def line(N:int) -> nx.MultiGraph:
    """Exactly what you think it is."""
    G = nx.MultiGraph()
    if N == 0: return G
    if N == 1:
        G.add_node(0)
        return G

    G.add_edges_from(tuple((i,i+1) for i in range(N-1)))
    return G

# -------------------------------------------------------------------------------
#                   plotting
# -------------------------------------------------------------------------------

def loop_hist(G:nx.MultiGraph,bin_edges:np.ndarray=()) -> tuple[np.ndarray,np.ndarray]:
    """
    Creates a histogram of the loop lengths of `G`.
    """
    # are we trying to find loops in a tree?
    if nx.is_tree(G):
        if len(bin_edges) == 0:
            return [0,],[0,np.inf]
        else:
            return [0 for i in range(len(bin_edges)-1)],bin_edges

    # investigating the cycles that occur in the network
    cycle_lengths = [len(cycle) for cycle in nx.simple_cycles(G)]
    if len(bin_edges) == 0:
        hist,edges = np.histogram(cycle_lengths,bins=np.arange(min(cycle_lengths),max(cycle_lengths)+2))
    else:
        hist,edges = np.histogram(cycle_lengths,bins=bin_edges)
    return hist,edges

def plot_loop_hist(G:nx.MultiGraph,show_plot=True) -> plt.Figure:
    """
    Plots the histogram of the loop lengths of `G`.
    """
    hist,edges = loop_hist(G)

    plt.figure("Loop length histogram")
    # investigating the cycles that occur in the network
    plt.bar(edges[:-1],hist,align="edge")
    plt.suptitle(f"Graph with {G.number_of_nodes()} nodes.")
    plt.xlabel("cycle length")
    plt.ylabel("count")
    if show_plot:
        plt.show()
        return None
    return plt.gcf()

if __name__ == "__main__":
    G = tree(10)
    G = line(10)
    #G = short_loop_graph(30,3,0.6)
    #G = heavyhex(m=3,n=2)
    print("Network created")

    # drawing the network
    plt.figure("G")
    nx.draw(G,with_labels=True,font_weight="bold")
    plt.show()
    plt.close()
