import numpy as np
import networkx as nx
import warnings

def delta_tensor(nLegs:int,chi:int) -> np.ndarray:
    T = np.zeros(shape = nLegs * [chi])
    idx = nLegs * (np.arange(chi),)
    T[idx] = 1
    return T

def crandn(size=None,rng:np.random.Generator=None):
    """
    Draw random samples from the standard complex normal (Gaussian) distribution.
    """
    if rng is None:
        rng = np.random.default_rng()
    # 1/sqrt(2) is a normalization factor
    return (rng.normal(size=size) + 1j*rng.normal(size=size)) / np.sqrt(2)

def delta_network(G:nx.graph,chi:int,) -> None:
    """
    Constructs a tensor network with bond dimension `chi`, where the topology is taken from the graph `G`.
    Each tensor is a delta-tensor, or a unit vector with only one non-zero entry. The graph `G` is manipulated in-place.
    """
    # random number generation
    for edge in G.edges:
        # each edge carries with it an indices dictionary. The keys are the labels
        # of the adjacent nodes, and their values are the indices of the tensor legs this edge connects
        G[edge[0]][edge[1]][0]["legs"] = {}

    for node in G.nodes:
        nLegs = len(G.adj[node])
        dim = nLegs * [chi]

        # adding the tensor to this node
        G.nodes[node]["T"] = delta_tensor(nLegs,chi)

        # adding to the adjacent edges which index they correspond to
        for i,neighbor in enumerate(G.adj[node]):
            G[node][neighbor][0]["legs"][node] = i

def network_sanity_check(G:nx.Graph) -> bool:
    """
    Checks if the given tensor network `G` is intact.
    """
    # as many tensor legs as adjacent nodes?
    for node in G.nodes:
        if len(G.nodes[node]["T"].shape) != len(G.edges(node)):
            warnings.warn(f"Unequal number of tensor legs and neighbors in node {node}.")
            return False

    # two legs in every edge's legs attribute?
    for node1,node2,key in G.edges(keys=True):
        if len(G[node1][node2][key]["legs"].keys()) != 2:
            warnings.warn(f"Wrong number of legs in edge ({node1},{node2},{key}).")
            return False

    # correct leg indices around each node?
    for node in G.nodes:
        legs = [leg for leg in range(len(G.nodes[node]["T"].shape))]
        for node1,node2,key in G.edges(node,keys=True):
            try:
                legs.remove(G[node1][node2][key]["legs"][node])
            except ValueError:
                warnings.warn(f"Wrong leg in edge ({node1},{node2},{key}).")
                return False

    return True

if __name__ == "__main__":
    pass