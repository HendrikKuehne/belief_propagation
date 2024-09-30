"""
Random stuff that is useful here or there.
"""
import numpy as np
import networkx as nx
import warnings
import matplotlib.pyplot as plt

def crandn(size=None,rng:np.random.Generator=None):
    """
    Draw random samples from the standard complex normal (Gaussian) distribution.
    """
    if rng is None: rng = np.random.default_rng()
    # 1/sqrt(2) is a normalization factor
    return (rng.normal(size=size) + 1j*rng.normal(size=size)) / np.sqrt(2)

def delta_tensor(nLegs:int,chi:int) -> np.ndarray:
    T = np.zeros(shape = nLegs * [chi])
    idx = nLegs * (np.arange(chi),)
    T[idx] = 1
    return T

# -------------------------------------------------------------------------------
#                   sanity checks & diagnosis
# -------------------------------------------------------------------------------

def network_intact_check(G:nx.MultiGraph) -> bool:
    """
    Checks if the given tensor network `G` is intact.
    """
    # two legs in every edge's legs attribute?
    for node1,node2,key in G.edges(keys=True):
        if G[node1][node2][key]["trace"]:
            # trace edge
            if len(G[node1][node2][key]["indices"]) != 2:
                warnings.warn(f"Wrong number of legs in trace edge ({node1},{node2},{key}).")
                return False
        else:
            # default edge
            if len(G[node1][node2][key]["legs"].keys()) != 2:
                warnings.warn(f"Wrong number of legs in edge ({node1},{node2},{key}).")
                return False

    # correct leg indices around each node?
    for node in G.nodes:
        legs = [leg for leg in range(len(G.nodes[node]["T"].shape))]
        for node1,node2,key in G.edges(node,keys=True):
            try:
                if not G[node1][node2][key]["trace"]:
                    legs.remove(G[node1][node2][key]["legs"][node])
                else:
                    # trace edge
                    i1,i2 = G[node1][node2][key]["indices"]
                    legs.remove(i1)
                    legs.remove(i2)
            except ValueError:
                warnings.warn(f"Wrong leg in edge ({node1},{node2},{key}).")
                return False

    return True

def network_message_check(G:nx.MultiGraph) -> bool:
    """
    Verifies that there are no double edges and no trace edges in `G`. If messages are present,
    checks if there is one message in each direction on each edge.
    """
    for node1,node2,key,data in G.edges(keys=True,data=True):
        if node1 == node2:
            warnings.warn(f"Trace edge from {node1} to {node2}.")
            return False
        if key != 0:
            warnings.warn(f"Multiple edges connecting {node1} and {node2}.")
            return False
        if "msg" in data.keys():
            if not node1 in data["msg"].keys() or not node2 in data["msg"].keys():
                warnings.warn(f"Wrong nodes in msg-value of edge ({node1},{node2}).")
                return False
            if len(data["msg"].values()) != 2:
                warnings.warn(f"Wrong number of messages on edge ({node1},{node2}).")
                return False
            #if not np.isclose(np.sum(data["msg"][node1]),1):
            #    warnings.warn(f"Message to f{node1} on edge ({node1},{node2}) is not normalized.")
            #    return False
            #if not np.isclose(np.sum(data["msg"][node2]),1):
            #    warnings.warn(f"Message to {node2} on edge ({node1},{node2}) is not normalized.")
            #    return False

    return True

if __name__ == "__main__":
    pass
