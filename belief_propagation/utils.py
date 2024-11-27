"""
Random stuff that is useful here or there.
"""
import numpy as np
import networkx as nx
import warnings
import matplotlib.pyplot as plt

def crandn(size=None,rng:np.random.Generator=np.random.default_rng()) -> np.ndarray:
    """
    Draw random samples from the standard complex normal (Gaussian) distribution.
    """
    # 1/sqrt(2) is a normalization factor
    return (rng.normal(size=size) + 1j*rng.normal(size=size)) / np.sqrt(2)

def delta_tensor(nLegs:int,chi:int) -> np.ndarray:
    T = np.zeros(shape = nLegs * [chi])
    idx = nLegs * (np.arange(chi),)
    T[idx] = 1
    return T

def multi_kron(*ops):
    """
    Tensor product of all the given operators.
    """
    res_op = 1
    for op in ops: res_op = np.kron(op,res_op)
    return res_op

def proportional(A:np.ndarray,B:np.ndarray,decimals:int=None,verbose:bool=False) -> bool:
    """
    Returns `True` if `A` and `B` are proportional to each other.
    Zero is defined to be proportional to zero.

    This is accurate up to `decimals` decimal places.

    raises `ValueError` if `A` and `B` have different shapes.
    """
    if np.isnan(A).any() or np.isnan(B).any():
        warnings.warn("A or B contain NaN, and I don't know what happes then.")

    if not A.shape == B.shape:
        raise ValueError("A and B must have the same shapes.")

    if np.allclose(A,0) and np.allclose(B,0):
        warnings.warn("Assuming zero to be proportional to zero.")
        return True

    A0 = A.flatten()[np.logical_not(np.isclose(A.flatten(),0))]
    B0 = B.flatten()[np.logical_not(np.isclose(B.flatten(),0))]

    if not A0.shape == B0.shape:
        if verbose: print("A and B have different amounts of zeros.")
        return False

    div = (A0 / B0)

    if decimals != None:
        div = np.unique(np.round(div,decimals=decimals))
    else:
        div = np.unique(div)
    if len(div) != 1:
        if verbose: print("There is no unique proportionality factor.")
        return False

    return np.allclose(div[0] * B,A)

def is_hermitian(A): return np.allclose(A,A.T.conj())

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

    # all edges in the graph accounted for?
    for node in G.nodes:
        legs = [leg for leg in range(len(G.adj[node]))]
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
    Checks if the network is intact, and verifies that there are
    no double edges and no trace edges. If messages are present,
    checks if there is one message in each direction on each edge.
    """
    if not network_intact_check(G): return False

    for node1,node2,key,data in G.edges(keys=True,data=True):
        if node1 == node2:
            warnings.warn(f"Trace edge from {node1} to {node2}.")
            return False
        if key != 0:
            warnings.warn(f"Multiple edges connecting {node1} and {node2}.")
            return False
        if "msg" in data.keys():
            if data["msg"] != {}:
                if not node1 in data["msg"].keys() or not node2 in data["msg"].keys():
                    warnings.warn(f"Wrong nodes in msg-value of edge ({node1},{node2}).")
                    return False
                if len(data["msg"].values()) != 2:
                    warnings.warn(f"Wrong number of messages on edge ({node1},{node2}).")
                    return False

    return True

if __name__ == "__main__":
    pass
