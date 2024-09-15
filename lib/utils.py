"""
Random stuff that is useful here or there.
"""
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
    if rng is None: rng = np.random.default_rng()
    # 1/sqrt(2) is a normalization factor
    return (rng.normal(size=size) + 1j*rng.normal(size=size)) / np.sqrt(2)

def delta_network(G:nx.MultiGraph,chi:int,) -> None:
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

def network_sanity_check(G:nx.MultiGraph) -> bool:
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

# -------------------------------------------------------------------------------
#                   dummy tensor networks for testing contract_edge
# -------------------------------------------------------------------------------

def dummynet1(real:bool=False,chi:int=3) -> tuple[nx.MultiGraph,float]:
    """
    Returns network and contraction value. Network structure:

         4    5
         |    |
         |    |
    0 -- 1 -- 2 -- 3
    """
    rng = np.random.default_rng()

    G = nx.MultiGraph(incoming_graph_data=(
        (0,1,{"legs":{0:0,1:0},"trace":False,"indices":None}),
        (1,2,{"legs":{1:1,2:0},"trace":False,"indices":None}),
        (2,3,{"legs":{2:1,3:0},"trace":False,"indices":None}),
        (4,1,{"legs":{4:0,1:2},"trace":False,"indices":None}),
        (5,2,{"legs":{5:0,2:2},"trace":False,"indices":None}),
    ))

    G.nodes[0]["T"] = rng.standard_normal(size=(chi))         if real else crandn(size=(chi))
    G.nodes[1]["T"] = rng.standard_normal(size=(chi,chi,chi)) if real else crandn(size=(chi,chi,chi))
    G.nodes[2]["T"] = rng.standard_normal(size=(chi,chi,chi)) if real else crandn(size=(chi,chi,chi))
    G.nodes[3]["T"] = rng.standard_normal(size=(chi))         if real else crandn(size=(chi))
    G.nodes[4]["T"] = rng.standard_normal(size=(chi))         if real else crandn(size=(chi))
    G.nodes[5]["T"] = rng.standard_normal(size=(chi))         if real else crandn(size=(chi))

    # sanity check
    assert network_sanity_check(G)

    # computing the reference contraction value
    refval = np.einsum(
        "i,ijl,jkn,k,l,n->",
        G.nodes[0]["T"],
        G.nodes[1]["T"],
        G.nodes[2]["T"],
        G.nodes[3]["T"],
        G.nodes[4]["T"],
        G.nodes[5]["T"]
    )

    return G,refval

def dummynet2(real:bool=False,chi:int=3) -> tuple[nx.MultiGraph,float]:
    """
    Returns network and contraction value. Network structure:

         4 -- + 
         |    |
         |    |
    0 -- 1 -- 2 -- 3
    """
    rng = np.random.default_rng()

    G = nx.MultiGraph(incoming_graph_data=(
        (0,1,{"legs":{0:0,1:0},"trace":False,"indices":None}),
        (1,2,{"legs":{1:1,2:2},"trace":False,"indices":None}),
        (2,3,{"legs":{2:0,3:0},"trace":False,"indices":None}),
        (4,1,{"legs":{4:0,1:2},"trace":False,"indices":None}),
        (4,2,{"legs":{4:1,2:1},"trace":False,"indices":None}),
    ))

    G.nodes[0]["T"] = rng.standard_normal(size=(chi))         if real else crandn(size=(chi))
    G.nodes[1]["T"] = rng.standard_normal(size=(chi,chi,chi)) if real else crandn(size=(chi,chi,chi))
    G.nodes[2]["T"] = rng.standard_normal(size=(chi,chi,chi)) if real else crandn(size=(chi,chi,chi))
    G.nodes[3]["T"] = rng.standard_normal(size=(chi))         if real else crandn(size=(chi))
    G.nodes[4]["T"] = rng.standard_normal(size=(chi,chi))     if real else crandn(size=(chi,chi))

    # sanity check
    assert network_sanity_check(G)

    # computing the reference contraction value
    refval = np.einsum(
        "i,ijl,knj,k,ln->",
        G.nodes[0]["T"],
        G.nodes[1]["T"],
        G.nodes[2]["T"],
        G.nodes[3]["T"],
        G.nodes[4]["T"]
    )

    return G,refval

def dummynet3(real:bool=False,chi:int=3) -> tuple[nx.MultiGraph,float]:
    """
    Returns network and contraction value. Network structure:

         + -- + 
         |    |
         |    |
    0 -- 1 -- +
         |
         |
         2
    """
    rng = np.random.default_rng()

    G = nx.MultiGraph(incoming_graph_data=(
        (0,1,{"legs":{0:0,1:1},"trace":False,"indices":None}),
        (1,2,{"legs":{1:2,2:0},"trace":False,"indices":None}),
        (1,1,{"legs":None,"trace":True,"indices":{0,3}}),
    ))

    G.nodes[0]["T"] = rng.standard_normal(size=(chi))             if real else crandn(size=(chi))
    G.nodes[1]["T"] = rng.standard_normal(size=(chi,chi,chi,chi)) if real else crandn(size=(chi,chi,chi,chi))
    G.nodes[2]["T"] = rng.standard_normal(size=(chi))             if real else crandn(size=(chi))

    # sanity check
    assert network_sanity_check(G)

    # computing the reference contraction value
    refval = np.einsum(
        "i,jikj,k->",
        G.nodes[0]["T"],
        G.nodes[1]["T"],
        G.nodes[2]["T"]
    )

    return G,refval

def dummynet4(real:bool=False,chi:int=3) -> tuple[nx.MultiGraph,float]:
    """
    Returns network and contraction value. Network structure:

    0 -- 4 -- 5
         |    |
         |    |
    + -- 1 -- 2 -- 3 -- +
    |    |         |    |
    |    |         |    |
    + -- +         + -- 6
    """
    rng = np.random.default_rng()

    G = nx.MultiGraph(incoming_graph_data=(
        (0,4,{"legs":{0:0,4:2},"trace":False,"indices":None}),
        (4,5,{"legs":{4:1,5:1},"trace":False,"indices":None}),
        (4,1,{"legs":{4:0,1:3},"trace":False,"indices":None}),
        (5,2,{"legs":{5:0,2:4},"trace":False,"indices":None}),
        (1,2,{"legs":{1:2,2:0},"trace":False,"indices":None}),
        (2,3,{"legs":{2:3,3:1},"trace":False,"indices":None}),
        (3,6,{"legs":{3:2,6:0},"trace":False,"indices":None}),
        (3,6,{"legs":{3:0,6:1},"trace":False,"indices":None}),
        (1,1,{"legs":None,"trace":True,"indices":{0,1}}),
        (2,2,{"legs":None,"trace":True,"indices":{1,2}}),
    ))

    G.nodes[0]["T"] = rng.standard_normal(size=(chi))                 if real else crandn(size=(chi))
    G.nodes[1]["T"] = rng.standard_normal(size=(chi,chi,chi,chi))     if real else crandn(size=(chi,chi,chi,chi))
    G.nodes[2]["T"] = rng.standard_normal(size=(chi,chi,chi,chi,chi)) if real else crandn(size=(chi,chi,chi,chi,chi))
    G.nodes[3]["T"] = rng.standard_normal(size=(chi,chi,chi))         if real else crandn(size=(chi,chi,chi))
    G.nodes[4]["T"] = rng.standard_normal(size=(chi,chi,chi))         if real else crandn(size=(chi,chi,chi))
    G.nodes[5]["T"] = rng.standard_normal(size=(chi,chi))             if real else crandn(size=(chi,chi))
    G.nodes[6]["T"] = rng.standard_normal(size=(chi,chi))             if real else crandn(size=(chi,chi))

    # sanity check
    assert network_sanity_check(G)

    # computing the reference contraction value
    refval = np.einsum(
        "i,jjkp,krrln,qlm,poi,no,mq->",
        G.nodes[0]["T"],
        G.nodes[1]["T"],
        G.nodes[2]["T"],
        G.nodes[3]["T"],
        G.nodes[4]["T"],
        G.nodes[5]["T"],
        G.nodes[6]["T"]
    )

    return G,refval

if __name__ == "__main__":
    pass