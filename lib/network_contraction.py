import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from graph_creation import short_loop_graph,regular_graph,bipartite_regular_graph
from utils import crandn,delta_network,network_sanity_check

# NOT ADAPTED TO MULTIGRAPH CLASS
def contract_edge(node1:int,node2:int,G:nx.Graph) -> None:
    """
    Contracts the nodes `node1` and `node2` in the tensor network `G`. `G` is modified in-place.
    
    Every edge in the network corresponds to the contraction of the tensors in the adjacent nodes. To execute this contraction, we need to know
    which legs of the tensors correspond to the edge. This is what the attribute `legs` of the edges gives: The edge `(node1,node2)` has
    `legs = {node1:i1,node2:i2}`, which means that the contraction runs over the `i1`th leg of `node1` and the `i2`th leg of `node2`. This function thus has
    to accomplish the following:
    * contract the tensors,
    * and update the values in `legs`.

    We first contract the tensors, then we re-wire the legs from `node2` to connect to `node2`, and re-label the legs. Afterwards, node `node2` is deleted.
    """
    # initialization
    i1 = G[node1][node2]["legs"][node1]
    i2 = G[node1][node2]["legs"][node2]
    nLegs1 = len(G.nodes[node1]["T"].shape)

    # contracting the tensors, and inserting the new tensor in node1
    T_res = np.tensordot(G.nodes[node1]["T"],G.nodes[node2]["T"],(i1,i2))
    G.nodes[node1]["T"] = T_res

    # removing the contracted edge
    G.remove_edge(node1,node2)

    # updating leg entries incident to node1
    for node in G.adj[node1]:
        old_leg = G[node][node1]["legs"][node1]
        G[node1][node]["legs"][node1] = old_leg if old_leg < i1 else old_leg - 1

    # re-wiring edges incident to node2 to connect to node1
    for node in G.adj[node2]:
        G.add_edge(node1,node,legs=G[node][node2]["legs"])

        # new legs entry
        old_leg = G[node][node2]["legs"][node2]
        G[node1][node]["legs"][node1] = nLegs1 - 1 + old_leg if old_leg < i2 else nLegs1 - 1 + old_leg - 1

        # removing the old legs entry
        G[node1][node]["legs"].pop(node2)

    # removing node2
    G.remove_node(node2)

def construct_network(
    G:nx.MultiGraph,
    chi:int,
    rng:np.random.Generator=np.random.default_rng(),
    real:bool=True,
    psd:bool=False
) -> None:
    """
    Constructs a tensor network with bond dimension `chi`, where the topology is taken from the graph `G`.
    The graph `G` is manipulated in-place.
    """
    # random number generation
    if real:
        randn = lambda size: rng.standard_normal(size)
    else:
        randn = lambda size: crandn(size,rng)

    for edge in G.edges:
        # each edge carries with it an indices dictionary. The keys are the labels
        # of the adjacent nodes, and their values are the indices of the tensor legs this edge connects
        G[edge[0]][edge[1]][0]["legs"] = {}

    for node in G.nodes:
        nLegs = len(G.adj[node])
        dim = nLegs * [chi]
        # constructing a new tensor
        if psd:
            h = int(np.sqrt(chi))
            assert h**2 == chi, "if psd=True, chi must have an integer root."
            s = randn(size = nLegs * [h,] + [chi,])
            T = np.einsum(
                s, [2*i for i in range(nLegs)] + [2*nLegs,],
                s.conj(), [2*i+1 for i in range(nLegs)] + [2*nLegs,],
                np.arange(2*nLegs)
            ).reshape(dim) / chi**(3/4)
        else:
            T = randn(size = dim) / chi**(3/4)

        # adding the tensor to this node
        G.nodes[node]["T"] = T

        # adding to the adjacent edges which index they correspond to
        for i,neighbor in enumerate(G.adj[node]):
            G[node][neighbor][0]["legs"][node] = i

# NOT ADAPTED TO MULTIGRAPH CLASS
def contract_network(G:nx.Graph) -> float:
    """
    Contracts the tensor network `G`.
    """
    while G.number_of_edges() > 0:
        print(G.number_of_edges())
        print("Network intact?",network_sanity_check(G))
        node1,node2 = list(G.edges)[0]
        contract_edge(node1,node2,G)

    return G.nodes.data("T")

if __name__ == "__main__":
    G = short_loop_graph(10,3,.5)
    #construct_network(G,3)
    delta_network(G,3)
    print("Network intact?",network_sanity_check(G))

    #res = contract_network(G)
    #print(res.shape)