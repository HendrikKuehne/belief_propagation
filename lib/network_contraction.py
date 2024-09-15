import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from graph_creation import short_loop_graph,regular_graph,bipartite_regular_graph
from utils import crandn,delta_network,network_sanity_check,dummynet1,dummynet2,dummynet3,dummynet4

def contract_edge(node1:int,node2:int,key:int,G:nx.MultiGraph) -> None:
    """
    Contracts the edge `(node1,node2,key)` in the tensor network `G`. `G` is modified in-place.
    `node2` is removed, and the new node assumes the label `node1`.
    """
    if node1 == node2:
        # trace edge

        # initialization
        i1,i2 = G[node1][node2][key]["indices"]
        update_leg = lambda i: i - (i >= i1) - (i >= i2)

        # contracting the tensor and inserting in node1
        G.nodes[node1]["T"] = np.trace(G.nodes[node1]["T"],axis1=i1,axis2=i2)

        # removing the contracted edge
        G.remove_edge(node1,node2,key)

        # updating leg entries incident to the node
        for _,neighbor,key1 in G.edges(node1,keys=True):
            assert node1 == _
            if G[node1][neighbor][key1]["trace"]:
                # trace edge on node1: Both legs need to be updated
                G[node1][neighbor][key1]["indices"] = {update_leg(i) for i in G[node1][neighbor][key1]["indices"]}
            else:
                old_leg = G[node1][neighbor][key1]["legs"][node1]
                G[node1][neighbor][key1]["legs"][node1] = update_leg(old_leg)

        return

    # initialization
    i1 = G[node1][node2][key]["legs"][node1]
    i2 = G[node1][node2][key]["legs"][node2]
    nLegs1 = len(G.nodes[node1]["T"].shape)
    update_leg1 = lambda i: i if i < i1 else i - 1
    update_leg2 = lambda i: nLegs1 - 1 + i if i < i2 else nLegs1 - 1 + i - 1

    # contracting the tensors, and inserting the new tensor in node1
    T_res = np.tensordot(G.nodes[node1]["T"],G.nodes[node2]["T"],(i1,i2))
    G.nodes[node1]["T"] = T_res

    # removing the contracted edge
    G.remove_edge(node1,node2,key)

    # updating leg entries incident to node1
    for _,neighbor,key1 in G.edges(node1,keys=True):
        assert node1 == _
        if G[node1][neighbor][key1]["trace"]:
            # trace edge on node1: Both legs need to be updated
            G[node1][neighbor][key1]["indices"] = {update_leg1(i) for i in G[node1][neighbor][key1]["indices"]}
        else:
            old_leg = G[node1][neighbor][key1]["legs"][node1]
            G[node1][neighbor][key1]["legs"][node1] = update_leg1(old_leg)

    # re-wiring edges incident to node2 to connect to node1
    for _,neighbor,key2 in G.edges(node2,keys=True):
        assert node2 == _
        if node2 == neighbor:
            # trace edge on node2, needs to be transferred to node1
            keyvals = {
                "legs":None,
                "trace":True,
                "indices":{update_leg2(i) for i in G[node2][neighbor][key2]["indices"]},
            }
            # adding the new edge
            G.add_edge(node1,node1,**keyvals)
        elif node1 == neighbor:
            # edge between node1 and node2 becomes trace edge on node1
            keyvals = {
                "legs":None,
                "trace":True,
                "indices":{
                    G[node2][neighbor][key2]["legs"][node1],                    # this leg was updated when the legs incident to node1 were updated
                    update_leg2(G[node2][neighbor][key2]["legs"][node2]),
                },
            }
            # adding the new edge
            G.add_edge(node1,neighbor,**keyvals)
        else:
            # edge from another node to node2
            old_leg = G[node2][neighbor][key2]["legs"][node2]
            new_leg = update_leg2(old_leg)

            keyvals = {
                "legs":{
                    node1:new_leg,
                    neighbor:G[node2][neighbor][key2]["legs"][neighbor]
                },
                "trace":False,
                "indices":None,
            }
            # adding the new edge
            G.add_edge(node1,neighbor,**keyvals)

    # removing node2
    G.remove_node(node2)

    return

def construct_network(G:nx.MultiGraph,chi:int,rng:np.random.Generator=np.random.default_rng(),real:bool=True,psd:bool=False) -> None:
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
        # each edge has a "legs" key, whose value is itself a dictionary. The keys are the labels of the adjacent nodes, and their values are the indices of the tensor legs this edge connects
        G[edge[0]][edge[1]][0]["legs"] = {}
        # each ede has a "trace" key, which is true if this edge corresponds to the trace of a tensor (i.e. if this edge connects a node to itself)
        G[edge[0]][edge[1]][0]["trace"] = False
        # each edge has an "indices" key, which holds the legs that the adjacent tensors are summed over as a set (only used for edges that represent a trace)
        G[edge[0]][edge[1]][0]["indices"] = None

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

def contract_network(G:nx.MultiGraph,sanity_check:False) -> float:
    """
    Contracts the tensor network `G`. The contraction order is random; this might be incredibly inefficient.
    """
    while G.number_of_edges() > 0:
        iEdge = np.random.randint(G.number_of_edges())
        node1,node2,key = list(G.edges(keys=True))[iEdge]
        contract_edge(node1,node2,key,G)
        if sanity_check: assert network_sanity_check(G)

    return G.nodes(data=True)[0]["T"]

if __name__ == None:#"__main__":
    G = short_loop_graph(10,3,.5)
    #construct_network(G,3)
    delta_network(G,3)
    print("Network intact?",network_sanity_check(G))

    res = contract_network(G)
    print(res.shape)

if __name__ == "__main__":
    G,refval = dummynet4()

    cntr = contract_network(G,sanity_check=True)
    print(np.isclose(cntr,refval))

    #nx.draw(G,with_labels=True,font_weight="bold")
    #plt.show()