"""
Functions for network creation and handling.
"""
import numpy as np
import networkx as nx
import itertools
import cotengra as ctg
import copy

from belief_propagation.utils import network_intact_check,network_message_check,crandn,delta_tensor

# -------------------------------------------------------------------------------
#                   Manipulating and contracting networks
# -------------------------------------------------------------------------------

def contract_edge(node1:int,node2:int,key:int,G:nx.MultiGraph,sanity_check:bool=False) -> None:
    """
    Contracts the edge `(node1,node2,key)` in the tensor network `G`. `G` is modified in-place.
    `node2` is removed, and the new node assumes the label `node1`.
    """
    # sanity check
    if sanity_check: assert network_intact_check(G)

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

def contract_network(G:nx.MultiGraph,sanity_check:bool=False) -> float:
    """
    Contracts the tensor network `G` using `ctg.einsum`.
    """
    if sanity_check: assert network_intact_check(G)

    if G.number_of_nodes() == 1:
        # the network is trivial
        return np.sum(tuple(G.nodes(data="T"))[0][1])

    args = ()

    # enumerating the edges in the graph
    for i,nodes in enumerate(G.edges()):
        node1,node2 = nodes
        G[node1][node2][0]["label"] = i

    # extracting the einsum arguments
    for node,T in G.nodes(data="T"):
        args += (T,)
        legs = [None for i in range(T.ndim)]
        for _,neighbor,edge_label in G.edges(nbunch=node,data="label"):
            legs[G[node][neighbor][0]["legs"][node]] = edge_label
        args += (tuple(legs),)

    return ctg.einsum(*args,optimize="greedy")

def merge_edges(node1:int,node2:int,G:nx.multigraph,sanity_check:bool=False) -> None:
    """
    Merges all parallel edges between `node1` and `node2`. `G` is modified in-place.
    """
    if len(G[node1][node2]) < 2:
        # there are no edges we could merge
        return

    # sanity check
    if sanity_check: assert network_intact_check(G)

    axes1 = []
    """Makes up the `axes` argument for `np.transpose` together with `legs1`."""
    axes2 = []
    """Makes up the `axes` argument for `np.transpose` together with `legs2`."""
    legs1 = list(range(G.nodes[node1]["T"].ndim))
    """Makes up the `axes` argument for `np.transpose` together with `axes1`."""
    legs2 = list(range(G.nodes[node2]["T"].ndim))
    """Makes up the `axes` argument for `np.transpose` together with `axes2`."""
    newshape1 = list(G.nodes[node1]["T"].shape)
    """Part of the `shape` arhument for `np.reshape`."""
    newshape2 = list(G.nodes[node2]["T"].shape)
    """Part of the `shape` argument for `np.reshape`."""
    keys = ()
    """Keys of the parallel edges, for removal later."""

    # the tensors of node1 and node2 must be transposed s.t. duplicate edges are the leading edges
    for i,key in enumerate(G[node1][node2]):
        legs1.remove(G[node1][node2][key]["legs"][node1])
        legs2.remove(G[node1][node2][key]["legs"][node2])

        axes1 += [G[node1][node2][key]["legs"][node1],]
        axes2 += [G[node1][node2][key]["legs"][node2],]

        keys += (key,)

        newshape1[G[node1][node2][key]["legs"][node1]] = None
        newshape2[G[node1][node2][key]["legs"][node2]] = None

    newshape1 = [-1,] + [dim for dim in newshape1 if dim != None]
    newshape2 = [-1,] + [dim for dim in newshape2 if dim != None]

    # removing the parallel edges
    for key in keys: G.remove_edge(node1,node2,key)

    # transposing the tensors, s.t. the parallel edges connect to the leading dimensions
    G.nodes[node1]["T"] = np.transpose(G.nodes[node1]["T"],axes1 + legs1)
    G.nodes[node2]["T"] = np.transpose(G.nodes[node2]["T"],axes2 + legs2)

    # re-shaping the tensors
    G.nodes[node1]["T"] = np.reshape(G.nodes[node1]["T"],newshape1)
    G.nodes[node2]["T"] = np.reshape(G.nodes[node2]["T"],newshape2)

    # shifts of the leg indices for both nodes
    legshift1 = lambda i: 1 - sum([i > leg for leg in axes1])
    legshift2 = lambda i: 1 - sum([i > leg for leg in axes2])

    # updating the leg entries of the edges to the rest of the network
    for _,neighbor,key in G.edges(node1,keys=True):
        if G[node1][neighbor][key]["trace"]:
            G[node1][neighbor][key]["indices"] = {leg + legshift1(leg) for leg in G[node1][neighbor][key]["indices"]}
        else:
            G[node1][neighbor][key]["legs"][node1] += legshift1(G[node1][neighbor][key]["legs"][node1])
    for _,neighbor,key in G.edges(node2,keys=True):
        if G[node2][neighbor][key]["trace"]:
            G[node2][neighbor][key]["indices"] = {leg + legshift2(leg) for leg in G[node2][neighbor][key]["indices"]}
        else:
            G[node2][neighbor][key]["legs"][node2] += legshift2(G[node2][neighbor][key]["legs"][node2])

    # adding a new edge
    G.add_edge(node1,node2,legs={node1:0,node2:0},trace=False,indices=None)

    return

def expose_edge(G:nx.MultiGraph,node1:int,node2:int,sanity_check:bool=False) -> None:
    """
    Transposes the tensors in `node1` and `node2` such that the edge
    connecting them sums over their trailing dimensions. `G` is modified in-place.
    """
    if sanity_check:
        assert network_intact_check(G)
        assert network_message_check(G)
        assert node1 in G.adj[node2]

    axes1 = [None for _ in range(G.nodes[node1]["T"].ndim)]
    """Makes up the `axes` argument for `np.transpose` at `node1`."""
    axes2 = [None for _ in range(G.nodes[node2]["T"].ndim)]
    """Makes up the `axes` argument for `np.transpose` at `node2`."""

    axes1[-1] = G[node1][node2][0]["legs"][node1]
    axes2[-1] = G[node2][node1][0]["legs"][node2]

    legshift1 = lambda i: i > G[node1][node2][0]["legs"][node1]
    legshift2 = lambda i: i > G[node2][node1][0]["legs"][node2]

    for neighbor in G.adj[node1]:
        if neighbor == node2: continue

        old_leg = G[node1][neighbor][0]["legs"][node1]
        # assembling the axes-argument
        axes1[old_leg - legshift1(old_leg)] = old_leg
        # overwriting the leg of this edge
        G[node1][neighbor][0]["legs"][node1] = old_leg - legshift1(old_leg)
    for neighbor in G.adj[node2]:
        if neighbor == node1: continue

        old_leg = G[node2][neighbor][0]["legs"][node2]
        # assembling the axes-argument
        axes2[old_leg - legshift2(old_leg)] = old_leg
        # overwriting the leg of this edge
        G[node2][neighbor][0]["legs"][node2] = old_leg - legshift2(old_leg)

    # overwriting the legs of the edge (node1,node2)
    G[node1][node2][0]["legs"][node1] = G.nodes[node1]["T"].ndim - 1
    G[node2][node1][0]["legs"][node2] = G.nodes[node2]["T"].ndim - 1

    # permuting the legs of u and v such that edge (u,v) corresponds to the trailing index in nodes u and v
    G.nodes[node1]["T"] = np.transpose(G.nodes[node1]["T"],axes1)
    G.nodes[node2]["T"] = np.transpose(G.nodes[node2]["T"],axes2)

    return

def feynman_cut(G:nx.MultiGraph,node1:int,node2:int,sanity_check:bool=False) -> tuple[nx.MultiGraph]:
    """
    Executes a Feynman-cut on the network `G`, and returns a list that contains the respective networks.
    Assumes that the network is intact and message-ready.
    """
    expose_edge(G,node1,node2,sanity_check)

    nCuts = G.nodes[node1]["T"].shape[-1]
    cut_graphs = ()
    for i in range(nCuts):
        Gcut = copy.deepcopy(G)
        Gcut.nodes[node1]["T"] = Gcut.nodes[node1]["T"][...,i]
        Gcut.nodes[node2]["T"] = Gcut.nodes[node2]["T"][...,i]
        Gcut.remove_edge(node1,node2)
        cut_graphs += (Gcut,)

    return cut_graphs

# -------------------------------------------------------------------------------
#                   Network creation
# -------------------------------------------------------------------------------

def construct_network(G:nx.MultiGraph,chi:int,rng:np.random.Generator=np.random.default_rng(),real:bool=True,psd:bool=False):
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

    tensors = {}

    for node in G.nodes:
        nLegs = len(G.adj[node])
        dim = nLegs * [chi]
        # constructing a new tensor
        if psd:
            h = int(np.sqrt(chi))
            assert h**2 == chi, "if psd=True, chi must have an integer root."
            s = randn(size = nLegs * [h,] + [chi,]) # last dimension is physical leg
            T = np.einsum(
                s, [2*i for i in range(nLegs)] + [2*nLegs,],
                s.conj(), [2*i+1 for i in range(nLegs)] + [2*nLegs,],
                np.arange(2*nLegs)
            ).reshape(dim) / chi**(3/4)

            # saving the physical tensor
            tensors[node] = s

        else:
            T = randn(size = dim) / chi**(3/4)

            # saving the physical tensor
            tensors[node] = T

        # adding the tensor to this node
        G.nodes[node]["T"] = T

        # adding to the adjacent edges which index they correspond to
        for i,neighbor in enumerate(G.adj[node]):
            G[node][neighbor][0]["legs"][node] = i

    return tensors

def construct_initial_messages(G:nx.MultiGraph,normalize:bool=True,sanity_check:bool=False) -> None:
    """
    Initializes messages one the edges of `G`. Random initialisation except for leaf nodes, where the initial value
    is the tensor of the leaf node. `G` is modified in-place.
    """
    # sanity check
    if sanity_check: assert network_message_check(G)

    for node1,node2 in G.edges():
        # add messages to every edge, in both directions
        G[node1][node2][0]["msg"] = {}

        for receiving_node in (node1,node2):
            sending_node = node2 if receiving_node == node1 else node1
            if len(G.adj[sending_node]) == 1:
                # message from leaf node
                G[node1][node2][0]["msg"][receiving_node] = G.nodes[sending_node]["T"] / np.sum(G.nodes[sending_node]["T"]) if normalize else G.nodes[sending_node]["T"]
            else:
                iLeg = G[node1][node2][0]["legs"][receiving_node]
                # bond dimension
                chi = G.nodes[receiving_node]["T"].shape[iLeg]
                # initialization with normalized vector
                #msg = np.ones(shape=(chi,))
                msg = np.random.normal(size=(chi,))
                G[node1][node2][0]["msg"][receiving_node] = msg / np.sum(msg) if normalize else msg

    return

def delta_network(G:nx.MultiGraph,chi:int,) -> None:
    """
    Constructs a tensor network with bond dimension `chi`, where the topology is taken from the graph `G`.
    Each tensor is a delta-tensor, or a unit vector with only one non-zero entry. The graph `G` is manipulated in-place.
    """
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

        # adding the tensor to this node
        G.nodes[node]["T"] = delta_tensor(nLegs,chi)

        # adding to the adjacent edges which index they correspond to
        for i,neighbor in enumerate(G.adj[node]):
            G[node][neighbor][0]["legs"][node] = i

    return

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
    assert network_intact_check(G)

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
    assert network_intact_check(G)

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
    assert network_intact_check(G)

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
    assert network_intact_check(G)

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

def dummynet5(real:bool=False,chi:int=3) -> tuple[nx.MultiGraph,float]:
    """
    Returns network and contraction value. Network structure:

    0 -- 4 -- 5
         |    |
         |    |
         1 -- 2 -- 3 -- 6
    """
    rng = np.random.default_rng()

    G = nx.MultiGraph(incoming_graph_data=(
        (0,4,{"legs":{0:0,4:2},"trace":False,"indices":None}),
        (4,5,{"legs":{4:1,5:1},"trace":False,"indices":None}),
        (4,1,{"legs":{4:0,1:1},"trace":False,"indices":None}),
        (5,2,{"legs":{5:0,2:2},"trace":False,"indices":None}),
        (1,2,{"legs":{1:0,2:0},"trace":False,"indices":None}),
        (2,3,{"legs":{2:1,3:1},"trace":False,"indices":None}),
        (3,6,{"legs":{3:0,6:0},"trace":False,"indices":None}),
    ))

    G.nodes[0]["T"] = rng.standard_normal(size=(chi))         if real else crandn(size=(chi))
    G.nodes[1]["T"] = rng.standard_normal(size=(chi,chi))     if real else crandn(size=(chi,chi))
    G.nodes[2]["T"] = rng.standard_normal(size=(chi,chi,chi)) if real else crandn(size=(chi,chi,chi))
    G.nodes[3]["T"] = rng.standard_normal(size=(chi,chi))     if real else crandn(size=(chi,chi))
    G.nodes[4]["T"] = rng.standard_normal(size=(chi,chi,chi)) if real else crandn(size=(chi,chi,chi))
    G.nodes[5]["T"] = rng.standard_normal(size=(chi,chi))     if real else crandn(size=(chi,chi))
    G.nodes[6]["T"] = rng.standard_normal(size=(chi))         if real else crandn(size=(chi))

    # sanity check
    assert network_intact_check(G)

    # computing the reference contraction value
    refval = np.einsum(
        "i,kp,kln,ql,poi,no,q->",
        G.nodes[0]["T"],
        G.nodes[1]["T"],
        G.nodes[2]["T"],
        G.nodes[3]["T"],
        G.nodes[4]["T"],
        G.nodes[5]["T"],
        G.nodes[6]["T"]
    )

    return G,refval

def grid_net(chi:int,width:int,height:int,rng:np.random.Generator=np.random.default_rng(),real:bool=True,psd:bool=False):
    r"""
    Construct random tensors forming a network on a two-dimensional square lattice. `chi` is the bonding dimension.
    Returns a list that contains the grid, as well as a `nx.MultiGraph`.

    Tensor axis ordering convention:

        __|__
       /  2  \
     --|0   1|--
       \__3__/
          |
    """
    # random number generation
    if real:
        randn = lambda size: rng.standard_normal(size)
    else:
        randn = lambda size: crandn(size, rng)

    tensors = []
    """Tuple that contains the grid."""
    G = nx.MultiGraph()
    G.add_nodes_from([i*width+j for i,j in itertools.product(range(height),range(width))])
    """Graph that represents the tensor network."""

    for i in range(height):
        row = []
        for j in range(width):
            # defining the tensor
            dim = (
                1 if j == 0        else chi,
                1 if j == width-1  else chi,
                1 if i == 0        else chi,
                1 if i == height-1 else chi)
            if psd:
                h = int(np.sqrt(chi))
                assert h**2 == chi, "if psd=True, chi must have an integer root."
                s = 0.3 * randn((
                    1 if j == 0        else h,
                    1 if j == width-1  else h,
                    1 if i == 0        else h,
                    1 if i == height-1 else h,
                    chi))
                t = np.einsum(s, (0, 2, 4, 6, 8), s.conj(), (1, 3, 5, 7, 8), (0, 1, 2, 3, 4, 5, 6, 7)).reshape(dim) / chi**(3/4)
            else:
                t = randn(dim) / chi**(3/4)

            # adding to the list of tensors
            row.append(t)

            # adding to the graph
            G.nodes[i*width+j]["T"] = t

            # adding edges
            if i != height - 1: G.add_edge(i*width+j,(i+1)*width+j,legs={i*width+j:3,(i+1)*width+j:2},trace=False,indices=None)
            if j != width - 1: G.add_edge(i*width+j,i*width+j+1,legs={i*width+j:1,i*width+j+1:0},trace=False,indices=None)
        tensors.append(row)

    outer_counter = height * width
    # adding outer edges and nodes
    for i in range(height):
        # left edge
        G.add_node(outer_counter,T=np.ones(shape=(1,)))
        G.add_edge(outer_counter,i*width,legs={i*width:0,outer_counter:0},trace=False,indices=None)
        outer_counter += 1

        # right edge
        G.add_node(outer_counter,T=np.ones(shape=(1,)))
        G.add_edge(outer_counter,(i+1)*width-1,legs={(i+1)*width-1:1,outer_counter:0},trace=False,indices=None)
        outer_counter += 1
    for j in range(width):
        # upper edge
        G.add_node(outer_counter,T=np.ones(shape=(1,)))
        G.add_edge(outer_counter,j,legs={j:2,outer_counter:0},trace=False,indices=None)
        outer_counter += 1

        # lower edge
        G.add_node(outer_counter,T=np.ones(shape=(1,)))
        G.add_edge(outer_counter,(height-1)*width+j,legs={(height-1)*width+j:3,outer_counter:0},trace=False,indices=None)
        outer_counter += 1

    outer_counter = height * width
    # contracting outer edges
    for i in range(height):
        # left edge
        contract_edge(i*width,outer_counter,0,G)
        outer_counter += 1

        # right edge
        contract_edge((i+1)*width-1,outer_counter,0,G)
        outer_counter += 1
    for j in range(width):
        # upper edge
        contract_edge(j,outer_counter,0,G)
        outer_counter += 1

        # lower edge
        contract_edge((height-1)*width+j,outer_counter,0,G)
        outer_counter += 1

    assert network_intact_check(G)
    return tensors,G

if __name__ == "__main__":
    pass
