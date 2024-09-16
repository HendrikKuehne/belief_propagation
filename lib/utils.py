"""
Random stuff that is useful here or there.
"""
import numpy as np
import networkx as nx
import warnings
import itertools

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

# -------------------------------------------------------------------------------
#                   sanity checks
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

# -------------------------------------------------------------------------------
#                   dummy tensor networks for testing
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
