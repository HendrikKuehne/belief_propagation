"""
Belief propagation on graphs, i.e. on various geometries.
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import copy

from graph_creation import short_loop_graph,regular_graph,bipartite_regular_graph
from utils import crandn,contract_edge,network_intact_check,network_message_check,delta_network,dummynet5,grid_net

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

def contract_network(G:nx.MultiGraph,sanity_check:bool=False) -> float:
    """
    Contracts the tensor network `G`. The contraction order is random; this might be highly inefficient. `G` is modified in-place.
    """
    while G.number_of_edges() > 0:
        iEdge = 0#np.random.randint(G.number_of_edges())
        node1,node2,key = list(G.edges(keys=True))[iEdge]
        contract_edge(node1,node2,key,G)
        if sanity_check: assert network_intact_check(G)

    return tuple(G.nodes(data=True))[0][1]["T"]

def block_bp(G:nx.MultiGraph,width:int,height:int,sanity_check:bool=False) -> None:
    """
    A kind of coarse-grainig inspired by the Block Belief Propagation
    algorithm (Arad, 2023: [Phys. Rev. B 108, 125111 (2023)](https://doi.org/10.1103/PhysRevB.108.125111)), which is the
    initialization of said algorithm. `G` is modified in-place.
    """
    # not yet implemented; this is a little subtle because implementing this might require merging edges, which
    # I have not implemented
    if width < 4 or height < 4: return

    grid_to_node = lambda i,j: i*width+j

    # s0: Turning the upper-left 3x3 plaquettes into one plaquette

def construct_initial_messages(G:nx.MultiGraph,sanity_check:bool=False) -> None:
    """
    Initializes messages one the edges of `G`. Random initialisation except for leaf nodes, where the initial value
    is the tensor of the leaf node. `G` is modified in-place.
    """
    # sanity check
    if sanity_check: assert network_message_check(G)

    for node1,node2 in G.edges():
        G[node1][node2][0]["msg"] = {}

        for receiving_node in (node1,node2):
            sending_node = node2 if receiving_node == node1 else node1
            if len(G.adj[sending_node]) == 1:
                # message from leaf node
                G[node1][node2][0]["msg"][receiving_node] = G.nodes[sending_node]["T"]
            else:
                iLeg = G[node1][node2][0]["legs"][receiving_node]
                # bond dimension
                chi = G.nodes[receiving_node]["T"].shape[iLeg]
                # initialization with normalized vector
                G[node1][node2][0]["msg"][receiving_node] = np.ones(shape=(chi,)) / chi

def message_passing_step(G:nx.MultiGraph,sanity_check:bool=False) -> float:
    """
    Performs a message passing iteration. Algorithm taken from Kirkley, 2021 ([Sci. Adv. 7, eabf1211 (2021)](https://doi.org/10.1126/sciadv.abf1211)).
    'G' is modified in-place. Returns the maximum change `eps` of message norm over the entire graph.
    """
    # sanity check
    if sanity_check: assert network_message_check(G)

    old_G = copy.deepcopy(G)
    """Copy of the graph used to store the old messages."""

    eps = ()

    for node1,node2 in G.edges():
        for receiving_node in (node1,node2):
            sending_node = node2 if receiving_node == node1 else node1

            if len(G.adj[sending_node]) == 1:
                # leaf node; no action necessary
                continue

            # The outcoming message on one edge is the result of absorbing all incoming messages on all other edges into the tensor
            nLegs = G.nodes[sending_node]["T"].ndim
            args = ()

            for neighbor in G.adj[sending_node]:
                if neighbor == receiving_node: continue
                args += (old_G[sending_node][neighbor][0]["msg"][sending_node],(G[sending_node][neighbor][0]["legs"][sending_node],))
            T_res = np.einsum(G.nodes[sending_node]["T"],list(range(nLegs)),*args)

            # saving the normalized message
            G[node1][node2][0]["msg"][receiving_node] = T_res / np.sum(T_res)

            # saving the change in message norm
            eps += (np.linalg.norm(G[node1][node2][0]["msg"][receiving_node] - old_G[node1][node2][0]["msg"][receiving_node]),)

    return max(eps)

def message_passing_iteration(G:nx.MultiGraph,numiter:int=30,verbose:bool=False,sanity_check:bool=False) -> tuple:
    """
    Performs a message passing iteration. `G` is modified in-place. Returns the change `eps` in maximum
    message norm for every iteration.
    """
    # sanity check
    if sanity_check: assert network_message_check(G)

    # initialization
    construct_initial_messages(G,sanity_check)

    if verbose: print(f"Message passing: {numiter} iterations.")

    eps_list = ()
    for i in range(numiter):
        eps = message_passing_step(G,sanity_check)
        if verbose: print("    iteration {:3}: eps = {:.3e}".format(i,eps))
        eps_list += (eps,)

    return eps_list

def normalize_messages(G:nx.MultiGraph,sanity_check:bool=False) -> None:
    """
    Normalize messages such that the inner product between messages traveling along
    the same edge but in opposite directions is one. `G` is modified in-place.
    """
    # sanity check
    if sanity_check: assert network_message_check(G)

    for node1,node2 in G.edges():
        norm = np.dot(G[node1][node2][0]["msg"][node1],G[node1][node2][0]["msg"][node2])
        G[node1][node2][0]["msg"][node1] /= np.sqrt(np.abs(norm))
        G[node1][node2][0]["msg"][node2] /= np.sqrt(np.abs(norm))

def contract_tensors_messages(G:nx.MultiGraph,sanity_check:bool=False) -> None:
    """
    Contracts all messages into the respective nodes, and adds the value to each node.
    """
    # sanity check
    if sanity_check: assert network_message_check(G)

    for node in G.nodes():
        nLegs = G.nodes[node]["T"].ndim
        args = ()
        for neighbor in G.adj[node]:
            args += (G[node][neighbor][0]["msg"][node],(G[node][neighbor][0]["legs"][node],))
        G.nodes[node]["cntr"] = np.einsum(G.nodes[node]["T"],list(range(nLegs)),*args)

if __name__ == "__main__":
    G = short_loop_graph(10,3,.6)
    construct_network(G,4,real=False,psd=True)

    print("Sanity checks:")
    print("    Network intact?",network_intact_check(G))
    print("    Network message-ready?",network_message_check(G),"\n")

    num_iter = 30
    eps_list = message_passing_iteration(G,num_iter,sanity_check=True)
    normalize_messages(G,True)

    if False: # plotting
        plt.figure("Tensor network")
        nx.draw(G,with_labels=True,font_weight="bold")

        plt.figure("Message norm vs. iterations")
        plt.semilogy(np.arange(num_iter),eps_list)
        plt.xlabel("Iteration")
        plt.ylabel(r"max $\Delta |\mathrm{Message}|$")
        plt.grid()

        plt.show()

    contract_tensors_messages(G,False)
    cntr = 1
    for node,val in G.nodes(data="cntr"): cntr *= val
    refval = contract_network(G,True)
    rel_err = np.abs(refval - cntr) / np.abs(refval)

    print("\nComparing direct contraction and message passing:\n    Contraction:     {}\n    Message passing: {}".format(np.real_if_close(refval),np.real_if_close(cntr)))
    print("Relative error = {:.3e}".format(rel_err))
