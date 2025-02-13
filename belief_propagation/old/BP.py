"""
Belief propagation on graphs. Taken from Kirkley et Al, 2021 ([Sci. Adv. 7, eabf1211 (2021)](https://doi.org/10.1126/sciadv.abf1211)).
"""
import numpy as np
import networkx as nx
import copy
import cotengra as ctg

from belief_propagation.utils import network_intact_check,network_message_check
from belief_propagation.old.networks_old import contract_edge,merge_edges,construct_initial_messages

def block_bp(G:nx.MultiGraph,width:int,height:int,blocksize:int=3,sanity_check:bool=False) -> None:
    """
    A kind of coarse-grainig inspired by the Block Belief Propagation
    algorithm (Arad, 2023: [Phys. Rev. B 108, 125111 (2023)](https://doi.org/10.1103/PhysRevB.108.125111)), which is the
    initialization of said algorithm. `G` is modified in-place.
    """
    if width <= blocksize or height <= blocksize: return

    # sanity check
    if sanity_check: assert network_intact_check(G)

    grid_to_node = lambda i,j: i*width+j

    # s0: Turning blocks into one plaquette by contracting their interior edges
    iBlock = 0
    jBlock = 0
    while blocksize * iBlock < height:
        while blocksize * jBlock < width:
            # edges in block (iBlock,jBlock)
            for i in range(blocksize * iBlock,min(blocksize * (iBlock + 1),height)):
                for j in range(blocksize * jBlock,min(blocksize * (jBlock + 1),width)):
                    if i == blocksize * iBlock and j == blocksize * jBlock: continue
                    contract_edge(grid_to_node(blocksize * iBlock,blocksize * jBlock),grid_to_node(i,j),0,G)
            jBlock += 1
        jBlock = 0
        iBlock += 1

    # contract trace edges
    trace_edges = ()
    for node1,node2,key in G.edges(keys=True):
        if node1 == node2: trace_edges += ((node1,node2,key),)
    for edge in trace_edges:
        contract_edge(*edge,G)

    # merge parallel edges
    parallel_edges = ()
    for node1,node2 in G.edges():
        if len(G[node1][node2]) > 1: parallel_edges += ({node1,node2},)
    for edge in parallel_edges:
        merge_edges(*edge,G)

    return

def message_passing_step(G:nx.MultiGraph,normalize:bool=True,sanity_check:bool=False) -> float:
    """
    Performs a message passing iteration. Algorithm taken from Kirkley, 2021 ([Sci. Adv. 7, eabf1211 (2021)](https://doi.org/10.1126/sciadv.abf1211)).
    'G' is modified in-place. Returns the maximum change `eps` of message norm over the entire graph.
    """
    # sanity check
    if sanity_check: assert network_message_check(G)

    if all([len(G.adj[node]) <= 1 for node in G.nodes]):
        # there are only leaf nodes in the graph; we don't need to do anything
        return None

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
            T_res = np.einsum(G.nodes[sending_node]["T"],list(range(nLegs)),*args,optimize=True)

            if normalize: T_res /= np.sum(T_res)

            # saving the normalized message
            G[node1][node2][0]["msg"][receiving_node] = T_res

            # saving the change in message norm
            eps += (np.linalg.norm(G[node1][node2][0]["msg"][receiving_node] - old_G[node1][node2][0]["msg"][receiving_node]),)

    return max(eps)

def message_passing_iteration(G:nx.MultiGraph,numiter:int=30,normalize:bool=True,verbose:bool=False,sanity_check:bool=False) -> tuple:
    """
    Performs a message passing iteration. `G` is modified in-place. Returns the change `eps` in maximum
    message norm for every iteration.
    """
    # sanity check
    if sanity_check: assert network_message_check(G)

    # initialization
    construct_initial_messages(G,normalize=normalize,sanity_check=sanity_check)

    if verbose: print(f"Message passing: {numiter} iterations.")

    eps_list = ()
    for i in range(numiter):
        eps = message_passing_step(G,normalize=normalize,sanity_check=sanity_check)
        if verbose: print("    iteration {:3}: eps = {:.3e}".format(i,eps))
        eps_list += (eps,)

    return eps_list

def normalize_messages(G:nx.MultiGraph,sanity_check:bool=False,norm_check:bool=False) -> None:
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

    if norm_check: # check if the messages, interpreted as matrices, are positive semi-definite
        print("Message normalization check:")
        h = int(np.sqrt(G[node1][node2][0]["msg"][node1].shape[0]))
        if h**2 == G[node1][node2][0]["msg"][node1].shape[0]:
            for node1,node2 in G.edges():
                # check normalization
                norm = np.dot(G[node1][node2][0]["msg"][node1],G[node1][node2][0]["msg"][node2])
                if not np.isclose(norm,1): print("    Edge ({},{}) normalized to {:.3f}".format(node1,node2,norm))

                # check positive semi-definite
                for node in (node1,node2):
                    # calculate the eigenvalues
                    eigvals = np.linalg.eigvals(G[node1][node2][0]["msg"][node].reshape(h,h))
                    # are they non-negative real numbers?
                    all_positive = all([np.real(eigval) >= 0 if np.real_if_close(eigval) == np.real(eigval) else False for eigval in eigvals])
                    if not all_positive: print(f"    Message G[{node1}][{node2}][0][\"msg\"][{node}] is not positive semi-definite.")

                    # are the matrices hermitian?
                    is_hermitian = np.allclose(G[node1][node2][0]["msg"][node].reshape(h,h),G[node1][node2][0]["msg"][node].reshape(h,h).conj())
                    if not is_hermitian: print(f"    Message G[{node1}][{node2}][0][\"msg\"][{node}] is not hermitian.")

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
        G.nodes[node]["cntr"] = ctg.einsum(G.nodes[node]["T"],list(range(nLegs)),*args,optimize="greedy")

def contract_opposing_messages(G:nx.MultiGraph,sanity_check:bool=False) -> None:
    """
    Contracts the two messages on every edge, and adds the value to each edge.
    """
    # sanity check
    if sanity_check: assert network_message_check(G)

    for node1,node2 in G.edges():
        T_res = np.dot(G[node1][node2][0]["msg"][node1],G[node1][node2][0]["msg"][node2])
        G[node1][node2][0]["cntr"] = T_res

if __name__ == "__main__":
    pass
