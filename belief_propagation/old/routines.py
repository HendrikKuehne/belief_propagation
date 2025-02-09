"""
All the different algorithms that are implemented, contained in single functions that accept a network and give back a contraction value.
"""
import networkx as nx

from belief_propagation import BP,loopyNBP,networks,plaquette

def BP_routine(G:nx.MultiGraph,num_iter:int=30) -> float:
    """
    Complete pipeline of the tensor network code.
    """
    eps_list = BP.message_passing_iteration(G,num_iter,sanity_check=True)
    BP.normalize_messages(G)
    BP.contract_tensors_messages(G)

    cntr = 1
    for node,val in G.nodes(data="cntr"):
        cntr *= val
    return cntr

def BP_routine_blocking(G:nx.MultiGraph,width:int=4,height:int=4,blocksize:int=3,num_iter:int=30) -> float:
    """
    Complete pipeline of the tensor network code on a grid, including blocking.
    """
    BP.block_bp(G,width,height,blocksize)

    return BP_routine(G)

def loopyNBP_routine(G:nx.MultiGraph,r:int=2,num_iter:int=30) -> float:
    """
    Complete pipeline of the tensor network code, including neighborhood grouping.
    """
    neighborhood_list = loopyNBP.construct_neighborhoods(G,r)

    for neighborhood in neighborhood_list:
        edges,nodes = neighborhood
        loopyNBP.contract_neighborhood(G,nodes,True)

    return BP_routine(G)

def loopyNBP_feynman_routine(G:nx.MultiGraph,r:int=2,num_iter:int=30) -> float:
    """
    Complete pipeline of the tensor network code, including neighborhoods and a feynman cut.
    """
    neighborhood_list = loopyNBP.construct_neighborhoods(G,r)

    for neighborhood in neighborhood_list:
        edges,nodes = neighborhood
        loopyNBP.contract_neighborhood(G,nodes,True)

    # labeling the nodes based on their size
    for node1,node2 in G.edges():
        G[node1][node2][0]["size"] = G.nodes[node1]["T"].shape[G[node1][node2][0]["legs"][node1]]

    # sorting the edges based on their size
    sorted_edge_list = sorted(G.edges(),key=lambda edge: G[edge[0]][edge[1]][0]["size"],reverse=True)

    # feynman cut of the largest edge
    node1,node2 = sorted_edge_list[0]
    cut_graphs = networks.feynman_cut(G,node1,node2)

    cntr = 0
    for Gcut in cut_graphs:
        cntr += BP_routine(Gcut)

    return cntr

def PQ_routine(tensors:list,num_iter:int=30) -> float:
    """
    Complete pipeline of the plaquette code.
    """
    msg_in_l, msg_in_r, msg_in_u, msg_in_d, eps_iter = plaquette.message_passing_iteration(tensors,num_iter)
    msg_in_l, msg_in_r, msg_in_u, msg_in_d = plaquette.normalize_messages(msg_in_l,msg_in_r,msg_in_u,msg_in_d)
    cntr = plaquette.contract_tensors_messages(tensors,msg_in_l,msg_in_r,msg_in_u,msg_in_d)

    return cntr

def PQ_routine_blocking(tensors:list,num_iter:int=30) -> float:
    """
    Complete pipeline of the plaquette code with block belief propagation.
    """
    s_tensors = plaquette.block_bp(tensors)
    return PQ_routine(s_tensors)

if __name__ == "__main__":
    pass