"""
Comparison between the tensor network code and the plaquette code.
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import network_contraction as tn
import plaquette_contraction as pq
import utils
import graph_creation as graphs

def tn_routine(G:nx.MultiGraph,num_iter:int=30) -> float:
    """
    Complete pipeline of the tensor network code.
    """
    eps_list = tn.message_passing_iteration(G,num_iter,sanity_check=True)
    tn.normalize_messages(G)
    tn.contract_tensors_messages(G)

    cntr = 1
    for node,val in G.nodes(data="cntr"): cntr *= val
    refval = tn.contract_network(G,True)

    rel_err = np.abs(refval - cntr) / np.abs(refval)

    return rel_err

def pq_routine(tensors:list,num_iter:int=30) -> float:
    """
    Complete pipeline of the plaquette code.
    """
    c_ref = pq.contract_network(tensors)
    msg_in_l, msg_in_r, msg_in_u, msg_in_d, eps_iter = pq.message_passing_iteration(tensors,num_iter)
    msg_in_l, msg_in_r, msg_in_u, msg_in_d = pq.normalize_messages(msg_in_l,msg_in_r,msg_in_u,msg_in_d)
    cntr = pq.contract_tensors_messages(tensors,msg_in_l,msg_in_r,msg_in_u,msg_in_d)
    rel_err = abs(np.prod(cntr) - c_ref) / abs(c_ref)

    return rel_err

def pq_routine_blocking(tensors:list,num_iter:int=30) -> float:
    """
    Complete pipeline of the plaquette code.
    """
    c_ref = pq.contract_network(tensors)
    s_tensors = pq.block_bp(tensors)
    msg_in_l, msg_in_r, msg_in_u, msg_in_d, eps_iter = pq.message_passing_iteration(s_tensors,num_iter)
    msg_in_l, msg_in_r, msg_in_u, msg_in_d = pq.normalize_messages(msg_in_l,msg_in_r,msg_in_u,msg_in_d)
    cntr = pq.contract_tensors_messages(s_tensors,msg_in_l,msg_in_r,msg_in_u,msg_in_d)
    rel_err = abs(np.prod(cntr) - c_ref) / abs(c_ref)

    return rel_err

if __name__ == "__main__":
    nTrials = 50
    chi = 4
    connectivity = 3
    p = .6
    results = ()
    for nNodes in range(5,20):
        for iTrial in range(nTrials):
            # constructing the tensors
            G = graphs.short_loop_graph(nNodes,connectivity,p)
            numer_of_nodes = G.number_of_nodes()
            tn.construct_network(G,chi,real=False,psd=True)

            tn_err = tn_routine(G)

            results += ((numer_of_nodes,connectivity,p,tn_err),)
    results = np.array(results)

    # preparing bins for a logarithmic hstogram
    #min_err = np.min(results[psd_mask,1:4])
    #max_err = np.max(results[psd_mask,1:4])
    #numbins = 40
    #bins = np.logspace(start=np.log10(min_err / (max_err - min_err)**(1/numbins)),stop=np.log10(max_err * (max_err - min_err)**(1/numbins)),num=numbins)

    plt.scatter(results[:,0],results[:,-1],alpha=.5)
    plt.suptitle("Bonding dimension chi = {}, node connectivity {}, edge discarding ratio {}".format(chi,connectivity,p))
    plt.xlabel("Number of nodes")
    plt.ylabel(r"$\frac{\Delta C}{C}$")
    plt.yscale("log")
    plt.show()