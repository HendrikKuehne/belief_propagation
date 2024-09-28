import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from lib.BP import message_passing_iteration,normalize_messages,contract_opposing_messages,contract_tensors_messages
from lib.graphs import short_loop_graph
from lib.utils import network_intact_check,network_message_check
from lib.networks import construct_network,contract_network

if __name__ == "__main__": # loopy Belief Propagation
    #G = tree(20)
    G = short_loop_graph(30,3,.6)
    construct_network(G,9,real=False,psd=True)
    nNodes = G.number_of_nodes()

    print("Sanity checks:")
    print("    Network intact?",network_intact_check(G))
    print("    Network message-ready?",network_message_check(G),"\n")

    num_iter = 30
    eps_list = message_passing_iteration(G,num_iter,sanity_check=True)
    normalize_messages(G,True)

    if True: # plotting
        plt.figure("Tensor network")
        nx.draw(G,with_labels=True,font_weight="bold")

        plt.figure("Message norm vs. iterations")
        plt.semilogy(np.arange(num_iter),eps_list)
        plt.xlabel("Iteration")
        plt.ylabel(r"max $\Delta |\mathrm{Message}|$")
        plt.grid()

        plt.show()

    # contracting the network
    contract_opposing_messages(G,True)
    contract_tensors_messages(G,True)

    rel_err = lambda true,approx: np.real_if_close(true - approx) / np.abs(true)

    node_cntr_list = ()
    for node,val in G.nodes(data="cntr"):
        node_cntr_list += (val,)

    edge_cntr_list = ()
    for node1,node2,val in G.edges(data="cntr"):
        edge_cntr_list += (val,)

    refval = contract_network(G,True)

    #for cntr in node_cntr_list: print(np.isclose(cntr,refval))
    #for cntr in edge_cntr_list: print(np.isclose(cntr,refval))

    print("Comparing direct contraction and message passing:\n    Contraction:     {}\n    Message passing: {}".format(np.real_if_close(refval),np.real_if_close(np.prod(node_cntr_list))))
    print("Relative error = {:.3e}".format(rel_err(refval,np.prod(node_cntr_list))))