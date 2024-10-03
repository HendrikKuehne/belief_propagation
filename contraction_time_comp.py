import numpy as np
import copy
import time
import cotengra as ctg
import matplotlib.pyplot as plt
import networkx as nx

from belief_propagation.utils import network_message_check
from belief_propagation.graphs import short_loop_graph
from belief_propagation.networks import construct_network,contract_network

def cotengra_contractiontree(G:nx.MultiGraph) -> float:
    tensors = ()
    legs = ()
    sizes = {}

    # enumerating the edges in the graph
    for i,nodes in enumerate(G.edges()):
        node1,node2 = nodes
        G[node1][node2][0]["label"] = str(i)
    # extracting edge labels from G
    for node,T in G.nodes(data="T"):
        tensors += (T,)
        legs_ = [None for i in range(T.ndim)]
        for _,neighbor,edge_label in G.edges(nbunch=node,data="label"):
            legs_[G[node][neighbor][0]["legs"][node]] = edge_label
            if edge_label not in sizes.keys(): sizes[edge_label] = T.shape[G[node][neighbor][0]["legs"][node]]
        legs += (tuple(legs_),)

    opt = ctg.HyperOptimizer()
    # optimal contraction order
    tree = opt.search(inputs=legs,output="",size_dict=sizes)

    return ctg.array_contract(tensors,legs,size_dict=sizes)#,optimize=tree)

def cotengra_array_contract(G:nx.MultiGraph) -> float:
    tensors = ()
    legs = ()
    sizes = {}

    # enumerating the edges in the graph
    for i,nodes in enumerate(G.edges()):
        node1,node2 = nodes
        G[node1][node2][0]["label"] = str(i)
    # extracting edge labels from G
    for node,T in G.nodes(data="T"):
        tensors += (T,)
        legs_ = [None for i in range(T.ndim)]
        for _,neighbor,edge_label in G.edges(nbunch=node,data="label"):
            legs_[G[node][neighbor][0]["legs"][node]] = edge_label
            if edge_label not in sizes.keys(): sizes[edge_label] = T.shape[G[node][neighbor][0]["legs"][node]]
        legs += (tuple(legs_),)

    return ctg.array_contract(tensors,legs,size_dict=sizes,optimize="auto-hq")

def using_einsum(G:nx.MultiGraph) -> float:
    args = ()

    # enumerating the edges in the graph
    for i,nodes in enumerate(G.edges()):
        node1,node2 = nodes
        G[node1][node2][0]["label"] = i
    # extracting edge labels from G
    for node,T in G.nodes(data="T"):
        args += (T,)
        legs = [None for i in range(T.ndim)]
        for _,neighbor,edge_label in G.edges(nbunch=node,data="label"):
            legs[G[node][neighbor][0]["legs"][node]] = edge_label
        args += (tuple(legs),)

    return np.einsum(*args,optimize=True)

def using_einsum_path(G:nx.MultiGraph) -> float:
    args = ()

    # enumerating the edges in the graph
    for i,nodes in enumerate(G.edges()):
        node1,node2 = nodes
        G[node1][node2][0]["label"] = i
    # extracting edge labels from G
    for node,T in G.nodes(data="T"):
        args += (T,)
        legs = [None for i in range(T.ndim)]
        for _,neighbor,edge_label in G.edges(nbunch=node,data="label"):
            legs[G[node][neighbor][0]["legs"][node]] = edge_label
        args += (tuple(legs),)

    path,path_info = np.einsum_path(*args,optimize=True)
    return np.einsum(*args,optimize=path)

if __name__ == "__main__":
    times = ()
    nSamples = 10

    for i in range(nSamples):
        G = short_loop_graph(15,3,.6)
        construct_network(G,3,real=False,psd=False)
        assert network_message_check(G)

        t0 = time.time()
        ref0 = contract_network(copy.deepcopy(G))
        t1 = time.time()
        ref1 = cotengra_array_contract(copy.deepcopy(G))
        t2 = time.time()
        ref2 = using_einsum(copy.deepcopy(G))
        t3 = time.time()
        ref3 = using_einsum_path(copy.deepcopy(G))
        t4 = time.time()
        ref4 = cotengra_contractiontree(copy.deepcopy(G))
        t5 = time.time()

        sample_times = (t1-t0,)

        sample_times = sample_times + (t2-t1,) if np.isclose(ref1,ref0) else sample_times + (None,)
        sample_times = sample_times + (t3-t2,) if np.isclose(ref2,ref0) else sample_times + (None,)
        sample_times = sample_times + (t4-t3,) if np.isclose(ref3,ref0) else sample_times + (None,)
        sample_times = sample_times + (t5-t4,) if np.isclose(ref4,ref0) else sample_times + (None,)

        times += (sample_times,)

    times = np.array(times)
    method_labels = ("Random","Cotengra","Einsum","Einsum path","Cotengra tree")

    for iMethod in range(times.shape[1]):
        plt.scatter(iMethod * np.ones(times.shape[0]),times[:,iMethod],alpha=.5,label=method_labels[iMethod])
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.show()

    #if np.isclose(ref0,ref1) and np.isclose(ref0,ref2):
    #    print(f"Success.\n    Random:      {t1-t0}\n    Cotengra:    {t2-t1}\n    Einsum:      {t3-t2}\n    Einsum path: {t4-t3}")