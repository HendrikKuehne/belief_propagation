"""
Comparison between the tensor network code and the plaquette code.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import itertools
import time
import copy
import plotly.express as px
import pickle
from datetime import datetime

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

def tn_routine_blocking(G:nx.MultiGraph,width:int,height:int,blocksize:int,num_iter:int=30) -> float:
    """
    Complete pipeline of the tensor network code with block belief propagation.
    """
    tn.block_bp(G,width,height,blocksize,sanity_check=True)
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
    Complete pipeline of the plaquette code with block belief propagation.
    """
    c_ref = pq.contract_network(tensors)
    s_tensors = pq.block_bp(tensors)
    msg_in_l, msg_in_r, msg_in_u, msg_in_d, eps_iter = pq.message_passing_iteration(s_tensors,num_iter)
    msg_in_l, msg_in_r, msg_in_u, msg_in_d = pq.normalize_messages(msg_in_l,msg_in_r,msg_in_u,msg_in_d)
    cntr = pq.contract_tensors_messages(s_tensors,msg_in_l,msg_in_r,msg_in_u,msg_in_d)
    rel_err = abs(np.prod(cntr) - c_ref) / abs(c_ref)

    return rel_err

if __name__ == "__main__":
    nTrials = 20
    results = {
        "nNodes":(),
        "chi":(),
        "width":(),
        "height":(),
        "tn_blocksize":(),
        "psd":(),
        "pq_err":(),
        "pq_block_err":(),
        "tn_err":(),
        "tn_block_err":(),
        "pq_time":(),
        "pq_block_time":(),
        "tn_time":(),
        "tn_block_time":(),
    }
    t0 = time.time()

    for iTrial,chi,blocksize,psd,width_height in itertools.product(
        range(nTrials),
        (4,),
        (2,3),
        (False,True),
        itertools.product(np.arange(2,5),repeat=2)
    ):
        width,height = width_height
        tensors,G = utils.grid_net(chi,width,height,real=False,psd=psd)

        if time.time() - t0 > 120:
            t0 = time.time()
            print(f"    still going ...")

        t1 = time.time()
        pq_err = pq_routine(tensors)
        t2 = time.time()
        try:
            pq_block_err = pq_routine_blocking(tensors)
        except IndexError:
            pq_block_err = None
        t3 = time.time()
        tn_err = tn_routine(copy.deepcopy(G))
        t4 = time.time()
        tn_block_err = tn_routine_blocking(G,width,height,blocksize)
        t5 = time.time()

        # simulation parameters
        results["nNodes"] += (width*height,)
        results["chi"] += (chi,)
        results["width"] += (width,)
        results["height"] += (height,)
        results["tn_blocksize"] += (blocksize,)
        results["psd"] += (psd,)

        # results
        results["pq_err"] += (pq_err,)
        results["pq_block_err"] += (pq_block_err,)
        results["tn_err"] += (tn_err,)
        results["tn_block_err"] += (tn_block_err,)

        # runtimes
        results["pq_time"] += (t2-t1,)
        results["pq_block_time"] += (t3 - t2 if pq_block_err != None else None,)
        results["tn_time"] += (t4 - t3,)
        results["tn_block_time"] += (t5 - t4,)

    for key in results.keys():
        results[key] = np.array(results[key])

    # saving the data in a file
    now = datetime.now()
    timestr = now.strftime("%m-%d_%H-%M-%S")

    with open("doc/data/" + timestr + ".pickle","wb") as file:
        pickle.dump(results,file)

if __name__ == "dings":#"__main__":
    nTrials = 20
    chi = 4
    connectivity = 3
    blocksize = 3
    p = .6
    rng = np.random.default_rng()
    results = ()
    t0 = time.time()

    for iTrial,psd,width_height in itertools.product(
        range(nTrials),
        (False,True),
        itertools.product(np.arange(2,5),repeat=2)
    ):
        width,height = width_height
        tensors,G = utils.grid_net(chi,width,height,real=False,psd=psd)

        if time.time() - t0 > 120:
            t0 = time.time()
            print(f"    still going ...")

        t1 = time.time()
        try:
            pq_err = pq_routine(tensors)
        except IndexError:
            pq_err = None
        t2 = time.time()
        try:
            pq_block_err = pq_routine_blocking(tensors)
        except IndexError:
            pq_block_err = None
        t3 = time.time()
        tn_err = tn_routine(copy.deepcopy(G))
        t4 = time.time()
        tn_block_err = tn_routine_blocking(G,width,height)
        t5 = time.time()

        results += ((width*height,psd,pq_err,pq_block_err,tn_err,tn_block_err),)
    results = np.array(results)

    psd_mask = results[:,1] == True

    minmax_errs = ()
    for nNodes in np.sort(list(set(results[:,0]))):
        nNodes_mask = results[:,0] == nNodes

        # finding the minimum relative error when psd = False
        vals = results[np.logical_and(nNodes_mask,np.logical_not(psd_mask)),2:].flatten()
        min_nonpsd_err = np.min(vals[vals != np.array(None)])

        # finding the maximum relative error when psd = True
        vals = results[np.logical_and(nNodes_mask,psd_mask),2:].flatten()
        max_psd_err = np.nanmax(vals[vals != np.array(None)])

        minmax_errs += ((nNodes,min_nonpsd_err,max_psd_err),)
    minmax_errs = np.array(minmax_errs)

    # cosmetics
    add_noise = lambda x: x + np.random.normal(scale=.1,size=x.shape[0])

    plt.plot(minmax_errs[:,0],minmax_errs[:,1],linestyle="dashed",c="grey",label=r"$\mathrm{Min}_{\mathrm{psd=False}}$")
    plt.plot(minmax_errs[:,0],minmax_errs[:,2],linestyle="dotted",c="grey",label=r"$\mathrm{Max}_{\mathrm{psd=True}}$")

    plt.scatter(add_noise(results[:,0]),results[:,2],marker=mpl.markers.MarkerStyle("s",fillstyle="none"),c="b",label="pq")
    plt.scatter(add_noise(results[:,0]),results[:,3],marker=mpl.markers.MarkerStyle("D",fillstyle="none"),c="b",label="pq_block")
    plt.scatter(add_noise(results[:,0]),results[:,4],marker="+",c="r",label="tn")
    plt.scatter(add_noise(results[:,0]),results[:,5],marker="x",c="r",label="tn_block")
    plt.suptitle(f"Bonding dimension chi = {chi}. {nTrials} samples. Blocksize 3x3.")
    plt.xlabel("Number of nodes")
    plt.ylabel(r"$\frac{\Delta C}{C}$")
    plt.yscale("log")
    plt.legend()
    plt.show()