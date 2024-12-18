"""
Creating sandwiches of the form
* MPS - PEPO - MPS, or
* MPS - MPS,

by combining the classes MPS and PEPO.
These classes implement the Belief Propagation
algorithm on graphs, as well as DMRG.
"""

import numpy as np
import networkx as nx
import cotengra as ctg
import warnings
import itertools
import ray
import tqdm
import scipy.linalg as scialg

from belief_propagation.utils import network_message_check,crandn,is_hermitian,rel_err
from belief_propagation.sandwich_BP.PEPO import PEPO
from belief_propagation.sandwich_BP.PEPS import PEPS

@ray.remote(num_cpus=1)
def contract_tensor_msg(
        msg:dict,
        sending_node:int,
        receiving_node:int,
        bra_legs:dict,
        op_legs:dict,
        ket_legs:dict,
        bra_T:np.ndarray,
        op_T:np.ndarray,
        ket_T:np.ndarray,
    ) -> np.ndarray:
    """
    Contracts tensor and messages at `sending_node`, and returns the message
    that `sending_node` sends to `receiving_node`. Same functionality as
    `Braket.__contract_tensor_msg`.
    """
    # The outcoming message on one edge is the result of absorbing all incoming messages on all other edges into the tensor sandwich
    nLegs = len(msg)
    args = ()
    """Arguments for einsum"""

    out_legs = list(range(3 * nLegs))

    for neighbor in msg.keys():
        if neighbor == receiving_node: continue
        args += (
            msg[neighbor],
            (
                bra_legs[neighbor][sending_node], # bra leg
                nLegs + op_legs[neighbor][sending_node], # operator leg
                2 * nLegs + ket_legs[neighbor][sending_node], # ket leg
            )
        )
        out_legs.remove(bra_legs[neighbor][sending_node])
        out_legs.remove(nLegs + op_legs[neighbor][sending_node])
        out_legs.remove(2 * nLegs + ket_legs[neighbor][sending_node])

    args += (
        # bra tensor
        bra_T,tuple(range(nLegs)) + (3 * nLegs,),
        # operator tensor
        op_T,tuple(nLegs + iLeg for iLeg in range(nLegs)) + (3 * nLegs,3 * nLegs + 1),
        # ket tensor
        ket_T,tuple(2 * nLegs + iLeg for iLeg in range(nLegs)) + (3 * nLegs + 1,),
    )

    msg = np.einsum(*args,out_legs,optimize=True)

    return msg

class Braket:
    """
    Base class for sandwiches of MPS and PEPOs. Contains code for belief propagation.
    """

    numiter:int = 500
    """Maximum number of message update iterations."""
    numretries:int = 10
    """Maximum number of message update iteration retries."""
    threshold:float = 1e-10
    """Default threshold for breaking of the BP iteration."""

    def intact_check(self) -> bool:
        """
        Cheks if the braket is intact. This includes:
        * Is the network `G` message-ready?
        * Are `bra`, `op`, and `ket` themselves intact?
        * Do the physical dimensions match?
        * Do all the messages contain finite values?
        """
        assert hasattr(self,"bra")
        assert hasattr(self,"op")
        assert hasattr(self,"ket")
        assert hasattr(self,"D")

        if not network_message_check(self.G): return False

        if not self.bra.intact_check(): return False
        if not self.op.intact_check(): return False
        if not self.ket.intact_check(): return False

        # do the physical dimensions match?
        if not self.bra.D == self.D and self.ket.D == self.D and self.op.D == self.D:
            warnings.warn("Physical dimensions in braket do not match.")
            return False

        # are there any messages with non-finite values?
        for node1,node2,data in self.G.edges(data=True):
            if "msg" in data.keys():
                if node1 in data["msg"].keys():
                    if not np.isfinite(data["msg"][node1]).all():
                        warnings.warn(f"Non-finite message sent from {node2} to {node1}.")
                        return False
                if node2 in data["msg"].keys():
                    if not np.isfinite(data["msg"][node2]).all():
                        warnings.warn(f"Non-finite message sent from {node1} to {node2}.")
                        return False

        return True

    def __construct_initial_messages(self,real:bool,normalize:bool,sanity_check:bool,rng:np.random.Generator=np.random.default_rng()) -> None:
        """
        Initial messages for BP iteration. Saved under the key `msg`
        in `self.G`.

        Messages are three-index tensors, where the first index belongs
        to the bra, the second index belongs to the operator and the third
        one belongs to the ket.
        """
        # sanity check
        if sanity_check: assert self.intact_check()

        # random number generation
        if real:
            randn = lambda size: rng.standard_normal(size)
        else:
            randn = lambda size: crandn(size,rng)

        for node1,node2 in self.G.edges():
            for sending_node,receiving_node in itertools.permutations((node1,node2)):
                # messages in both directions
                if len(self.G.adj[sending_node]) > 1:
                    # ket and bra leg indices
                    iBra = self.bra.G[sending_node][receiving_node][0]["legs"][receiving_node]
                    iKet = self.ket.G[sending_node][receiving_node][0]["legs"][receiving_node]
                    bra_size = self.bra.G.nodes[receiving_node]["T"].shape[iBra]
                    ket_size = self.ket.G.nodes[receiving_node]["T"].shape[iKet]
                    # calculating the message
                    msg = randn((bra_size,self.op.chi,ket_size))
                else:
                    # sending node is leaf node
                    msg = ctg.einsum(
                        "ij,kjl,rl->ikr",
                        self.bra.G.nodes[sending_node]["T"],
                        self.op.G.nodes[sending_node]["T"],
                        self.ket.G.nodes[sending_node]["T"]
                    )

                if normalize: msg /= np.sum(msg)
                self.G[sending_node][receiving_node][0]["msg"][receiving_node] = msg

        return

    def contract_tensor_msg(self,sending_node:int,receiving_node:int,sanity_check:bool) -> np.ndarray:
        """
        Contracts tensor and messages at `sending_node`, and returns the message
        that flows from `sending_node` to `receiving_node`.
        """
        if sanity_check:
            assert self.G.has_node(sending_node) and self.G.has_node(receiving_node)

        # The outcoming message on one edge is the result of absorbing all incoming messages on all other edges into the tensor sandwich
        nLegs = len(self.G.adj[sending_node])
        args = ()
        """Arguments for einsum"""

        out_legs = list(range(3 * nLegs))

        for neighbor in self.G.adj[sending_node]:
            if neighbor == receiving_node: continue
            args += (
                self.G[sending_node][neighbor][0]["msg"][sending_node],
                (
                    self.bra.G[sending_node][neighbor][0]["legs"][sending_node], # bra leg
                    nLegs + self.op.G[sending_node][neighbor][0]["legs"][sending_node], # operator leg
                    2 * nLegs + self.ket.G[sending_node][neighbor][0]["legs"][sending_node], # ket leg
                )
            )
            out_legs.remove(self.bra.G[sending_node][neighbor][0]["legs"][sending_node])
            out_legs.remove(nLegs + self.op.G[sending_node][neighbor][0]["legs"][sending_node])
            out_legs.remove(2 * nLegs + self.ket.G[sending_node][neighbor][0]["legs"][sending_node])

        args += (
            # bra tensor
            self.bra.G.nodes[sending_node]["T"],
            tuple(range(nLegs)) + (3 * nLegs,),
            # operator tensor
            self.op.G.nodes[sending_node]["T"],
            tuple(nLegs + iLeg for iLeg in range(nLegs)) + (3 * nLegs,3 * nLegs + 1),
            # ket tensor
            self.ket.G.nodes[sending_node]["T"],
            tuple(2 * nLegs + iLeg for iLeg in range(nLegs)) + (3 * nLegs + 1,),
        )

        msg = np.einsum(*args,out_legs,optimize=True)

        return msg

    def __message_passing_step(self,normalize:bool,parallel:bool,sanity_check:bool) -> float:
        """
        Performs a message passing iteration. Algorithm taken from Kirkley, 2021
        ([Sci. Adv. 7, eabf1211 (2021)](https://doi.org/10.1126/sciadv.abf1211)).
        Returns the maximum change of message norm over the entire graph.

        Parallelized using ray.
        """
        # sanity check
        if sanity_check: assert self.intact_check()

        if all([len(self.G.adj[node]) <= 1 for node in self.G.nodes]):
            # there are only leaf nodes in the graph; we don't need to do anything
            return 0

        eps = ()

        if not parallel:
            # old, non-parallel version of the code
            newG = self.prepare_graph(self.G,keep_legs=True)
            """Copy of the graph used to store the new messages."""

            for node1,node2 in self.G.edges():
                for sending_node,receiving_node in itertools.permutations((node1,node2)):
                    # messages in both directions

                    if len(self.G.adj[sending_node]) == 1:
                        # leaf node; no action necessary
                        newG[sending_node][receiving_node][0]["msg"][receiving_node] = self.G[sending_node][receiving_node][0]["msg"][receiving_node]
                        continue

                    msg = self.contract_tensor_msg(sending_node,receiving_node,sanity_check)

                    # saving the new message
                    newG[sending_node][receiving_node][0]["msg"][receiving_node] = msg / np.sum(msg) if normalize else msg
                    # change in message norm
                    eps += (np.linalg.norm(self.G[sending_node][receiving_node][0]["msg"][receiving_node] - newG[sending_node][receiving_node][0]["msg"][receiving_node]),)

            # put new messages in the graph
            self.G = newG

            return np.max(eps)

        if not ray.is_initialized(): ray.init()

        ray_refs = []
        msg_ids = ()

        for node1,node2 in self.G.edges():
            for sending_node,receiving_node in itertools.permutations((node1,node2)):
                # messages in both directions

                if len(self.G.adj[sending_node]) == 1:
                    # leaf node; no action necessary
                    continue

                ray_refs += [contract_tensor_msg.remote(**self.neighborhood(sending_node,receiving_node,sanity_check)),]
                msg_ids += ((sending_node,receiving_node),)

        # get new messages
        new_msg = ray.get(ray_refs)

        for msg_id,msg in zip(msg_ids,new_msg):
            sending_node,receiving_node = msg_id
            # change in message norm
            eps += (np.linalg.norm(self.G[sending_node][receiving_node][0]["msg"][receiving_node] - msg / np.sum(msg) if normalize else msg),)
            # saving the new message
            self.G[sending_node][receiving_node][0]["msg"][receiving_node] = msg / np.sum(msg) if normalize else msg

        return max(eps)

    def __message_passing_iteration(self,numiter:int,real:bool,normalize:bool,threshold:float,parallel:bool,iterator_desc_prefix:str,verbose:bool,sanity_check:bool) -> tuple[float]:
        """
        Performs a message passing iteration. Returns the change `eps` in maximum
        message norm for every iteration.
        """
        # sanity check
        if sanity_check: assert self.intact_check()

        iterator = tqdm.tqdm(range(numiter),desc=iterator_desc_prefix + f"BP iteration",disable=not verbose)

        eps_list = ()
        # message initialization
        self.__construct_initial_messages(real=real,normalize=normalize,sanity_check=sanity_check)

        for i in iterator:
            eps = self.__message_passing_step(normalize=normalize,parallel=parallel,sanity_check=sanity_check)
            iterator.set_postfix_str(f"eps = {eps:.3e}")
            eps_list += (eps,)

            if eps < threshold:
                if verbose: iterator.set_postfix_str(f"eps = {eps:.3e}; threshold reached; returning.")
                iterator.close()
                return eps_list

        return eps_list

    def __normalize_messages(self,sanity_check:bool=False) -> None:
        """
        Normalize messages, such that the inner product between messages
        traveling along the same edge, but in opposite directions, is one.
        Saves the inner product on the edge under the key `cntr`.
        """
        # sanity check
        if sanity_check: assert self.intact_check()

        for node1,node2 in self.G.edges():
            norm = ctg.einsum("ijk,ijk->",self.G[node1][node2][0]["msg"][node1],self.G[node1][node2][0]["msg"][node2])
            self.G[node1][node2][0]["msg"][node1] /= np.sqrt(np.abs(norm))
            self.G[node1][node2][0]["msg"][node2] /= np.sqrt(np.abs(norm))
            self.G[node1][node2][0]["cntr"] = norm

        return

    def __contract_tensors_messages(self,sanity_check:bool=False) -> None:
        """
        Contracts all messages into the respective nodes, and adds the value to each node.
        """
        # sanity check
        if sanity_check: assert self.intact_check()

        for node in self.G.nodes():
            nLegs = len(self.G.adj[node])
            args = ()

            for neighbor in self.G.adj[node]:
                args += (
                    self.G[node][neighbor][0]["msg"][node],
                    (
                        self.bra.G[node][neighbor][0]["legs"][node], # bra leg
                        nLegs + self.op.G[node][neighbor][0]["legs"][node], # operator leg
                        2 * nLegs + self.ket.G[node][neighbor][0]["legs"][node], # ket leg
                    )
                )

            args += (
                # bra tensor
                self.bra.G.nodes[node]["T"],
                tuple(iLeg for iLeg in range(nLegs)) + (3 * nLegs,),
                # operator tensor
                self.op.G.nodes[node]["T"],
                tuple(nLegs + iLeg for iLeg in range(nLegs)) + (3 * nLegs,3 * nLegs + 1),
                # ket tensor
                self.ket.G.nodes[node]["T"],
                tuple(2 * nLegs + iLeg for iLeg in range(nLegs)) + (3 * nLegs + 1,),
            )

            self.G.nodes[node]["cntr"] = ctg.einsum(*args,optimize="greedy")

        return

    def BP(self,numiter:int=None,numretries:float=None,real:bool=False,normalize:bool=True,threshold:float=1e-10,parallel:bool=False,verbose:bool=True,sanity_check:bool=False,**kwargs) -> None:
        """
        Runs the BP algorithm with `numiter` iterations on the network. Parameters:
        * `numretries`: Number of times the BP iteration is starting over when it does not converge.
        The value `np.inf` can be supplied, in which case the algorithm runs until `threshold` is reached.
        * `real`: Initialization of messages with real values (otherwise complex).
        * `normalize`: Normalization of messages after new message calculation. If `normalize=True`,
        this function implements the BP algorithm from Kirkley, 2021
        ([Sci. Adv. 7, eabf1211 (2021)](https://doi.org/10.1126/sciadv.abf1211)). Otherwise,
        this function becomes Belief Propagation on trees.
        * `threshold`: When to abort the BP iteration.

        Writes the network contraction value to `self.cntr`.
        """
        # sanity check
        if sanity_check: assert self.intact_check()

        if self.G.number_of_nodes() == 1:
            warnings.warn("The network is trivial.")
            return None

        # handling kwargs
        kwargs["iterator_desc_prefix"] = kwargs["iterator_desc_prefix"] + " | " if "iterator_desc_prefix" in kwargs.keys() else ""

        # initially, the messages are not converged
        self.converged = False

        numiter = numiter if numiter != None else self.numiter
        numretries = numretries if numretries != None else self.numretries
        threshold = threshold if threshold != None else self.threshold

        iRetry = 0
        while iRetry < numretries:
            # message passing iteration
            eps_list = self.__message_passing_iteration(
                numiter=numiter,
                real=real,
                normalize=normalize,
                threshold=threshold,
                parallel=parallel,
                iterator_desc_prefix=kwargs["iterator_desc_prefix"] + f"retry {iRetry} | ",
                verbose=verbose,
                sanity_check=sanity_check,
            )

            iRetry += 1

            if eps_list[-1] < threshold:
                self.converged = True
                break

        # contract tensors and messages
        self.__contract_tensors_messages(sanity_check=sanity_check)

        # opposing message normalization
        if normalize: self.__normalize_messages(sanity_check=sanity_check)

        if normalize:
            # the network value is the product of all node values, divided by all edge values
            self.cntr = 1
            for node,node_cntr in self.G.nodes(data="cntr"): self.cntr *= node_cntr
            for node1,node2,edge_cntr in self.G.edges(data="cntr"): self.cntr /= edge_cntr
        else:
            # each node carries the network value
            self.cntr = self.G.nodes[self.op.root]["cntr"]

        return

    def __contract_ctg_einsum(self,sanity_check:bool=False,**kwargs) -> float:
        """
        Exact contraction using `ctg.einsum`.
        """
        # sanity check
        if sanity_check: assert self.intact_check()

        if self.G.number_of_nodes() == 1:
            # the network is trivial
            bra = tuple(self.bra.G.nodes(data="T"))[0][1]
            op = tuple(self.op.G.nodes(data="T"))[0][1]
            ket = tuple(self.ket.G.nodes(data="T"))[0][1]

            return np.einsum("i,ij,j->",bra,op,ket)

        N = 0
        # enumerating the virtual edges in the network
        for node1,node2 in self.G.edges():
            self.bra.G[node1][node2][0]["label"] = N
            self.op.G[node1][node2][0]["label"] = N + 1
            self.ket.G[node1][node2][0]["label"] = N + 2
            N += 3
        # enumerating the physical edges in the network
        for node in self.G.nodes():
            self.bra.G.nodes[node]["label"] = [N,]
            self.op.G.nodes[node]["label"] = [N,N+1]
            self.ket.G.nodes[node]["label"] = [N+1,]
            N += 2

        args = ()
        # extracting the einsum arguments
        for node in self.G.nodes():
            for layer in (self.bra.G,self.op.G,self.ket.G):
                args += (layer.nodes[node]["T"],)
                # virtual edges
                legs = [None for _ in range(len(layer.adj[node]))]
                for _,neighbor,edge_label in layer.edges(nbunch=node,data="label"):
                    legs[layer[node][neighbor][0]["legs"][node]] = edge_label
                # physical edges
                legs += layer.nodes[node]["label"]

                args += (legs,)

        return ctg.einsum(*args,optimize="greedy")

    def __contract_ctg_hyperopt(self,target_size:int=2**20,parallel:bool=False,verbose:bool=False,sanity_check:bool=False,**kwargs) -> float:
        """
        Exact contraction using a `cotengra.HyperOptimizer` object.
        * `target_size`: Maximum intermediate tensor size (see [basic slicing](https://cotengra.readthedocs.io/en/latest/advanced.html#basic-slicing-slicing-opts)).
        Standard value was chosen basically arbitrarily.
        * `parallel`: Whether Cotengra uses parallelization.
        """
        # sanity check
        if sanity_check: assert self.intact_check()

        if self.G.number_of_nodes() == 1:
            # the network is trivial
            bra = tuple(self.bra.G.nodes(data="T"))[0][1]
            op = tuple(self.op.G.nodes(data="T"))[0][1]
            ket = tuple(self.ket.G.nodes(data="T"))[0][1]

            return np.einsum("i,ij,j->",bra,op,ket)

        size_dict = {}
        arrays = ()
        inputs = ()
        output = ()

        N = 0
        # enumerating the virtual edges in the network, extracting the size of every edge
        for node1,node2 in self.G.edges():
            # bra
            self.bra.G[node1][node2][0]["label"] = ctg.get_symbol(N)
            size_dict[ctg.get_symbol(N)] = self.bra.G[node1][node2][0]["size"]
            # operator
            self.op.G[node1][node2][0]["label"] = ctg.get_symbol(N+1)
            size_dict[ctg.get_symbol(N+1)] = self.op.chi
            # ket
            self.ket.G[node1][node2][0]["label"] = ctg.get_symbol(N+2)
            size_dict[ctg.get_symbol(N+2)] = self.ket.G[node1][node2][0]["size"]
            N += 3
        # enumerating the physical edges in the network, extracting the size of every edge
        for node in self.G.nodes():
            self.bra.G.nodes[node]["label"] = [ctg.get_symbol(N),]
            self.op.G.nodes[node]["label"] = [ctg.get_symbol(N),ctg.get_symbol(N+1)]
            self.ket.G.nodes[node]["label"] = [ctg.get_symbol(N+1),]
            size_dict[ctg.get_symbol(N)] = self.D
            size_dict[ctg.get_symbol(N+1)] = self.D
            N += 2

        # extracting the einsum arguments
        for node in self.G.nodes():
            for layer in (self.bra.G,self.op.G,self.ket.G):
                arrays += (layer.nodes[node]["T"],)
                # virtual edges
                legs = [None for _ in range(len(layer.adj[node]))]
                for _,neighbor,edge_label in layer.edges(nbunch=node,data="label"):
                    legs[layer[node][neighbor][0]["legs"][node]] = edge_label
                # physical edges
                legs += layer.nodes[node]["label"]

                inputs += (legs,)

        opt = ctg.HyperOptimizer(parallel=parallel,slicing_reconf_opts={"target_size":target_size},progbar=verbose)
        tree = opt.search(inputs=inputs,output=output,size_dict=size_dict)

        try:
            return tree.contract(arrays)
        except ValueError:
            warnings.warn("Contraction not possible using cotengra due to ValueError.")
            return np.nan

    def contract(self,sanity_check:bool=False,**kwargs) -> float:
        """
        Exact contraction.
        """
        # I'm using a guess to distinguish between cases where I should use a HyperOptimizer object, and where I shouldn't.
        #if self.G.number_of_nodes() > 25:
        #    return self.__contract_ctg_einsum(sanity_check=sanity_check)

        return self.__contract_ctg_hyperopt(sanity_check=sanity_check,**kwargs)

    def neighborhood(self,sending_node:int,receiving_node:int,sanity_check:bool=False) -> dict:
        """
        Returns the signature of `contract_tensor_msg` as a dictionary.
        """
        # sanity check
        if sanity_check:
            assert self.intact_check()
            assert self.G.has_node(sending_node) and self.G.has_node(receiving_node)

        return dict(
            msg = {neighbor:self.G[sending_node][neighbor][0]["msg"][sending_node] for neighbor in self.G.adj[sending_node]},
            sending_node = sending_node,
            receiving_node = receiving_node,
            bra_legs = self.bra.legs_dict(sending_node),
            op_legs = self.op.legs_dict(sending_node),
            ket_legs = self.ket.legs_dict(sending_node),
            bra_T = self.bra.G.nodes[sending_node]["T"],
            op_T = self.op.G.nodes[sending_node]["T"],
            ket_T = self.ket.G.nodes[sending_node]["T"]
        )

    @staticmethod
    def graph_compatible(G1:nx.MultiGraph,G2:nx.MultiGraph) -> bool:
        """
        Tests if `G1` and `G2` can be combined into a sandwich, that is
        if their geometry is the same. This amounts to checking if every edge in `G1` is
        contained in `G2`.

        Throws `ValueError` if there are two edges between any two
        nodes in `G1` or `G2`.
        """
        # sanity check
        assert network_message_check(G1)
        assert network_message_check(G2)
        for node1,node2,key in G1.edges(keys=True):
            if key != 0: raise ValueError(f"Edge ({node1},{node2}) is contained multiple times in G1.")
        for node1,node2,key in G2.edges(keys=True):
            if key != 0: raise ValueError(f"Edge ({node1},{node2}) is contained multiple times in G2.")

        # let's check
        for node1,node2 in G1.edges():
            if not G2.has_edge(node1,node2,0): return False

        return True

    @staticmethod
    def prepare_graph(G:nx.MultiGraph,keep_legs:bool=False) -> nx.MultiGraph:
        """
        Creates a shallow copy of `G`, and adds the keys `legs`,
        `trace`, and `indices` to the edges.

        This can be used to remove unwanted data from a graph.
        """
        # shallow copy of G
        newG = nx.MultiGraph(G.edges())
        # adding legs attribute to each edge
        for node1,node2,legs in G.edges(data="legs",keys=False):
            newG[node1][node2][0]["legs"] = legs if keep_legs else {}
            newG[node1][node2][0]["trace"] = False
            newG[node1][node2][0]["indices"] = None
            newG[node1][node2][0]["msg"] = {}

        return newG

    @classmethod
    def Cntr(cls,G:nx.MultiGraph,sanity_check:bool=False):
        """
        Contraction of the tensor network contained in `G`.
        """
        return cls(
            bra = PEPS.Dummy(G,sanity_check=sanity_check),
            op = PEPO.Identity(G=G,D=1,sanity_check=sanity_check),
            ket = PEPS.init_from_TN(G,sanity_check=sanity_check),
            sanity_check = sanity_check
        )

    @classmethod
    def Overlap(cls,psi1:PEPS,psi2:PEPS,sanity_check:bool=False):
        """
        Overlap <`psi1`,`psi2`> of two MPS. Returns the corresponding `Braket` object.
        """
        return cls(
            bra = psi1.conj(sanity_check=sanity_check),
            op = PEPO.Identity(G=psi1.G,D=psi1.D,sanity_check=sanity_check),
            ket = psi2,
            sanity_check = sanity_check
        )

    @classmethod
    def Expval(cls,psi:PEPS,op:PEPO,sanity_check:bool=False):
        """
        Expectation value of the operator `op` for the state `psi`.
        """
        return cls(psi.conj(sanity_check=sanity_check),op,psi,sanity_check=sanity_check)

    def __init__(self,bra:PEPS,op:PEPO,ket:PEPS,sanity_check:bool=False) -> None:
        # sanity check
        if sanity_check:
            assert self.graph_compatible(bra.G,ket.G)
            assert self.graph_compatible(bra.G,op.G)
            assert bra.D == op.D and ket.D == op.D

        self.G:nx.MultiGraph = self.prepare_graph(ket.G,True)
        self.ket:PEPS = ket
        self.bra:PEPS = bra
        self.op:PEPO = op
        self.D:int = op.D

        self.converged:bool=False
        """Indicates whether the messages in `self.G` are converged."""
        self.cntr:float = np.nan
        """Value of the network, calculated by BP."""

        if sanity_check: assert self.intact_check()

        return

class DMRG:
    """
    Single-site DMRG on graphs. Environments are calculated using belief propagation.
    """

    nSweeps:int=5
    """Default number of sweeps."""

    def intact_check(self) -> bool:
        """
        Checks if the DMRG algrithm can be run. This amounts to:
        * Checking if the underlying braket is intact.
        + Checking if expval graph and overlap graph are compatible.
        * Checking if bra and ket are adjoint to one another.
        * Checking if the leg orderings in `self.overlap` and `self.expval` are the same.
        """
        if not self.expval.intact_check(): return False
        if not self.overlap.intact_check(): return False

        # re the physical dimensions the same?
        if not self.overlap.D == self.expval.D:
            warnings.warn("Physical dimensions do not match.")
            return False

        # are expval graph and overlap graph compatible?
        if not Braket.graph_compatible(self.expval.G,self.overlap.G):
            warnings.warn("Graphs of overlap and expval not compatible.")
            return False

        # are bra and ket adjoint?
        for node in self.overlap.G.nodes():
            if not np.allclose(self.overlap.bra.G.nodes[node]["T"].conj(),self.overlap.ket.G.nodes[node]["T"]):
                warnings.warn(f"Bra- and ket-tensors at node {node} in overlap not complex conjugates of one another.")
                return False
        for node in self.expval.G.nodes():
            if not np.allclose(self.expval.bra.G.nodes[node]["T"].conj(),self.expval.ket.G.nodes[node]["T"]):
                warnings.warn(f"Bra- and ket-tensors at node {node} in expval not complex conjugates of one another.")
                return False

        # do overlap and expval contain the same tensors?
        for node in self.expval.G.nodes():
            if not np.allclose(self.expval.ket.G.nodes[node]["T"],self.overlap.ket.G.nodes[node]["T"]):
                warnings.warn(f"Tensor at node {node} is not the same in overlap and expval.")
                return False

        # are the leg orderings the same?
        for node1,node2,legs in self.expval.G.edges(data="legs"):
            if not self.overlap.G.has_edge(node1,node2):
                warnings.warn(f"Edge ({node1},{node2}) present in expval, but not present in overlap.")
                return False
            if self.expval.G[node1][node2][0]["legs"] != legs:
                warnings.warn(f"Leg indices of edge ({node1},{node2}) different in expval and overlap.")
                return False

        return True

    def local_H(self,node:int,threshold:float=1e-6,sanity_check:bool=False) -> np.ndarray:
        """
        Hamiltonian at `node`, by taking messages to be environments.
        `threshold` is the absolute allowed error in the hermiticity of the
        hamiltonian obtained (checked if `sanity_check=True`).
        """
        # sanity check
        if sanity_check: assert self.intact_check()

        # The hamiltonian at node is obtained by tracing out the rest of the network. The environments are approximated by messages
        nLegs = len(self.expval.G.adj[node])
        args = ()
        """Arguments for einsum"""

        out_legs = tuple(range(nLegs)) + (3*nLegs,) + tuple(range(2*nLegs,3*nLegs)) + (3*nLegs+1,)
        # Leg ordering of the local hamiltonian: (bra virtual dimensions, bra physical dimension, ket virtual dimensions, ket physical dimension).
        # The order of the virtual legs is inherited from the "legs" indices on the edges
        vir_dim = 1

        for neighbor in self.expval.G.adj[node]:
            # collecting einsum arguments
            args += (
                self.expval.G[node][neighbor][0]["msg"][node],
                (
                    self.expval.bra.G[node][neighbor][0]["legs"][node], # bra leg
                    nLegs + self.expval.op.G[node][neighbor][0]["legs"][node], # operator leg
                    2 * nLegs + self.expval.ket.G[node][neighbor][0]["legs"][node], # ket leg
                )
            )

            # compiling virtual dimensions for later reshape
            vir_dim *= self.expval.ket.G.nodes[node]["T"].shape[self.expval.ket.G[node][neighbor][0]["legs"][node]]

        args += (
            # operator tensor
            self.expval.op.G.nodes[node]["T"],
            tuple(nLegs + iLeg for iLeg in range(nLegs)) + (3*nLegs,3*nLegs+1),
        )

        Hloc = np.einsum(*args,out_legs,optimize=True)
        Hloc = np.reshape(Hloc,(vir_dim * self.expval.D,vir_dim * self.expval.D))

        if sanity_check: assert is_hermitian(Hloc,threshold=threshold)

        return Hloc

    def local_env(self,node:int,threshold:float=1e-6,sanity_check:bool=False) -> np.ndarray:
        """
        Environment at node. Calculated from `self.overlap`. This amounts to
        stacking and re-shaping messages, that are inflowing to `node`.
        `threshold` is the absolute allowed error in the hermiticity of the
        resulting matrix (checked if `sanity_check=True`).
        """
        # sanity check
        if sanity_check: assert self.intact_check()

        nLegs = len(self.overlap.G.adj[node])
        args = ()
        """Arguments for einsum"""

        out_legs = tuple(range(nLegs)) + (3*nLegs,) + tuple(range(2*nLegs,3*nLegs)) + (3*nLegs+1,)
        # Leg ordering of the local hamiltonian: (bra virtual dimensions, bra physical dimension, ket virtual dimensions, ket physical dimension)
        # The order of the virtual legs is inherited from the "legs" indices on the edges
        vir_dim = 1

        for neighbor in self.overlap.G.adj[node]:
            # collecting einsum arguments
            args += (
                self.overlap.G[node][neighbor][0]["msg"][node],
                (
                    self.overlap.bra.G[node][neighbor][0]["legs"][node], # bra leg
                    nLegs + self.overlap.op.G[node][neighbor][0]["legs"][node], # operator leg (overlap inserts an identity); is traced out
                    2 * nLegs + self.overlap.ket.G[node][neighbor][0]["legs"][node], # ket leg
                )
            )

            # compiling virtual dimensions for later reshape
            vir_dim *= self.overlap.ket.G.nodes[node]["T"].shape[self.overlap.ket.G[node][neighbor][0]["legs"][node]]

        # identity for the physical dimension
        args += (self.overlap.op.I,(3*nLegs,3*nLegs+1))

        N = np.einsum(*args,out_legs,optimize=True)
        N = np.reshape(N,(vir_dim * self.overlap.D,vir_dim * self.overlap.D))

        if sanity_check: assert is_hermitian(N,threshold=threshold)

        return N

    def sweep(self,sanity_check:bool=False,**kwargs) -> float:
        """
        Local update at all sites. `kwargs` are passed to `Braket.BP`.
        Returns the change in energy after the sweep.

        The graph is traversed in breadth-first manner. After each
        local update, new outgoing messages are calculated,
        thereby updating the environments.
        """
        if sanity_check: assert self.intact_check()

        # calculating environments and previous energy
        if not self.overlap.converged: self.overlap.BP(sanity_check=sanity_check,**kwargs)
        if not self.expval.converged: self.expval.BP(sanity_check=sanity_check,**kwargs)
        Eprev = self.E0

        # we'll update tensors and matrices, so as a precaution, we'll set the converged attributes to False
        self.overlap.converged = False
        self.expval.converged = False

        for node in nx.dfs_postorder_nodes(self.expval.G,source=self.expval.op.root):
            H = self.local_H(node,sanity_check=sanity_check)
            N = self.local_env(node,sanity_check=sanity_check)
            # Hloc and env are hermitian, if the BP iteration converged - which it must have if this line is executed

            # generalized eigenvalue problem
            eigvals,eigvecs = scialg.eig(
                a=H,
                #b=N,
                overwrite_a=True,
                overwrite_b=True,
            )

            # re-shaping new statevector
            newshape = [np.nan for neighbor in self.overlap.G.adj[node]]
            for neighbor in self.overlap.G.adj[node]: newshape[self.overlap.G[node][neighbor][0]["legs"][node]] = self.overlap.ket.G[node][neighbor][0]["size"]
            newshape += [self.overlap.D,]
            T = np.reshape(eigvecs[:,np.argmin(eigvals)],newshape)

            # inserting it into PEPS and PEPO
            self.overlap.ket.G.nodes[node]["T"] = T
            self.overlap.bra.G.nodes[node]["T"] = T.conj()
            self.expval.ket.G.nodes[node]["T"] = T
            self.expval.bra.G.nodes[node]["T"] = T.conj()

            # updating messages that node sends
            for neighbor in self.overlap.G.adj[node]:
                msg = self.overlap.contract_tensor_msg(sending_node=node,receiving_node=neighbor,sanity_check=sanity_check)
                self.overlap.G[node][neighbor][0]["msg"][neighbor] = msg
            for neighbor in self.expval.G.adj[node]:
                msg = self.expval.contract_tensor_msg(sending_node=node,receiving_node=neighbor,sanity_check=sanity_check)
                self.expval.G[node][neighbor][0]["msg"][neighbor] = msg

        # calculatig new environments
        self.overlap.BP(sanity_check=sanity_check,**kwargs)
        self.expval.BP(sanity_check=sanity_check,**kwargs)
        Enext = self.E0

        return np.abs(Eprev - Enext)

    def run(self,nSweeps:int=None,verbose:bool=False,sanity_check:bool=False,**kwargs):
        """
        Runs single-site DMRG on the underlying braket.
        `kwargs` are passed to `self.BP`.
        """
        if sanity_check: assert self.intact_check()

        # preparing kwargs
        kwargs["numretries"] = np.inf
        kwargs["verbose"] = False

        nSweeps = nSweeps if nSweeps != None else self.nSweeps
        iterator = tqdm.tqdm(range(nSweeps),desc=f"DMRG sweeps",disable=not verbose)
        eps_list = ()

        for iSweep in iterator:
            eps = self.sweep(sanity_check=sanity_check,**kwargs)
            iterator.set_postfix_str(f"eps = {eps:.3e}")
            eps_list += (eps,)

        return

    def contract(self,sanity_check:bool=True) -> float:
        """
        Exact calculation of the current expectation value.
        """
        expval_cntr = self.expval.contract(sanity_check=sanity_check)
        overlap_cntr = self.overlap.contract(sanity_check=sanity_check)

        return expval_cntr / overlap_cntr

    @property
    def converged(self):
        """Indicates whether the messages in `self.overlap.G` and `self.expval.G` are converged."""
        return self.overlap.converged and self.expval.converged

    @property
    def E0(self):
        """Current best guess of the ground state energy."""
        return self.expval.cntr / self.overlap.cntr

    def __init__(self,op:PEPO,psi_init:PEPS=None,chi:int=None,sanity_check:bool=False):
        # if not given, initial state is chosen randomly
        if psi_init == None:
            psi_init = PEPS.init_random(G=op.G,D=op.D,chi=chi)

        self.expval = Braket.Expval(psi=psi_init,op=op,sanity_check=sanity_check)
        self.overlap = Braket.Overlap(psi1=psi_init,psi2=psi_init)

        if sanity_check: assert self.intact_check()

if __name__ == "__main__":
    pass
