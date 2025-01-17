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
import scipy.linalg as scialg
import ray
import warnings
import itertools
import tqdm

from belief_propagation.utils import network_message_check,crandn
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

class SubscriptableNode:
    """
    Auxiliary class to make the `.msg` property
    of the `Braket` class doubly subscriptable.

    **UNUSED** because this would induce a huge
    memory overhead - and I thought of a more
    elegant solution.
    """
    def __init__(self,neighborhood:dict[int,np.ndarray]):
        self.neighborhood:dict[int,np.ndarray] = neighborhood

    def __getitem__(self,node:int) -> np.ndarray:
        """Access to the messages in a `Braket` instance."""
        return self.neighborhood[node]

class SubscriptableGraph:
    """
    Auxiliary class to make the `.msg` property
    of the `Braket` class doubly subscriptable.

    **UNUSED** because this would induce a huge
    memory overhead - and I thought of a more
    elegant solution.
    """
    def __init__(self,G:nx.MultiGraph):
        self.G:nx.MultiGraph = G

    def __getitem__(self,node:int) -> SubscriptableNode:
        """Access to the messages that are outbound from `node`."""
        return SubscriptableNode({neighbor:self.G[node][neighbor][0]["msg"][neighbor] for neighbor in self.G.adj[node]})

class Braket:
    """
    Base class for sandwiches of MPS and PEPOs. Contains code for belief propagation.
    """

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

        if self.msg != None:
            # are there any messages with non-finite values?
            for sending_node in self.msg.keys():
                for receiving_node in self.msg[sending_node].keys():
                    if not np.isfinite(self.msg[sending_node][receiving_node]).all():
                        warnings.warn(f"Non-finite message sent from {sending_node} to {receiving_node}.")
                        return False

        return True

    def __construct_initial_messages(self,real:bool,normalize:bool,sanity_check:bool,rng:np.random.Generator=np.random.default_rng()) -> None:
        """
        Initial messages for BP iteration. Saved in the dictionary
        `self.msg`.

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

        def get_new_message(bra_size:int,op_size:int,ket_size:int) -> np.ndarray:
            """
            Generates a new message with shape `[bra_size,op_size,ket_size]`.

            If `bra_size=ket_size`, then `msg[:,i,:]` is, for all `i`,
            positive-semidefinite and hermitian.
            """
            if bra_size == ket_size:
                msg = np.zeros(shape=(bra_size,op_size,ket_size)) if real else np.zeros(shape=(bra_size,op_size,ket_size)) + 0j
                for i in range(op_size):
                    A = randn(size=(bra_size,bra_size))
                    msg[:,i,:] = A.T.conj() @ A

                return msg
            else:
                # TODO psd-analogous matrices using scipy.stats.ortho_group / scipy.stats.unitary_group?
                pass

            return randn(size=(bra_size,op_size,ket_size))

        self.msg = {node:{} for node in self.G.nodes()}

        for node1,node2 in self.G.edges():
            for sending_node,receiving_node in itertools.permutations((node1,node2)):
                # messages in both directions
                if len(self.G.adj[sending_node]) > 1:
                    # ket and bra leg indices, and sizes
                    bra_size = self.bra.G[sending_node][receiving_node][0]["size"]
                    ket_size = self.ket.G[sending_node][receiving_node][0]["size"]
                    # calculating the message
                    msg = get_new_message(bra_size,self.op.chi,ket_size)
                else:
                    # sending node is leaf node
                    msg = ctg.einsum(
                        "ij,kjl,rl->ikr",
                        self.bra.G.nodes[sending_node]["T"],
                        self.op.G.nodes[sending_node]["T"],
                        self.ket.G.nodes[sending_node]["T"]
                    )

                if normalize: msg /= np.sum(msg)
                self.msg[sending_node][receiving_node] = msg

        return

    def contract_tensor_msg(self,sending_node:int,receiving_node:int,sanity_check:bool) -> np.ndarray:
        """
        Contracts tensor and messages at `sending_node`, and returns the message
        that flows from `sending_node` to `receiving_node`.
        """
        if sanity_check: assert self.G.has_node(sending_node) and self.G.has_node(receiving_node)

        # The outcoming message on one edge is the result of absorbing all incoming messages on all other edges into the tensor sandwich
        nLegs = len(self.G.adj[sending_node])
        args = ()
        """Arguments for einsum"""

        out_legs = list(range(3 * nLegs))

        for neighbor in self.G.adj[sending_node]:
            if neighbor == receiving_node: continue
            args += (
                self.msg[neighbor][sending_node],
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
        Performs a message passing step. Algorithm taken from Kirkley, 2021
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
            new_msg = {node:{} for node in self.G.nodes()}

            for node1,node2 in self.G.edges():
                for sending_node,receiving_node in itertools.permutations((node1,node2)):
                    # messages in both directions

                    if len(self.G.adj[sending_node]) == 1:
                        # leaf node; no action necessary
                        new_msg[sending_node][receiving_node] = self.msg[sending_node][receiving_node]
                        continue

                    msg = self.contract_tensor_msg(sending_node,receiving_node,sanity_check)

                    # saving the new message
                    new_msg[sending_node][receiving_node] = msg / np.sum(msg) if normalize else msg
                    # change in message norm
                    eps += (np.linalg.norm(self.msg[sending_node][receiving_node] - new_msg[sending_node][receiving_node]),)

            # put new messages in the graph
            self.msg = new_msg

            return np.max(eps)

        # parallel version
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
            eps += (np.linalg.norm(self.msg[sending_node][receiving_node] - msg / np.sum(msg) if normalize else msg),)
            # saving the new message
            self.msg[sending_node][receiving_node] = msg / np.sum(msg) if normalize else msg

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
        Normalize messages, such that at each site, contraction
        of a tensor with it's inbound messages yields the
        complete network value.

        When `self.BP` is executed with `normalize=False`, this
        is already the case. This function is thus only necessary
        when messages were obtained with `normalize=True`.
        """
        # sanity check
        if sanity_check: assert self.intact_check()

        # contract messages into tensors first to obtain tensor values, if necessary
        if not "cntr" in self.G.nodes[self.op.root].keys(): self.__contract_tensors_inbound_messages(sanity_check=sanity_check)

        for node in self.G.nodes():
            norm = np.real_if_close((self.cntr / self.G.nodes[node]["cntr"]) ** (1 / len(self.G.adj[node])))

            for neighbor in self.G.adj[node]: self.msg[neighbor][node] *= norm

        return
    
    def __contract_edge_opposite_messages(self,sanity_check:bool=False) -> None:
        """
        Contracts the messages travelling in each direction of an
        edge, on every edge. Value is saved under the key `cntr`.
        """
        # sanity check
        if sanity_check: assert self.intact_check()

        for node1,node2 in self.G.edges():
            self.G[node1][node2][0]["cntr"] = ctg.einsum("ijk,ijk->",self.msg[node1][node2],self.msg[node2][node1])

        return

    def __contract_tensors_inbound_messages(self,sanity_check:bool=False) -> None:
        """
        Contracts all messages into the respective nodes, and saves the value in each node.
        """
        # sanity check
        if sanity_check: assert self.intact_check()

        for node in self.G.nodes():
            nLegs = len(self.G.adj[node])
            args = ()

            for neighbor in self.G.adj[node]:
                args += (
                    self.msg[neighbor][node],
                    (
                        self.bra.G[node][neighbor][0]["legs"][node], # bra leg
                        nLegs + self.op.G[node][neighbor][0]["legs"][node], # operator leg
                        2 * nLegs + self.ket.G[node][neighbor][0]["legs"][node], # ket leg
                    )
                )

            args += (
                # bra tensor
                self.bra.G.nodes[node]["T"],
                tuple(range(nLegs)) + (3 * nLegs,),
                # operator tensor
                self.op.G.nodes[node]["T"],
                tuple(nLegs + iLeg for iLeg in range(nLegs)) + (3 * nLegs,3 * nLegs + 1),
                # ket tensor
                self.ket.G.nodes[node]["T"],
                tuple(2 * nLegs + iLeg for iLeg in range(nLegs)) + (3 * nLegs + 1,),
            )

            self.G.nodes[node]["cntr"] = ctg.einsum(*args,optimize="greedy")

        return

    def BP(self,numiter:int=500,numretries:float=10,real:bool=False,normalize:bool=True,threshold:float=1e-10,parallel:bool=False,verbose:bool=True,sanity_check:bool=False,**kwargs) -> None:
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

        Writes the network contraction value to `self.cntr`. Converged messages are normalized
        such that the contraction of a tensor with it's inbound messages gives the complete network
        value, at every site.

        If the algorithm converges, the flag `self.converged` is set to `True`.
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

        # contract tensors and messages, opposite messages
        self.__contract_tensors_inbound_messages(sanity_check=sanity_check)
        self.__contract_edge_opposite_messages(sanity_check=sanity_check)

        if normalize:
            # the network value is the product of all node values, divided by all edge values
            self.cntr = 1
            for node,node_cntr in self.G.nodes(data="cntr"): self.cntr *= node_cntr
            for node1,node2,edge_cntr in self.G.edges(data="cntr"): self.cntr /= edge_cntr

            # normalizing messages
            self.__normalize_messages(sanity_check=sanity_check)
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
        * `target_size`: Maximum intermediate tensor size (see
        [basic slicing](https://cotengra.readthedocs.io/en/latest/advanced.html#basic-slicing-slicing-opts)).
        Standard value was chosen basically arbitrarily.
        * `parallel`: Whether Cotengra uses parallelization.
        """
        # sanity check
        if sanity_check: assert self.intact_check()

        if self.nsites == 1:
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
            msg = {neighbor:self.msg[neighbor][sending_node] for neighbor in self.G.adj[sending_node]},
            sending_node = sending_node,
            receiving_node = receiving_node,
            bra_legs = self.bra.legs_dict(sending_node),
            op_legs = self.op.legs_dict(sending_node),
            ket_legs = self.ket.legs_dict(sending_node),
            bra_T = self.bra.G.nodes[sending_node]["T"],
            op_T = self.op.G.nodes[sending_node]["T"],
            ket_T = self.ket.G.nodes[sending_node]["T"]
        )

    @property
    def nsites(self) -> int:
        """
        Number of sites on which the braket is defined.
        """
        return self.op.nsites

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
        Overlap <`psi1`,`psi2`> of two PEPS. Returns the corresponding `Braket` object.
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
        self.bra:PEPS = bra
        self.op:PEPO = op
        self.ket:PEPS = ket
        self.D:int = op.D

        self.msg:dict[int,dict[int,np.ndarray]] = None
        """First key sending node, second key receiving node."""

        self.converged:bool=False
        """Indicates whether the messages in `self.G` are converged."""
        self.cntr:float = np.nan
        """Value of the network, calculated by BP."""

        if sanity_check: assert self.intact_check()

        return

    def __getitem__(self,node:int) -> tuple[np.ndarray]:
        """
        Subscripting with a node gives the tensor stack `(bra[node],op[node],ket[node])` at that node.
        """
        return (self.bra[node],self.op[node],self.ket[node])

    def __setitem__(self,node:int,Tstack:tuple[np.ndarray]) -> None:
        """
        Changing tensors directly.
        """
        if not len(Tstack) == 3: raise ValueError(f"Tensor stacks must consist of three tensors. received {len(Tstack)} tensors.")

        self.bra[node] = Tstack[0]
        self.op[node] = Tstack[1]
        self.ket[node] = Tstack[2]

        return

def BP_compression(psi:PEPS,singval_threshold:float=1e-8,sanity_check:bool=False,**kwargs) -> PEPS:
    """
    L2BP compression from [Sci. Adv. 10, eadk4321 (2024)](https://doi.org/10.1126/sciadv.adk4321).
    Singular values below `singval_threshold` are discarded. `kwargs` are passed to
    `Braket.BP`. `psi` is manipulated in-place.
    """
    if sanity_check: assert psi.intact_check()

    # handling kwargs
    kwargs["iterator_desc_prefix"] = kwargs["iterator_desc_prefix"] + " | L2BP compression" if "iterator_desc_prefix" in kwargs.keys() else "L2BP compression"
    kwargs["sanity_check"] = sanity_check

    # BP iteration
    overlap = Braket.Overlap(psi,psi,sanity_check=sanity_check)
    overlap.BP(**kwargs)

    # compressing every edge
    for node1,node2 in psi.G.edges():
        size = overlap.ket.G[node1][node2][0]["size"]
        ndim1 = psi.G.nodes[node1]["T"].ndim
        ndim2 = psi.G.nodes[node2]["T"].ndim

        # get messages
        msg_12 = np.reshape(overlap.msg[node1][node2][:,0,:],newshape=(size,size))
        msg_21 = np.reshape(overlap.msg[node2][node1][:,0,:],newshape=(size,size))

        # splitting the messages
        eigvals1,W1 = scialg.eigh(msg_12,overwrite_a=True)
        eigvals2,W2 = scialg.eigh(msg_21,overwrite_a=True)
        R1 = np.diag(np.sqrt(eigvals1)) @ W1.conj().T
        R2 = np.diag(np.sqrt(eigvals2)) @ W2.conj().T

        # SVD over the bond, and truncation
        U,singvals,Vh = scialg.svd(R1 @ R2,full_matrices=False,overwrite_a=True)
        nonzero_mask = np.logical_not(np.isclose(singvals,0,atol=singval_threshold))

        if np.sum(nonzero_mask) == 0:
            warnings.warn(f"Threshold {singval_threshold:.3e} cuts edge ({node1},{node2}). Setting bond dimension to one.")
            nonzero_mask[0] = True

        U = U[:,nonzero_mask]
        Vh = Vh[nonzero_mask,:]
        singvals = singvals[nonzero_mask]

        # projectors
        P1 = np.einsum("ij,jk,kl->il",R2,Vh.conj().T,np.diag(1 / np.sqrt(singvals)),optimize=True)
        P2 = np.einsum("ij,jk,kl->il",np.diag(1 / np.sqrt(singvals)),U.conj().T,R1,optimize=True)

        print(f"edge ({node1},{node2}): projector distance = {np.linalg.norm(np.eye(N=P1.shape[0]) - P1 @ P2)}")

        # absorbing projector 1 tensor into node 1
        Tlegs = tuple(range(ndim1))
        Plegs = (overlap.ket.G[node1][node2][0]["legs"][node1],ndim1)
        outlegs = list(range(ndim1))
        outlegs[overlap.ket.G[node1][node2][0]["legs"][node1]] = ndim1
        psi[node1] = np.einsum(
            psi[node1],Tlegs,
            P1,Plegs,
            outlegs,
            optimize=True
        )

        # absorbing projector 2 tensor into node 2
        Tlegs = tuple(range(ndim2))
        Plegs = (ndim2,overlap.ket.G[node1][node2][0]["legs"][node2])
        outlegs = list(range(ndim2))
        outlegs[overlap.ket.G[node1][node2][0]["legs"][node2]] = ndim2
        psi[node2] = np.einsum(
            P2,Plegs,
            psi[node2],Tlegs,
            outlegs,
            optimize=True
        )

        # updating size of edge
        overlap.ket.G[node1][node2][0]["size"] = P1.shape[1]

    if sanity_check: assert psi.intact_check()

    return

if __name__ == "__main__":
    pass
