"""
Creating sandwiches of the form
* MPS @ PEPO @ MPS, or
* MPS @ MPS,

by combining the classes MPS and PEPO.
The class `Braket` contained herein implements the Belief Propagation
algorithm.
"""

import numpy as np
import networkx as nx
import cotengra as ctg
import scipy.linalg as scialg
import ray
import warnings
import itertools
import tqdm
from typing import Union
import copy

from belief_propagation.utils import network_message_check,crandn
from belief_propagation.networks import expose_edge
from belief_propagation.hamiltonians import Identity
from belief_propagation.PEPO import PEPO
from belief_propagation.PEPS import PEPS

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

class BaseBraket:
    """
    Base class for sandwiches of MPS and PEPOs.
    Always describes a braket-object of the form `<bra|op|ket>`.
    """

    def __contract_ctg_einsum(self,sanity_check:bool=False,**kwargs) -> float:
        """
        Exact contraction using `ctg.einsum`.
        """
        # sanity check
        if sanity_check: assert self.intact

        if self.G.number_of_nodes() == 1:
            # the network is trivial
            bra = tuple(self._bra.G.nodes(data="T"))[0][1]
            op = tuple(self._op.G.nodes(data="T"))[0][1]
            ket = tuple(self._ket.G.nodes(data="T"))[0][1]

            return np.einsum("i,ij,j->",bra,op,ket)

        N = 0
        # enumerating the virtual edges in the network
        for node1,node2 in self.G.edges():
            self._bra.G[node1][node2][0]["label"] = N
            self._op.G[node1][node2][0]["label"] = N + 1
            self._ket.G[node1][node2][0]["label"] = N + 2
            N += 3
        # enumerating the physical edges in the network
        for node in self.G.nodes():
            self._bra.G.nodes[node]["label"] = [N,]
            self._op.G.nodes[node]["label"] = [N,N+1]
            self._ket.G.nodes[node]["label"] = [N+1,]
            N += 2

        args = ()
        # extracting the einsum arguments
        for node in self.G.nodes():
            for layer in (self._bra.G,self._op.G,self._ket.G):
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
        if sanity_check: assert self.intact

        if self.nsites == 1:
            # the network is trivial
            bra = tuple(self._bra.G.nodes(data="T"))[0][1]
            op = tuple(self._op.G.nodes(data="T"))[0][1]
            ket = tuple(self._ket.G.nodes(data="T"))[0][1]

            return np.einsum("i,ij,j->",bra,op,ket)

        size_dict = {}
        arrays = ()
        inputs = ()
        output = ()

        N = 0
        # enumerating the virtual edges in the network, extracting the size of every edge
        for node1,node2 in self.G.edges():
            # bra
            self._bra.G[node1][node2][0]["label"] = ctg.get_symbol(N)
            size_dict[ctg.get_symbol(N)] = self._bra.G[node1][node2][0]["size"]
            # operator
            self._op.G[node1][node2][0]["label"] = ctg.get_symbol(N+1)
            size_dict[ctg.get_symbol(N+1)] = self._op.chi
            # ket
            self._ket.G[node1][node2][0]["label"] = ctg.get_symbol(N+2)
            size_dict[ctg.get_symbol(N+2)] = self._ket.G[node1][node2][0]["size"]
            N += 3
        # enumerating the physical edges in the network, extracting the size of every edge
        for node in self.G.nodes():
            self._bra.G.nodes[node]["label"] = [ctg.get_symbol(N),]
            self._op.G.nodes[node]["label"] = [ctg.get_symbol(N),ctg.get_symbol(N+1)]
            self._ket.G.nodes[node]["label"] = [ctg.get_symbol(N+1),]
            size_dict[ctg.get_symbol(N)] = self.D
            size_dict[ctg.get_symbol(N+1)] = self.D
            N += 2

        # extracting the einsum arguments
        for node in self.G.nodes():
            for layer in (self._bra.G,self._op.G,self._ket.G):
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

    @property
    def nsites(self) -> int:
        """
        Number of sites on which the braket is defined.
        """
        return self._op.nsites

    @property
    def intact(self) -> bool:
        """
        Whether the braket is intact or not. This includes:
        * Is the network `G` message-ready?
        * Are `bra`, `op`, and `ket` themselves intact?
        * Do the physical dimensions match?
        * Do all the messages contain finite values?
        """
        assert hasattr(self,"_bra")
        assert hasattr(self,"_op")
        assert hasattr(self,"_ket")
        assert hasattr(self,"D")

        if not network_message_check(self.G): return False

        if not self._bra.intact: return False
        if not self._op.intact: return False
        if not self._ket.intact: return False

        # do the physical dimensions match?
        if not self._bra.D == self.D and self._ket.D == self.D and self._op.D == self.D:
            warnings.warn("Physical dimensions in braket do not match.")
            return False

        return True

    @property
    def converged(self) -> bool:
        """Whether the messages in `self.msg` are converged."""
        return self._converged

    @property
    def ket(self) -> PEPS:
        """
        The ket-state in the braket `bra @ op @ ket`.
        """
        return self._ket

    @ket.setter
    def ket(self,ket:PEPS) -> None:
        # sanity check
        assert self.graph_compatible(self.G,ket.G)
        self._ket = ket

        # it is no longer guaranteed that the messages are converged
        self._converged = False

        return

    @property
    def bra(self) -> PEPS:
        """
        The bra-state in the braket `bra @ op @ ket`.
        """
        return self._bra

    @bra.setter
    def bra(self,bra:PEPS) -> None:
        # sanity check
        assert self.graph_compatible(self.G,bra.G)
        self._bra = bra

        # it is no longer guaranteed that the messages are converged
        self._converged = False

        return

    @property
    def op(self) -> PEPO:
        """
        The operator in the braket `bra @ op @ ket`.
        """
        return self._op

    @op.setter
    def op(self,op:PEPO) -> None:
        # sanity check
        assert self.graph_compatible(self.G,op.G)
        self._op = op

        # it is no longer guaranteed that the messages are converged
        self._converged = False

        return

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
        assert nx.utils.nodes_equal(G1.nodes(),G2.nodes())
        assert nx.utils.edges_equal(G1.edges(),G2.edges())

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

            # default: no transformation on any edge
            newG[node1][node2][0]["T"] = np.full(shape=(1,),fill_value=np.nan)

        return newG

    def __init__(self,bra:PEPS,op:PEPO,ket:PEPS,sanity_check:bool=False) -> None:
        # sanity check
        if sanity_check:
            assert self.graph_compatible(bra.G,ket.G)
            assert self.graph_compatible(bra.G,op.G)
            assert bra.D == op.D and ket.D == op.D

        self.G:nx.MultiGraph = self.prepare_graph(ket.G,True)
        self._bra:PEPS = bra
        self._op:PEPO = op
        self._ket:PEPS = ket
        self.D:int = op.D
        """Physical dimension."""

        self._converged:bool = False
        """Whether the messages in `self.msg` are converged."""
        # this attribute does not really belong in BaseBraket, of course;
        # I included it here to be able to have the property setters in
        # this base class. The whole point of the base class was to avoid
        # clutter in the BP code

        if sanity_check: assert self.intact

        return

    def __getitem__(self,node:int) -> tuple[np.ndarray]:
        """
        Subscripting with a node gives the tensor stack `(bra[node],op[node],ket[node])` at that node.
        """
        return (self._bra[node],self._op[node],self._ket[node])

    def __setitem__(self,node:int,Tstack:tuple[np.ndarray]) -> None:
        """
        Changing tensors directly.
        """
        if not len(Tstack) == 3: raise ValueError(f"Tensor stacks must consist of three tensors. received {len(Tstack)} tensors.")

        self._bra[node] = Tstack[0]
        self._op[node] = Tstack[1]
        self._ket[node] = Tstack[2]

        # it is no longer guaranteed that the messages are converged
        self._converged = False

        return

class Braket(BaseBraket):
    """
    Code for belief propagation.
    """

    def __construct_initial_messages(self,real:bool,normalize_during:bool,sanity_check:bool,rng:np.random.Generator=np.random.default_rng()) -> None:
        """
        Initial messages for BP iteration. Saved in the dictionary
        `self.msg`.

        Messages are three-index tensors, where the first index belongs
        to the bra, the second index belongs to the operator and the third
        one belongs to the ket.
        """
        # sanity check
        if sanity_check: assert self.intact

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
                    bra_size = self._bra.G[sending_node][receiving_node][0]["size"]
                    ket_size = self._ket.G[sending_node][receiving_node][0]["size"]
                    # calculating the message
                    msg = get_new_message(bra_size,self._op.chi,ket_size)
                else:
                    # sending node is leaf node
                    msg = ctg.einsum(
                        "ij,kjl,rl->ikr",
                        self._bra.G.nodes[sending_node]["T"],
                        self._op.G.nodes[sending_node]["T"],
                        self._ket.G.nodes[sending_node]["T"]
                    )

                if normalize_during: msg /= np.sum(msg)
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
                    self._bra.G[sending_node][neighbor][0]["legs"][sending_node], # bra leg
                    nLegs + self._op.G[sending_node][neighbor][0]["legs"][sending_node], # operator leg
                    2 * nLegs + self._ket.G[sending_node][neighbor][0]["legs"][sending_node], # ket leg
                )
            )
            out_legs.remove(self._bra.G[sending_node][neighbor][0]["legs"][sending_node])
            out_legs.remove(nLegs + self._op.G[sending_node][neighbor][0]["legs"][sending_node])
            out_legs.remove(2 * nLegs + self._ket.G[sending_node][neighbor][0]["legs"][sending_node])

        args += (
            # bra tensor
            self._bra.G.nodes[sending_node]["T"],
            tuple(range(nLegs)) + (3 * nLegs,),
            # operator tensor
            self._op.G.nodes[sending_node]["T"],
            tuple(nLegs + iLeg for iLeg in range(nLegs)) + (3 * nLegs,3 * nLegs + 1),
            # ket tensor
            self._ket.G.nodes[sending_node]["T"],
            tuple(2 * nLegs + iLeg for iLeg in range(nLegs)) + (3 * nLegs + 1,),
        )

        msg = np.einsum(*args,out_legs,optimize=True)

        return msg

    def __pass_msg_through_edges(self,sanity_check:bool) -> None:
        """
        Passes messages along edges. This involves linear transformations
        that live on the edges of the braket, such as projections.
        """
        # sanity check
        if sanity_check: assert self.intact

        for node1,node2,T in self.G.edges(data="T"):
            if np.isnan(T).any():
                # no transformation on this edge
                continue

            sending_node = min(node1,node2)
            receiving_node = max(node1,node2)

            self.msg[sending_node][receiving_node] = np.einsum("ijkabc,abc->ijk",T,self.msg[sending_node][receiving_node])
            self.msg[receiving_node][sending_node] = np.einsum("abcijk,abc->ijk",T,self.msg[receiving_node][sending_node])

    def __message_passing_step(self,normalize_during:bool,parallel:bool,sanity_check:bool) -> float:
        """
        Performs a message passing step. Algorithm taken from Kirkley, 2021
        ([Sci. Adv. 7, eabf1211 (2021)](https://doi.org/10.1126/sciadv.abf1211)).
        Returns the maximum change of message norm over the entire graph.

        Parallelized using ray.
        """
        # sanity check
        if sanity_check: assert self.intact

        if all([len(self.G.adj[node]) <= 1 for node in self.G.nodes]):
            # there are only leaf nodes in the graph; we don't need to do anything
            return 0

        eps = ()

        if not parallel: # non-parallel version of the code
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
                    new_msg[sending_node][receiving_node] = msg
                    # change in message norm
                    eps += (np.linalg.norm(self.msg[sending_node][receiving_node] - (msg / np.sum(msg) if normalize_during else msg)),)

            # put new messages in the graph
            self.msg = new_msg

        else: # parallel version
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
                eps += (np.linalg.norm(self.msg[sending_node][receiving_node] - msg / np.sum(msg) if normalize_during else msg),)
                # saving the new message
                self.msg[sending_node][receiving_node] = msg

        # passing messages through the edges
        self.__pass_msg_through_edges(sanity_check=sanity_check)

        if normalize_during:
            # normalize messages to unity
            self.__normalize_messages(normalize_to="unity",sanity_check=sanity_check)

        return max(eps)

    def __message_passing_iteration(self,numiter:int,real:bool,normalize_during:bool,threshold:float,parallel:bool,iterator_desc_prefix:str,verbose:bool,new_messages:bool,sanity_check:bool) -> tuple[float]:
        """
        Performs a message passing iteration. Returns the change `eps` in maximum
        message norm for every iteration.
        """
        # sanity check
        if sanity_check: assert self.intact

        iterator = tqdm.tqdm(range(numiter),desc=iterator_desc_prefix + f"BP iteration",disable=not verbose)

        eps_list = ()
        # message initialization
        if new_messages: self.__construct_initial_messages(real=real,normalize_during=normalize_during,sanity_check=sanity_check)

        for i in iterator:
            eps = self.__message_passing_step(normalize_during=normalize_during,parallel=parallel,sanity_check=sanity_check)
            iterator.set_postfix_str(f"eps = {eps:.3e}")
            eps_list += (eps,)

            if eps < threshold:
                if verbose: iterator.set_postfix_str(f"eps = {eps:.3e}; threshold reached; returning.")
                iterator.close()
                return eps_list

        return eps_list

    def __normalize_messages(self,normalize_to:str,sanity_check:bool) -> None:
        """
        Normalize messages, such that at each site, contraction
        of a tensor with it's inbound messages yields the
        complete network value.

        When `self.BP` is executed with `normalize=False`, this
        is already the case. This function is thus only necessary
        when messages were obtained with `normalize=True`.
        """
        # sanity check
        if sanity_check: assert self.intact

        if normalize_to == "unity":
            for sending_node in self.msg.keys():
                for receiving_node in self.msg[sending_node].keys():
                    self.msg[sending_node][receiving_node] /= np.sum(self.msg[sending_node][receiving_node])

            return

        if normalize_to == "cntr":
            # contract messages into tensors first to obtain tensor values, if necessary
            if not "cntr" in self.G.nodes[self._op.root].keys(): self.__contract_tensors_inbound_messages(sanity_check=sanity_check)

            for node in self.G.nodes():
                norm = np.real_if_close((self.cntr / self.G.nodes[node]["cntr"]) ** (1 / len(self.G.adj[node])))

                for neighbor in self.G.adj[node]: self.msg[neighbor][node] *= norm

            return
    
        raise NotImplementedError("Message normalization " + normalize_to + " not implemented.")

    def __contract_edge_opposite_messages(self,sanity_check:bool=False) -> None:
        """
        Contracts the messages travelling in each direction of an
        edge, on every edge. Value is saved under the key `cntr`.
        """
        # sanity check
        if sanity_check: assert self.intact

        for node1,node2 in self.G.edges():
            self.G[node1][node2][0]["cntr"] = ctg.einsum("ijk,ijk->",self.msg[node1][node2],self.msg[node2][node1])

        return

    def __contract_tensors_inbound_messages(self,sanity_check:bool=False) -> None:
        """
        Contracts all messages into the respective nodes, and saves the value in each node.
        """
        # sanity check
        if sanity_check: assert self.intact

        for node in self.G.nodes():
            nLegs = len(self.G.adj[node])
            args = ()

            for neighbor in self.G.adj[node]:
                args += (
                    self.msg[neighbor][node],
                    (
                        self._bra.G[node][neighbor][0]["legs"][node], # bra leg
                        nLegs + self._op.G[node][neighbor][0]["legs"][node], # operator leg
                        2 * nLegs + self._ket.G[node][neighbor][0]["legs"][node], # ket leg
                    )
                )

            args += (
                # bra tensor
                self._bra.G.nodes[node]["T"],
                tuple(range(nLegs)) + (3 * nLegs,),
                # operator tensor
                self._op.G.nodes[node]["T"],
                tuple(nLegs + iLeg for iLeg in range(nLegs)) + (3 * nLegs,3 * nLegs + 1),
                # ket tensor
                self._ket.G.nodes[node]["T"],
                tuple(2 * nLegs + iLeg for iLeg in range(nLegs)) + (3 * nLegs + 1,),
            )

            self.G.nodes[node]["cntr"] = ctg.einsum(*args,optimize="greedy")

        return

    def BP(self,numiter:int=500,trials:int=10,real:bool=False,normalize_during:bool=True,normalize_after:bool=True,threshold:float=1e-10,parallel:bool=False,verbose:bool=True,new_messages:bool=True,sanity_check:bool=False,**kwargs) -> None:
        """
        Layz BP algorithm from [Sci. Adv. 10, eadk4321 (2024)](https://doi.org/10.1126/sciadv.adk4321). Parameters:
        * `numiter`: Number of iterations.
        * `trials`: Number of times the BP iteration is attempted. The value `np.inf` can be supplied, in which
        case the algorithm terminates only when a trial reaches `threshold`.
        * `real`: Initialization of messages with real values (otherwise complex).
        * `normalize_during`: Normalization of messages after each message passing iteration. If `normalize=True`,
        this function implements the BP algorithm from
        [Sci. Adv. 7, eabf1211 (2021)](https://doi.org/10.1126/sciadv.abf1211). Otherwise,
        the algorithm becomes Belief Propagation on trees.
        * `normalize_after`: Normalization of messages after each the complete BP iteration.
        If `True`, contraction of a node with it's incoming messages yields the complete network
        value. Only relevant if `normalize_during = True`.
        * `threshold`: When to abort the BP iteration.
        * `new_messages`: Whether or not to initialize new messages.

        Writes the network contraction value to `self.cntr`. Converged messages are normalized
        such that the contraction of a tensor with it's inbound messages gives the complete network
        value, at every site.

        If the algorithm converges, the flag `self.converged` is set to `True`.
        """
        # sanity check
        if sanity_check: assert self.intact

        if self.G.number_of_nodes() == 1:
            warnings.warn("The network is trivial.")
            return

        # handling kwargs
        kwargs["iterator_desc_prefix"] = kwargs["iterator_desc_prefix"] + " | " if "iterator_desc_prefix" in kwargs.keys() else ""
        if trials == 0: warnings.warn(f"Braket.BP received trials = 0. This results in no BP iteration attempt.")

        # initially, the messages are not converged
        self._converged = False

        iTrial = 0
        while iTrial < trials:
            # message passing iteration
            eps_list = self.__message_passing_iteration(
                numiter=numiter,
                real=real,
                normalize_during=normalize_during,
                threshold=threshold,
                parallel=parallel,
                iterator_desc_prefix=kwargs["iterator_desc_prefix"] + f"trial {iTrial+1} | ",
                verbose=verbose,
                new_messages=new_messages,
                sanity_check=sanity_check,
            )

            iTrial += 1

            if eps_list[-1] < threshold:
                self._converged = True
                break

        # contract tensors and messages, opposite messages
        self.__contract_tensors_inbound_messages(sanity_check=sanity_check)
        self.__contract_edge_opposite_messages(sanity_check=sanity_check)

        if normalize_during:
            # the network value is the product of all node values, divided by all edge values
            self.cntr = 1
            for node,node_cntr in self.G.nodes(data="cntr"): self.cntr *= node_cntr
            for node1,node2,edge_cntr in self.G.edges(data="cntr"): self.cntr /= edge_cntr

            # normalizing messages
            if normalize_after: self.__normalize_messages(normalize_to="cntr",sanity_check=sanity_check)
        else:
            # each node carries the network value
            self.cntr = self.G.nodes[self._op.root]["cntr"]

        return

    def neighborhood(self,sending_node:int,receiving_node:int,sanity_check:bool=False) -> dict:
        """
        Returns the signature of `contract_tensor_msg` as a dictionary.
        """
        # sanity check
        if sanity_check:
            assert self.intact
            assert self.G.has_node(sending_node) and self.G.has_node(receiving_node)

        return dict(
            msg = {neighbor:self.msg[neighbor][sending_node] for neighbor in self.G.adj[sending_node]},
            sending_node = sending_node,
            receiving_node = receiving_node,
            bra_legs = self._bra.legs_dict(sending_node),
            op_legs = self._op.legs_dict(sending_node),
            ket_legs = self._ket.legs_dict(sending_node),
            bra_T = self._bra.G.nodes[sending_node]["T"],
            op_T = self._op.G.nodes[sending_node]["T"],
            ket_T = self._ket.G.nodes[sending_node]["T"]
        )

    @property
    def intact(self) -> bool:
        """
        Whether the braket is intact or not. This includes:
        * Is the network `G` message-ready?
        * Are `bra`, `op`, and `ket` themselves intact?
        * Do the physical dimensions match?
        * Do all the messages contain finite values?
        * Do all edge transformations have correct domains
        and images?
        """
        if not super().intact: return False

        if self.msg != None:
            # are there any messages with non-finite values?
            for sending_node in self.msg.keys():
                for receiving_node in self.msg[sending_node].keys():
                    if not np.isfinite(self.msg[sending_node][receiving_node]).all():
                        warnings.warn(f"Non-finite message sent from {sending_node} to {receiving_node}.")
                        return False

        for node1,node2,T in self.G.edges(data="T"):
            if np.isnan(T).any():
                # no transformation on this edge
                continue

            if not T.shape == (
                self._bra.G[node1][node2][0]["size"],
                self._op.G[node1][node2][0]["size"],
                self._ket.G[node1][node2][0]["size"],
                self._bra.G[node1][node2][0]["size"],
                self._op.G[node1][node2][0]["size"],
                self._ket.G[node1][node2][0]["size"],
            ):
                return False

        return True

    @classmethod
    def Cntr(cls,G:nx.MultiGraph,sanity_check:bool=False):
        """
        Contraction of the tensor network contained in `G`.
        """
        return cls(
            bra = PEPS.Dummy(G,sanity_check=sanity_check),
            op = Identity(G=G,D=1,sanity_check=sanity_check),
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
            op = Identity(G=psi1.G,D=psi1.D,sanity_check=sanity_check),
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
        super().__init__(bra=bra,op=op,ket=ket,sanity_check=False)

        self.msg:dict[int,dict[int,np.ndarray]] = None
        """First key sending node, second key receiving node."""

        self.cntr:float = np.nan
        """Value of the network, calculated by BP."""

        if sanity_check: assert self.intact

        return

def L2BP_compression(psi:PEPS,singval_threshold:float=1e-10,sanity_check:bool=False,**kwargs) -> None:
    """
    L2BP compression from [Sci. Adv. 10, eadk4321 (2024)](https://doi.org/10.1126/sciadv.adk4321).
    Singular values below `singval_threshold` are discarded. `kwargs` are passed to
    `Braket.BP`. `psi` is manipulated in-place.
    """
    if sanity_check: assert psi.intact

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
        eigvals1,W1 = scialg.eig(msg_12,overwrite_a=True)
        eigvals2,W2 = scialg.eig(msg_21,overwrite_a=True)
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

        # absorbing projector 1 into node 1
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

        # absorbing projector 2 into node 2
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
        psi.G[node1][node2][0]["size"] = P1.shape[1]

    if sanity_check: assert psi.intact

    return

def QR_gauging(psi:PEPS,tree:nx.DiGraph=None,nodes:tuple[int]=None,sanity_check:bool=False,**kwargs) -> None:
    """
    Gauging of a state using QR decompositions. The root node of `tree` is
    the orthogonality center; if `tree` is not given, a breadth-first search spanning
    tree will be used. If given, only the nodes in `nodes` will be gauged.
    """
    if sanity_check: assert psi.intact

    if tree == None:
        # orthogonality center will be the node with the largest number of neighborhoods
        ortho_center = 0
        max_degree = 0
        for node in psi.G.nodes():
            if len(psi.G.adj[node]) > max_degree:
                ortho_center = node
                max_degree = len(psi.G.adj[node])
        tree = nx.bfs_tree(G=psi.G,source=ortho_center)
    else:
        if not isinstance(tree,nx.DiGraph): raise ValueError("tree must be an oriented graph.")
        if not nx.is_tree(tree): raise ValueError("Given spanning tree is not actually a tree.")
        # finding the orthogonality center
        ortho_center = None
        for node in tree.nodes():
            if tree.in_degree(node) == 0:
                ortho_center = node
                break

    if nodes == None: nodes = tuple(psi.G.nodes())

    # QR decompositions in upstream direction of the tree
    for node in nx.dfs_postorder_nodes(tree,source=ortho_center):
        if node not in nodes: continue

        if node == ortho_center:
            # we have reached the source
            continue

        # finding the upstream neighbor
        assert tree.in_degree(node) == 1
        pred = [_ for _ in tree.pred[node]][0]

        # exposing the upstream leg of the site tensor, and re-shaping
        T_exposed = np.moveaxis(psi[node],source=psi.G[pred][node][0]["legs"][node],destination=-1)
        oldshape = T_exposed.shape
        T_exposed = np.reshape(T_exposed,newshape=(-1,psi.G[pred][node][0]["size"]))

        # QR decomposition
        Q,R = np.linalg.qr(T_exposed,mode="reduced")

        # re-shaping Q, and inserting into the state
        Q = np.reshape(Q,newshape=oldshape)
        Q = np.moveaxis(Q,source=-1,destination=psi.G[pred][node][0]["legs"][node])
        psi[node] = Q

        # absorbing R into upstream node
        upstream_legs = tuple(range(psi[pred].ndim))
        out_legs = tuple(psi[pred].ndim if i == psi.G[pred][node][0]["legs"][pred] else i for i in range(psi[pred].ndim))
        psi[pred] = np.einsum(
            psi[pred],upstream_legs,
            R,(psi[pred].ndim,psi.G[pred][node][0]["legs"][pred]),
            out_legs,
        )

    return

def feynman_cut(obj:Union[PEPS,PEPO,Braket],node1:int,node2:int,sanity_check:bool=False) -> Union[tuple[PEPS],tuple[PEPO],tuple[Braket]]:
    """
    Cuts the edge `(node1,node2)`, and returns all resulting objects.
    """
    # sanity check
    assert obj.G.has_edge(node1,node2,0)
    if sanity_check: assert obj.intact

    if isinstance(obj,PEPS):
        oldG = copy.deepcopy(obj.G)
        expose_edge(oldG,node1=node1,node2=node2,sanity_check=sanity_check)
        leg1 = oldG[node1][node2][0]["legs"][node1]
        leg2 = oldG[node1][node2][0]["legs"][node2]
        idx1 = lambda i: tuple(i if _ == leg1 else slice(0,obj.G.nodes[node1]["T"].shape[_]) for _ in range(obj.G.nodes[node1]["T"].ndim))
        idx2 = lambda i: tuple(i if _ == leg2 else slice(0,obj.G.nodes[node2]["T"].shape[_]) for _ in range(obj.G.nodes[node2]["T"].ndim))

        res_objs = ()
        for i in range(obj.G[node1][node2][0]["size"]):
            newG = copy.deepcopy(oldG)
            newG.remove_edge(node1,node2,key=0)
            newG.nodes[node1]["T"] = oldG.nodes[node1]["T"][idx1(i)]
            newG.nodes[node2]["T"] = oldG.nodes[node2]["T"][idx2(i)]
            res_objs += (PEPS(newG,sanity_check=sanity_check),)

        return res_objs

    if isinstance(obj,PEPO):
        oldG = copy.deepcopy(obj.G)
        expose_edge(oldG,node1=node1,node2=node2,sanity_check=sanity_check)
        leg1 = oldG[node1][node2][0]["legs"][node1]
        leg2 = oldG[node1][node2][0]["legs"][node2]
        idx1 = lambda i: tuple(i if _ == leg1 else slice(0,obj.G.nodes[node1]["T"].shape[_]) for _ in range(obj.G.nodes[node1]["T"].ndim))
        idx2 = lambda i: tuple(i if _ == leg2 else slice(0,obj.G.nodes[node2]["T"].shape[_]) for _ in range(obj.G.nodes[node2]["T"].ndim))

        res_objs = ()
        for i in range(obj.G[node1][node2][0]["size"]):
            newG = copy.deepcopy(oldG)
            newG.remove_edge(node1,node2,key=0)
            newG.nodes[node1]["T"] = oldG.nodes[node1]["T"][idx1(i)]
            newG.nodes[node2]["T"] = oldG.nodes[node2]["T"][idx2(i)]
            res_objs += (PEPO.from_graphs(newG,obj.tree,check_tree=False,sanity_check=sanity_check),)

        return res_objs

    if isinstance(obj,Braket):
        bra_cuts = feynman_cut(obj.bra,node1,node2,sanity_check=sanity_check)
        op_cuts = feynman_cut(obj.op,node1,node2,sanity_check=sanity_check)
        ket_cuts = feynman_cut(obj.ket,node1,node2,sanity_check=sanity_check)

        res_objs = ()
        for bra,op,ket in itertools.product(bra_cuts,op_cuts,ket_cuts):
            res_objs += (Braket(bra,op,ket,sanity_check=sanity_check),)

        return res_objs

    raise NotImplementedError("feynman_cut not implemented for object of type " + str(type(obj)) + ".")

if __name__ == "__main__":
    pass
