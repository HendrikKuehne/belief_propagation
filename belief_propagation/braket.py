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
import scipy.stats as scistats
import ray
import warnings
import itertools
import tqdm
import copy

from belief_propagation.utils import network_message_check,crandn,graph_compatible
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
            size_dict[ctg.get_symbol(N+1)] = self._op.G[node1][node2][0]["size"]
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
            warnings.warn("Physical dimensions in braket do not match.",RuntimeWarning)
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
        assert graph_compatible(self.G,ket.G)
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
        assert graph_compatible(self.G,bra.G)
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
        assert graph_compatible(self.G,op.G)
        self._op = op

        # it is no longer guaranteed that the messages are converged
        self._converged = False

        return

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

    def __init__(self,bra:PEPS,op:PEPO,ket:PEPS,sanity_check:bool=False) -> None:
        # sanity check
        if sanity_check:
            assert graph_compatible(bra.G,ket.G)
            assert graph_compatible(bra.G,op.G)
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
        # this base class. After all, the whole point of this base class
        # was to avoid clutter in the BP code

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

    def __repr__(self) -> str:
        return f"Braket on {self.nsites} sites. Physical dimension {self.D}. Braket is " + ("intact." if self.intact else "not intact.")

class Braket(BaseBraket):
    """
    Code for belief propagation.
    """

    def construct_initial_messages(self,real:bool=False,normalize_during:bool=True,msg_init:str="normal",sanity_check:bool=False,rng:np.random.Generator=np.random.default_rng(),**kwargs) -> None:
        """
        Initial messages for BP iteration. Saved in the dictionary
        `self.msg`.

        Messages are three-index tensors, where the first index belongs
        to the bra, the second index belongs to the operator and the third
        one belongs to the ket.
        """
        # sanity check
        if sanity_check: assert self.intact

        self.msg = {node:{} for node in self.G.nodes()}

        for node1,node2 in self.G.edges():
            for sending_node,receiving_node in itertools.permutations((node1,node2)): # messages in both directions
                if len(self.G.adj[sending_node]) > 1:
                    # ket and bra leg indices, and sizes
                    bra_size = self._bra.G[sending_node][receiving_node][0]["size"]
                    op_size = self._op.G[sending_node][receiving_node][0]["size"]
                    ket_size = self._ket.G[sending_node][receiving_node][0]["size"]
                    # new message
                    self.msg[sending_node][receiving_node] = self.get_new_message(bra_size=bra_size,op_size=op_size,ket_size=ket_size,real=real,method=msg_init,rng=rng)
                else:
                    # sending node is leaf node
                    self.msg[sending_node][receiving_node] = ctg.einsum(
                        "ij,kjl,rl->ikr",
                        self._bra.G.nodes[sending_node]["T"],
                        self._op.G.nodes[sending_node]["T"],
                        self._ket.G.nodes[sending_node]["T"]
                    )

        if normalize_during: self.__normalize_messages(normalize_to="unity",sanity_check=sanity_check)

        return

    def contract_tensor_msg(self,sending_node:int,receiving_node:int,sanity_check:bool) -> np.ndarray:
        """
        Contracts tensor and messages at `sending_node`, and returns the message
        that flows from `sending_node` to `receiving_node`.
        """
        if sanity_check: assert self.intact
        if not self.G.has_edge(sending_node,receiving_node,key=0): raise ValueError(f"Edge ({sending_node},{receiving_node}) not present in graph.")

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

        for node1,node2 in self.G.edges():
            for sending_node,receiving_node in itertools.permutations((node1,node2)):
                if np.isnan(self._edge_T[sending_node][receiving_node]).any():
                    # no transformation on this edge
                    continue

                # applying the transformation
                self.msg[sending_node][receiving_node] = np.einsum(
                    "ijkabc,abc->ijk",
                    self._edge_T[sending_node][receiving_node],
                    self.msg[sending_node][receiving_node]
                )

        return

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

        old_msg = copy.deepcopy(self.msg)

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

                    ray_refs += [contract_tensor_msg.remote(**self.__neighborhood(sending_node,receiving_node,sanity_check)),]
                    msg_ids += ((sending_node,receiving_node),)

            # get new messages
            new_msg = ray.get(ray_refs)

            for msg_id,msg in zip(msg_ids,new_msg):
                sending_node,receiving_node = msg_id
                # saving the new message
                self.msg[sending_node][receiving_node] = msg

        # passing messages through the edges
        self.__pass_msg_through_edges(sanity_check=sanity_check)

        if normalize_during:
            # normalize messages to unity
            self.__normalize_messages(normalize_to="unity",sanity_check=sanity_check)

        eps = ()
        # change in message norm
        for node1,node2 in self.G.edges():
            for sending_node,receiving_node in itertools.permutations((node1,node2)):
                eps += (np.linalg.norm(self.msg[sending_node][receiving_node] - old_msg[sending_node][receiving_node]),)

        return max(eps)

    def __message_passing_iteration(self,numiter:int,real:bool,normalize_during:bool,threshold:float,parallel:bool,iterator_desc_prefix:str,verbose:bool,new_messages:bool,msg_init:str,sanity_check:bool) -> tuple[float]:
        """
        Performs a message passing iteration. Returns the change `eps` in maximum
        message norm for every iteration.
        """
        # sanity check
        if sanity_check: assert self.intact

        iterator = tqdm.tqdm(range(numiter),desc=iterator_desc_prefix + f"BP iteration",disable=not verbose)

        eps_list = ()
        # message initialization
        if new_messages: self.construct_initial_messages(real=real,normalize_during=normalize_during,msg_init=msg_init,sanity_check=sanity_check)

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
                    # vanishing messages will be blown up by normalization, which is something we do not want
                    if np.allclose(self.msg[sending_node][receiving_node],0):
                        # why not set them to zero? Because we incur a divide-by-zero during cntr calculation at the
                        # end of BP. How I handle this instead is I set node.cntr = 0 for zero-messages in
                        # self.__contract_tensors_inbound_messages
                        continue

                    self.msg[sending_node][receiving_node] /= np.sum(self.msg[sending_node][receiving_node])

            return

        if normalize_to == "cntr":
            if self.cntr == 0:
                warnings.warn("When the network value is zero, normalizing messages to the contraction value will not work. Skipping.",RuntimeWarning)
                return

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

            node_cntr = ctg.einsum(*args,optimize="greedy")
            self.G.nodes[node]["cntr"] = 0 if np.isclose(node_cntr,0) else node_cntr

        return

    def __neighborhood(self,sending_node:int,receiving_node:int,sanity_check:bool=False) -> dict:
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

    def BP(self,numiter:int=500,trials:int=10,real:bool=False,normalize_during:bool=True,normalize_after:bool=True,threshold:float=1e-10,parallel:bool=False,verbose:bool=False,new_messages:bool=True,msg_init:str="normal",sanity_check:bool=False,**kwargs) -> None:
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
        * `normalize_after`: Normalization of messages after the completed BP iteration.
        If `True`, contraction of a node with it's incoming messages yields the complete network
        value. Only relevant if `normalize_during = True`.
        * `threshold`: When to abort the BP iteration.
        * `new_messages`: Whether or not to initialize new messages.
        * `msg_init`: Method used for message initialization.

        Writes the network contraction value to `self.cntr`. Converged messages are normalized
        such that the contraction of a tensor with it's inbound messages gives the complete network
        value, at every site.

        If the algorithm converges, the flag `self.converged` is set to `True`.
        """
        # sanity check
        if sanity_check: assert self.intact

        if self.G.number_of_nodes() == 1:
            warnings.warn("The network consists of one node only. Exiting.",UserWarning)
            return

        # handling kwargs
        kwargs["iterator_desc_prefix"] = kwargs["iterator_desc_prefix"] + " | " if "iterator_desc_prefix" in kwargs.keys() else ""
        if trials == 0: warnings.warn(f"Braket.BP received trials = 0. This results in no BP iteration attempt.",UserWarning)

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
                msg_init=msg_init,
                sanity_check=sanity_check,
            )

            iTrial += 1

            if eps_list[-1] < threshold:
                self._converged = True
                self.iter_until_conv = len(eps_list)
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

    def insert_edge_T(self,T:np.ndarray,sending_node:int,receiving_node:int,sanity_check:bool=False) -> None:
        """
        Adds the transformation `T` to messages that are sent from `sending_node` to `receiving_node`.
        """
        # sanity check
        if sanity_check: assert self.intact
        if not self.G.has_edge(sending_node,receiving_node,0): raise ValueError(f"Edge ({sending_node},{receiving_node}) not present in graph.")
        vec_size = (
            self._bra.G[sending_node][receiving_node][0]["size"],
            self._op.G[sending_node][receiving_node][0]["size"],
            self._ket.G[sending_node][receiving_node][0]["size"]
        )
        if not T.shape == 2*vec_size: raise ValueError("Transformation T as wrong shape. Expected " + str(2*vec_size) + ", got " + str(T.shape) + ".")

        # inserting T
        if np.isnan(self._edge_T[sending_node][receiving_node]).any():
            # no transformation on this edge so far
            self._edge_T[sending_node][receiving_node] = T
        else:
            # transformations are executed successively
            self._edge_T[sending_node][receiving_node] = np.einsum("abcijk,ijklmn->abclmn",T,self._edge_T[sending_node][receiving_node])

        return

    def perturb_messages(self,real:bool=False,d:float=1e-3,msg_init:str="zero-normal",sanity_check:bool=False,rng:np.random.Generator=np.random.default_rng()) -> None:
        """
        Perturbs all messages in the braket. `d` is the magnitude
        of the perturbation relative to the unperturbed message.
        """
        # sanity check
        if sanity_check: assert self.intact

        if self.msg == None:
            warnings.warn("Message are not initialized. Skipping.",RuntimeWarning)

        for node1,node2 in self.G.edges():
            for sending_node,receiving_node in itertools.permutations((node1,node2)):
                norm = np.sum(self.msg[sending_node][receiving_node])
                delta_msg = self.get_new_message(*self.msg[sending_node][receiving_node].shape,real=real,method=msg_init,rng=rng)
                self.msg[sending_node][receiving_node] += delta_msg * norm * d / np.max(np.abs(delta_msg))

        # convergence cannot be guaranteed anymore
        self._converged = False

        return

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
                        warnings.warn(f"Non-finite message sent from {sending_node} to {receiving_node}.",RuntimeWarning)
                        return False

        for node1,node2 in self.G.edges():
            for sending_node,receiving_node in itertools.permutations((node1,node2)):
                if np.isnan(self._edge_T[sending_node][receiving_node]).any():
                    # no transformation on this edge
                    continue

                if not self._edge_T[sending_node][receiving_node].shape == (
                    self._bra.G[node1][node2][0]["size"],
                    self._op.G[node1][node2][0]["size"],
                    self._ket.G[node1][node2][0]["size"],
                    self._bra.G[node1][node2][0]["size"],
                    self._op.G[node1][node2][0]["size"],
                    self._ket.G[node1][node2][0]["size"],
                ):
                    warnings.warn(f"Transformation for message pass {sending_node} -> {receiving_node} ash wrong shape.",RuntimeWarning)
                    return False

        return True

    @property
    def edge_T(self) -> dict[int,dict[int,np.ndarray]]:
        """Transformations on edges. First key sending node, second key receiving node."""
        return self._edge_T

    @staticmethod
    def get_new_message(bra_size:int,op_size:int,ket_size:int,real:bool=False,method:str="normal",rng:np.random.Generator=np.random.default_rng()) -> np.ndarray:
        """
        Generates a new message with shape `[bra_size,op_size,ket_size]`.
        """
        # random number generation
        if real:
            randn = lambda size: rng.standard_normal(size)
        else:
            randn = lambda size: crandn(size,rng)

        # random matrix generation
        if real:
            matrixgen = lambda N: scistats.ortho_group.rvs(dim=N,size=1)
        else:
            matrixgen = lambda N: scistats.unitary_group.rvs(dim=N,size=1)

        if bra_size == ket_size:
            if method == "normal":
                # positive-semidefinite and hermitian
                msg = np.zeros(shape=(bra_size,op_size,ket_size),dtype=np.float128) if real else np.zeros(shape=(bra_size,op_size,ket_size),dtype=np.complex128)
                for i in range(op_size):
                    A = randn(size=(bra_size,bra_size))
                    msg[:,i,:] = A.T.conj() @ A

                return msg

            if method == "unitary":
                # positive-semidefinite and hermitian
                msg = np.zeros(shape=(bra_size,op_size,ket_size),dtype=np.float128) if real else np.zeros(shape=(bra_size,op_size,ket_size),dtype=np.complex128)
                for i in range(op_size):
                    eigvals = rng.uniform(low=0,high=1,size=bra_size)
                    U = matrixgen(bra_size)
                    msg[:,i,:] = U.conj().T @ np.diag(eigvals) @ U

                return msg

            if method == "zero-normal":
                # positive-semidefinite, hermitian, sums to zero
                msg = Braket.get_new_message(bra_size=bra_size,op_size=op_size,ket_size=ket_size,real=real,method="normal",rng=rng)
                return msg - np.sum(msg)

            raise ValueError("Message initialisation method " + method + " not implemented.")

        return randn(size=(bra_size,op_size,ket_size))

    @classmethod
    def Cntr(cls,G:nx.MultiGraph,sanity_check:bool=False):
        """
        Contraction of the tensor network contained in `G`. The TN will
        be contained in `self.ket`.
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
        """Messages. First key sending node, second key receiving node."""

        self._edge_T:dict[int,dict[int,np.ndarray]] = {
            sending_node:{
                receiving_node:np.full(shape=(1,),fill_value=np.nan)
                for receiving_node in self.G.adj[sending_node]
            }
            for sending_node in self.G.nodes
        }
        """Transformations on edges. First key sending node, second key receiving node."""

        self.cntr:float = np.nan
        """Value of the network, calculated by BP."""

        self.iter_until_conv:int = np.nan
        """Iterations necessary until convergence."""

        if sanity_check: assert self.intact

        return

    def __repr__(self):
        return super().__repr__() + " Messages are " + ("converged." if self.converged else "not converged.")

if __name__ == "__main__":
    pass
