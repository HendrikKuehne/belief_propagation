"""
Creating sandwiches of the form
* MPS - PEPO - MPS, or
* MPS - MPS,

by combining the classes MPS and PEPO.
These classes implement the Belief Propagation
algorithm on their graphs.
"""

import numpy as np
import networkx as nx
import cotengra as ctg
import warnings
import itertools
import ray
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
                ket_legs[neighbor][sending_node], # ket leg
                nLegs + op_legs[neighbor][sending_node], # operator leg
                2 * nLegs + bra_legs[neighbor][sending_node], # bra leg
            )
        )
        out_legs.remove(ket_legs[neighbor][sending_node])
        out_legs.remove(nLegs + op_legs[neighbor][sending_node])
        out_legs.remove(2 * nLegs + bra_legs[neighbor][sending_node])

    args += (
        # ket tensor
        ket_T,tuple(range(nLegs)) + (3 * nLegs,),
        # operator tensor
        op_T,tuple(nLegs + iLeg for iLeg in range(nLegs)) + (3 * nLegs + 1,3 * nLegs),
        # bra tensor
        bra_T,tuple(2 * nLegs + iLeg for iLeg in range(nLegs)) + (3 * nLegs + 1,),
    )

    msg = np.einsum(*args,out_legs,optimize=True)

    return msg

class Braket:
    """
    Base class for sandwiches of MPS and PEPOs.
    """

    maxiter = 10000
    """Maximum number of message update iterations."""
    targeteps = 1e-10
    """Default threshold for breaking of the BP iteration."""

    def intact_check(self) -> bool:
        """
        Cheks if the braket is intact. This includes:
        * Are the networks `G`, `bra.G`, `op.G`, and `ket.G` message-ready?
        * Are `bra`, `op`, and `ket` themselves intact?
        * Do the physical dimensions match?
        """
        assert hasattr(self,"bra")
        assert hasattr(self,"op")
        assert hasattr(self,"ket")

        if not network_message_check(self.G): return False

        if not self.bra.intact_check(): return False
        if not self.op.intact_check(): return False
        if not self.ket.intact_check(): return False

        if not self.bra.D == self.op.D and self.ket.D == self.op.D:
            warnings.warn("Physical dimensions in braket do not match.")
            return False

        return True

    def __construct_initial_messages(self,real:bool,normalize:bool,sanity_check:bool,rng:np.random.Generator=np.random.default_rng()) -> None:
        """
        Initial messages for BP iteration. Saved under the key `msg`
        in `self.G`.

        Messages are three-index tensors, where the first index belongs
        to the ket, the second index belongs to the operator and the third
        one belongs to the bra.
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
                    iKet = self.ket.G[sending_node][receiving_node][0]["legs"][receiving_node]
                    iBra = self.bra.G[sending_node][receiving_node][0]["legs"][receiving_node]
                    ket_size = self.ket.G.nodes[receiving_node]["T"].shape[iKet]
                    bra_size = self.bra.G.nodes[receiving_node]["T"].shape[iBra]
                    # calculating the message
                    msg = randn((ket_size,self.op.chi,bra_size))
                else:
                    # sending node is leaf node
                    msg = ctg.einsum(
                        "ij,klj,rl->ikr",
                        self.ket.G.nodes[sending_node]["T"],
                        self.op.G.nodes[sending_node]["T"],
                        self.bra.G.nodes[sending_node]["T"]
                    )

                if normalize: msg /= np.sum(msg)
                self.G[sending_node][receiving_node][0]["msg"][receiving_node] = msg

        return

    def __contract_tensor_msg(self,sending_node:int,receiving_node:int,sanity_check:bool) -> np.ndarray:
        """
        Contracts tensor and messages at `sending_node`, and returns the message
        that `sending_node` sends to `receiving_node`.
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
                    self.ket.G[sending_node][neighbor][0]["legs"][sending_node], # ket leg
                    nLegs + self.op.G[sending_node][neighbor][0]["legs"][sending_node], # operator leg
                    2 * nLegs + self.bra.G[sending_node][neighbor][0]["legs"][sending_node], # bra leg
                )
            )
            out_legs.remove(self.ket.G[sending_node][neighbor][0]["legs"][sending_node])
            out_legs.remove(nLegs + self.op.G[sending_node][neighbor][0]["legs"][sending_node])
            out_legs.remove(2 * nLegs + self.bra.G[sending_node][neighbor][0]["legs"][sending_node])

        args += (
            # ket tensor
            self.ket.G.nodes[sending_node]["T"],
            tuple(range(nLegs)) + (3 * nLegs,),
            # operator tensor
            self.op.G.nodes[sending_node]["T"],
            tuple(nLegs + iLeg for iLeg in range(nLegs)) + (3 * nLegs + 1,3 * nLegs),
            # bra tensor
            self.bra.G.nodes[sending_node]["T"],
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

                    msg = self.__contract_tensor_msg(sending_node,receiving_node,sanity_check)

                    # saving the new message
                    newG[sending_node][receiving_node][0]["msg"][receiving_node] = msg / np.sum(msg) if normalize else msg
                    # change in message norm
                    eps += (np.linalg.norm(self.G[sending_node][receiving_node][0]["msg"][receiving_node] - newG[sending_node][receiving_node][0]["msg"][receiving_node]),)

            # put new messages in the graph
            self.G = newG

            return max(eps)

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

    def __message_passing_iteration(self,numiter:int,real:bool,normalize:bool,threshold:float,parallel:bool,sanity_check:bool) -> tuple[float]:
        """
        Performs a message passing iteration. Returns the change `eps` in maximum
        message norm for every iteration.
        """
        # sanity check
        if sanity_check: assert self.intact_check()

        # message initialization
        self.__construct_initial_messages(real=real,normalize=normalize,sanity_check=sanity_check)

        # progress bar and threshold
        threshold = threshold if threshold != None else self.targeteps
        numiter = numiter if numiter != None else self.maxiter
        iterator = tqdm.tqdm(range(numiter),desc="BP iteration")

        eps_list = ()
        for i in iterator:
            eps = self.__message_passing_step(normalize=normalize,parallel=parallel,sanity_check=sanity_check)
            iterator.set_postfix_str(f"eps = {eps:.3e}")
            eps_list += (eps,)

            if eps < threshold:
                print("    Threshold reached; breaking.")
                iterator.close()
                break

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
                                    self.ket.G[node][neighbor][0]["legs"][node], # ket leg
                            nLegs + self.op.G[node][neighbor][0]["legs"][node], # operator leg
                        2 * nLegs + self.bra.G[node][neighbor][0]["legs"][node], # bra leg
                    )
                )

            args += (
                # ket tensor
                self.ket.G.nodes[node]["T"],
                tuple(range(nLegs)) + (3 * nLegs,),
                # operator tensor
                self.op.G.nodes[node]["T"],
                tuple(nLegs + iLeg for iLeg in range(nLegs)) + (3 * nLegs + 1,3 * nLegs),
                # bra tensor
                self.bra.G.nodes[node]["T"],
                tuple(2 * nLegs + iLeg for iLeg in range(nLegs)) + (3 * nLegs + 1,),
            )

            self.G.nodes[node]["cntr"] = ctg.einsum(*args,optimize="greedy")

        return

    def BP(self,numiter:int=None,real:bool=True,normalize:bool=True,threshold:float=1e-10,parallel:bool=False,sanity_check:bool=False) -> float:
        """
        Runs the BP algorithm with `numiter` iterations on the network. Parameters:
        * `real`: Initialization of messages with real values (otherwise complex).
        * `normalize`: Normalization of messages after new message calculation. If `normalize=True`,
        this function implements the BP algorithm from Kirkley, 2021
        ([Sci. Adv. 7, eabf1211 (2021)](https://doi.org/10.1126/sciadv.abf1211)). Otherwise,
        this function becomes Belief Propagation on trees.
        * `threshold`: When to abort the BP iteration.

        Returns the network contraction value.
        """
        # sanity check
        if sanity_check: assert self.intact_check()

        if self.G.number_of_nodes() == 1:
            warnings.warn("The network is trivial.")
            return None

        # message passing iteration
        eps_list = self.__message_passing_iteration(numiter=numiter,real=real,normalize=normalize,threshold=threshold,parallel=parallel,sanity_check=sanity_check)

        # contract tensors and messages
        self.__contract_tensors_messages(sanity_check=sanity_check)

        # opposing message normalization
        self.__normalize_messages(sanity_check=sanity_check)

        if normalize:
            # the network value is the product of all node values, divided by all edge values
            cntr = 1
            for node,node_cntr in self.G.nodes(data="cntr"): cntr *= node_cntr
            for node1,node2,edge_cntr in self.G.edges(data="cntr"): cntr /= edge_cntr
            return cntr
        else:
            # each node carries the network value
            return self.G.nodes[self.op.root]["cntr"]

    def contract(self,sanity_check:bool=False) -> float:
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
        Creates a shallow copy of `G`, and adds the keys `legs`, `trace`, `indices`,
        and `msg` to the edges.

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

        self.G = self.prepare_graph(ket.G,True)
        self.ket = ket
        self.bra = bra
        self.op = op

        if sanity_check: assert self.intact_check()

        return

if __name__ == "__main__":
    pass
