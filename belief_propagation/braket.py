"""
Creating sandwiches of the form
* MPS @ PEPO @ MPS, or
* MPS @ MPS,

by combining the classes MPS and PEPO. The class `Braket` contained
herein implements the Belief Propagation algorithm.
"""

__all__ = [
    "Braket",
    "ExcBraket",
    "contract_tensor_inbound_messages",
    "contract_braket_physical_indices",
    "edge_transf_to_tensor_stack",
    "assemble_excitation_brakets",
    "BP_convergence_test"
]

import copy
import warnings
import itertools
from typing import Iterator, Union, Iterable, Callable

import numpy as np
import networkx as nx
import cotengra as ctg
import scipy.linalg as scialg
import scipy.stats as scistats
import ray
import tqdm

from belief_propagation.utils import (
    network_message_check,
    crandn,
    graph_compatible,
    same_legs,
    check_msg_intact
)
from belief_propagation.hamiltonians import Identity
from belief_propagation.PEPO import PEPO
from belief_propagation.PEPS import PEPS


@ray.remote(num_cpus=1)
def contract_tensor_msg(
        msg: dict,
        sender: int,
        receiver: int,
        bra_legs: dict,
        op_legs: dict,
        ket_legs: dict,
        bra_T: np.ndarray,
        op_T: np.ndarray,
        ket_T: np.ndarray,
    ) -> np.ndarray:
    """
    Contracts tensor and messages at `sender`, and returns the
    message that `sender` sends to `receiver`. Same
    functionality as `Braket.__contract_tensor_msg`.
    """
    # The outcoming message on one edge is the result of absorbing all incoming
    # messages on all other edges into the tensor sandwich
    nLegs = len(msg)
    args = ()
    """Arguments for einsum"""

    out_legs = list(range(3 * nLegs))

    for neighbor in msg.keys():
        if neighbor == receiver: continue
        args += (
            msg[neighbor],
            (
                bra_legs[neighbor][sender], # bra leg
                nLegs + op_legs[neighbor][sender], # operator leg
                2 * nLegs + ket_legs[neighbor][sender], # ket leg
            )
        )
        out_legs.remove(bra_legs[neighbor][sender])
        out_legs.remove(nLegs + op_legs[neighbor][sender])
        out_legs.remove(2 * nLegs + ket_legs[neighbor][sender])

    args += (
        # bra tensor
        bra_T, tuple(range(nLegs)) + (3 * nLegs,),
        # operator tensor
        op_T, (tuple(nLegs + iLeg for iLeg in range(nLegs))
               + (3 * nLegs,3 * nLegs + 1)),
        # ket tensor
        ket_T, (tuple(2 * nLegs + iLeg for iLeg in range(nLegs))
                + (3 * nLegs + 1,)),
    )

    msg = np.einsum(*args, out_legs, optimize=True)

    return msg


class BaseBraket:
    """
    Base class for sandwiches of MPS and PEPOs.
    Always describes a braket of the form `<bra|op|ket>`.
    """

    def __contract_ctg_hyperopt(
            self,
            inputs: list[list[str]],
            arrays: list[np.ndarray],
            size_dict: dict[str, int],
            target_width: int = 20,
            parallel: bool = False,
            verbose: bool = False,
            **kwargs
        ) -> float:
        """
        Exact contraction using a `cotengra.HyperOptimizer` object. This
        is only recommended for large networks, as it incurs a
        significant computational overhead.
        * `target_size`: Maximum number of dimensions of any
        intermediate tensor. Is passed to dynamic slicing options of
        `ctg.HyperOptimizer` as `target_size = max_edge_size **
        target_width`, where `max_edge_size` is the largest edge size
        (see [dynamic slicing](https://cotengra.readthedocs.io/en/latest/advanced.html#dynamic-slicing-slicing-reconf-opts)
        in the cotengra docs).
        Standard value was chosen basically arbitrarily.
        * `parallel`: Whether Cotengra uses parallelization.
        * `kwargs` are passed to `ctg.HyperOptimizer`.
        """

        # Handling kwargs.
        max_edge_size = max(size_dict.values())
        target_size = max_edge_size ** target_width

        opt = ctg.HyperOptimizer(
            parallel=parallel,
            slicing_reconf_opts={"target_size": target_size},
            progbar=verbose,
            **kwargs
        )
        tree = opt.search(inputs=inputs, output=(), size_dict=size_dict)

        return tree.contract(arrays)

    def contract(
            self,
            hole: int = None,
            sanity_check: bool = False,
            **kwargs
        ) -> Union[float, np.ndarray]:
        """
        Exact contraction. `kwargs` are passed to `ctg.HyperOptimizer`.

        If `hole` is given, `self` is contracted around `hole`,
        returning the exact environment at the node `hole`.
        """

        if hole is not None:
            # Exact contraction around node; an environment is returned.
            return contract_braket_with_hole(
                braket=self,
                hole=hole,
                sanity_check=sanity_check
            )

        # Gathering cotengra contraction information.
        inputs, shapes, arrays, size_dict = braket_to_ctg_arguments(
            braket=self,
            sanity_check=sanity_check
        )
        # Slicing singleton dimensions.
        inputs, shapes, arrays, size_dict = slice_singleton_dimensions(
            inputs=inputs,
            shapes=shapes,
            arrays=arrays,
            size_dict=size_dict
        )

        # I'm using a guess to distinguish between cases where I should use a
        # HyperOptimizer object, and where I shouldn't. Optimal contraction
        # path construction with HyperOptimizers incurs a significant
        # computational overhead, and is only recommended for large networks.

        if self.nsites < 100:
            eq = ctg.utils.inputs_output_to_eq(inputs=inputs, output=[])
            return ctg.einsum(eq, *arrays)

        return self.__contract_ctg_hyperopt(
            inputs=inputs,
            arrays=arrays,
            size_dict=size_dict,
            **kwargs
        )

    def _permute_virtual_dimensions(
            self,
            G: nx.MultiGraph,
            sanity_check: bool = False
        ) -> None:
        """
        Changes the leg ordering to the one given in `G`.
        """
        self._bra._permute_virtual_dimensions(G=G, sanity_check=sanity_check)
        self._op._permute_virtual_dimensions(G=G, sanity_check=sanity_check)
        self._ket._permute_virtual_dimensions(G=G, sanity_check=sanity_check)

        return

    def get_network(self, sanity_check: bool = False) -> nx.MultiGraph:
        """
        Contracts all physical dimensions in `self`, and returns the
        resulting tenwor network, contained in a graph.
        """
        if not check_contracted_physical_dims(
            braket=self, sanity_check=sanity_check
        ):
            cntr_network = contract_braket_physical_indices(
                braket=self,
                sanity_check=sanity_check
            )
        else:
            cntr_network = copy.deepcopy(self)

        # Removing dummy physical dimensions.
        for node in cntr_network:
            cntr_network.G.nodes[node]["T"] = cntr_network[node][-1][...,0]

        if sanity_check: assert network_message_check(G=cntr_network.G)
        return cntr_network.G

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
        * Do the graphs in bra, operator, ket and the internal graph
        match?
        * Are `bra`, `op`, and `ket` themselves intact?
        * Do the physical dimensions match?
        * Do all the messages contain finite values?
        """
        assert hasattr(self, "_bra")
        assert hasattr(self, "_op")
        assert hasattr(self, "_ket")

        # Is the internal graph message-ready?
        if not network_message_check(self.G): return False

        # Do the internal graphs match, in terms of geometry and physical
        # dimension?
        for G, name in zip(
            (self.bra.G, self.op.G, self.ket.G),
            ("bra", "op", "ket")
        ):
            if not graph_compatible(self.G, G):
                warnings.warn(
                    "".join((
                        "Graph in ",
                        name,
                        " does not match internal graph."
                    )),
                    RuntimeWarning
                )
                return False

        if not self.bra.intact: return False
        if not self.op.intact: return False
        if not self.ket.intact: return False

        return True

    @property
    def converged(self) -> bool:
        """Whether the messages in `self.msg` are converged."""
        return self._converged

    @property
    def D(self) -> dict[int, int]:
        """Physical dimension at every node."""
        return {
            node: self.G.nodes[node]["D"]
            for node in self
        }

    @property
    def ket(self) -> PEPS:
        """
        The ket-state in the braket `bra @ op @ ket`.
        """
        return self._ket

    @ket.setter
    def ket(self, ket: PEPS) -> None:
        # sanity check
        assert graph_compatible(self.G, ket.G, sanity_check=True)
        self._ket = ket

        # It is no longer guaranteed that the messages are converged.
        self._converged = False

        return

    @property
    def bra(self) -> PEPS:
        """
        The bra-state in the braket `bra @ op @ ket`.
        """
        return self._bra

    @bra.setter
    def bra(self, bra: PEPS) -> None:
        # sanity check
        assert graph_compatible(self.G, bra.G, sanity_check=True)
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
    def op(self, op: PEPO) -> None:
        # sanity check
        assert graph_compatible(self.G, op.G, sanity_check=True)
        self._op = op

        # it is no longer guaranteed that the messages are converged
        self._converged = False

        return

    @staticmethod
    def prepare_graph(
            G: nx.MultiGraph,
            keep_legs: bool = False,
            D: Union[int, dict[int, int]] = None
        ) -> nx.MultiGraph:
        """
        Creates a shallow copy of `G`, and adds the keys `legs`,
        `trace`, and `indices` to the edges. If given, adds the physical
        dimensions to the graph.
        """
        # shallow copy of G
        newG = nx.MultiGraph(G.edges())
        # adding legs attribute to each edge
        for node1, node2, legs in G.edges(data="legs", keys=False):
            newG[node1][node2][0]["legs"] = legs if keep_legs else {}
            newG[node1][node2][0]["trace"] = False
            newG[node1][node2][0]["indices"] = None

        if D is not None:
            if np.isscalar(D):
                # Preparing physical dimensions.
                D = {node: D for node in G}

            else:
                # Sanity check for physical dimensions.
                if not isinstance(D, dict):
                    raise ValueError("".join((
                        "Physical dimensions must be given as dictionary, ",
                        "where the local physical dimension is given for ",
                        "every node."
                    )))

                if not nx.utils.nodes_equal(
                    nodes1=G.nodes(),
                    nodes2=D.keys()
                ):
                    raise ValueError("Nodes in D don't match nodes in graph.")

            for node in G:
                newG.nodes[node]["D"] = D[node]

        return newG

    @classmethod
    def Cntr(
            cls,
            G: nx.MultiGraph,
            sanity_check: bool = False,
            **kwargs
        ):
        """
        Contraction of the tensor network contained in `G`. The TN will
        be contained in `self.ket`. `kwargs` are passed to
        `cls.__init__`. `G` needs to contain a tensor on every site, and
        the `legs` attribute on every edge.
        """
        return cls(
            bra=PEPS.Dummy(G=G, sanity_check=sanity_check),
            op=Identity(G=G, D=1, sanity_check=sanity_check),
            ket=PEPS.init_from_TN(G=G, sanity_check=sanity_check),
            sanity_check=sanity_check,
            **kwargs
        )

    @classmethod
    def Overlap(
            cls,
            psi1: PEPS,
            psi2: PEPS,
            sanity_check: bool = False,
            **kwargs
        ):
        """
        Overlap <`psi1`,`psi2`> of two PEPS. Returns the corresponding
        `Braket` object. `kwargs` are passed to `cls.__init__`.
        """
        return cls(
            bra=psi1.conj(sanity_check=sanity_check),
            op=Identity(G=psi1.G, D=psi1.D, sanity_check=sanity_check),
            ket=psi2,
            sanity_check=sanity_check,
            **kwargs
        )

    @classmethod
    def Expval(
            cls,
            psi: PEPS,
            op: PEPO,
            sanity_check: bool = False,
            **kwargs
        ):
        """
        Expectation value of the operator `op` for the state `psi`.
        `kwargs` are passed to `cls.__init__`.
        """
        return cls(
            bra=psi.conj(sanity_check=sanity_check),
            op=op,
            ket=psi,
            sanity_check=sanity_check,
            **kwargs
        )

    def __getitem__(self, node:int) -> tuple[np.ndarray]:
        """
        Subscripting with a node gives the tensor stack
        `(bra[node], op[node], ket[node])` at that node.
        """
        return (self.bra[node], self.op[node], self.ket[node])

    def __setitem__(self, node: int, Tstack: tuple[np.ndarray]) -> None:
        """
        Changing tensor stacks directly.
        """
        if not len(Tstack) == 3:
            raise ValueError("".join((
                "Tensor stacks must consist of three tensors. received "
                ,f"{len(Tstack)} tensors."
            )))
        if not (Tstack[0].shape[-1] == Tstack[1].shape[-2]
                and Tstack[2].shape[-1] == Tstack[1].shape[-1]):
            raise ValueError(
                "Physical dimension mismatch in tensor stack."
            )

        self._bra[node] = Tstack[0]
        self._op[node] = Tstack[1]
        self._ket[node] = Tstack[2]

        # it is no longer guaranteed that the messages are converged
        self._converged = False

        return

    def __repr__(self) -> str:
        return "".join((
            f"Braket on {self.nsites} sites. Braket is ",
            "intact." if self.intact else "not intact."
        ))

    def __len__(self) -> int: return self.nsites

    def __iter__(self) -> Iterator[int]:
        """
        Iterator over the nodes in the graph `self.G`.
        """
        return iter(self.G.nodes(data=False))

    def __contains__(self, node: int) -> bool:
        """Does the graph `self.G` contain the node `node`?"""
        return node in self.G

    def __eq__(self, rhs: "BaseBraket") -> bool:
        """
        Two Brakets are considered equal if they contain the same local
        tensors on the same graph. Different leg orderings are accounted
        for. Keep in mind that this notion of equality is not invariant
        with respect to the gauge freedom of the virtual bond
        dimensions!
        """
        # Do self and rhs live on the same graphs?
        if not graph_compatible(self.G, rhs.G): return False

        # Since we might need to permute the virtual dimensions, we will
        # continue with a copy of self.
        lhs = copy.deepcopy(self)

        if not same_legs(lhs.G, rhs.G):
            # Permute dimensions of lhs to make both Brakets compatible.
            lhs._permute_virtual_dimensions(rhs.G)

        # Are all site tensors the same?
        for node in lhs:
            if not all(
                lhs_T.shape == rhs_T.shape
                for lhs_T, rhs_T in zip(lhs[node], rhs[node])
            ): 
                # Two tensors in the respective tensor stacks have different
                # shapes.
                return False

            if not all(
                np.allclose(lhs_T, rhs_T)
                for lhs_T, rhs_T in zip(lhs[node], rhs[node])
            ):
                # Two tensors in the respective tensor stacks have different
                # components.
                return False

        return True

    def __init__(
            self,
            bra: PEPS,
            op: PEPO,
            ket: PEPS,
            sanity_check: bool = False
        ) -> None:
        # sanity check
        if sanity_check:
            assert graph_compatible(bra.G, ket.G, sanity_check=sanity_check)
            assert graph_compatible(bra.G, op.G, sanity_check=sanity_check)
            assert bra.D == op.D and ket.D == op.D

        self.G: nx.MultiGraph = self.__class__.prepare_graph(
            G=op.G,
            keep_legs=True,
            D=op.D,
        )
        """
        Graph that contains leg ordering, and physical dimensions.
        """

        self._bra: PEPS = bra
        self._op: PEPO = op
        self._ket: PEPS = ket

        self._converged: bool = False
        """Whether the messages in `self.msg` are converged."""
        # This attribute does not really belong in BaseBraket, of course; I
        # included it here to be able to have the property setters in this base
        # class. After all, the whole point of this base class was to avoid
        # clutter in the BP code.

        if sanity_check: assert self.intact

        return


class Braket(BaseBraket):
    """
    Code for Belief Propagation.
    """

    def construct_initial_messages(
            self,
            real: bool = False,
            normalize: bool = True,
            msg_init: str = "normal",
            sanity_check: bool = False,
            rng: np.random.Generator = np.random.default_rng(),
            **kwargs
        ) -> None:
        """
        Initial messages for BP iteration. Saved in the dictionary
        `self.msg`. Sets convergence marker to `False`.

        Messages are three-index tensors, where the first index belongs
        to the bra, the second index belongs to the operator and the
        third one belongs to the ket.
        """
        # sanity check
        if sanity_check: assert self.intact

        self.msg = {node: {} for node in self.G.nodes()}

        for node1, node2 in self.G.edges():
            for sender, receiver in itertools.permutations(
                (node1, node2)
            ):
                if len(self.G.adj[sender]) > 1:
                    # ket and bra leg indices, and sizes
                    bra_size = self.bra.G[sender][receiver][0]["size"]
                    op_size = self.op.G[sender][receiver][0]["size"]
                    ket_size = self.ket.G[sender][receiver][0]["size"]
                    # new message
                    self.msg[sender][receiver] = self.get_new_message(
                        bra_size=bra_size,
                        op_size=op_size,
                        ket_size=ket_size,
                        real=real,
                        method=msg_init,
                        rng=rng
                    )
                else:
                    # sending node is leaf node
                    self.msg[sender][receiver] = ctg.einsum(
                        "ij,kjl,rl->ikr",
                        self.bra.G.nodes[sender]["T"],
                        self.op.G.nodes[sender]["T"],
                        self.ket.G.nodes[sender]["T"]
                    )

        if normalize:
            self.normalize_messages(
                normalize_to="unity",
                sanity_check=sanity_check
            )

        # After initializing new messages, the messages are - of course - not
        # converged.
        self._converged = False

        return

    def contract_tensor_msg(
            self,
            sender: int,
            receiver: int,
            sanity_check: bool
        ) -> np.ndarray:
        """
        Contracts tensor and messages at `sender`, and returns the
        message that flows from `sender` to `receiver`.
        """
        if sanity_check: assert self.intact
        if not self.G.has_edge(sender, receiver, key=0):
            raise ValueError(
                f"Edge ({sender}, {receiver}) not present in graph."
            )

        # The outcoming message on one edge is the result of absorbing all
        # incoming messages on all other edges into the tensor stack.
        neighbors = tuple(
            node 
            for node in self.G.adj[sender]
            if node != receiver
        )

        return contract_tensor_inbound_messages(
            braket=self,
            node=sender,
            neighbors=neighbors,
            sanity_check=sanity_check
        )

    def __pass_msg_through_edges(self, sanity_check: bool) -> None:
        """
        Passes messages along edges. This involves linear
        transformations that live on the edges of the braket, such as
        projections.
        """
        # sanity check
        if sanity_check: assert self.intact

        for node1, node2 in self.G.edges():
            for sender,receiver in itertools.permutations((node1, node2)):
                if np.isnan(self._edge_T[sender][receiver]).any():
                    # no transformation on this edge
                    continue

                # applying the transformation
                self.msg[sender][receiver] = np.einsum(
                    "ijkabc,abc->ijk",
                    self._edge_T[sender][receiver],
                    self.msg[sender][receiver]
                )

        return

    def __message_passing_step(
            self,
            damping: float,
            normalize: bool,
            parallel: bool,
            sanity_check: bool
        ) -> float:
        """
        Performs a message passing step. Algorithm taken from Kirkley,
        2021 ([Sci. Adv. 7, eabf1211
        (2021)](https://doi.org/10.1126/sciadv.abf1211)). Returns the
        maximum change of message norm over the entire graph.

        Parallelization using ray.
        """
        # sanity check
        if sanity_check: assert self.intact

        if all(len(self.G.adj[node]) <= 1 for node in self.G.nodes):
            # there are only leaf nodes in the graph; we don't need to do
            # anything
            return 0

        old_msg = copy.deepcopy(self.msg)

        if not parallel: # non-parallel version of the code
            new_msg = {node: {} for node in self.G.nodes()}

            for node1, node2 in self.G.edges():
                for sender,receiver in itertools.permutations((node1, node2)):
                    # messages in both directions

                    if len(self.G.adj[sender]) == 1:
                        # leaf node; no action necessary
                        new_msg[sender][receiver] = self.msg[sender][receiver]
                        continue

                    msg = self.contract_tensor_msg(
                        sender=sender,
                        receiver=receiver,
                        sanity_check=sanity_check
                    )

                    # saving the new message
                    new_msg_ = (msg * (1 - damping)
                                + damping * old_msg[sender][receiver])
                    new_msg[sender][receiver] = new_msg_

            # put new messages in the graph
            self.msg = new_msg

        else: # parallel version
            if not ray.is_initialized(): ray.init()

            ray_refs = []
            msg_ids = ()

            for node1, node2 in self.G.edges():
                for sender, receiver in itertools.permutations((node1, node2)):
                    # messages in both directions

                    if len(self.G.adj[sender]) == 1:
                        # leaf node; no action necessary
                        continue

                    ray_refs += [contract_tensor_msg.remote(
                        **self.__neighborhood(sender, receiver, sanity_check)
                    ),]
                    msg_ids += ((sender, receiver),)

            # get new messages
            new_msg = ray.get(ray_refs)

            for msg_id, msg in zip(msg_ids, new_msg):
                sender, receiver = msg_id
                # Calculating the new message.
                new_msg_ = (msg * (1 - damping)
                            + damping * self.msg[sender][receiver])
                self.msg[sender][receiver] = new_msg_

        # passing messages through the edges
        self.__pass_msg_through_edges(sanity_check=sanity_check)

        if normalize:
            # normalize messages to unity
            self.normalize_messages(
                normalize_to="unity",
                sanity_check=sanity_check
            )

        eps = ()
        # change in message norm
        for node1, node2 in self.G.edges():
            for sender, receiver in itertools.permutations((node1, node2)):
                eps += (np.linalg.norm(
                    self.msg[sender][receiver] - old_msg[sender][receiver]
                ),)

        return max(eps)

    def __message_passing_iteration(
            self,
            numiter: int,
            damping: float,
            real: bool,
            normalize: bool,
            threshold: float,
            parallel: bool,
            iterator_desc_prefix: str,
            verbose: bool,
            new_messages: bool,
            msg_init: str,
            sanity_check: bool
        ) -> tuple[float]:
        """
        Performs a message passing iteration. Returns the change `eps`
        in maximum message norm for every iteration.
        """
        # sanity check
        if sanity_check: assert self.intact

        iterator = tqdm.tqdm(
            range(numiter),
            desc="".join((iterator_desc_prefix, "BP iteration")),
            disable=not verbose
        )

        eps_list = ()
        # Message initialization.
        if new_messages: self.construct_initial_messages(
            real=real,
            normalize=normalize,
            msg_init=msg_init,
            sanity_check=sanity_check
        )

        for i in iterator:
            eps = self.__message_passing_step(
                damping=damping,
                normalize=normalize,
                parallel=parallel,
                sanity_check=sanity_check
            )
            iterator.set_postfix_str(f"eps = {eps:.3e}")
            eps_list += (eps,)

            if eps < threshold:
                if verbose: iterator.set_postfix_str(
                    f"eps = {eps:.3e}; threshold reached; returning."
                )
                iterator.close()
                return eps_list

        return eps_list

    def normalize_messages(
            self,
            normalize_to: Union[str, float],
            sanity_check: bool
        ) -> None:
        """
        If `normalize_to = "unity"`, normalizes messages such that they
        sum to one.

        If `normalize_to = "cntr"`, normalizes messages, such that at
        each site, contraction of a tensor with it's inbound messages
        yields the complete network value. When `self.BP` is executed
        with `normalize = False`, this is already the case. This
        normalization is thus only necessary when messages were obtained
        with `normalize = True`.

        If `normalize_to = dotp`, normalizes messages such the inner
        product of opposing messages on any edge is one.

        If `normalize_to` is numeric, messages are normalized to unity
        in the respective Lp-norm. It is also possible to supply
        `np.inf`.
        """
        # sanity check
        if sanity_check: assert self.intact

        if normalize_to is np.inf:
            for sender in self.msg.keys():
                for receiver in self.msg[sender].keys():
                    # Vanishing messages will be blown up by normalization,
                    # which is something we do not want.
                    if np.allclose(self.msg[sender][receiver], 0):
                        # Why not set them to zero? Because we incur a
                        # divide-by-zero during cntr calculation at the end of
                        # BP. How I handle this instead is I set node.cntr = 0
                        # for zero-messages in
                        # self.__contract_tensors_inbound_messages
                        continue

                    msg_norm = np.max(np.abs(self.msg[sender][receiver]))
                    self.msg[sender][receiver] /= msg_norm

            return

        if isinstance(normalize_to, (int, float)):
            for sender in self.msg.keys():
                for receiver in self.msg[sender].keys():
                    # Vanishing messages will be blown up by normalization,
                    # which is something we do not want.
                    if np.allclose(self.msg[sender][receiver], 0):
                        # Why not set them to zero? Because we incur a
                        # divide-by-zero during cntr calculation at the end of
                        # BP. How I handle this instead is I set node.cntr = 0
                        # for zero-messages in
                        # self.__contract_tensors_inbound_messages
                        continue

                    msg_norm = np.sum(
                        np.abs(self.msg[sender][receiver]) ** normalize_to
                    ) ** (1 / normalize_to)
                    self.msg[sender][receiver] /= msg_norm

            return

        if normalize_to == "unity":
            for sender in self.msg.keys():
                for receiver in self.msg[sender].keys():
                    # Vanishing messages will be blown up by normalization,
                    # which is something we do not want.
                    if np.allclose(self.msg[sender][receiver], 0):
                        # Why not set them to zero? Because we incur a
                        # divide-by-zero during cntr calculation at the end of
                        # BP. How I handle this instead is I set node.cntr = 0
                        # for zero-messages in
                        # self.__contract_tensors_inbound_messages
                        continue

                    msg_sum = np.sum(self.msg[sender][receiver])

                    if np.isclose(msg_sum, 0):
                        with tqdm.tqdm.external_write_mode():
                            warnings.warn(
                                "".join((
                                    f"Message from {sender} to {receiver} ",
                                    "sums to zero. Skipping normalization."
                                )),
                                RuntimeWarning
                            )
                        continue

                    self.msg[sender][receiver] /= msg_sum

            return

        if normalize_to == "cntr":
            if np.isnan(self.cntr):
                raise RuntimeError("Trying to normalize to nan network value.")

            if self.cntr == 0:
                with tqdm.tqdm.external_write_mode():
                    warnings.warn(
                        "".join((
                            "When the network value is zero, normalizing ",
                            "messages to the contraction value will not ",
                            "work. Skipping."
                        )),
                        RuntimeWarning
                    )
                return

            if not "cntr" in self.G.nodes[self._op.root].keys():
                # Contract messages into tensors to obtain tensor values.
                self.__contract_tensors_inbound_messages(
                    sanity_check=sanity_check
                )

            for node in self:
                norm = self.cntr / self.G.nodes[node]["cntr"]
                sign_factors = [1 for _ in self.G.adj[node]]

                if np.isclose(np.real(norm), norm):
                    # The contraction value of the network is stripped of it's
                    # sign, because negative-definite networks have a negative
                    # contraction value. This also leads to some nodes having
                    # negative node contraction values. If we do not account
                    # for this, messages will not be hermitian.
                    norm = np.real(norm)
                    norm_sign = np.sign(norm)
                    norm *= norm_sign
                    sign_factors[0] = norm_sign

                else:
                    # Complex numbers do not have a sign; skipping.
                    with tqdm.tqdm.external_write_mode():
                        warnings.warn(
                            "".join((
                                "Complex normalization factor in node ",
                                f"{node}; messages might not be hermitian ",
                                "after normaliation."
                            )),
                            RuntimeWarning
                        )
                    pass

                msg_norm = norm ** (1 / len(self.G.adj[node]))

                for neighbor, sign_factor in zip(
                    self.G.adj[node],
                    sign_factors
                ):
                    self.msg[neighbor][node] *= (sign_factor * msg_norm)

            return
    
        if normalize_to == "dotp":
            # sanity check
            if not all(
                "cntr" in self.G[node1][node2][0].keys()
                for node1, node2 in self.G.edges()
            ):
                self.__contract_edge_opposite_messages(
                    sanity_check=sanity_check
                )

            for node1, node2, cntr in self.G.edges(data="cntr"):
                norm = np.sqrt(cntr)
                if np.isclose(np.real(norm), 0):
                    # Negative message contraction value; this needs to be
                    # taken into account to prevent imaginary node contraction
                    # values.
                    norm1 = 1j * norm
                    norm2 = (-1j) * norm
                else:
                    norm1 = norm
                    norm2 = norm
                assert np.isclose(norm1 * norm2, cntr)

                self.msg[node1][node2] /= norm1
                self.msg[node2][node1] /= norm2

            return

        raise NotImplementedError("".join((
            "Message normalization ", str(normalize_to), " not implemented."
        )))

    def __contract_edge_opposite_messages(
            self,
            sanity_check: bool = False
        ) -> None:
        """
        Contracts the messages travelling in each direction of an edge,
        on every edge. Value is saved under the key `cntr`.
        """
        # sanity check
        if sanity_check: assert self.intact

        for node1, node2 in self.G.edges():
            # Contracting opposing messages.
            self.G[node1][node2][0]["cntr"] = ctg.einsum(
                "ijk,ijk->",
                self.msg[node1][node2],
                self.msg[node2][node1]
            )

        return

    def __strip_node_cntr_phases(
            self,
            raise_complex_warning: bool = True,
            sanity_check: bool = False
        ) -> None:
        """
        Strips node contraction values of their phases.
        """
        # sanity check
        if sanity_check: assert self.intact
        if not all(
            "cntr" in data.keys()
            for _, data in self.G.nodes(data=True)
        ):
            raise RuntimeError(
                "There are nodes without associated node contraction value."
            )

        phases = ()
        # Stripping all node contraction values from their complex phases.
        for node, cntr in self.G.nodes(data="cntr"):
            phases += (np.angle(cntr),)
            self.G.nodes[node]["cntr"] = np.abs(cntr)

        # Gathering phases, and putting them into the operator root.
        phase_factor = np.exp(1j * np.sum(phases))
        self.G.nodes[self.op.root]["cntr"] *= phase_factor

        if (not np.isclose(phase_factor, 1)) and raise_complex_warning:
            with tqdm.tqdm.external_write_mode():
                warnings.warn(
                    "".join((
                        "Braket node contraction values have overall ",
                        "complex phase."
                    )),
                    RuntimeWarning
                )

        return

    def __contract_tensors_inbound_messages(
            self,
            sanity_check: bool = False
        ) -> None:
        """
        Contracts all messages into the respective nodes, and saves the
        value in each node.
        """
        # sanity check
        if sanity_check: assert self.intact

        for node in self:
            node_cntr = contract_tensor_inbound_messages(
                braket=self,
                node=node,
                sanity_check=sanity_check
            )

            if np.isclose(node_cntr, 0):
                self.G.nodes[node]["cntr"] = 0
            else:
                self.G.nodes[node]["cntr"] = node_cntr

        return

    def __neighborhood(
            self,
            sender: int,
            receiver: int,
            sanity_check: bool = False
        ) -> dict:
        """
        Returns the signature of `contract_tensor_msg` as a dictionary.
        """
        # sanity check
        if sanity_check:
            assert self.intact
            assert self.G.has_node(sender) and self.G.has_node(receiver)

        return dict(
            msg = {
                neighbor: self.msg[neighbor][sender]
                for neighbor in self.G.adj[sender]
            },
            sender=sender,
            receiver=receiver,
            bra_legs=self._bra.legs_dict(sender),
            op_legs=self._op.legs_dict(sender),
            ket_legs=self._ket.legs_dict(sender),
            bra_T=self._bra.G.nodes[sender]["T"],
            op_T=self._op.G.nodes[sender]["T"],
            ket_T=self._ket.G.nodes[sender]["T"]
        )

    def write_cntr_value(self, sanity_check: bool = False) -> None:
        """
        Normalizes messages s.t. the dot product of opposing messages on
        any edge is one, calculates the contraction value, and writes to
        `self.cntr`.
        """
        # sanity check
        if sanity_check: assert self.intact
        if self.msg is None: raise RuntimeError("No messages available.")

        # contract tensors and messages, opposite messages, normalize messages.
        self.normalize_messages(normalize_to="dotp", sanity_check=sanity_check)
        self.__contract_tensors_inbound_messages(sanity_check=sanity_check)

        # With the current message normalization, the network value is the
        # product of all node contraction values.
        self.cntr = 1
        for node, node_cntr in self.G.nodes(data="cntr"):
            self.cntr *= node_cntr

        return

    def BP(
            self,
            numiter: int = 1000,
            trials: int = 1,
            damping: float = 0,
            real: bool = False,
            normalize: bool = True,
            threshold: float = 1e-10,
            parallel: bool = False,
            new_messages: bool = True,
            msg_init: str = "normal",
            verbose: bool = False,
            sanity_check: bool = False,
            **kwargs
        ) -> tuple[float]:
        """
        Layz BP algorithm from [Sci. Adv. 10, eadk4321
        (2024)](https://doi.org/10.1126/sciadv.adk4321). Parameters:
        * `numiter`: Number of iterations.
        * `trials`: Number of times the BP iteration is attempted. The
        value `np.inf` can be supplied, in which case the algorithm
        terminates only when a trial reaches `threshold`.
        * `damping`: Damping factor for messages. Messages are updated
        according to `m' = (1-damping)*m' + damping*m`, where `m'` is
        the new message.
        * `real`: Initialization of messages with real values (otherwise
        complex).
        * `normalize`: Normalization of messages after each
        message passing iteration. If `normalize=True`, this function
        implements the BP algorithm from [Sci. Adv. 7, eabf1211
        (2021)](https://doi.org/10.1126/sciadv.abf1211). Otherwise, the
        algorithm becomes Belief Propagation on trees.
        * `threshold`: When to abort the BP iteration.
        * `new_messages`: Whether or not to initialize new messages.
        * `msg_init`: Method used for message initialization.

        Writes the network contraction value to `self.cntr`. Messages
        are normalized such that the dot product of messages on any
        respective edge is one.

        Returns change in message norm for the last trial.

        If the algorithm converges, the flag `self.converged` is set to
        `True`.
        """
        # sanity check
        if sanity_check: assert self.intact

        if not (damping >= 0 and damping <= 1):
            raise ValueError("Damping factor must be between 0 and 1.")

        if not normalize:
            with tqdm.tqdm.external_write_mode():
                warnings.warn(
                    "".join((
                        "BP without normalization might soon break, or be ",
                        "broken alraedy, and should not be used. On trees, ",
                        "BP is exact with normalization, too."
                    )),
                    FutureWarning
                )

        if self.G.number_of_nodes() == 1:
            with tqdm.tqdm.external_write_mode():
                warnings.warn(
                    "The network consists of one node only. Exciting.",
                    UserWarning
                )
            return

        if (not nx.is_tree(self.G)) and (not normalize):
            with tqdm.tqdm.external_write_mode():
                warnings.warn(
                    "".join((
                        "Normalization during BP disabled on a graph with ",
                        "loops. This likely leads to diverging messages.",
                        UserWarning
                    ))
                )

        # Handling kwargs.
        if "iterator_desc_prefix" in kwargs.keys():
            kwargs["iterator_desc_prefix"] = "".join((
                kwargs["iterator_desc_prefix"],
                " | "
            ))
        else:
            kwargs["iterator_desc_prefix"] = ""

        if trials == 0:
            with tqdm.tqdm.external_write_mode():
                warnings.warn(
                    "".join((
                        "Braket.BP received trials = 0. This results in no ",
                        "BP iteration attempt."
                    )),
                    UserWarning
                )

        eps_list = ()
        iTrial = 0
        while iTrial < trials:
            # Message passing iteration.
            eps_list = self.__message_passing_iteration(
                numiter=numiter,
                damping=damping,
                real=real,
                normalize=normalize,
                threshold=threshold,
                parallel=parallel,
                iterator_desc_prefix="".join((
                    kwargs["iterator_desc_prefix"],
                    f"trial {iTrial} | "
                )),
                verbose=verbose,
                new_messages=new_messages,
                msg_init=msg_init,
                sanity_check=sanity_check,
            )

            iTrial += 1

            if eps_list[-1] < threshold:
                break

        if eps_list[-1] < threshold:
            self._converged = True
            self.iter_until_conv = len(eps_list)
        else:
            self.iter_until_conv = np.nan

        # Contract tensors and messages, opposite messages, normalize messages,
        # calculate total contraction value.
        self.write_cntr_value(sanity_check=sanity_check)

        return eps_list

    def insert_edge_T(
            self,
            T: np.ndarray,
            sender: int,
            receiver: int,
            sanity_check: bool = False
        ) -> None:
        """
        Adds the transformation `T` to messages that are sent from
        `sender` to `receiver`.
        """
        # sanity check
        if sanity_check: assert self.intact
        if not self.G.has_edge(sender, receiver, 0):
            raise ValueError(
                f"Edge ({sender},{receiver}) not present in graph."
            )

        vec_size = (
            self.bra.G[sender][receiver][0]["size"],
            self.op.G[sender][receiver][0]["size"],
            self.ket.G[sender][receiver][0]["size"]
        )
        if not T.shape == 2*vec_size:
            raise ValueError("".join((
                "Transformation T as wrong shape. Expected ",
                str(2*vec_size),
                ", got ",
                str(T.shape),
                "."
            )))

        # Inserting T.
        if np.isnan(self.edge_T[sender][receiver]).any():
            # No transformation on this edge so far.
            self._edge_T[sender][receiver] = T
        else:
            # Transformations are executed successively.
            self._edge_T[sender][receiver] = np.einsum(
                "abcijk,ijklmn->abclmn",
                T,
                self.edge_T[sender][receiver]
            )

        return

    def perturb_messages(
            self,
            real: bool = False,
            d: float = 1e-3,
            method: str = "zero-normal",
            sanity_check: bool = False,
            rng: np.random.Generator = np.random.default_rng()
        ) -> None:
        """
        Perturbs all messages in the braket. `d` is the magnitude of the
        perturbation relative to the unperturbed message.
        """
        # sanity check
        if sanity_check: assert self.intact

        if self.msg == None:
            with tqdm.tqdm.external_write_mode():
                warnings.warn(
                    "Message are not initialized. Skipping.",
                    RuntimeWarning
                )

        for node1, node2 in self.G.edges():
            for sender, receiver in itertools.permutations((node1, node2)):
                norm = np.sum(self.msg[sender][receiver])
                delta_msg = self.get_new_message(
                    *self.msg[sender][receiver].shape,
                    real=real,
                    method=method,
                    rng=rng
                )
                # adjusting the strength of the perturbation
                delta_msg *= (norm * d / np.max(np.abs(delta_msg)))

                self.msg[sender][receiver] += delta_msg

        # convergence cannot be guaranteed anymore
        self._converged = False

        return

    def msg_stack(
            self,
            new_messages: bool = True,
            sanity_check: bool = False,
            **kwargs
        ) -> np.ndarray:
        """
        Stacks the current messages and returns them as a vector that
        could be passed to the closed-form of the BP iteration step.
        Intended to be used together with
        `self.get_closed_form_msg_update()`. `kwargs` are passed to
        `self.construct_initial_messages`.
        """
        if sanity_check: assert self.intact
        if new_messages: self.construct_initial_messages(
            **kwargs,
            sanity_check=sanity_check
        )

        msg_in_order = ()
        for sender in sorted(self.msg.keys()):
            for receiver in sorted(self.msg[sender].keys()):
                msg_in_order += (self.msg[sender][receiver].flatten(),)

        return np.concatenate(msg_in_order, axis=0)

    def get_closed_form_msg_update(
            self,
            sanity_check: bool = False,
            **kwargs
        ) -> Callable[[np.ndarray, bool], np.ndarray]:
        """
        Returns the BP update of a stack of messages in closed form.
        This function takes as it's input a vector that is considered to
        be the direct sum of all messages, applies the BP update, and
        returns the new message stack. `**kwargs` are passed to
        `self.BP`. The messages in the message stack are assumed to be
        ordered by ascending node value in sender first, receiver
        second. The second (boolean) argument of the returned funcion is
        the sanity check.
        """
        if sanity_check: assert self.intact

        flat_braket = contract_braket_physical_indices(
            self, sanity_check=sanity_check
        )

        kwargs["numiter"] = 1
        kwargs["new_messages"] = False

        def msg_update(
                msg_stack: np.ndarray,
                sanity_check: bool = False
            ) -> np.ndarray:
            i_saved = 0
            msg = {}
            # Extracting messages from tensor stack.
            for sender in sorted(flat_braket):
                msg[sender] = {}
                for receiver in sorted(flat_braket.G.adj[sender]):
                    size = flat_braket.ket.G[sender][receiver][0]["size"]
                    msg[sender][receiver] = np.expand_dims(
                        msg_stack[i_saved:i_saved + size],
                        axis=(0, 1)
                    )
                    i_saved += size
            if not i_saved == len(msg_stack):
                raise RuntimeError("".join((
                    "Number of elements in tensor stack does not match the ",
                    f"braket. Expected length {i_saved}, got {len(msg_stack)}."
                )))

            # Executing a single BP step.
            flat_braket.msg = msg
            flat_braket.BP(
                **kwargs,
                sanity_check=sanity_check
            )

            # Assembling new tensor stack.
            msg_in_order = ()
            for sender in sorted(flat_braket):
                for receiver in sorted(flat_braket.G.adj[sender]):
                    msg_in_order += (flat_braket.msg[sender][receiver][0,0,:],)

            return np.concatenate(msg_in_order, axis=0)

        return msg_update

    @property
    def intact(self) -> bool:
        """
        Whether the braket is intact or not. This includes:
        * Is the underlying `BaseBraket` object intact?
        * Do all the messages contain finite values?
        * Do all edge transformations have correct domains
        and images?
        """
        if not super().intact: return False

        if (self.msg is not None) and self.converged:
            # Checking if the messages are intact is only necessary if this
            # braket contains a converged set of messages.

            for sender in self.msg.keys():
                for receiver in self.msg[sender].keys():
                    target_shape = (
                        self.bra.G[sender][receiver][0]["size"],
                        self.op.G[sender][receiver][0]["size"],
                        self.ket.G[sender][receiver][0]["size"]
                    )

                    if not check_msg_intact(
                        msg=self.msg[sender][receiver],
                        target_shape=target_shape,
                        sender=sender,
                        receiver=receiver
                    ):
                        return False

        for node1, node2 in self.G.edges():
            for sender, receiver in itertools.permutations((node1, node2)):
                if np.isnan(self.edge_T[sender][receiver]).any():
                    # No transformation on this edge.
                    continue

                if not self.edge_T[sender][receiver].shape == (
                    self.bra.G[node1][node2][0]["size"],
                    self.op.G[node1][node2][0]["size"],
                    self.ket.G[node1][node2][0]["size"],
                    self.bra.G[node1][node2][0]["size"],
                    self.op.G[node1][node2][0]["size"],
                    self.ket.G[node1][node2][0]["size"],
                ):
                    with tqdm.tqdm.external_write_mode():
                        warnings.warn(
                            "".join((
                                f"Transformation for message pass {sender} ",
                                f"-> {receiver} has wrong shape."
                            )),
                            RuntimeWarning
                        )
                    return False

        return True

    @property
    def edge_T(self) -> dict[int, dict[int, np.ndarray]]:
        """
        Transformations on edges. First key sending node, second key
        receiving node.
        """
        return self._edge_T

    @staticmethod
    def get_new_message(
            bra_size: int,
            op_size: int,
            ket_size: int,
            real: bool = False,
            method: str = "normal",
            rng: np.random.Generator = np.random.default_rng()
        ) -> np.ndarray:
        """
        Generates a new message with shape
        `(bra_size, op_size, ket_size)`. Available methods if
        `bra_size == ket_size`:
        * `normal`: Positive-semidefinite and hermitian.
        * `unitary`: Positive-semidefinite and unitary.
        * `unitary_neg`: Negative-semidefinite and unitary.
        * `zero-normal`: Positive-semidefinite, hermitian, sums to zero.
        * `randn`: Random message from normal distribution.

        If bra- and ket-sizes are different, completely random messages
        are returned.
        """
        # Random number generation.
        if real:
            randn = lambda size: rng.standard_normal(size)
        else:
            randn = lambda size: crandn(size, rng)

        # Random matrix generation.
        if real:
            matrixgen = lambda N: scistats.ortho_group.rvs(dim=N, size=1)
        else:
            matrixgen = lambda N: scistats.unitary_group.rvs(dim=N, size=1)

        dtype = np.float128 if real else np.complex128

        if bra_size == ket_size:
            msg = np.zeros(shape=(bra_size, op_size, ket_size), dtype=dtype)

            if method == "normal":
                # Positive-semidefinite and hermitian.
                for i in range(op_size):
                    A = randn(size=(bra_size, bra_size))
                    msg[:,i,:] = A.T.conj() @ A

                return msg

            if method == "unitary":
                # Positive-semidefinite and unitary.
                for i in range(op_size):
                    eigvals = rng.uniform(low=0, high=1, size=bra_size)
                    U = matrixgen(bra_size)
                    msg[:,i,:] = U.conj().T @ np.diag(eigvals) @ U

                return msg

            if method == "unitary_neg":
                # Negative-semidefinite and unitary.
                for i in range(op_size):
                    eigvals = rng.uniform(low=-1, high=0, size=bra_size)
                    U = matrixgen(bra_size)
                    msg[:,i,:] = U.conj().T @ np.diag(eigvals) @ U

                return msg

            if method == "zero-normal":
                # Positive-semidefinite, hermitian, sums to zero.
                msg = Braket.get_new_message(
                    bra_size=bra_size,
                    op_size=op_size,
                    ket_size=ket_size,
                    real=real,
                    method="normal",
                    rng=rng
                )
                return msg - np.sum(msg)

            if method == "randn":
                # Random message.
                return randn(size=(bra_size, op_size, ket_size))

            raise ValueError(
                "Message initialisation method " + method + " not implemented."
            )

        return randn(size=(bra_size, op_size, ket_size))

    def __repr__(self):
        return "".join((
            super().__repr__(),
            " Messages are ",
            "converged." if self.converged else "not converged."
        ))

    def __init__(
            self,
            bra: PEPS,
            op: PEPO,
            ket: PEPS,
            msg: dict[int, dict[int, np.ndarray]] = None,
            edge_T: dict[int, dict[int, np.ndarray]] = None,
            converged: bool = False,
            sanity_check: bool = False
        ) -> None:
        """
        Initialize a new braket.
        """
        super().__init__(bra=bra, op=op, ket=ket, sanity_check=False)

        self.msg: dict[int, dict[int, np.ndarray]] = msg
        """
        Messages. First key sending node, second key receiving node.
        """

        if edge_T is None:
            self._edge_T: dict[int, dict[int, np.ndarray]] = {
                sender: {
                    receiver: np.nan
                    for receiver in self.G.adj[sender]
                }
                for sender in self.G.nodes
            }
        else:
            self._edge_T = edge_T

        self.cntr: float = np.nan
        """Value of the network, calculated by BP."""

        self.iter_until_conv: int = np.nan
        """Iterations `self.BP` took unttil convergence."""

        if msg is not None:
            # If a set of messages is supplied, the user might supply the
            # information whether it is converged or not.
            self._converged = converged

        if sanity_check: assert self.intact

        return


class ExcBraket(Braket):
    """
    Braket-object that contains an excitation. This object is
    distinguished from the `Braket` class by the fact that is carries an
    additional graph `ExcBraket.exc`, which is the excitation that the
    respective object contains.
    """

    def __init__(
            self,
            bra: PEPS,
            op: PEPO,
            ket: PEPS,
            exc: nx.MultiGraph,
            msg: dict[int, dict[int, np.ndarray]] = None,
            edge_T: dict[int, dict[int, np.ndarray]] = None,
            converged: bool = False,
            sanity_check: bool = False
        ) -> None:
        """
        Initialize a new excitation braket.
        """
        super().__init__(
            bra=bra,
            op=op,
            ket=ket,
            msg=msg,
            edge_T=edge_T,
            converged=converged,
            sanity_check=sanity_check
        )

        self.exc = exc
        """The excitation that this object contains."""


def check_contracted_physical_dims(
        braket: Braket,
        sanity_check: bool = False
    ) -> bool:
    """
    Checks if `braket` contains dummy networks in `braket.op` and
    `braket.bra`.
    """
    if sanity_check: assert braket.intact

    for node in braket:
        if not (np.allclose(braket.bra[node], np.ones(shape=1))
                and np.allclose(braket.op[node], np.ones(shape=1))):
            return False

    return True


def __message_passthrough_factor_N(
        G: nx.MultiGraph,
        node: int,
        child: int,
        parent: int,
        sanity_check: bool = False,
    ) -> float:
    """
    Calculates the quantity `N(node, child, parent)` from
    [IEEE Trans.Inf.Theory 53, 12, 2007](https://doi.org/10.1109/TIT.2007.909166).
    """
    if sanity_check: assert network_message_check(G=G)

    if not (node in G and child in G and parent in G):
        raise ValueError("node, child and parent must be contained in G.")

    # Extracting the tensor at node, and re-shaping it s.t. the leg ordering is
    # (leg_to_child, leg_to_parent, remaining_legs).
    leg_child = G[node][child][0]["legs"][node]
    leg_parent = G[node][parent][0]["legs"][node]
    T = G.nodes[node]["T"]
    leg_child_size = T.shape[leg_child]
    leg_parent_size = T.shape[leg_parent]
    axes = tuple(i for i in range(T.ndim) if i not in (leg_child, leg_parent))
    T = np.transpose(T, axes=(leg_child, leg_parent) + axes)
    T = np.reshape(T, newshape=(leg_child_size, leg_parent_size, -1))

    # Innermost term in N from the paper.
    if T.ndim == 3:
        quot_prod = np.einsum(
            "abc,def,dbc,aef->abcdef", T, T, 1/T, 1/T
        )
    elif T.ndim == 2:
        quot_prod = np.einsum(
            "ab,de,db,ae->abde", T, T, 1/T, 1/T
        )
    else:
        raise RuntimeError("".join((
            "T has the wrong number of dimensions; expected two or three, ",
            f"got {T.ndim}."
        )))

    quot_prod = np.tanh(np.log(quot_prod) / 4)
    # Setting slices to zero that we do not include in supremum calculation.
    for alpha in range(T.shape[0]):
        zero_slice = (
            alpha, slice(quot_prod.shape[1]), slice(quot_prod.shape[2]),
            alpha, slice(quot_prod.shape[4]), slice(quot_prod.shape[5])
        ) if T.ndim == 3 else (
            alpha, slice(quot_prod.shape[1]),
            alpha, slice(quot_prod.shape[3])
        )
        quot_prod[zero_slice] = 0
    for beta in range(T.shape[0]):
        zero_slice = (
            slice(quot_prod.shape[0]), beta, slice(quot_prod.shape[2]),
            slice(quot_prod.shape[3]), beta, slice(quot_prod.shape[5])
        ) if T.ndim == 3 else (
            slice(quot_prod.shape[0]), beta,
            slice(quot_prod.shape[2]), beta
        )
        quot_prod[zero_slice] = 0

    return np.max(quot_prod)


def BP_convergence_test(
        network: Union[nx.MultiGraph, BaseBraket, Braket, ExcBraket],
        sanity_check: bool = False
    ) -> bool:
    """
    Test whether BP converges to a unique fixed point on the given
    network, irrespective of given messages. Taken from
    [IEEE Trans.Inf.Theory 53, 12, 2007](https://doi.org/10.1109/TIT.2007.909166).
    """
    raise NotImplementedError("Does not yet work!")

    # Contracting physical dimensions if necessary.
    if (isinstance(network, BaseBraket)
          or isinstance(network, Braket)
          or isinstance(network, ExcBraket)):
        network = network.get_network(sanity_check=sanity_check)

    if isinstance(network, nx.MultiGraph):
        if sanity_check: assert network_message_check(G=network)
    else:
        raise ValueError(
            "".join(("Unknown network type: ", str(type(network))))
        )

    msg_influence_factors = ()
    for parent, node in network.edges():
        msg_influence_sum = 0

        if len(network.adj[node]) == 1:
            # Messages from leaf nodes are constant; we may skip this one.
            continue

        for child in set(network.adj[node]) - set((parent,)):
            msg_influence_sum += __message_passthrough_factor_N(
                G=network,
                node=node,
                child=child,
                parent=parent,
                sanity_check=sanity_check
            )

        msg_influence_factors += (msg_influence_sum,)

    return max(msg_influence_factors) < 1


def braket_to_ctg_arguments(
        braket: Union[BaseBraket, Braket, ExcBraket],
        exclude: tuple[int] = (),
        exclude_policy: str = "skip",
        sanity_check: bool = False
    ) -> tuple[
        list[list[str]],
        list[list[int]],
        list[np.ndarray],
        dict[str, int]
    ]:
    """
    Given a `Braket` object, assembles and returns the cotengra
    arguments `inputs`, `shapes`, `arrays` and `size_dict`. Arguments
    are ordered by node ascending label, with bra, op and ket being
    listed consecutively.

    Optionally, nodes can be excluded from the cotengra arguments,
    enabling partial contraction of `braket`. Nodes in `exclude` are
    excluded. If `exclude_policy = "skip"` (default), ecluded nodes are
    simply ignored. If `exclude_policy = "fill"`, the inputs of excluded
    nodes are returned and shapes and arrays are filled with
    placeholders.

    Refer to the [cotengra documentation](https://cotengra.readthedocs.io/en/latest/basics.html)
    for an introduction to how the return values represent a
    contraction.
    """
    # sanity check
    if sanity_check: assert braket.intact

    if exclude_policy not in ("skip", "fill"):
        raise ValueError(
            "".join(("Unknown excluded policy ", exclude_policy, ".")),
        )

    if braket.nsites == 1:
        node = braket.op.root
        # The network is trivial.
        inputs = [["a",], ["a", "b"], ["b",]]
        shapes = [
            [braket._bra.D[node],],
            [braket._op.D[node], braket._op.D[node]],
            [braket._ket.D[node],]
        ]
        arrays = list((braket[node],))
        size_dict = {
            "a": braket._bra.D[node],
            "b": braket._ket.D[node]
        }

        return inputs, shapes, arrays, size_dict

    inputs = []
    shapes = []
    arrays = []
    size_dict = {}

    N = 0

    for node1, node2 in braket.G.edges():
        for i, layer in enumerate((braket.bra, braket.op, braket.ket)):
            # Enumerating the virtual edges in the network.
            edge_size = layer.G[node1][node2][0]["size"]
            layer.G[node1][node2][0]["label"] = ctg.get_symbol(N + i)

            # Extracting the size of every edge.
            size_dict[ctg.get_symbol(N + i)] = edge_size
        N += 3

    for node in braket:
        # Enumerating the physical edges in the network.
        braket.bra.G.nodes[node]["label"] = [ctg.get_symbol(N),]
        braket.op.G.nodes[node]["label"] = [
            ctg.get_symbol(N), ctg.get_symbol(N+1)
        ]
        braket.ket.G.nodes[node]["label"] = [ctg.get_symbol(N+1),]

        # Extracting the size of every edge.
        size_dict[ctg.get_symbol(N)] = braket.D[node]
        size_dict[ctg.get_symbol(N + 1)] = braket.D[node]
        N += 2

    # Extracting the einsum arguments.
    for node in sorted(braket):
        if node in exclude and exclude_policy == "skip": continue

        for layer in (braket.bra.G, braket.op.G, braket.ket.G):
            # Assembling the inputs argument that contains the leg ordering.
            inputs_ = [None for _ in range(len(layer.adj[node]))]
            for _, neighbor, edge_label in layer.edges(
                nbunch=node,
                data="label"
            ):
                leg = layer[node][neighbor][0]["legs"][node]
                inputs_[leg] = edge_label

            # Physical edges
            inputs_ += layer.nodes[node]["label"]

            if node in exclude and exclude_policy == "fill":
                # Tensor and it's shape at this site are substituted for
                # placeholders.
                arrays += [np.full(shape=1, fill_value=np.nan),]
                shapes += [np.full(shape=1, fill_value=np.nan),]
                # Leg ordering at this site.
                inputs += [inputs_,]
            else:
                # Tensor at this site.
                arrays += [layer.nodes[node]["T"],]
                # Shape of the tensor at this site.
                shapes += [layer.nodes[node]["T"].shape,]
                # Leg ordering at this site.
                inputs += [inputs_,]

    return inputs, shapes, arrays, size_dict


def slice_singleton_dimensions(
        inputs: list[list[str]],
        shapes: list[list[int]],
        arrays: list[np.ndarray],
        size_dict: dict[int, int]
    ) -> tuple[
        list[list[str]],
        list[list[int]],
        list[np.ndarray],
        dict[str, int]
    ]:
    """
    Slices singleton in the given cotengra contraction data. Optionally,
    parts of the input can be excluded from slicing via the argument
    'exclude'.
    """
    # If we use cotengra to contract a braket that was created using
    # Braket.Cntr - i.e. one that contains dummies in self.bra and self.ket -
    # the code so far might fail. The reason is that the size objectives of
    # cotengra only account for the numbers of elements of intermediate
    # tensors, not the number of dimensions. When there are dummies in the
    # braket, however, there are many dimensions of size one in the tensors.
    # Cotengra does not care about those, and leaves them mostly untouched, so
    # they accumulate. This causes problems not in cotengra but in the
    # underlying numpy code, because the maximum number of dimensions of a
    # numpy array is 32. Large brakets with dummies in bra and operator might
    # reach this threshold. What I'll do to prevent this is I'll slice all
    # edges in the tree that have size one. Slicing an edge with size one is
    # free, after all (e.g. incurs no additional computational cost).

    # Counting the occurences of each leg to ensure that we do not slice
    # dangling indices.
    concatenated_labels = sum(inputs, start=[])
    label_counts = {
        label: concatenated_labels.count(label)
        for label in set(concatenated_labels)
    }

    for i_tensor, T, input in zip(range(len(arrays)), arrays, inputs):
        if np.isnan(T).any():
            # a nan-valued tensor might indicate that this node is excluded
            # from the contraction; skipping.
            continue

        # Getting the singleton dimensions, making sure that no dangling legs
        # are sliced.
        singleton_legs = tuple(
            j for j in range(T.ndim)
            if T.shape[j] == 1 and label_counts[input[j]] == 2
        )

        if len(singleton_legs) == 0:
            # This tensor has no singleton dimension; continuing.
            continue

        # Removing the singleton dimensions from the inputs and the size
        # dictionary.
        singleton_leg_labels = tuple(input[j] for j in singleton_legs)
        inputs_ = copy.deepcopy(input)
        for label in singleton_leg_labels:
            inputs_.remove(label)
            size_dict.pop(label, None)
        inputs[i_tensor] = inputs_

        # Removing the singleton dimensions from the array.
        idx = tuple(
            0 if j in singleton_legs else slice(size)
            for j, size in enumerate(T.shape)
        )
        arrays[i_tensor] = arrays[i_tensor][idx]

        # Removing the singleton dimensions from the shapes.
        shapes[i_tensor] = arrays[i_tensor].shape

    return inputs, shapes, arrays, size_dict


def contract_braket_with_hole(
        braket: Braket,
        hole: int,
        sanity_check: bool = False
    ) -> np.ndarray:
    """
    Contracts `braket` around `hole`, thereby calculating the
    environment at `hole`.

    The environment at a node is a tensor with `3 * len(adj[hole])`
    legs. It's legs come in three groups: First the bra legs, followed
    by the operator legs, and finally the ket legs. Each group contains
    `len(adj[hole])` legs, leading to the total of `3 * len(adj[hole])`
    legs. The ordering within the groups follows the leg ordering of the
    tensor at respective `hole`.
    """
    # sanity check
    if sanity_check: assert braket.intact
    if not hole in braket:
        raise ValueError(f"Node {hole} is not contained in braket.")

    inputs, shapes, arrays, size_dict = braket_to_ctg_arguments(
        braket=braket,
        exclude=(hole,),
        exclude_policy="fill",
        sanity_check=sanity_check
    )

    # Extracting the leg ordering at hole.
    bra_edge_labels, op_edge_labels, ket_edge_labels = [
        inputs_ for inputs_, arrays_ in zip(inputs, arrays)
        if np.isnan(arrays_).any()
    ]

    # Dropping the placeholder inputs and shapes, and the arrays
    # they correspond to.
    inputs = [
        inputs_ for inputs_, arrays_ in zip(inputs, arrays)
        if not np.isnan(arrays_).any()
    ]
    shapes = [
        shapes_ for shapes_, arrays_ in zip(shapes, arrays)
        if not np.isnan(arrays_).any()
    ]
    arrays = [
        arrays_ for arrays_ in arrays
        if not np.isnan(arrays_).any()
    ]

    # Slicing all singleton dimensions.
    inputs, shapes, arrays, size_dict = slice_singleton_dimensions(
        inputs=inputs,
        shapes=shapes,
        arrays=arrays,
        size_dict=size_dict,
    )

    # Leg ordering of the output (dropping physical dimension
    # labels).
    output = (bra_edge_labels[:-1]
              + op_edge_labels[:-2]
              + ket_edge_labels[:-1])

    if sanity_check:
        # Have we correctly accounted for all legs?
        concatenated_labels = sum(inputs, start=output)
        label_counts = {
            label: concatenated_labels.count(label)
            for label in set(concatenated_labels)
        }
        assert all(
            label_count == 2
            for label_count in label_counts.values()
        )
        assert all(
            len(T.shape) == len(inputs_)
            for T, inputs_ in zip(arrays, inputs)
        )
        # Have we correctly accounted for all tensor shapes?
        assert all(
            T.shape == shape for T, shape in zip(arrays, shapes)
        )

    # Executing the contraction.
    eq = ctg.utils.inputs_output_to_eq(
        inputs=inputs,
        output=output
    )
    env = ctg.einsum(eq, *arrays)

    return env


def edge_transf_to_tensor_stack(T: np.ndarray) -> tuple[np.ndarray]:
    """
    Given a linear edge transformation `T` from a braket, splits it into
    three tensors via two SVDs. This transforms the edge transformation
    into a tensor stack that could be inserted into a braket.
    """
    # sanity check
    if not (T.shape[0] == T.shape[3]
            and T.shape[1] == T.shape[4]
            and T.shape[2] == T.shape[5]):
        raise ValueError("Leg size mismatch in tensor T.")

    bra_size = T.shape[0]
    op_size = T.shape[1]
    ket_size = T.shape[2]

    # Transposing T, s.t. the in- and out legs of each layer are consecutive.
    T_ = np.transpose(T, axes=(0, 3, 1, 4, 2, 5))

    # Splitting between operator legs and ket legs.
    T_ = np.reshape(T_, shape=(bra_size**2 * op_size**2, ket_size**2))
    U, singvals, Vh = scialg.svd(
        a=T_,
        full_matrices=False,
        overwrite_a=True
    )
    D_op_ket = len(singvals)
    T_bra_op = np.einsum("ij,j->ij" ,U, np.sqrt(singvals))
    T_ket = np.einsum("i,ij->ij", np.sqrt(singvals), Vh)

    # Re-shaping T_bra_op s.t. bra legs and operator legs are separated.
    T_bra_op = np.reshape(
        T_bra_op,
        shape=(bra_size**2, op_size**2 * D_op_ket)
    )

    # Splitting between bra legs and operator legs.
    U, singvals, Vh = scialg.svd(
        a=T_bra_op,
        full_matrices=False,
        overwrite_a=True
    )
    D_bra_op = len(singvals)
    T_bra = np.einsum("ij,j->ij", U, np.sqrt(singvals))
    T_op = np.einsum("i,ij->ij", np.sqrt(singvals), Vh)

    # Re-shaping an transposing T_bra T_op and T_ket s.t. they form a tensor
    # stack.
    T_bra = np.reshape(T_bra, newshape=(bra_size, bra_size, D_bra_op))
    T_op = np.transpose(
        np.reshape(
            T_op,
            newshape=(D_bra_op, op_size, op_size, D_op_ket)
        ),
        axes=(1, 2, 0, 3)
    )
    T_ket = np.transpose(
        np.reshape(
            T_ket,
            newshape=(D_op_ket, ket_size, ket_size)
        ),
        axes=(1, 2, 0)
    )

    # Braket objects require bra, op and ket to have the same physical
    # dimension at any respective node. If the physical dimensions from the
    # SVDs do not match, we have to enlarge them.

    if D_bra_op != D_op_ket:
        D_larger = max(D_bra_op, D_op_ket)
        # Adding zeros to the physical dimension of the bra tensor, and
        # inserting the old (smaller) tensor.
        T_bra_larger = np.zeros(
            shape=(bra_size, bra_size, D_larger),
            dtype=np.complex128
        )
        bra_slice = (slice(bra_size), slice(bra_size), slice(D_bra_op))
        T_bra_larger[bra_slice] = T_bra

        # Adding zeros to the physical dimension of the operator tensor, and
        # inserting the old (smaller) tensor.
        T_op_larger = np.zeros(
            shape=(op_size, op_size, D_larger, D_larger),
            dtype=np.complex128
        )
        op_slice = (
            slice(op_size),
            slice(op_size),
            slice(D_bra_op),
            slice(D_op_ket)
        )
        T_op_larger[op_slice] = T_op

        # Adding zeros to the physical dimension of the ket tensor, and
        # inserting the old (smaller) tensor.
        T_ket_larger = np.zeros(
            shape=(ket_size, ket_size, D_larger),
            dtype=np.complex128
        )
        ket_slice = (slice(ket_size), slice(ket_size), slice(D_op_ket))
        T_ket_larger[ket_slice] = T_ket

        return T_bra_larger, T_op_larger, T_ket_larger

    else:
        return T_bra, T_op, T_ket


def contract_tensor_inbound_messages(
        braket: Braket,
        node: int,
        neighbors: Iterable[int] = None,
        sanity_check: bool = False
    ) -> Union[float, np.ndarray]:
    """
    Contracts the tensor at `node` with it's incoming messages. If
    given, only messages from nodes in `neighbors` are absorbed. By
    default, all incoming messages are absorbed, yielding a scalar.
    """
    # sanity check
    if sanity_check: assert braket.intact

    # If neighbors is not given, we absorb all incoming messages.
    if neighbors is None: neighbors = tuple(braket.G.adj[node])

    # We use the number of virtual legs of the tensor to enumerate the edges
    # during einsum contraction.
    nLegs = len(braket.G.adj[node])

    args = ()

    for neighbor in neighbors:
        args += (
            braket.msg[neighbor][node],
            (
                # bra leg
                braket.bra.G[node][neighbor][0]["legs"][node],
                # operator leg
                nLegs + braket.op.G[node][neighbor][0]["legs"][node],
                # ket leg
                2*nLegs + braket.ket.G[node][neighbor][0]["legs"][node],
            )
        )

    args += (
        # Bra tensor.
        braket.bra.G.nodes[node]["T"],
        tuple(range(nLegs)) + (3 * nLegs,),
        # Operator tensor.
        braket.op.G.nodes[node]["T"],
        (tuple(nLegs + iLeg for iLeg in range(nLegs))
         + (3 * nLegs, 3*nLegs + 1)),
        # Ket tensor.
        braket.ket.G.nodes[node]["T"],
        tuple(2*nLegs + iLeg for iLeg in range(nLegs)) + (3*nLegs + 1,)
    )

    node_cntr = ctg.einsum(*args)

    return node_cntr


def __contract_tstack_physical_index(tstack: tuple[np.ndarray]) -> np.ndarray:
    """
    Contracts the physical index in a tensor stack, and returns the
    resulting tensor while keeping the leg ordering.
    """
    # Sizes of the legs that will remain after
    bra_sizes = tuple(tstack[0].shape[:-1])
    op_sizes = tuple(tstack[1].shape[:-2])
    ket_sizes = tuple(tstack[2].shape[:-1])

    # sanity check
    if not len(tstack) == 3:
        raise NotImplementedError("".join((
            "Tensor stacks with more than three components are not yet ",
            "supported."
        )))
    if not len(bra_sizes) == len(op_sizes) and len(op_sizes) == len(ket_sizes):
        raise ValueError(
            "Number of virtual legs in tensor stack components do not match."
        )
    if not (tstack[0].shape[-1] == tstack[1].shape[-2]
            and tstack[1].shape[-1] == tstack[2].shape[-1]):
        raise ValueError("Physical dimensions in tensor stack do not match.")

    nLegs = len(bra_sizes)

    args = (
        # bra tensor
        tstack[0],
        tuple(3 * i for i in range(nLegs)) + (3 * nLegs,),
        # operator tensor
        tstack[1],
        tuple(3*i + 1 for i in range(nLegs)) + (3 * nLegs, 3*nLegs + 1),
        # ket tensor
        tstack[2],
        tuple(3*i + 2 for i in range(nLegs)) + (3*nLegs + 1,),
    )
    out_legs = tuple(range(3 * nLegs))

    T = np.einsum(*args, out_legs, optimize=True)

    # sizes of the new legs.
    new_sizes = tuple(
        bra_sizes[i] * op_sizes[i] * ket_sizes[i]
        for i in range(nLegs)
    )
    T = T.reshape(new_sizes)

    return T


def contract_braket_physical_indices(
        braket: Union[BaseBraket, Braket, ExcBraket],
        sanity_check: bool = False
    ) -> Union[BaseBraket, Braket, ExcBraket]:
    """
    Contracts the physical indices at each site. Returns a new braket,
    where the network is contained in `newbraket.ket`. Edge
    transformations and messages are added to the returned braket, if
    present in `braket`.
    """
    if sanity_check: assert braket.intact

    newG = copy.deepcopy(braket.G)

    # Contracting physical dimension in every tensor stack.
    for node in braket:
        newG.nodes[node]["T"] = __contract_tstack_physical_index(braket[node])

    # writing new sizes to the graph.
    for node1, node2 in newG.edges():
        leg1 = newG[node1][node2][0]["legs"][node1]
        size = newG.nodes[node1]["T"].shape[leg1]
        newG[node1][node2][0]["size"] = size

    kwargs = {}
    # Flattening edge transformations, if present.
    if hasattr(braket,"edge_T"):
        kwargs["edge_T"] = {}
        for sender in braket.edge_T.keys():
            kwargs["edge_T"][sender] = {}
            for receiver, proj in braket.edge_T[sender].items():
                size = newG[sender][receiver][0]["size"]
                kwargs["edge_T"][sender][receiver] = (
                    np.nan if np.isnan(proj).any()
                    else np.expand_dims(
                        proj.reshape(size, size),
                        axis=(0, 1, 3, 4)
                    )
                )

    # Flattening messages, if present.
    if hasattr(braket,"msg"):
        if braket.msg is not None:
            kwargs["msg"] = {}
            for sender in braket.msg.keys():
                kwargs["msg"][sender] = {}
                for receiver, msg in braket.msg[sender].items():
                    flat_msg = np.expand_dims(msg.flatten(), axis=(0, 1))
                    kwargs["msg"][sender][receiver] = flat_msg

    newbraket = braket.__class__.Cntr(
        G=newG, sanity_check=sanity_check, **kwargs
    )
    newbraket._converged = braket.converged

    return newbraket


def BP_excitations(
        G: nx.MultiGraph,
        max_order: int = np.inf,
        holes: tuple[int] = (),
        sanity_check: bool = False
    ) -> tuple[nx.MultiGraph]:
    """
    Given the graph `G`, returns the excitations from
    [arXiv:2409.03108](https://arxiv.org/abs/2409.03108) up to order
    `max_order`. The order of an excitation is the number of edges that
    are excited.

    Optionally, excitations for environments can be calculated. If
    given, the nodes in `holes` will be treated as if their legs were
    open legs, which allows for dangling excitations at these nodes.
    Thus, this method returns excitations with which the environment
    at all nodes in `holes` can be calculated.
    """

    if nx.is_tree(G): return ()

    # The loop excitations are the operator chains of a PEPO with bond
    # dimension 2 on the graph G, where the PEPOs local tensors are defined
    # s.t. the only non-zero components are located in indices that ensure
    # that there are no dangling edges.

    # Root node of the PEPO is node with smallest degree.
    root = sorted(G.nodes(), key=lambda x: len(G.adj[x]))[0]

    # Depth-first search tree is PEPO traversal tree.
    tree = nx.dfs_tree(G,root)

    pepoG = PEPO.prepare_graph(G=G, chi=2, sanity_check=sanity_check)

    # Filling the PEPO with tensors that are zero-valued if there is a dangling
    # edge.
    for node in pepoG:
        nLegs = len(pepoG.adj[node])
        # Baseline PEPO tensor, which allows dangling excitations. This is
        # sufficient if node is a hole.
        pepoG.nodes[node]["T"] = np.reshape(
            np.outer(np.ones(shape=2**nLegs), (1, 0, 0, 1)),
            newshape=tuple(2 for _ in range(nLegs + 2))
        )

        if node not in holes:
            # No hole at this node, so no dangling excitations allowed. A
            # dangling excitation is present at a node where only one incoming
            # leg is excited, so we'll set these to zero. Recall that an edge
            # is considered excited, if it is unit-valued.
            for leg in range(nLegs):
                # A unit-valued edge is considered to be excited. This means
                # configuration with one adjacent, unit-valued edge represent
                # dangling excitations.
                idx = (tuple(1 if i == leg else 0 for i in range(nLegs))
                       + (slice(2), slice(2)))
                pepoG.nodes[node]["T"][idx] = np.zeros(shape=(2, 2))

    exc_pepo = PEPO.from_graphs(
        G=pepoG,
        tree=tree,
        check_tree=False,
        sanity_check=sanity_check
    )

    _, virt_idx_list = exc_pepo.operator_chains(
        return_virtidx=True,
        sanity_check=sanity_check
    )

    excitations = ()
    # Assembling the excitation graphs from virt_idx_list.
    for virt_idx in virt_idx_list:
        if sum(virt_idx.values()) > max_order:
            # This excitation has more edges than we asked for. Skipping.
            continue

        if sum(virt_idx.values()) == 0:
            # The BP vacuum is not an excited state.
            continue

        excitations += (nx.MultiGraph(incoming_graph_data=(
            tuple(edge)
            for edge, idx in virt_idx.items()
            if idx == 1
        )),)

    return excitations


def assemble_excitation_brakets(
        braket: Braket,
        excitation: nx.MultiGraph,
        sanity_check: bool = False,
        **kwargs
    ) -> tuple[ExcBraket]:
    """
    Assembles the brakets that contain the excitation `excitation`.
    Brakets are returned as a tuple, with as many brakets as there are
    disjoint components to the excitation. The product of the
    contraction values of the brakets gives the contribution of the
    excitation.
    
    Method taken from
    [arXiv:2409.03108](https://arxiv.org/abs/2409.03108). `excitation`
    contains the excited edges, `braket` contains the tensor network,
    the projectors and the messages. `kwargs` are passed to
    `Braket.contract`.
    """

    if not nx.is_connected(G=excitation):
        # The excitation is not connected; finding connected components.
        connected_excitations = tuple(
            excitation.subgraph(component)
            for component in nx.connected_components(excitation)
        )

        if sanity_check:
            # Are the excitations disjoint?
            for i, j in itertools.combinations(
                range(len(connected_excitations)),
                r=2
            ):
                exc1 = connected_excitations[i]
                exc2 = connected_excitations[j]
                shared_nodes = set(exc1.nodes()) & set(exc2.nodes())
                if len(shared_nodes) > 0:
                    raise RuntimeError(
                        f"Excitations {i} and {j} are not disjoint."
                    )

        # For disjoint excitations, the contribution is the product of the
        # components.
        all_excs = sum(
            (assemble_excitation_brakets(
                braket=braket,
                excitation=exc,
                sanity_check=sanity_check,
                **kwargs
            )
            for exc in connected_excitations),
            start=()
        )

        return all_excs

    # How will this work under the hood? We truncate the network. All edges
    # that are not contained in the excitation will be removed, and new edges
    # will be added that connect the messages that flow into the excitation.
    # On the excitations edges we add the projectors.

    # The set of nodes inside the excitation, and the set of edges that are not
    # excited.
    excitation_nodes = set(excitation.nodes())
    non_excitation_nodes = set(braket.G.nodes()) - excitation_nodes
    non_excitation_edges = nx.MultiGraph(incoming_graph_data=braket.G.edges())
    non_excitation_edges.remove_edges_from(excitation.edges())

    # The graphs that we will use to construct a new braket.
    G_bra = copy.deepcopy(braket.bra.G)
    G_op = copy.deepcopy(braket.op.G)
    G_ket = copy.deepcopy(braket.ket.G)

    # Removing nodes and edges that are not present in the excitation.
    for G in (G_bra, G_op, G_ket):
        G.remove_edges_from(non_excitation_edges.edges())
        G.remove_nodes_from(non_excitation_nodes)

    for node in excitation_nodes:
        non_contained_neighbors = set(braket.G.adj[node]) - set(G_op.adj[node])
        # Inserting messages as new sites in the graph.
        while len(non_contained_neighbors) > 0:
            # If this is the case, there are dangling tensor legs to which no
            # message is attached. Identifying the dangling neighbor, and the
            # tensor leg it connects to.
            next_node_label = max(node_ for node_ in G) + 1
            neighbor = non_contained_neighbors.pop()

            # The new node will contain a message that is inbound to the
            # excitation. Since messages are dense tensors, the message will be
            # inserted into the operator graph, and it's dimensions will be
            # permuted s.t. the operator leg is the first leg.
            op_T = np.transpose(braket.msg[neighbor][node], axes=(1, 0, 2))

            # Inserting a new node that contains the respective message.
            G_op.add_node(
                node_for_adding=next_node_label,
                T=op_T
            )
            G_bra.add_node(
                node_for_adding=next_node_label,
                T=np.eye(op_T.shape[-2])
            )
            G_ket.add_node(
                node_for_adding=next_node_label,
                T=np.eye(op_T.shape[-1])
            )

            # Connecting the node to the excitation graph.
            for G, legG in zip(
                (G_bra, G_op, G_ket),
                (braket.bra.G, braket.op.G, braket.ket.G)
            ):
                leg = legG[node][neighbor][0]["legs"][node]
                G.add_edge(
                    u_for_edge=node,
                    v_for_edge=next_node_label,
                    legs={node: leg, next_node_label: 0},
                    trace=False,
                    indices=None
                )

    # Inserting projectors as nodes on the excited edges.
    for node1, node2 in excitation.edges():
        # Label of the node we will add.
        next_node_label = max(node_ for node_ in G) + 1

        Tstack = edge_transf_to_tensor_stack(braket.edge_T[node1][node2])

        for G, legG, T in zip(
            (G_bra, G_op, G_ket),
            (braket.bra.G, braket.op.G, braket.ket.G),
            Tstack
        ):
            leg1 = legG[node1][node2][0]["legs"][node1]
            leg2 = legG[node1][node2][0]["legs"][node2]
            # Adding a node that contains the projector, and connecting it to
            # the graph.
            G.add_node(
                node_for_adding=next_node_label,
                T=T
            )
            G.add_edge(
                u_for_edge=node1,
                v_for_edge=next_node_label,
                legs={node1: leg1, next_node_label: 1},
                trace=False,
                indices=None
            )
            G.add_edge(
                u_for_edge=node2,
                v_for_edge=next_node_label,
                legs={node2: leg2, next_node_label: 0},
                trace=False,
                indices=None
            )
            # Removing the old edge.
            G.remove_edge(node1, node2)

    # Search tree for the excitation braket.
    exc_root = sorted(G_op.nodes(), key=lambda x: len(G_op.adj[x]))[0]
    exc_tree = nx.dfs_tree(G_op, exc_root)

    # Constructing a new braket for this excitation, and contracting it to get
    # the contribution of this excitation.
    exc_braket = ExcBraket(
        bra=PEPS(G=G_bra),
        op=PEPO.from_graphs(G=G_op, tree=exc_tree, check_tree=False),
        ket=PEPS(G=G_ket),
        exc=excitation,
        sanity_check=sanity_check
    )

    # Saving the excitation in the braket.
    exc_braket.exc = excitation

    return (exc_braket,)


if __name__ == "__main__":
    pass
