"""
Projector-entangled pair operators on arbitrary graphs.
"""

__all__ = ["PEPO", "PauliPEPO"]

import warnings
import itertools
from typing import Union, Iterable, Iterator, Tuple, List, Dict, FrozenSet
import copy

import numpy as np
import networkx as nx
import sparse
import cotengra as ctg
import scipy.sparse as scisparse

from belief_propagation.utils import (
    network_message_check,
    multi_kron,
    proportional,
    same_legs,
    graph_compatible
)
from belief_propagation.PEPS import PEPS


# -----------------------------------------------------------------------------
#                   Operator classes
# -----------------------------------------------------------------------------


class PEPO:
    """
    Base class for tensor product operators, that are constructed
    mathematically as sums of operator chains. Subclasses must provide
    `__init__`. Therein, the following attributes must be defined:
    * `self.G`: Graph on which the Hamiltonian is defined.
    * `self.D`: Physical dimension.
    * `self.tree`: Tree that determines the traversal of `G`, along
    which the PEPO is oriented.
    * `self.root`: The root node of the tree.

    Writing down a PEPO on an arbitrary graph `G` can be achieved by
    finding a spanning tree of `G`. The flow of finite state automaton
    information is then defined by the tree: The origin is at the root,
    and it terminates at the leaves. This method is similar to what is
    presented in [SciPost Phys. Core 7, 036
    (2024)](https://doi.org/10.21468/SciPostPhysCore.7.2.036).

    Every node in the graph has one inbound leg, passive legs, and
    outbound legs. The initial and final state of the finite state
    automaton are the first and last components, respectively. The
    initial state ("particle state") is only passed along the tree,
    while intermediate states ("decay states") can be passed along any
    edge. The final state ("vacuum state") is passed along every edge.
    The physical legs are the last two dimensions if the PEPO tensors.
    All other legs are virtual bond dimensions. The correspondence
    between legs and neighbors is determined by the `legs` attribute on
    each edge.

    During initialisation, the root and each leaf are equipped with an
    additional leg. These legs connect to the initial and final states
    of the finite state automaton, respectively. At the end of
    initialisation, these additional legs should be contracted such that
    the PEPO has the same structure as the underlying graph.

    It is assumed that, within the virtual dimension, the finite state
    automaton initial state is the 0th component, and the final state is
    the last component. When traversing the graph along operator chains
    from root to leaves, the virtual bond dimension index must be
    non-decreasing.
    """

    def contract_boundaries(self):
        """
        Contracts boundary legs. Boundary legs are assumed to be located
        in the last virtual dimension.

        This means contraction of the root node with the initial state,
        and each leaf with the final state. It is assumed that the
        initial state is the 0th component of the bond dimension, and
        that the final state is the last component.
        """
        for node in self.G.nodes():
            if node == self.root: # root node
                self[node] = self[node][...,0,:,:]
                continue

            if len(self.tree.succ[node]) == 0: # leaf node
                self[node] = self[node][...,-1,:,:]
                continue

        return

    def toarray(
            self,
            create_using: str = "cotengra",
            sanity_check: bool = False
        ):
        """
        Construct a matrix representation of this operator. Different
        methods are implemented, which can be selected by the argument
        `create_using`:
        * `cotengra`: Dense Numpy-array from contraction of the PEPO.
        * `scipy.csr`: Scipy csr-sparse array, constructed using
        operator chains.
        * `sparse`: Scipy csr-sparse matrix from sparse contraction of
        the network. Works for small PEPOs only.

        The order of the physical dimensions is inherited from the
        labels of the nodes: the nodes are sorted in ascending order.        
        """

        if create_using == "cotengra":
            return self.__to_dense(sanity_check=sanity_check)

        if create_using == "scipy.csr":
            return self.__to_sparse(
                create_using="scipy.csr",
                sanity_check=sanity_check
            )

        if create_using == "sparse":
            return self.__to_sparse(
                create_using="sparse",
                sanity_check=sanity_check
            )

        raise ValueError("".join((
            "toarray not implemented for method ",
            create_using,
            "."
        )))

    def __to_dense(self, sanity_check: bool) -> np.ndarray:
        """
        Constructs the dense operator using `ctg.einsum`.
        """
        if sanity_check: assert self.intact

        if self.nsites == 1:
            # the network is trivial
            return tuple(self.G.nodes(data="T"))[0][1]

        inputs = ()
        tensors = ()

        # Enumerating the edges in the graph.
        for i, nodes in enumerate(self.G.edges()):
            node1, node2 = nodes
            self.G[node1][node2][0]["label"] = ctg.get_symbol(i)

        N_edges = self.G.number_of_edges()
        # Assembling the einsum arguments.
        for i, node in enumerate(sorted(self)):
            T = self[node]
            legs = [
                None
                for _ in range(T.ndim-2)
            ] + [
                ctg.get_symbol(N_edges + i),
                ctg.get_symbol(N_edges + self.G.number_of_nodes() + i)
            ] # last two indices are the physical legs

            for _, neighbor, edge_label in self.G.edges(
                nbunch=node,
                data="label"
            ):
                legs[self.G[node][neighbor][0]["legs"][node]] = edge_label

            inputs += (legs,)
            tensors += (T,)

        # output ordering
        output = tuple(
            ctg.get_symbol(i)
            for i in range(N_edges, N_edges + 2 * self.G.number_of_nodes())
        )

        # getting the einsum expression, and contracting
        expr = ctg.utils.inputs_output_to_eq(inputs=inputs, output=output)
        H = ctg.einsum(expr, *tensors)
        H = np.reshape(
            H,
            newshape=(
                self.D ** self.G.number_of_nodes(),
                self.D ** self.G.number_of_nodes()
            )
        )

        return H

    def __to_sparse(
            self,
            create_using: str,
            sanity_check: bool
        ) -> Union[scisparse.csr_array, scisparse.csr_matrix]:
        """
        Constructs the operator using sparse operations.
        """
        if sanity_check: assert self.intact

        if self.nsites == 1:
            # The network is trivial.
            return tuple(self.G.nodes(data="T"))[0][1]

        if create_using == "scipy.csr":
            # H will be constructed by summing the contributions from all
            # operator chains. array construction is fastest using the coo
            # format, but I'm returning csr because this is optimal for
            # matrix-vector multiplication; an operation that the Lanczos
            # algorithm heavily relies on.
            chains = self.operator_chains(sanity_check=sanity_check)

            H = scisparse.csr_array((self.D**self.nsites, self.D**self.nsites))
            nodes = tuple(sorted(self))

            for chain in chains:
                ops = tuple(
                    scisparse.coo_array(chain[node]) if node in chain.keys()
                    else scisparse.eye_array(self.D,format="coo")
                    for node in nodes[::-1]
                )
                H += multi_kron(*ops,create_using="scipy.coo")

            return H.tocsr()

        if create_using == "sparse":
            if self.G.number_of_edges() > len(
                np.core.einsumfunc.einsum_symbols
            ):
                raise RuntimeError("".join((
                    f"The sparse package allows ",
                    "einsum contractions with up to ",
                    f"{len(np.core.einsumfunc.einsum_symbols)} indices. The ",
                    f"operator has ({self.G.number_of_edges()} edges)."
                )))

            inputs = ()
            tensors = ()

            # Enumerating the edges in the graph.
            for i, nodes in enumerate(self.G.edges()):
                node1, node2 = nodes
                self.G[node1][node2][0]["label"] = ctg.get_symbol(i)

            N_edges = self.G.number_of_edges()
            output = tuple(
                ctg.get_symbol(N_edges + i)
                for i in range(2 * self.nsites)
            )

            # Assembling the einsum arguments.
            for i, node in enumerate(sorted(self)):
                legs = [
                    None
                    for _ in range(self[node].ndim-2)
                ] + [
                    ctg.get_symbol(N_edges + i),
                    ctg.get_symbol(N_edges + self.G.number_of_nodes() + i)
                ]

                for _, neighbor, edge_label in self.G.edges(
                    nbunch=node,
                    data="label"
                ):
                    legs[self.G[node][neighbor][0]["legs"][node]] = edge_label

                inputs += (tuple(legs),)
                tensors += (sparse.GCXS(self[node]),)

            # Einsum expression, with all physical dimensions in ellipsis.
            einsum_expr = ctg.utils.inputs_output_to_eq(
                inputs=inputs,
                output=output
            )
            # Contraction using einsum.
            H = sparse.einsum(einsum_expr, *tensors)
            H = sparse.reshape(
                H,
                shape=(
                    self.D ** self.G.number_of_nodes(),
                    self.D ** self.G.number_of_nodes()
                )
            )

            return H.to_scipy_sparse()

        raise ValueError(
            "__to_sparse not implemented for method " + create_using + "."
        )

    def conj(self, sanity_check: bool = False):
        """
        Adjoint operator.
        """
        if sanity_check: assert self.intact

        newPEPO = copy.deepcopy(self)
        for node in self.G.nodes(): newPEPO[node] = self[node].conj()

        return newPEPO

    def view_site(self,node: int):
        """
        Prints all components of the tensor at node `node`.
        """
        # sanity check
        assert node in self.G

        legs_in = tuple(
            self.G[node][neighbor][0]["legs"][node]
            for neighbor in self.tree.pred[node].keys()
        )
        legs_out = tuple(
            self.G[node][neighbor][0]["legs"][node]
            for neighbor in self.tree.succ[node].keys()
        )

        print("".join((
            f"Displaying node {node}:" + "\n    ",
            f"legs {legs_in} incoming" + "\n    ",
            f"legs {legs_out} outgoing" + "\n"
        )))
        for virtual_index in itertools.product(*[
            range(self[node].shape[i])
            for i in range(self[node].ndim - 2)
        ]):
            index = virtual_index + (slice(0,self.D), slice(0,self.D))

            if not np.allclose(self.G.nodes[node]["T"][index], 0):
                print(index[:-2], ":\n" ,self[node][index], "\n")

        return

    def legs_dict(self, node: int, sanity_check: bool = False) -> dict:
        """
        Returns the `legs` attributes of the adjacent edges of `node` in
        a dictionary structure: `legs_dict[neighbor]` is the same as
        `self.G[node][neighbor][0]["legs"]`.
        """
        if sanity_check: assert self.intact

        out = dict()

        for neighbor in self.G.adj[node]:
            out[neighbor] = self.G[node][neighbor][0]["legs"]

        return out

    def traversal_tensor(
            self,
            chi: Union[int, Iterable[int]],
            N_pas: int,
            N_out: int,
            dtype=np.complex128
        ) -> np.ndarray:
        """
        Returns the minimum tensor for PEPOs, that is, a tensor that
        ensures correct tree traversal.

        `T` has the canonical leg ordering: The first leg is the
        incoming leg, afterwards follow the passive legs, and finally
        the outgoing legs.

        If a tuple of integers is passed as `chi`, the contents
        determine the bond dimension for every leg.        
        """
        # sanity check
        assert N_pas >= 0
        assert N_out >= 0

        if isinstance(chi, int):
            return self.traversal_tensor(
                chi=tuple(chi for _ in range(1 + N_pas + N_out)),
                N_pas=N_pas,
                N_out=N_out,
                dtype=dtype
            )

        if hasattr(chi, "__iter__"):
            # sanity check
            for i, chi_ in enumerate(chi):
                if not isinstance(chi_, int):
                    raise ValueError(f"chi[{i}] is not an integer.")
            if not i + 1 == 1 + N_pas + N_out:
                raise ValueError(
                    "chi contains the wrong number of bond dimensions."
                )

            T = np.zeros(
                shape=[chi_ for chi_ in chi] + [self.D, self.D],
                dtype=dtype
            )

            # particle index
            for i_out in range(N_out):
                index = list(
                    chi_ - 1
                    for chi_ in chi
                ) + [
                    slice(0, self.D), slice(0, self.D)
                ]
                index[0] = 0
                index[1 + N_pas + i_out] = 0
                T[tuple(index)] = self.I

            # vacuum index
            index = tuple(
                chi_ - 1
                for chi_ in chi
            ) + (
                slice(0, self.D), slice(0, self.D)
            )
            T[index] = self.I

            return T

        raise ValueError("chi must be an integer or an iterable of ints.")

    def operator_chains(
            self,
            save_tensors: bool = True,
            remove_ids: bool = True,
            return_virtidx: bool = False,
            sanity_check: bool = False
        ) -> Tuple[Dict[int, np.ndarray]]:
        """
        Returns all operator chains. An operator chain is a collection
        of operators. The summation of the tensor products of all
        operator chains gives the operator.

        Operator chains are returned as a dict, where nodes are keys.
        If `save_tensors = True` (default), local operators are values;
        otherwise, indices to PEPO tensors are values. If `remove_ids =
        True` (default), identities are removed. If `return_virtidx =
        True`, the virtual indices on every edge are returned as a
        dictionary alognside the operator chains.
        """
        # Sanity checks.
        if sanity_check: assert self.intact

        # Why a recursion? For large graphs, simply iterating through all
        # indices to find valid chains might take prohibitively long.

        operator_chains: List[Dict[int, Tuple[int]]] = []
        operator_chain_virtidx: List[Dict[FrozenSet[int], int]] = []

        self.__chain_construction_recursion(
            edges_to_indices={},
            chains=operator_chains,
            edges_to_indices_list=operator_chain_virtidx,
            sanity_check=sanity_check
        )

        # removing identity operators from the chain, substituting indices with
        # operators
        for iChain, chain in enumerate(operator_chains):
            # any identities in the chain?
            is_id = {
                node: np.allclose(self[node][chain[node]], self.I)
                for node in chain.keys()
            }

            if all(is_id.values()):
                # this chain consists of identities only, and needs special
                # treatment
                if remove_ids:
                    operator_chains[iChain] = {self.root: chain[self.root]}

                if save_tensors:
                    for node in operator_chains[iChain].keys():
                        idx = operator_chains[iChain][node]
                        operator_chains[iChain][node] = self[node][idx]

            for key in is_id.keys():
                T = self[key][chain[key]]

                # removing identities
                if remove_ids and is_id[key]:
                    chain.pop(key, None)
                    continue

                # substituting indices with tensors
                if save_tensors: chain[key] = T

        if return_virtidx:
            return tuple(operator_chains), operator_chain_virtidx
        else:
            return tuple(operator_chains)

    def __chain_construction_recursion(
            self,
            edges_to_indices: Dict[FrozenSet[int], int],
            chains: List[Dict[int, Tuple[int]]],
            edges_to_indices_list: List[Dict[FrozenSet[int], int]],
            sanity_check: bool = False
        ) -> None:
        """
        Traversing the PEPO graph, collecting the operator chains.
        Chains are saved in the argument `chains`, which is manipulated
        in-place. `edges_to_indices_list` contains the edge indices of
        every chain.

        This function recursively traverses the graph by moving one step
        out from the current set `edges_to_indices`.
        """
        if all_edges_present(
            G=self.G,
            edges_to_indices=edges_to_indices,
            sanity_check=sanity_check
        ):
            # The current chain has traversed the entire graph, and is thus
            # complete. Let's save it.
            chain = {}

            # Saving virtual indices for every node.
            for node in self:
                chain[node] = edge_indices_to_site_index(
                    G=self.G,
                    node=node,
                    edges_to_indices=edges_to_indices,
                    sanity_check=sanity_check
                ) + (
                    slice(0, self.D), slice(0, self.D)
                )

            # Saving the chain.
            chains.append(chain)
            edges_to_indices_list.append(edges_to_indices)

            return

        # which edges have we not yet visited?
        if len(edges_to_indices) == 0:
            next_edges = tuple(edge for edge in self.G.edges(nbunch=self.root))
        else:
            next_edges = get_next_edges(
                G=self.G,
                edge_set=tuple(
                    tuple(edge)
                    for edge in edges_to_indices.keys()
                ),
                sanity_check=sanity_check
            )

        if len(next_edges) == 0:
            # nothing to do here
            return

        # which nodes belonog to the unvisited edges?
        next_nodes = set().union(*[set(edge) for edge in next_edges])

        for neighbor_indices in itertools.product(*[
            range(self.G[edge[0]][edge[1]][0]["size"])
            for edge in next_edges]
        ):
            # preparing the next iteration
            new_edges_to_indices = copy.deepcopy(edges_to_indices)

            # adding new edge indices
            for i, edge in enumerate(next_edges):
                new_edges_to_indices[frozenset(edge)] = neighbor_indices[i]

            # do these indices contain a zero-valued local operator?
            zero_valued_chain = False
            for node in next_nodes:
                if all_edges_present(
                    G=self.G,
                    edges_to_indices=new_edges_to_indices,
                    node=node,
                    sanity_check=sanity_check
                ):
                    index = edge_indices_to_site_index(
                        G=self.G,
                        node=node,
                        edges_to_indices=new_edges_to_indices,
                        sanity_check=sanity_check
                    ) + (
                        slice(0, self.D), slice(0,self.D)
                    )
                    if np.allclose(self[node][index], 0):
                        # zero-valued index; this operator chain is zero;
                        # stopping recursion
                        zero_valued_chain = True
                        break

            if zero_valued_chain: continue

            # continuing recursion deeper into the graph
            self.__chain_construction_recursion(
                edges_to_indices=new_edges_to_indices,
                chains=chains,
                sanity_check=sanity_check,
                edges_to_indices_list=edges_to_indices_list
            )

        return

    def _canonical_to_correct_legs(
            self,
            T: np.ndarray,
            node: int
        ) -> np.ndarray:
        """
        Re-shapes the PEPO-tensor `T` at node `node` such that it fits
        into the graph `self.G`.

        It is assumed that the dimensions of `T` are in the canonical
        order: First the virtual dimensions, followed by two physical
        legs. The leading virtual dimension is the incoming leg, which
        is followed by the passive legs and, finally, the outgoing legs.

        For a node in `G` with N neighbors, `T` must have at least N+2
        legs. The last two legs are assumed to be the physical legs, and
        the first N legs are the virtual bonds. Legs that are neither
        among the first N or the last 2 remain untouched; these are
        boundary legs that connect to the initial and final states of
        the finite state automaton. This function thus permutes only the
        first N dimensions.
        """
        # sanity check
        assert node in self.G
        assert node in self.tree

        N_in = len(self.tree.pred[node])
        N_out = len(self.tree.succ[node])
        N_pas = len(self.G.adj[node]) - N_out - N_in

        # sanity check
        if T.ndim - N_in - N_pas - N_out - 2  not in (0,1):
            raise ValueError("Tensor may have up to one boundary leg.")

        newshape = [np.nan for _ in self.G.adj[node]]
        out_counter = 0
        pas_counter = 0
        for neighbor in self.G.adj[node]:
            leg = self.G[node][neighbor][0]["legs"][node]

            if neighbor in self.tree.pred[node]:
                # the incoming edge
                newshape[leg] = 0
                continue

            if neighbor in self.tree.succ[node]:
                # outgoing edge
                newshape[leg] = N_in + N_pas + out_counter
                out_counter += 1
                continue

            # passive edge
            newshape[leg] = N_in + pas_counter
            pas_counter += 1

        return np.transpose(
            T,
            axes=newshape + [_ for _ in range(len(newshape), T.ndim)]
        )

    def _permute_virtual_dimensions(
            self,
            G: nx.MultiGraph,
            sanity_check: bool = False
        ) -> None:
        """
        Changes the leg ordering to the one given in `G`.
        """
        # sanity check
        if not network_message_check(G):
            raise ValueError(
                "Given graph does not contain a valid leg ordering."
            )
        if not graph_compatible(self.G, G, sanity_check=sanity_check):
            raise ValueError("Given graph is not compatible with self.G.")

        # transposing site tensors
        for node in self.G.nodes():
            N_neighbors = len(self.G.adj[node])
            # assembling axis for transpose
            axes = ([None for _ in range(N_neighbors)]
                    + [N_neighbors, N_neighbors+1])

            for neighbor in self.G.adj[node]:
                leg = G[node][neighbor][0]["legs"][node]
                axes[leg] = self.G[node][neighbor][0]["legs"][node]

            self[node] = np.transpose(self[node],axes=axes)

        # updating leg orderings
        for node1,node2 in self.G.edges():
            self.G[node1][node2][0]["legs"] = G[node1][node2][0]["legs"]

        if sanity_check: assert self.intact

        return

    @property
    def I(self) -> np.ndarray:
        """Identity matrix with the dimensions `(self.D, self.D)`."""
        return np.eye(self.D)

    @property
    def nsites(self) -> int:
        """
        Number of sites on which the operator is defined.
        """
        return self.G.number_of_nodes()

    @property
    def intact(self) -> bool:
        """
        Whether the PEPO is intact:
        * Is the underlying network message-ready?
        * Is the size of every edge saved?
        * Are the physical legs the last two dimensions in each tensor?
        * Do the physical legs have the correct sizes?
        * is the information flow in the tree intact?
        """
        # Are all the necessary attributes defined?
        assert hasattr(self, "D")
        assert hasattr(self, "G")
        assert hasattr(self, "tree")
        assert hasattr(self, "root")

        # Is the underlying network message-ready?
        if not network_message_check(self.G):
            warnings.warn("Network not intact.", RuntimeWarning)
            return False

        # Size attribute given on every edge?
        for node1, node2, data in self.G.edges(data=True):
            if not "size" in data.keys():
                warnings.warn(
                    f"No size saved in edge ({node1},{node2}).",
                    RuntimeWarning
                )
                return False

            if data["size"] != self.G[node1][node2][0]["size"]:
                warnings.warn(
                    f"Wrong size saved in edge ({node1},{node2}).",
                    RuntimeWarning
                )
                return False

        # Are the physical legs the last dimension in each tensor?
        for node, T in self.G.nodes(data="T"):
            legs = [leg for leg in range(T.ndim)]
            # Accounting for virtual dimensions.
            for node1, node2, key in self.G.edges(node, keys=True):
                try:
                    if not self.G[node1][node2][key]["trace"]:
                        legs.remove(self.G[node1][node2][key]["legs"][node])
                    else:
                        # trace edge
                        i1, i2 = self.G[node1][node2][key]["indices"]
                        legs.remove(i1)
                        legs.remove(i2)
                except ValueError:
                    warnings.warn(
                        f"Wrong leg in edge ({node1},{node2},{key}).",
                        RuntimeWarning
                    )
                    return False

            if not legs == [T.ndim - 2, T.ndim - 1]:
                warnings.warn(
                    "".join((
                        "Physical legs are not the last two dimensions in ",
                        f"node {node}."
                    )),
                    RuntimeWarning
                )
                return False

            if not (T.shape[-2] == self.D and T.shape[-1] == self.D):
                warnings.warn(
                    f"Hilbert space at node {node} has wrong size.",
                    RuntimeWarning
                )
                return False

        if not self.check_tree:
            # The following tests fail if the PEPO is constructed using
            # PEPO.__add__ (status on 5th of February).
            return True

        # Tree traversal correct?
        for node in self.tree.nodes():
            if (len(self.tree.succ[node]) > 0) and (node != self.root):
                # Node is an intermediate node in the tree.

                # Checking if the particle state is pased along.
                for parent, child in itertools.product(
                    self.tree.pred[node],
                    self.tree.succ[node]
                ):
                    index = tuple(
                        0 if _ in (
                            self.G[node][parent][0]["legs"][node],
                            self.G[node][child][0]["legs"][node]
                        )
                        else -1
                        for _ in range(self[node].ndim - 2)
                    ) + (
                        slice(0, self.D), slice(0, self.D)
                    )
                    if not np.allclose(self[node][index], self.I):
                        warnings.warn(
                            "".join((
                                "Wrong indices for particle state ",
                                f"passthrough in node {node}."
                            )),
                            RuntimeWarning
                        )
                        return False

                # Checking if the vacuum state is passed along.
                index = (tuple(-1 for _ in range(self[node].ndim - 2))
                         + (slice(0, self.D), slice(0, self.D)))
                if not np.allclose(self[node][index], self.I):
                    warnings.warn(
                        "".join((
                            "Wrong indices for vacuum state passthrough in ",
                            f"node {node}."
                        )),
                        RuntimeWarning
                    )
                    return False

                # Checking if a particle state is passed along a passive edge.
                for parent, child in itertools.product(
                    self.tree.pred[node],
                    self.G.adj[node]
                ):
                    if (
                        child in self.tree.succ[node]
                        or child in self.tree.pred[node]
                    ):
                        continue

                    index = tuple(
                        0 if _ in (
                            self.G[node][parent][0]["legs"][node],
                            self.G[node][child][0]["legs"][node]
                        )
                        else -1
                        for _ in range(self[node].ndim - 2)
                    ) + (
                        slice(0, self.D), slice(0, self.D)
                    )
                    if np.allclose(self[node][index], self.I):
                        warnings.warn(
                            "".join((
                                "Particle state passed along passive ",
                                f"edge ({node},{child})."
                            )),
                            RuntimeWarning
                        )
                        return False

        return True

    @property
    def hermitian(self) -> bool:
        """
        A PEPO is hermitian, if all it's site tensors are hermitian.
        """
        # this test fails when I multiply two TFI PEPOs using PEPO.__matmul__,
        # although dense matrix of TFI @ TFI is hermitian; TODO revise this

        # site tensors hermitian?
        for node in self.G.nodes():
            axes = (tuple(range(len(self.G.adj[node])))
                    + (len(self.G.adj[node]) + 1, len(self.G.adj[node])))
            if not np.allclose(
                self[node],
                np.transpose(self[node], axes=axes)
            ):
                return False

        return True

    @staticmethod
    def view_tensor(T:np.ndarray):
        """
        Prints all extractable information from tensor `T`
        """
        D = T.shape[-1]
        # sanity check
        for i in range(T.ndim):
            if i >= T.ndim - 2:
                assert T.shape[i] == D

        # printing cellular automaton components
        for virtual_index in itertools.product(*[
            range(T.shape[i])
            for i in range(T.ndim - 2)
        ]):
            index = virtual_index + (slice(0, D), slice(0, D))

            if not np.allclose(T[index], 0):
                print(index[:-2], ":\n", T[index], "\n")

    @staticmethod
    def prepare_graph(
            G: nx.MultiGraph,
            chi: int = None,
            sanity_check: bool = False
        ) -> nx.MultiGraph:
        """
        Creates a shallow copy of G, and adds the keys `legs`, `trace`
        and `indices` to the edges. If `np.isnan(chi)` is False, the
        size `chi` will be added to each edge.

        The leg ordering is preserved, `trace` is set to `False` and,
        accordingly, `indices` to `None`.

        To be used in `__init__` of subclasses of `PEPO`: `G` is the
        graph from which the operator inherits it's underlying graph.
        """
        # shallow copy of G
        newG = nx.MultiGraph(G.edges())

        # adding additional information to every edge
        for node1, node2, legs in newG.edges(data="legs", keys=False):
            newG[node1][node2][0]["trace"] = False
            newG[node1][node2][0]["indices"] = None
            newG[node1][node2][0]["legs"] = {}

        for node in newG.nodes:
            # adding to the adjacent edges which index they correspond to
            for i, neighbor in enumerate(newG.adj[node]):
                newG[node][neighbor][0]["legs"][node] = i

        if chi is not None:
            if not np.isinf(chi):
                if not np.isclose(int(chi), chi):
                    raise ValueError("Size must be an integer.")

                # writing size chi to each edge
                for node1, node2 in newG.edges(keys=False):
                    newG[node1][node2][0]["size"] = int(chi)

            else:
                # writing size chi to each edge
                for node1, node2 in newG.edges(keys=False):
                    newG[node1][node2][0]["size"] = chi

        if sanity_check: assert network_message_check(newG)

        return newG

    @classmethod
    def from_graphs(
            cls,
            G: nx.MultiGraph,
            tree: nx.DiGraph,
            check_tree: bool = True,
            sanity_check: bool = False
        ):
        """
        Initialisation from a graph `G` that contains PEPO tensors, and
        a tree `tree` that determines the graph traversal by the finite
        state automaton.
        """
        # inferring physical dimension
        D = tuple(T.shape[-1] for node, T in G.nodes(data="T"))[0]
        # inferring root node
        root = sorted(tuple(tree.nodes), key=lambda x:len(tree.pred[x]))[0]

        # initialising the new PEPO
        op = cls(D=D)
        op.G = G
        op.tree = tree
        op.root = root
        op.check_tree = check_tree

        if sanity_check: assert op.intact
        return op

    def __init__(self, D: int) -> None:
        self.D = D
        """Physical dimension."""
        self.G: nx.MultiGraph
        """Grapth that contains PEPO local tensors."""
        self.tree: nx.DiGraph
        """Spanning tree of the graph."""
        self.root: int
        """Root node of the spanning tree."""
        self.check_tree: bool = True
        """
        False if this PEPO is the result of a summation. Means that the
        tree traversal checks in `self.intact` are disabled.
        """
        # TODO: I don't like that I have to disable the tree traversal checks;
        # maybe find a workaround?

        return

    def __getitem__(self, node: int) -> np.ndarray:
        """
        Subscripting with a node gives the tensor at that node.
        """
        if not self.G.has_node(node):
            raise ValueError(f"Node {node} not present in graph.")

        return self.G.nodes[node]["T"]

    def __setitem__(self, node: int, T: np.ndarray) -> None:
        """
        Changing tensors directly.
        """
        if not self.G.has_node(node):
            raise ValueError(f"Node {node} not present in graph.")

        if "T" in self.G.nodes[node].keys():
            if not (
                T.ndim == self.G.nodes[node]["T"].ndim
                or T.ndim == len(self.G.adj[node]) + 2
            ):
                # first checks against previous tensor, second checks against
                # number of legs that are necessary in the given graph
                raise ValueError(
                    "Attempting to set site tensor with wrong number of legs."
                )
        else:
            # I do not check the dimensions of the tensor here because the
            # dimensions are different from the above cases, while the PEPO is
            # constructed. It is advised to check self.intact after
            # construction of the PEPO.
            pass

        self.G.nodes[node]["T"] = T

        return

    def __mul__(self, x: float):
        """
        Multiplication of the whole PEPO with a scalar.
        """
        if not np.isscalar(x): raise ValueError("x must be a scalar.")
        newPEPO = copy.deepcopy(self)

        # since we are inserting additional factors into the PEPO, the tree
        # traversal check will fail
        newPEPO.check_tree = False

        # what we ae really doing is multiplying every operator chain by x;
        # this is more computationaly intensive, but has the advantage that the
        # sanity check still works (otherwise, identity operators would be
        # multiplied by x, which makes the sanity check fail)
        chains = newPEPO.operator_chains(save_tensors=False)

        for chain in chains:
            for node, index in chain.items():
                newPEPO[node][index] *= x
                break

        return newPEPO

    def __rmul__(self, x: float): return self.__mul__(x)

    def __matmul__(
            self,
            psi: Union["PEPO", PEPS, np.ndarray]
        ) -> Union["PEPO", PEPS, np.ndarray]:
        """
        Action of the operator on the object `psi`.
        """

        if isinstance(psi, self.__class__):
            # TODO: implement this using lazy belief propagation; what I should
            # do here is what Gray is doing in Sci. Adv. 10, eadk4321 (2024)
            # (https://doi.org/10.1126/sciadv.adk4321)

            # returns newPEPO, where newPEPO = self @ psi

            # sanity checks
            if not graph_compatible(self.G, psi.G, sanity_check=True):
                raise ValueError("Graphs of PEPO and PEPS cannot be combined.")

            if not same_legs(self.G, psi.G):
                # permute lhs virtual dimensions s.t. they match the leg
                # ordering of the rhs
                self._permute_virtual_dimensions(psi.G)

            newPEPO = copy.deepcopy(self)

            # new sizes
            for node1, node2 in newPEPO.G.edges():
                rhs_edge_size = psi.G[node1][node2][0]["size"]
                newPEPO.G[node1][node2][0]["size"] *= rhs_edge_size

            # multiplying site tensors
            for node in self.G.nodes():
                N_neighbors = len(self.G.adj[node])

                # multiplying site tensors
                lhs_legs = (tuple(range(N_neighbors))
                            + (2*N_neighbors + 2, 2*N_neighbors + 1))
                rhs_legs = (tuple(range(N_neighbors, 2 * N_neighbors))
                            + (2*N_neighbors + 1, 2*N_neighbors))
                out_legs = tuple(
                    i + j*N_neighbors
                    for i, j in itertools.product(
                        range(N_neighbors),
                        (0,1)
                    )
                ) + (
                    2*N_neighbors + 2, 2*N_neighbors
                )

                T = np.einsum(
                    self[node], lhs_legs,
                    psi[node], rhs_legs,
                    out_legs
                )

                # preparing a re-shape
                newshape = ([None for _ in range(N_neighbors)]
                            + [newPEPO.D, newPEPO.D])
                for neighbor in newPEPO.G.adj[node]:
                    leg = newPEPO.G[node][neighbor][0]["legs"][node]
                    newshape[leg] = newPEPO.G[node][neighbor][0]["size"]

                # inserting the re-shaped tensor
                newPEPO[node] = np.reshape(T, newshape=newshape)

            return newPEPO

        if isinstance(psi, PEPS):
            # TODO: implement this using lazy belief propagation; what I should
            # do here is what Gray is doing in Sci. Adv. 10, eadk4321 (2024)
            # (https://doi.org/10.1126/sciadv.adk4321)

            # the action of self on psi is computed, and the new PEPS is
            # returned. It will inherit the leg ordering from psi.

            # sanity checks
            if not graph_compatible(self.G, psi.G, sanity_check=True):
                raise ValueError("Graphs of PEPO and PEPS cannot be combined.")

            if not same_legs(self.G, psi.G):
                # permute PEPO virtual dimensions s.t. they match the leg
                # ordering of the PEPS
                self._permute_virtual_dimensions(psi.G)

            newPEPS = copy.deepcopy(psi)

            # new sizes
            for node1,node2 in newPEPS.G.edges():
                rhs_edge_size = self.G[node1][node2][0]["size"]
                newPEPS.G[node1][node2][0]["size"] *= rhs_edge_size

            # multiplying site tensors
            for node in self.G.nodes():
                N_neighbors = len(self.G.adj[node])

                # multiplying site tensors
                op_legs = (tuple(range(N_neighbors))
                           + (2*N_neighbors + 1, 2*N_neighbors))
                ket_legs = tuple(range(N_neighbors, 2*N_neighbors + 1))
                out_legs = tuple(
                    i + j*N_neighbors
                    for i,j in itertools.product(
                        range(N_neighbors),
                        (0,1)
                    )
                ) + (
                    2*N_neighbors + 1,
                )
                T = np.einsum(
                    self[node], op_legs,
                    psi[node], ket_legs,
                    out_legs
                )

                # preparing a re-shape
                newshape = [None for _ in range(N_neighbors)] + [newPEPS.D,]
                for neighbor in newPEPS.G.adj[node]:
                    leg = newPEPS.G[node][neighbor][0]["legs"][node]
                    newshape[leg] = newPEPS.G[node][neighbor][0]["size"]

                # inserting the re-shaped tensor
                newPEPS[node] = np.reshape(T, newshape=newshape)

            return newPEPS

        if isinstance(psi, np.ndarray):
            # sanity check
            if not psi.ndim == 1: raise ValueError("psi must be a vector.")
            if not psi.shape[0] == self.D ** self.nsites:
                raise ValueError("".join((
                    "psi has the wrong number of ",
                    f"components. Expected {self.D ** self.nsites}, got ",
                    f"{psi.shape[0]}."
                )))

            # re-shaping. Order of sites will be determined by the order in
            # which self.G.nodes() iterates through the graph
            psi = np.reshape(
                psi,
                newshape=[self.D for _ in range(self.nsites)]
            )

            # enumerating the edges in the graph
            for i, nodes in enumerate(self.G.edges()):
                node1, node2 = nodes
                self.G[node1][node2][0]["label"] = i

            args = ()
            N_edges = self.G.number_of_edges()
            # assembling einsum arguments for the operator
            for i, nodeT in enumerate(self.G.nodes(data="T")):
                node, T = nodeT
                legs = ([None for _ in range(T.ndim - 2)]
                        + [N_edges + i, N_edges + self.nsites + i])
                for _, neighbor, edge_label in self.G.edges(
                    nbunch=node,
                    data="label"
                ):
                    legs[self.G[node][neighbor][0]["legs"][node]] = edge_label
                args += (T, tuple(legs),)

            # einsum arguments for the state
            args += (
                psi,
                tuple(
                    range(
                        N_edges + self.G.number_of_nodes(),
                        N_edges + 2 * self.G.number_of_nodes()
                    )
                )
            )

            out_legs = tuple(range(N_edges, N_edges + self.nsites))
            psi = np.einsum(*args, out_legs, optimize=True)
            psi = psi.flatten()

            return psi

        raise ValueError("".join((
            "PEPO.__matmul__ not implemented for type ",
            str(type(psi)),
            "."
        )))

    def __add__(lhs, rhs: "PEPO"):
        """
        Addition of two PEPOs. The bond dimension of the new operator is
        the sum of the two old bond dimensions.
        """
        # Notice that lhs == self !!! I chose this variable name to keep track
        # of what's going on.

        # sanity check
        if not nx.utils.nodes_equal(lhs.G.nodes(), rhs.G.nodes()):
            raise ValueError(
                "Operands have different geometries; nodes do not match."
            )
        if not nx.utils.edges_equal(lhs.G.edges(), rhs.G.edges()):
            raise ValueError(
                "Operands have different geometries; edges do not match."
            )
        if not lhs.D == rhs.D:
            raise ValueError(
                "Operands must have the same physical dimensions."
            )
        if not nx.utils.graphs_equal(lhs.tree, rhs.tree):
            warnings.warn(
                "".join((
                    "The trees of the operands are not the same. This is not ",
                    "a problem (as of 12th of March)."
                )),
                UserWarning
            )

        if not same_legs(lhs.G, rhs.G):
            # Permute dimensions of lhs to make both PEPOs compatible.
            lhs._permute_virtual_dimensions(rhs.G)

        res = PEPO(D=lhs.D)
        res.root = lhs.root
        res.tree = lhs.tree

        res.check_tree = False
        # TODO: This is unelegant, and could be (more or less) easily avoided;
        # see TODO in README.

        # Graph for the result with correct legs and sizes.
        res.G = PEPO.prepare_graph(lhs.G)

        # Saving new sizes in the edges.
        for node1, node2 in res.G.edges():
            res.G[node1][node2][0]["size"] = (lhs.G[node1][node2][0]["size"]
                                              + rhs.G[node1][node2][0]["size"])

        for node in res.G.nodes():
            shape = tuple(
                (lhs.G[node][neighbor][0]["size"]
                 + rhs.G[node][neighbor][0]["size"])
                for neighbor in res.G.adj[node]
            ) + (
                res.D, res.D
            )
            T = np.zeros(shape=shape, dtype=np.complex128)
            index_lhs = tuple(
                slice(0, lhs.G[node][neighbor][0]["size"])
                for neighbor in res.G.adj[node]
            ) + (
                slice(0, res.D), slice(0, res.D)
            )
            index_rhs = tuple(
                slice(
                    lhs.G[node][neighbor][0]["size"],
                    (lhs.G[node][neighbor][0]["size"]
                     +rhs.G[node][neighbor][0]["size"])
                )
                for neighbor in res.G.adj[node]
            ) + (
                slice(0, res.D), slice(0, res.D)
            )
            T[index_lhs] = lhs.G.nodes[node]["T"]
            T[index_rhs] = rhs.G.nodes[node]["T"]
            res.G.nodes[node]["T"] = T

        if not res.intact:
            raise RuntimeError("PEPO not intact.")

        return res

    def __repr__(self) -> str:
        return "".join((
            f"Operator on {self.nsites} sites with Hilbert space of size ",
            f"{self.D} at each. PEPO is ",
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
        return self.G.has_node(node)


class PauliPEPO(PEPO):
    """
    Tensor product operators on spin systems,
    composed of Pauli operators.
    """
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    """Pauli $X$-matrix."""
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    """Pauli $Y$-matrix."""
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    """Pauli $Z$-matrix."""

    @property
    def intact(self) -> bool:
        """
        Whether the PEPO is intact:
        * Checks `super().intact`.
        * Checks if the physical dimension is two.
        * Checks if the hamiltonian is composed of Pauli operators.
        """
        if not super().intact: return False

        if not self.D == 2:
            warnings.warn("Physical dimension unequal to two.", RuntimeWarning)
            return False

        # Hamiltonian composed of pauli operators?
        for node, T in self.G.nodes(data="T"):
            for virtual_index in itertools.product(*[
                range(T.shape[i])
                for i in range(T.ndim - 2)
            ]):
                index = virtual_index + (slice(0, self.D), slice(0, self.D))

                if np.allclose(T[index], 0): continue

                proportional_to_pauli = [
                    proportional(T[index], op, 10)
                    for op in (self.X, self.Y, self.Z, self.I)
                ]

                if not any(proportional_to_pauli):
                    warnings.warn(
                        "".join((
                            f"Unknown operator in index {virtual_index} at ",
                            f"node {node}."
                        )),
                        RuntimeWarning
                    )
                    return False

        return True

    def __init__(self) -> None:
        super().__init__(2)

        return



# -----------------------------------------------------------------------------
#                   Functions for PEPO.operator_chains
# -----------------------------------------------------------------------------


def get_next_edges(
        G: nx.MultiGraph,
        edge_set: Tuple[Tuple[int]],
        sanity_check: bool = False
    ) -> Tuple[Tuple[int]]:
    """
    Returns the edges that are adjacent to, but not contained in, the
    set `edge_set`.
    """
    if sanity_check:
        assert network_message_check(G)
        if not all(G.has_edge(*edge) for edge in edge_set):
            raise ValueError("Some edges are not contained in the graph G.")

    next_edges = ()

    interior = nx.Graph(incoming_graph_data=edge_set).nodes()

    for node1, node2 in G.edges(nbunch=interior):
        if (node1 not in interior) or (node2 not in interior):
            next_edges += ((node1, node2),)

    return next_edges


def edge_indices_to_site_index(
        G: nx.MultiGraph,
        node: int,
        edges_to_indices: Dict[FrozenSet[int], int],
        sanity_check: bool = False
    ) -> Tuple[int]:
    """
    Given a dictionary of edges to indices, returns the index to the
    local tensor at `node`.
    """
    if sanity_check:
        assert network_message_check(G)
        if not G.has_node(node):
            raise ValueError(f"Node {node} not contained in graph G.")

    index = [np.nan for _ in G.adj[node]]

    for neighbor in G.adj[node]:
        leg = G[node][neighbor][0]["legs"][node]
        index[leg] = edges_to_indices[frozenset((node, neighbor))]

    return tuple(index)


def all_edges_present(
        G: nx.MultiGraph,
        edges_to_indices: Dict[FrozenSet[int], int],
        node: int = None,
        sanity_check: bool = False
    ) -> bool:
    """
    Returns `True` if all edges adjacent to `node` have an associated
    index in `edges_to_indices`.
    """
    if sanity_check: assert network_message_check(G)

    if node is not None:
        # checking if al edges adjacent to node are present
        if sanity_check:
            if not G.has_node(node):
                raise ValueError(f"Node {node} not contained in graph G.")

        return all(
            frozenset((node, neighbor)) in edges_to_indices.keys()
            for neighbor in G.adj[node]
        )

    # checking if all edges are present
    return nx.utils.edges_equal(
        G.edges(),
        [tuple(edge) for edge in edges_to_indices.keys()]
    )


if __name__ == "__main__":
    pass
