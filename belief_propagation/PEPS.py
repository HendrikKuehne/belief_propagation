"""
Projector-entangled pair states on arbitrary graphs.
"""

__all__ = ["PEPS",]

import warnings
import copy
from typing import Union, Iterator

import numpy as np
import networkx as nx
import cotengra as ctg
import tqdm

from belief_propagation.utils import (
    network_message_check,
    crandn,
    write_exp_bonddim_to_graph,
    multi_tensor_rank,
    graph_compatible,
    same_legs
)

class PEPS:
    """
    Base class for matrix-product states with arbitrary geometry.
    """

    def toarray(self, sanity_check: bool = False) -> np.ndarray:
        """
        Contracts the MPS using `ctg.einsum`.
        """
        if sanity_check: assert self.intact

        if self.nsites == 1:
            # the network is trivial
            return tuple(self.G.nodes(data="T"))[0][1]

        N = self.G.number_of_edges()
        args = ()

        # Enumerating the edges in the graph.
        for i, nodes in enumerate(self.G.edges()):
            node1, node2 = nodes
            self.G[node1][node2][0]["label"] = i

        # Extracting the einsum arguments.
        for node in sorted(self):
            T = self[node]
            args += (T,)

            # The first legs are thy virtual bond dimensions, the last leg is
            # the physical dimension.
            legs = [None for i in range(T.ndim-1)] + [N,]

            for _, neighbor, edge_label in self.G.edges(
                nbunch=node,
                data="label"
            ):
                legs[self.G[node][neighbor][0]["legs"][node]] = edge_label
            args += (tuple(legs),)
            N += 1

        psi = ctg.einsum(*args, optimize="greedy")
        return psi.flatten()

    def conj(self, sanity_check: bool = False):
        """
        bra to this state's ket: All site tensors are conjugated.
        """
        if sanity_check: assert self.intact

        newG = copy.deepcopy(self.G)

        for node, T in newG.nodes(data="T"): newG.nodes[node]["T"] = T.conj()

        return type(self)(G=newG, sanity_check=sanity_check)

    def legs_dict(self, node: int, sanity_check: bool = False) -> dict:
        """
        Returns the `legs` attributes of the adjacent edges of `node`
        in a dictionary structure: `legs_dict[neighbor]` is the
        same as `self.G[node][neighbor][0]["legs"]`.
        """
        if sanity_check: assert self.intact

        val = dict()

        for neighbor in self.G.adj[node]:
            val[neighbor] = self.G[node][neighbor][0]["legs"]

        return val

    def enlarge_bond_dimensions(
            self,
            size: Union[int, nx.MultiGraph],
            sanity_check: bool = False
        ) -> "PEPS":
        """
        Enlarges the virtual bond dimensions of the state. They new bond
        dimensions can be given as integer `size`, which applies to all
        bonds, or as a graph that contains the new bond dimension with
        the key `size` on every edge. If the old size of any edge
        exceeds the new size, nothing is done on this edge.
        """
        if sanity_check: assert self.intact

        # Preparing new bond dimensions.
        if not isinstance(size, nx.MultiGraph):
            size_graph = nx.MultiGraph(incoming_graph_data=self.G)
            for node1, node2 in size_graph.edges():
                size_graph[node1][node2][0]["size"] = size
        else:
            size_graph = size
        if not graph_compatible(self.G, size_graph, sanity_check=sanity_check):
            raise ValueError("".join((
                "Graph with new bond dimensions is not compatible with ",
                "geometry of the state."
            )))

        new_peps = copy.deepcopy(self)

        # Adding new bond dimensions.
        for node1, node2 in new_peps.G.edges():
            newsize = size_graph[node1][node2][0]["size"]
            new_peps.G[node1][node2][0]["size"] = newsize

        for node in new_peps:
            # Assembling numpy pad widths.
            pad_width = [None for _ in new_peps.G.adj[node]] + [(0, 0,),]
            for neighbor in new_peps.G.adj[node]:
                leg = new_peps.G[node][neighbor][0]["legs"][node]
                old_size = self.G[node][neighbor][0]["size"]
                pad_width[leg] = (
                    0,
                    max(size_graph[node][neighbor][0]["size"] - old_size, 0)
                )

            # Padding tensor, and inserting into new PEPS.
            new_peps[node] = np.pad(
                self[node],
                pad_width=pad_width,
                mode="constant",
                constant_values=0
            )

        if sanity_check: assert new_peps.intact

        return new_peps

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
            axes = [None for _ in range(N_neighbors)] + [N_neighbors,]

            for neighbor in self.G.adj[node]:
                leg = G[node][neighbor][0]["legs"][node]
                axes[leg] = self.G[node][neighbor][0]["legs"][node]

            self[node] = np.transpose(self[node], axes=axes)

        # updating leg orderings
        for node1, node2 in self.G.edges():
            self.G[node1][node2][0]["legs"] = G[node1][node2][0]["legs"]

        if sanity_check: assert self.intact

        return

    @property
    def nsites(self) -> int:
        """
        Number of sites on which the state is defined.
        """
        return self.G.number_of_nodes()

    @property
    def intact(self) -> bool:
        """
        Checks if the PEPS is intact:
        * Is the underlying network message-ready?
        * Is the size of every edge saved?
        * Are the physical legs the last dimension in each tensor?
        * Do the physical legs have the correct sizes?
        """
        # Is the underlying network message-ready?
        if not network_message_check(self.G):
            with tqdm.tqdm.external_write_mode():
                warnings.warn("Network not intact.", RuntimeWarning)
                return False

        # Size attribute given on every edge?
        for node1, node2, data in self.G.edges(data=True):
            if not "size" in data.keys():
                with tqdm.tqdm.external_write_mode():
                    warnings.warn(
                        f"No size saved in edge ({node1},{node2}).",
                        RuntimeWarning
                    )
                return False

            if data["size"] == 0:
                with tqdm.tqdm.external_write_mode():
                    warnings.warn(
                        f"Edge ({node1},{node2}) has size 0.",
                        RuntimeWarning
                    )
                return False

            if (
                data["size"] != self[node1].shape[data["legs"][node1]]
                or data["size"] != self[node2].shape[data["legs"][node2]]
            ):
                with tqdm.tqdm.external_write_mode():
                    warnings.warn(
                        f"Wrong size saved in edge ({node1},{node2}).",
                        RuntimeWarning
                    )
                return False

        # Are the physical legs the last dimension in each tensor? Do the
        # tensors have the correct physical dimensions?
        for node, data in self.G.nodes(data=True):
            if not "T" in data.keys():
                raise ValueError(f"No tensor in node {node}.")
            if not "D" in data.keys():
                raise ValueError(f"No physical dimension in node {node}.")

            T = data["T"]
            D = data["D"]

            legs = [leg for leg in range(T.ndim)]
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
                    with tqdm.tqdm.external_write_mode():
                        warnings.warn(
                            f"Wrong leg in edge ({node1},{node2},{key}).",
                            RuntimeWarning
                        )
                    return False

            if not legs == [T.ndim - 1,]:
                with tqdm.tqdm.external_write_mode():
                    warnings.warn(
                        f"Physical leg is not the last dimension in node {node}.",
                        RuntimeWarning
                    )
                return False

            # Correct size of physical leg?
            if not T.shape[-1] == D:
                with tqdm.tqdm.external_write_mode():
                    warnings.warn(
                        "".join((
                            f"Physical dimension mismatch at node {node}. ",
                            f"Expected {D}, tensor has {T.shape[-1]}."
                        )),
                        RuntimeWarning
                    )
                return False

        return True

    @property
    def D(self) -> dict[int, int]:
        """Physical dimension at every node."""
        return {
            node: self.G.nodes[node]["D"]
            for node in self
        }

    @classmethod
    def init_random(
            cls,
            G: nx.MultiGraph,
            D: Union[int, dict[int, int]],
            chi: int,
            rng: np.random.Generator = np.random.default_rng(),
            real: bool = False,
            bond_dim_strategy: str = "uniform",
            keep_legs: bool = True,
            sanity_check: bool = False,
        ) -> "PEPS":
        """
        Initializes a MPS randomly. The virtual bond dimension is `chi`,
        the physical dimension is `D`. Leg ordering in `G` is included
        based on value of `keep_legs` (default is `True`). Bond
        dimensions are initialized using `bond_dim_strategy`; see
        `PEPS.set_bond_dimensions`.
        """
        # Random number generation.
        if real:
            randn = lambda size: rng.standard_normal(size)
        else:
            randn = lambda size: crandn(size, rng)

        G = cls.prepare_graph(G, D=D, keep_legs=keep_legs)

        # Determining bond dimensions.
        cls.set_bond_dimensions(
            G=G,
            bond_dim_strategy=bond_dim_strategy,
            D=D,
            max_chi=chi
        )

        for node in G.nodes:
            # Constructing the shape of the tensor at this site.
            dim = [None for i in G.adj[node]] + [G.nodes[node]["D"],]
            for i, neighbor in enumerate(G.adj[node]):
                leg = G[node][neighbor][0]["legs"][node]
                dim[leg] = G[node][neighbor][0]["size"]

            # Adding the tensor to this node.
            G.nodes[node]["T"] = randn(size = dim) / chi**(3/4)

        return cls(G, sanity_check=sanity_check)

    @classmethod
    def init_from_TN(
            cls,
            G: nx.MultiGraph,
            sanity_check: bool = False
        ) -> "PEPS":
        """
        Initialises a PEPS from a TN by appending dummy physical
        dimensions of size one to the site tensors. `G` needs to contain
        a tensor on every site, and the `legs` attribute on every edge.
        """
        newG = cls.prepare_graph(G, keep_legs=True, D=1)

        # appending a dummy physical dimension with size one to the tensors
        for node in G.nodes:
            newG.nodes[node]["T"] = np.expand_dims(G.nodes[node]["T"], axis=-1)

        # adding sizes to the edges
        for node1, node2 in G.edges():
            leg = G[node1][node2][0]["legs"][node1]
            size = newG.nodes[node1]["T"].shape[leg]
            newG[node1][node2][0]["size"] = size

        return cls(G=newG, sanity_check=sanity_check)

    @classmethod
    def Dummy(
            cls,
            G: nx.MultiGraph,
            dtype=np.complex128,
            sanity_check: bool = False
        ) -> "PEPS":
        """
        Returns a dummy PEPS on graph `G` with physical dimension one.
        """
        G = cls.prepare_graph(G=G, D=1)
        # adding tensors
        for node in G.nodes:
            G.nodes[node]["T"] = np.ones(
                shape = [1 for _ in range(len(G.adj[node]) + 1)],
                dtype=dtype
            )

        # adding sizes to edges
        for node1, node2 in G.edges(): G[node1][node2][0]["size"] = 1

        return cls(G=G, sanity_check=sanity_check)

    @classmethod
    def ProductState(
            cls,
            G: nx.MultiGraph,
            state: Union[np.ndarray, dict[int, np.ndarray]],
            normalize: bool = True,
            sanity_check: bool = False
        ) -> "PEPS":
        """
        Initialises a product state PEPO. If `state` is an array, it is
        broadcasted to all sites. If `state` is a dictionary, it is
        assumed to contain a pure state for every site. The PEPO is
        normalized to unity, if `normalize=True` (default).
        """
        if isinstance(state, np.ndarray):
            return cls.ProductState(
                G=G,
                state={node: state for node in G.nodes()},
                sanity_check=sanity_check
            )

        if isinstance(state, dict):
            # sanity check
            if not nx.utils.nodes_equal(state.keys(), G.nodes()):
                raise ValueError("".join((
                    "State and graph do not agree in the nodes that they ",
                    "contain."
                )))
            if any(state[node].ndim != 1 for node in G.nodes()):
                raise ValueError("All local states must be pure states.")

            psi = cls.Dummy(G=G, sanity_check=sanity_check)

            for node in psi:
                # Re-sizing the tensor from psi, s.t. the tensor from the state
                # dictionary can be inserted.
                old_shape = list(psi[node].shape)
                old_shape[-1] = state[node].shape[0]
                newT = np.resize(psi[node], old_shape)

                # Inserting the normalized state.
                norm = np.sqrt(np.dot(state[node].conj(), state[node]))
                newT[...,:] = state[node] / norm if normalize else state[node]
                psi[node] = newT

                # Saving physical dimension.
                psi.G.nodes[node]["D"] = state[node].shape[0]

            if sanity_check: assert psi.intact

            return psi

        raise NotImplementedError("".join((
            "PEPS.ProductState not implemented for state of type ",
            str(type(state)),
            "."
        )))

    @staticmethod
    def prepare_graph(
            G: nx.MultiGraph,
            keep_legs: bool = False,
            keep_size: bool = False,
            D: Union[int, dict[int, int]] = None,
            sanity_check: bool = False
        ) -> nx.MultiGraph:
        """
        Creates a shallow copy of G, and adds the keys `legs`, `trace`
        and `indices` to the edges. If `D` is given, adds physical
        dimensions to the nodes.

        To be used in `__init__` of subclasses of `PEPO`: `G` is the
        graph from which the operator inherits it's underlying graph.
        """
        # shallow copy of G
        newG = nx.MultiGraph(G.edges())

        if any(
            legs is None
            for node1, node2, legs in G.edges(data="legs", keys=False)
        ) and keep_legs:
            # The graph does not contain a leg ordering. Setting keep_legs to
            # False, s.t. a leg ordering is added.
            with tqdm.tqdm.external_write_mode():
                warnings.warn(
                    "Graph does not contain leg ordering. Setting keep_legs to False.",
                    RuntimeWarning
                )
            keep_legs = False

        # adding legs attribute to each edge
        for node1, node2, legs in G.edges(data="legs", keys=False):
            newG[node1][node2][0]["legs"] = legs if keep_legs else {}
            newG[node1][node2][0]["trace"] = False
            newG[node1][node2][0]["indices"] = None

            if "size" in G[node1][node2][0].keys() and keep_size:
                newG[node1][node2][0]["size"] = G[node1][node2][0]["size"]

        if not keep_legs:
            for node in newG.nodes:
                # adding to the adjacent edges which index they correspond to
                for i, neighbor in enumerate(newG.adj[node]):
                    newG[node][neighbor][0]["legs"][node] = i

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

        if sanity_check: assert network_message_check(newG)

        return newG

    @staticmethod
    def set_bond_dimensions(
            G: nx.MultiGraph,
            bond_dim_strategy: str,
            D: Union[int, dict[int, int]] = None,
            max_chi: int = None
        ) -> None:
        """
        Initializes the bond dimensions in the graph `G` in-place. The
        string `bond_dim_strategy` determines how bond dimensions are
        intialized. There are several options:
        * `None` (default): Bond dimension `D` on every edge.
        * `exp`: Exact solution on trees. Edge size grows exponentially
        with distance from leaf nodes (internally calls
        `belief_propagation.utils.write_exp_size_to_graph`). Requires
        physical dimension `D`.
        * `exp_cutoff`: Same as `exp`, with maximum size `max_chi`.

        `D` is the physical dimension, and `max_chi` is the bond
        dimension cutoff.
        """
        if bond_dim_strategy in ("exp_cutoff", "exp") and D is None:
            raise ValueError("".join((
                "Bond dimension strategy ",
                bond_dim_strategy,
                " requires physical dimension."
            )))

        if np.isscalar(D):
            D = {node: D for node in G}

        if bond_dim_strategy == "uniform":
            for node1, node2, key in G.edges(keys=True):
                G[node1][node2][key]["size"] = max_chi
            return

        if bond_dim_strategy == "exp":
            #if not np.isscalar(D):
            #    raise NotImplementedError("".join((
            #        "PEPS.set_bond_dimensions is not implemented for bond ",
            #        "dimensions that change between nodes."
            #    )))

            write_exp_bonddim_to_graph(G=G, D=D)
            return

        if bond_dim_strategy == "exp_cutoff":
            #if not np.isscalar(D):
            #    raise NotImplementedError("".join((
            #        "PEPS.set_bond_dimensions is not implemented for bond ",
            #        "dimensions that change between nodes."
            #    )))

            if max_chi is None:
                raise ValueError("".join((
                    "Bond dimension strategy ",
                    bond_dim_strategy,
                    " requires cutoff value."
                )))
            write_exp_bonddim_to_graph(G=G, D=D, max_chi=max_chi)

            return

        raise ValueError("".join((
            "Bond dimension strategy ",
            bond_dim_strategy,
            " not implemented."
        )))

    def __getitem__(self, node:int) -> np.ndarray:
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
        if not T.ndim == self.G.nodes[node]["T"].ndim:
            raise ValueError(
                "Attempting to set site tensor with wrong number of legs."
            )

        self.G.nodes[node]["T"] = T

        return

    def __mul__(self, x: float):
        """
        Multiplication of the whole PEPS with a scalar.
        """
        if not np.isscalar(x): raise ValueError("x must be a scalar.")
        newPEPS = copy.deepcopy(self)

        N = newPEPS.nsites
        for node in newPEPS: newPEPS[node] = newPEPS[node] * (x**(1/N))

        return newPEPS

    def __rmul__(self, x: float): return self.__mul__(x)

    def __repr__(
            self,
            edge_details: bool = False,
            node_details: bool = False
        ) -> str:
        out = "".join((
            f"State on {self.nsites} sites.",
            " PEPS is ",
            ("intact." if self.intact else "not intact.")))

        if edge_details:
            # bond dimensions and multilinear tensor ranks
            digits = int(np.log10(self.nsites))
            out = out.join(("\n  Bond dimensions:",))
            for node1, node2, size in self.G.edges(data="size"):
                out = out.join((
                    "\n    (" + str(node1).zfill(digits),
                    ",",
                    str(node2).zfill(digits) + f") : size = {size}"
                ))

        if node_details:
            # multininear tensor ranks
            out = out.join(("\n  Multilinear tensor ranks:",))
            for node in self.G.nodes():
                out = out.join((
                    "\n    ",
                    str(node).zfill(digits),
                    f" : {multi_tensor_rank(self[node])}"
                ))

        return out

    def __len__(self) -> int: return self.nsites

    def __iter__(self) -> Iterator[int]:
        """
        Iterator over the nodes in the graph `self.G`.
        """
        return iter(self.G.nodes(data=False))

    def __contains__(self, node: int) -> bool:
        """Does the graph `self.G` contain the node `node`?"""
        return self.G.has_node(node)

    def __eq__(self, rhs: "PEPS") -> bool:
        """
        Two PEPS are considered equal if they contain the same local
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
            if not lhs[node].shape == rhs[node].shape: return False

            if not np.allclose(lhs[node], rhs[node]): return False

        return True

    def __init__(self, G: nx.MultiGraph, sanity_check: bool = False) -> None:
        """
        Initialisation from a graph that contains site tensors.
        """
        # Inferring physical dimension.
        for node, T in G.nodes(data="T"):
            G.nodes[node]["D"] = T.shape[-1]

        # Inferring edge sizes.
        for node1, node2, data in G.edges(data=True):
            if "legs" not in data.keys():
                raise ValueError("".join((
                    f"Edge ({node1}, {node2}) in graph does not contain leg ",
                    "ordering."
                )))

            if "size" in data.keys(): continue

            leg1 = data["legs"][node1]
            leg2 = data["legs"][node2]
            size1 = G.nodes[node1]["T"].shape[leg1]
            size2 = G.nodes[node2]["T"].shape[leg2]

            if size1 != size2:
                raise ValueError("".join((
                    f"Tensor leg sizes on edge ({node1}, {node2}) do not ",
                    "match."
                )))

            G[node1][node2][0]["size"] = size1

        self.G: nx.MultiGraph = G
        """
        Graph that contains PEPO local tensors, leg ordering, virtual
        bond dimension sizes, and physical dimensions.
        """

        if sanity_check: assert self.intact

        return


if __name__ == "__main__":
    pass
