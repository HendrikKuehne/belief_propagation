"""
Example hamiltonians as PEPOs.
"""

__all__ = [
    "TFI",
    "Heisenberg",
    "Zero",
    "Identity",
    "posneg_TFI",
    "operator_chain",
    "operator_layer"
]

import copy
import warnings
from typing import Union

import numpy as np
import networkx as nx
import tqdm

from belief_propagation.utils import multi_kron, op_layer_intact_check
from belief_propagation.PEPO import PEPO, PauliPEPO


class TFI(PauliPEPO):
    """
    Travsverse Field Ising model.
    """

    def __ising_PEPO_tensor_without_coupling(
            self,
            node: int,
            N_pas: int,
            N_out: int,
            g: float
        ) -> np.ndarray:
        """
        Returns a Transverse Field Ising PEPO tensor.

        `T` has the canonical leg ordering: The first leg is the
        incoming leg, afterwards follow the passive legs, and finally
        the outgoing legs.
        """
        # sanity check
        assert N_pas >= 0
        assert N_out >= 0
        assert node in self

        T = self.traversal_tensor(node=node, chi=3, N_pas=N_pas, N_out=N_out)

        # Transverse field.
        index = ((0,)
                 + tuple(-1 for _ in range(N_pas + N_out))
                 + (slice(2), slice(2)))
        T[index] = g * self.X

        return T

    def __add_ising_coupling(self, node1: int, node2: int, Jz: float) -> None:
        """
        Adds Ising-type coupling to the edge `(node1, node2)`. This
        means that both decay stages (of the finite state automaton) are
        added to this edge, NOT that an operator `Jz * sz * sz` is added
        to the Hamiltonian!
        """
        # Why is the construction of the PEPO this convoluted? Why do I not
        # assemble the tensors in `__ising_PEPO_tensor_without_coupling`,
        # re-shape them according to the tree structure, and insert them into
        # the PEPO? The code would be much more intelligible. The problem is
        # that I want only one ising coupling per edge. Since my graph might
        # have any structure, there's no way to know where to add coupling in a
        # graph-agnostic way. Put another way, I have to take the graph (and
        # thus the tree) into account to avoid adding double the coupling to
        # some edges. The edges that are affected by this are edges that are
        # not contained in the tree.

        if node1 in self.tree.succ[node2]:
            # node1 is downstream from node2.
            child = node1
            parent = node2
        else:
            # node2 is downstream from node1, or the edge is not contained in
            # the tree (in which case the order does not matter).
            child = node2
            parent = node1

        # First particle decay stage at this edge; at parent node.
        if parent != self.root:
            grandparent = tuple(_ for _ in self.tree.pred[parent].keys())[0]
            index = tuple(
                0 if i == self.G[grandparent][parent][0]["legs"][parent]
                else 1 if i == self.G[parent][child][0]["legs"][parent]
                else 2
                for i in range(self[parent].ndim - 2)
            ) + (slice(2), slice(2))
        else:
            # The incoming leg of the root node is a boundary leg, and thus
            # located just before the physical legs. This is why this case
            # distinction is necessary.
            index = tuple(
                1 if i == self.G[parent][child][0]["legs"][parent]
                else 2
                for i in range(self[parent].ndim - 3)
            ) + (0, slice(2), slice(2))
        self[parent][index] = Jz * self.Z

        # Second particle decay stage at this edge; at child node.
        index = tuple(
            1 if i == self.G[parent][child][0]["legs"][child]
            else 2
            for i in range(self[child].ndim - 2)
        ) + (slice(2), slice(2))
        self[child][index] = self.Z

        return

    def __init__(
            self,
            G: nx.MultiGraph,
            J: float = 1,
            g: float = 0,
            sanity_check: bool = False
        ) -> None:
        """
        Travsverse Field Ising model `J * sz * sz + g * sx` PEPO on
        graph `G`, with coupling `J` and external field `h`.

        Ordering of legs in the PEPO virtual dimensions is inherited
        from `G`. The last two dimensions of every PEPO tensor are the
        physical dimensions.
        """
        super().__init__()

        self.G = PEPO.prepare_graph(
            G=G,
            chi=3,
            D=2,
            sanity_check=sanity_check
        )

        # Saving coupling strength and transversal field.
        self.J = J
        self.g = g

        # Root node is node with smallest degree.
        self.root = sorted(G.nodes(), key=lambda x: len(G.adj[x]))[0]

        # Depth-first search tree.
        self.tree = nx.dfs_tree(G, self.root)

        # Adding PEPO tensors (without coupling).
        for node in self:
            N_in = len(self.tree.pred[node])
            N_out = len(self.tree.succ[node])
            N_pas = len(self.G.adj[node]) - N_out - N_in

            if N_out == 0:
                # Node is a leaf.
                N_out = 1

            # PEPO tensor, where the first dimension is the incoming leg, the
            # passive legs and the outgoing legs follow, and the last two
            # dimensions are the physical legs
            T = self.__ising_PEPO_tensor_without_coupling(
                node=node, N_pas=N_pas, N_out=N_out, g=g
            )

            if node == self.root:
                # Root node; we need to put the (incoming) boundary leg between
                # the virtual dimensions and the physical dimensions.
                T = np.moveaxis(T, 0, -3)

            # Re-shaping PEPO tensor to match the graph leg ordering
            T = self._canonical_to_correct_legs(T=T, node=node)

            self[node] = T

        # Adding incoming and outgoing coupling to every node but the root.
        for node1, node2 in self.G.edges():
            self.__add_ising_coupling(node1=node1, node2=node2, Jz=J)

        # Contracting boundary legs.
        self.contract_boundaries()

        if sanity_check: assert self.intact

        return

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------
    #                   dummy test cases
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def H1(J: float = 1,h: float = 0) -> np.ndarray:
        """
        Graph:

             4    5
             |    |
             |    |
        0 -- 1 -- 2 -- 3

        Notice that this is the geometry of
        `belief_propagation.old.networks.dummynet1`.
        """
        N = 6
        H = np.zeros(shape=(2**N, 2**N))
        I = np.eye(2)

        # Transverse field.
        for i in range(N):
            ops = tuple(PauliPEPO.X if _ == i else I for _ in range(N))
            H += h * multi_kron(*ops)

        # Two-body terms.
        for ops in (
            (I, I, I, I, PauliPEPO.Z, PauliPEPO.Z),
            (I, I, I, PauliPEPO.Z, PauliPEPO.Z, I),
            (I, I, PauliPEPO.Z, PauliPEPO.Z, I, I),
            (I, PauliPEPO.Z, I, I, PauliPEPO.Z, I),
            (PauliPEPO.Z, I, I, PauliPEPO.Z, I, I),
        ): H += J * multi_kron(*ops)

        return H

    @staticmethod
    def line(N: int, J: float = 1,h: float = 0) -> np.ndarray:
        """TFI mddel in one dimension, on `N` spins."""
        if N == 1: return h * PauliPEPO.X

        H = np.zeros(shape=(2**N, 2**N))
        I = np.eye(2)

        # Coupling terms.
        for i in range(N-1):
            ops = tuple(PauliPEPO.Z if _ in (i, i+1) else I for _ in range(N))
            H += J * multi_kron(*ops)
        # Transverse field.
        for i in range(N):
            ops = tuple(PauliPEPO.X if _ == i else I for _ in range(N))
            H += h * multi_kron(*ops)

        return H


class Heisenberg(PauliPEPO):
    """
    Heisenberg model with transverse field in x.
    """

    def __heisenberg_PEPO_tensor_without_coupling(
            self,
            node: int,
            N_pas: int,
            N_out: int,
            g: float
        ) -> np.ndarray:
        """
        Returns a Heisenberg PEPO tensor, that is missing the incoming
        and outgoing coupling.

        `T` has the canonical leg ordering: The first leg is the
        incoming leg, afterwards follow the passive legs, and finally
        the outgoing legs.
        """
        # sanity check
        assert N_pas >= 0
        assert N_out >= 0
        assert node in self

        T = self.traversal_tensor(node=node, chi=5, N_pas=N_pas, N_out=N_out)

        # Transverse field.
        index = ((0,)
                 + tuple(-1 for _ in range(N_pas + N_out))
                 + (slice(2), slice(2)))
        T[index] = g * self.X

        return T

    def __add_heisenberg_coupling(
            self,
            node1: int,
            node2: int,
            Jx: float,
            Jy: float,
            Jz: float
        ) -> None:
        """
        Adds Heisenberg-type coupling to the edge `(node1,node2)`. This
        means that both decay stages (of the finite state automaton) are
        added to this edge, NOT that an operator `Jz * sz * sz` is added
        to the Hamiltonian!
        """
        if node1 in self.tree.succ[node2]:
            # node1 is downstream from node2
            child = node1
            parent = node2
        else:
            # node2 is downstream from node1, or the edge is not contained in
            # the tree (in which case the order does not matter).
            child = node2
            parent = node1

        # First particle decay stage at this edge; at parent node.
        if parent != self.root:
            grandparent = tuple(_ for _ in self.tree.pred[parent].keys())[0]
            x_index = tuple(
                0 if i == self.G[grandparent][parent][0]["legs"][parent]
                else 1 if i == self.G[parent][child][0]["legs"][parent]
                else 4
                for i in range(self.G.nodes[parent]["T"].ndim - 2)
            ) + (slice(2), slice(2))
            y_index = tuple(
                0 if i == self.G[grandparent][parent][0]["legs"][parent]
                else 2 if i == self.G[parent][child][0]["legs"][parent]
                else 4
                for i in range(self.G.nodes[parent]["T"].ndim - 2)
            ) + (slice(2), slice(2))
            z_index = tuple(
                0 if i == self.G[grandparent][parent][0]["legs"][parent]
                else 3 if i == self.G[parent][child][0]["legs"][parent]
                else 4
                for i in range(self.G.nodes[parent]["T"].ndim - 2)
            ) + (slice(2), slice(2))
        else:
            x_index = tuple(
                1 if i == self.G[parent][child][0]["legs"][parent]
                else 4
                for i in range(self.G.nodes[parent]["T"].ndim - 3)
            ) + (0, slice(2), slice(2))
            y_index = tuple(
                2 if i == self.G[parent][child][0]["legs"][parent]
                else 4
                for i in range(self.G.nodes[parent]["T"].ndim - 3)
            ) + (0, slice(2), slice(2))
            z_index = tuple(
                3 if i == self.G[parent][child][0]["legs"][parent]
                else 4
                for i in range(self.G.nodes[parent]["T"].ndim - 3)
            ) + (0, slice(2), slice(2))
        self.G.nodes[parent]["T"][x_index] = Jx * self.X
        self.G.nodes[parent]["T"][y_index] = Jy * self.Y
        self.G.nodes[parent]["T"][z_index] = Jz * self.Z

        # Second particle decay stage at this edge; at child node.
        x_index = tuple(
            1 if i == self.G[parent][child][0]["legs"][child]
            else 4
            for i in range(self.G.nodes[child]["T"].ndim - 2)
        ) + (slice(2), slice(2))
        y_index = tuple(
            2 if i == self.G[parent][child][0]["legs"][child]
            else 4
            for i in range(self.G.nodes[child]["T"].ndim - 2)
        ) + (slice(2), slice(2))
        z_index = tuple(
            3 if i == self.G[parent][child][0]["legs"][child]
            else 4
            for i in range(self.G.nodes[child]["T"].ndim - 2)
        ) + (slice(2), slice(2))
        self.G.nodes[child]["T"][x_index] = self.X
        self.G.nodes[child]["T"][y_index] = self.Y
        self.G.nodes[child]["T"][z_index] = self.Z

        return

    def __init__(
            self,
            G: nx.MultiGraph,
            Jx: float = 1,
            Jy: float = 1,
            Jz: float = 1,
            g: float = 0,
            sanity_check: bool = False
        ):
        """
        Travsverse Field Ising model PEPO on graph `G`, with couplings
        `Jx`, `Jy`, `Jz`, and external field `g`.

        Ordering of legs in the PEPO virtual dimensions is inherited
        from `G`. The last two dimensions of every PEPO tensor are the
        physical dimensions.
        """
        super().__init__()

        self.G = PEPO.prepare_graph(G=G, chi=5, D=2, sanity_check=sanity_check)

        # Saving coupling strength and transversal field.
        self.Jx = Jx
        self.Jy = Jy
        self.Jz = Jz
        self.g = g

        # Root node is node with smallest degree.
        self.root = sorted(G.nodes(), key=lambda x: len(G.adj[x]))[0]

        # Depth-first search tree.
        self.tree = nx.dfs_tree(G, self.root)

        # Adding PEPO tensors (without coupling).
        for node in self:
            N_in = len(self.tree.pred[node])
            N_out = len(self.tree.succ[node])
            N_pas = len(self.G.adj[node]) - N_out - N_in

            if N_out == 0:
                # Node is a leaf.
                N_out = 1

            # PEPO tensor, where the first dimension is the incoming leg, the
            # passive legs and the outgoing legs follow, and the last two
            # dimensions are the physical legs.
            T = self.__heisenberg_PEPO_tensor_without_coupling(
                node=node, N_pas=N_pas, N_out=N_out, g=g
            )

            if node == self.root:
                # Root node; we need to put the (incoming) boundary leg at the
                # last place within the virtual dimensions.
                T = np.moveaxis(T, 0, -3)

            # Re-shaping PEPO tensor to match the graph leg ordering.
            T = self._canonical_to_correct_legs(T=T, node=node)

            self.G.nodes[node]["T"] = T

        # Adding incoming and outgoing coupling to every node but the root.
        for node1,node2 in self.G.edges():
            self.__add_heisenberg_coupling(
                node1=node1,
                node2=node2,
                Jx=Jx,
                Jy=Jy,
                Jz=Jz
            )

        # Contracting boundary legs.
        self.contract_boundaries()

        if sanity_check: assert self.intact

        return


def Zero(
        G: nx.MultiGraph,
        D: Union[int, dict[int, int]],
        dtype=np.complex128,
        sanity_check: bool = False
    ) -> PEPO:
    """
    Returns a zero-valued PEPO on graph `G`. Physical dimension `D`.
    If `D` is a dict, it must contain the physical dimension for every
    site in `G`.
    """
    # sanity check.
    if isinstance(D, dict):
        if not nx.utils.nodes_equal(nodes1=G.nodes(), nodes2=D.keys()):
            raise ValueError(
                "D must define the physical dimension on every site of G."
            )

    op = PEPO()
    op.G = PEPO.prepare_graph(G=G, chi=1, D=D)

    # Root node is node with smallest degree.
    op.root = sorted(G.nodes(), key=lambda x: len(G.adj[x]))[0]

    # Depth-first search tree.
    op.tree = nx.dfs_tree(G, op.root)

    # Since the PEPO contains only zeros, the tree traversal checks are not
    # applicable.
    op.check_tree = False

    # Adding local operators.
    for node in op:
        op[node] = np.zeros(
            shape = (tuple(1 for _ in range(len(G.adj[node])))
                     + (op.D[node], op.D[node])),
            dtype=dtype
        )

    if sanity_check: assert op.intact

    return op


def Identity(
        G: nx.MultiGraph,
        D: Union[int, dict[int, int]],
        dtype=np.complex128,
        sanity_check: bool = False
    ) -> PEPO:
    """
    Returns the identity PEPO on graph `G`. Physical dimension `D`.
    If `D` is a dict, it must contain the physical dimension for every
    site in `G`.
    """
    Id = Zero(G=G, D=D, dtype=dtype, sanity_check=sanity_check)

    # Adding local identities.
    for node in Id:
        Id[node][...,:,:] = Id.I(node=node)

    if nx.is_tree(G):
        # Enabling tree traversal checks.
        Id.check_tree = True

    if sanity_check: assert Id.intact

    return Id


def posneg_TFI(
        G: nx.MultiGraph,
        J: float = 1,
        g: float = 0,
        sanity_check: bool = False
    ) -> tuple[PEPO, PEPO]:
    """
    Constructs two PEPOs, where one contains the positive-semidefinite
    part of the TFI and the other contains the negative-semidefinite
    part.
    """
    # Spectral decompositions of X and Z.
    X_pos = np.ones(shape=(2, 2)) / 2
    X_neg = np.array([[-1, 1], [1, -1]]) / 2
    Z_pos = np.array([[1, 0], [0, 0]])
    Z_neg = np.array([[0, 0], [0, -1]])

    pos_op = PEPO()
    neg_op = PEPO()
    # Why not PauliPEPO? Because pos_op and neg_op will contain operators that
    # are not pauli matrices (e.g. projectors), so the sanity check of
    # PauliPEPO would not work.

    chi = 4
    """
    Virtual bond dimension. 0 is the moving particle state, 1 & 2 are
    decay states, and 3 is the vacuum state.
    """

    G = PEPO.prepare_graph(G=G, chi=chi, D=2, sanity_check=sanity_check)
    pos_op.G = copy.deepcopy(G)
    neg_op.G = copy.deepcopy(G)
    pos_op.J = J
    pos_op.g = g
    neg_op.J = J
    neg_op.g = g

    # Root node is node with smallest degree.
    root = sorted(G.nodes(), key=lambda x: len(G.adj[x]))[0]
    pos_op.root = root
    neg_op.root = root

    # Depth-first search trees.
    tree = nx.dfs_tree(G, root)
    pos_op.tree = copy.deepcopy(tree)
    neg_op.tree = copy.deepcopy(tree)

    # Adding PEPO tensors (without coupling).
    for node in G.nodes():
        N_in = len(tree.pred[node])
        N_out = len(tree.succ[node])
        N_pas = len(G.adj[node]) - N_out - N_in

        if N_out == 0:
            # Node is a leaf.
            N_out = 1

        # PEPO tensor, where the first dimension is the incoming leg, the
        # passive legs and the outgoing legs follow, and the last two
        # dimensions are the physical legs.
        pos_T = pos_op.traversal_tensor(
            node=node, chi=chi, N_pas=N_pas, N_out=N_out
        )
        neg_T = neg_op.traversal_tensor(
            node=node, chi=chi, N_pas=N_pas, N_out=N_out
        )

        # Transverse field.
        index = ((0,)
                 + tuple(-1 for _ in range(N_pas + N_out))
                 + (slice(2), slice(2)))
        pos_T[index] = g * X_pos
        neg_T[index] = g * X_neg

        if node == root:
            # Root node; we need to put the (incoming) boundary leg between the
            # virtual dimensions and the physical dimensions.
            pos_T = np.moveaxis(pos_T, 0, -3)
            neg_T = np.moveaxis(neg_T, 0, -3)

        # Re-shaping PEPO tensor to match the graph leg ordering.
        pos_op[node] = pos_op._canonical_to_correct_legs(T=pos_T, node=node)
        neg_op[node] = neg_op._canonical_to_correct_legs(T=neg_T, node=node)

    # Adding incoming and outgoing coupling to every node but the root.
    for node1, node2 in G.edges():
        if node1 in tree.succ[node2]:
            # Node1 is downstream from node2.
            child = node1
            parent = node2
        else:
            # Node2 is downstream from node1, or the edge is not contained in
            # the tree (in which case the order does not matter).
            child = node2
            parent = node1

        # First particle decay stages at this edge; at parent node.
        if parent != root:
            grandparent = tuple(_ for _ in tree.pred[parent].keys())[0]
            index = lambda x: tuple(
                0 if i == G[grandparent][parent][0]["legs"][parent]
                else x if i == G[parent][child][0]["legs"][parent]
                else -1
                for i in range(pos_op[parent].ndim - 2)
            ) + (slice(2), slice(2))
        else:
            # The incoming leg of the root node is a boundary leg, and thus
            # located just before the physical legs. This is why this case
            # distinction is necessary.
            index = lambda x: tuple(
                x if i == G[parent][child][0]["legs"][parent]
                else -1
                for i in range(pos_op[parent].ndim - 3)
            ) + (0, slice(2), slice(2))
        pos_op[parent][index(1)] = J * Z_pos
        pos_op[parent][index(2)] = J * Z_neg
        neg_op[parent][index(1)] = J * Z_pos
        neg_op[parent][index(2)] = J * Z_neg

        # Final particle decay stage at this edge; at child node.
        index = lambda x: tuple(
            x if i == G[parent][child][0]["legs"][child]
            else -1
            for i in range(pos_op[child].ndim - 2)
        ) + (slice(0,2),slice(0,2))
        pos_op[child][index(1)] = Z_pos
        pos_op[child][index(2)] = Z_neg
        neg_op[child][index(1)] = Z_neg
        neg_op[child][index(2)] = Z_pos

    # Contracting boundary legs.
    pos_op.contract_boundaries()
    neg_op.contract_boundaries()

    if sanity_check: assert pos_op.intact and neg_op.intact

    return pos_op,neg_op


def operator_chain(
        G: nx.MultiGraph,
        ops: dict[int, np.ndarray],
        sanity_check: bool = False
    ) -> PEPO:
    """
    Product of single-site operators. The dictionary `ops` contains the
    operators as values, and the sites on which they act as keys.
    """
    if ops == {}:
        raise ValueError(
            "operator_chain() received empty operator chain."
        )

    # Extracting physical dimension.
    D = {node: op.shape[0] for node, op in ops.items()}

    if not nx.utils.nodes_equal(nodes1=G.nodes(), nodes2=D.keys()):
        # Physical dimension is not defined on all sites. Trying to infer
        # physical dimension.
        phys_dim_set = set(op.shape[-1] for op in ops.values())
        if len(phys_dim_set) == 1:
            # There is a unique physical dimension; adding it to the remaining
            # sites.
            D_inferred = phys_dim_set.pop()
            for node in G:
                if node not in D.keys(): D[node] = D_inferred
        else:
            raise ValueError("".join((
                "Physical dimension cannot be inferred sites. If ops ",
                "contains operators with different dimensions, ops must ",
                "contain an operator for every site."
            )))

    H = Identity(G=G, D=D, sanity_check=sanity_check)

    for node, op in ops.items():
        # sanity check
        if not G.has_node(node):
            raise ValueError(f"Node {node} not contained in graph.")
        if not op.shape == (D[node], D[node]):
            raise ValueError("".join((
                f"Operator on node {node} has wrong shape: Expected ",
                f"({D[node]}, {D[node]}), got " + str(op.shape) + "."
            )))

        index = (tuple(0 for _ in H.G.adj[node])
                 + (slice(D[node]), slice(D[node])))
        H[node][index] = op

    # Since we inserted non-identity operators, the tree traversal checks are
    # not applicable.
    H.check_tree = False

    if sanity_check: assert H.intact

    return H


def operator_layer(
        G: nx.MultiGraph,
        op_chains: tuple[dict[int, np.ndarray]],
        sanity_check: bool = False
    ) -> PEPO:
    """
    A PEPO that contains disjoint operator chains.
    """
    with tqdm.tqdm.external_write_mode():
        warnings.warn(
            "".join((
                "So far, this method simply adds operator chains. There is a ",
                "more elegant method: if the operator chains are disjoint, ",
                "they can be compressed into a PEPO with smaller bond ",
                "dimension. This has yet to be implemented."
            )),
            FutureWarning
        )

    # sanity check
    assert op_layer_intact_check(
        G=G,
        op_chains=op_chains,
        test_same_length=False,
        test_disjoint=True
    )

    op = operator_chain(G=G, ops=op_chains[0], sanity_check=sanity_check)

    for ops in op_chains[1:]:
        op = op + operator_chain(
            G=G, ops=ops, sanity_check=sanity_check
        )

    if sanity_check: assert op.intact

    return op


if __name__ == "__main__":
    pass
