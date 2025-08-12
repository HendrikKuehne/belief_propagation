"""
Example hamiltonians as PEPOs.
"""

__all__ = [
    "TFI",
    "Heisenberg",
]

import copy

import numpy as np
import networkx as nx

from belief_propagation.utils import multi_kron
from belief_propagation.PEPO import PEPO, PauliPEPO


class TFI(PauliPEPO):
    """
    Travsverse Field Ising model: `J * sz * sz + g * sx`.
    """

    def __init__(
            self,
            G: nx.MultiGraph,
            J: float = 1,
            g: float = 0,
            dtype: np.dtype = np.complex128,
            sanity_check: bool = False
        ) -> None:
        """
        Travsverse Field Ising model `J * sz * sz + g * sx` PEPO on
        graph `G`, with coupling `J` and external field `h`.

        Ordering of legs in the PEPO virtual dimensions is inherited
        from `G`. The last two dimensions of every PEPO tensor are the
        physical dimensions.
        """
        super().__init__(dtype=dtype)

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
            T = self.traversal_tensor(
                node=node,
                chi=3,
                N_pas=N_pas,
                N_out=N_out,
                decay_op=g * self.X
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
            self.add_twosite_operator(
                node1=node1,
                node2=node2,
                ch_idx=1,
                first_decay_op=J * self.Z,
                second_decay_op=self.Z
            )

        # Contracting boundary legs.
        self.contract_boundaries()

        if sanity_check: assert self.intact

        return

    @staticmethod
    def posneg(
            G: nx.MultiGraph,
            J: float = 1,
            g: float = 0,
            dtype: np.dtype = np.complex128,
            sanity_check: bool = False
        ) -> tuple[PEPO, PEPO]:
        """
        Constructs two PEPOs, where one contains the positive-semidefinite
        part of the TFI and the other contains the negative-semidefinite
        part.
        """
        if any(c < 0 for c in (J, g)):
            raise NotImplementedError(
                "Negative Hamiltonian parameters are currently not supported."
            )

        # Spectral decompositions of X and Z.
        X_pos = np.ones(shape=(2, 2)) / 2
        X_neg = np.array([[-1, 1], [1, -1]]) / 2
        Z_pos = np.array([[1, 0], [0, 0]])
        Z_neg = np.array([[0, 0], [0, -1]])

        pos_op = PEPO(dtype=dtype)
        neg_op = PEPO(dtype=dtype)
        # Why not PauliPEPO? Because pos_op and neg_op will contain operators
        # that are not pauli matrices (e.g. projectors), so the sanity check of
        # PauliPEPO would not work.

        chi = 4
        """
        Virtual bond dimension. 0 is the moving particle state, 1 & 2 are
        decay states, and 3 is the vacuum state.
        """

        G = PEPO.prepare_graph(G=G, chi=chi, D=2, sanity_check=sanity_check)
        pos_op.G = copy.deepcopy(G)
        neg_op.G = copy.deepcopy(G)

        # Saving model parameters.
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

            # PEPO tensors, where the first dimension is the incoming leg, the
            # passive legs and the outgoing legs follow, and the last two
            # dimensions are the physical legs.
            pos_T = pos_op.traversal_tensor(
                node=node,
                chi=chi,
                N_pas=N_pas,
                N_out=N_out,
                decay_op=g * X_pos
            )
            neg_T = neg_op.traversal_tensor(
                node=node,
                chi=chi,
                N_pas=N_pas,
                N_out=N_out,
                decay_op=g * X_neg
            )

            if node == root:
                # Root node; we need to put the (incoming) boundary leg between
                # the virtual dimensions and the physical dimensions.
                pos_T = np.moveaxis(pos_T, 0, -3)
                neg_T = np.moveaxis(neg_T, 0, -3)

            # Re-shaping PEPO tensor to match the graph leg ordering.
            pos_op[node] = pos_op._canonical_to_correct_legs(
                T=pos_T, node=node
            )
            neg_op[node] = neg_op._canonical_to_correct_legs(
                T=neg_T, node=node
            )

        # Adding incoming and outgoing coupling to every node but the root.
        for node1, node2 in G.edges():
            pos_op.add_twosite_operator(
                node1=node1,
                node2=node2,
                ch_idx=1,
                first_decay_op=J * Z_pos,
                second_decay_op=Z_pos
            )
            pos_op.add_twosite_operator(
                node1=node1,
                node2=node2,
                ch_idx=2,
                first_decay_op=J * Z_neg,
                second_decay_op=Z_neg
            )
            neg_op.add_twosite_operator(
                node1=node1,
                node2=node2,
                ch_idx=1,
                first_decay_op=J * Z_pos,
                second_decay_op=Z_neg
            )
            neg_op.add_twosite_operator(
                node1=node1,
                node2=node2,
                ch_idx=2,
                first_decay_op=J * Z_neg,
                second_decay_op=Z_pos
            )

        # Contracting boundary legs.
        pos_op.contract_boundaries()
        neg_op.contract_boundaries()

        if sanity_check: assert pos_op.intact and neg_op.intact

        return pos_op, neg_op

    # -------------------------------------------------------------------------
    #                   dummy test cases
    # -------------------------------------------------------------------------

    @staticmethod
    def H1(J: float = 1, h: float = 0) -> np.ndarray:
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
    def line(N: int, J: float = 1, h: float = 0) -> np.ndarray:
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

    def __init__(
            self,
            G: nx.MultiGraph,
            Jx: float = 1,
            Jy: float = 1,
            Jz: float = 1,
            g: float = 0,
            dtype: np.dtype = np.complex128,
            sanity_check: bool = False
        ):
        """
        Travsverse Field Ising model PEPO on graph `G`, with couplings
        `Jx`, `Jy`, `Jz`, and external field `g`.

        Ordering of legs in the PEPO virtual dimensions is inherited
        from `G`. The last two dimensions of every PEPO tensor are the
        physical dimensions.
        """
        super().__init__(dtype=dtype)

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
            T = self.traversal_tensor(
                node=node, N_pas=N_pas, N_out=N_out, chi=5, decay_op=g * self.X
            )

            if node == self.root:
                # Root node; we need to put the (incoming) boundary leg at the
                # last place within the virtual dimensions.
                T = np.moveaxis(T, 0, -3)

            # Re-shaping PEPO tensor to match the graph leg ordering.
            T = self._canonical_to_correct_legs(T=T, node=node)

            self.G.nodes[node]["T"] = T

        # Adding incoming and outgoing coupling to every node but the root.
        for node1, node2 in self.G.edges():
            for J, ch_idx, op in zip(
                (Jx, Jy, Jz),
                range(1, 4),
                (self.X, self.Y, self.Z)
            ):
                self.add_twosite_operator(
                    node1=node1,
                    node2=node2,
                    ch_idx=ch_idx,
                    first_decay_op=J * op,
                    second_decay_op=op
                )

        # Contracting boundary legs.
        self.contract_boundaries()

        if sanity_check: assert self.intact

        return

    @staticmethod
    def posneg(
            G: nx.MultiGraph,
            Jx: float = 1,
            Jy: float = 1,
            Jz: float = 1,
            g: float = 0,
            dtype: np.dtype = np.complex128,
            sanity_check: bool = False
        ) -> tuple[PEPO, PEPO]:
        """
        Constructs two PEPOs, where one contains the
        positive-semidefinite part of the Heisenberg model and the other
        contains the negative-semidefinite part.
        """
        if any(c < 0 for c in (Jx, Jy, Jz, g)):
            raise NotImplementedError(
                "Negative Hamiltonian parameters are currently not supported."
            )

        # Spectral decompositions of X, Y and Z.
        X_pos = np.ones(shape=(2, 2)) / 2
        X_neg = np.array([[-1, 1], [1, -1]]) / 2
        Y_pos = np.array([[1, -1j], [1j, 1]]) / 2
        Y_neg = np.array([[-1, -1j], [1j, -1]]) / 2
        Z_pos = np.array([[1, 0], [0, 0]])
        Z_neg = np.array([[0, 0], [0, -1]])

        pos_op = PEPO(dtype=dtype)
        neg_op = PEPO(dtype=dtype)
        # Why not PauliPEPO? Because pos_op and neg_op will contain operators
        # that are not pauli matrices (e.g. projectors), so the sanity check of
        # PauliPEPO would not work.

        chi = 8
        """
        Virtual bond dimension. 0 is the moving particle state, 1 - 6 are
        decay states, and 7 is the vacuum state.
        """

        G = PEPO.prepare_graph(G=G, chi=chi, D=2, sanity_check=sanity_check)
        pos_op.G = copy.deepcopy(G)
        neg_op.G = copy.deepcopy(G)

        # Saving model parameters.
        pos_op.Jx = Jx
        pos_op.Jy = Jy
        pos_op.Jz = Jz
        pos_op.g = g
        neg_op.Jx = Jx
        neg_op.Jy = Jy
        neg_op.Jz = Jz
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

            # PEPO tensors, where the first dimension is the incoming leg, the
            # passive legs and the outgoing legs follow, and the last two
            # dimensions are the physical legs.
            pos_T = pos_op.traversal_tensor(
                node=node,
                chi=chi,
                N_pas=N_pas,
                N_out=N_out,
                decay_op=g * X_pos
            )
            neg_T = neg_op.traversal_tensor(
                node=node,
                chi=chi,
                N_pas=N_pas,
                N_out=N_out,
                decay_op=g * X_neg
            )

            if node == root:
                # Root node; we need to put the (incoming) boundary leg between
                # the virtual dimensions and the physical dimensions.
                pos_T = np.moveaxis(pos_T, 0, -3)
                neg_T = np.moveaxis(neg_T, 0, -3)

            # Re-shaping PEPO tensor to match the graph leg ordering.
            pos_op[node] = pos_op._canonical_to_correct_legs(
                T=pos_T, node=node
            )
            neg_op[node] = neg_op._canonical_to_correct_legs(
                T=neg_T, node=node
            )

        # Adding incoming and outgoing coupling to every node but the root.
        for node1, node2 in G.edges():

            # Coupling for the positive-semidefinite part:
            for J, ch_idx, first_decay_op, second_decay_op in zip(
                (Jx, Jx, Jy, Jy, Jz, Jz),
                range(1, 7),
                (X_pos, X_neg, Y_pos, Y_neg, Z_pos, Z_neg),
                (X_pos, X_neg, Y_pos, Y_neg, Z_pos, Z_neg),
            ):
                pos_op.add_twosite_operator(
                    node1=node1,
                    node2=node2,
                    ch_idx=ch_idx,
                    first_decay_op=J * first_decay_op,
                    second_decay_op=second_decay_op
                )

            # Coupling for the negative-semidefinite part:
            for J, ch_idx, first_decay_op, second_decay_op in zip(
                (Jx, Jx, Jy, Jy, Jz, Jz),
                range(1, 7),
                (X_pos, X_neg, Y_pos, Y_neg, Z_pos, Z_neg),
                (X_neg, X_pos, Y_neg, Y_pos, Z_neg, Z_pos),
            ):
                neg_op.add_twosite_operator(
                    node1=node1,
                    node2=node2,
                    ch_idx=ch_idx,
                    first_decay_op=J * first_decay_op,
                    second_decay_op=second_decay_op
                )

        # Contracting boundary legs.
        pos_op.contract_boundaries()
        neg_op.contract_boundaries()

        if sanity_check: assert pos_op.intact and neg_op.intact

        return pos_op, neg_op


if __name__ == "__main__":
    pass
