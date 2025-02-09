"""
Example hamiltonians as PEPOs.
"""

import numpy as np
import networkx as nx
import copy

from belief_propagation.utils import network_message_check,multi_kron,is_hermitian
from belief_propagation.PEPO import PEPO,PauliPEPO

class TFI(PauliPEPO):
    """
    Travsverse Field Ising model.
    """

    def __ising_PEPO_tensor_without_coupling(self,N_pas:int,N_out:int,g:float) -> np.ndarray:
        """
        Returns a Transverse Field Ising PEPO tensor.

        `T` has the canonical leg ordering: The first leg is the incoming leg, afterwards follow the
        passive legs, and finally the outgoing legs.
        """
        # sanity check
        assert N_pas >= 0
        assert N_out >= 0
        assert hasattr(self,"chi")

        T = self.traversal_tensor(N_pas=N_pas,N_out=N_out)

        # transverse field
        index = (0,) + tuple(-1 for _ in range(N_pas + N_out)) + (slice(0,2),slice(0,2))
        T[index] = g * self.X

        return T

    def __ising_coupling(self,node1:int,node2:int,Jz:float) -> None:
        """
        Adds Ising-type coupling to the edge `(node1,node2)`. This means that
        both decay stages (of the finite state automaton) are added to this
        edge, NOT that an operator `Jz * sz * sz` is added to the
        Hamiltonian!
        """
        # Why is the construction of the PEPO this convoluted? Why do I not
        # assemble the tensors in `__ising_PEPO_tensor_without_coupling`,
        # re-shape them according to the tree structure, and insert them
        # into the PEPO? The code would be much more intelligible. The
        # problem is that I want only one ising coupling per edge. Since
        # my graph might have any structure, there's no way to know where
        # to add coupling in a graph-agnostic way. Put another way, I have
        # to take the graph (and thus the tree) into account to avoid adding
        # double the coupling to some edges. The edges that are affected by
        # this are edges that are not contained in the tree.

        if node1 in self.tree.succ[node2]:
            # node1 is downstream from node2
            child = node1
            parent = node2
        else:
            # node2 is downstream from node1, or the edge is not contained in the tree (in which case the order does not matter)
            child = node2
            parent = node1

        # first particle decay stage at this edge; at parent node
        if parent != self.root:
            grandparent = tuple(_ for _ in self.tree.pred[parent].keys())[0]
            index = tuple(
                0 if i == self.G[grandparent][parent][0]["legs"][parent]
                else 1 if i == self.G[parent][child][0]["legs"][parent]
                else 2
                for i in range(self[parent].ndim - 2)
            ) + (slice(0,2),slice(0,2))
        else:
            # the incoming leg of the root node is a boundary leg, and thus
            # located just before the physical legs. This is why this case
            # distinction is necessary.
            index = tuple(
                1 if i == self.G[parent][child][0]["legs"][parent]
                else 2
                for i in range(self[parent].ndim - 3)
            ) + (0,slice(0,2),slice(0,2))
        self[parent][index] = Jz * self.Z

        # second particle decay stage at this edge; at child node
        index = tuple(
            1 if i == self.G[parent][child][0]["legs"][child]
            else 2
            for i in range(self[child].ndim - 2)
        ) + (slice(0,2),slice(0,2))
        self[child][index] = self.Z

        return

    def __init__(self,G:nx.MultiGraph,J:float=1,g:float=0,sanity_check:bool=False) -> None:
        """
        Travsverse Field Ising model `J * sz * sz + g * sx` PEPO on graph `G`, with coupling `J` and external field `h`.

        Ordering of legs in the PEPO virtual dimensions is inherited from `G`. The last two dimensions of every PEPO tensor are the physical dimensions.
        """
        # sanity check
        if sanity_check: assert network_message_check(G)

        super().__init__()

        self.chi = 3
        """
        Virtual bond dimension.
        0 is the moving particle state, 1 the decay state, and 2 is the vacuum state.
        """

        self.G = super().prepare_graph(G)

        # root node is node with smallest degree
        self.root = sorted(G.nodes(),key=lambda x: len(G.adj[x]))[0]

        # depth-first search tree
        self.tree = nx.dfs_tree(G,self.root)

        # adding PEPO tensors (without coupling)
        for node in self.G.nodes():
            N_in = len(self.tree.pred[node])
            N_out = len(self.tree.succ[node])
            N_pas = len(self.G.adj[node]) - N_out - N_in

            if N_out == 0:
                # node is a leaf
                N_out = 1

            # PEPO tensor, where the first dimension is the incoming leg, the passive legs
            # and the outgoing legs follow, and the last two dimensions are the physical legs
            T = self.__ising_PEPO_tensor_without_coupling(N_pas=N_pas,N_out=N_out,g=g)

            if node == self.root:
                # root node; we need to put the (incoming) boundary leg between the virtual dimensions and the physical dimensions
                T = np.moveaxis(T,0,-3)

            # re-shaping PEPO tensor to match the graph leg ordering
            T = self.permute_PEPO(T,node)

            self[node] = T

        # adding incoming and outgoing coupling to every node but the root
        for node1,node2 in self.G.edges():
            self.__ising_coupling(node1,node2,Jz=J)

        # contracting boundary legs
        self.contract_boundaries()

        if sanity_check: assert self.intact

        return

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------
    #                   dummy test cases
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def H1(J:float=1,h:float=0) -> np.ndarray:
        """
        Graph:

             4    5
             |    |
             |    |
        0 -- 1 -- 2 -- 3

        Notice that this is the geometry of `belief_propagation.networks.dummynet1`.
        """
        N = 6
        H = np.zeros(shape=(2**N,2**N))
        I = np.eye(2)

        # transverse field
        for i in range(N):
            ops = tuple(PauliPEPO.X if _ == i else I for _ in range(N))
            H += h * multi_kron(*ops)

        # two-body terms
        for ops in (
            (I,I,I,I,PauliPEPO.Z,PauliPEPO.Z),
            (I,I,I,PauliPEPO.Z,PauliPEPO.Z,I),
            (I,I,PauliPEPO.Z,PauliPEPO.Z,I,I),
            (I,PauliPEPO.Z,I,I,PauliPEPO.Z,I),
            (PauliPEPO.Z,I,I,PauliPEPO.Z,I,I),
        ): H += J * multi_kron(*ops)

        return H

    @staticmethod
    def line(N:int,J:float=1,h:float=0) -> np.ndarray:
        """TFI mddel in one dimension, on `N` spins."""
        if N == 1: return h * PauliPEPO.X

        H = np.zeros(shape=(2**N,2**N))
        I = np.eye(2)

        # coupling terms
        for i in range(N-1):
            ops = tuple(PauliPEPO.Z if _ in (i,i+1) else I for _ in range(N))
            H += J * multi_kron(*ops)
        # transverse field
        for i in range(N):
            ops = tuple(PauliPEPO.X if _ == i else I for _ in range(N))
            H += h * multi_kron(*ops)

        return H

class Heisenberg(PauliPEPO):
    """
    Heisenberg model with transverse field in x.
    """

    def __heisenberg_PEPO_tensor_without_coupling(self,N_pas:int,N_out:int,g:float) -> np.ndarray:
        """
        Returns a Heisenberg PEPO tensor, that is missing the incoming and outgoing coupling.

        `T` has the canonical leg ordering: The first leg is the incoming leg, afterwards follow the
        passive legs, and finally the outgoing legs.
        """
        # sanity check
        assert N_pas >= 0
        assert N_out >= 0
        assert hasattr(self,"chi")

        T = self.traversal_tensor(N_pas=N_pas,N_out=N_out)

        # transverse field
        index = (0,) + tuple(-1 for _ in range(N_pas + N_out)) + (slice(0,2),slice(0,2))
        T[index] = g * self.X

        return T

    def __heisenberg__coupling(self,node1:int,node2:int,Jx:float,Jy:float,Jz:float) -> None:
        """
        Adds Heisenberg-type coupling to the edge `(node1,node2)`. This means that
        both decay stages (of the finite state automaton) are added to this
        edge, NOT that an operator `Jz * sz * sz` is added to the
        Hamiltonian!
        """
        if node1 in self.tree.succ[node2]:
            # node1 is downstream from node2
            child = node1
            parent = node2
        else:
            # node2 is downstream from node1, or the edge is not contained in the tree (in which case the order does not matter)
            child = node2
            parent = node1

        # first particle decay stage at this edge; at parent node
        if parent != self.root:
            grandparent = tuple(_ for _ in self.tree.pred[parent].keys())[0]
            x_index = tuple(
                0 if i == self.G[grandparent][parent][0]["legs"][parent]
                else 1 if i == self.G[parent][child][0]["legs"][parent]
                else 4
                for i in range(self.G.nodes[parent]["T"].ndim - 2)
            ) + (slice(0,2),slice(0,2))
            y_index = tuple(
                0 if i == self.G[grandparent][parent][0]["legs"][parent]
                else 2 if i == self.G[parent][child][0]["legs"][parent]
                else 4
                for i in range(self.G.nodes[parent]["T"].ndim - 2)
            ) + (slice(0,2),slice(0,2))
            z_index = tuple(
                0 if i == self.G[grandparent][parent][0]["legs"][parent]
                else 3 if i == self.G[parent][child][0]["legs"][parent]
                else 4
                for i in range(self.G.nodes[parent]["T"].ndim - 2)
            ) + (slice(0,2),slice(0,2))
        else:
            x_index = tuple(
                1 if i == self.G[parent][child][0]["legs"][parent]
                else 4
                for i in range(self.G.nodes[parent]["T"].ndim - 3)
            ) + (0,slice(0,2),slice(0,2))
            y_index = tuple(
                2 if i == self.G[parent][child][0]["legs"][parent]
                else 4
                for i in range(self.G.nodes[parent]["T"].ndim - 3)
            ) + (0,slice(0,2),slice(0,2))
            z_index = tuple(
                3 if i == self.G[parent][child][0]["legs"][parent]
                else 4
                for i in range(self.G.nodes[parent]["T"].ndim - 3)
            ) + (0,slice(0,2),slice(0,2))
        self.G.nodes[parent]["T"][x_index] = Jx * self.X
        self.G.nodes[parent]["T"][y_index] = Jy * self.Y
        self.G.nodes[parent]["T"][z_index] = Jz * self.Z

        # second particle decay stage at this edge; at child node
        x_index = tuple(
            1 if i == self.G[parent][child][0]["legs"][child]
            else 4
            for i in range(self.G.nodes[child]["T"].ndim - 2)
        ) + (slice(0,2),slice(0,2))
        y_index = tuple(
            2 if i == self.G[parent][child][0]["legs"][child]
            else 4
            for i in range(self.G.nodes[child]["T"].ndim - 2)
        ) + (slice(0,2),slice(0,2))
        z_index = tuple(
            3 if i == self.G[parent][child][0]["legs"][child]
            else 4
            for i in range(self.G.nodes[child]["T"].ndim - 2)
        ) + (slice(0,2),slice(0,2))
        self.G.nodes[child]["T"][x_index] = self.X
        self.G.nodes[child]["T"][y_index] = self.Y
        self.G.nodes[child]["T"][z_index] = self.Z

        return

    def __init__(self,G:nx.MultiGraph,Jx:float=1,Jy:float=1,Jz:float=1,g:float=0,sanity_check:bool=False):
        """
        Travsverse Field Ising model PEPO on graph `G`, with coupling `J` and external field `g`.

        Ordering of legs in the PEPO virtual dimensions is inherited from `G`. The last two dimensions of every PEPO tensor are the physical dimensions.
        """
        # sanity check
        if sanity_check: assert network_message_check(G)

        super().__init__()

        self.chi = 5
        """
        Virtual bond dimension.
        0 is the moving particle state, 1/2/3 the decay state in x/y/z, and 4 is the vacuum state.
        """

        self.G = super().prepare_graph(G)

        # root node is node with smallest degree
        self.root = sorted(G.nodes(),key=lambda x: len(G.adj[x]))[0]

        # depth-first search tree
        self.tree = nx.dfs_tree(G,self.root)

        # adding PEPO tensors (without coupling)
        for node in self.G.nodes():
            N_in = len(self.tree.pred[node])
            N_out = len(self.tree.succ[node])
            N_pas = len(self.G.adj[node]) - N_out - N_in

            if N_out == 0:
                # node is a leaf
                N_out = 1

            # PEPO tensor, where the first dimension is the incoming leg, the passive legs
            # and the outgoing legs follow, and the last two dimensions are the physical legs
            T = self.__heisenberg_PEPO_tensor_without_coupling(N_pas=N_pas,N_out=N_out,g=g)

            if node == self.root:
                # root node; we need to put the (incoming) boundary leg at the last place within the virtual dimensions
                T = np.moveaxis(T,0,-3)

            # re-shaping PEPO tensor to match the graph leg ordering
            T = self.permute_PEPO(T,node)

            self.G.nodes[node]["T"] = T

        # adding incoming and outgoing coupling to every node but the root
        for node1,node2 in self.G.edges():
            self.__heisenberg__coupling(node1,node2,Jx,Jy,Jz)

        # contracting boundary legs
        self.contract_boundaries()

        if sanity_check: assert self.intact

        return

def posneg_TFI(G:nx.MultiGraph,J:float=1,g:float=0,sanity_check:bool=False) -> tuple[PEPO,PEPO]:
    """
    Constructs two PEPOs, where one contains the positive-semidefinite part of the TFI
    and the other contains the negative-semidefinite part.
    """
    # sanity check
    if sanity_check: assert network_message_check(G)

    # spectral decompositions of X and Z
    X_pos = np.ones(shape=(2,2)) / 2
    X_neg = np.array([[-1,1],[1,-1]]) / 2
    Z_pos = np.array([[1,0],[0,0]])
    Z_neg = np.array([[0,0],[0,-1]])

    pos_op = PEPO(D=2)
    neg_op = PEPO(D=2)
    # why not PauliPEPO? Because pos_op and neg_op will contain
    # operators that are not pauli matrices (e.g. projectors),
    # so the sanity check of PauliPEPO would not work

    chi = 4
    """
    Virtual bond dimension.
    0 is the moving particle state, 1 & 2 are decay states, and 3 is the vacuum state.
    """

    pos_op.chi = chi
    neg_op.chi = chi

    G = pos_op.prepare_graph(G)
    pos_op.G = copy.deepcopy(G)
    neg_op.G = copy.deepcopy(G)

    # root node is node with smallest degree
    root = sorted(G.nodes(),key=lambda x: len(G.adj[x]))[0]
    pos_op.root = root
    neg_op.root = root

    # depth-first search trees
    tree = nx.dfs_tree(G,root)
    pos_op.tree = copy.deepcopy(tree)
    neg_op.tree = copy.deepcopy(tree)

    # adding PEPO tensors (without coupling)
    for node in G.nodes():
        N_in = len(tree.pred[node])
        N_out = len(tree.succ[node])
        N_pas = len(G.adj[node]) - N_out - N_in

        if N_out == 0:
            # node is a leaf
            N_out = 1

        # PEPO tensor, where the first dimension is the incoming leg, the passive legs
        # and the outgoing legs follow, and the last two dimensions are the physical legs
        pos_T = pos_op.traversal_tensor(N_pas=N_pas,N_out=N_out)
        neg_T = neg_op.traversal_tensor(N_pas=N_pas,N_out=N_out)

        # transverse field
        index = (0,) + tuple(-1 for _ in range(N_pas + N_out)) + (slice(0,2),slice(0,2))
        pos_T[index] = g * X_pos
        neg_T[index] = g * X_neg

        if node == root:
            # root node; we need to put the (incoming) boundary leg between the virtual dimensions and the physical dimensions
            pos_T = np.moveaxis(pos_T,0,-3)
            neg_T = np.moveaxis(neg_T,0,-3)

        # re-shaping PEPO tensor to match the graph leg ordering
        pos_op[node] = pos_op.permute_PEPO(pos_T,node)
        neg_op[node] = neg_op.permute_PEPO(neg_T,node)

    # adding incoming and outgoing coupling to every node but the root
    for node1,node2 in G.edges():
        if node1 in tree.succ[node2]:
            # node1 is downstream from node2
            child = node1
            parent = node2
        else:
            # node2 is downstream from node1, or the edge is not contained in the tree (in which case the order does not matter)
            child = node2
            parent = node1

        # first particle decay stages at this edge; at parent node
        if parent != root:
            grandparent = tuple(_ for _ in tree.pred[parent].keys())[0]
            index = lambda x: tuple(
                0 if i == G[grandparent][parent][0]["legs"][parent]
                else x if i == G[parent][child][0]["legs"][parent]
                else -1
                for i in range(pos_op[parent].ndim - 2)
            ) + (slice(0,2),slice(0,2))
        else:
            # the incoming leg of the root node is a boundary leg, and thus
            # located just before the physical legs. This is why this case
            # distinction is necessary.
            index = lambda x: tuple(
                x if i == G[parent][child][0]["legs"][parent]
                else -1
                for i in range(pos_op[parent].ndim - 3)
            ) + (0,slice(0,2),slice(0,2))
        pos_op[parent][index(1)] = J * Z_pos
        pos_op[parent][index(2)] = J * Z_neg
        neg_op[parent][index(1)] = J * Z_pos
        neg_op[parent][index(2)] = J * Z_neg

        # final particle decay stage at this edge; at child node
        index = lambda x: tuple(
            x if i == G[parent][child][0]["legs"][child]
            else -1
            for i in range(pos_op[child].ndim - 2)
        ) + (slice(0,2),slice(0,2))
        pos_op[child][index(1)] = Z_pos
        pos_op[child][index(2)] = Z_neg
        neg_op[child][index(1)] = Z_neg
        neg_op[child][index(2)] = Z_pos

    # contracting boundary legs
    pos_op.contract_boundaries()
    neg_op.contract_boundaries()

    if sanity_check: assert pos_op.intact and neg_op.intact

    return pos_op,neg_op

def operator_chain(G:nx.MultiGraph,ops:dict[int,np.ndarray],sanity_check:bool=False) -> PEPO:
    """
    Product of single-site operators. The dictionary `ops`
    contains the operators as values, and the sites on which they act
    as keys.
    """
    # extracting physical dimension
    D = tuple(ops.values())[0].shape[0]

    H = PEPO.Identity(G=G,D=D,sanity_check=sanity_check)

    for node,op in ops.items():
        # sanity check
        if not G.has_node(node): raise ValueError(f"Node {node} not contained in graph.")
        if not op.shape == (D,D): raise ValueError(f"Operator on node {node} has wrong shape: Expected ({D},{D}), got " + str(op.shape) + ".")
        if not is_hermitian(op): raise ValueError(f"Operator at node {node} is not hermitian.")

        index = tuple(0 for _ in H.G.adj[node]) + (slice(0,D),slice(0,D))
        H.G.nodes[node]["T"][index] = op

    if sanity_check: assert H.intact
    return H

if __name__ == "__main__":
    pass
