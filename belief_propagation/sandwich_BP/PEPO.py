"""
Projector-entangled pair operators on arbitrary graphs.
"""

import numpy as np
import networkx as nx
import cotengra as ctg
import warnings
import itertools

from belief_propagation.utils import network_message_check,multi_kron,proportional,is_hermitian

class PEPO:
    """
    Base class for tensor product operators, composed of Pauli matrices. Subclasses must provide `__init__`.
    Therein, the following attributes must be defined:
    * `self.G`: Graph on which the Hamiltonian is defined.
    * `self.chi`: Virtual bond dimension.
    * `self.D`: Physical dimension.
    * `self.tree`: Tree that determines the traversal of `G`, along which the PEPO is oriented.
    * `self.root`: The root node of the tree.

    Writing down a PEPO on an arbitrary graph `G` can be achieved by
    finding a spanning tree of `G`. The flow of finite state automaton
    information is then defined by the tree: The origin is at the root,
    and it terminates at the leaves.

    During initialisation, the root and each leaf are equipped with an
    additional leg. These legs connect to the initial and final states of
    the finite state automaton, respectively. At the end of initialisation,
    these additional legs should be contracted such that the PEPO has the
    same structure as the underlying graph.

    It is assumed that, within the virtual dimension, the finite state automaton
    initial state is the 0th component, and the final state is the last
    component.
    """

    def intact_check(self) -> bool:
        """
        Checks if the PEPO is intact:
        * Is the underlying network message-ready?
        * Is the size of every edge saved?
        * Are the physical legs the last two dimensions in each tensor?
        * Do the physical legs have the correct sizes?
        * is the information flow in the tree intact?
        """
        # are all the necessary attributes defined?
        assert hasattr(self,"D")
        assert hasattr(self,"chi")
        assert hasattr(self,"G")
        assert hasattr(self,"tree")
        assert hasattr(self,"root")

        # is the underlying network message-ready?
        if not network_message_check(self.G):
            warnings.warn("Network not intact.")
            return False

        # size attribute given on every edge?
        for node1,node2,data in self.G.edges(data=True):
            if not "size" in data.keys():
                warnings.warn(f"No size saved in edge ({node1},{node2}).")
                return False
            if data["size"] != self.chi:
                warnings.warn(f"Wrong size saved in edge ({node1},{node2}).")
                return False

        # are the physical legs the last dimension in each tensor?
        for node,T in self.G.nodes(data="T"):
            legs = [leg for leg in range(T.ndim)]
            for node1,node2,key in self.G.edges(node,keys=True):
                try:
                    if not self.G[node1][node2][key]["trace"]:
                        legs.remove(self.G[node1][node2][key]["legs"][node])
                    else:
                        # trace edge
                        i1,i2 = self.G[node1][node2][key]["indices"]
                        legs.remove(i1)
                        legs.remove(i2)
                except ValueError:
                    warnings.warn(f"Wrong leg in edge ({node1},{node2},{key}).")
                    return False

            if not legs == [T.ndim - 2,T.ndim - 1]:
                warnings.warn(f"Physical legs are not the last two dimensions in node {node}.")
                return False

            if not (T.shape[-2] == self.D and T.shape[-1] == self.D):
                warnings.warn(f"Hilbert space at node {node} has wrong size.")
                return False

        # tree traversal correct?
        for node in self.tree.nodes():
            if (len(self.tree.succ[node]) > 0) and (node != self.root):
                # node is an intermediate node in the tree

                # checking if the particle state is pased along
                for parent,child in itertools.product(self.tree.pred[node],self.tree.succ[node]):
                    index = tuple(
                        0 if _ in (self.G[node][parent][key]["legs"][node],self.G[node][child][key]["legs"][node])
                        else -1
                        for _ in range(self.G.nodes[node]["T"].ndim - 2)
                    ) + (slice(0,2),slice(0,2))
                    if not np.allclose(self.G.nodes[node]["T"][index],self.I):
                        warnings.warn(f"Wrong indices for particle state passthrough in node {node}.")
                        return False

                # checking if the vacuum state is passed along
                index = tuple(-1 for _ in range(self.G.nodes[node]["T"].ndim - 2)) + (slice(0,2),slice(0,2))
                if not np.allclose(self.G.nodes[node]["T"][index],self.I):
                    warnings.warn(f"Wrong indices for vacuum state passthrough in node {node}.")
                    return False

        return True

    def permute_PEPO(self,T:np.ndarray,node:int) -> np.ndarray:
        """
        Re-shapes the PEPO-tensor `T` at node `node` such that it fits into
        the graph `self.G`.

        It is assumed that the dimensions of `T` are in the canonical
        order: First the virtual dimensions, followed by two physical legs.
        The leading virtual dimension is the incoming leg, which is
        followed by the passive legs and, finally, the outgoing legs.

        For a node in `G` with N neighbors, `T` must have at least N+2 legs.
        The last two legs are assumed to be the physical legs, and the first N
        legs are the virtual bonds. Legs that are neither among the first N or
        the last 2 remain untouched; these are boundary legs that connect to
        the initial and final states of the finite state automaton. This
        function thus permutes only the first N dimensions.
        """
        # sanity check
        assert node in self.G
        assert node in self.tree

        N_in = len(self.tree.pred[node])
        N_out = len(self.tree.succ[node])
        N_pas = len(self.G.adj[node]) - N_out - N_in

        # sanity check
        assert T.ndim - N_in - N_pas - N_out - 2 in (0,1), "Tensor may have up to one boundary leg."

        newshape = [np.nan for _ in self.G.adj[node]]
        out_counter = 0
        pas_counter = 0
        for neighbor in self.G.adj[node]:
            if neighbor in self.tree.pred[node]:
                # the incoming edge
                newshape[self.G[node][neighbor][0]["legs"][node]] = 0
                continue
            if neighbor in self.tree.succ[node]:
                # outgoing edge
                newshape[self.G[node][neighbor][0]["legs"][node]] = N_in + N_pas + out_counter
                out_counter += 1
                continue

            # passive edge
            newshape[self.G[node][neighbor][0]["legs"][node]] = N_in + pas_counter
            pas_counter += 1

        return np.transpose(T,newshape + [_ for _ in range(len(newshape),T.ndim)])

    def contract_boundaries(self):
        """
        Contracts boundary legs.

        This means contraction of the root node with the initial state,
        and each leaf with the final state. It is assumed that the initial
        state is the 0th component of the bond dimension, and that the
        final state is the last component.
        """
        for node in self.G.nodes():
            if node == self.root: # root node
                self.G.nodes[node]["T"] = self.G.nodes[node]["T"][...,0,:,:]
                continue

            if len(self.tree.succ[node]) == 0: # leaf node
                self.G.nodes[node]["T"] = self.G.nodes[node]["T"][...,-1,:,:]
                continue

        return

    def to_dense(self,sanity_check:bool=False) -> np.ndarray:
        """
        Contracts the PEPO using `ctg.einsum`.
        """
        if sanity_check: assert self.intact_check()

        if self.G.number_of_nodes() == 1:
            # the network is trivial
            return tuple(self.G.nodes(data="T"))[0][1]

        N = self.G.number_of_edges()
        args = ()

        # enumerating the edges in the graph
        for i,nodes in enumerate(self.G.edges()):
            node1,node2 = nodes
            self.G[node1][node2][0]["label"] = i

        # extracting the einsum arguments
        for node,T in self.G.nodes(data="T"):
            args += (T,)
            legs = [None for _ in range(T.ndim-2)] + [N,N+1] # last two indices are the physical legs
            for _,neighbor,edge_label in self.G.edges(nbunch=node,data="label"):
                legs[self.G[node][neighbor][0]["legs"][node]] = edge_label
            args += (tuple(legs),)
            N += 2

        H = ctg.einsum(*args,optimize="greedy")
        # reshaping
        H = np.transpose(H,[2*i for i in range(self.G.number_of_nodes())] + [2*i+1 for i in range(self.G.number_of_nodes())])
        H = np.reshape(H,newshape=(self.D ** self.G.number_of_nodes(),self.D ** self.G.number_of_nodes()))

        if sanity_check: assert is_hermitian(H)

        return H

    def view_site(self,node:int):
        """
        Prints all components of the tensor at node `node`.
        """
        # sanity check
        assert node in self.G

        legs_in = tuple(self.G[node][neighbor][0]["legs"][node] for neighbor in self.tree.pred[node].keys())
        legs_out = tuple(self.G[node][neighbor][0]["legs"][node] for neighbor in self.tree.succ[node].keys())

        print(
            f"Displaying node {node}:" + "\n    " +
            f"legs {legs_in} incoming" + "\n    " +
            f"legs {legs_out} outgoing" + "\n"
        )
        for virtual_index in itertools.product(range(self.chi),repeat=self.G.nodes[node]["T"].ndim-2):
            index = virtual_index + (slice(0,self.D),slice(0,self.D))

            if not np.allclose(self.G.nodes[node]["T"][index],0):
                print(index[:-2],":\n",self.G.nodes[node]["T"][index],"\n")

        return

    def prepare_graph(self,G:nx.MultiGraph) -> nx.MultiGraph:
        """
        Creates a shallow copy of G, and adds the keys `legs`, `trace`, `indices`
        and `size` to the edges.
        
        The leg ordering is preserved, `trace` is set to `False` and, accordingly,
        `indices` to `None`. The size of each axis is set to `self.chi`.
        
        To be used in `__init__` of subclasses of `PEPO`: `G` is the graph from which
        the operator inherits it's underlying graph.
        """
        # shallow copy of G
        newG = nx.MultiGraph(G.edges())
        # adding legs attribute to each edge
        for node1,node2,legs in G.edges(data="legs",keys=False):
            newG[node1][node2][0]["legs"] = legs
            newG[node1][node2][0]["trace"] = False
            newG[node1][node2][0]["indices"] = None
            newG[node1][node2][0]["size"] = self.chi

        return newG

    def legs_dict(self,node,sanity_check:bool=False) -> dict:
        """
        Returns the `legs` attributes of the adjacent edges of `node`
        in a dictionary structure: `legs_dict[neighbor]` is the
        same as `self.G[node][neighbor][0]["legs"]`.
        """
        if sanity_check: assert self.intact_check()

        out = dict()

        for neighbor in self.G.adj[node]:
            out[neighbor] = self.G[node][neighbor][0]["legs"]

        return out

    @property
    def I(self) -> np.ndarray:
        """Identity matrix with the respective dimensions."""
        return np.eye(self.D)

    @staticmethod
    def view_tensor(T:np.ndarray):
        """
        Prints all extractable information from tensor `T`
        """
        chi = T.shape[0]
        D = T.shape[-1]
        # sanity check
        for i in range(T.ndim):
            if i < T.ndim - 2:
                assert T.shape[i] == chi
            else:
                assert T.shape[i] == D

        print(f"Bond dimension {chi}.\n")

        # printing cellular automaton components
        for virtual_index in itertools.product(range(chi),repeat=T.ndim-2):
            index = virtual_index + (slice(0,D),slice(0,D))

            if not np.allclose(T[index],0):
                print(index[:-2],":\n",T[index],"\n")

    @classmethod
    def Identity(cls,G:nx.MultiGraph,D:int,sanity_check:bool=False):
        """
        Returns the identity PEPO on graph `G`.
        Physical dimension `D`.
        """
        Id = cls(D)
        Id.chi = 1
        Id.G = Id.prepare_graph(G)

        # root node is node with smallest degree
        Id.root = sorted(G.nodes(),key=lambda x: len(G.adj[x]))[0]

        # depth-first search tree
        Id.tree = nx.dfs_tree(G,Id.root)

        # adding physical dimensions, putting identities in the physical dimensions
        for node in Id.G.nodes():
            T = np.zeros(shape = tuple(Id.chi for _ in range(len(G.adj[node]))) + (Id.D,Id.D))
            T[...,:,:] = Id.I
            Id.G.nodes[node]["T"] = T

        if sanity_check: assert Id.intact_check()

        return Id

    def __init__(self,D:int) -> None:
        self.D = D
        """Physical dimension."""
        self.G:nx.MultiGraph
        self.tree:nx.DiGraph
        """Spanning tree of the graph."""
        self.root:int
        """Root node of the spanning tree."""
        self.chi:int
        """Virtual bond dimension."""

        return

class PauliPEPO(PEPO):
    """
    Tensor product operators on spin systems,
    composed of Pauli operators.
    """
    X=np.array([[0,1],[1,0]])
    """Pauli $X$-matrix."""
    Y=np.array([[0,-1j],[1j,0]])
    """Pauli $Y$-matrix."""
    Z=np.array([[1,0],[0,-1]])
    """Pauli $Z$-matrix."""

    def intact_check(self) -> bool:
        """
        Checks if the PEPO is intact:
        * Calls `super().intact_check()`.
        * Checks if the hamiltonian is composed of Pauli operators.
        """
        if not super().intact_check(): return False

        # Hamiltonian composed of pauli operators?
        for node,T in self.G.nodes(data="T"):
            for virtual_index in itertools.product(range(self.chi),repeat=self.G.nodes[node]["T"].ndim-2):
                index = virtual_index + (slice(0,self.D),slice(0,self.D))

                if np.allclose(T[index],0): continue

                closeness = [proportional(T[index],op,10) for op in (self.X,self.Y,self.Z,self.I)]

                if not any(closeness):
                    warnings.warn(f"Unknown operator in index {virtual_index} at node {node}.")
                    return False

        return True

    def __init__(self) -> None:
        super().__init__(2)

        return

class TFI(PauliPEPO):
    """
    Travsverse Field Ising model.
    """

    def __init__(self,G:nx.MultiGraph,J:float=1,h:float=0,sanity_check:bool=False) -> nx.MultiGraph:
        """
        Travsverse Field Ising model PEPO on graph `G`, with coupling `J` and external field `h`.

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

        def ising_PEPO_tensor_without_coupling(N_pas:int,N_out:int,h:float) -> np.ndarray:
            """
            Returns a Transverse Field Ising PEPO tensor, that is missing the incoming and outgoing coupling.

            `T` has the canonical leg ordering: The first leg is the incoming leg, afterwards follow the
            passive legs, and finally the outgoing legs.
            """
            # sanity check
            assert N_pas >= 0
            assert N_out >= 0

            T = np.zeros(shape=[self.chi for _ in range(1 + N_pas + N_out)] + [2,2])

            # particle index
            for i_out in range(N_out):
                index = (0,) + tuple(self.chi - 1 for _ in range(N_pas)) + tuple(0 if _ == i_out else self.chi - 1 for _ in range(N_out)) + (slice(0,2),slice(0,2))
                T[index] = self.I

            # vacuum index
            index = tuple(self.chi - 1 for _ in range(1 + N_pas + N_out)) + (slice(0,2),slice(0,2))
            T[index] = self.I

            # transverse field
            index = (0,) + tuple(-1 for _ in range(N_pas + N_out)) + (slice(0,2),slice(0,2))
            T[index] = h * self.X

            return T

        self.G = super().prepare_graph(G)

        # root node is node with smallest degree
        self.root = sorted(G.nodes(),key=lambda x: len(G.adj[x]))[0]

        # depth-first search tree
        self.tree = nx.dfs_tree(G,self.root)

        # adding PEPO tensors (without coupling)
        for node in self.G.nodes():
            N_in = len(self.tree.pred[node])
            N_out = len(self.tree.succ[node])
            N_pas = len(G.adj[node]) - N_out - N_in

            if N_out == 0:
                # node is a leaf
                N_out = 1

            # PEPO tensor, where the first dimension is the incoming leg, the passive legs
            # and the outgoing legs follow, and the last two dimensions are the physical legs
            T = ising_PEPO_tensor_without_coupling(N_pas=N_pas,N_out=N_out,h=h)

            if node == self.root:
                # root node; we need to put the (incoming) boundary leg between the virtual dimensions and the physical dimensions
                T = np.moveaxis(T,0,-3)

            # re-shaping PEPO tensor to match the graph leg ordering
            T = self.permute_PEPO(T,node)

            self.G.nodes[node]["T"] = T

        # adding incoming and outgoing coupling to every node but the root
        for node1,node2 in self.G.edges():
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
                    0 if i == G[grandparent][parent][0]["legs"][parent]
                    else 1 if i == G[parent][child][0]["legs"][parent]
                    else 2
                    for i in range(self.G.nodes[parent]["T"].ndim - 2)
                ) + (slice(0,2),slice(0,2))
            else:
                index = tuple(
                    1 if i == G[parent][child][0]["legs"][parent]
                    else 2
                    for i in range(self.G.nodes[parent]["T"].ndim - 3)
                ) + (0,slice(0,2),slice(0,2))
            self.G.nodes[parent]["T"][index] = J * self.Z

            # second particle decay stage at this edge; at child node
            index = tuple(
                1 if i == G[parent][child][0]["legs"][child]
                else 2
                for i in range(self.G.nodes[child]["T"].ndim - 2)
            ) + (slice(0,2),slice(0,2))
            self.G.nodes[child]["T"][index] = self.Z

        # contracting boundary legs
        self.contract_boundaries()

        if sanity_check: assert self.intact_check()

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

    def __init__(self,G:nx.MultiGraph,Jx:float=1,Jy:float=1,Jz:float=1,h:float=0,sanity_check:bool=False) -> nx.MultiGraph:
        """
        Travsverse Field Ising model PEPO on graph `G`, with coupling `J` and external field `h`.

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

        def heisenberg_PEPO_tensor_without_coupling(N_pas:int,N_out:int,h:float) -> np.ndarray:
            """
            Returns a Heisenberg PEPO tensor, that is missing the incoming and outgoing coupling.

            `T` has the canonical leg ordering: The first leg is the incoming leg, afterwards follow the
            passive legs, and finally the outgoing legs.
            """
            # sanity check
            assert N_pas >= 0
            assert N_out >= 0

            T = np.zeros(shape=[self.chi for _ in range(1 + N_pas + N_out)] + [2,2],dtype=np.complex128)

            # particle index
            for i_out in range(N_out):
                index = (0,) + tuple(self.chi - 1 for _ in range(N_pas)) + tuple(0 if _ == i_out else self.chi - 1 for _ in range(N_out)) + (slice(0,2),slice(0,2))
                T[index] = self.I

            # vacuum index
            index = tuple(self.chi - 1 for _ in range(1 + N_pas + N_out)) + (slice(0,2),slice(0,2))
            T[index] = self.I

            # transverse field
            index = (0,) + tuple(self.chi - 1 for _ in range(N_pas + N_out)) + (slice(0,2),slice(0,2))
            T[index] = h * self.X

            return T

        self.G = super().prepare_graph(G)

        # root node is node with smallest degree
        self.root = sorted(G.nodes(),key=lambda x: len(G.adj[x]))[0]

        # depth-first search tree
        self.tree = nx.dfs_tree(G,self.root)

        # adding PEPO tensors (without coupling)
        for node in self.G.nodes():
            N_in = len(self.tree.pred[node])
            N_out = len(self.tree.succ[node])
            N_pas = len(G.adj[node]) - N_out - N_in

            if N_out == 0:
                # node is a leaf
                N_out = 1

            # PEPO tensor, where the first dimension is the incoming leg, the passive legs
            # and the outgoing legs follow, and the last two dimensions are the physical legs
            T = heisenberg_PEPO_tensor_without_coupling(N_pas=N_pas,N_out=N_out,h=h)

            if node == self.root:
                # root node; we need to put the (incoming) boundary leg at the last place within the virtual dimensions
                T = np.moveaxis(T,0,-3)

            # re-shaping PEPO tensor to match the graph leg ordering
            T = self.permute_PEPO(T,node)

            self.G.nodes[node]["T"] = T

        # adding incoming and outgoing coupling to every node but the root
        for node1,node2 in self.G.edges():
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
                    0 if i == G[grandparent][parent][0]["legs"][parent]
                    else 1 if i == G[parent][child][0]["legs"][parent]
                    else 4
                    for i in range(self.G.nodes[parent]["T"].ndim - 2)
                ) + (slice(0,2),slice(0,2))
                y_index = tuple(
                    0 if i == G[grandparent][parent][0]["legs"][parent]
                    else 2 if i == G[parent][child][0]["legs"][parent]
                    else 4
                    for i in range(self.G.nodes[parent]["T"].ndim - 2)
                ) + (slice(0,2),slice(0,2))
                z_index = tuple(
                    0 if i == G[grandparent][parent][0]["legs"][parent]
                    else 3 if i == G[parent][child][0]["legs"][parent]
                    else 4
                    for i in range(self.G.nodes[parent]["T"].ndim - 2)
                ) + (slice(0,2),slice(0,2))
            else:
                x_index = tuple(
                    1 if i == G[parent][child][0]["legs"][parent]
                    else 4
                    for i in range(self.G.nodes[parent]["T"].ndim - 3)
                ) + (0,slice(0,2),slice(0,2))
                y_index = tuple(
                    2 if i == G[parent][child][0]["legs"][parent]
                    else 4
                    for i in range(self.G.nodes[parent]["T"].ndim - 3)
                ) + (0,slice(0,2),slice(0,2))
                z_index = tuple(
                    3 if i == G[parent][child][0]["legs"][parent]
                    else 4
                    for i in range(self.G.nodes[parent]["T"].ndim - 3)
                ) + (0,slice(0,2),slice(0,2))
            self.G.nodes[parent]["T"][x_index] = Jx * self.X
            self.G.nodes[parent]["T"][y_index] = Jy * self.Y
            self.G.nodes[parent]["T"][z_index] = Jz * self.Z

            # second particle decay stage at this edge; at child node
            x_index = tuple(
                1 if i == G[parent][child][0]["legs"][child]
                else 4
                for i in range(self.G.nodes[child]["T"].ndim - 2)
            ) + (slice(0,2),slice(0,2))
            y_index = tuple(
                2 if i == G[parent][child][0]["legs"][child]
                else 4
                for i in range(self.G.nodes[child]["T"].ndim - 2)
            ) + (slice(0,2),slice(0,2))
            z_index = tuple(
                3 if i == G[parent][child][0]["legs"][child]
                else 4
                for i in range(self.G.nodes[child]["T"].ndim - 2)
            ) + (slice(0,2),slice(0,2))
            self.G.nodes[child]["T"][x_index] = self.X
            self.G.nodes[child]["T"][y_index] = self.Y
            self.G.nodes[child]["T"][z_index] = self.Z

        # contracting boundary legs
        self.contract_boundaries()

        if sanity_check: assert self.intact_check()

        return

if __name__ == "__main__":
    pass
