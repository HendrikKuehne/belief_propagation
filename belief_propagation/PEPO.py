"""
Projector-entangled pair operators on arbitrary graphs.
"""

import numpy as np
import networkx as nx
import sparse
import cotengra as ctg
import scipy.sparse as scisparse
import warnings
import itertools
import copy
from typing import Union

from belief_propagation.utils import network_message_check,multi_kron,proportional,is_hermitian,same_legs
from belief_propagation.PEPS import PEPS

class PEPO:
    """
    Base class for tensor product operators, that are constructed
    mathematically as sums of operator chains. Subclasses must provide `__init__`. Therein,
    the following attributes must be defined:
    * `self.G`: Graph on which the Hamiltonian is defined.
    * `self.chi`: Virtual bond dimension.
    * `self.D`: Physical dimension.
    * `self.tree`: Tree that determines the traversal of `G`, along which the PEPO is oriented.
    * `self.root`: The root node of the tree.

    Writing down a PEPO on an arbitrary graph `G` can be achieved by
    finding a spanning tree of `G`. The flow of finite state automaton
    information is then defined by the tree: The origin is at the root,
    and it terminates at the leaves. This method is similar to what is
    presented in [SciPost Phys. Core 7, 036 (2024)](https://doi.org/10.21468/SciPostPhysCore.7.2.036).

    Every node in the graph has one inbound leg, passive legs, and outbound
    legs. The initial and final
    state of the finite state automaton are the first and last components,
    respectively. The initial state ("particle state") is only passed along
    the tree, while intermediate states ("decay states") can be passed along
    any edge. The final state ("vacuum state") is passed along every edge.
    The physical legs are the last two dimensions if the PEPO tensors. All
    other legs are virtual bond dimensions. The correspondence between legs
    and neighbors is determined by the `legs` attribute on each edge.

    During initialisation, the root and each leaf are equipped with an
    additional leg. These legs connect to the initial and final states of
    the finite state automaton, respectively. At the end of initialisation,
    these additional legs should be contracted such that the PEPO has the
    same structure as the underlying graph.

    It is assumed that, within the virtual dimension, the finite state automaton
    initial state is the 0th component, and the final state is the last
    component.
    """

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
        Contracts boundary legs. Boundary legs are assumed to be located
        in the last virtual dimension.

        This means contraction of the root node with the initial state,
        and each leaf with the final state. It is assumed that the initial
        state is the 0th component of the bond dimension, and that the
        final state is the last component.
        """
        for node in self.G.nodes():
            if node == self.root: # root node
                self[node] = self[node][...,0,:,:]
                continue

            if len(self.tree.succ[node]) == 0: # leaf node
                self[node] = self[node][...,-1,:,:]
                continue

        return

    def toarray(self,create_using:str="cotengra",sanity_check:bool=False):
        """
        Construct a matrix representation of this operator. Different methods are implemented,
        which can be selected by the argument `create_using`:
        * `cotengra`: Dense Numpy-array from contraction of the PEPO.
        * `scipy.csr`: Scipy csr-sparse array, constructed using operator chains.
        * `sparse`: Scipy csr-sparse matrix from sparse contraction of the network. Works for small PEPOs only.
        """

        if create_using == "cotengra": return self.__to_dense(sanity_check=sanity_check)

        if create_using == "scipy.csr": return self.__to_sparse(create_using="scipy.csr",sanity_check=sanity_check)

        if create_using == "sparse": return self.__to_sparse(create_using="sparse",sanity_check=sanity_check)

        raise ValueError("toarray not implemented for method " + create_using + ".")

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
                print(index[:-2],":\n",self[node][index],"\n")

        return

    def prepare_graph(self,G:nx.MultiGraph,sanity_check:bool=False) -> nx.MultiGraph:
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

        # adding additional information to every edge
        for node1,node2,legs in newG.edges(data="legs",keys=False):
            newG[node1][node2][0]["trace"] = False
            newG[node1][node2][0]["indices"] = None
            newG[node1][node2][0]["size"] = self.chi
            newG[node1][node2][0]["legs"] = {}

        for node in newG.nodes:
            # adding to the adjacent edges which index they correspond to
            for i,neighbor in enumerate(newG.adj[node]):
                newG[node][neighbor][0]["legs"][node] = i

        if sanity_check: assert network_message_check(newG)

        return newG

    def legs_dict(self,node,sanity_check:bool=False) -> dict:
        """
        Returns the `legs` attributes of the adjacent edges of `node`
        in a dictionary structure: `legs_dict[neighbor]` is the
        same as `self.G[node][neighbor][0]["legs"]`.
        """
        if sanity_check: assert self.intact

        out = dict()

        for neighbor in self.G.adj[node]:
            out[neighbor] = self.G[node][neighbor][0]["legs"]

        return out

    def traversal_tensor(self,N_pas:int,N_out:int,dtype=np.complex128) -> np.ndarray:
        """
        Returns the minimum tensor for PEPOs, that is, a tensor
        that ensures correct tree traversal.

        `T` has the canonical leg ordering: The first leg is the incoming leg, afterwards follow the
        passive legs, and finally the outgoing legs.
        """
        # sanity check
        assert N_pas >= 0
        assert N_out >= 0
        assert hasattr(self,"chi")

        T = np.zeros(shape=[self.chi for _ in range(1 + N_pas + N_out)] + [self.D,self.D],dtype=dtype)

        # particle index
        for i_out in range(N_out):
            index = (0,) + tuple(self.chi - 1 for _ in range(N_pas)) + tuple(0 if _ == i_out else self.chi - 1 for _ in range(N_out)) + (slice(0,self.D),slice(0,self.D))
            T[index] = self.I

        # vacuum index
        index = tuple(self.chi - 1 for _ in range(1 + N_pas + N_out)) + (slice(0,self.D),slice(0,self.D))
        T[index] = self.I

        return T

    def operator_chains(self,sanity_check:bool=False) -> tuple[dict[int:tuple]]:
        """
        Returns all operator chains. An operator chain is a collection
        of operators. The summation of the tensor products of all operator chains
        gives the operator.

        Operator chains are returned as a dict, where nodes are keys
        and indices are values. The index is an index for the node
        tensor, s.t. `self[node][index]` is part of the respective
        operator chain.
        """
        # sanity checks
        if sanity_check: assert self.intact

        # why a recursion? For large graphs, simply iterating through all
        # indices to find valid chains might take prohibitively long

        operator_chains = []

        self.__chain_construction_recursion(node=self.root,i_upstream=np.nan,chain=dict(),chains=operator_chains)

        return tuple(operator_chains)

    def __chain_construction_recursion(self,node:int,i_upstream:int,chain:dict[int,tuple],chains:list[dict[int,tuple]]) -> None:
        """
        Traversing the PEPO either downstream along the tree, or hopping
        from one branch (of the tree) to another, collecting the operator
        chains. Chains are saved in the argument `chains`, which is
        amnipulated in-place.
        """
        # this recursion breaks off eventually because the finite state automaton
        # that defines the tree does not have feedback loops

        if node != self.root:
            assert len(self.tree.pred[node]) == 1
            # parameters of the upstream node (parent)
            parent = tuple(self.tree.pred[node])[0]
            upstream_leg = self.G[node][parent][0]["legs"][node]
        else:
            # an upstream node only exists if node is not the root node
            parent = np.nan
            upstream_leg = np.nan

        # checking if the operator chain that terminates in node is non-zero
        terminal_index = tuple(i_upstream if _ == upstream_leg else self.chi - 1 for _ in range(self[node].ndim - 2)) + (slice(0,self.D),slice(0,self.D))
        if not np.allclose(self[node][terminal_index],0):
            chain = copy.deepcopy(chain)
            chain[node] = terminal_index
            chains.append(chain)

        for child,i_downstream in itertools.product(self.G.adj[node],range(self.chi)):
            if child == parent:
                # this leg is the upstream leg in the PEPO tree; this is not where the operator chain continues
                continue

            downstream_leg = self.G[node][child][0]["legs"][node]
            # assembling an index that takes us downstream in the operator chain
            index = tuple(
                i_downstream if _ == downstream_leg else
                i_upstream if _ == upstream_leg else
                self.chi - 1
                for _ in range(self[node].ndim-2)
            ) + (slice(0,self.D),slice(0,self.D))

            if np.allclose(self[node][index],0):
                # not part of any operator chain
                continue

            if index == terminal_index:
                # this operator chain terminates here
                continue

            nextchain = copy.deepcopy(chain)
            nextchain[node] = index

            self.__chain_construction_recursion(node=child,i_upstream=i_downstream,chain=nextchain,chains=chains)

    def __to_dense(self,sanity_check:bool) -> np.ndarray:
        """
        Constructs the operator using `ctg.einsum`.
        """
        if sanity_check: assert self.intact

        if self.nsites == 1:
            # the network is trivial
            return tuple(self.G.nodes(data="T"))[0][1]

        inputs = ()
        tensors = ()

        # enumerating the edges in the graph
        for i,nodes in enumerate(self.G.edges()):
            node1,node2 = nodes
            self.G[node1][node2][0]["label"] = ctg.get_symbol(i)

        N_edges = self.G.number_of_edges()
        # assembling the einsum arguments
        for i,nodeT in enumerate(self.G.nodes(data="T")):
            node,T = nodeT
            legs = [None for _ in range(T.ndim-2)] + [ctg.get_symbol(N_edges + i),ctg.get_symbol(N_edges + self.G.number_of_nodes() + i)] # last two indices are the physical legs
            for _,neighbor,edge_label in self.G.edges(nbunch=node,data="label"):
                legs[self.G[node][neighbor][0]["legs"][node]] = edge_label

            inputs += (legs,)
            tensors += (T,)

        # output ordering
        output = tuple(ctg.get_symbol(i) for i in range(N_edges,N_edges + 2 * self.G.number_of_nodes()))

        # getting the einsum expression, and contracting
        expr = ctg.utils.inputs_output_to_eq(inputs=inputs,output=output)
        H = ctg.einsum(expr,*tensors)
        H = np.reshape(H,newshape=(self.D ** self.G.number_of_nodes(),self.D ** self.G.number_of_nodes()))

        if sanity_check: assert is_hermitian(H)

        return H

        inputs = ()
        arrays = ()
        size_dict = {}

        # enumerating the edges in the graph
        for i,nodes in enumerate(self.G.edges()):
            node1,node2 = nodes
            self.G[node1][node2][0]["label"] = i
            size_dict[i] = self.G[node1][node2][0]["size"]

        N_edges = self.G.number_of_edges()
        # assembling the einsum arguments
        for i,nodeT in enumerate(self.G.nodes(data="T")):
            node,T = nodeT
            legs = [None for _ in range(T.ndim-2)] + [N_edges + i,N_edges + self.G.number_of_nodes() + i] # last two indices are the physical legs
            for _,neighbor,edge_label in self.G.edges(nbunch=node,data="label"):
                legs[self.G[node][neighbor][0]["legs"][node]] = edge_label

            inputs += (legs,)
            arrays += (T,)

        # output ordering
        output = tuple(_ for _ in range(N_edges,N_edges + 2 * self.G.number_of_nodes()))

        H = ctg.array_contract(arrays=arrays,inputs=inputs,output=output,size_dict=size_dict)
        H = np.reshape(H,newshape=(self.D ** self.G.number_of_nodes(),self.D ** self.G.number_of_nodes()))

        if sanity_check: assert is_hermitian(H)

        return H

    def __to_sparse(self,create_using:str,sanity_check:bool) -> Union[scisparse.csr_array,scisparse.csr_matrix]:
        """
        Constructs the operator using sparse operations.
        """
        if sanity_check: assert self.intact

        if self.nsites == 1:
            # the network is trivial
            return tuple(self.G.nodes(data="T"))[0][1]

        if create_using == "scipy.csr":
            # H will be constructed by summing the contributions from all operator chains.
            # array construction is fastest using the coo format, but I'm returning csr
            # because this is optimal for matrix-vector multiplication; an operation that
            # the Lanczos algorithm heavily relies on
            chains = self.operator_chains(sanity_check=sanity_check)

            H = scisparse.csr_array((self.D**self.nsites,self.D**self.nsites))
            nodes = tuple(self.G.nodes())

            for chain in chains:
                ops = tuple(
                    scisparse.coo_array(self[node][chain[node]]) if node in chain.keys()
                    else scisparse.eye_array(self.D,format="coo")
                    for node in nodes[::-1]
                )
                H += multi_kron(*ops,create_using="scipy.coo")

            return H.tocsr()

        if create_using == "sparse":
            #if self.G.number_of_edges() > len(np.core.einsumfunc.einsum_symbols):
            #    raise RuntimeError(f"The sparse package allows einsum contractions with up to {len(np.core.einsumfunc.einsum_symbols)} indices. The operator has too many edges ({self.G.number_of_edges()} edges).")

            inputs = ()
            tensors = ()

            # enumerating the edges in the graph
            for i,nodes in enumerate(self.G.edges()):
                node1,node2 = nodes
                self.G[node1][node2][0]["label"] = ctg.get_symbol(i)#np.core.einsumfunc.einsum_symbols[i]

            N_edges = self.G.number_of_edges()
            output = tuple(ctg.get_symbol(N_edges + i) for i in range(2 * self.nsites))

            # assembling the einsum arguments
            for i,node in enumerate(self.G.nodes()):
                legs = [None for _ in range(self[node].ndim-2)] + [ctg.get_symbol(N_edges + i),ctg.get_symbol(N_edges + self.G.number_of_nodes() + i)] # last two indices are the physical legs
                for _,neighbor,edge_label in self.G.edges(nbunch=node,data="label"):
                    legs[self.G[node][neighbor][0]["legs"][node]] = edge_label

                inputs += (tuple(legs),)
                tensors += (sparse.GCXS(self[node]),)

            # einsum expression, with all physical dimensions contained in ellipsis
            einsum_expr = ctg.utils.inputs_output_to_eq(inputs=inputs,output=output)
            # contraction using einsum
            H = sparse.einsum(einsum_expr,*tensors)
            H = sparse.reshape(H,shape=(self.D ** self.G.number_of_nodes(),self.D ** self.G.number_of_nodes()))

            return H.to_scipy_sparse()

        raise ValueError("__to_sparse not implemented for method " + create_using + ".")

    @property
    def I(self) -> np.ndarray:
        """Identity matrix with the dimensions `(self.D,self.D)`."""
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
        * Is every constituent operator hermitian?
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
            # accounting for virtual dimensions
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

        # local operators hermitian?
        for node in self.G.nodes():
            iterables = tuple(range(size) for size in self[node].shape[:-2])
            for index in itertools.product(*iterables):
                index += (slice(0,self.D),slice(0,self.D))
                if not is_hermitian(self[node][index]):
                    warnings.warn(f"Non-hermitian operator at node {node} in index " + str(index) + ".")
                    return False

        if not self.check_tree:
            # the following tests fail if the PEPO is constructed using PEPO.__add__ (status on 5th of February)
            return True

        if self.chi == 1:
            # PEPOs with bond dimension 1 are product operators, for which tree traversal checks are not necessary
            # (one does not need a finite state automaton to write a product operator as a PEPO)
            return True

        # tree traversal correct?
        for node in self.tree.nodes():
            if (len(self.tree.succ[node]) > 0) and (node != self.root):
                # node is an intermediate node in the tree

                # checking if the particle state is pased along
                for parent,child in itertools.product(self.tree.pred[node],self.tree.succ[node]):
                    index = tuple(
                        0 if _ in (self.G[node][parent][0]["legs"][node],self.G[node][child][0]["legs"][node])
                        else -1
                        for _ in range(self[node].ndim - 2)
                    ) + (slice(0,self.D),slice(0,self.D))
                    if not np.allclose(self[node][index],self.I):
                        warnings.warn(f"Wrong indices for particle state passthrough in node {node}.")
                        return False

                # checking if the vacuum state is passed along
                index = tuple(-1 for _ in range(self[node].ndim - 2)) + (slice(0,self.D),slice(0,self.D))
                if not np.allclose(self[node][index],self.I):
                    warnings.warn(f"Wrong indices for vacuum state passthrough in node {node}.")
                    return False

                # checking if a particle state is passed along a passive edge
                for parent,child in itertools.product(self.tree.pred[node],self.G.adj[node]):
                    if child in self.tree.succ[node] or child in self.tree.pred[node]: continue

                    index = tuple(
                        0 if _ in (self.G[node][parent][0]["legs"][node],self.G[node][child][0]["legs"][node])
                        else -1
                        for _ in range(self[node].ndim - 2)
                    ) + (slice(0,self.D),slice(0,self.D))
                    if np.allclose(self[node][index],self.I):
                        warnings.warn(f"Particle state passed along passive edge ({node},{child}).")
                        return False

        return True

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
    def from_graphs(cls,G:nx.MultiGraph,tree:nx.DiGraph,check_tree:bool=True,sanity_check:bool=False):
        """
        Initialisation from a graph `G` that contains PEPO tensors, and a tree `tree` that
        determines the graph traversal of the finite state automaton.
        """
        # inferring physical dimension
        D = tuple(T.shape[-1] for node,T in G.nodes(data="T"))[0]
        # inferring root node
        root = sorted(tuple(tree.nodes),key=lambda x:len(tree.pred[x]))[0]
        # inferring virtual dimension
        chi = tuple(T.shape[0] for node,T in G.nodes(data="T"))[0]

        # initialising the new PEPO
        op = cls(D=D)
        op.G = G
        op.tree = tree
        op.root = root
        op.chi = chi
        op.check_tree = check_tree

        if sanity_check: assert op.intact
        return op

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
        self.check_tree:bool=True
        """False if this PEPO is the result of a summation. Means that the tree traversal checks in `self.intact` are disabled."""
        # TODO: I don't like that I have to disable the tree traversal checks; maybe find a workaround?

        return

    def __getitem__(self,node:int) -> np.ndarray:
        """
        Subscripting with a node gives the tensor at that node.
        """
        if not self.G.has_node(node): raise ValueError(f"Node {node} not present in graph.")

        return self.G.nodes[node]["T"]

    def __setitem__(self,node:int,T:np.ndarray) -> None:
        """
        Changing tensors directly.
        """
        if not self.G.has_node(node): raise ValueError(f"Node {node} not present in graph.")

        if "T" in self.G.nodes[node].keys():
            if not (T.ndim == self.G.nodes[node]["T"].ndim or T.ndim == len(self.G.adj[node]) + 2):
                # first checks against previous tensor, second checks against number of legs that are necessary in the given graph
                raise ValueError("Attempting to set site tensor with wrong number of legs.")
        else:
            # I do not check the dimensions of the tensor here because the dimensions are different
            # from the above cases, while the PEPO is constructed. It is advised to check self.intact
            # after construction of the PEPO.
            pass

        self.G.nodes[node]["T"] = T

        return

    def __matmul__(self,psi:Union[PEPS,np.ndarray]) -> Union[PEPS,np.ndarray]:
        """
        Action of the operator on the state `psi`.
        """

        if isinstance(psi,self.__class__):
            # TODO: implement; what I should do here is what Gray is doing in Sci. Adv. 10, eadk4321 (2024) (https://doi.org/10.1126/sciadv.adk4321)
            raise NotImplementedError("PEPO action on PEPO is not yet implemented.")

        if isinstance(psi,PEPS):
            # TODO: implement; what I should do here is what Gray is doing in Sci. Adv. 10, eadk4321 (2024) (https://doi.org/10.1126/sciadv.adk4321)
            raise NotImplementedError("PEPO action on PEPS is not yet implemented.")

        if isinstance(psi,np.ndarray):
            # sanity check
            if not psi.ndim == 1: raise ValueError("psi must be a vector.")
            if not psi.shape[0] == self.D ** self.nsites: raise ValueError(f"psi has the wrong number of components. Expected {self.D ** self.nsites}, got {psi.shape[0]}.")

            # re-shaping. Order of sites will be determined by the order in which self.G.nodes() iterates through the graph
            psi = np.reshape(psi,newshape=[self.D for _ in range(self.nsites)])

            # enumerating the edges in the graph
            for i,nodes in enumerate(self.G.edges()):
                node1,node2 = nodes
                self.G[node1][node2][0]["label"] = i

            args = ()
            N_edges = self.G.number_of_edges()
            # assembling einsum arguments for the operator
            for i,nodeT in enumerate(self.G.nodes(data="T")):
                node,T = nodeT
                legs = [None for _ in range(T.ndim-2)] + [N_edges + i,N_edges + self.G.number_of_nodes() + i] # last two indices are the physical legs
                for _,neighbor,edge_label in self.G.edges(nbunch=node,data="label"):
                    legs[self.G[node][neighbor][0]["legs"][node]] = edge_label
                args += (T,tuple(legs),)

            # einsum arguments for the state
            args += (psi,tuple(range(N_edges + self.G.number_of_nodes(),N_edges + 2 * self.G.number_of_nodes())))

            out_legs = tuple(range(N_edges,N_edges + self.G.number_of_nodes()))
            psi = np.einsum(*args,out_legs,optimize=True)
            psi = psi.flatten()

            return psi

        raise ValueError("PEPO.__matmul__ not implemented for type " + str(type(psi)) + ".")

    def __add__(lhs,rhs):
        """
        Addition of two PEPOs. The bond dimension of the new operator
        is the sum of the two old bond dimensions.
        """
        # notice that lhs == self !!!

        # sanity check
        if not nx.utils.nodes_equal(lhs.G.nodes(),rhs.G.nodes()): raise ValueError(f"Operands have different geometries; nodes do not match.")
        if not nx.utils.edges_equal(lhs.G.edges(),rhs.G.edges()): raise ValueError(f"Operands have different geometries; edges do not match.")
        if not lhs.D == rhs.D: raise ValueError("Operands must have the same physical dimensions.")
        if not nx.utils.graphs_equal(lhs.tree,rhs.tree): warnings.warn("The trees of the operands are not the same. This is not a problem (as of 29th of January).")

        if not same_legs(lhs.G,rhs.G):
            # permute dimensions of rhs to make both PEPOs compatible
            raise NotImplementedError

        res = PEPO(D=lhs.D)
        res.chi = lhs.chi + rhs.chi
        res.root = lhs.root
        res.tree = lhs.tree
        res.check_tree = False # TODO: this is unelegant, and could be (more or less) easily avoided; see TODO in README

        # graph for the result with correct legs and sizes
        res.G = res.prepare_graph(lhs.G)

        for node in res.G.nodes():
            shape = tuple(res.chi for _ in res.G.adj[node]) + (res.D,res.D)
            T = np.zeros(shape=shape,dtype=np.complex128)
            index_lhs = tuple(slice(0,lhs.chi) for _ in res.G.adj[node]) + (slice(0,res.D),slice(0,res.D))
            index_rhs = tuple(slice(lhs.chi,res.chi) for _ in res.G.adj[node]) + (slice(0,res.D),slice(0,res.D))
            T[index_lhs] = lhs.G.nodes[node]["T"]
            T[index_rhs] = rhs.G.nodes[node]["T"]
            res.G.nodes[node]["T"] = T

        if not res.intact:
            raise RuntimeError("PEPO not intact.")

        return res

    def __repr__(self) -> str:
        return f"Hamiltonian on {self.nsites} sites with Hilbert space of size {self.D} at each. Bond dimension {self.chi}. PEPO is " + ("intact." if self.intact else "not intact.")

class PauliPEPO(PEPO):
    """
    Tensor product operators on spin systems,
    composed of Pauli operators.
    """
    X = np.array([[0,1],[1,0]],dtype=np.complex128)
    """Pauli $X$-matrix."""
    Y = np.array([[0,-1j],[1j,0]],dtype=np.complex128)
    """Pauli $Y$-matrix."""
    Z = np.array([[1,0],[0,-1]],dtype=np.complex128)
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
            warnings.warn("Physical dimension unequal to two.")
            return False

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

def print_operator_chain(op:PEPO,chain:dict[int:tuple],sanity_check:bool=False) -> None:
    """
    Prints the given operator chain.
    """
    if sanity_check: assert op.intact

    for node in chain.keys():
        print(f"op[{node}] : \n",op[node][chain[node]])

    return

if __name__ == "__main__":
    pass
