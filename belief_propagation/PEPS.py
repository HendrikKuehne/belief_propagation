"""
Projector-entangled pair states on arbitrary graphs.
"""

import numpy as np
import networkx as nx
import cotengra as ctg
import warnings
import copy

from belief_propagation.utils import network_message_check,crandn,write_exp_bonddim_to_graph,multi_tensor_rank

class PEPS:
    """
    Base class for matrix-product states with arbitrary geometry.
    """

    def toarray(self,sanity_check:bool=False) -> np.ndarray:
        """
        Contracts the MPS using `ctg.einsum`.
        """
        if sanity_check: assert self.intact

        if self.nsites == 1:
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
            legs = [None for i in range(T.ndim-1)] + [N,] # last index is the physical leg
            for _,neighbor,edge_label in self.G.edges(nbunch=node,data="label"):
                legs[self.G[node][neighbor][0]["legs"][node]] = edge_label
            args += (tuple(legs),)
            N += 1

        psi = ctg.einsum(*args,optimize="greedy")
        return psi.flatten()

    def conj(self,sanity_check:bool=False):
        """
        bra to this state's ket: All site tensors are conjugated.
        """
        if sanity_check: assert self.intact

        newG = copy.deepcopy(self.G)

        for node,T in newG.nodes(data="T"): newG.nodes[node]["T"] = T.conj()

        return type(self)(G=newG,sanity_check=sanity_check)

    def legs_dict(self,node,sanity_check:bool=False) -> dict:
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

    def multiply(self,x:float,sanity_check:bool=False) -> None:
        """
        Multiplies all tensors in the PEPS by `x`.
        """
        if sanity_check: assert self.intact

        for node in self.G.nodes(): self.G.nodes[node]["T"] *= x

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
        Checks if the MPS is intact:
        * Is the underlying network message-ready?
        * Is the size of every edge saved?
        * Are the physical legs the last dimension in each tensor?
        * Do the physical legs have the correct sizes?
        """
        # are all the necessary attributes defined?
        assert hasattr(self,"D")
        assert hasattr(self,"G")

        # is the underlying network message-ready?
        if not network_message_check(self.G):
            warnings.warn("Network not intact.")
            return False

        # size attribute given on every edge?
        for node1,node2,data in self.G.edges(data=True):
            if not "size" in data.keys():
                warnings.warn(f"No size saved in edge ({node1},{node2}).")
                return False
            if data["size"] != self[node1].shape[data["legs"][node1]] or data["size"] != self[node2].shape[data["legs"][node2]]:
                warnings.warn(f"Wrong size saved in edge ({node1},{node2}).")
                return False

        # are the physical legs the last dimension in each tensor? Do the tensors have the correct physical dimensions?
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

            if not legs == [T.ndim - 1,]:
                warnings.warn(f"Physical leg is not the last dimension in node {node}.")
                return False

            # correct size of physical leg?
            if not T.shape[-1] == self.D:
                warnings.warn(f"Hilbert space at node {node} has wrong size.")
                return False

        return True

    @classmethod
    def init_random(cls,G:nx.MultiGraph,D:int,chi:int,rng:np.random.Generator=np.random.default_rng(),real:bool=False,bond_dim_strategy:str="uniform",sanity_check:bool=False,**kwargs):
        """
        Initializes a MPS randomly. The virtual bond dimension is `chi`,
        the physical dimension is `D`. Any leg ordering in `G` is not
        incorporated in the MPS that is returned. Bond dimensions are
        initialized using `bond_dim_strategy`; see
        `PEPS.set_bond_dimensions`.
        """
        # random number generation
        if real:
            randn = lambda size: rng.standard_normal(size)
        else:
            randn = lambda size: crandn(size,rng)

        G = cls.prepare_graph(G)

        # determining bond dimensions
        cls.set_bond_dimensions(G=G,bond_dim_strategy=bond_dim_strategy,D=D,max_chi=chi)

        for node in G.nodes:
            # telling the adjacent edges which index they correspond to
            for i,neighbor in enumerate(G.adj[node]):
                G[node][neighbor][0]["legs"][node] = i

            # constructing the shape of the tensor at this site
            dim = [None for i in G.adj[node]] + [D,]
            for i,neighbor in enumerate(G.adj[node]): dim[G[node][neighbor][0]["legs"][node]] = G[node][neighbor][0]["size"]

            # adding the tensor to this node
            G.nodes[node]["T"] = randn(size = dim) / chi**(3/4)

        return cls(G,sanity_check=sanity_check)

    @classmethod
    def init_from_TN(cls,G:nx.MultiGraph,sanity_check:bool=False):
        """
        Initialises a PEPS from a TN by appending dummy physical dimensions
        of size one to the site tensors.
        """
        if sanity_check: assert network_message_check(G)

        newG = cls.prepare_graph(G,keep_legs=True)
        # appending a dummy physical dimension with size one to the tensors in G
        for node in G.nodes:
            newG.nodes[node]["T"] = np.expand_dims(G.nodes[node]["T"],-1)

        for node1,node2 in G.edges(): newG[node1][node2][0]["size"] = newG.nodes[node1]["T"].shape[G[node1][node2][0]["legs"][node1]]

        return cls(G=newG,sanity_check=sanity_check)

    @classmethod
    def Dummy(cls,G:nx.MultiGraph,sanity_check:bool=False):
        """
        Returns a dummy PEPS on graph `G` with physical dimension one.
        """
        G = cls.prepare_graph(G=G)
        # adding tensors
        for node in G.nodes:
            G.nodes[node]["T"] = np.ones(shape = [1 for _ in range(len(G.adj[node])+1)])

        # adding sizes to edges
        for node1,node2 in G.edges(): G[node1][node2][0]["size"] = 1

        return cls(G=G,sanity_check=sanity_check)

    @staticmethod
    def prepare_graph(G:nx.MultiGraph,keep_legs:bool=False,keep_size:bool=False,sanity_check:bool=False) -> nx.MultiGraph:
        """
        Creates a shallow copy of G, and adds the keys `legs`, `trace` and `indices` to the edges.

        To be used in `__init__` of subclasses of `PEPO`: `G` is the graph from which
        the operator inherits it's underlying graph.
        """
        # shallow copy of G
        newG = nx.MultiGraph(G.edges())

        # adding legs attribute to each edge
        for node1,node2,legs in G.edges(data="legs",keys=False):
            newG[node1][node2][0]["legs"] = legs if keep_legs else {}
            newG[node1][node2][0]["trace"] = False
            newG[node1][node2][0]["indices"] = None

            if "size" in G[node1][node2][0].keys() and keep_size:
                newG[node1][node2][0]["size"] = G[node1][node2][0]["size"]

        if not keep_legs:
            for node in newG.nodes:
                # adding to the adjacent edges which index they correspond to
                for i,neighbor in enumerate(newG.adj[node]):
                    newG[node][neighbor][0]["legs"][node] = i

        if sanity_check: assert network_message_check(newG)

        return newG

    @staticmethod
    def set_bond_dimensions(G:nx.MultiGraph,bond_dim_strategy:str,D:int=None,max_chi:int=None) -> None:
        """
        Initializes the bond dimensions in the graph `G` in-place. The
        string `bond_dim_strategy` determines how bond dimensions
        are intialized. There are several options:
        * `None` (default): Bond dimension `D` on every edge.
        * `exp`: Exact solution on trees. Edge size grows
        exponentially with distance from leaf nodes (internally
        calls `belief_propagation.utils.write_exp_size_to_graph`).
        Requires physical dimension `D`.
        * `exp_cutoff`: Same as `exp`, with maximum size `max_chi`.

        `D` is the physical dimension, and `max_chi` is the bond
        dimension cutoff.
        """
        if bond_dim_strategy not in ("exp_cutoff","exp") and D == None:
            raise ValueError("Bond dimension strategy " + bond_dim_strategy + " requires physical dimension.")

        if bond_dim_strategy == "uniform":
            for node1,node2,key in G.edges(keys=True): G[node1][node2][key]["size"] = max_chi
            return

        if bond_dim_strategy == "exp":
            write_exp_bonddim_to_graph(G=G,D=D)
            return

        if bond_dim_strategy == "exp_cutoff":
            if max_chi == None: raise ValueError("Bond dimension strategy " + bond_dim_strategy + " requires cutoff value.")
            write_exp_bonddim_to_graph(G=G,D=D,max_chi=max_chi)
            return

        raise ValueError("Bond dimension strategy " + bond_dim_strategy + " not implemented.")

    def __init__(self,G:nx.MultiGraph,sanity_check:bool=False) -> None:
        """
        Initialisation from a graph that contains site tensors.
        """
        # sanity check
        if sanity_check: assert network_message_check(G)
        if not isinstance(G,nx.MultiGraph): raise TypeError("G must be a MultiGraph.")

        # inferring physical dimension
        self.D:int = tuple(T.shape[-1] for _,T in G.nodes(data="T"))[0]
        """Physical dimension."""

        self.G:nx.MultiGraph = G

        if sanity_check: assert self.intact

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
        if not T.ndim == self.G.nodes[node]["T"].ndim: raise ValueError("Attempting to set site tensor with wrong number of legs.")

        self.G.nodes[node]["T"] = T

        return

    def __repr__(self) -> str:
        out = ""
        digits = int(np.log10(self.nsites))
        out += f"PEPS with {self.nsites} sites."
        #out += "\n  Bond dimensions:"
        #for node1,node2,size in self.G.edges(data="size"): out += "\n    (" + str(node1).zfill(digits) + "," + str(node2).zfill(digits) + f") : size = {size}"
        #out += "\n  Multilinear tensor ranks:"
        #for node in self.G.nodes(): out += "\n    " + str(node).zfill(digits) + f" : {multi_tensor_rank(self[node])}"

        out += "\n  PEPS is " + "intact." if self.intact else "not intact."

        return out

if __name__ == "__main__":
    pass
