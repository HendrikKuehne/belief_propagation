"""
Matrix product states on arbitrary graphs.
"""

import numpy as np
import networkx as nx
import cotengra as ctg
import warnings
import copy

from belief_propagation.utils import network_message_check,crandn

class PEPS:
    """
    Base class for matrix-product states with arbitrary geometry.
    """

    def intact_check(self) -> bool:
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
            if data["size"] != self.G.nodes[node1]["T"].shape[data["legs"][node1]] or data["size"] != self.G.nodes[node2]["T"].shape[data["legs"][node2]]:
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

    def to_dense(self,sanity_check:bool=False) -> np.ndarray:
        """
        Contracts the MPS using `ctg.einsum`.
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
        if sanity_check: assert self.intact_check()

        newG = copy.deepcopy(self.G)

        for node,T in newG.nodes(data="T"): newG.nodes[node]["T"] = T.conj()

        return type(self)(G=newG,sanity_check=sanity_check)

    def legs_dict(self,node,sanity_check:bool=False) -> dict:
        """
        Returns the `legs` attributes of the adjacent edges of `node`
        in a dictionary structure: `legs_dict[neighbor]` is the
        same as `self.G[node][neighbor][0]["legs"]`.
        """
        if sanity_check: assert self.intact_check()

        val = dict()

        for neighbor in self.G.adj[node]:
            val[neighbor] = self.G[node][neighbor][0]["legs"]

        return val

    def multiply(self,x:float,sanity_check:bool=False) -> None:
        """
        Multiplies all tensors in the PEPS by `x`.
        """
        if sanity_check: assert self.intact_check()

        for node in self.G.nodes(): self.G.nodes[node]["T"] *= x

        return

    @classmethod
    def init_random(cls,G:nx.MultiGraph,D:int,chi:int,rng:np.random.Generator=np.random.default_rng(),real:bool=False,sanity_check:bool=False):
        """
        Initializes a MPS randomly. The virtual bond dimension is `chi`,
        the physical dimension is `D`. Any leg ordering in `G` is not
        incorporated in the MPS that is returned.
        """
        # random number generation
        if real:
            randn = lambda size: rng.standard_normal(size)
        else:
            randn = lambda size: crandn(size,rng)

        G = cls.prepare_graph(G)

        tensors = {}

        for node in G.nodes:
            nLegs = len(G.adj[node])
            dim = nLegs * [chi] + [D,]
            # constructing a new tensor
            T = randn(size = dim) / chi**(3/4)

            # saving the physical tensor
            tensors[node] = T

            # adding the tensor to this node
            G.nodes[node]["T"] = T

            # adding to the adjacent edges which index they correspond to, and the size of this edge
            for i,neighbor in enumerate(G.adj[node]):
                G[node][neighbor][0]["legs"][node] = i
                G[node][neighbor][0]["size"] = T.shape[i]

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
        G = cls.prepare_graph(G=G,keep_legs=True)
        # adding tensors
        for node in G.nodes:
            G.nodes[node]["T"] = np.ones(shape = [1 for _ in range(len(G.adj[node])+1)])

        # adding sizes to edges
        for node1,node2 in G.edges(): G[node1][node2][0]["size"] = 1

        return cls(G=G,sanity_check=sanity_check)

    @staticmethod
    def prepare_graph(G:nx.MultiGraph,keep_legs:bool=False,keep_size:bool=False) -> nx.MultiGraph:
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

        return newG

    def __init__(self,G:nx.MultiGraph,sanity_check:bool=False) -> None:
        """
        Initialisation from a graph that contains site tensors.
        """
        # sanity check
        if sanity_check: assert network_message_check(G)

        # inferring physical dimension
        self.D = tuple(T.shape[-1] for _,T in G.nodes(data="T"))[0]
        """Physical dimension."""

        self.G = G

        if sanity_check: assert self.intact_check()

        return

if __name__ == "__main__":
    pass
