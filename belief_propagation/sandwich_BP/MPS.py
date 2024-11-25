"""
Matrix product states on arbitrary graphs.
"""

import numpy as np
import networkx as nx
import cotengra as ctg
import warnings

from belief_propagation.utils import network_message_check,crandn

class MPS:
    """
    Base class for matrix-product states on spin systems with arbitrary geometry.
    """

    D = 2
    """
    Physical dimension; so far, only spin-1/2 is implemented.
    """

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

    def intact_check(self) -> bool:
        """
        Checks if the MPS is intact:
        * Are the legs in the graph labeled correctly?
        """
        # are all the necessary attributes defined?
        assert hasattr(self,"D")
        assert hasattr(self,"G")

        # two legs in every edge's legs attribute?
        for node1,node2,key in self.G.edges(keys=True):
            if self.G[node1][node2][key]["trace"]:
                # trace edge
                if len(self.G[node1][node2][key]["indices"]) != 2:
                    warnings.warn(f"Wrong number of legs in trace edge ({node1},{node2},{key}).")
                    return False
            else:
                # default edge
                if len(self.G[node1][node2][key]["legs"].keys()) != 2:
                    warnings.warn(f"Wrong number of legs in edge ({node1},{node2},{key}).")
                    return False

        # correct leg indices around each node?
        for node in self.G.nodes:
            legs = [leg for leg in range(self.G.nodes[node]["T"].ndim)]
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

            if not legs == [self.G.nodes[node]["T"].ndim - 1,]:
                warnings.warn(f"Physical leg is not the last dimension in node {node}.")
                return False

        return True

    @classmethod
    def init_random(cls,G:nx.MultiGraph,chi:int,rng:np.random.Generator=np.random.default_rng(),real:bool=True,sanity_check:bool=False):
        """
        Initializes a MPS randomly. The virtual bond dimension is `chi`.
        Any leg ordering in `G` is not incorporated in the MPS that
        is returned.
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
            dim = nLegs * [chi] + [cls.D,]
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

    @staticmethod
    def prepare_graph(G:nx.MultiGraph,keep_legs:bool=False,keep_size:bool=False) -> nx.MultiGraph:
        """
        Creates a shallow copy of G, and adds the keys `legs`, `trace` and `indices` to the edges.
        The `size` key is retained, if present.

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

        self.G = G

        if sanity_check: assert self.intact_check()

        return

if __name__ == "__main__":
    pass
