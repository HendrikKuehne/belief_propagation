"""
Functions for network creation and handling.
"""

__all__ = [
    "expose_edge",
    "construct_network",
    "delta_network",
]

import numpy as np
import networkx as nx

from belief_propagation.utils import (
    network_intact_check, network_message_check, crandn, delta_tensor
)

# -----------------------------------------------------------------------------
#                   Manipulating and contracting networks
# -----------------------------------------------------------------------------


def expose_edge(
        G: nx.MultiGraph,
        node1: int,
        node2: int,
        sanity_check: bool = False
    ) -> None:
    """
    Transposes the tensors in `node1` and `node2` such that the edge
    connecting them sums over the trailing virtual dimensions. `G` is
    modified in-place.
    """
    if sanity_check:
        assert network_intact_check(G)
        assert network_message_check(G)
        assert node1 in G.adj[node2]

    # the virtual dimensions, which will be permuted
    axes1 = [None for _ in range(len(G.adj[node1]))]
    """The `axes` argument for `np.transpose` at `node1`."""
    axes2 = [None for _ in range(len(G.adj[node2]))]
    """The `axes` argument for `np.transpose` at `node2`."""

    axes1[-1] = G[node1][node2][0]["legs"][node1]
    axes2[-1] = G[node2][node1][0]["legs"][node2]

    legshift1 = lambda i: i > G[node1][node2][0]["legs"][node1]
    legshift2 = lambda i: i > G[node2][node1][0]["legs"][node2]

    # adding physical dimensions, which will not be permuted
    while len(axes1) < G.nodes[node1]["T"].ndim: axes1 += [len(axes1),]
    while len(axes2) < G.nodes[node2]["T"].ndim: axes2 += [len(axes2),]

    for neighbor in G.adj[node1]:
        if neighbor == node2: continue

        old_leg = G[node1][neighbor][0]["legs"][node1]
        # assembling the axes-argument
        axes1[old_leg - legshift1(old_leg)] = old_leg
        # overwriting the leg of this edge
        G[node1][neighbor][0]["legs"][node1] = old_leg - legshift1(old_leg)
    for neighbor in G.adj[node2]:
        if neighbor == node1: continue

        old_leg = G[node2][neighbor][0]["legs"][node2]
        # assembling the axes-argument
        axes2[old_leg - legshift2(old_leg)] = old_leg
        # overwriting the leg of this edge
        G[node2][neighbor][0]["legs"][node2] = old_leg - legshift2(old_leg)

    # overwriting the legs of the edge (node1,node2)
    G[node1][node2][0]["legs"][node1] = len(G.adj[node1]) - 1
    G[node2][node1][0]["legs"][node2] = len(G.adj[node2]) - 1

    # permuting the legs of u and v such that edge (u,v) corresponds to the
    # trailing index in nodes u and v
    G.nodes[node1]["T"] = np.transpose(G.nodes[node1]["T"],axes1)
    G.nodes[node2]["T"] = np.transpose(G.nodes[node2]["T"],axes2)

    return


# -----------------------------------------------------------------------------
#                   Network creation
# -----------------------------------------------------------------------------


def construct_network(
        G: nx.MultiGraph,
        chi: int = None,
        rng: np.random.Generator = np.random.default_rng(),
        positive: bool = False,
        real: bool = False,
        psd: bool = True,
        tensors: bool = False
    ) -> dict:
    """
    Constructs a tensor network with bond dimension `chi`, where the
    topology is taken from the graph `G`. The graph `G` is manipulated
    in-place. Tensors are only added if `tensors=True` (default:
    `False`). If tensors were added, returns the tensors in a dictionary
    where the nodes are keys, and the tensors are the values.

    The arguments `positive`, `real`, and `psd` control tensor
    initialization.
    * If `positive` is `True`, tensors are drawn from a uniform
    distribution over [0, 1].
    * If `positive` is `False`, tensors are drawn from a normal
    distribution, and contain either real or complex numbers (controlled
    by the `real` argument).
    * If `psd`, tensors are constructed to be positive-semidefinite
    (method inspired by [Cao and Vontobel, 2017 IEEE
    ITW](https://doi.org/10.1109/ITW.2017.8277985)).
    """
    # TODO accept graph argument for chi, s.t. edge sized can be
    # specified for each edge individually. The code can basically be
    # copied from L2BP compression.

    # Sanity check.
    if tensors and chi is None:
        raise ValueError("No virtual bond dimension given.")

    if positive and psd:
        #with tqdm.tqdm.external_write_mode():
        #    warnings.warn(
        #        "".join((
        #            "Options positive=True and psd=True are incompatible. ",
        #            "Defaulting to positive."
        #        )),
        #        UserWarning
        #    )
        pass#psd = False

    if positive and (not real):
        #with tqdm.tqdm.external_write_mode():
        #    warnings.warn(
        #        "".join((
        #            "Options positive=True and real=False are incompatible. ",
        #            "Defaulting to positive."
        #        )),
        #        UserWarning
        #    )
        pass#real = True

    # Random number generation.
    if positive:
        rand = lambda size: rng.uniform(low=0, high=1, size=size)
    elif real:
        rand = lambda size: rng.standard_normal(size=size)
    else:
        rand = lambda size: crandn(size=size, rng=rng)

    for node1, node2, key in G.edges(keys=True):
        if node1 == node2: raise ValueError("Graph contains a self-loop.")
        if key != 0: raise ValueError("graph contains double edges.")

        # Each edge has a "legs" key, whose value is itself a dictionary. The
        # keys are the labels of the adjacent nodes, and their values are the
        # indices of the tensor legs this edge connects.
        G[node1][node2][0]["legs"] = {}
        # Each ede has a "trace" key, which is true if this edge corresponds to
        # a trace over a tensor (i.e. if this edge connects a node to itself).
        G[node1][node2][0]["trace"] = False
        # Each edge has an "indices" key, which holds the legs that the
        # adjacent tensors are summed over as a set (only used for edges that
        # represent a trace).
        G[node1][node2][0]["indices"] = None

    tensor_list = {}

    for node in G.nodes:
        # Adding to the adjacent edges which index they correspond to.
        stumps = list(range(len(G.adj[node])))
        for neighbor in G.adj[node]:
            stump = rng.choice(stumps)
            G[node][neighbor][0]["legs"][node] = stump
            stumps.remove(stump)

        if not tensors: continue

        nLegs = len(G.adj[node])
        dim = nLegs * [chi]

        # Constructing a new tensor.

        if positive:
            T = rand(size=dim)

            # Saving the physical tensor.
            tensor_list[node] = T

        elif psd:
            h = int(np.sqrt(chi))
            if not h**2 == chi:
                raise ValueError("if psd=True, chi must have an integer root.")

            s = rand(size = nLegs * [h,] + [chi,])
            T = np.einsum(
                s, [2*i for i in range(nLegs)] + [2*nLegs,],
                s.conj(), [2*i+1 for i in range(nLegs)] + [2*nLegs,],
                np.arange(2*nLegs)
            ).reshape(dim) / chi**(3/4)

            # Saving the physical tensor.
            tensor_list[node] = s

        else:
            T = rand(size=dim) / chi**(3/4)

            # Saving the physical tensor.
            tensor_list[node] = T

        # Adding the tensor to this node.
        G.nodes[node]["T"] = T

    return tensor_list if tensors else None


def delta_network(G: nx.MultiGraph,chi: int) -> None:
    """
    Constructs a tensor network with bond dimension `chi`, where the
    topology is taken from the graph `G`. Each tensor is a delta-tensor,
    or a unit vector with only one non-zero entry. The graph `G` is
    manipulated in-place.
    """
    for edge in G.edges:
        # each edge has a "legs" key, whose value is itself a dictionary. The
        # keys are the labels of the adjacent nodes, and their values are the
        # indices of the tensor legs this edge connects
        G[edge[0]][edge[1]][0]["legs"] = {}
        # each ede has a "trace" key, which is true if this edge corresponds to
        # the trace of a tensor (i.e. if this edge connects a node to itself)
        G[edge[0]][edge[1]][0]["trace"] = False
        # each edge has an "indices" key, which holds the legs that the
        # adjacent tensors are summed over as a set (only used for edges that
        # represent a trace)
        G[edge[0]][edge[1]][0]["indices"] = None

    for node in G.nodes:
        nLegs = len(G.adj[node])
        dim = nLegs * (chi,)

        # adding the tensor to this node
        G.nodes[node]["T"] = delta_tensor(nLegs, chi)

        # adding to the adjacent edges which index they correspond to
        for i, neighbor in enumerate(G.adj[node]):
            G[node][neighbor][0]["legs"][node] = i

    return

if __name__ == "__main__":
    pass
