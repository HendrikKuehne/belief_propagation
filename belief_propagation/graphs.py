"""
Creation of various graphs.
"""

__all__ = [
    "k_wheel",
    "regular_graph",
    "bipartite_regular_graph",
    "short_loop_graph",
    "loop_capped_graph",
    "global_loop",
    "tree",
    "hex",
    "heavyhex",
    "grid",
    "line",
]

import itertools

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def k_wheel(k: int) -> nx.MultiGraph:
    """
    A [regular polygon](https://en.wikipedia.org/wiki/Regular_polygon)
    with `k` edges and a node in the middle, s.t. all vertices of the
    regular polygon are connected to the center node. Effectively a
    wheel with `k` spokes.
    """
    if k == 1: raise ValueError("k = 1 is undefined.")

    if k == 2: return nx.MultiGraph(incoming_graph_data=(
        (1, 2), (0, 1), (0, 2)
    ))

    edges = [
        (i + 1, 1 + int((i + 1) % k))
        for i in range(k)
    ] + [
        (i + 1, 0) for i in range(k)
    ]
    return nx.MultiGraph(incoming_graph_data=edges)


def regular_graph(
        nNodes: int,
        D: int,
        maxiter: int = 1000,
        verbose: bool=False
    ) -> nx.MultiGraph:
    """
    Generates a `D`-regular graph with `nNodes` nodes. WORK IN PROGRESS;
    this algorithm might not terminate, which is why I have included an
    ugly brake that re-initializes the graph and starts again. Off the
    top of my head I don't know how to generate a D-regular graph, and
    it is not as important right now to look this up.
    """
    # Sanity check.
    if nNodes < D + 1 or (nNodes * D) % 2 == 1:
        raise ValueError(
            f"There is no {D}-regular graph with {nNodes} nodes."
        )

    # Defining edges.
    stubs = D * [node for node in range(nNodes)]
    edges = []
    while len(stubs) > 1:
        node1 = np.random.choice(stubs)
        node2 = np.random.choice(stubs)

        i = 0
        while node1 == node2 or {node1, node2} in edges:
            node2 = np.random.choice(stubs)
            i += 1

            if i >= maxiter:
                if verbose:
                    print("".join((
                        f"Algorithm has not terminated after {maxiter} ",
                        "iterations; starting again."
                    )))
                stubs = D * [node for node in range(nNodes)]
                edges = []
                break

        if i < maxiter:
            stubs.remove(node1)
            stubs.remove(node2)

            edges += [{node1, node2},]

    G = nx.MultiGraph(incoming_graph_data=edges)
    return G


def bipartite_regular_graph(
        nNodes: int,
        D: int,
        maxiter: int = 1000,
        verbose: bool = False
    ) -> nx.MultiGraph:
    """
    Algorithm from Kirkley, 2021 ([Sci. Adv. 7, eabf1211
    (2021)](https://doi.org/10.1126/sciadv.abf1211)), which generates a
    bipartite, regular graph.
    """
    # Blue nodes' labels run from 0 to nNodes, red nodes' labels run from
    # nNodes to 2 * nNodes.
    blue_stubs = D * [node for node in range(nNodes)]
    red_stubs = D * [node + nNodes for node in range(nNodes)]
    edges = []

    while len(blue_stubs) > 0:
        blue_node = np.random.choice(blue_stubs)
        red_node = np.random.choice(red_stubs)

        i = 0
        while {blue_node, red_node} in edges:
            blue_node = np.random.choice(blue_stubs)
            red_node = np.random.choice(red_stubs)

            i += 1
            if i > maxiter:
                if verbose:
                    print("".join((
                        f"Algorithm has not terminated after {maxiter} ",
                        "iterations; starting again."
                    )))
                blue_stubs = D * [node for node in range(nNodes)]
                red_stubs = D * [node + nNodes for node in range(nNodes)]
                edges = []
                break

        if i <= maxiter:
            if len({blue_node,red_node}) == 1:
                print({blue_node, red_node})
            edges += [{blue_node, red_node},]
            blue_stubs.remove(blue_node)
            red_stubs.remove(red_node)

    G = nx.MultiGraph(incoming_graph_data=edges)

    return G


def short_loop_graph(
        nNodes: int,
        D: int,
        p: float = 0,
        verbose: bool = False
    ) -> nx.MultiGraph:
    """
    Algorithm from Kirkley, 2021 ([Sci. Adv. 7, eabf1211
    (2021)](https://doi.org/10.1126/sciadv.abf1211)), which generates a
    network with few short primitive cycles.
    """
    # Sanity check.
    if p > 1 or p < 0:
        raise ValueError("p must be a value between zero and one.")

    # Initial bipartite regular graph.
    biG = bipartite_regular_graph(nNodes, D, verbose=verbose)

    edges = []
    for red_node in np.arange(nNodes, 2*nNodes):
        # Projecting onto the blue nodes.
        for blue1,blue2 in itertools.combinations(biG.adj[red_node], r=2):
            if {blue1, blue2} not in edges: edges += [{blue1, blue2},]
        
        # Removing the red node.
        biG.remove_node(red_node)

    biG.add_edges_from(edges)

    # Removing some of the edges randomly.
    for iRemoval in range(int(p * biG.number_of_edges())):
        iEdge = np.random.randint(low=0, high=biG.number_of_edges())
        edge = list(biG.edges)[iEdge]
        biG.remove_edge(*edge)

    # Extracting the largest connected component.
    largest_cc = max(nx.connected_components(biG), key=len)

    # We need to copy because this removes the freeze of the subgraph.
    return biG.subgraph(largest_cc).copy()


def loop_capped_graph(
        nNodes: int,
        maxlength: int,
        p: float = .5,
        rng: np.random.Generator = np.random.default_rng()
    ) -> nx.MultiGraph:
    """
    Generates a graph that is globally tree-like. This is achieved by
    constructing a tree composed of clusters of nodes, where each
    cluster obeys the cycle maximum length. The parameter `p` determines
    cluster connectivity. For `p=1`, clusters are fully connected; for
    `p=0`, a tree is returned.
    """
    if p == 0: return tree(nNodes)

    # Initialization.
    iNode = 0
    clusters = []

    nodes = set()
    # Generating clusters.
    while len(nodes) < nNodes:
        cluster = [
            (iNode + i, iNode + j)
            for i,j in itertools.combinations(range(maxlength), r=2)
            if rng.uniform() <= p
        ]
        if len(cluster) == 0: continue
        nodes = nodes.union(*[set(edge) for edge in cluster])
        iNode += maxlength
        clusters += [cluster,]

    # Adding all clusters to a graph and connecting them tree-like.
    G = nx.MultiGraph()
    G.add_edges_from(clusters.pop(0))

    for cluster in clusters:
        old_docking_node = rng.choice(np.array(G.nodes()))
        new_docking_node = rng.choice(
            list(set().union(*[set(edge) for edge in cluster]))
            )
        G.add_edges_from(cluster + [(old_docking_node, new_docking_node,)])

        # Do we add a new edge or simply merge an existing node and a new one?
        if rng.uniform() < .5:
            G = nx.contracted_edge(
                G = G,
                edge = (old_docking_node, new_docking_node),
                self_loops = False
            )

    # Extracting the largest connected component.
    largest_cc = max(nx.connected_components(G), key=len)

    # We need to copy the graph because this removes the freeze of the
    # subgraph.
    return G.subgraph(largest_cc).copy()


def global_loop(
        global_cycle_length: int,
        nNodes: int,
        maxlength: int,
        p: float = .5,
        rng: np.random.Generator = np.random.default_rng()
    ) -> nx.MultiGraph:
    """
    Generates a graph that is globally tree-like. This is achieved by constructing
    clusters of nodes, where each cluster obeys the cycle maximum length. The clusters
    are attached to a global loop.
    """
    # Initialization.
    clusters = [[
        (i, (i+1) % global_cycle_length)
        for i in range(global_cycle_length)
    ],]
    iNode = global_cycle_length

    if maxlength > 2:
        nodes = {i for i in range(global_cycle_length)}
        # Generating clusters.
        while len(nodes) < nNodes + global_cycle_length:
            cluster = [
                (iNode + i, iNode + j)
                for i,j in itertools.combinations(range(maxlength), r=2)
                if rng.uniform() <= p
            ]
            if len(cluster) == 0: continue
            nodes = nodes.union(*[
                set(edge)
                for edge in cluster
            ])
            iNode += maxlength
            clusters += [cluster,]

        # Adding all clusters to a graph and connecting them tree-like.
        G = nx.MultiGraph()
        G.add_edges_from(clusters.pop(0))

        for cluster in clusters:
            old_docking_node = rng.choice(list(G.nodes()))
            new_docking_node = rng.choice(list(
                set().union(*[
                    set(edge)
                    for edge in cluster])
                ))
            G.add_edges_from(cluster + [(old_docking_node, new_docking_node),])

            # Do we add a new edge or simply merge the existing node and the
            # new one at a node?
            if rng.uniform() < .5:
                G = nx.contracted_edge(
                    G=G,
                    edge=(old_docking_node, new_docking_node),
                    self_loops=False
                )

        # Extracting the largest connected component.
        largest_cc = max(nx.connected_components(G), key=len)

        return G.subgraph(largest_cc).copy()
        # We need to copy because this removes the freeze of the subgraph.

    else:
        G = nx.MultiGraph(clusters[0])
        # Adding nodes in a tree-like fashion.
        while G.number_of_nodes() < nNodes + global_cycle_length:
            docking_node = rng.choice(list(G.nodes()))
            G.add_edge(docking_node, G.number_of_nodes())

        return G


def tree(
        nNodes: int,
        rng: np.random.Generator = np.random.default_rng()
    ) -> nx.MultiGraph:
    """
    Generates a tree by appending nodes at random to the tree.
    """
    not_connected = [i for i in range(1, nNodes)]
    connected = [0]
    G = nx.MultiGraph()
    G.add_node(0)

    while len(not_connected) > 0:
        node = rng.choice(not_connected)
        neighbor = rng.choice(connected)
        G.add_edge(node,neighbor)
        connected += [node,]
        not_connected.remove(node)

    return G


def hex(m: int, n:int) -> nx.MultiGraph:
    """
    Hexagonal graph.
    """
    G = nx.hexagonal_lattice_graph(
        m=m,
        n=n,
        create_using=nx.MultiGraph,
        with_positions=False
    )

    # Removing the pos key.
    for node in G.nodes():
        try:
            del G.nodes[node]["pos"]
        except KeyError:
            continue

    # Re-labeling nodes.
    mapping = {label: i for i, label in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)

    return G


def heavyhex(m: int,n: int) -> nx.MultiGraph:
    """
    Heavy-hex graph, as defined in [Phys. Rev. X 10, 011022
    (2020)](https://doi.org/10.1103/PhysRevX.10.011022).
    """
    G = hex(m=m, n=n)
    N = G.number_of_nodes()

    edges_to_add = ()
    edges_to_remove = ()
    # Adding qubits to the edges.
    for node1, node2 in G.edges():
        edges_to_add += ((node1, N), (N, node2))
        edges_to_remove += ((node1, node2),)
        N += 1
    G.add_edges_from(edges_to_add)
    G.remove_edges_from(edges_to_remove)

    return G


def grid(m: int, n: int) -> nx.MultiGraph:
    """
    Creates a grid graph with `m` rows and `n` columns.
    """
    G = nx.grid_2d_graph(m=m, n=n, create_using=nx.MultiGraph)
    # Re-labeling nodes.
    mapping = {(i, j): i * n + j for i in range(m) for j in range(n)}
    G = nx.relabel_nodes(G, mapping)

    return G


def line(N: int) -> nx.MultiGraph:
    """Exactly what you think it is."""
    G = nx.MultiGraph()
    if N == 0: return G
    if N == 1:
        G.add_node(0)
        return G

    G.add_edges_from(tuple((i, i + 1) for i in range(N - 1)))
    return G


# -----------------------------------------------------------------------------
#                   plotting
# -----------------------------------------------------------------------------


def loop_hist(
        G: nx.MultiGraph,
        bin_edges: np.ndarray = ()
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Creates a histogram of the loop lengths of `G`.
    """
    # are we trying to find loops in a tree?
    if nx.is_tree(G):
        if len(bin_edges) == 0:
            return [0,], [0, np.inf]
        else:
            return [0 for i in range(len(bin_edges)-1)], bin_edges

    # investigating the cycles that occur in the network
    cycle_lengths = [len(cycle) for cycle in nx.simple_cycles(G)]
    if len(bin_edges) == 0:
        hist, edges = np.histogram(
            cycle_lengths,
            bins=np.arange(min(cycle_lengths), max(cycle_lengths)+2)
        )
    else:
        hist, edges = np.histogram(cycle_lengths, bins=bin_edges)
    return hist, edges


def plot_loop_hist(G: nx.MultiGraph, show_plot: bool = True) -> plt.Figure:
    """
    Plots the histogram of the loop lengths of `G`.
    """
    hist, edges = loop_hist(G)

    plt.figure("Loop length histogram")
    # investigating the cycles that occur in the network
    plt.bar(edges[:-1], hist, align="edge")
    plt.suptitle(f"Graph with {G.number_of_nodes()} nodes.")
    plt.xlabel("cycle length")
    plt.ylabel("count")
    if show_plot:
        plt.show()
        return None
    return plt.gcf()


if __name__ == "__main__":
    pass