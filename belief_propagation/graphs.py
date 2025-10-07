"""
Creation of various graphs.
"""

__all__ = [
    "k_wheel",
    "tree",
    "hex",
    "heavyhex",
    "grid",
    "heavygrid",
    "line",
    "loop",
    "regular_graph",
    "bipartite_regular_graph",
    "min_girth_graph",
    "short_loop_graph",
    "loop_capped_cluster",
    "composed_cluster_graph",
    "global_loop",
]

import itertools
import warnings
from typing import Callable, Any

import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


# -----------------------------------------------------------------------------
#                   Constructing graphs
# -----------------------------------------------------------------------------


def tree(
        N: int,
        rng: np.random.Generator = np.random.default_rng()
    ) -> nx.MultiGraph:
    """
    Generates a tree by appending nodes at random to the tree.
    """
    not_connected = [i for i in range(1, N)]
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


def heavy_graph_decorator(
        smallgraph: Callable[[Any], nx.MultiGraph]
    ) -> Callable[[Any], nx.MultiGraph]:
    """
    Decorator for turning any graph into its heavy version. Inspired by
    the heavy-hexagonal graph from [Phys. Rev. X 10, 011022
    (2020)](https://doi.org/10.1103/PhysRevX.10.011022).
    """
    def make_heavy(*args, **kwargs) -> nx.MultiGraph:
        G = smallgraph(*args, **kwargs)
        if not isinstance(G, nx.MultiGraph): raise ValueError(
            "Function didi not return MultiGraph."
        )

        N = G.number_of_nodes()
        edges_to_add = ()
        edges_to_remove = ()

        # Adding sites on the edges.
        for node1, node2 in G.edges():
            edges_to_add += ((node1, N), (N, node2))
            edges_to_remove += ((node1, node2),)
            N += 1

        G.add_edges_from(edges_to_add)
        G.remove_edges_from(edges_to_remove)

        return G

    return make_heavy


@heavy_graph_decorator
def heavyhex(m: int, n: int) -> nx.MultiGraph:
    """
    Heavy-hex graph, as defined in [Phys. Rev. X 10, 011022
    (2020)](https://doi.org/10.1103/PhysRevX.10.011022).
    """
    G = hex(m=m, n=n)
    return G


def grid(m: int, n: int) -> nx.MultiGraph:
    """
    Rectangular grid with `m x n` unit cells.
    """
    G = nx.grid_2d_graph(m=m+1, n=n+1, create_using=nx.MultiGraph)
    # Re-labeling nodes.
    G = nx.relabel_nodes(
        G=G,
        mapping={node: i for i, node in enumerate(G.nodes)}
    )

    return G


@heavy_graph_decorator
def heavygrid(m: int, n: int) -> nx.MultiGraph:
    G = grid(m=m, n=n)
    return G


def line(N: int) -> nx.MultiGraph:
    """
    Exactly what you think it is. Contains nodes 0 to `N-1` (inclusive).
    """
    G = nx.MultiGraph()

    if N == 0: return G
    if N == 1:
        G.add_node(0)
        return G

    G.add_edges_from(tuple((i, i + 1) for i in range(N - 1)))
    return G


def loop(N: int) -> nx.MultiGraph:
    """
    Exactly what you think it is. Contains nodes 0 to `N-1` (inclusive).
    """
    G = line(N=N)
    G.add_edge(0, N-1)
    return G


def k_wheel(k: int) -> nx.MultiGraph:
    """
    A [regular polygon](https://en.wikipedia.org/wiki/Regular_polygon)
    with `k` edges and a node in the middle, s.t. all vertices of the
    regular polygon are connected to the center node. Effectively a
    wheel with `k` spokes. The center node is labeled `0`.
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
        N: int,
        D: int,
        maxiter: int = 1000,
        rng: np.random.Generator = np.random.default_rng(),
        verbose: bool=False,
    ) -> nx.MultiGraph:
    """
    Generates a `D`-regular graph with `N` nodes. WORK IN PROGRESS;
    this algorithm might not terminate, which is why I have included an
    ugly brake that re-initializes the graph and starts again. Off the
    top of my head I don't know how to generate a D-regular graph, and
    it is not as important right now to look this up.
    """
    # Sanity check.
    if N < D + 1 or (N * D) % 2 == 1:
        raise ValueError(
            f"There is no {D}-regular graph with {N} nodes."
        )

    # Defining edges.
    stubs = D * [node for node in range(N)]
    edges = []
    while len(stubs) > 1:
        node1 = rng.choice(stubs)
        node2 = rng.choice(stubs)

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
                stubs = D * [node for node in range(N)]
                edges = []
                break

        if i < maxiter:
            stubs.remove(node1)
            stubs.remove(node2)

            edges += [{node1, node2},]

    G = nx.MultiGraph(incoming_graph_data=edges)
    return G


def bipartite_regular_graph(
        N: int,
        D: int,
        maxiter: int = 1000,
        rng: np.random.Generator = np.random.default_rng(),
        verbose: bool = False,
    ) -> nx.MultiGraph:
    """
    Algorithm from Kirkley, 2021 ([Sci. Adv. 7, eabf1211 (2021)](https://doi.org/10.1126/sciadv.abf1211)),
    which generates a bipartite, regular graph.
    """
    # Blue nodes' labels run from 0 to nNodes, red nodes' labels run from
    # nNodes to 2 * nNodes.
    blue_stubs = D * [node for node in range(N)]
    red_stubs = D * [node + N for node in range(N)]
    edges = []

    while len(blue_stubs) > 0:
        blue_node = rng.choice(blue_stubs)
        red_node = rng.choice(red_stubs)

        i = 0
        while {blue_node, red_node} in edges:
            blue_node = rng.choice(blue_stubs)
            red_node = rng.choice(red_stubs)

            i += 1
            if i > maxiter:
                if verbose:
                    print("".join((
                        f"Algorithm has not terminated after {maxiter} ",
                        "iterations; starting again."
                    )))
                blue_stubs = D * [node for node in range(N)]
                red_stubs = D * [node + N for node in range(N)]
                edges = []
                break

        if i <= maxiter:
            edges += [{blue_node, red_node},]
            blue_stubs.remove(blue_node)
            red_stubs.remove(red_node)

    G = nx.MultiGraph(incoming_graph_data=edges)

    return G


def min_girth_graph(
        N: int = None,
        g: int = 5,
        max_edges: int = None,
        max_D: int = np.inf,
        rng: np.random.Generator = np.random.default_rng(),
        G_init: nx.MultiGraph = None,
    ) -> nx.MultiGraph:
    """
    Generates a graph with minimum loop length `g` and `N` nodes. This
    is done by randomly choosing edges `(u, v)` and adding them (I) if
    they do not create a loop, or (II) if they create a loop that is
    longer than or equal to `g`. At most `max_edges` are added (default:
    all possible edges are added). The edge `(u, v)` is only added, if
    none of the nodes exceeds degree `max_D`.

    An initial graph can be given, in which case edges are added to it
    according to the procedure described above. Returns a copy.
    """

    if G_init is None:
        if N is None: raise ValueError("No number of nodes given.")
        G = nx.MultiGraph()
        G.add_nodes_from(range(N))
    else:
        G = nx.MultiGraph(incoming_graph_data=G_init)

    all_possible_edges = [
        (u, v)
        for u, v in itertools.combinations(G, r=2)
        if not G.has_edge(u, v)
    ]

    # Shuffling the available edges, to achieve even density in the graph.
    rng.shuffle(all_possible_edges)

    nodes_added = 0
    for (u, v) in all_possible_edges:
        # Adding edge (u, v) will create a loop of length dist(u, v) + 1.
        # Starting from node u, we conduct an edge BFS from node v until depth
        # g - 2. If node v is found, adding the edge (u, v) would create a
        # loop with length smaller than g - which we do not want. We thus move
        # on to the next edge. If node v is not found, edge (u, v) can be
        # safely added.
        v_found = False
        for x, y in nx.bfs_edges(G=G, source=u, depth_limit=g-2):
            if x == v or y == v:
                v_found = True
                break

        if not v_found and (len(G.adj[u]) < max_D) and (len(G.adj[v]) < max_D):
            G.add_edge(u, v)
            nodes_added += 1

        if nodes_added == max_edges: break

    # Extracting the largest connected component.
    largest_cc = max(nx.connected_components(G), key=len)

    # We need to copy because this removes the freeze of the subgraph.
    return G.subgraph(largest_cc).copy()


def short_loop_graph(
        N: int,
        D: int,
        p: float = 0,
        rng: np.random.Generator = np.random.default_rng(),
        verbose: bool = False
    ) -> nx.MultiGraph:
    """
    Algorithm from [Sci. Adv. 7, eabf1211 (2021)](https://doi.org/10.1126/sciadv.abf1211),
    which generates a network with few short primitive cycles.
    """
    # Sanity check.
    if p > 1 or p < 0:
        raise ValueError("p must be a value between zero and one.")

    # Initial bipartite regular graph.
    biG = bipartite_regular_graph(N=N, D=D, rng=rng, verbose=verbose)

    edges = []
    for red_node in np.arange(N, 2 * N):
        # Projecting onto the blue nodes.
        for blue1,blue2 in itertools.combinations(biG.adj[red_node], r=2):
            if {blue1, blue2} not in edges: edges += [{blue1, blue2},]
        
        # Removing the red node.
        biG.remove_node(red_node)

    biG.add_edges_from(edges)

    # Removing some of the edges randomly.
    for _ in range(biG.number_of_edges()):
        if p >= rng.uniform(low=0, high=1):
            iEdge = rng.integers(low=0, high=biG.number_of_edges())
            edge = list(biG.edges)[iEdge]
            biG.remove_edge(*edge)

    # Extracting the largest connected component.
    largest_cc = max(nx.connected_components(biG), key=len)

    # We need to copy because this removes the freeze of the subgraph.
    return biG.subgraph(largest_cc).copy()


def loop_capped_cluster(
        N: int,
        maxlength: int,
        p: float = .5,
        rng: np.random.Generator = np.random.default_rng()
    ) -> nx.MultiGraph:
    """
    Generates a cluster of about `N` nodes. The cluster is constructed
    by weaving together loops of length `maxlength`. Note that this
    means that the cluster may end up containing loops longer than
    `maxlength`. The probability `p` controls the weaving of loops: If
    `p = 0`, the cluster consists of many loops that only intersect in
    single nodes. For increasing values of `p`, loops are increasingly
    intertwined.

    Intended to construct clusters of nodes that can be contained in the
    neighborhoods of [Sci. Adv. 7, eabf1211 (2021)](https://doi.org/10.1126/sciadv.abf1211),
    where `maxlength = r + 2`.
    """
    if maxlength < 3: return tree(N=N, rng=rng)

    # Sanity check.
    if not 0 <= p and p <= 1: raise ValueError("p is not a probability.")

    if p == 1:
        # As p increases, loops are more and more interwoven - and more
        # numerous. In the limit p = 1, the graph contains all possible
        # paths of length maxlength among N nodes that intersect 0.
        G = nx.MultiGraph(incoming_graph_data=(
            (u, v) for u, v in itertools.combinations(range(N), r=2)
        ))
        return G

    G = loop(N=maxlength)

    # The cluster will be constructed by waving together loops. The loops all
    # have length maxlength, but their nodes are drawn at random from both (I)
    # the nodes already present in the graph and (II) new nodes.

    retries = 1
    while G.number_of_nodes() < N:
        # The loop that will be woven into the existing cluster.
        new_loop = loop(N=maxlength)

        # Re-labeling nodes in the loop.
        avail_nodes = list(G.nodes)
        mapping = {}
        for i, node in enumerate(new_loop.nodes()):
            if rng.uniform(low=0, high=1, size=1).item() > p:
                newlabel = G.number_of_nodes() + i
            else:
                newlabel = rng.choice(avail_nodes, size=1).item()
                avail_nodes.remove(newlabel)
            mapping[node] = newlabel

        if p == 0:
            # If the probability, that an existing node from the cluster is
            # woven into the new loop, is zero, the new loops will never be
            # connected to the cluster. To prevent this, one node in the new
            # loop is replaced with one node in the cluster.
            dock_cluster = rng.choice(G.nodes(), size=1).item()
            dock_newloop = rng.choice(new_loop.nodes(), size=1).item()
            mapping[dock_newloop] = dock_cluster

        new_loop = nx.relabel_nodes(
            G=new_loop,
            mapping=mapping,
            copy=True
        )

        if len(set(G.nodes()).intersection(new_loop.nodes())) == 0:
            # The new loop and the cluster have no nodes in common. This would
            # lead to the cluster being disconnected; let's try again.

            if retries % 10000 == 0:
                warnings.warn(
                    "".join((
                        f"loop_capped_cluster is trying for the {retries}th ",
                        "time to weave a new loop into the existing cluster. ",
                        "Did you set the connection probability too low? ",
                        f"Current value: p = {p}."
                    )),
                    RuntimeWarning,
                )

            retries += 1
            continue

        # Adding edges from this loop to the graph.
        edges_to_add = [(node1, node2, 0) for node1, node2 in new_loop.edges()]
        G.add_edges_from(edges_to_add)

    for node1, node2, key in G.edges(keys=True):
        if node1 == node2:
            print("Self-loop")
        if key != 0:
            print("Double edge")

    return G


def composed_cluster_graph(
        N: int,
        N_cluster: int,
        r: int,
        p_cluster: float = .5,
        p_connect: float = .5,
        G_init: nx.MultiGraph = None,
        rng: np.random.Generator = np.random.default_rng()
    ) -> nx.MultiGraph:
    """
    Generates a graph composed of clusters of `N_cluster` nodes each,
    where each cluster is constructed by weaving together loops of
    length `r+2`. The parameter `p_cluster` determines cluster
    connectivity; see `loop_capped_cluster` for details.

    Clusters of `N_cluster` nodes each are added to `G_init`, until `N`
    nodes have been added in total. By default, `G_init` is a graph with
    one node. With probability `p_connect`, clusters are added to the
    graph by replacing an existing node. Otherwise, they are appended.

    Nodes in `G_init` may be re-labelled or replaced, s.t. the nodes in
    the returned graph do - in the general case - not correspond to the
    nodes in `G_init`. `G_init` is not changed.

    Intended to create graphs that can be nicely decomposed into
    neighborhoods, according to the method presented in
    [Sci. Adv. 7, eabf1211 (2021)](https://doi.org/10.1126/sciadv.abf1211).
    """

    if G_init is not None:
        G = nx.MultiGraph(G_init)
    else:
        G = nx.MultiGraph()
        G.add_node(0)

    # Re-labeling nodes to be consecutive integers. The following code assumes
    # that this is the case. This means that the naming of nodes in the
    # returned graph my be different to G_init; if this turns out to be a
    # problem later, this could (probably) be circumvented easily.
    G = nx.relabel_nodes(
        G=G,
        mapping={
            u: i
            for i, u in enumerate(G.nodes())
        },
        copy=True
    )

    clusters: list[nx.MultiGraph] = []
    nodes = G.number_of_nodes()

    # Generating clusters.
    while nodes < N:
        cluster = loop_capped_cluster(
            N=N_cluster,
            maxlength=r+2,
            p=p_cluster,
            rng=rng
        )

        cluster = nx.relabel_nodes(
            G=cluster,
            mapping={
                u: nodes + i
                for i, u in enumerate(cluster.nodes())
            },
            copy=True
        )

        nodes += cluster.number_of_nodes()
        clusters.append(cluster)

    # Adding all clusters to the graph.
    for cluster in clusters:

        if rng.uniform() < p_connect:
            # The cluster will replace a node in the graph.
            repl_node = rng.choice(list(G.nodes()))
            dock_nodes = rng.choice(
                list(cluster.nodes()),
                size=len(G.adj[repl_node]),
                replace=True
            )

            G = nx.union(G, cluster)
            edges_to_add = ()
            for dock_G, dock_cl in zip(G.adj[repl_node], dock_nodes):
                edges_to_add += ((dock_G, dock_cl.item()),)
            G.add_edges_from(edges_to_add)
            G.remove_node(repl_node)

        else:
            # The cluster will be appended to the graph.
            old_docking_node = rng.choice(list(G.nodes()))
            new_docking_node = rng.choice(list(cluster.nodes()))

            G = nx.union(G, cluster)
            G.add_edge(old_docking_node, new_docking_node)

            # Do we add a new edge or merge an existing node and a new one?
            if rng.uniform() < 0.5:
                G = nx.contracted_edge(
                    G = G,
                    edge = (old_docking_node, new_docking_node),
                    self_loops = False,
                )

    # Re-labeling nodes to be consecutive integers.
    G = nx.relabel_nodes(
        G=G,
        mapping={
            u: i
            for i, u in enumerate(G.nodes())
        },
        copy=True
    )

    return G


# -----------------------------------------------------------------------------
#                   Plotting
# -----------------------------------------------------------------------------


def loop_hist(
        G: nx.MultiGraph,
        bin_edges: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Creates a histogram of the loop lengths of `G`. Returns counts and
    bin edges.
    """

    # Are we trying to find loops in a tree?
    if nx.is_tree(G):
        if bin_edges is None:
            return [0,], [0, np.inf]
        else:
            return [0 for i in range(len(bin_edges)-1)], bin_edges

    # Cycle basis finding is not implemented for multigraphs.
    G_single = nx.Graph(incoming_graph_data=G)

    # Investigating the cycles that occur in the network.
    cycle_lengths = [len(cycle) for cycle in nx.cycle_basis(G_single)]
    if bin_edges is None:
        hist, edges = np.histogram(
            cycle_lengths,
            bins=np.arange(min(cycle_lengths), max(cycle_lengths)+2)
        )
    else:
        hist, edges = np.histogram(cycle_lengths, bins=bin_edges)

    return hist, edges


def plot_loop_hist(
        G: nx.MultiGraph,
        show: bool = True,
        ax: mpl.axes = None
    ) -> None:
    """
    Plots the histogram of the loop lengths of `G`.
    """
    hist, edges = loop_hist(G=G)

    if ax is None: ax = plt.gca()
    ax.bar(edges[:-1], hist, align="edge")
    ax.set_xticks(edges[:-1])
    ax.set_yticks(np.arange(0, hist.max() + 1, 1))

    if show: plt.show()
    return


if __name__ == "__main__":
    pass