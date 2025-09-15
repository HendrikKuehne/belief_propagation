"""
Belief propagation on graphs using neighbor regions. Inspired by [Sci.
Adv. 7, eabf1211 (2021)](https://doi.org/10.1126/sciadv.abf1211).
"""

__all__ = [
    "neighborhood",
    "construct_neighborhoods",
    "contract_neighborhood",
    "plot_neighborhoods",
]

from typing import Union

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import cotengra as ctr

from belief_propagation.utils import (
    network_intact_check,
    network_message_check
)
from belief_propagation.old.networks import merge_edges


# TODO Encapsulate neighborhoods in classes.
def neighborhood(
        G: nx.MultiGraph,
        rootnode: int,
        r: int = 0,
        sanity_check: bool = False
    ) -> tuple[set[frozenset[int]], set[int]]:
    """
    Recursively identifies the neighborhood with connectivity `r` around
    `node`. The neighborhood with connectivity `r` is defined as
    containing loops with length `r+2`.

    Returns a set that contains all edges in the neighborhood, and a set
    that contains all nodes in the neighborhood.
    """
    # sanity check
    if sanity_check: assert network_message_check(G)

    def traverse(
            G: nx.MultiGraph,
            node: int,
            home: int,
            steps: int,
            path: list[set[int]],
            edges_found: set[frozenset[int]],
            nodes_found: set[int]
        ) -> bool:
        """
        Traverses the graph recursively, taking `steps` steps. If a loop
        is found, the nodes and edges along the loop are added to
        `edges_found` and `nodes_found`. This function cannot cross
        existing neighborhoods; if
        `G.nodes[node]["neighborhood"] != None`
        is true, `traverse` immediately returns `False`.
        """
        # Are we back at home?
        if node == home: return True

        # Did we hit an existing neighborhood?
        try:
            if G.nodes[node]["neighborhood"] != None: return False
        except KeyError:
            pass

        if steps > 0:
            # There are steps left to take.
            path_found = False
            for next_node in G.adj[node]:
                if {node, next_node} in path:
                    # We have been here before; every edge should be traversed
                    # only once.
                    continue

                # Depth-first search.
                path.append({node, next_node})

                if traverse(
                    G=G,
                    node=next_node,
                    home=home,
                    steps=steps - 1,
                    path=path,
                    edges_found=edges_found,
                    nodes_found=nodes_found
                ):
                    # The next node is on a path back to home; we need to add
                    # this node and the edges we travelled along
                    for edge in path: edges_found.add(frozenset(edge))
                    nodes_found.add(next_node)
                    path_found = True

                # Backtracking.
                path.pop()

            return path_found

        return False

    edges_found = set()
    nodes_found = {rootnode,}

    for node in G.adj[rootnode]: traverse(
        G=G,
        node=node,
        home=rootnode,
        steps=r+1,
        path=[{rootnode, node},],
        edges_found=edges_found,
        nodes_found=nodes_found
    )

    edges_found = tuple([tuple(edge) for edge in edges_found])
    nodes_found = tuple(nodes_found)

    return edges_found, nodes_found


def construct_neighborhoods(
        G: nx.MultiGraph,
        r: int = 0,
        method: str = "loopy",
        rng: np.random.Generator = np.random.default_rng(),
        sanity_check: bool = False
    ) -> tuple:
    """
    Constructs neighborhoods in the graph `G` with connectivity `r`.
    `G` is modified in-place.

    The argument `method` denotes the method that is used to construct
    the neighborhoods. These methods only differ in the order in which
    the neighborhoods are created, i.e. the order in which nodes from
    the graph are chosen as root nodes. The default case (`method =
    loopy`) is a heuristic; the nodes are sorted based on the number of
    adjacent nodes. In case a single node of the graph is given as
    argument, neighborhoods are created starting at this node and moving
    outwards.
    """

    # TODO the "kirkley" method does not work too well - sometimes, small loops
    # remain outside of any neighborhood. I think this is due to the fact that
    # the focus of Kirkley et Al (https://doi.org/10.1126/sciadv.abf1211) is on
    # neighborhoods that contain edges (they allow nodes to be contained in
    # multiple neighborhoods), while my focus lies on nodes (in my scenario,
    # edges may be left inbetween neighborhoods without any association).

    # The choice of neighborhoods should ensure that as many loops as possible
    # are contained within the neighborhoods. This version of the code achieves
    # this using a (possibly crude) heuristic: We begin neighborhood
    # construction with the nodes that have the largest number of neighbors

    # Sanity check.
    if sanity_check: assert network_message_check(G)

    # Initialization.
    for node in G.nodes():
        G.nodes[node]["neighborhood"] = None

    if method == "loopy" or method == "kirkley":
        # Sorting nodes based on the number of neighbors; construction of
        # neighborhoods begins at the node with the highest degree.
        sorted_node_list = sorted(
            G.nodes(),
            key=lambda node: len(G.adj[node]),
            reverse=True
        )
    elif method in G:
        # Construction of neighborhoods begins at the specified node, and
        # moving outwards afterwards.
        sorted_node_list = [method,]
        method = "kirkley"
    else:
        raise ValueError("".join(("Method ", str(method), " udefined.")))

    neighborhood_list = ()

    def next_node() -> Union[None, int]:
        """
        Helper function that returns the next node to construct a
        neighborhood around. Depends on the `method` argument and the
        nodes that have been encountered already. Should return `None`
        once the entire graph has been visited.
        """
        if method == "loopy":
            # Removing nodes from the sorted node list that are already
            # contained in neighborhoods.
            for (edges, nodes) in neighborhood_list:
                for node in nodes:
                    try: sorted_node_list.remove(node)
                    except ValueError: pass

            if len(sorted_node_list) == 0: return None
            return sorted_node_list[0]

        if method == "kirkley":
            if len(neighborhood_list) == 0:
                # Initial value: Node with the highest degree.
                return sorted_node_list[0]

            seen_nodes = set().union(*[
                set(nodes) for (edges, nodes) in neighborhood_list
            ])
            seen_and_adj_nodes = set().union(*[
                set(G.adj[node]) for node in seen_nodes
            ])
            adj_nodes = seen_and_adj_nodes - seen_nodes

            if len(adj_nodes) == 0: return None
            return rng.choice(list(adj_nodes), size=1).item()

        raise ValueError("".join(("Method ", str(method), " udefined.")))

    # Constructing neighborhoods until we have exhausted all nodes.
    next = next_node()
    while next is not None:
        edges, nodes = neighborhood(
            G=G,
            rootnode=next,
            r=r,
            sanity_check=sanity_check
        )

        # Marking the nodes as belonging to a neighborhood.
        for node in nodes: G.nodes[node]["neighborhood"] = next

        neighborhood_list += ((edges, nodes),)

        next = next_node()

    return neighborhood_list


def contract_neighborhood(
        G: nx.MultiGraph,
        nodes: tuple,
        sanity_check: bool = False
    ) -> None:
    """
    Contracts the neighborhood in `G`, that is contracting all the edges
    connecting nodes in `nodes`, using `np.einsum` and `np.einsum_path`.
    `G` is manipulated in-place.
    """
    # Sanity check.
    if sanity_check: assert network_intact_check(G)

    if len(nodes) == 1:
        # Trivial case; we must not do anything.
        return

    args = ()

    out = ()
    """`out`-argument to `np.einsum`."""

    rootnode = G.nodes[nodes[0]]["neighborhood"]
    """Root node of the neighborhood."""

    new_edges = ()
    """
    These edges need to be re-added after we have contracted and removed
    the neighborhood.
    """

    interior_edge_label = 0
    exterior_edge_label = 0
    # Labeling the edges within the neighborhood.
    for node in nodes:
        for _, neighbor in G.edges(nbunch=node):
            if neighbor in nodes:
                G[node][neighbor][0]["label"] = interior_edge_label
            interior_edge_label += 1

    # Labeling the edges adjacent to the neighborhood.
    for node in nodes:
        for _, neighbor in G.edges(nbunch=node):
            if neighbor not in nodes:
                G[node][neighbor][0]["label"] = (interior_edge_label
                                                 + exterior_edge_label)

                # Adding this exterior edge to the out-argument of np.einsum.
                out += (interior_edge_label + exterior_edge_label,)

                # Saving this exterior edge for re-insertion into the network
                # later.
                new_edges += ((
                    rootnode,
                    neighbor,
                    {
                        "legs":{
                            rootnode: exterior_edge_label,
                            neighbor: G[node][neighbor][0]["legs"][neighbor]
                        },
                        "trace":False,
                        "indices":None
                    }
                ),)

                exterior_edge_label += 1

    # Extracting the einsum arguments.
    for node in nodes:
        args += (G.nodes[node]["T"],)
        legs = [None for i in range(G.nodes[node]["T"].ndim)]
        for _, neighbor, edge_label in G.edges(nbunch=node, data="label"):
            legs[G[node][neighbor][0]["legs"][node]] = edge_label
        args += (tuple(legs),)
    args += (out,)

    T_res = ctr.einsum(
        *args,
        optimize="greedy",
        #ctr.HyperOptimizer(minimize=ctr.scoring.SizeObjective())
    )

    # Removing the neighborhood and adding the contraction.
    G.remove_nodes_from(nodes)
    G.add_node(rootnode, T=T_res)
    G.add_edges_from(new_edges)

    # Merging any double edges.
    edges_to_be_merged = ()
    for node1, node2 in G.edges(nbunch=rootnode):
        if len(G[node1][node2]) > 1: edges_to_be_merged += ((node1, node2),)
    for edge in edges_to_be_merged:
        merge_edges(*edge, G)

    return


# -------------------------------------------------------------------------------
#                   Cosmetics
# -------------------------------------------------------------------------------


def plot_neighborhoods(
        G: nx.MultiGraph,
        neighborhood_list: tuple,
        pos: dict = None,
        ax: mpl.axes = None,
        show: bool = True,
        draw_labels: bool = False,
        **kwargs
    ) -> None:
    """
    Plot `G` along with it's neighborhood decomposition. Graph is drawn
    according to `pos`, if given (see the [networkx documentation](https://networkx.org/documentation/stable/reference/drawing.html#module-networkx.drawing.layout)
    for details). Figure is shown if `show` is `True`. If `draw_graph`
    is true, an additional figure is constructed and the original graph
    drawn inside it. `kwargs` are passed to plotting functions of
    networkx.
    """
    if pos is None: pos = nx.spring_layout(G)

    # Colormap from which neighborhood colors are drawn.
    cmap = mpl.colormaps["viridis"]

    # Grab the current axis, if none are supplied.
    if ax is None: ax = plt.gca()

    # Drawing all edges in gray first.
    all_edges = [(node1, node2) for node1, node2 in G.edges()]
    nx.draw_networkx_edges(
        G=G,
        pos=pos,
        edgelist=all_edges,
        edge_color="tab:gray",
        ax=ax,
        **kwargs
    )

    n_different_colors = sum([
        1 if len(neighborhood_tuple[1]) > 1 else 0
        for neighborhood_tuple in neighborhood_list]
    )

    # Drawing every enighborhood with a different color.
    iColor = 0
    for neighborhood_tuple in neighborhood_list:
        edges,nodes = neighborhood_tuple
        # Drawing edges within the neighborhood.
        nx.draw_networkx_edges(
            G,
            pos,
            edges,
            width=2,
            edge_color=[iColor for edge in edges],
            edge_cmap=cmap,
            edge_vmin=0,
            edge_vmax=n_different_colors-1,
            ax=ax,
            **kwargs
        )

        # Drawing nodes within the neigborhood.
        node_color = ([iColor for node in nodes] if len(nodes) > 1
                      else "tab:gray")
        nx.draw_networkx_nodes(
            G,
            pos,
            nodes,
            node_color=node_color,
            cmap=cmap,
            vmin=0,
            vmax=n_different_colors-1,
            ax=ax,
            **kwargs
        )
        if len(nodes) > 1: iColor += 1

    if draw_labels:
        # Extracting labels for the nodes.
        node_labels = {}
        for node, label in G.nodes(data="neighborhood"):
            node_labels[node] = r"$R$" if label == node else ""
        nx.draw_networkx_labels(
            G=G,
            pos=pos,
            labels=node_labels,
            font_color="whitesmoke",
            ax=ax,
        )

    plt.tight_layout()
    ax.axis("off")

    if show: plt.show()


if __name__ == "__main__":
    pass