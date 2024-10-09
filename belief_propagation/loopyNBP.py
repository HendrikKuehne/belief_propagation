"""
Belief propagation on graphs using neighbor regions. Inspired by Kirkley et Al, 2021 ([Sci. Adv. 7, eabf1211 (2021)](https://doi.org/10.1126/sciadv.abf1211)).
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import cotengra as ctr

from belief_propagation.utils import network_intact_check,network_message_check
from belief_propagation.networks import merge_edges

def neighborhood(G:nx.MultiGraph,rootnode:int,r:int=0,sanity_check:bool=False) -> tuple[set,set]:
    """
    Recursively identifies the neighborhood with connectivity `r` around `node`.
    The neighborhood with connectivity `r` is defined as containing loops with length `r+2`.
    Returns a set that contains all edges in the neighborhood, and a set that contains
    all nodes in the neighborhood.
    """
    # sanity check
    if sanity_check: assert network_message_check(G)

    def traverse(G:nx.MultiGraph,node:int,home:int,steps:int,path:list,edges_found:set,nodes_found:set) -> bool:
        """
        Traverses the graph recursively, taking `steps` steps. If a loop is found, the
        nodes and edges along the loop are added to `edges_found` and `nodes_found`.
        This function cannot cross existing neighborhoods; if `G.nodes[node]["neighborhood"]!=None`
        is true, `traverse` immediately returns `False`.
        """
        # are we back at home?
        if node == home: return True

        # did we hit an existing neighborhood?
        try:
            if G.nodes[node]["neighborhood"] != None: return False
        except KeyError:
            pass

        if steps > 0:
            # there are steps left to take
            path_found = False
            for next_node in G.adj[node]:
                if {node,next_node} in path:
                    # we have been here before; every edge should be traversed only once
                    continue

                # depth-first search
                path.append({node,next_node})
                if traverse(G,next_node,home,steps-1,path,edges_found,nodes_found):
                    # the next node is on a path back to home; we need to add this node and the edges we travelled along
                    for edge in path: edges_found.add(frozenset(edge))
                    nodes_found.add(next_node)
                    path_found = True

                # backtracking
                path.pop()

            return path_found

        return False

    edges_found = set()
    nodes_found = {rootnode,}

    for node in G.adj[rootnode]: traverse(G,node,rootnode,r+1,[{rootnode,node},],edges_found,nodes_found)

    edges_found = tuple([tuple(edge) for edge in edges_found])
    nodes_found = tuple(nodes_found)

    return edges_found,nodes_found

def construct_neighborhoods(G:nx.MultiGraph,r:int=0,sanity_check:bool=False) -> tuple:
    """
    Constructs neighborhoods in the graph `G` with connectivity `r`. `G` is modified in-place.
    """
    # The choice of neighborhoods should ensure that as many loops as possible are contained within the neighborhoods. This version of the code achieves this using a (possibly crude) heuristic: We begin neighborhood construction with the nodes that have the largest number of neighbors

    # sanity check
    if sanity_check: assert network_message_check(G)

    # initialization
    for node in G.nodes():
        G.nodes[node]["neighborhood"] = None

    # sorting the nodes based on the number of neighbors
    sorted_node_list = sorted(G.nodes(),key=lambda node: len(G.adj[node]),reverse=True)

    neighborhood_list = ()
    # constructing neighborhoods until we have exhausted all nodes
    while len(sorted_node_list) > 0:
        edges,nodes = neighborhood(G,sorted_node_list[0],r,sanity_check)

        # marking the nodes as belonging to a neighborhood
        for node in nodes: G.nodes[node]["neighborhood"] = sorted_node_list[0]

        # discarding the nodes we found
        for node in nodes: sorted_node_list.remove(node)

        neighborhood_list += ((edges,nodes),)

    return neighborhood_list

def contract_neighborhood(G:nx.MultiGraph,nodes:tuple,sanity_check:bool=False) -> None:
    """
    Contracts the neighborhood in `G`, that is contracting all the edges connecting nodes in
    `nodes`, using `np.einsum` and `np.einsum_path`. `G` is manipulated in-place.
    """
    if sanity_check: assert network_intact_check(G)

    if len(nodes) == 1:
        # trivial case; we must not do anything
        return

    args = ()

    out = ()
    """`out`-argument to `np.einsum`."""

    rootnode = G.nodes[nodes[0]]["neighborhood"]
    """root node of the neighborhood."""

    new_edges = ()
    """These edges need to be re-added after we have contracted and removed the neighborhood."""

    interior_edge_label = 0
    exterior_edge_label = 0
    # labeling the edges within the neighborhood
    for node in nodes:
        for _,neighbor in G.edges(nbunch=node):
            if neighbor in nodes: G[node][neighbor][0]["label"] = interior_edge_label
            interior_edge_label += 1
    # labeling the edges adjacent to the neighborhood
    for node in nodes:
        for _,neighbor in G.edges(nbunch=node):
            if neighbor not in nodes:
                G[node][neighbor][0]["label"] = interior_edge_label + exterior_edge_label

                # adding this exterior edge to the out-argument of np.einsum
                out += (interior_edge_label + exterior_edge_label,)

                # saving this exterior edge for re-insertion into the network later
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

    # extracting the einsum arguments
    for node in nodes:
        args += (G.nodes[node]["T"],)
        legs = [None for i in range(G.nodes[node]["T"].ndim)]
        for _,neighbor,edge_label in G.edges(nbunch=node,data="label"):
            legs[G[node][neighbor][0]["legs"][node]] = edge_label
        args += (tuple(legs),)
    args += (out,)

    T_res = ctr.einsum(
        *args,
        optimize="greedy",#ctr.HyperOptimizer(minimize=ctr.scoring.SizeObjective())
    )

    # removing the neighborhood and adding the contraction
    G.remove_nodes_from(nodes)
    G.add_node(rootnode,T=T_res)
    G.add_edges_from(new_edges)

    # merging any double edges
    edges_to_be_merged = ()
    for node1,node2 in G.edges(nbunch=rootnode):
        if len(G[node1][node2]) > 1: edges_to_be_merged += ((node1,node2),)
    for edge in edges_to_be_merged:
        merge_edges(*edge,G)

    return

# -------------------------------------------------------------------------------
#                   cosmetics
# -------------------------------------------------------------------------------

def plot_neighborhoods(G:nx.MultiGraph,neighborhood_list:tuple) -> None:
    """
    Plot `G` along with it's neighborhood decomposition.
    """
    N = len(neighborhood_list)
    pos = nx.spring_layout(G)

    all_edges = [(node1,node2) for node1,node2 in G.edges()]
    # drawing all edges in gray first
    nx.draw_networkx_edges(G,pos,all_edges,edge_color="tab:gray")

    nColors = sum([1 if len(neighborhood_double[1]) > 1 else 0 for neighborhood_double in neighborhood_list])

    iColor = 0
    for neighborhood_doublet in neighborhood_list:
        edges,nodes = neighborhood_doublet
        # drawing edges within the neighborhood
        nx.draw_networkx_edges(
            G,
            pos,
            edges,
            width=3,
            edge_color=[iColor for edge in edges],
            edge_cmap=mpl.colormaps["plasma"],
            edge_vmin=0,
            edge_vmax=nColors-1
        )
        # drawing nodes within the neigborhood
        node_color = [iColor for node in nodes] if len(nodes) > 1 else "tab:gray"
        nx.draw_networkx_nodes(
            G,
            pos,
            nodes,
            node_color=node_color,
            cmap=mpl.colormaps["plasma"],
            vmin=0,
            vmax=nColors-1
        )
        if len(nodes) > 1: iColor += 1

    # extracting labels for the nodes
    node_labels = {}
    for node,label in G.nodes(data="neighborhood"): node_labels[node] = "R" if label == node else ""
    nx.draw_networkx_labels(G,pos,node_labels,font_color="whitesmoke")

    plt.tight_layout()
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    pass