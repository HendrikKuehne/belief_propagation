"""
Belief propagation on graphs using neighbor regions. Inspired by Kirkley et Al, 2021 ([Sci. Adv. 7, eabf1211 (2021)](https://doi.org/10.1126/sciadv.abf1211)).
"""
import numpy as np
import networkx as nx
import copy

from lib.utils import network_intact_check,network_message_check

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
        """
        # are we back at home?
        if node == home: return True

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

    return edges_found,nodes_found

def message_passing_step():
    # TODO: The edges in a neighborhood are contracted. Messages into a neighborhood are messages that are incident to a node in the neighborhood, from an edge that is not in the neighborhood.
    pass

def message_passing_iteration():
    # TODO
    pass

def contract_tensors_messages():
    #TODO
    pass

if __name__ == "__main__":
    pass