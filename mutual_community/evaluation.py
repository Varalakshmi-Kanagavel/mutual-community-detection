"""Evaluation metrics for community detection."""
import networkx as nx
import numpy as np

def modularity(G: nx.Graph, communities: list[set]) -> float:
    """
    Compute the Newman-Girvan modularity of a partition.
    
    Args:
        G: Undirected graph.
        communities: List of sets, where each set contains nodes in a community.
        
    Returns:
        float: Modularity score [-0.5, 1.0]. Higher is better.
    """
    return nx.community.modularity(G, communities)

def conductance(G: nx.Graph, community: set) -> float:
    """
    Compute conductance of a single community.
    Conductance is the ratio of edges cut to the total volume (sum of degrees) 
    of the community (or the rest of the graph, whichever is smaller).
    
    Formula: phi(S) = cut(S, V \ S) / min(vol(S), vol(V \ S))
    
    Args:
        G: Undirected graph
        community: Set of nodes in the community.
        
    Returns:
        float: Conductance score [0, 1]. Lower is better.
    """
    if len(community) == 0 or len(community) == len(G.nodes()):
        return 0.0
        
    cut_size = nx.cut_size(G, community)
    vol_S = sum(G.degree(n) for n in community)
    vol_V = sum(G.degree(n) for n in G.nodes())
    vol_not_S = vol_V - vol_S
    
    min_vol = min(vol_S, vol_not_S)
    if min_vol == 0:
        return 0.0
        
    return cut_size / min_vol

def average_conductance(G: nx.Graph, communities: list[set]) -> float:
    """
    Compute average conductance across all communities.
    
    Args:
        G: Undirected graph
        communities: List of sets, where each set contains nodes in a community.
        
    Returns:
        float: Average conductance. Lower is better.
    """
    scores = [conductance(G, comm) for comm in communities if len(comm) > 0]
    if not scores:
        return 0.0
    return sum(scores) / len(scores)

def cut_ratio(G: nx.Graph, communities: list[set]) -> float:
    """
    Compute the cut ratio of the partition.
    Ratio of edges between different communities to the total number of edges.
    
    Args:
        G: Undirected graph
        communities: List of sets, where each set contains nodes in a community.
        
    Returns:
        float: Cut ratio. Lower is better.
    """
    m = G.number_of_edges()
    if m == 0:
        return 0.0
        
    total_cut_edges = 0
    # To avoid double counting, we iterate over all edges and check if endpoints are in different communities
    # First, build a map of node -> community_id
    node_to_comm = {}
    for i, comm in enumerate(communities):
        for node in comm:
            node_to_comm[node] = i
            
    for u, v in G.edges():
        if node_to_comm.get(u) != node_to_comm.get(v):
            total_cut_edges += 1
            
    return total_cut_edges / m

def evaluate_all(G: nx.Graph, labels: np.ndarray) -> dict:
    """
    Compute all evaluating metrics for a given partition.
    
    Args:
        G: Undirected graph
        labels: Array of cluster labels for each node (assumes node IDs are 0 to n-1, or matches order of G.nodes())
        
    Returns:
        dict: Dictionary containing Modularity, Conductance, and Cut Ratio.
    """
    # Convert labels array to list of sets
    communities_dict = {}
    nodes = list(G.nodes())
    for i, label in enumerate(labels):
        if label not in communities_dict:
            communities_dict[label] = set()
        communities_dict[label].add(nodes[i])
        
    communities = list(communities_dict.values())
    
    return {
        "Modularity": modularity(G, communities),
        "Conductance": average_conductance(G, communities),
        "Cut Ratio": cut_ratio(G, communities)
    }
