"""Vertex-Pair Closeness (VPC) calculations."""
import networkx as nx
import numpy as np
import scipy.sparse as sp
from typing import Tuple

def compute_vpc(G: nx.Graph, u, v) -> float:
    """
    Compute the Vertex-Pair Closeness between two nodes.
    
    Formula: VPC(u,v) = 2 * |N(u) intersection N(v)| / (|N(u)| + |N(v)|)
    
    Args:
        G: Undirected graph.
        u: Node 1
        v: Node 2
        
    Returns:
        float: VPC score.
    """
    nu = set(G.neighbors(u))
    nv = set(G.neighbors(v))
    
    len_nu = len(nu)
    len_nv = len(nv)
    
    if len_nu + len_nv == 0:
        return 0.0
        
    intersection_size = len(nu.intersection(nv))
    return 2.0 * intersection_size / (len_nu + len_nv)

def compute_vpc_weights(G: nx.Graph) -> Tuple[nx.Graph, sp.csr_matrix]:
    """
    Compute VPC weights for all edges and construct the weighted adjacency matrix.
    
    Formula: W_uv = weight_uv * VPC(u,v)
    
    Args:
        G: Undirected simple graph.
        
    Returns:
        Tuple containing:
        - Weighted graph (nx.Graph)
        - Sparse adjacency matrix W (scipy.sparse.csr_matrix)
    """
    W_G = G.copy()
    
    # Precompute neighbor sets to speed up computation
    neighbors = {n: set(G.neighbors(n)) for n in G.nodes()}
    
    for u, v, data in W_G.edges(data=True):
        nu = neighbors[u]
        nv = neighbors[v]
        len_nu = len(nu)
        len_nv = len(nv)
        
        if len_nu + len_nv == 0:
            vpc = 0.0
        else:
            intersection_size = len(nu.intersection(nv))
            vpc = 2.0 * intersection_size / (len_nu + len_nv)
            
        original_weight = data.get('weight', 1.0)
        new_weight = original_weight * vpc
        
        data['weight'] = new_weight
        data['vpc'] = vpc
        
    # Construct adjacency matrix W
    nodes = list(W_G.nodes())
    W = nx.adjacency_matrix(W_G, nodelist=nodes, weight='weight')
    
    return W_G, W
