"""Test script for vpc.py."""
import sys
import numpy as np
import networkx as nx
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mutual_community.vpc import compute_vpc, compute_vpc_weights

def test_vpc():
    G = nx.Graph()
    # Create a small graph:
    # 1 - 2 - 3
    # | \ | / |
    # 4 - 5 - 6
    edges = [
        (1, 2), (2, 3), (1, 4), (1, 5),
        (2, 5), (3, 5), (3, 6), (4, 5), (5, 6)
    ]
    G.add_edges_from(edges)
    
    # Node 1 neighbors: {2, 4, 5}
    # Node 2 neighbors: {1, 3, 5}
    # Intersection: {5} -> size 1
    # VPC(1, 2) = 2 * 1 / (3 + 3) = 2/6 = 1/3 ~ 0.333
    vpc_1_2 = compute_vpc(G, 1, 2)
    assert np.isclose(vpc_1_2, 1/3), f"Expected 0.333, got {vpc_1_2}"
    
    # Node 1 neighbors: {2, 4, 5} -> len 3
    # Node 5 neighbors: {1, 2, 3, 4, 6} -> len 5
    # Intersection: {2, 4} -> size 2
    # VPC(1, 5) = 2 * 2 / (3 + 5) = 4/8 = 0.5
    vpc_1_5 = compute_vpc(G, 1, 5)
    assert np.isclose(vpc_1_5, 0.5), f"Expected 0.5, got {vpc_1_5}"
    
    # Test matrix computation
    W_G, W = compute_vpc_weights(G)
    
    # Edge (1, 2) should have weight 1/3
    assert np.isclose(W_G[1][2]['weight'], 1/3)
    assert np.isclose(W_G[1][2]['vpc'], 1/3)
    
    # Check if W is symmetric
    W_dense = W.todense()
    assert np.allclose(W_dense, W_dense.T)
    
    # Check W shape corresponds to nodes
    assert W.shape == (6, 6)
    
    print("All vpc.py tests passed!")

if __name__ == "__main__":
    test_vpc()
