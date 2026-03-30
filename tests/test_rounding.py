"""Test script for rounding and size repair."""
import sys
import numpy as np
import scipy.sparse as sp
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mutual_community.rounding import hyperplane_rounding, repair_sizes

def test_rounding():
    print("Testing hyperplane_rounding...")
    # Create simple block matrix representing 2 communities
    n = 6
    X_dense = np.array([
        [1.0, 0.9, 0.8, -0.9, -0.8, -0.9],
        [0.9, 1.0, 0.9, -0.8, -0.9, -0.8],
        [0.8, 0.9, 1.0, -0.9, -0.8, -0.9],
        [-0.9, -0.8, -0.9, 1.0, 0.9, 0.8],
        [-0.8, -0.9, -0.8, 0.9, 1.0, 0.9],
        [-0.9, -0.8, -0.9, 0.8, 0.9, 1.0]
    ])
    
    # We pass it through rounding
    labels = hyperplane_rounding(X_dense, n_trials=10)
    
    # Nodes 0,1,2 should be grouped. Nodes 3,4,5 grouped.
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]
    print("Rounding OK.")
    
    print("Testing repair_sizes...")
    # Simulate labels where one community is too big
    # Com 0: nodes 0, 1, 2, 3, 4
    # Com 1: node 5
    bad_labels = np.array([0, 0, 0, 0, 0, 1])
    
    # Mock adjacency matrix where node 4 is closely connected to node 5
    W_dense = np.zeros((n, n))
    W_dense[4, 5] = W_dense[5, 4] = 10.0 # strong connection
    W_dense[0, 1] = W_dense[1, 0] = 10.0
    W_dense[1, 2] = W_dense[2, 1] = 10.0
    W_dense[2, 3] = W_dense[3, 2] = 10.0
    W = sp.csr_matrix(W_dense)
    
    # Repair with s_min=2, s_max=4
    repaired = repair_sizes(W, bad_labels, k=2, s_min=2, s_max=4)
    
    sizes = [np.sum(repaired == c) for c in range(2)]
    assert 2 <= sizes[0] <= 4
    assert 2 <= sizes[1] <= 4
    
    # Because node 4 is strongly connected to node 5, it should be the one moved to Com 1
    assert repaired[4] == repaired[5]
    
    print("Repair OK.")
    
    print("All rounding tests completed!")

if __name__ == "__main__":
    test_rounding()
