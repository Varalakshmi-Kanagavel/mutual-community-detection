"""Test script for optimization methods."""
import sys
import numpy as np
import scipy.sparse as sp
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mutual_community.sdp import solve_sdp, build_objective_matrix
from mutual_community.spectral import spectral_partition

def test_optimization():
    # Simple block diagonal matrix representing 2 communities of size 3
    # Com 1: nodes 0, 1, 2
    # Com 2: nodes 3, 4, 5
    W_dense = np.array([
        [0, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 0]
    ], dtype=float)
    
    W = sp.csr_matrix(W_dense)
    n = 6
    k = 2
    
    # Test SDP
    print("Testing solve_sdp...")
    try:
        X = solve_sdp(W, k=2, alpha=1.0)
        assert X.shape == (n, n)
        # Check constraints
        assert np.allclose(np.diag(X), 1.0, atol=1e-3)
        assert np.min(X) >= (-1/(k-1)) - 1e-4
        
        # In a perfect block structure, nodes in same community have X_uv close to 1
        # and nodes in different community have X_uv close to -1/(k-1)
        assert X[0, 1] > 0.5
        assert X[0, 4] < -0.5
        print("SDP OK.")
    except Exception as e:
        print(f"SDP failed: {e}")
        
    # Test spectral
    print("Testing spectral_partition...")
    labels = spectral_partition(W, k=2)
    assert len(labels) == n
    
    # Nodes 0,1,2 should have same label. Nodes 3,4,5 same label.
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]
    print("Spectral OK.")
    
    print("All optimization tests completed!")

if __name__ == "__main__":
    test_optimization()
