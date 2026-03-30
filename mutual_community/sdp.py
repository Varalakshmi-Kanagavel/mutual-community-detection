"""SDP optimization for community detection."""
import networkx as nx
import numpy as np
import scipy.sparse as sp
import cvxpy as cp
from typing import Tuple

def build_objective_matrix(W: sp.csr_matrix, alpha: float = 1.0) -> np.ndarray:
    """
    Build the objective matrix M for the SDP.
    The prompt formulates this conceptually as: Maximize Tightness - alpha * Looseness.
    We represent this using a modularity-like matrix:
    M_uv = W_uv - alpha * (d_u * d_v) / (2 * m)
    
    Args:
        W: Sparse weighted adjacency matrix.
        alpha: Resolution parameter.
        
    Returns:
        np.ndarray: The dense objective matrix.
    """
    n = W.shape[0]
    W_dense = W.toarray()
    
    # Degrees are the sum of weights for each node
    degrees = np.sum(W_dense, axis=1)
    m2 = np.sum(degrees) # 2 * Total weight (2m)
    
    if m2 == 0:
        return W_dense
        
    expected_edges = np.outer(degrees, degrees) / m2
    
    M = W_dense - alpha * expected_edges
    
    # Zero out diagonal to avoid self-loops contribution
    np.fill_diagonal(M, 0)
    
    return M

def solve_sdp(W: sp.csr_matrix, k: int, alpha: float = 1.0) -> np.ndarray:
    """
    Solve the SDP relaxation for k-community partitioning.
    
    Objective: 
        Maximize Tightness - alpha * Looseness
    Subject to:
        X >= 0 (Positive Semidefinite)
        diag(X) = 1
        X_uv >= -1/(k-1)
        
    Args:
        W: Sparse weighted adjacency matrix.
        k: Number of communities to find.
        alpha: Parameter balancing tightness and looseness.
        
    Returns:
        np.ndarray: The optimized continuous matrix X.
    """
    n = W.shape[0]
    
    # Build objective matrix
    M = build_objective_matrix(W, alpha)
    
    # Define matrix variable X
    X = cp.Variable((n, n), symmetric=True)
    
    # Define objective function
    objective = cp.Maximize(cp.trace(M @ X))
    
    # Add constraints
    constraints = [
        X >> 0,  # Positive semidefinite
        cp.diag(X) == 1,
        X >= -1 / (k - 1)  # Element-wise constraint for k-partition
    ]
    
    # Solve SDP
    prob = cp.Problem(objective, constraints)
    print(f"Solving SDP with SCS for graph of size {n} and k={k}...")
    prob.solve(solver=cp.SCS, verbose=False)
    
    if X.value is None:
        raise RuntimeError("SDP solver failed to find a solution.")
        
    return X.value
