"""Rounding and constraint repair algorithms."""
import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Tuple
from collections import defaultdict

def hyperplane_rounding(X: np.ndarray, W: sp.csr_matrix = None, n_trials: int = 100) -> np.ndarray:
    """
    Apply Random Hyperplane Rounding to continuous vectors derived from SDP.
    Since X is the gram matrix (X = V^T V), we first do Cholesky factorization 
    to obtain V, then use random hyperplanes.
    
    This splits the graph into 2 communities.
    
    Args:
        X: Positive semidefinite matrix from SDP (n x n).
        W: Optional edge weights to evaluate the best cut among trials.
        n_trials: Number of random hyperplanes to try.
        
    Returns:
        np.ndarray: Array of cluster labels (0 or 1).
    """
    n = X.shape[0]
    
    # Ensure X is positive semi-definite and symmetric
    X = (X + X.T) / 2
    
    # Small eigenvalue bump for numerical stability of Cholesky
    try:
        V = np.linalg.cholesky(X + np.eye(n) * 1e-6)
    except np.linalg.LinAlgError:
        # Fallback to Eigendecomposition if Cholesky fails
        eigenvalues, eigenvectors = np.linalg.eigh(X)
        # Keep positive eigenvalues
        eigenvalues[eigenvalues < 0] = 0
        V = eigenvectors @ np.diag(np.sqrt(eigenvalues))
        
    best_labels = None
    best_score = float('-inf')
    W_dense = W.toarray() if W is not None else None
    
    for _ in range(n_trials):
        # Sample random Gaussian vector
        g = np.random.randn(n)
        
        # Compute projection V * g
        projection = V @ g
        
        # Assign clusters based on sign
        labels = (projection >= 0).astype(int)
        
        # If no weighted matrix is given, just return the first trial
        if W is None:
            return labels
            
        # Optional: evaluate cut score (tightness) to pick the best rounding
        # For simplicity, we want to maximize weights within same cluster
        # Score = sum_{i,j} W_ij * (1 if label_i == label_j else 0)
        same_cluster_mask = labels[:, None] == labels[None, :]
        score = np.sum(W_dense[same_cluster_mask])
        
        if score > best_score:
            best_score = score
            best_labels = labels
            
    return best_labels

def get_community_centroids(W: sp.csr_matrix, labels: np.ndarray, k: int) -> np.ndarray:
    """
    Calculate internal degree 'centroid' score for each node within its community to help 
    decide which nodes to move during repair.
    """
    n = W.shape[0]
    W_dense = W.toarray()
    
    scores = np.zeros((n, k))
    for c in range(k):
        # Calculate sum of weights to nodes in community c for each node
        c_mask = (labels == c)
        if np.any(c_mask):
            scores[:, c] = np.sum(W_dense[:, c_mask], axis=1)
            
    return scores

def repair_sizes(W: sp.csr_matrix, labels: np.ndarray, k: int, 
                 s_min: int | None = None, s_max: int | None = None) -> np.ndarray:
    """
    Repair community sizes to satisfy constraints.
    
    Args:
        W: Sparse weighted adjacency matrix.
        labels: Current community assignments.
        k: Number of communities.
        s_min: Minimum community size.
        s_max: Maximum community size.
        
    Returns:
        np.ndarray: Repaired labels.
    """
    n = len(labels)
    # Default constraints if not provided
    if s_min is None:
        s_min = 1
    if s_max is None:
        s_max = n
        
    repaired_labels = labels.copy()
    
    # Calculate connection strength to all communities
    affinity_scores = get_community_centroids(W, repaired_labels, k)
    
    while True:
        # Check current sizes
        sizes = [np.sum(repaired_labels == c) for c in range(k)]
        
        # Are all constraints satisfied?
        if all(s_min <= size <= s_max for size in sizes):
            break
            
        oversized = [c for c, size in enumerate(sizes) if size > s_max]
        undersized = [c for c, size in enumerate(sizes) if size < s_min]
        
        # If no strict violations can be easily fixed without causing others, we break out.
        # This is a simplified greedy repair mechanism.
        if not oversized and not undersized:
            break
            
        made_change = False
        
        # Fix oversized communities by moving nodes out
        for c_over in oversized:
            # Nodes currently in this oversized community
            nodes_in_c = np.where(repaired_labels == c_over)[0]
            
            # Sort these nodes by how weakly they are connected to this community
            # We want to move the most weakly connected ones
            weak_nodes = sorted(nodes_in_c, key=lambda node: affinity_scores[node, c_over])
            
            # Move nodes until s_max is satisfied
            nodes_to_move = sizes[c_over] - s_max
            for node in weak_nodes[:nodes_to_move]:
                # Find the best valid community to move to (must not violate s_max)
                best_new_c = None
                best_affinity = -1
                
                for candidate_c in range(k):
                    if candidate_c == c_over:
                        continue
                    if sizes[candidate_c] < s_max:
                        if affinity_scores[node, candidate_c] > best_affinity:
                            best_affinity = affinity_scores[node, candidate_c]
                            best_new_c = candidate_c
                            
                if best_new_c is not None:
                    repaired_labels[node] = best_new_c
                    sizes[c_over] -= 1
                    sizes[best_new_c] += 1
                    made_change = True
                    
        # Fix undersized communities by pulling nodes in
        for c_under in undersized:
            nodes_needed = s_min - sizes[c_under]
            if nodes_needed <= 0:
                continue
                
            # Nodes NOT in this undersized community
            nodes_outside = np.where(repaired_labels != c_under)[0]
            
            # Sort by how strongly they want to join this community
            strong_nodes = sorted(nodes_outside, key=lambda node: affinity_scores[node, c_under], reverse=True)
            
            for node in strong_nodes:
                current_c = repaired_labels[node]
                # Can we pull this node without making its current community undersized?
                if sizes[current_c] > s_min:
                    repaired_labels[node] = c_under
                    sizes[c_under] += 1
                    sizes[current_c] -= 1
                    nodes_needed -= 1
                    made_change = True
                    if nodes_needed == 0:
                        break
                        
        if not made_change:
            # Could not find valid moves to fix constraints
            print("Warning: Size constraints could not be fully repaired.")
            break
            
    return repaired_labels
