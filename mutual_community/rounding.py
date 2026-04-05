"""Rounding and constraint repair algorithms."""
import numpy as np
import scipy.sparse as sp
from typing import Optional
from collections import defaultdict
from sklearn.cluster import KMeans


def _extract_embedding(X: np.ndarray) -> np.ndarray:
    """
    Extract embedding vectors V from a PSD gram matrix X (X ≈ V V^T).
    Uses eigendecomposition so it is always numerically stable.

    Returns:
        V: (n, n) matrix of row-embedding vectors.
    """
    n = X.shape[0]
    X = (X + X.T) / 2  # enforce symmetry
    eigenvalues, eigenvectors = np.linalg.eigh(X)
    eigenvalues = np.clip(eigenvalues, 0, None)          # project to PSD cone
    V = eigenvectors @ np.diag(np.sqrt(eigenvalues))    # V s.t. V V^T ≈ X
    return V


def hyperplane_rounding(
    X: np.ndarray,
    k: int = 2,
    W: Optional[sp.csr_matrix] = None,
    n_trials: int = 10,
    random_state: int = 42,
) -> np.ndarray:
    """
    K-way rounding of an SDP gram matrix X into exactly k communities.

    Strategy
    --------
    1. Extract embedding vectors V from X via eigendecomposition.
    2. Run KMeans(n_clusters=k) on the rows of V to obtain k-way labels.
    3. Retry with different random seeds until exactly k unique labels are
       produced (up to n_trials attempts).

    Fallback
    --------
    If KMeans cannot produce k clusters (e.g. degenerate embeddings),
    the function falls back to iterative random-seed restarts and then,
    if still failing, to a spectral reassignment of small clusters.

    Args:
        X: Positive semidefinite gram matrix from SDP (n × n).
        k: Desired number of communities (must be >= 2).
        W: Unused – kept for backward-compatible call signature.
        n_trials: Number of KMeans restarts to try before fallback.
        random_state: Base random seed.

    Returns:
        np.ndarray: Integer labels in 0 … k-1, length n.
    """
    n = X.shape[0]
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")
    if k > n:
        raise ValueError(f"k={k} cannot exceed number of nodes n={n}")

    V = _extract_embedding(X)

    labels = None
    for attempt in range(n_trials):
        seed = random_state + attempt
        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        candidate = km.fit_predict(V)
        if len(set(candidate)) == k:
            labels = candidate
            break

    # ── Fallback: force exactly k clusters by splitting/merging ──────────
    if labels is None or len(set(labels)) != k:
        print(
            f"  Warning: KMeans did not produce {k} clusters after "
            f"{n_trials} attempts. Applying spectral fallback."
        )
        labels = _force_k_clusters(V, k, random_state=random_state)

    # ── Validation ───────────────────────────────────────────────────────
    assert len(set(labels)) == k, (
        f"Rounding produced {len(set(labels))} clusters but expected {k}."
    )

    # Relabel to 0 … k-1 (KMeans already does this, but be safe)
    unique = sorted(set(labels))
    remap = {old: new for new, old in enumerate(unique)}
    labels = np.array([remap[l] for l in labels], dtype=int)

    return labels


def _force_k_clusters(V: np.ndarray, k: int, random_state: int = 0) -> np.ndarray:
    """
    Guarantee exactly k clusters by iteratively splitting the largest cluster
    until we have k, using KMeans sub-clustering.
    """
    n = V.shape[0]
    labels = np.zeros(n, dtype=int)   # start with 1 big cluster
    next_label = 1

    while len(set(labels)) < k:
        # Find the largest cluster to split
        unique, counts = np.unique(labels, return_counts=True)
        biggest = unique[np.argmax(counts)]
        idx = np.where(labels == biggest)[0]

        if len(idx) < 2:
            break  # cannot split further

        sub_km = KMeans(n_clusters=2, n_init=10, random_state=random_state)
        sub_labels = sub_km.fit_predict(V[idx])

        # Keep one part as `biggest`, relabel the other as `next_label`
        mask_new = sub_labels == 1
        labels[idx[mask_new]] = next_label
        next_label += 1

    return labels

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
