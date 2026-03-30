"""Spectral approximation for community detection."""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from sklearn.cluster import KMeans

def spectral_partition(W: sp.csr_matrix, k: int) -> np.ndarray:
    """
    Apply spectral approximation for k-community partitioning.
    
    Steps:
    1 Compute Laplacian L = D - W
    2 Compute eigenvectors
    3 Apply k-means clustering
    
    Args:
        W: Sparse weighted adjacency matrix.
        k: Number of communities to find.
        
    Returns:
        np.ndarray: An array of cluster labels for each node.
    """
    n = W.shape[0]
    
    # Calculate degree matrix
    degrees = np.array(W.sum(axis=1)).flatten()
    D = sp.diags(degrees)
    
    # Compute Laplacian L = D - W
    L = D - W
    
    # Compute the first k eigenvectors (corresponding to smallest eigenvalues)
    # Using shift-invert mode or simply targeting smallest magnitude (SM)
    # If k < n - 1, use eigsh
    if k < n - 1:
        # Get k smallest eigenvalues/vectors
        # We use k+1 because the smallest eigenvector is trivial (all 1s) for standard Laplacian
        eigvals, eigvecs = sla.eigsh(L.astype(float), k=k, which='SM', tol=1e-5)
    else:
        # Fallback to dense if k is large
        L_dense = L.toarray()
        eigvals, eigvecs = np.linalg.eigh(L_dense)
        eigvecs = eigvecs[:, :k]
        
    # Apply k-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(eigvecs)
    
    return labels
