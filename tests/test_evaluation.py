"""Test script for evaluation metrics."""
import sys
import numpy as np
import networkx as nx
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mutual_community.evaluation import evaluate_all, modularity, conductance, cut_ratio

def test_evaluation():
    # Simple block graph
    G = nx.Graph()
    G.add_edges_from([
        (0, 1), (1, 2), (0, 2),  # Community 0
        (3, 4), (4, 5), (3, 5),  # Community 1
        (2, 3)                   # Single bridge edge
    ])
    
    # Perfect partition
    labels = np.array([0, 0, 0, 1, 1, 1])
    metrics = evaluate_all(G, labels)
    
    print("Metrics for perfect partition:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
        
    # Cut ratio: 1 cross edge, 7 total edges -> 1/7 = 0.1428...
    assert np.isclose(metrics["Cut Ratio"], 1/7)
    
    # Conductance of community 0 (nodes 0,1,2):
    # cut_size = 1 (edge 2-3)
    # vol_S = 2+2+3 = 7
    # conductance = 1/7
    assert np.isclose(metrics["Conductance"], 1/7)
    
    # Modularity > 0 (good partition)
    assert metrics["Modularity"] > 0.3
    
    # Bad partition
    bad_labels = np.array([0, 1, 0, 1, 0, 1])
    bad_metrics = evaluate_all(G, bad_labels)
    
    print("\nMetrics for bad partition:")
    for k, v in bad_metrics.items():
        print(f"  {k}: {v:.4f}")
        
    assert bad_metrics["Modularity"] < metrics["Modularity"]
    assert bad_metrics["Conductance"] > metrics["Conductance"]
    assert bad_metrics["Cut Ratio"] > metrics["Cut Ratio"]
    
    print("\nAll evaluation tests passed!")

if __name__ == "__main__":
    test_evaluation()
