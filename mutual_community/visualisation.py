"""Visualization module for community detection."""
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional, Union

def draw_communities(
    G: nx.Graph, 
    labels: np.ndarray, 
    title: str = "Detected Communities",
    save_path: Optional[Union[str, Path]] = None,
    showplot: bool = True
):
    """
    Visualize detected communities on a graph.
    
    Args:
        G: Undirected graph.
        labels: Array of cluster labels corresponding to node order in G.
        title: Plot title.
        save_path: Path to save the figure (optional).
        showplot: Whether to display the plot interactively.
    """
    # Create colormap based on unique communities
    unique_labels = np.unique(labels)
    cmap = plt.cm.get_cmap('tab10') if len(unique_labels) <= 10 else plt.cm.get_cmap('tab20')
    
    node_colors = [cmap(i % cmap.N) for i in labels]
    
    plt.figure(figsize=(10, 8))
    
    # Generate layout (use pre-computed if available, else spring layout)
    if 'pos' in G.nodes[list(G.nodes())[0]]:
        pos = nx.get_node_attributes(G, 'pos')
    else:
        pos = nx.spring_layout(G, seed=42)
        
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.5, style="solid")
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color=node_colors, edgecolors="white", linewidths=1.5)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")
    
    plt.title(title, fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Visualization saved to {save_path}")
        
    if showplot:
        plt.show()
    
    # Clear plot to free memory
    plt.close()
