"""Graph input/output and processing utilities."""
import pickle
import networkx as nx
from pathlib import Path
from typing import Tuple, Dict, List, Any

def load_graph(filepath: str | Path, format: str | None = None) -> nx.Graph:
    """
    Load a graph from various formats and process it.
    
    Supported formats (inferred from extension if not provided):
    - gml (.gml)
    - graphml (.graphml)
    - gexf (.gexf)
    - edgelist (.edgelist, .txt, .csv)
    - adjlist (.adjlist)
    - gpickle (.gpickle)
    
    Args:
        filepath: Path to the graph file.
        format: Format of the graph file. If None, inferred from extension.
        
    Returns:
        nx.Graph: An undirected simple graph.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Graph file not found: {filepath}")
        
    if format is None:
        format = filepath.suffix.lower().lstrip('.')
        
    try:
        if format == 'gml':
            G = nx.read_gml(str(filepath))
        elif format == 'graphml':
            G = nx.read_graphml(str(filepath))
        elif format == 'gexf':
            G = nx.read_gexf(str(filepath))
        elif format in ('edgelist', 'txt', 'csv'):
            G = nx.read_edgelist(str(filepath))
        elif format == 'adjlist':
            G = nx.read_adjlist(str(filepath))
        elif format == 'gpickle':
            with open(filepath, 'rb') as f:
                G = pickle.load(f)
        else:
            raise ValueError(f"Unsupported graph format: {format}")
    except Exception as e:
        raise RuntimeError(f"Error loading graph {filepath}: {e}")
        
    # Convert to undirected simple graph
    G = nx.Graph(G)
    
    return G

def extract_graph_properties(G: nx.Graph) -> Tuple[List, List, Dict[Any, List], Dict[Any, int]]:
    """
    Extract essential properties from the graph.
    
    Args:
        G: The undirected simple graph.
        
    Returns:
        Tuple containing:
        - nodes (list): List of nodes
        - edges (list): List of edges
        - adjacency list (dict): Dictionary mapping node to list of neighbors
        - degrees (dict): Dictionary mapping node to degree
    """
    nodes = list(G.nodes())
    edges = list(G.edges())
    
    # Extract adjacency list
    adj_list = {node: list(G.neighbors(node)) for node in G.nodes()}
    
    # Extract degrees
    degrees = dict(G.degree())
    
    return nodes, edges, adj_list, degrees
