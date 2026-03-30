"""Test script for io.py."""
import sys
import pickle
import networkx as nx
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mutual_community.io import load_graph, extract_graph_properties

def prepare_karate_club():
    """Download/Generate the Karate club dataset and save as different formats."""
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    G = nx.karate_club_graph()
    
    # Save as gpickle
    with open(data_dir / "karate.gpickle", 'wb') as f:
        pickle.dump(G, f)
        
    # Save as Edge list
    nx.write_edgelist(G, data_dir / "karate.edgelist")
    
    print(f"Generated test graphs in {data_dir.absolute()}")
    return data_dir / "karate.gpickle"

def test_io():
    karate_path = prepare_karate_club()
    
    # Test loading
    print("Testing io.load_graph...")
    G = load_graph(karate_path)
    print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    assert G.number_of_nodes() == 34, f"Expected 34 nodes, got {G.number_of_nodes()}"
    assert G.number_of_edges() == 78, f"Expected 78 edges, got {G.number_of_edges()}"
    
    # Test property extraction
    print("Testing io.extract_graph_properties...")
    nodes, edges, adj_list, degrees = extract_graph_properties(G)
    
    assert len(nodes) == 34
    assert len(edges) == 78
    assert len(adj_list) == 34
    assert len(degrees) == 34
    
    print("All io.py tests passed!")

if __name__ == "__main__":
    test_io()
