"""Command Line Interface for Community Detection."""
import argparse
import sys
import numpy as np
from pathlib import Path

from mutual_community.io import load_graph
from mutual_community.vpc import compute_vpc_weights
from mutual_community.sdp import solve_sdp
from mutual_community.spectral import spectral_partition
from mutual_community.rounding import hyperplane_rounding, repair_sizes
from mutual_community.evaluation import evaluate_all
from mutual_community.visualisation import draw_communities

def parse_args():
    parser = argparse.ArgumentParser(description="VPC-based Mutual Community Detection")
    parser.add_argument("--graph", type=str, required=True, help="Path to the graph file")
    parser.add_argument("--format", type=str, default=None, help="Graph format (inferred if None)")
    parser.add_argument("--k", type=int, default=2, help="Number of communities to detect")
    parser.add_argument("--method", choices=["sdp", "spectral"], default="sdp", help="Optimization method")
    parser.add_argument("--alpha", type=float, default=1.0, help="Resolution parameter for SDP")
    parser.add_argument("--smin", type=int, default=None, help="Minimum community size")
    parser.add_argument("--smax", type=int, default=None, help="Maximum community size")
    parser.add_argument("--no-plot", action="store_true", help="Disable interactive plotting")
    parser.add_argument("--save-plot", type=str, default=None, help="Path to save the plot")
    
    return parser.parse_args()

def main():
    args = parse_args()
    graph_path = Path(args.graph)
    
    if not graph_path.exists():
        print(f"Error: Graph file {graph_path} not found.")
        sys.exit(1)
        
    print(f"=== Mutual Community Detection ===")
    print(f"Graph: {graph_path}")
    print(f"Target communities (k): {args.k}")
    print(f"Method: {args.method.upper()}")
    
    # 1. Load Graph
    print(f"\n[1/5] Loading graph...")
    G = load_graph(graph_path, format=args.format)
    n, m = G.number_of_nodes(), G.number_of_edges()
    print(f"      Nodes: {n}, Edges: {m}")
    
    # 2. Compute VPC Weights
    print(f"\n[2/5] Computing Vertex-Pair Closeness weights...")
    _, W = compute_vpc_weights(G)
    
    # 3. Optimization
    print(f"\n[3/5] Solving optimization ({args.method})...")
    if args.method == "sdp":
        try:
            X = solve_sdp(W, k=args.k, alpha=args.alpha)
            # 4. Rounding
            print(f"\n[4/5] Applying Random Hyperplane Rounding...")
            labels = hyperplane_rounding(X, W=W, n_trials=100)
        except Exception as e:
            print(f"      SDP failed ({e}). Fallback to spectral...")
            labels = spectral_partition(W, k=args.k)
    else:
        labels = spectral_partition(W, k=args.k)
        print(f"\n[4/5] Skipping Rounding (using spectral k-means result)")
        
    # Apply Size constraints if provided
    if args.smin is not None or args.smax is not None:
        print(f"      Repairing size constraints (smin={args.smin}, smax={args.smax})...")
        labels = repair_sizes(W, labels, args.k, s_min=args.smin, s_max=args.smax)
        
    # 5. Evaluation
    print(f"\n[5/5] Evaluating partition metrics...")
    metrics = evaluate_all(G, labels)
    for k, v in metrics.items():
        print(f"      {k}: {v:.4f}")
        
    # Visualization
    if not args.no_plot or args.save_plot:
        print(f"\n      Generating visualization...")
        title = f"Communities ({args.method.upper()}) | Modularity: {metrics['Modularity']:.2f}"
        
        # Disable interactive mode warning if running from script
        import matplotlib
        if args.no_plot:
            matplotlib.use('Agg')
            
        draw_communities(
            G, labels, 
            title=title, 
            save_path=args.save_plot, 
            showplot=not args.no_plot
        )
        
    print("\nDone.")

if __name__ == "__main__":
    main()
