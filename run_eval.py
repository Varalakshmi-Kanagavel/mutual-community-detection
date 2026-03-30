import json
from pathlib import Path
from mutual_community.io import load_graph
from mutual_community.vpc import compute_vpc_weights
from mutual_community.sdp import solve_sdp
from mutual_community.rounding import hyperplane_rounding
from mutual_community.evaluation import evaluate_all

def test_karate():
    G = load_graph("data/karate.gpickle")
    _, W = compute_vpc_weights(G)
    X = solve_sdp(W, k=2, alpha=1.0)
    labels = hyperplane_rounding(X, W=W, n_trials=100)
    metrics = evaluate_all(G, labels)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    test_karate()
