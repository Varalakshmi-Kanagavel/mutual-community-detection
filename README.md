# Mutual Community Detection

Implementation of a community detection system for social networks using the Vertex-Pair Closeness (VPC) based mutual relationship method.

## Requirements

This project uses Python 3. `cvxpy` with the `SCS` solver is used for SDP optimization.

\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Architecture

The project consists of the following phases:
1. **Graph Processing** loading and processing standard structures (`mutual_community.io`)
2. **Vertex Pair Closeness** Calculation (`mutual_community.vpc`)
3. **Optimization Layer** SDP and Spectral falling back (`mutual_community.sdp`, `mutual_community.spectral`)
4. **Rounding** Random Hyperplane rounding to discrete communities (`mutual_community.rounding`)
5. **Constraint Repair** Ensure sized communities restrictions (`mutual_community.rounding.repair_sizes`)
6. **Evaluation Metrics** Modularity, Conductance, Cut ratio (`mutual_community.evaluation`)
7. **Visualization** (`mutual_community.visualisation`)

## Usage

### Command Line Interface

You can run the full pipeline using the CLI:

\`\`\`bash
python -m mutual_community --graph data/karate.gpickle --k 2 --method sdp
\`\`\`

**Options:**
- `--graph`: Path to the graph file (e.g. `gml`, `gpickle`, `edgelist`)
- `--k`: Number of communities to detect (default: 2)
- `--method`: `sdp` or `spectral` (default: `sdp`)
- `--alpha`: Resolution parameter for SDP (default: 1.0)
- `--smin` / `--smax`: Enforce min/max size constraints on communities.
- `--no-plot`: Disable interactive plot displaying.
- `--save-plot`: Save plot to specified file.

### Programmatic Usage

Check out the `notebooks/demo_karate.ipynb` for a Python walkthrough that you can embed in your code.

## Results on Benchmark Datasets

### Karate Club (34 nodes, 2 communities)
Typical configuration: `--method sdp --k 2`
* **Modularity:** ≈ 0.38 - 0.41
* **Conductance:** ≈ 0.13 - 0.14
* **Cut Ratio:** ≈ 0.12 - 0.14

*Note: Results may vary slightly depending on the exact rounding due to random hyperplanes.*
