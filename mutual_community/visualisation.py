"""Visualization module for community detection.

Provides two rendering backends:
  - draw_communities()      → high-quality static PNG via matplotlib
  - draw_communities_pyvis() → interactive HTML via pyvis (optional)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Union

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_colormap(k: int):
    """Return a discrete colormap with k colours.

    Uses tab10 for k ≤ 10 and tab20 for k > 10.
    For k > 20 falls back to a continuous HSV palette.
    """
    if k <= 10:
        cmap = plt.cm.get_cmap("tab10", k)
    elif k <= 20:
        cmap = plt.cm.get_cmap("tab20", k)
    else:
        cmap = plt.cm.get_cmap("hsv", k)
    return cmap


def _community_colors(labels: np.ndarray, cmap) -> list:
    """Map integer labels → RGBA colours using *cmap*."""
    unique = np.unique(labels)
    label_to_idx = {lbl: i for i, lbl in enumerate(unique)}
    k = len(unique)
    return [cmap(label_to_idx[lbl] / max(k - 1, 1)) for lbl in labels]


def _node_sizes(G: nx.Graph, nodes: list, scale: float = 50.0,
                min_size: float = 80.0) -> list:
    """Scale node sizes by degree."""
    return [max(G.degree(n) * scale, min_size) for n in nodes]


def _label_filter(G: nx.Graph, nodes: list,
                  degree_threshold: int = 3) -> dict:
    """Return a {node: node} mapping only for high-degree nodes."""
    return {n: n for n in nodes if G.degree(n) > degree_threshold}


def _spring_layout(G: nx.Graph, k: Optional[float] = None) -> dict:
    """Return a spring layout with sensible defaults."""
    if k is None:
        n = G.number_of_nodes()
        k = 1.5 / (n ** 0.5) if n > 0 else 0.3
    return nx.spring_layout(G, seed=42, k=k)


def _build_legend(unique_labels: np.ndarray, cmap, k: int) -> list:
    label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
    patches = []
    for lbl in unique_labels:
        colour = cmap(label_to_idx[lbl] / max(k - 1, 1))
        patches.append(
            mpatches.Patch(facecolor=colour, edgecolor="white",
                           linewidth=0.8, label=f"Community {lbl}")
        )
    return patches


# ──────────────────────────────────────────────────────────────────────────────
# Public API – static matplotlib figure
# ──────────────────────────────────────────────────────────────────────────────

def draw_communities(
    G: nx.Graph,
    labels: np.ndarray,
    *,
    title: str = "Detected Communities",
    k: Optional[int] = None,
    modularity: Optional[float] = None,
    method: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    showplot: bool = True,
    layout: str = "spring",
    spring_k: Optional[float] = None,
    node_scale: float = 50.0,
    edge_alpha: float = 0.35,
    label_degree_threshold: int = 3,
    label_font_size: int = 7,
    figsize: tuple = (14, 10),
    dpi: int = 300,
) -> None:
    """Render a publication-ready community graph.

    Parameters
    ----------
    G : nx.Graph
        The undirected input graph.
    labels : np.ndarray
        Integer community label for each node (in the same order as G.nodes()).
    title : str
        Base title string (overridden when *k* / *modularity* / *method* are given).
    k : int, optional
        Number of communities – used in the auto-generated title.
    modularity : float, optional
        Modularity score – used in the auto-generated title.
    method : str, optional
        Algorithm name, e.g. ``"SDP"`` or ``"Spectral"``.
    save_path : str or Path, optional
        Where to write the PNG file (high-res, 300 dpi by default).
    showplot : bool
        Whether to call ``plt.show()``.
    layout : {"spring", "kamada_kawai"}
        Graph layout algorithm.
    spring_k : float, optional
        Repulsion factor for ``spring_layout``.  Inferred from graph size when
        *None*.
    node_scale : float
        Multiplier applied to node degree to determine node size (px²).
    edge_alpha : float
        Opacity of drawn edges (0 = invisible, 1 = opaque).
    label_degree_threshold : int
        Only nodes with degree strictly greater than this value receive a label.
    label_font_size : int
        Font size for node labels.
    figsize : tuple
        Matplotlib figure size in inches.
    dpi : int
        Resolution for saved figures.
    """
    nodes = list(G.nodes())
    unique_labels = np.unique(labels)
    num_communities = len(unique_labels)

    # ── Colourmap ──────────────────────────────────────────────────────────────
    effective_k = k if k is not None else num_communities
    cmap = _build_colormap(effective_k)
    node_colors = _community_colors(labels, cmap)

    # ── Layout ────────────────────────────────────────────────────────────────
    if "pos" in (G.nodes[nodes[0]] if nodes else {}):
        pos = nx.get_node_attributes(G, "pos")
    elif layout == "kamada_kawai":
        try:
            pos = nx.kamada_kawai_layout(G)
        except Exception:
            warnings.warn("kamada_kawai_layout failed; falling back to spring_layout.")
            pos = _spring_layout(G, spring_k)
    else:
        pos = _spring_layout(G, spring_k)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("#1a1a2e")          # dark navy background
    fig.patch.set_facecolor("#1a1a2e")

    # ── Edges ─────────────────────────────────────────────────────────────────
    nx.draw_networkx_edges(
        G, pos,
        ax=ax,
        edge_color="#aaaaaa",
        alpha=edge_alpha,
        width=0.6,
        style="solid",
    )

    # ── Nodes ─────────────────────────────────────────────────────────────────
    sizes = _node_sizes(G, nodes, scale=node_scale)
    nx.draw_networkx_nodes(
        G, pos,
        ax=ax,
        nodelist=nodes,
        node_size=sizes,
        node_color=node_colors,
        edgecolors="white",
        linewidths=0.8,
    )

    # ── Labels (selective) ────────────────────────────────────────────────────
    label_map = _label_filter(G, nodes, degree_threshold=label_degree_threshold)
    if label_map:
        nx.draw_networkx_labels(
            G, pos,
            labels=label_map,
            ax=ax,
            font_size=label_font_size,
            font_color="white",
            font_family="DejaVu Sans",
        )

    # ── Title ─────────────────────────────────────────────────────────────────
    if k is not None and modularity is not None:
        method_str = f"{method.upper()} | " if method else ""
        title = (
            f"Community Detection — {method_str}"
            f"k={k} communities | Modularity = {modularity:.3f}"
        )
    ax.set_title(title, fontsize=13, color="white", pad=14,
                 fontweight="bold", fontfamily="DejaVu Sans")

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_patches = _build_legend(unique_labels, cmap, effective_k)
    leg = ax.legend(
        handles=legend_patches,
        loc="upper left",
        fontsize=8,
        framealpha=0.25,
        facecolor="#0f3460",
        edgecolor="#e94560",
        labelcolor="white",
        title="Communities",
        title_fontsize=9,
    )
    leg.get_title().set_color("white")

    # ── Metadata text (bottom-right) ──────────────────────────────────────────
    info_parts = [f"Nodes: {G.number_of_nodes()}", f"Edges: {G.number_of_edges()}"]
    if num_communities:
        info_parts.append(f"Communities: {num_communities}")
    ax.text(
        0.99, 0.01, "  |  ".join(info_parts),
        transform=ax.transAxes,
        fontsize=7, color="#cccccc",
        ha="right", va="bottom",
        fontfamily="DejaVu Sans",
    )

    ax.axis("off")
    plt.tight_layout()

    # ── Save ──────────────────────────────────────────────────────────────────
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[visualisation] Saved static plot → {save_path}")

    if showplot:
        plt.show()

    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Public API – interactive pyvis HTML (optional)
# ──────────────────────────────────────────────────────────────────────────────

def draw_communities_pyvis(
    G: nx.Graph,
    labels: np.ndarray,
    *,
    output_path: Union[str, Path] = "graph.html",
    k: Optional[int] = None,
    modularity: Optional[float] = None,
    method: Optional[str] = None,
    node_scale: float = 5.0,
    label_degree_threshold: int = 3,
    height: str = "750px",
    width: str = "100%",
    bgcolor: str = "#1a1a2e",
    font_color: str = "#ffffff",
) -> None:
    """Render an interactive community graph as an HTML file using *pyvis*.

    Parameters
    ----------
    G : nx.Graph
        The undirected input graph.
    labels : np.ndarray
        Community label for each node (same ordering as G.nodes()).
    output_path : str or Path
        Destination HTML file.
    k : int, optional
        Number of communities (for title / colour count).
    modularity : float, optional
        Modularity score shown in the page title.
    method : str, optional
        Algorithm name.
    node_scale : float
        Scaling factor applied to node degree for visual size.
    label_degree_threshold : int
        Only nodes with degree strictly above this value show their ID label.
    height, width : str
        CSS dimensions of the canvas.
    bgcolor : str
        Canvas background colour (CSS colour string).
    font_color : str
        Default label colour.

    Raises
    ------
    ImportError
        If *pyvis* is not installed.  Install with ``pip install pyvis``.
    """
    try:
        from pyvis.network import Network
    except ImportError as exc:
        raise ImportError(
            "pyvis is required for interactive visualizations. "
            "Install with: pip install pyvis"
        ) from exc

    import matplotlib.colors as mcolors

    nodes = list(G.nodes())
    unique_labels = np.unique(labels)
    effective_k = k if k is not None else len(unique_labels)
    cmap = _build_colormap(effective_k)

    label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}

    def _rgba_to_hex(rgba) -> str:
        return mcolors.to_hex(rgba[:3])

    # Build page heading (ASCII-safe for pyvis injection)
    method_str = f"{method.upper()} | " if method else ""
    mod_str = f"| Modularity {modularity:.3f}" if modularity is not None else ""
    heading_text = f"Community Detection | {method_str}k={effective_k} {mod_str}"

    net = Network(
        height=height,
        width=width,
        bgcolor=bgcolor,
        font_color=font_color,
        heading="",          # injected manually below to avoid pyvis char-encoding bugs
    )
    net.barnes_hut(gravity=-8000, central_gravity=0.3,
                   spring_length=200, spring_strength=0.04,
                   damping=0.09, overlap=0)

    node_label_arr = np.array(labels)
    for i, node in enumerate(nodes):
        lbl = node_label_arr[i]
        colour_hex = _rgba_to_hex(cmap(label_to_idx[lbl] / max(effective_k - 1, 1)))
        deg = G.degree(node)
        size = max(deg * node_scale, 8)
        show_label = str(node) if deg > label_degree_threshold else ""
        title_html = (
            f"<b>Node {node}</b><br>"
            f"Community: {lbl}<br>"
            f"Degree: {deg}"
        )
        net.add_node(
            node,
            label=show_label,
            title=title_html,
            color=colour_hex,
            size=size,
            borderWidth=1,
            borderWidthSelected=3,
        )

    for u, v in G.edges():
        net.add_edge(u, v, color="#888888", width=0.8)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    net.save_graph(str(output_path))

    # Post-process: inject properly styled heading (pyvis template mangles special chars)
    import re as _re
    html = output_path.read_text(encoding="utf-8")
    styled_h1 = (
        f'<h1 style="font-family: Arial, sans-serif; font-size: 18px; '
        f'color: #e0e0e0; text-align: center; padding: 10px 0 4px; '
        f'margin: 0; background: #16213e; letter-spacing: 0.5px;">'
        f"{heading_text}"
        f"</h1>"
    )
    html = _re.sub(r"<h1>.*?</h1>", styled_h1, html, count=1, flags=_re.DOTALL)
    output_path.write_text(html, encoding="utf-8")

    print(f"[visualisation] Saved interactive graph -> {output_path}")
