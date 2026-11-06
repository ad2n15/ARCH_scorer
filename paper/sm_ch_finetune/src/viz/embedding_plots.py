"""PCA, t-SNE, and UMAP plots for pooled 1536-d AlphaGenome embeddings."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def _label_mask(labels: np.ndarray, value: int) -> np.ndarray:
    return labels.astype(int) == value


def _scatter_2d(
    coords: np.ndarray,
    labels: np.ndarray,
    *,
    title: str,
    out_path: Path,
    xlabel: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    styles = [(0, "SM", "#2563eb"), (1, "CH", "#dc2626")]
    for lab, name, color in styles:
        m = _label_mask(labels, lab)
        if m.any():
            ax.scatter(coords[m, 0], coords[m, 1], c=color, label=f"{name} (n={m.sum()})", alpha=0.75, s=28, edgecolors="none")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=True, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_pca(embeddings: np.ndarray, labels: np.ndarray, out_dir: Path, *, prefix: str = "") -> Path:
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(embeddings)
    out = out_dir / f"{prefix}pca_sm_ch.png"
    _scatter_2d(
        coords, labels,
        title=f"PCA (PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%})",
        out_path=out,
        xlabel="PC1",
        ylabel="PC2",
    )
    return out


def plot_tsne(embeddings: np.ndarray, labels: np.ndarray, out_dir: Path, *, prefix: str = "") -> Path:
    n = len(labels)
    perplexity = min(30, max(5, (n - 1) // 3))
    coords = TSNE(n_components=2, random_state=42, perplexity=perplexity, init="pca", learning_rate="auto").fit_transform(embeddings)
    out = out_dir / f"{prefix}tsne_sm_ch.png"
    _scatter_2d(coords, labels, title=f"t-SNE (perplexity={perplexity})", out_path=out, xlabel="t-SNE 1", ylabel="t-SNE 2")
    return out


def plot_umap(embeddings: np.ndarray, labels: np.ndarray, out_dir: Path, *, prefix: str = "") -> Path:
    try:
        import umap
    except ImportError as e:
        raise ImportError("umap-learn required for UMAP plots: pip install umap-learn") from e
    n = len(labels)
    n_neighbors = min(15, max(2, n - 1))
    coords = umap.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors, min_dist=0.25).fit_transform(embeddings)
    out = out_dir / f"{prefix}umap_sm_ch.png"
    _scatter_2d(coords, labels, title=f"UMAP (n_neighbors={n_neighbors})", out_path=out, xlabel="UMAP 1", ylabel="UMAP 2")
    return out


def generate_embedding_plots(
    embeddings: np.ndarray,
    labels: np.ndarray,
    out_dir: Path,
    *,
    prefix: str = "",
    methods: tuple[str, ...] = ("pca", "tsne", "umap"),
) -> list[Path]:
    out_dir = Path(out_dir)
    saved: list[Path] = []
    if "pca" in methods:
        saved.append(plot_pca(embeddings, labels, out_dir, prefix=prefix))
    if "tsne" in methods:
        saved.append(plot_tsne(embeddings, labels, out_dir, prefix=prefix))
    if "umap" in methods:
        saved.append(plot_umap(embeddings, labels, out_dir, prefix=prefix))
    return saved
