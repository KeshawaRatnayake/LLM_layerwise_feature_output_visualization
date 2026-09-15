"""
Plot layer-wise token features from a saved hidden_states.pt file
(produced by gpt_neo_layerwise.py with --save_hidden_states).

hidden_states.pt is a list of tensors, one per "layer" (13 for a 12-layer
model: index 0 = input embeddings, indices 1-12 = each transformer block's
output). Each tensor has shape [1, seq_len, hidden_dim] -- in your case
[1, 10, 768].

Produces two figures:
  1. layerwise_pca_grid.png  - one PCA scatter per layer (snapshot view)
  2. token_trajectories.png  - each token's path through PCA space across
                                all layers, connected layer-to-layer (shows
                                how much each word's representation moves
                                as it passes through the network)

Usage:
    python plot_layer_features.py \
        --hidden_states hidden_states.pt \
        --summary summary.json \
        --output_dir ./plots
"""

import argparse
import json
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def load_tokens(summary_path, n_tokens):
    if summary_path and os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        toks = summary.get("tokens", None)
        if toks and len(toks) == n_tokens:
            return [t.strip() if t.strip() else "_" for t in toks]
    # fallback: generic labels if summary.json isn't available or mismatched
    return [f"tok{i}" for i in range(n_tokens)]


def pca_2d(mat):
    """mat: [seq_len, hidden_dim] -> [seq_len, 2] via SVD-based PCA."""
    centered = mat - mat.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_states", default="hidden_states.pt")
    parser.add_argument("--summary", default="summary.json")
    parser.add_argument("--output_dir", default="./plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    hidden_states = torch.load(args.hidden_states, map_location="cpu")
    n_layers = len(hidden_states)          # 13 in your case
    seq_len = hidden_states[0].shape[1]    # 10 in your case
    tokens = load_tokens(args.summary, seq_len)

    print(f"Loaded {n_layers} layers, seq_len={seq_len}, hidden_dim={hidden_states[0].shape[-1]}")
    print("Tokens:", tokens)

    # ------------------------------------------------------------------
    # Figure 1: per-layer PCA grid (one snapshot per layer)
    # ------------------------------------------------------------------
    cols = 5
    rows = int(np.ceil(n_layers / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)

    colors = cm.tab10(np.linspace(0, 1, seq_len))

    for l in range(n_layers):
        mat = hidden_states[l][0].float().numpy()  # [seq_len, hidden_dim]
        proj = pca_2d(mat)
        ax = axes[l]
        for i, tok in enumerate(tokens):
            ax.scatter(proj[i, 0], proj[i, 1], color=colors[i], s=50)
            ax.annotate(tok, (proj[i, 0], proj[i, 1]), fontsize=7)
        label = "embeddings" if l == 0 else f"layer {l}"
        ax.set_title(label)

    for ax in axes[n_layers:]:
        ax.axis("off")

    plt.suptitle("Per-layer PCA snapshots of token features")
    plt.tight_layout()
    grid_path = os.path.join(args.output_dir, "layerwise_pca_grid.png")
    plt.savefig(grid_path, dpi=130)
    plt.close(fig)
    print(f"Saved {grid_path}")

    # ------------------------------------------------------------------
    # Figure 2: token trajectories across layers (one line per word)
    # Uses a SHARED PCA basis (fit on all layers stacked together) so
    # positions are comparable across layers, unlike the per-layer grid
    # above which refits PCA independently at each layer.
    # ------------------------------------------------------------------
    all_layers = np.stack([hs[0].float().numpy() for hs in hidden_states])  # [n_layers, seq_len, hidden_dim]
    flat = all_layers.reshape(-1, all_layers.shape[-1])                     # [n_layers*seq_len, hidden_dim]
    flat_centered = flat - flat.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(flat_centered, full_matrices=False)
    basis = vt[:2].T  # [hidden_dim, 2]

    proj_all = (flat_centered @ basis).reshape(n_layers, seq_len, 2)  # [n_layers, seq_len, 2]

    fig, ax = plt.subplots(figsize=(8, 7))
    for i, tok in enumerate(tokens):
        path = proj_all[:, i, :]  # [n_layers, 2]
        ax.plot(path[:, 0], path[:, 1], color=colors[i], alpha=0.6, linewidth=1.5)
        ax.scatter(path[:, 0], path[:, 1], color=colors[i], s=20)
        ax.annotate(tok, (path[-1, 0], path[-1, 1]), fontsize=9, color=colors[i], fontweight="bold")
        ax.scatter(path[0, 0], path[0, 1], color=colors[i], marker="x", s=80)  # start = embedding layer

    ax.set_title("Token trajectories across layers\n(x = embedding layer, label = final layer, shared PCA basis)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.tight_layout()
    traj_path = os.path.join(args.output_dir, "token_trajectories.png")
    plt.savefig(traj_path, dpi=130)
    plt.close(fig)
    print(f"Saved {traj_path}")


if __name__ == "__main__":
    main()
