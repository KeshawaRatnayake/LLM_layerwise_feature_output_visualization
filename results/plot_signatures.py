"""
Plot the raw 768-dimensional feature vector ("signature") for each word,
at each layer -- no PCA, no dimensionality reduction. Just the actual
768 numbers per word, plotted as a line so you can see the shape/pattern.

Input: hidden_states.pt (list of 13 tensors, each [1, seq_len, hidden_dim])
       summary.json (for token labels)

Output: one PNG per layer, each containing 10 overlaid lines (one per word),
        x-axis = dimension index (0-767), y-axis = activation value.
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
    return [f"tok{i}" for i in range(n_tokens)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_states", default="hidden_states.pt")
    parser.add_argument("--summary", default="summary.json")
    parser.add_argument("--output_dir", default="./plots")
    parser.add_argument("--layers", default="all",
                         help='"all" to plot every layer, or comma-separated indices e.g. "0,6,12"')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    hidden_states = torch.load(args.hidden_states, map_location="cpu")
    n_layers = len(hidden_states)
    seq_len = hidden_states[0].shape[1]
    hidden_dim = hidden_states[0].shape[-1]
    tokens = load_tokens(args.summary, seq_len)

    print(f"Loaded {n_layers} layers, seq_len={seq_len}, hidden_dim={hidden_dim}")
    print("Tokens:", tokens)

    if args.layers == "all":
        layer_indices = list(range(n_layers))
    else:
        layer_indices = [int(x) for x in args.layers.split(",")]

    colors = cm.tab10(np.linspace(0, 1, seq_len))
    x = np.arange(hidden_dim)  # 0..767

    for l in layer_indices:
        mat = hidden_states[l][0].float().numpy()  # [seq_len, hidden_dim]

        fig, ax = plt.subplots(figsize=(14, 5))
        for i, tok in enumerate(tokens):
            ax.plot(x, mat[i], color=colors[i], linewidth=0.8, alpha=0.8, label=tok)

        label = "embeddings" if l == 0 else f"layer {l}"
        ax.set_title(f"Raw feature signature per word -- {label}")
        ax.set_xlabel("Feature dimension (0-767)")
        ax.set_ylabel("Activation value")
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        plt.tight_layout()

        out_path = os.path.join(args.output_dir, f"signature_{label.replace(' ', '_')}.png")
        plt.savefig(out_path, dpi=130)
        plt.close(fig)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
