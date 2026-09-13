"""
Layer-wise feature extraction for GPT-Neo (HuggingFace transformers).

Loads a GPT-Neo checkpoint (same family used in joeljang/knowledge-unlearning:
EleutherAI/gpt-neo-125M / 1.3B / 2.7B), runs text through it, and extracts:
  - hidden states at every layer (the residual stream)
  - attention weights at every layer
  - summary stats: activation norm growth, consecutive-layer cosine similarity
  - a 2D PCA plot of token representations per layer
  - optional: aggregate stats over many examples (not just one sentence)

Usage (on your remote GPU box):
    pip install -r requirements.txt   # see bottom of this file
    python gpt_neo_layerwise.py --model_name EleutherAI/gpt-neo-125M \
        --text "The quick brown fox jumps over the lazy dog." \
        --output_dir ./layerwise_out

For bigger checkpoints (1.3B / 2.7B), add --fp16 and/or --device_map auto
to spread across GPU memory the same way the knowledge-unlearning repo does.
"""

import argparse
import os
import json

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model(model_name: str, device: str, fp16: bool, device_map: str | None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = dict(output_hidden_states=True, output_attentions=True)
    if fp16:
        load_kwargs["torch_dtype"] = torch.float16
    if device_map:
        load_kwargs["device_map"] = device_map  # e.g. "auto" -> needs `accelerate`

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    if not device_map:
        model = model.to(device)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def run_forward(tokenizer, model, text: str, device: str):
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device if hasattr(model, "device") else device) for k, v in inputs.items()}
    out = model(**inputs)
    # hidden_states: tuple of (n_layers + 1) tensors, each [batch, seq_len, hidden_dim]
    # index 0 is the input embedding layer; index i is the output of transformer block i
    return inputs, out


def layer_norms(hidden_states):
    # skip index 0 (embeddings) to line up with "layer 1..N" naming
    return [hs[0].norm(dim=-1).mean().item() for hs in hidden_states[1:]]


def consecutive_cosine_sim(hidden_states):
    sims = []
    for i in range(1, len(hidden_states) - 1):
        a = hidden_states[i][0].flatten(0, 0)   # [seq_len, hidden_dim] for batch item 0
        b = hidden_states[i + 1][0].flatten(0, 0)
        sims.append(F.cosine_similarity(a, b, dim=-1).mean().item())
    return sims


def pca_plot_per_layer(hidden_states, token_strs, out_path):
    n_layers = len(hidden_states) - 1  # exclude embedding layer for the grid
    fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 4))
    if n_layers == 1:
        axes = [axes]

    for l in range(n_layers):
        seq = hidden_states[l + 1][0].float().cpu().numpy()  # [seq_len, hidden_dim]
        seq_centered = seq - seq.mean(axis=0, keepdims=True)
        u, s, vt = np.linalg.svd(seq_centered, full_matrices=False)
        proj = seq_centered @ vt[:2].T

        ax = axes[l]
        for i, tok in enumerate(token_strs):
            ax.scatter(proj[i, 0], proj[i, 1], s=40)
            ax.annotate(tok, (proj[i, 0], proj[i, 1]), fontsize=7)
        ax.set_title(f"Layer {l + 1}")

    plt.suptitle("Per-layer PCA of token representations")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)


def aggregate_stats(tokenizer, model, texts, device):
    """Average norms / cosine sims over many examples instead of one sentence."""
    all_norms, all_sims = [], []
    for text in texts:
        _, out = run_forward(tokenizer, model, text, device)
        all_norms.append(layer_norms(out.hidden_states))
        all_sims.append(consecutive_cosine_sim(out.hidden_states))
    return np.array(all_norms).mean(axis=0), np.array(all_sims).mean(axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="EleutherAI/gpt-neo-125M")
    parser.add_argument("--text", default="The quick brown fox jumps over the lazy dog.")
    parser.add_argument("--output_dir", default="./layerwise_out")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--device_map", default=None, help='e.g. "auto" for large checkpoints (needs `accelerate`)')
    parser.add_argument("--save_hidden_states", action="store_true", help="dump raw tensors to disk")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Loading {args.model_name} on {args.device} (fp16={args.fp16}, device_map={args.device_map}) ...")
    tokenizer, model = load_model(args.model_name, args.device, args.fp16, args.device_map)
    print(f"Loaded. n_layers={model.config.num_layers if hasattr(model.config, 'num_layers') else model.config.num_hidden_layers}, "
          f"hidden_size={model.config.hidden_size}")

    inputs, out = run_forward(tokenizer, model, args.text, args.device)
    token_strs = [tokenizer.decode([t]) for t in inputs["input_ids"][0]]

    norms = layer_norms(out.hidden_states)
    sims = consecutive_cosine_sim(out.hidden_states)
    print("\nMean activation norm per layer:", [round(n, 2) for n in norms])
    print("Cosine similarity between consecutive layers:", [round(s, 3) for s in sims])

    pca_path = os.path.join(args.output_dir, "layerwise_pca.png")
    pca_plot_per_layer(out.hidden_states, token_strs, pca_path)
    print(f"Saved PCA plot to {pca_path}")

    summary = {
        "model_name": args.model_name,
        "text": args.text,
        "tokens": token_strs,
        "layer_norms": norms,
        "consecutive_cosine_sim": sims,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if args.save_hidden_states:
        # each tensor is [1, seq_len, hidden_dim]; index 0 = embeddings, 1..N = layer outputs
        torch.save([hs.cpu() for hs in out.hidden_states],
                    os.path.join(args.output_dir, "hidden_states.pt"))
        torch.save([att.cpu() for att in out.attentions],
                    os.path.join(args.output_dir, "attentions.pt"))
        print("Saved raw hidden_states.pt and attentions.pt")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# requirements.txt (create this alongside the script):
#
#   torch
#   transformers>=4.30
#   accelerate      # needed for --device_map auto with 1.3B / 2.7B checkpoints
#   matplotlib
#   numpy
#
# ---------------------------------------------------------------------------
# For aggregate stats over a dataset instead of one sentence, e.g. reusing
# the same validation data the knowledge-unlearning repo uses:
#
#   texts = [...]  # list of strings, e.g. loaded from their Datasets.py / data/
#   mean_norms, mean_sims = aggregate_stats(tokenizer, model, texts, args.device)
#
# ---------------------------------------------------------------------------
