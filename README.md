# GPT-Neo Layer-wise Feature Extraction

Small utility script for inspecting the internal, layer-by-layer representations
of a GPT-Neo model (e.g. `EleutherAI/gpt-neo-125M`, `1.3B`, `2.7B` — the same
family used in [joeljang/knowledge-unlearning](https://github.com/joeljang/knowledge-unlearning)).

For a given input text, it extracts:
- hidden states (the residual stream) at every transformer layer
- attention weights at every layer
- summary stats: mean activation norm per layer, cosine similarity between
  consecutive layers
- a 2D PCA plot of token representations per layer

## Setup

```bash
python -m venv .venv
source .venv/bin/activate     # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requires a machine with internet access to the Hugging Face Hub (to download
model weights) and, ideally, a CUDA-capable GPU for anything larger than the
125M checkpoint.

## Usage

Basic run on the smallest checkpoint:

```bash
python gpt_neo_layerwise.py \
    --model_name EleutherAI/gpt-neo-125M \
    --text "The quick brown fox jumps over the lazy dog." \
    --output_dir ./layerwise_out
```

This prints per-layer stats to the console and writes to `layerwise_out/`:
- `layerwise_pca.png` — PCA plot of token representations per layer
- `summary.json` — norms, cosine similarities, and the tokenized input

### Larger checkpoints

For `EleutherAI/gpt-neo-1.3B` or `2.7B`, use half precision and let
`accelerate` place the model across available GPU memory:

```bash
python gpt_neo_layerwise.py \
    --model_name EleutherAI/gpt-neo-1.3B \
    --fp16 --device_map auto \
    --text "Your text here." \
    --output_dir ./layerwise_out
```

### Saving raw tensors

Add `--save_hidden_states` to dump the raw hidden-state and attention tensors
(`hidden_states.pt`, `attentions.pt`) for further analysis (probing,
logit-lens decoding, comparing checkpoints before/after fine-tuning or
unlearning, etc.):

```bash
python gpt_neo_layerwise.py --model_name EleutherAI/gpt-neo-125M \
    --text "..." --output_dir ./layerwise_out --save_hidden_states
```

### Aggregating over many examples

`aggregate_stats()` in the script averages layer norms and consecutive-layer
cosine similarity over a list of texts instead of a single sentence — useful
if you want a stable layer-wise trend rather than one noisy example, e.g.
when comparing a base checkpoint against an unlearned one.

## Files

| File | Purpose |
|---|---|
| `gpt_neo_layerwise.py` | Main script: load model, run forward pass, extract and plot layer-wise features |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Excludes generated outputs, caches, and virtual envs from version control |

## Notes

- The smallest checkpoint (125M) runs fine on CPU for quick testing, but a
  GPU is strongly recommended for anything larger.
- `--device_map auto` requires `accelerate` to be installed and only makes
  sense on a machine with a GPU (or multiple GPUs).
