"""
What does the model look at? Attention maps, one per head, on a real sentence.

    python scripts/attention_maps.py --ckpt showcase/run1/ckpt.pt --tokenizer data/tokenizer.json \
        --prompt "Capitú olhou para mim e disse" --layer -1 --out docs/img/attention_run1.png

Each panel is one head of one block. Row = the position doing the looking, column = the
token it looks at; darker = more weight. Everything above the diagonal is zero (causal mask).

Inspired by the "Que regarde le modèle ?" figure of a class demo in IFT6758 (UdeM, Mila), which follows Karpathy's
"Let's build GPT", adapted to this model: BPE tokens instead of characters, RoPE applied to
q and k, and grouped-query attention (heads sharing K/V).

The weights are recomputed here from the model's own q/k projections, because
`causal_attention` in model.py returns only the output, not the weights.
"""
import argparse
import math
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ModelConfig            # noqa: E402
from model import TinyGPT, apply_rope, repeat_kv  # noqa: E402
from tokenizer import BPETokenizer        # noqa: E402

matplotlib.use("Agg")
INK, MUTED, PAPER = "#1A2130", "#5E6778", "#F4EEE3"


@torch.no_grad()
def attention_weights(model: TinyGPT, idx: torch.Tensor) -> list:
    """Run the model and return, for every block, the attention weights (H, T, T) of sequence 0."""
    model.eval()
    B, T = idx.shape
    cos, sin = model.rope_cos[:T], model.rope_sin[:T]
    x = model.tok_emb(idx)
    maps = []
    for block in model.blocks:
        attn = block.attn
        h = block.norm1(x)
        q = attn.wq(h).view(B, T, attn.n_head, attn.head_dim).transpose(1, 2)
        k = attn.wk(h).view(B, T, attn.n_kv_head, attn.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        k = repeat_kv(k, attn.n_rep)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(attn.head_dim)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        weights = F.softmax(scores.masked_fill(mask, float("-inf")), dim=-1)   # (B, H, T, T)
        maps.append(weights[0])
        x = block(x, cos, sin)   # the real forward pass, so the next block sees the right input
    return maps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/ckpt.pt")
    ap.add_argument("--tokenizer", default="data/tokenizer.json")
    ap.add_argument("--prompt", default="Capitú olhou para mim e disse que não")
    ap.add_argument("--layer", type=int, default=-1, help="block index (-1 = last)")
    ap.add_argument("--out", default="out/attention.png")
    args = ap.parse_args()

    tok = BPETokenizer.load(args.tokenizer)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = TinyGPT(ModelConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])

    ids = tok.encode(args.prompt)
    labels = [tok.decode([i]).replace(" ", "␣") for i in ids]
    maps = attention_weights(model, torch.tensor([ids]))
    layer = args.layer % len(maps)
    w = maps[layer]                                             # (H, T, T)
    H, T = w.shape[0], w.shape[1]

    n_cols = math.ceil(H / 2)
    fig, axes = plt.subplots(2, n_cols, figsize=(3.4 * n_cols, 7.2), dpi=130)
    fig.patch.set_facecolor("white")
    for i, ax in enumerate(axes.flat):
        if i >= H:
            ax.axis("off")
            continue
        ax.imshow(w[i], cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"head {i + 1}" + ("  (K/V group A)" if i < H // 2 else "  (K/V group B)")
                     if model.config.n_kv_head == 2 else f"head {i + 1}", fontsize=10, color=INK)
        ax.set_xticks(range(T), labels, rotation=90, fontsize=8)
        ax.set_yticks(range(T), labels, fontsize=8)
        ax.tick_params(length=0, colors=MUTED)
        for s in ax.spines.values():
            s.set_color("#D6CBB5")
    fig.suptitle(f"Attention weights, block {layer + 1} of {len(maps)} — "
                 f"\"{args.prompt}\"  (row looks at column)", fontsize=11, color=INK, y=0.995)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, facecolor="white")
    print(f"saved {args.out}")

    # a text summary: where does each head put most of its weight, averaged over positions?
    for i in range(H):
        self_w = w[i].diagonal().mean().item()
        prev_w = w[i].diagonal(-1).mean().item() if T > 1 else 0
        first_w = w[i][:, 0].mean().item()
        print(f"head {i + 1}: self {self_w:.2f} | previous token {prev_w:.2f} | first token {first_w:.2f}")


if __name__ == "__main__":
    main()
