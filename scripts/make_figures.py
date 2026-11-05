"""
Figures for the library (docs/img/fig_*.png), computed from this project's own code and data.

    python scripts/make_figures.py                 # all figures that have their inputs available
    python scripts/make_figures.py --only rope lr  # a subset

Every figure is generated, not drawn by hand: the BPE example runs the real get_stats/merge,
the RoPE plots call precompute_rope/apply_rope, the learning-rate curves call get_lr with the
real presets, the temperature plot uses the run-1 model's actual predictions, and so on.
Change the code, rerun the script, and the figures follow.

Style: white background, the project palette (ink, sage, brass), IBM Plex fonts (SIL Open Font
License, in scripts/fonts/). The idea of illustrating each step with small plots and heatmaps
is inspired by a class demo in IFT6758 (UdeM, Mila); the figures themselves are this project's.

Inputs (each figure is skipped with a message if its input is missing):
    data/tokenizer.json             run-1 tokenizer (Machado, 2,048)
    data/pretrain/tokenizer.json    run-2 tokenizer (4,096)
    data/pretrain/manifest.csv      pretraining corpus manifest
    showcase/run1/ckpt.pt           run-1 best checkpoint
"""
import argparse
import csv
import math
import os
import sys
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config import PRESETS, ModelConfig            # noqa: E402
from model import TinyGPT, apply_rope, precompute_rope  # noqa: E402
from tokenizer import BPETokenizer, get_stats, merge  # noqa: E402
from train import get_lr                           # noqa: E402

matplotlib.use("Agg")
OUT = os.path.join(ROOT, "docs", "img")

# ------------------------------------------------------------------ style
INK, MUTED, LINE, GRID = "#1A2130", "#5E6778", "#D6CBB5", "#EEE8DD"
SAGE, SAGE_D, SAGE_L = "#5B8A72", "#2F5D4A", "#DCE8E0"
BRASS, BRASS_D, BRASS_L = "#A88443", "#7A5E2A", "#F1E6D2"
CMAP = LinearSegmentedColormap.from_list("tl", ["#FFFFFF", SAGE_L, SAGE, SAGE_D])

FONT_DIR = os.path.join(ROOT, "scripts", "fonts")
SANS, MONO, SERIF = "DejaVu Sans", "DejaVu Sans Mono", "DejaVu Serif"
if os.path.isdir(FONT_DIR):
    for f in os.listdir(FONT_DIR):
        if f.endswith(".ttf"):
            font_manager.fontManager.addfont(os.path.join(FONT_DIR, f))
    SANS, MONO, SERIF = "IBM Plex Sans", "IBM Plex Mono", "Fraunces"
# The Plex files here are the Latin subset: symbols like →, √, ∞, ␣, σ, ₁ fall back to DejaVu.
SANS = [SANS, "DejaVu Sans"]
MONO = [MONO, "DejaVu Sans Mono"]

plt.rcParams.update({
    "font.family": SANS, "font.size": 10.5, "text.color": INK,
    "axes.edgecolor": LINE, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "medium", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.15,
})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved docs/img/{name}")


def suptitle(fig, text, sub=None):
    """Title and subtitle above the plot, offset in points so spacing is the same at any figure size."""
    from matplotlib.transforms import offset_copy
    fig.text(0.0, 1.0, text, fontsize=14, fontweight="medium", color=INK, ha="left", va="bottom",
             transform=offset_copy(fig.transFigure, fig=fig, y=30 if sub else 10, units="points"))
    if sub:
        fig.text(0.0, 1.0, sub, fontsize=10.5, color=MUTED, ha="left", va="bottom",
                 transform=offset_copy(fig.transFigure, fig=fig, y=10, units="points"))


def chips(ax, tokens, y, x0=0.0, colors=None, h=0.62, gap=0.18, size=10.5, unit=0.62):
    """Draw tokens as rounded chips on a line; returns the x after the last chip."""
    x = x0
    centers = []
    for i, t in enumerate(tokens):
        label = t.replace(" ", "␣").replace("\n", "↵")
        w = max(len(label), 1) * unit + 0.5
        face, edge = colors[i] if colors else (SAGE_L, SAGE)
        ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.14",
                                    linewidth=1, facecolor=face, edgecolor=edge))
        ax.text(x + w / 2, y, label, ha="center", va="center", fontsize=size, family=MONO, color=INK)
        centers.append(x + w / 2)
        x += w + gap
    return x, centers


def blank(ax):
    ax.set_axis_off()
    ax.grid(False)


def load_tok(path):
    p = os.path.join(ROOT, path)
    return BPETokenizer.load(p) if os.path.exists(p) else None


# ================================================================== 00 — data
def fig_machado_books():
    books = [("Quincas Borba", 453), ("Esaú e Jacó", 415), ("Dom Casmurro", 378),
             ("Memórias Póstumas de Brás Cubas", 364), ("Helena", 336), ("Iaiá Garcia", 330),
             ("Papéis Avulsos", 322), ("Histórias sem Data", 299), ("Memorial de Aires", 285),
             ("Relíquias de Casa Velha", 277), ("Poesias Completas", 252), ("A Mão e a Luva", 204)]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    names = [b[0] for b in books][::-1]
    vals = [b[1] for b in books][::-1]
    bars = ax.barh(names, vals, color=SAGE, height=0.62)
    ax.bar_label(bars, labels=[f"{v}k" for v in vals], padding=4, fontsize=9.5, color=MUTED)
    ax.set_xlabel("characters (thousands)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 520)
    suptitle(fig, "The run-1 corpus: Machado de Assis on Project Gutenberg",
             "12 books, 3.91 M characters — the whole training set of run 1")
    save(fig, "fig_machado_books.png")


# ================================================================== 01 — tokenizer
def fig_bpe_merges():
    """The não / ação worked example, computed with the real get_stats and merge."""
    words = {("n", "ã", "o"): 6, ("a", "ç", "ã", "o"): 3, ("m", "ã", "o"): 2,
             ("e", "m", "o", "ç", "ã", "o"): 2}
    rounds = [("start: characters", words, None, None)]
    for _ in range(3):
        stats = get_stats(words)
        pair = max(stats, key=stats.get)
        new = pair[0] + pair[1]
        words = {merge(w, pair, new): f for w, f in words.items()}
        rounds.append((f"merge  {pair[0]} + {pair[1]}  →  {new}", words, new, stats[pair]))

    fig, ax = plt.subplots(figsize=(10, 3.7))
    blank(ax)
    y = len(rounds) - 1
    for label, ws, new, count in rounds:
        ax.text(-0.3, y, label, ha="right", va="center", fontsize=10.5,
                color=BRASS_D if new else MUTED, family=MONO if new else SANS)
        if count:
            ax.text(-0.3, y - 0.32, f"pair seen {count}×", ha="right", va="center", fontsize=9, color=MUTED)
        x = 0.4
        for w, f in ws.items():
            cols = [(BRASS_L, BRASS) if t == new else (SAGE_L, SAGE) for t in w]
            x, _ = chips(ax, list(w), y, x0=x, colors=cols, size=11, gap=0.06, unit=0.7)
            ax.text(x + 0.05, y, f"×{f}", va="center", fontsize=9.5, color=MUTED)
            x += 1.3
        y -= 1
    ax.set_xlim(-6.2, 27)
    ax.set_ylim(-0.75, len(rounds) - 0.45)
    suptitle(fig, "Byte Pair Encoding, three rounds",
             "count adjacent pairs (weighted by word frequency) → merge the most frequent everywhere → repeat. "
             "Computed with tokenizer.get_stats / merge.")
    save(fig, "fig_bpe_merges.png")


def fig_tokenizer_cuts():
    t1, t2 = load_tok("data/tokenizer.json"), load_tok("data/pretrain/tokenizer.json")
    if not (t1 and t2):
        print("  skip fig_tokenizer_cuts (needs both tokenizers)")
        return
    sentence = "Nem por isso me zanguei. Contei a anecdota aos amigos da cidade."
    fig, ax = plt.subplots(figsize=(13, 3.0))
    blank(ax)
    rows = [("run 1 · 2,048 tokens · trained on Machado", t1, 1.45),
            ("run 2 · 4,096 tokens · trained on 90 M words", t2, 0.0)]
    xmax = 0
    for label, tok, y in rows:
        pieces = [tok.decode([i]) for i in tok.encode(sentence)]
        cols = [((SAGE_L, SAGE) if p.strip() and len(p.strip()) > 2 and p.startswith(" ") else (BRASS_L, BRASS))
                for p in pieces]
        ax.text(0, y + 0.52, f"{label}  —  {len(pieces)} tokens", fontsize=10, color=MUTED, va="bottom")
        x_end, _ = chips(ax, pieces, y, colors=cols, size=9.5, gap=0.1, unit=0.58)
        xmax = max(xmax, x_end)
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.45, 2.25)
    suptitle(fig, "The same sentence, two tokenizers",
             "sage = whole word in one token · brass = piece of a word, punctuation or a short word")
    save(fig, "fig_tokenizer_cuts.png")


# ================================================================== 02 — attention
def fig_attention_mask():
    torch.manual_seed(3)
    toks = ["Não", "␣tive", "␣filhos", ",", "␣não"]
    T, d = len(toks), 8
    q, k = torch.randn(T, d), torch.randn(T, d)
    scores = q @ k.T / math.sqrt(d)
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    masked = scores.masked_fill(mask, float("-inf"))
    weights = F.softmax(masked, dim=-1)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.1), gridspec_kw={"wspace": 0.35})
    panels = [("1 · scores  q·k / √d", scores, False), ("2 · causal mask: future → −∞", masked, True),
              ("3 · softmax: each row sums to 1", weights, False)]
    for ax, (title, m, is_masked) in zip(axes, panels):
        ax.grid(False)
        shown = m.clone()
        if is_masked:
            shown[mask] = float("nan")
        vmin, vmax = (0, 1) if title.startswith("3") else (-2.2, 2.2)
        cmap = CMAP if title.startswith("3") else LinearSegmentedColormap.from_list("d", [BRASS_L, "#FFFFFF", SAGE_L])
        cmap.set_bad("#ECECEC")
        ax.imshow(shown.numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
        for i in range(T):
            for j in range(T):
                v = m[i, j].item()
                txt = "−∞" if v == float("-inf") else f"{v:.2f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                        color=("white" if title.startswith("3") and v > 0.6 else INK if v != float("-inf") else MUTED))
        ax.set_xticks(range(T), toks, family=MONO, fontsize=9)
        ax.set_yticks(range(T), toks, family=MONO, fontsize=9)
        ax.tick_params(length=0)
        ax.set_title(title, fontsize=11)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("token doing the looking (query)", color=MUTED)
    axes[1].set_xlabel("token being looked at (key)", color=MUTED)
    suptitle(fig, "Causal attention in three steps",
             "random q and k for five tokens; the first row can only see itself, the last row sees everything before it")
    save(fig, "fig_attention_mask.png")


def fig_sqrt_d():
    torch.manual_seed(0)
    d, T = 64, 16
    q, k = torch.randn(d), torch.randn(T, d)
    raw = k @ q
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, (title, s) in zip(axes, [("without scaling: softmax(q·k)", raw),
                                     (f"with scaling: softmax(q·k / √{d})", raw / math.sqrt(d))]):
        p = F.softmax(s, dim=0).numpy()
        cols = [BRASS if i == p.argmax() else SAGE for i in range(T)]
        ax.bar(range(T), p, color=cols, width=0.7)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(T), [str(i) for i in range(T)], fontsize=8.5)
        ax.set_xlabel("key position")
        ax.grid(axis="x", visible=False)
        ax.text(0.02, 0.94, f"largest weight {p.max():.2f}\nscores range ±{s.abs().max():.0f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=9.5, color=MUTED)
    axes[0].set_ylabel("attention weight")
    suptitle(fig, "Why divide by √d",
             f"one query against 16 keys, d = {d}: unscaled scores are ~8× larger, softmax becomes nearly one-hot "
             "and the other keys get almost no gradient")
    save(fig, "fig_sqrt_d.png")


# ================================================================== 03 — modern upgrades
def fig_rope():
    cos, sin = precompute_rope(64, 256)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.1))
    ax = axes[0]
    pos = np.arange(0, 64)
    for i, c in [(3, BRASS), (8, SAGE), (16, SAGE_D), (31, MUTED)]:
        ax.plot(pos, cos[:64, i].numpy(), color=c, lw=2, label=f"pair {i}  (θ = {10000 ** (-2 * i / 64):.2g})")
    ax.set_xlabel("position p")
    ax.set_ylabel("cos(p · θ)")
    ax.set_title("Each pair of dimensions turns at its own speed", fontsize=11)
    ax.legend(loc="upper center", fontsize=9, ncol=2, bbox_to_anchor=(0.5, 1.0))
    ax.set_ylim(-1.15, 1.9)

    ax = axes[1]
    torch.manual_seed(1)
    q, k = torch.randn(64), torch.randn(64)
    offsets = np.arange(0, 41)
    for start, c, ls in [(0, BRASS, "-"), (60, SAGE, "--"), (150, SAGE_D, ":")]:
        vals = []
        for off in offsets:
            m, n = start + off, start
            qm = apply_rope(q.view(1, 1, 1, 64), cos[m:m + 1], sin[m:m + 1])
            kn = apply_rope(k.view(1, 1, 1, 64), cos[n:n + 1], sin[n:n + 1])
            vals.append((qm * kn).sum().item())
        ax.plot(offsets, vals, color=c, lw=2.2 if ls == "-" else 1.8, ls=ls,
                label=f"key at position {start}")
    ax.set_xlabel("distance m − n between query and key")
    ax.set_ylabel("score  rope(q, m) · rope(k, n)")
    ax.set_title("The score depends only on the distance", fontsize=11)
    ax.legend(fontsize=9, title="the three curves coincide", title_fontsize=9)
    suptitle(fig, "RoPE, computed with precompute_rope and apply_rope",
             "left: 4 of the 32 pairs, fast to slow, like the hands of a clock · right: the same q and k placed at three "
             "different absolute positions give identical curves")
    save(fig, "fig_rope.png")


def fig_activations():
    z = torch.linspace(-5, 4, 400)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9))
    ax = axes[0]
    ax.plot(z, F.relu(z), color=MUTED, lw=2, label="ReLU  max(0, z)  — 2017")
    ax.plot(z, F.silu(z), color=SAGE, lw=2.4, label="SiLU  z·σ(z)  — SwiGLU's gate")
    ax.axhline(0, color=LINE, lw=1)
    ax.axvline(0, color=LINE, lw=1)
    ax.set_xlabel("z")
    ax.set_title("ReLU vs SiLU", fontsize=11)
    ax.legend(fontsize=9.5, loc="upper left")
    ax.annotate("small negative values pass:\ngradient still flows", xy=(-1.3, -0.28), xytext=(-4.8, 1.6),
                fontsize=9.5, color=MUTED, arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))

    ax = axes[1]
    ax.grid(False)
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_title("SwiGLU in one picture (per token)", fontsize=11)

    def box(x, y, w, h, fc, ec, text, sub=None):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12", fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h / 2 + (0.18 if sub else 0), text, ha="center", va="center", fontsize=10)
        if sub:
            ax.text(x + w / 2, y + h / 2 - 0.3, sub, ha="center", va="center", fontsize=8.5, color=MUTED)

    box(0.2, 2.4, 1.3, 1.2, "white", LINE, "x", "384")
    box(2.6, 3.8, 2.4, 1.2, BRASS_L, BRASS, "SiLU(W₁x)", "gate · 1,024")
    box(2.6, 1.0, 2.4, 1.2, SAGE_L, SAGE, "W₃x", "content · 1,024")
    box(5.9, 2.4, 1.0, 1.2, "white", LINE, "⊙")
    box(7.8, 2.4, 1.9, 1.2, "white", LINE, "W₂", "back to 384")
    for (x1, y1, x2, y2) in [(1.5, 3.0, 2.6, 4.4), (1.5, 3.0, 2.6, 1.6), (5.0, 4.4, 5.9, 3.2),
                             (5.0, 1.6, 5.9, 2.8), (6.9, 3.0, 7.8, 3.0)]:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.1))
    ax.text(5, 0.25, "the gate decides, per dimension, how much of the content passes", ha="center",
            fontsize=9.5, color=MUTED)
    suptitle(fig, "The feed-forward network: from ReLU to SwiGLU")
    save(fig, "fig_activations.png")


def fig_kv_cache():
    cfg = ModelConfig()
    d, L, ctx = cfg.head_dim, cfg.n_layer, cfg.block_size
    designs = [("multi-head (2017)\n6 K/V heads", 6, MUTED), ("grouped-query (this model)\n2 K/V heads", 2, SAGE),
               ("multi-query\n1 K/V head", 1, SAGE_L)]
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    names, vals, cols = [], [], []
    for name, hkv, c in designs:
        per_token = 2 * hkv * d * 4 * L      # K and V · heads · dims · 4 bytes (float32) · layers
        names.append(name)
        vals.append(per_token * ctx / 1e6)
        cols.append(c)
    bars = ax.barh(names[::-1], vals[::-1], color=cols[::-1], height=0.55, edgecolor=[LINE] * 3)
    ax.bar_label(bars, labels=[f"{v:.2f} MB" for v in vals[::-1]], padding=5, fontsize=10)
    ax.set_xlabel(f"KV cache for one {ctx}-token sequence (float32, {L} layers, head dim {d})")
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.set_xlim(0, max(vals) * 1.25)
    suptitle(fig, "What GQA saves: the KV cache",
             "memory kept per sequence during generation = 2 (K and V) × K/V heads × head dim × 4 bytes × layers × tokens")
    save(fig, "fig_kv_cache.png")


# ================================================================== 04 — assembling
def fig_cross_entropy():
    p = np.logspace(-4.2, 0, 400)
    fig, ax = plt.subplots(figsize=(9, 3.9))
    ax.plot(p, -np.log(p), color=SAGE, lw=2.4)
    ax.set_xscale("log")
    ax.set_xlabel("probability the model gave to the real next token")
    ax.set_ylabel("loss  −ln p")
    pts = [(1.0, "p = 1 → 0: certain and right", (-175, -16)), (0.5, "p = 0.5 → 0.69", (-100, 10)),
           (0.05, "p = 0.05 → 3.00", (10, 6)),
           (1 / 2048, "p = 1/2,048 → 7.62: uniform guess (run 1)", (12, -4)),
           (1 / 4096, "p = 1/4,096 → 8.32: uniform guess (run 2)", (12, 6))]
    for x, txt, off in pts:
        y = -math.log(x)
        ax.scatter([x], [y], color=BRASS, zorder=5, s=36, edgecolor="white", lw=1)
        ax.annotate(txt, (x, y), xytext=off, textcoords="offset points", fontsize=9, color=INK)
    ax.set_ylim(-0.3, 10)
    suptitle(fig, "Cross-entropy: the price of a wrong guess",
             "the log makes confident mistakes expensive; a fresh model guesses uniformly, so its loss starts at ln V")
    save(fig, "fig_cross_entropy.png")


def fig_temperature():
    """Same prompt, three temperatures, two models: run 1 (Machado only) and run 2 fine-tuned."""
    runs = [("run 1 · Machado only", "showcase/run1/ckpt.pt", "data/tokenizer.json", BRASS_D, BRASS, BRASS_L),
            ("run 2 · pretrained + fine-tuned", "out/finetune/ckpt.pt", "data/pretrain/tokenizer.json", SAGE_D, SAGE, SAGE_L)]
    for _, ck, tk, *_ in runs:
        if not (os.path.exists(os.path.join(ROOT, ck)) and os.path.exists(os.path.join(ROOT, tk))):
            print(f"  skip fig_temperature (needs {ck} and {tk})")
            return
    prompt = "Capitú olhou para"
    temps = [0.5, 1.0, 1.5]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.6), sharey=True)
    for row, (name, ck, tk, *colors) in enumerate(runs):
        tok = load_tok(tk)
        ckpt = torch.load(os.path.join(ROOT, ck), map_location="cpu")
        model = TinyGPT(ModelConfig(**ckpt["model_config"]))
        model.load_state_dict(ckpt["model"])
        model.eval()
        with torch.no_grad():
            logits, _ = model(torch.tensor([tok.encode(prompt)]))
        logits = logits[0, -1]
        top = torch.topk(logits, 10).indices
        labels = [tok.decode([i]).replace(" ", "␣").replace("\n", "↵") for i in top.tolist()]
        for ax, tau, c in zip(axes[row], temps, colors):
            p = F.softmax(logits / tau, dim=-1)[top].numpy()
            ax.bar(range(10), p, color=c, edgecolor=colors[1], width=0.7)
            ax.set_xticks(range(10), labels, rotation=55, ha="right", family=MONO, fontsize=9)
            ax.set_title(f"temperature {tau}", fontsize=10.5)
            ax.grid(axis="x", visible=False)
            ax.text(0.97, 0.93, f"top 10 hold {p.sum():.0%}\nof the probability", transform=ax.transAxes,
                    ha="right", va="top", fontsize=9, color=MUTED)
        axes[row][0].set_ylabel(f"{name}\nprobability of the next token")
    fig.subplots_adjust(hspace=0.62, top=0.93)
    suptitle(fig, f"Temperature on a real prediction: \"{prompt} …\"",
             "top 10 next tokens of each model · below 1: sharper and safer · above 1: flatter and wilder")
    save(fig, "fig_temperature.png")


# ================================================================== 05 — training loop
def fig_batch_shift():
    tok = load_tok("data/tokenizer.json")
    text = "Não tive filhos, não transmiti a nenhuma criatura"
    pieces = [tok.decode([i]) for i in tok.encode(text)] if tok else text.split()
    T = min(len(pieces) - 1, 9)
    x, y = pieces[:T], pieces[1:T + 1]
    fig, ax = plt.subplots(figsize=(11, 2.8))
    blank(ax)
    unit = 0.62
    widths = [max(len(p.replace(" ", "␣")), len(q.replace(" ", "␣"))) * unit + 0.5 for p, q in zip(x, y)]
    xpos = 0.0
    for i in range(T):
        for row, toks, (fc, ec) in [(1.0, x, (SAGE_L, SAGE)), (0.0, y, (BRASS_L, BRASS))]:
            label = toks[i].replace(" ", "␣")
            ax.add_patch(FancyBboxPatch((xpos, row - 0.3), widths[i], 0.6, boxstyle="round,pad=0,rounding_size=0.14",
                                        fc=fc, ec=ec, lw=1))
            ax.text(xpos + widths[i] / 2, row, label, ha="center", va="center", family=MONO, fontsize=10)
        ax.annotate("", xy=(xpos + widths[i] / 2, 0.33), xytext=(xpos + widths[i] / 2, 0.67),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1))
        xpos += widths[i] + 0.2
    ax.text(-0.4, 1.0, "x  input", ha="right", va="center", fontsize=10.5, color=SAGE_D)
    ax.text(-0.4, 0.0, "y  target", ha="right", va="center", fontsize=10.5, color=BRASS_D)
    ax.set_xlim(-4.5, xpos + 0.2)
    ax.set_ylim(-0.6, 1.6)
    suptitle(fig, "One training window: the target is the input shifted by one token",
             f"{T} positions → {T} \"guess the next token\" exercises; a real window has 256, a batch 32 windows = 8,192")
    save(fig, "fig_batch_shift.png")


def fig_lr_schedule():
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9))
    for ax, preset, title in [(axes[0], "mac", "run 1 · mac · 5,000 steps"),
                              (axes[1], "mac-pretrain", "run 2 pretraining · 20,000 steps")]:
        _, tcfg = PRESETS[preset]
        import copy
        tcfg = copy.deepcopy(tcfg)
        if preset == "mac-pretrain":
            tcfg.max_iters = 20000
        steps = np.arange(0, tcfg.max_iters + 1, max(tcfg.max_iters // 1000, 1))
        lrs = [get_lr(int(s), tcfg) for s in steps]
        ax.plot(steps, lrs, color=SAGE, lw=2.4)
        ax.axvspan(0, tcfg.warmup_iters, color=BRASS_L, lw=0)
        ax.text(tcfg.max_iters * 0.06, tcfg.lr * 0.35, f"← warmup\n   {tcfg.warmup_iters} steps",
                fontsize=9, color=BRASS_D, va="top")
        mid = (tcfg.warmup_iters + tcfg.max_iters) // 2
        ax.scatter([mid], [get_lr(mid, tcfg)], color=BRASS, s=30, zorder=5)
        ax.annotate(f"halfway: {get_lr(mid, tcfg):.2e}", (mid, get_lr(mid, tcfg)), xytext=(8, 6),
                    textcoords="offset points", fontsize=9)
        ax.axhline(tcfg.min_lr, color=MUTED, lw=1, ls=(0, (3, 3)))
        ax.text(tcfg.max_iters, tcfg.min_lr * 1.6, f"floor {tcfg.min_lr:.0e}", ha="right", fontsize=9, color=MUTED)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("training step")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[0].set_ylabel("learning rate")
    suptitle(fig, "The learning-rate schedule, computed with train.get_lr",
             "linear warmup, then cosine decay from the peak to the floor")
    save(fig, "fig_lr_schedule.png")


# ================================================================== 06 — corpus
def fig_corpus():
    path = os.path.join(ROOT, "data", "pretrain", "manifest.csv")
    if not os.path.exists(path):
        print("  skip fig_corpus (needs data/pretrain/manifest.csv)")
        return
    rows = list(csv.DictReader(open(path, encoding="utf-8", newline="")))
    by_src = Counter()
    by_author, display = Counter(), {}
    for r in rows:
        w = int(r["words"])
        by_src[r["source"]] += w
        a = (r["creator"] or "Anonymous").split(";")[0].split("(")[0].split("|")[0].strip()
        a = ",".join(a.split(",")[:2]).strip() if "," in a else a
        a = a.rstrip(" ,0123456789-")
        if not a:
            continue
        key = " ".join(sorted(w_.lower() for w_ in a.replace(",", " ").split()))   # "Castelo Branco, Camilo" = "Camilo Castelo Branco"
        display.setdefault(key, a if len(a) <= 34 else a[:33] + "…")
        by_author[key] += w
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1.6], "wspace": 0.75})
    ax = axes[0]
    labels = {"gutenberg": "Project Gutenberg\n(proofread)", "pleias": "Portuguese-PD\n(OCR, filtered)"}
    srcs = ["gutenberg", "pleias"]
    vals = [by_src[s] / 1e6 for s in srcs]
    n_docs = [sum(1 for r in rows if r["source"] == s) for s in srcs]
    bars = ax.bar([labels[s] for s in srcs], vals, color=[SAGE, BRASS], width=0.55)
    ax.bar_label(bars, labels=[f"{v:.1f} M words\n{n} documents" for v, n in zip(vals, n_docs)],
                 padding=4, fontsize=9.5)
    ax.set_ylabel("words (millions)")
    ax.set_ylim(0, max(vals) * 1.3)
    ax.grid(axis="x", visible=False)
    ax.set_title("By source", fontsize=11)

    ax = axes[1]
    top = by_author.most_common(14)
    ax.barh([display[a] for a, _ in top][::-1], [w / 1e6 for _, w in top][::-1], color=SAGE, height=0.62)
    ax.set_xlabel("words (millions)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_title("Largest contributors (author field as catalogued)", fontsize=11)
    suptitle(fig, f"The run-2 pretraining corpus: {sum(by_src.values()) / 1e6:.0f} M words, {len(rows):,} documents",
             "open 19th-century Portuguese, Machado excluded · from data/pretrain/manifest.csv")
    save(fig, "fig_corpus.png")


# ================================================================== 06 — pretrain → fine-tune
def fig_finetune():
    import copy
    fig = plt.figure(figsize=(12, 6.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1], width_ratios=[3.2, 1], hspace=0.5, wspace=0.06, top=0.97)
    top = fig.add_subplot(gs[0, :]); blank(top)
    top.set_xlim(-0.05, 12.05); top.set_ylim(-1.0, 2.05)

    boxes = [  # title, body, face, edge
        ("Library", "1,760 books\n19th-century Portuguese\n160 M tokens\n(Machado left out)", BRASS_L, BRASS),
        ("1 · Pretrain", "20,000 steps · lr 1e-3\nlearns the language:\ngrammar, spelling,\ngenres", SAGE_L, SAGE),
        ("Checkpoint", "11.0 M weights\nbest at step 18,000\nval loss 3.434\nout/pretrain/ckpt.pt", "white", INK),
        ("2 · Fine-tune", "800 steps · lr 1e-4\nMachado only\n1.1 M tokens\nlearns his voice", SAGE_L, SAGE),
        ("Machado model", "same 11.0 M weights,\nnudged toward\nCapitu, Bentinho,\nthe ironic narrator", "white", SAGE_D),
    ]
    w, gap, y0, h = 2.04, 0.45, 0.0, 2.0
    for i, (title, body, face, edge) in enumerate(boxes):
        x = i * (w + gap)
        top.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                     facecolor=face, edgecolor=edge, linewidth=1.4))
        top.text(x + w / 2, y0 + h - 0.27, title, ha="center", va="center", fontsize=11.5,
                 fontweight="medium", color=INK)
        top.text(x + w / 2, y0 + h - 0.55, body, ha="center", va="top", fontsize=9.3, color=INK,
                 linespacing=1.45)
        if i < len(boxes) - 1:
            top.annotate("", xy=(x + w + gap - 0.04, y0 + h / 2), xytext=(x + w + 0.04, y0 + h / 2),
                         arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.4))
    # Machado's text feeding the fine-tune from below
    fx = 3 * (w + gap) + w / 2
    top.annotate("", xy=(fx, y0 - 0.02), xytext=(fx, -0.6),
                 arrowprops=dict(arrowstyle="-|>", color=BRASS, lw=1.4))
    top.text(fx, -0.72, "Machado's 12 books (data/machado)", ha="center", va="top", fontsize=9.3, color=BRASS_D)
    top.text(0, -0.72, "One tokenizer for both stages (4,096 tokens): token 812 must mean the same\n"
             "thing to embedding row 812 before and after — so Machado is re-encoded with it.",
             ha="left", va="top", fontsize=9.3, color=MUTED, linespacing=1.4)

    # bottom: the two learning-rate schedules, on one log axis
    ax1 = fig.add_subplot(gs[1, 0]); ax2 = fig.add_subplot(gs[1, 1], sharey=ax1)
    _, pre = PRESETS["mac-pretrain"]; pre = copy.deepcopy(pre); pre.max_iters = 20000
    _, ft = PRESETS["mac-finetune"]
    for ax, cfg, color, title in [(ax1, pre, SAGE, "pretraining · 20,000 steps"),
                                  (ax2, ft, BRASS, f"fine-tune · {ft.max_iters} steps")]:
        steps = np.arange(0, cfg.max_iters + 1, max(cfg.max_iters // 800, 1))
        ax.plot(steps, [get_lr(int(t), cfg) for t in steps], color=color, lw=2.4)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("step")
        ax.axhline(pre.min_lr, color=MUTED, lw=1, ls=(0, (3, 3)))
    ax1.set_yscale("log"); ax1.set_ylim(5e-6, 2e-3); ax1.set_ylabel("learning rate (log scale)")
    plt.setp(ax2.get_yticklabels(), visible=False)
    ax2.set_xlim(0, ft.max_iters)
    ax1.text(pre.max_iters * 0.55, pre.min_lr * 0.62, f"pretraining ends at {pre.min_lr:.0e} …",
             fontsize=9, color=MUTED, va="top")
    ax2.text(ft.max_iters * 0.5, ft.lr * 1.35, f"… fine-tuning starts there\nand goes down to {ft.min_lr:.0e}",
             fontsize=9, color=BRASS_D, ha="center", va="bottom")
    suptitle(fig, "Pretrain on a library, then fine-tune on one author",
             "the same weights go through two trainings; the second one is short and gentle so it adds Machado "
             "without erasing Portuguese")
    save(fig, "fig_finetune.png")


# ================================================================== 06b — run 1 vs run 2
def fig_run2():
    import csv as _csv
    def read(path):
        p = os.path.join(ROOT, path)
        if not os.path.exists(p):
            return None
        rows = list(_csv.DictReader(open(p)))
        return [int(r["iter"]) for r in rows], [float(r["train_loss"]) for r in rows], [float(r["val_loss"]) for r in rows]
    r1, pre, ft = read("showcase/run1/log.csv"), read("showcase/run2_pretrain_log.csv"), read("showcase/run2_finetune_log.csv")
    if not (r1 and pre and ft):
        print("  skipped: needs showcase/run1/log.csv, run2_pretrain_log.csv, run2_finetune_log.csv"); return
    fig = plt.figure(figsize=(13, 4.4))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.15, 1.6, 0.55, 1.1], wspace=0.32)
    a, b, c, d = (fig.add_subplot(gs[0, i]) for i in range(4))
    for ax, (it, tr, va), title in [(a, r1, "run 1 · Machado only"), (b, pre, "run 2 · 1 pretrain (library)")]:
        ax.plot(it, tr, color=MUTED, lw=1.8, label="train")
        ax.plot(it, va, color=SAGE if ax is b else BRASS, lw=2.4, label="validation")
        ax.set_title(title, fontsize=11); ax.set_xlabel("step"); ax.set_ylim(1.0, 5.0)
    a.set_ylabel("loss (nats per token)")
    best = min(range(len(r1[2])), key=lambda i: r1[2][i])
    a.scatter([r1[0][best]], [r1[2][best]], color=BRASS, s=30, zorder=5)
    a.annotate(f"best {r1[2][best]:.3f}", (r1[0][best], r1[2][best]), xytext=(4, -16), textcoords="offset points", fontsize=9)
    a.annotate(f"gap {r1[2][-1] - r1[1][-1]:.2f}", (r1[0][-1], (r1[1][-1] + r1[2][-1]) / 2), xytext=(-62, 0),
               textcoords="offset points", fontsize=9, color=BRASS_D, va="center")
    a.legend(loc="upper right", fontsize=9)
    b.text(pre[0][-1], pre[2][-1] + 0.22, "train and validation\nstay together", ha="right", fontsize=9, color=SAGE_D)
    it, tr, va = ft
    c.plot(it, tr, color=MUTED, lw=1.8); c.plot(it, va, color=SAGE_D, lw=2.4)
    c.set_title("2 fine-tune", fontsize=11); c.set_xlabel("step"); c.set_ylim(1.0, 5.0)
    c.set_xticks([0, 400, 800]); plt.setp(c.get_yticklabels(), visible=False)
    # per-character loss on Machado's validation text
    vals = [(3.513 / 2.96, "run 1\nbest", BRASS), (ft[2][0] / 3.1914, "run 2\nbefore\nfine-tune", SAGE_L),
            (min(ft[2]) / 3.1914, "run 2\nfine-tuned", SAGE)]
    xs = range(len(vals))
    d.bar(xs, [v for v, _, _ in vals], color=[col for _, _, col in vals], width=0.62,
          edgecolor=[BRASS_D, SAGE, SAGE_D], linewidth=1)
    for x, (v, _, _) in zip(xs, vals):
        d.text(x, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=9.5)
    d.set_xticks(list(xs), [l for _, l, _ in vals], fontsize=8.8)
    d.set_ylim(0.8, 1.27); d.grid(axis="x", visible=False)
    d.set_title("Machado val, per character", fontsize=11); d.set_ylabel("nats per character")
    suptitle(fig, "Run 1 vs run 2: more data instead of more passes",
             "run 1 sees 1.1 M tokens again and again and overfits; run 2 reads 160 M tokens once, then 800 gentle steps on Machado")
    save(fig, "fig_run2.png")


# ================================================================== 04 — the weights
def fig_weights():
    """Every weight matrix of TinyGPT drawn to scale; area = number of weights."""
    from matplotlib.patches import Rectangle
    model = TinyGPT(ModelConfig(vocab_size=4096))
    shapes = {n: tuple(p.shape) for n, p in model.named_parameters()}
    L = model.config.n_layer

    fig = plt.figure(figsize=(13, 10.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[8.2, 1], hspace=0.06, top=0.95, bottom=0.03, left=0.02, right=0.98)
    ax = fig.add_subplot(gs[0]); blank(ax)

    def rect(x, y, name, label, color, light, dashed=False, inside=True):
        """A weight matrix of shape (out, in), drawn as width = out, height = in."""
        w, h = shapes[name]
        ax.add_patch(Rectangle((x, y), w, h, facecolor="none" if dashed else light, edgecolor=color,
                               lw=1.4, ls=(0, (4, 3)) if dashed else "-"))
        if inside:
            ax.text(x + w / 2, y + h / 2, f"{label}\n{w} × {h}", ha="center", va="center", fontsize=10,
                    color=INK, linespacing=1.5)
        return x + w

    # row 3 (bottom): output layer
    y3 = 0
    rect(0, y3, "tok_emb.weight", "lm_head = tok_emb", BRASS, BRASS_L, dashed=True)
    ax.text(0, y3 + 384 + 50, "output layer (lm_head): the same matrix as the embedding, used in reverse (weight tying)",
            fontsize=10.5, color=MUTED, va="bottom")

    # row 2: one block
    y2 = 384 + 560
    small = 50
    x = 0
    x = rect(x, y2, "blocks.0.attn.wq.weight", "wq", SAGE, SAGE_L) + small
    xk = x
    x = rect(x, y2, "blocks.0.attn.wk.weight", "", SAGE, SAGE_L, inside=False) + small
    x = rect(x, y2, "blocks.0.attn.wv.weight", "", SAGE, SAGE_L, inside=False) + small
    ax.text((xk + x - small) / 2, y2 + 384 + 40, "wk, wv\n128 × 384 each", ha="center", va="bottom",
            fontsize=9.5, color=INK, linespacing=1.4)
    x = rect(x, y2, "blocks.0.attn.wo.weight", "wo", SAGE, SAGE_L)
    att_end = x
    x += 180
    ff0 = x
    x = rect(x, y2, "blocks.0.ffn.w1.weight", "w1 (gate)", BRASS, BRASS_L) + small
    x = rect(x, y2, "blocks.0.ffn.w3.weight", "w3 (up)", BRASS, BRASS_L) + small
    x_end = rect(x, y2, "blocks.0.ffn.w2.weight", "w2\n(down)", BRASS, BRASS_L)
    for a, b, c, cd, title, sub in [
            (0, att_end, SAGE, SAGE_D, "attention", "scores, mask, softmax and RoPE have no weights"),
            (ff0, x_end, BRASS, BRASS_D, "SwiGLU feed-forward", "SiLU and ⊙ have no weights")]:
        ax.plot([a, b], [y2 - 90, y2 - 90], color=c, lw=1.2)
        ax.text((a + b) / 2, y2 - 130, title, ha="center", va="top", fontsize=10.5, color=cd)
        ax.text((a + b) / 2, y2 - 250, sub, ha="center", va="top", fontsize=9, color=MUTED)
    ax.text(0, y2 + 1024 + 90, f"one block, repeated × {L}", fontsize=14, color=INK, va="bottom", family=SERIF)
    ax.text(x_end, y2 + 1024 + 100, "plus 2 RMSNorm gains of 384 numbers", fontsize=9.5, color=MUTED,
            va="bottom", ha="right")

    # row 1 (top): embedding
    y1 = y2 + 1024 + 520
    rect(0, y1, "tok_emb.weight", "tok_emb", BRASS, BRASS_L)
    ax.text(0, y1 + 384 + 50, "token embedding: one row of 384 numbers for each of the 4,096 tokens",
            fontsize=10.5, color=MUTED, va="bottom")

    ax.set_xlim(-30, max(4096, x_end) + 30)
    ax.set_ylim(-40, y1 + 384 + 230)
    ax.set_aspect("equal")

    # bottom: where the numbers live
    bx = fig.add_subplot(gs[1])
    count = lambda pred: sum(p.numel() for n, p in model.named_parameters() if pred(n))
    parts = [("embedding", count(lambda n: n.startswith("tok_emb")), BRASS),
             (f"attention × {L}", count(lambda n: ".attn." in n), SAGE),
             (f"feed-forward × {L}", count(lambda n: ".ffn." in n), BRASS_D),
             ("norms", count(lambda n: "norm" in n), MUTED)]
    total = model.num_params()
    left = 0
    for label, n, c in parts:
        bx.barh(0, n, left=left, color=c, height=0.6, edgecolor="white", lw=1.5)
        if n / total > 0.05:
            bx.text(left + n / 2, 0, f"{label}\n{n / 1e6:.2f} M · {n / total:.0%}", ha="center", va="center",
                    fontsize=9.5, color="white", linespacing=1.3)
        left += n
    bx.text(0, 0.42, f"all {total:,} weights", ha="left", va="bottom", fontsize=10.5, color=INK)
    bx.set_xlim(0, total); bx.set_ylim(-0.4, 0.75)
    bx.set_yticks([]); bx.set_xticks([]); bx.grid(False)
    for sp in bx.spines.values():
        sp.set_visible(False)
    suptitle(fig, "Where the 11 million weights live",
             "each rectangle is one weight matrix of the run-2 model, drawn to scale: its area is its number of weights")
    save(fig, "fig_weights.png")

def fig_network():
    """TinyGPT drawn as a classic neural network, with real numbers from the fine-tuned model."""
    ck, tk = os.path.join(ROOT, "out/finetune/ckpt.pt"), "data/pretrain/tokenizer.json"
    tok = load_tok(tk)
    if not (os.path.exists(ck) and tok):
        print("  skip fig_network (needs out/finetune/ckpt.pt and data/pretrain/tokenizer.json)")
        return
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from attention_maps import attention_weights
    ckpt = torch.load(ck, map_location="cpu")
    model = TinyGPT(ModelConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])
    model.eval()

    prompt = "Capitú olhou para"
    ids = tok.encode(prompt)
    labels = [tok.decode([i]).replace(" ", "␣") for i in ids]
    T = len(ids)
    att = attention_weights(model, torch.tensor([ids]))[0].mean(0).numpy()   # block 1, averaged over heads
    with torch.no_grad():
        logits, _ = model(torch.tensor([ids]))
    probs = F.softmax(logits[0, -1].float(), -1)
    pv, pi = probs.topk(5)
    w1 = model.blocks[0].ffn.w1.weight.detach().numpy()     # (1024, 384)
    w2 = model.blocks[0].ffn.w2.weight.detach().numpy()     # (384, 1024)

    D, Hn = 4, 8                    # neurons drawn per token vector / per hidden layer
    row = lambda t: (T - 1 - t) * 1.25
    xs = dict(tok=0.0, emb=1.9, att=4.6, hid=6.8, out=9.0, nxt=11.8)
    dy, hy = 0.2, 0.13

    fig, ax = plt.subplots(figsize=(13, 8.4))
    blank(ax)
    node = lambda x, y, c, r=0.07: ax.add_patch(plt.Circle((x, y), r, facecolor="white", edgecolor=c, lw=1.3, zorder=5))

    def column(x, t, n, sp):
        y0 = row(t)
        return [y0 + (k - (n - 1) / 2) * sp for k in range(n)]

    for t in range(T):
        # token chip
        ax.text(xs["tok"], row(t), labels[t], ha="center", va="center", family=MONO, fontsize=11, color=INK,
                bbox=dict(boxstyle="round,pad=0.35", facecolor=SAGE_L, edgecolor=SAGE, lw=1))
        e = column(xs["emb"], t, D, dy)
        for y in e:
            ax.plot([xs["tok"] + 0.42, xs["emb"]], [row(t), y], color=MUTED, lw=0.6, alpha=0.5, zorder=1)
        # attention: dimension k of every earlier token feeds dimension k of token t, weighted by attention
        a_out = column(xs["att"], t, D, dy)
        for j in range(t + 1):
            src = column(xs["emb"], j, D, dy)
            for k in range(D):
                ax.plot([xs["emb"], xs["att"]], [src[k], a_out[k]], color=SAGE, lw=0.4 + 3.2 * att[t, j],
                        alpha=min(1, 0.08 + 1.1 * att[t, j]), zorder=2)
        # feed-forward: 4 -> 8 -> 4, edges coloured by the sign of the real w1 / w2 weights
        h = column(xs["hid"], t, Hn, hy)
        o = column(xs["out"], t, D, dy)
        for a in range(D):
            for b in range(Hn):
                v = w1[b * 37, a * 61]
                ax.plot([xs["att"], xs["hid"]], [a_out[a], h[b]], color=BRASS if v > 0 else MUTED,
                        lw=0.3 + 18 * abs(v), alpha=0.55, zorder=2)
                v2 = w2[a * 61, b * 37]
                ax.plot([xs["hid"], xs["out"]], [h[b], o[a]], color=BRASS if v2 > 0 else MUTED,
                        lw=0.3 + 18 * abs(v2), alpha=0.55, zorder=2)
        for y in e: node(xs["emb"], y, BRASS)
        for y in a_out: node(xs["att"], y, SAGE)
        for y in h: node(xs["hid"], y, BRASS_D, 0.05)
        for y in o: node(xs["out"], y, BRASS)

    # output: only the last position predicts the next token
    last = column(xs["out"], T - 1, D, dy)
    ny = [row(T - 1) + 2.4 - k * 0.62 for k in range(5)]
    for k in range(5):
        p = pv[k].item()
        for y in last:
            ax.plot([xs["out"], xs["nxt"]], [y, ny[k]], color=SAGE_D, lw=0.4 + 6 * p, alpha=0.25 + 2.2 * p, zorder=2)
        ax.add_patch(plt.Circle((xs["nxt"], ny[k]), 0.08 + 0.35 * p, facecolor=SAGE_L, edgecolor=SAGE_D, lw=1.3, zorder=5))
        ax.text(xs["nxt"] + 0.55, ny[k], f"{tok.decode([pi[k].item()]).replace(' ', '␣')}   {p:.2f}",
                va="center", family=MONO, fontsize=10.5, color=INK)

    top = row(0) + 0.95
    heads = [("tok", "tokens"), ("emb", "embedding\n4 of 384 numbers"), ("att", "after attention\n(mixes tokens)"),
             ("hid", "feed-forward\n8 of 1,024 hidden"), ("out", "block output"), ("nxt", "next token")]
    for key, txt in heads:
        ax.text(xs[key], top, txt, ha="center", va="bottom", fontsize=10, color=INK, linespacing=1.35)
    ax.add_patch(FancyBboxPatch((xs["emb"] + 0.35, row(T - 1) - 0.75), xs["out"] - xs["emb"] - 0.1,
                                row(0) - row(T - 1) + 1.45, boxstyle="round,pad=0,rounding_size=0.25",
                                facecolor="none", edgecolor=LINE, lw=1.1, ls=(0, (4, 3)), zorder=0))
    ax.text((xs["emb"] + xs["out"]) / 2 + 0.15, row(T - 1) - 0.95, "one block: in the model, 6 of them in a row",
            ha="center", va="top", fontsize=10, color=MUTED)
    ax.text(xs["nxt"] - 0.3, row(T - 1) - 0.95, "only the last position\npredicts the next token",
            ha="center", va="top", fontsize=9.5, color=MUTED, linespacing=1.35)
    ax.set_xlim(-0.8, 14.4)
    ax.set_ylim(row(T - 1) - 1.6, top + 0.8)
    ax.set_aspect("equal")
    fig.subplots_adjust(top=0.99, bottom=0.01, left=0.01, right=0.99)
    suptitle(fig, f"TinyGPT as a neural network: \"{prompt}\"",
             "green lines: the fine-tuned model's real attention weights (block 1, heads averaged) · "
             "brass/grey lines: real feed-forward weights, positive/negative")
    save(fig, "fig_network.png")


def fig_attention_flow():
    """CausalSelfAttention.forward step by step: every tensor with its shape, filled with real values."""
    from matplotlib.patches import Rectangle, Circle
    import math as _m
    ck, tk = os.path.join(ROOT, "out/finetune/ckpt.pt"), "data/pretrain/tokenizer.json"
    tok = load_tok(tk)
    if not (os.path.exists(ck) and tok):
        print("  skip fig_attention_flow (needs out/finetune/ckpt.pt and data/pretrain/tokenizer.json)")
        return
    ckpt = torch.load(ck, map_location="cpu")
    model = TinyGPT(ModelConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])
    model.eval()
    prompt = "Capitú olhou para"
    ids = tok.encode(prompt)
    T = len(ids)
    labels = [tok.decode([i]).replace(" ", "␣") for i in ids]

    # real tensors from the last block, sequence 0, head 1
    with torch.no_grad():
        idx = torch.tensor([ids])
        cos, sin = model.rope_cos[:T], model.rope_sin[:T]
        h = model.tok_emb(idx)
        for blk in model.blocks[:-1]:
            h = blk(h, cos, sin)
        blk = model.blocks[-1]
        a = blk.attn
        xin = blk.norm1(h)
        q_lin, k_lin, v_lin = a.wq(xin), a.wk(xin), a.wv(xin)
        q = q_lin.view(1, T, a.n_head, a.head_dim).transpose(1, 2)
        k = k_lin.view(1, T, a.n_kv_head, a.head_dim).transpose(1, 2)
        v = v_lin.view(1, T, a.n_kv_head, a.head_dim).transpose(1, 2)
        q_r, k_r = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        from model import repeat_kv
        k_rep, v_rep = repeat_kv(k_r, a.n_rep), repeat_kv(v, a.n_rep)
        scores = (q_r @ k_rep.transpose(-2, -1))[0, 0] / _m.sqrt(a.head_dim)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
        w = F.softmax(scores.masked_fill(mask, float("-inf")), -1)
        y = (w @ v_rep[0, 0])
        y_all = F.softmax((q_r @ k_rep.transpose(-2, -1)).masked_fill(mask, float("-inf")) / 1, -1)
        merged = (F.softmax(((q_r @ k_rep.transpose(-2, -1)) / _m.sqrt(a.head_dim)).masked_fill(mask, float("-inf")), -1)
                  @ v_rep).transpose(1, 2).reshape(1, T, -1)
        out = a.wo(merged)

    DIV = LinearSegmentedColormap.from_list("div", [BRASS, "#FFFFFF", SAGE_D])
    fig, ax = plt.subplots(figsize=(15, 11))
    blank(ax)
    S = 1.0   # cell size

    def grid(x, y, M, title, shape, cmap=DIV, stack=0, rows=None, masked=None, signed=True, show_rows=False):
        """Draw matrix M (rows x cols) with its top-left at (x, y); y grows downward in rows."""
        M = np.asarray(M, dtype=float)
        nr, nc = M.shape
        for s in range(stack, 0, -1):
            ax.add_patch(Rectangle((x + 0.45 * s, y + 0.45 * s - nr), nc, nr, facecolor="white",
                                   edgecolor=LINE, lw=0.9, zorder=1))
        m = np.abs(M[np.isfinite(M)]).max() or 1
        for r in range(nr):
            for c in range(nc):
                val = M[r, c]
                if masked is not None and masked[r, c]:
                    fc = "#ECE7DE"
                else:
                    fc = cmap(0.5 + 0.5 * val / m) if signed else cmap(val / m)
                ax.add_patch(Rectangle((x + c, y - r - 1), 1, 1, facecolor=fc, edgecolor="white", lw=0.6, zorder=2))
        ax.add_patch(Rectangle((x, y - nr), nc, nr, facecolor="none", edgecolor=MUTED, lw=0.8, zorder=3))
        ax.text(x + nc / 2, y + 0.7 + (0.45 * stack), title, ha="center", va="bottom", fontsize=11.5,
                fontweight="medium", color=INK, family=MONO)
        ax.text(x + nc / 2, y - nr - 0.6, shape, ha="center", va="top", fontsize=9.5, color=MUTED, family=MONO)
        if show_rows:
            for r, lab in enumerate(labels):
                ax.text(x - 0.4, y - r - 0.5, lab, ha="right", va="center", fontsize=8.5, family=MONO, color=MUTED)
        return x + nc, y - nr / 2

    def box(xc, yc, text, sub=None, color=SAGE):
        ax.text(xc, yc, text, ha="center", va="center", fontsize=10.5, family=MONO, color=INK, zorder=6,
                bbox=dict(boxstyle="round,pad=0.45", facecolor=SAGE_L if color == SAGE else BRASS_L,
                          edgecolor=color, lw=1.2))
        if sub:
            ax.text(xc, yc - 1.25, sub, ha="center", va="top", fontsize=8.8, color=MUTED, linespacing=1.3)

    def arrow(x0, y0, x1, y1, label=None, color=MUTED, dy=0.45, ls="-"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=4,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3, linestyle=ls, shrinkA=2, shrinkB=2))
        if label:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, label, ha="center", va="bottom", fontsize=9, color=color)

    def at(xc, yc):
        ax.add_patch(Circle((xc, yc), 0.95, facecolor="white", edgecolor=SAGE_D, lw=1.4, zorder=5))
        ax.text(xc, yc, "@", ha="center", va="center", fontsize=12, color=SAGE_D, family=MONO, zorder=6)

    C_SHOW, KV_SHOW, D_SHOW = 12, 6, 6       # columns drawn for 384, 128 and 64
    # ---------------- row 1: project and split into heads
    top = 30
    ax.text(-5, top + 4.2, "1 · project each token, then split into heads", fontsize=13, color=INK, family=SERIF,
            fontweight="semibold")
    lanes = {"q": top, "k": top - 13, "v": top - 26}
    xin_np = xin[0].numpy()
    x_end, x_mid = grid(0, lanes["k"], xin_np[:, :C_SHOW], "x", "(B, T, 384)", show_rows=True)
    ax.text(C_SHOW / 2, lanes["k"] - T - 1.8, "RMSNorm of the residual stream", ha="center", va="top",
            fontsize=8.8, color=MUTED)
    spec = {"q": (q_lin, a.n_head, "wq", "384 → 384", C_SHOW, "q", q_r),
            "k": (k_lin, a.n_kv_head, "wk", "384 → 128", KV_SHOW, "k", k_r),
            "v": (v_lin, a.n_kv_head, "wv", "384 → 128", KV_SHOW, "v", v)}
    bx = x_end + 4.5
    for name, (lin, nh, wname, dims, ncols, short, heads) in spec.items():
        ly = lanes[name] - T / 2
        ax.plot([x_end + 0.3, x_end + 1.5, x_end + 1.5], [x_mid, x_mid, ly], color=MUTED, lw=1.1, zorder=1)
        arrow(x_end + 1.5, ly, bx - 2.0, ly)
        box(bx, ly, wname, f"nn.Linear\n{dims}")
        xp = bx + 3.6
        arrow(bx + 1.8, ly, xp - 0.2, ly)
        e, _ = grid(xp, lanes[name], lin[0].numpy()[:, :ncols], short, f"(B, T, {lin.shape[-1]})")
        xh = xp + ncols + 7.5
        arrow(e + 0.2, ly, xh - 0.3, ly, ".view .transpose")
        e2, _ = grid(xh, lanes[name], heads[0, 0].numpy()[:, :D_SHOW], short, f"(B, {nh}, T, 64)", stack=min(nh, 5) - 1 if nh > 2 else 1)
        if name in "qk":
            ax.text(xh + D_SHOW / 2, lanes[name] - T - 2.3, "RoPE applied", ha="center",
                    va="top", fontsize=8.8, color=SAGE_D)
        if name in "kv":
            xr = e2 + 8.5
            arrow(e2 + 1.4, ly, xr - 0.3, ly, "repeat_kv (GQA)")
            rep = k_rep if name == "k" else v_rep
            grid(xr, lanes[name], rep[0, 0].numpy()[:, :D_SHOW], short, "(B, 6, T, 64)", stack=4)
            ax.text(xr + D_SHOW + 2.6, ly, "each K/V head\nshared by 3\nquery heads", va="center", fontsize=8.8,
                    color=MUTED, linespacing=1.3)
    # ---------------- row 2: attend and merge (head 1 shown)
    r2 = top - 26 - T - 9
    ax.text(-5, r2 + 4.2, "2 · attend (one head shown), then merge the heads", fontsize=13, color=INK, family=SERIF,
            fontweight="semibold")
    gx = 0
    e, _ = grid(gx, r2, q_r[0, 0].numpy()[:, :D_SHOW], "q", "(B, 6, T, 64)", stack=4, show_rows=True)
    kt = k_rep[0, 0].numpy()[:, :D_SHOW].T
    e_k, _ = grid(gx, r2 - 14, kt, "kᵀ", "(B, 6, 64, T)", stack=4)
    ax.text(gx + T + 3, r2 - 14 - D_SHOW - 1.3, ".transpose(-2, -1)", fontsize=9, color=MUTED, va="center")
    ax_ = e + 6.5
    at(ax_, r2 - T / 2)
    arrow(e + 1.6, r2 - T / 2, ax_ - 1.0, r2 - T / 2)
    ax.plot([e_k + 2.2, ax_, ax_], [r2 - 14 - 3, r2 - 14 - 3, r2 - T / 2 - 1.0], color=MUTED, lw=1.3, zorder=1)
    ax.annotate("", xy=(ax_, r2 - T / 2 - 1.0), xytext=(ax_, r2 - T / 2 - 1.6),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.3))
    sx = ax_ + 3
    arrow(ax_ + 1.0, r2 - T / 2, sx - 0.2, r2 - T / 2)
    e, _ = grid(sx, r2, (scores * _m.sqrt(a.head_dim)).numpy(), "q @ kᵀ", "(B, 6, T, T)", show_rows=False)
    ax.text(sx + T / 2, r2 - T - 1.8, "one score per pair of tokens", ha="center", va="top", fontsize=8.8, color=MUTED)
    mx = e + 7
    arrow(e + 0.3, r2 - T / 2, mx - 0.3, r2 - T / 2, "÷ √64, mask")
    sm = scores.masked_fill(mask, 0).numpy()
    e, _ = grid(mx, r2, sm, "scores", "(B, 6, T, T)", masked=mask.numpy())
    ax.text(mx + T / 2, r2 - T - 1.8, "future = −∞ (grey)", ha="center", va="top", fontsize=8.8, color=MUTED)
    wx = e + 7
    arrow(e + 0.3, r2 - T / 2, wx - 0.3, r2 - T / 2, "softmax")
    e, _ = grid(wx, r2, w.numpy(), "weights", "(B, 6, T, T)", cmap=CMAP, signed=False, masked=mask.numpy())
    ax.text(wx + T / 2, r2 - T - 1.8, "each row sums to 1", ha="center", va="top", fontsize=8.8, color=MUTED)
    at2 = e + 5
    at(at2, r2 - T / 2)
    arrow(e + 0.3, r2 - T / 2, at2 - 1.0, r2 - T / 2)
    vx = at2 - D_SHOW - 4
    grid(vx, r2 - 14, v_rep[0, 0].numpy()[:, :D_SHOW], "v", "(B, 6, T, 64)", stack=4)
    ax.plot([vx + D_SHOW + 2.2, at2, at2], [r2 - 14 - 3.5, r2 - 14 - 3.5, r2 - T / 2 - 1.6], color=MUTED, lw=1.3, zorder=1)
    ax.annotate("", xy=(at2, r2 - T / 2 - 1.0), xytext=(at2, r2 - T / 2 - 1.6),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.3))
    yx = at2 + 3.5
    arrow(at2 + 1.0, r2 - T / 2, yx - 0.2, r2 - T / 2)
    e, _ = grid(yx, r2, y.numpy()[:, :D_SHOW], "y", "(B, 6, T, 64)", stack=4)
    ax.text(yx + D_SHOW / 2, r2 - T - 1.8, "a weighted mix\nof the values", ha="center", va="top", fontsize=8.8,
            color=MUTED, linespacing=1.3)
    mgx = e + 9
    arrow(e + 1.9, r2 - T / 2, mgx - 0.3, r2 - T / 2, ".transpose .view")
    e, _ = grid(mgx, r2, merged[0].numpy()[:, :C_SHOW], "merged", "(B, T, 384)")
    ax.text(mgx + C_SHOW / 2, r2 - T - 1.8, "6 heads × 64, side by side", ha="center", va="top", fontsize=8.8,
            color=MUTED)
    wox = e + 4
    arrow(e + 0.3, r2 - T / 2, wox - 1.6, r2 - T / 2)
    box(wox, r2 - T / 2, "wo", "nn.Linear\n384 → 384")
    ox = wox + 3.4
    arrow(wox + 1.6, r2 - T / 2, ox - 0.2, r2 - T / 2)
    e, _ = grid(ox, r2, out[0].numpy()[:, :C_SHOW], "out", "(B, T, 384)")
    ax.text(ox + C_SHOW / 2, r2 - T - 1.8, "added to the\nresidual stream", ha="center", va="top", fontsize=8.8,
            color=MUTED, linespacing=1.3)

    ax.set_xlim(-6, e + 1.5)
    ax.set_ylim(r2 - 25, top + 6)
    ax.set_aspect("equal")
    fig.subplots_adjust(top=0.99, bottom=0.01, left=0.01, right=0.99)
    suptitle(fig, "Inside CausalSelfAttention.forward",
             f"real values from the fine-tuned model, last block, on \"{prompt}\" · B = batch, T = {T} tokens · "
             "12 of 384 and 6 of 64 columns drawn")
    save(fig, "fig_attention_flow.png")


FIGURES = {
    "books": fig_machado_books, "bpe": fig_bpe_merges, "cuts": fig_tokenizer_cuts,
    "mask": fig_attention_mask, "sqrtd": fig_sqrt_d, "rope": fig_rope, "act": fig_activations,
    "kv": fig_kv_cache, "ce": fig_cross_entropy, "temp": fig_temperature, "batch": fig_batch_shift,
    "lr": fig_lr_schedule, "corpus": fig_corpus, "finetune": fig_finetune, "run2": fig_run2, "weights": fig_weights, "network": fig_network, "flow": fig_attention_flow,
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=list(FIGURES))
    args = ap.parse_args()
    for name in args.only or FIGURES:
        print(name)
        FIGURES[name]()
