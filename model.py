"""
STAGES 2-4 — The Transformer, decoder-only, with the 2023+ upgrades.

"Attention Is All You Need" (Vaswani et al., 2017) introduced the Transformer.
GPT kept only its decoder. Modern open models (Llama, Mistral, Qwen...) keep the
same skeleton but swap several parts. This file has both the core and the swaps:

    2017 original                  ->  what is built here (modern)
    ---------------------------------------------------------------------
    sinusoidal / learned positions -> RoPE (rotary embeddings, applied to q and k)
    LayerNorm, after each sublayer -> RMSNorm, BEFORE each sublayer (pre-norm)
    ReLU feed-forward              -> SwiGLU feed-forward (gated)
    every head has its own K, V    -> Grouped-Query Attention (heads share K, V)
    separate output matrix         -> weight tying (output = embedding matrix)

Shapes used in the comments: B = batch, T = tokens, H = heads, D = dims per head.

Explained step by step in docs/02-attention.md, docs/03-modern-upgrades.md and
docs/04-assembling.md.
Tests: tests/test_stage2_attention.py, test_stage3_modern.py, test_stage4_model.py
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


# =========================================================================== STAGE 2
# The heart of the paper: Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
# plus a causal mask, so position t can only look at positions <= t.

def causal_attention(q, k, v, dropout_p: float = 0.0, training: bool = False):
    """q, k, v: (B, H, T, D) -> (B, H, T, D)

    Each token compares its query with every key; the better the match, the more
    of that token's value it takes. The output is a weighted average of values.
    """
    # Scores: row i, column j = how well token i's query matches token j's key.
    # Divided by sqrt(D): q·k is a sum of D products, so its spread grows with D;
    # without the scaling, softmax becomes nearly one-hot and gradients vanish.
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))   # (B, H, T, T)

    # Causal mask: True strictly above the diagonal, i.e. where j > i (the future).
    # Setting those scores to -inf makes their softmax weight exactly 0.
    mask = torch.triu(torch.ones(scores.size(-2), scores.size(-1), device=scores.device), diagonal=1).bool()
    scores = scores.masked_fill(mask, float('-inf'))

    # Softmax in float32 (exponentials lose precision first in bf16), then back.
    weights = F.softmax(scores.float(), dim=-1).to(q.dtype)                 # each row sums to 1
    if training and dropout_p > 0:
        weights = F.dropout(weights, p=dropout_p)
    return torch.matmul(weights, v)                                          # (B, H, T, D)


def repeat_kv(x, n_rep: int):
    """Grouped-Query Attention helper. (B, H_kv, T, D) -> (B, H_kv * n_rep, T, D)

    Each K/V head is shared by n_rep consecutive query heads:
    kv heads [A, B], n_rep=3  ->  [A, A, A, B, B, B]
    """
    # Fewer K/V heads means a smaller KV cache at generation time (3x smaller here),
    # with almost no loss in quality. repeat_interleave keeps the copies adjacent;
    # torch.repeat would give [A, B, A, B, A, B] and pair the wrong heads.
    if n_rep == 1:
        return x  # classic multi-head attention: nothing to share
    else:
        return torch.repeat_interleave(x, n_rep, dim=1)


# =========================================================================== STAGE 3
class RMSNorm(nn.Module):
    """y = x / sqrt(mean(x^2) + eps) * weight

    LayerNorm subtracts the mean and divides by the std; RMSNorm skips the mean.
    Cheaper, and works just as well — Llama, Mistral, Gemma all use it.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # learned gain, one per dimension

    def forward(self, x):
        # Rescale each token's vector to root-mean-square 1, then apply the gain.
        # The mean of squares is computed in float32 (squares overflow first in low
        # precision); rsqrt(t) = 1 / sqrt(t).
        return x * torch.rsqrt(
            x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True) + self.eps
        ).to(x.dtype) * self.weight


def precompute_rope(head_dim: int, max_len: int, theta: float = 10000.0):
    """Angles for Rotary Position Embeddings. Returns cos, sin, each (max_len, head_dim // 2).

    Dimension pair i rotates at frequency  inv_freq[i] = 1 / theta^(2i / head_dim);
    at position p the rotation angle is  p * inv_freq[i].
    Low i -> fast rotation (fine position), high i -> slow (coarse position).
    """
    # Computed once when the model is created and stored as a buffer.
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))   # (D/2,): one speed per pair
    positions = torch.arange(max_len).float()                                         # (T,)
    angles = torch.outer(positions, inv_freq)                                         # (T, D/2): p * speed
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """Rotate q or k. x: (B, H, T, D); cos, sin: (T, D/2).

    Split the head dim in two halves x1, x2 and rotate each (x1[i], x2[i]) pair:
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
    Because q and k are both rotated by their own position, q·k ends up depending
    only on the distance between the two tokens, not on where they are.
    """
    # The 2D rotation formula applied to all D/2 pairs at once. Pairs are
    # (first half[i], second half[i]) — the GPT-NeoX / Hugging Face Llama layout.
    # cos and sin broadcast over B and H.
    x1, x2 = x.chunk(2, dim=-1)
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    # cos/sin are float32: without .to(x.dtype) a bf16 input would come back as float32.
    return torch.cat([out1, out2], dim=-1).to(x.dtype)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE and Grouped-Query Attention.

    Several heads run in parallel, each with its own projections, so each can learn
    to look for something different. In the trained models some heads look at the
    previous word, some at the verb, some rest on the first token ("attention sink"):
    see the heatmaps in docs/02-attention.md (scripts/attention_maps.py).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        assert cfg.n_head % cfg.n_kv_head == 0
        self.n_head, self.n_kv_head, self.head_dim = cfg.n_head, cfg.n_kv_head, cfg.head_dim
        self.n_rep = cfg.n_head // cfg.n_kv_head  # query heads per K/V head
        self.wq = nn.Linear(cfg.d_model, cfg.n_head * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_head * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_head * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_head * self.head_dim, cfg.d_model, bias=False)  # mixes the heads back
        self.dropout = cfg.dropout
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.use_flash = cfg.use_flash

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        # Project, then split each token's vector into heads: view() cuts the last
        # dimension into (heads, D) without moving data; transpose puts heads before
        # tokens so each head is an independent (T, D) matrix.
        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)     # (B, H, T, D)
        k = self.wk(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)  # (B, H_kv, T, D)
        v = self.wv(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)  # (B, H_kv, T, D)

        # Position goes into q and k only: it should change WHO attends to whom,
        # not WHAT information is passed along (v).
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        k, v = repeat_kv(k, self.n_rep), repeat_kv(v, self.n_rep)                 # (B, H, T, D)
        if self.use_flash:
            # Same math through PyTorch's fused kernel (faster on GPU); a test checks
            # it matches causal_attention.
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                               dropout_p=self.dropout if self.training else 0.0)
        else:
            y = causal_attention(q, k, v, self.dropout, self.training)

        # Merge heads back: transpose only changes indexing, so contiguous() lays the
        # data out in order before view() glues the H heads into one vector per token.
        y = y.transpose(1, 2).contiguous().view(B, T, -1)                         # (B, T, d_model)
        return self.resid_dropout(self.wo(y))


class SwiGLU(nn.Module):
    """Gated feed-forward: w2( silu(w1 x) * w3 x ).

    The 2017 FFN was w2(relu(w1 x)). The gate lets the network decide, per
    dimension, how much of the content to let through. The hidden size is
    ~8/3 * d_model so the 3 matrices cost about the same as the old 2 with 4*d.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = int(8 * cfg.d_model / 3)
        hidden = 64 * ((hidden + 63) // 64)  # round up to a multiple of 64 (hardware-friendly)
        self.w1 = nn.Linear(cfg.d_model, hidden, bias=False)  # gate
        self.w3 = nn.Linear(cfg.d_model, hidden, bias=False)  # up: the content
        self.w2 = nn.Linear(hidden, cfg.d_model, bias=False)  # down: back to d_model
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        # Attention moves information BETWEEN tokens; this processes each token on
        # its own. silu(w1 x) is the gate, w3 x the content, * mixes them element by
        # element, w2 projects back so the result fits the residual stream.
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# =========================================================================== STAGE 4
class Block(nn.Module):
    """One transformer layer, pre-norm:
        x = x + attention(norm(x))
        x = x + feedforward(norm(x))
    The `x + ...` is the residual stream: each layer *adds* information to it.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        # Each sublayer reads a normalized copy of x and ADDS its result back.
        # x itself is never normalized, so the path from input to output stays an
        # uninterrupted sum: gradients reach the first layer directly, and a sublayer
        # that outputs zeros leaves x unchanged.
        x = x + self.attn(self.norm1(x), cos, sin)   # gather context from earlier tokens
        x = x + self.ffn(self.norm2(x))              # process each token on its own
        return x


class TinyGPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.config = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying: one matrix reads AND writes tokens

        # RoPE tables, computed once. persistent=False: not saved in checkpoints,
        # they are rebuilt from the config.
        cos, sin = precompute_rope(cfg.head_dim, cfg.block_size, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # GPT-2 trick: shrink the init of layers that write into the residual stream,
        # so the stream's variance doesn't grow with depth.
        for name, p in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())  # tied weights counted once

    def forward(self, idx, targets=None):
        """idx: (B, T) token ids. Returns (logits (B, T, vocab), loss or None)."""
        B, T = idx.shape
        assert T <= self.config.block_size, f"sequence {T} > block_size {self.config.block_size}"
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]

        # No positional embedding is added here: position enters inside attention (RoPE).
        x = self.tok_emb(idx)           # (B, T, d_model): one vector per token id
        x = self.drop(x)
        for block in self.blocks:       # each block reads and adds to the residual stream
            x = block(x, cos, sin)
        x = self.norm_f(x)
        logits = self.lm_head(x)        # (B, T, vocab): one score per possible next token

        loss = None
        if targets is not None:
            # Flatten (B, T, V) -> (B*T, V): every position of every sequence is a
            # separate next-token prediction, so one forward pass trains B*T of them.
            # A fresh model scores ~ln(vocab) here, a uniform guess: ln(2,048) = 7.62 in
            # run 1, ln(4,096) = 8.32 in run 2 (both runs started within 0.03 of it).
            loss = F.cross_entropy(logits.reshape(-1, self.config.vocab_size), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 0.8,
                 top_k: int = 50, eot_id: int = None):
        """Autoregressive sampling: predict one token, append it, repeat."""
        for _ in range(max_new_tokens):
            # The model sees at most block_size tokens: keep the most recent ones.
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            # Only the last position's scores matter: they predict the next token.
            # temperature < 1 sharpens the distribution (safer), > 1 flattens it (wilder).
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                # Keep the k best scores; everything below the k-th gets -inf,
                # i.e. probability 0. Cuts off the long tail of unlikely tokens.
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)  # draw one token, following the probabilities
            idx = torch.cat([idx, nxt], dim=1)
            if eot_id is not None and bool((idx[:, -1] == eot_id).all()):
                break
        return idx

    def configure_optimizer(self, weight_decay, lr, betas, device_type):
        """AdamW. Weight decay only on matrices (2D+), never on norms' gains."""
        params = [p for p in self.parameters() if p.requires_grad]
        decay = [p for p in params if p.dim() >= 2]
        no_decay = [p for p in params if p.dim() < 2]
        groups = [{"params": decay, "weight_decay": weight_decay},
                  {"params": no_decay, "weight_decay": 0.0}]
        extra = {"fused": True} if device_type == "cuda" else {}
        return torch.optim.AdamW(groups, lr=lr, betas=betas, **extra)
