import torch

from config import ModelConfig
from model import CausalSelfAttention, RMSNorm, SwiGLU, apply_rope, precompute_rope, repeat_kv


def test_3_1_rmsnorm():
    torch.manual_seed(0)
    norm = RMSNorm(32)
    x = torch.randn(4, 7, 32) * 5 + 3
    y = norm(x)
    ref = x / torch.sqrt((x ** 2).mean(-1, keepdim=True) + 1e-6)
    assert torch.allclose(y, ref, atol=1e-5)
    rms = y.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3), "output should have RMS 1"


def test_3_2_rope_tables():
    cos, sin = precompute_rope(head_dim=8, max_len=5)
    assert cos.shape == sin.shape == (5, 4)
    assert torch.allclose(cos[0], torch.ones(4)) and torch.allclose(sin[0], torch.zeros(4)), \
        "position 0 must be no rotation"
    assert torch.isclose(sin[1, 0], torch.sin(torch.tensor(1.0))), "pair 0 rotates 1 rad per position"


def test_3_3_rope_preserves_norm():
    cos, sin = precompute_rope(16, 10)
    x = torch.randn(2, 3, 10, 16)
    assert torch.allclose(apply_rope(x, cos, sin).norm(dim=-1), x.norm(dim=-1), atol=1e-5), \
        "a rotation must not change vector length"


def test_3_3_rope_depends_only_on_relative_position():
    """The key property: <rope(q, m), rope(k, n)> depends only on m - n."""
    torch.manual_seed(0)
    cos, sin = precompute_rope(16, 64)
    q, k = torch.randn(16), torch.randn(16)

    def score(m, n):
        qm = apply_rope(q.view(1, 1, 1, 16), cos[m:m + 1], sin[m:m + 1])
        kn = apply_rope(k.view(1, 1, 1, 16), cos[n:n + 1], sin[n:n + 1])
        return (qm * kn).sum()

    assert torch.isclose(score(5, 2), score(40, 37), atol=1e-4)
    assert torch.isclose(score(10, 10), score(0, 0), atol=1e-4)
    assert not torch.isclose(score(5, 2), score(5, 4), atol=1e-4)


def test_3_4_repeat_kv():
    x = torch.arange(2).float().view(1, 2, 1, 1)          # kv heads [A=0, B=1]
    y = repeat_kv(x, 3)
    assert y.shape == (1, 6, 1, 1)
    assert y.flatten().tolist() == [0, 0, 0, 1, 1, 1]
    assert repeat_kv(x, 1) is x


def test_3_5_swiglu():
    cfg = ModelConfig(d_model=48, dropout=0.0)
    ffn = SwiGLU(cfg)
    x = torch.randn(2, 5, 48)
    assert ffn(x).shape == x.shape
    with torch.no_grad():
        ffn.w3.weight.zero_()                              # close the gate...
    assert torch.allclose(ffn(x), torch.zeros_like(x)), "...and nothing should pass"


def test_3_6_multihead_module_shapes_and_flash_equivalence():
    torch.manual_seed(0)
    cfg = ModelConfig(d_model=64, n_head=4, n_kv_head=2, block_size=16, dropout=0.0)
    attn = CausalSelfAttention(cfg).eval()
    cos, sin = precompute_rope(cfg.head_dim, 16)
    x = torch.randn(2, 16, 64)
    y = attn(x, cos, sin)
    assert y.shape == x.shape
    attn.use_flash = True
    assert torch.allclose(y, attn(x, cos, sin), atol=1e-5), "manual path != fused path"
