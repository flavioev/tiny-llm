import math

import torch

from config import ModelConfig
from model import Block, TinyGPT, precompute_rope

CFG = ModelConfig(vocab_size=97, block_size=32, n_layer=2, n_head=4, n_kv_head=2,
                  d_model=64, dropout=0.0)


def test_4_1_block_is_residual():
    blk = Block(CFG)
    cos, sin = precompute_rope(CFG.head_dim, 32)
    x = torch.randn(2, 32, 64)
    assert blk(x, cos, sin).shape == x.shape
    with torch.no_grad():                                  # zero both sublayers' outputs...
        blk.attn.wo.weight.zero_()
        blk.ffn.w2.weight.zero_()
    assert torch.allclose(blk(x, cos, sin), x), "...then the block must be the identity"


def test_4_2_forward_shapes_and_initial_loss():
    torch.manual_seed(0)
    model = TinyGPT(CFG)
    idx = torch.randint(0, 97, (4, 32))
    targets = torch.randint(0, 97, (4, 32))
    logits, loss = model(idx, targets)
    assert logits.shape == (4, 32, 97)
    # A fresh model should be ~uniformly unsure: loss ≈ ln(vocab)
    assert abs(loss.item() - math.log(97)) < 0.5, f"initial loss {loss.item():.2f}, expected ~{math.log(97):.2f}"
    _, none = model(idx)
    assert none is None


def test_4_2_model_is_causal_end_to_end():
    torch.manual_seed(0)
    model = TinyGPT(CFG).eval()
    a = torch.randint(0, 97, (1, 20))
    b = a.clone()
    b[0, 12:] = (b[0, 12:] + 1) % 97
    la, _ = model(a)
    lb, _ = model(b)
    assert torch.allclose(la[0, :12], lb[0, :12], atol=1e-5)


def test_4_2_can_memorize_one_batch():
    """The classic sanity check: a model that can't overfit one batch has a bug."""
    torch.manual_seed(0)
    model = TinyGPT(CFG)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    x = torch.randint(0, 97, (4, 33))
    for _ in range(150):
        _, loss = model(x[:, :-1], x[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.3, f"loss stuck at {loss.item():.2f}"


def test_4_3_generate():
    torch.manual_seed(0)
    model = TinyGPT(CFG).eval()
    out = model.generate(torch.zeros(2, 3, dtype=torch.long), max_new_tokens=40, top_k=5)
    assert out.shape == (2, 43)                            # longer than block_size: cropping works
    assert out.max() < 97 and out.min() >= 0
    assert torch.equal(out[:, :3], torch.zeros(2, 3, dtype=torch.long))
