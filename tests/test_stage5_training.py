import numpy as np
import pytest
import torch

from config import ModelConfig, TrainConfig
from model import TinyGPT
from train import get_batch, get_lr, train_step


def test_5_1_get_batch_targets_are_shifted_inputs():
    data = np.arange(1000, dtype=np.uint16)
    x, y = get_batch(data, block_size=16, batch_size=8, device="cpu")
    assert x.shape == y.shape == (8, 16)
    assert x.dtype == torch.int64
    assert torch.equal(y[:, :-1], x[:, 1:])
    assert torch.equal(y[:, -1], x[:, -1] + 1)


def test_5_2_lr_schedule():
    cfg = TrainConfig(lr=1e-3, min_lr=1e-4, warmup_iters=100, max_iters=1100)
    assert get_lr(0, cfg) == pytest.approx(1e-5)          # starts tiny
    assert get_lr(99, cfg) == pytest.approx(1e-3)         # peak at end of warmup
    assert get_lr(600, cfg) == pytest.approx(5.5e-4)      # halfway down the cosine
    assert get_lr(1100, cfg) == pytest.approx(1e-4)
    assert get_lr(5000, cfg) == pytest.approx(1e-4)
    lrs = [get_lr(i, cfg) for i in range(100, 1100)]
    assert all(a >= b for a, b in zip(lrs, lrs[1:])), "must decrease after warmup"


def test_5_3_train_step_reduces_loss():
    torch.manual_seed(0)
    mcfg = ModelConfig(vocab_size=50, block_size=16, n_layer=1, n_head=2, n_kv_head=1,
                       d_model=32, dropout=0.0)
    tcfg = TrainConfig(batch_size=8, grad_accum=2)
    data = np.tile(np.arange(50, dtype=np.uint16), 40)    # a perfectly predictable sequence
    model = TinyGPT(mcfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = train_step(model, opt, data, mcfg, tcfg, "cpu")
    for _ in range(60):
        last = train_step(model, opt, data, mcfg, tcfg, "cpu")
    assert isinstance(first, float)
    assert last < first * 0.3, f"loss went {first:.2f} -> {last:.2f}"
    assert all(p.grad is None for p in model.parameters()), "zero the grads after stepping"
