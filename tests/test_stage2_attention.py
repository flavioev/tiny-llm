import torch
import torch.nn.functional as F

from model import causal_attention


def test_2_1_matches_pytorch_reference():
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 3, 10, 16) for _ in range(3))
    ours = causal_attention(q, k, v)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert ours.shape == (2, 3, 10, 16)
    assert torch.allclose(ours, ref, atol=1e-5), "your attention differs from PyTorch's"


def test_2_1_is_causal():
    """Changing a FUTURE value must not change any PAST output."""
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 1, 8, 4) for _ in range(3))
    out1 = causal_attention(q, k, v)
    v2 = v.clone()
    v2[..., 5:, :] += 100.0
    out2 = causal_attention(q, k, v2)
    assert torch.allclose(out1[..., :5, :], out2[..., :5, :]), "position t sees tokens after t"
    assert not torch.allclose(out1[..., 5:, :], out2[..., 5:, :])


def test_2_1_first_token_attends_only_to_itself():
    q, k, v = (torch.randn(1, 1, 6, 4) for _ in range(3))
    assert torch.allclose(causal_attention(q, k, v)[..., 0, :], v[..., 0, :], atol=1e-6)
