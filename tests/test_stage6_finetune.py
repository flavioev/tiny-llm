import pytest
import torch

from config import ModelConfig
from model import TinyGPT
from train import load_pretrained

CFG = ModelConfig(vocab_size=97, block_size=32, n_layer=2, n_head=4, n_kv_head=2, d_model=64)


def save_ckpt(model, path, it=123, val_loss=3.21):
    torch.save({"model": model.state_dict(), "model_config": model.config.to_dict(),
                "iter": it, "val_loss": val_loss}, path)


def test_6_1_weights_are_loaded(tmp_path):
    torch.manual_seed(0)
    pretrained = TinyGPT(CFG)
    path = tmp_path / "ckpt.pt"
    save_ckpt(pretrained, path)

    torch.manual_seed(1)
    fresh = TinyGPT(CFG)                                   # different random weights
    assert not torch.equal(fresh.tok_emb.weight, pretrained.tok_emb.weight)
    load_pretrained(fresh, str(path), "cpu")
    for (n1, p1), (n2, p2) in zip(fresh.named_parameters(), pretrained.named_parameters()):
        assert n1 == n2 and torch.equal(p1, p2), f"{n1} differs after loading"


def test_6_1_dropout_may_differ(tmp_path):
    pretrained = TinyGPT(ModelConfig(**{**CFG.to_dict(), "dropout": 0.05}))
    path = tmp_path / "ckpt.pt"
    save_ckpt(pretrained, path)
    load_pretrained(TinyGPT(ModelConfig(**{**CFG.to_dict(), "dropout": 0.1})), str(path), "cpu")


@pytest.mark.parametrize("key,value", [("vocab_size", 98), ("n_layer", 3), ("d_model", 128)])
def test_6_1_architecture_mismatch_is_rejected(tmp_path, key, value):
    pretrained = TinyGPT(CFG)
    path = tmp_path / "ckpt.pt"
    save_ckpt(pretrained, path)
    other = TinyGPT(ModelConfig(**{**CFG.to_dict(), key: value}))
    with pytest.raises(ValueError, match=key):
        load_pretrained(other, str(path), "cpu")
