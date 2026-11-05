"""
Configuration for the model and the training run.

Nothing to implement here — but read it: every number is a design decision
you will be able to explain in an interview after this project.
"""
from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    vocab_size: int = 2048     # overwritten at train time from meta.json (2,048 in run 1, 4,096 in run 2)
    block_size: int = 256      # context length: how many tokens the model sees at once
    n_layer: int = 6           # number of transformer blocks stacked
    n_head: int = 6            # query heads
    n_kv_head: int = 2         # key/value heads (GQA). n_kv_head == n_head -> classic multi-head attention
    d_model: int = 384         # width of the residual stream
    dropout: float = 0.1
    rope_theta: float = 10000.0
    use_flash: bool = False    # False = YOUR attention code; True = PyTorch's fused kernel

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    def to_dict(self):
        return asdict(self)


@dataclass
class TrainConfig:
    batch_size: int = 32
    grad_accum: int = 1        # effective batch = batch_size * grad_accum
    max_iters: int = 5000
    warmup_iters: int = 200
    lr: float = 1e-3           # peak learning rate (small models tolerate higher LRs)
    min_lr: float = 1e-4       # floor of the cosine decay
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 250
    eval_iters: int = 50
    sample_interval: int = 500
    data_dir: str = "data"
    out_dir: str = "out"
    seed: int = 1337
    compile: bool = False      # torch.compile — CUDA only, worth it on the NVIDIA PC

    def to_dict(self):
        return asdict(self)


# Presets: pick one with `python train.py --preset mac`
PRESETS = {
    # ~0.6M params. Debug run: proves the pipeline works end to end in a couple of minutes.
    "tiny": (
        ModelConfig(block_size=128, n_layer=2, n_head=4, n_kv_head=2, d_model=128, dropout=0.0),
        TrainConfig(batch_size=16, max_iters=500, warmup_iters=50, eval_interval=100,
                    eval_iters=10, sample_interval=250),
    ),
    # ~10M params (10.2M with run 1's 2,048-token vocabulary). Apple Silicon (MPS).
    # Run 1, Machado only: 5,000 steps in 22 min; best val 3.513 at step 1,000, then overfits.
    "mac": (ModelConfig(), TrainConfig()),
    # ---- Stage 6, run 2: pretrain on open 19th-century Portuguese ...
    # Same architecture as `mac`; 11.0M params because the vocabulary is 4,096 (the embedding
    # table doubles). Dropout lowered: with this much data, overfitting is not the risk.
    # The corpus came out at 160M training tokens, so run 2 used --max_iters 20000
    # (20,000 x 8,192 tokens ≈ 1 pass): 82 min, best val 3.434 at step 18,000, no overfitting.
    "mac-pretrain": (
        ModelConfig(dropout=0.05),
        TrainConfig(max_iters=12000, warmup_iters=500, eval_interval=500, eval_iters=40,
                    sample_interval=2000, data_dir="data/pretrain", out_dir="out/pretrain"),
    ),
    # ... then fine-tune on Machado from that checkpoint (train.py --init_from out/pretrain/ckpt.pt).
    # Low learning rate (starting where pretraining ended, 1e-4), few steps, evaluate often: the
    # Machado set is small, so these 800 steps are ~6 passes over it. Run 2: 5 min, best val
    # 3.143 at step 650 (from 3.443 at step 0); the best-checkpoint rule keeps that one.
    "mac-finetune": (
        ModelConfig(dropout=0.1),
        TrainConfig(max_iters=800, warmup_iters=50, lr=1e-4, min_lr=1e-5, eval_interval=50,
                    eval_iters=40, sample_interval=200, data_dir="data/machado", out_dir="out/finetune"),
    ),
    # ~25M params. NVIDIA GPU (CUDA): longer context, fused attention, bf16, torch.compile.
    "pc": (
        ModelConfig(block_size=512, n_layer=8, n_head=8, n_kv_head=4, d_model=512,
                    dropout=0.15, use_flash=True),
        TrainConfig(batch_size=32, max_iters=8000, lr=6e-4, min_lr=6e-5,
                    eval_interval=500, sample_interval=1000, compile=True),
    ),
}
