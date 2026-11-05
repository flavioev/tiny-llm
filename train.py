"""
STAGE 5 — The training loop.

    python train.py --preset tiny      # smoke test, a few minutes
    python train.py --preset mac       # run 1: Machado only, Apple Silicon
    python train.py --preset pc        # bigger run on the NVIDIA machine

    # run 2: pretrain on the library, then fine-tune on Machado
    python train.py --preset mac-pretrain --max_iters 20000
    python train.py --preset mac-finetune --init_from out/pretrain/ckpt.pt

One step: sample a batch -> forward pass -> loss -> backward pass -> update the
weights. Repeated max_iters times. See docs/the-big-picture.md for the diagram
and docs/05-training-loop.md for the details.
Tests: tests/test_stage5_training.py
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch

from config import PRESETS, ModelConfig, TrainConfig
from model import TinyGPT
from tokenizer import BPETokenizer


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_batch(data: np.ndarray, block_size: int, batch_size: int, device: str):
    """Sample `batch_size` random windows from the token array.

    x = data[i : i + block_size]
    y = data[i + 1 : i + 1 + block_size]     <- the same window shifted by one:
                                                at every position, the target is the NEXT token.
    Returns two int64 tensors of shape (batch_size, block_size) on `device`.
    """
    # Random starting points; the upper bound keeps the window AND the extra token
    # needed for y inside the data.
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    # Convert only the slices we use (not the whole dataset): `data` is a memory map,
    # so only what is read gets loaded, and the cost stays constant however large the
    # corpus. Tokens are stored as uint16 to save disk; PyTorch needs int64 ids.
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)  # same device as the model, or PyTorch refuses to mix them


def get_lr(it: int, cfg: TrainConfig) -> float:
    """Linear warmup, then cosine decay from cfg.lr down to cfg.min_lr.

        it < warmup_iters:   lr * (it + 1) / warmup_iters
        it >= max_iters:     min_lr
        otherwise:           progress = (it - warmup) / (max_iters - warmup)
                             min_lr + 0.5 * (1 + cos(pi * progress)) * (lr - min_lr)
    Why warmup: at step 0 the weights are random and gradients are wild; big steps
    would wreck the model. Why decay: finer steps at the end settle into a better minimum.
    """
    if it < cfg.warmup_iters:
        return cfg.lr * (it + 1) / cfg.warmup_iters
    if it >= cfg.max_iters:
        return cfg.min_lr
    # 0.5 * (1 + cos(pi * progress)) slides smoothly from 1 to 0: flat near the peak,
    # steepest in the middle, flat again near the floor.
    progress = (it - cfg.warmup_iters) / (cfg.max_iters - cfg.warmup_iters)
    return cfg.min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (cfg.lr - cfg.min_lr)


def train_step(model, optimizer, data, mcfg: ModelConfig, tcfg: TrainConfig,
               device: str, ctx=nullcontext()) -> float:
    """One optimizer step (with gradient accumulation). Returns the loss as a float."""
    loss = 0.0
    # Gradient accumulation: backward() ADDS to existing gradients, so several small
    # batches can be combined into one step (a bigger effective batch, same memory).
    # Dividing by grad_accum makes the total an average.
    for _ in range(tcfg.grad_accum):
        x, y = get_batch(data, mcfg.block_size, tcfg.batch_size, device)   # 1. sample a batch
        with ctx:                                                          #    (bf16 on CUDA)
            _, batch_loss = model(x, y)                                    # 2-3. forward + loss
        (batch_loss / tcfg.grad_accum).backward()                          # 4. gradients for every weight
        loss += batch_loss.item()
    # Safety: if one odd batch produces huge gradients, scale them down to a max norm.
    torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
    optimizer.step()                                                       # 5. update the weights
    optimizer.zero_grad(set_to_none=True)  # clear, or the next step would add onto these
    return loss / tcfg.grad_accum


@torch.no_grad()
def estimate_loss(model, splits: dict, mcfg, tcfg, device, ctx) -> dict:
    """Average loss over eval_iters random batches, for each split (train / val).

    One batch is noisy; averaging many gives a stable number. eval() turns dropout
    off so the measurement is not randomized.
    """
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(tcfg.eval_iters)
        for k in range(tcfg.eval_iters):
            x, y = get_batch(data, mcfg.block_size, tcfg.batch_size, device)
            with ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def load_pretrained(model, ckpt_path: str, device: str):
    """STAGE 6 — Fine-tuning: start from a pretrained checkpoint instead of random weights.

    The checkpoint was saved by this script as a dict:
        {"model": state_dict, "model_config": {...}, "iter": int, "val_loss": float}
    Fine-tuning only works if the architecture is identical: same vocabulary (the tokenizer
    must be the same one), same d_model, layers, heads, context. Check before loading, so a
    mismatch fails here with a clear message instead of deep inside PyTorch.
    """
    # map_location: a checkpoint saved on the Mac (mps) loads on a CPU or CUDA machine too.
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_config = ckpt["model_config"]
    model_config = model.config.to_dict()
    # Only the keys that decide the shape of a weight. dropout and use_flash may differ:
    # they change how the model runs, not what it stores (fine-tuning uses dropout 0.1,
    # pretraining 0.05).
    for key in ["vocab_size", "block_size", "n_layer", "n_head", "n_kv_head", "d_model"]:
        if ckpt_config[key] != model_config[key]:
            raise ValueError(f"Checkpoint architecture mismatch: {key} "
                             f"{ckpt_config[key]} != {model_config[key]}")
    # Every weight is replaced by its pretrained value. From here on the training loop runs
    # exactly as before; only the starting point, the data and the learning rate differ.
    model.load_state_dict(ckpt["model"])
    print(f"Loaded checkpoint {ckpt_path} from step {ckpt['iter']} with val loss {ckpt['val_loss']:.4f}")
    return ckpt["iter"], ckpt["val_loss"]


SNAPSHOT_STEPS = [0, 25, 50, 100, 150, 200, 300, 400, 500, 750, 1000, 1500,
                  2000, 2500, 3000, 4000, 5000, 6000, 8000, 10000, 12000, 15000, 20000]
# "Capitú" with the accent, as in the Machado edition. Runs 1 and 2 used "Capitu", which
# tokenizes like "Capítulo" and so never reached the character (docs/06b-run2-log.md, step 5).
SNAPSHOT_PROMPTS = ["Nem por isso me zanguei.", "Capitú"]


@torch.no_grad()
def save_snapshot(model, tok, it, val_loss, device, path, n_tokens=120, seed=42):
    """Generate from fixed prompts with a fixed seed and append them to a JSONL file.

    Same prompts + same seed at every snapshot, so the only thing that changes between
    two snapshots is the model's weights: this is the raw material for the
    "watch it learn" page.
    """
    model.eval()
    with open(path, "a", encoding="utf-8") as f:
        for prompt in SNAPSHOT_PROMPTS:
            torch.manual_seed(seed)
            start = torch.tensor([tok.encode(prompt)], device=device)
            out = model.generate(start, n_tokens, temperature=0.8, top_k=50)
            text = tok.decode(out[0].tolist())
            f.write(json.dumps({"iter": it, "val_loss": round(val_loss, 4),
                                "prompt": prompt, "text": text}, ensure_ascii=False) + "\n")
    model.train()
    torch.manual_seed(seed + it + 1)  # give training back its own randomness


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="mac", choices=PRESETS)
    ap.add_argument("--max_iters", type=int)
    ap.add_argument("--device")
    ap.add_argument("--prompt", default="\n\n", help="prompt for the samples printed during training")
    ap.add_argument("--data_dir", help="override the preset's data_dir")
    ap.add_argument("--out_dir", help="override the preset's out_dir")
    ap.add_argument("--init_from", help="start from this checkpoint instead of random weights (fine-tuning)")
    args = ap.parse_args()
    # Print each line immediately, even when piped into `tee` (otherwise Python buffers it).
    sys.stdout.reconfigure(line_buffering=True)

    # ---- setup: config, data, model
    mcfg, tcfg = PRESETS[args.preset]
    if args.max_iters:
        tcfg.max_iters = args.max_iters
    if args.data_dir:
        tcfg.data_dir = args.data_dir
    if args.out_dir:
        tcfg.out_dir = args.out_dir
    device = args.device or pick_device()
    torch.manual_seed(tcfg.seed)

    with open(os.path.join(tcfg.data_dir, "meta.json")) as f:
        meta = json.load(f)
    mcfg.vocab_size = meta["vocab_size"]
    # The tokenizer may live elsewhere: run 2 encodes Machado with the pretraining tokenizer,
    # and prepare_tokens.py records its path in meta.json.
    tok_path = meta.get("tokenizer") or os.path.join(tcfg.data_dir, "tokenizer.json")
    tok = BPETokenizer.load(tok_path)
    # memmap: the token files are read from disk on demand, not loaded all at once.
    train_data = np.memmap(os.path.join(tcfg.data_dir, "train.bin"), dtype=np.uint16, mode="r")
    val_data = np.memmap(os.path.join(tcfg.data_dir, "val.bin"), dtype=np.uint16, mode="r")

    model = TinyGPT(mcfg).to(device)
    if args.init_from:
        load_pretrained(model, args.init_from, device)
    print(f"device={device} | params={model.num_params() / 1e6:.2f}M | "
          f"train tokens={len(train_data) / 1e6:.2f}M | val tokens={len(val_data) / 1e6:.2f}M")
    tokens_per_step = tcfg.batch_size * tcfg.grad_accum * mcfg.block_size
    print(f"tokens/step={tokens_per_step} | epochs over the data ≈ "
          f"{tokens_per_step * tcfg.max_iters / len(train_data):.1f}")

    optimizer = model.configure_optimizer(tcfg.weight_decay, tcfg.lr, (tcfg.beta1, tcfg.beta2), device)
    # Mixed precision: bf16 on recent NVIDIA GPUs. MPS/CPU stay in float32 (simpler, robust).
    if device == "cuda" and torch.cuda.is_bf16_supported():
        ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        ctx = nullcontext()
    step_model = torch.compile(model) if (tcfg.compile and device == "cuda") else model

    # ---- outputs: loss log (for the curve) and snapshots (for "watch it learn")
    os.makedirs(tcfg.out_dir, exist_ok=True)
    log_path = os.path.join(tcfg.out_dir, "log.csv")
    log = open(log_path, "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["iter", "train_loss", "val_loss", "lr"])

    snap_path = os.path.join(tcfg.out_dir, "snapshots.jsonl")
    open(snap_path, "w").close()
    snap_steps = {s for s in SNAPSHOT_STEPS if s <= tcfg.max_iters} | {tcfg.max_iters}
    last_val = float("nan")

    # ---- the loop
    best_val = float("inf")
    t0 = run_start = time.time()
    for it in range(tcfg.max_iters + 1):
        lr = get_lr(it, tcfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # Measure train AND val loss. Val is text the model never trains on: when train
        # keeps falling but val stops, the model is memorizing (overfitting).
        if it % tcfg.eval_interval == 0 or it == tcfg.max_iters or it in snap_steps:
            losses = estimate_loss(step_model, {"train": train_data, "val": val_data},
                                   mcfg, tcfg, device, ctx)
            writer.writerow([it, f"{losses['train']:.4f}", f"{losses['val']:.4f}", f"{lr:.2e}"])
            log.flush()
            print(f"--- iter {it}: train {losses['train']:.3f} | val {losses['val']:.3f} "
                  f"| ppl {math.exp(losses['val']):.1f}")
            last_val = losses["val"]
            # Early stopping in practice: keep only the checkpoint with the best val loss.
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save({"model": model.state_dict(), "model_config": mcfg.to_dict(),
                            "iter": it, "val_loss": best_val},
                           os.path.join(tcfg.out_dir, "ckpt.pt"))

        # A quick sample in the terminal, to watch progress.
        if it > 0 and it % tcfg.sample_interval == 0:
            model.eval()
            start = torch.tensor([tok.encode(args.prompt)], device=device)
            out = model.generate(start, 80, temperature=0.8, top_k=50)
            print(">>> " + tok.decode(out[0].tolist()).replace("\n", " ")[:400])
            model.train()

        if it in snap_steps:
            save_snapshot(model, tok, it, last_val, device, snap_path)

        if it == tcfg.max_iters:
            break
        loss = train_step(step_model, optimizer, train_data, mcfg, tcfg, device, ctx)
        if it % 20 == 0:
            dt = time.time() - t0
            t0 = time.time()
            tps = tokens_per_step * (20 if it else 1) / max(dt, 1e-9)
            elapsed = time.time() - run_start
            eta = elapsed / (it + 1) * (tcfg.max_iters - it - 1)
            print(f"iter {it:5d}/{tcfg.max_iters} | loss {loss:.3f} | lr {lr:.2e} | "
                  f"{tps / 1e3:.1f}k tok/s | elapsed {elapsed / 60:4.1f} min | left ~{eta / 60:4.1f} min")

    log.close()
    print(f"done. best val loss {best_val:.3f} -> {tcfg.out_dir}/ckpt.pt, curve data -> {log_path}, "
          f"samples during training -> {snap_path}")


if __name__ == "__main__":
    main()
