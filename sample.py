"""
STAGE 6 — Talk to your model.

    python sample.py --prompt "Capítulo I" --n 3                        # run 1 (defaults)
    python sample.py --prompt "A mulher" --temperature 1.1 --top_k 100
    python sample.py --ckpt out/finetune/ckpt.pt --tokenizer data/pretrain/tokenizer.json \
                     --prompt "Capitú" --seed 42                       # run 2, fine-tuned

The model is not a chatbot: it continues the prompt, one token at a time. Start a
sentence the way Machado would, and use his edition's spelling ("Capitú", with the
accent) — a different spelling is a different sequence of tokens.
The checkpoint and the tokenizer must come from the same run.
Side by side, in a notebook: notebooks/try_the_model.ipynb
"""
import argparse

import torch

from config import ModelConfig
from model import TinyGPT
from tokenizer import BPETokenizer
from train import pick_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/ckpt.pt")
    ap.add_argument("--tokenizer", default="data/tokenizer.json")
    ap.add_argument("--prompt", default="\n\n")
    ap.add_argument("--n", type=int, default=3, help="number of samples")
    ap.add_argument("--max_new_tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.8,
                    help="<1 = safer and more repetitive, >1 = wilder")
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = pick_device()
    ckpt = torch.load(args.ckpt, map_location=device)
    model = TinyGPT(ModelConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = BPETokenizer.load(args.tokenizer)
    print(f"(checkpoint from iter {ckpt['iter']}, val loss {ckpt['val_loss']:.3f})\n")

    start = torch.tensor([tok.encode(args.prompt)], device=device)
    for i in range(args.n):
        out = model.generate(start, args.max_new_tokens, args.temperature, args.top_k, tok.eot_id)
        print(f"===== sample {i + 1} =====")
        print(tok.decode(out[0].tolist()))
        print()


if __name__ == "__main__":
    main()
