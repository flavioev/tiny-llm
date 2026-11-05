"""
After STAGE 1 — Train the tokenizer on a corpus and turn the corpus into token ids.

    python prepare_tokens.py                                   # Machado, vocab 2048 (run 1)
    python prepare_tokens.py --data_dir data/pretrain --vocab_size 4096          # run 2, step 1
    python prepare_tokens.py --data_dir data/machado --corpus data/corpus.txt \\
                             --tokenizer data/pretrain/tokenizer.json           # run 2, step 2

Writes, in --data_dir:
    tokenizer.json          the learned merges (unless --tokenizer reuses an existing one)
    train.bin, val.bin      token ids as uint16 (90% / 10% of each document)
    meta.json               vocab size, token counts

For pretraining + fine-tuning, BOTH corpora must be encoded with the SAME tokenizer
(the model's embedding table is indexed by token id), hence --tokenizer.
"""
import argparse
from array import array
import sys
import json
import os
import random
import time

import numpy as np

from tokenizer import EOT, BPETokenizer


def sample_for_training(corpus: str, n_chars: int, seed: int = 0) -> str:
    """A spread-out sample of the corpus: paragraphs drawn from everywhere, not just the start."""
    if len(corpus) <= n_chars:
        return corpus
    paragraphs = corpus.replace(EOT, "\n").split("\n")
    random.Random(seed).shuffle(paragraphs)
    out, total = [], 0
    for p in paragraphs:
        if total >= n_chars:
            break
        out.append(p)
        total += len(p) + 1
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data", help="where to write tokenizer.json / train.bin / val.bin")
    ap.add_argument("--corpus", default=None, help="input text (default: <data_dir>/corpus.txt)")
    ap.add_argument("--tokenizer", default=None, help="reuse this tokenizer.json instead of training one")
    ap.add_argument("--vocab_size", type=int, default=2048)
    ap.add_argument("--train_chars", type=int, default=10_000_000,
                    help="train the tokenizer on a sample of this many characters (pure Python is slow)")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # show progress live, even through `tee`
    corpus_path = args.corpus or os.path.join(args.data_dir, "corpus.txt")
    os.makedirs(args.data_dir, exist_ok=True)

    with open(corpus_path, encoding="utf-8") as f:
        corpus = f.read()
    print(f"corpus: {len(corpus) / 1e6:.2f}M characters from {corpus_path}")

    t0 = time.time()
    if args.tokenizer:
        tok = BPETokenizer.load(args.tokenizer)
        print(f"reusing tokenizer {args.tokenizer} (vocab_size={tok.vocab_size})")
    else:
        tok = BPETokenizer()
        tok.train(sample_for_training(corpus, args.train_chars), args.vocab_size, verbose=True)
        tok.save(os.path.join(args.data_dir, "tokenizer.json"))
        print(f"tokenizer trained in {time.time() - t0:.0f}s, vocab_size={tok.vocab_size}")

    # Split each document 90/10 so validation covers every document's style.
    docs = corpus.split(EOT)
    del corpus  # free ~0.5 GB before encoding; it is re-read below only for the sample
    # array('H') stores each id in 2 bytes (uint16) instead of a Python list's 8-byte pointers:
    # ~120 M tokens fit in ~240 MB instead of ~1 GB.
    train_ids, val_ids = array("H"), array("H")
    t0 = time.time()
    for i, doc in enumerate(docs):
        ids = tok.encode(doc.strip()) + [tok.eot_id]
        cut = int(0.9 * len(ids))
        train_ids.extend(ids[:cut])
        val_ids.extend(ids[cut:])
        if (i + 1) % 100 == 0 or i + 1 == len(docs):
            done = (i + 1) / len(docs)
            print(f"  encoded {i + 1}/{len(docs)} documents | {(len(train_ids) + len(val_ids)) / 1e6:.1f}M tokens "
                  f"| {(time.time() - t0) / 60:.1f} min elapsed, ~{(time.time() - t0) / done * (1 - done) / 60:.1f} left")

    assert tok.vocab_size <= 65535, "uint16 storage"
    for name, ids in [("train", train_ids), ("val", val_ids)]:
        np.frombuffer(ids, dtype=np.uint16).tofile(os.path.join(args.data_dir, f"{name}.bin"))
    n_chars = os.path.getsize(corpus_path)  # bytes ≈ characters (UTF-8, mostly ASCII)
    meta = {"vocab_size": tok.vocab_size, "train_tokens": len(train_ids), "val_tokens": len(val_ids),
            "chars_per_token": n_chars / (len(train_ids) + len(val_ids)),
            "tokenizer": args.tokenizer or os.path.join(args.data_dir, "tokenizer.json"),
            "corpus": corpus_path}
    with open(os.path.join(args.data_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))

    with open(corpus_path, encoding="utf-8") as f:
        sample = f.read(1200)[1000:1200]
    pieces = [tok.decode([i]) for i in tok.encode(sample)]
    print("\nHow the tokenizer cuts text:\n" + "|".join(pieces))


if __name__ == "__main__":
    main()
