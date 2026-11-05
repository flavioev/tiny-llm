"""
STAGE 6 — The loss curve (the image for your LinkedIn post).

    python plot_loss.py            -> out/loss_curve.png
"""
import csv
import math

import matplotlib.pyplot as plt


def main(log_path="out/log.csv", out_path="out/loss_curve.png"):
    its, tr, va = [], [], []
    with open(log_path) as f:
        for row in csv.DictReader(f):
            its.append(int(row["iter"]))
            tr.append(float(row["train_loss"]))
            va.append(float(row["val_loss"]))

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.plot(its, tr, label="train", linewidth=2)
    ax.plot(its, va, label="validation", linewidth=2)
    ax.set_xlabel("training step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title(f"Tiny LLM from scratch — best val perplexity {math.exp(min(va)):.1f}")
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
