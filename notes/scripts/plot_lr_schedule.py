"""Plot the cosine learning-rate schedule with linear warmup, saved as SVG.

Labels in English (env font lacks CJK glyphs). Prose stays in the Markdown note.

Run:
    uv run --with matplotlib python notes/scripts/plot_lr_schedule.py
Output:
    notes/images/lr_cosine_schedule.svg
"""

import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG_DIR = os.path.join(os.path.dirname(__file__), "..", "images")


def lr_at(it, max_lr, min_lr, warmup, cosine_end):
    if it < warmup:
        return it / warmup * max_lr
    if it <= cosine_end:
        coeff = 0.5 * (1 + math.cos((it - warmup) / (cosine_end - warmup) * math.pi))
        return min_lr + coeff * (max_lr - min_lr)
    return min_lr


def main():
    max_lr, min_lr, Tw, Tc = 1.0, 0.1, 7, 21
    end = 28
    xs = [t / 4 for t in range(0, end * 4 + 1)]  # 细分以画出平滑曲线
    ys = [lr_at(t, max_lr, min_lr, Tw, Tc) for t in xs]

    fig, ax = plt.subplots(figsize=(9, 4.8))

    # 三段背景着色
    ax.axvspan(0, Tw, color="#e3f0ff", alpha=0.7, label="warmup (linear up)")
    ax.axvspan(Tw, Tc, color="#fff0e0", alpha=0.7, label="cosine annealing")
    ax.axvspan(Tc, end, color="#eaeaea", alpha=0.7, label="post (constant min)")

    ax.plot(xs, ys, color="#c0392b", linewidth=2.2, zorder=5)

    # 关键点
    for t, txt in [(0, "t=0: lr=0"), (Tw, "t=Tw=7: lr=max=1.0"), (Tc, "t=Tc=21: lr=min=0.1")]:
        y = lr_at(t, max_lr, min_lr, Tw, Tc)
        ax.scatter([t], [y], color="#c0392b", zorder=6, s=36)
        ax.annotate(txt, (t, y), textcoords="offset points", xytext=(6, 10 if t < Tc else -16),
                    fontsize=9, color="#7a1f16")

    ax.axhline(max_lr, ls="--", lw=0.8, color="#888")
    ax.axhline(min_lr, ls="--", lw=0.8, color="#888")
    ax.set_xlabel("iteration t")
    ax.set_ylabel("learning rate")
    ax.set_title("Cosine LR schedule with linear warmup\n(max=1.0, min=0.1, Tw=7, Tc=21)",
                 fontsize=12, weight="bold")
    ax.set_xlim(0, end)
    ax.set_ylim(0, 1.12)
    ax.legend(loc="center right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = os.path.join(IMG_DIR, "lr_cosine_schedule.svg")
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"saved={os.path.abspath(out)}")


if __name__ == "__main__":
    main()
