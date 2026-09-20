"""Plot Multi-Head Self-Attention diagrams, saved as SVG.

Labels are in English because the default matplotlib font in this env lacks
CJK glyphs (Chinese would render as tofu boxes). The prose explanation stays
in the Markdown note.

Run:
    uv run --with matplotlib python notes/scripts/plot_mha.py
Output:
    notes/images/mha_pipeline.svg      (SDPA vs MHA nested dataflow)
    notes/images/mha_split_heads.svg   (tensor shape transforms for head splitting)
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

IMG_DIR = os.path.join(os.path.dirname(__file__), "..", "images")


def _box(ax, x, y, w, h, text, fc, ec="#333333", fontsize=10, tc="#111111"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.4, edgecolor=ec, facecolor=fc,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=tc, zorder=5)


def _arrow(ax, x1, y1, x2, y2, color="#555555"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                 arrowstyle="-|>", mutation_scale=14,
                 linewidth=1.3, color=color, zorder=1))


def plot_pipeline():
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    y = 1.6
    h = 1.0
    param = "#cfe3f7"   # has learnable params (Linear)
    comp = "#ffe1c2"    # param-free reshape/transpose
    core = "#f7c6c6"    # SDPA core

    _box(ax, 0.1, y, 1.1, h, "x\n(.., seq,\nd_model)", "#e8e8e8", fontsize=8.5)
    _arrow(ax, 1.2, y + h / 2, 1.6, y + h / 2)
    _box(ax, 1.6, y, 1.5, h, "q/k/v_proj\n(Linear x3)\n[params]", param, fontsize=8.5)
    _arrow(ax, 3.1, y + h / 2, 3.5, y + h / 2)
    _box(ax, 3.5, y, 1.5, h, "split heads\nreshape+\ntranspose", comp, fontsize=8.5)
    _arrow(ax, 5.0, y + h / 2, 5.4, y + h / 2)
    _box(ax, 5.4, y, 1.7, h, "SDPA\nsoftmax(QKt/sqrt d)V\n+causal mask", core, fontsize=8.0)
    _arrow(ax, 7.1, y + h / 2, 7.5, y + h / 2)
    _box(ax, 7.5, y, 1.4, h, "merge heads\ntranspose+\nreshape", comp, fontsize=8.5)
    _arrow(ax, 8.9, y + h / 2, 9.3, y + h / 2)
    _box(ax, 9.3, y, 1.5, h, "output_proj\n(Linear)\n[params]", param, fontsize=8.5)

    # SDPA bracket (below)
    ax.annotate("", xy=(5.4, y - 0.25), xytext=(7.1, y - 0.25),
                arrowprops=dict(arrowstyle="-", color="#c0392b", lw=1.2))
    ax.text(6.25, y - 0.52, "this step IS the SDPA function", ha="center",
            va="center", fontsize=8.5, color="#c0392b")

    # MHA bracket (above)
    ax.annotate("", xy=(1.6, y + h + 0.3), xytext=(10.8, y + h + 0.3),
                arrowprops=dict(arrowstyle="-", color="#2c3e50", lw=1.2))
    ax.text(6.2, y + h + 0.6,
            "MHA = proj + split + SDPA + merge + output_proj",
            ha="center", va="center", fontsize=10.5, color="#2c3e50", weight="bold")

    # legend
    _box(ax, 0.3, 0.15, 0.45, 0.3, "", param)
    ax.text(0.85, 0.3, "has params (Linear)", va="center", fontsize=8.5)
    _box(ax, 3.1, 0.15, 0.45, 0.3, "", comp)
    ax.text(3.65, 0.3, "param-free reshape", va="center", fontsize=8.5)
    _box(ax, 5.9, 0.15, 0.45, 0.3, "", core)
    ax.text(6.45, 0.3, "SDPA core (param-free)", va="center", fontsize=8.5)

    ax.set_title("SDPA vs MHA: SDPA is one step inside MHA",
                 fontsize=12.5, weight="bold")
    fig.tight_layout()
    out = os.path.join(IMG_DIR, "mha_pipeline.svg")
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"saved={os.path.abspath(out)}")


def plot_split_heads():
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.6)
    ax.axis("off")

    y = 1.5
    h = 1.0
    c1 = "#cfe3f7"
    c2 = "#d6f0d6"
    c3 = "#ffe1c2"

    _box(ax, 0.2, y, 2.0, h, "proj output\n(.., seq, d_model)\nd_model=64", c1, fontsize=9)
    _arrow(ax, 2.2, y + h / 2, 2.8, y + h / 2)
    _box(ax, 2.8, y, 2.3, h, "reshape\n(.., seq,\nnum_heads, head_dim)\n(4, 16)", c2, fontsize=9)
    _arrow(ax, 5.1, y + h / 2, 5.7, y + h / 2)
    _box(ax, 5.7, y, 2.4, h, "transpose(-3,-2)\n(.., num_heads,\nseq, head_dim)", c3, fontsize=9)

    ax.text(6.9, y - 0.5, "num_heads becomes a batch dim\n-> SDPA runs per head in parallel",
            ha="center", va="center", fontsize=8.5, color="#7f5500")

    ax.text(5.0, y + h + 0.45,
            "split heads: d_model = num_heads x head_dim  (64 = 4 x 16)",
            ha="center", va="center", fontsize=10.5, weight="bold", color="#2c3e50")

    ax.set_title("Tensor shape transforms for head splitting",
                 fontsize=12.5, weight="bold")
    fig.tight_layout()
    out = os.path.join(IMG_DIR, "mha_split_heads.svg")
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"saved={os.path.abspath(out)}")


def plot_transpose_why():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))

    for ax in axes:
        ax.set_xlim(0, 5.4)
        ax.set_ylim(0, 3.4)
        ax.axis("off")

    good = "#d6f0d6"
    bad = "#f7c6c6"
    hl = "#ffe1c2"

    # LEFT: without transpose (WRONG)
    ax = axes[0]
    ax.set_title("WITHOUT transpose  (WRONG)", fontsize=11, weight="bold", color="#c0392b")
    _box(ax, 0.4, 2.2, 4.6, 0.8, "(.., seq, num_heads, head_dim)", bad, fontsize=10.5)
    ax.text(2.7, 1.75, "last two dims = (num_heads, head_dim)",
            ha="center", va="center", fontsize=8.5, color="#7f5500")
    ax.text(2.7, 1.05, "SDPA treats last two dims as (seq, feat)\n"
                       "=> attends across num_heads (WRONG!)",
            ha="center", va="center", fontsize=9.5, color="#c0392b")

    # RIGHT: with transpose (CORRECT)
    ax = axes[1]
    ax.set_title("WITH transpose(-3,-2)  (CORRECT)", fontsize=11, weight="bold", color="#2c7a2c")
    _box(ax, 0.4, 2.2, 4.6, 0.8, "(.., num_heads, seq, head_dim)", good, fontsize=10.5)
    ax.text(2.7, 1.75, "last two dims = (seq, head_dim)",
            ha="center", va="center", fontsize=8.5, color="#7f5500")
    ax.text(2.7, 1.05, "num_heads is a batch dim\n"
                       "=> attends across seq, per head (CORRECT)",
            ha="center", va="center", fontsize=9.5, color="#2c7a2c")

    fig.suptitle("Why transpose is required: it decides WHICH dim SDPA attends over",
                 fontsize=12, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(IMG_DIR, "mha_transpose_why.svg")
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"saved={os.path.abspath(out)}")


if __name__ == "__main__":
    plot_pipeline()
    plot_split_heads()
    plot_transpose_why()
