"""Plot a concrete get_batch example, saved as SVG.

A worked example with dataset = arange(16), context_length=4, batch_size=3,
starts = [2, 7, 11] (11 is the max legal start, showing the boundary).

Top panel : the 1D token stream, the legal-start range [0, len-ctx], and the
            three sampled input windows (x).
Bottom    : one sample (start=2), showing y is x shifted right by one, and each
            x[k] predicts y[k] = x[k+1] ("predict the next token").

Labels in English (env font lacks CJK glyphs). Prose stays in the Markdown note.

Run:
    uv run --with matplotlib python notes/scripts/plot_get_batch.py
Output:
    notes/images/get_batch_example.svg
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

IMG_DIR = os.path.join(os.path.dirname(__file__), "..", "images")

BLUE = "#2c7fb8"
ORANGE = "#e6811a"
GREEN = "#2ca25f"
GREY = "#b8b8b8"


def cell(ax, x, y, text, w=1.0, h=0.8, fc="white", ec="#333", tc="#222", lw=1.0, fs=11, weight="normal"):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, str(text), ha="center", va="center", fontsize=fs, color=tc, zorder=3, weight=weight)


def main():
    n, ctx = 16, 4
    dataset = list(range(n))
    starts = [2, 7, 11]
    colors = [BLUE, ORANGE, GREEN]
    max_start = n - ctx  # 12; legal start indices are [0, 11]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.6), gridspec_kw={"height_ratios": [1.15, 1]})

    # ================= Panel 1: the token stream + sampled windows =================
    ax1.set_xlim(-0.7, n + 0.2)
    ax1.set_ylim(-4.4, 3.4)
    ax1.axis("off")
    ax1.set_title("1) get_batch: sample random windows from a 1D token stream "
                  "(dataset=arange(16), ctx=4, batch_size=3)", fontsize=12, weight="bold")

    # the dataset row
    for i in range(n):
        cell(ax1, i, 0, dataset[i], fc="#f7f7f7")
        ax1.text(i + 0.5, -0.28, f"{i}", ha="center", va="top", fontsize=8, color="#999")
    ax1.text(-0.6, 0.4, "dataset\n(token id)", ha="right", va="center", fontsize=9, color="#555")
    ax1.text(n / 2, -0.75, "index i", ha="center", va="top", fontsize=9, color="#999")

    # legal-start bracket: start i can be 0..len-ctx-1 = 0..11 (left edges)
    y_br = 1.35
    ax1.annotate("", xy=(max_start, y_br), xytext=(0, y_br),
                 arrowprops=dict(arrowstyle="<->", color="#c0392b", lw=1.6))
    ax1.text(max_start / 2, y_br + 0.22,
             "legal start  i in [0, len-ctx) = [0, 11]   (so y can still reach index len-1=15)",
             ha="center", va="bottom", fontsize=9.5, color="#c0392b")
    # markers at the three chosen starts
    for s, c in zip(starts, colors):
        ax1.annotate("", xy=(s + 0.05, 0.85), xytext=(s + 0.05, 1.3),
                     arrowprops=dict(arrowstyle="-|>", color=c, lw=1.8))

    # the three sampled x-windows, each on its own row to avoid label collisions
    for k, (s, c) in enumerate(zip(starts, colors)):
        row_y = -1.7 - k * 0.95  # stack downward: -1.7, -2.65, -3.6
        # highlight the window on the stream
        ax1.add_patch(Rectangle((s, 0), ctx, 0.8, facecolor="none", edgecolor=c, lw=2.4, zorder=4))
        # the extracted x row (aligned under its real position)
        for j in range(ctx):
            cell(ax1, s + j, row_y, dataset[s + j], fc="white", ec=c, lw=1.8, tc=c, weight="bold")
        ax1.text(s - 0.15, row_y + 0.4, f"x[{k}]", ha="right", va="center", fontsize=9.5, color=c, weight="bold")
        ax1.text(s + ctx + 0.15, row_y + 0.4, f"start={s}", ha="left", va="center", fontsize=8.5, color=c)

    # ================= Panel 2: x vs y shift for one sample =================
    s = starts[0]  # 2
    ax2.set_xlim(s - 0.8, s + ctx + 1.4)
    ax2.set_ylim(-1.2, 3.3)
    ax2.axis("off")
    ax2.set_title("2) one sample (start=2): y is x shifted right by one -> each x[k] predicts y[k] = x[k+1]",
                  fontsize=12, weight="bold")

    x_y_row = 2.0
    y_y_row = 0.4

    # x row: indices [s, s+ctx)
    for j in range(ctx):
        cell(ax2, s + j, x_y_row, dataset[s + j], fc="#eaf3fa", ec=BLUE, lw=1.8, tc=BLUE, weight="bold")
        ax2.text(s + j + 0.5, x_y_row + 0.95, f"x[{j}]", ha="center", va="bottom", fontsize=9, color=BLUE)
    ax2.text(s - 0.65, x_y_row + 0.4, "x", ha="right", va="center", fontsize=13, color=BLUE, weight="bold")

    # y row: indices [s+1, s+1+ctx), shifted right by one cell
    for j in range(ctx):
        cell(ax2, s + 1 + j, y_y_row, dataset[s + 1 + j], fc="#eafaf1", ec=GREEN, lw=1.8, tc=GREEN, weight="bold")
        ax2.text(s + 1 + j + 0.5, y_y_row - 0.32, f"y[{j}]", ha="center", va="top", fontsize=9, color=GREEN)
    ax2.text(s - 0.65, y_y_row + 0.4, "y", ha="right", va="center", fontsize=13, color=GREEN, weight="bold")

    # arrows: x[k] -> y[k] (down-right by one), meaning "target = next token"
    for j in range(ctx):
        arr = FancyArrowPatch((s + j + 0.5, x_y_row),
                              (s + 1 + j + 0.5, y_y_row + 0.8),
                              arrowstyle="-|>", mutation_scale=12,
                              color="#c0392b", lw=1.3, zorder=5,
                              connectionstyle="arc3,rad=-0.15")
        ax2.add_patch(arr)
    ax2.text(s + ctx / 2 + 0.5, (x_y_row + y_y_row) / 2 + 0.55,
             "predict next", ha="center", va="center", fontsize=9.5, color="#c0392b",
             style="italic", rotation=0)

    # note the overlap: values 3,4,5 appear in both x and y
    ax2.text((2 * s + ctx + 1) / 2 + 0.5, -0.95,
             "values 3,4,5 appear in both rows -> y is literally x moved forward one step",
             ha="center", va="center", fontsize=9, color="#666")

    fig.tight_layout(h_pad=2.0)
    out = os.path.join(IMG_DIR, "get_batch_example.svg")
    fig.savefig(out, format="svg", bbox_inches="tight")
    # also a PNG for quick eyeballing
    fig.savefig(out.replace(".svg", ".png"), format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved={os.path.abspath(out)}")


if __name__ == "__main__":
    main()
