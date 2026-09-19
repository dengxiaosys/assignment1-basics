"""Plot common activation functions used around SwiGLU, saved as SVG.

Run:
    uv run --with matplotlib python notes/scripts/plot_activations.py
Output:
    notes/images/activation_functions.svg
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def silu(x):
    return x * sigmoid(x)


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def relu(x):
    return np.maximum(0.0, x)


def main():
    x = np.linspace(-6.0, 6.0, 1000)

    fig, ax = plt.subplots(figsize=(8.0, 5.2), dpi=100)

    ax.plot(x, silu(x), color="#ef4444", lw=3.0, label=r"SiLU = x·σ(x)", zorder=5)
    ax.plot(x, gelu(x), color="#22c55e", lw=2.2, label="GELU")
    ax.plot(x, relu(x), color="#94a3b8", lw=2.2, ls="--", label="ReLU = max(0, x)")
    ax.plot(x, sigmoid(x), color="#a855f7", lw=2.2, label=r"Sigmoid = σ(x)")
    ax.plot(x, np.tanh(x), color="#0ea5e9", lw=2.2, label="Tanh")

    # SiLU minimum near x = -1.278, y = -0.278
    xm = -1.2784
    ym = silu(xm)
    ax.plot([xm], [ym], "o", color="#b45309", ms=6, zorder=6)
    ax.annotate(
        f"SiLU min ≈ ({xm:.2f}, {ym:.2f})",
        xy=(xm, ym),
        xytext=(-5.6, -1.7),
        fontsize=10,
        color="#b45309",
        arrowprops=dict(arrowstyle="->", color="#b45309", lw=1.0),
    )

    ax.axhline(0, color="#334155", lw=1.2)
    ax.axvline(0, color="#334155", lw=1.2)
    ax.grid(True, color="#e2e8f0", lw=0.8)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-3, 6)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title("Activation functions: SiLU, GELU, ReLU, Sigmoid, Tanh")
    ax.legend(loc="upper left", framealpha=0.95)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "images")
    out_dir = os.path.abspath(out_dir)
    out_path = os.path.join(out_dir, "activation_functions.svg")
    fig.tight_layout()
    fig.savefig(out_path, format="svg")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
