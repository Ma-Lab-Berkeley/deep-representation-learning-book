r"""
Plot contraction-style product coefficients.

This script generates a plot of:

    \prod_{i=1}^{n} (1 - i / (a n^2 + i^2))

for multiple values of a, as a function of n.

Run:
    uv run contraction_coefficients.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def product_value(n: int, a: float) -> float:
    """Compute prod_{i=1..n} (1 - i / (a n^2 + i^2)) in float64."""
    i = np.arange(1, n + 1, dtype=np.float64)
    denom = a * (n**2) + i**2
    frac = i / denom
    # Numerically stable: log(prod(1 - frac)) = sum(log1p(-frac)).
    return float(np.exp(np.sum(np.log1p(-frac))))


def compute_curve(n_max: int, a: float) -> np.ndarray:
    """Compute the product for n = 1..n_max for a fixed a."""
    values = np.empty(n_max, dtype=np.float64)
    for n in range(1, n_max + 1):
        values[n - 1] = product_value(n=n, a=a)
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-max", type=int, default=1000, help="Maximum n to plot."
    )
    parser.add_argument(
        "--a-values",
        type=float,
        nargs="+",
        default=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        help="List of a values to plot.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).with_name("contraction_coefficients_plot.png"),
        help="Output image path.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive plot window.",
    )
    args = parser.parse_args()

    n_max: int = args.n_max
    a_values: list[float] = list(args.a_values)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))

    n_grid = np.arange(1, n_max + 1, dtype=int)
    for a in a_values:
        y = compute_curve(n_max=n_max, a=float(a))
        ax.plot(n_grid, y, linewidth=2, label=rf"$a = {a}$")

    ax.set_title(
        r"Plot of $\prod_{i=1}^{n}\left(1 - \frac{i}{a n^{2} + i^{2}}\right)$"
        r" for different values of $a$"
    )
    ax.set_xlabel("n")
    ax.set_ylabel("Product Value")
    ax.legend(loc="lower left", frameon=True)
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
