#!/usr/bin/env python
"""unit_square_covering.py

Estimate the minimal radius required to cover the unit square [0,1]^2
with M = 2**n disks (closed Euclidean balls) of that radius, and output
both the radius and the centers of the disks.

The script formulates the problem as a continuous optimisation task:
  centres = argmin  max_{x in [0,1]^2}  min_{i} ||x - centres_i||_2
The objective is approximated on a dense grid of sample points, and the
`max`/`min` operations are implemented via `jnp.max`/`jnp.min`, whose
sub-gradients are handled automatically by JAX. Stochastic gradient
updates are applied using `optax.adam`.

Usage
-----
python unit_square_covering.py --n 4 --steps 8000 --grid 64 --lr 3e-2

Requires JAX and Optax.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple, Dict, Any

import jax
import jax.numpy as jnp
import optax
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

Array = jax.Array  # type: ignore[attr-defined]


def sample_grid(resolution: int) -> Array:
    """Return a (R*R, 2) array of uniformly spaced points on the unit square."""
    xs = jnp.linspace(0.0, 1.0, resolution)
    grid_x, grid_y = jnp.meshgrid(xs, xs, indexing="ij")
    return jnp.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)


def init_centres(key: jax.Array, m: int) -> Array:
    """Initialise M centres uniformly in the unit square."""
    return jax.random.uniform(key, shape=(m, 2))


def covering_radius(centres: Array, points: Array) -> Array:
    """Compute the worst-case distance from any point to its nearest centre."""
    diff = points[:, None, :] - centres[None, :, :]  # (N, M, 2)
    d = jnp.sum(diff**2, axis=-1)  # (N, M)
    min_d_per_point = jnp.min(d, axis=1)  # (N,)
    max_min_d = jnp.max(min_d_per_point)  # ()
    return max_min_d


def covering_loss(params: Dict[str, Array], points: Array, lam: float) -> Array:
    """Lagrangian loss: r^2  + lam * mean(relu(min_dist_sq - r^2))."""
    centres = params["centres"]
    log_r2 = params["log_r2"]  # unconstrained scalar
    r2 = jnp.exp(log_r2)  # ensures positivity

    diff = points[:, None, :] - centres[None, :, :]
    d_sq = jnp.sum(diff ** 2, axis=-1)
    min_d_sq = jnp.min(d_sq, axis=1)

    violations = jnp.maximum(min_d_sq - r2, 0.0)
    # Objective: radius squared + penalty for uncovered points
    return r2 + lam * jnp.mean(violations)


def optimise_covering(
    key: jax.Array,
    n: int,
    grid_res: int = 64,
    steps: int = 5000,
    lr: float = 1e-2,
    lam: float = 50.0,
) -> Tuple[Array, float]:
    """Optimise centres and radius using a Lagrangian loss."""
    m = 2 ** n
    points = sample_grid(grid_res)

    key_c, key_r = jax.random.split(key)
    params: Dict[str, Array] = {
        "centres": jax.random.uniform(key_c, shape=(m, 2)),
        "log_r2": jnp.array(jnp.log(0.05), dtype=jnp.float32),  # start small
    }

    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    opt_state = opt.init(params)

    @jax.jit
    def step(params: Dict[str, Array], opt_state: optax.OptState):
        loss, grads = jax.value_and_grad(covering_loss)(params, points, lam)
        updates, opt_state_new = opt.update(grads, opt_state)
        params_new = optax.apply_updates(params, updates)
        # Keep centres inside the unit square.
        params_new["centres"] = jnp.clip(params_new["centres"], 0.0, 1.0)  # type: ignore[arg-type]
        return params_new, opt_state_new, loss

    loss = jnp.inf
    for _ in tqdm(range(steps)):
        params, opt_state, loss = step(params, opt_state)

    r2_final = float(jnp.exp(params["log_r2"]))
    radius = float(jnp.sqrt(r2_final))
    centres = params["centres"]
    return centres, radius


def visualize_unit_square(centres: Array, radius: float) -> None:
    """Display the unit square and the covering disks."""
    num_discs = len(centres)
    fig, ax = plt.subplots(figsize=(6, 6))
    # Draw unit square
    square = patches.Rectangle((0, 0), 1, 1, linewidth=1.5, edgecolor="black", facecolor="none")
    ax.add_patch(square)

    # Draw circles
    centres_np = np.array(centres)
    for cx, cy in centres_np:
        circ = patches.Circle((cx, cy), radius, alpha=0.3, edgecolor="C0", facecolor="C0")
        ax.add_patch(circ)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal", "box")
    ax.set_title(f"{num_discs} discs, radius ≈ {radius:.4f}", fontsize=20)
    plt.show()


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Unit-square covering with 2^n balls.")
    parser.add_argument("--n", type=int, default=4, help="Exponent so that M=2^n disks are used.")
    parser.add_argument("--steps", type=int, default=8000, help="Gradient descent steps.")
    parser.add_argument("--grid", type=int, default=500, help="Sampling grid resolution.")
    parser.add_argument("--lr", type=float, default=2e-2, help="Learning rate.")
    parser.add_argument("--lam", type=float, default=1000.0, help="Lagrange penalty weight.")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed.")
    parser.add_argument("--no-plot", action="store_true", help="Disable the matplotlib visualization.")
    args = parser.parse_args(argv)

    if args.n <= 0:
        raise ValueError("n must be positive")

    key = jax.random.PRNGKey(args.seed)
    centres, radius = optimise_covering(key, args.n, args.grid, args.steps, args.lr, args.lam)

    print(f"n = {args.n} ⇒ M = {2**args.n} centres")
    print(f"Approx. covering radius: {radius:.6f}")
    print("Centres (x, y):")
    for i, c in enumerate(centres):
        x, y = map(float, c)
        print(f"  {i:3d}: {x:.6f}, {y:.6f}")

    if not args.no_plot:
        visualize_unit_square(centres, radius)


if __name__ == "__main__":
    main(sys.argv[1:])
