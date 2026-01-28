from typing import Union

import jax
import jax.numpy as jnp
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from diffusion.gmm import gmm_sample
from diffusion.processes import DenoisingProcess
from jax import Array

# Set up dimensions and parameters
D = 2  # 2D data
K = 3  # 3 components
N = 60  # ground truth samples
N_sample = 60  # samples to denoise

seed = 4
key = jax.random.PRNGKey(seed)

# Create 2D data: two lines (rank 1) and one full-rank 2D Gaussian (rank 2).
# Use separated means so the three components are visually distinct.
mus = jnp.array([[-3.0, 0.0], [3.0, 0.0], [0.0, 3.0]])

scale_for_plotting = 3.0

# First component: a line (rank 1 covariance)
key, subkey = jax.random.split(key)
factor_1 = jax.random.normal(subkey, shape=(D, 1))
Sigma_1 = factor_1 @ factor_1.T * 2.0

# Second component: another line (rank 1 covariance)
key, subkey = jax.random.split(key)
factor_2 = jax.random.normal(subkey, shape=(D, 1))
Sigma_2 = factor_2 @ factor_2.T * 2.0

# Third component: full-rank (rank 2) covariance in 2D
key, subkey = jax.random.split(key)
factor_3 = jax.random.normal(subkey, shape=(D, 2))
Sigma_3 = factor_3 @ factor_3.T

Sigmas = jnp.stack([Sigma_1, Sigma_2, Sigma_3], axis=0) * scale_for_plotting

pi = jnp.ones((K,)) / K

key, subkey = jax.random.split(key)
X, y = gmm_sample(subkey, N, D, K, pi, mus, Sigmas)

# Set up the denoising process
def alpha_fn(t: Union[Array, float]) -> Array:
    t_arr = jnp.asarray(t)
    return jnp.sqrt(1 - t_arr**2)


def sigma_fn(t: Union[Array, float]) -> Array:
    return jnp.asarray(t)

noising_process = DenoisingProcess(alpha_fn, sigma_fn)

# Set up the noise schedule
L = 500
t_min = 0.001
t_max = 0.999
ts = jnp.linspace(t_max, t_min, L + 1)

# Generate "pure noise" by noising the training samples at t_max
key, subkey = jax.random.split(key)
_, sk2 = jax.random.split(subkey)
assert N == N_sample
X_noise_seed = X
X_noise = alpha_fn(t_max) * X_noise_seed + sigma_fn(t_max) * jax.random.normal(
    sk2, shape=(N_sample, D)
)


def memorizing_score_ce(
    X_noisy: Array, alpha: Array, sigma: Array, X_train: Array
) -> Array:
    """
    Conditional expectation using a "memorizing score" based on a GMM where each
    component corresponds to a training sample with (approx) zero covariance.
    """
    Np, Dp = X_noisy.shape
    assert Dp == X_train.shape[1]

    # Broadcast: X_noisy (Np, 1, Dp), X_train (1, M, Dp)
    X_noisy_expanded = X_noisy[:, None, :]
    X_train_expanded = X_train[None, :, :]

    # |x_noisy - alpha * x_train|^2
    sq_distances = jnp.sum(
        (X_noisy_expanded - alpha * X_train_expanded) ** 2, axis=-1
    )  # (Np, M)

    # Posterior weights (temperature sigma^2)
    weights = jax.nn.softmax(-sq_distances / (2 * sigma**2), axis=-1)  # (Np, M)

    # Weighted sum of training points
    return jnp.sum(weights[:, :, None] * X_train_expanded, axis=1)  # (Np, Dp)


def ce_func(x: Array, t: Array) -> Array:
    return memorizing_score_ce(x, alpha_fn(t), sigma_fn(t), X)


# We visualize 6 panels: original data + 5 denoising snapshots (3 rows × 2 cols).
steps_to_visualize = [0, 200, 300, 400, 500]

fig = plt.figure(figsize=(10, 14))
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.085, wspace=0.06)

colors = ["red", "orange", "gray"]
color_map = {0: colors[0], 1: colors[1], 2: colors[2]}
point_colors = [color_map[int(label)] for label in y]

axis_lim = 7
marker_size = 55

# Panel 0: original data
ax0 = fig.add_subplot(gs[0, 0])
ax0.scatter(X[:, 0], X[:, 1], c=point_colors, s=marker_size, alpha=1.0)
ax0.set_title("Original Data", fontsize=18, pad=4)
ax0.set_xlim(-axis_lim, axis_lim)
ax0.set_ylim(-axis_lim, axis_lim)
ax0.set_aspect("equal", adjustable="box")
ax0.set_xticks([])
ax0.set_yticks([])
ax0.grid(False)

# Precompute the full denoising trajectory once (much faster than re-running for
# each step with a different `ts[:step+1]`, which can trigger multiple compilations).
def _step_fn(x: Array, m: Array):
    t_m = ts[m]
    t_m_plus_1 = ts[m + 1]
    alpha_m = alpha_fn(t_m)
    alpha_m_plus_1 = alpha_fn(t_m_plus_1)
    sigma_m = sigma_fn(t_m)
    sigma_m_plus_1 = sigma_fn(t_m_plus_1)
    ce_m = ce_func(x, t_m)
    x_new = (sigma_m_plus_1 / sigma_m) * x + (
        alpha_m_plus_1 - (sigma_m_plus_1 / sigma_m) * alpha_m
    ) * ce_m
    return x_new, x_new  # carry, output


_, traj = jax.lax.scan(_step_fn, X_noise, jnp.arange(L))  # traj[m] is at t_{m+1}


def _state_at_step(step: int) -> Array:
    if step == 0:
        return X_noise
    return traj[step - 1]


# Remaining 5 panels: denoising snapshots
for i, step in enumerate(steps_to_visualize):
    # Fill panels in row-major order, skipping (0,0) which is used by original data.
    panel_idx = i + 1  # 1..5
    row = panel_idx // 2
    col = panel_idx % 2
    ax = fig.add_subplot(gs[row, col])

    # Always show original data (for reference)
    ax.scatter(X[:, 0], X[:, 1], c=point_colors, s=marker_size, alpha=1.0)

    if step == 0:
        X_step = _state_at_step(step)
        ax.scatter(X_step[:, 0], X_step[:, 1], c="blue", s=marker_size, alpha=0.20)
        title = f"$\\ell = L = {L - step}$ | $t_\\ell = {ts[step].item():.1f}$"
    else:
        X_step = _state_at_step(step)
        ax.scatter(
            X_step[:, 0],
            X_step[:, 1],
            c="blue",
            s=marker_size,
            alpha=0.20,
        )
        title = f"$\\ell = {L - step}$ | $t_\\ell = {ts[step].item():.1f}$"

    ax.set_title(title, fontsize=18, pad=4)
    ax.set_xlim(-axis_lim, axis_lim)
    ax.set_ylim(-axis_lim, axis_lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

plt.tight_layout()
plt.savefig("memorizing_score_denoising_2d.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
plt.show()

