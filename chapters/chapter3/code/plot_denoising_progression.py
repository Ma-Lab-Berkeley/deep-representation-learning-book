import jax
import jax.numpy as jnp
from jax import Array
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

from diffusion.gmm import gmm_sample, gmm_ce
from diffusion.processes import DenoisingProcess

# jax.config.update("jax_debug_nans", True)  # Raises an error when NaNs are produced
# jax.config.update("jax_disable_jit", True)  # Optional: disables JIT for easier debugging



D = 3  # 3D data
K = 3  # 3 components
N = 100  # 100 ground truth samples
N_sample = 100  # 100 samples

seed = 4
key = jax.random.PRNGKey(seed)

# Create 3D data with two lines (rank 1) and one plane (rank 2)
mus = jnp.zeros((K, D))

scale_for_plotting = 3

# First component: a line (rank 1 covariance)
key, subkey = jax.random.split(key)
factor_1 = jax.random.normal(subkey, shape=(D, 1))
Sigma_1 = factor_1 @ factor_1.T * 2

# Second component: another line (rank 1 covariance)
key, subkey = jax.random.split(key)
factor_2 = jax.random.normal(subkey, shape=(D, 1))
Sigma_2 = factor_2 @ factor_2.T * 2

# Third component: a plane (rank 2 covariance)
key, subkey = jax.random.split(key)
factor_3 = jax.random.normal(subkey, shape=(D, 2))
Sigma_3 = factor_3 @ factor_3.T

Sigmas = jnp.stack([Sigma_1, Sigma_2, Sigma_3], axis=0) * scale_for_plotting

pi = jnp.ones((K,)) / K

key, subkey = jax.random.split(key)
X, y = gmm_sample(subkey, N, D, K, pi, mus, Sigmas)

# Set up the denoising process
# alpha_fn = lambda t: jnp.ones_like(t)
alpha_fn = lambda t: jnp.sqrt(1 - t**2)
sigma_fn = lambda t: t
noising_process = DenoisingProcess(alpha_fn, sigma_fn)

# Set up the noise schedule
L = 500
t_min = 0.001
t_max = 0.999
ts = jnp.linspace(t_max, t_min, L+1)

# Generate pure noise
key, subkey = jax.random.split(key)
# X_noise = jax.random.normal(subkey, shape=(N_sample, D))
sk1, sk2 = jax.random.split(subkey)
# X_noise_seed, y_noise_seed = gmm_sample(sk1, N_sample, D, K, pi, mus, Sigmas)
assert N == N_sample
X_noise_seed = X
X_noise = alpha_fn(t_max) * X_noise_seed + sigma_fn(t_max) * jax.random.normal(sk2, shape=(N_sample, D))

# Define the conditional expectation function
def ce_func(x, t):
    return gmm_ce(x, alpha_fn(t), sigma_fn(t), pi, mus, Sigmas)

# Define the steps we want to visualize - original data, then steps 0, 460, 480, 490, 500
steps_to_visualize = [0, 300, 400, 450, 475, 500]

# Create a figure with a grid of subplots
fig = plt.figure(figsize=(15, 10))
fig.subplots_adjust(left=0.0, right=1.0, bottom=0.05, top=0.95)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.25)

# Create a color map for the components of the original data
colors = ['red', 'orange', 'gray']
color_map = {0: colors[0], 1: colors[1], 2: colors[2]}
point_colors = [color_map[int(label)] for label in y]

# Removed the separate plot for original data

# Plot each denoising step
for i, step in enumerate(steps_to_visualize):
    # Calculate the grid position for a 2x3 grid
    if i < 3:  # First three steps go in the top row
        row, col = 0, i
    else:      # Last three steps go in the bottom row
        row, col = 1, i - 3
    
    ax = fig.add_subplot(gs[row, col], projection='3d')
    
    # First plot the original data with their original colors (smaller and more transparent)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=point_colors, s=30, alpha=1, label='Original')
    
    if step == 0:
        # For step 0, show the pure noise
        ax.scatter(X_noise[:, 0], X_noise[:, 1], X_noise[:, 2], c='blue', s=30, alpha=0.1, label='Noisy')
        title = f"$\\ell = L = {L - step}$ | $t_\\ell = {ts[step].item():.2f}$"
    else:
        # For other steps, show the denoised state at that step
        X_step = noising_process.denoise(X_noise, ce_func, ts[:step+1])
        ax.scatter(X_step[:, 0], X_step[:, 1], X_step[:, 2], c='blue', s=30, alpha=0.1, label='Denoised')
        title = f"$\\ell = {L - step}$ | $t_\\ell = {ts[step].item():.2f}$"
    
    ax.set_title(title, fontsize=20)
    
    # Set consistent view angle for all plots
    ax.view_init(elev=30, azim=30)
    
    # Set axis limits to be consistent
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_zlim(-3, 3)
    
    # Add grid lines
    ax.grid(True)
    
    # Add legend to the first plot only to avoid clutter
    # if i == 0:
    #     ax.legend(loc='upper right', fontsize='small')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'denoising_progression_3d.png', dpi=300)
plt.show()
