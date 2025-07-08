import jax
import jax.numpy as jnp
from jax import Array

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from pathlib import Path

from .gmm_jax import gmm_sample, gmm_ce
from .processes_jax import DenoisingProcess

# jax.config.update("jax_debug_nans", True)  # Raises an error when NaNs are produced
# jax.config.update("jax_disable_jit", True)  # Optional: disables JIT for easier debugging



D = 3  # 3D data
K = 3  # 3 components
N = 100  # 100 ground truth samples
N_sample = 200  # 20 samples

seed = 4
key = jax.random.PRNGKey(seed)

# Create 3D data with two lines (rank 1) and one plane (rank 2)
mus = jnp.zeros((K, D))

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

Sigmas = jnp.stack([Sigma_1, Sigma_2, Sigma_3], axis=0)

pi = jnp.ones((K,)) / K

key, subkey = jax.random.split(key)
X, y = gmm_sample(subkey, N, D, K, pi, mus, Sigmas)

# Set up the denoising process
alpha_fn = lambda t: jnp.sqrt(1 - t**2)
sigma_fn = lambda t: t
noising_process = DenoisingProcess(alpha_fn, sigma_fn)

# Generate pure noise
key, subkey = jax.random.split(key)
X_noise = jax.random.normal(subkey, shape=(N_sample, D))

# Set up the noise schedule
M = 500
t_min = 0.0
t_max = 0.999
ts = jnp.linspace(t_max, t_min, M+1)
print(ts)

# Define the conditional expectation function
def ce_func(x, t):
    return gmm_ce(x, alpha_fn(t), sigma_fn(t), pi, mus, Sigmas)

# Define the steps we want to visualize - original data, then steps 0, 460, 480, 490, 500
steps_to_visualize = [0, 200, 300, 400, 500]

# Create a figure with a grid of subplots
fig = plt.figure(figsize=(15, 10))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.25)

# Create a color map for the components of the original data
colors = ['red', 'orange', 'gray']
color_map = {0: colors[0], 1: colors[1], 2: colors[2]}
point_colors = [color_map[int(label)] for label in y]

# Plot the original data in the top left
ax_orig = fig.add_subplot(gs[0, 0], projection='3d')
ax_orig.scatter(X[:, 0], X[:, 1], X[:, 2], c=point_colors, s=30, alpha=1)
ax_orig.set_title("Original Data | M = 500")
ax_orig.view_init(elev=30, azim=30)
ax_orig.set_xlim(-6, 6)
ax_orig.set_ylim(-6, 6)
ax_orig.set_zlim(-3, 3)
ax_orig.grid(True)

# Plot each denoising step
for i, step in enumerate(steps_to_visualize):
    # Calculate the grid position (skip the first cell which has the original data)
    if i < 2:  # First two steps go in the top row
        row, col = 0, i + 1  # +1 to skip the first cell which has original data
    else:      # Last three steps go in the bottom row
        row, col = 1, i - 2  # -2 to start from column 0 in the second row
    
    ax = fig.add_subplot(gs[row, col], projection='3d')
    
    # First plot the original data with their original colors (smaller and more transparent)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=point_colors, s=15, alpha=1, label='Original')
    
    if step == 0:
        # For step 0, show the pure noise
        ax.scatter(X_noise[:, 0], X_noise[:, 1], X_noise[:, 2], c='blue', s=25, alpha=0.1, label='Noisy')
        title = f"m = 0 | M = {M}"
    else:
        # For other steps, show the denoised state at that step
        X_step = noising_process.denoise(X_noise, ce_func, ts[:step+1])
        ax.scatter(X_step[:, 0], X_step[:, 1], X_step[:, 2], c='blue', s=25, alpha=0.1, label='Denoised')
        title = f"m = {step} | M = {M}"
    
    ax.set_title(title)
    
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
