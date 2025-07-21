import jax
import jax.numpy as jnp
from jax import Array

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

from diffusion.gmm import gmm_sample, gmm_ce
from diffusion.processes import DenoisingProcess
from pathlib import Path

# Set up dimensions and parameters
D = 3  # 3D data
K = 3  # 3 components
N = 100  # 100 ground truth samples

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

# Sample from the GMM
key, subkey = jax.random.split(key)
X, y = gmm_sample(subkey, N, D, K, pi, mus, Sigmas)

# Set up the denoising process
alpha_fn = lambda t: jnp.ones_like(t)
sigma_fn = lambda t: t
noising_process = DenoisingProcess(alpha_fn, sigma_fn)

# Set t_max to approximately 10 as requested
t_max = 10.0

# Generate two different noise vectors
key, subkey1, subkey2 = jax.random.split(key, 3)
noise1 = jax.random.normal(subkey1, shape=(N, D))
noise2 = jax.random.normal(subkey2, shape=(N, D))

# Create the three different versions of data
original_data = X
data_with_alpha_and_noise = alpha_fn(t_max) * X + sigma_fn(t_max) * noise1
data_with_noise_only = sigma_fn(t_max) * noise2

# Create a figure with a grid of 2 subplots
fig = plt.figure(figsize=(16, 7))

# Create a color map for the components of the original data
colors = ['red', 'orange', 'gray']
color_map = {0: colors[0], 1: colors[1], 2: colors[2]}
point_colors = [color_map[int(label)] for label in y]

# Plot 1: Original Data with closer view
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.scatter(original_data[:, 0], original_data[:, 1], original_data[:, 2], c=point_colors, s=40, alpha=1)
ax1.set_title(f"Original Data", fontsize=32)
ax1.view_init(elev=30, azim=30)
# Set tighter limits for a closer view
# ax1.set_xlim(-8, 8)
# ax1.set_ylim(-8, 8)
# ax1.set_zlim(-8, 8)
ax1.grid(True)

# Plot 2: Combined plot with both noise types
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
# First plot original data (smaller and more transparent)
ax2.scatter(original_data[:, 0], original_data[:, 1], original_data[:, 2], c=point_colors, s=20, alpha=0.4)#, label='Original')
# Plot data with alpha and noise
ax2.scatter(data_with_alpha_and_noise[:, 0], data_with_alpha_and_noise[:, 1], data_with_alpha_and_noise[:, 2], 
           c='blue', s=30, alpha=0.7, label='$x_{T}$')
# Plot data with noise only
ax2.scatter(data_with_noise_only[:, 0], data_with_noise_only[:, 1], data_with_noise_only[:, 2], 
           c='green', s=30, alpha=0.7, label='$\mathcal{N}(0, T^{2}I)$')
ax2.set_title(f"Noise Approximations ($T = {t_max}$)", fontsize=32)
ax2.view_init(elev=30, azim=30)
# ax2.set_xlim(-40, 40)
# ax2.set_ylim(-40, 40)
# ax2.set_zlim(-40, 40)
ax2.grid(True)
ax2.legend(loc='upper right', fontsize=28)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'two_plots_comparison.png', dpi=300)
plt.show()
