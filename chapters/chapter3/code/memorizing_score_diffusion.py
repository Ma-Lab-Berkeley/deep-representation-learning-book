import jax
import jax.numpy as jnp
from jax import Array, vmap

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

from gmm import gmm_sample
from processes import DenoisingProcess

# Set up dimensions and parameters
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
sk1, sk2 = jax.random.split(subkey)
X_noise_seed = X
X_noise = alpha_fn(t_max) * X_noise_seed + sigma_fn(t_max) * jax.random.normal(sk2, shape=(N_sample, D))

# Define the memorizing score function
# This creates a GMM where each component corresponds to a training sample
# with zero covariance (i.e., a Dirac delta)
def memorizing_score_ce(X_noisy: Array, alpha: Array, sigma: Array, X_train: Array) -> Array:
    """
    Compute the conditional expectation using a "memorizing score" based on a GMM
    where each component corresponds to a training sample with zero covariance.
    
    Args:
        X_noisy (Array): Noisy input data of shape (N, D)
        alpha (Array): Scale factor for the clean data
        sigma (Array): Scale factor for the noise
        X_train (Array): Training data of shape (M, D)
    
    Returns:
        Array: Denoised data of shape (N, D)
    """
    N, D = X_noisy.shape
    M = X_train.shape[0]
    
    # For each noisy point, compute squared distance to each training point
    # Reshape for broadcasting: X_noisy (N, 1, D), X_train (1, M, D)
    X_noisy_expanded = X_noisy[:, None, :]  # (N, 1, D)
    X_train_expanded = X_train[None, :, :]  # (1, M, D)
    
    # Compute squared distances: |x_noisy - alpha * x_train|^2
    # This is proportional to the negative log likelihood under a GMM with zero covariance
    sq_distances = jnp.sum((X_noisy_expanded - alpha * X_train_expanded) ** 2, axis=-1)  # (N, M)
    
    # Compute weights using softmax to normalize (with temperature parameter sigma^2)
    # This approximates the posterior probabilities in the limit of zero covariance
    weights = jax.nn.softmax(-sq_distances / (2 * sigma**2), axis=-1)  # (N, M)
    
    # Compute the conditional expectation as a weighted sum of training points
    return jnp.sum(weights[:, :, None] * X_train_expanded, axis=1)  # (N, D)

# Define the conditional expectation function using our memorizing score
def ce_func(x, t):
    return memorizing_score_ce(x, alpha_fn(t), sigma_fn(t), X)

# Define the steps we want to visualize - original data, then steps 0, 200, 300, 400, 500
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
ax_orig.set_title("Original Data", fontsize=20)
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
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=point_colors, s=30, alpha=1, label='Original')
    
    if step == 0:
        # For step 0, show the pure noise
        ax.scatter(X_noise[:, 0], X_noise[:, 1], X_noise[:, 2], c='blue', s=30, alpha=0.1, label='Noisy')
        title = f"$\\ell = L = {L - step}$ | $t_\\ell = {ts[step].item():.1f}$"
    else:
        # For other steps, show the denoised state at that step
        X_step = noising_process.denoise(X_noise, ce_func, ts[:step+1])
        ax.scatter(X_step[:, 0], X_step[:, 1], X_step[:, 2], c='blue', s=30, alpha=0.1, label='Denoised')
        title = f"$\\ell = {L - step}$ | $t_\\ell = {ts[step].item():.1f}$"
    
    ax.set_title(title, fontsize=20)
    
    # Set consistent view angle for all plots
    ax.view_init(elev=30, azim=30)
    
    # Set axis limits to be consistent
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_zlim(-3, 3)
    
    # Add grid lines
    ax.grid(True)

plt.tight_layout()
plt.savefig('memorizing_score_denoising_3d.png', dpi=300)
plt.show()
