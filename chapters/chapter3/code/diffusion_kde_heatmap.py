import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

from processes import DenoisingProcess
from gmm import gmm_sample, gmm_ce

# Set random seed for reproducibility
seed = 42
key = jax.random.PRNGKey(seed)

# Parameters for the Gaussian Mixture Model
D = 2  # 2D data for visualization
K = 3  # 3 components
N = 100  # Number of samples for ground truth
grid_size = 100  # Grid size for density visualization

# Create 2D data with three components
mus = jnp.array([
    [-3.0, -3.0],  # Component 1
    [3.0, 3.0],    # Component 2
    [0.0, 3.0]     # Component 3
])

# Create covariance matrices with different shapes
key, subkey = jax.random.split(key)
factor_1 = jnp.array([[2.0, 0.5], [0.5, 0.5]])
Sigma_1 = factor_1 @ factor_1.T

key, subkey = jax.random.split(key)
factor_2 = jnp.array([[0.5, 0.0], [0.0, 2.0]])
Sigma_2 = factor_2 @ factor_2.T

key, subkey = jax.random.split(key)
factor_3 = jnp.array([[1.0, -0.8], [-0.8, 1.0]])
Sigma_3 = factor_3 @ factor_3.T

Sigmas = jnp.stack([Sigma_1, Sigma_2, Sigma_3], axis=0)

# Equal mixture weights
pi = jnp.ones((K,)) / K

# Sample from GMM for ground truth
key, subkey = jax.random.split(key)
X, y = gmm_sample(subkey, N, D, K, pi, mus, Sigmas)

# Set up the diffusion process
def alpha_fn(t):
    return jnp.ones_like(t)

def sigma_fn(t):
    return t

noising_process = DenoisingProcess(alpha_fn, sigma_fn)

# Define the time points for the forward process
t_values = [0.0, 1.0, 2.0, 3.0, 5.0, 10.0]

# Create a grid for density visualization
x_min, x_max = -8, 8
y_min, y_max = -8, 8
x_grid = np.linspace(x_min, x_max, grid_size)
y_grid = np.linspace(y_min, y_max, grid_size)
xx, yy = np.meshgrid(x_grid, y_grid)
grid_points = np.stack([xx.flatten(), yy.flatten()], axis=1)

# Function to compute the GMM density at time t
def compute_gmm_density(grid_points, pi, mus, Sigmas, alpha_t, sigma_t):
    """
    Compute the GMM density at time t using the formula:
    p(x, t) = sum_k pi_k * N(x | alpha_t * mu_k, alpha_t^2 * Sigma_k + sigma_t^2 * I)
    
    Args:
        grid_points: Points where to evaluate the density, shape (N, D)
        pi: Mixture weights, shape (K,)
        mus: Component means, shape (K, D)
        Sigmas: Component covariances, shape (K, D, D)
        alpha_t: Scaling factor at time t
        sigma_t: Noise level at time t
        
    Returns:
        Density values at grid_points, shape (N,)
    """
    N, D = grid_points.shape
    K = len(pi)
    density = np.zeros(N)
    
    # Identity matrix for noise covariance
    identity_matrix = np.eye(D)
    
    # Compute density for each component and sum
    for k in range(K):
        # Compute mean and covariance at time t
        mean_t = alpha_t * mus[k]
        cov_t = alpha_t**2 * Sigmas[k] + sigma_t**2 * identity_matrix
        
        # Convert to numpy for scipy
        mean_t_np = np.array(mean_t)
        cov_t_np = np.array(cov_t)
        
        # Compute component density
        component_density = multivariate_normal.pdf(
            grid_points, mean=mean_t_np, cov=cov_t_np
        )
        
        # Add weighted component to total density
        density += pi[k] * component_density
    
    return density

# Create figure for visualization with dynamic width based on number of time points
fig, axes = plt.subplots(1, len(t_values), figsize=(5 * len(t_values), 3), constrained_layout=True)

# Generate noisy samples for each time step
samples_at_t = []
for t in t_values:
    key, subkey = jax.random.split(key)
    # Use add_noise to apply the forward process
    if t == 0.0:
        # For t=0, use the original data
        noisy_samples = X
    else:
        # For t>0, add noise according to the diffusion process
        noisy_samples = noising_process.add_noise(subkey, X, jnp.array(t))
    
    samples_at_t.append(np.array(noisy_samples))

# Plot density for each time step
for i, t in enumerate(t_values):
    # Compute alpha_t and sigma_t
    alpha_t = alpha_fn(jnp.array(t))
    sigma_t = sigma_fn(jnp.array(t))
    
    # Compute GMM density at time t
    density = compute_gmm_density(
        grid_points, 
        np.array(pi), 
        np.array(mus), 
        np.array(Sigmas), 
        float(alpha_t), 
        float(sigma_t)
    )
    
    # Reshape density to grid shape
    density_grid = density.reshape(grid_size, grid_size)
    
    # Plot density
    axes[i].imshow(
        density_grid, 
        extent=[x_min, x_max, y_min, y_max],
        origin='lower', 
        cmap='Blues',
        vmin=0.00,
        vmax=0.01
    )
    
    # Plot samples
    samples = samples_at_t[i]
    axes[i].scatter(samples[:, 0], samples[:, 1], c='red', s=10, alpha=0.5)
    
    # Set title and labels
    axes[i].set_title(f"$t={t}$", fontsize=32)
    axes[i].set_xlabel('$x_{1}$', fontsize=20)
    axes[i].set_ylabel('$x_{2}$', fontsize=20)
    
    # Ensure all plots have the same aspect ratio and limits
    axes[i].set_aspect('equal')
    axes[i].set_xlim(x_min, x_max)
    axes[i].set_ylim(y_min, y_max)

# Save and show the figure
plt.savefig('forward_diffusion_density.png', dpi=300, bbox_inches='tight')
plt.show()

# Add a new function to plot denoised estimates with arrows
def plot_denoised_estimates():
    """
    Create a plot showing the denoised estimates of noisy points with arrows
    connecting each noisy point to its denoised estimate.
    Uses a small number of points (N=10) for clarity.
    """
    # Use a smaller number of samples for clarity
    N_small = 10
    
    # Sample from GMM for ground truth (small sample)
    key_small = jax.random.PRNGKey(seed + 1)  # Use a different seed
    key_small, subkey = jax.random.split(key_small)
    X_small, _ = gmm_sample(subkey, N_small, D, K, pi, mus, Sigmas)
    
    # Create figure for visualization with dynamic width based on number of time points
    fig, axes = plt.subplots(1, len(t_values), figsize=(5 * len(t_values), 3), constrained_layout=True)
    
    # Generate noisy samples and their denoised estimates for each time step
    for i, t in enumerate(t_values):
        if t == 0.0:
            # For t=0, use the original data (no noise)
            noisy_samples = X_small
            denoised_samples = X_small  # At t=0, denoised = original
        else:
            # For t>0, add noise according to the diffusion process
            key_small, subkey = jax.random.split(key_small)
            noisy_samples = noising_process.add_noise(subkey, X_small, jnp.array(t))
            
            # Compute denoised estimates using gmm_ce
            alpha_t = alpha_fn(jnp.array(t))
            sigma_t = sigma_fn(jnp.array(t))
            denoised_samples = gmm_ce(noisy_samples, alpha_t, sigma_t, pi, mus, Sigmas)
        
        # Compute alpha_t and sigma_t for density
        alpha_t = alpha_fn(jnp.array(t))
        sigma_t = sigma_fn(jnp.array(t))
        
        # Compute GMM density at time t
        density = compute_gmm_density(
            grid_points, 
            np.array(pi), 
            np.array(mus), 
            np.array(Sigmas), 
            float(alpha_t), 
            float(sigma_t)
        )
        
        # Reshape density to grid shape
        density_grid = density.reshape(grid_size, grid_size)
        
        # Plot density with the same settings as the first plot
        axes[i].imshow(
            density_grid, 
            extent=[x_min, x_max, y_min, y_max],
            origin='lower', 
            cmap='Blues',
            vmin=0.00,
            vmax=0.01
        )
        
        # Convert to numpy arrays
        noisy_np = np.array(noisy_samples)
        denoised_np = np.array(denoised_samples)
        
        # Plot noisy points in blue
        axes[i].scatter(noisy_np[:, 0], noisy_np[:, 1], c='red', s=80, alpha=0.7, label='Noisy')
        
        # Plot denoised points in brown
        axes[i].scatter(denoised_np[:, 0], denoised_np[:, 1], c='green', s=80, alpha=0.7, label='Denoised')
        
        # Draw arrows from noisy to denoised points
        for j in range(N_small):
            # Only draw arrows when t > 0 (no arrows needed at t=0)
            if t > 0:
                axes[i].arrow(
                    noisy_np[j, 0], noisy_np[j, 1],
                    denoised_np[j, 0] - noisy_np[j, 0],
                    denoised_np[j, 1] - noisy_np[j, 1],
                    head_width=0.2, head_length=0.3, fc='black', ec='black', linewidth=2, alpha=0.5
                )
        
        # Set title and labels using the same format as the first plot
        axes[i].set_title(f"$t={t}$", fontsize=32)
        axes[i].set_xlabel('$x_{1}$', fontsize=20)
        axes[i].set_ylabel('$x_{2}$', fontsize=20)
        
        # Add legend only to the first plot
        if i == 0:
            axes[i].legend(loc='lower right', fontsize=16)
        
        # Ensure all plots have the same aspect ratio and limits
        axes[i].set_aspect('equal')
        axes[i].set_xlim(x_min, x_max)
        axes[i].set_ylim(y_min, y_max)
    
    # Save and show the figure
    plt.savefig('denoised_estimates_with_arrows.png', dpi=300, bbox_inches='tight')
    plt.show()

# Run the denoised estimates plot
plot_denoised_estimates()
