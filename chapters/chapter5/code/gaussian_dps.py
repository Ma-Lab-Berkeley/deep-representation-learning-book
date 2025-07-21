import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.linalg as jsl
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os
from datetime import datetime
import jax.scipy.stats.multivariate_normal as mvn
from pathlib import Path

# --- Setup Results Directory ---
now = datetime.now()
timestamp = now.strftime("%Y%m%d-%H%M%S")
results_base_dir = Path(__file__).parent / "results"
results_dir = os.path.join(results_base_dir, timestamp)
os.makedirs(results_dir, exist_ok=True)
print(f"Saving results to: {results_dir}")

# --- Hyperparameters ---
SEED = 4
key = jr.PRNGKey(SEED)

D = 2  # Dimension of data x
d = 1  # Dimension of measurement y

# Data distribution parameters (Gaussian: N(mu, Sigma))
mu = jnp.zeros(D)
# Ellipsoidal covariance
Sigma = jnp.diag(jnp.array([1.0, 0.25]))

# Measurement model parameters (y = A x + sigma * w)
# Project onto the first coordinate
A = jnp.array([[1.0, 0.0]]) # Shape (d, D)
assert A.shape == (d, D)
sigma_noise = 0.1  # Measurement noise std dev

# Diffusion parameters
T = 1.0  # Max noise schedule time
L = 100  # Number of discretization steps
N_SAMPLES = 100 # Number of posterior samples to draw

# --- Time Discretization ---
ts = jnp.linspace(0, T, L + 1) # t_0, t_1, ..., t_L
ts_rev = ts[::-1] # T = t_L, ..., t_1

# --- Utility Functions ---
def vec(x):
    return x.reshape(-1, 1)

def mat(x):
    return x.reshape(D, D)

print(f"Setting: D={D}, d={d}")
print(f"Mu:\n{mu}")
print(f"Sigma:\n{Sigma}")
print(f"A:\n{A}")
print(f"sigma_noise: {sigma_noise}")
print(f"T: {T}, L: {L}")


# --- Gaussian Case Closed-Form Expressions ---

@jax.jit
def E_x_given_xt(xi, t, mu, Sigma):
    """Computes E[x | x_t=xi] using Eq. (10)."""
    # E[x | x_t=xi] = mu + Sigma @ inv(Sigma + t^2*I) @ (xi - mu)
    D = mu.shape[0]
    Sigma_t_inv = jsl.solve(Sigma + t**2 * jnp.eye(D), Sigma, assume_a='pos') # More stable than inv
    # Sigma_t_inv = jnp.linalg.inv(Sigma + t^2 * jnp.eye(D)) @ Sigma # Alternative
    return mu + Sigma_t_inv @ (xi - mu)

@jax.jit
def Sigma_y_given_xt(t, Sigma, A, sigma_noise):
    """Computes Sigma_{y|xt}."""
    # Sigma_{y|xt} = A Sigma A^T + sigma^2 I - A Sigma inv(Sigma + t^2 I) Sigma A^T
    D = Sigma.shape[0]
    d = A.shape[0]
    I_d = jnp.eye(d)
    I_D = jnp.eye(D)

    Sigma_plus_t2I = Sigma + t**2 * I_D
    term1 = A @ Sigma @ A.T + sigma_noise**2 * I_d
    # Use solve: inv(Sigma + t^2 I) @ Sigma = solve(Sigma + t^2 I, Sigma)
    term2 = A @ jsl.solve(Sigma_plus_t2I, Sigma, assume_a='pos') @ Sigma @ A.T
    # Alternative with inv:
    # term2 = A @ jnp.linalg.inv(Sigma_plus_t2I) @ Sigma @ Sigma @ A.T
    return term1 - term2


@jax.jit
def grad_log_p_y_given_xt(xi, nu, t, mu, Sigma, A, sigma_noise):
    """Computes the exact measurement matching term t^2 * grad_xi log p(y|xt). Eq. (13)."""
    # t^2 * (Sigma+t^2 I)^-1 @ Sigma @ A^T @ Sigma_{y|xt}^-1 @ (nu - A @ E[x | xt=xi])
    D = mu.shape[0]
    d = A.shape[0]

    E_x_xt = E_x_given_xt(xi, t, mu, Sigma)
    Sigma_y_xt = Sigma_y_given_xt(t, Sigma, A, sigma_noise)

    # Precompute terms
    Sigma_plus_t2I_inv_Sigma = jsl.solve(Sigma + t**2 * jnp.eye(D), Sigma, assume_a='pos')
    Sigma_y_xt_inv = jnp.linalg.inv(Sigma_y_xt) # Okay for small d

    term1 = t**2 * Sigma_plus_t2I_inv_Sigma @ A.T
    term2 = Sigma_y_xt_inv @ (nu - A @ E_x_xt)

    return term1 @ term2

@jax.jit
def E_x_given_xt_y(xi, nu, t, mu, Sigma, A, sigma_noise):
    """Computes the exact conditional denoiser E[x | xt=xi, y=nu]. Eq. (8)."""
    unconditional_term = E_x_given_xt(xi, t, mu, Sigma)
    measurement_term = grad_log_p_y_given_xt(xi, nu, t, mu, Sigma, A, sigma_noise)
    return unconditional_term + measurement_term

# --- Approximate Measurement Matching Term (DPS style) ---

def log_p_y_given_x(x, nu, A, sigma_noise):
    """Computes log N(nu | Ax, sigma_noise^2 I)."""
    d = nu.shape[0]
    diff = nu - A @ x
    log_det_term = -0.5 * d * jnp.log(2 * jnp.pi * sigma_noise**2)
    exponent_term = -0.5 * jnp.sum(diff**2) / sigma_noise**2
    return log_det_term + exponent_term

@jax.jit
def approx_grad_log_p_y_given_xt(xi, nu, t, mu, Sigma, A, sigma_noise):
    """Computes the approximate measurement matching term using Eq. (16)."""
    # t^2 * grad_xi [ log p(y | x) (nu | E[x | xt=xi]) ]

    # Define a function whose gradient w.r.t. xi we need
    def loss_fn(current_xi):
        E_x_xt = E_x_given_xt(current_xi, t, mu, Sigma)
        return log_p_y_given_x(E_x_xt, nu, A, sigma_noise)

    grad_val = jax.grad(loss_fn)(xi)
    return t**2 * grad_val

@jax.jit
def approx_E_x_given_xt_y(xi, nu, t, mu, Sigma, A, sigma_noise):
    """Computes the approximate conditional denoiser E[x | xt=xi, y=nu]."""
    unconditional_term = E_x_given_xt(xi, t, mu, Sigma)
    measurement_term = approx_grad_log_p_y_given_xt(xi, nu, t, mu, Sigma, A, sigma_noise)
    return unconditional_term + measurement_term


# --- True Posterior Calculation ---

@jax.jit
def mu_x_given_y(nu, mu, Sigma, A, sigma_noise):
    """Computes the true posterior mean mu_{x|y}."""
    # mu + Sigma @ A.T @ inv(A @ Sigma @ A.T + sigma^2 I) @ (nu - A @ mu)
    d = A.shape[0]
    term1 = A @ Sigma @ A.T + sigma_noise**2 * jnp.eye(d)
    term2 = nu - A @ mu
    correction = Sigma @ A.T @ jsl.solve(term1, term2, assume_a='pos')
    return mu + correction

@jax.jit
def Sigma_x_given_y(Sigma, A, sigma_noise):
    """Computes the true posterior covariance Sigma_{x|y}."""
    # Sigma - Sigma @ A.T @ inv(A @ Sigma @ A.T + sigma^2 I) @ A @ Sigma
    d = A.shape[0]
    term1 = A @ Sigma @ A.T + sigma_noise**2 * jnp.eye(d)
    term2 = A @ Sigma
    correction = Sigma @ A.T @ jsl.solve(term1, term2, assume_a='pos')
    return Sigma - correction


# --- DDIM-style Sampler ---

def ddim_sampler(key, nu, ts_rev, L, D, T, denoiser_func, mu, Sigma, A, sigma_noise):
    """Samples x_0 using the DDIM-like iteration Eq. (1)."""
    key, subkey = jr.split(key)
    # Initial sample from N(0, T^2 I)
    x_hat_t = T * jr.normal(subkey, (D,)) # Equivalent to sqrt(T^2)*normal

    # Store trajectory for visualization (optional)
    # traj = [x_hat_t]
    traj = [x_hat_t] # Store x_hat_tL

    # Iterate backwards from t_L to t_1
    # ts_rev contains [t_L, t_{L-1}, ..., t_1, t_0]
    # We iterate L times, using t_l and updating to t_{l-1}
    for l in tqdm(range(L, 0, -1), leave=False): # Removed desc for cleaner nested loops
        t_l = ts_rev[L-l] # Current time t_l = T*l/L
        assert jnp.isclose(t_l, T * l / L)

        # Apply the denoiser E[x | xt=x_hat_t, y=nu]
        x_bar_star = denoiser_func(x_hat_t, nu, t_l, mu, Sigma, A, sigma_noise)

        # Update using Eq. (1)
        x_hat_t = (1.0 - 1.0/l) * x_hat_t + (1.0/l) * x_bar_star
        # traj.append(x_hat_t)
        traj.append(x_hat_t) # Store x_hat_{t_{l-1}}

    # Final result is x_hat_0
    # return x_hat_t #, jnp.array(traj)
    # Return the full trajectory, reversing it to match [t0, t1, ..., tL]
    # This makes indexing easier later: traj_array[l] corresponds to t_l
    return jnp.stack(traj[::-1])


# --- Simulation ---
if __name__ == "__main__":
    # 1. Generate true data and measurement
    key, subkey1, subkey2 = jr.split(key, 3)
    x_true = jr.multivariate_normal(subkey1, mu, Sigma)
    noise_w = jr.normal(subkey2, (d,))
    y_obs = A @ x_true + sigma_noise * noise_w

    print("\n--- Simulation Data ---")
    print(f"x_true: {x_true}")
    print(f"y_obs (nu): {y_obs}")

    # 2. Calculate true posterior parameters
    true_posterior_mean = mu_x_given_y(y_obs, mu, Sigma, A, sigma_noise)
    true_posterior_cov = Sigma_x_given_y(Sigma, A, sigma_noise)

    print("\n--- True Posterior ---")
    print(f"True Posterior Mean: {true_posterior_mean}")
    print(f"True Posterior Covariance:\n{true_posterior_cov}")

    # 3. Run samplers
    print("\n--- Running Samplers ---")
    # key, subkey_exact, subkey_approx = jr.split(key, 3)
    key, subkey_samples = jr.split(key)
    sample_keys = jr.split(subkey_samples, N_SAMPLES)

    # samples_exact = []
    # samples_approx = []
    trajectories_exact = []
    trajectories_approx = []

    print(f"Generating {N_SAMPLES} trajectories for each method...")
    # Disable inner tqdm progress bar by replacing tqdm with range if desired
    # Or keep it for detailed progress
    for i in tqdm(range(N_SAMPLES), desc="Overall Sampling Progress"):
        key_i = sample_keys[i]
        key_exact_i, key_approx_i = jr.split(key_i)

        # Run with exact denoiser
        x_hat_exact_i = ddim_sampler(
            key_exact_i, y_obs, ts_rev, L, D, T,
            E_x_given_xt_y, # Exact denoiser
            mu, Sigma, A, sigma_noise
        )
        # samples_exact.append(x_hat_exact_i)
        trajectories_exact.append(x_hat_exact_i) # Store full trajectory

        # Run with approximate denoiser
        x_hat_approx_i = ddim_sampler(
            key_approx_i, y_obs, ts_rev, L, D, T,
            approx_E_x_given_xt_y, # Approximate denoiser
            mu, Sigma, A, sigma_noise
        )
        # samples_approx.append(x_hat_approx_i)
        trajectories_approx.append(x_hat_approx_i) # Store full trajectory

    # samples_exact = jnp.stack(samples_exact) # Shape (N_SAMPLES, D)
    # samples_approx = jnp.stack(samples_approx) # Shape (N_SAMPLES, D)
    trajectories_exact = jnp.stack(trajectories_exact)   # Shape (N_SAMPLES, L+1, D)
    trajectories_approx = jnp.stack(trajectories_approx) # Shape (N_SAMPLES, L+1, D)

    # Final samples are at index 0 (time t0)
    samples_exact = trajectories_exact[:, 0, :]   # Shape (N_SAMPLES, D)
    samples_approx = trajectories_approx[:, 0, :] # Shape (N_SAMPLES, D)

    # 4. Output Results (using final samples)
    print("\n--- Results (Final Samples t=0) ---")
    # print(f"Sample (Exact Denoiser):   {x_hat_exact}")
    # print(f"Sample (Approx Denoiser):  {x_hat_approx}")
    # Calculate sample means
    mean_exact = jnp.mean(samples_exact, axis=0)
    mean_approx = jnp.mean(samples_approx, axis=0)
    print(f"Sample Mean (Exact Denoiser):  {mean_exact}")
    print(f"Sample Mean (Approx Denoiser): {mean_approx}")
    print(f"True Posterior Mean:           {true_posterior_mean}")

    # error_exact = jnp.linalg.norm(x_hat_exact - true_posterior_mean)
    # error_approx = jnp.linalg.norm(x_hat_approx - true_posterior_mean)
    # Calculate error of sample means
    error_mean_exact = jnp.linalg.norm(mean_exact - true_posterior_mean)
    error_mean_approx = jnp.linalg.norm(mean_approx - true_posterior_mean)
    print(f"\nL2 Error of Sample Mean vs True Mean (Exact):   {error_mean_exact:.4f}")
    print(f"L2 Error of Sample Mean vs True Mean (Approx):  {error_mean_approx:.4f}")

    # Calculate average L2 error of individual samples (optional, but good)
    avg_error_exact = jnp.mean(jax.vmap(lambda x: jnp.linalg.norm(x - true_posterior_mean))(samples_exact))
    avg_error_approx = jnp.mean(jax.vmap(lambda x: jnp.linalg.norm(x - true_posterior_mean))(samples_approx))
    print(f"Avg L2 Error of Samples vs True Mean (Exact):   {avg_error_exact:.4f}")
    print(f"Avg L2 Error of Samples vs True Mean (Approx):  {avg_error_approx:.4f}")


    # 5. Visualization (Modified to save intermediate plots)
    print("\n--- Plotting Trajectories ---")

    # Define fixed plot limits
    xlim = [-2, 2]
    ylim = [-1.25, 1.25]

    # Ensure matplotlib.patches.Ellipse is imported if needed
    from matplotlib.patches import Ellipse
    def plot_ellipse(ax, mean, cov, color, label=None, alpha=0.3, n_std=2, linewidth=1.5):
        """Plots an ellipse representing the covariance matrix."""
        vals, vecs = jnp.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]

        # Avoid sqrt of negative eigenvalues (numerical instability)
        safe_vals = jnp.maximum(vals, 1e-9)
        theta = jnp.arctan2(*vecs[:, 0][::-1])
        width, height = 2 * n_std * jnp.sqrt(safe_vals)

        ell = Ellipse(xy=mean, width=width, height=height, angle=jnp.degrees(theta),
                      edgecolor=color, facecolor='none', linestyle='--', label=label, linewidth=linewidth)
        ax.add_patch(ell)
        ell_fill = Ellipse(xy=mean, width=width, height=height, angle=jnp.degrees(theta),
                           color=color, alpha=alpha)
        ax.add_patch(ell_fill)

    def plot_snapshot(filename, l, t_l, samples_exact_tl, samples_approx_tl,
                      mu, Sigma, true_posterior_mean, true_posterior_cov, x_true, y_obs,
                      xlim, ylim, grid_points, xx, yy,
                      show_legend=False): # Default legend off
        """Generates and saves a plot for a single timestep."""
        fig, ax = plt.subplots(1, 1, figsize=(5, 4)) # Adjusted figsize slightly

        # Calculate p(xt | y) = N(mu_{x|y}, Sigma_{x|y} + t^2 I)
        t_sq = t_l**2 + 1e-9
        Sigma_xt_given_y = true_posterior_cov + t_sq * jnp.eye(D)

        # Evaluate PDF on grid
        Z = mvn.pdf(grid_points, mean=true_posterior_mean, cov=Sigma_xt_given_y)
        contour_levels = 10

        # Plot contour map first (underneath)
        ax.contourf(xx, yy, Z.reshape(xx.shape), levels=contour_levels, cmap='Greens', alpha=0.4, zorder=1)
        ax.contour(xx, yy, Z.reshape(xx.shape), levels=contour_levels, colors='green', alpha=0.5, linewidths=1.0, zorder=2)

        # --- Plot elements (add labels conditionally) ---
        prior_label = 'Prior Cov (2 std)' if show_legend else None
        plot_ellipse(ax, mu, Sigma, 'gray', label=prior_label, alpha=0.1, linewidth=2.0)

        post_cov_label = 'True Posterior p(x|y) Cov (2 std)' if show_legend else None
        plot_ellipse(ax, true_posterior_mean, true_posterior_cov, 'blue', label=post_cov_label, alpha=0.1, linewidth=2.0)

        obs_label = 'Observation y' if show_legend else None
        ax.scatter(y_obs[0], 0, c='black', marker='o', s=120, label=obs_label, zorder=3, facecolors='none', edgecolors='black', linewidths=1.5)

        prior_mean_label = 'Prior Mean' if show_legend else None
        # ax.scatter(mu[0], mu[1], c='gray', marker='s', s=130, label=prior_mean_label, zorder=4)

        signal_label = 'Signal x' if show_legend else None
        ax.scatter(x_true[0], x_true[1], c='black', marker='*', s=120, label=signal_label, zorder=5)

        post_mean_label = 'True Posterior p(x|y) Mean' if show_legend else None
        ax.scatter(true_posterior_mean[0], true_posterior_mean[1], c='blue', marker='x', s=130, label=post_mean_label, zorder=4)

        exact_label = f'Samples (Exact, N={N_SAMPLES})' if show_legend else None
        ax.scatter(samples_exact_tl[:, 0], samples_exact_tl[:, 1], c='cyan', marker='o', s=25, alpha=0.6, label=exact_label, zorder=6, edgecolors='blue')

        approx_label = f'Samples (Approx, N={N_SAMPLES})' if show_legend else None
        ax.scatter(samples_approx_tl[:, 0], samples_approx_tl[:, 1], c='red', marker='^', s=25, alpha=0.6, label=approx_label, zorder=7, edgecolors='darkred')
        # --- End Plot elements ---

        # Apply fixed plot limits
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])

        ax.set_xlabel("x_1", fontsize=12)
        ax.set_ylabel("x_2", fontsize=12)
        ax.set_title(f"t_{l} = {t_l:.3f}", fontsize=12)

        if show_legend:
            ax.legend(fontsize=10, loc='best')

        ax.grid(True, linestyle=':', alpha=0.5)
        # ax.set_aspect('equal', adjustable='box') # Removed aspect
        # plt.tight_layout() # Removed tight_layout

        plt.tight_layout(pad=0.1) # Re-add tight_layout with padding
        plt.savefig(filename, dpi=150)
        plt.close(fig)

    # Loop through time steps (L down to 0) and create plots
    # ts contains [t0, t1, ..., tL]
    # trajectories contain data corresponding to [t0, t1, ..., tL]
    # Grid for contour plot
    grid_res = 100
    x_grid = jnp.linspace(xlim[0], xlim[1], grid_res)
    y_grid = jnp.linspace(ylim[0], ylim[1], grid_res)
    xx, yy = jnp.meshgrid(x_grid, y_grid)
    grid_points = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)

    for l in tqdm(range(L, -1, -1), desc="Generating Plots"): # L, L-1, ..., 1, 0
        t_l = ts[l]
        samples_exact_tl = trajectories_exact[:, l, :]
        samples_approx_tl = trajectories_approx[:, l, :]

        # Call the plotting function
        plot_filename = os.path.join(results_dir, f"samples_step_{L-l:03d}_t_{l}.png")
        plot_snapshot(plot_filename,
                      l, t_l, samples_exact_tl, samples_approx_tl,
                      mu, Sigma, true_posterior_mean, true_posterior_cov, x_true, y_obs,
                      xlim, ylim, grid_points, xx, yy,
                      show_legend=False) # Legend off by default

    # print("Plot saved to gaussian_diffusion_sim_multi_sample.png")
    print(f"Finished plotting. Plots saved in {results_dir}")
    # plt.show() # Uncomment to display plot interactively 