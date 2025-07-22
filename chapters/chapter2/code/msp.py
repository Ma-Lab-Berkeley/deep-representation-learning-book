"""
Implementation of the Matching, Stretching, and Projection (MSP) algorithm for orthogonal dictionary learning.

This implements the algorithm from [ZMZM20] which solves the optimization problem:
    min (1/4)||AX||_4^4  subject to  A^T A = I

The algorithm uses power iteration with orthogonal projection to find the optimal orthogonal dictionary.
"""

# Configure JAX to use double precision
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import random, jit, grad
from typing import Tuple, Optional, Union
from functools import partial
import matplotlib.pyplot as plt
import torch
import torchvision
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import time


def project_orthogonal(A: jnp.ndarray) -> jnp.ndarray:
    """Project a matrix onto the space of orthogonal matrices O(D) using SVD.
    
    Args:
        A: Input matrix of shape (D, D)
        
    Returns:
        Projected orthogonal matrix of shape (D, D)
    """
    U, _, Vh = jnp.linalg.svd(A, full_matrices=True)
    return U @ Vh


@partial(jax.jit, static_argnums=(3, 4))
def msp_iteration(A: jnp.ndarray, 
                 X: jnp.ndarray, 
                 key: jnp.ndarray,
                 max_iters: int = 100,
                 tol: float = 1e-6) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run the MSP algorithm to learn an orthogonal dictionary.
    
    Args:
        A: Initial orthogonal dictionary matrix of shape (D, D)
        X: Data matrix of shape (D, N) where N is number of samples
        key: JAX random key for reproducibility
        max_iters: Maximum number of iterations
        tol: Convergence tolerance on the Frobenius norm of successive iterates
        
    Returns:
        Tuple of:
            - Learned dictionary matrix transposed (atoms are rows)
            - Array of errors (Frobenius norm of successive iterates)
            - Array of objective values
    """
    def body_fun(A_prev, _):
        # MSP update step: A_{t+1} = P_O(D)[(A_t X)^⊙3 X^T]
        AX = A_prev @ X
        AX_cubed = AX ** 3  # Element-wise cube
        update = AX_cubed @ X.T
        A_next = project_orthogonal(update)
        
        # Compute error and objective value
        error = jnp.linalg.norm(A_next - A_prev, ord='fro')
        obj_value = 0.25 * jnp.sum(jnp.power(A_next @ X, 4))
        
        return A_next, (error, obj_value)

    A_final, (errors, obj_values) = jax.lax.scan(
        body_fun, A, None, length=max_iters)
    
    return A_final.T, errors, obj_values


def initialize_orthogonal(D: int, key: jnp.ndarray) -> jnp.ndarray:
    """Initialize a random orthogonal matrix using QR decomposition.
    
    Args:
        D: Dimension of the square matrix
        key: JAX random key
        
    Returns:
        Random orthogonal matrix of shape (D, D)
    """
    key1, key2 = random.split(key)
    A = random.normal(key1, (D, D))
    Q, _ = jnp.linalg.qr(A)
    return Q


def load_mnist_data(num_samples: int = 1000, key: Optional[jnp.ndarray] = None, train_split_ratio: float = 1.0) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    """Load MNIST digits and convert to normalized column vectors.
    
    Args:
        num_samples: Total number of digits to load.
        key: JAX random key for shuffling and splitting. Required if train_split_ratio < 1.0.
        train_split_ratio: Fraction of data to use for training (0.0 to 1.0). 
                           If 1.0, all data is returned as a single array.
                           If < 1.0, returns a tuple (X_train, X_test).
        
    Returns:
        If train_split_ratio == 1.0: Matrix of shape (784, num_samples).
        If train_split_ratio < 1.0: Tuple of (X_train, X_test) with shapes 
                                     (784, N_train) and (784, N_test).
    """
    # Ensure key is provided if splitting is needed
    if train_split_ratio < 1.0 and key is None:
        raise ValueError("A JAX random key must be provided for train/test splitting.")

    # Create data directory if it doesn't exist
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)
    
    # Download and load MNIST
    mnist = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor()
    )
    
    # Convert to array and flatten images
    X = torch.stack([img.reshape(-1) for img, _ in mnist])  # Shape: (60000, 784)
    X = X.numpy()
    
    # Convert to JAX array
    X = jnp.array(X)
    
    # Randomly select num_samples if key provided
    if key is not None and num_samples < len(X):
        indices = random.choice(key, len(X), shape=(num_samples,), replace=False)
        X = X[indices]
    else:
        X = X[:num_samples]
    
    # Transpose to get (784, num_samples) and normalize columns
    X = X.T
    # X = X / jnp.linalg.norm(X, axis=0)

    # Split data if requested
    if train_split_ratio < 1.0:
        if key is None:
            raise ValueError("A JAX random key must be provided for train/test splitting.")
        num_total = X.shape[1]
        num_train = int(num_total * train_split_ratio)
        
        # Shuffle columns using the key
        indices = random.permutation(key, num_total)
        X_shuffled = X[:, indices]
        
        X_train = X_shuffled[:, :num_train]
        X_test = X_shuffled[:, num_train:]
        return X_train, X_test
    else:
        return X


def load_mnist_patches(key: jnp.ndarray, num_patches: int = 10000, patch_size: int = 8, train_split_ratio: float = 1.0, min_l1_threshold: Optional[float] = None) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    """Load MNIST digits and extract random patches, filtering low-intensity patches.

    Args:
        key: JAX random key for reproducibility and splitting.
        num_patches: Total number of valid random patches to extract.
        patch_size: The dimension (K) of the KxK patches.
        train_split_ratio: Fraction of data to use for training (0.0 to 1.0).
                           If 1.0, all data is returned as a single array.
                           If < 1.0, returns a tuple (X_train, X_test).
        min_l1_threshold: Minimum L1 norm (sum of pixel intensities) for a patch to be included.
                          If None, defaults to sqrt(patch_size).

    Returns:
        If train_split_ratio == 1.0: Matrix of shape (patch_size*patch_size, num_patches).
        If train_split_ratio < 1.0: Tuple of (X_train, X_test) with shapes
                                     (patch_size*patch_size, N_train) and (patch_size*patch_size, N_test).
    """
    # Set default L1 threshold if not provided
    if min_l1_threshold is None:
        min_l1_threshold = float(jnp.sqrt(patch_size))

    # Split the key for patch generation and potential shuffling
    key_gen, key_split = random.split(key)

    # Create data directory if it doesn't exist
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)

    # Download and load MNIST
    mnist = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor()
    )

    # Convert to array and keep as (N, H, W)
    X_images = torch.stack([img.squeeze() for img, _ in mnist])  # Shape: (60000, 28, 28)
    X_images = jnp.array(X_images.numpy())
    num_images, img_h, img_w = X_images.shape

    # Check if patch size is valid
    if patch_size > img_h or patch_size > img_w:
        raise ValueError(f"Patch size ({patch_size}) cannot be larger than image dimensions ({img_h}x{img_w})")

    def get_random_patch(key, images):
        """Helper function to extract a single random patch."""
        key_img, key_row, key_col = random.split(key, 3)
        
        # Select a random image
        img_idx = random.randint(key_img, (), 0, num_images)
        img = images[img_idx] # Shape (28, 28)

        # Select random top-left corner for the patch
        max_row = img_h - patch_size
        max_col = img_w - patch_size
        row_start = random.randint(key_row, (), 0, max_row + 1)
        col_start = random.randint(key_col, (), 0, max_col + 1)

        # Extract the patch using dynamic_slice for JIT compatibility
        patch = jax.lax.dynamic_slice(img, (row_start, col_start), (patch_size, patch_size))
        
        # Return patch and its L1 norm
        patch_flat = patch.reshape(-1)
        l1_norm = jnp.sum(jnp.abs(patch_flat)) # abs is technically redundant for MNIST [0,1]
        return patch_flat, l1_norm

    # Generate keys for each patch
    keys = random.split(key_gen, num_patches)

    # --- Rejection Sampling Loop ---
    valid_patches = []
    pbar = tqdm(total=num_patches, desc="Generating valid patches")
    num_generated = 0
    keys_stream = random.split(key_gen, num_patches * 10) # Generate more keys upfront, assuming < 90% rejection

    while len(valid_patches) < num_patches:
        # Simple safeguard against infinite loops if threshold is too high
        if num_generated >= len(keys_stream):
            raise RuntimeError(f"Could not generate enough valid patches. Only found {len(valid_patches)}/{num_patches}. "
                               f"Consider lowering min_l1_threshold ({min_l1_threshold:.2f}).")

        patch_key = keys_stream[num_generated]
        patch_flat, l1_norm = get_random_patch(patch_key, X_images)
        num_generated += 1

        if l1_norm >= min_l1_threshold:
            valid_patches.append(patch_flat)
            pbar.update(1)

    pbar.close()
    print(f"Generated {num_generated} candidates to get {num_patches} valid patches.")

    # Stack valid patches
    patches_flat = jnp.stack(valid_patches, axis=0) # Shape (num_patches, patch_size*patch_size)

    # Transpose to get (patch_size*patch_size, num_patches)
    X = patches_flat.T

    # Split data if requested
    if train_split_ratio < 1.0:
        num_total = X.shape[1]
        num_train = int(num_total * train_split_ratio)
        
        # Shuffle columns using the split key
        indices = random.permutation(key_split, num_total)
        X_shuffled = X[:, indices]
        
        X_train = X_shuffled[:, :num_train]
        X_test = X_shuffled[:, num_train:]
        return X_train, X_test
    else:
        return X


def visualize_dictionary_interactive(A: jnp.ndarray, patch_dim: int, nrows: int = 4, ncols: int = 16, figsize: Tuple[int, int] = (16, 4)):
    """Interactive visualization of dictionary atoms with keyboard controls.
    
    Args:
        A: Dictionary matrix where columns are atoms (patch_dim*patch_dim, n_components)
        patch_dim: The dimension (K) of the KxK patches/atoms.
        nrows: Number of rows in the plot grid.
        ncols: Number of columns in the plot grid.
        figsize: Figure size.
    
    Controls:
        - Space/Enter: Show new random subset
        - S: Save current figure
        - Q: Quit
    """
    plt.ion()  # Turn on interactive mode
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    
    # Manually adjust subplot spacing for tighter layout
    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    
    # Number of atoms to display
    n_display = min(nrows * ncols, A.shape[1])
    key = random.PRNGKey(0)
    
    def update_plot(key):
        # Clear all axes
        for ax in axes.ravel():
            ax.clear()
            ax.axis('off')
        
        # Generate new random indices if enough atoms available
        if A.shape[1] > 0:
            indices = random.choice(key, A.shape[1], shape=(min(n_display, A.shape[1]),), replace=False)
        else:
            indices = jnp.array([], dtype=int)

        # Calculate the global vmin/vmax across the selected atoms
        if len(indices) > 0:
            selected_atoms = A[:, indices]
            global_vmax = jnp.abs(selected_atoms).max()
        else:
            global_vmax = 1.0 # Default if no atoms are shown

        for i, idx in enumerate(indices):
            # Reshape atom
            atom = A[:, idx].reshape(patch_dim, patch_dim)
            # Use gray colormap and symmetric normalization around 0 using the global max
            axes.ravel()[i].imshow(atom, cmap='gray', vmin=-global_vmax, vmax=global_vmax)
        
        plt.tight_layout()
        plt.draw()
        return indices
    
    def on_key_press(event):
        nonlocal key
        if event.key in [' ', 'enter']:  # Space or Enter for new subset
            key, subkey = random.split(key)
            indices = update_plot(subkey)
            print(f"Showing atoms: {indices}")
        elif event.key == 's':  # 's' to save
            plt.savefig(Path(__file__).parent / 'dictionary_atoms.png')
            print("Saved figure as dictionary_atoms.png")
        elif event.key == 'q':  # 'q' to quit
            plt.close(fig)
            plt.ioff()
    
    # Show initial plot
    key, subkey = random.split(key)
    indices = update_plot(subkey)
    print(f"Initial atoms shown: {indices}")
    print("\nControls:")
    print("Space/Enter: Show new random subset")
    print("S: Save current figure")
    print("Q: Quit")
    
    # Connect the key press event
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    
    # Ensure tight layout is applied after initial draw
    fig.canvas.draw() # Force canvas draw
    plt.tight_layout()
    
    plt.show(block=True)  # This will block until the window is closed
    plt.ioff()  # Turn off interactive mode


def prox_l1(x, lam):
    """Proximal operator for L1 norm."""
    return jnp.sign(x) * jnp.maximum(jnp.abs(x) - lam, 0)

def prox_oblique(A):
    """Proximal operator for oblique constraint (columns with norm <= 1)."""
    col_norms = jnp.linalg.norm(A, axis=0, keepdims=True)
    return A / jnp.maximum(col_norms, 1.0)

def objective(Y, A, X, lam):
    """Compute the objective function value."""
    reconstruction_loss = 0.5 * jnp.sum((Y - A @ X) ** 2)
    l1_penalty = lam * jnp.sum(jnp.abs(X))
    return reconstruction_loss + l1_penalty

@partial(jit, static_argnums=(1, 2, 3, 4))
def palm_dictionary_learning(Y: jnp.ndarray, n_components: int, lam: float, max_iter: int = 100, stepsize_A: float = 0.1, stepsize_X: float = 0.1) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """PALM algorithm for dictionary learning with oblique constraints.
    
    Args:
        Y: Data matrix (n_features, n_samples)
        n_components: Number of dictionary atoms
        lam: L1 regularization parameter
        max_iter: Maximum number of iterations
        stepsize_A: Initial step size for dictionary updates (will be adapted)
        stepsize_X: Initial step size for coefficient updates (will be adapted)
        
    Returns:
        Tuple of:
            - Dictionary matrix A (n_features, n_components)
            - Coefficient matrix X (n_components, n_samples)
            - Array of objective values
    """
    n_features, n_samples = Y.shape
    
    # Initialize dictionary using SVD if possible, otherwise random
    if min(n_features, n_samples) >= n_components:
        U, S, Vt = jnp.linalg.svd(Y, full_matrices=False)
        A = U[:, :n_components]  # Shape: (n_features, n_components)
    else:
        key = jax.random.PRNGKey(0)
        A = jax.random.normal(key, (n_features, n_components))
    
    # Normalize dictionary atoms
    A = prox_oblique(A)
    
    # Initialize coefficients
    X = jnp.zeros((n_components, n_samples))
    
    # Removed initial calculation of Lipschitz constants and step sizes here

    def update_A(A, X, Y):
        """Update dictionary using proximal gradient step."""
        # Calculate L_A and stepsize_A based on current X
        L_A = jnp.linalg.norm(X @ X.T, ord=2)
        stepsize_A = 1.0 / (L_A + 1e-10) # Add epsilon for stability
        
        grad_A = (A @ X - Y) @ X.T
        A_new = A - stepsize_A * grad_A
        return prox_oblique(A_new)
    
    def update_X(A, X, Y):
        """Update coefficients using proximal gradient step."""
        # Calculate L_X and stepsize_X based on current A
        L_X = jnp.linalg.norm(A.T @ A, ord=2)
        stepsize_X = 1.0 / (L_X + 1e-10) # Add epsilon for stability
        
        grad_X = A.T @ (A @ X - Y)
        X_new = X - stepsize_X * grad_X
        return prox_l1(X_new, lam * stepsize_X)
    
    def body_fun(carry, _):
        A, X = carry
        
        # Update dictionary
        A_new = update_A(A, X, Y)
        
        # Update coefficients
        X_new = update_X(A_new, X, Y) # Use A_new for update_X
        
        # Compute objective
        obj = objective(Y, A_new, X_new, lam)
        
        return (A_new, X_new), obj
    
    # Run PALM iterations
    init_carry = (A, X)
    (A_final, X_final), obj_vals = jax.lax.scan(body_fun, init_carry, None, length=max_iter)
    
    return A_final, X_final, obj_vals

@partial(jit, static_argnums=(2, 3, 4))
def solve_lasso(Y: jnp.ndarray, A: jnp.ndarray, lam: float, max_iter: int = 1000, tol: float = 1e-4) -> jnp.ndarray:
    """Solves the LASSO problem: min_X 0.5*||Y - AX||_F^2 + lam*||X||_1 for X, given Y and A.

    Uses the Iterative Shrinkage-Thresholding Algorithm (ISTA).

    Args:
        Y: Data matrix (n_features, n_samples).
        A: Dictionary matrix (n_features, n_components).
        lam: L1 regularization parameter.
        max_iter: Maximum number of iterations.
        tol: Convergence tolerance for the relative change in X.

    Returns:
        Coefficient matrix X (n_components, n_samples).
    """
    n_components, n_samples = A.shape[1], Y.shape[1]
    
    # Initialize coefficients
    X = jnp.zeros((n_components, n_samples))
    
    # Precompute A.T and Lipschitz constant for ISTA step
    At = A.T
    L = jnp.linalg.norm(At @ A, ord=2)
    stepsize = 1.0 / (L + 1e-10) # Add epsilon for stability

    def body_fun(X_prev, _):
        grad_X = At @ (A @ X_prev - Y)
        X_cand = X_prev - stepsize * grad_X
        X_next = prox_l1(X_cand, lam * stepsize)
        
        # Optional: Add convergence check (cannot be done inside scan easily)
        # For simplicity here, we run for fixed max_iter within JIT
        return X_next, None # Return X_next as carry, no per-iteration output needed

    # Run ISTA iterations
    X_final, _ = jax.lax.scan(body_fun, X, None, length=max_iter)
    
    # Note: A proper convergence check based on `tol` would typically require a 
    # standard Python loop or `jax.lax.while_loop`, which can be less efficient 
    # to JIT compile than a fixed-length `scan`.
    # For this use case, running a fixed number of iterations is often sufficient.

    return X_final

def palm_dictionary_learning_with_progress(Y: jnp.ndarray, n_components: int, lam: float, max_iter: int = 100, stepsize_A: float = 0.1, stepsize_X: float = 0.1) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Wrapper for palm_dictionary_learning that shows progress bar."""
    # JIT the PALM function itself
    palm_jitted = jit(palm_dictionary_learning, static_argnums=(1, 2, 3, 4))
    A, X, obj_vals = palm_jitted(Y, n_components, lam, max_iter, stepsize_A, stepsize_X)
    
    # Convert to numpy for progress bar
    obj_vals_np = np.array(obj_vals)
    
    # Show progress bar with objective value
    pbar = tqdm(range(max_iter), desc="PALM Dictionary Learning")
    for i in pbar:
        if i < len(obj_vals_np):
            pbar.set_description(f"PALM Dictionary Learning (obj: {obj_vals_np[i]:.2e})")
    
    return A, X, obj_vals


# Example usage
if __name__ == "__main__":
    # Parameters
    N = 12500  # Number of samples
    D = 10    # Dictionary dimension (Used only for synthetic data initially)
    train_split_ratio = 0.8 # Fraction for training set
    sparsity_threshold = 1e-1 # Threshold for counting non-zeros in codes
    max_iter = 6000 # Max iterations for solvers
    lam = 0.1     # L1 regularization parameter for PALM
    debug = False # If True, visualize data samples before running algorithm
    overcomplete_factor = 8 # Overcomplete factor for K-SVD/PALM
    
    # Initialize random key
    key = random.PRNGKey(2)
    key_data, key_init, key_msp, key_palm = random.split(key, 4)
    
    # Algorithm choice
    algorithm = "msp"   # "msp" or "palm"
    data_source = "patches" # "mnist", "patches", or "synthetic"
    patch_size = 8       # Patch size if data_source="patches"
    
    if data_source == "mnist":
        X_train, X_test = load_mnist_data(num_samples=N, key=key_data, train_split_ratio=train_split_ratio)
        print(f"\nLoaded MNIST data: Train shape {X_train.shape}, Test shape {X_test.shape}")
        # For MNIST:
        # - MSP: square dictionary (784x784)
        # - PALM: overcomplete dictionary (e.g., 8x784 = 6272 atoms)
        n_components = X_train.shape[0] if algorithm == "msp" else overcomplete_factor * X_train.shape[0] # Adjusted overcompleteness for PALM
    elif data_source == "patches":
        X_train, X_test = load_mnist_patches(key=key_data, num_patches=N, patch_size=patch_size, train_split_ratio=train_split_ratio)
        print(f"\nLoaded MNIST patches: Train shape {X_train.shape}, Test shape {X_test.shape}")
        # For patches:
        # - MSP: square dictionary (patch_size*patch_size)
        # - PALM: Overcomplete, e.g., 8x features
        n_features = X_train.shape[0]
        n_components = n_features if algorithm == "msp" else overcomplete_factor * n_features
    else:
        # Synthetic data not supported without generate_sparse_data function
        raise ValueError("Synthetic data source is not supported. Please use 'mnist' or 'patches'.")
    
    # Get problem dimension from data
    D = X_train.shape[0]
    
    # Debug: Visualize some training data samples
    if debug:
        print("\nDebug: Visualizing training data samples (interactive viewer)...")
        # Determine the dimension for reshaping (patch size or 28 for MNIST)
        viz_dim = patch_size if data_source == "patches" else (28 if data_source == "mnist" else None)
        if viz_dim:
            # Visualize requires samples as columns (D, n_samples) -> (viz_dim*viz_dim, n_samples)
            visualize_dictionary_interactive(X_train, patch_dim=viz_dim, nrows=4, ncols=8, figsize=(15, 8))
        else:
            print("(Skipping visualization for synthetic data)")

    if algorithm == "msp":
        # Initialize random orthogonal dictionary
        A_init = initialize_orthogonal(D, key_init)
        # Run MSP algorithm
        A_learned, errors, obj_values = msp_iteration(A_init, X_train, key_msp, max_iters=max_iter)
        
        # Create figure with subplots
        if data_source == "mnist" or data_source == "patches":
            # For MNIST, show convergence plots and dictionary atoms
            fig = plt.figure(figsize=(15, 10))
            
            # Plot error convergence
            ax1 = plt.subplot(221)
            iterations = jnp.arange(len(errors))
            ax1.semilogy(iterations, errors)
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Error (Frobenius norm)')
            ax1.set_title('Convergence of Successive Iterates')
            ax1.grid(True)
            
            # Plot objective value
            ax2 = plt.subplot(222)
            ax2.plot(iterations, obj_values)
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Objective Value')
            ax2.set_title('Objective Value vs Iteration')
            ax2.grid(True)
            
            plt.tight_layout()
            plt.show()
            
            # Show learned dictionary atoms interactively
            print("\nLearned dictionary atoms (interactive viewer):")
            # Reshape according to patch size if needed
            atom_dim = int(jnp.sqrt(A_learned.shape[0])) # Assumes square patches/images
            # Visualize requires atoms as columns (D, n_components) -> (atom_dim*atom_dim, n_components)
            visualize_dictionary_interactive(A_learned, patch_dim=atom_dim)
            
        else:
            # Original visualization for sparse case
            # Create figure with three subplots
            fig = plt.figure(figsize=(15, 5))
            
            # Plot error convergence
            ax1 = plt.subplot(131)
            iterations = jnp.arange(len(errors))
            ax1.semilogy(iterations, errors)
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Error (Frobenius norm)')
            ax1.set_title('Convergence of Successive Iterates')
            ax1.grid(True)
            
            # Plot objective value
            ax2 = plt.subplot(132)
            ax2.plot(iterations, obj_values)
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Objective Value')
            ax2.set_title('Objective Value vs Iteration')
            ax2.grid(True)
            
            # Plot A_learned to check for signed permutation of I
            ax3 = plt.subplot(133)
            im = ax3.imshow(A_learned, cmap='RdBu', vmin=-1, vmax=1)
            ax3.set_title('A_learned\n(Should be signed permutation of I)')
            plt.colorbar(im, ax=ax3)
            
            plt.tight_layout()
            plt.show()
        
        # Print final metrics
        print(f"\nFinal error between successive iterates: {errors[-1]:.6e}")
        print(f"Final objective value: {obj_values[-1]:.6f}")
        
        # Verify orthogonality of learned matrix
        error_ortho = jnp.linalg.norm(A_learned.T @ A_learned - jnp.eye(D), ord='fro')
        print(f"Orthogonality error: {error_ortho:.6e}")
        
        # Evaluate sparsity on test set
        if X_test is not None:
            # gamma has shape (n_components, N_train)
            # Compute codes for test set using the LASSO solver
            print(f"\nSolving LASSO for test codes ({X_test.shape[1]} samples)...")
            start_lasso_time = time.time()
            gamma_test = solve_lasso(X_test, A_learned, lam, max_iter=500) # Use dedicated LASSO solver
            end_lasso_time = time.time()
            print(f"LASSO solve took {end_lasso_time - start_lasso_time:.2f} seconds.")
            
            l0_norms = jnp.sum(jnp.abs(gamma_test) > sparsity_threshold, axis=0)
            avg_l0 = jnp.mean(l0_norms)
            sparsity_percentage = (avg_l0 / D) * 100
            print(f"Average code sparsity on test set ({X_test.shape[1]} samples): {sparsity_percentage:.2f}% (L0 norm / {D})")
        else:
            print("\nSkipping test set sparsity evaluation (no test set provided).")
        
        if data_source == "synthetic":
            # Only show permutation metrics for sparse case
            # For MNIST we don't expect to recover I
            max_entries = jnp.maximum(jnp.abs(A_learned).max(axis=0), jnp.abs(A_learned).max(axis=1))
            min_max_entry = jnp.min(max_entries)
            print(f"\nMin of max absolute entries per row/col: {min_max_entry:.6f} (should be close to 1)")
            
            close_to_one = jnp.sum(jnp.abs(A_learned) > 0.9)
            print(f"Number of entries with |value| > 0.9: {close_to_one} (should be {D} for perfect recovery)")
            
            rows_good = jnp.sum(jnp.sum(jnp.abs(A_learned) > 0.9, axis=1) == 1) == D
            cols_good = jnp.sum(jnp.sum(jnp.abs(A_learned) > 0.9, axis=0) == 1) == D
            print(f"Each row has exactly one |value| > 0.9: {rows_good}")
            print(f"Each col has exactly one |value| > 0.9: {cols_good}")
    elif algorithm == "palm":
        # Run PALM algorithm
        print(f"\nRunning PALM algorithm with {n_components} atoms...")
        A_learned, gamma, obj_vals = palm_dictionary_learning_with_progress(
            X_train, n_components, lam, max_iter=max_iter
        )
        
        if data_source == "mnist" or data_source == "patches":
            # Show learned dictionary atoms interactively
            print("\nLearned dictionary atoms (interactive viewer):")
            # Adjust grid size for larger dictionary
            nrows = int(jnp.sqrt(n_components)) if n_components > 0 else 1
            ncols = (n_components + nrows - 1) // nrows  # Ceiling division
            atom_dim = int(jnp.sqrt(A_learned.shape[0])) # Assumes square patches/images
            visualize_dictionary_interactive(A_learned, patch_dim=atom_dim)
            
            # Show reconstruction error
            X_recon = A_learned @ gamma
            recon_error = jnp.linalg.norm(X_train - X_recon, ord='fro') / jnp.linalg.norm(X_train, ord='fro')
            print(f"\nRelative reconstruction error: {recon_error:.6f}")
            
            # Evaluate sparsity on test set
            if X_test is not None:
                # gamma has shape (n_components, N_train)
                # Compute codes for test set using the LASSO solver
                print(f"\nSolving LASSO for test codes ({X_test.shape[1]} samples)...")
                start_lasso_time = time.time()
                gamma_test = solve_lasso(X_test, A_learned, lam, max_iter=500) # Use dedicated LASSO solver
                end_lasso_time = time.time()
                print(f"LASSO solve took {end_lasso_time - start_lasso_time:.2f} seconds.")
                
                l0_norms = jnp.sum(jnp.abs(gamma_test) > sparsity_threshold, axis=0)
                avg_l0 = jnp.mean(l0_norms)
                sparsity_percentage = (avg_l0 / n_components) * 100
                print(f"Average code sparsity on test set ({X_test.shape[1]} samples): {sparsity_percentage:.2f}% (L0 norm / {n_components})")
            else:
                print("\nSkipping test set sparsity evaluation (no test set provided).")
            
            # Plot objective values
            plt.figure(figsize=(10, 6))
            plt.semilogy(obj_vals)
            plt.xlabel('Iteration')
            plt.ylabel('Objective Value')
            plt.title('PALM Dictionary Learning Convergence')
            plt.grid(True)
            plt.show()
        else:
            # For sparse case, visualization not implemented
            print("\nVisualization for sparse case not implemented for PALM") 