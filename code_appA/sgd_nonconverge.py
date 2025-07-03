import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Define three different quadratic functions centered at different points
    centers = [
        jnp.array([-0.5, 0.5]),   # x_1
        jnp.array([0.5, 0.5]),    # x_2
        jnp.array([0.0, -0.5])    # x_3
    ]
    
    def f1(x):
        return jnp.sum((x - centers[0])**2)
    
    def f2(x):
        return jnp.sum((x - centers[1])**2)
    
    def f3(x):
        return jnp.sum((x - centers[2])**2)
    
    functions = [f1, f2, f3]
    
    # Define the average objective function
    def avg_f(x):
        return (f1(x) + f2(x) + f3(x)) / 3.0
    
    # Compute gradients using JAX
    grads = [jax.grad(f) for f in functions]
    avg_grad = jax.grad(avg_f)
    
    # Create a grid of points
    x = np.linspace(-1.5, 1.5, 30)
    y = np.linspace(-1.5, 1.5, 30)
    X, Y = np.meshgrid(x, y)
    
    # Compute gradient field for the average function
    avg_field = np.zeros((30, 30, 2))
    for i in range(30):
        for j in range(30):
            point = jnp.array([X[i, j], Y[i, j]])
            avg_field[i, j] = avg_grad(point)
    
    # Normalize gradient field for visualization
    max_norm = np.sqrt((avg_field**2).sum(axis=2)).max()
    avg_field_normalized = avg_field / max_norm
    
    # Simulation parameters
    num_iterations = 100
    learning_rate = 0.3
    beta = 0.9  # Momentum parameter for Nesterov SGD
    
    # Start from the same point for both algorithms
    start_point = jnp.array([0.8, 0.8])
    
    # Simulate vanilla SGD trajectory
    current_point_sgd = start_point.copy()
    sgd_trajectory = [current_point_sgd.copy()]
    
    for i in range(num_iterations):
        # Cycle through the functions
        func_idx = i % 3
        grad = grads[func_idx](current_point_sgd)
        
        # Update using gradient descent
        current_point_sgd = current_point_sgd - learning_rate * grad
        sgd_trajectory.append(current_point_sgd.copy())
    
    sgd_trajectory = np.array(sgd_trajectory)
    
    # Simulate Nesterov SGD trajectory
    current_point_nesterov = start_point.copy()
    nesterov_trajectory = [current_point_nesterov.copy()]
    g_prev = jnp.zeros(2)  # Initialize momentum term
    
    for i in range(num_iterations):
        # Cycle through the functions
        func_idx = i % 3
        grad = grads[func_idx](current_point_nesterov)
        
        # Update momentum term with Nesterov
        g_t = beta * g_prev + (1 - beta) * grad
        
        # Update position
        current_point_nesterov = current_point_nesterov - learning_rate * g_t
        nesterov_trajectory.append(current_point_nesterov.copy())
        
        # Update previous momentum
        g_prev = g_t
    
    nesterov_trajectory = np.array(nesterov_trajectory)
    
    # Compute average function values over the grid
    Z = np.array([[avg_f(jnp.array([X[i, j], Y[i, j]])) for j in range(30)] for i in range(30)])
    
    # Create side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Common plotting function for both subplots
    def plot_common(ax, title):
        # Plot contours of the average function
        contour = ax.contour(X, Y, Z, levels=20, alpha=0.6)
        
        # Plot the centers of the individual functions
        colors = ['blue', 'green', 'red']
        for i, center in enumerate(centers):
            ax.scatter(center[0], center[1], color=colors[i], s=100, marker='*', 
                      label=f'Center {i+1}')
        
        # Plot the vector field of the average gradient
        ax.quiver(X[::2, ::2], Y[::2, ::2], 
                 -avg_field_normalized[::2, ::2, 0], -avg_field_normalized[::2, ::2, 1], 
                 color='purple', scale=4, scale_units='inches')
        
        # Mark the optimal point of the average function
        optimal_point = np.mean(centers, axis=0)
        ax.scatter(optimal_point[0], optimal_point[1], color='purple', s=150, marker='o', 
                  label='Average Optimum')
        
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
    
    # Plot 1: Vanilla SGD
    plot_common(axes[0], 'Vanilla SGD (Non-convergent)')
    
    # Plot SGD trajectory
    axes[0].plot(sgd_trajectory[:, 0], sgd_trajectory[:, 1], 'k-', alpha=0.7, linewidth=1.5, 
                label='SGD Trajectory')
    
    # Add arrows to show direction
    arrow_indices = np.linspace(0, len(sgd_trajectory)-2, 15, dtype=int)
    for idx in arrow_indices:
        axes[0].arrow(sgd_trajectory[idx, 0], sgd_trajectory[idx, 1], 
                     sgd_trajectory[idx+1, 0] - sgd_trajectory[idx, 0], 
                     sgd_trajectory[idx+1, 1] - sgd_trajectory[idx, 1],
                     head_width=0.03, head_length=0.05, fc='k', ec='k')
    
    # Plot 2: Nesterov SGD
    plot_common(axes[1], f'Nesterov SGD (β={beta}, Convergent)')
    
    # Plot Nesterov trajectory
    axes[1].plot(nesterov_trajectory[:, 0], nesterov_trajectory[:, 1], 'k-', alpha=0.7, linewidth=1.5, 
                label='Nesterov Trajectory')
    
    # Add arrows to show direction
    arrow_indices = np.linspace(0, len(nesterov_trajectory)-2, 15, dtype=int)
    for idx in arrow_indices:
        axes[1].arrow(nesterov_trajectory[idx, 0], nesterov_trajectory[idx, 1], 
                     nesterov_trajectory[idx+1, 0] - nesterov_trajectory[idx, 0], 
                     nesterov_trajectory[idx+1, 1] - nesterov_trajectory[idx, 1],
                     head_width=0.03, head_length=0.05, fc='k', ec='k')
    
    # Add legends
    for ax in axes:
        ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('sgd_vs_nesterov.png', dpi=300)
    plt.show()
    
    print("Visualization complete! Check 'sgd_vs_nesterov.png' for the output.")


if __name__ == "__main__":
    main()
