import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Set up a badly conditioned function f: R^2 -> R
    # Using a quadratic function with very different eigenvalues
    def f(x):
        # Less badly conditioned function: 20*x^2 + y^2
        # Condition number = 20 (reduced from 100)
        return 20 * x[0]**2 + x[1]**2
    
    # Compute gradient and Hessian using JAX
    grad_f = jax.grad(f)
    hess_f = jax.hessian(f)
    
    # Function to compute Newton direction: [hess(f)]^-1 grad(f)
    def newton_direction(x):
        g = grad_f(x)
        H = hess_f(x)
        # Solve the linear system H * direction = g
        # This is equivalent to direction = H^-1 * g
        return jnp.linalg.solve(H, g)
    
    # Create a grid of points
    x = np.linspace(-1, 1, 20)
    y = np.linspace(-1, 1, 20)
    X, Y = np.meshgrid(x, y)
    
    # Compute gradients and Newton directions at each point
    grad_field = np.zeros((20, 20, 2))
    newton_field = np.zeros((20, 20, 2))
    
    for i in range(20):
        for j in range(20):
            point = jnp.array([X[i, j], Y[i, j]])
            grad_field[i, j] = grad_f(point)
            newton_field[i, j] = newton_direction(point)
    
    # Calculate max norms for scaling within each field
    grad_max_norm = np.sqrt((grad_field**2).sum(axis=2)).max()
    newton_max_norm = np.sqrt((newton_field**2).sum(axis=2)).max()
    
    # Scale each field by its maximum norm for better visualization
    grad_field_scaled = grad_field / grad_max_norm
    newton_field_scaled = newton_field / newton_max_norm
    
    # Create contour plot of the function
    Z = np.array([[f(jnp.array([X[i, j], Y[i, j]])) for j in range(20)] for i in range(20)])
    
    # Plot the results
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot the function contours
    contour = axes[0].contour(X, Y, Z, levels=20, cmap='viridis')
    axes[0].set_title('Contour Plot of f(x,y) = 20x² + y²')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    fig.colorbar(contour, ax=axes[0])
    
    # Plot the gradient field with arrows normalized per-picture
    axes[1].quiver(X, Y, -grad_field_scaled[:, :, 0], -grad_field_scaled[:, :, 1], 
                   color='blue', scale=4, scale_units='inches')
    axes[1].set_title('Gradient Field -∇f')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    
    # Plot the Newton direction field with arrows normalized per-picture
    axes[2].quiver(X, Y, -newton_field_scaled[:, :, 0], -newton_field_scaled[:, :, 1], 
                   color='red', scale=4, scale_units='inches')
    axes[2].set_title('Newton Direction Field -[∇²f]⁻¹∇f')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    
    plt.tight_layout()
    plt.savefig('optimization_fields.png', dpi=300)
    plt.show()
    
    print("Visualization complete! Check 'optimization_fields.png' for the output.")


if __name__ == "__main__":
    main()
