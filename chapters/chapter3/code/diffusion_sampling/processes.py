from typing import Callable, Any

import jax
import jax.numpy as jnp
from jax import Array


class DenoisingProcess:
    def __init__(self, alpha: Callable[[Array], Array], sigma: Callable[[Array], Array]):
        self.alpha: Callable[[Array], Array] = alpha
        self.sigma: Callable[[Array], Array] = sigma

    def add_noise(self, key: jax.random.PRNGKey, x: Array, t: Array) -> Array:
        """
        Add noise to the input tensor x using the process:
        x_sigma = alpha(sigma) * x + sigma * N(0, I)

        Args:
                key (jax.random.PRNGKey): Random key for JAX's PRNG.
                x (Array): Input array to add noise to. (N, D)
                t (Array): Noise level. ()

        Returns:
                Array: Noisy array x_sigma = alpha(sigma) * x + sigma * N(0, I).
        """
        alpha = self.alpha(t)
        sigma = self.sigma(t)
        return alpha * x + sigma * jax.random.normal(key, shape=x.shape)

    def denoise(self, x0: Array, ce_func: Callable[[Array, Array], Array], ts: Array) -> Array:
        """
        Denoise the input array using a conditional expectation function and timesteps.
        
        Args:
                x0 (Array): Initial noisy array to denoise.
                ce_func (Callable): Conditional expectation function that takes (x, t) and returns denoised x.
                ts (Array): Schedule of timesteps, must end with 0 (or very small value).
                
        Returns:
                Array: Denoised array.
        """
        # comment: ts[-1] must be 0 (or very small if wanting to early stop)
        
        # Define the step function for a single denoising step
        def step_fn(carry, m):
            x = carry
            t_m = ts[m]
            t_m_plus_1 = ts[m + 1]
            alpha_m = self.alpha(t_m)
            alpha_m_plus_1 = self.alpha(t_m_plus_1)
            sigma_m = self.sigma(t_m)
            sigma_m_plus_1 = self.sigma(t_m_plus_1)
            ce_m = ce_func(x, t_m)
            x_new = (sigma_m_plus_1 / sigma_m) * x + (alpha_m_plus_1 - (sigma_m_plus_1 / sigma_m) * alpha_m) * ce_m
            return x_new, None
        
        # Run the denoising process through the timesteps
        M = ts.shape[0] - 1
        x_final, _ = jax.lax.scan(step_fn, x0, jnp.arange(M))
        
        return x_final
