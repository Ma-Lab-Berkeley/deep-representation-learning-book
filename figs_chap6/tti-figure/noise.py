#!/usr/bin/env python3
"""
Noise application script for images using JAX.
Applies diffusion-style noise to an image based on a noise schedule.
"""

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
import tyro
from pathlib import Path


def load_and_resize_image(image_path: str, size: tuple[int, int] = (256, 256)) -> jnp.ndarray:
    """Load an image and resize it to the specified dimensions."""
    image = Image.open(image_path).convert('RGB')
    image = image.resize(size, Image.Resampling.LANCZOS)
    # Convert to float32 array normalized to [0, 1]
    image_array = np.array(image).astype(np.float32) / 255.0
    return jnp.array(image_array)


def apply_noise(x: jnp.ndarray, t: float, key: jax.random.PRNGKey) -> jnp.ndarray:
    """
    Apply noise to image according to: x_t = α_t * x + σ_t * g
    where α_t = sqrt(1 - t²) and σ_t = t
    """
    alpha_t = jnp.sqrt(1 - t**2)
    sigma_t = t
    
    # Generate Gaussian noise with same shape as image
    noise = jax.random.normal(key, x.shape)
    
    # Apply noise formula
    x_t = alpha_t * x + sigma_t * noise
    
    return x_t


def save_image(image_array: jnp.ndarray, output_path: str):
    """Save JAX array as PNG image."""
    # Clamp values to [0, 1] and convert to uint8
    image_clamped = jnp.clip(image_array, 0.0, 1.0)
    image_uint8 = (image_clamped * 255).astype(jnp.uint8)
    
    # Convert to PIL Image and save
    pil_image = Image.fromarray(np.array(image_uint8))
    pil_image.save(output_path)


def main(t: float = 0.5, filename: str = "corgi.png") -> None:
    """
    Apply noise to image and save result.
    
    Args:
        t: Noise level parameter (0.0 to 1.0)
        filename: Input image filename (default: corgi.png)
    """
    if not (0.0 <= t <= 1.0):
        raise ValueError("Parameter t must be between 0.0 and 1.0")
    
    # Set up paths
    input_path = Path(filename)
    # Extract base name without extension for output filename
    base_name = input_path.stem
    output_path = f"{base_name}_noise_{t:.3f}.png"
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")
    
    # Load and process image
    print(f"Loading image from {input_path}")
    image = load_and_resize_image(str(input_path))
    print(f"Image shape: {image.shape}")
    
    # Generate random key for noise
    key = jax.random.PRNGKey(42)
    
    # Apply noise
    print(f"Applying noise with t={t:.3f} (α_t={jnp.sqrt(1-t**2):.3f}, σ_t={t:.3f})")
    noisy_image = apply_noise(image, t, key)
    
    # Save result
    print(f"Saving result to {output_path}")
    save_image(noisy_image, output_path)
    print("Done!")


if __name__ == "__main__":
    tyro.cli(main) 