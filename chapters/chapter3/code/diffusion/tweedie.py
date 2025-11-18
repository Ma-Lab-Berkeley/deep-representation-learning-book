import jax.numpy as jnp
from jax import Array


def tweedie_score_to_denoise(X: Array, score_X: Array, alpha: Array, sigma: Array) -> Array:
    """
    Compute the denoised data via Tweedie's formula.

    :param X: data (N, D)
    :param score_X: score function evaluated at X (N, D)
    :param alpha: scale ()
    :param sigma: noise level ()
    :return: denoised X (N, D)
    """
    return (X + (sigma ** 2) * score_X) / alpha


def tweedie_denoise_to_score(X: Array, denoised_X: Array, alpha: Array, sigma: Array) -> Array:
    """
    Compute the gradient of the log-likelihood via Tweedie's formula.

    :param X: data (N, D)
    :param denoised_X: denoised X (N, D)
    :param alpha: scale ()
    :param sigma: noise level ()
    :return: score function evaluated at X (N, D)
    """
    return (alpha * denoised_X - X) / sigma**2
