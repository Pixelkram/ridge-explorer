"""
Grid coordinate computation and latent noise generation.

Grid axes:
  - Alpha (horizontal): interpolates text embedding from prompt A to prompt B
  - Beta (vertical): perpendicular direction in text embedding space
  - Noise: fixed z0 from seed (same for all cells)
"""
import numpy as np
from backend import config


def build_grid_coords(grid_size: int,
                      alpha_range: tuple[float, float] = config.ALPHA_RANGE,
                      beta_range: tuple[float, float] = config.BETA_RANGE):
    """Return alpha and beta arrays for the grid."""
    alphas = np.linspace(alpha_range[0], alpha_range[1], grid_size)
    betas = np.linspace(beta_range[0], beta_range[1], grid_size)
    return alphas, betas


def generate_noise(seed: int, latent_shape=(1, 4, 64, 64)):
    """Generate a single fixed noise vector from a seed."""
    rng = np.random.RandomState(seed)
    z0 = rng.randn(*latent_shape).astype(np.float16)
    return z0


def generate_perp_seed(seed: int):
    """Generate a seed for the perpendicular direction (deterministic from main seed)."""
    return seed * 31337 + 7
