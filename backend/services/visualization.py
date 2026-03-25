"""
Server-side matplotlib rendering: heatmap, image grid, overlay, cluster map.

Axis convention:
  - data[i, j]: i indexes alphas (x-axis), j indexes betas (y-axis)
  - Heatmap uses .T with origin='lower' so x=alpha, y=beta
  - Image grid: col=alpha (left→right), row flipped for y=beta (bottom→top)
"""
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

from backend import config


def render_heatmap(sensitivity: np.ndarray, alphas: np.ndarray,
                   betas: np.ndarray, output_path: Path,
                   prompt_a: str = "", prompt_b: str = "", prompt_c: str = ""):
    extent = [alphas[0], alphas[-1], betas[0], betas[-1]]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(sensitivity.T, origin='lower', extent=extent,
                   cmap='hot', aspect='auto')
    ratio = sensitivity.max() / (np.median(sensitivity) + 1e-10)
    xlabel = f'→ {prompt_b[:20]}' if prompt_b else 'alpha'
    ylabel = f'→ {prompt_c[:20]}' if prompt_c else 'beta'
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f'DINOv2 Sensitivity (max/med={ratio:.1f})', fontsize=13)
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def assemble_image_grid(thumbnail_grid: dict[tuple[int, int], bytes],
                        grid_size: int, output_path: Path):
    ts = config.THUMBNAIL_SIZE
    canvas = np.zeros((grid_size * ts, grid_size * ts, 3), dtype=np.uint8)

    for (alpha_idx, beta_idx), thumb_bytes in thumbnail_grid.items():
        img = Image.open(io.BytesIO(thumb_bytes))
        arr = np.array(img.convert('RGB'))
        if arr.shape[:2] != (ts, ts):
            img = img.resize((ts, ts), Image.LANCZOS)
            arr = np.array(img.convert('RGB'))
        row = grid_size - 1 - beta_idx
        col = alpha_idx
        canvas[row * ts:(row + 1) * ts, col * ts:(col + 1) * ts] = arr

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(output_path)
    return canvas


def render_overlay(sensitivity: np.ndarray, thumbnail_grid: dict,
                   grid_size: int, alphas: np.ndarray, betas: np.ndarray,
                   tau: float, output_path: Path):
    extent = [alphas[0], alphas[-1], betas[0], betas[-1]]

    # Build canvas from thumbnails
    ts = config.THUMBNAIL_SIZE
    canvas = np.zeros((grid_size * ts, grid_size * ts, 3), dtype=np.uint8)
    for (ai, bi), thumb_bytes in thumbnail_grid.items():
        img = Image.open(io.BytesIO(thumb_bytes))
        arr = np.array(img.convert('RGB'))
        if arr.shape[:2] != (ts, ts):
            img = img.resize((ts, ts), Image.LANCZOS)
            arr = np.array(img.convert('RGB'))
        row = grid_size - 1 - bi
        canvas[row * ts:(row + 1) * ts, ai * ts:(ai + 1) * ts] = arr

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.imshow(canvas, extent=extent, origin='upper', aspect='auto')

    sens_norm = (sensitivity - sensitivity.min()) / (sensitivity.max() - sensitivity.min() + 1e-10)
    overlay = plt.cm.hot(sens_norm.T)
    overlay[:, :, 3] = sens_norm.T * 0.6
    ax.imshow(overlay, extent=extent, origin='lower', aspect='auto')

    median_s = np.median(sensitivity)
    if median_s > 0:
        # Three tau levels for context
        tau_levels = [
            (1.2, 'yellow', 1.0, '--'),
            (1.5, 'cyan', 2.0, '-'),
            (1.8, 'magenta', 1.0, '--'),
        ]
        for t_val, color, lw, ls in tau_levels:
            level = median_s * t_val
            if level < sensitivity.max():
                try:
                    ax.contour(alphas, betas, sensitivity.T,
                               levels=[level], colors=color, linewidths=lw, linestyles=ls)
                except ValueError:
                    pass

    ax.set_xlabel('alpha', fontsize=11)
    ax.set_ylabel('beta', fontsize=11)
    ax.set_title('Ridge Overlay (yellow=1.2τ, cyan=1.5τ, magenta=1.8τ)', fontsize=12)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def render_clusters(clusters: np.ndarray, thumbnail_grid: dict,
                    grid_size: int, alphas: np.ndarray, betas: np.ndarray,
                    output_path: Path):
    """Render cluster label map overlaid on image grid."""
    extent = [alphas[0], alphas[-1], betas[0], betas[-1]]

    ts = config.THUMBNAIL_SIZE
    canvas = np.zeros((grid_size * ts, grid_size * ts, 3), dtype=np.uint8)
    for (ai, bi), thumb_bytes in thumbnail_grid.items():
        img = Image.open(io.BytesIO(thumb_bytes))
        arr = np.array(img.convert('RGB'))
        if arr.shape[:2] != (ts, ts):
            img = img.resize((ts, ts), Image.LANCZOS)
            arr = np.array(img.convert('RGB'))
        row = grid_size - 1 - bi
        canvas[row * ts:(row + 1) * ts, ai * ts:(ai + 1) * ts] = arr

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.imshow(canvas, extent=extent, origin='upper', aspect='auto')

    # Cluster boundaries as contours
    for i in range(grid_size):
        for j in range(grid_size):
            for di, dj in [(1, 0), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < grid_size and 0 <= nj < grid_size:
                    if clusters[i, j] != clusters[ni, nj]:
                        # Draw boundary line
                        a1, a2 = alphas[i], alphas[min(ni, grid_size - 1)]
                        b1, b2 = betas[j], betas[min(nj, grid_size - 1)]
                        ax.plot([(a1 + a2) / 2], [(b1 + b2) / 2], 'c.', markersize=3)

    # Also show cluster IDs as colored overlay
    n_clusters = len(set(clusters.ravel()) - {-1})
    cluster_colors = plt.cm.tab20(np.linspace(0, 1, max(n_clusters, 1)))
    overlay = np.zeros((*clusters.T.shape, 4))
    for k in range(clusters.max() + 1):
        mask = clusters.T == k
        if mask.any():
            color = cluster_colors[k % len(cluster_colors)]
            overlay[mask] = [*color[:3], 0.25]
    # Noise points (cluster -1) in red
    noise_mask = clusters.T == -1
    overlay[noise_mask] = [1, 0, 0, 0.3]

    ax.imshow(overlay, extent=extent, origin='lower', aspect='auto')

    ax.set_xlabel('alpha', fontsize=11)
    ax.set_ylabel('beta', fontsize=11)
    ax.set_title(f'Semantic Regions ({n_clusters} clusters)', fontsize=13)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
