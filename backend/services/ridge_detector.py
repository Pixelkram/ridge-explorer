"""
Ridge detection from DINOv2 embeddings + DBSCAN clustering.
Supports both 2D and 3D grids.
"""
import numpy as np
from scipy import ndimage
from sklearn.cluster import DBSCAN
from backend import config


def compute_sensitivity(embeddings: np.ndarray) -> np.ndarray:
    """Compute DINOv2 neighbor distance map (2D or 3D).

    Args:
        embeddings: (gs, gs, 768) for 2D or (gs, gs, gs, 768) for 3D

    Returns:
        sensitivity: same spatial shape as input (without embedding dim)
    """
    ndim = len(embeddings.shape) - 1  # spatial dimensions

    if ndim == 2:
        return _sensitivity_2d(embeddings)
    elif ndim == 3:
        return _sensitivity_3d(embeddings)
    else:
        raise ValueError(f"Unsupported dimensionality: {ndim}")


def _sensitivity_2d(embeddings):
    """Mean cosine distance to 4-neighbors.
    Skips cells or neighbors with zero-norm embeddings (e.g. non-generated cells
    from fast scan partial generation)."""
    gs = embeddings.shape[0]
    gs_b = embeddings.shape[1]
    sensitivity = np.zeros((gs, gs_b))
    for i in range(gs):
        for j in range(gs_b):
            if np.dot(embeddings[i, j], embeddings[i, j]) < 0.01:
                continue  # skip cells without real embeddings
            dists = []
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < gs and 0 <= nj < gs_b:
                    if np.dot(embeddings[ni, nj], embeddings[ni, nj]) < 0.01:
                        continue  # skip zero-embedding neighbors
                    dists.append(1 - np.dot(embeddings[i, j], embeddings[ni, nj]))
            sensitivity[i, j] = np.mean(dists) if dists else 0
    return sensitivity


def _sensitivity_3d(embeddings):
    """Mean cosine distance to 6-neighbors (±1 along each axis)."""
    gs = embeddings.shape[0]
    gs_y = embeddings.shape[1]
    gs_z = embeddings.shape[2]
    sensitivity = np.zeros((gs, gs_y, gs_z))
    for i in range(gs):
        for j in range(gs_y):
            for k in range(gs_z):
                dists = []
                for di, dj, dk in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
                    ni, nj, nk = i + di, j + dj, k + dk
                    if 0 <= ni < gs and 0 <= nj < gs_y and 0 <= nk < gs_z:
                        dists.append(1 - np.dot(embeddings[i,j,k], embeddings[ni,nj,nk]))
                sensitivity[i, j, k] = np.mean(dists) if dists else 0
    return sensitivity


def compute_jacobian_sensitivity(latents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute 2×2 Jacobian Gram matrix spectral norm from grid-neighbor finite differences.

    Uses coarse grid-neighbor distances (NOT fine epsilon) — ridges are mesoscale features.

    Args:
        latents: (gs, gs, D) array of L2-normalized latent vectors

    Returns:
        spectral_norm: (gs, gs) — primary ridge indicator (max singular value)
        anisotropy: (gs, gs) — directional ratio (0=ridge, 1=isotropic)
    """
    gs = latents.shape[0]
    gs_b = latents.shape[1]
    da = 1.0 / (gs - 1) if gs > 1 else 1.0
    db = 1.0 / (gs_b - 1) if gs_b > 1 else 1.0

    spectral = np.zeros((gs, gs_b))
    aniso = np.zeros((gs, gs_b))

    for i in range(gs):
        for j in range(gs_b):
            # ∂f/∂α — second-order accurate everywhere (matches np.gradient)
            if i > 0 and i < gs - 1:
                # Central difference: O(h²)
                df_da = (latents[i + 1, j] - latents[i - 1, j]) / (2 * da)
            elif i == 0 and gs >= 3:
                # Forward 2nd-order: (-3f₀ + 4f₁ - f₂) / 2h
                df_da = (-3 * latents[0, j] + 4 * latents[1, j] - latents[2, j]) / (2 * da)
            elif i == gs - 1 and gs >= 3:
                # Backward 2nd-order: (3f_n - 4f_{n-1} + f_{n-2}) / 2h
                df_da = (3 * latents[-1, j] - 4 * latents[-2, j] + latents[-3, j]) / (2 * da)
            elif i < gs - 1:
                df_da = (latents[i + 1, j] - latents[i, j]) / da
            else:
                df_da = (latents[i, j] - latents[i - 1, j]) / da

            # ∂f/∂β — same treatment
            if j > 0 and j < gs_b - 1:
                df_db = (latents[i, j + 1] - latents[i, j - 1]) / (2 * db)
            elif j == 0 and gs_b >= 3:
                df_db = (-3 * latents[i, 0] + 4 * latents[i, 1] - latents[i, 2]) / (2 * db)
            elif j == gs_b - 1 and gs_b >= 3:
                df_db = (3 * latents[i, -1] - 4 * latents[i, -2] + latents[i, -3]) / (2 * db)
            elif j < gs_b - 1:
                df_db = (latents[i, j + 1] - latents[i, j]) / db
            else:
                df_db = (latents[i, j] - latents[i, j - 1]) / db

            # 2×2 Gram matrix J^T J
            a11 = np.dot(df_da, df_da)  # ||∂f/∂α||²
            a22 = np.dot(df_db, df_db)  # ||∂f/∂β||²
            a12 = np.dot(df_da, df_db)  # ⟨∂f/∂α, ∂f/∂β⟩

            # Closed-form eigenvalues for 2×2 symmetric matrix
            trace = a11 + a22
            det = a11 * a22 - a12 * a12
            disc = max(trace * trace - 4 * det, 0)
            lambda_max = (trace + np.sqrt(disc)) / 2
            lambda_min = (trace - np.sqrt(disc)) / 2

            spectral[i, j] = np.sqrt(max(lambda_max, 0))
            if lambda_max > 1e-10:
                aniso[i, j] = max(lambda_min, 0) / lambda_max

    # Clamp boundary cells to nearest interior value to eliminate edge artifacts.
    # Finite difference stencils (even 2nd-order) produce systematically different
    # values at boundaries vs interior, creating a visible border in the heatmap.
    if gs >= 4:
        spectral[0, :] = spectral[1, :]
        spectral[-1, :] = spectral[-2, :]
        spectral[:, 0] = spectral[:, 1]
        spectral[:, -1] = spectral[:, -2]
    if gs_b >= 4 and gs_b != gs:
        spectral[:, 0] = spectral[:, 1]
        spectral[:, -1] = spectral[:, -2]

    return spectral, aniso


def compute_jacobian_sensitivity_3d(latents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute 3×3 Jacobian Gram matrix spectral norm for 3D grids.

    Vectorized implementation using np.gradient (2nd-order accurate everywhere).

    Args:
        latents: (gs_a, gs_b, gs_c, D) array of L2-normalized latent vectors

    Returns:
        spectral_norm: (gs_a, gs_b, gs_c) — primary ridge indicator
        anisotropy: (gs_a, gs_b, gs_c) — min/max eigenvalue ratio
    """
    gs_a, gs_b, gs_c, D = latents.shape
    da = 1.0 / (gs_a - 1) if gs_a > 1 else 1.0
    db = 1.0 / (gs_b - 1) if gs_b > 1 else 1.0
    dc = 1.0 / (gs_c - 1) if gs_c > 1 else 1.0

    # Vectorized gradients via np.gradient (2nd-order, handles boundaries)
    # Each is (gs_a, gs_b, gs_c, D)
    df_da = np.gradient(latents, da, axis=0)
    df_db = np.gradient(latents, db, axis=1)
    df_dc = np.gradient(latents, dc, axis=2)

    # Gram matrix elements: dot products over the D dimension
    # Each is (gs_a, gs_b, gs_c)
    g00 = np.einsum('ijkd,ijkd->ijk', df_da, df_da)
    g11 = np.einsum('ijkd,ijkd->ijk', df_db, df_db)
    g22 = np.einsum('ijkd,ijkd->ijk', df_dc, df_dc)
    g01 = np.einsum('ijkd,ijkd->ijk', df_da, df_db)
    g02 = np.einsum('ijkd,ijkd->ijk', df_da, df_dc)
    g12 = np.einsum('ijkd,ijkd->ijk', df_db, df_dc)

    # Build 3×3 Gram matrices and compute eigenvalues
    # Stack into (gs_a, gs_b, gs_c, 3, 3)
    gram = np.zeros((gs_a, gs_b, gs_c, 3, 3))
    gram[..., 0, 0] = g00
    gram[..., 1, 1] = g11
    gram[..., 2, 2] = g22
    gram[..., 0, 1] = gram[..., 1, 0] = g01
    gram[..., 0, 2] = gram[..., 2, 0] = g02
    gram[..., 1, 2] = gram[..., 2, 1] = g12

    # Batched eigenvalue computation
    eigs = np.linalg.eigvalsh(gram)  # (gs_a, gs_b, gs_c, 3), sorted ascending
    lambda_max = np.maximum(eigs[..., -1], 0)
    lambda_min = np.maximum(eigs[..., 0], 0)

    spectral = np.sqrt(lambda_max)
    aniso = np.where(lambda_max > 1e-10, lambda_min / lambda_max, 0.0)

    # Boundary clamping
    if gs_a >= 4:
        spectral[0, :, :] = spectral[1, :, :]
        spectral[-1, :, :] = spectral[-2, :, :]
    if gs_b >= 4:
        spectral[:, 0, :] = spectral[:, 1, :]
        spectral[:, -1, :] = spectral[:, -2, :]
    if gs_c >= 4:
        spectral[:, :, 0] = spectral[:, :, 1]
        spectral[:, :, -1] = spectral[:, :, -2]

    return spectral, aniso


def compute_clusters(embeddings: np.ndarray, eps: float = 0.1) -> np.ndarray:
    """DBSCAN clustering (works for any spatial shape)."""
    spatial_shape = embeddings.shape[:-1]
    n_points = int(np.prod(spatial_shape))
    flat = embeddings.reshape(n_points, -1)
    cos_dist = 1 - flat @ flat.T
    np.fill_diagonal(cos_dist, 0)
    cos_dist = np.maximum(cos_dist, 0)

    db = DBSCAN(eps=eps, min_samples=3, metric='precomputed')
    labels = db.fit_predict(cos_dist)
    return labels.reshape(spatial_shape)


def extract_ridge_mesh(sensitivity_3d: np.ndarray, tau: float = 1.5):
    """Extract ridge surface from 3D sensitivity field using marching cubes.

    Returns vertices, faces as lists for JSON serialization, or None if no surface.
    """
    from skimage.measure import marching_cubes

    threshold = np.median(sensitivity_3d) * tau
    if threshold <= 0 or threshold >= sensitivity_3d.max():
        return None

    try:
        verts, faces, normals, values = marching_cubes(
            sensitivity_3d, level=threshold,
            spacing=(1.0 / max(sensitivity_3d.shape[0] - 1, 1),
                     1.0 / max(sensitivity_3d.shape[1] - 1, 1),
                     1.0 / max(sensitivity_3d.shape[2] - 1, 1)),
        )
        return {
            "vertices": verts.tolist(),
            "faces": faces.tolist(),
            "threshold": float(threshold),
            "max_sensitivity": float(sensitivity_3d.max()),
            "median_sensitivity": float(np.median(sensitivity_3d)),
        }
    except (ValueError, RuntimeError):
        return None


def classify_ridges(sensitivity: np.ndarray, tau: float = config.RIDGE_THRESHOLD_TAU):
    """Classify cells as ridge or non-ridge."""
    threshold = np.median(sensitivity) * tau
    binary_map = sensitivity > threshold
    labeled, n_components = ndimage.label(binary_map)
    non_ridge = ~binary_map
    _, n_regions = ndimage.label(non_ridge)
    return binary_map, n_components, n_regions
