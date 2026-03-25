"""
Ridge detection from DINOv2 embeddings + DBSCAN clustering.

Computes:
1. Neighbor distance map (mean cosine distance to 4-neighbors)
2. DBSCAN clustering for semantic region labeling
3. Ridge classification (threshold-based)
"""
import numpy as np
from scipy import ndimage
from sklearn.cluster import DBSCAN
from backend import config


def compute_sensitivity(embeddings: np.ndarray) -> np.ndarray:
    """Compute DINOv2 neighbor distance map.

    Args:
        embeddings: (grid_size, grid_size, 768) normalized DINOv2 embeddings

    Returns:
        sensitivity: (grid_size, grid_size) mean cosine distance to 4-neighbors
    """
    gs = embeddings.shape[0]
    sensitivity = np.zeros((gs, gs))
    for i in range(gs):
        for j in range(gs):
            dists = []
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < gs and 0 <= nj < gs:
                    dists.append(1 - np.dot(embeddings[i, j], embeddings[ni, nj]))
            sensitivity[i, j] = np.mean(dists) if dists else 0
    return sensitivity


def compute_clusters(embeddings: np.ndarray, eps: float = 0.1) -> np.ndarray:
    """Compute DBSCAN clusters from DINOv2 embeddings.

    Args:
        embeddings: (grid_size, grid_size, 768) normalized DINOv2 embeddings

    Returns:
        labels: (grid_size, grid_size) cluster labels (-1 = noise/boundary)
    """
    gs = embeddings.shape[0]
    flat = embeddings.reshape(gs * gs, -1)
    cos_dist = 1 - flat @ flat.T
    np.fill_diagonal(cos_dist, 0)
    cos_dist = np.maximum(cos_dist, 0)

    db = DBSCAN(eps=eps, min_samples=3, metric='precomputed')
    labels = db.fit_predict(cos_dist)
    return labels.reshape(gs, gs)


def classify_ridges(sensitivity: np.ndarray, tau: float = config.RIDGE_THRESHOLD_TAU):
    """Classify grid cells as ridge or non-ridge.

    Returns:
        binary_map: bool array, True = ridge
        n_components: number of connected ridge components
        n_clusters: estimated number of semantic regions
    """
    threshold = np.median(sensitivity) * tau
    binary_map = sensitivity > threshold
    labeled, n_components = ndimage.label(binary_map)

    # Enclosed regions (non-ridge connected components)
    non_ridge = ~binary_map
    _, n_regions = ndimage.label(non_ridge)

    return binary_map, n_components, n_regions
