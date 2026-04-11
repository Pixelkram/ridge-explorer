"""Multi-fidelity Gaussian Process ridge detection.

Combines cheap Jacobian spectral norm (available at all grid points after fast scan)
with targeted DINOv2 evaluations at adaptively selected points. A Kennedy-O'Hagan
autoregressive GP calibrates the cheap signal against expensive ground truth.

Usage:
    from backend.services.mf_gp import MFRidgeDetector

    detector = MFRidgeDetector(jacobian_map, grid_coords, valid_mask)
    seed_indices = detector.select_seed_points(n=15)
    # ... generate DINOv2 at seed_indices ...
    detector.observe(seed_indices, dino_values)

    for round in range(n_rounds):
        next_indices = detector.acquire(batch_size=10)
        # ... generate DINOv2 at next_indices ...
        detector.observe(next_indices, dino_values)

    predicted_sensitivity = detector.predict()
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.linear_model import LinearRegression


class MFRidgeDetector:
    """Multi-fidelity GP ridge detector with straddle acquisition."""

    def __init__(self, jacobian: np.ndarray, coords: np.ndarray,
                 valid_mask: np.ndarray, tau_mf: float = 1.3):
        """
        Args:
            jacobian: (N_valid,) Jacobian spectral norm values at valid grid points.
            coords: (N_valid, 2) barycentric coordinates of valid points.
            valid_mask: (gs, gs) boolean mask of valid simplex points.
            tau_mf: Ridge detection threshold multiplier on MF prediction median.
        """
        self.jacobian = jacobian.copy()
        self.coords = coords.copy()
        self.valid_mask = valid_mask
        self.n_valid = len(jacobian)
        self.tau_mf = tau_mf

        # Observed DINOv2 values
        self.observed_indices = set()
        self.observed_values = {}  # index -> DINOv2 sensitivity value

        # GP state
        self._gp = None
        self._rho = 0.0
        self._intercept = 0.0
        self._mu = None
        self._sigma = None

    def select_seed_points(self, n: int = 15, rng: np.random.RandomState = None) -> list[int]:
        """Select initial seed points for DINOv2 evaluation (random)."""
        if rng is None:
            rng = np.random.RandomState(42)
        indices = rng.choice(self.n_valid, size=min(n, self.n_valid), replace=False)
        return indices.tolist()

    def observe(self, indices: list[int], values: list[float]):
        """Record DINOv2 sensitivity observations at given indices."""
        for idx, val in zip(indices, values):
            self.observed_indices.add(idx)
            self.observed_values[idx] = val
        self._fit_gp()

    def _fit_gp(self):
        """Fit the Kennedy-O'Hagan multi-fidelity GP on current observations."""
        if len(self.observed_indices) < 3:
            return

        obs = np.array(sorted(self.observed_indices))
        y_obs = np.array([self.observed_values[i] for i in obs])
        jac_obs = self.jacobian[obs]

        # Linear calibration: DINOv2 ≈ rho * Jacobian + intercept
        lr = LinearRegression()
        lr.fit(jac_obs.reshape(-1, 1), y_obs)
        self._rho = lr.coef_[0]
        self._intercept = lr.intercept_

        # Residuals
        residuals = y_obs - (self._rho * jac_obs + self._intercept)

        # GP on residuals
        kernel = Matern(nu=2.5, length_scale=0.1) + WhiteKernel(noise_level=0.01)
        gp = GaussianProcessRegressor(
            kernel=kernel, normalize_y=True,
            n_restarts_optimizer=1, random_state=0
        )
        gp.fit(self.coords[obs], residuals)
        self._gp = gp

        # Predict everywhere
        delta_mu, delta_sigma = gp.predict(self.coords, return_std=True)
        self._mu = self._rho * self.jacobian + self._intercept + delta_mu
        self._sigma = delta_sigma

    def acquire(self, batch_size: int = 10) -> list[int]:
        """Select next batch of points using straddle heuristic.

        Targets points that are simultaneously uncertain and near the ridge threshold.
        """
        if self._mu is None:
            # No GP yet — fall back to random
            remaining = [i for i in range(self.n_valid) if i not in self.observed_indices]
            rng = np.random.RandomState(len(self.observed_indices))
            n = min(batch_size, len(remaining))
            return rng.choice(remaining, size=n, replace=False).tolist()

        candidates = np.array([i for i in range(self.n_valid) if i not in self.observed_indices])
        if len(candidates) == 0:
            return []

        n = min(batch_size, len(candidates))
        threshold = self.tau_mf * np.median(self._mu)

        # Straddle: maximize sigma - |mu - threshold|
        score = self._sigma[candidates] - np.abs(self._mu[candidates] - threshold)
        top = np.argsort(score)[-n:]
        return candidates[top].tolist()

    def predict(self) -> np.ndarray:
        """Return the current GP-predicted sensitivity map (N_valid,)."""
        if self._mu is not None:
            return self._mu.copy()
        # Fallback: return raw Jacobian (normalized to DINOv2-like range)
        return self.jacobian.copy()

    def predict_grid(self) -> np.ndarray:
        """Return prediction mapped back to (gs, gs) grid with NaN outside simplex."""
        pred = self.predict()
        gs = self.valid_mask.shape[0]
        grid = np.full((gs, gs), np.nan)
        grid[self.valid_mask] = pred
        return grid

    def get_ridge_mask(self) -> np.ndarray:
        """Return boolean (N_valid,) array of predicted ridge cells."""
        pred = self.predict()
        return pred > self.tau_mf * np.median(pred)

    def get_ridge_cells(self) -> set[tuple[int, int]]:
        """Return set of (row, col) grid indices predicted as ridge cells."""
        ridge = self.get_ridge_mask()
        valid_indices = np.argwhere(self.valid_mask)  # (N_valid, 2) of (i, j)
        return {(int(valid_indices[k, 0]), int(valid_indices[k, 1]))
                for k in range(self.n_valid) if ridge[k]}

    @property
    def n_observed(self) -> int:
        return len(self.observed_indices)

    @property
    def correlation(self) -> float:
        """Spearman rho between Jacobian and observed DINOv2 values."""
        if len(self.observed_indices) < 5:
            return 0.0
        from scipy.stats import spearmanr
        obs = sorted(self.observed_indices)
        rho, _ = spearmanr(self.jacobian[obs],
                           [self.observed_values[i] for i in obs])
        return float(rho) if not np.isnan(rho) else 0.0


def build_mf_detector(job: dict) -> MFRidgeDetector | None:
    """Create an MFRidgeDetector from a fast-scan job's Jacobian data.

    Args:
        job: A fast-scan job dict with 'sensitivity' (Jacobian spectral norm)
             and 'alphas', 'betas' arrays.

    Returns:
        MFRidgeDetector or None if the job doesn't have the required data.
    """
    sensitivity = job.get("sensitivity")
    if sensitivity is None:
        return None

    gs = job["grid_size"]
    alphas = job["alphas"]
    betas = job["betas"]

    # Build valid mask and coordinates
    AA, BB = np.meshgrid(alphas, betas, indexing="ij")
    valid_mask = (AA + BB) <= 1.02
    coords = np.column_stack([AA[valid_mask], BB[valid_mask]])
    jacobian = sensitivity[valid_mask].copy()

    # Handle NaN
    nan_mask = np.isnan(jacobian)
    if nan_mask.any():
        jacobian[nan_mask] = np.nanmedian(jacobian)

    return MFRidgeDetector(jacobian, coords, valid_mask)
