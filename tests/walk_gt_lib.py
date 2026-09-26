"""Dense ground truth for k = 3 cascades (tests/PREREG_same_ridge_trace.md).

A k = 3 simplex is 2-D, so a dense lattice rendered through the run's own generation path (POST .../render) shows
every front. Definitions (all at the walk seed and fidelity):
- lattice: w = (i, j, N - i - j) / N; one step moves 1/N between two prompts (Euclidean sqrt(2)/N).
- boundary point: along some lattice axis, its two neighbours `sep` steps away differ by 1 - cos > COS_T.
- basin: a connected component (6-neighbourhood) of non-boundary points.
- species of a point on a front: the two basins most represented on a small ring around it (unordered pair);
  None if the ring does not see two basins (not on a ground-truth front).
- crossing weights: virtual chords drawn like the cascade's (uniform anchor, isotropic direction, probes every
  STRIDE) crossing the lattice field; the share of their crossings per species is what Good-Turing estimates.
"""
from collections import Counter
import numpy as np

COS_T, STRIDE = 0.35, 0.025
E1 = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)
E2 = np.array([1.0, 1.0, -2.0]) / np.sqrt(6)
AXES = ((1, 0), (0, 1), (1, -1))
NBRS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1))


def lattice_weights(N):
    return [[i / N, j / N, (N - i - j) / N] for i in range(N + 1) for j in range(N + 1 - i)]


class Lattice:
    def __init__(self, path, sep=1):
        z = np.load(path, allow_pickle=True)
        self.w = z["weights"]
        e = z["emb"].astype(np.float64)
        self.e = e / np.linalg.norm(e, axis=1, keepdims=True)
        self.N = int(round(1 / np.min(np.abs(np.diff(np.unique(self.w[:, 0]))))))
        self.row = {(int(round(a * self.N)), int(round(b * self.N))): r for r, (a, b, _) in enumerate(self.w)}
        self.sep = sep
        self.boundary = self._boundary()
        self.labels = self._basins()

    def _at(self, i, j):
        return self.row.get((i, j))

    def _boundary(self):
        B = np.zeros(len(self.w), bool)
        for (i, j), r in self.row.items():
            for di, dj in AXES:
                for s in range(1, self.sep + 1):
                    a, b = self._at(i - s * di, j - s * dj), self._at(i + s * di, j + s * dj)
                    if a is not None and b is not None and 1 - self.e[a] @ self.e[b] > COS_T:
                        B[r] = True
        return B

    def _basins(self):
        lab = np.full(len(self.w), -1)
        cur = 0
        for (i, j), r in self.row.items():
            if self.boundary[r] or lab[r] >= 0:
                continue
            stack = [(i, j)]
            lab[r] = cur
            while stack:
                a, b = stack.pop()
                for di, dj in NBRS:
                    q = self._at(a + di, b + dj)
                    if q is not None and not self.boundary[q] and lab[q] < 0:
                        lab[q] = cur
                        stack.append((a + di, b + dj))
            cur += 1
        return lab

    def nearest(self, w):
        w = np.asarray(w, dtype=float)
        if w.min() < -1e-9:
            return None
        i, j = int(round(w[0] * self.N)), int(round(w[1] * self.N))
        i, j = max(i, 0), max(j, 0)
        while i + j > self.N:
            if w[0] * self.N - i < w[1] * self.N - j:
                i -= 1
            else:
                j -= 1
        return self._at(i, j)

    # rings out to 0.08 reach past rough patches (dev: 39/57 crossings get a region pair vs 22/57 with rings to 0.045);
    # fronts inside one region (cracks) stay None -- 18/57 on the dev field
    def species(self, w, radii=(0.03, 0.045, 0.06, 0.08), n_ang=24, min_share=0.2):
        """Unordered basin pair seen on rings around w, or None if w is not on a ground-truth front."""
        w = np.asarray(w, dtype=float)
        c = Counter()
        for rad in radii:
            for a in np.linspace(0, 2 * np.pi, n_ang, endpoint=False):
                r = self.nearest(w + rad * (np.cos(a) * E1 + np.sin(a) * E2))
                if r is not None and not self.boundary[r]:
                    c[int(self.labels[r])] += 1
        tot = sum(c.values())
        top = c.most_common(2)
        if tot < 0.3 * len(radii) * n_ang or len(top) < 2 or top[1][1] < min_share * tot:
            return None
        return frozenset((top[0][0], top[1][0]))

    def emb_at(self, w):
        r = self.nearest(w)
        return None if r is None else self.e[r]

    def chord_weights(self, n=20000, seed=0):
        """Crossings of n virtual cascade chords, per species (None = not a ground-truth front)."""
        rng = np.random.default_rng(seed)
        cnt = Counter()
        for _ in range(n):
            w0 = rng.dirichlet(np.ones(3))
            u = rng.standard_normal(3); u -= u.mean(); u /= np.linalg.norm(u)
            tpos = min(w0[i] / -u[i] for i in range(3) if u[i] < 0)
            tneg = max(-w0[i] / u[i] for i in range(3) if u[i] > 0)
            if tpos - tneg < 6 * STRIDE:
                continue
            ts = tneg + STRIDE * np.arange(int((tpos - tneg) / STRIDE) + 1)
            es = [self.emb_at(np.clip(w0 + t * u, 0, None)) for t in ts]
            for a in range(len(ts) - 1):
                if es[a] is None or es[a + 1] is None or 1 - es[a] @ es[a + 1] <= COS_T:
                    continue
                cnt[self.species(np.clip(w0 + (ts[a] + ts[a + 1]) / 2 * u, 0, None))] += 1
        return cnt
