"""Local boundary density: a kernel-smoothed Crofton S_V map read off a cascade's own chords.

The survey already pays for this. Its chords are uniform-anchor + isotropic-direction
lines through the simplex, so they sample boundary AREA fairly -- the same unbiasedness
that makes the Good-Turing coverage certificate estimable (coverage_stats). Crofton then
turns crossings per unit chord length into surface area per unit volume, and smoothing
that ratio with a Gaussian kernel instead of pooling it over the whole space turns ONE
global number into a map of WHERE the boundaries are dense.

Calibration of record: search_problem/outputs/h23_local_sv_kde (PREREG + RESULTS.md, plus
an independent re-implementation in verify/VERIFY.md). At h = 0.10 the map beats the best
possible global constant from N* = 20 oracle chords; under single-seed labels -- which is
what a real Cascade run reads -- the budget degrades 4x to N* = 80. Read that as a hard
split between two products of the same estimator:

  * RANKING (which regions are busier) is reliable early: median Spearman 0.78-0.90
    against the converged map at 20-40 single-seed chords, >= 0.90 at 80.
  * CALIBRATED VALUES in S_V units need N >= 80 single-seed chords (`calibrated_ok`).
    Below that the single-seed spurious-crossing inflation (x1.24, nearly uniform) is a
    bias no constant absorbs and the numbers are worse than the global constant.

Calibration is established at k = 4 only; the estimator itself is k-agnostic.

Absolute S_V is sampling-step dependent (h23 §d.4: the crossing count is not converged in
Delta for a partition whose boundary occupies ~26% of cells), so values compare only
within one Delta -- which is why Delta is fixed here rather than exposed.

Pure numpy, no GPU: this reads a finished survey's geometry and generates nothing.
"""
from math import pi, sqrt

import numpy as np

# Calibration of record (search_problem h23, 2026-09-30)
H_DEFAULT = 0.10        # kernel width in orthonormal tangent coords; ~4 cells of a 60-lattice at k=4
SAMPLE_DIV = 120        # chord sample stations per simplex edge -> Delta ~ half a lattice cell
N_CALIBRATED = 80       # single-seed chords before the VALUES mean anything (ranking holds from ~20)
COV_FRAC = 0.05         # kernel mass below this share of the median = no local evidence


def tangent_basis(k):
    """Orthonormal basis of the sum-zero subspace {x in R^k : sum x = 0}, as a k x (k-1) matrix.

    The simplex's own chart: an isometry on sum-zero differences, so a chord's length and
    the kernel width mean the same thing in either coordinate system, and the simplex edge
    is sqrt(2) at every k. Same construction as the h23 study (QR of I - 11^T/k).
    """
    return np.linalg.qr(np.eye(k) - np.ones((k, k)) / k)[0][:, :k - 1]


def to_tangent(w, basis=None):
    """Weight vectors (barycentric, rows) -> tangent coordinates, centred on the barycentre."""
    w = np.atleast_2d(np.asarray(w, dtype=np.float64))
    k = w.shape[1]
    B = tangent_basis(k) if basis is None else basis
    return (w - 1.0 / k) @ B


def crofton_constant(d):
    """c_d = sqrt(pi)*Gamma((d+1)/2)/Gamma(d/2): the Crofton factor turning crossings per
    unit length into (d-1)-area per unit d-volume. E69a validated c_3 = 2.002 on analytic
    spheres.

    Evaluated by the recursion c_d = c_{d-2}*(d-1)/(d-2) off c_1 = 1 and c_2 = pi/2 rather
    than by the Gammas: it is EXACT for odd d (c_3 = 2.0 to the bit, where every ordering of
    the closed form lands on 1.9999999999999998), and no Gamma overflows at large k.
    """
    c = 1.0 if d % 2 else pi / 2.0
    for j in range(3 if d % 2 else 4, d + 1, 2):
        c *= (j - 1) / (j - 2)
    return c


def _seg_dist(p, a, b):
    """Distance from each row of p to the segment [a, b] (rows of p, one segment)."""
    ab = b - a
    denom = max(float(np.dot(ab, ab)), 1e-18)
    t = np.clip((p - a) @ ab / denom, 0.0, 1.0)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


def _kernel_sums(Q, P, h, chunk=512):
    """sum_j exp(-|q - p_j|^2 / (2h^2)) for every row q of Q.

    Unnormalised on purpose: the Gaussian's constant cancels in the numerator/denominator
    ratio, exactly as in the h23 reference implementation (which also works in float32 --
    the ratio is a smooth function of two large sums, so the precision is not the limit
    here; the label noise is).
    """
    out = np.zeros(len(Q), dtype=np.float64)
    if len(P) == 0:
        return out
    inv = np.float32(-1.0 / (2.0 * h * h))
    P32 = P.astype(np.float32)
    sqP = (P32 * P32).sum(1)
    for i in range(0, len(Q), chunk):
        q = Q[i:i + chunk].astype(np.float32)
        d2 = (q * q).sum(1)[:, None] + sqP[None, :] - 2.0 * (q @ P32.T)
        np.maximum(d2, 0.0, out=d2)          # the expansion can go slightly negative
        np.multiply(d2, inv, out=d2)
        np.exp(d2, out=d2)
        out[i:i + chunk] = d2.sum(1, dtype=np.float64)
    return out


def _ranks(vals, keep):
    """Percentile rank (0-100) within the covered population; uncovered rows get 0.0.

    Ordinal ranking, ties broken by order -- the map is read as "busier than here", and the
    UI colours by this rather than by value whenever the run is short of N_CALIBRATED.
    """
    pct = np.zeros(len(vals), dtype=np.float64)
    idx = np.flatnonzero(keep)
    if len(idx) == 0:
        return pct
    order = idx[np.argsort(vals[idx], kind="stable")]
    pct[order] = 100.0 * np.arange(len(order)) / max(len(order) - 1, 1)
    return pct


def local_sv(chords, crossings, k, h=H_DEFAULT, mode="all", eval_points=None,
             cloud=0, seed=0):
    """Kernel-smoothed local S_V from a survey's chords and the crossings read off them.

    chords     -- [(w_start, w_end)] endpoints in weight space (k-vectors), as laid down.
    crossings  -- [(chord_id, position, certified)]. `position` is either a weight vector
                  or a scalar t in [0, 1] along its chord; `chord_id` may be None, in which
                  case the chord is recovered as the nearest one (crossings lie ON a chord,
                  so this is exact up to the bisection bracket).
    mode       -- "all" uses every crossing. "certified" keeps only the flagged ones in the
                  NUMERATOR while the denominator stays the full chord length, so it
                  estimates the density of certified boundary, NOT a cleaner S_V: h23's
                  own filter variant retained 0.395 of genuine crossings and ranked worse
                  than raw in 12/12 cells. Offered as a diagnostic, never as the default.
    eval_points -- extra weight vectors to read the map at (returned under "extra").
    cloud      -- if > 0, also read the map at this many Dirichlet(1,...,1) points.

    The denominator is built by walking every chord at a fixed step with NO labels -- it is
    pure geometry, costs no images, and is what makes the ratio an S_V rather than a count.
    """
    B = tangent_basis(k)
    c_d = crofton_constant(k - 1)
    delta = sqrt(2.0) / SAMPLE_DIV

    # ---- denominator: chord-length mass, sampled uniformly (no labels, no images)
    st_w, st_chord, lengths = [], [], []
    for ci, (wa, wb) in enumerate(chords):
        a = np.asarray(wa, dtype=np.float64)
        b = np.asarray(wb, dtype=np.float64)
        L = float(np.linalg.norm(b - a))
        lengths.append(L)
        if L <= 0:
            continue
        n = int(L // delta) + 1                       # floor(L/Delta)+1, as in h23
        t = np.arange(n) * (delta / L)
        st_w.append(a + t[:, None] * (b - a))
        st_chord.append(np.full(n, ci, dtype=np.int64))
    S_w = np.vstack(st_w) if st_w else np.zeros((0, k))
    S_chord = np.concatenate(st_chord) if st_chord else np.zeros(0, dtype=np.int64)
    S_t = to_tangent(S_w, B)
    lengths = np.asarray(lengths, dtype=np.float64)

    # ---- numerator: the crossings, placed on their chord
    A_t = np.stack([to_tangent(c[0], B)[0] for c in chords]) if chords else np.zeros((0, k - 1))
    Bt = np.stack([to_tangent(c[1], B)[0] for c in chords]) if chords else np.zeros((0, k - 1))
    x_w, x_cert, x_chord = [], [], []
    for cid, pos, certified in crossings:
        pos = np.asarray(pos, dtype=np.float64)
        if pos.ndim == 0:                             # scalar t along a known chord
            if cid is None or not (0 <= int(cid) < len(chords)):
                continue
            a = np.asarray(chords[int(cid)][0], dtype=np.float64)
            b = np.asarray(chords[int(cid)][1], dtype=np.float64)
            x_w.append(a + float(pos) * (b - a))
            x_chord.append(int(cid))
        else:
            x_w.append(pos)
            x_chord.append(-1 if cid is None else int(cid))
        x_cert.append(bool(certified))
    X_w = np.stack(x_w) if x_w else np.zeros((0, k))
    X_t = to_tangent(X_w, B) if len(X_w) else np.zeros((0, k - 1))
    x_cert = np.asarray(x_cert, dtype=bool)
    x_chord = np.asarray(x_chord, dtype=np.int64)
    # chords do not record which of them produced a crossing; recover it geometrically
    for i in np.flatnonzero(x_chord < 0):
        if len(chords) == 0:
            continue
        d = [float(_seg_dist(X_t[i:i + 1], A_t[j], Bt[j])[0]) for j in range(len(chords))]
        x_chord[i] = int(np.argmin(d))
    use = x_cert if mode == "certified" else np.ones(len(X_t), dtype=bool)

    # ---- global mean-of-ratios constant, from the same chords (the baseline to beat)
    n_chords = int(len(chords))
    ok = lengths > 0
    s_global = None
    if n_chords and ok.any():
        cnt = np.bincount(x_chord[use], minlength=n_chords).astype(np.float64)
        s_global = float(c_d * np.mean(cnt[ok] / lengths[ok]))

    # ---- evaluate: crossings, every chord station, an optional cloud, plus extras
    cloud_w = np.zeros((0, k))
    if cloud and cloud > 0:
        cloud_w = np.random.default_rng(seed).dirichlet(np.ones(k), size=int(cloud))
    extra_w = np.atleast_2d(np.asarray(eval_points, dtype=np.float64)) \
        if eval_points is not None and len(eval_points) else np.zeros((0, k))
    blocks = [("crossings", X_t), ("stations", S_t),
              ("cloud", to_tangent(cloud_w, B) if len(cloud_w) else np.zeros((0, k - 1))),
              ("extra", to_tangent(extra_w, B) if len(extra_w) else np.zeros((0, k - 1)))]
    Q = np.vstack([q for _, q in blocks]) if any(len(q) for _, q in blocks) \
        else np.zeros((0, k - 1))

    den = _kernel_sums(Q, S_t, h) * delta
    num = _kernel_sums(Q, X_t[use], h)
    med = float(np.median(den)) if len(den) else 0.0
    covered = den > max(COV_FRAC * med, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        vals = np.where(covered, c_d * num / np.where(den > 0, den, 1.0), 0.0)
    pct = _ranks(vals, covered)

    out = {"h": float(h), "mode": mode, "delta": float(delta), "c_d": float(c_d),
           "k": int(k), "n_chords": n_chords, "n_crossings": int(use.sum()),
           "s_global": s_global, "calibrated_ok": bool(n_chords >= N_CALIBRATED)}
    off = 0
    for name, q in blocks:
        sl = slice(off, off + len(q))
        out[name] = {"weights": (X_w if name == "crossings" else S_w if name == "stations"
                                 else cloud_w if name == "cloud" else extra_w),
                     "values": vals[sl], "ranks": pct[sl], "uncovered": ~covered[sl]}
        off += len(q)
    out["stations"]["chord"] = S_chord
    out["crossings"]["chord"] = x_chord
    out["crossings"]["certified"] = x_cert
    return out


def from_run(run_state):
    """Build (chords, crossings, k) from a finished cascade.

    Accepts either the live in-memory run (services.cascade.CascadeRun: `chords_geo` pairs
    and `Crossing` records) or a serialised CascadeStatus dict -- the shape the research
    harnesses persist (e.g. tests/walk_gt_dev/*.json, whose `status` key holds one). Runs
    are not otherwise written to disk: app.state.cascades is memory-only.

    `significant` is the certified flag: B above the run's OWN measured background p95
    (E89), which is a stronger test than the h23 filter variant and the only per-crossing
    verdict the survey produces.
    """
    d = run_state
    if isinstance(d, dict) and "status" in d and isinstance(d["status"], dict):
        d = d["status"]                              # harness wrapper around a CascadeStatus
    if isinstance(d, dict):
        chords = [(np.asarray(c["a"], dtype=np.float64), np.asarray(c["b"], dtype=np.float64))
                  for c in d.get("chords") or ()]
        crossings = [(None, np.asarray(x["weights"], dtype=np.float64),
                      bool(x.get("significant")))
                     for x in d.get("crossings") or ()]
        k = len(d.get("prompts") or ()) or (len(chords[0][0]) if chords else 0)
        return chords, crossings, int(k)
    chords = [(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))
              for a, b in getattr(run_state, "chords_geo", None) or ()]
    crossings = []
    for x in getattr(run_state, "crossings", None) or ():
        w = x.mid if x.mid is not None else (x.wa + x.wb) / 2
        crossings.append((None, np.asarray(w, dtype=np.float64), bool(x.significant)))
    k = getattr(run_state, "k", 0) or (len(chords[0][0]) if chords else 0)
    return chords, crossings, int(k)
