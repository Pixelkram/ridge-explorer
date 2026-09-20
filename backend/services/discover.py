"""Discovery: the procedure our ablations actually support.

WHAT THIS IS, AND WHY IT IS SO MUCH SIMPLER THAN THE HIKER. The hiker (services/hiker.py)
builds a sensitivity field, extracts its ridge network as a graph, walks arcs, and carries
the walked exit into the next simplex as an embedding coordinate. Every one of those
components was ablated in Aug 2026 and none of them survived:

  * the ridge graph did not beat one-line argmax(sensitivity)      (E32, all p > 0.15)
  * guidance did not beat sampling well, once the survey it needs
    was charged against the same budget                            (E30 p = 0.192; E32 no
                                                                    crossover at 400-1200)
  * carrying the exit coordinate did not beat restarting at four
    RANDOM fresh triples                                           (E35, p = 0.967)

What did survive is the part nobody was looking at:

  * a fixed simplex EXHAUSTS. Its diversity decays -0.040 within one 960-image run, and
    splitting the same budget across four simplices raises diversity 0.742 -> 0.913
    (E34/E35, p = 0.0000). Leaving matters; how you leave does not.
  * HOW you sample inside a simplex dominates everything else. Coherence rises with
    commitment -- how concentrated the mixing weights are -- and saturates near 0.60, while
    diversity peaks near 0.50 (E28: within-set Spearman +0.94 in all three prompt sets;
    E29: the optimum is stable at k = 3, 4, 5, 8).
  * prompt spread has an interior optimum at mean pairwise CLIP-text similarity ~0.56
    (E10: novelty rho = -0.817, coherence +0.634, product peaks there).

So the loop is: pick k prompts at the right spread, sample the simplex at the right
commitment, and move on before it saturates. No field, no graph, no walk.

THE TRAP THIS EXISTS TO AVOID. Sampling a simplex "uniformly" is not neutral. Under
Dirichlet(1) the expected maximum weight falls as k grows -- 0.611 at k=3, 0.521 at k=4,
0.339 at k=8 -- so adding prompts silently slides you out of the good band and into the
regime where output is texture rather than subject matter. This module never samples
uniformly: it solves for the alpha that puts E[max weight] where you asked, at whatever k
you chose.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import numpy as np

from backend import config

# ---------------------------------------------------------------------------
# Defaults, each traceable to a measurement. See DEFAULTS_PROVENANCE below for
# what the UI shows the user.
# ---------------------------------------------------------------------------
# TARGET_SIM. E10 claimed "novelty x coherence peaks at 0.56"; that reasoning does not
# reproduce -- its own bins give a product falling monotonically with similarity. E36
# (2026-08-16) re-measured it properly: 5 spread levels x 3 prompt sets x 3 seeds, scored
# on USABLE yield (coherent AND novel) rather than novelty x mean coherence.
#
#     target sim   0.35   0.45   0.55   0.65   0.75
#     usable      54.4%  23.8%  48.8%   7.9%   0.2%
#     coherent      85%    84%    97%    96%    98%
#
# What that settles:
#   * there is NO peak at 0.56. Usable yield declines with similarity overall
#     (Spearman -0.807, p < 1e-4) and the response is non-monotone, with a reproducible
#     trough at 0.45 (all three prompt sets, between-level SD 21.5 pp vs within-level
#     3.7 pp) that we cannot explain.
#   * 0.55 is nonetheless NOT beaten: 0.35 scores 54.4% against 48.8%, p = 0.113.
#   * the real, large effect is the CEILING. At 0.65 usable yield is 7.9% and at 0.75 it
#     is 0.2% -- near-perfectly coherent images of exactly what you asked for.
#   * the mechanism E10's mean-coherence term was standing in for is confirmed: the
#     FRACTION clearing the gate rises with similarity (rho +0.450, p = 0.002).
#
# So 0.55 stays as a defensible default, its stated justification is replaced, and the
# actionable rule is the ceiling rather than an optimum: stay below 0.60.
TARGET_SIM = 0.55
COMMITMENT = 0.55          # E28/E29: coherence saturates ~0.60, diversity peaks ~0.50
BATCH = 300                # E32/E34: no decay at 240/simplex; -0.040 by 960
DEFAULT_K = 3              # E29: k buys variety, not coherence; 3-4 is the sweet default

DEFAULTS_PROVENANCE = {
    "k": "3-4. At matched commitment more prompts buy DIVERSITY (partial rho +0.898) at a "
         "small coherence cost (-0.205), so raise k when you want variety. [E29]",
    "commitment": "0.50-0.60. Coherence rises with commitment and saturates near 0.60; "
                  "diversity peaks near 0.50. Optimum stable at k=3,4,5,8. [E28, E29]",
    "target_sim": "STAY BELOW 0.60 -- that is the finding, not an optimum. Usable yield "
                  "(coherent AND novel) by spread level: 54.4% at 0.35, 23.8% at 0.45, "
                  "48.8% at 0.55, then it falls off a cliff: 7.9% at 0.65 and 0.2% at 0.75, "
                  "where output is near-perfectly coherent and shows exactly what you asked "
                  "for. E10's claimed peak at 0.56 does not reproduce; the trend is downward "
                  "(rho -0.807) with an unexplained trough at 0.45. 0.35 and 0.55 are "
                  "statistically indistinguishable (p = 0.113), so 0.55 is kept as a "
                  "default on grounds of coherence (97% vs 85% of images clearing the "
                  "gate). [E10 superseded by E36, 2026-08-16]",
    "batch": "200-400. A simplex exhausts: diversity decays -0.040 over 960 images in one "
             "simplex, and 240/simplex showed no decay. [E32, E34]",
    "steps": "8-16, default 8. 4 steps is NOT converged (3.2x less high-frequency detail "
             "than 16); the knee is 8. [E24]",
    "seed_count": "1-3. A single-seed sensitivity FIELD is 77% signal / 23% noise and "
                  "averaging recovers it at the Spearman-Brown rate, n=3 being the knee. "
                  "This loop reads no field, so seeds here are simply more samples. [E26]",
    "guidance_scale": "UNSWEPT. Left at the pipeline default; we have no measurement.",
}


def solve_alpha(k: int, target: float, n: int = 60_000, seed: int = 0,
                lo: float = 0.02, hi: float = 60.0, tol: float = 1e-4) -> float:
    """alpha such that E[max of Dirichlet(alpha * 1_k)] == target.

    E[max weight] decreases monotonically in alpha (large alpha -> balanced -> 1/k; small
    alpha -> one component dominates -> 1), so a bisection is enough. A fixed RNG stream
    keeps the result deterministic for a given (k, target), which matters because the value
    is reported to the user and stored with the run.
    """
    floor = 1.0 / k
    if target <= floor:
        raise ValueError(
            f"commitment {target:.3f} is at or below the k={k} floor of {floor:.3f}; "
            f"a {k}-prompt mixture cannot concentrate less than 1/k")
    rng = np.random.default_rng(seed)
    f = lambda a: float(rng.dirichlet(np.full(k, a), size=n).max(1).mean())
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if f(mid) > target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def choose_prompts(pool, k, target_sim, rng, text_enc, tries=4000, exclude=()):
    """Random-restart search for a k-subset whose mean pairwise CLIP-text similarity is
    closest to `target_sim`.

    Deliberately NOT nearest-neighbour retrieval: E10 measured novelty falling with
    similarity (rho = -0.817), so an embedding index used the obvious way walks straight
    into the collapsed end of the curve. Selection targets a BAND, not a minimum distance.

    `exclude` holds prompts already spent by earlier legs, so a long run keeps finding
    fresh material rather than recycling the same corpus subset; it falls back to the full
    pool once the corpus is exhausted rather than failing the run.
    """
    avail = [p for p in pool if p not in exclude]
    if len(avail) < k:
        avail = list(pool)
    P = text_enc(avail)
    best, best_d = None, None
    for _ in range(tries):
        idx = rng.sample(range(len(avail)), k)
        M = P[idx] @ P[idx].T
        iu = np.triu_indices(k, 1)
        ms = float(M[iu].mean())
        d = abs(ms - target_sim)
        if best_d is None or d < best_d:
            best, best_d, best_ms = [avail[i] for i in idx], d, ms
    return best, best_ms


@dataclass
class Simplex:
    """One leg of the loop: k prompts, the alpha that hits the requested commitment, and
    the points drawn from it."""
    index: int
    prompts: list
    mean_sim: float
    alpha: float
    n_points: int
    weights: list = field(default_factory=list)
    done: int = 0
    blend_frac: float = float("nan")   # fraction of draws that are real mixtures (max<0.5)
    scout_idx: list = field(default_factory=list)   # lattice cells, if this leg scouts
    scout_n: int = 0                                # survey cost charged to this leg
    aimed: bool = False                             # True once points were re-aimed
    aim_lo: int = -1                                # global index range of the AIMED
    aim_hi: int = -1                                # images, so the UI can show only them
    # The two numbers that predict whether a leg is worth looking at. Nothing measured
    # these before, and they are what separates the run that produced winged giraffes
    # (7:1, 1.0) from the ablations that found no benefit (1.1:1, 9.0).
    draws_per_cell: float = float("nan")
    survey_ratio: float = float("nan")


@dataclass
class DiscoverRun:
    run_id: str
    k: int
    commitment: float
    target_sim: float
    batch: int
    total: int
    seed: int
    steps: int
    height: int
    width: int
    guidance_scale: float
    pool: list
    # Reject vertex-dominated draws so a mixture is the typical sample, not the exception.
    commit_band: tuple | None = None
    min_weight: float = 0.0
    # Ridge targeting. scout_gs > 0 spends a triangular lattice of that size measuring the
    # sensitivity field first, then draws the leg's points near its top-decile cells.
    # Costs the survey; buys hybrids instead of source prompts (10/10 triplets, 2026-08-16).
    scout_gs: int = 0
    top_frac: float = 0.10
    # Jitter applied to an aimed cell's barycentric coordinate, in CELL widths. The ridge
    # is thin: on a coarse lattice a half-cell nudge can land back inside a basin, which
    # is the suspected reason aimed harvests still contain clean source prompts. 0 = the
    # measured cell exactly.
    aim_jitter: float = 0.5
    # How far below zero a mixing weight may go. 0 = inside the prompt hull only, which
    # is a hard ceiling: 26 of the top 36 sensitivity cells of a 400-cell grid lie OUTSIDE
    # the hull (redundancy 0.735 vs 0.865 for the best in-hull cells). Raises the failure
    # rate too -- pair with the coherence gate.
    extrapolate: float = 0.0
    prompts_override: list | None = None
    status: str = "running"
    error: str | None = None
    simplices: list = field(default_factory=list)
    used_prompts: list = field(default_factory=list)
    generated: int = 0
    embeddings: dict = field(default_factory=dict)   # global index -> np.ndarray
    notes: list = field(default_factory=list)
    thumbs: object = None                            # ThumbnailStore, set by the router

    # ---- derived metrics, recomputed on demand -------------------------------
    def diversity(self, since: int = 0) -> float:
        """Mean pairwise DINOv2 distance over the harvest (optionally a suffix).

        This is the measure the ablations were decided on, and it is reference-free -- it
        depends only on the generated images, never on the prompts -- which is why it is
        the one reported live.
        """
        E = [v for k_, v in sorted(self.embeddings.items()) if k_ >= since]
        if len(E) < 2:
            return float("nan")
        A = np.stack(E)
        A = A / np.linalg.norm(A, axis=1, keepdims=True)
        G = A @ A.T
        iu = np.triu_indices(len(A), 1)
        return float(1.0 - G[iu].mean())

    def redundancy(self, since: int = 0) -> float:
        """Mean cosine to the NEAREST OTHER image in the harvest. Lower is better.

        This exists because mean pairwise distance -- the number this panel used to report
        on its own -- cannot tell two opposite outcomes apart. Three source prompts rendered
        separately form a few tight, far-apart clusters and score ~0.74; genuinely varied
        hybrids also score ~0.75. A run that returned nothing but wheat fields, clovers and
        red hearts passed every check at 0.740 diversity, and only looking at the images
        caught it.

        Redundancy catches it directly: if every image has a near-twin the score approaches
        1.0 no matter how far apart the clusters are. Measured over 10 prompt triplets,
        ridge-targeted sets scored 0.766 against 0.912 for balanced random sampling.

        Note it is size-dependent -- a bigger set has closer neighbours -- so it is only
        comparable between runs of similar length.
        """
        E = [v for k_, v in sorted(self.embeddings.items()) if k_ >= since]
        if len(E) < 2:
            return float("nan")
        A = np.stack(E)
        A = A / np.linalg.norm(A, axis=1, keepdims=True)
        G = A @ A.T
        np.fill_diagonal(G, -2.0)
        return float(G.max(1).mean())

    def per_simplex_diversity(self) -> list:
        """Diversity within each simplex's own slice of the index space.

        The slice is [start, start + n_points), NOT [start, start + done): points are
        sharded across the GPUs, so they arrive out of order and `done` is a COUNT, not a
        high-water mark. Using it as a bound sampled whichever indices happened to be low
        and reported a diversity for a set that was never generated together.
        """
        out, start = [], 0
        for s in self.simplices:
            E = [self.embeddings[i] for i in range(start, start + s.n_points)
                 if i in self.embeddings]
            if len(E) < 2:
                out.append(float("nan"))
            else:
                A = np.stack(E); A = A / np.linalg.norm(A, axis=1, keepdims=True)
                G = A @ A.T; iu = np.triu_indices(len(A), 1)
                out.append(float(1.0 - G[iu].mean()))
            start += s.n_points
        return out


def sample_weights(k, alpha, n, seed, band=None, min_weight=0.0, oversample=40,
                   extrapolate=0.0):
    """Draw n weight vectors over k prompts, optionally reaching OUTSIDE the prompt hull.

    WHY NEGATIVE WEIGHTS. A Dirichlet draw is non-negative by construction, so it can only
    ever produce points inside the convex hull of the k prompts. Measured on a 400-cell
    grid over the giraffe/airplane/unicorn triple, that is a hard ceiling on quality:

        top-36 cells by sensitivity   26 of 36 lie OUTSIDE the hull   redundancy 0.735
        best-36 cells INSIDE the hull                                 redundancy 0.865

    The striking material -- winged plane-creatures, bird-planes, pegasus forms -- is at
    negative weights, where the model is extrapolating past every prompt it was given. No
    amount of commitment banding, selectivity or jitter reaches it, which is why every
    in-hull variant of this sampler plateaued around 0.84.

    `extrapolate` is how far below zero a weight may go (0 = the old in-hull behaviour,
    0.5 = weights down to -0.5). Weights are still normalised to sum to 1, so the mixture
    stays affine and one prompt is effectively SUBTRACTED from the others.

    THIS IS NOT FREE. Section 3.4 measured the annulus outside the hull as significantly
    LESS novel on a coherence-gated basis (-10.5 pp on 3/3 triplets), and the top-sensitivity
    panel visibly contains washed-out gradients alongside its best images. Out-of-hull
    sampling raises both the ceiling and the failure rate, so it is meant to be paired with
    a coherence gate rather than used raw.
    """
    rng = np.random.default_rng(seed)
    if extrapolate > 0:
        # Uniform over a box that extends below zero, then renormalise. Rejection keeps
        # the draw from degenerating: without a floor on the sum the normaliser can pass
        # through zero and send a weight to +-inf.
        keep = np.empty((0, k))
        for _ in range(oversample):
            W = rng.uniform(-extrapolate, 1.0, size=(max(n * 4, 256), k))
            ssum = W.sum(1)
            ok = W[np.abs(ssum) > 0.25]
            if len(ok):
                ok = ok / ok.sum(1, keepdims=True)
                # keep draws that actually leave the hull, else this is just noisy Dirichlet,
                # but not deeper than MAX_DEPTH: past ~0.6 the images stop having subjects
                # (40% empty beyond depth 0.8) and the sensitivity field starts rewarding
                # that breakdown rather than semantics.
                d = -ok.min(1)
                ok = ok[(d > 0) & (d <= MAX_DEPTH)]
            keep = np.concatenate([keep, ok], 0) if len(keep) else ok
            if len(keep) >= n:
                return keep[:n]
        if len(keep) == 0:
            return rng.dirichlet(np.full(k, alpha), size=n)
        reps = int(np.ceil(n / len(keep)))
        return np.concatenate([keep] * reps, 0)[:n]

    if band is None and min_weight <= 0:
        return rng.dirichlet(np.full(k, alpha), size=n)
    lo, hi = band if band else (0.0, 1.0)
    keep = np.empty((0, k))
    for _ in range(oversample):
        W = rng.dirichlet(np.full(k, alpha), size=max(n * 4, 256))
        mx, mn = W.max(1), W.min(1)
        ok = W[(mx >= lo) & (mx <= hi) & (mn >= min_weight)]
        keep = np.concatenate([keep, ok], 0) if len(keep) else ok
        if len(keep) >= n:
            return keep[:n]
    if len(keep) == 0:
        return rng.dirichlet(np.full(k, alpha), size=n)
    reps = int(np.ceil(n / len(keep)))
    return np.concatenate([keep] * reps, 0)[:n]


# How far outside the prompt hull the survey may reach.
#
# RETRACTED CAP. This was 0.6, derived from a single prompt triple where structure was flat
# to depth 0.6 and 40% of images past 0.8 were empty. E38 (10 triplets, 4,000 cells) does
# not reproduce any of it: pooled Spearman(depth, structure) = +0.037, and the empty rate is
# 0% at EVERY depth band including 0.8+. The deep cells are not degraded -- on one triplet
# they are log piles, on another seascapes -- they are coherent images that have collapsed
# into a PRIOR ATTRACTOR, which is the failure mode E11/E15 already documented, not an
# off-manifold breakdown.
#
# 1.0 admits the whole square, matching /api/grid. Callers who want a tighter bound should
# set it per prompt set, because where the ridge sits relative to the hull turns out to be
# strongly prompt-dependent: 26 of 36 top cells outside for giraffe/airplane/unicorn, 0 of
# 36 for two other triplets, 19% averaged over ten.
MAX_DEPTH = 1.0


def depth_of(w):
    """How far outside the prompt hull a weight vector sits: 0 inside, else |most negative|."""
    return float(max(0.0, -np.min(w)))


def bary(i, j, gs, extrapolate=0.0):
    """Barycentric coordinate of a lattice cell.

    When extrapolating, alpha and beta each span [0, 1] over the SQUARE lattice, so the
    third weight 1 - a - b runs down to -1. That is deliberately the exact parameterisation
    the /api/grid endpoint uses, because that is the grid on which 26 of the top 36
    sensitivity cells were found to lie outside the hull.
    """
    a, b = i / (gs - 1), j / (gs - 1)
    return np.array([1.0 - a - b, a, b])


def lattice(gs, extrapolate=0.0):
    """Lattice over the simplex, optionally extended past its edges.

    With extrapolate = 0 this is the triangular in-hull lattice. With extrapolate > 0 it is
    the full square, exactly as the grid endpoint uses -- which is how the ridge was found
    to sit OUTSIDE the hull in the first place: 26 of the top 36 sensitivity cells of a
    400-cell square grid had a negative third weight, and a triangular lattice cannot see
    any of them.

    Cells deeper than MAX_DEPTH outside the hull are dropped; see `depth_of`.
    """
    if extrapolate > 0:
        cells = [(i, j) for i in range(gs) for j in range(gs)]
        return [c for c in cells if depth_of(bary(*c, gs, extrapolate)) <= MAX_DEPTH]
    return [(i, j) for i in range(gs) for j in range(gs - i)]


def sensitivity_field(emb, idx, gs):
    """Mean DINOv2 distance to lattice neighbours -- the ridge signal.

    High sensitivity marks conditioning points where a small move changes the OUTPUT a
    lot, i.e. where the model's own semantic basins meet. Those cells are where the
    hybrids live: measured over 10 prompt triplets, the top-decile cells of this field
    gave redundancy 0.766 against 0.912 for balanced random sampling from the same
    simplex, and beat it on all 10.
    """
    pos = {c: n for n, c in enumerate(idx)}
    S = {}
    for (i, j) in idx:
        n = pos[(i, j)]
        ds = []
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            m = pos.get((i + di, j + dj))
            if m is not None and n < len(emb) and m < len(emb):
                ds.append(1.0 - float(emb[n] @ emb[m]))
        if ds:
            S[(i, j)] = float(np.mean(ds))
    return S


def aim_points(S, idx, gs, n, rng, top_frac=0.10, jitter_scale=0.5, extrapolate=0.0):
    """Generate at the n most sensitive cells, ONE draw each.

    SELECTIVITY IS THE ACTIVE INGREDIENT, and it is easy to spend and then throw away.
    Three runs of the same method differ only in how many draws each selected cell got:

        survey:aimed   draws/cell   outcome
        7:1            1.0          winged giraffes, plane-birds -- 10/10 triplets
        1.6:1          6.5          modest; some hybrids, mostly source prompts
        1.1:1          9.0          no measurable benefit at all (E30, E32)

    Taking the top decile and sampling each cell once IS a 10:1 survey-to-output ratio, by
    construction -- 90% of what you measured is discarded. That is the real price of ridge
    targeting, and E30/E32 failed precisely because they paid for the survey and then
    re-drew a handful of cells to fill the budget, collapsing the selectivity they had
    bought. So this returns AT MOST one point per cell and lets the caller get fewer images
    rather than worse ones.

    Jitter is kept small and only breaks the exact lattice coordinate; it is not a way to
    manufacture extra samples from the same cell.
    """
    if not S:
        return None
    cells = sorted(S, key=lambda c: -S[c])
    k = max(1, int(round(len(cells) * top_frac)))
    pool = cells[:k]
    take = min(n, len(pool))          # never re-draw a cell to pad the batch
    out = []
    for m in range(take):
        y, x = pool[m]
        w = bary(y, x, gs, extrapolate) + rng.normal(0, jitter_scale / (gs - 1), size=3)
        # No clipping when extrapolating: clipping to positive is exactly what would put
        # the point back inside the hull and undo the aim.
        if extrapolate <= 0:
            w = np.clip(w, 1e-4, None)
        out.append(w / w.sum())
    return np.stack(out)


def plan_simplex(run: DiscoverRun, rng, text_enc) -> Simplex:
    """Draw the next leg: fresh prompts at the target spread, alpha for the commitment."""
    if run.prompts_override:
        prompts = list(run.prompts_override)
        P = text_enc(prompts)
        M = P @ P.T
        iu = np.triu_indices(len(prompts), 1)
        ms = float(M[iu].mean())
    else:
        prompts, ms = choose_prompts(run.pool, run.k, run.target_sim, rng, text_enc,
                                     exclude=run.used_prompts)
    alpha = solve_alpha(run.k, run.commitment)
    n = min(run.batch, run.total - run.generated)
    idx = len(run.simplices)
    remaining = run.total - run.generated
    # Only start a scouting leg if the remaining budget can actually pay for the survey
    # AND leave something to aim with. Without this the scout path queued its full lattice
    # regardless of budget: a run with 14 images left still launched a 351-cell survey and
    # overshot its total by 30%. A run that cannot afford another survey should finish.
    scout_cells = len(lattice(run.scout_gs, run.extrapolate)) if run.scout_gs else 0
    can_scout = (run.scout_gs and run.k == 3
                 and remaining >= scout_cells + max(1, int(round(scout_cells * run.top_frac))))
    if can_scout:
        # Ridge targeting: spend the first part of the leg measuring the field on a
        # lattice, then aim the rest at its steepest cells. The survey is charged to this
        # leg's budget -- a method that needs a map is not free, and E30 showed that
        # accounting is what decides whether guidance pays.
        cells = lattice(run.scout_gs, run.extrapolate)
        W = np.stack([bary(i, j, run.scout_gs, run.extrapolate) for (i, j) in cells])
        s = Simplex(index=idx, prompts=prompts, mean_sim=ms, alpha=alpha,
                    n_points=len(cells), weights=[w.tolist() for w in W])
        s.scout_idx = cells
        s.scout_n = len(cells)
        s.blend_frac = float((W.max(1) < 0.5).mean())
        run.used_prompts.extend(prompts)
        run.simplices.append(s)
        return s
    W = sample_weights(run.k, alpha, n,
                       seed=abs(hash((run.seed, idx))) % (2**32),
                       band=run.commit_band, min_weight=run.min_weight,
                       extrapolate=run.extrapolate)
    s = Simplex(index=len(run.simplices), prompts=prompts, mean_sim=ms,
                alpha=alpha, n_points=n, weights=[w.tolist() for w in W])
    mx = W.max(1)
    s.blend_frac = float((mx < 0.5).mean())      # how much of this leg is a real mixture
    run.used_prompts.extend(prompts)
    run.simplices.append(s)
    return s
