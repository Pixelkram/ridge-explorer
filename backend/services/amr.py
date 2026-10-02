"""AMR runs: nested barycentric lattices, refined only where the field changes.

The octree/adaptive-mesh-refinement sampler validated in search_problem h25f
(`outputs/h25_pathtracing/f_amr/RESULTS.md`). Instead of laying chords through the
simplex, it tiles the (k-1)-simplex with a barycentric lattice of n points per edge,
evaluates every cell, and then REFINES only the neighbourhood of the cells whose images
disagree -- the same pairwise crossing detector the Cascade uses (`COS_T` on DINOv2
cosine distance), read across lattice adjacency instead of along a chord. The levels are
nested (each divides the next), so a cell evaluated coarse is never re-probed, and each
level is ONE batched evaluate on the GPU pool.

Verdict of record, and the reason for the hard k cap below:

  * AMR PAYS AT k=4. At the 60/edge operating point it reached the full-lattice ceiling
    on every metric (108 true boundary pairs, area 1.0, M2 1.0) at 55 % of exhaustive
    cost, and chords needed a median 7.05x that cost to match its M2.
  * AMR DOES NOT SCALE. A codim-1 boundary still has O(phi * C(n+d,d) * factor^d)
    boundary-adjacent cells, and factor^d with d = 8 is 256 per flagged cell: the
    projected k=9 ladder is 4.785e9 probes, five orders of magnitude past the budget of
    record. Already k=5 misses it by ~8x.

So this module supports k in {3, 4} and refuses k >= 5 with the projected probe count
(`projected_probes`), rather than letting a request tie up the pool for a week.

Nothing here is a second crossing detector: `_cosd` and `COS_T` are imported from the
Cascade, so a boundary means the same thing in both surfaces.

Note for later (NOT implemented): the detected boundary edges are already (position,
divergence) pairs on known-length lattice segments, which is exactly the numerator
`services/local_sv.py` wants -- an AMR run could feed the kernel-smoothed Crofton S_V map
by passing its edges as crossings and its evaluated adjacency as the chord-length mass.
"""
from dataclasses import dataclass, field
import itertools
import math
import threading

import numpy as np

from backend.services import cascade as cs
from backend.services.cascade import COS_T, _cosd

# ---- accounting of record (h25f) ------------------------------------------------------
# One cheap probe = half an image: the lattice is read at `probe_steps` (4 of 8 denoising
# steps), the convention every cost in the h25f tables is quoted in. A run that probes at
# full fidelity (probe_steps None) pays a whole image per cell and says so.
PROBE_COST = 0.5
# Boundary-adjacent cell fraction phi_L, MEASURED at k=4 (h25f section 3). Used only to
# project the cost of a refused k >= 5 request; assumption A1 (phi is k-independent) is the
# study's own, and it is optimistic -- E70 measures the branching factor rising 4.72 -> 8.47
# from k=4 to k=9, so a projection built on this is a lower bound.
PHI_K4 = {5: 0.7679, 15: 0.5159, 30: 0.3783, 60: 0.2598}
# Supported arity. k=3 and k=4 only; see the module docstring.
K_MAX = 4
# Biggest lattice the finest level may ask for. 39,711 cells is k=4 at 60/edge -- the
# study's own finest ladder step and its measured ceiling -- so this admits every operating
# point h25f validated and refuses the (legal-looking) parameter combinations beyond it,
# which run into millions of cells and would hold the shared pool for days.
MAX_CELLS = 40000


# ---- pure lattice helpers -------------------------------------------------------------

def lattice_counts(k, n):
    """Integer compositions of n into k parts, in combinations_with_replacement order.

    Row i of the result is the multiplicity vector of the i-th multiset of k symbols of
    size n, i.e. exactly the enumeration `itertools.combinations_with_replacement` gives
    -- fixed, so a point's index means the same thing in every helper here and in the
    h25f study, which enumerated its lattices the same way.
    """
    return np.array([np.bincount(c, minlength=k)
                     for c in itertools.combinations_with_replacement(range(k), int(n))],
                    dtype=np.int64)


def lattice_points(k, n):
    """The barycentric lattice with n points per simplex edge, as (m, k) weight vectors.

    weights = counts / n, so every row sums to 1 exactly in the rational sense (and to
    within round-off in float64). m = C(n+k-1, k-1).
    """
    return lattice_counts(k, n).astype(np.float64) / float(n)


def lattice_spacing(n):
    """Euclidean weight-space distance of one unit lattice move (+e_i -e_j) at level n.

    The simplex edge is sqrt(2) in weight space at every k (two coordinates change by 1),
    so one cell is sqrt(2)/n. This is the unit `r_ref` is measured in.
    """
    return math.sqrt(2.0) / float(n)


def neighbours(k, n):
    """Unit-move adjacency of the level-n lattice, as sorted (i, j) index pairs with i < j.

    Two cells are adjacent when one unit of weight moves from coordinate i to coordinate j
    (+e_j -e_i): the lattice's own nearest-neighbour graph, and the pairs the boundary
    detector compares. Undirected, each pair kept once.
    """
    comps = lattice_counts(k, n)
    at = {tuple(int(v) for v in c): i for i, c in enumerate(comps)}
    out = set()
    for i, c in enumerate(comps):
        for a in range(k):
            if c[a] == 0:
                continue
            for b in range(k):
                if b == a:
                    continue
                c2 = c.copy()
                c2[a] -= 1
                c2[b] += 1
                j = at.get(tuple(int(v) for v in c2))
                if j is not None and i < j:
                    out.add((i, j))
    return sorted(out)


def n_pairs(k, n):
    """Analytic count of the pairs `neighbours(k, n)` returns.

    Every unit move out of a cell lands on another cell of the same lattice (the sum is
    preserved and no coordinate goes negative), so the ordered degree of a cell is
    (nonzero coordinates) x (k-1); a cell with j nonzero coordinates is one of
    C(k, j) * C(n-1, j-1) of them. Halve for undirected pairs. 210 at k=4, n=5 -- the
    count h25f reports for its coarsest level.
    """
    tot = sum(j * math.comb(k, j) * math.comb(n - 1, j - 1)
              for j in range(1, min(k, n) + 1))
    return (k - 1) * tot // 2


def nested_schedule(base, levels, factor):
    """The refinement ladder base, base*factor, ... -- `levels` entries.

    e.g. nested_schedule(5, 4, 2) -> [5, 10, 20, 40]. Every level divides the next, which
    is what makes the lattices NESTED: a level-n point is also a level-(n*factor) point, so
    it is evaluated once and reused at every finer level. The divisibility is checked
    rather than assumed -- a ladder that is not nested would re-probe the whole coarse grid
    at every level and quietly double the bill.
    """
    base, levels, factor = int(base), int(levels), int(factor)
    if base < 1:
        raise ValueError(f"base must be >= 1, got {base}")
    if levels < 1:
        raise ValueError(f"levels must be >= 1, got {levels}")
    if factor < 2:
        raise ValueError(f"factor must be >= 2 (1 would not refine), got {factor}")
    sched = [base * factor ** i for i in range(levels)]
    for a, b in zip(sched, sched[1:]):
        if b % a:
            raise ValueError(f"ladder {sched} is not nested: {a} does not divide {b}")
    return sched


def refine_set(coarse_pts, boundary_edges, fine_n, r_ref):
    """Which cells of the level-`fine_n` lattice the next pass evaluates: a boolean mask.

    A fine cell is selected when it lies within `r_ref` COARSE cells of the midpoint of a
    detected boundary edge, measured in weight space and expressed in units of the coarse
    unit move (`lattice_spacing`) -- the barycentric unit-move distance. r_ref = 1 is the
    setting of record: on the exact k=4 oracle it lost nothing against r_ref = 2 (same 108
    pairs, same M2 1.0) for 22 % fewer probes.

    The coarse spacing is read off the edges themselves rather than passed in: every unit
    move at one level has the same length, so any boundary edge measures it exactly, and
    there is no second place for the level to be stated wrongly.

    With no boundary edges the mask is empty -- a level that found nothing refines nothing,
    which is the honest answer and not an error (the caller still carries the already
    evaluated cells forward; nesting is its job, not this helper's).
    """
    coarse = np.asarray(coarse_pts, dtype=np.float64)
    k = coarse.shape[1]
    fine = lattice_points(k, fine_n)
    if not len(boundary_edges):
        return np.zeros(len(fine), dtype=bool)
    ij = np.asarray([(e[0], e[1]) for e in boundary_edges], dtype=np.int64)
    mids = 0.5 * (coarse[ij[:, 0]] + coarse[ij[:, 1]])
    # the coarse unit move, measured on the edges we were handed
    step = float(np.median(np.linalg.norm(coarse[ij[:, 0]] - coarse[ij[:, 1]], axis=1)))
    rad = float(r_ref) * step * (1.0 + 1e-9)       # the epsilon admits the exact-radius tie
    from scipy.spatial import cKDTree              # local: scipy is a heavy import
    dmin = cKDTree(mids).query(fine, workers=-1)[0]
    return dmin <= rad


def detect_edges(embeddings, pairs, threshold=COS_T):
    """Boundary edges among adjacent evaluated cells: [(i, j, divergence)], strongest first.

    `embeddings` is indexed by cell (None where the cell was never evaluated or its probe
    never arrived), `pairs` is the level's adjacency. An edge is a pair whose two images
    differ by more than `threshold` in DINOv2 cosine distance -- the Cascade's pairwise
    crossing detector (`_cosd`, `COS_T`, protocol of record), read across a lattice edge
    instead of along a chord stride. A pair with a missing end is evidence of nothing and
    is skipped rather than counted as quiet.
    """
    out = []
    for i, j in pairs:
        ea, eb = embeddings[i], embeddings[j]
        if ea is None or eb is None:
            continue
        d = _cosd(ea, eb)
        if d > threshold:
            out.append((int(i), int(j), float(d)))
    # strongest first, ties by index, so the order a run records is deterministic
    return sorted(out, key=lambda t: (-t[2], t[0], t[1]))


def pair_divergences(embeddings, pairs):
    """Per-cell local divergence: {cell index: max 1-cos over its measured neighbours}.

    The lattice analogue of the Cascade's probe-to-probe divergence (and of
    discover.sensitivity_field's 4-neighbour read): one number per evaluated cell, in the
    same units as `detect_edges` thresholds, so an AMR point joins the map's shared
    blue->red colouring. Cells with no measured neighbour are absent from the mapping.
    """
    out = {}
    for i, j in pairs:
        ea, eb = embeddings[i], embeddings[j]
        if ea is None or eb is None:
            continue
        d = _cosd(ea, eb)
        for t in (int(i), int(j)):
            cur = out.get(t)
            if cur is None or d > cur:
                out[t] = float(d)
    return out


# ---- cost model -----------------------------------------------------------------------

def probe_cost(probe_steps):
    """Image-equivalents of ONE lattice probe.

    0.5 is the h25f convention: the lattice is read on the cheap field (4 of 8 denoising
    steps), so two probes cost about one image. A run that probes at full fidelity pays a
    whole one.
    """
    return 1.0 if probe_steps is None else PROBE_COST


def boundary_fraction(n):
    """phi at n points per edge: the share of cells bordering a label change.

    Measured at k=4 (h25f section 3) and log-interpolated between the measured levels,
    held flat outside them. Only the k >= 5 refusal uses this; nothing a run reports
    depends on it.
    """
    ns = sorted(PHI_K4)
    if n <= ns[0]:
        return PHI_K4[ns[0]]
    if n >= ns[-1]:
        return PHI_K4[ns[-1]]
    for a, b in zip(ns, ns[1:]):
        if a <= n <= b:
            t = (math.log(n) - math.log(a)) / (math.log(b) - math.log(a))
            return PHI_K4[a] + t * (PHI_K4[b] - PHI_K4[a])
    return PHI_K4[ns[-1]]


def projected_probes(k, base, levels, factor):
    """(total probes, per-level rows) an AMR ladder would cost at this k.

    The h25f projection: the coarsest level is evaluated in full, and every refinement
    costs phi(previous level) x (previous level's cells) x factor^d, capped at that
    level's own cell count (refinement can never exceed exhaustive). d = k-1. Validated
    where it can be: at k=4 on the study's own 5/15/30/60 ladder it lands within 5 % of
    the measured probe count.
    """
    d = k - 1
    sched = nested_schedule(base, levels, factor)
    rows, total = [], 0
    for i, n in enumerate(sched):
        cells = math.comb(n + d, d)
        if i == 0:
            p = cells
        else:
            prev = sched[i - 1]
            p = min(cells, int(round(boundary_fraction(prev)
                                     * math.comb(prev + d, d) * factor ** d)))
        total += p
        rows.append({"level": int(n), "cells": int(cells), "probes": int(p)})
    return int(total), rows


# The study's published projections, so a refusal can quote them beside the projection for
# whatever ladder was actually asked for. Its ladder (5 -> 15 -> 30 -> 60) mixes a x3 and a
# x2 step, so it is not one of the single-factor ladders a run may ask for -- these are
# quoted as the measured reference, never recomputed here.
H25F_K5_PROBES = 316702        # 5 -> 15 -> 30 -> 60 at k=5 (RESULTS.md section 4)
H25F_K9_PROBES = 4785432981    # ... and at k=9
H25F_BAR_IMAGE_EQ = 20000      # the 10x-chord-cost bar those miss


@dataclass
class AmrRun:
    """One adaptive-refinement survey.

    The fields down to `_lock` are the contract `cascade.evaluate` reads (prompts, image
    settings, the thumbnail store, the index allocator, the live cloud); everything after
    is this sampler's own state. `probe_geo`/`probe_div`/`_geo_pos` exist for evaluate's
    sake and are not served: an AMR point is identified by its lattice cell, and `points`
    below carries it with its level, which the capped live cloud cannot.
    """
    run_id: str
    prompts: list
    seed: int
    steps: int
    height: int
    width: int
    guidance_scale: float
    # the ladder: `levels` lattices starting at `base` points per simplex edge, each
    # `factor` times finer than the last (5, 2, 4 -> 5, 10, 20, 40)
    base: int = 5
    levels: int = 4
    factor: int = 2
    # refinement radius in COARSE cells around a detected boundary edge (1 = of record)
    r_ref: int = 1
    # lattice probes run on the cheap field, as the Cascade's chord probes do; None = full
    probe_steps: int | None = 4
    status: str = "running"
    phase: str = ""
    phase_done: int = 0
    phase_total: int = 0
    schedule: list = field(default_factory=list)
    # one row per level: {level, cells, n_candidates, n_evaluated, n_edges, cost_image_eq}
    level_stats: list = field(default_factory=list)
    # every evaluated cell, in the order it was first evaluated (its id is its index):
    # {weights, level, image, div}. A cell evaluated coarse keeps the level that paid for it.
    points: list = field(default_factory=list)
    # detected boundary edges: {a, b, level, divergence}, a/b being point ids above
    edges: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    error: str | None = None
    generated: int = 0
    embeddings: dict = field(default_factory=dict)
    thumbs: object = None                             # ThumbnailStore, set by the router
    next_idx: int = 0
    recent_thumbs: list = field(default_factory=list)
    probe_geo: list = field(default_factory=list)
    probe_div: list = field(default_factory=list)
    _geo_pos: dict = field(default_factory=dict, repr=False)
    _lock: object = field(default_factory=threading.Lock, repr=False)

    @property
    def k(self):
        return len(self.prompts)

    @property
    def cost_image_eq(self):
        return float(sum(r["cost_image_eq"] for r in self.level_stats))


def run_amr(app, run, pool):
    """The whole ladder. Called from the router's guarded thread body.

    One batched evaluate per level: all of the coarsest lattice, then only the cells within
    r_ref coarse cells of a detected boundary edge, down to the finest level. Cells are
    identified by their FINEST-level integer counts, so nesting is exact integer
    arithmetic -- a level-5 cell and its level-40 twin are the same key, evaluated once.
    Cancellation is the Cascade's: `run.status` stops the run between (and inside) levels.
    """
    k = run.k
    sched = nested_schedule(run.base, run.levels, run.factor)
    run.schedule = list(sched)
    fine = sched[-1]
    unit = probe_cost(run.probe_steps)

    emb = {}            # canonical key -> embedding
    pid = {}            # canonical key -> point id (index into run.points)
    prev_pts = None     # the previous level's lattice points
    prev_edges = None   # ... and the boundary edges detected on it

    for li, n in enumerate(sched):
        if run.status != "running":
            return
        pts = lattice_points(k, n)
        scale = fine // n
        keys = [tuple(int(v) * scale for v in c) for c in lattice_counts(k, n)]
        carried = np.array([key in emb for key in keys], dtype=bool)
        if li == 0:
            want = np.ones(len(pts), dtype=bool)       # evaluate ALL coarse cells
        else:
            want = refine_set(prev_pts, prev_edges, n, run.r_ref)
        # nesting: a cell already paid for at a coarser level is part of this level's
        # evaluated set without a new probe, and must be, or the detector would see holes
        take = want | carried
        todo = [i for i in range(len(pts)) if take[i] and not carried[i]]

        run.phase = f"level {n}"
        run.phase_done, run.phase_total = 0, len(todo)
        # the tids evaluate() builds are namespaced by run_id, so sharing its "cascade:"
        # prefix cannot collide with a Cascade run's inbox
        gidx = cs.evaluate(app, run, pool, [pts[i] for i in todo], run.seed,
                           f"amr{n}", steps=run.probe_steps)
        if run.status != "running":
            return
        n_new = 0
        for i, gi in zip(todo, gidx):
            e = run.embeddings.get(gi) if gi is not None else None
            if e is None:
                continue                              # a probe that never arrived
            emb[keys[i]] = e
            pid[keys[i]] = len(run.points)
            run.points.append({"weights": [float(v) for v in pts[i]],
                               "level": int(n), "image": int(gi), "div": None})
            n_new += 1

        pairs = neighbours(k, n)
        level_emb = [emb.get(key) for key in keys]
        edges = detect_edges(level_emb, pairs)
        for i, d in pair_divergences(level_emb, pairs).items():
            p = pid.get(keys[i])
            if p is not None:
                cur = run.points[p]["div"]
                run.points[p]["div"] = d if cur is None else max(cur, d)
                cs._set_div(run, run.points[p]["image"], d)
        for i, j, d in edges:
            a, b = pid.get(keys[i]), pid.get(keys[j])
            if a is not None and b is not None:
                run.edges.append({"a": a, "b": b, "level": int(n),
                                  "divergence": round(float(d), 4)})

        cells = len(pts)
        run.level_stats.append({
            "level": int(n), "cells": int(cells),
            "n_candidates": int(take.sum()), "n_evaluated": int(n_new),
            "n_edges": int(len(edges)), "cost_image_eq": float(unit * n_new)})
        run.notes.append(
            f"level {n}: {int(take.sum())}/{cells} cells selected "
            f"({100.0 * take.sum() / max(cells, 1):.0f} % of exhaustive), {n_new} new probes, "
            f"{len(edges)} boundary edges, {unit * n_new:.1f} image-eq")
        prev_pts, prev_edges = pts, edges
        if li + 1 < len(sched) and not edges:
            run.notes.append(
                "no boundary edge at this level: nothing to refine, stopping early")
            break

    run.phase = "done"
    tot_probes = sum(r["n_evaluated"] for r in run.level_stats)
    full = math.comb(sched[-1] + k - 1, k - 1)
    run.notes.append(
        f"done: {tot_probes} probes = {run.cost_image_eq:.1f} image-eq, "
        f"{len(run.edges)} boundary edges, "
        f"{100.0 * tot_probes / max(full, 1):.0f} % of the {full}-cell exhaustive lattice "
        f"(h25f reached the ceiling at 55 % at k=4)")
    if run.status == "running":
        run.status = "complete"
