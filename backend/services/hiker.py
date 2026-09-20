"""Hikers: population search over prompt-simplex chains.

A ridge walk inside one simplex is bounded — every point is a mixture of the same
prompts. To keep travelling, a walk that reaches the simplex boundary carries its EXIT as
an embedding coordinate (never as a caption; captioning collapses, an exit flanked by dogs
captions to "dog" and re-enters a simplex that already has "dog" as a vertex) and forms a
new simplex from that coordinate plus two fresh prompts.

Several chains ("hikers") advance in parallel against a SHARED novelty archive, so each is
rewarded for finding material none of the others has produced. Each hop:

  1. generate one grid per surviving chain           <- the only GPU cost
  2. extract the ridge graph, walk the best arc, score it against the archive
  3. every arc yields a candidate exit whose image ALREADY EXISTS, so candidates are
     scored without generating anything
  4. rank all candidates globally, keep the best N subject to the guards

Measured against a per-chain-greedy ablation at matched grid budget and matched width,
global ranking gave +7.9 pp novel yield per cell (59.0% vs 51.0%). That is a single run
per arm — directional, not an effect size. It is also STALE: both arms were generated
with the arc splitter that classified corners as junctions (fixed 2026-08-14), so the
arcs walked and the exits proposed would differ on a re-run. Both arms shared the
splitter, so the comparison is internally fair, but the absolute yields are not current.

Guards, each of which fixes a failure observed in development:
  * coherence is NOT gated here (no VLM in this service); instead candidates far from the
    archive are additionally required to be non-degenerate by a cheap image statistic.
    A VLM gate is the right thing if one is available — see `min_chroma` below.
  * triplet diversity: reject a successor leaving two vertices closer than DIV_MAX in
    CLIP-text space (a chain once collapsed with two vertices at 0.708).
  * prompt draws target mean pairwise similarity TARGET_SIM: novelty falls with prompt
    similarity (Spearman -0.817) while coherence rises (+0.634), and the product peaks
    near 0.56. Nearest-neighbour retrieval sits at the dead end of that curve.
"""

from __future__ import annotations

import io
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from backend import config
from backend.services import ridge_graph as rg

TAU_IMG = 0.4218      # archive-novelty threshold, calibrated at AUC 1.000
DIV_MAX = 0.65
TARGET_SIM = 0.56
MIN_ARC = 3          # absolute floor; walk() also scales this with the lattice
REACH = 0.80         # lambda at which an arc counts as reaching the simplex boundary

_GRAM: dict = {}     # pool -> its CLIP-text Gram matrix; see choose_prompts
_GRAM_MAX = 4        # bounded: ~17 MB per entry at the shipped 2,095-prompt pool


@dataclass
class Chain:
    basis: list
    coefs: list
    labels: list
    lineage: str = ""
    cum_novel: float = float("nan")
    arc: list = field(default_factory=list)
    grid_key: str = ""


def bary(i, j, gs, m=0):
    """Barycentric weights for a lattice cell, with an optional padding margin.

    `i, j` are ARRAY indices into a (gs+3m, gs+3m) grid; the lattice coordinate is
    (i-m, j-m), so the three hull edges sit at i=m, j=m and i+j=gs-1+2m, each with m
    cells of padding beyond it. With m>0 the
    outer ring has a NEGATIVE weight on one prompt -- affine extrapolation past the hull,
    which the unnormalised mixing handles exactly. m=0 reproduces the unpadded lattice.
    """
    a, b = (i - m) / (gs - 1), (j - m) / (gs - 1)
    return np.array([1 - a - b, a, b])


def simplex_points(gs, m=0):
    """Array indices of every cell, including the padding ring when m > 0.

    The hypotenuse bound is gs-1+3m, not gs-1+2m: 2m of that is the origin shift (a hull
    cell sits at i+j = gs-1+2m) and the remaining m is the actual padding beyond it.
    Getting this wrong pads two edges and leaves the third exactly as it was.
    """
    n = gs + 3 * m
    return [(i, j) for i in range(n) for j in range(n)
            if i + j <= gs - 1 + 3 * m]


def on_hull(i, j, gs, m=0):
    """True if the cell is inside the true simplex, i.e. not in the padding ring.

    The margin exists to give hull cells a full neighbourhood so their sensitivity is
    measured rather than extrapolated. It is NOT somewhere to walk to: past the hull the
    annulus is significantly LESS novel than the interior (-10.5/-4.5/-9.5 pp on 3/3
    triplets, E3c), so exits stay inside.
    """
    return (i - m) >= 0 and (j - m) >= 0 and (i - m) + (j - m) <= gs - 1


def sensitivity(dino, valid):
    """Mean cosine distance to the 4 lattice neighbours, averaged PER AXIS.

    Kept in step with ridge_detector._sensitivity_2d, which this duplicates; the two
    disagreed until 2026-08-14 and the hiker was the stale one.

    A flat mean over whichever neighbours happen to exist weights an edge cell
    differently from an interior one: on the alpha=0 and beta=0 edges a cell has three
    neighbours, so one axis contributes 1/3 of the score and the other 2/3, and the whole
    edge reads high or low purely by which direction the field varies in. Averaging each
    axis first and then across axes gives every cell the same directional weighting and
    reduces to the old value in the interior.

    This does NOT fix the hypotenuse, where a cell keeps one neighbour per axis: that
    estimate is already balanced, just built from half the differences. Only padding the
    lattice past the hull can supply the missing side.
    """
    gs = valid.shape[0]
    S = np.full((gs, gs), np.nan)
    axes = (((1, 0), (-1, 0)), ((0, 1), (0, -1)))
    for i in range(gs):
        for j in range(gs):
            if not valid[i, j]:
                continue
            axis_means = []
            for offs in axes:
                d = [1.0 - float(dino[i, j] @ dino[i + di, j + dj])
                     for di, dj in offs
                     if 0 <= i + di < gs and 0 <= j + dj < gs and valid[i + di, j + dj]]
                if d:
                    axis_means.append(float(np.mean(d)))
            if axis_means:
                S[i, j] = float(np.mean(axis_means))
    return S


def walk(dino, valid, gs, k_basins=6, h_frac=0.10, m=0, S=None):
    """Ridge graph on this grid; return every arc, longest-to-boundary first.

    `S` overrides the field computed from `dino`. The caller passes one when it has a
    better estimate than a single noise draw can give: E25 (2026-08-15) measured the
    seed-to-seed Spearman of this field at only 0.69-0.79 -- seed moves the ridge MORE
    than any sampler setting does -- so a field averaged over several seeds is a
    materially more reliable ridge than one computed from seed 42 alone. Everything
    downstream (watershed, contraction, arc ranking) is unchanged; only the surface
    they run on gets better.
    """
    if S is None:
        S = sensitivity(dino, valid)
    v = valid & np.isfinite(S)
    lab = rg.semantic_labels(dino, v, k_basins)
    ws = rg.watershed_line(S, v, h_frac)
    ret, flank, _ = rg.contract(ws, lab, v)
    arcs, junctions = rg.split_arcs(ret)
    min_len = max(MIN_ARC, gs // 8)   # a fixed 4 was a large fraction of a 10x10 grid
    arcs = [a for a in arcs if len(a) >= min_len]
    if not arcs:
        return None
    escore = lambda cells: max(1.0 - 3.0 * float(min(bary(y, x, gs, m))) for y, x in cells)
    hull_of = lambda a: [p for p in a if on_hull(p[0], p[1], gs, m)]
    # "Reaches the boundary" is judged on the arc's ON-HULL cells only. escore measures
    # distance TO the hull and says nothing about being inside it: a padding-ring cell has
    # a negative barycentric weight, so it scores above 1.0 and clears REACH for free.
    # An arc lying wholly in the ring therefore won the top tier here while the on_hull
    # filter on candidate exits rejected every one of its cells -- the arc shown was one
    # the chain could NOT have exited from, the exact opposite of the note below (gs=14,
    # m=2, ring-only 13-cell arc: escore 1.46, zero candidate exits, still arcs[0]).
    # Clamping escore would not have helped; the missing test is membership, not range.
    # The second key breaks the tie one tier down, so a ring-only arc is walked only when
    # nothing touching the true simplex was found at all.
    def reaches(a):
        h = hull_of(a)
        return bool(h) and escore(h) >= REACH

    # Which arc is WALKED (arcs[0]) is a display choice, not a search one: every arc
    # contributes a candidate exit and a refinement target regardless, and `cum_novel` is
    # a display figure too (the UI's "N% new"). So rank for what the user sees.
    #
    # lambda is a feasibility FILTER, not the ranking key. It saturates at 1.0 on 66% of
    # arcs and ties at the top in 86% of grids, so ranking by it mostly does nothing, and
    # within-triplet it carries no signal about content (r = -0.02, p > 0.39). Mean
    # sensitivity does (r = +0.23, p = 0.001), and picking on it gives measurably more
    # varied filmstrips at all three cached seeds (frame self-similarity 0.488 vs 0.571,
    # Wilcoxon p = 2e-4) despite ~25% fewer frames. Arcs that reach the boundary still
    # come first, so the arc shown is one a chain could actually have exited from.
    msens = lambda a: float(np.nanmean([S[y, x] for y, x in a]))
    arcs.sort(key=lambda a: (reaches(a), bool(hull_of(a)), msens(a), len(a)), reverse=True)
    out = []
    for a in arcs:
        if escore([a[0]]) > escore([a[-1]]):
            a = a[::-1]
        out.append(a)
    return dict(S=S, valid=v, labels=lab, arcs=out, junctions=junctions, flank=flank)


def approach_path(entry, target, valid):
    """Lattice path from the entry cell to the walk's first ridge station.

    `extend()` makes the previous hop's exit the FIRST VERTEX of this triangle, so with a
    fixed seed the image at vertex A is pixel-identical to the exit image. The ridge the
    chain then walks starts wherever the ridge is, which made the filmstrip cut from the
    exit to an unrelated cell. Prefixing this path closes the gap exactly: the walk's
    first frame becomes the entry image, i.e. the exit image (measured jump 0.794 -> 0.000
    over 150 grids).

    Display only, deliberately. Candidate exits and refinement targets are taken from the
    ridge arcs before this is applied, so the approach cannot move the search -- the two
    alternatives that could (picking entry-proximal arcs, or re-orienting toward the
    entry) measured worse: the first raised the RELATIVE jump to 2.05x by landing on
    flatter ridges, the second cut the hop rate from 100% to 97% for a null.

    Returns the cells from `entry` up to but excluding `target`, so `path + arc` is one
    contiguous walk. Empty when the entry is unusable or already at the target.

    Steps are STRIDE-AWARE, finest first. After r refine rounds `upscale_lattice` leaves
    the un-refined ground on the even sublattice only (stride 2**r), and only a radius-1
    band around the ridge is filled in, so a single-cell step out of the entry corner
    always landed on a cell that was never generated and the whole path was abandoned --
    silently, from the first refined hop on, which is exactly the case that asked for the
    most resolution. Doubling the step until a generated cell is found walks coarse cell
    to coarse cell out in the basins and drops back to 1 as soon as the fine band starts.
    A stride-k hop is still contiguous ON THE LATTICE THAT EXISTS: there is no image
    between two carried coarse cells to show.

    `valid` must be the GENERATED mask, not walk()'s `valid & isfinite(S)`: a carried
    coarse cell whose fine neighbours were never filled has NaN sensitivity but a
    perfectly good image, and this path is display-only, so unmeasurable sensitivity is
    no reason to refuse to show it.
    """
    tgt = (int(target[0]), int(target[1]))
    ci, cj = int(entry[0]), int(entry[1])
    H, W = valid.shape
    if not (0 <= ci < H and 0 <= cj < W) or not valid[ci, cj]:
        return []
    out, guard = [], 0
    while (ci, cj) != tgt and guard < 4 * max(H, W):
        guard += 1
        nxt, step = None, 1
        while nxt is None and step <= max(H, W):
            # clamp each axis at the remaining distance so a wide step cannot overshoot
            oi = int(np.sign(tgt[0] - ci)) * min(step, abs(tgt[0] - ci))
            oj = int(np.sign(tgt[1] - cj)) * min(step, abs(tgt[1] - cj))
            # diagonal first, then either axis; skip cells that were never generated
            nxt = next((q for q in ((ci + oi, cj + oj), (ci + oi, cj), (ci, cj + oj))
                        if q != (ci, cj) and 0 <= q[0] < H and 0 <= q[1] < W
                        and valid[q]), None)
            step *= 2
        if nxt is None:
            return []                    # no clean route; show the ridge walk alone
        out.append((ci, cj))
        ci, cj = nxt
    return out


def novelty(vecs, archive):
    if not archive:
        return float("nan")
    A = np.concatenate(archive, 0)
    return float(((vecs @ A.T).max(axis=1) < TAU_IMG).mean())


def reach(vec, archive):
    """Max cosine to everything seen. Lower = further from all known material."""
    if not archive:
        return 0.0
    A = np.concatenate(archive, 0)
    return float((vec @ A.T).max())


def choose_prompts(x, basis, pool, text_enc):
    """Pick the fresh pair whose triangle with the exit lands nearest TARGET_SIM, subject
    to the diversity cap. `x` is the exit's TEXT-SPACE PROXY: the exit is an embedding
    coordinate with no phrase attached, so captioning it would need a VLM and would lose
    what the image has that no phrase names. Instead the exit's barycentric weights are
    applied to the vertex label embeddings — exact in text space, no captioner required.
    The cap is applied DURING selection: optimising the mean while constraining the max
    caused every proposal to be rejected and the population to starve."""
    fresh = [p for p in pool if p not in basis]
    if len(fresh) < 2:
        return None
    P = text_enc(pool)
    sx = P @ x
    # The pool is constant for a hike but P @ P.T was rebuilt on every call: 45 ms and a
    # 17 MB allocation at the shipped 2,095-prompt pool, half of choose_prompts. Cache it
    # on the pool contents. String hashes are interned so keying on the tuple is cheap.
    key = tuple(pool)
    SS = _GRAM.get(key)
    if SS is None:
        SS = P @ P.T
        # Keep a few pools, not one. A single entry meant two hikes running concurrently
        # with different pools (one on an uploaded CSV, say) evicted each other on every
        # extend(), so each paid the 45 ms / 17 MB rebuild this cache exists to avoid.
        # Bounded FIFO so it still cannot grow without limit; a caller holds its own
        # reference to SS across an eviction, so dropping an entry is always safe.
        while len(_GRAM) >= _GRAM_MAX:
            _GRAM.pop(next(iter(_GRAM)))
        _GRAM[key] = SS
    ok = np.where(np.array([q in fresh for q in pool]))[0]
    best = None
    for i in ok:
        js = ok[ok > i]
        if not len(js):
            continue
        mx = np.maximum(np.maximum(sx[i], sx[js]), SS[i, js])
        feas = np.where(mx <= DIV_MAX)[0]
        if not len(feas):
            continue
        ms = (sx[i] + sx[js] + SS[i, js]) / 3.0
        kb = int(feas[np.argmin(np.abs(ms[feas] - TARGET_SIM))])
        d = abs(float(ms[kb]) - TARGET_SIM)
        if best is None or d < best[0]:
            best = (d, pool[i], pool[int(js[kb])], float(ms[kb]))
    return None if best is None else best[1:]


def exit_text_proxy(chain, exit_ij, gs, text_enc, m=0):
    """Text-space embedding of the exit: vertex label embeddings, barycentrically mixed."""
    w = bary(*exit_ij, gs, m)
    L = text_enc(chain.labels)
    x = (w[:, None] * L).sum(0)
    n = np.linalg.norm(x)
    return x / n if n > 1e-12 else x


def extend(chain, exit_ij, gs, pool, text_enc, m=0):
    x = exit_text_proxy(chain, exit_ij, gs, text_enc, m)
    picked = choose_prompts(x, chain.basis, pool, text_enc)
    if picked is None:
        return None
    p1, p2, sim = picked
    w = bary(*exit_ij, gs, m)
    ecoef = [sum(w[k] * chain.coefs[k][n] for k in range(3))
             for n in range(len(chain.basis))]
    nb = chain.basis + [p1, p2]
    ec = ecoef + [0.0, 0.0]
    n = len(nb)
    coefs = [ec,
             [1.0 if nb[i] == p1 else 0.0 for i in range(n)],
             [1.0 if nb[i] == p2 else 0.0 for i in range(n)]]
    dom = chain.labels[int(np.argmax(w))]
    if dom.startswith("exit("):            # don't nest exit(exit(exit(...
        dom = dom.split("\u00b7", 1)[-1].rstrip(")")
    label = f"exit({w.max():.2f}\u00b7{dom[:34]})"
    return Chain(nb, coefs, [label, p1, p2],
                 lineage=chain.lineage + f">{exit_ij[0]},{exit_ij[1]}")


def upscale_lattice(dino, valid, thumbs, gs, spans=None):
    """Double the lattice, keeping every generated cell exactly where it was.

    `bary(i, j, gs) == bary(2i, 2j, 2gs-1)` identically, so a coarse cell keeps its
    barycentric weights and its image stays valid at the finer resolution. Refinement
    therefore only ever fills the gaps -- nothing is regenerated, and the coarse pass is
    never wasted. Cells not yet generated are left invalid, which `sensitivity()` and
    `walk()` already skip.

    Spans double with the lattice: a cell that covered one cell now covers two, because
    the region it stands for did not shrink. Renderers need this or a refined map shows
    every image at one size and the unrefined ground reads as holes.
    """
    gs2 = 2 * gs - 1
    n2 = 2 * valid.shape[0] - 1          # array dim doubles with the lattice
    d2 = np.zeros((n2, n2, dino.shape[-1]), dtype=dino.dtype)
    v2 = np.zeros((n2, n2), bool)
    t2, s2 = {}, {}
    # With margin m the padded array index doubles too (m2 = 2m), because
    # bary(2I, 2gs-1, 2m) == bary(I, gs, m) identically -- so iterate the array, not a
    # hard-coded triangle, and the margin ring carries over for free.
    for i in range(valid.shape[0]):
        for j in range(valid.shape[1]):
            if not valid[i, j]:
                continue
            d2[2 * i, 2 * j] = dino[i, j]
            v2[2 * i, 2 * j] = True
            if (i, j) in thumbs:
                t2[(2 * i, 2 * j)] = thumbs[(i, j)]
            s2[(2 * i, 2 * j)] = 2 * (spans or {}).get((i, j), 1)
    return d2, v2, t2, gs2, s2


def refine_targets(arcs, gs2, have, radius: int = 1, cap: int = 400, max_sum=None):
    """Ungenerated cells of the doubled lattice lying near the ridge network.

    The analogue of the main tool's refine: spend the extra budget only where the
    structure is, leaving basins coarse. Targets are taken from EVERY arc, not just the
    one walked -- the selector compares exits across arcs, so sharpening only the chosen
    one would bias which exit looks best at the finer resolution.
    """
    want = []
    seen = set()
    for a in arcs:
        for (i, j) in a:
            ci, cj = 2 * i, 2 * j
            for di in range(-radius, radius + 1):
                for dj in range(-radius, radius + 1):
                    p = (ci + di, cj + dj)
                    if p in seen or p in have:
                        continue
                    lim = gs2 - 1 if max_sum is None else max_sum
                    if p[0] < 0 or p[1] < 0 or p[0] + p[1] > lim:
                        continue        # outside the (possibly padded) simplex
                    seen.add(p)
                    want.append(p)
    # deterministic, and closest-to-the-ridge first so a cap truncates the periphery
    want.sort(key=lambda p: (min(abs(p[0] - 2 * i) + abs(p[1] - 2 * j)
                                 for a in arcs for (i, j) in a), p))
    return want[:cap]


def _hot(v: float):
    """matplotlib's `hot` ramp, by hand.

    visualization.py renders with matplotlib, but a hike runs on a worker thread and
    pyplot's global figure state is not thread-safe; two chains rendering at once would
    race. PIL only here.
    """
    return (int(255 * min(1.0, max(0.0, 3 * v))),
            int(255 * min(1.0, max(0.0, 3 * v - 1))),
            int(255 * min(1.0, max(0.0, 3 * v - 2))))


def render_hike_map(thumbs, S, valid, arcs, best, gs, out_path,
                    exit_ij=None, junctions=(), cell: int = 30, spans=None, m=0,
                    approach: int = 0):
    """The simplex this chain walked: thumbnails, sensitivity, and the route on top.

    The filmstrip answers "what did the walk look like"; this answers "where on the
    simplex was it, and what else was on offer". Both are needed to judge a hop: an arc
    is only interesting relative to the arcs the selector passed over.

    Layout matches the stored montage and `render_itinerary`: display row = gs-1-beta,
    display col = alpha, so a station's position here is the same position it has in the
    main grid view.
    """
    from PIL import Image, ImageDraw

    n = gs + 3 * m                     # array dim; the padding ring is drawn too
    img = Image.new("RGB", (n * cell, n * cell), (12, 12, 20))
    meas = S[np.isfinite(S) & valid] if valid is not None else S[np.isfinite(S)]
    lo, hi = (float(meas.min()), float(meas.max())) if meas.size else (0.0, 1.0)
    rng = max(hi - lo, 1e-9)

    def xy(i, j):                      # cell (alpha_idx, beta_idx) -> pixel box
        return i * cell, (n - 1 - j) * cell

    # A cell of span S stands for an SxS block: after a refine round the coarse cells
    # still cover the ground between the new fine cells, so drawing everything at one
    # size leaves the unrefined region looking like holes. Largest span first, so finer
    # cells paint over the coarse cell they subdivide.
    order = sorted(thumbs.items(), key=lambda kv: -(spans or {}).get(kv[0], 1))
    for (i, j), buf in order:
        sp = max(1, int((spans or {}).get((i, j), 1)))
        w = min(sp, n - i) * cell                        # clamp at the canvas edge
        h = min(sp, n - j) * cell
        if w <= 0 or h <= 0:
            continue
        x0 = i * cell
        y0 = max(0, (n - j - min(sp, n - j)) * cell)     # block grows upward in beta
        try:
            t = Image.open(io.BytesIO(buf)).convert("RGB").resize((w, h))
        except Exception:
            continue
        img.paste(t, (x0, y0))
        if np.isfinite(S[i, j]):       # tint by sensitivity, keep the image readable
            img.paste(Image.new("RGB", (w, h), _hot((S[i, j] - lo) / rng)),
                      (x0, y0), Image.new("L", (w, h), 90))

    dr = ImageDraw.Draw(img)
    ctr = lambda p: (xy(*p)[0] + cell // 2, xy(*p)[1] + cell // 2)
    for a in arcs:                     # arcs not taken, so the choice is visible
        if a is best or len(a) < 2:
            continue
        dr.line([ctr(p) for p in a], fill=(120, 130, 160), width=2)
    if best and len(best) >= 2:
        # the approach from the entry is drawn dimmer than the ridge walk: it is how the
        # hiker reaches the ridge, not part of it
        if approach > 0:
            dr.line([ctr(p) for p in best[:approach + 1]], fill=(90, 150, 210), width=3)
            cx, cy = ctr(best[0])
            dr.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], outline=(150, 200, 255), width=3)
        dr.line([ctr(p) for p in best[approach:]], fill=(80, 255, 180), width=4)
    for p in (best or ())[approach:]:
        cx, cy = ctr(p)
        dr.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=(255, 255, 255))
    for p in junctions:
        cx, cy = ctr(tuple(p))
        dr.rectangle([cx - 3, cy - 3, cx + 3, cy + 3], outline=(255, 210, 80), width=2)
    if exit_ij is not None:            # where this chain leaves for the next simplex
        cx, cy = ctr(tuple(exit_ij))
        dr.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], outline=(255, 90, 120), width=3)
    img.save(out_path, quality=88)


def hike_dir(hike_id: str) -> Path:
    return config.RESULTS_DIR / f"hike_{hike_id}"


def new_hike_id() -> str:
    return uuid.uuid4().hex[:8]


def save_state(hike_id: str, state: dict):
    """Persist the hike state as VALID json.

    cum_novel is NaN on the first hop (nothing in the archive to be novel against), and
    Python's json emits a bare `NaN`, which is not JSON: every strict parser rejects the
    file, including every browser. FastAPI converts it to null on the wire, so this only
    bit anything reading hike.json directly -- but that is exactly what a restart does.
    """
    d = hike_dir(hike_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / "hike.json").write_text(_dumps_finite(state))


def _dumps_finite(state: dict) -> str:
    import math

    def clean(o):
        if isinstance(o, float):
            return None if (math.isnan(o) or math.isinf(o)) else o
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        return o
    return json.dumps(clean(state), default=str, allow_nan=False)


def load_state(hike_id: str):
    p = hike_dir(hike_id) / "hike.json"
    return json.loads(p.read_text()) if p.exists() else None
