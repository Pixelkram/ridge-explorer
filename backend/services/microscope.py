"""Ridge microscope: a dense, human-readable image lattice on a 2-D plane through a ridge.

Every other surface either works in the full (k-1)-dimensional simplex -- whose lattices cost
~n^(k-1) images -- or along 1-D lines (chords, walks) that a person cannot read as a picture.
This one cuts a PLANE through a point and shows a G x G lattice of full-fidelity images on it,
so the cost is G^2 at every k. Precedents it follows:

  * Sequential Gallery (Koyama, Sato & Goto, ACM TOG 2020): a 2-D plane subtask shown as a
    5 x 5 (or 3 x 3) grid of images, a click zooming in by 2x around the chosen cell.
  * Sohns, Garth & Leitte, CGF 42(1) 2023 ("Decision Boundary Visualization for Counterfactual
    Reasoning"): a plane chosen by PCA of a BALANCED local set -- half on one side of the
    boundary, half on the other -- so a nearby boundary is in view, with a grey biplot of the
    original axes on top.

The plane
---------
Two orthonormal SUM-ZERO directions e1, e2 (the simplex's own tangent space, so every lattice
point is still an affine recipe), chosen one of three ways:

  * "crossing"    -- e1 = the crossing's hi-res CLOUD normal when the run measured one (the
                     Cascade's own precedence, `cascade._cloud_apply`), else the chord direction
                     that found it; e2 = the first principal direction, orthogonal to e1, of the
                     crossing's cloud points taken as a BALANCED set (equal numbers from each
                     side, as Sohns et al.), else a seeded random tangent direction orthogonal to
                     e1. Centre = the crossing's midpoint.
  * "prompt-swap" -- e1 = the tangent image of e_i - e_j (trade prompt j for prompt i), e2 the
                     same for another pair, Gram-Schmidt'ed.
  * "random"      -- two seeded random orthonormal tangent directions; `redraw` re-draws e2
                     alone, keeping e1.

The lattice is w = w0 + s (a e1 + b e2), a, b in linspace(-1, 1, G): e1 runs left -> right on
screen and e2 bottom -> top. A point outside the simplex is an "outside" cell -- shown empty,
never rendered, never charged. The biplot arrow of prompt i is the projection of e_i's tangent
image (e_i - 1/k) onto (e1, e2), which is exactly (e1[i], e2[i]) because both directions are
sum-zero: it says which prompt grows in which screen direction.

Reading it
----------
Every inside cell is rendered at FULL fidelity through `cascade.evaluate` and embedded with
DINOv2. Two 4-neighbour cells are separated by a boundary when their cosine distance passes
`cascade.COS_T` -- the Cascade's crossing threshold, so "boundary" means here what it means on
the cascade map. Single seed: a boundary drawn here is the Cascade's uncertified kind.

Zoom: clicking a cell opens a new LEVEL centred on it with s/2, in the same plane. Every other
point of the new lattice coincides with one of the parent's (9 of 25 at G = 5, 9 of 49 at G = 7,
1 of 9 at G = 3), and those -- like any point an earlier level already rendered -- are read back
from the run's render cache rather than rendered again. Levels are cached by (centre,
s), so going back and clicking the same cell again costs nothing.

Cost: one image per in-simplex cell not already rendered; a run (all its levels) is refused past
MAX_IMAGE_EQ with the reason.
"""
from dataclasses import dataclass, field
import threading

import numpy as np

from backend.services import cascade as cs
from backend.services.cascade import COS_T, _cosd
from backend.services import gridfree as gf

GRID_CHOICES = (3, 5, 7)
GRID_DEFAULT = 5          # Sequential Gallery's default subtask
S_DEFAULT = 0.10          # half-width in tangent units: 4 Cascade strides from centre to edge
S_MAX = 0.5
# Below this half-width the whole lattice spans ~1/6 of a chord stride: the images are
# indistinguishable at a single seed and a further zoom only spends images on round-off.
S_MIN = 0.002
INSIDE_TOL = 1e-9         # a lattice point this far below a face still counts as on it
IMAGE_COST = 1.0          # every cell is a full-fidelity render: one image-eq
# Ceiling on one run, all levels together. A 7 x 7 start is 49; this is ~8 full zooms deep,
# which is far past the depth (s_min) where the lattice stops resolving anything.
MAX_IMAGE_EQ = 400
MODES = ("crossing", "prompt-swap", "random")


# ---- pure geometry --------------------------------------------------------------------

def _rng(*parts):
    """A generator seeded by a tuple of ints (negative seeds folded into range)."""
    return np.random.default_rng([int(p) % (2 ** 63) for p in parts])


def tangent_unit(v):
    """v projected onto the sum-zero subspace and normalised; None for a (near) zero vector."""
    v = np.asarray(v, dtype=np.float64)
    v = v - v.mean()
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else None


def orthogonalise(v, against):
    """Gram-Schmidt v against orthonormal sum-zero `against`, in the tangent space; None if
    nothing is left (v was in their span)."""
    v = np.asarray(v, dtype=np.float64)
    v = v - v.mean()
    for b in against:
        v = v - float(np.dot(v, b)) * np.asarray(b, dtype=np.float64)
    return tangent_unit(v)


def random_tangent(k, rng, against=(), tries=8):
    """A uniformly random unit sum-zero direction, orthogonal to `against`."""
    for _ in range(tries):
        u = orthogonalise(rng.standard_normal(int(k)), against)
        if u is not None:
            return u
    raise ValueError(f"no tangent direction orthogonal to {len(against)} given at k={k}")


def plane_random(k, seed, redraw=0):
    """Two seeded random orthonormal tangent directions. e1 depends on `seed` alone, e2 on
    (seed, redraw) -- so "re-draw e2" moves the second axis and keeps the first."""
    e1 = random_tangent(k, _rng(seed, 11))
    e2 = random_tangent(k, _rng(seed, 12, redraw), against=[e1])
    return e1, e2, "random plane: e1 and e2 seeded random tangent directions"


def plane_prompt_swap(k, pair_a, pair_b):
    """e1 = tangent image of e_i - e_j, e2 = that of e_p - e_q made orthogonal to e1.

    Raises ValueError with the reason for a pair that is not one (i == j), an index out of
    range, or a second pair that adds no direction (it is the first pair, either way round).
    """
    for nm, (x, y) in (("first", pair_a), ("second", pair_b)):
        if not (0 <= x < k and 0 <= y < k):
            raise ValueError(f"{nm} prompt pair ({x}, {y}) is out of range for k={k}")
        if x == y:
            raise ValueError(f"{nm} prompt pair trades prompt {x} for itself")
    eye = np.eye(int(k))
    e1 = tangent_unit(eye[pair_a[0]] - eye[pair_a[1]])
    e2 = orthogonalise(eye[pair_b[0]] - eye[pair_b[1]], [e1])
    if e2 is None:
        raise ValueError("the two prompt pairs span one direction; pick a second pair that "
                         "involves a different prompt")
    note = (f"prompt-swap plane: e1 = P{pair_a[0] + 1} up / P{pair_a[1] + 1} down, "
            f"e2 = P{pair_b[0] + 1} up / P{pair_b[1] + 1} down (orthogonalised)")
    return e1, e2, note


def balanced_principal(side_a, side_b, e1):
    """First principal direction, orthogonal to e1, of a BALANCED two-sided point set.

    Equal numbers from each side (the first n of each, n = the smaller side), as in Sohns et
    al.: an unbalanced set's principal axis follows whichever side was sampled more. The points
    are centred, e1 is projected out, and the top right-singular vector is returned with a
    deterministic sign (its largest-magnitude component positive). Returns (direction, n) or
    (None, n) when there is no spread left to name a direction.
    """
    n = min(len(side_a), len(side_b))
    if n < 2:
        return None, n
    X = np.vstack([np.asarray(side_a[:n], dtype=float), np.asarray(side_b[:n], dtype=float)])
    X = X - X.mean(axis=0)
    X = X - X.mean(axis=1, keepdims=True)        # rows are recipe differences: sum-zero
    X = X - np.outer(X @ e1, e1)
    _u, sv, vt = np.linalg.svd(X, full_matrices=False)
    if sv.size == 0 or float(sv[0]) < 1e-12:
        return None, n
    v = orthogonalise(vt[0], [e1])
    if v is None:
        return None, n
    if v[int(np.argmax(np.abs(v)))] < 0:
        v = -v
    return v, n


def plane_crossing(k, wa, wb, cloud_normal=None, cloud_pts=None, seed=0, redraw=0, cid=0):
    """The plane through a Cascade crossing: (e1, e2, note).

    e1 = the cloud normal when present (it outranks the bracket chord, `cascade._cloud_apply`),
    else the chord direction wb - wa. e2 = `balanced_principal` of the cloud's side-A and side-B
    points, else a random tangent direction orthogonal to e1 seeded by (seed, cid, redraw).
    """
    e1 = tangent_unit(cloud_normal) if cloud_normal is not None else None
    src1 = "the hi-res cloud normal"
    if e1 is None:
        e1 = tangent_unit(np.asarray(wb, dtype=float) - np.asarray(wa, dtype=float))
        src1 = "the chord direction that found it"
    if e1 is None:
        raise ValueError("this crossing has neither a cloud normal nor a chord direction "
                         "(its bracket collapsed to a point)")
    side_a = [p[0] for p in (cloud_pts or ()) if len(p) > 1 and p[1] == "A"]
    side_b = [p[0] for p in (cloud_pts or ()) if len(p) > 1 and p[1] == "B"]
    e2, n = balanced_principal(side_a, side_b, e1) if cloud_pts else (None, 0)
    if e2 is not None:
        return e1, e2, (f"crossing plane: e1 = {src1}; e2 = first principal direction of a "
                        f"balanced cloud set ({n} side-A + {n} side-B points), orthogonal to e1")
    why = ("no hi-res cloud" if not cloud_pts
           else f"cloud too one-sided ({len(side_a)} A / {len(side_b)} B, need 2 + 2)")
    e2 = random_tangent(k, _rng(seed, 13, cid, redraw), against=[e1])
    return e1, e2, f"crossing plane: e1 = {src1}; e2 = seeded random tangent direction ({why})"


def grid_offsets(G):
    return np.linspace(-1.0, 1.0, int(G))


def in_simplex(w, tol=INSIDE_TOL):
    return bool(float(np.min(w)) >= -tol)


def lattice(w0, e1, e2, s, G):
    """The G x G lattice: one dict per cell {ia, ib, a, b, w, inside}, ib-major.

    ia indexes a (along e1, screen x), ib indexes b (along e2, screen y up). With odd G the
    centre cell IS w0. `inside` = every weight >= -1e-9; outside cells are kept (the panel
    shows them as empty) but never rendered.
    """
    w0 = np.asarray(w0, dtype=np.float64)
    e1 = np.asarray(e1, dtype=np.float64)
    e2 = np.asarray(e2, dtype=np.float64)
    offs = grid_offsets(G)
    out = []
    for ib, b in enumerate(offs):
        for ia, a in enumerate(offs):
            w = w0 + float(s) * (float(a) * e1 + float(b) * e2)
            out.append({"ia": ia, "ib": ib, "a": float(a), "b": float(b),
                        "w": [float(v) for v in w], "inside": in_simplex(w)})
    return out


def biplot(e1, e2):
    """Per prompt i, the projection of its tangent image e_i - 1/k onto (e1, e2).

    Because e1 and e2 sum to zero, <e_i - 1/k, e1> = e1[i]: the arrow is just the i-th
    components, which is why this is cheap enough to recompute every poll.
    """
    return [[float(x), float(y)] for x, y in zip(e1, e2)]


def boundary_edges(G, emb):
    """4-neighbour edges with both cells rendered: {a, b, div, boundary}.

    `emb` maps (ia, ib) -> unit embedding. boundary = div > COS_T, the Cascade's rule.
    """
    out = []
    for ib in range(int(G)):
        for ia in range(int(G)):
            e = emb.get((ia, ib))
            if e is None:
                continue
            for ja, jb in ((ia + 1, ib), (ia, ib + 1)):
                f = emb.get((ja, jb))
                if f is None:
                    continue
                d = float(_cosd(e, f))
                out.append({"a": [ia, ib], "b": [ja, jb], "div": d, "boundary": d > COS_T})
    return out


def zoom_target(centre, s, G, ia, ib, e1, e2):
    """(new centre, new half-width) for a click on cell (ia, ib): that cell's recipe, s/2."""
    offs = grid_offsets(G)
    w = (np.asarray(centre, dtype=np.float64)
         + float(s) * (float(offs[ia]) * np.asarray(e1, dtype=np.float64)
                       + float(offs[ib]) * np.asarray(e2, dtype=np.float64)))
    return w, float(s) / 2.0


def level_key(centre, s):
    return (tuple(round(float(v), gf.KEY_DIGITS) for v in centre), round(float(s), 12))


def new_images(cells, run=None, steps=None):
    """In-simplex cells a render would actually pay for: those not in `run`'s render cache."""
    n = 0
    for c in cells:
        if not c["inside"]:
            continue
        if run is not None:
            eff = int(run.steps) if steps is None else int(steps)
            w = cs._clipn(np.asarray(c["w"], dtype=float), len(c["w"]))
            gi = run.point_cache.get(gf.point_key(w, eff))
            if gi is not None and gi in run.embeddings:
                continue
        n += 1
    return n


def projected_cost(cells, run=None):
    """Image-eq a level would cost: one full-fidelity image per NEW in-simplex cell."""
    return IMAGE_COST * new_images(cells, run)


# ---- the run --------------------------------------------------------------------------

@dataclass
class Level:
    """One lattice of the zoom stack. Also the `ctl` its render runs under, so cancelling a
    level stops exactly its batch (`cascade.evaluate` reads ctl.status / ctl.notes)."""
    level: int
    parent: int | None
    parent_cell: list | None        # [ia, ib] of the parent cell this level zooms into
    centre: list
    s: float
    grid: int
    cells: list                     # lattice() dicts + "image" (-1 until rendered)
    edges: list = field(default_factory=list)
    status: str = "pending"
    n_new: int = 0                  # renders this level paid for
    n_reused: int = 0               # cells read from the run's render cache instead
    notes: list = field(default_factory=list)
    error: str | None = None


@dataclass
class MicroRun:
    """One microscope: a fixed plane (e1, e2) and the stack of levels zoomed into it.

    The fields from `generated` down are the contract `cascade.evaluate` reads (as on
    MetroRun); `point_cache` is the gridfree render cache.
    """
    run_id: str
    prompts: list
    seed: int
    steps: int
    height: int
    width: int
    guidance_scale: float
    mode: str
    e1: list
    e2: list
    grid: int = GRID_DEFAULT
    plane_note: str = ""
    source_run: str | None = None
    source_cid: int | None = None
    levels: list = field(default_factory=list)
    level_index: dict = field(default_factory=dict)
    n_images: int = 0
    notes: list = field(default_factory=list)
    error: str | None = None
    point_cache: dict = field(default_factory=dict)
    generated: int = 0
    phase_done: int = 0
    phase_total: int = 0
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
    def status(self):
        """running while any level renders, else the newest level's own status."""
        if any(lv.status in ("pending", "running") for lv in self.levels):
            return "running"
        return self.levels[-1].status if self.levels else "complete"

    @property
    def cost_image_eq(self):
        return IMAGE_COST * self.n_images


def add_level(run, centre, s, parent=None, parent_cell=None):
    """The level at (centre, s), created if new. Returns (level, created)."""
    key = level_key(centre, s)
    if key in run.level_index:
        return run.levels[run.level_index[key]], False
    cells = lattice(centre, run.e1, run.e2, s, run.grid)
    for c in cells:
        c["image"] = -1
    lv = Level(level=len(run.levels), parent=parent, parent_cell=parent_cell,
               centre=[float(v) for v in centre], s=float(s), grid=int(run.grid), cells=cells)
    run.level_index[key] = lv.level
    run.levels.append(lv)
    return lv, True


def render_level(app, run, pool, lv):
    """Render one level's in-simplex cells at full fidelity, then read its boundaries.

    One batched `evaluate` (through the render cache: cells an earlier level already rendered
    are read, not re-rendered or charged). Cells fill in live as images land; the edges are
    computed once the batch is back.
    """
    lv.status = "running"
    inside = [c for c in lv.cells if c["inside"]]
    ws = [cs._clipn(np.asarray(c["w"], dtype=float), run.k) for c in inside]
    run.phase_done, run.phase_total = 0, len(ws)

    def _fill(i, gi):
        inside[i]["image"] = int(gi)

    gis, n_new = gf.render_cached(app, run, pool, ws, None, f"scope{lv.level}", lv,
                                  on_arrival=_fill)
    lv.n_new = int(n_new)
    lv.n_reused = sum(1 for g in gis if g is not None) - int(n_new)
    run.n_images += int(n_new)
    for c, gi in zip(inside, gis):
        c["image"] = -1 if gi is None else int(gi)
    emb = {(c["ia"], c["ib"]): run.embeddings[c["image"]] for c in inside
           if c["image"] >= 0 and c["image"] in run.embeddings}
    lv.edges = boundary_edges(lv.grid, emb)
    missing = sum(1 for c in inside if c["image"] < 0)
    nb = sum(1 for e in lv.edges if e["boundary"])
    lv.notes.append(f"level {lv.level}: {len(inside)} cells inside the simplex "
                    f"({lv.n_new} rendered, {lv.n_reused} reused), "
                    f"{len(lv.cells) - len(inside)} outside; {nb} boundary edges past {COS_T}"
                    + (f"; {missing} renders never arrived" if missing else ""))
    if lv.status == "running":
        lv.status = "complete"
