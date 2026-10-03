"""Cascade runs: fast boundary isolation + dense patch refinement.

The two-tier cascade validated in the search_problem experiments (E85/E86/E92):
tier 1 finds boundary crossings with isotropic chords + pairwise bisection (~10-16
images per crossing); every crossing is scored with the coupled-seed paired statistic
B = mean_s(1 - cos) across the crossing (E84b: AUROC 1.000 against the exact k=4 field)
and judged against a background distribution measured in the SAME run (random
eps-separated pairs) -- never against a no-step null (E89). Tier 2 refines the top
regions with dense 5x5 image patches that walk across the boundary at 1-cell steps.
One patch slot always goes to a random non-top crossing (the exploration floor, E77):
ranking must not decide what we never look at.

Calibrated constants below carry their experiment of record. Settings that look like
free knobs (m=1 probes, m=4 scoring) are not -- see e91 (stratified allocation).
"""
from dataclasses import dataclass, field
import math
import threading
import time

import numpy as np

from backend.services.gpu_pool import DiscoverTask
from backend.services import staged as st

# Calibration of record (search_problem, 2026-08-22)
STRIDE = 0.025          # chord probe spacing (E92 used 0.02; 0.025 trims ~20% cost)
BRACKET = 0.012         # bisection stop (E72: localisation quoted as median/p90, not this)
EPS = 0.047             # B-step separation (E84 pilot separation; 2 fine cells)
CELL = 0.0236           # patch spacing (1 fine cell)
COS_T = 0.35            # pairwise crossing threshold (protocol of record)
M_SCORE = 4             # coupled seeds for B (e91: boundary stratum)
PATCH_N = 5             # 5x5 patch (transversal x one tangent)
STALL_TIMEOUT = 15 * 60


@dataclass
class Crossing:
    cid: int
    wa: np.ndarray
    wb: np.ndarray
    ea: np.ndarray | None = None
    eb: np.ndarray | None = None
    mid: np.ndarray | None = None
    n: np.ndarray | None = None
    b: float | None = None
    significant: bool = False
    thumb: int = -1
    # generation of the chord this crossing was found on: 0 = the fair root survey, >= 1 =
    # a branching child ray. Only generation 0 may feed fair-area statistics.
    gen: int = 0
    # final high-resolution pass (_hires_refine): the bracket was re-probed at stride/factor.
    # hires_width = the sub-bracket spacing the position is now quoted at (None when the pass
    # kept the original bracket); hires_profile = the divergences along the refined sequence
    # (kept in memory, not served); hires_note = "diffuse" when no sub-step cleared COS_T;
    # split_from = the crossing this one was split out of (a second boundary in one bracket).
    hires: bool = False
    hires_width: float | None = None
    hires_profile: list | None = None
    hires_note: str | None = None
    split_from: int | None = None
    # which mode of the pass touched this crossing ("bracket" leaves it None, as it was the
    # only mode; "cloud" = the ball of random probes below).
    hires_mode: str | None = None
    # CLOUD mode (_hires_cloud): the summary of that ball -- {n, r, frac_a, frac_b, frac_other,
    # normal, mid_est, junction_hint, k, div_median, div_max, boundary_frac} -- and the per-point
    # record the map draws, [[weights...], side, image index, divergence]. Both None unless the
    # cloud mode ran here.
    cloud: dict | None = None
    cloud_pts: list | None = None
    # both bracket ends carry FULL-fidelity labels already (staged probes: resumed images), so the
    # certify half's rebracket has nothing to re-render for this crossing
    exact: bool = False


@dataclass
class Patch:
    region: int
    cid: int
    b: float
    significant: bool
    exploration: bool
    grid: list = field(default_factory=list)      # 5x5 of thumb indices (or -1)
    cols_with_crossing: int = 0


@dataclass
class CascadeRun:
    run_id: str
    prompts: list
    seed: int
    steps: int
    height: int
    width: int
    guidance_scale: float
    n_chords: int
    n_patches: int
    # focused exploration: confine the survey to a ball around this recipe
    focus: list | None = None
    focus_radius: float = 0.18
    # tier-1 detection at fewer denoising steps (gated 2026-08-23: 93% crossing
    # recall, 94% certified recall vs full steps, 1 spurious; bracket endpoints
    # are re-rendered at full fidelity before bisection). None = full steps.
    probe_steps: int | None = 4
    # how chord probes read their label (services/staged.py). "steps" = a complete probe_steps
    # image per probe (the behaviour above, unchanged). "staged" = the full schedule to step
    # staged_t, DINOv2 of the predicted clean image x̂0, the latent cached; chord segments whose
    # x̂0 readouts are >= staged_theta apart have both probes RESUMED to the full image, and only
    # those exact labels are tested against COS_T. probe_steps is unused for chords then, and the
    # high-resolution pass renders at full fidelity (its probes compare against exact ends).
    probe_mode: str = "steps"
    staged_t: int = st.T_DEFAULT
    staged_theta: float = st.THETA_DEFAULT
    # full pipeline per detected crossing (rebracket, bisection, coupled-seed B, patches), or
    # detection alone. OFF by default: detection leaves every crossing on its cheap bracket --
    # mid quoted at bracket precision (+-stride/2), b None, significant False -- and skips
    # patches, walks and the trace phase. The chord geometry, the cheap side signatures and
    # everything read off them (the coverage certificate, the local-density map) are the same
    # objects a full run leaves.
    certify: bool = False
    # chord probe spacing, in Euclidean weight-space units along the unit chord direction.
    # STRIDE (0.025, ~1 fine cell) is the protocol of record; finer resolves boundaries that
    # sit closer together than one stride and costs probes as 1/stride. Detection only: the
    # continuation walk's corrector spacing (PC_DS) stays at STRIDE whatever this says.
    stride: float = STRIDE
    # recursive ("branching") chord exploration: children per selected crossing (0 = off) and
    # how many generations of them to spawn. Generation 0 is the fair isotropic-uniform survey;
    # a child is a RAY out of a crossing, so children are PREFERENTIAL samples -- they land
    # where a boundary already is. Fair-area statistics (the Good-Turing coverage certificate)
    # are therefore quoted on generation 0 alone; the local-density map, whose ratio estimator
    # only needs isotropic DIRECTIONS, uses every chord.
    branch: int = 0
    depth: int = 0
    # which crossings of a generation become the next generation's origins: the top this-many %
    # by probe-to-probe divergence (_branch_origins).
    branch_top_pct: int = 20
    # final high-resolution pass. After the WHOLE chord phase (all generations), the top
    # hires_top_pct % of all detected crossings by divergence get hires_factor - 1 extra cheap
    # probes evenly spaced inside their bracket, and the bracket is re-detected at
    # stride/hires_factor: the position sharpens by the factor, and a jump that hid several
    # boundaries splits into several crossings. Off by default. No new chord is laid down --
    # the probes sit on the chord that found the crossing -- so chords_geo/chords_meta and the
    # fair-area certificate's generation-0 filter are untouched.
    hires: bool = False
    hires_top_pct: int = 20
    hires_factor: int = 4
    # which mode the pass spends those probes in. "bracket" (default) is the sweep along the
    # chord described above. "cloud" instead draws hires_cloud_n random points in the tangent
    # ball of radius hires_cloud_r AROUND each selected crossing (_hires_cloud): it measures the
    # directions the chord says nothing about -- the local normal, a chord-free position
    # estimate, and a third basin (junction) a single line cannot see -- and leaves the bracket
    # alone. hires_cloud_r is in tangent units (0.0236 = one fine cell). hires_cloud_k is how many
    # nearest neighbours each cloud point averages its divergence over (_cloud_knn_div), the
    # cloud's stand-in for the 4 grid neighbours of a lattice sensitivity.
    hires_mode: str = "bracket"
    hires_cloud_n: int = 12
    hires_cloud_r: float = 0.05
    hires_cloud_k: int = 4
    # survey randomness (chords, background pairs, patches) apart from the image seed; None = seed. Lets several
    # surveys of ONE image field be compared against a single dense ground truth.
    chord_seed: int | None = None
    # trace phase: walk every significant crossing both ways and link the crossings the walks reach
    trace: bool = False
    trace_steps: int = 8
    trace_certify: bool = True
    traces: list = field(default_factory=list)        # [{cid, direction, points, cert}]
    trace_links: list = field(default_factory=list)   # [(cid, cid)] crossings joined by a walk
    status: str = "running"
    phase: str = "chords"
    chords_geo: list = field(default_factory=list)   # [(a_weights, b_weights)] for the map
    # parallel to chords_geo: [{gen, parent, origin_cid}], children also [+ min_angle_deg, n_near]
    chords_meta: list = field(default_factory=list)
    probe_geo: list = field(default_factory=list)    # completed-point positions, live map cloud
    phase_done: int = 0
    phase_total: int = 0
    generated: int = 0
    crossings: list = field(default_factory=list)
    bg_mean: float | None = None
    bg_p95: float | None = None
    bg_seed_vals: list = field(default_factory=list)   # per background pair: 1-cos per scoring seed j (seed + 997*j)
    patches: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    error: str | None = None
    embeddings: dict = field(default_factory=dict)   # global idx -> np.ndarray (768,)
    thumbs: object = None                            # ThumbnailStore, set by router
    next_idx: int = 0
    recent_thumbs: list = field(default_factory=list)   # last ~12 generated indices
    probe_div: list = field(default_factory=list)       # local divergence per cloud point
    _geo_pos: dict = field(default_factory=dict, repr=False)  # gi -> index in probe_geo
    # staged probes: x̂0 readout embedding, recipe and latent-cache key per read-out image index;
    # the indices whose thumbnail is still the x̂0 preview (not a finished image); the cost ledger
    xhat: dict = field(default_factory=dict, repr=False)
    staged_w: dict = field(default_factory=dict, repr=False)
    staged_lat: dict = field(default_factory=dict, repr=False)
    staged_preview: set = field(default_factory=set, repr=False)
    staged_ledger: object = field(default=None, repr=False)
    # Two concurrent walks (or a walk racing a status poll) share this run; the index
    # allocator is the one read-modify-write that must be atomic.
    _lock: object = field(default_factory=threading.Lock, repr=False)

    @property
    def k(self):
        return len(self.prompts)


def _blue_anchor(k, rng, chosen, n_cand=32, draw=None):
    """Mitchell best-candidate blue noise: draw n_cand uniform simplex points, keep the
    one farthest from every already-chosen anchor. Spreads chord anchors evenly (the
    variance-reduction overlay the round-1 survey adopted: unbiasedness is untouched
    because each candidate is uniform and the direction stays isotropic -- we blue-noise
    the ANCHORS, never endpoint pairs, which would be the sigma^(d+1) length-bias trap
    of the E64 autopsy)."""
    if draw is None:
        cands = rng.dirichlet(np.ones(k), size=n_cand)
    else:
        cands = np.stack([draw() for _ in range(n_cand)])
    if not chosen:
        return cands[0]
    ch = np.stack(chosen)
    d = np.linalg.norm(cands[:, None, :] - ch[None, :, :], axis=2).min(axis=1)
    return cands[int(np.argmax(d))]


def _dir_batch(k, rng, n):
    """n near-orthogonal isotropic directions (Gram-Schmidt over fresh Gaussians in the
    sum-zero subspace; falls back to fresh isotropic draws once the subspace is spent).
    A mini systematic fan: spreads directions like the anchors spread positions."""
    out = []
    basis = []
    for _ in range(n):
        u = rng.standard_normal(k)
        u -= u.mean()
        for b in basis:
            u -= np.dot(u, b) * b
        nn = np.linalg.norm(u)
        if nn < 1e-6:
            basis = []
            u = rng.standard_normal(k)
            u -= u.mean()
            nn = np.linalg.norm(u)
        u /= nn
        basis.append(u)
        if len(basis) >= k - 1:
            basis = []
        out.append(u)
    return out


def _iso_dir(k, rng):
    u = rng.standard_normal(k)
    u -= u.mean()
    return u / np.linalg.norm(u)


BRANCH_NEAR_R = 0.15     # a chord passing this close to an origin (tangent units, ~6 fine cells) is its neighbour
BRANCH_N_CAND = 64       # isotropic candidates one child direction is picked from (best-candidate, as _blue_anchor)


def _near_dirs(origin_w, geo, r=BRANCH_NEAR_R):
    """Unit directions of the stored chords whose SEGMENT passes within r of origin_w.

    The parent always qualifies: the origin is the midpoint of one of its brackets, so its
    distance is 0. Chords further than r are left out on purpose -- a parallel chord elsewhere
    samples different territory, and only the local neighbours can shadow this fan.
    """
    p = np.asarray(origin_w, dtype=float)
    out = []
    for a, b in geo:
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        d = b - a
        nn = np.linalg.norm(d)
        if nn < 1e-12 or _seg_dist(p, a, b) > r:
            continue
        out.append(d / nn)
    return out


def _spread_dir(existing, rng, k, n_cand=BRANCH_N_CAND):
    """Mitchell best-candidate over ANGLES: draw n_cand isotropic directions and keep the one
    whose smallest LINE angle to `existing` is largest. Returns (direction, that angle in deg).

    Chords are undirected lines, so the angle is arccos(|u.v|) -- parallel and anti-parallel both
    count as parallel. Candidates come from the run's rng, so the survey stays reproducible.
    With the parent alone and k=4 the first child comes out near-orthogonal to it; the caller
    feeds each chosen direction back in, so the next sibling spreads away from both.
    """
    cands = [_iso_dir(k, rng) for _ in range(n_cand)]
    if not existing:
        return cands[0], 90.0        # nothing to spread away from
    # per candidate: the most parallel neighbour, i.e. the one that bounds its minimum angle
    worst = np.abs(np.stack(cands) @ np.stack(existing).T).max(axis=1)
    i = int(np.argmin(worst))
    return cands[i], float(np.degrees(np.arccos(min(1.0, worst[i]))))


def _extent(w, u):
    tpos = min((w[i] / -u[i]) for i in range(len(w)) if u[i] < 0)
    tneg = max((-w[i] / u[i]) for i in range(len(w)) if u[i] > 0)
    return tneg, tpos


def _chord_offsets(tneg, tpos, stride):
    """Probe offsets along one chord: tneg, tneg+stride, ... while inside tpos.

    One source for the probes and for the brackets a crossing is quoted at, so probe
    index i means the same offset in both places.
    """
    return [tneg + i * stride for i in range(int((tpos - tneg) / stride) + 1)]


def _min_chord_len(stride):
    """Shortest chord worth probing. 0.15 is the 6*STRIDE of record; the 2*stride arm
    takes over at coarse strides and keeps >= 3 probes (>= 2 neighbour pairs) per chord."""
    return max(0.15, 2 * stride)


def _chord_cap(n_chords, branch, depth):
    """Total chords a branching survey may draw: the geometric sum n*(1 + b + ... + b^depth).

    branch = 0 is the single fair generation of record; branch = 1 is the ratio formula's
    removable singularity, n*(depth+1). A hard cap, not a target: a generation only fills
    from parents that actually crossed something.
    """
    if branch <= 0:
        return n_chords
    if branch == 1:
        return n_chords * (depth + 1)
    return n_chords * (branch ** (depth + 1) - 1) // (branch - 1)


def _cosd(a, b):
    return 1.0 - float(np.dot(a, b))


def _norm_emb(e):
    e = np.asarray(e, dtype=np.float64)
    n = np.linalg.norm(e)
    return e / n if n > 0 else e


def _branch_origins(crossings, pct):
    """Which crossings of one generation spawn the next: the top `pct` % by probe-to-probe
    divergence, at least one, strongest first.

    Replaces the one-origin-per-chord rule of 733f3d1. A chord is an accident of the survey
    geometry, so spending one fan per crossing-bearing chord spread the branching budget evenly
    over chords instead of over evidence; ranking the generation's crossings puts the fans where
    the field diverged most, and lets one chord contribute several origins or none.
    """
    xs = list(crossings)
    if not xs:
        return []
    n = max(1, math.ceil(len(xs) * pct / 100.0))
    # stable sort: equally divergent crossings keep their detection order (cid), so the fans a
    # given survey spawns do not depend on how numpy happened to break a tie
    return sorted(xs, key=lambda x: _cosd(x.ea, x.eb), reverse=True)[:n]


# Final high-resolution pass (owner, 2026-10-02). The chord phase quotes every crossing at ONE
# stride, so a bracket straddling two boundaries closer together than that reads as a single
# strong jump, and the position carries +-stride/2. Re-probing the most divergent brackets is the
# cheapest refinement in the cascade: the extra probes sit INSIDE a bracket of a chord that
# already exists, on the same cheap field, so nothing has to be re-aimed -- no new chord, no new
# direction, no bisection -- and the position sharpens by the factor for (factor - 1) probes.


def _hires_points(wa, wb, factor):
    """The factor - 1 probe positions evenly spaced strictly inside the bracket [wa, wb].

    Spacing is |wb - wa| / factor, so the refined sequence [wa, p1, ..., p_{f-1}, wb] has f
    sub-steps and re-detection localises a crossing to stride/factor. The bracket ends lie on
    the chord, so every interpolated point does too.
    """
    a = np.asarray(wa, dtype=float)
    b = np.asarray(wb, dtype=float)
    return [a + (i / factor) * (b - a) for i in range(1, factor)]


def _hires_select(crossings, pct):
    """Which crossings the high-resolution pass refines: the top `pct` % of ALL crossings by
    probe-to-probe divergence, at least one, strongest first.

    The branching selection rule (_branch_origins) applied ONCE over every generation instead of
    per generation: the pass runs after the whole chord phase, so a child ray's crossing competes
    with a root's on the same evidence.
    """
    return _branch_origins(crossings, pct)


def _hires_refine(x, seq, em, width, gis=None):
    """Re-detect inside one refined bracket and move the crossing onto the sub-bracket it found.

    seq   -- [wa, p1, ..., p_{f-1}, wb], the refined positions along the chord.
    em    -- their embeddings in the same order (None where a probe never arrived).
    width -- the sub-bracket spacing stride/factor, recorded as the new position precision.
    gis   -- the probes' global image indices (None at the ends, which were rendered earlier),
             for the detection-only thumbnail convention.

    Consecutive pairs above COS_T are the boundaries inside the bracket. The strongest becomes
    this crossing; any further one is a SECOND boundary the one-stride probe had merged into the
    same jump, returned as a new crossing with split_from set (same chord, same generation, cid
    unassigned). None above COS_T means the change was spread over the bracket -- "diffuse": no
    sub-step is evidence of a boundary on its own, so the original bracket is the sharpest honest
    statement and stands.
    """
    prof = [None if em[i] is None or em[i + 1] is None else _cosd(em[i], em[i + 1])
            for i in range(len(em) - 1)]
    # strongest first; ties keep the lower sub-bracket, so the pass is order-deterministic
    subs = sorted(((d, i) for i, d in enumerate(prof) if d is not None and d > COS_T),
                  key=lambda t: (-t[0], t[1]))
    x.hires = True
    x.hires_profile = [None if d is None else round(float(d), 4) for d in prof]
    if not subs:
        x.hires_note = "diffuse"
        return []
    extra = []
    for rank, (_d, i) in enumerate(subs):
        t = x if rank == 0 else Crossing(cid=-1, wa=x.wa, wb=x.wb, gen=x.gen, thumb=x.thumb,
                                         hires=True, split_from=x.cid)
        t.wa = np.asarray(seq[i], dtype=float)
        t.wb = np.asarray(seq[i + 1], dtype=float)
        t.ea, t.eb = em[i], em[i + 1]
        t.hires_width = float(width)
        if gis is not None and x.thumb >= 0:
            # detection-only runs quote a bracket END as the thumbnail (both ends sit one
            # half-bracket from the position): keep that, now on the refined bracket
            t.thumb = next((g for g in (gis[i], gis[i + 1]) if g is not None), x.thumb)
        if rank:
            extra.append(t)
    return extra


# CLOUD mode of the pass (owner, 2026-10-02). The bracket mode above sharpens a crossing ALONG
# the chord that found it -- the one direction the survey has already measured. A cloud spends
# the same cheap probes AROUND it instead: random points in the tangent ball of radius r, each
# labelled by the side it matches. That buys what no single line can report: the local NORMAL
# (the direction between the two side means), a position estimate not tied to the chord, and the
# first evidence of a THIRD basin, i.e. a junction. It is a measurement of orientation, not of
# position precision, so the bracket is left exactly as the chord phase quoted it.


def _cloud_points(centre, r, n, rng, tries=20):
    """Up to n points drawn uniformly in the tangent-space ball of radius r around `centre`.

    Uniform in the (k-1)-dimensional ball the simplex's affine hull really has: an isotropic
    sum-zero direction times radius r*U^(1/(k-1)) (the radial inverse-CDF; a uniform radius would
    crowd the centre). The offsets are sum-zero, so the weights keep summing to 1 and only
    non-negativity can fail -- a point outside the simplex is REDRAWN, never clipped, because
    clipping piles points onto the faces and biases the very means the normal is built from. Up to
    `tries` draws per point, then that point is given up on: close to a face most of the ball lies
    outside the simplex, and a short cloud is honest where a stretched one is not. Deterministic
    in `rng`.
    """
    c = np.asarray(centre, dtype=float)
    k = len(c)
    out = []
    for _ in range(int(n)):
        for _t in range(tries):
            u = rng.standard_normal(k)
            u -= u.mean()
            nn = np.linalg.norm(u)
            if nn < 1e-12:
                continue
            w = c + (r * rng.random() ** (1.0 / max(k - 1, 1))) * (u / nn)
            if w.min() >= 0:
                out.append(w)
                break
    return out


def _cloud_side(e, ea, eb):
    """Which side of the crossing one cloud image fell on: "A", "B" or "other".

    The nearer of the two side signatures -- except that a point further than COS_T from BOTH is
    on neither side. That is a third basin, the one thing a cloud sees and a chord cannot, so
    calling it the nearer of two strangers would hide a junction inside frac_a/frac_b.
    """
    da, db = _cosd(e, ea), _cosd(e, eb)
    if min(da, db) > COS_T:
        return "other"
    return "A" if da < db else "B"


def _cloud_normal(ws, sides):
    """(unit normal, midpoint estimate) of a labelled cloud, or (None, None).

    The normal is mean_B - mean_A, normalised: the direction the boundary separates, measured
    from points on both sides of it rather than along one chord, and sum-zero by construction
    (both means sum to 1) so it is a tangent direction. The midpoint is (mean_A + mean_B)/2.
    Needs at least 2 points per side -- one point names a direction with no evidence of its
    spread -- and "other" points enter neither mean: a third basin is not a side of THIS
    boundary.
    """
    a = [w for w, s in zip(ws, sides) if s == "A"]
    b = [w for w, s in zip(ws, sides) if s == "B"]
    if len(a) < 2 or len(b) < 2:
        return None, None
    ma, mb = np.mean(np.stack(a), axis=0), np.mean(np.stack(b), axis=0)
    d = mb - ma
    nn = np.linalg.norm(d)
    if nn < 1e-12:
        return None, None
    return d / nn, (ma + mb) / 2


def _cloud_knn_div(pts_t, embs, k):
    """Local divergence of every pool member: the mean cosine distance to its k NEAREST others.

    The lattice fields (discover.sensitivity_field, hiker.sensitivity) read a point's divergence
    off its 4 GRID neighbours -- a grid hands "neighbour" over for free. A cloud has no grid, so
    the k nearest other members of the pool in tangent coordinates stand in for them. Same
    quantity and same units as the lattice S and as the chord's probe-to-probe divergence, which
    is why a cloud point can now join the map's divergence colouring.

    `pts_t` / `embs` are the WHOLE pool: the cloud's own points plus the two bracket ends, which
    anchor the cloud to the chord that found the crossing (without them a point sitting between
    the two ends has no evidence of the boundary that runs between them, because the cloud alone
    need not have sampled either basin nearby). k is clipped to len(pool) - 1, so a pool of 3 with
    k=16 averages over the 2 others instead of failing. One divergence per entry is returned, in
    order; the caller reads the cloud's own and drops the ends' -- an end is a neighbour, not a
    measurement of this ball.
    """
    P = np.asarray(pts_t, dtype=float)
    m = len(P)
    if m < 2:
        return [None] * m
    kk = max(1, min(int(k), m - 1))
    out = []
    for i in range(m):
        d = np.linalg.norm(P - P[i], axis=1)
        d[i] = np.inf                     # a point is not its own neighbour
        nb = np.argsort(d, kind="stable")[:kk]
        out.append(float(np.mean([_cosd(embs[i], embs[j]) for j in nb])))
    return out


def _cloud_apply(x):
    """Hand a cloud crossing's geometry back to its cloud; True when it did.

    PRECEDENCE: the cloud normal is a local estimate of how the boundary is ORIENTED, measured in
    every tangent direction; the bracket chord is merely the line the survey happened to cross on.
    So the cloud wins -- but both places that derive mid/n from the bracket (the detection-only
    finish and the end of bisection, which legitimately re-reads the bracket it narrowed) run
    AFTER the pass, which is why this is re-applied there instead of only at cloud time. A
    bracket-mode crossing, or a cloud with too few points on a side to name a normal, keeps the
    bracket geometry.
    """
    c = getattr(x, "cloud", None)
    if not c or c.get("normal") is None:
        return False
    x.n = np.asarray(c["normal"], dtype=float)
    if c.get("mid_est") is not None:
        x.mid = np.asarray(c["mid_est"], dtype=float)
    return True


def _hires_cloud(app, run, pool, sel_x, rng):
    """CLOUD mode: one ball of random cheap probes around each selected crossing, in ONE batch.

    Per crossing, hires_cloud_n points in the tangent ball of radius hires_cloud_r around the
    centre of its bracket, each labelled against the crossing's OWN side signatures. The summary
    lands on x.cloud and the per-point record on x.cloud_pts (for the map); where the cloud names
    a normal it also becomes the crossing's n and mid (_cloud_apply). wa/wb are untouched, so
    bisection and the detection-only finish still work from the bracket.

    Each point also carries a DIVERGENCE (_cloud_knn_div): the mean cosine distance to its
    hires_cloud_k nearest neighbours among the cloud and the two bracket ends. That is the cloud's
    analogue of the lattice sensitivity, so -- unlike the labels, which only exist against this
    crossing's signatures -- it is the same reading the chord probes feed, and every cloud point
    goes into the run's divergence colouring (_set_div) through it.
    """
    n, r = int(run.hires_cloud_n), float(run.hires_cloud_r)
    pts, meta, by_sel = [], [], {}
    for si, x in enumerate(sel_x):
        for p in _cloud_points(np.clip((x.wa + x.wb) / 2, 0, None), r, n, rng):
            pts.append(p)
            meta.append(si)
    run.phase_total, run.phase_done = len(pts), 0
    gh = evaluate(app, run, pool, pts, run.seed, "hires", steps=_extra_steps(run))
    if run.status != "running":
        return
    for si, p, gi in zip(meta, pts, gh):
        by_sel.setdefault(si, []).append((p, gi))
    n_norm, n_junc, coss, all_div = 0, 0, [], []
    for si, x in enumerate(sel_x):
        ws, sides, gis, es = [], [], [], []
        for p, gi in by_sel.get(si, ()):
            e = run.embeddings.get(gi) if gi is not None else None
            if e is None:
                continue                  # a probe that never arrived labels nothing
            ws.append(p)
            sides.append(_cloud_side(e, x.ea, x.eb))
            gis.append(gi)
            es.append(e)
        # the neighbour pool: the cloud plus the two bracket ends (the clip is the helper's own,
        # repeated here only so the stored k is the one actually averaged over)
        kk = max(1, min(int(run.hires_cloud_k), len(ws) + 1))
        divs = _cloud_knn_div(ws + [x.wa, x.wb], es + [x.ea, x.eb], kk)[:len(ws)]
        for gi, dv in zip(gis, divs):
            _set_div(run, gi, dv)
        all_div += divs
        nrm, mid = _cloud_normal(ws, sides)
        tot = max(len(sides), 1)          # an empty cloud reports zero of every side
        fa, fb, fo = (sides.count("A") / tot, sides.count("B") / tot,
                      sides.count("other") / tot)
        x.cloud = {"n": len(sides), "r": r,
                   "frac_a": float(fa), "frac_b": float(fb), "frac_other": float(fo),
                   "normal": None if nrm is None else [float(v) for v in nrm],
                   "mid_est": None if mid is None else [float(v) for v in mid],
                   "junction_hint": bool(fo > 0),
                   "k": int(kk),
                   "div_median": float(np.median(divs)) if divs else 0.0,
                   "div_max": float(max(divs)) if divs else 0.0,
                   "boundary_frac": float(sum(1 for d in divs if d > COS_T) / tot)}
        x.cloud_pts = [[[float(v) for v in w], s, int(g), float(d)]
                       for w, s, g, d in zip(ws, sides, gis, divs)]
        x.hires, x.hires_mode = True, "cloud"
        n_junc += 1 if x.cloud["junction_hint"] else 0
        if _cloud_apply(x):
            n_norm += 1
            d = x.wb - x.wa
            dn = float(np.linalg.norm(d))
            if dn > 1e-12:
                coss.append(abs(float(np.dot(nrm, d / dn))))
    run.notes.append(
        f"hires cloud: {len(sel_x)} crossings (top {run.hires_top_pct} %), {n} pts @ {r:.3f}, "
        f"k={int(run.hires_cloud_k)}; normals for {n_norm}; junction hints {n_junc}; "
        f"median |cos(normal, chord dir)| "
        + (f"{float(np.median(coss)):.2f}" if coss else "n/a")
        + "; median div "
        + (f"{float(np.median(all_div)):.2f} · boundary frac "
           f"{sum(1 for d in all_div if d > COS_T) / len(all_div):.2f}" if all_div else "n/a"))


_EVAL_SEQ = [0]


def evaluate(app, run, pool, weights, seed, label, ctl=None, on_arrival=None, steps=None):
    """Generate one image per weight vector; returns list of global indices (or None
    where the result never arrived). Follows the discover shard/inbox discipline:
    inbox lists are pre-registered before submit, results come only from the inbox,
    the queue is kept short so a cooperative stop stays responsive, and a pool-wide
    cancel epoch bump marks this run cancelled rather than being re-raised."""
    if not weights:
        return []
    ctl = ctl if ctl is not None else run
    with run._lock:
        idxs = list(range(run.next_idx, run.next_idx + len(weights)))
        run.next_idx += len(weights)
        _EVAL_SEQ[0] += 1
        label = f"{label}.{_EVAL_SEQ[0]}"  # tids unique across rounds/retries
    local_by_gi = {gi: li for li, gi in enumerate(idxs)}
    points = [(gi, [float(x) for x in w]) for gi, w in zip(idxs, weights)]
    pt_by_gi = {gi: w for gi, w in points}
    ep = pool.cancel_epoch.value
    n_gpus = max(1, min(getattr(pool, "n_gpus", 1), len(points)))
    CHUNK = 8
    step = n_gpus * CHUNK
    got_idx = set()
    for off in range(0, len(points), step):
        if ctl.status != "running":
            break
        block = points[off:off + step]
        shards = []
        for g in range(n_gpus):
            shard = block[g::n_gpus]
            if not shard:
                continue
            tid = f"cascade:{run.run_id}:{label}:{off}:{g}"
            app.state.hike_inbox.setdefault(tid, [])
            shards.append((tid, len(shard)))
            pool.submit(DiscoverTask(
                job_id=tid, basis=run.prompts, points=shard, seed=seed,
                height=run.height, width=run.width,
                steps=steps if steps is not None else run.steps,
                guidance_scale=run.guidance_scale))
        want = sum(n for _, n in shards)
        got, last = 0, time.time()
        while got < want and ctl.status == "running":
            moved = False
            for tid, _n in shards:
                ib = app.state.hike_inbox.get(tid) or []
                while ib:
                    r = ib.pop(0)
                    if r.row not in pt_by_gi:
                        continue  # stale result from a superseded round
                    run.thumbs[r.row] = r.thumbnail_bytes
                    run.embeddings[r.row] = _norm_emb(r.dino_embedding)
                    run.recent_thumbs.append(r.row)
                    del run.recent_thumbs[:-12]
                    if on_arrival is not None:
                        try:
                            on_arrival(local_by_gi[r.row], r.row)
                        except Exception:
                            pass  # colouring must never sink a run
                    if len(run.probe_geo) < 6000 and r.row in pt_by_gi:
                        run._geo_pos[r.row] = len(run.probe_geo)
                        run.probe_geo.append(pt_by_gi[r.row])
                        run.probe_div.append(None)
                    got_idx.add(r.row)
                    run.generated += 1
                    run.phase_done += 1
                    got += 1
                    moved = True
            if moved:
                last = time.time()
                continue
            if pool.cancel_epoch.value != ep:
                ctl.notes.append("cancelled by a pool-wide cancel")
                ctl.status = "cancelled"
                break
            he = getattr(app.state, "hike_task_errors", {})
            err = next((he.pop(tid) for tid, _n in shards if tid in he), None)
            if err is not None:
                ctl.notes.append(f"worker error: {err}")
                break
            if time.time() - last > STALL_TIMEOUT:
                ctl.notes.append(f"{label} stalled at {got}/{want}; moving on")
                break
            time.sleep(0.25)
        for tid, _n in shards:
            app.state.hike_inbox.pop(tid, None)
    return [gi if gi in got_idx else None for gi in idxs]


def coverage_stats(crossings, links=None, use_signature=True):
    """Group crossings into distinct ridges and estimate unexplored boundary share.

    Two crossings portray the same ridge if their side-embedding pairs match
    (order-invariant, both sides within the protocol threshold COS_T). Because chords
    sample boundary AREA fairly, Good-Turing applies: the share of ridges crossed
    exactly ONCE estimates the share of boundary area never crossed at all -- the
    survey's own measure of its ignorance. Returns (n_groups, n_singletons, gt_share)
    or (None, None, None) below n=2 crossings.
    """
    xs = [x for x in crossings if x.ea is not None and x.eb is not None]
    n = len(xs)
    if n < 2:
        return None, None, None, None
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if not use_signature:
                break
            a1, b1 = xs[i].ea, xs[i].eb
            a2, b2 = xs[j].ea, xs[j].eb
            straight = max(_cosd(a1, a2), _cosd(b1, b2))
            flipped = max(_cosd(a1, b2), _cosd(b1, a2))
            if min(straight, flipped) < COS_T:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    # trace links: a walk from one crossing reached the other -- same ridge by construction, even where the
    # side images have drifted too far apart for the signature test
    pos = {x.cid: i for i, x in enumerate(xs)}
    for ca, cb in links or ():
        if ca in pos and cb in pos:
            ri, rj = find(pos[ca]), find(pos[cb])
            if ri != rj:
                parent[ri] = rj
    from collections import Counter
    roots = [find(i) for i in range(n)]
    sizes = Counter(roots)
    n1 = sum(1 for c in sizes.values() if c == 1)
    gid = {r: g for g, r in enumerate(sorted(sizes))}
    group_of = {xs[i].cid: gid[roots[i]] for i in range(n)}
    return len(sizes), n1, n1 / n, group_of


@dataclass
class Walk:
    """A human-in-the-loop traversal along one ridge. Each step: move tangentially,
    re-bisect transversally to stay on the boundary, record the boundary image and its
    local contrast. Deliberately NOT an optimizer -- the automated argmax search
    saturated in every benchmark (E86/E86b/E87); the eye does the judging here."""
    wid: str
    cid: int
    direction: int
    n_steps: int
    status: str = "running"
    use_jvp: bool = False
    jvp: dict | None = None
    sig_mode: str = "relative"      # "absolute" (same_ridge) | "relative" (same_ridge_relative); relative won its A/B 24/24
    mode: str = "continuation"      # "continuation" (predictor-corrector, won tests/walk_pc_results.json) | "fan" (old)
    images: int = 0                 # images the walk rendered (efficiency = images per captured arclength)
    plane: dict | None = None       # {x0, t, n}: the walk plane span(t, n) through the origin crossing
    cert: dict | None = None        # certify_walk: held-out-seed B per station
    trace: list = field(default_factory=list)   # continuation: one entry per corrector attempt (diagnostics)
    side_embs: list = field(default_factory=list, repr=False)   # continuation: (a, b) side embeddings per station
    cont_gamma: float | None = None   # research override of PC_CONT_GAMMA
    steps: list = field(default_factory=list)
    segs: list = field(default_factory=list)    # transversal probe lines, for the map
    notes: list = field(default_factory=list)
    error: str | None = None


def same_ridge(ea1, eb1, ea2, eb2):
    """Order-invariant side-signature match -- the coverage-grouping criterion."""
    straight = max(_cosd(ea1, ea2), _cosd(eb1, eb2))
    flipped = max(_cosd(ea1, eb2), _cosd(eb1, ea2))
    return min(straight, flipped) < COS_T


def same_ridge_relative(ea0, eb0, ea1, eb1):
    """Relative side match for the WALK: the new pair straddles the same boundary if each new side is closer to
    the origin's matching side than to its opposite (or the flipped assignment). Unlike same_ridge it does not
    require the sides to stay similar images -- content along a front is location-specific (h07b-d), so the
    absolute test rejected 28/36 first stations (tests/walk_diag_*.json)."""
    straight = _cosd(ea1, ea0) < _cosd(ea1, eb0) and _cosd(eb1, eb0) < _cosd(eb1, ea0)
    flipped = _cosd(ea1, eb0) < _cosd(ea1, ea0) and _cosd(eb1, ea0) < _cosd(eb1, eb0)
    return straight or flipped


WALK_STEP = 0.05          # tangent step per click-step
WALK_PROBE = 0.024        # transversal probe arm (matches the patch cell scale)
WALK_WIDE = 0.06          # one widened retry before declaring the ridge lost


def _jvp_normal(app, run, pool, x0, walk, timeout=240.0):
    """One exact-JVP probe at x0 through the pool; blocks this worker thread until the collector posts the
    result into app.state.jobs. Returns the unit sum-zero normal in barycentric coordinates, or None."""
    from .gpu_pool import ProbeTask
    import uuid
    pid = f"jvp_{uuid.uuid4().hex[:8]}"
    app.state.jobs[pid] = {"type": "jvp_probe", "status": "running", "kind": "jvp", "job_id": run.run_id,
                           "weights": [float(v) for v in x0], "k": run.k, "names": [f"P{i + 1}" for i in range(run.k)],
                           "result": None, "error": "", "started_at": time.time()}
    pool.submit(ProbeTask(probe_id=pid, job_id=run.run_id, prompts=list(run.prompts), weights=[float(v) for v in x0],
                          seed=run.seed, height=run.height, width=run.width, steps=run.steps,
                          guidance_scale=run.guidance_scale, use_slerp=False))
    t0 = time.time()
    while time.time() - t0 < timeout and walk.status == "running":
        e = app.state.jobs.get(pid) or {}
        if e.get("status") == "done" and e.get("result"):
            r = e["result"]
            n = np.asarray(r["normal_bary"], dtype=np.float64)
            n -= n.mean()
            nn = np.linalg.norm(n)
            if nn < 1e-9:
                return None
            walk.jvp = {"rank1_share": float(r["rank1_share"]), "participation_ratio": float(r["participation_ratio"]),
                        "wall_s": float(r.get("wall_s", 0.0)), "probe_id": pid}
            return n / nn
        if e.get("status") == "error":
            walk.notes.append(f"JVP probe failed: {e.get('error')}")
            return None
        time.sleep(1.0)
    return None


def _clipn(w, k):
    w = np.clip(w, 0, None)
    s = w.sum()
    return w / s if s > 0 else np.full(k, 1.0 / k)


def _walk_setup(app, run, pool, walk):
    """Shared by both walk modes: the origin crossing, the transversal (the bracket chord, or the exact JVP
    normal), the side-signature test, and the tangent t -- seeded per crossing, so both modes of one
    (crossing, direction) walk the SAME plane span(t, n). Returns (x0, nrm, t, match), or None after
    flagging the walk as an error."""
    origin = None
    for c in run.crossings:
        if c.cid == walk.cid and c.mid is not None and c.n is not None:
            origin = c
            break
    if origin is None:
        walk.status = "error"
        walk.error = "crossing not found or not yet bisected"
        return None
    x0, nrm = origin.mid.copy(), origin.n.copy()
    sig_a, sig_b = origin.ea, origin.eb
    if getattr(walk, "sig_mode", "absolute") in ("relative", "continuity"):
        _match = lambda ea, eb: same_ridge_relative(sig_a, sig_b, ea, eb)
    else:
        _match = lambda ea, eb: same_ridge(ea, eb, sig_a, sig_b)
    if getattr(walk, "use_jvp", False):
        # Replace the bracket chord (an isotropic line that happened to cross) by the exact whitened normal at
        # the origin: the shortest direction across the front, and the true tangent plane for the stations.
        # Sign-aligned with the bracket so the a/b side signatures keep their meaning.
        nj = _jvp_normal(app, run, pool, x0, walk)
        if nj is not None:
            walk.jvp["cos_with_bracket"] = float(abs(np.dot(nj, nrm)))
            nrm = nj if np.dot(nj, nrm) >= 0 else -nj
            walk.notes.append(f"transversal = JVP normal (|cos| with bracket {walk.jvp['cos_with_bracket']:.2f}, "
                              f"rank-1 {walk.jvp['rank1_share']:.2f}, PR {walk.jvp['participation_ratio']:.2f})")
        else:
            walk.notes.append("JVP normal unavailable; fell back to the bracket chord")
    k = run.k
    rng = np.random.default_rng(run.seed + 7919 * walk.cid)
    for _ in range(4):
        t = rng.standard_normal(k)
        t -= t.mean()
        t -= np.dot(t, nrm) * nrm
        tn = np.linalg.norm(t)
        # cid 0 seeds default_rng(run.seed) -- the very stream run_cascade drew chord 0's direction from, so a
        # crossing on chord 0 got t == n and failed here (the "other" rows of tests/walk_diag_*.json): redraw
        if tn > 1e-6:
            break
    if tn < 1e-9:
        walk.status = "error"
        walk.error = (f"no tangent direction (k={k}, tn={tn:.2e}, "
                      f"|nrm|={float(np.linalg.norm(nrm)):.3f}, "
                      f"nrm_dim={len(nrm)}, mid_dim={len(x0)})")
        return None
    t = walk.direction * t / tn      # at k=3: THE ridge direction (up to sign)
    walk.plane = {"x0": [float(v) for v in x0], "t": [float(v) for v in t], "n": [float(v) for v in nrm]}
    return x0, nrm, t, _match, sig_a, sig_b


def run_walk(app, run, pool, walk):
    """Fan walk: follow the ridge by LINE SAMPLING, not sequential stepping.

    Stations sit on a straight tangent line from the origin crossing; each station gets
    a short transversal probe line whose arm widens with lookahead distance (measured
    sheet drift ~1 cell per cell travelled). All stations generate in one batch, then
    all brackets bisect in two batched rounds -- the whole walk costs ~3-4 GPU rounds
    instead of one per step. Each captured crossing must MATCH the origin's side
    signature (same-ridge check); a mismatch means a junction or a neighbouring ridge
    (spacing ~0.2), and the walk truncates there with a note rather than derailing.
    """
    if getattr(walk, "mode", "fan") == "continuation":
        return run_walk_continuation(app, run, pool, walk)
    su = _walk_setup(app, run, pool, walk)
    if su is None:
        return
    x0, nrm, t, _match, _, _ = su
    k = run.k

    def clipn(w):
        return _clipn(w, k)

    # round 1: all stations' probe pairs in one batch (arm grows with lookahead)
    stations = []
    for j in range(1, walk.n_steps + 1):
        base = x0 + j * WALK_STEP * t
        if np.min(base) < 0.003:
            walk.notes.append(f"edge of the space at station {j}")
            break
        arm = min(WALK_PROBE + 0.020 * j, WALK_WIDE + 0.03)
        # cap by feasible extent: probes leaving the simplex used to get
        # clipped+renormalised, silently bending the straddle near faces
        bb = clipn(base)
        feas = min((bb[i] / abs(nrm[i]) for i in range(k)
                    if abs(nrm[i]) > 1e-9), default=arm)
        arm = min(arm, max(feas - 1e-4, 0.0))
        if arm < WALK_PROBE / 2:
            walk.notes.append(f"too close to a face at station {j}")
            break
        stations.append(dict(j=j, base=bb, arm=arm))
    if not stations:
        walk.status = "complete"
        return
    ws = []
    for s in stations:
        pa = clipn(s["base"] - s["arm"] * nrm)
        pb = clipn(s["base"] + s["arm"] * nrm)
        walk.segs.append([[float(v) for v in pa],
                          [float(v) for v in pb]])
        ws.append(pa)
        ws.append(pb)
    gp = evaluate(app, run, pool, ws, run.seed, f"walk{walk.wid}fan", ctl=walk)
    walk.images += len(ws)
    for i, s in enumerate(stations):
        ga, gb = gp[2 * i], gp[2 * i + 1]
        s["ea"] = run.embeddings.get(ga) if ga is not None else None
        s["eb"] = run.embeddings.get(gb) if gb is not None else None
        s["lo"], s["hi"] = -s["arm"], s["arm"]
        s["ok"] = (s["ea"] is not None and s["eb"] is not None
                   and _cosd(s["ea"], s["eb"]) > COS_T
                   and _match(s["ea"], s["eb"]))
    # widen retry (one batched round) for stations that failed the graded arm --
    # the V1 setting from the ground-truth sweep: conservative signature, but a second
    # look at 0.12 before giving up
    retry = [s for s in stations if not s["ok"]]
    if retry and walk.status == "running":
        ws = []
        for s in retry:
            wide = min(0.12, max(s["arm"], 0.001))
            base = s["base"]
            feas = min((base[i] / abs(nrm[i]) for i in range(k)
                        if abs(nrm[i]) > 1e-9), default=wide)
            s["arm2"] = min(0.12, max(feas - 1e-4, 0.001))
            pa = clipn(base - s["arm2"] * nrm)
            pb = clipn(base + s["arm2"] * nrm)
            walk.segs.append([[float(v) for v in pa],
                              [float(v) for v in pb]])
            ws.append(pa)
            ws.append(pb)
        gp = evaluate(app, run, pool, ws, run.seed, f"walk{walk.wid}wide", ctl=walk)
        walk.images += len(ws)
        for i, s in enumerate(retry):
            ga, gb = gp[2 * i], gp[2 * i + 1]
            ea2 = run.embeddings.get(ga) if ga is not None else None
            eb2 = run.embeddings.get(gb) if gb is not None else None
            if (ea2 is not None and eb2 is not None and _cosd(ea2, eb2) > COS_T
                    and _match(ea2, eb2)):
                s["ea"], s["eb"] = ea2, eb2
                s["lo"], s["hi"] = -s["arm2"], s["arm2"]
                s["ok"] = True
    # contiguous prefix only: the first failed station truncates the walk
    good = []
    for s in stations:
        if not s["ok"]:
            if s["ea"] is not None and s["eb"] is not None and \
               _cosd(s["ea"], s["eb"]) > COS_T:
                walk.notes.append(
                    f"ridge changed identity at station {s['j']} (junction?)")
            else:
                walk.notes.append(f"ridge ended before station {s['j']}")
            break
        good.append(s)
    # rounds 2-3: batched bisection of all captured brackets
    for _ in range(2):
        if walk.status != "running" or not good:
            break
        ws = [clipn(s["base"] + ((s["lo"] + s["hi"]) / 2) * nrm) for s in good]
        gm = evaluate(app, run, pool, ws, run.seed,
                      f"walk{walk.wid}bis", ctl=walk)
        walk.images += len(ws)
        for s, g in zip(good, gm):
            e = run.embeddings.get(g) if g is not None else None
            if e is None:
                continue
            s["gmid"] = g
            mid = (s["lo"] + s["hi"]) / 2
            s["gmid_off"] = mid   # the offset this thumb was rendered at
            if _cosd(e, s["ea"]) < _cosd(e, s["eb"]):
                s["lo"], s["ea"] = mid, e
            else:
                s["hi"], s["eb"] = mid, e
    for s in good:
        off = s.get("gmid_off", (s["lo"] + s["hi"]) / 2)
        pos = clipn(s["base"] + off * nrm)
        ctr = clipn(s["base"] + ((s["lo"] + s["hi"]) / 2) * nrm)   # best estimate: centre of the final bracket
        walk.steps.append(dict(weights=[float(v) for v in pos],
                               contrast=float(_cosd(s["ea"], s["eb"])
                                              if s["ea"] is not None
                                              and s["eb"] is not None else 0.0),
                               thumb=int(s.get("gmid", -1) if s.get("gmid") is not None
                                         else -1),
                               center=[float(v) for v in ctr]))
    if walk.status == "running":
        walk.status = "complete"


# Continuation walk (RESEARCH_ridge_following_k4.md, fix 1). Under the relative rule the fan's remaining losses
# are drift (12/36) and edge exits (7/36): its straight line leaves a front that bends -- or never ran along the
# line at all, since t is only perpendicular to the CHORD that found the crossing, not to the front.
PC_DS = STRIDE            # corrector spacing = the chord probe spacing of record (the detection protocol)
PC_ARM_FIRST = 0.10       # first corrector half-width: t can miss the front's direction by ~60 deg
PC_ARM = 0.05             # afterwards the secant predictor errs only at second order
PC_H_MIN = 0.0125         # below half a stride a secant is mostly localisation noise
PC_GROW = 1.5             # step growth after a success (Allgower-Georg step control), capped at WALK_STEP
PC_EDGE = 0.003           # face margin, as in the fan walk
PC_SEP_MAX = 4            # widest straddle pair on a corrector line: 0.10 (the fan's pairs span 0.09-0.24)
PC_CONT_GAMMA = 0.5       # sig_mode "continuity": a new station's largest side move, as a share of the previous station's
                          # contrast, must stay below this. Dev (k3_T0 + dense ground truth, tests/walk_gt_dev/): the
                          # ratio separates ridge switches from continuing steps with AUC 0.87; at 0.5 it rejected
                          # 16/16 switches and 27 % of continuing steps on the first try (they retry at a shorter step)


def _pc_candidates(offs, em, match):
    """Every localised crossing on a corrector line (see _pc_bracket), as (key, k_lo, k_hi, i, j), one per
    localised bracket (the narrowest evidence pair wins it), plus whether anything straddled at all."""
    best, straddled = {}, False
    n = len(offs)
    for i in range(n - 1):
        for j in range(i + 1, min(i + PC_SEP_MAX, n - 1) + 1):
            ea, eb = em[i], em[j]
            if ea is None or eb is None or _cosd(ea, eb) <= COS_T:
                continue
            straddled = True
            if not match(ea, eb):
                continue
            lo, hi = i, j
            for m in range(i + 1, j):
                if em[m] is None:
                    continue
                if _cosd(em[m], ea) < _cosd(em[m], eb):
                    lo = m
                else:
                    hi = m
                    break
            key = (abs(offs[lo] + offs[hi]), j - i)
            if (lo, hi) not in best or key < best[(lo, hi)][0]:
                best[(lo, hi)] = (key, lo, hi, i, j)
    return sorted(best.values()), straddled


def _orient(prev_a, prev_b, s1, s2):
    """Assign a new side pair to the previous station's sides (the assignment that moves them least).
    Returns (s_a, s_b, d_a, d_b) with d_* = 1-cos to the matching previous side."""
    st = (_cosd(s1, prev_a), _cosd(s2, prev_b))
    fl = (_cosd(s2, prev_a), _cosd(s1, prev_b))
    return (s1, s2, *st) if max(st) <= max(fl) else (s2, s1, *fl)


def _pc_bracket(offs, em, match):
    """Locate the front on a corrector line. Any pair (i, j), 1 <= j - i <= PC_SEP_MAX, with 1-cos > COS_T that
    passes the side-signature test is evidence of it: soft fronts spread their change over several spacings (dev
    traces: line ends at 1-cos 0.6-0.95 while no adjacent pair cleared 0.35). Each such pair is localised to one
    spacing by labelling its interior points by the nearer end; the flip nearest the prediction (offset 0) wins,
    the narrower pair on ties. Returns ((k_lo, k_hi, i, j) or None, straddled)."""
    best, straddled = None, False
    n = len(offs)
    for i in range(n - 1):
        for j in range(i + 1, min(i + PC_SEP_MAX, n - 1) + 1):
            ea, eb = em[i], em[j]
            if ea is None or eb is None or _cosd(ea, eb) <= COS_T:
                continue
            straddled = True
            if not match(ea, eb):
                continue
            lo, hi = i, j
            for m in range(i + 1, j):
                if em[m] is None:
                    continue
                if _cosd(em[m], ea) < _cosd(em[m], eb):
                    lo = m
                else:
                    hi = m
                    break
            key = (abs(offs[lo] + offs[hi]), j - i)
            if best is None or key < best[0]:
                best = (key, (lo, hi, i, j))
    return (best[1] if best else None), straddled


def run_walk_continuation(app, run, pool, walk):
    """Predictor-corrector walk (pseudo-arclength continuation) in the plane span(t, n) -- the plane the fan
    samples, so for one (crossing, direction) both modes trace the same slice of the front.

    Each step predicts along the current tangent (t at first, then the secant through the last two captured
    points), renders a corrector line perpendicular to it at the chord spacing in ONE GPU round, keeps the
    straddle nearest the prediction that matches the origin's side signature, and bisects it once (station
    localised to +-PC_DS/4). A success grows the step x1.5 up to WALK_STEP; a miss halves it and retries from the
    same point; below PC_H_MIN the walk ends with the fan's failure notes. The budget is the fan's reach,
    n_steps * WALK_STEP of arclength, so the two modes are compared over the same distance."""
    su = _walk_setup(app, run, pool, walk)
    if su is None:
        return
    x0, nrm, t, match, prev_a, prev_b = su          # prev_*: the sides the next station must continue
    k = run.k
    E = np.stack([t, nrm], axis=1)                  # orthonormal frame of the walk plane
    budget = walk.n_steps * WALK_STEP
    p = np.zeros(2)                                 # in-plane position; the origin crossing
    tau = np.array([1.0, 0.0])                      # predictor tangent: t until the first secant exists
    h, arc, attempt = WALK_STEP, 0.0, 0
    while arc < budget - 1e-9 and walk.status == "running" and attempt < 4 * walk.n_steps + 4:
        attempt += 1
        j = len(walk.steps) + 1
        perp = np.array([-tau[1], tau[0]])          # +90 deg: the +n side at the origin
        # predictor, shortened to stay inside the simplex
        here, d = x0 + E @ p, E @ tau
        room = min(((here[i] - PC_EDGE) / -d[i] for i in range(k) if d[i] < -1e-12), default=np.inf)
        h_try = min(h, room)
        if h_try < PC_H_MIN:
            walk.notes.append(f"edge of the space at station {j}")
            break
        q = p + h_try * tau
        # corrector: chord-spaced points across the predicted point, inside the simplex
        m = int(round((PC_ARM_FIRST if j == 1 else PC_ARM) / PC_DS))
        offs = [i * PC_DS for i in range(-m, m + 1) if (x0 + E @ (q + i * PC_DS * perp)).min() >= 0]
        if len(offs) < 2:
            walk.notes.append(f"too close to a face at station {j}")
            break
        ws = [_clipn(x0 + E @ (q + s * perp), k) for s in offs]
        walk.segs.append([[float(v) for v in ws[0]], [float(v) for v in ws[-1]]])
        g = evaluate(app, run, pool, ws, run.seed, f"walk{walk.wid}pc{attempt}", ctl=walk)
        walk.images += len(ws)
        if walk.status != "running":
            break
        em = [run.embeddings.get(gi) if gi is not None else None for gi in g]
        cands, straddled = _pc_candidates(offs, em, match)
        # continuity: how far each side moved since the previous station (the origin for station 1), relative to
        # that station's own contrast -- a third basin appearing at a junction moves one side by about a full contrast
        c_prev = max(_cosd(prev_a, prev_b), 1e-6)
        scored = []
        for key, k_lo, k_hi, i_p, j_p in cands:
            s_a, s_b, d_a, d_b = _orient(prev_a, prev_b, em[i_p], em[j_p])
            scored.append((key, k_lo, k_hi, i_p, j_p, max(d_a, d_b) / c_prev, s_a, s_b))
        gamma = PC_CONT_GAMMA if walk.cont_gamma is None else walk.cont_gamma
        ok = [c for c in scored if walk.sig_mode != "continuity" or c[5] < gamma]
        br = min(ok, key=lambda c: c[0]) if ok else None
        adj = [None if em[i] is None or em[i + 1] is None else round(_cosd(em[i], em[i + 1]), 3)
               for i in range(len(offs) - 1)]
        ends = _cosd(em[0], em[-1]) if em[0] is not None and em[-1] is not None else None
        walk.trace.append({"j": j, "h": round(float(h_try), 4), "offs": [round(float(o), 4) for o in offs], "adj": adj,
                           "ends": None if ends is None else round(ends, 3),
                           "ends_match": bool(ends is not None and match(em[0], em[-1])),
                           "cands": [[int(c[1]), int(c[2]), int(c[3]), int(c[4]), round(float(c[0][0]), 4),
                                      round(float(c[5]), 3)] for c in scored],
                           "chosen": None if br is None else [int(br[1]), int(br[2]), int(br[3]), int(br[4])]})
        if br is None:
            h = h_try / 2
            if h < PC_H_MIN:
                walk.notes.append(f"ridge changed identity at station {j} (junction?)" if straddled
                                  else f"ridge ended before station {j}")
                break
            continue
        _, k_lo, k_hi, i_p, j_p, r_cont, prev_a, prev_b = br   # the accepted sides become the next reference
        walk.side_embs.append((prev_a, prev_b))
        lo, hi, e_lo, e_hi = offs[k_lo], offs[k_hi], em[k_lo], em[k_hi]
        contrast = _cosd(em[i_p], em[j_p])            # the pair that evidenced the front
        mid = (lo + hi) / 2
        gm = evaluate(app, run, pool, [_clipn(x0 + E @ (q + mid * perp), k)], run.seed,
                      f"walk{walk.wid}pcb{attempt}", ctl=walk)
        walk.images += 1
        e_m = run.embeddings.get(gm[0]) if gm and gm[0] is not None else None
        if e_m is not None:
            if _cosd(e_m, e_lo) < _cosd(e_m, e_hi):
                lo = mid
            else:
                hi = mid
        p_new = q + ((lo + hi) / 2) * perp
        step = float(np.linalg.norm(p_new - p))
        tau, p = (p_new - p) / step, p_new           # secant: the next predictor direction
        arc += step
        h = min(WALK_STEP, h_try * PC_GROW)
        pos = _clipn(x0 + E @ p, k)
        walk.steps.append(dict(weights=[float(v) for v in pos], contrast=float(contrast),
                               thumb=int(gm[0]) if e_m is not None else -1,
                               center=[float(v) for v in pos], cont=round(float(r_cont), 4)))
    if walk.status == "running":
        walk.status = "complete"


def render_points(app, run, pool, weights, seed, job):
    """Research tool: render arbitrary weight vectors through the cascade's own path (evaluate: same pipeline,
    mixing, steps and size as every walk and chord of this run) and save the DINOv2 embeddings to job["path"]
    (npz: weights, emb, idx). Used to build dense ground-truth lattices (tests/walk_gt_*)."""
    import types
    ctl = types.SimpleNamespace(status="running", notes=job.setdefault("notes", []))
    job["total"], job["done"] = len(weights), 0
    out = []
    for off in range(0, len(weights), 280):
        chunk = [np.asarray(w, dtype=float) for w in weights[off:off + 280]]
        out += evaluate(app, run, pool, chunk, seed, f"render{job['id']}o{off}", ctl=ctl)
        job["done"] = off + len(chunk)
        if ctl.status != "running":
            raise RuntimeError("render cancelled")
    dim = next(len(run.embeddings[gi]) for gi in out if gi is not None and gi in run.embeddings)
    emb = np.stack([run.embeddings[gi] if gi is not None and gi in run.embeddings else np.full(dim, np.nan)
                    for gi in out]).astype(np.float32)
    np.savez(job["path"], weights=np.asarray(weights, dtype=np.float64), emb=emb,
             idx=np.array([-1 if gi is None else gi for gi in out]), seed=seed, prompts=np.array(run.prompts),
             steps=run.steps, height=run.height, width=run.width, guidance=run.guidance_scale)
    job["status"] = "done"


CERT_SEEDS = (1, 2, 3)    # held out: seed = run.seed + 997*j -- the scoring seeds minus j = 0, which the walk used


def certify_walk(app, run, pool, walk):
    """Re-measure every captured station with the cascade's own statistic B (coupled-seed mean 1-cos across it at
    +-EPS/2) on seeds the walk never used, against the run's background at the same seeds. The direction is
    mode-agnostic -- the in-plane perpendicular of the walk's own polyline (origin -> stations) at each station --
    so fan and continuation walks are judged the same way. B above the threshold = a certified boundary point."""
    import types
    k = run.k
    pl = walk.plane
    if not walk.steps or pl is None:
        walk.cert = {"status": "done", "b": [], "significant": [], "threshold": None, "images": 0}
        return
    x0 = np.asarray(pl["x0"], dtype=float)
    E = np.stack([np.asarray(pl["t"], dtype=float), np.asarray(pl["n"], dtype=float)], axis=1)
    ctrs = [np.asarray(s.get("center") or s["weights"], dtype=float) for s in walk.steps]
    uv = [np.zeros(2)] + [E.T @ (c - x0) for c in ctrs]
    pairs = []
    for i, c in enumerate(ctrs, start=1):
        tan = (uv[i + 1] if i + 1 < len(uv) else uv[i]) - uv[i - 1]
        tn = np.linalg.norm(tan)
        tan = tan / tn if tn > 1e-12 else np.array([1.0, 0.0])
        nv = E @ np.array([-tan[1], tan[0]])
        pairs += [_clipn(c - (EPS / 2) * nv, k), _clipn(c + (EPS / 2) * nv, k)]
    ctl = types.SimpleNamespace(status="running", notes=walk.notes)
    seeds = [run.seed + 997 * j for j in CERT_SEEDS]
    per = [evaluate(app, run, pool, pairs, sd, f"walk{walk.wid}cert{sd}", ctl=ctl) for sd in seeds]
    b = []
    for i in range(len(ctrs)):
        ds = [_cosd(run.embeddings[g[2 * i]], run.embeddings[g[2 * i + 1]]) for g in per
              if g[2 * i] in run.embeddings and g[2 * i + 1] in run.embeddings]
        b.append(float(np.mean(ds)) if ds else None)
    bg = [float(np.mean([r[j] for j in CERT_SEEDS if r[j] is not None])) for r in run.bg_seed_vals
          if any(r[j] is not None for j in CERT_SEEDS)]
    thr, kind = (float(np.percentile(bg, 95)), "held-out seeds") if bg else (run.bg_p95, "run p95, all seeds")
    walk.cert = {"status": "done", "b": b, "threshold": thr, "threshold_kind": kind, "seeds": seeds,
                 "images": len(pairs) * len(seeds),
                 "significant": [bool(v is not None and thr is not None and v > thr) for v in b]}


# Trace phase (RESEARCH_ridge_following_k4.md §7). After the survey, walk every significant crossing both ways with
# the continuation walk, certify the stations, and link crossings a walk reaches: two crossings on one ridge are
# often too far apart for their side images to match (content changes along a ridge, h07), which splits one ridge
# into several "singletons" and inflates the Good-Turing unexplored share.
TRACE_LINK = 0.02         # a crossing this close to another crossing's walk polyline is on that walk's ridge
TRACE_SIG_MODE = "continuity"   # traces must not change ridge -- a switch would link two different ridges
TRACE_WORKERS = 4


def _seg_dist(p, a, b):
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / max(float(np.dot(ab, ab)), 1e-18), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def trace_links(run, walks):
    """Crossing pairs joined by a walk: crossing Y lies within TRACE_LINK of the polyline of a walk from X, and Y's
    sides pass the relative side test against the station ending that segment (so a walk that merely passes a
    different ridge's crossing does not claim it)."""
    by_cid = {x.cid: x for x in run.crossings}
    links = set()
    for w in walks:
        if not w.steps or len(w.side_embs) != len(w.steps):
            continue
        pts = [np.asarray(by_cid[w.cid].mid, dtype=float)] + [np.asarray(st["center"], dtype=float) for st in w.steps]
        for y in run.crossings:
            if y.cid == w.cid or y.mid is None or y.ea is None or y.eb is None:
                continue
            d = [_seg_dist(np.asarray(y.mid, dtype=float), pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
            i = int(np.argmin(d))
            if d[i] < TRACE_LINK and same_ridge_relative(*w.side_embs[i], y.ea, y.eb):
                links.add(tuple(sorted((w.cid, y.cid))))
    return sorted(links)


def run_trace(app, run, pool):
    """Walk both ways from every significant crossing (TRACE_WORKERS at a time), certify, and link."""
    from concurrent.futures import ThreadPoolExecutor
    jobs = [(x, d) for x in run.crossings if x.significant and x.mid is not None and x.n is not None
            for d in (1, -1)]
    if not jobs:
        run.notes.append("trace: no significant crossing to walk from")
        return
    run.phase, run.phase_total, run.phase_done = "trace", len(jobs) * 40, 0

    def go(xd):
        x, d = xd
        if run.status != "running":
            return None
        w = Walk(wid=f"t{x.cid}{'p' if d > 0 else 'm'}", cid=x.cid, direction=d, n_steps=run.trace_steps,
                 mode="continuation", sig_mode=TRACE_SIG_MODE)
        run_walk_continuation(app, run, pool, w)
        if run.trace_certify and w.steps and run.status == "running":
            certify_walk(app, run, pool, w)
        return w

    with ThreadPoolExecutor(TRACE_WORKERS) as ex:
        walks = [w for w in ex.map(go, jobs) if w is not None]
    run.walks = getattr(run, "walks", {})
    for w in walks:
        run.walks[w.wid] = w
    run.traces = [{"cid": w.cid, "direction": w.direction, "walk_id": w.wid,
                   "points": [[float(v) for v in next(x.mid for x in run.crossings if x.cid == w.cid)]]
                             + [st["center"] for st in w.steps],
                   "cert": (w.cert or {}).get("significant"), "end": (w.notes or ["budget reached"])[-1]}
                  for w in walks]
    run.trace_links = trace_links(run, walks)
    run.notes.append(f"trace: {len(walks)} walks, {sum(len(w.steps) for w in walks)} stations, "
                     f"{len(run.trace_links)} crossing links")


def _set_div(run, gi, d):
    """Record a point's local divergence for the live map colouring (max over
    the measurements that touched it)."""
    pos = run._geo_pos.get(gi)
    if pos is not None and d is not None:
        cur = run.probe_div[pos]
        run.probe_div[pos] = float(d) if cur is None else max(cur, float(d))


def _staged(run):
    return getattr(run, "probe_mode", "steps") == "staged"


def _extra_steps(run):
    """Steps for the high-resolution pass's probes: the cheap field beside cheap chord probes,
    full fidelity beside staged ones (their bracket ends are exact labels, and a sub-probe must be
    read on the same field as the ends it is compared with)."""
    return None if _staged(run) else run.probe_steps


def _staged_chords(app, run, pool, pts, meta, label, skip_first):
    """The chord probes of one generation in staged mode: x̂0 readouts of every probe, then the
    flagged segments' probes resumed to full fidelity (services/staged.py). Returns one image
    index per probe like evaluate(); only the RESUMED ones carry a label (run.embeddings), so
    _detect tests exactly the segments whose two ends were finished.

    The map is coloured by the x̂0 divergence as readouts land; a resumed probe's colour is then
    cleared so _detect's exact divergences replace it."""
    arrived = {}

    def _live(li, gi):
        arrived[li] = gi
        ci, ii = meta[li]
        for nb in (li - 1, li + 1):
            gj = arrived.get(nb)
            if gj is None:
                continue
            cj, jj = meta[nb]
            if cj != ci or abs(jj - ii) != 1:
                continue
            e1, e2 = run.xhat.get(gi), run.xhat.get(gj)
            if e1 is not None and e2 is not None:
                dd = _cosd(e1, e2)
                _set_div(run, gi, dd)
                _set_div(run, gj, dd)

    gidx = st.readout(app, run, pool, pts, label, on_arrival=_live)
    if run.status != "running":
        return gidx
    per = {}
    for li, (ci, i) in enumerate(meta):
        per.setdefault(ci, []).append((i, li))
    seqs = [[gidx[li] for _i, li in sorted(per[ci])] for ci in sorted(per)]
    _flags, sel = st.run_sequences(app, run, pool, seqs, label + "r",
                                   skip_first=[skip_first] * len(seqs))
    for gi in sel:
        pos = run._geo_pos.get(gi)
        if pos is not None and gi in run.embeddings:
            run.probe_div[pos] = None
    return gidx


def run_cascade(app, run, pool):
    """The full cascade. Called from the router's guarded thread body."""
    rng = np.random.default_rng(run.seed if run.chord_seed is None else run.chord_seed)
    k = run.k

    # ---------------- phase 1: chords (m=1, seed = run.seed), depth+1 generations
    run.phase = "chords"
    chords = []          # (w0, u, offs) per chord, indexed across ALL generations
    cand = []            # crossings, likewise
    max_chords = _chord_cap(run.n_chords, run.branch, run.depth)
    focus = np.asarray(run.focus, dtype=float) if run.focus else None
    if focus is not None:
        focus = np.clip(focus, 0, None)
        focus = focus / focus.sum()
    R = float(run.focus_radius)

    def _draw_anchor():
        if focus is None:
            return rng.dirichlet(np.ones(k))
        for _ in range(60):
            g = rng.standard_normal(k)
            g -= g.mean()
            gn = np.linalg.norm(g)
            if gn < 1e-9:
                continue
            w = focus + R * (rng.random() ** (1.0 / max(k - 1, 1))) * g / gn
            if w.min() >= 0:
                return w / w.sum()
        return focus.copy()

    def _focus_clip(w0, u, tneg, tpos):
        """Clip [tneg, tpos] to the focus ball so the budget stays in the region;
        None when the line misses the ball entirely."""
        d0 = w0 - focus
        b = float(np.dot(u, d0))
        cq = float(np.dot(d0, d0)) - R * R
        disc = b * b - cq
        if disc <= 0:
            return None
        root = float(np.sqrt(disc))
        return max(tneg, -b - root), min(tpos, -b + root)

    def _add_chord(w0, u, tneg, tpos, gen, parent, origin_cid, pts, meta):
        """Register one chord and queue its probes; -1 if it is too short to be worth probing.

        Keeps the PROBED offsets: focus mode and ray origins clip them, so recomputing from
        the simplex walls would misplace the bracket a crossing is quoted at.
        """
        if tpos - tneg < _min_chord_len(run.stride):
            return -1
        ci = len(chords)
        offs = _chord_offsets(tneg, tpos, run.stride)
        chords.append((w0, u, offs))
        run.chords_geo.append((
            [float(v) for v in np.clip(w0 + tneg * u, 0, None)],
            [float(v) for v in np.clip(w0 + tpos * u, 0, None)]))
        run.chords_meta.append({"gen": gen, "parent": parent, "origin_cid": origin_cid})
        for i, off in enumerate(offs):
            pts.append(np.clip(w0 + off * u, 0, None))
            meta.append((ci, i))
        return ci

    def _detect(gidx, meta, gen, skip_first):
        """Neighbour-pair crossing detection over ONE generation's probes; numbers the new
        crossings into `cand` and returns {chord index: [Crossing]} for the next generation.

        skip_first drops every chord's probe0-probe1 bracket: a child ray starts ON the
        crossing it was spawned from, so that bracket would re-count the parent's sheet.
        """
        per = {}
        for (ci, i), gi in zip(meta, gidx):
            if gi is not None and gi in run.embeddings:
                per.setdefault(ci, []).append((i, gi))
        found = {}
        for ci in sorted(per):
            seq = sorted(per[ci])
            for (i1, g1), (i2, g2) in zip(seq, seq[1:]):
                if i2 != i1 + 1 or (skip_first and i1 == 0):
                    continue
                e1, e2 = run.embeddings[g1], run.embeddings[g2]
                dd = _cosd(e1, e2)
                _set_div(run, g1, dd)
                _set_div(run, g2, dd)
                if dd > COS_T:
                    w0, u, offs = chords[ci]
                    found.setdefault(ci, []).append(Crossing(
                        cid=-1,
                        wa=np.clip(w0 + offs[i1] * u, 0, None),
                        wb=np.clip(w0 + offs[i2] * u, 0, None),
                        ea=e1, eb=e2, gen=gen,
                        # detection-only runs never render a mid image: show a bracket end
                        # instead (both are one half-stride from the midpoint they quote)
                        thumb=g1 if not run.certify else -1))
        for ci in sorted(found):
            for x in found[ci]:
                x.cid = len(cand)
                cand.append(x)
        return found

    run.phase_total = 0
    run.phase_done = 0
    found_prev = {}
    prev_x, origins = [], []      # generation g-1's crossings and the ones that spawned gen g
    for g in range(run.depth + 1 if run.branch > 0 else 1):
        if run.status != "running" or len(chords) >= max_chords:
            break
        chord_pts, chord_meta = [], []
        if g == 0:
            tries = 0
            max_tries = 200 * run.n_chords   # high k rejects many short chords; never spin
            anchors = []
            dirs = _dir_batch(k, rng, run.n_chords)
            while len(chords) < run.n_chords and tries < max_tries:
                tries += 1
                w0 = _blue_anchor(k, rng, anchors, draw=_draw_anchor)
                u = dirs[len(chords)] if len(chords) < len(dirs) else _iso_dir(k, rng)
                tneg, tpos = _extent(w0, u)
                if focus is not None:
                    cl = _focus_clip(w0, u, tneg, tpos)
                    if cl is None:
                        continue
                    tneg, tpos = cl
                if _add_chord(w0, u, tneg, tpos, 0, -1, -1,
                              chord_pts, chord_meta) < 0:
                    continue
                anchors.append(w0)
            if len(chords) < run.n_chords:
                run.notes.append(
                    f"only {len(chords)}/{run.n_chords} chords long enough at k={k}; "
                    "continuing")
        else:
            # A fan of RAYS out of the previous generation's most divergent crossings (the top
            # branch_top_pct %), from the midpoint of each one's bracket and forward only
            # (t >= 0), so the sheet it was spawned from sits at the origin and _detect's
            # first-bracket skip drops it.
            parent_of = {x.cid: ci for ci in found_prev for x in found_prev[ci]}
            prev_x = [x for ci in sorted(found_prev) for x in found_prev[ci]]
            origins = _branch_origins(prev_x, run.branch_top_pct)
            for x in origins:
                if len(chords) >= max_chords:
                    break
                w0 = np.clip((x.wa + x.wb) / 2, 0, None)
                # the fan maximises its angle to the chords ALREADY passing this origin (the
                # parent first of all) and, greedily, to the siblings drawn before it: near-parallel
                # rays would re-probe the sheet their neighbour already resolved
                near = _near_dirs(w0, run.chords_geo)
                for _ in range(run.branch):
                    if len(chords) >= max_chords:
                        break
                    u, ang = _spread_dir(near, rng, k)
                    _t, tpos = _extent(w0, u)
                    lo, hi = 0.0, tpos
                    if focus is not None:
                        cl = _focus_clip(w0, u, lo, hi)
                        if cl is None:
                            continue
                        lo, hi = cl
                    ci = _add_chord(w0, u, lo, hi, g, parent_of[x.cid], x.cid,
                                    chord_pts, chord_meta)
                    if ci < 0:
                        continue      # too short to probe: not a chord, so not a neighbour either
                    run.chords_meta[ci].update(min_angle_deg=ang, n_near=len(near))
                    near.append(u)
            if not chord_pts:
                break
        run.phase_total += len(chord_pts)
        # live colouring: the moment a probe AND its chord-neighbour both exist,
        # their pairwise divergence lands on the map -- no waiting for phase end
        arrived = {}

        def _live_div(li, gi, _meta=chord_meta, _arrived=arrived):
            _arrived[li] = gi
            ci, ii = _meta[li]
            for nb in (li - 1, li + 1):
                gj = _arrived.get(nb)
                if gj is None:
                    continue
                cj, jj = _meta[nb]
                if cj != ci or abs(jj - ii) != 1:
                    continue
                e1, e2 = run.embeddings.get(gi), run.embeddings.get(gj)
                if e1 is not None and e2 is not None:
                    dd = _cosd(e1, e2)
                    _set_div(run, gi, dd)
                    _set_div(run, gj, dd)

        if _staged(run):
            gidx = _staged_chords(app, run, pool, chord_pts, chord_meta,
                                  "chords" if g == 0 else f"chords{g}", skip_first=g > 0)
        else:
            gidx = evaluate(app, run, pool, chord_pts, run.seed,
                            "chords" if g == 0 else f"chords{g}",
                            on_arrival=_live_div, steps=run.probe_steps)
        if run.status != "running":
            return
        found_prev = _detect(gidx, chord_meta, g, skip_first=g > 0)
        if run.branch > 0:
            n_gen = sum(1 for m in run.chords_meta if m["gen"] == g)
            n_new = sum(len(v) for v in found_prev.values())
            spread = [m["min_angle_deg"] for m in run.chords_meta
                      if m["gen"] == g and m.get("min_angle_deg") is not None]
            run.notes.append(
                f"gen {g}: {n_gen} chords"
                + ("" if g == 0 else
                   f" from {len(origins)} origins "
                   f"(top {run.branch_top_pct} % of {len(prev_x)} crossings)")
                + f", {n_new} crossings"
                + (f", median spread {np.median(spread):.0f}°" if spread else ""))
        else:
            run.notes.append(f"{len(chords)} chords, {len(cand)} crossings")
    if run.branch > 0:
        n_root = sum(1 for m in run.chords_meta if m["gen"] == 0)
        n_rootx = sum(1 for x in cand if x.gen == 0)
        run.notes.append(
            f"{len(chords)} chords ({n_root} roots), {len(cand)} crossings "
            f"({n_rootx} on roots); the coverage certificate uses the roots only "
            "-- child rays are preferential samples"
            + (f"; chord cap {max_chords} reached" if len(chords) >= max_chords else ""))

    # ---------------- phase 1b: final high-resolution pass (optional, cheap field)
    # Runs on the finished chord phase and BEFORE anything in the certify half, so bisection
    # starts from the refined bracket and the detection-only finish quotes the refined centre.
    # Two modes over the SAME top-% selection: "bracket" probes finer along the chord (sharper
    # position), "cloud" probes a ball around the crossing (orientation, junctions) and leaves the
    # bracket -- so the width the certify half reads below stays None in cloud mode.
    hires_width = None
    if cand and run.hires and run.hires_mode == "cloud" and run.status == "running":
        run.phase = "hires"
        _hires_cloud(app, run, pool, _hires_select(cand, run.hires_top_pct), rng)
        if run.status != "running":
            return
    elif cand and run.hires and run.hires_factor >= 2 and run.status == "running":
        run.phase = "hires"
        f = int(run.hires_factor)
        n_before = len(cand)
        sel_x = _hires_select(cand, run.hires_top_pct)
        w_before = [float(np.linalg.norm(x.wa - x.wb)) for x in sel_x]
        pts, meta, ins_by = [], [], []     # meta: (index in sel_x, index among its probes)
        for si, x in enumerate(sel_x):
            ins = [np.clip(p, 0, None) for p in _hires_points(x.wa, x.wb, f)]
            ins_by.append(ins)
            for ii, p in enumerate(ins):
                pts.append(p)
                meta.append((si, ii))
        run.phase_total, run.phase_done = len(pts), 0
        arrived = {}

        def _live_hires(li, gi, _meta=meta, _arrived=arrived):
            """Live colouring as in the chord phase: two neighbouring probes of ONE refined
            bracket colour each other the moment both exist. The sub-steps touching the bracket
            ENDS are set after the batch -- the ends were rendered a phase ago and this run no
            longer holds their image indices."""
            _arrived[li] = gi
            si, ii = _meta[li]
            for nb in (li - 1, li + 1):
                gj = _arrived.get(nb)
                if gj is None or _meta[nb][0] != si or abs(_meta[nb][1] - ii) != 1:
                    continue
                e1, e2 = run.embeddings.get(gi), run.embeddings.get(gj)
                if e1 is not None and e2 is not None:
                    dd = _cosd(e1, e2)
                    _set_div(run, gi, dd)
                    _set_div(run, gj, dd)

        gh = evaluate(app, run, pool, pts, run.seed, "hires",
                      on_arrival=_live_hires, steps=_extra_steps(run))
        if run.status != "running":
            return
        by_sel = {}
        for (si, _ii), gi in zip(meta, gh):
            by_sel.setdefault(si, []).append(gi)
        hires_width = run.stride / f
        n_split, n_diffuse, w_after = 0, 0, []
        for si, x in enumerate(sel_x):
            gis = [None] + by_sel.get(si, [None] * (f - 1)) + [None]
            em = ([x.ea] + [run.embeddings.get(g) if g is not None else None
                            for g in gis[1:-1]] + [x.eb])
            seq = [x.wa] + ins_by[si] + [x.wb]
            for xx in _hires_refine(x, seq, em, hires_width, gis):
                xx.cid = len(cand)
                cand.append(xx)
                n_split += 1
            n_diffuse += 1 if x.hires_note == "diffuse" else 0
            w_after.append(float(np.linalg.norm(x.wa - x.wb)))
            # the refined probes are cloud points too: colour them by the sub-steps they end
            for i, d in enumerate(x.hires_profile or []):
                if d is None:
                    continue
                _set_div(run, gis[i], d)
                _set_div(run, gis[i + 1], d)
        mb = float(np.median(w_before)) if w_before else 0.0
        ma = float(np.median(w_after)) if w_after else 0.0
        run.notes.append(
            f"hires: refined {len(sel_x)} of {n_before} crossings "
            f"(top {run.hires_top_pct} %), ×{f}, {n_split} split, {n_diffuse} diffuse; "
            f"median width {mb:.4f}→{ma:.4f}")

    if _staged(run):
        # every bracket a staged survey quotes sits between two RESUMED probes (or full-fidelity
        # high-res sub-probes): its side labels are already the full-fidelity ones
        for x in cand:
            x.exact = True
        lg = st.ledger(run)
        rep = lg.report()
        run.notes.append(
            f"staged readout at step {run.staged_t}/{run.steps}, θ {run.staged_theta:g}: "
            f"{rep['flagged']}/{rep['segments']} segments flagged, {rep['resumed'] + rep['from_scratch']}"
            f"/{rep['readouts']} probes resumed"
            + (f" ({100 * rep['resumed_share']:.0f} %)" if rep['resumed_share'] is not None else "")
            + f", {rep['image_eq']:.1f} image-eq; crossings tested on exact labels only")
    if cand and run.certify and (_staged(run) or (run.probe_steps is not None
                                                  and run.probe_steps != run.steps)):
        # detection ran on the cheap field; hand bisection full-fidelity side images -- except for
        # brackets whose ends already are full-fidelity images (staged probes): nothing to redo
        todo = [x for x in cand if not x.exact]
        if len(todo) < len(cand):
            st.ledger(run).rebracket_skipped += len(cand) - len(todo)
            run.notes.append(f"rebracket skipped for {len(cand) - len(todo)} brackets whose ends "
                             "are resumed full-fidelity images")
        ws = []
        for x in todo:
            ws.append(np.clip(x.wa, 0, None))
            ws.append(np.clip(x.wb, 0, None))
        run.phase_total += len(ws)
        gi2 = evaluate(app, run, pool, ws, run.seed, "rebracket") if ws else []
        for i, x in enumerate(todo):
            ea = run.embeddings.get(gi2[2 * i]) if gi2[2 * i] is not None else None
            eb = run.embeddings.get(gi2[2 * i + 1]) if gi2[2 * i + 1] is not None else None
            if ea is not None:
                x.ea = ea
            if eb is not None:
                x.eb = eb
    run.crossings = cand
    if not cand:
        run.status = "complete"
        run.phase = "done"
        run.notes.append("no boundary crossings found -- try more chords or other prompts")
        return
    if not run.certify:
        # Detection only: every crossing is quoted at the centre of the cheap bracket that
        # found it, with the chord direction as its normal. Nothing downstream of the chords
        # runs, so b/significant stay unset -- but ea/eb are the probe embeddings the coverage
        # certificate and the local-density map read either way, so both still work.
        for x in cand:
            x.mid = (x.wa + x.wb) / 2
            d = x.wb - x.wa
            nn = np.linalg.norm(d)
            x.n = d / nn if nn > 1e-9 else _iso_dir(k, rng)
            x.b, x.significant = None, False
            _cloud_apply(x)      # a cloud outranks the bracket chord (see _cloud_apply)
        run.notes.append("detection only: crossings uncertified, positions at bracket "
                         "precision (±stride/2"
                         + (f"; ±{hires_width / 2:.4f} where the high-resolution pass refined"
                            if hires_width else "")
                         + "); no patches")
        if run.trace:
            run.notes.append("trace phase skipped: it walks certified crossings only")
        run.phase = "done"
        run.status = "complete"
        run.notes.append(f"total images: {run.generated}")
        return

    # ---------------- phase 2: bisection (parallel rounds)
    run.phase = "bisect"
    # progress denominator only: at a stride below BRACKET the brackets already arrive
    # narrow enough and the loop below runs zero rounds (the bar jumps straight to done).
    # The high-resolution pass hands bisection brackets of stride/factor, so it counts from
    # there -- at the default stride, a x4 pass already lands under BRACKET on its own.
    est_rounds = max(1, math.ceil(math.log2((hires_width or run.stride) / BRACKET)))
    run.phase_total = len(cand) * est_rounds
    run.phase_done = 0
    while run.status == "running":
        todo = [x for x in cand
                if np.linalg.norm(x.wa - x.wb) > BRACKET and x.ea is not None]
        if not todo:
            break
        mids = [(x.wa + x.wb) / 2 for x in todo]
        gidx = evaluate(app, run, pool, mids, run.seed, f"bisect{run.phase_done}")
        for x, wm, gi in zip(todo, mids, gidx):
            if gi is None or gi not in run.embeddings:
                x.wa = x.wb = (x.wa + x.wb) / 2      # give up on this one gracefully
                continue
            em = run.embeddings[gi]
            if x.ea is not None and x.eb is not None:
                _set_div(run, gi, _cosd(x.ea, x.eb))
            if _cosd(em, x.ea) < _cosd(em, x.eb):
                x.wa, x.ea = wm, em
            else:
                x.wb, x.eb = wm, em
            x.thumb = gi
    for x in cand:
        x.mid = (x.wa + x.wb) / 2
        d = x.wb - x.wa
        nn = np.linalg.norm(d)
        x.n = d / nn if nn > 1e-9 else _iso_dir(k, rng)
        # bisection narrowed the bracket, so its chord direction is the sharpest TRANSVERSAL it
        # has -- but a cloud measured the orientation itself, so it takes the normal (and the
        # position it estimated) back here. Precedence documented in _cloud_apply.
        _cloud_apply(x)
    if run.status != "running":
        return

    # ---------------- phase 3: B scoring (m=4 coupled seeds) + measured background
    run.phase = "score"
    n_bg = 12
    bg_pairs = []
    tries = 0
    while len(bg_pairs) < n_bg and tries < 200:
        tries += 1
        w0 = _draw_anchor()
        u = _iso_dir(k, rng)
        if (w0 + EPS * u).min() >= 0 and (w0 - EPS * u).min() >= 0:
            bg_pairs.append((w0, u))
    side_pts = []
    for x in cand:
        side_pts.append(np.clip(x.mid - (EPS / 2) * x.n, 0, None))
        side_pts.append(np.clip(x.mid + (EPS / 2) * x.n, 0, None))
    for w0, u in bg_pairs:
        side_pts.append(np.clip(w0 - (EPS / 2) * u, 0, None))
        side_pts.append(np.clip(w0 + (EPS / 2) * u, 0, None))
    seeds = [run.seed + 997 * j for j in range(M_SCORE)]
    run.phase_total = len(side_pts) * M_SCORE
    run.phase_done = 0
    per_seed = []
    for s in seeds:
        if run.status != "running":
            return
        per_seed.append(evaluate(app, run, pool, side_pts, s, f"score{s}"))
    n_units = len(cand) + len(bg_pairs)
    # Refresh side signatures at the CALIBRATED separation: bisection left
    # ea/eb ~0.006 apart (both blends), making same-ridge grouping and walk
    # checks spuriously strict. The scoring pass just rendered clean
    # EPS-separated sides -- store those instead.
    for ui, x in enumerate(cand):
        ga, gb = per_seed[0][2 * ui], per_seed[0][2 * ui + 1]
        if ga in run.embeddings and gb in run.embeddings:
            x.ea, x.eb = run.embeddings[ga], run.embeddings[gb]
    b_vals = []
    for ui in range(n_units):
        ds = []
        for gs in per_seed:
            ga, gb = gs[2 * ui], gs[2 * ui + 1]
            if ga in run.embeddings and gb in run.embeddings:
                ds.append(_cosd(run.embeddings[ga], run.embeddings[gb]))
        b_vals.append(float(np.mean(ds)) if ds else None)
    bg_vals = [v for v in b_vals[len(cand):] if v is not None]
    run.bg_seed_vals = [[_cosd(run.embeddings[gs[2 * ui]], run.embeddings[gs[2 * ui + 1]])
                         if gs[2 * ui] in run.embeddings and gs[2 * ui + 1] in run.embeddings else None
                         for gs in per_seed] for ui in range(len(cand), n_units)]
    if bg_vals:
        run.bg_mean = float(np.mean(bg_vals))
        run.bg_p95 = float(np.percentile(bg_vals, 95))
    for ui in range(n_units):
        for gs in per_seed:
            _set_div(run, gs[2 * ui], b_vals[ui])
            _set_div(run, gs[2 * ui + 1], b_vals[ui])
    for x, v in zip(cand, b_vals[:len(cand)]):
        x.b = v
        x.significant = bool(v is not None and run.bg_p95 is not None
                             and v > run.bg_p95)
    n_sig = sum(1 for x in cand if x.significant)
    run.notes.append(
        f"background mean {run.bg_mean:.3f} p95 {run.bg_p95:.3f}; "
        f"{n_sig}/{len(cand)} crossings significant" if bg_vals else
        "background scoring failed")
    if run.status != "running":
        return

    # ---------------- phase 4: patches (top-B dedup + one exploration slot)
    run.phase = "patches"
    ranked = sorted([x for x in cand if x.b is not None], key=lambda x: -x.b)
    sel = []
    for x in ranked:
        if all(np.linalg.norm(x.mid - s.mid) >= 0.15 for s in sel):
            sel.append(x)
        if len(sel) >= run.n_patches - 1:
            break
    rest = [x for x in ranked if all(x is not s for s in sel)]
    expl = None
    if rest:
        expl = rest[int(rng.integers(len(rest)))]
        sel.append(expl)
    run.phase_total = len(sel) * PATCH_N * PATCH_N
    run.phase_done = 0
    for ri, x in enumerate(sel):
        if run.status != "running":
            return
        # publish the patch BEFORE generating so the panel shows the contact sheet
        # assembling cell by cell (grid cells flip from -1 as images land)
        live = Patch(region=ri, cid=x.cid, b=float(x.b), significant=x.significant,
                     exploration=(x is expl),
                     grid=[[-1] * PATCH_N for _ in range(PATCH_N)],
                     cols_with_crossing=0)
        run.patches.append(live)
        t1 = rng.standard_normal(k)
        t1 -= t1.mean()
        t1 -= np.dot(t1, x.n) * x.n
        t1 /= max(np.linalg.norm(t1), 1e-9)
        pts, keys = [], []
        for i in range(-2, 3):          # tangent
            for l in range(-2, 3):      # across the boundary
                w = x.mid + CELL * (i * t1 + l * x.n)
                if w.min() < 0:
                    keys.append(None)
                    continue
                keys.append((i, l))
                pts.append(w)
        valid_keys = [kk for kk in keys if kk is not None]

        def _fill(li, gi, _keys=valid_keys, _live=live):
            if li < len(_keys):
                i0, l0 = _keys[li]
                _live.grid[i0 + 2][l0 + 2] = int(gi)

        gidx = evaluate(app, run, pool, pts, run.seed, f"patch{ri}",
                        on_arrival=_fill)
        gmap = {}
        it = iter(gidx)
        for kk in keys:
            if kk is not None:
                gmap[kk] = next(it)
        grid = [[gmap.get((i, l)) if gmap.get((i, l)) is not None else -1
                 for l in range(-2, 3)] for i in range(-2, 3)]
        for kk, gg in gmap.items():
            if gg is None or gg not in run.embeddings:
                continue
            i0, l0 = kk
            nb = gmap.get((i0, l0 + 1))
            if nb is not None and nb in run.embeddings:
                _set_div(run, gg, _cosd(run.embeddings[gg],
                                        run.embeddings[nb]))
        cols = 0
        for i in range(-2, 3):
            col = [(l, run.embeddings[gmap[(i, l)]]) for l in range(-2, 3)
                   if gmap.get((i, l)) is not None and gmap[(i, l)] in run.embeddings]
            if any(l2 == l1 + 1 and _cosd(e1, e2) > COS_T
                   for (l1, e1), (l2, e2) in zip(col, col[1:])):
                cols += 1
        live.grid = grid
        live.cols_with_crossing = cols
    if run.trace and run.status == "running":
        run_trace(app, run, pool)
    if run.status != "running":
        return
    run.phase = "done"
    run.status = "complete"
    run.notes.append(f"total images: {run.generated}")
