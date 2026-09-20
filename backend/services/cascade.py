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
    status: str = "running"
    phase: str = "chords"
    chords_geo: list = field(default_factory=list)   # [(a_weights, b_weights)] for the map
    probe_geo: list = field(default_factory=list)    # completed-point positions, live map cloud
    phase_done: int = 0
    phase_total: int = 0
    generated: int = 0
    crossings: list = field(default_factory=list)
    bg_mean: float | None = None
    bg_p95: float | None = None
    patches: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    error: str | None = None
    embeddings: dict = field(default_factory=dict)   # global idx -> np.ndarray (768,)
    thumbs: object = None                            # ThumbnailStore, set by router
    next_idx: int = 0
    recent_thumbs: list = field(default_factory=list)   # last ~12 generated indices
    probe_div: list = field(default_factory=list)       # local divergence per cloud point
    _geo_pos: dict = field(default_factory=dict, repr=False)  # gi -> index in probe_geo
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


def _extent(w, u):
    tpos = min((w[i] / -u[i]) for i in range(len(w)) if u[i] < 0)
    tneg = max((-w[i] / u[i]) for i in range(len(w)) if u[i] > 0)
    return tneg, tpos


def _cosd(a, b):
    return 1.0 - float(np.dot(a, b))


def _norm_emb(e):
    e = np.asarray(e, dtype=np.float64)
    n = np.linalg.norm(e)
    return e / n if n > 0 else e


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


def coverage_stats(crossings):
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
            a1, b1 = xs[i].ea, xs[i].eb
            a2, b2 = xs[j].ea, xs[j].eb
            straight = max(_cosd(a1, a2), _cosd(b1, b2))
            flipped = max(_cosd(a1, b2), _cosd(b1, a2))
            if min(straight, flipped) < COS_T:
                ri, rj = find(i), find(j)
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
    steps: list = field(default_factory=list)
    segs: list = field(default_factory=list)    # transversal probe lines, for the map
    notes: list = field(default_factory=list)
    error: str | None = None


def same_ridge(ea1, eb1, ea2, eb2):
    """Order-invariant side-signature match -- the coverage-grouping criterion."""
    straight = max(_cosd(ea1, ea2), _cosd(eb1, eb2))
    flipped = max(_cosd(ea1, eb2), _cosd(eb1, ea2))
    return min(straight, flipped) < COS_T


WALK_STEP = 0.05          # tangent step per click-step
WALK_PROBE = 0.024        # transversal probe arm (matches the patch cell scale)
WALK_WIDE = 0.06          # one widened retry before declaring the ridge lost


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
    origin = None
    for c in run.crossings:
        if c.cid == walk.cid and c.mid is not None and c.n is not None:
            origin = c
            break
    if origin is None:
        walk.status = "error"
        walk.error = "crossing not found or not yet bisected"
        return
    x0, nrm = origin.mid.copy(), origin.n.copy()
    sig_a, sig_b = origin.ea, origin.eb
    k = run.k
    rng = np.random.default_rng(run.seed + 7919 * walk.cid)
    t = rng.standard_normal(k)
    t -= t.mean()
    t -= np.dot(t, nrm) * nrm
    tn = np.linalg.norm(t)
    if tn < 1e-9:
        walk.status = "error"
        walk.error = (f"no tangent direction (k={k}, tn={tn:.2e}, "
                      f"|nrm|={float(np.linalg.norm(nrm)):.3f}, "
                      f"nrm_dim={len(nrm)}, mid_dim={len(x0)})")
        return
    t = walk.direction * t / tn      # at k=3: THE ridge direction (up to sign)

    def clipn(w):
        w = np.clip(w, 0, None)
        s = w.sum()
        return w / s if s > 0 else np.full(k, 1.0 / k)

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
    for i, s in enumerate(stations):
        ga, gb = gp[2 * i], gp[2 * i + 1]
        s["ea"] = run.embeddings.get(ga) if ga is not None else None
        s["eb"] = run.embeddings.get(gb) if gb is not None else None
        s["lo"], s["hi"] = -s["arm"], s["arm"]
        s["ok"] = (s["ea"] is not None and s["eb"] is not None
                   and _cosd(s["ea"], s["eb"]) > COS_T
                   and same_ridge(s["ea"], s["eb"], sig_a, sig_b))
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
        for i, s in enumerate(retry):
            ga, gb = gp[2 * i], gp[2 * i + 1]
            ea2 = run.embeddings.get(ga) if ga is not None else None
            eb2 = run.embeddings.get(gb) if gb is not None else None
            if (ea2 is not None and eb2 is not None and _cosd(ea2, eb2) > COS_T
                    and same_ridge(ea2, eb2, sig_a, sig_b)):
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
        walk.steps.append(dict(weights=[float(v) for v in pos],
                               contrast=float(_cosd(s["ea"], s["eb"])
                                              if s["ea"] is not None
                                              and s["eb"] is not None else 0.0),
                               thumb=int(s.get("gmid", -1) if s.get("gmid") is not None
                                         else -1)))
    if walk.status == "running":
        walk.status = "complete"


def _set_div(run, gi, d):
    """Record a point's local divergence for the live map colouring (max over
    the measurements that touched it)."""
    pos = run._geo_pos.get(gi)
    if pos is not None and d is not None:
        cur = run.probe_div[pos]
        run.probe_div[pos] = float(d) if cur is None else max(cur, float(d))


def run_cascade(app, run, pool):
    """The full cascade. Called from the router's guarded thread body."""
    rng = np.random.default_rng(run.seed)
    k = run.k

    # ---------------- phase 1: chords (m=1, seed = run.seed)
    run.phase = "chords"
    chords = []
    chord_pts, chord_meta = [], []
    tries = 0
    max_tries = 200 * run.n_chords          # high k rejects many short chords; never spin
    anchors = []
    dirs = _dir_batch(k, rng, run.n_chords)
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
    while len(chords) < run.n_chords and tries < max_tries:
        tries += 1
        w0 = _blue_anchor(k, rng, anchors, draw=_draw_anchor)
        u = dirs[len(chords)] if len(chords) < len(dirs) else _iso_dir(k, rng)
        tneg, tpos = _extent(w0, u)
        if focus is not None:
            # clip the chord to the focus ball so the budget stays in the region
            d0 = w0 - focus
            b = float(np.dot(u, d0))
            cq = float(np.dot(d0, d0)) - R * R
            disc = b * b - cq
            if disc <= 0:
                continue
            root = float(np.sqrt(disc))
            tneg = max(tneg, -b - root)
            tpos = min(tpos, -b + root)
        if tpos - tneg < 6 * STRIDE:
            continue
        ci = len(chords)
        chords.append((w0, u, tneg))     # keep the PROBED offset: focus mode clips it,
        anchors.append(w0)               # so recomputing from walls would misplace brackets
        run.chords_geo.append((
            [float(v) for v in np.clip(w0 + tneg * u, 0, None)],
            [float(v) for v in np.clip(w0 + tpos * u, 0, None)]))
        for i in range(int((tpos - tneg) / STRIDE) + 1):
            chord_pts.append(np.clip(w0 + (tneg + i * STRIDE) * u, 0, None))
            chord_meta.append((ci, i))
    if len(chords) < run.n_chords:
        run.notes.append(
            f"only {len(chords)}/{run.n_chords} chords long enough at k={k}; continuing")
    run.phase_total = len(chord_pts)
    run.phase_done = 0
    # live colouring: the moment a probe AND its chord-neighbour both exist,
    # their pairwise divergence lands on the map -- no waiting for phase end
    arrived = {}

    def _live_div(li, gi):
        arrived[li] = gi
        ci, ii = chord_meta[li]
        for nb in (li - 1, li + 1):
            gj = arrived.get(nb)
            if gj is None:
                continue
            cj, jj = chord_meta[nb]
            if cj != ci or abs(jj - ii) != 1:
                continue
            e1, e2 = run.embeddings.get(gi), run.embeddings.get(gj)
            if e1 is not None and e2 is not None:
                dd = _cosd(e1, e2)
                _set_div(run, gi, dd)
                _set_div(run, gj, dd)

    gidx = evaluate(app, run, pool, chord_pts, run.seed, "chords",
                    on_arrival=_live_div, steps=run.probe_steps)
    if run.status != "running":
        return
    per = {}
    for (ci, i), gi in zip(chord_meta, gidx):
        if gi is not None and gi in run.embeddings:
            per.setdefault(ci, []).append((i, gi))
    cand = []
    for ci, seq in per.items():
        seq.sort()
        for (i1, g1), (i2, g2) in zip(seq, seq[1:]):
            if i2 == i1 + 1:
                e1, e2 = run.embeddings[g1], run.embeddings[g2]
                dd = _cosd(e1, e2)
                _set_div(run, g1, dd)
                _set_div(run, g2, dd)
                if dd > COS_T:
                    w0, u, tneg = chords[ci]
                    cand.append(Crossing(
                        cid=len(cand),
                        wa=np.clip(w0 + (tneg + i1 * STRIDE) * u, 0, None),
                        wb=np.clip(w0 + (tneg + i2 * STRIDE) * u, 0, None),
                        ea=e1, eb=e2))
    run.notes.append(f"{len(chords)} chords, {len(cand)} crossings")
    if cand and run.probe_steps is not None and run.probe_steps != run.steps:
        # detection ran on the cheap field; hand bisection full-fidelity side images
        ws = []
        for x in cand:
            ws.append(np.clip(x.wa, 0, None))
            ws.append(np.clip(x.wb, 0, None))
        run.phase_total += len(ws)
        gi2 = evaluate(app, run, pool, ws, run.seed, "rebracket")
        for i, x in enumerate(cand):
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

    # ---------------- phase 2: bisection (parallel rounds)
    run.phase = "bisect"
    est_rounds = max(1, math.ceil(math.log2(STRIDE / BRACKET)))
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
    run.phase = "done"
    run.status = "complete"
    run.notes.append(f"total images: {run.generated}")
