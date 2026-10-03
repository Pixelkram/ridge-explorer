"""Mixing desk: "what happens if I turn prompt i up or down", and which prompt flips first.

WeightLifter (Pajer, Streit, Torsney-Weir, Spechtenhauser, Moeller & Piringer, IEEE TVCG 23(1)
2017, sections 4.2-4.5) gives one slider per criterion, each a straight line through the
current weighting along which that criterion is traded against ALL the others with the others'
proportions preserved, and draws inside each slider the places where the best outcome switches.
Here the criteria are the k prompts and "the outcome" is the image.

The line of prompt i through the current mix w0
-----------------------------------------------
    w_i = alpha in [0, 1],     w_j = w0_j * (1 - alpha) / (1 - w0_i)   for j != i,

so the weights keep summing to 1, the others keep their proportions, alpha = w0_i is w0 itself,
alpha = 1 is prompt i alone and alpha = 0 is the face without it. When w0_i = 1 the others'
proportions are undefined (w0 is the vertex); the line then shares the rest EQUALLY among the
others and says so (`degenerate`).

A line depends on w0 only through the others' proportions, so it is the same line for every
mix along it: moving prompt i's marker keeps line i, and only the other k - 1 lines are new.
That is the cache key here (`line_key`) -- it covers every revisit of (w0, i) and every move
along line i too.

Reading a line
--------------
Each line is sampled every dalpha (default 0.05, both ends included) with the Cascade chord
phase's cheap probes (`cascade.evaluate` at probe_steps, 0.5 image-eq each), plus ONE probe at
w0 itself, shared by all k lines. A label change between two consecutive samples is the
Cascade's own rule, cosine distance > COS_T. The optional REFINE re-renders the two ends of
each change at full fidelity and bisects it three times (to dalpha/8), batched across every
line -- the Cascade's rebracket + bisection, at most REFINE_MAX_PER_LINE changes per line,
strongest first, so its worst case is known before it starts.

Readout: per prompt, the alpha-distance from w0_i to the nearest change upward and downward,
with the image just beyond it, prompts sorted by the nearer of the two. This is a single-seed,
one-line-per-prompt reading of a margin whose calibration is the subject of the frozen h26a
pre-registration, so it carries READOUT_LABEL everywhere it is shown.
"""
from dataclasses import dataclass, field
import math
import threading

import numpy as np

from backend.services import cascade as cs
from backend.services.cascade import COS_T, _cosd
from backend.services import gridfree as gf
from backend.services import staged as stg

DALPHA_DEFAULT = 0.05
PROBE_COST = 0.5               # a cheap probe (probe_steps = 4 of 8): half an image
FULL_COST = 1.0
REFINE_ROUNDS = 3              # dalpha / 2^3: the refined bracket is dalpha/8 wide
REFINE_MAX_PER_LINE = 4        # changes refined per line, largest divergence first
REFINE_IMAGES = 2 + REFINE_ROUNDS   # full-fidelity ends + one midpoint per round
# Ceilings. One request (a start or a move) at k = 8 and the default stride is 169 probes =
# 85 image-eq, 245 with refine; a stride of 0.02 with refine at k = 8 is refused.
MAX_REQUEST_IMAGE_EQ = 300
MAX_SESSION_IMAGE_EQ = 3000
DEGENERATE_EPS = 1e-9
PROP_DIGITS = 6                # a line's identity: the others' proportions, to 1e-6
READOUT_LABEL = "exploratory — calibration pending (h26a)"


# ---- pure geometry --------------------------------------------------------------------

def is_degenerate(w0, i):
    """True when w0 is (numerically) vertex i: the others' proportions are undefined."""
    return 1.0 - float(w0[i]) <= DEGENERATE_EPS


def line_point(w0, i, alpha):
    """The recipe at alpha on prompt i's line through w0 (WeightLifter's slider line)."""
    w0 = np.asarray(w0, dtype=np.float64)
    k = len(w0)
    rest = 1.0 - float(w0[i])
    a = float(alpha)
    if rest > DEGENERATE_EPS:
        w = w0 * ((1.0 - a) / rest)
    else:
        w = np.full(k, (1.0 - a) / (k - 1))
    w[i] = a
    return w


def alpha_grid(dalpha):
    """0, h, 2h, ..., 1 with h = 1/ceil(1/dalpha) <= dalpha: both ends always sampled."""
    n = max(1, int(math.ceil(1.0 / float(dalpha) - 1e-9)))
    return [t / n for t in range(n + 1)]


def line_key(w0, i, dalpha, refine):
    """A line's identity: (i, the others' proportions, the stride, refine).

    Independent of w0_i, so every mix ON the line has the same key.
    """
    w0 = np.asarray(w0, dtype=np.float64)
    k = len(w0)
    if is_degenerate(w0, i):
        props = tuple(round(1.0 / (k - 1), PROP_DIGITS) if j != i else 0.0 for j in range(k))
    else:
        rest = 1.0 - float(w0[i])
        props = tuple(round(float(w0[j]) / rest, PROP_DIGITS) if j != i else 0.0
                      for j in range(k))
    return (int(i), props, round(float(dalpha), 9), bool(refine))


def detect_flips(alphas, embs, gis=None):
    """Consecutive sample pairs further apart than COS_T: the Cascade's crossing rule.

    `embs` aligned with `alphas`, None where a sample never arrived (a pair with a missing end
    is no evidence either way). Each flip is quoted at its bracket's midpoint.
    """
    out = []
    for t in range(len(alphas) - 1):
        a, b = embs[t], embs[t + 1]
        if a is None or b is None:
            continue
        d = float(_cosd(a, b))
        if d > COS_T:
            out.append({"lo": t, "hi": t + 1, "alpha": (alphas[t] + alphas[t + 1]) / 2.0,
                        "div": d, "refined": False, "width": alphas[t + 1] - alphas[t],
                        "confirmed": None, "full_div": None,
                        "img_lo": -1 if gis is None or gis[t] is None else int(gis[t]),
                        "img_hi": -1 if gis is None or gis[t + 1] is None else int(gis[t + 1])})
    return out


def segments(alphas, flips, gis):
    """The stretches between changes: sample range, alpha extent (flip to flip, 0 and 1 at the
    ends) and one thumbnail -- the rendered sample nearest the stretch's middle."""
    cuts = sorted(flips, key=lambda f: f["lo"])
    out, start, a_lo = [], 0, 0.0
    bounds = [(f["lo"], f["hi"], f["alpha"]) for f in cuts] + [(len(alphas) - 1, None, 1.0)]
    for lo, hi, a_hi in bounds:
        end = lo
        idx = [t for t in range(start, end + 1) if gis[t] is not None]
        mid = (start + end) / 2.0
        thumb = int(gis[min(idx, key=lambda t: (abs(t - mid), t))]) if idx else -1
        out.append({"lo": start, "hi": end, "alpha_lo": float(a_lo), "alpha_hi": float(a_hi),
                    "thumb": thumb})
        if hi is None:
            break
        start, a_lo = hi, a_hi
    return out


def nearest_flips(alpha0, alphas, flips, side=None):
    """(up, down): the nearest change above and below alpha0, or None either way.

    A change's direction is the sign of (its alpha - alpha0), except for the one whose sample
    bracket strictly CONTAINS alpha0: there the midpoint can sit on the wrong side of the mix,
    so `side(flip)` decides ("up" = the mix matched the bracket's lower end) when it can.
    Each is {dist, alpha, thumb, refined}; the thumbnail is the image just beyond the change.
    """
    up = down = None
    for f in flips:
        lo_a, hi_a = alphas[f["lo"]], alphas[f["hi"]]
        if lo_a < alpha0 < hi_a and side is not None:
            way = side(f) or ("up" if f["alpha"] >= alpha0 else "down")
        else:
            way = "up" if f["alpha"] > alpha0 else "down" if f["alpha"] < alpha0 else (
                (side(f) if side is not None else None) or "up")
        ref = {"dist": abs(float(f["alpha"]) - float(alpha0)), "alpha": float(f["alpha"]),
               "thumb": int(f["img_hi"] if way == "up" else f["img_lo"]),
               "refined": bool(f["refined"])}
        if way == "up" and (up is None or ref["dist"] < up["dist"]):
            up = ref
        if way == "down" and (down is None or ref["dist"] < down["dist"]):
            down = ref
    return up, down


def probe_cost(probe_steps):
    return FULL_COST if probe_steps is None else PROBE_COST


def projected_cost(k, n_samples, refine, probe_steps=4, n_cached_lines=0, w0_cached=False,
                   probe_mode="steps", staged_t=stg.T_DEFAULT, steps=8):
    """(image-eq, rows) one request would spend at most: k x samples cheap probes, the mix's
    own probe, and -- with refine -- REFINE_MAX_PER_LINE changes x REFINE_IMAGES full images
    per new line (the worst case; a line with fewer changes costs less). Cached lines are free.

    Staged: every sample read out at t/S, the worst case resumes every sample ((S-t)/S each),
    the mix is read out AND finished (1), and refine needs only its REFINE_ROUNDS midpoints per
    change -- the change's ends are finished images already.
    """
    n_lines = max(0, int(k) - int(n_cached_lines))
    if probe_mode == "staged":
        S, t = int(steps), int(staged_t)
        n = n_lines * int(n_samples)
        rows = [{"what": f"{n_lines} lines x {int(n_samples)} samples read out to step {t}/{S}",
                 "probes": n, "image_eq": t / S * n},
                {"what": "resume, worst case (every segment flagged)", "probes": n,
                 "image_eq": (S - t) / S * n}]
        if not w0_cached:
            rows.append({"what": "the current mix (read out and finished)", "probes": 1,
                         "image_eq": FULL_COST})
        if refine and n_lines:
            nr = n_lines * REFINE_MAX_PER_LINE * REFINE_ROUNDS
            rows.append({"what": f"refine (worst case {REFINE_MAX_PER_LINE} changes/line x "
                                 f"{REFINE_ROUNDS} midpoints; the ends are finished already)",
                         "probes": nr, "image_eq": FULL_COST * nr})
        return float(sum(r["image_eq"] for r in rows)), rows
    unit = probe_cost(probe_steps)
    rows = [{"what": f"{n_lines} lines x {int(n_samples)} samples",
             "probes": n_lines * int(n_samples),
             "image_eq": unit * n_lines * int(n_samples)}]
    if not w0_cached:
        rows.append({"what": "the current mix", "probes": 1, "image_eq": unit})
    if refine and n_lines:
        n = n_lines * REFINE_MAX_PER_LINE * REFINE_IMAGES
        rows.append({"what": f"refine (worst case {REFINE_MAX_PER_LINE} changes/line x "
                             f"{REFINE_IMAGES} full images)", "probes": n,
                     "image_eq": FULL_COST * n})
    return float(sum(r["image_eq"] for r in rows)), rows


# ---- the desk -------------------------------------------------------------------------

@dataclass
class Position:
    """One current mix and its k lines. Also the `ctl` its batches run under, so a newer move
    can cancel exactly this one (`cascade.evaluate` reads ctl.status / ctl.notes)."""
    pid: int
    w0: list
    dalpha: float
    refine: bool
    status: str = "pending"
    lines: list = field(default_factory=list)
    cached: list = field(default_factory=list)      # per line: served from the line cache
    w0_image: int = -1
    readout: list = field(default_factory=list)
    cost: float = 0.0
    notes: list = field(default_factory=list)
    error: str | None = None


@dataclass
class DeskRun:
    """A desk: the prompts and image settings, every mix visited, and the line cache.

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
    probe_steps: int | None = 4
    # "steps" (the cheap probes above) or "staged" (services/staged.py): samples read out at step
    # staged_t of the full schedule, segments >= staged_theta apart resumed to exact labels
    probe_mode: str = "steps"
    staged_t: int = stg.T_DEFAULT
    staged_theta: float = stg.THETA_DEFAULT
    dalpha: float = DALPHA_DEFAULT
    refine: bool = False
    positions: list = field(default_factory=list)
    position_index: dict = field(default_factory=dict)
    current: int = -1
    line_cache: dict = field(default_factory=dict)
    n_probes: int = 0
    n_full: int = 0
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
    # staged probes (the contract staged.readout / staged.resume read): x̂0 readouts, recipes,
    # latent-cache keys, unfinished previews, the cost ledger, and the readout cache
    # (recipe -> image index) that keeps a recipe from being read out twice
    xhat: dict = field(default_factory=dict, repr=False)
    staged_w: dict = field(default_factory=dict, repr=False)
    staged_lat: dict = field(default_factory=dict, repr=False)
    staged_preview: set = field(default_factory=set, repr=False)
    staged_ledger: object = field(default=None, repr=False)
    xhat_cache: dict = field(default_factory=dict, repr=False)

    @property
    def k(self):
        return len(self.prompts)

    @property
    def staged(self):
        return self.probe_mode == "staged"

    @property
    def status(self):
        if 0 <= self.current < len(self.positions):
            return self.positions[self.current].status
        return "complete"

    @property
    def cost_image_eq(self):
        lg = self.staged_ledger
        return float(probe_cost(self.probe_steps) * self.n_probes + FULL_COST * self.n_full
                     + (lg.image_eq if lg is not None else 0.0))

    def cached_lines(self, w0, dalpha, refine):
        """How many of the k lines through w0 are already in the line cache."""
        return sum(1 for i in range(self.k)
                   if line_key(w0, i, dalpha, refine) in self.line_cache)

    def w0_cached(self, w0):
        if self.staged:          # the mix is always finished in staged mode
            gi = self.point_cache.get(gf.point_key(w0, self.steps))
            return gi is not None and gi in self.embeddings
        gi = self.point_cache.get(gf.point_key(w0, self.steps if self.probe_steps is None
                                               else self.probe_steps))
        return gi is not None and gi in self.embeddings

    def projected(self, w0, dalpha, refine, n_cached_lines=None):
        """projected_cost of a request at (w0, dalpha, refine) on this desk."""
        n_cached = self.cached_lines(w0, dalpha, refine) if n_cached_lines is None else n_cached_lines
        return projected_cost(self.k, len(alpha_grid(dalpha)), refine, self.probe_steps,
                              n_cached_lines=n_cached, w0_cached=self.w0_cached(w0),
                              probe_mode=self.probe_mode, staged_t=self.staged_t, steps=self.steps)


def position_key(w0, dalpha, refine):
    return (tuple(round(float(v), gf.KEY_DIGITS) for v in w0), round(float(dalpha), 9),
            bool(refine))


def add_position(run, w0, dalpha, refine):
    """The position at (w0, dalpha, refine), created if new; it becomes the current one.
    Returns (position, created). A finished position is reused as it stands."""
    key = position_key(w0, dalpha, refine)
    if key in run.position_index:
        pos = run.positions[run.position_index[key]]
        if pos.status == "complete":
            run.current = pos.pid
            return pos, False
    pos = Position(pid=len(run.positions), w0=[float(v) for v in w0], dalpha=float(dalpha),
                   refine=bool(refine))
    run.position_index[key] = pos.pid
    run.positions.append(pos)
    run.current = pos.pid
    return pos, True


def _new_line(w0, i, dalpha, refine):
    alphas = alpha_grid(dalpha)
    return {"i": int(i), "key": line_key(w0, i, dalpha, refine), "w0": [float(v) for v in w0],
            "alphas": alphas, "ws": [line_point(w0, i, a) for a in alphas],
            "images": [-1] * len(alphas), "divs": [None] * (len(alphas) - 1), "flips": [],
            "segments": [], "status": "running", "degenerate": is_degenerate(w0, i),
            "resumed": [], "flagged": []}


def _refine(app, run, pool, pos, lines):
    """Full-fidelity rebracket + REFINE_ROUNDS bisections of every change on `lines` (at most
    REFINE_MAX_PER_LINE per line, largest divergence first), batched across all of them."""
    todo = []
    for ln in lines:
        for f in sorted(ln["flips"], key=lambda f: (-f["div"], f["lo"]))[:REFINE_MAX_PER_LINE]:
            todo.append((ln, f))
    if not todo:
        return
    ends = []
    for ln, f in todo:
        ends += [ln["ws"][f["lo"]], ln["ws"][f["hi"]]]
    g, n_new = gf.render_cached(app, run, pool, ends, None, f"desk{pos.pid}ends", pos)
    run.n_full += n_new
    pos.cost += FULL_COST * n_new
    state = []
    for j, (ln, f) in enumerate(todo):
        glo, ghi = g[2 * j], g[2 * j + 1]
        elo = run.embeddings.get(glo) if glo is not None else None
        ehi = run.embeddings.get(ghi) if ghi is not None else None
        if elo is None or ehi is None:
            continue                          # an end never arrived: the cheap bracket stands
        f["full_div"] = float(_cosd(elo, ehi))
        f["confirmed"] = bool(f["full_div"] > COS_T)
        state.append([ln, f, ln["alphas"][f["lo"]], ln["alphas"][f["hi"]], elo, ehi, glo, ghi])
    for r in range(REFINE_ROUNDS):
        if not state or pos.status != "running":
            return
        mids = [line_point(st[0]["w0"], st[0]["i"], (st[2] + st[3]) / 2.0) for st in state]
        gm, n_new = gf.render_cached(app, run, pool, mids, None, f"desk{pos.pid}bis{r}", pos)
        run.n_full += n_new
        pos.cost += FULL_COST * n_new
        for st, gi in zip(state, gm):
            em = run.embeddings.get(gi) if gi is not None else None
            if em is None:
                continue
            am = (st[2] + st[3]) / 2.0
            if _cosd(em, st[4]) < _cosd(em, st[5]):
                st[2], st[4], st[6] = am, em, gi
            else:
                st[3], st[5], st[7] = am, em, gi
    for ln, f, alo, ahi, _elo, _ehi, glo, ghi in state:
        f["alpha"] = (alo + ahi) / 2.0
        f["width"] = ahi - alo
        f["refined"] = True
        f["img_lo"], f["img_hi"] = int(glo), int(ghi)


def readout(run, pos):
    """Per prompt: the nearest change up and down from w0_i, sorted by the nearer one."""
    e0 = run.embeddings.get(pos.w0_image) if pos.w0_image >= 0 else None
    rows = []
    for ln in pos.lines:
        a0 = float(pos.w0[ln["i"]])

        def side(f, ln=ln):
            if e0 is None:
                return None
            glo, ghi = ln["images"][f["lo"]], ln["images"][f["hi"]]
            elo, ehi = run.embeddings.get(glo), run.embeddings.get(ghi)
            if elo is None or ehi is None:
                return None
            return "up" if _cosd(e0, elo) <= _cosd(e0, ehi) else "down"
        up, down = nearest_flips(a0, ln["alphas"], ln["flips"], side)
        ds = [d["dist"] for d in (up, down) if d is not None]
        rows.append({"i": ln["i"], "up": up, "down": down,
                     "nearest": min(ds) if ds else None})
    rows.sort(key=lambda r: (r["nearest"] is None, r["nearest"] or 0.0, r["i"]))
    return rows


def _readout_cached(app, run, pool, weights, label, ctl, on_arrival=None):
    """staged.readout through the desk's readout cache: a recipe is read out at most once per
    desk, duplicates in one batch once (render_cached's discipline). Returns gis per input."""
    tag = ("staged", int(run.staged_t))
    gis = [None] * len(weights)
    todo, todo_of = [], {}
    for i, w in enumerate(weights):
        key = gf.point_key(w, tag)
        gi = run.xhat_cache.get(key)
        if gi is not None and gi in run.xhat:
            gis[i] = gi
            if on_arrival is not None:
                on_arrival(i, gi)
            continue
        if key in todo_of:
            todo[todo_of[key]][1].append(i)
        else:
            todo_of[key] = len(todo)
            todo.append((np.asarray(w, dtype=float), [i], key))
    if not todo:
        return gis

    def _arrive(li, gi):
        for i in todo[li][1]:
            gis[i] = gi
            if on_arrival is not None:
                on_arrival(i, gi)

    got = stg.readout(app, run, pool, [t[0] for t in todo], label, ctl=ctl, on_arrival=_arrive)
    for (_w, idxs, key), gi in zip(todo, got):
        if gi is None:
            continue
        run.xhat_cache[key] = gi
        for i in idxs:
            gis[i] = gi
    return gis


def _staged_batch(app, run, pool, pos, need, pts, arrive):
    """The staged half of run_position: x̂0 readouts of every new line's samples and the mix, the
    flagged segments' probes resumed (and the mix, always), then each line's labels. A change is
    only ever claimed between two FINISHED samples; divs are exact there and x̂0 elsewhere."""
    lg = stg.ledger(run)
    before = lg.image_eq
    g = _readout_cached(app, run, pool, pts, f"desk{pos.pid}x", pos, on_arrival=arrive)
    if pos.status != "running":
        pos.cost += lg.image_eq - before
        return None
    seqs, off = [], 0
    for ln in need:
        n = len(ln["alphas"])
        seqs.append(g[off:off + n])
        off += n
    flags, _sel = stg.run_sequences(app, run, pool, seqs, f"desk{pos.pid}r", ctl=pos, always=[g[-1]])
    pos.cost += lg.image_eq - before
    # a finished sample IS the full-fidelity render of its recipe: refine's ends and later
    # requests find it in the render cache instead of rendering it again
    for gi, w in list(run.staged_w.items()):
        if gi in run.embeddings:
            run.point_cache.setdefault(gf.point_key(w, run.steps), gi)
    if pos.status != "running":
        return None
    for ln, seq, fl in zip(need, seqs, flags):
        full = [run.embeddings.get(x) if x is not None else None for x in seq]
        xh = [run.xhat.get(x) if x is not None else None for x in seq]
        ln["images"] = [-1 if x is None else int(x) for x in seq]
        ln["resumed"] = [x is not None and x in run.embeddings for x in seq]
        ln["flagged"] = list(fl)
        ln["divs"] = [float(_cosd(full[t], full[t + 1])) if full[t] is not None and full[t + 1] is not None
                      else (float(_cosd(xh[t], xh[t + 1])) if xh[t] is not None and xh[t + 1] is not None
                            else None) for t in range(len(seq) - 1)]
        ln["flips"] = detect_flips(ln["alphas"], full, seq)
    return g


def run_position(app, run, pool, pos):
    """Compute (or read from the cache) the k lines through pos.w0, then the readout.

    One cheap batch for every line not in the cache plus the mix itself; with refine, one
    full-fidelity batch of bracket ends and REFINE_ROUNDS of midpoints. Lines are cached only
    once complete, so a move cancelled half-way leaves nothing half-measured behind.
    """
    pos.status = "running"
    k = run.k
    w0 = np.asarray(pos.w0, dtype=np.float64)
    lines, need = [], []
    for i in range(k):
        key = line_key(w0, i, pos.dalpha, pos.refine)
        ln = run.line_cache.get(key)
        if ln is None:
            ln = _new_line(w0, i, pos.dalpha, pos.refine)
            need.append(ln)
        lines.append(ln)
    pos.lines = lines
    pos.cached = [all(ln is not x for x in need) for ln in lines]   # identity, not ==
    pts, owner = [], []
    for li, ln in enumerate(need):
        for t, w in enumerate(ln["ws"]):
            pts.append(w)
            owner.append((li, t))
    pts.append(w0)
    owner.append(None)
    run.phase_done, run.phase_total = 0, len(pts)

    def _arrive(j, gi):
        o = owner[j]
        if o is None:
            pos.w0_image = int(gi)
        else:
            need[o[0]]["images"][o[1]] = int(gi)

    if run.staged:
        g = _staged_batch(app, run, pool, pos, need, pts, _arrive)
        if g is None or pos.status != "running":
            return
        pos.w0_image = -1 if g[-1] is None else int(g[-1])
    else:
        g, n_new = gf.render_cached(app, run, pool, pts, run.probe_steps, f"desk{pos.pid}", pos,
                                    on_arrival=_arrive)
        run.n_probes += n_new
        pos.cost += probe_cost(run.probe_steps) * n_new
        if pos.status != "running":
            return
        pos.w0_image = -1 if g[-1] is None else int(g[-1])
        off = 0
        for ln in need:
            n = len(ln["alphas"])
            gis = g[off:off + n]
            off += n
            ln["images"] = [-1 if x is None else int(x) for x in gis]
            embs = [run.embeddings.get(x) if x is not None else None for x in gis]
            ln["divs"] = [None if embs[t] is None or embs[t + 1] is None
                          else float(_cosd(embs[t], embs[t + 1])) for t in range(n - 1)]
            ln["flips"] = detect_flips(ln["alphas"], embs, gis)
    if pos.refine:
        _refine(app, run, pool, pos, need)
        if pos.status != "running":
            return
    for ln in need:
        gis = [None if x < 0 else x for x in ln["images"]]
        if run.staged:
            # a stretch shows a FINISHED image only (never an x̂0 preview); the one stretch of a
            # line without changes has no finished sample of its own, and contains the mix
            gis = [x if x is not None and x in run.embeddings else None for x in gis]
        ln["segments"] = segments(ln["alphas"], ln["flips"], gis)
        if run.staged:
            a0 = float(pos.w0[ln["i"]])
            for sg in ln["segments"]:
                if sg["thumb"] < 0 and sg["alpha_lo"] <= a0 <= sg["alpha_hi"]:
                    sg["thumb"] = int(pos.w0_image)
        ln["status"] = "complete"
        run.line_cache[ln["key"]] = ln
    pos.readout = readout(run, pos)
    n_flip = sum(len(ln["flips"]) for ln in lines)
    if run.staged:
        rep = stg.ledger(run).report()
        pos.notes.append(f"staged readout at step {run.staged_t}/{run.steps}, θ {run.staged_theta:g}: "
                         f"{rep['resumed'] + rep['from_scratch']}/{rep['readouts']} samples finished "
                         f"so far on this desk; changes are claimed between finished samples only")
    pos.notes.append(f"{k} lines ({k - len(need)} from the cache), {n_flip} changes past "
                     f"{COS_T}, {pos.cost:.1f} image-eq"
                     + (f"; {sum(1 for ln in need if ln['degenerate'])} degenerate line(s): the "
                        f"mix is a vertex, so the others share the rest equally"
                        if any(ln["degenerate"] for ln in need) else ""))
    pos.status = "complete"
