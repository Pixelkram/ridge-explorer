"""Metropolis ridge sampler: draw images in proportion to how sharp the field is there.

The one positive result of the h25 path-tracing programme
(`search_problem/outputs/h25_pathtracing/a_mcmc/RESULTS.md`, verdict replicated in
`a_mcmc/verify/`). Everything else in the tool SEARCHES for boundaries -- chords, lattices,
walks -- and then reports where they are. This does something different: it produces a FAIR
SAMPLE of the sharpness field, so a gallery of its accepted states is a stimulus set drawn
from the law S(w)^beta rather than a hand-picked tour of the sharpest places found.

The chain
---------
State is a recipe w on the simplex. Its energy is the tool's own divergence, read at the
chord stride:

    S(w) = mean over m random unit tangent directions u of cosd( e(w), e(w + delta*u) )

with delta = cascade.STRIDE (0.025, ~1 fine cell) -- the same spacing a Cascade chord probes
at, so "sharp" means here what it means on the cascade map. Target pi(w) ~ S(w)^beta,
proposals are Gaussian random walks in tangent coordinates (symmetric, so the acceptance is
the plain ratio min(1, (S'/S)^beta)), and a proposal outside the simplex is rejected.

Calibration of record (NOISY = single-seed energies, which is what this tool reads):

  * beta = 1 WORKS: on-boundary fraction 0.673 against the 0.677 the exact S^beta law would
    give (0.99x), KS 0.083, a 2.3-2.6x enrichment over uniform sampling.
  * beta = 2 FAILS: KS 0.19-0.28. Squaring a 1-2-direction Monte-Carlo estimate of S
    amplifies the ESTIMATOR noise, so the chain locks onto cells whose estimate happened to
    be large -- the on-boundary fraction stays high while the whole shape of the S histogram
    is wrong. The models cap beta at 1.5 and the UI warns above 1.
  * ERPT-style C short chains x L steps (60 x 50) seeded at crossings beat one long chain on
    the LAW at small sigma (on-boundary ratio 0.98-0.99 vs 0.90), because the seeds already
    sit on boundaries. 4.5-8.1 basins per chain; 1-5 % of chains never leave their seed.
  * NOT competitive for DISCOVERY: 0.91-1.05x the distinct-pair count of plain IUR chords at
    equal cost. This is a sampler, not a search -- the Cascade stays the instrument for
    "where are the boundaries".

Cost (h25 accounting, shared with amr.py): a cheap probe is 0.5 image-eq, and one energy
evaluation at a proposal is (1 + m) probes -- the current state's e(w) is cached from the
round that accepted it and is never re-rendered. Each MCMC step across ALL chains is ONE
batched evaluate, so a C x L run is L batched rounds plus one for the chains' initial
energies.

Caveats carried from the study, and what is NOT claimed here:

  * The energy is re-estimated at every proposal from m directions, so the chain targets
    S^beta only up to that estimator noise (the study's KS column IS that gap).
  * Equal weight per chain omits ERPT's seed-density weighting, so crossing-seeded chains
    inherit the seeds' bias toward boundaries (visible as on-boundary ratios slightly ABOVE
    1 in the study's oracle runs).
  * The summary below is quoted over the ACCEPTED states (what the gallery shows), not over
    the time-average of the chains with their repeats, so it is a statistic of the sample
    this run produced, not an estimate of the stationary law.
  * Probe points w + delta*u are deliberately NOT constrained to the simplex (study caveat
    5): only STATES are. Mixing is affine in the worker, so a probe just past a face is an
    extrapolation that renders fine, and clipping it would shorten delta exactly where a
    boundary runs along a face.
"""
from dataclasses import dataclass, field
import threading

import numpy as np

from backend.services import cascade as cs
from backend.services.cascade import COS_T, _cosd, _iso_dir
from backend.services.local_sv import tangent_basis

# ---- calibration of record (h25a) -----------------------------------------------------
# the offset the energy is read at: the Cascade's chord stride, so one number means one thing
DELTA = cs.STRIDE
BETA_DEFAULT = 1.0
BETA_MAX = 1.5                 # past 1.5 nothing was measured; 2 is known to fail under noise
BETA_WARN = 1.0                # above this the single-seed KS degrades (0.19-0.28 at beta 2)
SIGMA_DEFAULT = 0.03           # best LAW cell (on-bnd 0.99x ideal, KS 0.083) with m = 2
M_DEFAULT = 2
CHAINS_DEFAULT = 60            # the study's ERPT design
CHAIN_STEPS_DEFAULT = 50
N_SEED_CHORDS_DEFAULT = 20
# accounting: a cheap probe is half an image (probe_steps = 4 of 8 denoising steps)
PROBE_COST = 0.5
# S = 0 happens: a single-direction probe stays inside the same cell ~1.3 % of the time at
# delta = 0.025 (study caveat 5), and the two images are then identical. Floor it rather
# than divide by zero -- the ratio (S'/S)^beta is the only place S appears in a denominator,
# and 1e-6 makes a move away from an exactly-flat state certain and a move INTO one
# vanishingly unlikely, which is what the target law says.
S_FLOOR = 1e-6
# Mean IUR chord length at k=4 with UNIFORM anchors (h25b, 4,000 picks). Used only to
# project a run's cost before it starts; nothing a run reports depends on it.
MEAN_CHORD_LEN = 0.573
# Ceiling on a projected run, in image-eq. The h25 programme's own bar; the default
# 60 x 50 design projects ~4.8e3, so this refuses the configurations that would hold the
# shared pool for days rather than hours.
MAX_IMAGE_EQ = 20000


# ---- pure helpers ---------------------------------------------------------------------

def propose(w, sigma, rng, basis):
    """One Gaussian random-walk step from w, in TANGENT coordinates.

    `basis` is k x (k-1) with orthonormal sum-zero columns (local_sv.tangent_basis), so the
    displacement is isotropic in the simplex's own chart and sums to zero exactly: the
    proposal is still an affine recipe (sum 1) whatever sigma is. Symmetric by construction,
    which is what lets `accept` be the plain ratio with no proposal correction.

    Nothing is clipped here. A proposal may land outside the simplex, and the caller rejects
    it (see `in_simplex`): rejecting is the correct treatment of a hard support boundary,
    where re-drawing until inside would truncate the proposal kernel asymmetrically and
    quietly bias the chain toward the faces.
    """
    w = np.asarray(w, dtype=np.float64)
    z = rng.standard_normal(np.shape(basis)[1])
    return w + float(sigma) * (np.asarray(basis, dtype=np.float64) @ z)


def in_simplex(w, tol=1e-12):
    """Is this a legal recipe: non-negative and summing to 1 (to round-off)?"""
    w = np.asarray(w, dtype=np.float64)
    return bool(w.size and w.min() >= -tol and abs(float(w.sum()) - 1.0) <= 1e-9)


def energy_dirs(k, rng, m):
    """m random unit directions in the sum-zero subspace: the probe fan of ONE energy.

    Fresh at every evaluation (`cs._iso_dir`, the Cascade's own isotropic draw), so S is an
    unbiased m-sample estimate of the mean divergence at the stride scale rather than a
    reading of m fixed axes.
    """
    return [_iso_dir(int(k), rng) for _ in range(int(m))]


def energy(e0, probes):
    """S = mean cosine distance from the state's image to its probe images, floored.

    None when the state's own image never arrived or no probe did -- no evidence, which the
    caller treats as a rejection rather than as a flat field.
    """
    if e0 is None:
        return None
    vals = [_cosd(e0, e) for e in probes if e is not None]
    if not vals:
        return None
    return max(S_FLOOR, float(np.mean(vals)))


def accept_prob(s, s_prop, beta):
    """min(1, (S'/S)^beta) -- Metropolis for a symmetric proposal and target pi ~ S^beta.

    Both energies are floored at S_FLOOR (see the constant), so S = 0 is handled rather than
    excluded: a state with S = 0 accepts anything, and a proposal with S = 0 is accepted with
    probability (1e-6/S)^beta.
    """
    a = max(float(s), S_FLOOR)
    b = max(float(s_prop), S_FLOOR)
    return float(min(1.0, (b / a) ** float(beta)))


def accept(s, s_prop, beta, u):
    """True when the uniform draw u in [0, 1) falls under `accept_prob`."""
    return bool(float(u) < accept_prob(s, s_prop, beta))


# ---- cost model ----------------------------------------------------------------------

def probe_cost(probe_steps):
    """Image-equivalents of one probe: 0.5 on the cheap field, 1.0 at full fidelity."""
    return 1.0 if probe_steps is None else PROBE_COST


def round_probes(n_states, m):
    """Probes one batched round costs: (1 + m) per state whose energy is being measured."""
    return int(n_states) * (1 + int(m))


def projected_cost(k, m, chains, chain_steps, n_seed_chords, seed_mode, probe_steps,
                   render_full=False):
    """(image-eq, rows) this configuration would spend, before any of it is spent.

    Three terms, each named in the rows so a refusal can be read:
      * seed chords (IUR mode only), at the mean k=4 chord length measured in h25b;
      * the chains' INITIAL energies, one round of (1 + m) probes per chain -- a crossing
        midpoint has no rendered image, so it has to be measured like any other state;
      * L rounds of the same, which is the upper branch: a proposal outside the simplex is
        rejected without rendering anything, so a real run comes in under this.
    An optional full-fidelity pass over the accepted states is bounded by C x L images.
    """
    unit = probe_cost(probe_steps)
    per_chord = int(MEAN_CHORD_LEN / cs.STRIDE) + 1
    rows = []
    if seed_mode == "iur":
        rows.append({"what": f"{int(n_seed_chords)} seed chords",
                     "probes": int(n_seed_chords) * per_chord})
    rows.append({"what": "initial energies", "probes": round_probes(chains, m)})
    rows.append({"what": f"{int(chain_steps)} rounds",
                 "probes": round_probes(chains, m) * int(chain_steps)})
    total = unit * sum(r["probes"] for r in rows)
    if render_full:
        rows.append({"what": "full-fidelity pass (worst case)",
                     "probes": int(chains) * int(chain_steps)})
        total += float(int(chains) * int(chain_steps))
    for r in rows:
        r["image_eq"] = float(unit * r["probes"]) if "full-fidelity" not in r["what"] \
            else float(r["probes"])
    return float(total), rows


@dataclass
class MetroRun:
    """One Metropolis sampling run.

    The fields down to `_lock` are the contract `cascade.evaluate` reads (prompts, image
    settings, the thumbnail store, the index allocator, the live cloud); everything above is
    this sampler's own state.
    """
    run_id: str
    prompts: list
    seed: int
    steps: int
    height: int
    width: int
    guidance_scale: float
    # --- the chain
    beta: float = BETA_DEFAULT
    sigma: float = SIGMA_DEFAULT
    m: int = M_DEFAULT
    chains: int = CHAINS_DEFAULT
    # steps per chain (L). NOT `steps`, which is the denoising-step count cascade.evaluate
    # reads off this run like it does off a CascadeRun.
    chain_steps: int = CHAIN_STEPS_DEFAULT
    delta: float = DELTA
    # where the chains start: "iur" draws its own root chords and seeds at their crossings,
    # "cascade" reuses the crossings of a finished Cascade run
    seed_mode: str = "iur"
    cascade_run_id: str | None = None
    n_seed_chords: int = N_SEED_CHORDS_DEFAULT
    probe_steps: int | None = 4
    # optional post-pass: re-render every accepted state at full denoising steps
    render_full: bool = False
    status: str = "running"
    phase: str = ""
    phase_done: int = 0
    phase_total: int = 0
    round_done: int = 0
    # the crossings the chains were seeded at: {weights, divergence}
    seed_crossings: list = field(default_factory=list)
    n_chords: int = 0
    # one per chain: {chain, weights, s, image, seed_weights, seed_s, seed_image,
    #                 n_propose, n_accept, n_outside, moved}
    chains_state: list = field(default_factory=list)
    # every ACCEPTED state: {chain, step, weights, s, image, full_image}
    samples: list = field(default_factory=list)
    n_probes: int = 0                 # cheap probes rendered (what the cost is charged on)
    n_full: int = 0                   # full-fidelity renders of accepted states
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
        return float(probe_cost(self.probe_steps) * self.n_probes + 1.0 * self.n_full)

    def summary(self):
        """The numbers the panel reads, all over the ACCEPTED states (see the module header).

        `seed_mean_s` is the mean energy of the chains' START states, i.e. of the crossings
        they were seeded at -- the comparison that says whether the chain found sharper
        places than the survey handed it, or merely held on to them.
        """
        ss = [x["s"] for x in self.samples if x["s"] is not None]
        seed_s = [c["seed_s"] for c in self.chains_state if c["seed_s"] is not None]
        prop = sum(c["n_propose"] for c in self.chains_state)
        acc = sum(c["n_accept"] for c in self.chains_state)
        return {
            "n_samples": len(self.samples),
            "mean_s": float(np.mean(ss)) if ss else None,
            "seed_mean_s": float(np.mean(seed_s)) if seed_s else None,
            "on_boundary_frac": (float(np.mean([s > COS_T for s in ss])) if ss else None),
            "acceptance": (float(acc) / prop) if prop else None,
            "n_propose": int(prop),
            "n_accept": int(acc),
            "n_outside": int(sum(c["n_outside"] for c in self.chains_state)),
            "chains_never_moved": int(sum(1 for c in self.chains_state if not c["moved"])),
            "n_chains": len(self.chains_state),
            "n_seed_crossings": len(self.seed_crossings),
            "n_chords": int(self.n_chords),
            "n_probes": int(self.n_probes),
            "n_full": int(self.n_full),
            "rounds_done": int(self.round_done),
            "cost_image_eq": self.cost_image_eq,
        }


# ---- seeding -------------------------------------------------------------------------

def seed_chords(app, run, pool, rng):
    """IUR root chords and the crossings along them: the Cascade's chord phase, reused.

    Uniform anchor + isotropic direction + probes every `cs.STRIDE` along the chord, read on
    the cheap field, crossings detected as adjacent probe pairs past `COS_T`. The anchors are
    drawn UNIFORMLY rather than through `cs._blue_anchor`: h25b measured the tool's Mitchell
    best-candidate anchors sitting at the simplex faces (41.9 % within 0.02 of a face vs
    21.3 % uniform, mean chord length 0.524 vs 0.573), and seeds are the one thing in this
    sampler that should not carry a position bias.

    One batched evaluate for every chord's probes together. Returns [{weights, divergence}].
    """
    k = run.k
    pts, meta, geo = [], [], []
    for _ in range(int(run.n_seed_chords)):
        w0 = rng.dirichlet(np.ones(k))
        u = _iso_dir(k, rng)
        tneg, tpos = cs._extent(w0, u)
        if tpos - tneg < cs._min_chord_len(cs.STRIDE):
            continue                                  # too short to carry a bracket
        offs = cs._chord_offsets(tneg, tpos, cs.STRIDE)
        ci = len(geo)
        geo.append((w0, u, offs))
        for i, off in enumerate(offs):
            pts.append(np.clip(w0 + off * u, 0, None))
            meta.append((ci, i))
    run.n_chords = len(geo)
    if not pts:
        return []
    run.phase = "seed chords"
    run.phase_done, run.phase_total = 0, len(pts)
    gidx = cs.evaluate(app, run, pool, pts, run.seed, "metroseed", steps=run.probe_steps)
    run.n_probes += sum(1 for g in gidx if g is not None)
    per = {}
    for (ci, i), gi in zip(meta, gidx):
        if gi is not None and gi in run.embeddings:
            per.setdefault(ci, []).append((i, gi))
    out = []
    for ci in sorted(per):
        seq = sorted(per[ci])
        w0, u, offs = geo[ci]
        for (i1, g1), (i2, g2) in zip(seq, seq[1:]):
            if i2 != i1 + 1:
                continue                              # a probe in between never arrived
            d = _cosd(run.embeddings[g1], run.embeddings[g2])
            cs._set_div(run, g1, d)
            cs._set_div(run, g2, d)
            if d > COS_T:
                mid = cs._clipn(w0 + 0.5 * (offs[i1] + offs[i2]) * u, k)
                out.append({"weights": [float(v) for v in mid], "divergence": float(d)})
    run.notes.append(f"seed chords: {len(geo)} chords, {len(pts)} cheap probes, "
                     f"{len(out)} crossings past {COS_T}")
    return out


def cascade_seeds(run, src):
    """The crossings of a finished Cascade run, as seed states: their MIDPOINTS.

    A detection-only run quotes its midpoint at bracket precision (+- stride/2) and a
    certified one at bisection precision; either is well inside one cell of the boundary,
    which is all a chain start needs. `divergence` is the cosine distance across the bracket
    where the run recorded both sides -- reported, never used as an energy: it is measured
    ACROSS the boundary, where S is measured around a point.
    """
    out = []
    k = run.k
    for x in getattr(src, "crossings", None) or ():
        w = getattr(x, "mid", None)
        if w is None:
            wa, wb = getattr(x, "wa", None), getattr(x, "wb", None)
            if wa is None or wb is None:
                continue
            w = (np.asarray(wa, dtype=float) + np.asarray(wb, dtype=float)) / 2.0
        w = cs._clipn(np.asarray(w, dtype=float), k)
        ea, eb = getattr(x, "ea", None), getattr(x, "eb", None)
        out.append({"weights": [float(v) for v in w],
                    "divergence": float(_cosd(ea, eb)) if ea is not None and eb is not None
                    else None})
    return out


def _start_states(crossings, chains, rng, k):
    """Which crossing each chain starts on: a random permutation, cycled if there are fewer
    crossings than chains (several chains then share a sheet, which the notes say).

    With no crossings at all the chains start at uniform Dirichlet points: the chain still
    targets S^beta from anywhere, it just loses the ERPT advantage the study measured (the
    crossing seeds are what buy the better LAW at small sigma).
    """
    if not crossings:
        return [np.asarray(w, dtype=float)
                for w in rng.dirichlet(np.ones(k), size=int(chains))], True
    order = rng.permutation(len(crossings))
    return [np.asarray(crossings[int(order[i % len(order)])]["weights"], dtype=float)
            for i in range(int(chains))], False


# ---- the batched energy round ---------------------------------------------------------

def energy_batch(app, run, pool, rng, states, label):
    """Measure S at every state in `states` with ONE batched evaluate.

    (1 + m) probes per state: the state's own image, then m probes one `delta` away along
    fresh random unit tangent directions. Returns [(image index, S)] aligned with `states`,
    S None where the state's image or all of its probes never arrived.
    """
    k, m = run.k, int(run.m)
    pts = []
    for w in states:
        w = np.asarray(w, dtype=np.float64)
        pts.append(w)
        for u in energy_dirs(k, rng, m):
            # deliberately unclipped: only STATES live in the simplex (study caveat 5)
            pts.append(w + float(run.delta) * u)
    run.phase_done, run.phase_total = 0, len(pts)
    gidx = cs.evaluate(app, run, pool, pts, run.seed, label, steps=run.probe_steps)
    run.n_probes += sum(1 for g in gidx if g is not None)
    out = []
    for i in range(len(states)):
        base = i * (1 + m)
        g0 = gidx[base]
        e0 = run.embeddings.get(g0) if g0 is not None else None
        es = [run.embeddings.get(g) if g is not None else None
              for g in gidx[base + 1:base + 1 + m]]
        s = energy(e0, es)
        if s is not None:
            cs._set_div(run, g0, s)      # the live cloud colours by the same number
        out.append((g0, s))
    return out


def run_metro(app, run, pool):
    """The whole run. Called from the router's guarded thread body.

    Phases: seed crossings (a chord survey, or a finished Cascade's) -> the chains' initial
    energies -> L batched rounds, one per MCMC step across all chains -> an optional
    full-fidelity pass over the accepted states. Cancellation is the Cascade's: `run.status`
    stops the run between rounds and inside `evaluate`.
    """
    k = run.k
    rng = np.random.default_rng(run.seed)
    basis = tangent_basis(k)

    # ---- seeds
    if run.seed_mode == "cascade":
        src = (getattr(app.state, "cascades", {}) or {}).get(run.cascade_run_id) \
            if app is not None else None
        run.seed_crossings = cascade_seeds(run, src) if src is not None else []
        run.notes.append(f"seeded from cascade run {run.cascade_run_id}: "
                         f"{len(run.seed_crossings)} crossings")
    else:
        run.seed_crossings = seed_chords(app, run, pool, rng)
    if run.status != "running":
        return

    starts, uniform = _start_states(run.seed_crossings, run.chains, rng, k)
    if uniform:
        run.notes.append("no seed crossings: the chains start at uniform points, so the "
                         "ERPT advantage h25a measured (crossing seeds buy a better law at "
                         "small sigma) does not apply to this run")
    elif len(run.seed_crossings) < run.chains:
        run.notes.append(f"{run.chains} chains over {len(run.seed_crossings)} crossings: "
                         f"some chains share a starting sheet")

    # ---- the chains' initial energies: one batched round
    run.phase = "initial energies"
    init = energy_batch(app, run, pool, rng, starts, "metroinit")
    if run.status != "running":
        return
    for i, (gi, s) in enumerate(init):
        run.chains_state.append({
            "chain": i, "weights": [float(v) for v in starts[i]], "s": s,
            "image": int(gi) if gi is not None else -1,
            "seed_weights": [float(v) for v in starts[i]], "seed_s": s,
            "seed_image": int(gi) if gi is not None else -1,
            "n_propose": 0, "n_accept": 0, "n_outside": 0, "moved": False})
    dead = [c["chain"] for c in run.chains_state if c["s"] is None]
    if dead:
        run.notes.append(f"{len(dead)} chains never got a starting energy (missing probes) "
                         f"and sit out the run")

    # ---- L rounds, each ONE batched evaluate across every chain that proposed
    for r in range(1, int(run.chain_steps) + 1):
        if run.status != "running":
            break
        run.phase = f"round {r}/{int(run.chain_steps)}"
        live, props = [], []
        for c in run.chains_state:
            if c["s"] is None:
                continue
            c["n_propose"] += 1
            wp = propose(c["weights"], run.sigma, rng, basis)
            if not in_simplex(wp):
                c["n_outside"] += 1          # rejected for free: nothing is rendered
                continue
            live.append(c)
            props.append(wp)
        if not props:
            run.round_done = r
            continue
        got = energy_batch(app, run, pool, rng, props, f"metro{r}")
        if run.status != "running":
            break
        for c, wp, (gi, s) in zip(live, props, got):
            if s is None:
                continue                     # no evidence -> the state stays where it is
            if accept(c["s"], s, run.beta, rng.random()):
                c["weights"] = [float(v) for v in wp]
                c["s"] = s
                c["image"] = int(gi) if gi is not None else -1
                c["n_accept"] += 1
                c["moved"] = True
                run.samples.append({
                    "chain": c["chain"], "step": r, "weights": c["weights"], "s": float(s),
                    "image": c["image"], "full_image": None})
        run.round_done = r

    # ---- optional: re-render the accepted states at full denoising steps
    if run.render_full and run.samples and run.status == "running":
        run.phase = "full-fidelity pass"
        uniq, at = [], {}
        for x in run.samples:
            key = tuple(round(float(v), 9) for v in x["weights"])
            if key not in at:
                at[key] = len(uniq)
                uniq.append(np.asarray(x["weights"], dtype=float))
        run.phase_done, run.phase_total = 0, len(uniq)
        gidx = cs.evaluate(app, run, pool, uniq, run.seed, "metrofull", steps=None)
        run.n_full += sum(1 for g in gidx if g is not None)
        for x in run.samples:
            gi = gidx[at[tuple(round(float(v), 9) for v in x["weights"])]]
            x["full_image"] = int(gi) if gi is not None else None
        run.notes.append(f"full-fidelity pass: {len(uniq)} distinct accepted states of "
                         f"{len(run.samples)} samples, {run.n_full} rendered at "
                         f"{run.steps} steps")

    run.phase = "done"
    s = run.summary()
    ratio = ("n/a" if not s["mean_s"] or not s["seed_mean_s"]
             else f"{s['mean_s'] / s['seed_mean_s']:.2f}x")
    run.notes.append(
        f"done: {s['n_samples']} accepted states from {s['n_chains']} chains x "
        f"{run.round_done} rounds, acceptance "
        + (f"{s['acceptance']:.2f}" if s["acceptance"] is not None else "n/a")
        + f", mean S " + (f"{s['mean_s']:.3f}" if s["mean_s"] is not None else "n/a")
        + f" ({ratio} the seeds'), on-boundary "
        + (f"{s['on_boundary_frac']:.2f}" if s["on_boundary_frac"] is not None else "n/a")
        + f", {s['chains_never_moved']}/{s['n_chains']} chains never moved, "
        f"{s['cost_image_eq']:.1f} image-eq "
        f"(h25a beta=1: on-boundary 0.67, 2.3-2.6x uniform)")
    if run.status == "running":
        run.status = "complete"
