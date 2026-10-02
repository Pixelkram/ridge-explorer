"""Offline checks of the Metropolis ridge sampler (backend/services/metro.py). No GPU.

Three parts, all analytic or stubbed -- nothing renders:

  (i)   PURE -- `propose` steps in the simplex's TANGENT space (sum-zero displacement, so a
        proposal is still an affine recipe) and a proposal that leaves the simplex is
        REJECTED rather than redrawn; `accept_prob`/`accept` are the Metropolis ratio
        min(1, (S'/S)^beta) over a table of (S, S', beta) cases including S = 0, which is
        floored at metro.S_FLOOR = 1e-6 (at delta = 0.025 a single-direction probe stays
        inside the same cell ~1.3 % of the time and the two images are then identical);
        `energy` averages exactly m directions and is the floored mean cosine distance; the
        cost model charges (1 + m) probes per energy evaluation at 0.5 image-eq each.
  (ii)  LOOP -- run_metro over a synthetic TWO-BASIN field (two sides of a plane joined
        through a tanh sheet of width `band`, plus a low-frequency background so S is small
        but never zero off the sheet -- the real field's off-boundary S is ~1/6 of its
        on-boundary S, not 0), with cascade.evaluate stubbed as in tests/amr_test.py. The
        samples' mean S must beat the mean S of UNIFORM points, acceptance must land inside
        (0.1, 0.9), every round must be ONE batch of (1 + m) probes per PROPOSING chain --
        checked as an exact total, so an out-of-simplex rejection cannot hide a lost batch --
        and the cost must be 0.5 image-eq per probe.
  (iii) ROUTER through FastAPI's TestClient with a stubbed app.state and metro.run_metro
        replaced by a no-op (as tests/amr_test.py does for the AMR router): the defaults of
        record (beta 1, sigma 0.03, m 2, 60 x 50, iur, 20 chords), beta 2.0 and k = 2 refused
        with a 422, the cascade seed mode refused with a 404 for an unknown run id (and 400 /
        409 for the other ways it cannot be honoured), a configuration past the image-eq
        ceiling refused with the projection, plus /status, /cancel, /samples.json,
        /cascade-runs and the image route.

Run: python tests/metro_test.py
"""
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.services import metro                      # noqa: E402
from backend.services import cascade as cs              # noqa: E402
from backend.services.cascade import COS_T, _cosd       # noqa: E402
from backend.services.local_sv import tangent_basis     # noqa: E402

K = 4
fails = []


def check(ok, label, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(label)


# ------------------------------------------------------------------------- (i) pure helpers
def part1():
    print("\n(i) propose / accept / energy / cost, as arithmetic")
    rng = np.random.default_rng(0)
    B = tangent_basis(K)
    check(float(np.abs(B.sum(axis=0)).max()) < 1e-12
          and np.allclose(B.T @ B, np.eye(K - 1), atol=1e-12),
          "the tangent basis is orthonormal with sum-zero columns",
          f"worst |column sum| {float(np.abs(B.sum(axis=0)).max()):.1e}")

    w = np.full(K, 1.0 / K)
    props = [metro.propose(w, 0.03, rng, B) for _ in range(2000)]
    check(max(abs(float(p.sum()) - 1.0) for p in props) < 1e-12,
          "every proposal is still an affine recipe (sums to 1 exactly)",
          f"worst |sum - 1| {max(abs(float(p.sum()) - 1.0) for p in props):.1e}")
    disp = np.stack([p - w for p in props])
    norms = np.linalg.norm(disp, axis=1)
    # |displacement| is sigma * chi_{k-1}: mean sigma*sqrt(2)*Gamma(k/2)/Gamma((k-1)/2)
    want = 0.03 * math.sqrt(2.0) * math.gamma(K / 2.0) / math.gamma((K - 1) / 2.0)
    check(abs(float(norms.mean()) - want) < 0.1 * want,
          "the step length is sigma x chi_(k-1): isotropic in the tangent chart",
          f"mean |dw| {float(norms.mean()):.4f} vs {want:.4f}")
    check(all(metro.in_simplex(p) for p in props),
          "a 0.03 step from the barycentre never leaves the simplex")

    # near a vertex a big step DOES leave it, and the caller rejects rather than redraws
    corner = np.array([0.94, 0.02, 0.02, 0.02])
    far = [metro.propose(corner, 0.12, rng, B) for _ in range(2000)]
    outside = [p for p in far if not metro.in_simplex(p)]
    check(len(outside) > 100 and all(float(p.min()) < 0 for p in outside)
          and all(abs(float(p.sum()) - 1.0) < 1e-12 for p in outside),
          "a proposal CAN leave the simplex, and leaves it with a negative weight (not an "
          "unnormalised one) -- `propose` never clips, `in_simplex` rejects",
          f"{len(outside)}/2000 outside near a vertex")
    check(all(float(p.min()) >= 0 for p in far if metro.in_simplex(p)),
          "in_simplex accepts exactly the non-negative proposals")
    check(not metro.in_simplex([0.5, 0.5, 0.5, -0.5]) and not metro.in_simplex([0.5, 0.6, 0, 0])
          and metro.in_simplex([0.25] * 4),
          "in_simplex rejects negative weights and recipes that do not sum to 1")

    # --- the acceptance rule
    cases = [
        # (S, S', beta, expected probability)
        (0.1, 0.2, 1.0, 1.0),            # uphill is always taken
        (0.2, 0.1, 1.0, 0.5),
        (0.2, 0.1, 2.0, 0.25),           # beta sharpens the target, and the downhill move
        (0.2, 0.1, 0.5, math.sqrt(0.5)),
        (0.2, 0.2, 1.0, 1.0),
        (0.0, 0.1, 1.0, 1.0),            # S = 0 -> floored, so anything is uphill
        (0.1, 0.0, 1.0, 1e-5),           # ... and a move INTO a flat state is 1e-6/0.1
        (0.0, 0.0, 1.0, 1.0),
        (metro.S_FLOOR / 10, metro.S_FLOOR / 10, 1.0, 1.0),
    ]
    worst = 0.0
    for s, sp, b, want_p in cases:
        got = metro.accept_prob(s, sp, b)
        worst = max(worst, abs(got - want_p))
    check(worst < 1e-12, "accept_prob is min(1, (S'/S)^beta), with both ends floored at 1e-6",
          f"{len(cases)} cases, worst error {worst:.1e}")
    check(metro.accept(0.2, 0.1, 1.0, 0.4) and not metro.accept(0.2, 0.1, 1.0, 0.6)
          and metro.accept(0.1, 0.2, 1.0, 0.999999)
          and not metro.accept(0.1, 0.0, 1.0, 1e-4),
          "accept(S, S', beta, u) fires exactly when u < that probability")
    # a flat chain is a free random walk, which is what "no evidence" should look like
    check(all(metro.accept_prob(0.0, 0.0, b) == 1.0 for b in (0.5, 1.0, 1.5)),
          "a chain on an exactly flat field accepts everything at every beta")

    # --- the energy
    a = np.zeros(8)
    a[0] = 1.0
    def at(d):
        """a unit vector at cosine distance exactly d from a"""
        c = 1.0 - d
        e = np.zeros(8)
        e[0], e[1] = c, math.sqrt(max(0.0, 1.0 - c * c))
        return e
    probes = [at(0.2), at(0.4), at(0.6)]
    check(abs(metro.energy(a, probes) - 0.4) < 1e-12,
          "energy is the MEAN cosine distance over the m probes",
          f"{metro.energy(a, probes):.6f} for 0.2/0.4/0.6")
    check(abs(metro.energy(a, [at(0.2), None, at(0.6)]) - 0.4) < 1e-12,
          "a probe that never arrived is left out of the mean, not counted as 0")
    check(metro.energy(a, []) is None and metro.energy(a, [None]) is None
          and metro.energy(None, probes) is None,
          "no evidence (no state image, or no probe at all) is None, not 0")
    check(metro.energy(a, [a, a]) == metro.S_FLOOR,
          "two identical images give the 1e-6 floor, not an exact 0",
          f"{metro.energy(a, [a, a]):.1e}")
    for m in (1, 2, 4):
        ds = metro.energy_dirs(K, np.random.default_rng(3), m)
        check(len(ds) == m
              and all(abs(float(np.linalg.norm(u)) - 1.0) < 1e-12 for u in ds)
              and all(abs(float(u.sum())) < 1e-12 for u in ds),
              f"energy_dirs gives m={m} unit sum-zero directions")

    # --- the cost model
    check(metro.probe_cost(4) == 0.5 and metro.probe_cost(None) == 1.0,
          "a cheap probe is 0.5 image-eq, a full-fidelity one 1.0")
    check(metro.round_probes(60, 2) == 180 and metro.round_probes(1, 1) == 2,
          "one energy evaluation is (1 + m) probes: the state's own image plus its fan")
    tot, rows = metro.projected_cost(4, 2, 60, 50, 20, "iur", 4)
    names = [r["what"] for r in rows]
    check(abs(tot - 0.5 * (20 * (int(metro.MEAN_CHORD_LEN / cs.STRIDE) + 1)
                           + 180 + 180 * 50)) < 1e-9
          and any("seed chords" in n for n in names) and any("rounds" in n for n in names),
          "the projection is seed chords + initial energies + L rounds, at 0.5 each",
          f"{tot:.0f} image-eq for the 60 x 50 design of record")
    tot_c, rows_c = metro.projected_cost(4, 2, 60, 50, 20, "cascade", 4)
    check(tot_c < tot and not any("seed chords" in r["what"] for r in rows_c),
          "the cascade seed mode pays no chord survey (it reuses one)",
          f"{tot_c:.0f} vs {tot:.0f} image-eq")
    check(metro.projected_cost(4, 2, 60, 50, 20, "iur", None)[0] == 2 * tot,
          "full-fidelity probes double the projection")
    check(metro.projected_cost(4, 2, 60, 50, 20, "iur", 4, render_full=True)[0]
          == tot + 60 * 50,
          "the optional full pass adds at most one image per accepted state")


# ------------------------------------------------- (ii) the loop on a synthetic two-basin field
def _two_basin(seed=1, band=0.02, amp=2.5, bg=0.25, freq=2.5, modes=3):
    """Two basins joined through a thin sheet, with a quiet background everywhere.

    e(w) = normalise( a + amp*tanh(g/band)*b + bg * (low-frequency wiggle) ), where
    g = <w - 1/k, p> is the signed distance to a sum-zero plane. The tanh term makes a
    codim-1 SHEET of width ~band across which the image flips basin, so a chord probe pair
    straddling it clears COS_T; the wiggle makes S small but nonzero off the sheet, which is
    the real field's behaviour (h25a: off-boundary S_bar 0.019 vs 0.112 on it, a factor 6 --
    not a factor of infinity, and a field that is exactly flat off the sheet would make the
    chain a free random walk with acceptance 1 there).

    Returns (field, plane normal).
    """
    rng = np.random.default_rng(seed)
    p = rng.standard_normal(K)
    p -= p.mean()
    p /= np.linalg.norm(p)
    Q = np.linalg.qr(rng.standard_normal((8, 2 + modes)))[0]      # orthonormal columns
    a, b, Cvec = Q[:, 0], Q[:, 1], Q[:, 2:]
    Cdir = rng.standard_normal((modes, K))
    Cdir -= Cdir.mean(axis=1, keepdims=True)
    Cdir /= np.linalg.norm(Cdir, axis=1, keepdims=True)
    ph = rng.uniform(0.0, 2.0 * math.pi, modes)

    def f(w):
        x = np.asarray(w, dtype=float) - 1.0 / K
        e = (a + amp * math.tanh(float(x @ p) / band) * b
             + bg * (Cvec @ np.sin(freq * (Cdir @ x) + ph)))
        n = float(np.linalg.norm(e))
        return e / n if n > 1e-12 else a
    return f, p


def _stub_evaluate(run, field, rounds):
    """Stand-in for cascade.evaluate (the shape tests/amr_test.py uses): embeds every weight
    vector through the analytic field, keeps the live-cloud bookkeeping the real one keeps,
    and records (label, batch size) per GPU round -- which is how the batching is counted."""
    def stub(app, r, pool, weights, seed, label, ctl=None, on_arrival=None, steps=None):
        rounds.append((label.split(".")[0], len(weights), steps))
        idxs = []
        for w in weights:
            gi = run.next_idx
            run.next_idx += 1
            run.embeddings[gi] = field(w)
            run.generated += 1
            run.phase_done += 1
            run._geo_pos[gi] = len(run.probe_geo)
            run.probe_geo.append([float(v) for v in w])
            run.probe_div.append(None)
            idxs.append(gi)
        return idxs
    return stub


def _uniform_mean_s(field, k, n, m, delta, seed=99):
    """Mean energy of UNIFORM recipes, measured with the sampler's own estimator -- the
    baseline the chain has to beat (uniform sampling is what "no sampler" looks like)."""
    rng = np.random.default_rng(seed)
    vals = []
    for w in rng.dirichlet(np.ones(k), size=n):
        e0 = field(w)
        vals.append(metro.energy(e0, [field(w + delta * u)
                                      for u in metro.energy_dirs(k, rng, m)]))
    return float(np.mean(vals)), vals


def _run(field, chains=24, chain_steps=20, beta=1.0, sigma=0.03, m=2, n_seed_chords=14,
         seed=5, render_full=False, stop_after=None):
    """One stubbed Metropolis run; returns (run, rounds)."""
    run = metro.MetroRun(
        run_id="m", prompts=[f"p{i}" for i in range(K)], seed=seed, steps=8, height=64,
        width=64, guidance_scale=3.5, beta=beta, sigma=sigma, m=m, chains=chains,
        chain_steps=chain_steps, seed_mode="iur", n_seed_chords=n_seed_chords,
        render_full=render_full)
    rounds = []
    orig = cs.evaluate
    stub = _stub_evaluate(run, field, rounds)
    if stop_after is not None:
        def cancelling(*args, **kw):
            out = stub(*args, **kw)
            if len(rounds) >= stop_after:
                run.status = "cancelled"          # the UI's Cancel, mid-run
            return out
        cs.evaluate = cancelling
    else:
        cs.evaluate = stub
    try:
        metro.run_metro(None, run, None)
    finally:
        cs.evaluate = orig
    return run, rounds


def part2():
    print("\n(ii) run_metro on a synthetic two-basin field (stubbed, no GPU)")
    field, p = _two_basin()
    m, chains, L = 2, 24, 20
    base, base_vals = _uniform_mean_s(field, K, 400, m, metro.DELTA)
    run, rounds = _run(field, chains=chains, chain_steps=L, m=m)
    s = run.summary()

    check(run.status == "complete" and run.phase == "done" and s["rounds_done"] == L,
          "the run finishes every round", f"{run.status}/{run.phase}, {s['rounds_done']}/{L}")
    check(len(run.seed_crossings) > 0 and run.n_chords > 0,
          "the sampler's own IUR chord phase found crossings to seed at",
          f"{run.n_chords} chords, {len(run.seed_crossings)} crossings")
    check(rounds[0][0] == "metroseed" and rounds[1][0] == "metroinit"
          and [r[0] for r in rounds[2:2 + L]] == [f"metro{i}" for i in range(1, L + 1)]
          and len(rounds) == 2 + L,
          "one batched round for the chords, one for the initial energies, then exactly one "
          "per MCMC step", f"{len(rounds)} rounds")
    check(all(r[2] == 4 for r in rounds),
          "every probe is rendered on the cheap field (4 of 8 steps)")

    # the batching identity: (1 + m) probes per PROPOSING chain, per round
    init = rounds[1][1]
    per_round = [r[1] for r in rounds[2:]]
    check(init == (1 + m) * chains,
          "the initial energies are one batch of (1 + m) x C probes",
          f"{init} = ({1 + m}) x {chains}")
    check(all(n % (1 + m) == 0 and n <= (1 + m) * chains for n in per_round)
          and sum(per_round) == (1 + m) * (s["n_propose"] - s["n_outside"]),
          "each round is ONE batch of (1 + m) probes per chain that proposed inside the "
          "simplex (an out-of-simplex proposal is rejected for free)",
          f"{sum(per_round)} probes over {len(per_round)} rounds, "
          f"{s['n_outside']} outside of {s['n_propose']} proposals")
    check(s["n_probes"] == sum(r[1] for r in rounds)
          and abs(run.cost_image_eq - 0.5 * s["n_probes"]) < 1e-9,
          "cost is 0.5 image-eq per probe rendered, and every probe is counted",
          f"{s['n_probes']} probes = {run.cost_image_eq:.1f} image-eq")

    # the point of the sampler: the samples are drawn where the field is sharp
    check(s["n_samples"] > 0 and s["mean_s"] > base,
          "the samples' mean S beats the mean S of uniform recipes",
          f"{s['mean_s']:.4f} vs uniform {base:.4f} "
          f"({s['mean_s'] / base:.2f}x, {s['n_samples']} samples)")
    check(s["on_boundary_frac"] > float(np.mean([v > COS_T for v in base_vals])),
          "and more of them are past the crossing threshold than uniform points are",
          f"{s['on_boundary_frac']:.3f} vs uniform "
          f"{float(np.mean([v > COS_T for v in base_vals])):.3f} (COS_T {COS_T})")
    check(0.1 < s["acceptance"] < 0.9,
          "acceptance sits in the usable band (h25a measured 0.59 at sigma 0.03, m 2)",
          f"{s['acceptance']:.3f}")
    check(s["seed_mean_s"] is not None and s["chains_never_moved"] <= chains,
          "the seeds' own mean S is reported beside the samples'",
          f"seeds {s['seed_mean_s']:.4f}, {s['chains_never_moved']}/{chains} chains never "
          f"moved")
    check(all(x["s"] is not None and x["image"] >= 0 and len(x["weights"]) == K
              and metro.in_simplex(x["weights"]) for x in run.samples),
          "every sample carries a legal recipe, its energy and the image rendered at it")
    check(all(0 < x["step"] <= L and 0 <= x["chain"] < chains for x in run.samples)
          and len({x["chain"] for x in run.samples}) > 1,
          "samples are labelled by chain and round, and more than one chain contributed",
          f"{len({x['chain'] for x in run.samples})} chains contributed")
    moved = [c for c in run.chains_state if c["moved"]]
    check(all(c["n_accept"] > 0 for c in moved)
          and all(c["n_accept"] == 0 for c in run.chains_state if not c["moved"])
          and sum(c["n_accept"] for c in run.chains_state) == s["n_samples"],
          "a chain 'moved' exactly when it accepted something, and every acceptance is a "
          "sample", f"{len(moved)}/{chains} chains moved")

    # beta: a sharper target cannot accept MORE downhill moves than a flatter one
    hot, _r = _run(field, chains=chains, chain_steps=L, m=m, beta=1.5)
    cold, _r2 = _run(field, chains=chains, chain_steps=L, m=m, beta=0.5)
    check(hot.summary()["acceptance"] < cold.summary()["acceptance"],
          "a larger beta accepts less (the same proposals, a sharper target)",
          f"beta 1.5 -> {hot.summary()['acceptance']:.3f}, "
          f"beta 0.5 -> {cold.summary()['acceptance']:.3f}")
    check(hot.summary()["mean_s"] > cold.summary()["mean_s"],
          "... and concentrates the samples harder",
          f"{hot.summary()['mean_s']:.4f} vs {cold.summary()['mean_s']:.4f}")

    # the optional full-fidelity pass, and cancellation
    full, frounds = _run(field, chains=8, chain_steps=6, m=1, render_full=True)
    uniq = {tuple(round(v, 9) for v in x["weights"]) for x in full.samples}
    check(frounds[-1][0] == "metrofull" and frounds[-1][2] is None
          and frounds[-1][1] == len(uniq) and full.n_full == len(uniq)
          and all(x["full_image"] is not None for x in full.samples),
          "the full pass is one final batch of the DISTINCT accepted states, at full steps",
          f"{len(uniq)} distinct of {len(full.samples)} samples")
    check(abs(full.cost_image_eq
              - (0.5 * full.n_probes + 1.0 * full.n_full)) < 1e-9
          and full.n_full > 0,
          "a full-fidelity render costs a whole image, not half of one",
          f"{full.n_probes} probes + {full.n_full} images = "
          f"{full.cost_image_eq:.1f} image-eq")
    cut, crounds = _run(field, chains=8, chain_steps=20, m=1, stop_after=3)
    check(cut.status == "cancelled" and len(crounds) == 3
          and cut.summary()["rounds_done"] < 20,
          "a cancel mid-run stops the chain and stays cancelled",
          f"{cut.status} after {len(crounds)} rounds, "
          f"{cut.summary()['rounds_done']} rounds done")


# ------------------------------------------------------------------------- (iii) the router
def _fake_cascade(k=4, n_cross=3, status="complete", seed=0):
    """A finished-looking CascadeRun with crossings, for the cascade seed mode. No GPU: the
    crossings are built directly, as a finished run's would be after bisection."""
    run = cs.CascadeRun(run_id="c", prompts=[f"q{i}" for i in range(k)], seed=7, steps=8,
                        height=64, width=64, guidance_scale=3.5, n_chords=4, n_patches=1)
    run.status = status
    rng = np.random.default_rng(seed)
    ea, eb = np.zeros(8), np.zeros(8)
    ea[0] = 1.0
    eb[0], eb[1] = 0.2, math.sqrt(1 - 0.2 ** 2)
    for cid in range(n_cross):
        w = rng.dirichlet(np.ones(k))
        run.crossings.append(cs.Crossing(cid=cid, wa=w, wb=w, ea=ea, eb=eb, mid=w))
    return run


def part3():
    print("\n(iii) router through TestClient (stubbed state, run_metro no-op)")
    try:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
    except Exception as exc:                         # noqa: BLE001
        check(False, "TestClient available", str(exc))
        return
    from backend.routers import metro as router_mod

    class _NoCache:
        """A truthy stand-in ThumbnailStore keeps as-is, so nothing touches the disk (the
        shape tests/amr_test.py uses)."""
        def has(self, h):
            return False

        def save(self, h, data):
            pass

        def get_path(self, h):
            return None

    orig = metro.run_metro
    metro.run_metro = lambda app, run, pool: None     # the worker thread renders nothing
    try:
        app = FastAPI()
        app.include_router(router_mod.router)
        app.state.metro_runs = {}
        app.state.cascades = {"good4": _fake_cascade(4, 3), "k3": _fake_cascade(3, 2),
                              "empty": _fake_cascade(4, 0),
                              "busy": _fake_cascade(4, 5, status="running")}
        app.state.cache = _NoCache()
        app.state.gpu_pool = None
        c = TestClient(app, raise_server_exceptions=False)

        r = c.post("/api/metro/start", json={"prompts": ["a", "b", "c"]})
        rid3 = r.json().get("run_id", "")
        run3 = app.state.metro_runs.get(rid3)
        check(r.status_code == 200 and run3 is not None and run3.k == 3
              and (run3.beta, run3.sigma, run3.m) == (1.0, 0.03, 2)
              and (run3.chains, run3.chain_steps) == (60, 50)
              and run3.seed_mode == "iur" and run3.n_seed_chords == 20
              and run3.probe_steps == 4 and run3.render_full is False
              and run3.delta == cs.STRIDE,
              "k=3 starts with the h25a design of record as the defaults",
              f"HTTP {r.status_code}, beta {getattr(run3, 'beta', None)}, "
              f"{getattr(run3, 'chains', None)} x {getattr(run3, 'chain_steps', None)}")
        check(not run3.notes, "a beta of record leaves no warning on the run")

        r = c.post("/api/metro/start",
                   json={"prompts": ["a", "b", "c", "d"], "beta": 1.4, "sigma": 0.12,
                         "m": 1, "chains": 10, "chain_steps": 5, "seed": 7,
                         "probe_steps": None, "render_full": True})
        rid4 = r.json().get("run_id", "")
        run4 = app.state.metro_runs.get(rid4)
        check(r.status_code == 200 and run4 is not None and run4.k == 4
              and (run4.beta, run4.sigma, run4.m) == (1.4, 0.12, 1)
              and (run4.chains, run4.chain_steps) == (10, 5) and run4.seed == 7
              and run4.probe_steps is None and run4.render_full is True,
              "k=4 starts and every setting is threaded into the run",
              f"HTTP {r.status_code}")
        check(any("above the" in n and "KS" in n for n in run4.notes),
              "a beta past the one of record is recorded on the run, not silently accepted",
              (run4.notes or ["MISSING"])[0][:90])

        for bad, why in (({"beta": 2.0}, "beta 2.0 (max 1.5)"), ({"beta": 0.4}, "beta 0.4"),
                         ({"sigma": 0.3}, "sigma 0.3"), ({"sigma": 0.0}, "sigma 0"),
                         ({"m": 5}, "m 5"), ({"chains": 0}, "chains 0"),
                         ({"chain_steps": 0}, "chain_steps 0"),
                         ({"n_seed_chords": 0}, "n_seed_chords 0"),
                         ({"seed_mode": "mcmc"}, "an unknown seed mode")):
            r = c.post("/api/metro/start", json=dict({"prompts": ["a", "b", "c"]}, **bad))
            check(r.status_code == 422, f"{why} is rejected by the bounds",
                  f"got HTTP {r.status_code}")
        r = c.post("/api/metro/start", json={"prompts": ["a", "b"]})
        check(r.status_code == 422, "k=2 is rejected (fewer than 3 prompts)",
              f"got HTTP {r.status_code}")

        # --- the cascade seed mode, and every way it cannot be honoured
        r = c.post("/api/metro/start",
                   json={"prompts": ["a", "b", "c", "d"], "seed_mode": "cascade",
                         "cascade_run_id": "good4", "chains": 4, "chain_steps": 3})
        good = app.state.metro_runs.get(r.json().get("run_id", ""))
        check(r.status_code == 200 and good is not None
              and good.seed_mode == "cascade" and good.cascade_run_id == "good4",
              "the cascade seed mode starts against a finished run of the same k",
              f"HTTP {r.status_code}")
        r = c.post("/api/metro/start",
                   json={"prompts": ["a", "b", "c", "d"], "seed_mode": "cascade",
                         "cascade_run_id": "nope"})
        check(r.status_code == 404 and "nope" in r.text,
              "an unknown cascade run id -> 404", f"HTTP {r.status_code}: {r.text[:90]}")
        r = c.post("/api/metro/start",
                   json={"prompts": ["a", "b", "c", "d"], "seed_mode": "cascade"})
        check(r.status_code == 400 and "cascade_run_id" in r.text,
              "the cascade seed mode without a run id -> 400 naming the field",
              f"HTTP {r.status_code}")
        r = c.post("/api/metro/start",
                   json={"prompts": ["a", "b", "c", "d"], "seed_mode": "cascade",
                         "cascade_run_id": "k3"})
        check(r.status_code == 400 and "k=3" in r.text,
              "a cascade run of a different k -> 400 (its crossings are over its prompts)",
              f"HTTP {r.status_code}: {r.text[:90]}")
        r = c.post("/api/metro/start",
                   json={"prompts": ["a", "b", "c", "d"], "seed_mode": "cascade",
                         "cascade_run_id": "empty"})
        check(r.status_code == 400 and "no crossings" in r.text,
              "a cascade run with no crossings -> 400 rather than silent uniform seeding",
              f"HTTP {r.status_code}")
        r = c.post("/api/metro/start",
                   json={"prompts": ["a", "b", "c", "d"], "seed_mode": "cascade",
                         "cascade_run_id": "busy"})
        check(r.status_code == 409 and "still running" in r.text,
              "a cascade run that is still running -> 409", f"HTTP {r.status_code}")

        # --- the budget ceiling
        r = c.post("/api/metro/start",
                   json={"prompts": ["a", "b", "c"], "chains": 200, "chain_steps": 200,
                         "m": 4})
        proj = metro.projected_cost(3, 4, 200, 200, 20, "iur", 4)[0]
        check(r.status_code == 400 and "ceiling" in r.text and f"{proj:.0f}" in r.text,
              "a configuration past the image-eq ceiling is refused with its projection",
              f"HTTP {r.status_code}: {r.text[:110]}")

        # --- the seed-source listing the panel offers
        j = c.get("/api/metro/cascade-runs").json()
        ids = [row["run_id"] for row in j]
        check("busy" not in ids and {"good4", "k3", "empty"} <= set(ids)
              and ids == sorted(ids, key=lambda i: (-{"good4": 3, "k3": 2, "empty": 0}[i], i)),
              "/cascade-runs lists the finished runs, most crossings first, and hides the "
              "running one", f"{ids}")
        j4 = c.get("/api/metro/cascade-runs?k=4").json()
        check([row["run_id"] for row in j4] == ["good4", "empty"]
              and j4[0]["n_crossings"] == 3 and j4[0]["k"] == 4,
              "... and filters to one arity on request",
              f"{[row['run_id'] for row in j4]}")

        # --- status / export / cancel / images
        j = c.get(f"/api/metro/{rid4}/status").json()
        check(j["status"] == "running" and j["k"] == 4 and j["samples"] == []
              and j["chains_stats"] == [] and j["summary"]["n_samples"] == 0
              and j["summary"]["cost_image_eq"] == 0.0 and j["beta"] == 1.4
              and j["chain_steps"] == 5 and j["delta"] == cs.STRIDE,
              "/status shape on a fresh run", f"phase {j['phase']!r}")
        check(c.get("/api/metro/nope/status").json()["status"] == "unknown",
              "an unknown run answers status 'unknown', not a 500")

        # a finished-looking run: the records must survive the round trip
        run4.chains_state = [
            {"chain": 0, "weights": [0.25] * 4, "s": 0.5, "image": 3,
             "seed_weights": [0.1, 0.3, 0.3, 0.3], "seed_s": 0.2, "seed_image": 1,
             "n_propose": 5, "n_accept": 2, "n_outside": 1, "moved": True},
            {"chain": 1, "weights": [0.4, 0.2, 0.2, 0.2], "s": 0.1, "image": 2,
             "seed_weights": [0.4, 0.2, 0.2, 0.2], "seed_s": 0.1, "seed_image": 2,
             "n_propose": 5, "n_accept": 0, "n_outside": 0, "moved": False}]
        run4.samples = [
            {"chain": 0, "step": 2, "weights": [0.2, 0.3, 0.3, 0.2], "s": 0.42,
             "image": 7, "full_image": None},
            {"chain": 0, "step": 4, "weights": [0.25] * 4, "s": 0.5, "image": 3,
             "full_image": 9}]
        run4.seed_crossings = [{"weights": [0.1, 0.3, 0.3, 0.3], "divergence": 0.8},
                               {"weights": [0.4, 0.2, 0.2, 0.2], "divergence": None}]
        run4.n_probes, run4.n_full, run4.round_done = 40, 2, 5
        j = c.get(f"/api/metro/{rid4}/status").json()
        sm = j["summary"]
        check(len(j["samples"]) == 2 and j["samples"][1]["full_image"] == 9
              and len(j["chains_stats"]) == 2 and j["chains_stats"][1]["moved"] is False
              and j["seeds"] == [[0.1, 0.3, 0.3, 0.3], [0.4, 0.2, 0.2, 0.2]]
              and j["seed_divs"] == [0.8, None],
              "/status serves the samples, the per-chain record and the seed crossings")
        check(sm["n_samples"] == 2 and abs(sm["mean_s"] - 0.46) < 1e-9
              and abs(sm["seed_mean_s"] - 0.15) < 1e-9
              and abs(sm["on_boundary_frac"] - 1.0) < 1e-9
              and abs(sm["acceptance"] - 0.2) < 1e-9 and sm["chains_never_moved"] == 1
              and sm["n_outside"] == 1 and sm["rounds_done"] == 5
              # this run was started with probe_steps null, so its probes are whole images
              and abs(sm["cost_image_eq"] - (1.0 * 40 + 2)) < 1e-9,
              "the summary is computed from those records, over the ACCEPTED states",
              f"mean S {sm['mean_s']}, on-boundary {sm['on_boundary_frac']}, "
              f"acceptance {sm['acceptance']}, {sm['cost_image_eq']} image-eq")
        # probe_steps None means every probe was a whole image
        check(abs(c.get(f"/api/metro/{rid3}/status").json()["summary"]["cost_image_eq"]) < 1e-9,
              "a run that has rendered nothing has cost 0")

        e = c.get(f"/api/metro/{rid4}/samples.json")
        ej = e.json()
        check(e.status_code == 200 and len(ej["samples"]) == 2 and ej["k"] == 4
              and ej["beta"] == 1.4 and ej["cos_t"] == COS_T
              and ej["summary"]["n_samples"] == 2 and len(ej["chains_stats"]) == 2,
              "samples.json exports the sample set with the settings it was drawn under",
              f"HTTP {e.status_code}")
        check(c.get("/api/metro/nope/samples.json").status_code == 404,
              "samples.json on an unknown run -> 404")
        check(c.get(f"/api/metro/{rid4}/image/999").status_code == 404,
              "an image index the store never saw -> 404")

        r = c.post(f"/api/metro/{rid4}/cancel")
        check(r.status_code == 200 and r.json() == {"ok": True}
              and app.state.metro_runs[rid4].status == "cancelled",
              "/cancel flips a running run to cancelled")
        r = c.post("/api/metro/nope/cancel")
        check(r.status_code == 200, "/cancel on an unknown run is a no-op, not a 500")
    finally:
        metro.run_metro = orig


if __name__ == "__main__":
    part1()
    part2()
    part3()
    print(f"\n{'ALL CHECKS PASS' if not fails else 'FAILED: ' + ', '.join(fails)}")
    sys.exit(1 if fails else 0)
