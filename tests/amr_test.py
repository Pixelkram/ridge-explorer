"""Offline checks of the adaptive-refinement sampler (backend/services/amr.py). No GPU.

Three parts, all stored-data or analytic -- nothing renders:

  (i)   LATTICE / NESTING -- the level-n lattice is exactly a SUBSET of the level-2n one
        (by weights, 1e-12), the unit-move adjacency count matches the analytic formula at
        several (k, n), `nested_schedule` validates its arguments, and the h25f reference
        numbers (56 cells / 210 pairs at k=4, n=5) are reproduced.
  (ii)  REFINEMENT on a synthetic PLANAR boundary in k=4: embeddings are e_A / e_B by side
        of a plane (cosine distance 0.8, well past COS_T), so the truth is known exactly.
        The detected edges must straddle the plane, the refined share must shrink level by
        level, every true boundary-crossing fine edge must be inside the refined set
        (recall 1.0 at r_ref = 1), and the cost must be 0.5 image-eq per probe.
  (ii-c) DIVERGENCE -- the number AmrMap now fills every dot with by default (the Cascade's
        own divColor ramp): each point's `div` is recomputed from the run's points and the
        lattice adjacency alone, on a SMOOTH field where every adjacent pair has its own
        cosine distance, and must equal the max over the cell's evaluated neighbours (and
        must have reached the shared probe cloud through cascade._set_div).
  (iii) ROUTER through FastAPI's TestClient with a stubbed app.state and amr.run_amr
        replaced by a no-op (as tests/stride_test.py does for the cascade): k=3 and k=4
        start, k=5 is refused with a 400 that names the projected probe count, the bounds
        reject base 2 / levels 6 / factor 4 with a 422, /status has the documented shape,
        and /cancel flips a running run.

Run: python tests/amr_test.py
"""
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.services import amr                       # noqa: E402
from backend.services.cascade import COS_T            # noqa: E402

K = 4
PLANE_SEEDS = range(6)

fails = []


def check(ok, label, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(label)


# ---------------------------------------------------------------- (i) lattice and nesting
def part1():
    print("\n(i) nested lattices, adjacency counts, schedule validation")
    check(amr.nested_schedule(5, 4, 2) == [5, 10, 20, 40],
          "nested_schedule(5, 4, 2) = 5 -> 10 -> 20 -> 40",
          str(amr.nested_schedule(5, 4, 2)))
    check(amr.nested_schedule(5, 3, 3) == [5, 15, 45], "factor 3 ladder")
    check(amr.nested_schedule(7, 1, 2) == [7], "a one-level ladder is just the base")
    for bad, why in (((0, 4, 2), "base 0"), ((5, 0, 2), "levels 0"), ((5, 4, 1), "factor 1")):
        try:
            amr.nested_schedule(*bad)
            check(False, f"{why} rejected", "no ValueError")
        except ValueError as exc:
            check(True, f"{why} rejected", str(exc)[:60])
    sched = amr.nested_schedule(5, 4, 2)
    check(all(b % a == 0 for a, b in zip(sched, sched[1:])),
          "every level of the ladder divides the next (the nesting precondition)")

    # the h25f reference rung
    P = amr.lattice_points(4, 5)
    pr = amr.neighbours(4, 5)
    check(len(P) == 56 and len(pr) == 210,
          "k=4, n=5: 56 cells and 210 adjacency pairs (h25f section 3)",
          f"{len(P)} cells, {len(pr)} pairs")
    check(np.allclose(P.sum(axis=1), 1.0, atol=1e-12),
          "every lattice point is a weight vector summing to 1")

    for k in (3, 4):
        for n in (4, 5, 6, 10):
            got, want = len(amr.neighbours(k, n)), amr.n_pairs(k, n)
            check(got == want, f"k={k}, n={n}: adjacency count matches the analytic formula",
                  f"{got} vs {want}")
            cells = len(amr.lattice_points(k, n))
            check(cells == math.comb(n + k - 1, k - 1),
                  f"k={k}, n={n}: {cells} cells = C(n+k-1, k-1)")
            d = [float(np.linalg.norm(P2[i] - P2[j]))
                 for P2 in (amr.lattice_points(k, n),) for i, j in amr.neighbours(k, n)[:40]]
            check(max(abs(x - amr.lattice_spacing(n)) for x in d) < 1e-12,
                  f"k={k}, n={n}: every unit move is sqrt(2)/n long",
                  f"spacing {amr.lattice_spacing(n):.6f}")

    # NESTING: level n is a subset of level 2n (and of level 3n), by weights
    for k in (3, 4):
        for n in (4, 5):
            for f in (2, 3):
                coarse = amr.lattice_points(k, n)
                fine = amr.lattice_points(k, n * f)
                keyf = {tuple(np.round(w * n * f).astype(int)) for w in fine}
                inside = sum(1 for w in coarse
                             if tuple(np.round(w * n * f).astype(int)) in keyf)
                # and the weights really coincide, not just the integer keys
                worst = max(float(np.abs(fine - w).sum(axis=1).min()) for w in coarse)
                check(inside == len(coarse) and worst < 1e-12,
                      f"k={k}: every level-{n} point is a level-{n * f} point",
                      f"{inside}/{len(coarse)} matched, worst |dw| {worst:.1e}")

    # cost model
    check(amr.probe_cost(4) == 0.5 and amr.probe_cost(None) == 1.0,
          "a cheap probe is 0.5 image-eq, a full-fidelity one 1.0")
    check(abs(amr.boundary_fraction(5) - 0.7679) < 1e-12
          and abs(amr.boundary_fraction(60) - 0.2598) < 1e-12
          and 0.2598 < amr.boundary_fraction(40) < 0.3783,
          "phi(n) pins the measured k=4 levels and interpolates between them",
          f"phi(40) = {amr.boundary_fraction(40):.4f}")
    tot4, rows4 = amr.projected_probes(4, 5, 4, 2)
    check(rows4[0]["probes"] == 56 and tot4 > 56
          and all(r["probes"] <= r["cells"] for r in rows4),
          "the projection evaluates the coarse level in full and caps every rung at its"
          " own cell count", f"k=4 ladder 5->40: {tot4} probes")
    tot5, _ = amr.projected_probes(5, 5, 4, 2)
    check(tot5 > 10 * tot4, "the same ladder at k=5 costs an order of magnitude more"
                            " (2^d fan-out)", f"{tot5} vs {tot4} probes")


# ------------------------------------------------- (ii) refinement on a planar boundary
def _plane(seed):
    """A sum-zero unit normal p; g(w) = (w - 1/k).p is the exact signed tangent distance."""
    rng = np.random.default_rng(seed)
    p = rng.standard_normal(K)
    p -= p.mean()
    p /= np.linalg.norm(p)
    return p


def _side_embeddings(pts, p):
    """e_A on one side of the plane, e_B on the other: orthogonal unit vectors scaled so
    1 - cos across the plane is 0.8 (far above COS_T = 0.35) and 0 within a side."""
    a, b = np.zeros(8), np.zeros(8)
    a[0] = 1.0
    b[0], b[1] = 0.2, math.sqrt(1 - 0.2 ** 2)      # 1 - cos(a, b) = 0.8
    s = (pts - 1.0 / K) @ p
    return [a if v > 0 else b for v in s], s


def part2():
    print("\n(ii) refinement on a synthetic plane (k=4, exact truth)")
    a, b = np.zeros(8), np.zeros(8)
    a[0] = 1.0
    b[0], b[1] = 0.2, math.sqrt(1 - 0.2 ** 2)
    from backend.services.cascade import _cosd
    check(abs(_cosd(a, b) - 0.8) < 1e-12 and _cosd(a, a) == 0.0,
          "the synthetic sides are 0.8 apart and 0 within a side (COS_T = 0.35)",
          f"cosd {_cosd(a, b):.3f}")

    sched = amr.nested_schedule(5, 3, 2)              # 5 -> 10 -> 20
    worst_recall, shrank, costs_ok = 1.0, True, True
    for seed in PLANE_SEEDS:
        p = _plane(seed)
        emb_by_key = {}
        fine_n = sched[-1]
        prev_pts, prev_edges = None, None
        fracs, evaluated_tot = [], 0
        for li, n in enumerate(sched):
            pts = amr.lattice_points(K, n)
            scale = fine_n // n
            keys = [tuple(int(v) * scale for v in c) for c in amr.lattice_counts(K, n)]
            carried = np.array([key in emb_by_key for key in keys], dtype=bool)
            if li == 0:
                want = np.ones(len(pts), dtype=bool)
            else:
                want = amr.refine_set(prev_pts, prev_edges, n, r_ref=1)
                # RECALL: every fine pair that really straddles the plane must have both
                # ends inside the refined set, or the pass has lost a boundary for good
                s = (pts - 1.0 / K) @ p
                true_pairs = [(i, j) for i, j in amr.neighbours(K, n)
                              if (s[i] > 0) != (s[j] > 0)]
                take_r = want | carried
                kept = sum(1 for i, j in true_pairs if take_r[i] and take_r[j])
                worst_recall = min(worst_recall, kept / max(len(true_pairs), 1))
            take = want | carried
            fracs.append(float(take.mean()))
            evaluated_tot += int((take & ~carried).sum())
            side, s = _side_embeddings(pts, p)
            lvl_emb = [side[i] if take[i] else None for i in range(len(pts))]
            for i in range(len(pts)):
                if take[i]:
                    emb_by_key[keys[i]] = side[i]
            edges = amr.detect_edges(lvl_emb, amr.neighbours(K, n))
            # every detected edge must really straddle the plane (no false positives on a
            # field whose only structure IS the plane)
            if any((s[i] > 0) == (s[j] > 0) for i, j, _d in edges):
                check(False, f"seed {seed}: a detected edge did not straddle the plane")
            if any(abs(d - 0.8) > 1e-12 for _i, _j, d in edges):
                check(False, f"seed {seed}: edge divergence is not the synthetic 0.8")
            prev_pts, prev_edges = pts, edges
        if not all(x > y for x, y in zip(fracs, fracs[1:])):
            shrank = False
        cost = sum(amr.probe_cost(4) * c for c in [evaluated_tot])
        if abs(cost - 0.5 * evaluated_tot) > 1e-12:
            costs_ok = False
        if seed == PLANE_SEEDS[0]:
            print(f"      seed {seed}: refined share {[round(f, 3) for f in fracs]}, "
                  f"{evaluated_tot} probes = {cost:.1f} image-eq")
    check(worst_recall == 1.0,
          "r_ref=1 keeps every true boundary-crossing fine edge (recall 1.0)",
          f"worst recall over {len(list(PLANE_SEEDS))} planes {worst_recall:.4f}")
    check(shrank, "the refined share shrinks at every level (refinement is selective)")
    check(costs_ok, "cost = 0.5 image-eq per evaluated probe")

    # the detector's own contract: a missing end is evidence of nothing
    pts = amr.lattice_points(K, 5)
    side, _s = _side_embeddings(pts, _plane(0))
    pairs = amr.neighbours(K, 5)
    full = amr.detect_edges(side, pairs)
    holed = amr.detect_edges([None if i % 3 == 0 else e for i, e in enumerate(side)], pairs)
    check(len(holed) < len(full) and all(
        side[i] is not None and side[j] is not None for i, j, _d in holed),
        "detect_edges skips pairs with an unevaluated end rather than counting them quiet",
        f"{len(holed)} of {len(full)} edges survive a third of the cells going missing")
    divs = amr.pair_divergences(side, pairs)
    check(set(divs) <= set(range(len(pts)))
          and all(0.0 <= v <= 0.8 + 1e-12 for v in divs.values())
          and max(divs.values()) == 0.8,
          "pair_divergences reports the max 1-cos per cell, in the detector's units",
          f"{len(divs)} cells carry a divergence")
    check(amr.refine_set(pts, [], 10, 1).sum() == 0,
          "no boundary edge -> an empty refined set (not an error)")


# ------------------------------------------- (ii-b) the real run_amr on the same plane
def _stub_evaluate(run, field, labels):
    """Stand-in for cascade.evaluate (the shape tests/certify_test.py uses): embeds every
    weight vector through the analytic field and records the label of each GPU round."""
    def stub(app, r, pool, weights, seed, label, ctl=None, on_arrival=None, steps=None):
        labels.append(label.split(".")[0])
        idxs = []
        for w in weights:
            gi = run.next_idx
            run.next_idx += 1
            run.embeddings[gi] = field(w)
            run.generated += 1
            run.phase_done += 1
            # the live-cloud bookkeeping cascade.evaluate does, so cascade._set_div has
            # somewhere to write and the map's shared colouring can be checked too
            run._geo_pos[gi] = len(run.probe_geo)
            run.probe_geo.append([float(v) for v in w])
            run.probe_div.append(None)
            idxs.append(gi)
        return idxs
    return stub


def _smooth_field(seed=0, scale=7.0, modes=5):
    """A field whose every adjacent pair has its OWN cosine distance: a few random sinusoidal
    modes of the position, mixed into an 8-vector and normalised, so e(w) wanders over the
    unit sphere instead of along one circle.

    The plane field above cannot tell a max-over-neighbours rule from several wrong ones
    (every answer is 0 or 0.8), and a field linear in w cannot either (the distance then
    depends only on WHICH unit move it is, so one level has k(k-1)/2 answers in total). This
    one gives a different number per pair, and still crosses COS_T often enough that the
    ladder refines.
    """
    rng = np.random.default_rng(seed)
    C = rng.standard_normal((modes, K))
    C -= C.mean(axis=1, keepdims=True)          # sum-zero: only tangent motion is seen
    ph = rng.uniform(0.0, 2.0 * math.pi, modes)
    A = rng.standard_normal((modes, 8))

    def f(w):
        x = np.asarray(w, dtype=float) - 1.0 / K
        e = A.T @ np.sin(scale * (C @ x) + ph)
        nn = float(np.linalg.norm(e))
        return e / nn if nn > 1e-12 else np.eye(8)[0]
    return f


def _survey(levels=3, r_ref=1, seed=3, stop_after=None, field=None):
    """One stubbed AMR survey over the planar field; returns (run, plane normal, labels)."""
    from backend.services import cascade as cs
    p = _plane(seed)
    a, b = np.zeros(8), np.zeros(8)
    a[0] = 1.0
    b[0], b[1] = 0.2, math.sqrt(1 - 0.2 ** 2)
    if field is None:
        field = lambda w: a if float(np.dot(np.asarray(w) - 1.0 / K, p)) > 0 else b
    run = amr.AmrRun(run_id="a", prompts=[f"p{i}" for i in range(K)], seed=7, steps=8,
                     height=64, width=64, guidance_scale=3.5, base=5, levels=levels,
                     factor=2, r_ref=r_ref)
    labels = []
    orig = cs.evaluate
    stub = _stub_evaluate(run, field, labels)
    if stop_after is not None:
        def cancelling(*args, **kw):
            out = stub(*args, **kw)
            if len(labels) >= stop_after:
                run.status = "cancelled"       # the UI's Cancel, mid-level
            return out
        cs.evaluate = cancelling
    else:
        cs.evaluate = stub
    try:
        amr.run_amr(None, run, None)
    finally:
        cs.evaluate = orig
    return run, p, labels


def part2b():
    print("\n(ii-b) run_amr itself, stubbed on the same plane (no GPU)")
    run, p, labels = _survey()
    sched = amr.nested_schedule(5, 3, 2)
    check(run.status == "complete" and run.phase == "done"
          and [r["level"] for r in run.level_stats] == sched,
          "the ladder runs to the end, one round per level",
          f"{run.status}/{run.phase}, rounds {labels}")
    check(labels == [f"amr{n}" for n in sched],
          "exactly one batched evaluate per level", str(labels))

    probes = sum(r["n_evaluated"] for r in run.level_stats)
    check(probes == len(run.points) == run.generated,
          "every probe becomes exactly one point (nesting never re-probes a cell)",
          f"{probes} probes, {len(run.points)} points")
    keys = {tuple(np.round(np.asarray(pt["weights"]) * sched[-1]).astype(int))
            for pt in run.points}
    check(len(keys) == len(run.points), "no cell is evaluated twice across the levels")
    check(abs(run.cost_image_eq - 0.5 * probes) < 1e-9
          and all(abs(r["cost_image_eq"] - 0.5 * r["n_evaluated"]) < 1e-9
                  for r in run.level_stats),
          "per-level and total cost are 0.5 image-eq per probe",
          f"{run.cost_image_eq:.1f} image-eq for {probes} probes")
    shares = [r["n_candidates"] / r["cells"] for r in run.level_stats]
    check(shares[0] == 1.0 and all(x > y for x, y in zip(shares, shares[1:])),
          "level 0 is exhaustive and every refinement is a smaller share of its lattice",
          " ".join(f"{s:.2f}" for s in shares))
    check(all(r["n_edges"] > 0 for r in run.level_stats)
          and len(run.edges) == sum(r["n_edges"] for r in run.level_stats),
          "every level finds boundary edges and all of them are recorded",
          f"{len(run.edges)} edges")
    g = {i: float(np.dot(np.asarray(pt["weights"]) - 1.0 / K, p))
         for i, pt in enumerate(run.points)}
    check(all((g[e["a"]] > 0) != (g[e["b"]] > 0) for e in run.edges)
          and all(abs(e["divergence"] - 0.8) < 1e-9 for e in run.edges),
          "every recorded edge straddles the plane, at the synthetic divergence")
    check(all(pt["div"] is not None for pt in run.points)
          and max(pt["div"] for pt in run.points) == 0.8,
          "each point carries its local divergence for the map colouring")

    # r_ref = 2 is a superset of r_ref = 1: strictly more probes, never fewer edges
    wide, _p2, _l = _survey(r_ref=2)
    check(sum(r["n_evaluated"] for r in wide.level_stats) > probes
          and len(wide.edges) >= len(run.edges),
          "r_ref=2 spends more probes and finds at least as many edges",
          f"{sum(r['n_evaluated'] for r in wide.level_stats)} vs {probes} probes")

    # cancel: the run stops between levels and keeps the verdict
    cut, _p3, lab = _survey(levels=3, stop_after=1)
    check(cut.status == "cancelled" and len(lab) == 1 and len(cut.level_stats) <= 1,
          "a cancel during the first level stops the ladder and stays cancelled",
          f"{cut.status}, {len(lab)} rounds, {len(cut.level_stats)} level rows")


# --------------------------- (ii-c) the `div` the map colours by, recomputed independently
def part2c():
    """`div` is what AmrMap fills a dot with (on the Cascade's own divColor ramp), so it is
    worth pinning to its definition rather than to a range: the max cosine distance from the
    cell to the EVALUATED LATTICE NEIGHBOURS it ever had. Recomputed here from the run's own
    points and the lattice adjacency alone -- no refine_set, no pair_divergences -- on the
    smooth field, where every pair has a different answer."""
    print("\n(ii-c) point `div` = max cosd to the evaluated lattice neighbours (smooth field)")
    from backend.services.cascade import _cosd
    field = _smooth_field()
    run, _p, _labels = _survey(levels=3, field=field)
    sched = [r["level"] for r in run.level_stats]
    fine_n = amr.nested_schedule(5, 3, 2)[-1]

    # the run's points, keyed the way run_amr keys them (finest-level integer counts)
    at = {tuple(int(round(v * fine_n)) for v in pt["weights"]): i
          for i, pt in enumerate(run.points)}
    check(len(at) == len(run.points), "every point has a distinct finest-level lattice key")
    want = {}
    for n in sched:
        scale = fine_n // n
        here = [at.get(tuple(int(v) * scale for v in c))
                for c in amr.lattice_counts(K, n)]
        for i, j in amr.neighbours(K, n):
            pa, pb = here[i], here[j]
            if pa is None or pb is None:        # one end of the pair was never evaluated
                continue
            d = _cosd(field(run.points[pa]["weights"]), field(run.points[pb]["weights"]))
            for t in (pa, pb):
                cur = want.get(t)
                want[t] = d if cur is None else max(cur, d)
    got = {i: pt["div"] for i, pt in enumerate(run.points)}

    n_distinct = len(set(round(v, 9) for v in want.values()))
    check(len(sched) > 1 and len(run.points) > 56 and n_distinct > 20,
          "the survey refined past its coarse level and the field is genuinely varied",
          f"levels {sched}, {len(run.points)} points, {n_distinct} distinct divergences")
    check(all(v is not None for v in got.values()),
          "`div` is present on every evaluated point",
          f"{sum(1 for v in got.values() if v is None)} missing of {len(got)}")
    check(set(want) == set(got),
          "exactly the points with a measured neighbour carry a divergence",
          f"{len(want)} expected, {len(got)} served")
    worst = max((abs(got[i] - want[i]) for i in want), default=0.0)
    check(worst < 1e-12,
          "every point's `div` equals the max cosd over its evaluated lattice neighbours",
          f"worst |served - recomputed| {worst:.1e} over {len(want)} points")
    # and the same number reached the live-cloud colouring the Cascade shares with us
    cloud = {run.points[i]["image"]: run.points[i]["div"] for i in range(len(run.points))}
    bad = [gi for gi, d in cloud.items()
           if run._geo_pos.get(gi) is not None
           and abs((run.probe_div[run._geo_pos[gi]] or -1) - d) > 1e-12]
    check(not bad, "_set_div carried the same divergence into the shared probe cloud",
          f"{len(cloud)} cells, {len(bad)} disagreeing")


# ------------------------------------------------------------------- (iii) router
def part3():
    print("\n(iii) router through TestClient (stubbed state, run_amr no-op)")
    try:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
    except Exception as exc:                         # noqa: BLE001
        check(False, "TestClient available", str(exc))
        return
    from backend.routers import amr as router_mod

    class _NoCache:
        """A truthy stand-in ThumbnailStore keeps as-is, so nothing touches the disk. It
        answers the one call the image route makes (an index the store never saw), which a
        bare object() cannot: __getitem__ looks up `get_path` BEFORE the missing key."""
        def has(self, h):
            return False

        def save(self, h, data):
            pass

        def get_path(self, h):
            return None

    orig = amr.run_amr
    amr.run_amr = lambda app, run, pool: None        # the worker thread renders nothing
    try:
        app = FastAPI()
        app.include_router(router_mod.router)
        app.state.amr_runs = {}
        app.state.cache = _NoCache()
        app.state.gpu_pool = None
        c = TestClient(app, raise_server_exceptions=False)

        r = c.post("/api/amr/start", json={"prompts": ["a", "b", "c"]})
        rid3 = r.json().get("run_id", "")
        run3 = app.state.amr_runs.get(rid3)
        check(r.status_code == 200 and run3 is not None and run3.k == 3
              and (run3.base, run3.levels, run3.factor, run3.r_ref) == (5, 4, 2, 1)
              and run3.probe_steps == 4,
              "k=3 starts, with the ladder of record as the defaults",
              f"HTTP {r.status_code}")

        r = c.post("/api/amr/start",
                   json={"prompts": ["a", "b", "c", "d"], "base": 4, "levels": 3,
                         "factor": 3, "r_ref": 2, "seed": 7, "probe_steps": None})
        rid4 = r.json().get("run_id", "")
        run4 = app.state.amr_runs.get(rid4)
        check(r.status_code == 200 and run4 is not None and run4.k == 4
              and (run4.base, run4.levels, run4.factor, run4.r_ref) == (4, 3, 3, 2)
              and run4.seed == 7 and run4.probe_steps is None,
              "k=4 starts and every setting is threaded into the run",
              f"HTTP {r.status_code}")

        r = c.post("/api/amr/start", json={"prompts": ["a", "b", "c", "d", "e"]})
        probes, _rows = amr.projected_probes(5, 5, 4, 2)
        body = r.text
        check(r.status_code == 400 and str(probes) in body
              and str(amr.H25F_K9_PROBES) in body,
              "k=5 is refused with a 400 naming the projected probe count",
              f"HTTP {r.status_code}: {body[:150]}")

        for bad, why in (({"base": 2}, "base 2"), ({"levels": 6}, "levels 6"),
                         ({"factor": 4}, "factor 4"), ({"r_ref": 3}, "r_ref 3")):
            r = c.post("/api/amr/start", json=dict({"prompts": ["a", "b", "c"]}, **bad))
            check(r.status_code == 422, f"{why} is rejected by the bounds",
                  f"got HTTP {r.status_code}")
        r = c.post("/api/amr/start", json={"prompts": ["a", "b"]})
        check(r.status_code == 422, "fewer than 3 prompts is rejected",
              f"got HTTP {r.status_code}")
        # a legal-looking ladder whose finest lattice is past the cell ceiling
        r = c.post("/api/amr/start",
                   json={"prompts": ["a", "b", "c", "d"], "base": 10, "levels": 5,
                         "factor": 3})
        check(r.status_code == 400 and "ceiling" in r.text,
              "a ladder past the cell ceiling is refused, not queued",
              f"HTTP {r.status_code}: {r.text[:120]}")

        j = c.get(f"/api/amr/{rid4}/status").json()
        check(j["status"] == "running" and j["k"] == 4 and j["schedule"] == [4, 12, 36]
              and j["points"] == [] and j["edges"] == [] and j["levels_stats"] == []
              and j["cost_image_eq"] == 0.0 and j["r_ref"] == 2,
              "/status shape on a fresh run", f"schedule {j['schedule']}")
        check(c.get("/api/amr/nope/status").json()["status"] == "unknown",
              "an unknown run answers status 'unknown', not a 500")

        # a finished-looking run: the parallel arrays must stay aligned, and the export
        # must say the same thing in record form
        run4.points = [{"weights": [0.25] * 4, "level": 4, "image": 3, "div": 0.5},
                       {"weights": [0.5, 0.5, 0.0, 0.0], "level": 12, "image": 4,
                        "div": None}]
        run4.edges = [{"a": 0, "b": 1, "level": 12, "divergence": 0.8}]
        run4.level_stats = [{"level": 4, "cells": 35, "n_candidates": 35,
                             "n_evaluated": 35, "n_edges": 1, "cost_image_eq": 17.5}]
        j = c.get(f"/api/amr/{rid4}/status").json()
        check(len(j["points"]) == len(j["point_levels"]) == len(j["point_images"])
              == len(j["point_divs"]) == 2
              and j["point_levels"] == [4, 12] and j["point_divs"] == [0.5, None]
              and j["edges"] == [[0, 1]] and j["edge_divs"] == [0.8]
              and j["edge_levels"] == [12] and j["cost_image_eq"] == 17.5,
              "/status serves points and edges as aligned parallel arrays")
        e = c.get(f"/api/amr/{rid4}/points.json")
        ej = e.json()
        check(e.status_code == 200 and [p["id"] for p in ej["points"]] == [0, 1]
              and ej["points"][0]["level"] == 4 and ej["edges"][0]["divergence"] == 0.8
              and ej["schedule"] == [4, 12, 36] and ej["k"] == 4,
              "points.json exports the same survey as records", f"HTTP {e.status_code}")
        check(c.get("/api/amr/nope/points.json").status_code == 404,
              "points.json on an unknown run -> 404")
        check(c.get(f"/api/amr/{rid4}/image/999").status_code == 404,
              "an image index the store never saw -> 404")

        r = c.post(f"/api/amr/{rid4}/cancel")
        check(r.status_code == 200 and r.json() == {"ok": True}
              and app.state.amr_runs[rid4].status == "cancelled",
              "/cancel flips a running run to cancelled")
        r = c.post("/api/amr/nope/cancel")
        check(r.status_code == 200, "/cancel on an unknown run is a no-op, not a 500")
    finally:
        amr.run_amr = orig


if __name__ == "__main__":
    part1()
    part2()
    part2b()
    part2c()
    part3()
    print(f"\n{'ALL CHECKS PASS' if not fails else 'FAILED: ' + ', '.join(fails)}")
    sys.exit(1 if fails else 0)
