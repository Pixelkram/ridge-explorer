"""Offline checks of the k-NEAREST-NEIGHBOUR DIVERGENCE of the hi-res cloud points
(backend/services/cascade.py). No GPU: the stubbed-survey machinery of tests/hires_cloud_test.py
is reused, so the real chord phase, the real cloud phase and the real certify half run unchanged.

A cloud point used to carry only a LABEL -- which of the crossing's two sides its image matched.
It now also carries a DIVERGENCE: the mean cosine distance to its k nearest neighbours among the
cloud and the crossing's two bracket ends (_cloud_knn_div). That is the cloud's stand-in for the
4 GRID neighbours a lattice sensitivity averages over (discover.sensitivity_field,
hiker.sensitivity), in the same units, which is why a cloud point can now join the run's
divergence colouring.

  (a) HELPER -- _cloud_knn_div on synthetic pools: on a two-basin cloud (cosd between the basins
      0.8) the points within one neighbour spacing of the dividing plane read above COS_T while
      deep points read 0; on a random ball the high divergences sit nearest the plane; k is
      clipped to the pool (3 cloud points + 2 bracket ends, k=16 -> 4 neighbours); the bracket
      ends act as neighbours, so a point whose only near neighbours ARE the ends reads their
      divergence; a pool of one has nothing to average.
  (b) PIPELINE -- a stubbed cloud survey: every stored point carries its divergence, each cloud
      reports k / div_median / div_max / boundary_frac consistent with its own points, the
      divergences are exactly what the helper gives for the pool (cloud + bracket ends), k is
      threaded from the run, and every cloud image reached the run's live divergence map
      (_set_div).
  (c) REQUEST -- /api/cascade/start through FastAPI's TestClient: hires_cloud_k defaults to 4,
      is threaded into the run and the status, and 0 / 17 are rejected.

Run: python tests/hires_cloud_knn_test.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.services import cascade as cs            # noqa: E402
from hires_cloud_test import _survey                  # noqa: E402  (same stubbed survey)

fails = []


def check(ok, label, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(label)


def _two_basins(gap=0.8):
    """Two unit embeddings a cosine distance `gap` apart -- one basin each side of a plane."""
    ea = np.zeros(8)
    ea[0] = 1.0
    perp = np.zeros(8)
    perp[1] = 1.0
    dot = 1.0 - gap
    return ea, dot * ea + np.sqrt(1.0 - dot * dot) * perp


def _iso(k, rng):
    """A unit sum-zero direction -- the tangent space the cloud lives in."""
    u = rng.standard_normal(k)
    u -= u.mean()
    return u / np.linalg.norm(u)


def part_a():
    print("\n(a) the helper: mean distance to the k nearest members of the pool")
    k, h = 4, 0.01
    c = np.full(k, 1.0 / k)
    v = _iso(k, np.random.default_rng(5))
    ea, eb = _two_basins(0.8)
    check(abs(cs._cosd(ea, eb) - 0.8) < 1e-12, "control: the two basins are 0.8 apart",
          f"1-cos {cs._cosd(ea, eb):.3f} vs COS_T {cs.COS_T}")

    # A CHAIN along the plane normal, spacing h: the one geometry whose neighbour spacing is
    # exact, so what "within one spacing of the plane" buys can be read off. At |t| = h/2 two of
    # the four nearest lie across the plane (0.8 each) and two beside it (0), i.e. 0.4 > COS_T;
    # from |t| = 5h/2 outwards all four are on the same side, i.e. 0.
    ts = np.array([(i + 0.5) * h for i in range(-9, 9)])
    pts = [c + t * v for t in ts]
    embs = [eb if t > 0 else ea for t in ts]
    divs = cs._cloud_knn_div(pts, embs, 4)
    near = [d for t, d in zip(ts, divs) if abs(t) < h]
    deep = [d for t, d in zip(ts, divs) if abs(t) > 2 * h]
    check(len(near) == 2 and all(d > cs.COS_T for d in near),
          "a point within one neighbour spacing of the plane reads above COS_T",
          f"{len(near)} such points, div {['%.2f' % d for d in near]}")
    check(len(deep) > 0 and max(deep) < 1e-9,
          "a point deeper than its neighbour spacing reads 0 -- inside one basin",
          f"{len(deep)} such points, max div {max(deep):.2e}")
    check(len(divs) == len(pts) and all(d is not None for d in divs),
          "one divergence per pool entry, in order", f"{len(divs)} for {len(pts)} points")

    # the same reading on the ball the pass actually draws: the field has to PEAK at the plane
    rng = np.random.default_rng(17)
    ball = cs._cloud_points(c, 0.05, 80, rng)
    bd = cs._cloud_knn_div(ball, [eb if float(np.dot(p - c, v)) > 0 else ea for p in ball], 4)
    off = np.array([abs(float(np.dot(p - c, v))) for p in ball])
    order = np.argsort(bd)
    q = max(1, len(ball) // 4)
    hi, lo = off[order[-q:]].mean(), off[order[:q]].mean()
    check(hi < lo, "on a random ball the highest divergences sit nearest the plane",
          f"mean |t·v| {hi:.4f} (top quartile by div) vs {lo:.4f} (bottom)")

    # k is CLIPPED to the pool: 3 cloud points + 2 bracket ends is 4 possible neighbours
    small = [c, c + h * v, c + 2 * h * v, c - h * v, c + 3 * h * v]          # 3 cloud, 2 ends
    se = [ea, eb, _two_basins(0.3)[1], _two_basins(0.5)[1], _two_basins(0.9)[1]]
    got = cs._cloud_knn_div(small, se, 16)[0]
    want4 = float(np.mean([cs._cosd(se[0], e) for e in se[1:]]))
    want2 = float(np.mean([cs._cosd(se[0], se[1]), cs._cosd(se[0], se[3])]))
    check(abs(got - want4) < 1e-12 and abs(got - want2) > 1e-6,
          "k=16 on a pool of 5 averages over the 4 others, not over fewer",
          f"div {got:.4f} vs all-4 {want4:.4f} / nearest-2 {want2:.4f}")

    # the bracket ends are neighbours: a point they are the nearest members of reads THEIR
    # divergence, which is how a cloud that sampled only one basin still sees the boundary
    pool = [c, c + 0.20 * v, c + 0.22 * v, c + 0.24 * v,                     # 4 cloud points
            c - h * v, c + h * v]                                            # 2 bracket ends
    pe = [_two_basins(0.5)[1], ea, ea, ea, ea, eb]
    pd = cs._cloud_knn_div(pool, pe, 2)
    want = float(np.mean([cs._cosd(pe[0], ea), cs._cosd(pe[0], eb)]))
    check(abs(pd[0] - want) < 1e-12,
          "a point whose nearest neighbours are the two bracket ends reads their divergence",
          f"div {pd[0]:.4f} vs expected {want:.4f}")
    check(max(pd[1:4]) < 1e-9,
          "the cloud points far from the ends read off each other instead",
          f"div {['%.2e' % d for d in pd[1:4]]}")
    check(cs._cloud_knn_div([c], [ea], 4) == [None] and cs._cloud_knn_div([], [], 4) == [],
          "a pool of one has no neighbour to average over: None")


def part_b():
    print("\n(b) the divergence inside the real loop (stubbed field)")
    run, _r = _survey(hires=True)
    clouded = [x for x in run.crossings if x.cloud is not None]
    check(bool(clouded)
          and all(isinstance(p[3], float) and np.isfinite(p[3])
                  for x in clouded for p in x.cloud_pts),
          "every stored cloud point carries a finite divergence",
          f"{sum(len(x.cloud_pts) for x in clouded)} points over {len(clouded)} clouds")
    # the stored divergences ARE the helper's, on the pool of the cloud plus the two bracket ends
    same = True
    for x in clouded:
        want = cs._cloud_knn_div(
            [np.asarray(p[0]) for p in x.cloud_pts] + [x.wa, x.wb],
            [run.embeddings[p[2]] for p in x.cloud_pts] + [x.ea, x.eb],
            run.hires_cloud_k)[:len(x.cloud_pts)]
        same = same and all(abs(p[3] - w) < 1e-12 for p, w in zip(x.cloud_pts, want))
    check(same, "each divergence is the k-NN mean over the cloud AND the two bracket ends")
    summ = all(
        x.cloud["k"] == run.hires_cloud_k
        and abs(x.cloud["div_median"] - float(np.median([p[3] for p in x.cloud_pts]))) < 1e-12
        and abs(x.cloud["div_max"] - max(p[3] for p in x.cloud_pts)) < 1e-12
        and abs(x.cloud["boundary_frac"]
                - sum(1 for p in x.cloud_pts if p[3] > cs.COS_T) / len(x.cloud_pts)) < 1e-12
        for x in clouded)
    check(summ, "each cloud's k / div_median / div_max / boundary_frac match its own points",
          f"cloud 0: k {clouded[0].cloud['k']}, median "
          f"{clouded[0].cloud['div_median']:.2f}, max {clouded[0].cloud['div_max']:.2f}, "
          f"boundary frac {clouded[0].cloud['boundary_frac']:.2f}" if clouded else "none")
    check(any(x.cloud["boundary_frac"] > 0 for x in clouded),
          "on a field of bands some of the ball really does sit on the boundary",
          f"boundary fracs {['%.2f' % x.cloud['boundary_frac'] for x in clouded]}")

    # the whole point of defining a divergence: the cloud joins the live colouring
    fed = [(run.probe_div[run._geo_pos[p[2]]], p[3])
           for x in clouded for p in x.cloud_pts if p[2] in run._geo_pos]
    check(len(fed) == sum(len(x.cloud_pts) for x in clouded)
          and all(d is not None and d >= pd - 1e-12 for d, pd in fed),
          "every cloud image is in the run's divergence map, at least at its own reading",
          f"{len(fed)} points fed")

    # k is a setting, not a constant: a different k reads the same cloud differently
    k1, _r1 = _survey(hires=True, hires_cloud_k=1)
    c1 = [x for x in k1.crossings if x.cloud is not None]
    check(all(x.cloud["k"] == 1 for x in c1)
          and any(abs(a.cloud["div_median"] - b.cloud["div_median"]) > 1e-9
                  for a, b in zip(c1, clouded)),
          "hires_cloud_k is threaded through and changes the reading",
          f"k=1 medians {['%.2f' % x.cloud['div_median'] for x in c1]} vs "
          f"k=4 {['%.2f' % x.cloud['div_median'] for x in clouded]}")


def part_c():
    print("\n(c) /start and /status through TestClient (stubbed state, run_cascade no-op)")
    try:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
    except Exception as exc:                     # noqa: BLE001
        check(False, "TestClient available", str(exc))
        return
    from backend.routers import cascade as router_mod

    orig = cs.run_cascade
    cs.run_cascade = lambda app, run, pool: None      # the worker thread renders nothing
    try:
        app = FastAPI()
        app.include_router(router_mod.router)
        app.state.cascades = {}
        app.state.cache = object()      # ThumbnailStore keeps a truthy cache as-is (no disk)
        app.state.gpu_pool = None
        c = TestClient(app)
        body = dict(k=3, prompts=["a", "b", "c"], n_chords=4, n_patches=2)

        r = c.post("/api/cascade/start", json=body)
        rid = r.json().get("run_id", "")
        run = app.state.cascades.get(rid)
        check(r.status_code == 200 and run is not None and run.hires_cloud_k == 4,
              "an omitted hires_cloud_k defaults to 4 -- the lattice's own 4 neighbours",
              f"got {getattr(run, 'hires_cloud_k', None)!r}")
        if rid:
            j = c.get(f"/api/cascade/{rid}/status").json()
            check(j["hires_cloud_k"] == 4, "status reports the run's own k")

        r = c.post("/api/cascade/start",
                   json=dict(body, hires=True, hires_mode="cloud", hires_cloud_k=9))
        rid = r.json().get("run_id", "")
        run = app.state.cascades.get(rid)
        check(r.status_code == 200 and run is not None and run.hires_cloud_k == 9,
              "a given k is threaded into the run", f"got {getattr(run, 'hires_cloud_k', None)!r}")
        if rid:
            j = c.get(f"/api/cascade/{rid}/status").json()
            check(j["hires_cloud_k"] == 9, "status reports a run started with that k")

        for bad in (0, 17):
            r = c.post("/api/cascade/start", json=dict(body, hires_cloud_k=bad))
            check(r.status_code == 422, f"hires_cloud_k={bad} rejected",
                  f"got HTTP {r.status_code}")
    finally:
        cs.run_cascade = orig


if __name__ == "__main__":
    part_a()
    part_b()
    part_c()
    print(f"\n{'ALL CHECKS PASS' if not fails else 'FAILED: ' + ', '.join(fails)}")
    sys.exit(1 if fails else 0)
