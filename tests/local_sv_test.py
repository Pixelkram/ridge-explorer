"""Offline checks of the local boundary-density map (backend/services/local_sv.py). No GPU.

Two parts, both stored-data only:

  (i)  SYNTHETIC -- a single flat boundary plane through the barycentre of a k = 4 simplex,
       read by 200 chords. The truth is known exactly (distance to the plane), so the map
       must rank near-the-plane above far-from-it: Spearman(S_hat, |distance|) <= -0.7. Also
       pins c_3 = 2 exactly and checks the geometry the estimator rests on.
  (ii) ADAPTER -- from_run() on a REAL finished cascade. Runs live in app.state only, so the
       fixture is a persisted CascadeStatus: tests/walk_gt_dev/dev_k3T0_c1.json (k = 3,
       16 chords, 57 crossings, from the walk ground-truth development harness). The router
       is exercised through FastAPI's TestClient with a stubbed app.state -- no server.

Run: python tests/local_sv_test.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.services import local_sv as lsv           # noqa: E402

K = 4
N_CHORDS = 200
REAL_RUN = ROOT / "tests" / "walk_gt_dev" / "dev_k3T0_c1.json"

fails = []


def check(ok, label, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(label)


def _extent(w, u):
    """Chord half-extents to the simplex walls -- the cascade's own (services/cascade.py)."""
    tpos = min((w[i] / -u[i]) for i in range(len(w)) if u[i] < 0)
    tneg = max((-w[i] / u[i]) for i in range(len(w)) if u[i] > 0)
    return tneg, tpos


def synthetic(seed=7):
    """200 chords across ONE flat boundary plane; the plane's normal is a sum-zero unit
    vector, so g(w) = (w - 1/k).p is exactly the signed tangent distance to it."""
    rng = np.random.default_rng(seed)
    p = rng.standard_normal(K)
    p -= p.mean()
    p /= np.linalg.norm(p)
    g = lambda w: float(np.dot(np.asarray(w) - 1.0 / K, p))

    chords, crossings, straddled = [], [], 0
    while len(chords) < N_CHORDS:
        w0 = rng.dirichlet(np.ones(K))
        u = rng.standard_normal(K)
        u -= u.mean()
        u /= np.linalg.norm(u)
        tneg, tpos = _extent(w0, u)
        if tpos - tneg < 0.1:
            continue
        a, b = np.clip(w0 + tneg * u, 0, None), np.clip(w0 + tpos * u, 0, None)
        chords.append((a, b))
        ga, gb = g(a), g(b)
        if ga * gb < 0:                       # a plane and a line meet at most once
            straddled += 1
            crossings.append((None, a + (-ga / (gb - ga)) * (b - a), straddled % 3 == 0))
    return chords, crossings, g, straddled


def part1():
    print("\n(i) synthetic: one flat plane, k = 4, 200 chords")
    check(lsv.crofton_constant(3) == 2.0, "c_3 = 2 exactly",
          f"got {lsv.crofton_constant(3)!r}")
    B = lsv.tangent_basis(K)
    edge = float(np.linalg.norm(B.T @ (np.eye(K)[0] - np.eye(K)[1])))
    check(np.allclose(B.T @ B, np.eye(K - 1)) and np.allclose(B.T @ np.ones(K), 0, atol=1e-12)
          and abs(edge - np.sqrt(2)) < 1e-12,
          "tangent basis orthonormal, sum-zero, edge = sqrt(2)", f"edge {edge:.12f}")

    chords, crossings, g, straddled = synthetic()
    rng = np.random.default_rng(23)
    probes = rng.dirichlet(np.ones(K), size=1200)
    m = lsv.local_sv(chords, crossings, K, h=lsv.H_DEFAULT, eval_points=probes)
    print(f"      {straddled} crossings; S_glob = {m['s_global']:.3f}; "
          f"Delta = {m['delta']:.6f}; stations = {len(m['stations']['values'])}")

    ex = m["extra"]
    dist = np.abs(np.array([g(w) for w in probes]))
    cov = ~ex["uncovered"]
    rho = float(spearmanr(ex["values"][cov], dist[cov]).statistic)
    check(rho <= -0.7, "map ranks against distance to the plane",
          f"Spearman {rho:.3f} over {int(cov.sum())}/{len(probes)} covered probes")

    near, far = dist < 0.05, dist > 0.40
    mn, mf = float(ex["values"][near].mean()), float(ex["values"][far].mean())
    check(mn > 5 * mf, "high on the plane, low away from it",
          f"mean S_hat {mn:.2f} at |d|<0.05 vs {mf:.2f} at |d|>0.40")
    check(bool(m["calibrated_ok"]) and m["n_chords"] == N_CHORDS,
          "200 chords clears the calibrated bar (>= 80)")
    check(np.all(np.isfinite(ex["values"])) and np.all(ex["values"] >= 0),
          "values finite and non-negative")

    cert = lsv.local_sv(chords, crossings, K, mode="certified", eval_points=probes)
    check(cert["n_crossings"] < m["n_crossings"]
          and float(cert["extra"]["values"][near].mean()) < mn,
          "certified mode filters the numerator only (a smaller density, not a cleaner one)",
          f"{cert['n_crossings']}/{m['n_crossings']} crossings kept")

    d = lsv.local_sv(chords, crossings, K, h=lsv.H_DEFAULT, cloud=64, seed=1)
    check(len(d["cloud"]["values"]) == 64, "cloud=64 returns 64 readings")


def part2():
    print("\n(ii) adapter on a real finished run")
    if not REAL_RUN.exists():
        check(False, "real run fixture present", f"missing {REAL_RUN}")
        return
    raw = json.loads(REAL_RUN.read_text())
    chords, crossings, k = lsv.from_run(raw)
    st = raw["status"]
    print(f"      {REAL_RUN.name}: run {st['run_id']} · {st['status']} · k={k} · "
          f"{len(chords)} chords · {len(crossings)} crossings")
    check(k == len(st["prompts"]) and len(chords) == len(st["chords"])
          and len(crossings) == len(st["crossings"]),
          "from_run reads every chord and crossing of the persisted status")
    check(all(len(a) == k and len(b) == k for a, b in chords)
          and all(abs(float(np.sum(a)) - 1.0) < 1e-6 for a, _ in chords),
          "chord endpoints are k-vectors on the simplex")
    n_cert = sum(1 for _, _, c in crossings if c)
    check(n_cert == sum(1 for x in st["crossings"] if x["significant"]),
          "certified flag = the run's own `significant`", f"{n_cert} certified")

    m = lsv.local_sv(chords, crossings, k, h=lsv.H_DEFAULT)
    xs = m["crossings"]
    print(f"      S_glob = {m['s_global']:.3f} · c_d = {m['c_d']:.4f} · "
          f"S_hat at crossings: min {xs['values'].min():.2f} / "
          f"med {np.median(xs['values']):.2f} / max {xs['values'].max():.2f}")
    check(m["n_chords"] == len(chords) and not m["calibrated_ok"],
          "a 16-chord run is NOT calibrated (ranking only)")
    check(m["s_global"] is not None and m["s_global"] > 0
          and np.all(np.isfinite(xs["values"])), "global constant and values finite")
    check(float(xs["uncovered"].mean()) < 0.10,
          "crossings sit on covered ground", f"{100 * xs['uncovered'].mean():.1f}% uncovered")
    # every crossing must be attributed to a chord it actually lies on
    A = lsv.to_tangent(np.stack([a for a, _ in chords]))
    Bt = lsv.to_tangent(np.stack([b for _, b in chords]))
    X = lsv.to_tangent(xs["weights"])
    d = [float(lsv._seg_dist(X[i:i + 1], A[j], Bt[j])[0])
         for i, j in enumerate(xs["chord"])]
    check(max(d) < 0.02, "each crossing assigned to the chord it lies on",
          f"worst distance {max(d):.2e} (bisection bracket is 0.012)")

    ranks = xs["ranks"]
    check(0.0 <= ranks.min() and ranks.max() <= 100.0
          and spearmanr(ranks, xs["values"]).statistic > 0.99,
          "percentile ranks are monotone in the values")


def part3():
    print("\n(iii) router through TestClient (stubbed app.state, no server)")
    try:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
    except Exception as exc:                     # noqa: BLE001
        check(False, "TestClient available", str(exc))
        return
    from backend.routers import cascade as router_mod
    from backend.services import cascade as cs

    raw = json.loads(REAL_RUN.read_text())["status"]
    run = cs.CascadeRun(run_id="stub", prompts=list(raw["prompts"]), seed=42, steps=8,
                        height=256, width=256, guidance_scale=3.5, n_chords=16, n_patches=4)
    run.status, run.phase = "complete", "done"
    run.chords_geo = [(c["a"], c["b"]) for c in raw["chords"]]
    run.crossings = [cs.Crossing(cid=x["cid"], wa=np.asarray(x["weights"]),
                                 wb=np.asarray(x["weights"]),
                                 mid=np.asarray(x["weights"]),
                                 significant=bool(x["significant"])) for x in raw["crossings"]]
    app = FastAPI()
    app.include_router(router_mod.router)
    app.state.cascades = {"stub": run}
    c = TestClient(app)

    r = c.get("/api/cascade/nope/local-sv")
    check(r.status_code == 404, "unknown run -> 404", f"got {r.status_code}")
    r = c.get("/api/cascade/stub/local-sv?mode=bogus")
    check(r.status_code == 400, "bad mode -> 400", f"got {r.status_code}")
    r = c.get("/api/cascade/stub/local-sv?h=0.10&mode=all&cloud=0")
    check(r.status_code == 200, "GET local-sv -> 200", f"got {r.status_code}")
    if r.status_code != 200:
        return
    j = r.json()
    check(j["n_chords"] == len(run.chords_geo) and len(j["chords"]) == len(run.chords_geo)
          and j["k"] == len(raw["prompts"]) and j["calibrated_ok"] is False,
          "payload shape matches the run",
          f"{j['n_chords']} chords, k={j['k']}, c_d={j['c_d']:.3f}")
    check(j["crossing_cids"] == [x["cid"] for x in raw["crossings"]]
          and len(j["crossing_values"]) == len(j["crossing_cids"]),
          "crossing readings align with the run's crossing ids")
    ch = j["chords"][0]
    check(len(ch["points"]) == len(ch["values"]) == len(ch["ranks"]) == len(ch["uncovered"])
          and len(ch["points"][0]) == j["k"],
          "per-chord arrays are parallel and in weight space",
          f"chord 0 has {len(ch['points'])} stations")

    empty = cs.CascadeRun(run_id="e", prompts=list(raw["prompts"]), seed=1, steps=8,
                          height=256, width=256, guidance_scale=3.5, n_chords=16, n_patches=4)
    empty.chords_geo = list(run.chords_geo)
    app.state.cascades["e"] = empty
    r = c.get("/api/cascade/e/local-sv")
    check(r.status_code == 200 and r.json()["n_chords"] == len(run.chords_geo)
          and r.json()["chords"] == [],
          "chords but no crossings -> empty map with n_chords, not a 500",
          f"got {r.status_code}")


if __name__ == "__main__":
    part1()
    part2()
    part3()
    print(f"\n{'ALL CHECKS PASS' if not fails else 'FAILED: ' + ', '.join(fails)}")
    sys.exit(1 if fails else 0)
