"""Offline checks of CLOUD mode of the final high-resolution pass (backend/services/cascade.py).
No GPU: cascade.evaluate is replaced by the band-field stub pattern of tests/hires_test.py, so the
real chord phase, the real hires phase and the real certify half run unchanged.

Bracket mode (tests/hires_test.py) spends its probes ALONG the chord, sharpening the position.
Cloud mode spends them AROUND the crossing instead: `hires_cloud_n` random points in the tangent
ball of radius `hires_cloud_r`, each labelled by the side it matches, which measures what the
chord cannot -- the local NORMAL between the two side means, a chord-free position estimate, and a
third basin (junction hint). The bracket is left as the chord phase quoted it.

  (a) SAMPLING -- _cloud_points: inside the ball, on the simplex (sum 1, min >= 0), at most n
      points, deterministic in the rng, and short rather than clipped beside a face.
  (b) CLASSIFICATION -- _cloud_side on synthetic embeddings: near side A -> "A", near side B ->
      "B", far from both -> "other" (a third basin, not the nearer stranger).
  (c) NORMAL -- _cloud_normal on a synthetic half-space split (side = sign of t.v) recovers v to
      |cos| >= 0.9 at n = 32, and answers None where a side holds fewer than 2 points.
  (d) PIPELINE -- stubbed surveys. Cloud mode fills `cloud` (fractions, normal, mid_est),
      re-quotes mid/n from it, marks hires with hires_mode "cloud", and leaves wa/wb at the
      one-stride bracket; a ball wide enough to reach a third band raises junction hints; with
      certify on the cloud normal survives bisection, which re-derives n from the bracket. Then
      the finished run goes through the real /status serialiser (the clouds, their points, and
      the untouched chords).
  (e) REQUEST -- /api/cascade/start through FastAPI's TestClient with a stubbed app.state and
      cs.run_cascade replaced by a no-op (as tests/hires_test.py does): the defaults and the
      bounds of the three new settings.

Run: python tests/hires_cloud_test.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.services import cascade as cs            # noqa: E402
from branch_test import _band_field                   # noqa: E402  (same analytic field)

fails = []


def check(ok, label, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(label)


def _e(i, n=8):
    """A unit embedding far from every other _e(j): 1-cos = 1 between any two of them."""
    v = np.zeros(n)
    v[i] = 1.0
    return v


def _tilt(a, b, t):
    """A unit embedding t of the way from _e(a) to _e(b) -- a GRADED change, so a point can be
    placed just inside or just outside the crossing threshold of a side."""
    v = (1.0 - t) * _e(a) + t * _e(b)
    return v / np.linalg.norm(v)


def _iso(k, rng):
    """A unit sum-zero direction -- the tangent space the cloud lives in."""
    u = rng.standard_normal(k)
    u -= u.mean()
    return u / np.linalg.norm(u)


def part_a():
    print("\n(a) sampling: a uniform ball in the tangent space, on the simplex")
    k, r, n = 4, 0.05, 24
    c = np.full(k, 1.0 / k)
    pts = cs._cloud_points(c, r, n, np.random.default_rng(3))
    d = [float(np.linalg.norm(p - c)) for p in pts]
    check(len(pts) == n and max(d) <= r + 1e-12,
          f"{n} points, every one within r of the centre",
          f"{len(pts)} points, max |p-c| {max(d):.4f} vs r {r}")
    sums = [abs(float(np.sum(p)) - 1.0) for p in pts]
    check(max(sums) < 1e-12 and min(float(np.min(p)) for p in pts) >= 0.0,
          "every point is a recipe: weights sum to 1 and none is negative",
          f"max |sum-1| {max(sums):.2e}, min weight "
          f"{min(float(np.min(p)) for p in pts):.4f}")
    # the offsets fill the ball rather than hugging its surface (the U^(1/(k-1)) radius): with a
    # uniform radius the mean would sit at r/2, with the correct one at r*(k-1)/k
    check(abs(float(np.mean(d)) - r * (k - 1) / k) < 0.1 * r,
          "radii follow the ball's own radial density, not a uniform radius",
          f"mean |p-c| {float(np.mean(d)):.4f} vs r(k-1)/k {r * (k - 1) / k:.4f}")
    again = cs._cloud_points(c, r, n, np.random.default_rng(3))
    check(len(again) == len(pts) and all(np.allclose(a, b) for a, b in zip(pts, again)),
          "the same seed draws the same cloud")
    other = cs._cloud_points(c, r, n, np.random.default_rng(4))
    check(not all(np.allclose(a, b) for a, b in zip(pts, other)),
          "a different seed draws a different cloud (the check above is not vacuous)")
    # beside a face most of the ball lies outside the simplex: the pass keeps what it gets
    near = np.array([0.94, 0.02, 0.02, 0.02])
    few = cs._cloud_points(near, 0.25, n, np.random.default_rng(7))
    check(len(few) < n and all(float(np.min(p)) >= 0.0 for p in few)
          and all(abs(float(np.sum(p)) - 1.0) < 1e-12 for p in few),
          "beside a face the cloud comes out SHORT, never clipped onto the face",
          f"{len(few)} of {n} points")


def part_b():
    print("\n(b) classification: the side a cloud image matched, or a third basin")
    ea, eb = _e(0), _e(1)
    check(cs._cloud_side(_tilt(0, 1, 0.05), ea, eb) == "A", "an image near side A reads 'A'")
    check(cs._cloud_side(_tilt(1, 0, 0.05), ea, eb) == "B", "an image near side B reads 'B'")
    far = _e(2)
    das = (cs._cosd(far, ea), cs._cosd(far, eb))
    check(min(das) > cs.COS_T and cs._cloud_side(far, ea, eb) == "other",
          "an image far from BOTH sides reads 'other' -- a third basin, not the nearer stranger",
          f"1-cos to the sides {das[0]:.2f} / {das[1]:.2f} vs COS_T {cs.COS_T}")
    # just inside the threshold of one side: a side, not a junction
    e = _tilt(0, 2, 0.5)
    check(cs._cosd(e, ea) < cs.COS_T and cs._cloud_side(e, ea, eb) == "A",
          "an image still within COS_T of side A is side A, however far side B is",
          f"1-cos to A {cs._cosd(e, ea):.2f}")


def part_c():
    print("\n(c) normal: the direction between the two side means")
    rng = np.random.default_rng(11)
    k, n, r = 4, 32, 0.05
    c = np.full(k, 1.0 / k)
    v = _iso(k, rng)
    pts = cs._cloud_points(c, r, n, rng)
    sides = ["B" if float(np.dot(p - c, v)) > 0 else "A" for p in pts]
    nrm, mid = cs._cloud_normal(pts, sides)
    cos = abs(float(np.dot(nrm, v))) if nrm is not None else 0.0
    check(nrm is not None and cos >= 0.9,
          "a half-space split recovers the splitting direction (|cos| >= 0.9 at n=32)",
          f"|cos| {cos:.3f}, sides {sides.count('A')}/{sides.count('B')}")
    check(nrm is not None and abs(float(np.linalg.norm(nrm)) - 1.0) < 1e-12
          and abs(float(np.sum(nrm))) < 1e-12,
          "the normal is a unit SUM-ZERO direction, i.e. a tangent of the simplex")
    check(mid is not None and abs(float(np.sum(mid)) - 1.0) < 1e-12
          and float(np.linalg.norm(mid - c)) <= r + 1e-12,
          "the midpoint estimate is a recipe inside the ball",
          f"|mid-c| {float(np.linalg.norm(mid - c)):.4f} vs r {r}")
    # "other" points belong to no side: they must not drag either mean
    nrm2, _m2 = cs._cloud_normal(pts + [c + 0.9 * r * v], sides + ["other"])
    check(nrm2 is not None and float(np.dot(nrm2, nrm)) > 1 - 1e-12,
          "a third-basin point enters neither mean, so the normal does not move")
    one = cs._cloud_normal([c - 0.01 * v, c - 0.02 * v, c + 0.01 * v], ["A", "A", "B"])
    check(one == (None, None), "one point on a side is no evidence of a direction: None")
    check(cs._cloud_normal([], []) == (None, None), "an empty cloud: None")


class _StopAt(Exception):
    """Raised by the stub at a chosen round: the run object already carries what is checked."""


def _stub_evaluate(run, field, rounds, stop_at=None):
    """Stand-in for cascade.evaluate: embeds every weight vector through the analytic field and
    records (label, weights, steps) per GPU round, optionally stopping at a round label."""
    def stub(app, r, pool, weights, seed, label, ctl=None, on_arrival=None, steps=None):
        base = label.split(".")[0]
        rounds.append((base, [np.asarray(w, dtype=float) for w in weights], steps))
        if stop_at is not None and base.startswith(stop_at):
            raise _StopAt()
        idxs = []
        for w in weights:
            gi = run.next_idx
            run.next_idx += 1
            run.embeddings[gi] = field(w)
            run.generated += 1
            idxs.append(gi)
        for li, gi in enumerate(idxs):
            if on_arrival is not None:
                on_arrival(li, gi)
        return idxs
    return stub


def _survey(k=4, n_chords=4, seed=11, certify=False, probe_steps=4, hires=False,
            hires_mode="cloud", hires_cloud_n=12, hires_cloud_r=0.05, hires_top_pct=20,
            period=None, stop_at=None):
    """One stubbed survey; returns (run, [(round label, weights, steps)])."""
    run = cs.CascadeRun(
        run_id="hc", prompts=[f"p{i}" for i in range(k)], seed=seed, steps=8, height=64,
        width=64, guidance_scale=3.5, n_chords=n_chords, n_patches=2,
        probe_steps=probe_steps, certify=certify,
        hires=hires, hires_mode=hires_mode, hires_cloud_n=hires_cloud_n,
        hires_cloud_r=hires_cloud_r, hires_top_pct=hires_top_pct)
    rounds = []
    field = _band_field(k) if period is None else _band_field(k, period)
    orig = cs.evaluate
    cs.evaluate = _stub_evaluate(run, field, rounds, stop_at=stop_at)
    try:
        cs.run_cascade(None, run, None)
    except _StopAt:
        pass
    finally:
        cs.evaluate = orig
    return run, rounds


def part_d():
    print("\n(d) the cloud inside the real loop (stubbed field)")
    base, _br = _survey()
    run, rounds = _survey(hires=True)
    labels = [l for l, _w, _s in rounds]
    hr = [(w, s) for l, w, s in rounds if l == "hires"]
    sel = cs._hires_select(base.crossings, 20)
    check(labels.count("hires") == 1
          and labels.index("hires") == max(i for i, l in enumerate(labels)
                                           if l.startswith("chords")) + 1,
          "one cloud round, after every chord generation", f"rounds {labels}")
    check(hr and hr[0][1] == run.probe_steps,
          "the cloud renders on the cheap field (probe_steps), like the chord probes",
          f"steps {hr[0][1] if hr else None} vs probe_steps {run.probe_steps}")
    check(hr and len(hr[0][0]) <= len(sel) * run.hires_cloud_n
          and len(hr[0][0]) >= len(sel) * (run.hires_cloud_n - 2),
          "the round holds up to hires_cloud_n points per selected crossing",
          f"{len(hr[0][0]) if hr else 0} points for {len(sel)} of "
          f"{len(base.crossings)} crossings")

    clouded = [x for x in run.crossings if x.cloud is not None]
    check(len(clouded) == len(sel)
          and all(x.hires and x.hires_mode == "cloud" for x in clouded),
          "exactly the selected crossings carry a cloud, marked hires / hires_mode 'cloud'",
          f"{len(clouded)} clouded of {len(run.crossings)}")
    check(len(run.crossings) == len(base.crossings)
          and all(x.hires_width is None and x.split_from is None and x.hires_note is None
                  for x in run.crossings),
          "a cloud neither splits nor re-quotes a bracket: no new crossing, no hires_width",
          f"{len(run.crossings)} vs {len(base.crossings)} crossings")
    by_cid = {x.cid: x for x in base.crossings}
    kept = all(np.allclose(x.wa, by_cid[x.cid].wa) and np.allclose(x.wb, by_cid[x.cid].wb)
               for x in run.crossings)
    check(kept, "wa/wb stay exactly where the chord phase left them")
    shapes = all(
        set(x.cloud) == {"n", "r", "frac_a", "frac_b", "frac_other", "normal", "mid_est",
                         "junction_hint"}
        and x.cloud["r"] == run.hires_cloud_r
        and x.cloud["n"] == len(x.cloud_pts)
        and abs(x.cloud["frac_a"] + x.cloud["frac_b"] + x.cloud["frac_other"] - 1.0) < 1e-9
        for x in clouded)
    check(shapes, "each cloud reports n/r, three side fractions summing to 1, and its points",
          f"{clouded[0].cloud['frac_a']:.2f}/{clouded[0].cloud['frac_b']:.2f}/"
          f"{clouded[0].cloud['frac_other']:.2f}" if clouded else "none")
    pts_ok = all(len(p) == 3 and len(p[0]) == run.k and p[1] in ("A", "B", "other")
                 and isinstance(p[2], int) and p[2] in run.embeddings
                 for x in clouded for p in x.cloud_pts)
    check(pts_ok, "every stored point is [[weights...], side, image index] of a rendered image")
    with_n = [x for x in clouded if x.cloud["normal"] is not None]
    check(len(with_n) > 0
          and all(np.allclose(x.n, x.cloud["normal"]) for x in with_n)
          and all(np.allclose(x.mid, x.cloud["mid_est"]) for x in with_n),
          "where the cloud named a normal, the crossing's n and mid ARE the cloud's",
          f"{len(with_n)} of {len(clouded)} clouds named a normal")
    no_n = [x for x in clouded if x.cloud["normal"] is None]
    check(all(np.allclose(x.mid, (x.wa + x.wb) / 2) for x in no_n),
          "a cloud with too few points on a side leaves the bracket centre as the position",
          f"{len(no_n)} such clouds")
    # the field's boundaries are flat sheets normal to one fixed sum-zero direction (branch_test's
    # _band_field), so the cloud's normals must line up with it -- the chord's never do
    bn = np.arange(1, run.k + 1, dtype=float)
    bn -= bn.mean()
    bn /= np.linalg.norm(bn)
    coss = [abs(float(np.dot(x.cloud["normal"], bn))) for x in with_n]
    check(coss and float(np.median(coss)) > 0.9,
          "on a field of flat bands the cloud normals line up with the band normal",
          f"median |cos| {float(np.median(coss)):.3f} over {len(coss)} normals")
    note = next((n for n in run.notes if n.startswith("hires cloud:")), "MISSING")
    check("crossings" in note and "pts @" in note and "normals for" in note
          and "junction hints" in note and "median |cos(normal, chord dir)|" in note,
          "the summary note reports the count, the ball, the normals and the junction hints",
          note)
    check(not any(n.startswith("hires: refined") for n in run.notes),
          "cloud mode does not also run the bracket sweep")

    # a ball wide enough to reach PAST the neighbouring band sees a third basin
    run2, _r2 = _survey(hires=True, hires_cloud_n=24, hires_cloud_r=0.09, period=0.03)
    hints = [x for x in run2.crossings if x.cloud and x.cloud["junction_hint"]]
    check(len(hints) > 0 and all(x.cloud["frac_other"] > 0 for x in hints),
          "a third basin inside the ball is flagged as a junction hint",
          f"{len(hints)} hints of {sum(1 for x in run2.crossings if x.cloud)} clouds")

    # certify on: bisection re-derives n from the bracket it narrowed -- the cloud takes it back
    run3, rounds3 = _survey(certify=True, hires=True)
    labels3 = [l for l, _w, _s in rounds3]
    cl3 = [x for x in run3.crossings if x.cloud and x.cloud["normal"] is not None]
    check(any(l.startswith("bisect") for l in labels3) and len(cl3) > 0,
          "control: the certify half really ran, with clouded crossings in it",
          f"{len(cl3)} clouded normals, rounds {sorted(set(labels3))}")
    check(all(np.allclose(x.n, x.cloud["normal"]) for x in cl3)
          and all(np.allclose(x.mid, x.cloud["mid_est"]) for x in cl3),
          "the cloud normal (and position) survive bisection",
          f"{len(cl3)} crossings")
    narrowed = [x for x in cl3 if float(np.linalg.norm(x.wa - x.wb)) <= cs.BRACKET]
    chord_n = []
    for x in narrowed:
        d = x.wb - x.wa
        chord_n.append(abs(float(np.dot(x.cloud["normal"], d / np.linalg.norm(d)))))
    check(len(narrowed) > 0 and any(c < 0.999 for c in chord_n),
          "bisection did narrow those brackets, and the kept normal is not just the chord's",
          f"{len(narrowed)} pinned, max |cos| with the bracket chord "
          f"{max(chord_n) if chord_n else float('nan'):.3f}")
    check(all(x.b is not None for x in run3.crossings) and bool(run3.patches),
          "the rest of the certify half ran on the cloud geometry (scored, patched)",
          f"{len(run3.patches)} patches")
    return run


def part_d2(run):
    print("\n(d2) the finished cloud run through /status (the real serialiser)")
    try:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
    except Exception as exc:                     # noqa: BLE001
        check(False, "TestClient available", str(exc))
        return
    from backend.routers import cascade as router_mod

    app = FastAPI()
    app.include_router(router_mod.router)
    app.state.cascades = {"hc": run}
    app.state.cache = object()
    app.state.gpu_pool = None
    r = TestClient(app, raise_server_exceptions=False).get("/api/cascade/hc/status")
    j = r.json() if r.status_code == 200 else {}
    xs = [x for x in j.get("crossings", []) if x.get("cloud")]
    check(r.status_code == 200 and len(xs) == sum(1 for x in run.crossings if x.cloud),
          "status serialises every cloud (the dict matches the CascadeCloud model)",
          f"HTTP {r.status_code}, {len(xs)} clouds")
    ok = all(x["hires_mode"] == "cloud" and x["hires"] is True
             and set(x["cloud"]) >= {"n", "r", "frac_a", "frac_b", "frac_other", "normal",
                                     "mid_est", "junction_hint"}
             and len(x["cloud_pts"]) == x["cloud"]["n"]
             and all(len(p) == 3 and p[1] in ("A", "B", "other") for p in x["cloud_pts"])
             for x in xs)
    check(bool(xs) and ok, "each served crossing carries hires_mode, the summary and its points")
    wmid = all(np.allclose(x["weights"], x["cloud"]["mid_est"])
               for x in xs if x["cloud"]["normal"] is not None)
    check(wmid, "the served position of a clouded crossing IS the cloud's estimate")
    check(len(j.get("chords", [])) == len(run.chords_geo)
          and len(j.get("chords_meta", [])) == len(run.chords_geo)
          and j.get("hires_mode") == "cloud",
          "chords/chords_meta are untouched (a cloud is not a chord)",
          f"{len(j.get('chords', []))} chords")


def part_e():
    print("\n(e) /start and /status through TestClient (stubbed state, run_cascade no-op)")
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
        got = (run.hires_mode, run.hires_cloud_n, run.hires_cloud_r) if run else None
        check(r.status_code == 200 and got == ("bracket", 12, 0.05),
              "omitted cloud settings default to bracket mode, 12 points, r 0.05",
              f"got {got!r}")
        if rid:
            j = c.get(f"/api/cascade/{rid}/status").json()
            check(j["hires_mode"] == "bracket" and j["hires_cloud_n"] == 12
                  and j["hires_cloud_r"] == 0.05,
                  "status reports the run's own cloud settings")

        r = c.post("/api/cascade/start",
                   json=dict(body, hires=True, hires_mode="cloud", hires_cloud_n=32,
                             hires_cloud_r=0.12))
        rid = r.json().get("run_id", "")
        run = app.state.cascades.get(rid)
        got = (run.hires, run.hires_mode, run.hires_cloud_n, run.hires_cloud_r) if run else None
        check(r.status_code == 200 and got == (True, "cloud", 32, 0.12),
              "cloud mode with its two settings is threaded into the run", f"got {got!r}")
        if rid:
            j = c.get(f"/api/cascade/{rid}/status").json()
            check(j["hires_mode"] == "cloud" and j["hires_cloud_n"] == 32
                  and j["hires_cloud_r"] == 0.12,
                  "status reports a run started in cloud mode")

        for field, bad in (("hires_mode", "foo"), ("hires_cloud_n", 3),
                           ("hires_cloud_n", 65), ("hires_cloud_r", 0.5),
                           ("hires_cloud_r", 0.005)):
            r = c.post("/api/cascade/start", json=dict(body, **{field: bad}))
            check(r.status_code == 422, f"{field}={bad!r} rejected",
                  f"got HTTP {r.status_code}")
    finally:
        cs.run_cascade = orig


if __name__ == "__main__":
    part_a()
    part_b()
    part_c()
    part_d2(part_d())
    part_e()
    print(f"\n{'ALL CHECKS PASS' if not fails else 'FAILED: ' + ', '.join(fails)}")
    sys.exit(1 if fails else 0)
