"""Offline checks of recursive ("branching") chord exploration (backend/services/cascade.py).
No GPU: cascade.evaluate is replaced by a stub that embeds each weight vector through an
analytic field, so the real generational loop, detection and cap run unchanged.

The field is a stack of parallel BANDS in one fixed sum-zero direction: consecutive probes in
the same band are identical images (1-cos = 0), probes in neighbouring bands differ far above
COS_T, and the contrast varies band to band so "strongest crossing" is not a tie.

  (i)   BRANCHING on that field: child rays start on a crossing in the previous generation's
        top-divergence selection (_branch_origins; tests/branch_top_test.py covers the rule
        itself) and skip the probe0-probe1 bracket (so the origin's own sheet is not
        re-counted), carry gen/parent/origin_cid, and never exceed the geometric chord cap --
        checked for (branch, depth) in {(2,2), (3,1), (1,3)}.
  (ii)  DETERMINISM: with branch=0 the probe list is bit-identical to the pre-refactor chord
        phase, replayed here from the same helpers and rng stream; and turning branching on
        does not disturb generation 0.
  (iii) REQUEST/STATUS through FastAPI's TestClient with a stubbed app.state and
        cs.run_cascade replaced by a no-op (as tests/stride_test.py does): bounds on branch
        and depth, the defaults, chords_meta, and the roots-only coverage certificate.

Run: python tests/branch_test.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.services import cascade as cs            # noqa: E402

CONFIGS = [(2, 2), (3, 1), (1, 3)]
PERIOD = 0.06        # band width: >= 2 strides, so a bracket holds at most one sheet
NDIM = 32            # bands before the one-hot index wraps (a chord spans far fewer)

fails = []


def check(ok, label, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(label)


class _StopAfterChords(Exception):
    """Raised by the stub once the chord phase is over: bisection and scoring are not under
    test here and the run object already carries everything the checks read."""


def _band_field(k, period=PERIOD):
    """Analytic image field: parallel bands along one fixed sum-zero direction.

    Band b embeds as a one-hot at b plus a small graded tilt onto b+1, so neighbouring bands
    sit far apart (1-cos ~ 0.8) while the exact contrast depends on b -- the strongest-crossing
    pick would be untestable on a field where every crossing scores the same.
    """
    n = np.arange(1, k + 1, dtype=float)
    n -= n.mean()
    n /= np.linalg.norm(n)

    def field(w):
        b = int(np.floor(float(np.dot(np.asarray(w, dtype=float), n)) / period))
        e = np.zeros(NDIM)
        e[b % NDIM] = 1.0
        e[(b + 1) % NDIM] = 0.15 + 0.02 * (b % 5)
        return e / np.linalg.norm(e)
    return field


def _stub_evaluate(run, field, probes):
    """Stand-in for cascade.evaluate: embeds every weight vector, files the probes of each
    chord generation under its label, and stops the run after the chord phase."""
    def stub(app, r, pool, weights, seed, label, ctl=None, on_arrival=None, steps=None):
        if not label.startswith("chords"):
            raise _StopAfterChords()
        probes.setdefault(label.split(".")[0], []).extend(
            np.asarray(w, dtype=float) for w in weights)
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


def _survey(k=4, n_chords=4, branch=0, depth=0, seed=11, stride=cs.STRIDE):
    """One stubbed survey; returns (run, {generation label: probe list})."""
    run = cs.CascadeRun(
        run_id="t", prompts=[f"p{i}" for i in range(k)], seed=seed, steps=8, height=64,
        width=64, guidance_scale=3.5, n_chords=n_chords, n_patches=2, stride=stride,
        branch=branch, depth=depth, probe_steps=None,   # None skips the rebracket pass
        certify=True)   # the full pipeline, so the stub still stops the run AT the bisection
    probes = {}
    orig = cs.evaluate
    cs.evaluate = _stub_evaluate(run, _band_field(k), probes)
    try:
        cs.run_cascade(None, run, None)
    except _StopAfterChords:
        pass
    finally:
        cs.evaluate = orig
    return run, probes


def _gen_probes(run, probes, g):
    """The recorded probe list of generation g, in the order it was submitted."""
    return probes.get("chords" if g == 0 else f"chords{g}", [])


def _first_brackets(run, gen_probes, g):
    """(chord index, start weights, second probe) per chord of generation g, read out of the
    recorded probe list -- no reconstruction, so no chance of an fp band flip."""
    out, pos = [], 0
    for ci, m in enumerate(run.chords_meta):
        if m["gen"] != g:
            continue
        a = np.asarray(run.chords_geo[ci][0], dtype=float)
        while pos < len(gen_probes) and not np.array_equal(gen_probes[pos], a):
            pos += 1
        if pos + 1 < len(gen_probes):
            out.append((ci, a, gen_probes[pos + 1]))
        pos += 1
    return out


def _on_chord(run, ci, w):
    """Does weight vector w lie on chord ci? (used to attribute crossings to parents)"""
    a = np.asarray(run.chords_geo[ci][0], dtype=float)
    b = np.asarray(run.chords_geo[ci][1], dtype=float)
    u = b - a
    nn = np.linalg.norm(u)
    if nn < 1e-12:
        return False
    u = u / nn
    d = np.asarray(w, dtype=float) - a
    return float(np.linalg.norm(d - np.dot(d, u) * u)) < 1e-9


def part1():
    print("\n(i) branching on the band field: first-bracket skip, provenance, cap")
    field = _band_field(4)
    run, probes = _survey(branch=2, depth=2)
    gens = [m["gen"] for m in run.chords_meta]
    check(len(run.chords_geo) == len(run.chords_meta) == len(gens) and max(gens) == 2,
          "branch=2 depth=2 produced three generations of chords",
          f"gen counts {[gens.count(g) for g in range(3)]}, {len(run.crossings)} crossings")

    # the skip: no crossing may be quoted at a child chord's FIRST bracket, and the test is
    # only meaningful if some of those first brackets really do straddle a sheet
    straddling, offending = 0, 0
    for g in (1, 2):
        for ci, a, p1 in _first_brackets(run, _gen_probes(run, probes, g), g):
            if cs._cosd(field(a), field(p1)) > cs.COS_T:
                straddling += 1
            if any(np.allclose(x.wa, a, atol=0, rtol=0) for x in run.crossings):
                offending += 1
    check(straddling > 0, "child rays do start on a sheet (the skip is not vacuous)",
          f"{straddling} first brackets straddle a band boundary")
    check(offending == 0, "no crossing is quoted at a child ray's first bracket",
          f"{offending} offending crossings")

    # provenance: a child's origin is the MIDPOINT of the bracket it was spawned from, and that
    # crossing lies on the parent chord one generation up
    check(all(m["parent"] == -1 and m["origin_cid"] == -1
              for m in run.chords_meta if m["gen"] == 0),
          "root chords carry parent -1 / origin -1")
    bad_parent, bad_origin, bad_chord = 0, 0, 0
    by_cid = {x.cid: x for x in run.crossings}
    for ci, m in enumerate(run.chords_meta):
        if m["gen"] == 0:
            continue
        p, x = m["parent"], by_cid.get(m["origin_cid"])
        if not (0 <= p < len(run.chords_meta)
                and run.chords_meta[p]["gen"] == m["gen"] - 1):
            bad_parent += 1
            continue
        if x is None or x.gen != m["gen"] - 1 or not np.allclose(
                np.asarray(run.chords_geo[ci][0], dtype=float), (x.wa + x.wb) / 2):
            bad_origin += 1
            continue
        if not _on_chord(run, p, (x.wa + x.wb) / 2):
            bad_chord += 1
    check(bad_parent == 0, "every child names a parent chord one generation up")
    check(bad_origin == 0, "every child ray starts at its origin crossing's bracket midpoint")
    check(bad_chord == 0, "the origin crossing sits on the parent chord it is attributed to")

    # the origins of a generation are exactly the top branch_top_pct % of the PREVIOUS
    # generation's crossings by divergence -- not one per crossing-bearing chord (the rule
    # this replaced), so they need not be spread over the chords at all
    bad_sel, detail = [], []
    for g in (1, 2):
        prev = [y for y in run.crossings if y.gen == g - 1]
        want = {x.cid for x in cs._branch_origins(prev, run.branch_top_pct)}
        got = {m["origin_cid"] for m in run.chords_meta if m["gen"] == g}
        detail.append(f"gen {g}: {len(got)} of {len(want)} selected from {len(prev)}")
        if not got or not got <= want:
            bad_sel.append(g)
    check(not bad_sel,
          f"origins come from the previous generation's top {run.branch_top_pct} % "
          "by divergence", "; ".join(detail))

    # crossings inherit their chord's generation, and the roots-only filter keeps a subset
    gset = sorted({x.gen for x in run.crossings})
    n_root = sum(1 for x in run.crossings if x.gen == 0)
    check(gset == [0, 1, 2] and 0 < n_root < len(run.crossings),
          "crossings carry their chord's generation",
          f"gens {gset}, {n_root}/{len(run.crossings)} on roots")

    # the geometric cap, and the per-generation growth it comes from
    for b, d in CONFIGS:
        cap = cs._chord_cap(4, b, d)
        want = 4 * sum(b ** i for i in range(d + 1))
        r, _ = _survey(branch=b, depth=d)
        gs = [m["gen"] for m in r.chords_meta]
        cnt = [gs.count(g) for g in range(d + 1)]
        grew = all(cnt[g] <= b * cnt[g - 1] for g in range(1, d + 1))
        check(cap == want and len(r.chords_geo) <= cap and cnt[0] <= 4 and grew
              and max(gs) <= d,
              f"branch={b} depth={d}: cap {cap} respected, growth <= branch per generation",
              f"{len(r.chords_geo)} chords {cnt}")

    check(cs._chord_cap(24, 0, 3) == 24 and cs._chord_cap(24, 1, 3) == 96
          and cs._chord_cap(24, 2, 2) == 168,
          "cap formula: branch=0 is one generation, branch=1 is n*(depth+1)")

    notes = " | ".join(run.notes)
    check(any(n.startswith("gen 0:") for n in run.notes)
          and any(n.startswith("gen 2:") for n in run.notes)
          and "roots" in notes,
          "notes carry a line per generation and say the certificate is roots-only")


def _replay_roots(k, n_chords, seed, stride):
    """The chord phase as it stood BEFORE the generational refactor, replayed from the same
    helpers and the same rng stream. Any drift in the root draw shows up as a probe mismatch."""
    rng = np.random.default_rng(seed)
    chords, pts, anchors = [], [], []
    dirs = cs._dir_batch(k, rng, n_chords)
    tries, max_tries = 0, 200 * n_chords
    while len(chords) < n_chords and tries < max_tries:
        tries += 1
        w0 = cs._blue_anchor(k, rng, anchors, draw=lambda: rng.dirichlet(np.ones(k)))
        u = dirs[len(chords)] if len(chords) < len(dirs) else cs._iso_dir(k, rng)
        tneg, tpos = cs._extent(w0, u)
        if tpos - tneg < cs._min_chord_len(stride):
            continue
        offs = cs._chord_offsets(tneg, tpos, stride)
        chords.append((w0, u, tneg, tpos))
        anchors.append(w0)
        for off in offs:
            pts.append(np.clip(w0 + off * u, 0, None))
    return chords, pts


def part2():
    print("\n(ii) determinism: branch=0 reproduces the pre-refactor chord phase")
    for k, n_chords, stride in ((4, 6, cs.STRIDE), (3, 5, 0.05)):
        run, probes = _survey(k=k, n_chords=n_chords, branch=0, depth=0, seed=11,
                              stride=stride)
        got = _gen_probes(run, probes, 0)
        old_chords, want = _replay_roots(k, n_chords, 11, stride)
        same = (len(got) == len(want)
                and all(np.array_equal(a, b) for a, b in zip(got, want)))
        check(same, f"k={k} stride={stride}: probe list is bit-identical to the old phase",
              f"{len(got)} probes over {len(run.chords_geo)} chords")
        geo_same = len(run.chords_geo) == len(old_chords) and all(
            np.array_equal(np.asarray(run.chords_geo[i][0]),
                           np.clip(w0 + tneg * u, 0, None))
            and np.array_equal(np.asarray(run.chords_geo[i][1]),
                               np.clip(w0 + tpos * u, 0, None))
            for i, (w0, u, tneg, tpos) in enumerate(old_chords))
        check(geo_same, f"k={k} stride={stride}: chord endpoints unchanged")
        check(all(m["gen"] == 0 and m["parent"] == -1 for m in run.chords_meta)
              and all(x.gen == 0 for x in run.crossings),
              f"k={k} stride={stride}: everything is generation 0 with branching off")
        check(any(n == f"{len(run.chords_geo)} chords, {len(run.crossings)} crossings"
                  for n in run.notes),
              f"k={k} stride={stride}: the note keeps its pre-refactor wording")

    # branching must not disturb the fair survey it grows out of
    base, pb = _survey(k=4, n_chords=6, branch=0, depth=0, seed=11)
    br, pr = _survey(k=4, n_chords=6, branch=3, depth=1, seed=11)
    p0b, p0r = _gen_probes(base, pb, 0), _gen_probes(br, pr, 0)
    check(len(p0b) == len(p0r) and all(np.array_equal(a, b) for a, b in zip(p0b, p0r)),
          "generation 0 is identical with branching on and off (same seed)",
          f"{len(p0b)} root probes")
    nb = len(base.crossings)
    check([x.cid for x in base.crossings] == [x.cid for x in br.crossings[:nb]]
          and all(np.array_equal(a.wa, b.wa) and np.array_equal(a.wb, b.wb)
                  for a, b in zip(base.crossings, br.crossings[:nb])),
          "root crossings keep their ids and brackets when children are added",
          f"{nb} root crossings of {len(br.crossings)}")


def _mixed_run(branch):
    """A finished run whose crossings straddle two generations. Roots: 4 crossings over 3
    ridges, 2 of them singletons (share 2/4). All six: 4 ridges, 2 singletons (share 2/6).
    The two certificates therefore disagree in every field, so the filter cannot pass by
    coincidence."""
    run = cs.CascadeRun(run_id="m", prompts=["a", "b", "c"], seed=1, steps=8, height=64,
                        width=64, guidance_scale=3.5, n_chords=4, n_patches=2,
                        branch=branch, depth=1 if branch else 0)
    run.status, run.phase = "complete", "done"
    run.chords_geo = [([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])]
    run.chords_meta = [{"gen": 0, "parent": -1, "origin_cid": -1}]

    def e(i):
        v = np.zeros(8)
        v[i] = 1.0
        return v
    sides = [(0, 1), (0, 1), (2, 3), (4, 5), (6, 7), (6, 7)]
    for cid, (ia, ib) in enumerate(sides):
        w = np.array([0.2 + 0.05 * cid, 0.4, 0.4 - 0.05 * cid])
        run.crossings.append(cs.Crossing(cid=cid, wa=w, wb=w, mid=w, ea=e(ia), eb=e(ib),
                                         gen=0 if cid < 4 else 1))
    return run


def part3():
    print("\n(iii) /start and /status through TestClient (stubbed state, run_cascade no-op)")
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

        r = c.post("/api/cascade/start", json=dict(body, branch=2, depth=2))
        rid = r.json().get("run_id", "")
        run = app.state.cascades.get(rid)
        check(r.status_code == 200 and run is not None and run.branch == 2 and run.depth == 2,
              "branch=2 depth=2 accepted and threaded into the run", f"HTTP {r.status_code}")
        if rid:
            j = c.get(f"/api/cascade/{rid}/status").json()
            check(j["chords_meta"] == [] and j["stats_scope"] == "roots",
                  "status exposes chords_meta and a roots-only stats_scope",
                  f"stats_scope {j['stats_scope']!r}")

        r = c.post("/api/cascade/start", json=body)
        rid = r.json().get("run_id", "")
        run = app.state.cascades.get(rid)
        got = (run.branch, run.depth) if run else None
        check(r.status_code == 200 and got == (0, 0),
              "omitted branch/depth default to 0/0", f"got {got!r}")
        if rid:
            check(c.get(f"/api/cascade/{rid}/status").json()["stats_scope"] == "all",
                  "without branching the certificate is quoted over all crossings")

        for field, bad in (("branch", 7), ("depth", 5), ("branch", -1), ("depth", -1)):
            r = c.post("/api/cascade/start", json=dict(body, **{field: bad}))
            check(r.status_code == 422, f"{field}={bad} rejected (out of range)",
                  f"got HTTP {r.status_code}")

        # the roots-only certificate itself, on a run with crossings in two generations
        app.state.cascades["mix"] = _mixed_run(branch=2)
        app.state.cascades["flat"] = _mixed_run(branch=0)
        j = c.get("/api/cascade/mix/status").json()
        jf = c.get("/api/cascade/flat/status").json()
        check(j["stats_scope"] == "roots" and j["distinct_ridges"] == 3
              and j["singleton_ridges"] == 2 and abs(j["unexplored_share"] - 2 / 4) < 1e-9,
              "certificate counts the 4 root crossings only",
              f"{j['distinct_ridges']} ridges, share {j['unexplored_share']:.3f}")
        check(jf["stats_scope"] == "all" and jf["distinct_ridges"] == 4
              and jf["singleton_ridges"] == 2
              and abs(jf["unexplored_share"] - 2 / 6) < 1e-9,
              "the same crossings without branching give the all-crossings certificate",
              f"{jf['distinct_ridges']} ridges, share {jf['unexplored_share']:.3f}")
        groups = [x["ridge_group"] for x in j["crossings"]]
        check(all(g is not None for g in groups) and len(set(groups)) == 4
              and [x["gen"] for x in j["crossings"]] == [0, 0, 0, 0, 1, 1],
              "ridge identities still cover every generation, and gen is reported",
              f"groups {groups}")
    finally:
        cs.run_cascade = orig


if __name__ == "__main__":
    part1()
    part2()
    part3()
    print(f"\n{'ALL CHECKS PASS' if not fails else 'FAILED: ' + ', '.join(fails)}")
    sys.exit(1 if fails else 0)
