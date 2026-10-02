"""Offline checks of the ORIGIN RULE for recursive ("branching") chords
(backend/services/cascade.py, _branch_origins). No GPU: cascade.evaluate is replaced by the
band-field stub of tests/branch_test.py, so the real generational loop runs unchanged.

Generation g grows out of the crossings of generation g-1 whose probe-to-probe divergence is in
the top `branch_top_pct` % -- max(1, ceil(pct/100 * n)) of them, strongest first -- and each of
those spawns `branch` rays. This replaced the "one origin per crossing-bearing chord" rule of
733f3d1: a chord is an accident of the survey geometry, so spreading the branching budget over
chords spent it evenly on evidence of very different strength. One chord may now contribute
several origins, or none.

  (i)   PURE -- _branch_origins on 10 synthetic crossings of known, distinct divergences, handed
        in scrambled order: pct=20 keeps the top 2 strongest-first, pct=100 all ten, pct=1 the
        single strongest, an empty generation nothing, and n=3 at pct=20 one. Ties keep detection
        order and the input list is left alone.
  (ii)  LOOP -- stubbed surveys with the minimum-length rule neutralised, so every selected
        origin MUST contribute exactly `branch` rays until the geometric cap bites: the chord
        count per generation, the fan size per origin including the fan the cap cuts short, and
        the per-generation note. pct=100 is run as the control (every crossing spawns).
  (iii) REQUEST/STATUS -- branch_top_pct through FastAPI's TestClient with a stubbed app.state
        and cs.run_cascade replaced by a no-op (as tests/branch_test.py does): 50 accepted and
        threaded into the run, 0 and 101 rejected, the default is 20, and /status echoes the
        branching settings the run was started with.

Run: python tests/branch_top_test.py
"""
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.services import cascade as cs            # noqa: E402
from branch_test import (_StopAfterChords, _band_field,   # noqa: E402  (same analytic field)
                         _stub_evaluate)

fails = []


def check(ok, label, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(label)


def _xing(cid, div):
    """A crossing whose only measured property is its divergence: sides one apart by 1-cos=div."""
    ea = np.array([1.0, 0.0])
    eb = np.array([1.0 - div, math.sqrt(max(0.0, 1.0 - (1.0 - div) ** 2))])
    w = np.zeros(3)
    return cs.Crossing(cid=cid, wa=w, wb=w, ea=ea, eb=eb)


def part1():
    print("\n(i) _branch_origins as a pure function")
    divs = [0.05 * (i + 1) for i in range(10)]        # 0.05 .. 0.50, all distinct
    # handed in scrambled order, so "strongest first" cannot pass by echoing the input
    order = [4, 9, 0, 7, 2, 8, 5, 1, 6, 3]
    xs = [_xing(cid, divs[cid]) for cid in order]
    got = [round(cs._cosd(x.ea, x.eb), 6) for x in xs]
    check(got == [round(divs[cid], 6) for cid in order],
          "the synthetic crossings carry the divergences they were built with")

    sel = cs._branch_origins(xs, 20)
    check([x.cid for x in sel] == [9, 8],
          "pct=20 of 10 crossings selects exactly the top 2, strongest first",
          f"cids {[x.cid for x in sel]}")
    sel = cs._branch_origins(xs, 100)
    check([x.cid for x in sel] == list(range(9, -1, -1)),
          "pct=100 selects all 10, in descending divergence order",
          f"cids {[x.cid for x in sel]}")
    sel = cs._branch_origins(xs, 1)
    check([x.cid for x in sel] == [9],
          "pct=1 selects the single strongest crossing")
    check(cs._branch_origins([], 20) == [] and cs._branch_origins([], 100) == [],
          "an empty generation selects no origins at all (no forced minimum)")
    sel = cs._branch_origins(xs[:3], 20)
    check(len(sel) == 1, "n=3 at pct=20 rounds up to 1 origin", f"{len(sel)} origins")

    # the size is max(1, ceil(n*pct/100)) at every pct, and never more than n
    bad = [p for p in range(1, 101)
           if len(cs._branch_origins(xs, p)) != max(1, math.ceil(10 * p / 100))]
    check(not bad, "the selection size is max(1, ceil(pct/100 * n)) for pct = 1..100",
          f"wrong at {bad}")

    # ties: two crossings of equal divergence keep detection (cid) order
    tied = [_xing(5, 0.4), _xing(2, 0.4), _xing(7, 0.1)]
    check([x.cid for x in cs._branch_origins(tied, 100)] == [5, 2, 7],
          "equal divergences keep their detection order (stable sort)")

    before = [x.cid for x in xs]
    cs._branch_origins(xs, 30)
    check([x.cid for x in xs] == before, "the input list is not reordered or consumed")


def _survey(branch, depth, pct, k=4, n_chords=4, seed=11, min_len=None):
    """One stubbed survey; returns (run, cap). min_len overrides the shortest-chord rule, so a
    ray is never dropped for being short and the fan sizes below are exact."""
    run = cs.CascadeRun(
        run_id="t", prompts=[f"p{i}" for i in range(k)], seed=seed, steps=8, height=64,
        width=64, guidance_scale=3.5, n_chords=n_chords, n_patches=2,
        branch=branch, depth=depth, branch_top_pct=pct,
        probe_steps=None,    # None skips the rebracket pass
        certify=True)        # the full pipeline, so the stub still stops the run AT the bisection
    orig_ev, orig_len = cs.evaluate, cs._min_chord_len
    cs.evaluate = _stub_evaluate(run, _band_field(k), {})
    if min_len is not None:
        cs._min_chord_len = lambda stride, _v=min_len: _v
    try:
        cs.run_cascade(None, run, None)
    except _StopAfterChords:
        pass
    finally:
        cs.evaluate, cs._min_chord_len = orig_ev, orig_len
    return run, cs._chord_cap(n_chords, branch, depth)


def _fans(run, cap, branch, pct, label):
    """Per generation: the chord count, the fan each selected origin got, and the note."""
    for g in range(1, run.depth + 1):
        prev = [x for x in run.crossings if x.gen == g - 1]
        origins = cs._branch_origins(prev, pct)
        before = sum(1 for m in run.chords_meta if m["gen"] < g)
        budget = min(branch * len(origins), max(0, cap - before))
        got = sum(1 for m in run.chords_meta if m["gen"] == g)
        check(got == budget,
              f"{label} gen {g}: {budget} chords = branch x origins, capped",
              f"{got} chords from {len(origins)} of {len(prev)} crossings, cap {cap}")
        # every origin gets a full fan until the budget runs out; the one it runs out on gets
        # the remainder, and the origins behind it get nothing
        cnt = Counter(m["origin_cid"] for m in run.chords_meta if m["gen"] == g)
        want = [branch] * (budget // branch) if branch else []
        if branch and budget % branch:
            want.append(budget % branch)
        want += [0] * (len(origins) - len(want))
        check([cnt.get(x.cid, 0) for x in origins] == want,
              f"{label} gen {g}: each selected origin spawns {branch} rays until the cap",
              f"fans {[cnt.get(x.cid, 0) for x in origins]}")
        n_new = sum(1 for x in run.crossings if x.gen == g)
        # the angle-spreading suffix (tests/branch_spread_test.py covers the spreading itself)
        spread = [m["min_angle_deg"] for m in run.chords_meta if m["gen"] == g]
        note = (f"gen {g}: {got} chords from {len(origins)} origins "
                f"(top {pct} % of {len(prev)} crossings), {n_new} crossings"
                + (f", median spread {np.median(spread):.0f}°" if spread else ""))
        check(note in run.notes, f"{label} gen {g}: the note names origins and the percentile",
              note)


def part2():
    print("\n(ii) the generational loop: fans per selected origin, cap, notes")
    run, cap = _survey(branch=2, depth=2, pct=20, min_len=0.0)
    gens = [m["gen"] for m in run.chords_meta]
    check(max(gens) == 2 and len(run.crossings) > 0,
          "pct=20 branch=2 depth=2 produced three generations",
          f"gen counts {[gens.count(g) for g in range(3)]}, {len(run.crossings)} crossings")
    _fans(run, cap, 2, 20, "pct=20")
    check(run.notes[0] == f"gen 0: {gens.count(0)} chords, "
                          f"{sum(1 for x in run.crossings if x.gen == 0)} crossings",
          "generation 0 has no origins, so its note keeps the plain wording", run.notes[0])

    # the control: at pct=100 every crossing of a generation is an origin
    run, cap = _survey(branch=1, depth=2, pct=100, min_len=0.0)
    n0 = sum(1 for x in run.crossings if x.gen == 0)
    check(len(cs._branch_origins([x for x in run.crossings if x.gen == 0], 100)) == n0
          and sum(1 for m in run.chords_meta if m["gen"] == 1) == min(n0, cap - 4),
          "pct=100: every crossing spawns, so generation 1 has one chord per root crossing",
          f"{n0} root crossings")
    _fans(run, cap, 1, 100, "pct=100")

    # a run whose later generation is cut by the geometric cap, so the truncated fan is real
    run, cap = _survey(branch=6, depth=2, pct=100, min_len=0.0)
    check(len(run.chords_meta) == cap, "branch=6 pct=100 fills the geometric cap exactly",
          f"{len(run.chords_meta)}/{cap} chords")
    _fans(run, cap, 6, 100, "capped")

    # a single origin is always kept, and a generation with no crossings ends the survey
    run, _ = _survey(branch=2, depth=1, pct=1, min_len=0.0)
    n1 = sum(1 for m in run.chords_meta if m["gen"] == 1)
    check(n1 == 2, "pct=1 still selects one origin, which spawns its full fan",
          f"{n1} chords in generation 1")


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

        r = c.post("/api/cascade/start",
                   json=dict(body, branch=2, depth=1, branch_top_pct=50))
        rid = r.json().get("run_id", "")
        run = app.state.cascades.get(rid)
        check(r.status_code == 200 and run is not None and run.branch_top_pct == 50,
              "branch_top_pct=50 accepted and threaded into the run",
              f"HTTP {r.status_code}")
        if rid:
            j = c.get(f"/api/cascade/{rid}/status").json()
            check(j["branch_top_pct"] == 50 and j["branch"] == 2 and j["depth"] == 1,
                  "status echoes the branching settings the run was started with",
                  f"branch {j['branch']} depth {j['depth']} top {j['branch_top_pct']} %")

        r = c.post("/api/cascade/start", json=body)
        rid = r.json().get("run_id", "")
        run = app.state.cascades.get(rid)
        got = run.branch_top_pct if run else None
        check(r.status_code == 200 and got == 20,
              "an omitted branch_top_pct defaults to 20", f"got {got!r}")
        if rid:
            j = c.get(f"/api/cascade/{rid}/status").json()
            check(j["branch_top_pct"] == 20 and j["branch"] == 0,
                  "the default shows up in the status of a run without branching")

        for bad in (0, 101, -5):
            r = c.post("/api/cascade/start", json=dict(body, branch_top_pct=bad))
            check(r.status_code == 422, f"branch_top_pct={bad} rejected (out of range)",
                  f"got HTTP {r.status_code}")
    finally:
        cs.run_cascade = orig


if __name__ == "__main__":
    part1()
    part2()
    part3()
    print(f"\n{'ALL CHECKS PASS' if not fails else 'FAILED: ' + ', '.join(fails)}")
    sys.exit(1 if fails else 0)
