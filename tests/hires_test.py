"""Offline checks of the final high-resolution pass (backend/services/cascade.py). No GPU:
cascade.evaluate is replaced by the band-field stub pattern of tests/branch_test.py, so the real
chord phase, the real hires phase and the real certify branch run unchanged.

The pass is OFF by default. With it on, the top `hires_top_pct` % of ALL detected crossings get
`hires_factor - 1` extra cheap probes evenly spaced inside their bracket, and the bracket is
re-detected at stride/factor: the position sharpens by the factor, a bracket that hid two
boundaries splits into two crossings, and a bracket whose change is spread over every sub-step
("diffuse") is kept as it was.

  (i)   PURE -- _hires_points (count and spacing) and _hires_refine on synthetic embedding
        sequences: one sharp jump -> one sub-bracket at the right position and width; two jumps
        -> one crossing plus one split_from child; a diffuse ramp (every sub-step below COS_T
        while the ends are far apart) -> the original bracket, hires_note "diffuse".
  (ii)  SELECTION -- _hires_select: the top 20 % of 10 crossings are the 2 strongest, and the
        rule never selects nothing.
  (iii) PIPELINE -- stubbed surveys. With certify off the detection-only finish quotes the centre
        of the REFINED bracket (checked against the same survey without the pass, which has a
        bit-identical chord phase); with certify on the first bisection round is handed the
        refined brackets. The hires round runs at the probe (cheap-field) steps.
  (iv)  REQUEST -- /api/cascade/start through FastAPI's TestClient with a stubbed app.state and
        cs.run_cascade replaced by a no-op (as tests/stride_test.py does): the defaults, the
        bounds, and what /status reports.

Run: python tests/hires_test.py
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
    """A unit embedding t of the way from _e(a) to _e(b) -- a GRADED change, so a ramp can be
    built whose neighbouring steps stay below COS_T while its ends do not."""
    v = (1.0 - t) * _e(a) + t * _e(b)
    return v / np.linalg.norm(v)


# one bracket exactly STRIDE long on a unit sum-zero chord direction, as the chord phase lays
# them down -- so the geometric width after an x f pass is stride/f, the number the pass records
_U = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)
_WA = np.array([0.4, 0.3, 0.3])
_WB = _WA + cs.STRIDE * _U


def _crossing(cid=3, thumb=-1):
    return cs.Crossing(cid=cid, wa=_WA.copy(), wb=_WB.copy(), ea=_e(0), eb=_e(1), thumb=thumb)


def part1():
    print("\n(i) pure: subdivision and re-detection inside one bracket")
    wa, wb = _WA, _WB
    L = float(np.linalg.norm(wb - wa))
    check(abs(L - cs.STRIDE) < 1e-15, "the fixture bracket is exactly one stride long",
          f"{L:.6f} vs {cs.STRIDE}")
    for f in (2, 3, 4, 8):
        pts = cs._hires_points(wa, wb, f)
        seq = [wa] + pts + [wb]
        gaps = [float(np.linalg.norm(b - a)) for a, b in zip(seq, seq[1:])]
        inside = all(0.0 < float(np.dot(p - wa, wb - wa)) / (L * L) < 1.0 for p in pts)
        check(len(pts) == f - 1 and inside
              and max(abs(g - L / f) for g in gaps) < 1e-12,
              f"×{f}: {f - 1} points strictly inside, spacing |wb-wa|/{f}",
              f"{len(pts)} points, spacing {gaps[0]:.6f} vs {L / f:.6f}")
        on_line = max(float(np.linalg.norm(np.cross(p - wa, wb - wa))) for p in pts)
        check(on_line < 1e-15, f"×{f}: every inserted point stays on the chord",
              f"max off-line {on_line:.2e}")

    # one sharp jump, factor 4, in the third sub-bracket
    f, width = 4, cs.STRIDE / 4
    x = _crossing()
    seq = [x.wa] + cs._hires_points(x.wa, x.wb, f) + [x.wb]
    em = [_e(0), _e(0), _e(0), _e(1), _e(1)]
    extra = cs._hires_refine(x, seq, em, width)
    got_w = float(np.linalg.norm(x.wa - x.wb))
    check(not extra and x.hires and x.hires_note is None
          and abs(got_w - width) < 1e-12 and x.hires_width == width
          and np.allclose(x.wa, seq[2]) and np.allclose(x.wb, seq[3])
          and x.ea is em[2] and x.eb is em[3],
          "one sharp jump: a single sub-bracket at the right position, width stride/4",
          f"width {got_w:.6f} vs {width:.6f}, {len(extra)} extra crossings")
    check(x.hires_profile == [0.0, 0.0, 1.0, 0.0],
          "the profile records the divergence of every sub-step",
          f"{x.hires_profile}")

    # two jumps in one bracket: the strongest stays, the other becomes a split child
    x = _crossing(cid=7)
    seq = [x.wa] + cs._hires_points(x.wa, x.wb, f) + [x.wb]
    #        e0 -> e1 (1.0)        e1 -> e1 (0)   e1 -> tilt (weaker, still > COS_T)
    em = [_e(0), _e(1), _e(1), _tilt(1, 2, 0.75), _tilt(1, 2, 0.75)]
    extra = cs._hires_refine(x, seq, em, width)
    d0 = cs._cosd(em[0], em[1])
    d2 = cs._cosd(em[2], em[3])
    ok = (len(extra) == 1 and d0 > d2 > cs.COS_T
          and np.allclose(x.wa, seq[0]) and np.allclose(x.wb, seq[1])
          and x.split_from is None
          and np.allclose(extra[0].wa, seq[2]) and np.allclose(extra[0].wb, seq[3])
          and extra[0].split_from == 7 and extra[0].cid == -1 and extra[0].gen == x.gen
          and extra[0].hires and extra[0].hires_width == width
          and extra[0].ea is em[2] and extra[0].eb is em[3])
    check(ok, "two jumps: the strongest stays the crossing, the other splits off with split_from",
          f"{len(extra)} split, divergences {d0:.2f} / {d2:.2f}")

    # diffuse: every sub-step below COS_T, ends far above it -- the bracket is kept
    x = _crossing(cid=9)
    seq = [x.wa] + cs._hires_points(x.wa, x.wb, f) + [x.wb]
    em = [_tilt(0, 1, t) for t in (0.0, 0.25, 0.5, 0.75, 1.0)]
    before = (x.wa.copy(), x.wb.copy())
    extra = cs._hires_refine(x, seq, em, width)
    steps = [cs._cosd(em[i], em[i + 1]) for i in range(4)]
    ends = cs._cosd(em[0], em[-1])
    check(max(steps) < cs.COS_T < ends,
          "the ramp is a real diffuse case (every step below COS_T, ends above it)",
          f"max step {max(steps):.2f}, ends {ends:.2f}")
    check(not extra and x.hires and x.hires_note == "diffuse" and x.hires_width is None
          and np.allclose(x.wa, before[0]) and np.allclose(x.wb, before[1]),
          "diffuse: the original bracket is kept, flagged hires with hires_note 'diffuse'",
          f"note {x.hires_note!r}, width {x.hires_width!r}")

    # a detection-only crossing keeps a bracket-END thumbnail, now a refined probe
    x = _crossing(cid=11, thumb=5)
    seq = [x.wa] + cs._hires_points(x.wa, x.wb, f) + [x.wb]
    cs._hires_refine(x, seq, [_e(0), _e(0), _e(1), _e(1), _e(1)], width,
                     gis=[None, 31, 32, 33, None])
    check(x.thumb in (31, 32), "the thumbnail moves to a probe of the refined bracket",
          f"thumb {x.thumb}")


def part2():
    print("\n(ii) selection: the top % of ALL crossings, at least one")
    xs = [cs.Crossing(cid=i, wa=np.zeros(3), wb=np.zeros(3),
                      ea=_e(0), eb=_tilt(0, 1, 0.2 + 0.08 * i)) for i in range(10)]
    divs = [cs._cosd(x.ea, x.eb) for x in xs]
    check(divs == sorted(divs), "the fixture's divergence rises with cid (so ranking is testable)")
    sel = cs._hires_select(xs, 20)
    check([x.cid for x in sel] == [9, 8],
          "top 20 % of 10 crossings = the 2 strongest, strongest first",
          f"cids {[x.cid for x in sel]}")
    check([x.cid for x in cs._hires_select(xs, 1)] == [9],
          "a percentage that rounds below one crossing still selects the strongest")
    check(len(cs._hires_select(xs, 100)) == 10, "100 % selects every crossing")
    check(cs._hires_select([], 20) == [], "no crossings: nothing to refine")


class _StopAt(Exception):
    """Raised by the stub at a chosen round: the run object already carries what is checked."""


def _stub_evaluate(run, field, rounds, stop_at=None):
    """Stand-in for cascade.evaluate: embeds every weight vector through the analytic field and
    records (label, weights, steps) per GPU round, optionally stopping at a round label."""
    def stub(app, r, pool, weights, seed, label, ctl=None, on_arrival=None, steps=None):
        base = label.split(".")[0]
        if stop_at is not None and base.startswith(stop_at):
            rounds.append((base, [np.asarray(w, dtype=float) for w in weights], steps))
            raise _StopAt()
        rounds.append((base, [np.asarray(w, dtype=float) for w in weights], steps))
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


def _survey(k=4, n_chords=4, seed=11, stride=cs.STRIDE, certify=False, probe_steps=4,
            hires=False, hires_factor=4, hires_top_pct=20, period=None, stop_at=None):
    """One stubbed survey; returns (run, [(round label, weights, steps)])."""
    run = cs.CascadeRun(
        run_id="h", prompts=[f"p{i}" for i in range(k)], seed=seed, steps=8, height=64,
        width=64, guidance_scale=3.5, n_chords=n_chords, n_patches=2, stride=stride,
        probe_steps=probe_steps, certify=certify,
        hires=hires, hires_factor=hires_factor, hires_top_pct=hires_top_pct)
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


def part3():
    print("\n(iii) the pass inside the real loop (stubbed field)")
    base, _br = _survey()
    run, rounds = _survey(hires=True)
    labels = [l for l, _w, _s in rounds]
    hr = [(w, s) for l, w, s in rounds if l == "hires"]
    check(labels.count("hires") == 1 and "hires" in labels,
          "one hires round, after the chord rounds",
          f"rounds {labels}")
    check(labels.index("hires") == max(i for i, l in enumerate(labels)
                                      if l.startswith("chords")) + 1,
          "the hires round comes after every chord generation")
    check(hr and hr[0][1] == run.probe_steps,
          "the extra probes run on the cheap field (probe_steps), like the chord probes",
          f"steps {hr[0][1] if hr else None} vs probe_steps {run.probe_steps}")

    sel = cs._hires_select(base.crossings, 20)
    n_sel = len(sel)
    check(hr and len(hr[0][0]) == n_sel * (run.hires_factor - 1),
          "the round holds (factor - 1) probes per selected crossing",
          f"{len(hr[0][0]) if hr else 0} probes for {n_sel} of {len(base.crossings)} crossings")

    # the chord phase must be untouched: same crossings, same brackets, up to the refinement
    nb = len(base.crossings)
    same_phase = ([x.cid for x in base.crossings] == [x.cid for x in run.crossings[:nb]]
                  and all(x.gen == y.gen for x, y in zip(base.crossings, run.crossings[:nb])))
    check(same_phase, "the chord phase is identical with the pass on and off (same seed)",
          f"{nb} crossings, {len(run.crossings) - nb} split off")

    refined = [x for x in run.crossings if x.hires and x.hires_note is None]
    width = run.stride / run.hires_factor
    check(len(refined) > 0, "the pass refined something on this field",
          f"{len(refined)} refined, "
          f"{sum(1 for x in run.crossings if x.hires_note == 'diffuse')} diffuse, "
          f"{sum(1 for x in run.crossings if x.split_from is not None)} split")
    check(all(abs(float(np.linalg.norm(x.wa - x.wb)) - width) < 1e-12 for x in refined)
          and all(x.hires_width == width for x in refined),
          "every refined bracket is stride/factor wide", f"{width:.5f}")
    # detection-only finish: mid/n read off the REFINED bracket, inside the original one
    by_cid = {x.cid: x for x in base.crossings}
    mid_ok = all(np.allclose(x.mid, (x.wa + x.wb) / 2) for x in run.crossings)
    moved = [x for x in refined if x.cid in by_cid
             and not np.allclose(x.mid, by_cid[x.cid].mid)]
    inside = all(
        float(np.linalg.norm(x.mid - by_cid[x.cid].mid)) <= run.stride / 2 + 1e-12
        for x in refined if x.cid in by_cid)
    check(mid_ok and inside and len(moved) > 0,
          "detection only: mid is the centre of the refined bracket, inside the original one",
          f"{len(moved)}/{len(refined)} refined positions moved")
    n_ok = all(abs(float(np.linalg.norm(x.n)) - 1.0) < 1e-9 for x in run.crossings)
    check(n_ok, "every crossing (splits included) still carries a unit normal")
    splits = [x for x in run.crossings if x.split_from is not None]
    check(all(x.b is None and not x.significant and x.cid >= nb for x in splits),
          "split children are numbered after the chord phase and left uncertified",
          f"{len(splits)} splits")
    note = next((n for n in run.notes if n.startswith("hires:")), "MISSING")
    check("refined" in note and "split" in note and "diffuse" in note and "→" in note,
          "the summary note reports refined/split/diffuse and the width before→after", note)

    # certify on: the first bisection round starts from the refined brackets. A x2 pass at the
    # default stride leaves them at 0.0125, still above BRACKET (0.012), so bisection really runs
    # -- a x4 pass would land under it and the loop would legitimately do nothing.
    run2, rounds2 = _survey(certify=True, probe_steps=None, hires=True,
                            hires_factor=2, stop_at="bisect")
    mids = next((w for l, w, _s in rounds2 if l.startswith("bisect")), None)
    todo = [x for x in run2.crossings
            if float(np.linalg.norm(x.wa - x.wb)) > cs.BRACKET and x.ea is not None]
    want = [(x.wa + x.wb) / 2 for x in todo]
    ref2 = [x for x in todo if x.hires and x.hires_note is None]
    check(mids is not None and len(mids) == len(want)
          and all(np.allclose(a, b) for a, b in zip(mids, want)),
          "certify on: bisection is handed the refined brackets, not the one-stride ones",
          f"{0 if mids is None else len(mids)} mids, {len(ref2)} of them refined")
    check(len(ref2) > 0 and run2.stride / 2 > cs.BRACKET
          and all(abs(float(np.linalg.norm(x.wa - x.wb)) - run2.stride / 2) < 1e-12
                  for x in ref2),
          "those refined brackets are stride/2 wide, still above BRACKET so bisection runs",
          f"{len(ref2)} refined of {len(todo)} to bisect, "
          f"width {run2.stride / 2:.4f} vs BRACKET {cs.BRACKET}")

    # bands packed closer than one stride: a chord bracket really does hide several boundaries,
    # so the splitting path runs inside the loop and not only in the pure test above
    run3, _r3 = _survey(period=0.012, hires=True)
    sp = [x for x in run3.crossings if x.split_from is not None]
    parents = {x.split_from for x in sp}
    check(len(sp) > 0 and all(x.hires and x.cid >= len(run3.crossings) - len(sp)
                              and x.split_from < x.cid for x in sp),
          "bands closer than one stride: brackets split into extra crossings in the real loop",
          f"{len(sp)} splits from {len(parents)} brackets of {len(run3.crossings)} crossings")
    okw = all(abs(float(np.linalg.norm(x.wa - x.wb)) - run3.stride / run3.hires_factor) < 1e-12
              for x in sp)
    check(okw and all(np.allclose(x.mid, (x.wa + x.wb) / 2) for x in sp),
          "a split child is quoted at its own sub-bracket, stride/factor wide")

    # the pass lays down no chord: provenance and the fair-area filter are untouched
    check(len(run.chords_geo) == len(base.chords_geo)
          and len(run.chords_meta) == len(base.chords_meta)
          and all(m == n for m, n in zip(run.chords_meta, base.chords_meta)),
          "no new chord is registered, so chords_geo/chords_meta are unchanged",
          f"{len(run.chords_geo)} chords")
    check(all(x.gen == by_cid[x.split_from].gen for x in splits if x.split_from in by_cid),
          "a split child inherits the generation of the crossing it came from")


def part4():
    print("\n(iv) /start and /status through TestClient (stubbed state, run_cascade no-op)")
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
        got = (run.hires, run.hires_top_pct, run.hires_factor) if run else None
        check(r.status_code == 200 and got == (False, 20, 4),
              "omitted hires defaults to off, top 20 %, ×4", f"got {got!r}")
        if rid:
            j = c.get(f"/api/cascade/{rid}/status").json()
            check(j["hires"] is False and j["hires_top_pct"] == 20 and j["hires_factor"] == 4,
                  "status reports the run's own hires settings")

        r = c.post("/api/cascade/start",
                   json=dict(body, hires=True, hires_top_pct=35, hires_factor=8))
        rid = r.json().get("run_id", "")
        run = app.state.cascades.get(rid)
        got = (run.hires, run.hires_top_pct, run.hires_factor) if run else None
        check(r.status_code == 200 and got == (True, 35, 8),
              "hires=true with its two settings is threaded into the run", f"got {got!r}")
        if rid:
            j = c.get(f"/api/cascade/{rid}/status").json()
            check(j["hires"] is True and j["hires_factor"] == 8,
                  "status reports a run started with the pass on")

        for field, bad in (("hires_factor", 1), ("hires_factor", 9), ("hires_top_pct", 0),
                           ("hires_top_pct", 101), ("hires", "maybe")):
            r = c.post("/api/cascade/start", json=dict(body, **{field: bad}))
            check(r.status_code == 422, f"{field}={bad!r} rejected",
                  f"got HTTP {r.status_code}")
    finally:
        cs.run_cascade = orig


if __name__ == "__main__":
    part1()
    part2()
    part3()
    part4()
    print(f"\n{'ALL CHECKS PASS' if not fails else 'FAILED: ' + ', '.join(fails)}")
    sys.exit(1 if fails else 0)
