"""Offline checks of the Mixing desk (backend/services/desk.py + its router). No GPU.

Three parts, all analytic or stubbed -- nothing renders:

  (i)   GEOMETRY -- WeightLifter's line w_i = alpha, w_j = w0_j (1 - alpha)/(1 - w0_i) keeps
        sum(w) = 1 and the others' proportions at k = 3..8, passes through w0 at alpha = w0_i,
        reaches the vertex at 1 and the opposite face at 0; a vertex mix (w0_i = 1) is
        handled (the others share the rest equally) and flagged. The alpha grid always samples
        both ends at a spacing <= dalpha. A line's cache key is the same for every mix ON the
        line (so a move along line i keeps line i) and differs for the other lines. Flip
        detection, segments and the nearest-flip readout on known inputs (including the
        bracket that contains the mix, where the side test decides the direction). The cost
        projection, and which configurations the ceilings refuse.
  (ii)  LOOP -- run_position over a synthetic TWO-BASIN field with cascade.evaluate stubbed as
        tests/metro_test.py stubs it, and a CHEAP field whose sheet is offset from the FULL
        field's (so the refine provably reads full fidelity). Along every line the sheet is
        crossed at a known alpha*: the detected bracket must contain it, and the refined
        position must land within dalpha/16 of the FULL field's alpha* and not the cheap one.
        One cheap batch for all lines + the mix (a mix that is itself a grid sample is not
        rendered twice), refine batched as ends + 3 bisection rounds at full fidelity, a move
        along line 0 re-renders only the other k - 1 lines, a revisit renders nothing, a
        cancelled move caches nothing.
  (iii) ROUTER through FastAPI's TestClient with the same stub, live (positions compute on the
        router's own worker thread): start -> status, move by (line, alpha) and by w0, the
        revisit, every refusal (400 with the projection past either ceiling, bad w0, missing
        target, line out of range; 404 unknown desk; 422 bounds), the readout label, cancel and
        the image route.

Run: python tests/desk_test.py
"""
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.services import desk as dk                 # noqa: E402
from backend.services import cascade as cs              # noqa: E402
from backend.services.cascade import COS_T              # noqa: E402

fails = []


def check(ok, label, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(label)


# ------------------------------------------------------------------------- (i) geometry
def part1():
    print("\n(i) the WeightLifter line, the grid, the cache key, flips, readout, cost")
    rng = np.random.default_rng(0)
    worst_sum = worst_prop = worst_ai = 0.0
    for k in range(3, 9):
        for _ in range(40):
            w0 = rng.dirichlet(np.ones(k))
            i = int(rng.integers(k))
            a = float(rng.random())
            w = dk.line_point(w0, i, a)
            o = [j for j in range(k) if j != i]
            worst_sum = max(worst_sum, abs(float(w.sum()) - 1.0))
            worst_ai = max(worst_ai, abs(float(w[i]) - a))
            worst_prop = max(worst_prop, float(np.abs(w[o] / w[o].sum()
                                                      - w0[o] / w0[o].sum()).max()))
    check(worst_sum < 1e-12 and worst_ai == 0.0 and worst_prop < 1e-12,
          "every line point sums to 1, has w_i = alpha, and keeps the others' proportions "
          "(k = 3..8)", f"|sum-1| {worst_sum:.1e}, proportion defect {worst_prop:.1e}")
    w0 = np.array([0.1, 0.2, 0.3, 0.4])
    ends = [dk.line_point(w0, 2, x) for x in (0.3, 1.0, 0.0)]
    check(np.allclose(ends[0], w0, atol=1e-15) and np.allclose(ends[1], [0, 0, 1, 0])
          and np.allclose(ends[2], [1 / 7, 2 / 7, 0, 4 / 7], atol=1e-15),
          "alpha = w0_i is w0 itself, alpha = 1 the vertex, alpha = 0 the opposite face")
    v = np.array([0.0, 1.0, 0.0, 0.0])
    wd = dk.line_point(v, 1, 0.4)
    check(dk.is_degenerate(v, 1) and not dk.is_degenerate(v, 0)
          and np.allclose(wd, [0.2, 0.4, 0.2, 0.2]) and abs(wd.sum() - 1) < 1e-15
          and np.allclose(dk.line_point(v, 0, 0.3), [0.3, 0.7, 0, 0]),
          "a vertex mix: its own line shares the rest equally (flagged degenerate), the "
          "other lines run along the edges out of it")

    g = dk.alpha_grid(0.05)
    g3 = dk.alpha_grid(0.03)
    check(len(g) == 21 and g[0] == 0.0 and g[-1] == 1.0
          and max(b - a for a, b in zip(g, g[1:])) <= 0.05 + 1e-12
          and g3[-1] == 1.0 and max(b - a for a, b in zip(g3, g3[1:])) <= 0.03 + 1e-12,
          "the alpha grid samples both ends at a spacing <= dalpha",
          f"{len(g)} samples at 0.05, {len(g3)} at 0.03")

    w0 = rng.dirichlet(np.ones(5))
    w1 = dk.line_point(w0, 2, 0.7)
    same = [dk.line_key(w0, i, 0.05, False) == dk.line_key(w1, i, 0.05, False)
            for i in range(5)]
    check(same == [False, False, True, False, False]
          and dk.line_key(w0, 2, 0.05, False) != dk.line_key(w0, 2, 0.05, True)
          and dk.line_key(w0, 2, 0.05, False) != dk.line_key(w0, 2, 0.04, False),
          "a line's key is the same for every mix ON it (a move along line 2 keeps line 2) and "
          "differs for the other lines, the stride and refine", f"{same}")

    # flips / segments / readout on a known label sequence
    alphas = dk.alpha_grid(0.1)                                       # 11 samples
    ea, eb, ec = np.eye(3)
    lab = [ea] * 4 + [eb] * 4 + [None] + [ec] * 2                     # one sample missing
    gis = [100 + t if lab[t] is not None else None for t in range(11)]
    fl = dk.detect_flips(alphas, lab, gis)
    check([(f["lo"], f["hi"]) for f in fl] == [(3, 4)] and abs(fl[0]["alpha"] - 0.35) < 1e-12
          and fl[0]["img_lo"] == 103 and fl[0]["img_hi"] == 104 and abs(fl[0]["div"] - 1) < 1e-12
          and abs(fl[0]["width"] - 0.1) < 1e-12,
          "a change is a consecutive pair past COS_T, quoted at the bracket midpoint; a pair with "
          "a missing end is no evidence (the b|c change behind the gap is not claimed)")
    fl2 = dk.detect_flips(alphas, [ea] * 4 + [eb] * 4 + [ec] * 3, list(range(11)))
    seg = dk.segments(alphas, fl2, list(range(11)))
    check([(s["lo"], s["hi"]) for s in seg] == [(0, 3), (4, 7), (8, 10)]
          and [s["alpha_lo"] for s in seg] == [0.0, fl2[0]["alpha"], fl2[1]["alpha"]]
          and seg[-1]["alpha_hi"] == 1.0
          and all(s["lo"] <= s["thumb"] <= s["hi"] for s in seg),
          "segments run flip to flip (0 and 1 at the ends), each with a thumbnail from inside it",
          f"{[(s['lo'], s['hi'], s['thumb']) for s in seg]}")
    up, down = dk.nearest_flips(0.5, alphas, fl2)
    check(abs(up["dist"] - 0.25) < 1e-12 and up["thumb"] == 8
          and abs(down["dist"] - 0.15) < 1e-12 and down["thumb"] == 3,
          "nearest change up and down, each with the image just BEYOND it",
          f"up {up['dist']:.2f}, down {down['dist']:.2f}")
    up2, down2 = dk.nearest_flips(0.33, alphas, fl2, side=lambda f: "up")
    up3, down3 = dk.nearest_flips(0.37, alphas, fl2, side=lambda f: "up")
    check(abs(up2["dist"] - 0.02) < 1e-12 and abs(up3["dist"] - 0.02) < 1e-12
          and down3 is None and down2 is None,
          "inside a change's own bracket the side test decides the direction (the midpoint may "
          "sit on the wrong side of the mix)")
    check(dk.nearest_flips(0.5, alphas, []) == (None, None), "no change: nothing either way")

    # cost
    tot, rows = dk.projected_cost(4, 21, False, 4)
    tot_r, _ = dk.projected_cost(4, 21, True, 4)
    tot_c, _ = dk.projected_cost(4, 21, False, 4, n_cached_lines=1, w0_cached=True)
    check(tot == 0.5 * (4 * 21 + 1) and tot_r == tot + 4 * dk.REFINE_MAX_PER_LINE * 5
          and tot_c == 0.5 * 3 * 21 and dk.projected_cost(4, 21, False, None)[0] == 2 * tot,
          "cost = k x samples x 0.5 + the mix's probe (+ refine's worst case 4 x 5 full images "
          "per line); cached lines and a cached mix are free; full-fidelity probes double it",
          f"{tot} / {tot_r} / {tot_c} image-eq")
    ok_default = dk.projected_cost(8, 21, True, 4)[0] <= dk.MAX_REQUEST_IMAGE_EQ
    refused = dk.projected_cost(8, 51, True, 4)[0] > dk.MAX_REQUEST_IMAGE_EQ
    ok_norefine = dk.projected_cost(8, 51, False, 4)[0] <= dk.MAX_REQUEST_IMAGE_EQ
    check(ok_default and refused and ok_norefine,
          "the request ceiling admits k = 8 at the default stride with refine, refuses "
          "k = 8 at 0.02 with refine, admits it without")


# ----------------------------------------------- (ii) run_position on a synthetic field
def _fields(k, seed=2, band=0.0005, cheap_shift=0.006):
    """(cheap, full, p): two basins across the sheet <w - 1/k, p> = c, with the CHEAP field's
    sheet offset by cheap_shift -- the cheap probes and the full renders disagree on where the
    change is, which is what lets the test see which one the refine read."""
    rng = np.random.default_rng(seed)
    p = rng.standard_normal(k)
    p -= p.mean()
    p /= np.linalg.norm(p)
    Q = np.linalg.qr(rng.standard_normal((8, 3)))[0]

    def mk(shift):
        def f(w):
            g = float((np.asarray(w, dtype=float) - 1.0 / k) @ p) + shift
            e = Q[:, 0] + 2.5 * math.tanh(g / band) * Q[:, 1] + 0.1 * math.sin(7 * g) * Q[:, 2]
            return e / float(np.linalg.norm(e))
        return f
    return mk(cheap_shift), mk(0.0), p


def _key(w, steps):
    return (tuple(round(float(v), 9) for v in w), steps)


def _stub(run, cheap, full, calls, cancel_at=None, seen=None):
    """tests/metro_test.py's stub shape. `seen` collects (recipe, steps) of every render, so a
    test can count what a batch SHOULD contain independently of the render cache."""
    def stub(app, r, pool, weights, seed, label, ctl=None, on_arrival=None, steps=None):
        calls.append((label.split(".")[0], len(weights), steps))
        if seen is not None:
            seen.update(_key(w, steps) for w in weights)
        if cancel_at is not None and len(calls) >= cancel_at:
            ctl.status = "cancelled"
            return [None] * len(weights)
        f = full if steps is None else cheap
        out = []
        for li, w in enumerate(weights):
            gi = r.next_idx
            r.next_idx += 1
            r.embeddings[gi] = f(w)
            r.generated += 1
            if r.thumbs is not None:
                r.thumbs[gi] = f"img{gi}".encode()
            if on_arrival is not None:
                on_arrival(li, gi)
            out.append(gi)
        return out
    return stub


def _alpha_star(w0, i, p, shift):
    """Where line i crosses <w - 1/k, p> + shift = 0 (g is affine in alpha), or None."""
    k = len(w0)
    g0 = float((dk.line_point(w0, i, 0.0) - 1.0 / k) @ p) + shift
    g1 = float((dk.line_point(w0, i, 1.0) - 1.0 / k) @ p) + shift
    if (g0 > 0) == (g1 > 0):
        return None
    return g0 / (g0 - g1)


def _mk(k, refine=False):
    return dk.DeskRun(run_id="d", prompts=[f"p{i}" for i in range(k)], seed=3, steps=8,
                      height=64, width=64, guidance_scale=3.5, probe_steps=4, refine=refine)


def part2():
    print("\n(ii) run_position on a synthetic two-basin field (stubbed, no GPU)")
    k = 4
    cheap, full, p = _fields(k)
    w0 = np.array([0.31, 0.22, 0.16, 0.31])      # no weight on the 0.05 grid
    run = _mk(k)
    calls, seen = [], set()
    orig = cs.evaluate
    cs.evaluate = _stub(run, cheap, full, calls, seen=seen)
    try:
        pos, created = dk.add_position(run, w0, 0.05, False)
        dk.run_position(None, run, None, pos)
        check(created and pos.status == "complete" and len(calls) == 1
              and calls[0] == ("desk0", k * 21 + 1, 4)
              and run.n_probes == k * 21 + 1 and run.cost_image_eq == 0.5 * (k * 21 + 1),
              "ONE cheap batch: k lines x 21 samples + the mix, at probe_steps 4, 0.5 each",
              f"{calls[0]}, {run.cost_image_eq} image-eq")
        ok_lines, n_cross = True, 0
        for ln in pos.lines:
            a_star = _alpha_star(w0, ln["i"], p, 0.006)
            if a_star is None:
                ok_lines &= ln["flips"] == []
                continue
            n_cross += 1
            ok_lines &= (len(ln["flips"]) == 1
                         and ln["alphas"][ln["flips"][0]["lo"]] <= a_star
                         <= ln["alphas"][ln["flips"][0]["hi"]])
        check(ok_lines and n_cross >= 2,
              "every line that crosses the cheap sheet shows exactly one change, bracketing the "
              "known crossing alpha*; the lines that do not cross show none",
              f"{n_cross}/{k} lines cross")
        ok_read = True
        for r in pos.readout:
            a_star = _alpha_star(w0, r["i"], p, 0.006)
            a0 = w0[r["i"]]
            if a_star is None:
                ok_read &= r["up"] is None and r["down"] is None and r["nearest"] is None
            elif a_star > a0:
                ok_read &= r["down"] is None and abs(r["up"]["dist"] - (a_star - a0)) <= 0.025
            else:
                ok_read &= r["up"] is None and abs(r["down"]["dist"] - (a0 - a_star)) <= 0.025
        near = [r["nearest"] for r in pos.readout if r["nearest"] is not None]
        check(ok_read and near == sorted(near)
              and [r["nearest"] for r in pos.readout][len(near):] == [None] * (k - len(near)),
              "the readout: the change on the right side of the mix at |alpha* - w0_i| "
              "(+- dalpha/2), prompts sorted by the nearer change, no-change prompts last")
        check(all(len(ln["segments"]) == len(ln["flips"]) + 1 for ln in pos.lines)
              and all(dk.line_key(w0, ln["i"], 0.05, False) in run.line_cache for ln in pos.lines),
              "segments = changes + 1 per line, and every finished line is cached")

        # refine at the same mix: the cheap samples are the same recipes, so only refine renders
        n0 = len(calls)
        pos_r, _ = dk.add_position(run, w0, 0.05, True)
        dk.run_position(None, run, None, pos_r)
        labels = [c[0] for c in calls[n0:]]
        n_fl = sum(len(ln["flips"]) for ln in pos_r.lines)
        check(labels == ["desk1ends", "desk1bis0", "desk1bis1", "desk1bis2"]
              and all(c[2] is None for c in calls[n0:])
              and calls[n0][1] == 2 * n_fl and all(c[1] == n_fl for c in calls[n0 + 1:]),
              "refine: the cheap probes come back out of the render cache; then one full-"
              "fidelity batch of bracket ends and three of midpoints, across all lines at once",
              f"{[(c[0], c[1]) for c in calls[n0:]]}")
        # The refine re-renders the CHEAP bracket's ends at full fidelity and bisects inside it.
        # Where the full field's crossing lies inside that bracket it must be found to dalpha/16
        # (and the change confirmed); where the cheap probes put it in the wrong bracket the
        # full-fidelity ends agree, and the change must be flagged unconfirmed, not "found".
        ok_ref, inside, outside, told = True, 0, 0, 0
        for ln in pos_r.lines:
            a_full = _alpha_star(w0, ln["i"], p, 0.0)
            a_cheap = _alpha_star(w0, ln["i"], p, 0.006)
            for f in ln["flips"]:
                ok_ref &= f["refined"] and abs(f["width"] - 0.05 / 8) < 1e-12
                lo_a, hi_a = ln["alphas"][f["lo"]], ln["alphas"][f["hi"]]
                if a_full is not None and lo_a < a_full < hi_a:
                    inside += 1
                    ok_ref &= bool(f["confirmed"]) and abs(f["alpha"] - a_full) <= 0.05 / 16 + 1e-9
                    told += abs(a_full - a_cheap) > 0.05 / 16
                else:
                    outside += 1
                    ok_ref &= f["confirmed"] is False
        check(ok_ref and inside >= 1 and told >= 1,
              "each change is bisected to dalpha/8 at FULL fidelity: inside its bracket the full "
              "field's crossing is found to dalpha/16 (where the cheap field's crossing is "
              "measurably elsewhere), and a change the full field does not confirm is flagged",
              f"{inside} found ({told} distinguishable from the cheap one), {outside} unconfirmed")

        # a move along line 0: line 0 is the same line, so only k - 1 lines are new
        n0 = len(calls)
        w1 = dk.line_point(w0, 0, 0.6)
        new = {_key(dk.line_point(w1, i, a), 4) for i in range(1, k) for a in dk.alpha_grid(0.05)}
        new.add(_key(w1, 4))
        want = len(new - seen)
        pos2, _ = dk.add_position(run, w1, 0.05, False)
        dk.run_position(None, run, None, pos2)
        check(pos2.cached == [True, False, False, False] and calls[n0] == ("desk2", want, 4)
              and want == (k - 1) * 21 - (k - 1),
              "a marker released on line 0 keeps line 0 (cache) and renders only the other k - 1 "
              "lines' new points -- their vertex ends and the new mix (a line-0 sample) are "
              "already rendered", f"{calls[n0]}")
        n0 = len(calls)
        again, created = dk.add_position(run, w0, 0.05, False)
        check(again is pos and not created and run.current == pos.pid and len(calls) == n0,
              "revisiting a mix serves the finished position as it stands: nothing rendered")
    finally:
        cs.evaluate = orig

    # the barycentre is a grid sample on every line: rendered once, not k + 1 times
    run = _mk(4)
    calls = []
    cs.evaluate = _stub(run, cheap, full, calls)
    try:
        pos, _ = dk.add_position(run, np.full(4, 0.25), 0.05, False)
        dk.run_position(None, run, None, pos)
        check(calls[0][1] == 4 * 21 + 1 - 4 and pos.w0_image == pos.lines[0]["images"][5]
              and run.cost_image_eq == 0.5 * 81,
              "a mix that is itself a sample on every line is rendered once (81 of 85 probes)",
              f"batch {calls[0][1]}")
    finally:
        cs.evaluate = orig

    # a vertex mix
    run = _mk(4)
    calls = []
    cs.evaluate = _stub(run, cheap, full, calls)
    try:
        pos, _ = dk.add_position(run, np.array([1.0, 0, 0, 0]), 0.05, False)
        dk.run_position(None, run, None, pos)
        check(pos.status == "complete" and pos.lines[0]["degenerate"]
              and not any(ln["degenerate"] for ln in pos.lines[1:])
              and all(abs(float(w.sum()) - 1) < 1e-12 for ln in pos.lines for w in ln["ws"])
              and "degenerate" in pos.notes[-1],
              "a vertex mix computes: its own line is flagged degenerate and says so")
    finally:
        cs.evaluate = orig

    # a move cancelled mid-batch leaves nothing cached and charges nothing
    run = _mk(4)
    calls = []
    cs.evaluate = _stub(run, cheap, full, calls, cancel_at=1)
    try:
        pos, _ = dk.add_position(run, w0, 0.05, False)
        dk.run_position(None, run, None, pos)
        check(pos.status == "cancelled" and not run.line_cache and run.cost_image_eq == 0
              and pos.readout == [],
              "a cancelled position caches no line and charges nothing")
    finally:
        cs.evaluate = orig


# ------------------------------------------------------------------------- (iii) the router
def _wait(c, did, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        j = c.get(f"/api/desk/{did}/status").json()
        if j["status"] not in ("running", "pending"):
            return j
        time.sleep(0.02)
    return j


def part3():
    print("\n(iii) router through TestClient (stubbed evaluate, live worker thread)")
    try:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
    except Exception as exc:                         # noqa: BLE001
        check(False, "TestClient available", str(exc))
        return
    from backend.routers import desk as router_mod
    from backend.cache.thumbnail_cache import ThumbnailCache

    fields = {}
    calls = []
    orig_eval = cs.evaluate
    orig_req, orig_ses = dk.MAX_REQUEST_IMAGE_EQ, dk.MAX_SESSION_IMAGE_EQ

    def stub(app, r, pool, weights, seed, label, ctl=None, on_arrival=None, steps=None):
        cheap, full, _p = fields.setdefault(r.k, _fields(r.k))
        return _stub(r, cheap, full, calls)(app, r, pool, weights, seed, label, ctl,
                                           on_arrival, steps)
    cs.evaluate = stub
    tmp = tempfile.TemporaryDirectory()
    try:
        app = FastAPI()
        app.include_router(router_mod.router)
        app.state.cache = ThumbnailCache(Path(tmp.name))
        app.state.gpu_pool = None
        with TestClient(app, raise_server_exceptions=False) as c:
            P = ["a", "b", "c", "d"]
            w0 = [0.31, 0.22, 0.16, 0.31]
            r = c.post("/api/desk/start", json={"prompts": P, "w0": w0})
            did = r.json().get("desk_id", "")
            check(r.status_code == 200 and r.json()["cost_image_eq"] == 0.5 * 85,
                  "/start answers with the desk id and its projected cost", f"HTTP {r.status_code}")
            j = _wait(c, did)
            check(j["status"] == "complete" and j["k"] == 4 and len(j["lines"]) == 4
                  and all(len(ln["alphas"]) == 21 and min(ln["images"]) >= 0 for ln in j["lines"])
                  and [ln["alpha0"] for ln in j["lines"]] == w0 and j["w0_image"] >= 0
                  and j["cost_image_eq"] == 42.5 and j["position"] == 0,
                  "the position computes on the router's worker thread; /status serves k lines "
                  "of 21 samples, each marker at w0_i, and the mix's own image",
                  f"{j['status']}, {j['cost_image_eq']} image-eq")
            check(j["readout_label"] == "exploratory — calibration pending (h26a)"
                  and len(j["readout"]) == 4 and {r["i"] for r in j["readout"]} == {0, 1, 2, 3},
                  "the nearest-flip readout carries its label: exploratory, calibration pending")
            check(any(ln["flips"] for ln in j["lines"])
                  and all(len(ln["segments"]) == len(ln["flips"]) + 1 for ln in j["lines"]),
                  "lines carry their changes and segments")

            r = c.post(f"/api/desk/{did}/move", json={"line": 0, "alpha": 0.6})
            mv = r.json()
            check(r.status_code == 200 and mv["position"] == 1 and mv["cached_lines"] == 1
                  and mv["cost_image_eq"] == 0.5 * 3 * 21,
                  "a released marker (line 0 -> 0.6) is a move: line 0 cached, 3 lines projected "
                  "(the new mix is a line-0 sample, so its probe is free)",
                  f"{mv}")
            j1 = _wait(c, did)
            check(j1["status"] == "complete" and abs(j1["w0"][0] - 0.6) < 1e-12
                  and abs(sum(j1["w0"]) - 1) < 1e-12 and j1["lines"][0]["cached"]
                  and not any(ln["cached"] for ln in j1["lines"][1:])
                  and abs(j1["w0"][1] / j1["w0"][3] - w0[1] / w0[3]) < 1e-12
                  and len(j1["positions"]) == 2,
                  "the new mix keeps the others' proportions, and the history lists both mixes")
            r = c.post(f"/api/desk/{did}/move", json={"w0": w0})
            check(r.status_code == 200 and r.json()["status"] == "complete"
                  and r.json()["position"] == 0 and r.json()["cost_image_eq"] == 0,
                  "moving back to a visited mix is served at once, at no cost")
            check(c.get(f"/api/desk/{did}/status").json()["position"] == 0,
                  "... and becomes the current position")

            for body, why in (({}, "neither w0 nor (line, alpha)"),
                              ({"line": 0}, "a line without alpha"),
                              ({"line": 7, "alpha": 0.5}, "a line that does not exist"),
                              ({"w0": [0.5, 0.5, 0.5, -0.5]}, "a w0 that is no recipe"),
                              ({"w0": [0.5, 0.5]}, "a w0 of the wrong length")):
                r = c.post(f"/api/desk/{did}/move", json=body)
                check(r.status_code == 400 and len(r.json().get("detail", "")) > 10,
                      f"a move with {why} -> 400 with the reason", f"HTTP {r.status_code}")
            check(c.post("/api/desk/nope/move", json={"w0": w0}).status_code == 404,
                  "a move on an unknown desk -> 404")
            for body, why in (({"alpha": 1.5, "line": 0}, "alpha 1.5"),
                              ({"dalpha": 0.005, "w0": w0}, "dalpha 0.005")):
                r = c.post(f"/api/desk/{did}/move", json=body)
                check(r.status_code == 422, f"{why} is rejected by the bounds",
                      f"HTTP {r.status_code}")
            for body, why in (({"prompts": ["a", "b"]}, "k=2"),
                              ({"prompts": P, "dalpha": 0.3}, "dalpha 0.3")):
                r = c.post("/api/desk/start", json=body)
                check(r.status_code == 422, f"/start with {why} -> 422", f"HTTP {r.status_code}")

            P8 = [f"q{i}" for i in range(8)]
            r = c.post("/api/desk/start", json={"prompts": P8, "dalpha": 0.02, "refine": True})
            want = dk.projected_cost(8, 51, True, 4)[0]
            check(r.status_code == 400 and "ceiling" in r.text and f"{want:.1f}" in r.text,
                  "k = 8 at dalpha 0.02 with refine -> 400 with the projection", r.text[:90])
            r = c.post("/api/desk/start", json={"prompts": P8, "dalpha": 0.02})
            check(r.status_code == 200, "... and the same without refine starts")
            did8 = r.json().get("desk_id", "")
            _wait(c, did8)
            dk.MAX_SESSION_IMAGE_EQ = 210
            r = c.post(f"/api/desk/{did8}/move", json={"line": 1, "alpha": 0.5})
            check(r.status_code == 400 and "has spent" in r.text,
                  "a move past the desk's session ceiling -> 400 naming what is spent",
                  r.text[:90])
            dk.MAX_SESSION_IMAGE_EQ = orig_ses

            # refine through the router
            r = c.post(f"/api/desk/{did}/move", json={"w0": w0, "refine": True})
            jr = _wait(c, did)
            fl = [f for ln in jr["lines"] for f in ln["flips"]]
            check(r.status_code == 200 and jr["refine"] and fl
                  and all(f["refined"] and abs(f["width"] - 0.05 / 8) < 1e-12 for f in fl)
                  and all(f["img_lo"] >= 0 and f["img_hi"] >= 0 for f in fl),
                  "refine through /move: every change bisected to dalpha/8 with full-fidelity "
                  "images either side", f"{len(fl)} changes")

            check(c.post(f"/api/desk/{did}/cancel").json() == {"ok": True}
                  and c.post("/api/desk/nope/cancel").status_code == 200,
                  "/cancel answers ok, also for an unknown desk")
            check(c.get("/api/desk/nope/status").json()["status"] == "unknown",
                  "an unknown desk answers status 'unknown', not a 500")
            img = jr["w0_image"]
            r = c.get(f"/api/desk/{did}/image/{img}")
            check(r.status_code == 200 and r.content == f"img{img}".encode(),
                  "the mix's image is served from the thumbnail store")
            check(c.get(f"/api/desk/{did}/image/999999").status_code == 404
                  and c.get("/api/desk/nope/image/0").status_code == 404,
                  "an unknown image or desk -> 404")
    finally:
        cs.evaluate = orig_eval
        dk.MAX_REQUEST_IMAGE_EQ, dk.MAX_SESSION_IMAGE_EQ = orig_req, orig_ses
        tmp.cleanup()


if __name__ == "__main__":
    part1()
    part2()
    part3()
    print(f"\n{'ALL CHECKS PASS' if not fails else 'FAILED: ' + ', '.join(fails)}")
    sys.exit(1 if fails else 0)
