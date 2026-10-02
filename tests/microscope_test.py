"""Offline checks of the Ridge microscope (backend/services/microscope.py + its router). No GPU.

Three parts, all analytic or stubbed -- nothing renders:

  (i)   GEOMETRY -- every plane mode (random, prompt-swap, crossing with and without a hi-res
        cloud) gives two ORTHONORMAL SUM-ZERO directions at k = 3..8; prompt-swap's e1 is
        exactly (e_i - e_j)/sqrt(2) and a pair that adds no direction is refused; the crossing
        plane takes the cloud normal over the chord, and its e2 is the principal direction of a
        BALANCED cloud set (re-computed independently here from the first n of each side) or a
        seeded random direction when the cloud is missing or one-sided; "re-draw e2" moves e2
        and keeps e1. The lattice: G^2 cells, the centre cell IS w0, every cell sums to 1,
        neighbours step by exactly s*2/(G-1)*e1, corners sit s*sqrt(2) out; cells outside the
        simplex are flagged exactly when a weight is negative and the cost counts only the
        inside ones. The biplot arrow of prompt i equals the projection of e_i - 1/k computed
        the long way. Zoom: the target is the clicked cell's recipe at s/2, and an interior
        click reuses ((G+1)/2)^2 of the parent's cells (9 of 25 at G = 5).
  (ii)  RENDER -- render_level over a synthetic TWO-BASIN field (two sides of a plane joined
        through a sharp tanh sheet, plus a low-frequency background) with cascade.evaluate
        stubbed as tests/metro_test.py stubs it. The plane is laid across the sheet so the
        truth is known: the boundary edges must be exactly the edges whose two cells lie on
        opposite sides; one batch per level, only the inside cells, all at FULL fidelity; a
        zoom renders only the cells the parent did not, and is charged only for those; a
        level cancelled mid-batch stays cancelled and charges nothing.
  (iii) ROUTER through FastAPI's TestClient with the same stub, live (the level renders on the
        router's own worker thread): /plan for all three modes and every way crossing mode can
        be refused (400 / 404 / 409), /start -> /status to complete, /zoom (new, cached,
        outside 400, unknown 404, below the smallest half-width 400, past the ceiling 400 with
        the projection), the per-cell zoom cost, /cancel and the image route.

Run: python tests/microscope_test.py
"""
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.services import microscope as ms           # noqa: E402
from backend.services import cascade as cs              # noqa: E402
from backend.services import gridfree as gf             # noqa: E402
from backend.services.cascade import COS_T, _cosd       # noqa: E402

fails = []


def check(ok, label, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(label)


def _orthonormal_sum_zero(e1, e2):
    e1, e2 = np.asarray(e1), np.asarray(e2)
    return max(abs(float(e1.sum())), abs(float(e2.sum())),
               abs(float(np.linalg.norm(e1)) - 1.0), abs(float(np.linalg.norm(e2)) - 1.0),
               abs(float(np.dot(e1, e2))))


# ------------------------------------------------------------------------- (i) geometry
def part1():
    print("\n(i) planes, lattice, simplex clipping, biplot, zoom, cost -- as arithmetic")
    worst = 0.0
    for k in range(3, 9):
        for seed in (0, 7, -3):
            e1, e2, _ = ms.plane_random(k, seed, 0)
            worst = max(worst, _orthonormal_sum_zero(e1, e2))
    check(worst < 1e-12, "random planes are orthonormal and sum-zero at k = 3..8",
          f"worst defect {worst:.1e}")
    a1, a2, _ = ms.plane_random(5, 42, 0)
    b1, b2, _ = ms.plane_random(5, 42, 1)
    c1, c2, _ = ms.plane_random(5, 42, 0)
    check(np.allclose(a1, b1, atol=0) and not np.allclose(a2, b2, atol=1e-6)
          and np.array_equal(a1, c1) and np.array_equal(a2, c2)
          and _orthonormal_sum_zero(b1, b2) < 1e-12,
          "re-draw e2 moves e2 and keeps e1; the same (seed, redraw) gives the same plane",
          f"|cos(e2, e2')| {abs(float(np.dot(a2, b2))):.2f}")

    worst = 0.0
    for k in range(3, 9):
        e1, e2, _ = ms.plane_prompt_swap(k, (0, 1), (2, 0))
        want = np.zeros(k)
        want[0], want[1] = 1 / math.sqrt(2), -1 / math.sqrt(2)
        worst = max(worst, _orthonormal_sum_zero(e1, e2), float(np.abs(e1 - want).max()))
    check(worst < 1e-12, "prompt-swap: e1 = (e_i - e_j)/sqrt(2) exactly, e2 Gram-Schmidt'ed, "
          "both orthonormal and sum-zero at k = 3..8", f"worst defect {worst:.1e}")
    bad = 0
    for pa, pb in (((0, 1), (1, 0)), ((0, 1), (0, 1)), ((0, 0), (1, 2)), ((0, 5), (1, 2))):
        try:
            ms.plane_prompt_swap(4, pa, pb)
        except ValueError:
            bad += 1
    check(bad == 4, "a pair that adds no direction, trades a prompt for itself, or is out of "
          "range is refused with a reason", f"{bad}/4 refused")

    # --- the crossing plane
    k = 5
    rng = np.random.default_rng(3)
    c = np.full(k, 0.2)
    n = ms.tangent_unit(rng.standard_normal(k))
    d1 = ms.orthogonalise(rng.standard_normal(k), [n])
    d2 = ms.orthogonalise(rng.standard_normal(k), [n, d1])
    side_a = [c - 0.02 * n + 0.04 * rng.standard_normal() * d1 for _ in range(10)]
    side_b = [c + 0.02 * n + 0.04 * rng.standard_normal() * d2 for _ in range(3)]
    pts = ([[list(w), "A", i, 0.1] for i, w in enumerate(side_a)]
           + [[list(w), "B", 20 + i, 0.1] for i, w in enumerate(side_b)]
           + [[list(c), "other", 99, 0.9]])
    wa, wb = c - 0.0125 * d1, c + 0.0125 * d1           # a chord that is NOT the normal
    e1, e2, note = ms.plane_crossing(k, wa, wb, cloud_normal=list(n), cloud_pts=pts, seed=1)
    # independent PCA of the balanced set: first 3 of A + first 3 of B, centred, e1 removed
    X = np.vstack([np.stack(side_a[:3]), np.stack(side_b[:3])])
    X = X - X.mean(axis=0)
    X = X - np.outer(X @ n, n)
    v = np.linalg.svd(X)[2][0]
    v = v - np.dot(v, n) * n
    v /= np.linalg.norm(v)
    check(_orthonormal_sum_zero(e1, e2) < 1e-12 and np.allclose(e1, n, atol=1e-12),
          "crossing plane: e1 is the cloud normal (it outranks the chord), plane orthonormal")
    check(abs(abs(float(np.dot(e2, v))) - 1.0) < 1e-9 and "balanced" in note and "3 side-A" in note,
          "e2 = first principal direction of a BALANCED set (3 + 3 of 10 A / 3 B), "
          "orthogonal to e1 -- matches an independent PCA", f"|cos| {abs(float(np.dot(e2, v))):.12f}")
    Xu = np.vstack([np.stack(side_a), np.stack(side_b)])
    Xu = Xu - Xu.mean(axis=0)
    Xu = Xu - np.outer(Xu @ n, n)
    vu = np.linalg.svd(Xu)[2][0]
    check(abs(float(np.dot(vu, v))) < 0.9,
          "... and balancing matters here: the unbalanced 13-point PCA points elsewhere",
          f"|cos(balanced, unbalanced)| {abs(float(np.dot(vu, v))):.2f}")
    f1, f2, fnote = ms.plane_crossing(k, wa, wb, cloud_normal=None, cloud_pts=None, seed=1,
                                      redraw=0, cid=4)
    g1, g2, _ = ms.plane_crossing(k, wa, wb, seed=1, redraw=1, cid=4)
    check(np.allclose(f1, ms.tangent_unit(wb - wa), atol=1e-12)
          and _orthonormal_sum_zero(f1, f2) < 1e-12 and "random" in fnote
          and np.array_equal(f1, g1) and not np.allclose(f2, g2, atol=1e-6),
          "no cloud: e1 = the chord direction, e2 a seeded random tangent direction that "
          "re-draws on request", fnote[:70])
    h1, h2, hnote = ms.plane_crossing(k, wa, wb, cloud_normal=list(n),
                                      cloud_pts=[p for p in pts if p[1] != "B"], seed=1)
    check(np.allclose(h1, n, atol=1e-12) and "one-sided" in hnote
          and _orthonormal_sum_zero(h1, h2) < 1e-12,
          "a one-sided cloud keeps its normal but cannot name e2 (random, and says why)",
          hnote[-40:])
    try:
        ms.plane_crossing(k, c, c)
        collapsed = False
    except ValueError:
        collapsed = True
    check(collapsed, "a crossing with no direction at all is refused, not given a fake plane")

    # --- the lattice
    for G in ms.GRID_CHOICES:
        w0 = np.array([0.3, 0.25, 0.25, 0.2])
        e1, e2, _ = ms.plane_random(4, 5, 0)
        s = 0.08
        cells = ms.lattice(w0, e1, e2, s, G)
        mid = cells[(G // 2) * G + G // 2]
        by = {(x["ia"], x["ib"]): np.asarray(x["w"]) for x in cells}
        step = max(float(np.abs(by[(ia + 1, ib)] - by[(ia, ib)] - s * 2 / (G - 1) * e1).max())
                   for ia in range(G - 1) for ib in range(G))
        check(len(cells) == G * G and np.allclose(mid["w"], w0, atol=1e-15)
              and mid["a"] == 0.0 and mid["b"] == 0.0
              and max(abs(sum(x["w"]) - 1.0) for x in cells) < 1e-12 and step < 1e-15
              and abs(float(np.linalg.norm(by[(G - 1, G - 1)] - w0)) - s * math.sqrt(2)) < 1e-12,
              f"G={G}: {G * G} cells, centre cell = w0, all sum to 1, neighbours step by "
              f"s*2/(G-1)*e1, corners s*sqrt(2) out", f"step defect {step:.1e}")
    # near a vertex the lattice leaves the simplex
    w0 = np.array([0.9, 0.05, 0.05])
    e1, e2, _ = ms.plane_random(3, 2, 0)
    cells = ms.lattice(w0, e1, e2, 0.1, 5)
    n_in = sum(1 for x in cells if x["inside"])
    check(0 < n_in < 25 and all(x["inside"] == (min(x["w"]) >= -1e-9) for x in cells)
          and ms.projected_cost(cells) == n_in * ms.IMAGE_COST,
          "near a vertex some cells fall outside: flagged exactly when a weight is negative, "
          "and the cost counts only the inside ones", f"{n_in}/25 inside")

    # --- the biplot
    worst = 0.0
    for k in range(3, 9):
        e1, e2, _ = ms.plane_random(k, k, 0)
        bp = np.asarray(ms.biplot(e1, e2))
        for i in range(k):
            t = np.eye(k)[i] - 1.0 / k
            worst = max(worst, abs(bp[i, 0] - float(t @ e1)), abs(bp[i, 1] - float(t @ e2)))
        worst = max(worst, float(np.abs(bp.sum(axis=0)).max()))
    check(worst < 1e-12, "biplot arrow i = projection of e_i - 1/k onto (e1, e2), "
          "and the k arrows sum to zero", f"worst defect {worst:.1e}")

    # --- zoom arithmetic and cache reuse
    # zoom point j (of -(G-1)/2..(G-1)/2) sits at s*j/(G-1) from the click, the parent's
    # lattice at multiples of 2s/(G-1): the even j coincide -- 1, 3, 3 per axis at G = 3, 5, 7
    for G, reuse in ((3, 1), (5, 9), (7, 9)):
        k = 4
        e1, e2, _ = ms.plane_random(k, 9, 0)
        w0 = np.full(k, 0.25)
        cells = ms.lattice(w0, e1, e2, 0.08, G)
        ia = ib = G // 2 + 1 if G > 3 else 1
        tgt = next(x for x in cells if x["ia"] == ia and x["ib"] == ib)
        centre, s2 = ms.zoom_target(w0, 0.08, G, ia, ib, e1, e2)
        zc = ms.lattice(centre, e1, e2, s2, G)
        parent = {gf.point_key(x["w"], 8) for x in cells}
        shared = sum(1 for x in zc if gf.point_key(x["w"], 8) in parent)
        check(np.allclose(centre, tgt["w"], atol=1e-15) and s2 == 0.04
              and np.allclose(zc[(G // 2) * G + G // 2]["w"], tgt["w"], atol=1e-15)
              and shared == reuse,
              f"G={G}: a zoom is centred on the clicked cell at s/2, and shares "
              f"(2*floor((G-1)/4) + 1)^2 = {reuse} cells with its parent", f"{shared} shared")
    check(ms.level_key([0.25] * 4, 0.1) == ms.level_key([0.25 + 1e-13] * 4, 0.1)
          and ms.level_key([0.25] * 4, 0.1) != ms.level_key([0.25] * 4, 0.05),
          "levels are keyed by (centre, s), deaf to float round-off")

    # --- boundary edges on a known label map
    G = 5
    ea, eb = np.zeros(8), np.zeros(8)
    ea[0], eb[1] = 1.0, 1.0
    lab = {(ia, ib): (ia + ib >= 4) for ia in range(G) for ib in range(G)}
    emb = {key: (eb if v else ea) for key, v in lab.items()}
    del emb[(0, 0)]                                       # one cell never arrived
    edges = ms.boundary_edges(G, emb)
    want = {((ia, ib), (ja, jb)) for (ia, ib) in lab for (ja, jb) in ((ia + 1, ib), (ia, ib + 1))
            if (ja, jb) in lab and (ia, ib) in emb and lab[(ia, ib)] != lab[(ja, jb)]}
    got = {(tuple(e["a"]), tuple(e["b"])) for e in edges if e["boundary"]}
    check(got == want and len(edges) == 2 * G * (G - 1) - 2
          and all(abs(e["div"] - (1.0 if e["boundary"] else 0.0)) < 1e-12 for e in edges),
          "boundary edges = the 4-neighbour pairs past COS_T, and a missing cell drops only "
          "its own edges", f"{len(got)} boundaries of {len(edges)} edges")


# --------------------------------------------- (ii) render_level on a synthetic two-basin field
def _two_basin(k, seed=1, band=0.003, amp=2.5, bg=0.2, freq=2.5, modes=3):
    """tests/metro_test.py's field at any k: a sharp tanh sheet across the plane
    <w - 1/k, p> = 0, plus a quiet low-frequency background. Returns (field, p)."""
    rng = np.random.default_rng(seed)
    p = ms.tangent_unit(rng.standard_normal(k))
    Q = np.linalg.qr(rng.standard_normal((8, 2 + modes)))[0]
    a, b, Cvec = Q[:, 0], Q[:, 1], Q[:, 2:]
    Cdir = rng.standard_normal((modes, k))
    Cdir -= Cdir.mean(axis=1, keepdims=True)
    Cdir /= np.linalg.norm(Cdir, axis=1, keepdims=True)
    ph = rng.uniform(0.0, 2.0 * math.pi, modes)

    def f(w):
        x = np.asarray(w, dtype=float) - 1.0 / k
        e = a + amp * math.tanh(float(x @ p) / band) * b + bg * (Cvec @ np.sin(freq * (Cdir @ x) + ph))
        return e / float(np.linalg.norm(e))
    return f, p


def _stub(run, field, calls, cancel_after=None):
    """Stand-in for cascade.evaluate (tests/metro_test.py's shape): embeds through the field,
    keeps the live-cloud bookkeeping, calls on_arrival per point, records (label, n, steps).
    With cancel_after=j, the j-th call flips its ctl to cancelled and returns nothing."""
    def stub(app, r, pool, weights, seed, label, ctl=None, on_arrival=None, steps=None):
        calls.append((label.split(".")[0], len(weights), steps))
        if cancel_after is not None and len(calls) >= cancel_after:
            ctl.status = "cancelled"
            return [None] * len(weights)
        out = []
        for li, w in enumerate(weights):
            gi = r.next_idx
            r.next_idx += 1
            r.embeddings[gi] = field(w)
            r.generated += 1
            if r.thumbs is not None:
                r.thumbs[gi] = f"img{gi}".encode()
            if on_arrival is not None:
                on_arrival(li, gi)
            out.append(gi)
        return out
    return stub


def _mk_run(k, e1, e2, grid=5):
    return ms.MicroRun(run_id="m", prompts=[f"p{i}" for i in range(k)], seed=1, steps=8,
                       height=64, width=64, guidance_scale=3.5, mode="crossing",
                       e1=[float(v) for v in e1], e2=[float(v) for v in e2], grid=grid)


def part2():
    print("\n(ii) render_level on a synthetic two-basin field (stubbed, no GPU)")
    k = 4
    field, p = _two_basin(k)
    e1 = p
    e2 = ms.random_tangent(k, np.random.default_rng(5), against=[p])
    g0 = 0.0125
    w0 = np.full(k, 1.0 / k) + g0 * p                     # the sheet sits a quarter-step left
    run = _mk_run(k, e1, e2)
    calls = []
    orig = cs.evaluate
    cs.evaluate = _stub(run, field, calls)
    try:
        lv, created = ms.add_level(run, w0, 0.10)
        ms.render_level(None, run, None, lv)
        inside = [c for c in lv.cells if c["inside"]]
        check(created and lv.status == "complete" and len(calls) == 1
              and calls[0] == ("scope0", len(inside), None),
              "one batch per level, only the inside cells, at FULL fidelity (steps None)",
              f"{calls[0]}, {len(inside)}/{len(lv.cells)} inside")
        check(all(c["image"] >= 0 for c in inside)
              and all(c["image"] == -1 for c in lv.cells if not c["inside"]),
              "every inside cell got its image; outside cells stay -1")

        def side(w):
            return float((np.asarray(w) - 1.0 / k) @ p) > 0

        by = {(c["ia"], c["ib"]): c for c in inside}
        truth = {((ia, ib), (ja, jb)) for (ia, ib) in by for (ja, jb) in ((ia + 1, ib), (ia, ib + 1))
                 if (ja, jb) in by and side(by[(ia, ib)]["w"]) != side(by[(ja, jb)]["w"])}
        got = {(tuple(e["a"]), tuple(e["b"])) for e in lv.edges if e["boundary"]}
        margin = min(abs(float((np.asarray(c["w"]) - 1.0 / k) @ p)) for c in inside)
        check(got == truth and len(truth) > 0,
              "the boundary edges are exactly the neighbour pairs on opposite sides of the sheet",
              f"{len(got)} boundary edges (nearest cell {margin:.4f} from the sheet, band 0.003)")
        check(run.n_images == len(inside) and run.cost_image_eq == len(inside) * ms.IMAGE_COST
              and lv.n_new == len(inside) and lv.n_reused == 0,
              "cost = one image-eq per rendered inside cell", f"{run.cost_image_eq:.0f} image-eq")

        # zoom into the centre cell: s/2, same plane, the parent's shared cells reused
        centre, s2 = ms.zoom_target(lv.centre, lv.s, lv.grid, 2, 2, run.e1, run.e2)
        lv1, created1 = ms.add_level(run, centre, s2, parent=0, parent_cell=[2, 2])
        before = run.n_images
        ms.render_level(None, run, None, lv1)
        n_in1 = sum(1 for c in lv1.cells if c["inside"])
        check(created1 and lv1.level == 1 and lv1.parent == 0 and lv1.s == 0.05
              and calls[1][1] == n_in1 - 9 and lv1.n_reused == 9
              and run.n_images - before == n_in1 - 9,
              "a zoom renders only the cells its parent did not (9 of 25 reused at G = 5) and is "
              "charged only for those", f"batch {calls[1][1]}, reused {lv1.n_reused}")
        got1 = {(tuple(e["a"]), tuple(e["b"])) for e in lv1.edges if e["boundary"]}
        check(got1 == {((1, ib), (2, ib)) for ib in range(5)},
              "at s/2 the sheet still runs between columns 1 and 2 (it sits g0 = s/8 left of "
              "the centre)", f"{sorted(got1)[:2]}...")
        same, created2 = ms.add_level(run, centre + 1e-13, s2)
        check(same is lv1 and not created2 and len(run.levels) == 2,
              "the same (centre, s) is the same level: zooming there again renders nothing")
        cells2 = ms.lattice(centre, run.e1, run.e2, s2, 5)
        check(ms.projected_cost(cells2, run) == 0.0
              and ms.projected_cost(cells2) == n_in1 * ms.IMAGE_COST,
              "the projected cost reads the render cache: a lattice already rendered costs 0")
    finally:
        cs.evaluate = orig

    # a lattice reaching outside the simplex: the outside cells never reach the batch
    run = _mk_run(3, *ms.plane_random(3, 2, 0)[:2])
    calls = []
    cs.evaluate = _stub(run, _two_basin(3)[0], calls)
    try:
        lv, _ = ms.add_level(run, [0.9, 0.05, 0.05], 0.1)
        ms.render_level(None, run, None, lv)
        n_in = sum(1 for c in lv.cells if c["inside"])
        check(0 < n_in < 25 and calls[0][1] == n_in and run.n_images == n_in
              and all(min(np.asarray(c["w"])) >= -1e-9 for c in lv.cells if c["image"] >= 0),
              "outside cells are not rendered and not charged",
              f"{n_in} rendered of 25 ({25 - n_in} outside)")
    finally:
        cs.evaluate = orig

    # cancellation mid-batch: nothing charged, the level stays cancelled
    k = 4
    run = _mk_run(k, *ms.plane_random(k, 3, 0)[:2])
    calls = []
    cs.evaluate = _stub(run, _two_basin(k)[0], calls, cancel_after=1)
    try:
        lv, _ = ms.add_level(run, np.full(k, 0.25), 0.05)
        ms.render_level(None, run, None, lv)
        check(lv.status == "cancelled" and run.n_images == 0 and lv.edges == []
              and all(c["image"] == -1 for c in lv.cells) and run.status == "cancelled",
              "a level cancelled mid-batch stays cancelled and charges nothing",
              f"{lv.status}, {run.n_images} images")
    finally:
        cs.evaluate = orig


# ------------------------------------------------------------------------- (iii) the router
def _fake_cascade(k=4, status="complete", cloud=True):
    run = cs.CascadeRun(run_id="c", prompts=[f"q{i}" for i in range(k)], seed=7, steps=8,
                        height=64, width=64, guidance_scale=3.5, n_chords=4, n_patches=1)
    run.status = status
    rng = np.random.default_rng(0)
    mid = np.full(k, 1.0 / k)
    u = ms.tangent_unit(rng.standard_normal(k))
    n = ms.tangent_unit(rng.standard_normal(k))
    x = cs.Crossing(cid=0, wa=mid - 0.0125 * u, wb=mid + 0.0125 * u, mid=mid, n=u)
    if cloud:
        pts = []
        for j in range(12):
            w = mid + 0.04 * ms.tangent_unit(rng.standard_normal(k))
            pts.append([list(w), "A" if float((w - mid) @ n) < 0 else "B", 100 + j, 0.2])
        x.cloud = {"normal": list(n), "n": 12, "r": 0.05}
        x.cloud_pts = pts
    run.crossings = [x, cs.Crossing(cid=1, wa=mid - 0.0125 * n, wb=mid + 0.0125 * n, mid=mid)]
    return run, n, u


def _wait(c, rid, level=None, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        j = c.get(f"/api/microscope/{rid}/status"
                  + ("" if level is None else f"?level={level}")).json()
        if j["status"] != "running":
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
    from backend.routers import microscope as router_mod
    from backend.cache.thumbnail_cache import ThumbnailCache

    fields = {}                                       # one synthetic field per k
    calls = []
    orig_eval, orig_max, orig_smin = cs.evaluate, ms.MAX_IMAGE_EQ, ms.S_MIN

    def stub(app, r, pool, weights, seed, label, ctl=None, on_arrival=None, steps=None):
        field = fields.setdefault(r.k, _two_basin(r.k)[0])
        return _stub(r, field, calls)(app, r, pool, weights, seed, label, ctl, on_arrival,
                                     steps)
    cs.evaluate = stub
    tmp = tempfile.TemporaryDirectory()
    try:
        app = FastAPI()
        app.include_router(router_mod.router)
        good, n_cloud, u_chord = _fake_cascade(4)
        app.state.cascades = {"good": good, "busy": _fake_cascade(4, status="running")[0]}
        app.state.cache = ThumbnailCache(Path(tmp.name))
        app.state.gpu_pool = None
        with TestClient(app, raise_server_exceptions=False) as c:
            P = ["a", "b", "c", "d"]
            r = c.post("/api/microscope/plan", json={"prompts": P, "grid": 5})
            j = r.json()
            check(r.status_code == 200 and j["grid"] == 5 and len(j["cells"]) == 25
                  and j["cost_image_eq"] == j["n_inside"] == 25 and j["mode"] == "random"
                  and _orthonormal_sum_zero(j["e1"], j["e2"]) < 1e-12
                  and len(j["biplot"]) == 4 and j["centre"] == [0.25] * 4,
                  "/plan (random, barycentre): 25 cells, cost = in-simplex cells, "
                  "orthonormal plane, one biplot arrow per prompt", f"HTTP {r.status_code}")
            check(not calls, "/plan renders nothing")
            r = c.post("/api/microscope/plan",
                       json={"prompts": P, "mode": "prompt-swap", "swap_a": [1, 3],
                             "swap_b": [0, 2], "grid": 3, "s": 0.05})
            j = r.json()
            want = np.zeros(4)
            want[1], want[3] = 1 / math.sqrt(2), -1 / math.sqrt(2)
            check(r.status_code == 200 and np.allclose(j["e1"], want, atol=1e-12)
                  and len(j["cells"]) == 9 and "P2 up / P4 down" in j["plane_note"],
                  "/plan prompt-swap: e1 = (e_2 - e_4)/sqrt(2)", j.get("plane_note", "")[:60])
            for body, why in (({"mode": "prompt-swap"}, "no pairs"),
                              ({"mode": "prompt-swap", "swap_a": [0, 1], "swap_b": [1, 0]},
                               "a parallel second pair"),
                              ({"centre": [0.5, 0.5, 0.5, -0.5]}, "a centre that is no recipe"),
                              ({"centre": [0.5, 0.5]}, "a centre of the wrong length")):
                r = c.post("/api/microscope/plan", json=dict({"prompts": P}, **body))
                check(r.status_code == 400 and len(r.json().get("detail", "")) > 10,
                      f"{why} -> 400 with the reason", f"HTTP {r.status_code}")
            for body, why in (({"grid": 4}, "grid 4"), ({"s": 0.0}, "s 0"), ({"s": 0.6}, "s 0.6"),
                              ({"mode": "lasso"}, "an unknown mode")):
                r = c.post("/api/microscope/plan", json=dict({"prompts": P}, **body))
                check(r.status_code == 422, f"{why} is rejected by the bounds",
                      f"HTTP {r.status_code}")
            r = c.post("/api/microscope/plan", json={"prompts": ["a", "b"]})
            check(r.status_code == 422, "k=2 is rejected", f"HTTP {r.status_code}")

            # --- crossing mode, and every way it cannot be honoured
            r = c.post("/api/microscope/plan",
                       json={"mode": "crossing", "cascade_run_id": "good", "cid": 0})
            j = r.json()
            check(r.status_code == 200 and np.allclose(j["e1"], n_cloud, atol=1e-12)
                  and "balanced" in j["plane_note"] and j["prompts"] == good.prompts
                  and j["seed"] == 7 and j["source_cid"] == 0
                  and np.allclose(j["centre"], [0.25] * 4),
                  "/plan crossing: the run's prompts and seed, centre = the crossing midpoint, "
                  "e1 = its cloud normal, e2 from the balanced cloud", j.get("plane_note", "")[:70])
            r = c.post("/api/microscope/plan",
                       json={"mode": "crossing", "cascade_run_id": "good", "cid": 1})
            j = r.json()
            check(r.status_code == 200 and "random" in j["plane_note"]
                  and abs(abs(float(np.dot(j["e1"], good.crossings[1].wb - good.crossings[1].wa))
                              / np.linalg.norm(good.crossings[1].wb - good.crossings[1].wa)) - 1)
                  < 1e-9,
                  "a crossing without a cloud: e1 = its chord, e2 random (and says so)")
            for body, code, why in (
                    ({"cid": 0}, 400, "no run id"),
                    ({"cascade_run_id": "nope", "cid": 0}, 404, "an unknown run"),
                    ({"cascade_run_id": "busy", "cid": 0}, 409, "a run still running"),
                    ({"cascade_run_id": "good", "cid": 9}, 404, "an unknown crossing"),
                    ({"cascade_run_id": "good", "cid": 0, "prompts": P}, 400,
                     "prompts that differ from the run's")):
                r = c.post("/api/microscope/plan", json=dict({"mode": "crossing"}, **body))
                check(r.status_code == code, f"crossing mode with {why} -> {code}",
                      f"HTTP {r.status_code}: {r.text[:70]}")

            # --- start -> status -> complete
            r = c.post("/api/microscope/start", json={"prompts": P, "grid": 5, "s": 0.1})
            rid = r.json().get("run_id", "")
            check(r.status_code == 200 and r.json()["cost_image_eq"] == 25,
                  "/start answers with the run id and the cost it is about to spend")
            j = _wait(c, rid)
            lv0 = j["levels"][0] if j.get("levels") else {}
            check(j["status"] == "complete" and lv0.get("status") == "complete"
                  and all(x["image"] >= 0 for x in lv0["cells"] if x["inside"])
                  and len(lv0["edges"]) == 40 and j["cost_image_eq"] == 25
                  and calls[-1][1:] == (25, None),
                  "the level renders on the router's worker thread at full fidelity; /status "
                  "serves its cells and its 40 neighbour edges",
                  f"{j['status']}, {len(lv0.get('edges', []))} edges, {j['cost_image_eq']} image-eq")
            check(all(x["zoom_cost"] == 16 for x in lv0["cells"]
                      if 0 < x["ia"] < 4 and 0 < x["ib"] < 4),
                  "every interior cell quotes its zoom cost: 25 - 9 reused = 16 new images")

            # --- zoom
            n_calls = len(calls)
            r = c.post(f"/api/microscope/{rid}/zoom", json={"level": 0, "ia": 2, "ib": 1})
            z = r.json()
            check(r.status_code == 200 and z["level"] == 1 and not z["cached"]
                  and z["cost_image_eq"] == 16, "/zoom opens level 1 and states its cost",
                  f"{z}")
            j = _wait(c, rid)
            lv1 = j["levels"][1]
            check(j["status"] == "complete" and lv1["parent"] == 0 and lv1["parent_cell"] == [2, 1]
                  and abs(lv1["s"] - 0.05) < 1e-15 and lv1["n_reused"] == 9 and lv1["n_new"] == 16
                  and calls[n_calls][1] == 16 and j["cost_image_eq"] == 41,
                  "the zoom level renders 16 new cells, reuses 9, and the run is charged 41",
                  f"reused {lv1['n_reused']}, cost {j['cost_image_eq']}")
            r = c.post(f"/api/microscope/{rid}/zoom", json={"level": 0, "ia": 2, "ib": 1})
            check(r.status_code == 200 and r.json()["cached"] and r.json()["level"] == 1
                  and len(calls) == n_calls + 1,
                  "the same click again is a cached level: nothing rendered")
            j0 = c.get(f"/api/microscope/{rid}/status?level=0").json()
            zc = {(x["ia"], x["ib"]): x["zoom_cost"] for x in j0["levels"][0]["cells"]}
            check(zc[(2, 1)] == 0 and zc[(1, 1)] is not None and zc[(1, 1)] > 0,
                  "/status quotes 0 for a cell whose level already exists")
            r = c.post("/api/microscope/start",
                       json={"prompts": ["a", "b", "c"], "centre": [0.9, 0.05, 0.05],
                             "s": 0.1, "grid": 5})
            rid3 = r.json()["run_id"]
            j3 = _wait(c, rid3)
            out_cell = next(x for x in j3["levels"][0]["cells"] if not x["inside"])
            r = c.post(f"/api/microscope/{rid3}/zoom",
                       json={"level": 0, "ia": out_cell["ia"], "ib": out_cell["ib"]})
            check(r.status_code == 400 and "outside" in r.text,
                  "zooming into an outside cell -> 400", f"HTTP {r.status_code}")
            n3 = sum(1 for x in j3["levels"][0]["cells"] if x["inside"])
            check(j3["status"] == "complete" and 0 < n3 < 25 and j3["cost_image_eq"] == n3,
                  "a lattice reaching past a face is charged only its inside cells",
                  f"{j3['status']}, {n3} inside, {j3['cost_image_eq']} image-eq")
            for body, code, why in (({"level": 5, "ia": 0, "ib": 0}, 404, "an unknown level"),
                                    ({"level": 0, "ia": 9, "ib": 0}, 404, "a cell off the grid")):
                r = c.post(f"/api/microscope/{rid}/zoom", json=body)
                check(r.status_code == code, f"zoom into {why} -> {code}", f"HTTP {r.status_code}")
            r = c.post("/api/microscope/nope/zoom", json={"level": 0, "ia": 0, "ib": 0})
            check(r.status_code == 404, "zoom on an unknown run -> 404")
            ms.S_MIN = 0.06
            r = c.post(f"/api/microscope/{rid}/zoom", json={"level": 0, "ia": 1, "ib": 1})
            check(r.status_code == 400 and "half-width" in r.text,
                  "a zoom below the smallest half-width -> 400 with the reason")
            ms.S_MIN = orig_smin
            ms.MAX_IMAGE_EQ = 50
            r = c.post(f"/api/microscope/{rid}/zoom", json={"level": 0, "ia": 1, "ib": 3})
            # (1, 3)'s lattice also meets level 1's at one point, so it would be 15 new, not 16:
            # the refusal quotes the cache-aware number /status advertised for that cell
            check(r.status_code == 400 and "ceiling" in r.text
                  and f"{zc[(1, 3)]} new images" in r.text and "41" in r.text,
                  "a zoom past the run's ceiling -> 400 with the projection and what is spent",
                  r.text[:100])
            ms.MAX_IMAGE_EQ = 20
            r = c.post("/api/microscope/start", json={"prompts": P, "grid": 5})
            check(r.status_code == 400 and "ceiling" in r.text,
                  "a start past the ceiling is refused with its projection, as Metro refuses")
            ms.MAX_IMAGE_EQ = orig_max

            # --- cancel / unknown / images
            r = c.post(f"/api/microscope/{rid}/cancel")
            check(r.status_code == 200 and r.json() == {"ok": True},
                  "/cancel answers ok (on a finished run it is a no-op)")
            check(c.post("/api/microscope/nope/cancel").status_code == 200,
                  "/cancel on an unknown run is a no-op, not a 500")
            check(c.get("/api/microscope/nope/status").json()["status"] == "unknown",
                  "an unknown run answers status 'unknown', not a 500")
            img = lv0["cells"][0]["image"]
            r = c.get(f"/api/microscope/{rid}/image/{img}")
            check(r.status_code == 200 and r.content == f"img{img}".encode()
                  and r.headers["content-type"] == "image/jpeg",
                  "a rendered cell's image is served from the thumbnail store")
            check(c.get(f"/api/microscope/{rid}/image/99999").status_code == 404
                  and c.get("/api/microscope/nope/image/0").status_code == 404,
                  "an image the store never saw, or of an unknown run -> 404")
            r = c.post("/api/microscope/start",
                       json={"mode": "crossing", "cascade_run_id": "good", "cid": 0, "grid": 3})
            jc = _wait(c, r.json()["run_id"])
            check(jc["status"] == "complete" and jc["seed"] == 7 and jc["source_run"] == "good"
                  and jc["prompts"] == good.prompts and np.allclose(jc["e1"], n_cloud, atol=1e-12),
                  "a crossing-mode run renders the cascade run's own field (its seed and prompts)")
    finally:
        cs.evaluate, ms.MAX_IMAGE_EQ, ms.S_MIN = orig_eval, orig_max, orig_smin
        tmp.cleanup()


if __name__ == "__main__":
    part1()
    part2()
    part3()
    print(f"\n{'ALL CHECKS PASS' if not fails else 'FAILED: ' + ', '.join(fails)}")
    sys.exit(1 if fails else 0)
