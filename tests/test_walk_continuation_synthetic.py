"""Offline check of the two walk modes on a synthetic k = 4 field with a known curved front (no GPU).

The renderer is mocked: the "image embedding" at weights w is one of two orthogonal side vectors, chosen by the
sign of f(w) = |w - c|^2 - r^2 (a sphere in barycentric space: a front that curves everywhere), plus a smooth
location-dependent component (so side content changes along the front, as h07 found for real fronts). A walk
station is on the front iff |f| is small there. Run: python tests/test_walk_continuation_synthetic.py
"""
import sys, types
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import backend.services.cascade as cs

K = 4
C = np.array([0.55, 0.15, 0.15, 0.15])
R = 0.22
rng_g = np.random.default_rng(0)
A_side, B_side = np.eye(64)[0], np.eye(64)[1]
G = rng_g.standard_normal((K, 64)); G[:, :2] = 0; G /= np.linalg.norm(G, axis=1, keepdims=True)


def f(w):
    return float(np.sum((w - C) ** 2) - R ** 2)


SOFT = [0.0]                             # transition half-width in |w - c| units; 0 = a sharp front


def emb(w):
    if SOFT[0] > 0:                      # soft front: the side mix changes smoothly over ~2 * SOFT
        g = 1.0 / (1.0 + np.exp(-(np.sqrt(np.sum((w - C) ** 2)) - R) / (SOFT[0] / 3)))
        side = (1 - g) * A_side + g * B_side
    else:
        side = A_side if f(w) < 0 else B_side
    e = side + 0.35 * (w @ G)            # content drifts with position along the front
    return e / np.linalg.norm(e)


def make_run():
    run = cs.CascadeRun(run_id="syn", prompts=[f"p{i}" for i in range(K)], seed=42, steps=8, height=64, width=64,
                        guidance_scale=4.0, n_chords=0, n_patches=0)
    run.thumbs = {}
    return run


def fake_evaluate(app, run, pool, weights, seed, label, ctl=None, on_arrival=None, steps=None):
    out = []
    for w in weights:
        gi = run.next_idx; run.next_idx += 1
        run.embeddings[gi] = emb(np.asarray(w, dtype=float))
        out.append(gi)
    return out


cs.evaluate = fake_evaluate


def crossing(run, rng):
    """A random chord through the sphere's shell, bisected to the front -- like the cascade's tier 1."""
    while True:
        w0 = rng.dirichlet(np.ones(K))
        u = rng.standard_normal(K); u -= u.mean(); u /= np.linalg.norm(u)
        ts = np.linspace(-0.4, 0.4, 161)
        pts = [w0 + t * u for t in ts]
        ok = [p.min() >= 0 for p in pts]
        sg = [f(p) < 0 for p in pts]
        for i in range(len(ts) - 1):
            if ok[i] and ok[i + 1] and sg[i] != sg[i + 1]:
                a, b = pts[i], pts[i + 1]
                for _ in range(30):
                    m = (a + b) / 2
                    if (f(m) < 0) == (f(a) < 0):
                        a = m
                    else:
                        b = m
                mid = (a + b) / 2
                if mid.min() < 0.05:
                    break
                x = cs.Crossing(cid=len(run.crossings), wa=a, wb=b, mid=mid, n=u.copy())
                x.ea = emb(np.clip(mid - cs.EPS / 2 * u, 0, None))
                x.eb = emb(np.clip(mid + cs.EPS / 2 * u, 0, None))
                run.crossings.append(x)
                return x


def arclength(x0, steps, cap):
    pts = [np.asarray(x0)] + [np.asarray(s["weights"]) for s in steps]
    return min(cap, float(sum(np.linalg.norm(b - a) for a, b in zip(pts, pts[1:]))))


def main(n=60, n_steps=6):
    rng = np.random.default_rng(1)
    run = make_run()
    rows = []
    for _ in range(n):
        x = crossing(run, rng)
        for d in (1, -1):
            res = {}
            for mode in ("fan", "continuation"):
                w = cs.Walk(wid=f"{mode}{x.cid}{d}", cid=x.cid, direction=d, n_steps=n_steps, mode=mode)
                cs.run_walk(None, run, types.SimpleNamespace(n_gpus=6), w)
                off = [abs(f(np.asarray(s["center"]))) for s in w.steps]
                res[mode] = dict(arc=arclength(x.mid, w.steps, n_steps * cs.WALK_STEP), n=len(w.steps),
                                 img=w.images, max_f=max(off) if off else 0.0, note=(w.notes or [""])[-1])
            rows.append(res)
    fa = np.array([r["fan"]["arc"] for r in rows]); pc = np.array([r["continuation"]["arc"] for r in rows])
    print(f"{len(rows)} walks on a curved synthetic front (radius {R}, transition half-width {SOFT[0]})")
    print(f"  mean captured arclength  fan {fa.mean():.3f}   continuation {pc.mean():.3f}   (cap {n_steps * cs.WALK_STEP:.2f})")
    print(f"  continuation longer in {np.sum(pc > fa + 0.01)} walks, shorter in {np.sum(pc < fa - 0.01)}")
    print(f"  images/walk  fan {np.mean([r['fan']['img'] for r in rows]):.1f}   continuation {np.mean([r['continuation']['img'] for r in rows]):.1f}")
    mf = [r["continuation"]["max_f"] for r in rows if r["continuation"]["n"]]
    mff = [r["fan"]["max_f"] for r in rows if r["fan"]["n"]]
    # |f| = |d^2 - r^2| ~ 2 r * distance from the sphere -> distance ~ |f| / (2r)
    print(f"  max distance of a station from the true front: continuation {max(mf) / (2 * R):.4f}, fan {max(mff) / (2 * R) if mff else 0:.4f}")
    from collections import Counter
    for mode in ("fan", "continuation"):
        print(f"  {mode} end notes:", Counter(r[mode]["note"].split(" at ")[0] if r[mode]["note"] else "budget reached" for r in rows).most_common())
    assert pc.mean() > fa.mean(), "continuation should follow a curved front further than the straight fan"
    assert max(mf) / (2 * R) < 0.02 + SOFT[0], "continuation stations must sit on the front"


if __name__ == "__main__":
    main()
    SOFT[0] = 0.04                       # dev traces: real fronts spread their change over 2-4 spacings
    main()
