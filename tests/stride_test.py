"""Offline checks of the per-run chord stride (backend/services/cascade.py). No GPU.

The stride is the chord probe spacing in weight space. 0.025 (~1 fine cell) is the
protocol of record; a run may now ask for 0.01-0.2 instead. Two parts:

  (i)  PURE -- _chord_offsets / _min_chord_len at strides 0.01 / 0.025 / 0.2: probe counts,
       spacing, the >= 3-probes floor the minimum chord length must guarantee, and a
       regression guard that the default stride still reproduces the old constants
       (6*STRIDE = 0.15 minimum, offsets tneg + i*STRIDE). Also pins PC_DS: the
       continuation walk's corrector spacing is NOT the per-run knob.
  (ii) REQUEST -- /api/cascade/start through FastAPI's TestClient with a stubbed app.state
       and cs.run_cascade replaced by a no-op, so the request is validated and the run
       object is built but nothing renders (as tests/local_sv_test.py stubs the router).

Run: python tests/stride_test.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.services import cascade as cs            # noqa: E402

STRIDES = [0.01, 0.025, 0.2]

fails = []


def check(ok, label, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(label)


def part1():
    print("\n(i) pure: chord offsets and the minimum chord length")
    check(cs.STRIDE == 0.025 and cs.PC_DS == cs.STRIDE,
          "PC_DS is still the module STRIDE (the walk protocol is not a per-run knob)",
          f"STRIDE {cs.STRIDE}, PC_DS {cs.PC_DS}")
    check(cs.CascadeRun(run_id="d", prompts=["a", "b", "c"], seed=1, steps=8, height=256,
                        width=256, guidance_scale=3.5, n_chords=4, n_patches=2).stride
          == cs.STRIDE, "run default stride = STRIDE")

    # the old chord phase: reject below 6*STRIDE, probe at tneg + i*STRIDE
    # 6 * 0.025 is 0.15000000000000002 in binary, so compare to the double the old
    # expression produced rather than to it literally (the 2e-17 gap cannot move a chord)
    check(cs._min_chord_len(cs.STRIDE) == 0.15
          and abs(cs._min_chord_len(cs.STRIDE) - 6 * cs.STRIDE) < 1e-15,
          "at the default stride the minimum chord length is the 6*STRIDE of record")
    tneg, tpos = -0.31, 0.42
    old = [tneg + i * cs.STRIDE
           for i in range(int((tpos - tneg) / cs.STRIDE) + 1)]
    new = cs._chord_offsets(tneg, tpos, cs.STRIDE)
    check(new == old, "offsets at the default stride are the old formula exactly",
          f"{len(new)} probes on a chord of length {tpos - tneg:.2f}")

    for s in STRIDES:
        offs = cs._chord_offsets(tneg, tpos, s)
        L = tpos - tneg
        gaps = [b - a for a, b in zip(offs, offs[1:])]
        ok = (len(offs) == int(L / s) + 1
              and abs(offs[0] - tneg) < 1e-12
              and all(abs(g - s) < 1e-12 for g in gaps)
              and offs[-1] <= tpos + 1e-12 and offs[-1] + s > tpos - 1e-12)
        check(ok, f"stride {s}: {len(offs)} probes, spacing exact, last inside the chord",
              f"L={L:.2f}, first {offs[0]:.4f}, last {offs[-1]:.4f}")

    # cost is ~1/stride: the frontend estimate scales the 110/k probe term by 0.025/stride
    n = {s: len(cs._chord_offsets(tneg, tpos, s)) for s in STRIDES}
    check(abs(n[0.01] / n[0.025] - 2.5) < 0.1 and abs(n[0.2] / n[0.025] - 0.125) < 0.05,
          "probe count scales as 1/stride", f"{n[0.01]}/{n[0.025]}/{n[0.2]} probes")

    # the min-length rule has to leave at least 3 probes (>= 2 neighbour pairs to compare)
    for s in STRIDES:
        m = cs._min_chord_len(s)
        cnt = len(cs._chord_offsets(0.0, m, s))
        check(m == max(0.15, 2 * s) and cnt >= 3,
              f"stride {s}: shortest accepted chord {m:.3f} still carries >= 3 probes",
              f"{cnt} probes")
    check(cs._min_chord_len(0.2) == 0.4,
          "at a coarse stride the 2*stride arm takes over from 0.15")


def part2():
    print("\n(ii) /api/cascade/start through TestClient (stubbed state, run_cascade no-op)")
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

        r = c.post("/api/cascade/start", json=dict(body, stride=0.05))
        rid = r.json().get("run_id", "")
        check(r.status_code == 200 and app.state.cascades.get(rid) is not None
              and app.state.cascades[rid].stride == 0.05,
              "stride=0.05 accepted and threaded into the run", f"HTTP {r.status_code}")
        if rid:
            check(c.get(f"/api/cascade/{rid}/status").json()["stride"] == 0.05,
                  "status reports the run's own stride")

        r = c.post("/api/cascade/start", json=body)
        rid = r.json().get("run_id", "")
        got = app.state.cascades[rid].stride if rid else None
        check(r.status_code == 200 and got == 0.025,
              "omitted stride defaults to 0.025", f"got {got!r}")

        for bad in (0.005, 0.5):
            r = c.post("/api/cascade/start", json=dict(body, stride=bad))
            check(r.status_code == 422, f"stride={bad} rejected (outside [0.01, 0.2])",
                  f"got HTTP {r.status_code}")
    finally:
        cs.run_cascade = orig


if __name__ == "__main__":
    part1()
    part2()
    print(f"\n{'ALL CHECKS PASS' if not fails else 'FAILED: ' + ', '.join(fails)}")
    sys.exit(1 if fails else 0)
