"""Offline checks of the per-run `certify` switch (backend/services/cascade.py). No GPU:
cascade.evaluate is replaced by the band-field stub of tests/branch_test.py, so the real
chord phase runs and every later phase would show up as a GPU round we can count.

`certify` is OFF by default: a run then stops after detection. Its crossings keep the cheap
bracket that found them (mid at bracket precision, no B, no significance), and nothing is
rebracketed, bisected, scored or refined -- roughly 6x cheaper per crossing, and everything
that reads chord GEOMETRY rather than a verdict still works.

  (i)   REQUEST -- /api/cascade/start through FastAPI's TestClient with a stubbed app.state
        and cs.run_cascade replaced by a no-op (as tests/stride_test.py does): certify
        defaults to False, accepts true, and the status reports the run's own setting.
  (ii)  PIPELINE -- one stubbed survey with certify off: only the chord phase submits GPU
        work, every crossing gets mid/n with b None and significant False, the thumb reuses
        a bracket-end probe, no patches, and the note says so. The same survey with certify
        on is run as the control, so "no rebracket/bisect/score round" cannot pass
        vacuously. coverage_stats and local_sv.from_run/local_sv then read the finished
        detection-only run, in memory and through /status and /local-sv.
  (iii) GUARDS -- walk start (and walk certification, and the certified-only density map) on
        a detection-only run answer 4xx with the reason, never a 500.

Run: python tests/certify_test.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.services import cascade as cs            # noqa: E402
from backend.services import local_sv as lsv          # noqa: E402
from branch_test import _band_field                   # noqa: E402  (same analytic field)

fails = []


def check(ok, label, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(label)


def _stub_evaluate(run, field, labels):
    """Stand-in for cascade.evaluate: embeds every weight vector through the analytic field
    and records the label of each GPU round, which is how the phases are counted here."""
    def stub(app, r, pool, weights, seed, label, ctl=None, on_arrival=None, steps=None):
        labels.append(label.split(".")[0])
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


def _survey(certify, k=4, n_chords=4, seed=11):
    """One stubbed survey to completion; returns (run, [round label])."""
    run = cs.CascadeRun(
        run_id="c", prompts=[f"p{i}" for i in range(k)], seed=seed, steps=8, height=64,
        width=64, guidance_scale=3.5, n_chords=n_chords, n_patches=2, certify=certify)
    labels = []
    orig = cs.evaluate
    cs.evaluate = _stub_evaluate(run, _band_field(k), labels)
    try:
        cs.run_cascade(None, run, None)
    finally:
        cs.evaluate = orig
    return run, labels


def part1():
    print("\n(i) /api/cascade/start through TestClient (stubbed state, run_cascade no-op)")
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
        got = app.state.cascades[rid].certify if rid else None
        check(r.status_code == 200 and got is False,
              "omitted certify defaults to False (detection only)", f"got {got!r}")
        if rid:
            check(c.get(f"/api/cascade/{rid}/status").json()["certify"] is False,
                  "status reports the run's own certify setting")

        r = c.post("/api/cascade/start", json=dict(body, certify=True))
        rid = r.json().get("run_id", "")
        got = app.state.cascades[rid].certify if rid else None
        check(r.status_code == 200 and got is True,
              "certify=true accepted and threaded into the run", f"HTTP {r.status_code}")
        if rid:
            check(c.get(f"/api/cascade/{rid}/status").json()["certify"] is True,
                  "status reports certify on for a full run")

        r = c.post("/api/cascade/start", json=dict(body, certify="maybe"))
        check(r.status_code == 422, "a non-boolean certify is rejected",
              f"got HTTP {r.status_code}")
    finally:
        cs.run_cascade = orig


def part2():
    print("\n(ii) a detection-only survey: which phases run, and what the crossings carry")
    run, labels = _survey(certify=False)
    full, full_labels = _survey(certify=True)

    phases = sorted({l for l in labels})
    check(all(l.startswith("chords") for l in labels) and len(run.crossings) > 0,
          "certify off: every GPU round is a chord round",
          f"rounds {phases}, {len(run.crossings)} crossings")
    check(any(l == "rebracket" for l in full_labels)
          and any(l.startswith("bisect") for l in full_labels)
          and any(l.startswith("score") for l in full_labels)
          and any(l.startswith("patch") for l in full_labels),
          "control: the same survey with certify on runs all four phases",
          f"{len(set(full_labels))} distinct rounds")
    check(run.generated < full.generated,
          "detection only costs strictly fewer images",
          f"{run.generated} vs {full.generated}")

    bad = [x.cid for x in run.crossings
           if x.mid is None or x.n is None or x.b is not None or x.significant]
    check(not bad, "every crossing carries mid and n, with b None and significant False",
          f"{len(run.crossings)} crossings, {len(bad)} bad")
    mid_ok = all(np.allclose(x.mid, (x.wa + x.wb) / 2) for x in run.crossings)
    n_ok = all(abs(float(np.linalg.norm(x.n)) - 1.0) < 1e-9
               and abs(float(np.dot(x.n, (x.wb - x.wa) / np.linalg.norm(x.wb - x.wa)))
                       - 1.0) < 1e-9
               for x in run.crossings)
    check(mid_ok, "mid is the centre of the bracket that found the crossing")
    check(n_ok, "n is the unit chord direction across that bracket")
    half = float(np.max([np.linalg.norm(x.mid - x.wa) for x in run.crossings]))
    check(half <= run.stride / 2 + 1e-9,
          "quoted position is within half a stride of both bracket ends",
          f"max {half:.4f} vs stride/2 {run.stride / 2:.4f}")
    check(all(x.thumb >= 0 for x in run.crossings)
          and all(x.thumb in run.embeddings for x in run.crossings),
          "the thumb reuses a rendered bracket-end probe")
    check(run.status == "complete" and run.phase == "done" and not run.patches,
          "the run completes with no patches", f"{run.status}/{run.phase}")
    check(any("detection only: crossings uncertified" in n for n in run.notes),
          "the notes say the run is uncertified",
          next((n for n in run.notes if n.startswith("detection only")), "MISSING"))

    # the two products that read chord geometry rather than a verdict
    cov = cs.coverage_stats(run.crossings)
    check(cov[0] is not None and cov[0] >= 1 and 0.0 <= cov[2] <= 1.0,
          "coverage_stats groups the cheap-probe side signatures",
          f"{cov[0]} ridges, {cov[1]} singletons, share {cov[2]:.2f}")
    chords, crossings, k = lsv.from_run(run)
    check(len(chords) == len(run.chords_geo) and len(crossings) == len(run.crossings)
          and k == run.k and not any(c[2] for c in crossings),
          "local_sv.from_run reads the run, with nothing flagged certified",
          f"{len(chords)} chords, {len(crossings)} crossings, k={k}")
    m = lsv.local_sv(chords, crossings, k)
    check(np.isfinite(m["crossings"]["values"]).all()
          and np.isfinite(m["stations"]["values"]).all()
          and m["n_crossings"] == len(crossings),
          "the density map is finite everywhere on a detection-only run",
          f"s_global {m['s_global']:.2f}")
    return run


def part3(det_run):
    print("\n(iii) endpoints that need a bisected or scored crossing")
    try:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
    except Exception as exc:                     # noqa: BLE001
        check(False, "TestClient available", str(exc))
        return
    from backend.routers import cascade as router_mod

    app = FastAPI()
    app.include_router(router_mod.router)
    app.state.cascades = {"d": det_run}
    app.state.cache = object()
    app.state.gpu_pool = None
    c = TestClient(app, raise_server_exceptions=False)
    cid = det_run.crossings[0].cid

    j = c.get("/api/cascade/d/status").json()
    check(j["certify"] is False and len(j["crossings"]) == len(det_run.crossings)
          and all(x["b"] is None and x["significant"] is False for x in j["crossings"])
          and all(x["gen"] == 0 for x in j["crossings"]),
          "status serialises the uncertified crossings (with gen) and the certificate",
          f"{j['distinct_ridges']} ridges, {len(j['chords_meta'])} chord metas")
    check(len(j["chords"]) == len(det_run.chords_geo)
          and len(j["chords_meta"]) == len(det_run.chords_geo),
          "chords_geo and chords_meta are served as for a full run")

    r = c.get("/api/cascade/d/local-sv")
    check(r.status_code == 200 and r.json()["n_chords"] == len(det_run.chords_geo)
          and len(r.json()["crossing_cids"]) == len(det_run.crossings),
          "/local-sv answers the estimator of record on a detection-only run",
          f"HTTP {r.status_code}")

    r = c.post("/api/cascade/d/walk", json={"cid": cid, "direction": 1, "n_steps": 5})
    body = r.text
    check(r.status_code == 409 and "detection-only" in body and "certify on" in body,
          "walk start is refused with the reason, not a 500",
          f"HTTP {r.status_code}: {body[:120]}")

    r = c.post("/api/cascade/d/walk/nosuch/certify")
    check(r.status_code == 409 and "detection-only" in r.text,
          "walk certification is refused the same way", f"HTTP {r.status_code}")

    r = c.get("/api/cascade/d/local-sv?mode=certified")
    check(r.status_code == 400 and "detection-only" in r.text,
          "the certified-only density map is refused rather than answering all zeros",
          f"HTTP {r.status_code}")


if __name__ == "__main__":
    part1()
    det = part2()
    part3(det)
    print(f"\n{'ALL CHECKS PASS' if not fails else 'FAILED: ' + ', '.join(fails)}")
    sys.exit(1 if fails else 0)
