"""End-to-end check of the exact-JVP probes against a running backend (default http://127.0.0.1:8001).

1. start a small 3-prompt grid job and wait for it
2. POST /jvp-probe at a cell, poll until done; print the normal, rank-1 share, PR, wall time
3. POST /token-probe on prompt A, poll; print the ranking
4. start an identical grid job AFTER the probes and compare embeddings.npy with the first: the probe must leave
   the tool's own generation bit-identical (norm patch / VAE dtype restored)
Usage: python tests/probe_e2e.py [--base http://127.0.0.1:8001] [--grid 4]
"""
import argparse, json, time, sys, urllib.request
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument('--base', default='http://127.0.0.1:8001'); ap.add_argument('--grid', type=int, default=4); a = ap.parse_args()
RES = Path(__file__).resolve().parents[1] / 'cache_data' / 'results'
PROMPTS = dict(prompt_a="a photograph of an airplane", prompt_b="a photograph of a giraffe", prompt_c="a surrealist oil painting of a roll of toilet paper")


def call(path, body=None):
    req = urllib.request.Request(a.base + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"}, method="POST" if body is not None else "GET")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def wait_job(job_id, timeout=1200):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = call(f"/api/grid/{job_id}/status")
        if st.get("status") in ("complete", "completed", "done") or st.get("phase") in ("complete", "completed", "done"):
            return st
        if st.get("status") == "error": sys.exit(f"job {job_id} failed: {st}")
        time.sleep(2)
    sys.exit("grid job timed out")


def wait_probe(kind, pid, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = call(f"/api/grid/{kind}/{pid}")
        if st["status"] in ("done", "error"): return st
        time.sleep(1.5)
    sys.exit("probe timed out")


body = dict(**PROMPTS, grid_size=a.grid, seed=42, steps=8, guidance_scale=4.0, height=512, width=512, dimensions=2)
j1 = call("/api/grid/start", body); jid1 = j1.get("job_id") or j1.get("id"); print("job 1:", jid1, flush=True); wait_job(jid1); print("job 1 complete", flush=True)

t0 = time.time(); p = call(f"/api/grid/{jid1}/jvp-probe", {"alpha": 0.43, "beta": 0.22}); print("jvp probe:", p, flush=True)
st = wait_probe("jvp-probe", p["probe_id"]); r = st["result"] or {}
print(f"JVP probe {st['status']} in {time.time() - t0:.0f}s: normal_theta={np.round(r.get('normal_theta', []), 3).tolist()} rank1={r.get('rank1_share', float('nan')):.3f} "
      f"PR={r.get('participation_ratio', float('nan')):.2f} sigma1={r.get('sigma1_raw', float('nan')):.1f} wall={r.get('wall_s', float('nan')):.1f}s | {r.get('normal_reading')} | err={st['error']!r}", flush=True)

t0 = time.time(); p = call(f"/api/grid/{jid1}/token-probe", {"which": "a"}); print("token probe:", p, flush=True)
st = wait_probe("token-probe", p["probe_id"]); r = st["result"] or {}
print(f"token probe {st['status']} in {time.time() - t0:.0f}s: ranking={r.get('ranking')} n_jvp={r.get('n_jvp')} wall={r.get('wall_s', float('nan')):.0f}s | err={st['error']!r}", flush=True)

j2 = call("/api/grid/start", body); jid2 = j2.get("job_id") or j2.get("id"); wait_job(jid2); print("job 2 complete", flush=True)
dir1, dir2 = (next(iter(sorted(RES.glob(f"{j}*"))), RES / j) for j in (jid1, jid2))   # result dirs carry a _s<seed> suffix
for name in ('sensitivity.npy', 'embeddings.npy'):
    f1, f2 = dir1 / name, dir2 / name
    if f1.exists() and f2.exists():
        d = float(np.nanmax(np.abs(np.load(f1).astype(np.float64) - np.load(f2).astype(np.float64))))
        print(f"bit-identity after probes ({name}): max|Δ| = {d:.2e} ->", "PASS" if d == 0.0 else "DIFFERS (generation altered by probe?)")
    else:
        print(f"({name} not persisted for one of the jobs; skipped)")
