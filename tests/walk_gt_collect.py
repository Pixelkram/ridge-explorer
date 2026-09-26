"""Collect k = 3 cascade runs with a dense ground-truth lattice (tests/walk_gt_lib.py) and continuation walks.

For one prompt triplet and image seed: start a cascade (full-fidelity detection, optional chord seed, optional trace
phase), render the N-lattice through the run's own generation path once per (prompts, seed, N) into tests/gt/, then
walk from every crossing in both directions in each requested sig_mode. Records raw data only.

Usage: python tests/walk_gt_collect.py --set k3_T0 --seed 42 --chord-seed 1 --modes relative --out tests/x.json
"""
import argparse, hashlib, json, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from walk_gt_lib import lattice_weights

SETS = "/home/student/ai/search_problem/outputs/h08_vector_uses/1_highdim/sets.json"
ap = argparse.ArgumentParser()
ap.add_argument("--set", default=None)
ap.add_argument("--prompts", default=None, help="JSON list of 3 prompts (instead of --set)")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--chord-seed", type=int, default=None)
ap.add_argument("--n-chords", type=int, default=16)
ap.add_argument("--N", type=int, default=80)
ap.add_argument("--modes", default="relative")
ap.add_argument("--steps", type=int, default=6)
ap.add_argument("--trace", action="store_true", help="run the cascade's trace phase")
ap.add_argument("--trace-certify", action="store_true", help="certify trace stations (not needed for linking)")
ap.add_argument("--gt-filter", action="store_true", help="walk only from crossings on a ground-truth two-region front")
ap.add_argument("--workers", type=int, default=3)
ap.add_argument("--out", required=True)
ap.add_argument("--base", default="http://127.0.0.1:8001")
a = ap.parse_args()


def call(p, b=None, timeout=300):
    r = urllib.request.Request(a.base + p, data=json.dumps(b).encode() if b is not None else None,
                               headers={"Content-Type": "application/json"}, method="POST" if b is not None else "GET")
    return json.loads(urllib.request.urlopen(r, timeout=timeout).read())


def wait(path, done, limit=3600, every=2.0):
    t0 = time.time()
    while time.time() - t0 < limit:
        s = call(path)
        if done(s):
            return s
        time.sleep(every)
    raise TimeoutError(path)


prompts = json.loads(a.prompts) if a.prompts else next(s for s in json.load(open(SETS)) if s["name"] == a.set)["prompts"]
assert len(prompts) == 3
req = {"k": 3, "prompts": prompts, "n_chords": a.n_chords, "n_patches": 2, "seed": a.seed, "steps": 8,
       "probe_steps": None, "chord_seed": a.chord_seed}
if a.trace:
    req["trace"], req["trace_certify"] = True, bool(a.trace_certify)
t0 = time.time()
rid = call("/api/cascade/start", req)["run_id"]
st = wait(f"/api/cascade/{rid}/status", lambda s: s["status"] != "running", every=5)
print(f"cascade {rid} ({a.set}, seed {a.seed}, chords seed {a.chord_seed}): {st['status']}, {len(st['crossings'])} "
      f"crossings, {st.get('distinct_ridges')} ridges, unexplored {st.get('unexplored_share')} [{time.time() - t0:.0f}s]",
      flush=True)
tag = hashlib.md5(json.dumps(prompts).encode()).hexdigest()[:8]
lat = Path(__file__).resolve().parent / "gt" / f"k3_{tag}_s{a.seed}_N{a.N}.npz"
if not lat.exists():
    t1 = time.time()
    job = call(f"/api/cascade/{rid}/render", {"weights": lattice_weights(a.N), "name": lat.stem})
    job = wait(f"/api/cascade/{rid}/render/{job['id']}", lambda s: s["status"] != "running", every=10)
    print(f"lattice {lat.name}: {job['status']} {job.get('error', '')} [{time.time() - t1:.0f}s]", flush=True)
walks = []
xs = st["crossings"]
if a.gt_filter:
    from walk_gt_lib import Lattice
    L = Lattice(str(lat))
    xs = [x for x in xs if L.species(x["weights"]) is not None]
    print(f"gt filter: {len(xs)}/{len(st['crossings'])} crossings on a two-region front", flush=True)
units = [(x, d, m) for m in a.modes.split(",") if m for x in xs for d in (1, -1)]


def one(u):
    x, d, m = u
    mode, _, gam = m.partition(":")               # e.g. "continuity:0.7" overrides the threshold
    body = {"cid": x["cid"], "direction": d, "n_steps": a.steps, "use_jvp": False, "sig_mode": mode, "mode": "continuation"}
    if gam:
        body["cont_gamma"] = float(gam)
    w = call(f"/api/cascade/{rid}/walk", body)
    if w.get("error"):
        return {"cid": x["cid"], "dir": d, "sig_mode": m, "error": w["error"]}
    s = wait(f"/api/cascade/{rid}/walk/{w['walk_id']}", lambda s: s["status"] != "running")
    return {"cid": x["cid"], "dir": d, "sig_mode": m, "walk": s}


with ThreadPoolExecutor(a.workers) as ex:
    walks = list(ex.map(one, units))
json.dump({"run_id": rid, "set": a.set, "prompts": prompts, "seed": a.seed, "chord_seed": a.chord_seed,
           "lattice": str(lat), "status": st, "walks": walks, "wall_s": time.time() - t0}, open(a.out, "w"), indent=1)
print(f"saved {a.out}: {len(walks)} walks [{time.time() - t0:.0f}s]", flush=True)
