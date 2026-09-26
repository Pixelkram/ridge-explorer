"""Collect paired walks for the continuation-vs-fan A/B (tests/PREREG_walk_continuation.md). Records raw data only;
tests/walk_pc_analyze.py computes the registered metrics.

For every crossing of a k = 4 cascade and both directions: one fan walk and one continuation walk (sig_mode
relative, no JVP, n_steps N), each followed by held-out-seed certification of its stations. Both modes of a unit
walk the same plane (same crossing, same seeded tangent). Units run in a small thread pool.

Usage: python tests/walk_pc_collect.py --set k4_T5 --seed 42 --out tests/walk_pc_raw_dev.json [--max-units 8]
       python tests/walk_pc_collect.py --run <run_id> --out ...        (reuse a finished cascade run)
"""
import argparse, json, time, threading, urllib.request
from concurrent.futures import ThreadPoolExecutor

SETS = "/home/student/ai/search_problem/outputs/h08_vector_uses/1_highdim/sets.json"
ap = argparse.ArgumentParser()
ap.add_argument("--set", default=None, help="prompt set name from h08 sets.json (k = 4)")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--run", default=None)
ap.add_argument("--steps", type=int, default=6, help="walk n_steps (budget = n_steps x 0.05 arclength)")
ap.add_argument("--max-units", type=int, default=0)
ap.add_argument("--workers", type=int, default=3)
ap.add_argument("--out", required=True)
ap.add_argument("--base", default="http://127.0.0.1:8001")
a = ap.parse_args()


def call(p, b=None, timeout=120):
    r = urllib.request.Request(a.base + p, data=json.dumps(b).encode() if b is not None else None,
                               headers={"Content-Type": "application/json"}, method="POST" if b is not None else "GET")
    return json.loads(urllib.request.urlopen(r, timeout=timeout).read())


def wait(path, done, limit=1800, every=2.0):
    t0 = time.time()
    while time.time() - t0 < limit:
        s = call(path)
        if done(s):
            return s, time.time() - t0
        time.sleep(every)
    return call(path), time.time() - t0


rid = a.run
if rid is None:
    prompts = next(s for s in json.load(open(SETS)) if s["name"] == a.set)["prompts"]
    rid = call("/api/cascade/start", {"k": 4, "prompts": prompts, "n_chords": 10, "n_patches": 2, "seed": a.seed,
                                      "steps": 8})["run_id"]
    print(f"cascade {rid} started ({a.set}, seed {a.seed})", flush=True)
    wait(f"/api/cascade/{rid}/status", lambda s: s["status"] != "running", limit=3000, every=5)
st = call(f"/api/cascade/{rid}/status")
xs = st["crossings"]
print(f"run {rid}: status {st['status']}, {len(xs)} crossings, {sum(x['significant'] for x in xs)} significant, "
      f"bg_p95 {st.get('bg_p95')}", flush=True)
units = [(x, d) for x in xs for d in (1, -1)]
if a.max_units:
    units = units[:a.max_units]
lock = threading.Lock()


def one(x, d, mode):
    w = call(f"/api/cascade/{rid}/walk", {"cid": x["cid"], "direction": d, "n_steps": a.steps, "use_jvp": False,
                                          "sig_mode": "relative", "mode": mode})
    if w.get("error"):
        return {"error": w["error"]}
    s, wall = wait(f"/api/cascade/{rid}/walk/{w['walk_id']}", lambda s: s["status"] != "running")
    call(f"/api/cascade/{rid}/walk/{w['walk_id']}/certify", {})
    c, cwall = wait(f"/api/cascade/{rid}/walk/{w['walk_id']}", lambda s: (s.get("cert") or {}).get("status") != "running")
    return c | {"wall_s": wall, "cert_wall_s": cwall}


def unit(xd):
    x, d = xd
    r = {"cid": x["cid"], "dir": d, "significant": x["significant"], "b": x["b"], "origin": x["weights"],
         "fan": one(x, d, "fan"), "continuation": one(x, d, "continuation")}
    with lock:
        f, c = r["fan"], r["continuation"]
        print(f"cid {x['cid']:>3} dir {d:+d} sig={int(x['significant'])}: fan {len(f.get('steps', []))} st "
              f"{f.get('images')} img {f.get('wall_s', 0):.0f}s | cont {len(c.get('steps', []))} st {c.get('images')} img "
              f"{c.get('wall_s', 0):.0f}s | {(c.get('notes') or ['budget'])[-1]}", flush=True)
    return r


t0 = time.time()
with ThreadPoolExecutor(a.workers) as ex:
    rows = list(ex.map(unit, units))
json.dump({"run_id": rid, "set": a.set, "seed": a.seed, "prompts": st["prompts"], "n_steps": a.steps,
           "bg_p95": st.get("bg_p95"), "crossings": xs, "units": rows, "wall_s": time.time() - t0},
          open(a.out, "w"), indent=1)
print(f"saved {a.out}: {len(rows)} units in {time.time() - t0:.0f}s", flush=True)
