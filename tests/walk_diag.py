"""Diagnostic for the cascade fan walk: HOW does it lose the front? (RESEARCH_ridge_following_k4.md §4)

For every crossing of a run, both directions, n_steps = N: classify the walk by its first failure note —
  captured_all        every station kept the ridge
  ended_no_straddle   "ridge ended before station j": no straddle inside the arm (drift or weak front) -> (a)/(b)
  changed_identity    "ridge changed identity": straddles a different ridge (junction / signature) -> (c)
  edge/face           left the simplex
— split by the origin crossing's `significant` flag. Writes tests/walk_diag_<run>.json.
Usage: python tests/walk_diag.py [--run <run_id>] [--steps 4] [--base http://127.0.0.1:8001]
Without --run, starts a fresh k=4 cascade (10 chords) and waits for it.
"""
import argparse, json, time, collections, urllib.request

ap = argparse.ArgumentParser(); ap.add_argument('--run', default=None); ap.add_argument('--steps', type=int, default=4); ap.add_argument('--base', default='http://127.0.0.1:8001'); ap.add_argument('--sig-mode', default='absolute', choices=['absolute','relative']); a = ap.parse_args()


def call(p, b=None):
    r = urllib.request.Request(a.base + p, data=json.dumps(b).encode() if b is not None else None, headers={"Content-Type": "application/json"}, method="POST" if b is not None else "GET")
    return json.loads(urllib.request.urlopen(r, timeout=120).read())


rid = a.run
if rid is None:
    prompts = ["a photograph of an airplane", "a photograph of a giraffe", "a surrealist oil painting of a roll of toilet paper", "a photograph of sushi on a plate"]
    rid = call("/api/cascade/start", {"k": 4, "prompts": prompts, "n_chords": 10, "n_patches": 2, "seed": 42, "steps": 8})["run_id"]; t0 = time.time()
    while call(f"/api/cascade/{rid}/status")["status"] == "running" and time.time() - t0 < 1500: time.sleep(5)
s = call(f"/api/cascade/{rid}/status"); xs = s["crossings"]
print(f"run {rid}: {len(xs)} crossings, significant {sum(x['significant'] for x in xs)}, bg_p95={s.get('bg_p95')}", flush=True)
cls, rows = collections.Counter(), []
for x in xs:
    for d in (1, -1):
        w = call(f"/api/cascade/{rid}/walk", {"cid": x["cid"], "direction": d, "n_steps": a.steps, "use_jvp": False, "sig_mode": a.sig_mode, "mode": "fan"})
        if w.get("error"): cls["walk_error"] += 1; continue
        while True:
            st = call(f"/api/cascade/{rid}/walk/{w['walk_id']}")
            if st["status"] != "running": break
            time.sleep(2)
        notes, n = st.get("notes", []), len(st["steps"])
        kind = ("captured_all" if n >= a.steps else "changed_identity" if any("changed identity" in t for t in notes) else
                "ended_no_straddle" if any("ended before" in t for t in notes) else "edge/face" if any(("edge" in t or "face" in t) for t in notes) else "other")
        cls[kind] += 1
        rows.append(dict(cid=x["cid"], dir=d, significant=x["significant"], b=x["b"], bracket_w=x.get("bracket_w"), captured=n, kind=kind, notes=notes[:2]))
        print(f"cid {x['cid']:>3} dir {d:+d} sig={int(x['significant'])} b={x['b'] if x['b'] is None else round(x['b'], 2)}: captured {n} -> {kind} | {notes[:1]}", flush=True)
sig = [r for r in rows if r['significant']]; non = [r for r in rows if not r['significant']]
summ = {"classes_all": dict(cls), "classes_significant": dict(collections.Counter(r['kind'] for r in sig)), "classes_nonsignificant": dict(collections.Counter(r['kind'] for r in non)),
        "mean_captured_significant": (sum(r['captured'] for r in sig) / len(sig)) if sig else None, "mean_captured_nonsignificant": (sum(r['captured'] for r in non) / len(non)) if non else None}
print(json.dumps(summ, indent=1))
json.dump(dict(run=rid, steps=a.steps, summary=summ, rows=rows), open(f'tests/walk_diag_{rid}_{a.sig_mode}.json', 'w'), indent=1)
