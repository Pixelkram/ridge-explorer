"""A/B test of the cascade fan walk: bracket-chord transversal (iso) vs exact-JVP normal (jvp), same origins.

PRE-REGISTERED (written before any walk was run, 2026-09-24):
  Unit = (crossing, direction) pair, both walks from the same origin with n_steps = N.
  Metric = number of captured stations (len(steps): the contiguous prefix that kept the same ridge).
  H_walk: the JVP walk captures MORE stations than the iso walk in >= 65 % of pairs where they differ,
          and the paired Wilcoxon on captured counts is significant (p < 0.05, two-sided) in the JVP direction.
  Secondary: |cos| between the JVP normal and the bracket chord (how different the two transversals are);
             wall time overhead of the JVP; stations captured only after the widen retry.
  Verdict rule: SUPPORTED iff both conditions hold; REFUTED iff iso wins >= 65 % of differing pairs with p < 0.05;
                INCONCLUSIVE otherwise (including too few pairs, n < 8).
Usage: python tests/walk_ab.py --run <run_id> [--steps 8] [--base http://127.0.0.1:8001]
"""
import argparse, json, time, sys, urllib.request
import numpy as np
from scipy.stats import wilcoxon

ap = argparse.ArgumentParser(); ap.add_argument('--run', required=True); ap.add_argument('--steps', type=int, default=8); ap.add_argument('--base', default='http://127.0.0.1:8001'); ap.add_argument('--arm', default='jvp', choices=['jvp','relative','both'], help='what arm B changes vs the default walk'); ap.add_argument('--tag', default=''); a = ap.parse_args()


def call(p, b=None):
    r = urllib.request.Request(a.base + p, data=json.dumps(b).encode() if b is not None else None, headers={"Content-Type": "application/json"}, method="POST" if b is not None else "GET")
    return json.loads(urllib.request.urlopen(r, timeout=120).read())


def walk(cid, direction, armB):
    body = {"cid": cid, "direction": direction, "n_steps": a.steps, "mode": "fan", "use_jvp": bool(armB and a.arm in ('jvp', 'both')),
            "sig_mode": "relative" if (armB and a.arm in ('relative', 'both')) else "absolute"}
    w = call(f"/api/cascade/{a.run}/walk", body)
    if w.get("error"): return None
    t0 = time.time()
    while time.time() - t0 < 900:
        s = call(f"/api/cascade/{a.run}/walk/{w['walk_id']}")
        if s["status"] in ("complete", "error", "cancelled"): return s | {"wall": time.time() - t0}
        time.sleep(2)
    return None


st = call(f"/api/cascade/{a.run}/status"); cids = [c["cid"] for c in st.get("crossings", [])]
print(f"run {a.run}: {len(cids)} crossings, k={len(st.get('prompts', []))}", flush=True)
rows = []
for cid in cids:
    for d in (1, -1):
        wi = walk(cid, d, False); wj = walk(cid, d, True)
        if wi is None or wj is None: print(f"cid {cid} dir {d}: skipped ({'iso' if wi is None else 'jvp'} walk unavailable)"); continue
        rows.append(dict(cid=cid, dir=d, iso=len(wi["steps"]), jvp=len(wj["steps"]), cos=(wj.get("jvp") or {}).get("cos_with_bracket"),
                         rank1=(wj.get("jvp") or {}).get("rank1_share"), wall_iso=wi["wall"], wall_jvp=wj["wall"], notes_jvp=wj.get("notes", [])[:2]))
        print(f"cid {cid:>3} dir {d:+d}: iso {rows[-1]['iso']} vs jvp {rows[-1]['jvp']} stations | |cos| {rows[-1]['cos']} | wall {rows[-1]['wall_iso']:.0f}s vs {rows[-1]['wall_jvp']:.0f}s", flush=True)
iso, jvp = np.array([r['iso'] for r in rows]), np.array([r['jvp'] for r in rows]); diff = jvp - iso; nd = int(np.sum(diff != 0))
frac = float(np.mean(diff[diff != 0] > 0)) if nd else float('nan'); p = float(wilcoxon(diff).pvalue) if nd else 1.0
verdict = "INCONCLUSIVE" if len(rows) < 8 or nd == 0 else ("SUPPORTED" if frac >= 0.65 and p < 0.05 and np.median(diff) > 0 else "REFUTED" if (1 - frac) >= 0.65 and p < 0.05 else "INCONCLUSIVE")
res = dict(n_pairs=len(rows), n_differing=nd, frac_jvp_wins=frac, wilcoxon_p=p, mean_iso=float(iso.mean()) if len(rows) else None, mean_jvp=float(jvp.mean()) if len(rows) else None,
           median_cos_with_bracket=float(np.nanmedian([r['cos'] for r in rows if r['cos'] is not None])) if rows else None, jvp_overhead_s=float(np.median([r['wall_jvp'] - r['wall_iso'] for r in rows])) if rows else None, verdict=verdict, rows=rows)
json.dump(res, open(f"tests/walk_ab_{a.run}_{a.arm}{a.tag}.json", "w"), indent=1); print(json.dumps({k: v for k, v in res.items() if k != 'rows'}, indent=1))
