"""DEV calibration of the continuity guard (not a registered analysis). Labels every accepted walk step with the
dense ground truth (tests/walk_gt_lib.py): does the station sit on the same ground-truth ridge (basin pair) as
the previous station / the origin? Then: does the continuity statistic r (largest side move / previous contrast,
steps[].cont) separate continuing steps from switches?
Usage: python tests/walk_gt_dev/calibrate.py tests/walk_gt_dev/dev_k3T0_c1.json [--sep 1]"""
import argparse, json, sys
from collections import Counter
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from walk_gt_lib import Lattice

ap = argparse.ArgumentParser(); ap.add_argument("raw"); ap.add_argument("--sep", type=int, default=1); a = ap.parse_args()
d = json.load(open(a.raw))
L = Lattice(d["lattice"], sep=a.sep)
nb = int(L.labels.max()) + 1
print(f"lattice N={L.N}: {len(L.w)} points, boundary share {L.boundary.mean():.2f}, {nb} basins "
      f"(sizes {sorted(Counter(L.labels[L.labels >= 0]).values(), reverse=True)[:8]} ...)")
cross = {x["cid"]: x for x in d["status"]["crossings"]}
sp_cross = {c: L.species(x["weights"]) for c, x in cross.items()}
print(f"crossings on a ground-truth front: {sum(v is not None for v in sp_cross.values())}/{len(sp_cross)}; "
      f"distinct species among them {len({v for v in sp_cross.values() if v is not None})}")
steps, walks = [], []
for u in d["walks"]:
    w = u.get("walk")
    if not w or w["status"] != "complete":
        continue
    s0 = sp_cross[u["cid"]]
    prev = s0
    labs = []
    for st in w["steps"]:
        s = L.species(st["center"])
        lab_prev = "none" if s is None or prev is None else ("same" if s == prev else "switch")
        lab_orig = "none" if s is None or s0 is None else ("same" if s == s0 else "switch")
        steps.append(dict(r=st.get("cont"), prev=lab_prev, orig=lab_orig, j=len(labs) + 1, cid=u["cid"]))
        labs.append(lab_orig)
        prev = s
    walks.append(dict(cid=u["cid"], dir=u["dir"], mode=u["sig_mode"], origin_ok=s0 is not None, n=len(labs),
                      switched=any(l == "switch" for l in labs), none=any(l == "none" for l in labs),
                      end=(w["notes"] or ["budget"])[-1].split(" at ")[0]))
ok = [w for w in walks if w["origin_ok"]]
print(f"\nwalks: {len(walks)}, from ground-truth fronts {len(ok)}; stations {sum(w['n'] for w in ok)}")
print(f"  walks that switched ridge at least once (vs origin): {sum(w['switched'] for w in ok)}/{len(ok)}; "
      f"walks with an off-front station: {sum(w['none'] for w in ok)}/{len(ok)}")
print("  end notes:", Counter(w["end"] for w in ok).most_common())
S = [s for s in steps if s["prev"] in ("same", "switch") and s["r"] is not None]
same = np.array([s["r"] for s in S if s["prev"] == "same"]); sw = np.array([s["r"] for s in S if s["prev"] == "switch"])
print(f"\nsteps with both ends on ground-truth fronts: {len(S)} (continue {len(same)}, switch {len(sw)})")
if len(same) and len(sw):
    for q in (0.1, 0.25, 0.5, 0.75, 0.9):
        print(f"  r quantile {q:.2f}: continue {np.quantile(same, q):.2f} | switch {np.quantile(sw, q):.2f}")
    from sklearn.metrics import roc_auc_score
    print(f"  AUC(r separates switch from continue) = {roc_auc_score([0] * len(same) + [1] * len(sw), list(same) + list(sw)):.3f}")
    print("  gamma | keeps continue | rejects switch")
    for g in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5):
        print(f"  {g:5.2f} | {np.mean(same < g):.2f} | {np.mean(sw >= g):.2f}")
first = [s for s in S if s["j"] == 1]; later = [s for s in S if s["j"] > 1]
print(f"station 1: continue {sum(s['prev'] == 'same' for s in first)}, switch {sum(s['prev'] == 'switch' for s in first)}; "
      f"later: continue {sum(s['prev'] == 'same' for s in later)}, switch {sum(s['prev'] == 'switch' for s in later)}")
