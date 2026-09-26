"""DEV comparison of walk rules against the dense ground truth (not a registered analysis): for each unit (crossing,
direction) and each sig_mode, did the walk ever sit on a different ground-truth ridge (basin pair) than its origin,
how far did it get (A, capped at the budget), and how far before the first switch (A_cons)?
Usage: python tests/walk_gt_dev/ab_dev.py RAW.json"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from walk_gt_lib import Lattice

d = json.load(open(sys.argv[1]))
L = Lattice(d["lattice"])
cap = 0.05 * 6
cross = {x["cid"]: x for x in d["status"]["crossings"]}
res = defaultdict(dict)
for u in d["walks"]:
    w = u.get("walk")
    if not w or w["status"] != "complete":
        continue
    s0 = L.species(cross[u["cid"]]["weights"])
    if s0 is None:
        continue
    prev = np.asarray(cross[u["cid"]]["weights"]); A = Acons = 0.0; sw = False
    for st in w["steps"]:
        c = np.asarray(st["center"]); seg = min(float(np.linalg.norm(c - prev)), max(cap - A, 0.0))
        s = L.species(c)
        if s is not None and s != s0:
            sw = True
        A += seg
        if not sw:
            Acons += seg
        prev = c
    res[(u["cid"], u["dir"])][u["sig_mode"]] = dict(sw=sw, A=A, Acons=Acons, n=len(w["steps"]), img=w["images"],
                                                    end=(w["notes"] or ["budget"])[-1].split(" at ")[0])
modes = sorted({m for r in res.values() for m in r})
units = [r for r in res.values() if all(m in r for m in modes)]
print(f"{len(units)} paired units from ground-truth fronts; modes {modes}")
for m in modes:
    print(f"  {m:11s}: switched {sum(u[m]['sw'] for u in units)}/{len(units)} | mean A {np.mean([u[m]['A'] for u in units]):.3f} | "
          f"mean A before switch {np.mean([u[m]['Acons'] for u in units]):.3f} | stations {np.mean([u[m]['n'] for u in units]):.1f} | "
          f"images {np.mean([u[m]['img'] for u in units]):.1f}")
if len(modes) == 2:
    a, b = modes
    print(f"  switch discordant: only {a} switched {sum(u[a]['sw'] and not u[b]['sw'] for u in units)}, "
          f"only {b} switched {sum(u[b]['sw'] and not u[a]['sw'] for u in units)}")
    from collections import Counter
    for m in modes:
        print(f"  {m} ends:", Counter(u[m]["end"] for u in units).most_common())
