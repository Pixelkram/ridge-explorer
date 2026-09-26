"""DEV check of the trace phase against the dense ground truth (not a registered analysis).

For a k = 3 cascade run with a trace phase: which ground-truth ridge (basin pair) is each crossing on; how well do
the signature grouping (ridge_group) and the traced grouping (traced_group) recover that identity (pairwise
precision / recall); and how close is each Good-Turing unexplored share to the true share, i.e. the share of
virtual-chord crossings that land on species this run never crossed.
Usage: python tests/walk_gt_dev/coverage_dev.py RAW.json [RAW2.json ...]"""
import json, sys
from collections import Counter
from itertools import combinations
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from walk_gt_lib import Lattice

CACHE = {}


def gt(path):
    if path not in CACHE:
        L = Lattice(path)
        CACHE[path] = (L, L.chord_weights(n=20000, seed=0))
    return CACHE[path]


def good_turing(groups):
    c = Counter(groups.values())
    return sum(1 for v in c.values() if v == 1) / len(groups) if groups else None


for raw in sys.argv[1:]:
    d = json.load(open(raw))
    st = d["status"]
    L, W = gt(d["lattice"])
    tot = sum(v for s, v in W.items() if s is not None)
    xs = st["crossings"]
    sp = {x["cid"]: L.species(x["weights"]) for x in xs}
    crossed = {s for s in sp.values() if s is not None}
    truth = sum(v for s, v in W.items() if s is not None and s not in crossed) / tot
    # every estimate on the SAME judged crossings (those with a ground-truth region pair)
    judged = [x for x in xs if sp[x["cid"]] is not None]
    oracle = good_turing({x["cid"]: sp[x["cid"]] for x in judged})
    est_sig = good_turing({x["cid"]: x["ridge_group"] for x in judged})
    est_tr = good_turing({x["cid"]: x.get("traced_group") for x in judged}) if st.get("traces") else None
    print(f"{Path(raw).name}: {len(xs)} crossings ({len(judged)} on GT two-region fronts, "
          f"{len(crossed)} species of {len([s for s in W if s is not None])}); links {len(st.get('trace_links', []))}")
    print(f"  unexplored share on judged crossings: truth {truth:.3f} | signature {est_sig:.3f} | traced "
          f"{est_tr if est_tr is None else round(est_tr, 3)} | oracle (true identity) {oracle:.3f}   "
          f"[tool, all crossings: signature {st['unexplored_share']}, traced {st.get('traced_unexplored_share')}]")
    for name, key in (("signature", "ridge_group"), ("traced", "traced_group")):
        pairs = [(sp[a["cid"]] == sp[b["cid"]], a[key] is not None and a[key] == b[key])
                 for a, b in combinations(xs, 2) if sp[a["cid"]] is not None and sp[b["cid"]] is not None]
        tp = sum(t and p for t, p in pairs); fp = sum(p and not t for t, p in pairs); fn = sum(t and not p for t, p in pairs)
        print(f"  {name:9s} grouping: same-ridge pairs found {tp}/{tp + fn} (recall {tp / max(tp + fn, 1):.2f}), "
              f"wrong merges {fp} (precision {tp / max(tp + fp, 1):.2f})")
    lk = st.get("trace_links", [])
    if lk:
        good = sum(sp[a] is not None and sp[a] == sp[b] for a, b in lk)
        und = sum(sp[a] is None or sp[b] is None for a, b in lk)
        print(f"  trace links: {good}/{len(lk)} join the same GT ridge, {und} involve an off-front crossing")
