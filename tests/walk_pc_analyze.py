"""Analysis for PREREG_walk_continuation.md (written after the freeze). Reads tests/walk_pc_raw_C*.json (confirmatory)
and tests/walk_pc_raw_D1.json (descriptive); writes tests/walk_pc_results.json.
Usage: python tests/walk_pc_analyze.py"""
import glob, json, re
from collections import Counter
import numpy as np
from scipy.stats import wilcoxon

CAP, DIFF, ARMS = 0.30, 0.01, ("fan", "continuation")
LATTICE = "/home/student/ai/search_problem/outputs/h08_vector_uses/1_highdim/grid4_quad0.npz"


def arcs(origin, walk, cap=CAP):
    """(A, A_cert): polyline origin -> station centres, capped; A_cert counts segments ending at certified stations."""
    cert = (walk.get("cert") or {}).get("significant") or []
    a = ac = 0.0
    prev = np.asarray(origin, dtype=float)
    for i, s in enumerate(walk.get("steps", [])):
        c = np.asarray(s.get("center") or s["weights"], dtype=float)
        if a >= cap:
            break
        take = min(float(np.linalg.norm(c - prev)), cap - a)
        a += take
        if i < len(cert) and cert[i]:
            ac += take
        prev = c
    return a, ac


def end_class(walk):
    n = (walk.get("notes") or [""])[-1]
    return ("budget" if not n else "identity" if "changed identity" in n else "ended" if "ended before" in n
            else "edge" if ("edge" in n or "face" in n) else "other")


def load(path):
    d = json.load(open(path))
    rows, excluded = [], 0
    for u in d["units"]:
        if any(u[a].get("error") or u[a].get("status") != "complete" for a in ARMS):
            excluded += 1
            continue
        r = {"run": re.search(r"raw_(\w+)\.json", path).group(1), "cid": u["cid"], "dir": u["dir"],
             "significant": bool(u["significant"])}
        for a in ARMS:
            w = u[a]
            r[a] = dict(zip(("A", "A_cert"), arcs(u["origin"], w)))
            cert = w.get("cert") or {}
            r[a].update(stations=len(w["steps"]), certified=int(sum(cert.get("significant") or [])),
                        cert_ok=cert.get("status") == "done", images=w.get("images", 0), wall=w.get("wall_s"),
                        end=end_class(w), b=cert.get("b") or [], flags=cert.get("significant") or [],
                        centers=[s.get("center") or s["weights"] for s in w["steps"]])
        rows.append(r)
    return d, rows, excluded


def paired(rows, key="A"):
    D = np.array([r["continuation"][key] - r["fan"][key] for r in rows])
    dif = np.abs(D) > DIFF
    wins = int(np.sum(D > DIFF)); losses = int(np.sum(D < -DIFF))
    try:
        p = float(wilcoxon(D).pvalue) if np.any(D != 0) else 1.0
    except ValueError:
        p = 1.0
    return {"n": len(rows), "n_differing": int(dif.sum()), "cont_longer": wins, "fan_longer": losses,
            "frac_cont_wins": wins / int(dif.sum()) if dif.sum() else float("nan"), "wilcoxon_p": p,
            "mean_D": float(D.mean()) if len(D) else float("nan"),
            "mean_fan": float(np.mean([r["fan"][key] for r in rows])) if rows else float("nan"),
            "mean_cont": float(np.mean([r["continuation"][key] for r in rows])) if rows else float("nan")}


def describe(rows):
    out = {"A": paired(rows, "A"), "A_cert": paired(rows, "A_cert")}
    for a in ARMS:
        st = sum(r[a]["stations"] for r in rows); ce = sum(r[a]["certified"] for r in rows)
        img = sum(r[a]["images"] for r in rows); A = sum(r[a]["A"] for r in rows); Ac = sum(r[a]["A_cert"] for r in rows)
        by_idx = {}
        for r in rows:
            for i, f in enumerate(r[a]["flags"]):
                by_idx.setdefault(i + 1, []).append(bool(f))
        out[a] = {"stations": st, "certified": ce, "certified_rate": ce / st if st else float("nan"),
                  "certified_rate_by_station": {k: [round(float(np.mean(v)), 3), len(v)] for k, v in sorted(by_idx.items())},
                  "images_per_walk": img / len(rows) if rows else float("nan"),
                  "images_per_unit_A": img / A if A else float("nan"), "images_per_unit_A_cert": img / Ac if Ac else float("nan"),
                  "median_wall_s": float(np.median([r[a]["wall"] for r in rows if r[a]["wall"] is not None])) if rows else float("nan"),
                  "end_classes": dict(Counter(r[a]["end"] for r in rows)),
                  "cert_missing": sum(1 for r in rows if not r[a]["cert_ok"])}
    return out


def verdict(prim, runs):
    s = paired(prim, "A")
    guard = np.mean([r["continuation"]["A_cert"] for r in prim]) >= np.mean([r["fan"]["A_cert"] for r in prim]) - 0.01
    elig = {k: v for k, v in runs.items() if len(v) >= 2}
    run_ok = sum(1 for v in elig.values() if np.mean([r["continuation"]["A"] for r in v]) > np.mean([r["fan"]["A"] for r in v]))
    majority = run_ok > len(elig) / 2
    i_ = s["frac_cont_wins"] >= 0.65; ii = s["wilcoxon_p"] < 0.05; iii = s["mean_D"] > 0
    if s["n_differing"] < 8:
        v = "INCONCLUSIVE (fewer than 8 differing units)"
    elif i_ and ii and iii and majority and guard:
        v = "SUPPORTED"
    elif i_ and ii and iii and majority:
        v = "INCONCLUSIVE (longer but not validated: guard failed)"
    elif (1 - s["frac_cont_wins"]) >= 0.65 and ii and s["mean_D"] < 0:
        v = "REFUTED"
    else:
        v = "INCONCLUSIVE"
    return v, {"i_frac>=0.65": bool(i_), "ii_p<0.05": bool(ii), "iii_meanD>0": bool(iii),
               "iv_runs": f"{run_ok}/{len(elig)} eligible runs favour continuation", "iv_majority": bool(majority),
               "v_guard": bool(guard)}


def lattice_check(rows, prompts):
    z = np.load(LATTICE, allow_pickle=True)
    if list(z["prompts"]) != list(prompts):
        return {"skipped": "prompt order differs from the lattice"}
    th = z["theta"]; cells = np.column_stack([1 - th.sum(1), th]); S = z["S"]
    pct = np.argsort(np.argsort(S)) / (len(S) - 1)
    out = {}
    for a in ARMS:
        vals, vc, vu = [], [], []
        for r in rows:
            for c, f in zip(r[a]["centers"], r[a]["flags"] + [None] * len(r[a]["centers"])):
                q = pct[int(np.argmin(np.linalg.norm(cells - np.asarray(c), axis=1)))]
                vals.append(q); (vc if f else vu).append(q)
        out[a] = {"n": len(vals), "mean_S_percentile": float(np.mean(vals)) if vals else None,
                  "certified": float(np.mean(vc)) if vc else None, "uncertified": float(np.mean(vu)) if vu else None}
    return out


def main():
    conf, runs, excl, meta = [], {}, {}, {}
    for p in sorted(glob.glob("tests/walk_pc_raw_C*.json")):
        d, rows, e = load(p)
        lab = rows[0]["run"] if rows else p
        runs[lab] = [r for r in rows if r["significant"]]
        excl[lab] = e
        meta[lab] = {"run_id": d["run_id"], "set": d["set"], "seed": d["seed"], "units": len(d["units"]),
                     "crossings": len(d["crossings"]), "significant": sum(x["significant"] for x in d["crossings"]),
                     "bg_p95": d.get("bg_p95")}
        conf += rows
    prim = [r for r in conf if r["significant"]]
    v, checks = verdict(prim, runs)
    res = {"runs": meta, "excluded_units": excl, "verdict": v, "checks": checks,
           "primary": describe(prim), "all_units": describe(conf),
           "per_run_primary": {k: paired(vv, "A") for k, vv in runs.items()},
           "per_run_all": {k: paired([r for r in conf if r["run"] == k], "A") for k in runs}}
    try:
        d, rows, e = load("tests/walk_pc_raw_D1.json")
        res["D1"] = {"run_id": d["run_id"], "excluded": e, "all_units": describe(rows),
                     "primary": describe([r for r in rows if r["significant"]]), "lattice": lattice_check(rows, d["prompts"])}
    except FileNotFoundError:
        res["D1"] = None
    json.dump(res, open("tests/walk_pc_results.json", "w"), indent=1, default=float)
    P, A = res["primary"], res["all_units"]
    print(f"VERDICT: {v}")
    print("checks:", checks)
    for name, blk in (("primary", P), ("all units", A)):
        s = blk["A"]
        print(f"[{name}] n={s['n']} differing={s['n_differing']} cont longer {s['cont_longer']} / fan longer {s['fan_longer']} "
              f"(frac {s['frac_cont_wins']:.2f}) p={s['wilcoxon_p']:.2g} mean A fan {s['mean_fan']:.3f} cont {s['mean_cont']:.3f}")
        c = blk["A_cert"]
        print(f"   A_cert: fan {c['mean_fan']:.3f} cont {c['mean_cont']:.3f} (p={c['wilcoxon_p']:.2g}); certified rate fan "
              f"{blk['fan']['certified_rate']:.2f} cont {blk['continuation']['certified_rate']:.2f}; images/walk fan "
              f"{blk['fan']['images_per_walk']:.1f} cont {blk['continuation']['images_per_walk']:.1f}")
    for k, s in res["per_run_primary"].items():
        print(f"   {k}: n={s['n']} mean A fan {s['mean_fan']:.3f} cont {s['mean_cont']:.3f} ({s['cont_longer']} vs {s['fan_longer']})")


if __name__ == "__main__":
    main()
