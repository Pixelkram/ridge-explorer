#!/usr/bin/env python3
"""Independent verifier for tests/PREREG_walk_continuation.md (continuation vs fan walk).

Written from the PREREG text and the raw-data format (tests/walk_pc_collect.py, backend/services/cascade.py) BEFORE
reading tests/walk_pc_analyze.py or tests/walk_pc_results.json. Reads only the raw files
tests/walk_pc_raw_C*.json (confirmatory runs) and writes tests/walk_pc_verify/verify_output.json and
tests/walk_pc_verify/verify_summary.txt.

Registered definitions implemented here (PREREG section in brackets):
  [Data]      unit = (crossing, direction); primary population = units whose origin crossing is significant, pooled
              over the confirmatory runs; secondary = all units; a unit with an errored (or non-complete) walk in
              either arm is excluded and counted.
  [Metrics]   A = min(0.30, sum ||c_i - c_{i-1}||_2) along the polyline origin -> steps[].center (fallback
              steps[].weights); origin = the unit's "origin" field (the crossing position).
              A_cert = the part of A made of polyline segments whose END station is certified
              (walk["cert"]["significant"][i]); segments taken in order up to the 0.30 cap, the last one truncated.
  [Decision]  D = A_cont - A_fan; differs iff |D| > 0.01.
              SUPPORTED iff (i) cont longer in >= 65 % of differing units, (ii) two-sided paired Wilcoxon on D
              (scipy default, zero_method="wilcox") p < 0.05, (iii) mean D > 0, (iv) mean A_cont > mean A_fan in a
              majority of the confirmatory runs contributing >= 2 primary units, (v) mean A_cert,cont >=
              mean A_cert,fan - 0.01.
              REFUTED iff fan longer in >= 65 % of differing units, p < 0.05 and mean D < 0.
              INCONCLUSIVE otherwise, incl. < 8 differing units, or (i)-(iv) with (v) failing.
"""
import glob
import json
import os
import re
import sys

import numpy as np
import scipy
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
CAP = 0.30          # budget 6 x 0.05
DIFF = 0.01         # a unit differs iff |D| > DIFF
WIN = 0.65          # (i)
ALPHA = 0.05        # (ii)
GUARD = 0.01        # (v)
MIN_DIFF_UNITS = 8  # fewer differing units -> INCONCLUSIVE
ARMS = ("fan", "continuation")


def run_label_key(path):
    m = re.search(r"walk_pc_raw_C(\d+)\.json$", path)
    return int(m.group(1)) if m else 10 ** 9


def load_confirmatory():
    paths = sorted(glob.glob(os.path.join(REPO, "tests", "walk_pc_raw_C*.json")), key=run_label_key)
    runs = []
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        d["_label"] = "C" + str(run_label_key(p))
        d["_path"] = os.path.relpath(p, REPO)
        runs.append(d)
    return runs


def centres(walk):
    out = []
    for s in walk.get("steps") or []:
        c = s.get("center")
        if c is None:
            c = s["weights"]
        out.append(np.asarray(c, dtype=float))
    return out


def seg_lengths(origin, cs):
    pts = [np.asarray(origin, dtype=float)] + list(cs)
    return [float(np.linalg.norm(pts[i] - pts[i - 1])) for i in range(1, len(pts))]


def metric_A(origin, walk):
    return min(CAP, float(sum(seg_lengths(origin, centres(walk)))))


def metric_A_cert(origin, walk):
    """Segments in order; segment i (ending at station i) contributes the part of it that fits under the 0.30 cap of
    the TOTAL polyline, and counts towards A_cert iff station i is certified."""
    L = seg_lengths(origin, centres(walk))
    sig = cert_flags(walk)
    cum, acc = 0.0, 0.0
    for li, si in zip(L, sig):
        fit = max(0.0, min(li, CAP - cum))
        if si:
            acc += fit
        cum += li
        if cum >= CAP:
            break
    return acc


def metric_A_cert_alt(origin, walk):
    """Sensitivity reading: sum of certified segment lengths, the cap applied to that sum only."""
    L = seg_lengths(origin, centres(walk))
    sig = cert_flags(walk)
    return min(CAP, float(sum(li for li, si in zip(L, sig) if si)))


def cert_flags(walk):
    cert = walk.get("cert") or {}
    sig = cert.get("significant") or []
    n = len(walk.get("steps") or [])
    if len(sig) != n:
        raise ValueError(f"cert flags {len(sig)} != stations {n} (walk {walk.get('walk_id')})")
    return [bool(v) for v in sig]


def walk_problem(walk):
    """None if the walk is usable; else a reason (errored / non-complete / certification not done)."""
    if walk is None:
        return "missing"
    if walk.get("error"):
        return f"error: {walk.get('error')}"
    if walk.get("status") != "complete":
        return f"status {walk.get('status')}"
    cert = walk.get("cert") or {}
    if cert.get("status") != "done":
        return f"cert status {cert.get('status')}"
    return None


def wilcoxon_default(D):
    D = np.asarray(D, dtype=float)
    if D.size == 0 or np.all(D == 0):
        return {"statistic": None, "pvalue": None, "note": "no non-zero differences"}
    r = wilcoxon(D)   # scipy default: zero_method="wilcox", correction=False, two-sided, method="auto"
    n_zero = int(np.sum(D == 0))
    nz = D[D != 0]
    has_ties = len(np.unique(np.abs(nz))) < len(nz)
    if D.size > 50:
        meth = "asymptotic (n > 50)"
    elif not (has_ties or n_zero > 0):
        meth = "exact"
    elif D.size <= 13:
        meth = "permutation"
    else:
        meth = "asymptotic (zeros/ties present, no continuity correction)"
    return {"statistic": float(r.statistic), "pvalue": float(r.pvalue), "n": int(D.size), "n_zero": n_zero,
            "n_nonzero": int(nz.size), "abs_ties_among_nonzero": bool(has_ties), "method_used": meth}


def summarise(units):
    """Registered statistics over a list of analysed units (dicts with A_fan, A_cont, ...)."""
    if not units:
        return {"n": 0}
    Af = np.array([u["A_fan"] for u in units])
    Ac = np.array([u["A_cont"] for u in units])
    Cf = np.array([u["Acert_fan"] for u in units])
    Cc = np.array([u["Acert_cont"] for u in units])
    D = Ac - Af
    diff = np.abs(D) > DIFF
    wins = int(np.sum(D > DIFF))
    losses = int(np.sum(D < -DIFF))
    ndiff = int(np.sum(diff))
    return {
        "n": len(units),
        "n_differing": ndiff,
        "cont_longer": wins,
        "fan_longer": losses,
        "win_fraction_cont": (wins / ndiff) if ndiff else None,
        "win_fraction_fan": (losses / ndiff) if ndiff else None,
        "n_D_exactly_zero": int(np.sum(D == 0)),
        "n_small_nonzero_D": int(np.sum((D != 0) & ~diff)),
        "mean_D": float(D.mean()),
        "median_D": float(np.median(D)),
        "mean_A_fan": float(Af.mean()),
        "mean_A_cont": float(Ac.mean()),
        "mean_Acert_fan": float(Cf.mean()),
        "mean_Acert_cont": float(Cc.mean()),
        "mean_Dcert": float((Cc - Cf).mean()),
        "wilcoxon_D": wilcoxon_default(D),
        "wilcoxon_Dcert": wilcoxon_default(Cc - Cf),
        "wilcoxon_D_differing_only": wilcoxon_default(D[diff]),
    }


def verdict(units, per_run_primary):
    s = summarise(units)
    if s["n"] == 0:
        return "INCONCLUSIVE", {"reason": "no units"}, s
    p = s["wilcoxon_D"]["pvalue"]
    ndiff = s["n_differing"]
    elig = {r: v for r, v in per_run_primary.items() if v["n"] >= 2}
    run_ok = {r: bool(v["mean_A_cont"] > v["mean_A_fan"]) for r, v in elig.items()}
    crit = {
        "i_cont_longer_ge_65pct_of_differing": bool(ndiff > 0 and s["win_fraction_cont"] >= WIN),
        "ii_wilcoxon_p_lt_0.05": bool(p is not None and p < ALPHA),
        "iii_mean_D_gt_0": bool(s["mean_D"] > 0),
        "iv_majority_of_runs": bool(len(elig) > 0 and sum(run_ok.values()) > len(elig) / 2),
        "v_guard_Acert_cont_ge_Acert_fan_minus_0.01": bool(s["mean_Acert_cont"] >= s["mean_Acert_fan"] - GUARD),
        "n_differing_ge_8": bool(ndiff >= MIN_DIFF_UNITS),
        "iv_detail": {"eligible_runs": sorted(elig), "cont_mean_higher": run_ok},
    }
    refuted = bool(ndiff > 0 and s["win_fraction_fan"] >= WIN and p is not None and p < ALPHA and s["mean_D"] < 0)
    if ndiff < MIN_DIFF_UNITS:
        v = "INCONCLUSIVE"
        why = f"fewer than {MIN_DIFF_UNITS} differing units"
    elif all(crit[c] for c in ("i_cont_longer_ge_65pct_of_differing", "ii_wilcoxon_p_lt_0.05", "iii_mean_D_gt_0",
                               "iv_majority_of_runs", "v_guard_Acert_cont_ge_Acert_fan_minus_0.01")):
        v = "SUPPORTED"
        why = "(i)-(v) all hold"
    elif refuted:
        v = "REFUTED"
        why = "fan longer in >= 65 % of differing units, p < 0.05, mean D < 0"
    elif all(crit[c] for c in ("i_cont_longer_ge_65pct_of_differing", "ii_wilcoxon_p_lt_0.05", "iii_mean_D_gt_0",
                               "iv_majority_of_runs")):
        v = "INCONCLUSIVE"
        why = "longer but not validated ((i)-(iv) hold, (v) fails)"
    else:
        v = "INCONCLUSIVE"
        why = "criteria not met: " + ", ".join(c for c in crit if c != "iv_detail" and not crit[c])
    crit["reason"] = why
    crit["refuted_condition"] = refuted
    return v, crit, s


def main():
    runs = load_confirmatory()
    if not runs:
        print("no confirmatory raw files found", file=sys.stderr)
        sys.exit(1)
    all_units, excluded = [], []
    per_run = {}
    for r in runs:
        lab = r["_label"]
        rows = []
        for u in r["units"]:
            fan, cont = u.get("fan"), u.get("continuation")
            prob = {a: walk_problem(u.get(a)) for a in ARMS}
            base = {"run": lab, "cid": u["cid"], "dir": u["dir"], "significant": bool(u["significant"]),
                    "b": u["b"]}
            if prob["fan"] or prob["continuation"]:
                excluded.append(base | {"problems": prob})
                continue
            o = u["origin"]
            row = base | {
                "A_fan": metric_A(o, fan), "A_cont": metric_A(o, cont),
                "Acert_fan": metric_A_cert(o, fan), "Acert_cont": metric_A_cert(o, cont),
                "Acert_alt_fan": metric_A_cert_alt(o, fan), "Acert_alt_cont": metric_A_cert_alt(o, cont),
                "raw_len_fan": float(sum(seg_lengths(o, centres(fan)))),
                "raw_len_cont": float(sum(seg_lengths(o, centres(cont)))),
                "n_st_fan": len(fan["steps"]), "n_st_cont": len(cont["steps"]),
                "n_cert_fan": int(sum(cert_flags(fan))), "n_cert_cont": int(sum(cert_flags(cont))),
                "img_fan": fan.get("images"), "img_cont": cont.get("images"),
                "end_fan": (fan.get("notes") or ["budget"])[-1], "end_cont": (cont.get("notes") or ["budget"])[-1],
            }
            row["D"] = row["A_cont"] - row["A_fan"]
            rows.append(row)
        all_units += rows
        per_run[lab] = {"run_id": r["run_id"], "set": r["set"], "seed": r["seed"], "n_units_raw": len(r["units"]),
                        "n_crossings": len(r["crossings"]),
                        "n_sig_crossings": int(sum(bool(x["significant"]) for x in r["crossings"])),
                        "bg_p95": r["bg_p95"], "rows": rows}

    primary = [u for u in all_units if u["significant"]]
    per_run_primary = {lab: summarise([u for u in v["rows"] if u["significant"]]) for lab, v in per_run.items()}
    per_run_all = {lab: summarise(v["rows"]) for lab, v in per_run.items()}
    v_prim, crit_prim, s_prim = verdict(primary, per_run_primary)
    # the rule applied to all units (secondary, descriptive only -- no verdict is registered for it)
    v_all, crit_all, s_all = verdict(all_units, per_run_all)

    # stopping-rule count as run_all.py counts it (units from significant crossings, before exclusions)
    pooled = 0
    stop_trace = []
    for lab, v in per_run.items():
        n_sig_units_raw = 2 * v["n_sig_crossings"]
        pooled += n_sig_units_raw
        stop_trace.append({"run": lab, "primary_units_raw": n_sig_units_raw, "pooled_after": pooled})

    out = {
        "scipy": scipy.__version__, "numpy": np.__version__,
        "runs": [{k: v for k, v in pr.items() if k != "rows"} | {"label": lab} for lab, pr in per_run.items()],
        "n_units_total": len(all_units) + len(excluded),
        "n_excluded": len(excluded), "excluded": excluded,
        "primary": {"verdict": v_prim, "criteria": crit_prim, "stats": s_prim},
        "all_units_rule_applied_descriptive": {"verdict_if_rule_applied": v_all, "criteria": crit_all,
                                               "stats": s_all},
        "per_run_primary": per_run_primary,
        "per_run_all": per_run_all,
        "stopping_rule_counts": stop_trace,
        "units": all_units,
    }
    with open(os.path.join(HERE, "verify_output.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)

    L = []
    L.append(f"Independent verification of PREREG_walk_continuation.md (scipy {scipy.__version__})")
    L.append(f"runs: " + ", ".join(f"{lab}={v['run_id']} ({v['set']}, seed {v['seed']}, {v['n_crossings']} crossings, "
                                    f"{v['n_sig_crossings']} sig, {v['n_units_raw']} units)" for lab, v in per_run.items()))
    L.append(f"units: {out['n_units_total']} total, {len(excluded)} excluded")
    for e in excluded:
        L.append(f"  excluded {e['run']} cid {e['cid']} dir {e['dir']:+d}: {e['problems']}")
    for name, s, v, c in (("PRIMARY", s_prim, v_prim, crit_prim), ("ALL UNITS (descriptive)", s_all, v_all, crit_all)):
        w = s["wilcoxon_D"]
        L.append(f"{name}: n={s['n']}  differing={s['n_differing']}  cont longer={s['cont_longer']}  "
                 f"fan longer={s['fan_longer']}  win frac={s['win_fraction_cont']}")
        L.append(f"  mean A fan={s['mean_A_fan']:.4f} cont={s['mean_A_cont']:.4f}  mean D={s['mean_D']:.4f}  "
                 f"median D={s['median_D']:.4f}")
        L.append(f"  Wilcoxon D: W={w['statistic']} p={w['pvalue']} ({w.get('method_used')}, n_zero={w.get('n_zero')})")
        L.append(f"  mean A_cert fan={s['mean_Acert_fan']:.4f} cont={s['mean_Acert_cont']:.4f}  "
                 f"guard (cont >= fan - 0.01): {c['v_guard_Acert_cont_ge_Acert_fan_minus_0.01']}")
        L.append(f"  Wilcoxon D_cert: p={s['wilcoxon_Dcert']['pvalue']}")
        L.append(f"  criteria: " + ", ".join(f"{k}={c[k]}" for k in c if k not in ("iv_detail", "reason")))
        L.append(f"  (iv) detail: {c['iv_detail']}")
        L.append(f"  VERDICT: {v}  ({c['reason']})")
    L.append("per-run (primary):")
    for lab, s in per_run_primary.items():
        if s["n"]:
            L.append(f"  {lab}: n={s['n']} diff={s['n_differing']} +{s['cont_longer']}/-{s['fan_longer']} "
                     f"A fan={s['mean_A_fan']:.4f} cont={s['mean_A_cont']:.4f} meanD={s['mean_D']:+.4f} "
                     f"Acert fan={s['mean_Acert_fan']:.4f} cont={s['mean_Acert_cont']:.4f}")
        else:
            L.append(f"  {lab}: n=0")
    L.append("per-run (all units):")
    for lab, s in per_run_all.items():
        L.append(f"  {lab}: n={s['n']} diff={s['n_differing']} +{s['cont_longer']}/-{s['fan_longer']} "
                 f"A fan={s['mean_A_fan']:.4f} cont={s['mean_A_cont']:.4f} meanD={s['mean_D']:+.4f} "
                 f"Acert fan={s['mean_Acert_fan']:.4f} cont={s['mean_Acert_cont']:.4f}")
    L.append("stopping rule (pooled units from significant crossings, as run_all counts them): "
             + ", ".join(f"{t['run']}:{t['primary_units_raw']}->{t['pooled_after']}" for t in stop_trace))
    txt = "\n".join(L)
    with open(os.path.join(HERE, "verify_summary.txt"), "w") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
