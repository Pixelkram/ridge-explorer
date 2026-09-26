#!/usr/bin/env python3
"""Integrity + adversarial checks for the continuation-vs-fan A/B (PREREG_walk_continuation.md).
Reads only tests/walk_pc_raw_C*.json (+ tests/walk_pc_results.json for the number-by-number comparison) and the
verifier's own verify.py functions. Writes tests/walk_pc_verify/checks_output.json and checks_summary.txt."""
import itertools
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import wilcoxon, binomtest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify as V  # noqa: E402

HERE, REPO = V.HERE, V.REPO
CAP, DIFF = V.CAP, V.DIFF
EPS_HALF = 0.047 / 2
L = []          # text summary lines
OUT = {}


def say(s=""):
    L.append(s)
    print(s)


def uv_of(walk, pts):
    pl = walk["plane"]
    x0 = np.asarray(pl["x0"], float)
    E = np.stack([np.asarray(pl["t"], float), np.asarray(pl["n"], float)], axis=1)
    return [E.T @ (np.asarray(p, float) - x0) for p in pts], E, x0


def stats_paired(a_cont, a_fan):
    a_cont, a_fan = np.asarray(a_cont, float), np.asarray(a_fan, float)
    D = a_cont - a_fan
    dif = np.abs(D) > DIFF
    w, l = int(np.sum(D > DIFF)), int(np.sum(D < -DIFF))
    p = float(wilcoxon(D).pvalue) if np.any(D != 0) else None
    return {"n": int(D.size), "mean_fan": float(a_fan.mean()), "mean_cont": float(a_cont.mean()),
            "mean_D": float(D.mean()), "differing": int(dif.sum()), "cont_longer": w, "fan_longer": l,
            "frac_cont": (w / dif.sum()) if dif.sum() else None, "wilcoxon_p": p}


def fmt(s):
    return (f"n={s['n']} fan={s['mean_fan']:.4f} cont={s['mean_cont']:.4f} meanD={s['mean_D']:+.4f} "
            f"diff={s['differing']} +{s['cont_longer']}/-{s['fan_longer']} p={s['wilcoxon_p']}")


# ------------------------------------------------------------------ load
runs = V.load_confirmatory()
units = []   # flat list of (run, raw unit)
for r in runs:
    for u in r["units"]:
        units.append((r, u))

# ------------------------------------------------------------------ 1. integrity
say("=== INTEGRITY ===")
issues = []
integ = {}
for r in runs:
    lab = r["_label"]
    xs = {x["cid"]: x for x in r["crossings"]}
    # significance flags vs b > bg_p95
    bad_sig = [x["cid"] for x in r["crossings"] if bool(x["significant"]) != bool(x["b"] is not None and x["b"] > r["bg_p95"])]
    margins = sorted(abs(x["b"] - r["bg_p95"]) for x in r["crossings"])
    # units = 2 x crossings, every (cid, dir) exactly once
    keys = Counter((u["cid"], u["dir"]) for u in r["units"])
    want = Counter((c, d) for c in xs for d in (1, -1))
    # unit fields vs crossing list
    bad_unit = [(u["cid"], u["dir"]) for u in r["units"]
                if u["significant"] != xs[u["cid"]]["significant"] or u["b"] != xs[u["cid"]]["b"]
                or u["origin"] != xs[u["cid"]]["weights"]]
    integ[lab] = {"n_crossings": len(xs), "n_units": len(r["units"]), "units_eq_2x": len(r["units"]) == 2 * len(xs),
                  "every_cid_dir_once": keys == want, "sig_flag_mismatch": bad_sig,
                  "min_abs_b_minus_p95": margins[0], "unit_vs_crossing_mismatch": bad_unit}
    if bad_sig or bad_unit or keys != want:
        issues.append(f"{lab}: flag/unit mismatch {bad_sig} {bad_unit} keys_ok={keys == want}")
    say(f"{lab}: crossings {len(xs)}, units {len(r['units'])} (2x: {len(r['units']) == 2 * len(xs)}), every (cid,dir) once: "
        f"{keys == want}; sig flag == (b > bg_p95): {not bad_sig} (closest |b - p95| = {margins[0]:.4f}); "
        f"unit sig/b/origin == crossing: {not bad_unit}")

plane_dev, x0_dev, orth_dev = 0.0, 0.0, 0.0
simplex_min, simplex_sum_dev, weights_min = 1.0, 0.0, 1.0
offplane_max = {"fan": 0.0, "continuation": 0.0}
fan_u_dev, fan_v_max = 0.0, 0.0
mode_bad, jvp_bad, n_fan_over6 = [], [], 0
cont_budget_viol, cont_attempts_max, cont_img_mismatch, cont_trace_mismatch = [], 0, [], []
cert_issues, thr_by_run = [], defaultdict(set)
walk_ids, thumbs_by_run = Counter(), defaultdict(Counter)
step_sigs = Counter()
for r, u in units:
    lab = r["_label"]
    f, c = u["fan"], u["continuation"]
    for key in ("x0", "t", "n"):
        plane_dev = max(plane_dev, float(np.max(np.abs(np.asarray(f["plane"][key]) - np.asarray(c["plane"][key])))))
    x0_dev = max(x0_dev, float(np.max(np.abs(np.asarray(f["plane"]["x0"]) - np.asarray(u["origin"])))))
    t, n = np.asarray(f["plane"]["t"]), np.asarray(f["plane"]["n"])
    orth_dev = max(orth_dev, abs(t @ n), abs(np.linalg.norm(t) - 1), abs(np.linalg.norm(n) - 1), abs(t.sum()), abs(n.sum()))
    for arm, w in (("fan", f), ("continuation", c)):
        if w["mode"] != arm:
            mode_bad.append((lab, u["cid"], u["dir"], arm))
        if w.get("jvp") is not None:
            jvp_bad.append((lab, u["cid"], u["dir"], arm))
        walk_ids[w["walk_id"]] += 1
        cs = V.centres(w)
        for s in w["steps"]:
            for key in ("center", "weights"):
                v = np.asarray(s[key], float)
                simplex_min = min(simplex_min, float(v.min()))
                simplex_sum_dev = max(simplex_sum_dev, abs(float(v.sum()) - 1))
            if s.get("thumb", -1) >= 0:
                thumbs_by_run[lab][s["thumb"]] += 1
        uv, E, x0 = uv_of(w, cs)
        for cc, q in zip(cs, uv):
            offplane_max[arm] = max(offplane_max[arm], float(np.linalg.norm(cc - x0 - E @ q)))
        if cs:
            step_sigs[tuple(np.round(np.concatenate(cs), 12))] += 1
        # certification
        ce = w["cert"]
        ns = len(w["steps"])
        if ce.get("status") != "done" or len(ce.get("b") or []) != ns or len(ce.get("significant") or []) != ns:
            cert_issues.append((lab, u["cid"], u["dir"], arm, "status/len"))
        if ns:
            thr_by_run[lab].add(ce["threshold"])
            if ce.get("threshold_kind") != "held-out seeds":
                cert_issues.append((lab, u["cid"], u["dir"], arm, "threshold_kind " + str(ce.get("threshold_kind"))))
            if ce.get("seeds") != [r["seed"] + 997 * j for j in (1, 2, 3)]:
                cert_issues.append((lab, u["cid"], u["dir"], arm, f"seeds {ce.get('seeds')}"))
            if ce.get("images") != 6 * ns:
                cert_issues.append((lab, u["cid"], u["dir"], arm, f"images {ce.get('images')} != {6 * ns}"))
            if any(bb is None for bb in ce["b"]):
                cert_issues.append((lab, u["cid"], u["dir"], arm, "b None"))
            if [bool(bb > ce["threshold"]) for bb in ce["b"]] != [bool(v) for v in ce["significant"]]:
                cert_issues.append((lab, u["cid"], u["dir"], arm, "flag != b > thr"))
    # fan geometry: tangential coordinate exactly 0.05 j, transversal within the widest arm
    uvf, _, _ = uv_of(f, V.centres(f))
    for j, q in enumerate(uvf, start=1):
        fan_u_dev = max(fan_u_dev, abs(q[0] - 0.05 * j))
        fan_v_max = max(fan_v_max, abs(q[1]))
    if len(f["steps"]) > 6:
        n_fan_over6 += 1
    # continuation: stations == successful attempts, images == sum(len(offs)) + stations, budget respected
    succ = sum(1 for tr in c["trace"] if tr["chosen"] is not None)
    if succ != len(c["steps"]):
        cont_trace_mismatch.append((lab, u["cid"], u["dir"], succ, len(c["steps"])))
    img = sum(len(tr["offs"]) for tr in c["trace"]) + succ
    if img != c["images"]:
        cont_img_mismatch.append((lab, u["cid"], u["dir"], img, c["images"]))
    cont_attempts_max = max(cont_attempts_max, len(c["trace"]))
    uvc, _, _ = uv_of(c, V.centres(c))
    pts = [np.zeros(2)] + uvc
    arc_inplane = np.cumsum([np.linalg.norm(pts[i] - pts[i - 1]) for i in range(1, len(pts))]) if uvc else []
    if len(arc_inplane) >= 2 and arc_inplane[-2] >= 0.30 - 1e-9:
        cont_budget_viol.append((lab, u["cid"], u["dir"], float(arc_inplane[-2])))

dup_walk_ids = [k for k, v in walk_ids.items() if v > 1]
dup_thumbs = {lab: [k for k, v in cnt.items() if v > 1] for lab, cnt in thumbs_by_run.items()}
dup_steps = sum(1 for v in step_sigs.values() if v > 1)
integ.update({
    "max_plane_dev_between_arms": plane_dev, "max_x0_minus_origin": x0_dev, "max_t_n_orthonormality_dev": orth_dev,
    "station_min_coordinate": simplex_min, "station_max_abs_sum_minus_1": simplex_sum_dev,
    "max_offplane_residual": offplane_max, "fan_max_dev_tangential_coord_from_0.05j": fan_u_dev,
    "fan_max_abs_transversal_coord": fan_v_max, "fan_walks_over_6_stations": n_fan_over6,
    "mode_mismatch": mode_bad, "jvp_not_none": jvp_bad, "cont_stations_ne_successful_attempts": cont_trace_mismatch,
    "cont_images_ne_offsets_plus_bisections": cont_img_mismatch, "cont_max_attempts": cont_attempts_max,
    "cont_station_added_after_budget": cont_budget_viol, "cert_issues": cert_issues,
    "cert_threshold_values_per_run": {k: sorted(v) for k, v in thr_by_run.items()},
    "duplicate_walk_ids": dup_walk_ids, "duplicate_thumbs_per_run": dup_thumbs,
    "identical_station_lists_across_walks": dup_steps,
})
say(f"plane identical between arms: max |diff| = {plane_dev:.2e}; plane x0 vs unit origin: {x0_dev:.2e}; "
    f"t,n orthonormal & sum-zero: max dev {orth_dev:.2e}")
say(f"station centres/weights in simplex: min coord {simplex_min:.3e}, max |sum-1| {simplex_sum_dev:.2e}; "
    f"off-plane residual fan {offplane_max['fan']:.2e} cont {offplane_max['continuation']:.2e}")
say(f"fan: tangential coord == 0.05 j (max dev {fan_u_dev:.2e}), max |transversal| {fan_v_max:.4f}, walks > 6 stations: {n_fan_over6}")
say(f"cont: stations == successful attempts: {not cont_trace_mismatch}; images == sum(offs)+bisections: "
    f"{not cont_img_mismatch}; max attempts {cont_attempts_max} (limit 28); station after budget: {cont_budget_viol}")
say(f"modes ok: {not mode_bad}; jvp None: {not jvp_bad}; cert issues: {cert_issues}; "
    f"cert threshold per run: { {k: [round(x, 4) for x in v] for k, v in thr_by_run.items()} }")
say(f"duplicate walk ids: {dup_walk_ids}; duplicate station thumbs: {dup_thumbs}; identical station lists: {dup_steps}")
notes = Counter()
for r, u in units:
    for arm in ("fan", "continuation"):
        for nt in u[arm]["notes"]:
            notes[(arm, nt.split(" at station")[0].split(" before station")[0])] += 1
integ["note_kinds"] = {f"{a}|{k}": v for (a, k), v in sorted(notes.items())}
say(f"note kinds: {integ['note_kinds']}")
multi_notes = [(r['_label'], u['cid'], u['dir'], a, u[a]['notes']) for r, u in units for a in ("fan", "continuation")
               if len(u[a]['notes']) > 1]
integ["walks_with_multiple_notes"] = multi_notes
say(f"walks with >1 note: {len(multi_notes)}: {multi_notes}")
OUT["integrity"] = integ

# ------------------------------------------------------------------ 2. per-unit geometry
rows = []
for r, u in units:
    row = {"run": r["_label"], "cid": u["cid"], "dir": u["dir"], "sig": bool(u["significant"])}
    for arm, key in (("fan", "f"), ("continuation", "c")):
        w = u[arm]
        cs = V.centres(w)
        o = np.asarray(u["origin"], float)
        segs = V.seg_lengths(o, cs)
        uv, E, x0 = uv_of(w, cs)
        pts = [np.zeros(2)] + uv
        # turning angle of each segment relative to the previous one (first: relative to +t)
        ang = []
        prev = np.array([1.0, 0.0])
        for i in range(1, len(pts)):
            d = pts[i] - pts[i - 1]
            nd = np.linalg.norm(d)
            if nd < 1e-12:
                ang.append(0.0)
                continue
            dd = d / nd
            ang.append(float(np.degrees(np.arctan2(prev[0] * dd[1] - prev[1] * dd[0], prev @ dd))))
            prev = dd
        full = [o] + cs
        disp = float(np.linalg.norm(cs[-1] - o)) if cs else 0.0
        # decimated polyline (origin, every 2nd station, always the last)
        dec_idx = list(range(2, len(full), 2))
        if cs and (len(full) - 1) not in dec_idx:
            dec_idx.append(len(full) - 1)
        dec = [full[0]] + [full[i] for i in dec_idx]
        a_dec = min(CAP, sum(float(np.linalg.norm(dec[i] - dec[i - 1])) for i in range(1, len(dec))))
        # 3-point moving average of interior vertices, endpoints (origin, last) fixed
        if len(full) >= 3:
            sm = [full[0]] + [(full[i - 1] + full[i] + full[i + 1]) / 3 for i in range(1, len(full) - 1)] + [full[-1]]
        else:
            sm = full
        a_sm = min(CAP, sum(float(np.linalg.norm(sm[i] - sm[i - 1])) for i in range(1, len(sm))))
        a_segcap = min(CAP, sum(min(s, 0.05) for s in segs))
        # progress along the walk's own tangent seed t (in-plane u coordinate of the last station, >= 0)
        u_last = float(max(0.0, uv[-1][0])) if uv else 0.0
        # max backward excursion along t (a curl back past the origin)
        u_min = float(min([q[0] for q in uv], default=0.0))
        row[key] = {
            "A": V.metric_A(u["origin"], w), "A_cert": V.metric_A_cert(u["origin"], w),
            "raw": float(sum(segs)), "segs": segs, "ang": ang, "disp": min(CAP, disp), "dec": a_dec, "smooth": a_sm,
            "segcap": a_segcap, "u_last": min(CAP, u_last), "u_min": u_min, "n": len(cs),
            "ncert": int(sum(V.cert_flags(w))), "flags": V.cert_flags(w), "b": w["cert"].get("b") or [],
            "thr": w["cert"].get("threshold"), "images": w["images"], "wall": w.get("wall_s"),
            "end": (w.get("notes") or [""])[-1],
            # fan localisation: |weights - center| = half-width of the final bracket
            "halfw": [float(np.linalg.norm(np.asarray(s["weights"]) - np.asarray(s["center"]))) for s in w["steps"]],
            "trace": w.get("trace") or [],
        }
    rows.append(row)
prim = [r for r in rows if r["sig"]]

# ------------------------------------------------------------------ 3. compare every registered number
say("\n=== COMPARISON WITH tests/walk_pc_results.json ===")
res = json.load(open(os.path.join(REPO, "tests", "walk_pc_results.json")))
cmp_rows = []


def check(name, mine, theirs, tol=1e-12):
    if isinstance(mine, (int, float, np.floating, np.integer)) and isinstance(theirs, (int, float)):
        ok = abs(float(mine) - float(theirs)) <= tol * max(1.0, abs(float(theirs)))
    else:
        ok = mine == theirs
    cmp_rows.append({"name": name, "mine": mine, "theirs": theirs, "match": bool(ok)})
    return ok


def desc(rs):
    out = {}
    for arm, key in (("fan", "f"), ("continuation", "c")):
        st = sum(r[key]["n"] for r in rs)
        ce = sum(r[key]["ncert"] for r in rs)
        img = sum(r[key]["images"] for r in rs)
        A = sum(r[key]["A"] for r in rs)
        Ac = sum(r[key]["A_cert"] for r in rs)
        by = defaultdict(list)
        for r in rs:
            for i, fl in enumerate(r[key]["flags"], start=1):
                by[str(i)].append(fl)
        ends = Counter()
        for r in rs:
            nt = r[key]["end"]
            ends["budget" if not nt else "identity" if "changed identity" in nt else "ended" if "ended before" in nt
                 else "edge" if ("edge" in nt or "face" in nt) else "other"] += 1
        out[arm] = {"stations": st, "certified": ce, "certified_rate": ce / st,
                    "certified_rate_by_station": {k: [round(float(np.mean(v)), 3), len(v)] for k, v in sorted(by.items())},
                    "images_per_walk": img / len(rs), "images_per_unit_A": img / A, "images_per_unit_A_cert": img / Ac,
                    "median_wall_s": float(np.median([r[key]["wall"] for r in rs])), "end_classes": dict(ends)}
    return out


for pop_name, rs in (("primary", prim), ("all_units", rows)):
    mine_A = stats_paired([r["c"]["A"] for r in rs], [r["f"]["A"] for r in rs])
    mine_C = stats_paired([r["c"]["A_cert"] for r in rs], [r["f"]["A_cert"] for r in rs])
    for metric, mine in (("A", mine_A), ("A_cert", mine_C)):
        th = res[pop_name][metric]
        check(f"{pop_name}.{metric}.n", mine["n"], th["n"])
        check(f"{pop_name}.{metric}.n_differing", mine["differing"], th["n_differing"])
        check(f"{pop_name}.{metric}.cont_longer", mine["cont_longer"], th["cont_longer"])
        check(f"{pop_name}.{metric}.fan_longer", mine["fan_longer"], th["fan_longer"])
        check(f"{pop_name}.{metric}.frac_cont_wins", mine["frac_cont"], th["frac_cont_wins"])
        check(f"{pop_name}.{metric}.wilcoxon_p", mine["wilcoxon_p"], th["wilcoxon_p"], tol=1e-9)
        check(f"{pop_name}.{metric}.mean_D", mine["mean_D"], th["mean_D"], tol=1e-9)
        check(f"{pop_name}.{metric}.mean_fan", mine["mean_fan"], th["mean_fan"], tol=1e-9)
        check(f"{pop_name}.{metric}.mean_cont", mine["mean_cont"], th["mean_cont"], tol=1e-9)
    d = desc(rs)
    for arm in ("fan", "continuation"):
        for k, v in d[arm].items():
            tv = res[pop_name][arm][k]
            if isinstance(v, dict):
                check(f"{pop_name}.{arm}.{k}", {kk: list(vv) if isinstance(vv, list) else vv for kk, vv in v.items()},
                      {kk: list(vv) if isinstance(vv, list) else vv for kk, vv in tv.items()})
            else:
                check(f"{pop_name}.{arm}.{k}", v, tv, tol=1e-9)
        check(f"{pop_name}.{arm}.cert_missing", 0, res[pop_name][arm]["cert_missing"])
for lab in [r["_label"] for r in runs]:
    for pop_name, rs in (("per_run_primary", [r for r in prim if r["run"] == lab]),
                         ("per_run_all", [r for r in rows if r["run"] == lab])):
        s = stats_paired([r["c"]["A"] for r in rs], [r["f"]["A"] for r in rs])
        th = res[pop_name][lab]
        for mk, tk in (("n", "n"), ("differing", "n_differing"), ("cont_longer", "cont_longer"),
                       ("fan_longer", "fan_longer"), ("frac_cont", "frac_cont_wins"), ("wilcoxon_p", "wilcoxon_p"),
                       ("mean_D", "mean_D"), ("mean_fan", "mean_fan"), ("mean_cont", "mean_cont")):
            check(f"{pop_name}.{lab}.{tk}", s[mk], th[tk], tol=1e-9)
vo = json.load(open(os.path.join(HERE, "verify_output.json")))
check("verdict", vo["primary"]["verdict"], res["verdict"])
crit = vo["primary"]["criteria"]
check("checks.i", crit["i_cont_longer_ge_65pct_of_differing"], res["checks"]["i_frac>=0.65"])
check("checks.ii", crit["ii_wilcoxon_p_lt_0.05"], res["checks"]["ii_p<0.05"])
check("checks.iii", crit["iii_mean_D_gt_0"], res["checks"]["iii_meanD>0"])
check("checks.iv", crit["iv_majority_of_runs"], res["checks"]["iv_majority"])
check("checks.iv_text", f"{sum(crit['iv_detail']['cont_mean_higher'].values())}/{len(crit['iv_detail']['eligible_runs'])} "
      "eligible runs favour continuation", res["checks"]["iv_runs"])
check("checks.v", crit["v_guard_Acert_cont_ge_Acert_fan_minus_0.01"], res["checks"]["v_guard"])
for lab, m in res["runs"].items():
    rr = next(r for r in runs if r["_label"] == lab)
    check(f"runs.{lab}", m, {"run_id": rr["run_id"], "set": rr["set"], "seed": rr["seed"], "units": len(rr["units"]),
                              "crossings": len(rr["crossings"]),
                              "significant": sum(x["significant"] for x in rr["crossings"]), "bg_p95": rr["bg_p95"]})
check("excluded_units", res["excluded_units"], {r["_label"]: 0 for r in runs})
check("D1", res["D1"], None)
mism = [c for c in cmp_rows if not c["match"]]
say(f"compared {len(cmp_rows)} registered numbers/fields; mismatches: {len(mism)}")
for c in mism:
    say(f"  MISMATCH {c['name']}: mine={c['mine']} theirs={c['theirs']}")
OUT["comparison"] = {"n_compared": len(cmp_rows), "mismatches": mism}

# ------------------------------------------------------------------ 4. adversarial
say("\n=== ADVERSARIAL (primary population unless noted) ===")
adv = {}


def pooled(rs, key, field):
    return [x for r in rs for x in r[key][field]]


for pop_name, rs in (("primary", prim), ("all", rows)):
    blk = {}
    for arm, key in (("fan", "f"), ("cont", "c")):
        sg = np.array(pooled(rs, key, "segs"))
        first = np.array([r[key]["segs"][0] for r in rs if r[key]["segs"]])
        later = np.array([s for r in rs for s in r[key]["segs"][1:]])
        ang = np.abs(np.array([a for r in rs for a in r[key]["ang"][1:]]))
        ang1 = np.abs(np.array([r[key]["ang"][0] for r in rs if r[key]["ang"]]))
        blk[arm] = {
            "n_segs": int(sg.size),
            "seg_len_quantiles_10_25_50_75_90_max": [round(float(q), 4) for q in np.quantile(sg, [.1, .25, .5, .75, .9, 1])],
            "seg_mean": float(sg.mean()),
            "first_seg_median_mean_max": [float(np.median(first)), float(first.mean()), float(first.max())],
            "later_seg_median_mean_max": [float(np.median(later)), float(later.mean()), float(later.max())] if later.size else None,
            "frac_seg_gt_0.06": float(np.mean(sg > 0.06)), "frac_seg_gt_0.08": float(np.mean(sg > 0.08)),
            "turn_angle_deg_median_p90_max(later)": [float(np.median(ang)), float(np.quantile(ang, .9)), float(ang.max())] if ang.size else None,
            "frac_turn_gt_45deg(later)": float(np.mean(ang > 45)) if ang.size else None,
            "first_seg_angle_to_t_deg_median_p90": [float(np.median(ang1)), float(np.quantile(ang1, .9))],
            "tortuosity_raw_over_disp_median": float(np.median([r[key]["raw"] / max(np.linalg.norm(1e-12), 1e-12)
                                                                 if False else (r[key]["raw"] / r[key]["disp"] if r[key]["disp"] > 0 else 1.0)
                                                                 for r in rs if r[key]["n"] >= 2])),
            "walks_curling_behind_origin_along_t(u<0)": int(sum(1 for r in rs if r[key]["u_min"] < 0)),
        }
    adv[f"segments_{pop_name}"] = blk
say("consecutive-station distances (primary):")
for arm in ("fan", "cont"):
    b = adv["segments_primary"][arm]
    say(f"  {arm}: segs {b['n_segs']}, quantiles(10,25,50,75,90,max) {b['seg_len_quantiles_10_25_50_75_90_max']}, "
        f"mean {b['seg_mean']:.4f}; first seg median/mean/max {np.round(b['first_seg_median_mean_max'], 4).tolist()}; "
        f"later {np.round(b['later_seg_median_mean_max'], 4).tolist()}; >0.06: {b['frac_seg_gt_0.06']:.2f}, "
        f">0.08: {b['frac_seg_gt_0.08']:.2f}; |turn| later median/p90/max {np.round(b['turn_angle_deg_median_p90_max(later)'], 1).tolist()} "
        f"(>45deg {b['frac_turn_gt_45deg(later)']:.2f}); first seg angle to t median/p90 "
        f"{np.round(b['first_seg_angle_to_t_deg_median_p90'], 1).tolist()}; tortuosity median {b['tortuosity_raw_over_disp_median']:.3f}; "
        f"walks with a station behind the origin along t: {b['walks_curling_behind_origin_along_t(u<0)']}")

alt = {}
for pop_name, rs in (("primary", prim), ("all", rows)):
    alt[pop_name] = {}
    for m in ("A", "A_cert", "disp", "dec", "smooth", "segcap", "u_last"):
        alt[pop_name][m] = stats_paired([r["c"][m] for r in rs], [r["f"][m] for r in rs])
    # station counts and certified-station counts (not arclength; descriptive)
    alt[pop_name]["n_stations"] = stats_paired([r["c"]["n"] for r in rs], [r["f"]["n"] for r in rs])
    alt[pop_name]["n_certified_stations"] = stats_paired([r["c"]["ncert"] for r in rs], [r["f"]["ncert"] for r in rs])
adv["alternative_metrics"] = alt
say("alternative metrics (D = cont - fan; differing |D| > 0.01; scipy-default Wilcoxon):")
for pop_name in ("primary", "all"):
    for m, s in alt[pop_name].items():
        say(f"  [{pop_name}] {m:>20}: {fmt(s)}")

# certification by station index and by arclength bin, B margins
cert_tab = {}
for arm, key in (("fan", "f"), ("cont", "c")):
    by_idx = defaultdict(list)
    by_arc = defaultdict(list)
    margins = []
    for r in prim:
        cum = 0.0
        for i, (s, fl, bb) in enumerate(zip(r[key]["segs"], r[key]["flags"], r[key]["b"]), start=1):
            cum += s
            by_idx[i].append(fl)
            by_arc[min(int(cum / 0.05), 6)].append(fl)
            margins.append(bb - r[key]["thr"])
    cert_tab[arm] = {"by_index": {i: [round(float(np.mean(v)), 3), len(v)] for i, v in sorted(by_idx.items())},
                     "by_arclength_bin_0.05": {f"[{b * 0.05:.2f},{(b + 1) * 0.05:.2f})": [round(float(np.mean(v)), 3), len(v)]
                                               for b, v in sorted(by_arc.items())},
                     "B_minus_thr_median_mean": [float(np.median(margins)), float(np.mean(margins))]}
adv["certification"] = cert_tab
say("certified rate by station index (primary): fan " + str(cert_tab["fan"]["by_index"]) + " | cont " + str(cert_tab["cont"]["by_index"]))
say("certified rate by arclength bin (primary): fan " + str(cert_tab["fan"]["by_arclength_bin_0.05"]) + " | cont "
    + str(cert_tab["cont"]["by_arclength_bin_0.05"]))
say(f"cert B - threshold median/mean: fan {np.round(cert_tab['fan']['B_minus_thr_median_mean'], 4).tolist()} "
    f"cont {np.round(cert_tab['cont']['B_minus_thr_median_mean'], 4).tolist()}")

# fan localisation half-width vs the certification half-separation
hw = np.array([h for r in prim for h in r["f"]["halfw"]])
flw = np.array([fl for r in prim for fl in r["f"]["flags"]])
adv["fan_localisation"] = {"halfwidth_quantiles_25_50_75_90": [float(q) for q in np.quantile(hw, [.25, .5, .75, .9])],
                           "frac_halfwidth_gt_EPS/2": float(np.mean(hw > EPS_HALF)),
                           "cert_rate_halfwidth_le_EPS/2": float(flw[hw <= EPS_HALF].mean()),
                           "cert_rate_halfwidth_gt_EPS/2": float(flw[hw > EPS_HALF].mean()) if np.any(hw > EPS_HALF) else None,
                           "n_gt": int(np.sum(hw > EPS_HALF))}
cont_hw = [h for r in prim for h in r["c"]["halfw"]]
say(f"fan final-bracket half-width quantiles(25,50,75,90) {np.round(adv['fan_localisation']['halfwidth_quantiles_25_50_75_90'], 4).tolist()}; "
    f"> EPS/2 ({EPS_HALF:.4f}): {adv['fan_localisation']['frac_halfwidth_gt_EPS/2']:.2f} "
    f"(cert rate {adv['fan_localisation']['cert_rate_halfwidth_gt_EPS/2']}) vs <=: {adv['fan_localisation']['cert_rate_halfwidth_le_EPS/2']:.3f}; "
    f"cont |weights-center| max {max(cont_hw) if cont_hw else 0:.2e} (station = bracket centre, +-0.00625 by construction)")

# continuation corrector: evidence-pair separation, lateral offset of the chosen flip, multiple flips on a line
sep, lat, multi, soft_flags, adj_flags, lat_flags = [], [], 0, [], [], []
n_lines = 0
for r in prim:
    tr_ok = [tr for tr in r["c"]["trace"] if tr["chosen"] is not None]
    for tr, fl in zip(tr_ok, r["c"]["flags"]):
        k_lo, k_hi, i_p, j_p = tr["chosen"]
        sep.append(j_p - i_p)
        off = (tr["offs"][k_lo] + tr["offs"][k_hi]) / 2
        lat.append(abs(off))
        (soft_flags if j_p - i_p >= 2 else adj_flags).append(fl)
        lat_flags.append((abs(off), fl, abs(off) / max(tr["h"], 1e-9)))
    for tr in r["c"]["trace"]:
        n_lines += 1
        flips = sum(1 for a in tr["adj"] if a is not None and a > 0.35)
        if flips >= 2:
            multi += 1
lat = np.array(lat)
lf = np.array([x[1] for x in lat_flags])
ratio = np.array([x[2] for x in lat_flags])
adv["continuation_corrector"] = {
    "evidence_pair_separation_counts": dict(Counter(sep)),
    "cert_rate_adjacent_pair": float(np.mean(adj_flags)) if adj_flags else None, "n_adjacent": len(adj_flags),
    "cert_rate_soft_pair(sep>=2)": float(np.mean(soft_flags)) if soft_flags else None, "n_soft": len(soft_flags),
    "abs_lateral_offset_quantiles_50_75_90_max": [float(q) for q in np.quantile(lat, [.5, .75, .9, 1])],
    "frac_lateral_offset_ge_0.05": float(np.mean(lat >= 0.05 - 1e-9)),
    "cert_rate_lateral_offset_ge_0.05": float(lf[lat >= 0.05 - 1e-9].mean()) if np.any(lat >= 0.05 - 1e-9) else None,
    "cert_rate_lateral_offset_lt_0.05": float(lf[lat < 0.05 - 1e-9].mean()),
    "frac_offset_over_h_gt_1(turn>45deg)": float(np.mean(ratio > 1)),
    "cert_rate_offset_over_h_gt_1": float(lf[ratio > 1].mean()) if np.any(ratio > 1) else None,
    "lines_with_>=2_adjacent_flips": multi, "n_corrector_lines": n_lines,
}
say(f"continuation corrector (primary): evidence-pair separation {dict(sorted(Counter(sep).items()))}; cert rate adjacent "
    f"{adv['continuation_corrector']['cert_rate_adjacent_pair']:.3f} (n={len(adj_flags)}) vs soft (sep>=2) "
    f"{adv['continuation_corrector']['cert_rate_soft_pair(sep>=2)']} (n={len(soft_flags)}); |lateral offset of chosen flip| "
    f"quantiles(50,75,90,max) {np.round(adv['continuation_corrector']['abs_lateral_offset_quantiles_50_75_90_max'], 4).tolist()}; "
    f">=0.05: {adv['continuation_corrector']['frac_lateral_offset_ge_0.05']:.2f} (cert {adv['continuation_corrector']['cert_rate_lateral_offset_ge_0.05']}) "
    f"vs <0.05 cert {adv['continuation_corrector']['cert_rate_lateral_offset_lt_0.05']:.3f}; offset/h > 1: "
    f"{adv['continuation_corrector']['frac_offset_over_h_gt_1(turn>45deg)']:.2f} (cert {adv['continuation_corrector']['cert_rate_offset_over_h_gt_1']}); "
    f"lines with >=2 adjacent flips: {multi}/{n_lines}")

# fan: transversal jumps between consecutive stations
fj, ff = [], []
for r in prim:
    uvf = [np.zeros(2)]
    for s, a in zip(r["f"]["segs"], r["f"]["ang"]):
        pass
for r, u in [(r, u) for r, u in units if u["significant"]]:
    cs = V.centres(u["fan"])
    uv, _, _ = uv_of(u["fan"], cs)
    v = [0.0] + [q[1] for q in uv]
    fl = V.cert_flags(u["fan"])
    for i in range(1, len(v)):
        fj.append(abs(v[i] - v[i - 1]))
        ff.append(fl[i - 1])
fj, ff = np.array(fj), np.array(ff)
adv["fan_transversal_jumps"] = {"abs_dv_quantiles_50_75_90_max": [float(q) for q in np.quantile(fj, [.5, .75, .9, 1])],
                                "frac_dv_gt_0.05(turn>45deg)": float(np.mean(fj > 0.05)),
                                "cert_rate_dv_gt_0.05": float(ff[fj > 0.05].mean()) if np.any(fj > 0.05) else None,
                                "cert_rate_dv_le_0.05": float(ff[fj <= 0.05].mean())}
say(f"fan |dv| between consecutive stations quantiles(50,75,90,max) "
    f"{np.round(adv['fan_transversal_jumps']['abs_dv_quantiles_50_75_90_max'], 4).tolist()}; >0.05 (turn>45deg): "
    f"{adv['fan_transversal_jumps']['frac_dv_gt_0.05(turn>45deg)']:.2f} (cert {adv['fan_transversal_jumps']['cert_rate_dv_gt_0.05']}) "
    f"vs <=0.05 cert {adv['fan_transversal_jumps']['cert_rate_dv_le_0.05']:.3f}")

# per-run contributions and leave-one-run-out
labs = [r["_label"] for r in runs]
contrib = {}
totD = sum(r["c"]["A"] - r["f"]["A"] for r in prim)
for lab in labs:
    rs = [r for r in prim if r["run"] == lab]
    Dl = [r["c"]["A"] - r["f"]["A"] for r in rs]
    contrib[lab] = {"n": len(rs), "sum_D": float(np.sum(Dl)), "share_of_total_D": float(np.sum(Dl) / totD),
                    "mean_D": float(np.mean(Dl)), "wins": int(sum(d > DIFF for d in Dl)), "losses": int(sum(d < -DIFF for d in Dl)),
                    "zeros": int(sum(d == 0 for d in Dl))}
loro = {}
for lab in labs:
    rs = [r for r in prim if r["run"] != lab]
    s = stats_paired([r["c"]["A"] for r in rs], [r["f"]["A"] for r in rs])
    sc = stats_paired([r["c"]["A_cert"] for r in rs], [r["f"]["A_cert"] for r in rs])
    loro[f"without_{lab}"] = s | {"guard_ok": bool(sc["mean_cont"] >= sc["mean_fan"] - 0.01)}
adv["per_run_contribution_primary"] = contrib
adv["leave_one_run_out_primary"] = loro
say("per-run contribution to sum of D (primary): " + "; ".join(
    f"{k}: n={v['n']} sumD={v['sum_D']:+.3f} ({100 * v['share_of_total_D']:.0f}%) meanD={v['mean_D']:+.4f} +{v['wins']}/-{v['losses']} zeros={v['zeros']}"
    for k, v in contrib.items()))
for k, s in loro.items():
    say(f"  {k}: {fmt(s)} guard_ok={s['guard_ok']}")

# clustering: crossing-level (average the two directions) and exact sign test
cl = defaultdict(list)
for r in prim:
    cl[(r["run"], r["cid"])].append(r["c"]["A"] - r["f"]["A"])
Dx = np.array([np.mean(v) for v in cl.values()])
p_cross = float(wilcoxon(Dx).pvalue) if np.any(Dx != 0) else None
s_all = stats_paired([r["c"]["A"] for r in prim], [r["f"]["A"] for r in prim])
p_sign = float(binomtest(s_all["cont_longer"], s_all["cont_longer"] + s_all["fan_longer"]).pvalue)
Dp = np.array([r["c"]["A"] - r["f"]["A"] for r in prim])
p_pratt = float(wilcoxon(Dp, zero_method="pratt").pvalue)
rng = np.random.default_rng(0)
nz = Dp[Dp != 0]
from scipy.stats import rankdata
rk = rankdata(np.abs(nz))
obs = rk[nz > 0].sum()
sims = np.array([(rk * (rng.random(nz.size) < 0.5)).sum() for _ in range(200000)])
mu = rk.sum() / 2
p_perm = float(np.mean(np.abs(sims - mu) >= abs(obs - mu) - 1e-12))
adv["robustness_primary"] = {"crossing_level_n": int(Dx.size), "crossing_level_mean_D": float(Dx.mean()),
                             "crossing_level_wins_losses": [int(np.sum(Dx > DIFF)), int(np.sum(Dx < -DIFF))],
                             "crossing_level_wilcoxon_p": p_cross, "sign_test_p_differing": p_sign,
                             "wilcoxon_pratt_p": p_pratt, "signed_rank_permutation_p_200k(zeros dropped)": p_perm}
say(f"robustness (primary): crossing-level (n={Dx.size}, dirs averaged) meanD={Dx.mean():+.4f} "
    f"+{int(np.sum(Dx > DIFF))}/-{int(np.sum(Dx < -DIFF))} Wilcoxon p={p_cross:.4f}; sign test on differing p={p_sign:.4f}; "
    f"Wilcoxon pratt p={p_pratt:.4f}; signed-rank permutation (200k) p={p_perm:.4f}")

# A_cert reading sensitivity (cap on the certified sum only)
alt_c = stats_paired([V.metric_A_cert_alt(u["origin"], u["continuation"]) for r, u in units if u["significant"]],
                     [V.metric_A_cert_alt(u["origin"], u["fan"]) for r, u in units if u["significant"]])
adv["A_cert_alt_reading_primary"] = alt_c
say(f"A_cert alternative reading (cap on certified sum only), primary: {fmt(alt_c)}")

# where do the wins come from? end-class cross-tab of the differing units
xt = Counter()
for r in prim:
    D = r["c"]["A"] - r["f"]["A"]
    if abs(D) <= DIFF:
        continue
    def ec(nt):
        return ("budget" if not nt else "identity" if "changed identity" in nt else "ended" if "ended before" in nt
                else "edge" if ("edge" in nt or "face" in nt) else "other")
    xt[("cont" if D > 0 else "fan") + " longer | fan:" + ec(r["f"]["end"]) + " cont:" + ec(r["c"]["end"])] += 1
adv["differing_units_end_class_crosstab"] = dict(xt)
say("differing units by end class: " + str(dict(sorted(xt.items()))))

# unit table for the primary population
tab = [{"run": r["run"], "cid": r["cid"], "dir": r["dir"], "A_fan": round(r["f"]["A"], 4), "A_cont": round(r["c"]["A"], 4),
        "D": round(r["c"]["A"] - r["f"]["A"], 4), "Acert_fan": round(r["f"]["A_cert"], 4), "Acert_cont": round(r["c"]["A_cert"], 4),
        "disp_fan": round(r["f"]["disp"], 4), "disp_cont": round(r["c"]["disp"], 4), "n_fan": r["f"]["n"], "n_cont": r["c"]["n"],
        "cert_fan": r["f"]["ncert"], "cert_cont": r["c"]["ncert"], "end_fan": r["f"]["end"], "end_cont": r["c"]["end"]}
       for r in prim]
OUT["primary_unit_table"] = tab
OUT["adversarial"] = adv
json.dump(OUT, open(os.path.join(HERE, "checks_output.json"), "w"), indent=1, default=float)
with open(os.path.join(HERE, "checks_summary.txt"), "w") as fh:
    fh.write("\n".join(L) + "\n")
