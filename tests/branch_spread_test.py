"""Offline checks of ANGLE SPREADING for recursive ("branching") child chords
(backend/services/cascade.py, _spread_dir and _near_dirs). No GPU: cascade.evaluate is replaced
by the band-field stub of tests/branch_test.py, so the real generational loop runs unchanged.

A child ray no longer takes a plain isotropic direction. It is the best of BRANCH_N_CAND
isotropic candidates by LINE angle arccos(|u.v|) -- chords are undirected, so parallel and
anti-parallel are equally bad -- against the chords that already pass NEAR its origin (within
BRANCH_NEAR_R, the parent first of all), and the rays of one fan are drawn greedily so each
spreads away from the siblings before it. Near-parallel rays would re-probe the sheet their
neighbour already resolved; a parallel chord FAR away is left alone, it samples other territory.

  (i)   PURE -- _spread_dir over 40 fixed seeds: against the parent alone it lands within 0.15
        of perpendicular (|cos|; the isotropic median is 0.50), the greedy sibling spreads away
        from parent AND sibling, k=3 falls back to the 45 deg bisector its 2-D tangent space
        allows, every direction is a unit sum-zero tangent vector, an empty neighbourhood returns
        the first candidate, and -v counts exactly like +v.
  (ii)  NEIGHBOURHOOD -- _near_dirs as a distance filter: the parent (distance 0) is in, a chord
        offset past BRANCH_NEAR_R is out, and it is the SEGMENT that counts, not its infinite
        line (a chord whose line passes through the origin but whose ends are far is out).
  (iii) LOOP -- a stubbed branch=3 depth=2 survey: every child's chords_meta carries
        min_angle_deg and n_near >= 1, both EXACTLY reproduced from the stored geometry (the
        neighbours at fan start plus the siblings drawn before it), the per-generation note
        quotes the median spread, and generation 0 is untouched (bit-identical probes).

Run: python tests/branch_spread_test.py
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from backend.services import cascade as cs            # noqa: E402
from branch_test import (_gen_probes, _replay_roots,   # noqa: E402  (same stub and band field)
                         _survey)

SEEDS = range(1000, 1040)        # 40 fixed seeds: the spread is random, its statistics are not

fails = []


def check(ok, label, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(label)


def _acos_d(a, b):
    """Line angle between two unit directions, in degrees."""
    return float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(a, b)))))))


def _trio(k, seed):
    """A parent direction and the two children a greedy fan would spread away from it."""
    rng = np.random.default_rng(seed)
    p = cs._iso_dir(k, rng)
    u1, a1 = cs._spread_dir([p], rng, k)
    u2, a2 = cs._spread_dir([p, u1], rng, k)
    return p, (u1, a1), (u2, a2)


def part1():
    print("\n(i) _spread_dir as a pure function")
    check(cs.BRANCH_NEAR_R == 0.15 and cs.BRANCH_N_CAND == 64,
          "the two constants of record are unchanged",
          f"near {cs.BRANCH_NEAR_R}, {cs.BRANCH_N_CAND} candidates")

    trios = [_trio(4, s) for s in SEEDS]
    cos1 = np.array([abs(float(np.dot(p, u1))) for p, (u1, _), _ in trios])
    ang1 = np.array([a for _, (_, a), _ in trios])
    check(cos1.max() <= 0.15,
          "against the parent alone the child is within |cos| 0.15 of perpendicular (k=4)",
          f"max |cos| {cos1.max():.3f}, median angle {np.median(ang1):.1f} deg")

    # the same draw without spreading: a plain isotropic direction sits at |cos| ~ 0.5
    rng = np.random.default_rng(7)
    iso = np.array([abs(float(np.dot(cs._iso_dir(4, rng), cs._iso_dir(4, rng))))
                    for _ in range(2000)])
    check(np.median(iso) > 0.4 > cos1.max(),
          "the spreading is not vacuous: isotropic directions sit near |cos| 0.5",
          f"isotropic median {np.median(iso):.2f} vs spread max {cos1.max():.3f}")

    cos2 = np.array([max(abs(float(np.dot(p, u2))), abs(float(np.dot(u1, u2))))
                     for p, (u1, _), (u2, _) in trios])
    ang2 = np.array([a for _, _, (_, a) in trios])
    # 64 candidates in the 3-D tangent space of k=4 meet the DOUBLE constraint on ~93% of draws
    # (the admissible solid angle is ~2t^2/pi per candidate), so the bound is quoted as a
    # proportion plus a hard outer bound -- an all-seeds 0.30 would be a flaky test, not a result
    check(np.median(cos2) <= 0.30 and np.mean(cos2 <= 0.30) >= 0.9 and cos2.max() <= 0.45,
          "the greedy sibling spreads away from parent AND sibling (k=4)",
          f"median |cos| {np.median(cos2):.3f}, {np.mean(cos2 <= 0.30):.0%} <= 0.30, "
          f"max {cos2.max():.3f}, median angle {np.median(ang2):.1f} deg")

    # k=3: the tangent space is 2-D, so once parent and first child span it the second child
    # CANNOT be perpendicular to both -- the best it can do is bisect them at 45 deg
    a2_k3 = np.array([a for _, _, (_, a) in (_trio(3, s) for s in SEEDS)])
    check(43.0 <= np.median(a2_k3) <= 46.0,
          "k=3 degrades to the 45 deg bisector its 2-D tangent space allows",
          f"median angle {np.median(a2_k3):.1f} deg, min {a2_k3.min():.1f}")

    # unit length, sum-zero (the tangent space of the simplex), and the recorded angle is the one
    # the direction actually achieves against what it was spread away from
    bad_u, bad_a = 0, 0
    for k in (3, 4, 6):
        for s in SEEDS:
            p, (u1, a1), (u2, a2) = _trio(k, s)
            for u in (u1, u2):
                if abs(np.linalg.norm(u) - 1) > 1e-12 or abs(u.sum()) > 1e-12:
                    bad_u += 1
            if abs(a1 - _acos_d(p, u1)) > 1e-9 or abs(
                    a2 - min(_acos_d(p, u2), _acos_d(u1, u2))) > 1e-9:
                bad_a += 1
    check(bad_u == 0, "every chosen direction is a unit sum-zero tangent vector")
    check(bad_a == 0, "the returned angle is the minimum line angle to the neighbours")

    # an empty neighbourhood: the first candidate, i.e. exactly the old _iso_dir draw
    rng_a, rng_b = np.random.default_rng(5), np.random.default_rng(5)
    u, a = cs._spread_dir([], rng_a, 4)
    check(np.array_equal(u, cs._iso_dir(4, rng_b)) and a == 90.0,
          "an empty neighbourhood returns the first candidate (unconstrained, 90 deg)")

    # undirected lines: an existing -v must steer the choice exactly like +v
    same = 0
    for s in SEEDS:
        v = cs._iso_dir(4, np.random.default_rng(s))
        up, ap = cs._spread_dir([v], np.random.default_rng(s + 1), 4)
        um, am = cs._spread_dir([-v], np.random.default_rng(s + 1), 4)
        same += int(np.array_equal(up, um) and ap == am)
    check(same == len(SEEDS), "an anti-parallel neighbour -v counts exactly like +v",
          f"{same}/{len(SEEDS)} seeds identical")


def part2():
    print("\n(ii) _near_dirs: which chords the origin can see")
    p = np.full(4, 0.25)
    u = np.array([1.0, -1.0, 0.0, 0.0]) / np.sqrt(2)      # sum-zero, perpendicular to v
    v = np.array([0.0, 0.0, 1.0, -1.0]) / np.sqrt(2)
    parent = (p - 0.3 * u, p + 0.3 * u)                   # through the origin: distance 0
    cases = [
        ("the parent through the origin", parent, True),
        ("a chord 0.14 away (inside 0.15)", (p + 0.14 * v - 0.3 * u, p + 0.14 * v + 0.3 * u),
         True),
        ("a chord 0.16 away (outside 0.15)", (p + 0.16 * v - 0.3 * u, p + 0.16 * v + 0.3 * u),
         False),
        ("a chord 0.40 away", (p + 0.4 * v - 0.3 * u, p + 0.4 * v + 0.3 * u), False),
        # its infinite LINE runs through the origin, but the segment stops 0.5 short
        ("a collinear segment 0.5 further along", (p + 0.5 * u, p + 0.9 * u), False),
    ]
    for label, seg, want in cases:
        got = len(cs._near_dirs(p, [seg])) == 1
        check(got == want, f"{label} is {'in' if want else 'out'}",
              f"distance {cs._seg_dist(p, *[np.asarray(x) for x in seg]):.3f}")

    geo = [parent] + [c[1] for c in cases[1:]]
    near = cs._near_dirs(p, geo)
    check(len(near) == 2 and all(abs(np.linalg.norm(d) - 1) < 1e-12 for d in near),
          "a mixed neighbourhood keeps only the two near chords, as unit directions",
          f"{len(near)} of {len(geo)} chords")
    check(abs(abs(float(np.dot(near[0], u))) - 1) < 1e-12,
          "the parent's direction comes back (up to sign: a chord is a line)")
    check(cs._near_dirs(p, []) == [] and cs._near_dirs(p, [(p, p)]) == [],
          "no chords, or a degenerate zero-length chord, give an empty neighbourhood")
    check(len(cs._near_dirs(p, [cases[3][1]], r=0.5)) == 1,
          "the radius is a parameter: at r=0.5 the 0.40 chord is a neighbour again")


def part3():
    print("\n(iii) the generational loop: recorded spread, notes, roots untouched")
    run, probes = _survey(k=4, n_chords=6, branch=3, depth=2, seed=11)
    gens = [m["gen"] for m in run.chords_meta]
    check(max(gens) == 2 and gens.count(1) > 0,
          "branch=3 depth=2 produced children to check",
          f"gen counts {[gens.count(g) for g in range(3)]}, {len(run.crossings)} crossings")

    check(all("min_angle_deg" not in m and "n_near" not in m
              for m in run.chords_meta if m["gen"] == 0),
          "root chords carry no spread fields (they are the fair isotropic survey)")
    kids = [(ci, m) for ci, m in enumerate(run.chords_meta) if m["gen"] > 0]
    check(all(m.get("min_angle_deg") is not None and m.get("n_near", 0) >= 1
              for _ci, m in kids),
          "every child carries min_angle_deg and n_near >= 1 (the parent always counts)",
          f"{len(kids)} children, n_near {min(m['n_near'] for _c, m in kids)}.."
          f"{max(m['n_near'] for _c, m in kids)}")

    def direction(ci):
        a = np.asarray(run.chords_geo[ci][0], dtype=float)
        b = np.asarray(run.chords_geo[ci][1], dtype=float)
        return (b - a) / np.linalg.norm(b - a)

    # the exact reconstruction: a fan's chords are contiguous, so the s-th ray of the fan at ci0
    # was spread away from the chords near the origin at fan start (geo[:ci0]) plus its s elder
    # siblings. Both recorded numbers must come out of the stored geometry alone.
    fans = defaultdict(list)
    for ci, m in kids:
        fans[(m["gen"], m["origin_cid"])].append(ci)
    bad_n, bad_ang, bad_parent = 0, 0, 0
    for cis in fans.values():
        ci0 = min(cis)
        base = cs._near_dirs(np.asarray(run.chords_geo[ci0][0], dtype=float),
                             run.chords_geo[:ci0])
        for s, ci in enumerate(sorted(cis)):
            near = base + [direction(c) for c in sorted(cis)[:s]]
            if run.chords_meta[ci]["n_near"] != len(near):
                bad_n += 1
                continue
            want = min(_acos_d(direction(ci), d) for d in near)
            if abs(run.chords_meta[ci]["min_angle_deg"] - want) > 1e-2:
                bad_ang += 1
            if _acos_d(direction(ci), direction(run.chords_meta[ci]["parent"])) \
                    < run.chords_meta[ci]["min_angle_deg"] - 1e-6:
                bad_parent += 1
    check(bad_n == 0, "n_near = the chords near the origin at fan start + the elder siblings",
          f"{len(fans)} fans")
    check(bad_ang == 0, "min_angle_deg is the angle the child achieves against that set")
    check(bad_parent == 0, "and the parent, being in that set, never sits below the recorded "
          "minimum angle")

    med = [np.median([m["min_angle_deg"] for ci, m in kids if m["gen"] == g]) for g in (1, 2)]
    check(all(f", median spread {m:.0f}°" in n for m, n in zip(med, run.notes[1:3])),
          "each generation's note quotes its median spread",
          " | ".join(run.notes[1:3]))
    check(all("median spread" not in n for n in run.notes if n.startswith("gen 0:")),
          "generation 0 has nothing to spread away from, so its note is unchanged")

    # the roots are the fair survey the certificate rests on: spreading must not touch them,
    # even though the children now draw 64 candidates from the same rng stream
    got = _gen_probes(run, probes, 0)
    _old, want = _replay_roots(4, 6, 11, cs.STRIDE)
    check(len(got) == len(want) and all(np.array_equal(a, b) for a, b in zip(got, want)),
          "generation 0 is still bit-identical to the pre-branching chord phase",
          f"{len(got)} root probes")
    base, pb = _survey(k=4, n_chords=6, branch=0, depth=0, seed=11)
    p0 = _gen_probes(base, pb, 0)
    nb = len(base.crossings)
    check(len(p0) == len(got) and all(np.array_equal(a, b) for a, b in zip(p0, got))
          and [x.cid for x in base.crossings] == [x.cid for x in run.crossings[:nb]],
          "and identical to the same survey with branching off (root crossings included)",
          f"{nb} root crossings of {len(run.crossings)}")


if __name__ == "__main__":
    part1()
    part2()
    part3()
    print(f"\n{'ALL CHECKS PASS' if not fails else 'FAILED: ' + ', '.join(fails)}")
    sys.exit(1 if fails else 0)
