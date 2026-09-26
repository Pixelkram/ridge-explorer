"""DEV: how noisy is the Good-Turing unexplored share with PERFECT ridge identity? Simulate cascade surveys of m chords
on the dense ground truth (virtual chords, species = region pairs; off-front crossings dropped) and compare the
estimate (share of species crossed exactly once) with the true unseen share (chord-crossing weight of species the
survey never crossed)."""
import sys
from collections import Counter
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from walk_gt_lib import Lattice, STRIDE, COS_T

L = Lattice(sys.argv[1])
rng = np.random.default_rng(0)
chords = []                      # species crossed by each virtual chord (region pairs only)
while len(chords) < 6000:
    w0 = rng.dirichlet(np.ones(3)); u = rng.standard_normal(3); u -= u.mean(); u /= np.linalg.norm(u)
    tpos = min(w0[i] / -u[i] for i in range(3) if u[i] < 0); tneg = max(-w0[i] / u[i] for i in range(3) if u[i] > 0)
    if tpos - tneg < 6 * STRIDE:
        continue
    ts = tneg + STRIDE * np.arange(int((tpos - tneg) / STRIDE) + 1)
    es = [L.emb_at(np.clip(w0 + t * u, 0, None)) for t in ts]
    sp = []
    for a in range(len(ts) - 1):
        if es[a] is not None and es[a + 1] is not None and 1 - es[a] @ es[a + 1] > COS_T:
            s = L.species(np.clip(w0 + (ts[a] + ts[a + 1]) / 2 * u, 0, None))
            if s is not None:
                sp.append(s)
    chords.append(sp)
W = Counter(s for c in chords for s in c); tot = sum(W.values())
print(f"{len(W)} species; crossing weights {[round(v / tot, 3) for _, v in W.most_common()]}")
for m in (10, 16, 24, 40):
    est, tru, nn = [], [], []
    for _ in range(3000):
        pick = [chords[i] for i in rng.integers(len(chords), size=m)]
        c = Counter(s for p in pick for s in p); n = sum(c.values())
        if n < 2:
            continue
        est.append(sum(1 for v in c.values() if v == 1) / n)
        tru.append(sum(v for s, v in W.items() if s not in c) / tot); nn.append(n)
    est, tru = np.array(est), np.array(tru)
    print(f"m={m:2d} chords (~{np.mean(nn):.0f} judged crossings): true unseen {tru.mean():.3f} | estimate {est.mean():.3f} "
          f"| error mean {np.mean(est - tru):+.3f}, abs {np.mean(np.abs(est - tru)):.3f}, 90% range [{np.quantile(est - tru, .05):+.2f}, {np.quantile(est - tru, .95):+.2f}]")
