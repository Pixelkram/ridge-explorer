# Why the fan walk loses the front at k ≥ 4, and how to fix it
*2026-09-24. Evidence from `tests/walk_ab_4985c1b7.json` (25 origin/direction pairs, k = 4) and the walk code
(`backend/services/cascade.py::run_walk`). Proposal, not implementation; each fix carries a test that decides it.
Update 2026-09-26: fix 1 is built and won its pre-registered test — see §6.*

## 1. What the walk does, and where it breaks

`run_walk` is a *one-shot* line sampler: from a crossing `x0` it lays `n_steps` stations on a straight line
`x0 + j·0.05·t` (t = a random direction in the tangent plane of the bracket chord), renders a transversal pair
`base ± arm·nrm` at every station **in one GPU round**, and keeps the contiguous prefix of stations whose pair
(i) straddles a boundary (`cosd > COS_T = 0.35`) and (ii) matches the origin's side signatures. The arm grows
`0.024 + 0.020·j`, capped at 0.09; one widen retry at 0.12.

The A/B measured the transversal direction (bracket chord vs exact JVP normal) and found no difference — because
**18 of 25 walks captured nothing at either setting**, and both settings totalled 5 stations. The direction is
not what fails. Three mechanisms, in the order the evidence supports them:

**(a) The predictor is straight; the front is not.** The code's own comment records "measured sheet drift ~1 cell
per cell travelled". If the front leaves the tangent line at ~45°, at station j it sits ~0.05·j off the line
transversally, while the arm reaches only 0.024 + 0.02·j: **the arm is shorter than the drift at every j ≥ 1**
(0.044 vs 0.05 at j = 1; 0.064 vs 0.10 at j = 2). A straight predictor with a linearly growing arm cannot follow a
front that curves at that rate. The widen retry (0.12) rescues j ≤ 2 at best.

**(b) Weak origins.** The run's crossings were mostly below the significance bar (`bg_p95`); walking from a
crossing that is barely a front cannot capture stations. The A/B did not condition on `significant`.

**(c) A single-seed straddle criterion at the finest scale.** `cosd(ea, eb) > 0.35` on one seed at ±0.024 is a
chord at the scale where h07d showed the map is roughest; the same-ridge signature adds a second threshold. Both
are exactly the kind of small-scale, single-seed reading the h07 chain found least reproducible.

The diagnostic that separates (a)/(b)/(c) is the walk's own note per failed station — `ridge ended before
station j` (no straddle in the arm: drift or weak front) vs `ridge changed identity` (straddles a different
ridge: junction or (c)) — split by origin significance.

**Diagnostic result (`tests/walk_diag_<run>.json`, fresh k = 4 run, 18 crossings × 2 directions, n_steps = 4):**

| failure class | all 36 | significant origins (24) | non-significant (12) |
|---|---|---|---|
| `changed_identity` (pair straddles, signature rejects) | **28** | 17 | 11 |
| `ended_no_straddle` (drift / weak front) | 3 | 3 | 0 |
| edge of the space | 3 | 2 | 1 |
| other | 2 | 2 | 0 |
| mean stations captured | 0.19 | 0.21 | 0.17 |

**So (c) dominates, not (a).** In 28 of 36 walks the first station's pair *does* straddle a boundary
(`cosd > 0.35`), and the walk still stops because `same_ridge` demands that **both** new side images lie within
cosine distance 0.35 of the origin's side images. That is an *absolute* similarity test, and the h07 chain says
it must fail along a front: what changes across a ridge is location-specific (h07b–d), so 0.05 further along the
same front the two sides are already different images. Significance of the origin makes no difference. The
arm arithmetic in (a) is still wrong, but it is not what stops the walk at station 1.

**The fix this points to is cheap: a *relative* signature.** Accept a station iff its pair straddles and each
new side is closer to the origin's *matching* side than to the *opposite* one (order-invariant, as now):
`cosd(ea′, ea₀) < cosd(ea′, eb₀)` and `cosd(eb′, eb₀) < cosd(eb′, ea₀)`, or the flipped assignment. This keeps
the walk from jumping to a ridge whose sides are swapped or unrelated, without requiring the sides to stay the
same images. Implemented as `sig_mode = "relative"` on the walk and decided by the A/B of §4:

**A/B result (`tests/walk_ab_<run>_relative.json`, fresh k = 4 run, 18 crossings × 2 directions, n_steps = 6,
rule pre-registered in `tests/walk_ab.py`): SUPPORTED.** 24 of 36 pairs differed and the relative walk won
**24/24** (paired Wilcoxon p = 1.5e-5); mean captured stations **0.19 → 1.81**; overhead +1 s. The relative
signature is now the walk's default (`absolute` remains available). Caveat: one prompt quad, one seed, 8
steps; the same harness should be re-run on a second quad before this is quoted as general.

With the acceptance rule fixed, the walk still averages under 2 of 6 stations. **Re-diagnosis under
`sig_mode = relative` (`tests/walk_diag_<run>_relative.json`, same run, n_steps = 6, 36 walks):**

| failure class | count |
|---|---|
| `captured_all` (6/6 stations) | 3 |
| `ended_no_straddle` — front left the arm: **drift, (a)** | 12 |
| `changed_identity` — straddles, sides swapped/unrelated: junction or criterion | 12 |
| edge of the space — straight line leaves the simplex | 7 |
| other | 2 |
| stations captured: distribution over 0…6 | 10 · 10 · 5 · 6 · 1 · 1 · 3 |

The remaining loss splits evenly between **drift** and **identity changes**, with a third of the drift cases
being the straight predictor walking out of the simplex. That is exactly the case for fix **1** (secant
predictor–corrector with adaptive step: follows curvature and turns before a face) and, for the identity
changes, for reading the JVP's participation ratio at each captured station (a rising PR flags an approaching
junction before the walk crosses it). Fix 5 alone (a wider arm) would recover part of the drift cases at the
cost of more false straddles; the diagnostic makes 1 the better next step.

## 2. What the JVP contributes, and what it cannot

The exact normal is seed-invariant (h07e: 0.93; whitened 0.91) and the local spectrum reads codimension (h08-1).
It is a good *direction* at a point. It says nothing about the front 0.05 away — and (a) is a statement about
the front 0.05 away. So the JVP belongs in the **corrector** (which way to bisect) and in **re-estimating the
tangent after each captured station**, not as a one-shot replacement for a random chord. That is what the A/B
tested, and it was the wrong place.

## 3. Fixes, ranked by evidence and cost

| # | Fix | What it changes | Cost per station | Evidence it should help |
|---|---|---|---|---|
| **1** | **Predictor–corrector continuation (sequential, secant tangent, adaptive step)** | Step j uses the *secant* through the last two captured crossings as tangent (no JVP needed); corrector = bisection along the current normal; step halves on failure, ×1.5 on success (Allgower–Georg step control). Batched variant: at each step render a small fan (3 step lengths × 2 tangent directions) in one round and keep the best straddle. | 1 GPU round per step (vs 3–4 per walk now); ~6–10 images | Directly addresses (a): a secant follows curvature at first order; adaptive steps shrink where the sheet turns. Standard pseudo-arclength continuation ([Allgower & Georg](https://link.springer.com/book/10.1007/978-3-642-61257-2), [Acta Numerica 1993](https://assets.cambridge.org/97805214/43562/excerpt/9780521443562_excerpt.pdf)). |
| **2** | **HopSkipJump-style boundary walk** | Alternate: (i) tangential step, (ii) *binary search back onto the boundary along an estimated normal*, (iii) re-estimate the normal at the new boundary point. The normal estimate can be the JVP (exact, 20–35 s) or the cheap Monte-Carlo estimate HSJA uses (many 1-step latent probes). | 1 JVP or ~16 fast-scan probes + a bisection round | This is the decision-boundary analogue of the walk and is query-efficient by design ([Chen, Jordan & Wainwright 2020](https://arxiv.org/abs/1904.02144)); the step-size rule (geometric progression until the boundary is crossed) is exactly what (a) needs. |
| **3** | **Local level-set estimation in the tangent plane** | Around a crossing, fit a GP to the straddle field (cheap 1-step latent proxy for fidelity 1, full images for fidelity 2) over a (k − 2)-dim disc in the tangent plane, sample by the straddle rule `1.96σ − |μ − h|`, and read the front as the GP's level set. `mf_gp.MFRidgeDetector` already implements the multi-fidelity straddle for the 3-prompt simplex; this restricts it to a local chart. | ~20–40 cheap probes per chart | The tool's own straddle machinery, principled ([Bryan et al. 2005](https://www.researchgate.net/publication/262273291_Active_learning_for_level_set_estimation), [Gotovos et al. 2013](https://www.ijcai.org/Proceedings/13/Papers/202.pdf)); replaces "follow a line" by "map the sheet locally", which is what a (k − 2)-dimensional front actually needs. |
| **4** | **Condition on origin quality; seed-average the straddle** | Walk only from `significant` crossings; evaluate the straddle pair on 2–3 seeds and require the *mean* contrast to clear `COS_T`. | ×2–3 images | Addresses (b) and (c); cheap to test; the h07c result (seed averaging is the honest lever) predicts a gain. |
| **5** | **Arm schedule matched to measured drift** | Set the arm at station j to `c·0.05·j` with c ≈ 1.2 from the drift measurement, capped by the simplex; keep the one-shot batch. | none | Minimal patch that fixes the arithmetic in (a) without restructuring; predicted to help only for the first 2–3 stations. |

**Recommendation.** Do 4 and 5 first (an afternoon; they are the controls every other fix needs), then 1 as the
structural change — it reuses the existing bisection and thumbnail code, costs one GPU round per step, and needs
no JVP. Reserve 2 for the case where 1 loses the front at junctions (the JVP's participation ratio flags those
before the walk arrives), and 3 for a later "map this sheet" feature rather than a walk.

## 4. Test design (pre-register before running)

- **Diagnostic** (`tests/walk_diag.py`): walks from every crossing, both directions, n_steps = 4; classify each
  walk by its first failure note; split by `significant`. Decides the (a)/(b)/(c) weights.
- **Fix gate:** for each fix, the A/B of `tests/walk_ab.py` with the fix as arm B against the current walk, from
  *significant* origins only, n_steps = 6. Metric: contiguous captured stations; rule: B wins ≥ 65 % of
  differing pairs and paired Wilcoxon p < 0.05; also report images per captured station (efficiency), since a
  fix that captures more by spending 5× more is not a win.
- **Ground truth at k = 4:** `outputs/h08_vector_uses/1_highdim/grid4_quad0.npz` (455-cell lattice) gives an
  independent "is this station on a front" reference for one prompt quad — use it to check that captured
  stations really sit on high-S cells, not just on any straddle.

## 5. What not to do
Do not tune `COS_T` or the signature threshold by hand until the diagnostic says (c) matters; do not claim the
JVP improves ridge following (the A/B says it does not, in the walk as it stands); do not ship a standalone
tracer at k ≥ 4 — a front there is a surface, and 3 is the honest version of "trace it".

*Sources: HopSkipJumpAttack — https://arxiv.org/abs/1904.02144 ; Allgower & Georg, Numerical Continuation
Methods — https://link.springer.com/book/10.1007/978-3-642-61257-2 ; Bryan et al. 2005 / Gotovos et al. 2013,
active learning for level set estimation — https://www.ijcai.org/Proceedings/13/Papers/202.pdf .*

## 6. Fix 1 built and tested (2026-09-26)

**Implementation** (`run_walk_continuation`, `mode = "continuation"`, now the default). Sequential
predictor–corrector continuation in the fan's own plane span(t, n) — same crossing, same seeded tangent — so both
modes trace the same slice of the front. Predictor: t for the first station, then the secant through the last two
captured points. Corrector: points every 0.025 across the prediction, ±0.10 for the first station (t can miss the
front's direction by ~60°: t is only perpendicular to the chord, not the front — itself a large share of what (a)
called drift), ±0.05 afterwards, one GPU round; one bisection round. Step 0.05, ×1.5 after success, halved after a
miss, stop below 0.0125; budget = the fan's reach (n_steps × 0.05 of arclength).

**One change on development data** (k4_T5, excluded from the test): with straddles only between adjacent probes,
4/22 walks failed at station 1 although the line's ends differed strongly (1 − cos 0.6–0.95) — no adjacent pair
cleared 0.35 (0.339, 0.343, 0.344 …): real fronts spread their change over several probes. The corrector now
accepts any pair up to 4 probes (0.10) apart as evidence and localises the crossing between adjacent probes by
labelling the points in between by the nearer end (`_pc_bracket`); station-1 failures 4/22 → 0/22. The offline
test `tests/test_walk_continuation_synthetic.py` covers sharp and soft synthetic fronts.

**Also found:** the walk from crossing 0 errored whenever that crossing lay on chord 0 (the tangent seed
`default_rng(seed + 7919·cid)` equals the cascade's chord-direction stream for cid 0) — the unexplained "other"
rows of §1's diagnostics. Fixed by a redraw; other walks unchanged.

**Pre-registered A/B** (`tests/PREREG_walk_continuation.md`, frozen with the code hash; deviations in
`tests/DEVIATIONS_walk_continuation.md`; analysis `tests/walk_pc_analyze.py`; independent re-computation in
`tests/walk_pc_verify/`, 142 numbers matched). Three fresh k = 4 runs (k4_T0 seed 7, k4_T2, k4_T3), primary
population = walks from significant crossings (46), both modes from the same origin, n_steps = 6.

| | fan | continuation |
|---|---|---|
| mean captured arclength A (cap 0.30) | 0.156 | **0.197** |
| walks further (of 27 that differ) | 6 | **21** (p = 0.008; clustered by crossing 0.027) |
| arclength on stations certified on 3 held-out seeds | 0.112 | **0.150** (p = 0.01) |
| certified-station rate | 0.71 | 0.76 |
| end: drift ("ended before") / identity / edge / full budget | 10 / 15 / 12 / 9 | **1** / 11 / 17 / **17** |
| images per walk · per unit A · median wall | 18.3 · 117 · 14 s | 33.5 · 170 · 31 s |

**Verdict: SUPPORTED** (all five registered conditions). Robustness (verifier): zig-zag inflates the *fan's*
arclength, not the continuation's — displacement, smoothed-path and capped-segment metrics all widen the lead
(e.g. displacement 24/5, p = 0.0013). Per prompt set: C1 8–0, C3 8–1, **C2 5–5 (no effect)**; leaving out C1 or C3
loses significance, so "generalises across prompt sets" is not established. D1 (descriptive, the design run's own
crossings): 12–5, A 0.115 → 0.181; stations of both modes sit on high-S cells of the h08 lattice (mean S percentile
0.80 fan, 0.87 continuation; approximate, 10 vs 8 steps).

**What is left.** (1) *Identity*: 11/46 walks still end at a probable junction, and 18 % of continuation stations
were picked on corrector lines that crossed more than one boundary — certification shows a station is on *a*
boundary, not on the origin's one. The next lever is the one §3 reserved: read the JVP participation ratio at
captured stations as a junction early-warning, and prefer the crossing whose sides best continue the previous
station's (chained signature) when a line shows several. (2) *Edges*: 17/46 walks now end at a face of the
simplex — mostly fronts that genuinely run into it; walking along the face is a separate feature. (3) *Cost*:
~1.45× the images per unit distance; a batched fan of step lengths per round (§3, fix 1 variant) would trade images
for latency.


## 7. Same-ridge rule, trace phase, coverage — development findings (2026-09-26, NOT confirmatory)

**Ground truth.** At k = 3 the simplex is 2-D, so a dense lattice (N = 80, 3,321 images, rendered through the run's own
path via `POST /api/cascade/{run}/render`) shows every front (`tests/walk_gt_lib.py`): boundary points (1 − cos > 0.35
across 2 lattice steps), regions (connected non-boundary components), and a point's *ridge* = the pair of regions on
rings around it (radii 0.03–0.08). On the development field (k3_T0, seed 42, full-fidelity detection) 56/57 crossings
sit on ground-truth boundary points; 39/57 separate two regions (the rest are cracks inside one region — not
judgeable by this definition). Crossing weights per ridge come from 20,000 virtual cascade chords.

**How often does the current walk switch ridges?** Continuation walk, relative rule: 25 of 78 walks from judgeable
crossings ended up on a different region pair than their origin (dev c1).

**Continuity rule** (`sig_mode = "continuity"`, `PC_CONT_GAMMA`): a new station's largest side move, as a share of
the previous station's contrast (`steps[].cont`), must stay below γ. It separates switching steps from continuing
steps with AUC 0.87 (296 steps, 16 switches). Live dev check (fresh chords, 28 paired walks, γ = 0.5): switches
9 → 4, but distance before any switch 0.166 → 0.128 (walks stop more often at "junctions"). Offline replay (dev c1):
at every γ the rule beats simply stopping every walk sooner — e.g. γ = 0.7: 5 switches at 0.125 vs keep-3-stations:
10 switches at 0.132. **It is smart stopping, i.e. a strictness dial, not a free win**: a rule that only stops walks
can never increase the distance before the first switch. Shipped as an option ("strict"); the trace phase uses it.

**Coverage.** (i) With *perfect* ridge identity the Good-Turing unexplored share is well calibrated on this field
(simulated surveys: mean abs error 0.11 at 10 chords, 0.06 at 16, 0.035 at 24). (ii) The signature grouping finds
only about half of the true same-ridge crossing pairs (recall 0.48–0.50, precision 0.92–1.00), so it over-splits.
(iii) Joining crossings by walks is dangerous: individual walk links are ~80 % right, but union-find chains the wrong
ones — with distance-only links from 6-station walks every judgeable crossing merged into one group (precision 0.18)
and the estimate fell to 0 (truth 0.155). The implemented trace phase (strict walks + side check) made 14 links on the
second dev run, none of them wrong where judgeable, but it did not change the grouping of any judgeable crossing, so
the estimate did not move there. **Verdict so far: the trace phase draws useful ridge maps, but it does not (yet)
improve the coverage estimate; the traced number is labelled experimental in the UI. The better lever for a
reliable estimate is more chords (default 24).** A confirmatory test on fresh prompt sets has not been run.
