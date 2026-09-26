# PREREG — continuation walk vs fan walk (Ridge Explorer cascade, k = 4)

*Frozen before any confirmatory walk was run; sha256 in `PREREG_walk_continuation.sha256`. Departures go to
`DEVIATIONS_walk_continuation.md`. Design source: `RESEARCH_ridge_following_k4.md` (fix 1 and its §4 gate).*

## Question
Under the relative side-signature rule the fan walk still loses most fronts: in the re-diagnosis
(`walk_diag_f91e00f7_relative.json`, 36 walks) 12 ended by drift, 7 at the edge of the simplex, 12 by identity
changes, 3 used the whole budget. Does a sequential **predictor–corrector (continuation) walk** follow a front
further than the fan, without buying the extra distance with points that are not boundaries?

## Arms (both: `sig_mode = relative`, `use_jvp = false`, `n_steps = 6`)
- **Fan** (`mode = fan`, the current default): stations on the straight line `x0 + j·0.05·t`, j = 1..6; transversal
  probe pairs along the bracket chord n (arm 0.024 + 0.02·j, capped at 0.09), one widened retry (0.12), contiguous
  prefix, two batched bisection rounds.
- **Continuation** (`mode = continuation`): in the same plane span(t, n) (same crossing, same seeded tangent t),
  predict along the current tangent (t, then the secant through the last two captured points); corrector line
  perpendicular to it with points every 0.025 over ±0.10 (first station) / ±0.05 (later), rendered in one round.
  Any pair on the line at most 4 spacings apart with `1 − cos > 0.35` that passes the relative signature is
  evidence of the front; each is localised to one spacing by labelling its interior points by the nearer end; the
  localised crossing nearest the prediction wins (narrower pair on ties); one bisection round. Step 0.05, ×1.5 after
  a success (capped at 0.05), halved after a miss; stop below 0.0125, at a face (margin 0.003), or when the captured
  arclength reaches the budget 6 × 0.05 = 0.30.
- Parameters frozen as in `backend/services/cascade.py` (sha256 below): `PC_DS = 0.025`, `PC_ARM_FIRST = 0.10`,
  `PC_ARM = 0.05`, `PC_H_MIN = 0.0125`, `PC_GROW = 1.5`, `PC_EDGE = 0.003`, `PC_SEP_MAX = 4`, `WALK_STEP = 0.05`.
- Fixed for both arms before the freeze: the tangent seed collision (a crossing on chord 0 got t = n and the walk
  errored — the "other" rows of the earlier diagnostics) now redraws t; all other walks keep their exact plane.

## Data
Fresh cascade runs, k = 4, `n_chords = 10`, `n_patches = 2`, `steps = 8`, `probe_steps = 4`, 512², guidance 4.0,
FLUX.2-klein as served by the Explorer (7 GPU workers). Prompt sets from `h08_vector_uses/1_highdim/sets.json`.
Runs are collected in this fixed order; C1–C3 always, then C4, C5 only while the pooled primary population (below)
has fewer than 30 units (a stopping rule on counts, never on outcomes):

| run | prompt set | seed | note |
|---|---|---|---|
| C1 | k4_T0 (airplane / giraffe / toilet-paper painting / sushi) | 7 | the design quad, fresh crossings |
| C2 | k4_T2 (skyscraper / jellyfish watercolour / knight sketch / insect macro) | 42 | new prompts |
| C3 | k4_T3 (insect macro / hurricane / old books / Einstein) | 42 | new prompts |
| C4 | k4_T1 (sushi / astronaut / haunted forest / skyscraper) | 42 | only if needed |
| C5 | k4_T4 (Einstein / ballerina / sixties poster / Greek temple) | 42 | only if needed |
| D1 | k4_T0 | 42 | descriptive only: the design run's crossings (f91e00f7) + the lattice check |

Development data (the corrector was changed after these; excluded): k4_T5 seed 42, runs 734e0500, 61782f33 and the
re-check run after the multi-scale corrector.

**Unit** = (crossing, direction); both arms walk from the same origin. **Primary population**: units whose origin
crossing is `significant` (B above its run's background p95), pooled over the confirmatory runs. Secondary: all
units. A unit with an errored walk in either arm is excluded and counted.

## Metrics
- **A (primary)**: captured arclength = `min(0.30, Σ‖c_i − c_{i−1}‖₂)` along the polyline origin → station centres
  (`steps[].center`, barycentric coordinates), identical code for both arms.
- **Certification** (after each walk, `POST …/certify`): each station is re-measured with the cascade's statistic
  B (mean 1 − cos across it at ±EPS/2 = ±0.0235, along the in-plane perpendicular of the walk's own polyline) on
  the held-out seeds seed + 997·j, j = 1, 2, 3 (every walk decision used j = 0 only). A station is **certified**
  iff B > the 95th percentile of its run's background pairs at the same three seeds.
- **A_cert**: the part of A made of polyline segments whose end station is certified (segments taken in order up to
  the 0.30 cap, the last one truncated to fit).
- Also recorded: stations, certified stations, images and wall time per walk, the end note.

## Decision rule (primary population)
D = A_cont − A_fan per unit; a unit **differs** iff |D| > 0.01.
- **SUPPORTED** iff all of: (i) continuation longer in ≥ 65 % of differing units; (ii) two-sided paired Wilcoxon
  signed-rank on D (scipy default, `zero_method = "wilcox"`) p < 0.05; (iii) mean D > 0; (iv) mean A_cont > mean
  A_fan in a majority of the confirmatory runs that contribute ≥ 2 primary units; (v) **validity guard**: mean
  A_cert,cont ≥ mean A_cert,fan − 0.01 (the extra distance is not bought by losing certified distance).
- **REFUTED** iff the fan is longer in ≥ 65 % of differing units, p < 0.05 and mean D < 0.
- **INCONCLUSIVE** otherwise, including fewer than 8 differing units, or (i)–(iv) holding with (v) failing
  (reported as "longer but not validated").

## Secondary / descriptive (no verdict)
The same statistics over all units; per-run tables; paired Wilcoxon on A_cert; certified-station rates per arm and
by station index; efficiency (images per unit of A and of A_cert); wall time; end-note classes per arm (budget
reached / edge / ended = drift / changed identity); D1: the same tables on the design run's crossings, plus the
h08 k = 4 lattice (`grid4_quad0.npz`, 455 cells, spacing 1/12, rendered at 10 steps vs the Explorer's 8, so only
approximate): the S percentile of the lattice cell nearest each station, per arm.

## Analysis and verification
`tests/walk_pc_analyze.py` (written after this freeze) computes everything from the raw files
`tests/walk_pc_raw_<run>.json` written by `tests/walk_pc_collect.py`. An independent verifier recomputes A, D, the
rule and the guard from the raw JSON with its own code before reading the analysis script.

## Frozen code
`backend/services/cascade.py` sha256: `ecd34395874371bcbed7ed7d1f595b12dcfa2ec66d9a1d82547da4ac533d4988` (frozen 2026-09-26T13:45:38Z; server pid 3307149 loaded this file)
