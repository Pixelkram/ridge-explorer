# Experiment Log: Phase Boundaries in Text-to-Image Embedding Spaces

## Project Goal

Two potential papers from this work:

1. **Paper 1 (Computational)**: "Phase Boundary Structure in Text-to-Image Embedding Spaces" — characterize ridges in text-embedding interpolation, validate across metrics and architectures.
2. **Paper 2 (Perceptual)**: "Do Phase Boundaries Predict Visual Interestingness?" — human evaluation of whether ridge images are more interesting.

Paper 1 is largely ready. Paper 2 depends on collecting human annotations.

**Core repo**: https://github.com/Pixelkram/ridge_evaluator
**Hardware**: 6× RTX 4090 (24GB each)
**Working directory**: `/home/student/ai/search_problem/`

---

## Critical Findings (Verified)

### 1. Ridges Exist in Text-Embedding Interpolation Space
- Barycentric interpolation between 3 text prompts on a 50×50 simplex grid produces clear ridge structures (regions of high DINOv2 neighbor distance)
- Validated on 25 hand-picked prompt triplets across 7 groups (Exp 22)
- Ridges appear in ALL triplets, including mundane ones (parking lot/filing cabinet/concrete wall)

### 2. Three Independent Metrics Detect the Same Ridges (Exp 21)
- DINOv2 vs CLIP: ρ=0.94
- DINOv2 vs Pixel MSE: ρ=0.65
- CLIP vs Pixel MSE: ρ=0.75
- **Conclusion**: Ridges are real structural features, not DINOv2 artifacts

### 3. Cross-Architecture Analysis (Exp 29-30, n=25 triplets, VERIFIED)

| Pair | Mean ρ | 95% CI | p (>0) | Cohen's d |
|------|--------|--------|--------|-----------|
| **Flux↔SD3.5** | **+0.491** | [+0.42, +0.56] | <10⁻⁶ | 2.67 |
| **Flux↔DS8** | **+0.361** | [+0.26, +0.46] | <10⁻⁵ | 1.41 |
| **DS8↔SD3.5** | **+0.255** | [+0.16, +0.35] | <10⁻⁴ | 1.01 |
| **Flux↔PixArt** | **-0.214** | [-0.33, -0.10] | Negative | -0.72 |
| **DS8↔PixArt** | **-0.164** | [-0.24, -0.09] | Negative | -0.82 |
| **SD3.5↔PixArt** | **-0.162** | [-0.25, -0.08] | Negative | -0.73 |

**Key finding**: CLIP-containing models (Flux, DS8, SD3.5) produce positively correlated ridge maps. PixArt (T5-XXL only) is **anti-correlated** — where CLIP-based models see ridges, PixArt sees valleys. All CIs exclude zero.

**Important nuance**: PixArt shares T5-XXL with Flux and SD3.5 but still anti-correlates. This means **diffusion architecture matters, not just text encoder**. The original claim "text encoder family determines boundaries" is **falsified** by PixArt. The correct claim: "Models sharing CLIP encoder components produce correlated ridge maps. T5-only models (PixArt) produce an inverted sensitivity landscape."

**Cross-architecture agreement by prompt group**:
- Groups A+B (diverse): highest agreement (ρ ≈ 0.4-0.6)
- Group D (close semantics, e.g. 3 cat breeds): lowest (ρ ≈ 0.1-0.4)
- Group E (mundane): low for DS8 pairs but moderate for Flux↔SD3.5

### 4. Seed Stability (Exp 28)
- Mean seed-to-seed Spearman ρ = 0.64 (range: 0.27-0.87 across 25 triplets)
- Ridges are **moderately stable** across seeds — broad structure is reproducible but fine-grained ranking shifts
- Some triplets much more stable (T01, T02: ρ≈0.85) than others (T00, T07: ρ≈0.30)
- **Implication**: Averaging across 3 seeds (as we do in Exp 22) is the right approach
- **Important context**: Lobashev et al. (ICML 2025) already showed noise-dependent boundary manifestation in their Hessian geometry work. Our finding in text-embedding space is consistent but not novel per se.

### 5. Step-Count Stability (Exp 23)
- 4-step vs 10-step DINOv2 sensitivity: ρ=0.76
- Top-10% ridge Jaccard overlap: 0.43
- 10-step produces wider dynamic range (0.001-0.878 vs 0.007-0.677)
- A/B human test (Exp 25): 54% vs 53% boundary preference, p=0.93 — no difference

### 6. Low-Level Image Metrics (Exp 27)
Boundary vs non-boundary images in the eval dataset:

| Metric | Boundary | Non-boundary | Diff | p |
|--------|----------|-------------|------|---|
| JPEG size | 54,592 | 53,520 | +2.0% | 0.12 |
| Edge density | 0.10 | 0.10 | +0.6% | 0.62 |
| **Colorfulness** | **34.21** | **40.49** | **-15.5%** | **<0.001** |
| Brightness | 119.96 | 122.38 | -2.0% | 0.02 |
| Entropy | 6.68 | 6.74 | -0.9% | 0.008 |

Boundary images are **less colorful** — favorable for the interestingness hypothesis (can't be explained by "flashier" images).

### 7. Noise-Space Lipschitz Does NOT Predict Text-Space Ridges (Exp 14-15)
- Lipschitz sensitivity in noise space: ρ=0.13-0.44 with DINOv2 text-space ridges (inconsistent)
- Lipschitz in noise space predicts interestingness at chance (52%)
- **This is a genuine negative result** — text-embedding ridges and noise-space sensitivity measure different phenomena

### 8. Human Evaluation Status
- **Pilot (Exp 12-13)**: 79% boundary preference, p<10⁻⁸ — BUT only 1-2 annotators (likely the authors). Invalid.
- **Confound control (Exp 19)**: Boundary vs random-unusual = 52%, p=0.80 (CHANCE). Suggests pilot effect was novelty, not phase boundaries. BUT: also 1-2 annotators, small sample.
- **Definitive dataset (Exp 26)**: 1,750 pairs ready. NO independent annotations collected yet.
- **The critical question**: Does ridge_vs_random_unusual show significant preference with 20+ independent annotators after covariate control?

---

## Relationship to Prior Work

### Lobashev et al. (ICML 2025) — "Hessian Geometry of Diffusion Models"
- They characterize phase boundaries via Jacobian/Hessian of the noise-to-image map
- Found fractal boundary structure in noise space
- Used Dreamshaper8 (same model family we started with)
- **Our work extends this to text-embedding interpolation space** — different input space, simpler detection method (DINOv2 neighbor distance vs Hessian computation)
- The seed-dependent boundary manifestation we observe (ρ=0.64) is consistent with their findings
- **Novelty concern**: The core phenomenon (generative models have phase boundaries) is theirs. Our contribution is: (a) text-embedding space, (b) cross-architecture comparison, (c) simpler detection, (d) human perception (if it works)

### Schmidhuber (2009) — Interestingness as Compression Progress
- Theoretical framework: interesting stimuli are those that yield learning progress for the observer
- Phase boundaries (high sensitivity = rapid change in output) could map to compression progress
- Our work is the first empirical test of this in generated images (if human eval succeeds)

### Berlyne (1971) — Experimental Aesthetics
- "Interesting" and "pleasing" are distinct hedonic responses
- Novelty, complexity, and incongruity drive interest
- Our random-unusual control tests whether boundary images are interesting beyond novelty

---

## Key Design Decisions and Rationale

### Why No Definition of "Interesting" in Annotations
Literature review (Berlyne 1971, Dhar et al. 2011 CVPR, Orne 1962) strongly supports:
- 2AFC with undefined "interesting" captures the ecological/folk concept
- Definitions create demand characteristics (annotators match a rubric, not their intuition)
- Anchoring effects from examples bias subsequent judgments
- Our current instruction: "Choose the image you find more interesting. There are no right or wrong answers — just go with your gut feeling."

### Why Linear Interpolation (Not On-Manifold)
- `emb = (1-α-β)*A + α*B + β*C` goes **off-manifold** — no real prompt produces these embeddings
- Ridges may mark where interpolation leaves the manifold of "sensible" embeddings
- This is a **weaker claim** than "the model has intrinsic phase boundaries"
- Better framing: "Linear interpolation in text-embedding space produces regions of high output sensitivity" — relevant for anyone doing prompt interpolation, embedding arithmetic, or latent space exploration

### Why Mixed 4-Step + 10-Step Dataset
- Serves as within-study replication
- Step count as recorded covariate
- A/B test showed no effect difference (p=0.93)
- More robust than single step count

---

## Experiment Inventory

| Exp | Description | Status | Key Result |
|-----|-------------|--------|------------|
| 01-09 | Fisher/Lipschitz/Jacobian in noise space | Done | Noise-space sensitivity doesn't predict interestingness |
| 10-13 | Ridge Explorer on DS8, 50×50 grids | Done | Pilot: 79% (invalid — author annotations) |
| 14-15 | Directional Lipschitz along interpolation | Done | ρ=0.13-0.44 with DINOv2 (inconsistent) |
| 16-18 | Cross-arch on Flux/SD3.5/PixArt/Z-Image (3 triplets) | Done | Preliminary, superseded by Exp 29-30 |
| 19 | Novelty confound control | Done | Boundary vs random-unusual = 52% (chance) — BUT small sample |
| 20 | Expanded confound (multi-model) | Done | Superseded by Exp 22 |
| 21 | Ridge method comparison (DINOv2/CLIP/pixel) | Done | ρ=0.65-0.94 convergence |
| 22 | Definitive 10-step Flux (25 triplets, 3 seeds) | Done | 95,625 images, sensitivity maps, pools |
| 23 | 4-step vs 10-step sensitivity comparison | Done | ρ=0.76, Jaccard=0.43 |
| 25 | 4-step vs 10-step A/B human test | Done | 54% vs 53%, p=0.93 (no difference) |
| 26 | Mixed eval dataset (4+10 step) | Done | 1,750 pairs, 7 comparisons, covariates |
| 27 | Low-level image metrics | Done | Boundary less colorful (-15.5%), similar complexity |
| 28 | Seed stability analysis | Done | Mean ρ=0.64 across seeds |
| 29 | Cross-arch grids: DS8+SD3.5+PixArt (25 triplets) | Done | All 4 models × 25 triplets generated |
| 30 | Cross-arch correlation analysis (4 models) | Done | See table above — PixArt anti-correlates |
| 31 | Random triplet validation (25 from 160-prompt bank) | **Running (GPUs 3-5)** | ~4h remaining, resumable |

---

## Currently Running

### Experiment 31: Random Triplet Flux Grids
- **Purpose**: Show ridges exist in randomly-sampled triplets, not just hand-picked ones
- **Running on**: GPUs 3, 4, 5 (GPUs 0-2 free for other work)
- **Resume command**: `python -m experiments.31_random_triplets --phase generate`
- Has skip-if-done logic — completed triplets are preserved on restart
- After generation: `python -m experiments.31_random_triplets --phase analyze`

### Annotator
- **Dataset**: `outputs/26_mixed_eval/` (1,750 pairs, mixed 4+10 step)
- **Start command**:
```bash
nohup python -u -m evaluation.annotate \
  --eval-dataset outputs/26_mixed_eval/pairs.json \
  --images-dir outputs/26_mixed_eval/images \
  --responses-dir outputs/26_mixed_eval/responses \
  --port 7860 --share > /tmp/annotator_mixed.log 2>&1 &
```
- **Dark mode UI**, 1.25s delay, skip button, gender defaults to "Prefer not to say"

---

## What's Needed for Paper 1 (Computational)

### Ready Now
- [x] Ridges exist (25 hand-picked triplets, Flux 10-step)
- [x] Three-metric convergence (DINOv2, CLIP, pixel MSE)
- [x] Cross-architecture analysis (4 models, 25 triplets, proper statistics)
- [x] Seed stability (mean ρ=0.64)
- [x] Step-count stability (ρ=0.76)
- [x] Noise-space Lipschitz negative result
- [x] Low-level image metrics
- [x] Curvature-based ridge vs slope classification

### Still Needed
- [ ] **Random triplet validation** (Exp 31, running) — shows ridges aren't cherry-picked
- [ ] Paper writing (methods, results, figures)
- [ ] Decide on Z-Image inclusion (lower resolution confound, only 3 triplets — probably drop)

### Nice to Have
- [ ] Permutation test (shuffle grid, show real ridges significantly different)
- [ ] Small-scale human eval pilot (5-10 Prolific annotators, ~$100) as preliminary evidence
- [ ] Ridge topology characterization (count, length, branching)

---

## What's Needed for Paper 2 (Perceptual)

- [ ] **Recruit 20-30 independent annotators** (Prolific/MTurk, no hypothesis knowledge)
- [ ] **Collect annotations** on the 1,750-pair mixed dataset
- [ ] **Run analysis**: `python -m experiments.22_analyze --eval-dir outputs/26_mixed_eval`
- [ ] **Critical test**: ridge_vs_random_unusual significant after covariate control?
- [ ] If positive: write perceptual paper
- [ ] If null: honest negative result, fold into Paper 1 as "preliminary" or publish separately

---

## Publication Strategy

### If human eval is positive (ridge > random-unusual after covariates)
- **One strong paper** combining computational + perceptual findings
- Target: NeurIPS/ICML main venue
- "We characterize phase boundaries in T2I models and show they predict human interestingness"

### If human eval is null
- **Paper 1**: Computational paper (phase boundary geometry + cross-architecture)
- **Paper 2**: Negative result / methodology paper (if desired)
- Target for Paper 1: NeurIPS workshop, TMLR, or Entropy journal

### Key concern for any venue
- Lobashev et al. (ICML 2025) already showed phase boundaries in noise space
- Our novelty: text-embedding space, cross-architecture comparison, simpler detection method
- Must cite and clearly differentiate

---

## File Map

```
search_problem/
├── config.py                              # Model config (Dreamshaper8 base)
├── evaluation/
│   ├── annotate.py                        # Gradio 2AFC annotator (dark mode)
│   ├── analysis.py                        # Basic analysis utilities
│   └── pairs.py                           # Pair generation utilities
├── experiments/
│   ├── 01-15: Early experiments           # Noise-space, Lipschitz, Ridge Explorer
│   ├── 16-18: Cross-arch (old, 3 triplets)
│   ├── 19: Novelty confound control
│   ├── 21: Ridge method comparison        # DINOv2 vs CLIP vs pixel
│   ├── 22_definitive.py                   # Main pipeline (5 phases, 25 triplets)
│   ├── 22_analyze.py                      # Full analysis with regression + covariates
│   ├── 22_run_all.sh                      # Run all phases
│   ├── 23_compare_steps.py                # 4-step vs 10-step comparison
│   ├── 25_step_ab_test.py                 # Human A/B: 4-step vs 10-step
│   ├── 25_analyze_ab.py                   # A/B analysis
│   ├── 26_mixed_dataset.py                # Build mixed 4+10 step eval
│   ├── 27_add_lowlevel_metrics.py         # Compute image covariates
│   ├── 28_seed_stability.py               # Seed-to-seed correlation analysis
│   ├── 29_crossarch_grids.py              # Generate DS8/SD3.5/PixArt grids (25 triplets)
│   ├── 30_crossarch_analysis.py           # 4-model correlation analysis
│   └── 31_random_triplets.py              # Random prompt validation (resumable)
├── fisher/                                # Jacobian, Lipschitz, sensitivity
├── generation/                            # SD 1.5 / Dreamshaper8 pipeline wrapper
├── outputs/
│   ├── 22_definitive_10step/              # Flux 10-step: 25 triplets, sensitivity, pools
│   │   ├── triplet_00..24/                # Per-triplet: *.npy, pools.json, sensitivity_maps.png
│   │   ├── eval/                          # 10-step only eval pairs
│   │   └── seed_stability_analysis.json   # Exp 28 results
│   ├── 26_mixed_eval/                     # ← CURRENT EVAL DATASET
│   │   ├── pairs.json                     # 1,750 pairs with covariates
│   │   ├── images/                        # 3,500 JPEG images
│   │   └── responses/                     # Annotator JSONL files
│   ├── 29_crossarch_ds8_g50/              # DS8 sensitivity maps (25 triplets)
│   ├── 29_crossarch_sd35_g50/             # SD3.5 sensitivity maps (25 triplets)
│   ├── 29_crossarch_pixart_g50/           # PixArt sensitivity maps (25 triplets)
│   ├── 30_crossarch_analysis/             # Cross-arch correlation results + plots
│   │   ├── summary.json
│   │   ├── per_triplet.json
│   │   └── crossarch_correlations.png
│   └── 31_random_triplets/                # Random prompt validation (in progress)
│       ├── triplets.json                  # 25 random triplets from 160-prompt bank
│       └── triplet_00..24/                # Sensitivity maps (being generated)
├── PAPER_SUMMARY.md                       # Paper draft summary (OUTDATED — use this file)
├── EXPERIMENT_LOG.md                      # ← THIS FILE
└── RESEARCH_DIRECTIONS.md                 # Ideation results
```

---

## Key Commands

```bash
# Start annotator (dark mode, port 7860)
nohup python -u -m evaluation.annotate \
  --eval-dataset outputs/26_mixed_eval/pairs.json \
  --images-dir outputs/26_mixed_eval/images \
  --responses-dir outputs/26_mixed_eval/responses \
  --port 7860 --share > /tmp/annotator_mixed.log 2>&1 &

# Run analysis on collected annotations
python -m experiments.22_analyze --eval-dir outputs/26_mixed_eval

# Resume random triplet generation (GPUs 3-5)
python -m experiments.31_random_triplets --phase generate

# Analyze random vs hand-picked triplet ridges
python -m experiments.31_random_triplets --phase analyze

# Run cross-architecture analysis (already done, just re-run if needed)
python -m experiments.30_crossarch_analysis

# Generate grids for a new model (e.g. pixart on GPUs 0-5)
python -m experiments.29_crossarch_grids --model pixart --gpu-start 0 --gpu-end 6
```

---

## Known Issues and Honest Weaknesses

1. **Circularity**: DINOv2 defines ridges AND determines visual character of selected images. Partially mitigated by CLIP/pixel convergence and low-level covariates.
2. **Off-manifold interpolation**: Linear interpolation goes through regions no real prompt produces. Ridges may mark manifold departure, not model-intrinsic boundaries.
3. **Lobashev overlap**: Core phenomenon (phase boundaries in generative models) is not novel. Our contribution is the text-embedding space angle and cross-architecture comparison.
4. **No valid human data**: Pilot data is from the authors. Exp 19 confound control showed null. The interestingness hypothesis is **unvalidated**.
5. **PixArt falsifies "text encoder determines boundaries"**: PixArt (T5-XXL) anti-correlates with Flux/SD3.5 despite sharing T5. Architecture matters too.
6. **Z-Image data at lower resolution (256×256, 30×30 grid)**: Confounded. Only 3 triplets. Best to drop from Paper 1.
7. **Hand-picked triplets**: 25 triplets chosen by us. Exp 31 (random triplets) addresses this but is still running.
