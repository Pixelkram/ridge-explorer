# Fast Ridge Detection via Coarse Jacobian Spectral Norm

## Overview

Detect sensitivity ridges in text-embedding interpolation space using a single denoising step per grid point, then computing the 2×2 Jacobian Gram matrix from grid-neighbor finite differences. This is the Pareto-optimal method: Jaccard@20% = 0.328 with DINOv2 ground truth at 1,275 forward passes (vs 42,000 for full DINOv2).

**Key finding**: Ridges are mesoscale features (grid spacing ~0.02), not differential features. Fine-epsilon Jacobian (ε=1e-4) gives ρ=-0.013 — zero correlation. Coarse grid-neighbor Jacobian gives ρ=0.569. Use grid-neighbor finite differences, not infinitesimal derivatives.

---

## Algorithm

### Input
- Three text prompts: A, B, C
- Grid size N (default: 50)
- Fixed random seed
- Model pipeline (Flux Klein or similar)

### Step 1: Encode Prompts
```python
ea = pipe.encode_prompt(prompt=A)[0]  # (1, seq_len, dim)
eb = pipe.encode_prompt(prompt=B)[0]
ec = pipe.encode_prompt(prompt=C)[0]
```

### Step 2: Generate One-Step Latents on Grid
For each valid grid point (i, j) where α_i + β_j ≤ 1:
```python
alpha = i / (N - 1)
beta = j / (N - 1)
emb = (1 - alpha - beta) * ea + alpha * eb + beta * ec

latent = pipe(
    prompt_embeds=emb,
    height=512, width=512,
    num_inference_steps=1,
    guidance_scale=4.0,
    generator=fixed_seed_generator,
    output_type='latent',
).images[0]

# Flatten and L2-normalize
vec = latent.flatten().float()
vec = vec / vec.norm()
latent_grid[i, j] = vec
```

**Cost**: 1 forward pass per point × ~1,275 valid points = **1,275 forward passes**

### Step 3: Compute 2×2 Jacobian Gram Matrix at Each Point

For each grid point (i, j), estimate the Jacobian columns via finite differences with grid neighbors:

```python
da = 1.0 / (N - 1)  # grid spacing

# ∂f/∂α via central differences (or one-sided at boundary)
if (i+1, j) in grid and (i-1, j) in grid:
    df_da = (grid[i+1, j] - grid[i-1, j]) / (2 * da)
elif (i+1, j) in grid:
    df_da = (grid[i+1, j] - grid[i, j]) / da
else:
    df_da = (grid[i, j] - grid[i-1, j]) / da

# ∂f/∂β similarly
if (i, j+1) in grid and (i, j-1) in grid:
    df_db = (grid[i, j+1] - grid[i, j-1]) / (2 * da)
# ... same one-sided fallback

# 2×2 Gram matrix J^T J
a11 = dot(df_da, df_da)  # ||∂f/∂α||²
a22 = dot(df_db, df_db)  # ||∂f/∂β||²
a12 = dot(df_da, df_db)  # ⟨∂f/∂α, ∂f/∂β⟩

# Eigenvalues (closed-form for 2×2 symmetric matrix)
trace = a11 + a22
det = a11 * a22 - a12 * a12
disc = max(trace² - 4 * det, 0)
lambda_max = (trace + sqrt(disc)) / 2
lambda_min = (trace - sqrt(disc)) / 2

# Outputs
spectral_norm = sqrt(lambda_max)  # max singular value
anisotropy = lambda_min / lambda_max  # 0 = pure ridge, 1 = isotropic
```

**Cost**: Zero additional forward passes (computed from Step 2 data)

### Step 4: Build Sensitivity Map

The `spectral_norm` at each grid point is the ridge detection signal.
- High spectral norm = high sensitivity = near a ridge
- Low anisotropy at high spectral norm = directional sensitivity = on a ridge

---

## Output Maps

| Map | What it shows | Use case |
|-----|--------------|----------|
| `spectral_norm[i,j]` | Overall sensitivity (max singular value) | Primary ridge indicator |
| `anisotropy[i,j]` | Directional ratio (0=ridge, 1=uniform) | Ridge vs isotropic sensitivity |
| `spectral_norm * (1-anisotropy)` | Ridge-specific sensitivity | Hybrid indicator |

---

## Performance

Validated on Triplet 0 (airplane / giraffe / toilet paper), Flux Klein 4B:

| Metric | Coarse Jacobian | 1-step cosine | DINOv2 (reference) |
|--------|----------------|---------------|-------------------|
| Spearman ρ with DINOv2 | **0.569** | 0.409 | 1.000 |
| Jaccard@20% | **0.328** | 0.272 | 1.000 |
| F1@20% | **0.494** | 0.427 | 1.000 |
| Forward passes | 1,275 | 1,275 | 42,000 |
| Time (1× RTX 4090) | ~4 min | ~4 min | ~2.5 hours |

---

## Critical Implementation Notes

### 1. Use Grid-Neighbor Distances, NOT Fine Epsilon
```
WRONG:  df_da = (f(α+1e-4) - f(α)) / 1e-4     → ρ = -0.013 (useless)
RIGHT:  df_da = (f(α+Δα) - f(α-Δα)) / (2·Δα)  → ρ = 0.569 (works)
```
Ridges are mesoscale features at Δα ≈ 0.02, not differential features.

### 2. L2-Normalize Latents Before Computing Distances
Without normalization, the Gram matrix is dominated by latent magnitude variation rather than directional change.

### 3. Fixed Seed is Essential
All grid points must use the same random seed. Otherwise, stochastic variation (seed-to-seed ρ=0.64) dominates the signal.

### 4. guidance_scale=1.0 for Single-Step
With `guidance_scale=1.0`, the pipeline runs one forward pass (no CFG). With `guidance_scale>1.0`, it runs two (conditional + unconditional). Use 1.0 for speed if only doing ridge detection (not generating viewable images). Our experiments used 4.0 but 1.0 would halve the cost.

### 5. One Denoising Step is Optimal
More steps (4-step: Jaccard=0.214) are WORSE because:
- Additional denoising refines texture/detail, not semantic structure
- Cosine distance on refined latents captures texture variation, not concept transitions
- The first step captures the broadest semantic "sketch" — exactly what ridges are

---

## Interactive Ridge Explorer Integration

### Real-Time Mode (for interactive exploration)

For a Ridge Explorer tool where users navigate the simplex:

1. **Precompute phase**: Generate the full 50×50 one-step latent grid (~4 min on RTX 4090). Compute the spectral norm map. Store both.

2. **Display phase**: Show the spectral norm heatmap on the simplex. Overlay with the anisotropy map to highlight directional ridges.

3. **On-demand refinement**: When user zooms into a region, generate full 10-step images at selected points for display. Use the precomputed spectral norm to highlight which images are at ridges.

4. **Multi-model comparison**: Precompute spectral norm maps for multiple models on the same triplet. Show side-by-side or overlay to visualize cross-architecture ridge agreement/disagreement.

### Streaming Mode (for real-time feedback)

For a tool where users input 3 prompts and see ridges immediately:

1. Encode 3 prompts (~0.1s)
2. Generate one-step latents on a 20×20 coarse grid (~80 points, ~16s at 5 pts/s)
3. Compute coarse spectral norm map (<0.1s)
4. Display immediately — user sees approximate ridge structure in ~17s
5. Background: refine to 50×50 grid (~4 min), update display when ready

### API

```python
def detect_ridges(pipe, prompt_a, prompt_b, prompt_c, grid_size=50, seed=42):
    """
    Returns:
        spectral_norm: (N, N) array — ridge sensitivity map
        anisotropy: (N, N) array — directional indicator (0=ridge, 1=isotropic)
        alphas: (N,) array — alpha coordinates
        betas: (N,) array — beta coordinates
    """
```

---

## Barycentric Coordinate Convention

```
emb(α, β) = (1 - α - β) · emb_A + α · emb_B + β · emb_C

Grid: alphas[i] = i / (N-1), betas[j] = j / (N-1)
Valid region: α + β ≤ 1 (simplex constraint)

Image filename: g{i:03d}_{j:03d}.jpg
  - i indexes alpha (prompt B weight)
  - j indexes beta (prompt C weight)

Vertex A (α=0, β=0): pure prompt A — grid point (0, 0)
Vertex B (α=1, β=0): pure prompt B — grid point (N-1, 0)
Vertex C (α=0, β=1): pure prompt C — grid point (0, N-1)
```

---

## Reference Implementation

See: `experiments/45_jacobian_ridge_detection.py`

Key functions:
- `get_onestep_latent(emb)` — single denoising step, returns normalized latent
- Main loop — computes Gram matrix from grid-neighbor finite differences
- Eigenvalue computation — closed-form 2×2 spectral decomposition
