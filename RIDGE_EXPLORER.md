# Ridge Explorer — Technical Documentation

## Overview

Ridge Explorer is an interactive tool for detecting and visualizing **phase boundary ridges** in text-to-image diffusion model latent spaces. Users provide 2-4 text prompts, and the tool generates a grid of images along the interpolation between them, revealing regions of high sensitivity where small perturbations in the text embedding cause abrupt semantic changes in the generated image.

**Core thesis**: The latent spaces of text-to-image models contain sharp phase boundaries — ridges where the output transitions between distinct semantic modes (e.g., "giraffe" to "airplane"). These ridges are mesoscale features that are omnipresent, cross-architecture, and exhibit critical exponents consistent with the 3D Ising universality class.

**Repository**: https://github.com/Pixelkram/ridge-explorer
**Hardware**: 6× NVIDIA RTX 4090 (24GB VRAM each)
**Model**: FLUX.2 Klein 4B (black-forest-labs/FLUX.2-klein-base-4B)

---

## System Architecture

```
Browser (React + Zustand)               FastAPI Backend (Python)
├── PromptInput ─── POST ──────────▶   /api/grid/start          (full explore)
│                   POST ──────────▶   /api/grid/fast-scan       (Jacobian scan)
├── ProgressBar
├── ScanCompletePanel ── POST ─────▶   /api/grid/{id}/generate-selected
├── RefinePanel ──────── POST ─────▶   /api/grid/{id}/refine
├── UnifiedViewport (2D)                /api/grid/{id}/status     (polling)
│   ├── image layer                     /cache/thumbnails/{hash}.jpg
│   ├── heatmap layer
│   └── tau selection layer
├── RidgeViewer3D (Plotly)              /api/grid/{id}/ridge_mesh.json
└── CellDetail / SeedProbe              /api/grid/{id}/seed-probe

GPU Pool (6× RTX 4090):
  Each GPU: FLUX Klein 4B (bfloat16, ~16GB) + DINOv2 ViT-B/14 (~0.5GB)
  Persistent worker processes via torch.multiprocessing (spawn)
  Task types: GenerateTask, FastScanTask, HQTask
  Results via shared mp.Queue → async result_collector in main process
```

### Directory Structure

```
ridge_explorer/
├── backend/
│   ├── main.py                    # FastAPI app, lifespan, result collector
│   ├── config.py                  # GPU count, model config, grid defaults
│   ├── models.py                  # Pydantic schemas (requests + responses)
│   ├── services/
│   │   ├── gpu_pool.py            # Multi-GPU worker pool, task dispatch
│   │   ├── ridge_detector.py      # DINOv2 sensitivity, Jacobian spectral norm, DBSCAN
│   │   ├── visualization.py       # Matplotlib heatmap, overlay, cluster rendering
│   │   └── grid_builder.py        # Grid coordinate computation
│   ├── routers/
│   │   ├── health.py              # GET /health
│   │   └── grid.py                # All grid endpoints (start, fast-scan, refine, status)
│   └── cache/
│       └── thumbnail_cache.py     # Content-addressed disk cache (MD5 hash)
├── frontend/
│   ├── vite.config.ts             # Vite + React, proxies /api → backend
│   └── src/
│       ├── App.tsx                 # All UI components (PromptInput, ProgressBar, etc.)
│       ├── api/client.ts           # HTTP client functions
│       ├── api/types.ts            # TypeScript interfaces
│       └── stores/ridgeStore.ts    # Zustand state + polling + actions
├── RIDGE_EXPLORER.md              # This file
├── ARCHITECTURE.md                # Original architecture plan
├── FAST_RIDGE_DETECTION.md        # Jacobian method specification
└── SPEEDUP_RESEARCH.md            # Performance optimization research
```

---

## Two Detection Methods

### Method 1: Full DINOv2 Exploration (Original)

For each grid point on the N×N interpolation grid:

1. **Interpolate text embeddings**: `emb = (1-α-β)·A + α·B + β·C`
2. **Generate image**: FLUX Klein 4B, 4 steps, 256×256, guidance_scale=4.0
3. **Extract DINOv2 embedding**: ViT-B/14 → 768-dim L2-normalized vector
4. **Compute sensitivity**: Mean cosine distance to 4 grid neighbors

**Cost**: ~36 seconds for 20×20 grid on 6 GPUs (4 denoising steps + VAE decode + DINOv2 per point)
**Quality**: Ground truth — defines the ridge structure

### Method 2: Fast Jacobian Scan

For each grid point:

1. **Interpolate text embeddings**: Same as above
2. **One denoising step**: FLUX Klein, 1 step, 256×256, guidance_scale=1.0, output_type='latent'
3. **L2-normalize** the flattened latent vector
4. **Compute 2×2 Jacobian Gram matrix** from grid-neighbor finite differences:
   - `∂f/∂α ≈ (f[i+1,j] - f[i-1,j]) / 2Δα` (central, O(h²))
   - `∂f/∂β ≈ (f[i,j+1] - f[i,j-1]) / 2Δβ` (central, O(h²))
   - Boundary: 2nd-order one-sided stencil `(-3f₀ + 4f₁ - f₂) / 2h`
5. **Spectral norm** = √(max eigenvalue of J^T J) → ridge sensitivity

**Cost**: ~5 seconds for 20×20 grid on 6 GPUs (1 forward pass per point, no image decode, no DINOv2)
**Quality**: Spearman ρ = 0.797 vs DINOv2, Jaccard@20% = 0.455

**Key insight**: Ridges are mesoscale features at grid spacing ~0.02. Fine-epsilon Jacobian (ε=1e-4) gives ρ = -0.013 — zero correlation. Coarse grid-neighbor finite differences are the correct approach.

### Comparison

| Metric | Full DINOv2 | Fast Jacobian |
|--------|------------|---------------|
| Time (20×20, 6 GPUs) | 36s | **5s** |
| Speedup | 1× | **7×** |
| Forward passes/point | ~10 (4 steps × CFG + VAE + DINOv2) | **1** |
| Produces images | Yes | No (latents only) |
| ρ vs DINOv2 ground truth | 1.000 | 0.797 |
| Jaccard@20% | 1.000 | 0.455 |

---

## Batched Direct-Transformer Inference

The fast scan bypasses the diffusers pipeline and calls the FLUX Klein transformer directly:

```python
# Pre-compute once per scan:
emb_a, text_ids = pipe.encode_prompt(prompt_a)
noise, latent_ids = pipe.prepare_latents(...)
timestep = scheduler.timesteps[0]  # single step

# Per batch of 8 grid points:
batch_embeds = interpolate_embeddings(alphas, betas)  # (8, seq, dim)
velocity = transformer(
    hidden_states=noise.expand(8, -1, -1),
    timestep=(t / 1000).expand(8),
    encoder_hidden_states=batch_embeds,
    txt_ids=text_ids.expand(8, -1, -1),
    img_ids=latent_ids.expand(8, -1, -1),
)
denoised = noise - velocity  # Euler step for rectified flow
```

This eliminates ~78ms of Python pipeline overhead per point (input validation, scheduler setup, generator creation, queue serialization). Results are sent as row-level batches (`LatentBatchResult`) to reduce multiprocessing queue overhead.

**Attempted optimizations that did NOT work:**
- `torch.compile(mode="reduce-overhead")`: 60s startup, no runtime gain — CUDA graph overhead dominates at 256 image tokens
- Early exit via ModuleList replacement: 20× regression — PyTorch re-registers parameters internally
- Hierarchical subsampling within workers: Too complex with distributed row assignment

---

## User Workflow

### Fast Scan → Selective Generation

1. Enter 3 prompts, set grid size (up to 100×100)
2. Click **Fast Scan** → Jacobian sensitivity heatmap in ~5s (20×20)
3. Adjust τ threshold to select ridge regions
4. Click **Generate N Images** → only generates images where ridges are
5. View images overlaid with sensitivity heatmap
6. Optionally **Refine** selected regions at higher resolution (2× multiplier)

### Full Exploration

1. Enter 3 prompts, set grid size, resolution, steps
2. Click **Explore** → progressive image generation with DINOv2 sensitivity
3. View images/heatmap/tau layers, toggle between them
4. Refine ridge regions iteratively
5. Right-click cells for manual selection, double-click for full-res overlay with seed probe

### 3D Exploration

1. Switch to 3D mode, enter 4 prompts
2. Explore generates a 3D grid with Plotly visualization
3. Marching cubes extracts ridge isosurface
4. Z-slice browser for 2D cross-sections
5. Click points for image overlay, hover for 160px preview

---

## Frontend Features

| Feature | Description |
|---------|-------------|
| Pan/zoom viewport | Pointer-drag pan, scroll-wheel zoom centered on cursor |
| Layer compositing | Toggle images / heatmap / tau selection independently |
| Multi-seed support | 1-5 seeds, averaged sensitivity, per-seed thumbnails |
| Manual selection | Right-click cells to add/remove from refinement set |
| Seed probe | Double-click any cell → generate across 20 seeds to see variability |
| H key recenter | Fits grid to viewport |
| Cancel button | Abort any running scan/generation |
| 3D Plotly viewer | Interactive 3D scatter + mesh3d isosurface, camera orbit |

---

## Backend Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/grid/start` | Full explore: generate images + DINOv2 embeddings |
| POST | `/api/grid/fast-scan` | Fast Jacobian scan: 1-step latents + spectral norm |
| POST | `/api/grid/{id}/generate-selected` | Generate images for cells above τ threshold |
| POST | `/api/grid/{id}/refine` | Subdivide and regenerate high-sensitivity cells |
| GET | `/api/grid/{id}/status` | Poll job progress, get cell data + sensitivity |
| POST | `/api/grid/{id}/seed-probe` | Generate one point across many seeds |
| GET | `/api/grid/{id}/heatmap.png` | Server-rendered matplotlib heatmap |
| GET | `/api/grid/{id}/overlay.png` | Heatmap + thumbnails + contour lines |
| GET | `/api/grid/{id}/ridge_mesh.json` | 3D marching cubes isosurface |
| GET | `/health` | GPU status |

---

## Ridge Detection Pipeline (ridge_detector.py)

### DINOv2 Sensitivity (`compute_sensitivity`)

- **2D**: Mean cosine distance to 4 grid neighbors (±1 in each axis)
- **3D**: Mean cosine distance to 6 grid neighbors
- Skips zero-norm embeddings (non-generated cells from partial fast-scan generation)
- Output: scalar sensitivity per cell

### Jacobian Spectral Norm (`compute_jacobian_sensitivity`)

- Input: (gs, gs, D) array of L2-normalized 1-step latent vectors
- Central finite differences interior, 2nd-order one-sided at boundaries
- 2×2 Gram matrix J^T J with closed-form eigenvalues
- Boundary clamping: edge cells set to nearest interior value (eliminates stencil artifacts)
- Output: spectral_norm (ridge indicator) + anisotropy (0=ridge, 1=isotropic)

### DBSCAN Clustering (`compute_clusters`)

- Pairwise cosine distance matrix
- DBSCAN(eps=0.1, min_samples=3, metric='precomputed')
- Labels semantic regions separated by ridges

### 3D Ridge Mesh (`extract_ridge_mesh`)

- `skimage.measure.marching_cubes` on 3D sensitivity field
- Threshold: median × τ
- Returns vertices + faces as JSON for Plotly mesh3d

---

## Refinement System

The iterative refinement system allows progressively higher resolution exploration of ridge regions:

1. **Initial grid**: N×N at user-selected resolution/steps
2. **Tau selection**: `threshold = median(sensitivity) × τ` — cells above threshold are selected
3. **Subdivision**: Each selected cell splits into `multiplier × multiplier` sub-cells
4. **Span system**: Non-selected cells retain their parent image at `span > 1`, covering a block of the refined grid
5. **Re-analysis**: After refinement generation, DINOv2 sensitivity is recomputed

For fast-scan jobs, the Jacobian sensitivity is preserved for non-generated cells, and DINOv2 sensitivity computed only for cells with real embeddings.

---

## Research Results

### Ridge Universality

From the parallel research effort (`search_problem/`), validated on 25 prompt triplets across 4 architectures:

| Model Pair | Mean ρ | 95% CI | Interpretation |
|------------|--------|--------|----------------|
| Flux ↔ SD3.5 | +0.491 | [+0.42, +0.56] | Strong agreement |
| Flux ↔ DreamShaper8 | +0.361 | [+0.26, +0.46] | Moderate agreement |
| DS8 ↔ SD3.5 | +0.255 | [+0.16, +0.35] | Weak agreement |
| Flux ↔ PixArt | **-0.214** | [-0.33, -0.10] | **Anti-correlated** |

**Key finding**: CLIP-containing models produce positively correlated ridge maps. PixArt (T5-XXL only) produces an **inverted** sensitivity landscape — where CLIP-based models see ridges, PixArt sees valleys. This falsifies the simple hypothesis that "text encoder determines boundaries."

### Three-Metric Convergence

Ridges are not DINOv2 artifacts:
- DINOv2 vs CLIP: ρ = 0.94
- DINOv2 vs Pixel MSE: ρ = 0.65
- CLIP vs Pixel MSE: ρ = 0.75

### Critical Exponents

From earlier ridge_explorer work:
- β ≈ 0.346, 95% CI [0.325, 0.394]
- Consistent with 3D Ising universality class
- 512 fits with bootstrap confidence intervals

### Seed Stability

- Mean seed-to-seed Spearman ρ = 0.64 (range: 0.27-0.87 across 25 triplets)
- Ridges are moderately stable — broad structure is reproducible but fine-grained ranking shifts
- Averaging across 3+ seeds recommended for robust detection

### Noise-Space Independence

- Noise-space Lipschitz sensitivity: ρ = 0.13-0.44 with text-space DINOv2 ridges (inconsistent)
- **Genuine negative result**: text-embedding ridges and noise-space sensitivity measure different phenomena

### Low-Level Image Metrics

Boundary images are **less colorful** (-15.5%, p<0.001) than non-boundary images. No significant difference in JPEG complexity, edge density, or entropy. This is favorable for the interestingness hypothesis.

---

## Performance Summary

| Operation | Time (20×20, 6 GPUs) | Notes |
|-----------|----------------------|-------|
| Fast Jacobian scan | **~5s** | 1 fwd pass/point, no images |
| Full DINOv2 explore | ~36s | 4 steps + DINOv2/point, produces images |
| Generate selected (after scan) | ~15-30s | Only ridge cells, varies with τ |
| Refine 2× | ~20-40s | Depends on selected region size |

### Speedup Evolution

| Version | Time (20×20) | Speedup | Change |
|---------|-------------|---------|--------|
| Full DINOv2 explore | 36s | 1× | Baseline |
| Unbatched fast scan (pipe per point) | 6.2s | 5.8× | 1-step + Jacobian |
| + guidance_scale=1.0, 256px | 6.2s | 5.9× | Halved forward passes |
| + Batched direct transformer | 5.0s | 7.2× | Bypass pipeline overhead |
| + Row-level result batching | 5.0s | 7.2× | Reduced queue serialization |
| + Adaptive poll interval | 5.0s | 7.2× | Faster result collection |

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RIDGE_N_GPUS` | 6 | Number of GPUs to use |

### Key Defaults (config.py)

| Parameter | Value | Description |
|-----------|-------|-------------|
| MODEL_ID | FLUX.2-klein-base-4B | Generation model |
| DEFAULT_HEIGHT/WIDTH | 256 | Fast generation resolution |
| DEFAULT_NUM_INFERENCE_STEPS | 4 | Denoising steps |
| DEFAULT_GUIDANCE_SCALE | 4.0 | CFG scale |
| HQ_HEIGHT/WIDTH | 512 | High-quality re-render |
| RIDGE_THRESHOLD_TAU | 1.5 | Default ridge threshold |
| RESULT_POLL_INTERVAL | 0.5s | Base polling interval (adaptive: 0.05s when active) |

### Starting the Explorer

```bash
cd ridge_explorer/

# Backend (from ridge_explorer/)
RIDGE_N_GPUS=6 uvicorn backend.main:app --host 0.0.0.0 --port 8000

# Frontend (from ridge_explorer/frontend/)
cd frontend && npx vite --host 0.0.0.0 --port 5173
```

---

## Known Limitations

1. **Off-manifold interpolation**: Linear interpolation in embedding space passes through regions no real prompt produces. Ridges may mark manifold departure, not model-intrinsic boundaries.

2. **DINOv2 circularity**: DINOv2 defines ridges AND determines visual character of selected images. Partially mitigated by three-metric convergence (CLIP, pixel MSE agree).

3. **Square grid, not simplex**: The grid covers [0,1]² rather than the simplex α+β≤1. Points where α+β>1 produce embeddings with negative weight on prompt A, which is off-simplex.

4. **No valid human evaluation data**: Author-only pilot showed 79% preference; confound control showed chance (52%). 1,750-pair dataset awaits independent annotators.

5. **torch.compile / early exit not viable**: Runtime patching of FLUX Klein's ModuleList causes severe performance regression. Would require modifying diffusers source directly.

---

## Publication Status

### Paper 1 (Computational) — Largely Ready

- [x] Ridges exist (25 hand-picked + random triplets, Flux 10-step)
- [x] Three-metric convergence (DINOv2, CLIP, Pixel MSE)
- [x] Cross-architecture analysis (4 models, 25 triplets, proper CIs)
- [x] Seed stability (mean ρ=0.64)
- [x] Critical exponents (β≈0.346, 3D Ising)
- [x] Noise-space negative result
- [x] Fast Jacobian detection method (ρ=0.797, 7× speedup)
- [ ] Paper writing

### Paper 2 (Perceptual) — Awaiting Human Data

- [ ] 20-30 independent annotators on 1,750-pair dataset
- [ ] Covariate-controlled analysis
- [ ] If positive: combined computational + perceptual paper
- [ ] If null: honest negative result

### Interactive Tool

- [x] Full 2D exploration with iterative refinement
- [x] 3D exploration with marching cubes + Plotly
- [x] Fast Jacobian scan (7× speedup)
- [x] Multi-seed evaluation
- [x] Manual cell selection + seed probe
- [x] Pan/zoom viewport with layer compositing
