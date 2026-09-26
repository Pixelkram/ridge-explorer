# Ridge Explorer — Project Description

Ridge Explorer is an interactive tool for discovering and visualizing **phase boundary ridges** in the latent space of text-to-image diffusion models. It maps the semantic landscape between text prompts by interpolating their embeddings on a grid, generating images at each point, and computing sensitivity metrics that reveal where small changes in the text embedding cause abrupt visual transitions. These transitions form ridge structures analogous to phase boundaries in statistical physics.

The system targets **FLUX.2 Klein 4B** (`black-forest-labs/FLUX.2-klein-base-4B`) and runs on a multi-GPU backend (default 6x RTX 4090).

---

## Core Concept

Given 3 text prompts (A, B, C) in 2D mode, or 4 prompts (A, B, C, D) in 3D mode, the tool interpolates their text embeddings across a grid:

```
emb(α, β) = (1 - α - β) · A + α · B + β · C       (2D)
emb(α, β, γ) = (1-α-β-γ)·A + α·B + β·C + γ·D      (3D)
```

where α, β, γ ∈ [0, 1]. At each grid point, the model generates an image. The tool then computes how rapidly the output changes across the grid — regions of high sensitivity are **ridges**, marking sharp semantic transitions (e.g., where "a photo of a cat" abruptly becomes "a photo of a dog").

Research has shown these ridges exhibit **3D Ising universality** with critical exponent β ≈ 0.346, suggesting deep connections between diffusion model geometry and statistical physics.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Frontend (React + Zustand + Plotly)            │
│  Vite dev server :5173                          │
│  Proxies /api/* and /cache/* to backend         │
└───────────────────┬─────────────────────────────┘
                    │ HTTP (polling every 1.5s)
┌───────────────────▼─────────────────────────────┐
│  Backend (FastAPI, async)                       │
│  Uvicorn :8000                                  │
│                                                 │
│  ┌─────────────────────────────────────────┐    │
│  │  Result Collector (background task)      │    │
│  │  Drains result_queue, stores embeddings  │    │
│  │  Triggers analysis when all cells done   │    │
│  └─────────────────────────────────────────┘    │
│                                                 │
│  ┌─────────────────────────────────────────┐    │
│  │  GPU Pool (multiprocessing, spawn)      │    │
│  │  task_queue → N workers → result_queue  │    │
│  │  Each worker: FLUX Klein + DINOv2       │    │
│  │  on dedicated GPU                       │    │
│  └─────────────────────────────────────────┘    │
└─────────────────────────────────────────────────┘
```

### Backend Structure

```
backend/
├── main.py                  # FastAPI app, lifespan, result collector, analysis pipelines
├── config.py                # Model ID, GPU count, defaults, thresholds, paths
├── models.py                # Pydantic request/response schemas
├── routers/
│   ├── grid.py              # All API endpoints (grid start, status, refine, fast-scan, etc.)
│   └── health.py            # GPU pool health check
├── services/
│   ├── gpu_pool.py          # Multi-GPU worker pool, FLUX + DINOv2 loading, inference
│   ├── ridge_detector.py    # Sensitivity, Jacobian, DBSCAN, marching cubes
│   ├── visualization.py     # Matplotlib rendering (heatmap, overlay, clusters)
│   ├── grid_builder.py      # Grid coordinate generation, noise tensor creation
│   └── point_store.py       # UMAP-ready point collection (unused currently)
└── cache/
    └── thumbnail_cache.py   # Content-addressed thumbnail storage (MD5 hashed)
```

### Frontend Structure

```
frontend/
├── src/
│   ├── main.tsx             # React entry point
│   ├── App.tsx              # Main application component (~1073 lines, all UI)
│   ├── index.css            # Global dark theme styles
│   ├── api/
│   │   ├── client.ts        # HTTP request wrappers (8 API functions)
│   │   └── types.ts         # TypeScript interfaces for API payloads
│   └── stores/
│       └── ridgeStore.ts    # Zustand state management + polling logic
├── vite.config.ts           # Vite + API proxy to localhost:8000
├── package.json             # React 18, Zustand 4, Plotly.js 3
└── tsconfig.json            # Strict mode, ES2020
```

---

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/grid/start` | Start full grid generation (image + DINOv2 per cell) |
| GET | `/api/grid/{job_id}/status` | Poll job state, get cells, visualization URLs |
| POST | `/api/grid/fast-scan` | 1-step latent-only scan (no images), ~7x faster |
| POST | `/api/grid/{job_id}/generate-selected` | Generate images only for cells above tau threshold |
| POST | `/api/grid/{job_id}/refine` | Hierarchical grid refinement in high-sensitivity regions |
| POST | `/api/grid/{job_id}/seed-probe` | Generate single point across multiple seeds |
| GET | `/api/grid/{job_id}/heatmap.png` | Sensitivity heatmap image |
| GET | `/api/grid/{job_id}/overlay.png` | Images + heatmap overlay with contours |
| GET | `/api/grid/{job_id}/clusters.png` | DBSCAN cluster visualization |
| GET | `/api/grid/{job_id}/images.png` | Assembled image grid |
| GET | `/api/grid/{job_id}/export/{layer}.jpg` | Full-resolution export (images/heatmap/overlay) |
| GET | `/api/grid/{job_id}/ridge_mesh.json` | 3D marching cubes mesh (vertices + faces) |
| POST | `/api/grid/cancel` | Cancel pending GPU tasks |
| GET | `/health` | GPU pool status |

---

## Algorithms

### 1. Prompt Embedding Interpolation

The system encodes 3 or 4 text prompts into embedding vectors using FLUX's text encoders (CLIP + T5). Grid points are created by interpolating these embeddings:

**NLERP (Normalized Linear Interpolation):**
The system uses NLERP rather than plain LERP to better approximate SLERP on the embedding hypersphere:
1. Linearly interpolate: `emb = Σ wᵢ · embᵢ`
2. Compute average norm: `avg_norm = mean(||embᵢ||)`
3. Normalize: `emb = emb · (avg_norm / ||emb||)`

This preserves embedding magnitude, which matters for stable generation.

### 2. DINOv2-Based Sensitivity (Ground Truth)

The primary ridge detection method uses DINOv2 (ViT-B/14) visual embeddings:

1. Generate a full image at each grid point (4 diffusion steps, 256px default)
2. Extract 768-dimensional DINOv2 embedding from each image, L2-normalize
3. For each cell, compute **mean cosine distance to neighbors**:
   - 2D: 4-neighbors (up, down, left, right)
   - 3D: 6-neighbors (±1 in each axis)
   - `sensitivity(i,j) = mean(1 - dot(e_ij, e_neighbor))` for valid neighbors
4. High sensitivity = output changes rapidly = ridge region

**Performance:** ~36s for a 20×20 grid on 6 GPUs.

### 3. Jacobian Spectral Norm (Fast Scan)

A much faster alternative that operates on raw latents without generating images:

1. Run only **1 denoising step** at each grid point, keeping the latent tensor
2. Flatten and **L2-normalize** the latent vectors
3. Compute **finite-difference Jacobian** using grid neighbors:
   - Central differences for interior: `∂f/∂α ≈ (f[i+1,j] - f[i-1,j]) / (2Δα)`
   - Forward/backward differences at boundaries
4. Build the **Gram matrix** G = JᵀJ:
   - 2D: 2×2 matrix with entries `a₁₁=||∂f/∂α||²`, `a₂₂=||∂f/∂β||²`, `a₁₂=⟨∂f/∂α, ∂f/∂β⟩`
   - 3D: 3×3 matrix (vectorized via `np.einsum`)
5. Compute eigenvalues:
   - 2D: Closed-form quadratic solution for λ₁, λ₂
   - 3D: `np.linalg.eigvalsh` on batched (gs, gs, gs, 3, 3) array
6. **Spectral norm** = √λ_max (primary ridge indicator)
7. **Anisotropy** = λ_min / λ_max (0 = strongly directional ridge, 1 = isotropic change)

**Critical insight:** Ridges are **mesoscale features** (grid spacing ~0.02). Using infinitesimal-epsilon Jacobians gives ρ ≈ -0.01 (useless). The grid-neighbor finite-difference Jacobian gives ρ ≈ 0.80 correlation with DINOv2 ground truth.

**Performance:** ~5s for a 20×20 grid on 6 GPUs (7x faster than full DINOv2).

### 4. DBSCAN Clustering

Groups grid cells into semantic clusters based on DINOv2 embedding similarity:

1. Flatten all embeddings to (N, 768)
2. Compute pairwise cosine distance matrix: `D = 1 - (E · Eᵀ)`
3. Run DBSCAN with `eps=0.1`, `min_samples=3`, `metric='precomputed'`
4. Reshape cluster labels to spatial grid shape
5. Noise points (cluster = -1) shown in red; clusters colored with tab20 colormap

### 5. 3D Ridge Mesh Extraction (Marching Cubes)

For 3D grids, the tool extracts an isosurface of the sensitivity field:

1. Compute threshold = `median(sensitivity) × tau`
2. Run **marching cubes** (scikit-image) on the binary sensitivity volume
3. Normalize spacing by grid dimensions
4. Output: JSON with vertices, faces, threshold, and statistics
5. Rendered as a 3D mesh in Plotly alongside the scatter plot

### 6. Hierarchical Refinement (Span System)

After initial exploration, users can refine high-sensitivity regions:

1. Compute `threshold = median(sensitivity) × tau` (only from `span=1` cells)
2. Cells above threshold are subdivided by multiplier M (2-8x)
3. **Span system** (2D): Each coarse cell covers `span × span` sub-cells. On refinement:
   - High-sensitivity cells: subdivided into M×M new cells (span=1 each)
   - Low-sensitivity cells: kept as-is with `span = M` (no GPU cost)
   - Parent thumbnails/embeddings are reused for non-refined regions
4. **3D refinement**: Simple grid upsampling — regenerates above-threshold cells at higher resolution
5. Users can also manually select cells via right-click for forced inclusion

### 7. Multi-Seed Averaging

To reduce stochastic noise from the random seed:

1. User specifies `seed_count` (1-5)
2. Backend creates sub-jobs, one per seed
3. Sensitivity maps from each seed are averaged
4. Frontend stores per-seed thumbnails; user can toggle which seed's images to view
5. Empirically, seed-to-seed ridge correlation is ρ ≈ 0.64, so averaging helps

---

## Visualization Renders (Backend, Matplotlib)

The backend generates four static visualization images:

1. **Heatmap** (`heatmap.png`): Sensitivity values with `hot` colormap, axes labeled α and β, title shows max/median ratio
2. **Image Grid** (`images.png`): Montage of all cell thumbnails, y-axis flipped to match coordinate space
3. **Overlay** (`overlay.png`): Thumbnails with semi-transparent sensitivity heatmap overlaid, contour lines at τ = 1.2, 1.5, 1.8 in different colors
4. **Clusters** (`clusters.png`): DBSCAN cluster boundaries with colored regions, noise points in red, cluster IDs labeled

---

## Frontend UI

### Layout

The UI is a vertically stacked single-page app with a dark theme (#1a1a2e background, #e94560 accent red, #4ecca3 accent teal):

```
┌────────────────────────────────────────────┐
│  Header: "Ridge Explorer" + subtitle       │
├────────────────────────────────────────────┤
│  PromptInput Panel                         │
│  [2D/3D] [Prompt A] [Prompt B] [Prompt C]  │
│  [Prompt D (3D only)]                      │
│  Grid: [slider 3-100]  Res: [dropdown]     │
│  Steps: [dropdown]  Seed: [input]          │
│  Seeds: [1-5]                              │
│  [Explore] [Cancel] [Fast Scan]            │
├────────────────────────────────────────────┤
│  ProgressBar (phase-aware)                 │
│  "Fast scan: 245/400 latents" [██████░░░]  │
├────────────────────────────────────────────┤
│  ScanCompletePanel (after fast scan)       │
│  τ threshold: [slider 0-4]                 │
│  "147/400 cells above threshold"           │
│  Res: [dropdown]  Steps: [dropdown]        │
│  [Generate 147 Images]                     │
├────────────────────────────────────────────┤
│  RefinePanel (after completion)            │
│  τ: [slider]  Multiplier: [slider 2-8]    │
│  "Above τ: 42 + 5 manual = 47 cells"      │
│  "New cells: 47 × 4 = 188"                │
│  [Refine]                                  │
├────────────────────────────────────────────┤
│  MainViewport                              │
│  ┌────────────────────────────────────┐    │
│  │ Layer toggles: [images][heatmap]   │    │
│  │ [tau] | Seed selector | [Export]   │    │
│  │                                    │    │
│  │  ┌──┬──┬──┬──┬──┬──┐             │    │
│  │  │  │  │  │  │  │  │  Interactive │    │
│  │  ├──┼──┼──┼──┼──┼──┤  pan/zoom   │    │
│  │  │  │  │  │  │  │  │  canvas      │    │
│  │  ├──┼──┼──┼──┼──┼──┤             │    │
│  │  │  │  │  │  │  │  │             │    │
│  │  └──┴──┴──┴──┴──┴──┘             │    │
│  │                                    │    │
│  │ "scroll=zoom | drag=pan |          │    │
│  │  dblclick=fullres | rclick=select  │    │
│  │  | H=recenter"                     │    │
│  └────────────────────────────────────┘    │
└────────────────────────────────────────────┘
```

### 2D Viewport (UnifiedViewport)

An interactive canvas with pan and zoom for exploring the grid:

- **Scroll wheel**: Zoom in/out, centered on the hovered tile
- **Left-click drag**: Pan the viewport
- **Double-click**: Open full-resolution image overlay with seed probe controls
- **Right-click**: Toggle manual cell selection (red border) for forced refinement
- **H key**: Recenter viewport to fit entire grid
- **ResizeObserver**: Auto-recenters when the viewport container resizes

**Layer toggles** (composited on each cell):
- **images**: Show cell thumbnails
- **heatmap**: Color overlay by sensitivity (red=high, green=medium, blue=low)
- **tau**: Green border on cells above the current τ threshold

**Seed selector**: When multiple seeds are used, buttons switch which seed's thumbnails are displayed.

**Export buttons**: Download full-resolution JPEGs of the images, heatmap, or overlay layers.

### Full-Resolution Overlay

Clicking a cell opens a large image view with:
- The cell's coordinates (α, β)
- **Seed Probe** controls: specify a range of seeds, click "Generate" to see the same point rendered with different noise realizations
- A gallery of seed variant thumbnails (128×128)
- Click outside to dismiss

### 3D Viewport (RidgeViewer3D)

For 3D grids, switches to a Plotly-based 3D scatter plot:

- **Scatter3D**: Points at (α, β, γ) coordinates, colored by sensitivity
- **Ridge mesh**: Optional marching cubes isosurface overlaid (toggle on/off)
- **Z-slice slider**: Browse 2D slices through the volume
- **Hover**: 160×160 thumbnail preview at cursor position
- **Click**: Full-resolution image overlay
- **H key**: Reset camera to default view

### Color Scheme

| Element | Color | Hex |
|---------|-------|-----|
| Background | Very dark blue | #1a1a2e |
| Panel background | Dark blue | #16213e |
| Input background | Darker blue | #0f3460 |
| Primary accent (buttons, selections) | Red/pink | #e94560 |
| Secondary accent (success, tau borders) | Teal/cyan | #4ecca3 |
| Text | Light gray | #e0e0e0 |
| Borders | Dark gray | #333 |

---

## Workflows

### Workflow 1: Full Exploration

1. Enter 3 prompts (e.g., "a cat", "a dog", "a bird")
2. Set grid size (e.g., 15), resolution (256), steps (4), seed
3. Click **Explore**
4. Watch progress bar: generating → analyzing → complete
5. Toggle layers (images/heatmap/tau) to explore results
6. Double-click interesting cells for full-resolution view
7. Use seed probe to check stability across noise realizations
8. Adjust τ, right-click to manually select cells, click **Refine** to zoom in on ridges

### Workflow 2: Fast Scan + Selective Generation

1. Enter prompts and parameters
2. Click **Fast Scan** (runs 1-step latent-only evaluation, ~7x faster)
3. After scan completes, adjust τ threshold to select interesting cells
4. See count of cells above threshold
5. Set desired resolution and steps for final generation
6. Click **Generate N Images** to render only the selected cells
7. Results combine Jacobian sensitivity with DINOv2 embeddings

### Workflow 3: 3D Exploration

1. Select "3D" mode
2. Enter 4 prompts (A, B, C, D)
3. Run fast scan or full generation
4. Explore 3D scatter plot with sensitivity coloring
5. Toggle ridge mesh to see marching cubes isosurface
6. Use z-slice slider to browse 2D cross-sections
7. Click points for full-resolution images

---

## GPU Worker Details

Each GPU worker process loads:
- **FLUX.2 Klein 4B** (bfloat16) — the diffusion model
- **DINOv2 ViT-B/14** — for computing visual embeddings

Workers accept three task types:
- **GenerateTask**: Full pipeline — interpolate embeddings, generate image (N steps), compute DINOv2 embedding, save thumbnail
- **FastScanTask**: 1-step latent evaluation only — interpolate embeddings, run 1 denoising step, return flattened L2-normalized latent vector
- **HQTask**: High-quality render — 512×512, 20 steps, full JPEG output

Communication uses `multiprocessing.Queue`:
- `task_queue`: Main process → workers (tasks distributed in row chunks)
- `result_queue`: Workers → main process (results collected by background task)

Workers filter cells via an `active_cells` set, allowing selective generation after fast scan.

---

## Configuration Defaults

| Parameter | Default | Notes |
|-----------|---------|-------|
| Model | FLUX.2 Klein 4B | bfloat16 |
| GPU count | 6 | `RIDGE_N_GPUS` env var |
| Grid size | 15×15 | Configurable 3-100 |
| Resolution | 256px | Options: 128, 256, 384, 512 |
| Steps | 4 | Options: 2, 4, 8, 12, 20, 50 |
| Guidance scale | 4.0 | Fixed |
| τ threshold | 1.5 | Multiplier on median sensitivity |
| HQ resolution | 512px, 20 steps | For high-quality re-renders |
| Thumbnail | 64px, 85% JPEG | Content-addressed cache |
| Polling interval | 1.5s | Frontend → backend |
| DBSCAN eps | 0.1 | Cosine distance threshold |
| DBSCAN min_samples | 3 | Minimum cluster size |

---

## Key Design Decisions

1. **Mesoscale sensitivity, not differential**: Ridges are features at grid-spacing scale (~0.02), not infinitesimal. Fine-epsilon Jacobians give useless results (ρ ≈ -0.01). Grid-neighbor finite differences work (ρ ≈ 0.80).

2. **Dual metrics**: DINOv2 cosine distance is the ground truth but slow; Jacobian spectral norm is 7x faster with good correlation, enabling interactive workflows.

3. **Span system for refinement**: Avoids regenerating low-sensitivity cells. Coarse cells simply cover larger areas (span > 1) while refined cells are individually generated.

4. **NLERP over LERP**: Normalized interpolation preserves embedding magnitude on the hypersphere, critical for stable generation.

5. **Multi-process GPU pool**: Spawn-context workers with dedicated GPUs. Avoids GIL issues and GPU memory conflicts.

6. **Content-addressed thumbnail cache**: MD5-hashed storage prevents duplicate writes and enables efficient serving.

---

## Dependencies

### Backend
- Python 3.13
- FastAPI + Uvicorn
- PyTorch (bfloat16, CUDA)
- diffusers (FLUX pipeline)
- transformers (DINOv2)
- scikit-learn (DBSCAN)
- scikit-image (marching cubes)
- matplotlib (visualization renders)
- numpy
- Pillow

### Frontend
- React 18.3
- Zustand 4.5 (state management)
- Plotly.js 3.4 (3D visualization)
- react-plotly.js 2.6
- Vite 6.0 (build/dev server)
- TypeScript 5.6

---

## Starting the Application

From the project root (`ridge_explorer/`):

```bash
# Backend (must run from project root for imports)
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Frontend (from frontend directory)
cd frontend && npm run dev
```

The Vite dev server proxies `/api/*` and `/cache/*` to `localhost:8000`.
