# Ridge Explorer — Architecture & Algorithm Documentation

## Purpose

Interactive tool for exploring phase boundary ridges in text-to-image diffusion model latent spaces. Users define a 2D plane through semantic space using 2-3 text prompts, generate a grid of images, detect ridges via DINOv2 perceptual embeddings, and iteratively refine interesting regions at higher resolution.

## System Overview

```
Browser (React + Zustand + Vite)     FastAPI Backend (Python, port 8000)
├── PromptInput ──POST──────────▶   /api/grid/start
│   (3 prompts, grid size,           Creates sub-jobs per seed
│    resolution, steps, seed,        Distributes rows across 6 GPUs
│    seed_count)                     Workers: FLUX Klein + DINOv2 inline
│
├── UnifiedViewport ──GET(poll)──▶  /api/grid/{id}/status
│   Pan/zoom canvas                  Returns cells + per-seed data
│   Layer toggles (images,           Triggers analysis on completion
│   heatmap, tau)                    (sensitivity + DBSCAN clustering)
│
├── RefinePanel ──POST───────────▶  /api/grid/{id}/refine
│   tau slider + multiplier          In-place grid expansion
│                                    Only generates in selected regions
│
├── SeedProbe ──POST─────────────▶  /api/grid/{id}/seed-probe
│   (double-click cell)              Generates single point × N seeds
│   Gallery display                  Progressive polling
│
└── Static assets ──GET──────────▶  /cache/thumbnails/{hash}.jpg
                                    /api/grid/{id}/heatmap.png
                                    /api/grid/{id}/overlay.png
                                    /api/grid/{id}/clusters.png
```

## Hardware Requirements

- 6× NVIDIA GPUs with ≥20GB VRAM each (tested on RTX 4090 24GB)
- Each GPU loads: FLUX.2 Klein 4B (~16GB) + DINOv2 ViT-B/14 (~0.3GB)

## Core Algorithm

### Phase 1: Grid Generation

**Input**: 3 prompts (A, B, C), grid_size N, seed(s), resolution, steps

1. Encode prompts A, B, C into text embeddings using FLUX Klein's text encoder
2. For each grid position (i, j) where α = i/(N-1), β = j/(N-1):
   - Interpolate: `emb = (1-α-β)*A + α*B + β*C`
   - Generate image with fixed noise (from seed) at specified resolution/steps
   - Compute DINOv2 embedding (768-dim, L2-normalized)
   - Save full-resolution JPEG + embedding
3. Distribute rows across 6 GPU workers via `torch.multiprocessing` spawn context
4. Workers put `CellResult` (thumbnail_bytes + dino_embedding) on a shared result queue
5. Main process `result_collector` async task drains queue, updates job state

**Multi-seed**: Creates independent sub-jobs per seed, all sharing the same grid coordinates. Each sub-job runs the full pipeline independently.

### Phase 2: Ridge Analysis (automatic on completion)

1. **DINOv2 Neighbor Distance**: For each cell, compute mean cosine distance to 4-neighbors:
   ```
   sensitivity[i,j] = mean(1 - dot(emb[i,j], emb[ni,nj])) for (ni,nj) in 4-neighbors
   ```
   High sensitivity = large visual change between adjacent cells = ridge

2. **DBSCAN Clustering**: Run DBSCAN (eps=0.1, min_samples=3) on cosine distance matrix of all DINOv2 embeddings. Labels semantic regions; boundaries align with ridges.

3. **Multi-seed averaging**: If multiple seeds, average sensitivity maps element-wise across all seeds. This highlights ridges that are consistent across noise realizations.

4. **Visualization**: Render matplotlib heatmap, overlay (with contours at τ=1.2, 1.5, 1.8 × median), and cluster map.

### Phase 3: Iterative Refinement

**Input**: tau threshold, resolution multiplier M

**Algorithm** (`_refine_single_job`):

1. **Compute refine positions** from master job's averaged sensitivity:
   - Median computed from **only span=1 (real) cells** to avoid bias from tiled parent cells
   - Threshold = median × tau
   - `refine_positions` = set of (row, col) where span=1 AND sensitivity ≥ threshold

2. **Expand grid** from N to N×M:
   - New alpha/beta coordinates: each old cell's range subdivided into M equal parts
   - `new_alphas[i*M + si] = old_alphas[i] - da/2 + da*(si+0.5)/M`

3. **For each existing cell entry**:
   - If `(row, col) in refine_positions` (span=1, above threshold):
     → Create M×M new pending cells at fine positions → `needs_generation`
   - Else (below threshold OR already a spanned parent tile):
     → Scale position: `tl = (row*M, col*M)`
     → Scale span: `new_span = old_span * M` (clamped to grid boundary)
     → Copy parent's thumbnail + embedding to all covered positions
     → Create ONE cell entry at top-left with the new span

4. **If no cells to generate**: return without modifying the job (prevents empty expansions)

5. **Replace job in-place** with expanded grid, submit GPU tasks for `needs_generation` only

6. **Active cells filter**: `GenerateTask.active_cells` (frozenset) tells workers to skip positions not in the set, avoiding regeneration of parent-tiled areas

7. **Multi-seed**: Same `refine_positions` computed once from master, applied identically to all sub-jobs → all seeds refine the same cells

8. **On completion**: Re-run ridge analysis on the full expanded grid. Parent-tiled regions have duplicated embeddings (sensitivity=0 within blocks), but real cells at boundaries get accurate new sensitivity values.

### Span System

Cells have a `span` field indicating how many grid positions they cover:
- `span=1`: Real cell, individually generated image
- `span=N`: Parent tile from N-1 refinement iterations ago, displayed at N×N size

Span behavior across iterations:
- Initial grid: all span=1
- After 1st refine (M=2): refined cells → span=1, others → span=2
- After 2nd refine (M=2): new fine → span=1, prev fine → span=2, original → span=4
- After 3rd refine: spans = {1, 2, 4, 8}

Frontend handles spans by:
- Rendering each cell at `span × tileSize` pixels
- Tracking `coveredSet` of all positions covered by any cell (including sub-positions of spanned cells)
- Only rendering empty placeholders for uncovered positions

### Tau Threshold Consistency

Critical invariant: **frontend and backend must agree on which cells are selected**.

- Median computed from **span=1 cells only** (both frontend `computeRealMedian()` and backend)
- Only span=1 cells can be selected for refinement (spanned parent tiles are never refined)
- Fallback: if no span=1 cells exist (shouldn't happen), use all cells with sensitivity > 0

### Seed Probe

**Input**: (alpha, beta) coordinates, seed range [start, end]

1. Creates a 1×1 grid sub-job per seed
2. Each generates one image at the specified coordinates with that seed
3. Results polled progressively, displayed as a gallery in the overlay modal

## File Structure

```
ridge_explorer/
├── backend/
│   ├── __init__.py
│   ├── config.py                 # MODEL_ID, N_GPUS, defaults
│   ├── main.py                   # FastAPI app, lifespan, result_collector
│   ├── models.py                 # Pydantic: GridStartRequest, CellStatus, etc.
│   ├── routers/
│   │   ├── grid.py               # /start, /refine, /status, /seed-probe
│   │   └── health.py             # /health
│   ├── services/
│   │   ├── gpu_pool.py           # GPUPool, GenerateTask, CellResult, worker_main
│   │   ├── ridge_detector.py     # compute_sensitivity, compute_clusters
│   │   ├── visualization.py      # render_heatmap, render_overlay, render_clusters
│   │   ├── grid_builder.py       # (legacy, unused)
│   │   └── point_store.py        # (legacy, unused)
│   └── cache/
│       └── thumbnail_cache.py    # Disk cache with hash-based paths
├── frontend/
│   ├── package.json
│   ├── vite.config.ts            # Proxy /api, /cache → localhost:8000
│   ├── tsconfig.json
│   └── src/
│       ├── main.tsx
│       ├── index.css
│       ├── App.tsx               # All UI components (single file)
│       ├── api/
│       │   ├── client.ts         # startGrid, getGridStatus, refineGrid, etc.
│       │   └── types.ts          # TypeScript interfaces
│       └── stores/
│           └── ridgeStore.ts     # Zustand state + polling logic
└── cache_data/
    ├── thumbnails/               # Hash-based JPEG cache
    └── results/                  # Per-job matplotlib renders
```

## Frontend State (Zustand)

Key state fields:
- `jobId`, `phase` (idle/generating/analyzing/complete)
- `cells` (CellStatus[]) — primary cell data with averaged sensitivity
- `seedCells` (Record<string, CellStatus[]>) — per-seed cell data
- `seeds`, `activeSeedIdx` — multi-seed navigation
- `tau`, `multiplier` — refinement controls
- `currentGridSize` — tracks grid size (grows on refine)
- `shouldCenter` — true on fresh generate, false on refine (preserves pan/zoom)
- `steps`, `resolution`, `seedCount` — generation parameters

Polling: `startPolling()` fires immediately then every 1.5s. Updates cells, grid_size, phase, image URLs. Stops on `phase === 'complete'`.

## Frontend Viewport

- **Pan**: Pointer events on container div, `setPointerCapture` for reliable drag
- **Zoom**: Native wheel listener (passive:false), zooms toward hovered tile with 30% interpolation
- **Tile hover detection**: Converts screen→world coords `(screen - pan) / zoom`, then `floor(world / tileSize)` → grid row/col
- **Layer compositing**: Each cell renders as absolutely-positioned div with stacked layers (image, heatmap color, tau border)
- **Inner grid**: `pointerEvents: 'none'` so it doesn't steal pointer events from the container

## Backend Concurrency

- Workers: 6 `torch.multiprocessing.Process` (spawn context, daemon=True)
- Communication: `mp.Queue` for tasks (main→workers) and results (workers→main)
- Result collector: `asyncio` background task in the main process, polls queue every 0.5s
- No locks needed: only the result_collector writes to job state, only the main thread reads it for API responses
- `active_cells` (frozenset) passed to workers for refine — workers skip non-active positions

## Starting

```bash
cd ridge_explorer

# Terminal 1: Backend (takes ~60-80s for GPU model loading)
uvicorn backend.main:app --host 0.0.0.0 --port 8000

# Terminal 2: Frontend
cd frontend && node ./node_modules/vite/bin/vite.js --host 0.0.0.0 --port 5173
```

## Known Issues / TODOs

1. **Grid size mismatch after refine**: `max(row)+1` from cells can differ from `grid_size` because spanned cells don't have entries at every position. Frontend uses `grid_size` from status response (not inferred).

2. **Stale GPU workers**: If workers crash silently (e.g., OOM from previous session), generation hangs at 0/N. Fix: kill all GPU processes (`nvidia-smi --query-compute-apps=pid | xargs kill -9`) and restart.

3. **Coverage overflow**: Spanned cells near grid edges can extend beyond boundary. Spans are clamped via `min(new_span, new_gs - tl_i, new_gs - tl_j)`.

4. **Sensitivity at tiled regions**: Parent-tiled blocks have identical embeddings → sensitivity=0 within blocks. This is correct behavior (no visual change within a tiled region) but means the heatmap shows artificial "cold zones" at coarse areas.

5. **DBSCAN eps not adaptive**: Fixed at 0.1. Works well on average (ρ=0.611) but fails on some grids. Could be made adaptive based on the sensitivity distribution.
