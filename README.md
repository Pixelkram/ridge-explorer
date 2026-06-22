# Ridge Explorer

An interactive, multi-GPU tool for discovering and visualizing **phase-boundary
"sensitivity ridges"** in the embedding space of text-to-image diffusion models.

Ridge Explorer interpolates between text prompts on a grid, generates an image at
each grid point, and computes a sensitivity field that reveals where small changes
in the text embedding cause abrupt visual transitions. These high-sensitivity
regions form ridge structures analogous to phase boundaries in statistical physics.
The backend targets **FLUX.2 Klein (4B)** with **DINOv2** sensitivity scoring; the
frontend is a React + Zustand + Plotly single-page app.

![stack](https://img.shields.io/badge/backend-FastAPI%20%2B%20PyTorch-blue) ![stack](https://img.shields.io/badge/frontend-React%20%2B%20Vite-green)

## Features

- **Two detection methods** — full DINOv2 perceptual sensitivity (ground truth,
  ~36 s for a 20×20 grid) and a fast 1-step **Jacobian spectral-norm scan**
  (~5 s, ≈7× faster, Spearman ρ ≈ 0.80 vs DINOv2).
- **Embedding interpolation** across the grid with NLERP (normalized linear
  interpolation) to preserve magnitude on the embedding hypersphere.
- **Scan → select → generate** flow: fast-scan, drag a `τ` threshold to select
  ridge cells, then render full images only in the selected regions.
- **Hierarchical refinement** — subdivide high-sensitivity cells (2–8× multiplier)
  while reusing parent thumbnails for low-sensitivity cells (no extra GPU cost).
- **2D mode** (3 prompts): pan/zoom viewport with toggleable image / heatmap / `τ`
  layers, manual right-click cell selection, per-cell multi-seed probe.
- **3D mode** (4 prompts): Plotly scatter colored by sensitivity, marching-cubes
  ridge isosurface, and a z-slice browser for 2D cross-sections.
- **Multi-GPU worker pool** — each GPU loads FLUX.2 Klein 4B + DINOv2 ViT-B/14,
  with cancellable, batched inference.

## Requirements

- **NVIDIA GPU(s) with CUDA — required.** There is no CPU fallback. Each worker
  process holds a full FLUX.2 Klein 4B (bf16) pipeline **and** DINOv2 on one GPU,
  so you need one suitable GPU per worker (default 6; set `RIDGE_N_GPUS` lower for
  fewer GPUs).
- **Python 3.11+** (developed on 3.13) with:
  `torch` (CUDA build), `torchvision`, `fastapi`, `uvicorn`, `pydantic`, `numpy`,
  `scipy`, `scikit-learn`, `scikit-image`, `matplotlib`, `Pillow`, `transformers`,
  `sentencepiece`.
  - ⚠️ **`diffusers` must be a dev build that includes the FLUX.2 pipeline**
    (`Flux2KleinPipeline` + `diffusers.pipelines.flux2`). A stable release will
    not have it and the workers crash on import. Install from source:
    `pip install "git+https://github.com/huggingface/diffusers.git"`
- **Node.js 18+** (developed on 22) and **npm** for the frontend.
- **Model weights** (downloaded automatically on first run, then cached):
  - `black-forest-labs/FLUX.2-klein-base-4B` (~23 GB) — **gated on Hugging Face**.
    Accept the license, then `export HF_TOKEN=hf_…` so the first download
    succeeds. Subsequent runs load from `~/.cache/huggingface`.
  - DINOv2 ViT-B/14 (reg) via `torch.hub` — cached under `~/.cache/torch/hub`.

## Running it

The app is two processes: a FastAPI backend (port **8001**) and a Vite dev server
(port **5173**) that proxies API calls to the backend.

### 1. Backend

Run from the repo root (the directory that contains the `backend/` package — the
package uses absolute `from backend import …` imports, so the working directory
matters):

```bash
cd ridge_explorer
export HF_TOKEN=hf_...          # only needed the first time, for the gated FLUX download
RIDGE_N_GPUS=6 uvicorn backend.main:app --host 0.0.0.0 --port 8001
```

- `RIDGE_N_GPUS` — number of GPU worker processes to spawn (default `6`). Set it to
  your GPU count, e.g. `RIDGE_N_GPUS=1`.
- Startup loads the models into every worker and takes **~60–80 s**; wait for the
  `Ridge Explorer ready` log line. Check health with `curl localhost:8001/health`.
- `CUDA_VISIBLE_DEVICES` controls which physical GPUs the workers use.

### 2. Frontend

In a second terminal:

```bash
cd ridge_explorer/frontend
npm install        # first time only
npm run dev        # serves on http://localhost:5173
```

Open **http://localhost:5173**. The dev server proxies `/api`, `/cache`, and
`/health` to `http://localhost:8001` (configured in `frontend/vite.config.ts`).

> **Port note:** the frontend proxy and the backend `--port` must match. Both are
> set to **8001** here. If you change one, change the other in `vite.config.ts`.

### Production build (optional)

```bash
cd ridge_explorer/frontend
npm run build      # outputs to dist/
npx vite preview   # serve the built bundle
```

## Typical workflow

1. Pick **2D** (3 prompts) or **3D** (4 prompts) mode and enter your prompts, grid
   size, resolution, steps, seed, and seed count.
2. Click **Fast Scan** for a Jacobian sensitivity heatmap in a few seconds.
3. Drag the **`τ` threshold** slider to select ridge cells (live count shown), then
   **Generate N Images** to render full images only where the ridges are.
   - Or click **Explore** for a full DINOv2 pass that generates every cell with
     automatic ridge analysis on completion.
4. In the viewport: pan (drag), zoom (scroll, centered on the hovered tile), toggle
   layers, recenter with **H**, double-click a cell for a full-res overlay with
   seed-probe controls, right-click to force cells into the refinement set.
5. Set a `τ` and multiplier and click **Refine** to zoom into ridge regions at
   higher resolution; iterate as needed.
6. In 3D: orbit the Plotly scatter, toggle the marching-cubes mesh, and scrub the
   z-slice slider.

## Architecture

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full design. In brief:

- `backend/main.py` — FastAPI app; a `lifespan` hook starts the multiprocessing
  `GPUPool`.
- `backend/services/gpu_pool.py` — spawn-based worker pool; each worker loads
  FLUX.2 + DINOv2 and serves generate / fast-scan / HQ tasks with mid-task
  cancellation.
- `backend/routers/grid.py` — the grid API (`/api/grid/start`, `/fast-scan`,
  `/mf-scan`, `/refine`, `/generate-selected`, `/seed-probe`, `/cancel`, …).
- `frontend/src/` — React + Zustand state, Plotly visualizations, the 2D viewport,
  and the API client.

Generated thumbnails and results are cached on disk under `cache_data/`
(gitignored).
