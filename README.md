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

### Hiding panels (demo / recording builds)

The research surfaces added after the original tool — **Hikers**, **Cascade**,
**Discovery**, the **ridge itinerary**, **MF Scan** and the **surprise** sampler — can be
switched off without touching the code, so a screen recording can show just the original
scan → threshold → generate → refine loop.

Set it once at server start:

```bash
VITE_RIDGE_HIDE=og npm run dev        # every tab opens on the original surface
```

…or per tab, which wins over the env var and needs no restart between takes:

| URL | Effect |
|---|---|
| `?hide=og` | the preset: hides hikers, cascade, discovery, itinerary, MF scan, surprise |
| `?hide=hikers,cascade` | hand-picked |
| `?hide=og&show=itinerary` | preset minus one |
| `?hide=none` | everything back, even if `VITE_RIDGE_HIDE` hid it |

Individually gateable: `hikers`, `cascade`, `discovery`, `itinerary`, `mfscan`,
`surprise`, `explore`, `fastscan`, `refine`, `seeds`, `threed`. The `og` preset keeps
Explore, Fast Scan, Refine, the seed controls and the 2D/3D selector — add them by name
(`?hide=og,explore`) to strip further. Names are case- and separator-insensitive
(`MF-Scan` = `mfscan`) and a few aliases work (`hike`, `discover`, `3d`); an unrecognised
name logs a console warning and is ignored rather than breaking the page. The resolved
set is logged at startup and readable at `window.__ridgeFeatures`.

The same flags drive the mobile view's tab bar, which hides itself when only one tab is
left. Definitions live in `frontend/src/featureFlags.ts`; `node test/features.test.mjs`
covers the resolution rules.

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

## Exact-JVP probes (grid-free)

Two measurements computed at a *single* point without a grid, by exact forward-mode Jacobian-vector products
through the generator (`backend/services/jvp_probe.py`; evidence in
`search_problem/outputs/h07_chain_SUMMARY.md` and `h08_vector_uses/RESULTS.md`):

- **Crossing direction** — double-click a cell → *Probe crossing direction* (~20 s). Two JVPs give the direction
  in prompt space along which the image changes fastest at that point: the ridge normal, whitened by the prompt
  embeddings' Gram matrix so the text encoder's own anisotropy is removed. It is seed-invariant (cross-seed
  |cos| 0.93 raw / 0.91 whitened, out of sample) and drawn as a sign-free segment on the map, with the rank-1
  share (~0.99 on a front) and participation ratio (~1 front, ~2 corner). σ₁ is comparable only between probes
  of the same job — a global "ridge / not ridge" threshold was refuted.
- **Which words matter** — per prompt, one JVP per token position (~2 min). Rankings are seed-invariant
  (Spearman 0.66 across seeds) and follow the word, not its slot. They say which word the generation depends on
  at this point — not how the image would change; that direction is seed noise.

Both run on the resident pool workers: the probe patches the norms for forward AD, casts the VAE to fp32, parks
the text encoder on the CPU and freezes weights for the duration, then restores everything — the tool's own
generation is bit-identical afterwards (`tests/probe_e2e.py`). Routes: `POST /api/grid/{job}/jvp-probe`,
`GET /api/grid/jvp-probe/{id}`, `POST /api/grid/{job}/token-probe`, `GET /api/grid/token-probe/{id}`.

- **k ≥ 4 without a grid** — the same probe on a Cascade run (`POST /api/cascade/{run}/jvp-probe` with a
  crossing `cid` or a k-vector of `weights`; in the UI, select a crossing → *Probe crossing direction*) or on any
  prompt basis at all (`POST /api/probe/jvp` with `prompts` and `weights`). A grid over k prompts costs ~n^(k−1)
  generations; the probe costs k − 1 JVPs (~10 s each). Its whitened spectrum reads the local geometry: rank-1
  share ≈ 0.98 = a single front, participation ratio ≥ 1.5 = a corner where fronts overlap. Out of sample at
  k = 4–5 the sharpest points are fronts (rank-1 0.976, 10/10 prompt sets) and corners grow with k. Use it for
  direction and codimension; its absolute magnitude is not calibrated against a lattice.
- **Following a ridge at k ≥ 4** — a front is a (k − 2)-dimensional surface, not a curve, so the standalone
  k = 3 tracer (crest-tracking by chord score; stayed on the crest only 2/4 times) is not shipped. Instead the
  cascade's existing fan walk gets a **JVP normal** option (checkbox next to the walk buttons, `use_jvp` in
  `POST /api/cascade/{run}/walk`): one exact JVP at the origin crossing (~30 s) replaces the bracket chord — an
  isotropic line that happened to cross — by the true normal for the transversal probes and the true tangent
  plane for the stations. Whether that keeps the ridge longer was measured, not assumed: `tests/walk_ab.py`
  ran both walks from the same origins (rule pre-registered in its docstring). **Result (run `4985c1b7`, k = 4,
  25 origin/direction pairs): INCONCLUSIVE.** Both walks captured 5 stations in total; 18/25 pairs captured
  nothing either way; of the 7 that differed, the JVP walk won 3 (Wilcoxon p = 0.86), at +28 s per walk. The
  JVP normal and the bracket chord do differ (median |cos| 0.73), so the transversal direction is simply not the
  walk's bottleneck — staying on the front is. The option is therefore shipped **off by default**; the
  measurement is in `tests/walk_ab_4985c1b7.json`.
- **What actually kept the walk from following the ridge** (`RESEARCH_ridge_following_k4.md`): the diagnostic
  `tests/walk_diag.py` showed 28/36 walks stopping at the first station with "ridge changed identity" — the pair
  straddled a boundary but failed the *absolute* same-ridge test (both new sides within cosine distance 0.35 of
  the origin's sides), which cannot hold along a front whose content is location-specific. The walk now uses a
  **relative** signature by default (`sig_mode` in `POST /api/cascade/{run}/walk`: each new side must be closer to
  its matching origin side than to the opposite one). Pre-registered A/B on a fresh k = 4 run: relative won
  **24/24** differing pairs (Wilcoxon p = 1.5e-5), mean captured stations 0.19 → 1.81, +1 s per walk.
- **Following ridges that bend: the continuation walk (default since 2026-09-26)** — `mode` in
  `POST /api/cascade/{run}/walk` (`continuation` | `fan`); UI: *walk* selector next to the ridge test. With the
  relative rule in place, the fan's remaining losses were drift and edge exits: its stations sit on one straight
  line, and that line is only perpendicular to the random chord that found the crossing, not to the ridge. The
  continuation walk is predictor–corrector continuation in the same plane: predict along the secant through the
  last two captured points, render a short line of probes 0.025 apart across the prediction in one GPU round, take
  the boundary nearest the prediction (pairs up to 0.10 apart count as evidence — real fronts spread their change
  over several probes), bisect once; the step grows ×1.5 after a success and halves after a miss. Stations stream
  into the UI as they are found. **Pre-registered A/B** (`tests/PREREG_walk_continuation.md`, 3 fresh k = 4 runs,
  46 walks from significant crossings, independently re-computed in `tests/walk_pc_verify/`): **SUPPORTED** —
  further in 21 of 27 differing walks, mean distance along the boundary 0.156 → 0.197 (Wilcoxon p = 0.008;
  0.02–0.03 when the two directions from one crossing are treated as one cluster); distance on stations that
  re-certify on held-out seeds 0.112 → 0.150; drift endings 10 → 1, full-budget walks 9 → 17. Caveats: the gain
  comes from 2 of the 3 prompt sets (the third was a 5–5 tie); it costs ~1.8× the images per walk (33.5 vs 18.3,
  median 31 s vs 14 s); and "same ridge" is still soft — 18 % of stations were chosen on probe lines that crossed
  more than one boundary. Results: `tests/walk_pc_results.json`; design and diagnosis:
  `RESEARCH_ridge_following_k4.md` §6.
- **Certifying a walk** — `POST /api/cascade/{run}/walk/{walk}/certify` re-measures every station with the
  cascade's boundary statistic B on three seeds the walk never used, against the run's background at the same
  seeds (`cert` in the walk status). A walk's stations are single-seed until certified.
  In the UI: **certify on 3 unseen seeds** next to a finished walk; stations turn green (hold) or grey (do not),
  in the thumbnail strip and on the map.
- **Strict ridge test** (`sig_mode = "continuity"`, UI: *ridge test → strict*): each new station's sides may change
  by at most half the previous station's contrast (`PC_CONT_GAMMA`). Development test against a dense ground truth
  (`RESEARCH_ridge_following_k4.md` §7, not yet confirmed on fresh prompts): about half the ridge switches, about a
  quarter less distance — a strictness dial, so `relative` stays the default.
- **Trace phase** (`trace: true` in `POST /api/cascade/start`, UI: *trace ridges*): after the survey, walk both ways
  from every significant crossing with the strict rule, draw the walked ridges on the map, and link crossings a walk
  reaches (`traces`, `trace_links`, `traced_*` in the status). The traced unexplored share is **experimental**: on
  the development field it did not improve on the plain estimate, and walk links can chain different ridges together
  (§7). More chords is the reliable way to a better coverage estimate.
- **Ground-truth tooling** — `POST /api/cascade/{run}/render` renders any list of weight vectors through a run's own
  generation path and saves the DINOv2 embeddings (`tests/gt/`); `chord_seed` lets several surveys share one image
  field; `tests/walk_gt_lib.py` turns a dense k = 3 lattice into regions, ridges and chord-crossing weights.
- Fixed 2026-09-26: a walk from crossing 0 errored ("no tangent direction") whenever that crossing lay on chord 0 —
  the walk's tangent seed and chord 0's direction came from the same random stream. It now redraws; all other
  walks keep their exact plane.

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
