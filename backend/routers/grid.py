import uuid
import numpy as np
from fastapi import APIRouter, Request
from fastapi.responses import FileResponse

from backend.models import (
    GridStartRequest, GridStartResponse, GridStatusResponse, CellStatus,
    RenderHQRequest, RefineRequest, RefineResponse,
    SeedProbeRequest, SeedProbeResponse, SeedProbeStatus,
    FastScanRequest, FastScanResponse,
    GenerateSelectedRequest, GenerateSelectedResponse,
    MFScanRequest, MFScanResponse,
)
from backend.services.gpu_pool import GenerateTask, HQTask, FastScanTask
from backend import config


router = APIRouter(prefix="/api/grid")


def _flatten_cells(job):
    """Flatten 2D or 3D cell arrays into a flat list of CellStatus."""
    flat = []
    is_3d = job.get("dimensions", 2) == 3

    if is_3d:
        for row in job["cells"]:
            for col_list in row:
                for cell in col_list:
                    if cell is not None:
                        flat.append(CellStatus(**cell))
    else:
        for row in job["cells"]:
            for cell in row:
                if cell is not None:
                    flat.append(CellStatus(**cell))
    return flat


@router.post("/cancel")
async def cancel_job(request: Request):
    """Cancel all pending GPU work. Drains the task queue and clears old jobs."""
    pool = request.app.state.gpu_pool
    drained = pool.drain_pending()
    # Also drain any completed results sitting in the result queue
    # so they don't get attributed to future jobs
    stale_results = pool.collect_results()
    # Clear all jobs so in-flight GPU results get discarded by the collector
    jobs = request.app.state.jobs
    n_cleared = len(jobs)
    jobs.clear()
    return {"status": "ok", "drained": drained, "stale_results": len(stale_results), "jobs_cleared": n_cleared}


@router.post("/start", response_model=GridStartResponse)
async def start_grid(req: GridStartRequest, request: Request):
    pool = request.app.state.gpu_pool
    pool.clear_cancel()  # Allow workers to process new tasks
    master_id = str(uuid.uuid4())[:8]
    gs = req.grid_size
    alphas = np.linspace(config.ALPHA_RANGE[0], config.ALPHA_RANGE[1], gs)
    betas = np.linspace(config.BETA_RANGE[0], config.BETA_RANGE[1], gs)

    is_3d = req.dimensions == 3
    gammas = np.linspace(config.ALPHA_RANGE[0], config.ALPHA_RANGE[1], gs) if is_3d else None
    use_slerp = is_3d  # enable SLERP for 3D by default

    jobs = request.app.state.jobs
    pool = request.app.state.gpu_pool

    seeds = list(range(req.seed, req.seed + req.seed_count))
    cells_per_seed = gs * gs * gs if is_3d else gs * gs
    total_cells_all = cells_per_seed * len(seeds)

    # Create a sub-job per seed
    sub_job_ids = []
    for seed in seeds:
        sub_id = f"{master_id}_s{seed}"
        jobs[sub_id] = _make_job(sub_id, gs, alphas, betas,
                                 req.prompt_a, req.prompt_b, req.prompt_c, seed,
                                 height=req.height, width=req.width,
                                 steps=req.steps, guidance_scale=req.guidance_scale,
                                 dimensions=req.dimensions, prompt_d=req.prompt_d,
                                 gammas=gammas, use_slerp=use_slerp)
        sub_job_ids.append(sub_id)

        chunks = [list(range(gs))[i::pool.n_gpus] for i in range(pool.n_gpus)]
        for chunk in chunks:
            if chunk:
                pool.submit(GenerateTask(
                    job_id=sub_id, row_indices=chunk,
                    alphas=alphas, betas=betas,
                    prompt_a=req.prompt_a, prompt_b=req.prompt_b, prompt_c=req.prompt_c,
                    grid_size=gs, seed=seed,
                    height=req.height, width=req.width,
                    steps=req.steps, guidance_scale=req.guidance_scale,
                    prompt_d=req.prompt_d, gammas=gammas,
                    grid_size_z=gs if is_3d else 0,
                    use_slerp=use_slerp,
                ))

    # Master job aggregates all seeds
    # For single seed, master IS the sub-job (backward compatible)
    if len(seeds) == 1:
        # Alias master to the single sub-job
        jobs[master_id] = jobs[sub_job_ids[0]]
        jobs[master_id]["seeds"] = seeds
        jobs[master_id]["sub_job_ids"] = sub_job_ids
    else:
        jobs[master_id] = {
            "phase": "generating",
            "status": "running",
            "dimensions": req.dimensions,
            "grid_size": gs,
            "total_cells": total_cells_all,
            "cells_generated": 0,
            "prompt_a": req.prompt_a,
            "prompt_b": req.prompt_b,
            "prompt_c": req.prompt_c,
            "prompt_d": req.prompt_d,
            "seed": req.seed,
            "seeds": seeds,
            "sub_job_ids": sub_job_ids,
            "height": req.height, "width": req.width,
            "steps": req.steps, "guidance_scale": req.guidance_scale,
            "use_slerp": use_slerp,
            "alphas": alphas, "betas": betas, "gammas": gammas,
            "embeddings": np.zeros((gs, gs, gs, 768)) if is_3d else np.zeros((gs, gs, 768)),
            "sensitivity": None, "clusters": None,
            "thumbnails": {}, "thumbnail_hashes": {},
            "cells": jobs[sub_job_ids[0]]["cells"],  # use first sub's cell structure
            "heatmap_path": None, "overlay_path": None,
            "cluster_path": None, "image_grid_path": None,
            "ridge_mesh_path": None,
            "render_version": 0,
        }

    return GridStartResponse(job_id=master_id, total_cells=total_cells_all,
                             dimensions=req.dimensions, status="running")


def _make_job(job_id, gs, alphas, betas, prompt_a, prompt_b, prompt_c, seed,
              height=256, width=256, steps=4, guidance_scale=4.0,
              dimensions=2, prompt_d="", gammas=None, use_slerp=False):
    if dimensions == 3 and gammas is not None:
        gs_z = len(gammas)
        total = gs * gs * gs_z
        cells = [[[{
            "row": i, "col": j, "depth": k,
            "alpha": float(alphas[i]), "beta": float(betas[j]), "gamma": float(gammas[k]),
            "status": "pending",
            "sensitivity": None, "cluster": None,
            "thumbnail_url": None, "hq_url": None, "span": 1,
        } for k in range(gs_z)] for j in range(gs)] for i in range(gs)]
        embeddings = np.zeros((gs, gs, gs_z, 768))
    else:
        total = gs * gs
        cells = [[{
            "row": i, "col": j,
            "alpha": float(alphas[i]), "beta": float(betas[j]),
            "status": "pending",
            "sensitivity": None, "cluster": None,
            "thumbnail_url": None, "hq_url": None, "span": 1,
        } for j in range(gs)] for i in range(gs)]
        embeddings = np.zeros((gs, gs, 768))

    return {
        "phase": "generating",
        "status": "running",
        "dimensions": dimensions,
        "grid_size": gs,
        "total_cells": total,
        "cells_generated": 0,
        "prompt_a": prompt_a,
        "prompt_b": prompt_b,
        "prompt_c": prompt_c,
        "prompt_d": prompt_d,
        "seed": seed,
        "height": height,
        "width": width,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "use_slerp": use_slerp,
        "alphas": alphas,
        "betas": betas,
        "gammas": gammas,
        "embeddings": embeddings,
        "sensitivity": None,
        "clusters": None,
        "thumbnails": {},
        "thumbnail_hashes": {},
        "cells": cells,
        "heatmap_path": None, "overlay_path": None,
        "cluster_path": None, "image_grid_path": None,
        "ridge_mesh_path": None,
        "render_version": 0,
    }


def _refine_single_job(sub_id, jobs, refine_positions, mult, cache, pool):
    """Refine a single job in-place.

    Args:
        refine_positions: set of (row, col) in the OLD grid that should be refined.
            Computed once from the master's averaged sensitivity.
    Returns (needs_generation set, new_gs) or None.
    """
    old = jobs.get(sub_id)
    if not old or old["phase"] != "complete":
        return None

    old_gs = old["grid_size"]
    old_alphas = old["alphas"]
    old_betas = old["betas"]

    new_gs = old_gs * mult
    da = old_alphas[1] - old_alphas[0] if old_gs > 1 else 1.0
    db = old_betas[1] - old_betas[0] if old_gs > 1 else 1.0

    new_alphas = np.zeros(new_gs)
    new_betas = np.zeros(new_gs)
    for i in range(old_gs):
        for si in range(mult):
            new_alphas[i * mult + si] = old_alphas[i] - da / 2 + da * (si + 0.5) / mult
    for j in range(old_gs):
        for sj in range(mult):
            new_betas[j * mult + sj] = old_betas[j] - db / 2 + db * (sj + 0.5) / mult
    new_alphas = np.clip(new_alphas, 0, 1)
    new_betas = np.clip(new_betas, 0, 1)

    new_embeddings = np.zeros((new_gs, new_gs, 768))
    new_thumbnails = {}
    new_thumbnail_hashes = {}
    new_cells = [[None for _ in range(new_gs)] for _ in range(new_gs)]
    needs_generation = set()

    for row in old["cells"]:
        for cell in row:
            if cell is None:
                continue
            ci, cj = cell["row"], cell["col"]
            old_span = cell.get("span", 1)

            # Use the pre-computed refine_positions from master (averaged sensitivity + manual)
            should_refine = (ci, cj) in refine_positions

            if should_refine:
                # Subdivide the full span×span area into fine cells
                # A span=S cell at (ci,cj) covers old positions ci..ci+S-1, cj..cj+S-1
                # In the new grid, that maps to ci*mult .. (ci+S)*mult - 1
                total_sub = old_span * mult
                base_i, base_j = ci * mult, cj * mult
                for si in range(total_sub):
                    for sj in range(total_sub):
                        ni, nj = base_i + si, base_j + sj
                        if 0 <= ni < new_gs and 0 <= nj < new_gs:
                            needs_generation.add((ni, nj))
                            new_cells[ni][nj] = {
                                "row": ni, "col": nj,
                                "alpha": float(new_alphas[ni]),
                                "beta": float(new_betas[nj]),
                                "status": "pending",
                                "sensitivity": None, "cluster": None,
                                "thumbnail_url": None, "hq_url": None,
                                "span": 1,
                            }
            else:
                new_span = old_span * mult
                tl_i, tl_j = ci * mult, cj * mult
                new_span = min(new_span, new_gs - tl_i, new_gs - tl_j)

                if 0 <= tl_i < new_gs and 0 <= tl_j < new_gs and new_span > 0:
                    for di in range(new_span):
                        for dj in range(new_span):
                            ei, ej = tl_i + di, tl_j + dj
                            if 0 <= ei < new_gs and 0 <= ej < new_gs:
                                new_embeddings[ei, ej] = old["embeddings"][ci, cj]

                    thumb_bytes = old["thumbnails"].get((ci, cj))
                    if thumb_bytes:
                        new_thumbnails[(tl_i, tl_j)] = thumb_bytes
                        h = old["thumbnail_hashes"].get((ci, cj), "")
                        new_thumbnail_hashes[(tl_i, tl_j)] = h
                        thumb_url = cache.url(h) if h else None
                    else:
                        thumb_url = None

                    new_cells[tl_i][tl_j] = {
                        "row": tl_i, "col": tl_j,
                        "alpha": float(new_alphas[tl_i]),
                        "beta": float(new_betas[tl_j]),
                        "status": "generated",
                        "sensitivity": None, "cluster": None,
                        "thumbnail_url": thumb_url, "hq_url": None,
                        "span": new_span,
                    }

    if len(needs_generation) == 0:
        return None

    pre_filled = sum(1 for row in new_cells for c in row if c is not None and c["status"] == "generated")

    new_job = {
        "phase": "generating", "status": "running",
        "grid_size": new_gs,
        "total_cells": pre_filled + len(needs_generation),
        "cells_generated": pre_filled,
        "prompt_a": old["prompt_a"], "prompt_b": old["prompt_b"],
        "prompt_c": old.get("prompt_c", ""),
        "seed": old["seed"],
        "height": old.get("height", config.DEFAULT_HEIGHT),
        "width": old.get("width", config.DEFAULT_WIDTH),
        "steps": old.get("steps", config.DEFAULT_NUM_INFERENCE_STEPS),
        "guidance_scale": old.get("guidance_scale", config.DEFAULT_GUIDANCE_SCALE),
        "alphas": new_alphas, "betas": new_betas,
        "embeddings": new_embeddings,
        "sensitivity": None, "clusters": None,
        "thumbnails": new_thumbnails, "thumbnail_hashes": new_thumbnail_hashes,
        "cells": new_cells,
        "heatmap_path": None, "overlay_path": None,
        "cluster_path": None, "image_grid_path": None,
        "render_version": old.get("render_version", 0) + 1,
    }
    # Propagate job type (e.g. fast_scan) so analysis phase knows the context
    if old.get("type"):
        new_job["type"] = old["type"]
    # Refine always runs full analysis (unlike initial generate-selected)
    new_job["_run_analysis"] = True
    jobs[sub_id] = new_job

    # Submit GPU tasks
    active_frozen = frozenset(needs_generation)
    rows_needing = sorted(set(ni for ni, nj in needs_generation))
    chunks = [rows_needing[i::pool.n_gpus] for i in range(pool.n_gpus)]
    for chunk in chunks:
        if chunk:
            pool.submit(GenerateTask(
                job_id=sub_id, row_indices=chunk,
                alphas=new_alphas, betas=new_betas,
                prompt_a=old["prompt_a"], prompt_b=old["prompt_b"],
                prompt_c=old.get("prompt_c", ""),
                grid_size=new_gs, seed=old["seed"],
                height=old.get("height", config.DEFAULT_HEIGHT),
                width=old.get("width", config.DEFAULT_WIDTH),
                steps=old.get("steps", config.DEFAULT_NUM_INFERENCE_STEPS),
                guidance_scale=old.get("guidance_scale", config.DEFAULT_GUIDANCE_SCALE),
                active_cells=active_frozen,
            ))

    return needs_generation, new_gs


async def _refine_3d(job_id: str, req: RefineRequest, request: Request):
    """3D refine: expand grid, only generate new cells where sensitivity >= tau threshold.
    Non-selected cells get their embeddings copied from the nearest old cell."""
    jobs = request.app.state.jobs
    master = jobs.get(job_id)
    pool = request.app.state.gpu_pool
    cache = request.app.state.cache
    mult = req.multiplier
    old_gs = master["grid_size"]
    new_gs = old_gs * mult
    sensitivity = master["sensitivity"]

    # Compute threshold from 3D sensitivity
    real_sens = sensitivity[sensitivity > 0]
    if len(real_sens) == 0:
        return RefineResponse(refine_job_id=job_id, parent_job_id=job_id,
                              total_cells=0, status="no_cells")
    median = float(np.median(real_sens))
    threshold = median * req.tau

    old_alphas = master["alphas"]
    old_betas = master["betas"]
    old_gammas = master["gammas"]

    # New coordinates
    da = old_alphas[1] - old_alphas[0] if old_gs > 1 else 1.0
    new_alphas = np.zeros(new_gs)
    new_betas = np.zeros(new_gs)
    new_gammas = np.zeros(new_gs)
    for i in range(old_gs):
        for si in range(mult):
            new_alphas[i * mult + si] = old_alphas[i] - da / 2 + da * (si + 0.5) / mult
            new_betas[i * mult + si] = old_betas[i] - da / 2 + da * (si + 0.5) / mult
            new_gammas[i * mult + si] = old_gammas[i] - da / 2 + da * (si + 0.5) / mult
    new_alphas = np.clip(new_alphas, 0, 1)
    new_betas = np.clip(new_betas, 0, 1)
    new_gammas = np.clip(new_gammas, 0, 1)

    # Determine which old cells are above threshold → need generation
    needs_generation = set()  # (ni, nj, nk) in new grid
    new_embeddings = np.zeros((new_gs, new_gs, new_gs, 768))

    for i in range(old_gs):
        for j in range(old_gs):
            for k in range(old_gs):
                above = sensitivity[i, j, k] >= threshold
                for si in range(mult):
                    for sj in range(mult):
                        for sk in range(mult):
                            ni, nj, nk = i*mult+si, j*mult+sj, k*mult+sk
                            if above:
                                needs_generation.add((ni, nj, nk))
                            else:
                                # Copy parent embedding
                                new_embeddings[ni, nj, nk] = master["embeddings"][i, j, k]

    cells_to_gen = len(needs_generation)
    if cells_to_gen == 0:
        return RefineResponse(refine_job_id=job_id, parent_job_id=job_id,
                              total_cells=0, status="no_cells")

    # Build new 3D cell array
    new_cells = [[[None for _ in range(new_gs)] for _ in range(new_gs)] for _ in range(new_gs)]
    pre_filled = 0

    for i in range(new_gs):
        for j in range(new_gs):
            for k in range(new_gs):
                if (i, j, k) in needs_generation:
                    new_cells[i][j][k] = {
                        "row": i, "col": j, "depth": k,
                        "alpha": float(new_alphas[i]), "beta": float(new_betas[j]),
                        "gamma": float(new_gammas[k]),
                        "status": "pending",
                        "sensitivity": None, "cluster": None,
                        "thumbnail_url": None, "hq_url": None, "span": 1,
                    }
                else:
                    # Copy from nearest parent
                    oi, oj, ok = i // mult, j // mult, k // mult
                    parent_key = (oi, oj, ok) if (oi, oj, ok) in master["thumbnails"] else None
                    thumb_url = None
                    if parent_key and parent_key in master["thumbnail_hashes"]:
                        h = master["thumbnail_hashes"][parent_key]
                        thumb_url = cache.url(h)

                    new_cells[i][j][k] = {
                        "row": i, "col": j, "depth": k,
                        "alpha": float(new_alphas[i]), "beta": float(new_betas[j]),
                        "gamma": float(new_gammas[k]),
                        "status": "generated",
                        "sensitivity": None, "cluster": None,
                        "thumbnail_url": thumb_url, "hq_url": None, "span": 1,
                    }
                    pre_filled += 1

    total = pre_filled + cells_to_gen

    jobs[job_id] = {
        "phase": "generating", "status": "running",
        "dimensions": 3,
        "grid_size": new_gs,
        "total_cells": total,
        "cells_generated": pre_filled,
        "prompt_a": master["prompt_a"], "prompt_b": master["prompt_b"],
        "prompt_c": master.get("prompt_c", ""), "prompt_d": master.get("prompt_d", ""),
        "seed": master["seed"],
        "height": master.get("height", 256), "width": master.get("width", 256),
        "steps": master.get("steps", 4), "guidance_scale": master.get("guidance_scale", 4.0),
        "use_slerp": master.get("use_slerp", True),
        "alphas": new_alphas, "betas": new_betas, "gammas": new_gammas,
        "embeddings": new_embeddings,
        "sensitivity": None, "clusters": None,
        "thumbnails": {}, "thumbnail_hashes": {},
        "cells": new_cells,
        "heatmap_path": None, "overlay_path": None,
        "cluster_path": None, "image_grid_path": None,
        "ridge_mesh_path": None,
        "render_version": master.get("render_version", 0) + 1,
    }

    # Copy parent thumbnails for pre-filled cells
    for i in range(new_gs):
        for j in range(new_gs):
            for k in range(new_gs):
                if (i, j, k) not in needs_generation:
                    oi, oj, ok = i // mult, j // mult, k // mult
                    if (oi, oj, ok) in master["thumbnails"]:
                        jobs[job_id]["thumbnails"][(i, j, k)] = master["thumbnails"][(oi, oj, ok)]
                        jobs[job_id]["thumbnail_hashes"][(i, j, k)] = master["thumbnail_hashes"].get((oi, oj, ok), "")

    # Submit GPU tasks only for cells above threshold
    active_frozen = frozenset(needs_generation)
    rows_needing = sorted(set(ni for ni, nj, nk in needs_generation))
    chunks = [rows_needing[i::pool.n_gpus] for i in range(pool.n_gpus)]

    for chunk in chunks:
        if chunk:
            pool.submit(GenerateTask(
                job_id=job_id, row_indices=chunk,
                alphas=new_alphas, betas=new_betas,
                prompt_a=master["prompt_a"], prompt_b=master["prompt_b"],
                prompt_c=master.get("prompt_c", ""),
                grid_size=new_gs, seed=master["seed"],
                height=master.get("height", 256), width=master.get("width", 256),
                steps=master.get("steps", 4), guidance_scale=master.get("guidance_scale", 4.0),
                prompt_d=master.get("prompt_d", ""), gammas=new_gammas,
                grid_size_z=new_gs, use_slerp=master.get("use_slerp", True),
                active_cells=active_frozen,
            ))

    return RefineResponse(
        refine_job_id=job_id, parent_job_id=job_id,
        total_cells=cells_to_gen, status="running",
    )


@router.post("/fast-scan", response_model=FastScanResponse)
async def fast_scan(req: FastScanRequest, request: Request):
    """Fast ridge detection: 1-step latents + Jacobian spectral norm.

    Supports both 2D (3 prompts) and 3D (4 prompts) grids.
    ~7x faster than full DINOv2 exploration.
    """
    request.app.state.gpu_pool.clear_cancel()  # Allow workers to process new tasks
    job_id = str(uuid.uuid4())[:8]
    gs = req.grid_size
    alphas = np.linspace(config.ALPHA_RANGE[0], config.ALPHA_RANGE[1], gs)
    betas = np.linspace(config.BETA_RANGE[0], config.BETA_RANGE[1], gs)

    is_3d = req.dimensions == 3 and req.prompt_d
    gammas = np.linspace(config.ALPHA_RANGE[0], config.ALPHA_RANGE[1], gs) if is_3d else None

    jobs = request.app.state.jobs
    pool = request.app.state.gpu_pool

    total = gs * gs * gs if is_3d else gs * gs

    if is_3d:
        cells = [[[{
            "row": i, "col": j, "depth": k,
            "alpha": float(alphas[i]), "beta": float(betas[j]), "gamma": float(gammas[k]),
            "status": "scanning",
            "sensitivity": None, "cluster": None,
            "thumbnail_url": None, "hq_url": None, "span": 1,
        } for k in range(gs)] for j in range(gs)] for i in range(gs)]
    else:
        cells = [[{
            "row": i, "col": j,
            "alpha": float(alphas[i]), "beta": float(betas[j]),
            "status": "scanning",
            "sensitivity": None, "cluster": None,
            "thumbnail_url": None, "hq_url": None, "span": 1,
        } for j in range(gs)] for i in range(gs)]

    jobs[job_id] = {
        "type": "fast_scan",
        "phase": "scanning",
        "status": "running",
        "dimensions": 3 if is_3d else 2,
        "grid_size": gs,
        "grid_size_b": gs,
        "grid_size_z": gs if is_3d else 0,
        "total_cells": total,
        "cells_generated": 0,
        "prompt_a": req.prompt_a,
        "prompt_b": req.prompt_b,
        "prompt_c": req.prompt_c,
        "prompt_d": req.prompt_d if is_3d else "",
        "seed": req.seed,
        "seeds": [req.seed],
        "sub_job_ids": [],
        "height": req.height, "width": req.width,
        "guidance_scale": req.guidance_scale,
        "alphas": alphas, "betas": betas,
        "gammas": gammas,
        "latents": None,
        "anisotropy": None,
        "sensitivity": None,
        "clusters": None,
        "embeddings": np.zeros((gs, gs, gs, 768)) if is_3d else np.zeros((gs, gs, 768)),
        "thumbnails": {}, "thumbnail_hashes": {},
        "cells": cells,
        "heatmap_path": None, "overlay_path": None,
        "cluster_path": None, "image_grid_path": None,
        "ridge_mesh_path": None,
        "render_version": 0,
    }

    # Distribute rows across GPUs
    chunks = [list(range(gs))[i::pool.n_gpus] for i in range(pool.n_gpus)]
    for chunk in chunks:
        if chunk:
            pool.submit(FastScanTask(
                job_id=job_id, row_indices=chunk,
                alphas=alphas, betas=betas,
                prompt_a=req.prompt_a, prompt_b=req.prompt_b, prompt_c=req.prompt_c,
                grid_size=gs, seed=req.seed,
                height=req.height, width=req.width,
                guidance_scale=req.guidance_scale,
                prompt_d=req.prompt_d if is_3d else "",
                gammas=gammas,
                grid_size_z=gs if is_3d else 0,
            ))

    return FastScanResponse(job_id=job_id, total_cells=total, status="running")


@router.post("/{job_id}/generate-selected", response_model=GenerateSelectedResponse)
async def generate_selected(job_id: str, req: GenerateSelectedRequest, request: Request):
    """Generate full images for cells above tau threshold after a fast scan.
    Supports both 2D and 3D grids."""
    jobs = request.app.state.jobs
    job = jobs.get(job_id)
    if not job or job.get("type") != "fast_scan" or job["phase"] != "scan_complete":
        return GenerateSelectedResponse(status="error", total_cells=0)

    pool = request.app.state.gpu_pool
    gs = job["grid_size"]
    sensitivity = job["sensitivity"]
    is_3d = job.get("dimensions", 2) == 3

    # Compute threshold
    real_sens = sensitivity[sensitivity > 0]
    if len(real_sens) == 0:
        return GenerateSelectedResponse(status="no_cells", total_cells=0)
    median = float(np.median(real_sens))
    threshold = median * req.tau

    if is_3d:
        gs_z = job.get("grid_size_z", gs)
        gammas = job.get("gammas")

        # Find 3D cells above threshold
        active_cells = set()
        for i in range(gs):
            for j in range(gs):
                for k in range(gs_z):
                    if sensitivity[i, j, k] >= threshold:
                        active_cells.add((i, j, k))

        if not active_cells:
            return GenerateSelectedResponse(status="no_cells", total_cells=0)

        # Transition job
        job["phase"] = "generating"
        job["status"] = "running"
        job["total_cells"] = len(active_cells)
        job["cells_generated"] = 0
        job["height"] = req.height
        job["width"] = req.width
        job["steps"] = req.steps
        job["guidance_scale"] = req.guidance_scale
        job["embeddings"] = np.zeros((gs, gs, gs_z, 768))
        job["render_version"] = job.get("render_version", 0) + 1

        for i in range(gs):
            for j in range(gs):
                for k in range(gs_z):
                    if (i, j, k) in active_cells:
                        job["cells"][i][j][k]["status"] = "pending"
                    else:
                        job["cells"][i][j][k]["status"] = "skipped"

        # Submit GPU tasks
        active_frozen = frozenset(active_cells)
        rows_needing = sorted(set(i for i, j, k in active_cells))
        chunks = [rows_needing[i::pool.n_gpus] for i in range(pool.n_gpus)]

        for chunk in chunks:
            if chunk:
                pool.submit(GenerateTask(
                    job_id=job_id, row_indices=chunk,
                    alphas=job["alphas"], betas=job["betas"],
                    prompt_a=job["prompt_a"], prompt_b=job["prompt_b"],
                    prompt_c=job.get("prompt_c", ""),
                    grid_size=gs, seed=job["seed"],
                    height=req.height, width=req.width,
                    steps=req.steps, guidance_scale=req.guidance_scale,
                    active_cells=active_frozen,
                    prompt_d=job.get("prompt_d", ""),
                    gammas=gammas,
                    grid_size_z=gs_z,
                    use_slerp=job.get("use_slerp", True),
                ))

    else:
        # 2D path
        active_cells = set()
        for i in range(gs):
            for j in range(gs):
                if sensitivity[i, j] >= threshold:
                    active_cells.add((i, j))

        if not active_cells:
            return GenerateSelectedResponse(status="no_cells", total_cells=0)

        job["phase"] = "generating"
        job["status"] = "running"
        job["total_cells"] = len(active_cells)
        job["cells_generated"] = 0
        job["height"] = req.height
        job["width"] = req.width
        job["steps"] = req.steps
        job["guidance_scale"] = req.guidance_scale
        job["embeddings"] = np.zeros((gs, gs, 768))
        job["render_version"] = job.get("render_version", 0) + 1

        for i in range(gs):
            for j in range(gs):
                if (i, j) in active_cells:
                    job["cells"][i][j]["status"] = "pending"
                else:
                    job["cells"][i][j]["status"] = "skipped"

        active_frozen = frozenset(active_cells)
        rows_needing = sorted(set(i for i, j in active_cells))
        chunks = [rows_needing[i::pool.n_gpus] for i in range(pool.n_gpus)]

        for chunk in chunks:
            if chunk:
                pool.submit(GenerateTask(
                    job_id=job_id, row_indices=chunk,
                    alphas=job["alphas"], betas=job["betas"],
                    prompt_a=job["prompt_a"], prompt_b=job["prompt_b"],
                    prompt_c=job.get("prompt_c", ""),
                    grid_size=gs, seed=job["seed"],
                    height=req.height, width=req.width,
                    steps=req.steps, guidance_scale=req.guidance_scale,
                    active_cells=active_frozen,
                ))

    return GenerateSelectedResponse(status="running", total_cells=len(active_cells))


@router.post("/{job_id}/refine", response_model=RefineResponse)
async def refine_grid(job_id: str, req: RefineRequest, request: Request):
    """Refine grid in-place. For multi-seed, refines all sub-jobs."""
    jobs = request.app.state.jobs
    master = jobs.get(job_id)
    if not master or master["phase"] != "complete":
        return RefineResponse(refine_job_id=job_id, parent_job_id=job_id,
                              total_cells=0, status="error")

    # 3D refine: regenerate at higher resolution (no span system)
    if master.get("dimensions", 2) == 3:
        return await _refine_3d(job_id, req, request)

    sensitivity = master["sensitivity"]
    if sensitivity is None:
        return RefineResponse(refine_job_id=job_id, parent_job_id=job_id,
                              total_cells=0, status="error")

    # Compute threshold from master (averaged) sensitivity
    real_sensitivities = []
    for row in master["cells"]:
        for cell in row:
            if cell is not None and cell.get("span", 1) == 1 and cell.get("sensitivity") is not None:
                real_sensitivities.append(cell["sensitivity"])
    if not real_sensitivities:
        real_sensitivities = [s for s in sensitivity.ravel() if s > 0]
    if not real_sensitivities:
        return RefineResponse(refine_job_id=job_id, parent_job_id=job_id,
                              total_cells=0, status="no_cells")

    median = float(np.median(real_sensitivities))
    threshold = median * req.tau
    mult = req.multiplier
    cache = request.app.state.cache
    pool = request.app.state.gpu_pool

    # Compute which positions to refine from tau threshold + manual selection
    refine_positions = set()
    for row in master["cells"]:
        for cell in row:
            if cell is not None and cell.get("span", 1) == 1:
                sens = cell.get("sensitivity")
                if sens is not None and sens >= threshold:
                    refine_positions.add((cell["row"], cell["col"]))

    # Add manually selected positions (any span — will be subdivided)
    all_cell_positions = set()
    for row in master["cells"]:
        for cell in row:
            if cell is not None:
                all_cell_positions.add((cell["row"], cell["col"]))

    for pos in req.extra_positions:
        p = (pos[0], pos[1])
        if p in all_cell_positions:
            refine_positions.add(p)

    if not refine_positions:
        return RefineResponse(refine_job_id=job_id, parent_job_id=job_id,
                              total_cells=0, status="no_cells")

    sub_ids = master.get("sub_job_ids", [])
    seeds = master.get("seeds", [master.get("seed", 42)])
    is_multi = len(seeds) > 1 and len(sub_ids) > 1

    total_new = 0

    if is_multi:
        # Refine each sub-job with the same threshold
        new_gs = None
        for sid in sub_ids:
            result = _refine_single_job(sid, jobs, refine_positions, mult, cache, pool)
            if result:
                needs, new_gs = result
                total_new += len(needs)

        if total_new == 0:
            return RefineResponse(refine_job_id=job_id, parent_job_id=job_id,
                                  total_cells=0, status="no_cells")

        # Update master to reflect new grid size, reset sensitivity
        if new_gs:
            old_alphas = master["alphas"]
            old_betas = master["betas"]
            old_gs = master["grid_size"]
            da = old_alphas[1] - old_alphas[0] if old_gs > 1 else 1.0
            db = old_betas[1] - old_betas[0] if old_gs > 1 else 1.0
            new_alphas = np.zeros(new_gs)
            new_betas = np.zeros(new_gs)
            for i in range(old_gs):
                for si in range(mult):
                    new_alphas[i * mult + si] = old_alphas[i] - da / 2 + da * (si + 0.5) / mult
            for j in range(old_gs):
                for sj in range(mult):
                    new_betas[j * mult + sj] = old_betas[j] - db / 2 + db * (sj + 0.5) / mult

            master["phase"] = "generating"
            master["status"] = "running"
            master["grid_size"] = new_gs
            master["alphas"] = np.clip(new_alphas, 0, 1)
            master["betas"] = np.clip(new_betas, 0, 1)
            master["sensitivity"] = None
            master["embeddings"] = np.zeros((new_gs, new_gs, 768))
            master["total_cells"] = total_new * len(seeds)
            master["cells_generated"] = 0
            master["render_version"] = master.get("render_version", 0) + 1
    else:
        # Single seed: refine the master directly
        result = _refine_single_job(job_id, jobs, refine_positions, mult, cache, pool)
        if not result:
            return RefineResponse(refine_job_id=job_id, parent_job_id=job_id,
                                  total_cells=0, status="no_cells")
        needs, new_gs = result
        total_new = len(needs)
        # Carry over seeds/sub_job_ids
        jobs[job_id]["seeds"] = seeds
        jobs[job_id]["sub_job_ids"] = sub_ids

    return RefineResponse(
        refine_job_id=job_id, parent_job_id=job_id,
        total_cells=total_new, status="running",
    )


@router.get("/{job_id}/status", response_model=GridStatusResponse)
async def grid_status(job_id: str, request: Request):
    jobs = request.app.state.jobs
    job = jobs.get(job_id)
    if not job:
        return GridStatusResponse(
            job_id=job_id, status="not_found", phase="unknown",
            grid_size=0, cells_generated=0, cells_total=0, cells=[],
            prompt_a="", prompt_b="", prompt_c="",
        )

    seeds = job.get("seeds", [job.get("seed", 42)])
    sub_ids = job.get("sub_job_ids", [])
    is_multi = len(seeds) > 1 and len(sub_ids) > 1

    if is_multi:
        # Aggregate multi-seed status
        all_gen = 0
        all_total = 0
        all_complete = True
        any_analyzing = False
        seed_cell_map = {}

        for sid in sub_ids:
            sub = jobs.get(sid)
            if not sub:
                continue
            all_gen += sub["cells_generated"]
            all_total += sub["total_cells"]
            if sub["phase"] != "complete":
                all_complete = False
            if sub["phase"] == "analyzing":
                any_analyzing = True

            # Collect per-seed cells
            seed_num = sub["seed"]
            sub_cells = []
            for row in sub["cells"]:
                for cell in row:
                    if cell is not None:
                        sub_cells.append(CellStatus(**cell))
            seed_cell_map[str(seed_num)] = sub_cells

        # Use first seed's cells as the "primary" cells (for grid display)
        first_sub = jobs.get(sub_ids[0], {})
        flat_cells = []
        for row in first_sub.get("cells", []):
            for cell in row:
                if cell is not None:
                    flat_cells.append(CellStatus(**cell))

        # If all subs complete, compute averaged sensitivity on the master
        phase = "complete" if all_complete else ("analyzing" if any_analyzing else "generating")
        if all_complete and job.get("sensitivity") is None:
            _compute_averaged_sensitivity(job, jobs, sub_ids)

        # Override primary cells' sensitivity with averaged values
        if job.get("sensitivity") is not None:
            gs = job["grid_size"]
            sens = job["sensitivity"]
            for c in flat_cells:
                if c.row < gs and c.col < gs:
                    c.sensitivity = float(sens[c.row, c.col])

        v = job.get("render_version", 0)
        return GridStatusResponse(
            job_id=job_id, status="running" if not all_complete else "complete",
            phase=phase, grid_size=job["grid_size"],
            cells_generated=all_gen, cells_total=all_total,
            cells=flat_cells, seed_cells=seed_cell_map, seeds=seeds,
            prompt_a=job["prompt_a"], prompt_b=job["prompt_b"],
            prompt_c=job.get("prompt_c", ""),
            heatmap_url=f"/api/grid/{job_id}/heatmap.png?v={v}" if job.get("heatmap_path") else None,
            overlay_url=f"/api/grid/{job_id}/overlay.png?v={v}" if job.get("overlay_path") else None,
            cluster_url=f"/api/grid/{job_id}/clusters.png?v={v}" if job.get("cluster_path") else None,
            image_grid_url=f"/api/grid/{job_id}/images.png?v={v}" if job.get("image_grid_path") else None,
        )

    # Single seed
    flat_cells = _flatten_cells(job)

    v = job.get("render_version", 0)
    return GridStatusResponse(
        job_id=job_id,
        status=job["status"],
        phase=job["phase"],
        dimensions=job.get("dimensions", 2),
        grid_size=job["grid_size"],
        cells_generated=job["cells_generated"],
        cells_total=job["total_cells"],
        cells=flat_cells,
        seeds=seeds,
        prompt_a=job["prompt_a"],
        prompt_b=job["prompt_b"],
        prompt_c=job.get("prompt_c", ""),
        prompt_d=job.get("prompt_d", ""),
        heatmap_url=f"/api/grid/{job_id}/heatmap.png?v={v}" if job.get("heatmap_path") else None,
        overlay_url=f"/api/grid/{job_id}/overlay.png?v={v}" if job.get("overlay_path") else None,
        cluster_url=f"/api/grid/{job_id}/clusters.png?v={v}" if job.get("cluster_path") else None,
        image_grid_url=f"/api/grid/{job_id}/images.png?v={v}" if job.get("image_grid_path") else None,
        ridge_mesh_url=f"/api/grid/{job_id}/ridge_mesh.json?v={v}" if job.get("ridge_mesh_path") else None,
    )


@router.get("/{job_id}/ridge_mesh.json")
async def get_ridge_mesh(job_id: str, request: Request):
    job = request.app.state.jobs.get(job_id)
    if not job or not job.get("ridge_mesh_path"):
        return {"error": "not ready"}
    return FileResponse(job["ridge_mesh_path"], media_type="application/json")


def _compute_averaged_sensitivity(master, jobs, sub_ids):
    """Average sensitivity across all sub-jobs into the master."""
    from backend.services.ridge_detector import compute_sensitivity, compute_clusters

    gs = master["grid_size"]
    all_sens = []
    for sid in sub_ids:
        sub = jobs.get(sid)
        if sub and sub.get("sensitivity") is not None:
            all_sens.append(sub["sensitivity"])

    if not all_sens:
        return

    avg_sens = np.mean(all_sens, axis=0)
    master["sensitivity"] = avg_sens
    master["phase"] = "complete"
    master["status"] = "complete"

    # Use first sub's embeddings for clustering
    first_sub = jobs.get(sub_ids[0])
    if first_sub:
        master["embeddings"] = first_sub["embeddings"]
        master["clusters"] = compute_clusters(first_sub["embeddings"])
        master["thumbnails"] = first_sub["thumbnails"]
        master["thumbnail_hashes"] = first_sub["thumbnail_hashes"]
        master["cells"] = first_sub["cells"]

        # Update cells with averaged sensitivity
        for i in range(gs):
            for j in range(gs):
                if master["cells"][i][j] is not None:
                    master["cells"][i][j]["sensitivity"] = float(avg_sens[i, j])


@router.get("/{job_id}/heatmap.png")
async def get_heatmap(job_id: str, request: Request):
    job = request.app.state.jobs.get(job_id)
    if not job or not job.get("heatmap_path"):
        return {"error": "not ready"}
    return FileResponse(job["heatmap_path"], media_type="image/png")


@router.get("/{job_id}/overlay.png")
async def get_overlay(job_id: str, request: Request):
    job = request.app.state.jobs.get(job_id)
    if not job or not job.get("overlay_path"):
        return {"error": "not ready"}
    return FileResponse(job["overlay_path"], media_type="image/png")


@router.get("/{job_id}/clusters.png")
async def get_clusters(job_id: str, request: Request):
    job = request.app.state.jobs.get(job_id)
    if not job or not job.get("cluster_path"):
        return {"error": "not ready"}
    return FileResponse(job["cluster_path"], media_type="image/png")


@router.get("/{job_id}/images.png")
async def get_image_grid(job_id: str, request: Request):
    job = request.app.state.jobs.get(job_id)
    if not job or not job.get("image_grid_path"):
        return {"error": "not ready"}
    return FileResponse(job["image_grid_path"], media_type="image/png")


@router.get("/{job_id}/export/{layer}.jpg")
async def export_grid(job_id: str, layer: str, request: Request):
    """Export grid as full-resolution PNG. Layers: images, heatmap, overlay."""
    from io import BytesIO
    from PIL import Image
    from fastapi.responses import StreamingResponse

    jobs = request.app.state.jobs
    job = jobs.get(job_id)
    if not job:
        return {"error": "not found"}

    gs = job["grid_size"]
    thumbnails = job.get("thumbnails", {})
    sensitivity = job.get("sensitivity")

    # Determine tile size from first thumbnail
    tile_size = 256
    for key, thumb_bytes in thumbnails.items():
        img = Image.open(BytesIO(thumb_bytes))
        tile_size = img.size[0]
        break

    if layer == "heatmap":
        tile_size = 8  # small tiles for heatmap-only (fast)
    elif layer not in ("images", "overlay"):
        return {"error": f"unknown layer: {layer}"}

    canvas_w = gs * tile_size
    canvas_h = gs * tile_size

    import numpy as np_export
    canvas = np_export.zeros((canvas_h, canvas_w, 3), dtype=np_export.uint8)
    canvas[:] = [10, 10, 18]  # dark background

    if layer in ("images", "overlay"):
        # Use cell data for proper span handling
        cells = job.get("cells", [])
        is_3d = job.get("dimensions", 2) == 3
        cell_list = []
        if not is_3d:
            for row in cells:
                for cell in row:
                    if cell is not None:
                        cell_list.append(cell)

        for cell in cell_list:
            alpha_idx = cell["row"]
            beta_idx = cell["col"]
            span = cell.get("span", 1)
            key = (alpha_idx, beta_idx)
            thumb_bytes = thumbnails.get(key)
            if not thumb_bytes:
                continue
            try:
                img = Image.open(BytesIO(thumb_bytes)).convert("RGB")
                # Scale to span × tile_size
                target = span * tile_size
                if img.size != (target, target):
                    img = img.resize((target, target), Image.LANCZOS)
                arr = np_export.array(img)
                col = alpha_idx
                row_top = gs - 1 - beta_idx - (span - 1)
                if 0 <= row_top and row_top + span <= gs and 0 <= col and col + span <= gs:
                    canvas[row_top * tile_size:(row_top + span) * tile_size,
                           col * tile_size:(col + span) * tile_size] = arr
            except Exception:
                continue

    if layer in ("heatmap", "overlay") and sensitivity is not None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.cm as cm

        sens = sensitivity
        s_min = float(np_export.nanmin(sens[sens > 0])) if (sens > 0).any() else 0
        s_max = float(np_export.nanmax(sens)) if not np_export.all(np_export.isnan(sens)) else 1

        for i in range(gs):
            gs_b = sens.shape[1] if len(sens.shape) > 1 else gs
            for j in range(gs_b):
                val = float(sens[i, j]) if not np_export.isnan(sens[i, j]) else 0
                if val <= 0:
                    continue
                t = (val - s_min) / (s_max - s_min + 1e-10)
                r = int(255 * min(1, t * 2))
                g = int(255 * max(0, t - 0.5) * 2)
                row = gs - 1 - j
                col = i
                if 0 <= row < gs and 0 <= col < gs:
                    y0, y1 = row * tile_size, (row + 1) * tile_size
                    x0, x1 = col * tile_size, (col + 1) * tile_size
                    if layer == "heatmap":
                        canvas[y0:y1, x0:x1] = [r, g, 0]
                    else:  # overlay
                        alpha = 0.5
                        canvas[y0:y1, x0:x1] = (
                            canvas[y0:y1, x0:x1].astype(float) * (1 - alpha) +
                            np_export.array([r, g, 0], dtype=float) * alpha
                        ).astype(np_export.uint8)

    # Encode as PNG
    img_out = Image.fromarray(canvas)
    buf = BytesIO()
    img_out.save(buf, format="JPEG", quality=92)
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg",
                             headers={"Content-Disposition": f'attachment; filename="ridge_{layer}_{gs}x{gs}.jpg"'})


@router.post("/{job_id}/seed-probe", response_model=SeedProbeResponse)
async def seed_probe(job_id: str, req: SeedProbeRequest, request: Request):
    """Generate a single (alpha, beta) point across many seeds.

    Drains pending generation tasks so the probe runs immediately.
    """
    pool = request.app.state.gpu_pool
    # Preempt: drain pending tasks so probe runs next (after in-flight tasks finish)
    drained = pool.drain_pending()
    pool.clear_cancel()  # Allow workers to process the probe tasks
    if drained:
        print(f"Seed probe preempted {drained} queued generation tasks", flush=True)

    jobs = request.app.state.jobs
    master = jobs.get(job_id)
    if not master:
        return SeedProbeResponse(probe_id="", total=0, status="error")

    probe_id = f"probe_{uuid.uuid4().hex[:6]}"
    seeds = list(range(req.seed_start, req.seed_end + 1))
    n = len(seeds)

    alphas = np.array([req.alpha])
    betas = np.array([req.beta])

    # Use requested steps, or fall back to parent job's steps
    probe_steps = req.steps if req.steps > 0 else master.get("steps", 4)

    # Store probe job
    jobs[probe_id] = {
        "type": "probe",
        "alpha": req.alpha,
        "beta": req.beta,
        "seeds": seeds,
        "images": [None] * n,  # thumbnail URLs
        "thumbnails": {},  # seed -> bytes
        "thumbnail_hashes": {},
        "complete_count": 0,
        "total": n,
        "prompt_a": master["prompt_a"],
        "prompt_b": master["prompt_b"],
        "prompt_c": master.get("prompt_c", ""),
    }

    pool = request.app.state.gpu_pool
    cache = request.app.state.cache

    # Submit one GenerateTask per seed (each is a 1×1 grid)
    for seed in seeds:
        sub_id = f"{probe_id}_s{seed}"
        jobs[sub_id] = {
            "phase": "generating", "status": "running",
            "grid_size": 1, "total_cells": 1, "cells_generated": 0,
            "prompt_a": master["prompt_a"], "prompt_b": master["prompt_b"],
            "prompt_c": master.get("prompt_c", ""), "seed": seed,
            "height": master.get("height", 256), "width": master.get("width", 256),
            "steps": probe_steps,
            "guidance_scale": master.get("guidance_scale", 4.0),
            "alphas": alphas, "betas": betas,
            "embeddings": np.zeros((1, 1, 768)),
            "sensitivity": None, "clusters": None,
            "thumbnails": {}, "thumbnail_hashes": {},
            "cells": [[{
                "row": 0, "col": 0,
                "alpha": req.alpha, "beta": req.beta,
                "status": "pending", "sensitivity": None, "cluster": None,
                "thumbnail_url": None, "hq_url": None, "span": 1,
            }]],
            "heatmap_path": None, "overlay_path": None,
            "cluster_path": None, "image_grid_path": None,
            "render_version": 0,
        }
        pool.submit(GenerateTask(
            job_id=sub_id, row_indices=[0],
            alphas=alphas, betas=betas,
            prompt_a=master["prompt_a"], prompt_b=master["prompt_b"],
            prompt_c=master.get("prompt_c", ""),
            grid_size=1, seed=seed,
            height=master.get("height", 256), width=master.get("width", 256),
            steps=probe_steps,
            guidance_scale=master.get("guidance_scale", 4.0),
        ))

    return SeedProbeResponse(probe_id=probe_id, total=n, status="running")


@router.get("/probe/{probe_id}/status", response_model=SeedProbeStatus)
async def probe_status(probe_id: str, request: Request):
    jobs = request.app.state.jobs
    probe = jobs.get(probe_id)
    if not probe or probe.get("type") != "probe":
        return SeedProbeStatus(probe_id=probe_id, alpha=0, beta=0,
                               seeds=[], images=[], complete=True)

    cache = request.app.state.cache
    seeds = probe["seeds"]
    images = []
    done = 0

    for i, seed in enumerate(seeds):
        sub_id = f"{probe_id}_s{seed}"
        sub = jobs.get(sub_id)
        if sub and sub["cells"][0][0].get("thumbnail_url"):
            images.append(sub["cells"][0][0]["thumbnail_url"])
            done += 1
        else:
            images.append(None)

    return SeedProbeStatus(
        probe_id=probe_id,
        alpha=probe["alpha"], beta=probe["beta"],
        seeds=seeds, images=images,
        complete=(done >= len(seeds)),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MF-Scan: Multi-fidelity Jacobian + GP + targeted DINOv2
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/mf-scan", response_model=MFScanResponse)
async def mf_scan(req: MFScanRequest, request: Request):
    """Multi-fidelity ridge detection.

    Phase 1: Fast Jacobian sweep (all grid points, 1-step latents)
    Phase 2: GP-guided adaptive DINOv2 evaluation (budget points only)
    Phase 3: Final GP-predicted sensitivity map

    ~2.5x faster than full DINOv2 at 50x50 with F1≈0.85 ridge detection.
    """
    request.app.state.gpu_pool.clear_cancel()
    # Step 1: Start a fast scan (reuse existing infrastructure)
    job_id = str(uuid.uuid4())[:8]
    gs = req.grid_size
    alphas = np.linspace(config.ALPHA_RANGE[0], config.ALPHA_RANGE[1], gs)
    betas = np.linspace(config.BETA_RANGE[0], config.BETA_RANGE[1], gs)

    jobs = request.app.state.jobs
    pool = request.app.state.gpu_pool

    total = gs * gs
    cells = [[{
        "row": i, "col": j,
        "alpha": float(alphas[i]), "beta": float(betas[j]),
        "status": "scanning",
        "sensitivity": None, "cluster": None,
        "thumbnail_url": None, "hq_url": None, "span": 1,
    } for j in range(gs)] for i in range(gs)]

    jobs[job_id] = {
        "type": "mf_scan",
        "phase": "mf_scanning",
        "status": "running",
        "dimensions": 2,
        "grid_size": gs,
        "grid_size_b": gs,
        "total_cells": total,
        "cells_generated": 0,
        "prompt_a": req.prompt_a,
        "prompt_b": req.prompt_b,
        "prompt_c": req.prompt_c,
        "prompt_d": "",
        "seed": req.seed,
        "seeds": [req.seed],
        "sub_job_ids": [],
        "height": req.height, "width": req.width,
        "steps": req.steps,
        "guidance_scale": req.guidance_scale,
        "alphas": alphas, "betas": betas,
        "gammas": None,
        "latents": None,
        "anisotropy": None,
        "sensitivity": None,
        "clusters": None,
        "embeddings": np.zeros((gs, gs, 768)),
        "thumbnails": {}, "thumbnail_hashes": {},
        "cells": cells,
        "heatmap_path": None, "overlay_path": None,
        "cluster_path": None, "image_grid_path": None,
        "ridge_mesh_path": None,
        "render_version": 0,
        # MF-specific fields
        "mf_budget": req.budget,
        "mf_tau": req.tau_mf,
        "mf_detector": None,
        "mf_n_observed": 0,
        "mf_predicted_sensitivity": None,
    }

    # Submit fast scan tasks to GPUs
    chunks = [list(range(gs))[i::pool.n_gpus] for i in range(pool.n_gpus)]
    for chunk in chunks:
        if chunk:
            pool.submit(FastScanTask(
                job_id=job_id, row_indices=chunk,
                alphas=alphas, betas=betas,
                prompt_a=req.prompt_a, prompt_b=req.prompt_b, prompt_c=req.prompt_c,
                grid_size=gs, seed=req.seed,
                height=req.height, width=req.width,
                guidance_scale=1.0,  # no CFG for fast scan
                prompt_d="",
                gammas=None,
                grid_size_z=0,
            ))

    return MFScanResponse(
        job_id=job_id, total_cells=total,
        budget=req.budget, status="running"
    )
