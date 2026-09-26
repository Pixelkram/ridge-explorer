import time
import re
import uuid

import numpy as np
from fastapi import HTTPException
from fastapi import APIRouter, Request
from fastapi.responses import FileResponse

from backend.models import (
    JvpProbeRequest, ProbeStartResponse, ProbeStatusResponse, TokenProbeRequest,
    SurpriseSampleRequest, SurpriseSampleResponse,
    GridStartRequest, GridStartResponse, GridStatusResponse, CellStatus,
    RenderHQRequest, RefineRequest, RefineResponse,
    SeedProbeRequest, SeedProbeResponse, SeedProbeStatus,
    FastScanRequest, FastScanResponse,
    GenerateSelectedRequest, GenerateSelectedResponse,
    MFScanRequest, MFScanResponse,
    RidgeGraphRequest, RidgeGraphResponse,
    HikeBranchRequest, HikeStartRequest, HikeStartResponse, HikeStatus,
)
from backend.services.gpu_pool import GenerateTask, HQTask, FastScanTask, HikeTask, ProbeTask, TokenProbeTask
from backend.cache.thumbnail_cache import ThumbnailStore
from backend.services.ridge_detector import measured_mask
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
    # so they don't get attributed to future jobs. Results belonging to a live hike
    # must still reach its inbox -- the collector does this dispatch, and dropping it
    # here silently starved the hike's wait loop until its deadline.
    inbox = getattr(request.app.state, "hike_inbox", {})
    stale_results = []
    for r in pool.collect_results():
        if r.job_id in inbox:
            inbox[r.job_id].append(r)
        else:
            stale_results.append(r)
    # Clear all jobs so in-flight GPU results get discarded by the collector
    jobs = request.app.state.jobs
    n_cleared = len(jobs)
    jobs.clear()
    return {"status": "ok", "drained": drained, "stale_results": len(stale_results), "jobs_cleared": n_cleared}


@router.post("/start", response_model=GridStartResponse)
async def start_grid(req: GridStartRequest, request: Request):
    pool = request.app.state.gpu_pool
    master_id = str(uuid.uuid4())[:8]
    gs = req.grid_size
    alphas = np.linspace(config.ALPHA_RANGE[0], config.ALPHA_RANGE[1], gs)
    betas = np.linspace(config.BETA_RANGE[0], config.BETA_RANGE[1], gs)

    is_3d = req.dimensions == 3
    gammas = np.linspace(config.ALPHA_RANGE[0], config.ALPHA_RANGE[1], gs) if is_3d else None
    use_slerp = is_3d  # enable SLERP for 3D by default

    jobs = request.app.state.jobs
    pool = request.app.state.gpu_pool
    cache = request.app.state.cache
    _evict_old_jobs(jobs)

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
                                 gammas=gammas, use_slerp=use_slerp,
                                 cache=request.app.state.cache)
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
            "thumbnails": ThumbnailStore(cache), "thumbnail_hashes": {},
            "cells": jobs[sub_job_ids[0]]["cells"],  # use first sub's cell structure
            "heatmap_path": None, "overlay_path": None,
            "cluster_path": None, "image_grid_path": None,
            "ridge_mesh_path": None,
            "render_version": 0,
        }

    return GridStartResponse(job_id=master_id, total_cells=total_cells_all,
                             dimensions=req.dimensions, status="running")


def _make_job(job_id, gs, alphas, betas, prompt_a, prompt_b, prompt_c, seed,
              height=config.DEFAULT_HEIGHT, width=config.DEFAULT_WIDTH,
              steps=config.DEFAULT_NUM_INFERENCE_STEPS,
              guidance_scale=config.DEFAULT_GUIDANCE_SCALE,
              dimensions=2, prompt_d="", gammas=None, use_slerp=False, cache=None):
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
        # read-through: the bytes live on disk, only the hash stays resident
        "thumbnails": ThumbnailStore(cache),
        "thumbnail_hashes": {},
        "cells": cells,
        "heatmap_path": None, "overlay_path": None,
        "cluster_path": None, "image_grid_path": None,
        "ridge_mesh_path": None,
        "render_version": 0,
    }


_TERMINAL_PHASES = {"complete", "scan_complete", "error"}


def _evict_old_jobs(jobs: dict):
    """Bound app.state.jobs, which otherwise only ever grows.

    Called when a new job is registered. Oldest-first by insertion order, and only
    jobs that have finished — a live job's results are still being routed to it by the
    collector. Sub-jobs still referenced by a retained master are kept whatever their
    age, or grid_status would aggregate over holes.
    """
    if len(jobs) <= config.MAX_RESIDENT_JOBS:
        return
    protected = set()
    for jid in list(jobs)[-config.MAX_RESIDENT_JOBS:]:
        protected.add(jid)
        protected.update(jobs[jid].get("sub_job_ids") or [])
    for jid in list(jobs):
        if jid in protected:
            continue
        if jobs[jid].get("phase", "generating") in _TERMINAL_PHASES:
            del jobs[jid]
        if len(jobs) <= config.MAX_RESIDENT_JOBS:
            break


def _subdivided_axis(old_axis, new_n, full_range):
    """Coordinates for a refined axis: the old span resampled at `new_n` points.

    Keeps the parent's endpoints (the pure-prompt vertices stay in the grid) and spaces
    every sample uniformly, so no two columns can land on the same coordinate.
    `full_range` is only used when the parent has a single sample and there is no span
    to subdivide.
    """
    if len(old_axis) > 1:
        lo, hi = float(old_axis[0]), float(old_axis[-1])
    else:
        lo, hi = float(full_range[0]), float(full_range[1])
    return np.linspace(lo, hi, new_n)


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
    # Subdivide INSIDE the old range. Treating each old sample as the centre of a cell
    # of width da put the outermost sub-samples at -da/2 and 1+da/2, and the np.clip
    # that followed collapsed ceil(mult/2) of them per axis-end onto the boundary: with
    # a fixed seed those columns generated byte-identical images, so the border read as
    # a band of zero sensitivity and the duplicate cells cost GPU time for nothing.
    new_alphas = _subdivided_axis(old_alphas, new_gs, config.ALPHA_RANGE)
    new_betas = _subdivided_axis(old_betas, new_gs, config.BETA_RANGE)

    new_embeddings = np.zeros((new_gs, new_gs, 768))
    new_thumbnails = ThumbnailStore(cache)
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
                    # The coarse cell was measured ONCE, so its embedding is written to
                    # one cell of the block it expands into and the rest stay zero, i.e.
                    # unmeasured (compute_sensitivity's zero-norm guard skips them).
                    # Replicating it over the whole span x span block made every pair
                    # inside the block bit-identical, which fabricated an exactly-zero
                    # sensitivity plateau per block and a lattice of bright seams around
                    # it -- the block edges were the only cells left with a neighbour to
                    # differ from, and that difference spans a whole coarse step while
                    # the field is being read at the fine step.
                    new_embeddings[tl_i, tl_j] = old["embeddings"][ci, cj]

                    # carry the hash, not the bytes: both jobs read the same cache file
                    h = old["thumbnail_hashes"].get((ci, cj))
                    if h:
                        new_thumbnails[(tl_i, tl_j)] = h
                        new_thumbnail_hashes[(tl_i, tl_j)] = h
                        thumb_url = cache.url(h)
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
        "use_slerp": old.get("use_slerp", False),
        "embeddings": new_embeddings,
        "sensitivity": None, "clusters": None,
        # "sensitivity" must stay None (refine_grid and build_mf_detector both None-check
        # it), so the parent's Jacobian field is carried under its own key. Without it
        # _analyze_and_render's fast-scan recovery branch had nothing to read and every
        # sub-threshold cell of a refined fast scan rendered as 0.
        "sensitivity_prev": old.get("sensitivity"),
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
                use_slerp=old.get("use_slerp", False),
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

    # Compute threshold from 3D sensitivity. measured_mask, not `> 0`: it is the one
    # definition of "this cell was measured" (ridge_detector uses it for every median it
    # takes), and `> 0` additionally drops the cells whose neighbours are near-identical
    # and land at ~-1e-7 from float error -- those are real, very flat measurements.
    real_sens = sensitivity[measured_mask(sensitivity)]
    if len(real_sens) == 0:
        return RefineResponse(refine_job_id=job_id, parent_job_id=job_id,
                              total_cells=0, status="no_cells")
    median = float(np.median(real_sens))
    threshold = median * req.tau

    old_alphas = master["alphas"]
    old_betas = master["betas"]
    old_gammas = master["gammas"]

    # New coordinates. Subdivided inside the old span, per axis: the cell-centred
    # scheme this replaces ran past [0,1] and clipped, which gave several planes at
    # each face the same coordinate (identical images, zero measured sensitivity), and
    # it reused the alpha step for all three axes.
    new_alphas = _subdivided_axis(old_alphas, new_gs, config.ALPHA_RANGE)
    new_betas = _subdivided_axis(old_betas, new_gs, config.BETA_RANGE)
    new_gammas = _subdivided_axis(old_gammas, new_gs, config.ALPHA_RANGE)

    # Determine which old cells are above threshold → need generation
    needs_generation = set()  # (ni, nj, nk) in new grid
    new_embeddings = np.zeros((new_gs, new_gs, new_gs, 768))

    for i in range(old_gs):
        for j in range(old_gs):
            for k in range(old_gs):
                above = sensitivity[i, j, k] >= threshold
                if above:
                    for si in range(mult):
                        for sj in range(mult):
                            for sk in range(mult):
                                needs_generation.add((i*mult+si, j*mult+sj, k*mult+sk))
                else:
                    # One measurement, one cell: the parent embedding goes to the corner
                    # of the block it expands into and the rest of the block stays
                    # unmeasured. Copying it into all mult^3 sub-voxels made the block
                    # interior exactly zero-distance and left a shell of coarse-scale
                    # differences on its faces, read at the fine step -- a fabricated
                    # cubic lattice on top of the real field.
                    new_embeddings[i*mult, j*mult, k*mult] = master["embeddings"][i, j, k]

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
                    parent_key = (oi, oj, ok) if (oi, oj, ok) in master["thumbnail_hashes"] else None
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
        "height": master.get("height", config.DEFAULT_HEIGHT),
            "width": master.get("width", config.DEFAULT_WIDTH),
        "steps": master.get("steps", config.DEFAULT_NUM_INFERENCE_STEPS),
        "guidance_scale": master.get("guidance_scale", config.DEFAULT_GUIDANCE_SCALE),
        "use_slerp": master.get("use_slerp", False),
        "alphas": new_alphas, "betas": new_betas, "gammas": new_gammas,
        "embeddings": new_embeddings,
        "sensitivity": None, "clusters": None,
        "thumbnails": ThumbnailStore(cache), "thumbnail_hashes": {},
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
                    h = master["thumbnail_hashes"].get((oi, oj, ok))
                    if h:
                        # the hash, not the bytes: mult**3 sub-voxels share one image
                        jobs[job_id]["thumbnails"][(i, j, k)] = h
                        jobs[job_id]["thumbnail_hashes"][(i, j, k)] = h

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
                height=master.get("height", config.DEFAULT_HEIGHT),
                width=master.get("width", config.DEFAULT_WIDTH),
                steps=master.get("steps", config.DEFAULT_NUM_INFERENCE_STEPS),
                guidance_scale=master.get("guidance_scale", config.DEFAULT_GUIDANCE_SCALE),
                prompt_d=master.get("prompt_d", ""), gammas=new_gammas,
                grid_size_z=new_gs, use_slerp=master.get("use_slerp", False),
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
    job_id = str(uuid.uuid4())[:8]
    gs = req.grid_size
    alphas = np.linspace(config.ALPHA_RANGE[0], config.ALPHA_RANGE[1], gs)
    betas = np.linspace(config.BETA_RANGE[0], config.BETA_RANGE[1], gs)

    is_3d = req.dimensions == 3 and req.prompt_d
    gammas = np.linspace(config.ALPHA_RANGE[0], config.ALPHA_RANGE[1], gs) if is_3d else None

    jobs = request.app.state.jobs
    pool = request.app.state.gpu_pool
    cache = request.app.state.cache
    _evict_old_jobs(jobs)

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
        # Recorded so generate-selected and refine mix embeddings exactly the way this
        # scan did; the scan and the generation it selects cells for have to agree.
        "use_slerp": is_3d,
        "latents": None,
        "anisotropy": None,
        "sensitivity": None,
        "clusters": None,
        "embeddings": np.zeros((gs, gs, gs, 768)) if is_3d else np.zeros((gs, gs, 768)),
        "thumbnails": ThumbnailStore(cache), "thumbnail_hashes": {},
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
                use_slerp=is_3d,
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

    # Compute threshold over measured cells only -- same rule as ridge_detector
    real_sens = sensitivity[measured_mask(sensitivity)]
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
                    use_slerp=job.get("use_slerp", False),
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
    # span==1 excludes retained coarse blocks; sensitivity 0 excludes cells that were
    # never generated (skipped by generate-selected, or an un-refined block's interior).
    # Counting those as the flattest points in the space pulled the median down and the
    # tau threshold with it -- the same rule ridge_detector's measured_mask applies.
    real_sensitivities = []
    for row in master["cells"]:
        for cell in row:
            if (cell is not None and cell.get("span", 1) == 1
                    and cell.get("sensitivity")):
                real_sensitivities.append(cell["sensitivity"])
    if not real_sensitivities:
        real_sensitivities = list(sensitivity[measured_mask(sensitivity)])
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
            # Same subdivision the sub-jobs got: the master's axes must agree with them
            # cell for cell, or the heatmap axes label a different point than the one
            # each sub-job generated.
            new_alphas = _subdivided_axis(master["alphas"], new_gs, config.ALPHA_RANGE)
            new_betas = _subdivided_axis(master["betas"], new_gs, config.BETA_RANGE)

            master["phase"] = "generating"
            master["status"] = "running"
            master["grid_size"] = new_gs
            master["alphas"] = new_alphas
            master["betas"] = new_betas
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
            # _flatten_cells dispatches on dimensions; the hand-rolled 2-level loop
            # that was here raised TypeError on every poll of a 3-D multi-seed job,
            # because cells[i][j] is a LIST for a tetrahedron, not a dict.
            seed_cell_map[str(seed_num)] = _flatten_cells(sub)

        # Use first seed's cells as the "primary" cells (for grid display)
        first_sub = jobs.get(sub_ids[0], {})
        flat_cells = _flatten_cells(first_sub) if first_sub.get("cells") else []

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
        # a sub-job that a worker failed marks the whole aggregate failed: its cells can
        # never arrive, so `phase` would otherwise sit at 'generating' forever
        sub_error = next((jobs[s]["error"] for s in sub_ids
                          if jobs.get(s) and jobs[s].get("error")), None)
        if sub_error:
            phase = "error"
        return GridStatusResponse(
            job_id=job_id, status="error" if sub_error else ("running" if not all_complete else "complete"),
            phase=phase, grid_size=job["grid_size"],
            cells_generated=all_gen, cells_total=all_total,
            cells=flat_cells, seed_cells=seed_cell_map, seeds=seeds,
            prompt_a=job["prompt_a"], prompt_b=job["prompt_b"],
            prompt_c=job.get("prompt_c", ""),
            heatmap_url=f"/api/grid/{job_id}/heatmap.png?v={v}" if job.get("heatmap_path") else None,
            overlay_url=f"/api/grid/{job_id}/overlay.png?v={v}" if job.get("overlay_path") else None,
            cluster_url=f"/api/grid/{job_id}/clusters.png?v={v}" if job.get("cluster_path") else None,
            image_grid_url=f"/api/grid/{job_id}/images.png?v={v}" if job.get("image_grid_path") else None,
            error=sub_error,
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
        error=job.get("error"),
    )


def _results_dir(job_id: str):
    """Ridge-graph endpoints read from disk, not from app.state.jobs — that dict is
    in-memory and POST /api/grid/cancel clears it, and old jobs need to stay analysable.

    /start returns the MASTER id but writes analysis under "<master>_s<seed>" (one
    directory per seed), while /refine writes under the master id directly. Resolving
    only the master meant the graph endpoints never found a freshly generated grid.
    Prefer an exact directory; otherwise take the seed directory, lowest seed first so
    the choice is deterministic.
    """
    # job_id arrives from the URL path: reject anything that is not a plain id before
    # it reaches Path.glob, where '*' or '?' would match OTHER jobs' directories and
    # silently return an unrelated job's results.
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", job_id):
        raise HTTPException(status_code=400, detail="invalid job id")
    d = config.RESULTS_DIR / job_id
    if (d / "sensitivity.npy").exists():
        return d
    # sort by the integer seed, not lexicographically: "_s100" sorts before "_s98"
    def _seed_of(p):
        tail = p.name[len(job_id) + 2:]
        return (0, int(tail)) if tail.isdigit() else (1, 0)
    seeded = sorted((p for p in config.RESULTS_DIR.glob(f"{job_id}_s*")
                     if (p / "sensitivity.npy").exists()), key=_seed_of)
    return seeded[0] if seeded else d


def _clip_text_encoder(app):
    """CLIP-text encoder for the prompt-spread and diversity guards (small, CPU).

    Cached on app.state: ~10 s to load, and both the hike loop and the branch endpoint
    need it.
    """
    import open_clip
    import torch
    cached = getattr(app.state, "clip_text", None)
    if cached is None:
        m, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k", device="cpu")
        m.eval()
        cached = (m, open_clip.get_tokenizer("ViT-B-32"), {})
        app.state.clip_text = cached
    m, tok, memo = cached

    def enc(texts):
        # choose_prompts re-encodes the WHOLE pool on every call, and the pool is constant
        # for a hike: 431 ms of CPU per call at the shipped 60-prompt pool, which is >97%
        # of choose_prompts and sits on the critical path at every hop boundary (extend()
        # runs once per surviving chain). Memoise per prompt string -- bit-identical, and
        # takes extend() from ~1.1 s to ~0.3 ms.
        texts = list(texts)
        miss = [t for t in texts if t not in memo]
        if miss:
            with torch.no_grad():
                f = m.encode_text(tok(miss))
                f = (f / f.norm(dim=-1, keepdim=True)).numpy()
            for t, row in zip(miss, f):
                memo[t] = row
        return np.stack([memo[t] for t in texts])
    return enc


def _hike_prompt_pool(req) -> list:
    """The prompt pool a hike draws its fresh vertices from.

    One definition for both entry points. /hike/start built it as an OVERRIDE
    (`req.prompt_pool or HIKE_PROMPT_POOL`, which is what models.py documents the field
    to be) while /hike/{id}/branch UNIONED the two, so a branch started with an uploaded
    pool drew its seed triangle from a set the hike it launched would then never draw
    from again, and the two disagreed about how many prompts were available.

    Blanks are dropped: a BRANCHED hike takes its first triangle from seed_chain, so
    HikeBranchRequest leaves prompt_a/b/c at "" (models.py) -- and the empty string was
    landing at index 0 of the pool. choose_prompts only filters `p not in basis`, and ""
    is never in a branched chain's basis, so it stayed eligible as a simplex vertex for
    every later hop (an unconditioned vertex with an empty label), inflated the `need`
    check in _run_hike_inner by one, and changed the _GRAM cache key for a wasted Gram
    rebuild.
    """
    corpus = [p for p in (getattr(req, "prompt_a", ""), getattr(req, "prompt_b", ""),
                          getattr(req, "prompt_c", "")) if p and p.strip()]
    extra = req.prompt_pool or getattr(config, "HIKE_PROMPT_POOL", [])
    return list(dict.fromkeys(corpus + list(extra)))


def _run_hike(app, hike_id: str, req, seed_chain=None):
    """Beam loop for one hike. Runs on a worker thread; each hop shards a grid per
    surviving chain across the GPU pool and blocks on its CellResults."""
    st = app.state.hikes[hike_id]
    try:
        _run_hike_inner(app, hike_id, req, st, seed_chain)
    except Exception as exc:      # noqa: BLE001 - the poller is the only observer
        # Setup (CLIP download, mkdir, pool lookup) used to run outside the try, so a
        # failure there left the hike reporting "running" forever with no error.
        import traceback
        traceback.print_exc()
        st["status"] = "error"
        st["error"] = f"{type(exc).__name__}: {exc}"
        from backend.services import hiker as _hk
        _hk.save_state(hike_id, st)


def _run_hike_inner(app, hike_id: str, req, st, seed_chain=None):
    import numpy as np
    from PIL import Image, ImageDraw
    import io as _io
    from backend.services import hiker as hk

    d = hk.hike_dir(hike_id)
    d.mkdir(parents=True, exist_ok=True)
    pool = app.state.gpu_pool
    gs = req.grid_size
    mg = int(getattr(req, "margin", 0) or 0)
    # The padding ring exists so hull cells have a full neighbourhood: unpadded they keep
    # 2.6 of 4 neighbours and their sensitivity carries ~30% error on the hypotenuse,
    # which is exactly where exits are chosen. Its cells are generated and MEASURED but
    # never walked to -- see the on_hull filter on candidate exits below.
    pts = hk.simplex_points(gs, mg)

    # Every cell is rendered once per seed, and the walk runs on the field AVERAGED over
    # them -- the same construction the grid view uses in _compute_averaged_sensitivity.
    # This is the cheapest available increase in ridge reliability: E25 (2026-08-15)
    # measured this field's seed-to-seed Spearman at only 0.69-0.79, i.e. seed moves the
    # ridge MORE than any sampler setting does, so a single-seed ridge is substantially a
    # property of one noise draw rather than of the conditioning space.
    #
    # Cost is linear in seed_count, and it is charged in IMAGES, not in grids: grids_done
    # still counts triangles, because req.budget also sets the search's shape (n_left
    # below decides how many chains run, and max_hops above sizes the prompt pool). Making
    # a seed cost a grid would silently narrow the beam instead of deepening the evidence.
    seeds = list(range(req.seed, req.seed + max(1, int(getattr(req, "seed_count", 1) or 1))))

    text_enc = _clip_text_encoder(app)

    # Drop blanks: a BRANCHED hike takes its first triangle from seed_chain, so
    # HikeBranchRequest leaves prompt_a/b/c at "" (models.py) -- and the empty string was
    # landing at index 0 of the pool. choose_prompts only filters `p not in basis`, and ""
    # is never in a branched chain's basis, so it stayed eligible as a simplex vertex for
    # every later hop (an unconditioned vertex with an empty label), inflated the `need`
    # check below by one, and changed the _GRAM cache key for a wasted Gram rebuild.
    prompt_pool = _hike_prompt_pool(req)
    # Two fresh prompts are consumed per HOP per chain, and choose_prompts filters the
    # pool per chain (`p not in basis`), so chains do not compete. The requirement is
    # therefore per-chain hops, not total grids -- charging per grid over-rejected by a
    # factor of `beam` and aborted valid hikes before any GPU work.
    max_hops = -(-max(1, req.budget) // max(1, req.beam))   # ceil
    need = 2 * max_hops + 3
    if len(prompt_pool) < need:
        st["status"] = "error"
        st["error"] = (f"prompt pool has {len(prompt_pool)} entries; a budget of "
                       f"{req.budget} grids at beam {req.beam} runs up to {max_hops} "
                       f"hops per chain and can consume up to {need}")
        hk.save_state(hike_id, st)
        return

    beam = [seed_chain] if seed_chain is not None else [
        hk.Chain([req.prompt_a, req.prompt_b, req.prompt_c],
                 [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]],
                 [req.prompt_a, req.prompt_b, req.prompt_c])]
    archive, grids_done, hop = [], 0, 0
    refined_cells = 0
    # How many walk rows st["chains"] keeps. This was a flat 40, which predates the beam
    # ceiling being raised to 32: ONE hop then writes 32 rows, so by hop 2 the root row
    # is already gone. That is not just cosmetic -- /hike/{id}/branch looks its walk up by
    # (hop, chain) in this list, so branching at an early hop of a wide hike 404s with
    # "no such walk, or it predates branching", and the UI's lineage tree and journeys
    # lose the ancestors they reconstruct paths from. Scale it with the width instead,
    # with a floor that covers an ordinary hike end to end. A row is ~1 KB (stations plus
    # a coefficient basis that grows 2 per hop), so this bounds hike.json at ~1 MB.
    row_cap = max(200, 8 * max(1, req.beam))

    import time as _t

    def _shard_failure(shards):
        """First worker-side failure among these shard ids, or None.

        result_collector records them in app.state.hike_task_errors keyed by shard id.
        Without this a crashed shard is indistinguishable from a slow one and both
        drains below waited out their full 15-minute deadline before giving up. Popping
        the key here keeps the map bounded and means each failure is reported once.
        """
        errs = getattr(app.state, "hike_task_errors", None)
        if not errs:
            return None
        # shards are (tid, n) here and (tid, n, seed_idx) in the hop drain
        for tid, *_rest in shards:
            if tid in errs:
                return f"{tid}: {errs.pop(tid)}"
        return None

    def _drop_shard_errors(shards):
        errs = getattr(app.state, "hike_task_errors", None) or {}
        for tid, *_rest in shards:
            errs.pop(tid, None)

    def _gen_points(ch, points, gs_local, tag, m_local=0, seed=None):
        """Generate an arbitrary point list for one chain and block for it.

        Used by ridge refinement, which needs the same shard-and-drain as a full hop but
        over a sparse set of cells at a finer lattice. Returns None if the work was
        cancelled or timed out, so the caller can fall back to the coarse result.

        `seed` defaults to the hike's base seed; a multi-seed hike calls this once per
        seed over the SAME point list, so the refined lattice exists at every seed and
        the averaged field can be recomputed on it.
        """
        app.state.hike_inbox = getattr(app.state, "hike_inbox", {})
        nsh_l = max(1, min(getattr(pool, "n_gpus", 1), len(points)))
        # Read the epoch BEFORE submitting, not after. POST /api/grid/cancel bumps it
        # and drains the queue; a bump landing part-way through the loop below cancels
        # the shards already queued (they carry the older epoch) while `ep`, if read
        # afterwards, is already the post-cancel value -- so the drain could never tell,
        # and sat out its full 15-minute deadline instead. Reading first makes any bump
        # at or after this instant visible. The converse error is harmless: if a cancel
        # lands between this read and the submits, the tasks do carry the new epoch and
        # would have run, but a cancel really was issued, so aborting is correct anyway.
        ep = pool.cancel_epoch.value
        shards = []
        for g in range(nsh_l):
            sub = points[g::nsh_l]
            if not sub:
                continue
            tid = f"{hike_id}_{tag}_g{g}"
            app.state.hike_inbox.setdefault(tid, [])
            pool.submit(HikeTask(
                job_id=tid, basis=ch.basis, vertex_coefs=ch.coefs,
                points=[(i, j, *hk.bary(i, j, gs_local, m_local)) for i, j in sub],
                seed=req.seed if seed is None else int(seed),
                height=req.height, width=req.width,
                steps=req.steps, guidance_scale=req.guidance_scale))
            shards.append((tid, len(sub)))
        want_r = sum(n for _, n in shards)
        out, got_r, last_r = [], 0, _t.time()
        while got_r < want_r and _t.time() - last_r < 15 * 60:
            moved = False
            for tid, _n in shards:
                ib = app.state.hike_inbox.get(tid) or []
                while ib:
                    out.append(ib.pop(0)); got_r += 1; moved = True
            if moved:
                last_r = _t.time()
            elif got_r < want_r:
                if st.get("cancel_requested") or pool.cancel_epoch.value != ep:
                    break
                fail = _shard_failure(shards)
                if fail is not None:
                    # a dead shard is never coming; keep the coarse walk now rather than
                    # 15 minutes from now. NOT a cancel -- one OOM must not end the hike.
                    st["notes"] = (st.get("notes") or []) + [
                        f"refine shard failed, keeping the coarse walk — {fail}"]
                    break
                _t.sleep(0.25)
        for tid, _n in shards:
            app.state.hike_inbox.pop(tid, None)
        _drop_shard_errors(shards)
        return out if got_r >= want_r else None

    try:
        while grids_done < req.budget and beam and not st.get("cancel_requested"):
            n_left = req.budget - grids_done
            live = beam[:n_left]
            if not live:
                break
            if len(live) < len(beam):
                st["notes"] = (st.get("notes") or []) + [
                    f"hop {hop}: budget allows {len(live)} of {len(beam)} chains; "
                    f"{len(beam) - len(live)} dropped. Raise budget to run full width."]
            elif hop == 0 and req.beam > 1:
                st["notes"] = (st.get("notes") or []) + [
                    f"hop 0 runs 1 chain (one starting simplex); width {req.beam} "
                    f"applies from hop 1."]

            # Submit EVERY chain's work before draining any of it, sharded across GPUs.
            # One HikeTask per grid pinned all len(pts) images to a single worker while
            # the rest idled, and the per-chain submit-then-block meant beam width
            # bought no parallelism either.
            app.state.hike_inbox = getattr(app.state, "hike_inbox", {})
            nsh = max(1, min(getattr(pool, "n_gpus", 1), len(pts)))
            # POST /api/grid/cancel (and every store action that calls it first) bumps
            # the pool's cancel epoch, which aborts the HikeTasks queued below. Their
            # cells will never arrive, so watching the epoch is the only way to tell a
            # cancel apart from slow generation -- otherwise every chain sat out its full
            # 15-minute deadline and the hike still reported "complete".
            #
            # Read it BEFORE the submit loop. It used to be read after, on the reasoning
            # that submit() stamps the epoch so this is "the epoch the tasks carry" --
            # but a cancel landing part-way through the loop cancels the shards already
            # queued (older epoch) and leaves the recorded value ALREADY equal to the
            # post-cancel one, so the drain below could never fire and the hop hung for
            # the full deadline per chain, plus another per refine round. Reading first
            # makes any bump at or after this instant visible; the converse case (cancel
            # between this read and the submits) aborts work that would have run, which
            # is what a cancel asked for.
            hop_epoch = pool.cancel_epoch.value
            pending = []
            for ci, ch in enumerate(live):
                shards = []
                # seed-major, shard-minor. Every seed renders the SAME point list at the
                # same barycentric coordinates, so the only thing that differs between
                # them is the noise draw -- which is exactly what averaging is meant to
                # integrate out. Shards carry their seed index so the drain can keep the
                # per-seed arrays apart.
                for si, sd in enumerate(seeds):
                    for g in range(nsh):
                        sub = pts[g::nsh]
                        if not sub:
                            continue
                        tid = f"{hike_id}_h{hop}_c{ci}_s{si}_g{g}"
                        app.state.hike_inbox.setdefault(tid, [])
                        pool.submit(HikeTask(
                            job_id=tid, basis=ch.basis, vertex_coefs=ch.coefs,
                            points=[(i, j, *hk.bary(i, j, gs, mg)) for i, j in sub],
                            seed=sd, height=req.height, width=req.width,
                            steps=req.steps, guidance_scale=req.guidance_scale))
                        shards.append((tid, len(sub), si))
                pending.append((ci, ch, shards))
            if hop == 0 and len(seeds) > 1:
                st["notes"] = (st.get("notes") or []) + [
                    f"averaging the sensitivity field over {len(seeds)} seeds "
                    f"({seeds[0]}-{seeds[-1]}): {len(seeds)}x the images per grid, and "
                    f"the budget of {req.budget} grids is unchanged."]

            cands = []
            arch = list(archive)      # frozen for the whole hop -- see below
            fresh_emb = []
            cancelled = False
            for ci, ch, shards in pending:
                nn_ = gs + 3 * mg
                # one array per seed; `valid` is derived from all of them below
                dinos = [np.zeros((nn_, nn_, 768)) for _ in seeds]
                valids = [np.zeros((nn_, nn_), bool) for _ in seeds]
                thumb_sets = [{} for _ in seeds]
                got = 0
                shard_fail = None
                want = sum(n for _, n, _s in shards)
                # deadline is per hop and measured from first progress, so queue time
                # behind other chains is not charged as generation time
                last = _t.time()
                while got < want and _t.time() - last < 15 * 60:
                    # two ways to stop: this hike was cancelled by name, or the whole
                    # pool was drained under us (global /cancel bumps the epoch)
                    if st.get("cancel_requested") or pool.cancel_epoch.value != hop_epoch:
                        cancelled = True
                        break
                    moved = False
                    for tid, _n, si in shards:
                        ib = app.state.hike_inbox.get(tid) or []
                        while ib:
                            r = ib.pop(0)
                            dinos[si][r.row, r.col] = r.dino_embedding
                            valids[si][r.row, r.col] = True
                            thumb_sets[si][(r.row, r.col)] = r.thumbnail_bytes
                            got += 1
                            moved = True
                    if moved:
                        last = _t.time()
                    elif got < want:
                        shard_fail = _shard_failure(shards)
                        if shard_fail is not None:
                            # this chain's cells are never arriving; fall through to the
                            # got < want handling below, which charges the grid and drops
                            # the chain. Not a cancel: the other chains still run.
                            break
                        _t.sleep(0.25)
                for tid, _n, _s in shards:
                    app.state.hike_inbox.pop(tid, None)
                _drop_shard_errors(shards)
                if cancelled:
                    st["notes"] = (st.get("notes") or []) + [
                        f"hop {hop}: cancelled while hiker {ci} was generating "
                        f"({got}/{want} cells received)"]
                    break
                # A cell counts as generated only where EVERY seed produced it. Both the
                # averaged field and the archive read across seeds, and a row one seed
                # never filled is a zero vector -- cosine 0 with everything, so it would
                # read as maximally novel and maximally far in reach()/novelty().
                valid = (valids[0] if len(seeds) == 1
                         else np.logical_and.reduce(np.stack(valids)))
                # Normalise before any early exit below: `archive` is fed from `dino`,
                # and raw-magnitude rows would corrupt every later reach/novelty cosine.
                for a in dinos:
                    n = np.linalg.norm(a, axis=-1, keepdims=True)
                    np.divide(a, np.where(n > 0, n, 1), out=a)
                # Seed 0 is the hike's canonical draw: its embeddings steer the archive,
                # the beam and walk()'s semantic contraction, and its images are what the
                # UI shows unless asked for another -- exactly as the grid view keeps
                # first_sub's embeddings for clustering. The other seeds contribute to the
                # sensitivity field and to the per-seed filmstrips, nothing else.
                dino, thumbs = dinos[0], thumb_sets[0]
                # Everything that was generated enters the archive, whatever happens to
                # this chain. Archiving only on the arc path meant a dropped or arcless
                # grid was never seen by hk.reach, so a later chain landing on that same
                # material scored as maximally far from everything and the beam was
                # steered back into ground already covered. Skip an empty grid: it would
                # put a (0, 768) block into `archive`, and novelty/reach only guard the
                # empty-LIST case, so the .max() over no columns would raise.
                if valid.any():
                    fresh_emb.append(dino[valid])
                # Charge the budget for every grid that was submitted and generated,
                # including one whose cells were lost -- the GPU work is spent either
                # way. Incrementing only on the success path let each drop buy an extra
                # uncharged grid, so a hike overran req.budget by the number of drops.
                grids_done += 1
                st["grids_done"] = grids_done
                hk.save_state(hike_id, st)
                if got < want:
                    # one chain's lost cells must not abort the other chains: the
                    # function-level handler would have ended the whole hike
                    st["grids_dropped"] = st.get("grids_dropped", 0) + 1
                    why = (f"GPU shard failed ({shard_fail})" if shard_fail
                           else "timed out")
                    st["notes"] = (st.get("notes") or []) + [
                        f"hop {hop} hiker {ci}: {why} at {got}/{want} cells; "
                        f"chain dropped"]
                    hk.save_state(hike_id, st)
                    continue

                gs_cur, m_cur = gs, mg
                spans = {}          # cell -> how many fine cells it stands for
                # The surface the ridge is found on. One seed: unchanged, walk() computes
                # it. Several: the mean of the PER-SEED fields -- not the field of averaged
                # embeddings, which would cancel exactly the neighbour-to-neighbour
                # differences sensitivity() measures and flatten the ridge away. Same order
                # of operations as _compute_averaged_sensitivity.
                S_avg = (None if len(seeds) == 1 else
                         np.mean([hk.sensitivity(dd, valid) for dd in dinos], axis=0))
                w = hk.walk(dino, valid, gs_cur, req.k_basins, req.h_frac, m_cur, S=S_avg)
                if w is None:
                    # record it: a silent `continue` here made a working run look idle
                    st["chains"] = [dict(chain=ci, hop=hop, labels=list(ch.labels),
                                         lineage=ch.lineage or "root",
                                         arc_len=0, cum_novel=float("nan"), n_arcs=0,
                                         note="no ridge arc on this grid")] + st["chains"][:row_cap]
                    st["notes"] = (st.get("notes") or []) + [
                        f"hop {hop} hiker {ci}: no arc (grid {gs}x{gs} may be too coarse)"]
                    hk.save_state(hike_id, st)
                    continue

                # Iterative ridge refinement, the hiker analogue of /refine on the main
                # grid: double the lattice and generate ONLY near the ridge network, so
                # the walk and the exit it hands to the next hop are located at a finer
                # resolution than the basins ever pay for. bary() is identical at
                # (2i, 2j) of the doubled lattice, so nothing already generated is
                # regenerated or moved. Any failure keeps the coarse result -- a refine
                # is an improvement, never a precondition.
                #
                # The lattice the walk was found on before any refinement, and how far
                # apart its cells sit once refinement has doubled the array. Both are
                # needed for the approach path below, which must be routed on the COARSE
                # mask -- see there.
                valid0, stride = valid, 1
                for rnd in range(int(getattr(req, "refine_rounds", 0) or 0)):
                    # upscale every seed onto the same finer lattice; v2/gs2/s2 depend
                    # only on `valid`, which the seeds share, so take them from seed 0
                    ups = [hk.upscale_lattice(dd, valid, tt, gs_cur, spans)
                           for dd, tt in zip(dinos, thumb_sets)]
                    d2s = [u[0] for u in ups]
                    t2s = [u[2] for u in ups]
                    v2, gs2, s2 = ups[0][1], ups[0][3], ups[0][4]
                    have = {(i, j) for i in range(v2.shape[0])
                            for j in range(v2.shape[1]) if v2[i, j]}
                    tgt = hk.refine_targets(w["arcs"], gs2, have,
                                            radius=1, cap=int(req.refine_cap),
                                            max_sum=gs2 - 1 + 3 * (2 * m_cur))
                    if not tgt:
                        break
                    # Every seed refines the SAME targets. Refining only seed 0 would
                    # leave the finer cells with a one-seed field and the coarse ones with
                    # an averaged field -- a surface whose reliability varies cell to cell,
                    # which is worse than either done consistently.
                    per_seed = []
                    for si, sd in enumerate(seeds):
                        gr = _gen_points(ch, tgt, gs2, f"h{hop}_c{ci}_r{rnd}_s{si}",
                                         2 * m_cur, sd)
                        if gr is None:
                            per_seed = None
                            break
                        per_seed.append(gr)
                    got_r = None if per_seed is None else per_seed[0]
                    if got_r is None:
                        # _gen_points returns None for a cancel AND for a timeout, and
                        # treating both as "refine incomplete" swallowed the cancel: the
                        # hop-level flag is only ever set by the base-grid drain, which
                        # has already finished for this chain (and on the last chain of a
                        # hop there is nobody left to notice), so the loop fell through
                        # and the NEXT hop submitted a full grid after the user pressed
                        # cancel. A global /cancel bumps the pool epoch and never sets
                        # cancel_requested, so test both, exactly as the drain does.
                        if st.get("cancel_requested") or pool.cancel_epoch.value != hop_epoch:
                            cancelled = True
                            st["notes"] = (st.get("notes") or []) + [
                                f"hop {hop} hiker {ci}: cancelled during refine round {rnd}"]
                        else:
                            st["notes"] = (st.get("notes") or []) + [
                                f"hop {hop} hiker {ci}: refine round {rnd} incomplete; "
                                f"kept the {gs_cur}x{gs_cur} walk"]
                        break
                    for si, gr in enumerate(per_seed):
                        for r in gr:
                            d2s[si][r.row, r.col] = r.dino_embedding
                            t2s[si][(r.row, r.col)] = r.thumbnail_bytes
                    # _gen_points returns non-None only when it received every point it
                    # submitted, so all seeds delivered the same cells; mark them once.
                    for r in got_r:
                        v2[r.row, r.col] = True
                        s2[(r.row, r.col)] = 1        # newly refined: covers one cell
                    for a in d2s:
                        nn = np.linalg.norm(a, axis=-1, keepdims=True)
                        np.divide(a, np.where(nn > 0, nn, 1), out=a)
                    S2 = (None if len(seeds) == 1 else
                          np.mean([hk.sensitivity(dd, v2) for dd in d2s], axis=0))
                    w2 = hk.walk(d2s[0], v2, gs2, req.k_basins, req.h_frac, 2 * m_cur,
                                 S=S2)
                    if w2 is None:
                        break
                    dinos, thumb_sets = d2s, t2s
                    dino, thumbs = dinos[0], thumb_sets[0]
                    valid, gs_cur, w, spans = v2, gs2, w2, s2
                    m_cur = 2 * m_cur
                    stride *= 2
                    refined_cells += len(tgt)
                    st["refined_cells"] = refined_cells
                    hk.save_state(hike_id, st)
                    # Refined cells enter the archive too. The append above ran BEFORE
                    # this loop and `dino[valid]` is a copy, so rebinding dino to the
                    # upscaled array left every refined cell invisible to hk.reach and
                    # hk.novelty for the rest of the hike -- up to refine_cap per round
                    # per chain, and precisely the ridge-adjacent material the beam is
                    # most likely to revisit. The chosen exit is usually one of these
                    # cells (an odd index exists only on the refined lattice), so the
                    # very point handed to the next hop was unarchived. Append the delta
                    # only, and to `archive` via fresh_emb rather than to the frozen
                    # `arch`, which would reintroduce the first-chain bias noted below.
                    new = [(r.row, r.col) for r in got_r]
                    if new:
                        fresh_emb.append(np.stack([dino[y, x] for y, x in new]))

                # a cancel seen inside a refine round must end the hike, not just the
                # round: the per-chain drain that normally notices is already past.
                if cancelled:
                    break

                arc = w["arcs"][0]
                # Stop the displayed walk at the hull. walk() orients each arc so the end
                # with the SMALLEST barycentric weight is last, and a padding-ring cell's
                # weight is negative, so with margin>0 arcs[0] essentially always ends in
                # the ring -- a cell nothing exits from, since the candidate loop below
                # walks the same arc back to the hull. Uncorrected, the map's red exit
                # ring, the last filmstrip frame and the last `stations` entry (what
                # /hike/{id}/branch seeds a new hike from) all pointed at a cell outside
                # the simplex, a couple of cells from the exit that really seeds the next
                # hop -- breaking the very continuity the approach prefix exists for.
                on = [k for k, (y, x) in enumerate(arc)
                      if hk.on_hull(y, x, gs_cur, m_cur)]
                if on:
                    arc = arc[:on[-1] + 1]
                # From hop 1 on, vertex A of this triangle IS the previous hop's exit
                # (extend() carries its coefficient vector verbatim), so walking in from
                # there makes the filmstrip continuous across the hand-off. Hop 0 has no
                # predecessor, so nothing to walk in from.
                approach = []
                if hop > 0:
                    # Route on the REFINED lattice, against the GENERATED mask -- not
                    # w["valid"], which is `valid & isfinite(S)`: a carried coarse cell
                    # whose fine neighbours were never filled has NaN sensitivity but a
                    # perfectly good image, and this route is display-only, so
                    # unmeasurable sensitivity is no reason to refuse to show it. That
                    # mask alone was not enough either -- after a refine round the
                    # un-refined ground sits on the stride-2**r sublattice, so a
                    # single-cell step out of the entry corner always missed and the
                    # whole path was abandoned, silently, from the first refined hop on.
                    # approach_path is stride-aware now (finest step first, doubling
                    # until it lands on a generated cell), so it walks coarse-to-coarse
                    # through the basins and drops back to 1 inside the fine band, ending
                    # exactly on arc[0].
                    approach = hk.approach_path((m_cur, m_cur), arc[0], valid)
                    if not approach and stride > 1:
                        # Fall back to routing on the pre-refinement lattice and scaling
                        # up: coarse (i, j) sits at exactly (stride*i, stride*j) after
                        # refinement, so the scaled path is still a real sequence of
                        # generated images. Costs a bridge of up to 2*stride at the join,
                        # where rounding to the coarse lattice lands beside arc[0].
                        t0 = (int(round(arc[0][0] / stride)),
                              int(round(arc[0][1] / stride)))
                        approach = [(y * stride, x * stride)
                                    for y, x in hk.approach_path((mg, mg), t0, valid0)]
                    if not approach:
                        st["notes"] = (st.get("notes") or []) + [
                            f"hop {hop} hiker {ci}: no approach path from the entry cell; "
                            f"this filmstrip starts at the ridge, not at the exit image"]
                # Novelty is the RIDGE arc's figure, so take it before the prefix. The
                # approach frames are display-only re-treads: they begin at vertex A,
                # which is pixel-identical to the previous hop's exit, and cross its
                # neighbourhood -- material the archive holds by construction -- so
                # including them pushed the user-facing "N% new" down by path length alone.
                vecs = np.stack([dino[y, x] for y, x in arc])
                best = approach + arc
                # scored against the archive as it stood at the START of the hop.
                # Appending inside this loop made chain i compete against i earlier
                # grids that chain 0 never saw; since reach and 1-novelty only grow
                # with archive size, the global sort then systematically favoured the
                # first chain in `live` and collapsed the beam onto one lineage.
                ch.cum_novel = hk.novelty(vecs, arch)
                # filmstrip for this hop
                # Tiles keep their GENERATED size. They used to be forced to 96px, so a
                # 256px hike was downsampled 2.7x on the server and the UI then scaled the
                # result back up -- soft frames that no client-side setting could recover.
                # The strip is now resolution-independent and both UIs slide it by a
                # FRACTION of its own width, so nothing depends on a fixed tile size.
                def _write_strip(tset, out_path):
                    tiles, tile_px = [], 0
                    for k, (y, x) in enumerate(best):
                        if (y, x) not in tset:
                            continue
                        im = Image.open(_io.BytesIO(tset[(y, x)])).convert("RGB")
                        if tile_px == 0:
                            tile_px = max(im.width, im.height)
                        if im.size != (tile_px, tile_px):
                            im = im.resize((tile_px, tile_px), Image.LANCZOS)
                        lab = max(12, tile_px // 8)      # index band scales with the tile
                        t = Image.new("RGB", (tile_px, tile_px + lab), "black")
                        t.paste(im, (0, 0))
                        ImageDraw.Draw(t).text((3, tile_px + max(1, lab // 6)), str(k),
                                               fill="white")
                        tiles.append(t)
                    if tiles:
                        h = tiles[0].height
                        strip = Image.new("RGB", (tile_px * len(tiles), h), "black")
                        for k, t in enumerate(tiles):
                            strip.paste(t, (k * tile_px, 0))
                        # quality 94: these are now the sharpest copy the UI ever shows
                        strip.save(out_path, quality=94)

                # One strip per seed, over the SAME walk: the route was chosen on the
                # averaged field, so switching seed in the UI re-renders an identical
                # itinerary under a different noise draw rather than showing a different
                # hike. Seed 0 keeps the historic filename, so every existing URL, cached
                # client and stored hike directory still resolves.
                for si in range(len(seeds)):
                    _write_strip(thumb_sets[si],
                                 d / (f"hop{hop}_c{ci}.jpg" if si == 0
                                      else f"hop{hop}_c{ci}_s{si}.jpg"))
                # the simplex map: where the walk ran, which arcs it passed over, and
                # where it exits. The filmstrip alone cannot show any of that.
                try:
                    hk.render_hike_map(thumbs, w["S"], w["valid"], w["arcs"], best, gs_cur,
                                       d / f"hop{hop}_c{ci}_map.jpg",
                                       exit_ij=best[-1], junctions=w.get("junctions") or [],
                                       spans=spans, m=m_cur, approach=len(approach))
                except Exception as exc:      # a map is a nicety; never fail a hop for it
                    st["notes"] = (st.get("notes") or []) + [
                        f"hop {hop} hiker {ci}: map render failed ({type(exc).__name__})"]
                for a in w["arcs"]:
                    ey, ex = a[-1]
                    # never exit into the padding ring: past the hull the annulus is
                    # significantly less novel than the interior (3/3 triplets, E3c).
                    # Walk the arc back until it re-enters the true simplex.
                    if not hk.on_hull(ey, ex, gs_cur, m_cur):
                        inside = [p for p in a if hk.on_hull(p[0], p[1], gs_cur, m_cur)]
                        if not inside:
                            continue
                        ey, ex = inside[-1]
                    # carry the lattice the exit was found on: after refinement it is
                    # finer than `gs`, and extend() converts the exit to barycentric
                    # weights, so the wrong gs would silently relocate the exit.
                    cands.append(dict(chain=ch, exit=(int(ey), int(ex)), gs=gs_cur, m=m_cur,
                                      reach=hk.reach(dino[ey, ex], arch)))
                st["chains"] = [dict(chain=ci, hop=hop, labels=list(ch.labels),
                                     lineage=ch.lineage or "root",
                                     arc_len=len(best), cum_novel=ch.cum_novel,
                                     # the walk itself, so the UI can animate a marker
                                     # along the map instead of only showing the baked
                                     # JPEG. Array indices; pair with grid+margin to
                                     # place them (map layout is x=i, y=n-1-j).
                                     grid=gs_cur, margin=m_cur, cell=30,
                                     frame_px=int(req.width), n_frames=len(best),
                                     approach=len(approach),
                                     # how many per-seed filmstrips exist for this row,
                                     # and which draws they are; the UI offers a switcher
                                     # only when there is more than one
                                     n_seeds=len(seeds), seeds=[int(s) for s in seeds],
                                     # the chain's coefficient basis, so any station on
                                     # this walk can be turned back into an embedding
                                     # coordinate and used to seed a new hike
                                     basis=list(ch.basis),
                                     coefs=[list(map(float, c)) for c in ch.coefs],
                                     stations=[[int(y), int(x)] for y, x in best],
                                     n_arcs=len(w["arcs"]))] + st["chains"][:row_cap]
                hk.save_state(hike_id, st)

            if cancelled:
                st["status"] = "cancelled"
                break
            archive.extend(fresh_emb)
            if not cands:
                st["notes"] = (st.get("notes") or []) + [
                    f"stopped at hop {hop}: no viable candidate exits"]
                break
            cands.sort(key=lambda c: c["reach"])
            nxt, seen = [], set()
            for c in cands:
                nc = hk.extend(c["chain"], c["exit"], c.get("gs", gs), prompt_pool,
                               text_enc, c.get("m", 0))
                if nc is None or "|".join(nc.labels) in seen:
                    continue
                seen.add("|".join(nc.labels)); nxt.append(nc)
                if len(nxt) >= req.beam:
                    break
            if not nxt:
                break
            beam = nxt
            hop += 1
            st["hop"] = hop
            hk.save_state(hike_id, st)
        if st["status"] == "running":
            st["status"] = "cancelled" if st.get("cancel_requested") else "complete"
    except Exception as exc:  # surface the failure instead of hanging the poller
        st["status"] = "error"
        st["error"] = f"{type(exc).__name__}: {exc}"
    hk.save_state(hike_id, st)


@router.post("/hike/start", response_model=HikeStartResponse)
async def start_hike(req: HikeStartRequest, request: Request):
    """Launch a population of hikers. Runs in the background against the shared GPU pool;
    poll /api/grid/hike/{id}/status."""
    import asyncio
    from backend.services import hiker as hk
    hid = hk.new_hike_id()
    state = dict(hike_id=hid, status="running", hop=0, grids_done=0,
                 budget=req.budget, beam=req.beam, chains=[], error=None,
                 params=req.model_dump())
    hk.save_state(hid, state)
    request.app.state.hikes = getattr(request.app.state, "hikes", {})
    request.app.state.hikes[hid] = state
    asyncio.create_task(asyncio.to_thread(_run_hike, request.app, hid, req))
    return {"hike_id": hid, "budget": req.budget, "beam": req.beam, "status": "running"}


@router.get("/hike/{hike_id}/status", response_model=HikeStatus)
async def hike_status(hike_id: str, request: Request):
    from backend.services import hiker as hk
    st = getattr(request.app.state, "hikes", {}).get(hike_id) or hk.load_state(hike_id)
    if st is None:
        return {"hike_id": hike_id, "status": "unknown", "hop": 0, "grids_done": 0,
                "budget": 0, "chains": [], "error": "no such hike"}
    return st


@router.post("/hike/{hike_id}/branch", response_model=HikeStartResponse)
async def branch_hike(hike_id: str, req: HikeBranchRequest, request: Request):
    """Start a new hike from one station of an existing walk.

    A station is a barycentric point in that chain's simplex, i.e. an embedding
    coordinate with no phrase attached. `extend()` turns it into the first vertex of a
    fresh triangle alongside two newly drawn prompts -- exactly what an edge-hop does at
    the end of a hop, except the user picks the point instead of the selector. A station
    in the padding ring is not such a point; it is backed up to the hull below.
    """
    import asyncio

    from backend.services import hiker as hk
    hikes = getattr(request.app.state, "hikes", {})
    st = hikes.get(hike_id) or hk.load_state(hike_id)
    if st is None:
        raise HTTPException(status_code=404, detail="no such hike")
    row = next((r for r in st.get("chains", [])
                if r.get("hop") == req.hop and r.get("chain") == req.chain), None)
    if row is None or not row.get("basis"):
        raise HTTPException(status_code=404, detail="no such walk, or it predates branching")
    stations = row.get("stations") or []
    if not (0 <= req.station < len(stations)):
        raise HTTPException(status_code=400,
                            detail=f"station out of range (0..{len(stations)-1})")

    parent = hk.Chain(list(row["basis"]), [list(c) for c in row["coefs"]],
                      list(row.get("labels") or row["basis"]))
    # the same pool the hike this call launches will draw from, so the seed triangle and
    # every later hop agree about what is available
    pool = _hike_prompt_pool(req)
    enc = _clip_text_encoder(request.app)
    row_gs = int(row.get("grid") or req.grid_size)
    row_m = int(row.get("margin") or 0)
    # A ring station is NOT a barycentric point in the simplex: past the hull one weight
    # goes negative and extend() would anchor the whole new hike at an affine
    # extrapolation, in an annulus measured 4.5-10.5 pp less novel than the interior
    # (3/3 triplets, E3c). Walks recorded before the hull truncation in _run_hike_inner
    # can still carry such stations, so apply the same walk-back the hop loop applies to
    # candidate exits: back up to the last station at or before this one that is inside.
    sy, sx = stations[req.station]
    moved = None
    if not hk.on_hull(int(sy), int(sx), row_gs, row_m):
        back = [p for p in stations[:req.station + 1]
                if hk.on_hull(int(p[0]), int(p[1]), row_gs, row_m)]
        if not back:
            raise HTTPException(
                status_code=400,
                detail=(f"station {req.station} is in the padding ring, and no earlier "
                        f"station on this walk is inside the simplex"))
        moved = (int(sy), int(sx))
        sy, sx = back[-1]
    seed_chain = hk.extend(parent, (int(sy), int(sx)), row_gs, pool, enc, row_m)
    if seed_chain is None:
        raise HTTPException(status_code=409,
                            detail="could not draw two fresh prompts for that point")

    hid = hk.new_hike_id()
    notes = [f"branched from hike {hike_id} hop {req.hop} station {req.station}"]
    if moved is not None:
        notes.append(f"station {req.station} at {list(moved)} is in the padding ring; "
                     f"seeded from the last station inside the simplex, {[sy, sx]}")
    state = dict(hike_id=hid, status="running", hop=0, grids_done=0,
                 budget=req.budget, beam=req.beam, chains=[], notes=notes,
                 error=None, params=req.model_dump())
    hk.save_state(hid, state)
    request.app.state.hikes = getattr(request.app.state, "hikes", {})
    request.app.state.hikes[hid] = state
    asyncio.create_task(asyncio.to_thread(_run_hike, request.app, hid, req, seed_chain))
    return {"hike_id": hid, "budget": req.budget, "beam": req.beam, "status": "running"}


@router.post("/hike/{hike_id}/cancel")
async def cancel_hike(hike_id: str, request: Request):
    """Stop one hike by name, without touching any other job.

    POST /api/grid/cancel bumps a single process-wide epoch, so using it to stop a hike
    also aborts whatever grid a colleague is generating. This is cooperative instead: it
    sets a flag the hike loop checks between hops, inside each chain's drain, and inside
    a refinement round, so the hike stops within a poll rather than at the next hop.

    The GPUs are only drained when nothing else is using them. Otherwise the images
    already queued for this hike finish and are discarded, which wastes at most one hop
    of generation but cannot disturb another user's job.
    """
    from backend.services import hiker as hk
    hikes = getattr(request.app.state, "hikes", {})
    st = hikes.get(hike_id) or hk.load_state(hike_id)
    if st is None:
        raise HTTPException(status_code=404, detail="no such hike")
    if st.get("status") != "running":
        return {"hike_id": hike_id, "status": st.get("status"), "drained": 0,
                "note": "already finished"}
    st["cancel_requested"] = True
    st["notes"] = (st.get("notes") or []) + ["cancelled by request"]
    hk.save_state(hike_id, st)

    pool = request.app.state.gpu_pool
    jobs = request.app.state.jobs
    busy = any(isinstance(j, dict) and j.get("phase") in ("generating", "scanning")
               for j in jobs.values())
    others = any(h is not st and h.get("status") == "running" for h in hikes.values())
    drained = 0 if (busy or others) else pool.drain_pending()
    return {"hike_id": hike_id, "status": "cancelling", "drained": drained}


def _strip_tiles(w_px: int, h_px: int) -> int:
    """How many tiles a filmstrip holds, from its dimensions alone.

    A tile is a square image of side s with an index band under it, so the strip is
    (s*n) x (s + max(12, s//8)). The estimate this replaces was round(width/height),
    which ignores the band and therefore under-counts by about 1/9 -- an 8-tile 256px
    strip read as 7. That made the derived tile width 293 instead of 256, so `?w=256`
    "downscaled" a strip already at exactly the requested size, costing the mobile UI
    ~12% of its resolution at every hike resolution. Invert the construction instead:
    the strip width is an exact multiple of the tile side, so test the divisors.
    """
    for n in range(1, 4097):
        if w_px % n:
            continue
        s = w_px // n
        if s < 1:
            break
        if s + max(12, s // 8) == h_px:
            return n
    return max(1, round(w_px / max(h_px, 1)))


@router.get("/hike/{hike_id}/chain/{chain_id}/hop/{hop}.jpg")
async def hike_filmstrip(hike_id: str, chain_id: int, hop: int, w: int = 0,
                         seed: int = 0):
    """Filmstrip along the arc this chain walked at this hop.

    `seed` is an INDEX into the hike's seed list, not a seed value: 0 is the canonical
    draw and the only one that exists unless the hike ran with seed_count > 1. The walk
    is identical across the variants -- it was found on the field averaged over all of
    them -- so switching seed re-renders the same itinerary under a different noise draw,
    which is what makes the comparison worth showing.
    """
    from fastapi.responses import JSONResponse, Response
    from backend.services import hiker as hk
    p = hk.hike_dir(hike_id) / (f"hop{hop}_c{chain_id}.jpg" if seed <= 0
                                else f"hop{hop}_c{chain_id}_s{int(seed)}.jpg")
    if not p.exists():
        return JSONResponse({"error": "not ready"}, status_code=404)
    if not w:
        return Response(content=p.read_bytes(), media_type="image/jpeg")

    # Strips are stored at the hike's generated resolution, so a 512px run is a
    # 2048x576 JPEG per hop per chain -- fine on a desktop, heavy over mobile data.
    # `?w=` serves a variant scaled so each TILE is w px wide; never upscales, so
    # asking for more than was generated returns the original. The variant is written
    # beside the source, so the resize is paid once per (hop, chain, w).
    var = p.with_name(f"{p.stem}_w{int(w)}.jpg")
    if not var.exists():
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        try:
            im = Image.open(p).convert("RGB")
            n = _strip_tiles(im.width, im.height)
            tile = im.width / n
            if tile <= w:
                return Response(content=p.read_bytes(), media_type="image/jpeg")
            k = w / tile
            im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))),
                      Image.LANCZOS).save(var, quality=90)
        except Exception:
            return Response(content=p.read_bytes(), media_type="image/jpeg")
    return Response(content=var.read_bytes(), media_type="image/jpeg")


@router.get("/hike/{hike_id}/chain/{chain_id}/hop/{hop}/map.jpg")
async def hike_map(hike_id: str, chain_id: int, hop: int):
    """The simplex this chain walked: thumbnails tinted by sensitivity, every arc that
    was available, the one taken highlighted, and the exit that seeds the next hop."""
    from fastapi.responses import JSONResponse, Response
    from backend.services import hiker as hk
    p = hk.hike_dir(hike_id) / f"hop{hop}_c{chain_id}_map.jpg"
    if not p.exists():
        return JSONResponse({"error": "not ready"}, status_code=404)
    return Response(content=p.read_bytes(), media_type="image/jpeg")


@router.post("/{job_id}/graph", response_model=RidgeGraphResponse)
async def build_graph(job_id: str, req: RidgeGraphRequest):
    """Extract the ridge network as walkable arcs. No GPU: rebuilt from the stored
    sensitivity field and DINOv2 embeddings, so K and the persistence depth can be swept
    interactively."""
    from starlette.concurrency import run_in_threadpool

    from backend.services import ridge_graph as rg
    d = _results_dir(job_id)
    if not (d / "sensitivity.npy").exists():
        return {"job_id": job_id, "n_arcs": 0, "n_basins": 0, "n_junctions": 0,
                "params": {"error": "no sensitivity.npy for this job"}}
    try:
        # seconds of clustering and watershed on a 192x192 grid: on the event loop it
        # froze every other request, including the grid-status poll
        graph = await run_in_threadpool(
            rg.build_ridge_graph, d, k=req.k, h_frac=req.h_frac, min_arc=req.min_arc)
    except (FileNotFoundError, ValueError) as exc:
        # ValueError is the 3-D / shape-mismatch path and was reaching the client as a
        # bare 500 that the UI reported as a network failure
        return {"job_id": job_id, "n_arcs": 0, "n_basins": 0, "n_junctions": 0,
                "params": {"error": str(exc)}}
    rg.save_graph(d, graph)
    params = dict(graph["params"])
    if graph.get("note"):
        params["note"] = graph["note"]
    return {"job_id": job_id, "n_arcs": graph["n_arcs"], "n_basins": graph["n_basins"],
            "n_junctions": len(graph["junctions"]), "params": params}


@router.get("/{job_id}/ridge_graph.json")
async def get_ridge_graph(job_id: str):
    from backend.services import ridge_graph as rg
    p = rg.graph_path(_results_dir(job_id))
    if not p.exists():
        return {"error": "not built — POST /api/grid/{job_id}/graph first"}
    return FileResponse(p, media_type="application/json")


@router.get("/{job_id}/itinerary/{edge_id}.jpg")
async def get_itinerary(job_id: str, edge_id: int):
    """Filmstrip of the generations along one arc, in walk order."""
    import json as _json
    from fastapi.responses import JSONResponse, Response
    from starlette.concurrency import run_in_threadpool

    from backend.services import ridge_graph as rg
    d = _results_dir(job_id)
    p = rg.graph_path(d)
    # the only consumer is an <img> tag: a 200 with a JSON body renders as a broken
    # image with no way for onError to tell "not ready" from "no such arc"
    if not p.exists():
        return JSONResponse({"error": "graph not built"}, status_code=404)
    if not (d / "images.png").exists():
        return JSONResponse({"error": "no image montage for this job"}, status_code=404)
    try:
        data = await run_in_threadpool(
            rg.render_itinerary, d, _json.loads(p.read_text()), edge_id)
    except KeyError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    return Response(content=data, media_type="image/jpeg")


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


# These are read by <img src>, so a bare dict is HTTP 200 + application/json: the
# browser sees a successful request that is not an image and onError never fires.
def _render_layer(request: Request, job_id: str, path_key: str):
    job = request.app.state.jobs.get(job_id)
    if not job or not job.get(path_key):
        raise HTTPException(status_code=404, detail="not ready")
    return FileResponse(job[path_key], media_type="image/png")


@router.get("/{job_id}/heatmap.png")
async def get_heatmap(job_id: str, request: Request):
    return _render_layer(request, job_id, "heatmap_path")


@router.get("/{job_id}/overlay.png")
async def get_overlay(job_id: str, request: Request):
    return _render_layer(request, job_id, "overlay_path")


@router.get("/{job_id}/clusters.png")
async def get_clusters(job_id: str, request: Request):
    return _render_layer(request, job_id, "cluster_path")


@router.get("/{job_id}/images.png")
async def get_image_grid(job_id: str, request: Request):
    return _render_layer(request, job_id, "image_grid_path")


@router.get("/{job_id}/export/{layer}.jpg")
async def export_grid(job_id: str, layer: str, request: Request):
    """Export grid as full-resolution PNG. Layers: images, heatmap, overlay."""
    from io import BytesIO
    from PIL import Image
    from fastapi.responses import StreamingResponse

    jobs = request.app.state.jobs
    job = jobs.get(job_id)
    # the client saves the body to <layer>.jpg: a bare dict is HTTP 200 +
    # application/json, so a missing job downloaded as a "successful" broken image
    if not job:
        raise HTTPException(status_code=404, detail="no such job")

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
        raise HTTPException(status_code=404, detail=f"unknown layer: {layer}")

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
    jobs = request.app.state.jobs
    master = jobs.get(job_id)
    if not master:
        return SeedProbeResponse(probe_id="", total=0, status="error")

    # Only preempt when nothing else is generating. drain_pending() discards queued
    # tasks belonging to whatever job is running and never re-submits them, so the
    # victim's cells_generated stalls below total forever and its poll never
    # completes -- a probe should not silently destroy a grid in flight.
    busy = [jid for jid, j in jobs.items()
            if isinstance(j, dict) and j.get("phase") == "generating" and jid != job_id]
    if busy:
        print(f"Seed probe queued behind {len(busy)} generating job(s); not draining",
              flush=True)
    else:
        drained = pool.drain_pending()
        if drained:
            print(f"Seed probe preempted {drained} queued generation tasks", flush=True)

    probe_id = f"probe_{uuid.uuid4().hex[:6]}"
    seeds = list(range(req.seed_start, req.seed_end + 1))
    n = len(seeds)

    alphas = np.array([req.alpha])
    betas = np.array([req.beta])

    # Use requested steps, or fall back to parent job's steps
    probe_steps = req.steps if req.steps > 0 else master.get("steps", config.DEFAULT_NUM_INFERENCE_STEPS)

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
            "height": master.get("height", config.DEFAULT_HEIGHT),
            "width": master.get("width", config.DEFAULT_WIDTH),
            "steps": probe_steps,
            "guidance_scale": master.get("guidance_scale", 4.0),
            "alphas": alphas, "betas": betas,
            "embeddings": np.zeros((1, 1, 768)),
            "sensitivity": None, "clusters": None,
            "thumbnails": ThumbnailStore(cache), "thumbnail_hashes": {},
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
            height=master.get("height", config.DEFAULT_HEIGHT),
                width=master.get("width", config.DEFAULT_WIDTH),
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
    # Step 1: Start a fast scan (reuse existing infrastructure)
    job_id = str(uuid.uuid4())[:8]
    gs = req.grid_size
    alphas = np.linspace(config.ALPHA_RANGE[0], config.ALPHA_RANGE[1], gs)
    betas = np.linspace(config.BETA_RANGE[0], config.BETA_RANGE[1], gs)

    jobs = request.app.state.jobs
    pool = request.app.state.gpu_pool
    cache = request.app.state.cache
    _evict_old_jobs(jobs)

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
        "thumbnails": ThumbnailStore(cache), "thumbnail_hashes": {},
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
                use_slerp=False,  # MF is 2D; _run_mf_pipeline generates with plain LERP
            ))

    return MFScanResponse(
        job_id=job_id, total_cells=total,
        budget=req.budget, status="running"
    )


@router.post("/{job_id}/surprise-sample", response_model=SurpriseSampleResponse)
async def surprise_sample_cells(job_id: str, req: SurpriseSampleRequest, request: Request):
    """The surprise slider: draw n cells biased along the argmin(S) <-> argmax(S) axis.

    -1 = calm interiors (coherent, on-prompt), +1 = boundaries (novel, hybrid). Measured
    basis: search_problem E31/E33/E62 -- novelty is monotone in boundary degree at a
    coherence price. Uses the job's current (possibly partial) sensitivity field.
    """
    from backend.services.surprise import surprise_sample
    jobs = request.app.state.jobs
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    flat = _flatten_cells(job)
    cells = [c for c in flat if c.span == 1]
    idxs, probs = surprise_sample([c.sensitivity for c in cells],
                                  req.surprise, req.n, req.seed)
    return SurpriseSampleResponse(
        job_id=job_id, surprise=req.surprise,
        cells=[(cells[i].row, cells[i].col) for i in idxs],
        probabilities={f"{cells[i].row},{cells[i].col}": p for i, p in probs.items()},
    )


# ---------------------------------------------------------------------------------------------
# Exact-JVP probes: grid-free ridge normal at a point, and per-token sensitivity of a prompt.
# Evidence: search_problem/outputs/h07_chain_SUMMARY.md and h08_vector_uses/RESULTS.md.
# ---------------------------------------------------------------------------------------------
def _reading(normal_bary: list, k: int, names: list | None = None) -> str:
    names = names or (list("ABCD") if k <= 4 else [f"P{i + 1}" for i in range(k)])
    parts = sorted(zip(names, normal_bary), key=lambda x: -abs(x[1]))
    return (f"±({parts[0][0]} {'+' if parts[0][1] > 0 else '-'}, {parts[1][0]} {'+' if parts[1][1] > 0 else '-'}): "
            f"crossing here trades {parts[0][0]} against {parts[1][0]}")


@router.post("/{job_id}/jvp-probe", response_model=ProbeStartResponse)
async def jvp_probe_start(job_id: str, req: JvpProbeRequest, request: Request):
    """Exact local Jacobian at (alpha, beta[, gamma]) of this job: the whitened ridge normal (seed-invariant),
    rank-1 share and participation ratio. ~25 s on one GPU; the direction is the output, sigma1 is comparable
    only within this job."""
    jobs = request.app.state.jobs
    master = jobs.get(job_id)
    if not master or not isinstance(master, dict):
        return ProbeStartResponse(probe_id="", status="error")
    prompts = [master["prompt_a"], master["prompt_b"]]
    weights = [1.0 - req.alpha - req.beta - req.gamma, req.alpha]
    if master.get("prompt_c"):
        prompts.append(master["prompt_c"]); weights.append(req.beta)
    if master.get("prompt_d") and master.get("dimensions", 2) == 3:
        prompts.append(master["prompt_d"]); weights.append(req.gamma)
    # alpha + beta > 1 (negative weight on A) is outside the simplex but inside the square the 2-D grid draws;
    # the mixing is affine there (see gpu_pool._nlerp), so the probe serves those cells too. The h07/h08 evidence
    # for seed-invariance was gathered on the simplex proper; treat readings there as the same map, less tested.
    probe_id = f"jvp_{uuid.uuid4().hex[:8]}"
    jobs[probe_id] = {"type": "jvp_probe", "status": "running", "kind": "jvp", "job_id": job_id, "alpha": req.alpha,
                      "beta": req.beta, "gamma": req.gamma, "k": len(prompts), "result": None, "error": "", "started_at": time.time()}
    request.app.state.gpu_pool.submit(ProbeTask(
        probe_id=probe_id, job_id=job_id, prompts=prompts, weights=weights,
        seed=req.seed if req.seed is not None else int(master.get("seed", config.DEFAULT_SEED)),
        height=master.get("height", config.DEFAULT_HEIGHT), width=master.get("width", config.DEFAULT_WIDTH),
        steps=master.get("steps", config.DEFAULT_NUM_INFERENCE_STEPS),
        guidance_scale=master.get("guidance_scale", config.DEFAULT_GUIDANCE_SCALE), use_slerp=bool(master.get("use_slerp", False))))
    return ProbeStartResponse(probe_id=probe_id, status="running")


@router.post("/{job_id}/token-probe", response_model=ProbeStartResponse)
async def token_probe_start(job_id: str, req: TokenProbeRequest, request: Request):
    """One exact JVP per token of prompt `which` (a-d): which word the generation is load-bearing on.
    Rankings are seed-invariant and follow the word, not its slot (h08-3, h08-6). ~3 min on one GPU."""
    jobs = request.app.state.jobs
    master = jobs.get(job_id)
    prompt = (master or {}).get(f"prompt_{req.which}") if isinstance(master, dict) else None
    if not prompt:
        return ProbeStartResponse(probe_id="", status="error")
    probe_id = f"tok_{uuid.uuid4().hex[:8]}"
    jobs[probe_id] = {"type": "token_probe", "status": "running", "kind": "token", "job_id": job_id, "which": req.which,
                      "prompt": prompt, "result": None, "error": "", "started_at": time.time()}
    request.app.state.gpu_pool.submit(TokenProbeTask(
        probe_id=probe_id, job_id=job_id, prompt=prompt,
        seed=req.seed if req.seed is not None else int(master.get("seed", config.DEFAULT_SEED)),
        height=master.get("height", config.DEFAULT_HEIGHT), width=master.get("width", config.DEFAULT_WIDTH),
        steps=master.get("steps", config.DEFAULT_NUM_INFERENCE_STEPS),
        guidance_scale=master.get("guidance_scale", config.DEFAULT_GUIDANCE_SCALE)))
    return ProbeStartResponse(probe_id=probe_id, status="running")


@router.get("/jvp-probe/{probe_id}", response_model=ProbeStatusResponse)
@router.get("/token-probe/{probe_id}", response_model=ProbeStatusResponse)
async def probe_status(probe_id: str, request: Request):
    entry = request.app.state.jobs.get(probe_id)
    if not entry or entry.get("type") not in ("jvp_probe", "token_probe"):
        return ProbeStatusResponse(probe_id=probe_id, status="error", kind="", error="unknown probe")
    result = entry.get("result")
    if entry.get("status") == "done" and entry["kind"] == "jvp" and result and "normal_reading" not in result:
        result = dict(result, alpha=entry.get("alpha"), beta=entry.get("beta"), gamma=entry.get("gamma"),
                      weights=entry.get("weights"), cid=entry.get("cid"),
                      normal_reading=_reading(result["normal_bary"], result["k"], entry.get("names")))
        entry["result"] = result
    return ProbeStatusResponse(probe_id=probe_id, status=entry.get("status", "running"), kind=entry["kind"],
                               error=entry.get("error", ""), result=result)
