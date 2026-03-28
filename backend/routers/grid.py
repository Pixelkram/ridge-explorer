import uuid
import numpy as np
from fastapi import APIRouter, Request
from fastapi.responses import FileResponse

from backend.models import (
    GridStartRequest, GridStartResponse, GridStatusResponse, CellStatus,
    RenderHQRequest, RefineRequest, RefineResponse,
    SeedProbeRequest, SeedProbeResponse, SeedProbeStatus,
)
from backend.services.gpu_pool import GenerateTask, HQTask
from backend import config


router = APIRouter(prefix="/api/grid")


@router.post("/start", response_model=GridStartResponse)
async def start_grid(req: GridStartRequest, request: Request):
    master_id = str(uuid.uuid4())[:8]
    gs = req.grid_size
    alphas = np.linspace(config.ALPHA_RANGE[0], config.ALPHA_RANGE[1], gs)
    betas = np.linspace(config.BETA_RANGE[0], config.BETA_RANGE[1], gs)

    jobs = request.app.state.jobs
    pool = request.app.state.gpu_pool

    seeds = list(range(req.seed, req.seed + req.seed_count))
    total_cells_all = gs * gs * len(seeds)

    # Create a sub-job per seed
    sub_job_ids = []
    for seed in seeds:
        sub_id = f"{master_id}_s{seed}"
        jobs[sub_id] = _make_job(sub_id, gs, alphas, betas,
                                 req.prompt_a, req.prompt_b, req.prompt_c, seed,
                                 height=req.height, width=req.width,
                                 steps=req.steps, guidance_scale=req.guidance_scale)
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
            "grid_size": gs,
            "total_cells": total_cells_all,
            "cells_generated": 0,
            "prompt_a": req.prompt_a,
            "prompt_b": req.prompt_b,
            "prompt_c": req.prompt_c,
            "seed": req.seed,
            "seeds": seeds,
            "sub_job_ids": sub_job_ids,
            "height": req.height, "width": req.width,
            "steps": req.steps, "guidance_scale": req.guidance_scale,
            "alphas": alphas, "betas": betas,
            "embeddings": np.zeros((gs, gs, 768)),
            "sensitivity": None, "clusters": None,
            "thumbnails": {}, "thumbnail_hashes": {},
            "cells": [[{
                "row": i, "col": j,
                "alpha": float(alphas[i]), "beta": float(betas[j]),
                "status": "pending",
                "sensitivity": None, "cluster": None,
                "thumbnail_url": None, "hq_url": None, "span": 1,
            } for j in range(gs)] for i in range(gs)],
            "heatmap_path": None, "overlay_path": None,
            "cluster_path": None, "image_grid_path": None,
            "render_version": 0,
        }

    return GridStartResponse(job_id=master_id, total_cells=total_cells_all, status="running")


def _make_job(job_id, gs, alphas, betas, prompt_a, prompt_b, prompt_c, seed,
              height=256, width=256, steps=4, guidance_scale=4.0):
    return {
        "phase": "generating",
        "status": "running",
        "grid_size": gs,
        "total_cells": gs * gs,
        "cells_generated": 0,
        "prompt_a": prompt_a,
        "prompt_b": prompt_b,
        "prompt_c": prompt_c,
        "seed": seed,
        "height": height,
        "width": width,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "alphas": alphas,
        "betas": betas,
        "embeddings": np.zeros((gs, gs, 768)),
        "sensitivity": None,
        "clusters": None,
        "thumbnails": {},
        "thumbnail_hashes": {},
        "cells": [[{
            "row": i, "col": j,
            "alpha": float(alphas[i]), "beta": float(betas[j]),
            "status": "pending",
            "sensitivity": None, "cluster": None,
            "thumbnail_url": None, "hq_url": None,
            "span": 1,
        } for j in range(gs)] for i in range(gs)],
        "heatmap_path": None, "overlay_path": None,
        "cluster_path": None, "image_grid_path": None,
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

    jobs[sub_id] = {
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


@router.post("/{job_id}/refine", response_model=RefineResponse)
async def refine_grid(job_id: str, req: RefineRequest, request: Request):
    """Refine grid in-place. For multi-seed, refines all sub-jobs."""
    jobs = request.app.state.jobs
    master = jobs.get(job_id)
    if not master or master["phase"] != "complete":
        return RefineResponse(refine_job_id=job_id, parent_job_id=job_id,
                              total_cells=0, status="error")

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

    # Single seed — same as before
    flat_cells = []
    for row in job["cells"]:
        for cell in row:
            if cell is not None:
                flat_cells.append(CellStatus(**cell))

    v = job.get("render_version", 0)
    return GridStatusResponse(
        job_id=job_id,
        status=job["status"],
        phase=job["phase"],
        grid_size=job["grid_size"],
        cells_generated=job["cells_generated"],
        cells_total=job["total_cells"],
        cells=flat_cells,
        seeds=seeds,
        prompt_a=job["prompt_a"],
        prompt_b=job["prompt_b"],
        prompt_c=job.get("prompt_c", ""),
        heatmap_url=f"/api/grid/{job_id}/heatmap.png?v={v}" if job.get("heatmap_path") else None,
        overlay_url=f"/api/grid/{job_id}/overlay.png?v={v}" if job.get("overlay_path") else None,
        cluster_url=f"/api/grid/{job_id}/clusters.png?v={v}" if job.get("cluster_path") else None,
        image_grid_url=f"/api/grid/{job_id}/images.png?v={v}" if job.get("image_grid_path") else None,
    )


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


@router.post("/{job_id}/seed-probe", response_model=SeedProbeResponse)
async def seed_probe(job_id: str, req: SeedProbeRequest, request: Request):
    """Generate a single (alpha, beta) point across many seeds."""
    jobs = request.app.state.jobs
    master = jobs.get(job_id)
    if not master:
        return SeedProbeResponse(probe_id="", total=0, status="error")

    probe_id = f"probe_{uuid.uuid4().hex[:6]}"
    seeds = list(range(req.seed_start, req.seed_end + 1))
    n = len(seeds)

    alphas = np.array([req.alpha])
    betas = np.array([req.beta])

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
            "steps": master.get("steps", 4),
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
            steps=master.get("steps", 4),
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
