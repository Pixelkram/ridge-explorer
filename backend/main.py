import time
import asyncio
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend import config
from backend.services.gpu_pool import GPUPool, CellResult, LatentResult, LatentBatchResult, ProbeResult
from backend.services.ridge_detector import compute_sensitivity, compute_clusters, classify_ridges, measured_mask
from backend.services.visualization import render_heatmap, assemble_image_grid, render_overlay, render_clusters
from backend.cache.thumbnail_cache import ThumbnailCache
from backend.routers import health, grid, discover, cascade, amr, metro, probe

import numpy as np


async def result_collector(app: FastAPI):
    """Background task that drains GPU results and updates job state."""
    pool: GPUPool = app.state.gpu_pool
    cache: ThumbnailCache = app.state.cache
    jobs: dict = app.state.jobs

    while True:
        # Worker-side failures are drained separately from results (they are dicts, not
        # dataclasses). Without this the failed task was an invisible hole in the job's
        # cell count: cells_generated never reached total_cells, so the job stayed in
        # phase 'generating' and the UI polled it forever.
        for err in pool.pop_task_errors():
            msg = f"GPU {err.get('gpu_id')}: {err.get('error', 'worker failed')}"
            tid = err.get("job_id") or ""
            print(f"[Collector] {err.get('type')} for job {tid} — {msg}", flush=True)
            failed = jobs.get(tid)
            if failed is not None:
                failed["phase"] = "error"
                failed["status"] = "error"
                failed["error"] = msg
                continue
            # A hike's shards are not entries in `jobs`: their ids are
            # "{hike_id}_h{hop}_c{chain}_g{shard}" (and "{hike_id}_h..._r{round}_g{shard}"
            # for a refine round), and a hike id is hex, so the prefix before the first
            # '_' names the hike. Without this branch a worker-side failure was printed
            # and dropped, and the hike's drain -- which can only tell "no cells yet" from
            # "no cells ever" by waiting -- sat out its 15-minute deadline and then said
            # "timed out; chain dropped" with nothing anywhere naming the GPU error. The
            # wait itself belongs to the hiker loop; this makes the run SAY what happened
            # while it is happening. Not persisted here: the hike thread owns this dict
            # and saves it, so writing hike.json from the collector would race it.
            hike = getattr(app.state, "hikes", {}).get(tid.split("_")[0])
            if hike is not None:
                note = f"GPU task {tid} failed — {msg}"
                notes = hike.get("notes") or []
                if note not in notes:
                    hike["notes"] = notes + [note]
                # Fail FAST as well as loudly. The note above makes the failure visible,
                # but the drain still had no way to learn its shard is never coming and
                # burned the whole 15-minute deadline first. Record it per shard id; the
                # drains in grid.py watch their own tids here and give up immediately.
                # Deliberately NOT a cancel: a single OOM must drop one chain, not the
                # hike. The drain pops its own keys, so this cannot grow unbounded.
                errs = getattr(app.state, "hike_task_errors", None)
                if errs is None:
                    errs = app.state.hike_task_errors = {}
                errs[tid] = msg
            else:
                # Non-hike run types (cascade walks etc.) watch this dict
                # by their full shard tid; drains pop their own keys, so
                # this cannot grow unbounded.
                errs = getattr(app.state, "hike_task_errors", None)
                if errs is None:
                    errs = app.state.hike_task_errors = {}
                errs[tid] = msg

        results = pool.collect_results()
        for r in results:
            if isinstance(r, ProbeResult):
                entry = jobs.get(r.probe_id)
                if entry is not None:
                    entry["status"] = "error" if r.error else "done"
                    entry["error"] = r.error
                    entry["result"] = r.result
                    entry["finished_at"] = time.time()
                continue
            job = jobs.get(r.job_id)
            if not job:
                # Hike tasks are not entries in `jobs`; route them to their inbox rather
                # than dropping the result on the floor (the default `continue` here
                # silently discarded every hike cell).
                inbox = getattr(app.state, "hike_inbox", None)
                if inbox is not None:
                    # check-then-use raced _run_hike's pop() on timeout/cleanup and
                    # killed the collector with a KeyError, stalling every job
                    lst = inbox.get(r.job_id)
                    if lst is not None:
                        lst.append(r)
                continue

            try:
                if isinstance(r, LatentBatchResult):
                    # Fast scan batch result — row of latent vectors (2D or 3D)
                    is_3d_scan = job.get("dimensions", 2) == 3
                    if job["latents"] is None:
                        gs = job["grid_size"]
                        dim = r.latent_vectors.shape[1]
                        if is_3d_scan:
                            gs_z = job.get("grid_size_z", gs)
                            job["latents"] = np.zeros((gs, gs, gs_z, dim))
                        else:
                            job["latents"] = np.zeros((gs, gs, dim))

                    if is_3d_scan and r.depths is not None:
                        for idx in range(len(r.cols)):
                            col, depth = r.cols[idx], r.depths[idx]
                            job["latents"][r.row, col, depth] = r.latent_vectors[idx]
                            job["cells"][r.row][col][depth]["status"] = "scanned"
                    else:
                        for idx, col in enumerate(r.cols):
                            job["latents"][r.row, col] = r.latent_vectors[idx]
                            job["cells"][r.row][col]["status"] = "scanned"
                    job["cells_generated"] += len(r.cols)

                    if job["cells_generated"] >= job["total_cells"] and job["phase"] == "scanning":
                        job["phase"] = "analyzing"
                        asyncio.create_task(asyncio.to_thread(
                            _analyze_fast_scan, job, r.job_id
                        ))

                    # MF-scan: after Jacobian sweep, run MF-GP pipeline
                    if job["cells_generated"] >= job["total_cells"] and job["phase"] == "mf_scanning":
                        job["phase"] = "mf_jacobian_done"
                        asyncio.create_task(asyncio.to_thread(
                            _run_mf_pipeline, job, r.job_id, app
                        ))

                elif isinstance(r, LatentResult):
                    # Fast scan single result (legacy path)
                    if job["latents"] is None:
                        gs = job["grid_size"]
                        dim = r.latent_vector.shape[0]
                        job["latents"] = np.zeros((gs, gs, dim))
                    job["latents"][r.row, r.col] = r.latent_vector
                    job["cells"][r.row][r.col]["status"] = "scanned"
                    job["cells_generated"] += 1

                    if job["cells_generated"] >= job["total_cells"] and job["phase"] == "scanning":
                        job["phase"] = "analyzing"
                        asyncio.create_task(asyncio.to_thread(
                            _analyze_fast_scan, job, r.job_id
                        ))

                    if job["cells_generated"] >= job["total_cells"] and job["phase"] == "mf_scanning":
                        job["phase"] = "mf_jacobian_done"
                        asyncio.create_task(asyncio.to_thread(
                            _run_mf_pipeline, job, r.job_id, app
                        ))

                elif isinstance(r, CellResult):
                    is_3d = job.get("dimensions", 2) == 3

                    # Find the cell entry
                    if is_3d:
                        cell = None
                        try:
                            cell = job["cells"][r.row][r.col][r.depth]
                        except (IndexError, TypeError):
                            pass
                    else:
                        cell = None
                        try:
                            cell = job["cells"][r.row][r.col]
                        except (IndexError, TypeError):
                            pass

                    if cell is None:
                        continue

                    # Save thumbnail
                    key = (r.row, r.col, r.depth) if is_3d else (r.row, r.col)
                    cache.save(r.thumbnail_hash, r.thumbnail_bytes)
                    # store the hash, not the bytes: job["thumbnails"] is a read-through
                    # ThumbnailStore, so the job dict no longer pins a second
                    # full-resolution copy of every render for the life of the process
                    job["thumbnails"][key] = r.thumbnail_hash
                    job["thumbnail_hashes"][key] = r.thumbnail_hash

                    # Store DINOv2 embedding
                    if is_3d:
                        job["embeddings"][r.row, r.col, r.depth] = r.dino_embedding
                    else:
                        job["embeddings"][r.row, r.col] = r.dino_embedding

                    # Update cell status
                    status = "hq" if r.is_hq else "generated"
                    cell["status"] = status
                    url_key = "hq_url" if r.is_hq else "thumbnail_url"
                    cell[url_key] = cache.url(r.thumbnail_hash)
                    job["cells_generated"] += 1

                    # When all cells are generated, compute ridges
                    if job["cells_generated"] >= job["total_cells"] and job["phase"] == "generating":
                        if job.get("type") == "mf_scan":
                            # MF-scan: finalize with GP-predicted sensitivity
                            job["phase"] = "mf_finalizing"
                            asyncio.create_task(asyncio.to_thread(
                                _finalize_mf_scan, job, r.job_id
                            ))
                        elif job.get("type") == "fast_scan" and not job.get("_run_analysis"):
                            job["phase"] = "complete"
                            job["status"] = "complete"
                            job["render_version"] = job.get("render_version", 0) + 1
                            print(f"Job {r.job_id}: fast scan image generation complete", flush=True)
                        else:
                            job["phase"] = "analyzing"
                            asyncio.create_task(asyncio.to_thread(
                                _analyze_and_render, job, r.job_id
                            ))

            except Exception as e:
                print(f"[Collector] Error processing {type(r).__name__}: {e}", flush=True)
                import traceback; traceback.print_exc()

        # Use shorter poll interval when results are flowing
        interval = 0.05 if results else config.RESULT_POLL_INTERVAL
        await asyncio.sleep(interval)


def _cell_spans(job: dict) -> dict:
    """(row, col) -> span for a 2D job's cells.

    A refine keeps a retained coarse cell's single thumbnail under its top-left key and
    gives the cell a span of `multiplier`. The montage needs that span to fill the block;
    without it every montage after a refine was mostly black tiles.
    """
    spans = {}
    for row in job.get("cells", []):
        for cell in row:
            if cell is not None and cell.get("span", 1) > 1:
                spans[(cell["row"], cell["col"])] = cell["span"]
    return spans


def _analyze_and_render(job: dict, job_id: str):
    """Compute ridge map + clusters + render all visualizations."""
    from backend.services.ridge_detector import extract_ridge_mesh
    import json

    results_dir = config.RESULTS_DIR / job_id
    results_dir.mkdir(parents=True, exist_ok=True)

    gs = job["grid_size"]
    is_3d = job.get("dimensions", 2) == 3
    embs = job["embeddings"]

    # Ridge sensitivity
    sensitivity = compute_sensitivity(embs)

    # For fast_scan jobs: cells without DINOv2 embeddings get sensitivity=0 from
    # compute_sensitivity. Preserve their Jacobian sensitivity from the scan phase.
    # A refine builds a fresh job dict whose "sensitivity" must stay None (refine_grid
    # and build_mf_detector both None-check it), so the parent's Jacobian field arrives
    # under "sensitivity_prev". Reading "sensitivity" alone made this branch dead on the
    # only path that reaches it, and every sub-threshold cell of a refined fast scan
    # rendered as 0 — exactly the map the fast scan exists to produce.
    old_sensitivity = job.get("sensitivity_prev")
    if old_sensitivity is None:
        old_sensitivity = job.get("sensitivity")
    if job.get("type") == "fast_scan" and old_sensitivity is not None:
        # Where DINOv2 produced 0 (no embedding), keep old Jacobian value
        mask = sensitivity == 0
        if old_sensitivity.shape == sensitivity.shape:
            sensitivity[mask] = old_sensitivity[mask]
        else:
            # Grid was refined — scale up old sensitivity to new grid
            from scipy.ndimage import zoom as ndizoom
            scale = tuple(n / o for n, o in zip(sensitivity.shape, old_sensitivity.shape))
            old_upsampled = ndizoom(old_sensitivity, scale, order=1)
            sensitivity[mask] = old_upsampled[mask]

    job["sensitivity"] = sensitivity
    np.save(results_dir / "sensitivity.npy", sensitivity)

    # Persist the per-cell DINOv2 embeddings. Without these, every rebuild
    # (re-clustering at a different K, any vector-valued readout) needs the
    # images regenerated on GPU — sensitivity.npy alone is a scalar reduction
    # and cannot be inverted. Stored as fp16: ~2 MB at 50x50, ~8 MB at 100x100.
    try:
        np.save(results_dir / "embeddings.npy", np.asarray(embs, dtype=np.float16))
    except Exception as exc:  # never let a cache write kill the job
        print(f"[analyze] could not persist embeddings for {job_id}: {exc}", flush=True)

    if is_3d:
        gs_z = embs.shape[2]
        # Update 3D cell sensitivities
        for i in range(gs):
            for j in range(gs):
                for k in range(gs_z):
                    if job["cells"][i][j][k] is not None:
                        job["cells"][i][j][k]["sensitivity"] = float(sensitivity[i, j, k])

        # DBSCAN clustering
        clusters = compute_clusters(embs)
        job["clusters"] = clusters
        for i in range(gs):
            for j in range(gs):
                for k in range(gs_z):
                    if job["cells"][i][j][k] is not None:
                        job["cells"][i][j][k]["cluster"] = int(clusters[i, j, k])

        # Marching cubes ridge mesh
        mesh = extract_ridge_mesh(sensitivity, tau=1.5)
        if mesh:
            mesh_path = results_dir / "ridge_mesh.json"
            with open(mesh_path, 'w') as f:
                json.dump(mesh, f)
            job["ridge_mesh_path"] = str(mesh_path)

    else:
        gs_b = job.get("grid_size_b", gs)
        # Update 2D cell sensitivities
        for i in range(gs):
            for j in range(gs_b):
                if job["cells"][i][j] is not None:
                    job["cells"][i][j]["sensitivity"] = float(sensitivity[i, j])

        # DBSCAN clustering
        clusters = compute_clusters(embs)
        job["clusters"] = clusters
        for i in range(gs):
            for j in range(gs_b):
                if job["cells"][i][j] is not None:
                    job["cells"][i][j]["cluster"] = int(clusters[i, j])

    # 2D-only rendering (skip for 3D — visualization is done client-side with Plotly)
    if not is_3d:
        spans = _cell_spans(job)
        heatmap_path = results_dir / "heatmap.png"
        render_heatmap(sensitivity, job["alphas"], job["betas"], heatmap_path,
                       prompt_a=job["prompt_a"], prompt_b=job["prompt_b"], prompt_c=job["prompt_c"])
        job["heatmap_path"] = str(heatmap_path)

        image_grid_path = results_dir / "images.png"
        assemble_image_grid(job["thumbnails"], gs, image_grid_path, spans=spans)
        job["image_grid_path"] = str(image_grid_path)

        overlay_path = results_dir / "overlay.png"
        render_overlay(sensitivity, job["thumbnails"], gs, job["alphas"], job["betas"],
                       config.RIDGE_THRESHOLD_TAU, overlay_path, spans=spans)
        job["overlay_path"] = str(overlay_path)

        cluster_path = results_dir / "clusters.png"
        render_clusters(clusters, job["thumbnails"], gs, job["alphas"], job["betas"],
                        cluster_path, spans=spans)
        job["cluster_path"] = str(cluster_path)

    job["phase"] = "complete"
    job["status"] = "complete"
    job["render_version"] = job.get("render_version", 0) + 1
    n_clusters = len(set(clusters.ravel()) - {-1}) if clusters is not None else 0
    # over measured cells only: after a refine most of the array is the unmeasured
    # sentinel 0, and the raw median then collapses to 0 (ZeroDivisionError / inf here)
    meas = sensitivity[measured_mask(sensitivity)]
    ratio = float(meas.max() / np.median(meas)) if meas.size else float("nan")
    print(f"Job {job_id}: analysis complete "
          f"({'3D' if is_3d else '2D'}, "
          f"m/m={ratio:.1f} over {meas.size} measured cells, "
          f"{n_clusters} clusters)", flush=True)


def _analyze_fast_scan(job: dict, job_id: str):
    """Compute Jacobian spectral norm from 1-step latents. Supports 2D and 3D."""
    from backend.services.ridge_detector import compute_jacobian_sensitivity, compute_jacobian_sensitivity_3d

    gs = job["grid_size"]
    is_3d = job.get("dimensions", 2) == 3
    latents = job["latents"]

    if is_3d:
        spectral, anisotropy = compute_jacobian_sensitivity_3d(latents)
    else:
        spectral, anisotropy = compute_jacobian_sensitivity(latents)

    job["sensitivity"] = spectral
    job["anisotropy"] = anisotropy

    results_dir = config.RESULTS_DIR / job_id
    results_dir.mkdir(parents=True, exist_ok=True)

    if is_3d:
        gs_z = latents.shape[2]
        for i in range(gs):
            for j in range(gs):
                for k in range(gs_z):
                    if job["cells"][i][j][k] is not None:
                        job["cells"][i][j][k]["sensitivity"] = float(spectral[i, j, k])
                        job["cells"][i][j][k]["status"] = "scanned"

        # Extract ridge mesh for 3D visualization
        from backend.services.ridge_detector import extract_ridge_mesh
        import json
        mesh = extract_ridge_mesh(spectral, tau=1.5)
        if mesh:
            mesh_path = results_dir / "ridge_mesh.json"
            with open(mesh_path, 'w') as f:
                json.dump(mesh, f)
            job["ridge_mesh_path"] = str(mesh_path)
    else:
        for i in range(gs):
            for j in range(gs):
                if job["cells"][i][j] is not None:
                    job["cells"][i][j]["sensitivity"] = float(spectral[i, j])
                    job["cells"][i][j]["status"] = "scanned"

        heatmap_path = results_dir / "heatmap.png"
        render_heatmap(spectral, job["alphas"], job["betas"], heatmap_path,
                       prompt_a=job["prompt_a"], prompt_b=job["prompt_b"],
                       prompt_c=job.get("prompt_c", ""))
        job["heatmap_path"] = str(heatmap_path)

    np.save(results_dir / "spectral_norm.npy", spectral)
    np.save(results_dir / "anisotropy.npy", anisotropy)

    job["phase"] = "scan_complete"
    job["status"] = "scan_complete"
    job["render_version"] = job.get("render_version", 0) + 1

    print(f"Fast scan {job_id}: Jacobian spectral norm computed "
          f"(max={spectral.max():.4f}, median={np.median(spectral):.4f}, "
          f"grid={gs}x{gs})", flush=True)


def _run_mf_pipeline(job: dict, job_id: str, app):
    """After Jacobian sweep: compute spectral norm, build MF-GP, select points, submit generation."""
    from backend.services.ridge_detector import compute_jacobian_sensitivity
    from backend.services.mf_gp import build_mf_detector
    from backend.services.gpu_pool import GenerateTask

    gs = job["grid_size"]
    latents = job["latents"]

    # Step 1: Compute Jacobian spectral norm
    spectral, anisotropy = compute_jacobian_sensitivity(latents)
    job["sensitivity"] = spectral
    job["anisotropy"] = anisotropy

    # Update ALL cell-level sensitivities with Jacobian values
    for i in range(gs):
        for j in range(gs):
            if job["cells"][i][j] is not None:
                job["cells"][i][j]["sensitivity"] = float(spectral[i, j])
                job["cells"][i][j]["status"] = "scanned"

    results_dir = config.RESULTS_DIR / job_id
    results_dir.mkdir(parents=True, exist_ok=True)
    np.save(results_dir / "spectral_norm.npy", spectral)

    # Render initial Jacobian heatmap (visible while GP runs)
    render_heatmap(spectral, job["alphas"], job["betas"],
                   results_dir / "heatmap.png",
                   prompt_a=job["prompt_a"], prompt_b=job["prompt_b"],
                   prompt_c=job.get("prompt_c", ""))
    job["heatmap_path"] = str(results_dir / "heatmap.png")
    job["render_version"] = job.get("render_version", 0) + 1

    print(f"MF-scan {job_id}: Jacobian done (max={spectral.max():.4f}, "
          f"median={np.median(spectral):.4f})", flush=True)

    # Step 2: Build MF detector
    detector = build_mf_detector(job)
    if detector is None:
        print(f"MF-scan {job_id}: ERROR building detector", flush=True)
        job["phase"] = "scan_complete"
        job["status"] = "scan_complete"
        return

    budget = job.get("mf_budget", 80)
    tau_mf = job.get("mf_tau", 1.3)
    detector.tau_mf = tau_mf

    # Step 3: Select seed points + straddle-acquired points (all upfront)
    # We select ALL points now, then generate them in one batch for GPU efficiency
    seed_indices = detector.select_seed_points(n=15)
    # Use Jacobian values as pseudo-observations for initial GP
    # (bootstrapping: use rank-preserved Jacobian values scaled to estimated DINOv2 range)
    jac_vals = detector.jacobian
    jac_min, jac_max = jac_vals.min(), jac_vals.max()
    # Rough scaling: map Jacobian range to [0.01, 0.31] (typical DINOv2 sensitivity range)
    pseudo_scale = 0.3 / (jac_max - jac_min + 1e-10)
    pseudo_dino = (jac_vals - jac_min) * pseudo_scale + 0.01

    # NOTE on what this loop actually is. pseudo_dino is affine in the Jacobian, so
    # _fit_gp's linear calibration reproduces it exactly: residuals are rounding noise
    # (~1e-17) and sigma collapses with them. The straddle score sigma - |mu - threshold|
    # therefore reduces to "cells whose Jacobian is closest to one fixed level", and on
    # real Jacobian maps it picks the same set as argsort(|jac - level|) with no GP at
    # all. This is one-shot Jacobian level-set sampling, not uncertainty-driven active
    # learning. The GP only does real work in _finalize_mf_scan, which discards these
    # pseudo values and refits a fresh detector on true DINOv2 observations.
    all_selected = set(seed_indices)
    detector.observe(seed_indices, [float(pseudo_dino[i]) for i in seed_indices])

    remaining = budget - len(all_selected)
    while remaining > 0:
        batch = detector.acquire(batch_size=min(20, remaining))
        if not batch:
            break
        all_selected.update(batch)
        detector.observe(batch, [float(pseudo_dino[i]) for i in batch])
        remaining = budget - len(all_selected)

    job["mf_detector"] = detector
    job["mf_selected_indices"] = sorted(all_selected)

    # Step 4: Map selected indices back to (row, col) grid coordinates
    valid_mask = detector.valid_mask
    valid_positions = np.argwhere(valid_mask)  # (N_valid, 2) of (i, j)
    active_cells = set()
    for idx in all_selected:
        i, j = int(valid_positions[idx, 0]), int(valid_positions[idx, 1])
        active_cells.add((i, j))

    print(f"MF-scan {job_id}: selected {len(active_cells)} cells for DINOv2 "
          f"(budget={budget})", flush=True)

    # Step 5: Submit generation tasks
    job["phase"] = "generating"
    job["total_cells"] = len(active_cells)
    job["cells_generated"] = 0
    job["render_version"] = job.get("render_version", 0) + 1

    for i in range(gs):
        for j in range(gs):
            if (i, j) in active_cells:
                job["cells"][i][j]["status"] = "pending"
            else:
                job["cells"][i][j]["status"] = "skipped"

    pool = app.state.gpu_pool
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
                height=job["height"], width=job["width"],
                steps=job["steps"], guidance_scale=job["guidance_scale"],
                active_cells=active_frozen,
            ))


def _finalize_mf_scan(job: dict, job_id: str):
    """After DINOv2 generation: re-fit GP with real values, compute final sensitivity map."""
    from backend.services.mf_gp import build_mf_detector

    gs = job["grid_size"]
    embeddings = job["embeddings"]
    detector = job.get("mf_detector")

    if detector is None:
        detector = build_mf_detector(job)
        if detector is None:
            job["phase"] = "complete"
            job["status"] = "complete"
            return

    # Compute DINOv2 sensitivity at generated cells from their embeddings
    valid_mask = detector.valid_mask
    valid_positions = np.argwhere(valid_mask)
    selected = job.get("mf_selected_indices", [])

    # Re-observe with REAL DINOv2 sensitivity values
    real_indices = []
    real_values = []
    for idx in selected:
        i, j = int(valid_positions[idx, 0]), int(valid_positions[idx, 1])
        emb = embeddings[i, j]
        if np.linalg.norm(emb) < 0.01:
            continue  # No embedding generated

        # Compute local sensitivity: mean cosine distance to neighbors
        neighbors = []
        for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
            ni, nj = i+di, j+dj
            if 0 <= ni < gs and 0 <= nj < gs:
                n_emb = embeddings[ni, nj]
                if np.linalg.norm(n_emb) > 0.01:
                    cos_dist = 1 - np.dot(emb, n_emb) / (np.linalg.norm(emb) * np.linalg.norm(n_emb) + 1e-10)
                    neighbors.append(cos_dist)

        if neighbors:
            sens = float(np.mean(neighbors))
            real_indices.append(idx)
            real_values.append(sens)

    print(f"MF-scan {job_id}: {len(real_indices)}/{len(selected)} cells have DINOv2 sensitivity", flush=True)

    if real_indices:
        # Rebuild detector with real observations
        detector_final = build_mf_detector(job)
        detector_final.tau_mf = job.get("mf_tau", 1.3)
        detector_final.observe(real_indices, real_values)

        # Get GP-predicted sensitivity — blend with Jacobian outside simplex.
        # Off-simplex cells only ever have the cheap Jacobian, and pasting it in raw put
        # two units into one array, one colorbar and one max/med statistic: on a real run
        # the off-simplex Jacobian median was 13x the in-simplex GP median, so the entire
        # simplex — the only region actually measured — collapsed into the bottom of the
        # colour ramp. rho/intercept is the calibration the GP fits for exactly this
        # cheap->DINOv2 conversion, so send the off-simplex values through it. In-simplex
        # values are unchanged.
        # calibrate() is the identity when <3 observations were made: no calibration was
        # fitted, predict() falls back to the raw Jacobian, and the whole map is
        # therefore already in Jacobian units.
        pred_flat = detector_final.predict()
        blended = detector_final.calibrate(job["sensitivity"])
        for k in range(detector_final.n_valid):
            i, j = int(valid_positions[k, 0]), int(valid_positions[k, 1])
            blended[i, j] = pred_flat[k]  # overwrite simplex cells with GP prediction

        # Keep the per-cell values the UI reads in the same units as the rendered map;
        # off-simplex cells still carried the raw Jacobian written by _run_mf_pipeline.
        for i in range(gs):
            for j in range(gs):
                if job["cells"][i][j] is not None:
                    job["cells"][i][j]["sensitivity"] = float(blended[i, j])

        job["sensitivity"] = blended
        job["mf_predicted_sensitivity"] = blended
        job["mf_n_observed"] = len(real_indices)
        job["mf_detector"] = detector_final

        # Render heatmap — values everywhere (calibrated Jacobian outside, GP inside)
        results_dir = config.RESULTS_DIR / job_id
        results_dir.mkdir(parents=True, exist_ok=True)

        render_heatmap(blended, job["alphas"], job["betas"],
                       results_dir / "heatmap.png",
                       prompt_a=job["prompt_a"], prompt_b=job["prompt_b"],
                       prompt_c=job.get("prompt_c", ""))
        job["heatmap_path"] = str(results_dir / "heatmap.png")

        # Assemble image grid (only generated cells have thumbnails)
        if job["thumbnails"]:
            image_grid_path = results_dir / "images.png"
            assemble_image_grid(job["thumbnails"], gs, image_grid_path, spans=_cell_spans(job))
            job["image_grid_path"] = str(image_grid_path)

        np.save(results_dir / "mf_sensitivity.npy", blended)

        print(f"MF-scan {job_id}: complete "
              f"(observed={len(real_indices)}, "
              f"rho_jac={detector_final.correlation:.3f})", flush=True)
    else:
        # No cell had a generated 4-neighbour, so no DINOv2 value could be formed. The
        # job still completes, but job["sensitivity"] is the pure Jacobian and nothing
        # is re-rendered — say so rather than serving a Jacobian map labelled DINOv2.
        print(f"MF-scan {job_id}: no DINOv2 observations could be formed from "
              f"{len(selected)} generated cells; sensitivity map is the raw Jacobian",
              flush=True)

    job["phase"] = "complete"
    job["status"] = "complete"
    job["render_version"] = job.get("render_version", 0) + 1


def _reap_orphaned_hikes() -> int:
    """Mark hikes the PREVIOUS process was running as interrupted. Returns how many.

    A running hike is a dict on app.state plus the thread mutating it; only the json
    snapshot survives a restart. /hike/{id}/status falls back to that snapshot, and the
    browser resumes a hike id from localStorage, so after a restart (or a crash, or a
    redeploy) the resumed panel polled "running" for ever: the progress bar never moved,
    `send hikers` stayed disabled because the panel believes a hike is live, and
    /hike/{id}/cancel could not free it either -- it sets cancel_requested on a dict no
    loop is reading and leaves the status alone.

    Startup is the only moment at which "the process that owned this hike is gone" is
    knowable, which is why this lives here and not in the status endpoint. Any hop
    already recorded is kept; only the verdict changes. A second server sharing
    cache_data would see its peer's live hike here, but that peer holds the authoritative
    in-memory dict and rewrites the file on its next save_state, so the mislabel cannot
    outlive one hop.
    """
    from backend.services import hiker as hk
    n = 0
    for d in sorted(config.RESULTS_DIR.glob("hike_*")):
        if not d.is_dir():
            continue
        try:
            hid = d.name[len("hike_"):]
            st = hk.load_state(hid)
            if not st or st.get("status") != "running":
                continue
            st["status"] = "interrupted"
            st["error"] = "the server restarted while this hike was running"
            st["notes"] = (st.get("notes") or []) + [
                "interrupted by a server restart; the hops already recorded are kept"]
            hk.save_state(hid, st)
            n += 1
        except Exception as exc:      # a malformed snapshot must not block startup
            print(f"[startup] could not reconcile {d.name}: {exc}", flush=True)
    return n


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting Ridge Explorer (FLUX Klein)...", flush=True)

    app.state.jobs = {}
    app.state.hikes = {}
    app.state.hike_inbox = {}
    # shard id -> worker-side failure message, written by result_collector and consumed
    # by the hike drains so a dead shard costs a poll instead of a 15-minute deadline
    app.state.hike_task_errors = {}
    app.state.cache = ThumbnailCache()

    orphaned = _reap_orphaned_hikes()
    if orphaned:
        print(f"{orphaned} hike(s) left running by a previous process marked interrupted",
              flush=True)

    pool = GPUPool(n_gpus=config.N_GPUS)
    pool.start()

    # Build the CLIP text encoder HERE, between pool.start() and wait_ready(), so its ~6 s
    # overlaps the far longer FLUX+DINOv2 load happening in the worker processes and costs
    # no startup time. It is also what makes the warm-up below worth anything.
    # _clip_text_encoder caches (model, tokenizer, memo) on app.state behind an unlocked
    # check-then-set. A hike started inside the model-build window found app.state.clip_text
    # still unset, built a SECOND encoder and overwrote the warm one, so that hike ran
    # against an empty memo and paid the whole pool encode at its first hop boundary --
    # precisely the stall the warm-up exists to remove -- while the warm thread went on
    # filling a memo nothing would ever read. Creating it before any request can be served
    # closes the window: every later caller, warm thread included, hits the same cache.
    enc = None
    try:
        from backend.routers.grid import _clip_text_encoder
        enc = _clip_text_encoder(app)
    except Exception as exc:      # never let a text-side optimisation break startup
        print(f"CLIP text encoder unavailable ({exc}); hikes will load it lazily",
              flush=True)

    # wait_ready() reports whether the whole pool came up, and shrinks pool.n_gpus to
    # the live worker count when it did not — everything that shards rows must read
    # pool.n_gpus, not config.N_GPUS, or it hands chunks to processes that do not exist.
    app.state.pool_ready = pool.wait_ready()
    app.state.gpu_pool = pool

    collector = asyncio.create_task(result_collector(app))

    # Warm the CLIP text pool off the request path. Encoding the 2,095-prompt pool takes
    # 15.3 s on CPU and used to be paid at the FIRST hop boundary of the first hike after
    # every restart -- a stall in the middle of a run, attributed to nothing. A daemon
    # thread keeps startup unblocked; the memo it fills is the one every hike reads,
    # because the encoder itself was built above instead of being raced for here.
    def _warm_prompt_pool():
        import time as _t
        try:
            t0 = _t.time()
            enc(list(config.HIKE_PROMPT_POOL))
            print(f"Prompt pool embeddings cached ({len(config.HIKE_PROMPT_POOL)} prompts, "
                  f"{_t.time() - t0:.1f}s)", flush=True)
        except Exception as exc:
            print(f"Prompt pool warm-up skipped: {exc}", flush=True)

    if enc is not None:
        threading.Thread(target=_warm_prompt_pool, daemon=True).start()
    if not app.state.pool_ready:
        print(f"Ridge Explorer DEGRADED: {pool.ready_count}/{config.N_GPUS} GPUs ready; "
              f"serving on {pool.n_gpus} worker(s)", flush=True)
    else:
        print(f"Ridge Explorer ready: {pool.ready_count}/{config.N_GPUS} GPUs", flush=True)
    yield

    collector.cancel()
    pool.shutdown()
    print("Ridge Explorer shut down.", flush=True)


app = FastAPI(title="Ridge Explorer", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

config.THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)
config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/cache/thumbnails", StaticFiles(directory=str(config.THUMBNAILS_DIR)), name="thumbnails")
app.mount("/cache/results", StaticFiles(directory=str(config.RESULTS_DIR)), name="results")

app.include_router(health.router)
app.include_router(grid.router)
app.include_router(discover.router)
app.include_router(cascade.router)
app.include_router(amr.router)
app.include_router(metro.router)
app.include_router(probe.router)
