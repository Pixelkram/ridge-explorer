import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend import config
from backend.services.gpu_pool import GPUPool, CellResult, LatentResult, LatentBatchResult
from backend.services.ridge_detector import compute_sensitivity, compute_clusters, classify_ridges
from backend.services.visualization import render_heatmap, assemble_image_grid, render_overlay, render_clusters
from backend.cache.thumbnail_cache import ThumbnailCache
from backend.routers import health, grid

import numpy as np


async def result_collector(app: FastAPI):
    """Background task that drains GPU results and updates job state."""
    pool: GPUPool = app.state.gpu_pool
    cache: ThumbnailCache = app.state.cache
    jobs: dict = app.state.jobs

    while True:
        results = pool.collect_results()
        for r in results:
            job = jobs.get(r.job_id)
            if not job:
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
                    job["thumbnails"][key] = r.thumbnail_bytes
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
        heatmap_path = results_dir / "heatmap.png"
        render_heatmap(sensitivity, job["alphas"], job["betas"], heatmap_path,
                       prompt_a=job["prompt_a"], prompt_b=job["prompt_b"], prompt_c=job["prompt_c"])
        job["heatmap_path"] = str(heatmap_path)

        image_grid_path = results_dir / "images.png"
        assemble_image_grid(job["thumbnails"], gs, image_grid_path)
        job["image_grid_path"] = str(image_grid_path)

        overlay_path = results_dir / "overlay.png"
        render_overlay(sensitivity, job["thumbnails"], gs, job["alphas"], job["betas"],
                       config.RIDGE_THRESHOLD_TAU, overlay_path)
        job["overlay_path"] = str(overlay_path)

        cluster_path = results_dir / "clusters.png"
        render_clusters(clusters, job["thumbnails"], gs, job["alphas"], job["betas"], cluster_path)
        job["cluster_path"] = str(cluster_path)

    job["phase"] = "complete"
    job["status"] = "complete"
    job["render_version"] = job.get("render_version", 0) + 1
    n_clusters = len(set(clusters.ravel()) - {-1}) if clusters is not None else 0
    print(f"Job {job_id}: analysis complete "
          f"({'3D' if is_3d else '2D'}, "
          f"m/m={sensitivity.max()/np.median(sensitivity):.1f}, "
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
    # Rough scaling: map Jacobian range to [0, 0.5] (typical DINOv2 sensitivity range)
    pseudo_scale = 0.3 / (jac_max - jac_min + 1e-10)
    pseudo_dino = (jac_vals - jac_min) * pseudo_scale + 0.01

    # Initialize with seed points using pseudo values, then acquire the rest
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

        # Get GP-predicted sensitivity — blend with Jacobian outside simplex
        pred_flat = detector_final.predict()
        blended = job["sensitivity"].copy()  # start with full-grid Jacobian
        for k in range(detector_final.n_valid):
            i, j = int(valid_positions[k, 0]), int(valid_positions[k, 1])
            blended[i, j] = pred_flat[k]  # overwrite simplex cells with GP prediction
            job["cells"][i][j]["sensitivity"] = float(pred_flat[k])

        job["sensitivity"] = blended
        job["mf_predicted_sensitivity"] = blended
        job["mf_n_observed"] = len(real_indices)
        job["mf_detector"] = detector_final

        # Render heatmap — now has values everywhere (Jacobian outside, GP inside)
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
            assemble_image_grid(job["thumbnails"], gs, image_grid_path)
            job["image_grid_path"] = str(image_grid_path)

        np.save(results_dir / "mf_sensitivity.npy", blended)

        print(f"MF-scan {job_id}: complete "
              f"(observed={len(real_indices)}, "
              f"rho_jac={detector_final.correlation:.3f})", flush=True)

    job["phase"] = "complete"
    job["status"] = "complete"
    job["render_version"] = job.get("render_version", 0) + 1


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting Ridge Explorer (FLUX Klein)...", flush=True)

    app.state.jobs = {}
    app.state.cache = ThumbnailCache()

    pool = GPUPool(n_gpus=config.N_GPUS)
    pool.start()
    pool.wait_ready()
    app.state.gpu_pool = pool

    collector = asyncio.create_task(result_collector(app))
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
